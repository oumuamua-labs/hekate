// SPDX-License-Identifier: Apache-2.0
// This file is part of the hekate project.
// Copyright (C) 2026 Andrei Kochergin <andrei@oumuamua.dev>
// Copyright (C) 2026 Oumuamua Labs <info@oumuamua.dev>. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;
use flatbuffers::FlatBufferBuilder;
use hekate_core::errors::{Error, Result};
use hekate_program::permutation::{BusKind, ChallengeLabel, PermutationCheckSpec, Source};

use crate::generated::program as fb;

const MAX_WAIVER_LEN: usize = 1024;

pub fn serialize_source<'a>(
    fbb: &mut FlatBufferBuilder<'a>,
    source: &Source,
    label: ChallengeLabel,
) -> flatbuffers::WIPOffset<fb::SourceEntry<'a>> {
    let fb_source = match source {
        Source::Column(idx) => fb::Source::create(
            fbb,
            &fb::SourceArgs {
                kind: fb::SourceKind::Column,
                column_index: *idx as u32,
                ..Default::default()
            },
        ),
        Source::Columns(indices) => {
            let vec = fbb.create_vector(&indices.iter().map(|i| *i as u32).collect::<Vec<_>>());

            fb::Source::create(
                fbb,
                &fb::SourceArgs {
                    kind: fb::SourceKind::Columns,
                    column_indices: Some(vec),
                    ..Default::default()
                },
            )
        }
        Source::RowIndexLeBytes(n) => fb::Source::create(
            fbb,
            &fb::SourceArgs {
                kind: fb::SourceKind::RowIndexLeBytes,
                byte_index: *n as u32,
                ..Default::default()
            },
        ),
        Source::RowIndexByte(n) => fb::Source::create(
            fbb,
            &fb::SourceArgs {
                kind: fb::SourceKind::RowIndexByte,
                byte_index: *n as u32,
                ..Default::default()
            },
        ),
        Source::Const(val) => {
            let lo = *val as u64;
            let hi = (*val >> 64) as u64;
            let block = fb::Block128::new(lo, hi);

            fb::Source::create(
                fbb,
                &fb::SourceArgs {
                    kind: fb::SourceKind::Constant,
                    constant_value: Some(&block),
                    ..Default::default()
                },
            )
        }
    };

    let label_str = fbb.create_string(core::str::from_utf8(label).unwrap_or(""));

    fb::SourceEntry::create(
        fbb,
        &fb::SourceEntryArgs {
            source: Some(fb_source),
            challenge_label: Some(label_str),
        },
    )
}

pub fn serialize_bus_endpoint<'a>(
    fbb: &mut FlatBufferBuilder<'a>,
    bus_id: &str,
    spec: &PermutationCheckSpec,
) -> flatbuffers::WIPOffset<fb::BusEndpoint<'a>> {
    let source_offsets: Vec<_> = spec
        .sources
        .iter()
        .map(|(source, label)| serialize_source(fbb, source, label))
        .collect();
    let sources = fbb.create_vector(&source_offsets);

    let (has_selector, selector_val) = match spec.selector {
        Some(idx) => (true, idx as u32),
        None => (false, 0),
    };

    let (has_recv_selector, recv_selector_val) = match spec.recv_selector {
        Some(idx) => (true, idx as u32),
        None => (false, 0),
    };

    let fb_kind = match spec.kind {
        BusKind::Permutation => fb::BusKind::Permutation,
        BusKind::Lookup => fb::BusKind::Lookup,
    };

    let waiver_offset = spec
        .clock_waiver
        .as_deref()
        .map(|reason| fbb.create_string(reason));

    let fb_spec = fb::PermutationCheckSpec::create(
        fbb,
        &fb::PermutationCheckSpecArgs {
            kind: fb_kind,
            sources: Some(sources),
            selector: selector_val,
            has_selector,
            clock_waiver_reason: waiver_offset,
            recv_selector: recv_selector_val,
            has_recv_selector,
        },
    );

    let bus_id_str = fbb.create_string(bus_id);

    fb::BusEndpoint::create(
        fbb,
        &fb::BusEndpointArgs {
            bus_id: Some(bus_id_str),
            spec: Some(fb_spec),
        },
    )
}

pub fn deserialize_bus_endpoint(
    fb_ep: fb::BusEndpoint<'_>,
) -> Result<(String, PermutationCheckSpec)> {
    let bus_id = fb_ep
        .bus_id()
        .ok_or(Error::Protocol {
            protocol: "wire",
            message: "missing bus_id",
        })?
        .to_string();

    let fb_spec = fb_ep.spec().ok_or(Error::Protocol {
        protocol: "wire",
        message: "missing PermutationCheckSpec",
    })?;

    let fb_sources = fb_spec.sources().ok_or(Error::Protocol {
        protocol: "wire",
        message: "missing sources",
    })?;

    let mut sources = Vec::with_capacity(fb_sources.len());
    for i in 0..fb_sources.len() {
        let entry = fb_sources.get(i);
        let source = deserialize_source(&entry)?;
        let label = deserialize_label(&entry)?;

        sources.push((source, label));
    }

    let selector = if fb_spec.has_selector() {
        Some(fb_spec.selector() as usize)
    } else {
        None
    };

    let kind = match fb_spec.kind() {
        fb::BusKind::Permutation => BusKind::Permutation,
        fb::BusKind::Lookup => BusKind::Lookup,
        _ => {
            return Err(Error::Protocol {
                protocol: "wire",
                message: "unknown BusKind variant",
            });
        }
    };

    let mut spec = if fb_spec.has_recv_selector() {
        let send = selector.ok_or(Error::Protocol {
            protocol: "wire",
            message: "paired spec missing send selector",
        })?;

        PermutationCheckSpec::new_paired(sources, send, fb_spec.recv_selector() as usize, kind)
    } else {
        match kind {
            BusKind::Permutation => PermutationCheckSpec::new(sources, selector),
            BusKind::Lookup => PermutationCheckSpec::new_lookup(sources, selector),
        }
    };

    if let Some(reason) = fb_spec.clock_waiver_reason() {
        if reason.len() > MAX_WAIVER_LEN {
            return Err(Error::Protocol {
                protocol: "wire",
                message: "clock_waiver_reason exceeds 1 KiB",
            });
        }

        spec = spec.with_clock_waiver(reason);
    }

    spec.validate_clock_stitching(&bus_id)?;

    Ok((bus_id, spec))
}

fn deserialize_source(entry: &fb::SourceEntry<'_>) -> Result<Source> {
    let fb_source = entry.source().ok_or(Error::Protocol {
        protocol: "wire",
        message: "missing source",
    })?;

    match fb_source.kind() {
        fb::SourceKind::Column => Ok(Source::Column(fb_source.column_index() as usize)),
        fb::SourceKind::Columns => {
            let indices = fb_source.column_indices().ok_or(Error::Protocol {
                protocol: "wire",
                message: "Columns source missing indices",
            })?;

            let vec: Vec<usize> = (0..indices.len())
                .map(|i| indices.get(i) as usize)
                .collect();

            Ok(Source::Columns(vec))
        }
        fb::SourceKind::RowIndexLeBytes => {
            Ok(Source::RowIndexLeBytes(fb_source.byte_index() as usize))
        }
        fb::SourceKind::RowIndexByte => Ok(Source::RowIndexByte(fb_source.byte_index() as usize)),
        fb::SourceKind::Constant => {
            let block = fb_source.constant_value().ok_or(Error::Protocol {
                protocol: "wire",
                message: "Constant source missing value",
            })?;

            let val = block.lo() as u128 | ((block.hi() as u128) << 64);

            Ok(Source::Const(val))
        }
        _ => Err(Error::Protocol {
            protocol: "wire",
            message: "unknown SourceKind",
        }),
    }
}

fn deserialize_label(entry: &fb::SourceEntry<'_>) -> Result<ChallengeLabel> {
    let s = entry.challenge_label().unwrap_or("");
    let leaked = super::leak_str(s)?;

    Ok(leaked.as_bytes())
}
