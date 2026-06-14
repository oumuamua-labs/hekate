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

use alloc::vec::Vec;
use flatbuffers::FlatBufferBuilder;
use hekate_core::errors::{Error, Result};
use hekate_core::trace::ColumnType;
use hekate_program::expander::{ExpansionEntry, VirtualExpander};

use crate::generated::program as fb;
use crate::wire::trace::column_type_to_fb;

pub fn serialize_expander<'a>(
    fbb: &mut FlatBufferBuilder<'a>,
    expander: &VirtualExpander,
) -> flatbuffers::WIPOffset<fb::VirtualExpander<'a>> {
    let specs = expander.expansion_entries();
    let entry_offsets: Vec<_> = specs
        .iter()
        .map(|entry| {
            let (kind, count, storage, phy_col_start) = match *entry {
                ExpansionEntry::ExpandBits { count, storage } => {
                    (fb::ExpansionKind::ExpandBits, count, storage, 0)
                }
                ExpansionEntry::PassThrough { count, storage } => {
                    (fb::ExpansionKind::PassThrough, count, storage, 0)
                }
                ExpansionEntry::ControlBits { count } => {
                    (fb::ExpansionKind::ControlBits, count, ColumnType::Bit, 0)
                }
                ExpansionEntry::ReusePassThrough {
                    phy_col_start,
                    count,
                    storage,
                } => (
                    fb::ExpansionKind::ReusePassThrough,
                    count,
                    storage,
                    phy_col_start,
                ),
                ExpansionEntry::ReuseExpandBits {
                    phy_col_start,
                    count,
                    storage,
                } => (
                    fb::ExpansionKind::ReuseExpandBits,
                    count,
                    storage,
                    phy_col_start,
                ),
            };

            fb::ExpansionEntry::create(
                fbb,
                &fb::ExpansionEntryArgs {
                    kind,
                    count: count as u32,
                    storage: column_type_to_fb(storage),
                    phy_col_start: phy_col_start as u32,
                },
            )
        })
        .collect();

    let entries = fbb.create_vector(&entry_offsets);

    fb::VirtualExpander::create(
        fbb,
        &fb::VirtualExpanderArgs {
            entries: Some(entries),
        },
    )
}

pub fn deserialize_expander(fb_exp: fb::VirtualExpander<'_>) -> Result<VirtualExpander> {
    let fb_entries = fb_exp.entries().ok_or(Error::Protocol {
        protocol: "wire",
        message: "missing expander entries",
    })?;

    let mut builder = VirtualExpander::new();

    for i in 0..fb_entries.len() {
        let entry = fb_entries.get(i);
        let count = entry.count() as usize;
        let storage = crate::wire::trace::column_type_from_fb(entry.storage())?;

        builder = match entry.kind() {
            fb::ExpansionKind::ExpandBits => builder.expand_bits(count, storage),
            fb::ExpansionKind::PassThrough => builder.pass_through(count, storage),
            fb::ExpansionKind::ControlBits => builder.control_bits(count),
            fb::ExpansionKind::ReusePassThrough => {
                builder.reuse_pass_through(entry.phy_col_start() as usize, count)
            }
            fb::ExpansionKind::ReuseExpandBits => {
                builder.reuse_expand_bits(entry.phy_col_start() as usize, count)
            }
            _ => {
                return Err(Error::Protocol {
                    protocol: "wire",
                    message: "unknown ExpansionKind",
                });
            }
        };
    }

    builder.build()
}
