// SPDX-License-Identifier: Apache-2.0
// This file is part of the hekate project.
// Copyright (C) 2026 Andrei Kochergin <andrei@oumuamua.dev>
// Copyright (C) 2026 Oumuamua Labs <info@oumuamua.dev>.
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
use hekate_core::errors::Result;
use hekate_math::TowerField;
use hekate_program::{FixedColumn, FixedShape};

use super::field::{field_to_lo_hi, lo_hi_to_field};
use super::wire_err;
use crate::generated::program as fb;

pub fn serialize_fixed_column<'a, F: TowerField>(
    fbb: &mut FlatBufferBuilder<'a>,
    fc: &FixedColumn<F>,
) -> flatbuffers::WIPOffset<fb::FixedColumn<'a>> {
    let mut args = fb::FixedColumnArgs {
        col_idx: fc.col_idx as u32,
        ..Default::default()
    };

    match &fc.shape {
        FixedShape::LastRow => args.kind = fb::FixedShapeKind::LastRow,
        FixedShape::FirstRow => args.kind = fb::FixedShapeKind::FirstRow,
        FixedShape::Custom(bits) => {
            let bytes: Vec<u8> = bits.iter().map(|&b| b as u8).collect();

            args.kind = fb::FixedShapeKind::Custom;
            args.custom_bits = Some(fbb.create_vector(&bytes));
        }
        FixedShape::Periodic { period, values } => {
            let blocks: Vec<fb::Block128> = values.iter().map(to_block).collect();

            args.kind = fb::FixedShapeKind::Periodic;
            args.period = *period as u32;
            args.values = Some(fbb.create_vector(&blocks));
        }
        FixedShape::Sparse(entries) => {
            let rows: Vec<u64> = entries.iter().map(|&(r, _)| r as u64).collect();
            let blocks: Vec<fb::Block128> = entries.iter().map(|(_, v)| to_block(v)).collect();

            args.kind = fb::FixedShapeKind::Sparse;
            args.sparse_rows = Some(fbb.create_vector(&rows));
            args.sparse_values = Some(fbb.create_vector(&blocks));
        }
        FixedShape::Dense(values) => {
            let blocks: Vec<fb::Block128> = values.iter().map(to_block).collect();

            args.kind = fb::FixedShapeKind::Dense;
            args.values = Some(fbb.create_vector(&blocks));
        }
    }

    fb::FixedColumn::create(fbb, &args)
}

pub fn deserialize_fixed_column<F: TowerField>(
    fb_fc: fb::FixedColumn<'_>,
) -> Result<FixedColumn<F>> {
    let col_idx = fb_fc.col_idx() as usize;

    let shape = match fb_fc.kind() {
        fb::FixedShapeKind::LastRow => FixedShape::LastRow,
        fb::FixedShapeKind::FirstRow => FixedShape::FirstRow,
        fb::FixedShapeKind::Custom => {
            let bytes = fb_fc
                .custom_bits()
                .ok_or(wire_err("missing custom_bits for Custom fixed column"))?;

            let mut bits = Vec::with_capacity(bytes.len());
            for i in 0..bytes.len() {
                let b = bytes.get(i);
                if b > 1 {
                    return Err(wire_err("Custom fixed column bit must be 0 or 1"));
                }

                bits.push(b == 1);
            }

            FixedShape::Custom(bits)
        }
        fb::FixedShapeKind::Periodic => FixedShape::Periodic {
            period: fb_fc.period() as usize,
            values: read_values(fb_fc.values())?,
        },
        fb::FixedShapeKind::Sparse => {
            let rows = fb_fc
                .sparse_rows()
                .ok_or(wire_err("missing sparse_rows for Sparse fixed column"))?;
            let values: Vec<F> = read_values(fb_fc.sparse_values())?;

            if rows.len() != values.len() {
                return Err(wire_err("Sparse fixed column rows/values length mismatch"));
            }

            let mut entries = Vec::with_capacity(values.len());
            for (i, v) in values.into_iter().enumerate() {
                entries.push((rows.get(i) as usize, v));
            }

            FixedShape::Sparse(entries)
        }
        fb::FixedShapeKind::Dense => FixedShape::Dense(read_values(fb_fc.values())?),
        _ => return Err(wire_err("unknown FixedShapeKind")),
    };

    Ok(FixedColumn { col_idx, shape })
}

pub fn serialize_fixed_columns<'a, F: TowerField>(
    fbb: &mut FlatBufferBuilder<'a>,
    fixed: &[FixedColumn<F>],
) -> flatbuffers::WIPOffset<
    flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<fb::FixedColumn<'a>>>,
> {
    let offsets: Vec<_> = fixed
        .iter()
        .map(|fc| serialize_fixed_column(fbb, fc))
        .collect();

    fbb.create_vector(&offsets)
}

pub fn deserialize_fixed_columns<F: TowerField>(
    fb_cols: flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<fb::FixedColumn<'_>>>,
) -> Result<Vec<FixedColumn<F>>> {
    let mut out = Vec::with_capacity(fb_cols.len());
    for i in 0..fb_cols.len() {
        out.push(deserialize_fixed_column(fb_cols.get(i))?);
    }

    Ok(out)
}

fn to_block<F: TowerField>(f: &F) -> fb::Block128 {
    let (lo, hi) = field_to_lo_hi(f);
    fb::Block128::new(lo, hi)
}

fn read_values<F: TowerField>(
    values: Option<flatbuffers::Vector<'_, fb::Block128>>,
) -> Result<Vec<F>> {
    let blocks = values.ok_or(wire_err("fixed column missing field values"))?;

    let mut out = Vec::with_capacity(blocks.len());
    for i in 0..blocks.len() {
        let b = blocks.get(i);
        out.push(lo_hi_to_field(b.lo(), b.hi())?);
    }

    Ok(out)
}
