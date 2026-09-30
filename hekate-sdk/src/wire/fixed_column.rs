// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec::Vec;
use flatbuffers::{Allocator, FlatBufferBuilder};
use hekate_core::errors::Result;
use hekate_math::TowerField;
use hekate_program::{CadenceSegment, FixedColumn, FixedShape};

use super::field::{field_to_lo_hi, lo_hi_to_field};
use super::wire_err;
use crate::generated::program as fb;

pub fn serialize_fixed_column<'a, F: TowerField, A: Allocator + 'a>(
    fbb: &mut FlatBufferBuilder<'a, A>,
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
        FixedShape::Cadence {
            stride,
            count,
            origin,
            values,
        } => {
            let blocks: Vec<fb::Block128> = values.iter().map(to_block).collect();

            args.kind = fb::FixedShapeKind::Cadence;
            args.stride = *stride as u64;
            args.count = *count as u64;
            args.origin = *origin as u64;
            args.values = Some(fbb.create_vector(&blocks));
        }
        FixedShape::Segments(segments) => {
            let seg_offsets: Vec<_> = segments
                .iter()
                .map(|seg| {
                    let blocks: Vec<fb::Block128> = seg.values.iter().map(to_block).collect();
                    let values = fbb.create_vector(&blocks);

                    fb::CadenceSegment::create(
                        fbb,
                        &fb::CadenceSegmentArgs {
                            stride: seg.stride as u64,
                            count: seg.count as u64,
                            origin: seg.origin as u64,
                            values: Some(values),
                        },
                    )
                })
                .collect();

            args.kind = fb::FixedShapeKind::Segments;
            args.segments = Some(fbb.create_vector(&seg_offsets));
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
        fb::FixedShapeKind::Cadence => FixedShape::Cadence {
            stride: usize::try_from(fb_fc.stride())
                .map_err(|_| wire_err("Cadence stride exceeds usize"))?,
            count: usize::try_from(fb_fc.count())
                .map_err(|_| wire_err("Cadence count exceeds usize"))?,
            origin: usize::try_from(fb_fc.origin())
                .map_err(|_| wire_err("Cadence origin exceeds usize"))?,
            values: read_values(fb_fc.values())?,
        },
        fb::FixedShapeKind::Segments => {
            let fb_segs = fb_fc
                .segments()
                .ok_or(wire_err("missing segments for Segments fixed column"))?;

            let mut segments = Vec::with_capacity(fb_segs.len());
            for i in 0..fb_segs.len() {
                let seg = fb_segs.get(i);

                segments.push(CadenceSegment {
                    stride: usize::try_from(seg.stride())
                        .map_err(|_| wire_err("Segments stride exceeds usize"))?,
                    count: usize::try_from(seg.count())
                        .map_err(|_| wire_err("Segments count exceeds usize"))?,
                    origin: usize::try_from(seg.origin())
                        .map_err(|_| wire_err("Segments origin exceeds usize"))?,
                    values: read_values(seg.values())?,
                });
            }

            FixedShape::Segments(segments)
        }
        _ => return Err(wire_err("unknown FixedShapeKind")),
    };

    Ok(FixedColumn { col_idx, shape })
}

pub fn serialize_fixed_columns<'a, F: TowerField, A: Allocator + 'a>(
    fbb: &mut FlatBufferBuilder<'a, A>,
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
