// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec::Vec;
use flatbuffers::{Allocator, FlatBufferBuilder};
use hekate_core::errors::Result;
use hekate_math::TowerField;
use hekate_program::constraint::{BoundaryConstraint, BoundaryTarget};

use super::field::{field_to_lo_hi, lo_hi_to_field};
use crate::generated::program as fb;

pub fn serialize_boundary<'a, F: TowerField, A: Allocator + 'a>(
    fbb: &mut FlatBufferBuilder<'a, A>,
    bc: &BoundaryConstraint<F>,
) -> flatbuffers::WIPOffset<fb::BoundaryConstraint<'a>> {
    let (kind, public_input_idx, constant_value) = match &bc.target {
        BoundaryTarget::PublicInput(idx) => (
            fb::BoundaryTargetKind::PublicInput,
            *idx as u32,
            fb::Block128::new(0, 0),
        ),
        BoundaryTarget::Constant(v) => {
            let (lo, hi) = field_to_lo_hi(v);
            (
                fb::BoundaryTargetKind::Constant,
                0,
                fb::Block128::new(lo, hi),
            )
        }
    };

    fb::BoundaryConstraint::create(
        fbb,
        &fb::BoundaryConstraintArgs {
            col_idx: bc.col_idx as u32,
            row_idx: bc.row_idx as u64,
            kind,
            public_input_idx,
            constant_value: Some(&constant_value),
        },
    )
}

pub fn deserialize_boundary<F: TowerField>(
    fb_bc: fb::BoundaryConstraint<'_>,
) -> Result<BoundaryConstraint<F>> {
    let col_idx = fb_bc.col_idx() as usize;
    let row_idx = fb_bc.row_idx() as usize;

    match fb_bc.kind() {
        fb::BoundaryTargetKind::PublicInput => Ok(BoundaryConstraint::with_public_input(
            col_idx,
            row_idx,
            fb_bc.public_input_idx() as usize,
        )),
        fb::BoundaryTargetKind::Constant => {
            let block = fb_bc
                .constant_value()
                .ok_or(super::wire_err("Constant boundary missing constant_value"))?;
            let val: F = lo_hi_to_field(block.lo(), block.hi())?;

            Ok(BoundaryConstraint::with_constant(col_idx, row_idx, val))
        }
        _ => Err(super::wire_err("unknown BoundaryTargetKind")),
    }
}

pub fn serialize_boundaries<'a, F: TowerField, A: Allocator + 'a>(
    fbb: &mut FlatBufferBuilder<'a, A>,
    bcs: &[BoundaryConstraint<F>],
) -> flatbuffers::WIPOffset<
    flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<fb::BoundaryConstraint<'a>>>,
> {
    let offsets: Vec<_> = bcs.iter().map(|bc| serialize_boundary(fbb, bc)).collect();

    fbb.create_vector(&offsets)
}

pub fn deserialize_boundaries<F: TowerField>(
    fb_bcs: flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<fb::BoundaryConstraint<'_>>>,
) -> Result<Vec<BoundaryConstraint<F>>> {
    let mut out = Vec::with_capacity(fb_bcs.len());
    for i in 0..fb_bcs.len() {
        out.push(deserialize_boundary(fb_bcs.get(i))?);
    }

    Ok(out)
}
