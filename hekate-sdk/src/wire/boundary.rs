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
use hekate_core::errors::Result;
use hekate_math::TowerField;
use hekate_program::constraint::{BoundaryConstraint, BoundaryTarget};

use super::field::{field_to_lo_hi, lo_hi_to_field};
use crate::generated::program as fb;

pub fn serialize_boundary<'a, F: TowerField>(
    fbb: &mut FlatBufferBuilder<'a>,
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

pub fn serialize_boundaries<'a, F: TowerField>(
    fbb: &mut FlatBufferBuilder<'a>,
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
