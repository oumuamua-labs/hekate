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

use super::wire_err;
use alloc::string::ToString;
use alloc::vec::Vec;
use flatbuffers::FlatBufferBuilder;
use hekate_core::errors::Result;
use hekate_math::TowerField;
use hekate_program::Air;
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::BoundaryConstraint;

use crate::generated::program as fb;
use crate::wire::{ast, boundary, expander, lagrange, permutation, trace};

pub fn serialize_chiplet<'a, F: TowerField>(
    fbb: &mut FlatBufferBuilder<'a>,
    chiplet: &ChipletDef<F>,
) -> flatbuffers::WIPOffset<fb::ChipletDef<'a>> {
    let name = fbb.create_string(&chiplet.name());

    let layout = trace::serialize_column_layout(fbb, chiplet.column_layout());
    let virtual_layout = trace::serialize_column_layout(fbb, chiplet.virtual_column_layout());

    let constraint_ast = ast::serialize_ast(fbb, &chiplet.constraint_ast());

    let boundaries = boundary::serialize_boundaries(fbb, &chiplet.boundary_constraints());

    let perm_offsets: Vec<_> = chiplet
        .permutation_checks()
        .iter()
        .map(|(bus_id, spec)| permutation::serialize_bus_endpoint(fbb, bus_id, spec))
        .collect();
    let perms = fbb.create_vector(&perm_offsets);

    let virtual_expander = chiplet
        .virtual_expander()
        .map(|e| expander::serialize_expander(fbb, e));

    let pins = Air::<F>::lagrange_pinned_columns(chiplet);
    let lagrange_pins = lagrange::serialize_pins(fbb, &pins);

    fb::ChipletDef::create(
        fbb,
        &fb::ChipletDefArgs {
            name: Some(name),
            num_columns: chiplet.num_columns() as u32,
            column_layout: Some(layout),
            virtual_column_layout: Some(virtual_layout),
            constraint_ast: Some(constraint_ast),
            boundary_constraints: Some(boundaries),
            permutation_checks: Some(perms),
            virtual_expander,
            lagrange_pins: Some(lagrange_pins),
        },
    )
}

pub fn deserialize_chiplet<F: TowerField>(fb_cd: fb::ChipletDef<'_>) -> Result<ChipletDef<F>> {
    let name = fb_cd
        .name()
        .ok_or(wire_err("missing chiplet name"))?
        .to_string();

    let num_columns = fb_cd.num_columns() as usize;

    let column_layout = fb_cd
        .column_layout()
        .map(|v| trace::deserialize_column_layout(v))
        .transpose()?
        .unwrap_or_default();

    let virtual_column_layout = fb_cd
        .virtual_column_layout()
        .map(|v| trace::deserialize_column_layout(v))
        .transpose()?
        .unwrap_or_default();

    let constraint_ast = fb_cd
        .constraint_ast()
        .map(|a| ast::deserialize_ast::<F>(a))
        .transpose()?
        .ok_or(wire_err("missing chiplet constraint_ast"))?;

    let boundary_constraints: Vec<BoundaryConstraint<F>> = match fb_cd.boundary_constraints() {
        Some(bcs) => boundary::deserialize_boundaries(bcs)?,
        None => Vec::new(),
    };

    let permutation_checks = match fb_cd.permutation_checks() {
        Some(eps) => {
            let mut checks = Vec::with_capacity(eps.len());
            for i in 0..eps.len() {
                checks.push(permutation::deserialize_bus_endpoint(eps.get(i))?);
            }

            checks
        }
        None => Vec::new(),
    };

    let virtual_expander = fb_cd
        .virtual_expander()
        .map(|e| expander::deserialize_expander(e))
        .transpose()?;

    let lagrange_pins = match fb_cd.lagrange_pins() {
        Some(v) => lagrange::deserialize_pins(v)?,
        None => Vec::new(),
    };

    ChipletDef::from_wire(
        name,
        num_columns,
        constraint_ast,
        column_layout,
        virtual_column_layout,
        boundary_constraints,
        lagrange_pins,
        virtual_expander,
        permutation_checks,
    )
}
