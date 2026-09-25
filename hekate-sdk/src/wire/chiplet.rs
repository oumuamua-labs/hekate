// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use super::wire_err;
use alloc::string::ToString;
use alloc::vec::Vec;
use flatbuffers::FlatBufferBuilder;
use hekate_core::errors::Result;
use hekate_math::TowerField;
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::BoundaryConstraint;
use hekate_program::{Air, InlineKernelHint};

use crate::generated::program as fb;
use crate::wire::{ast, boundary, expander, fixed_column, permutation, trace};

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

    let fixed = Air::<F>::fixed_columns(chiplet);
    let fixed_columns = fixed_column::serialize_fixed_columns(fbb, &fixed);

    let inline_chiplets = serialize_chiplets(fbb, chiplet.inline_defs());
    let inline_chiplet_kernels = serialize_kernel_hints(fbb, chiplet.inline_hints());

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
            fixed_columns: Some(fixed_columns),
            inline_chiplets: Some(inline_chiplets),
            inline_chiplet_kernels: Some(inline_chiplet_kernels),
        },
    )
}

pub fn serialize_chiplets<'a, F: TowerField>(
    fbb: &mut FlatBufferBuilder<'a>,
    defs: &[ChipletDef<F>],
) -> flatbuffers::WIPOffset<flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<fb::ChipletDef<'a>>>>
{
    let offsets: Vec<_> = defs.iter().map(|cd| serialize_chiplet(fbb, cd)).collect();

    fbb.create_vector(&offsets)
}

pub fn serialize_kernel_hints<'a>(
    fbb: &mut FlatBufferBuilder<'a>,
    hints: &[InlineKernelHint],
) -> flatbuffers::WIPOffset<
    flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<fb::InlineKernelHint<'a>>>,
> {
    let offsets: Vec<_> = hints
        .iter()
        .map(|h| {
            fb::InlineKernelHint::create(
                fbb,
                &fb::InlineKernelHintArgs {
                    chiplet_idx: h.chiplet_idx as u32,
                    root_offset: h.root_offset as u32,
                    column_offset: h.column_offset as u32,
                },
            )
        })
        .collect();

    fbb.create_vector(&offsets)
}

pub fn deserialize_chiplets<'a, F: TowerField>(
    cds: flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<fb::ChipletDef<'a>>>,
) -> Result<Vec<ChipletDef<F>>> {
    let mut defs = Vec::with_capacity(cds.len());
    for i in 0..cds.len() {
        defs.push(deserialize_chiplet::<F>(cds.get(i))?);
    }

    Ok(defs)
}

pub fn deserialize_kernel_hints<'a>(
    hints: flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<fb::InlineKernelHint<'a>>>,
) -> Vec<InlineKernelHint> {
    (0..hints.len())
        .map(|i| {
            let h = hints.get(i);

            InlineKernelHint {
                chiplet_idx: h.chiplet_idx() as usize,
                root_offset: h.root_offset() as usize,
                column_offset: h.column_offset() as usize,
            }
        })
        .collect()
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

    let fixed_columns = match fb_cd.fixed_columns() {
        Some(v) => fixed_column::deserialize_fixed_columns(v)?,
        None => Vec::new(),
    };

    let inline_chiplets = match fb_cd.inline_chiplets() {
        Some(cds) => deserialize_chiplets::<F>(cds)?,
        None => Vec::new(),
    };

    let inline_kernels = match fb_cd.inline_chiplet_kernels() {
        Some(hints) => deserialize_kernel_hints(hints),
        None => Vec::new(),
    };

    ChipletDef::from_wire(
        name,
        num_columns,
        constraint_ast,
        column_layout,
        virtual_column_layout,
        boundary_constraints,
        fixed_columns,
        virtual_expander,
        permutation_checks,
        inline_chiplets,
        inline_kernels,
    )
}
