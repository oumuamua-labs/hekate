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
use alloc::string::String;
use alloc::vec::Vec;
use flatbuffers::FlatBufferBuilder;
use hekate_core::config::Config;
use hekate_core::errors::{Error, Result};
use hekate_core::trace::{ColumnType, Trace};
use hekate_math::TowerField;
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::{BoundaryConstraint, ConstraintAst};
use hekate_program::expander::VirtualExpander;
use hekate_program::permutation::PermutationCheckSpec;
use hekate_program::{
    Air, InlineKernelHint, LagrangePin, Program, ProgramInstance, ProgramWitness,
};

use crate::generated::program as fb;
use crate::wire::{ast, boundary, chiplet, config, expander, lagrange, permutation, trace};

const WIRE_FORMAT_VERSION: u32 = 1;

pub struct DeserializedBundle<F: TowerField> {
    pub constraint_ast: ConstraintAst<F>,
    pub column_layout: Vec<ColumnType>,
    pub virtual_column_layout: Vec<ColumnType>,
    pub boundary_constraints: Vec<BoundaryConstraint<F>>,
    pub permutation_checks: Vec<(String, PermutationCheckSpec)>,
    pub virtual_expander: Option<VirtualExpander>,
    pub chiplet_defs: Vec<ChipletDef<F>>,
    pub inline_chiplets: Vec<ChipletDef<F>>,
    pub inline_chiplet_kernels: Vec<InlineKernelHint>,
    pub num_columns: usize,
    pub num_public_inputs: usize,
    pub lagrange_pins: Vec<LagrangePin>,
    pub instance: ProgramInstance<F>,
    pub witness: ProgramWitness<F>,
    pub config: Config,
}

pub fn serialize_bundle<F, P, T>(
    program: &P,
    instance: &ProgramInstance<F>,
    witness: &ProgramWitness<F, T>,
    cfg: &Config,
) -> Result<Vec<u8>>
where
    F: TowerField,
    P: Program<F>,
    T: Trace,
{
    let mut fbb = FlatBufferBuilder::with_capacity(1024 * 1024);

    let main_trace = trace::serialize_trace(&mut fbb, &witness.trace);

    let chiplet_trace_offsets: Vec<_> = witness
        .chiplet_traces
        .iter()
        .map(|t| trace::serialize_trace(&mut fbb, t))
        .collect();
    let chiplet_traces = fbb.create_vector(&chiplet_trace_offsets);

    finish_bundle(fbb, program, instance, cfg, main_trace, chiplet_traces)
}

/// Serialize program + instance + config + chiplet defs
/// without witness data. `main_trace` is a zero-column
/// `ColumnTrace` with `num_rows = instance.num_rows()`;
/// `chiplet_traces` is empty.
pub fn serialize_bundle_header<F, P>(
    program: &P,
    instance: &ProgramInstance<F>,
    cfg: &Config,
) -> Result<Vec<u8>>
where
    F: TowerField,
    P: Program<F>,
{
    let mut fbb = FlatBufferBuilder::with_capacity(1024 * 1024);

    let empty_columns = fbb.create_vector::<flatbuffers::WIPOffset<fb::TraceColumn>>(&[]);
    let main_trace = fb::ColumnTrace::create(
        &mut fbb,
        &fb::ColumnTraceArgs {
            columns: Some(empty_columns),
            num_rows: instance.num_rows() as u64,
        },
    );

    let chiplet_traces = fbb.create_vector::<flatbuffers::ForwardsUOffset<fb::ColumnTrace>>(&[]);

    finish_bundle(fbb, program, instance, cfg, main_trace, chiplet_traces)
}

pub fn deserialize_bundle<F: TowerField>(bytes: &[u8]) -> Result<DeserializedBundle<F>> {
    super::reset_leak_budget();

    let bundle = flatbuffers::root::<fb::ProgramBundle>(bytes).map_err(|_| Error::Protocol {
        protocol: "wire",
        message: "invalid FlatBuffer",
    })?;

    if bundle.version() != WIRE_FORMAT_VERSION {
        return Err(Error::Protocol {
            protocol: "wire",
            message: "wire format version mismatch",
        });
    }

    let column_layout = bundle
        .column_layout()
        .map(|v| trace::deserialize_column_layout(v))
        .transpose()?
        .unwrap_or_default();

    let virtual_column_layout = bundle
        .virtual_column_layout()
        .map(|v| trace::deserialize_column_layout(v))
        .transpose()?
        .unwrap_or_else(|| column_layout.clone());

    let constraint_ast = bundle
        .constraint_ast()
        .map(|a| ast::deserialize_ast::<F>(a))
        .transpose()?
        .ok_or(wire_err("missing constraint_ast"))?;

    let boundary_constraints: Vec<BoundaryConstraint<F>> = match bundle.boundary_constraints() {
        Some(bcs) => boundary::deserialize_boundaries(bcs)?,
        None => Vec::new(),
    };

    let permutation_checks = match bundle.permutation_checks() {
        Some(eps) => {
            let mut checks = Vec::with_capacity(eps.len());
            for i in 0..eps.len() {
                checks.push(permutation::deserialize_bus_endpoint(eps.get(i))?);
            }

            checks
        }
        None => Vec::new(),
    };

    let virtual_expander = bundle
        .virtual_expander()
        .map(|e| expander::deserialize_expander(e))
        .transpose()?;

    let chiplet_defs = match bundle.chiplet_defs() {
        Some(cds) => {
            let mut defs = Vec::with_capacity(cds.len());
            for i in 0..cds.len() {
                defs.push(chiplet::deserialize_chiplet::<F>(cds.get(i))?);
            }

            defs
        }
        None => Vec::new(),
    };

    let inline_chiplets = match bundle.inline_chiplets() {
        Some(cds) => {
            let mut defs = Vec::with_capacity(cds.len());
            for i in 0..cds.len() {
                defs.push(chiplet::deserialize_chiplet::<F>(cds.get(i))?);
            }

            defs
        }
        None => Vec::new(),
    };

    let inline_chiplet_kernels: Vec<InlineKernelHint> = match bundle.inline_chiplet_kernels() {
        Some(hs) => (0..hs.len())
            .map(|i| {
                let h = hs.get(i);
                InlineKernelHint {
                    chiplet_idx: h.chiplet_idx() as usize,
                    root_offset: h.root_offset() as usize,
                    column_offset: h.column_offset() as usize,
                }
            })
            .collect(),
        None => Vec::new(),
    };

    let public_inputs: Vec<F> = match bundle.public_inputs() {
        Some(pis) => {
            let mut inputs = Vec::with_capacity(pis.len());
            for i in 0..pis.len() {
                let block = pis.get(i);
                inputs.push(super::field::lo_hi_to_field(block.lo(), block.hi())?);
            }

            inputs
        }
        None => Vec::new(),
    };

    let num_rows = bundle.num_rows() as usize;
    if num_rows == 0 || !num_rows.is_power_of_two() {
        return Err(wire_err("num_rows must be a non-zero power of two"));
    }

    let instance = ProgramInstance::new(num_rows, public_inputs);

    let main_trace = bundle
        .main_trace()
        .map(|t| trace::deserialize_trace(t))
        .transpose()?
        .ok_or(wire_err("missing main_trace"))?;

    let chiplet_traces = match bundle.chiplet_traces() {
        Some(cts) => {
            let mut traces = Vec::with_capacity(cts.len());
            for i in 0..cts.len() {
                traces.push(trace::deserialize_trace(cts.get(i))?);
            }

            traces
        }
        None => Vec::new(),
    };

    let witness = ProgramWitness::new(main_trace).with_chiplets(chiplet_traces);

    let cfg = bundle
        .config()
        .map(|c| config::deserialize_config(c))
        .transpose()?
        .ok_or(wire_err("missing config"))?;

    let lagrange_pins = match bundle.lagrange_pins() {
        Some(v) => lagrange::deserialize_pins(v)?,
        None => Vec::new(),
    };

    Ok(DeserializedBundle {
        constraint_ast,
        column_layout,
        virtual_column_layout,
        boundary_constraints,
        permutation_checks,
        virtual_expander,
        chiplet_defs,
        inline_chiplets,
        inline_chiplet_kernels,
        num_columns: bundle.num_columns() as usize,
        num_public_inputs: bundle.num_public_inputs() as usize,
        lagrange_pins,
        instance,
        witness,
        config: cfg,
    })
}

fn finish_bundle<'a, F, P>(
    mut fbb: FlatBufferBuilder<'a>,
    program: &P,
    instance: &ProgramInstance<F>,
    cfg: &Config,
    main_trace: flatbuffers::WIPOffset<fb::ColumnTrace<'a>>,
    chiplet_traces: flatbuffers::WIPOffset<
        flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<fb::ColumnTrace<'a>>>,
    >,
) -> Result<Vec<u8>>
where
    F: TowerField,
    P: Program<F>,
{
    let layout = trace::serialize_column_layout(&mut fbb, program.column_layout());
    let virtual_layout = trace::serialize_column_layout(&mut fbb, program.virtual_column_layout());

    let constraint_ast = ast::serialize_ast(&mut fbb, &program.constraint_ast());

    let boundaries = boundary::serialize_boundaries(&mut fbb, &program.boundary_constraints());

    let perm_offsets: Vec<_> = program
        .permutation_checks()
        .iter()
        .map(|(bus_id, spec)| permutation::serialize_bus_endpoint(&mut fbb, bus_id, spec))
        .collect();
    let perms = fbb.create_vector(&perm_offsets);

    let virtual_exp = program
        .virtual_expander()
        .map(|e| expander::serialize_expander(&mut fbb, e));

    let chiplet_defs_list = program.chiplet_defs()?;
    let chiplet_offsets: Vec<_> = chiplet_defs_list
        .iter()
        .map(|cd| chiplet::serialize_chiplet(&mut fbb, cd))
        .collect();
    let chiplets = fbb.create_vector(&chiplet_offsets);

    let inline_chiplet_defs = <P as Air<F>>::inline_chiplets(program)?;
    let inline_chiplet_offsets: Vec<_> = inline_chiplet_defs
        .iter()
        .map(|cd| chiplet::serialize_chiplet(&mut fbb, cd))
        .collect();
    let inline_chiplets = fbb.create_vector(&inline_chiplet_offsets);

    let hint_offsets: Vec<_> = <P as Air<F>>::inline_chiplet_kernels(program)
        .iter()
        .map(|h| {
            fb::InlineKernelHint::create(
                &mut fbb,
                &fb::InlineKernelHintArgs {
                    chiplet_idx: h.chiplet_idx as u32,
                    root_offset: h.root_offset as u32,
                    column_offset: h.column_offset as u32,
                },
            )
        })
        .collect();

    let inline_chiplet_kernels = fbb.create_vector(&hint_offsets);

    let public_inputs_blocks: Vec<fb::Block128> = instance
        .public_inputs()
        .iter()
        .map(|f| {
            let (lo, hi) = super::field::field_to_lo_hi(f);
            fb::Block128::new(lo, hi)
        })
        .collect();
    let public_inputs = fbb.create_vector(&public_inputs_blocks);

    let cfg_offset = config::serialize_config(&mut fbb, cfg);

    let pins = program.lagrange_pinned_columns();
    let lagrange_pins = lagrange::serialize_pins(&mut fbb, &pins);

    let bundle = fb::ProgramBundle::create(
        &mut fbb,
        &fb::ProgramBundleArgs {
            version: WIRE_FORMAT_VERSION,
            num_columns: program.num_columns() as u32,
            num_public_inputs: program.num_public_inputs() as u32,
            column_layout: Some(layout),
            virtual_column_layout: Some(virtual_layout),
            constraint_ast: Some(constraint_ast),
            boundary_constraints: Some(boundaries),
            permutation_checks: Some(perms),
            virtual_expander: virtual_exp,
            chiplet_defs: Some(chiplets),
            inline_chiplets: Some(inline_chiplets),
            inline_chiplet_kernels: Some(inline_chiplet_kernels),
            num_rows: instance.num_rows() as u64,
            public_inputs: Some(public_inputs),
            main_trace: Some(main_trace),
            chiplet_traces: Some(chiplet_traces),
            config: Some(cfg_offset),
            lagrange_pins: Some(lagrange_pins),
        },
    );

    fbb.finish(bundle, None);

    Ok(fbb.finished_data().to_vec())
}
