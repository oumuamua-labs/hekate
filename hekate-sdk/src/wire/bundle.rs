// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

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
    Air, FixedColumn, InlineKernelHint, Program, ProgramInstance, ProgramWitness,
};

use crate::generated::program as fb;
use crate::wire::{ast, boundary, chiplet, config, expander, fixed_column, permutation, trace};

const WIRE_FORMAT_VERSION: u32 = 6;

pub struct DeserializedBundle<F: TowerField> {
    pub name: String,
    pub num_columns: usize,
    pub num_public_inputs: usize,
    pub column_layout: Vec<ColumnType>,
    pub virtual_column_layout: Vec<ColumnType>,
    pub virtual_expander: Option<VirtualExpander>,
    pub constraint_ast: ConstraintAst<F>,
    pub boundary_constraints: Vec<BoundaryConstraint<F>>,
    pub fixed_columns: Vec<FixedColumn<F>>,
    pub permutation_checks: Vec<(String, PermutationCheckSpec)>,
    pub chiplet_defs: Vec<ChipletDef<F>>,
    pub inline_chiplets: Vec<ChipletDef<F>>,
    pub inline_chiplet_kernels: Vec<InlineKernelHint>,
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

    let name = String::from(
        bundle
            .name()
            .ok_or(wire_err("bundle missing program name"))?,
    );

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
        Some(cds) => chiplet::deserialize_chiplets::<F>(cds)?,
        None => Vec::new(),
    };

    let inline_chiplets = match bundle.inline_chiplets() {
        Some(cds) => chiplet::deserialize_chiplets::<F>(cds)?,
        None => Vec::new(),
    };

    let inline_chiplet_kernels = match bundle.inline_chiplet_kernels() {
        Some(hints) => chiplet::deserialize_kernel_hints(hints),
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

    let fixed_columns = match bundle.fixed_columns() {
        Some(v) => fixed_column::deserialize_fixed_columns(v)?,
        None => Vec::new(),
    };

    Ok(DeserializedBundle {
        name,
        num_columns: bundle.num_columns() as usize,
        num_public_inputs: bundle.num_public_inputs() as usize,
        column_layout,
        virtual_column_layout,
        virtual_expander,
        constraint_ast,
        boundary_constraints,
        fixed_columns,
        permutation_checks,
        chiplet_defs,
        inline_chiplets,
        inline_chiplet_kernels,
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

    let chiplets = chiplet::serialize_chiplets(&mut fbb, &program.chiplet_defs()?);

    let inline_chiplets =
        chiplet::serialize_chiplets(&mut fbb, &<P as Air<F>>::inline_chiplets(program)?);

    let inline_chiplet_kernels =
        chiplet::serialize_kernel_hints(&mut fbb, &<P as Air<F>>::inline_chiplet_kernels(program));

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

    let fixed = program.fixed_columns();
    let fixed_columns = fixed_column::serialize_fixed_columns(&mut fbb, &fixed);

    let name = fbb.create_string(&program.name());

    let bundle = fb::ProgramBundle::create(
        &mut fbb,
        &fb::ProgramBundleArgs {
            version: WIRE_FORMAT_VERSION,
            name: Some(name),
            num_columns: program.num_columns() as u32,
            num_public_inputs: program.num_public_inputs() as u32,
            column_layout: Some(layout),
            virtual_column_layout: Some(virtual_layout),
            virtual_expander: virtual_exp,
            constraint_ast: Some(constraint_ast),
            boundary_constraints: Some(boundaries),
            fixed_columns: Some(fixed_columns),
            permutation_checks: Some(perms),
            chiplet_defs: Some(chiplets),
            inline_chiplets: Some(inline_chiplets),
            inline_chiplet_kernels: Some(inline_chiplet_kernels),
            num_rows: instance.num_rows() as u64,
            public_inputs: Some(public_inputs),
            main_trace: Some(main_trace),
            chiplet_traces: Some(chiplet_traces),
            config: Some(cfg_offset),
        },
    );

    fbb.finish(bundle, None);

    Ok(fbb.finished_data().to_vec())
}
