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

#[path = "common/mod.rs"]
mod common;

use hekate::core::trace::{ColumnTrace, ColumnType};
use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::Block128;
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::trace::TraceBuilder;
use hekate_gadgets::{
    ArithmeticOpcode, CpuArithColumns, CpuFetchColumns, CpuFetchUnit, CpuIntArithmeticUnit,
    Instruction, IntArithmeticChiplet, IntArithmeticLayout, IntArithmeticOp, RomChiplet,
    generate_arithmetic_trace, generate_rom_trace,
};
use hekate_math::{Bit, Block32, TowerField};
use hekate_program::chiplet::{
    ChipletDef, CompositeChiplet, compose_chiplet_defs, compose_external_buses,
};
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::permutation::{PermutationCheckSpec, Source};
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;
use rand::TryRngCore;
use rand::rngs::OsRng;

type F = Block128;
type H = DefaultHasher;

// =================================================================
// 1. MAIN TRACE COLUMN LAYOUT
// =================================================================

const CPU_FETCH: usize = 0;
const CPU_ARITH: usize = CpuFetchColumns::NUM_COLUMNS;
const NUM_CPU_COLS: usize = CPU_ARITH + CpuArithColumns::NUM_COLUMNS;

// =================================================================
// 2. DUMMY CHIPLET (chiplet<>chiplet internal bus)
// =================================================================

const DUMMY_DATA: usize = 0;
const DUMMY_SEL: usize = 1;
const DUMMY_BUS_ID: &str = "data_pipe";

#[derive(Clone)]
struct DummyChiplet;

impl Air<F> for DummyChiplet {
    fn num_columns(&self) -> usize {
        2
    }

    fn column_layout(&self) -> &[ColumnType] {
        &[ColumnType::B32, ColumnType::Bit]
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        let spec = PermutationCheckSpec::new(
            vec![(Source::Column(DUMMY_DATA), b"kappa_data")],
            Some(DUMMY_SEL),
        )
        .with_clock_waiver(
            "see hekate/examples/composite_chiplets.rs: synthetic demo bus; \
             example only, not security-critical",
        );

        vec![(DUMMY_BUS_ID.into(), spec)]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(DUMMY_SEL));

        cs.build()
    }
}

fn generate_dummy_trace(data: &[u32], num_rows: usize) -> errors::Result<ColumnTrace> {
    let layout = [ColumnType::B32, ColumnType::Bit];
    let num_vars = num_rows.trailing_zeros() as usize;

    let mut tb = TraceBuilder::new(&layout, num_vars)?;

    for (i, &val) in data.iter().enumerate() {
        tb.set_b32(DUMMY_DATA, i, Block32::from(val))?;
    }

    tb.fill_selector(DUMMY_SEL, data.len())?;

    Ok(tb.build())
}

// =================================================================
// 3. COMPOSITE WRAPPERS
// =================================================================
//
// Each wrapper owns a CompositeChiplet and provides
// domain-specific trace generation. Trace ordering
// is encapsulated, callers never see sub-chiplet indices.

/// ROM + Arithmetic composite.
/// External buses connect both
/// chiplets to the main trace.
struct Pipeline {
    composite: CompositeChiplet<F>,
    rom_rows: usize,
    arith_rows: usize,
    arith_layout: IntArithmeticLayout,
}

impl Pipeline {
    fn new(rom_rows: usize, arith_rows: usize) -> Self {
        let cpu_fetch = CpuFetchUnit::linking_spec();

        let mut cpu_arith = CpuIntArithmeticUnit::linking_spec();
        cpu_arith.shift_column_indices(CPU_ARITH);

        let arith_layout = IntArithmeticLayout::compute(32);

        let composite = CompositeChiplet::<F>::builder("pipeline")
            .chiplet(RomChiplet::new(rom_rows))
            .chiplet(
                IntArithmeticChiplet::new(32, arith_rows)
                    .expect("IntArithmeticChiplet::new(32, arith_rows)"),
            )
            .external_bus(RomChiplet::BUS_ID, cpu_fetch)
            .external_bus(IntArithmeticChiplet::BUS_ID, cpu_arith)
            .build()
            .unwrap();

        Self {
            composite,
            rom_rows,
            arith_rows,
            arith_layout,
        }
    }

    /// Generate sub-chiplet traces
    /// in flatten_defs() order.
    fn generate_traces(
        &self,
        instrs: &[Instruction],
        ariths: &[IntArithmeticOp],
    ) -> errors::Result<Vec<ColumnTrace>> {
        Ok(vec![
            generate_rom_trace(instrs, self.rom_rows)?,
            generate_arithmetic_trace(ariths, &self.arith_layout, self.arith_rows)?,
        ])
    }

    fn composite(&self) -> &CompositeChiplet<F> {
        &self.composite
    }
}

/// Two DummyChiplets connected by internal
/// bus "audit::data_pipe". No external buses,
/// pure chiplet<>chiplet communication.
struct Audit {
    composite: CompositeChiplet<F>,
    num_rows: usize,
}

impl Audit {
    fn new(num_rows: usize) -> Self {
        let composite = CompositeChiplet::<F>::builder("audit")
            .chiplet(DummyChiplet)
            .chiplet(DummyChiplet)
            .build()
            .unwrap();

        Self {
            composite,
            num_rows,
        }
    }

    /// Generate identical traces for both
    /// endpoints. Matching data > matching
    /// GPA products on "audit::data_pipe".
    fn generate_traces(&self, data: &[u32]) -> errors::Result<Vec<ColumnTrace>> {
        Ok(vec![
            generate_dummy_trace(data, self.num_rows)?,
            generate_dummy_trace(data, self.num_rows)?,
        ])
    }

    fn composite(&self) -> &CompositeChiplet<F> {
        &self.composite
    }
}

// =================================================================
// 4. PROGRAM DEFINITION
// =================================================================

#[derive(Clone)]
struct CompositeChipletsProgram {
    pipeline: CompositeChiplet<F>,
    audit: CompositeChiplet<F>,
}

impl CompositeChipletsProgram {
    fn composites(&self) -> [&CompositeChiplet<F>; 2] {
        [&self.pipeline, &self.audit]
    }
}

impl Air<F> for CompositeChipletsProgram {
    fn name(&self) -> String {
        "CompositeExample".into()
    }

    fn num_columns(&self) -> usize {
        NUM_CPU_COLS
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(|| {
            let mut cols = CpuFetchColumns::build_layout();
            cols.extend(CpuArithColumns::build_layout());

            cols
        })
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        compose_external_buses(&self.composites())
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(CPU_FETCH + CpuFetchColumns::SELECTOR));
        cs.assert_boolean(cs.col(CPU_ARITH + CpuArithColumns::SELECTOR));

        cs.build()
    }
}

impl Program<F> for CompositeChipletsProgram {
    fn chiplet_defs(&self) -> errors::Result<Vec<ChipletDef<F>>> {
        compose_chiplet_defs(&self.composites())
    }
}

// =================================================================
// 5. WORKLOAD & TRACE GENERATION
// =================================================================

fn generate_workload(num_ops: usize) -> (Vec<Instruction>, Vec<IntArithmeticOp>, Vec<u32>) {
    let mut instrs = Vec::with_capacity(num_ops);
    let mut ariths = Vec::with_capacity(num_ops);
    let mut audit_data = Vec::with_capacity(num_ops);

    for i in 0..num_ops {
        instrs.push(Instruction::new((i * 4) as u32, 0x01, [0, 0, 0]));

        let val_a = (i * 100) as u32;
        let val_b = 0xAAAA_BBBB;

        let (opcode, b_val, result) = match i % 6 {
            0 => (ArithmeticOpcode::ADD, val_b, val_a.wrapping_add(val_b)),
            1 => (ArithmeticOpcode::SUB, val_b, val_a.wrapping_sub(val_b)),
            2 => (ArithmeticOpcode::AND, val_b, val_a & val_b),
            3 => (ArithmeticOpcode::XOR, val_b, val_a ^ val_b),
            4 => (ArithmeticOpcode::NOT, 0, !val_a),
            5 => (ArithmeticOpcode::LT, val_b, (val_a < val_b) as u32),
            _ => unreachable!(),
        };

        ariths.push(IntArithmeticOp::U32 {
            op: opcode,
            a: val_a,
            b: b_val,
            request_idx: i as u32,
        });

        audit_data.push(result);
    }

    (instrs, ariths, audit_data)
}

fn generate_cpu_trace(
    instrs: &[Instruction],
    ariths: &[IntArithmeticOp],
    num_rows: usize,
) -> errors::Result<ColumnTrace> {
    let num_vars = num_rows.trailing_zeros() as usize;

    let mut layout = CpuFetchColumns::build_layout();
    layout.extend(CpuArithColumns::build_layout());

    let mut tb = TraceBuilder::new(&layout, num_vars)?;

    for (i, instr) in instrs.iter().enumerate() {
        let r = i;
        if r >= num_rows {
            break;
        }

        let pc = instr.pc_bytes();
        let args = instr.args();

        tb.set_b32(
            CPU_FETCH + CpuFetchColumns::PC_B0,
            r,
            Block32::from(pc[0] as u32),
        )?;
        tb.set_b32(
            CPU_FETCH + CpuFetchColumns::PC_B1,
            r,
            Block32::from(pc[1] as u32),
        )?;
        tb.set_b32(
            CPU_FETCH + CpuFetchColumns::PC_B2,
            r,
            Block32::from(pc[2] as u32),
        )?;
        tb.set_b32(
            CPU_FETCH + CpuFetchColumns::PC_B3,
            r,
            Block32::from(pc[3] as u32),
        )?;
        tb.set_b32(
            CPU_FETCH + CpuFetchColumns::OPCODE,
            r,
            Block32::from(instr.opcode as u32),
        )?;
        tb.set_b32(
            CPU_FETCH + CpuFetchColumns::ARG0,
            r,
            Block32::from(args[0] as u32),
        )?;
        tb.set_b32(
            CPU_FETCH + CpuFetchColumns::ARG1,
            r,
            Block32::from(args[1] as u32),
        )?;
        tb.set_b32(
            CPU_FETCH + CpuFetchColumns::ARG2,
            r,
            Block32::from(args[2] as u32),
        )?;
        tb.set_bit(CPU_FETCH + CpuFetchColumns::SELECTOR, r, Bit::ONE)?;

        let IntArithmeticOp::U32 {
            op,
            a,
            b,
            request_idx: _,
        } = ariths[i]
        else {
            unreachable!("composite_chiplets is u32-only");
        };

        let res = match op {
            ArithmeticOpcode::ADD => a.wrapping_add(b),
            ArithmeticOpcode::SUB => a.wrapping_sub(b),
            ArithmeticOpcode::AND => a & b,
            ArithmeticOpcode::XOR => a ^ b,
            ArithmeticOpcode::NOT => !a,
            ArithmeticOpcode::LT => (a < b) as u32,
        };

        tb.set_b32(CPU_ARITH + CpuArithColumns::VAL_A, r, Block32::from(a))?;
        tb.set_b32(CPU_ARITH + CpuArithColumns::VAL_B, r, Block32::from(b))?;
        tb.set_b32(CPU_ARITH + CpuArithColumns::VAL_RES, r, Block32::from(res))?;
        tb.set_b32(
            CPU_ARITH + CpuArithColumns::OPCODE,
            r,
            Block32::from(op as u8 as u32),
        )?;
        tb.set_bit(CPU_ARITH + CpuArithColumns::SELECTOR, r, Bit::ONE)?;
    }

    Ok(tb.build())
}

// =================================================================
// 6. MAIN
// =================================================================

fn main() {
    common::init("Composite Chiplets");

    let num_ops: usize = 1 << 20;

    let main_num_rows = num_ops.next_power_of_two();
    let main_num_vars = main_num_rows.trailing_zeros() as usize;
    let rom_num_rows = num_ops.next_power_of_two();
    let arith_num_rows = num_ops.next_power_of_two();
    let dummy_num_rows = num_ops.next_power_of_two();

    // Composites own row counts and
    // encapsulate trace generation.
    let pipeline = Pipeline::new(rom_num_rows, arith_num_rows);
    let audit = Audit::new(dummy_num_rows);

    let mut config = Config {
        sumcheck_blinding_factor: 2,
        ..Config::default()
    };
    OsRng.try_fill_bytes(&mut config.matrix_seed).unwrap();

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    println!("  Operations:     {}", num_ops);
    println!(
        "  Main trace:     2^{} ({} rows)",
        main_num_vars, main_num_rows
    );
    println!(
        "  Pipeline:       {} chiplets (ROM 2^{}, Arith 2^{})",
        pipeline.composite().len(),
        rom_num_rows.trailing_zeros(),
        arith_num_rows.trailing_zeros(),
    );
    println!(
        "  Audit:          {} chiplets (Dummy 2^{})",
        audit.composite().len(),
        dummy_num_rows.trailing_zeros(),
    );

    // Each composite generates its own
    // traces in flatten_defs() order.
    // The caller never deals with
    // sub-chiplet indices.
    let (cpu_trace, chiplet_traces) = common::phase("Trace Generation", || {
        let (instrs, ariths, audit_data) = generate_workload(num_ops);

        let cpu_trace = generate_cpu_trace(&instrs, &ariths, main_num_rows).unwrap();

        let mut chiplet_traces = pipeline.generate_traces(&instrs, &ariths).unwrap();
        chiplet_traces.extend(audit.generate_traces(&audit_data).unwrap());

        (cpu_trace, chiplet_traces)
    });

    let air = CompositeChipletsProgram {
        pipeline: pipeline.composite().clone(),
        audit: audit.composite().clone(),
    };

    let instance = ProgramInstance::new(main_num_rows, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(chiplet_traces);

    let proof = common::phase("Proving", || {
        prove(
            b"Composite_Example",
            &air,
            &instance,
            &witness,
            &config,
            blinding_seed,
            None,
        )
        .expect("Prover failed")
    });

    common::proof_breakdown(&proof);

    let mut verifier_transcript = Transcript::<H>::new(b"Composite_Example");

    let is_valid = common::phase("Verifying", || {
        HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut verifier_transcript, &config)
            .expect("Verifier failed")
    });

    common::result(is_valid);
}
