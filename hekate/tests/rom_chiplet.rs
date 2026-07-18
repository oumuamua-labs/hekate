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

use hekate::core::config::Config;
use hekate::core::trace::{ColumnTrace, ColumnType, TraceColumn};
use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::{Block128, TowerField};
use hekate_core::trace::IntoTraceColumn;
use hekate_gadgets::{
    CpuFetchColumns, CpuFetchUnit, Instruction, RomChiplet, RomColumns, generate_rom_trace,
};
use hekate_math::{Bit, Block32};
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::permutation::PermutationCheckSpec;
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_scribble::{MutationKind, ScribbleConfig, Target, assert_all_caught};
use hekate_verifier::HekateVerifier;

type F = Block128;
type H = DefaultHasher;

// Define Combined AIR for Testing

#[derive(Clone)]
struct RomTestAir;

impl Air<F> for RomTestAir {
    fn num_columns(&self) -> usize {
        CpuFetchColumns::NUM_COLUMNS + RomColumns::NUM_COLUMNS
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();

        LAYOUT.get_or_init(|| {
            let mut cols = CpuFetchColumns::build_layout();
            cols.extend(RomColumns::build_layout());

            cols
        })
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        // Both endpoints live on the main
        // trace; share the ROM's canonical
        // bus_id so their LogUp sums cancel.
        let cpu_spec = CpuFetchUnit::linking_spec();
        let mut rom_spec = RomChiplet::linking_spec();
        rom_spec.shift_column_indices(CpuFetchColumns::NUM_COLUMNS);

        vec![
            (RomChiplet::BUS_ID.into(), cpu_spec),
            (RomChiplet::BUS_ID.into(), rom_spec),
        ]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(CpuFetchColumns::SELECTOR));

        let mut ast = cs.build();

        let rom_chiplet = RomChiplet { num_rows: 0 };

        let mut rom_ast = rom_chiplet.constraint_ast();
        rom_ast.arena.shift_cells(CpuFetchColumns::NUM_COLUMNS);

        ast.merge(rom_ast);

        ast
    }
}

impl Program<F> for RomTestAir {}

// --- 2. Trace Generation Helpers ---

/// Generates a combined trace containing both CPU fetch events and ROM data.
fn generate_combined_trace(instructions: &[Instruction], num_rows: usize) -> ColumnTrace {
    let num_vars = (num_rows as f64).log2() as usize;
    let mut trace = ColumnTrace::new(num_vars).unwrap();

    // 1. Generate CPU columns (simulation of execution)
    // CPU Fetch Unit (6 cols): PC(4), Opcode(1), Selector(1)

    // Vectors for CPU columns
    let mut pc_cols = (0..4)
        .map(|_| Vec::with_capacity(num_rows))
        .collect::<Vec<_>>();
    let mut op_col = Vec::with_capacity(num_rows);
    let mut arg_cols = (0..3)
        .map(|_| Vec::with_capacity(num_rows))
        .collect::<Vec<_>>();
    let mut sel_col = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        if i < instructions.len() {
            let instr = &instructions[i];
            let pc_bytes = instr.pc_bytes();

            for (col, &byte) in pc_cols.iter_mut().zip(pc_bytes.iter()) {
                col.push(Block32::from(byte as u32));
            }

            op_col.push(Block32::from(instr.opcode() as u32));

            let args = instr.args();
            for (col, &arg) in arg_cols.iter_mut().zip(args.iter()) {
                col.push(Block32::from(arg as u32));
            }

            sel_col.push(Bit::ONE);
        } else {
            // Padding
            for col in &mut pc_cols {
                col.push(Block32::ZERO);
            }

            op_col.push(Block32::ZERO);

            for col in &mut arg_cols {
                col.push(Block32::ZERO);
            }

            sel_col.push(Bit::ZERO);
        }
    }

    // Add CPU columns to trace
    for col in pc_cols {
        trace.add_column(col.into_trace_column()).unwrap();
    }

    trace.add_column(op_col.into_trace_column()).unwrap();

    for col in arg_cols {
        trace.add_column(col.into_trace_column()).unwrap();
    }

    // Insert argument columns
    trace.add_column(TraceColumn::Bit(sel_col)).unwrap();

    // 2. Generate ROM columns using library helper
    let rom = generate_rom_trace(instructions, num_rows).unwrap();
    for col in rom.into_columns() {
        trace.add_column(col).unwrap();
    }

    trace
}

// --- 3. Tests ---

/// E2E ROM-CPU Linking Test.
///
/// This test demonstrates:
/// 1. ROM trace generation with static instructions
/// 2. CPU trace generation with fetch events (execution order)
/// 3. GPA proving on both sides via HekateProver
/// 4. Verification that products match (bus consistency)
#[test]
fn rom_cpu_linking() {
    // Define a simple program (8 instructions)
    let program = vec![
        Instruction::new(0x1000, 0x01, [0x11, 0x12, 0x13]), // ADD
        Instruction::new(0x1004, 0x02, [0x21, 0x22, 0x23]), // SUB
        Instruction::new(0x1008, 0x03, [0x31, 0x32, 0x33]), // MUL
        Instruction::new(0x100C, 0x04, [0x41, 0x42, 0x43]), // DIV
        Instruction::new(0x1010, 0x05, [0x51, 0x52, 0x53]), // LOAD
        Instruction::new(0x1014, 0x06, [0x61, 0x62, 0x63]), // STORE
        Instruction::new(0x1018, 0x07, [0x71, 0x72, 0x73]), // JUMP
        Instruction::new(0x101C, 0xFF, [0x00, 0x00, 0x00]), // HALT
    ];

    let num_vars = 4; // 16 rows
    let num_rows = 1 << num_vars;
    let seed = [0xAAu8; 32];

    // 1. Setup
    let air = RomTestAir;
    let trace = generate_combined_trace(&program, num_rows);
    let witness = ProgramWitness::new(trace);
    let instance = ProgramInstance::new(num_rows, vec![]); // No public inputs

    let config = Config {
        num_queries: 4,
        min_security_bits: 0,
        sumcheck_blinding_factor: 2,
        ldt_support_size: 4,
        ..Config::default()
    };

    // 2. Prove
    println!("-> Proving...");
    let proof =
        prove(b"ROM_E2E", &air, &instance, &witness, &config, seed, None).expect("Proving failed");

    // 3. Verify
    println!("-> Verifying...");
    let mut verifier_transcript = Transcript::<H>::new(b"ROM_E2E");
    let result =
        HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut verifier_transcript, &config);

    assert!(result.unwrap(), "Verification failed");

    // 4. Bus Consistency
    // Both spec endpoints sit on the main trace
    // under a sharedbus_id; `verify()` rejects
    // if their LogUp sums do not cancel.
    assert_eq!(proof.main_logup_aux.claimed_sums.len(), 2);
    assert!(proof.chiplet_logup_aux.is_empty());
}

/// Correctness test: ROM GPA for large trace via HekateProver.
///
/// This test verifies that the full proving stack handles large ROMs correctly.
#[test]
fn rom_gpa_large_trace() {
    let num_vars = 12; // 2^12 = 4096 rows
    let num_rows = 1 << num_vars;
    let seed = [0xAAu8; 32];

    // Generate synthetic instructions
    let mut instructions = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        let pc = (i * 4) as u32;
        let opcode = ((i % 256) as u8).wrapping_add(1);
        let args = [(i & 0xFF) as u8, 0, 0];
        instructions.push(Instruction::new(pc, opcode, args));
    }

    let air = RomTestAir;
    let trace = generate_combined_trace(&instructions, num_rows);
    let witness = ProgramWitness::new(trace);
    let instance = ProgramInstance::new(num_rows, vec![]);

    let config = Config {
        num_queries: 2,
        min_security_bits: 0,
        sumcheck_blinding_factor: 0, // Disable ZK for speed in this large test
        ..Config::default()
    };

    let proof = prove(b"ROM_Large", &air, &instance, &witness, &config, seed, None)
        .expect("Large trace proving failed");

    // `check_bus_sum_matching` inside `verify()`
    // enforces endpoint cancellation across
    // the paired (main, ROM) bus.
    let mut vt = Transcript::<H>::new(b"ROM_Large");
    let ok = HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut vt, &config).unwrap();

    assert!(ok, "Large trace verification failed");
}

/// Test: ROM Padding Transparency via HekateProver.
///
/// This test verifies that padding rows (selector=0) do not affect the product.
/// Specifically: a trace with ONLY padding should result in product = 1.
#[test]
fn rom_padding_transparency() {
    let num_vars = 4;
    let num_rows = 1 << num_vars;
    let seed = [0xAAu8; 32];

    let air = RomTestAir;
    let instance = ProgramInstance::new(num_rows, vec![]);

    // Case 1: All Padding (Empty program)
    let empty_instructions = vec![];
    let empty_trace = generate_combined_trace(&empty_instructions, num_rows);
    let empty_witness = ProgramWitness::new(empty_trace);

    let config = Config {
        ldt_support_size: 4,
        min_security_bits: 0,
        ..Config::default()
    };

    let proof = prove(
        b"Padding",
        &air,
        &instance,
        &empty_witness,
        &config,
        seed,
        None,
    )
    .expect("Proving empty trace failed");

    // Padding rows carry selector=0,
    // so `h·selector = 0` on every
    // row and both main-side
    // claimed_sums are zero.
    assert_eq!(proof.main_logup_aux.claimed_sums.len(), 2);
    assert_eq!(proof.main_logup_aux.claimed_sums[0].1, F::ZERO);
    assert_eq!(proof.main_logup_aux.claimed_sums[1].1, F::ZERO);
}

#[test]
fn scribble_rom_flip_selector_caught() {
    let program = vec![
        Instruction::new(0x1000, 0x01, [0x11, 0x12, 0x13]),
        Instruction::new(0x1004, 0x02, [0x21, 0x22, 0x23]),
        Instruction::new(0x1008, 0x03, [0x31, 0x32, 0x33]),
        Instruction::new(0x100C, 0x04, [0x41, 0x42, 0x43]),
        Instruction::new(0x1010, 0x05, [0x51, 0x52, 0x53]),
        Instruction::new(0x1014, 0x06, [0x61, 0x62, 0x63]),
        Instruction::new(0x1018, 0x07, [0x71, 0x72, 0x73]),
        Instruction::new(0x101C, 0xFF, [0x00, 0x00, 0x00]),
    ];

    let num_vars = 4;
    let num_rows = 1 << num_vars;

    let air = RomTestAir;
    let trace = generate_combined_trace(&program, num_rows);
    let witness = ProgramWitness::new(trace);
    let instance = ProgramInstance::new(num_rows, vec![]);

    assert_all_caught(
        &air,
        &instance,
        &witness,
        ScribbleConfig::default()
            .target(Target::Main)
            .mutations([MutationKind::FlipSelector])
            .cases(64),
    );
}
