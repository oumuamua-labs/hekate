// SPDX-License-Identifier: Apache-2.0
// This file is part of the hekate-math project.
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
use hekate::core::trace::{ColumnTrace, ColumnType, Trace, TraceColumn};
use hekate::crypto::transcript::Transcript;
use hekate::crypto::DefaultHasher;
use hekate::math::{Block128, TowerField};
use hekate_core::trace::IntoTraceColumn;
use hekate_gadgets::{
    generate_rom_trace, CpuFetchColumns, CpuFetchUnit, Instruction, RomChiplet, RomColumns,
};
use hekate_math::{Bit, Block32};
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::constraint::ConstraintAst;
use hekate_program::permutation::PermutationCheckSpec;
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;

type F = Block128;
type H = DefaultHasher;

// --- 1. Define Combined AIR using Library Specs ---

#[derive(Clone)]
struct CpuRomAir {
    #[allow(dead_code)]
    num_rows: usize,
}

impl Air<F> for CpuRomAir {
    fn num_columns(&self) -> usize {
        // CPU Fetch (6 cols) defined in CpuFetchUnit
        // ROM (9 cols) defined in RomChiplet
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
        // Both endpoints sit on the main trace
        // for this combined AIR, so they share
        // the ROM's canonical bus_id and their
        // paired LogUp sums must cancel.
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

        cs.build()
    }
}

impl Program<F> for CpuRomAir {}

// --- 2. Trace Generation ---

fn generate_combined_trace(num_vars: usize) -> ColumnTrace {
    let num_rows = 1 << num_vars;

    // A. Generate ROM Instructions
    let mut instructions = Vec::new();
    for i in 0..num_rows {
        instructions.push(Instruction::new(i as u32, 1, [0, 0, 0]));
    }

    // Initialize Trace
    let mut trace = ColumnTrace::new(num_vars).unwrap();

    // B. Generate CPU Data manually (Simulation)
    // CPU Fetch Unit has 6 columns: PC(4) + Opcode(1) + Selector(1)

    // PC Bytes (0-3)
    let mut pc_cols = (0..4)
        .map(|_| Vec::with_capacity(num_rows))
        .collect::<Vec<_>>();

    // Opcode (4)
    let mut op_col = Vec::with_capacity(num_rows);

    // CPU side argument buffers
    let mut arg_cols = (0..3)
        .map(|_| Vec::with_capacity(num_rows))
        .collect::<Vec<_>>();

    // Selector (5)
    let mut sel_col = Vec::with_capacity(num_rows);

    for instr in &instructions {
        let bytes = instr.pc_bytes();
        for b in 0..4 {
            pc_cols[b].push(Block32::from(bytes[b] as u32));
        }

        op_col.push(Block32::from(instr.opcode as u32));

        let args = instr.args();
        for a in 0..3 {
            arg_cols[a].push(Block32::from(args[a] as u32));
        }

        sel_col.push(Bit::ONE);
    }

    // Add CPU Columns to Trace (Indices 0..5)
    for col in pc_cols {
        trace.add_column(col.into_trace_column()).unwrap();
    }

    trace.add_column(op_col.into_trace_column()).unwrap();

    for col in arg_cols {
        trace.add_column(col.into_trace_column()).unwrap();
    }

    trace.add_column(TraceColumn::Bit(sel_col)).unwrap();

    // C. Generate ROM Data using Library Helper
    let rom = generate_rom_trace(&instructions, num_rows).unwrap();
    for col in rom.into_columns() {
        trace.add_column(col).unwrap();
    }

    trace
}

#[test]
fn chiplets_integration() {
    let num_vars = 8; // 256 rows
    let num_rows = 1 << num_vars;
    let seed = [0u8; 32];

    let air = CpuRomAir { num_rows };
    let trace = generate_combined_trace(num_vars);

    // Verify trace dimensions match AIR expectation
    assert_eq!(trace.num_cols(), air.num_columns());

    let witness = ProgramWitness::new(trace);
    let instance = ProgramInstance::new(num_rows, vec![]);

    let config = Config {
        num_queries: 4,
        min_security_bits: 0,
        sumcheck_blinding_factor: 2,
        ..Config::default()
    };

    // 1. Prove
    println!("-> Proving...");
    let proof = prove(
        b"ChipletTestRefactored",
        &air,
        &instance,
        &witness,
        &config,
        seed,
        None,
    )
    .expect("Proving failed");

    // 2. Verify
    println!("-> Verifying...");
    let mut verifier_transcript = Transcript::<H>::new(b"ChipletTestRefactored");
    let result =
        HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut verifier_transcript, &config);

    assert!(result.unwrap(), "Verification failed");

    // 3. Bus Consistency
    // Both spec endpoints share a single bus_id
    // on the main trace; `verify()` rejects if
    // their LogUp sums do not cancel.
    assert_eq!(proof.main_logup_aux.claimed_sums.len(), 2);
    assert!(proof.chiplet_logup_aux.is_empty());

    println!("BUS SECURE: Chiplets are cryptographically linked.");
}
