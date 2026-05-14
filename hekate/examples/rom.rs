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

#[path = "common/mod.rs"]
mod common;

use hekate::core::trace::ColumnTrace;
use hekate::core::trace::ColumnType;
use hekate::crypto::transcript::Transcript;
use hekate::crypto::DefaultHasher;
use hekate::math::{Block128, TowerField};
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::trace::TraceBuilder;
use hekate_gadgets::{
    generate_rom_trace, CpuFetchColumns, CpuFetchUnit, Instruction, RomChiplet, RomColumns,
};
use hekate_math::{Bit, Block32};
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::constraint::ConstraintAst;
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;
use rand::{rngs::OsRng, TryRngCore};

type F = Block128;
type H = DefaultHasher;

// =================================================================
// 1. ROM TEST PROGRAM DEFINITION
// =================================================================
#[derive(Clone)]
struct RomInlineChipletProgram {
    #[allow(dead_code)]
    num_rows: usize,
}

impl Air<F> for RomInlineChipletProgram {
    fn num_columns(&self) -> usize {
        // CPU Fetch (6 cols) + ROM Chiplet (9 cols)
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

    fn permutation_checks(
        &self,
    ) -> Vec<(String, hekate_program::permutation::PermutationCheckSpec)> {
        let cpu_spec = CpuFetchUnit::linking_spec();
        let mut rom_spec = RomChiplet::linking_spec();

        // Adjust ROM indices to account for
        // CPU columns offset in the combined trace.
        rom_spec.shift_column_indices(CpuFetchColumns::NUM_COLUMNS);

        vec![
            (RomChiplet::BUS_ID.to_string(), cpu_spec),
            (RomChiplet::BUS_ID.to_string(), rom_spec),
        ]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(CpuFetchColumns::SELECTOR));

        let mut ast = cs.build();

        let mut rom_ast = RomChiplet::new(self.num_rows).constraint_ast();
        rom_ast.arena.shift_cells(CpuFetchColumns::NUM_COLUMNS);

        ast.merge(rom_ast);

        ast
    }
}

impl Program<F> for RomInlineChipletProgram {}

// =================================================================
// 2. TRACE GENERATION HELPERS
// =================================================================
fn generate_rom_instructions(num_rows: usize) -> Vec<Instruction> {
    (0..num_rows)
        .map(|i| {
            Instruction::new(
                (i * 4) as u32,
                ((i % 256) as u8).wrapping_add(1),
                [(i & 0xFF) as u8, ((i >> 8) & 0xFF) as u8, 0],
            )
        })
        .collect()
}

/// Generates a combined trace containing
/// both CPU fetch events and ROM data.
fn generate_combined_trace(
    instructions: &[Instruction],
    num_rows: usize,
) -> errors::Result<ColumnTrace> {
    let num_vars = num_rows.trailing_zeros() as usize;
    let mut tb = TraceBuilder::new(&CpuFetchColumns::build_layout(), num_vars)?;

    for (i, instr) in instructions.iter().enumerate() {
        let pc_bytes = instr.pc_bytes();
        tb.set_b32(CpuFetchColumns::PC_B0, i, Block32::from(pc_bytes[0] as u32))?;
        tb.set_b32(CpuFetchColumns::PC_B1, i, Block32::from(pc_bytes[1] as u32))?;
        tb.set_b32(CpuFetchColumns::PC_B2, i, Block32::from(pc_bytes[2] as u32))?;
        tb.set_b32(CpuFetchColumns::PC_B3, i, Block32::from(pc_bytes[3] as u32))?;

        tb.set_b32(
            CpuFetchColumns::OPCODE,
            i,
            Block32::from(instr.opcode() as u32),
        )?;

        let args = instr.args();
        tb.set_b32(CpuFetchColumns::ARG0, i, Block32::from(args[0] as u32))?;
        tb.set_b32(CpuFetchColumns::ARG1, i, Block32::from(args[1] as u32))?;
        tb.set_b32(CpuFetchColumns::ARG2, i, Block32::from(args[2] as u32))?;
        tb.set_bit(CpuFetchColumns::SELECTOR, i, Bit::ONE)?;
    }

    let mut trace = tb.build();

    // Append ROM chiplet columns
    let rom = generate_rom_trace(instructions, num_rows)?;
    for col in rom.into_columns() {
        trace.add_column(col)?;
    }

    Ok(trace)
}

fn main() {
    common::init("ROM Chiplet");

    let num_vars = 20;
    let num_rows = 1 << num_vars;

    // Use ZK mode with 2 blinding columns
    // to see maximum memory pressure.
    let mut config = Config {
        sumcheck_blinding_factor: 2,
        ..Config::default()
    };

    OsRng.try_fill_bytes(&mut config.matrix_seed).unwrap();

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    println!(
        "Rows: 2^{} ({} million)",
        num_vars,
        num_rows as f64 / 1_000_000.0
    );

    let trace = common::phase("Trace Generation", || {
        let instructions = generate_rom_instructions(num_rows);
        generate_combined_trace(&instructions, num_rows).unwrap()
    });

    let air = RomInlineChipletProgram { num_rows };
    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(trace);

    let proof = common::phase("Proving", || {
        prove(
            b"ROM_Example",
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

    let mut verifier_transcript = Transcript::<H>::new(b"ROM_Example");

    let is_valid = common::phase("Verifying", || {
        HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut verifier_transcript, &config)
            .expect("Verifier failed")
    });

    common::result(is_valid);
}
