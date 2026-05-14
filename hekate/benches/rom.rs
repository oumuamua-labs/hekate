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

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hekate::core::config::Config;
use hekate::core::trace::{ColumnTrace, ColumnType, TraceColumn};
use hekate::crypto::transcript::Transcript;
use hekate::crypto::DefaultHasher;
use hekate::math::{Block128, TowerField};
use hekate::program::{Air, Program, ProgramInstance, ProgramWitness};
use hekate_core::trace::IntoTraceColumn;
use hekate_gadgets::{
    generate_rom_trace, CpuFetchColumns, CpuFetchUnit, Instruction, RomChiplet, RomColumns,
};
use hekate_math::{Bit, Block32};
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::constraint::ConstraintAst;
use hekate_program::permutation::PermutationCheckSpec;
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;
use std::hint::black_box;
use std::time::Duration;

type F = Block128;
type H = DefaultHasher;

// --- 1. Define Combined AIR for Benchmarking ---

#[derive(Clone)]
struct RomBenchAir;

impl Air<F> for RomBenchAir {
    fn num_columns(&self) -> usize {
        // CPU Fetch (6 cols) + ROM (9 cols)
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
        let cpu_spec = CpuFetchUnit::linking_spec();
        let mut rom_spec = RomChiplet::linking_spec();

        // Adjust ROM indices to account
        // for CPU columns offset.
        rom_spec.shift_column_indices(CpuFetchColumns::NUM_COLUMNS);

        vec![
            (RomChiplet::BUS_ID.to_string(), cpu_spec),
            (RomChiplet::BUS_ID.to_string(), rom_spec),
        ]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(CpuFetchColumns::SELECTOR));

        cs.build()
    }
}

impl Program<F> for RomBenchAir {}

// --- 2. Trace Generation Helpers ---

fn generate_rom_instructions(num_rows: usize) -> Vec<Instruction> {
    let mut instructions = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        let pc = (i * 4) as u32;
        let opcode = ((i % 256) as u8).wrapping_add(1);
        let args = [
            (i & 0xFF) as u8,
            ((i >> 8) & 0xFF) as u8,
            ((i >> 16) & 0xFF) as u8,
        ];

        instructions.push(Instruction::new(pc, opcode, args));
    }

    instructions
}

fn generate_combined_trace(instructions: &[Instruction], num_rows: usize) -> ColumnTrace {
    let num_vars = (num_rows as f64).log2() as usize;
    let mut trace = ColumnTrace::new(num_vars).unwrap();

    // 1. Generate CPU columns (simulation: CPU fetches instructions linearly)
    // CPU Fetch Unit (6 cols): PC(4), Opcode(1), Selector(1)

    let mut pc_cols = (0..4)
        .map(|_| Vec::with_capacity(num_rows))
        .collect::<Vec<_>>();
    let mut op_col = Vec::with_capacity(num_rows);
    let mut sel_col = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        if i < instructions.len() {
            let instr = &instructions[i];
            let pc_bytes = instr.pc_bytes();

            for (col, &byte) in pc_cols.iter_mut().zip(pc_bytes.iter()) {
                col.push(Block32::from(byte as u32));
            }

            op_col.push(Block32::from(instr.opcode() as u32));
            sel_col.push(Bit::ONE);
        } else {
            // Padding
            for col in &mut pc_cols {
                col.push(Block32::ZERO);
            }

            op_col.push(Block32::ZERO);
            sel_col.push(Bit::ZERO);
        }
    }

    // Add CPU columns to trace
    for col in pc_cols {
        trace.add_column(col.into_trace_column()).unwrap();
    }

    trace.add_column(op_col.into_trace_column()).unwrap();
    trace.add_column(TraceColumn::Bit(sel_col)).unwrap();

    // 2. Generate ROM columns (9 cols) using library helper
    let rom = generate_rom_trace(instructions, num_rows).unwrap();
    for col in rom.into_columns() {
        trace.add_column(col).unwrap();
    }

    trace
}

// --- 3. Benchmarks ---

fn bench_rom_e2e(c: &mut Criterion) {
    let mut group = c.benchmark_group("ROM");

    // Proving is heavy, reduce sample size
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    // Config for benchmarking (No ZK for speed, minimal security check)
    let seed = [0xAAu8; 32];
    let config = Config {
        num_queries: 4,
        min_security_bits: 0,
        sumcheck_blinding_factor: 0,
        ..Config::default()
    };

    // Benchmark sizes: 2^14 (16k rows) to 2^24 (1M rows)
    for &num_vars in &[16, 18, 20, 24] {
        let num_rows = 1usize << num_vars;
        group.throughput(Throughput::Elements(num_rows as u64));

        let instructions = generate_rom_instructions(num_rows);
        let trace = generate_combined_trace(&instructions, num_rows);
        let air = RomBenchAir;
        let instance = ProgramInstance::new(num_rows, vec![]);
        let witness = ProgramWitness::new(trace);

        // BENCHMARK: PROVE
        group.bench_with_input(BenchmarkId::new("Prove", num_vars), &num_vars, |b, _| {
            b.iter(|| {
                let proof = prove(
                    black_box(b"ROM_Bench"),
                    black_box(&air),
                    black_box(&instance),
                    black_box(&witness),
                    black_box(&config),
                    black_box(seed),
                    None,
                )
                .unwrap();
                black_box(proof);
            })
        });

        // Pre-generate proof for Verification benchmark
        let proof = prove(b"ROM_Bench", &air, &instance, &witness, &config, seed, None).unwrap();

        // BENCHMARK: VERIFY
        group.bench_with_input(BenchmarkId::new("Verify", num_vars), &num_vars, |b, _| {
            b.iter(|| {
                let mut transcript = Transcript::<H>::new(b"ROM_Bench");
                let result = HekateVerifier::<F, H>::verify(
                    black_box(&air),
                    black_box(&instance),
                    black_box(&proof),
                    black_box(&mut transcript),
                    black_box(&config),
                )
                .unwrap();
                assert!(result, "Verification failed in benchmark!");
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_rom_e2e);
criterion_main!(benches);
