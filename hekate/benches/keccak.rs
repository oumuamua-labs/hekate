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
use hekate::core::errors;
use hekate::core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate::math::{Block128, TowerField};
use hekate_gadgets::{
    generate_keccak_trace, CpuKeccakColumns, CpuKeccakUnit, KeccakChiplet, KeccakColumns,
    KeccakWitness,
};
use hekate_math::{Bit, Block64};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::constraint::ConstraintAst;
use hekate_program::expander::VirtualExpander;
use hekate_program::permutation::PermutationCheckSpec;
use hekate_program::{Air, InlineKernelHint, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use rand::{rngs::OsRng, TryRngCore};
use std::hint::black_box;
use std::time::Duration;

type F = Block128;

const KECCAK_OFFSET: usize = CpuKeccakColumns::NUM_COLUMNS;

// =======================================
// 1. KECCAK BENCH AIR
// =======================================

#[derive(Clone)]
struct KeccakBenchAir {
    num_rows: usize,
}

impl Air<F> for KeccakBenchAir {
    fn num_columns(&self) -> usize {
        CpuKeccakColumns::NUM_COLUMNS + KeccakColumns::NUM_COLUMNS
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(|| {
            let mut cols = CpuKeccakColumns::build_layout();
            cols.extend(vec![ColumnType::B64; 26]);
            cols.push(ColumnType::B32);
            cols.extend(vec![ColumnType::Bit; 2]);

            cols
        })
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        let cpu_spec = CpuKeccakUnit::linking_spec();

        let mut keccak_spec = KeccakChiplet::linking_spec();
        keccak_spec.shift_column_indices(KECCAK_OFFSET);

        vec![
            (KeccakChiplet::BUS_ID.into(), cpu_spec),
            (KeccakChiplet::BUS_ID.into(), keccak_spec),
        ]
    }

    fn virtual_expander(&self) -> Option<&VirtualExpander> {
        static E: std::sync::OnceLock<VirtualExpander> = std::sync::OnceLock::new();
        Some(E.get_or_init(|| {
            VirtualExpander::new()
                .pass_through(25, ColumnType::B64)
                .control_bits(1)
                .expand_bits(25, ColumnType::B64)
                .expand_bits(1, ColumnType::B64)
                .reuse_pass_through(KECCAK_OFFSET, 25)
                .pass_through(1, ColumnType::B32)
                .control_bits(2)
                .build()
                .expect("keccak bench expander")
        }))
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(CpuKeccakColumns::SELECTOR));

        let mut ast = cs.build();

        let mut keccak_ast = KeccakChiplet::new(self.num_rows).constraint_ast();
        keccak_ast.arena.shift_cells(KECCAK_OFFSET);

        ast.merge(keccak_ast);

        ast
    }

    fn inline_chiplets(&self) -> errors::Result<Vec<ChipletDef<F>>> {
        Ok(vec![ChipletDef::from_air(&KeccakChiplet::new(
            self.num_rows,
        ))?])
    }

    fn inline_chiplet_kernels(&self) -> Vec<InlineKernelHint> {
        vec![InlineKernelHint {
            chiplet_idx: 0,
            root_offset: 1,
            column_offset: KECCAK_OFFSET,
        }]
    }
}

impl Program<F> for KeccakBenchAir {}

// =================================================================
// 2. TRACE GENERATION
// =================================================================

fn sponge_calls(message: &[u8]) -> Vec<([Block64; 25], [Block64; 25])> {
    let rate_bytes = 136;

    let mut padded = message.to_vec();
    padded.push(0x01);

    while (padded.len() % rate_bytes) != (rate_bytes - 1) {
        padded.push(0x00);
    }

    padded.push(0x80);

    let mut calls = Vec::new();
    let mut state = [0u64; 25];

    for block in padded.chunks_exact(rate_bytes) {
        for i in 0..17 {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&block[i * 8..(i + 1) * 8]);
            state[i] ^= u64::from_le_bytes(bytes);
        }

        let mut input = [Block64::ZERO; 25];
        for i in 0..25 {
            input[i] = Block64::from(state[i]);
        }

        for round in 0..24 {
            state = KeccakWitness::keccak_f_round(state, KeccakChiplet::ROUND_CONSTANTS[round]);
        }

        let mut output = [Block64::ZERO; 25];
        for i in 0..25 {
            output[i] = Block64::from(state[i]);
        }

        calls.push((input, output));
    }

    calls
}

fn generate_combined_trace(
    calls: &[([Block64; 25], [Block64; 25])],
    inputs: &[[Block64; 25]],
    num_rows: usize,
) -> errors::Result<ColumnTrace> {
    let num_vars = num_rows.trailing_zeros() as usize;

    let layout = CpuKeccakColumns::build_layout();

    let mut tb = TraceBuilder::new(&layout, num_vars)?;
    let mut row = 0;

    for (input, output) in calls {
        assert!(row + 25 <= num_rows, "CPU trace overflow");

        for i in 0..25 {
            tb.set_b64(i, row, input[i])?;
        }

        tb.set_bit(CpuKeccakColumns::SELECTOR, row, Bit::ONE)?;

        row += 24;

        for i in 0..25 {
            tb.set_b64(i, row, output[i])?;
        }

        tb.set_bit(CpuKeccakColumns::SELECTOR, row, Bit::ONE)?;

        row += 1;
    }

    let mut trace = tb.build();

    let pairs: Vec<(u32, u32)> = (0..inputs.len() as u32)
        .map(|k| (25 * k, 25 * k + 24))
        .collect();
    let keccak_trace = generate_keccak_trace(inputs, Some(&pairs), num_rows)?;

    for col in keccak_trace.into_columns() {
        trace.add_column(col)?;
    }

    Ok(trace)
}

// =================================================================
// 3. BENCHMARK EXECUTION
// =================================================================

fn bench_keccak_prove(c: &mut Criterion) {
    let mut group = c.benchmark_group("Keccak");

    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));

    {
        let &num_vars = &15;
        let num_rows = 1usize << num_vars;
        group.throughput(Throughput::Elements(num_rows as u64));

        let max_blocks = num_rows / 25;
        let message_len = max_blocks * 136 - 136;

        let mut message = vec![0u8; message_len];
        OsRng.try_fill_bytes(&mut message).unwrap();

        let calls = sponge_calls(&message);
        let inputs: Vec<[Block64; 25]> = calls.iter().map(|(inp, _)| *inp).collect();
        let trace = generate_combined_trace(&calls, &inputs, num_rows).unwrap();

        let air = KeccakBenchAir { num_rows };
        let instance = ProgramInstance::new(num_rows, vec![]);
        let witness = ProgramWitness::new(trace);

        let mut config = Config {
            sumcheck_blinding_factor: 2,
            ..Config::default()
        };

        OsRng.try_fill_bytes(&mut config.matrix_seed).unwrap();

        let mut blinding_seed = [0u8; 32];
        OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

        group.bench_with_input(BenchmarkId::new("Prove", num_vars), &num_vars, |b, _| {
            b.iter(|| {
                let proof = prove(
                    black_box(b"Keccak_Bench"),
                    black_box(&air),
                    black_box(&instance),
                    black_box(&witness),
                    black_box(&config),
                    black_box(blinding_seed),
                    None,
                );

                black_box(proof.unwrap());
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_keccak_prove);
criterion_main!(benches);
