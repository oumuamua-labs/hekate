// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::trace::{ColumnTrace, TraceBuilder};
use hekate_keccak::{CpuKeccakColumns, KeccakChiplet, KeccakWitness, generate_keccak_trace};
use hekate_math::{Bit, Block64, Block128, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram, Col};
use hekate_program::{ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use rand::{TryRngCore, rngs::OsRng};
use std::hint::black_box;
use std::time::Duration;

type F = Block128;

// =======================================
// 1. KECCAK BENCH AIR
// =======================================

fn build_program(num_rows: usize) -> CircuitProgram<F> {
    let num_blocks = num_rows / KeccakChiplet::BLOCK_ROWS;

    let mut cx = Circuit::<F>::new("KeccakBench", num_rows).unwrap();
    let cpu = cx.schema(&CpuKeccakColumns::build_layout());

    let selector = cpu.at(CpuKeccakColumns::SELECTOR);

    let call_values: Vec<Col> = (0..25)
        .map(|lane| cpu.at(CpuKeccakColumns::LANES + lane))
        .collect();

    cx.call(&KeccakChiplet::service(), &call_values, selector)
        .unwrap();

    cx.fix(
        selector,
        KeccakChiplet::host_selector_shape(KeccakChiplet::BLOCK_ROWS, num_blocks),
    );

    cx.mount(ChipletDef::from_air(&KeccakChiplet::new(num_rows, num_blocks)).unwrap());

    cx.compile().unwrap()
}

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

        for (i, &val) in input.iter().enumerate() {
            tb.set_b64(i, row, val)?;
        }

        tb.set_bit(CpuKeccakColumns::SELECTOR, row, Bit::ONE)?;

        row += 24;

        for (i, &val) in output.iter().enumerate() {
            tb.set_b64(i, row, val)?;
        }

        tb.set_bit(CpuKeccakColumns::SELECTOR, row, Bit::ONE)?;

        row += 1;
    }

    let mut trace = tb.build();

    let keccak_trace = generate_keccak_trace(inputs, num_rows)?;

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

    for &num_vars in &[12usize, 15, 20] {
        let num_rows = 1usize << num_vars;
        let max_blocks = num_rows / KeccakChiplet::BLOCK_ROWS;
        let message_len = max_blocks * 136 - 136;

        group.throughput(Throughput::Bytes((max_blocks * 136) as u64));

        let mut message = vec![0u8; message_len];
        OsRng.try_fill_bytes(&mut message).unwrap();

        let calls = sponge_calls(&message);
        let inputs: Vec<[Block64; 25]> = calls.iter().map(|(inp, _)| *inp).collect();
        let trace = generate_combined_trace(&calls, &inputs, num_rows).unwrap();

        let air = build_program(num_rows);
        let instance = ProgramInstance::new(num_rows, vec![]);
        let witness = ProgramWitness::new(trace);

        let config = Config {
            zero_knowledge: true,
            ..Config::default()
        };

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
