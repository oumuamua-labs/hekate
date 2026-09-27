// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

#[path = "common/mod.rs"]
mod common;

use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::{Block128, TowerField};
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::trace::{ColumnTrace, TraceBuilder};
use hekate_keccak::{CpuKeccakColumns, KeccakChiplet, KeccakColumns, generate_keccak_trace};
use hekate_math::{Bit, Block64};
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram, Col};
use hekate_program::{ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;
use rand::{TryRngCore, rngs::OsRng};

type F = Block128;
type H = DefaultHasher;

// =================================================================
// 1. KECCAK PROGRAM DEFINITION
// =================================================================
//
// CPU-only main trace. Keccak runs as an independent
// chiplet with its own trace, commitment, and ZeroCheck.
// The kernel activates automatically through ChipletDef.

/// CPU columns in `CpuKeccakColumns` order;
/// the circuit handles match the trace generator's schema.
/// The digest is the first 4 lanes of the final state,
/// read where the cadence forces the last output emit.
fn build_program(num_rows: usize) -> errors::Result<CircuitProgram<F>> {
    let num_blocks = num_rows / KeccakChiplet::BLOCK_ROWS;

    let mut cx = Circuit::<F>::new("KeccakIsolated", num_rows)?;

    let cpu = cx.schema(&CpuKeccakColumns::build_layout());

    let selector = cpu.at(CpuKeccakColumns::SELECTOR);

    let call_values: Vec<Col> = (0..25)
        .map(|lane| cpu.at(CpuKeccakColumns::LANES + lane))
        .collect();

    cx.call(&KeccakChiplet::service(), &call_values, selector)?;

    cx.fix(
        selector,
        KeccakChiplet::host_selector_shape(KeccakChiplet::BLOCK_ROWS, num_blocks),
    );

    cx.attach(ChipletDef::from_air(&KeccakChiplet::new(
        num_rows, num_blocks,
    ))?);

    let last_output_row = KeccakChiplet::BLOCK_ROWS * num_blocks - 1;
    for i in 0..4 {
        cx.publish(cpu.at(CpuKeccakColumns::LANES + i), last_output_row);
    }

    cx.compile()
}

// =================================================================
// 2. TRACE GENERATION
// =================================================================

/// Sponge computation:
/// absorb message, produce (input, output)
/// state pairs per permutation.
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

        // Record the input state for the chiplet
        let mut input = [Block64::ZERO; 25];
        for i in 0..25 {
            input[i] = Block64::from(state[i]);
        }

        keccak::Keccak::new().with_f1600(|f| f(&mut state));

        let mut output = [Block64::ZERO; 25];
        for i in 0..25 {
            output[i] = Block64::from(state[i]);
        }

        calls.push((input, output));
    }

    calls
}

/// Generate CPU-side main trace.
/// Writes lanes + selector at I/O rows.
fn generate_cpu_trace(calls: &[([Block64; 25], [Block64; 25])], num_rows: usize) -> ColumnTrace {
    let num_vars = num_rows.trailing_zeros() as usize;
    let layout = CpuKeccakColumns::build_layout();

    let mut tb = TraceBuilder::new(&layout, num_vars).unwrap();
    let mut row = 0;

    for (input, output) in calls {
        assert!(row + 25 <= num_rows, "CPU trace overflow");

        // Input row:
        // write pre-permutation state
        for (i, &val) in input.iter().enumerate() {
            tb.set_b64(i, row, val).unwrap();
        }

        tb.set_bit(CpuKeccakColumns::SELECTOR, row, Bit::ONE)
            .unwrap();

        // 24 rounds (no CPU activity)
        row += 24;

        // Output row:
        // write post-permutation state
        for (i, &val) in output.iter().enumerate() {
            tb.set_b64(i, row, val).unwrap();
        }

        tb.set_bit(CpuKeccakColumns::SELECTOR, row, Bit::ONE)
            .unwrap();

        row += 1;
    }

    tb.build()
}

// =================================================================
// 3. MAIN
// =================================================================
fn main() {
    common::init("Keccak-f[1600]");

    // Setup parameters
    let num_vars = common::num_vars(15);
    let num_rows = 1 << num_vars;

    let config = Config {
        zero_knowledge: common::zero_knowledge(),
        ..Config::default()
    };

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    println!(
        "Rows: 2^{} ({} permutations)",
        num_vars,
        num_rows / KeccakChiplet::BLOCK_ROWS
    );
    println!(
        "Total Columns: {} CPU + {} Keccak chiplet",
        CpuKeccakColumns::NUM_COLUMNS,
        KeccakColumns::NUM_COLUMNS
    );

    let (cpu_trace, keccak_trace, air, digest) = common::phase("Trace Generation", || {
        let max_blocks = num_rows / KeccakChiplet::BLOCK_ROWS;
        let message_len = max_blocks * 136 - 136;

        println!("   Max Blocks: {}", max_blocks);
        println!(
            "   Message Len: {} bytes (leaving 1 block for padding)",
            message_len
        );

        let mut message = vec![0u8; message_len];
        OsRng.try_fill_bytes(&mut message).unwrap();

        let calls = sponge_calls(&message);
        let inputs: Vec<[Block64; 25]> = calls.iter().map(|(inp, _)| *inp).collect();

        let final_state = calls.last().expect("at least one block").1;
        let digest: [Block64; 4] = [
            final_state[0],
            final_state[1],
            final_state[2],
            final_state[3],
        ];

        let cpu = generate_cpu_trace(&calls, num_rows);
        let keccak = generate_keccak_trace(&inputs, num_rows).unwrap();

        let air = build_program(num_rows).unwrap();

        (cpu, keccak, air, digest)
    });

    print!("Keccak-256 digest (via `keccak` crate's f1600): 0x");
    for lane in &digest {
        for byte in lane.0.to_le_bytes() {
            print!("{:02x}", byte);
        }
    }
    println!();

    let public_inputs: Vec<F> = digest.iter().map(|&lane| F::from(lane)).collect();

    let instance = ProgramInstance::new(num_rows, public_inputs);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![keccak_trace]);

    let proof = common::phase("Proving", || {
        prove(
            b"Keccak_E2E",
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

    let mut verifier_transcript = Transcript::<H>::new(b"Keccak_E2E");
    let pinned_id = common::audited_id(&air);

    let is_valid = common::phase_with_mem("Verifying", || {
        HekateVerifier::<F, H>::verify(
            &pinned_id,
            &air,
            &instance,
            &proof,
            &mut verifier_transcript,
            &config,
        )
        .expect("Verifier failed")
    });

    common::result(is_valid);
}
