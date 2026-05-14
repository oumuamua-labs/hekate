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

use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::{Bit, Block8, Block128, TowerField};
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate_aes::{
    Aes128Chiplet, Aes256Chiplet, AesRound128Air, AesRound256Air, CpuAes128Columns, CpuAes128Unit,
    CpuAes256Columns, CpuAes256Unit, PhysAes128Columns, PhysAes256Columns,
    trace::{Aes128Call, Aes256Call, expand_key, expand_key_256},
};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::permutation::PermutationCheckSpec;
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;
use rand::{TryRngCore, rngs::OsRng};
type F = Block128;
type H = DefaultHasher;

// =================================================================
// 1. INPUTS
// =================================================================

const NUM_BLOCKS: usize = 31250;

#[rustfmt::skip]
const FIPS128_KEY: [u8; 16] = [
    0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
    0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
];

/// FIPS 197 Appendix C.3.
#[rustfmt::skip]
const FIPS256_KEY: [u8; 32] = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
];

// =================================================================
// 2. AES PROGRAM DEFINITIONS
// =================================================================

const CPU_IO_PER_BLOCK: usize = 2;

#[derive(Clone)]
struct Aes128ExampleProgram {
    aes: Aes128Chiplet<F>,
}

impl Air<F> for Aes128ExampleProgram {
    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(CpuAes128Columns::build_layout)
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![
            (
                AesRound128Air::LINK_BUS_ID.into(),
                CpuAes128Unit::linking_spec(),
            ),
            (
                AesRound128Air::KEY_BUS_ID.into(),
                CpuAes128Unit::key_linking_spec(),
            ),
        ]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(CpuAes128Columns::SELECTOR));
        cs.assert_boolean(cs.col(CpuAes128Columns::KEY_SELECTOR));

        cs.build()
    }
}

impl Program<F> for Aes128ExampleProgram {
    fn chiplet_defs(&self) -> errors::Result<Vec<ChipletDef<F>>> {
        self.aes.composite().flatten_defs()
    }
}

#[derive(Clone)]
struct Aes256ExampleProgram {
    aes: Aes256Chiplet<F>,
}

impl Air<F> for Aes256ExampleProgram {
    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(CpuAes256Columns::build_layout)
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![
            (
                AesRound256Air::LINK_BUS_ID.into(),
                CpuAes256Unit::linking_spec(),
            ),
            (
                AesRound256Air::KEY_BUS_ID.into(),
                CpuAes256Unit::key_linking_spec(),
            ),
        ]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(CpuAes256Columns::SELECTOR));
        cs.assert_boolean(cs.col(CpuAes256Columns::KEY_SELECTOR));

        cs.build()
    }
}

impl Program<F> for Aes256ExampleProgram {
    fn chiplet_defs(&self) -> errors::Result<Vec<ChipletDef<F>>> {
        self.aes.composite().flatten_defs()
    }
}

// =================================================================
// 3. SHARED HELPERS
// =================================================================

/// Output row = block_idx * rows_per_block + (rows_per_block - 1).
fn extract_ciphertext(
    chiplet_trace: &ColumnTrace,
    state_in_col: usize,
    rows_per_block: usize,
    block_idx: usize,
) -> [u8; 16] {
    let output_row = block_idx * rows_per_block + (rows_per_block - 1);

    let mut ct = [0u8; 16];
    for (j, byte) in ct.iter_mut().enumerate() {
        *byte = chiplet_trace.columns[state_in_col + j]
            .as_b8_slice()
            .unwrap()[output_row]
            .to_tower()
            .0;
    }

    ct
}

fn prove_and_verify<P: Program<F> + Air<F>>(
    air: &P,
    cpu_rows: usize,
    cpu_trace: ColumnTrace,
    chiplet_traces: Vec<ColumnTrace>,
    transcript_label: &'static [u8],
) {
    // Phase 3:
    // Prove
    let mut config = Config {
        sumcheck_blinding_factor: 2,
        ..Config::default()
    };

    OsRng.try_fill_bytes(&mut config.matrix_seed).unwrap();

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    let instance = ProgramInstance::new(cpu_rows, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(chiplet_traces);

    let proof = common::phase("Proving", || {
        prove(
            transcript_label,
            air,
            &instance,
            &witness,
            &config,
            blinding_seed,
            None,
        )
        .expect("Prover failed")
    });

    common::proof_breakdown(&proof);

    // Phase 4:
    // Verify
    let is_valid = common::phase("Verifying", || {
        let mut vt = Transcript::<H>::new(transcript_label);
        HekateVerifier::<F, H>::verify(air, &instance, &proof, &mut vt, &config)
            .expect("Verifier failed")
    });

    common::result(is_valid);
}

fn print_layout(key_hex: &str, rows_per_block: usize, sbox_rounds_per_block: usize) {
    let chiplet_rows = (NUM_BLOCKS * rows_per_block).next_power_of_two();
    let cpu_rows = (NUM_BLOCKS * CPU_IO_PER_BLOCK).next_power_of_two();
    let sbox_rom_rows = (NUM_BLOCKS * sbox_rounds_per_block).next_power_of_two();

    println!("Key:             {}", key_hex);
    println!("Blocks:          {}", NUM_BLOCKS);
    println!(
        "Chiplet rows:    {} (2^{})",
        chiplet_rows,
        chiplet_rows.trailing_zeros()
    );
    println!(
        "CPU rows:        {} (2^{})",
        cpu_rows,
        cpu_rows.trailing_zeros()
    );
    println!(
        "S-box ROM rows:  {} (2^{})",
        sbox_rom_rows,
        sbox_rom_rows.trailing_zeros()
    );
}

fn generate_plaintexts() -> Vec<[u8; 16]> {
    let mut plaintexts = vec![[0u8; 16]; NUM_BLOCKS];
    for pt in &mut plaintexts {
        OsRng.try_fill_bytes(pt).unwrap();
    }

    plaintexts
}

fn print_sample_blocks(plaintexts: &[[u8; 16]], ciphertexts: &[[u8; 16]]) {
    println!(
        "   Block 0:   {:02x?} → {:02x?}",
        plaintexts[0], ciphertexts[0]
    );

    if NUM_BLOCKS > 1 {
        println!(
            "   Block {}: {:02x?} → {:02x?}",
            NUM_BLOCKS - 1,
            plaintexts[NUM_BLOCKS - 1],
            ciphertexts[NUM_BLOCKS - 1]
        );
    }
}

// =================================================================
// 4. AES-128 PIPELINE
// =================================================================

fn run_aes128() {
    const ROWS_PER_BLOCK: usize = 11;
    const SBOX_ROUNDS: usize = 10;

    // Phase 1:
    // Setup
    let round_keys = expand_key(&FIPS128_KEY);
    let chiplet_rows = (NUM_BLOCKS * ROWS_PER_BLOCK).next_power_of_two();
    let cpu_rows = (NUM_BLOCKS * CPU_IO_PER_BLOCK).next_power_of_two();
    let sbox_rom_rows = (NUM_BLOCKS * SBOX_ROUNDS).next_power_of_two();

    print_layout(
        &format!("{:02x?}", FIPS128_KEY),
        ROWS_PER_BLOCK,
        SBOX_ROUNDS,
    );

    // Phase 2:
    // Trace generation
    let (air, cpu_trace, chiplet_traces) = common::phase("Trace Generation", || {
        let plaintexts = generate_plaintexts();

        let calls: Vec<Aes128Call> = plaintexts
            .iter()
            .map(|pt| Aes128Call {
                key: FIPS128_KEY,
                plaintext: *pt,
                round_keys,
            })
            .collect();

        let aes = Aes128Chiplet::<F>::new(chiplet_rows, sbox_rom_rows).unwrap();
        let chiplet_traces = aes.generate_traces(&calls).unwrap();

        let ciphertexts: Vec<[u8; 16]> = (0..NUM_BLOCKS)
            .map(|i| {
                extract_ciphertext(
                    &chiplet_traces[0],
                    PhysAes128Columns::P_STATE_IN,
                    ROWS_PER_BLOCK,
                    i,
                )
            })
            .collect();

        print_sample_blocks(&plaintexts, &ciphertexts);

        let cpu_trace = build_cpu128_trace(&calls, &ciphertexts, cpu_rows);
        let air = Aes128ExampleProgram { aes };

        (air, cpu_trace, chiplet_traces)
    });

    prove_and_verify(&air, cpu_rows, cpu_trace, chiplet_traces, b"AES128_Example");
}

fn build_cpu128_trace(
    calls: &[Aes128Call],
    ciphertexts: &[[u8; 16]],
    num_rows: usize,
) -> ColumnTrace {
    let num_vars = num_rows.trailing_zeros() as usize;

    let mut row = 0;
    let mut tb = TraceBuilder::new(&CpuAes128Columns::build_layout(), num_vars).unwrap();

    for (call, ct) in calls.iter().zip(ciphertexts) {
        // Input row:
        // plaintext XOR K0 + key binding
        for j in 0..16 {
            let whitened = call.plaintext[j] ^ call.round_keys[0][j];
            tb.set_b8(CpuAes128Columns::DATA + j, row, Block8(whitened))
                .unwrap();
            tb.set_b8(CpuAes128Columns::KEY + j, row, Block8(call.key[j]))
                .unwrap();
        }

        tb.set_bit(CpuAes128Columns::SELECTOR, row, Bit::ONE)
            .unwrap();
        tb.set_bit(CpuAes128Columns::KEY_SELECTOR, row, Bit::ONE)
            .unwrap();

        row += 1;

        // Output row:
        // ciphertext
        for (j, &byte) in ct.iter().enumerate() {
            tb.set_b8(CpuAes128Columns::DATA + j, row, Block8(byte))
                .unwrap();
        }

        tb.set_bit(CpuAes128Columns::SELECTOR, row, Bit::ONE)
            .unwrap();

        row += 1;
    }

    tb.build()
}

// =================================================================
// 5. AES-256 PIPELINE
// =================================================================

fn run_aes256() {
    const ROWS_PER_BLOCK: usize = 15;
    const SBOX_ROUNDS: usize = 14;

    // Phase 1:
    // Setup
    let round_keys = expand_key_256(&FIPS256_KEY);
    let chiplet_rows = (NUM_BLOCKS * ROWS_PER_BLOCK).next_power_of_two();
    let cpu_rows = (NUM_BLOCKS * CPU_IO_PER_BLOCK).next_power_of_two();
    let sbox_rom_rows = (NUM_BLOCKS * SBOX_ROUNDS).next_power_of_two();

    print_layout(
        &format!("{:02x?}", FIPS256_KEY),
        ROWS_PER_BLOCK,
        SBOX_ROUNDS,
    );

    // Phase 2:
    // Trace generation
    let (air, cpu_trace, chiplet_traces) = common::phase("Trace Generation", || {
        let plaintexts = generate_plaintexts();

        let calls: Vec<Aes256Call> = plaintexts
            .iter()
            .map(|pt| Aes256Call {
                key: FIPS256_KEY,
                plaintext: *pt,
                round_keys,
            })
            .collect();

        let aes = Aes256Chiplet::<F>::new(chiplet_rows, sbox_rom_rows).unwrap();
        let chiplet_traces = aes.generate_traces(&calls).unwrap();

        let ciphertexts: Vec<[u8; 16]> = (0..NUM_BLOCKS)
            .map(|i| {
                extract_ciphertext(
                    &chiplet_traces[0],
                    PhysAes256Columns::P_STATE_IN,
                    ROWS_PER_BLOCK,
                    i,
                )
            })
            .collect();

        print_sample_blocks(&plaintexts, &ciphertexts);

        let cpu_trace = build_cpu256_trace(&calls, &ciphertexts, cpu_rows);
        let air = Aes256ExampleProgram { aes };

        (air, cpu_trace, chiplet_traces)
    });

    prove_and_verify(&air, cpu_rows, cpu_trace, chiplet_traces, b"AES256_Example");
}

fn build_cpu256_trace(
    calls: &[Aes256Call],
    ciphertexts: &[[u8; 16]],
    num_rows: usize,
) -> ColumnTrace {
    let num_vars = num_rows.trailing_zeros() as usize;

    let mut row = 0;
    let mut tb = TraceBuilder::new(&CpuAes256Columns::build_layout(), num_vars).unwrap();

    for (call, ct) in calls.iter().zip(ciphertexts) {
        for j in 0..16 {
            let whitened = call.plaintext[j] ^ call.round_keys[0][j];
            tb.set_b8(CpuAes256Columns::DATA + j, row, Block8(whitened))
                .unwrap();
        }

        for j in 0..32 {
            tb.set_b8(CpuAes256Columns::KEY + j, row, Block8(call.key[j]))
                .unwrap();
        }

        tb.set_bit(CpuAes256Columns::SELECTOR, row, Bit::ONE)
            .unwrap();
        tb.set_bit(CpuAes256Columns::KEY_SELECTOR, row, Bit::ONE)
            .unwrap();

        row += 1;

        for (j, &byte) in ct.iter().enumerate() {
            tb.set_b8(CpuAes256Columns::DATA + j, row, Block8(byte))
                .unwrap();
        }

        tb.set_bit(CpuAes256Columns::SELECTOR, row, Bit::ONE)
            .unwrap();

        row += 1;
    }

    tb.build()
}

// =================================================================
// 6. MAIN
// =================================================================

fn main() {
    let level = std::env::args().nth(1).unwrap_or_else(|| "128".to_string());

    match level.as_str() {
        "128" => {
            common::init("AES-128 Chiplet");
            run_aes128();
        }
        "256" => {
            common::init("AES-256 Chiplet");
            run_aes256();
        }
        other => {
            eprintln!("Usage: aes_chiplet [128|256] (got {:?})", other);
            std::process::exit(1);
        }
    }
}
