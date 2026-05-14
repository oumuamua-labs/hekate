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

use hekate::core::trace::{ColumnTrace, ColumnType};
use hekate::crypto::transcript::Transcript;
use hekate::crypto::DefaultHasher;
use hekate::math::{Block128, TowerField};
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::trace::TraceBuilder;
use hekate_gadgets::{
    generate_keccak_trace, CpuKeccakColumns, CpuKeccakUnit, KeccakChiplet, KeccakColumns,
};
use hekate_math::{Bit, Block64};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::constraint::{BoundaryConstraint, ConstraintAst};
use hekate_program::expander::VirtualExpander;
use hekate_program::permutation::PermutationCheckSpec;
use hekate_program::{Air, InlineKernelHint, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;
use rand::{rngs::OsRng, TryRngCore};

type F = Block128;
type H = DefaultHasher;

const KECCAK_OFFSET: usize = CpuKeccakColumns::NUM_COLUMNS;

// =================================================================
// 1. KECCAK PROGRAM DEFINITION
// =================================================================
//
// Mounted trace:
// CPU columns + Keccak columns in one trace.
// Keccak's 28 physical columns are virtually
// expanded to 1691 columns via
// KeccakTrace::get_poly_variants().
// The constraint AST references virtual
// column indices (shifted by KECCAK_OFFSET).

#[derive(Clone)]
struct KeccakInlineChipletProgram {
    num_rows: usize,
}

impl Air<F> for KeccakInlineChipletProgram {
    fn num_columns(&self) -> usize {
        CpuKeccakColumns::NUM_COLUMNS + KeccakColumns::NUM_COLUMNS
    }

    fn boundary_constraints(&self) -> Vec<BoundaryConstraint<F>> {
        // Keccak-256 digest = first 4 lanes of final state.
        // Last output row sits at 25*max_blocks-1, which
        // only equals num_rows-1 when 25 divides num_rows;
        // trailing rows beyond that are zero-padded and
        // must not be pinned.
        let max_blocks = self.num_rows / 25;
        let last_output_row = 25 * max_blocks - 1;

        (0..4)
            .map(|i| BoundaryConstraint::with_public_input(i, last_output_row, i))
            .collect()
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(|| {
            let mut cols = CpuKeccakColumns::build_layout();

            // Keccak physical:
            // 25 lanes (B64)
            //   + 1 RC (B64)
            //   + 1 REQUEST_IDX (B32)
            //   + 2 selectors (Bit)
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

        // Both endpoints must share the bus_id,
        // otherwise LogUp's cross-bus cancellation
        // cannot pair them.
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
                .expect("keccak inline expander")
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
        // Program AST:
        //   root 0            = CPU SELECTOR booleanity
        //   roots [1, 1+1629) = Keccak chiplet
        // Keccak columns sit at `KECCAK_OFFSET` after the CPU block.
        vec![InlineKernelHint {
            chiplet_idx: 0,
            root_offset: 1,
            column_offset: KECCAK_OFFSET,
        }]
    }
}

impl Program<F> for KeccakInlineChipletProgram {
    fn num_public_inputs(&self) -> usize {
        4
    }
}

// =================================================================
// 2. TRACE GENERATION
// =================================================================

/// Sponge:
/// absorb message, produce (input, output)
/// pairs per permutation.
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

        keccak::Keccak::new().with_f1600(|f| f(&mut state));

        let mut output = [Block64::ZERO; 25];
        for i in 0..25 {
            output[i] = Block64::from(state[i]);
        }

        calls.push((input, output));
    }

    calls
}

/// Build combined trace:
/// CPU columns + Keccak columns.
fn generate_combined_trace(
    calls: &[([Block64; 25], [Block64; 25])],
    inputs: &[[Block64; 25]],
    num_rows: usize,
) -> errors::Result<ColumnTrace> {
    let num_vars = num_rows.trailing_zeros() as usize;

    // CPU trace (26 physical columns)
    let layout = CpuKeccakColumns::build_layout();

    let mut tb = TraceBuilder::new(&layout, num_vars)?;
    let mut row = 0;

    for (input, output) in calls {
        assert!(row + 25 <= num_rows, "CPU trace overflow");

        // Input row
        for i in 0..25 {
            tb.set_b64(i, row, input[i])?;
        }

        tb.set_bit(CpuKeccakColumns::SELECTOR, row, Bit::ONE)?;

        row += 24;

        // Output row
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
// 4. MAIN
// =================================================================
fn main() {
    common::init("Keccak-f[1600]");
    hekate_prover_sys::init_tracing();

    // Setup parameters
    let num_vars: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);

    let num_rows = 1 << num_vars;

    let mut config = Config {
        sumcheck_blinding_factor: 2, // Enable ZK
        ..Config::default()
    };

    OsRng.try_fill_bytes(&mut config.matrix_seed).unwrap();

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    println!("Rows: 2^{} ({} permutations)", num_vars, num_rows / 25);
    println!(
        "Physical: {} CPU + {} Keccak = {} columns",
        CpuKeccakColumns::NUM_COLUMNS,
        28,
        CpuKeccakColumns::NUM_COLUMNS + 28
    );
    println!(
        "Virtual: {} CPU + {} Keccak = {} columns",
        CpuKeccakColumns::NUM_COLUMNS,
        KeccakColumns::NUM_COLUMNS,
        CpuKeccakColumns::NUM_COLUMNS + KeccakColumns::NUM_COLUMNS
    );

    let (trace, air, digest) = common::phase("Trace Generation", || {
        let max_blocks = num_rows / 25;
        let message_len = max_blocks * 136 - 136;

        println!("   Max Blocks: {}", max_blocks);
        println!("   Message Len: {} bytes", message_len);

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

        let trace = generate_combined_trace(&calls, &inputs, num_rows).unwrap();
        let air = KeccakInlineChipletProgram { num_rows };

        (trace, air, digest)
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
    let witness = ProgramWitness::new(trace);

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

    let is_valid = common::phase("Verifying", || {
        HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut verifier_transcript, &config)
            .expect("Verifier failed")
    });

    common::result(is_valid);
}
