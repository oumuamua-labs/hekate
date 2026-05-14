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
use hekate::core::trace::ColumnType;
use hekate::crypto::transcript::Transcript;
use hekate::crypto::DefaultHasher;
use hekate::math::{Block128, TowerField};
use hekate_core::trace::TraceBuilder;
use hekate_math::matrix::ByteSparseMatrix;
use hekate_math::{Bit, Block32, Flat, HardwareField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::constraint::ConstraintAst;
use hekate_program::expander::VirtualExpander;
use hekate_program::{Air, LagrangePin, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;

type F = Block128;
type H = DefaultHasher;

const MAGIC: u32 = 0xDEAD_BEEF;

const NUM_VARS: usize = 4;
const NUM_ROWS: usize = 1 << NUM_VARS;

// =========================================================
// Synthetic Chiplet:
// 1×B32 -> 32 virtual bits + 1 control
// =========================================================

#[derive(Clone)]
struct PackedBitChiplet {
    expander: VirtualExpander,
}

impl PackedBitChiplet {
    fn new() -> Self {
        Self {
            expander: VirtualExpander::new()
                .expand_bits(1, ColumnType::B32)
                .control_bits(1)
                .build()
                .expect("PackedBitChiplet expander"),
        }
    }
}

impl Air<F> for PackedBitChiplet {
    fn name(&self) -> String {
        "packed_bit".to_string()
    }

    fn column_layout(&self) -> &'static [ColumnType] {
        &[ColumnType::B32, ColumnType::Bit]
    }

    fn lagrange_pinned_columns(&self) -> Vec<LagrangePin> {
        vec![LagrangePin::last_row(32)]
    }

    fn virtual_expander(&self) -> Option<&VirtualExpander> {
        Some(&self.expander)
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        let bit0 = cs.col(0);
        let sel = cs.col(32);

        cs.constrain(sel * bit0 * (bit0 + cs.one()));

        cs.build()
    }
}

#[derive(Clone)]
struct PackedBitHost {
    defs: Vec<ChipletDef<F>>,
}

impl Air<F> for PackedBitHost {
    fn name(&self) -> String {
        "packed_bit_host".to_string()
    }

    fn column_layout(&self) -> &'static [ColumnType] {
        &[ColumnType::Bit]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(0));

        cs.build()
    }
}

impl Program<F> for PackedBitHost {
    fn chiplet_defs(&self) -> hekate_core::errors::Result<Vec<ChipletDef<F>>> {
        Ok(self.defs.clone())
    }
}

// =========================================================
// Helpers
// =========================================================

fn make_chiplet_trace(magic: u32) -> hekate_core::trace::ColumnTrace {
    let layout = [ColumnType::B32, ColumnType::Bit];

    let mut tb = TraceBuilder::new(&layout, NUM_VARS).unwrap();

    tb.set_b32(0, 0, Block32(magic)).unwrap();

    for i in 0..NUM_ROWS {
        tb.set_bit(
            1,
            i,
            if i < NUM_ROWS - 1 {
                Bit::ONE
            } else {
                Bit::ZERO
            },
        )
        .unwrap();
    }

    tb.build()
}

fn make_main_trace() -> hekate_core::trace::ColumnTrace {
    let layout = [ColumnType::Bit];

    let mut tb = TraceBuilder::new(&layout, NUM_VARS).unwrap();

    for i in 0..NUM_ROWS {
        tb.set_bit(
            0,
            i,
            if i < NUM_ROWS - 1 {
                Bit::ONE
            } else {
                Bit::ZERO
            },
        )
        .unwrap();
    }

    tb.build()
}

fn make_test_system() -> (
    PackedBitHost,
    ProgramInstance<F>,
    ProgramWitness<F, hekate_core::trace::ColumnTrace>,
) {
    let chiplet = PackedBitChiplet::new();
    let air = PackedBitHost {
        defs: vec![ChipletDef::from_air(&chiplet).unwrap()],
    };
    let instance = ProgramInstance::new(NUM_ROWS, vec![]);
    let witness =
        ProgramWitness::new(make_main_trace()).with_chiplets(vec![make_chiplet_trace(MAGIC)]);

    (air, instance, witness)
}

fn zk_config() -> Config {
    Config {
        num_queries: 4,
        min_security_bits: 0,
        sumcheck_blinding_factor: 2,
        ldt_blinding_factor: 4,
        ..Config::default()
    }
}

fn no_zk_config() -> Config {
    Config {
        num_queries: 4,
        min_security_bits: 0,
        sumcheck_blinding_factor: 0,
        ldt_blinding_factor: 0,
        ..Config::default()
    }
}

// =========================================================
// TEST 1:
// Binary SpMV Weights
//
// Virtual packing soundness requires tower_bit
// to commute with SpMV. This only holds for binary
// (0/1) weights. Probe the matrix with unit vectors
// to verify every weight is exactly 0 or 1 in GF(2^128).
// =========================================================

#[test]
fn brakedown_binary_weights_verified() {
    let zero = Flat::from_raw(F::ZERO);
    let one = Flat::from_raw(F::ONE);

    for &seed_byte in &[0u8, 42, 0xFF, 0xDE] {
        let seed = [seed_byte; 32];

        for &(dim, degree) in &[(20, 16), (36, 16), (64, 8), (128, 16)] {
            let matrix = ByteSparseMatrix::generate_random(dim, dim, degree, seed);

            for c in 0..dim {
                let mut unit_vec = vec![zero; dim];
                unit_vec[c] = one;

                let result = matrix.spmv(unit_vec.as_slice());

                for (r, &val) in result.iter().enumerate() {
                    assert!(
                        val == zero || val == one,
                        "non-binary weight at ({r},{c}) seed=0x{seed_byte:02X} dim={dim}",
                    );
                }
            }
        }
    }
}

// =========================================================
// TEST 2:
// Virtual Packing Eval Forgery
//
// Corrupt a virtual bit column evaluation in
// the chiplet's point_evaluations. The TensorPCS
// proximity check must catch the mismatch between
// claimed virtual values and the physically
// committed trace data.
// =========================================================

#[test]
fn virtual_packing_eval_forgery_rejected() {
    let (air, instance, witness) = make_test_system();
    let config = zk_config();
    let seed = [0xAAu8; 32];

    let mut proof = prove(
        b"VirtualPackEvalForgery",
        &air,
        &instance,
        &witness,
        &config,
        seed,
        None,
    )
    .expect("honest proof must succeed");

    let mut vt = Transcript::<H>::new(b"VirtualPackEvalForgery");
    let ok = HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut vt, &config)
        .expect("verification must not error");

    assert!(ok, "baseline must verify");

    // Corrupt virtual bit column 17
    let evals = &mut proof.chiplet_eval_proofs[0].point_evaluations[0].1;
    assert!(evals.len() > 17);
    evals[17] += F::ONE;

    let mut at = Transcript::<H>::new(b"VirtualPackEvalForgery");
    let attack = HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut at, &config);

    assert!(
        attack.is_err() || !attack.unwrap(),
        "forged virtual bit column 17 accepted",
    );
}

// =========================================================
// TEST 3:
// Virtual Expansion Witness Isolation
//
// Stage A:
// With ZK enabled, the magic witness bytes must
// NOT appear in chiplet LDT openings. The noise
// generation path through parse_virtual_row must
// mask the data.
//
// Stage B:
// Without ZK, the magic bytes MUST appear
// (control group validating the scan is reliable).
// =========================================================

#[test]
fn virtual_expansion_witness_isolation() {
    let (air, instance, witness) = make_test_system();
    let seed = [0xBBu8; 32];
    let needle = Block32(MAGIC).to_hardware().into_raw().0.to_le_bytes();

    let scan_chiplet_openings = |proof: &hekate_core::proofs::InnerProof<F>| -> bool {
        for c_idx in 0..proof.chiplet_eval_proofs.len() {
            let ldt = &proof.chiplet_eval_proofs[c_idx].ldt_proof;
            for col in &ldt.opened_columns {
                if col.windows(needle.len()).any(|w| w == needle) {
                    return true;
                }
            }
        }

        false
    };

    // STAGE A:
    // ZK enabled
    let config_zk = zk_config();

    let proof_zk = prove(
        b"VirtualPackWitnessIsolation",
        &air,
        &instance,
        &witness,
        &config_zk,
        seed,
        None,
    )
    .expect("ZK proof must succeed");

    let mut vt_zk = Transcript::<H>::new(b"VirtualPackWitnessIsolation");
    assert!(
        HekateVerifier::<F, H>::verify(&air, &instance, &proof_zk, &mut vt_zk, &config_zk).unwrap(),
        "ZK proof must verify"
    );

    assert!(
        !scan_chiplet_openings(&proof_zk),
        "ZK enabled but witness bytes leaked in chiplet LDT openings",
    );

    // STAGE B:
    // ZK disabled (control)
    let config_no_zk = no_zk_config();

    let proof_no = prove(
        b"VirtualPackWitnessIsolation",
        &air,
        &instance,
        &witness,
        &config_no_zk,
        seed,
        None,
    )
    .expect("non-ZK proof must succeed");

    assert!(
        scan_chiplet_openings(&proof_no),
        "ZK disabled but witness bytes absent, control group broken",
    );
}
