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

//! Adversarial security tests for the
//! independent chiplet pipeline.
//!
//! Exercises forgery vectors specific
//! to chiplet isolation, transcript binding,
//! and evaluation argument integrity.

use hekate::core::config::Config;
use hekate::core::trace::{ColumnTrace, ColumnType, TraceColumn};
use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::{Block128, TowerField};
use hekate_core::trace::IntoTraceColumn;
use hekate_gadgets::{CpuFetchColumns, CpuFetchUnit, Instruction, RomChiplet, generate_rom_trace};
use hekate_math::{Bit, Block32};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::permutation::PermutationCheckSpec;
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;

type F = Block128;
type H = DefaultHasher;

// ==========================================================
// Minimal independent-chiplet AIR for security tests.
// CPU has 6 columns (CpuFetchColumns).
// ROM is an independent chiplet via chiplet_defs().
// ==========================================================

#[derive(Clone)]
struct ChipletTestAir {
    rom_num_rows: usize,
}

impl Air<F> for ChipletTestAir {
    fn num_columns(&self) -> usize {
        CpuFetchColumns::NUM_COLUMNS
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(CpuFetchColumns::build_layout)
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![(RomChiplet::BUS_ID.into(), CpuFetchUnit::linking_spec())]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(CpuFetchColumns::SELECTOR));

        cs.build()
    }
}

impl Program<F> for ChipletTestAir {
    fn chiplet_defs(&self) -> hekate_core::errors::Result<Vec<ChipletDef<F>>> {
        let rom = RomChiplet::new(self.rom_num_rows);
        Ok(vec![ChipletDef::from_air(&rom)?])
    }
}

// ==========================================================
// Helpers
// ==========================================================

fn build_test_system(
    num_vars: usize,
) -> (
    ChipletTestAir,
    ProgramInstance<F>,
    ProgramWitness<F, ColumnTrace>,
    Config,
) {
    let num_rows = 1 << num_vars;

    let instructions: Vec<Instruction> = (0..num_rows)
        .map(|i| Instruction::new(i as u32, 1, [0, 0, 0]))
        .collect();

    // CPU trace
    let mut cpu_trace = ColumnTrace::new(num_vars).unwrap();
    let mut pc_cols: Vec<Vec<Block32>> = (0..4).map(|_| Vec::with_capacity(num_rows)).collect();
    let mut op_col = Vec::with_capacity(num_rows);
    let mut arg_cols: Vec<Vec<Block32>> = (0..3).map(|_| Vec::with_capacity(num_rows)).collect();
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

    for col in pc_cols {
        cpu_trace.add_column(col.into_trace_column()).unwrap();
    }

    cpu_trace.add_column(op_col.into_trace_column()).unwrap();

    for col in arg_cols {
        cpu_trace.add_column(col.into_trace_column()).unwrap();
    }

    cpu_trace.add_column(TraceColumn::Bit(sel_col)).unwrap();

    // ROM chiplet trace
    let rom_trace = generate_rom_trace(&instructions, num_rows).unwrap();

    let air = ChipletTestAir {
        rom_num_rows: num_rows,
    };
    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![rom_trace]);

    let config = Config {
        num_queries: 4,
        min_security_bits: 0,
        sumcheck_blinding_factor: 2,
        ldt_support_size: 4,
        ..Config::default()
    };

    (air, instance, witness, config)
}

fn prove_and_verify(
    air: &ChipletTestAir,
    instance: &ProgramInstance<F>,
    witness: &ProgramWitness<F, ColumnTrace>,
    config: &Config,
) -> (hekate_core::proofs::InnerProof<F>, bool) {
    let seed = [0xBBu8; 32];
    let proof = prove(
        b"ChipletSecurity",
        air,
        instance,
        witness,
        config,
        seed,
        None,
    )
    .expect("proving failed");

    let mut vt = Transcript::<H>::new(b"ChipletSecurity");
    let ok =
        HekateVerifier::<F, H>::verify(air, instance, &proof, &mut vt, config).unwrap_or(false);

    (proof, ok)
}

// ==========================================================
// EXPLOIT: Extra chiplet commitments in proof
//
// A malicious prover adds extra chiplet_commitments
// that don't correspond to any chiplet_defs().
// The verifier must reject this immediately.
// ==========================================================

#[test]
fn extra_chiplet_commitments_rejected() {
    let (air, instance, witness, config) = build_test_system(6);
    let (mut proof, ok) = prove_and_verify(&air, &instance, &witness, &config);
    assert!(ok, "Baseline proof must verify");

    // ATTACK:
    // Duplicate the first chiplet commitment
    let extra_comm = proof.chiplet_commitments[0].clone();
    proof.chiplet_commitments.push(extra_comm);

    let mut vt = Transcript::<H>::new(b"ChipletSecurity");
    let result = HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut vt, &config);

    assert!(
        result.is_err() || !result.unwrap(),
        "SECURITY FAILURE: Extra chiplet commitment accepted"
    );
}

// ==========================================================
// EXPLOIT: Missing chiplet commitments in proof
//
// A malicious prover strips chiplet_commitments
// to bypass chiplet ZeroCheck verification.
// ==========================================================

#[test]
fn missing_chiplet_commitments_rejected() {
    let (air, instance, witness, config) = build_test_system(6);
    let (mut proof, ok) = prove_and_verify(&air, &instance, &witness, &config);
    assert!(ok, "Baseline proof must verify");

    // ATTACK:
    // Remove all chiplet commitments
    proof.chiplet_commitments.clear();

    let mut vt = Transcript::<H>::new(b"ChipletSecurity");
    let result = HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut vt, &config);

    assert!(
        result.is_err(),
        "SECURITY FAILURE: Missing chiplet commitments accepted"
    );
}

// ==========================================================
// EXPLOIT: Corrupted chiplet evaluation values
//
// A malicious prover modifies the claimed evaluation
// values for a chiplet. The TensorPCS proximity check
// must detect the mismatch.
// ==========================================================

#[test]
fn chiplet_eval_values_forgery() {
    let (air, instance, witness, config) = build_test_system(6);
    let (mut proof, ok) = prove_and_verify(&air, &instance, &witness, &config);
    assert!(ok, "Baseline proof must verify");

    // ATTACK:
    // Corrupt first chiplet's
    // claimed evaluation at r_final.
    let c_eval = &mut proof.chiplet_eval_proofs[0];
    assert!(!c_eval.point_evaluations.is_empty());
    assert!(!c_eval.point_evaluations[0].1.is_empty());

    c_eval.point_evaluations[0].1[0] += F::ONE;

    let mut vt = Transcript::<H>::new(b"ChipletSecurity");
    let result = HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut vt, &config);

    assert!(
        result.is_err() || !result.unwrap(),
        "SECURITY FAILURE: Chiplet eval value forgery accepted"
    );
}

// ==========================================================
// EXPLOIT: Swap chiplet commitment root
//
// A malicious prover replaces the chiplet's Merkle
// root with an all-zero root. This desynchronizes the
// Fiat-Shamir transcript (because the root is absorbed
// into the transcript before challenges are drawn).
// ==========================================================

#[test]
fn chiplet_root_swap_rejected() {
    let (air, instance, witness, config) = build_test_system(6);
    let (mut proof, ok) = prove_and_verify(&air, &instance, &witness, &config);
    assert!(ok, "Baseline proof must verify");

    // ATTACK:
    // Replace chiplet root with zeros
    proof.chiplet_commitments[0].root = [0u8; 32];

    let mut vt = Transcript::<H>::new(b"ChipletSecurity");
    let result = HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut vt, &config);

    assert!(
        result.is_err() || !result.unwrap(),
        "SECURITY FAILURE: Forged chiplet Merkle root accepted"
    );
}

// ==========================================================
// EXPLOIT: Truncate chiplet claimed values
//
// A malicious prover strips entries from the
// chiplet's combined evaluation vector. The
// verifier must reject with an error, not panic.
// ==========================================================

#[test]
fn chiplet_eval_values_truncated() {
    let (air, instance, witness, config) = build_test_system(6);
    let (mut proof, ok) = prove_and_verify(&air, &instance, &witness, &config);
    assert!(ok, "Baseline proof must verify");

    // ATTACK:
    // Truncate chiplet's combined trace values
    // to 1 entry; the verifier's length check
    // rejects without panicking.
    proof.chiplet_eval_proofs[0].point_evaluations[0]
        .1
        .truncate(1);

    let mut vt = Transcript::<H>::new(b"ChipletSecurity");
    let result = HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut vt, &config);

    assert!(
        result.is_err(),
        "SECURITY FAILURE: Truncated chiplet eval values accepted (should reject)"
    );
}

// ==========================================================
// EXPLOIT: Forge chiplet LogUp claimed_sum
//
// A malicious prover modifies the chiplet's LogUp
// claimed_sum so the paired (main, chiplet) endpoints
// no longer cancel. `check_bus_sum_matching` must reject.
// ==========================================================

#[test]
fn chiplet_logup_sum_mismatch() {
    let (air, instance, witness, config) = build_test_system(6);
    let (mut proof, ok) = prove_and_verify(&air, &instance, &witness, &config);
    assert!(ok, "Baseline proof must verify");

    assert_eq!(proof.chiplet_logup_aux.len(), 1);
    assert!(!proof.chiplet_logup_aux[0].claimed_sums.is_empty());

    // ATTACK:
    // Corrupt the chiplet's claimed_sum
    // so the bus no longer cancels.
    proof.chiplet_logup_aux[0].claimed_sums[0].1 += F::ONE;

    let mut vt = Transcript::<H>::new(b"ChipletSecurity");
    let result = HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut vt, &config);

    assert!(
        result.is_err() || !result.unwrap(),
        "SECURITY FAILURE: Chiplet LogUp claimed_sum forgery accepted"
    );
}

// ==========================================================
// EXPLOIT: Unmatched main-trace bus_id
//
// A program declares a main-trace bus endpoint
// ("phantom_bus") that no chiplet or gadget supplies.
// The verifier's exhaustiveness check must reject
// this because the bus has no counterpart.
// ==========================================================

/// AIR with an extra phantom bus_id that
/// no chiplet provides. Used only for
/// verification (not proving).
#[derive(Clone)]
struct PhantomBusAir {
    rom_num_rows: usize,
}

impl Air<F> for PhantomBusAir {
    fn num_columns(&self) -> usize {
        CpuFetchColumns::NUM_COLUMNS
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(CpuFetchColumns::build_layout)
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![
            (RomChiplet::BUS_ID.into(), CpuFetchUnit::linking_spec()),
            // Phantom bus:
            // no chiplet supplies this.
            ("phantom_bus".into(), CpuFetchUnit::linking_spec()),
        ]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(CpuFetchColumns::SELECTOR));
        cs.build()
    }
}

impl Program<F> for PhantomBusAir {
    fn chiplet_defs(&self) -> hekate_core::errors::Result<Vec<ChipletDef<F>>> {
        let rom = RomChiplet::new(self.rom_num_rows);
        Ok(vec![ChipletDef::from_air(&rom)?])
    }
}

#[test]
fn unmatched_main_bus_rejected() {
    let num_vars = 6;
    let num_rows = 1 << num_vars;

    // Prove with the normal AIR (1 main bus + 1 chiplet).
    let (air, instance, witness, config) = build_test_system(num_vars);
    let seed = [0xBBu8; 32];

    let proof = prove(
        b"ChipletSecurity",
        &air,
        &instance,
        &witness,
        &config,
        seed,
        None,
    )
    .expect("proving failed");

    // Verify with the phantom AIR that declares
    // an extra bus_id no chiplet supplies.
    // The verifier must reject: "phantom_bus"
    // has no chiplet/gadget counterpart.
    let phantom_air = PhantomBusAir {
        rom_num_rows: num_rows,
    };

    let mut vt = Transcript::<H>::new(b"ChipletSecurity");
    let result = HekateVerifier::<F, H>::verify(&phantom_air, &instance, &proof, &mut vt, &config);

    assert!(
        result.is_err(),
        "SECURITY FAILURE: Unmatched main-trace bus_id accepted"
    );
}

// =====================================================
// Chiplet sumcheck round-poly degree is strict
// =====================================================

#[test]
fn chiplet_sumcheck_degree_inflation_rejected() {
    let (air, instance, witness, config) = build_test_system(6);
    let (mut proof, ok) = prove_and_verify(&air, &instance, &witness, &config);

    assert!(ok, "Baseline proof must verify");

    let pad = proof.chiplet_zerocheck_proofs[0].round_polys[0].evals[0];
    proof.chiplet_zerocheck_proofs[0].round_polys[0]
        .evals
        .push(pad);

    let mut vt = Transcript::<H>::new(b"ChipletSecurity");
    let result = HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut vt, &config);

    assert!(
        result.is_err() || !result.unwrap(),
        "SECURITY FAILURE: degree-inflated chiplet round poly accepted"
    );
}

// =====================================================
// Chiplet eval_proof must carry exactly one point
// =====================================================

#[test]
fn chiplet_eval_proof_multiple_points_rejected() {
    let (air, instance, witness, config) = build_test_system(6);
    let (mut proof, ok) = prove_and_verify(&air, &instance, &witness, &config);

    assert!(ok, "Baseline proof must verify");

    let dup = proof.chiplet_eval_proofs[0].point_evaluations[0].clone();
    proof.chiplet_eval_proofs[0].point_evaluations.push(dup);

    let mut vt = Transcript::<H>::new(b"ChipletSecurity");
    let result = HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut vt, &config);

    assert!(
        result.is_err(),
        "SECURITY FAILURE: multi-point chiplet eval_proof accepted"
    );
}

// =====================================================
// chiplet_commitments[i].num_rows
// must be non-zero power of two.
// =====================================================

#[test]
fn chiplet_commitment_num_rows_non_power_of_two_rejected() {
    let (air, instance, witness, config) = build_test_system(6);
    let (mut proof, ok) = prove_and_verify(&air, &instance, &witness, &config);

    assert!(ok, "Baseline proof must verify");

    proof.chiplet_commitments[0].num_rows = 5;

    let mut vt = Transcript::<H>::new(b"ChipletSecurity");
    let result = HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut vt, &config);

    assert!(
        result.is_err() || !result.unwrap(),
        "SECURITY FAILURE: chiplet_commitments[0].num_rows = 5 accepted"
    );
}

#[test]
fn chiplet_commitment_num_rows_zero_rejected() {
    let (air, instance, witness, config) = build_test_system(6);
    let (mut proof, ok) = prove_and_verify(&air, &instance, &witness, &config);

    assert!(ok, "Baseline proof must verify");

    proof.chiplet_commitments[0].num_rows = 0;

    let mut vt = Transcript::<H>::new(b"ChipletSecurity");
    let result = HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut vt, &config);

    assert!(
        result.is_err() || !result.unwrap(),
        "SECURITY FAILURE: chiplet_commitments[0].num_rows = 0 accepted"
    );
}
