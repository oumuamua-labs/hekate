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

use hekate::core::config::Config;
use hekate::core::trace::{ColumnTrace, ColumnType, TraceColumn};
use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::{Block128, HardwareField, TowerField};
use hekate_core::trace::{IntoTraceColumn, Trace, TraceBuilder};
use hekate_core::utils::compute_split_vars;
use hekate_math::{Bit, Block32};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::constraint::{BoundaryConstraint, ConstraintAst};
use hekate_program::permutation::{PermutationCheckSpec, REQUEST_IDX_LABEL, Source};
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;
use rand::{TryRngCore, rngs::OsRng};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

type F = Block128;
type H = DefaultHasher;

#[derive(Clone)]
struct FibAir {
    num_cols: usize,
    num_rows: usize,
}

impl Air<F> for FibAir {
    fn num_columns(&self) -> usize {
        self.num_cols
    }

    fn boundary_constraints(&self) -> Vec<BoundaryConstraint<F>> {
        vec![BoundaryConstraint::with_public_input(
            1,
            self.num_rows - 1,
            0,
        )]
    }

    fn column_layout(&self) -> &'static [ColumnType] {
        &[ColumnType::B32, ColumnType::B32, ColumnType::Bit]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();

        let [a, b, q] = [cs.col(0), cs.col(1), cs.col(2)];
        let [na, nb] = [cs.next(0), cs.next(1)];

        cs.constrain(q * (na + b));
        cs.constrain(q * (nb + a + b));

        cs.build()
    }
}

impl Program<F> for FibAir {
    fn num_public_inputs(&self) -> usize {
        1
    }
}

fn generate_fib_trace(num_vars: usize) -> ColumnTrace {
    let num_rows = 1 << num_vars;

    let mut a_col: Vec<Block32> = Vec::with_capacity(num_rows);
    let mut b_col: Vec<Block32> = Vec::with_capacity(num_rows);
    let mut sel_col: Vec<Bit> = Vec::with_capacity(num_rows);

    let mut a = Block32::ZERO;
    let mut b = Block32::ONE;

    for i in 0..num_rows {
        a_col.push(a);
        b_col.push(b);

        if i == num_rows - 1 {
            sel_col.push(Bit::ZERO);
        } else {
            sel_col.push(Bit::ONE);
        }

        let tmp = a + b;
        a = b;
        b = tmp;
    }

    let mut trace = ColumnTrace::new(num_vars).unwrap();
    trace.add_column(a_col.into_trace_column()).unwrap();
    trace.add_column(b_col.into_trace_column()).unwrap();
    trace.add_column(TraceColumn::Bit(sel_col)).unwrap();

    trace
}

#[test]
fn noise_entropy_inspection() {
    // OBJECTIVE:
    // Ensure that blinding appends non-zero,
    // high-entropy bytes to the opened data rows.

    let num_vars = 12;
    let num_rows = 1usize << num_vars;

    let trace = generate_fib_trace(num_vars);
    let expected_pub = trace.get_element(1, num_rows - 1).unwrap().to_tower();
    let instance = ProgramInstance::new(num_rows, vec![expected_pub]);
    let witness = ProgramWitness::new(trace);

    let air = FibAir {
        num_cols: 3,
        num_rows,
    };

    let config = Config {
        sumcheck_blinding_factor: 2,
        ldt_blinding_factor: 2,
        num_queries: 8,
        min_security_bits: 0,
        ..Config::default()
    };

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    let proof = prove(
        b"ZK_Entropy",
        &air,
        &instance,
        &witness,
        &config,
        blinding_seed,
        None,
    )
    .unwrap();

    // In Tensor PCS, we open full columns
    let columns = &proof.eval_proof.ldt_proof.opened_columns;
    assert!(!columns.is_empty(), "Must have opened columns");

    // Architecture:
    // FibAir uses [B32, B32, Bit]
    // Physical size = 4 + 4 + 1 = 9 bytes
    // per base row. With interleaved shifted
    // columns: 9 * 2 = 18 bytes.
    let data_bytes_per_row = (4 + 4 + 1) * 2;

    // ZK Sumcheck noise is always Block128 (16 bytes).
    let noise_bytes_per_row = config.sumcheck_blinding_factor * 16 * 2;
    let bytes_per_row = data_bytes_per_row + noise_bytes_per_row;

    // Calculate grid_rows based on the
    // asymmetric grid formula used in Prover.
    let grid_rows = {
        let split_vars = compute_split_vars(
            num_vars,
            config.num_queries,
            config.expansion_degree,
            bytes_per_row,
        );

        1 << (num_vars - split_vars)
    };

    for col_data in columns {
        assert_eq!(
            col_data.len(),
            grid_rows * bytes_per_row,
            "Opened column must include virtual blinding bytes for all grid rows"
        );

        // DYNAMIC ENTROPY ANALYSIS:
        // Extract noise from ALL grid
        // rows, not just the first one.
        let mut all_noise_bytes = Vec::with_capacity(grid_rows * noise_bytes_per_row);
        for row_idx in 0..grid_rows {
            let start = row_idx * bytes_per_row + data_bytes_per_row;
            let end = start + noise_bytes_per_row;

            all_noise_bytes.extend_from_slice(&col_data[start..end]);
        }

        // Calculate Shannon Entropy:
        // H = - sum(p_i * log2(p_i))
        let mut counts = [0usize; 256];
        for &b in &all_noise_bytes {
            counts[b as usize] += 1;
        }

        let mut entropy = 0.0f64;
        let total = all_noise_bytes.len() as f64;

        for &count in &counts {
            if count > 0 {
                let p = count as f64 / total;
                entropy -= p * p.log2();
            }
        }

        // BIRTHDAY PARADOX MATH:
        // The prover aggressively folds the matrix,
        // which may result in small sample sizes (e.g., 256 bytes).
        // Due to natural collisions, 256 random bytes
        // will mathematically never reach 8.0 bits.
        // Expected unique values = 256 * (1 - (255/256)^N)
        let expected_unique = 256.0 * (1.0 - (255.0 / 256.0_f64).powf(total));
        let theoretical_expected_entropy = expected_unique.log2();

        // Demand that the AES PRNG hits at least 95%
        // of the mathematically expected entropy for
        // this exact sample size.
        let expected_threshold = theoretical_expected_entropy * 0.95;

        assert!(
            entropy > expected_threshold,
            "Noise entropy is fatally low ({} bits vs {} expected on {} bytes). The PRNG is compromised.",
            entropy,
            theoretical_expected_entropy,
            total
        );
    }
}

#[test]
fn seed_nondeterminism() {
    // OBJECTIVE: changing the blinding seed must change:
    // - the trace commitment root
    // - the sumcheck transcript (round polys)
    // - the opened row bytes (noise)

    let num_vars = 6;
    let num_rows = 1usize << num_vars;

    let trace = generate_fib_trace(num_vars);
    let expected_pub = trace.get_element(1, num_rows - 1).unwrap().to_tower();
    let air = FibAir {
        num_cols: 3,
        num_rows,
    };
    let instance = ProgramInstance::new(num_rows, vec![expected_pub]);

    let config_a = Config {
        sumcheck_blinding_factor: 1,
        num_queries: 8,
        min_security_bits: 0,
        ..Config::default()
    };

    let mut seed1 = [0u8; 32];
    seed1[0] = 1;

    let config_b = Config {
        sumcheck_blinding_factor: 1,
        num_queries: 8,
        min_security_bits: 0,
        ..Config::default()
    };

    let mut seed2 = [0u8; 32];
    seed2[0] = 2;

    let witness_a = ProgramWitness::new(trace.clone());
    let witness_b = ProgramWitness::new(trace);

    let proof_a = prove(
        b"ZK_Seed", &air, &instance, &witness_a, &config_a, seed1, None,
    )
    .unwrap();

    let proof_b = prove(
        b"ZK_Seed", &air, &instance, &witness_b, &config_b, seed2, None,
    )
    .unwrap();

    assert_ne!(
        proof_a.trace_commitment.root, proof_b.trace_commitment.root,
        "Merkle root must depend on blinding seed"
    );

    let poly_a_r0 = &proof_a.zerocheck_proof.round_polys[0].evals;
    let poly_b_r0 = &proof_b.zerocheck_proof.round_polys[0].evals;

    assert_ne!(
        poly_a_r0, poly_b_r0,
        "Sumcheck rounds must differ when blinding seed changes"
    );

    let col_a = &proof_a.eval_proof.ldt_proof.opened_columns[0];
    let col_b = &proof_b.eval_proof.ldt_proof.opened_columns[0];

    assert_ne!(col_a, col_b, "Opened noise bytes must differ");

    let mut vt_a = Transcript::<H>::new(b"ZK_Seed");
    let ok_a =
        HekateVerifier::<F, H>::verify(&air, &instance, &proof_a, &mut vt_a, &config_a).unwrap();
    assert!(ok_a, "Proof A must verify");

    let mut vt_b = Transcript::<H>::new(b"ZK_Seed");
    let ok_b =
        HekateVerifier::<F, H>::verify(&air, &instance, &proof_b, &mut vt_b, &config_b).unwrap();
    assert!(ok_b, "Proof B must verify");
}

#[test]
fn noise_integrity_check() {
    // OBJECTIVE:
    // verifier must reject if the
    // blinding bytes are modified.

    let num_vars = 5;
    let num_rows = 1usize << num_vars;

    let trace = generate_fib_trace(num_vars);
    let expected_pub = trace.get_element(1, num_rows - 1).unwrap().to_tower();
    let instance = ProgramInstance::new(num_rows, vec![expected_pub]);
    let witness = ProgramWitness::new(trace);

    let air = FibAir {
        num_cols: 3,
        num_rows,
    };

    let config = Config {
        sumcheck_blinding_factor: 1,
        ldt_blinding_factor: 1,
        num_queries: 8,
        min_security_bits: 0,
        ..Config::default()
    };

    let mut blinding_seed = [42u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    let mut proof = prove(
        b"ZK_Integrity",
        &air,
        &instance,
        &witness,
        &config,
        blinding_seed,
        None,
    )
    .unwrap();

    let col_data = &mut proof.eval_proof.ldt_proof.opened_columns[0];

    // Architecture:
    // FibAir uses [B32, B32, Bit] = 9 bytes.
    let data_bytes_per_row = (4 + 4 + 1) * 2;
    let bytes_per_row = data_bytes_per_row + (config.sumcheck_blinding_factor * 16 * 2);

    // ATTACK:
    // maliciously corrupt the noise suffix
    // in the first grid row of the column.
    for b in &mut col_data[data_bytes_per_row..bytes_per_row] {
        *b = 0;
    }

    let mut verifier_transcript = Transcript::<H>::new(b"ZK_Integrity");
    let result =
        HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut verifier_transcript, &config);

    match result {
        Ok(true) => panic!("SECURITY FAILURE: Verifier accepted modified noise!"),
        Ok(false) => { /* Logic rejection (Good) */ }
        Err(_) => { /* Protocol rejection / Merkle mismatch (Good) */ }
    }
}

/// Malicious AIR that bypasses degree checks
/// but zeroes out the sum for the Prover
#[derive(Clone)]
struct MaliciousAir {
    num_cols: usize,
    num_rows: usize,
    is_prover: Arc<AtomicBool>,
}

impl Air<F> for MaliciousAir {
    fn num_columns(&self) -> usize {
        self.num_cols
    }

    fn boundary_constraints(&self) -> Vec<BoundaryConstraint<F>> {
        if self.is_prover.load(Ordering::SeqCst) {
            vec![]
        } else {
            vec![BoundaryConstraint::with_public_input(
                1,
                self.num_rows - 1,
                0,
            )]
        }
    }

    fn column_layout(&self) -> &'static [ColumnType] {
        &[ColumnType::B32, ColumnType::B32, ColumnType::Bit]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let coeff = if self.is_prover.load(Ordering::SeqCst) {
            F::ZERO
        } else {
            F::ONE
        };

        let cs = ConstraintSystem::<F>::new();

        let [b, q] = [cs.col(1), cs.col(2)];
        let na = cs.next(0);

        cs.constrain(cs.scale(coeff, q * (na + b)));

        cs.build()
    }
}

impl Program<F> for MaliciousAir {
    fn num_public_inputs(&self) -> usize {
        1
    }
}

#[test]
fn trust_me_bro_knowledge() {
    let num_vars = 4;
    let num_rows = 1 << num_vars;

    // 1. Create a FAKE trace (all zeros).
    // This physically violates the boundary
    // constraint (expected 999999).
    let mut trace = ColumnTrace::new(num_vars).unwrap();
    trace
        .add_column(vec![Block32::ZERO; num_rows].into_trace_column())
        .unwrap();
    trace
        .add_column(vec![Block32::ZERO; num_rows].into_trace_column())
        .unwrap();
    trace
        .add_column(TraceColumn::Bit(vec![Bit::ZERO; num_rows]))
        .unwrap();

    let instance = ProgramInstance::new(num_rows, vec![F::from(999999u128)]);
    let witness = ProgramWitness::new(trace);

    let air = MaliciousAir {
        num_cols: 3,
        num_rows,
        is_prover: Arc::new(AtomicBool::new(true)),
    };

    let mut blinding_seed = [42u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    // ==========================================
    // ATTACK WITH ZK ENABLED (blinding_factor = 2)
    // ==========================================
    let config_zk = Config {
        num_queries: 8,
        min_security_bits: 0,
        sumcheck_blinding_factor: 2, // ZK ACTIVATED
        ldt_blinding_factor: 2,
        ..Config::default()
    };

    let proof_zk = prove(
        b"Exploit",
        &air,
        &instance,
        &witness,
        &config_zk,
        blinding_seed,
        None,
    )
    .unwrap();

    // Switch to Verifier mode
    air.is_prover.store(false, Ordering::SeqCst);

    let mut verifier_transcript = Transcript::<H>::new(b"Exploit");
    let result_zk = HekateVerifier::<F, H>::verify(
        &air,
        &instance,
        &proof_zk,
        &mut verifier_transcript,
        &config_zk,
    );

    // TThe verifier must reject the proof
    let is_valid_zk = result_zk.unwrap_or(false);
    assert!(
        !is_valid_zk,
        "SECURITY FAILURE: Verifier blindly accepted the masking_bias"
    );

    // ==========================================
    // PROVE IT FAILS WITHOUT ZK
    // ==========================================
    air.is_prover.store(true, Ordering::SeqCst);

    let config_no_zk = Config {
        num_queries: 8,
        min_security_bits: 0,
        sumcheck_blinding_factor: 0, // ZK DISABLED
        ldt_blinding_factor: 1,
        ..Config::default()
    };

    let proof_no_zk = prove(
        b"Exploit2",
        &air,
        &instance,
        &witness,
        &config_no_zk,
        blinding_seed,
        None,
    )
    .unwrap();

    air.is_prover.store(false, Ordering::SeqCst);

    let mut verifier_transcript2 = Transcript::<H>::new(b"Exploit2");
    let result_no_zk = HekateVerifier::<F, H>::verify(
        &air,
        &instance,
        &proof_no_zk,
        &mut verifier_transcript2,
        &config_no_zk,
    );

    let is_valid_no_zk = result_no_zk.unwrap_or(false);
    assert!(
        !is_valid_no_zk,
        "Without ZK, the Verifier should correctly reject the forgery"
    );
}

#[test]
fn algebraic_and_evaluation_perfect_hiding() {
    // OBJECTIVE:
    // Prove that ZK mathematically preserves
    // the target claim while perfectly hiding
    // both the Sumcheck polynomials and
    // the Trace Evaluations.

    let num_vars = 6;
    let num_rows = 1 << num_vars;
    let trace = generate_fib_trace(num_vars);
    let expected_pub = trace.get_element(1, num_rows - 1).unwrap().to_tower();
    let instance = ProgramInstance::new(num_rows, vec![expected_pub]);
    let witness = ProgramWitness::new(trace.clone());
    let air = FibAir {
        num_cols: 3,
        num_rows,
    };

    let config_no_zk = Config {
        sumcheck_blinding_factor: 0,
        ldt_blinding_factor: 4,
        num_queries: 4,
        min_security_bits: 0,
        ..Config::default()
    };

    let config_zk = Config {
        sumcheck_blinding_factor: 2,
        ldt_blinding_factor: 4,
        num_queries: 4,
        min_security_bits: 0,
        ..Config::default()
    };

    // 1. Proof WITHOUT ZK
    let p_no_zk = prove(
        b"ZK_Hiding",
        &air,
        &instance,
        &witness,
        &config_no_zk,
        [0u8; 32],
        None,
    )
    .unwrap();

    // 2. Proof WITH ZK (Seed A)
    // Use asymmetric seeds to avoid AES XOR
    // cancellation in JitNoiseGenerator!
    let mut seed_a = [0u8; 32];
    seed_a[0] = 1;

    let p_zk_a = prove(
        b"ZK_Hiding",
        &air,
        &instance,
        &witness,
        &config_zk,
        seed_a,
        None,
    )
    .unwrap();

    // 3. Proof WITH ZK (Seed B)
    let mut seed_b = [0u8; 32];
    seed_b[0] = 2;

    let p_zk_b = prove(
        b"ZK_Hiding",
        &air,
        &instance,
        &witness,
        &config_zk,
        seed_b,
        None,
    )
    .unwrap();

    // GUARANTEE 1:
    // Sumcheck Target Invariance
    // The algebraic telescopic sum must perfectly cancel
    // out over the hypercube. This means the initial sum
    // of the hypercube (g_0(0) + g_0(1)) must always
    // be exactly ZERO. We cannot compare the final
    // claimed_evaluation because Fiat-Shamir challenges
    // diverge. Since Degree Stripping removes P(0),
    // we verify this invariant by asserting that the
    // Verifier successfully reconstructs P(0) from 0
    // and validates the proof.
    let mut vt1 = Transcript::<H>::new(b"ZK_Hiding");
    assert!(
        HekateVerifier::<F, H>::verify(&air, &instance, &p_no_zk, &mut vt1, &config_no_zk).unwrap(),
        "No-ZK proof failed to verify"
    );

    let mut vt2 = Transcript::<H>::new(b"ZK_Hiding");
    assert!(
        HekateVerifier::<F, H>::verify(&air, &instance, &p_zk_a, &mut vt2, &config_zk).unwrap(),
        "ZK proof A failed to verify"
    );

    let mut vt3 = Transcript::<H>::new(b"ZK_Hiding");
    assert!(
        HekateVerifier::<F, H>::verify(&air, &instance, &p_zk_b, &mut vt3, &config_zk).unwrap(),
        "ZK proof B failed to verify"
    );

    // GUARANTEE 2:
    // Polynomial Obfuscation
    // The round polynomials sent to the
    // verifier must be completely different.
    let poly_no_zk = &p_no_zk.zerocheck_proof.round_polys[0].evals;
    let poly_zk_a = &p_zk_a.zerocheck_proof.round_polys[0].evals;
    let poly_zk_b = &p_zk_b.zerocheck_proof.round_polys[0].evals;

    assert_ne!(
        poly_no_zk, poly_zk_a,
        "ZK failed to hide the Sumcheck polynomials"
    );
    assert_ne!(
        poly_zk_a, poly_zk_b,
        "ZK failed to hide the Sumcheck polynomials"
    );

    // GUARANTEE 3:
    // Trace Evaluation Perfect Hiding
    // The evaluations `trace_values` at r_final must
    // be perfectly masked by AES noise. They must
    // leak ZERO information about the underlying data.
    let eval_no_zk = &p_no_zk.eval_proof.point_evaluations[0].1;
    let eval_zk_a = &p_zk_a.eval_proof.point_evaluations[0].1;
    let eval_zk_b = &p_zk_b.eval_proof.point_evaluations[0].1;

    assert_ne!(eval_no_zk, eval_zk_a, "ZK failed to hide trace evaluations");
    assert_ne!(eval_zk_a, eval_zk_b, "ZK failed to hide trace evaluations");

    // K = 0, ldt_blinding = 200 ablation pair.
    // Isolates whether ldt_blinding alone
    // produces seed-dependent round polys and
    // evaluations without any sumcheck masks.
    let config_only_ldt = Config {
        sumcheck_blinding_factor: 0,
        ldt_blinding_factor: 200,
        num_queries: 4,
        min_security_bits: 0,
        ..Config::default()
    };

    let p_only_ldt_a = prove(
        b"ZK_Hiding",
        &air,
        &instance,
        &witness,
        &config_only_ldt,
        seed_a,
        None,
    )
    .unwrap();

    let p_only_ldt_b = prove(
        b"ZK_Hiding",
        &air,
        &instance,
        &witness,
        &config_only_ldt,
        seed_b,
        None,
    )
    .unwrap();

    let mut vt4 = Transcript::<H>::new(b"ZK_Hiding");
    assert!(
        HekateVerifier::<F, H>::verify(&air, &instance, &p_only_ldt_a, &mut vt4, &config_only_ldt)
            .unwrap(),
        "K=0/ldt=200 proof A must verify"
    );

    let mut vt5 = Transcript::<H>::new(b"ZK_Hiding");
    assert!(
        HekateVerifier::<F, H>::verify(&air, &instance, &p_only_ldt_b, &mut vt5, &config_only_ldt)
            .unwrap(),
        "K=0/ldt=200 proof B must verify"
    );

    let poly_only_ldt_a = &p_only_ldt_a.zerocheck_proof.round_polys[0].evals;
    let poly_only_ldt_b = &p_only_ldt_b.zerocheck_proof.round_polys[0].evals;

    assert_ne!(
        poly_no_zk, poly_only_ldt_a,
        "K=0 round polys must differ when seed/ldt differ"
    );
    assert_ne!(
        poly_only_ldt_a, poly_only_ldt_b,
        "K=0/ldt=200: round polys must diverge across seeds"
    );

    let eval_only_ldt_a = &p_only_ldt_a.eval_proof.point_evaluations[0].1;
    let eval_only_ldt_b = &p_only_ldt_b.eval_proof.point_evaluations[0].1;

    assert_ne!(
        eval_no_zk, eval_only_ldt_a,
        "K=0 evaluations must differ when seed/ldt differ"
    );
    assert_ne!(
        eval_only_ldt_a, eval_only_ldt_b,
        "K=0/ldt=200: trace evaluations must diverge across seeds"
    );
}

#[test]
fn true_zk_memory_isolation() {
    // OBJECTIVE
    //
    // The main-trace LDT commitment must hide raw
    // physical trace bytes. Runs as a paired
    // positive/negative control to guarantee the
    // assertion has bite:
    //
    //   STAGE A (production blinding):
    //   inject a magic value at a single row,
    //   generate a proof, the magic hardware-basis
    //   bytes must NOT appear in main LDT openings.
    //
    //   STAGE B (zero blinding negative control):
    //   same construction with both blinding factors
    //   at zero, the magic hardware-basis bytes MUST
    //   appear at least once - otherwise the positive
    //   case is passing vacuously.

    let num_vars = 4;
    let num_rows = 1 << num_vars;

    let magic_val = 0xDEADBEEF_u32;

    let mut a_col = vec![Block32::ZERO; num_rows];
    a_col[0] = Block32::from(magic_val);

    let b_col = vec![Block32::ZERO; num_rows];
    let sel_col = vec![Bit::ZERO; num_rows];

    let mut trace = ColumnTrace::new(num_vars).unwrap();
    trace.add_column(a_col.into_trace_column()).unwrap();
    trace.add_column(b_col.into_trace_column()).unwrap();
    trace.add_column(TraceColumn::Bit(sel_col)).unwrap();

    let air = FibAir {
        num_cols: 3,
        num_rows,
    };

    // FibAir's transition constraint is
    // `selector * (next_a + b) = 0`. With sel_col
    // all zeros it holds trivially regardless of a/b
    // content. The boundary constraint pins
    // `b[num_rows - 1] = public_input[0] = ZERO`,
    // which the all-zero b_col satisfies.
    let instance = ProgramInstance::new(num_rows, vec![F::ZERO]);
    let witness = ProgramWitness::new(trace);

    let needle: [u8; 4] = Block32::from(magic_val)
        .to_hardware()
        .into_raw()
        .0
        .to_le_bytes();

    let scan_main_openings = |proof: &hekate_core::proofs::InnerProof<F>| -> bool {
        for col_data in &proof.eval_proof.ldt_proof.opened_columns {
            if col_data.windows(needle.len()).any(|w| w == needle) {
                return true;
            }
        }

        false
    };

    // STAGE A
    // Production blinding.
    // Magic bytes MUST be hidden.
    //
    // expansion_degree pinned to 16
    // (= grid_cols for num_vars=4,
    // num_queries=4) so the Stage B
    // negative control below stays
    // deterministic.
    let config_zk = Config {
        sumcheck_blinding_factor: 2,
        ldt_blinding_factor: 2,
        num_queries: 4,
        min_security_bits: 0,
        expansion_degree: 16,
        ..Config::default()
    };

    let proof_zk = prove(
        b"TrueZK", &air, &instance, &witness, &config_zk, [7u8; 32], None,
    )
    .unwrap();

    assert!(
        !scan_main_openings(&proof_zk),
        "STAGE A FAILURE: magic hardware bytes appeared in main LDT openings"
    );

    // STAGE B
    // Zero blinding control.
    // Magic bytes MUST be visible.
    let config_no_zk = Config {
        sumcheck_blinding_factor: 0,
        ldt_blinding_factor: 0,
        num_queries: 4,
        min_security_bits: 0,
        expansion_degree: 16,
        ..Config::default()
    };

    let proof_no_zk = prove(
        b"TrueZK",
        &air,
        &instance,
        &witness,
        &config_no_zk,
        [7u8; 32],
        None,
    )
    .unwrap();

    assert!(
        scan_main_openings(&proof_no_zk),
        "STAGE B FAILURE: magic hardware bytes did NOT appear in main LDT openings"
    );

    // STAGE C
    // K = 0, ldt_blinding = 200.
    // Isolates whether ldt_blinding alone
    // hides raw bytes without sumcheck masks.
    let config_only_ldt = Config {
        sumcheck_blinding_factor: 0,
        ldt_blinding_factor: 200,
        num_queries: 4,
        min_security_bits: 0,
        expansion_degree: 16,
        ..Config::default()
    };

    let proof_only_ldt = prove(
        b"TrueZK",
        &air,
        &instance,
        &witness,
        &config_only_ldt,
        [7u8; 32],
        None,
    )
    .unwrap();

    assert!(
        !scan_main_openings(&proof_only_ldt),
        "STAGE C FAILURE: magic hardware bytes appeared with K=0, ldt_blinding=200"
    );
}

#[test]
fn truncation_overflow_injection() {
    // OBJECTIVE:
    // Ensure that the Verifier strictly rejects
    // out-of-bounds bytes for truncated types (like `Bit`)
    // in the opened LDT leaves. If an attacker injects
    // `0xFE` into a `Bit` column, the protocol must fail cleanly.

    let num_vars = 5;
    let num_rows = 1usize << num_vars;
    let trace = generate_fib_trace(num_vars);
    let expected_pub = trace.get_element(1, num_rows - 1).unwrap().to_tower();
    let instance = ProgramInstance::new(num_rows, vec![expected_pub]);
    let witness = ProgramWitness::new(trace);
    let air = FibAir {
        num_cols: 3,
        num_rows,
    };

    let config = Config {
        sumcheck_blinding_factor: 1,
        ldt_blinding_factor: 1,
        num_queries: 8,
        min_security_bits: 0,
        ..Config::default()
    };

    let mut blinding_seed = [42u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    // 1. Generate a perfectly valid proof
    let mut proof = prove(
        b"ZK_Truncation",
        &air,
        &instance,
        &witness,
        &config,
        blinding_seed,
        None,
    )
    .unwrap();

    let col_data = &mut proof.eval_proof.ldt_proof.opened_columns[0];

    // DYNAMIC ATTACK VECTOR:
    // Automatically calculate the byte
    // offset for the first `Bit` column
    // based on the AIR layout, accounting
    // for base+shift interleaving.
    let mut target_offset = 0;
    for col_type in air.column_layout() {
        if matches!(col_type, ColumnType::Bit) {
            break;
        }

        let col_size = match col_type {
            ColumnType::Bit => 1,
            ColumnType::B8 => 1,
            ColumnType::B16 => 2,
            ColumnType::B32 => 4,
            ColumnType::B64 => 8,
            ColumnType::B128 => 16,
        };

        // Architecture interleaves
        // Base and Shift physical bytes.
        target_offset += col_size * 2;
    }

    // ATTACK:
    // Inject an invalid byte (0xFE)
    // into the dynamic offset.
    // A `Bit` must strictly be
    // evaluated as 0x00 or 0x01.
    col_data[target_offset] = 0xFE;

    // 2. Verify the corrupted proof
    let mut verifier_transcript = Transcript::<H>::new(b"ZK_Truncation");
    let result =
        HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut verifier_transcript, &config);

    // 3. Guarantee rejection
    match result {
        Ok(true) => panic!("Verifier accepted a Truncation Overflow (0xFE) for a Bit column"),
        Ok(false) => {
            // Rejected due to evaluation mismatch (Good)
        }
        Err(_) => {
            // Rejected due to Merkle mismatch / Protocol Error (Good)
        }
    }
}

#[test]
fn noise_shift_sign_forgery() {
    // OBJECTIVE:
    // Ensure that the Verifier detects if the Prover
    // swaps or modifies the base/shifted noise values
    // (B(r) vs B(x_next)). Even if the AIR check passes
    // logically, the TensorPCS binding must fail because
    // the claimed values won't match the committed LDT data.

    let num_vars = 5;
    let num_rows = 1usize << num_vars;
    let trace = generate_fib_trace(num_vars);
    let expected_pub = trace.get_element(1, num_rows - 1).unwrap().to_tower();
    let instance = ProgramInstance::new(num_rows, vec![expected_pub]);
    let witness = ProgramWitness::new(trace);
    let air = FibAir {
        num_cols: 3,
        num_rows,
    };

    let config = Config {
        sumcheck_blinding_factor: 1, // Use 1 noise column (Base + Shifted)
        ldt_blinding_factor: 1,
        num_queries: 4,
        min_security_bits: 0,
        ..Config::default()
    };

    let mut blinding_seed = [42u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    // 1. Generate a perfectly valid proof
    let mut proof = prove(
        b"ZK_NoiseShift",
        &air,
        &instance,
        &witness,
        &config,
        blinding_seed,
        None,
    )
    .unwrap();

    // In Hekate architecture, noise columns
    // are appended after physical data.
    // FibAir has 3 data columns (indices 0, 1, 2).
    // The sumcheck noise column is at index 3.
    let noise_col_idx = 3;

    // ATTACK:
    // Swap the base noise value and the next-row
    // noise value. This "disconnects" the algebraic
    // claim from the physical Merkle tree data.
    let expected_trace_len = air.num_columns() + config.sumcheck_blinding_factor;
    let base_noise = proof.eval_proof.point_evaluations[0].1[noise_col_idx];
    let next_noise = proof.eval_proof.point_evaluations[0].1[expected_trace_len + noise_col_idx];

    proof.eval_proof.point_evaluations[0].1[noise_col_idx] = next_noise;
    proof.eval_proof.point_evaluations[0].1[expected_trace_len + noise_col_idx] = base_noise;

    // 2. Verify the forged proof
    let mut verifier_transcript = Transcript::<H>::new(b"ZK_NoiseShift");
    let result =
        HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut verifier_transcript, &config);

    // 3. Guarantee rejection
    match result {
        Ok(true) => panic!(
            "Verifier accepted swapped noise values! TensorPCS failed to bind trace_values to Merkle root"
        ),
        Ok(false) => {
            // Success: Rejected by Batch Evaluation (TensorPCS)
        }
        Err(_) => {
            // Success: Protocol/Merkle mismatch detected
        }
    }
}

#[test]
fn ghost_protocol_indistinguishability() {
    // OBJECTIVE:
    // Prove that ZK noise makes the padding zone (Ghost Protocol)
    // indistinguishable from real data rows in the LDT openings.
    // A verifier/observer should not be able to guess the program's
    // effective length by looking at the proof's noise distribution.

    let num_vars = 6;
    let num_rows = 1 << num_vars;
    let air = FibAir {
        num_cols: 3,
        num_rows,
    };

    let config = Config {
        sumcheck_blinding_factor: 2,
        ldt_blinding_factor: 2,
        num_queries: 4,
        min_security_bits: 0,
        ..Config::default()
    };

    // 1. Trace A:
    // Program halts early at row 8.
    let mut a_col_short = vec![Block32::ZERO; num_rows];
    for (i, slot) in a_col_short.iter_mut().take(8).enumerate() {
        *slot = Block32::from(i as u32);
    }

    let mut trace_short = ColumnTrace::new(num_vars).unwrap();
    trace_short
        .add_column(a_col_short.into_trace_column())
        .unwrap();
    trace_short
        .add_column(vec![Block32::ZERO; num_rows].into_trace_column())
        .unwrap();
    trace_short
        .add_column(TraceColumn::Bit(vec![Bit::ZERO; num_rows]))
        .unwrap();

    // 2. Trace B:
    // Program runs longer, halts at row 32.
    let mut a_col_long = vec![Block32::ZERO; num_rows];
    for (i, slot) in a_col_long.iter_mut().take(32).enumerate() {
        *slot = Block32::from(i as u32);
    }

    let mut trace_long = ColumnTrace::new(num_vars).unwrap();
    trace_long
        .add_column(a_col_long.into_trace_column())
        .unwrap();
    trace_long
        .add_column(vec![Block32::ZERO; num_rows].into_trace_column())
        .unwrap();
    trace_long
        .add_column(TraceColumn::Bit(vec![Bit::ZERO; num_rows]))
        .unwrap();

    // Generate proofs for both (different data, same ZK config)
    let instance = ProgramInstance::new(num_rows, vec![F::ZERO]);
    let proof_short = prove(
        b"GhostZK",
        &air,
        &instance,
        &ProgramWitness::new(trace_short),
        &config,
        [1u8; 32],
        None,
    )
    .unwrap();

    let proof_long = prove(
        b"GhostZK",
        &air,
        &instance,
        &ProgramWitness::new(trace_long),
        &config,
        [2u8; 32],
        None,
    )
    .unwrap();

    // Check the LDT openings in the zone where
    // Trace A is padding but Trace B is data.
    // In True ZK, all opened bytes must look
    // like high-entropy noise.
    let col_short = &proof_short.eval_proof.ldt_proof.opened_columns[0];
    let col_long = &proof_long.eval_proof.ldt_proof.opened_columns[0];

    // Guarantee:
    // Both columns must be
    // fully populated with noise.
    assert!(
        !col_short.iter().all(|&x| x == 0),
        "Short trace LDT is not blinded"
    );
    assert!(
        !col_long.iter().all(|&x| x == 0),
        "Long trace LDT is not blinded"
    );

    // Guarantee:
    // The "fingerprint" (distribution) should
    // not reveal the halt point. We check that
    // the padding zone in Trace A is not just
    // "zeros" but actual ZK noise.
    let data_bytes_per_row = (4 + 4 + 1) * 2;
    let noise_start = data_bytes_per_row;
    let noise_end = data_bytes_per_row + (config.sumcheck_blinding_factor * 16 * 2);

    let padding_noise_short = &col_short[noise_start..noise_end];
    let padding_noise_long = &col_long[noise_start..noise_end];

    // Statistical check:
    // the noise must be present
    // and non-trivial in both cases.
    assert!(padding_noise_short.iter().filter(|&&x| x != 0).count() > 10);
    assert!(padding_noise_long.iter().filter(|&&x| x != 0).count() > 10);
}

// =================================================================
// Multi-Bus Chiplet ZK Test Fixtures
//
// Synthetic minimal chiplet + program used to exercise
// the chiplet pipeline's multi-open code path.
// =================================================================

/// Smallest possible chiplet with N > 1 GPA buses.
///
/// Two columns:
/// - col 0: B32 payload sourced by every bus
/// - col 1: Bit selector, sticky-end pattern
///   `[1,...,1,0]` (set in `make_minimal_bus_trace`)
///   so the MLE has support but the last row is silent.
#[derive(Clone)]
struct MinimalBusChiplet;

impl Air<F> for MinimalBusChiplet {
    fn name(&self) -> String {
        "minimal_bus".to_string()
    }

    fn num_columns(&self) -> usize {
        2
    }

    fn column_layout(&self) -> &'static [ColumnType] {
        &[ColumnType::B32, ColumnType::Bit]
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        let make_spec = || {
            PermutationCheckSpec::new(
                vec![
                    (Source::Column(0), b"kappa_payload" as &[u8]),
                    (Source::RowIndexLeBytes(4), REQUEST_IDX_LABEL),
                ],
                Some(1),
            )
        };

        vec![
            ("multi_open_bus_0".to_string(), make_spec()),
            ("multi_open_bus_1".to_string(), make_spec()),
            ("multi_open_bus_2".to_string(), make_spec()),
        ]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(1));

        cs.build()
    }
}

/// Bare main program with
/// no permutation checks.
#[derive(Clone)]
struct MultiBusProgram {
    defs: Vec<ChipletDef<F>>,
}

impl Air<F> for MultiBusProgram {
    fn name(&self) -> String {
        "multi_bus_main".to_string()
    }

    fn num_columns(&self) -> usize {
        1
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

impl Program<F> for MultiBusProgram {
    fn chiplet_defs(&self) -> hekate_core::errors::Result<Vec<ChipletDef<F>>> {
        Ok(self.defs.clone())
    }
}

fn make_minimal_main_trace(num_vars: usize) -> ColumnTrace {
    let layout = [ColumnType::Bit];
    let tb = TraceBuilder::new(&layout, num_vars).unwrap();

    tb.build()
}

/// Builds a chiplet trace where the data column
/// carries `magic` at row 0 and zeros elsewhere.
///
/// Selector uses the sticky-end pattern `[1, ..., 1, 0]`
/// so its MLE is non-trivial and the bus has well-defined
/// active multisets.
fn make_minimal_bus_trace(num_vars: usize, magic: u32) -> ColumnTrace {
    let layout = [ColumnType::B32, ColumnType::Bit];
    let mut tb = TraceBuilder::new(&layout, num_vars).unwrap();
    let num_rows = tb.num_rows();

    tb.set_b32(0, 0, Block32::from(magic)).unwrap();
    tb.fill_selector(1, num_rows - 1).unwrap();

    tb.build()
}

#[test]
fn chiplet_pipeline_witness_isolation() {
    // OBJECTIVE
    //
    // The chiplet Brakedown LDT openings must
    // not contain raw witness bytes in cleartext.
    // This test runs as a paired positive/negative
    // control:
    //
    //   STAGE A (production blinding):
    //   inject a magic byte sequence into the
    //   chiplet trace, generate a proof, scan
    //   ALL chiplet LDT opened bytes, the
    //   magic sequence must NOT appear.
    //
    //   STAGE B (zero blinding negative control):
    //   same construction with all blinding
    //   factors at zero, the magic sequence
    //   MUST appear at least once. This proves
    //   assertion has bite, without the negative,
    //   "bytes don't appear" could be coincidence
    //   from matrix XOR cancellation.
    //
    // What this catches end-to-end:
    // - chiplet pipeline forgets to
    //   apply LDT noise injection.
    // - chiplet path silently overrides
    //   ldt_blinding_factor.
    // - any code change that lets
    //   raw chiplet bytes reach:
    //   `chiplet_eval_proofs[..].ldt_proof.opened_columns`

    const MAGIC: u32 = 0xDEAD_BEEF;

    let needle: [u8; 4] = Block32::from(MAGIC)
        .to_hardware()
        .into_raw()
        .0
        .to_le_bytes();

    let num_vars = 4;
    let num_rows = 1usize << num_vars;

    // Two copies share bus_ids by design,
    // `Program::chiplet_defs()`does NOT
    // auto-namespace (only `CompositeChiplet` does),
    // so each bus pairs across the two copies
    // (chiplet0::bus_n <> chiplet1::bus_n)
    // and satisfies GPA bus exhaustiveness
    // with exactly 2 endpoints.
    let chiplet = MinimalBusChiplet;
    let air = MultiBusProgram {
        defs: vec![
            ChipletDef::from_air(&chiplet).unwrap(),
            ChipletDef::from_air(&chiplet).unwrap(),
        ],
    };

    let main_trace = make_minimal_main_trace(num_vars);
    let chiplet_trace_0 = make_minimal_bus_trace(num_vars, MAGIC);
    let chiplet_trace_1 = make_minimal_bus_trace(num_vars, MAGIC);

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness =
        ProgramWitness::new(main_trace).with_chiplets(vec![chiplet_trace_0, chiplet_trace_1]);

    let seed = [0xA5u8; 32];

    // Local helper:
    // scan every byte of every opened column
    // for both chiplets and report whether
    // the magic needle appears anywhere.
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

    // STAGE A
    // Production blinding.
    // Magic bytes MUST be hidden.
    //
    // expansion_degree MUST equal grid_cols
    // (=16 for num_vars=4, num_queries=4) so
    // the Stage B negative control below is
    // deterministic. Pinned here so the
    // test does not silently break if
    // Config::default().expansion_degree changes.
    let config_zk = Config {
        num_queries: 4,
        min_security_bits: 0,
        sumcheck_blinding_factor: 2,
        ldt_blinding_factor: 4,
        expansion_degree: 16,
        ..Config::default()
    };

    let proof_zk = prove(
        b"ChipletWitnessIsolation",
        &air,
        &instance,
        &witness,
        &config_zk,
        seed,
        None,
    )
    .expect("ZK proof generation must succeed");

    let mut vt_zk = Transcript::<H>::new(b"ChipletWitnessIsolation");
    assert!(
        HekateVerifier::<F, H>::verify(&air, &instance, &proof_zk, &mut vt_zk, &config_zk).unwrap(),
        "ZK proof must verify",
    );

    let leaked_zk = scan_chiplet_openings(&proof_zk);
    assert!(
        !leaked_zk,
        "STAGE A FAILURE: magic witness bytes 0x{MAGIC:08X} appeared in chiplet LDT openings",
    );

    // STAGE B
    // Zero blinding control.
    // Magic bytes MUST be visible.
    //
    // Stage B works because the test sizing is
    // engineered to make the leak deterministic:
    // with num_vars=4 and num_queries=4,
    // compute_split_vars yields grid_cols=16,
    // matching the pinned expansion_degree=16.
    // The rejection sampler in the binary expander
    // matrix has no choice, every output row must
    // select every input column. Each encoded row
    // therefore equals magic XOR 0 XOR ... XOR 0
    // = magic, and the needle appears in every
    // opened column. With both blinding factors at
    // 0, no noise columns are appended, so those
    // raw encoded bytes reach the LDT openings.
    let config_no_zk = Config {
        num_queries: 4,
        min_security_bits: 0,
        sumcheck_blinding_factor: 0,
        ldt_blinding_factor: 0,
        expansion_degree: 16,
        ..Config::default()
    };

    let proof_no_zk = prove(
        b"ChipletWitnessIsolation",
        &air,
        &instance,
        &witness,
        &config_no_zk,
        seed,
        None,
    )
    .expect("zero-blinding proof generation must succeed");

    let leaked_no_zk = scan_chiplet_openings(&proof_no_zk);
    assert!(
        leaked_no_zk,
        "STAGE B FAILURE: magic witness bytes 0x{MAGIC:08X} did NOT appear in chiplet LDT openings",
    );
}
