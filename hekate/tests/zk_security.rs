// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate::core::config::Config;
use hekate::core::trace::{ColumnTrace, ColumnType, TraceColumn};
use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::{Block128, Flat, HardwareField, TowerField};
use hekate_core::poly::PolyVariant;
use hekate_core::trace::{IntoTraceColumn, Trace, TraceBuilder};
use hekate_math::{Bit, Block32};
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::constraint::{BoundaryConstraint, BoundaryTarget, ConstraintAst};
use hekate_program::digest::program_id;
use hekate_program::fold_pool::FoldPool;
use hekate_program::outer::{TableParts, TableShape, table_plan};
use hekate_program::permutation::{PermutationCheckSpec, Source};
use hekate_program::{Air, FixedColumn, FixedShape, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;
use rand::{TryRngCore, rngs::OsRng};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

type F = Block128;
type H = DefaultHasher;

fn fib_air(num_rows: usize) -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("FibZk", num_rows).unwrap();

    let words = cx.columns(2, ColumnType::B32);
    let q = cx.column(ColumnType::Bit);

    let cs = cx.cs();

    let [a, b] = [cs.col(words.at(0).index()), cs.col(words.at(1).index())];
    let [na, nb] = [cs.next(words.at(0).index()), cs.next(words.at(1).index())];

    let q_cell = cs.col(q.index());

    cs.constrain(q_cell * (na + b));
    cs.constrain(q_cell * (nb + a + b));

    cx.fix(q, FixedShape::LastRow);
    cx.boundary(words.at(0), 0, F::ZERO);
    cx.boundary(words.at(1), 0, F::ONE);
    cx.publish(words.at(1), num_rows - 1);

    cx.compile().unwrap()
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

        sel_col.push(if i == num_rows - 1 {
            Bit::ZERO
        } else {
            Bit::ONE
        });

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

fn fold_pool(air: &CircuitProgram<F>, num_vars: usize, config: &Config) -> FoldPool {
    let parts = TableParts::of(air);
    let shape = TableShape::from_air(air, &parts.statics()).unwrap();
    let plan = table_plan(air, &shape, config).unwrap();

    FoldPool::new(&[(&plan, num_vars)], config).unwrap()
}

fn column_mle_pair(trace: &ColumnTrace, col: usize, weights: &[Flat<F>]) -> (Flat<F>, Flat<F>) {
    let num_rows = trace.num_rows().unwrap();

    let mut base = Flat::from_raw(F::ZERO);
    let mut next = Flat::from_raw(F::ZERO);

    for (row, &weight) in weights.iter().enumerate().take(num_rows) {
        let cur = trace.get_element::<F>(col, row).unwrap();
        let nxt = trace.get_element::<F>(col, (row + 1) % num_rows).unwrap();

        base += cur * weight;
        next += nxt * weight;
    }

    (base, next)
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

    let air = fib_air(num_rows);

    let config = Config {
        zero_knowledge: true,
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

    let leaves = &proof.eval_proof.openings[0].leaves;
    let leaf_bytes = fold_pool(&air, num_vars, &config).layout().trees()[0].leaf_bytes;

    assert!(!leaves.is_empty(), "Must open trace leaves");
    assert_eq!(
        leaves.len() % leaf_bytes,
        0,
        "Opened trace leaves must be whole leaves"
    );

    // Calculate Shannon Entropy:
    // H = - sum(p_i * log2(p_i))
    let mut counts = [0usize; 256];
    for &b in leaves {
        counts[usize::from(b)] += 1;
    }

    let mut entropy = 0.0f64;
    let total = leaves.len() as f64;

    for &count in &counts {
        if count > 0 {
            let p = count as f64 / total;
            entropy -= p * p.log2();
        }
    }

    // BIRTHDAY PARADOX MATH:
    // The openings may be small samples (e.g., 256 bytes).
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
    let air = fib_air(num_rows);
    let instance = ProgramInstance::new(num_rows, vec![expected_pub]);

    let config_a = Config {
        zero_knowledge: true,
        num_queries: 8,
        fold_arity_log: 1,
        early_stop_vars: 1,
        min_security_bits: 0,
        ..Config::default()
    };

    let mut seed1 = [0u8; 32];
    seed1[0] = 1;

    let config_b = Config {
        zero_knowledge: true,
        num_queries: 8,
        fold_arity_log: 1,
        early_stop_vars: 1,
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
        proof_a.trace_root, proof_b.trace_root,
        "Merkle root must depend on blinding seed"
    );

    let poly_a_r0 = &proof_a.zerocheck_proof.round_polys[0].evals;
    let poly_b_r0 = &proof_b.zerocheck_proof.round_polys[0].evals;

    assert_ne!(
        poly_a_r0, poly_b_r0,
        "Sumcheck rounds must differ when blinding seed changes"
    );

    let leaves_a = &proof_a.eval_proof.openings[0].leaves;
    let leaves_b = &proof_b.eval_proof.openings[0].leaves;

    assert_ne!(leaves_a, leaves_b, "Opened leaves must differ");

    let mut vt_a = Transcript::<H>::new(b"ZK_Seed");
    let ok_a = HekateVerifier::<F, H>::verify(
        &program_id(&air).unwrap(),
        &air,
        &instance,
        &proof_a,
        &mut vt_a,
        &config_a,
    )
    .unwrap();

    assert!(ok_a, "Proof A must verify");

    let mut vt_b = Transcript::<H>::new(b"ZK_Seed");
    let ok_b = HekateVerifier::<F, H>::verify(
        &program_id(&air).unwrap(),
        &air,
        &instance,
        &proof_b,
        &mut vt_b,
        &config_b,
    )
    .unwrap();

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

    let air = fib_air(num_rows);

    let config = Config {
        zero_knowledge: true,
        num_queries: 8,
        fold_arity_log: 1,
        early_stop_vars: 1,
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

    let leaf_bytes = fold_pool(&air, num_vars, &config).layout().trees()[0].leaf_bytes;

    // ATTACK:
    // maliciously zero the first
    // opened leaf of the trace tree.
    for b in &mut proof.eval_proof.openings[0].leaves[..leaf_bytes] {
        *b = 0;
    }

    let mut verifier_transcript = Transcript::<H>::new(b"ZK_Integrity");
    let result = HekateVerifier::<F, H>::verify(
        &program_id(&air).unwrap(),
        &air,
        &instance,
        &proof,
        &mut verifier_transcript,
        &config,
    );

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
            vec![BoundaryConstraint {
                col_idx: 1,
                row_idx: self.num_rows - 1,
                target: BoundaryTarget::PublicInput(0),
            }]
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
    let num_vars = 5;
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
    // ATTACK WITH ZK ENABLED
    // ==========================================
    let config_zk = Config {
        num_queries: 8,
        fold_arity_log: 1,
        early_stop_vars: 1,
        min_security_bits: 0,
        zero_knowledge: true,
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
        &program_id(&air).unwrap(),
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
        zero_knowledge: false,
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
        &program_id(&air).unwrap(),
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
    let air = fib_air(num_rows);

    let config_no_zk = Config {
        zero_knowledge: false,
        num_queries: 4,
        fold_arity_log: 1,
        early_stop_vars: 1,
        min_security_bits: 0,
        ..Config::default()
    };

    let config_zk = Config {
        zero_knowledge: true,
        num_queries: 4,
        fold_arity_log: 1,
        early_stop_vars: 1,
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
        HekateVerifier::<F, H>::verify(
            &program_id(&air).unwrap(),
            &air,
            &instance,
            &p_no_zk,
            &mut vt1,
            &config_no_zk,
        )
        .unwrap(),
        "No-ZK proof failed to verify"
    );

    let mut vt2 = Transcript::<H>::new(b"ZK_Hiding");
    assert!(
        HekateVerifier::<F, H>::verify(
            &program_id(&air).unwrap(),
            &air,
            &instance,
            &p_zk_a,
            &mut vt2,
            &config_zk,
        )
        .unwrap(),
        "ZK proof A failed to verify"
    );

    let mut vt3 = Transcript::<H>::new(b"ZK_Hiding");
    assert!(
        HekateVerifier::<F, H>::verify(
            &program_id(&air).unwrap(),
            &air,
            &instance,
            &p_zk_b,
            &mut vt3,
            &config_zk,
        )
        .unwrap(),
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
    // Seed dependence only:
    // r_final differs across seeds.
    let eval_no_zk = &p_no_zk.main_point_evaluation.1;
    let eval_zk_a = &p_zk_a.main_point_evaluation.1;
    let eval_zk_b = &p_zk_b.main_point_evaluation.1;

    assert_ne!(eval_no_zk, eval_zk_a, "ZK failed to hide trace evaluations");
    assert_ne!(eval_zk_a, eval_zk_b, "ZK failed to hide trace evaluations");

    // Raw-mode control: the proof draws no randomness without zero knowledge
    let p_raw_a = prove(
        b"ZK_Hiding",
        &air,
        &instance,
        &witness,
        &config_no_zk,
        seed_a,
        None,
    )
    .unwrap();

    let p_raw_b = prove(
        b"ZK_Hiding",
        &air,
        &instance,
        &witness,
        &config_no_zk,
        seed_b,
        None,
    )
    .unwrap();

    assert_eq!(
        poly_no_zk, &p_raw_a.zerocheck_proof.round_polys[0].evals,
        "raw round polys must not depend on the seed"
    );
    assert_eq!(
        poly_no_zk, &p_raw_b.zerocheck_proof.round_polys[0].evals,
        "raw round polys must not depend on the seed"
    );
    assert_eq!(
        eval_no_zk, &p_raw_a.main_point_evaluation.1,
        "raw evaluations must not depend on the seed"
    );
    assert_eq!(
        eval_no_zk, &p_raw_b.main_point_evaluation.1,
        "raw evaluations must not depend on the seed"
    );
}

/// Per column: `(claim, MLE(r_final))` for the
/// base and next-row halves of the trace opening.
fn point_evaluation_claims_vs_mle(zero_knowledge: bool) -> Vec<[(F, F); 2]> {
    let num_vars = 6;
    let num_rows = 1 << num_vars;
    let trace = generate_fib_trace(num_vars);
    let expected_pub = trace.get_element(1, num_rows - 1).unwrap().to_tower();
    let instance = ProgramInstance::new(num_rows, vec![expected_pub]);
    let witness = ProgramWitness::new(trace.clone());
    let air = fib_air(num_rows);

    let config = Config {
        zero_knowledge,
        num_queries: 4,
        fold_arity_log: 1,
        early_stop_vars: 1,
        min_security_bits: 0,
        ..Config::default()
    };

    let mut seed = [0u8; 32];
    seed[0] = 7;

    let proof = prove(b"ZK_Hiding", &air, &instance, &witness, &config, seed, None).unwrap();

    let mut vt = Transcript::<H>::new(b"ZK_Hiding");
    assert!(
        HekateVerifier::<F, H>::verify(
            &program_id(&air).unwrap(),
            &air,
            &instance,
            &proof,
            &mut vt,
            &config,
        )
        .unwrap()
    );

    let (r_final, claims) = &proof.main_point_evaluation;
    let num_cols = air.num_columns();
    let k = config.blind_units();

    assert_eq!(r_final.len(), num_vars);
    assert_eq!(claims.len(), 2 * (num_cols + k));

    let r_hw: Vec<Flat<F>> = r_final.iter().map(|x| x.to_hardware()).collect();
    let weights = PolyVariant::<F>::expand_mle_weights(&r_hw);

    (0..num_cols)
        .map(|col| {
            let (base, next) = column_mle_pair(&trace, col, &weights);

            [
                (claims[col], base.to_tower()),
                (claims[num_cols + k + col], next.to_tower()),
            ]
        })
        .collect()
}

#[test]
fn point_evaluation_claims_are_not_witness_functionals() {
    for (col, [base, next]) in point_evaluation_claims_vs_mle(false)
        .into_iter()
        .enumerate()
    {
        assert_eq!(base.0, base.1, "raw base claim of column {col}");
        assert_eq!(next.0, next.1, "raw next-row claim of column {col}");
    }

    for (col, [base, next]) in point_evaluation_claims_vs_mle(true).into_iter().enumerate() {
        assert_ne!(base.0, base.1, "base claim of column {col}");
        assert_ne!(next.0, next.1, "next-row claim of column {col}");
    }
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
    //   bytes must NOT appear in any fold opening.
    //
    //   STAGE B (positive control):
    //   a needle planted in an opened leaf
    //   must be found by the same scan.

    let num_vars = 5;
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

    let air = fib_air(num_rows);

    let instance = ProgramInstance::new(num_rows, vec![F::ZERO]);
    let witness = ProgramWitness::new(trace);

    let needle: [u8; 4] = Block32::from(magic_val)
        .to_hardware()
        .into_raw()
        .0
        .to_le_bytes();

    let scan_main_openings = |proof: &hekate_core::proofs::InnerProof<F>| -> bool {
        proof
            .eval_proof
            .openings
            .iter()
            .any(|opening| opening.leaves.windows(needle.len()).any(|w| w == needle))
    };

    // STAGE A
    // Production blinding.
    // Magic bytes MUST be hidden.
    let config_zk = Config {
        zero_knowledge: true,
        num_queries: 4,
        fold_arity_log: 1,
        early_stop_vars: 1,
        min_security_bits: 0,
        ..Config::default()
    };

    let proof_zk = prove(
        b"TrueZK", &air, &instance, &witness, &config_zk, [7u8; 32], None,
    )
    .unwrap();

    assert!(
        !scan_main_openings(&proof_zk),
        "STAGE A FAILURE: magic hardware bytes appeared in fold openings"
    );

    // STAGE B
    // The non-systematic code emits no verbatim
    // cell, no real leak exists to detect.
    let mut planted = proof_zk.clone();

    let target = planted
        .eval_proof
        .openings
        .iter_mut()
        .map(|opening| &mut opening.leaves)
        .find(|leaves| leaves.len() >= needle.len())
        .expect("an opening wide enough to hold the needle");

    target[..needle.len()].copy_from_slice(&needle);

    assert!(
        scan_main_openings(&planted),
        "positive control: a planted needle must be detected by the scan",
    );
}

#[test]
fn truncation_overflow_injection() {
    // OBJECTIVE:
    // Ensure that the Verifier strictly rejects an opened
    // code symbol whose top byte an attacker set to `0xFE`.

    let num_vars = 5;
    let num_rows = 1usize << num_vars;
    let trace = generate_fib_trace(num_vars);
    let expected_pub = trace.get_element(1, num_rows - 1).unwrap().to_tower();
    let instance = ProgramInstance::new(num_rows, vec![expected_pub]);
    let witness = ProgramWitness::new(trace);
    let air = fib_air(num_rows);

    let config = Config {
        zero_knowledge: true,
        num_queries: 8,
        fold_arity_log: 1,
        early_stop_vars: 1,
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

    let target_offset = fold_pool(&air, num_vars, &config).layout().polys()[0]
        .field
        .bytes()
        - 1;

    let leaves = &mut proof.eval_proof.openings[0].leaves;

    // ATTACK:
    // Inject 0xFE into the top byte of the first opened symbol.
    leaves[target_offset] = match leaves[target_offset] {
        0xFE => 0xFD,
        _ => 0xFE,
    };

    // 2. Verify the corrupted proof
    let mut verifier_transcript = Transcript::<H>::new(b"ZK_Truncation");
    let result = HekateVerifier::<F, H>::verify(
        &program_id(&air).unwrap(),
        &air,
        &instance,
        &proof,
        &mut verifier_transcript,
        &config,
    );

    // 3. Guarantee rejection
    match result {
        Ok(true) => panic!("Verifier accepted an opened symbol with a forged top byte"),
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
    // logically, the PCS binding must fail because
    // the claimed values won't match the committed data.

    let num_vars = 5;
    let num_rows = 1usize << num_vars;
    let trace = generate_fib_trace(num_vars);
    let expected_pub = trace.get_element(1, num_rows - 1).unwrap().to_tower();
    let instance = ProgramInstance::new(num_rows, vec![expected_pub]);
    let witness = ProgramWitness::new(trace);
    let air = fib_air(num_rows);

    let config = Config {
        zero_knowledge: true,
        num_queries: 4,
        fold_arity_log: 1,
        early_stop_vars: 1,
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
    let expected_trace_len = proof.main_point_evaluation.1.len() / 2;
    let base_noise = proof.main_point_evaluation.1[noise_col_idx];
    let next_noise = proof.main_point_evaluation.1[expected_trace_len + noise_col_idx];

    proof.main_point_evaluation.1[noise_col_idx] = next_noise;
    proof.main_point_evaluation.1[expected_trace_len + noise_col_idx] = base_noise;

    // 2. Verify the forged proof
    let mut verifier_transcript = Transcript::<H>::new(b"ZK_NoiseShift");
    let result = HekateVerifier::<F, H>::verify(
        &program_id(&air).unwrap(),
        &air,
        &instance,
        &proof,
        &mut verifier_transcript,
        &config,
    );

    // 3. Guarantee rejection
    match result {
        Ok(true) => panic!(
            "Verifier accepted swapped noise values! The PCS failed to bind trace_values to the root"
        ),
        Ok(false) => {
            // Success: Rejected by the pool eval
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
    // indistinguishable from real data rows in the fold openings.
    // A verifier/observer should not be able to guess the program's
    // effective length by looking at the proof's noise distribution.

    let num_vars = 6;
    let num_rows = 1 << num_vars;
    let air = fib_air(num_rows);

    let config = Config {
        zero_knowledge: true,
        num_queries: 4,
        fold_arity_log: 1,
        early_stop_vars: 1,
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

    // Every opened symbol mixes all rows with the noise block:
    // in True ZK, all opened bytes must look like high-entropy noise.
    let leaves_short = &proof_short.eval_proof.openings[0].leaves;
    let leaves_long = &proof_long.eval_proof.openings[0].leaves;

    // Guarantee:
    // Both openings must be fully populated with noise.
    assert!(
        !leaves_short.iter().all(|&x| x == 0),
        "Short trace openings are not blinded"
    );
    assert!(
        !leaves_long.iter().all(|&x| x == 0),
        "Long trace openings are not blinded"
    );

    // Statistical check:
    // the noise must be present and non-trivial in both cases.
    assert!(leaves_short.iter().filter(|&&x| x != 0).count() > 10);
    assert!(leaves_long.iter().filter(|&&x| x != 0).count() > 10);
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
struct MinimalBusChiplet {
    num_rows: usize,
}

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
                    (Source::RowIndexLeBytes(4), b"kappa_clk" as &[u8]),
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

    fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
        vec![FixedColumn::prefix(1, self.num_rows - 1)]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
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
    // The fold openings must not contain raw
    // chiplet witness bytes in cleartext.
    // This test runs as a paired positive/negative
    // control:
    //
    //   STAGE A (production blinding):
    //   inject a magic byte sequence into the
    //   chiplet trace, generate a proof, scan
    //   ALL opened fold bytes, the
    //   magic sequence must NOT appear.
    //
    //   STAGE B (positive control):
    //   a needle planted in an opened leaf
    //   must be found by the same scan.
    //
    // What this catches end-to-end:
    // - any code change that lets
    //   raw chiplet bytes reach:
    //   `eval_proof.openings`

    const MAGIC: u32 = 0xDEAD_BEEF;

    let needle: [u8; 4] = Block32::from(MAGIC)
        .to_hardware()
        .into_raw()
        .0
        .to_le_bytes();

    let num_vars = 5;
    let num_rows = 1usize << num_vars;

    // Two copies share bus_ids by design,
    // `Program::chiplet_defs()` does NOT
    // auto-namespace (only `Circuit::attach_namespaced` does),
    // so each bus pairs across the two copies
    // (chiplet0::bus_n <> chiplet1::bus_n)
    // and satisfies GPA bus exhaustiveness
    // with exactly 2 endpoints.
    let chiplet = MinimalBusChiplet { num_rows };
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

    let scan_openings = |proof: &hekate_core::proofs::InnerProof<F>| -> bool {
        proof
            .eval_proof
            .openings
            .iter()
            .any(|opening| opening.leaves.windows(needle.len()).any(|w| w == needle))
    };

    // STAGE A
    // Production blinding.
    // Magic bytes MUST be hidden.
    let config_zk = Config {
        num_queries: 4,
        fold_arity_log: 1,
        early_stop_vars: 1,
        min_security_bits: 0,
        zero_knowledge: true,
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
        HekateVerifier::<F, H>::verify(
            &program_id(&air).unwrap(),
            &air,
            &instance,
            &proof_zk,
            &mut vt_zk,
            &config_zk,
        )
        .unwrap(),
        "ZK proof must verify",
    );

    let leaked_zk = scan_openings(&proof_zk);
    assert!(
        !leaked_zk,
        "STAGE A FAILURE: magic witness bytes 0x{MAGIC:08X} appeared in fold openings",
    );

    // STAGE B
    // The non-systematic code emits no verbatim
    // cell, no real leak exists to detect.
    let mut planted = proof_zk.clone();

    let target = planted
        .eval_proof
        .openings
        .iter_mut()
        .map(|opening| &mut opening.leaves)
        .find(|leaves| leaves.len() >= needle.len())
        .expect("opening wide enough to hold the needle");

    target[..needle.len()].copy_from_slice(&needle);

    assert!(
        scan_openings(&planted),
        "positive control: a planted needle must be detected by the scan",
    );
}
