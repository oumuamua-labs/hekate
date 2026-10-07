// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate::core::config::Config;
use hekate::core::trace::{ColumnTrace, ColumnType, TraceColumn};
use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::{Block128, TowerField};
use hekate_core::errors::Error;
use hekate_core::trace::{IntoTraceColumn, Trace};
use hekate_math::{Bit, Block32};
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::digest::program_id;
use hekate_program::{FixedShape, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;
use hekate_verifier::prepared::{PreparedProgram, VerifierScratch};
use proptest::prelude::*;
use rand::{TryRngCore, rngs::OsRng};

type F = Block128;
type H = DefaultHasher;

#[allow(dead_code)]
fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_test_writer()
        .with_max_level(tracing::Level::DEBUG)
        .try_init();
}

fn test_config() -> Config {
    Config {
        // Integration tests must be fast.
        // We keep security checks disabled here.
        num_queries: 8,
        min_security_bits: 0,
        ..Config::default()
    }
}

fn fib_program(num_rows: usize) -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("Fib", num_rows).unwrap();

    let words = cx.columns(2, ColumnType::B32);
    let q = cx.column(ColumnType::Bit);

    let cs = cx.cs();

    let [a, b] = [cs.col(words.at(0).index()), cs.col(words.at(1).index())];
    let q_cell = cs.col(q.index());
    let [na, nb] = [cs.next(words.at(0).index()), cs.next(words.at(1).index())];

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

#[test]
fn air_fib_e2e() {
    // init_tracing();

    let num_vars = 8;
    let num_rows = 1 << num_vars;
    let seed = [0xAAu8; 32];

    let trace = generate_fib_trace(num_vars);
    let expected_pub = trace.get_element(1, num_rows - 1).unwrap().to_tower();
    let instance = ProgramInstance::new(num_rows, vec![expected_pub]);
    let witness = ProgramWitness::new(trace);

    let air = fib_program(num_rows);

    let config = test_config();

    let proof = prove(
        b"FibAir_E2E",
        &air,
        &instance,
        &witness,
        &config,
        seed,
        None,
    )
    .unwrap();

    let mut verifier_transcript = Transcript::<H>::new(b"FibAir_E2E");
    let result = HekateVerifier::<F, H>::verify(
        &program_id(&air).unwrap(),
        &air,
        &instance,
        &proof,
        &mut verifier_transcript,
        &config,
    );

    match result {
        Ok(true) => {}
        Ok(false) => panic!("Program verification returned false"),
        Err(e) => panic!("Program verification error: {:?}", e),
    }
}

#[test]
fn transcript_binding_security_trace_root_changes_challenges() {
    // Security test: if the trace commitment root changes,
    // the prover/verifier transcript challenges MUST change.

    let num_vars = 4;
    let num_rows = 1 << num_vars;
    let seed = [0xAAu8; 32];

    let trace = generate_fib_trace(num_vars);
    let expected_pub = trace.get_element(1, num_rows - 1).unwrap().to_tower();
    let instance = ProgramInstance::new(num_rows, vec![expected_pub]);
    let witness = ProgramWitness::new(trace);

    let air = fib_program(num_rows);

    let config_a = Config {
        num_queries: 8,
        min_security_bits: 0,
        ldt_support_size: 4,
        zero_knowledge: true,
        ..Config::default()
    };

    let config_b = Config {
        num_queries: 8,
        min_security_bits: 0,
        ldt_support_size: 4,
        zero_knowledge: false,
        ..Config::default()
    };

    let proof_a = prove(
        b"BindingTest",
        &air,
        &instance,
        &witness,
        &config_a,
        seed,
        None,
    )
    .unwrap();

    let proof_b = prove(
        b"BindingTest",
        &air,
        &instance,
        &witness,
        &config_b,
        seed,
        None,
    )
    .unwrap();

    assert_ne!(
        proof_a.trace_root, proof_b.trace_root,
        "Sanity: different configs must yield different trace roots"
    );

    let alpha_a = {
        let mut t = Transcript::<H>::new(b"BindingTest");
        t.append_message(b"trace_root", &proof_a.trace_root);
        t.challenge_field::<F>(b"alpha").unwrap()
    };

    let alpha_b = {
        let mut t = Transcript::<H>::new(b"BindingTest");
        t.append_message(b"trace_root", &proof_b.trace_root);
        t.challenge_field::<F>(b"alpha").unwrap()
    };

    assert_ne!(
        alpha_a, alpha_b,
        "CRITICAL SECURITY FAIL: Transcript challenge did not change when trace root changed"
    );
}

#[test]
fn zk_air_happy_path() {
    // Scenario:
    // End-to-end Program proving
    // with blinding enabled.

    let num_vars = 8;
    let num_rows = 1 << num_vars;

    let trace = generate_fib_trace(num_vars);
    let expected_pub = trace.get_element(1, num_rows - 1).unwrap().to_tower();
    let instance = ProgramInstance::new(num_rows, vec![expected_pub]);
    let witness = ProgramWitness::new(trace);

    let air = fib_program(num_rows);

    let mut config = test_config();
    config.zero_knowledge = true;

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    let proof = prove(
        b"FibAir_ZK",
        &air,
        &instance,
        &witness,
        &config,
        blinding_seed,
        None,
    )
    .unwrap();

    let mut verifier_transcript = Transcript::<H>::new(b"FibAir_ZK");
    let ok = HekateVerifier::<F, H>::verify(
        &program_id(&air).unwrap(),
        &air,
        &instance,
        &proof,
        &mut verifier_transcript,
        &config,
    )
    .unwrap();

    assert!(ok, "ZK Program verification failed");
}

#[test]
fn prepared_verify_matches_verify_across_reuse() {
    let mut config = test_config();
    config.zero_knowledge = true;

    let mut scratch = VerifierScratch::<F>::new();

    for num_vars in [8, 10] {
        let num_rows = 1 << num_vars;
        let air = fib_program(num_rows);
        let id = program_id(&air).unwrap();
        let prepared = PreparedProgram::new(&air, &config).unwrap();

        assert_eq!(prepared.program_id(), &id);

        let trace = generate_fib_trace(num_vars);
        let expected_pub = trace.get_element(1, num_rows - 1).unwrap().to_tower();

        let honest = ProgramInstance::new(num_rows, vec![expected_pub]);
        let forged = ProgramInstance::new(num_rows, vec![expected_pub + F::ONE]);

        let proof = prove(
            b"FibAir_Prepared",
            &air,
            &honest,
            &ProgramWitness::new(trace),
            &config,
            [0x5A; 32],
            None,
        )
        .unwrap();

        for (instance, accepts) in [(&honest, true), (&forged, false), (&honest, true)] {
            let fresh = HekateVerifier::<F, H>::verify(
                &id,
                &air,
                instance,
                &proof,
                &mut Transcript::<H>::new(b"FibAir_Prepared"),
                &config,
            );

            let reused = HekateVerifier::<F, H>::verify_prepared(
                &id,
                &prepared,
                instance,
                &proof,
                &mut Transcript::<H>::new(b"FibAir_Prepared"),
                &mut scratch,
            );

            assert_eq!(fresh.as_ref().ok(), reused.as_ref().ok(), "2^{num_vars}");
            assert_eq!(matches!(reused, Ok(true)), accepts, "2^{num_vars}");
        }
    }
}

#[test]
fn prepared_program_rejects_another_programs_id() {
    let config = test_config();
    let num_vars = 8;
    let num_rows = 1 << num_vars;

    let air = fib_program(num_rows);
    let other = fib_program(2 * num_rows);
    let other_id = program_id(&other).unwrap();

    let prepared = PreparedProgram::new(&air, &config).unwrap();

    assert_ne!(prepared.program_id(), &other_id);

    let trace = generate_fib_trace(num_vars);
    let expected_pub = trace.get_element(1, num_rows - 1).unwrap().to_tower();

    let instance = ProgramInstance::new(num_rows, vec![expected_pub]);

    let proof = prove(
        b"FibAir_OtherId",
        &air,
        &instance,
        &ProgramWitness::new(trace),
        &config,
        [0x3C; 32],
        None,
    )
    .unwrap();

    let outcome = HekateVerifier::<F, H>::verify_prepared(
        &other_id,
        &prepared,
        &instance,
        &proof,
        &mut Transcript::<H>::new(b"FibAir_OtherId"),
        &mut VerifierScratch::new(),
    );

    assert!(matches!(outcome, Err(Error::ProgramIdMismatch { .. })));
}

#[test]
fn verify_batch_rejects_exactly_forged_position() {
    let mut config = test_config();
    config.zero_knowledge = true;

    let num_vars = 8;
    let num_rows = 1 << num_vars;

    let air = fib_program(num_rows);
    let id = program_id(&air).unwrap();
    let prepared = PreparedProgram::new(&air, &config).unwrap();

    let trace = generate_fib_trace(num_vars);
    let expected_pub = trace.get_element(1, num_rows - 1).unwrap().to_tower();

    let honest = ProgramInstance::new(num_rows, vec![expected_pub]);
    let forged = ProgramInstance::new(num_rows, vec![expected_pub + F::ONE]);

    let proof = prove(
        b"FibAir_Batch",
        &air,
        &honest,
        &ProgramWitness::new(trace),
        &config,
        [0x77; 32],
        None,
    )
    .unwrap();

    for bad in 0..4 {
        let batch: Vec<_> = (0..4)
            .map(|i| match i == bad {
                true => (&forged, &proof),
                false => (&honest, &proof),
            })
            .collect();

        let results = HekateVerifier::<F, H>::verify_batch(&id, &prepared, b"FibAir_Batch", &batch);

        for (i, result) in results.iter().enumerate() {
            assert_eq!(
                matches!(result, Ok(true)),
                i != bad,
                "forged at {bad}, item {i}"
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn fuzz_air_completeness(
        num_vars in 4usize..=10,
        blinding_seed in any::<[u8; 32]>(),
        zero_knowledge in any::<bool>(),
    ) {
        let num_rows = 1usize << num_vars;

        let trace = generate_fib_trace(num_vars);
        let expected_pub = trace.get_element(1, num_rows - 1).unwrap().to_tower();
        let instance = ProgramInstance::new(num_rows, vec![expected_pub]);
        let witness = ProgramWitness::new(trace);

        let air = fib_program(num_rows);

        let config = Config {
            zero_knowledge,
            ldt_support_size: 6,
            num_queries: 4,
            min_security_bits: 0,
            ..Config::default()
        };

        let proof = prove(
            b"FibAir_Fuzz",
            &air,
            &instance,
            &witness,
            &config,
            blinding_seed,
            None,
        )
        .unwrap();

        let mut verifier_transcript = Transcript::<H>::new(b"FibAir_Fuzz");
        let ok = HekateVerifier::<F, H>::verify(
        &program_id(&air).unwrap(),
        &air,
        &instance,
        &proof,
        &mut verifier_transcript,
        &config,
    )
            .unwrap();

        prop_assert!(ok, "Program verification failed for num_vars={num_vars}");
    }
}
