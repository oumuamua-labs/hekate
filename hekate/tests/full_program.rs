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
use hekate::math::{Block128, TowerField};
use hekate_core::trace::{IntoTraceColumn, Trace};
use hekate_math::{Bit, Block32};
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::constraint::{BoundaryConstraint, ConstraintAst};
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;
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

#[derive(Clone)]
struct FibProgram {
    num_cols: usize,
    num_rows: usize,
}

impl Air<F> for FibProgram {
    fn num_columns(&self) -> usize {
        self.num_cols
    }

    fn boundary_constraints(&self) -> Vec<BoundaryConstraint<F>> {
        // Enforce:
        // b[last_row] == public_inputs[0].
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

impl Program<F> for FibProgram {
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
fn air_fib_e2e() {
    // init_tracing();

    let num_vars = 8;
    let num_rows = 1 << num_vars;
    let seed = [0xAAu8; 32];

    let trace = generate_fib_trace(num_vars);
    let expected_pub = trace.get_element(1, num_rows - 1).unwrap().to_tower();
    let instance = ProgramInstance::new(num_rows, vec![expected_pub]);
    let witness = ProgramWitness::new(trace);

    let air = FibProgram {
        num_cols: 3,
        num_rows,
    };

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
    let result =
        HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut verifier_transcript, &config);

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

    let air = FibProgram {
        num_cols: 3,
        num_rows,
    };

    let config_a = Config {
        matrix_seed: [1u8; 32],
        num_queries: 8,
        min_security_bits: 0,
        ..Config::default()
    };

    let config_b = Config {
        matrix_seed: [2u8; 32],
        num_queries: 8,
        min_security_bits: 0,
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
        proof_a.trace_commitment.root, proof_b.trace_commitment.root,
        "Sanity: different configs must yield different trace roots"
    );

    let alpha_a = {
        let mut t = Transcript::<H>::new(b"BindingTest");
        t.append_message(b"trace_root", &proof_a.trace_commitment.root);
        t.challenge_field::<F>(b"alpha").unwrap()
    };

    let alpha_b = {
        let mut t = Transcript::<H>::new(b"BindingTest");
        t.append_message(b"trace_root", &proof_b.trace_commitment.root);
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

    let air = FibProgram {
        num_cols: 3,
        num_rows,
    };

    let mut config = test_config();
    config.sumcheck_blinding_factor = 2;

    OsRng.try_fill_bytes(&mut config.matrix_seed).unwrap();

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
    let ok =
        HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut verifier_transcript, &config)
            .unwrap();

    assert!(ok, "ZK Program verification failed");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn fuzz_air_completeness(
        num_vars in 4usize..=10,
        matrix_seed in any::<[u8; 32]>(),
        blinding_seed in any::<[u8; 32]>(),
        sumcheck_blinding_factor in 0usize..=2,
    ) {
        let num_rows = 1usize << num_vars;

        let trace = generate_fib_trace(num_vars);
        let expected_pub = trace.get_element(1, num_rows - 1).unwrap().to_tower();
        let instance = ProgramInstance::new(num_rows, vec![expected_pub]);
        let witness = ProgramWitness::new(trace);

        let air = FibProgram {
            num_cols: 3,
            num_rows,
        };

        let config = Config {
            matrix_seed,
            sumcheck_blinding_factor,
            ldt_blinding_factor: 6,
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
        let ok = HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut verifier_transcript, &config)
            .unwrap();

        prop_assert!(ok, "Program verification failed for num_vars={num_vars}");
    }
}
