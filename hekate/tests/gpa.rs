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

//! End-to-end LogUp bus tests against
//! the high-level prover/verifier.

use hekate::core::config::Config;
use hekate::core::trace::{ColumnTrace, ColumnType, IntoTraceColumn, TraceColumn};
use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::{Bit, Block32, Block128, Flat};
use hekate_math::TowerField;
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::permutation::{PermutationCheckSpec, Source};
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;

type F = Block128;
type H = DefaultHasher;

/// Intra-table LogUp bus:
/// two specs on the same `bus_id`, one reading
/// column 0, the other reading column 1. Endpoint
/// sums cancel iff the two columns carry the same
/// multiset under the shared selector.
#[derive(Clone)]
struct SameBusPermutationAir {
    num_cols: usize,
}

impl Air<F> for SameBusPermutationAir {
    fn num_columns(&self) -> usize {
        self.num_cols
    }

    fn column_layout(&self) -> &'static [ColumnType] {
        &[ColumnType::B32, ColumnType::B32, ColumnType::Bit]
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        let waiver = "see hekate/tests/gpa.rs: synthetic intra-table reverse-permutation \
                      test, both endpoints positional on same trace; not a production \
                      bus shape, exercises LogUp algebra only";

        let spec_a =
            PermutationCheckSpec::new(vec![(Source::Column(0), b"kappa_data" as &[u8])], Some(2))
                .with_clock_waiver(waiver);

        let spec_b =
            PermutationCheckSpec::new(vec![(Source::Column(1), b"kappa_data" as &[u8])], Some(2))
                .with_clock_waiver(waiver);

        vec![
            ("same_bus".to_string(), spec_a),
            ("same_bus".to_string(), spec_b),
        ]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(2));

        cs.build()
    }
}

impl Program<F> for SameBusPermutationAir {}

fn generate_permutation_trace(num_vars: usize) -> ColumnTrace {
    let num_rows = 1 << num_vars;

    let mut col_a: Vec<Block32> = Vec::with_capacity(num_rows);
    let mut col_b: Vec<Block32> = Vec::with_capacity(num_rows);
    let mut selector: Vec<Bit> = Vec::with_capacity(num_rows);

    // Col A:
    // [0, 1, ..., N-1]
    // Col B:
    // [N-1, ..., 1, 0]  (permutation of A)
    for i in 0..num_rows {
        col_a.push(Block32::from(i as u32));
        col_b.push(Block32::from((num_rows - 1 - i) as u32));
        selector.push(Bit::ONE);
    }

    let mut trace = ColumnTrace::new(num_vars).unwrap();
    trace.add_column(col_a.into_trace_column()).unwrap();
    trace.add_column(col_b.into_trace_column()).unwrap();
    trace.add_column(TraceColumn::Bit(selector)).unwrap();

    trace
}

/// Happy path:
/// identical multisets -> paired
/// LogUp sums cancel -> verify passes.
#[test]
fn logup_bus_happy_path() {
    let num_vars = 8;
    let num_rows = 1 << num_vars;
    let seed = [0xAAu8; 32];

    let air = SameBusPermutationAir { num_cols: 3 };
    let trace = generate_permutation_trace(num_vars);
    let witness = ProgramWitness::new(trace);
    let instance = ProgramInstance::new(num_rows, vec![]);

    let config = Config {
        num_queries: 4,
        min_security_bits: 0,
        sumcheck_blinding_factor: 2,
        ..Config::default()
    };

    let proof = prove(
        b"LogUp_Happy",
        &air,
        &instance,
        &witness,
        &config,
        seed,
        None,
    )
    .expect("Proving failed");

    assert_eq!(proof.main_logup_aux.claimed_sums.len(), 2);

    let mut verifier_transcript = Transcript::<H>::new(b"LogUp_Happy");
    let ok =
        HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut verifier_transcript, &config)
            .expect("Verifier returned error");

    assert!(ok, "Verifier rejected a valid permutation proof");
}

/// Divergent multisets -> paired LogUp
/// sums do NOT cancel → verifier rejects.
#[test]
fn logup_bus_divergence_rejected() {
    let num_vars = 4;
    let num_rows = 1 << num_vars;
    let seed = [0xAAu8; 32];

    let air = SameBusPermutationAir { num_cols: 3 };

    // Corrupt column B so its
    // multiset differs from column A.
    let mut trace = generate_permutation_trace(num_vars);
    if let TraceColumn::B32(ref mut vals) = trace.columns[1] {
        vals[num_rows - 1] = Flat::from_raw(Block32::from(999u32));
    }

    let witness = ProgramWitness::new(trace);
    let instance = ProgramInstance::new(num_rows, vec![]);
    let config = Config::default();

    let proof = prove(
        b"LogUp_Divergence",
        &air,
        &instance,
        &witness,
        &config,
        seed,
        None,
    )
    .expect("Prover generates an honest proof of a non-permutation");

    let mut verifier_transcript = Transcript::<H>::new(b"LogUp_Divergence");
    let result =
        HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut verifier_transcript, &config);

    assert!(
        result.is_err(),
        "Verifier must reject a non-permutation (bus sums do not cancel)",
    );
}
