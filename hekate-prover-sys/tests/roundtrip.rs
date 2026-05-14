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

use hekate_core::config::Config;
use hekate_core::trace::{ColumnTrace, ColumnType, Trace, TraceBuilder};
use hekate_crypto::transcript::Transcript;
use hekate_crypto::DefaultHasher;
use hekate_math::{Bit, Block128, Block32, TowerField};
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::constraint::{BoundaryConstraint, ConstraintAst};
use hekate_program::define_columns;
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::{prove, CancelToken, Error, ErrorCode};
use hekate_verifier::HekateVerifier;

type F = Block128;
type H = DefaultHasher;

define_columns! {
    FibCols {
        A: B32,
        B: B32,
        Q: Bit,
    }
}

#[derive(Clone)]
struct FibAir {
    num_rows: usize,
    layout: Vec<ColumnType>,
}

impl FibAir {
    fn new(num_rows: usize) -> Self {
        Self {
            num_rows,
            layout: FibCols::build_layout(),
        }
    }
}

impl Air<F> for FibAir {
    fn num_columns(&self) -> usize {
        FibCols::NUM_COLUMNS
    }

    fn boundary_constraints(&self) -> Vec<BoundaryConstraint<F>> {
        vec![BoundaryConstraint::with_public_input(
            FibCols::B,
            self.num_rows - 1,
            0,
        )]
    }

    fn column_layout(&self) -> &[ColumnType] {
        &self.layout
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();

        let [a, b, q] = [cs.col(FibCols::A), cs.col(FibCols::B), cs.col(FibCols::Q)];
        let [na, nb] = [cs.next(FibCols::A), cs.next(FibCols::B)];

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

fn build_fib_trace(num_rows: usize) -> ColumnTrace {
    let num_vars = num_rows.trailing_zeros() as usize;
    let mut tb = TraceBuilder::new(&FibCols::build_layout(), num_vars).expect("trace builder");

    let (mut a, mut b) = (Block32::ZERO, Block32::ONE);

    for i in 0..num_rows {
        tb.set_b32(FibCols::A, i, a).unwrap();
        tb.set_b32(FibCols::B, i, b).unwrap();
        tb.set_bit(
            FibCols::Q,
            i,
            if i < num_rows - 1 {
                Bit::ONE
            } else {
                Bit::ZERO
            },
        )
        .unwrap();

        let tmp = a + b;
        a = b;
        b = tmp;
    }

    tb.build()
}

fn fib_setup(num_vars: usize) -> (FibAir, ProgramInstance<F>, ProgramWitness<F>, Config) {
    let num_rows = 1 << num_vars;
    let trace = build_fib_trace(num_rows);

    let pub_input = trace
        .get_element::<F>(FibCols::B, num_rows - 1)
        .unwrap()
        .to_tower();

    let air = FibAir::new(num_rows);
    let instance = ProgramInstance::new(num_rows, vec![pub_input]);
    let witness = ProgramWitness::<F>::new(trace);
    let config = Config {
        num_queries: 8,
        min_security_bits: 0,
        ..Config::default()
    };

    (air, instance, witness, config)
}

#[test]
fn shim_prove_verifies_against_hekate_verifier() {
    let (air, instance, witness, config) = fib_setup(8);
    let label: &[u8] = b"hekate-prover-sys-roundtrip";
    let seed = [0xA5u8; 32];

    let proof = prove(label, &air, &instance, &witness, &config, seed, None).expect("prove");

    let mut t = Transcript::<H>::new(label);
    let ok =
        HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut t, &config).expect("verify");

    assert!(ok, "verifier rejected shim-produced proof");
}

#[test]
fn shim_cancel_pre_set_returns_cancelled() {
    let (air, instance, witness, config) = fib_setup(8);
    let label: &[u8] = b"hekate-prover-sys-cancel";
    let seed = [0u8; 32];

    let cancel = CancelToken::new();
    cancel.request();

    let err = prove(
        label,
        &air,
        &instance,
        &witness,
        &config,
        seed,
        Some(&cancel),
    )
    .expect_err("prove must fail when cancel is pre-set");

    match err {
        Error::Ffi { code, .. } => assert_eq!(code, ErrorCode::Cancelled, "expected Cancelled"),
        other => panic!("expected Ffi(Cancelled); got {other:?}"),
    }
}

#[test]
fn shim_version_is_non_empty() {
    let v = hekate_prover_sys::version();
    assert!(!v.is_empty(), "version string should not be empty");
}
