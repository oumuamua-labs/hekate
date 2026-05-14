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

//! End-to-end prover + verifier coverage of
//! chiplet `BoundaryConstraint::with_constant`.

use hekate::core::config::Config;
use hekate::core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate::crypto::transcript::Transcript;
use hekate::crypto::DefaultHasher;
use hekate::math::{Bit, Block128, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::constraint::{BoundaryConstraint, ConstraintAst};
use hekate_program::{define_columns, Air, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;
use rand::{rngs::OsRng, TryRngCore};

type F = Block128;
type H = DefaultHasher;

fn test_config() -> Config {
    Config {
        num_queries: 8,
        min_security_bits: 0,
        ..Config::default()
    }
}

fn run_prove_verify<P: Program<F>>(
    program: &P,
    instance: &ProgramInstance<F>,
    witness: &ProgramWitness<F>,
) -> bool {
    run_split_air(program, program, instance, witness)
}

fn run_split_air<PProver, PVerifier>(
    prover_air: &PProver,
    verifier_air: &PVerifier,
    instance: &ProgramInstance<F>,
    witness: &ProgramWitness<F>,
) -> bool
where
    PProver: Program<F>,
    PVerifier: Program<F>,
{
    let mut seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut seed).unwrap();

    let cfg = test_config();
    let domain = b"chiplet_boundary";

    let proof =
        prove(domain, prover_air, instance, witness, &cfg, seed, None).expect("prove failed");

    let mut verifier_t = Transcript::<H>::new(domain);

    HekateVerifier::<F, H>::verify(verifier_air, instance, &proof, &mut verifier_t, &cfg)
        .unwrap_or(false)
}

// =================================================================
// Single-boundary chiplet
// =================================================================

define_columns! {
    SingleBndCols {
        FLAG: Bit,
    }
}

#[derive(Clone)]
struct SingleBndChiplet {
    pinned_value: F,
}

impl Air<F> for SingleBndChiplet {
    fn num_columns(&self) -> usize {
        SingleBndCols::NUM_COLUMNS
    }

    fn boundary_constraints(&self) -> Vec<BoundaryConstraint<F>> {
        vec![BoundaryConstraint::with_constant(
            SingleBndCols::FLAG,
            0,
            self.pinned_value,
        )]
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(SingleBndCols::build_layout)
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
    }
}

#[derive(Clone)]
struct SingleBndHost {
    pinned_value: F,
}

impl Air<F> for SingleBndHost {
    fn num_columns(&self) -> usize {
        1
    }

    fn column_layout(&self) -> &[ColumnType] {
        &[ColumnType::Bit]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
    }
}

impl Program<F> for SingleBndHost {
    fn chiplet_defs(&self) -> hekate_core::errors::Result<Vec<ChipletDef<F>>> {
        Ok(vec![ChipletDef::from_air(&SingleBndChiplet {
            pinned_value: self.pinned_value,
        })?])
    }
}

fn build_traces(num_vars: usize, row0_flag: Bit) -> (ColumnTrace, ColumnTrace) {
    let main_layout: Vec<ColumnType> = vec![ColumnType::Bit];
    let main = TraceBuilder::new(&main_layout, num_vars).unwrap().build();

    let layout = SingleBndCols::build_layout();

    let mut tb = TraceBuilder::new(&layout, num_vars).unwrap();
    tb.set_bit(SingleBndCols::FLAG, 0, row0_flag).unwrap();

    let chiplet = tb.build();

    (main, chiplet)
}

#[test]
fn chiplet_boundary_happy_path() {
    let num_vars = 3;
    let num_rows = 1 << num_vars;

    let air = SingleBndHost {
        pinned_value: F::ONE,
    };
    let (main, chiplet) = build_traces(num_vars, Bit::ONE);

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(main).with_chiplets(vec![chiplet]);

    assert!(
        run_prove_verify(&air, &instance, &witness),
        "honest trace must verify"
    );
}

#[test]
fn chiplet_boundary_pin_zero_happy_path() {
    let num_vars = 3;
    let num_rows = 1 << num_vars;

    let air = SingleBndHost {
        pinned_value: F::ZERO,
    };

    let (main, chiplet) = build_traces(num_vars, Bit::ZERO);

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(main).with_chiplets(vec![chiplet]);

    assert!(
        run_prove_verify(&air, &instance, &witness),
        "pin-zero trace must verify"
    );
}

#[test]
fn chiplet_boundary_violation_rejected() {
    let num_vars = 3;
    let num_rows = 1 << num_vars;

    let air = SingleBndHost {
        pinned_value: F::ONE,
    };

    let (main, chiplet) = build_traces(num_vars, Bit::ZERO);

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(main).with_chiplets(vec![chiplet]);

    assert!(
        !run_prove_verify(&air, &instance, &witness),
        "trace violating chiplet boundary must reject"
    );
}

#[test]
fn chiplet_boundary_constant_swap_rejected() {
    let num_vars = 3;
    let num_rows = 1 << num_vars;

    let prover_air = SingleBndHost {
        pinned_value: F::ONE,
    };
    let verifier_air = SingleBndHost {
        pinned_value: F::ZERO,
    };

    let (main, chiplet) = build_traces(num_vars, Bit::ONE);

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(main).with_chiplets(vec![chiplet]);

    assert!(
        !run_split_air(&prover_air, &verifier_air, &instance, &witness),
        "constant swap between prover and verifier must reject (transcript binding)"
    );
}

// =================================================================
// Multi-boundary chiplet
// =================================================================

define_columns! {
    MultiBndCols {
        FLAG_FIRST: Bit,
        FLAG_LAST: Bit,
    }
}

#[derive(Clone)]
struct MultiBndChiplet {
    num_rows: usize,
}

impl Air<F> for MultiBndChiplet {
    fn num_columns(&self) -> usize {
        MultiBndCols::NUM_COLUMNS
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(MultiBndCols::build_layout)
    }

    fn boundary_constraints(&self) -> Vec<BoundaryConstraint<F>> {
        vec![
            BoundaryConstraint::with_constant(MultiBndCols::FLAG_FIRST, 0, F::ONE),
            BoundaryConstraint::with_constant(MultiBndCols::FLAG_LAST, self.num_rows - 1, F::ONE),
        ]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
    }
}

#[derive(Clone)]
struct MultiBndHost {
    num_rows: usize,
}

impl Air<F> for MultiBndHost {
    fn num_columns(&self) -> usize {
        1
    }

    fn column_layout(&self) -> &[ColumnType] {
        &[ColumnType::Bit]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
    }
}

impl Program<F> for MultiBndHost {
    fn chiplet_defs(&self) -> hekate_core::errors::Result<Vec<ChipletDef<F>>> {
        Ok(vec![ChipletDef::from_air(&MultiBndChiplet {
            num_rows: self.num_rows,
        })?])
    }
}

fn build_multi_traces(
    num_vars: usize,
    first_val: Bit,
    last_val: Bit,
) -> (ColumnTrace, ColumnTrace) {
    let num_rows = 1 << num_vars;

    let main_layout: Vec<ColumnType> = vec![ColumnType::Bit];
    let main = TraceBuilder::new(&main_layout, num_vars).unwrap().build();

    let layout = MultiBndCols::build_layout();

    let mut tb = TraceBuilder::new(&layout, num_vars).unwrap();

    tb.set_bit(MultiBndCols::FLAG_FIRST, 0, first_val).unwrap();
    tb.set_bit(MultiBndCols::FLAG_LAST, num_rows - 1, last_val)
        .unwrap();

    let chiplet = tb.build();

    (main, chiplet)
}

#[test]
fn chiplet_multiple_boundaries_all_satisfied() {
    let num_vars = 3;
    let num_rows = 1 << num_vars;

    let air = MultiBndHost { num_rows };

    let (main, chiplet) = build_multi_traces(num_vars, Bit::ONE, Bit::ONE);

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(main).with_chiplets(vec![chiplet]);

    assert!(run_prove_verify(&air, &instance, &witness));
}

#[test]
fn chiplet_multiple_boundaries_first_violated() {
    let num_vars = 3;
    let num_rows = 1 << num_vars;

    let air = MultiBndHost { num_rows };

    let (main, chiplet) = build_multi_traces(num_vars, Bit::ZERO, Bit::ONE);

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(main).with_chiplets(vec![chiplet]);

    assert!(!run_prove_verify(&air, &instance, &witness));
}

#[test]
fn chiplet_multiple_boundaries_last_violated() {
    let num_vars = 3;
    let num_rows = 1 << num_vars;

    let air = MultiBndHost { num_rows };

    let (main, chiplet) = build_multi_traces(num_vars, Bit::ONE, Bit::ZERO);

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(main).with_chiplets(vec![chiplet]);

    assert!(!run_prove_verify(&air, &instance, &witness));
}

#[test]
fn chiplet_multiple_boundaries_both_violated() {
    let num_vars = 3;
    let num_rows = 1 << num_vars;

    let air = MultiBndHost { num_rows };

    let (main, chiplet) = build_multi_traces(num_vars, Bit::ZERO, Bit::ZERO);

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(main).with_chiplets(vec![chiplet]);

    assert!(!run_prove_verify(&air, &instance, &witness));
}

// =================================================================
// Chiplet with PublicInput target rejected at snapshot
// =================================================================

#[test]
fn chiplet_with_public_input_target_rejected_at_snapshot() {
    #[derive(Clone)]
    struct PubInputChiplet;

    impl Air<F> for PubInputChiplet {
        fn num_columns(&self) -> usize {
            1
        }

        fn boundary_constraints(&self) -> Vec<BoundaryConstraint<F>> {
            vec![BoundaryConstraint::with_public_input(0, 0, 0)]
        }

        fn column_layout(&self) -> &[ColumnType] {
            &[ColumnType::Bit]
        }

        fn constraint_ast(&self) -> ConstraintAst<F> {
            ConstraintSystem::<F>::new().build()
        }
    }

    let result = ChipletDef::<F>::from_air(&PubInputChiplet);
    assert!(matches!(
        result,
        Err(hekate_core::errors::Error::Protocol {
            protocol: "boundary",
            ..
        })
    ));
}
