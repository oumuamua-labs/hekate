// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! End-to-end prover + verifier coverage of
//! chiplet `BoundaryConstraint::with_constant`.

use hekate::core::config::Config;
use hekate::core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::{Bit, Block128, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::constraint::{BoundaryConstraint, BoundaryTarget, ConstraintAst};
use hekate_program::digest::program_id;
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness, define_columns};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;
use rand::{TryRngCore, rngs::OsRng};

type F = Block128;
type H = DefaultHasher;

fn test_config() -> Config {
    Config {
        num_queries: 8,
        min_security_bits: 0,
        fold_arity_log: 1,
        early_stop_vars: 1,
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
    let pinned_id = program_id(verifier_air).unwrap();

    HekateVerifier::<F, H>::verify(
        &pinned_id,
        verifier_air,
        instance,
        &proof,
        &mut verifier_t,
        &cfg,
    )
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

fn single_bnd_host(num_rows: usize, pinned_value: F) -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("SingleBndHost", num_rows).unwrap();

    cx.column(ColumnType::Bit);
    cx.attach(ChipletDef::from_air(&SingleBndChiplet { pinned_value }).unwrap());

    cx.compile().unwrap()
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
    let num_vars = 5;
    let num_rows = 1 << num_vars;

    let air = single_bnd_host(num_rows, F::ONE);
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
    let num_vars = 5;
    let num_rows = 1 << num_vars;

    let air = single_bnd_host(num_rows, F::ZERO);
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
    let num_vars = 5;
    let num_rows = 1 << num_vars;

    let air = single_bnd_host(num_rows, F::ONE);

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
    let num_vars = 5;
    let num_rows = 1 << num_vars;

    let prover_air = single_bnd_host(num_rows, F::ONE);
    let verifier_air = single_bnd_host(num_rows, F::ZERO);

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

    fn boundary_constraints(&self) -> Vec<BoundaryConstraint<F>> {
        vec![
            BoundaryConstraint::with_constant(MultiBndCols::FLAG_FIRST, 0, F::ONE),
            BoundaryConstraint::with_constant(MultiBndCols::FLAG_LAST, self.num_rows - 1, F::ONE),
        ]
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(MultiBndCols::build_layout)
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
    }
}

fn multi_bnd_host(num_rows: usize) -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("MultiBndHost", num_rows).unwrap();

    cx.column(ColumnType::Bit);
    cx.attach(ChipletDef::from_air(&MultiBndChiplet { num_rows }).unwrap());

    cx.compile().unwrap()
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
    let num_vars = 5;
    let num_rows = 1 << num_vars;

    let air = multi_bnd_host(num_rows);
    let (main, chiplet) = build_multi_traces(num_vars, Bit::ONE, Bit::ONE);

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(main).with_chiplets(vec![chiplet]);

    assert!(run_prove_verify(&air, &instance, &witness));
}

#[test]
fn chiplet_multiple_boundaries_first_violated() {
    let num_vars = 5;
    let num_rows = 1 << num_vars;

    let air = multi_bnd_host(num_rows);
    let (main, chiplet) = build_multi_traces(num_vars, Bit::ZERO, Bit::ONE);

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(main).with_chiplets(vec![chiplet]);

    assert!(!run_prove_verify(&air, &instance, &witness));
}

#[test]
fn chiplet_multiple_boundaries_last_violated() {
    let num_vars = 5;
    let num_rows = 1 << num_vars;

    let air = multi_bnd_host(num_rows);
    let (main, chiplet) = build_multi_traces(num_vars, Bit::ONE, Bit::ZERO);

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(main).with_chiplets(vec![chiplet]);

    assert!(!run_prove_verify(&air, &instance, &witness));
}

#[test]
fn chiplet_multiple_boundaries_both_violated() {
    let num_vars = 5;
    let num_rows = 1 << num_vars;

    let air = multi_bnd_host(num_rows);
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
            vec![BoundaryConstraint {
                col_idx: 0,
                row_idx: 0,
                target: BoundaryTarget::PublicInput(0),
            }]
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
