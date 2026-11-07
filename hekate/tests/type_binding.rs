// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate::core::config::Config;
use hekate::core::trace::{ColumnTrace, ColumnType, TraceColumn};
use hekate::math::{Block128, Block256, TowerField};
use hekate_core::proofs::InnerProof;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::constraint::{BoundaryConstraint, ConstraintAst};
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_testkit::site::{EVAL_SUMCHECK, OUTER_LINEAR};
use hekate_testkit::verdict::{rejected_at, verify_logged};

type F = Block256;

const SKIP_VARS: usize = 9;
const HEIGHTS: [usize; 2] = [6, SKIP_VARS];
const LABEL: &[u8] = b"TypeBinding";
const VALUE: usize = 0;
const NARROW: usize = 1;

const NARROW_TYPES: [(ColumnType, u32); 5] = [
    (ColumnType::Bit, 1),
    (ColumnType::B8, 8),
    (ColumnType::B16, 16),
    (ColumnType::B32, 32),
    (ColumnType::B64, 64),
];

const WIDE_TYPES: [(ColumnType, u32); 3] = [
    (ColumnType::B32, 32),
    (ColumnType::B64, 64),
    (ColumnType::B128, 128),
];

/// Proves the public input fits the narrow column's type.
/// A constraint ranging that column would void the guard.
#[derive(Clone)]
struct RangeByType {
    layout: [ColumnType; 2],
}

impl Air<F> for RangeByType {
    fn boundary_constraints(&self) -> Vec<BoundaryConstraint<F>> {
        vec![BoundaryConstraint::with_public_input(VALUE, 0, 0)]
    }

    fn column_layout(&self) -> &[ColumnType] {
        &self.layout
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.constrain(cs.col(VALUE) + cs.col(NARROW));

        cs.build()
    }
}

impl Program<F> for RangeByType {
    fn num_public_inputs(&self) -> usize {
        1
    }
}

fn config(zero_knowledge: bool) -> Config {
    Config {
        fold_arity_log: 2,
        early_stop_vars: 2,
        zero_knowledge,
        ..Config::dev()
    }
}

fn instance(vars: usize, value: u128) -> ProgramInstance<F> {
    ProgramInstance::new(1 << vars, vec![F::from(value)])
}

fn prove_with(
    air: &RangeByType,
    config: &Config,
    vars: usize,
    value: u128,
    storage: ColumnType,
) -> InnerProof<F> {
    let mut cells = vec![Block128::ZERO; 1 << vars];
    cells[0] = Block128::from(value);

    let mut trace = ColumnTrace::new(vars).unwrap();
    trace
        .add_column(TraceColumn::from_data(cells.clone(), air.layout[VALUE]))
        .unwrap();
    trace
        .add_column(TraceColumn::from_data(cells, storage))
        .unwrap();

    prove(
        LABEL,
        air,
        &instance(vars, value),
        &ProgramWitness::new(trace),
        config,
        [0x94; 32],
        None,
    )
    .unwrap()
}

#[test]
fn whole_read_cell_outside_type_rejected() {
    let mut escaped = Vec::new();
    for (vars, zero_knowledge) in HEIGHTS.into_iter().flat_map(|v| [(v, false), (v, true)]) {
        let config = config(zero_knowledge);
        let site = match zero_knowledge {
            false => EVAL_SUMCHECK,
            true => OUTER_LINEAR,
        };

        for (wide, wide_bits) in WIDE_TYPES {
            for (narrow, bits) in NARROW_TYPES {
                if bits >= wide_bits {
                    continue;
                }

                let air = RangeByType {
                    layout: [wide, narrow],
                };

                let label =
                    format!("{wide:?}, {narrow:?}, 2^{vars}, zero knowledge {zero_knowledge}");

                let fits = (1u128 << bits) - 1;
                let honest = prove_with(&air, &config, vars, fits, narrow);
                let (outcome, _) =
                    verify_logged(&air, &instance(vars, fits), &config, &honest, LABEL);

                assert!(matches!(outcome, Ok(true)), "{label}: {outcome:?}");
                assert_eq!(
                    !honest.zerocheck_proof.skip.is_empty(),
                    vars == SKIP_VARS,
                    "{label}"
                );

                let outside = 1u128 << bits;
                let forged = prove_with(&air, &config, vars, outside, wide);

                let (outcome, log) =
                    verify_logged(&air, &instance(vars, outside), &config, &forged, LABEL);

                if !rejected_at(&outcome, &log, site) {
                    escaped.push((label, log));
                }
            }
        }
    }

    assert!(escaped.is_empty(), "{escaped:?}");
}
