// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::trace::{ColumnTrace, ColumnType, TraceColumn};
use hekate_math::{Bit, Block32, Block128, Flat, TowerField};
use hekate_program::FixedShape;
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::{ProgramInstance, ProgramWitness};
use hekate_scribble::{MutationKind, ScribbleConfig, assert_all_caught};

const NUM_VARS: usize = 3;
const N: usize = 1 << NUM_VARS;
const X: usize = 0;

type F = Block128;

fn odd_gated_fixture() -> (
    CircuitProgram<F>,
    ProgramInstance<F>,
    ProgramWitness<F, ColumnTrace>,
) {
    let mut cx = Circuit::<F>::new("OddGated", N).unwrap();

    cx.column(ColumnType::B32);

    let odd = cx.column(ColumnType::Bit);

    cx.fix(
        odd,
        FixedShape::Cadence {
            stride: 2,
            count: N / 2,
            origin: 0,
            values: vec![F::ZERO, F::ONE],
        },
    );

    let cs = cx.cs();
    let q = cs.col(odd.index());

    cs.constrain(q * cs.next(X));
    cs.constrain(q * cs.col(X));

    let gate: Vec<Bit> = (0..N)
        .map(|r| if r % 2 == 1 { Bit::ONE } else { Bit::ZERO })
        .collect();

    let mut trace = ColumnTrace::new(NUM_VARS).unwrap();
    trace
        .add_column(TraceColumn::B32(vec![Flat::from_raw(Block32(0)); N]))
        .unwrap();
    trace.add_column(TraceColumn::Bit(gate)).unwrap();

    (
        cx.compile().unwrap(),
        ProgramInstance::new(N, vec![]),
        ProgramWitness::new(trace),
    )
}

fn even_row_flips(rows: impl IntoIterator<Item = usize>) -> ScribbleConfig {
    ScribbleConfig::default()
        .mutations([MutationKind::BitFlip])
        .include_cols([X])
        .include_rows(rows)
}

#[test]
fn local_oracle_rechecks_previous_row() {
    let (air, instance, witness) = odd_gated_fixture();

    assert_all_caught(&air, &instance, &witness, even_row_flips([2, 4, 6]));
}

#[test]
fn local_oracle_rechecks_last_row_for_row_zero() {
    let (air, instance, witness) = odd_gated_fixture();

    assert_all_caught(&air, &instance, &witness, even_row_flips([0]));
}
