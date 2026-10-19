// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::trace::{ColumnTrace, ColumnType, TraceColumn};
use hekate_math::{Block32, Block128, Flat};
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::{ProgramInstance, ProgramWitness};
use hekate_scribble::{ScribbleConfig, assert_all_caught, assert_all_caught_all_targets};

const NUM_VARS: usize = 2;
const N: usize = 1 << NUM_VARS;

type F = Block128;

fn fixture(
    row0: u32,
) -> (
    CircuitProgram<F>,
    ProgramInstance<F>,
    ProgramWitness<F, ColumnTrace>,
) {
    let mut cx = Circuit::<F>::new("ZeroColumn", N).unwrap();
    cx.column(ColumnType::B32);

    let cs = cx.cs();
    cs.constrain(cs.col(0));

    let mut data = vec![Flat::from_raw(Block32(0)); N];
    data[0] = Flat::from_raw(Block32(row0));

    let mut trace = ColumnTrace::new(NUM_VARS).unwrap();
    trace.add_column(TraceColumn::B32(data)).unwrap();

    (
        cx.compile().unwrap(),
        ProgramInstance::new(N, vec![]),
        ProgramWitness::new(trace),
    )
}

#[test]
fn clean_witness_runs_cases() {
    let (air, instance, witness) = fixture(0);

    assert_all_caught(&air, &instance, &witness, ScribbleConfig::default());
}

#[test]
#[should_panic(expected = "clean witness fails preflight")]
fn dirty_witness_panics_before_first_case() {
    let (air, instance, witness) = fixture(1);

    assert_all_caught(&air, &instance, &witness, ScribbleConfig::default());
}

#[test]
#[should_panic(expected = "clean witness fails preflight")]
fn dirty_witness_panics_before_any_target() {
    let (air, instance, witness) = fixture(1);

    assert_all_caught_all_targets(&air, &instance, &witness, ScribbleConfig::default());
}
