// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::errors;
use hekate_core::trace::{self, ColumnTrace, ColumnType, TraceColumn};
use hekate_math::{Bit, Block8, Block32, Block128, Flat, TowerField};
use hekate_program::FixedShape;
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_sdk::preflight::{PreflightReport, TableId, check_air_constraints, check_air_rows};

const NUM_VARS: usize = 3;
const N: usize = 1 << NUM_VARS;
const PINNED_ROWS: usize = N / 2;

const X: usize = 0;
const PIN: usize = 1;

type F = Block128;

type Violations = (
    Vec<(usize, usize, Flat<F>)>,
    Vec<(usize, usize, Flat<F>, Flat<F>)>,
);

fn constant_column_program() -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("ConstantColumn", N).unwrap();

    cx.column(ColumnType::B32);

    let pin = cx.column(ColumnType::Bit);

    cx.fix(
        pin,
        FixedShape::Cadence {
            stride: 1,
            count: PINNED_ROWS,
            origin: 0,
            values: vec![F::ONE],
        },
    );

    let cs = cx.cs();

    cs.constrain(cs.next(X) + cs.col(X));

    cx.compile().unwrap()
}

fn trace_with(tamper: impl FnOnce(&mut ColumnTrace)) -> ColumnTrace {
    let x = vec![Flat::from_raw(Block32(5)); N];
    let pin = (0..N)
        .map(|r| if r < PINNED_ROWS { Bit::ONE } else { Bit::ZERO })
        .collect();

    let mut trace = ColumnTrace::new(NUM_VARS).unwrap();
    trace.add_column(TraceColumn::B32(x)).unwrap();
    trace.add_column(TraceColumn::Bit(pin)).unwrap();

    tamper(&mut trace);

    trace
}

fn set_x(trace: &mut ColumnTrace, row: usize, value: u32) {
    let TraceColumn::B32(x) = &mut trace.columns[X] else {
        panic!("X is a B32 column");
    };

    x[row] = Flat::from_raw(Block32(value));
}

fn set_pin(trace: &mut ColumnTrace, row: usize, value: Bit) {
    let TraceColumn::Bit(pin) = &mut trace.columns[PIN] else {
        panic!("PIN is a Bit column");
    };

    pin[row] = value;
}

fn full_check(program: &CircuitProgram<F>, trace: &ColumnTrace) -> Violations {
    let mut report = PreflightReport::new();
    check_air_constraints(program, trace, TableId::Main, &mut report).unwrap();

    violations(&report)
}

fn row_check(
    program: &CircuitProgram<F>,
    trace: &ColumnTrace,
    rows: &[usize],
) -> errors::Result<Violations> {
    let mut report = PreflightReport::new();
    check_air_rows(program, trace, TableId::Main, rows, &mut report)?;

    Ok(violations(&report))
}

fn violations(report: &PreflightReport<F>) -> Violations {
    (
        report
            .constraint_violations
            .iter()
            .map(|v| (v.constraint_idx, v.row_idx, v.value))
            .collect(),
        report
            .fixed_column_violations
            .iter()
            .map(|v| (v.pin_idx, v.row_idx, v.actual, v.expected))
            .collect(),
    )
}

fn packed_program() -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("PackedByte", N).unwrap();

    cx.expand_bits(1, ColumnType::B8);

    let cs = cx.cs();

    cs.constrain(cs.col(0) * (cs.one() + cs.col(0)));

    cx.compile().unwrap()
}

fn packed_trace() -> ColumnTrace {
    let mut trace = ColumnTrace::new(NUM_VARS).unwrap();
    trace
        .add_column(TraceColumn::B8(vec![Flat::from_raw(Block8(0)); N]))
        .unwrap();

    trace
}

#[test]
fn row_check_matches_full_check_at_each_row() {
    let program = constant_column_program();
    let trace = trace_with(|t| {
        set_x(t, 3, 9);
        set_pin(t, 6, Bit::ONE);
    });

    let full = full_check(&program, &trace);

    assert_eq!(full.0.iter().map(|v| v.1).collect::<Vec<_>>(), [2, 3]);
    assert_eq!(full.1.iter().map(|v| v.1).collect::<Vec<_>>(), [6]);

    for row in 0..N {
        let at_row: Violations = (
            full.0.iter().filter(|v| v.1 == row).copied().collect(),
            full.1.iter().filter(|v| v.1 == row).copied().collect(),
        );

        assert_eq!(row_check(&program, &trace, &[row]).unwrap(), at_row);
    }
}

#[test]
fn row_check_reads_row_zero_from_last_row() {
    let program = constant_column_program();
    let trace = trace_with(|t| set_x(t, 0, 9));

    let (constraints, pins) = row_check(&program, &trace, &[N - 1]).unwrap();

    assert_eq!(constraints.iter().map(|v| v.1).collect::<Vec<_>>(), [N - 1]);
    assert!(pins.is_empty());
}

#[test]
fn row_check_over_every_row_matches_full_check() {
    let program = constant_column_program();
    let trace = trace_with(|t| {
        set_x(t, 0, 9);
        set_x(t, 5, 7);
        set_pin(t, 1, Bit::ZERO);
    });

    let rows: Vec<usize> = (0..N).rev().chain(0..N).collect();

    assert_eq!(
        row_check(&program, &trace, &rows).unwrap(),
        full_check(&program, &trace)
    );
}

#[test]
fn row_check_rejects_row_past_trace_height() {
    let program = packed_program();
    let trace = packed_trace();

    let err = row_check(&program, &trace, &[1, N]).unwrap_err();

    assert!(matches!(
        err,
        errors::Error::Trace(trace::Error::RowIndexOutOfBounds {
            row_idx: N,
            num_rows: N,
        })
    ));
}
