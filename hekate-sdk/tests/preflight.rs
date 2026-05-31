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

use hekate_core::errors;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder, TraceColumn};
use hekate_math::{Bit, Block32, Block128, HardwareField, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::constraint::{BoundaryConstraint, ConstraintAst};
use hekate_program::expander::VirtualExpander;
use hekate_program::permutation::{BusKind, PermutationCheckSpec, REQUEST_IDX_LABEL, Source};
use hekate_program::{Air, LagrangePin, Program, ProgramInstance, ProgramWitness, define_columns};
use hekate_sdk::preflight;
use hekate_sdk::preflight::TableId;

type F = Block128;

// =================================================================
// COLUMN SCHEMAS
// =================================================================

define_columns! {
    FibCols {
        A: B32,
        B: B32,
        Q: Bit,
    }
}

define_columns! {
    BusCols {
        SEL: Bit,
        ADDR: B32,
        VAL: B32,
        REQUEST_IDX: B32,
    }
}

// =================================================================
// MINIMAL AIR PROGRAMS
// =================================================================

#[derive(Clone)]
struct FibAir;

impl Air<F> for FibAir {
    fn num_columns(&self) -> usize {
        FibCols::NUM_COLUMNS
    }

    fn boundary_constraints(&self) -> Vec<BoundaryConstraint<F>> {
        vec![
            BoundaryConstraint::with_public_input(FibCols::A, 0, 0),
            BoundaryConstraint::with_public_input(FibCols::B, 0, 1),
        ]
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(FibCols::build_layout)
    }

    fn lagrange_pinned_columns(&self) -> Vec<LagrangePin> {
        vec![LagrangePin::last_row(FibCols::Q)]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();

        let [a, b, q] = [cs.col(FibCols::A), cs.col(FibCols::B), cs.col(FibCols::Q)];
        let [na, nb] = [cs.next(FibCols::A), cs.next(FibCols::B)];

        cs.constrain_named("fib_a", q * (na + b));
        cs.constrain_named("fib_b", q * (nb + a + b));

        cs.build()
    }
}

impl Program<F> for FibAir {
    fn num_public_inputs(&self) -> usize {
        2
    }
}

fn valid_fib_trace(num_vars: usize) -> ColumnTrace {
    let num_rows = 1 << num_vars;

    let mut tb = TraceBuilder::new(&FibCols::build_layout(), num_vars).unwrap();

    let mut a = Block32::ZERO;
    let mut b = Block32::ONE;

    for i in 0..num_rows {
        tb.set_b32(FibCols::A, i, a).unwrap();
        tb.set_b32(FibCols::B, i, b).unwrap();
        tb.set_bit(
            FibCols::Q,
            i,
            if i == num_rows - 1 {
                Bit::ZERO
            } else {
                Bit::ONE
            },
        )
        .unwrap();

        let tmp = a + b;
        a = b;
        b = tmp;
    }

    tb.build()
}

// CPU + chiplet with shared bus
#[derive(Clone)]
struct CpuWithBus;

impl Air<F> for CpuWithBus {
    fn num_columns(&self) -> usize {
        BusCols::NUM_COLUMNS
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(BusCols::build_layout)
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![(
            "test_bus".to_string(),
            PermutationCheckSpec::new(
                vec![
                    (Source::Column(BusCols::ADDR), b"k0"),
                    (Source::Column(BusCols::VAL), b"k1"),
                    (Source::RowIndexLeBytes(4), REQUEST_IDX_LABEL),
                ],
                Some(BusCols::SEL),
            ),
        )]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
    }
}

impl Program<F> for CpuWithBus {
    fn chiplet_defs(&self) -> errors::Result<Vec<ChipletDef<F>>> {
        Ok(vec![ChipletDef::from_air(&MemChiplet)?])
    }
}

// Chiplet side:
// receives the same (selector, addr, value) tuples
#[derive(Clone)]
struct MemChiplet;

impl Air<F> for MemChiplet {
    fn num_columns(&self) -> usize {
        BusCols::NUM_COLUMNS
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(BusCols::build_layout)
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![(
            "test_bus".to_string(),
            PermutationCheckSpec::new(
                vec![
                    (Source::Column(BusCols::ADDR), b"k0"),
                    (Source::Column(BusCols::VAL), b"k1"),
                    (Source::Column(BusCols::REQUEST_IDX), REQUEST_IDX_LABEL),
                ],
                Some(BusCols::SEL),
            ),
        )]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(BusCols::SEL));

        cs.build()
    }
}

fn matching_bus_traces(num_vars: usize, active: usize) -> (ColumnTrace, ColumnTrace) {
    let layout = BusCols::build_layout();

    let mut cpu_tb = TraceBuilder::new(&layout, num_vars).unwrap();
    let mut mem_tb = TraceBuilder::new(&layout, num_vars).unwrap();

    for i in 0..active {
        let addr = Block32::from(i as u32);
        let val = Block32::from((i * 10) as u32);

        cpu_tb.set_bit(BusCols::SEL, i, Bit::ONE).unwrap();
        cpu_tb.set_b32(BusCols::ADDR, i, addr).unwrap();
        cpu_tb.set_b32(BusCols::VAL, i, val).unwrap();

        mem_tb.set_bit(BusCols::SEL, i, Bit::ONE).unwrap();
        mem_tb.set_b32(BusCols::ADDR, i, addr).unwrap();
        mem_tb.set_b32(BusCols::VAL, i, val).unwrap();
        mem_tb
            .set_b32(BusCols::REQUEST_IDX, i, Block32::from(i as u32))
            .unwrap();
    }

    (cpu_tb.build(), mem_tb.build())
}

// =================================================================
// TESTS
// =================================================================

#[test]
fn clean_trace_passes() {
    let trace = valid_fib_trace(3);
    let instance = ProgramInstance::new(8, vec![F::ZERO, F::ONE]);
    let witness = ProgramWitness::<F>::new(trace);

    let report = preflight(&FibAir, &instance, &witness).unwrap();
    eprintln!("{}", report);

    assert!(report.is_clean());
}

#[test]
fn detects_air_constraint_violation() {
    let num_vars = 3;
    let mut trace = valid_fib_trace(num_vars);

    // Corrupt row 2,
    // col A breaks fib_a at row 1.
    if let TraceColumn::B32(ref mut v) = trace.columns[FibCols::A] {
        v[2] = Block32::from(0xDEADu32).to_hardware();
    }

    let instance = ProgramInstance::new(8, vec![F::ZERO, F::ONE]);
    let witness = ProgramWitness::<F>::new(trace);

    let report = preflight(&FibAir, &instance, &witness).unwrap();
    eprintln!("{}", report);

    assert!(!report.is_clean());
    assert!(!report.constraint_violations.is_empty());

    let first = &report.constraint_violations[0];
    assert_eq!(first.label, Some("fib_a"));
    assert_eq!(first.row_idx, 1);
}

#[test]
fn detects_boundary_violation() {
    let trace = valid_fib_trace(3);
    let instance = ProgramInstance::new(8, vec![F::from(99u128), F::from(77u128)]);
    let witness = ProgramWitness::<F>::new(trace);

    let report = preflight(&FibAir, &instance, &witness).unwrap();
    eprintln!("{}", report);

    assert!(!report.is_clean());
    assert_eq!(report.boundary_violations.len(), 2);
    assert_eq!(report.boundary_violations[0].col_idx, FibCols::A);
    assert_eq!(report.boundary_violations[1].col_idx, FibCols::B);
}

#[test]
fn gpa_matching_buses_pass() {
    let (cpu_trace, mem_trace) = matching_bus_traces(3, 4);
    let instance = ProgramInstance::new(8, vec![]);
    let witness = ProgramWitness::<F>::new(cpu_trace).with_chiplets(vec![mem_trace]);

    let report = preflight(&CpuWithBus, &instance, &witness).unwrap();
    eprintln!("{}", report);

    assert!(report.is_clean());
}

#[test]
fn detects_gpa_bus_mismatch() {
    let (cpu_trace, mut mem_trace) = matching_bus_traces(3, 4);

    // Corrupt chiplet row 1 value
    if let TraceColumn::B32(ref mut v) = mem_trace.columns[BusCols::VAL] {
        v[1] = Block32::from(0xBADu32).to_hardware();
    }

    let instance = ProgramInstance::new(8, vec![]);
    let witness = ProgramWitness::<F>::new(cpu_trace).with_chiplets(vec![mem_trace]);

    let report = preflight(&CpuWithBus, &instance, &witness).unwrap();
    eprintln!("{}", report);

    assert!(!report.is_clean());
    assert_eq!(report.bus_diagnostics.len(), 1);

    let d = &report.bus_diagnostics[0];

    assert_eq!(d.bus_id, "test_bus");
    assert_eq!(d.endpoints.len(), 2);
    assert!(
        d.bus_imbalance,
        "Permutation bus imbalance must surface as a named flag"
    );
    assert!(d.has_failures());
    assert_ne!(
        d.endpoints[0].claimed_sum, d.endpoints[1].claimed_sum,
        "endpoints' claimed_sums must diverge under multiset mismatch",
    );

    // mismatching_rows is the Lookup pointwise signal;
    // Permutation buses must leave it empty.
    assert!(d.mismatching_rows.is_empty());
}

#[test]
fn detects_chiplet_constraint_violation() {
    let (cpu_trace, mut mem_trace) = matching_bus_traces(3, 4);

    // Corrupt selector to non-boolean
    if let TraceColumn::Bit(ref mut v) = mem_trace.columns[BusCols::SEL] {
        v[0] = Bit(2);
    }

    let instance = ProgramInstance::new(8, vec![]);
    let witness = ProgramWitness::<F>::new(cpu_trace).with_chiplets(vec![mem_trace]);

    let report = preflight(&CpuWithBus, &instance, &witness).unwrap();
    eprintln!("{}", report);

    assert!(!report.is_clean());

    let chiplet_violations: Vec<_> = report
        .constraint_violations
        .iter()
        .filter(|v| matches!(v.table, TableId::Chiplet(0)))
        .collect();

    assert!(!chiplet_violations.is_empty());
    assert_eq!(chiplet_violations[0].label, Some("boolean"));
    assert_eq!(chiplet_violations[0].row_idx, 0);
}

// =================================================================
// VIRTUAL COLUMN EXPANSION TESTS
// =================================================================

#[derive(Clone)]
struct HostForBitUnpack;

impl Air<F> for HostForBitUnpack {
    fn num_columns(&self) -> usize {
        0
    }

    fn column_layout(&self) -> &[ColumnType] {
        &[]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
    }
}

impl Program<F> for HostForBitUnpack {
    fn chiplet_defs(&self) -> errors::Result<Vec<ChipletDef<F>>> {
        Ok(vec![ChipletDef::from_air(&BitUnpackChiplet)?])
    }
}

#[derive(Clone)]
struct BitUnpackChiplet;

impl BitUnpackChiplet {
    const VIRTUAL_COLS: usize = 32;
}

impl Air<F> for BitUnpackChiplet {
    fn column_layout(&self) -> &[ColumnType] {
        &[ColumnType::B32]
    }

    fn virtual_expander(&self) -> Option<&VirtualExpander> {
        static E: std::sync::OnceLock<VirtualExpander> = std::sync::OnceLock::new();
        Some(E.get_or_init(|| {
            VirtualExpander::new()
                .expand_bits(1, ColumnType::B32)
                .build()
                .expect("BitUnpackChiplet expander")
        }))
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();

        for i in 0..Self::VIRTUAL_COLS {
            cs.assert_boolean(cs.col(i));
        }

        // bit[0] XOR bit[1] = 0 ⟹ bit[0] == bit[1]
        cs.constrain_named("bits_01_equal", cs.col(0) + cs.col(1));

        cs.build()
    }
}

impl Program<F> for BitUnpackChiplet {}

fn bit_unpack_trace(num_vars: usize, values: &[u32]) -> ColumnTrace {
    let num_rows = 1 << num_vars;
    let mut tb = TraceBuilder::new(&[ColumnType::B32], num_vars).unwrap();

    for (i, &val) in values.iter().enumerate() {
        if i < num_rows {
            tb.set_b32(0, i, Block32::from(val)).unwrap();
        }
    }

    tb.build()
}

#[test]
fn virtual_expansion_clean_passes() {
    let trace = bit_unpack_trace(2, &[0x0, 0x0, 0x0, 0x0]);
    let instance = ProgramInstance::new(4, vec![]);
    let witness = ProgramWitness::<F>::new(trace);

    let report = preflight(&BitUnpackChiplet, &instance, &witness).unwrap();
    eprintln!("{}", report);

    assert!(report.is_clean());
}

#[test]
fn virtual_expansion_detects_constraint_violation() {
    // 0x1:
    // bit[0]=1, bit[1]=0 -> bits_01_equal fails at row 0
    let trace = bit_unpack_trace(2, &[0x1, 0x0, 0x0, 0x0]);
    let instance = ProgramInstance::new(4, vec![]);
    let witness = ProgramWitness::<F>::new(trace);

    let report = preflight(&BitUnpackChiplet, &instance, &witness).unwrap();
    eprintln!("{}", report);

    assert!(!report.is_clean());

    let named: Vec<_> = report
        .constraint_violations
        .iter()
        .filter(|v| v.label == Some("bits_01_equal"))
        .collect();

    assert_eq!(named.len(), 1);
    assert_eq!(named[0].row_idx, 0);
}

#[test]
fn virtual_expansion_corrupt_physical_detected() {
    // 0x3
    // bit[0]=1, bit[1]=1 -> equal ✓
    // 0x2:
    // bit[0]=0, bit[1]=1 -> equal ✗ at row 1
    let trace = bit_unpack_trace(2, &[0x3, 0x2, 0x0, 0x0]);
    let instance = ProgramInstance::new(4, vec![]);
    let witness = ProgramWitness::<F>::new(trace);

    let report = preflight(&BitUnpackChiplet, &instance, &witness).unwrap();
    eprintln!("{}", report);

    assert!(!report.is_clean());

    let named: Vec<_> = report
        .constraint_violations
        .iter()
        .filter(|v| v.label == Some("bits_01_equal"))
        .collect();

    assert_eq!(named.len(), 1);
    assert_eq!(named[0].row_idx, 1);
}

#[test]
fn chiplet_virtual_expansion_clean() {
    let host_trace = TraceBuilder::new(&[], 2).unwrap().build();
    let chiplet_trace = bit_unpack_trace(2, &[0x0, 0x0, 0x0, 0x0]);
    let instance = ProgramInstance::new(4, vec![]);
    let witness = ProgramWitness::<F>::new(host_trace).with_chiplets(vec![chiplet_trace]);

    let report = preflight(&HostForBitUnpack, &instance, &witness).unwrap();
    eprintln!("{}", report);
    assert!(report.is_clean());
}

#[test]
fn chiplet_virtual_expansion_detects_violation() {
    let host_trace = TraceBuilder::new(&[], 2).unwrap().build();
    // 0x1:
    // bit[0]=1, bit[1]=0 -> bits_01_equal fails at row 0.
    let chiplet_trace = bit_unpack_trace(2, &[0x1, 0x0, 0x0, 0x0]);
    let instance = ProgramInstance::new(4, vec![]);
    let witness = ProgramWitness::<F>::new(host_trace).with_chiplets(vec![chiplet_trace]);

    let report = preflight(&HostForBitUnpack, &instance, &witness).unwrap();
    eprintln!("{}", report);

    assert!(!report.is_clean());

    let chiplet_violations: Vec<_> = report
        .constraint_violations
        .iter()
        .filter(|v| matches!(v.table, TableId::Chiplet(0)))
        .filter(|v| v.label == Some("bits_01_equal"))
        .collect();

    assert_eq!(chiplet_violations.len(), 1);
    assert_eq!(chiplet_violations[0].row_idx, 0);
}

// =================================================================
// LOOKUP-KIND BUS TESTS
// =================================================================

#[derive(Clone)]
struct CpuWithLookupBus;

fn lookup_spec() -> PermutationCheckSpec {
    PermutationCheckSpec::new_lookup(
        vec![
            (Source::Column(BusCols::ADDR), b"k0"),
            (Source::Column(BusCols::VAL), b"k1"),
        ],
        Some(BusCols::SEL),
    )
}

impl Air<F> for CpuWithLookupBus {
    fn num_columns(&self) -> usize {
        BusCols::NUM_COLUMNS
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(BusCols::build_layout)
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![("lookup_bus".to_string(), lookup_spec())]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
    }
}

impl Program<F> for CpuWithLookupBus {
    fn chiplet_defs(&self) -> errors::Result<Vec<ChipletDef<F>>> {
        Ok(vec![ChipletDef::from_air(&MemLookupChiplet)?])
    }
}

#[derive(Clone)]
struct MemLookupChiplet;

impl Air<F> for MemLookupChiplet {
    fn num_columns(&self) -> usize {
        BusCols::NUM_COLUMNS
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(BusCols::build_layout)
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![("lookup_bus".to_string(), lookup_spec())]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(BusCols::SEL));

        cs.build()
    }
}

/// Builds two identical traces, both filled with
/// `(addr=i, val=i*10)` on `active` rows. Honest
/// Lookup witness, pointwise equal on the padded
/// hypercube.
fn aligned_lookup_traces(num_vars: usize, active: usize) -> (ColumnTrace, ColumnTrace) {
    matching_bus_traces(num_vars, active)
}

/// Same multisets as `aligned_lookup_traces`,
/// but the chiplet side writes rows in reverse
/// order. Product check passes; pointwise check
/// must fail.
fn shuffled_lookup_traces(num_vars: usize, active: usize) -> (ColumnTrace, ColumnTrace) {
    let layout = BusCols::build_layout();

    let mut cpu_tb = TraceBuilder::new(&layout, num_vars).unwrap();
    let mut mem_tb = TraceBuilder::new(&layout, num_vars).unwrap();

    for i in 0..active {
        let addr = Block32::from(i as u32);
        let val = Block32::from((i * 10) as u32);

        cpu_tb.set_bit(BusCols::SEL, i, Bit::ONE).unwrap();
        cpu_tb.set_b32(BusCols::ADDR, i, addr).unwrap();
        cpu_tb.set_b32(BusCols::VAL, i, val).unwrap();

        // Chiplet writes the same
        // multiset but reversed.
        let j = active - 1 - i;
        mem_tb.set_bit(BusCols::SEL, j, Bit::ONE).unwrap();
        mem_tb.set_b32(BusCols::ADDR, j, addr).unwrap();
        mem_tb.set_b32(BusCols::VAL, j, val).unwrap();
    }

    (cpu_tb.build(), mem_tb.build())
}

/// CPU emits a duplicate key `X` on rows that
/// would cancel under plain char-2 Permutation.
/// Chiplet has only the non-forged keys. Both
/// product and pointwise checks must fail.
fn forged_pair_lookup_traces(num_vars: usize) -> (ColumnTrace, ColumnTrace) {
    let layout = BusCols::build_layout();

    let mut cpu_tb = TraceBuilder::new(&layout, num_vars).unwrap();
    let mut mem_tb = TraceBuilder::new(&layout, num_vars).unwrap();

    // CPU honest rows 0..2.
    for i in 0..2 {
        let addr = Block32::from(i as u32);
        let val = Block32::from((i * 10) as u32);

        cpu_tb.set_bit(BusCols::SEL, i, Bit::ONE).unwrap();
        cpu_tb.set_b32(BusCols::ADDR, i, addr).unwrap();
        cpu_tb.set_b32(BusCols::VAL, i, val).unwrap();

        mem_tb.set_bit(BusCols::SEL, i, Bit::ONE).unwrap();
        mem_tb.set_b32(BusCols::ADDR, i, addr).unwrap();
        mem_tb.set_b32(BusCols::VAL, i, val).unwrap();
    }

    // CPU forges an `X, X` pair on rows 2, 3.
    let x_addr = Block32::from(0xDEADu32);
    let x_val = Block32::from(0xBEEFu32);

    for i in 2..4 {
        cpu_tb.set_bit(BusCols::SEL, i, Bit::ONE).unwrap();
        cpu_tb.set_b32(BusCols::ADDR, i, x_addr).unwrap();
        cpu_tb.set_b32(BusCols::VAL, i, x_val).unwrap();
    }

    (cpu_tb.build(), mem_tb.build())
}

#[test]
fn lookup_aligned_buses_pass() {
    let (cpu_trace, mem_trace) = aligned_lookup_traces(3, 4);
    let instance = ProgramInstance::new(8, vec![]);
    let witness = ProgramWitness::<F>::new(cpu_trace).with_chiplets(vec![mem_trace]);

    let report = preflight(&CpuWithLookupBus, &instance, &witness).unwrap();
    eprintln!("{}", report);

    assert!(
        report.is_clean(),
        "aligned Lookup endpoints must pass preflight"
    );
}

#[test]
fn lookup_catches_row_shuffle() {
    let (cpu_trace, mem_trace) = shuffled_lookup_traces(3, 4);
    let instance = ProgramInstance::new(8, vec![]);
    let witness = ProgramWitness::<F>::new(cpu_trace).with_chiplets(vec![mem_trace]);

    let report = preflight(&CpuWithLookupBus, &instance, &witness).unwrap();
    eprintln!("{}", report);

    assert_eq!(report.bus_diagnostics.len(), 1);
    let d = &report.bus_diagnostics[0];

    assert_eq!(d.bus_id, "lookup_bus");
    assert_eq!(d.kind, BusKind::Lookup);

    // Products match (same multiset), so only
    // the pointwise check fires.
    let products_match = d.endpoints.windows(2).all(|w| w[0].product == w[1].product);
    assert!(
        products_match,
        "shuffled multisets must still match on product"
    );
    assert!(
        !d.mismatching_rows.is_empty(),
        "pointwise check must flag shuffled rows under Lookup"
    );

    // Rows 0..active are all shuffled against each
    // other, so every active row reports a mismatch.
    assert_eq!(d.mismatching_rows.len(), 4);
}

#[test]
fn lookup_catches_parity_forgery() {
    let (cpu_trace, mem_trace) = forged_pair_lookup_traces(3);
    let instance = ProgramInstance::new(8, vec![]);
    let witness = ProgramWitness::<F>::new(cpu_trace).with_chiplets(vec![mem_trace]);

    let report = preflight(&CpuWithLookupBus, &instance, &witness).unwrap();
    eprintln!("{}", report);

    assert_eq!(report.bus_diagnostics.len(), 1);
    let d = &report.bus_diagnostics[0];

    assert_eq!(d.kind, BusKind::Lookup);

    // Products diverge because CPU has extra
    // `(γ + X)` factors that the chiplet lacks.
    let products_differ = !d.endpoints.windows(2).all(|w| w[0].product == w[1].product);
    assert!(products_differ, "forged keys must diverge products");

    // Pointwise check must also flag the forged rows.
    assert!(d.mismatching_rows.contains(&2));
    assert!(d.mismatching_rows.contains(&3));
}

// =================================================================
// LAGRANGE PIN TESTS
// =================================================================

define_columns! {
    PinCols {
        FLAG: Bit,
    }
}

#[derive(Clone)]
struct PinAir {
    pin: LagrangePin,
}

impl Air<F> for PinAir {
    fn num_columns(&self) -> usize {
        PinCols::NUM_COLUMNS
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(PinCols::build_layout)
    }

    fn lagrange_pinned_columns(&self) -> Vec<LagrangePin> {
        vec![self.pin.clone()]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
    }
}

impl Program<F> for PinAir {}

fn pin_trace(num_vars: usize, on_row: impl Fn(usize) -> Bit) -> ColumnTrace {
    let num_rows = 1 << num_vars;

    let mut tb = TraceBuilder::new(&PinCols::build_layout(), num_vars).unwrap();

    for i in 0..num_rows {
        tb.set_bit(PinCols::FLAG, i, on_row(i)).unwrap();
    }

    tb.build()
}

#[test]
fn lagrange_pin_last_row_clean_passes() {
    let num_vars = 3;
    let num_rows = 1 << num_vars;

    let air = PinAir {
        pin: LagrangePin::last_row(PinCols::FLAG),
    };

    let trace = pin_trace(num_vars, |i| {
        if i == num_rows - 1 {
            Bit::ZERO
        } else {
            Bit::ONE
        }
    });

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::<F>::new(trace);

    let report = preflight(&air, &instance, &witness).unwrap();
    eprintln!("{}", report);

    assert!(report.is_clean());
    assert!(report.lagrange_pin_violations.is_empty());
}

#[test]
fn lagrange_pin_last_row_detects_corrupted_last_row() {
    let num_vars = 3;
    let num_rows = 1 << num_vars;

    let air = PinAir {
        pin: LagrangePin::last_row(PinCols::FLAG),
    };

    let trace = pin_trace(num_vars, |_| Bit::ONE);

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::<F>::new(trace);

    let report = preflight(&air, &instance, &witness).unwrap();
    eprintln!("{}", report);

    assert!(!report.is_clean());
    assert_eq!(report.lagrange_pin_violations.len(), 1);

    let v = &report.lagrange_pin_violations[0];
    assert_eq!(v.col_idx, PinCols::FLAG);
    assert_eq!(v.row_idx, num_rows - 1);
}

#[test]
fn lagrange_pin_last_row_detects_corrupted_mid_row() {
    let num_vars = 3;
    let num_rows = 1 << num_vars;

    let air = PinAir {
        pin: LagrangePin::last_row(PinCols::FLAG),
    };

    let trace = pin_trace(num_vars, |i| {
        if i == 5 || i == num_rows - 1 {
            Bit::ZERO
        } else {
            Bit::ONE
        }
    });

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::<F>::new(trace);

    let report = preflight(&air, &instance, &witness).unwrap();
    eprintln!("{}", report);

    assert!(!report.is_clean());
    assert_eq!(report.lagrange_pin_violations.len(), 1);
    assert_eq!(report.lagrange_pin_violations[0].row_idx, 5);
}

#[test]
fn lagrange_pin_first_row_clean_passes() {
    let num_vars = 3;
    let num_rows = 1 << num_vars;

    let air = PinAir {
        pin: LagrangePin::first_row(PinCols::FLAG),
    };

    let trace = pin_trace(num_vars, |i| if i == 0 { Bit::ONE } else { Bit::ZERO });

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::<F>::new(trace);

    let report = preflight(&air, &instance, &witness).unwrap();
    eprintln!("{}", report);

    assert!(report.is_clean());
}

#[test]
fn lagrange_pin_first_row_detects_violation() {
    let num_vars = 3;
    let num_rows = 1 << num_vars;

    let air = PinAir {
        pin: LagrangePin::first_row(PinCols::FLAG),
    };

    let trace = pin_trace(num_vars, |_| Bit::ZERO);

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::<F>::new(trace);

    let report = preflight(&air, &instance, &witness).unwrap();
    eprintln!("{}", report);

    assert_eq!(report.lagrange_pin_violations.len(), 1);
    assert_eq!(report.lagrange_pin_violations[0].row_idx, 0);
}

#[test]
fn lagrange_pin_custom_clean_passes() {
    let num_vars = 3;
    let num_rows = 1 << num_vars;
    let target_row = 5;

    let bits: Vec<bool> = (0..num_vars).map(|k| (target_row >> k) & 1 == 1).collect();

    let air = PinAir {
        pin: LagrangePin::custom(PinCols::FLAG, bits),
    };

    let trace = pin_trace(
        num_vars,
        |i| {
            if i == target_row { Bit::ONE } else { Bit::ZERO }
        },
    );

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::<F>::new(trace);

    let report = preflight(&air, &instance, &witness).unwrap();
    eprintln!("{}", report);

    assert!(report.is_clean());
}

#[test]
fn lagrange_pin_custom_detects_violation_at_target_row() {
    let num_vars = 3;
    let num_rows = 1 << num_vars;
    let target_row = 5;

    let bits: Vec<bool> = (0..num_vars).map(|k| (target_row >> k) & 1 == 1).collect();

    let air = PinAir {
        pin: LagrangePin::custom(PinCols::FLAG, bits),
    };

    let trace = pin_trace(num_vars, |_| Bit::ZERO);

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::<F>::new(trace);

    let report = preflight(&air, &instance, &witness).unwrap();
    eprintln!("{}", report);

    assert_eq!(report.lagrange_pin_violations.len(), 1);
    assert_eq!(report.lagrange_pin_violations[0].row_idx, target_row);
}

#[test]
fn lagrange_pin_out_of_range_col_idx_rejected_in_preflight() {
    let num_vars = 3;
    let num_rows = 1 << num_vars;

    let air = PinAir {
        pin: LagrangePin::last_row(99),
    };

    let trace = pin_trace(num_vars, |_| Bit::ZERO);

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::<F>::new(trace);

    let result = preflight(&air, &instance, &witness);

    match &result {
        Err(e) => eprintln!("\n  PREFLIGHT rejected out-of-range pin: {:?}\n", e),
        Ok(report) => {
            eprintln!("\n  PREFLIGHT unexpectedly returned Ok\n");
            eprintln!("{}", report);
        }
    }

    assert!(matches!(
        result,
        Err(errors::Error::Protocol {
            protocol: "lagrange_pin",
            ..
        })
    ));
}

// =================================================================
// CHIPLET BOUNDARY TESTS
// =================================================================

define_columns! {
    BoundaryCols {
        FLAG: Bit,
    }
}

#[derive(Clone)]
struct BoundaryChiplet;

impl Air<F> for BoundaryChiplet {
    fn num_columns(&self) -> usize {
        BoundaryCols::NUM_COLUMNS
    }

    fn boundary_constraints(&self) -> Vec<BoundaryConstraint<F>> {
        vec![BoundaryConstraint::with_constant(
            BoundaryCols::FLAG,
            0,
            F::ONE,
        )]
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(BoundaryCols::build_layout)
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
    }
}

#[derive(Clone)]
struct BoundaryHost;

impl Air<F> for BoundaryHost {
    fn num_columns(&self) -> usize {
        0
    }

    fn column_layout(&self) -> &[ColumnType] {
        &[]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
    }
}

impl Program<F> for BoundaryHost {
    fn chiplet_defs(&self) -> errors::Result<Vec<ChipletDef<F>>> {
        Ok(vec![ChipletDef::from_air(&BoundaryChiplet)?])
    }
}

fn boundary_chiplet_trace(num_vars: usize, row0_flag: Bit) -> ColumnTrace {
    let layout = BoundaryCols::build_layout();

    let mut tb = TraceBuilder::new(&layout, num_vars).unwrap();
    tb.set_bit(BoundaryCols::FLAG, 0, row0_flag).unwrap();

    tb.build()
}

#[test]
fn detects_chiplet_boundary_violation() {
    let num_vars = 3;
    let num_rows = 1 << num_vars;

    let main_layout: Vec<ColumnType> = vec![];
    let main_trace = TraceBuilder::new(&main_layout, num_vars).unwrap().build();

    // Boundary expects FLAG[0] == 1;
    // trace sets it to 0.
    let chiplet_trace = boundary_chiplet_trace(num_vars, Bit::ZERO);

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::<F>::new(main_trace).with_chiplets(vec![chiplet_trace]);

    let report = preflight(&BoundaryHost, &instance, &witness).unwrap();
    eprintln!("{}", report);

    assert!(!report.is_clean());
    assert_eq!(report.boundary_violations.len(), 1);

    let v = &report.boundary_violations[0];
    assert!(matches!(v.table, TableId::Chiplet(0)));
    assert_eq!(v.col_idx, BoundaryCols::FLAG);
    assert_eq!(v.row_idx, 0);
}

#[test]
fn chiplet_boundary_clean_passes() {
    let num_vars = 3;
    let num_rows = 1 << num_vars;

    let main_layout: Vec<ColumnType> = vec![];
    let main_trace = TraceBuilder::new(&main_layout, num_vars).unwrap().build();

    let chiplet_trace = boundary_chiplet_trace(num_vars, Bit::ONE);

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::<F>::new(main_trace).with_chiplets(vec![chiplet_trace]);

    let report = preflight(&BoundaryHost, &instance, &witness).unwrap();
    eprintln!("{}", report);

    assert!(report.is_clean());
    assert!(report.boundary_violations.is_empty());
}

// =================================================================
// PAIRED BUS MUTEX DIAGNOSTICS
// =================================================================

define_columns! {
    PairedBusCols {
        KEY: B32,
        S_SEND: Bit,
        S_RECV: Bit,
    }
}

#[derive(Clone)]
struct PairedBusChiplet;

impl Air<F> for PairedBusChiplet {
    fn num_columns(&self) -> usize {
        PairedBusCols::NUM_COLUMNS
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(PairedBusCols::build_layout)
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        let sources = vec![
            (Source::Column(PairedBusCols::KEY), b"k_a" as &[u8]),
            (Source::RowIndexLeBytes(4), b"k_clk" as &[u8]),
        ];

        vec![(
            "paired_diag_bus".into(),
            PermutationCheckSpec::new_paired(
                sources,
                PairedBusCols::S_SEND,
                PairedBusCols::S_RECV,
                BusKind::Permutation,
            ),
        )]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();

        cs.assert_paired_bus_mutex(PairedBusCols::S_SEND, PairedBusCols::S_RECV);

        cs.build()
    }
}

#[derive(Clone)]
struct PairedBusHost;

impl Air<F> for PairedBusHost {
    fn num_columns(&self) -> usize {
        0
    }

    fn column_layout(&self) -> &[ColumnType] {
        &[]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
    }
}

impl Program<F> for PairedBusHost {
    fn chiplet_defs(&self) -> errors::Result<Vec<ChipletDef<F>>> {
        Ok(vec![ChipletDef::from_air(&PairedBusChiplet)?])
    }
}

fn paired_bus_chiplet_trace(num_vars: usize, rows: &[(u32, Bit, Bit)]) -> ColumnTrace {
    let layout = PairedBusCols::build_layout();
    let mut tb = TraceBuilder::new(&layout, num_vars).unwrap();

    for (i, (key, send, recv)) in rows.iter().enumerate() {
        tb.set_b32(PairedBusCols::KEY, i, hekate_math::Block32::from(*key))
            .unwrap();
        tb.set_bit(PairedBusCols::S_SEND, i, *send).unwrap();
        tb.set_bit(PairedBusCols::S_RECV, i, *recv).unwrap();
    }

    tb.build()
}

#[test]
fn exploit_preflight_misses_bidirectional_mutex_violation() {
    let num_vars = 3;
    let num_rows = 1 << num_vars;

    let main_layout: Vec<ColumnType> = vec![];
    let main = TraceBuilder::new(&main_layout, num_vars).unwrap().build();

    let chip = paired_bus_chiplet_trace(
        num_vars,
        &[(0xA1A1_A1A1, Bit::ONE, Bit::ONE), (0, Bit::ZERO, Bit::ZERO)],
    );

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::<F>::new(main).with_chiplets(vec![chip]);

    let report = preflight(&PairedBusHost, &instance, &witness).unwrap();
    eprintln!("{}", report);

    assert!(!report.is_clean());

    let diag = report
        .bus_diagnostics
        .iter()
        .find(|d| d.bus_id == "paired_diag_bus")
        .expect("diagnostic for paired_diag_bus must exist");

    assert!(
        diag.selector_mutex_violations
            .iter()
            .any(|(_, row)| *row == 0),
        "preflight must report row 0 as a mutex violation"
    );
}

#[test]
fn preflight_bidirectional_aligned_witness_clean() {
    let num_vars = 3;
    let num_rows = 1 << num_vars;

    let main_layout: Vec<ColumnType> = vec![];
    let main = TraceBuilder::new(&main_layout, num_vars).unwrap().build();

    let chip = paired_bus_chiplet_trace(
        num_vars,
        &[
            (0xA1A1_A1A1, Bit::ONE, Bit::ZERO),
            (0xB2B2_B2B2, Bit::ZERO, Bit::ONE),
        ],
    );

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::<F>::new(main).with_chiplets(vec![chip]);

    let report = preflight(&PairedBusHost, &instance, &witness).unwrap();
    eprintln!("{}", report);

    let bus_with_mutex_violations = report
        .bus_diagnostics
        .iter()
        .any(|d| !d.selector_mutex_violations.is_empty());

    assert!(
        !bus_with_mutex_violations,
        "honest disjoint witness must not report mutex violations"
    );
}

#[test]
fn chiplet_boundary_public_input_rejected_at_snapshot() {
    #[derive(Clone)]
    struct BadChiplet;

    impl Air<F> for BadChiplet {
        fn num_columns(&self) -> usize {
            BoundaryCols::NUM_COLUMNS
        }

        fn boundary_constraints(&self) -> Vec<BoundaryConstraint<F>> {
            vec![BoundaryConstraint::with_public_input(
                BoundaryCols::FLAG,
                0,
                0,
            )]
        }

        fn column_layout(&self) -> &[ColumnType] {
            static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
            LAYOUT.get_or_init(BoundaryCols::build_layout)
        }

        fn constraint_ast(&self) -> ConstraintAst<F> {
            ConstraintSystem::<F>::new().build()
        }
    }

    let result = ChipletDef::<F>::from_air(&BadChiplet);

    assert!(matches!(
        result,
        Err(errors::Error::Protocol {
            protocol: "boundary",
            ..
        })
    ));
}

// Two Permutation specs on a SINGLE main trace
// sharing one `bus_id` (both endpoints on main,
// products diverge).
define_columns! {
    DualBusCols {
        SEL_SEND: Bit,
        SEL_RECV: Bit,
        SEND_KEY: B32,
        RECV_KEY: B32,
    }
}

#[derive(Clone)]
struct MainPairBus;

impl Air<F> for MainPairBus {
    fn num_columns(&self) -> usize {
        DualBusCols::NUM_COLUMNS
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(DualBusCols::build_layout)
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![
            (
                "main_pair_bus".to_string(),
                PermutationCheckSpec::new(
                    vec![
                        (Source::Column(DualBusCols::SEND_KEY), b"k0"),
                        (Source::RowIndexLeBytes(4), REQUEST_IDX_LABEL),
                    ],
                    Some(DualBusCols::SEL_SEND),
                ),
            ),
            (
                "main_pair_bus".to_string(),
                PermutationCheckSpec::new(
                    vec![
                        (Source::Column(DualBusCols::RECV_KEY), b"k0"),
                        (Source::RowIndexLeBytes(4), REQUEST_IDX_LABEL),
                    ],
                    Some(DualBusCols::SEL_RECV),
                ),
            ),
        ]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
    }
}

impl Program<F> for MainPairBus {}

fn main_pair_trace(num_vars: usize, active: usize, recv_key_offset: u32) -> ColumnTrace {
    let layout = DualBusCols::build_layout();

    let mut tb = TraceBuilder::new(&layout, num_vars).unwrap();

    for i in 0..active {
        tb.set_bit(DualBusCols::SEL_SEND, i, Bit::ONE).unwrap();
        tb.set_bit(DualBusCols::SEL_RECV, i, Bit::ONE).unwrap();
        tb.set_b32(DualBusCols::SEND_KEY, i, Block32::from(i as u32))
            .unwrap();
        tb.set_b32(
            DualBusCols::RECV_KEY,
            i,
            Block32::from(i as u32 + recv_key_offset),
        )
        .unwrap();
    }

    tb.build()
}

#[test]
fn detects_main_main_permutation_bus_imbalance() {
    let trace = main_pair_trace(3, 4, 100);
    let instance = ProgramInstance::new(8, vec![]);
    let witness = ProgramWitness::<F>::new(trace);

    let report = preflight(&MainPairBus, &instance, &witness).unwrap();
    eprintln!("{}", report);

    assert!(!report.is_clean());

    let d = report
        .bus_diagnostics
        .iter()
        .find(|d| d.bus_id == "main_pair_bus")
        .expect("diagnostic for main_pair_bus must exist");

    assert_eq!(d.kind, BusKind::Permutation);
    assert_eq!(d.endpoints.len(), 2);
    assert!(matches!(d.endpoints[0].source, TableId::Main));
    assert!(matches!(d.endpoints[1].source, TableId::Main));
    assert!(d.bus_imbalance);
    assert!(d.has_failures());
    assert_ne!(d.endpoints[0].claimed_sum, d.endpoints[1].claimed_sum);
    assert!(d.mismatching_rows.is_empty());
}

#[test]
fn main_main_permutation_matching_keys_pass() {
    let trace = main_pair_trace(3, 4, 0);
    let instance = ProgramInstance::new(8, vec![]);
    let witness = ProgramWitness::<F>::new(trace);

    let report = preflight(&MainPairBus, &instance, &witness).unwrap();
    eprintln!("{}", report);

    assert!(report.is_clean());
}

// 1-SEND + 2-RECV partition on a single main trace.
define_columns! {
    PartitionCols {
        SEL_SEND: Bit,
        SEL_A: Bit,
        SEL_B: Bit,
        KEY_SEND: B32,
        KEY_A: B32,
        KEY_B: B32,
    }
}

#[derive(Clone)]
struct MainPartitionBus;

impl Air<F> for MainPartitionBus {
    fn num_columns(&self) -> usize {
        PartitionCols::NUM_COLUMNS
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(PartitionCols::build_layout)
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        let waiver =
            "see preflight test: per-row uniqueness via disjoint AIR selectors on the partition";
        vec![
            (
                "partition_bus".to_string(),
                PermutationCheckSpec::new(
                    vec![(Source::Column(PartitionCols::KEY_SEND), b"k0")],
                    Some(PartitionCols::SEL_SEND),
                )
                .with_clock_waiver(waiver),
            ),
            (
                "partition_bus".to_string(),
                PermutationCheckSpec::new(
                    vec![(Source::Column(PartitionCols::KEY_A), b"k0")],
                    Some(PartitionCols::SEL_A),
                )
                .with_clock_waiver(waiver),
            ),
            (
                "partition_bus".to_string(),
                PermutationCheckSpec::new(
                    vec![(Source::Column(PartitionCols::KEY_B), b"k0")],
                    Some(PartitionCols::SEL_B),
                )
                .with_clock_waiver(waiver),
            ),
        ]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
    }
}

impl Program<F> for MainPartitionBus {}

fn partition_trace(
    num_vars: usize,
    send_keys: &[u32],
    a_keys: &[u32],
    b_keys: &[u32],
) -> ColumnTrace {
    let layout = PartitionCols::build_layout();

    let mut tb = TraceBuilder::new(&layout, num_vars).unwrap();

    for (i, &k) in send_keys.iter().enumerate() {
        tb.set_bit(PartitionCols::SEL_SEND, i, Bit::ONE).unwrap();
        tb.set_b32(PartitionCols::KEY_SEND, i, Block32::from(k))
            .unwrap();
    }

    for (i, &k) in a_keys.iter().enumerate() {
        tb.set_bit(PartitionCols::SEL_A, i, Bit::ONE).unwrap();
        tb.set_b32(PartitionCols::KEY_A, i, Block32::from(k))
            .unwrap();
    }

    for (i, &k) in b_keys.iter().enumerate() {
        tb.set_bit(PartitionCols::SEL_B, i, Bit::ONE).unwrap();
        tb.set_b32(PartitionCols::KEY_B, i, Block32::from(k))
            .unwrap();
    }

    tb.build()
}

#[test]
fn multi_endpoint_permutation_partition_passes_preflight() {
    let trace = partition_trace(3, &[10, 11, 12, 13], &[10, 11], &[12, 13]);
    let instance = ProgramInstance::new(8, vec![]);
    let witness = ProgramWitness::<F>::new(trace);

    let report = preflight(&MainPartitionBus, &instance, &witness).unwrap();
    eprintln!("{}", report);

    assert!(report.is_clean());
}

#[test]
fn multi_endpoint_permutation_partition_imbalance_detected() {
    let trace = partition_trace(3, &[10, 11, 12, 13], &[10, 11], &[12, 99]);
    let instance = ProgramInstance::new(8, vec![]);
    let witness = ProgramWitness::<F>::new(trace);

    let report = preflight(&MainPartitionBus, &instance, &witness).unwrap();
    eprintln!("{}", report);

    assert!(!report.is_clean());

    let d = report
        .bus_diagnostics
        .iter()
        .find(|d| d.bus_id == "partition_bus")
        .expect("diagnostic for partition_bus must exist");

    assert_eq!(d.kind, BusKind::Permutation);
    assert_eq!(d.endpoints.len(), 3);
    assert!(d.bus_imbalance);
    assert!(d.has_failures());
}

// =================================================================
// BOUNDARY ON EXPANDED VIRTUAL COLUMN
// =================================================================

#[derive(Clone)]
struct ExpBoundaryChiplet;

impl Air<F> for ExpBoundaryChiplet {
    fn boundary_constraints(&self) -> Vec<BoundaryConstraint<F>> {
        // Virtual bit 31 sits past the single physical
        // column; it resolves only through expansion,
        // never physical indexing.
        vec![BoundaryConstraint::with_constant(31, 0, F::ZERO)]
    }

    fn column_layout(&self) -> &[ColumnType] {
        &[ColumnType::B32]
    }

    fn virtual_expander(&self) -> Option<&VirtualExpander> {
        static E: std::sync::OnceLock<VirtualExpander> = std::sync::OnceLock::new();
        Some(E.get_or_init(|| {
            VirtualExpander::new()
                .expand_bits(1, ColumnType::B32)
                .build()
                .expect("ExpBoundaryChiplet expander")
        }))
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
    }
}

#[derive(Clone)]
struct ExpBoundaryHost;

impl Air<F> for ExpBoundaryHost {
    fn num_columns(&self) -> usize {
        0
    }

    fn column_layout(&self) -> &[ColumnType] {
        &[]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
    }
}

impl Program<F> for ExpBoundaryHost {
    fn chiplet_defs(&self) -> errors::Result<Vec<ChipletDef<F>>> {
        Ok(vec![ChipletDef::from_air(&ExpBoundaryChiplet)?])
    }
}

#[test]
fn boundary_on_expanded_bit_clean_passes() {
    let num_vars = 2;
    let num_rows = 1 << num_vars;

    let main_trace = TraceBuilder::new(&[], num_vars).unwrap().build();

    // Bit 31 is 0 on every row,
    // matching the constant-0 boundary.
    let chiplet_trace = bit_unpack_trace(num_vars, &[]);

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::<F>::new(main_trace).with_chiplets(vec![chiplet_trace]);

    let report = preflight(&ExpBoundaryHost, &instance, &witness)
        .expect("boundary on an expanded virtual column must not error");
    eprintln!("{}", report);

    assert!(report.is_clean());
    assert!(report.boundary_violations.is_empty());
}

#[test]
fn boundary_on_expanded_bit_detects_violation() {
    let num_vars = 2;
    let num_rows = 1 << num_vars;

    let main_trace = TraceBuilder::new(&[], num_vars).unwrap().build();

    // 0x8000_0000 raises bit 31 at row 0,
    // diverging from the constant-0 boundary.
    let chiplet_trace = bit_unpack_trace(num_vars, &[0x8000_0000]);

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::<F>::new(main_trace).with_chiplets(vec![chiplet_trace]);

    let report = preflight(&ExpBoundaryHost, &instance, &witness)
        .expect("boundary on an expanded virtual column must not error");
    eprintln!("{}", report);

    assert!(!report.is_clean());
    assert_eq!(report.boundary_violations.len(), 1);

    let v = &report.boundary_violations[0];
    assert!(matches!(v.table, TableId::Chiplet(0)));
    assert_eq!(v.col_idx, 31);
    assert_eq!(v.row_idx, 0);
}
