// SPDX-License-Identifier: Apache-2.0
// This file is part of the hekate project.
// Copyright (C) 2026 Andrei Kochergin <andrei@oumuamua.dev>
// Copyright (C) 2026 Oumuamua Labs <info@oumuamua.dev>.
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

//! Unit-level coverage for the trace-wide
//! mutation kinds: `ColumnUniformWrite`,
//! `RowSegmentZero`, `MonotonicReplace`.
//!
//! Verifies apply correctness via direct
//! read-back, `Mutation::kind` mapping,
//! `ScribbleConfig` gating, proptest-strategy
//! generation, and the engine round-trip
//! through `check_single_mutation`.

use hekate_core::trace::{ColumnTrace, ColumnType, TraceColumn};
use hekate_math::{Bit, Block32, Block128, Flat, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness};
use hekate_scribble::{
    Mutation, MutationKind, ScribbleConfig, Target, apply_mutation, check_single_mutation,
    mutation_strategy,
};
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::{Config as PtConfig, TestRunner};

const N: usize = 4;
const NUM_VARS: usize = 2;

type F = Block128;

#[derive(Clone)]
struct TrivialAir;

impl Air<F> for TrivialAir {
    fn column_layout(&self) -> &[ColumnType] {
        static L: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        L.get_or_init(|| vec![ColumnType::Bit, ColumnType::B32])
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
    }
}

impl Program<F> for TrivialAir {
    fn chiplet_defs(&self) -> hekate_core::errors::Result<Vec<ChipletDef<F>>> {
        Ok(vec![])
    }
}

fn build_trace() -> ColumnTrace {
    let bit_data: Vec<Bit> = (0..N)
        .map(|r| if r % 2 == 0 { Bit::ZERO } else { Bit::ONE })
        .collect();
    let b32_data: Vec<Flat<Block32>> = (0..N)
        .map(|r| Flat::from_raw(Block32(10 * (r as u32 + 1))))
        .collect();

    let mut t = ColumnTrace::new(NUM_VARS).unwrap();
    t.add_column(TraceColumn::Bit(bit_data)).unwrap();
    t.add_column(TraceColumn::B32(b32_data)).unwrap();

    t
}

fn build_witness() -> ProgramWitness<F, ColumnTrace> {
    ProgramWitness::new(build_trace())
}

fn build_dual_witness() -> ProgramWitness<F, ColumnTrace> {
    let main = build_trace();

    let chiplet_data: Vec<Flat<Block32>> = (0..N)
        .map(|r| Flat::from_raw(Block32(100 + r as u32)))
        .collect();

    let mut chiplet = ColumnTrace::new(NUM_VARS).unwrap();
    chiplet.add_column(TraceColumn::B32(chiplet_data)).unwrap();

    ProgramWitness::new(main).with_chiplets(vec![chiplet])
}

fn read_bit(trace: &ColumnTrace, col: usize, row: usize) -> u8 {
    match &trace.columns[col] {
        TraceColumn::Bit(v) => v[row].0,
        _ => panic!("expected Bit column"),
    }
}

fn read_b32(trace: &ColumnTrace, col: usize, row: usize) -> u32 {
    match &trace.columns[col] {
        TraceColumn::B32(v) => v[row].into_raw().0,
        _ => panic!("expected B32 column"),
    }
}

fn sample_one<S>(strategy: &S) -> Mutation
where
    S: Strategy<Value = Mutation>,
{
    let mut runner = TestRunner::new(PtConfig::default());
    strategy.new_tree(&mut runner).unwrap().current()
}

#[test]
fn column_uniform_write_writes_every_row_b32() {
    let mut witness = build_witness();
    apply_mutation(
        &mut witness,
        &Mutation::ColumnUniformWrite {
            target: Target::Main,
            col: 1,
            value: 0xDEAD_BEEF,
        },
    );

    for row in 0..N {
        assert_eq!(read_b32(&witness.trace, 1, row), 0xDEAD_BEEF);
    }
}

#[test]
fn column_uniform_write_truncates_to_bit_width() {
    let mut witness = build_witness();
    apply_mutation(
        &mut witness,
        &Mutation::ColumnUniformWrite {
            target: Target::Main,
            col: 0,
            value: 0xFE,
        },
    );

    for row in 0..N {
        assert_eq!(read_bit(&witness.trace, 0, row), 0);
    }
}

#[test]
fn column_uniform_write_leaves_other_columns_untouched() {
    let mut witness = build_witness();
    let bit_before: Vec<u8> = (0..N).map(|r| read_bit(&witness.trace, 0, r)).collect();

    apply_mutation(
        &mut witness,
        &Mutation::ColumnUniformWrite {
            target: Target::Main,
            col: 1,
            value: 0,
        },
    );

    let bit_after: Vec<u8> = (0..N).map(|r| read_bit(&witness.trace, 0, r)).collect();
    assert_eq!(bit_before, bit_after);
}

#[test]
fn row_segment_zero_zeroes_specified_cells() {
    let mut witness = build_witness();
    apply_mutation(
        &mut witness,
        &Mutation::RowSegmentZero {
            target: Target::Main,
            rows: vec![1, 3],
            cols: vec![1],
        },
    );

    assert_eq!(read_b32(&witness.trace, 1, 1), 0);
    assert_eq!(read_b32(&witness.trace, 1, 3), 0);
    assert_eq!(read_b32(&witness.trace, 1, 0), 10);
    assert_eq!(read_b32(&witness.trace, 1, 2), 30);
}

#[test]
fn row_segment_zero_cross_product() {
    let mut witness = build_witness();
    apply_mutation(
        &mut witness,
        &Mutation::RowSegmentZero {
            target: Target::Main,
            rows: vec![0, 1],
            cols: vec![0, 1],
        },
    );

    for row in 0..=1 {
        assert_eq!(read_bit(&witness.trace, 0, row), 0);
        assert_eq!(read_b32(&witness.trace, 1, row), 0);
    }

    assert_eq!(read_b32(&witness.trace, 1, 2), 30);
    assert_eq!(read_b32(&witness.trace, 1, 3), 40);
    assert_eq!(read_bit(&witness.trace, 0, 2), 0);
    assert_eq!(read_bit(&witness.trace, 0, 3), 1);
}

#[test]
fn row_segment_zero_on_zeroed_cell_is_noop_not_escape() {
    let air = TrivialAir;
    let instance = ProgramInstance::new(N, vec![]);
    let witness = build_witness();

    assert_eq!(read_bit(&witness.trace, 0, 0), 0);
    assert_eq!(read_bit(&witness.trace, 0, 2), 0);

    let noop = Mutation::RowSegmentZero {
        target: Target::Main,
        rows: vec![0, 2],
        cols: vec![0],
    };

    assert!(
        check_single_mutation(&air, &instance, &witness, &noop).is_ok(),
        "zeroing already-zero cells must not be flagged as an escape"
    );

    let real = Mutation::RowSegmentZero {
        target: Target::Main,
        rows: vec![1],
        cols: vec![1],
    };

    assert!(
        check_single_mutation(&air, &instance, &witness, &real).is_err(),
        "a genuine change under a constraint-free AIR must escape, proving the no-op guard is load-bearing"
    );
}

#[test]
fn monotonic_replace_writes_progression() {
    let mut witness = build_witness();
    apply_mutation(
        &mut witness,
        &Mutation::MonotonicReplace {
            target: Target::Main,
            col: 1,
            base: 100,
            step: 7,
        },
    );

    for row in 0..N {
        assert_eq!(
            read_b32(&witness.trace, 1, row),
            100 + 7 * row as u32,
            "row {row}"
        );
    }
}

#[test]
fn monotonic_replace_truncates_to_bit_width() {
    let mut witness = build_witness();
    apply_mutation(
        &mut witness,
        &Mutation::MonotonicReplace {
            target: Target::Main,
            col: 0,
            base: 0,
            step: 1,
        },
    );

    for row in 0..N {
        assert_eq!(read_bit(&witness.trace, 0, row), (row as u8) & 1);
    }
}

#[test]
fn monotonic_replace_step_zero_writes_constant() {
    let mut witness = build_witness();
    apply_mutation(
        &mut witness,
        &Mutation::MonotonicReplace {
            target: Target::Main,
            col: 1,
            base: 42,
            step: 0,
        },
    );

    for row in 0..N {
        assert_eq!(read_b32(&witness.trace, 1, row), 42);
    }
}

#[test]
fn kind_mapping_for_new_variants() {
    let m = Mutation::ColumnUniformWrite {
        target: Target::Main,
        col: 0,
        value: 0,
    };
    assert_eq!(m.kind(), Some(MutationKind::ColumnUniformWrite));

    let m = Mutation::RowSegmentZero {
        target: Target::Main,
        rows: vec![0],
        cols: vec![0],
    };
    assert_eq!(m.kind(), Some(MutationKind::RowSegmentZero));

    let m = Mutation::MonotonicReplace {
        target: Target::Main,
        col: 0,
        base: 0,
        step: 0,
    };
    assert_eq!(m.kind(), Some(MutationKind::MonotonicReplace));
}

#[test]
fn config_gates_new_kinds_independently() {
    let cfg = ScribbleConfig::default().mutations([MutationKind::ColumnUniformWrite]);
    assert!(cfg.is_kind_allowed(MutationKind::ColumnUniformWrite));
    assert!(!cfg.is_kind_allowed(MutationKind::BitFlip));
    assert!(!cfg.is_kind_allowed(MutationKind::RowSegmentZero));
    assert!(!cfg.is_kind_allowed(MutationKind::MonotonicReplace));
}

#[test]
fn config_default_does_not_filter_new_kinds() {
    // Empty `kinds` list means "no restriction".
    let cfg = ScribbleConfig::default();
    assert!(cfg.is_kind_allowed(MutationKind::ColumnUniformWrite));
    assert!(cfg.is_kind_allowed(MutationKind::RowSegmentZero));
    assert!(cfg.is_kind_allowed(MutationKind::MonotonicReplace));
}

#[test]
fn strategy_emits_column_uniform_write_when_opt_in() {
    let witness = build_witness();
    let cfg = ScribbleConfig::default().mutations([MutationKind::ColumnUniformWrite]);
    let strat = mutation_strategy(&witness, &cfg);

    let m = sample_one(&strat);
    assert_eq!(m.kind(), Some(MutationKind::ColumnUniformWrite));
}

#[test]
fn strategy_emits_row_segment_zero_when_opt_in() {
    let witness = build_witness();
    let cfg = ScribbleConfig::default().mutations([MutationKind::RowSegmentZero]);
    let strat = mutation_strategy(&witness, &cfg);

    let m = sample_one(&strat);
    assert_eq!(m.kind(), Some(MutationKind::RowSegmentZero));
}

#[test]
fn strategy_emits_monotonic_replace_when_opt_in() {
    let witness = build_witness();
    let cfg = ScribbleConfig::default().mutations([MutationKind::MonotonicReplace]);
    let strat = mutation_strategy(&witness, &cfg);

    let m = sample_one(&strat);
    assert_eq!(m.kind(), Some(MutationKind::MonotonicReplace));
}

#[test]
fn check_single_mutation_preserves_input_witness_for_new_kinds() {
    // Trivial AIR has no constraints; preflight
    // always returns clean. We don't care about
    // the verdict; we verify the engine round-trip
    // (clone + apply + preflight + restore) does
    // not mutate the caller's witness.
    let air = TrivialAir;
    let instance = ProgramInstance::new(N, vec![]);
    let witness = build_witness();

    let mutations = [
        Mutation::ColumnUniformWrite {
            target: Target::Main,
            col: 1,
            value: 0xCAFE_BABE,
        },
        Mutation::RowSegmentZero {
            target: Target::Main,
            rows: vec![0, 1, 2, 3],
            cols: vec![0, 1],
        },
        Mutation::MonotonicReplace {
            target: Target::Main,
            col: 1,
            base: 7,
            step: 3,
        },
    ];

    for m in &mutations {
        let _ = check_single_mutation(&air, &instance, &witness, m);
        assert_eq!(read_bit(&witness.trace, 0, 1), 1, "{m:?}");
        assert_eq!(read_b32(&witness.trace, 1, 0), 10, "{m:?}");
        assert_eq!(read_b32(&witness.trace, 1, 3), 40, "{m:?}");
    }
}

#[test]
fn compound_of_new_kinds_round_trips() {
    // Stress the recursion path in
    // `apply_mutation` + `collect_patches`
    // for `Compound` containing the new
    // variants.
    let air = TrivialAir;
    let instance = ProgramInstance::new(N, vec![]);
    let witness = build_witness();

    let m = Mutation::Compound(vec![
        Mutation::ColumnUniformWrite {
            target: Target::Main,
            col: 1,
            value: 0,
        },
        Mutation::RowSegmentZero {
            target: Target::Main,
            rows: vec![0, 2],
            cols: vec![0],
        },
        Mutation::MonotonicReplace {
            target: Target::Main,
            col: 1,
            base: 1,
            step: 1,
        },
    ]);

    let _ = check_single_mutation(&air, &instance, &witness, &m);
    assert_eq!(read_bit(&witness.trace, 0, 1), 1);
    assert_eq!(read_b32(&witness.trace, 1, 2), 30);
}

// =================================================================
// Shallow variants:
// BitFlip, OutOfBounds, FlipSelector
// =================================================================

#[test]
fn bitflip_changes_only_target_b32_cell() {
    let mut witness = build_witness();

    let before_0 = read_b32(&witness.trace, 1, 0);
    let before_1 = read_b32(&witness.trace, 1, 1);

    apply_mutation(
        &mut witness,
        &Mutation::BitFlip {
            target: Target::Main,
            col: 1,
            row: 0,
            mask: 0x0000_00FF,
        },
    );

    assert_ne!(read_b32(&witness.trace, 1, 0), before_0);
    assert_eq!(read_b32(&witness.trace, 1, 1), before_1);
}

#[test]
fn bitflip_truncates_mask_to_column_width() {
    let mut witness = build_witness();
    apply_mutation(
        &mut witness,
        &Mutation::BitFlip {
            target: Target::Main,
            col: 0,
            row: 0,
            mask: 0xFE,
        },
    );

    assert_eq!(read_bit(&witness.trace, 0, 0), 0);
}

#[test]
fn bitflip_with_mask_one_on_bit_toggles() {
    let mut witness = build_witness();
    apply_mutation(
        &mut witness,
        &Mutation::BitFlip {
            target: Target::Main,
            col: 0,
            row: 1,
            mask: 1,
        },
    );

    assert_eq!(read_bit(&witness.trace, 0, 1), 0);
}

#[test]
fn out_of_bounds_writes_value_truncated() {
    let mut witness = build_witness();
    apply_mutation(
        &mut witness,
        &Mutation::OutOfBounds {
            target: Target::Main,
            col: 1,
            row: 2,
            value: 0x1_0000_0042,
        },
    );

    assert_eq!(read_b32(&witness.trace, 1, 2), 0x42);
    assert_eq!(read_b32(&witness.trace, 1, 1), 20);
}

#[test]
fn flip_selector_toggles_bit_cell() {
    let mut witness = build_witness();
    apply_mutation(
        &mut witness,
        &Mutation::FlipSelector {
            target: Target::Main,
            col: 0,
            row: 0,
        },
    );
    apply_mutation(
        &mut witness,
        &Mutation::FlipSelector {
            target: Target::Main,
            col: 0,
            row: 1,
        },
    );

    assert_eq!(read_bit(&witness.trace, 0, 0), 1);
    assert_eq!(read_bit(&witness.trace, 0, 1), 0);
}

#[test]
#[should_panic]
fn flip_selector_panics_on_b32_column() {
    let mut witness = build_witness();
    apply_mutation(
        &mut witness,
        &Mutation::FlipSelector {
            target: Target::Main,
            col: 1,
            row: 0,
        },
    );
}

// =================================================================
// Structural variants:
// SwapRows, SwapColumns, DuplicateRow, CopyColumns
// =================================================================

#[test]
fn swap_rows_exchanges_every_column() {
    let mut witness = build_witness();
    apply_mutation(
        &mut witness,
        &Mutation::SwapRows {
            target: Target::Main,
            row_a: 0,
            row_b: 3,
        },
    );

    assert_eq!(read_bit(&witness.trace, 0, 0), 1);
    assert_eq!(read_bit(&witness.trace, 0, 3), 0);
    assert_eq!(read_b32(&witness.trace, 1, 0), 40);
    assert_eq!(read_b32(&witness.trace, 1, 3), 10);

    assert_eq!(read_b32(&witness.trace, 1, 1), 20);
    assert_eq!(read_b32(&witness.trace, 1, 2), 30);
}

#[test]
fn swap_columns_only_swaps_listed_columns() {
    let mut witness = build_witness();
    apply_mutation(
        &mut witness,
        &Mutation::SwapColumns {
            target: Target::Main,
            cols: vec![1],
            row_a: 0,
            row_b: 2,
        },
    );

    assert_eq!(read_b32(&witness.trace, 1, 0), 30);
    assert_eq!(read_b32(&witness.trace, 1, 2), 10);
    assert_eq!(read_bit(&witness.trace, 0, 0), 0);
    assert_eq!(read_bit(&witness.trace, 0, 2), 0);
}

#[test]
fn duplicate_row_copies_every_column() {
    let mut witness = build_witness();
    apply_mutation(
        &mut witness,
        &Mutation::DuplicateRow {
            target: Target::Main,
            src_row: 1,
            dst_row: 2,
        },
    );

    assert_eq!(read_bit(&witness.trace, 0, 2), 1);
    assert_eq!(read_b32(&witness.trace, 1, 2), 20);
    assert_eq!(read_bit(&witness.trace, 0, 1), 1);
    assert_eq!(read_b32(&witness.trace, 1, 1), 20);
}

#[test]
fn copy_columns_only_writes_listed_columns() {
    let mut witness = build_witness();
    apply_mutation(
        &mut witness,
        &Mutation::CopyColumns {
            target: Target::Main,
            cols: vec![1],
            src_row: 3,
            dst_row: 0,
        },
    );

    assert_eq!(read_b32(&witness.trace, 1, 0), 40);
    assert_eq!(read_bit(&witness.trace, 0, 0), 0);
}

// =================================================================
// kind() coverage for shallow + structural
// =================================================================

#[test]
fn kind_mapping_for_shallow_and_structural() {
    let cases: Vec<(Mutation, Option<MutationKind>)> = vec![
        (
            Mutation::BitFlip {
                target: Target::Main,
                col: 0,
                row: 0,
                mask: 0,
            },
            Some(MutationKind::BitFlip),
        ),
        (
            Mutation::OutOfBounds {
                target: Target::Main,
                col: 0,
                row: 0,
                value: 0,
            },
            Some(MutationKind::OutOfBounds),
        ),
        (
            Mutation::FlipSelector {
                target: Target::Main,
                col: 0,
                row: 0,
            },
            Some(MutationKind::FlipSelector),
        ),
        (
            Mutation::SwapRows {
                target: Target::Main,
                row_a: 0,
                row_b: 1,
            },
            Some(MutationKind::SwapRows),
        ),
        (
            Mutation::DuplicateRow {
                target: Target::Main,
                src_row: 0,
                dst_row: 1,
            },
            Some(MutationKind::DuplicateRow),
        ),
        (
            Mutation::SwapColumns {
                target: Target::Main,
                cols: vec![0],
                row_a: 0,
                row_b: 1,
            },
            None,
        ),
        (
            Mutation::CopyColumns {
                target: Target::Main,
                cols: vec![0],
                src_row: 0,
                dst_row: 1,
            },
            None,
        ),
        (Mutation::Compound(vec![]), None),
    ];

    for (m, k) in &cases {
        assert_eq!(m.kind(), *k, "{m:?}");
    }
}

// =================================================================
// Strategy coverage for shallow + structural
// =================================================================

#[test]
fn strategy_emits_bitflip_when_opt_in() {
    let witness = build_witness();
    let cfg = ScribbleConfig::default().mutations([MutationKind::BitFlip]);
    let m = sample_one(&mutation_strategy(&witness, &cfg));

    assert_eq!(m.kind(), Some(MutationKind::BitFlip));
}

#[test]
fn strategy_emits_out_of_bounds_when_opt_in() {
    let witness = build_witness();
    let cfg = ScribbleConfig::default().mutations([MutationKind::OutOfBounds]);
    let m = sample_one(&mutation_strategy(&witness, &cfg));

    assert_eq!(m.kind(), Some(MutationKind::OutOfBounds));
}

#[test]
fn strategy_emits_flip_selector_when_opt_in() {
    let witness = build_witness();
    let cfg = ScribbleConfig::default().mutations([MutationKind::FlipSelector]);
    let m = sample_one(&mutation_strategy(&witness, &cfg));

    assert_eq!(m.kind(), Some(MutationKind::FlipSelector));
}

#[test]
fn strategy_emits_swap_rows_when_opt_in() {
    let witness = build_witness();
    let cfg = ScribbleConfig::default().mutations([MutationKind::SwapRows]);
    let m = sample_one(&mutation_strategy(&witness, &cfg));

    assert_eq!(m.kind(), Some(MutationKind::SwapRows));
}

#[test]
fn strategy_emits_duplicate_row_when_opt_in() {
    let witness = build_witness();
    let cfg = ScribbleConfig::default().mutations([MutationKind::DuplicateRow]);
    let m = sample_one(&mutation_strategy(&witness, &cfg));

    assert_eq!(m.kind(), Some(MutationKind::DuplicateRow));
}

// =================================================================
// Engine round-trip for every shallow/structural variant
// =================================================================

#[test]
fn check_single_mutation_preserves_input_witness_for_all_variants() {
    let air = TrivialAir;
    let instance = ProgramInstance::new(N, vec![]);
    let witness = build_witness();

    let mutations = vec![
        Mutation::BitFlip {
            target: Target::Main,
            col: 1,
            row: 0,
            mask: 0xFFFF,
        },
        Mutation::OutOfBounds {
            target: Target::Main,
            col: 1,
            row: 1,
            value: 0xDEAD,
        },
        Mutation::FlipSelector {
            target: Target::Main,
            col: 0,
            row: 1,
        },
        Mutation::SwapRows {
            target: Target::Main,
            row_a: 0,
            row_b: 3,
        },
        Mutation::SwapColumns {
            target: Target::Main,
            cols: vec![1],
            row_a: 0,
            row_b: 2,
        },
        Mutation::DuplicateRow {
            target: Target::Main,
            src_row: 1,
            dst_row: 2,
        },
        Mutation::CopyColumns {
            target: Target::Main,
            cols: vec![1],
            src_row: 0,
            dst_row: 3,
        },
    ];

    for m in &mutations {
        let _ = check_single_mutation(&air, &instance, &witness, m);
        assert_eq!(read_bit(&witness.trace, 0, 0), 0, "{m:?}");
        assert_eq!(read_bit(&witness.trace, 0, 1), 1, "{m:?}");
        assert_eq!(read_b32(&witness.trace, 1, 0), 10, "{m:?}");
        assert_eq!(read_b32(&witness.trace, 1, 1), 20, "{m:?}");
        assert_eq!(read_b32(&witness.trace, 1, 2), 30, "{m:?}");
        assert_eq!(read_b32(&witness.trace, 1, 3), 40, "{m:?}");
    }
}

#[test]
fn nested_compound_round_trips() {
    let air = TrivialAir;
    let instance = ProgramInstance::new(N, vec![]);
    let witness = build_witness();

    let m = Mutation::Compound(vec![
        Mutation::Compound(vec![
            Mutation::BitFlip {
                target: Target::Main,
                col: 1,
                row: 0,
                mask: 1,
            },
            Mutation::SwapRows {
                target: Target::Main,
                row_a: 0,
                row_b: 1,
            },
        ]),
        Mutation::DuplicateRow {
            target: Target::Main,
            src_row: 2,
            dst_row: 3,
        },
    ]);

    let _ = check_single_mutation(&air, &instance, &witness, &m);

    assert_eq!(read_bit(&witness.trace, 0, 0), 0);
    assert_eq!(read_bit(&witness.trace, 0, 1), 1);
    assert_eq!(read_b32(&witness.trace, 1, 0), 10);
    assert_eq!(read_b32(&witness.trace, 1, 3), 40);
}

// =================================================================
// Chiplet target coverage
// =================================================================

#[test]
fn apply_mutation_targets_chiplet_trace() {
    let mut witness = build_dual_witness();
    apply_mutation(
        &mut witness,
        &Mutation::BitFlip {
            target: Target::Chiplet(0),
            col: 0,
            row: 1,
            mask: 0xFF,
        },
    );

    let main_unchanged = read_b32(&witness.trace, 1, 1);
    assert_eq!(main_unchanged, 20);

    let chiplet_after = match &witness.chiplet_traces[0].columns[0] {
        TraceColumn::B32(v) => v[1].into_raw().0,
        _ => unreachable!(),
    };

    assert_ne!(chiplet_after, 101);
}

#[test]
#[should_panic]
fn target_chiplet_out_of_bounds_panics() {
    let mut witness = build_dual_witness();
    apply_mutation(
        &mut witness,
        &Mutation::BitFlip {
            target: Target::Chiplet(7),
            col: 0,
            row: 0,
            mask: 1,
        },
    );
}
