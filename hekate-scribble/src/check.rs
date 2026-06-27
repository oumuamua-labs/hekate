use crate::apply::apply_mutation;
use crate::config::ScribbleConfig;
use crate::mutation::Mutation;
use crate::strategy::mutation_strategy;
use crate::target::Target;
use hekate_core::trace::{ColumnTrace, Trace, TraceColumn, TraceCompatibleField};
use hekate_math::{Bit, Block8, Block16, Block32, Block64, Block128, Flat, TowerField};
use hekate_program::{Program, ProgramInstance, ProgramWitness};
use hekate_sdk::preflight::{PreflightReport, preflight};
use proptest::test_runner::{Config as PtConfig, TestCaseError, TestRunner};
use std::cell::RefCell;

#[derive(Clone, Copy)]
enum CellValue {
    Bit(Bit),
    B8(Flat<Block8>),
    B16(Flat<Block16>),
    B32(Flat<Block32>),
    B64(Flat<Block64>),
    B128(Flat<Block128>),
}

struct CellPatch {
    target: Target,
    col: usize,
    row: usize,
    value: CellValue,
}

/// Restores `witness` on drop so a panic
/// inside `preflight` cannot leak mutated
/// state into the next proptest case.
struct RestoreGuard<'a, F: TraceCompatibleField> {
    witness: &'a mut ProgramWitness<F, ColumnTrace>,
    patches: Vec<CellPatch>,
}

impl<'a, F: TraceCompatibleField> Drop for RestoreGuard<'a, F> {
    fn drop(&mut self) {
        restore_all(self.witness, &self.patches);
    }
}

/// Apply `mutation` to a fresh copy
/// of `witness` and run preflight.
///
/// `Ok(report)`, the mutation was caught
/// (`report` carries the violations that
/// proved it). `Err(report)`, the mutation
/// escaped (`report.is_clean()` is true);
/// this is a soundness gap in the AIR.
pub fn check_single_mutation<P, F>(
    air: &P,
    instance: &ProgramInstance<F>,
    witness: &ProgramWitness<F, ColumnTrace>,
    mutation: &Mutation,
) -> Result<PreflightReport<F>, PreflightReport<F>>
where
    P: Program<F>,
    F: TraceCompatibleField + Into<Block128>,
{
    let mut scratch = clone_witness(witness);
    check_in_place(air, instance, &mut scratch, mutation)
}

/// Run `config.case_count()` proptest cases
/// against `(air, instance, witness)`,
/// failing the test with the shrunk minimal
/// `Mutation` if any tamper escapes.
pub fn assert_all_caught<P, F>(
    air: &P,
    instance: &ProgramInstance<F>,
    witness: &ProgramWitness<F, ColumnTrace>,
    config: ScribbleConfig,
) where
    P: Program<F>,
    F: TraceCompatibleField + Into<Block128>,
{
    let strategy = mutation_strategy(witness, &config);
    let mut runner = TestRunner::new(PtConfig {
        cases: config.case_count(),
        ..PtConfig::default()
    });

    let scratch = RefCell::new(clone_witness(witness));

    let result = runner.run(&strategy, |mutation: Mutation| {
        let mut tampered = scratch.borrow_mut();
        check_in_place(air, instance, &mut tampered, &mutation)
            .map(|_| ())
            .map_err(|_report| TestCaseError::fail("mutation escaped preflight"))
    });

    if let Err(err) = result {
        panic!("scribble: {err}");
    }
}

/// Run [`assert_all_caught`] once per
/// target derived from the witness:
/// `Target::Main` followed by `Target::Chiplet(i)`
/// for every chiplet trace present.
///
/// Any per-target filter on `config` is
/// overridden by the loop; pass a config
/// configured for kinds, cases, and column
/// filters only.
pub fn assert_all_caught_all_targets<P, F>(
    air: &P,
    instance: &ProgramInstance<F>,
    witness: &ProgramWitness<F, ColumnTrace>,
    config: ScribbleConfig,
) where
    P: Program<F>,
    F: TraceCompatibleField + Into<Block128>,
{
    let n = witness.chiplet_traces.len();
    for target in std::iter::once(Target::Main).chain((0..n).map(Target::Chiplet)) {
        assert_all_caught(air, instance, witness, config.clone().target(target));
    }
}

fn check_in_place<P, F>(
    air: &P,
    instance: &ProgramInstance<F>,
    witness: &mut ProgramWitness<F, ColumnTrace>,
    mutation: &Mutation,
) -> Result<PreflightReport<F>, PreflightReport<F>>
where
    P: Program<F>,
    F: TraceCompatibleField + Into<Block128>,
{
    let patches = snapshot_for_mutation(witness, mutation);

    let guard = RestoreGuard { witness, patches };

    apply_mutation(guard.witness, mutation);

    if is_noop(guard.witness, &guard.patches) {
        return Ok(PreflightReport::new());
    }

    let report = preflight(air, instance, guard.witness)
        .expect("preflight failed during scribble check (mutation produced invalid trace shape)");

    drop(guard);

    if report.is_clean() {
        Err(report)
    } else {
        Ok(report)
    }
}

/// Mirror of `ProgramWitness`'s public fields;
/// upstream additions will silently drop here.
fn clone_witness<F, T>(witness: &ProgramWitness<F, T>) -> ProgramWitness<F, T>
where
    F: TowerField,
    T: Trace + Clone,
{
    ProgramWitness::new(witness.trace.clone()).with_chiplets(witness.chiplet_traces.clone())
}

fn snapshot_for_mutation<F: TraceCompatibleField>(
    witness: &ProgramWitness<F, ColumnTrace>,
    mutation: &Mutation,
) -> Vec<CellPatch> {
    let mut patches = Vec::new();
    collect_patches(witness, mutation, &mut patches);

    patches
}

fn collect_patches<F: TraceCompatibleField>(
    witness: &ProgramWitness<F, ColumnTrace>,
    mutation: &Mutation,
    patches: &mut Vec<CellPatch>,
) {
    match mutation {
        Mutation::BitFlip {
            target, col, row, ..
        }
        | Mutation::OutOfBounds {
            target, col, row, ..
        }
        | Mutation::FlipSelector { target, col, row } => {
            patches.push(snapshot_cell(witness, *target, *col, *row));
        }
        Mutation::SwapRows {
            target,
            row_a,
            row_b,
        } => {
            let trace = target.resolve(witness);
            for col in 0..trace.columns.len() {
                patches.push(snapshot_cell(witness, *target, col, *row_a));
                patches.push(snapshot_cell(witness, *target, col, *row_b));
            }
        }
        Mutation::SwapColumns {
            target,
            cols,
            row_a,
            row_b,
        } => {
            for &col in cols {
                patches.push(snapshot_cell(witness, *target, col, *row_a));
                patches.push(snapshot_cell(witness, *target, col, *row_b));
            }
        }
        Mutation::DuplicateRow {
            target, dst_row, ..
        } => {
            let trace = target.resolve(witness);
            for col in 0..trace.columns.len() {
                patches.push(snapshot_cell(witness, *target, col, *dst_row));
            }
        }
        Mutation::CopyColumns {
            target,
            cols,
            dst_row,
            ..
        } => {
            for &col in cols {
                patches.push(snapshot_cell(witness, *target, col, *dst_row));
            }
        }
        Mutation::ColumnUniformWrite { target, col, .. } => {
            let trace = target.resolve(witness);
            let n = trace.columns[*col].len();

            for row in 0..n {
                patches.push(snapshot_cell(witness, *target, *col, row));
            }
        }
        Mutation::RowSegmentZero { target, rows, cols } => {
            for &col in cols {
                for &row in rows {
                    patches.push(snapshot_cell(witness, *target, col, row));
                }
            }
        }
        Mutation::MonotonicReplace { target, col, .. } => {
            let trace = target.resolve(witness);
            let n = trace.columns[*col].len();

            for row in 0..n {
                patches.push(snapshot_cell(witness, *target, *col, row));
            }
        }
        Mutation::Compound(ms) => {
            for sub in ms {
                collect_patches(witness, sub, patches);
            }
        }
    }
}

fn snapshot_cell<F: TraceCompatibleField>(
    witness: &ProgramWitness<F, ColumnTrace>,
    target: Target,
    col: usize,
    row: usize,
) -> CellPatch {
    let trace = target.resolve(witness);
    let value = read_cell(&trace.columns[col], row);

    CellPatch {
        target,
        col,
        row,
        value,
    }
}

fn read_cell(column: &TraceColumn, row: usize) -> CellValue {
    match column {
        TraceColumn::Bit(v) => CellValue::Bit(v[row]),
        TraceColumn::B8(v) => CellValue::B8(v[row]),
        TraceColumn::B16(v) => CellValue::B16(v[row]),
        TraceColumn::B32(v) => CellValue::B32(v[row]),
        TraceColumn::B64(v) => CellValue::B64(v[row]),
        TraceColumn::B128(v) => CellValue::B128(v[row]),
    }
}

fn restore_all<F: TraceCompatibleField>(
    witness: &mut ProgramWitness<F, ColumnTrace>,
    patches: &[CellPatch],
) {
    for patch in patches.iter().rev() {
        let trace = patch.target.resolve_mut(witness);
        write_cell(&mut trace.columns[patch.col], patch.row, patch.value);
    }
}

fn write_cell(column: &mut TraceColumn, row: usize, value: CellValue) {
    match (column, value) {
        (TraceColumn::Bit(v), CellValue::Bit(x)) => v[row] = x,
        (TraceColumn::B8(v), CellValue::B8(x)) => v[row] = x,
        (TraceColumn::B16(v), CellValue::B16(x)) => v[row] = x,
        (TraceColumn::B32(v), CellValue::B32(x)) => v[row] = x,
        (TraceColumn::B64(v), CellValue::B64(x)) => v[row] = x,
        (TraceColumn::B128(v), CellValue::B128(x)) => v[row] = x,
        _ => unreachable!("cell type mismatch in snapshot/restore"),
    }
}

fn is_noop<F: TraceCompatibleField>(
    witness: &ProgramWitness<F, ColumnTrace>,
    patches: &[CellPatch],
) -> bool {
    patches.iter().all(|p| {
        let trace = p.target.resolve(witness);
        cell_equal(&trace.columns[p.col], p.row, p.value)
    })
}

fn cell_equal(column: &TraceColumn, row: usize, value: CellValue) -> bool {
    match (column, value) {
        (TraceColumn::Bit(v), CellValue::Bit(x)) => v[row].0 == x.0,
        (TraceColumn::B8(v), CellValue::B8(x)) => v[row].into_raw().0 == x.into_raw().0,
        (TraceColumn::B16(v), CellValue::B16(x)) => v[row].into_raw().0 == x.into_raw().0,
        (TraceColumn::B32(v), CellValue::B32(x)) => v[row].into_raw().0 == x.into_raw().0,
        (TraceColumn::B64(v), CellValue::B64(x)) => v[row].into_raw().0 == x.into_raw().0,
        (TraceColumn::B128(v), CellValue::B128(x)) => v[row].into_raw().0 == x.into_raw().0,
        _ => false,
    }
}
