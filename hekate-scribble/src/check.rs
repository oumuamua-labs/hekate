// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use crate::apply::apply_mutation;
use crate::config::ScribbleConfig;
use crate::mutation::Mutation;
use crate::strategy::mutation_strategy;
use crate::target::Target;
use hekate_core::errors;
use hekate_core::trace::{ColumnTrace, Trace, TraceColumn, TraceCompatibleField};
use hekate_math::{Bit, Block8, Block16, Block32, Block64, Block128, Flat, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness};
use hekate_sdk::preflight::{
    PreflightReport, TableId, check_air_constraints, check_air_rows, check_boundary_constraints,
    check_bus_multisets, preflight,
};
use proptest::test_runner::{Config as PtConfig, TestCaseError, TestRunner};
use std::cell::RefCell;
use std::env;

const ORACLE_ENV: &str = "HEKATE_SCRIBBLE_ORACLE";
const PREFLIGHT_FAILED: &str =
    "preflight failed during scribble check (mutation produced invalid trace shape)";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Oracle {
    Local,
    Both,
}

impl Oracle {
    fn from_env() -> Self {
        match env::var(ORACLE_ENV) {
            Ok(value) => Self::parse(Some(&value)),
            Err(env::VarError::NotPresent) => Self::parse(None),
            Err(env::VarError::NotUnicode(_)) => panic!("scribble: {ORACLE_ENV} is not unicode"),
        }
    }

    fn parse(value: Option<&str>) -> Self {
        match value {
            None | Some("local") => Self::Local,
            Some("both") => Self::Both,
            Some(other) => {
                panic!("scribble: {ORACLE_ENV} must be `local` or `both`, got `{other}`")
            }
        }
    }
}

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

/// Apply `mutation` to a fresh copy of `witness`
/// and run preflight. `witness` must pass preflight.
///
/// `Ok(report)`, the mutation was caught (`report` carries
/// the violations that proved it). `Err(report)`, the
/// mutation escaped (`report.is_clean()` is true);
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
    check_in_place(&mut scratch, mutation, |tampered, _| {
        preflight(air, instance, tampered).expect(PREFLIGHT_FAILED)
    })
}

/// Run `config.case_count()` proptest cases against `(air, instance, witness)`,
/// failing the test with the shrunk minimal `Mutation` if any tamper escapes.
/// Panics before the first case when `witness` fails preflight.
/// `HEKATE_SCRIBBLE_ORACLE=both` also runs the full
/// preflight per case and panics where verdicts differ.
pub fn assert_all_caught<P, F>(
    air: &P,
    instance: &ProgramInstance<F>,
    witness: &ProgramWitness<F, ColumnTrace>,
    config: ScribbleConfig,
) where
    P: Program<F>,
    F: TraceCompatibleField + Into<Block128>,
{
    assert_clean_baseline(air, instance, witness);
    assert_cases_caught(air, instance, witness, config);
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
    assert_clean_baseline(air, instance, witness);

    let n = witness.chiplet_traces.len();
    for target in std::iter::once(Target::Main).chain((0..n).map(Target::Chiplet)) {
        assert_cases_caught(air, instance, witness, config.clone().target(target));
    }
}

fn assert_clean_baseline<P, F>(
    air: &P,
    instance: &ProgramInstance<F>,
    witness: &ProgramWitness<F, ColumnTrace>,
) where
    P: Program<F>,
    F: TraceCompatibleField + Into<Block128>,
{
    let report = preflight(air, instance, witness).expect("preflight failed on the clean witness");

    assert!(
        report.is_clean(),
        "scribble: clean witness fails preflight, every tamper would count as caught\n{report}"
    );
}

fn assert_cases_caught<P, F>(
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

    let oracle = Oracle::from_env();
    let chiplet_defs = air
        .chiplet_defs()
        .expect("chiplet_defs failed on the clean witness's program");

    let scratch = RefCell::new(clone_witness(witness));

    let result = runner.run(&strategy, |mutation: Mutation| {
        let mut tampered = scratch.borrow_mut();
        check_in_place(&mut tampered, &mutation, |witness, patches| {
            judge_case(
                air,
                instance,
                &chiplet_defs,
                oracle,
                witness,
                patches,
                &mutation,
            )
        })
        .map(|_| ())
        .map_err(|_report| TestCaseError::fail("mutation escaped preflight"))
    });

    if let Err(err) = result {
        panic!("scribble: {err}");
    }
}

fn check_in_place<F>(
    witness: &mut ProgramWitness<F, ColumnTrace>,
    mutation: &Mutation,
    judge: impl FnOnce(&ProgramWitness<F, ColumnTrace>, &[CellPatch]) -> PreflightReport<F>,
) -> Result<PreflightReport<F>, PreflightReport<F>>
where
    F: TraceCompatibleField,
{
    let patches = snapshot_for_mutation(witness, mutation);
    let guard = RestoreGuard { witness, patches };

    apply_mutation(guard.witness, mutation);

    if is_noop(guard.witness, &guard.patches) {
        return Ok(PreflightReport::new());
    }

    let report = judge(guard.witness, &guard.patches);

    drop(guard);

    if report.is_clean() {
        Err(report)
    } else {
        Ok(report)
    }
}

fn judge_case<P, F>(
    air: &P,
    instance: &ProgramInstance<F>,
    chiplet_defs: &[ChipletDef<F>],
    oracle: Oracle,
    witness: &ProgramWitness<F, ColumnTrace>,
    patches: &[CellPatch],
    mutation: &Mutation,
) -> PreflightReport<F>
where
    P: Program<F>,
    F: TraceCompatibleField + Into<Block128>,
{
    let local =
        local_report(air, instance, chiplet_defs, witness, patches).expect(PREFLIGHT_FAILED);

    match oracle {
        Oracle::Local => {}
        Oracle::Both => {
            let full = preflight(air, instance, witness).expect(PREFLIGHT_FAILED);

            assert_eq!(
                local.is_clean(),
                full.is_clean(),
                "scribble: local and full oracles disagree on {mutation:?}\nlocal:\n{local}\nfull:\n{full}"
            );
        }
    }

    local
}

fn local_report<P, F>(
    air: &P,
    instance: &ProgramInstance<F>,
    chiplet_defs: &[ChipletDef<F>],
    witness: &ProgramWitness<F, ColumnTrace>,
    patches: &[CellPatch],
) -> errors::Result<PreflightReport<F>>
where
    P: Program<F>,
    F: TraceCompatibleField + Into<Block128>,
{
    let chiplet_instance = ProgramInstance::new(1, vec![]);

    let mut report = PreflightReport::new();
    for (target, rows) in written_rows(witness, patches) {
        match target {
            Target::Main => check_table(
                air,
                instance,
                &witness.trace,
                TableId::Main,
                &rows,
                &mut report,
            )?,
            Target::Chiplet(i) => check_table(
                &chiplet_defs[i],
                &chiplet_instance,
                &witness.chiplet_traces[i],
                TableId::Chiplet(i),
                &rows,
                &mut report,
            )?,
        }
    }

    check_bus_multisets(air, chiplet_defs, witness, &mut report)?;

    Ok(report)
}

fn check_table<A, F>(
    air: &A,
    instance: &ProgramInstance<F>,
    trace: &ColumnTrace,
    table: TableId,
    rows: &[usize],
    report: &mut PreflightReport<F>,
) -> errors::Result<()>
where
    A: Air<F>,
    F: TraceCompatibleField,
{
    match rows.len() >= 1 << trace.num_vars {
        true => check_air_constraints(air, trace, table, report)?,
        false => check_air_rows(air, trace, table, rows, report)?,
    }

    check_boundary_constraints(air, instance, trace, table, report)
}

fn written_rows<F: TraceCompatibleField>(
    witness: &ProgramWitness<F, ColumnTrace>,
    patches: &[CellPatch],
) -> Vec<(Target, Vec<usize>)> {
    let mut tables: Vec<(Target, Vec<usize>)> = Vec::new();
    for patch in patches {
        let num_rows = 1usize << patch.target.resolve(witness).num_vars;

        let idx = match tables
            .iter()
            .position(|(target, _)| *target == patch.target)
        {
            Some(idx) => idx,
            None => {
                tables.push((patch.target, Vec::new()));

                tables.len() - 1
            }
        };

        tables[idx].1.push(patch.row);
        tables[idx].1.push((patch.row + num_rows - 1) % num_rows);
    }

    for (_, rows) in &mut tables {
        rows.sort_unstable();
        rows.dedup();
    }

    tables
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
        (TraceColumn::Bit(v), CellValue::Bit(x)) => v[row].get() == x.get(),
        (TraceColumn::B8(v), CellValue::B8(x)) => v[row].into_raw().0 == x.into_raw().0,
        (TraceColumn::B16(v), CellValue::B16(x)) => v[row].into_raw().0 == x.into_raw().0,
        (TraceColumn::B32(v), CellValue::B32(x)) => v[row].into_raw().0 == x.into_raw().0,
        (TraceColumn::B64(v), CellValue::B64(x)) => v[row].into_raw().0 == x.into_raw().0,
        (TraceColumn::B128(v), CellValue::B128(x)) => v[row].into_raw().0 == x.into_raw().0,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hekate_core::trace::ColumnType;
    use hekate_program::circuit::Circuit;

    const NUM_VARS: usize = 3;
    const N: usize = 1 << NUM_VARS;

    #[test]
    fn oracle_reads_local_and_both() {
        assert_eq!(Oracle::parse(None), Oracle::Local);
        assert_eq!(Oracle::parse(Some("local")), Oracle::Local);
        assert_eq!(Oracle::parse(Some("both")), Oracle::Both);
    }

    #[test]
    #[should_panic(expected = "must be `local` or `both`, got `bth`")]
    fn oracle_rejects_unknown_value() {
        Oracle::parse(Some("bth"));
    }

    #[test]
    #[should_panic(expected = "local and full oracles disagree")]
    fn both_oracle_panics_when_verdicts_differ() {
        let mut cx = Circuit::<Block128>::new("ZeroColumn", N).unwrap();
        cx.column(ColumnType::B32);

        let cs = cx.cs();
        cs.constrain(cs.col(0));

        let air = cx.compile().unwrap();

        let mut x = vec![Flat::from_raw(Block32(0)); N];
        x[5] = Flat::from_raw(Block32(1));

        let mut trace = ColumnTrace::new(NUM_VARS).unwrap();
        trace.add_column(TraceColumn::B32(x)).unwrap();

        let witness = ProgramWitness::new(trace);
        let instance = ProgramInstance::new(N, vec![]);

        let mutation = Mutation::BitFlip {
            target: Target::Main,
            col: 0,
            row: 1,
            mask: 1,
        };

        let patches = snapshot_for_mutation(&witness, &mutation);

        judge_case(
            &air,
            &instance,
            &[],
            Oracle::Both,
            &witness,
            &patches,
            &mutation,
        );
    }
}
