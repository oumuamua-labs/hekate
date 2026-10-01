// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use hashbrown::HashMap;
use hekate_core::errors;
use hekate_core::trace::{Trace, TraceColumn, TraceCompatibleField};
use hekate_math::{Block128, Flat, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::ConstraintAst;
use hekate_program::permutation::{
    BusKind, PermutationCheckSpec, RankClock, RankTable, Source, TableHeight, rank_clocks,
};
use hekate_program::{
    Air, FixedColumn, Program, ProgramInstance, ProgramWitness, validate_fixed_columns,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

const MAX_REPORTED_MISMATCH_ROWS: usize = 32;

const FMT_MAX_ROW_PREVIEW: usize = 8;
const FMT_MAX_MISMATCH_PREVIEW: usize = 16;

type AirViolations<F> = (Vec<ConstraintViolation<F>>, Vec<FixedColumnViolation<F>>);

type GroupedConstraintRows = BTreeMap<(TableId, usize), (Option<&'static str>, Vec<usize>)>;

pub struct ConstraintViolation<F> {
    pub table: TableId,
    pub constraint_idx: usize,
    pub label: Option<&'static str>,
    pub row_idx: usize,
    pub value: Flat<F>,
}

pub struct BoundaryViolation<F> {
    pub table: TableId,
    pub bc_idx: usize,
    pub row_idx: usize,
    pub col_idx: usize,
    pub actual: Flat<F>,
    pub expected: Flat<F>,
}

pub struct FixedColumnViolation<F> {
    pub table: TableId,
    pub pin_idx: usize,
    pub col_idx: usize,
    pub row_idx: usize,
    pub actual: Flat<F>,
    pub expected: Flat<F>,
}

/// `bus_imbalance` mirrors runtime LogUp closure
/// `Σ_e claimed_sum_e = 0` (char-2). `mismatching_rows`
/// is the stricter Lookup pointwise check.
pub struct BusDiagnostic<F> {
    pub bus_id: String,
    pub kind: BusKind,
    pub endpoints: Vec<BusEndpoint<F>>,

    /// Bus closure failed:
    /// `XOR_e claimed_sum_e != 0` across endpoints.
    /// Topology-agnostic; correct for any N-endpoint
    /// bus (1-SEND + N-1 RECVs, chiplet<>chiplet, etc.).
    pub bus_imbalance: bool,

    /// Lookup-only:
    /// row indices where endpoint `h` values fail the pointwise
    /// XOR-to-zero check. Permutation buses leave this empty
    /// and signal via `bus_imbalance` instead.
    pub mismatching_rows: Vec<usize>,

    /// Endpoints on this bus_id disagree on `BusKind`.
    pub kind_conflict: bool,

    /// Emit rows whose row-index clock sources fold to
    /// one value. The rest of the key is witness-derived.
    pub clock_collisions: Vec<(TableId, usize, usize)>,

    /// No endpoint emits:
    /// balanced, and enforcing nothing.
    pub inert: bool,
}

impl<F> BusDiagnostic<F> {
    pub fn has_failures(&self) -> bool {
        self.bus_imbalance
            || self.kind_conflict
            || self.inert
            || !self.mismatching_rows.is_empty()
            || !self.clock_collisions.is_empty()
    }
}

pub struct BusEndpoint<F> {
    pub source: TableId,
    pub row_count: usize,
    pub active_rows: usize,

    /// `Π_{active row} (γ + key_row)`. Informational
    /// only, pairwise equality of products across
    /// endpoints is the wrong predicate for N ≥ 3.
    pub product: Flat<F>,

    /// `XOR_{active row} 1 / (γ + key_row)` in char-2.
    /// The bus closes iff `XOR_e claimed_sum_e == 0`.
    pub claimed_sum: Flat<F>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TableId {
    Main,
    Chiplet(usize),
}

impl fmt::Display for TableId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TableId::Main => f.write_str("Main"),
            TableId::Chiplet(i) => write!(f, "Chiplet {}", i),
        }
    }
}

pub struct PreflightReport<F> {
    pub constraint_violations: Vec<ConstraintViolation<F>>,
    pub boundary_violations: Vec<BoundaryViolation<F>>,
    pub fixed_column_violations: Vec<FixedColumnViolation<F>>,
    pub bus_diagnostics: Vec<BusDiagnostic<F>>,
}

impl<F> PreflightReport<F> {
    pub fn new() -> Self {
        Self {
            constraint_violations: Vec::new(),
            boundary_violations: Vec::new(),
            fixed_column_violations: Vec::new(),
            bus_diagnostics: Vec::new(),
        }
    }

    pub fn is_clean(&self) -> bool {
        self.constraint_violations.is_empty()
            && self.boundary_violations.is_empty()
            && self.fixed_column_violations.is_empty()
            && self.bus_diagnostics.is_empty()
    }
}

impl<F> Default for PreflightReport<F> {
    fn default() -> Self {
        Self::new()
    }
}

pub fn check_air_constraints<F, P, T>(
    air: &P,
    trace: &T,
    table: TableId,
    report: &mut PreflightReport<F>,
) -> errors::Result<()>
where
    F: TraceCompatibleField,
    P: Air<F>,
    T: Trace,
{
    let (mut constraints, mut pins) = collect_air_violations(air, trace, table)?;
    report.constraint_violations.append(&mut constraints);
    report.fixed_column_violations.append(&mut pins);

    Ok(())
}

struct RowScratch<F: TowerField> {
    current_row: Vec<Flat<F>>,
    next_row: Vec<Flat<F>>,
    eval_buf: Vec<Flat<F>>,
    row_bytes: Vec<u8>,
}

impl<F: TowerField> RowScratch<F> {
    fn new(num_virtual_cols: usize, ast_arena_len: usize, phys_row_bytes: usize) -> Self {
        Self {
            current_row: Vec::with_capacity(num_virtual_cols),
            next_row: Vec::with_capacity(num_virtual_cols),
            eval_buf: Vec::with_capacity(ast_arena_len),
            row_bytes: Vec::with_capacity(phys_row_bytes),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate_air_row<F, P, T>(
    air: &P,
    trace: &T,
    table: TableId,
    ast: &ConstraintAst<F>,
    consts: &[Flat<F>],
    pins: &[FixedColumn<F>],
    num_virtual_cols: usize,
    has_virtual_expansion: bool,
    num_rows: usize,
    row_idx: usize,
    scratch: &mut RowScratch<F>,
    constraints_out: &mut Vec<ConstraintViolation<F>>,
    pins_out: &mut Vec<FixedColumnViolation<F>>,
) -> errors::Result<()>
where
    F: TraceCompatibleField,
    P: Air<F>,
    T: Trace,
{
    let zero = Flat::from_raw(F::ZERO);

    scratch.current_row.clear();
    scratch.next_row.clear();

    let next_idx = (row_idx + 1) % num_rows;
    let columns = trace.columns();

    if has_virtual_expansion {
        extract_row_bytes(columns, row_idx, &mut scratch.row_bytes);
        air.parse_virtual_row(&scratch.row_bytes, &mut scratch.current_row);

        extract_row_bytes(columns, next_idx, &mut scratch.row_bytes);
        air.parse_virtual_row(&scratch.row_bytes, &mut scratch.next_row);
    } else {
        for col in 0..num_virtual_cols {
            scratch
                .current_row
                .push(trace.get_element::<F>(col, row_idx)?);
        }

        for col in 0..num_virtual_cols {
            scratch
                .next_row
                .push(trace.get_element::<F>(col, next_idx)?);
        }
    }

    let num_vars = num_rows.trailing_zeros() as usize;

    for (pin_idx, pin) in pins.iter().enumerate() {
        let actual = scratch.current_row[pin.col_idx];
        let expected = pin.shape.value_at_row(row_idx, num_vars);

        if actual != expected {
            pins_out.push(FixedColumnViolation {
                table,
                pin_idx,
                col_idx: pin.col_idx,
                row_idx,
                actual,
                expected,
            });
        }

        scratch.current_row[pin.col_idx] = expected;
    }

    ast.evaluate_into(
        consts,
        &scratch.current_row,
        &scratch.next_row,
        &mut scratch.eval_buf,
    );

    for (ci, root) in ast.roots.iter().enumerate() {
        let val = scratch.eval_buf[root.0 as usize];
        if val != zero {
            let label = ast.labels.get(ci).copied().flatten();
            constraints_out.push(ConstraintViolation {
                table,
                constraint_idx: ci,
                label,
                row_idx,
                value: val,
            });
        }
    }

    Ok(())
}

fn collect_air_violations<F, P, T>(
    air: &P,
    trace: &T,
    table: TableId,
) -> errors::Result<AirViolations<F>>
where
    F: TraceCompatibleField,
    P: Air<F>,
    T: Trace,
{
    let num_rows = trace.num_rows()?;
    let num_vars = num_rows.trailing_zeros() as usize;
    let num_virtual_cols = air.num_columns();

    let ast = air.constraint_ast();
    let pins = air.fixed_columns();

    validate_fixed_columns(&pins, air.virtual_column_layout(), Some(num_vars))?;

    if ast.roots.is_empty() && pins.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let has_virtual_expansion = air.virtual_expander().is_some();
    let phys_row_bytes: usize = if has_virtual_expansion {
        air.column_layout().iter().map(|c| c.byte_size()).sum()
    } else {
        0
    };

    let ast_arena_len = ast.arena.len();
    let consts = ast.precompute_hardware_consts();

    #[cfg(not(feature = "parallel"))]
    {
        let mut scratch = RowScratch::<F>::new(num_virtual_cols, ast_arena_len, phys_row_bytes);
        let mut constraints: Vec<ConstraintViolation<F>> = Vec::new();
        let mut pin_violations: Vec<FixedColumnViolation<F>> = Vec::new();

        for row_idx in 0..num_rows {
            evaluate_air_row(
                air,
                trace,
                table,
                &ast,
                &consts,
                &pins,
                num_virtual_cols,
                has_virtual_expansion,
                num_rows,
                row_idx,
                &mut scratch,
                &mut constraints,
                &mut pin_violations,
            )?;
        }

        Ok((constraints, pin_violations))
    }

    #[cfg(feature = "parallel")]
    {
        let chunks: errors::Result<Vec<AirViolations<F>>> = (0..num_rows)
            .into_par_iter()
            .try_fold(
                || {
                    (
                        RowScratch::<F>::new(num_virtual_cols, ast_arena_len, phys_row_bytes),
                        Vec::<ConstraintViolation<F>>::new(),
                        Vec::<FixedColumnViolation<F>>::new(),
                    )
                },
                |(mut scratch, mut cs, mut ps), row_idx| {
                    evaluate_air_row(
                        air,
                        trace,
                        table,
                        &ast,
                        &consts,
                        &pins,
                        num_virtual_cols,
                        has_virtual_expansion,
                        num_rows,
                        row_idx,
                        &mut scratch,
                        &mut cs,
                        &mut ps,
                    )?;

                    Ok((scratch, cs, ps))
                },
            )
            .map(|res| res.map(|(_scratch, cs, ps)| (cs, ps)))
            .collect();

        let mut constraints: Vec<ConstraintViolation<F>> = Vec::new();
        let mut pin_violations: Vec<FixedColumnViolation<F>> = Vec::new();

        for (chunk_cs, chunk_ps) in chunks? {
            constraints.extend(chunk_cs);
            pin_violations.extend(chunk_ps);
        }

        Ok((constraints, pin_violations))
    }
}

pub fn extract_row_bytes(columns: &[TraceColumn], row_idx: usize, buf: &mut Vec<u8>) {
    buf.clear();

    for col in columns {
        match col {
            TraceColumn::Bit(v) => buf.push(v[row_idx].get()),
            TraceColumn::B8(v) => buf.push(v[row_idx].into_raw().0),
            TraceColumn::B16(v) => buf.extend_from_slice(&v[row_idx].into_raw().0.to_le_bytes()),
            TraceColumn::B32(v) => buf.extend_from_slice(&v[row_idx].into_raw().0.to_le_bytes()),
            TraceColumn::B64(v) => buf.extend_from_slice(&v[row_idx].into_raw().0.to_le_bytes()),
            TraceColumn::B128(v) => buf.extend_from_slice(&v[row_idx].into_raw().0.to_le_bytes()),
        }
    }
}

pub fn check_boundary_constraints<F, P, T>(
    air: &P,
    instance: &ProgramInstance<F>,
    trace: &T,
    table: TableId,
    report: &mut PreflightReport<F>,
) -> errors::Result<()>
where
    F: TraceCompatibleField,
    P: Air<F>,
    T: Trace,
{
    let bcs = air.boundary_constraints();
    if bcs.is_empty() {
        return Ok(());
    }

    let num_virtual = air.num_columns();
    let num_rows = trace.num_rows()?;
    let has_expansion = air.virtual_expander().is_some();
    let columns = trace.columns();

    let (mut row_bytes, mut virtual_row) = if has_expansion {
        let phys_row_bytes: usize = air.column_layout().iter().map(|c| c.byte_size()).sum();
        (
            Vec::<u8>::with_capacity(phys_row_bytes),
            Vec::<Flat<F>>::with_capacity(num_virtual),
        )
    } else {
        (Vec::new(), Vec::new())
    };

    for (bc_idx, bc) in bcs.iter().enumerate() {
        if bc.col_idx >= num_virtual {
            return Err(errors::Error::Protocol {
                protocol: "boundary",
                message: "boundary col_idx out of range",
            });
        }

        if bc.row_idx >= num_rows {
            return Err(errors::Error::Protocol {
                protocol: "boundary",
                message: "boundary row_idx exceeds trace height",
            });
        }

        let actual: Flat<F> = if has_expansion {
            extract_row_bytes(columns, bc.row_idx, &mut row_bytes);

            virtual_row.clear();

            air.parse_virtual_row(&row_bytes, &mut virtual_row);

            virtual_row[bc.col_idx]
        } else {
            trace.get_element(bc.col_idx, bc.row_idx)?
        };

        let expected = bc.resolve_target(instance).unwrap_or(F::ZERO).to_hardware();

        if actual != expected {
            report.boundary_violations.push(BoundaryViolation {
                table,
                bc_idx,
                row_idx: bc.row_idx,
                col_idx: bc.col_idx,
                actual,
                expected,
            });
        }
    }

    Ok(())
}

pub fn check_chiplet_constraints<F>(
    chiplet_defs: &[ChipletDef<F>],
    chiplet_traces: &[hekate_core::trace::ColumnTrace],
    report: &mut PreflightReport<F>,
) -> errors::Result<()>
where
    F: TraceCompatibleField,
{
    let empty_instance = ProgramInstance::<F>::new(1, vec![]);

    for (idx, (def, trace)) in chiplet_defs.iter().zip(chiplet_traces.iter()).enumerate() {
        check_boundary_constraints(def, &empty_instance, trace, TableId::Chiplet(idx), report)?;
    }

    #[cfg(not(feature = "parallel"))]
    {
        for (idx, (def, trace)) in chiplet_defs.iter().zip(chiplet_traces.iter()).enumerate() {
            check_air_constraints(def, trace, TableId::Chiplet(idx), report)?;
        }
    }

    #[cfg(feature = "parallel")]
    {
        let per_chiplet: Vec<errors::Result<AirViolations<F>>> = chiplet_defs
            .par_iter()
            .zip(chiplet_traces.par_iter())
            .enumerate()
            .map(|(idx, (def, trace))| collect_air_violations(def, trace, TableId::Chiplet(idx)))
            .collect();

        for result in per_chiplet {
            let (cs, ps) = result?;
            report.constraint_violations.extend(cs);
            report.fixed_column_violations.extend(ps);
        }
    }

    Ok(())
}

// =================================================================
// BUS MULTISET DIAGNOSTIC
// =================================================================

struct BusEndpointAccum<F> {
    source: TableId,
    kind: BusKind,
    row_count: usize,
    active_rows: usize,
    product: Flat<F>,
    claimed_sum: Flat<F>,

    /// Empty unless `kind == Lookup`.
    h_rows: Vec<Flat<F>>,

    clock_collision: Option<(usize, usize)>,
}

struct EndpointStats<F> {
    product: Flat<F>,
    claimed_sum: Flat<F>,
    row_count: usize,
    active_rows: usize,
    h_rows: Vec<Flat<F>>,
    clock_collision: Option<(usize, usize)>,
}

/// Fixed γ, β for char-2 bus closure check;
/// SZ collisions are 2^-128 rare.
fn fixed_gamma<F: TraceCompatibleField>() -> Flat<F> {
    F::from(0x9e3779b97f4a7c15u128).to_hardware()
}

fn fixed_beta<F: TraceCompatibleField>() -> Flat<F> {
    F::from(0x517cc1b727220a95u128).to_hardware()
}

fn resolve_source<F: TraceCompatibleField>(
    source: &Source,
    row: &[Flat<F>],
    row_idx: usize,
    rank: Flat<F>,
    beta: Flat<F>,
    current_beta: Flat<F>,
) -> (Flat<F>, Flat<F>) {
    match source {
        Source::Column(col_idx) | Source::PhaseColumn(col_idx) => {
            let val = row[*col_idx];
            (val * current_beta, current_beta * beta)
        }
        Source::Columns(col_indices) => {
            let mut acc = Flat::from_raw(F::ZERO);
            let mut curr = current_beta;

            for &idx in col_indices {
                acc += row[idx] * curr;
                curr *= beta;
            }

            (acc, curr)
        }
        Source::RowIndexLeBytes(num_bytes) => {
            let limit = (*num_bytes).min(8);

            let mut val: u128 = 0;
            for b in 0..limit {
                let byte_val = ((row_idx >> (8 * b)) & 0xFF) as u128;
                val += byte_val << (8 * b);
            }

            let v = F::from(val).to_hardware();

            (v * current_beta, current_beta * beta)
        }
        Source::RowIndexByte(byte_k) => {
            let byte_val = ((row_idx >> (8 * byte_k)) & 0xFF) as u128;
            let v = F::from(byte_val).to_hardware();

            (v * current_beta, current_beta * beta)
        }
        Source::Const(val) => {
            let v = F::from(*val).to_hardware();
            (v * current_beta, current_beta * beta)
        }
        Source::EmitRank(_) => (rank * current_beta, current_beta * beta),
    }
}

/// Compute `Π(γ + key)`, `claimed_sum = XOR
/// 1/(γ + key)` over active rows, and per-row
/// `h[i] = s_eff[i] / (γ + key[i])` for
/// `BusKind::Lookup` specs (empty otherwise).
/// `s_eff = s_send + s_recv` in char-2 when paired.
fn compute_endpoint_product<F, A, T>(
    spec: &PermutationCheckSpec,
    clock: Option<&RankClock>,
    air: &A,
    trace: &T,
    gamma: Flat<F>,
    beta: Flat<F>,
) -> errors::Result<EndpointStats<F>>
where
    F: TraceCompatibleField,
    A: Air<F>,
    T: Trace,
{
    let num_rows = trace.num_rows()?;
    let num_virtual = air.num_columns();

    let has_expansion = air.virtual_expander().is_some();
    let want_h = spec.kind == BusKind::Lookup;

    let zero = Flat::from_raw(F::ZERO);
    let one = Flat::from_raw(F::ONE);

    let has_rank = spec
        .sources
        .iter()
        .any(|(source, _)| matches!(source, Source::EmitRank(_)));

    let clock_column = match (has_rank, clock) {
        (false, _) => Vec::new(),
        (true, Some(clock)) => {
            let mut column = vec![zero; num_rows];
            clock.accumulate(&mut column, one)?;

            column
        }
        (true, None) => {
            return Err(errors::Error::Protocol {
                protocol: "preflight",
                message: "ordered bus spec has no rank clock",
            });
        }
    };

    let mut product = one;
    let mut claimed_sum = zero;
    let mut active_rows = 0usize;

    let clock_mask = separating_clock_mask(spec, num_rows.trailing_zeros() as usize);

    let mut clock_seen: BTreeMap<usize, usize> = BTreeMap::new();
    let mut clock_collision: Option<(usize, usize)> = None;

    let mut row_vec: Vec<Flat<F>> = Vec::with_capacity(num_virtual);
    let mut row_bytes: Vec<u8> = Vec::new();

    let mut h_rows: Vec<Flat<F>> = if want_h {
        Vec::with_capacity(num_rows)
    } else {
        Vec::new()
    };

    if has_expansion {
        let phys_row_bytes: usize = air.column_layout().iter().map(|c| c.byte_size()).sum();
        row_bytes = Vec::with_capacity(phys_row_bytes);
    }

    let columns = trace.columns();

    for row_idx in 0..num_rows {
        row_vec.clear();

        if has_expansion {
            extract_row_bytes(columns, row_idx, &mut row_bytes);
            air.parse_virtual_row(&row_bytes, &mut row_vec);
        } else {
            for col in 0..num_virtual {
                row_vec.push(trace.get_element::<F>(col, row_idx)?);
            }
        }

        let s_send = match spec.selector {
            Some(sel_col) => row_vec[sel_col],
            None => one,
        };

        let s_recv = match spec.recv_selector {
            Some(sel_col) => row_vec[sel_col],
            None => zero,
        };

        let selector_val = s_send + s_recv;

        if selector_val == zero {
            if want_h {
                h_rows.push(zero);
            }

            continue;
        }

        active_rows += 1;

        if let Some(mask) = clock_mask
            && let Some(prev) = clock_seen.insert(row_idx & mask, row_idx)
            && clock_collision.is_none()
        {
            clock_collision = Some((prev, row_idx));
        }

        let mut key = gamma;
        let mut current_beta = one;

        let rank = clock_column.get(row_idx).copied().unwrap_or(zero);

        for (source, _label) in &spec.sources {
            let (contrib, next_beta) =
                resolve_source(source, &row_vec, row_idx, rank, beta, current_beta);

            key += contrib;
            current_beta = next_beta;
        }

        product *= key;

        // invert(0) = 0 by hekate-math convention.
        // Preflight γ is fixed (not random), so a
        // user trace engineered to land on `key = 0`
        // will deterministically produce h = 0 here;
        // caller responsibility to avoid that input.
        let inv = key.to_tower().invert().to_hardware();
        let h_row = selector_val * inv;

        claimed_sum += h_row;

        if want_h {
            h_rows.push(h_row);
        }
    }

    Ok(EndpointStats {
        product,
        claimed_sum,
        row_count: num_rows,
        active_rows,
        h_rows,
        clock_collision,
    })
}

pub fn check_bus_multisets<F, A, T>(
    main: &A,
    chiplet_defs: &[ChipletDef<F>],
    witness: &ProgramWitness<F, T>,
    report: &mut PreflightReport<F>,
) -> errors::Result<()>
where
    F: TraceCompatibleField,
    A: Air<F> + Sync,
    T: Trace,
{
    let gamma = fixed_gamma::<F>();
    let beta = fixed_beta::<F>();

    if chiplet_defs.len() != witness.chiplet_traces.len() {
        return Err(errors::Error::Protocol {
            protocol: "preflight",
            message: "chiplet trace count differs from chiplet_defs",
        });
    }

    let main_perm_checks = main.permutation_checks();
    let main_fixed = main.fixed_columns();

    let mut rank_tables = Vec::with_capacity(1 + chiplet_defs.len());

    rank_tables.push(RankTable {
        specs: &main_perm_checks,
        fixed: &main_fixed,
        height: TableHeight::Main(witness.trace.num_vars()),
    });

    for (def, trace) in chiplet_defs.iter().zip(&witness.chiplet_traces) {
        rank_tables.push(RankTable {
            specs: &def.permutation_checks,
            fixed: def.pins(),
            height: TableHeight::Chiplet(Some(trace.num_vars())),
        });
    }

    let clocks = rank_clocks(&rank_tables)?;
    let clocks = &clocks;

    #[cfg(not(feature = "parallel"))]
    let main_endpoints: Vec<(String, BusEndpointAccum<F>)> = main_perm_checks
        .iter()
        .enumerate()
        .map(|(i, (bus_id, spec))| {
            let clock = clocks[0][i].as_ref();
            let stats = compute_endpoint_product(spec, clock, main, &witness.trace, gamma, beta)?;

            Ok((
                bus_id.clone(),
                BusEndpointAccum {
                    source: TableId::Main,
                    kind: spec.kind,
                    row_count: stats.row_count,
                    active_rows: stats.active_rows,
                    product: stats.product,
                    claimed_sum: stats.claimed_sum,
                    h_rows: stats.h_rows,
                    clock_collision: stats.clock_collision,
                },
            ))
        })
        .collect::<errors::Result<_>>()?;

    #[cfg(feature = "parallel")]
    let main_endpoints: Vec<(String, BusEndpointAccum<F>)> = main_perm_checks
        .par_iter()
        .enumerate()
        .map(|(i, (bus_id, spec))| {
            let clock = clocks[0][i].as_ref();
            let stats = compute_endpoint_product(spec, clock, main, &witness.trace, gamma, beta)?;

            Ok((
                bus_id.clone(),
                BusEndpointAccum {
                    source: TableId::Main,
                    kind: spec.kind,
                    row_count: stats.row_count,
                    active_rows: stats.active_rows,
                    product: stats.product,
                    claimed_sum: stats.claimed_sum,
                    h_rows: stats.h_rows,
                    clock_collision: stats.clock_collision,
                },
            ))
        })
        .collect::<errors::Result<_>>()?;

    #[cfg(not(feature = "parallel"))]
    let chiplet_endpoints: Vec<(String, BusEndpointAccum<F>)> = chiplet_defs
        .iter()
        .zip(witness.chiplet_traces.iter())
        .enumerate()
        .flat_map(|(c_idx, (def, trace))| {
            def.permutation_checks
                .iter()
                .enumerate()
                .map(move |(i, (bus_id, spec))| {
                    let clock = clocks[1 + c_idx][i].as_ref();
                    let stats = compute_endpoint_product(spec, clock, def, trace, gamma, beta)?;

                    Ok((
                        bus_id.clone(),
                        BusEndpointAccum {
                            source: TableId::Chiplet(c_idx),
                            kind: spec.kind,
                            row_count: stats.row_count,
                            active_rows: stats.active_rows,
                            product: stats.product,
                            claimed_sum: stats.claimed_sum,
                            h_rows: stats.h_rows,
                            clock_collision: stats.clock_collision,
                        },
                    ))
                })
        })
        .collect::<errors::Result<_>>()?;

    #[cfg(feature = "parallel")]
    let chiplet_endpoints: Vec<(String, BusEndpointAccum<F>)> = chiplet_defs
        .par_iter()
        .zip(witness.chiplet_traces.par_iter())
        .enumerate()
        .flat_map_iter(|(c_idx, (def, trace))| {
            def.permutation_checks
                .iter()
                .enumerate()
                .map(move |(i, (bus_id, spec))| {
                    let clock = clocks[1 + c_idx][i].as_ref();
                    let stats = compute_endpoint_product(spec, clock, def, trace, gamma, beta)?;

                    Ok((
                        bus_id.clone(),
                        BusEndpointAccum {
                            source: TableId::Chiplet(c_idx),
                            kind: spec.kind,
                            row_count: stats.row_count,
                            active_rows: stats.active_rows,
                            product: stats.product,
                            claimed_sum: stats.claimed_sum,
                            h_rows: stats.h_rows,
                            clock_collision: stats.clock_collision,
                        },
                    ))
                })
        })
        .collect::<errors::Result<_>>()?;

    let mut bus_map: HashMap<String, Vec<BusEndpointAccum<F>>> = HashMap::new();
    for (bus_id, endpoint) in main_endpoints {
        bus_map.entry(bus_id).or_default().push(endpoint);
    }

    for (bus_id, endpoint) in chiplet_endpoints {
        bus_map.entry(bus_id).or_default().push(endpoint);
    }

    let zero = Flat::from_raw(F::ZERO);

    for (bus_id, endpoints) in &bus_map {
        // Runtime LogUp closure:
        // `Σ_e claimed_sum_e = 0` (char-2).
        // Topology-agnostic for any N ≥ 2.
        let bus_imbalance = endpoints.iter().fold(zero, |acc, e| acc + e.claimed_sum) != zero;

        let kind_conflict = !endpoints.windows(2).all(|w| w[0].kind == w[1].kind);

        let bus_kind = endpoints.first().map(|e| e.kind).unwrap_or_default();

        // Lookup buses additionally require
        // pointwise `XOR_k h_k[i] = 0` on
        // the padded hypercube.
        let mismatching_rows = if bus_kind == BusKind::Lookup && !kind_conflict {
            find_lookup_mismatch_rows(endpoints)
        } else {
            Vec::new()
        };

        let clock_collisions: Vec<(TableId, usize, usize)> = endpoints
            .iter()
            .filter_map(|e| e.clock_collision.map(|(a, b)| (e.source, a, b)))
            .collect();

        let inert = endpoints.iter().all(|e| e.active_rows == 0);

        let diag = BusDiagnostic {
            bus_id: bus_id.clone(),
            kind: bus_kind,
            endpoints: endpoints
                .iter()
                .map(|e| BusEndpoint {
                    source: e.source,
                    row_count: e.row_count,
                    active_rows: e.active_rows,
                    product: e.product,
                    claimed_sum: e.claimed_sum,
                })
                .collect(),
            bus_imbalance,
            mismatching_rows,
            kind_conflict,
            clock_collisions,
            inert,
        };

        if diag.has_failures() {
            report.bus_diagnostics.push(diag);
        }
    }

    Ok(())
}

fn find_lookup_mismatch_rows<F: TraceCompatibleField>(
    endpoints: &[BusEndpointAccum<F>],
) -> Vec<usize> {
    let zero = Flat::from_raw(F::ZERO);

    let n_max = endpoints.iter().map(|e| e.h_rows.len()).max().unwrap_or(0);

    let mut rows = Vec::new();

    for i in 0..n_max {
        let mut xor_sum = zero;
        for e in endpoints {
            // Shorter endpoints are zero-padded
            // up to `n_max`, padding contributes 0.
            if i < e.h_rows.len() {
                xor_sum += e.h_rows[i];
            }
        }

        if xor_sum != zero {
            rows.push(i);

            if rows.len() >= MAX_REPORTED_MISMATCH_ROWS {
                break;
            }
        }
    }

    rows
}

// =================================================================
// TOP-LEVEL API
// =================================================================

pub fn preflight<F, P, T>(
    program: &P,
    instance: &ProgramInstance<F>,
    witness: &ProgramWitness<F, T>,
) -> errors::Result<PreflightReport<F>>
where
    F: TraceCompatibleField + Into<Block128>,
    P: Program<F>,
    T: Trace,
{
    let mut report = PreflightReport::new();

    let chiplet_defs = program.chiplet_defs()?;

    check_air_constraints(program, &witness.trace, TableId::Main, &mut report)?;
    check_boundary_constraints(
        program,
        instance,
        &witness.trace,
        TableId::Main,
        &mut report,
    )?;
    check_chiplet_constraints(&chiplet_defs, &witness.chiplet_traces, &mut report)?;
    check_bus_multisets(program, &chiplet_defs, witness, &mut report)?;

    Ok(report)
}

// =================================================================
// Formatted preflight report
// =================================================================

impl<F: TraceCompatibleField> fmt::Display for PreflightReport<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "PREFLIGHT: {} constraint, {} boundary, {} fixed column, {} bus",
            self.constraint_violations.len(),
            self.boundary_violations.len(),
            self.fixed_column_violations.len(),
            self.bus_diagnostics.len(),
        )?;

        write_grouped_constraint_violations(f, &self.constraint_violations)?;

        for v in &self.boundary_violations {
            writeln!(
                f,
                "  [{}] boundary #{}: col={} row={} actual={:?} expected={:?}",
                v.table, v.bc_idx, v.col_idx, v.row_idx, v.actual, v.expected,
            )?;
        }

        for v in &self.fixed_column_violations {
            writeln!(
                f,
                "  [{}] fixed column #{}: col={} row={} actual={:?} expected={:?}",
                v.table, v.pin_idx, v.col_idx, v.row_idx, v.actual, v.expected,
            )?;
        }

        for d in &self.bus_diagnostics {
            fmt::Display::fmt(d, f)?;
        }

        Ok(())
    }
}

impl<F: TraceCompatibleField> fmt::Display for BusDiagnostic<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self.kind {
            BusKind::Permutation => "Permutation",
            BusKind::Lookup => "Lookup",
        };

        let mut failures: Vec<String> = Vec::new();

        if self.bus_imbalance {
            failures.push("bus_imbalance".to_string());
        }

        if self.kind_conflict {
            failures.push("kind_conflict".to_string());
        }

        if !self.mismatching_rows.is_empty() {
            failures.push(format!("mismatching_rows={}", self.mismatching_rows.len()));
        }

        if !self.clock_collisions.is_empty() {
            failures.push(format!("clock_collisions={}", self.clock_collisions.len()));
        }

        if self.inert {
            failures.push("inert".to_string());
        }

        writeln!(
            f,
            "  bus \"{}\" [{}] ({} endpoints) failures: {}",
            self.bus_id,
            kind,
            self.endpoints.len(),
            failures.join(", "),
        )?;

        for e in &self.endpoints {
            writeln!(
                f,
                "    {}: {} rows, {} active, product={:?}, claimed_sum={:?}",
                e.source, e.row_count, e.active_rows, e.product, e.claimed_sum,
            )?;
        }

        if !self.mismatching_rows.is_empty() {
            let total = self.mismatching_rows.len();
            let shown = total.min(FMT_MAX_MISMATCH_PREVIEW);

            writeln!(
                f,
                "    mismatching rows (first {} of {}): {:?}",
                shown,
                total,
                &self.mismatching_rows[..shown],
            )?;
        }

        for (table, a, b) in &self.clock_collisions {
            writeln!(
                f,
                "    [{}] emit rows {} and {} share a clock value; \
                 widen the row-index source or make the keys differ",
                table, a, b,
            )?;
        }

        if self.inert {
            writeln!(
                f,
                "    every endpoint is switched off on every row; \
                 the selector shapes pin no emit; this bus binds nothing",
            )?;
        }

        Ok(())
    }
}

fn write_grouped_constraint_violations<F>(
    f: &mut fmt::Formatter<'_>,
    violations: &[ConstraintViolation<F>],
) -> fmt::Result {
    let mut grouped: GroupedConstraintRows = BTreeMap::new();

    for v in violations {
        let entry = grouped
            .entry((v.table, v.constraint_idx))
            .or_insert_with(|| (v.label, Vec::new()));
        entry.1.push(v.row_idx);
    }

    for ((table, ci), (label, rows)) in &grouped {
        let label = label.unwrap_or("(unnamed)");
        let shown = rows.len().min(FMT_MAX_ROW_PREVIEW);

        writeln!(
            f,
            "  [{}] constraint #{} \"{}\" — {} rows: {:?}",
            table,
            ci,
            label,
            rows.len(),
            &rows[..shown],
        )?;
    }

    Ok(())
}

/// Row-index bits the spec's clock sources actually
/// fold, `None` when they already separate every row.
fn separating_clock_mask(spec: &PermutationCheckSpec, num_vars: usize) -> Option<usize> {
    let all = match num_vars >= usize::BITS as usize {
        true => usize::MAX,
        false => (1usize << num_vars) - 1,
    };

    let mut mask = 0usize;
    let mut has_clock = false;

    for (source, _) in &spec.sources {
        let (lo, hi) = match source {
            Source::RowIndexLeBytes(n) => (0, 8 * (*n).min(8)),
            Source::RowIndexByte(k) => (8 * *k, 8 * *k + 8),
            Source::EmitRank(_) => return None,
            Source::Column(_) | Source::Columns(_) | Source::Const(_) | Source::PhaseColumn(_) => {
                continue;
            }
        };

        has_clock = true;

        for bit in lo..hi.min(num_vars) {
            mask |= 1usize << bit;
        }
    }

    match has_clock && mask != all {
        true => Some(mask),
        false => None,
    }
}
