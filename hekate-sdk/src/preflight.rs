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

use alloc::string::String;
use alloc::vec::Vec;
use hashbrown::HashMap;
use hekate_core::errors;
use hekate_core::trace::{Trace, TraceColumn, TraceCompatibleField};
use hekate_math::{Block128, Flat, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::ConstraintAst;
use hekate_program::permutation::{BusKind, PermutationCheckSpec, Source};
use hekate_program::{
    Air, LagrangePin, Program, ProgramInstance, ProgramWitness, validate_lagrange_pins,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

const MAX_REPORTED_MISMATCH_ROWS: usize = 32;

type AirViolations<F> = (Vec<ConstraintViolation<F>>, Vec<LagrangePinViolation<F>>);

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

pub struct LagrangePinViolation<F> {
    pub table: TableId,
    pub pin_idx: usize,
    pub col_idx: usize,
    pub row_idx: usize,
    pub actual: Flat<F>,
    pub expected: Flat<F>,
}

pub struct BusDiagnostic<F> {
    pub bus_id: String,
    pub kind: BusKind,
    pub endpoints: Vec<BusEndpoint<F>>,

    /// Lookup-only:
    /// row indices where endpoint `h`
    /// values don't XOR to zero.
    pub mismatching_rows: Vec<usize>,

    /// Endpoints on this bus_id
    /// disagree on `BusKind`.
    pub kind_conflict: bool,

    /// Paired-spec rows where `s_send · s_recv = 1`;
    /// the AIR mutex constraint must reject these.
    pub selector_mutex_violations: Vec<(TableId, usize)>,
}

pub struct BusEndpoint<F> {
    pub source: TableId,
    pub row_count: usize,
    pub active_rows: usize,
    pub product: Flat<F>,
}

#[derive(Clone, Copy)]
pub enum TableId {
    Main,
    Chiplet(usize),
}

pub struct PreflightReport<F> {
    pub constraint_violations: Vec<ConstraintViolation<F>>,
    pub boundary_violations: Vec<BoundaryViolation<F>>,
    pub lagrange_pin_violations: Vec<LagrangePinViolation<F>>,
    pub bus_diagnostics: Vec<BusDiagnostic<F>>,
}

impl<F> PreflightReport<F> {
    pub fn new() -> Self {
        Self {
            constraint_violations: Vec::new(),
            boundary_violations: Vec::new(),
            lagrange_pin_violations: Vec::new(),
            bus_diagnostics: Vec::new(),
        }
    }

    pub fn is_clean(&self) -> bool {
        self.constraint_violations.is_empty()
            && self.boundary_violations.is_empty()
            && self.lagrange_pin_violations.is_empty()
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
    report.lagrange_pin_violations.append(&mut pins);

    Ok(())
}

struct RowScratch<F: TowerField> {
    current_row: Vec<Flat<F>>,
    next_row: Vec<Flat<F>>,
    eval_buf: Vec<Flat<F>>,
    row_bytes: Vec<u8>,
    row_bits: Vec<Flat<F>>,
}

impl<F: TowerField> RowScratch<F> {
    fn new(
        num_virtual_cols: usize,
        ast_arena_len: usize,
        phys_row_bytes: usize,
        num_vars: usize,
    ) -> Self {
        Self {
            current_row: Vec::with_capacity(num_virtual_cols),
            next_row: Vec::with_capacity(num_virtual_cols),
            eval_buf: Vec::with_capacity(ast_arena_len),
            row_bytes: Vec::with_capacity(phys_row_bytes),
            row_bits: Vec::with_capacity(num_vars),
        }
    }
}

/// Fill `dst` with the LSB-first bit decomposition
/// of `row_idx`, promoted to field elements.
fn fill_row_bits<F: TowerField>(dst: &mut Vec<Flat<F>>, row_idx: usize, num_vars: usize) {
    dst.clear();

    let zero = Flat::from_raw(F::ZERO);
    let one = Flat::from_raw(F::ONE);

    for k in 0..num_vars {
        if (row_idx >> k) & 1 == 1 {
            dst.push(one);
        } else {
            dst.push(zero);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate_air_row<F, P, T>(
    air: &P,
    trace: &T,
    table: TableId,
    ast: &ConstraintAst<F>,
    pins: &[LagrangePin],
    row_bits: &[Flat<F>],
    num_virtual_cols: usize,
    has_virtual_expansion: bool,
    num_rows: usize,
    row_idx: usize,
    scratch: &mut RowScratch<F>,
    constraints_out: &mut Vec<ConstraintViolation<F>>,
    pins_out: &mut Vec<LagrangePinViolation<F>>,
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

    for (pin_idx, pin) in pins.iter().enumerate() {
        let actual = scratch.current_row[pin.col_idx];
        let expected = pin.point.evaluate::<F>(row_bits);

        if actual != expected {
            pins_out.push(LagrangePinViolation {
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
    let num_physical_cols = trace.num_cols();

    let ast = air.constraint_ast();
    let pins = air.lagrange_pinned_columns();

    validate_lagrange_pins(&pins, num_virtual_cols, Some(num_vars))?;

    if ast.roots.is_empty() && pins.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let has_virtual_expansion = num_virtual_cols != num_physical_cols;
    let phys_row_bytes: usize = if has_virtual_expansion {
        air.column_layout().iter().map(|c| c.byte_size()).sum()
    } else {
        0
    };

    let ast_arena_len = ast.arena.len();

    #[cfg(not(feature = "parallel"))]
    {
        let mut scratch =
            RowScratch::<F>::new(num_virtual_cols, ast_arena_len, phys_row_bytes, num_vars);
        let mut constraints: Vec<ConstraintViolation<F>> = Vec::new();
        let mut pin_violations: Vec<LagrangePinViolation<F>> = Vec::new();

        for row_idx in 0..num_rows {
            fill_row_bits::<F>(&mut scratch.row_bits, row_idx, num_vars);

            // Borrow split:
            // row_bits read, current_row written.
            let row_bits = core::mem::take(&mut scratch.row_bits);

            let res = evaluate_air_row(
                air,
                trace,
                table,
                &ast,
                &pins,
                &row_bits,
                num_virtual_cols,
                has_virtual_expansion,
                num_rows,
                row_idx,
                &mut scratch,
                &mut constraints,
                &mut pin_violations,
            );

            scratch.row_bits = row_bits;

            res?;
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
                        RowScratch::<F>::new(
                            num_virtual_cols,
                            ast_arena_len,
                            phys_row_bytes,
                            num_vars,
                        ),
                        Vec::<ConstraintViolation<F>>::new(),
                        Vec::<LagrangePinViolation<F>>::new(),
                    )
                },
                |(mut scratch, mut cs, mut ps), row_idx| {
                    fill_row_bits::<F>(&mut scratch.row_bits, row_idx, num_vars);

                    let row_bits = core::mem::take(&mut scratch.row_bits);

                    let res = evaluate_air_row(
                        air,
                        trace,
                        table,
                        &ast,
                        &pins,
                        &row_bits,
                        num_virtual_cols,
                        has_virtual_expansion,
                        num_rows,
                        row_idx,
                        &mut scratch,
                        &mut cs,
                        &mut ps,
                    );

                    scratch.row_bits = row_bits;

                    res?;

                    Ok((scratch, cs, ps))
                },
            )
            .map(|res| res.map(|(_scratch, cs, ps)| (cs, ps)))
            .collect();

        let mut constraints: Vec<ConstraintViolation<F>> = Vec::new();
        let mut pin_violations: Vec<LagrangePinViolation<F>> = Vec::new();

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
            TraceColumn::Bit(v) => buf.push(v[row_idx].0),
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
    for (bc_idx, bc) in air.boundary_constraints().iter().enumerate() {
        let actual: Flat<F> = trace.get_element(bc.col_idx, bc.row_idx)?;
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
    let empty_instance = ProgramInstance::<F>::new(1, alloc::vec![]);

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
            report.lagrange_pin_violations.extend(ps);
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

    /// Empty unless `kind == Lookup`.
    h_rows: Vec<Flat<F>>,

    /// Paired-spec rows with both selectors high
    /// (capped at `MAX_REPORTED_MISMATCH_ROWS`).
    mutex_violations: Vec<usize>,
}

struct EndpointStats<F> {
    product: Flat<F>,
    row_count: usize,
    active_rows: usize,
    h_rows: Vec<Flat<F>>,
    mutex_violations: Vec<usize>,
}

/// Fixed non-cryptographic challenges for
/// bus multiset detection. Equivalent to
/// LogUp bus-matching at runtime:
/// if two endpoints share a bus_id their
/// selected row multisets must coincide,
/// which makes their `Π (γ + key)` products
/// equal by Schwartz-Zippel.
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
    beta: Flat<F>,
    current_beta: Flat<F>,
) -> (Flat<F>, Flat<F>) {
    match source {
        Source::Column(col_idx) => {
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
    }
}

/// Compute `Π(γ + key)` over active rows,
/// plus per-row `h[i] = s_eff[i] / (γ + key[i])`
/// for `BusKind::Lookup` specs (empty otherwise).
/// `s_eff = s_send + s_recv` in char-2 when paired.
fn compute_endpoint_product<F, A, T>(
    spec: &PermutationCheckSpec,
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
    let num_physical = trace.num_cols();

    let has_expansion = num_virtual != num_physical;
    let want_h = spec.kind == BusKind::Lookup;

    let zero = Flat::from_raw(F::ZERO);
    let one = Flat::from_raw(F::ONE);

    let mut product = one;
    let mut active_rows = 0usize;

    let mut row_vec: Vec<Flat<F>> = Vec::with_capacity(num_virtual);
    let mut row_bytes: Vec<u8> = Vec::new();

    let mut h_rows: Vec<Flat<F>> = if want_h {
        Vec::with_capacity(num_rows)
    } else {
        Vec::new()
    };

    let mut mutex_violations: Vec<usize> = Vec::new();

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

        if spec.recv_selector.is_some()
            && s_send == one
            && s_recv == one
            && mutex_violations.len() < MAX_REPORTED_MISMATCH_ROWS
        {
            mutex_violations.push(row_idx);
        }

        let selector_val = s_send + s_recv;

        if selector_val == zero {
            if want_h {
                h_rows.push(zero);
            }

            continue;
        }

        active_rows += 1;

        let mut key = gamma;
        let mut current_beta = one;

        for (source, _label) in &spec.sources {
            let (contrib, next_beta) =
                resolve_source(source, &row_vec, row_idx, beta, current_beta);

            key += contrib;
            current_beta = next_beta;
        }

        product *= key;

        if want_h {
            // invert(0) = 0 by hekate-math convention.
            // Preflight γ is fixed (not random), so a
            // user trace engineered to land on `key = 0`
            // will deterministically produce h = 0 here;
            // caller responsibility to avoid that input.
            let inv = key.to_tower().invert().to_hardware();
            h_rows.push(selector_val * inv);
        }
    }

    Ok(EndpointStats {
        product,
        row_count: num_rows,
        active_rows,
        h_rows,
        mutex_violations,
    })
}

pub fn check_bus_multisets<F, P, T>(
    program: &P,
    witness: &ProgramWitness<F, T>,
    report: &mut PreflightReport<F>,
) -> errors::Result<()>
where
    F: TraceCompatibleField,
    P: Program<F>,
    T: Trace,
{
    let gamma = fixed_gamma::<F>();
    let beta = fixed_beta::<F>();

    let main_perm_checks = program.permutation_checks();
    let chiplet_defs = program.chiplet_defs()?;

    #[cfg(not(feature = "parallel"))]
    let main_endpoints: Vec<(String, BusEndpointAccum<F>)> = main_perm_checks
        .iter()
        .map(|(bus_id, spec)| {
            let stats = compute_endpoint_product(spec, program, &witness.trace, gamma, beta)?;

            Ok((
                bus_id.clone(),
                BusEndpointAccum {
                    source: TableId::Main,
                    kind: spec.kind,
                    row_count: stats.row_count,
                    active_rows: stats.active_rows,
                    product: stats.product,
                    h_rows: stats.h_rows,
                    mutex_violations: stats.mutex_violations,
                },
            ))
        })
        .collect::<errors::Result<_>>()?;

    #[cfg(feature = "parallel")]
    let main_endpoints: Vec<(String, BusEndpointAccum<F>)> = main_perm_checks
        .par_iter()
        .map(|(bus_id, spec)| {
            let stats = compute_endpoint_product(spec, program, &witness.trace, gamma, beta)?;
            Ok((
                bus_id.clone(),
                BusEndpointAccum {
                    source: TableId::Main,
                    kind: spec.kind,
                    row_count: stats.row_count,
                    active_rows: stats.active_rows,
                    product: stats.product,
                    h_rows: stats.h_rows,
                    mutex_violations: stats.mutex_violations,
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
            def.permutation_checks.iter().map(move |(bus_id, spec)| {
                let stats = compute_endpoint_product(spec, def, trace, gamma, beta)?;
                Ok((
                    bus_id.clone(),
                    BusEndpointAccum {
                        source: TableId::Chiplet(c_idx),
                        kind: spec.kind,
                        row_count: stats.row_count,
                        active_rows: stats.active_rows,
                        product: stats.product,
                        h_rows: stats.h_rows,
                        mutex_violations: stats.mutex_violations,
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
            def.permutation_checks.iter().map(move |(bus_id, spec)| {
                let stats = compute_endpoint_product(spec, def, trace, gamma, beta)?;
                Ok((
                    bus_id.clone(),
                    BusEndpointAccum {
                        source: TableId::Chiplet(c_idx),
                        kind: spec.kind,
                        row_count: stats.row_count,
                        active_rows: stats.active_rows,
                        product: stats.product,
                        h_rows: stats.h_rows,
                        mutex_violations: stats.mutex_violations,
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

    for (bus_id, endpoints) in &bus_map {
        // All endpoints sharing a bus_id must
        // have identical Π(γ + key) products.
        let product_mismatch = !endpoints.windows(2).all(|w| w[0].product == w[1].product);

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

        let selector_mutex_violations: Vec<(TableId, usize)> = endpoints
            .iter()
            .flat_map(|e| e.mutex_violations.iter().map(|row| (e.source, *row)))
            .take(MAX_REPORTED_MISMATCH_ROWS)
            .collect();

        if product_mismatch
            || kind_conflict
            || !mismatching_rows.is_empty()
            || !selector_mutex_violations.is_empty()
        {
            report.bus_diagnostics.push(BusDiagnostic {
                bus_id: bus_id.clone(),
                kind: bus_kind,
                endpoints: endpoints
                    .iter()
                    .map(|e| BusEndpoint {
                        source: e.source,
                        row_count: e.row_count,
                        active_rows: e.active_rows,
                        product: e.product,
                    })
                    .collect(),
                mismatching_rows,
                kind_conflict,
                selector_mutex_violations,
            });
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

    check_air_constraints(program, &witness.trace, TableId::Main, &mut report)?;
    check_boundary_constraints(
        program,
        instance,
        &witness.trace,
        TableId::Main,
        &mut report,
    )?;
    check_chiplet_constraints(
        &program.chiplet_defs()?,
        &witness.chiplet_traces,
        &mut report,
    )?;
    check_bus_multisets(program, witness, &mut report)?;

    Ok(report)
}
