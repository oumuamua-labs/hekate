// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! The circuit DSL of Hekate, the Rust zero-knowledge proof engine.
//!
//! A [`circuit::Circuit`] declares a table's columns with [`define_columns!`],
//! relates its rows with constraints, pins cells and fixed columns,
//! publishes public inputs and calls chiplets over LogUp buses.
//! It compiles into a [`Program`]: the prover runs it on a [`ProgramWitness`],
//! and the verifier checks it against a [`ProgramInstance`].
//! [`digest::program_id`] hashes the program for the verifier to pin.
//!
//! - [Your First ZK Program][first]
//! - [AIR Constraints][air]
//! - [LogUp Buses][logup]
//! - [Cryptographic Chiplets][chiplets]
//!
//! [first]: https://oumuamua.dev/hekate/docs/getting-started/your-first-zk-program
//! [air]: https://oumuamua.dev/hekate/docs/basics/air-constraints
//! [logup]: https://oumuamua.dev/hekate/docs/basics/logup-buses
//! [chiplets]: https://oumuamua.dev/hekate/docs/basics/cryptographic-chiplets

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
extern crate core;

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use constraint::{BoundaryConstraint, Constraint, ConstraintAst};
use core::marker::PhantomData;
use expander::VirtualExpander;
use hekate_core::errors;
use hekate_core::skip::SkipPoint;
use hekate_core::trace::{ColumnTrace, ColumnType, Trace, TraceCompatibleField};
use hekate_math::{Flat, HardwareField, TowerField};
use permutation::PermutationCheckSpec;

pub mod chiplet;
pub mod circuit;
pub mod constraint;
pub mod digest;
pub mod expander;
pub mod fold_pool;
pub mod linearized;
pub mod outer;
pub mod permutation;
pub mod predicate;
pub mod schema;

// =================================================================
// Core Algebraic Intermediate Representation
// =================================================================

/// Defines the algebraic structure, trace
/// layout, and constraints of an AIR table.
///
/// Implemented by both standalone
/// programs and independent chiplets.
pub trait Air<F: TowerField>: Sized + Clone + Sync {
    fn name(&self) -> String {
        "HekateAir".to_string()
    }

    fn num_columns(&self) -> usize {
        self.virtual_column_layout().len()
    }

    /// Flat expansion of `constraint_ast()`.
    fn constraints(&self) -> Vec<Constraint<F>> {
        self.constraint_ast().to_constraints()
    }

    /// Returns the list of boundary constraints. Each
    /// constraint ties a specific trace cell to a public
    /// input value. By default, returns an empty list.
    fn boundary_constraints(&self) -> Vec<BoundaryConstraint<F>> {
        Vec::new()
    }

    /// Returns the physical layout of the columns in the trace.
    ///
    /// [`ColumnType`] states what the commitment binds.
    fn column_layout(&self) -> &[ColumnType];

    /// Returns the virtual layout of the columns
    /// (after unpacking). Defaults to the expander's
    /// layout if present, else the physical layout.
    fn virtual_column_layout(&self) -> &[ColumnType] {
        match self.virtual_expander() {
            Some(e) => e.virtual_layout(),
            None => self.column_layout(),
        }
    }

    /// Returns the permutation check
    /// specifications for this AIR table.
    ///
    /// Each tuple contains:
    /// - `String`:
    ///   Unique bus identifier (e.g., `RomChiplet::BUS_ID`)
    /// - `PermutationCheckSpec`:
    ///   The GPA specification (sources, selector)
    ///
    /// Default:
    /// No permutation checks.
    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        Vec::new()
    }

    /// Columns pinned to a fixed shape, bound by MLE
    /// equality at `r_final`. Each shape must be a pure
    /// function of the row index, not the witness.
    fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
        Vec::new()
    }

    /// Returns the `VirtualExpander` for chiplets
    /// with physical to virtual column expansion.
    /// Non-reuse entries must tile `column_layout()`
    /// exactly; an uncovered column is bound by nothing.
    fn virtual_expander(&self) -> Option<&VirtualExpander> {
        None
    }

    /// Parses a raw physical row (bytes) into the full
    /// Virtual Row (fields). Used by the Verifier to
    /// reconstruct the virtual trace from committed data.
    ///
    /// Delegates to `virtual_expander().parse_row()` when present.
    /// Falls back to 1:1 parsing from `column_layout()`.
    fn parse_virtual_row(&self, bytes: &[u8], res: &mut Vec<Flat<F>>)
    where
        F: TraceCompatibleField,
    {
        res.clear();

        if let Some(e) = self.virtual_expander() {
            e.parse_row(bytes, res)
                .expect("committed row byte length must match physical_row_bytes");
            return;
        }

        let mut offset = 0;
        for col_type in self.column_layout() {
            let size = col_type.byte_size();
            if offset + size <= bytes.len() {
                res.push(col_type.parse_from_bytes(&bytes[offset..offset + size]));
                offset += size;
            }
        }
    }

    /// Returns the constraint system as an AST-DAG.
    fn constraint_ast(&self) -> ConstraintAst<F>;

    /// Chiplet defs used only for kernel dispatch.
    fn inline_chiplets(&self) -> errors::Result<Vec<chiplet::ChipletDef<F>>> {
        Ok(Vec::new())
    }

    /// Each hint's `chiplet_idx` indexes into `inline_chiplets()`.
    fn inline_chiplet_kernels(&self) -> Vec<InlineKernelHint> {
        Vec::new()
    }
}

// =================================================================
// Composition over Air
// =================================================================

/// Extends `Air<F>` with multi-table composition:
/// independent chiplets, GKR gadgets, and public inputs.
///
/// The top-level prover and verifier require `Program<F>`.
/// Internal sub-protocols (ZeroCheck, chiplet verification)
/// operate on `Air<F>` alone.
pub trait Program<F: TowerField>: Air<F> {
    /// Number of public inputs for this program.
    fn num_public_inputs(&self) -> usize {
        0
    }

    /// Returns independent AIR chiplet definitions.
    /// Each chiplet gets its own trace, commitment,
    /// ZeroCheck, and evaluation argument.
    /// Connected to the main trace via GPA bus.
    fn chiplet_defs(&self) -> errors::Result<Vec<chiplet::ChipletDef<F>>> {
        Ok(Vec::new())
    }
}

/// Represents a reference to a trace cell within
/// the program's execution trace. Points to a specific
/// column and relative row offset (current or next).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ProgramCell {
    pub col_idx: usize,

    /// false = current row (i),
    /// true = next row (i+1)
    pub next_row: bool,
}

impl ProgramCell {
    /// Reference to a cell in the current row.
    pub fn current(col_idx: usize) -> Self {
        Self {
            col_idx,
            next_row: false,
        }
    }

    /// Reference to a cell in the next row.
    pub fn next(col_idx: usize) -> Self {
        Self {
            col_idx,
            next_row: true,
        }
    }
}

// =================================================================
// INSTANCE & WITNESS
// =================================================================

/// Public Instance (Common inputs)
/// of the program execution.
#[derive(Clone, Debug)]
pub struct ProgramInstance<F: TowerField> {
    num_rows: usize,
    public_inputs: Vec<F>,
}

impl<F: TowerField> ProgramInstance<F> {
    pub fn new(num_rows: usize, public_inputs: Vec<F>) -> Self {
        assert!(
            num_rows.is_power_of_two(),
            "Program trace height must be power of 2"
        );

        Self {
            num_rows,
            public_inputs,
        }
    }

    #[inline(always)]
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Public inputs in canonical basis.
    #[inline(always)]
    pub fn public_inputs(&self) -> &[F] {
        &self.public_inputs
    }

    #[inline(always)]
    pub fn public_input(&self, idx: usize) -> Option<F> {
        self.public_inputs.get(idx).copied()
    }
}

/// Secret Witness (The Execution Trace) of the program.
/// Holds the trace data. Generic over T to support both
/// raw ColumnTrace and specialized wrappers.
pub struct ProgramWitness<F: TowerField, T: Trace = ColumnTrace> {
    pub trace: T,
    pub chiplet_traces: Vec<ColumnTrace>,
    _marker: PhantomData<F>,
}

impl<F: TowerField, T: Trace> ProgramWitness<F, T> {
    pub fn new(trace: T) -> Self {
        Self {
            trace,
            chiplet_traces: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Attach independent chiplet traces.
    /// Each entry corresponds by index to `chiplet_defs()`.
    pub fn with_chiplets(mut self, chiplet_traces: Vec<ColumnTrace>) -> Self {
        self.chiplet_traces = chiplet_traces;
        self
    }
}

/// Locates a chiplet's inlined sub-AST in
/// the program's merged `constraint_ast()`
/// so the prover can dispatch its kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InlineKernelHint {
    /// Index into `Air::inline_chiplets()`.
    pub chiplet_idx: usize,

    /// Absolute index of the chiplet's
    /// first root in the program's `roots`.
    pub root_offset: usize,

    /// Absolute column index where
    /// the chiplet's columns start.
    pub column_offset: usize,
}

// =================================================================
// FIXED COLUMNS
// =================================================================

/// Most variables [`ShapeEvaluator`] builds eq tables for:
/// a proof picks a chiplet's height, and the larger
/// table holds `2^⌈num_vars / 2⌉` cells.
const EQ_TABLE_MAX_VARS: usize = 32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CadenceSegment<F> {
    pub stride: usize,
    pub count: usize,
    pub origin: usize,
    pub values: Vec<F>,
}

/// Row-index-determined shape a fixed column is pinned to.
/// `LastRow` is one on every row but the last.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedShape<F> {
    LastRow,
    FirstRow,
    Custom(Vec<bool>),
    Periodic {
        period: usize,
        values: Vec<F>,
    },
    Sparse(Vec<(usize, F)>),
    Dense(Vec<F>),

    /// `values[(row - origin) % stride]` on rows
    /// `[origin, origin + stride·count)`, zero outside.
    Cadence {
        stride: usize,
        count: usize,
        origin: usize,
        values: Vec<F>,
    },

    /// Disjoint union of cadence blocks, sorted by
    /// origin; the MLE is the sum of the block MLEs.
    Segments(Vec<CadenceSegment<F>>),
}

impl<F> FixedShape<F> {
    /// Whether the shape keeps the committed column readable.
    /// The rest are supplied as indicators, never read back.
    pub fn is_overlay(&self) -> bool {
        match self {
            Self::Periodic { .. }
            | Self::Sparse(..)
            | Self::Dense(..)
            | Self::Cadence { .. }
            | Self::Segments(..) => true,
            Self::LastRow | Self::FirstRow | Self::Custom(..) => false,
        }
    }
}

impl<F: HardwareField> FixedShape<F> {
    /// Evaluates the shape's MLE at `point` in the flat basis;
    /// the point must span the table's variables.
    pub fn evaluate(&self, point: &SkipPoint<F>) -> Flat<F> {
        match self {
            FixedShape::LastRow => point.total() - point.weight((1 << point.num_vars()) - 1),
            FixedShape::FirstRow => point.weight(0),
            FixedShape::Custom(bits) => {
                let row = bits
                    .iter()
                    .enumerate()
                    .fold(0usize, |acc, (k, &b)| acc | ((b as usize) << k));

                point.weight(row)
            }
            FixedShape::Periodic { period, values } => {
                let p = period.trailing_zeros() as usize;

                let mut acc = Flat::from_raw(F::ZERO);
                for (j, &v) in values.iter().enumerate() {
                    acc += v.to_hardware() * point.low_bits_marginal(p, j);
                }

                acc
            }
            FixedShape::Sparse(entries) => {
                let mut acc = Flat::from_raw(F::ZERO);
                for &(row, v) in entries {
                    acc += v.to_hardware() * point.weight(row);
                }

                acc
            }
            FixedShape::Dense(values) => {
                let mut acc = Flat::from_raw(F::ZERO);
                for (i, &v) in values.iter().enumerate() {
                    acc += v.to_hardware() * point.weight(i);
                }

                acc
            }
            FixedShape::Cadence {
                stride,
                count,
                origin,
                values,
            } => cadence_mle(point, *stride, *count, *origin, values),
            FixedShape::Segments(segments) => {
                let mut acc = Flat::from_raw(F::ZERO);
                for seg in segments {
                    acc += cadence_mle(point, seg.stride, seg.count, seg.origin, &seg.values);
                }

                acc
            }
        }
    }

    /// Shape value at integer row `row`. O(1); prefer over `evaluate`
    /// at a vertex, which is O(N) for `Dense`. `Segments` unvalidated
    /// by `validate_fixed_columns` diverge from `evaluate`.
    pub fn value_at_row(&self, row: usize, num_vars: usize) -> Flat<F> {
        let one = Flat::from_raw(F::ONE);
        let zero = Flat::from_raw(F::ZERO);

        match self {
            FixedShape::FirstRow => {
                if row == 0 {
                    one
                } else {
                    zero
                }
            }
            FixedShape::LastRow => {
                if row == (1usize << num_vars) - 1 {
                    zero
                } else {
                    one
                }
            }
            FixedShape::Custom(bits) => {
                let target = bits
                    .iter()
                    .enumerate()
                    .fold(0usize, |acc, (k, &b)| acc | ((b as usize) << k));

                if row == target { one } else { zero }
            }
            FixedShape::Periodic { period, values } => values[row % period].to_hardware(),
            FixedShape::Sparse(entries) => {
                let mut acc = zero;
                for &(r, v) in entries {
                    if r == row {
                        acc += v.to_hardware();
                    }
                }

                acc
            }
            FixedShape::Dense(values) => values[row].to_hardware(),
            FixedShape::Cadence {
                stride,
                count,
                origin,
                values,
            } => cadence_value_at(row, *stride, *count, *origin, values),
            FixedShape::Segments(segments) => {
                // validate_shape rejects unsorted or overlapping spans:
                // at most one covers a row.
                let at = segments.partition_point(|seg| seg.origin <= row);

                match at.checked_sub(1) {
                    Some(i) => {
                        let seg = &segments[i];

                        cadence_value_at(row, seg.stride, seg.count, seg.origin, &seg.values)
                    }
                    None => zero,
                }
            }
        }
    }
}

/// One committed column pinned to a fixed shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedColumn<F> {
    pub col_idx: usize,
    pub shape: FixedShape<F>,
}

impl<F> FixedColumn<F> {
    pub fn last_row(col_idx: usize) -> Self {
        Self {
            col_idx,
            shape: FixedShape::LastRow,
        }
    }

    pub fn first_row(col_idx: usize) -> Self {
        Self {
            col_idx,
            shape: FixedShape::FirstRow,
        }
    }

    pub fn prefix(col_idx: usize, count: usize) -> Self
    where
        F: TowerField,
    {
        Self {
            col_idx,
            shape: FixedShape::Cadence {
                stride: 1,
                count,
                origin: 0,
                values: vec![F::ONE],
            },
        }
    }

    pub fn custom(col_idx: usize, bits: Vec<bool>) -> Self {
        Self {
            col_idx,
            shape: FixedShape::Custom(bits),
        }
    }

    pub fn periodic(col_idx: usize, period: usize, values: Vec<F>) -> Self {
        Self {
            col_idx,
            shape: FixedShape::Periodic { period, values },
        }
    }

    pub fn sparse(col_idx: usize, entries: Vec<(usize, F)>) -> Self {
        Self {
            col_idx,
            shape: FixedShape::Sparse(entries),
        }
    }

    pub fn dense(col_idx: usize, values: Vec<F>) -> Self {
        Self {
            col_idx,
            shape: FixedShape::Dense(values),
        }
    }

    pub fn segments(col_idx: usize, segments: Vec<CadenceSegment<F>>) -> Self {
        Self {
            col_idx,
            shape: FixedShape::Segments(segments),
        }
    }
}

/// One table's fixed shapes at one point, sharing
/// residue sums and eq tables across the shapes.
pub struct ShapeEvaluator<'a, F: HardwareField> {
    point: &'a SkipPoint<F>,
    eq: Option<EqTables<F>>,
    sums: BTreeMap<(usize, usize), Vec<Flat<F>>>,
}

impl<'a, F: HardwareField> ShapeEvaluator<'a, F> {
    pub fn new(point: &'a SkipPoint<F>) -> Self {
        Self {
            point,
            eq: (point.num_vars() <= EQ_TABLE_MAX_VARS)
                .then(|| EqTables::new(point))
                .flatten(),
            sums: BTreeMap::new(),
        }
    }

    /// MLE of `shape` at the point; equals
    /// [`FixedShape::evaluate`] on every input.
    pub fn evaluate(&mut self, shape: &FixedShape<F>) -> Flat<F> {
        let zero = Flat::from_raw(F::ZERO);

        match shape {
            FixedShape::Sparse(entries) => {
                let mut acc = zero;
                for &(row, value) in entries {
                    acc += scaled(value, self.eq(row));
                }

                acc
            }
            FixedShape::Dense(values) => {
                let mut acc = zero;
                for (row, &value) in values.iter().enumerate() {
                    acc += scaled(value, self.eq(row));
                }

                acc
            }
            FixedShape::Cadence {
                stride,
                count,
                origin,
                values,
            } => self.cadence(*stride, *count, *origin, values),
            FixedShape::Segments(segments) => {
                let mut acc = zero;
                for seg in segments {
                    acc += self.cadence(seg.stride, seg.count, seg.origin, &seg.values);
                }

                acc
            }
            FixedShape::LastRow
            | FixedShape::FirstRow
            | FixedShape::Custom(_)
            | FixedShape::Periodic { .. } => shape.evaluate(self.point),
        }
    }

    /// Returns the weight of `row`, its bits past the point ignored.
    fn eq(&self, row: usize) -> Flat<F> {
        match &self.eq {
            Some(tables) => tables.eq(row),
            None => self.point.weight(row),
        }
    }

    /// MLE of one cadence block: an interval of the eq
    /// tables at stride 1, cached residue sums otherwise.
    fn cadence(&mut self, stride: usize, count: usize, origin: usize, values: &[F]) -> Flat<F> {
        let zero = Flat::from_raw(F::ZERO);

        if stride == 0 || values.len() != stride {
            return zero;
        }

        let end = origin.saturating_add(stride.saturating_mul(count));

        let Self { point, eq, sums } = self;

        match (stride, eq) {
            (1, Some(tables)) => scaled(values[0], tables.below(end) + tables.below(origin)),
            _ => {
                for limit in [end, origin] {
                    sums.entry((stride, limit))
                        .or_insert_with(|| residue_sums(point, limit, stride, None));
                }

                let upper = &sums[&(stride, end)];
                let lower = &sums[&(stride, origin)];
                let shift = stride - origin % stride;

                let mut acc = zero;
                for (c, (&u, &l)) in upper.iter().zip(lower).enumerate() {
                    acc += scaled(values[(c + shift) % stride], u + l);
                }

                acc
            }
        }
    }
}

/// `eq(r, ·)` over the low and the high
/// half of `r`, each with its prefix sums.
struct EqTables<F> {
    low_bits: usize,
    eq_low: Vec<Flat<F>>,
    eq_high: Vec<Flat<F>>,
    below_low: Vec<Flat<F>>,
    below_high: Vec<Flat<F>>,
}

impl<F: HardwareField> EqTables<F> {
    fn new(point: &SkipPoint<F>) -> Option<Self> {
        let low_bits = (point.num_vars() / 2).max(point.skip_vars());
        let halves = point.halves(low_bits).ok()?;

        Some(Self {
            low_bits,
            below_low: running_sums(&halves.low),
            below_high: running_sums(&halves.high),
            eq_low: halves.low,
            eq_high: halves.high,
        })
    }

    /// `eq(r, row)`, ignoring the bits of `row` past `r.len()`.
    fn eq(&self, row: usize) -> Flat<F> {
        let low = row & (self.eq_low.len() - 1);
        let high = (row >> self.low_bits) & (self.eq_high.len() - 1);

        self.eq_low[low] * self.eq_high[high]
    }

    /// Sums the weights of the rows below `limit`;
    /// every row's once `limit` reaches the height.
    fn below(&self, limit: usize) -> Flat<F> {
        let high = limit >> self.low_bits;
        let total_low = self.below_low[self.eq_low.len()];

        if high >= self.eq_high.len() {
            return total_low;
        }

        let low = limit & (self.eq_low.len() - 1);

        self.below_high[high] * total_low + self.eq_high[high] * self.below_low[low]
    }
}

/// Declares a fixed column from a shape.
pub fn fix<F>(col_idx: usize, shape: FixedShape<F>) -> FixedColumn<F> {
    FixedColumn { col_idx, shape }
}

/// Rejects out-of-range `col_idx`, duplicate pins,
/// malformed shapes, and out-of-domain values
/// (`Bit` columns require values in {0, 1}).
pub fn validate_fixed_columns<F: TowerField>(
    fixed: &[FixedColumn<F>],
    layout: &[ColumnType],
    num_vars: Option<usize>,
) -> errors::Result<()> {
    for (i, fc) in fixed.iter().enumerate() {
        if fc.col_idx >= layout.len() {
            return Err(errors::Error::Protocol {
                protocol: "fixed_column",
                message: "col_idx out of range",
            });
        }

        validate_shape(&fc.shape, layout[fc.col_idx], num_vars)?;

        for prior in &fixed[..i] {
            if prior.col_idx == fc.col_idx {
                return Err(errors::Error::Protocol {
                    protocol: "fixed_column",
                    message: "duplicate pin on same column",
                });
            }
        }
    }

    Ok(())
}

/// Sums `w(x) · W^⌊x / stride⌋` over `x < limit` per residue
/// `x mod stride`, `w` the point's weights; `W = 1` without a step.
fn residue_sums<F: HardwareField>(
    point: &SkipPoint<F>,
    limit: usize,
    stride: usize,
    step: Option<Flat<F>>,
) -> Vec<Flat<F>> {
    let one = Flat::from_raw(F::ONE);
    let zero = Flat::from_raw(F::ZERO);

    let num_vars = point.num_vars();
    let block_vars = point.skip_vars();

    let full = match 1usize.checked_shl(num_vars as u32) {
        Some(cap) => limit >= cap,
        None => false,
    };

    let squares: Vec<Flat<F>> = step.map_or_else(Vec::new, |w| {
        (0..num_vars)
            .scan(w, |power, _| {
                let current = *power;
                *power *= *power;

                Some(current)
            })
            .collect()
    });

    let mut free = vec![zero; stride];
    let mut next = vec![zero; stride];

    let mut tight = if full { zero } else { one };
    let mut tight_residue = 0usize;

    if full {
        free[0] = one;
    }

    for t in (block_vars..num_vars).rev() {
        let r_t = point.tail()[t - block_vars];
        let step = step.map(|_| squares[t]);

        next.fill(zero);

        for (residue, &weight) in free.iter().enumerate() {
            let low = 2 * residue;
            let hi = weight * r_t;

            let (at, w) = division_step(low, weight + hi, stride, step);
            next[at] += w;

            let (at, w) = division_step(low + 1, hi, stride, step);
            next[at] += w;
        }

        let low = 2 * tight_residue;
        let hi = tight * r_t;

        (tight_residue, tight) = match (limit >> t) & 1 {
            1 => {
                let (at, w) = division_step(low, tight + hi, stride, step);
                next[at] += w;

                division_step(low + 1, hi, stride, step)
            }
            _ => division_step(low, tight + hi, stride, step),
        };

        core::mem::swap(&mut free, &mut next);
    }

    let block = point.low();

    let powers: Vec<Flat<F>> = step.map_or_else(Vec::new, |w| {
        (0..block.len())
            .scan(one, |power, _| {
                let current = *power;
                *power *= w;

                Some(current)
            })
            .collect()
    });

    let quotient_power = |q: usize| powers.get(q).copied().unwrap_or(one);

    let gcd = block.len().min(1 << stride.trailing_zeros());

    let mut groups = vec![zero; stride / gcd];
    for (residue, &weight) in free.iter().enumerate() {
        let shifted = residue * block.len();
        groups[shifted % stride / gcd] += weight * quotient_power(shifted / stride);
    }

    next.fill(zero);

    for (group, &weight) in groups.iter().enumerate() {
        for (x, &low) in block.iter().enumerate() {
            let value = group * gcd + x;
            next[value % stride] += weight * low * quotient_power(value / stride);
        }
    }

    let limit_low = limit & (block.len() - 1);
    for (x, &low) in block.iter().enumerate().take(limit_low) {
        let value = tight_residue * block.len() + x;
        next[value % stride] += tight * low * quotient_power(value / stride);
    }

    next
}

fn division_step<F: HardwareField>(
    value: usize,
    weight: Flat<F>,
    stride: usize,
    step: Option<Flat<F>>,
) -> (usize, Flat<F>) {
    match (value >= stride, step) {
        (true, Some(step)) => (value - stride, weight * step),
        (true, None) => (value - stride, weight),
        (false, _) => (value, weight),
    }
}

fn cadence_mle<F: HardwareField>(
    point: &SkipPoint<F>,
    stride: usize,
    count: usize,
    origin: usize,
    values: &[F],
) -> Flat<F> {
    if stride == 0 || values.len() != stride {
        return Flat::from_raw(F::ZERO);
    }

    let end = origin.saturating_add(stride.saturating_mul(count));

    let mut class_sums = residue_sums(point, end, stride, None);
    let start_sums = residue_sums(point, origin, stride, None);

    for (c, s) in start_sums.iter().enumerate() {
        class_sums[c] += *s;
    }

    let mut acc = Flat::from_raw(F::ZERO);
    for (c, sum) in class_sums.iter().enumerate() {
        let offset = (c + stride - origin % stride) % stride;
        acc += values[offset].to_hardware() * *sum;
    }

    acc
}

fn cadence_value_at<F: HardwareField>(
    row: usize,
    stride: usize,
    count: usize,
    origin: usize,
    values: &[F],
) -> Flat<F> {
    if stride == 0 || values.len() != stride {
        return Flat::from_raw(F::ZERO);
    }

    let end = origin.saturating_add(stride.saturating_mul(count));

    if row < origin || row >= end {
        Flat::from_raw(F::ZERO)
    } else {
        values[(row - origin) % stride].to_hardware()
    }
}

/// `sums[k] = Σ_{i < k} values[i]` for `k = 0..=len`.
fn running_sums<F: HardwareField>(values: &[Flat<F>]) -> Vec<Flat<F>> {
    let mut sums = Vec::with_capacity(values.len() + 1);
    let mut acc = Flat::from_raw(F::ZERO);

    sums.push(acc);

    for &v in values {
        acc += v;

        sums.push(acc);
    }

    sums
}

/// `value · x`, skipping the basis conversion when `value` is zero or one.
fn scaled<F: HardwareField>(value: F, x: Flat<F>) -> Flat<F> {
    if value == F::ZERO {
        Flat::from_raw(F::ZERO)
    } else if value == F::ONE {
        x
    } else {
        value.to_hardware() * x
    }
}

fn validate_shape<F: TowerField>(
    shape: &FixedShape<F>,
    col_type: ColumnType,
    num_vars: Option<usize>,
) -> errors::Result<()> {
    match shape {
        FixedShape::LastRow | FixedShape::FirstRow => Ok(()),
        FixedShape::Custom(bits) => match num_vars {
            Some(nv) if bits.len() != nv => Err(errors::Error::Protocol {
                protocol: "fixed_column",
                message: "Custom point bit width != num_vars",
            }),
            _ => Ok(()),
        },
        FixedShape::Periodic { period, values } => {
            if !period.is_power_of_two() {
                return Err(errors::Error::Protocol {
                    protocol: "fixed_column",
                    message: "Periodic period must be a power of two",
                });
            }

            if values.len() != *period {
                return Err(errors::Error::Protocol {
                    protocol: "fixed_column",
                    message: "Periodic values length != period",
                });
            }

            if let Some(nv) = num_vars
                && *period > (1usize << nv)
            {
                return Err(errors::Error::Protocol {
                    protocol: "fixed_column",
                    message: "Periodic period exceeds trace height",
                });
            }

            check_bit_domain(values.iter().copied(), col_type)
        }
        FixedShape::Sparse(entries) => {
            if let Some(nv) = num_vars {
                let n = 1usize << nv;
                for &(row, _) in entries {
                    if row >= n {
                        return Err(errors::Error::Protocol {
                            protocol: "fixed_column",
                            message: "Sparse row index exceeds trace height",
                        });
                    }
                }
            }

            let mut rows: Vec<usize> = entries.iter().map(|&(row, _)| row).collect();
            rows.sort_unstable();

            if rows.windows(2).any(|pair| pair[0] == pair[1]) {
                return Err(errors::Error::Protocol {
                    protocol: "fixed_column",
                    message: "duplicate Sparse row",
                });
            }

            check_bit_domain(entries.iter().map(|&(_, v)| v), col_type)
        }
        FixedShape::Dense(values) => {
            if let Some(nv) = num_vars
                && values.len() != (1usize << nv)
            {
                return Err(errors::Error::Protocol {
                    protocol: "fixed_column",
                    message: "Dense values length != trace height",
                });
            }

            check_bit_domain(values.iter().copied(), col_type)
        }
        FixedShape::Cadence {
            stride,
            count,
            origin,
            values,
        } => {
            if *stride == 0 {
                return Err(errors::Error::Protocol {
                    protocol: "fixed_column",
                    message: "Cadence stride must be non-zero",
                });
            }

            if values.len() != *stride {
                return Err(errors::Error::Protocol {
                    protocol: "fixed_column",
                    message: "Cadence values length != stride",
                });
            }

            let end = stride
                .checked_mul(*count)
                .and_then(|span| span.checked_add(*origin))
                .ok_or(errors::Error::Protocol {
                    protocol: "fixed_column",
                    message: "Cadence active range overflows",
                })?;

            if let Some(nv) = num_vars
                && end > (1usize << nv)
            {
                return Err(errors::Error::Protocol {
                    protocol: "fixed_column",
                    message: "Cadence active range exceeds trace height",
                });
            }

            check_bit_domain(values.iter().copied(), col_type)
        }
        FixedShape::Segments(segments) => {
            if segments.is_empty() {
                return Err(errors::Error::Protocol {
                    protocol: "fixed_column",
                    message: "Segments list is empty",
                });
            }

            let mut prev_end = 0usize;
            for seg in segments {
                if seg.stride == 0 {
                    return Err(errors::Error::Protocol {
                        protocol: "fixed_column",
                        message: "Segments stride must be non-zero",
                    });
                }

                if seg.count == 0 {
                    return Err(errors::Error::Protocol {
                        protocol: "fixed_column",
                        message: "Segments count must be non-zero",
                    });
                }

                if seg.values.len() != seg.stride {
                    return Err(errors::Error::Protocol {
                        protocol: "fixed_column",
                        message: "Segments values length != stride",
                    });
                }

                let end = seg
                    .stride
                    .checked_mul(seg.count)
                    .and_then(|span| span.checked_add(seg.origin))
                    .ok_or(errors::Error::Protocol {
                        protocol: "fixed_column",
                        message: "Segments active range overflows",
                    })?;

                if seg.origin < prev_end {
                    return Err(errors::Error::Protocol {
                        protocol: "fixed_column",
                        message: "Segments must be sorted by origin and disjoint",
                    });
                }

                prev_end = end;

                check_bit_domain(seg.values.iter().copied(), col_type)?;
            }

            if let Some(nv) = num_vars
                && prev_end > (1usize << nv)
            {
                return Err(errors::Error::Protocol {
                    protocol: "fixed_column",
                    message: "Segments active range exceeds trace height",
                });
            }

            Ok(())
        }
    }
}

fn check_bit_domain<F: TowerField>(
    values: impl Iterator<Item = F>,
    col_type: ColumnType,
) -> errors::Result<()> {
    if col_type != ColumnType::Bit {
        return Ok(());
    }

    for v in values {
        if v != F::ZERO && v != F::ONE {
            return Err(errors::Error::Protocol {
                protocol: "fixed_column",
                message: "Bit fixed column value not in {0,1}",
            });
        }
    }

    Ok(())
}
