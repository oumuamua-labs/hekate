// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
extern crate core;

use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use constraint::{BoundaryConstraint, Constraint, ConstraintAst};
use core::marker::PhantomData;
use expander::VirtualExpander;
use hekate_core::errors;
use hekate_core::trace::{ColumnTrace, ColumnType, Trace, TraceCompatibleField};
use hekate_math::{Flat, HardwareField, TowerField};
use permutation::PermutationCheckSpec;

pub mod chiplet;
pub mod circuit;
pub mod constraint;
pub mod digest;
pub mod expander;
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
    /// This describes the storage type
    /// (Bit, B8, B32, etc.) of each column.
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
    /// MLE of the shape at point `r` (LSB-first),
    /// in flat basis. `r.len()` must equal `num_vars`.
    pub fn evaluate(&self, r: &[Flat<F>]) -> Flat<F> {
        let one = Flat::from_raw(F::ONE);
        match self {
            FixedShape::LastRow => {
                let mut prod = one;
                for &r_k in r {
                    prod *= r_k;
                }

                one - prod
            }
            FixedShape::FirstRow => {
                let mut prod = one;
                for &r_k in r {
                    prod *= one - r_k;
                }

                prod
            }
            FixedShape::Custom(bits) => {
                debug_assert_eq!(bits.len(), r.len(), "Custom point bit width != r.len()");

                let mut prod = one;
                for (k, &b) in bits.iter().enumerate() {
                    let factor = if b { r[k] } else { one - r[k] };
                    prod *= factor;
                }

                prod
            }
            FixedShape::Periodic { period, values } => {
                // Low p = log2(period) coords only;
                // high coords each sum to 1.
                let p = period.trailing_zeros() as usize;

                let mut acc = Flat::from_raw(F::ZERO);
                for (j, &v) in values.iter().enumerate() {
                    acc += v.to_hardware() * eq_index(&r[..p], j);
                }

                acc
            }
            FixedShape::Sparse(entries) => {
                let mut acc = Flat::from_raw(F::ZERO);
                for &(row, v) in entries {
                    acc += v.to_hardware() * eq_index(r, row);
                }

                acc
            }
            FixedShape::Dense(values) => {
                let mut acc = Flat::from_raw(F::ZERO);
                for (i, &v) in values.iter().enumerate() {
                    acc += v.to_hardware() * eq_index(r, i);
                }

                acc
            }
            FixedShape::Cadence {
                stride,
                count,
                origin,
                values,
            } => cadence_mle(r, *stride, *count, *origin, values),
            FixedShape::Segments(segments) => {
                let mut acc = Flat::from_raw(F::ZERO);
                for seg in segments {
                    acc += cadence_mle(r, seg.stride, seg.count, seg.origin, &seg.values);
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

/// `Σ_{i < limit, i ≡ c (mod stride)} eq(r, i)` per residue class `c`.
fn prefix_residue_sums<F: HardwareField>(
    r: &[Flat<F>],
    limit: usize,
    stride: usize,
) -> Vec<Flat<F>> {
    let one = Flat::from_raw(F::ONE);
    let zero = Flat::from_raw(F::ZERO);
    let nv = r.len();

    let full = match 1usize.checked_shl(nv as u32) {
        Some(cap) => limit >= cap,
        None => false,
    };

    let mut pow2_mod = Vec::with_capacity(nv);
    let mut p2 = 1usize % stride;

    for _ in 0..nv {
        pow2_mod.push(p2);
        p2 = (p2 * 2) % stride;
    }

    let mut free = alloc::vec![zero; stride];
    let mut next_free = alloc::vec![zero; stride];
    let mut tight = if full { zero } else { one };
    let mut tight_residue = 0usize;

    if full {
        free[0] = one;
    }

    for k in (0..nv).rev() {
        let p2k = pow2_mod[k];
        for slot in next_free.iter_mut() {
            *slot = zero;
        }

        for c in 0..stride {
            let w = free[c];

            next_free[c] += w * (one - r[k]);
            next_free[(c + p2k) % stride] += w * r[k];
        }

        if (limit >> k) & 1 == 1 {
            next_free[tight_residue] += tight * (one - r[k]);
            tight *= r[k];
            tight_residue = (tight_residue + p2k) % stride;
        } else {
            tight *= one - r[k];
        }

        core::mem::swap(&mut free, &mut next_free);
    }

    free
}

fn cadence_mle<F: HardwareField>(
    r: &[Flat<F>],
    stride: usize,
    count: usize,
    origin: usize,
    values: &[F],
) -> Flat<F> {
    if stride == 0 || values.len() != stride {
        return Flat::from_raw(F::ZERO);
    }

    let end = origin.saturating_add(stride.saturating_mul(count));

    let mut class_sums = prefix_residue_sums(r, end, stride);
    let start_sums = prefix_residue_sums(r, origin, stride);

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

fn eq_index<F: HardwareField>(r: &[Flat<F>], index: usize) -> Flat<F> {
    let one = Flat::from_raw(F::ONE);
    let mut prod = one;

    for (k, &r_k) in r.iter().enumerate() {
        let factor = if (index >> k) & 1 == 1 {
            r_k
        } else {
            one - r_k
        };

        prod *= factor;
    }

    prod
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

            for (i, &(row, _)) in entries.iter().enumerate() {
                if entries[..i].iter().any(|&(prior, _)| prior == row) {
                    return Err(errors::Error::Protocol {
                        protocol: "fixed_column",
                        message: "duplicate Sparse row",
                    });
                }
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
