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

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
extern crate core;

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use constraint::{BoundaryConstraint, Constraint, ConstraintAst};
use core::marker::PhantomData;
use expander::VirtualExpander;
use hekate_core::errors;
use hekate_core::trace::{ColumnTrace, ColumnType, Trace, TraceCompatibleField};
use hekate_math::{Flat, HardwareField, TowerField};
use permutation::PermutationCheckSpec;

pub mod chiplet;
pub mod constraint;
pub mod expander;
pub mod permutation;
pub mod schema;

// =================================================================
// AIR TRAIT:
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

    /// Returns the physical layout
    /// of the columns in the trace.
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

    /// Columns whose MLE evaluation must equal a fixed
    /// Lagrange kernel at the verifier's r_final.
    fn lagrange_pinned_columns(&self) -> Vec<LagrangePin> {
        Vec::new()
    }

    /// Returns the `VirtualExpander` for chiplets
    /// with physical to virtual column expansion.
    fn virtual_expander(&self) -> Option<&VirtualExpander> {
        None
    }

    /// Parses a raw physical row (bytes) into
    /// the full Virtual Row (fields). Used by
    /// the Verifier to reconstruct the virtual
    /// trace from committed data.
    ///
    /// Delegates to `virtual_expander().parse_row()`
    /// when present. Falls back to 1:1 parsing
    /// from `column_layout()`.
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
// PROGRAM TRAIT — Composition over Air
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
#[derive(Clone, Copy, Debug)]
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
// LAGRANGE-PINNED COLUMNS
// =================================================================

/// Hypercube point at which a Lagrange MLE is anchored.
///
/// `Custom(bits)` carries the bit-decomposition of an arbitrary
/// row index, LSB first, length must equal the trace's `num_vars`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LagrangePoint {
    LastRow,
    FirstRow,
    Custom(Vec<bool>),
}

impl LagrangePoint {
    /// MLE evaluation of the pinned column
    /// at point `r` (LSB-first bit order).
    /// `r.len()` must equal `num_vars`.
    pub fn evaluate<F>(&self, r: &[Flat<F>]) -> Flat<F>
    where
        F: HardwareField,
    {
        let one = Flat::from_raw(F::ONE);
        match self {
            LagrangePoint::LastRow => {
                let mut prod = one;
                for &r_k in r {
                    prod *= r_k;
                }

                one - prod
            }
            LagrangePoint::FirstRow => {
                let mut prod = one;
                for &r_k in r {
                    prod *= one - r_k;
                }

                prod
            }
            LagrangePoint::Custom(bits) => {
                debug_assert_eq!(bits.len(), r.len(), "Custom point bit width != r.len()");

                let mut prod = one;
                for (k, &b) in bits.iter().enumerate() {
                    let factor = if b { r[k] } else { one - r[k] };
                    prod *= factor;
                }

                prod
            }
        }
    }
}

/// Single binding:
/// one virtual column pinned to a Lagrange MLE.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LagrangePin {
    pub col_idx: usize,
    pub point: LagrangePoint,
}

impl LagrangePin {
    pub fn last_row(col_idx: usize) -> Self {
        Self {
            col_idx,
            point: LagrangePoint::LastRow,
        }
    }

    pub fn first_row(col_idx: usize) -> Self {
        Self {
            col_idx,
            point: LagrangePoint::FirstRow,
        }
    }

    pub fn custom(col_idx: usize, bits: Vec<bool>) -> Self {
        Self {
            col_idx,
            point: LagrangePoint::Custom(bits),
        }
    }
}

/// Rejects out-of-range `col_idx`, mis-sized `Custom`
/// bit vectors, and duplicate pins on the same column
/// (a column anchored to two distinct points is unsatisfiable).
pub fn validate_lagrange_pins(
    pins: &[LagrangePin],
    num_columns: usize,
    num_vars: Option<usize>,
) -> errors::Result<()> {
    for (i, pin) in pins.iter().enumerate() {
        if pin.col_idx >= num_columns {
            return Err(errors::Error::Protocol {
                protocol: "lagrange_pin",
                message: "col_idx out of range",
            });
        }

        if let (LagrangePoint::Custom(bits), Some(nv)) = (&pin.point, num_vars)
            && bits.len() != nv
        {
            return Err(errors::Error::Protocol {
                protocol: "lagrange_pin",
                message: "Custom point bit width != num_vars",
            });
        }

        for prior in &pins[..i] {
            if prior.col_idx == pin.col_idx {
                return Err(errors::Error::Protocol {
                    protocol: "lagrange_pin",
                    message: "duplicate pin on same column",
                });
            }
        }
    }

    Ok(())
}
