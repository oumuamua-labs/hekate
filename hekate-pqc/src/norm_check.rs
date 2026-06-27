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

//! Norm Check Chiplet for ML-DSA.
//!
//! Verifies that polynomial coefficients
//! satisfy infinity norm bounds:
//! |z_i| < bound (in mod-q representation).
//!
//! Each row checks one coefficient:
//! 1. Compute complement = q - value
//! 2. Select abs = value or complement
//!    (based on sign bit)
//! 3. Range check abs < bound
//!
//! Bus: "norm_check" (internal to ML-DSA composite)

use crate::utils::{
    fill_add_carry_packed, fill_sub_borrow_packed, flush_bit_buffer, pack_bits, pack_one,
};
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder, TraceCompatibleField};
use hekate_gadgets::atoms::int_arith;
use hekate_math::{Block32, TowerField};
use hekate_program::Air;
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::expander::VirtualExpander;
use hekate_program::permutation::{PermutationCheckSpec, Source};

// =================================================================
// Column Layout
// =================================================================

/// Column index map for the NormCheck chiplet.
/// Computed at construction time from
/// `modulus`, `bit_width`, and `bound`.
#[derive(Clone, Debug)]
pub struct NormCheckLayout {
    pub bit_width: usize,

    // Arithmetic Bit columns (virtual indices)
    pub value_bits: usize,
    pub abs_bits: usize,
    pub comp_bits: usize,
    pub comp_carry: usize,
    pub range_result: usize,
    pub range_borrow: usize,
    pub is_negative: usize,

    pub num_bit_cols: usize,
    pub num_packed_b32_cols: usize,
    pub num_expanded_bits: usize,

    // B32 columns (bus, virtual indices)
    pub bus_value: usize,
    pub bus_idx: usize,

    // Control
    pub s_active: usize,

    /// Virtual column count (constraint AST).
    pub num_columns: usize,

    /// Physical column count (committed trace).
    pub num_physical_columns: usize,
}

impl NormCheckLayout {
    /// Compute the column layout for a
    /// given modulus and bound.
    pub fn compute(bit_width: usize) -> Self {
        let mut offset = 0usize;

        let mut alloc = |n: usize| -> usize {
            let start = offset;
            offset += n;

            start
        };

        let value_bits = alloc(bit_width);
        let abs_bits = alloc(bit_width);
        let comp_bits = alloc(bit_width);
        let comp_carry = alloc(bit_width + 1);
        let range_result = alloc(bit_width);
        let range_borrow = alloc(bit_width + 1);
        let is_negative = alloc(1);

        let num_bit_cols = offset;
        let num_packed_b32_cols = num_bit_cols.div_ceil(32);
        let num_expanded_bits = num_packed_b32_cols * 32;

        let bus_value = num_expanded_bits;
        let bus_idx = num_expanded_bits + 1;
        let s_active = num_expanded_bits + 2;

        let num_columns = num_expanded_bits + 3;
        let num_physical_columns = num_packed_b32_cols + 3;

        NormCheckLayout {
            bit_width,
            value_bits,
            abs_bits,
            comp_bits,
            comp_carry,
            range_result,
            range_borrow,
            is_negative,
            num_bit_cols,
            num_packed_b32_cols,
            num_expanded_bits,
            bus_value,
            bus_idx,
            s_active,
            num_columns,
            num_physical_columns,
        }
    }

    pub fn build_virtual_layout(&self) -> Vec<ColumnType> {
        let mut layout = Vec::with_capacity(self.num_columns);

        for _ in 0..self.num_expanded_bits {
            layout.push(ColumnType::Bit);
        }

        // B32 bus columns
        layout.push(ColumnType::B32);
        layout.push(ColumnType::B32);

        // s_active
        layout.push(ColumnType::Bit);

        debug_assert_eq!(layout.len(), self.num_columns);

        layout
    }

    pub fn build_physical_layout(&self) -> Vec<ColumnType> {
        let mut layout = Vec::with_capacity(self.num_physical_columns);

        for _ in 0..self.num_packed_b32_cols {
            layout.push(ColumnType::B32);
        }

        layout.push(ColumnType::B32);
        layout.push(ColumnType::B32);
        layout.push(ColumnType::Bit);

        debug_assert_eq!(layout.len(), self.num_physical_columns);

        layout
    }
}

// =================================================================
// NormCheck Chiplet
// =================================================================

/// Norm Check Chiplet.
///
/// Parameterized by modulus q and bound:
/// - q=8380417 for ML-DSA
/// - bound = γ₁ - β (e.g. 524092 for ML-DSA-65)
///
/// Each row verifies |z_i| < bound where
/// z_i is a coefficient mod q.
///
/// In mod-q representation:
/// - z_i ∈ [0, (q-1)/2] → positive, |z_i| = z_i
/// - z_i ∈ ((q-1)/2, q-1] → negative, |z_i| = q - z_i
#[derive(Clone, Debug)]
pub struct NormCheckChiplet {
    pub modulus: u32,
    pub bit_width: usize,
    pub bound: u32,
    pub num_rows: usize,

    layout: NormCheckLayout,
    expander: VirtualExpander,
}

impl NormCheckChiplet {
    pub const BUS_ID: &'static str = "norm_check";

    pub fn new(modulus: u32, bound: u32, num_rows: usize) -> Self {
        assert!(num_rows.is_power_of_two());
        assert!(bound > 0 && bound <= modulus / 2);

        let bit_width = 32 - modulus.leading_zeros() as usize;
        let layout = NormCheckLayout::compute(bit_width);

        let expander = VirtualExpander::new()
            .expand_bits(layout.num_packed_b32_cols, ColumnType::B32)
            .pass_through(2, ColumnType::B32)
            .control_bits(1)
            .build()
            .expect("NormCheckChiplet expander");

        Self {
            modulus,
            bit_width,
            bound,
            num_rows,
            layout,
            expander,
        }
    }

    pub fn layout(&self) -> &NormCheckLayout {
        &self.layout
    }

    pub fn linking_spec(&self) -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(self.layout.bus_value),
                    b"kappa_nc_value" as &[u8],
                ),
                (
                    Source::Column(self.layout.bus_idx),
                    b"kappa_nc_idx" as &[u8],
                ),
            ],
            Some(self.layout.s_active),
        )
        .with_clock_waiver(
            "see pqc/norm_check.rs: bus_idx is positional, AIR forces one row per \
             (idx) value; partner mldsa ctrl side carries the matching idx clock",
        )
    }
}

// =================================================================
// Air Implementation
// =================================================================

impl<F: TowerField + TraceCompatibleField> Air<F> for NormCheckChiplet {
    fn name(&self) -> String {
        "NormCheckChiplet".to_string()
    }

    fn column_layout(&self) -> &[ColumnType] {
        let layout = self.layout.build_physical_layout();
        Box::leak(layout.into_boxed_slice())
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![(Self::BUS_ID.into(), self.linking_spec())]
    }

    fn virtual_expander(&self) -> Option<&VirtualExpander> {
        Some(&self.expander)
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        build_norm_check_constraints(self.modulus, self.bit_width, self.bound, &self.layout)
    }
}

// =================================================================
// Constraint Generation
// =================================================================

/// Build the constraint AST for one
/// norm check row.
fn build_norm_check_constraints<F: TowerField>(
    modulus: u32,
    bit_width: usize,
    bound: u32,
    ly: &NormCheckLayout,
) -> ConstraintAst<F> {
    let cs = ConstraintSystem::<F>::new();
    let s_active = cs.col(ly.s_active);

    // Collect bit expressions
    let val: Vec<_> = (0..bit_width).map(|k| cs.col(ly.value_bits + k)).collect();
    let abs: Vec<_> = (0..bit_width).map(|k| cs.col(ly.abs_bits + k)).collect();
    let comp: Vec<_> = (0..bit_width).map(|k| cs.col(ly.comp_bits + k)).collect();
    let comp_c: Vec<_> = (0..=bit_width).map(|k| cs.col(ly.comp_carry + k)).collect();
    let rng_r: Vec<_> = (0..bit_width)
        .map(|k| cs.col(ly.range_result + k))
        .collect();
    let rng_w: Vec<_> = (0..=bit_width)
        .map(|k| cs.col(ly.range_borrow + k))
        .collect();
    let is_neg = cs.col(ly.is_negative);

    // Booleanity of input bits (s_active gated)
    for &bit in val.iter().chain(abs.iter()).chain(comp.iter()) {
        cs.assert_zero_when(s_active, bit * (bit + cs.one()));
    }

    cs.assert_zero_when(s_active, is_neg * (is_neg + cs.one()));

    // 1. Complement verification:
    //    value + complement = q (constant)
    //
    //    Pass q's bits as constant result_bits
    //    to constrain_add_carry_chain.
    let q_bits: Vec<_> = (0..bit_width)
        .map(|k| {
            if (modulus >> k) & 1 == 1 {
                cs.one()
            } else {
                cs.constant(F::ZERO)
            }
        })
        .collect();

    int_arith::add_carry_chain(&cs, &val, &comp, &q_bits, &comp_c);

    // Carry-out must be 0 (value + comp = q fits in bit_width)
    cs.constrain(comp_c[bit_width]);

    // 2. MUX:
    // abs = is_neg ? comp : value
    //
    //    At each bit k:
    //    abs[k] + val[k] + is_neg*(val[k] + comp[k]) = 0
    //
    //    When is_neg=0: abs[k] = val[k]
    //    When is_neg=1: abs[k] = comp[k]
    for k in 0..bit_width {
        cs.constrain(abs[k] + val[k] + is_neg * (val[k] + comp[k]));
    }

    // 3. Range check:
    // abs < bound
    int_arith::range_check(&cs, &abs, &rng_r, &rng_w, bound);

    // Tail-bit padding:
    // unused bits in the last
    // packed B32 must be zero.
    for k in ly.num_bit_cols..ly.num_expanded_bits {
        cs.constrain(cs.col(k));
    }

    cs.build()
}

// =================================================================
// Trace Generation
// =================================================================

/// A single norm check operation.
#[derive(Clone, Debug)]
pub struct NormCheckOp {
    /// Coefficient value mod q.
    pub value: u32,

    /// Coefficient index (for bus linking).
    pub idx: u32,

    /// RAM address where this coefficient
    /// was written. Used by ctrl to
    /// co-activate RAM for binding.
    pub ram_addr: u32,
}

/// Generate the norm check trace.
///
/// `modulus`: prime q
/// `bound`: norm bound (|z_i| < bound)
/// `ops`: list of norm check operations
/// `num_rows`: trace height (power of 2)
pub fn generate_norm_check_trace(
    modulus: u32,
    bound: u32,
    ops: &[NormCheckOp],
    num_rows: usize,
) -> hekate_core::errors::Result<ColumnTrace> {
    assert!(num_rows.is_power_of_two());
    assert!(ops.len() <= num_rows);

    let bit_width = 32 - modulus.leading_zeros() as usize;
    let layout = NormCheckLayout::compute(bit_width);
    let physical = layout.build_physical_layout();

    let num_vars = num_rows.trailing_zeros() as usize;
    let num_packed = layout.num_packed_b32_cols;

    let mut tb = TraceBuilder::new(&physical, num_vars)?;

    let phy_bus_value = num_packed;
    let phy_bus_idx = num_packed + 1;
    let phy_s_active = num_packed + 2;

    let half_q = (modulus - 1) / 2;

    let mut bits = vec![0u32; num_packed];

    for (row, op) in ops.iter().enumerate() {
        debug_assert!(op.value < modulus);

        let value = op.value;
        let is_negative = value > half_q;
        let complement = modulus - value;
        let abs_value = if is_negative { complement } else { value };

        debug_assert!(
            abs_value < bound,
            "norm check: |{}| = {} >= {}",
            value,
            abs_value,
            bound
        );

        bits.iter_mut().for_each(|w| *w = 0);

        pack_bits(&mut bits, layout.value_bits, value as u64, bit_width);
        pack_bits(&mut bits, layout.abs_bits, abs_value as u64, bit_width);
        pack_bits(&mut bits, layout.comp_bits, complement as u64, bit_width);

        // comp_carry:
        // value + complement = q
        fill_add_carry_packed(
            &mut bits,
            layout.comp_carry,
            bit_width + 1,
            value as u64,
            complement as u64,
        );

        if is_negative {
            pack_one(&mut bits, layout.is_negative, true);
        }

        // range_result + range_borrow:
        // abs < bound
        fill_sub_borrow_packed(
            &mut bits,
            layout.range_result,
            layout.range_borrow,
            bit_width,
            (bound - 1) as u64,
            abs_value as u64,
        );

        flush_bit_buffer(&bits, &mut tb, row)?;

        tb.set_b32(phy_bus_value, row, Block32::from(value))?;
        tb.set_b32(phy_bus_idx, row, Block32::from(op.idx))?;
    }

    // Ungated add_carry_chain requires
    // value + complement = q on every row.
    if ops.len() < num_rows {
        bits.iter_mut().for_each(|w| *w = 0);

        pack_bits(&mut bits, layout.comp_bits, modulus as u64, bit_width);

        fill_add_carry_packed(
            &mut bits,
            layout.comp_carry,
            bit_width + 1,
            0,
            modulus as u64,
        );
        fill_sub_borrow_packed(
            &mut bits,
            layout.range_result,
            layout.range_borrow,
            bit_width,
            (bound - 1) as u64,
            0,
        );

        for row in ops.len()..num_rows {
            flush_bit_buffer(&bits, &mut tb, row)?;
        }
    }

    tb.fill_selector(phy_s_active, ops.len())?;

    Ok(tb.build())
}

// =================================================================
// Tests
// =================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use hekate_core::trace::Trace;
    use hekate_math::{Bit, Block128};

    type F = Block128;

    const Q: u32 = 8380417;
    const BIT_WIDTH: usize = 23;

    // ML-DSA-65:
    // γ₁ = 2^19 = 524288,
    // β = 196
    const BOUND: u32 = 524288 - 196; // 524092

    fn read_packed_bit(trace: &ColumnTrace, virt_col: usize, row: usize) -> bool {
        let packed_col = virt_col / 32;
        let bit_idx = virt_col % 32;
        let word = trace.columns[packed_col].as_b32_slice().unwrap()[row]
            .to_tower()
            .0;

        (word >> bit_idx) & 1 == 1
    }

    fn phy_sel(ly: &NormCheckLayout) -> usize {
        ly.num_packed_b32_cols + 2
    }

    #[test]
    fn layout_q8380417() {
        let ly = NormCheckLayout::compute(BIT_WIDTH);

        // 6*bw + 3 Bit columns + 2 B32 + 1 Bit(selector)
        assert_eq!(ly.num_bit_cols, 6 * BIT_WIDTH + 3);

        // 141 bits -> 5 packed B32 -> 160 expanded
        assert_eq!(ly.num_packed_b32_cols, 5);
        assert_eq!(ly.num_expanded_bits, 160);
        assert_eq!(ly.num_columns, 160 + 3);
        assert_eq!(ly.num_physical_columns, 5 + 3);
    }

    #[test]
    fn constraint_ast_builds() {
        let chiplet = NormCheckChiplet::new(Q, BOUND, 1024);
        let ast = Air::<F>::constraint_ast(&chiplet);

        // Non-trivial constraint count
        assert!(ast.roots.len() > 50);
    }

    #[test]
    fn air_declares_one_bus() {
        let chiplet = NormCheckChiplet::new(Q, BOUND, 1024);
        let checks = Air::<F>::permutation_checks(&chiplet);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].0, "norm_check");
    }

    #[test]
    fn trace_positive_value() {
        // z = 100 (positive, |z| = 100 < 524092)
        let ops = vec![NormCheckOp {
            value: 100,
            idx: 0,
            ram_addr: 0,
        }];
        let trace = generate_norm_check_trace(Q, BOUND, &ops, 4).unwrap();
        let ly = NormCheckLayout::compute(BIT_WIDTH);

        assert_eq!(trace.num_rows().unwrap(), 4);

        // Selector:
        // first row active
        let sel = trace.columns[phy_sel(&ly)].as_bit_slice().unwrap();
        assert_eq!(sel[0], Bit::ONE);
        assert_eq!(sel[1], Bit::ZERO);
    }

    #[test]
    fn trace_negative_value() {
        // z = q - 200 (represents -200, |z| = 200 < 524092)
        let neg_200 = Q - 200;
        let ops = vec![NormCheckOp {
            value: neg_200,
            idx: 1,
            ram_addr: 0,
        }];
        let trace = generate_norm_check_trace(Q, BOUND, &ops, 4).unwrap();
        let ly = NormCheckLayout::compute(BIT_WIDTH);

        // is_negative should be 1
        assert!(read_packed_bit(&trace, ly.is_negative, 0));

        // abs bits should encode 200
        let mut abs_val = 0u32;
        for k in 0..BIT_WIDTH {
            if read_packed_bit(&trace, ly.abs_bits + k, 0) {
                abs_val |= 1 << k;
            }
        }

        assert_eq!(abs_val, 200);
    }

    #[test]
    fn trace_boundary_values() {
        let half_q = (Q - 1) / 2;

        let ops = vec![
            // value = 0, positive, |z| = 0
            NormCheckOp {
                value: 0,
                idx: 0,
                ram_addr: 0,
            },
            // value = bound-1, positive max
            NormCheckOp {
                value: BOUND - 1,
                idx: 1,
                ram_addr: 0,
            },
            // value = q-1, negative, |z| = 1
            NormCheckOp {
                value: Q - 1,
                idx: 2,
                ram_addr: 0,
            },
            // value = q - (bound-1), negative max
            NormCheckOp {
                value: Q - (BOUND - 1),
                idx: 3,
                ram_addr: 0,
            },
        ];

        // Must not panic
        let trace = generate_norm_check_trace(Q, BOUND, &ops, 4).unwrap();

        let ly = NormCheckLayout::compute(BIT_WIDTH);

        // Check is_negative flags
        assert!(!read_packed_bit(&trace, ly.is_negative, 0)); // 0 ≤ half_q
        assert!(!read_packed_bit(&trace, ly.is_negative, 1)); // bound-1 ≤ half_q
        assert!(read_packed_bit(&trace, ly.is_negative, 2)); // q-1 > half_q
        assert!(read_packed_bit(&trace, ly.is_negative, 3)); // q-(bound-1) > half_q

        let _ = half_q;
    }

    #[test]
    fn trace_with_padding() {
        let ops = vec![NormCheckOp {
            value: 42,
            idx: 0,
            ram_addr: 0,
        }];
        let trace = generate_norm_check_trace(Q, BOUND, &ops, 8).unwrap();
        let ly = NormCheckLayout::compute(BIT_WIDTH);

        let sel = trace.columns[phy_sel(&ly)].as_bit_slice().unwrap();
        assert_eq!(sel[0], Bit::ONE);

        for &s in &sel[1..8] {
            assert_eq!(s, Bit::ZERO);
        }
    }

    #[test]
    fn bus_labels() {
        let chiplet = NormCheckChiplet::new(Q, BOUND, 1024);
        let spec = chiplet.linking_spec();

        assert_eq!(spec.sources.len(), 2);
        assert_eq!(spec.sources[0].1, b"kappa_nc_value");
        assert_eq!(spec.sources[1].1, b"kappa_nc_idx");
    }

    #[test]
    fn packed_trace_roundtrip() {
        let ops = vec![
            NormCheckOp {
                value: 100,
                idx: 0,
                ram_addr: 0,
            },
            NormCheckOp {
                value: Q - 200,
                idx: 1,
                ram_addr: 0,
            },
            NormCheckOp {
                value: BOUND - 1,
                idx: 2,
                ram_addr: 0,
            },
        ];

        let trace = generate_norm_check_trace(Q, BOUND, &ops, 4).unwrap();
        let ly = NormCheckLayout::compute(BIT_WIDTH);

        assert_eq!(trace.columns.len(), ly.num_physical_columns);

        let chiplet = NormCheckChiplet::new(Q, BOUND, 4);
        let variants = Air::<F>::virtual_expander(&chiplet)
            .unwrap()
            .expand_variants::<F, _>(&trace, 0)
            .unwrap();

        assert_eq!(variants.len(), ly.num_columns);

        // Tail padding bits must be zero
        let pad_bits = ly.num_expanded_bits - ly.num_bit_cols;
        assert!(pad_bits > 0);

        for k in ly.num_bit_cols..ly.num_expanded_bits {
            for row in 0..4 {
                assert!(!read_packed_bit(&trace, k, row));
            }
        }
    }
}
