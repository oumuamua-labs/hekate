// SPDX-License-Identifier: Apache-2.0
// This file is part of the hekate-math project.
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

//! Basemul Chiplet for ML-KEM / ML-DSA.
//!
//! Verifies the NTT basemul addition structure:
//! each row proves `a + b ≡ c (mod q)`.
//!
//! The individual mod-q products (a0*b0, a1*b1, etc.)
//! are verified by the NTT chiplet's MulOnly mode.
//! This chiplet verifies how those products combine:
//!
//!   r0 = a0*b0 + a1*b1*ζ     (mod q)
//!   r1 = a0*b1 + a1*b0       (mod q)
//!   r2 = a2*b2 - a3*b3*ζ     (mod q)
//!   r3 = a2*b3 + a3*b2       (mod q)
//!
//! Each equation is one row. Subtraction is encoded
//! by swapping operands: `r2 + p33z ≡ p22 (mod q)`.
//!
//! 4 rows per basemul unit × 64 units per polynomial
//! pair = 256 rows per pointwise multiply.
//!
//! Bus: "basemul" (internal to ML-KEM/ML-DSA composite)

use super::utils::{
    fill_add_carry_packed, fill_sub_borrow_packed, flush_bit_buffer, pack_bits, pack_one,
};
use crate::atoms::int_arith;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder, TraceCompatibleField};
use hekate_math::{Block32, TowerField};
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::constraint::ConstraintAst;
use hekate_program::define_columns;
use hekate_program::expander::VirtualExpander;
use hekate_program::permutation::{PermutationCheckSpec, Source, REQUEST_IDX_LABEL};
use hekate_program::Air;

// =================================================================
// CPU-side Columns (for main trace linking)
// =================================================================

define_columns! {
    /// CPU-side columns for the basemul bus.
    /// Use in the main trace to link to the
    /// BasemulChiplet.
    pub CpuBasemulColumns {
        BM_A: B32,
        BM_B: B32,
        BM_C: B32,
        BM_IDX: B32,
        SELECTOR: Bit,
    }
}

/// CPU-side basemul linking unit.
pub struct CpuBasemulUnit;

impl CpuBasemulUnit {
    /// Linking spec matching BasemulChiplet.
    /// Challenge labels must be identical.
    pub fn linking_spec() -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(CpuBasemulColumns::BM_A),
                    b"kappa_bm_a" as &[u8],
                ),
                (
                    Source::Column(CpuBasemulColumns::BM_B),
                    b"kappa_bm_b" as &[u8],
                ),
                (
                    Source::Column(CpuBasemulColumns::BM_C),
                    b"kappa_bm_c" as &[u8],
                ),
                (
                    Source::Column(CpuBasemulColumns::BM_IDX),
                    b"kappa_bm_idx" as &[u8],
                ),
                (Source::RowIndexLeBytes(4), REQUEST_IDX_LABEL),
            ],
            Some(CpuBasemulColumns::SELECTOR),
        )
    }
}

// =================================================================
// Column Layout
// =================================================================

/// Column index map for the Basemul chiplet.
/// Each row verifies:
/// a + b = c + flag * q (mod q).
#[derive(Clone, Debug)]
pub struct BasemulLayout {
    pub bit_width: usize,

    // Operand bit decompositions
    pub a_bits: usize,
    pub b_bits: usize,
    pub c_bits: usize,

    // Mod-add scratch
    pub lhs_result: usize,
    pub lhs_carry: usize,
    pub rhs_result: usize,
    pub rhs_carry: usize,
    pub flag: usize,
    pub range_result: usize,
    pub range_borrow: usize,

    pub num_bit_cols: usize,
    pub num_packed_b32_cols: usize,
    pub num_expanded_bits: usize,

    // B32 bus columns
    pub bus_a: usize,
    pub bus_b: usize,
    pub bus_c: usize,
    pub bus_idx: usize,
    pub request_idx: usize,

    // Control
    pub s_active: usize,

    pub num_columns: usize,
    pub num_physical_columns: usize,

    // Sub-layout
    pub mod_add_layout: int_arith::ModAddLayout,
}

impl BasemulLayout {
    pub fn compute(bit_width: usize) -> Self {
        let mod_add_layout = int_arith::mod_add_scratch_count(bit_width);

        let mut offset = 0usize;
        let mut alloc = |n: usize| -> usize {
            let start = offset;
            offset += n;

            start
        };

        let a_bits = alloc(bit_width);
        let b_bits = alloc(bit_width);
        let c_bits = alloc(bit_width);

        let lhs_result = alloc(mod_add_layout.result_width);
        let lhs_carry = alloc(mod_add_layout.carry_width);
        let rhs_result = alloc(mod_add_layout.result_width);
        let rhs_carry = alloc(mod_add_layout.carry_width);
        let flag = alloc(1);
        let range_result = alloc(mod_add_layout.range_result_width);
        let range_borrow = alloc(mod_add_layout.range_borrow_width);

        let num_bit_cols = offset;
        let num_packed_b32_cols = num_bit_cols.div_ceil(32);
        let num_expanded_bits = num_packed_b32_cols * 32;

        let bus_a = num_expanded_bits;
        let bus_b = num_expanded_bits + 1;
        let bus_c = num_expanded_bits + 2;
        let bus_idx = num_expanded_bits + 3;
        let request_idx = num_expanded_bits + 4;

        let s_active = num_expanded_bits + 5;

        let num_columns = num_expanded_bits + 6;
        let num_physical_columns = num_packed_b32_cols + 6;

        BasemulLayout {
            bit_width,
            a_bits,
            b_bits,
            c_bits,
            lhs_result,
            lhs_carry,
            rhs_result,
            rhs_carry,
            flag,
            range_result,
            range_borrow,
            num_bit_cols,
            num_packed_b32_cols,
            num_expanded_bits,
            bus_a,
            bus_b,
            bus_c,
            bus_idx,
            request_idx,
            s_active,
            num_columns,
            num_physical_columns,
            mod_add_layout,
        }
    }

    pub fn build_virtual_layout(&self) -> Vec<ColumnType> {
        let mut layout = Vec::with_capacity(self.num_columns);

        for _ in 0..self.num_expanded_bits {
            layout.push(ColumnType::Bit);
        }

        for _ in 0..5 {
            layout.push(ColumnType::B32);
        }

        layout.push(ColumnType::Bit);

        debug_assert_eq!(layout.len(), self.num_columns);

        layout
    }

    pub fn build_physical_layout(&self) -> Vec<ColumnType> {
        let mut layout = Vec::with_capacity(self.num_physical_columns);

        for _ in 0..self.num_packed_b32_cols {
            layout.push(ColumnType::B32);
        }

        for _ in 0..5 {
            layout.push(ColumnType::B32);
        }

        layout.push(ColumnType::Bit);

        debug_assert_eq!(layout.len(), self.num_physical_columns);

        layout
    }
}

// =================================================================
// Basemul Chiplet
// =================================================================

/// Basemul Chiplet.
///
/// Each row verifies `a + b ≡ c (mod q)`.
///
/// For basemul subtraction (r2 = p22 - p33z),
/// the caller dispatches (a=r2, b=p33z, c=p22)
/// so the constraint checks r2 + p33z ≡ p22.
#[derive(Clone, Debug)]
pub struct BasemulChiplet {
    pub modulus: u32,
    pub bit_width: usize,
    pub num_rows: usize,

    layout: BasemulLayout,
    expander: VirtualExpander,
}

impl BasemulChiplet {
    pub const BUS_ID: &'static str = "basemul";

    pub fn new(modulus: u32, num_rows: usize) -> Self {
        assert!(num_rows.is_power_of_two());

        let bit_width = 32 - modulus.leading_zeros() as usize;
        let layout = BasemulLayout::compute(bit_width);

        let expander = VirtualExpander::new()
            .expand_bits(layout.num_packed_b32_cols, ColumnType::B32)
            .pass_through(5, ColumnType::B32)
            .control_bits(1)
            .build()
            .expect("BasemulChiplet expander");

        Self {
            modulus,
            bit_width,
            num_rows,
            layout,
            expander,
        }
    }

    pub fn layout(&self) -> &BasemulLayout {
        &self.layout
    }

    pub fn linking_spec(&self) -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (Source::Column(self.layout.bus_a), b"kappa_bm_a" as &[u8]),
                (Source::Column(self.layout.bus_b), b"kappa_bm_b" as &[u8]),
                (Source::Column(self.layout.bus_c), b"kappa_bm_c" as &[u8]),
                (
                    Source::Column(self.layout.bus_idx),
                    b"kappa_bm_idx" as &[u8],
                ),
                (Source::Column(self.layout.request_idx), REQUEST_IDX_LABEL),
            ],
            Some(self.layout.s_active),
        )
    }
}

// =================================================================
// Air Implementation
// =================================================================

impl<F: TowerField + TraceCompatibleField> Air<F> for BasemulChiplet {
    fn name(&self) -> String {
        "BasemulChiplet".to_string()
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
        build_basemul_constraints(self.modulus, self.bit_width, &self.layout)
    }
}

// =================================================================
// Constraint Generation
// =================================================================

fn build_basemul_constraints<F: TowerField>(
    modulus: u32,
    bit_width: usize,
    ly: &BasemulLayout,
) -> ConstraintAst<F> {
    let cs = ConstraintSystem::<F>::new();

    let s_active = cs.col(ly.s_active);
    cs.assert_boolean(s_active);

    let a: Vec<_> = (0..bit_width).map(|k| cs.col(ly.a_bits + k)).collect();
    let b: Vec<_> = (0..bit_width).map(|k| cs.col(ly.b_bits + k)).collect();
    let c: Vec<_> = (0..bit_width).map(|k| cs.col(ly.c_bits + k)).collect();

    // Booleanity (s_active gated)
    for &bit in a.iter().chain(b.iter()).chain(c.iter()) {
        cs.assert_zero_when(s_active, bit * (bit + cs.one()));
    }

    // Mod-add:
    // a + b = c + flag*q
    let lhs_r: Vec<_> = (0..ly.mod_add_layout.result_width)
        .map(|k| cs.col(ly.lhs_result + k))
        .collect();
    let lhs_c: Vec<_> = (0..ly.mod_add_layout.carry_width)
        .map(|k| cs.col(ly.lhs_carry + k))
        .collect();
    let rhs_r: Vec<_> = (0..ly.mod_add_layout.result_width)
        .map(|k| cs.col(ly.rhs_result + k))
        .collect();
    let rhs_c: Vec<_> = (0..ly.mod_add_layout.carry_width)
        .map(|k| cs.col(ly.rhs_carry + k))
        .collect();
    let flag_expr = cs.col(ly.flag);
    let rng_r: Vec<_> = (0..ly.mod_add_layout.range_result_width)
        .map(|k| cs.col(ly.range_result + k))
        .collect();
    let rng_w: Vec<_> = (0..ly.mod_add_layout.range_borrow_width)
        .map(|k| cs.col(ly.range_borrow + k))
        .collect();

    int_arith::mod_add(
        &cs,
        &a,
        &b,
        &c,
        &int_arith::ModAddWitness {
            lhs_result: &lhs_r,
            lhs_carry: &lhs_c,
            rhs_result: &rhs_r,
            rhs_carry: &rhs_c,
            flag: flag_expr,
            range_result: &rng_r,
            range_borrow: &rng_w,
        },
        modulus,
    );

    // Packing constraints:
    // bind B32 bus columns to bit decompositions.
    int_arith::bit_packing(&cs, cs.col(ly.bus_a), &a);
    int_arith::bit_packing(&cs, cs.col(ly.bus_b), &b);
    int_arith::bit_packing(&cs, cs.col(ly.bus_c), &c);

    // Tail padding bits in the
    // last packed B32 column
    // must be zero on every row.
    for k in ly.num_bit_cols..ly.num_expanded_bits {
        cs.constrain(cs.col(k));
    }

    // Pin bus_idx / request_idx to 0 on padding rows;
    // bit_packing already pins bus_a/b/c.
    let one = cs.one();
    let not_active = one - s_active;

    cs.assert_zero_when(not_active, cs.col(ly.bus_idx));
    cs.assert_zero_when(not_active, cs.col(ly.request_idx));

    cs.build()
}

// =================================================================
// Trace Generation
// =================================================================

/// A single basemul addition operation.
#[derive(Clone, Debug)]
pub struct BasemulOp {
    /// First operand (mod q).
    pub a: u32,

    /// Second operand (mod q).
    pub b: u32,

    /// Expected sum:
    /// (a + b) mod q.
    pub c: u32,

    /// Bus index.
    pub idx: u32,

    /// RAM address for BM-RAM binding.
    pub ram_addr: u32,

    /// Partner-side row index.
    pub request_idx: u32,
}

/// Generate the basemul chiplet trace.
pub fn generate_basemul_trace(
    modulus: u32,
    ops: &[BasemulOp],
    num_rows: usize,
) -> hekate_core::errors::Result<ColumnTrace> {
    assert!(num_rows.is_power_of_two());
    assert!(ops.len() <= num_rows);

    let bit_width = 32 - modulus.leading_zeros() as usize;
    let layout = BasemulLayout::compute(bit_width);
    let physical = layout.build_physical_layout();

    let num_vars = num_rows.trailing_zeros() as usize;
    let num_packed = layout.num_packed_b32_cols;

    let mut tb = TraceBuilder::new(&physical, num_vars)?;

    let phy_bus_a = num_packed;
    let phy_bus_b = num_packed + 1;
    let phy_bus_c = num_packed + 2;
    let phy_bus_idx = num_packed + 3;
    let phy_request_idx = num_packed + 4;
    let phy_s_active = num_packed + 5;

    let mut bits = vec![0u32; num_packed];

    for (row, op) in ops.iter().enumerate() {
        assert!(
            op.a < modulus,
            "basemul: op[{row}].a={} >= modulus {modulus}",
            op.a,
        );
        assert!(
            op.b < modulus,
            "basemul: op[{row}].b={} >= modulus {modulus}",
            op.b,
        );
        assert!(
            op.c < modulus,
            "basemul: op[{row}].c={} >= modulus {modulus}",
            op.c,
        );
        assert_eq!(
            (op.a + op.b) % modulus,
            op.c,
            "basemul: op[{row}] a+b mod q != c: {}+{} mod {modulus} = {}, expected {}",
            op.a,
            op.b,
            (op.a + op.b) % modulus,
            op.c,
        );

        bits.iter_mut().for_each(|w| *w = 0);

        let sum = op.a + op.b;
        let flag = if sum >= modulus { 1u32 } else { 0u32 };
        let rhs_b = flag * modulus;
        let rhs = op.c + rhs_b;

        pack_bits(&mut bits, layout.a_bits, op.a as u64, bit_width);
        pack_bits(&mut bits, layout.b_bits, op.b as u64, bit_width);
        pack_bits(&mut bits, layout.c_bits, op.c as u64, bit_width);

        // LHS:
        // a + b
        pack_bits(&mut bits, layout.lhs_result, sum as u64, bit_width);
        fill_add_carry_packed(
            &mut bits,
            layout.lhs_carry,
            layout.mod_add_layout.carry_width,
            op.a as u64,
            op.b as u64,
        );

        // RHS:
        // c + flag*q
        pack_bits(&mut bits, layout.rhs_result, rhs as u64, bit_width);
        fill_add_carry_packed(
            &mut bits,
            layout.rhs_carry,
            layout.mod_add_layout.carry_width,
            op.c as u64,
            rhs_b as u64,
        );

        pack_one(&mut bits, layout.flag, flag == 1);

        // Range check:
        // c < modulus
        fill_sub_borrow_packed(
            &mut bits,
            layout.range_result,
            layout.range_borrow,
            bit_width,
            (modulus - 1) as u64,
            op.c as u64,
        );

        flush_bit_buffer(&bits, &mut tb, row)?;

        tb.set_b32(phy_bus_a, row, Block32::from(op.a))?;
        tb.set_b32(phy_bus_b, row, Block32::from(op.b))?;
        tb.set_b32(phy_bus_c, row, Block32::from(op.c))?;
        tb.set_b32(phy_bus_idx, row, Block32::from(op.idx))?;
        tb.set_b32(phy_request_idx, row, Block32::from(op.request_idx))?;
    }

    // Ghost Protocol
    tb.fill_selector(phy_s_active, ops.len())?;

    // Padding range checks
    for pad_row in ops.len()..num_rows {
        bits.iter_mut().for_each(|w| *w = 0);

        fill_sub_borrow_packed(
            &mut bits,
            layout.range_result,
            layout.range_borrow,
            bit_width,
            (modulus - 1) as u64,
            0,
        );
        flush_bit_buffer(&bits, &mut tb, pad_row)?;
    }

    Ok(tb.build())
}

// =================================================================
// Tests
// =================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use hekate_core::trace::Trace;
    use hekate_math::{Bit, Block128, Flat};

    type F = Block128;

    const Q: u32 = 3329;
    const BW: usize = 12;

    fn phy_s_active(ly: &BasemulLayout) -> usize {
        ly.num_packed_b32_cols + 5
    }

    fn phy_bus(ly: &BasemulLayout, offset: usize) -> usize {
        ly.num_packed_b32_cols + offset
    }

    fn read_virtual_bit(trace: &ColumnTrace, virt_col: usize, row: usize) -> u32 {
        let packed_col = virt_col / 32;
        let bit = virt_col % 32;
        let word = trace.columns[packed_col].as_b32_slice().unwrap()[row]
            .to_tower()
            .0;

        (word >> bit) & 1
    }

    #[test]
    fn layout_q3329() {
        let ly = BasemulLayout::compute(BW);

        // 3*12 operand bits + mod_add scratch
        assert_eq!(ly.a_bits, 0);
        assert_eq!(ly.b_bits, BW);
        assert_eq!(ly.c_bits, 2 * BW);
        assert!(ly.num_columns > 3 * BW);
    }

    #[test]
    fn constraint_ast_builds() {
        let chiplet = BasemulChiplet::new(Q, 4);
        let ast = Air::<F>::constraint_ast(&chiplet);
        assert!(ast.roots.len() > 30);
    }

    #[test]
    fn air_declares_one_bus() {
        let chiplet = BasemulChiplet::new(Q, 4);
        let checks = Air::<F>::permutation_checks(&chiplet);
        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].0, "basemul");
    }

    #[test]
    fn trace_simple_addition() {
        // 1000 + 2000 = 3000 (mod 3329)
        let ops = vec![BasemulOp {
            a: 1000,
            b: 2000,
            c: 3000,
            idx: 0,
            ram_addr: 0,
            request_idx: 0,
        }];
        let trace = generate_basemul_trace(Q, &ops, 4).unwrap();
        assert_eq!(trace.num_rows().unwrap(), 4);

        let ly = BasemulLayout::compute(BW);
        let sel = trace.columns[phy_s_active(&ly)].as_bit_slice().unwrap();

        assert_eq!(sel[0], Bit::ONE);
        assert_eq!(sel[1], Bit::ZERO);
    }

    #[test]
    fn trace_overflow_addition() {
        // 2000 + 2000 = 4000 mod 3329 = 671
        let ops = vec![BasemulOp {
            a: 2000,
            b: 2000,
            c: (2000 + 2000) % Q,
            idx: 0,
            ram_addr: 0,
            request_idx: 0,
        }];
        let trace = generate_basemul_trace(Q, &ops, 4).unwrap();
        let ly = BasemulLayout::compute(BW);

        // Verify flag is set (overflow)
        assert_eq!(read_virtual_bit(&trace, ly.flag, 0), 1);
    }

    #[test]
    fn trace_subtraction_encoding() {
        // Verify p22 - p33z = r2 encoded as r2 + p33z = p22
        let p22 = 2500u32;
        let p33z = 1000u32;
        let r2 = (p22 + Q - p33z) % Q; // 1500

        let ops = vec![BasemulOp {
            a: r2,
            b: p33z,
            c: p22,
            idx: 0,
            ram_addr: 0,
            request_idx: 0,
        }];
        let trace = generate_basemul_trace(Q, &ops, 4).unwrap();
        let ly = BasemulLayout::compute(BW);

        let bus_c = trace.columns[phy_bus(&ly, 2)].as_b32_slice().unwrap();
        assert_eq!(bus_c[0].to_tower(), Block32::from(p22));
    }

    #[test]
    fn trace_full_basemul_unit() {
        // One complete basemul unit (4 rows):
        // a = (100, 200), b = (300, 400), ζ = 17
        let a0 = 100u32;
        let a1 = 200u32;
        let b0 = 300u32;
        let b1 = 400u32;
        let zeta = 17u32;

        let p00 = (a0 * b0) % Q; // 30000 % 3329 = 17
        let p11 = (a1 * b1) % Q; // 80000 % 3329 = 8
        let p11z = (p11 * zeta) % Q; // 136 % 3329 = 136
        let p01 = (a0 * b1) % Q; // 40000 % 3329 = 17
        let p10 = (a1 * b0) % Q; // 60000 % 3329 = 8

        let r0 = (p00 + p11z) % Q;
        let r1 = (p01 + p10) % Q;

        // Second pair with -ζ
        let a2 = 500u32;
        let a3 = 600u32;
        let b2 = 700u32;
        let b3 = 800u32;

        let p22 = (a2 * b2) % Q;
        let p33 = (a3 * b3) % Q;
        let p33z = (p33 * zeta) % Q;
        let r2 = (p22 + Q - p33z) % Q; // subtraction
        let p23 = (a2 * b3) % Q;
        let p32 = (a3 * b2) % Q;
        let r3 = (p23 + p32) % Q;

        let ops = vec![
            BasemulOp {
                a: p00,
                b: p11z,
                c: r0,
                idx: 0,
                ram_addr: 0,
                request_idx: 0,
            },
            BasemulOp {
                a: p01,
                b: p10,
                c: r1,
                idx: 1,
                ram_addr: 0,
                request_idx: 1,
            },
            BasemulOp {
                a: r2,
                b: p33z,
                c: p22,
                idx: 2,
                ram_addr: 0,
                request_idx: 2,
            }, // sub encoded as add
            BasemulOp {
                a: p23,
                b: p32,
                c: r3,
                idx: 3,
                ram_addr: 0,
                request_idx: 3,
            },
        ];

        let trace = generate_basemul_trace(Q, &ops, 4).unwrap();
        assert_eq!(trace.num_rows().unwrap(), 4);

        let ly = BasemulLayout::compute(BW);
        let sel = trace.columns[phy_s_active(&ly)].as_bit_slice().unwrap();

        for &s in &sel[..4] {
            assert_eq!(s, Bit::ONE);
        }
    }

    #[test]
    fn trace_zero_operands() {
        let ops = vec![BasemulOp {
            a: 0,
            b: 0,
            c: 0,
            idx: 0,
            ram_addr: 0,
            request_idx: 0,
        }];
        let trace = generate_basemul_trace(Q, &ops, 4).unwrap();
        assert_eq!(trace.num_rows().unwrap(), 4);
    }

    #[test]
    fn trace_boundary_values() {
        // (q-1) + (q-1) = 2*(q-1) mod q = q-2
        let ops = vec![BasemulOp {
            a: Q - 1,
            b: Q - 1,
            c: (2 * (Q - 1)) % Q,
            idx: 0,
            ram_addr: 0,
            request_idx: 0,
        }];
        let trace = generate_basemul_trace(Q, &ops, 4).unwrap();
        assert_eq!(trace.num_rows().unwrap(), 4);
    }

    #[test]
    fn bus_labels() {
        let chiplet = BasemulChiplet::new(Q, 4);
        let spec = chiplet.linking_spec();
        assert_eq!(spec.sources.len(), 5);
        assert_eq!(spec.sources[0].1, b"kappa_bm_a");
        assert_eq!(spec.sources[1].1, b"kappa_bm_b");
        assert_eq!(spec.sources[2].1, b"kappa_bm_c");
        assert_eq!(spec.sources[3].1, b"kappa_bm_idx");
        assert_eq!(spec.sources[4].1, REQUEST_IDX_LABEL);
    }

    #[test]
    fn packing_constraints_present() {
        // Build constraints for the basemul chiplet
        // and verify packing constraints are included.
        //
        // Without packing: the bus B32 columns would
        // be disconnected from the arithmetic Bit columns.
        // A malicious prover could transmit arbitrary
        // values through the bus while proving correct
        // arithmetic on different values.
        let chiplet = BasemulChiplet::new(Q, 4);
        let ast = Air::<F>::constraint_ast(&chiplet);

        // Build constraints WITHOUT packing for comparison.
        // The difference must be exactly 3 (bus_a, bus_b, bus_c).
        let ly = BasemulLayout::compute(BW);
        let cs_no_pack = ConstraintSystem::<F>::new();
        let s_active = cs_no_pack.col(ly.s_active);

        cs_no_pack.assert_boolean(s_active);

        let a: Vec<_> = (0..BW).map(|k| cs_no_pack.col(ly.a_bits + k)).collect();
        let b: Vec<_> = (0..BW).map(|k| cs_no_pack.col(ly.b_bits + k)).collect();
        let c: Vec<_> = (0..BW).map(|k| cs_no_pack.col(ly.c_bits + k)).collect();

        for &bit in a.iter().chain(b.iter()).chain(c.iter()) {
            cs_no_pack.assert_zero_when(s_active, bit * (bit + cs_no_pack.one()));
        }

        let lhs_r: Vec<_> = (0..ly.mod_add_layout.result_width)
            .map(|k| cs_no_pack.col(ly.lhs_result + k))
            .collect();
        let lhs_c: Vec<_> = (0..ly.mod_add_layout.carry_width)
            .map(|k| cs_no_pack.col(ly.lhs_carry + k))
            .collect();
        let rhs_r: Vec<_> = (0..ly.mod_add_layout.result_width)
            .map(|k| cs_no_pack.col(ly.rhs_result + k))
            .collect();
        let rhs_c: Vec<_> = (0..ly.mod_add_layout.carry_width)
            .map(|k| cs_no_pack.col(ly.rhs_carry + k))
            .collect();
        let flag_expr = cs_no_pack.col(ly.flag);
        let rng_r: Vec<_> = (0..ly.mod_add_layout.range_result_width)
            .map(|k| cs_no_pack.col(ly.range_result + k))
            .collect();
        let rng_w: Vec<_> = (0..ly.mod_add_layout.range_borrow_width)
            .map(|k| cs_no_pack.col(ly.range_borrow + k))
            .collect();

        int_arith::mod_add(
            &cs_no_pack,
            &a,
            &b,
            &c,
            &int_arith::ModAddWitness {
                lhs_result: &lhs_r,
                lhs_carry: &lhs_c,
                rhs_result: &rhs_r,
                rhs_carry: &rhs_c,
                flag: flag_expr,
                range_result: &rng_r,
                range_borrow: &rng_w,
            },
            Q,
        );

        let ast_no_pack = cs_no_pack.build();

        // 3 bus packings + tail-padding-bit zero pins
        // + 2 padding-row pins (bus_idx, request_idx).
        let pad_bits = ly.num_expanded_bits - ly.num_bit_cols;
        let expected_delta = 3 + pad_bits + 2;
        assert_eq!(
            ast.roots.len() - ast_no_pack.roots.len(),
            expected_delta,
            "Expected {expected_delta} packing+padding constraints \
             (3 bus + {pad_bits} pad + 2 padding-row pins), \
             got delta={} (with={}, without={})",
            ast.roots.len() as i64 - ast_no_pack.roots.len() as i64,
            ast.roots.len(),
            ast_no_pack.roots.len(),
        );
    }

    #[test]
    fn trace_padding_rows() {
        let ops = vec![BasemulOp {
            a: 42,
            b: 58,
            c: 100,
            idx: 0,
            ram_addr: 0,
            request_idx: 0,
        }];
        let trace = generate_basemul_trace(Q, &ops, 8).unwrap();
        let ly = BasemulLayout::compute(BW);

        let sel = trace.columns[phy_s_active(&ly)].as_bit_slice().unwrap();
        assert_eq!(sel[0], Bit::ONE);

        for &s in &sel[1..8] {
            assert_eq!(s, Bit::ZERO);
        }
    }

    #[test]
    fn packed_basemul_trace_roundtrip_q3329() {
        let ops = vec![BasemulOp {
            a: 1000,
            b: 2000,
            c: 3000,
            idx: 7,
            ram_addr: 0,
            request_idx: 0,
        }];

        let trace = generate_basemul_trace(Q, &ops, 2).unwrap();
        let layout = BasemulLayout::compute(BW);

        let chiplet = BasemulChiplet::new(Q, 2);
        let variants = Air::<F>::virtual_expander(&chiplet)
            .unwrap()
            .expand_variants(&trace, 0)
            .unwrap();

        assert_eq!(variants.len(), layout.num_columns);

        // Operand bits decode to original values.
        for k in 0..BW {
            let val = variants[layout.a_bits + k].get_at(0);
            let expected = if ((1000u32 >> k) & 1) == 1 {
                Flat::from_raw(F::ONE)
            } else {
                Flat::from_raw(F::ZERO)
            };

            assert_eq!(val, expected, "a_bit[{k}] mismatch");
        }

        for k in 0..BW {
            let val = variants[layout.b_bits + k].get_at(0);
            let expected = if ((2000u32 >> k) & 1) == 1 {
                Flat::from_raw(F::ONE)
            } else {
                Flat::from_raw(F::ZERO)
            };

            assert_eq!(val, expected, "b_bit[{k}] mismatch");
        }

        // Bus B32 columns are passthrough, verify
        // the underlying physical column directly.
        let bus_a_slice = trace.columns[phy_bus(&layout, 0)].as_b32_slice().unwrap();
        assert_eq!(bus_a_slice[0].to_tower(), Block32::from(1000u32));

        let bus_idx_slice = trace.columns[phy_bus(&layout, 3)].as_b32_slice().unwrap();
        assert_eq!(bus_idx_slice[0].to_tower(), Block32::from(7u32));

        // Selector decodes from physical Bit column.
        let sel = variants[layout.s_active].get_at(0);
        assert_eq!(sel, Flat::from_raw(F::ONE));

        // Tail padding bits are zero.
        for (k, variant) in variants
            .iter()
            .enumerate()
            .take(layout.num_expanded_bits)
            .skip(layout.num_bit_cols)
        {
            let val = variant.get_at(0);
            assert_eq!(val, Flat::from_raw(F::ZERO), "padding bit {k} not zero");
        }
    }
}
