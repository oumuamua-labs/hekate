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

//! 1-row-per-op ALU chiplet over
//! `int_arithmetic` primitives.
//! Bit width is u32 or u64;
//!
//! Opcodes:
//! ADD, SUB, AND, XOR, NOT, LT.

use crate::atoms::int_arith;
use alloc::borrow::ToOwned;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use hekate_core::errors;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate_math::{Bit, Block32, Block64, TowerField};
use hekate_program::Air;
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::{ConstraintSystem, Expr};
use hekate_program::define_columns;
use hekate_program::expander::VirtualExpander;
use hekate_program::permutation::{PermutationCheckSpec, REQUEST_IDX_LABEL, Source};
// =================================================================
// 0. LAYOUT
// =================================================================

const ARITH_OPERANDS: usize = 4;
const ARITH_SELECTORS: usize = 7;
const ARITH_BUS_TAIL_B32: usize = 2;
const ARITH_NUM_PHYSICAL: usize = ARITH_OPERANDS + ARITH_BUS_TAIL_B32 + ARITH_SELECTORS;

#[derive(Clone, Debug)]
pub struct IntArithmeticLayout {
    pub bit_width: usize,

    pub a_bits: usize,
    pub b_bits: usize,
    pub res_bits: usize,
    pub carry_bits: usize,

    pub val_a: usize,
    pub val_b: usize,
    pub val_res: usize,
    pub carry_packed: usize,
    pub opcode: usize,
    pub request_idx: usize,

    pub s_output: usize,
    pub s_add: usize,
    pub s_sub: usize,
    pub s_and: usize,
    pub s_xor: usize,
    pub s_not: usize,
    pub s_lt: usize,

    pub num_virtual_columns: usize,
    pub num_physical_columns: usize,
}

impl IntArithmeticLayout {
    pub fn compute(bit_width: usize) -> Self {
        assert!(
            bit_width == 32 || bit_width == 64,
            "IntArithmeticLayout::compute: bit_width must be 32 or 64, got {bit_width}"
        );

        let a_bits = 0;
        let b_bits = a_bits + bit_width;
        let res_bits = b_bits + bit_width;
        let carry_bits = res_bits + bit_width;

        let val_a = carry_bits + bit_width;
        let val_b = val_a + 1;
        let val_res = val_a + 2;
        let carry_packed = val_a + 3;
        let opcode = val_a + 4;
        let request_idx = val_a + 5;

        let s_output = request_idx + 1;
        let s_add = s_output + 1;
        let s_sub = s_output + 2;
        let s_and = s_output + 3;
        let s_xor = s_output + 4;
        let s_not = s_output + 5;
        let s_lt = s_output + 6;

        Self {
            bit_width,
            a_bits,
            b_bits,
            res_bits,
            carry_bits,
            val_a,
            val_b,
            val_res,
            carry_packed,
            opcode,
            request_idx,
            s_output,
            s_add,
            s_sub,
            s_and,
            s_xor,
            s_not,
            s_lt,
            num_virtual_columns: s_lt + 1,
            num_physical_columns: ARITH_NUM_PHYSICAL,
        }
    }

    pub fn operand_storage(&self) -> ColumnType {
        match self.bit_width {
            64 => ColumnType::B64,
            _ => ColumnType::B32,
        }
    }

    pub fn build_physical_layout(&self) -> Vec<ColumnType> {
        let operand = self.operand_storage();

        let mut layout = Vec::with_capacity(self.num_physical_columns);

        for _ in 0..ARITH_OPERANDS {
            layout.push(operand);
        }

        for _ in 0..ARITH_BUS_TAIL_B32 {
            layout.push(ColumnType::B32);
        }

        for _ in 0..ARITH_SELECTORS {
            layout.push(ColumnType::Bit);
        }

        assert_eq!(layout.len(), self.num_physical_columns);

        layout
    }

    pub fn build_virtual_layout(&self) -> Vec<ColumnType> {
        let operand = self.operand_storage();

        let mut layout = Vec::with_capacity(self.num_virtual_columns);

        for _ in 0..(ARITH_OPERANDS * self.bit_width) {
            layout.push(ColumnType::Bit);
        }

        for _ in 0..ARITH_OPERANDS {
            layout.push(operand);
        }

        for _ in 0..ARITH_BUS_TAIL_B32 {
            layout.push(ColumnType::B32);
        }

        for _ in 0..ARITH_SELECTORS {
            layout.push(ColumnType::Bit);
        }

        assert_eq!(layout.len(), self.num_virtual_columns);

        layout
    }

    pub fn build_expander(&self) -> errors::Result<VirtualExpander> {
        let operand = self.operand_storage();
        VirtualExpander::new()
            .expand_bits(ARITH_OPERANDS, operand)
            .reuse_pass_through(0, ARITH_OPERANDS)
            .pass_through(ARITH_BUS_TAIL_B32, ColumnType::B32)
            .control_bits(ARITH_SELECTORS)
            .build()
    }
}

// =================================================================
// 1. CHIPLET DEFINITION
// =================================================================

#[derive(Clone, Debug)]
pub struct IntArithmeticChiplet {
    pub bit_width: usize,
    pub num_rows: usize,

    bus_id: String,
    layout: IntArithmeticLayout,
    expander: VirtualExpander,
    physical_layout: Vec<ColumnType>,
}

impl IntArithmeticChiplet {
    pub const BUS_ID: &'static str = "int_arith_link";

    pub fn new(bit_width: usize, num_rows: usize) -> errors::Result<Self> {
        if bit_width != 32 && bit_width != 64 {
            return Err(errors::Error::Protocol {
                protocol: "arithmetic",
                message: "bit_width must be 32 or 64",
            });
        }

        if !num_rows.is_power_of_two() {
            return Err(errors::Error::Protocol {
                protocol: "arithmetic",
                message: "num_rows must be a power of 2",
            });
        }

        let layout = IntArithmeticLayout::compute(bit_width);
        let expander = layout.build_expander()?;
        let physical_layout = layout.build_physical_layout();

        Ok(Self {
            bit_width,
            num_rows,
            bus_id: Self::BUS_ID.to_owned(),
            layout,
            expander,
            physical_layout,
        })
    }

    pub fn with_bus_id(mut self, bus_id: impl Into<String>) -> Self {
        self.bus_id = bus_id.into();
        self
    }

    pub fn bus_id(&self) -> &str {
        &self.bus_id
    }

    pub fn linking_spec(&self) -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (Source::Column(self.layout.val_a), b"kappa_val_a" as &[u8]),
                (Source::Column(self.layout.val_b), b"kappa_val_b" as &[u8]),
                (
                    Source::Column(self.layout.val_res),
                    b"kappa_val_res" as &[u8],
                ),
                (Source::Column(self.layout.opcode), b"kappa_opcode" as &[u8]),
                (Source::Column(self.layout.request_idx), REQUEST_IDX_LABEL),
            ],
            Some(self.layout.s_output),
        )
    }

    pub fn layout(&self) -> &IntArithmeticLayout {
        &self.layout
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn num_columns(&self) -> usize {
        self.layout.num_virtual_columns
    }
}

define_columns! {
    pub CpuArithColumns {
        VAL_A: B32,
        VAL_B: B32,
        VAL_RES: B32,
        OPCODE: B32,
        SELECTOR: Bit,
    }
}

/// CPU arithmetic event column layout.
///
/// The CPU side emits arithmetic operation requests.
#[derive(Clone, Debug)]
pub struct CpuIntArithmeticUnit;

impl CpuIntArithmeticUnit {
    pub fn linking_spec() -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(CpuArithColumns::VAL_A),
                    b"kappa_val_a" as &[u8],
                ),
                (
                    Source::Column(CpuArithColumns::VAL_B),
                    b"kappa_val_b" as &[u8],
                ),
                (
                    Source::Column(CpuArithColumns::VAL_RES),
                    b"kappa_val_res" as &[u8],
                ),
                (
                    Source::Column(CpuArithColumns::OPCODE),
                    b"kappa_opcode" as &[u8],
                ),
                (Source::RowIndexLeBytes(4), REQUEST_IDX_LABEL),
            ],
            Some(CpuArithColumns::SELECTOR),
        )
    }

    pub fn num_columns(&self) -> usize {
        CpuArithColumns::NUM_COLUMNS
    }
}

// =================================================================
// 2. OPERATIONS & TRACE GENERATION
// =================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArithmeticOpcode {
    ADD = 0,
    SUB = 1,
    AND = 2,
    XOR = 3,
    NOT = 4,
    LT = 5,
}

impl<F: TowerField> Air<F> for IntArithmeticChiplet {
    fn name(&self) -> String {
        "ArithmeticChiplet".to_string()
    }

    fn num_columns(&self) -> usize {
        self.layout.num_virtual_columns
    }

    fn column_layout(&self) -> &[ColumnType] {
        &self.physical_layout
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![(self.bus_id.clone(), self.linking_spec())]
    }

    fn virtual_expander(&self) -> Option<&VirtualExpander> {
        Some(&self.expander)
    }

    /// Operand columns (val_a, val_b, val_res, opcode) are CPU-host owned.
    /// Chiplet pins only internal state: carry_packed, request_idx.
    fn constraint_ast(&self) -> ConstraintAst<F> {
        let ly = &self.layout;
        let cs = ConstraintSystem::<F>::new();

        let bw = ly.bit_width;
        let one = cs.one();
        let zero = cs.constant(F::ZERO);

        let a: Vec<_> = (0..bw).map(|k| cs.col(ly.a_bits + k)).collect();
        let b: Vec<_> = (0..bw).map(|k| cs.col(ly.b_bits + k)).collect();
        let r: Vec<_> = (0..bw).map(|k| cs.col(ly.res_bits + k)).collect();
        let w: Vec<_> = (0..bw).map(|k| cs.col(ly.carry_bits + k)).collect();

        let val_a = cs.col(ly.val_a);
        let val_b = cs.col(ly.val_b);
        let val_res = cs.col(ly.val_res);
        let carry_packed = cs.col(ly.carry_packed);
        let opcode = cs.col(ly.opcode);

        let s_out = cs.col(ly.s_output);
        let s_add = cs.col(ly.s_add);
        let s_sub = cs.col(ly.s_sub);
        let s_and = cs.col(ly.s_and);
        let s_xor = cs.col(ly.s_xor);
        let s_not = cs.col(ly.s_not);
        let s_lt = cs.col(ly.s_lt);

        for s in [s_out, s_add, s_sub, s_and, s_xor, s_not, s_lt] {
            cs.assert_boolean(s);
        }

        let opcode_selectors = [s_add, s_sub, s_and, s_xor, s_not, s_lt];
        for i in 0..opcode_selectors.len() {
            for j in (i + 1)..opcode_selectors.len() {
                cs.constrain(opcode_selectors[i] * opcode_selectors[j]);
            }
        }

        cs.constrain(s_add + s_sub + s_and + s_xor + s_not + s_lt + s_out);

        cs.constrain(
            opcode
                + s_sub
                + cs.scale(F::from(2u128), s_and)
                + cs.scale(F::from(3u128), s_xor)
                + cs.scale(F::from(4u128), s_not)
                + cs.scale(F::from(5u128), s_lt),
        );

        int_arith::bit_packing(&cs, val_a, &a);
        int_arith::bit_packing(&cs, val_b, &b);
        int_arith::bit_packing(&cs, val_res, &r);
        int_arith::bit_packing(&cs, carry_packed, &w);

        let mut carry_full: Vec<Expr<F>> = Vec::with_capacity(bw + 1);
        carry_full.push(zero);

        for &w_k in w.iter() {
            carry_full.push(w_k);
        }

        int_arith::add_carry_chain_with_carry_in_gated(&cs, s_add, &a, &b, &r, &carry_full);
        int_arith::sub_borrow_chain_gated(&cs, s_sub, &a, &b, &r, &carry_full);

        for k in 0..bw {
            cs.assert_zero_when(s_and, r[k] + a[k] * b[k]);
            cs.assert_zero_when(s_xor, r[k] + a[k] + b[k]);
            cs.assert_zero_when(s_not, r[k] + a[k] + one);
        }

        for k in 0..bw {
            let w_k = if k == 0 { zero } else { w[k - 1] };
            let w_next = w[k];

            cs.assert_zero_when(
                s_lt,
                w_next + b[k] + a[k] * b[k] + w_k + a[k] * w_k + b[k] * w_k,
            );
        }

        cs.assert_zero_when(s_lt, r[0] + w[bw - 1]);

        for &bit in &r[1..bw] {
            cs.assert_zero_when(s_lt, bit);
        }

        let request_idx = cs.col(ly.request_idx);
        let not_active = one + s_out;

        cs.assert_zero_when(not_active, request_idx);

        let carry_unused = one + s_add + s_sub + s_lt;
        cs.assert_zero_when(carry_unused, carry_packed);

        cs.build()
    }
}

// =================================================================
// 4. PACKED TRACE GENERATION
// (1 row per op, u32 + u64)
// =================================================================

const PHY_VAL_A: usize = 0;
const PHY_VAL_B: usize = 1;
const PHY_VAL_RES: usize = 2;
const PHY_CARRY: usize = 3;
const PHY_OPCODE: usize = 4;
const PHY_REQUEST_IDX: usize = 5;
const PHY_S_OUTPUT: usize = 6;
const PHY_S_ADD: usize = 7;
const PHY_S_SUB: usize = 8;
const PHY_S_AND: usize = 9;
const PHY_S_XOR: usize = 10;
const PHY_S_NOT: usize = 11;
const PHY_S_LT: usize = 12;

#[derive(Clone, Copy, Debug)]
pub enum IntArithmeticOp {
    U32 {
        op: ArithmeticOpcode,
        a: u32,
        b: u32,

        /// Partner-side emit row index.
        request_idx: u32,
    },
    U64 {
        op: ArithmeticOpcode,
        a: u64,
        b: u64,

        /// Partner-side emit row index.
        request_idx: u32,
    },
}

pub fn generate_arithmetic_trace(
    ops: &[IntArithmeticOp],
    layout: &IntArithmeticLayout,
    num_rows: usize,
) -> errors::Result<ColumnTrace> {
    if !num_rows.is_power_of_two() {
        return Err(errors::Error::Protocol {
            protocol: "arithmetic",
            message: "num_rows must be a power of 2",
        });
    }

    if ops.len() > num_rows {
        return Err(errors::Error::Protocol {
            protocol: "arithmetic",
            message: "ops overflow num_rows",
        });
    }

    let bw = layout.bit_width;
    let num_vars = num_rows.trailing_zeros() as usize;
    let phy = layout.build_physical_layout();

    let mut tb = TraceBuilder::new(&phy, num_vars)?;

    for (i, call) in ops.iter().enumerate() {
        let (opcode, a, b, request_idx) = match *call {
            IntArithmeticOp::U32 {
                op,
                a,
                b,
                request_idx,
            } => {
                if bw != 32 {
                    return Err(errors::Error::Protocol {
                        protocol: "arithmetic",
                        message: "U32 op given with layout.bit_width != 32",
                    });
                }

                (op, a as u64, b as u64, request_idx)
            }
            IntArithmeticOp::U64 {
                op,
                a,
                b,
                request_idx,
            } => {
                if bw != 64 {
                    return Err(errors::Error::Protocol {
                        protocol: "arithmetic",
                        message: "U64 op given with layout.bit_width != 64",
                    });
                }

                (op, a, b, request_idx)
            }
        };

        let (res, carry_word) = compute_packed_row(opcode, a, b, bw);

        write_packed_operand(&mut tb, PHY_VAL_A, i, bw, a)?;
        write_packed_operand(&mut tb, PHY_VAL_B, i, bw, b)?;
        write_packed_operand(&mut tb, PHY_VAL_RES, i, bw, res)?;
        write_packed_operand(&mut tb, PHY_CARRY, i, bw, carry_word)?;

        tb.set_b32(PHY_OPCODE, i, Block32::from(opcode as u8 as u32))?;
        tb.set_b32(PHY_REQUEST_IDX, i, Block32::from(request_idx))?;
        tb.set_bit(PHY_S_OUTPUT, i, Bit::ONE)?;

        let sel_col = match opcode {
            ArithmeticOpcode::ADD => PHY_S_ADD,
            ArithmeticOpcode::SUB => PHY_S_SUB,
            ArithmeticOpcode::AND => PHY_S_AND,
            ArithmeticOpcode::XOR => PHY_S_XOR,
            ArithmeticOpcode::NOT => PHY_S_NOT,
            ArithmeticOpcode::LT => PHY_S_LT,
        };

        tb.set_bit(sel_col, i, Bit::ONE)?;
    }

    Ok(tb.build())
}

fn write_packed_operand(
    tb: &mut TraceBuilder,
    phy_col: usize,
    row: usize,
    bit_width: usize,
    value: u64,
) -> errors::Result<()> {
    match bit_width {
        32 => tb.set_b32(phy_col, row, Block32::from(value as u32)),
        64 => tb.set_b64(phy_col, row, Block64::from(value)),
        _ => Err(errors::Error::Protocol {
            protocol: "arithmetic",
            message: "bit_width must be 32 or 64",
        }),
    }
}

fn compute_packed_row(op: ArithmeticOpcode, a: u64, b: u64, bit_width: usize) -> (u64, u64) {
    let mask = if bit_width == 64 {
        u64::MAX
    } else {
        (1u64 << bit_width) - 1
    };

    match op {
        ArithmeticOpcode::ADD => {
            let mut c = 0u64;
            let mut sum = 0u64;
            let mut carry_word = 0u64;

            for k in 0..bit_width {
                let a_k = (a >> k) & 1;
                let b_k = (b >> k) & 1;
                let s_k = a_k ^ b_k ^ c;
                let c_next = (a_k & b_k) | (c & (a_k ^ b_k));

                sum |= s_k << k;
                carry_word |= c_next << k;

                c = c_next;
            }

            (sum & mask, carry_word & mask)
        }
        ArithmeticOpcode::SUB => {
            let mut w = 0u64;
            let mut res = 0u64;
            let mut borrow_word = 0u64;

            for k in 0..bit_width {
                let a_k = (a >> k) & 1;
                let b_k = (b >> k) & 1;
                let not_a = a_k ^ 1;
                let r_k = a_k ^ b_k ^ w;
                let w_next = (not_a & b_k) | ((not_a ^ b_k) & w);

                res |= r_k << k;
                borrow_word |= w_next << k;

                w = w_next;
            }

            (res & mask, borrow_word & mask)
        }
        ArithmeticOpcode::AND => ((a & b) & mask, 0u64),
        ArithmeticOpcode::XOR => ((a ^ b) & mask, 0u64),
        ArithmeticOpcode::NOT => ((!a) & mask, 0u64),
        ArithmeticOpcode::LT => {
            let mut w = 0u64;
            let mut borrow_word = 0u64;

            for k in 0..bit_width {
                let a_k = (a >> k) & 1;
                let b_k = (b >> k) & 1;
                let not_a = a_k ^ 1;
                let w_next = (not_a & b_k) | ((not_a ^ b_k) & w);

                borrow_word |= w_next << k;
                w = w_next;
            }

            (w & mask, borrow_word & mask)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hekate_core::trace::TraceColumn;
    use hekate_math::{Block128, Flat, FlatPromote};

    type F = Block128;

    fn physical_row_bytes(layout: &[ColumnType]) -> usize {
        layout.iter().map(|c| c.byte_size()).sum()
    }

    fn op_add(a: u32, b: u32) -> IntArithmeticOp {
        IntArithmeticOp::U32 {
            op: ArithmeticOpcode::ADD,
            a,
            b,
            request_idx: 0,
        }
    }

    fn op_sub(a: u32, b: u32) -> IntArithmeticOp {
        IntArithmeticOp::U32 {
            op: ArithmeticOpcode::SUB,
            a,
            b,
            request_idx: 0,
        }
    }

    fn op_and(a: u32, b: u32) -> IntArithmeticOp {
        IntArithmeticOp::U32 {
            op: ArithmeticOpcode::AND,
            a,
            b,
            request_idx: 0,
        }
    }

    fn op_xor(a: u32, b: u32) -> IntArithmeticOp {
        IntArithmeticOp::U32 {
            op: ArithmeticOpcode::XOR,
            a,
            b,
            request_idx: 0,
        }
    }

    fn op_not(a: u32) -> IntArithmeticOp {
        IntArithmeticOp::U32 {
            op: ArithmeticOpcode::NOT,
            a,
            b: 0,
            request_idx: 0,
        }
    }

    fn op_lt(a: u32, b: u32) -> IntArithmeticOp {
        IntArithmeticOp::U32 {
            op: ArithmeticOpcode::LT,
            a,
            b,
            request_idx: 0,
        }
    }

    fn val_b32(trace: &ColumnTrace, col: usize, row: usize) -> u128 {
        F::promote_flat(trace.columns[col].as_b32_slice().unwrap()[row])
            .to_tower()
            .0
    }

    fn val_b64(trace: &ColumnTrace, col: usize, row: usize) -> u128 {
        F::promote_flat(trace.columns[col].as_b64_slice().unwrap()[row])
            .to_tower()
            .0
    }

    fn extract_row_bytes(trace: &ColumnTrace, row: usize, buf: &mut Vec<u8>) {
        buf.clear();

        for col in trace.columns.iter() {
            match col {
                TraceColumn::Bit(v) => buf.push(v[row].get()),
                TraceColumn::B8(v) => buf.push(v[row].into_raw().0),
                TraceColumn::B16(v) => buf.extend_from_slice(&v[row].into_raw().0.to_le_bytes()),
                TraceColumn::B32(v) => buf.extend_from_slice(&v[row].into_raw().0.to_le_bytes()),
                TraceColumn::B64(v) => buf.extend_from_slice(&v[row].into_raw().0.to_le_bytes()),
                TraceColumn::B128(v) => buf.extend_from_slice(&v[row].into_raw().0.to_le_bytes()),
            }
        }
    }

    fn expand_row(expander: &VirtualExpander, trace: &ColumnTrace, row: usize) -> Vec<Flat<F>> {
        let mut bytes = Vec::with_capacity(expander.physical_row_bytes());
        extract_row_bytes(trace, row, &mut bytes);

        let mut out = Vec::with_capacity(expander.num_virtual_columns());
        expander.parse_row(&bytes, &mut out).unwrap();

        out
    }

    #[test]
    fn compute_32_indices() {
        let ly = IntArithmeticLayout::compute(32);

        assert_eq!(ly.a_bits, 0);
        assert_eq!(ly.b_bits, 32);
        assert_eq!(ly.res_bits, 64);
        assert_eq!(ly.carry_bits, 96);

        assert_eq!(ly.val_a, 128);
        assert_eq!(ly.val_b, 129);
        assert_eq!(ly.val_res, 130);
        assert_eq!(ly.carry_packed, 131);
        assert_eq!(ly.opcode, 132);
        assert_eq!(ly.request_idx, 133);

        assert_eq!(ly.s_output, 134);
        assert_eq!(ly.s_add, 135);
        assert_eq!(ly.s_sub, 136);
        assert_eq!(ly.s_and, 137);
        assert_eq!(ly.s_xor, 138);
        assert_eq!(ly.s_not, 139);
        assert_eq!(ly.s_lt, 140);

        assert_eq!(ly.num_virtual_columns, 141);
        assert_eq!(ly.num_physical_columns, 13);
    }

    #[test]
    fn compute_64_indices() {
        let ly = IntArithmeticLayout::compute(64);

        assert_eq!(ly.a_bits, 0);
        assert_eq!(ly.b_bits, 64);
        assert_eq!(ly.res_bits, 128);
        assert_eq!(ly.carry_bits, 192);

        assert_eq!(ly.val_a, 256);
        assert_eq!(ly.val_b, 257);
        assert_eq!(ly.val_res, 258);
        assert_eq!(ly.carry_packed, 259);
        assert_eq!(ly.opcode, 260);
        assert_eq!(ly.request_idx, 261);

        assert_eq!(ly.s_output, 262);
        assert_eq!(ly.s_lt, 268);

        assert_eq!(ly.num_virtual_columns, 269);
        assert_eq!(ly.num_physical_columns, 13);
    }

    #[test]
    fn physical_layout_32_is_31_bytes_per_row() {
        let ly = IntArithmeticLayout::compute(32);
        let phy = ly.build_physical_layout();

        assert_eq!(phy.len(), 13);
        assert_eq!(physical_row_bytes(&phy), 31);
    }

    #[test]
    fn physical_layout_64_is_47_bytes_per_row() {
        let ly = IntArithmeticLayout::compute(64);
        let phy = ly.build_physical_layout();

        assert_eq!(phy.len(), 13);
        assert_eq!(physical_row_bytes(&phy), 47);
    }

    #[test]
    fn physical_column_types_32() {
        let phy = IntArithmeticLayout::compute(32).build_physical_layout();
        let expected = [
            ColumnType::B32,
            ColumnType::B32,
            ColumnType::B32,
            ColumnType::B32,
            ColumnType::B32,
            ColumnType::B32,
            ColumnType::Bit,
            ColumnType::Bit,
            ColumnType::Bit,
            ColumnType::Bit,
            ColumnType::Bit,
            ColumnType::Bit,
            ColumnType::Bit,
        ];
        assert_eq!(phy, expected);
    }

    #[test]
    fn physical_column_types_64() {
        let phy = IntArithmeticLayout::compute(64).build_physical_layout();
        let expected = [
            ColumnType::B64,
            ColumnType::B64,
            ColumnType::B64,
            ColumnType::B64,
            ColumnType::B32,
            ColumnType::B32,
            ColumnType::Bit,
            ColumnType::Bit,
            ColumnType::Bit,
            ColumnType::Bit,
            ColumnType::Bit,
            ColumnType::Bit,
            ColumnType::Bit,
        ];
        assert_eq!(phy, expected);
    }

    #[test]
    fn virtual_layout_32_matches_expander() {
        let ly = IntArithmeticLayout::compute(32);
        let expander = ly.build_expander().expect("expander 32");

        assert_eq!(expander.num_virtual_columns(), 141);
        assert_eq!(expander.num_physical_columns(), 13);
        assert_eq!(expander.physical_row_bytes(), 31);
        assert_eq!(
            expander.virtual_layout(),
            ly.build_virtual_layout().as_slice()
        );
    }

    #[test]
    fn virtual_layout_64_matches_expander() {
        let ly = IntArithmeticLayout::compute(64);
        let expander = ly.build_expander().expect("expander 64");

        assert_eq!(expander.num_virtual_columns(), 269);
        assert_eq!(expander.num_physical_columns(), 13);
        assert_eq!(expander.physical_row_bytes(), 47);
        assert_eq!(
            expander.virtual_layout(),
            ly.build_virtual_layout().as_slice()
        );
    }

    #[test]
    fn operand_storage_selects_by_width() {
        assert_eq!(
            IntArithmeticLayout::compute(32).operand_storage(),
            ColumnType::B32
        );
        assert_eq!(
            IntArithmeticLayout::compute(64).operand_storage(),
            ColumnType::B64
        );
    }

    #[test]
    fn new_rejects_invalid_bit_width() {
        assert!(IntArithmeticChiplet::new(16, 16).is_err());
        assert!(IntArithmeticChiplet::new(33, 16).is_err());
        assert!(IntArithmeticChiplet::new(128, 16).is_err());
    }

    #[test]
    fn new_rejects_non_power_of_two_rows() {
        assert!(IntArithmeticChiplet::new(32, 10).is_err());
        assert!(IntArithmeticChiplet::new(32, 3).is_err());
    }

    #[test]
    fn new_accepts_valid_shapes() {
        let chip32 = IntArithmeticChiplet::new(32, 16).unwrap();
        assert_eq!(chip32.num_columns(), 141);
        assert_eq!(chip32.num_rows(), 16);

        let chip64 = IntArithmeticChiplet::new(64, 8).unwrap();
        assert_eq!(chip64.num_columns(), 269);
        assert_eq!(chip64.num_rows(), 8);
    }

    #[test]
    fn add_u32_basic_and_overflow() {
        let ly = IntArithmeticLayout::compute(32);
        let ops = vec![
            op_add(10, 20),
            op_add(u32::MAX, 1),
            op_add(u32::MAX, u32::MAX),
        ];

        let trace = generate_arithmetic_trace(&ops, &ly, 8).unwrap();

        assert_eq!(val_b32(&trace, 2, 0), 30);
        assert_eq!(val_b32(&trace, 2, 1), 0);
        assert_eq!(val_b32(&trace, 2, 2), 0xFFFFFFFEu128);
    }

    #[test]
    fn sub_u32_basic_and_underflow() {
        let ly = IntArithmeticLayout::compute(32);
        let ops = vec![op_sub(50, 20), op_sub(10, 20), op_sub(50, 50)];

        let trace = generate_arithmetic_trace(&ops, &ly, 4).unwrap();

        assert_eq!(val_b32(&trace, 2, 0), 30);
        assert_eq!(val_b32(&trace, 2, 1), 0xFFFFFFF6u128);
        assert_eq!(val_b32(&trace, 2, 2), 0);
    }

    #[test]
    fn bitwise_u32_ops() {
        let ly = IntArithmeticLayout::compute(32);
        let ops = vec![
            op_and(0xFFFF0000, 0x00FFFF00),
            op_xor(0xAAAAAAAA, 0x55555555),
            op_not(0),
        ];

        let trace = generate_arithmetic_trace(&ops, &ly, 4).unwrap();

        assert_eq!(val_b32(&trace, 2, 0), 0x00FF0000);
        assert_eq!(val_b32(&trace, 2, 1), 0xFFFFFFFF);
        assert_eq!(val_b32(&trace, 2, 2), 0xFFFFFFFF);
    }

    #[test]
    fn lt_u32_boundaries() {
        let ly = IntArithmeticLayout::compute(32);
        let ops = vec![
            op_lt(10, 20),
            op_lt(20, 10),
            op_lt(20, 20),
            op_lt(0, u32::MAX),
        ];

        let trace = generate_arithmetic_trace(&ops, &ly, 4).unwrap();

        assert_eq!(val_b32(&trace, 2, 0), 1);
        assert_eq!(val_b32(&trace, 2, 1), 0);
        assert_eq!(val_b32(&trace, 2, 2), 0);
        assert_eq!(val_b32(&trace, 2, 3), 1);
    }

    #[test]
    fn add_u64_basic_and_overflow() {
        let ly = IntArithmeticLayout::compute(64);
        let ops = vec![
            IntArithmeticOp::U64 {
                op: ArithmeticOpcode::ADD,
                a: 10,
                b: 20,
                request_idx: 0,
            },
            IntArithmeticOp::U64 {
                op: ArithmeticOpcode::ADD,
                a: u64::MAX,
                b: 1,
                request_idx: 1,
            },
            IntArithmeticOp::U64 {
                op: ArithmeticOpcode::ADD,
                a: u64::MAX,
                b: u64::MAX,
                request_idx: 2,
            },
        ];

        let trace = generate_arithmetic_trace(&ops, &ly, 4).unwrap();

        assert_eq!(val_b64(&trace, 2, 0), 30);
        assert_eq!(val_b64(&trace, 2, 1), 0);
        assert_eq!(val_b64(&trace, 2, 2), u64::MAX as u128 - 1);
    }

    #[test]
    fn sub_u64_underflow() {
        let ly = IntArithmeticLayout::compute(64);
        let ops = vec![IntArithmeticOp::U64 {
            op: ArithmeticOpcode::SUB,
            a: 0,
            b: 1,
            request_idx: 0,
        }];

        let trace = generate_arithmetic_trace(&ops, &ly, 2).unwrap();

        assert_eq!(val_b64(&trace, 2, 0), u64::MAX as u128);
    }

    #[test]
    fn trace_rejects_width_mismatch_u32_on_64() {
        let ly = IntArithmeticLayout::compute(64);
        let ops = vec![op_add(1, 2)];

        assert!(generate_arithmetic_trace(&ops, &ly, 2).is_err());
    }

    #[test]
    fn trace_rejects_width_mismatch_u64_on_32() {
        let ly = IntArithmeticLayout::compute(32);
        let ops = vec![IntArithmeticOp::U64 {
            op: ArithmeticOpcode::ADD,
            a: 1,
            b: 2,
            request_idx: 0,
        }];

        assert!(generate_arithmetic_trace(&ops, &ly, 2).is_err());
    }

    #[test]
    fn trace_rejects_ops_overflow_num_rows() {
        let ly = IntArithmeticLayout::compute(32);
        let ops = vec![op_add(1, 2), op_add(3, 4), op_add(5, 6)];

        assert!(generate_arithmetic_trace(&ops, &ly, 2).is_err());
    }

    #[test]
    fn trace_rejects_non_power_of_two_rows() {
        let ly = IntArithmeticLayout::compute(32);
        let ops = vec![op_add(1, 2)];

        assert!(generate_arithmetic_trace(&ops, &ly, 3).is_err());
    }

    #[test]
    fn constraints_satisfied_on_u32_honest_trace() {
        let chiplet = IntArithmeticChiplet::new(32, 8).unwrap();
        let ly = chiplet.layout().clone();
        let ops = vec![
            op_add(10, 20),
            op_sub(100, 50),
            op_and(0xFF, 0x0F),
            op_xor(0xAA, 0x55),
            op_not(0),
            op_lt(10, 20),
        ];

        let trace = generate_arithmetic_trace(&ops, &ly, 8).unwrap();
        let expander = <IntArithmeticChiplet as Air<F>>::virtual_expander(&chiplet).unwrap();
        let ast = <IntArithmeticChiplet as Air<F>>::constraint_ast(&chiplet);

        let rows: Vec<Vec<Flat<F>>> = (0..8).map(|r| expand_row(expander, &trace, r)).collect();

        for row in 0..8 {
            let next_row = (row + 1) % 8;
            let evals = ast.evaluate(&rows[row], &rows[next_row]);

            for (i, val) in evals.iter().enumerate() {
                assert_eq!(
                    *val,
                    Flat::from_raw(F::ZERO),
                    "constraint {i} failed at row {row} (u32)",
                );
            }
        }
    }

    #[test]
    fn constraints_satisfied_on_u64_honest_trace() {
        let chiplet = IntArithmeticChiplet::new(64, 4).unwrap();
        let ly = chiplet.layout().clone();
        let ops = vec![
            IntArithmeticOp::U64 {
                op: ArithmeticOpcode::ADD,
                a: 0xDEADBEEF,
                b: 0xCAFEBABE,
                request_idx: 0,
            },
            IntArithmeticOp::U64 {
                op: ArithmeticOpcode::SUB,
                a: 100,
                b: 200,
                request_idx: 1,
            },
            IntArithmeticOp::U64 {
                op: ArithmeticOpcode::LT,
                a: u64::MAX - 1,
                b: u64::MAX,
                request_idx: 2,
            },
        ];

        let trace = generate_arithmetic_trace(&ops, &ly, 4).unwrap();
        let expander = <IntArithmeticChiplet as Air<F>>::virtual_expander(&chiplet).unwrap();
        let ast = <IntArithmeticChiplet as Air<F>>::constraint_ast(&chiplet);

        let rows: Vec<Vec<Flat<F>>> = (0..4).map(|r| expand_row(expander, &trace, r)).collect();

        for row in 0..4 {
            let next_row = (row + 1) % 4;
            let evals = ast.evaluate(&rows[row], &rows[next_row]);

            for (i, val) in evals.iter().enumerate() {
                assert_eq!(
                    *val,
                    Flat::from_raw(F::ZERO),
                    "constraint {i} failed at row {row} (u64)",
                );
            }
        }
    }
}
