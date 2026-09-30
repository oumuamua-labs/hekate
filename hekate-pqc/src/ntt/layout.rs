// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec::Vec;
use hekate_core::errors::{Error, Result};
use hekate_core::trace::ColumnType;
use hekate_gadgets::atoms::int_arith::{
    ModAddLayout, ModReductionLayout, SchoolbookMulLayout, mod_add_scratch_count,
    mod_reduction_scratch_count, schoolbook_mul_layout,
};
use hekate_math::{HardwareField, TowerField};
use hekate_program::circuit::{Circuit, Col, ColRange};

use crate::gadgets::{ModAddCols, MulModCols};

const B16_COLUMNS: usize = 5;
const B32_COLUMNS: usize = 5;
const FLAG_COLUMNS: usize = 10;

/// Column positions of the NTT table.
#[derive(Clone, Debug)]
pub struct NttLayout {
    pub bit_width: usize,
    pub registers: usize,

    pub a: usize,
    pub b: usize,
    pub w: usize,
    pub x: usize,
    pub y: usize,

    pub s1: usize,
    pub s1_add: ModAddCols,

    pub s2: usize,
    pub s2_sub: ModAddCols,
    pub s2_range_result: usize,
    pub s2_range_borrow: usize,

    pub mul_pp0: usize,
    pub mul_sums: Vec<(usize, usize)>,
    pub mul_carries: Vec<(usize, usize)>,
    pub product: usize,

    pub quot: usize,
    pub p: usize,
    pub quot_x_q: usize,
    pub red_results: Vec<(usize, usize)>,
    pub red_carries: Vec<(usize, usize)>,
    pub red_add_carry: usize,
    pub red_range_result: usize,
    pub red_range_borrow: usize,

    pub num_bits: usize,
    pub num_packed: usize,

    pub v_in1: Col,
    pub v_in2: Col,
    pub v_out1: Col,
    pub v_out2: Col,
    pub tw: Col,
    pub reg: ColRange,

    pub poly1: Col,
    pub poly2: Col,
    pub poly3: Col,
    pub pos1: Col,
    pub pos2: Col,

    pub gs: Col,
    pub mac: Col,
    pub neg: Col,
    pub za: Col,
    pub c1: Col,
    pub c2: Col,
    pub p1: Col,
    pub p2: Col,
    pub bcont: Col,
    pub rcont: Col,
    pub sel: ColRange,
    pub selin: ColRange,

    pub mul: SchoolbookMulLayout,
    pub red: ModReductionLayout,
    pub add: ModAddLayout,
}

impl NttLayout {
    pub(crate) fn declare<F: TowerField + HardwareField>(
        cx: &mut Circuit<F>,
        q: u32,
        bit_width: usize,
        registers: usize,
    ) -> Self {
        let mul = schoolbook_mul_layout(bit_width, bit_width);
        let red = mod_reduction_scratch_count(bit_width, q);
        let add = mod_add_scratch_count(bit_width);

        let mut next = 0usize;

        let mut alloc = |n: usize| {
            let start = next;
            next += n;

            start
        };

        let a = alloc(bit_width);
        let b = alloc(bit_width);
        let w = alloc(bit_width);
        let x = alloc(bit_width);
        let y = alloc(bit_width);

        let s1 = alloc(bit_width);
        let s1_add = ModAddCols::alloc(&add, &mut alloc);

        let s2 = alloc(bit_width);
        let s2_sub = ModAddCols::alloc(&add, &mut alloc);
        let s2_range_result = alloc(bit_width);
        let s2_range_borrow = alloc(bit_width + 1);

        let mul_pp0 = alloc(mul.pp0_width);
        let mul_sums = mul.sum_widths.iter().map(|&n| (alloc(n), n)).collect();
        let mul_carries = mul.carry_widths.iter().map(|&n| (alloc(n), n)).collect();
        let product = alloc(mul.product_width);

        let quot = alloc(bit_width);
        let p = alloc(bit_width);
        let quot_x_q = alloc(red.product_width);

        let red_results = red
            .mul_layout
            .scratch_result_widths
            .iter()
            .map(|&n| (alloc(n), n))
            .collect();

        let red_carries = red
            .mul_layout
            .scratch_carry_widths
            .iter()
            .map(|&n| (alloc(n), n))
            .collect();

        let red_add_carry = alloc(red.add_carry_width);
        let red_range_result = alloc(red.range_result_width);
        let red_range_borrow = alloc(red.range_borrow_width);

        let num_bits = next;
        let num_packed = num_bits.div_ceil(32);

        cx.expand_bits(num_packed, ColumnType::B32);

        let words = cx.columns(B32_COLUMNS, ColumnType::B32);
        let reg = cx.columns(registers, ColumnType::B32);
        let labels = cx.columns(B16_COLUMNS, ColumnType::B16);
        let flags = cx.columns(FLAG_COLUMNS, ColumnType::Bit);
        let sel = cx.columns(registers, ColumnType::Bit);
        let selin = cx.columns(registers, ColumnType::Bit);

        Self {
            bit_width,
            registers,
            a,
            b,
            w,
            x,
            y,
            s1,
            s1_add,
            s2,
            s2_sub,
            s2_range_result,
            s2_range_borrow,
            mul_pp0,
            mul_sums,
            mul_carries,
            product,
            quot,
            p,
            quot_x_q,
            red_results,
            red_carries,
            red_add_carry,
            red_range_result,
            red_range_borrow,
            num_bits,
            num_packed,
            v_in1: words.at(0),
            v_in2: words.at(1),
            v_out1: words.at(2),
            v_out2: words.at(3),
            tw: words.at(4),
            reg,
            poly1: labels.at(0),
            poly2: labels.at(1),
            poly3: labels.at(2),
            pos1: labels.at(3),
            pos2: labels.at(4),
            gs: flags.at(0),
            mac: flags.at(1),
            neg: flags.at(2),
            za: flags.at(3),
            c1: flags.at(4),
            c2: flags.at(5),
            p1: flags.at(6),
            p2: flags.at(7),
            bcont: flags.at(8),
            rcont: flags.at(9),
            sel,
            selin,
            mul,
            red,
            add,
        }
    }

    /// Physical column behind `col`.
    pub fn physical(&self, col: Col) -> Result<usize> {
        match col.index().checked_sub(32 * self.num_packed) {
            Some(plain) => Ok(self.num_packed + plain),
            None => Err(Error::Protocol {
                protocol: "ntt_layout",
                message: "virtual bit has no committed column of its own",
            }),
        }
    }

    pub(crate) fn mul_mod(&self) -> MulModCols {
        MulModCols {
            pp0: (self.mul_pp0, self.mul.pp0_width),
            sums: self.mul_sums.clone(),
            carries: self.mul_carries.clone(),
            product: (self.product, self.mul.product_width),
            quot: (self.quot, self.bit_width),
            rem: (self.p, self.bit_width),
            quot_x_q: (self.quot_x_q, self.red.product_width),
            red_results: self.red_results.clone(),
            red_carries: self.red_carries.clone(),
            red_add_carry: (self.red_add_carry, self.red.add_carry_width),
            red_range_result: (self.red_range_result, self.red.range_result_width),
            red_range_borrow: (self.red_range_borrow, self.red.range_borrow_width),
        }
    }
}
