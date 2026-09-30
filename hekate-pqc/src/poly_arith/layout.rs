// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::errors::{Error, Result};
use hekate_core::trace::ColumnType;
use hekate_gadgets::atoms::int_arith::{
    ModAddLayout, mod_add_scratch_count, mod_reduction_scratch_count, schoolbook_mul_layout,
};
use hekate_math::{HardwareField, TowerField};
use hekate_program::circuit::{Circuit, Col, ColRange};

use crate::gadgets::{ModAddCols, MulModCols};

const B16_COLUMNS: usize = 6;
const B32_COLUMNS: usize = 9;
const FLAG_COLUMNS: usize = 6;

/// Column positions of the PolyArith table.
#[derive(Clone, Debug)]
pub struct PolyArithLayout {
    pub bit_width: usize,
    pub registers: usize,

    pub a0: usize,
    pub a1: usize,
    pub b0: usize,
    pub b1: usize,
    pub in0: usize,
    pub in1: usize,
    pub gamma: usize,

    pub t00: MulModCols,
    pub t11: MulModCols,
    pub g: MulModCols,
    pub t01: MulModCols,
    pub t10: MulModCols,

    pub u0: usize,
    pub u0_add: ModAddCols,
    pub out0: usize,
    pub out0_add: ModAddCols,
    pub u1: usize,
    pub u1_add: ModAddCols,
    pub out1: usize,
    pub out1_add: ModAddCols,

    pub num_bits: usize,
    pub num_packed: usize,

    pub v_a0: Col,
    pub v_a1: Col,
    pub v_b0: Col,
    pub v_b1: Col,
    pub v_in0: Col,
    pub v_in1: Col,
    pub v_out0: Col,
    pub v_out1: Col,
    pub v_gamma: Col,
    pub h0: ColRange,
    pub h1: ColRange,

    pub poly_a: Col,
    pub poly_b: Col,
    pub poly_s: Col,
    pub poly_o: Col,
    pub pos0: Col,
    pub pos1: Col,

    pub active: Col,
    pub btake: Col,
    pub seed: Col,
    pub emit: Col,
    pub bcont: Col,
    pub rcont: Col,
    pub sel: ColRange,
    pub selin: ColRange,
    pub copy: Option<(Col, Col)>,

    pub add: ModAddLayout,
}

impl PolyArithLayout {
    pub(crate) fn declare<F: TowerField + HardwareField>(
        cx: &mut Circuit<F>,
        q: u32,
        bit_width: usize,
        registers: usize,
        copies: bool,
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

        let a0 = alloc(bit_width);
        let a1 = alloc(bit_width);
        let b0 = alloc(bit_width);
        let b1 = alloc(bit_width);
        let in0 = alloc(bit_width);
        let in1 = alloc(bit_width);
        let gamma = alloc(bit_width);

        let t00 = MulModCols::alloc(bit_width, &mul, &red, &mut alloc);
        let t11 = MulModCols::alloc(bit_width, &mul, &red, &mut alloc);
        let g = MulModCols::alloc(bit_width, &mul, &red, &mut alloc);
        let t01 = MulModCols::alloc(bit_width, &mul, &red, &mut alloc);
        let t10 = MulModCols::alloc(bit_width, &mul, &red, &mut alloc);

        let u0 = alloc(bit_width);
        let u0_add = ModAddCols::alloc(&add, &mut alloc);
        let out0 = alloc(bit_width);
        let out0_add = ModAddCols::alloc(&add, &mut alloc);
        let u1 = alloc(bit_width);
        let u1_add = ModAddCols::alloc(&add, &mut alloc);
        let out1 = alloc(bit_width);
        let out1_add = ModAddCols::alloc(&add, &mut alloc);

        let num_bits = next;
        let num_packed = num_bits.div_ceil(32);

        cx.expand_bits(num_packed, ColumnType::B32);

        let words = cx.columns(B32_COLUMNS, ColumnType::B32);
        let h0 = cx.columns(registers, ColumnType::B32);
        let h1 = cx.columns(registers, ColumnType::B32);
        let labels = cx.columns(B16_COLUMNS, ColumnType::B16);
        let flags = cx.columns(FLAG_COLUMNS, ColumnType::Bit);
        let sel = cx.columns(registers, ColumnType::Bit);
        let selin = cx.columns(registers, ColumnType::Bit);

        let copy = copies.then(|| (cx.column(ColumnType::B16), cx.column(ColumnType::Bit)));

        Self {
            bit_width,
            registers,
            a0,
            a1,
            b0,
            b1,
            in0,
            in1,
            gamma,
            t00,
            t11,
            g,
            t01,
            t10,
            u0,
            u0_add,
            out0,
            out0_add,
            u1,
            u1_add,
            out1,
            out1_add,
            num_bits,
            num_packed,
            v_a0: words.at(0),
            v_a1: words.at(1),
            v_b0: words.at(2),
            v_b1: words.at(3),
            v_in0: words.at(4),
            v_in1: words.at(5),
            v_out0: words.at(6),
            v_out1: words.at(7),
            v_gamma: words.at(8),
            h0,
            h1,
            poly_a: labels.at(0),
            poly_b: labels.at(1),
            poly_s: labels.at(2),
            poly_o: labels.at(3),
            pos0: labels.at(4),
            pos1: labels.at(5),
            active: flags.at(0),
            btake: flags.at(1),
            seed: flags.at(2),
            emit: flags.at(3),
            bcont: flags.at(4),
            rcont: flags.at(5),
            sel,
            selin,
            copy,
            add,
        }
    }

    pub fn physical(&self, col: Col) -> Result<usize> {
        match col.index().checked_sub(32 * self.num_packed) {
            Some(plain) => Ok(self.num_packed + plain),
            None => Err(Error::Protocol {
                protocol: "poly_arith_layout",
                message: "virtual bit has no committed column of its own",
            }),
        }
    }
}
