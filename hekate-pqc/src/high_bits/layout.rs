// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec::Vec;
use hekate_core::errors::{Error, Result};
use hekate_core::trace::ColumnType;
use hekate_gadgets::atoms::int_arith::{
    ModAddLayout, MulConstLayout, mod_add_scratch_count, mul_const_scratch_widths,
};
use hekate_math::{HardwareField, TowerField};
use hekate_program::circuit::{Circuit, Col, ColRange};

use crate::gadgets::ModAddCols;
use crate::mldsa::MlDsaParams;

pub(crate) const W_BITS: usize = 23;

const WORD_COLUMNS: usize = 2;
const LANE_COLUMNS: usize = 2;
const LABEL_COLUMNS: usize = 7;
const FLAG_COLUMNS: usize = 4;

/// Column positions of the HighBits table.
#[derive(Clone, Debug)]
pub struct HighBitsLayout {
    pub n: usize,
    pub b: usize,
    pub r0_width: usize,

    pub w: usize,
    pub r1u: usize,
    pub r0u: usize,
    pub qxd: usize,
    pub mul_results: Vec<(usize, usize)>,
    pub mul_carries: Vec<(usize, usize)>,
    pub add_carry: usize,
    pub rng_result: usize,
    pub rng_borrow: usize,
    pub neg_result: usize,
    pub neg_borrow: usize,
    pub w_rng_result: usize,
    pub w_rng_borrow: usize,

    pub r1p: usize,
    pub inc: usize,
    pub nz: usize,
    pub dir: usize,
    pub h: usize,
    pub down: usize,
    pub w1_add: ModAddCols,
    pub w1: usize,

    pub num_bits: usize,
    pub num_packed: usize,

    pub word: Col,
    pub inv0: Col,
    pub acc: Col,
    pub lane: Col,
    pub pw: ColRange,
    pub pn: ColRange,

    pub poly: Col,
    pub pos: Col,
    pub hpoly: Col,
    pub hp: Col,
    pub hm: Col,
    pub lidx: Col,
    pub lstream: Col,

    pub active: Col,
    pub start: Col,
    pub cont: Col,
    pub emit: Col,

    pub mul: MulConstLayout,
    pub add: ModAddLayout,
}

impl HighBitsLayout {
    pub(crate) fn declare<F: TowerField + HardwareField>(
        cx: &mut Circuit<F>,
        params: &MlDsaParams,
    ) -> Self {
        let m = params.high_bits_range();
        let n = bit_len(m);
        let b = params.w1_bits();
        let r0_width = bit_len(2 * params.gamma2() - 1);

        let mul = mul_const_scratch_widths(n, 2 * params.gamma2());
        let add = mod_add_scratch_count(n);

        let mut next = 0usize;

        let mut alloc = |width: usize| {
            let start = next;
            next += width;

            start
        };

        let w = alloc(W_BITS);
        let r1u = alloc(n);
        let r0u = alloc(r0_width);
        let qxd = alloc(mul.result_width);

        let mul_results = mul
            .scratch_result_widths
            .iter()
            .map(|&width| (alloc(width), width))
            .collect();

        let mul_carries = mul
            .scratch_carry_widths
            .iter()
            .map(|&width| (alloc(width), width))
            .collect();

        let add_carry = alloc(mul.result_width + 1);
        let rng_result = alloc(r0_width);
        let rng_borrow = alloc(r0_width + 1);
        let neg_result = alloc(r0_width);
        let neg_borrow = alloc(r0_width + 1);
        let w_rng_result = alloc(W_BITS);
        let w_rng_borrow = alloc(W_BITS + 1);

        let r1p = alloc(n);
        let inc = alloc(n - 1);
        let nz = alloc(1);
        let dir = alloc(1);
        let h = alloc(1);
        let down = alloc(1);

        let w1_add = ModAddCols::alloc(&add, &mut alloc);
        let w1 = alloc(n);

        let num_bits = next;
        let num_packed = num_bits.div_ceil(32);

        cx.expand_bits(num_packed, ColumnType::B32);

        let words = cx.columns(WORD_COLUMNS, ColumnType::B32);
        let lanes = cx.columns(LANE_COLUMNS, ColumnType::B64);
        let pw = cx.columns(b, ColumnType::B64);
        let pn = cx.columns(b, ColumnType::B64);
        let labels = cx.columns(LABEL_COLUMNS, ColumnType::B16);
        let flags = cx.columns(FLAG_COLUMNS, ColumnType::Bit);

        Self {
            n,
            b,
            r0_width,
            w,
            r1u,
            r0u,
            qxd,
            mul_results,
            mul_carries,
            add_carry,
            rng_result,
            rng_borrow,
            neg_result,
            neg_borrow,
            w_rng_result,
            w_rng_borrow,
            r1p,
            inc,
            nz,
            dir,
            h,
            down,
            w1_add,
            w1,
            num_bits,
            num_packed,
            word: words.at(0),
            inv0: words.at(1),
            acc: lanes.at(0),
            lane: lanes.at(1),
            pw,
            pn,
            poly: labels.at(0),
            pos: labels.at(1),
            hpoly: labels.at(2),
            hp: labels.at(3),
            hm: labels.at(4),
            lidx: labels.at(5),
            lstream: labels.at(6),
            active: flags.at(0),
            start: flags.at(1),
            cont: flags.at(2),
            emit: flags.at(3),
            mul,
            add,
        }
    }

    /// Physical column behind `col`.
    pub fn physical(&self, col: Col) -> Result<usize> {
        match col.index().checked_sub(32 * self.num_packed) {
            Some(plain) => Ok(self.num_packed + plain),
            None => Err(Error::Protocol {
                protocol: "high_bits_layout",
                message: "virtual bit has no committed column of its own",
            }),
        }
    }
}

fn bit_len(v: u32) -> usize {
    32 - v.leading_zeros() as usize
}
