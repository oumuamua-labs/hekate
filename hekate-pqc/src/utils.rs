// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Shared witness helpers for PQC chiplets.

use hekate_core::config::Config;
use hekate_core::errors::{Error, Result};
use hekate_core::trace::{ColumnType, TraceBuilder};
use hekate_math::{Bit, Block16, Block32, Block64};
use hekate_program::circuit::Col;
use subtle::{Choice, ConditionallySelectable};

use crate::gadgets::{ModAddCols, MulModCols};

pub(crate) struct Writer<'w, P: Fn(Col) -> Result<usize>> {
    pub(crate) tb: &'w mut TraceBuilder,
    pub(crate) physical: P,
    pub(crate) row: usize,
}

impl<P: Fn(Col) -> Result<usize>> Writer<'_, P> {
    pub(crate) fn word(&mut self, col: Col, value: u32) -> Result<()> {
        self.tb
            .set_b32((self.physical)(col)?, self.row, Block32::from(value))
    }

    pub(crate) fn lane(&mut self, col: Col, value: u64) -> Result<()> {
        self.tb
            .set_b64((self.physical)(col)?, self.row, Block64(value))
    }

    pub(crate) fn label(&mut self, col: Col, value: u16) -> Result<()> {
        self.tb
            .set_b16((self.physical)(col)?, self.row, Block16(value))
    }

    pub(crate) fn flag(&mut self, col: Col, on: bool) -> Result<()> {
        self.tb
            .set_bit((self.physical)(col)?, self.row, Bit::from(on as u8))
    }
}

/// Writes `values`, one per row, into committed column `at` of type `ty`.
pub(crate) fn write_column(
    tb: &mut TraceBuilder,
    ty: ColumnType,
    at: usize,
    values: impl IntoIterator<Item = u64>,
) -> Result<()> {
    for (r, value) in values.into_iter().enumerate().filter(|&(_, v)| v != 0) {
        match ty {
            ColumnType::Bit => tb.set_bit(at, r, Bit::from(value as u8))?,
            ColumnType::B16 => tb.set_b16(at, r, Block16(value as u16))?,
            ColumnType::B32 => tb.set_b32(at, r, Block32::from(value as u32))?,
            ColumnType::B64 => tb.set_b64(at, r, Block64(value))?,
            ColumnType::B8 | ColumnType::B128 => {
                return Err(Error::Protocol {
                    protocol: "pqc_trace",
                    message: "pinned column has no writer for its type",
                });
            }
        }
    }

    Ok(())
}

/// Pack `n` bits of `v` into `buf` at virtual
/// column offset `col_start`, LSB-first.
#[inline]
pub fn pack_bits(buf: &mut [u32], col_start: usize, v: u64, n: usize) {
    for k in 0..n {
        let virt_col = col_start + k;
        buf[virt_col / 32] |= (((v >> k) & 1) as u32) << (virt_col % 32);
    }
}

/// Set a single virtual bit position in `buf`.
#[inline]
pub fn pack_one(buf: &mut [u32], virt_col: usize, val: bool) {
    buf[virt_col / 32] |= (val as u32) << (virt_col % 32);
}

/// Flush the per-row packed buffer into
/// the first `bits.len()` packed B32
/// columns of the trace builder.
pub fn flush_bit_buffer(bits: &[u32], tb: &mut TraceBuilder, row: usize) -> Result<()> {
    for (col, &word) in bits.iter().enumerate() {
        tb.set_b32(col, row, Block32::from(word))?;
    }

    Ok(())
}

/// Fill the carry chain of `a + b` into `bits`
/// at virtual offset `carry_start`. Carry-out
/// at bit position `k` lands at `carry_start + k + 1`.
pub fn fill_add_carry_packed(
    bits: &mut [u32],
    carry_start: usize,
    carry_width: usize,
    a_val: u64,
    b_val: u64,
) {
    let mut carry = false;
    for k in 0..carry_width - 1 {
        let a_bit = ((a_val >> k) & 1) == 1;
        let b_bit = ((b_val >> k) & 1) == 1;

        let new_carry = (a_bit & b_bit) ^ (carry & (a_bit ^ b_bit));
        pack_one(bits, carry_start + k + 1, new_carry);

        carry = new_carry;
    }
}

/// Fill the result and borrow chain of `a - b` into `bits`.
/// Result bits land at `result_start..result_start+width`;
/// borrow at bit `k` lands at `borrow_start + k + 1`.
pub fn fill_sub_borrow_packed(
    bits: &mut [u32],
    result_start: usize,
    borrow_start: usize,
    width: usize,
    a_val: u64,
    b_val: u64,
) {
    let mut borrow = false;
    for k in 0..width {
        let a_bit = ((a_val >> k) & 1) == 1;
        let b_bit = ((b_val >> k) & 1) == 1;

        let result = a_bit ^ b_bit ^ borrow;
        let new_borrow = (!a_bit & b_bit) | ((!a_bit ^ b_bit) & borrow);

        pack_one(bits, result_start + k, result);
        pack_one(bits, borrow_start + k + 1, new_borrow);

        borrow = new_borrow;
    }
}

pub(crate) fn fill_mod_add(
    bits: &mut [u32],
    q: u32,
    bw: usize,
    (a, b, result): (u32, u32, u32),
    flag: bool,
    cols: &ModAddCols,
) {
    let flag_q = u64::conditional_select(&0, &(q as u64), Choice::from(flag as u8));

    pack_bits(bits, cols.lhs_result, a as u64 + b as u64, bw);
    fill_add_carry_packed(bits, cols.lhs_carry, bw + 1, a as u64, b as u64);

    pack_bits(bits, cols.rhs_result, result as u64 + flag_q, bw);
    fill_add_carry_packed(bits, cols.rhs_carry, bw + 1, result as u64, flag_q);

    pack_one(bits, cols.flag, flag);

    fill_sub_borrow_packed(
        bits,
        cols.range_result,
        cols.range_borrow,
        bw,
        (q - 1) as u64,
        result as u64,
    );
}

pub(crate) fn fill_mul_mod(
    bits: &mut [u32],
    cols: &MulModCols,
    q: u32,
    (x, y): (u32, u32),
    (quot, rem): (u32, u32),
) {
    let bw = cols.rem.1;

    pack_bits(bits, cols.product.0, x as u64 * y as u64, cols.product.1);
    pack_bits(bits, cols.quot.0, quot as u64, bw);
    pack_bits(bits, cols.rem.0, rem as u64, bw);

    let partial = |step: usize| {
        let hit = Choice::from(((y >> step) & 1) as u8);

        u64::conditional_select(&0, &((x as u64) << step), hit)
    };

    let pp0 = partial(0);

    pack_bits(bits, cols.pp0.0, pp0, cols.pp0.1);

    let mut acc = pp0;
    for (j, &(start, width)) in cols.sums.iter().enumerate() {
        let pp = partial(j + 1);
        let (carry_start, carry_width) = cols.carries[j];

        pack_bits(bits, start, acc + pp, width);
        fill_add_carry_packed(bits, carry_start, carry_width, acc, pp);

        acc += pp;
    }

    let last = cols.carries.len();
    let (carry_start, carry_width) = cols.carries[last - 1];

    fill_add_carry_packed(bits, carry_start, carry_width, acc, partial(last));

    let quot_x_q = quot as u64 * q as u64;

    pack_bits(bits, cols.quot_x_q.0, quot_x_q, cols.quot_x_q.1);

    fill_mul_const(bits, q, quot, &cols.red_results, &cols.red_carries);
    fill_add_carry_packed(
        bits,
        cols.red_add_carry.0,
        cols.red_add_carry.1,
        quot_x_q,
        rem as u64,
    );
    fill_sub_borrow_packed(
        bits,
        cols.red_range_result.0,
        cols.red_range_borrow.0,
        cols.red_range_result.1,
        (q - 1) as u64,
        rem as u64,
    );
}

pub(crate) fn fill_mul_const(
    bits: &mut [u32],
    constant: u32,
    operand: u32,
    results: &[(usize, usize)],
    carries: &[(usize, usize)],
) {
    let mut terms = (0..32)
        .filter(|&i| (constant >> i) & 1 == 1)
        .map(|i| (operand as u64) << i);

    let Some(mut acc) = terms.next() else {
        return;
    };

    let partials = (constant.count_ones() as usize).saturating_sub(2);

    for (j, term) in terms.enumerate() {
        fill_add_carry_packed(bits, carries[j].0, carries[j].1, acc, term);

        acc += term;

        if j < partials {
            pack_bits(bits, results[j].0, acc, results[j].1);
        }
    }
}

pub(crate) fn gcd(a: usize, b: usize) -> usize {
    match b {
        0 => a,
        _ => gcd(b, a % b),
    }
}

pub(crate) fn height(rows: usize) -> usize {
    rows.next_power_of_two()
        .max(Config::prod().min_table_rows())
}

pub(crate) fn le_words(bytes: &[u8]) -> impl Iterator<Item = u32> + '_ {
    bytes.chunks(4).map(|chunk| {
        let mut word = [0u8; 4];
        word[..chunk.len()].copy_from_slice(chunk);

        u32::from_le_bytes(word)
    })
}

pub(crate) fn le_lanes(bytes: &[u8]) -> impl Iterator<Item = u64> + '_ {
    bytes.chunks(8).map(|chunk| {
        let mut lane = [0u8; 8];
        lane[..chunk.len()].copy_from_slice(chunk);

        u64::from_le_bytes(lane)
    })
}
