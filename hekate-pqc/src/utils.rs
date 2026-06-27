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

//! Shared witness helpers for PQC chiplets.

use hekate_core::errors::Result;
use hekate_core::trace::TraceBuilder;
use hekate_math::{Bit, Block32};

/// Pack `n` bits of `v` into `buf` at virtual
/// column offset `col_start`, LSB-first.
#[inline]
pub fn pack_bits(buf: &mut [u32], col_start: usize, v: u64, n: usize) {
    for k in 0..n {
        if (v >> k) & 1 == 1 {
            let virt_col = col_start + k;
            buf[virt_col / 32] |= 1u32 << (virt_col % 32);
        }
    }
}

/// Set a single virtual bit position in `buf`.
#[inline]
pub fn pack_one(buf: &mut [u32], virt_col: usize, val: bool) {
    if val {
        buf[virt_col / 32] |= 1u32 << (virt_col % 32);
    }
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

/// Fill the result and borrow chain of `a - b`
/// into `bits`. Result bits land at
/// `result_start..result_start+width`;
/// borrow at bit `k` lands at
/// `borrow_start + k + 1`.
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

/// Write `n` LSBs of `value` to
/// consecutive Bit columns at `col`.
pub fn write_bits(
    tb: &mut TraceBuilder,
    col: usize,
    row: usize,
    value: u32,
    n: usize,
) -> Result<()> {
    for k in 0..n {
        let bit = ((value >> k) & 1) as u8;
        tb.set_bit(col + k, row, Bit::from(bit))?;
    }

    Ok(())
}

/// Fill carry chain witness
/// for a + b at `n` bits.
pub fn fill_add_carry_witness(
    tb: &mut TraceBuilder,
    carry_col: usize,
    row: usize,
    a: u32,
    b: u32,
    n: usize,
) -> Result<()> {
    let mut carry = 0u32;
    for k in 0..n {
        let a_bit = (a >> k) & 1;
        let b_bit = (b >> k) & 1;
        let sum = a_bit + b_bit + carry;

        carry = sum >> 1;

        tb.set_bit(carry_col + k + 1, row, Bit::from(carry as u8))?;
    }

    Ok(())
}

/// Borrow chain for value < bound.
/// Computes (bound-1) - value.
pub fn fill_range_check_witness(
    tb: &mut TraceBuilder,
    result_col: usize,
    borrow_col: usize,
    row: usize,
    value: u32,
    bound: u32,
    n: usize,
) -> Result<()> {
    let minuend = bound - 1;
    let mut borrow = 0u32;

    for k in 0..n {
        let m_bit = (minuend >> k) & 1;
        let v_bit = (value >> k) & 1;

        let diff = (m_bit as i32) - (v_bit as i32) - (borrow as i32);
        let result_bit = (diff & 1) as u32;
        let new_borrow = if diff < 0 { 1u32 } else { 0u32 };

        tb.set_bit(result_col + k, row, Bit::from(result_bit as u8))?;
        tb.set_bit(borrow_col + k + 1, row, Bit::from(new_borrow as u8))?;

        borrow = new_borrow;
    }

    Ok(())
}
