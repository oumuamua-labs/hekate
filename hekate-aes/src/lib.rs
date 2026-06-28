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

//! AES chiplets: shared constants and
//! level-specific Air implementations.
//!
//! Shared:
//! ShiftRows, MixColumns, RotWord.
//!
//! Level-specific:
//! - aes128 (AES-128)
//! - aes256 (AES-256).

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;
use hekate_math::TowerField;
use hekate_program::constraint::builder::ConstraintSystem;

pub(crate) mod sbox_rom;

pub mod aes128;
pub mod aes256;
pub mod trace;

pub use aes128::{
    Aes128Chiplet, Aes128Columns, AesRound128Air, CpuAes128Columns, CpuAes128Unit,
    PhysAes128Columns,
};
pub use aes256::{
    Aes256Chiplet, Aes256Columns, AesRound256Air, CpuAes256Columns, CpuAes256Unit,
    PhysAes256Columns,
};

/// FIPS 197 §5.1.2:
/// ShiftRows byte permutation.
/// `SHIFT_MAP[j]` = source byte
/// index for output position j.
/// AES state is column-major 4×4:
/// byte[i] = state[i%4][i/4].
#[rustfmt::skip]
const SHIFT_MAP: [usize; 16] = [
     0,  5, 10, 15,
     4,  9, 14,  3,
     8, 13,  2,  7,
    12,  1,  6, 11,
];

/// FIPS 197 §5.1.3:
/// MixColumns coefficient matrix.
/// MC[row][col] in GF(2^8).
#[rustfmt::skip]
const MC: [[u8; 4]; 4] = [
    [2, 3, 1, 1],
    [1, 2, 3, 1],
    [1, 1, 2, 3],
    [3, 1, 1, 2],
];

/// FIPS 197 §5.2:
/// RotWord byte permutation. Maps S-box
/// index j (0..4) to the source byte
/// offset in the key's last word.
const ROT_MAP: [usize; 4] = [13, 14, 15, 12];

pub const AES_BYTE_LABELS: [&[u8]; 16] = [
    b"aes_byte_0",
    b"aes_byte_1",
    b"aes_byte_2",
    b"aes_byte_3",
    b"aes_byte_4",
    b"aes_byte_5",
    b"aes_byte_6",
    b"aes_byte_7",
    b"aes_byte_8",
    b"aes_byte_9",
    b"aes_byte_10",
    b"aes_byte_11",
    b"aes_byte_12",
    b"aes_byte_13",
    b"aes_byte_14",
    b"aes_byte_15",
];

#[rustfmt::skip]
const SBOX_IN_LABELS: [&[u8]; 16] = [
    b"aes_sbox_in_0",  b"aes_sbox_in_1",
    b"aes_sbox_in_2",  b"aes_sbox_in_3",
    b"aes_sbox_in_4",  b"aes_sbox_in_5",
    b"aes_sbox_in_6",  b"aes_sbox_in_7",
    b"aes_sbox_in_8",  b"aes_sbox_in_9",
    b"aes_sbox_in_10", b"aes_sbox_in_11",
    b"aes_sbox_in_12", b"aes_sbox_in_13",
    b"aes_sbox_in_14", b"aes_sbox_in_15",
];

#[rustfmt::skip]
const SBOX_OUT_LABELS: [&[u8]; 16] = [
    b"aes_sbox_out_0",  b"aes_sbox_out_1",
    b"aes_sbox_out_2",  b"aes_sbox_out_3",
    b"aes_sbox_out_4",  b"aes_sbox_out_5",
    b"aes_sbox_out_6",  b"aes_sbox_out_7",
    b"aes_sbox_out_8",  b"aes_sbox_out_9",
    b"aes_sbox_out_10", b"aes_sbox_out_11",
    b"aes_sbox_out_12", b"aes_sbox_out_13",
    b"aes_sbox_out_14", b"aes_sbox_out_15",
];

/// FIPS 197 §5.1.2–5.1.4:
/// SubBytes + ShiftRows + MixColumns + AddRoundKey (full rounds),
/// SubBytes + ShiftRows + AddRoundKey (final round).
/// Shared across AES-128 and AES-256, the round
/// function is identical for all key sizes.
pub(crate) fn build_round_constraints<F: TowerField>(
    cs: &ConstraintSystem<F>,
    state_in: usize,
    sbox_out: usize,
    round_key: usize,
    s_round_col: usize,
    s_final_col: usize,
) {
    let s_round = cs.col(s_round_col);
    let s_final = cs.col(s_final_col);
    let two = cs.constant(F::from(2u8));
    let three = cs.constant(F::from(3u8));

    // Full rounds:
    // next.state = MixCol(ShiftRows(sbox_out)) + round_key
    for j in 0..16usize {
        let aes_col = j / 4;
        let aes_row = j % 4;

        let mut mc_terms = Vec::with_capacity(4);
        for k in 0..4 {
            let src = cs.col(sbox_out + SHIFT_MAP[aes_col * 4 + k]);
            mc_terms.push(match MC[aes_row][k] {
                1 => src,
                2 => two * src,
                3 => three * src,
                _ => unreachable!(),
            });
        }

        let body = cs.next(state_in + j) + cs.col(round_key + j) + cs.sum(&mc_terms);
        cs.assert_zero_when(s_round, body);
    }

    // Final round:
    // next.state = ShiftRows(sbox_out) + round_key (no MixColumns)
    for (j, &src_byte) in SHIFT_MAP.iter().enumerate() {
        let shifted = cs.col(sbox_out + src_byte);
        let body = cs.next(state_in + j) + cs.col(round_key + j) + shifted;

        cs.assert_zero_when(s_final, body);
    }
}

/// FIPS 197 S-box inversion witness:
/// SubWord(input) = sub via explicit
/// inverse bit decomposition.
/// 52 constraints per call (13 per byte).
pub(crate) fn build_sbox_inversion_constraints<F: TowerField>(
    cs: &ConstraintSystem<F>,
    input_cols: [usize; 4],
    sub_col: usize,
    inv_bits_col: usize,
    z_col: usize,
    gate_col: usize,
) {
    let gate = cs.col(gate_col);
    let one = cs.one();
    let affine_const = cs.constant(F::from(0x63u8));

    for (j, &in_col) in input_cols.iter().enumerate() {
        let input = cs.col(in_col);
        let sub = cs.col(sub_col + j);
        let z = cs.col(z_col + j);

        cs.assert_boolean(z);

        let bits: [_; 8] = core::array::from_fn(|k| {
            let b = cs.col(inv_bits_col + j * 8 + k);
            cs.assert_boolean(b);

            b
        });

        let inv_terms: Vec<_> = (0..8)
            .map(|k| cs.scale(F::from(1u8 << k), bits[k]))
            .collect();
        let inv_sum = cs.sum(&inv_terms);

        cs.assert_zero_when(gate, input * inv_sum + z + one);

        cs.constrain(z * input);
        cs.constrain(z * inv_sum);

        let affine_terms: Vec<_> = (0..8)
            .map(|k| cs.scale(F::from(sbox_rom::AFFINE_COLS[k]), bits[k]))
            .collect();
        let affine_sum = cs.sum(&affine_terms);

        cs.assert_zero_when(gate, sub + affine_const + affine_sum);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shift_map_is_permutation() {
        let mut seen = [false; 16];
        for &s in &SHIFT_MAP {
            assert!(!seen[s]);
            seen[s] = true;
        }
    }

    #[test]
    fn shift_map_row0_identity() {
        // FIPS 197:
        // row 0 is not shifted.
        assert_eq!(SHIFT_MAP[0], 0);
        assert_eq!(SHIFT_MAP[4], 4);
        assert_eq!(SHIFT_MAP[8], 8);
        assert_eq!(SHIFT_MAP[12], 12);
    }
}
