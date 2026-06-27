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
//! SBOX, ShiftRows, MixColumns, RotWord.
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

/// FIPS 197 Table 4:
/// AES S-box.
/// Maps each input byte
/// to its SubBytes output.
///
/// S(x) = AffineTransform(x^{-1}) in GF(2^8)
/// with irreducible polynomial x^8+x^4+x^3+x+1.
#[rustfmt::skip]
pub const SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
    0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
    0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc,
    0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a,
    0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
    0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b,
    0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85,
    0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
    0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17,
    0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88,
    0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
    0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9,
    0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6,
    0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
    0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94,
    0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68,
    0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

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
