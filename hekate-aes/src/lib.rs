// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! AES-128 and AES-256 chiplets for Hekate, the Rust zero-knowledge proof engine.
//!
//! [`Aes128Chiplet`] and [`Aes256Chiplet`] pair a round table,
//! [`AesRound128Air`] or [`AesRound256Air`], that proves the FIPS 197 round
//! function with an S-box ROM for the GF(2^8) inversion. [`trace`] expands
//! keys and builds the chiplet traces in constant time.
//!
//! - [AES-128 and AES-256 Encryption][aes]: what the
//!   proof states and what stays outside it
//!
//! [aes]: https://oumuamua.dev/primitives/encryption/aes

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;
use hekate_math::TowerField;
use hekate_program::FixedShape;
use hekate_program::constraint::builder::ConstraintSystem;

pub(crate) mod sbox_rom;

pub mod aes128;
pub mod aes256;
pub mod trace;

pub use aes128::{
    Aes128Chiplet, Aes128Columns, AesRound128Air, CpuAes128Columns, PhysAes128Columns,
};
pub use aes256::{
    Aes256Chiplet, Aes256Columns, AesRound256Air, CpuAes256Columns, PhysAes256Columns,
};
pub use sbox_rom::{PhysSboxRomColumns, SboxRomColumns};

/// Host emit schedule:
/// `SELECTOR` fires on each block's input row
/// at offset 0 and output row at offset 1.
pub fn host_selector_shape<F: TowerField>(stride: usize, count: usize) -> FixedShape<F> {
    assert!(stride >= 2);

    FixedShape::Cadence {
        stride,
        count,
        origin: 0,
        values: (0..stride)
            .map(|off| if off <= 1 { F::ONE } else { F::ZERO })
            .collect(),
    }
}

/// Host key emit schedule:
/// `KEY_SELECTOR` fires only on each
/// block's input row at offset 0.
pub fn host_key_selector_shape<F: TowerField>(stride: usize, count: usize) -> FixedShape<F> {
    assert!(stride >= 2);

    FixedShape::Cadence {
        stride,
        count,
        origin: 0,
        values: (0..stride)
            .map(|off| if off == 0 { F::ONE } else { F::ZERO })
            .collect(),
    }
}

/// FIPS 197 §5.1.2:
/// ShiftRows byte permutation. `SHIFT_MAP[j]` = source byte
/// index for output position j. AES state is column-major 4×4:
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
