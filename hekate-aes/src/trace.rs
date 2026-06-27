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

//! AES trace generation (128 and 256).
//!
//! xtime/mix_column use
//! FIPS 197 GF(2^8) arithmetic
//! with polynomial 0x11B.

use alloc::vec::Vec;
use hekate_core::errors::Error;
use hekate_core::trace::{ColumnTrace, TraceBuilder};
use hekate_math::{Bit, Block8, Block32, TowerField};

use super::aes128::PhysAes128Columns as P128;
use super::aes256::PhysAes256Columns as P256;
use super::sbox_rom::gf256_inv;
use super::{ROT_MAP, SBOX, SHIFT_MAP};

pub type Aes128Call = AesCall<16, 11>;
pub type Aes256Call = AesCall<32, 15>;

/// K=16,R=11 for AES-128.
/// K=32,R=15 for AES-256.
/// R = number of round keys = rows per block.
#[derive(Clone)]
pub struct AesCall<const K: usize, const R: usize> {
    pub key: [u8; K],
    pub plaintext: [u8; 16],
    pub round_keys: [[u8; 16]; R],
}

// Superset of AES-128 and AES-256
// row fields. Each level populates its own;
// the rest stays zeroed.
struct RowData<const K: usize> {
    state_in: [u8; 16],
    sbox_out: [u8; 16],
    round_key: [u8; 16],
    key_aux: [u8; 16],
    round_num: u8,
    rcon: u8,
    s_round: bool,
    s_final: bool,
    s_in_out: bool,
    s_input: bool,
    s_even: bool,
    k0: [u8; K],
    ks_input: [u8; 4],
    ks_sub: [u8; 4],
    ks_inv: [u8; 4],
    ks_z: [bool; 4],
    k0_sub: [u8; 4],
    k0_inv: [u8; 4],
    k0_z: [bool; 4],
    request_idx_link: u32,
    request_idx_key: u32,
}

pub fn generate_aes_trace<const K: usize, const R: usize>(
    calls: &[AesCall<K, R>],
    request_idx_triples: Option<&[(u32, u32, u32)]>,
    num_rows: usize,
) -> Result<ColumnTrace, Error> {
    if !num_rows.is_power_of_two() {
        return Err(Error::Protocol {
            protocol: "aes_trace",
            message: "trace size must be power of 2",
        });
    }

    let needed = calls.len() * R;
    if needed > num_rows {
        return Err(Error::Protocol {
            protocol: "aes_trace",
            message: "too many calls for allocated rows",
        });
    }

    let default_triples: Vec<(u32, u32, u32)> = match request_idx_triples {
        Some(_) => Vec::new(),
        None => (0..calls.len() as u32)
            .map(|k| (2 * k, 2 * k + 1, 2 * k))
            .collect(),
    };

    let triples: &[(u32, u32, u32)] = request_idx_triples.unwrap_or(&default_triples);

    if triples.len() != calls.len() {
        return Err(Error::Protocol {
            protocol: "aes_trace",
            message: "request_idx_triples length must match calls length",
        });
    }

    if P128::P_STATE_IN != P256::P_STATE_IN
        || P128::P_SBOX_OUT != P256::P_SBOX_OUT
        || P128::P_ROUND_KEY != P256::P_ROUND_KEY
    {
        return Err(Error::Protocol {
            protocol: "aes_trace",
            message: "AES-128/256 shared column offsets diverged",
        });
    }

    let num_full_rounds = R - 2;

    let mut rows = Vec::with_capacity(needed);

    for (k, call) in calls.iter().enumerate() {
        let (link_in_idx, link_out_idx, key_idx) = triples[k];

        let mut round_num: u8 = 1;
        let mut rcon: u8 = 1;
        let mut s_even = true;
        let mut prev_rk = call.round_keys[0];
        let mut state = call.plaintext;

        add_round_key(&mut state, &call.round_keys[0]);

        // AES-128:
        // init S-box witness (K0 -> K1)
        let (k0_sub, k0_inv, k0_z) = if K == 16 {
            sbox_witness_bytes(rotword_bytes(&call.key))
        } else {
            ([0u8; 4], [0u8; 4], [false; 4])
        };

        for r in 1..=num_full_rounds {
            let state_in = state;
            let mut sbox_out_state = state;

            sub_bytes(&mut sbox_out_state);

            let sbox_out = sbox_out_state;
            let mut after_shift = shift_rows(&sbox_out_state);

            mix_columns(&mut after_shift);
            add_round_key(&mut after_shift, &call.round_keys[r]);

            state = after_shift;

            let ks_bytes = if K == 32 && !s_even {
                direct_bytes(&call.round_keys[r])
            } else {
                rotword_bytes(&call.round_keys[r])
            };

            let (ks_sub, ks_inv, ks_z) = sbox_witness_bytes(ks_bytes);

            let is_input = r == 1;

            rows.push(RowData {
                state_in,
                sbox_out,
                round_key: call.round_keys[r],
                key_aux: if K == 32 { prev_rk } else { [0u8; 16] },
                round_num,
                rcon: if K == 32 { rcon } else { 0 },
                s_round: true,
                s_final: false,
                s_in_out: is_input,
                s_input: is_input,
                s_even: K == 32 && s_even,
                k0: if is_input { call.key } else { [0u8; K] },
                ks_input: if K == 32 { ks_bytes } else { [0u8; 4] },
                ks_sub,
                ks_inv,
                ks_z,
                k0_sub: if K == 16 && is_input {
                    k0_sub
                } else {
                    [0u8; 4]
                },
                k0_inv: if K == 16 && is_input {
                    k0_inv
                } else {
                    [0u8; 4]
                },
                k0_z: if K == 16 && is_input {
                    k0_z
                } else {
                    [false; 4]
                },
                request_idx_link: if is_input { link_in_idx } else { 0 },
                request_idx_key: if is_input { key_idx } else { 0 },
            });

            if K == 32 {
                prev_rk = call.round_keys[r];

                if !s_even {
                    rcon = xtime(rcon);
                }

                s_even = !s_even;
            }

            round_num = xtime(round_num);
        }

        // Final round (no MixColumns)
        let state_in = state;
        let mut sbox_out_state = state;

        sub_bytes(&mut sbox_out_state);

        let sbox_out = sbox_out_state;
        let mut after_shift = shift_rows(&sbox_out_state);

        add_round_key(&mut after_shift, &call.round_keys[R - 1]);

        state = after_shift;

        rows.push(RowData {
            state_in,
            sbox_out,
            round_key: call.round_keys[R - 1],
            key_aux: if K == 32 { prev_rk } else { [0u8; 16] },
            round_num,
            rcon,
            s_round: false,
            s_final: true,
            s_in_out: false,
            s_input: false,
            s_even: false,
            k0: [0u8; K],
            ks_input: [0u8; 4],
            ks_sub: [0u8; 4],
            ks_inv: [0u8; 4],
            ks_z: [false; 4],
            k0_sub: [0u8; 4],
            k0_inv: [0u8; 4],
            k0_z: [false; 4],
            request_idx_link: 0,
            request_idx_key: 0,
        });

        // Output row
        rows.push(RowData {
            state_in: state,
            sbox_out: [0u8; 16],
            round_key: [0u8; 16],
            key_aux: [0u8; 16],
            round_num: 0,
            rcon: 0,
            s_round: false,
            s_final: false,
            s_in_out: true,
            s_input: false,
            s_even: false,
            k0: [0u8; K],
            ks_input: [0u8; 4],
            ks_sub: [0u8; 4],
            ks_inv: [0u8; 4],
            ks_z: [false; 4],
            k0_sub: [0u8; 4],
            k0_inv: [0u8; 4],
            k0_z: [false; 4],
            request_idx_link: link_out_idx,
            request_idx_key: 0,
        });
    }

    let layout = if K == 16 {
        P128::build_layout()
    } else {
        P256::build_layout()
    };

    let num_vars = num_rows.trailing_zeros() as usize;

    let mut tb = TraceBuilder::new(&layout, num_vars)?;

    for (i, row) in rows.iter().enumerate() {
        tb.set_b8_array(P128::P_STATE_IN, i, &row.state_in.map(Block8))?;
        tb.set_b8_array(P128::P_SBOX_OUT, i, &row.sbox_out.map(Block8))?;
        tb.set_b8_array(P128::P_ROUND_KEY, i, &row.round_key.map(Block8))?;

        if K == 16 {
            write_128_row(&mut tb, i, row)?;
        } else {
            write_256_row(&mut tb, i, row)?;
        }
    }

    Ok(tb.build())
}

fn write_128_row<const K: usize>(
    tb: &mut TraceBuilder,
    i: usize,
    row: &RowData<K>,
) -> Result<(), Error> {
    if K != 16 {
        return Err(Error::Protocol {
            protocol: "aes_trace",
            message: "write_128_row requires K=16",
        });
    }

    if row.round_num != 0 {
        tb.set_b8(P128::P_ROUND_NUM, i, Block8(row.round_num))?;
    }

    if row.s_round {
        tb.set_bit(P128::P_S_ROUND, i, Bit::ONE)?;
    }

    if row.s_final {
        tb.set_bit(P128::P_S_FINAL, i, Bit::ONE)?;
    }

    if row.s_in_out {
        tb.set_bit(P128::P_S_IN_OUT, i, Bit::ONE)?;
    }

    if row.s_round || row.s_final {
        tb.set_bit(P128::P_S_ACTIVE, i, Bit::ONE)?;
    }

    if row.s_input {
        tb.set_bit(P128::P_S_INPUT, i, Bit::ONE)?;
        tb.set_b8_array(P128::P_K0, i, &row.k0.map(Block8))?;
        tb.set_b8_array(P128::P_K0_SUB, i, &row.k0_sub.map(Block8))?;
        tb.set_b8_array(P128::P_K0_INV, i, &row.k0_inv.map(Block8))?;

        for j in 0..4 {
            if row.k0_z[j] {
                tb.set_bit(P128::P_K0_Z + j, i, Bit::ONE)?;
            }
        }
    }

    if row.s_round {
        tb.set_b8_array(P128::P_KS_SUB, i, &row.ks_sub.map(Block8))?;
        tb.set_b8_array(P128::P_KS_INV, i, &row.ks_inv.map(Block8))?;

        for j in 0..4 {
            if row.ks_z[j] {
                tb.set_bit(P128::P_KS_Z + j, i, Bit::ONE)?;
            }
        }
    }

    if row.request_idx_link != 0 {
        tb.set_b32(
            P128::P_REQUEST_IDX_LINK,
            i,
            Block32::from(row.request_idx_link),
        )?;
    }

    if row.request_idx_key != 0 {
        tb.set_b32(
            P128::P_REQUEST_IDX_KEY,
            i,
            Block32::from(row.request_idx_key),
        )?;
    }

    Ok(())
}

fn write_256_row<const K: usize>(
    tb: &mut TraceBuilder,
    i: usize,
    row: &RowData<K>,
) -> Result<(), Error> {
    if K != 32 {
        return Err(Error::Protocol {
            protocol: "aes_trace",
            message: "write_256_row requires K=32",
        });
    }

    tb.set_b8_array(P256::P_KEY_AUX, i, &row.key_aux.map(Block8))?;

    if row.round_num != 0 {
        tb.set_b8(P256::P_ROUND_NUM, i, Block8(row.round_num))?;
    }

    if row.rcon != 0 {
        tb.set_b8(P256::P_RCON, i, Block8(row.rcon))?;
    }

    if row.s_round {
        tb.set_bit(P256::P_S_ROUND, i, Bit::ONE)?;
    }

    if row.s_final {
        tb.set_bit(P256::P_S_FINAL, i, Bit::ONE)?;
    }

    if row.s_in_out {
        tb.set_bit(P256::P_S_IN_OUT, i, Bit::ONE)?;
    }

    if row.s_round || row.s_final {
        tb.set_bit(P256::P_S_ACTIVE, i, Bit::ONE)?;
    }

    if row.s_even {
        tb.set_bit(P256::P_S_EVEN, i, Bit::ONE)?;
    }

    if row.s_input {
        tb.set_bit(P256::P_S_INPUT, i, Bit::ONE)?;
        tb.set_b8_array(P256::P_K0, i, &row.k0.map(Block8))?;
    }

    if row.s_round {
        tb.set_b8_array(P256::P_KS_INPUT, i, &row.ks_input.map(Block8))?;
        tb.set_b8_array(P256::P_KS_SUB, i, &row.ks_sub.map(Block8))?;
        tb.set_b8_array(P256::P_KS_INV, i, &row.ks_inv.map(Block8))?;

        for j in 0..4 {
            if row.ks_z[j] {
                tb.set_bit(P256::P_KS_Z + j, i, Bit::ONE)?;
            }
        }
    }

    if row.request_idx_link != 0 {
        tb.set_b32(
            P256::P_REQUEST_IDX_LINK,
            i,
            Block32::from(row.request_idx_link),
        )?;
    }

    if row.request_idx_key != 0 {
        tb.set_b32(
            P256::P_REQUEST_IDX_KEY,
            i,
            Block32::from(row.request_idx_key),
        )?;
    }

    Ok(())
}

/// AES-128 key expansion (FIPS 197 §5.2).
/// Expands a 16-byte key into 11 round keys.
pub fn expand_key(key: &[u8; 16]) -> [[u8; 16]; 11] {
    const RCON: [u8; 10] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36];

    let mut rk = [[0u8; 16]; 11];
    rk[0] = *key;

    for r in 1..11 {
        let prev = rk[r - 1];

        // RotWord + SubWord + Rcon on last word
        let rot = [prev[13], prev[14], prev[15], prev[12]];
        let sub = [
            SBOX[rot[0] as usize],
            SBOX[rot[1] as usize],
            SBOX[rot[2] as usize],
            SBOX[rot[3] as usize],
        ];

        rk[r][0] = prev[0] ^ sub[0] ^ RCON[r - 1];
        rk[r][1] = prev[1] ^ sub[1];
        rk[r][2] = prev[2] ^ sub[2];
        rk[r][3] = prev[3] ^ sub[3];

        for word in 1..4 {
            let base = word * 4;
            for b in 0..4 {
                rk[r][base + b] = prev[base + b] ^ rk[r][base + b - 4];
            }
        }
    }

    rk
}

/// AES-256 key expansion (FIPS 197 §5.2, Nk=8).
/// Expands a 32-byte key into 15 round keys.
pub fn expand_key_256(key: &[u8; 32]) -> [[u8; 16]; 15] {
    const RCON: [u8; 7] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40];

    let mut w = [0u8; 240];
    w[..32].copy_from_slice(key);

    for i in 8..60 {
        let prev = [
            w[(i - 1) * 4],
            w[(i - 1) * 4 + 1],
            w[(i - 1) * 4 + 2],
            w[(i - 1) * 4 + 3],
        ];
        let back = [
            w[(i - 8) * 4],
            w[(i - 8) * 4 + 1],
            w[(i - 8) * 4 + 2],
            w[(i - 8) * 4 + 3],
        ];

        let derived = if i % 8 == 0 {
            // SubWord(RotWord(prev)) + Rcon
            let rot = [prev[1], prev[2], prev[3], prev[0]];
            [
                back[0] ^ SBOX[rot[0] as usize] ^ RCON[i / 8 - 1],
                back[1] ^ SBOX[rot[1] as usize],
                back[2] ^ SBOX[rot[2] as usize],
                back[3] ^ SBOX[rot[3] as usize],
            ]
        } else if i % 8 == 4 {
            // SubWord(prev), no RotWord, no Rcon
            [
                back[0] ^ SBOX[prev[0] as usize],
                back[1] ^ SBOX[prev[1] as usize],
                back[2] ^ SBOX[prev[2] as usize],
                back[3] ^ SBOX[prev[3] as usize],
            ]
        } else {
            [
                back[0] ^ prev[0],
                back[1] ^ prev[1],
                back[2] ^ prev[2],
                back[3] ^ prev[3],
            ]
        };

        w[i * 4..i * 4 + 4].copy_from_slice(&derived);
    }

    let mut rk = [[0u8; 16]; 15];
    for r in 0..15 {
        rk[r].copy_from_slice(&w[r * 16..r * 16 + 16]);
    }

    rk
}

pub fn aes256_encrypt_block(round_keys: &[[u8; 16]; 15], block: &[u8; 16]) -> [u8; 16] {
    let mut state = *block;
    add_round_key(&mut state, &round_keys[0]);

    for rk in &round_keys[1..14] {
        sub_bytes(&mut state);
        state = shift_rows(&state);
        mix_columns(&mut state);
        add_round_key(&mut state, rk);
    }

    sub_bytes(&mut state);
    state = shift_rows(&state);
    add_round_key(&mut state, &round_keys[14]);

    state
}

/// xtime:
/// multiply by 0x02 in GF(2^8)
/// with irreducible polynomial 0x11B.
fn xtime(b: u8) -> u8 {
    let shifted = (b as u16) << 1;
    (shifted ^ if b & 0x80 != 0 { 0x1B } else { 0 }) as u8
}

fn sub_bytes(state: &mut [u8; 16]) {
    for b in state.iter_mut() {
        *b = SBOX[*b as usize];
    }
}

fn shift_rows(state: &[u8; 16]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for (j, &src) in SHIFT_MAP.iter().enumerate() {
        out[j] = state[src];
    }

    out
}

fn mix_column(col: [u8; 4]) -> [u8; 4] {
    let [a, b, c, d] = col;

    [
        xtime(a) ^ xtime(b) ^ b ^ c ^ d,
        a ^ xtime(b) ^ xtime(c) ^ c ^ d,
        a ^ b ^ xtime(c) ^ xtime(d) ^ d,
        xtime(a) ^ a ^ b ^ c ^ xtime(d),
    ]
}

fn mix_columns(state: &mut [u8; 16]) {
    for col in 0..4 {
        let base = col * 4;
        let mixed = mix_column([
            state[base],
            state[base + 1],
            state[base + 2],
            state[base + 3],
        ]);

        state[base..base + 4].copy_from_slice(&mixed);
    }
}

fn add_round_key(state: &mut [u8; 16], rk: &[u8; 16]) {
    for (s, &k) in state.iter_mut().zip(rk) {
        *s ^= k;
    }
}

fn rotword_bytes(rk: &[u8]) -> [u8; 4] {
    [
        rk[ROT_MAP[0]],
        rk[ROT_MAP[1]],
        rk[ROT_MAP[2]],
        rk[ROT_MAP[3]],
    ]
}

fn direct_bytes(rk: &[u8]) -> [u8; 4] {
    [rk[12], rk[13], rk[14], rk[15]]
}

fn sbox_witness_bytes(input: [u8; 4]) -> ([u8; 4], [u8; 4], [bool; 4]) {
    let mut sub = [0u8; 4];
    let mut inv = [0u8; 4];
    let mut z = [false; 4];

    for j in 0..4 {
        inv[j] = gf256_inv(input[j]);
        sub[j] = SBOX[input[j] as usize];
        z[j] = input[j] == 0;
    }

    (sub, inv, z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hekate_core::trace::Trace;

    /// FIPS 197 Appendix B test vector.
    #[rustfmt::skip]
    const FIPS_KEY: [u8; 16] = [
        0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
        0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
    ];

    #[rustfmt::skip]
    const FIPS_PLAIN: [u8; 16] = [
        0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d,
        0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34,
    ];

    #[rustfmt::skip]
    const FIPS_CIPHER: [u8; 16] = [
        0x39, 0x25, 0x84, 0x1d, 0x02, 0xdc, 0x09, 0xfb,
        0xdc, 0x11, 0x85, 0x97, 0x19, 0x6a, 0x0b, 0x32,
    ];

    fn fips_call() -> Aes128Call {
        let rk = expand_key(&FIPS_KEY);
        AesCall {
            key: FIPS_KEY,
            plaintext: FIPS_PLAIN,
            round_keys: rk,
        }
    }

    #[test]
    fn key_expansion_fips197() {
        let rk = expand_key(&FIPS_KEY);
        assert_eq!(rk[0], FIPS_KEY);

        // FIPS 197 Appendix A.1, last round key
        assert_eq!(
            rk[10],
            [
                0xd0, 0x14, 0xf9, 0xa8, 0xc9, 0xee, 0x25, 0x89, 0xe1, 0x3f, 0x0c, 0xc8, 0xb6, 0x63,
                0x0c, 0xa6
            ],
        );
    }

    #[test]
    fn xtime_known_values() {
        assert_eq!(xtime(0x57), 0xAE);
        assert_eq!(xtime(0xAE), 0x47);
        assert_eq!(xtime(0x47), 0x8E);
        assert_eq!(xtime(0x8E), 0x07);
    }

    #[test]
    fn single_block_ciphertext() {
        let call = fips_call();
        let trace = generate_aes_trace(&[call], None, 16).unwrap();

        let state_in_cols: Vec<_> = (0..16)
            .map(|c| trace.columns[P128::P_STATE_IN + c].as_b8_slice().unwrap())
            .collect();

        for (j, expected) in FIPS_CIPHER.iter().enumerate() {
            assert_eq!(
                state_in_cols[j][10].to_tower(),
                Block8(*expected),
                "ciphertext byte {j} mismatch",
            );
        }
    }

    #[test]
    fn selector_pattern() {
        let call = fips_call();
        let trace = generate_aes_trace(&[call], None, 16).unwrap();

        let s_round = trace.columns[P128::P_S_ROUND].as_bit_slice().unwrap();
        let s_final = trace.columns[P128::P_S_FINAL].as_bit_slice().unwrap();
        let s_in_out = trace.columns[P128::P_S_IN_OUT].as_bit_slice().unwrap();

        // Rows 0-8:
        // s_round=1
        assert!(s_round.iter().take(9).all(|&s| s == Bit::ONE));

        // Row 9:
        // s_final=1
        assert_eq!(s_final[9], Bit::ONE);
        assert_eq!(s_round[9], Bit::ZERO);

        // Row 10:
        // output row, s_in_out=1
        assert_eq!(s_in_out[10], Bit::ONE);
        assert_eq!(s_round[10], Bit::ZERO);
        assert_eq!(s_final[10], Bit::ZERO);

        // Row 0:
        // s_in_out=1 (input row)
        assert_eq!(s_in_out[0], Bit::ONE);

        // Padding rows 11-15
        assert!(s_round.iter().skip(11).take(5).all(|&s| s == Bit::ZERO));
        assert!(s_final.iter().skip(11).take(5).all(|&s| s == Bit::ZERO));
        assert!(s_in_out.iter().skip(11).take(5).all(|&s| s == Bit::ZERO));
    }

    #[test]
    fn trace_overflow() {
        let call = fips_call();
        assert!(generate_aes_trace(&[call], None, 8).is_err());
    }

    #[test]
    fn two_blocks() {
        let calls = [fips_call(), fips_call()];
        let trace = generate_aes_trace(&calls, None, 32).unwrap();
        assert_eq!(trace.num_rows().unwrap(), 32);

        // Both blocks produce
        // the same ciphertext.
        let col0 = trace.columns[P128::P_STATE_IN].as_b8_slice().unwrap();
        assert_eq!(col0[10].to_tower(), col0[21].to_tower());
    }

    #[test]
    fn sbox_out_matches_sub_bytes() {
        let call = fips_call();
        let trace = generate_aes_trace(&[call], None, 16).unwrap();

        for row in 0..10 {
            for j in 0..16 {
                let inp = trace.columns[P128::P_STATE_IN + j].as_b8_slice().unwrap()[row]
                    .to_tower()
                    .0;
                let out = trace.columns[P128::P_SBOX_OUT + j].as_b8_slice().unwrap()[row]
                    .to_tower()
                    .0;

                assert_eq!(out, SBOX[inp as usize], "row {row} byte {j}");
            }
        }
    }

    #[test]
    fn key_schedule_witness_fips197() {
        let call = fips_call();
        let trace = generate_aes_trace(&[call], None, 16).unwrap();

        let k0_col: Vec<_> = (0..16)
            .map(|j| trace.columns[P128::P_K0 + j].as_b8_slice().unwrap())
            .collect();

        // K0 populated on s_input row (row 0)
        for j in 0..16 {
            assert_eq!(k0_col[j][0].to_tower().0, FIPS_KEY[j], "K0 byte {j}");
        }

        // KS_INV on round rows holds
        // gf256_inv(RotWord(ROUND_KEY)).
        let rk = expand_key(&FIPS_KEY);
        for row in 0..9 {
            for j in 0..4 {
                let rk_byte = rk[row + 1][ROT_MAP[j]];
                let expected_inv = gf256_inv(rk_byte);
                let actual_inv = trace.columns[P128::P_KS_INV + j].as_b8_slice().unwrap()[row]
                    .to_tower()
                    .0;

                assert_eq!(
                    actual_inv, expected_inv,
                    "row {row} KS_INV[{j}]: rk_byte=0x{rk_byte:02X}",
                );
            }
        }
    }
}
