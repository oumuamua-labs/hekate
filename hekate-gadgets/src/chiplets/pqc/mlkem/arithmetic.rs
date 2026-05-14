// SPDX-License-Identifier: Apache-2.0
// This file is part of the hekate-math project.
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

use crate::chiplets::pqc::mlkem::{MLKEM_Q, N};
use crate::{shake128, shake256, KeccakCall};
use alloc::vec;
use alloc::vec::Vec;

/// Primitive 17th root of unity mod 3329.
/// ζ = 17, ζ^128 ≡ -1 (mod 3329).
const ZETA: u16 = 17;

/// Barrett reduction constant for q=3329.
/// floor(2^24 / 3329) = 5039.
const BARRETT_SHIFT: u32 = 24;
const BARRETT_MULT: u32 = 5039;

// =================================================================
// Native Mod-q Arithmetic
// =================================================================

pub fn mod_reduce(a: u32) -> u16 {
    let t = ((a as u64 * BARRETT_MULT as u64) >> BARRETT_SHIFT) as u32;
    let r = a - t * MLKEM_Q;

    if r >= MLKEM_Q {
        (r - MLKEM_Q) as u16
    } else {
        r as u16
    }
}

pub fn mod_add(a: u16, b: u16) -> u16 {
    let s = a as u32 + b as u32;
    if s >= MLKEM_Q {
        (s - MLKEM_Q) as u16
    } else {
        s as u16
    }
}

pub fn mod_sub(a: u16, b: u16) -> u16 {
    let s = a as u32 + MLKEM_Q - b as u32;
    if s >= MLKEM_Q {
        (s - MLKEM_Q) as u16
    } else {
        s as u16
    }
}

pub fn mod_mul(a: u16, b: u16) -> u16 {
    mod_reduce(a as u32 * b as u32)
}

// =================================================================
// NTT-256 over Z_3329
// =================================================================

/// Precomputed powers of ζ in bit-reversed
/// order for Cooley-Tukey NTT (FIPS 203 §4.4).
pub fn zeta_powers() -> [u16; 128] {
    let mut zetas = [0u16; 128];

    // ζ^(bit_rev(i)) for i = 0..127
    let mut power = 1u32;
    let mut table = [0u16; 256];

    table[0] = 1;

    for i in 1..256 {
        power = (power * ZETA as u32) % MLKEM_Q;
        table[i] = power as u16;
    }

    for i in 0..128 {
        let br = bit_rev_7(i as u8) as usize;
        zetas[i] = table[br];
    }

    zetas
}

fn bit_rev_7(x: u8) -> u8 {
    let mut r = 0u8;
    let mut v = x;

    for _ in 0..7 {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }

    r
}

/// Forward NTT (FIPS 203 Algorithm 9).
#[cfg(test)]
pub fn ntt_forward(f: &mut [u16; N]) {
    let zetas = zeta_powers();

    let mut k = 1usize;
    let mut len = 128;

    while len >= 2 {
        let mut start = 0;
        while start < N {
            let zeta = zetas[k];
            k += 1;

            for j in start..start + len {
                let t = mod_mul(zeta, f[j + len]);
                f[j + len] = mod_sub(f[j], t);
                f[j] = mod_add(f[j], t);
            }

            start += 2 * len;
        }

        len >>= 1;
    }
}

/// Inverse NTT (FIPS 203 Algorithm 10).
#[cfg(test)]
pub fn ntt_inverse(f: &mut [u16; N]) {
    let zetas = zeta_powers();

    let mut k = 127usize;
    let mut len = 2;

    while len <= 128 {
        let mut start = 0;
        while start < N {
            let zeta = zetas[k];

            // Final iteration underflows
            // to usize::MAX; harmless k is
            // unused after the loop exits.
            k = k.wrapping_sub(1);

            for j in start..start + len {
                let t = f[j];
                f[j] = mod_add(t, f[j + len]);
                f[j + len] = mod_mul(zeta, mod_sub(f[j + len], t));
            }

            start += 2 * len;
        }

        len <<= 1;
    }

    // Multiply by 128^{-1} mod q (FIPS 203 §4.4).
    // The NTT is 7-layer (128-point over degree-2
    // subrings), so normalization is 128^{-1},
    // not 256^{-1}. 128^{-1} mod 3329 = 3303.
    const NTT_NORM: u16 = 3303;
    for coeff in f.iter_mut() {
        *coeff = mod_mul(*coeff, NTT_NORM);
    }
}

/// Pointwise multiply two NTT-domain polys
/// (basemul, FIPS 203 Algorithm 11).
#[cfg(test)]
pub fn poly_basemul(a: &[u16; N], b: &[u16; N]) -> [u16; N] {
    let zetas = zeta_powers();
    let mut r = [0u16; N];

    for i in 0..64 {
        let z = zetas[64 + i];
        let (a0, a1) = (a[4 * i], a[4 * i + 1]);
        let (b0, b1) = (b[4 * i], b[4 * i + 1]);

        r[4 * i] = mod_add(mod_mul(a0, b0), mod_mul(mod_mul(a1, b1), z));
        r[4 * i + 1] = mod_add(mod_mul(a0, b1), mod_mul(a1, b0));

        let (a2, a3) = (a[4 * i + 2], a[4 * i + 3]);
        let (b2, b3) = (b[4 * i + 2], b[4 * i + 3]);

        let neg_z = mod_sub(0, z);

        r[4 * i + 2] = mod_add(mod_mul(a2, b2), mod_mul(mod_mul(a3, b3), neg_z));
        r[4 * i + 3] = mod_add(mod_mul(a2, b3), mod_mul(a3, b2));
    }

    r
}

pub fn poly_add(a: &[u16; N], b: &[u16; N]) -> [u16; N] {
    let mut r = [0u16; N];
    for i in 0..N {
        r[i] = mod_add(a[i], b[i]);
    }

    r
}

pub fn poly_sub(a: &[u16; N], b: &[u16; N]) -> [u16; N] {
    let mut r = [0u16; N];
    for i in 0..N {
        r[i] = mod_sub(a[i], b[i]);
    }

    r
}

// =================================================================
// Byte Encoding / Decoding (FIPS 203 §4.2)
// =================================================================

/// ByteDecode_d:
/// decode coefficients from bytes.
pub fn byte_decode(d: usize, bytes: &[u8]) -> [u16; N] {
    let mut f = [0u16; N];

    let mask = (1u32 << d) - 1;
    let mut bit_idx = 0usize;

    for coeff in f.iter_mut() {
        let mut val = 0u32;
        for b in 0..d {
            let byte_pos = (bit_idx + b) / 8;
            let bit_pos = (bit_idx + b) % 8;

            val |= (((bytes[byte_pos] >> bit_pos) & 1) as u32) << b;
        }

        *coeff = (val & mask) as u16;
        bit_idx += d;
    }

    f
}

/// ByteEncode_d:
/// encode coefficients to bytes.
pub fn byte_encode(d: usize, f: &[u16; N]) -> Vec<u8> {
    let total_bits = N * d;

    let mut bytes = vec![0u8; total_bits.div_ceil(8)];
    let mut bit_idx = 0usize;

    for &coeff in f {
        let val = coeff as u32;
        for b in 0..d {
            let byte_pos = (bit_idx + b) / 8;
            let bit_pos = (bit_idx + b) % 8;

            bytes[byte_pos] |= (((val >> b) & 1) as u8) << bit_pos;
        }

        bit_idx += d;
    }

    bytes
}

// =================================================================
// Compress / Decompress (FIPS 203 §4.2)
// =================================================================

pub fn compress_d(d: usize, x: u16) -> u16 {
    // ⌈(2^d / q) · x⌋ = ⌊(2^d · x + q/2) / q⌋
    let scaled = ((x as u64) << d) + (MLKEM_Q as u64 / 2);
    (scaled / MLKEM_Q as u64) as u16 & ((1 << d) - 1)
}

pub fn decompress_d(d: usize, y: u16) -> u16 {
    // ⌈(q / 2^d) · y⌋ = ⌊(q · y + 2^(d-1)) / 2^d⌋
    let scaled = MLKEM_Q * (y as u32) + (1 << (d - 1));
    (scaled >> d) as u16
}

pub fn compress_poly(d: usize, f: &[u16; N]) -> [u16; N] {
    let mut r = [0u16; N];
    for i in 0..N {
        r[i] = compress_d(d, f[i]);
    }

    r
}

pub fn decompress_poly(d: usize, f: &[u16; N]) -> [u16; N] {
    let mut r = [0u16; N];
    for i in 0..N {
        r[i] = decompress_d(d, f[i]);
    }

    r
}

// =================================================================
// Sampling (FIPS 203 §4.3)
// =================================================================

/// Sample NTT polynomial from XOF (SHAKE-128).
/// FIPS 203 Algorithm 6:
/// SampleNTT.
pub fn sample_ntt(
    rho: &[u8; 32],
    i: u8,
    j: u8,
    keccak_calls: &mut Vec<KeccakCall>,
    sponge_meta: &mut Vec<(bool, bool, bool)>,
) -> [u16; N] {
    let mut seed = Vec::with_capacity(34);
    seed.extend_from_slice(rho);
    seed.push(j);
    seed.push(i);

    let (xof_out, calls) = shake128(&seed, 3 * N);
    keccak_calls.extend_from_slice(&calls);

    for (k, _) in calls.iter().enumerate() {
        sponge_meta.push((k == 0, false, true));
    }

    let mut f = [0u16; N];
    let mut idx = 0;
    let mut byte_pos = 0;

    while idx < N {
        if byte_pos + 2 >= xof_out.len() {
            break;
        }

        let d1 = xof_out[byte_pos] as u16 | ((xof_out[byte_pos + 1] as u16 & 0x0F) << 8);
        let d2 = (xof_out[byte_pos + 1] as u16 >> 4) | ((xof_out[byte_pos + 2] as u16) << 4);

        byte_pos += 3;

        if d1 < MLKEM_Q as u16 {
            f[idx] = d1;
            idx += 1;
        }

        if idx < N && d2 < MLKEM_Q as u16 {
            f[idx] = d2;
            idx += 1;
        }
    }

    f
}

/// Sample polynomial from CBD
/// (FIPS 203 Algorithm 7).
pub fn sample_cbd(
    eta: usize,
    seed: &[u8],
    nonce: u8,
    keccak_calls: &mut Vec<KeccakCall>,
    sponge_meta: &mut Vec<(bool, bool, bool)>,
) -> [u16; N] {
    let mut prf_input = Vec::with_capacity(seed.len() + 1);
    prf_input.extend_from_slice(seed);
    prf_input.push(nonce);

    let (prf_out, calls) = shake256(&prf_input, 64 * eta);
    keccak_calls.extend_from_slice(&calls);

    for (k, _) in calls.iter().enumerate() {
        sponge_meta.push((k == 0, false, false));
    }

    let bits = &prf_out;
    let mut f = [0u16; N];

    for i in 0..N {
        let mut a = 0u16;
        let mut b = 0u16;

        for j in 0..eta {
            let bit_a = (bits[(2 * i * eta + j) / 8] >> ((2 * i * eta + j) % 8)) & 1;
            let bit_b = (bits[(2 * i * eta + eta + j) / 8] >> ((2 * i * eta + eta + j) % 8)) & 1;

            a += bit_a as u16;
            b += bit_b as u16;
        }

        f[i] = mod_sub(a, b);
    }

    f
}

// =====================================================
// Native ML-KEM arithmetic tests
// =====================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chiplets::pqc::mlkem::MlKemLevel;

    #[test]
    fn mod_arithmetic_basic() {
        assert_eq!(mod_add(3328, 1), 0);
        assert_eq!(mod_sub(0, 1), 3328);
        assert_eq!(mod_mul(17, 17), 289);
        assert_eq!(mod_reduce(3329), 0);
        assert_eq!(mod_reduce(3330), 1);
    }

    #[test]
    fn ntt_roundtrip() {
        let mut f = [0u16; N];
        for i in 0..N {
            f[i] = (i as u16 * 7 + 3) % MLKEM_Q as u16;
        }

        let original = f;
        ntt_forward(&mut f);
        ntt_inverse(&mut f);

        assert_eq!(f, original, "NTT roundtrip failed");
    }

    #[test]
    fn poly_basemul_commutativity() {
        let mut a = [0u16; N];
        let mut b = [0u16; N];

        for i in 0..N {
            a[i] = ((i * 13 + 5) % MLKEM_Q as usize) as u16;
            b[i] = ((i * 7 + 11) % MLKEM_Q as usize) as u16;
        }

        ntt_forward(&mut a);
        ntt_forward(&mut b);

        let ab = poly_basemul(&a, &b);
        let ba = poly_basemul(&b, &a);

        assert_eq!(ab, ba);
    }

    #[test]
    fn compress_decompress_roundtrip() {
        // 3328 (q-1) wraps around correctly — skip it.
        for x in [0u16, 1, 100, 1664, 1000] {
            let c = compress_d(MlKemLevel::MLKEM_768.du, x);
            let d = decompress_d(MlKemLevel::MLKEM_768.du, c);

            // Lossy, but should be close
            let diff = d.abs_diff(x);
            assert!(
                diff < 4,
                "compress/decompress error too large: {} -> {} -> {}, diff={}",
                x,
                c,
                d,
                diff,
            );
        }
    }

    #[test]
    fn byte_encode_decode_roundtrip() {
        let mut f = [0u16; N];
        for i in 0..N {
            f[i] = (i as u16 * 3) % MLKEM_Q as u16;
        }

        let encoded = byte_encode(12, &f);
        let decoded = byte_decode(12, &encoded);
        assert_eq!(f, decoded);
    }

    #[test]
    fn ntt_matches_fips203_reference() {
        // Reference values computed from FIPS 203 Algorithm 9
        // with input f[i] = (i*7 + 3) mod q
        let mut f = [0u16; N];
        for i in 0..N {
            f[i] = (i as u16 * 7 + 3) % MLKEM_Q as u16;
        }

        ntt_forward(&mut f);

        assert_eq!(f[0], 1606, "NTT output[0]");
        assert_eq!(f[1], 1189, "NTT output[1]");
        assert_eq!(f[2], 756, "NTT output[2]");
        assert_eq!(f[3], 17, "NTT output[3]");
        assert_eq!(f[254], 1132, "NTT output[254]");
        assert_eq!(f[255], 1563, "NTT output[255]");
    }

    // =====================================================
    // Compress / Decompress Systematic Tests
    // =====================================================

    #[test]
    fn compress_decompress_all_d_values() {
        // Test compress/decompress for d=1, 4, 10
        // at boundary values.
        for &d in &[1usize, 4, 10] {
            let max_compressed = (1u32 << d) - 1;
            for x in 0..MLKEM_Q as u16 {
                let c = compress_d(d, x);
                assert!(
                    (c as u32) <= max_compressed,
                    "compress_d({d}, {x}) = {c} > max {max_compressed}",
                );

                let y = decompress_d(d, c);
                assert!((y as u32) < MLKEM_Q, "decompress_d({d}, {c}) = {y} >= q",);
            }

            // Verify decompress(compress(x)) ≈ x
            // Error bound:
            // |x - decompress(compress(x))| < q/(2^(d+1))
            let error_bound = MLKEM_Q as f64 / (1u64 << (d + 1)) as f64;
            for &x in &[0u16, 1, 100, 1664, 3000, 3328] {
                let c = compress_d(d, x);
                let y = decompress_d(d, c);

                let diff = y.abs_diff(x);
                let diff = core::cmp::min(diff, MLKEM_Q as u16 - diff);

                assert!(
                    (diff as f64) <= error_bound.ceil(),
                    "compress_d({d}, {x}): error {diff} > bound {error_bound:.1}",
                );
            }
        }
    }

    // =====================================================
    // ByteEncode / ByteDecode for all d values
    // =====================================================

    #[test]
    fn byte_encode_decode_all_d_values() {
        for &d in &[1usize, 4, 10, 12] {
            let mask = (1u32 << d) - 1;

            // Test with structured input
            let mut f = [0u16; N];
            for i in 0..N {
                f[i] = ((i as u32 * 17 + 5) & mask) as u16;
            }

            let encoded = byte_encode(d, &f);
            let decoded = byte_decode(d, &encoded);

            assert_eq!(f, decoded, "ByteEncode/Decode roundtrip failed for d={d}",);
        }
    }

    #[test]
    fn byte_encode_decode_boundary_values() {
        // d=12:
        // max value is 4095,
        // but ML-KEM uses mod q=3329
        let d = 12;

        let mut f = [0u16; N];
        f[0] = 0;
        f[1] = 1;
        f[2] = 3328; // q-1
        f[3] = 4095; // 2^12 - 1 (max for d=12)

        let encoded = byte_encode(d, &f);
        let decoded = byte_decode(d, &encoded);

        assert_eq!(decoded[0], 0);
        assert_eq!(decoded[1], 1);
        assert_eq!(decoded[2], 3328);
        assert_eq!(decoded[3], 4095);
    }

    // =====================================================
    // SampleNTT / CBD Verification
    // =====================================================

    #[test]
    fn sample_ntt_produces_valid_coefficients() {
        // All coefficients must be in [0, q)
        let mut kc = Vec::new();
        let mut _sm = Vec::new();

        for i in 0..3u8 {
            for j in 0..3u8 {
                let rho = [i ^ j; 32];
                let poly = sample_ntt(&rho, i, j, &mut kc, &mut _sm);

                for (k, &coeff) in poly.iter().enumerate() {
                    assert!(
                        (coeff as u32) < MLKEM_Q,
                        "SampleNTT({i},{j})[{k}] = {coeff} >= q",
                    );
                }
            }
        }
    }

    #[test]
    fn cbd_produces_valid_coefficients() {
        // CBD_eta produces values in [0, q) that represent
        // [-eta, eta] mod q.
        let mut kc = Vec::new();
        for eta in [2usize] {
            for nonce in 0..6u8 {
                let mut _sm = Vec::new();

                let seed = [nonce; 32];
                let poly = sample_cbd(eta, &seed, nonce, &mut kc, &mut _sm);

                for (k, &coeff) in poly.iter().enumerate() {
                    assert!(
                        (coeff as u32) < MLKEM_Q,
                        "CBD_eta{eta}(nonce={nonce})[{k}] = {coeff} >= q",
                    );

                    // Value must be in {0, 1, ..., eta, q-eta, ..., q-1}
                    let signed = if coeff as u32 > MLKEM_Q / 2 {
                        coeff as i32 - MLKEM_Q as i32
                    } else {
                        coeff as i32
                    };

                    assert!(
                        signed.unsigned_abs() as usize <= eta,
                        "CBD_eta{eta}(nonce={nonce})[{k}]: |{signed}| > {eta}",
                    );
                }
            }
        }
    }

    #[test]
    fn sample_ntt_deterministic() {
        // Same input must produce same output
        let rho = [0x42u8; 32];

        let mut kc1 = Vec::new();
        let mut kc2 = Vec::new();
        let mut _sm = Vec::new();

        let a = sample_ntt(&rho, 0, 1, &mut kc1, &mut _sm);
        let b = sample_ntt(&rho, 0, 1, &mut kc2, &mut _sm);

        assert_eq!(a, b, "SampleNTT is not deterministic");
    }

    #[test]
    fn sample_ntt_different_indices_differ() {
        // Different (i,j) must produce
        // different polynomials.
        let rho = [0x42u8; 32];

        let mut kc = Vec::new();
        let mut _sm = Vec::new();

        let a01 = sample_ntt(&rho, 0, 1, &mut kc, &mut _sm);
        let a10 = sample_ntt(&rho, 1, 0, &mut kc, &mut _sm);

        assert_ne!(a01, a10, "SampleNTT(0,1) == SampleNTT(1,0)");
    }
}
