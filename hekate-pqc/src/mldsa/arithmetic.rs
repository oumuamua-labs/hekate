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

use crate::mldsa::{MLDSA_Q, N};
use alloc::vec;
use alloc::vec::Vec;
use hekate_keccak::{KeccakCall, shake128, shake256};

/// Primitive 512th root of
/// unity mod q=8380417 (FIPS 204).
/// ζ^256 ≡ -1 (mod q), ζ^512 ≡ 1 (mod q).
const ZETA: u32 = 1753;

// =================================================================
// Native Mod-q Arithmetic
// =================================================================

pub fn mod_reduce(a: u64) -> u32 {
    (a % MLDSA_Q as u64) as u32
}

pub fn mod_add(a: u32, b: u32) -> u32 {
    let s = a as u64 + b as u64;
    if s >= MLDSA_Q as u64 {
        (s - MLDSA_Q as u64) as u32
    } else {
        s as u32
    }
}

pub fn mod_sub(a: u32, b: u32) -> u32 {
    let s = a as u64 + MLDSA_Q as u64 - b as u64;
    if s >= MLDSA_Q as u64 {
        (s - MLDSA_Q as u64) as u32
    } else {
        s as u32
    }
}

pub fn mod_mul(a: u32, b: u32) -> u32 {
    mod_reduce(a as u64 * b as u64)
}

/// Modular negation:
/// -a mod q.
pub fn mod_neg(a: u32) -> u32 {
    if a == 0 { 0 } else { MLDSA_Q - a }
}

/// Modular exponentiation
/// (for tests / twiddle generation).
pub fn mod_pow(mut base: u32, mut exp: u32) -> u32 {
    let mut result = 1u32;
    base %= MLDSA_Q;

    while exp > 0 {
        if exp & 1 == 1 {
            result = mod_mul(result, base);
        }
        exp >>= 1;
        base = mod_mul(base, base);
    }

    result
}

// =================================================================
// NTT-256 over Z_q (q=8380417)
// =================================================================

/// Precomputed ζ^{BitRev8(i)} mod q for i=0..255.
pub fn zeta_powers() -> [u32; N] {
    // ζ^0..ζ^511 mod q
    let mut pow = [0u32; 512];
    pow[0] = 1;

    for i in 1..512 {
        pow[i] = mod_mul(pow[i - 1], ZETA);
    }

    let mut zetas = [0u32; N];
    for i in 0..N {
        let br = (i as u8).reverse_bits() as usize;
        zetas[i] = pow[br];
    }

    zetas
}

/// Forward NTT (Cooley-Tukey, FIPS 204 Algorithm 41).
#[allow(dead_code)]
pub fn ntt_forward(f: &mut [u32; N]) {
    let zetas = zeta_powers();

    let mut k = 0usize;
    let mut len = 128;

    while len >= 1 {
        let mut start = 0;
        while start < N {
            k += 1;

            let zeta = zetas[k];

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

/// Inverse NTT (Gentleman-Sande, FIPS 204 Algorithm 42).
#[allow(dead_code)]
pub fn ntt_inverse(f: &mut [u32; N]) {
    let zetas = zeta_powers();

    let mut k = 256usize;
    let mut len = 1;

    while len <= 128 {
        let mut start = 0;
        while start < N {
            k -= 1;

            let zeta = mod_neg(zetas[k]);

            for j in start..start + len {
                let t = f[j];
                f[j] = mod_add(t, f[j + len]);
                f[j + len] = mod_mul(zeta, mod_sub(t, f[j + len]));
            }

            start += 2 * len;
        }

        len <<= 1;
    }

    // 256^{-1} mod q.
    let norm = mod_pow(256, MLDSA_Q - 2);
    for coeff in f.iter_mut() {
        *coeff = mod_mul(*coeff, norm);
    }
}

/// Pointwise multiply in NTT domain.
#[allow(dead_code)]
pub fn poly_basemul(a: &[u32; N], b: &[u32; N]) -> [u32; N] {
    let mut r = [0u32; N];
    for i in 0..N {
        r[i] = mod_mul(a[i], b[i]);
    }

    r
}

pub fn poly_add(a: &[u32; N], b: &[u32; N]) -> [u32; N] {
    let mut r = [0u32; N];
    for i in 0..N {
        r[i] = mod_add(a[i], b[i]);
    }

    r
}

pub fn poly_sub(a: &[u32; N], b: &[u32; N]) -> [u32; N] {
    let mut r = [0u32; N];
    for i in 0..N {
        r[i] = mod_sub(a[i], b[i]);
    }

    r
}

// =================================================================
// HighBits / LowBits / MakeHint / UseHint (FIPS 204 §8.3–8.5)
// =================================================================

/// Decompose r into (r1, r0)
/// such that r ≡ r1·2γ₂ + r0 (mod q).
/// r0 ∈ (-γ₂, γ₂], r1 ∈ [0, (q-1)/(2γ₂)].
/// FIPS 204 Algorithm 35:
/// Decompose.
pub fn decompose(r: u32, gamma2: u32) -> (u32, i32) {
    let r_plus = r % MLDSA_Q;

    let mut r0 = (r_plus % (2 * gamma2)) as i32;
    if r0 as u32 > gamma2 {
        r0 -= (2 * gamma2) as i32;
    }

    // r_plus - r0 can underflow
    // if r0 > 0 and r_plus < r0,
    // but r0 ≤ γ₂ < q/2 and r_plus < q,
    // so subtraction is safe.
    let diff = if r0 >= 0 {
        r_plus - r0 as u32
    } else {
        r_plus + (-r0) as u32
    };

    let mut r1 = diff / (2 * gamma2);

    // Corner case:
    // r1 = (q-1)/(2γ₂) means r0
    // wraps (FIPS 204 line 6).
    if diff == MLDSA_Q - 1 {
        r1 = 0;
        r0 -= 1;
    }

    (r1, r0)
}

/// FIPS 204 Algorithm 36.
#[allow(dead_code)]
pub fn high_bits(r: u32, gamma2: u32) -> u32 {
    decompose(r, gamma2).0
}

/// FIPS 204 Algorithm 39:
/// UseHint.
pub fn use_hint(h: bool, r: u32, gamma2: u32) -> u32 {
    let (r1, r0) = decompose(r, gamma2);
    if !h {
        return r1;
    }

    let m = (MLDSA_Q - 1) / (2 * gamma2);
    if r0 > 0 {
        // r1 + 1 mod m
        if r1 + 1 >= m { 0 } else { r1 + 1 }
    } else {
        // r1 - 1 mod m
        if r1 == 0 { m - 1 } else { r1 - 1 }
    }
}

// =================================================================
// Sampling (FIPS 204 §8.1–8.2)
// =================================================================

/// ExpandA:
/// sample k×l matrix of NTT-domain polynomials
/// from ρ via SHAKE-128 (FIPS 204 Algorithm 32: ExpandA).
pub fn expand_a(
    rho: &[u8; 32],
    k: usize,
    l: usize,
    keccak_calls: &mut Vec<KeccakCall>,
    sponge_meta: &mut Vec<(bool, bool, bool)>,
) -> Vec<[u32; N]> {
    let mut a_hat = Vec::with_capacity(k * l);
    for r in 0..k {
        for s in 0..l {
            let poly = rejection_sample_ntt(rho, s as u8, r as u8, keccak_calls, sponge_meta);
            a_hat.push(poly);
        }
    }

    a_hat
}

/// RejNTTPoly:
/// rejection-sample NTT-domain poly from
/// SHAKE-128 stream (FIPS 204 Algorithm 33).
fn rejection_sample_ntt(
    rho: &[u8; 32],
    s: u8,
    r: u8,
    keccak_calls: &mut Vec<KeccakCall>,
    sponge_meta: &mut Vec<(bool, bool, bool)>,
) -> [u32; N] {
    let mut seed = Vec::with_capacity(34);
    seed.extend_from_slice(rho);
    seed.push(s);
    seed.push(r);

    // Over-request to handle rejections.
    // q ≈ 2^23, so ~3 bytes per sample,
    // rejection rate ≈ 0.5%.
    let (xof_out, calls) = shake128(&seed, 3 * N + 128);
    keccak_calls.extend_from_slice(&calls);

    for (i, _) in calls.iter().enumerate() {
        sponge_meta.push((i == 0, false, true));
    }

    let mut f = [0u32; N];
    let mut idx = 0;
    let mut byte_pos = 0;

    while idx < N && byte_pos + 2 < xof_out.len() {
        let b0 = xof_out[byte_pos] as u32;
        let b1 = xof_out[byte_pos + 1] as u32;
        let b2 = xof_out[byte_pos + 2] as u32;

        byte_pos += 3;

        // FIPS 204 Algorithm 14
        // (CoeffFromThreeBytes).
        let coeff = b0 | (b1 << 8) | ((b2 & 0x7F) << 16);
        if coeff < MLDSA_Q {
            f[idx] = coeff;
            idx += 1;
        }
    }

    assert_eq!(idx, N, "rejection sampling exhausted XOF output");

    f
}

/// SampleInBall:
/// sparse ternary polynomial with
/// exactly τ non-zero entries in {-1, +1}
/// (FIPS 204 Algorithm 34).
pub fn sample_in_ball(
    c_tilde: &[u8],
    tau: usize,
    keccak_calls: &mut Vec<KeccakCall>,
    sponge_meta: &mut Vec<(bool, bool, bool)>,
) -> [u32; N] {
    // 8 bytes for sign bits + rejection-sampled
    // positions. Worst-case rejection rate ~19%
    // at first iteration (i = N - tau).
    // 3× margin handles all practical tau values.
    let needed = 8 + tau * 3;
    let (stream, calls) = shake256(c_tilde, needed);

    keccak_calls.extend_from_slice(&calls);

    for (i, _) in calls.iter().enumerate() {
        sponge_meta.push((i == 0, false, false));
    }

    let mut c = [0u32; N];

    // First 8 bytes -> 64-bit sign mask
    let mut signs = u64::from_le_bytes(stream[..8].try_into().unwrap());
    let mut stream_pos = 8;

    // FIPS 204 Algorithm 34:
    // rejection sampling. Read bytes
    // from stream, reject any byte > i
    // to get uniform j ∈ [0, i].
    for i in (N - tau)..N {
        loop {
            assert!(
                stream_pos < stream.len(),
                "SampleInBall: exhausted SHAKE-256 stream"
            );

            let byte = stream[stream_pos] as usize;
            stream_pos += 1;

            if byte <= i {
                let j = byte;

                c[i] = c[j];
                c[j] = if signs & 1 == 0 { 1 } else { mod_neg(1) };

                signs >>= 1;

                break;
            }
        }
    }

    c
}

// =================================================================
// Coefficient Encoding / Decoding (FIPS 204 §8.6)
// =================================================================

/// SimpleBitPack:
/// pack d-bit unsigned coefficients.
pub fn bit_pack(coeffs: &[u32], d: usize) -> Vec<u8> {
    let total_bits = coeffs.len() * d;

    let mut bytes = vec![0u8; total_bits.div_ceil(8)];
    let mut bit_idx = 0;

    for &c in coeffs {
        for b in 0..d {
            let byte_pos = (bit_idx + b) / 8;
            let bit_pos = (bit_idx + b) % 8;
            bytes[byte_pos] |= (((c >> b) & 1) as u8) << bit_pos;
        }

        bit_idx += d;
    }

    bytes
}

/// SimpleBitUnpack:
/// unpack d-bit unsigned coefficients.
pub fn bit_unpack(bytes: &[u8], n: usize, d: usize) -> Vec<u32> {
    let mut coeffs = Vec::with_capacity(n);
    let mask = (1u32 << d) - 1;

    let mut bit_idx = 0;
    for _ in 0..n {
        let mut val = 0u32;
        for b in 0..d {
            let byte_pos = (bit_idx + b) / 8;
            let bit_pos = (bit_idx + b) % 8;

            if byte_pos < bytes.len() {
                val |= (((bytes[byte_pos] >> bit_pos) & 1) as u32) << b;
            }
        }

        coeffs.push(val & mask);
        bit_idx += d;
    }

    coeffs
}

/// w1Encode:
/// encode HighBits output for hashing
/// (FIPS 204 Algorithm 28).
/// Coefficients are in [0, (q-1)/(2γ₂) - 1].
pub fn w1_encode(w1: &[[u32; N]], gamma2: u32) -> Vec<u8> {
    let m = (MLDSA_Q - 1) / (2 * gamma2);

    // Bits needed for max value m-1.
    // m=16 (ML-DSA-65) -> 4 bits,
    // m=44 (ML-DSA-44) -> 6 bits.
    let bits = 32 - (m - 1).leading_zeros() as usize;

    let mut encoded = Vec::new();
    for poly in w1 {
        encoded.extend_from_slice(&bit_pack(poly, bits));
    }

    encoded
}

/// Decode z polynomial from signature
/// (FIPS 204 Algorithm 23: sigDecode, z part).
/// z coefficients encoded as
/// γ₁ - z mod q, with γ₁ bits per coeff.
pub fn decode_z_poly(bytes: &[u8], gamma1: u32) -> [u32; N] {
    let gamma1_bits = if gamma1 == (1 << 17) { 18 } else { 20 };
    let raw = bit_unpack(bytes, N, gamma1_bits);

    let mut z = [0u32; N];
    for i in 0..N {
        // Encoded as γ₁ - z_i,
        // so z_i = γ₁ - encoded (mod q).
        z[i] = mod_sub(gamma1, raw[i]);
    }

    z
}

/// Decode t1 from public key.
/// t1 coefficients are 10 bits
/// (d=13 dropped bits).
pub fn decode_t1(bytes: &[u8]) -> [u32; N] {
    let raw = bit_unpack(bytes, N, 10);

    let mut t1 = [0u32; N];
    t1[..N].copy_from_slice(&raw[..N]);

    t1
}

/// Decode hint vector h from signature
/// (FIPS 204 Algorithm 23: sigDecode, h part).
pub fn decode_hint(bytes: &[u8], k: usize, omega: usize) -> Option<Vec<[bool; N]>> {
    if bytes.len() != omega + k {
        return None;
    }

    let mut h = vec![[false; N]; k];
    let mut idx = 0;

    for i in 0..k {
        let limit = bytes[omega + i] as usize;
        if limit < idx || limit > omega {
            return None;
        }

        let mut prev: Option<usize> = None;
        while idx < limit {
            let pos = bytes[idx] as usize;
            if pos >= N {
                return None;
            }

            // FIPS 204 Algorithm 20:
            // positions within each polynomial
            // must be strictly ascending.
            if let Some(p) = prev
                && pos <= p
            {
                return None;
            }

            prev = Some(pos);

            h[i][pos] = true;
            idx += 1;
        }
    }

    // Total hint weight ≤ ω
    let total: usize = h.iter().map(|p| p.iter().filter(|&&b| b).count()).sum();
    if total > omega {
        return None;
    }

    Some(h)
}

// =================================================================
// Tests
// =================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mldsa::MlDsaLevel;

    #[test]
    fn mod_arithmetic_basic() {
        assert_eq!(mod_add(MLDSA_Q - 1, 1), 0);
        assert_eq!(mod_sub(0, 1), MLDSA_Q - 1);
        assert_eq!(mod_mul(1753, 1753), mod_reduce(1753u64 * 1753));
        assert_eq!(mod_reduce(MLDSA_Q as u64), 0);
        assert_eq!(mod_reduce(MLDSA_Q as u64 + 1), 1);
        assert_eq!(mod_neg(0), 0);
        assert_eq!(mod_neg(1), MLDSA_Q - 1);
        assert_eq!(mod_add(mod_neg(42), 42), 0);
    }

    #[test]
    fn mod_pow_basic() {
        assert_eq!(mod_pow(ZETA, 0), 1);
        assert_eq!(mod_pow(ZETA, 1), ZETA);

        // ζ is a 512th root:
        // ζ^256 ≡ -1,
        // ζ^512 ≡ 1
        assert_eq!(mod_pow(ZETA, 256), MLDSA_Q - 1);
        assert_eq!(mod_pow(ZETA, 512), 1);

        // Not a 256th root
        assert_ne!(mod_pow(ZETA, 128), MLDSA_Q - 1);
    }

    #[test]
    fn ntt_roundtrip() {
        let mut f = [0u32; N];
        for i in 0..N {
            f[i] = (i as u32 * 7 + 3) % MLDSA_Q;
        }

        let original = f;
        ntt_forward(&mut f);
        ntt_inverse(&mut f);

        assert_eq!(f, original, "NTT roundtrip failed");
    }

    #[test]
    fn ntt_linearity() {
        let mut a = [0u32; N];
        let mut b = [0u32; N];
        for i in 0..N {
            a[i] = (i as u32 * 13 + 5) % MLDSA_Q;
            b[i] = (i as u32 * 7 + 11) % MLDSA_Q;
        }

        let sum_ab = poly_add(&a, &b);

        let mut sum_ntt = [0u32; N];
        sum_ntt.copy_from_slice(&sum_ab);

        ntt_forward(&mut sum_ntt);

        ntt_forward(&mut a);
        ntt_forward(&mut b);

        let ntt_sum = poly_add(&a, &b);

        assert_eq!(sum_ntt, ntt_sum, "NTT(a+b) ≠ NTT(a)+NTT(b)");
    }

    #[test]
    fn poly_basemul_commutativity() {
        let mut a = [0u32; N];
        let mut b = [0u32; N];

        for i in 0..N {
            a[i] = ((i as u64 * 13 + 5) % MLDSA_Q as u64) as u32;
            b[i] = ((i as u64 * 7 + 11) % MLDSA_Q as u64) as u32;
        }

        ntt_forward(&mut a);
        ntt_forward(&mut b);

        let ab = poly_basemul(&a, &b);
        let ba = poly_basemul(&b, &a);

        assert_eq!(ab, ba);
    }

    #[test]
    fn ntt_multiply_via_basemul() {
        // Ring is Z_q[X]/(X^256+1).
        let mut a = [0u32; N];
        let mut b = [0u32; N];

        a[0] = 1;
        a[1] = 2;
        b[0] = 3;
        b[1] = 4;

        let mut a_ntt = a;
        let mut b_ntt = b;

        ntt_forward(&mut a_ntt);
        ntt_forward(&mut b_ntt);

        let c_ntt = poly_basemul(&a_ntt, &b_ntt);
        let mut c = c_ntt;
        ntt_inverse(&mut c);

        assert_eq!(c[0], 3);
        assert_eq!(c[1], 10);
        assert_eq!(c[2], 8);

        for i in 3..N {
            assert_eq!(c[i], 0, "c[{i}] should be 0");
        }
    }

    #[test]
    fn decompose_basic() {
        let gamma2 = 261888u32; // (q-1)/32

        // r = 0 -> r1=0, r0=0
        let (r1, r0) = decompose(0, gamma2);
        assert_eq!(r1, 0);
        assert_eq!(r0, 0);

        // r = 1 -> r1=0, r0=1
        let (r1, r0) = decompose(1, gamma2);
        assert_eq!(r1, 0);
        assert_eq!(r0, 1);

        // Roundtrip:
        // r ≡ r1·2γ₂ + r0 (mod q) for all r < q
        for r in (0..MLDSA_Q).step_by(1000) {
            let (r1, r0) = decompose(r, gamma2);
            let reconstructed = if r0 >= 0 {
                mod_add(r1 * 2 * gamma2, r0 as u32)
            } else {
                mod_sub(r1 * 2 * gamma2, (-r0) as u32)
            };
            assert_eq!(reconstructed, r, "decompose roundtrip failed for r={r}");
        }
    }

    #[test]
    fn use_hint_reverses_make_hint() {
        let gamma2 = 261888u32;
        let m = (MLDSA_Q - 1) / (2 * gamma2);

        for r in (0..MLDSA_Q).step_by(5000) {
            let r1 = high_bits(r, gamma2);
            assert!(r1 < m, "r1={r1} >= m={m} for r={r}");

            // UseHint(false, r) == HighBits(r)
            assert_eq!(use_hint(false, r, gamma2), r1);
        }
    }

    #[test]
    fn bit_pack_unpack_roundtrip() {
        let coeffs: Vec<u32> = (0..N as u32).collect();
        for d in [10usize, 18, 20, 23] {
            let mask = (1u32 << d) - 1;
            let masked: Vec<u32> = coeffs.iter().map(|&c| c & mask).collect();
            let packed = bit_pack(&masked, d);
            let unpacked = bit_unpack(&packed, N, d);

            assert_eq!(
                masked, unpacked,
                "bit_pack/unpack roundtrip failed for d={d}"
            );
        }
    }

    #[test]
    fn sample_ntt_deterministic() {
        let rho = [0x42u8; 32];

        let mut kc1 = Vec::new();
        let mut kc2 = Vec::new();
        let mut sm = Vec::new();

        let a = rejection_sample_ntt(&rho, 0, 1, &mut kc1, &mut sm);
        let b = rejection_sample_ntt(&rho, 0, 1, &mut kc2, &mut sm);

        assert_eq!(a, b, "rejection_sample_ntt not deterministic");
    }

    #[test]
    fn sample_ntt_valid_range() {
        let rho = [0xABu8; 32];

        let mut kc = Vec::new();
        let mut sm = Vec::new();

        for r in 0..3u8 {
            for s in 0..3u8 {
                let poly = rejection_sample_ntt(&rho, s, r, &mut kc, &mut sm);
                for (i, &c) in poly.iter().enumerate() {
                    assert!(c < MLDSA_Q, "expand_a[{r},{s}][{i}] = {c} >= q");
                }
            }
        }
    }

    #[test]
    fn sample_in_ball_weight() {
        let level = MlDsaLevel::MLDSA_65;
        let c_tilde = [0x99u8; 48];

        let mut kc = Vec::new();
        let mut sm = Vec::new();

        let c = sample_in_ball(&c_tilde, level.tau, &mut kc, &mut sm);

        let nonzero: usize = c.iter().filter(|&&x| x != 0).count();
        assert_eq!(nonzero, level.tau, "SampleInBall weight mismatch");

        // All non-zero entries must be ±1
        for (i, &x) in c.iter().enumerate() {
            if x != 0 {
                assert!(
                    x == 1 || x == MLDSA_Q - 1,
                    "c[{i}] = {x}, expected ±1 mod q"
                );
            }
        }
    }

    #[test]
    fn w1_encode_range() {
        let gamma2 = 261888u32;
        let m = (MLDSA_Q - 1) / (2 * gamma2);

        let mut poly = [0u32; N];
        for i in 0..N {
            poly[i] = (i as u32) % m;
        }

        let encoded = w1_encode(&[poly], gamma2);
        assert!(!encoded.is_empty());

        // Verify decode roundtrip
        let bits = 32 - (m - 1).leading_zeros() as usize;
        let decoded = bit_unpack(&encoded, N, bits);
        for i in 0..N {
            assert_eq!(decoded[i], poly[i], "w1_encode roundtrip mismatch at {i}");
        }
    }

    #[test]
    fn barrett_reduction_edge_cases() {
        assert_eq!(mod_reduce(0), 0);
        assert_eq!(mod_reduce(1), 1);
        assert_eq!(mod_reduce(MLDSA_Q as u64 - 1), MLDSA_Q - 1);
        assert_eq!(mod_reduce(MLDSA_Q as u64), 0);
        assert_eq!(mod_reduce(MLDSA_Q as u64 * 2), 0);

        // Large product:
        // (q-1)^2
        let big = (MLDSA_Q as u64 - 1) * (MLDSA_Q as u64 - 1);
        assert_eq!(mod_reduce(big), mod_mul(MLDSA_Q - 1, MLDSA_Q - 1));

        // Verify against naive reduction
        for a in [0u64, 1, 42, 8380416, 8380417, 8380418, 100_000_000] {
            assert_eq!(
                mod_reduce(a),
                (a % MLDSA_Q as u64) as u32,
                "mod_reduce({a})"
            );
        }
    }
}
