// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;

pub mod cpu;
pub mod sha256;
pub mod trace;

pub use cpu::CpuSha256Block;
pub use sha256::{Sha256Chiplet, Sha256Layout};
pub use trace::Sha256Call;

pub const STATE_WORDS: usize = 8;
pub const BLOCK_WORDS: usize = 16;
pub const ROUNDS: usize = 64;

/// FIPS 180-4 §5.3.3.
pub const IV: [u32; STATE_WORDS] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// FIPS 180-4 §4.2.2.
#[rustfmt::skip]
pub const K: [u32; ROUNDS] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

pub fn big_sigma0(x: u32) -> u32 {
    x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
}

pub fn big_sigma1(x: u32) -> u32 {
    x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
}

pub fn small_sigma0(x: u32) -> u32 {
    x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
}

pub fn small_sigma1(x: u32) -> u32 {
    x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
}

pub fn ch(e: u32, f: u32, g: u32) -> u32 {
    g ^ (e & (f ^ g))
}

pub fn maj(a: u32, b: u32, c: u32) -> u32 {
    b ^ ((a ^ b) & (b ^ c))
}

/// `(sum, carries)`: bit `k` of `carries` is
/// the carry out of bit `k`, bit 31 the overflow.
pub fn add_with_carries(a: u32, b: u32) -> (u32, u32) {
    let sum = a.wrapping_add(b);
    let carry_in = a ^ b ^ sum;
    let overflow = ((u64::from(a) + u64::from(b)) >> 32) as u32;

    (sum, (carry_in >> 1) | (overflow << 31))
}

pub fn schedule_word(w: &[u32; BLOCK_WORDS], t: usize) -> u32 {
    small_sigma1(w[(t + 14) % BLOCK_WORDS])
        .wrapping_add(w[(t + 9) % BLOCK_WORDS])
        .wrapping_add(small_sigma0(w[(t + 1) % BLOCK_WORDS]))
        .wrapping_add(w[t % BLOCK_WORDS])
}

pub fn round(state: &mut [u32; STATE_WORDS], k: u32, w: u32) {
    let [a, b, c, d, e, f, g, h] = *state;

    let t1 = h
        .wrapping_add(big_sigma1(e))
        .wrapping_add(ch(e, f, g))
        .wrapping_add(k)
        .wrapping_add(w);

    let t2 = big_sigma0(a).wrapping_add(maj(a, b, c));

    *state = [t1.wrapping_add(t2), a, b, c, d.wrapping_add(t1), e, f, g];
}

pub fn rounds(h_in: &[u32; STATE_WORDS], block: &[u32; BLOCK_WORDS]) -> [u32; STATE_WORDS] {
    let mut w = *block;
    let mut state = *h_in;

    for t in 0..ROUNDS {
        if t >= BLOCK_WORDS {
            w[t % BLOCK_WORDS] = schedule_word(&w, t);
        }

        round(&mut state, K[t], w[t % BLOCK_WORDS]);
    }

    state
}

pub fn feed_forward(h_in: &[u32; STATE_WORDS], state: &[u32; STATE_WORDS]) -> [u32; STATE_WORDS] {
    let mut h_out = *h_in;
    for (out, s) in h_out.iter_mut().zip(state) {
        *out = out.wrapping_add(*s);
    }

    h_out
}

/// FIPS 180-4 §6.2.2, big-endian words.
pub fn compress(h_in: &[u32; STATE_WORDS], block: &[u32; BLOCK_WORDS]) -> [u32; STATE_WORDS] {
    feed_forward(h_in, &rounds(h_in, block))
}

/// FIPS 180-4 §5.1.1, big-endian words.
pub fn pad_message(msg: &[u8]) -> Vec<[u32; BLOCK_WORDS]> {
    let bit_len = (msg.len() as u64) * 8;
    let padded_len = (msg.len() + 1 + 8).div_ceil(64) * 64;

    let mut bytes = Vec::with_capacity(padded_len);
    bytes.extend_from_slice(msg);
    bytes.push(0x80);
    bytes.resize(padded_len - 8, 0);
    bytes.extend_from_slice(&bit_len.to_be_bytes());

    bytes
        .as_chunks::<64>()
        .0
        .iter()
        .map(|chunk| {
            let mut words = [0u32; BLOCK_WORDS];
            for (word, quad) in words.iter_mut().zip(chunk.as_chunks::<4>().0) {
                *word = u32::from_be_bytes(*quad);
            }

            words
        })
        .collect()
}

pub fn sha256_words(msg: &[u8]) -> [u32; STATE_WORDS] {
    pad_message(msg)
        .iter()
        .fold(IV, |h, block| compress(&h, block))
}

pub fn digest_bytes(words: &[u32; STATE_WORDS]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (quad, word) in out.as_chunks_mut::<4>().0.iter_mut().zip(words) {
        *quad = word.to_be_bytes();
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abc() {
        assert_eq!(
            sha256_words(b"abc"),
            [
                0xba7816bf, 0x8f01cfea, 0x414140de, 0x5dae2223, 0xb00361a3, 0x96177a9c, 0xb410ff61,
                0xf20015ad
            ]
        );
    }

    #[test]
    fn two_blocks() {
        assert_eq!(
            sha256_words(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            [
                0x248d6a61, 0xd20638b8, 0xe5c02693, 0x0c3e6039, 0xa33ce459, 0x64ff2167, 0xf6ecedd4,
                0x19db06c1
            ]
        );
    }

    #[test]
    fn empty() {
        assert_eq!(
            sha256_words(b""),
            [
                0xe3b0c442, 0x98fc1c14, 0x9afbf4c8, 0x996fb924, 0x27ae41e4, 0x649b934c, 0xa495991b,
                0x7852b855
            ]
        );
    }

    #[test]
    fn padding_boundaries() {
        assert_eq!(pad_message(&[0u8; 55]).len(), 1);
        assert_eq!(pad_message(&[0u8; 56]).len(), 2);
        assert_eq!(pad_message(&[0u8; 64]).len(), 2);
        assert_eq!(pad_message(&[0u8; 119]).len(), 2);
        assert_eq!(pad_message(&[0u8; 120]).len(), 3);
    }

    #[test]
    fn carries_reproduce_the_sum() {
        let pairs = [
            (0u32, 0u32),
            (1, 1),
            (u32::MAX, 1),
            (u32::MAX, u32::MAX),
            (0x8000_0000, 0x8000_0000),
            (0x1234_5678, 0x9abc_def0),
        ];

        for (a, b) in pairs {
            let (sum, carries) = add_with_carries(a, b);
            assert_eq!(sum, a.wrapping_add(b));

            let mut c_prev = 0u32;
            for k in 0..32 {
                let a_k = (a >> k) & 1;
                let b_k = (b >> k) & 1;
                let expected = (a_k & b_k) | (a_k & c_prev) | (b_k & c_prev);
                let c_k = (carries >> k) & 1;

                assert_eq!(c_k, expected, "bit {k} of {a:#x} + {b:#x}");
                assert_eq!((sum >> k) & 1, a_k ^ b_k ^ c_prev);

                c_prev = c_k;
            }
        }
    }
}
