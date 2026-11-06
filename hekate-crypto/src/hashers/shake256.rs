// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! SHAKE256 with a 512-bit output (FIPS-202) on the `keccak` permutation:
//! the digest of the 220-bit profile; `DefaultHasher` when enabled.

use crate::Hasher;
#[cfg(target_arch = "aarch64")]
use crate::hashers::sponge::Batch;
use crate::hashers::sponge::{RATE, SHAKE_PAD, squeeze, xor_block};
use keccak::Keccak;

#[derive(Clone, Debug)]
pub struct Shake256Hasher {
    keccak: Keccak,
    state: [u64; 25],
    block: [u8; RATE],
    filled: usize,
}

impl Hasher for Shake256Hasher {
    type Digest = [u8; 64];

    fn new() -> Self {
        Self {
            keccak: Keccak::new(),
            state: [0; 25],
            block: [0; RATE],
            filled: 0,
        }
    }

    fn update(&mut self, mut data: &[u8]) {
        if self.filled > 0 {
            let take = (RATE - self.filled).min(data.len());

            self.block[self.filled..self.filled + take].copy_from_slice(&data[..take]);

            self.filled += take;

            data = &data[take..];

            if self.filled < RATE {
                return;
            }

            xor_block(&mut self.state, &self.block);
            self.permute();

            self.filled = 0;
        }

        let (blocks, tail) = data.as_chunks::<RATE>();
        for block in blocks {
            xor_block(&mut self.state, block);
            self.permute();
        }

        self.block[..tail.len()].copy_from_slice(tail);

        self.filled = tail.len();
    }

    fn finalize(mut self) -> [u8; 64] {
        self.finalize_reset()
    }

    fn finalize_reset(&mut self) -> [u8; 64] {
        self.block[self.filled..].fill(0);
        self.block[self.filled] ^= SHAKE_PAD;
        self.block[RATE - 1] ^= 0x80;

        xor_block(&mut self.state, &self.block);
        self.permute();

        let out = squeeze(&self.state);

        *self = Self::new();

        out
    }

    #[cfg(target_arch = "aarch64")]
    fn digest_many(data: &[u8], len: usize, out: &mut [[u8; 64]]) {
        Batch {
            data,
            len,
            out,
            pad: SHAKE_PAD,
        }
        .run();
    }
}

impl Shake256Hasher {
    fn permute(&mut self) {
        self.keccak.with_f1600(|f1600| f1600(&mut self.state));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// FIPS-202 SHAKE256 of the empty message, 512 bits.
    const SHAKE256_EMPTY: [u8; 64] = [
        0x46, 0xb9, 0xdd, 0x2b, 0x0b, 0xa8, 0x8d, 0x13, 0x23, 0x3b, 0x3f, 0xeb, 0x74, 0x3e, 0xeb,
        0x24, 0x3f, 0xcd, 0x52, 0xea, 0x62, 0xb8, 0x1b, 0x82, 0xb5, 0x0c, 0x27, 0x64, 0x6e, 0xd5,
        0x76, 0x2f, 0xd7, 0x5d, 0xc4, 0xdd, 0xd8, 0xc0, 0xf2, 0x00, 0xcb, 0x05, 0x01, 0x9d, 0x67,
        0xb5, 0x92, 0xf6, 0xfc, 0x82, 0x1c, 0x49, 0x47, 0x9a, 0xb4, 0x86, 0x40, 0x29, 0x2e, 0xac,
        0xb3, 0xb7, 0xc4, 0xbe,
    ];

    #[test]
    fn shake256_empty_message_matches_fips_202() {
        assert_eq!(Shake256Hasher::digest(&[]), SHAKE256_EMPTY);
        assert_eq!(Shake256Hasher::new().finalize(), SHAKE256_EMPTY);
    }

    #[test]
    fn shake256_update_splits_match_one_shot() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 37 + 11) as u8).collect();
        let whole = Shake256Hasher::digest(&data);

        for split in [0usize, 1, 135, 136, 137, 271, 272, 273, 999, 1000] {
            let mut hasher = Shake256Hasher::new();
            hasher.update(&data[..split]);
            hasher.update(&data[split..]);

            assert_eq!(hasher.finalize(), whole, "split {split}");
        }

        let mut hasher = Shake256Hasher::new();
        for byte in &data {
            hasher.update(&[*byte]);
        }

        assert_eq!(hasher.finalize_reset(), whole);
        assert_eq!(hasher.finalize(), SHAKE256_EMPTY);
    }
}
