// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! SHA-3-256 (FIPS-202, RustCrypto `sha3` crate).
use crate::Hasher;
#[cfg(target_arch = "aarch64")]
use keccak::{Backend, BackendClosure, Keccak, ParState1600};
use sha3::{Digest, Sha3_256};

#[cfg(target_arch = "aarch64")]
const RATE: usize = 136;

#[derive(Clone, Debug)]
pub struct Sha3_256Hasher {
    inner: Sha3_256,
}

impl Hasher for Sha3_256Hasher {
    const OUTPUT_SIZE: usize = 32;

    fn new() -> Self {
        Self {
            inner: Sha3_256::new(),
        }
    }

    fn update(&mut self, data: &[u8]) {
        self.inner.update(data);
    }

    fn finalize(self) -> [u8; 32] {
        self.inner.finalize().into()
    }

    fn finalize_reset(&mut self) -> [u8; 32] {
        let out = self.inner.finalize_reset();
        out.into()
    }

    #[cfg(target_arch = "aarch64")]
    fn digest_many(data: &[u8], len: usize, out: &mut [[u8; 32]]) {
        Keccak::new().with_backend(Batch { data, len, out });
    }
}

#[cfg(target_arch = "aarch64")]
struct Batch<'a> {
    data: &'a [u8],
    len: usize,
    out: &'a mut [[u8; 32]],
}

#[cfg(target_arch = "aarch64")]
impl BackendClosure for Batch<'_> {
    fn call_once<B: Backend>(self) {
        let permute_par = B::get_par_f1600();
        let permute = B::get_f1600();

        let mut par = ParState1600::<B>::default();

        let width = par.len();
        let blocks = self.len / RATE + 1;
        let paired = self.out.len() - self.out.len() % width;

        let (grouped, rest) = self.out.split_at_mut(paired);

        for (g, outs) in grouped.chunks_exact_mut(width).enumerate() {
            for state in par.iter_mut() {
                *state = [0u64; 25];
            }

            for b in 0..blocks {
                for (lane, state) in par.iter_mut().enumerate() {
                    absorb(state, message(self.data, self.len, g * width + lane), b);
                }

                permute_par(&mut par);
            }

            for (slot, state) in outs.iter_mut().zip(par.iter()) {
                *slot = squeeze(state);
            }
        }

        for (i, slot) in rest.iter_mut().enumerate() {
            let mut state = [0u64; 25];
            for b in 0..blocks {
                absorb(&mut state, message(self.data, self.len, paired + i), b);
                permute(&mut state);
            }

            *slot = squeeze(&state);
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn message(data: &[u8], len: usize, index: usize) -> &[u8] {
    &data[index * len..(index + 1) * len]
}

#[cfg(target_arch = "aarch64")]
fn absorb(state: &mut [u64; 25], msg: &[u8], b: usize) {
    let start = b * RATE;

    let mut block = [0u8; RATE];
    let full = match msg.get(start..start + RATE) {
        Some(bytes) => {
            block.copy_from_slice(bytes);

            true
        }
        None => false,
    };

    if !full {
        let tail = &msg[start..];

        block[..tail.len()].copy_from_slice(tail);
        block[tail.len()] ^= 0x06;
        block[RATE - 1] ^= 0x80;
    }

    for (lane, bytes) in state.iter_mut().zip(block.as_chunks::<8>().0) {
        *lane ^= u64::from_le_bytes(*bytes);
    }
}

#[cfg(target_arch = "aarch64")]
fn squeeze(state: &[u64; 25]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (bytes, lane) in out.as_chunks_mut::<8>().0.iter_mut().zip(state) {
        *bytes = lane.to_le_bytes();
    }

    out
}
