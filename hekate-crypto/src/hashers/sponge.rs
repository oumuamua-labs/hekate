// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! The Keccak sponge the SHA-3-256 and SHAKE256 backends share:
//! block absorption and the batched permutation (FIPS-202).

#[cfg(target_arch = "aarch64")]
use keccak::{Backend, BackendClosure, Keccak, ParState1600};

pub(super) const RATE: usize = 136;

#[cfg(all(feature = "sha3", target_arch = "aarch64"))]
pub(super) const SHA3_PAD: u8 = 0x06;

#[cfg(feature = "shake256")]
pub(super) const SHAKE_PAD: u8 = 0x1F;

#[cfg(target_arch = "aarch64")]
pub(super) struct Batch<'a, const OUT: usize> {
    pub data: &'a [u8],
    pub len: usize,
    pub out: &'a mut [[u8; OUT]],
    pub pad: u8,
}

#[cfg(target_arch = "aarch64")]
impl<const OUT: usize> Batch<'_, OUT> {
    pub fn run(self) {
        Keccak::new().with_backend(self);
    }
}

#[cfg(target_arch = "aarch64")]
impl<const OUT: usize> BackendClosure for Batch<'_, OUT> {
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
                    absorb(
                        state,
                        message(self.data, self.len, g * width + lane),
                        b,
                        self.pad,
                    );
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
                absorb(
                    &mut state,
                    message(self.data, self.len, paired + i),
                    b,
                    self.pad,
                );
                permute(&mut state);
            }

            *slot = squeeze(&state);
        }
    }
}

pub(super) fn xor_block(state: &mut [u64; 25], block: &[u8; RATE]) {
    for (lane, bytes) in state.iter_mut().zip(block.as_chunks::<8>().0) {
        *lane ^= u64::from_le_bytes(*bytes);
    }
}

pub(super) fn squeeze<const OUT: usize>(state: &[u64; 25]) -> [u8; OUT] {
    const { assert!(OUT <= RATE) };

    let mut out = [0u8; OUT];
    for (bytes, lane) in out.as_chunks_mut::<8>().0.iter_mut().zip(state) {
        *bytes = lane.to_le_bytes();
    }

    out
}

#[cfg(target_arch = "aarch64")]
fn message(data: &[u8], len: usize, index: usize) -> &[u8] {
    &data[index * len..(index + 1) * len]
}

#[cfg(target_arch = "aarch64")]
fn absorb(state: &mut [u64; 25], msg: &[u8], b: usize, pad: u8) {
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
        block[tail.len()] ^= pad;
        block[RATE - 1] ^= 0x80;
    }

    xor_block(state, &block);
}
