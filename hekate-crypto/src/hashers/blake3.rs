// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! BLAKE3 (Maximum Performance)
use crate::Hasher;
#[cfg(feature = "blake3")]
use blake3;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use core::arch::aarch64::*;
use core::fmt::Debug;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
const LANES: usize = 4;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
const BLOCK_LEN: usize = 64;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
const CHUNK_LEN: usize = 1024;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
const CHUNK_START: u32 = 1;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
const CHUNK_END: u32 = 2;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
const ROOT: u32 = 8;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
const MSG_SCHEDULE: [[usize; 16]; 7] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
    [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
    [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
    [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
    [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
    [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
];

#[cfg(feature = "blake3")]
#[derive(Clone, Debug)]
pub struct Blake3Hasher {
    inner: blake3::Hasher,
}

#[cfg(feature = "blake3")]
impl Hasher for Blake3Hasher {
    const OUTPUT_SIZE: usize = 32;

    fn new() -> Self {
        Self {
            inner: blake3::Hasher::new(),
        }
    }

    fn update(&mut self, data: &[u8]) {
        self.inner.update(data);
    }

    fn finalize(self) -> [u8; 32] {
        self.inner.finalize().into()
    }

    fn finalize_reset(&mut self) -> [u8; 32] {
        let res = self.inner.finalize();
        self.inner.reset();

        res.into()
    }

    fn digest(data: &[u8]) -> [u8; 32] {
        blake3::hash(data).into()
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn digest_many(data: &[u8], len: usize, out: &mut [[u8; 32]]) {
        let grouped = match len <= CHUNK_LEN {
            true => out.len() - out.len() % LANES,
            false => 0,
        };

        let (quads, rest) = out.split_at_mut(grouped);

        for (q, outs) in quads.as_chunks_mut::<LANES>().0.iter_mut().enumerate() {
            let at = q * LANES * len;

            // SAFETY:
            // the cfg gate admits only builds with NEON enabled.
            unsafe { hash_lanes(&data[at..at + LANES * len], len, outs) };
        }

        for (i, slot) in rest.iter_mut().enumerate() {
            let at = (grouped + i) * len;
            *slot = blake3::hash(&data[at..at + len]).into();
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon")]
fn hash_lanes(msgs: &[u8], len: usize, out: &mut [[u8; 32]; LANES]) {
    let blocks = len.div_ceil(BLOCK_LEN).max(1);

    let mut cv: [uint32x4_t; 8] = core::array::from_fn(|i| vdupq_n_u32(IV[i]));
    for b in 0..blocks {
        let start = b * BLOCK_LEN;
        let block_len = (len - start).min(BLOCK_LEN);

        let mut flags = 0;

        if b == 0 {
            flags |= CHUNK_START;
        }

        if b + 1 == blocks {
            flags |= CHUNK_END | ROOT;
        }

        let m = load_block(msgs, len, start, block_len);
        cv = compress(&cv, &m, block_len as u32, flags);
    }

    for (lane, digest) in out.iter_mut().enumerate() {
        for (bytes, &v) in digest.as_chunks_mut::<4>().0.iter_mut().zip(&cv) {
            let word = match lane {
                0 => vgetq_lane_u32::<0>(v),
                1 => vgetq_lane_u32::<1>(v),
                2 => vgetq_lane_u32::<2>(v),
                _ => vgetq_lane_u32::<3>(v),
            };

            *bytes = word.to_le_bytes();
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon")]
fn load_block(msgs: &[u8], len: usize, start: usize, block_len: usize) -> [uint32x4_t; 16] {
    let blocks: [[u8; BLOCK_LEN]; LANES] = core::array::from_fn(|lane| {
        let at = lane * len + start;

        let mut block = [0u8; BLOCK_LEN];
        block[..block_len].copy_from_slice(&msgs[at..at + block_len]);

        block
    });

    core::array::from_fn(|j| {
        let [w0, w1, w2, w3] = blocks
            .each_ref()
            .map(|block| u64::from(u32::from_le_bytes(block.as_chunks::<4>().0[j])));

        vcombine_u32(vcreate_u32(w0 | (w1 << 32)), vcreate_u32(w2 | (w3 << 32)))
    })
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon")]
fn compress(
    cv: &[uint32x4_t; 8],
    m: &[uint32x4_t; 16],
    block_len: u32,
    flags: u32,
) -> [uint32x4_t; 8] {
    let mut v = [
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        vdupq_n_u32(IV[0]),
        vdupq_n_u32(IV[1]),
        vdupq_n_u32(IV[2]),
        vdupq_n_u32(IV[3]),
        vdupq_n_u32(0),
        vdupq_n_u32(0),
        vdupq_n_u32(block_len),
        vdupq_n_u32(flags),
    ];

    for s in &MSG_SCHEDULE {
        g(&mut v, [0, 4, 8, 12], m[s[0]], m[s[1]]);
        g(&mut v, [1, 5, 9, 13], m[s[2]], m[s[3]]);
        g(&mut v, [2, 6, 10, 14], m[s[4]], m[s[5]]);
        g(&mut v, [3, 7, 11, 15], m[s[6]], m[s[7]]);
        g(&mut v, [0, 5, 10, 15], m[s[8]], m[s[9]]);
        g(&mut v, [1, 6, 11, 12], m[s[10]], m[s[11]]);
        g(&mut v, [2, 7, 8, 13], m[s[12]], m[s[13]]);
        g(&mut v, [3, 4, 9, 14], m[s[14]], m[s[15]]);
    }

    core::array::from_fn(|i| veorq_u32(v[i], v[i + 8]))
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon")]
#[inline]
fn g(v: &mut [uint32x4_t; 16], [a, b, c, d]: [usize; 4], x: uint32x4_t, y: uint32x4_t) {
    v[a] = vaddq_u32(vaddq_u32(v[a], v[b]), x);
    v[d] = rotr16(veorq_u32(v[d], v[a]));
    v[c] = vaddq_u32(v[c], v[d]);
    v[b] = rotr::<12, 20>(veorq_u32(v[b], v[c]));
    v[a] = vaddq_u32(vaddq_u32(v[a], v[b]), y);
    v[d] = rotr::<8, 24>(veorq_u32(v[d], v[a]));
    v[c] = vaddq_u32(v[c], v[d]);
    v[b] = rotr::<7, 25>(veorq_u32(v[b], v[c]));
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon")]
#[inline]
fn rotr16(x: uint32x4_t) -> uint32x4_t {
    vreinterpretq_u32_u16(vrev32q_u16(vreinterpretq_u16_u32(x)))
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon")]
#[inline]
fn rotr<const R: i32, const L: i32>(x: uint32x4_t) -> uint32x4_t {
    vsriq_n_u32::<R>(vshlq_n_u32::<L>(x), x)
}
