// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_keccak::KeccakSpongeNative;
use subtle::{Choice, ConstantTimeLess};
use zeroize::Zeroizing;

use super::{Q, compress, decompress};
use crate::ctrl::RATE;
use crate::ntt::NttParams;
use crate::wiring::N;

pub(super) const SHA3: u8 = 0x06;
pub(super) const SHAKE: u8 = 0x1f;
pub(super) const G_RATE: usize = 9;

pub(super) type Coeffs = [u32; N];

pub(super) fn add(f: &Coeffs, g: &Coeffs) -> Coeffs {
    core::array::from_fn(|i| NttParams::ML_KEM.add(f[i], g[i]))
}

pub(super) fn sub(f: &Coeffs, g: &Coeffs) -> Coeffs {
    core::array::from_fn(|i| NttParams::ML_KEM.sub(f[i], g[i]))
}

pub(super) fn mac(acc: &mut Coeffs, f: &Coeffs, g: &Coeffs) {
    let params = NttParams::ML_KEM;

    for (p, &gamma) in params.gammas().iter().enumerate() {
        let (a0, a1, b0, b1) = (f[2 * p], f[2 * p + 1], g[2 * p], g[2 * p + 1]);

        let c0 = params.add(params.mul(a0, b0), params.mul(params.mul(a1, b1), gamma));
        let c1 = params.add(params.mul(a0, b1), params.mul(a1, b0));

        acc[2 * p] = params.add(acc[2 * p], c0);
        acc[2 * p + 1] = params.add(acc[2 * p + 1], c1);
    }
}

pub(super) fn compress_poly(d: u32, f: &Coeffs) -> Coeffs {
    core::array::from_fn(|i| compress(d, f[i]))
}

pub(super) fn decompress_poly(d: u32, f: &Coeffs) -> Coeffs {
    core::array::from_fn(|i| decompress(d, f[i]))
}

pub(super) fn byte_encode(d: u32, f: &Coeffs, out: &mut [u8]) {
    out.fill(0);

    for (i, &x) in f.iter().enumerate() {
        for b in 0..d as usize {
            let bit = d as usize * i + b;

            out[bit / 8] |= (((x >> b) & 1) as u8) << (bit % 8);
        }
    }
}

pub(super) fn byte_decode(d: u32, bytes: &[u8]) -> Coeffs {
    core::array::from_fn(|i| {
        let x = field(bytes, d as usize, i);

        match d {
            12 => NttParams::ML_KEM.divmod(x as u64).1,
            _ => x,
        }
    })
}

pub(super) fn canonical12(bytes: &[u8]) -> Choice {
    (0..bytes.len() * 8 / 12).fold(Choice::from(1), |ok, i| ok & field(bytes, 12, i).ct_lt(&Q))
}

pub(super) fn g(input: &[u8]) -> Zeroizing<[u8; 64]> {
    let mut out = Zeroizing::new([0u8; 64]);
    sponge(input, G_RATE, SHA3, &mut out[..]);

    out
}

pub(super) fn h(input: &[u8]) -> Zeroizing<[u8; 32]> {
    let mut out = Zeroizing::new([0u8; 32]);
    sponge(input, RATE, SHA3, &mut out[..]);

    out
}

pub(super) fn j(input: &[u8]) -> Zeroizing<[u8; 32]> {
    let mut out = Zeroizing::new([0u8; 32]);
    sponge(input, RATE, SHAKE, &mut out[..]);

    out
}

fn field(bytes: &[u8], width: usize, i: usize) -> u32 {
    (0..width).fold(0, |acc, b| {
        let bit = width * i + b;

        acc | (((bytes[bit / 8] >> (bit % 8)) & 1) as u32) << b
    })
}

fn sponge(input: &[u8], rate: usize, domain: u8, out: &mut [u8]) {
    let mut sponge = KeccakSpongeNative::new();
    sponge.absorb(input, 8 * rate, domain);

    out.copy_from_slice(&Zeroizing::new(sponge.squeeze(out.len(), 8 * rate)));
}
