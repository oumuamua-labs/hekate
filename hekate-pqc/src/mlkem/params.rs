// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

pub const Q: u32 = 3329;

/// ⌈2^36/q⌉: `(v·BARRETT) >> 36` equals ⌊v/q⌋ for
/// v below 2^24, with no division instruction on v.
const BARRETT: u64 = 20_642_679;

/// One ML-KEM parameter set, FIPS 203 Table 2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MlKemParams {
    k: usize,
    eta1: u32,
    eta2: u32,
    du: u32,
    dv: u32,
}

impl MlKemParams {
    pub const ML_KEM_512: Self = Self {
        k: 2,
        eta1: 3,
        eta2: 2,
        du: 10,
        dv: 4,
    };

    pub const ML_KEM_768: Self = Self {
        k: 3,
        eta1: 2,
        eta2: 2,
        du: 10,
        dv: 4,
    };

    pub const ML_KEM_1024: Self = Self {
        k: 4,
        eta1: 2,
        eta2: 2,
        du: 11,
        dv: 5,
    };

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn eta1(&self) -> u32 {
        self.eta1
    }

    pub fn eta2(&self) -> u32 {
        self.eta2
    }

    pub fn du(&self) -> u32 {
        self.du
    }

    pub fn dv(&self) -> u32 {
        self.dv
    }
}

pub(crate) fn compress(d: u32, x: u32) -> u32 {
    let scaled = ((x as u64) << d) + (Q as u64 - 1) / 2;

    (div_q(scaled) & ((1 << d) - 1)) as u32
}

pub(crate) fn decompress(d: u32, y: u32) -> u32 {
    ((Q as u64 * y as u64 + (1 << (d - 1))) >> d) as u32
}

pub(crate) fn div_q(v: u64) -> u64 {
    (v * BARRETT) >> 36
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_rounds_half_up_modulo_two_to_the_d() {
        assert_eq!(compress(1, 832), 0);
        assert_eq!(compress(1, 833), 1);
        assert_eq!(compress(1, 2496), 1);
        assert_eq!(compress(1, 2497), 0);
        assert_eq!(compress(10, Q - 1), 0);
        assert_eq!(compress(4, 0), 0);
    }

    #[test]
    fn compress_matches_exact_division() {
        for d in 1..=11 {
            for x in 0..Q {
                let scaled = ((x as u64) << d) + (Q as u64 - 1) / 2;
                assert_eq!(compress(d, x), ((scaled / Q as u64) % (1 << d)) as u32);
            }
        }
    }

    #[test]
    fn decompress_then_compress_is_identity() {
        for d in [1, 4, 5, 10, 11] {
            for y in 0..1 << d {
                let x = decompress(d, y);

                assert!(x < Q);
                assert_eq!(compress(d, x), y);
            }
        }
    }
}
