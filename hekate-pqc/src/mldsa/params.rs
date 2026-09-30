// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::errors::{self, Error};
use subtle::{
    Choice, ConditionallySelectable, ConstantTimeEq, ConstantTimeGreater, ConstantTimeLess,
};

use crate::wiring::N;

pub const Q: u32 = 8_380_417;

/// One ML-DSA parameter set, FIPS 204 Table 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MlDsaParams {
    k: usize,
    l: usize,
    eta: u32,
    tau: usize,
    beta: u32,
    gamma1: u32,
    gamma2: u32,
    omega: usize,
    lambda: usize,
}

impl MlDsaParams {
    pub const ML_DSA_44: Self = Self {
        k: 4,
        l: 4,
        eta: 2,
        tau: 39,
        beta: 78,
        gamma1: 1 << 17,
        gamma2: (Q - 1) / 88,
        omega: 80,
        lambda: 128,
    };

    pub const ML_DSA_65: Self = Self {
        k: 6,
        l: 5,
        eta: 4,
        tau: 49,
        beta: 196,
        gamma1: 1 << 19,
        gamma2: (Q - 1) / 32,
        omega: 55,
        lambda: 192,
    };

    pub const ML_DSA_87: Self = Self {
        k: 8,
        l: 7,
        eta: 2,
        tau: 60,
        beta: 120,
        gamma1: 1 << 19,
        gamma2: (Q - 1) / 32,
        omega: 75,
        lambda: 256,
    };

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn l(&self) -> usize {
        self.l
    }

    pub fn eta(&self) -> u32 {
        self.eta
    }

    pub fn tau(&self) -> usize {
        self.tau
    }

    pub fn beta(&self) -> u32 {
        self.beta
    }

    pub fn gamma1(&self) -> u32 {
        self.gamma1
    }

    pub fn gamma2(&self) -> u32 {
        self.gamma2
    }

    pub fn omega(&self) -> usize {
        self.omega
    }

    pub fn lambda(&self) -> usize {
        self.lambda
    }

    /// Encoded public key length in bytes, FIPS 204 Table 2.
    pub fn pk_bytes(&self) -> usize {
        32 + 320 * self.k
    }

    /// Encoded signature length in bytes, FIPS 204 Table 2.
    pub fn sig_bytes(&self) -> usize {
        self.lambda / 4 + 32 * self.l * self.z_bits() + self.omega + self.k
    }

    /// Bits per z coefficient in sigEncode: 1 + bitlen(γ1 − 1).
    pub fn z_bits(&self) -> usize {
        33 - (self.gamma1 - 1).leading_zeros() as usize
    }

    /// m = (q − 1)/(2γ2), the count of HighBits values.
    pub fn high_bits_range(&self) -> u32 {
        (Q - 1) / (2 * self.gamma2)
    }

    /// Bits per coefficient in w1Encode, FIPS 204 Algorithm 28.
    pub fn w1_bits(&self) -> usize {
        32 - (self.high_bits_range() - 1).leading_zeros() as usize
    }

    /// (r1, r0) with r ≡ r1·2γ2 + r0 mod q for r below q,
    /// FIPS 204 Algorithm 36, with no branch or division on r.
    pub fn decompose(&self, r: u32) -> (u32, i32) {
        let m = self.high_bits_range();
        let (quot, rem) = self.split(r);

        let high = rem.ct_gt(&self.gamma2);
        let r0 = rem as i32 - i32::conditional_select(&0, &(2 * self.gamma2 as i32), high);
        let r1 = quot + u32::conditional_select(&0, &1, high);

        let corner = r1.ct_eq(&m);

        (
            u32::conditional_select(&r1, &0, corner),
            r0 - i32::conditional_select(&0, &1, corner),
        )
    }

    /// The HighBits of r moved by hint h, FIPS 204 Algorithm 40,
    /// for r below q, with no branch on h or r.
    pub fn use_hint(&self, h: bool, r: u32) -> u32 {
        let m = self.high_bits_range();
        let (r1, r0) = self.decompose(r);

        let positive = Choice::from((r0.wrapping_neg() as u32 >> 31) as u8);
        let wrapped = |v: u32| u32::conditional_select(&v, &v.wrapping_sub(m), v.ct_gt(&(m - 1)));

        let up = wrapped(r1 + 1);
        let down = wrapped(r1 + m - 1);

        let moved = u32::conditional_select(&down, &up, positive);

        u32::conditional_select(&r1, &moved, Choice::from(h as u8))
    }

    /// ⌊w / 2γ2⌋ and w mod 2γ2 for w below q,
    /// by m comparisons and no division.
    pub(crate) fn split(&self, w: u32) -> (u32, u32) {
        let two_g2 = 2 * self.gamma2;

        let quot = (1..=self.high_bits_range())
            .map(|i| (!w.ct_lt(&(i * two_g2))).unwrap_u8() as u32)
            .sum::<u32>();

        (quot, w - quot * two_g2)
    }

    /// w1Encode of one polynomial, FIPS 204 Algorithm 28,
    /// into `lanes` as little-endian 64-bit lanes.
    pub fn w1_lanes(&self, w1: &[u32; N], lanes: &mut [u64]) -> errors::Result<()> {
        let b = self.w1_bits();

        if lanes.len() != N * b / 64 {
            return Err(Error::Protocol {
                protocol: "mldsa_params",
                message: "w1Encode fills 256·b/64 lanes",
            });
        }

        lanes.fill(0);

        for (j, &c) in w1.iter().enumerate() {
            for t in 0..b {
                let pos = j * b + t;
                lanes[pos / 64] |= (((c >> t) & 1) as u64) << (pos % 64);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn table_one_derived_values() {
        assert_eq!(MlDsaParams::ML_DSA_44.gamma2(), 95_232);
        assert_eq!(MlDsaParams::ML_DSA_65.gamma2(), 261_888);
        assert_eq!(MlDsaParams::ML_DSA_44.high_bits_range(), 44);
        assert_eq!(MlDsaParams::ML_DSA_87.high_bits_range(), 16);
        assert_eq!(MlDsaParams::ML_DSA_44.w1_bits(), 6);
        assert_eq!(MlDsaParams::ML_DSA_65.w1_bits(), 4);

        for p in [
            MlDsaParams::ML_DSA_44,
            MlDsaParams::ML_DSA_65,
            MlDsaParams::ML_DSA_87,
        ] {
            assert_eq!(p.beta(), p.tau() as u32 * p.eta());
        }
    }

    #[test]
    fn decompose_corners() {
        let p = MlDsaParams::ML_DSA_44;
        let g2 = p.gamma2();

        assert_eq!(p.decompose(0), (0, 0));
        assert_eq!(p.decompose(g2), (0, g2 as i32));
        assert_eq!(p.decompose(g2 + 1), (1, -(g2 as i32) + 1));
        assert_eq!(p.decompose(Q - 1), (0, -1));
        assert_eq!(p.decompose(Q - 1 - g2 + 1), (0, -(g2 as i32)));
        assert_eq!(p.decompose(Q - 1 - g2), (43, g2 as i32));
    }

    #[test]
    fn decompose_recomposes_modulo_q() {
        for p in [MlDsaParams::ML_DSA_44, MlDsaParams::ML_DSA_65] {
            let two_g2 = 2 * p.gamma2() as i64;

            for r in (0..Q).step_by(997).chain(Q - 3000..Q) {
                let (r1, r0) = p.decompose(r);

                assert!(r1 < p.high_bits_range());
                assert!(r0.unsigned_abs() <= p.gamma2());
                assert_eq!(
                    (r1 as i64 * two_g2 + r0 as i64).rem_euclid(Q as i64),
                    r as i64
                );
            }
        }
    }

    #[test]
    fn use_hint_moves_one_step_toward_sign_of_r0() {
        let p = MlDsaParams::ML_DSA_65;
        let m = p.high_bits_range();

        assert_eq!(p.use_hint(true, 0), m - 1);
        assert_eq!(p.use_hint(true, 1), 1);
        assert_eq!(p.use_hint(true, Q - 1), m - 1);
        assert_eq!(p.use_hint(false, Q - 1), 0);
    }

    #[test]
    fn w1_lanes_pack_little_endian() {
        let p = MlDsaParams::ML_DSA_44;

        let mut w1 = [0u32; N];
        w1[0] = 0b10_1010;
        w1[10] = 0b11_0001;

        let mut lanes = vec![0u64; 24];
        p.w1_lanes(&w1, &mut lanes).unwrap();

        assert!(p.w1_lanes(&w1, &mut [0; 23]).is_err());
        assert_eq!(lanes[0] & 0x3f, 0b10_1010);
        assert_eq!(lanes[0] >> 60, 0b0001);
        assert_eq!(lanes[1] & 0b11, 0b11);
    }
}
