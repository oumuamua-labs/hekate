// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! GF(2^256) for the LogUp bus:
//! its challenges, fractions and claimed sums are GF(2^256)
//! elements whether the sumcheck runs over GF(2^128) or GF(2^256).

use core::ops::{Add, AddAssign};
use hekate_core::errors;
use hekate_math::{Block256, Flat, HardwareField, TowerField};

/// GF(2^128) limbs in one GF(2^256) bus value.
pub const BUS_LIMBS: usize = 2;

/// The bus field in the tower basis: the bus draws
/// come from it and their gate reads its width.
pub type BusElement = Block256;

/// A GF(2^256) bus value, `lo + hi·y` over flat GF(2^128) limbs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BusValue<F> {
    pub lo: Flat<F>,
    pub hi: Flat<F>,
}

/// GF(2^256) arithmetic over flat GF(2^128) limbs, `y² = y + τ`.
#[derive(Clone, Copy, Debug)]
pub struct BusField<F> {
    tau: Flat<F>,
}

impl<F: TowerField + HardwareField> BusValue<F> {
    /// Returns the zero bus value.
    pub fn zero() -> Self {
        Self::from_base(Flat::from_raw(F::ZERO))
    }

    /// Returns the unit bus value.
    pub fn one() -> Self {
        Self::from_base(Flat::from_raw(F::ONE))
    }

    /// Embeds a GF(2^128) element as the low limb.
    pub fn from_base(x: Flat<F>) -> Self {
        Self {
            lo: x,
            hi: Flat::from_raw(F::ZERO),
        }
    }

    /// Returns the limbs as `[lo, hi]`.
    pub fn limbs(self) -> [Flat<F>; BUS_LIMBS] {
        [self.lo, self.hi]
    }

    /// Multiplies both limbs by a GF(2^128) element.
    pub fn mul_base(self, x: Flat<F>) -> Self {
        Self {
            lo: self.lo * x,
            hi: self.hi * x,
        }
    }

    /// Tests whether the value is zero.
    pub fn is_zero(self) -> bool {
        self == Self::zero()
    }
}

impl<F: TowerField + HardwareField> Add for BusValue<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            lo: self.lo + rhs.lo,
            hi: self.hi + rhs.hi,
        }
    }
}

impl<F: TowerField + HardwareField> AddAssign for BusValue<F> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F: TowerField + HardwareField> BusField<F> {
    /// Builds the bus field over `F`;
    /// errors unless `F` is GF(2^128).
    pub fn new() -> errors::Result<Self> {
        if F::BITS != 128 {
            return Err(errors::Error::Protocol {
                protocol: "logup",
                message: "the bus field is GF(2^256) on two GF(2^128) limbs",
            });
        }

        Ok(Self {
            tau: F::EXTENSION_TAU.to_hardware(),
        })
    }

    /// Returns `τ` in the flat basis.
    pub fn tau(&self) -> Flat<F> {
        self.tau
    }

    /// Converts a tower `BusElement` into flat limbs.
    pub fn import(&self, x: BusElement) -> BusValue<F> {
        let (lo, hi) = x.split();

        BusValue {
            lo: F::from(lo.0).to_hardware(),
            hi: F::from(hi.0).to_hardware(),
        }
    }

    /// Multiplies two bus values.
    pub fn mul(&self, a: BusValue<F>, b: BusValue<F>) -> BusValue<F> {
        let v0 = a.lo * b.lo;
        let v1 = a.hi * b.hi;
        let vs = (a.lo + a.hi) * (b.lo + b.hi);

        BusValue {
            lo: v0 + v1 * self.tau,
            hi: v0 + vs,
        }
    }

    /// Evaluates `eq(r, x)` at a GF(2^256) point `r` and a GF(2^128) point `x`.
    pub fn eq_at(&self, r: &[BusValue<F>], x: &[Flat<F>]) -> BusValue<F> {
        let one = Flat::from_raw(F::ONE);

        r.iter().zip(x).fold(BusValue::one(), |acc, (&r_i, &x_i)| {
            self.mul(acc, r_i + BusValue::from_base(one + x_i))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hekate_math::{Block64, Block128};

    fn block256(lo: u128, hi: u128) -> Block256 {
        Block256::new(Block128(lo), Block128(hi))
    }

    #[test]
    fn limb_arithmetic_matches_flat_block256() {
        let field = BusField::<Block128>::new().unwrap();
        let pairs = [
            (block256(3, 0), block256(5, 0)),
            (block256(0, 1), block256(0, 1)),
            (
                block256(0x1234_5678_9abc_def0, 0x0fed_cba9_8765_4321),
                block256(0xdead_beef, 0xcafe_babe_f00d),
            ),
        ];

        for (a, b) in pairs {
            let expected = (a.to_hardware() * b.to_hardware()).to_tower();
            let got = field.mul(field.import(a), field.import(b));
            let (lo, hi) = expected.split();

            assert_eq!(got.lo, lo.to_hardware());
            assert_eq!(got.hi, hi.to_hardware());
        }
    }

    #[test]
    fn eq_at_sums_to_one_over_cube() {
        let field = BusField::<Block128>::new().unwrap();
        let r = [field.import(block256(7, 9)), field.import(block256(11, 13))];

        let zero = Flat::from_raw(Block128::ZERO);
        let one = Flat::from_raw(Block128::ONE);

        let total = [[zero, zero], [one, zero], [zero, one], [one, one]]
            .iter()
            .fold(BusValue::zero(), |acc, x| acc + field.eq_at(&r, x));

        assert_eq!(total, BusValue::one());
    }

    #[test]
    fn non_gf128_field_rejected() {
        assert!(BusField::<Block64>::new().is_err());
    }
}
