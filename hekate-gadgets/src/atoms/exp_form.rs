// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Integers as exponents of a primitive
//! element of GF(2^128)*.
//!
//! `x` travels as `G^x`: integer addition is a
//! field multiplication and `x · 2^j` is `j`
//! Frobenius squarings. Every integer an AIR
//! feeds through these chains must stay below
//! the group order `2^128 - 1`; past it the
//! exponent wraps and the relation is void.

use hekate_core::errors::{self, Error};
use hekate_math::{BinaryFieldExtras, Flat, HardwareField, TowerField};
use hekate_program::constraint::builder::{ConstraintSystem, Expr};

pub const GROUP_BITS: usize = 128;

const GROUP_ORDER: u128 = u128::MAX;
const GROUP_ORDER_PRIMES: [u128; 9] = [3, 5, 17, 257, 641, 65537, 274177, 6700417, 67280421310721];

/// Half of GF(2^128)* is primitive.
const SEARCH_WINDOW: u128 = 4096;

/// `pow2[i] = G^(2^i)` for a primitive `G`.
#[derive(Clone, Debug)]
pub struct ExpBasis<F: HardwareField> {
    generator: F,
    pow2: [Flat<F>; GROUP_BITS],
}

impl<F: TowerField + BinaryFieldExtras + HardwareField> ExpBasis<F> {
    /// # Errors
    /// `F` is not GF(2^128), or no primitive
    /// element was found in the search window.
    pub fn new() -> errors::Result<Self> {
        if F::BITS != GROUP_BITS {
            return Err(Error::Protocol {
                protocol: "exp_form",
                message: "exponent form needs GF(2^128); smaller towers have a \
                          group order below the column sums a bignum AIR produces",
            });
        }

        let generator = primitive_element::<F>().ok_or(Error::Protocol {
            protocol: "exp_form",
            message: "no primitive element of GF(2^128)* in the search window",
        })?;

        let mut pow2 = [Flat::from_raw(F::ZERO); GROUP_BITS];
        let mut acc = generator;

        for slot in pow2.iter_mut() {
            *slot = acc.to_hardware();
            acc = acc.square();
        }

        Ok(Self { generator, pow2 })
    }

    pub fn generator(&self) -> F {
        self.generator
    }

    pub fn pow2(&self) -> &[Flat<F>; GROUP_BITS] {
        &self.pow2
    }

    pub fn tower_pow2(&self) -> [F; GROUP_BITS] {
        core::array::from_fn(|i| self.pow2[i].to_tower())
    }

    /// `out[t] = G^(x mod 2^(2t+2))`, branch-free in `x`.
    pub fn chain2(&self, x: u128, out: &mut [Flat<F>]) {
        let one = Flat::from_raw(F::ONE);
        let mut acc = one;

        for (t, slot) in out.iter_mut().enumerate() {
            for i in [2 * t, 2 * t + 1] {
                let bit = Flat::from_raw(F::from(((x >> i) & 1) as u8));
                acc *= one + bit * (self.pow2[i] + one);
            }

            *slot = acc;
        }
    }
}

/// Two bits per step:
/// `cols[t] = cols[t-1] · Π_{i=2t,2t+1} (1 + bits[i] · (G^(2^i) + 1))`
/// with `cols[-1] = 1`; `cols.last() = G^x`, `x = Σ bits[i] 2^i`.
///
/// # Panics
/// `bits.len() != 2 · cols.len()`, or
/// `bits.len() != pow2.len()`.
pub fn constrain_exp_chain2<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    bits: &[Expr<'a, F>],
    cols: &[Expr<'a, F>],
    pow2: &[F],
) {
    assert_eq!(
        bits.len(),
        2 * cols.len(),
        "exp chain: {} bits for {} columns",
        bits.len(),
        cols.len()
    );
    assert_eq!(
        bits.len(),
        pow2.len(),
        "exp chain: {} bits against {} powers",
        bits.len(),
        pow2.len()
    );

    let one = cs.constant(F::ONE);

    let mut prev = one;
    for (t, &col) in cols.iter().enumerate() {
        let step0 = cs.constant(pow2[2 * t] + F::ONE);
        let step1 = cs.constant(pow2[2 * t + 1] + F::ONE);

        cs.constrain(col + prev * (one + bits[2 * t] * step0) * (one + bits[2 * t + 1] * step1));

        prev = col;
    }
}

pub fn constrain_squarings<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    base: Expr<'a, F>,
    cols: &[Expr<'a, F>],
) {
    let mut prev = base;
    for &col in cols {
        cs.constrain(col + prev * prev);

        prev = col;
    }
}

pub fn constrain_fourth_powers<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    base: Expr<'a, F>,
    cols: &[Expr<'a, F>],
) {
    let mut prev = base;
    for &col in cols {
        let sq = prev * prev;

        cs.constrain(col + sq * sq);

        prev = col;
    }
}

fn pow<F: TowerField + BinaryFieldExtras>(base: F, mut exp: u128) -> F {
    let mut acc = F::ONE;
    let mut sq = base;

    while exp != 0 {
        if exp & 1 == 1 {
            acc *= sq;
        }

        sq = sq.square();
        exp >>= 1;
    }

    acc
}

/// Candidates start above 2^64: lower tower coordinates
/// lie in GF(2^64), whose element orders divide 2^64 - 1.
fn primitive_element<F: TowerField + BinaryFieldExtras>() -> Option<F> {
    let base = 1u128 << 64;

    (base..base + SEARCH_WINDOW).map(F::from).find(|&g| {
        GROUP_ORDER_PRIMES
            .iter()
            .all(|&p| pow(g, GROUP_ORDER / p) != F::ONE)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hekate_math::{Block64, Block128};

    type F = Block128;

    #[test]
    fn generator_has_full_order() {
        let basis = ExpBasis::<F>::new().unwrap();

        for &p in &GROUP_ORDER_PRIMES {
            assert_ne!(pow(basis.generator(), GROUP_ORDER / p), F::ONE);
        }

        assert_eq!(pow(basis.generator(), GROUP_ORDER), F::ONE);
    }

    #[test]
    fn narrow_towers_are_rejected() {
        assert!(ExpBasis::<Block64>::new().is_err());
    }

    #[test]
    fn squarings_are_doublings_of_the_exponent() {
        let basis = ExpBasis::<F>::new().unwrap();
        for i in 0..GROUP_BITS {
            assert_eq!(
                basis.pow2()[i].to_tower(),
                pow(basis.generator(), 1u128 << i)
            );
        }

        assert_eq!(basis.tower_pow2()[..], basis.pow2().map(Flat::to_tower)[..]);
    }

    #[test]
    fn chain_reaches_the_exponent() {
        let basis = ExpBasis::<F>::new().unwrap();

        let mut out = [Flat::from_raw(F::ZERO); 16];
        for x in [0u128, 1, 0xffff_ffff, 0x9e37_79b9_7f4a_7c15] {
            basis.chain2(x, &mut out);

            for (t, &value) in out.iter().enumerate() {
                let masked = x & ((1u128 << (2 * t + 2)) - 1);
                assert_eq!(
                    value.to_tower(),
                    pow(basis.generator(), masked),
                    "x = {x:#x}, t = {t}"
                );
            }
        }
    }

    #[test]
    fn addition_is_multiplication() {
        let basis = ExpBasis::<F>::new().unwrap();
        let g = basis.generator();

        let a = 0x1234_5678_9abc_def0u128;
        let b = 0x0fed_cba9_8765_4321u128;

        assert_eq!(pow(g, a) * pow(g, b), pow(g, a + b));
    }

    #[test]
    fn frobenius_shifts_exponent() {
        let basis = ExpBasis::<F>::new().unwrap();
        let g = basis.generator();

        let x = 0xdead_beef_u128;

        let mut acc = pow(g, x);
        for j in 1..=32 {
            acc = acc.square();
            assert_eq!(acc, pow(g, x << j), "j = {j}");
        }
    }
}
