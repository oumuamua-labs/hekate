// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use crate::config::{Config, Error, LOG2_FRAC_BITS, OUTER_RATE_LOG2, OUTER_ROWS, log2_ratio_fixed};
use crate::errors;
use alloc::vec::Vec;
use hekate_crypto::Hasher;
use hekate_math::{Flat, HardwareField, TowerField};
use zeroize::Zeroize;
#[cfg(feature = "secure-memory")]
use zeroize::ZeroizeOnDrop;

/// Interleaved, then a low/high pair per
/// product-code test: zero-sum linear,
/// zero-vector quadratic (Ligero 2017 Fig 8-11).
pub const OUTER_MASK_ROWS: usize = 5;

/// Wires `x`, `y`, `z` of one Hadamard row.
pub const OUTER_WIRES_PER_MUL: usize = 3;

/// Additive-FFT domains are F_2-subspaces;
/// the cap is an exponent.
const MAX_OUTER_DOMAIN_LOG2: u32 = 24;

/// Ligero notation:
/// `message_len` = l, `code_len` = k,
/// `domain_len` = n, `queries` = t.
///
/// Both sides derive this from the statement
/// size alone and never read it from the proof:
/// a prover-chosen geometry is a free parameter
/// and would void the query bound.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OuterGeometry {
    pub message_len: usize,

    /// `k = l + t + 1`. The surplus over `l` is uniform,
    /// which is what leaves any `t` opened columns of
    /// a row independent of its message (Ligero Lemma 4.15).
    pub code_len: usize,

    pub domain_len: usize,
    pub queries: usize,
}

#[derive(Zeroize)]
#[cfg_attr(feature = "secure-memory", derive(ZeroizeOnDrop))]
pub struct Pad<F: TowerField> {
    values: Vec<Flat<F>>,

    #[zeroize(skip)]
    next: usize,
}

impl<F: TowerField + From<u128>> Pad<F> {
    /// Derives `len` entries from `seed` in counter mode.
    pub fn derive<H: Hasher>(seed: &[u8; 32], len: usize) -> Self {
        let mut values = Vec::with_capacity(len);
        for i in 0..len {
            let mut hasher = H::new();
            hasher.update(seed);
            hasher.update(&(i as u64).to_le_bytes());

            let digest = hasher.finalize();

            let mut bytes = [0u8; 16];
            bytes.copy_from_slice(&digest[..16]);

            values.push(Flat::from_raw(F::from(u128::from_le_bytes(bytes))));
        }

        Self { values, next: 0 }
    }

    pub fn values(&self) -> &[Flat<F>] {
        &self.values
    }

    pub fn get(&self, index: usize) -> Option<Flat<F>> {
        self.values.get(index).copied()
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn take(&mut self) -> Option<Flat<F>> {
        let entry = self.values.get(self.next).copied();
        if entry.is_some() {
            self.next += 1;
        }

        entry
    }

    pub fn consumed(&self) -> usize {
        self.next
    }
}

pub enum Masking<'a, F: TowerField> {
    Off,
    On(&'a mut Pad<F>),
}

impl<F: TowerField + HardwareField> Masking<'_, F> {
    pub fn mask(&mut self, value: Flat<F>) -> errors::Result<Flat<F>> {
        match self {
            Self::Off => Ok(value),
            Self::On(pad) => match pad.take() {
                None => Err(errors::Error::Protocol {
                    protocol: "outer",
                    message: "one-time pad exhausted",
                }),
                Some(entry) => Ok(value + entry),
            },
        }
    }

    pub fn is_on(&self) -> bool {
        matches!(self, Self::On(_))
    }

    pub fn consumed(&self) -> usize {
        match self {
            Self::Off => 0,
            Self::On(pad) => pad.consumed(),
        }
    }

    pub fn entry(&self, index: usize) -> Option<Flat<F>> {
        match self {
            Self::Off => None,
            Self::On(pad) => pad.get(index),
        }
    }
}

impl OuterGeometry {
    pub fn zk_padding(&self) -> usize {
        self.code_len - self.message_len
    }
}

impl Config {
    pub fn outer_geom(
        &self,
        masked_scalars: usize,
        mul_wires: usize,
        field_bits: usize,
    ) -> errors::Result<OuterGeometry> {
        if masked_scalars == 0 {
            return Err(Error::EmptyOuterStatement.into());
        }

        if self.outer_queries == 0 {
            return Err(Error::ZeroOuterQueries.into());
        }

        let aux_len = OUTER_WIRES_PER_MUL * mul_wires;
        let seed_len = aux_len.div_ceil(OUTER_ROWS).max(1);
        let seed_code = seed_len + self.outer_queries + 1;

        let domain_len = seed_code
            .checked_mul(1 << OUTER_RATE_LOG2)
            .ok_or(Error::OuterDomainTooLarge {
                code_len: seed_code,
                max_log2: MAX_OUTER_DOMAIN_LOG2,
            })?
            .next_power_of_two();

        if domain_len.ilog2() > MAX_OUTER_DOMAIN_LOG2 {
            return Err(Error::OuterDomainTooLarge {
                code_len: seed_code,
                max_log2: MAX_OUTER_DOMAIN_LOG2,
            }
            .into());
        }

        let code_len = domain_len >> OUTER_RATE_LOG2;
        let message_len = code_len - self.outer_queries - 1;

        let geom = OuterGeometry {
            message_len,
            code_len,
            domain_len,
            queries: self.outer_queries,
        };

        self.check_outer_security(field_bits, &geom)?;

        Ok(geom)
    }

    /// Query term only; pair with [`Config::outer_field_term_bits`].
    /// Ligero journal Lemmas 4.6 and 4.10: at `e = (n - 2k) / 2`
    /// interleaved `(n - e) / n` meets quadratic `(e + 2k) / n`.
    pub fn outer_security_bits(&self, field_bits: usize, geom: &OuterGeometry) -> usize {
        let n = geom.domain_len as u128;
        let k = geom.code_len as u128;

        let per_query = log2_ratio_fixed(2 * n, n + 2 * k);
        let bits = ((geom.queries as u128 * per_query) >> LOG2_FRAC_BITS) as usize;

        bits.min(field_bits)
    }

    /// Ligero journal Lemma 4.6's additive term `n / |F|`, in bits.
    pub fn outer_field_term_bits(&self, field_bits: usize, geom: &OuterGeometry) -> usize {
        field_bits.saturating_sub(geom.domain_len.next_power_of_two().ilog2() as usize)
    }

    pub fn check_outer_security(
        &self,
        field_bits: usize,
        geom: &OuterGeometry,
    ) -> errors::Result<()> {
        if self.min_security_bits == 0 {
            return Ok(());
        }

        let estimated_bits = self
            .outer_security_bits(field_bits, geom)
            .min(self.outer_field_term_bits(field_bits, geom));

        if estimated_bits < self.min_security_bits {
            return Err(Error::SecurityTooLow {
                estimated_bits,
                min_bits: self.min_security_bits,
            }
            .into());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MIN_PRODUCTION_BITS;

    const FIELD_BITS: usize = 128;

    #[test]
    fn prod_geometry_saturates_field_width() {
        let cfg = Config::prod();
        let geom = cfg.outer_geom(12_000, 20_000, FIELD_BITS).unwrap();

        assert!(cfg.outer_security_bits(FIELD_BITS, &geom) >= FIELD_BITS);
    }

    #[test]
    fn zk_padding_covers_every_opened_column() {
        let cfg = Config::prod();

        for (scalars, wires) in [(1, 0), (4_075, 2_384), (12_000, 20_000), (60_000, 90_000)] {
            let geom = cfg.outer_geom(scalars, wires, FIELD_BITS).unwrap();

            assert!(geom.zk_padding() > geom.queries);
        }
    }

    #[test]
    fn domain_is_subspace_at_target_rate() {
        let cfg = Config::prod();
        let geom = cfg.outer_geom(12_000, 20_000, FIELD_BITS).unwrap();

        assert!(geom.domain_len.is_power_of_two());
        assert_eq!(geom.code_len << OUTER_RATE_LOG2, geom.domain_len);
    }

    /// `AdditiveFft` takes only power-of-two sizes
    /// and `1 <= log_k < log_n <= min(F::BITS, 63)`.
    #[test]
    fn geometry_fits_additive_fft_domain_rules() {
        let cfg = Config::prod();

        for (scalars, wires) in [(1, 0), (4_075, 2_384), (12_000, 20_000), (60_000, 90_000)] {
            let geom = cfg.outer_geom(scalars, wires, FIELD_BITS).unwrap();

            assert!(geom.code_len.is_power_of_two());
            assert!(geom.domain_len.is_power_of_two());
            assert!(geom.code_len.ilog2() >= 1);
            assert!(geom.code_len.ilog2() < geom.domain_len.ilog2());
            assert!(geom.domain_len.ilog2() as usize <= FIELD_BITS.min(63));
        }
    }

    #[test]
    fn geometry_is_deterministic_function_of_statement() {
        let cfg = Config::prod();

        assert_eq!(
            cfg.outer_geom(12_000, 20_000, FIELD_BITS).unwrap(),
            cfg.outer_geom(12_000, 20_000, FIELD_BITS).unwrap()
        );
    }

    #[test]
    fn empty_statement_is_rejected() {
        assert!(Config::prod().outer_geom(0, 20_000, FIELD_BITS).is_err());
    }

    #[test]
    fn zero_outer_queries_is_rejected() {
        let cfg = Config {
            outer_queries: 0,
            ..Config::dev()
        };

        assert!(cfg.outer_geom(64, 32, FIELD_BITS).is_err());
    }

    #[test]
    fn oversized_statement_is_rejected() {
        let cfg = Config::prod();

        assert!(cfg.outer_geom(1 << 40, 1 << 40, FIELD_BITS).is_err());
    }

    #[test]
    fn query_count_matches_ligero_bound_at_rate_one_sixteenth() {
        let cfg = Config::prod();
        let geom = cfg.outer_geom(12_000, 20_000, FIELD_BITS).unwrap();

        let thinner = OuterGeometry {
            queries: 154,
            ..geom
        };

        assert!(cfg.outer_security_bits(FIELD_BITS, &thinner) < 128);
        assert!(cfg.outer_security_bits(FIELD_BITS, &geom) >= 128);
    }

    /// `128 - log2(2^18) = MIN_PRODUCTION_BITS` is the binding term;
    /// one more mul wire doubles the domain and loses a bit.
    #[test]
    fn field_term_caps_outer_statement() {
        let cfg = Config::prod();

        let largest = cfg.outer_geom(12_000, 173_098, FIELD_BITS).unwrap();

        assert_eq!(largest.domain_len, 1 << 18);
        assert_eq!(
            cfg.outer_field_term_bits(FIELD_BITS, &largest),
            MIN_PRODUCTION_BITS
        );

        let doubled = OuterGeometry {
            domain_len: 1 << 19,
            ..largest
        };

        assert!(doubled.domain_len.ilog2() <= MAX_OUTER_DOMAIN_LOG2);
        assert!(cfg.outer_field_term_bits(FIELD_BITS, &doubled) < MIN_PRODUCTION_BITS);
        assert!(cfg.outer_geom(12_000, 173_099, FIELD_BITS).is_err());
    }

    #[test]
    fn dev_config_waives_floor() {
        let cfg = Config::dev();

        assert!(cfg.outer_geom(64, 32, FIELD_BITS).is_ok());
    }
}
