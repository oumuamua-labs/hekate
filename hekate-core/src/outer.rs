// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use crate::config::{Config, Error, LOG2_FRAC_BITS, OUTER_RATE_LOG2, log2_ratio_fixed};
use crate::errors;
use alloc::vec::Vec;
use hekate_crypto::{Digest, Hasher};
use hekate_math::{Flat, HardwareField, TowerField};
use zeroize::Zeroize;
#[cfg(feature = "secure-memory")]
use zeroize::ZeroizeOnDrop;

/// Interleaved, then a low/high pair per product-code test:
/// zero-sum linear, zero-vector quadratic (Ligero 2017 Fig 8-11).
pub const OUTER_MASK_ROWS: usize = 5;

/// Wires `x`, `y`, `z` of one Hadamard row.
pub const OUTER_WIRES_PER_MUL: usize = 3;

/// Additive-FFT domains are F_2-subspaces; the cap is an exponent.
const MAX_OUTER_DOMAIN_LOG2: u32 = 24;

/// Response coefficients of one proof in code lengths, `k` for the
/// interleaved test and `2k` each for the linear and the quadratic one.
const RESPONSE_CODE_LENS: usize = 5;

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

impl<F: TowerField> Pad<F> {
    /// Derives `len` uniform entries of `F` from `seed` in counter mode.
    pub fn derive<H: Hasher>(seed: &[u8; 32], len: usize) -> Self {
        let mut values = Vec::with_capacity(len);
        for i in 0..len {
            let mut hasher = H::new();
            hasher.update(seed);
            hasher.update(&(i as u64).to_le_bytes());

            let entry = F::from_uniform_bytes(&hasher.finalize().leading());

            values.push(Flat::from_raw(entry));
        }

        Self { values, next: 0 }
    }

    pub fn values(&self) -> &[Flat<F>] {
        &self.values
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
}

impl OuterGeometry {
    pub fn zk_padding(&self) -> usize {
        self.code_len - self.message_len
    }
}

impl Config {
    /// Picks the power-of-two code length whose
    /// proof carries the fewest field elements.
    pub fn outer_geom(
        &self,
        masked_scalars: usize,
        claims: usize,
        mul_wires: usize,
        field_bits: usize,
    ) -> errors::Result<OuterGeometry> {
        if masked_scalars == 0 || claims == 0 {
            return Err(Error::EmptyOuterStatement.into());
        }

        if self.outer_queries == 0 {
            return Err(Error::ZeroOuterQueries.into());
        }

        let floor = self.outer_queries.saturating_add(2);
        let shortest = floor
            .checked_next_power_of_two()
            .ok_or(Error::OuterDomainTooLarge {
                code_len: floor,
                max_log2: MAX_OUTER_DOMAIN_LOG2,
            })?;

        let longest_log = MAX_OUTER_DOMAIN_LOG2 - OUTER_RATE_LOG2;
        let past_cap_log = longest_log + 1;

        let mut code_len = shortest;
        let mut fewest = usize::MAX;

        for log_k in shortest.ilog2()..=past_cap_log {
            let candidate = 1usize << log_k;
            let elements = self.outer_proof_elements(candidate, masked_scalars, claims, mul_wires);

            if elements < fewest {
                fewest = elements;
                code_len = candidate;
            }
        }

        if code_len.ilog2() > longest_log {
            return Err(Error::OuterDomainTooLarge {
                code_len,
                max_log2: MAX_OUTER_DOMAIN_LOG2,
            }
            .into());
        }

        let geom = OuterGeometry {
            message_len: code_len - self.outer_queries - 1,
            code_len,
            domain_len: code_len << OUTER_RATE_LOG2,
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
                round: "outer",
                estimated_bits,
                min_bits: self.min_security_bits,
            }
            .into());
        }

        Ok(())
    }

    fn outer_proof_elements(
        &self,
        code_len: usize,
        masked_scalars: usize,
        claims: usize,
        mul_wires: usize,
    ) -> usize {
        let message_len = code_len - self.outer_queries - 1;
        let wire_rows = mul_wires.div_ceil(message_len).max(1);
        let rows = masked_scalars
            .div_ceil(message_len)
            .saturating_add(claims.div_ceil(message_len))
            .saturating_add(OUTER_WIRES_PER_MUL.saturating_mul(wire_rows))
            .saturating_add(OUTER_MASK_ROWS);

        let opened = rows.saturating_mul(self.outer_queries);
        let responses = RESPONSE_CODE_LENS.saturating_mul(code_len);

        opened.saturating_add(responses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MIN_PRODUCTION_BITS;
    use hekate_crypto::DefaultHasher;
    use hekate_math::Block256;

    const FIELD_BITS: usize = 256;

    /// Statements as `(masked scalars, claims, wires)`, from the
    /// least one to several times the largest shipped at 0.39.
    const STATEMENTS: [(usize, usize, usize); 4] = [
        (1, 1, 0),
        (75, 4_000, 2_384),
        (2_000, 10_000, 20_000),
        (10_000, 50_000, 90_000),
    ];

    fn assert_fewest(cfg: &Config, scalars: usize, claims: usize, wires: usize) {
        let longest = 1usize << (MAX_OUTER_DOMAIN_LOG2 - OUTER_RATE_LOG2);

        let geom = cfg.outer_geom(scalars, claims, wires, FIELD_BITS).unwrap();
        let chosen = geom.code_len;
        let fewest = cfg.outer_proof_elements(chosen, scalars, claims, wires);

        let mut code_len = (cfg.outer_queries + 2).next_power_of_two();
        while code_len <= longest {
            let elements = cfg.outer_proof_elements(code_len, scalars, claims, wires);

            assert!(elements >= fewest, "{scalars} {claims} {wires} {code_len}");
            assert!(
                elements > fewest || code_len >= chosen,
                "{scalars} {claims} {wires} {code_len}"
            );

            code_len *= 2;
        }
    }

    #[test]
    fn prod_geometry_meets_the_production_floor() {
        let cfg = Config::prod();
        let geom = cfg.outer_geom(2_000, 10_000, 20_000, FIELD_BITS).unwrap();

        assert!(cfg.outer_security_bits(FIELD_BITS, &geom) >= MIN_PRODUCTION_BITS);
        assert!(cfg.outer_field_term_bits(FIELD_BITS, &geom) >= MIN_PRODUCTION_BITS);
    }

    #[test]
    fn zk_padding_covers_every_opened_column() {
        let cfg = Config::prod();

        for (scalars, claims, wires) in STATEMENTS {
            let geom = cfg.outer_geom(scalars, claims, wires, FIELD_BITS).unwrap();
            assert!(geom.zk_padding() > geom.queries);
        }
    }

    #[test]
    fn domain_is_subspace_at_target_rate() {
        let cfg = Config::prod();
        let geom = cfg.outer_geom(2_000, 10_000, 20_000, FIELD_BITS).unwrap();

        assert!(geom.domain_len.is_power_of_two());
        assert_eq!(geom.code_len << OUTER_RATE_LOG2, geom.domain_len);
    }

    /// `AdditiveFft` takes only power-of-two sizes
    /// and `1 <= log_k < log_n <= min(F::BITS, 63)`.
    #[test]
    fn geometry_fits_additive_fft_domain_rules() {
        let cfg = Config::prod();

        for (scalars, claims, wires) in STATEMENTS {
            let geom = cfg.outer_geom(scalars, claims, wires, FIELD_BITS).unwrap();

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
            cfg.outer_geom(2_000, 10_000, 20_000, FIELD_BITS).unwrap(),
            cfg.outer_geom(2_000, 10_000, 20_000, FIELD_BITS).unwrap()
        );
    }

    #[test]
    fn empty_statement_is_rejected() {
        let cfg = Config::prod();

        assert!(cfg.outer_geom(0, 10_000, 20_000, FIELD_BITS).is_err());
        assert!(cfg.outer_geom(2_000, 0, 20_000, FIELD_BITS).is_err());
    }

    #[test]
    fn zero_outer_queries_is_rejected() {
        let cfg = Config {
            outer_queries: 0,
            ..Config::dev()
        };

        assert!(cfg.outer_geom(16, 48, 32, FIELD_BITS).is_err());
    }

    #[test]
    fn oversized_statement_is_rejected() {
        let cfg = Config::prod();

        assert!(
            cfg.outer_geom(1 << 39, 1 << 39, 1 << 40, FIELD_BITS)
                .is_err()
        );
    }

    #[test]
    fn query_count_matches_ligero_bound_at_rate_one_sixteenth() {
        let cfg = Config::prod();
        let geom = cfg.outer_geom(2_000, 10_000, 20_000, FIELD_BITS).unwrap();

        let thinner = OuterGeometry {
            queries: 154,
            ..geom
        };

        assert_eq!(cfg.outer_queries, 155);
        assert!(cfg.outer_security_bits(FIELD_BITS, &thinner) < MIN_PRODUCTION_BITS);
        assert!(cfg.outer_security_bits(FIELD_BITS, &geom) >= MIN_PRODUCTION_BITS);
    }

    /// `MAX_OUTER_DOMAIN_LOG2` binds before the field term: at the
    /// largest domain `256 - 24 = 232` still clears `MIN_PRODUCTION_BITS`.
    #[test]
    fn domain_cap_bounds_outer_statement() {
        let cfg = Config::prod();

        let largest = cfg.outer_geom(2_000, 10_000, 1 << 34, FIELD_BITS).unwrap();

        let past_cap = cfg.outer_geom(2_000, 10_000, 1 << 36, FIELD_BITS);

        assert_eq!(largest.domain_len.ilog2(), MAX_OUTER_DOMAIN_LOG2);
        assert!(cfg.outer_field_term_bits(FIELD_BITS, &largest) > MIN_PRODUCTION_BITS);
        assert!(past_cap.is_err());
    }

    #[test]
    fn dev_config_waives_floor() {
        let cfg = Config::dev();

        assert!(cfg.outer_geom(16, 48, 32, FIELD_BITS).is_ok());
    }

    #[test]
    fn pad_entries_fill_high_limb() {
        let pad = Pad::<Block256>::derive::<DefaultHasher>(&[7u8; 32], 64);

        assert!(pad.values().iter().all(|entry| entry.into_raw().0[1] != 0));
    }

    #[test]
    fn geometry_takes_code_length_of_smallest_proof() {
        let cfg = Config::prod();

        let aes = cfg.outer_geom(1_400, 952, 743, FIELD_BITS).unwrap();
        let keccak = cfg.outer_geom(63, 3_581, 3_458, FIELD_BITS).unwrap();

        assert_eq!(aes.code_len, 512);
        assert_eq!(keccak.code_len, 1_024);
    }

    #[test]
    fn tie_takes_shorter_code() {
        let cfg = Config {
            outer_queries: 4,
            ..Config::dev()
        };

        assert_eq!(cfg.outer_proof_elements(8, 9, 9, 9), 120);
        assert_eq!(cfg.outer_proof_elements(16, 9, 9, 9), 120);
        assert_eq!(cfg.outer_geom(9, 9, 9, FIELD_BITS).unwrap().code_len, 8);
    }

    #[test]
    fn no_code_length_gives_smaller_proof() {
        let few_queries = Config {
            outer_queries: 4,
            ..Config::dev()
        };

        for cfg in [Config::prod(), few_queries] {
            // Each list climbs from the least count to the counts of
            // the fibonacci, Keccak, SHA-256 and RSA-2048 statements
            // at 0.39, and the scalars and wires go far past the largest.
            for scalars in [1, 7, 100, 987, 4_649, 250_000] {
                for claims in [1, 521, 3_581, 18_414] {
                    for wires in [0, 1, 9, 289, 3_458, 13_297, 400_000] {
                        assert_fewest(&cfg, scalars, claims, wires);
                    }
                }
            }
        }
    }

    #[test]
    fn query_count_past_longest_code_is_rejected() {
        let longest = 1usize << (MAX_OUTER_DOMAIN_LOG2 - OUTER_RATE_LOG2);

        let with_queries = |outer_queries| Config {
            outer_queries,
            ..Config::dev()
        };

        let too_large = |queries| {
            matches!(
                with_queries(queries).outer_geom(1, 1, 0, FIELD_BITS),
                Err(errors::Error::Config(Error::OuterDomainTooLarge { .. }))
            )
        };

        let one_cell = with_queries(longest - 2).outer_geom(1, 1, 0, FIELD_BITS);

        assert_eq!(one_cell.unwrap().message_len, 1);
        assert!(too_large(longest - 1));
        assert!(too_large(usize::MAX));
    }
}
