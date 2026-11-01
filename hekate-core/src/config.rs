// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use core::fmt;

use crate::errors;

/// Production soundness floor, per Fiat-Shamir round: a proof
/// is as strong as its weakest round (round-by-round soundness).
pub const MIN_PRODUCTION_BITS: usize = 100;

/// Most row variables one table may carry. The verifier
/// rejects a proof that claims a taller table.
pub const MAX_TABLE_VARS: usize = 32;

/// Largest `fold_arity_log`: bounds the leaf width and the
/// verifier's interpolation for a bundle-supplied config.
pub const MAX_FOLD_ARITY_LOG: usize = 4;

/// Outer row-code rate `1/2^OUTER_RATE_LOG2`.
/// Below 2 the quadratic test has no soundness.
pub const OUTER_RATE_LOG2: u32 = 4;

/// Target row count of the outer AUX oracle.
pub const OUTER_ROWS: usize = 32;

/// Precision of `log2_ratio_fixed`:
/// 32 holds the truncation error below
/// `num_queries · 2⁻³²`, under one bit.
pub(crate) const LOG2_FRAC_BITS: u32 = 32;

/// Failures of the `Config` checks and the `CodeRate` decode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    /// Estimated security of `round` fell below `min_security_bits`.
    SecurityTooLow {
        round: &'static str,
        estimated_bits: usize,
        min_bits: usize,
    },

    /// No scalar to mask;
    /// a pad of length zero hides nothing.
    EmptyOuterStatement,

    /// No column opening;
    /// an unopened oracle proves nothing.
    ZeroOuterQueries,

    /// The outer code outgrew the largest
    /// additive-FFT domain we encode.
    OuterDomainTooLarge { code_len: usize, max_log2: u32 },

    /// `log_inv_rate` decodes to no [`CodeRate`]:
    /// only 1 (rate 1/2) and 2 (rate 1/4) do.
    UnsupportedCodeRate { log_inv_rate: u32 },

    /// `fold_arity_log` is 0 or above [`MAX_FOLD_ARITY_LOG`].
    FoldArity { fold_arity_log: usize },

    /// `early_stop_vars + 1 < fold_arity_log`: some
    /// heights then admit no stop with whole rounds.
    EarlyStopBelowArity {
        early_stop_vars: usize,
        fold_arity_log: usize,
    },

    /// The largest polynomial has fewer variables
    /// than one committed round binds.
    PoolBelowArity {
        num_vars: usize,
        fold_arity_log: usize,
    },

    /// The codeword of `num_vars + log_inv_rate` index bits
    /// does not fit a machine word, where shifts would wrap.
    PoolAboveWord {
        num_vars: usize,
        log_inv_rate: usize,
    },

    /// The uniform blocks of polynomial `poly` cannot hold its
    /// noise: its unopened leaves would be fixed by the witness.
    TableBelowNoise {
        poly: usize,
        row_vars: usize,
        noise_cells: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SecurityTooLow {
                round,
                estimated_bits,
                min_bits,
            } => write!(
                f,
                "Security too low ({round}): estimated {estimated_bits} bits, \
                 but {min_bits} required",
            ),
            Self::EmptyOuterStatement => {
                write!(f, "outer statement masks no scalars")
            }
            Self::ZeroOuterQueries => write!(f, "outer_queries cannot be zero"),
            Self::OuterDomainTooLarge { code_len, max_log2 } => write!(
                f,
                "outer code length {code_len} needs a domain above the 2^{max_log2} cap",
            ),
            Self::UnsupportedCodeRate { log_inv_rate } => {
                write!(f, "code rate 1/2^{log_inv_rate} is not one of 1/2, 1/4")
            }
            Self::FoldArity { fold_arity_log } => write!(
                f,
                "fold_arity_log ({fold_arity_log}) must lie in 1..={MAX_FOLD_ARITY_LOG}",
            ),
            Self::EarlyStopBelowArity {
                early_stop_vars,
                fold_arity_log,
            } => write!(
                f,
                "early_stop_vars ({early_stop_vars}) must be at least \
                 fold_arity_log - 1 ({fold_arity_log} - 1)",
            ),
            Self::PoolBelowArity {
                num_vars,
                fold_arity_log,
            } => write!(
                f,
                "largest polynomial has {num_vars} variables, \
                 below one fold round of {fold_arity_log}",
            ),
            Self::PoolAboveWord {
                num_vars,
                log_inv_rate,
            } => write!(
                f,
                "largest polynomial has {num_vars} variables at rate 2^-{log_inv_rate}, \
                 past the {}-bit codeword index",
                usize::BITS,
            ),
            Self::TableBelowNoise {
                poly,
                row_vars,
                noise_cells,
            } => write!(
                f,
                "polynomial {poly}: uniform blocks of 2^{row_vars} rows \
                 cannot hold its {noise_cells} noise cells",
            ),
        }
    }
}

/// RS code rate `1/inv_rate` of every fold codeword.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum CodeRate {
    Half = 1,
    Quarter = 2,
}

impl CodeRate {
    pub const fn log_inv_rate(self) -> u32 {
        self as u32
    }

    pub const fn inv_rate(self) -> usize {
        1 << self.log_inv_rate()
    }
}

impl TryFrom<u32> for CodeRate {
    type Error = Error;

    fn try_from(log_inv_rate: u32) -> Result<Self, Error> {
        match log_inv_rate {
            1 => Ok(Self::Half),
            2 => Ok(Self::Quarter),
            _ => Err(Error::UnsupportedCodeRate { log_inv_rate }),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    /// Code rate of the fold's RS codewords;
    /// [`Config::prod_at`] derives the query count from it.
    pub code_rate: CodeRate,

    /// Number of LDT spot-check queries.
    pub num_queries: usize,

    /// Variables one committed round of the fold binds;
    /// its leaves hold `2^fold_arity_log` symbols.
    pub fold_arity_log: usize,

    /// Bound on the variables of the fold's final layer;
    /// [`crate::fold::FoldLayout`] derives the stop under it.
    pub early_stop_vars: usize,

    /// Fold layouts and draws whose estimated
    /// bits fall below this are rejected.
    pub min_security_bits: usize,

    /// Column openings of the outer argument.
    pub outer_queries: usize,

    /// Blind units, one-time pad and outer
    /// argument on; `false` proves in the clear.
    pub zero_knowledge: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self::prod()
    }
}

impl Config {
    /// Production parameters: 100-bit soundness with the
    /// `MIN_PRODUCTION_BITS` acceptance threshold. The `Default`.
    pub fn prod() -> Self {
        Self::prod_at(CodeRate::Half)
    }

    /// [`Config::prod`] at `code_rate`, with the fewest
    /// queries whose fold term clears the floor.
    pub fn prod_at(code_rate: CodeRate) -> Self {
        let num_queries = match code_rate {
            CodeRate::Half => const { fold_floor_queries(CodeRate::Half) },
            CodeRate::Quarter => const { fold_floor_queries(CodeRate::Quarter) },
        };

        Self {
            code_rate,
            num_queries,
            fold_arity_log: 4,
            early_stop_vars: 11,
            min_security_bits: MIN_PRODUCTION_BITS,
            outer_queries: 121,
            zero_knowledge: true,
        }
    }

    /// Fast, low-soundness parameters for tests and experiments;
    /// `min_security_bits = 0` accepts weak layouts. Never deploy.
    pub fn dev() -> Self {
        Self {
            num_queries: 4,
            min_security_bits: 0,
            outer_queries: 4,
            zero_knowledge: false,
            ..Self::prod()
        }
    }

    /// Blind columns committed with the whole eval master.
    pub fn blind_units(&self) -> usize {
        usize::from(self.zero_knowledge)
    }

    /// Rejects a fold arity outside `1..=MAX_FOLD_ARITY_LOG`
    /// and an early stop below one fold round.
    pub fn check_fold(&self) -> errors::Result<()> {
        let arity_log = self.fold_arity_log;

        if !(1..=MAX_FOLD_ARITY_LOG).contains(&arity_log) {
            return Err(Error::FoldArity {
                fold_arity_log: arity_log,
            }
            .into());
        }

        if self.early_stop_vars < arity_log - 1 {
            return Err(Error::EarlyStopBelowArity {
                early_stop_vars: self.early_stop_vars,
                fold_arity_log: arity_log,
            }
            .into());
        }

        Ok(())
    }

    /// Rejects a draw whose [`Config::draw_bits`]
    /// fall below `min_security_bits`; 0 waives it.
    pub fn check_draw(
        &self,
        round: &'static str,
        field_bits: usize,
        events: u128,
    ) -> errors::Result<()> {
        if self.min_security_bits == 0 {
            return Ok(());
        }

        let bits = self.draw_bits(field_bits, events);

        if bits < self.min_security_bits {
            return Err(Error::SecurityTooLow {
                round,
                estimated_bits: bits,
                min_bits: self.min_security_bits,
            }
            .into());
        }

        Ok(())
    }

    /// One Fiat-Shamir draw from a `2^field_bits` field that at most
    /// `events` values defeat, in bits: `field_bits − ⌈log₂ events⌉`.
    pub fn draw_bits(&self, field_bits: usize, events: u128) -> usize {
        let log_events = events
            .max(1)
            .checked_next_power_of_two()
            .map_or(u128::BITS, u128::ilog2) as usize;

        field_bits.saturating_sub(log_events)
    }
}

/// `floor(log₂(n / m) · 2^LOG2_FRAC_BITS)` for `n > m >= 1`.
/// The Q60 mantissa keeps the squaring `y² < 2¹²²`, inside `u128`.
pub(crate) const fn log2_ratio_fixed(n: u128, m: u128) -> u128 {
    const S: u32 = 60;

    let mut scaled_m = m;
    let mut int_part: u128 = 0;

    while scaled_m <= n / 2 {
        scaled_m <<= 1;
        int_part += 1;
    }

    // m·2^int_part ∈ (n/2, n],
    // y = n/(m·2^int_part) ∈ [1, 2) in Q_S.
    let mut y = (n << S) / scaled_m;
    let mut frac: u128 = 0;

    let mut i = 0;
    while i < LOG2_FRAC_BITS {
        y = (y * y) >> S;

        if y >= 2u128 << S {
            y >>= 1;
            frac |= 1u128 << (LOG2_FRAC_BITS - 1 - i);
        }

        i += 1;
    }

    (int_part << LOG2_FRAC_BITS) | frac
}

/// `floor(-log₂(((1 + ρ)/2)^q))` in bits, ρ = msg_len / code_width.
/// Integer-only, prover and verifier derive identical geometry;
/// libm `powf`/`log2` are not bit-reproducible across platforms.
pub(crate) const fn ldt_bits(num_queries: usize, msg_len: usize, code_width: usize) -> usize {
    if msg_len >= code_width {
        return 0;
    }

    let log2_ratio = log2_ratio_fixed(2 * code_width as u128, code_width as u128 + msg_len as u128);

    ((num_queries as u128 * log2_ratio) >> LOG2_FRAC_BITS) as usize
}

/// Fewest queries whose fold term `⌊t · log2(2 / (1 + ρ))⌋`
/// clears `MIN_PRODUCTION_BITS` at `code_rate`.
const fn fold_floor_queries(code_rate: CodeRate) -> usize {
    let mut queries = 1;
    while ldt_bits(queries, 1, code_rate.inv_rate()) < MIN_PRODUCTION_BITS {
        queries += 1;
    }

    queries
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_prod() {
        assert_eq!(Config::default().min_security_bits, MIN_PRODUCTION_BITS);
        assert_eq!(Config::default().num_queries, Config::prod().num_queries);
    }

    #[test]
    fn prod_queries_are_fewest_clearing_fold_floor() {
        for (rate, queries) in [(CodeRate::Half, 241), (CodeRate::Quarter, 148)] {
            let t = Config::prod_at(rate).num_queries;

            assert_eq!(t, queries);
            assert!(ldt_bits(t, 1, rate.inv_rate()) >= MIN_PRODUCTION_BITS);
            assert!(ldt_bits(t - 1, 1, rate.inv_rate()) < MIN_PRODUCTION_BITS);
        }
    }

    #[test]
    fn ldt_bits_matches_float_within_one_bit() {
        for rate in [CodeRate::Half, CodeRate::Quarter] {
            let inv_rate = rate.inv_rate();
            let per_query = (1.0 + 1.0 / inv_rate as f64) / 2.0;

            for num_queries in (50..=400).step_by(7) {
                let reference = (-per_query.powf(num_queries as f64).log2()).floor();
                let integer = ldt_bits(num_queries, 1, inv_rate) as f64;

                assert!(
                    (integer - reference).abs() <= 1.0,
                    "rate 1/{inv_rate}, {num_queries} queries: {integer} vs {reference}",
                );
            }
        }
    }

    #[test]
    fn draw_sits_on_production_floor_edge() {
        let prod = Config::prod();
        let edge = 1u128 << (128 - MIN_PRODUCTION_BITS);

        assert_eq!(prod.draw_bits(128, edge), MIN_PRODUCTION_BITS);
        assert_eq!(prod.draw_bits(128, edge + 1), MIN_PRODUCTION_BITS - 1);

        assert!(prod.check_draw("bus draw", 128, edge).is_ok());
        assert!(matches!(
            prod.check_draw("bus draw", 128, edge + 1),
            Err(errors::Error::Config(Error::SecurityTooLow {
                round: "bus draw",
                ..
            }))
        ));
    }

    #[test]
    fn draw_clears_for_one_event_and_dev_configs() {
        let prod = Config::prod();

        assert_eq!(prod.draw_bits(128, 0), 128);
        assert_eq!(prod.draw_bits(128, 1), 128);
        assert!(prod.check_draw("bus draw", 128, 0).is_ok());

        let dev = Config::dev();

        assert_eq!(dev.min_security_bits, 0);
        assert!(dev.check_draw("bus draw", 128, 1 << 60).is_ok());
    }

    #[test]
    fn draw_saturates_past_u128_events() {
        assert_eq!(Config::prod().draw_bits(256, u128::MAX), 128);
    }

    #[test]
    fn bus_weight_past_128_bit_floor_clears_at_256_bits() {
        let prod = Config::prod();
        let weight = 1u128 << 30;

        assert!(prod.check_draw("bus draw", 128, weight).is_err());
        assert!(prod.check_draw("bus draw", 256, weight).is_ok());
    }

    #[test]
    fn code_rate_decodes_only_half_and_quarter() {
        for rate in [CodeRate::Half, CodeRate::Quarter] {
            assert_eq!(CodeRate::try_from(rate.log_inv_rate()), Ok(rate));
        }

        for log_inv_rate in [0, 3, u32::MAX] {
            assert_eq!(
                CodeRate::try_from(log_inv_rate),
                Err(Error::UnsupportedCodeRate { log_inv_rate })
            );
        }
    }
}
