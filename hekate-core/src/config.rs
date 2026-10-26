// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use core::fmt;

use crate::errors;
use crate::utils::support_floor_vars;

/// Production soundness floor, per Fiat-Shamir round: a proof
/// is as strong as its weakest round (round-by-round soundness).
pub const MIN_PRODUCTION_BITS: usize = 100;

/// Most row variables one table may carry. The verifier
/// rejects a proof that claims a taller table.
pub const MAX_TABLE_VARS: usize = 32;

/// Outer row-code rate `1/2^OUTER_RATE_LOG2`.
/// Below 2 the quadratic test has no soundness.
pub const OUTER_RATE_LOG2: u32 = 4;

/// Target row count of the outer AUX oracle.
pub const OUTER_ROWS: usize = 32;

/// Precision of `log2_ratio_fixed`:
/// 32 holds the truncation error below
/// `num_queries · 2⁻³²`, under one bit.
pub(crate) const LOG2_FRAC_BITS: u32 = 32;

/// Smallest grid the fractional mode keeps at every
/// code rate; the derived query count clears it.
const FRACTIONAL_FLOOR_GRID: usize = 1 << 11;

/// Failures of the `Config` checks and the `CodeRate` decode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    /// Estimated security of `round` fell below `min_security_bits`.
    SecurityTooLow {
        round: &'static str,
        estimated_bits: usize,
        min_bits: usize,
    },

    /// `ldt_support_size <= num_queries`;
    /// the openings use up the support and
    /// the unopened leaves leak the witness.
    InsufficientSupport {
        ldt_support_size: usize,
        num_queries: usize,
    },

    /// No scalar to mask;
    /// a pad of length zero hides nothing.
    EmptyOuterStatement,

    /// No column opening;
    /// an unopened oracle proves nothing.
    ZeroOuterQueries,

    /// The outer code outgrew the largest
    /// additive-FFT domain we encode.
    OuterDomainTooLarge {
        code_len: usize,
        max_log2: u32,
    },

    UnsupportedCodeRate {
        log_inv_rate: u32,
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
            Self::InsufficientSupport {
                ldt_support_size,
                num_queries,
            } => write!(
                f,
                "ldt_support_size ({ldt_support_size}) must be > num_queries ({num_queries})",
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
        }
    }
}

/// Security metrics snapshot for a given `Config`.
#[derive(Clone, Copy, Debug)]
pub struct SecurityMetrics {
    /// Estimated relative distance δ of the linear code.
    pub relative_distance: f64,

    /// LDT spot-check count.
    pub num_queries: usize,

    /// LDT query bound: `-log₂(((1 + ρ)/2)^q)`.
    pub ldt_bits: usize,

    /// Proximity-gap term `(log2(rows) + units - 1) · n / |F|`.
    pub proximity_bits: usize,

    /// `min(ldt_bits, proximity_bits)`.
    pub security_bits: usize,
}

/// Fold the proximity term is stated over: the committed
/// grid, the eq-tensor's row count and the exponent count
/// of the `eta` and `lambda` walks, from `PoolLayout`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FoldShape {
    pub grid_cols: usize,
    pub grid_rows: usize,
    pub units: usize,
}

/// Row-code geometry chosen by `Config::table_geom`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TableGeom {
    /// Random low-coord support (LDT opening mask) length.
    pub support_size: usize,

    /// Committed codeword width (power of two).
    pub encoded_width: usize,
}

/// Brakedown row-code rate `1/inv_rate`;
/// small grids fall back to `1/(2·inv_rate)`.
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
    /// Row-code rate of the pool commitment;
    /// [`Config::prod_at`] derives the query count from it.
    pub code_rate: CodeRate,

    /// Number of LDT spot-check queries.
    pub num_queries: usize,

    /// Random low-coord support that masks the LDT
    /// column openings (data ZK). Lives inside the
    /// message, it does not widen the codeword.
    pub ldt_support_size: usize,

    /// `check_security` rejects configs
    /// whose estimated bits fall below this.
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

    /// [`Config::prod`] at `code_rate`, with that rate's
    /// query count and one support cell over it.
    pub fn prod_at(code_rate: CodeRate) -> Self {
        let num_queries = match code_rate {
            CodeRate::Half => const { fractional_floor_queries(CodeRate::Half) },
            CodeRate::Quarter => const { fractional_floor_queries(CodeRate::Quarter) },
        };

        Self {
            code_rate,
            num_queries,
            ldt_support_size: num_queries + 1,
            min_security_bits: MIN_PRODUCTION_BITS,
            outer_queries: 121,
            zero_knowledge: true,
        }
    }

    /// Fast, low-soundness parameters for tests and experiments;
    /// `min_security_bits = 0` accepts weak grids. Never deploy.
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

    /// Fewest rows whose unpacked grid holds the LDT support.
    pub fn min_table_rows(&self) -> usize {
        1 << support_floor_vars(self.ldt_support_size)
    }

    /// Committed row-code width for the chosen mode.
    pub fn encoded_width(&self, grid_cols: usize) -> usize {
        self.table_geom(grid_cols).encoded_width
    }

    /// Mode must derive from transcript-bound inputs and
    /// a fixed target only, never an unabsorbed field, or
    /// prover and verifier silently diverge on the geometry.
    pub fn table_geom(&self, grid_cols: usize) -> TableGeom {
        let inv_rate = self.code_rate.inv_rate();

        let frac = TableGeom {
            support_size: self.ldt_support_size,
            encoded_width: grid_cols * inv_rate,
        };

        let frac_msg = frac.support_size + grid_cols;

        if frac.support_size <= grid_cols
            && ldt_bits(self.num_queries, frac_msg, frac.encoded_width) >= MIN_PRODUCTION_BITS
        {
            return frac;
        }

        TableGeom {
            support_size: grid_cols,
            encoded_width: grid_cols * inv_rate * 2,
        }
    }

    /// `min(-log₂(((1 + ρ)/2)^q), proximity_gap_bits)`
    /// where ρ = msg_len / code_width, q = num_queries.
    ///
    /// Brakedown (Golovnev et al. 2022), Appendix B Theorem 7.
    pub fn estimated_security_bits(&self, field_bits: usize, shape: FoldShape) -> usize {
        let g = self.table_geom(shape.grid_cols);

        ldt_bits(
            self.num_queries,
            g.support_size + shape.grid_cols,
            g.encoded_width,
        )
        .min(self.proximity_gap_bits(field_bits, shape))
    }

    /// `(log2(rows) + units - 1) · n / |F|`: BCIKS20's `n / |F|`,
    /// Diamond-Gruen 2024/1351's eq-tensor factor over the rows,
    /// BCIKS20 Thm 1.5's curve factor over the `eta` and `lambda` walks.
    pub fn proximity_gap_bits(&self, field_bits: usize, shape: FoldShape) -> usize {
        let width = self.table_geom(shape.grid_cols).encoded_width;

        let theta = shape.grid_rows.next_power_of_two().ilog2() as usize;
        let fold_terms = (theta + shape.units.saturating_sub(1)).max(1);

        field_bits
            .saturating_sub(width.next_power_of_two().ilog2() as usize)
            .saturating_sub(fold_terms.next_power_of_two().ilog2() as usize)
    }

    /// Per-table terms at `shape`; the other draws
    /// are charged by [`Config::draw_bits`].
    pub fn security_metrics(&self, field_bits: usize, shape: FoldShape) -> SecurityMetrics {
        let g = self.table_geom(shape.grid_cols);

        SecurityMetrics {
            relative_distance: self.estimate_relative_distance(shape.grid_cols),
            num_queries: self.num_queries,
            ldt_bits: ldt_bits(
                self.num_queries,
                g.support_size + shape.grid_cols,
                g.encoded_width,
            ),
            proximity_bits: self.proximity_gap_bits(field_bits, shape),
            security_bits: self.estimated_security_bits(field_bits, shape),
        }
    }

    /// Rejects configs whose estimated soundness at
    /// `grid_cols` falls below `min_security_bits`.
    pub fn check_security(&self, field_bits: usize, shape: FoldShape) -> errors::Result<()> {
        // dev (min_security_bits == 0) waives the ZK floor
        let support = self.table_geom(shape.grid_cols).support_size;
        if self.min_security_bits > 0 && support <= self.num_queries {
            return Err(Error::InsufficientSupport {
                ldt_support_size: support,
                num_queries: self.num_queries,
            }
            .into());
        }

        let est_bits = self.estimated_security_bits(field_bits, shape);
        if est_bits < self.min_security_bits {
            return Err(Error::SecurityTooLow {
                round: "pool",
                estimated_bits: est_bits,
                min_bits: self.min_security_bits,
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

    /// Exact MDS (Singleton) distance of the chosen geometry:
    /// `δ = (encoded_width − support − grid_cols) / encoded_width`.
    /// Holds for both modes (full-half yields exactly 0.5).
    /// MDS: `hekate-math` `verus/fft.rs::fwd_semantics`.
    fn estimate_relative_distance(&self, grid_cols: usize) -> f64 {
        let g = self.table_geom(grid_cols);

        g.encoded_width.saturating_sub(g.support_size + grid_cols) as f64 / g.encoded_width as f64
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
const fn ldt_bits(num_queries: usize, msg_len: usize, code_width: usize) -> usize {
    if msg_len >= code_width {
        return 0;
    }

    let log2_ratio = log2_ratio_fixed(2 * code_width as u128, code_width as u128 + msg_len as u128);

    ((num_queries as u128 * log2_ratio) >> LOG2_FRAC_BITS) as usize
}

/// Fewest queries whose fractional mode clears `MIN_PRODUCTION_BITS`
/// at `FRACTIONAL_FLOOR_GRID` with one support cell over the count.
const fn fractional_floor_queries(code_rate: CodeRate) -> usize {
    let width = FRACTIONAL_FLOOR_GRID * code_rate.inv_rate();

    let mut queries = 1;
    while ldt_bits(queries, queries + 1 + FRACTIONAL_FLOOR_GRID, width) < MIN_PRODUCTION_BITS {
        queries += 1;
    }

    queries
}

#[cfg(test)]
mod tests {
    use super::*;

    const GRID_COLS: usize = 1024;

    /// A single-row, single-unit fold contributes no tensor
    /// and no curve factor, isolating the terms under test.
    fn flat(grid_cols: usize) -> FoldShape {
        FoldShape {
            grid_cols,
            grid_rows: 1,
            units: 1,
        }
    }

    #[test]
    fn default_is_prod() {
        assert_eq!(Config::default().min_security_bits, MIN_PRODUCTION_BITS);
        assert_eq!(Config::default().num_queries, Config::prod().num_queries);
    }

    #[test]
    fn prod_meets_production_floor() {
        let prod = Config::prod();

        assert!(prod.estimated_security_bits(128, flat(GRID_COLS)) >= MIN_PRODUCTION_BITS);
        assert!(prod.check_security(128, flat(GRID_COLS)).is_ok());
    }

    #[test]
    fn dev_is_lenient_on_weak_params() {
        let dev = Config::dev();

        assert!(dev.estimated_security_bits(128, flat(GRID_COLS)) < MIN_PRODUCTION_BITS);
        assert!(dev.check_security(128, flat(GRID_COLS)).is_ok());
    }

    #[test]
    fn prod_threshold_rejects_weak_queries() {
        let weak = Config {
            num_queries: 4,
            ..Config::prod()
        };

        assert!(weak.check_security(128, flat(GRID_COLS)).is_err());
    }

    #[test]
    fn full_half_fallback_admits_ml_dsa_grid() {
        assert!(Config::prod().check_security(128, flat(512)).is_ok());
    }

    #[test]
    fn grid_below_num_queries_rejected() {
        assert!(Config::prod().check_security(128, flat(128)).is_err());
    }

    #[test]
    fn ldt_bits_matches_float_within_one_bit() {
        let cfg = Config::prod();

        for log_g in 8usize..=20 {
            let grid_cols = 1usize << log_g;
            let g = cfg.table_geom(grid_cols);
            let msg: usize = g.support_size + grid_cols;

            if msg >= g.encoded_width {
                continue;
            }

            let rho = msg as f64 / g.encoded_width as f64;
            let per_query = (1.0 + rho) / 2.0;
            let reference = (-per_query.powf(cfg.num_queries as f64).log2()).floor();
            let integer = ldt_bits(cfg.num_queries, msg, g.encoded_width) as f64;

            assert!(
                (integer - reference).abs() <= 1.0,
                "grid 2^{log_g}: integer {integer} vs float {reference}",
            );
        }
    }

    #[test]
    fn table_geom_selects_integer_stable_modes() {
        let prod = Config::prod();

        let inv_rate = prod.code_rate.inv_rate();

        let big = prod.table_geom(1 << 12);
        assert_eq!(big.support_size, prod.ldt_support_size);
        assert_eq!(big.encoded_width, inv_rate << 12);

        let small = prod.table_geom(512);
        assert_eq!(small.support_size, 512);
        assert_eq!(small.encoded_width, inv_rate * 512 * 2);
    }

    #[test]
    fn prod_queries_sit_on_fractional_mode_edge() {
        let at = |t: usize| Config {
            num_queries: t,
            ldt_support_size: t + 1,
            ..Config::prod()
        };

        let binding = 1usize << 11;

        assert_eq!(Config::prod().num_queries, 287);
        assert_eq!(at(287).encoded_width(binding), binding * 2);
        assert_eq!(at(286).encoded_width(binding), binding * 2 * 2);
    }

    #[test]
    fn grid_1024_never_clears_floor_fractionally() {
        let best = (200..=1200)
            .map(|t| ldt_bits(t, t + 1024, 2048))
            .max()
            .unwrap();

        assert!(best < MIN_PRODUCTION_BITS, "peaked at {best}");
    }

    #[test]
    fn proximity_gap_charges_tensor_and_curve_factors() {
        let cfg = Config::prod();
        let grid_cols = 1usize << 14;

        let width_only = cfg.proximity_gap_bits(128, flat(grid_cols));

        let charged = cfg.proximity_gap_bits(
            128,
            FoldShape {
                grid_cols,
                grid_rows: 1 << 7,
                units: 80,
            },
        );

        assert_eq!(cfg.encoded_width(grid_cols), 1 << 15);
        assert_eq!(width_only, 128 - 15);
        assert_eq!(
            width_only - charged,
            (7 + 79usize).next_power_of_two().ilog2() as usize
        );
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
    fn support_below_queries_rejected() {
        let config = Config {
            ldt_support_size: 32,
            ..Config::prod()
        };

        assert!(matches!(
            config.check_security(128, flat(1 << 12)),
            Err(errors::Error::Config(Error::InsufficientSupport { .. }))
        ));
    }

    #[test]
    fn support_equal_to_queries_rejected() {
        let config = Config {
            ldt_support_size: Config::prod().num_queries,
            ..Config::prod()
        };

        assert!(matches!(
            config.check_security(128, flat(1 << 12)),
            Err(errors::Error::Config(Error::InsufficientSupport { .. }))
        ));
    }

    #[test]
    fn support_one_above_queries_accepted() {
        let prod = Config::prod();

        assert_eq!(prod.ldt_support_size, prod.num_queries + 1);
        assert!(prod.check_security(128, flat(1 << 12)).is_ok());
    }

    #[test]
    fn support_below_queries_waived_in_dev() {
        let config = Config {
            ldt_support_size: 32,
            min_security_bits: 0,
            ..Config::prod()
        };

        assert!(config.check_security(128, flat(1 << 12)).is_ok());
    }

    #[test]
    fn quarter_rate_queries_sit_on_fractional_mode_edge() {
        let at = |t: usize| Config {
            num_queries: t,
            ldt_support_size: t + 1,
            ..Config::prod_at(CodeRate::Quarter)
        };

        let binding = 1usize << 11;

        assert_eq!(Config::prod_at(CodeRate::Quarter).num_queries, 153);
        assert_eq!(at(153).encoded_width(binding), binding * 4);
        assert_eq!(at(152).encoded_width(binding), binding * 4 * 2);
    }

    #[test]
    fn quarter_rate_keeps_half_rate_mode_map() {
        let half = Config::prod();
        let quarter = Config::prod_at(CodeRate::Quarter);

        for log_g in 8..=20 {
            let grid_cols = 1usize << log_g;
            let mode = |cfg: &Config| {
                cfg.encoded_width(grid_cols) / (grid_cols * cfg.code_rate.inv_rate())
            };

            assert_eq!(mode(&half), mode(&quarter), "grid 2^{log_g}");
        }
    }

    #[test]
    fn quarter_rate_meets_production_floor() {
        let quarter = Config::prod_at(CodeRate::Quarter);

        assert_eq!(quarter.ldt_support_size, quarter.num_queries + 1);

        for grid_cols in [512, 1 << 11, 1 << 14] {
            assert!(quarter.check_security(128, flat(grid_cols)).is_ok());
        }
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

    #[test]
    fn quarter_rate_proximity_gap_pays_one_width_bit() {
        let shape = FoldShape {
            grid_cols: 1 << 14,
            grid_rows: 1 << 7,
            units: 80,
        };

        let half = Config::prod();
        let quarter = Config::prod_at(CodeRate::Quarter);

        assert_eq!(
            quarter.encoded_width(shape.grid_cols),
            2 * half.encoded_width(shape.grid_cols)
        );
        assert_eq!(
            quarter.proximity_gap_bits(128, shape),
            half.proximity_gap_bits(128, shape) - 1
        );
        assert!(quarter.check_security(128, shape).is_ok());
    }
}
