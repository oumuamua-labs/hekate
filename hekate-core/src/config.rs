// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use crate::errors;
use core::fmt;

/// Production soundness floor.
pub const MIN_PRODUCTION_BITS: usize = 110;

/// Brakedown row-code rate `1/INV_RATE`;
/// small grids fall back to `1/(2·INV_RATE)`.
pub const INV_RATE: usize = 2;

/// Outer row-code rate `1/2^OUTER_RATE_LOG2`.
/// Below 2 the quadratic test has no soundness.
pub const OUTER_RATE_LOG2: u32 = 4;

/// Target row count of the outer AUX oracle.
pub const OUTER_ROWS: usize = 32;

/// Precision of `log2_ratio_fixed`:
/// 32 holds the truncation error below
/// `num_queries · 2⁻³²`, under one bit.
pub(crate) const LOG2_FRAC_BITS: u32 = 32;

/// Fractional-mode target of `table_geom`.
const FRACTIONAL_MODE_BITS: usize = 128;

/// Failures produced by `Config::check_security`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    /// Estimated security fell below `min_security_bits`.
    SecurityTooLow {
        estimated_bits: usize,
        min_bits: usize,
    },

    /// `ldt_support_size < num_queries`;
    /// opened columns exhaust the noise
    /// budget and witness data leaks.
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
    OuterDomainTooLarge { code_len: usize, max_log2: u32 },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SecurityTooLow {
                estimated_bits,
                min_bits,
            } => write!(
                f,
                "Security too low: estimated {estimated_bits} bits, but {min_bits} required",
            ),
            Self::InsufficientSupport {
                ldt_support_size,
                num_queries,
            } => write!(
                f,
                "ldt_support_size ({ldt_support_size}) must be >= num_queries ({num_queries})",
            ),
            Self::EmptyOuterStatement => {
                write!(f, "outer statement masks no scalars")
            }
            Self::ZeroOuterQueries => write!(f, "outer_queries cannot be zero"),
            Self::OuterDomainTooLarge { code_len, max_log2 } => write!(
                f,
                "outer code length {code_len} needs a domain above the 2^{max_log2} cap",
            ),
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

    /// LDT proximity bound: `-log₂((1 - δ)^q)`.
    pub ldt_bits: usize,

    /// Proximity-gap term `n / |F|`.
    pub proximity_bits: usize,

    /// `min(ldt_bits, proximity_bits)`.
    pub security_bits: usize,
}

/// Per-table row-code geometry chosen by `Config::table_geom`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TableGeom {
    /// Random low-coord support (LDT opening mask) length.
    pub support_size: usize,

    /// Committed codeword width (power of two).
    pub encoded_width: usize,
}

#[derive(Clone, Debug)]
pub struct Config {
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
    /// Production parameters: ≈128-bit soundness with the
    /// `MIN_PRODUCTION_BITS` acceptance threshold. The `Default`.
    pub fn prod() -> Self {
        Self {
            num_queries: 176,
            ldt_support_size: 200,
            min_security_bits: MIN_PRODUCTION_BITS,
            outer_queries: 155,
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

    /// Committed row-code width for the chosen per-table mode.
    pub fn encoded_width(&self, grid_cols: usize) -> usize {
        self.table_geom(grid_cols).encoded_width
    }

    /// Mode must derive from transcript-bound inputs and
    /// a fixed target only, never an unabsorbed field, or
    /// prover and verifier silently diverge on the geometry.
    pub fn table_geom(&self, grid_cols: usize) -> TableGeom {
        let frac = TableGeom {
            support_size: self.ldt_support_size,
            encoded_width: grid_cols * INV_RATE,
        };

        let frac_msg = frac.support_size + grid_cols;

        if frac.support_size <= grid_cols
            && self.ldt_bits(frac_msg, frac.encoded_width) >= FRACTIONAL_MODE_BITS
        {
            return frac;
        }

        TableGeom {
            support_size: grid_cols,
            encoded_width: grid_cols * INV_RATE * 2,
        }
    }

    /// `min(-log₂((1 - δ)^q), proximity_gap_bits)`
    /// where δ = relative distance, q = num_queries.
    ///
    /// Brakedown (Golovnev et al. 2022), Section 3.2.
    pub fn estimated_security_bits(&self, field_bits: usize, grid_cols: usize) -> usize {
        let g = self.table_geom(grid_cols);

        self.ldt_bits(g.support_size + grid_cols, g.encoded_width)
            .min(self.proximity_gap_bits(field_bits, grid_cols))
    }

    /// Proximity-gap term `n / |F|` of the
    /// random column fold (BCIKS20), in bits.
    pub fn proximity_gap_bits(&self, field_bits: usize, grid_cols: usize) -> usize {
        let width = self.table_geom(grid_cols).encoded_width;

        field_bits.saturating_sub(width.next_power_of_two().ilog2() as usize)
    }

    /// `field_bits`: `size_of::<F>() * 8`.
    pub fn security_metrics(&self, field_bits: usize, grid_cols: usize) -> SecurityMetrics {
        let g = self.table_geom(grid_cols);

        SecurityMetrics {
            relative_distance: self.estimate_relative_distance(grid_cols),
            num_queries: self.num_queries,
            ldt_bits: self.ldt_bits(g.support_size + grid_cols, g.encoded_width),
            proximity_bits: self.proximity_gap_bits(field_bits, grid_cols),
            security_bits: self.estimated_security_bits(field_bits, grid_cols),
        }
    }

    /// Rejects configs whose estimated soundness at
    /// `grid_cols` falls below `min_security_bits`.
    pub fn check_security(&self, field_bits: usize, grid_cols: usize) -> errors::Result<()> {
        // dev (min_security_bits == 0) waives the ZK floor
        let support = self.table_geom(grid_cols).support_size;
        if self.min_security_bits > 0 && support < self.num_queries {
            return Err(Error::InsufficientSupport {
                ldt_support_size: support,
                num_queries: self.num_queries,
            }
            .into());
        }

        let est_bits = self.estimated_security_bits(field_bits, grid_cols);
        if est_bits < self.min_security_bits {
            return Err(Error::SecurityTooLow {
                estimated_bits: est_bits,
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

    /// `floor(-log₂((msg_len / code_width)^q))` in bits.
    /// Integer-only, prover and verifier derive identical geometry;
    /// libm `powf`/`log2` are not bit-reproducible across platforms.
    fn ldt_bits(&self, msg_len: usize, code_width: usize) -> usize {
        if msg_len >= code_width {
            return 0;
        }

        let log2_ratio = log2_ratio_fixed(code_width as u128, msg_len as u128);

        ((self.num_queries as u128 * log2_ratio) >> LOG2_FRAC_BITS) as usize
    }
}

/// `floor(log₂(n / m) · 2^LOG2_FRAC_BITS)` for `n > m >= 1`.
/// The Q60 mantissa keeps the squaring `y² < 2¹²²`, inside `u128`.
pub(crate) fn log2_ratio_fixed(n: u128, m: u128) -> u128 {
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

    for i in 0..LOG2_FRAC_BITS {
        y = (y * y) >> S;

        if y >= 2u128 << S {
            y >>= 1;
            frac |= 1u128 << (LOG2_FRAC_BITS - 1 - i);
        }
    }

    (int_part << LOG2_FRAC_BITS) | frac
}

#[cfg(test)]
mod tests {
    use super::*;

    const GRID_COLS: usize = 1024;

    #[test]
    fn default_is_prod() {
        assert_eq!(Config::default().min_security_bits, MIN_PRODUCTION_BITS);
        assert_eq!(Config::default().num_queries, Config::prod().num_queries);
    }

    #[test]
    fn prod_meets_production_floor() {
        let prod = Config::prod();

        assert!(prod.estimated_security_bits(128, GRID_COLS) >= MIN_PRODUCTION_BITS);
        assert!(prod.check_security(128, GRID_COLS).is_ok());
    }

    #[test]
    fn dev_is_lenient_on_weak_params() {
        let dev = Config::dev();

        assert!(dev.estimated_security_bits(128, GRID_COLS) < MIN_PRODUCTION_BITS);
        assert!(dev.check_security(128, GRID_COLS).is_ok());
    }

    #[test]
    fn prod_threshold_rejects_weak_queries() {
        let weak = Config {
            num_queries: 4,
            ..Config::prod()
        };

        assert!(weak.check_security(128, GRID_COLS).is_err());
    }

    #[test]
    fn full_half_fallback_admits_ml_dsa_grid() {
        assert!(Config::prod().check_security(128, 512).is_ok());
    }

    #[test]
    fn grid_below_num_queries_rejected() {
        assert!(Config::prod().check_security(128, 128).is_err());
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

            let one_minus_delta = msg as f64 / g.encoded_width as f64;
            let reference = (-one_minus_delta.powf(cfg.num_queries as f64).log2()).floor();
            let integer = cfg.ldt_bits(msg, g.encoded_width) as f64;

            assert!(
                (integer - reference).abs() <= 1.0,
                "grid 2^{log_g}: integer {integer} vs float {reference}",
            );
        }
    }

    #[test]
    fn table_geom_selects_integer_stable_modes() {
        let prod = Config::prod();

        let big = prod.table_geom(1 << 12);
        assert_eq!(big.support_size, prod.ldt_support_size);
        assert_eq!(big.encoded_width, INV_RATE << 12);

        let small = prod.table_geom(512);
        assert_eq!(small.support_size, 512);
        assert_eq!(small.encoded_width, INV_RATE * 512 * 2);
    }
}
