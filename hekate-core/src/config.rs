// SPDX-License-Identifier: Apache-2.0
// This file is part of the hekate project.
// Copyright (C) 2026 Andrei Kochergin <andrei@oumuamua.dev>
// Copyright (C) 2026 Oumuamua Labs <info@oumuamua.dev>. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::errors;
use crate::utils::compute_split_vars;
use core::fmt;
use tracing::warn;

/// Failures produced by `Config::check_security`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    /// Estimated security fell below `min_security_bits`.
    SecurityTooLow {
        estimated_bits: usize,
        min_bits: usize,
    },

    /// `ldt_blinding_factor < num_queries`;
    /// opened columns exhaust the noise
    /// budget and witness data leaks.
    InsufficientLdtBlinding {
        ldt_blinding_factor: usize,
        num_queries: usize,
    },
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
            Self::InsufficientLdtBlinding {
                ldt_blinding_factor,
                num_queries,
            } => write!(
                f,
                "ldt_blinding_factor ({ldt_blinding_factor}) must be >= num_queries ({num_queries})",
            ),
        }
    }
}

/// Security metrics snapshot for a given `Config`.
#[derive(Clone, Copy, Debug)]
pub struct SecurityMetrics {
    /// Estimated relative distance
    /// δ of the linear code.
    pub relative_distance: f64,

    /// LDT spot-check count.
    pub num_queries: usize,

    /// Soundness error:
    /// `(1 - δ)^q`.
    pub soundness_error: f64,

    /// LDT proximity bound:
    /// `-log₂(soundness_error)`.
    pub ldt_bits: usize,

    /// `min(ldt_bits, field_bits)`. Schwartz-Zippel
    /// caps Sumcheck / ZeroCheck / LogUp at field size.
    pub security_bits: usize,

    /// Non-zero entries per row of
    /// the expander matrix.
    pub expansion_degree: usize,
}

#[derive(Clone, Debug)]
pub struct Config {
    /// Non-zero entries per row in
    /// the expander matrix.
    pub expansion_degree: usize,

    /// Number of LDT spot-check queries.
    pub num_queries: usize,

    /// Seed for the deterministic RNG
    /// that samples the expander matrix.
    pub matrix_seed: [u8; 32],

    /// Blinding columns for algebraic ZK
    /// (Sumcheck), extends the 1D trace.
    pub sumcheck_blinding_factor: usize,

    /// Blinding columns for data ZK
    /// (LDT), extends the 2D grid width.
    /// Must be `>= num_queries`.
    pub ldt_blinding_factor: usize,

    /// `check_security` rejects configs
    /// whose estimated bits fall below this.
    pub min_security_bits: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            expansion_degree: 16,
            num_queries: 160,
            matrix_seed: [42u8; 32],
            min_security_bits: 99,
            sumcheck_blinding_factor: 2,
            ldt_blinding_factor: 200,
        }
    }
}

impl Config {
    /// `min(-log₂((1 - δ)^q), field_bits)` where
    /// δ = relative distance, q = num_queries.
    ///
    /// Brakedown (Golovnev et al. 2022), Section 3.2.
    pub fn estimated_security_bits(&self, field_bits: usize) -> usize {
        let delta = self.estimate_relative_distance();
        let q = self.num_queries as f64;

        let soundness_error = (1.0 - delta).powf(q);
        let ldt_bits = (-soundness_error.log2()).floor() as usize;

        ldt_bits.min(field_bits)
    }

    /// `field_bits`: `size_of::<F>() * 8`.
    pub fn security_metrics(&self, field_bits: usize) -> SecurityMetrics {
        let delta = self.estimate_relative_distance();
        let q = self.num_queries as f64;
        let soundness_error = (1.0 - delta).powf(q);
        let ldt_bits = (-soundness_error.log2()).floor() as usize;

        SecurityMetrics {
            relative_distance: delta,
            num_queries: self.num_queries,
            soundness_error,
            ldt_bits,
            security_bits: ldt_bits.min(field_bits),
            expansion_degree: self.expansion_degree,
        }
    }

    /// Rejects configs that can't meet
    /// `min_security_bits` for the given
    /// trace dimensions.
    pub fn check_security(&self, num_vars: usize, field_bits: usize) -> errors::Result<()> {
        if self.ldt_blinding_factor < self.num_queries {
            return Err(Error::InsufficientLdtBlinding {
                ldt_blinding_factor: self.ldt_blinding_factor,
                num_queries: self.num_queries,
            }
            .into());
        }

        let split_vars = compute_split_vars(num_vars, self.num_queries);
        let grid_cols = 1usize << split_vars;

        // Random-expander δ guarantees
        // break down on very narrow grids.
        if grid_cols > 0 && grid_cols < 128 && self.min_security_bits > 40 {
            warn!("Grid width ({grid_cols}) too small for random expander guarantees");
        }

        // degree >> grid_cols degrades δ.
        if grid_cols > 0 && self.expansion_degree > grid_cols / 4 {
            warn!(
                "Expansion degree ({}) too large for grid width ({}), need < {}",
                self.expansion_degree,
                grid_cols,
                grid_cols / 4
            );
        }

        let est_bits = self.estimated_security_bits(field_bits);
        if est_bits < self.min_security_bits {
            return Err(Error::SecurityTooLow {
                estimated_bits: est_bits,
                min_bits: self.min_security_bits,
            }
            .into());
        }

        Ok(())
    }

    /// Sipser-Spielman "Expander Codes"
    /// (1996) bound `δ ≥ (d - 2√(d-1)) / d`,
    /// scaled by an empirical correction
    /// for finite random graphs.
    fn estimate_relative_distance(&self) -> f64 {
        let d = self.expansion_degree as f64;
        if d < 2.0 {
            return 0.01;
        }

        let sqrt_term = 2.0 * (d - 1.0).sqrt();
        let theoretical_delta = (d - sqrt_term) / d;

        // Random-graph correction:
        // tighter as d grows.
        let correction_factor = if d >= 64.0 {
            0.95
        } else if d >= 32.0 {
            0.90
        } else if d >= 16.0 {
            // Standard Brakedown parameters
            0.85
        } else if d >= 8.0 {
            0.75
        } else {
            0.60
        };

        (theoretical_delta * correction_factor).max(0.01)
    }
}
