// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

pub const ZEROCHECK: &str = "zerocheck sumcheck failed";
pub const CHIPLET_ZEROCHECK: &str = "chiplet zerocheck failed";
pub const EVAL_SUMCHECK: &str = "eval sumcheck failed";
pub const FIXED_COLUMN: &str = "fixed column differs from its shape";
pub const CONSTRAINTS: &str = "constraints disagree with the zerocheck claim";
pub const OUTER_LINEAR: &str = "outer linear test failed";
pub const OUTER_COVERAGE: &str = "outer openings do not cover the queried columns";
pub const OUTER_OPENING: &str = "outer opening rejected";
pub const FOLD_FINAL: &str = "fold sumcheck final claim disagrees with the final layer";
pub const MULTIPROOF: &str = "fold tree multiproof rejected";
pub const H_EVAL_BINDING: &str = "h_eval does not match the committed h claim";
pub const BUS_SUM: &str = "bus endpoint sums do not cancel to zero";
