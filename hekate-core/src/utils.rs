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

#[cfg(feature = "std")]
pub use std::time::Instant;

#[cfg(not(feature = "std"))]
#[derive(Clone, Copy, Debug)]
pub struct Instant;

#[cfg(not(feature = "std"))]
impl Instant {
    pub fn now() -> Self {
        Self
    }

    pub fn elapsed(&self) -> core::time::Duration {
        core::time::Duration::from_secs(0)
    }
}

/// Splitting variable `c` minimising
/// `2^c · 16 + num_queries · 2^(num_vars - c) · row_bytes`,
/// floored toward `2^c >= max(2, support_size)`,
/// then capped at `num_vars`.
#[inline(always)]
pub fn compute_split_vars(
    num_vars: usize,
    num_queries: usize,
    support_size: usize,
    row_bytes: usize,
) -> usize {
    if num_vars == 0 {
        return 0;
    }

    let ratio = (row_bytes / 16).max(1);
    let factor = (num_queries * ratio).max(1);
    let optimal_c = (num_vars + factor.ilog2() as usize) / 2;

    let support_floor = if support_size > 1 {
        (support_size - 1).ilog2() as usize + 1
    } else {
        1
    };

    optimal_c.max(support_floor).clamp(1, num_vars)
}

/// Serialized width of one opened Brakedown grid row:
/// base+shift of every physical column
/// plus interleaved B128 noise.
#[inline(always)]
pub fn opened_row_bytes(physical_data_bytes: usize, sumcheck_blinding_factor: usize) -> usize {
    2 * physical_data_bytes + 2 * sumcheck_blinding_factor * 16
}
