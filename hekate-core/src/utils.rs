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
/// `2^c · 16 + num_queries · 2^(num_vars - c) · row_bytes`
/// at an assumed 128-byte row width.
#[inline(always)]
pub fn compute_split_vars(num_vars: usize, num_queries: usize) -> usize {
    if num_vars == 0 {
        return 0;
    }

    let factor = (num_queries * 8).max(1);
    let optimal_c = (num_vars + factor.ilog2() as usize) / 2;

    optimal_c.clamp(1, num_vars.max(1))
}
