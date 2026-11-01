// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

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
