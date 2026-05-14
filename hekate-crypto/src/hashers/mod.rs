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

use core::fmt::Debug;

#[cfg(feature = "blake3")]
pub mod blake3;
#[cfg(feature = "sha2")]
pub mod sha256;
#[cfg(feature = "sha3")]
pub mod sha3;

#[cfg(not(any(feature = "blake3", feature = "sha2", feature = "sha3")))]
compile_error!("At least one hashing feature must be enabled: 'blake3', 'sha2' or 'sha3'");

/// Defines a cryptographic hash function
/// interface for Fiat-Shamir and Merkle Trees.
/// Allows switching between SHA-3, SHA-256,
/// Blake3, etc. without changing core logic.
pub trait Hasher: Clone + Debug + Send + Sync + 'static {
    /// The size of the hash output in bytes.
    const OUTPUT_SIZE: usize;

    /// Create a new hasher instance.
    fn new() -> Self;

    /// Update the internal state with input bytes.
    fn update(&mut self, data: &[u8]);

    /// Finalize the hash and return the result.
    /// Resets the internal state if needed or consumes it.
    fn finalize(self) -> [u8; 32];

    /// Finalize and reset.
    fn finalize_reset(&mut self) -> [u8; 32];
}

#[cfg(feature = "sha3")]
pub type DefaultHasher = sha3::Sha3_256Hasher;

#[cfg(all(feature = "sha2", not(feature = "sha3")))]
pub type DefaultHasher = sha256::Sha256Hasher;

#[cfg(all(feature = "blake3", not(feature = "sha3"), not(feature = "sha2")))]
pub type DefaultHasher = blake3::Blake3Hasher;

#[cfg(feature = "sha3")]
#[inline]
pub const fn default_hasher_name() -> &'static str {
    "hekate_crypto::sha3::Sha3_256Hasher"
}

#[cfg(all(feature = "sha2", not(feature = "sha3")))]
#[inline]
pub const fn default_hasher_name() -> &'static str {
    "hekate_crypto::sha256::Sha256Hasher"
}

#[cfg(all(feature = "blake3", not(feature = "sha3"), not(feature = "sha2")))]
#[inline]
pub const fn default_hasher_name() -> &'static str {
    "hekate_crypto::blake3::Blake3Hasher"
}
