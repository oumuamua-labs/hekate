// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use core::fmt::Debug;
use hekate_math::TowerField;

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

    /// Hash of `data` in one call.
    fn digest(data: &[u8]) -> [u8; 32] {
        let mut hasher = Self::new();
        hasher.update(data);

        hasher.finalize()
    }

    /// `update` with `prefix` and one element's canonical
    /// bytes per element, batched through a stack buffer.
    fn update_fields<F: TowerField>(
        &mut self,
        prefix: &[u8],
        elements: impl IntoIterator<Item = F>,
    ) {
        let mut buf = [0u8; FIELD_BUFFER_BYTES];
        let mut filled = 0;

        for element in elements {
            let size = prefix.len() + element.serialized_size();

            if filled + size > buf.len() {
                self.update(&buf[..filled]);

                filled = 0;
            }

            let written = buf.get_mut(filled..filled + size).is_some_and(|slot| {
                let (head, tail) = slot.split_at_mut(prefix.len());
                head.copy_from_slice(prefix);

                element.serialize(tail).is_ok()
            });

            if written {
                filled += size;
            } else {
                self.update(&buf[..filled]);
                self.update(prefix);
                self.update(&element.to_bytes());

                filled = 0;
            }
        }

        self.update(&buf[..filled]);
    }
}

const FIELD_BUFFER_BYTES: usize = 16 * 1024;

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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use hekate_math::{Block32, Block128};

    fn incremental<H: Hasher>(data: &[u8]) -> [u8; 32] {
        let mut hasher = H::new();
        hasher.update(data);

        hasher.finalize()
    }

    fn per_element<H: Hasher, F: TowerField>(prefix: &[u8], elements: &[F]) -> [u8; 32] {
        let mut hasher = H::new();
        for element in elements {
            hasher.update(prefix);
            hasher.update(&element.to_bytes());
        }

        hasher.finalize()
    }

    fn buffered<H: Hasher, F: TowerField>(prefix: &[u8], elements: &[F]) -> [u8; 32] {
        let mut hasher = H::new();
        hasher.update_fields(prefix, elements.iter().copied());

        hasher.finalize()
    }

    fn check<H: Hasher>() {
        let data: Vec<u8> = (0..5000u32).map(|i| (i * 31 + 7) as u8).collect();
        for len in [0, 1, 65, 1024, 1025, 5000] {
            assert_eq!(
                H::digest(&data[..len]),
                incremental::<H>(&data[..len]),
                "len {len}"
            );
        }

        let wide: Vec<Block128> = (0..3000u128)
            .map(|i| Block128(i.wrapping_mul(0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C835)))
            .collect();

        let narrow: Vec<Block32> = (0..5000u32)
            .map(|i| Block32(i.wrapping_mul(0x9E37_79B9)))
            .collect();

        for prefix in [&b""[..], b"claimed_val"] {
            assert_eq!(
                buffered::<H, _>(prefix, &wide),
                per_element::<H, _>(prefix, &wide)
            );
            assert_eq!(
                buffered::<H, _>(prefix, &narrow),
                per_element::<H, _>(prefix, &narrow)
            );
            assert_eq!(
                buffered::<H, Block128>(prefix, &[]),
                per_element::<H, Block128>(prefix, &[])
            );
        }
    }

    #[cfg(feature = "blake3")]
    #[test]
    fn blake3_one_shot_and_buffered_fields_match_incremental() {
        check::<blake3::Blake3Hasher>();
    }

    #[cfg(feature = "sha2")]
    #[test]
    fn sha256_one_shot_and_buffered_fields_match_incremental() {
        check::<sha256::Sha256Hasher>();
    }

    #[cfg(feature = "sha3")]
    #[test]
    fn sha3_one_shot_and_buffered_fields_match_incremental() {
        check::<sha3::Sha3_256Hasher>();
    }
}
