// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! RSA PKCS#1 v1.5 signature encoding for SHA-256
//! (RFC 8017 §9.2), as base-2^32 limbs.
//!
//! `EM = 0x00 || 0x01 || 0xFF… || 0x00 || DigestInfo || H`,
//! 256 bytes big-endian. Limbs run little-endian, hence
//! the digest lands in limbs `0..8` with its word order
//! reversed, and limbs `8..64` are the encoding constants.

use hekate_gadgets::chiplets::bignum::modexp::LIMBS32;

pub const EM_BYTES: usize = LIMBS32 * 4;
pub const DIGEST_WORDS: usize = 8;
pub const DIGEST_LIMBS: usize = DIGEST_WORDS;

/// RFC 8017 §9.2 note 1, the `DigestInfo` DER prefix for `id-sha256`.
#[rustfmt::skip]
pub const DIGEST_INFO_SHA256: [u8; 19] = [
    0x30, 0x31, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01,
    0x65, 0x03, 0x04, 0x02, 0x01, 0x05, 0x00, 0x04, 0x20,
];

/// `digest` is big-endian words, `H[0]` first.
pub fn encode_sha256(digest: &[u32; DIGEST_WORDS]) -> [u32; LIMBS32] {
    let mut em = [0u8; EM_BYTES];
    let info_at = EM_BYTES - 4 * DIGEST_WORDS - DIGEST_INFO_SHA256.len();

    em[1] = 0x01;

    for byte in em[2..info_at - 1].iter_mut() {
        *byte = 0xff;
    }

    em[info_at..info_at + DIGEST_INFO_SHA256.len()].copy_from_slice(&DIGEST_INFO_SHA256);

    for (i, &word) in digest.iter().enumerate() {
        let at = EM_BYTES - 4 * DIGEST_WORDS + 4 * i;
        em[at..at + 4].copy_from_slice(&word.to_be_bytes());
    }

    limbs_from_be(&em)
}

/// Limbs below [`DIGEST_LIMBS`] are zero;
/// the rest are the encoding constants an AIR pins.
pub fn padding_limbs() -> [u32; LIMBS32] {
    encode_sha256(&[0; DIGEST_WORDS])
}

pub fn digest_word_of_limb(j: usize) -> usize {
    DIGEST_WORDS - 1 - j
}

fn limbs_from_be(em: &[u8; EM_BYTES]) -> [u32; LIMBS32] {
    core::array::from_fn(|j| {
        let at = EM_BYTES - 4 * (j + 1);

        u32::from_be_bytes([em[at], em[at + 1], em[at + 2], em[at + 3]])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const INFO_AT: usize = 205;
    const DIGEST_AT: usize = 224;

    /// `hekate rsa pkcs1 v1_5 sha256 statement`
    /// hashed with CPython `hashlib.sha256`.
    const DIGEST: [u32; DIGEST_WORDS] = [
        0xb692_4278,
        0xc87f_c712,
        0xefea_0807,
        0x2b20_75dc,
        0x134e_dafe,
        0x1c23_bf03,
        0x61a5_acdd,
        0x6bdb_36bc,
    ];

    fn to_be_bytes(em: &[u32; LIMBS32]) -> [u8; EM_BYTES] {
        let mut bytes = [0u8; EM_BYTES];
        for (j, &limb) in em.iter().enumerate() {
            let at = EM_BYTES - 4 * (j + 1);
            bytes[at..at + 4].copy_from_slice(&limb.to_be_bytes());
        }

        bytes
    }

    #[test]
    fn digest_occupies_low_limbs_reversed() {
        let em = encode_sha256(&DIGEST);

        for j in 0..DIGEST_LIMBS {
            assert_eq!(em[j], DIGEST[digest_word_of_limb(j)], "limb {j}");
        }
    }

    #[test]
    fn padding_is_independent_of_digest() {
        let padding = padding_limbs();
        let em = encode_sha256(&DIGEST);

        for j in DIGEST_LIMBS..LIMBS32 {
            assert_eq!(em[j], padding[j], "limb {j}");
        }

        for (j, &limb) in padding.iter().enumerate().take(DIGEST_LIMBS) {
            assert_eq!(limb, 0, "limb {j}");
        }
    }

    #[test]
    fn encoding_matches_rfc_8017_byte_layout() {
        let bytes = to_be_bytes(&encode_sha256(&DIGEST));

        assert_eq!(bytes[0], 0x00);
        assert_eq!(bytes[1], 0x01);
        assert!(bytes[2..INFO_AT - 1].iter().all(|&b| b == 0xff));
        assert_eq!(bytes[2..INFO_AT - 1].len(), 202);
        assert_eq!(bytes[INFO_AT - 1], 0x00);
        assert_eq!(bytes[INFO_AT..DIGEST_AT], DIGEST_INFO_SHA256);

        for (i, &word) in DIGEST.iter().enumerate() {
            let at = DIGEST_AT + 4 * i;
            assert_eq!(bytes[at..at + 4], word.to_be_bytes());
        }
    }

    #[test]
    fn top_limb_leads_with_zero_and_one_octets() {
        assert_eq!(padding_limbs()[LIMBS32 - 1], 0x0001_ffff);
    }
}
