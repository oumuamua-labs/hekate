// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! RSA-2048 PKCS#1 v1.5 signature verification
//! for Hekate, the Rust zero-knowledge proof engine.
//!
//! [`Pkcs1Statement`] builds the program, instance and
//! witness that prove `s^65537 mod N` equals the RFC 8017 §9.2
//! encoding of `H`, SHA-256 chained over committed block words
//! with FIPS 180-4 padding left to the caller. `N` is public,
//! and the signature and the message are in the witness.
//! [`encode_sha256`] computes that encoding natively.
//!
//! - [RSA-2048 PKCS#1 v1.5 Verification][rsa]: what the proof
//!   states and how the tables split the work
//!
//! [rsa]: https://oumuamua.dev/primitives/signatures/rsa

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod pkcs1;
pub mod statement;

pub use pkcs1::{DIGEST_INFO_SHA256, encode_sha256, padding_limbs};
pub use statement::Pkcs1Statement;
