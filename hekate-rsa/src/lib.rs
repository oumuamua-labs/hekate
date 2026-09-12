// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod pkcs1;
pub mod statement;

pub use pkcs1::{DIGEST_INFO_SHA256, encode_sha256, padding_limbs};
pub use statement::Pkcs1Statement;
