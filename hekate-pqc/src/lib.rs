// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Post-quantum chiplets for Hekate, the Rust zero-knowledge proof engine:
//! ML-KEM (FIPS 203) and ML-DSA (FIPS 204) at every parameter set.
//!
//! [`mlkem::MlKemChiplet`] proves key generation, encapsulation
//! and decapsulation, and [`mldsa::MlDsaChiplet`] proves signature
//! verification. Each is a pipeline of tables that a host table calls
//! over one service bus, built from an internal control table, Keccak
//! from `hekate-keccak`, and the tables in [`codec`], [`sampler`],
//! [`ntt`], [`poly_arith`], [`high_bits`] and [`kem_select`].
//!
//! - [ML-DSA Verification][mldsa]
//! - [ML-KEM Sender / Receiver][mlkem]
//!
//! [mldsa]: https://oumuamua.dev/primitives/signatures/mldsa
//! [mlkem]: https://oumuamua.dev/primitives/encryption/mlkem

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod codec;
pub mod gadgets;
pub mod high_bits;
pub mod kem_select;
pub mod mldsa;
pub mod mlkem;
pub mod ntt;
pub mod poly_arith;
pub mod sampler;
pub mod utils;
pub mod wiring;

pub(crate) mod ctrl;

#[cfg(test)]
mod census;
