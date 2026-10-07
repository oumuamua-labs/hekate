// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Trace mutation fuzzer for Hekate, the Rust zero-knowledge proof engine.
//!
//! [`assert_all_caught`] tampers with a valid trace and panics
//! with the smallest tamper that `hekate_sdk::preflight` misses,
//! and [`check_single_mutation`] runs one tamper you choose.
//! Scribble never invokes the prover or the verifier:
//! preflight, row-by-row constraint evaluation plus
//! bus multiset checking, is the oracle.
//!
//! - [Preflight: tests that tamper on purpose][tests]
//!
//! [tests]: https://oumuamua.dev/hekate/docs/basics/preflight#keep-it-in-your-tests

#![forbid(unsafe_code)]

pub mod apply;
pub mod check;
pub mod config;
pub mod mutation;
pub mod prelude;
pub mod strategy;
pub mod target;

pub use prelude::*;
