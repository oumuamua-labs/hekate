// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Preflight diagnostics and the proof wire format
//! of Hekate, the Rust zero-knowledge proof engine.
//!
//! [`preflight()`] checks a witness against every constraint,
//! pinned cell, fixed column and bus of a program on the concrete
//! trace, without proving. [`serialize_proof_bytes`] and
//! [`deserialize_proof`] carry a proof over the wire, and
//! [`serialize_bundle_header`] encodes the program, instance
//! and configuration for the prover binary.
//!
//! - [Preflight][guide]: reading the report,
//!   and what a clean report does not prove
//!
//! [guide]: https://oumuamua.dev/hekate/docs/basics/preflight

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod generated;
mod program;
mod wire;

pub mod preflight;

pub use preflight::preflight;
pub use program::BundleProgram;
pub use wire::bundle::{
    DeserializedBundle, deserialize_bundle, serialize_bundle, serialize_bundle_header,
};
pub use wire::proof::{deserialize_proof, serialize_proof_bytes};
