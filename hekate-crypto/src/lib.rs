// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Hash functions, Merkle trees and the Fiat-Shamir
//! transcript of Hekate, the Rust zero-knowledge proof engine.
//!
//! [`DefaultHasher`] is the hash every proof commits with:
//! BLAKE3 by default, SHA-256 with `sha2`, SHA3-256 with `sha3`,
//! which wins when several are on. The prover and the verifier
//! must agree on it and on the label a [`transcript::Transcript`]
//! opens with. [`merkle::MerkleTree`] builds the commitment trees
//! and their batch openings.
//!
//! - [Prover Engine: commit][commit]
//! - [Prover Engine: challenges][challenges]
//!
//! [commit]: https://oumuamua.dev/hekate/docs/basics/prover-engine#step-1-commit
//! [challenges]: https://oumuamua.dev/hekate/docs/basics/prover-engine#step-2-challenges

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
extern crate core;

mod hashers;

pub mod merkle;
pub mod transcript;

pub use hashers::*;
