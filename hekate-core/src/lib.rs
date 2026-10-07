// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Configuration, execution traces and proof types of Hekate,
//! the Rust zero-knowledge proof engine.
//!
//! A program's witness is a table of typed columns:
//! [`trace::TraceBuilder`] fills it into a [`trace::ColumnTrace`],
//! [`config::Config`] sets the proof parameters, and
//! [`proofs::InnerProof`] is what the prover returns.
//!
//! - [Execution Trace][trace]: column types, padding and secrets in memory
//! - [Prover Engine: configuration][config]
//!
//! [trace]: https://oumuamua.dev/hekate/docs/basics/execution-trace
//! [config]: https://oumuamua.dev/hekate/docs/basics/prover-engine#configuration

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod config;
pub mod errors;
pub mod ligero;
pub mod outer;
pub mod poly;
pub mod proofs;
pub mod protocol;
pub mod tensor;
pub mod trace;
pub mod utils;
