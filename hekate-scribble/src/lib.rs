//! Trace mutation fuzzer for
//! the Hekate ZK proving system.
//!
//! Tampers valid execution traces
//! and asserts `hekate_sdk::preflight`
//! rejects every mutation. Scribble
//! never invokes the prover or verifier,
//! preflight (row-by-row constraint evaluation
//! plus bus multiset checking) is the oracle.

#![forbid(unsafe_code)]

pub mod apply;
pub mod check;
pub mod config;
pub mod mutation;
pub mod prelude;
pub mod strategy;
pub mod target;

pub use prelude::*;
