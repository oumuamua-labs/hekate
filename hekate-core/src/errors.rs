// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use crate::poly::variant;
use crate::{config, trace};
use core::fmt;
use hekate_crypto::{DefaultHasher, Hasher, merkle, transcript};

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum Error {
    Config(config::Error),
    Trace(trace::Error),
    Merkle(merkle::Error),
    Transcript(transcript::Error),
    VirtualPoly(variant::Error),

    /// Cross-crate protocol failures that
    /// do not belong to a specific sub-error enum.
    Protocol {
        protocol: &'static str,
        message: &'static str,
    },

    /// The program the verifier holds is not
    /// the audited artifact the caller pinned.
    ProgramIdMismatch {
        actual: <DefaultHasher as Hasher>::Digest,
    },

    /// Internal invariant breach; a bug,
    /// not a verifier-observable soundness failure.
    InvariantViolation {
        message: &'static str,
    },
}

/// Not `#[derive(Debug)]`: `unwrap` and `{:?}` render
/// an id as decimal integers nobody can paste back.
impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(e) => e.fmt(f),
            Self::Trace(e) => e.fmt(f),
            Self::Merkle(e) => e.fmt(f),
            Self::Transcript(e) => e.fmt(f),
            Self::VirtualPoly(e) => e.fmt(f),
            Self::Protocol { protocol, message } => {
                write!(f, "Protocol error ({protocol}): {message}")
            }
            Self::ProgramIdMismatch { actual } => {
                write!(f, "program_id mismatch: this build computes ")?;
                write_hex(f, actual.as_ref())
            }
            Self::InvariantViolation { message } => {
                write!(f, "Invariant violation: {message}")
            }
        }
    }
}

impl core::error::Error for Error {}

impl From<config::Error> for Error {
    fn from(value: config::Error) -> Self {
        Self::Config(value)
    }
}

impl From<trace::Error> for Error {
    fn from(value: trace::Error) -> Self {
        Self::Trace(value)
    }
}

impl From<merkle::Error> for Error {
    fn from(value: merkle::Error) -> Self {
        Self::Merkle(value)
    }
}

impl From<transcript::Error> for Error {
    fn from(value: transcript::Error) -> Self {
        Self::Transcript(value)
    }
}

impl From<variant::Error> for Error {
    fn from(value: variant::Error) -> Self {
        Self::VirtualPoly(value)
    }
}

fn write_hex(f: &mut fmt::Formatter<'_>, bytes: &[u8]) -> fmt::Result {
    for byte in bytes {
        write!(f, "{byte:02x}")?;
    }

    Ok(())
}
