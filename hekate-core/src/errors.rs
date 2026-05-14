// SPDX-License-Identifier: Apache-2.0
// This file is part of the hekate-math project.
// Copyright (C) 2026 Andrei Kochergin <andrei@oumuamua.dev>
// Copyright (C) 2026 Oumuamua Labs <info@oumuamua.dev>. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::poly::variant;
use crate::{config, trace};
use core::fmt;
use hekate_crypto::{merkle, transcript};

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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

    /// Internal invariant breach; a bug,
    /// not a verifier-observable soundness failure.
    InvariantViolation {
        message: &'static str,
    },
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
            Self::InvariantViolation { message } => {
                write!(f, "Invariant violation: {message}")
            }
        }
    }
}

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
