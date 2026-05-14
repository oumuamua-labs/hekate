// SPDX-License-Identifier: Apache-2.0
// This file is part of the hekate project.
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

use crate::{DefaultHasher, Hasher};
#[cfg(feature = "transcript-trace")]
use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;
use hekate_math::TowerField;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    /// Field is wider than a single hash
    /// output, so a raw squeeze cannot
    /// deliver enough entropy for a challenge.
    FieldTooLargeForChallenge {
        field_bytes: usize,
        max_entropy_bytes: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FieldTooLargeForChallenge {
                field_bytes,
                max_entropy_bytes,
            } => write!(
                f,
                "Field too large for transcript entropy: field_bytes={field_bytes}, max_entropy_bytes={max_entropy_bytes}",
            ),
        }
    }
}

/// Op-by-op trace of transcript activity,
/// enabled under `transcript-trace`.
/// Prover/verifier traces must match
/// step-for-step or Fiat-Shamir diverges.
#[cfg(feature = "transcript-trace")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TranscriptOp {
    AppendMessage {
        label: &'static [u8],
        digest: [u8; 32],
    },
    AppendU64 {
        label: &'static [u8],
        value: u64,
    },
    AppendField {
        label: &'static [u8],
        digest: [u8; 32],
    },
    AppendFieldList {
        label: &'static [u8],
        count: u64,
        digest: [u8; 32],
    },
    ChallengeField {
        label: &'static [u8],
    },
}

/// Fiat-Shamir transcript as
/// a continuous hash chain:
/// inputs update the running state;
/// challenges are produced by finalizing
/// a clone and re-absorbing the digest
/// so each squeeze depends on every prior op.
///
/// Generic over `H` so SHA3 / Blake3 / Poseidon
/// backends are interchangeable.
#[derive(Clone, Debug)]
pub struct Transcript<H: Hasher = DefaultHasher> {
    hasher: H,
    #[cfg(feature = "transcript-trace")]
    trace: Vec<TranscriptOp>,
    _marker: PhantomData<H>,
}

impl<H: Hasher> Transcript<H> {
    /// Create a new transcript with a domain separator.
    pub fn new(label: &'static [u8]) -> Self {
        let mut hasher = H::new();
        hasher.update(b"hekate-transcript-v1");
        hasher.update(label);

        Self {
            hasher,
            #[cfg(feature = "transcript-trace")]
            trace: Vec::new(),
            _marker: PhantomData,
        }
    }

    #[cfg(feature = "transcript-trace")]
    pub fn take_trace(&mut self) -> Vec<TranscriptOp> {
        core::mem::take(&mut self.trace)
    }

    #[cfg(feature = "transcript-trace")]
    pub fn trace(&self) -> &[TranscriptOp] {
        &self.trace
    }

    /// Append labelled bytes. The length prefix
    /// is required to block length-extension
    /// collisions between messages.
    pub fn append_message(&mut self, label: &'static [u8], message: &[u8]) {
        self.hasher.update(label);

        self.hasher.update(&(message.len() as u64).to_le_bytes());
        self.hasher.update(message);

        #[cfg(feature = "transcript-trace")]
        self.trace.push(TranscriptOp::AppendMessage {
            label,
            digest: payload_digest::<H>(message),
        });
    }

    /// Append a `u64` for protocol context
    /// (`num_rows`, `num_cols`, bus heights, …).
    pub fn append_u64(&mut self, label: &'static [u8], value: u64) {
        self.hasher.update(label);
        self.hasher.update(&value.to_le_bytes());

        #[cfg(feature = "transcript-trace")]
        self.trace.push(TranscriptOp::AppendU64 { label, value });
    }

    pub fn append_field<F: TowerField>(&mut self, label: &'static [u8], element: F) {
        self.hasher.update(label);

        let bytes = element.to_bytes();
        self.hasher.update(&bytes);

        #[cfg(feature = "transcript-trace")]
        self.trace.push(TranscriptOp::AppendField {
            label,
            digest: payload_digest::<H>(&bytes),
        });
    }

    /// Append a list of field elements
    /// (e.g. a polynomial's round coefficients).
    /// Length-prefixed and serialized via
    /// `TowerField::to_bytes()` for canonical,
    /// padding-free, endian-agnostic hashing.
    pub fn append_field_list<F: TowerField>(&mut self, label: &'static [u8], elements: &[F]) {
        self.hasher.update(label);

        self.hasher.update(&(elements.len() as u64).to_le_bytes());

        #[cfg(feature = "transcript-trace")]
        let mut digest_h = H::new();
        for element in elements {
            let bytes = element.to_bytes();
            self.hasher.update(&bytes);

            #[cfg(feature = "transcript-trace")]
            digest_h.update(&bytes);
        }

        #[cfg(feature = "transcript-trace")]
        self.trace.push(TranscriptOp::AppendFieldList {
            label,
            count: elements.len() as u64,
            digest: digest_h.finalize(),
        });
    }

    /// Draw a field challenge via the
    /// wide-pipe Fiat-Shamir pattern:
    /// finalize a clone of the running hasher
    /// (preserving full internal entropy),
    /// then re-absorb the digest so the
    /// next challenge depends on this one.
    pub fn challenge_field<F: TowerField>(&mut self, label: &'static [u8]) -> Result<F> {
        self.hasher.update(label);

        let challenge_hasher = self.hasher.clone();
        let result = challenge_hasher.finalize();

        self.hasher.update(&result);

        #[cfg(feature = "transcript-trace")]
        self.trace.push(TranscriptOp::ChallengeField { label });

        Self::bytes_to_field(&result)
    }

    fn bytes_to_field<F: TowerField>(bytes: &[u8]) -> Result<F> {
        let size = size_of::<F>();
        let max_entropy_bytes = 32;

        if size > max_entropy_bytes {
            return Err(Error::FieldTooLargeForChallenge {
                field_bytes: size,
                max_entropy_bytes,
            });
        }

        // SAFETY:
        // every bit pattern is a valid
        // binary-field element, so raw-copy
        // from the digest is sound.
        let mut elem = F::default();
        unsafe {
            let elem_ptr = &mut elem as *mut F as *mut u8;
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), elem_ptr, size);
        }

        Ok(elem)
    }
}

#[cfg(feature = "transcript-trace")]
fn payload_digest<H: Hasher>(bytes: &[u8]) -> [u8; 32] {
    let mut h = H::new();
    h.update(bytes);

    h.finalize()
}
