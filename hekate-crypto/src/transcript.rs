// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use crate::{DefaultHasher, Hasher};
use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;
use core::num::NonZeroU64;
use hekate_math::TowerField;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    /// Field is wider than a single hash output; a raw squeeze
    /// cannot deliver enough entropy for a challenge.
    FieldTooLargeForChallenge {
        field_bytes: usize,
        max_entropy_bytes: usize,
    },
    EmptyIndexRange,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FieldTooLargeForChallenge {
                field_bytes,
                max_entropy_bytes,
            } => write!(
                f,
                "field too large for transcript entropy: field_bytes={field_bytes}, max_entropy_bytes={max_entropy_bytes}",
            ),
            Self::EmptyIndexRange => write!(f, "index challenge over an empty range"),
        }
    }
}

/// Op-by-op trace of transcript activity, enabled under `transcript-trace`.
/// Prover/verifier traces must match step-for-step or Fiat-Shamir diverges.
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
    ChallengeIndices {
        label: &'static [u8],
        count: u64,
    },
}

/// Fiat-Shamir transcript as a continuous hash chain:
/// inputs update the running state; challenges are
/// produced by finalizing a clone and re-absorbing the
/// digest and each squeeze depends on every prior op.
///
/// Generic over `H`:
/// SHA3 / Blake3 / SHA-256 backends are interchangeable.
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
        let mut transcript = Self {
            hasher: H::new(),
            #[cfg(feature = "transcript-trace")]
            trace: Vec::new(),
            _marker: PhantomData,
        };

        transcript.absorb_label(b"hekate-transcript-v2");
        transcript
            .hasher
            .update(&(label.len() as u64).to_le_bytes());
        transcript.absorb_label(label);

        transcript
    }

    #[cfg(feature = "transcript-trace")]
    pub fn take_trace(&mut self) -> Vec<TranscriptOp> {
        core::mem::take(&mut self.trace)
    }

    #[cfg(feature = "transcript-trace")]
    pub fn trace(&self) -> &[TranscriptOp] {
        &self.trace
    }

    /// Append labelled bytes. The length prefix is required
    /// to block length-extension collisions between messages.
    pub fn append_message(&mut self, label: &'static [u8], message: &[u8]) {
        self.absorb_label(label);

        self.hasher.update(&(message.len() as u64).to_le_bytes());
        self.hasher.update(message);

        self.pad(message.len());

        #[cfg(feature = "transcript-trace")]
        self.trace.push(TranscriptOp::AppendMessage {
            label,
            digest: payload_digest::<H>(message),
        });
    }

    /// Append a `u64` for protocol context
    /// (`num_rows`, `num_cols`, bus heights, …).
    pub fn append_u64(&mut self, label: &'static [u8], value: u64) {
        self.absorb_label(label);
        self.hasher.update(&value.to_le_bytes());

        #[cfg(feature = "transcript-trace")]
        self.trace.push(TranscriptOp::AppendU64 { label, value });
    }

    pub fn append_field<F: TowerField>(&mut self, label: &'static [u8], element: F) {
        self.absorb_label(label);

        let bytes = element.to_bytes();
        self.hasher.update(&bytes);

        self.pad(bytes.len());

        #[cfg(feature = "transcript-trace")]
        self.trace.push(TranscriptOp::AppendField {
            label,
            digest: payload_digest::<H>(&bytes),
        });
    }

    /// Append a list of field elements
    /// (e.g. a polynomial's round coefficients).
    /// Length-prefixed and serialized as `TowerField::to_bytes()`
    /// for canonical, endian-agnostic hashing.
    pub fn append_field_list<F: TowerField>(&mut self, label: &'static [u8], elements: &[F]) {
        self.absorb_label(label);

        self.hasher.update(&(elements.len() as u64).to_le_bytes());
        self.hasher.update_fields(elements.iter().copied());

        self.pad(elements.iter().map(|e| e.serialized_size()).sum());

        #[cfg(feature = "transcript-trace")]
        self.trace.push(TranscriptOp::AppendFieldList {
            label,
            count: elements.len() as u64,
            digest: {
                let mut digest_h = H::new();
                digest_h.update_fields(elements.iter().copied());

                digest_h.finalize()
            },
        });
    }

    /// Draw a field challenge via the wide-pipe Fiat-Shamir pattern:
    /// finalize a clone of the running hasher (preserving full
    /// internal entropy), then re-absorb the digest;
    /// the next challenge depends on this one.
    pub fn challenge_field<F: TowerField>(&mut self, label: &'static [u8]) -> Result<F> {
        let digest = self.squeeze(label);

        #[cfg(feature = "transcript-trace")]
        self.trace.push(TranscriptOp::ChallengeField { label });

        Self::bytes_to_field(&digest)
    }

    /// Draws `count` indices below `bound`, four per digest:
    /// each 8-byte little-endian word modulo `bound`.
    /// Errors with `EmptyIndexRange` if `bound` is zero.
    pub fn challenge_indices(
        &mut self,
        label: &'static [u8],
        count: usize,
        bound: usize,
    ) -> Result<Vec<usize>> {
        let Some(bound) = NonZeroU64::new(bound as u64) else {
            return Err(Error::EmptyIndexRange);
        };

        let mut indices = Vec::with_capacity(count);
        while indices.len() < count {
            let digest = self.squeeze(label);
            let missing = count - indices.len();

            indices.extend(
                digest
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .take(missing)
                    .map(|word| (u64::from_le_bytes(*word) % bound) as usize),
            );
        }

        #[cfg(feature = "transcript-trace")]
        self.trace.push(TranscriptOp::ChallengeIndices {
            label,
            count: count as u64,
        });

        Ok(indices)
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

    fn squeeze(&mut self, label: &'static [u8]) -> [u8; 32] {
        self.absorb_label(label);

        let digest = self.hasher.clone().finalize();

        self.hasher.update(&digest);

        digest
    }

    fn absorb_label(&mut self, label: &'static [u8]) {
        self.hasher.update(label);
        self.pad(label.len());
    }

    fn pad(&mut self, len: usize) {
        self.hasher.update(&[0u8; 8][..(8 - len % 8) % 8]);
    }
}

#[cfg(feature = "transcript-trace")]
fn payload_digest<H: Hasher>(bytes: &[u8]) -> [u8; 32] {
    let mut h = H::new();
    h.update(bytes);

    h.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hekate_math::{Block128, CanonicalSerialize};

    #[derive(Clone, Debug)]
    struct Recorder {
        bytes: Vec<u8>,
    }

    impl Hasher for Recorder {
        const OUTPUT_SIZE: usize = 32;

        fn new() -> Self {
            Self { bytes: Vec::new() }
        }

        fn update(&mut self, data: &[u8]) {
            self.bytes.extend_from_slice(data);
        }

        fn finalize(self) -> [u8; 32] {
            DefaultHasher::digest(&self.bytes)
        }

        fn finalize_reset(&mut self) -> [u8; 32] {
            DefaultHasher::digest(&core::mem::take(&mut self.bytes))
        }
    }

    #[test]
    fn stream_layout_matches_hand_built_bytes() {
        let mut transcript = Transcript::<Recorder>::new(b"t");

        transcript.append_message(b"root", &[0xAA; 32]);
        transcript.append_message(b"odd", &[1, 2, 3]);
        transcript.append_u64(b"num_rows", 7);
        transcript.append_field(b"claim", Block128(5));
        transcript.append_field_list(b"q", &[Block128(1), Block128(2)]);

        let _: Block128 = transcript.challenge_field(b"alpha").unwrap();
        let indices = transcript.challenge_indices(b"idx", 5, 1000).unwrap();

        let mut expected = Vec::new();
        expected.extend_from_slice(b"hekate-transcript-v2\0\0\0\0");
        expected.extend_from_slice(&1u64.to_le_bytes());
        expected.extend_from_slice(b"t\0\0\0\0\0\0\0");
        expected.extend_from_slice(b"root\0\0\0\0");
        expected.extend_from_slice(&32u64.to_le_bytes());
        expected.extend_from_slice(&[0xAA; 32]);
        expected.extend_from_slice(b"odd\0\0\0\0\0");
        expected.extend_from_slice(&3u64.to_le_bytes());
        expected.extend_from_slice(&[1, 2, 3, 0, 0, 0, 0, 0]);
        expected.extend_from_slice(b"num_rows");
        expected.extend_from_slice(&7u64.to_le_bytes());
        expected.extend_from_slice(b"claim\0\0\0");
        expected.extend_from_slice(&Block128(5).to_bytes());
        expected.extend_from_slice(b"q\0\0\0\0\0\0\0");
        expected.extend_from_slice(&2u64.to_le_bytes());
        expected.extend_from_slice(&Block128(1).to_bytes());
        expected.extend_from_slice(&Block128(2).to_bytes());
        expected.extend_from_slice(b"alpha\0\0\0");

        let digest = DefaultHasher::digest(&expected);
        expected.extend_from_slice(&digest);

        let mut words = Vec::new();
        for _ in 0..2 {
            expected.extend_from_slice(b"idx\0\0\0\0\0");

            let digest = DefaultHasher::digest(&expected);
            expected.extend_from_slice(&digest);

            words.extend(
                digest
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|w| u64::from_le_bytes(*w)),
            );
        }

        assert_eq!(transcript.hasher.bytes, expected);
        assert_eq!(
            indices,
            words[..5]
                .iter()
                .map(|w| (w % 1000) as usize)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn challenge_indices_rejects_empty_range() {
        let mut transcript = Transcript::<DefaultHasher>::new(b"t");

        assert_eq!(
            transcript.challenge_indices(b"idx", 3, 0),
            Err(Error::EmptyIndexRange)
        );
    }

    #[test]
    fn domain_labels_differing_in_trailing_zeros_diverge() {
        let short = Transcript::<Recorder>::new(b"app");
        let long = Transcript::<Recorder>::new(b"app\0");

        assert_ne!(short.hasher.bytes, long.hasher.bytes);
    }
}
