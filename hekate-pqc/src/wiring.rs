// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Labels, and the buses that move coefficients,
//! words, lanes and hint keys between PQC chiplets.
//! Label 0 marks a slot that carries nothing.

use alloc::boxed::Box;
use alloc::collections::btree_map::Entry;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;
use hekate_core::errors::{Error, Result};
use hekate_math::TowerField;
use hekate_program::permutation::{PermutationCheckSpec, Source};
use hekate_program::{CadenceSegment, FixedShape};
use subtle::ConstantTimeEq;
use zeroize::Zeroize;

/// Coefficients per polynomial.
pub const N: usize = 256;

/// Bus that carries `(poly, pos, value)` coefficient tokens.
pub const COEF_BUS_ID: &str = "coef";

/// Bus that carries `(stream, index, word)` 32-bit word tokens.
pub const WORD_BUS_ID: &str = "word";

/// Bus that carries `(stream, index, lane)` 64-bit lane tokens.
pub const LANE_BUS_ID: &str = "lane";

/// Bus that carries the `(16·call + i + 1, pos)` key of every set hint bit.
pub const HINT_BUS_ID: &str = "hint";

const COEF_WAIVER: &str = "see hekate-pqc/src/census.rs: every nonzero (poly, pos) \
     has exactly two emitters, and the table that produces it is one of them";

const WORD_WAIVER: &str = "see hekate-pqc/src/census.rs: stream and index are fixed \
     columns; every (stream, index) has exactly two emitters, a producer among them";

const LANE_WAIVER: &str = "see hekate-pqc/src/census.rs: stream and index are fixed \
     columns; every (stream, index) has exactly two emitters, a producer among them";

const HINT_WAIVER: &str = "see hekate-pqc/src/codec/air.rs: every nonzero key \
     (hint_poly(call, i), pos) is unique on its endpoint, by fixed labels in HighBits \
     and by a pinned call base and strictly increasing indices in the Codec; the zero \
     keys are even in number across the bus and cancel";

/// Label naming one polynomial on the `coef` bus.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Poly(u16);

impl Poly {
    /// The label as it appears in the bus key.
    pub fn id(self) -> u16 {
        self.0
    }
}

/// Label naming one word or lane stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Stream(u16);

impl Stream {
    /// The label as it appears in the bus key.
    pub fn id(self) -> u16 {
        self.0
    }
}

/// Hands out distinct polynomial labels for one composite.
#[derive(Debug)]
pub struct PolyLabels {
    next: u32,
}

impl PolyLabels {
    /// Starts a composite's labels at 1.
    pub fn new() -> Self {
        Self { next: 1 }
    }

    /// Returns the next unused label.
    pub fn fresh(&mut self) -> Result<Poly> {
        let id = u16::try_from(self.next).map_err(|_| Error::Protocol {
            protocol: "pqc_wiring",
            message: "coef bus labels exhausted: 65535 polynomials per composite",
        })?;

        self.next += 1;

        Ok(Poly(id))
    }

    /// Returns the next unused label for a stream.
    pub fn stream(&mut self) -> Result<Stream> {
        self.fresh().map(|poly| Stream(poly.0))
    }
}

impl Default for PolyLabels {
    fn default() -> Self {
        Self::new()
    }
}

/// Coefficients of each produced polynomial, by label.
#[derive(Default)]
pub struct PolyValues {
    map: BTreeMap<Poly, Box<[u32; N]>>,
}

impl PolyValues {
    /// Records the coefficients of `poly` or checks a repeat against them.
    pub fn insert(&mut self, poly: Poly, mut coeffs: [u32; N]) -> Result<()> {
        match self.map.entry(poly) {
            Entry::Vacant(slot) => {
                slot.insert(Box::new(coeffs));
                coeffs.zeroize();

                Ok(())
            }
            Entry::Occupied(slot) => {
                let same = bool::from(slot.get()[..].ct_eq(&coeffs[..]));

                coeffs.zeroize();

                same.then_some(()).ok_or(Error::Protocol {
                    protocol: "pqc_wiring",
                    message: "polynomial label produced twice with different coefficients",
                })
            }
        }
    }

    /// Returns the coefficients of `poly`.
    pub fn get(&self, poly: Poly) -> Result<&[u32; N]> {
        self.map
            .get(&poly)
            .map(|coeffs| &**coeffs)
            .ok_or(Error::Protocol {
                protocol: "pqc_wiring",
                message: "polynomial label read before it was produced",
            })
    }

    pub(crate) fn get_mut(&mut self, poly: Poly) -> Result<&mut [u32; N]> {
        self.map
            .get_mut(&poly)
            .map(|coeffs| &mut **coeffs)
            .ok_or(Error::Protocol {
                protocol: "pqc_wiring",
                message: "polynomial label read before it was produced",
            })
    }
}

impl Drop for PolyValues {
    fn drop(&mut self) {
        self.map.values_mut().for_each(|coeffs| coeffs.zeroize());
    }
}

/// Words of each stream, by label.
#[derive(Default)]
pub struct WordValues {
    map: BTreeMap<Stream, Vec<u32>>,
}

impl WordValues {
    /// Records the words of `stream` or checks a repeat against them.
    pub fn insert(&mut self, stream: Stream, mut words: Vec<u32>) -> Result<()> {
        match self.map.entry(stream) {
            Entry::Vacant(slot) => {
                slot.insert(words);

                Ok(())
            }
            Entry::Occupied(slot) => {
                let same = bool::from(slot.get().as_slice().ct_eq(words.as_slice()));

                words.zeroize();

                same.then_some(()).ok_or(Error::Protocol {
                    protocol: "pqc_wiring",
                    message: "word stream produced twice with different words",
                })
            }
        }
    }

    /// Returns the words of `stream`.
    pub fn get(&self, stream: Stream) -> Result<&[u32]> {
        self.map
            .get(&stream)
            .map(Vec::as_slice)
            .ok_or(Error::Protocol {
                protocol: "pqc_wiring",
                message: "word stream read before it was produced",
            })
    }
}

impl Drop for WordValues {
    fn drop(&mut self) {
        self.map.values_mut().for_each(Zeroize::zeroize);
    }
}

/// Lanes of each stream, by label.
#[derive(Default)]
pub struct LaneValues {
    map: BTreeMap<Stream, Vec<u64>>,
}

impl LaneValues {
    /// Records the lanes of `stream` or checks a repeat against them.
    pub fn insert(&mut self, stream: Stream, mut lanes: Vec<u64>) -> Result<()> {
        match self.map.entry(stream) {
            Entry::Vacant(slot) => {
                slot.insert(lanes);

                Ok(())
            }
            Entry::Occupied(slot) => {
                let same = bool::from(slot.get().as_slice().ct_eq(lanes.as_slice()));

                lanes.zeroize();

                same.then_some(()).ok_or(Error::Protocol {
                    protocol: "pqc_wiring",
                    message: "lane stream produced twice with different lanes",
                })
            }
        }
    }

    /// Returns the lanes of `stream`.
    pub fn get(&self, stream: Stream) -> Result<&[u64]> {
        self.map
            .get(&stream)
            .map(Vec::as_slice)
            .ok_or(Error::Protocol {
                protocol: "pqc_wiring",
                message: "lane stream read before it was produced",
            })
    }
}

impl Drop for LaneValues {
    fn drop(&mut self) {
        self.map.values_mut().for_each(Zeroize::zeroize);
    }
}

pub(crate) struct Pins<F> {
    segments: Vec<CadenceSegment<F>>,
}

impl<F: TowerField> Pins<F> {
    pub(crate) fn new() -> Self {
        Self {
            segments: Vec::new(),
        }
    }

    pub(crate) fn run(&mut self, origin: usize, len: usize, value: F) {
        if len == 0 || value == F::ZERO {
            return;
        }

        if let Some(last) = self.segments.last_mut()
            && last.stride == 1
            && last.values[0] == value
            && last.origin + last.count == origin
        {
            last.count += len;
            return;
        }

        self.segments.push(CadenceSegment {
            stride: 1,
            count: len,
            origin,
            values: vec![value],
        });
    }

    pub(crate) fn cadence(&mut self, origin: usize, count: usize, values: Vec<F>) {
        if count == 0 || values.iter().all(|v| *v == F::ZERO) {
            return;
        }

        self.segments.push(CadenceSegment {
            stride: values.len(),
            count,
            origin,
            values,
        });
    }

    pub(crate) fn shape(self) -> FixedShape<F> {
        if self.segments.is_empty() {
            FixedShape::Sparse(Vec::new())
        } else {
            FixedShape::Segments(self.segments)
        }
    }
}

/// Endpoint emitting `(poly, pos, value)` where `selector` is 1;
/// its waiver requires each nonzero `(poly, pos)` emitted once.
pub fn coef_spec(poly: usize, pos: usize, value: usize, selector: usize) -> PermutationCheckSpec {
    PermutationCheckSpec::new(
        vec![
            (Source::Column(poly), b"kappa_coef_poly" as &[u8]),
            (Source::Column(pos), b"kappa_coef_pos" as &[u8]),
            (Source::Column(value), b"kappa_coef_val" as &[u8]),
        ],
        Some(selector),
    )
    .with_clock_waiver(COEF_WAIVER)
}

/// Endpoint emitting `(stream, index, word)` where `selector`
/// is 1; its waiver requires `stream` and `index` pinned.
pub fn word_spec(
    stream: usize,
    index: usize,
    word: usize,
    selector: usize,
) -> PermutationCheckSpec {
    PermutationCheckSpec::new(
        vec![
            (Source::Column(stream), b"kappa_word_stream" as &[u8]),
            (Source::Column(index), b"kappa_word_index" as &[u8]),
            (Source::Column(word), b"kappa_word_value" as &[u8]),
        ],
        Some(selector),
    )
    .with_clock_waiver(WORD_WAIVER)
}

/// Endpoint emitting `(stream, index, lane)` where `selector`
/// is 1; its waiver requires `stream` and `index` pinned.
pub fn lane_spec(
    stream: usize,
    index: usize,
    lane: usize,
    selector: usize,
) -> PermutationCheckSpec {
    PermutationCheckSpec::new(
        vec![
            (Source::Column(stream), b"kappa_lane_stream" as &[u8]),
            (Source::Column(index), b"kappa_lane_index" as &[u8]),
            (Source::Column(lane), b"kappa_lane_value" as &[u8]),
        ],
        Some(selector),
    )
    .with_clock_waiver(LANE_WAIVER)
}

/// Endpoint emitting the hint key `(poly, pos)` where `selector`
/// is 1; its waiver requires each nonzero key emitted once.
pub fn hint_spec(poly: usize, pos: usize, selector: usize) -> PermutationCheckSpec {
    PermutationCheckSpec::new(
        vec![
            (Source::Column(poly), b"kappa_hint_poly" as &[u8]),
            (Source::Column(pos), b"kappa_hint_pos" as &[u8]),
        ],
        Some(selector),
    )
    .with_clock_waiver(HINT_WAIVER)
}

pub(crate) fn label<F: TowerField>(poly: Poly) -> F {
    F::from(poly.id() as u32)
}

/// Fails with `message` when `labels` holds a label twice:
/// two emissions of one key cancel in char 2.
pub(crate) fn distinct<T: Ord>(
    labels: impl IntoIterator<Item = T>,
    protocol: &'static str,
    message: &'static str,
) -> Result<()> {
    let mut seen = BTreeSet::new();

    match labels.into_iter().all(|l| seen.insert(l)) {
        true => Ok(()),
        false => Err(Error::Protocol { protocol, message }),
    }
}

/// Polynomial i of call `call` on the `hint` bus:
/// 16·call + i + 1. The Codec adds the call base in the field,
/// where addition is XOR, and XOR adds while i + 1 < 16.
pub(crate) fn hint_poly(call: u8, i: usize) -> u16 {
    16 * call as u16 + i as u16 + 1
}

pub(crate) fn bit<F: TowerField>(on: bool) -> F {
    if on { F::ONE } else { F::ZERO }
}

/// The shape pinning a column to `values`, one per row.
pub(crate) fn pinned_shape<F: TowerField>(values: impl IntoIterator<Item = u64>) -> FixedShape<F> {
    let mut pins = Pins::new();
    for (r, value) in values.into_iter().enumerate() {
        pins.run(r, 1, F::from(value as u128));
    }

    pins.shape()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hekate_math::Block128;

    type F = Block128;

    #[test]
    fn labels_start_at_one_and_never_repeat() {
        let mut labels = PolyLabels::new();

        let ids: Vec<u16> = (0..4).map(|_| labels.fresh().unwrap().id()).collect();
        assert_eq!(ids, [1, 2, 3, 4]);
    }

    #[test]
    fn labels_run_out_with_error() {
        let mut labels = PolyLabels {
            next: u32::from(u16::MAX),
        };

        assert_eq!(labels.fresh().unwrap().id(), u16::MAX);
        assert!(labels.fresh().is_err());
    }

    #[test]
    fn values_reject_double_production_and_early_reads() {
        let mut labels = PolyLabels::new();
        let p = labels.fresh().unwrap();
        let mut values = PolyValues::default();

        assert!(values.get(p).is_err());

        values.insert(p, [7; N]).unwrap();

        assert_eq!(values.get(p).unwrap()[0], 7);
        assert!(values.insert(p, [0; N]).is_err());
    }

    #[test]
    fn contiguous_equal_runs_merge() {
        let mut pins = Pins::<F>::new();

        pins.run(0, 4, F::ONE);
        pins.run(4, 2, F::ONE);
        pins.run(6, 3, F::ZERO);
        pins.run(9, 1, F::ONE);

        match pins.shape() {
            FixedShape::Segments(segs) => {
                assert_eq!(segs.len(), 2);
                assert_eq!((segs[0].origin, segs[0].count), (0, 6));
                assert_eq!((segs[1].origin, segs[1].count), (9, 1));
            }
            other => panic!("expected Segments, got {other:?}"),
        }
    }

    #[test]
    fn empty_pins_are_empty_sparse_shape() {
        let mut pins = Pins::<F>::new();
        pins.cadence(0, 3, vec![F::ZERO; 5]);

        assert_eq!(pins.shape(), FixedShape::Sparse(Vec::new()));
    }
}
