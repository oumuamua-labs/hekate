// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Converts ML-DSA and ML-KEM encodings on
//! the `word` bus to coefficients on the `coef`
//! bus and back, and emits the ML-DSA hint keys.

mod air;
mod layout;
mod trace;

pub use layout::CodecLayout;

use alloc::vec;
use alloc::vec::Vec;
use hekate_core::errors::{self, Error};
use hekate_core::trace::{ColumnTrace, TraceCompatibleField};
use hekate_math::{Flat, HardwareField, PackableField, TowerField};
use hekate_program::FixedShape;
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram, Col};
use zeroize::Zeroizing;

use crate::mldsa::MlDsaParams;
use crate::utils::gcd;
use crate::wiring::{
    COEF_BUS_ID, HINT_BUS_ID, N, Pins, Poly, PolyValues, Stream, WORD_BUS_ID, WordValues,
    coef_spec, distinct, hint_poly, hint_spec, label, word_spec,
};
use layout::{BYTE, HINT_SLOTS, KEM_WIDTH};

const T1_BITS: usize = 10;
const T1_SHIFT: usize = 13;
const T1_FIELDS: usize = 16;
const T1_WORDS: usize = T1_FIELDS * T1_BITS / 32;

const MAX_D: u32 = 11;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ZShape {
    pub(crate) width: usize,
    pub(crate) fields: usize,
    pub(crate) words: usize,
    pub(crate) gamma1: u32,
    pub(crate) beta: u32,
}

impl ZShape {
    fn new(params: &MlDsaParams) -> Self {
        let width = 32 - (2 * params.gamma1() - 1).leading_zeros() as usize;
        let (words, fields) = packing(width);

        Self {
            width,
            fields,
            words,
            gamma1: params.gamma1(),
            beta: params.beta(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HintShape {
    pub(crate) k: usize,
    pub(crate) omega: usize,
}

impl HintShape {
    fn rows(&self) -> usize {
        (self.omega + self.k).div_ceil(HINT_SLOTS)
    }
}

struct HintPins<F> {
    kind: Pins<F>,
    cont: Pins<F>,
    base: Pins<F>,
    idx: Vec<Pins<F>>,
    pad: Vec<Pins<F>>,
    pair: Vec<Pins<F>>,
    slot_value: Vec<Pins<F>>,
    count_role: Vec<Pins<F>>,
}

impl<F: TowerField> HintPins<F> {
    fn new(ly: &CodecLayout) -> Self {
        let k = ly.hint_cols.as_ref().map_or(0, |hc| hc.count_role.len());
        let per_slot = |n: usize| (0..n).map(|_| Pins::new()).collect();

        Self {
            kind: Pins::new(),
            cont: Pins::new(),
            base: Pins::new(),
            idx: per_slot(HINT_SLOTS),
            pad: per_slot(HINT_SLOTS),
            pair: per_slot(HINT_SLOTS),
            slot_value: per_slot(HINT_SLOTS),
            count_role: per_slot(k),
        }
    }

    fn step(&mut self, origin: usize, shape: HintShape, base: u16) {
        let rows = shape.rows();

        self.kind.run(origin, rows, F::ONE);
        self.cont.run(origin, rows - 1, F::ONE);
        self.base.run(origin, rows, F::from(base as u32));

        for r in 0..rows {
            for t in 0..HINT_SLOTS {
                let roles = (hint_role(shape, r, t), hint_role(shape, r, t + 1));

                self.slot(origin + r, t, roles);
            }
        }
    }

    fn slot(&mut self, row: usize, t: usize, roles: (HintRole, HintRole)) {
        match roles {
            (HintRole::Index(s), after) => {
                self.idx[t].run(row, 1, F::ONE);
                self.slot_value[t].run(row, 1, F::from(s as u32));

                if matches!(after, HintRole::Index(_)) {
                    self.pair[t].run(row, 1, F::ONE);
                }
            }
            (HintRole::Count(i), _) => self.count_role[i].run(row, 1, F::ONE),
            (HintRole::Pad, _) => self.pad[t].run(row, 1, F::ONE),
        }
    }

    fn into_columns(self, ly: &CodecLayout) -> Vec<(Col, FixedShape<F>)> {
        let Some(hc) = &ly.hint_cols else {
            return Vec::new();
        };

        let mut columns = vec![
            (hc.kind, self.kind.shape()),
            (hc.cont, self.cont.shape()),
            (hc.base, self.base.shape()),
        ];

        let slotwise = [
            (&hc.idx, self.idx),
            (&hc.pad, self.pad),
            (&hc.pair, self.pair),
            (&hc.slot_value, self.slot_value),
            (&hc.count_role, self.count_role),
        ];

        for (range, pins) in slotwise {
            for (c, p) in pins.into_iter().enumerate() {
                columns.push((range.at(c), p.shape()));
            }
        }

        columns
    }
}

/// What a slot of a hint row holds. HintBitUnpack (FIPS 204
/// Algorithm 21) reads ω position bytes, then k counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HintRole {
    /// Byte p of the ω position bytes.
    Index(usize),

    /// The running count that ends polynomial i's positions.
    Count(usize),

    /// A slot past the ω + k bytes.
    Pad,
}

/// The encoding a step converts. It fixes the field
/// width, the words per row and the checks on each value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// ML-DSA t1, 10-bit fields emitted times 2^13.
    T1,

    /// ML-DSA z, rejected at norm γ1 − β or above.
    Z(ZShape),

    /// ML-DSA hint bytes. The `u16` is a label base:
    /// keys of polynomial i carry label `base + i + 1`.
    Hint(HintShape, u16),

    /// ByteDecode12, with each value reduced mod q
    /// when `reduce`, or rejected at q or above otherwise.
    Decode12 { reduce: bool },

    /// ByteEncode12 of values below q.
    Encode12,

    /// ByteDecode_d then Decompress_d, for d from 1 to 11.
    Decompress(u32),

    /// Compress_d then ByteEncode_d, for d from 1 to 11.
    Compress(u32),
}

/// A cell of table row `row` on the other
/// witness that one check alone rejects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodecForgery {
    /// A 12-bit field left unreduced: flag 0, x = b ≥ q.
    Unreduced { row: usize, field: usize },

    /// A z field lifted by q: flag 1, z + q.
    Lifted { row: usize, field: usize },

    /// A compressed field one quotient short: y′ − 1, r + q.
    Borrowed { row: usize, field: usize },

    /// Hint count `poly` of a hint row set to `value`.
    Count { row: usize, poly: usize, value: u8 },
}

/// One decoding pass over a word stream,
/// laid out on consecutive table rows.
#[derive(Clone, Debug)]
pub struct CodecStep {
    kind: Kind,
    words: Stream,
    polys: Vec<Poly>,
    twin: Option<Stream>,
}

impl CodecStep {
    /// Unpacks the t1 part of a public key into `out`,
    /// emitting every coefficient multiplied by 2^13.
    pub fn t1(words: Stream, out: Vec<Poly>) -> Self {
        Self {
            kind: Kind::T1,
            words,
            polys: out,
            twin: None,
        }
    }

    /// Unpacks the z part of a signature into `out`. The table
    /// rejects a coefficient whose norm is at least γ1 − β.
    pub fn z(params: &MlDsaParams, words: Stream, out: Vec<Poly>) -> Self {
        Self {
            kind: Kind::Z(ZShape::new(params)),
            words,
            polys: out,
            twin: None,
        }
    }

    /// Unpacks the hint part of a signature and emits the key
    /// (16·call + i + 1, pos) of every set bit of polynomial i.
    pub fn hint(params: &MlDsaParams, words: Stream, call: u8) -> Self {
        Self {
            kind: Kind::Hint(
                HintShape {
                    k: params.k(),
                    omega: params.omega(),
                },
                hint_poly(call, 0) - 1,
            ),
            words,
            polys: Vec::new(),
            twin: None,
        }
    }

    /// Unpacks 12-bit fields into `out` with
    /// ByteDecode12, reducing each value mod q.
    pub fn decode12(words: Stream, out: Vec<Poly>) -> Self {
        Self {
            kind: Kind::Decode12 { reduce: true },
            words,
            polys: out,
            twin: None,
        }
    }

    /// Unpacks an encapsulation key into `out`. The table rejects
    /// a value at or above q, the FIPS 203 modulus check.
    pub fn decode12_canonical(words: Stream, out: Vec<Poly>) -> Self {
        Self {
            kind: Kind::Decode12 { reduce: false },
            words,
            polys: out,
            twin: None,
        }
    }

    /// Packs the coefficients of `input` into 12-bit fields
    /// with ByteEncode12. The table rejects a value at or above q.
    pub fn encode12(input: Vec<Poly>, words: Stream) -> Self {
        Self {
            kind: Kind::Encode12,
            words,
            polys: input,
            twin: None,
        }
    }

    /// Unpacks d-bit fields of `words` into `out` with
    /// ByteDecode_d then Decompress_d (FIPS 203 §4.2.1).
    pub fn decompress(d: u32, words: Stream, out: Vec<Poly>) -> errors::Result<Self> {
        Ok(Self {
            kind: Kind::Decompress(compression_width(d)?),
            words,
            polys: out,
            twin: None,
        })
    }

    /// Packs the coefficients of `input` into d-bit fields
    /// of `words` with Compress_d then ByteEncode_d
    /// (FIPS 203 §4.2.1). Every coefficient must be below q.
    pub fn compress(d: u32, input: Vec<Poly>, words: Stream) -> errors::Result<Self> {
        Ok(Self {
            kind: Kind::Compress(compression_width(d)?),
            words,
            polys: input,
            twin: None,
        })
    }

    /// Emits the step's words a second time
    /// under `twin`, for a second consumer.
    pub fn with_twin(mut self, twin: Stream) -> Self {
        self.twin = Some(twin);

        self
    }

    /// Table rows the step occupies.
    pub fn rows(&self) -> usize {
        match self.kind {
            Kind::Hint(shape, _) => shape.rows(),
            _ => self.polys.len() * N / self.fields(),
        }
    }

    fn fields(&self) -> usize {
        match self.kind {
            Kind::T1 => T1_FIELDS,
            Kind::Z(shape) => shape.fields,
            Kind::Hint(..) => 0,
            Kind::Decode12 { .. } | Kind::Encode12 => packing(KEM_WIDTH).1,
            Kind::Decompress(d) | Kind::Compress(d) => packing(d as usize).1,
        }
    }

    fn words_per_row(&self) -> usize {
        match self.kind {
            Kind::T1 => T1_WORDS,
            Kind::Z(shape) => shape.words,
            Kind::Hint(..) => 1,
            Kind::Decode12 { .. } | Kind::Encode12 => packing(KEM_WIDTH).0,
            Kind::Decompress(d) | Kind::Compress(d) => packing(d as usize).0,
        }
    }

    fn field_width(&self) -> usize {
        match self.kind {
            Kind::T1 => T1_BITS,
            Kind::Z(shape) => shape.width,
            Kind::Hint(..) => BYTE,
            Kind::Decode12 { .. } | Kind::Encode12 => KEM_WIDTH,
            Kind::Decompress(d) | Kind::Compress(d) => d as usize,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Kinds {
    pub(crate) t1: bool,
    pub(crate) z: Option<ZShape>,
    pub(crate) hint: Option<HintShape>,
    pub(crate) byte12: bool,
    pub(crate) decompress: u16,
    pub(crate) compress: u16,
    pub(crate) twin: bool,
}

impl Kinds {
    fn of(steps: &[CodecStep]) -> errors::Result<Self> {
        let mut kinds = Self::default();

        let mismatch = Error::Protocol {
            protocol: "codec_chiplet",
            message: "steps of one kind must share their parameter set",
        };

        for step in steps {
            match step.kind {
                Kind::T1 => kinds.t1 = true,
                Kind::Z(shape) => match kinds.z.replace(shape) {
                    Some(prev) if prev != shape => return Err(mismatch),
                    _ => {}
                },
                Kind::Hint(shape, _) => match kinds.hint.replace(shape) {
                    Some(prev) if prev != shape => return Err(mismatch),
                    _ => {}
                },
                Kind::Decode12 { .. } | Kind::Encode12 => kinds.byte12 = true,
                Kind::Decompress(d) => kinds.decompress |= 1 << d,
                Kind::Compress(d) => kinds.compress |= 1 << d,
            }

            if step.twin.is_some() {
                if !matches!(step.kind, Kind::Compress(_)) || step.words_per_row() != 1 {
                    return Err(Error::Protocol {
                        protocol: "codec_chiplet",
                        message: "twin words need a compress step with one word per row",
                    });
                }

                kinds.twin = true;
            }
        }

        Ok(kinds)
    }

    pub(crate) fn words(&self) -> usize {
        self.packings().map(|(words, _)| words).max().unwrap_or(0)
    }

    pub(crate) fn coefs(&self) -> usize {
        self.packings().map(|(_, fields)| fields).max().unwrap_or(0)
    }

    fn packings(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let t1 = self.t1.then_some((T1_WORDS, T1_FIELDS));
        let z = self.z.map(|shape| (shape.words, shape.fields));
        let hint = self.hint.map(|_| (1, 0));
        let byte12 = self.byte12.then(|| packing(KEM_WIDTH));

        let compressed = widths(self.decompress | self.compress).map(|d| packing(d as usize));

        [t1, z, hint, byte12]
            .into_iter()
            .flatten()
            .chain(compressed)
    }
}

/// Table coding ML-DSA and ML-KEM words with the checks
/// of FIPS 204 sigDecode and Verify and of FIPS 203.
#[derive(Clone)]
pub struct CodecChiplet<F: TowerField> {
    program: CircuitProgram<F>,
    steps: Vec<CodecStep>,
    kinds: Kinds,
    layout: CodecLayout,
    num_rows: usize,
}

impl<F> CodecChiplet<F>
where
    F: TowerField + TraceCompatibleField + PackableField + HardwareField + Send + 'static,
    <F as PackableField>::Packed: Copy + Send + Sync,
    Flat<F>: Send + Sync,
{
    /// A table holds steps of any kind but hint alone, one parameter
    /// set per kind and widths d from 1 to 11; anything else fails.
    pub fn new(steps: Vec<CodecStep>, num_rows: usize) -> errors::Result<Self> {
        let kinds = Kinds::of(&steps)?;

        if kinds.coefs() == 0 {
            return Err(Error::Protocol {
                protocol: "codec_chiplet",
                message: "steps carry no coefficient",
            });
        }

        if steps.iter().map(CodecStep::rows).sum::<usize>() > num_rows {
            return Err(Error::Protocol {
                protocol: "codec_chiplet",
                message: "steps need more rows than the table holds",
            });
        }

        let (encoders, decoders): (Vec<&CodecStep>, Vec<&CodecStep>) = steps
            .iter()
            .partition(|s| matches!(s.kind, Kind::Encode12 | Kind::Compress(_)));

        distinct(
            decoders.iter().map(|s| s.words),
            "codec_chiplet",
            "steps read a word stream twice",
        )?;

        distinct(
            encoders
                .iter()
                .flat_map(|s| [Some(s.words), s.twin])
                .flatten(),
            "codec_chiplet",
            "steps write a word stream twice",
        )?;

        distinct(
            encoders.iter().flat_map(|s| &s.polys),
            "codec_chiplet",
            "steps read a polynomial label twice",
        )?;

        distinct(
            decoders.iter().flat_map(|s| &s.polys),
            "codec_chiplet",
            "steps write a polynomial label twice",
        )?;

        distinct(
            steps.iter().filter_map(|s| match s.kind {
                Kind::Hint(_, base) => Some(base),
                _ => None,
            }),
            "codec_chiplet",
            "hint steps share a call base",
        )?;

        let mut cx = Circuit::<F>::new("CodecChiplet", num_rows)?;

        let layout = CodecLayout::declare(&mut cx, &kinds);

        for (col, shape) in pins(&steps, &layout) {
            cx.fix(col, shape);
        }

        let ly = &layout;

        for i in 0..ly.words {
            cx.bus(
                WORD_BUS_ID,
                word_spec(
                    ly.wstream.index(),
                    ly.widx.at(i).index(),
                    ly.word.at(i).index(),
                    ly.wsel.at(i).index(),
                ),
            );
        }

        if let Some((stream, sel)) = ly.twin {
            cx.bus(
                WORD_BUS_ID,
                word_spec(
                    stream.index(),
                    ly.widx.at(0).index(),
                    ly.word.at(0).index(),
                    sel.index(),
                ),
            );
        }

        for j in 0..ly.coefs {
            cx.bus(
                COEF_BUS_ID,
                coef_spec(
                    ly.poly.index(),
                    ly.pos.at(j).index(),
                    ly.coef.at(j).index(),
                    ly.csel.at(j).index(),
                ),
            );
        }

        if let Some(hc) = &ly.hint_cols {
            for t in 0..HINT_SLOTS {
                cx.bus(
                    HINT_BUS_ID,
                    hint_spec(hc.hp.at(t).index(), hc.hm.at(t).index(), hc.kind.index()),
                );
            }
        }

        air::constrain(cx.cs(), &kinds, &layout);

        let program = cx.compile()?;

        Ok(Self {
            program,
            steps,
            kinds,
            layout,
            num_rows,
        })
    }

    /// The table as a chiplet for a host program to attach.
    pub fn def(&self) -> errors::Result<ChipletDef<F>> {
        ChipletDef::from_air(&self.program)
    }

    /// The compiled table: constraints, bus endpoints and fixed columns.
    pub fn program(&self) -> &CircuitProgram<F> {
        &self.program
    }

    /// Column positions, for tests that forge cells of a trace.
    pub fn layout(&self) -> &CodecLayout {
        &self.layout
    }

    /// Traces the table, records each decoded polynomial in
    /// `values` and returns the hint vectors of the hint steps.
    pub fn trace(
        &self,
        words: &mut WordValues,
        values: &mut PolyValues,
    ) -> errors::Result<(ColumnTrace, Zeroizing<Vec<[bool; N]>>)> {
        trace::generate(self, words, values, true, &[])
    }

    /// Traces the table without the FIPS 203 and FIPS 204 checks,
    /// with the cells `forgeries` names on their other witness.
    /// A proof over a malformed encoding fails on the check it breaks.
    #[cfg(feature = "forgery")]
    pub fn trace_forged(
        &self,
        words: &mut WordValues,
        values: &mut PolyValues,
        forgeries: &[CodecForgery],
    ) -> errors::Result<(ColumnTrace, Zeroizing<Vec<[bool; N]>>)> {
        trace::generate(self, words, values, false, forgeries)
    }

    pub(crate) fn trace_checked(
        &self,
        words: &mut WordValues,
        values: &mut PolyValues,
        checked: bool,
    ) -> errors::Result<(ColumnTrace, Zeroizing<Vec<[bool; N]>>)> {
        trace::generate(self, words, values, checked, &[])
    }

    #[cfg(test)]
    pub(crate) fn produced(&self) -> Vec<(&'static str, u16)> {
        use crate::wiring::{COEF_BUS_ID, WORD_BUS_ID};

        self.steps
            .iter()
            .flat_map(|step| match step.kind {
                Kind::Encode12 | Kind::Compress(_) => step
                    .twin
                    .into_iter()
                    .chain([step.words])
                    .map(|words| (WORD_BUS_ID, words.id()))
                    .collect::<Vec<_>>(),
                Kind::Hint(..) => Vec::new(),
                _ => step.polys.iter().map(|p| (COEF_BUS_ID, p.id())).collect(),
            })
            .collect()
    }
}

pub(crate) fn hint_role(shape: HintShape, row: usize, slot: usize) -> HintRole {
    let p = HINT_SLOTS * row + slot;

    match p {
        p if p < shape.omega => HintRole::Index(p),
        p if p < shape.omega + shape.k => HintRole::Count(p - shape.omega),
        _ => HintRole::Pad,
    }
}

pub(crate) fn packing(width: usize) -> (usize, usize) {
    let words = width / gcd(width, 32);

    (words, 32 * words / width)
}

pub(crate) fn widths(mask: u16) -> impl Iterator<Item = u32> {
    (1..=MAX_D).filter(move |&d| (mask >> d) & 1 == 1)
}

fn compression_width(d: u32) -> errors::Result<u32> {
    match d {
        1..=MAX_D => Ok(d),
        _ => Err(Error::Protocol {
            protocol: "codec_chiplet",
            message: "compression width d must lie in 1..=11",
        }),
    }
}

fn pins<F: TowerField>(steps: &[CodecStep], ly: &CodecLayout) -> Vec<(Col, FixedShape<F>)> {
    let mut wsel: Vec<Pins<F>> = (0..ly.words).map(|_| Pins::new()).collect();
    let mut csel: Vec<Pins<F>> = (0..ly.coefs).map(|_| Pins::new()).collect();
    let mut pos: Vec<Pins<F>> = (0..ly.coefs).map(|_| Pins::new()).collect();
    let mut widx: Vec<Vec<(usize, F)>> = (0..ly.words).map(|_| Vec::new()).collect();

    let mut flags: Vec<(Col, Pins<F>)> = ly
        .kind_flags()
        .into_iter()
        .map(|col| (col, Pins::new()))
        .collect();

    let mut poly = Pins::new();
    let mut wstream = Pins::new();
    let mut twin_stream = Pins::new();
    let mut twin_sel = Pins::new();

    let mut hint = HintPins::new(ly);
    let mut origin = 0;

    for step in steps {
        let rows = step.rows();
        let (wpr, fields) = (step.words_per_row(), step.fields());

        wstream.run(origin, rows, F::from(step.words.id() as u32));

        if let Some(twin) = step.twin {
            twin_stream.run(origin, rows, F::from(twin.id() as u32));
            twin_sel.run(origin, rows, F::ONE);
        }

        for (i, pins) in wsel.iter_mut().enumerate().take(wpr) {
            pins.run(origin, rows, F::ONE);

            widx[i].extend(
                (0..rows)
                    .map(|r| (origin + r, F::from((wpr * r + i) as u32)))
                    .filter(|&(_, v)| v != F::ZERO),
            );
        }

        match step.kind {
            Kind::Hint(shape, base) => hint.step(origin, shape, base),
            _ => {
                let per_poly = N / fields;

                for (p, &out) in step.polys.iter().enumerate() {
                    poly.run(origin + p * per_poly, per_poly, label(out));
                }

                for j in 0..fields {
                    csel[j].run(origin, rows, F::ONE);
                    pos[j].cadence(
                        origin,
                        step.polys.len(),
                        (0..per_poly)
                            .map(|r| F::from((fields * r + j) as u32))
                            .collect(),
                    );
                }
            }
        }

        for col in ly.flags_of(step.kind) {
            if let Some((_, pins)) = flags.iter_mut().find(|(c, _)| *c == col) {
                pins.run(origin, rows, F::ONE);
            }
        }

        origin += rows;
    }

    let mut columns = Vec::new();

    for (i, pins) in wsel.into_iter().enumerate() {
        columns.push((ly.wsel.at(i), pins.shape()));
    }

    for (i, entries) in widx.into_iter().enumerate() {
        columns.push((ly.widx.at(i), FixedShape::Sparse(entries)));
    }

    for (j, (sel, p)) in csel.into_iter().zip(pos).enumerate() {
        columns.push((ly.csel.at(j), sel.shape()));
        columns.push((ly.pos.at(j), p.shape()));
    }

    columns.push((ly.poly, poly.shape()));
    columns.push((ly.wstream, wstream.shape()));

    columns.extend(flags.into_iter().map(|(col, pins)| (col, pins.shape())));
    columns.extend(hint.into_columns(ly));

    if let Some((stream, sel)) = ly.twin {
        columns.push((stream, twin_stream.shape()));
        columns.push((sel, twin_sel.shape()));
    }

    columns
}
