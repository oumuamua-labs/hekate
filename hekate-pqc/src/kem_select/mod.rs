// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! ML-KEM Decaps' implicit rejection (FIPS 203 Algorithm 18):
//! c against the re-encryption c', and the choice of K' or K̄.

mod air;
mod layout;
mod trace;

pub use layout::KemSelectLayout;

use alloc::vec;
use alloc::vec::Vec;
use hekate_core::errors::{self, Error};
use hekate_core::trace::{ColumnTrace, TraceCompatibleField};
use hekate_math::{Flat, HardwareField, PackableField, TowerField};
use hekate_program::FixedShape;
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram, Col};

use crate::wiring::{Pins, Stream, WORD_BUS_ID, WordValues, distinct, word_spec};

const KEY_WORDS: usize = 8;
const MAX_PART_WORDS: usize = 1 << 16;

/// One Decaps: K is K' when c equals c' word by word
/// and K̄ otherwise, and valid says which. Each part
/// of c goes on to the decrypt Codec under its `relay`.
#[derive(Clone, Debug)]
pub struct KemSelectStep {
    pub parts: Vec<CipherPart>,
    pub k_prime: Stream,
    pub k_bar: Stream,
    pub k: Stream,
    pub valid: Stream,
}

impl KemSelectStep {
    /// Table rows the step occupies.
    pub fn rows(&self) -> usize {
        self.words() + KEY_WORDS + 1
    }

    fn words(&self) -> usize {
        self.parts.iter().map(|p| p.words).sum()
    }

    fn streams(&self) -> impl Iterator<Item = Stream> + '_ {
        self.parts
            .iter()
            .flat_map(|p| [p.c, p.c_prime, p.relay])
            .chain([self.k_prime, self.k_bar, self.k, self.valid])
    }
}

/// c1 or c2: the streams of c and c′, compared word by
/// word, and the stream relaying c to the decrypt Codec.
#[derive(Clone, Copy, Debug)]
pub struct CipherPart {
    pub c: Stream,
    pub c_prime: Stream,
    pub relay: Stream,
    pub words: usize,
}

/// Table proving the implicit rejection of a list of Decaps
/// steps, with every word read from and written to the word bus.
#[derive(Clone)]
pub struct KemSelectChiplet<F: TowerField> {
    program: CircuitProgram<F>,
    steps: Vec<KemSelectStep>,
    layout: KemSelectLayout,
    num_rows: usize,
}

impl<F> KemSelectChiplet<F>
where
    F: TowerField + TraceCompatibleField + PackableField + HardwareField + Send + 'static,
    <F as PackableField>::Packed: Copy + Send + Sync,
    Flat<F>: Send + Sync,
{
    /// A table holds any list of steps whose
    /// ciphertext parts each hold 1 to 65536 words.
    pub fn new(steps: Vec<KemSelectStep>, num_rows: usize) -> errors::Result<Self> {
        if steps
            .iter()
            .any(|s| s.parts.is_empty() || s.parts.iter().any(|p| p.words == 0))
        {
            return Err(Error::Protocol {
                protocol: "kem_select_chiplet",
                message: "ciphertext of a step has an empty part",
            });
        }

        if steps
            .iter()
            .flat_map(|s| &s.parts)
            .any(|p| p.words > MAX_PART_WORDS)
        {
            return Err(Error::Protocol {
                protocol: "kem_select_chiplet",
                message: "ciphertext part holds more words than a 16-bit index counts",
            });
        }

        if steps.iter().map(KemSelectStep::rows).sum::<usize>() > num_rows {
            return Err(Error::Protocol {
                protocol: "kem_select_chiplet",
                message: "steps need more rows than the table holds",
            });
        }

        distinct(
            steps.iter().flat_map(KemSelectStep::streams),
            "kem_select_chiplet",
            "steps name a word stream twice",
        )?;

        let mut cx = Circuit::<F>::new("KemSelectChiplet", num_rows)?;

        let layout = KemSelectLayout::declare(&mut cx);

        for (col, shape) in pins(&steps, &layout) {
            cx.fix(col, shape);
        }

        let ly = &layout;

        for (stream, word, sel) in [
            (ly.stream_a, ly.word_a, ly.sel_ab),
            (ly.stream_b, ly.word_b, ly.sel_ab),
            (ly.stream_c, ly.word_c, ly.sel_c),
        ] {
            cx.bus(
                WORD_BUS_ID,
                word_spec(stream.index(), ly.idx.index(), word.index(), sel.index()),
            );
        }

        air::constrain(cx.cs(), &layout);

        let program = cx.compile()?;

        Ok(Self {
            program,
            steps,
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
    pub fn layout(&self) -> &KemSelectLayout {
        &self.layout
    }

    /// Reads c, c', K' and K̄ from `words`, writes K, valid
    /// and each part's `relay` there, and returns the trace.
    pub fn trace(&self, words: &mut WordValues) -> errors::Result<ColumnTrace> {
        trace::generate(
            &self.program,
            &self.steps,
            &self.layout,
            self.num_rows,
            words,
        )
    }

    #[cfg(test)]
    pub(crate) fn produced(&self) -> Vec<(&'static str, u16)> {
        self.steps
            .iter()
            .flat_map(|step| {
                step.parts
                    .iter()
                    .map(|part| part.relay)
                    .chain([step.k, step.valid])
            })
            .map(|words| (WORD_BUS_ID, words.id()))
            .collect()
    }
}

fn pins<F: TowerField>(steps: &[KemSelectStep], ly: &KemSelectLayout) -> Vec<(Col, FixedShape<F>)> {
    let mut stream_a = Pins::new();
    let mut stream_b = Pins::new();
    let mut stream_c = Pins::new();
    let mut idx = Pins::new();
    let mut sel_ab = Pins::new();
    let mut sel_c = Pins::new();
    let mut compare = Pins::new();
    let mut select = Pins::new();
    let mut report = Pins::new();
    let mut first = Pins::new();
    let mut chain = Pins::new();

    let id = |s: Stream| F::from(s.id() as u32);
    let count = |n: usize| (0..n as u32).map(F::from).collect::<Vec<F>>();

    let mut origin = 0;

    for s in steps {
        let keys = origin + s.words();
        let end = keys + KEY_WORDS;

        let mut at = origin;

        for part in &s.parts {
            stream_a.run(at, part.words, id(part.c));
            stream_b.run(at, part.words, id(part.c_prime));
            stream_c.run(at, part.words, id(part.relay));

            idx.cadence(at, 1, count(part.words));

            at += part.words;
        }

        stream_a.run(keys, KEY_WORDS, id(s.k_prime));
        stream_b.run(keys, KEY_WORDS, id(s.k_bar));
        stream_c.run(keys, KEY_WORDS, id(s.k));
        stream_c.run(end, 1, id(s.valid));

        idx.cadence(keys, 1, count(KEY_WORDS));

        sel_ab.run(origin, s.words() + KEY_WORDS, F::ONE);
        sel_c.run(origin, s.words() + KEY_WORDS + 1, F::ONE);
        compare.run(origin, s.words(), F::ONE);
        select.run(keys, KEY_WORDS, F::ONE);
        report.run(end, 1, F::ONE);
        first.run(origin, 1, F::ONE);
        chain.run(origin, s.words() + KEY_WORDS, F::ONE);

        origin = end + 1;
    }

    vec![
        (ly.stream_a, stream_a.shape()),
        (ly.stream_b, stream_b.shape()),
        (ly.stream_c, stream_c.shape()),
        (ly.idx, idx.shape()),
        (ly.sel_ab, sel_ab.shape()),
        (ly.sel_c, sel_c.shape()),
        (ly.compare, compare.shape()),
        (ly.select, select.shape()),
        (ly.report, report.shape()),
        (ly.first, first.shape()),
        (ly.chain, chain.shape()),
    ]
}
