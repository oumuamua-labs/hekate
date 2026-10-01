// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

mod air;
mod layout;
mod trace;

use layout::CtrlLayout;

use alloc::vec::Vec;
use hekate_core::errors::{self, Error};
use hekate_core::trace::{ColumnTrace, TraceCompatibleField};
use hekate_keccak::KeccakChiplet;
use hekate_math::{Flat, HardwareField, PackableField, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram, Col};
use hekate_program::permutation::Service;
use zeroize::Zeroizing;

use crate::wiring::{
    LANE_BUS_ID, LaneValues, Stream, WORD_BUS_ID, WordValues, distinct, lane_spec, pinned_shape,
    word_spec,
};
use layout::TAILS;

pub(crate) const RATE: usize = 17;

const LANES: usize = 25;
const PREFIX: usize = 8;
const LAST_BYTE: u64 = 0x80 << 56;

pub(crate) type Token = (Stream, u16);

/// How a row's sponge state follows from the previous row's.
/// The 17 rate lanes form a ring: a turn moves lane 0 to
/// lane 16 and XORs the row's input and pad into it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Effect {
    /// Unchanged. On a `kec` row this state enters Keccak-f.
    #[default]
    Copy,

    /// All 25 lanes zero: a new sponge.
    Reset,

    /// Zero except rate lanes 9..=16, where the last
    /// 8 squeezed lanes sit: a sponge that opens with that
    /// 64-byte digest, as ML-DSA's μ = H(tr ‖ M′) opens with tr.
    Prefix,

    /// No turn: the row's word XORed into the low half of lane 0.
    Lo,

    /// A turn whose input is the row's word, in the high half.
    Hi,

    /// A turn whose input is the row's 64-bit lane.
    Lane,

    /// A turn with no input but the pad.
    Rotate,

    /// Keccak-f of the previous row's state, bound by
    /// the Keccak service in place of the ring constraint.
    Out,
}

impl Effect {
    fn rotates(self) -> bool {
        matches!(self, Self::Hi | Self::Lane | Self::Rotate)
    }
}

/// Whether a row's word crosses the service bus to the host.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Io {
    /// No; the row's word, if any, travels only on the word bus.
    #[default]
    None,

    /// The host's next request word.
    Input,

    /// A word read from the word bus and returned to the host.
    Output,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CtrlRow {
    effect: Effect,
    io: Io,
    kec: bool,
    split: bool,
    high: bool,
    emit: bool,
    word: Option<Token>,
    lane: Option<Token>,
    pad: u64,
    tail: u8,
}

pub(crate) struct CtrlTrace {
    pub(crate) trace: ColumnTrace,
    pub(crate) keccak: Zeroizing<Vec<[u64; LANES]>>,
}

pub(crate) struct Assembler {
    rows: Vec<CtrlRow>,
    rate: usize,
    phase: usize,
    offset: usize,
    squeezed: Option<usize>,
    fresh: usize,
    finished: bool,
}

impl Assembler {
    pub(crate) fn new() -> Self {
        Self {
            rows: Vec::new(),
            rate: RATE,
            phase: 0,
            offset: 0,
            squeezed: None,
            fresh: 0,
            finished: false,
        }
    }

    pub(crate) fn into_rows(self) -> Vec<CtrlRow> {
        self.rows
    }

    pub(crate) fn reset(&mut self, rate: usize) -> errors::Result<()> {
        if !(1..=RATE).contains(&rate) {
            return Err(schedule("sponge rate spans 1 to 17 lanes"));
        }

        self.rows.push(CtrlRow {
            effect: Effect::Reset,
            ..Default::default()
        });

        self.rate = rate;
        self.phase = 0;
        self.offset = 0;
        self.squeezed = None;
        self.fresh = RATE;
        self.finished = false;

        Ok(())
    }

    pub(crate) fn reset_prefix(&mut self) -> errors::Result<()> {
        if self.phase != PREFIX {
            return Err(schedule(
                "a prefix reset needs the squeezed prefix at the back of the ring",
            ));
        }

        self.rows.push(CtrlRow {
            effect: Effect::Prefix,
            ..Default::default()
        });

        self.offset = 8 * PREFIX;
        self.squeezed = None;
        self.fresh = RATE - PREFIX;
        self.finished = false;

        Ok(())
    }

    pub(crate) fn absorb_word(
        &mut self,
        source: Io,
        word: Option<Token>,
        lane: Option<Token>,
        bytes: usize,
    ) -> errors::Result<()> {
        if self.finished {
            return Err(schedule("absorb after a finish needs a reset"));
        }

        if !(1..=4).contains(&bytes) || !self.offset.is_multiple_of(4) {
            return Err(schedule(
                "absorbed word carries 1 to 4 bytes and follows whole words",
            ));
        }

        if source != Io::Input && word.is_none() {
            return Err(schedule("word absorbed off the word bus names its token"));
        }

        self.permute_if_full()?;

        let hi = self.offset % 8 == 4;

        if lane.is_some() && !hi {
            return Err(schedule("only the word that completes a lane emits it"));
        }

        if lane.is_some() && (bytes < 4 || self.fresh == 0) {
            return Err(schedule(
                "emitted lane needs a whole word in a slot the sponge has not filled",
            ));
        }

        self.rows.push(CtrlRow {
            effect: if hi { Effect::Hi } else { Effect::Lo },
            io: source,
            emit: lane.is_some(),
            word,
            lane,
            tail: if bytes < 4 { bytes as u8 } else { 0 },
            ..Default::default()
        });

        self.offset += bytes;
        self.squeezed = None;

        if hi {
            self.advance();
        }

        Ok(())
    }

    pub(crate) fn absorb_lane(&mut self, lane: Token) -> errors::Result<()> {
        if self.finished {
            return Err(schedule("absorb after a finish needs a reset"));
        }

        if !self.offset.is_multiple_of(8) {
            return Err(schedule("absorbed lane starts on a lane boundary"));
        }

        self.permute_if_full()?;

        self.rows.push(CtrlRow {
            effect: Effect::Lane,
            lane: Some(lane),
            ..Default::default()
        });

        self.offset += 8;
        self.squeezed = None;

        self.advance();

        Ok(())
    }

    pub(crate) fn finish(&mut self, suffix: &[u8]) -> errors::Result<()> {
        if self.finished {
            return Err(schedule("finish after a finish needs a reset"));
        }

        self.permute_if_full()?;

        let (lane, byte) = (self.offset / 8, self.offset % 8);

        if suffix.is_empty() || byte + suffix.len() > 8 {
            return Err(schedule("sponge suffix is 1 to 8 bytes inside one lane"));
        }

        let domain = suffix
            .iter()
            .rev()
            .fold(0u64, |acc, &b| acc << 8 | b as u64)
            << (8 * byte);

        let first = match byte {
            5..=7 => {
                self.patch(domain)?;

                lane + 1
            }
            _ => lane,
        };

        if first == self.rate {
            self.patch(LAST_BYTE)?;
        }

        for l in first..RATE {
            let mut pad = 0;

            if l == lane {
                pad ^= domain;
            }

            if l == self.rate - 1 {
                pad ^= LAST_BYTE;
            }

            self.rows.push(CtrlRow {
                effect: Effect::Rotate,
                pad,
                ..Default::default()
            });

            self.advance();
        }

        self.permute()?;

        self.finished = true;

        Ok(())
    }

    pub(crate) fn split(
        &mut self,
        lanes: usize,
        words: Stream,
        emit: Option<Stream>,
    ) -> errors::Result<()> {
        self.squeeze(lanes)?;

        for l in 0..lanes {
            self.rows.push(CtrlRow {
                split: true,
                word: Some((words, 2 * l as u16)),
                ..Default::default()
            });

            self.rows.push(CtrlRow {
                effect: Effect::Rotate,
                high: true,
                word: Some((words, 2 * l as u16 + 1)),
                emit: emit.is_some(),
                lane: emit.map(|s| (s, l as u16)),
                ..Default::default()
            });

            self.advance();
        }

        Ok(())
    }

    pub(crate) fn squeeze_lanes(&mut self, lanes: usize, emit: Stream) -> errors::Result<()> {
        self.squeeze(lanes)?;

        for l in 0..lanes {
            self.rows.push(CtrlRow {
                effect: Effect::Rotate,
                emit: true,
                lane: Some((emit, l as u16)),
                ..Default::default()
            });

            self.advance();
        }

        Ok(())
    }

    pub(crate) fn input(&mut self, word: Option<Token>) {
        self.rows.push(CtrlRow {
            io: Io::Input,
            word,
            ..Default::default()
        });
    }

    pub(crate) fn output(&mut self, word: Token) {
        self.rows.push(CtrlRow {
            io: Io::Output,
            word: Some(word),
            ..Default::default()
        });
    }

    fn permute(&mut self) -> errors::Result<()> {
        if self.phase != 0 {
            return Err(schedule("block permutes with its lanes in order"));
        }

        self.rows.push(CtrlRow {
            kec: true,
            ..Default::default()
        });
        self.rows.push(CtrlRow {
            effect: Effect::Out,
            kec: true,
            ..Default::default()
        });

        self.offset = 0;
        self.squeezed = Some(0);
        self.fresh = 0;

        Ok(())
    }

    fn permute_if_full(&mut self) -> errors::Result<()> {
        if self.offset < 8 * self.rate {
            return Ok(());
        }

        for _ in self.rate..RATE {
            self.rows.push(CtrlRow {
                effect: Effect::Rotate,
                ..Default::default()
            });

            self.advance();
        }

        self.permute()
    }

    fn squeeze(&mut self, lanes: usize) -> errors::Result<()> {
        match self.squeezed {
            Some(done) if done + lanes <= self.rate => {
                self.squeezed = Some(done + lanes);

                Ok(())
            }
            Some(_) => Err(schedule("squeeze reads at most the rate's lanes")),
            None => Err(schedule(
                "squeeze reads the lanes a permutation just produced",
            )),
        }
    }

    fn patch(&mut self, pad: u64) -> errors::Result<()> {
        match self.rows.last_mut() {
            Some(row) if row.effect.rotates() => {
                row.pad ^= pad;

                Ok(())
            }
            _ => Err(schedule("pad byte lands on a rotating row")),
        }
    }

    fn advance(&mut self) {
        self.phase = (self.phase + 1) % RATE;
        self.fresh = self.fresh.saturating_sub(1);
    }
}

#[derive(Clone)]
pub struct CtrlChiplet<F: TowerField> {
    program: CircuitProgram<F>,
    rows: Vec<CtrlRow>,
    layout: CtrlLayout,
    num_rows: usize,
}

impl<F> CtrlChiplet<F>
where
    F: TowerField + TraceCompatibleField + PackableField + HardwareField + Send + 'static,
    <F as PackableField>::Packed: Copy + Send + Sync,
    Flat<F>: Send + Sync,
{
    pub(crate) fn new(
        name: &str,
        service: &Service,
        rows: Vec<CtrlRow>,
        num_rows: usize,
    ) -> errors::Result<Self> {
        if rows.first().map(|r| r.effect) != Some(Effect::Reset) {
            return Err(schedule("control program starts with a full reset"));
        }

        if rows.len() > num_rows {
            return Err(schedule("program needs more rows than the table holds"));
        }

        distinct(
            rows.iter()
                .filter_map(|r| r.word.map(|t| (r.split || r.high, r.io == Io::Input, t))),
            "ctrl_chiplet",
            "word token appears twice in one role",
        )?;

        distinct(
            rows.iter().filter_map(|r| r.lane.map(|t| (r.emit, t))),
            "ctrl_chiplet",
            "lane token appears twice in one role",
        )?;

        let mut cx = Circuit::<F>::new(name, num_rows)?;

        let layout = CtrlLayout::declare(&mut cx);

        pinned(&rows, &layout, num_rows, |col, value| {
            cx.fix(col, pinned_shape((0..num_rows).map(value)));

            Ok(())
        })?;

        let ly = &layout;
        let state: Vec<Col> = ly.state.iter().collect();

        cx.call(&KeccakChiplet::service(), &state, ly.kec)?;

        cx.bus(
            service.bus_id,
            service.respond(&[ly.word.index()], ly.io.index())?,
        );
        cx.bus(
            WORD_BUS_ID,
            word_spec(
                ly.wstream.index(),
                ly.widx.index(),
                ly.word.index(),
                ly.wsel.index(),
            ),
        );
        cx.bus(
            LANE_BUS_ID,
            lane_spec(
                ly.lstream.index(),
                ly.lidx.index(),
                ly.lane.index(),
                ly.lsel.index(),
            ),
        );

        air::constrain(cx.cs(), &layout);

        let program = cx.compile()?;

        Ok(Self {
            program,
            rows,
            layout,
            num_rows,
        })
    }

    pub fn def(&self) -> errors::Result<ChipletDef<F>> {
        ChipletDef::from_air(&self.program)
    }

    pub(crate) fn blocks(&self) -> usize {
        self.rows
            .iter()
            .filter(|r| r.kec && r.effect == Effect::Copy)
            .count()
    }

    /// Words the host sends in, and words it gets back.
    pub(crate) fn host_words(&self) -> (usize, usize) {
        let count = |io: Io| self.rows.iter().filter(|r| r.io == io).count();

        (count(Io::Input), count(Io::Output))
    }

    pub(crate) fn trace(
        &self,
        inputs: &[u32],
        lanes: &LaneValues,
        words: &mut WordValues,
    ) -> errors::Result<CtrlTrace> {
        trace::generate(self, inputs, lanes, words)
    }

    #[cfg(test)]
    pub(crate) fn produced(&self) -> Vec<(&'static str, u16)> {
        use crate::wiring::{LANE_BUS_ID, WORD_BUS_ID};

        self.rows
            .iter()
            .flat_map(|row| {
                let word = row
                    .word
                    .filter(|_| row.io == Io::Input || row.split || row.high)
                    .map(|(stream, _)| (WORD_BUS_ID, stream.id()));

                let lane = row
                    .lane
                    .filter(|_| row.emit)
                    .map(|(stream, _)| (LANE_BUS_ID, stream.id()));

                word.into_iter().chain(lane)
            })
            .collect()
    }
}

pub(crate) fn pinned(
    rows: &[CtrlRow],
    ly: &CtrlLayout,
    num_rows: usize,
    mut column: impl FnMut(Col, &dyn Fn(usize) -> u64) -> errors::Result<()>,
) -> errors::Result<()> {
    let row = |r: usize| rows.get(r).copied().unwrap_or_default();
    let ahead = |r: usize| row((r + 1) % num_rows);

    let stream = |t: Option<Token>| t.map_or(0, |(s, _)| s.id() as u64);
    let index = |t: Option<Token>| t.map_or(0, |(_, i)| i as u64);
    let is = |r: CtrlRow, e: Effect| (r.effect == e) as u64;

    column(ly.io, &|r| (row(r).io != Io::None) as u64)?;
    column(ly.kec, &|r| row(r).kec as u64)?;
    column(ly.split, &|r| row(r).split as u64)?;
    column(ly.emit, &|r| row(r).emit as u64)?;
    column(ly.wsel, &|r| row(r).word.is_some() as u64)?;
    column(ly.lsel, &|r| row(r).lane.is_some() as u64)?;
    column(ly.wstream, &|r| stream(row(r).word))?;
    column(ly.widx, &|r| index(row(r).word))?;
    column(ly.lstream, &|r| stream(row(r).lane))?;
    column(ly.lidx, &|r| index(row(r).lane))?;
    column(ly.out, &|r| is(ahead(r), Effect::Out))?;
    column(ly.rot, &|r| ahead(r).effect.rotates() as u64)?;
    column(ly.reset, &|r| is(ahead(r), Effect::Reset))?;
    column(ly.prefix, &|r| is(ahead(r), Effect::Prefix))?;
    column(ly.lo, &|r| is(ahead(r), Effect::Lo))?;
    column(ly.hi, &|r| is(ahead(r), Effect::Hi))?;
    column(ly.lane_in, &|r| is(ahead(r), Effect::Lane))?;
    column(ly.pad, &|r| ahead(r).pad)?;

    for t in 0..TAILS {
        column(ly.tail.at(t), &|r| (row(r).tail as usize == t + 1) as u64)?;
    }

    Ok(())
}

fn schedule(message: &'static str) -> Error {
    Error::Protocol {
        protocol: "ctrl_chiplet",
        message,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use hekate_core::config::Config;
    use hekate_core::trace::{ColumnType, TraceBuilder, TraceColumn};
    use hekate_crypto::DefaultHasher;
    use hekate_crypto::transcript::Transcript;
    use hekate_keccak::generate_keccak_trace;
    use hekate_math::{Bit, Block16, Block32, Block64, Block128};
    use hekate_program::digest::program_id;
    use hekate_program::permutation::{BusKind, ServiceSlot};
    use hekate_program::{FixedShape, ProgramInstance, ProgramWitness};
    use hekate_prover_sys::prove;
    use hekate_sdk::preflight::{PreflightReport, preflight};
    use hekate_verifier::HekateVerifier;
    use sha3::{Digest, Sha3_256, Sha3_512};

    use crate::wiring::PolyLabels;

    type F = Block128;

    const SHA3: u8 = 0x06;
    const SHA3_512_RATE: usize = 9;
    const DIGEST_WORDS: usize = 16;

    const HOST_LAYOUT: [ColumnType; 6] = [
        ColumnType::B32,
        ColumnType::Bit,
        ColumnType::B16,
        ColumnType::B16,
        ColumnType::B64,
        ColumnType::Bit,
    ];

    struct Sha3Program {
        ctrl: CtrlChiplet<F>,
        inputs: Vec<u32>,
        digest: Stream,
        lanes: Stream,
    }

    struct GProgram {
        ctrl: CtrlChiplet<F>,
        inputs: Vec<u32>,
        h: Stream,
        m: Stream,
        key: Stream,
        seed: Stream,
    }

    struct Proof {
        program: CircuitProgram<F>,
        instance: ProgramInstance<F>,
        witness: ProgramWitness<F>,
    }

    impl Proof {
        fn report(&self) -> PreflightReport<F> {
            preflight(&self.program, &self.instance, &self.witness).unwrap()
        }

        fn accepted(&self, zero_knowledge: bool) -> bool {
            let config = Config {
                zero_knowledge,
                ..Config::prod()
            };

            let proof = prove(
                b"CtrlChiplet",
                &self.program,
                &self.instance,
                &self.witness,
                &config,
                [7; 32],
                None,
            )
            .unwrap();

            let mut transcript = Transcript::<DefaultHasher>::new(b"CtrlChiplet");

            HekateVerifier::<F, DefaultHasher>::verify(
                &program_id(&self.program).unwrap(),
                &self.program,
                &self.instance,
                &proof,
                &mut transcript,
                &config,
            )
            .unwrap_or(false)
        }
    }

    fn sha3_512(message: &[u8]) -> Sha3Program {
        let mut labels = PolyLabels::new();

        let digest = labels.stream().unwrap();
        let lanes = labels.stream().unwrap();

        let mut asm = Assembler::new();
        asm.reset(SHA3_512_RATE).unwrap();

        for chunk in message.chunks(4) {
            asm.absorb_word(Io::Input, None, None, chunk.len()).unwrap();
        }

        asm.finish(&[SHA3]).unwrap();
        asm.split(DIGEST_WORDS / 2, digest, Some(lanes)).unwrap();

        for j in 0..DIGEST_WORDS {
            asm.output((digest, j as u16));
        }

        let rows = asm.into_rows();
        let height = rows
            .len()
            .next_power_of_two()
            .max(Config::prod().min_table_rows());

        Sha3Program {
            ctrl: CtrlChiplet::new("Sha3Ctrl", &service(), rows, height).unwrap(),
            inputs: words(message),
            digest,
            lanes,
        }
    }

    fn g_program(x: &[u8], y: &[u8], k: u8) -> GProgram {
        let mut labels = PolyLabels::new();
        let [h, m, key, seed] = [(); 4].map(|_| labels.stream().unwrap());

        let mut asm = Assembler::new();

        for (message, digest) in [(x, h), (y, m)] {
            asm.reset(RATE).unwrap();

            for chunk in message.chunks(4) {
                asm.absorb_word(Io::Input, None, None, chunk.len()).unwrap();
            }

            asm.finish(&[SHA3]).unwrap();
            asm.split(4, digest, None).unwrap();
        }

        asm.reset(SHA3_512_RATE).unwrap();

        for j in 0..8 {
            asm.absorb_word(Io::None, Some((m, j)), None, 4).unwrap();
        }

        for j in 0..8 {
            asm.absorb_word(Io::Output, Some((h, j)), None, 4).unwrap();
        }

        asm.finish(&[k, SHA3]).unwrap();
        asm.split(4, key, None).unwrap();
        asm.squeeze_lanes(4, seed).unwrap();

        for j in 0..8 {
            asm.output((key, j));
        }

        let rows = asm.into_rows();
        let height = rows.len().next_power_of_two();

        GProgram {
            ctrl: CtrlChiplet::new("GCtrl", &service(), rows, height).unwrap(),
            inputs: [words(x), words(y)].concat(),
            h,
            m,
            key,
            seed,
        }
    }

    fn service() -> Service {
        Service {
            bus_id: "ctrl_test",
            kind: BusKind::Permutation,
            slots: vec![
                ServiceSlot::Value(b"kappa_ctrl_test_word"),
                ServiceSlot::EmitRank,
            ],
        }
    }

    fn message(len: usize) -> Vec<u8> {
        (0..len)
            .map(|i| (i as u8).wrapping_mul(0x9d) ^ 0x3c)
            .collect()
    }

    fn words(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks(4)
            .map(|chunk| {
                let mut word = [0u8; 4];
                word[..chunk.len()].copy_from_slice(chunk);

                u32::from_le_bytes(word)
            })
            .collect()
    }

    fn host(
        ctrl: &CtrlChiplet<F>,
        trace: CtrlTrace,
        hosted: &[u32],
        seed: Stream,
        lanes: &[u64],
    ) -> Proof {
        let floor = Config::prod().min_table_rows();
        let rows = hosted.len().max(lanes.len()).next_power_of_two().max(floor);

        let blocks = trace.keccak.len();
        let keccak_rows = (blocks * KeccakChiplet::BLOCK_ROWS)
            .next_power_of_two()
            .max(floor);

        let mut cx = Circuit::<F>::new("CtrlHost", rows).unwrap();

        let cols = cx.schema(&HOST_LAYOUT);

        cx.fix(
            cols.at(1),
            FixedShape::Cadence {
                stride: 1,
                count: hosted.len(),
                origin: 0,
                values: vec![F::ONE],
            },
        );

        cx.fix(
            cols.at(2),
            pinned_shape(vec![seed.id() as u64; lanes.len()]),
        );
        cx.fix(cols.at(3), pinned_shape(0..lanes.len() as u64));
        cx.fix(cols.at(5), pinned_shape(vec![1; lanes.len()]));

        cx.call(&service(), &[cols.at(0)], cols.at(1)).unwrap();

        cx.bus(LANE_BUS_ID, lane_spec(2, 3, 4, 5));

        cx.attach(ctrl.def().unwrap());
        cx.attach(ChipletDef::from_air(&KeccakChiplet::new(keccak_rows, blocks)).unwrap());

        let host = cx.compile().unwrap();

        let mut tb = TraceBuilder::new(&HOST_LAYOUT, rows.trailing_zeros() as usize).unwrap();

        for (r, &w) in hosted.iter().enumerate() {
            tb.set_b32(0, r, Block32::from(w)).unwrap();
            tb.set_bit(1, r, Bit::from(1u8)).unwrap();
        }

        for (r, &lane) in lanes.iter().enumerate() {
            tb.set_b16(2, r, Block16(seed.id())).unwrap();
            tb.set_b16(3, r, Block16(r as u16)).unwrap();
            tb.set_b64(4, r, Block64(lane)).unwrap();
            tb.set_bit(5, r, Bit::from(1u8)).unwrap();
        }

        let states: Vec<[Block64; LANES]> = trace.keccak.iter().map(|s| s.map(Block64)).collect();
        let keccak = generate_keccak_trace(&states, keccak_rows).unwrap();

        Proof {
            program: host,
            instance: ProgramInstance::new(rows, Vec::new()),
            witness: ProgramWitness::new(tb.build()).with_chiplets(vec![trace.trace, keccak]),
        }
    }

    fn lanes_of(words: &[u32]) -> Vec<u64> {
        words
            .chunks(2)
            .map(|w| w[0] as u64 | (w[1] as u64) << 32)
            .collect()
    }

    fn traced(program: &Sha3Program) -> (CtrlTrace, Vec<u32>) {
        let mut produced = WordValues::default();

        let ctrl = program
            .ctrl
            .trace(&program.inputs, &LaneValues::default(), &mut produced)
            .unwrap();

        (ctrl, produced.get(program.digest).unwrap().to_vec())
    }

    fn row_where(ctrl: &CtrlChiplet<F>, pick: impl Fn(&CtrlRow) -> bool) -> usize {
        ctrl.rows.iter().position(pick).unwrap()
    }

    fn set_word(trace: &mut ColumnTrace, col: usize, row: usize, value: u32) {
        let TraceColumn::B32(cells) = &mut trace.columns[col] else {
            panic!("forged words live in B32 columns");
        };

        cells[row] = Block32::from(value).to_hardware();
    }

    fn set_lane(trace: &mut ColumnTrace, col: usize, row: usize, value: u64) {
        let TraceColumn::B64(cells) = &mut trace.columns[col] else {
            panic!("forged lanes live in B64 columns");
        };

        cells[row] = Block64(value).to_hardware();
    }

    fn assert_breaks_only(proof: &Proof, label: &str) {
        let report = proof.report();

        assert!(report.fixed_column_violations.is_empty());
        assert!(report.boundary_violations.is_empty());
        assert!(report.bus_diagnostics.iter().all(|d| !d.has_failures()));
        assert!(!report.constraint_violations.is_empty());
        assert!(
            report
                .constraint_violations
                .iter()
                .all(|v| v.label == Some(label))
        );
        assert!(!proof.accepted(false));
        assert!(!proof.accepted(true));
    }

    #[test]
    fn sha3_512_on_17_lane_ring_matches_reference() {
        for len in [0, 3, 33, 64, 68, 71, 72, 100, 143, 144, 150] {
            let m = message(len);
            let program = sha3_512(&m);

            let mut produced = WordValues::default();

            program
                .ctrl
                .trace(&program.inputs, &LaneValues::default(), &mut produced)
                .unwrap();

            let expected: Vec<u32> = Sha3_512::digest(&m)
                .chunks(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            assert_eq!(
                produced.get(program.digest).unwrap().to_vec(),
                expected,
                "length {len}"
            );
        }
    }

    #[test]
    fn sha3_512_program_satisfies_ctrl_air() {
        for len in [64, 150] {
            let program = sha3_512(&message(len));

            let mut produced = WordValues::default();

            let ctrl = program
                .ctrl
                .trace(&program.inputs, &LaneValues::default(), &mut produced)
                .unwrap();

            let digest = produced.get(program.digest).unwrap().to_vec();
            let hosted = [program.inputs.clone(), digest.clone()].concat();

            let proof = host(
                &program.ctrl,
                ctrl,
                &hosted,
                program.lanes,
                &lanes_of(&digest),
            );

            assert!(proof.report().is_clean());
        }
    }

    #[test]
    fn g_over_internal_words_matches_reference() {
        let (x, y, k) = (message(41), message(77), 3);
        let program = g_program(&x, &y, k);

        let mut produced = WordValues::default();

        program
            .ctrl
            .trace(&program.inputs, &LaneValues::default(), &mut produced)
            .unwrap();

        let h = Sha3_256::digest(&x);
        let m = Sha3_256::digest(&y);
        let g = Sha3_512::digest([m.as_slice(), h.as_slice(), &[k]].concat());

        assert_eq!(produced.get(program.h).unwrap(), words(&h));
        assert_eq!(produced.get(program.m).unwrap(), words(&m));
        assert_eq!(produced.get(program.key).unwrap(), words(&g[..32]));
    }

    #[test]
    fn g_over_internal_words_satisfies_ctrl_air() {
        let (x, y, k) = (message(41), message(77), 3);
        let program = g_program(&x, &y, k);

        let mut produced = WordValues::default();

        let ctrl = program
            .ctrl
            .trace(&program.inputs, &LaneValues::default(), &mut produced)
            .unwrap();

        let h = produced.get(program.h).unwrap().to_vec();
        let key = produced.get(program.key).unwrap().to_vec();

        let hosted = [program.inputs.clone(), h, key].concat();

        let g = Sha3_512::digest(
            [
                Sha3_256::digest(&y).as_slice(),
                Sha3_256::digest(&x).as_slice(),
                &[k],
            ]
            .concat(),
        );

        let proof = host(
            &program.ctrl,
            ctrl,
            &hosted,
            program.seed,
            &lanes_of(&words(&g[32..])),
        );

        assert!(proof.report().is_clean());
    }

    #[test]
    fn host_word_off_sponge_breaks_only_ctrl_sponge() {
        let program = sha3_512(&message(64));
        let (mut ctrl, digest) = traced(&program);

        let ly = &program.ctrl.layout;
        let row = row_where(&program.ctrl, |r| r.io == Io::Input);

        let mut inputs = program.inputs.clone();
        inputs[0] ^= 1 << 9;

        set_word(
            &mut ctrl.trace,
            ly.physical(ly.word).unwrap(),
            row,
            inputs[0],
        );

        let hosted = [inputs, digest.clone()].concat();
        let proof = host(
            &program.ctrl,
            ctrl,
            &hosted,
            program.lanes,
            &lanes_of(&digest),
        );

        assert_breaks_only(&proof, "ctrl_sponge");
    }

    #[test]
    fn digest_word_off_state_breaks_only_ctrl_split() {
        let program = sha3_512(&message(64));
        let (mut ctrl, mut digest) = traced(&program);

        let ly = &program.ctrl.layout;
        let word = ly.physical(ly.word).unwrap();
        let lanes = lanes_of(&digest);

        digest[0] ^= 1 << 5;

        for row in [
            row_where(&program.ctrl, |r| r.split),
            row_where(&program.ctrl, |r| r.io == Io::Output),
        ] {
            set_word(&mut ctrl.trace, word, row, digest[0]);
        }

        let hosted = [program.inputs.clone(), digest].concat();
        let proof = host(&program.ctrl, ctrl, &hosted, program.lanes, &lanes);

        assert_breaks_only(&proof, "ctrl_split");
    }

    #[test]
    fn emitted_lane_off_state_breaks_only_ctrl_lane() {
        let program = sha3_512(&message(64));
        let (mut ctrl, digest) = traced(&program);

        let ly = &program.ctrl.layout;
        let row = row_where(&program.ctrl, |r| r.emit);

        let mut lanes = lanes_of(&digest);
        lanes[0] ^= 1 << 40;

        set_lane(
            &mut ctrl.trace,
            ly.physical(ly.lane).unwrap(),
            row,
            lanes[0],
        );

        let hosted = [program.inputs.clone(), digest].concat();
        let proof = host(&program.ctrl, ctrl, &hosted, program.lanes, &lanes);

        assert_breaks_only(&proof, "ctrl_lane");
    }

    #[test]
    fn token_in_one_role_twice_is_refused() {
        let token = (PolyLabels::new().stream().unwrap(), 0);

        let twice = |absorb: &dyn Fn(&mut Assembler)| {
            let mut asm = Assembler::new();

            asm.reset(RATE).unwrap();

            absorb(&mut asm);
            absorb(&mut asm);

            asm.finish(&[SHA3]).unwrap();

            let rows = asm.into_rows();
            let height = rows.len().next_power_of_two();

            CtrlChiplet::<F>::new("TwiceCtrl", &service(), rows, height).err()
        };

        assert_eq!(
            twice(&|asm| asm.absorb_word(Io::None, Some(token), None, 4).unwrap()),
            Some(schedule("word token appears twice in one role"))
        );
        assert_eq!(
            twice(&|asm| asm.absorb_lane(token).unwrap()),
            Some(schedule("lane token appears twice in one role"))
        );
    }

    #[test]
    fn finish_after_finish_needs_reset() {
        let mut asm = Assembler::new();

        asm.reset(RATE).unwrap();
        asm.absorb_word(Io::Input, None, None, 4).unwrap();
        asm.finish(&[SHA3]).unwrap();

        assert_eq!(
            asm.finish(&[SHA3]),
            Err(schedule("finish after a finish needs a reset"))
        );

        asm.reset(RATE).unwrap();

        assert!(asm.finish(&[SHA3]).is_ok());
    }

    #[test]
    fn lane_emit_needs_whole_word_in_unfilled_slot() {
        let lane = (PolyLabels::new().stream().unwrap(), 0);
        let refused = Err(schedule(
            "emitted lane needs a whole word in a slot the sponge has not filled",
        ));

        let mut short = Assembler::new();

        short.reset(RATE).unwrap();
        short.absorb_word(Io::Input, None, None, 4).unwrap();

        assert_eq!(short.absorb_word(Io::Input, None, Some(lane), 3), refused);

        let mut permuted = Assembler::new();
        permuted.reset(SHA3_512_RATE).unwrap();

        for _ in 0..2 * SHA3_512_RATE {
            permuted.absorb_word(Io::Input, None, None, 4).unwrap();
        }

        permuted.absorb_word(Io::Input, None, None, 4).unwrap();

        assert_eq!(
            permuted.absorb_word(Io::Input, None, Some(lane), 4),
            refused
        );
    }

    #[test]
    fn absorb_after_finish_needs_reset() {
        let lane = (PolyLabels::new().stream().unwrap(), 0);
        let refused = Err(schedule("absorb after a finish needs a reset"));

        let mut asm = Assembler::new();

        asm.reset(RATE).unwrap();
        asm.absorb_word(Io::Input, None, None, 4).unwrap();
        asm.finish(&[SHA3]).unwrap();

        assert_eq!(asm.absorb_word(Io::Input, None, None, 4), refused);
        assert_eq!(asm.absorb_lane(lane), refused);

        asm.reset(RATE).unwrap();

        assert!(asm.absorb_word(Io::Input, None, None, 4).is_ok());
    }
}
