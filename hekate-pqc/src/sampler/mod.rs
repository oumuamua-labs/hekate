// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Polynomials sampled from seed streams for ML-DSA and ML-KEM:
//! RejNTTPoly, SampleNTT, SamplePolyCBD and SampleInBall, over SHAKE
//! blocks the Keccak table computes, emitted on the `coef` bus.

mod air;
mod layout;
mod trace;

pub use layout::SamplerLayout;

use alloc::vec;
use alloc::vec::Vec;
use hekate_core::errors::{self, Error};
use hekate_core::trace::{ColumnTrace, TraceCompatibleField};
use hekate_keccak::KeccakChiplet;
use hekate_math::{Flat, HardwareField, PackableField, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram, Col};
use hekate_program::permutation::{PermutationCheckSpec, Source};
use zeroize::Zeroizing;

use crate::mldsa::MlDsaParams;
use crate::wiring::{
    COEF_BUS_ID, LANE_BUS_ID, LaneValues, N, Poly, PolyValues, Stream, coef_spec, distinct,
    lane_spec, pinned_shape,
};
use crate::{mldsa, mlkem};
use layout::{BYTE, LANES, WINDOW};

pub(crate) const SIB_BUS_ID: &str = "sampler_sib";

const SEED_BYTES: usize = 32;
const SIGN_BYTES: usize = 8;
const SHAKE_PAD: u8 = 0x1f;
const PAD_LAST: u8 = 0x80;
const RATE_128: usize = 21;
const RATE_256: usize = 17;
const TAIL_LANES: usize = 2;
const STEP_ACCESSES: usize = 3;

const BALL_BUDGETS: [(usize, usize); 3] = [(39, 91), (49, 111), (60, 136)];

const SIB_WAIVER: &str = "see hekate-pqc/src/sampler/air.rs: every access carries its step's \
     pinned label and a fixed time unique within the step, and each sorted row consumes exactly one \
     in strict (address, time) order";

type Column<'c> = dyn FnMut(Col, &dyn Fn(usize) -> u64) -> errors::Result<()> + 'c;

/// What a row does in its step: carry a Keccak-f call,
/// read a window of the output or run SampleInBall's memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RowKind {
    /// Input of the step's next Keccak-f call: the padded
    /// seed for block 0, the previous output after that.
    In,

    /// Output of that call. A capture row also keeps
    /// the η = 3 tail lanes or the SampleInBall sign lane.
    Out,

    /// A 3-lane window at rate lane `offset`, or tail plus
    /// lane 0 when `None`. `index` numbers the window,
    /// or the candidate byte in SampleInBall.
    Work { offset: Option<usize>, index: usize },

    /// SampleInBall step t: c_i <- c_j, then c_j <- ±1,
    /// for i = 256 − τ + t, in three memory accesses.
    Step(usize),

    /// Cell p of the finished challenge, emitted on `coef`.
    Readout(usize),

    /// Access s in (address, time) order, which the
    /// offline memory check matches against program order.
    Sorted(usize),
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RowInfo {
    pub(crate) step: usize,
    pub(crate) block: usize,
    pub(crate) kind: RowKind,
    pub(crate) first: bool,
    pub(crate) capture: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RejShape {
    pub(crate) width: usize,
    pub(crate) stride: usize,
    pub(crate) slots: usize,
    pub(crate) q: u32,
    pub(crate) blocks: usize,
}

impl RejShape {
    const REJ_NTT: Self = Self::new(23, 24, mldsa::Q, 5);
    const SAMPLE_NTT: Self = Self::new(12, 12, mlkem::Q, 4);

    const fn new(width: usize, stride: usize, q: u32, blocks: usize) -> Self {
        Self {
            width,
            stride,
            slots: WINDOW * 64 / stride,
            q,
            blocks,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BallShape {
    pub(crate) tau: usize,
    pub(crate) budget: usize,
    pub(crate) seed_lanes: usize,
}

impl BallShape {
    /// τ, the byte budget and the c̃ lanes of `params`;
    /// fails for a τ with no checked-in budget.
    fn new(params: &MlDsaParams) -> errors::Result<Self> {
        let tau = params.tau();

        let Some(&(_, budget)) = BALL_BUDGETS.iter().find(|&&(t, _)| t == tau) else {
            return Err(Error::Protocol {
                protocol: "sampler_chiplet",
                message: "no SampleInBall byte budget for this tau",
            });
        };

        Ok(Self {
            tau,
            budget,
            seed_lanes: params.lambda() / 32,
        })
    }

    /// 256 − τ, the first cell a step writes and the start of i.
    pub(crate) fn first_cell(&self) -> usize {
        N - self.tau
    }

    /// Memory accesses of one SampleInBall: 3 per step, 1 per readout.
    pub(crate) fn accesses(&self) -> usize {
        STEP_ACCESSES * self.tau + N
    }
}

/// The algorithm a step runs, which
/// fixes its XOF, rate and row plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// Rejection sampling of NTT-domain coefficients from SHAKE128:
    /// RejNTTPoly (FIPS 204) or SampleNTT (FIPS 203).
    Rej(RejShape),

    /// SamplePolyCBD_η over SHAKE256, for η of 2 or 3.
    Cbd(usize),

    /// SampleInBall over SHAKE256 of c̃.
    Ball(BallShape),
}

/// One sampling pass over a seed stream,
/// laid out on consecutive table rows.
#[derive(Clone, Debug)]
pub struct SamplerStep {
    kind: Kind,
    seed: Stream,
    suffix: Vec<u8>,
    out: Poly,
}

impl SamplerStep {
    /// Samples `Â[r][s]` into `out` with RejNTTPoly(ρ‖s‖r),
    /// FIPS 204 Algorithm 30, over 5 SHAKE128 blocks.
    pub fn expand_a(seed: Stream, r: u8, s: u8, out: Poly) -> Self {
        Self {
            kind: Kind::Rej(RejShape::REJ_NTT),
            seed,
            suffix: vec![s, r],
            out,
        }
    }

    /// Samples `Â[i][j]` into `out` with SampleNTT(ρ‖j‖i),
    /// FIPS 203 Algorithm 7, over 4 SHAKE128 blocks.
    pub fn sample_ntt(seed: Stream, i: u8, j: u8, out: Poly) -> Self {
        Self {
            kind: Kind::Rej(RejShape::SAMPLE_NTT),
            seed,
            suffix: vec![j, i],
            out,
        }
    }

    /// Samples `out` with SamplePolyCBD_η(PRF_η(s, n)),
    /// FIPS 203 Algorithm 8, for η of 2 or 3.
    pub fn prf(seed: Stream, n: u8, eta: u32, out: Poly) -> Self {
        Self {
            kind: Kind::Cbd(eta as usize),
            seed,
            suffix: vec![n],
            out,
        }
    }

    /// Samples the challenge c into `out` with SampleInBall(c̃),
    /// FIPS 204 Algorithm 29; the stream carries c̃.
    /// Fails for a τ with no checked-in byte budget.
    pub fn sample_in_ball(params: &MlDsaParams, seed: Stream, out: Poly) -> errors::Result<Self> {
        Ok(Self {
            kind: Kind::Ball(BallShape::new(params)?),
            seed,
            suffix: Vec::new(),
            out,
        })
    }

    /// Table rows the step occupies.
    pub fn rows(&self) -> usize {
        self.plan().len()
    }

    /// Keccak permutations the step requests.
    pub fn blocks(&self) -> usize {
        self.plan()
            .iter()
            .filter(|&&(_, kind, _)| kind == RowKind::In)
            .count()
    }

    pub(crate) fn sample(&self, seeds: &LaneValues, values: &mut PolyValues) -> errors::Result<()> {
        values.insert(self.out, trace::coefficients(self, seeds)?)
    }

    /// Lanes of seed the step absorbs:
    /// λ/32 for SampleInBall, 4 otherwise.
    fn seed_lanes(&self) -> usize {
        match self.kind {
            Kind::Ball(shape) => shape.seed_lanes,
            Kind::Rej(_) | Kind::Cbd(_) => SEED_BYTES / 8,
        }
    }

    fn rate(&self) -> usize {
        match self.kind {
            Kind::Rej(_) => RATE_128,
            Kind::Cbd(_) | Kind::Ball(_) => RATE_256,
        }
    }

    /// Bytes the step squeezes:
    /// whole blocks for rejection, 64·η for CBD,
    /// the sign lane plus the byte budget for SampleInBall.
    fn squeeze_bytes(&self) -> usize {
        match self.kind {
            Kind::Rej(shape) => shape.blocks * RATE_128 * 8,
            Kind::Cbd(eta) => 64 * eta,
            Kind::Ball(shape) => SIGN_BYTES + shape.budget,
        }
    }

    /// The step's rows in table order, each as (Keccak block,
    /// row kind, whether its OUT feeds the tail or the sign lane).
    fn plan(&self) -> Vec<(usize, RowKind, bool)> {
        let work = |offset: Option<usize>, index: usize| RowKind::Work { offset, index };

        let mut rows = Vec::new();
        match self.kind {
            Kind::Rej(shape) => {
                for block in 0..shape.blocks {
                    rows.push((block, RowKind::In, false));
                    rows.push((block, RowKind::Out, false));

                    for w in 0..RATE_128 / WINDOW {
                        let index = block * (RATE_128 / WINDOW) + w;

                        rows.push((block, work(Some(WINDOW * w), index), false));
                    }
                }
            }
            Kind::Cbd(2) => {
                rows.push((0, RowKind::In, false));
                rows.push((0, RowKind::Out, false));

                for lane in 0..self.squeeze_bytes() / 8 {
                    rows.push((0, work(Some(lane), lane), false));
                }
            }
            Kind::Cbd(3) => {
                // 192 bytes over two blocks:
                // windows at lanes 0..15 of block 0, the straddle window of tail
                // lanes 15 and 16 with lane 0 of block 1, then lanes 1..7 of block 1.
                rows.push((0, RowKind::In, false));
                rows.push((0, RowKind::Out, true));

                for w in 0..RATE_256 / WINDOW {
                    rows.push((0, work(Some(WINDOW * w), w), false));
                }

                let straddled = RATE_256 / WINDOW;

                rows.push((1, RowKind::In, false));
                rows.push((1, RowKind::Out, false));
                rows.push((1, work(None, straddled), false));

                for w in 0..2 {
                    rows.push((1, work(Some(1 + WINDOW * w), straddled + 1 + w), false));
                }
            }
            Kind::Cbd(_) => {}
            Kind::Ball(shape) => {
                // IN, OUT with the sign lane captured, one row per candidate
                // byte from byte 8 with a new block at each rate boundary,
                // then τ step rows, 256 readout rows and 3τ + 256 sorted rows.
                let rate = RATE_256 * 8;

                rows.push((0, RowKind::In, false));
                rows.push((0, RowKind::Out, true));

                for c in 0..shape.budget {
                    let (block, at) = ((SIGN_BYTES + c) / rate, (SIGN_BYTES + c) % rate);

                    if at == 0 {
                        rows.push((block, RowKind::In, false));
                        rows.push((block, RowKind::Out, false));
                    }

                    rows.push((block, work(Some(at / 8), c), false));
                }

                rows.extend((0..shape.tau).map(|t| (0, RowKind::Step(t), false)));
                rows.extend((0..N).map(|p| (0, RowKind::Readout(p), false)));
                rows.extend((0..shape.accesses()).map(|s| (0, RowKind::Sorted(s), false)));
            }
        }

        rows
    }

    /// The absorbed message: the seed's bytes, then the suffix.
    fn message(&self, seed: &[u64]) -> Zeroizing<Vec<u8>> {
        let seed_bytes = 8 * self.seed_lanes();

        let mut message = Zeroizing::new(Vec::with_capacity(seed_bytes + self.suffix.len()));
        message.extend(
            seed.iter()
                .flat_map(|lane| lane.to_le_bytes())
                .take(seed_bytes),
        );
        message.extend_from_slice(&self.suffix);

        message
    }

    /// The first padded block with the seed bytes zero,
    /// as lanes; absorb rows add the seed lanes to it.
    fn template(&self) -> [u64; LANES] {
        let rate = self.rate() * 8;
        let start = 8 * self.seed_lanes();
        let end = start + self.suffix.len();

        let mut block = [0u8; RATE_128 * 8];

        block[start..end].copy_from_slice(&self.suffix);
        block[end] ^= SHAKE_PAD;
        block[rate - 1] ^= PAD_LAST;

        let mut lanes = [0u64; LANES];
        for (lane, bytes) in lanes.iter_mut().zip(block[..rate].chunks_exact(8)) {
            let mut le = [0u8; 8];
            le.copy_from_slice(bytes);

            *lane = u64::from_le_bytes(le);
        }

        lanes
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Kinds {
    pub(crate) rej: Option<RejShape>,
    pub(crate) ball: Option<BallShape>,
    pub(crate) eta2: bool,
    pub(crate) eta3: bool,
    pub(crate) seed_lanes: usize,
}

impl Kinds {
    /// The kinds a schedule mixes. Fails on a second rejection
    /// or SampleInBall shape and on η outside 2 and 3.
    fn of(steps: &[SamplerStep]) -> errors::Result<Self> {
        let mut kinds = Self::default();
        for step in steps {
            kinds.seed_lanes = kinds.seed_lanes.max(step.seed_lanes());

            match step.kind {
                Kind::Rej(shape) => match kinds.rej.replace(shape) {
                    Some(prev) if prev != shape => {
                        return Err(Error::Protocol {
                            protocol: "sampler_chiplet",
                            message: "one sampler table holds one rejection shape",
                        });
                    }
                    _ => {}
                },
                Kind::Ball(shape) => match kinds.ball.replace(shape) {
                    Some(prev) if prev != shape => {
                        return Err(Error::Protocol {
                            protocol: "sampler_chiplet",
                            message: "one sampler table holds one SampleInBall shape",
                        });
                    }
                    _ => {}
                },
                Kind::Cbd(2) => kinds.eta2 = true,
                Kind::Cbd(3) => kinds.eta3 = true,
                Kind::Cbd(_) => {
                    return Err(Error::Protocol {
                        protocol: "sampler_chiplet",
                        message: "CBD takes eta 2 or 3",
                    });
                }
            }
        }

        Ok(kinds)
    }

    /// The CBD η values present, ascending;
    /// the CBD blocks follow this order.
    pub(crate) fn etas(&self) -> impl Iterator<Item = usize> + '_ {
        [(2, self.eta2), (3, self.eta3)]
            .into_iter()
            .filter(|&(_, on)| on)
            .map(|(eta, _)| eta)
    }
}

struct Rows<'s> {
    steps: &'s [SamplerStep],
    plan: Vec<Option<RowInfo>>,
    templates: Vec<[u64; LANES]>,
}

impl Rows<'_> {
    fn is(&self, r: usize, pred: impl Fn(&RowInfo) -> bool) -> bool {
        self.plan[r].as_ref().is_some_and(pred)
    }

    fn of(&self, r: usize, pred: impl Fn(Kind, &RowInfo) -> bool) -> bool {
        self.plan[r]
            .as_ref()
            .is_some_and(|i| pred(self.steps[i.step].kind, i))
    }

    fn label(&self, r: usize, value: impl Fn(&RowInfo) -> u64) -> u64 {
        self.plan[r].as_ref().map_or(0, value)
    }

    fn next(&self, r: usize) -> usize {
        (r + 1) % self.plan.len()
    }

    fn absorb(&self, r: usize) -> bool {
        self.is(r, |i| i.kind == RowKind::In && i.block == 0)
    }

    fn first(&self, r: usize) -> bool {
        self.is(r, |i| i.first)
    }

    fn out(&self, r: usize) -> bool {
        self.is(r, |i| i.kind == RowKind::Out)
    }

    fn step(&self, r: usize) -> bool {
        self.is(r, |i| matches!(i.kind, RowKind::Step(_)))
    }

    fn work(&self, r: usize, want: impl Fn(Kind) -> bool) -> bool {
        self.of(r, |k, i| matches!(i.kind, RowKind::Work { .. }) && want(k))
    }

    fn captures(&self, r: usize, want: impl Fn(Kind) -> bool) -> bool {
        self.of(r, |k, i| i.capture && want(k))
    }

    fn seed_lanes(&self, r: usize) -> usize {
        self.plan[r].map_or(0, |i| self.steps[i.step].seed_lanes())
    }

    fn core(&self, ly: &SamplerLayout, column: &mut Column<'_>) -> errors::Result<()> {
        let next = |r: usize| self.next(r);

        column(ly.kec, &|r| {
            self.is(r, |i| matches!(i.kind, RowKind::In | RowKind::Out)) as u64
        })?;
        column(ly.absorb, &|r| self.absorb(r) as u64)?;
        column(ly.keep, &|r| {
            !(self.absorb(next(r)) || self.out(next(r))) as u64
        })?;
        column(ly.seed_keep, &|r| !self.first(next(r)) as u64)?;
        column(ly.carry, &|r| !self.absorb(next(r)) as u64)?;
        column(ly.poly, &|r| {
            self.label(r, |i| match i.kind {
                RowKind::Work { .. }
                | RowKind::Step(_)
                | RowKind::Readout(_)
                | RowKind::Sorted(_) => self.steps[i.step].out.id() as u64,
                _ => 0,
            })
        })?;
        column(ly.seed_stream, &|r| {
            self.label(r, |i| match i.first {
                true => self.steps[i.step].seed.id() as u64,
                false => 0,
            })
        })?;

        for &(o, col) in &ly.offsets {
            column(col, &|r| {
                self.is(
                    r,
                    |i| matches!(i.kind, RowKind::Work { offset: Some(x), .. } if x == o),
                ) as u64
            })?;
        }

        Ok(())
    }

    fn rejection(&self, ly: &SamplerLayout, column: &mut Column<'_>) -> errors::Result<()> {
        match &ly.rej_cols {
            Some(rc) => column(rc.work, &|r| {
                self.work(r, |k| matches!(k, Kind::Rej(_))) as u64
            }),
            None => Ok(()),
        }
    }

    fn cbd(&self, ly: &SamplerLayout, column: &mut Column<'_>) -> errors::Result<()> {
        for (block, cc) in ly.cbd.iter().zip(&ly.cbd_cols) {
            let eta = block.eta;
            let slots = cbd_slots(eta);

            column(cc.work, &|r| self.work(r, |k| k == Kind::Cbd(eta)) as u64)?;

            for m in 0..slots {
                column(cc.pos.at(m), &|r| {
                    self.label(r, |i| match (i.kind, self.steps[i.step].kind) {
                        (RowKind::Work { index, .. }, Kind::Cbd(e)) if e == eta => {
                            (index * slots + m) as u64
                        }
                        _ => 0,
                    })
                })?;
            }
        }

        Ok(())
    }

    fn tail(&self, ly: &SamplerLayout, column: &mut Column<'_>) -> errors::Result<()> {
        let Some(tc) = &ly.tail else {
            return Ok(());
        };

        let capture = |r: usize| self.captures(r, |k| matches!(k, Kind::Cbd(_)));
        let next = |r: usize| self.next(r);

        column(tc.capture, &|r| capture(r) as u64)?;
        column(tc.keep, &|r| {
            !(self.absorb(next(r)) || capture(next(r))) as u64
        })?;

        column(tc.straddle, &|r| {
            self.is(r, |i| matches!(i.kind, RowKind::Work { offset: None, .. })) as u64
        })
    }

    fn ball(&self, ly: &SamplerLayout, column: &mut Column<'_>) -> errors::Result<()> {
        let (Some(bits), Some(bc)) = (&ly.ball, &ly.ball_cols) else {
            return Ok(());
        };

        let shape = bits.shape;

        let cell = |t: usize| (shape.first_cell() + t) as u64;
        let ball = |r: usize| self.work(r, |k| matches!(k, Kind::Ball(_)));
        let capture = |r: usize| self.captures(r, |k| matches!(k, Kind::Ball(_)));
        let next = |r: usize| self.next(r);

        column(bc.work, &|r| ball(r) as u64)?;

        for b in 0..BYTE {
            column(bc.bytes.at(b), &|r| {
                (ball(r)
                    && self.is(
                        r,
                        |i| matches!(i.kind, RowKind::Work { index, .. } if index % BYTE == b),
                    )) as u64
            })?;
        }

        column(bc.sign.capture, &|r| capture(r) as u64)?;
        column(bc.sign.keep, &|r| {
            !(self.absorb(next(r)) || capture(next(r)) || self.step(r)) as u64
        })?;
        column(bc.sign.shift, &|r| self.step(r) as u64)?;

        column(bc.mem.step, &|r| self.step(r) as u64)?;
        column(bc.mem.read, &|r| {
            self.is(r, |i| {
                matches!(i.kind, RowKind::Step(_) | RowKind::Readout(_))
            }) as u64
        })?;
        column(bc.mem.readout, &|r| {
            self.is(r, |i| matches!(i.kind, RowKind::Readout(_))) as u64
        })?;
        column(bc.mem.pos, &|r| {
            self.label(r, |i| match i.kind {
                RowKind::Step(t) => N as u64 + cell(t),
                RowKind::Readout(p) => p as u64,
                _ => 0,
            })
        })?;
        column(bc.mem.cell, &|r| {
            self.label(r, |i| match i.kind {
                RowKind::Step(t) => cell(t),
                _ => 0,
            })
        })?;

        for k in 0..STEP_ACCESSES {
            column(bc.mem.times.at(k), &|r| {
                self.label(r, |i| match i.kind {
                    RowKind::Step(t) => (STEP_ACCESSES * t + k) as u64,
                    RowKind::Readout(p) if k == 0 => (STEP_ACCESSES * shape.tau + p) as u64,
                    _ => 0,
                })
            })?;
        }

        column(bc.sort.sorted, &|r| {
            self.is(r, |i| matches!(i.kind, RowKind::Sorted(_))) as u64
        })?;
        column(bc.sort.link, &|r| {
            self.is(
                r,
                |i| matches!(i.kind, RowKind::Sorted(s) if s + 1 < shape.accesses()),
            ) as u64
        })?;

        column(bc.sort.first, &|r| {
            self.is(r, |i| i.kind == RowKind::Sorted(0)) as u64
        })
    }

    fn seeds(&self, ly: &SamplerLayout, column: &mut Column<'_>) -> errors::Result<()> {
        for l in 0..ly.seed.len() {
            column(ly.seed_use.at(l), &|r| {
                (self.absorb(r) && l < self.seed_lanes(r)) as u64
            })?;
            column(ly.seed_load.at(l), &|r| {
                (self.first(r) && l < self.seed_lanes(r)) as u64
            })?;
            column(ly.seed_zero.at(l), &|r| {
                (self.first(r) && l >= self.seed_lanes(r)) as u64
            })?;
            column(ly.seed_index.at(l), &|r| {
                self.label(r, |i| match i.first {
                    true => l as u64,
                    false => 0,
                })
            })?;
        }

        Ok(())
    }

    fn pads(&self, ly: &SamplerLayout, column: &mut Column<'_>) -> errors::Result<()> {
        for &(lane, col) in &ly.pads {
            column(col, &|r| match self.plan[r] {
                Some(i) if self.absorb(r) => self.templates[i.step][lane],
                _ => 0,
            })?;
        }

        Ok(())
    }
}

/// A witness that leaves the algorithm at one point of step `step`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplerForgery {
    /// Candidate `candidate` takes the other decision.
    Flip { step: usize, candidate: usize },

    /// Taken candidates `candidate` and `candidate + 1` trade
    /// coefficient positions; both sit in one row, past its first slot.
    Swap { step: usize, candidate: usize },

    /// The η = 3 tail holds lanes 15 and 16 of the second OUT.
    Tail { step: usize },

    /// The step absorbs its suffix bytes in reverse order.
    Suffix { step: usize },

    /// Rejection window `window` reads the three lanes one lane on.
    Window { step: usize, window: usize },

    /// An address's first two sorted accesses,
    /// a write and a read of zero, trade places.
    Unsorted { step: usize },
}

/// Table sampling ML-DSA and ML-KEM polynomials from
/// seed streams through the Keccak table's permutation.
#[derive(Clone)]
pub struct SamplerChiplet<F: TowerField> {
    program: CircuitProgram<F>,
    steps: Vec<SamplerStep>,
    layout: SamplerLayout,
    num_rows: usize,
}

impl<F> SamplerChiplet<F>
where
    F: TowerField + TraceCompatibleField + PackableField + HardwareField + Send + 'static,
    <F as PackableField>::Packed: Copy + Send + Sync,
    Flat<F>: Send + Sync,
{
    /// One rejection and one SampleInBall shape, CBD with η of 2 or 3,
    /// distinct outputs and one run per seed stream; else it fails.
    pub fn new(steps: Vec<SamplerStep>, num_rows: usize) -> errors::Result<Self> {
        let kinds = validate(&steps, num_rows)?;

        let (pad_lanes, offsets) = lanes(&steps);

        let mut cx = Circuit::<F>::new("SamplerChiplet", num_rows)?;

        let layout = SamplerLayout::declare(&mut cx, &kinds, &pad_lanes, &offsets);

        pinned(&steps, &layout, num_rows, |col, value| {
            cx.fix(col, pinned_shape((0..num_rows).map(value)));

            Ok(())
        })?;

        buses(&mut cx, &layout)?;

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
    pub fn layout(&self) -> &SamplerLayout {
        &self.layout
    }

    /// Traces the table, records each sampled polynomial in
    /// `values` and returns the Keccak inputs in request order.
    pub fn trace(
        &self,
        seeds: &LaneValues,
        values: &mut PolyValues,
    ) -> errors::Result<(ColumnTrace, Zeroizing<Vec<[u64; LANES]>>)> {
        trace::generate(self, seeds, values, &[])
    }

    /// `trace` with `forgeries` applied and
    /// without the squeeze budget check.
    #[cfg(feature = "forgery")]
    pub fn trace_forged(
        &self,
        seeds: &LaneValues,
        values: &mut PolyValues,
        forgeries: &[SamplerForgery],
    ) -> errors::Result<(ColumnTrace, Zeroizing<Vec<[u64; LANES]>>)> {
        trace::generate(self, seeds, values, forgeries)
    }

    #[cfg(test)]
    pub(crate) fn produced(&self) -> Vec<(&'static str, u16)> {
        self.steps
            .iter()
            .map(|step| (COEF_BUS_ID, step.out.id()))
            .collect()
    }
}

/// Coefficients per CBD work row:
/// 16 for η = 2, 32 for η = 3.
pub(crate) fn cbd_slots(eta: usize) -> usize {
    cbd_lanes(eta) * 64 / (2 * eta)
}

/// Window lanes a CBD work row reads:
/// one for η = 2, three for η = 3.
pub(crate) fn cbd_lanes(eta: usize) -> usize {
    match eta {
        2 => 1,
        _ => WINDOW,
    }
}

/// The role of every table row, steps in order;
/// padding rows are `None`.
pub(crate) fn plan(steps: &[SamplerStep], num_rows: usize) -> Vec<Option<RowInfo>> {
    let mut rows = vec![None; num_rows];
    let mut origin = 0;

    for (index, step) in steps.iter().enumerate() {
        let first = index == 0 || steps[index - 1].seed != step.seed;
        let step_plan = step.plan();

        for (offset, &(block, kind, capture)) in step_plan.iter().enumerate() {
            rows[origin + offset] = Some(RowInfo {
                step: index,
                block,
                kind,
                first: first && offset == 0,
                capture,
            });
        }

        origin += step_plan.len();
    }

    rows
}

/// Every fixed column with its value on each row, handed to
/// `column`: `new` fixes the columns and the trace writes them.
pub(crate) fn pinned<C>(
    steps: &[SamplerStep],
    ly: &SamplerLayout,
    num_rows: usize,
    mut column: C,
) -> errors::Result<()>
where
    C: FnMut(Col, &dyn Fn(usize) -> u64) -> errors::Result<()>,
{
    let rows = Rows {
        steps,
        plan: plan(steps, num_rows),
        templates: steps.iter().map(SamplerStep::template).collect(),
    };

    rows.core(ly, &mut column)?;
    rows.rejection(ly, &mut column)?;
    rows.cbd(ly, &mut column)?;
    rows.tail(ly, &mut column)?;
    rows.ball(ly, &mut column)?;
    rows.seeds(ly, &mut column)?;
    rows.pads(ly, &mut column)
}

fn validate(steps: &[SamplerStep], num_rows: usize) -> errors::Result<Kinds> {
    if steps.is_empty() {
        return Err(Error::Protocol {
            protocol: "sampler_chiplet",
            message: "sampler table needs at least one step",
        });
    }

    let kinds = Kinds::of(steps)?;

    if steps.iter().map(SamplerStep::rows).sum::<usize>() > num_rows {
        return Err(Error::Protocol {
            protocol: "sampler_chiplet",
            message: "steps need more rows than the table holds",
        });
    }

    if steps
        .windows(2)
        .any(|w| w[0].seed == w[1].seed && w[0].seed_lanes() != w[1].seed_lanes())
    {
        return Err(Error::Protocol {
            protocol: "sampler_chiplet",
            message: "adjacent steps on one seed stream disagree on the seed length",
        });
    }

    distinct(
        steps.iter().map(|s| s.out),
        "sampler_chiplet",
        "two steps sample into one polynomial label",
    )?;

    distinct(
        steps
            .iter()
            .enumerate()
            .filter(|&(i, s)| i == 0 || steps[i - 1].seed != s.seed)
            .map(|(_, s)| s.seed),
        "sampler_chiplet",
        "seed stream loads in two separate runs of steps",
    )?;

    Ok(kinds)
}

fn lanes(steps: &[SamplerStep]) -> (Vec<usize>, Vec<usize>) {
    let (mut padded, mut read) = ([false; LANES], [false; LANES]);

    for step in steps {
        for (l, &lane) in step.template().iter().enumerate() {
            padded[l] |= lane != 0;
        }

        for (_, kind, _) in step.plan() {
            if let RowKind::Work {
                offset: Some(x), ..
            } = kind
            {
                read[x] = true;
            }
        }
    }

    let pad_lanes = (0..LANES).filter(|&l| padded[l]).collect();
    let offsets = (0..LANES).filter(|&o| read[o]).collect();

    (pad_lanes, offsets)
}

fn buses<F: TowerField + HardwareField>(
    cx: &mut Circuit<F>,
    ly: &SamplerLayout,
) -> errors::Result<()> {
    let state: Vec<Col> = ly.state.iter().collect();

    cx.call(&KeccakChiplet::service(), &state, ly.kec)?;

    for l in 0..ly.seed.len() {
        cx.bus(
            LANE_BUS_ID,
            lane_spec(
                ly.seed_stream.index(),
                ly.seed_index.at(l).index(),
                ly.seed.at(l).index(),
                ly.seed_load.at(l).index(),
            ),
        );
    }

    if let Some(rc) = &ly.rej_cols {
        for m in 0..rc.key_value.len() {
            cx.bus(
                COEF_BUS_ID,
                coef_spec(
                    rc.key_poly.at(m).index(),
                    rc.key_pos.at(m).index(),
                    rc.key_value.at(m).index(),
                    rc.work.index(),
                ),
            );
        }
    }

    for cc in &ly.cbd_cols {
        for m in 0..cc.value.len() {
            cx.bus(
                COEF_BUS_ID,
                coef_spec(
                    ly.poly.index(),
                    cc.pos.at(m).index(),
                    cc.value.at(m).index(),
                    cc.work.index(),
                ),
            );
        }
    }

    let (Some(bits), Some(bc)) = (&ly.ball, &ly.ball_cols) else {
        return Ok(());
    };

    let (mem, sort) = (&bc.mem, &bc.sort);

    cx.bus(
        COEF_BUS_ID,
        coef_spec(
            bc.key_poly.index(),
            bc.key_pos.index(),
            bc.key_value.index(),
            bc.work.index(),
        ),
    );
    cx.bus(
        COEF_BUS_ID,
        coef_spec(
            ly.poly.index(),
            mem.pos.index(),
            mem.addr.index(),
            mem.step.index(),
        ),
    );
    cx.bus(
        COEF_BUS_ID,
        coef_spec(
            ly.poly.index(),
            mem.pos.index(),
            mem.value.index(),
            mem.readout.index(),
        ),
    );

    cx.bus(
        SIB_BUS_ID,
        sib_spec(
            [ly.poly, mem.times.at(0), mem.addr, mem.value],
            Source::Const(0),
            mem.read,
        ),
    );
    cx.bus(
        SIB_BUS_ID,
        sib_spec(
            [ly.poly, mem.times.at(1), mem.cell, mem.value],
            Source::Const(1),
            mem.step,
        ),
    );
    cx.bus(
        SIB_BUS_ID,
        sib_spec(
            [ly.poly, mem.times.at(2), mem.addr, bc.sign.value],
            Source::Const(1),
            mem.step,
        ),
    );
    cx.bus(
        SIB_BUS_ID,
        sib_spec(
            [ly.poly, sort.time, sort.addr, sort.value],
            Source::Column(bits.sort.write),
            sort.sorted,
        ),
    );

    Ok(())
}

/// An endpoint of the sib bus, keyed
/// (step label, time, address, value, write).
fn sib_spec(
    [poly, time, addr, value]: [Col; 4],
    write: Source,
    selector: Col,
) -> PermutationCheckSpec {
    PermutationCheckSpec::new(
        vec![
            (Source::Column(poly.index()), b"kappa_sib_poly" as &[u8]),
            (Source::Column(time.index()), b"kappa_sib_time"),
            (Source::Column(addr.index()), b"kappa_sib_addr"),
            (Source::Column(value.index()), b"kappa_sib_value"),
            (write, b"kappa_sib_write"),
        ],
        Some(selector.index()),
    )
    .with_clock_waiver(SIB_WAIVER)
}
