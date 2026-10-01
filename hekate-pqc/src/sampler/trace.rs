// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Witness of the Sampler table: each step's stream and decisions,
//! then the rows, carrying the registers the AIR carries.

use alloc::vec;
use alloc::vec::Vec;
use hekate_core::errors::{self, Error};
use hekate_core::trace::{ColumnTrace, TraceBuilder};
use hekate_keccak::{KeccakCall, shake128, shake256};
use hekate_math::TowerField;
use hekate_program::Air;
use hekate_program::circuit::Col;
use subtle::{
    Choice, ConditionallySelectable, ConstantTimeEq, ConstantTimeGreater, ConstantTimeLess,
};
use zeroize::{Zeroize, ZeroizeOnDrop, Zeroizing};

use super::layout::{
    BYTE, BallBits, BallCols, COUNT_BITS, CbdBits, CbdCols, LANES, SEED_LANES, SamplerLayout,
    SortBits, SortCols, TIME_BITS, WINDOW,
};
use super::{
    BallShape, Kind, RATE_128, RATE_256, RejShape, RowInfo, RowKind, SIGN_BYTES, STEP_ACCESSES,
    SamplerChiplet, SamplerForgery, SamplerStep, TAIL_LANES, cbd_slots, pinned, plan,
};
use crate::utils::{
    Writer, fill_sub_borrow_packed, flush_bit_buffer, pack_bits, pack_one, write_column,
};
use crate::wiring::{LaneValues, N, PolyValues};
use crate::{mldsa, mlkem};

/// One SampleInBall memory access, in program order.
#[derive(Clone, Copy, Default, Zeroize)]
struct Access {
    time: u32,
    addr: u32,
    value: u32,
    write: bool,
}

impl Access {
    fn key(&self) -> u64 {
        (self.addr as u64) << 32 | self.time as u64
    }
}

impl ConditionallySelectable for Access {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        let write = Choice::conditional_select(
            &Choice::from(a.write as u8),
            &Choice::from(b.write as u8),
            choice,
        );

        Self {
            time: u32::conditional_select(&a.time, &b.time, choice),
            addr: u32::conditional_select(&a.addr, &b.addr, choice),
            value: u32::conditional_select(&a.value, &b.value, choice),
            write: bool::from(write),
        }
    }
}

#[derive(Zeroize, ZeroizeOnDrop)]
struct BallWitness {
    /// j_t of each step t.
    picks: Vec<u32>,

    /// `c[j_t]` as step t reads it.
    reads: Vec<u32>,

    /// The challenge c after the last step.
    cells: [u32; N],
    sorted: Vec<Access>,
}

#[derive(Zeroize, ZeroizeOnDrop)]
struct StepWitness {
    seed: [u64; SEED_LANES],
    calls: Vec<KeccakCall>,
    uses: Vec<bool>,
    coeffs: [u32; N],
    taken: usize,
    tail: [u64; TAIL_LANES],
    swaps: Vec<usize>,
    skews: Vec<usize>,
    ball: Option<BallWitness>,
}

#[derive(Default)]
struct RowInputs<'w> {
    window: [u64; WINDOW],
    uses: Option<&'w [bool]>,
    swap: Option<usize>,
    candidate: (u32, bool),
    memory: (u32, u32),
}

/// Registers carried row to row as the AIR carries them:
/// the use counters, the Keccak state, the seed,
/// the η = 3 tail and the sign lane.
#[derive(Default, Zeroize, ZeroizeOnDrop)]
struct Regs {
    count: u32,
    ball_count: u32,
    state: [u64; LANES],
    seed: [u64; SEED_LANES],
    tail: [u64; TAIL_LANES],
    sign: u64,
}

impl Regs {
    fn advance<'w>(
        &mut self,
        info: &RowInfo,
        witness: &'w StepWitness,
        kind: Kind,
        first_cell: u32,
    ) -> RowInputs<'w> {
        let mut inputs = RowInputs::default();

        if info.first {
            self.seed = witness.seed;
        }

        match info.kind {
            RowKind::In => {
                self.state = witness.calls[info.block].0;

                // An absorb row restarts both counters and zeroes the tail
                // and the sign lane, as the AIR's absorb constraints pin them.
                if info.block == 0 {
                    self.count = 0;
                    self.ball_count = first_cell;
                    self.tail = [0; TAIL_LANES];
                    self.sign = 0;
                }
            }
            RowKind::Out => {
                self.state = witness.calls[info.block].1;

                // A capture row loads the step's tail for η = 3,
                // or the sign lane for SampleInBall from this OUT.
                match (info.capture, kind) {
                    (true, Kind::Cbd(_)) => self.tail = witness.tail,
                    (true, Kind::Ball(_)) => self.sign = self.state[0],
                    _ => {}
                }
            }
            RowKind::Work { offset, index } => {
                inputs.window = match offset {
                    Some(o) => {
                        let o = o + witness.skews.contains(&index) as usize;

                        [self.state[o], self.state[o + 1], self.state[o + 2]]
                    }
                    None => [self.tail[0], self.tail[1], self.state[0]],
                };

                match kind {
                    Kind::Rej(shape) => {
                        inputs.uses = Some(&witness.uses[index * shape.slots..][..shape.slots]);
                        inputs.swap = witness
                            .swaps
                            .iter()
                            .find(|&&c| c / shape.slots == index)
                            .map(|&c| c % shape.slots);
                    }
                    Kind::Ball(_) => {
                        // Stream byte 8 + c is byte c mod 8 of its lane
                        let byte = (inputs.window[0] >> (BYTE * (index % BYTE))) as u8;
                        inputs.candidate = (byte as u32, witness.uses[index]);
                    }
                    Kind::Cbd(_) => {}
                }
            }
            RowKind::Step(_) | RowKind::Readout(_) | RowKind::Sorted(_) => {}
        }

        inputs.memory = match (info.kind, &witness.ball) {
            (RowKind::Step(t), Some(b)) => (b.picks[t], b.reads[t]),
            (RowKind::Readout(p), Some(b)) => (p as u32, b.cells[p]),
            _ => (0, 0),
        };

        inputs
    }
}

/// Traces the table, records each sampled polynomial
/// in `values` and returns the Keccak inputs in request
/// order, with `forgeries` applied for adaptive forgeries.
pub(super) fn generate<F: TowerField>(
    chiplet: &SamplerChiplet<F>,
    seeds: &LaneValues,
    values: &mut PolyValues,
    forgeries: &[SamplerForgery],
) -> errors::Result<(ColumnTrace, Zeroizing<Vec<[u64; LANES]>>)> {
    let ly = &chiplet.layout;
    let steps = &chiplet.steps;
    let num_rows = chiplet.num_rows;

    let mut witnesses = Vec::with_capacity(steps.len());
    for (index, step) in steps.iter().enumerate() {
        let mut witness = sample(step, seeds, (index, forgeries))?;

        if forgeries.is_empty() && witness.taken < N {
            return Err(Error::Protocol {
                protocol: "sampler_chiplet",
                message: "the squeeze budget ran out before 256 coefficients",
            });
        }

        forge(&mut witness, step, index, forgeries);

        values.insert(step.out, witness.coeffs)?;
        witnesses.push(witness);
    }

    let mut tb = TraceBuilder::new_secret(
        chiplet.program.column_layout(),
        num_rows.trailing_zeros() as usize,
    )?;

    let rows = plan(steps, num_rows);
    let first_cell = ly
        .ball
        .as_ref()
        .map_or(0, |bb| bb.shape.first_cell() as u32);

    let sorted_at = |r: usize| match rows[r % num_rows] {
        Some(RowInfo {
            kind: RowKind::Sorted(s),
            step,
            ..
        }) => witnesses[step].ball.as_ref().map(|b| b.sorted[s]),
        _ => None,
    };

    let mut regs = Regs::default();
    let mut bits = Zeroizing::new(vec![0u32; ly.num_packed]);

    for (r, &info) in rows.iter().enumerate() {
        bits.fill(0);

        let inputs = match info {
            Some(i) => regs.advance(&i, &witnesses[i.step], steps[i.step].kind, first_cell),
            None => RowInputs::default(),
        };

        let poly = match info {
            Some(i) => match i.kind {
                RowKind::Work { .. } | RowKind::Step(_) | RowKind::Readout(_) => {
                    steps[i.step].out.id()
                }
                _ => 0,
            },
            None => 0,
        };

        let mut w = Writer {
            tb: &mut tb,
            physical: |col| ly.physical(col),
            row: r,
        };

        // Every block fills on every row, work or not:
        // the AIR's chains and the ±1 value run ungated.
        fill_rejection(
            &mut w,
            &mut bits,
            ly,
            &inputs.window,
            (inputs.uses, inputs.swap),
            poly,
            &mut regs.count,
        )?;

        for (block, cc) in ly.cbd.iter().zip(&ly.cbd_cols) {
            fill_cbd(&mut w, &mut bits, block, cc, &inputs.window)?;
        }

        if let (Some(bb), Some(bc)) = (&ly.ball, &ly.ball_cols) {
            fill_ball(
                &mut w,
                &mut bits,
                bb,
                bc,
                inputs.candidate,
                poly,
                &mut regs.ball_count,
            )?;

            w.lane(bc.sign.lane, regs.sign)?;
            w.word(bc.sign.value, signed_one((regs.sign & 1) as u8))?;

            w.label(bc.mem.addr, inputs.memory.0 as u16)?;
            w.word(bc.mem.value, inputs.memory.1)?;

            fill_sorted(
                &mut w,
                &mut bits,
                &bb.sort,
                &bc.sort,
                sorted_at(r),
                sorted_at(r + 1),
            )?;
        }

        flush_bit_buffer(&bits, w.tb, r)?;

        for (k, &lane) in inputs.window.iter().enumerate() {
            w.lane(ly.window.at(k), lane)?;
        }

        for (l, &lane) in regs.state.iter().enumerate() {
            w.lane(ly.state.at(l), lane)?;
        }

        for (l, &lane) in regs.seed.iter().take(ly.seed.len()).enumerate() {
            w.lane(ly.seed.at(l), lane)?;
        }

        if let Some(tc) = &ly.tail {
            for (k, &lane) in regs.tail.iter().enumerate() {
                w.lane(tc.lanes.at(k), lane)?;
            }
        }

        // The shift constraint of a step row fixes S on the next row
        if info.is_some_and(|i| matches!(i.kind, RowKind::Step(_))) {
            regs.sign >>= 1;
        }
    }

    drop(regs);

    let types = chiplet.program.column_layout();

    pinned(steps, ly, num_rows, |col, value| {
        let at = ly.physical(col)?;

        write_column(&mut tb, types[at], at, (0..num_rows).map(value))
    })?;

    let blocks = witnesses.iter().map(|w| w.calls.len()).sum();

    let mut calls = Zeroizing::new(Vec::with_capacity(blocks));
    calls.extend(
        witnesses
            .iter()
            .flat_map(|w| w.calls.iter().map(|&(input, _)| input)),
    );

    Ok((tb.build(), calls))
}

pub(super) fn coefficients(step: &SamplerStep, seeds: &LaneValues) -> errors::Result<[u32; N]> {
    sample(step, seeds, (0, &[])).map(|witness| witness.coeffs)
}

/// Squeezes one step's stream and derives its decisions,
/// coefficients and, for SampleInBall, its memory log.
fn sample(
    step: &SamplerStep,
    seeds: &LaneValues,
    (index, forgeries): (usize, &[SamplerForgery]),
) -> errors::Result<StepWitness> {
    let lanes = seeds.get(step.seed)?;

    if lanes.len() < step.seed_lanes() {
        return Err(Error::Protocol {
            protocol: "sampler_chiplet",
            message: "seed stream is shorter than the step's seed",
        });
    }

    let mut seed = Zeroizing::new([0u64; SEED_LANES]);
    for (dst, &lane) in seed.iter_mut().zip(lanes).take(step.seed_lanes()) {
        *dst = lane;
    }

    let mut message = step.message(&seed[..]);
    if forgeries.contains(&SamplerForgery::Suffix { step: index }) {
        message[8 * step.seed_lanes()..].reverse();
    }

    let (bytes, calls) = match step.kind {
        Kind::Rej(_) => shake128(&message, step.squeeze_bytes()),
        Kind::Cbd(_) | Kind::Ball(_) => shake256(&message, step.squeeze_bytes()),
    };

    let mut bytes = Zeroizing::new(bytes);
    let mut skews = Vec::new();

    let flipped = |c: usize| {
        forgeries.contains(&SamplerForgery::Flip {
            step: index,
            candidate: c,
        })
    };

    let (uses, coeffs, taken, ball) = match step.kind {
        Kind::Rej(shape) => {
            for f in forgeries {
                if let SamplerForgery::Window { step: at, window } = *f
                    && at == index
                {
                    skew(&mut bytes, window)?;
                    skews.push(window);
                }
            }

            let (uses, coeffs, taken) = rejection_uses(shape, &bytes, flipped);

            (uses, coeffs, taken, None)
        }
        Kind::Cbd(eta) => (Vec::new(), cbd_coefficients(eta, &bytes), N, None),
        Kind::Ball(shape) => {
            let uses = ball_uses(shape, &bytes, flipped);
            let witness = ball_witness(shape, &bytes, &uses)?;

            (uses, witness.cells, N, Some(witness))
        }
    };

    let mut tail = [0u64; TAIL_LANES];
    if step.kind == Kind::Cbd(3) {
        tail.copy_from_slice(&calls[0].1[RATE_256 - TAIL_LANES..RATE_256]);
    }

    Ok(StepWitness {
        seed: *seed,
        calls,
        uses,
        coeffs,
        taken,
        tail,
        swaps: Vec::new(),
        skews,
        ball,
    })
}

fn skew(bytes: &mut [u8], window: usize) -> errors::Result<()> {
    let per_block = RATE_128 / WINDOW;
    let offset = WINDOW * (window % per_block);

    if offset + WINDOW >= RATE_128 {
        return Err(Error::Protocol {
            protocol: "sampler_chiplet",
            message: "skewed window reads past the rate",
        });
    }

    let start = 8 * (window / per_block * RATE_128 + offset);

    bytes.copy_within(start + 8..start + 8 * (WINDOW + 1), start);

    Ok(())
}

/// Use decisions of every candidate and the
/// coefficients they take, in stream order.
fn rejection_uses(
    shape: RejShape,
    bytes: &[u8],
    flipped: impl Fn(usize) -> bool,
) -> (Vec<bool>, [u32; N], usize) {
    let candidates = bytes.len() * 8 / shape.stride;

    let mut uses = Vec::with_capacity(candidates);
    let mut coeffs = [0u32; N];
    let mut taken = 0u32;

    for c in 0..candidates {
        let value = byte_field(bytes, shape.stride * c, shape.width);

        // Used when below q while fewer than 256 are taken;
        // `flipped` inverts chosen decisions for adaptive forgeries.
        let used =
            (value.ct_lt(&shape.q) & taken.ct_lt(&(N as u32))) ^ Choice::from(flipped(c) as u8);

        for (p, slot) in coeffs.iter_mut().enumerate() {
            slot.conditional_assign(&value, used & (p as u32).ct_eq(&taken));
        }

        taken += used.unwrap_u8() as u32;
        uses.push(bool::from(used));
    }

    (uses, coeffs, taken as usize)
}

/// SamplePolyCBD_η over the PRF bytes, FIPS 203 Algorithm 8.
fn cbd_coefficients(eta: usize, bytes: &[u8]) -> [u32; N] {
    core::array::from_fn(|i| {
        let x = byte_field(bytes, 2 * eta * i, eta).count_ones();
        let y = byte_field(bytes, 2 * eta * i + eta, eta).count_ones();

        cbd_value(x, y)
    })
}

/// Take decisions of SampleInBall's candidate
/// bytes, one per byte after the sign lane.
fn ball_uses(shape: BallShape, bytes: &[u8], flipped: impl Fn(usize) -> bool) -> Vec<bool> {
    let (tau, first) = (shape.tau as u32, shape.first_cell() as u32);
    let mut picked = 0u32;

    (0..shape.budget)
        .map(|c| {
            let j = bytes[SIGN_BYTES + c] as u32;

            // Algorithm 29 takes j for i = 256 − τ + picked when j ≤ i;
            // `flipped` inverts chosen decisions for adaptive forgeries.
            let used =
                (picked.ct_lt(&tau) & !j.ct_gt(&(first + picked))) ^ Choice::from(flipped(c) as u8);

            picked += used.unwrap_u8() as u32;

            bool::from(used)
        })
        .collect()
}

/// Runs SampleInBall over the taken bytes and logs every access.
/// Fails when the budget holds fewer than τ taken bytes.
fn ball_witness(shape: BallShape, bytes: &[u8], uses: &[bool]) -> errors::Result<BallWitness> {
    let mut picks = Zeroizing::new(vec![0u32; shape.tau]);
    let mut picked = 0u32;

    for (c, &used) in uses.iter().enumerate() {
        let used = Choice::from(used as u8);
        let j = bytes[SIGN_BYTES + c] as u32;

        for (t, pick) in picks.iter_mut().enumerate() {
            pick.conditional_assign(&j, used & (t as u32).ct_eq(&picked));
        }

        picked += used.unwrap_u8() as u32;
    }

    if (picked as usize) < shape.tau {
        return Err(Error::Protocol {
            protocol: "sampler_chiplet",
            message: "SampleInBall ran out of candidate bytes",
        });
    }

    let mut signs = [0u8; SIGN_BYTES];
    signs.copy_from_slice(&bytes[..SIGN_BYTES]);

    let signs = u64::from_le_bytes(signs);

    // Algorithm 29 with each access logged:
    // step t reads c[j] at 3t,
    // writes c[i] = v at 3t + 1
    // and c[j] = ±1 at 3t + 2.
    let mut cells = [0u32; N];
    let mut reads = Vec::with_capacity(shape.tau);
    let mut accesses = Zeroizing::new(Vec::with_capacity(shape.accesses()));

    for (t, &j) in picks.iter().enumerate() {
        let i = shape.first_cell() + t;
        let v = cells.iter().enumerate().fold(0, |acc, (p, &cell)| {
            u32::conditional_select(&acc, &cell, (p as u32).ct_eq(&j))
        });

        let signed = signed_one(((signs >> t) & 1) as u8);

        cells[i] = v;

        for (p, cell) in cells.iter_mut().enumerate() {
            cell.conditional_assign(&signed, (p as u32).ct_eq(&j));
        }

        reads.push(v);

        let time = (STEP_ACCESSES * t) as u32;

        accesses.push(Access {
            time,
            addr: j,
            value: v,
            write: false,
        });
        accesses.push(Access {
            time: time + 1,
            addr: i as u32,
            value: v,
            write: true,
        });
        accesses.push(Access {
            time: time + 2,
            addr: j,
            value: signed,
            write: true,
        });
    }

    for (p, &value) in cells.iter().enumerate() {
        accesses.push(Access {
            time: (STEP_ACCESSES * shape.tau + p) as u32,
            addr: p as u32,
            value,
            write: false,
        });
    }

    // The sorted rows replay the log in (address, time) order
    let sorted = sorted_by_key(&accesses);

    Ok(BallWitness {
        picks: core::mem::take(&mut *picks),
        reads,
        cells,
        sorted,
    })
}

/// Fills one row's rejection scratch and the keys of
/// its used slots; `count` carries n from row to row.
fn fill_rejection<P: Fn(Col) -> errors::Result<usize>>(
    w: &mut Writer<'_, P>,
    bits: &mut [u32],
    ly: &SamplerLayout,
    window: &[u64; WINDOW],
    (uses, swap): (Option<&[bool]>, Option<usize>),
    poly: u16,
    count: &mut u32,
) -> errors::Result<()> {
    let (Some(rej), Some(rc)) = (&ly.rej, &ly.rej_cols) else {
        return Ok(());
    };

    let shape = rej.shape;

    pack_bits(bits, rej.count(0), *count as u64, COUNT_BITS);

    for (m, slot) in rej.slots.iter().enumerate() {
        let candidate = field(window, shape.stride * m, shape.width);

        fill_sub_borrow_packed(
            bits,
            slot.result,
            slot.borrow,
            shape.width,
            (shape.q - 1) as u64,
            candidate as u64,
        );

        let n = *count;
        let u = uses.is_some_and(|uses| uses[m]);
        let hit = Choice::from(u as u8);

        *count = n + u as u32;

        let (pos, after) = match swap {
            Some(s) if m + 1 == s => (n, *count + 1),
            Some(s) if m == s => (n + 1, count.wrapping_sub(1)),
            Some(s) if m == s + 1 => (n.wrapping_sub(1), *count),
            _ => (n, *count),
        };

        pack_one(bits, slot.u, u);
        pack_bits(bits, rej.count(m + 1), after as u64, COUNT_BITS);

        fill_increment(bits, slot.carry, n, u);

        w.label(rc.key_poly.at(m), u16::conditional_select(&0, &poly, hit))?;
        w.label(
            rc.key_pos.at(m),
            u16::conditional_select(&0, &(pos as u16), hit),
        )?;
        w.word(
            rc.key_value.at(m),
            u32::conditional_select(&0, &candidate, hit),
        )?;
    }

    Ok(())
}

/// Fills one row's CBD scratch and coefficient values from its window.
fn fill_cbd<P: Fn(Col) -> errors::Result<usize>>(
    w: &mut Writer<'_, P>,
    bits: &mut [u32],
    block: &CbdBits,
    cc: &CbdCols,
    window: &[u64; WINDOW],
) -> errors::Result<()> {
    let eta = block.eta;

    for (m, slot) in block.slots.iter().enumerate() {
        let x = field(window, 2 * eta * m, eta).count_ones();
        let y = field(window, 2 * eta * m + eta, eta).count_ones();

        let negative = x.ct_lt(&y);
        let diff = u32::conditional_select(&x.wrapping_sub(y), &y.wrapping_sub(x), negative);
        let low = u32::conditional_select(&y, &x, negative);

        pack_bits(bits, slot.x, x as u64, 2);
        pack_bits(bits, slot.y, y as u64, 2);
        pack_one(bits, slot.sign, bool::from(negative));
        pack_bits(bits, slot.diff, diff as u64, 2);
        pack_one(bits, slot.carry, low & diff & 1 == 1);

        w.word(cc.value.at(m), cbd_value(x, y))?;
    }

    Ok(())
}

/// Fills one row's candidate scratch and, when j is taken,
/// its link key; `count` carries i from row to row.
fn fill_ball<P: Fn(Col) -> errors::Result<usize>>(
    w: &mut Writer<'_, P>,
    bits: &mut [u32],
    bb: &BallBits,
    bc: &BallCols,
    (byte, used): (u32, bool),
    poly: u16,
    count: &mut u32,
) -> errors::Result<()> {
    let n = *count;
    let hit = Choice::from(used as u8);

    *count = n + used as u32;

    pack_bits(bits, bb.byte, byte as u64, BYTE);
    pack_bits(bits, bb.count(0), n as u64, COUNT_BITS);
    pack_bits(bits, bb.count(1), *count as u64, COUNT_BITS);
    pack_one(bits, bb.u, used);

    // The AIR compares the low byte of i with j;
    // at i = 256 the done gate zeroes u.
    fill_sub_borrow_packed(
        bits,
        bb.diff,
        bb.borrow,
        BYTE,
        (n & 0xff) as u64,
        byte as u64,
    );

    fill_increment(bits, bb.carry, n, used);

    w.label(bc.key_poly, u16::conditional_select(&0, &poly, hit))?;
    w.label(
        bc.key_pos,
        u16::conditional_select(&0, &((N as u32 + n) as u16), hit),
    )?;
    w.label(
        bc.key_value,
        u16::conditional_select(&0, &(byte as u16), hit),
    )?;

    Ok(())
}

/// Fills one row's sorted access, `here`, and its
/// comparison with the next row's access, `next`.
fn fill_sorted<P: Fn(Col) -> errors::Result<usize>>(
    w: &mut Writer<'_, P>,
    bits: &mut [u32],
    sb: &SortBits,
    sc: &SortCols,
    here: Option<Access>,
    next: Option<Access>,
) -> errors::Result<()> {
    // Off sorted rows the access is zero, as the AIR pins it;
    // `same` needs both rows sorted, the last sorted row linking to nothing.
    let (a, b) = (here.unwrap_or_default(), next.unwrap_or_default());

    let both = Choice::from((here.is_some() && next.is_some()) as u8);
    let same = both & a.addr.ct_eq(&b.addr);

    pack_bits(bits, sb.addr, a.addr as u64, BYTE);
    pack_bits(bits, sb.time, a.time as u64, TIME_BITS);
    pack_one(bits, sb.write, a.write);
    pack_one(bits, sb.same, bool::from(same));

    fill_sub_borrow_packed(
        bits,
        sb.addr_diff,
        sb.addr_borrow,
        BYTE,
        a.addr as u64,
        b.addr as u64,
    );
    fill_sub_borrow_packed(
        bits,
        sb.time_diff,
        sb.time_borrow,
        TIME_BITS,
        a.time as u64,
        b.time as u64,
    );

    w.label(sc.addr, a.addr as u16)?;
    w.label(sc.time, a.time as u16)?;

    w.word(sc.value, a.value)?;
    w.word(sc.carried, u32::conditional_select(&0, &a.value, same))?;

    Ok(())
}

fn fill_increment(bits: &mut [u32], carry: usize, n: u32, u: bool) {
    // Carry k of n + u is u ∧ n_0 ∧ … ∧ n_k.
    let mut c = u;
    for k in 0..COUNT_BITS {
        c &= (n >> k) & 1 == 1;

        pack_one(bits, carry + k, c);
    }
}

fn cbd_value(x: u32, y: u32) -> u32 {
    let v = x + mlkem::Q - y;

    u32::conditional_select(&v, &v.wrapping_sub(mlkem::Q), !v.ct_lt(&mlkem::Q))
}

fn signed_one(bit: u8) -> u32 {
    u32::conditional_select(&1, &(mldsa::Q - 1), Choice::from(bit))
}

/// Sorts in time independent of the keys,
/// which carry SampleInBall's secret j_t.
fn sorted_by_key(log: &[Access]) -> Vec<Access> {
    let ranks = Zeroizing::new(
        log.iter()
            .map(|a| {
                log.iter().fold(0u32, |rank, b| {
                    rank + b.key().ct_lt(&a.key()).unwrap_u8() as u32
                })
            })
            .collect::<Vec<u32>>(),
    );

    (0..log.len() as u32)
        .map(|k| {
            log.iter()
                .zip(ranks.iter())
                .fold(Access::default(), |acc, (a, rank)| {
                    Access::conditional_select(&acc, a, rank.ct_eq(&k))
                })
        })
        .collect()
}

/// Bits `start..start + width` of little-endian words, bit 0 first.
fn field(words: &[u64], start: usize, width: usize) -> u32 {
    (0..width).fold(0, |acc, t| {
        let b = start + t;

        acc | ((((words[b / 64] >> (b % 64)) & 1) as u32) << t)
    })
}

/// Bits `start..start + width` of a byte stream, bit 0 first.
fn byte_field(bytes: &[u8], start: usize, width: usize) -> u32 {
    (0..width).fold(0, |acc, t| {
        let b = start + t;

        acc | ((((bytes[b / 8] >> (b % 8)) & 1) as u32) << t)
    })
}

/// Applies the Swap and Tail forgeries of step `index`
/// to its witness; Flip acts while the step samples.
fn forge(
    witness: &mut StepWitness,
    step: &SamplerStep,
    index: usize,
    forgeries: &[SamplerForgery],
) {
    for forgery in forgeries {
        match *forgery {
            SamplerForgery::Swap {
                step: at,
                candidate,
            } if at == index && matches!(step.kind, Kind::Rej(_)) => {
                let n = witness.uses.iter().take(candidate).filter(|&&u| u).count();

                if n + 1 < N {
                    witness.coeffs.swap(n, n + 1);
                }

                witness.swaps.push(candidate);
            }
            SamplerForgery::Tail { step: at } if at == index && step.kind == Kind::Cbd(3) => {
                let lanes = &witness.calls[1].1;
                let window = [lanes[RATE_256 - 2], lanes[RATE_256 - 1], lanes[0]];

                let (slots, straddled) = (cbd_slots(3), RATE_256 / WINDOW);

                for m in 0..slots {
                    let x = field(&window, 6 * m, 3).count_ones();
                    let y = field(&window, 6 * m + 3, 3).count_ones();

                    witness.coeffs[straddled * slots + m] = cbd_value(x, y);
                }

                witness
                    .tail
                    .copy_from_slice(&lanes[RATE_256 - TAIL_LANES..RATE_256]);
            }
            SamplerForgery::Unsorted { step: at } if at == index => {
                if let Some(ball) = witness.ball.as_mut() {
                    unsort(&mut ball.sorted);
                }
            }
            _ => {}
        }
    }
}

fn unsort(sorted: &mut [Access]) {
    let pair = (1..sorted.len()).find(|&k| {
        let (a, b) = (sorted[k - 1], sorted[k]);
        let first = k == 1 || sorted[k - 2].addr != a.addr;

        first && a.addr == b.addr && a.write && !b.write && a.value == 0 && b.value == 0
    });

    if let Some(k) = pair {
        sorted.swap(k - 1, k);
    }
}
