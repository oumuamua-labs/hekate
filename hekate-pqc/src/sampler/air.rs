// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Constraints of the Sampler table: the sponge rows and window every
//! kind shares, then rejection, CBD, and SampleInBall with its memory.

use alloc::vec::Vec;
use hekate_math::TowerField;
use hekate_program::circuit::Col;
use hekate_program::constraint::builder::{ConstraintSystem, Expr};

use super::layout::{
    BYTE, BallBits, BallCols, COUNT_BITS, CbdBits, CbdCols, LANES, MemCols, RejBits, RejCols,
    SamplerLayout, SignCols, SortBits, SortCols, TIME_BITS, WINDOW,
};
use super::{RATE_256, TAIL_LANES};
use crate::gadgets::{bits, borrow_chain, const_bits, packed};
use crate::wiring::N;
use crate::{mldsa, mlkem};

/// Constrains the sponge rows, seed register, window
/// and tail, then each sampling kind the layout carries.
pub(crate) fn constrain<F: TowerField>(cs: &ConstraintSystem<F>, ly: &SamplerLayout) {
    let col = |c: Col| cs.col(c.index());

    let (keep, absorb, seed_keep) = (col(ly.keep), col(ly.absorb), col(ly.seed_keep));

    for l in 0..LANES {
        let lane = col(ly.state.at(l));

        // The state holds row to row. `keep` is off into absorb rows,
        // which the template sets, and into OUT rows, which the Keccak
        // table answers; a squeeze IN row keeps the previous OUT.
        cs.constrain(keep * (cs.next(ly.state.at(l).index()) + lane));

        // On absorb rows a lane is its pad lane ⊕ its seed lane; pads and
        // `seed_use` are pinned zero on every other row.
        let mut template = absorb * lane;

        if let Some(&(_, pad)) = ly.pads.iter().find(|&&(p, _)| p == l) {
            template = template + col(pad);
        }

        if l < ly.seed.len() {
            template = template + col(ly.seed_use.at(l)) * col(ly.seed.at(l));
        }

        cs.constrain_named("sampler_absorb", template);
    }

    for l in 0..ly.seed.len() {
        let seed = col(ly.seed.at(l));

        // The seed register holds from a stream's first row, where the lane
        // bus loads it; lanes past the seed length are pinned zero there.
        cs.constrain(seed_keep * (cs.next(ly.seed.at(l).index()) + seed));
        cs.constrain(col(ly.seed_zero.at(l)) * seed);
    }

    // The window is three state lanes at the row's pinned offset, or on
    // the straddle row the two tail lanes and lane 0; other rows hold zero.
    for k in 0..WINDOW {
        let mut selected = col(ly.window.at(k));
        for &(o, sel) in &ly.offsets {
            selected = selected + col(sel) * col(ly.state.at(o + k));
        }

        if let Some(tc) = &ly.tail {
            let source = match k < TAIL_LANES {
                true => col(tc.lanes.at(k)),
                false => col(ly.state.at(0)),
            };

            selected = selected + col(tc.straddle) * source;
        }

        cs.constrain_named("sampler_window", selected);
    }

    // The tail keeps lanes 15 and 16 of an η = 3 step's first
    // OUT for the straddle window. It is zero at absorb rows,
    // leaving no free cell before its capture.
    if let Some(tc) = &ly.tail {
        for k in 0..TAIL_LANES {
            let tail = col(tc.lanes.at(k));
            let lane = col(ly.state.at(RATE_256 - TAIL_LANES + k));

            cs.constrain_named("sampler_tail", absorb * tail);
            cs.constrain_named("sampler_tail", col(tc.capture) * (tail + lane));
            cs.constrain_named(
                "sampler_tail",
                col(tc.keep) * (cs.next(tc.lanes.at(k).index()) + tail),
            );
        }
    }

    if let (Some(rej), Some(rc)) = (&ly.rej, &ly.rej_cols) {
        rejection(cs, ly, rej, rc);
    }

    for (block, cc) in ly.cbd.iter().zip(&ly.cbd_cols) {
        cbd(cs, ly, block, cc);
    }

    if let (Some(bb), Some(bc)) = (&ly.ball, &ly.ball_cols) {
        ball_candidates(cs, ly, bb, bc);
        ball_sign(cs, ly, &bc.sign);
        ball_memory(cs, &bc.mem);
        ball_sorted(cs, &bb.sort, &bc.sort);
    }

    for k in ly.num_bits..ly.num_packed * 32 {
        cs.constrain(cs.col(k));
    }
}

/// RejNTTPoly and SampleNTT: a comparison against q per candidate
/// slot, the use counter, and a coefficient key per slot.
fn rejection<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    ly: &SamplerLayout,
    rej: &RejBits,
    rc: &RejCols,
) {
    let one = cs.one();
    let shape = rej.shape;

    let work = cs.col(rc.work.index());
    let poly = cs.col(ly.poly.index());
    let top = const_bits(cs, shape.q - 1, shape.width);

    let first = bits(cs, rej.count(0), COUNT_BITS);
    let last = bits(cs, rej.count(shape.slots), COUNT_BITS);
    let next: Vec<Expr<'a, F>> = (0..COUNT_BITS).map(|k| cs.next(rej.count(0) + k)).collect();

    // The use counter n runs across a step's rows: `carry` links a row's
    // last slot to the next row's first, and absorb rows restart it at 0.
    cs.constrain(cs.col(ly.carry.index()) * (packed(cs, &next) + packed(cs, &last)));
    cs.constrain(cs.col(ly.absorb.index()) * packed(cs, &first));

    for (m, slot) in rej.slots.iter().enumerate() {
        let candidate = bits(cs, ly.window_bits + shape.stride * m, shape.width);
        let borrow = bits(cs, slot.borrow, shape.width + 1);

        // The borrow out of (q − 1) − c is set iff c ≥ q
        borrow_chain(
            cs,
            &top,
            &candidate,
            &bits(cs, slot.result, shape.width),
            &borrow,
        );

        let count = bits(cs, rej.count(m), COUNT_BITS);
        let u = cs.col(slot.u);

        let accept = one + borrow[shape.width];
        let done = count[COUNT_BITS - 1];

        // A candidate is used on a rejection row when below q
        // and while fewer than 256 are used; done is bit 8 of n.
        cs.constrain_named("sampler_accept", u + work * accept * (one + done));

        increment(
            cs,
            &count,
            &bits(cs, rej.count(m + 1), COUNT_BITS),
            &bits(cs, slot.carry, COUNT_BITS),
            u,
        );

        // A used slot emits (POLY, n, c) and every other slot JUNK = (0, 0, 0).
        // The block budget leaves an even JUNK count, which cancels in char 2.
        cs.constrain(cs.col(rc.key_poly.at(m).index()) + u * poly);
        cs.constrain(cs.col(rc.key_pos.at(m).index()) + u * packed(cs, &count));
        cs.constrain_named(
            "sampler_value",
            cs.col(rc.key_value.at(m).index()) + u * packed(cs, &candidate),
        );
    }
}

/// SamplePolyCBD_η per window slot: popcounts, the order
/// of x and y, |x − y|, and the coefficient x − y mod q.
fn cbd<F: TowerField>(cs: &ConstraintSystem<F>, ly: &SamplerLayout, block: &CbdBits, cc: &CbdCols) {
    let one = cs.one();
    let eta = block.eta;

    let below_q = |d: u32| cs.constant(F::from(mlkem::Q - d));
    let small = |d: u32| cs.constant(F::from(d));

    // For η ≤ 3 bits, the popcount's bit 0 is Σ b
    // and its bit 1 is Σ_{i<j} b_i·b_j, both mod 2.
    let popcount = |start: usize, into: usize| {
        let b = bits(cs, start, eta);

        let parity = b.iter().fold(cs.col(into), |acc, &bit| acc + bit);

        let mut majority = cs.col(into + 1);
        for i in 0..eta {
            for j in i + 1..eta {
                majority = majority + b[i] * b[j];
            }
        }

        cs.constrain(parity);
        cs.constrain(majority);

        [cs.col(into), cs.col(into + 1)]
    };

    for (m, slot) in block.slots.iter().enumerate() {
        let base = ly.window_bits + 2 * eta * m;

        let x = popcount(base, slot.x);
        let y = popcount(base + eta, slot.y);

        let s = cs.col(slot.sign);
        let a = [cs.col(slot.diff), cs.col(slot.diff + 1)];
        let c = cs.col(slot.carry);

        let hi = [x[0] + s * (x[0] + y[0]), x[1] + s * (x[1] + y[1])];
        let lo = [y[0] + s * (x[0] + y[0]), y[1] + s * (x[1] + y[1])];

        // s = [x < y] orders (hi, lo) = (max, min).
        // a = hi − lo is checked as lo + a = hi
        // with no carry out, which a wrong s violates.
        cs.constrain(hi[0] + lo[0] + a[0]);
        cs.constrain(c + lo[0] * a[0]);
        cs.constrain(hi[1] + lo[1] + a[1] + c);
        cs.constrain(lo[1] * a[1] + lo[1] * c + a[1] * c);

        // s = 1 needs a ≠ 0, which forces s = 0 when x = y
        cs.constrain(s * (one + a[0]) * (one + a[1]));

        // f = x − y mod q:
        // a when s = 0, q − a when s = 1.
        let positive = a[0] * small(1) + a[1] * small(2);
        let negative = a[0] * (one + a[1]) * below_q(1)
            + (one + a[0]) * a[1] * below_q(2)
            + a[0] * a[1] * below_q(3);

        cs.constrain_named(
            "sampler_cbd",
            cs.col(cc.value.at(m).index()) + (one + s) * positive + s * negative,
        );
    }
}

/// SampleInBall candidates, one byte per row:
/// the take test against i, the counter i,
/// and the link key of a taken byte.
fn ball_candidates<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    ly: &SamplerLayout,
    bb: &BallBits,
    bc: &BallCols,
) {
    let one = cs.one();
    let work = cs.col(bc.work.index());

    // j is the byte of window lane 0 that the pinned one-hot
    // selects; rows without a selected byte hold j = 0.
    let byte = bits(cs, bb.byte, BYTE);
    for (k, &bit) in byte.iter().enumerate() {
        let mut selected = bit;
        for b in 0..BYTE {
            selected =
                selected + cs.col(bc.bytes.at(b).index()) * cs.col(ly.window_bits + BYTE * b + k);
        }

        cs.constrain(selected);
    }

    let count = bits(cs, bb.count(0), COUNT_BITS);
    let after = bits(cs, bb.count(1), COUNT_BITS);
    let next: Vec<Expr<'a, F>> = (0..COUNT_BITS).map(|k| cs.next(bb.count(0) + k)).collect();

    let first_cell = cs.constant(F::from(bb.shape.first_cell() as u32));

    // i runs from 256 − τ across the step: `carry` links rows, absorb
    // rows set 256 − τ, and done = bit 8 marks i = 256 after τ picks.
    cs.constrain(cs.col(ly.carry.index()) * (packed(cs, &next) + packed(cs, &after)));
    cs.constrain(cs.col(ly.absorb.index()) * (packed(cs, &count) + first_cell));

    let borrow = bits(cs, bb.borrow, BYTE + 1);

    // The borrow out of i − j is set iff j > i,
    // the test of FIPS 204 Algorithm 29.
    borrow_chain(cs, &count[..BYTE], &byte, &bits(cs, bb.diff, BYTE), &borrow);

    let u = cs.col(bb.u);
    let done = count[COUNT_BITS - 1];

    cs.constrain_named(
        "sampler_ball_accept",
        u + work * (one + borrow[BYTE]) * (one + done),
    );

    increment(cs, &count, &after, &bits(cs, bb.carry, COUNT_BITS), u);

    // Link keys sit at N + i, clear of the readout's 0..N
    let pos = packed(cs, &count) + cs.constant(F::from(N as u32));

    // Unused bytes emit JUNK; the byte budget
    // makes their count, budget − τ, even.
    cs.constrain(cs.col(bc.key_poly.index()) + u * cs.col(ly.poly.index()));
    cs.constrain(cs.col(bc.key_pos.index()) + u * pos);
    cs.constrain_named(
        "sampler_ball_value",
        cs.col(bc.key_value.index()) + u * packed(cs, &byte),
    );
}

/// The sign register S and the ±1 each step writes from it.
fn ball_sign<F: TowerField>(cs: &ConstraintSystem<F>, ly: &SamplerLayout, sign: &SignCols) {
    let one = cs.one();

    let lane = cs.col(sign.lane.index());
    let next = cs.next(sign.lane.index());
    let s = bits(cs, sign.bits.start(), sign.bits.len());

    let minus_one = cs.constant(F::from(mldsa::Q - 1));

    // S holds the sign lane: zero at absorb rows, the first 8 squeezed
    // bytes at the capture, then shifted right once per step row.
    cs.constrain(cs.col(ly.absorb.index()) * lane);
    cs.constrain(cs.col(sign.capture.index()) * (lane + cs.col(ly.state.at(0).index())));
    cs.constrain(cs.col(sign.keep.index()) * (next + lane));
    cs.constrain(cs.col(sign.shift.index()) * (next + packed(cs, &s[1..])));

    // Step row t sets c_j = (−1)^{h_t} from bit 0 of S,
    // which after t shifts is h_t: 1 when clear, q − 1 when set.
    cs.constrain_named(
        "sampler_sign",
        cs.col(sign.value.index()) + one + s[0] * (one + minus_one),
    );
}

/// Binds the program-order memory columns of step and readout rows;
/// the sib bus specs in `new` carry their accesses.
fn ball_memory<F: TowerField>(cs: &ConstraintSystem<F>, mem: &MemCols) {
    let one = cs.one();

    let read = cs.col(mem.read.index());
    let addr = cs.col(mem.addr.index());

    // Step row t reads c[j] at time 3t, writes c[i_t] = v at 3t + 1
    // and c[j] = ±1 at 3t + 2; readout row p reads c[p] at 3τ + p.
    // The sib bus carries every access to the sorted rows.
    cs.constrain((one + read) * addr);
    cs.constrain((one + read) * cs.col(mem.value.index()));
    cs.constrain(cs.col(mem.readout.index()) * (addr + cs.col(mem.pos.index())));
}

/// Read-after-write over the sorted accesses: strict (address, time)
/// order, and every read returns the value its address last held.
fn ball_sorted<'a, F: TowerField>(cs: &'a ConstraintSystem<F>, sb: &SortBits, sort: &SortCols) {
    let one = cs.one();
    let col = |c: Col| cs.col(c.index());

    let addr = bits(cs, sb.addr, BYTE);
    let time = bits(cs, sb.time, TIME_BITS);

    let next_addr: Vec<Expr<'a, F>> = (0..BYTE).map(|k| cs.next(sb.addr + k)).collect();
    let next_time: Vec<Expr<'a, F>> = (0..TIME_BITS).map(|k| cs.next(sb.time + k)).collect();

    let (write, same) = (cs.col(sb.write), cs.col(sb.same));
    let (sorted, link) = (col(sort.sorted), col(sort.link));

    // Sorted rows hold one access each in (address, time) order.
    // Address and time are bound to their bits, and every cell
    // is zero off sorted rows.
    cs.constrain(col(sort.addr) + packed(cs, &addr));
    cs.constrain(col(sort.time) + packed(cs, &time));

    for cell in [col(sort.addr), col(sort.time), col(sort.value), write] {
        cs.constrain((one + sorted) * cell);
    }

    let addr_borrow = bits(cs, sb.addr_borrow, BYTE + 1);
    let time_borrow = bits(cs, sb.time_borrow, TIME_BITS + 1);

    // The borrow out of addr − next addr is set iff
    // the next address is larger, and the same for time.
    borrow_chain(
        cs,
        &addr,
        &next_addr,
        &bits(cs, sb.addr_diff, BYTE),
        &addr_borrow,
    );
    borrow_chain(
        cs,
        &time,
        &next_time,
        &bits(cs, sb.time_diff, TIME_BITS),
        &time_borrow,
    );

    // same = [the next access has this address], on linked rows only.
    // The same address repeats with a rising time; a new address rises.
    cs.constrain((one + link) * same);
    cs.constrain_named(
        "sampler_sib_order",
        link * same * (cs.next(sort.addr.index()) + col(sort.addr)),
    );
    cs.constrain_named(
        "sampler_sib_order",
        link * (one + same) * (one + addr_borrow[BYTE]),
    );
    cs.constrain_named(
        "sampler_sib_order",
        link * same * (one + time_borrow[TIME_BITS]),
    );

    // A read returns the previous access's value at its address,
    // or zero on the address's first access: `carried` = same · value.
    cs.constrain(col(sort.carried) + same * col(sort.value));
    cs.constrain_named(
        "sampler_sib_read",
        link * (one + cs.next(sb.write)) * (cs.next(sort.value.index()) + col(sort.carried)),
    );

    // The first sorted row starts the first address:
    // a read there returns zero.
    cs.constrain_named(
        "sampler_sib_read",
        col(sort.first) * (one + write) * col(sort.value),
    );
}

/// after = count + u by ripple carry, with no carry out of the top bit.
fn increment<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    count: &[Expr<'a, F>],
    after: &[Expr<'a, F>],
    carry: &[Expr<'a, F>],
    u: Expr<'a, F>,
) {
    for k in 0..count.len() {
        let carry_in = if k == 0 { u } else { carry[k - 1] };

        cs.constrain_named("sampler_count", after[k] + count[k] + carry_in);
        cs.constrain_named("sampler_count", carry[k] + count[k] * carry_in);
    }

    cs.constrain_named("sampler_count", carry[count.len() - 1]);
}
