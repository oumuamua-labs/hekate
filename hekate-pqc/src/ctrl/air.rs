// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec::Vec;
use hekate_math::TowerField;
use hekate_program::circuit::Col;
use hekate_program::constraint::builder::{ConstraintSystem, Expr};

use super::layout::{CtrlLayout, TAILS};
use super::{LANES, PREFIX, RATE};

pub(super) fn constrain<'a, F: TowerField>(cs: &'a ConstraintSystem<F>, ly: &CtrlLayout) {
    let one = cs.one();
    let zero = cs.constant(F::ZERO);
    let v = cs.constant(F::from(1u128 << 32));

    let col = |c: Col| cs.col(c.index());

    let state: Vec<Expr<'a, F>> = ly.state.iter().map(col).collect();
    let next: Vec<Expr<'a, F>> = ly.state.iter().map(|c| cs.next(c.index())).collect();

    let word = col(ly.word);
    let lane = col(ly.lane);

    let word_next = cs.next(ly.word.index());
    let lane_next = cs.next(ly.lane.index());

    let (out, rot) = (col(ly.out), col(ly.rot));
    let (reset, prefix) = (col(ly.reset), col(ly.prefix));

    let held = one + out;
    let kept = held + rot;

    for i in 0..LANES {
        let cleared = match i {
            i if PREFIX < i && i < RATE => reset,
            _ => reset + prefix,
        };

        let step = match i {
            i if i >= RATE => held * state[i],
            _ => kept * state[i] + rot * state[(i + 1) % RATE],
        };

        let mut e = held * next[i] + step + cleared * state[i];

        if i == 0 {
            e = e + col(ly.lo) * word_next;
        }

        if i == RATE - 1 {
            e = e + v * col(ly.hi) * word_next + col(ly.lane_in) * lane_next + col(ly.pad);
        }

        cs.constrain_named("ctrl_sponge", e);
    }

    cs.constrain_named(
        "ctrl_split",
        col(ly.split) * (state[0] + word + v * word_next),
    );
    cs.constrain_named("ctrl_lane", col(ly.emit) * (lane + state[RATE - 1]));

    cs.constrain((one + col(ly.io)) * (one + col(ly.wsel)) * word);
    cs.constrain((one + col(ly.lsel)) * lane);

    let bits: Vec<Expr<'a, F>> = ly.word_bits.iter().map(col).collect();

    for t in 0..TAILS {
        let high = bits
            .iter()
            .enumerate()
            .skip(8 * (t + 1))
            .fold(zero, |acc, (b, &bit)| {
                acc + bit * cs.constant(F::from(1u128 << b))
            });

        cs.constrain_named("ctrl_tail", col(ly.tail.at(t)) * high);
    }
}
