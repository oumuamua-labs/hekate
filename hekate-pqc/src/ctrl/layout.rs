// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::errors::{Error, Result};
use hekate_core::trace::ColumnType;
use hekate_math::{HardwareField, TowerField};
use hekate_program::circuit::{Circuit, Col, ColRange};

use super::LANES;

pub(crate) const TAILS: usize = 3;

#[derive(Clone, Debug)]
pub struct CtrlLayout {
    pub state: ColRange,
    pub word: Col,
    pub word_bits: ColRange,
    pub lane: Col,

    pub io: Col,
    pub kec: Col,
    pub split: Col,
    pub emit: Col,
    pub wsel: Col,
    pub lsel: Col,
    pub tail: ColRange,
    pub wstream: Col,
    pub widx: Col,
    pub lstream: Col,
    pub lidx: Col,

    pub out: Col,
    pub rot: Col,
    pub reset: Col,
    pub prefix: Col,
    pub lo: Col,
    pub hi: Col,
    pub lane_in: Col,
    pub pad: Col,
}

impl CtrlLayout {
    pub(crate) fn declare<F: TowerField + HardwareField>(cx: &mut Circuit<F>) -> Self {
        let packed = cx.expand_bits(1, ColumnType::B32);
        let word = cx.reuse_pass_through(&packed).at(0);

        let state = cx.columns(LANES, ColumnType::B64);
        let lane = cx.column(ColumnType::B64);
        let pad = cx.column(ColumnType::B64);

        let wstream = cx.column(ColumnType::B16);
        let widx = cx.column(ColumnType::B16);
        let lstream = cx.column(ColumnType::B16);
        let lidx = cx.column(ColumnType::B16);

        let io = cx.column(ColumnType::Bit);
        let kec = cx.column(ColumnType::Bit);
        let split = cx.column(ColumnType::Bit);
        let emit = cx.column(ColumnType::Bit);
        let wsel = cx.column(ColumnType::Bit);
        let lsel = cx.column(ColumnType::Bit);
        let tail = cx.columns(TAILS, ColumnType::Bit);

        let out = cx.column(ColumnType::Bit);
        let rot = cx.column(ColumnType::Bit);
        let reset = cx.column(ColumnType::Bit);
        let prefix = cx.column(ColumnType::Bit);
        let lo = cx.column(ColumnType::Bit);
        let hi = cx.column(ColumnType::Bit);
        let lane_in = cx.column(ColumnType::Bit);

        Self {
            state,
            word,
            word_bits: packed.bits(0),
            lane,
            io,
            kec,
            split,
            emit,
            wsel,
            lsel,
            tail,
            wstream,
            widx,
            lstream,
            lidx,
            out,
            rot,
            reset,
            prefix,
            lo,
            hi,
            lane_in,
            pad,
        }
    }

    pub fn physical(&self, col: Col) -> Result<usize> {
        match col.index().checked_sub(self.word.index()) {
            Some(offset) => Ok(offset),
            None => Err(Error::Protocol {
                protocol: "ctrl_layout",
                message: "virtual bit has no committed column of its own",
            }),
        }
    }
}
