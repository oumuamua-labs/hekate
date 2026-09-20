// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec::Vec;
use core::array;
use hekate_core::errors;
use hekate_core::trace::{ColumnType, TraceBuilder};
use hekate_math::{Block32, HardwareField, TowerField};
use hekate_program::circuit::{Circuit, Col, ColRange, Packed};
use hekate_program::constraint::builder::ConstraintSystem;

use crate::sha256::{Word, constrain_feed_forward, feed_forward_carries, word_bits};
use crate::trace::Sha256Call;
use crate::{BLOCK_WORDS, STATE_WORDS, Sha256Chiplet};

/// One compression request on the CPU table: `h_in`, the
/// block, the chiplet's `state_out`, the feed-forward carries
/// and `h_out`, in that physical order from `phys_base`.
pub struct CpuSha256Block {
    pub h_in: Packed,
    pub h_in_words: ColRange,
    pub msg: ColRange,
    pub state_out: Packed,
    pub state_out_words: ColRange,
    pub carry: Packed,
    pub h_out: Packed,
    pub h_out_words: ColRange,

    phys_base: usize,
}

impl CpuSha256Block {
    pub const COLUMNS: usize = 4 * STATE_WORDS + BLOCK_WORDS;

    pub const H_IN: usize = 0;
    pub const MSG: usize = Self::H_IN + STATE_WORDS;
    pub const STATE_OUT: usize = Self::MSG + BLOCK_WORDS;
    pub const CARRY: usize = Self::STATE_OUT + STATE_WORDS;
    pub const H_OUT: usize = Self::CARRY + STATE_WORDS;

    pub fn layout() -> [ColumnType; Self::COLUMNS] {
        [ColumnType::B32; Self::COLUMNS]
    }

    /// `phys_base` is the block's first physical column in the CPU layout;
    /// the caller's declarations before this call determine it.
    pub fn declare<F: TowerField + HardwareField>(cx: &mut Circuit<F>, phys_base: usize) -> Self {
        let h_in = cx.expand_bits(STATE_WORDS, ColumnType::B32);
        let h_in_words = cx.reuse_pass_through(&h_in);
        let msg = cx.columns(BLOCK_WORDS, ColumnType::B32);
        let state_out = cx.expand_bits(STATE_WORDS, ColumnType::B32);
        let state_out_words = cx.reuse_pass_through(&state_out);
        let carry = cx.expand_bits(STATE_WORDS, ColumnType::B32);
        let h_out = cx.expand_bits(STATE_WORDS, ColumnType::B32);
        let h_out_words = cx.reuse_pass_through(&h_out);

        Self {
            h_in,
            h_in_words,
            msg,
            state_out,
            state_out_words,
            carry,
            h_out,
            h_out_words,
            phys_base,
        }
    }

    /// Wires the requester half of `sha256_rounds` and emits
    /// `h_out = h_in + state_out` ungated: every host row
    /// must satisfy it, padding rows by staying zero.
    pub fn connect<F: TowerField + HardwareField>(
        &self,
        cx: &mut Circuit<F>,
        selector: Col,
    ) -> errors::Result<()> {
        let values: Vec<Col> = self
            .h_in_words
            .iter()
            .chain(self.msg.iter())
            .chain(self.state_out_words.iter())
            .collect();

        cx.call(&Sha256Chiplet::<F>::service(), &values, selector)?;

        let cs = cx.cs();

        constrain_feed_forward(
            cs,
            &self.words(cs, &self.h_in),
            &self.words(cs, &self.state_out),
            &self.words(cs, &self.carry),
            &self.words(cs, &self.h_out),
        );

        Ok(())
    }

    /// `call` must be the chiplet trace's call for this row;
    /// the bus key spans `h_in`, the block and `state_out`.
    ///
    /// # Errors
    /// `row` outside the trace or a column type mismatch.
    pub fn write(
        &self,
        tb: &mut TraceBuilder,
        row: usize,
        call: &Sha256Call,
    ) -> errors::Result<()> {
        let state_out = call.state_out();
        let h_out = crate::feed_forward(&call.h_in, &state_out);
        let carries = feed_forward_carries(&call.h_in, &state_out);

        let groups = [
            (Self::H_IN, &call.h_in[..]),
            (Self::MSG, &call.block[..]),
            (Self::STATE_OUT, &state_out[..]),
            (Self::CARRY, &carries[..]),
            (Self::H_OUT, &h_out[..]),
        ];

        for (offset, words) in groups {
            for (i, &word) in words.iter().enumerate() {
                tb.set_b32(self.phys_base + offset + i, row, Block32::from(word))?;
            }
        }

        Ok(())
    }

    fn words<'a, F: TowerField>(
        &self,
        cs: &'a ConstraintSystem<F>,
        group: &Packed,
    ) -> [Word<'a, F>; STATE_WORDS] {
        array::from_fn(|i| word_bits(cs, group.bits(i)))
    }
}
