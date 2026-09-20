// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::errors::{self, Error};
use hekate_core::trace::{ColumnTrace, TraceBuilder};
use hekate_math::{Bit, Block32, TowerField};

use crate::sha256::{ADDS_PER_ROUND, ROUND_ADDS, SCHEDULE_ADDS, Sha256Layout};
use crate::{
    BLOCK_WORDS, K, ROUNDS, STATE_WORDS, add_with_carries, big_sigma0, big_sigma1, ch, compress,
    maj, rounds, small_sigma0, small_sigma1,
};

const SCHEDULE_LEN: usize = ROUNDS + BLOCK_WORDS;

/// One compression request;
/// `request_idx` is the host row that emits it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sha256Call {
    pub h_in: [u32; STATE_WORDS],
    pub block: [u32; BLOCK_WORDS],
    pub request_idx: u32,
}

struct RoundWitness {
    ch: u32,
    maj: u32,
    carries: [u32; ROUND_ADDS],
}

struct RowWriter<'a> {
    tb: &'a mut TraceBuilder,
    row: usize,
}

impl Sha256Call {
    pub fn state_out(&self) -> [u32; STATE_WORDS] {
        rounds(&self.h_in, &self.block)
    }

    pub fn h_out(&self) -> [u32; STATE_WORDS] {
        compress(&self.h_in, &self.block)
    }
}

/// Lays each call out as `rows_per_block` contiguous
/// rows from row 0; padding rows stay zero.
///
/// # Errors
/// `calls.len()` differs from the `num_blocks` the
/// fixed schedule pins, `num_rows` is not a power
/// of two, or the blocks overflow the table.
pub fn generate_sha256_trace(
    layout: &Sha256Layout,
    calls: &[Sha256Call],
    num_blocks: usize,
    num_rows: usize,
) -> errors::Result<ColumnTrace> {
    let rows_per_block = layout.rows_per_block();

    if calls.len() != num_blocks {
        return Err(Error::Protocol {
            protocol: "sha256_chiplet",
            message: "call count differs from the num_blocks the fixed schedule pins",
        });
    }

    if !num_rows.is_power_of_two() {
        return Err(Error::Protocol {
            protocol: "sha256_chiplet",
            message: "num_rows must be a power of two",
        });
    }

    if calls.len().saturating_mul(rows_per_block) > num_rows {
        return Err(Error::Protocol {
            protocol: "sha256_chiplet",
            message: "trace overflow: more calls than the table holds",
        });
    }

    let num_vars = num_rows.trailing_zeros() as usize;

    let mut tb = TraceBuilder::new(layout.columns(), num_vars)?;
    for (b, call) in calls.iter().enumerate() {
        write_block(&mut tb, layout, b * rows_per_block, call)?;
    }

    Ok(tb.build())
}

fn write_block(
    tb: &mut TraceBuilder,
    layout: &Sha256Layout,
    first_row: usize,
    call: &Sha256Call,
) -> errors::Result<()> {
    let r = layout.rounds_per_row;
    let rows_per_block = layout.rows_per_block();
    let state_out = call.state_out();

    let mut schedule = [0u32; SCHEDULE_LEN];
    schedule[..BLOCK_WORDS].copy_from_slice(&call.block);

    let mut state = call.h_in;
    for j in 0..rows_per_block {
        let mut w = RowWriter {
            tb,
            row: first_row + j,
        };

        w.words(layout.state, &state)?;
        w.words(layout.window, &schedule[j * r..j * r + BLOCK_WORDS])?;
        w.words(layout.state_out, &state_out)?;

        for i in 0..r {
            let t = j * r + i;
            let round = round_with_carries(&mut state, K[t], schedule[t]);
            let (generated, schedule_carries) = schedule_with_carries(&schedule, t);

            schedule[t + BLOCK_WORDS] = generated;

            w.word(layout.k + i, K[t])?;
            w.word(layout.a_out + i, state[0])?;
            w.word(layout.e_out + i, state[4])?;
            w.word(layout.w_gen + i, generated)?;
            w.word(layout.ch + i, round.ch)?;
            w.word(layout.maj + i, round.maj)?;
            w.words(layout.carry + i * ADDS_PER_ROUND, &round.carries)?;
            w.words(
                layout.carry + i * ADDS_PER_ROUND + ROUND_ADDS,
                &schedule_carries,
            )?;
        }

        w.bit(layout.s_input, j == 0)?;
        w.bit(layout.s_mid, j + 1 < rows_per_block)?;
        w.bit(layout.s_last, j + 1 == rows_per_block)?;

        if j == 0 {
            w.word(layout.request_idx, call.request_idx)?;
        }
    }

    Ok(())
}

fn round_with_carries(state: &mut [u32; STATE_WORDS], k: u32, w: u32) -> RoundWitness {
    let [a, b, c, d, e, f, g, h] = *state;

    let ch = ch(e, f, g);
    let maj = maj(a, b, c);

    let (s1, c0) = add_with_carries(h, big_sigma1(e));
    let (s2, c1) = add_with_carries(s1, ch);
    let (s3, c2) = add_with_carries(s2, k);
    let (t1, c3) = add_with_carries(s3, w);
    let (t2, c4) = add_with_carries(big_sigma0(a), maj);
    let (e_new, c5) = add_with_carries(d, t1);
    let (a_new, c6) = add_with_carries(t1, t2);

    *state = [a_new, a, b, c, e_new, e, f, g];

    RoundWitness {
        ch,
        maj,
        carries: [c0, c1, c2, c3, c4, c5, c6],
    }
}

fn schedule_with_carries(w: &[u32; SCHEDULE_LEN], t: usize) -> (u32, [u32; SCHEDULE_ADDS]) {
    let (s1, c0) = add_with_carries(small_sigma1(w[t + 14]), w[t + 9]);
    let (s2, c1) = add_with_carries(s1, small_sigma0(w[t + 1]));
    let (generated, c2) = add_with_carries(s2, w[t]);

    (generated, [c0, c1, c2])
}

impl RowWriter<'_> {
    fn word(&mut self, col: usize, value: u32) -> errors::Result<()> {
        self.tb.set_b32(col, self.row, Block32::from(value))
    }

    fn words(&mut self, base: usize, values: &[u32]) -> errors::Result<()> {
        for (i, &value) in values.iter().enumerate() {
            self.word(base + i, value)?;
        }

        Ok(())
    }

    fn bit(&mut self, col: usize, on: bool) -> errors::Result<()> {
        self.tb
            .set_bit(col, self.row, if on { Bit::ONE } else { Bit::ZERO })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IV, pad_message, sha256_words};

    #[test]
    fn chained_calls_reach_the_digest() {
        let msg = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        let mut h = IV;

        for block in pad_message(msg) {
            let call = Sha256Call {
                h_in: h,
                block,
                request_idx: 0,
            };

            h = call.h_out();
        }

        assert_eq!(h, sha256_words(msg));
    }

    #[test]
    fn schedule_matches_the_reference() {
        let block: [u32; BLOCK_WORDS] = core::array::from_fn(|i| 0x0101_0101 * (i as u32 + 1));

        let mut ext = [0u32; SCHEDULE_LEN];
        ext[..BLOCK_WORDS].copy_from_slice(&block);

        let mut ring = block;
        for t in 0..ROUNDS {
            let (generated, _) = schedule_with_carries(&ext, t);
            ext[t + BLOCK_WORDS] = generated;

            if t >= BLOCK_WORDS {
                ring[t % BLOCK_WORDS] = crate::schedule_word(&ring, t);
                assert_eq!(ext[t], ring[t % BLOCK_WORDS], "W_{t}");
            }
        }
    }

    #[test]
    fn rows_fill_the_expected_span() {
        for r in [1, 4, 8, 16] {
            let layout = Sha256Layout::new(r).unwrap();
            let call = Sha256Call {
                h_in: IV,
                block: [0; BLOCK_WORDS],
                request_idx: 3,
            };

            let trace = generate_sha256_trace(&layout, &[call, call], 2, 256).unwrap();
            let s_last = trace.columns[layout.s_last].as_bit_slice().unwrap();
            let s_input = trace.columns[layout.s_input].as_bit_slice().unwrap();

            let ones = |bits: &[Bit]| bits.iter().filter(|&&b| b == Bit::ONE).count();

            assert_eq!(ones(s_input), 2);
            assert_eq!(ones(s_last), 2);
            assert_eq!(s_last[layout.rows_per_block() - 1], Bit::ONE);
            assert_eq!(s_input[layout.rows_per_block()], Bit::ONE);
        }

        let layout = Sha256Layout::new(8).unwrap();
        let call = Sha256Call {
            h_in: IV,
            block: [0; BLOCK_WORDS],
            request_idx: 0,
        };

        assert!(generate_sha256_trace(&layout, &[call; 3], 3, 16).is_err());
    }

    #[test]
    fn geometry_disagreements_are_rejected() {
        let layout = Sha256Layout::new(8).unwrap();
        let call = Sha256Call {
            h_in: IV,
            block: [0; BLOCK_WORDS],
            request_idx: 0,
        };

        assert!(generate_sha256_trace(&layout, &[call, call], 3, 256).is_err());
        assert!(generate_sha256_trace(&layout, &[call, call], 1, 256).is_err());
        assert!(generate_sha256_trace(&layout, &[call, call], 2, 100).is_err());
        assert!(generate_sha256_trace(&layout, &[call, call], 2, 256).is_ok());
    }
}
