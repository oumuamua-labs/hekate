// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec::Vec;
use core::array;
use hekate_core::errors::{self, Error};
use hekate_core::trace::{ColumnTrace, ColumnType, TraceCompatibleField};
use hekate_math::{Flat, HardwareField, PackableField, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram, Col, ColRange, Packed};
use hekate_program::constraint::builder::{ConstraintSystem, Expr};
use hekate_program::permutation::{BusKind, Service, ServiceSlot};
use hekate_program::{Air, FixedShape};

use crate::trace::{Sha256Call, generate_sha256_trace};
use crate::{BLOCK_WORDS, K, ROUNDS, STATE_WORDS};

pub const BUS_ID: &str = "sha256_rounds";
pub const WORD_BITS: usize = 32;
pub const ROUND_ADDS: usize = 7;
pub const SCHEDULE_ADDS: usize = 3;
pub const ADDS_PER_ROUND: usize = ROUND_ADDS + SCHEDULE_ADDS;
pub const MAX_ROUNDS_PER_ROW: usize = BLOCK_WORDS;

#[rustfmt::skip]
pub const STATE_IN_LABELS: [&[u8]; STATE_WORDS] = [
    b"sha256_in_0", b"sha256_in_1", b"sha256_in_2", b"sha256_in_3",
    b"sha256_in_4", b"sha256_in_5", b"sha256_in_6", b"sha256_in_7",
];

#[rustfmt::skip]
pub const MSG_LABELS: [&[u8]; BLOCK_WORDS] = [
    b"sha256_msg_0",  b"sha256_msg_1",  b"sha256_msg_2",  b"sha256_msg_3",
    b"sha256_msg_4",  b"sha256_msg_5",  b"sha256_msg_6",  b"sha256_msg_7",
    b"sha256_msg_8",  b"sha256_msg_9",  b"sha256_msg_10", b"sha256_msg_11",
    b"sha256_msg_12", b"sha256_msg_13", b"sha256_msg_14", b"sha256_msg_15",
];

#[rustfmt::skip]
pub const STATE_OUT_LABELS: [&[u8]; STATE_WORDS] = [
    b"sha256_out_0", b"sha256_out_1", b"sha256_out_2", b"sha256_out_3",
    b"sha256_out_4", b"sha256_out_5", b"sha256_out_6", b"sha256_out_7",
];

/// Physical column offsets in declaration order.
#[derive(Clone, Debug)]
pub struct Sha256Layout {
    pub rounds_per_row: usize,
    pub state: usize,
    pub window: usize,
    pub state_out: usize,
    pub a_out: usize,
    pub e_out: usize,
    pub w_gen: usize,
    pub ch: usize,
    pub maj: usize,
    pub carry: usize,
    pub k: usize,
    pub s_input: usize,
    pub s_mid: usize,
    pub s_last: usize,
    pub request_idx: usize,
    columns: Vec<ColumnType>,
}

pub type Word<'a, F> = [Expr<'a, F>; WORD_BITS];

#[derive(Clone, Copy, Debug)]
pub struct Sha256Cols {
    pub state: Packed,
    pub state_words: ColRange,
    pub window: Packed,
    pub window_words: ColRange,
    pub state_out_words: ColRange,
    pub a_out: Packed,
    pub a_out_words: ColRange,
    pub e_out: Packed,
    pub e_out_words: ColRange,
    pub w_gen: Packed,
    pub w_gen_words: ColRange,
    pub ch: Packed,
    pub maj: Packed,
    pub carry: Packed,
    pub k: Packed,
    pub k_words: ColRange,
    pub s_input: Col,
    pub s_mid: Col,
    pub s_last: Col,
    pub request_idx: Col,
}

struct RoundCols<'a, F: TowerField> {
    k: Word<'a, F>,
    ch: Word<'a, F>,
    maj: Word<'a, F>,
    a_out: Word<'a, F>,
    e_out: Word<'a, F>,
    carries: [Word<'a, F>; ROUND_ADDS],
}

#[derive(Clone, Copy)]
enum Slot {
    State(usize),
    A(usize),
    E(usize),
}

impl Sha256Layout {
    /// # Errors
    /// `rounds_per_row` zero, above 16, or not a divisor of 64.
    pub fn new(rounds_per_row: usize) -> errors::Result<Self> {
        if rounds_per_row == 0
            || rounds_per_row > MAX_ROUNDS_PER_ROW
            || !ROUNDS.is_multiple_of(rounds_per_row)
        {
            return Err(Error::Protocol {
                protocol: "sha256_chiplet",
                message: "rounds_per_row must divide 64 and be at most 16",
            });
        }

        let mut columns = Vec::new();

        let mut alloc = |ty: ColumnType, count: usize| {
            let start = columns.len();
            columns.extend(core::iter::repeat_n(ty, count));

            start
        };

        let state = alloc(ColumnType::B32, STATE_WORDS);
        let window = alloc(ColumnType::B32, BLOCK_WORDS);
        let state_out = alloc(ColumnType::B32, STATE_WORDS);
        let a_out = alloc(ColumnType::B32, rounds_per_row);
        let e_out = alloc(ColumnType::B32, rounds_per_row);
        let w_gen = alloc(ColumnType::B32, rounds_per_row);
        let ch = alloc(ColumnType::B32, rounds_per_row);
        let maj = alloc(ColumnType::B32, rounds_per_row);
        let carry = alloc(ColumnType::B32, ADDS_PER_ROUND * rounds_per_row);
        let k = alloc(ColumnType::B32, rounds_per_row);
        let s_input = alloc(ColumnType::Bit, 1);
        let s_mid = alloc(ColumnType::Bit, 1);
        let s_last = alloc(ColumnType::Bit, 1);
        let request_idx = alloc(ColumnType::B32, 1);

        Ok(Self {
            rounds_per_row,
            state,
            window,
            state_out,
            a_out,
            e_out,
            w_gen,
            ch,
            maj,
            carry,
            k,
            s_input,
            s_mid,
            s_last,
            request_idx,
            columns,
        })
    }

    pub fn columns(&self) -> &[ColumnType] {
        &self.columns
    }

    pub fn rows_per_block(&self) -> usize {
        ROUNDS / self.rounds_per_row
    }

    pub fn row_bytes(&self) -> usize {
        self.columns.iter().map(|t| t.byte_size()).sum()
    }
}

/// The 64 rounds as `64 / rounds_per_row` rows per block,
/// one bus emit per block on its first row. The feed-forward
/// add is the host's: see [`constrain_feed_forward`].
#[derive(Clone)]
pub struct Sha256Chiplet<F: TowerField> {
    program: CircuitProgram<F>,
    layout: Sha256Layout,
    cols: Sha256Cols,
    num_rows: usize,
    num_blocks: usize,
}

impl<F: TowerField> Sha256Chiplet<F> {
    /// Both endpoints derive from this schema:
    /// the input state, the message block,
    /// the output state, the request-index clock.
    pub fn service() -> Service {
        let mut slots = Vec::with_capacity(STATE_WORDS + BLOCK_WORDS + STATE_WORDS + 1);

        for label in STATE_IN_LABELS
            .iter()
            .chain(&MSG_LABELS)
            .chain(&STATE_OUT_LABELS)
        {
            slots.push(ServiceSlot::Value(label));
        }

        slots.push(ServiceSlot::RequestIdx { num_bytes: 4 });

        Service {
            bus_id: BUS_ID,
            kind: BusKind::Permutation,
            slots,
            clock_waiver: None,
        }
    }
}

impl<F> Sha256Chiplet<F>
where
    F: TowerField + TraceCompatibleField + PackableField + HardwareField + Send + 'static,
    <F as PackableField>::Packed: Copy + Send + Sync,
    Flat<F>: Send + Sync,
{
    /// Blocks sit contiguously at rows
    /// `0..rows_per_block · num_blocks`.
    ///
    /// # Errors
    /// `num_rows` not a power of two, `num_blocks` zero or
    /// exceeding the table, `rounds_per_row` not a divisor of 64.
    pub fn new(num_rows: usize, num_blocks: usize, rounds_per_row: usize) -> errors::Result<Self> {
        let layout = Sha256Layout::new(rounds_per_row)?;

        let span = num_blocks
            .checked_mul(layout.rows_per_block())
            .ok_or(Error::Protocol {
                protocol: "sha256_chiplet",
                message: "num_blocks exceeds the trace height",
            })?;

        if num_blocks == 0 || span > num_rows {
            return Err(Error::Protocol {
                protocol: "sha256_chiplet",
                message: "num_blocks must be 1..=num_rows / rows_per_block",
            });
        }

        let mut cx = Circuit::<F>::new("Sha256Chiplet", num_rows)?;

        let cols = declare(&mut cx, rounds_per_row);

        fix_schedule(&mut cx, &cols, &layout, num_blocks);

        let values: Vec<usize> = cols
            .state_words
            .iter()
            .chain(cols.window_words.iter())
            .chain(cols.state_out_words.iter())
            .map(Col::index)
            .collect();

        let spec =
            Self::service().respond(&values, &[cols.request_idx.index()], cols.s_input.index())?;

        cx.bus(BUS_ID, spec);

        constrain(&cx, &cols, rounds_per_row);

        let program = cx.compile()?;

        if program.column_layout() != layout.columns() {
            return Err(Error::Protocol {
                protocol: "sha256_chiplet",
                message: "physical layout diverged from the circuit declaration",
            });
        }

        Ok(Self {
            program,
            layout,
            cols,
            num_rows,
            num_blocks,
        })
    }

    pub fn def(&self) -> errors::Result<ChipletDef<F>> {
        ChipletDef::from_air(&self.program)
    }

    pub fn program(&self) -> &CircuitProgram<F> {
        &self.program
    }

    pub fn layout(&self) -> &Sha256Layout {
        &self.layout
    }

    pub fn cols(&self) -> &Sha256Cols {
        &self.cols
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// # Errors
    /// `calls.len()` differs from the `num_blocks`
    /// the fixed schedule was built for.
    pub fn trace(&self, calls: &[Sha256Call]) -> errors::Result<ColumnTrace> {
        generate_sha256_trace(&self.layout, calls, self.num_blocks, self.num_rows)
    }
}

/// Host-side `h_out = h_in + state`, eight 32-bit adds
/// over bit-expanded columns; `carry[i]` holds the carries
/// of word `i` as [`feed_forward_carries`] lays them out.
pub fn constrain_feed_forward<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    h_in: &[Word<'a, F>; STATE_WORDS],
    state: &[Word<'a, F>; STATE_WORDS],
    carry: &[Word<'a, F>; STATE_WORDS],
    h_out: &[Word<'a, F>; STATE_WORDS],
) {
    for i in 0..STATE_WORDS {
        let sum = add32(cs, &h_in[i], &state[i], &carry[i]);
        define_word(cs, &h_out[i], &sum);
    }
}

pub fn feed_forward_carries(
    h_in: &[u32; STATE_WORDS],
    state: &[u32; STATE_WORDS],
) -> [u32; STATE_WORDS] {
    array::from_fn(|i| crate::add_with_carries(h_in[i], state[i]).1)
}

pub fn word_bits<F: TowerField>(cs: &ConstraintSystem<F>, range: ColRange) -> Word<'_, F> {
    array::from_fn(|k| cs.col(range.at(k).index()))
}

fn declare<F: TowerField + HardwareField>(
    cx: &mut Circuit<F>,
    rounds_per_row: usize,
) -> Sha256Cols {
    let state = cx.expand_bits(STATE_WORDS, ColumnType::B32);
    let state_words = cx.reuse_pass_through(&state);
    let window = cx.expand_bits(BLOCK_WORDS, ColumnType::B32);
    let window_words = cx.reuse_pass_through(&window);
    let state_out_words = cx.columns(STATE_WORDS, ColumnType::B32);
    let a_out = cx.expand_bits(rounds_per_row, ColumnType::B32);
    let a_out_words = cx.reuse_pass_through(&a_out);
    let e_out = cx.expand_bits(rounds_per_row, ColumnType::B32);
    let e_out_words = cx.reuse_pass_through(&e_out);
    let w_gen = cx.expand_bits(rounds_per_row, ColumnType::B32);
    let w_gen_words = cx.reuse_pass_through(&w_gen);
    let ch = cx.expand_bits(rounds_per_row, ColumnType::B32);
    let maj = cx.expand_bits(rounds_per_row, ColumnType::B32);
    let carry = cx.expand_bits(ADDS_PER_ROUND * rounds_per_row, ColumnType::B32);
    let k = cx.expand_bits(rounds_per_row, ColumnType::B32);
    let k_words = cx.reuse_pass_through(&k);
    let s_input = cx.column(ColumnType::Bit);
    let s_mid = cx.column(ColumnType::Bit);
    let s_last = cx.column(ColumnType::Bit);
    let request_idx = cx.column(ColumnType::B32);

    Sha256Cols {
        state,
        state_words,
        window,
        window_words,
        state_out_words,
        a_out,
        a_out_words,
        e_out,
        e_out_words,
        w_gen,
        w_gen_words,
        ch,
        maj,
        carry,
        k,
        k_words,
        s_input,
        s_mid,
        s_last,
        request_idx,
    }
}

fn fix_schedule<F: TowerField + HardwareField>(
    cx: &mut Circuit<F>,
    cols: &Sha256Cols,
    layout: &Sha256Layout,
    num_blocks: usize,
) {
    let rows = layout.rows_per_block();

    let cadence = |values: Vec<F>| FixedShape::Cadence {
        stride: rows,
        count: num_blocks,
        origin: 0,
        values,
    };

    let indicator = |pred: &dyn Fn(usize) -> bool| {
        cadence(
            (0..rows)
                .map(|off| if pred(off) { F::ONE } else { F::ZERO })
                .collect(),
        )
    };

    cx.fix(cols.s_input, indicator(&|off| off == 0));
    cx.fix(cols.s_mid, indicator(&|off| off + 1 < rows));
    cx.fix(cols.s_last, indicator(&|off| off + 1 == rows));

    for r in 0..layout.rounds_per_row {
        let values = (0..rows)
            .map(|off| F::from(u128::from(K[off * layout.rounds_per_row + r])))
            .collect();

        cx.fix(cols.k_words.at(r), cadence(values));
    }
}

fn constrain<F: TowerField + HardwareField>(
    cx: &Circuit<F>,
    cols: &Sha256Cols,
    rounds_per_row: usize,
) {
    let cs = cx.cs();

    let s_mid = cs.col(cols.s_mid.index());
    let s_last = cs.col(cols.s_last.index());

    let slot_bits = |slot: Slot| match slot {
        Slot::State(i) => word_bits(cs, cols.state.bits(i)),
        Slot::A(r) => word_bits(cs, cols.a_out.bits(r)),
        Slot::E(r) => word_bits(cs, cols.e_out.bits(r)),
    };

    let slot_word = |slot: Slot| match slot {
        Slot::State(i) => cols.state_words.at(i),
        Slot::A(r) => cols.a_out_words.at(r),
        Slot::E(r) => cols.e_out_words.at(r),
    };

    let mut slots: [Slot; STATE_WORDS] = array::from_fn(Slot::State);

    let mut schedule: Vec<Word<'_, F>> = (0..BLOCK_WORDS)
        .map(|i| word_bits(cs, cols.window.bits(i)))
        .collect();

    for r in 0..rounds_per_row {
        let carry = |j: usize| word_bits(cs, cols.carry.bits(r * ADDS_PER_ROUND + j));

        let round_cols = RoundCols {
            k: word_bits(cs, cols.k.bits(r)),
            ch: word_bits(cs, cols.ch.bits(r)),
            maj: word_bits(cs, cols.maj.bits(r)),
            a_out: word_bits(cs, cols.a_out.bits(r)),
            e_out: word_bits(cs, cols.e_out.bits(r)),
            carries: array::from_fn(carry),
        };

        round(cs, &slots.map(slot_bits), &schedule[r], &round_cols);

        slots = [
            Slot::A(r),
            slots[0],
            slots[1],
            slots[2],
            Slot::E(r),
            slots[4],
            slots[5],
            slots[6],
        ];

        let schedule_carries: [Word<'_, F>; SCHEDULE_ADDS] =
            array::from_fn(|j| carry(ROUND_ADDS + j));

        let generated = schedule_word(cs, &schedule, r, &schedule_carries);
        let committed = word_bits(cs, cols.w_gen.bits(r));

        define_word(cs, &committed, &generated);

        schedule.push(committed);
    }

    for (i, &slot) in slots.iter().enumerate() {
        let word = cs.col(slot_word(slot).index());
        let state_out = cs.col(cols.state_out_words.at(i).index());

        cs.assert_zero_when(s_mid, cs.next(cols.state_words.at(i).index()) + word);
        cs.assert_zero_when(
            s_mid,
            cs.next(cols.state_out_words.at(i).index()) + state_out,
        );
        cs.assert_zero_when(s_last, state_out + word);
    }

    for i in 0..BLOCK_WORDS {
        let next = cs.next(cols.window_words.at(i).index());
        let word = match i + rounds_per_row < BLOCK_WORDS {
            true => cols.window_words.at(i + rounds_per_row),
            false => cols.w_gen_words.at(i + rounds_per_row - BLOCK_WORDS),
        };

        cs.assert_zero_when(s_mid, next + cs.col(word.index()));
    }
}

/// Ungated:
/// an all-zero row satisfies every round root,
/// which keeps the composition at degree 2.
fn round<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    st: &[Word<'a, F>; STATE_WORDS],
    w: &Word<'a, F>,
    rc: &RoundCols<'a, F>,
) {
    let [a, b, c, d, e, f, g, h] = st;

    define_word(cs, &rc.ch, &array::from_fn(|i| g[i] + e[i] * (f[i] + g[i])));
    define_word(
        cs,
        &rc.maj,
        &array::from_fn(|i| b[i] + (a[i] + b[i]) * (b[i] + c[i])),
    );

    let s1 = add32(cs, h, &big_sigma1(cs, e), &rc.carries[0]);
    let s2 = add32(cs, &s1, &rc.ch, &rc.carries[1]);
    let s3 = add32(cs, &s2, &rc.k, &rc.carries[2]);
    let t1 = add32(cs, &s3, w, &rc.carries[3]);
    let t2 = add32(cs, &big_sigma0(cs, a), &rc.maj, &rc.carries[4]);
    let e_new = add32(cs, d, &t1, &rc.carries[5]);
    let a_new = add32(cs, &t1, &t2, &rc.carries[6]);

    define_word(cs, &rc.e_out, &e_new);
    define_word(cs, &rc.a_out, &a_new);
}

fn schedule_word<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    w: &[Word<'a, F>],
    t: usize,
    carries: &[Word<'a, F>; SCHEDULE_ADDS],
) -> Word<'a, F> {
    let s1 = add32(cs, &small_sigma1(cs, &w[t + 14]), &w[t + 9], &carries[0]);
    let s2 = add32(cs, &s1, &small_sigma0(cs, &w[t + 1]), &carries[1]);

    add32(cs, &s2, &w[t], &carries[2])
}

/// Carry roots only; the sum bits are expressions.
/// `carry[k]` is the carry out of bit `k`.
fn add32<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    a: &Word<'a, F>,
    b: &Word<'a, F>,
    carry: &Word<'a, F>,
) -> Word<'a, F> {
    let mut carry_in: Option<Expr<'a, F>> = None;

    array::from_fn(|k| {
        let (sum, majority) = match carry_in {
            None => (a[k] + b[k], a[k] * b[k]),
            Some(c) => (cs.sum(&[a[k], b[k], c]), (a[k] + c) * (b[k] + c) + c),
        };

        cs.constrain(carry[k] + majority);

        carry_in = Some(carry[k]);

        sum
    })
}

fn define_word<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    committed: &Word<'a, F>,
    value: &Word<'a, F>,
) {
    for k in 0..WORD_BITS {
        cs.constrain(committed[k] + value[k]);
    }
}

fn rotr<'a, F: TowerField>(w: &Word<'a, F>, n: usize) -> Word<'a, F> {
    array::from_fn(|k| w[(k + n) % WORD_BITS])
}

fn xor3<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    x: &Word<'a, F>,
    y: &Word<'a, F>,
    z: &Word<'a, F>,
) -> Word<'a, F> {
    array::from_fn(|k| cs.sum(&[x[k], y[k], z[k]]))
}

fn big_sigma0<'a, F: TowerField>(cs: &'a ConstraintSystem<F>, a: &Word<'a, F>) -> Word<'a, F> {
    xor3(cs, &rotr(a, 2), &rotr(a, 13), &rotr(a, 22))
}

fn big_sigma1<'a, F: TowerField>(cs: &'a ConstraintSystem<F>, e: &Word<'a, F>) -> Word<'a, F> {
    xor3(cs, &rotr(e, 6), &rotr(e, 11), &rotr(e, 25))
}

fn rotr_rotr_shr<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    w: &Word<'a, F>,
    r1: usize,
    r2: usize,
    s: usize,
) -> Word<'a, F> {
    array::from_fn(|k| {
        let rot = [w[(k + r1) % WORD_BITS], w[(k + r2) % WORD_BITS]];

        match k + s < WORD_BITS {
            true => cs.sum(&[rot[0], rot[1], w[k + s]]),
            false => rot[0] + rot[1],
        }
    })
}

fn small_sigma0<'a, F: TowerField>(cs: &'a ConstraintSystem<F>, w: &Word<'a, F>) -> Word<'a, F> {
    rotr_rotr_shr(cs, w, 7, 18, 3)
}

fn small_sigma1<'a, F: TowerField>(cs: &'a ConstraintSystem<F>, w: &Word<'a, F>) -> Word<'a, F> {
    rotr_rotr_shr(cs, w, 17, 19, 10)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hekate_math::Block128;
    use hekate_program::permutation::REQUEST_IDX_LABEL;
    use hekate_program::predicate::{ClaimLayout, compile};

    type F = Block128;

    #[test]
    fn layout_rejects_non_divisors() {
        assert!(Sha256Layout::new(0).is_err());
        assert!(Sha256Layout::new(3).is_err());
        assert!(Sha256Layout::new(32).is_err());
        assert!(Sha256Layout::new(64).is_err());

        for r in [1, 2, 4, 8, 16] {
            let layout = Sha256Layout::new(r).unwrap();

            assert_eq!(layout.rows_per_block(), 64 / r);
            assert_eq!(layout.columns().len(), 36 + 16 * r);
        }
    }

    #[test]
    fn new_validates() {
        assert!(Sha256Chiplet::<F>::new(16, 1, 8).is_ok());
        assert!(Sha256Chiplet::<F>::new(16, 2, 8).is_ok());
        assert!(Sha256Chiplet::<F>::new(16, 3, 8).is_err());
        assert!(Sha256Chiplet::<F>::new(16, 0, 8).is_err());
        assert!(Sha256Chiplet::<F>::new(15, 1, 8).is_err());
        assert!(Sha256Chiplet::<F>::new(16, 1, 5).is_err());
    }

    #[test]
    fn service_slots() {
        let spec = Sha256Chiplet::<F>::new(8, 1, 8)
            .unwrap()
            .program()
            .permutation_checks();

        assert_eq!(spec.len(), 1);
        assert_eq!(spec[0].0, BUS_ID);
        assert_eq!(spec[0].1.num_sources(), 33);
        assert_eq!(spec[0].1.sources[32].1, REQUEST_IDX_LABEL);
    }

    #[test]
    fn degree_stays_at_two() {
        for r in [1, 4, 16] {
            let chiplet = Sha256Chiplet::<F>::new(64, 1, r).unwrap();
            let ast = chiplet.program().constraint_ast();

            assert_eq!(ast.max_degree(), 2, "rounds_per_row = {r}");
        }
    }

    #[test]
    fn root_count() {
        let roots = |r: usize| {
            Sha256Chiplet::<F>::new(64, 1, r)
                .unwrap()
                .program()
                .constraint_ast()
                .roots
                .len()
        };

        let per_round = (ROUND_ADDS + SCHEDULE_ADDS + 2 + 2 + 1) * WORD_BITS;
        let boundary = 3 * STATE_WORDS + BLOCK_WORDS;

        assert_eq!(roots(1), per_round + boundary);
        assert_eq!(roots(1), 520);
        assert_eq!(roots(8), 8 * per_round + boundary);
        assert_eq!(roots(16), 16 * per_round + boundary);
    }

    #[test]
    fn fixed_columns_cover_schedule() {
        let chiplet = Sha256Chiplet::<F>::new(32, 2, 4).unwrap();
        let pins = chiplet.program().fixed_columns();

        assert_eq!(pins.len(), 3 + 4);

        let one = Flat::from_raw(F::ONE);
        let zero = Flat::from_raw(F::ZERO);

        let at = |pin: usize, row: usize| pins[pin].shape.value_at_row(row, 5);
        let k = |t: usize| F::from(u128::from(K[t])).to_hardware();

        assert_eq!(at(0, 0), one);
        assert_eq!(at(0, 1), zero);
        assert_eq!(at(0, 16), one);
        assert_eq!(at(1, 14), one);
        assert_eq!(at(1, 15), zero);
        assert_eq!(at(2, 15), one);
        assert_eq!(at(2, 30), zero);
        assert_eq!(at(2, 31), one);
        assert_eq!(at(3, 0), k(0));
        assert_eq!(at(3, 1), k(4));
        assert_eq!(at(6, 1), k(7));
        assert_eq!(at(3, 17), k(4));
        assert_eq!(at(3, 32), zero);
    }

    #[test]
    fn outer_statement_stays_sparse() {
        for r in [1, 8, 16] {
            let chiplet = Sha256Chiplet::<F>::new(64, 1, r).unwrap();
            let program = chiplet.program();
            let half = program.num_columns() as u32;
            let rows = compile(
                &program.constraint_ast(),
                ClaimLayout {
                    pad_first: 2 * half,
                    half,
                },
            );

            let widest = rows
                .affine
                .iter()
                .map(|row| row.unknowns.len() + row.claims.len())
                .max()
                .unwrap();

            let per_mul = rows.affine.len() / rows.mul_nodes as usize;

            assert_eq!(per_mul, 2, "rounds_per_row = {r}");
            assert_eq!(widest, 25, "rounds_per_row = {r}");

            let per_round = (ADDS_PER_ROUND + 2) * WORD_BITS;
            let gated = 3 * STATE_WORDS + BLOCK_WORDS;

            assert_eq!(
                rows.mul_nodes as usize,
                r * per_round + gated,
                "rounds_per_row = {r}"
            );
        }
    }
}
