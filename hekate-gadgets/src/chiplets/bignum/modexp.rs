// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Proves `s^65537 mod N` for a 2048-bit `N`.
//!
//! # Table shape
//!
//! 32 modmul blocks of 32 rows. Blocks 0..15 square, block 16
//! multiplies by the base, blocks 17..31 keep squaring, which
//! keeps every block the same AIR. The result sits on the last
//! row of block 16 and is emitted once on the bus.
//!
//! # One modmul
//!
//! Proves `x · y = q · N + r` one base-2^32 digit at a time.
//! Row `i` multiplies x-limb `i` and q-limb `i` (64-bit) by all
//! 64 limbs of `y` and `N` (32-bit), accumulating both sides
//! into running column products. It then finalises columns
//! `2i` and `2i+1` of its own block, and `64+2i`, `65+2i` of
//! the previous block: the tail is pipelined cyclically across
//! the trace, which is what buys 32 rows per block instead of
//! 64. `r < N` is proven as the addition `r + w + 1 = N` with
//! one-bit carries.
//!
//! # Why the arithmetic is field work
//!
//! Integers travel as `G^x` for a primitive `G` of GF(2^128)*
//! (see [`crate::atoms::exp_form`]), turning an integer sum
//! into a field product and a shift into Frobenius squarings.
//! Column sums stay below `2^101` and carries below `2^70`,
//! both under the group order `2^128 - 1`. Past it the exponent
//! wraps and the relation proves nothing.

use alloc::vec;
use alloc::vec::Vec;
use hekate_core::errors;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate_math::{Bit, Block32, Block64, Block128, Flat, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram, Col, ColRange, Packed};
use hekate_program::constraint::builder::Expr;
use hekate_program::permutation::{BusKind, ChallengeLabel, Service, ServiceSlot};
use hekate_program::{Air, FixedShape};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::atoms::exp_form::{
    ExpBasis, constrain_exp_chain2, constrain_fourth_powers, constrain_squarings,
};
use crate::chiplets::bignum::bigint::{
    column_sums, divrem, is_less, limb_pair, normalise, sub_one_minus,
};

type F = Block128;

pub const BUS_ID: &str = "bignum_modexp";

const MODULUS_LABEL: ChallengeLabel = b"kappa_modexp_n";
const BASE_LABEL: ChallengeLabel = b"kappa_modexp_base";
const RESULT_LABEL: ChallengeLabel = b"kappa_modexp_result";

pub const LIMBS32: usize = 64;
pub const NUM_MODMULS: usize = 32;
pub const FINAL_BLOCK: usize = 16;
pub const ROWS_PER_MODMUL: usize = 32;
pub const NUM_ROWS: usize = NUM_MODMULS * ROWS_PER_MODMUL;
pub const RESULT_ROW: usize = (FINAL_BLOCK + 1) * ROWS_PER_MODMUL - 1;

pub const X_BITS: usize = 64;
pub const Y_BITS: usize = 32;
pub const DIGIT_BITS: usize = 32;
pub const CARRY_BITS: usize = 70;
pub const CARRY_HI_BITS: usize = CARRY_BITS - X_BITS;
pub const FROB_STEPS: usize = DIGIT_BITS / 2;
pub const CARRY_WORDS: usize = 8;

/// Tower coordinate of `2^32` in a packed B64 view:
/// `Block64(hi << 32 | lo) = lo + V32 · hi`.
pub const V32: u128 = 1 << 32;

pub const ARR_S: usize = 0;
pub const ARR_X: usize = 1;
pub const ARR_Y: usize = 2;
pub const ARR_N: usize = 3;
pub const ARR_R: usize = 4;
pub const ARR_W: usize = 5;

const COLUMNS: usize = 128;
const NUM_VARS: usize = NUM_ROWS.trailing_zeros() as usize;

const CARRY_HI_MASK: u64 = (1 << CARRY_HI_BITS) - 1;

/// One `s^65537 mod N` per table, emitted once
/// on [`RESULT_ROW`].
pub struct ModexpChiplet {
    program: CircuitProgram<F>,
    layout: Layout,
    cols: ModexpCols,
    basis: ExpBasis<F>,
}

impl ModexpChiplet {
    /// # Errors
    /// No primitive element, or the compiled layout
    /// diverges from the declaration.
    pub fn new() -> errors::Result<Self> {
        let basis = ExpBasis::<F>::new()?;
        let layout = Layout::new();
        let (program, cols) = build_program(&basis)?;

        if program.column_layout() != layout.physical.as_slice() {
            return Err(errors::Error::Protocol {
                protocol: "modexp_chiplet",
                message: "physical layout diverged from the circuit declaration",
            });
        }

        Ok(Self {
            program,
            layout,
            cols,
            basis,
        })
    }

    pub fn cols(&self) -> &ModexpCols {
        &self.cols
    }

    pub fn basis(&self) -> &ExpBasis<F> {
        &self.basis
    }

    /// # Errors
    /// The chiplet snapshot fails structural validation.
    pub fn def(&self) -> errors::Result<ChipletDef<F>> {
        ChipletDef::from_air(&self.program)
    }

    pub fn program(&self) -> &CircuitProgram<F> {
        &self.program
    }

    pub fn column_layout(&self) -> &[ColumnType] {
        &self.layout.physical
    }

    pub fn row_bytes(&self) -> usize {
        self.layout.row_bytes()
    }

    pub fn b128_columns(&self) -> usize {
        self.layout.b128_columns()
    }

    pub fn result_column(&self, limb: usize) -> usize {
        self.layout.arrays[ARR_R] + limb
    }

    pub fn emit_column(&self) -> usize {
        self.layout.emit
    }

    /// # Errors
    /// A row's bookkeeping does not satisfy the carry splits.
    pub fn trace(&self, modexp: &Modexp, request_idx: u32) -> errors::Result<ColumnTrace> {
        generate_trace(&self.basis, &self.layout, modexp, request_idx)
    }
}

// =================================================================
// Physical layout
// =================================================================

struct Layout {
    xs: usize,
    q: usize,
    arrays: [usize; 6],
    sel: usize,
    digits: usize,
    carry_lo: usize,
    carry_hi: usize,
    request_idx: usize,
    rc: usize,
    rcm: usize,
    first: usize,
    block_end: usize,
    advance: usize,
    into_final: usize,
    emit: usize,
    onehot: usize,
    xa: [usize; 2],
    yf: [usize; 2],
    p: [usize; 2],
    u: [usize; 2],
    ut: [usize; 2],
    c: [usize; 2],
    ct: [usize; 2],
    cm: [usize; 2],
    co: [usize; 2],
    ctm: [usize; 2],
    cto: [usize; 2],
    dg: [usize; 4],
    ce: [usize; 8],
    fr: [usize; 8],
    lch: [usize; 6],
    physical: Vec<ColumnType>,
}

impl Layout {
    fn new() -> Self {
        let mut physical = Vec::new();
        let mut alloc = |ty: ColumnType, count: usize| {
            let start = physical.len();
            physical.extend(core::iter::repeat_n(ty, count));

            start
        };

        let xs = alloc(ColumnType::B64, 1);
        let q = alloc(ColumnType::B64, 1);

        let arrays = [0; 6].map(|_| alloc(ColumnType::B32, LIMBS32));

        let sel = alloc(ColumnType::B32, 6);
        let digits = alloc(ColumnType::B32, 4);
        let carry_lo = alloc(ColumnType::B64, CARRY_WORDS);
        let carry_hi = alloc(ColumnType::B64, 1);
        let request_idx = alloc(ColumnType::B32, 1);
        let rc = alloc(ColumnType::Bit, 1);
        let rcm = alloc(ColumnType::Bit, 1);
        let first = alloc(ColumnType::Bit, 1);
        let block_end = alloc(ColumnType::Bit, 1);
        let advance = alloc(ColumnType::Bit, 1);
        let into_final = alloc(ColumnType::Bit, 1);
        let emit = alloc(ColumnType::Bit, 1);
        let onehot = alloc(ColumnType::Bit, ROWS_PER_MODMUL);

        let mut pair = |count: usize| {
            [
                alloc(ColumnType::B128, count),
                alloc(ColumnType::B128, count),
            ]
        };

        let xa = pair(half(X_BITS));
        let yf = pair(Y_BITS - 1);
        let p = pair(LIMBS32 * half(Y_BITS));
        let u = pair(LIMBS32);
        let ut = pair(LIMBS32);
        let c = pair(1);
        let ct = pair(1);
        let cm = pair(1);
        let co = pair(1);
        let ctm = pair(1);
        let cto = pair(1);

        let dg = [0; 4].map(|_| alloc(ColumnType::B128, half(DIGIT_BITS)));
        let ce = [0; 8].map(|_| alloc(ColumnType::B128, half(CARRY_BITS) - 1));
        let fr = [0; 8].map(|_| alloc(ColumnType::B128, FROB_STEPS));
        let lch = [0; 6].map(|_| alloc(ColumnType::B128, half(Y_BITS)));

        Self {
            xs,
            q,
            arrays,
            sel,
            digits,
            carry_lo,
            carry_hi,
            request_idx,
            rc,
            rcm,
            first,
            block_end,
            advance,
            into_final,
            emit,
            onehot,
            xa,
            yf,
            p,
            u,
            ut,
            c,
            ct,
            cm,
            co,
            ctm,
            cto,
            dg,
            ce,
            fr,
            lch,
            physical,
        }
    }

    fn row_bytes(&self) -> usize {
        self.physical.iter().map(|t| t.byte_size()).sum()
    }

    fn b128_columns(&self) -> usize {
        self.physical.len() - self.xa[0]
    }
}

/// One modmul `x · y = q · N + r` with the bookkeeping the AIR
/// mirrors: normalised digits and per-side carries.
struct Modmul {
    x: [u32; LIMBS32],
    y: [u32; LIMBS32],
    q: [u32; LIMBS32],
    r: [u32; LIMBS32],
    w: [u32; LIMBS32],
    digits: [u32; COLUMNS],
    carries: [[u128; COLUMNS + 1]; 2],
}

impl Modmul {
    fn compute(x: [u32; LIMBS32], y: [u32; LIMBS32], n: &[u32; LIMBS32]) -> errors::Result<Self> {
        let mut sums = [0u128; COLUMNS];
        let mut digits = [0u32; COLUMNS];
        let mut carries_a = [0u128; COLUMNS + 1];

        column_sums(&x, &y, &mut sums);
        normalise(&sums, &[], &mut digits, &mut carries_a);

        let mut q = [0u32; LIMBS32];
        let mut acc = [0u32; LIMBS32 + 1];

        divrem(&digits, n, &mut q, &mut acc)?;

        let mut r = [0u32; LIMBS32];
        r.copy_from_slice(&acc[..LIMBS32]);

        let mut w = [0u32; LIMBS32];
        sub_one_minus(n, &r, &mut w);

        let mut digits_b = [0u32; COLUMNS];
        let mut carries_b = [0u128; COLUMNS + 1];

        column_sums(&q, n, &mut sums);
        normalise(&sums, &r, &mut digits_b, &mut carries_b);

        if digits_b != digits {
            return Err(errors::Error::Protocol {
                protocol: "modexp_chiplet",
                message: "q · N + r does not reproduce x · y",
            });
        }

        Ok(Self {
            x,
            y,
            q,
            r,
            w,
            digits,
            carries: [carries_a, carries_b],
        })
    }
}

/// Per-row scratch, reused across rows; `co`, `cto` and
/// `rc_next` of the previous row seed the next row's carry-ins.
struct RowValues {
    xs: u64,
    q: u64,
    sel: [u32; 6],
    digits: [u32; 4],
    carries: [u128; CARRY_WORDS],
    rc: u8,
    rcm: u8,
    rc_next: u8,
    xa: [Vec<Flat<F>>; 2],
    yf: [Vec<Flat<F>>; 2],
    p: [Vec<Flat<F>>; 2],
    u: [Vec<Flat<F>>; 2],
    ut: [Vec<Flat<F>>; 2],
    c: [Flat<F>; 2],
    ct: [Flat<F>; 2],
    cm: [Flat<F>; 2],
    co: [Flat<F>; 2],
    ctm: [Flat<F>; 2],
    cto: [Flat<F>; 2],
    dg: [Vec<Flat<F>>; 4],
    ce: [Vec<Flat<F>>; 8],
    fr: [Vec<Flat<F>>; 8],
    lch: [Vec<Flat<F>>; 6],
}

impl RowValues {
    fn zeroed() -> Self {
        let one = Flat::from_raw(F::ONE);
        let cols = |n: usize| vec![Flat::from_raw(F::ZERO); n];

        Self {
            xs: 0,
            q: 0,
            sel: [0; 6],
            digits: [0; 4],
            carries: [0; CARRY_WORDS],
            rc: 0,
            rcm: 0,
            rc_next: 0,
            xa: [cols(half(X_BITS)), cols(half(X_BITS))],
            yf: [cols(Y_BITS - 1), cols(Y_BITS - 1)],
            p: [cols(LIMBS32 * half(Y_BITS)), cols(LIMBS32 * half(Y_BITS))],
            u: [cols(LIMBS32), cols(LIMBS32)],
            ut: [cols(LIMBS32), cols(LIMBS32)],
            c: [one; 2],
            ct: [one; 2],
            cm: [one; 2],
            co: [one; 2],
            ctm: [one; 2],
            cto: [one; 2],
            dg: [0; 4].map(|_| cols(half(DIGIT_BITS))),
            ce: [0; 8].map(|_| cols(half(CARRY_BITS))),
            fr: [0; 8].map(|_| cols(FROB_STEPS)),
            lch: [0; 6].map(|_| cols(half(Y_BITS))),
        }
    }
}

/// All [`NUM_MODMULS`] blocks, the dead squarings past [`FINAL_BLOCK`]
/// included: their tails close block 0's pipelined columns.
pub struct Modexp {
    n: [u32; LIMBS32],
    s: [u32; LIMBS32],
    blocks: Vec<Modmul>,
}

impl Modexp {
    /// # Errors
    /// `s >= n`, or a block's `q · N + r`
    /// does not reproduce `x · y`.
    pub fn new(n: &[u32; LIMBS32], s: &[u32; LIMBS32]) -> errors::Result<Self> {
        if !is_less(s, n) {
            return Err(errors::Error::Protocol {
                protocol: "modexp_chiplet",
                message: "base must be reduced: s < N",
            });
        }

        let mut blocks = Vec::with_capacity(NUM_MODMULS);
        let mut x = *s;

        for b in 0..NUM_MODMULS {
            let y = if b == FINAL_BLOCK { *s } else { x };
            let block = Modmul::compute(x, y, n)?;

            x = block.r;

            blocks.push(block);
        }

        Ok(Self {
            n: *n,
            s: *s,
            blocks,
        })
    }

    pub fn result(&self) -> &[u32; LIMBS32] {
        &self.blocks[FINAL_BLOCK].r
    }

    pub fn modulus(&self) -> &[u32; LIMBS32] {
        &self.n
    }

    pub fn base(&self) -> &[u32; LIMBS32] {
        &self.s
    }

    fn arrays(&self, b: usize) -> [&[u32; LIMBS32]; 6] {
        let block = &self.blocks[b];

        [&self.s, &block.x, &block.y, &self.n, &block.r, &block.w]
    }

    /// Exponent-form values of `row`; `u_state` and `ut_state`
    /// carry the running column products across rows, `out`
    /// carries the previous row's carry-outs.
    fn row(
        &self,
        basis: &ExpBasis<F>,
        row: usize,
        u_state: &mut [[Flat<F>; LIMBS32]; 2],
        ut_state: &mut [[Flat<F>; LIMBS32]; 2],
        out: &mut RowValues,
    ) {
        let b = row / ROWS_PER_MODMUL;
        let i = row % ROWS_PER_MODMUL;
        let block_start = i == 0;

        let one = Flat::from_raw(F::ONE);

        let block = &self.blocks[b];
        let prev = &self.blocks[(b + NUM_MODMULS - 1) % NUM_MODMULS];

        for side in 0..2 {
            out.c[side] = if block_start { one } else { out.co[side] };
            out.ct[side] = if block_start {
                out.co[side]
            } else {
                out.cto[side]
            };
        }

        out.rc = if block_start { 1 } else { out.rc_next };

        out.xs = limb_pair(&block.x, i);
        out.q = limb_pair(&block.q, i);

        out.sel = [
            block.r[2 * i],
            block.r[2 * i + 1],
            block.w[2 * i],
            block.w[2 * i + 1],
            self.n[2 * i],
            self.n[2 * i + 1],
        ];

        out.digits = [
            block.digits[2 * i],
            block.digits[2 * i + 1],
            prev.digits[LIMBS32 + 2 * i],
            prev.digits[LIMBS32 + 2 * i + 1],
        ];

        out.carries = [
            block.carries[0][2 * i + 1],
            block.carries[0][2 * i + 2],
            block.carries[1][2 * i + 1],
            block.carries[1][2 * i + 2],
            prev.carries[0][LIMBS32 + 2 * i + 1],
            prev.carries[0][LIMBS32 + 2 * i + 2],
            prev.carries[1][LIMBS32 + 2 * i + 1],
            prev.carries[1][LIMBS32 + 2 * i + 2],
        ];

        let sides = [(out.xs, &block.y), (out.q, &self.n)];
        for (side, (limb, other)) in sides.iter().enumerate() {
            basis.chain2(u128::from(*limb), &mut out.xa[side]);

            let g_a = out.xa[side][half(X_BITS) - 1];

            let mut frob = g_a;
            for slot in out.yf[side].iter_mut() {
                frob = frob * frob;
                *slot = frob;
            }

            let yf = &out.yf[side];

            let chain_of = |(chain, &limb): (&mut [Flat<F>], &u32)| {
                let mut acc = one;
                for (t, slot) in chain.iter_mut().enumerate() {
                    for s in [2 * t, 2 * t + 1] {
                        let frob = if s == 0 { g_a } else { yf[s - 1] };
                        let bit = Flat::from_raw(F::from(((limb >> s) & 1) as u8));

                        acc *= one + bit * (frob + one);
                    }

                    *slot = acc;
                }
            };

            #[cfg(feature = "parallel")]
            out.p[side]
                .par_chunks_mut(half(Y_BITS))
                .zip(other.par_iter())
                .for_each(chain_of);

            #[cfg(not(feature = "parallel"))]
            out.p[side]
                .chunks_mut(half(Y_BITS))
                .zip(other.iter())
                .for_each(chain_of);

            let mut next_u = [one; LIMBS32];
            let mut next_ut = [one; LIMBS32];

            for m in 0..LIMBS32 {
                let product = out.p[side][m * half(Y_BITS) + half(Y_BITS) - 1];
                let (prior_u, prior_ut) = match (block_start, m + 2 < LIMBS32) {
                    (_, false) => (one, one),
                    (true, true) => (one, u_state[side][m + 2]),
                    (false, true) => (u_state[side][m + 2], ut_state[side][m + 2]),
                };

                next_u[m] = prior_u * product;
                next_ut[m] = prior_ut;
            }

            u_state[side] = next_u;
            ut_state[side] = next_ut;

            out.u[side].copy_from_slice(&next_u);
            out.ut[side].copy_from_slice(&next_ut);
        }

        for (k, chain) in out.dg.iter_mut().enumerate() {
            basis.chain2(u128::from(out.digits[k]), chain);
        }

        for (k, chain) in out.ce.iter_mut().enumerate() {
            basis.chain2(out.carries[k], chain);
        }

        let end = |k: usize| out.ce[k][half(CARRY_BITS) - 1];

        out.cm = [end(0), end(2)];
        out.co = [end(1), end(3)];
        out.ctm = [end(4), end(6)];
        out.cto = [end(5), end(7)];

        for k in 0..CARRY_WORDS {
            let mut acc = end(k);
            for slot in out.fr[k].iter_mut() {
                acc = acc * acc;
                acc = acc * acc;
                *slot = acc;
            }
        }

        for (k, chain) in out.lch.iter_mut().enumerate() {
            basis.chain2(u128::from(out.sel[k]), chain);
        }

        let sum_lo = u64::from(out.sel[0]) + u64::from(out.sel[2]) + u64::from(out.rc);
        out.rcm = (sum_lo >> 32) as u8;

        let sum_hi = u64::from(out.sel[1]) + u64::from(out.sel[3]) + u64::from(out.rcm);
        out.rc_next = (sum_hi >> 32) as u8;
    }
}

impl RowValues {
    fn check(&self) -> errors::Result<()> {
        let digit = |d: usize| self.dg[d][half(DIGIT_BITS) - 1];
        let frob32 = |k: usize| self.fr[k][FROB_STEPS - 1];
        let r_term = |w: usize| self.lch[w][half(Y_BITS) - 1];

        let sum_lo = u64::from(self.sel[0]) + u64::from(self.sel[2]) + u64::from(self.rc);
        let sum_hi = u64::from(self.sel[1]) + u64::from(self.sel[3]) + u64::from(self.rcm);

        let checks = [
            self.u[0][0] * self.c[0] == digit(0) * frob32(0),
            self.u[0][1] * self.cm[0] == digit(1) * frob32(1),
            self.u[1][0] * self.c[1] * r_term(0) == digit(0) * frob32(2),
            self.u[1][1] * self.cm[1] * r_term(1) == digit(1) * frob32(3),
            self.ut[0][0] * self.ct[0] == digit(2) * frob32(4),
            self.ut[0][1] * self.ctm[0] == digit(3) * frob32(5),
            self.ut[1][0] * self.ct[1] == digit(2) * frob32(6),
            self.ut[1][1] * self.ctm[1] == digit(3) * frob32(7),
            (sum_lo as u32) == self.sel[4],
            (sum_hi as u32) == self.sel[5],
        ];

        if checks.iter().any(|ok| !ok) {
            return Err(errors::Error::Protocol {
                protocol: "modexp_chiplet",
                message: "row bookkeeping does not satisfy the carry splits",
            });
        }

        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ModexpCols {
    pub xs_bits: ColRange,
    pub q_bits: ColRange,
    pub xs_packed: Col,
    pub arrays: [ColRange; 6],
    pub y_bits: Packed,
    pub n_bits: Packed,
    pub sel_bits: Packed,
    pub sel_packed: ColRange,
    pub digit_bits: Packed,
    pub carry_lo_bits: Packed,
    pub carry_hi_bits: ColRange,
    pub request_idx: Col,
    pub rc: Col,
    pub rcm: Col,
    pub first: Col,
    pub block_end: Col,
    pub advance: Col,
    pub into_final: Col,
    pub emit: Col,
    pub onehot: ColRange,
    pub xa: [ColRange; 2],
    pub yf: [ColRange; 2],
    pub p: [ColRange; 2],
    pub u: [ColRange; 2],
    pub ut: [ColRange; 2],
    pub c: [Col; 2],
    pub ct: [Col; 2],
    pub cm: [Col; 2],
    pub co: [Col; 2],
    pub ctm: [Col; 2],
    pub cto: [Col; 2],
    pub dg: [ColRange; 4],
    pub ce: [ColRange; 8],
    pub fr: [ColRange; 8],
    pub lch: [ColRange; 6],
}

/// Slot order is the `β` power schedule;
/// both endpoints derive their spec from here.
pub fn service() -> Service {
    let mut slots = Vec::with_capacity(3 * LIMBS32 + 1);
    for label in [MODULUS_LABEL, BASE_LABEL, RESULT_LABEL] {
        slots.extend(core::iter::repeat_n(ServiceSlot::Value(label), LIMBS32));
    }

    slots.push(ServiceSlot::RequestIdx { num_bytes: 4 });

    Service {
        bus_id: BUS_ID,
        kind: BusKind::Permutation,
        slots,
        clock_waiver: None,
    }
}

const fn half(bits: usize) -> usize {
    bits / 2
}

fn write_row(
    tb: &mut TraceBuilder,
    layout: &Layout,
    modexp: &Modexp,
    row: usize,
    v: &RowValues,
    request_idx: u32,
) -> errors::Result<()> {
    let b = row / ROWS_PER_MODMUL;
    let i = row % ROWS_PER_MODMUL;

    tb.set_b64(layout.xs, row, Block64::from(v.xs))?;
    tb.set_b64(layout.q, row, Block64::from(v.q))?;

    for (which, limbs) in modexp.arrays(b).iter().enumerate() {
        for (j, &limb) in limbs.iter().enumerate() {
            tb.set_b32(layout.arrays[which] + j, row, Block32::from(limb))?;
        }
    }

    for (k, &limb) in v.sel.iter().enumerate() {
        tb.set_b32(layout.sel + k, row, Block32::from(limb))?;
    }

    for (k, &digit) in v.digits.iter().enumerate() {
        tb.set_b32(layout.digits + k, row, Block32::from(digit))?;
    }

    let mut carry_hi = 0u64;
    for (k, &carry) in v.carries.iter().enumerate() {
        tb.set_b64(layout.carry_lo + k, row, Block64::from(carry as u64))?;
        carry_hi |= (((carry >> X_BITS) as u64) & CARRY_HI_MASK) << (CARRY_HI_BITS * k);
    }

    tb.set_b64(layout.carry_hi, row, Block64::from(carry_hi))?;

    let block_end = i + 1 == ROWS_PER_MODMUL;

    tb.set_bit(layout.rc, row, Bit::new(v.rc))?;
    tb.set_bit(layout.rcm, row, Bit::new(v.rcm))?;
    tb.set_bit(layout.first, row, Bit::new(u8::from(row == 0)))?;
    tb.set_bit(layout.block_end, row, Bit::new(u8::from(block_end)))?;
    tb.set_bit(
        layout.advance,
        row,
        Bit::new(u8::from(block_end && b + 1 < NUM_MODMULS)),
    )?;
    tb.set_bit(
        layout.into_final,
        row,
        Bit::new(u8::from(row + 1 == FINAL_BLOCK * ROWS_PER_MODMUL)),
    )?;
    tb.set_bit(layout.emit, row, Bit::new(u8::from(row == RESULT_ROW)))?;

    if row == RESULT_ROW {
        tb.set_b32(layout.request_idx, row, Block32::from(request_idx))?;
    }

    for k in 0..ROWS_PER_MODMUL {
        tb.set_bit(layout.onehot + k, row, Bit::new(u8::from(k == i)))?;
    }

    for side in 0..2 {
        tb.set_b128_array_flat(layout.xa[side], row, &v.xa[side])?;
        tb.set_b128_array_flat(layout.yf[side], row, &v.yf[side])?;
        tb.set_b128_array_flat(layout.p[side], row, &v.p[side])?;
        tb.set_b128_array_flat(layout.u[side], row, &v.u[side])?;
        tb.set_b128_array_flat(layout.ut[side], row, &v.ut[side])?;
        tb.set_b128_flat(layout.c[side], row, v.c[side])?;
        tb.set_b128_flat(layout.ct[side], row, v.ct[side])?;
        tb.set_b128_flat(layout.cm[side], row, v.cm[side])?;
        tb.set_b128_flat(layout.co[side], row, v.co[side])?;
        tb.set_b128_flat(layout.ctm[side], row, v.ctm[side])?;
        tb.set_b128_flat(layout.cto[side], row, v.cto[side])?;
    }

    for (k, chain) in v.dg.iter().enumerate() {
        tb.set_b128_array_flat(layout.dg[k], row, chain)?;
    }

    for (k, chain) in v.ce.iter().enumerate() {
        tb.set_b128_array_flat(layout.ce[k], row, &chain[..half(CARRY_BITS) - 1])?;
    }

    for (k, chain) in v.fr.iter().enumerate() {
        tb.set_b128_array_flat(layout.fr[k], row, chain)?;
    }

    for (k, chain) in v.lch.iter().enumerate() {
        tb.set_b128_array_flat(layout.lch[k], row, chain)?;
    }

    Ok(())
}

/// The last block is walked silently first: running
/// products and carry-outs wrap cyclically into row 0.
fn generate_trace(
    basis: &ExpBasis<F>,
    layout: &Layout,
    modexp: &Modexp,
    request_idx: u32,
) -> errors::Result<ColumnTrace> {
    let mut tb = TraceBuilder::new(&layout.physical, NUM_VARS)?;
    let mut values = RowValues::zeroed();
    let mut u_state = [[Flat::from_raw(F::ONE); LIMBS32]; 2];
    let mut ut_state = [[Flat::from_raw(F::ONE); LIMBS32]; 2];

    for row in NUM_ROWS - ROWS_PER_MODMUL..NUM_ROWS {
        modexp.row(basis, row, &mut u_state, &mut ut_state, &mut values);
    }

    for row in 0..NUM_ROWS {
        modexp.row(basis, row, &mut u_state, &mut ut_state, &mut values);

        values.check()?;

        write_row(&mut tb, layout, modexp, row, &values, request_idx)?;
    }

    Ok(tb.build())
}

/// Declaration order is the physical order `Layout` mirrors.
fn declare(cx: &mut Circuit<F>) -> ModexpCols {
    let limbs = cx.expand_bits(2, ColumnType::B64);
    let s = cx.columns(LIMBS32, ColumnType::B32);
    let x = cx.columns(LIMBS32, ColumnType::B32);
    let y_bits = cx.expand_bits(LIMBS32, ColumnType::B32);
    let n_bits = cx.expand_bits(LIMBS32, ColumnType::B32);
    let r = cx.columns(LIMBS32, ColumnType::B32);
    let w = cx.columns(LIMBS32, ColumnType::B32);
    let sel_bits = cx.expand_bits(6, ColumnType::B32);
    let digit_bits = cx.expand_bits(4, ColumnType::B32);
    let carry_lo_bits = cx.expand_bits(CARRY_WORDS, ColumnType::B64);
    let carry_hi_word = cx.expand_bits(1, ColumnType::B64);

    let limbs_packed = cx.reuse_pass_through(&limbs);
    let y = cx.reuse_pass_through(&y_bits);
    let n = cx.reuse_pass_through(&n_bits);
    let sel_packed = cx.reuse_pass_through(&sel_bits);

    let request_idx = cx.column(ColumnType::B32);
    let rc = cx.column(ColumnType::Bit);
    let rcm = cx.column(ColumnType::Bit);
    let first = cx.column(ColumnType::Bit);
    let block_end = cx.column(ColumnType::Bit);
    let advance = cx.column(ColumnType::Bit);
    let into_final = cx.column(ColumnType::Bit);
    let emit = cx.column(ColumnType::Bit);
    let onehot = cx.columns(ROWS_PER_MODMUL, ColumnType::Bit);

    let mut pair = |count: usize| {
        [
            cx.columns(count, ColumnType::B128),
            cx.columns(count, ColumnType::B128),
        ]
    };

    let xa = pair(half(X_BITS));
    let yf = pair(Y_BITS - 1);
    let p = pair(LIMBS32 * half(Y_BITS));
    let u = pair(LIMBS32);
    let ut = pair(LIMBS32);
    let c = pair(1).map(|r| r.at(0));
    let ct = pair(1).map(|r| r.at(0));
    let cm = pair(1).map(|r| r.at(0));
    let co = pair(1).map(|r| r.at(0));
    let ctm = pair(1).map(|r| r.at(0));
    let cto = pair(1).map(|r| r.at(0));

    let dg = [0; 4].map(|_| cx.columns(half(DIGIT_BITS), ColumnType::B128));
    let ce = [0; 8].map(|_| cx.columns(half(CARRY_BITS) - 1, ColumnType::B128));
    let fr = [0; 8].map(|_| cx.columns(FROB_STEPS, ColumnType::B128));
    let lch = [0; 6].map(|_| cx.columns(half(Y_BITS), ColumnType::B128));

    ModexpCols {
        xs_bits: limbs.bits(0),
        q_bits: limbs.bits(1),
        xs_packed: limbs_packed.at(0),
        arrays: [s, x, y, n, r, w],
        y_bits,
        n_bits,
        sel_bits,
        sel_packed,
        digit_bits,
        carry_lo_bits,
        carry_hi_bits: carry_hi_word.bits(0),
        request_idx,
        rc,
        rcm,
        first,
        block_end,
        advance,
        into_final,
        emit,
        onehot,
        xa,
        yf,
        p,
        u,
        ut,
        c,
        ct,
        cm,
        co,
        ctm,
        cto,
        dg,
        ce,
        fr,
        lch,
    }
}

fn cadence(values: Vec<F>, count: usize, origin: usize) -> FixedShape<F> {
    FixedShape::Cadence {
        stride: ROWS_PER_MODMUL,
        count,
        origin,
        values,
    }
}

fn build_program(basis: &ExpBasis<F>) -> errors::Result<(CircuitProgram<F>, ModexpCols)> {
    let mut cx = Circuit::<F>::new("ModexpChiplet", NUM_ROWS)?;

    let k = declare(&mut cx);

    let mut end_of_block = vec![F::ZERO; ROWS_PER_MODMUL];
    end_of_block[ROWS_PER_MODMUL - 1] = F::ONE;

    cx.fix(k.first, FixedShape::FirstRow);
    cx.fix(k.block_end, cadence(end_of_block.clone(), NUM_MODMULS, 0));
    cx.fix(k.advance, cadence(end_of_block.clone(), NUM_MODMULS - 1, 0));
    cx.fix(
        k.into_final,
        cadence(end_of_block, 1, (FINAL_BLOCK - 1) * ROWS_PER_MODMUL),
    );
    cx.fix(k.emit, FixedShape::Sparse(vec![(RESULT_ROW, F::ONE)]));

    for i in 0..ROWS_PER_MODMUL {
        let mut values = vec![F::ZERO; ROWS_PER_MODMUL];
        values[i] = F::ONE;

        cx.fix(k.onehot.at(i), cadence(values, NUM_MODMULS, 0));
    }

    let values: Vec<usize> = k.arrays[ARR_N]
        .iter()
        .chain(k.arrays[ARR_S].iter())
        .chain(k.arrays[ARR_R].iter())
        .map(Col::index)
        .collect();

    cx.bus(
        BUS_ID,
        service().respond(&values, &[k.request_idx.index()], k.emit.index())?,
    );

    let cs = cx.cs();

    let pow2 = basis.tower_pow2();

    let one = cs.constant(F::ONE);
    let v32 = cs.constant(F::from(V32));
    let g_step = cs.constant(basis.generator() + F::ONE);
    let k32_step = cs.constant(pow2[DIGIT_BITS] + F::ONE);

    let col = |c: Col| cs.col(c.index());
    let next = |c: Col| cs.next(c.index());
    let range = |r: ColRange| -> Vec<Expr<'_, F>> { r.iter().map(col).collect() };
    let array = |which: usize, j: usize| k.arrays[which].at(j);

    let block_end = col(k.block_end);
    let not_block_end = one + block_end;
    let advance = col(k.advance);
    let first = col(k.first);
    let into_final = col(k.into_final);
    let onehot = range(k.onehot);

    cs.assert_boolean(col(k.rc));
    cs.assert_boolean(col(k.rcm));

    for cell in k.carry_hi_bits.iter().skip(CARRY_WORDS * CARRY_HI_BITS) {
        cs.constrain(col(cell));
    }

    for j in 0..LIMBS32 {
        cs.constrain(next(array(ARR_S, j)) + col(array(ARR_S, j)));
        cs.constrain(next(array(ARR_N, j)) + col(array(ARR_N, j)));

        for which in [ARR_X, ARR_Y, ARR_R, ARR_W] {
            cs.constrain(not_block_end * (next(array(which, j)) + col(array(which, j))));
        }

        let r_j = col(array(ARR_R, j));
        let s_j = col(array(ARR_S, j));

        cs.constrain(advance * (next(array(ARR_X, j)) + r_j));
        cs.constrain(advance * (next(array(ARR_Y, j)) + r_j + into_final * (r_j + s_j)));
        cs.constrain(first * (col(array(ARR_X, j)) + s_j));
        cs.constrain(first * (col(array(ARR_Y, j)) + s_j));
    }

    let select = |which: usize, parity: usize| -> Expr<'_, F> {
        let mut acc = onehot[0] * col(array(which, parity));
        for (i, &s_i) in onehot.iter().enumerate().skip(1) {
            acc = acc + s_i * col(array(which, 2 * i + parity));
        }

        acc
    };

    let mut xs_select = onehot[0] * (col(array(ARR_X, 0)) + v32 * col(array(ARR_X, 1)));
    for (i, &s_i) in onehot.iter().enumerate().skip(1) {
        xs_select =
            xs_select + s_i * (col(array(ARR_X, 2 * i)) + v32 * col(array(ARR_X, 2 * i + 1)));
    }

    cs.constrain(col(k.xs_packed) + xs_select);

    let selected = [
        (ARR_R, 0),
        (ARR_R, 1),
        (ARR_W, 0),
        (ARR_W, 1),
        (ARR_N, 0),
        (ARR_N, 1),
    ];

    for (w, (which, parity)) in selected.iter().enumerate() {
        cs.constrain(col(k.sel_packed.at(w)) + select(*which, *parity));
    }

    let limb_bits = [range(k.xs_bits), range(k.q_bits)];
    let other_bits = [k.y_bits, k.n_bits];

    let mut products = [Vec::with_capacity(LIMBS32), Vec::with_capacity(LIMBS32)];

    for side in 0..2 {
        let xa = range(k.xa[side]);
        let yf = range(k.yf[side]);

        constrain_exp_chain2(cs, &limb_bits[side], &xa, &pow2[..X_BITS]);

        let g_a = xa[half(X_BITS) - 1];
        constrain_squarings(cs, g_a, &yf);

        for j in 0..LIMBS32 {
            let bits = other_bits[side].bits(j);

            let mut prev = one;
            for t in 0..half(Y_BITS) {
                let cell = col(k.p[side].at(j * half(Y_BITS) + t));

                let mut step = prev;
                for s in [2 * t, 2 * t + 1] {
                    let frob = if s == 0 { g_a } else { yf[s - 1] };
                    step = step * (one + col(bits.at(s)) * (frob + one));
                }

                cs.constrain(cell + step);

                prev = cell;
            }

            products[side].push(k.p[side].at(j * half(Y_BITS) + half(Y_BITS) - 1));
        }

        let u = range(k.u[side]);
        let ut = range(k.ut[side]);

        for m in 0..LIMBS32 {
            let (prior_u, prior_ut) = if m + 2 < LIMBS32 {
                (
                    not_block_end * u[m + 2] + block_end,
                    not_block_end * ut[m + 2] + block_end * u[m + 2],
                )
            } else {
                (one, one)
            };

            cs.constrain(next(k.u[side].at(m)) + prior_u * next(products[side][m]));
            cs.constrain(next(k.ut[side].at(m)) + prior_ut);
        }

        let c = col(k.c[side]);
        let ct = col(k.ct[side]);
        let cm = col(k.cm[side]);
        let co = col(k.co[side]);
        let ctm = col(k.ctm[side]);
        let cto = col(k.cto[side]);

        cs.constrain(not_block_end * (next(k.c[side]) + co));
        cs.constrain(block_end * (next(k.ct[side]) + co));
        cs.constrain(block_end * (next(k.c[side]) + one));
        cs.constrain(not_block_end * (next(k.ct[side]) + cto));
        cs.constrain(block_end * (cto + one));

        let words = [2 * side, 2 * side + 1, 4 + 2 * side, 4 + 2 * side + 1];
        for (word, end) in words.into_iter().zip([cm, co, ctm, cto]) {
            let hi = CARRY_HI_BITS * word;
            let bits: Vec<_> = k
                .carry_lo_bits
                .bits(word)
                .iter()
                .chain(k.carry_hi_bits.iter().skip(hi).take(CARRY_HI_BITS))
                .map(col)
                .collect();

            let mut cols = range(k.ce[word]);
            cols.push(end);

            constrain_exp_chain2(cs, &bits, &cols, &pow2[..CARRY_BITS]);
            constrain_fourth_powers(cs, end, &range(k.fr[word]));
        }

        let frob32 = |word: usize| col(k.fr[word].at(FROB_STEPS - 1));
        let digit = |d: usize| col(k.dg[d].at(half(DIGIT_BITS) - 1));
        let r_term = |parity: usize| -> Expr<'_, F> {
            match side {
                0 => one,
                _ => col(k.lch[parity].at(half(Y_BITS) - 1)),
            }
        };

        cs.constrain(u[0] * c * r_term(0) + digit(0) * frob32(words[0]));
        cs.constrain(u[1] * cm * r_term(1) + digit(1) * frob32(words[1]));
        cs.constrain(ut[0] * ct + digit(2) * frob32(words[2]));
        cs.constrain(ut[1] * ctm + digit(3) * frob32(words[3]));
    }

    for d in 0..4 {
        constrain_exp_chain2(
            cs,
            &range(k.digit_bits.bits(d)),
            &range(k.dg[d]),
            &pow2[..DIGIT_BITS],
        );
    }

    for w in 0..6 {
        constrain_exp_chain2(
            cs,
            &range(k.sel_bits.bits(w)),
            &range(k.lch[w]),
            &pow2[..Y_BITS],
        );
    }

    let lch_end = |w: usize| col(k.lch[w].at(half(Y_BITS) - 1));

    let rc = col(k.rc);
    let rcm = col(k.rcm);
    let rc_out = not_block_end * next(k.rc);

    cs.constrain(
        lch_end(0) * lch_end(2) * (one + rc * g_step) + lch_end(4) * (one + rcm * k32_step),
    );
    cs.constrain(
        lch_end(1) * lch_end(3) * (one + rcm * g_step) + lch_end(5) * (one + rc_out * k32_step),
    );
    cs.constrain(block_end * (next(k.rc) + one));

    Ok((cx.compile()?, k))
}
