// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec;
use hekate_core::errors;
use hekate_core::trace::{ColumnTrace, TraceBuilder};
use hekate_math::TowerField;
use hekate_program::Air;
use hekate_program::circuit::CircuitProgram;
use subtle::{Choice, ConditionallySelectable};
use zeroize::Zeroizing;

use super::layout::{HighBitsLayout, W_BITS};
use super::{HighBitsRow, bit_position, emits, lane_index, slots};
use crate::mldsa::{MlDsaParams, Q};
use crate::utils::{
    Writer, fill_add_carry_packed, fill_mod_add, fill_mul_const, fill_sub_borrow_packed,
    flush_bit_buffer, pack_bits, pack_one,
};
use crate::wiring::{N, Poly, Stream, hint_poly};

pub(super) fn generate<F: TowerField>(
    program: &CircuitProgram<F>,
    params: &MlDsaParams,
    (inputs, lanes): (&[Poly], Stream),
    ly: &HighBitsLayout,
    num_rows: usize,
    rows: &[HighBitsRow],
) -> errors::Result<ColumnTrace> {
    let num_vars = num_rows.trailing_zeros() as usize;
    let mut tb = TraceBuilder::new_secret(program.column_layout(), num_vars)?;

    let padding = HighBitsRow::new(params, 0, false)?;
    let b = ly.b;

    let mut bits = Zeroizing::new(vec![0u32; ly.num_packed]);
    let mut acc = 0u64;

    for r in 0..num_rows {
        let row = rows.get(r).unwrap_or(&padding);

        bits.fill(0);

        fill_bits(&mut bits, params, ly, row);
        flush_bit_buffer(&bits, &mut tb, r)?;

        let mut w = Writer {
            tb: &mut tb,
            physical: |col| ly.physical(col),
            row: r,
        };

        w.word(ly.word, row.w)?;
        w.word(ly.inv0, row.inv0)?;

        if r >= rows.len() {
            continue;
        }

        let (i, pos) = (r / N, r % N);

        let slot = pos % slots(b);
        let emit = emits(slot, b);

        let mut current = 0u64;
        let mut carried = 0u64;

        for t in 0..b {
            let bit = ((row.w1 >> t) & 1) as u64;

            if let Some(p) = bit_position(slot, t, b, false) {
                w.lane(ly.pw.at(t), 1 << p)?;

                current |= bit << p;
            }

            if let Some(p) = bit_position(slot, t, b, true) {
                w.lane(ly.pn.at(t), 1 << p)?;

                carried |= bit << p;
            }
        }

        let lane = acc | current;

        w.lane(ly.acc, acc)?;
        w.lane(ly.lane, lane)?;

        let hpoly = hint_poly((i / params.k()) as u8, i % params.k());

        w.label(ly.poly, inputs[i].id())?;
        w.label(ly.pos, pos as u16)?;
        w.label(ly.hpoly, hpoly)?;

        let hit = Choice::from(row.h as u8);

        w.label(ly.hp, u16::conditional_select(&0, &hpoly, hit))?;
        w.label(ly.hm, u16::conditional_select(&0, &(pos as u16), hit))?;

        w.label(ly.lidx, if emit { lane_index(r, b) as u16 } else { 0 })?;
        w.label(ly.lstream, lanes.id())?;

        w.flag(ly.active, true)?;
        w.flag(ly.start, r == 0)?;
        w.flag(ly.cont, r + 1 < rows.len())?;
        w.flag(ly.emit, emit)?;

        acc = if emit { carried } else { lane };
    }

    Ok(tb.build())
}

fn fill_bits(bits: &mut [u32], params: &MlDsaParams, ly: &HighBitsLayout, row: &HighBitsRow) {
    let two_g2 = 2 * params.gamma2();
    let (n, r0_width) = (ly.n, ly.r0_width);
    let qxd = row.r1u as u64 * two_g2 as u64;

    pack_bits(bits, ly.w, row.w as u64, W_BITS);
    pack_bits(bits, ly.r1u, row.r1u as u64, n);
    pack_bits(bits, ly.r0u, row.r0u as u64, r0_width);
    pack_bits(bits, ly.qxd, qxd, ly.mul.result_width);

    fill_mul_const(bits, two_g2, row.r1u, &ly.mul_results, &ly.mul_carries);
    fill_add_carry_packed(
        bits,
        ly.add_carry,
        ly.mul.result_width + 1,
        qxd,
        row.r0u as u64,
    );
    fill_sub_borrow_packed(
        bits,
        ly.rng_result,
        ly.rng_borrow,
        r0_width,
        (two_g2 - 1) as u64,
        row.r0u as u64,
    );
    fill_sub_borrow_packed(
        bits,
        ly.neg_result,
        ly.neg_borrow,
        r0_width,
        params.gamma2() as u64,
        row.r0u as u64,
    );
    fill_sub_borrow_packed(
        bits,
        ly.w_rng_result,
        ly.w_rng_borrow,
        W_BITS,
        (Q - 1) as u64,
        row.w as u64,
    );

    pack_bits(bits, ly.r1p, row.r1p as u64, n);

    let mut carry = row.neg;
    for k in 0..n - 1 {
        carry &= (row.r1u >> k) & 1 == 1;

        pack_one(bits, ly.inc + k, carry);
    }

    pack_one(bits, ly.nz, row.nz);
    pack_one(bits, ly.dir, row.dir);
    pack_one(bits, ly.h, row.h);
    pack_one(bits, ly.down, row.down);

    fill_mod_add(
        bits,
        params.high_bits_range(),
        n,
        (row.r1p, row.delta, row.w1),
        row.wrap,
        &ly.w1_add,
    );

    pack_bits(bits, ly.w1, row.w1 as u64, n);
}
