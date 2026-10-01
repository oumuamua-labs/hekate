// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec::Vec;
use hekate_gadgets::atoms::int_arith::{add_carry_chain, mod_add, mul_const, range_check};
use hekate_math::TowerField;
use hekate_program::constraint::builder::{ConstraintSystem, Expr};

use super::layout::{HighBitsLayout, W_BITS};
use crate::gadgets::{ModAddExprs, bits, borrow_chain, const_bits, groups, packed, padded};
use crate::mldsa::{MlDsaParams, Q};

pub(crate) fn constrain<F: TowerField>(
    cs: &ConstraintSystem<F>,
    params: &MlDsaParams,
    ly: &HighBitsLayout,
) {
    let one = cs.one();
    let zero = cs.constant(F::ZERO);

    let gamma2 = params.gamma2();
    let m = params.high_bits_range();
    let n = ly.n;

    let w = bits(cs, ly.w, W_BITS);
    let r1u = bits(cs, ly.r1u, n);
    let r0u = bits(cs, ly.r0u, ly.r0_width);

    cs.constrain(cs.col(ly.word.index()) + packed(cs, &w));

    let width = ly.mul.result_width;
    let qxd = bits(cs, ly.qxd, width);

    let mul_results = groups(cs, &ly.mul_results);
    let mul_carries = groups(cs, &ly.mul_carries);

    let result_refs: Vec<&[Expr<'_, F>]> = mul_results.iter().map(Vec::as_slice).collect();
    let carry_refs: Vec<&[Expr<'_, F>]> = mul_carries.iter().map(Vec::as_slice).collect();

    mul_const(cs, &r1u, &qxd, &result_refs, &carry_refs, 2 * gamma2);

    let add_carry = bits(cs, ly.add_carry, width + 1);

    add_carry_chain(
        cs,
        &qxd,
        &padded(&r0u, width, zero),
        &padded(&w, width, zero),
        &add_carry,
    );

    cs.constrain(add_carry[width]);

    range_check(
        cs,
        &r0u,
        &bits(cs, ly.rng_result, ly.r0_width),
        &bits(cs, ly.rng_borrow, ly.r0_width + 1),
        2 * gamma2,
    );

    let w_borrow = bits(cs, ly.w_rng_borrow, W_BITS + 1);

    borrow_chain(
        cs,
        &const_bits(cs, Q - 1, W_BITS),
        &w,
        &bits(cs, ly.w_rng_result, W_BITS),
        &w_borrow,
    );

    cs.constrain_named("hb_w_range", w_borrow[W_BITS]);

    let neg_result = bits(cs, ly.neg_result, ly.r0_width);
    let neg_borrow = bits(cs, ly.neg_borrow, ly.r0_width + 1);

    cs.constrain(neg_borrow[0]);

    for i in 0..ly.r0_width {
        let (v, w_in, r, w_out) = (r0u[i], neg_borrow[i], neg_result[i], neg_borrow[i + 1]);

        match (gamma2 >> i) & 1 {
            1 => {
                cs.constrain(r + one + v + w_in);
                cs.constrain(w_out + v * w_in);
            }
            _ => {
                cs.constrain(r + v + w_in);
                cs.constrain(w_out + v + w_in + v * w_in);
            }
        }
    }

    let neg = neg_borrow[ly.r0_width];

    let r1p = bits(cs, ly.r1p, n);
    let inc = bits(cs, ly.inc, n - 1);

    for k in 0..n {
        let carry = if k == 0 { neg } else { inc[k - 1] };

        cs.constrain(r1p[k] + r1u[k] + carry);

        match k + 1 < n {
            true => cs.constrain(inc[k] + r1u[k] * carry),
            false => cs.constrain(r1u[k] * carry),
        }
    }

    let nz = cs.col(ly.nz);
    let d0 = packed(cs, &r0u);

    cs.constrain_named("hb_nonzero_set", d0 * (one + nz));
    cs.constrain_named("hb_nonzero_unset", nz + d0 * cs.col(ly.inv0.index()));

    cs.constrain((one + nz) * cs.col(ly.inv0.index()));

    let dir = cs.col(ly.dir);
    let h = cs.col(ly.h);
    let down = cs.col(ly.down);

    cs.constrain_named("hb_dir", dir + (one + neg) * nz);

    cs.constrain(down + h * (one + dir));

    let up = h + down;
    let delta: Vec<Expr<'_, F>> = (0..n)
        .map(|k| {
            let from_up = if k == 0 { up } else { zero };
            let from_down = if ((m - 1) >> k) & 1 == 1 { down } else { zero };

            from_up + from_down
        })
        .collect();

    let w1 = bits(cs, ly.w1, n);
    let w1_add = ModAddExprs::new(cs, &ly.w1_add, &ly.add);

    mod_add(cs, &r1p, &delta, &w1, &w1_add.witness(), m);

    cs.constrain(cs.col(ly.hp.index()) + h * cs.col(ly.hpoly.index()));
    cs.constrain(cs.col(ly.hm.index()) + h * cs.col(ly.pos.index()));

    let acc = cs.col(ly.acc.index());
    let lane = cs.col(ly.lane.index());

    let mut current = zero;
    let mut carried = zero;

    for (t, &bit) in w1.iter().take(ly.b).enumerate() {
        current = current + bit * cs.col(ly.pw.at(t).index());
        carried = carried + bit * cs.col(ly.pn.at(t).index());
    }

    cs.constrain_named("hb_lane", lane + acc + current);

    cs.constrain(
        cs.col(ly.cont.index())
            * (cs.next(ly.acc.index()) + (one + cs.col(ly.emit.index())) * lane + carried),
    );
    cs.constrain(cs.col(ly.start.index()) * acc);

    let idle = one + cs.col(ly.active.index());

    cs.constrain(idle * cs.col(ly.word.index()));
    cs.constrain(idle * h);
    cs.constrain(idle * acc);

    for k in ly.num_bits..ly.num_packed * 32 {
        cs.constrain(cs.col(k));
    }
}
