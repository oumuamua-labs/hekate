// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec::Vec;
use hekate_gadgets::atoms::int_arith::{
    ModReductionWitness, SchoolbookMulWitness, mod_add, mod_reduction, range_check, schoolbook_mul,
};
use hekate_math::TowerField;
use hekate_program::constraint::builder::{ConstraintSystem, Expr};

use super::layout::NttLayout;
use crate::gadgets::{ModAddExprs, bits, groups, packed};

pub(crate) fn constrain<F: TowerField>(cs: &ConstraintSystem<F>, q: u32, ly: &NttLayout) {
    let bw = ly.bit_width;

    let a = bits(cs, ly.a, bw);
    let b = bits(cs, ly.b, bw);
    let w = bits(cs, ly.w, bw);
    let x = bits(cs, ly.x, bw);
    let y = bits(cs, ly.y, bw);
    let s1 = bits(cs, ly.s1, bw);
    let s2 = bits(cs, ly.s2, bw);
    let quot = bits(cs, ly.quot, bw);
    let p = bits(cs, ly.p, bw);
    let product = bits(cs, ly.product, ly.mul.product_width);

    let s1_add = ModAddExprs::new(cs, &ly.s1_add, &ly.add);
    let s2_sub = ModAddExprs::new(cs, &ly.s2_sub, &ly.add);

    mod_add(cs, &a, &y, &s1, &s1_add.witness(), q);
    mod_add(cs, &s2, &y, &a, &s2_sub.witness(), q);

    range_check(
        cs,
        &s2,
        &bits(cs, ly.s2_range_result, bw),
        &bits(cs, ly.s2_range_borrow, bw + 1),
        q,
    );

    let pp0 = bits(cs, ly.mul_pp0, ly.mul.pp0_width);
    let sums = groups(cs, &ly.mul_sums);
    let carries = groups(cs, &ly.mul_carries);

    let sum_refs: Vec<&[Expr<'_, F>]> = sums.iter().map(Vec::as_slice).collect();
    let carry_refs: Vec<&[Expr<'_, F>]> = carries.iter().map(Vec::as_slice).collect();

    schoolbook_mul(
        cs,
        &w,
        &x,
        &product,
        &SchoolbookMulWitness {
            pp0: &pp0,
            sums: &sum_refs,
            carries: &carry_refs,
        },
    );

    let quot_x_q = bits(cs, ly.quot_x_q, ly.red.product_width);
    let red_results = groups(cs, &ly.red_results);
    let red_carries = groups(cs, &ly.red_carries);

    let red_result_refs: Vec<&[Expr<'_, F>]> = red_results.iter().map(Vec::as_slice).collect();
    let red_carry_refs: Vec<&[Expr<'_, F>]> = red_carries.iter().map(Vec::as_slice).collect();

    mod_reduction(
        cs,
        &product,
        &quot,
        &p,
        &ModReductionWitness {
            quot_x_mod_bits: &quot_x_q,
            mul_scratch_results: &red_result_refs,
            mul_scratch_carries: &red_carry_refs,
            add_carry_bits: &bits(cs, ly.red_add_carry, ly.red.add_carry_width),
            range_result_bits: &bits(cs, ly.red_range_result, ly.red.range_result_width),
            range_borrow_bits: &bits(cs, ly.red_range_borrow, ly.red.range_borrow_width),
        },
        q,
    );

    let gs = cs.col(ly.gs.index());

    for k in 0..bw {
        cs.constrain(y[k] + p[k] + gs * (b[k] + p[k]));
        cs.constrain(x[k] + b[k] + gs * (s2[k] + b[k]));
    }

    let mac = cs.col(ly.mac.index());
    let neg = cs.col(ly.neg.index());
    let za = cs.col(ly.za.index());
    let tw = cs.col(ly.tw.index());
    let v_in1 = cs.col(ly.v_in1.index());
    let v_in2 = cs.col(ly.v_in2.index());
    let v_out1 = cs.col(ly.v_out1.index());
    let v_out2 = cs.col(ly.v_out2.index());

    let packed_s1 = packed(cs, &s1);
    let packed_s2 = packed(cs, &s2);

    let mut acc_in = cs.constant(F::ZERO);
    for r in 0..ly.registers {
        acc_in = acc_in + cs.col(ly.selin.at(r).index()) * cs.col(ly.reg.at(r).index());
    }

    cs.constrain(v_in2 + packed(cs, &b));
    cs.constrain(packed(cs, &w) + tw + mac * v_in1);

    cs.constrain_named("ntt_acc", packed(cs, &a) + v_in1 + mac * v_in1 + acc_in);
    cs.constrain_named("ntt_scale_zero", za * v_in1);

    cs.constrain(v_out1 + packed_s1 + neg * (packed_s1 + packed_s2));
    cs.constrain(v_out2 + packed_s2 + gs * (packed_s2 + packed(cs, &p)));

    cs.constrain_named(
        "ntt_copy",
        cs.col(ly.bcont.index()) * (cs.next(ly.v_in2.index()) + v_in2),
    );

    let rcont = cs.col(ly.rcont.index());

    for r in 0..ly.registers {
        let reg = ly.reg.at(r).index();
        let sel = cs.col(ly.sel.at(r).index());

        cs.constrain_named(
            "ntt_register",
            rcont * (cs.next(reg) + cs.col(reg) + sel * (v_out1 + cs.col(reg))),
        );
    }

    for k in ly.num_bits..ly.num_packed * 32 {
        cs.constrain(cs.col(k));
    }
}
