// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_gadgets::atoms::int_arith::mod_add;
use hekate_math::TowerField;
use hekate_program::circuit::Col;
use hekate_program::constraint::builder::ConstraintSystem;

use super::layout::PolyArithLayout;
use crate::gadgets::{ModAddExprs, bits, packed};

pub(crate) fn constrain<F: TowerField>(cs: &ConstraintSystem<F>, q: u32, ly: &PolyArithLayout) {
    let bw = ly.bit_width;
    let col = |c: Col| cs.col(c.index());

    let a0 = bits(cs, ly.a0, bw);
    let a1 = bits(cs, ly.a1, bw);
    let b0 = bits(cs, ly.b0, bw);
    let b1 = bits(cs, ly.b1, bw);
    let in0 = bits(cs, ly.in0, bw);
    let in1 = bits(cs, ly.in1, bw);
    let gamma = bits(cs, ly.gamma, bw);

    let t00 = ly.t00.constrain(cs, &a0, &b0, q);
    let t11 = ly.t11.constrain(cs, &a1, &b1, q);
    let g = ly.g.constrain(cs, &t11, &gamma, q);
    let t01 = ly.t01.constrain(cs, &a0, &b1, q);
    let t10 = ly.t10.constrain(cs, &a1, &b0, q);

    let u0 = bits(cs, ly.u0, bw);
    let out0 = bits(cs, ly.out0, bw);
    let u1 = bits(cs, ly.u1, bw);
    let out1 = bits(cs, ly.out1, bw);

    let u0_add = ModAddExprs::new(cs, &ly.u0_add, &ly.add);
    let out0_add = ModAddExprs::new(cs, &ly.out0_add, &ly.add);
    let u1_add = ModAddExprs::new(cs, &ly.u1_add, &ly.add);
    let out1_add = ModAddExprs::new(cs, &ly.out1_add, &ly.add);

    mod_add(cs, &in0, &t00, &u0, &u0_add.witness(), q);
    mod_add(cs, &u0, &g, &out0, &out0_add.witness(), q);
    mod_add(cs, &in1, &t01, &u1, &u1_add.witness(), q);
    mod_add(cs, &u1, &t10, &out1, &out1_add.witness(), q);

    for (value, word) in [
        (&a0, ly.v_a0),
        (&a1, ly.v_a1),
        (&b0, ly.v_b0),
        (&b1, ly.v_b1),
        (&in0, ly.v_in0),
        (&in1, ly.v_in1),
        (&gamma, ly.v_gamma),
        (&out0, ly.v_out0),
        (&out1, ly.v_out1),
    ] {
        cs.constrain_named("poly_arith_pack", packed(cs, value) + col(word));
    }

    let seed = col(ly.seed);

    let mut acc0 = cs.constant(F::ZERO);
    let mut acc1 = cs.constant(F::ZERO);

    for r in 0..ly.registers {
        let selin = col(ly.selin.at(r));

        acc0 = acc0 + selin * col(ly.h0.at(r));
        acc1 = acc1 + selin * col(ly.h1.at(r));
    }

    let (v_in0, v_in1) = (col(ly.v_in0), col(ly.v_in1));

    cs.constrain_named("poly_arith_acc", v_in0 + seed * v_in0 + acc0);
    cs.constrain_named("poly_arith_acc", v_in1 + seed * v_in1 + acc1);

    let bcont = col(ly.bcont);

    for word in [ly.v_b0, ly.v_b1] {
        cs.constrain_named(
            "poly_arith_copy",
            bcont * (cs.next(word.index()) + col(word)),
        );
    }

    let rcont = col(ly.rcont);
    let (v_out0, v_out1) = (col(ly.v_out0), col(ly.v_out1));

    for r in 0..ly.registers {
        let sel = col(ly.sel.at(r));

        for (reg, out) in [(ly.h0.at(r), v_out0), (ly.h1.at(r), v_out1)] {
            let h = col(reg);

            cs.constrain_named(
                "poly_arith_reg",
                rcont * (cs.next(reg.index()) + h + sel * (out + h)),
            );
        }
    }

    let idle = cs.one() + col(ly.active);

    for word in [ly.v_a0, ly.v_a1, ly.v_b0, ly.v_b1] {
        cs.constrain(idle * col(word));
    }

    for r in 0..ly.registers {
        cs.constrain(idle * col(ly.h0.at(r)));
        cs.constrain(idle * col(ly.h1.at(r)));
    }

    for k in ly.num_bits..ly.num_packed * 32 {
        cs.constrain(cs.col(k));
    }
}
