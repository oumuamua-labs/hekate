// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_math::TowerField;
use hekate_program::circuit::Col;
use hekate_program::constraint::builder::ConstraintSystem;

use super::layout::KemSelectLayout;

pub(super) fn constrain<F: TowerField>(cs: &ConstraintSystem<F>, ly: &KemSelectLayout) {
    let one = cs.one();
    let col = |c: Col| cs.col(c.index());

    let (a, b, c, inv) = (col(ly.word_a), col(ly.word_b), col(ly.word_c), col(ly.inv));
    let (compare, select, report) = (col(ly.compare), col(ly.select), col(ly.report));
    let (nz, flag) = (col(ly.nz), col(ly.flag));

    let d = a + b;

    cs.constrain_named("kem_select_nonzero", compare * (d * inv + nz));
    cs.constrain_named("kem_select_zero", compare * d * (one + nz));
    cs.constrain_named(
        "kem_select_flag",
        col(ly.chain) * (cs.next(ly.flag.index()) + flag + nz + flag * nz),
    );
    cs.constrain_named("kem_select_first", col(ly.first) * flag);
    cs.constrain_named(
        "kem_select_out",
        c + compare * a + select * (a + flag * d) + report * (one + flag),
    );

    cs.constrain((one + col(ly.sel_ab)) * a);
    cs.constrain((one + col(ly.sel_ab)) * b);
    cs.constrain((one + nz) * inv);

    cs.constrain_named("kem_select_nz_gate", (one + compare) * nz);

    cs.constrain((one + compare + select + report) * flag);
}
