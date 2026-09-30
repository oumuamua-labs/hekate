// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec::Vec;
use hekate_math::TowerField;
use hekate_program::constraint::builder::{ConstraintSystem, Expr};

use super::layout::{
    BYTE, CodecLayout, Compress1Block, CompressBlock, DecompressBlock, HINT_SLOTS, HintBits,
    HintCols, HintSlot, KEM_WIDTH, Q_SHIFTS, Z_WIDTH, half_q,
};
use super::{HintShape, Kinds, T1_BITS, T1_FIELDS, T1_SHIFT, ZShape, packing};
use crate::gadgets::{bits, borrow_chain, carry_chain, const_bits, packed, padded};
use crate::mldsa::Q;
use crate::mlkem;

struct HintExprs<'a, F: TowerField> {
    one: Expr<'a, F>,
    zero: Expr<'a, F>,
    counts: Vec<Vec<Expr<'a, F>>>,
    bytes: Vec<Vec<Expr<'a, F>>>,
    owners: Vec<Vec<Expr<'a, F>>>,
    following: Vec<Expr<'a, F>>,
    following_owner: Vec<Expr<'a, F>>,
    base: Expr<'a, F>,
}

pub(crate) fn constrain<F: TowerField>(cs: &ConstraintSystem<F>, kinds: &Kinds, ly: &CodecLayout) {
    let one = cs.one();

    for i in 0..ly.words {
        let sel = cs.col(ly.wsel.at(i).index());
        cs.constrain((one + sel) * cs.col(ly.word.at(i).index()));
    }

    for j in 0..ly.coefs {
        let sel = cs.col(ly.csel.at(j).index());
        cs.constrain((one + sel) * cs.col(ly.coef.at(j).index()));
    }

    if let Some(on) = ly.t1 {
        t1(cs, ly, cs.col(on.index()));
    }

    if let (Some(shape), Some(on)) = (kinds.z, ly.zk) {
        z(cs, ly, shape, cs.col(on.index()));
    }

    if let (Some(shape), Some(hb), Some(hc)) = (kinds.hint, &ly.hint, &ly.hint_cols) {
        hint(cs, ly, shape, hb, hc);
    }

    if let (Some(on), Some(reduce)) = (ly.k12, ly.reduce) {
        byte12(cs, ly, cs.col(on.index()), cs.col(reduce.index()));
    }

    for block in &ly.decompress {
        decompress(cs, ly, block);
    }

    if let Some(on) = ly.decompress1 {
        decompress1(cs, ly, cs.col(on.index()));
    }

    for block in &ly.compress {
        compress(cs, ly, block);
    }

    if let Some(block) = &ly.compress1 {
        compress1(cs, ly, block);
    }

    for k in ly.num_bits..ly.num_packed * 32 {
        cs.constrain(cs.col(k));
    }
}

fn t1<'a, F: TowerField>(cs: &'a ConstraintSystem<F>, ly: &CodecLayout, on: Expr<'a, F>) {
    for j in 0..T1_FIELDS {
        let mut value = cs.col(ly.coef.at(j).index());
        for t in 0..T1_BITS {
            let weight = cs.constant(F::from(1u128 << (T1_SHIFT + t)));
            value = value + cs.col(ly.word_bit(T1_BITS * j + t)) * weight;
        }

        cs.constrain_named("codec_t1", on * value);
    }
}

fn z<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    ly: &CodecLayout,
    shape: ZShape,
    on: Expr<'a, F>,
) {
    let one = cs.one();
    let zero = cs.constant(F::ZERO);
    let w = shape.width;

    let idle = cs.constant(F::from(shape.beta + 1));
    let low = const_bits(cs, shape.beta + 1, w);
    let high = const_bits(cs, 2 * shape.gamma1 - shape.beta - 1, w);
    let top = const_bits(cs, Q - 1, Z_WIDTH);

    for (j, f) in ly.z.iter().enumerate() {
        let y = bits(cs, f.y, w);
        let field = word_field(cs, ly, j, w);

        cs.constrain_named(
            "codec_z_word",
            packed(cs, &y) + on * packed(cs, &field) + (one + on) * idle,
        );

        let lo_borrow = bits(cs, f.lo_borrow, w + 1);

        borrow_chain(cs, &y, &low, &bits(cs, f.lo_result, w), &lo_borrow);

        cs.constrain_named("codec_z_low", lo_borrow[w]);

        let hi_borrow = bits(cs, f.hi_borrow, w + 1);

        borrow_chain(cs, &high, &y, &bits(cs, f.hi_result, w), &hi_borrow);

        cs.constrain_named("codec_z_high", hi_borrow[w]);

        let z = bits(cs, f.z, Z_WIDTH);
        let flag = cs.col(f.flag);

        let sum: Vec<Expr<'a, F>> = (0..=Z_WIDTH)
            .map(
                |k| match ((shape.gamma1 >> k) & 1, ((shape.gamma1 + Q) >> k) & 1) {
                    (0, 0) => zero,
                    (1, 1) => one,
                    (0, _) => flag,
                    _ => one + flag,
                },
            )
            .collect();

        let carry = bits(cs, f.carry, Z_WIDTH + 2);

        carry_chain(
            cs,
            &padded(&z, Z_WIDTH + 1, zero),
            &padded(&y, Z_WIDTH + 1, zero),
            &sum,
            &carry,
        );

        cs.constrain(carry[Z_WIDTH + 1]);

        let canonical = bits(cs, f.rng_borrow, Z_WIDTH + 1);

        borrow_chain(cs, &top, &z, &bits(cs, f.rng_result, Z_WIDTH), &canonical);

        cs.constrain_named("codec_z_range", canonical[Z_WIDTH]);
        cs.constrain_named(
            "codec_z_coef",
            on * (cs.col(ly.coef.at(j).index()) + packed(cs, &z)),
        );
    }
}

fn hint<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    ly: &CodecLayout,
    shape: HintShape,
    hb: &HintBits,
    hc: &HintCols,
) {
    let one = cs.one();
    let zero = cs.constant(F::ZERO);
    let k = shape.k;

    let on = cs.col(hc.kind.index());
    let cont = cs.col(hc.cont.index());
    let base = cs.col(hc.base.index());

    let counts: Vec<Vec<Expr<'a, F>>> = (0..k)
        .map(|i| bits(cs, hb.counts + BYTE * i, BYTE))
        .collect();

    for (i, count) in counts.iter().enumerate() {
        let next: Vec<Expr<'a, F>> = (0..BYTE)
            .map(|b| cs.next(hb.counts + BYTE * i + b))
            .collect();

        cs.constrain_named(
            "codec_hint_cont",
            cont * (packed(cs, &next) + packed(cs, count)),
        );

        cs.constrain((one + on) * packed(cs, count));
    }

    for i in 1..k {
        let borrow = bits(cs, hb.mono_borrow[i - 1], BYTE + 1);
        let result = bits(cs, hb.mono_result[i - 1], BYTE);

        borrow_chain(cs, &counts[i], &counts[i - 1], &result, &borrow);

        cs.constrain_named("codec_hint_counts", borrow[BYTE]);
    }

    let omega = const_bits(cs, shape.omega as u32, BYTE);
    let bound = bits(cs, hb.last_borrow, BYTE + 1);

    borrow_chain(
        cs,
        &omega,
        &counts[k - 1],
        &bits(cs, hb.last_result, BYTE),
        &bound,
    );

    cs.constrain_named("codec_hint_bound", bound[BYTE]);

    let x = HintExprs {
        bytes: (0..HINT_SLOTS)
            .map(|t| bits(cs, ly.word_bit(BYTE * t), BYTE))
            .collect(),
        owners: hb.slots.iter().map(|slot| bits(cs, slot.p, k)).collect(),
        following: (0..BYTE).map(|b| cs.next(ly.word_bit(b))).collect(),
        following_owner: (0..k).map(|i| cs.next(hb.slots[0].p + i)).collect(),
        one,
        zero,
        counts,
        base,
    };

    for i in 0..k {
        let role = cs.col(hc.count_role.at(i).index());
        let byte = &x.bytes[(shape.omega + i) % HINT_SLOTS];

        cs.constrain_named(
            "codec_hint_count",
            role * (packed(cs, byte) + packed(cs, &x.counts[i])),
        );
    }

    for (t, slot) in hb.slots.iter().enumerate() {
        hint_slot(cs, hc, &x, t, slot);
    }
}

fn hint_slot<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    hc: &HintCols,
    x: &HintExprs<'a, F>,
    t: usize,
    slot: &HintSlot,
) {
    let (one, zero) = (x.one, x.zero);

    let k = x.counts.len();

    let idx = cs.col(hc.idx.at(t).index());
    let byte = packed(cs, &x.bytes[t]);
    let s = bits(cs, slot.s, BYTE);

    cs.constrain(packed(cs, &s) + cs.col(hc.slot_value.at(t).index()));

    let lt: Vec<Expr<'a, F>> = x
        .counts
        .iter()
        .zip(&slot.lt_result)
        .zip(&slot.lt_borrow)
        .map(|((count, &result), &borrow)| {
            let borrow = bits(cs, borrow, BYTE + 1);
            borrow_chain(cs, &s, count, &bits(cs, result, BYTE), &borrow);

            borrow[BYTE]
        })
        .collect();

    let p = &x.owners[t];

    let mut owner = zero;
    let mut owned = zero;

    for (i, &pi) in p.iter().enumerate() {
        let earlier = if i == 0 { zero } else { lt[i - 1] };
        cs.constrain(pi + lt[i] * (one + earlier));

        owner = owner + pi * cs.constant(F::from(i as u32 + 1));
        owned = owned + pi;
    }

    cs.constrain(cs.col(hc.hp.at(t).index()) + idx * (owner + x.base * owned));
    cs.constrain(cs.col(hc.hm.at(t).index()) + idx * byte);

    cs.constrain_named("codec_hint_unused", idx * (one + lt[k - 1]) * byte);
    cs.constrain_named("codec_hint_pad", cs.col(hc.pad.at(t).index()) * byte);

    let (after, after_owner) = match t + 1 < HINT_SLOTS {
        true => (&x.bytes[t + 1], &x.owners[t + 1]),
        false => (&x.following, &x.following_owner),
    };

    let mut shared = zero;
    for (&pi, &qi) in p.iter().zip(after_owner) {
        shared = shared + pi * qi;
    }

    let same = cs.col(slot.same);
    let order = bits(cs, slot.inc_borrow, BYTE + 1);

    cs.constrain(same + cs.col(hc.pair.at(t).index()) * shared);

    borrow_chain(
        cs,
        &x.bytes[t],
        after,
        &bits(cs, slot.inc_result, BYTE),
        &order,
    );

    cs.constrain_named("codec_hint_order", same * (one + order[BYTE]));
}

fn byte12<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    ly: &CodecLayout,
    on: Expr<'a, F>,
    reduce: Expr<'a, F>,
) {
    let one = cs.one();
    let zero = cs.constant(F::ZERO);
    let top = const_bits(cs, mlkem::Q - 1, KEM_WIDTH);

    for (j, f) in ly.byte12.iter().enumerate() {
        let b = bits(cs, f.b, KEM_WIDTH);
        let x = bits(cs, f.x, KEM_WIDTH);

        let field = word_field(cs, ly, j, KEM_WIDTH);

        cs.constrain(packed(cs, &b) + on * packed(cs, &field));

        let flag = cs.col(f.flag);
        let q: Vec<Expr<'a, F>> = (0..KEM_WIDTH)
            .map(|k| match (mlkem::Q >> k) & 1 {
                1 => flag,
                _ => zero,
            })
            .collect();

        let carry = bits(cs, f.carry, KEM_WIDTH + 1);

        carry_chain(cs, &x, &q, &b, &carry);

        cs.constrain(carry[KEM_WIDTH]);

        let canonical = bits(cs, f.rng_borrow, KEM_WIDTH + 1);

        borrow_chain(cs, &top, &x, &bits(cs, f.rng_result, KEM_WIDTH), &canonical);

        cs.constrain_named("codec_kem_range", canonical[KEM_WIDTH]);
        cs.constrain_named("codec_kem_modulus", (one + reduce) * flag);
        cs.constrain_named(
            "codec_kem_byte12",
            on * (cs.col(ly.coef.at(j).index()) + packed(cs, &x)),
        );
    }
}

fn decompress<F: TowerField>(cs: &ConstraintSystem<F>, ly: &CodecLayout, block: &DecompressBlock) {
    let one = cs.one();
    let zero = cs.constant(F::ZERO);
    let on = cs.col(block.on.index());

    let d = block.d;
    let w = d + KEM_WIDTH;

    for (j, f) in block.fields.iter().enumerate() {
        let y = bits(cs, f.y, d);
        let field = word_field(cs, ly, j, d);

        cs.constrain(packed(cs, &y) + on * packed(cs, &field));

        let rounded: Vec<Expr<'_, F>> = (0..w)
            .map(|k| match k {
                k if k + 1 < d => y[k],
                k if k + 1 == d => one + y[k],
                k if k == d => y[d - 1],
                _ => zero,
            })
            .collect();

        let x = bits(cs, f.x, KEM_WIDTH);
        let out: Vec<Expr<'_, F>> = bits(cs, f.r, d)
            .into_iter()
            .chain(x.iter().copied())
            .collect();

        let mut acc = rounded;
        for (i, &s) in Q_SHIFTS.iter().enumerate() {
            let carry = bits(cs, f.carries[i], w + 1);
            let sum = match f.sums.get(i) {
                Some(&start) => bits(cs, start, w),
                None => out.clone(),
            };

            carry_chain(cs, &acc, &shifted(&y, s, w, zero), &sum, &carry);

            cs.constrain(carry[w]);

            acc = sum;
        }

        cs.constrain_named(
            "codec_kem_decompress",
            on * (cs.col(ly.coef.at(j).index()) + packed(cs, &x)),
        );
    }
}

fn decompress1<'a, F: TowerField>(cs: &'a ConstraintSystem<F>, ly: &CodecLayout, on: Expr<'a, F>) {
    let scale = cs.constant(F::from(mlkem::decompress(1, 1)));

    for j in 0..packing(1).1 {
        cs.constrain_named(
            "codec_kem_decompress",
            on * (cs.col(ly.coef.at(j).index()) + cs.col(ly.word_bit(j)) * scale),
        );
    }
}

fn compress<F: TowerField>(cs: &ConstraintSystem<F>, ly: &CodecLayout, block: &CompressBlock) {
    let zero = cs.constant(F::ZERO);
    let on = cs.col(block.on.index());

    let d = block.d;
    let v = d + KEM_WIDTH + 1;

    let low = half_q() & ((1 << d) - 1);
    let high = (half_q() >> d) as u32;
    let top = const_bits(cs, mlkem::Q - 1, KEM_WIDTH);

    for (j, f) in block.fields.iter().enumerate() {
        let x = bits(cs, f.x, KEM_WIDTH);

        cs.constrain(packed(cs, &x) + on * cs.col(ly.coef.at(j).index()));

        let yp = bits(cs, f.yp, d + 1);
        let r = bits(cs, f.r, KEM_WIDTH);

        let lifted = match f.lift {
            Some((sum, carry)) => {
                let sum = bits(cs, sum, KEM_WIDTH + 1);
                let carry = bits(cs, carry, KEM_WIDTH + 2);

                carry_chain(
                    cs,
                    &padded(&x, KEM_WIDTH + 1, zero),
                    &const_bits(cs, high, KEM_WIDTH + 1),
                    &sum,
                    &carry,
                );

                cs.constrain(carry[KEM_WIDTH + 1]);

                sum
            }
            None => padded(&x, KEM_WIDTH + 1, zero),
        };

        let lhs: Vec<Expr<'_, F>> = const_bits(cs, low as u32, d)
            .into_iter()
            .chain(lifted)
            .collect();

        let terms = core::iter::once(padded(&r, v, zero))
            .chain(Q_SHIFTS.iter().map(|&s| shifted(&yp, s, v, zero)));

        let mut acc = padded(&yp, v, zero);
        for (i, term) in terms.enumerate() {
            let carry = bits(cs, f.carries[i], v + 1);
            let sum = match f.sums.get(i) {
                Some(&start) => bits(cs, start, v),
                None => lhs.clone(),
            };

            carry_chain(cs, &acc, &term, &sum, &carry);

            cs.constrain(carry[v]);

            acc = sum;
        }

        let canonical = bits(cs, f.rng_borrow, KEM_WIDTH + 1);

        borrow_chain(cs, &top, &r, &bits(cs, f.rng_result, KEM_WIDTH), &canonical);

        cs.constrain_named("codec_compress_range", canonical[KEM_WIDTH]);

        let field = word_field(cs, ly, j, d);

        cs.constrain_named(
            "codec_kem_compress",
            on * (packed(cs, &field) + packed(cs, &yp[..d])),
        );
    }
}

fn compress1<F: TowerField>(cs: &ConstraintSystem<F>, ly: &CodecLayout, block: &Compress1Block) {
    let one = cs.one();
    let on = cs.col(block.on.index());

    let lower = const_bits(cs, mlkem::Q.div_ceil(4), KEM_WIDTH);
    let upper = const_bits(cs, 3 * mlkem::Q / 4, KEM_WIDTH);

    for (j, f) in block.fields.iter().enumerate() {
        let x = bits(cs, f.x, KEM_WIDTH);

        cs.constrain(packed(cs, &x) + on * cs.col(ly.coef.at(j).index()));

        let below = bits(cs, f.lo_borrow, KEM_WIDTH + 1);

        borrow_chain(cs, &x, &lower, &bits(cs, f.lo_result, KEM_WIDTH), &below);

        let above = bits(cs, f.hi_borrow, KEM_WIDTH + 1);

        borrow_chain(cs, &upper, &x, &bits(cs, f.hi_result, KEM_WIDTH), &above);

        cs.constrain_named(
            "codec_kem_compress",
            on * (cs.col(ly.word_bit(j)) + one + below[KEM_WIDTH] + above[KEM_WIDTH]),
        );
    }
}

fn word_field<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    ly: &CodecLayout,
    j: usize,
    width: usize,
) -> Vec<Expr<'a, F>> {
    (0..width)
        .map(|t| cs.col(ly.word_bit(width * j + t)))
        .collect()
}

fn shifted<'a, F: TowerField>(
    v: &[Expr<'a, F>],
    shift: usize,
    width: usize,
    zero: Expr<'a, F>,
) -> Vec<Expr<'a, F>> {
    (0..width)
        .map(|k| match k.checked_sub(shift) {
            Some(i) if i < v.len() => v[i],
            _ => zero,
        })
        .collect()
}
