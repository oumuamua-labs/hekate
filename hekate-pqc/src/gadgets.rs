// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec::Vec;
use hekate_gadgets::atoms::int_arith::{
    ModAddLayout, ModAddWitness, ModReductionLayout, ModReductionWitness, SchoolbookMulLayout,
    SchoolbookMulWitness, borrow_out, mod_reduction, schoolbook_mul,
};
use hekate_math::TowerField;
use hekate_program::constraint::builder::{ConstraintSystem, Expr};

/// Columns of one modular-addition gadget.
#[derive(Clone, Debug)]
pub struct ModAddCols {
    pub carry: usize,
    pub borrow: usize,
    pub flag: usize,
    pub range_borrow: usize,
}

impl ModAddCols {
    pub(crate) fn alloc(add: &ModAddLayout, alloc: &mut impl FnMut(usize) -> usize) -> Self {
        Self {
            carry: alloc(add.carry_width),
            borrow: alloc(add.borrow_width),
            flag: alloc(1),
            range_borrow: alloc(add.range_borrow_width),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MulModCols {
    pub pp0: (usize, usize),
    pub sums: Vec<(usize, usize)>,
    pub carries: Vec<(usize, usize)>,
    pub product: (usize, usize),
    pub quot: (usize, usize),
    pub rem: (usize, usize),
    pub quot_x_q: (usize, usize),
    pub red_results: Vec<(usize, usize)>,
    pub red_carries: Vec<(usize, usize)>,
    pub red_add_carry: (usize, usize),
    pub red_range_borrow: (usize, usize),
}

impl MulModCols {
    pub(crate) fn alloc(
        bw: usize,
        mul: &SchoolbookMulLayout,
        red: &ModReductionLayout,
        alloc: &mut impl FnMut(usize) -> usize,
    ) -> Self {
        let mut span = |n: usize| (alloc(n), n);

        let pp0 = span(mul.pp0_width);
        let sums = mul.sum_widths.iter().map(|&n| span(n)).collect();
        let carries = mul.carry_widths.iter().map(|&n| span(n)).collect();
        let product = span(mul.product_width);

        let quot = span(bw);
        let rem = span(bw);
        let quot_x_q = span(red.quot_x_mod_width);

        let red_results = red
            .mul_layout
            .scratch_result_widths
            .iter()
            .map(|&n| span(n))
            .collect();

        let red_carries = red
            .mul_layout
            .scratch_carry_widths
            .iter()
            .map(|&n| span(n))
            .collect();

        Self {
            pp0,
            sums,
            carries,
            product,
            quot,
            rem,
            quot_x_q,
            red_results,
            red_carries,
            red_add_carry: span(red.add_carry_width),
            red_range_borrow: span(red.range_borrow_width),
        }
    }

    pub(crate) fn constrain<'a, F: TowerField>(
        &self,
        cs: &'a ConstraintSystem<F>,
        x: &[Expr<'a, F>],
        y: &[Expr<'a, F>],
        q: u32,
    ) -> Vec<Expr<'a, F>> {
        let span = |(start, n): (usize, usize)| bits(cs, start, n);

        let product = span(self.product);
        let rem = span(self.rem);

        let sums = groups(cs, &self.sums);
        let carries = groups(cs, &self.carries);

        let sum_refs: Vec<&[Expr<'_, F>]> = sums.iter().map(Vec::as_slice).collect();
        let carry_refs: Vec<&[Expr<'_, F>]> = carries.iter().map(Vec::as_slice).collect();

        schoolbook_mul(
            cs,
            x,
            y,
            &product,
            &SchoolbookMulWitness {
                pp0: &span(self.pp0),
                sums: &sum_refs,
                carries: &carry_refs,
            },
        );

        let red_results = groups(cs, &self.red_results);
        let red_carries = groups(cs, &self.red_carries);

        let result_refs: Vec<&[Expr<'_, F>]> = red_results.iter().map(Vec::as_slice).collect();
        let carry_refs: Vec<&[Expr<'_, F>]> = red_carries.iter().map(Vec::as_slice).collect();

        mod_reduction(
            cs,
            &product,
            &span(self.quot),
            &rem,
            &ModReductionWitness {
                quot_x_mod_bits: &span(self.quot_x_q),
                mul_scratch_results: &result_refs,
                mul_scratch_carries: &carry_refs,
                add_carry_bits: &span(self.red_add_carry),
                range_borrow_bits: &span(self.red_range_borrow),
            },
            q,
        );

        rem
    }
}

pub(crate) struct ModAddExprs<'a, F: TowerField> {
    carry: Vec<Expr<'a, F>>,
    borrow: Vec<Expr<'a, F>>,
    flag: Expr<'a, F>,
    range_borrow: Vec<Expr<'a, F>>,
}

impl<'a, F: TowerField> ModAddExprs<'a, F> {
    pub(crate) fn new(cs: &'a ConstraintSystem<F>, cols: &ModAddCols, add: &ModAddLayout) -> Self {
        Self {
            carry: bits(cs, cols.carry, add.carry_width),
            borrow: bits(cs, cols.borrow, add.borrow_width),
            flag: cs.col(cols.flag),
            range_borrow: bits(cs, cols.range_borrow, add.range_borrow_width),
        }
    }

    pub(crate) fn witness(&self) -> ModAddWitness<'a, '_, F> {
        ModAddWitness {
            carry: &self.carry,
            borrow: &self.borrow,
            flag: self.flag,
            range_borrow: &self.range_borrow,
        }
    }
}

pub(crate) fn bits<F: TowerField>(
    cs: &ConstraintSystem<F>,
    start: usize,
    n: usize,
) -> Vec<Expr<'_, F>> {
    (start..start + n).map(|c| cs.col(c)).collect()
}

pub(crate) fn groups<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    spans: &[(usize, usize)],
) -> Vec<Vec<Expr<'a, F>>> {
    spans.iter().map(|&(start, n)| bits(cs, start, n)).collect()
}

pub(crate) fn packed<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    bits: &[Expr<'a, F>],
) -> Expr<'a, F> {
    let mut acc = cs.constant(F::ZERO);
    for (k, &bit) in bits.iter().enumerate() {
        acc = acc + bit * cs.constant(F::from(1u128 << k));
    }

    acc
}

pub(crate) fn padded<'a, F: TowerField>(
    v: &[Expr<'a, F>],
    width: usize,
    zero: Expr<'a, F>,
) -> Vec<Expr<'a, F>> {
    (0..width)
        .map(|k| v.get(k).copied().unwrap_or(zero))
        .collect()
}

/// Borrows of `a − b` over bits only: `borrow[k]` is the
/// borrow out of bit `k`; no result cells, no booleanity root.
pub(crate) fn borrow_chain<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    a: &[Expr<'a, F>],
    b: &[Expr<'a, F>],
    borrow: &[Expr<'a, F>],
) {
    let mut borrow_in = None;
    for ((&x, &y), &w) in a.iter().zip(b).zip(borrow) {
        cs.constrain(w + borrow_out(cs, x, y, borrow_in));

        borrow_in = Some(w);
    }
}

pub(crate) fn const_bits<F: TowerField>(
    cs: &ConstraintSystem<F>,
    value: u32,
    width: usize,
) -> Vec<Expr<'_, F>> {
    (0..width)
        .map(|k| match (value >> k) & 1 {
            1 => cs.one(),
            _ => cs.constant(F::ZERO),
        })
        .collect()
}
