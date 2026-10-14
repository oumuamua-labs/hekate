// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Integer arithmetic over binary tower fields as carry and
//! borrow chains over bit cells. Every bit slice must be a
//! cell that holds only 0 or 1; no atom asserts booleanity.

use alloc::vec::Vec;
use hekate_math::TowerField;
use hekate_program::constraint::builder::{ConstraintSystem, Expr};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NafDigit {
    pub shift: usize,
    pub negative: bool,
}

/// Layout for `mul_const` scratch allocation.
#[derive(Clone, Debug)]
pub struct MulConstLayout {
    pub digits: Vec<NafDigit>,
    pub scratch_result_widths: Vec<usize>,
    pub scratch_carry_widths: Vec<usize>,
    pub result_width: usize,
}

/// Layout for modular reduction scratch allocation.
#[derive(Clone, Debug)]
pub struct ModReductionLayout {
    pub mul_layout: MulConstLayout,
    pub quot_x_mod_width: usize,
    pub add_carry_width: usize,
    pub range_borrow_width: usize,
}

/// Witness columns for modular reduction.
pub struct ModReductionWitness<'a, 'b, F: TowerField> {
    pub quot_x_mod_bits: &'b [Expr<'a, F>],
    pub mul_scratch_results: &'b [&'b [Expr<'a, F>]],
    pub mul_scratch_carries: &'b [&'b [Expr<'a, F>]],
    pub add_carry_bits: &'b [Expr<'a, F>],
    pub range_borrow_bits: &'b [Expr<'a, F>],
}

/// Layout for schoolbook multiplication scratch allocation.
#[derive(Clone, Debug)]
pub struct SchoolbookMulLayout {
    pub pp0_width: usize,
    pub sum_widths: Vec<usize>,
    pub carry_widths: Vec<usize>,
    pub product_width: usize,
}

/// Witness columns for schoolbook multiplication.
pub struct SchoolbookMulWitness<'a, 'b, F: TowerField> {
    pub pp0: &'b [Expr<'a, F>],
    pub sums: &'b [&'b [Expr<'a, F>]],
    pub carries: &'b [&'b [Expr<'a, F>]],
}

/// Witness columns for modular addition.
pub struct ModAddWitness<'a, 'b, F: TowerField> {
    pub carry: &'b [Expr<'a, F>],
    pub borrow: &'b [Expr<'a, F>],
    pub flag: Expr<'a, F>,
    pub range_borrow: &'b [Expr<'a, F>],
}

/// Scratch layout for modular addition.
#[derive(Clone, Debug)]
pub struct ModAddLayout {
    pub carry_width: usize,
    pub borrow_width: usize,
    pub range_borrow_width: usize,
}

/// Majority of `a`, `b`, `c` as `(a + c)(b + c) + c`;
/// exact on boolean `c` only.
pub fn carry_out<'a, F: TowerField>(
    a: Expr<'a, F>,
    b: Expr<'a, F>,
    carry_in: Option<Expr<'a, F>>,
) -> Expr<'a, F> {
    match carry_in {
        Some(c) => (a + c) * (b + c) + c,
        None => a * b,
    }
}

/// Borrow of `a - b` with borrow-in `w` as
/// `(1 + a + w)(b + w) + w`; exact on boolean `w` only.
pub fn borrow_out<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    a: Expr<'a, F>,
    b: Expr<'a, F>,
    borrow_in: Option<Expr<'a, F>>,
) -> Expr<'a, F> {
    let one = cs.one();

    match borrow_in {
        Some(w) => (one + a + w) * (b + w) + w,
        None => (one + a) * b,
    }
}

/// Constrain n-bit addition `a + b = result`.
/// `carry_bits[k]` is the carry out of position k;
/// `carry_bits[n-1]` is the overflow, left to the caller.
pub fn add_carry_chain<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    a_bits: &[Expr<'a, F>],
    b_bits: &[Expr<'a, F>],
    result_bits: &[Expr<'a, F>],
    carry_bits: &[Expr<'a, F>],
) {
    chain(cs, None, a_bits, b_bits, result_bits, carry_bits, false);
}

/// `add_carry_chain` with `carry_in` feeding position 0.
pub fn add_carry_chain_with_carry_in<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    carry_in: Expr<'a, F>,
    a_bits: &[Expr<'a, F>],
    b_bits: &[Expr<'a, F>],
    result_bits: &[Expr<'a, F>],
    carry_bits: &[Expr<'a, F>],
) {
    chain(
        cs,
        Some(carry_in),
        a_bits,
        b_bits,
        result_bits,
        carry_bits,
        false,
    );
}

/// Constrain n-bit subtraction `a - b = result`.
/// `borrow_bits[k]` is the borrow out of position k;
/// `borrow_bits[n-1] = 1` means `a < b`, left to the caller.
pub fn sub_borrow_chain<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    a_bits: &[Expr<'a, F>],
    b_bits: &[Expr<'a, F>],
    result_bits: &[Expr<'a, F>],
    borrow_bits: &[Expr<'a, F>],
) {
    chain(cs, None, a_bits, b_bits, result_bits, borrow_bits, true);
}

/// Constrain `value < bound` as the borrow chain of
/// `(bound - 1) - value`: `borrow_bits` holds the borrows
/// out of positions `0..n-1`, the last one is the root.
pub fn range_check<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    value_bits: &[Expr<'a, F>],
    borrow_bits: &[Expr<'a, F>],
    bound: u32,
) {
    let n = value_bits.len();

    assert!(bound > 0, "range_check: bound must be > 0");
    assert_eq!(
        borrow_bits.len(),
        n - 1,
        "range_check: borrow_bits length {} != n-1 ({})",
        borrow_bits.len(),
        n - 1
    );

    let bound_minus_1 = bound - 1;

    let mut borrow_in: Option<Expr<'a, F>> = None;
    for (k, &v) in value_bits.iter().enumerate() {
        let minuend = match (bound_minus_1 >> k) & 1 {
            1 => cs.one(),
            _ => cs.constant(F::ZERO),
        };

        let out = borrow_out(cs, minuend, v, borrow_in);

        match borrow_bits.get(k) {
            Some(&w) => {
                cs.constrain(w + out);

                borrow_in = Some(w);
            }
            None => cs.constrain(out),
        }
    }
}

/// Constrain `result = operand * constant` as a cascade
/// over the signed digits of `constant`, high to low.
pub fn mul_const<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    operand_bits: &[Expr<'a, F>],
    result_bits: &[Expr<'a, F>],
    scratch_results: &[&[Expr<'a, F>]],
    scratch_carries: &[&[Expr<'a, F>]],
    constant: u32,
) {
    let n = operand_bits.len();
    let layout = mul_const_scratch_widths(n, constant);
    let digits = &layout.digits;
    let m = digits.len();

    assert_eq!(
        result_bits.len(),
        layout.result_width,
        "mul_const: result_bits length {} != {}",
        result_bits.len(),
        layout.result_width
    );

    if m == 0 {
        for &r in result_bits {
            cs.constrain(r);
        }

        return;
    }

    let zero = cs.constant(F::ZERO);

    let shifted = |shift: usize, k: usize| -> Expr<'a, F> {
        match k.checked_sub(shift) {
            Some(i) if i < n => operand_bits[i],
            _ => zero,
        }
    };

    if m == 1 {
        for (k, &r) in result_bits.iter().enumerate() {
            cs.constrain(r + shifted(digits[0].shift, k));
        }

        return;
    }

    assert_eq!(
        scratch_results.len(),
        m - 2,
        "mul_const: scratch_results length {} != digits-2 ({})",
        scratch_results.len(),
        m - 2
    );
    assert_eq!(
        scratch_carries.len(),
        m - 1,
        "mul_const: scratch_carries length {} != digits-1 ({})",
        scratch_carries.len(),
        m - 1
    );

    let top = digits[0].shift;

    let mut acc: Vec<Expr<'a, F>> = (0..top + n).map(|k| shifted(top, k)).collect();
    for (j, digit) in digits.iter().enumerate().skip(1) {
        let width = layout.scratch_carry_widths[j - 1];

        let a: Vec<Expr<'a, F>> = (0..width)
            .map(|k| acc.get(k).copied().unwrap_or(zero))
            .collect();
        let b: Vec<Expr<'a, F>> = (0..width).map(|k| shifted(digit.shift, k)).collect();

        let result: &[Expr<'a, F>] = match j == m - 1 {
            true => result_bits,
            false => scratch_results[j - 1],
        };

        let padded: Vec<Expr<'a, F>> = (0..width)
            .map(|k| result.get(k).copied().unwrap_or(zero))
            .collect();

        chain(
            cs,
            None,
            &a,
            &b,
            &padded,
            scratch_carries[j - 1],
            digit.negative,
        );

        acc = padded;
    }
}

/// Constrain `product = quotient * modulus + remainder` with `remainder < modulus`.
pub fn mod_reduction<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    product_bits: &[Expr<'a, F>],
    quotient_bits: &[Expr<'a, F>],
    remainder_bits: &[Expr<'a, F>],
    scratch: &ModReductionWitness<'a, '_, F>,
    modulus: u32,
) {
    mul_const(
        cs,
        quotient_bits,
        scratch.quot_x_mod_bits,
        scratch.mul_scratch_results,
        scratch.mul_scratch_carries,
        modulus,
    );

    let width = scratch.add_carry_bits.len();

    assert!(
        width >= product_bits.len() && width >= scratch.quot_x_mod_bits.len(),
        "mod_reduction: add_carry_bits length {width} below the operand widths"
    );

    let zero = cs.constant(F::ZERO);

    let pad = |bits: &[Expr<'a, F>]| -> Vec<Expr<'a, F>> {
        (0..width)
            .map(|k| bits.get(k).copied().unwrap_or(zero))
            .collect()
    };

    add_carry_chain(
        cs,
        &pad(scratch.quot_x_mod_bits),
        &pad(remainder_bits),
        &pad(product_bits),
        scratch.add_carry_bits,
    );

    cs.constrain(scratch.add_carry_bits[width - 1]);

    range_check(cs, remainder_bits, scratch.range_borrow_bits, modulus);
}

/// Compute column layout for schoolbook
/// multiplication of two variable operands.
pub fn schoolbook_mul_layout(a_width: usize, b_width: usize) -> SchoolbookMulLayout {
    assert!(
        a_width > 0 && b_width > 0,
        "schoolbook_mul_layout: operand widths must be > 0 (a={a_width}, b={b_width})"
    );

    let product_width = a_width + b_width;
    let pp0_width = a_width;

    let carry_widths: Vec<usize> = (1..b_width).map(|j| a_width + j + 1).collect();
    let sum_widths: Vec<usize> = carry_widths[..carry_widths.len().saturating_sub(1)].to_vec();

    SchoolbookMulLayout {
        pp0_width,
        sum_widths,
        carry_widths,
        product_width,
    }
}

/// Constrain `product = a * b`. `pp0[k] = b[0] * a[k]` is
/// committed to keep the carry roots at degree 3; each later
/// `b[j] * a[k-j]` is one node shared by its sum and carry root.
pub fn schoolbook_mul<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    a_bits: &[Expr<'a, F>],
    b_bits: &[Expr<'a, F>],
    product_bits: &[Expr<'a, F>],
    scratch: &SchoolbookMulWitness<'a, '_, F>,
) {
    let a_width = a_bits.len();
    let b_width = b_bits.len();

    assert!(
        a_width > 0 && b_width > 0,
        "schoolbook_mul: operand widths must be > 0 (a={a_width}, b={b_width})"
    );
    assert_eq!(
        scratch.pp0.len(),
        a_width,
        "schoolbook_mul: pp0 length {} != a width {a_width}",
        scratch.pp0.len()
    );
    assert_eq!(
        scratch.carries.len(),
        b_width - 1,
        "schoolbook_mul: carries length {} != b_width-1 ({})",
        scratch.carries.len(),
        b_width - 1
    );
    assert_eq!(
        scratch.sums.len(),
        b_width.saturating_sub(2),
        "schoolbook_mul: sums length {} != b_width-2",
        scratch.sums.len()
    );

    for (k, &pp0_k) in scratch.pp0.iter().enumerate() {
        cs.constrain(pp0_k + b_bits[0] * a_bits[k]);
    }

    if b_width == 1 {
        for (k, &prod_k) in product_bits.iter().enumerate() {
            match scratch.pp0.get(k) {
                Some(&pp0_k) => cs.constrain(prod_k + pp0_k),
                None => cs.constrain(prod_k),
            }
        }

        return;
    }

    let zero = cs.constant(F::ZERO);

    let mut acc: &[Expr<'a, F>] = scratch.pp0;
    for (j, &b_j) in b_bits.iter().enumerate().skip(1) {
        let width = a_width + j + 1;

        let a: Vec<Expr<'a, F>> = (0..width)
            .map(|k| acc.get(k).copied().unwrap_or(zero))
            .collect();

        let b: Vec<Expr<'a, F>> = (0..width)
            .map(|k| match k.checked_sub(j) {
                Some(i) if i < a_width => b_j * a_bits[i],
                _ => zero,
            })
            .collect();

        let result: &[Expr<'a, F>] = match j == b_width - 1 {
            true => product_bits,
            false => scratch.sums[j - 1],
        };

        assert_eq!(
            result.len(),
            width,
            "schoolbook_mul: stage {j} result length {} != chain width {width}",
            result.len()
        );

        add_carry_chain(cs, &a, &b, result, scratch.carries[j - 1]);

        acc = result;
    }
}

/// Constrain `result = (a + b) mod modulus` for reduced
/// operands (`a + b < 2 · modulus`): carries of `a + b`,
/// borrows of `flag · modulus` off it, `result < modulus`.
pub fn mod_add<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    a_bits: &[Expr<'a, F>],
    b_bits: &[Expr<'a, F>],
    result_bits: &[Expr<'a, F>],
    scratch: &ModAddWitness<'a, '_, F>,
    modulus: u32,
) {
    let n = a_bits.len();

    assert_eq!(
        b_bits.len(),
        n,
        "mod_add: b_bits length {} != a_bits length {n}",
        b_bits.len()
    );
    assert_eq!(
        result_bits.len(),
        n,
        "mod_add: result_bits length {} != operand width {n}",
        result_bits.len()
    );
    assert_eq!(
        scratch.carry.len(),
        n,
        "mod_add: carry length {} != operand width {n}",
        scratch.carry.len()
    );
    assert_eq!(
        scratch.borrow.len(),
        n,
        "mod_add: borrow length {} != operand width {n}",
        scratch.borrow.len()
    );

    let zero = cs.constant(F::ZERO);

    let mut carry_in: Option<Expr<'a, F>> = None;
    for k in 0..n {
        cs.constrain(scratch.carry[k] + carry_out(a_bits[k], b_bits[k], carry_in));

        carry_in = Some(scratch.carry[k]);
    }

    let mut borrow_in: Option<Expr<'a, F>> = None;
    for k in 0..n {
        let sum = match k {
            0 => a_bits[0] + b_bits[0],
            _ => a_bits[k] + b_bits[k] + scratch.carry[k - 1],
        };

        let flag_q = match (modulus >> k) & 1 {
            1 => scratch.flag,
            _ => zero,
        };

        let diff = match borrow_in {
            Some(w) => result_bits[k] + sum + flag_q + w,
            None => result_bits[k] + sum + flag_q,
        };

        cs.constrain(diff);
        cs.constrain(scratch.borrow[k] + borrow_out(cs, sum, flag_q, borrow_in));

        borrow_in = Some(scratch.borrow[k]);
    }

    cs.constrain(scratch.carry[n - 1] + scratch.borrow[n - 1]);

    range_check(cs, result_bits, scratch.range_borrow, modulus);
}

pub fn mod_add_scratch_count(operand_width: usize) -> ModAddLayout {
    ModAddLayout {
        carry_width: operand_width,
        borrow_width: operand_width,
        range_borrow_width: operand_width - 1,
    }
}

/// Constrain bus_col = Σ bit_k · 2^k: F::from(2^k) is
/// the element with bit k set and XOR addition preserves
/// the bit pattern. One degree-1 constraint.
pub fn bit_packing<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    bus_col: Expr<'a, F>,
    bits: &[Expr<'a, F>],
) {
    assert!(
        bits.len() <= 64,
        "bit_packing: bits length {} exceeds 64-bit limit",
        bits.len()
    );

    let mut reconstruction = cs.constant(F::ZERO);
    for (k, &bit) in bits.iter().enumerate() {
        reconstruction = reconstruction + bit * cs.constant(F::from(1u128 << k));
    }

    cs.constrain(bus_col + reconstruction);
}

pub fn naf_digits(constant: u32) -> Vec<NafDigit> {
    let mut digits = Vec::new();
    let mut c = constant as i64;
    let mut shift = 0usize;

    while c != 0 {
        if c & 1 == 1 {
            let d = 2 - (c & 3);

            digits.push(NafDigit {
                shift,
                negative: d < 0,
            });

            c -= d;
        }

        c >>= 1;
        shift += 1;
    }

    digits.reverse();

    digits
}

/// Scratch widths for `mul_const`: stage `j` is sized to `partial_j · (2^n - 1)`,
/// which no stage can exceed; no carry-out needs a pin.
pub fn mul_const_scratch_widths(operand_width: usize, constant: u32) -> MulConstLayout {
    let digits = naf_digits(constant);
    let n = operand_width;
    let max_operand = (1u128 << n) - 1;

    let bits = |value: u128| -> usize { (u128::BITS - value.leading_zeros()) as usize };

    if digits.is_empty() {
        return MulConstLayout {
            digits,
            scratch_result_widths: Vec::new(),
            scratch_carry_widths: Vec::new(),
            result_width: 1,
        };
    }

    let mut partial: i128 = 1i128 << digits[0].shift;
    let mut prev_width = digits[0].shift + n;

    let mut result_widths = Vec::with_capacity(digits.len());
    let mut carry_widths = Vec::with_capacity(digits.len());

    for digit in &digits[1..] {
        let term = 1i128 << digit.shift;

        partial += if digit.negative { -term } else { term };

        let width = bits(partial as u128 * max_operand);
        let chain_width = match digit.negative {
            true => prev_width.max(digit.shift + n),
            false => width,
        };

        result_widths.push(width);
        carry_widths.push(chain_width);

        prev_width = width;
    }

    let result_width = result_widths.pop().unwrap_or(prev_width);

    MulConstLayout {
        digits,
        scratch_result_widths: result_widths,
        scratch_carry_widths: carry_widths,
        result_width,
    }
}

/// Compute scratch layout for modular reduction of a
/// `product_width`-bit product by `modulus`.
pub fn mod_reduction_scratch_count(
    quotient_width: usize,
    product_width: usize,
    modulus: u32,
) -> ModReductionLayout {
    let mul_layout = mul_const_scratch_widths(quotient_width, modulus);
    let quot_x_mod_width = mul_layout.result_width;
    let add_carry_width = product_width.max(quot_x_mod_width);
    let range_borrow_width = quotient_width - 1;

    ModReductionLayout {
        mul_layout,
        quot_x_mod_width,
        add_carry_width,
        range_borrow_width,
    }
}

fn chain<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    carry_in: Option<Expr<'a, F>>,
    a_bits: &[Expr<'a, F>],
    b_bits: &[Expr<'a, F>],
    result_bits: &[Expr<'a, F>],
    carry_bits: &[Expr<'a, F>],
    subtract: bool,
) {
    let n = a_bits.len();

    assert_eq!(
        b_bits.len(),
        n,
        "chain: b_bits length {} != a_bits length {n}",
        b_bits.len()
    );
    assert_eq!(
        result_bits.len(),
        n,
        "chain: result_bits length {} != operand width {n}",
        result_bits.len()
    );
    assert_eq!(
        carry_bits.len(),
        n,
        "chain: carry_bits length {} != operand width {n}",
        carry_bits.len()
    );

    let mut carry = carry_in;
    for k in 0..n {
        let (a, b, r) = (a_bits[k], b_bits[k], result_bits[k]);

        let sum = match carry {
            Some(c) => r + a + b + c,
            None => r + a + b,
        };

        let out = match subtract {
            true => borrow_out(cs, a, b, carry),
            false => carry_out(a, b, carry),
        };

        cs.constrain(sum);
        cs.constrain(carry_bits[k] + out);

        carry = Some(carry_bits[k]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hekate_math::{Block128, Flat, HardwareField};
    use hekate_program::constraint::ConstraintAst;

    type F = Block128;

    struct Cols<'a> {
        cs: &'a ConstraintSystem<F>,
        next: usize,
    }

    impl<'a> Cols<'a> {
        fn new(cs: &'a ConstraintSystem<F>) -> Self {
            Self { cs, next: 0 }
        }

        fn take(&mut self, n: usize) -> Vec<Expr<'a, F>> {
            let cols: Vec<_> = (self.next..self.next + n).map(|i| self.cs.col(i)).collect();
            self.next += n;

            cols
        }
    }

    struct Row {
        cells: Vec<Flat<F>>,
    }

    impl Row {
        fn new(num_cols: usize) -> Self {
            Self {
                cells: alloc::vec![F::ZERO.to_hardware(); num_cols],
            }
        }

        fn bits(&mut self, start: usize, width: usize, value: u128) {
            for k in 0..width {
                self.cells[start + k] = F::from((value >> k) & 1).to_hardware();
            }
        }

        fn bit(&mut self, col: usize, on: bool) {
            self.cells[col] = F::from(on as u128).to_hardware();
        }

        fn violations(&self, ast: &ConstraintAst<F>) -> Vec<usize> {
            ast.evaluate(&self.cells, &self.cells)
                .iter()
                .enumerate()
                .filter(|(_, v)| **v != F::ZERO.to_hardware())
                .map(|(i, _)| i)
                .collect()
        }
    }

    fn carries(a: u128, b: u128, width: usize) -> u128 {
        let mut word = 0u128;
        let mut c = false;

        for k in 0..width {
            let (x, y) = (((a >> k) & 1) == 1, ((b >> k) & 1) == 1);
            c = (x & y) | (c & (x ^ y));
            word |= (c as u128) << k;
        }

        word
    }

    fn borrows(a: u128, b: u128, width: usize) -> u128 {
        let mut word = 0u128;
        let mut w = false;

        for k in 0..width {
            let (x, y) = (((a >> k) & 1) == 1, ((b >> k) & 1) == 1);
            w = (!x & y) | (!(x ^ y) & w);
            word |= (w as u128) << k;
        }

        word
    }

    fn range_borrows(value: u128, bound: u32, width: usize) -> u128 {
        borrows((bound - 1) as u128, value, width) & ((1u128 << (width - 1)) - 1)
    }

    fn mask(width: usize) -> u128 {
        (1u128 << width) - 1
    }

    fn fill_mul_const(row: &mut Row, layout: &MulConstLayout, n: usize, x: u128, constant: u32) {
        let mut col = n;
        let result_start = col;
        col += layout.result_width;

        let result_starts: Vec<usize> = layout
            .scratch_result_widths
            .iter()
            .map(|&w| {
                let start = col;
                col += w;
                start
            })
            .collect();
        let carry_starts: Vec<usize> = layout
            .scratch_carry_widths
            .iter()
            .map(|&w| {
                let start = col;
                col += w;
                start
            })
            .collect();

        row.bits(0, n, x);
        row.bits(result_start, layout.result_width, x * constant as u128);

        let mut acc = x << layout.digits[0].shift;

        for (j, digit) in layout.digits.iter().enumerate().skip(1) {
            let term = x << digit.shift;
            let width = layout.scratch_carry_widths[j - 1];

            let (value, carry) = match digit.negative {
                true => (acc - term, borrows(acc, term, width)),
                false => (acc + term, carries(acc, term, width)),
            };

            row.bits(carry_starts[j - 1], width, carry);

            if j < layout.digits.len() - 1 {
                row.bits(
                    result_starts[j - 1],
                    layout.scratch_result_widths[j - 1],
                    value,
                );
            }

            acc = value;
        }
    }

    fn mul_const_case(n: usize, constant: u32) -> (ConstraintAst<F>, MulConstLayout, usize) {
        let layout = mul_const_scratch_widths(n, constant);

        let cs = ConstraintSystem::<F>::new();
        let mut cols = Cols::new(&cs);

        let operand = cols.take(n);
        let result = cols.take(layout.result_width);

        let scratch_results: Vec<Vec<Expr<'_, F>>> = layout
            .scratch_result_widths
            .iter()
            .map(|&w| cols.take(w))
            .collect();
        let scratch_carries: Vec<Vec<Expr<'_, F>>> = layout
            .scratch_carry_widths
            .iter()
            .map(|&w| cols.take(w))
            .collect();

        let sr_refs: Vec<&[Expr<'_, F>]> = scratch_results.iter().map(|v| v.as_slice()).collect();
        let sc_refs: Vec<&[Expr<'_, F>]> = scratch_carries.iter().map(|v| v.as_slice()).collect();

        mul_const(&cs, &operand, &result, &sr_refs, &sc_refs, constant);

        let num_cols = cols.next;

        (cs.build(), layout, num_cols)
    }

    fn schoolbook_case(n: usize) -> (ConstraintAst<F>, SchoolbookMulLayout, usize) {
        let layout = schoolbook_mul_layout(n, n);

        let cs = ConstraintSystem::<F>::new();
        let mut cols = Cols::new(&cs);

        let a = cols.take(n);
        let b = cols.take(n);
        let product = cols.take(layout.product_width);
        let pp0 = cols.take(layout.pp0_width);

        let sum_cols: Vec<Vec<Expr<'_, F>>> =
            layout.sum_widths.iter().map(|&w| cols.take(w)).collect();
        let carry_cols: Vec<Vec<Expr<'_, F>>> =
            layout.carry_widths.iter().map(|&w| cols.take(w)).collect();

        let sum_refs: Vec<&[Expr<'_, F>]> = sum_cols.iter().map(|v| v.as_slice()).collect();
        let carry_refs: Vec<&[Expr<'_, F>]> = carry_cols.iter().map(|v| v.as_slice()).collect();

        schoolbook_mul(
            &cs,
            &a,
            &b,
            &product,
            &SchoolbookMulWitness {
                pp0: &pp0,
                sums: &sum_refs,
                carries: &carry_refs,
            },
        );

        let num_cols = cols.next;

        (cs.build(), layout, num_cols)
    }

    fn mod_add_case(n: usize, q: u32) -> (ConstraintAst<F>, usize) {
        let layout = mod_add_scratch_count(n);

        let cs = ConstraintSystem::<F>::new();
        let mut cols = Cols::new(&cs);

        let a = cols.take(n);
        let b = cols.take(n);
        let result = cols.take(n);
        let carry = cols.take(layout.carry_width);
        let borrow = cols.take(layout.borrow_width);
        let flag = cols.take(1)[0];
        let range_borrow = cols.take(layout.range_borrow_width);

        mod_add(
            &cs,
            &a,
            &b,
            &result,
            &ModAddWitness {
                carry: &carry,
                borrow: &borrow,
                flag,
                range_borrow: &range_borrow,
            },
            q,
        );

        let num_cols = cols.next;

        (cs.build(), num_cols)
    }

    #[test]
    fn add_carry_chain_constraint_count() {
        let cs = ConstraintSystem::<F>::new();
        let n = 12;
        let mut cols = Cols::new(&cs);
        let (a, b, r, c) = (cols.take(n), cols.take(n), cols.take(n), cols.take(n));

        add_carry_chain(&cs, &a, &b, &r, &c);

        assert_eq!(cs.build().roots.len(), 2 * n);
    }

    #[test]
    fn add_carry_chain_accepts_honest_rows_and_rejects_tampered() {
        let cs = ConstraintSystem::<F>::new();
        let n = 12;
        let mut cols = Cols::new(&cs);
        let (a, b, r, c) = (cols.take(n), cols.take(n), cols.take(n), cols.take(n));

        add_carry_chain(&cs, &a, &b, &r, &c);

        let ast = cs.build();

        for (x, y) in [(0u128, 0u128), (4095, 1), (2730, 1365), (3329, 4000)] {
            let mut row = Row::new(4 * n);
            row.bits(0, n, x);
            row.bits(n, n, y);
            row.bits(2 * n, n, (x + y) & mask(n));
            row.bits(3 * n, n, carries(x, y, n));

            assert!(row.violations(&ast).is_empty(), "{x} + {y}");

            row.bits(2 * n, n, ((x + y) & mask(n)) ^ 1);

            assert!(!row.violations(&ast).is_empty(), "{x} + {y} tampered");
        }
    }

    #[test]
    fn sub_borrow_chain_constraint_count() {
        let cs = ConstraintSystem::<F>::new();
        let n = 8;
        let mut cols = Cols::new(&cs);
        let (a, b, r, w) = (cols.take(n), cols.take(n), cols.take(n), cols.take(n));

        sub_borrow_chain(&cs, &a, &b, &r, &w);

        assert_eq!(cs.build().roots.len(), 2 * n);
    }

    #[test]
    fn sub_borrow_chain_accepts_honest_rows_and_rejects_tampered() {
        let cs = ConstraintSystem::<F>::new();
        let n = 8;
        let mut cols = Cols::new(&cs);
        let (a, b, r, w) = (cols.take(n), cols.take(n), cols.take(n), cols.take(n));

        sub_borrow_chain(&cs, &a, &b, &r, &w);

        let ast = cs.build();

        for (x, y) in [(0u128, 0u128), (255, 1), (100, 200), (37, 37)] {
            let mut row = Row::new(4 * n);
            row.bits(0, n, x);
            row.bits(n, n, y);
            row.bits(2 * n, n, x.wrapping_sub(y) & mask(n));
            row.bits(3 * n, n, borrows(x, y, n));

            assert!(row.violations(&ast).is_empty(), "{x} - {y}");
            assert_eq!((borrows(x, y, n) >> (n - 1)) & 1 == 1, x < y);

            row.bits(3 * n, n, borrows(x, y, n) ^ 1);

            assert!(!row.violations(&ast).is_empty(), "{x} - {y} tampered");
        }
    }

    #[test]
    fn range_check_constraint_count() {
        let cs = ConstraintSystem::<F>::new();
        let n = 12;
        let mut cols = Cols::new(&cs);
        let (v, w) = (cols.take(n), cols.take(n - 1));

        range_check(&cs, &v, &w, 3329);

        assert_eq!(cs.build().roots.len(), n);
    }

    #[test]
    fn range_check_accepts_below_bound_and_rejects_at_bound() {
        let cs = ConstraintSystem::<F>::new();
        let n = 12;
        let mut cols = Cols::new(&cs);
        let (v, w) = (cols.take(n), cols.take(n - 1));

        range_check(&cs, &v, &w, 3329);

        let ast = cs.build();

        for value in [0u128, 1, 1664, 3328] {
            let mut row = Row::new(2 * n - 1);
            row.bits(0, n, value);
            row.bits(n, n - 1, range_borrows(value, 3329, n));

            assert!(row.violations(&ast).is_empty(), "{value}");
        }

        for value in [3329u128, 3330, 4095] {
            let mut row = Row::new(2 * n - 1);
            row.bits(0, n, value);
            row.bits(n, n - 1, range_borrows(value, 3329, n));

            assert!(!row.violations(&ast).is_empty(), "{value}");
        }
    }

    #[test]
    fn naf_digits_of_both_moduli() {
        let digit = |shift: usize, negative: bool| NafDigit { shift, negative };

        assert_eq!(
            naf_digits(3329),
            [
                digit(12, false),
                digit(10, true),
                digit(8, false),
                digit(0, false)
            ]
        );
        assert_eq!(
            naf_digits(8380417),
            [digit(23, false), digit(13, true), digit(0, false)]
        );
        assert_eq!(naf_digits(0), []);
        assert_eq!(naf_digits(1 << 9), [digit(9, false)]);
    }

    #[test]
    fn mul_const_layout_q3329() {
        let layout = mul_const_scratch_widths(12, 3329);

        assert_eq!(layout.digits.len(), 4);
        assert_eq!(layout.scratch_result_widths, [24, 24]);
        assert_eq!(layout.scratch_carry_widths, [24, 24, 24]);
        assert_eq!(layout.result_width, 24);
    }

    #[test]
    fn mul_const_layout_q8380417() {
        let layout = mul_const_scratch_widths(23, 8380417);

        assert_eq!(layout.digits.len(), 3);
        assert_eq!(layout.scratch_result_widths, [46]);
        assert_eq!(layout.scratch_carry_widths, [46, 46]);
        assert_eq!(layout.result_width, 46);
    }

    #[test]
    fn mul_const_constraint_count() {
        let (ast, layout, _) = mul_const_case(12, 3329);

        let expected: usize = layout.scratch_carry_widths.iter().map(|&w| 2 * w).sum();

        assert_eq!(ast.roots.len(), expected);
        assert_eq!(expected, 144);
    }

    #[test]
    fn mul_const_accepts_honest_rows_and_rejects_tampered() {
        for (n, constant, operands) in [
            (12usize, 3329u32, [0u128, 1, 4095, 2730]),
            (23, 8380417, [0, 1, (1 << 23) - 1, 4_194_303]),
            (8, 12, [0, 255, 7, 128]),
        ] {
            let (ast, layout, num_cols) = mul_const_case(n, constant);

            for x in operands {
                let mut row = Row::new(num_cols);
                fill_mul_const(&mut row, &layout, n, x, constant);

                assert!(row.violations(&ast).is_empty(), "{x} * {constant}");

                row.bits(n, layout.result_width, (x * constant as u128) ^ 1);

                assert!(
                    !row.violations(&ast).is_empty(),
                    "{x} * {constant} tampered"
                );
            }
        }
    }

    #[test]
    fn mod_reduction_layout_q3329() {
        let bl = mod_reduction_scratch_count(12, 24, 3329);

        assert_eq!(bl.quot_x_mod_width, 24);
        assert_eq!(bl.add_carry_width, 24);
        assert_eq!(bl.range_borrow_width, 11);
    }

    #[test]
    fn mod_reduction_accepts_honest_rows_and_rejects_tampered() {
        let (n, q) = (12usize, 3329u32);
        let product_width = 24;
        let layout = mod_reduction_scratch_count(n, product_width, q);

        let cs = ConstraintSystem::<F>::new();
        let mut cols = Cols::new(&cs);

        let product = cols.take(product_width);
        let quot = cols.take(n);
        let rem = cols.take(n);
        let quot_x_q = cols.take(layout.quot_x_mod_width);

        let result_cols: Vec<Vec<Expr<'_, F>>> = layout
            .mul_layout
            .scratch_result_widths
            .iter()
            .map(|&w| cols.take(w))
            .collect();

        let carry_cols: Vec<Vec<Expr<'_, F>>> = layout
            .mul_layout
            .scratch_carry_widths
            .iter()
            .map(|&w| cols.take(w))
            .collect();

        let add_carry = cols.take(layout.add_carry_width);
        let range_borrow = cols.take(layout.range_borrow_width);

        let result_refs: Vec<&[Expr<'_, F>]> = result_cols.iter().map(|v| v.as_slice()).collect();
        let carry_refs: Vec<&[Expr<'_, F>]> = carry_cols.iter().map(|v| v.as_slice()).collect();

        mod_reduction(
            &cs,
            &product,
            &quot,
            &rem,
            &ModReductionWitness {
                quot_x_mod_bits: &quot_x_q,
                mul_scratch_results: &result_refs,
                mul_scratch_carries: &carry_refs,
                add_carry_bits: &add_carry,
                range_borrow_bits: &range_borrow,
            },
            q,
        );

        let num_cols = cols.next;
        let ast = cs.build();

        for (x, y) in [(0u128, 0u128), (3328, 3328), (17, 1000), (2000, 2500)] {
            let p = x * y;
            let (quotient, remainder) = (p / q as u128, p % q as u128);

            let mut row = Row::new(num_cols);
            row.bits(0, product_width, p);
            row.bits(product_width, n, quotient);
            row.bits(product_width + n, n, remainder);

            let mut sub = Row::new(num_cols - product_width - n - n);
            fill_mul_const(&mut sub, &layout.mul_layout, n, quotient, q);

            let mul_cells = layout.quot_x_mod_width
                + layout
                    .mul_layout
                    .scratch_result_widths
                    .iter()
                    .sum::<usize>()
                + layout.mul_layout.scratch_carry_widths.iter().sum::<usize>();
            row.cells[product_width + 2 * n..product_width + 2 * n + mul_cells]
                .copy_from_slice(&sub.cells[n..n + mul_cells]);

            let add_start = product_width + 2 * n + mul_cells;
            row.bits(
                add_start,
                layout.add_carry_width,
                carries(quotient * q as u128, remainder, layout.add_carry_width),
            );
            row.bits(
                add_start + layout.add_carry_width,
                layout.range_borrow_width,
                range_borrows(remainder, q, n),
            );

            assert!(row.violations(&ast).is_empty(), "{x} * {y}");

            row.bits(product_width + n, n, remainder ^ 1);

            assert!(!row.violations(&ast).is_empty(), "{x} * {y} tampered");
        }
    }

    #[test]
    fn schoolbook_mul_layout_12x12() {
        let layout = schoolbook_mul_layout(12, 12);

        assert_eq!(layout.product_width, 24);
        assert_eq!(layout.pp0_width, 12);
        assert_eq!(layout.carry_widths.len(), 11);
        assert_eq!(layout.sum_widths.len(), 10);

        for (i, &w) in layout.sum_widths.iter().enumerate() {
            assert_eq!(w, 14 + i);
        }

        for (i, &w) in layout.carry_widths.iter().enumerate() {
            assert_eq!(w, 14 + i);
        }
    }

    #[test]
    fn schoolbook_mul_layout_23x23() {
        let layout = schoolbook_mul_layout(23, 23);

        assert_eq!(layout.product_width, 46);
        assert_eq!(layout.pp0_width, 23);
        assert_eq!(layout.carry_widths.len(), 22);
        assert_eq!(layout.sum_widths.len(), 21);
    }

    #[test]
    fn schoolbook_mul_layout_8x12() {
        let layout = schoolbook_mul_layout(8, 12);

        assert_eq!(layout.product_width, 20);
        assert_eq!(layout.pp0_width, 8);
        assert_eq!(layout.carry_widths.len(), 11);
        assert_eq!(layout.sum_widths.len(), 10);
    }

    #[test]
    fn schoolbook_mul_constraint_count_4x4() {
        let (ast, layout, _) = schoolbook_case(4);

        let expected = layout.pp0_width + layout.carry_widths.iter().map(|&w| 2 * w).sum::<usize>();

        assert_eq!(ast.roots.len(), expected);
        assert_eq!(expected, 46);
    }

    #[test]
    fn schoolbook_mul_accepts_honest_rows_and_rejects_tampered() {
        for n in [4usize, 12] {
            let (ast, layout, num_cols) = schoolbook_case(n);

            for (x, y) in [(0u128, 0u128), (mask(n), mask(n)), (5, 9), (mask(n), 1)] {
                let mut row = Row::new(num_cols);
                row.bits(0, n, x);
                row.bits(n, n, y);
                row.bits(2 * n, layout.product_width, x * y);

                let mut col = 2 * n + layout.product_width;

                let partial = |j: usize| -> u128 {
                    match (y >> j) & 1 {
                        1 => x << j,
                        _ => 0,
                    }
                };

                row.bits(col, layout.pp0_width, partial(0));

                col += layout.pp0_width;

                let sum_start = col;

                col += layout.sum_widths.iter().sum::<usize>();

                let carry_start = col;

                let mut acc = partial(0);
                let (mut sum_col, mut carry_col) = (sum_start, carry_start);

                for j in 1..n {
                    let width = layout.carry_widths[j - 1];

                    row.bits(carry_col, width, carries(acc, partial(j), width));
                    carry_col += width;

                    acc += partial(j);

                    if j < n - 1 {
                        row.bits(sum_col, layout.sum_widths[j - 1], acc);
                        sum_col += layout.sum_widths[j - 1];
                    }
                }

                assert!(row.violations(&ast).is_empty(), "{x} * {y} at {n} bits");

                row.bits(2 * n, layout.product_width, (x * y) ^ 1);

                assert!(!row.violations(&ast).is_empty(), "{x} * {y} tampered");
            }
        }
    }

    #[test]
    fn mod_add_layout_q3329() {
        let layout = mod_add_scratch_count(12);

        assert_eq!(layout.carry_width, 12);
        assert_eq!(layout.borrow_width, 12);
        assert_eq!(layout.range_borrow_width, 11);
    }

    #[test]
    fn mod_add_constraint_count_q3329() {
        let (ast, _) = mod_add_case(12, 3329);

        assert_eq!(ast.roots.len(), 4 * 12 + 1);
    }

    #[test]
    fn mod_add_accepts_honest_rows_and_rejects_tampered() {
        for (n, q) in [(12usize, 3329u32), (23, 8380417)] {
            let (ast, num_cols) = mod_add_case(n, q);
            let q = q as u128;

            for (x, y) in [
                (0u128, 0u128),
                (q - 1, q - 1),
                (q - 1, 1),
                (1, 2),
                (q / 2, q / 2 + 1),
            ] {
                let s = x + y;
                let flag = s >= q;
                let r = s - if flag { q } else { 0 };
                let fq = if flag { q } else { 0 };

                let mut row = Row::new(num_cols);
                row.bits(0, n, x);
                row.bits(n, n, y);
                row.bits(2 * n, n, r);
                row.bits(3 * n, n, carries(x, y, n));
                row.bits(4 * n, n, borrows(s, fq, n));
                row.bit(5 * n, flag);
                row.bits(5 * n + 1, n - 1, range_borrows(r, q as u32, n));

                assert!(row.violations(&ast).is_empty(), "{x} + {y} mod {q}");

                row.bit(5 * n, !flag);

                assert!(
                    !row.violations(&ast).is_empty(),
                    "{x} + {y} mod {q} flag flipped"
                );

                row.bit(5 * n, flag);
                row.bits(2 * n, n, r ^ 1);

                assert!(
                    !row.violations(&ast).is_empty(),
                    "{x} + {y} mod {q} tampered"
                );
            }
        }
    }

    #[test]
    fn bit_packing_constraint_count() {
        let cs = ConstraintSystem::<F>::new();
        let bus = cs.col(0);
        let bits: Vec<_> = (1..=12).map(|i| cs.col(i)).collect();

        bit_packing(&cs, bus, &bits);

        assert_eq!(cs.build().roots.len(), 1);
    }

    #[test]
    fn ast_sharing_across_multiple_add_chains() {
        let cs = ConstraintSystem::<F>::new();
        let mut cols = Cols::new(&cs);

        let (a1, b1, r1, c1) = (cols.take(12), cols.take(12), cols.take(12), cols.take(12));
        let (a2, b2, r2, c2) = (cols.take(12), cols.take(12), cols.take(12), cols.take(12));

        add_carry_chain(&cs, &a1, &b1, &r1, &c1);
        add_carry_chain(&cs, &a2, &b2, &r2, &c2);

        let ast = cs.build();

        assert_eq!(ast.roots.len(), 48);
        assert!(!ast.arena.is_empty());
    }
}
