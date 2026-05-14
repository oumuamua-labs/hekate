// SPDX-License-Identifier: Apache-2.0
// This file is part of the hekate-math project.
// Copyright (C) 2026 Andrei Kochergin <andrei@oumuamua.dev>
// Copyright (C) 2026 Oumuamua Labs <info@oumuamua.dev>. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Emulated integer arithmetic
//! over binary tower fields.
//!
//! GF(2^k) addition is XOR with no integer
//! carry. Proving mod-q operations requires
//! bit-decomposed operands with explicit
//! carry/borrow chain constraints.

use alloc::vec::Vec;
use hekate_math::TowerField;
use hekate_program::constraint::builder::{ConstraintSystem, Expr};

/// Constrain n-bit unsigned addition:
/// a + b = result.
///
/// GF(2) carry chain:
/// result[i] = a[i] + b[i] + carry[i],
/// carry[i+1] = a[i]*b[i] + a[i]*carry[i] + b[i]*carry[i].
///
/// carry_bits[0] forced to zero;
/// carry_bits[n] is overflow.
///
/// Bit slices:
/// length n.
///
/// Carry slice:
/// length n+1.
pub fn add_carry_chain<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    a_bits: &[Expr<'a, F>],
    b_bits: &[Expr<'a, F>],
    result_bits: &[Expr<'a, F>],
    carry_bits: &[Expr<'a, F>],
) {
    let n = a_bits.len();
    assert_eq!(
        b_bits.len(),
        n,
        "add_carry_chain: b_bits length {} != a_bits length {n}",
        b_bits.len()
    );
    assert_eq!(
        result_bits.len(),
        n,
        "add_carry_chain: result_bits length {} != operand width {n}",
        result_bits.len()
    );
    assert_eq!(
        carry_bits.len(),
        n + 1,
        "add_carry_chain: carry_bits length {} != n+1 ({})",
        carry_bits.len(),
        n + 1
    );

    // carry[0] = 0
    cs.constrain(carry_bits[0]);

    for i in 0..n {
        let a = a_bits[i];
        let b = b_bits[i];
        let c = carry_bits[i];
        let r = result_bits[i];
        let c_next = carry_bits[i + 1];

        cs.assert_boolean(r);
        cs.assert_boolean(c_next);

        cs.constrain(r + a + b + c);
        cs.constrain(c_next + a * b + a * c + b * c);
    }
}

/// Like `add_carry_chain` but carry_bits[0]
/// is not forced to zero, caller controls
/// the initial carry for multi-word chaining.
pub fn add_carry_chain_with_carry_in<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    a_bits: &[Expr<'a, F>],
    b_bits: &[Expr<'a, F>],
    result_bits: &[Expr<'a, F>],
    carry_bits: &[Expr<'a, F>],
) {
    let n = a_bits.len();

    assert_eq!(
        b_bits.len(),
        n,
        "add_carry_chain_with_carry_in: b_bits length {} != a_bits length {n}",
        b_bits.len()
    );
    assert_eq!(
        result_bits.len(),
        n,
        "add_carry_chain_with_carry_in: result_bits length {} != operand width {n}",
        result_bits.len()
    );
    assert_eq!(
        carry_bits.len(),
        n + 1,
        "add_carry_chain_with_carry_in: carry_bits length {} != n+1 ({})",
        carry_bits.len(),
        n + 1
    );

    for i in 0..n {
        let a = a_bits[i];
        let b = b_bits[i];
        let c = carry_bits[i];
        let r = result_bits[i];
        let c_next = carry_bits[i + 1];

        cs.assert_boolean(r);
        cs.assert_boolean(c_next);

        cs.constrain(r + a + b + c);
        cs.constrain(c_next + a * b + a * c + b * c);
    }
}

/// Like `add_carry_chain_with_carry_in` but
/// every emitted constraint is multiplied by
/// `selector`. Use for multi-opcode chiplets
/// where ADD applies only on `selector = 1` rows.
pub fn add_carry_chain_with_carry_in_gated<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    selector: Expr<'a, F>,
    a_bits: &[Expr<'a, F>],
    b_bits: &[Expr<'a, F>],
    result_bits: &[Expr<'a, F>],
    carry_bits: &[Expr<'a, F>],
) {
    let n = a_bits.len();

    assert_eq!(
        b_bits.len(),
        n,
        "add_carry_chain_with_carry_in_gated: b_bits length {} != a_bits length {n}",
        b_bits.len()
    );
    assert_eq!(
        result_bits.len(),
        n,
        "add_carry_chain_with_carry_in_gated: result_bits length {} != operand width {n}",
        result_bits.len()
    );
    assert_eq!(
        carry_bits.len(),
        n + 1,
        "add_carry_chain_with_carry_in_gated: carry_bits length {} != n+1 ({})",
        carry_bits.len(),
        n + 1
    );

    let one = cs.one();

    for i in 0..n {
        let a = a_bits[i];
        let b = b_bits[i];
        let c = carry_bits[i];
        let r = result_bits[i];
        let c_next = carry_bits[i + 1];

        cs.assert_zero_when(selector, r * (r + one));
        cs.assert_zero_when(selector, c_next * (c_next + one));

        cs.assert_zero_when(selector, r + a + b + c);
        cs.assert_zero_when(selector, c_next + a * b + a * c + b * c);
    }
}

// =====================================================================
// Integer Subtraction (Borrow Chain)
// =====================================================================

/// Constrain n-bit unsigned subtraction:
/// a - b = result.
///
/// GF(2) borrow chain (!a = a+1):
/// result[i] = a[i] + b[i] + borrow[i],
/// borrow[i+1] = b[i] + a[i]*b[i] + borrow[i]
///   + a[i]*borrow[i] + b[i]*borrow[i].
///
/// borrow_bits[0] forced to zero;
/// borrow_bits[n] = 1 means underflow (a < b).
///
/// Bit slices:
/// length n.
///
/// Borrow slice:
/// length n+1.
pub fn sub_borrow_chain<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    a_bits: &[Expr<'a, F>],
    b_bits: &[Expr<'a, F>],
    result_bits: &[Expr<'a, F>],
    borrow_bits: &[Expr<'a, F>],
) {
    let n = a_bits.len();

    assert_eq!(
        b_bits.len(),
        n,
        "sub_borrow_chain: b_bits length {} != a_bits length {n}",
        b_bits.len()
    );
    assert_eq!(
        result_bits.len(),
        n,
        "sub_borrow_chain: result_bits length {} != operand width {n}",
        result_bits.len()
    );
    assert_eq!(
        borrow_bits.len(),
        n + 1,
        "sub_borrow_chain: borrow_bits length {} != n+1 ({})",
        borrow_bits.len(),
        n + 1
    );

    // borrow[0] = 0
    cs.constrain(borrow_bits[0]);

    for i in 0..n {
        let a = a_bits[i];
        let b = b_bits[i];
        let w = borrow_bits[i];
        let r = result_bits[i];
        let w_next = borrow_bits[i + 1];

        cs.assert_boolean(r);
        cs.assert_boolean(w_next);

        cs.constrain(r + a + b + w);
        cs.constrain(w_next + b + a * b + w + a * w + b * w);
    }
}

/// Like `sub_borrow_chain` but every emitted
/// constraint is multiplied by `selector`.
/// Use for multi-opcode chiplets where SUB
/// applies only on `selector = 1` rows.
pub fn sub_borrow_chain_gated<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    selector: Expr<'a, F>,
    a_bits: &[Expr<'a, F>],
    b_bits: &[Expr<'a, F>],
    result_bits: &[Expr<'a, F>],
    borrow_bits: &[Expr<'a, F>],
) {
    let n = a_bits.len();

    assert_eq!(
        b_bits.len(),
        n,
        "sub_borrow_chain_gated: b_bits length {} != a_bits length {n}",
        b_bits.len()
    );
    assert_eq!(
        result_bits.len(),
        n,
        "sub_borrow_chain_gated: result_bits length {} != operand width {n}",
        result_bits.len()
    );
    assert_eq!(
        borrow_bits.len(),
        n + 1,
        "sub_borrow_chain_gated: borrow_bits length {} != n+1 ({})",
        borrow_bits.len(),
        n + 1
    );

    let one = cs.one();

    cs.assert_zero_when(selector, borrow_bits[0]);

    for i in 0..n {
        let a = a_bits[i];
        let b = b_bits[i];
        let w = borrow_bits[i];
        let r = result_bits[i];
        let w_next = borrow_bits[i + 1];

        cs.assert_zero_when(selector, r * (r + one));
        cs.assert_zero_when(selector, w_next * (w_next + one));

        cs.assert_zero_when(selector, r + a + b + w);
        cs.assert_zero_when(selector, w_next + b + a * b + w + a * w + b * w);
    }
}

// =====================================================================
// Range Check
// =====================================================================

/// Constrain value < bound (unsigned).
///
/// Computes (bound-1), value via borrow chain;
/// final borrow forced to zero proves value < bound.
pub fn range_check<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    value_bits: &[Expr<'a, F>],
    result_bits: &[Expr<'a, F>],
    borrow_bits: &[Expr<'a, F>],
    bound: u32,
) {
    let n = value_bits.len();

    assert!(bound > 0, "range_check: bound must be > 0");
    assert_eq!(
        result_bits.len(),
        n,
        "range_check: result_bits length {} != value width {n}",
        result_bits.len()
    );
    assert_eq!(
        borrow_bits.len(),
        n + 1,
        "range_check: borrow_bits length {} != n+1 ({})",
        borrow_bits.len(),
        n + 1
    );

    let bound_minus_1 = bound - 1;

    // borrow[0] = 0
    cs.constrain(borrow_bits[0]);

    for i in 0..n {
        let bm1_bit = (bound_minus_1 >> i) & 1;
        let v = value_bits[i];
        let w = borrow_bits[i];
        let r = result_bits[i];
        let w_next = borrow_bits[i + 1];

        cs.assert_boolean(r);
        cs.assert_boolean(w_next);

        if bm1_bit == 1 {
            let one = cs.one();
            cs.constrain(r + one + v + w);

            // GF(2):
            // pairs cancel → w_next + v*w
            cs.constrain(w_next + v * w);
        } else {
            cs.constrain(r + v + w);
            cs.constrain(w_next + v + w + v * w);
        }
    }

    // Final borrow must be 0
    cs.constrain(borrow_bits[n]);
}

// =====================================================================
// Constant Multiplication
// =====================================================================

/// Constrain result = operand * constant
/// (unsigned integer multiply).
///
/// Cascades additions of shifted operand
/// copies, one per set bit of the constant.
pub fn mul_const<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    operand_bits: &[Expr<'a, F>],
    result_bits: &[Expr<'a, F>],
    scratch_results: &[&[Expr<'a, F>]],
    scratch_carries: &[&[Expr<'a, F>]],
    constant: u32,
) {
    assert!(constant > 0, "mul_const: constant must be > 0");

    let n = operand_bits.len();
    let set_bits: Vec<usize> = (0..32).filter(|&i| (constant >> i) & 1 == 1).collect();
    let m = set_bits.len();

    if m == 0 {
        for r in result_bits {
            cs.constrain(*r);
        }

        return;
    }

    if m == 1 {
        let shift = set_bits[0];
        let result_width = result_bits.len();

        for k in 0..result_width {
            if k < shift || k >= shift + n {
                cs.constrain(result_bits[k]);
            } else {
                cs.constrain(result_bits[k] + operand_bits[k - shift]);
            }
        }

        return;
    }

    assert_eq!(
        scratch_results.len(),
        m - 2,
        "mul_const: scratch_results length {} != popcount-2 ({})",
        scratch_results.len(),
        m - 2
    );
    assert_eq!(
        scratch_carries.len(),
        m - 1,
        "mul_const: scratch_carries length {} != popcount-1 ({})",
        scratch_carries.len(),
        m - 1
    );

    let zero = cs.constant(F::ZERO);

    let shifted_bit = |j: usize, k: usize| -> Expr<'a, F> {
        let shift = set_bits[j];
        if k >= shift && k < shift + n {
            operand_bits[k - shift]
        } else {
            zero
        }
    };

    // First addition:
    // shifted[0] + shifted[1]
    let first_span = core::cmp::max(set_bits[0] + n, set_bits[1] + n);
    let width_0 = first_span + 1; // +1 for carry growth
    let carry_0 = scratch_carries[0];

    assert_eq!(
        carry_0.len(),
        width_0 + 1,
        "mul_const: carry_0 length {} != width+1 ({})",
        carry_0.len(),
        width_0 + 1
    );

    if m == 2 {
        assert_eq!(
            result_bits.len(),
            width_0,
            "mul_const: result_bits length {} != expected width {width_0}",
            result_bits.len()
        );

        let a_bits: Vec<Expr<'a, F>> = (0..width_0).map(|k| shifted_bit(0, k)).collect();
        let b_bits: Vec<Expr<'a, F>> = (0..width_0).map(|k| shifted_bit(1, k)).collect();

        add_carry_chain(cs, &a_bits, &b_bits, result_bits, carry_0);

        return;
    }

    // First addition -> scratch_results[0]
    {
        let a_bits: Vec<Expr<'a, F>> = (0..width_0).map(|k| shifted_bit(0, k)).collect();
        let b_bits: Vec<Expr<'a, F>> = (0..width_0).map(|k| shifted_bit(1, k)).collect();

        add_carry_chain(cs, &a_bits, &b_bits, scratch_results[0], carry_0);
    }

    // Middle additions
    for j in 2..m - 1 {
        let prev = scratch_results[j - 2];
        let cur_width = scratch_results[j - 1].len();
        let carry_j = scratch_carries[j - 1];

        assert_eq!(
            carry_j.len(),
            cur_width + 1,
            "mul_const: carry[{j}] length {} != width+1 ({})",
            carry_j.len(),
            cur_width + 1
        );

        let prev_width = prev.len();
        let a_bits: Vec<Expr<'a, F>> = (0..cur_width)
            .map(|k| if k < prev_width { prev[k] } else { zero })
            .collect();
        let b_bits: Vec<Expr<'a, F>> = (0..cur_width).map(|k| shifted_bit(j, k)).collect();

        add_carry_chain(cs, &a_bits, &b_bits, scratch_results[j - 1], carry_j);
    }

    // Final addition -> result_bits
    {
        let prev = scratch_results[m - 3];
        let final_width = result_bits.len();
        let carry_last = scratch_carries[m - 2];

        assert_eq!(
            carry_last.len(),
            final_width + 1,
            "mul_const: final carry length {} != width+1 ({})",
            carry_last.len(),
            final_width + 1
        );

        let prev_width = prev.len();
        let a_bits: Vec<Expr<'a, F>> = (0..final_width)
            .map(|k| if k < prev_width { prev[k] } else { zero })
            .collect();
        let b_bits: Vec<Expr<'a, F>> = (0..final_width).map(|k| shifted_bit(m - 1, k)).collect();

        add_carry_chain(cs, &a_bits, &b_bits, result_bits, carry_last);
    }
}

// =====================================================================
// Modular Reduction
// =====================================================================

/// Layout for modular reduction
/// scratch allocation.
#[derive(Clone, Debug)]
pub struct ModReductionLayout {
    pub mul_layout: MulConstLayout,
    pub product_width: usize,
    pub add_carry_width: usize,
    pub range_result_width: usize,
    pub range_borrow_width: usize,
    pub total_scratch_bits: usize,
}

/// Witness columns for modular
/// reduction verification.
pub struct ModReductionWitness<'a, 'b, F: TowerField> {
    /// quotient * modulus
    /// (product_width bits, witness).
    pub quot_x_mod_bits: &'b [Expr<'a, F>],

    /// Intermediate result columns
    /// for constant multiply tree.
    pub mul_scratch_results: &'b [&'b [Expr<'a, F>]],

    /// Carry chain columns for
    /// constant multiply tree.
    pub mul_scratch_carries: &'b [&'b [Expr<'a, F>]],

    /// Carry chain for
    /// quot_x_mod + remainder = product
    /// (product_width+1 bits).
    pub add_carry_bits: &'b [Expr<'a, F>],

    /// Result of (modulus-1) - remainder
    /// (remainder_width bits, witness).
    pub range_result_bits: &'b [Expr<'a, F>],

    /// Borrow chain for range check
    /// (remainder_width+1 bits, witness).
    pub range_borrow_bits: &'b [Expr<'a, F>],
}

/// Constrain modular reduction:
/// product = quotient * modulus + remainder,
/// with 0 <= remainder < modulus.
#[allow(clippy::too_many_arguments)]
pub fn mod_reduction<'a, F: TowerField>(
    cs: &'a ConstraintSystem<F>,
    product_bits: &[Expr<'a, F>],
    quotient_bits: &[Expr<'a, F>],
    remainder_bits: &[Expr<'a, F>],
    scratch: &ModReductionWitness<'a, '_, F>,
    modulus: u32,
) {
    let product_width = product_bits.len();
    let remainder_width = remainder_bits.len();

    // 1. quotient * modulus = quot_x_mod
    mul_const(
        cs,
        quotient_bits,
        scratch.quot_x_mod_bits,
        scratch.mul_scratch_results,
        scratch.mul_scratch_carries,
        modulus,
    );

    // 2. quot_x_mod + remainder = product
    let zero = cs.constant(F::ZERO);
    let remainder_padded: Vec<Expr<'a, F>> = (0..product_width)
        .map(|k| {
            if k < remainder_width {
                remainder_bits[k]
            } else {
                zero
            }
        })
        .collect();

    add_carry_chain(
        cs,
        scratch.quot_x_mod_bits,
        &remainder_padded,
        product_bits,
        scratch.add_carry_bits,
    );

    // Overflow carry must be 0
    cs.constrain(scratch.add_carry_bits[product_width]);

    // 3. remainder < modulus
    range_check(
        cs,
        remainder_bits,
        scratch.range_result_bits,
        scratch.range_borrow_bits,
        modulus,
    );
}

// =====================================================================
// Schoolbook Multiplication (Variable × Variable)
// =====================================================================

/// Layout for schoolbook multiplication
/// scratch allocation.
#[derive(Clone, Debug)]
pub struct SchoolbookMulLayout {
    /// Width of the materialized first
    /// partial product (= a_width bits).
    pub pp0_width: usize,

    /// Width of each intermediate sum.
    /// Length = b_width - 2.
    pub sum_widths: Vec<usize>,

    /// Width of each carry chain
    /// (includes +1 for carry-out).
    /// Length = b_width - 1.
    pub carry_widths: Vec<usize>,

    /// Width of the final product.
    pub product_width: usize,

    /// Total scratch bit columns needed
    /// (pp0 + intermediate sums + carries).
    pub total_scratch_bits: usize,
}

/// Compute column layout for schoolbook
/// multiplication of two variable operands.
///
/// First partial product is materialized
/// to keep constraint degree ≤ 4 with gating.
pub fn schoolbook_mul_layout(a_width: usize, b_width: usize) -> SchoolbookMulLayout {
    assert!(
        a_width > 0 && b_width > 0,
        "schoolbook_mul_layout: operand widths must be > 0 (a={a_width}, b={b_width})"
    );

    let product_width = a_width + b_width;
    let pp0_width = a_width;

    if b_width == 1 {
        return SchoolbookMulLayout {
            pp0_width,
            sum_widths: Vec::new(),
            carry_widths: Vec::new(),
            product_width,
            total_scratch_bits: pp0_width,
        };
    }

    let mut sum_widths = Vec::with_capacity(b_width - 2);
    let mut carry_widths = Vec::with_capacity(b_width - 1);

    let first_span = 1 + a_width;
    let width_1 = first_span + 1;

    carry_widths.push(width_1 + 1);

    if b_width == 2 {
        let scratch = pp0_width + carry_widths.iter().sum::<usize>();

        return SchoolbookMulLayout {
            pp0_width,
            sum_widths,
            carry_widths,
            product_width: width_1,
            total_scratch_bits: scratch,
        };
    }

    sum_widths.push(width_1);

    let mut current_width = width_1;
    for j in 2..b_width - 1 {
        let shifted_max = j + a_width;
        current_width = core::cmp::max(current_width, shifted_max) + 1;

        carry_widths.push(current_width + 1);
        sum_widths.push(current_width);
    }

    let shifted_max = (b_width - 1) + a_width;
    let final_width = core::cmp::max(current_width, shifted_max) + 1;

    carry_widths.push(final_width + 1);

    let total_scratch: usize =
        pp0_width + sum_widths.iter().sum::<usize>() + carry_widths.iter().sum::<usize>();

    SchoolbookMulLayout {
        pp0_width,
        sum_widths,
        carry_widths,
        product_width: final_width,
        total_scratch_bits: total_scratch,
    }
}

/// Witness columns for
/// schoolbook multiplication.
pub struct SchoolbookMulWitness<'a, 'b, F: TowerField> {
    /// Materialized first partial product
    /// (pp0_width bits).
    /// pp0[k] = b[0]*a[k].
    pub pp0: &'b [Expr<'a, F>],

    /// Intermediate sum columns.
    /// Length = b_width - 2 groups.
    pub sums: &'b [&'b [Expr<'a, F>]],

    /// Carry chain columns.
    /// Length = b_width - 1 groups.
    pub carries: &'b [&'b [Expr<'a, F>]],
}

/// Constrain schoolbook multiplication:
/// product = a * b (unsigned integers).
///
/// pp[j][k] = b[j] * a[k-j] (degree-2).
/// pp0 materialized as witness to keep
/// constraint degree ≤ 3 (gated: ≤ 4).
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
        a_width, b_width,
        "schoolbook_mul: operand widths must match ({a_width} vs {b_width})"
    );
    assert_eq!(
        scratch.carries.len(),
        b_width - 1,
        "schoolbook_mul: carries length {} != b_width-1 ({})",
        scratch.carries.len(),
        b_width - 1
    );

    let zero = cs.constant(F::ZERO);

    for (k, &pp0_k) in scratch.pp0.iter().enumerate() {
        cs.assert_boolean(pp0_k);

        if k < a_width {
            cs.constrain(pp0_k + b_bits[0] * a_bits[k]);
        } else {
            cs.constrain(pp0_k);
        }
    }

    let gated = |j: usize, k: usize, width: usize| -> Expr<'a, F> {
        if k >= j && k - j < a_width && k < width {
            b_bits[j] * a_bits[k - j]
        } else {
            zero
        }
    };

    if b_width == 1 {
        let pp0_len = scratch.pp0.len();
        for (k, &prod_k) in product_bits.iter().enumerate() {
            if k < pp0_len {
                cs.constrain(prod_k + scratch.pp0[k]);
            } else {
                cs.constrain(prod_k);
            }
        }

        return;
    }

    // First addition:
    // pp0 + pp[1]
    let width_1 = scratch.carries[0].len() - 1;
    let pp0_padded: Vec<Expr<'a, F>> = (0..width_1)
        .map(|k| {
            if k < scratch.pp0.len() {
                scratch.pp0[k]
            } else {
                zero
            }
        })
        .collect();

    let pp1: Vec<Expr<'a, F>> = (0..width_1).map(|k| gated(1, k, width_1)).collect();

    if b_width == 2 {
        add_carry_chain(cs, &pp0_padded, &pp1, product_bits, scratch.carries[0]);
        return;
    }

    // First addition -> sums[0]
    add_carry_chain(cs, &pp0_padded, &pp1, scratch.sums[0], scratch.carries[0]);

    // Middle additions
    for j in 2..b_width - 1 {
        let prev = scratch.sums[j - 2];
        let cur_width = scratch.sums[j - 1].len();

        let prev_width = prev.len();
        let a_padded: Vec<Expr<'a, F>> = (0..cur_width)
            .map(|k| if k < prev_width { prev[k] } else { zero })
            .collect();
        let b_shifted: Vec<Expr<'a, F>> = (0..cur_width).map(|k| gated(j, k, cur_width)).collect();

        add_carry_chain(
            cs,
            &a_padded,
            &b_shifted,
            scratch.sums[j - 1],
            scratch.carries[j - 1],
        );
    }

    // Final addition -> product
    let last_j = b_width - 1;
    let prev = scratch.sums[last_j - 2];
    let final_width = product_bits.len();

    let prev_width = prev.len();
    let a_padded: Vec<Expr<'a, F>> = (0..final_width)
        .map(|k| if k < prev_width { prev[k] } else { zero })
        .collect();
    let b_shifted: Vec<Expr<'a, F>> = (0..final_width)
        .map(|k| gated(last_j, k, final_width))
        .collect();

    add_carry_chain(
        cs,
        &a_padded,
        &b_shifted,
        product_bits,
        scratch.carries[last_j - 1],
    );
}

// =====================================================================
// Modular Addition / Subtraction
// =====================================================================

/// Witness columns for modular addition.
pub struct ModAddWitness<'a, 'b, F: TowerField> {
    /// LHS = a + b:
    /// result bits.
    pub lhs_result: &'b [Expr<'a, F>],

    /// LHS carry chain.
    pub lhs_carry: &'b [Expr<'a, F>],

    /// RHS = result + flag*modulus:
    /// result bits.
    pub rhs_result: &'b [Expr<'a, F>],

    /// RHS carry chain.
    pub rhs_carry: &'b [Expr<'a, F>],

    /// Reduction flag (boolean).
    pub flag: Expr<'a, F>,

    /// Range check result bits (result < modulus).
    pub range_result: &'b [Expr<'a, F>],

    /// Range check borrow chain.
    pub range_borrow: &'b [Expr<'a, F>],
}

/// Constrain modular addition:
/// result = (a + b) mod modulus.
///
/// Precondition:
/// a + b < 2 * modulus
/// (holds when both operands are reduced).
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

    cs.assert_boolean(scratch.flag);

    // LHS:
    // a + b
    add_carry_chain(cs, a_bits, b_bits, scratch.lhs_result, scratch.lhs_carry);

    // RHS:
    // result + flag * modulus
    let flag_q: Vec<Expr<'a, F>> = (0..n)
        .map(|k| {
            if (modulus >> k) & 1 == 1 {
                scratch.flag
            } else {
                cs.constant(F::ZERO)
            }
        })
        .collect();

    add_carry_chain(
        cs,
        result_bits,
        &flag_q,
        scratch.rhs_result,
        scratch.rhs_carry,
    );

    // LHS = RHS at each bit
    let lhs_width = scratch.lhs_result.len();
    let rhs_width = scratch.rhs_result.len();
    let check_width = core::cmp::max(lhs_width, rhs_width);
    let zero = cs.constant(F::ZERO);

    for k in 0..check_width {
        let l = if k < lhs_width {
            scratch.lhs_result[k]
        } else {
            zero
        };
        let r = if k < rhs_width {
            scratch.rhs_result[k]
        } else {
            zero
        };

        cs.constrain(l + r);
    }

    // Overflow carries must match
    cs.constrain(scratch.lhs_carry[lhs_width] + scratch.rhs_carry[rhs_width]);

    // result < modulus
    range_check(
        cs,
        result_bits,
        scratch.range_result,
        scratch.range_borrow,
        modulus,
    );
}

/// Scratch layout for modular
/// addition/subtraction.
#[derive(Clone, Debug)]
pub struct ModAddLayout {
    /// Width of LHS/RHS result columns.
    pub result_width: usize,

    /// Width of LHS/RHS carry columns.
    pub carry_width: usize,

    /// Width of range result columns.
    pub range_result_width: usize,

    /// Width of range borrow columns.
    pub range_borrow_width: usize,

    /// Total scratch bits
    /// (2×result + 2×carry + 1 flag +
    /// range_result + range_borrow).
    pub total_scratch_bits: usize,
}

/// Compute scratch layout for
/// modular addition (or subtraction).
pub fn mod_add_scratch_count(operand_width: usize) -> ModAddLayout {
    let result_width = operand_width;
    let carry_width = operand_width + 1;
    let range_result_width = operand_width;
    let range_borrow_width = operand_width + 1;

    ModAddLayout {
        result_width,
        carry_width,
        range_result_width,
        range_borrow_width,
        total_scratch_bits: 2 * result_width
            + 2 * carry_width
            + 1  // flag
            + range_result_width
            + range_borrow_width,
    }
}

// =====================================================================
// Bit Packing (Bus <> Decomposition Binding)
// =====================================================================

/// Constrain bus_col = Σ bit_k · 2^k.
///
/// F::from(2^k) maps to the element with
/// bit k set; XOR addition preserves the
/// bit pattern. Emits one degree-1 constraint.
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

// =====================================================================
// Helpers
// =====================================================================

/// Compute scratch column widths for
/// `mul_const` with given operand
/// width and constant.
pub fn mul_const_scratch_widths(operand_width: usize, constant: u32) -> MulConstLayout {
    let set_bits: Vec<usize> = (0..32).filter(|&i| (constant >> i) & 1 == 1).collect();
    let m = set_bits.len();

    if m <= 1 {
        let result_width = if m == 0 {
            1
        } else {
            set_bits[0] + operand_width
        };
        return MulConstLayout {
            scratch_result_widths: Vec::new(),
            scratch_carry_widths: Vec::new(),
            result_width,
        };
    }

    let mut scratch_result_widths = Vec::with_capacity(m.saturating_sub(2));
    let mut scratch_carry_widths = Vec::with_capacity(m - 1);

    let first_span = core::cmp::max(set_bits[0] + operand_width, set_bits[1] + operand_width);
    let first_width = first_span + 1; // +1 for carry growth

    scratch_carry_widths.push(first_width + 1);

    if m == 2 {
        return MulConstLayout {
            scratch_result_widths,
            scratch_carry_widths,
            result_width: first_width,
        };
    }

    scratch_result_widths.push(first_width);

    let mut current_width = first_width;
    for &shift in &set_bits[2..m - 1] {
        let shifted_max = shift + operand_width;
        current_width = core::cmp::max(current_width, shifted_max) + 1;
        scratch_carry_widths.push(current_width + 1);
        scratch_result_widths.push(current_width);
    }

    let shifted_max = set_bits[m - 1] + operand_width;
    let result_width = core::cmp::max(current_width, shifted_max) + 1;
    scratch_carry_widths.push(result_width + 1);

    MulConstLayout {
        scratch_result_widths,
        scratch_carry_widths,
        result_width,
    }
}

/// Layout for `mul_const` scratch allocation.
#[derive(Clone, Debug)]
pub struct MulConstLayout {
    /// Width of each intermediate result array.
    /// Length = popcount(constant) - 2.
    pub scratch_result_widths: Vec<usize>,

    /// Width of each carry chain
    /// (includes the +1 for carry-out).
    /// Length = popcount(constant) - 1.
    pub scratch_carry_widths: Vec<usize>,

    /// Width of the final product result.
    pub result_width: usize,
}

/// Compute scratch layout for modular
/// reduction with given parameters.
pub fn mod_reduction_scratch_count(quotient_width: usize, modulus: u32) -> ModReductionLayout {
    let mul_layout = mul_const_scratch_widths(quotient_width, modulus);
    let product_width = mul_layout.result_width;

    let mul_scratch_bits: usize = mul_layout.scratch_result_widths.iter().sum::<usize>()
        + mul_layout.scratch_carry_widths.iter().sum::<usize>();

    ModReductionLayout {
        mul_layout,
        product_width,
        add_carry_width: product_width + 1,
        range_result_width: quotient_width,
        range_borrow_width: quotient_width + 1,
        total_scratch_bits: product_width     // quot_x_mod
            + mul_scratch_bits                // mul intermediates
            + (product_width + 1)             // add carry
            + quotient_width                  // range result
            + (quotient_width + 1), // range borrow
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hekate_math::{Block128, Flat, HardwareField, TowerField};

    type F = Block128;

    #[test]
    fn add_carry_chain_constraint_count() {
        // 12-bit addition:
        // 12 sum + 12 carry + 1 carry_init = 25 constraints
        let cs = ConstraintSystem::<F>::new();
        let n = 12;
        let a: Vec<_> = (0..n).map(|i| cs.col(i)).collect();
        let b: Vec<_> = (0..n).map(|i| cs.col(n + i)).collect();
        let r: Vec<_> = (0..n).map(|i| cs.col(2 * n + i)).collect();
        let c: Vec<_> = (0..=n).map(|i| cs.col(3 * n + i)).collect();

        add_carry_chain(&cs, &a, &b, &r, &c);

        let ast = cs.build();

        // 1 (carry init) + n * 4 (bool_r + bool_c + sum + carry per bit) = 49
        assert_eq!(ast.roots.len(), 4 * n + 1);
    }

    #[test]
    fn add_carry_chain_with_carry_in_gated_constraint_count() {
        let cs = ConstraintSystem::<F>::new();
        let n = 8;
        let s = cs.col(0);
        let a: Vec<_> = (0..n).map(|i| cs.col(1 + i)).collect();
        let b: Vec<_> = (0..n).map(|i| cs.col(1 + n + i)).collect();
        let r: Vec<_> = (0..n).map(|i| cs.col(1 + 2 * n + i)).collect();
        let c: Vec<_> = (0..=n).map(|i| cs.col(1 + 3 * n + i)).collect();

        add_carry_chain_with_carry_in_gated(&cs, s, &a, &b, &r, &c);

        let ast = cs.build();

        assert_eq!(ast.roots.len(), 4 * n);
    }

    #[test]
    fn add_carry_chain_with_carry_in_gated_body_vanishes_when_selector_zero() {
        use hekate_math::{Flat, HardwareField, TowerField};

        let cs = ConstraintSystem::<F>::new();
        let n = 4;
        let s = cs.col(0);
        let a: Vec<_> = (0..n).map(|i| cs.col(1 + i)).collect();
        let b: Vec<_> = (0..n).map(|i| cs.col(1 + n + i)).collect();
        let r: Vec<_> = (0..n).map(|i| cs.col(1 + 2 * n + i)).collect();
        let c: Vec<_> = (0..=n).map(|i| cs.col(1 + 3 * n + i)).collect();

        add_carry_chain_with_carry_in_gated(&cs, s, &a, &b, &r, &c);

        let ast = cs.build();

        let zero = F::ZERO.to_hardware();
        let selector_off = zero;
        let garbage = F::from(0xDEADBEEFu128).to_hardware();

        let mut row: Vec<Flat<F>> = Vec::with_capacity(1 + 3 * n + (n + 1));
        row.push(selector_off);

        for _ in 0..3 * n {
            row.push(garbage);
        }

        for _ in 0..=n {
            row.push(garbage);
        }

        let evals = ast.evaluate(&row, &row);
        for (i, v) in evals.iter().enumerate() {
            assert_eq!(
                *v, zero,
                "constraint {i} non-zero on selector=0 garbage row"
            );
        }
    }

    #[test]
    fn sub_borrow_chain_gated_constraint_count() {
        let cs = ConstraintSystem::<F>::new();
        let n = 8;
        let s = cs.col(0);
        let a: Vec<_> = (0..n).map(|i| cs.col(1 + i)).collect();
        let b: Vec<_> = (0..n).map(|i| cs.col(1 + n + i)).collect();
        let r: Vec<_> = (0..n).map(|i| cs.col(1 + 2 * n + i)).collect();
        let w: Vec<_> = (0..=n).map(|i| cs.col(1 + 3 * n + i)).collect();

        sub_borrow_chain_gated(&cs, s, &a, &b, &r, &w);

        let ast = cs.build();

        assert_eq!(ast.roots.len(), 4 * n + 1);
    }

    #[test]
    fn sub_borrow_chain_gated_body_vanishes_when_selector_zero() {
        let cs = ConstraintSystem::<F>::new();
        let n = 4;
        let s = cs.col(0);
        let a: Vec<_> = (0..n).map(|i| cs.col(1 + i)).collect();
        let b: Vec<_> = (0..n).map(|i| cs.col(1 + n + i)).collect();
        let r: Vec<_> = (0..n).map(|i| cs.col(1 + 2 * n + i)).collect();
        let w: Vec<_> = (0..=n).map(|i| cs.col(1 + 3 * n + i)).collect();

        sub_borrow_chain_gated(&cs, s, &a, &b, &r, &w);

        let ast = cs.build();

        let zero = F::ZERO.to_hardware();
        let selector_off = zero;
        let garbage = F::from(0xBADF00Du128).to_hardware();

        let mut row: Vec<Flat<F>> = Vec::with_capacity(1 + 3 * n + (n + 1));
        row.push(selector_off);

        for _ in 0..3 * n {
            row.push(garbage);
        }

        for _ in 0..=n {
            row.push(garbage);
        }

        let evals = ast.evaluate(&row, &row);
        for (i, v) in evals.iter().enumerate() {
            assert_eq!(
                *v, zero,
                "constraint {i} non-zero on selector=0 garbage row"
            );
        }
    }

    #[test]
    fn sub_borrow_chain_constraint_count() {
        let cs = ConstraintSystem::<F>::new();
        let n = 8;
        let a: Vec<_> = (0..n).map(|i| cs.col(i)).collect();
        let b: Vec<_> = (0..n).map(|i| cs.col(n + i)).collect();
        let r: Vec<_> = (0..n).map(|i| cs.col(2 * n + i)).collect();
        let w: Vec<_> = (0..=n).map(|i| cs.col(3 * n + i)).collect();

        sub_borrow_chain(&cs, &a, &b, &r, &w);

        let ast = cs.build();

        // Same pattern as addition:
        // 1 + 4*n = 33
        assert_eq!(ast.roots.len(), 4 * n + 1);
    }

    #[test]
    fn range_check_constraint_count() {
        let cs = ConstraintSystem::<F>::new();
        let n = 12;
        let v: Vec<_> = (0..n).map(|i| cs.col(i)).collect();
        let r: Vec<_> = (0..n).map(|i| cs.col(n + i)).collect();
        let w: Vec<_> = (0..=n).map(|i| cs.col(2 * n + i)).collect();

        range_check(&cs, &v, &r, &w, 3329);

        let ast = cs.build();

        // 1 (borrow init) + 4*n (bool_r + bool_w + result + borrow per bit) + 1 (final borrow = 0) = 50
        assert_eq!(ast.roots.len(), 4 * n + 2);
    }

    #[test]
    fn mul_const_layout_q3329() {
        // q = 3329 = 0b110100000001
        // Set bits:
        // 0, 8, 10, 11 → popcount = 4
        let layout = mul_const_scratch_widths(12, 3329);

        assert_eq!(layout.scratch_result_widths.len(), 2); // popcount - 2
        assert_eq!(layout.scratch_carry_widths.len(), 3); // popcount - 1

        // Result width:
        // 24 (carry growth across 3 additions)
        assert_eq!(layout.result_width, 24);
    }

    #[test]
    fn mul_const_layout_q8380417() {
        // q = 8380417 = 0b11111111110000000000001
        // Set bits:
        // 0, 13..22 → popcount = 11
        let layout = mul_const_scratch_widths(23, 8380417);

        assert_eq!(layout.scratch_result_widths.len(), 9); // popcount - 2
        assert_eq!(layout.scratch_carry_widths.len(), 10); // popcount - 1

        // Result width:
        // 46 (carry growth across 10 additions)
        assert_eq!(layout.result_width, 46);
    }

    #[test]
    fn mul_const_constraint_count() {
        // Multiply 12-bit operand by 3329
        // (popcount 5 → 4 additions).
        let layout = mul_const_scratch_widths(12, 3329);
        let total_cols = 12 // operand
            + layout.result_width
            + layout.scratch_result_widths.iter().sum::<usize>()
            + layout.scratch_carry_widths.iter().sum::<usize>();

        let cs = ConstraintSystem::<F>::new();
        let mut col_idx = 0;
        let mut alloc = |n: usize| -> Vec<Expr<'_, F>> {
            let cols: Vec<_> = (col_idx..col_idx + n).map(|i| cs.col(i)).collect();
            col_idx += n;
            cols
        };

        let operand = alloc(12);
        let result = alloc(layout.result_width);

        let scratch_results: Vec<Vec<Expr<'_, F>>> = layout
            .scratch_result_widths
            .iter()
            .map(|&w| alloc(w))
            .collect();
        let scratch_carries: Vec<Vec<Expr<'_, F>>> = layout
            .scratch_carry_widths
            .iter()
            .map(|&w| alloc(w))
            .collect();

        let sr_refs: Vec<&[Expr<'_, F>]> = scratch_results.iter().map(|v| v.as_slice()).collect();
        let sc_refs: Vec<&[Expr<'_, F>]> = scratch_carries.iter().map(|v| v.as_slice()).collect();

        mul_const(&cs, &operand, &result, &sr_refs, &sc_refs, 3329);

        let ast = cs.build();

        // 3 additions (popcount-1=3), each
        // with (1 + 4*width) constraints
        // (carry_init + 4 per bit).
        let expected: usize = layout
            .scratch_carry_widths
            .iter()
            .map(|&cw| {
                let width = cw - 1;
                4 * width + 1
            })
            .sum();

        assert_eq!(ast.roots.len(), expected);

        let _ = total_cols; // suppress unused
    }

    #[test]
    fn mod_reduction_layout_q3329() {
        let bl = mod_reduction_scratch_count(12, 3329);

        // Product width:
        // 24 (12-bit quotient × 12-bit
        // modulus, with carry growth).
        assert_eq!(bl.product_width, 24);
        assert_eq!(bl.add_carry_width, 25);
        assert_eq!(bl.range_result_width, 12);
        assert_eq!(bl.range_borrow_width, 13);
    }

    #[test]
    fn ast_sharing_across_multiple_add_chains() {
        // Two independent 12-bit
        // additions sharing the same
        // ConstraintSystem should deduplicate
        // the constant nodes.
        let cs = ConstraintSystem::<F>::new();

        let a1: Vec<_> = (0..12).map(|i| cs.col(i)).collect();
        let b1: Vec<_> = (12..24).map(|i| cs.col(i)).collect();
        let r1: Vec<_> = (24..36).map(|i| cs.col(i)).collect();
        let c1: Vec<_> = (36..49).map(|i| cs.col(i)).collect();

        let a2: Vec<_> = (49..61).map(|i| cs.col(i)).collect();
        let b2: Vec<_> = (61..73).map(|i| cs.col(i)).collect();
        let r2: Vec<_> = (73..85).map(|i| cs.col(i)).collect();
        let c2: Vec<_> = (85..98).map(|i| cs.col(i)).collect();

        add_carry_chain(&cs, &a1, &b1, &r1, &c1);
        add_carry_chain(&cs, &a2, &b2, &r2, &c2);

        let ast = cs.build();

        // Two additions of
        // 12 bits = 2 * (4*12 + 1) = 98 constraints
        assert_eq!(ast.roots.len(), 98);

        // AST arena should be smaller than 2x
        // a single addition because the carry
        // formula structure (maj pattern) is
        // shared — the Mul nodes for a*b, a*c,
        // b*c have the same shape, though with
        // different column operands they won't
        // deduplicate (cell dedup only matches
        // same column index). Constants DO
        // deduplicate: the zero constant
        // for carry[0] is allocated once.
        //
        // Just verify the arena is non-empty and reasonable.
        assert!(!ast.arena.is_empty());
    }

    #[test]
    fn schoolbook_mul_layout_12x12() {
        let layout = schoolbook_mul_layout(12, 12);

        // 12-bit × 12-bit → 24-bit product
        assert_eq!(layout.product_width, 24);
        assert_eq!(layout.pp0_width, 12);

        // 11 additions (b_width - 1)
        assert_eq!(layout.carry_widths.len(), 11);

        // 10 intermediate sums (b_width - 2)
        assert_eq!(layout.sum_widths.len(), 10);

        // Sum widths:
        // 14, 15, 16, ..., 23
        for (i, &w) in layout.sum_widths.iter().enumerate() {
            assert_eq!(w, 14 + i);
        }

        // Carry widths:
        // sum_width + 1
        for (i, &w) in layout.carry_widths.iter().enumerate() {
            assert_eq!(w, 15 + i);
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
    fn schoolbook_mul_constraint_count_4x4() {
        // Small case:
        // 4-bit × 4-bit = 8-bit.
        let layout = schoolbook_mul_layout(4, 4);

        let cs = ConstraintSystem::<F>::new();
        let mut col_idx = 0;

        let mut alloc = |n: usize| -> Vec<Expr<'_, F>> {
            let cols: Vec<_> = (col_idx..col_idx + n).map(|i| cs.col(i)).collect();
            col_idx += n;

            cols
        };

        let a = alloc(4);
        let b = alloc(4);
        let product = alloc(layout.product_width);
        let pp0 = alloc(layout.pp0_width);

        let sums: Vec<Vec<Expr<'_, F>>> = layout.sum_widths.iter().map(|&w| alloc(w)).collect();
        let carries: Vec<Vec<Expr<'_, F>>> =
            layout.carry_widths.iter().map(|&w| alloc(w)).collect();

        let sum_refs: Vec<&[Expr<'_, F>]> = sums.iter().map(|v| v.as_slice()).collect();
        let carry_refs: Vec<&[Expr<'_, F>]> = carries.iter().map(|v| v.as_slice()).collect();

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

        let ast = cs.build();

        // pp0 materialization:
        // 4 × (boolean + constrain) = 8
        // 3 additions (b_width - 1):
        //   carry widths: [7, 8, 9]
        //   each: 4*width + 1 constraints
        //   = 29 + 33 + 37 = 99
        // Total: 8 + 99 = 107
        let pp0_constraints = 2 * layout.pp0_width;
        let add_constraints: usize = layout.carry_widths.iter().map(|&cw| 4 * (cw - 1) + 1).sum();

        assert_eq!(ast.roots.len(), pp0_constraints + add_constraints,);
    }

    #[test]
    fn mod_add_layout_q3329() {
        let layout = mod_add_scratch_count(12);

        assert_eq!(layout.result_width, 12);
        assert_eq!(layout.carry_width, 13);
        assert_eq!(layout.range_result_width, 12);
        assert_eq!(layout.range_borrow_width, 13);

        // 2×12 + 2×13 + 1 + 12 + 13 = 76
        assert_eq!(layout.total_scratch_bits, 76);
    }

    #[test]
    fn mod_add_constraint_count_q3329() {
        let layout = mod_add_scratch_count(12);
        let cs = ConstraintSystem::<F>::new();

        let mut col_idx = 0;

        let mut alloc = |n: usize| -> Vec<Expr<'_, F>> {
            let cols: Vec<_> = (col_idx..col_idx + n).map(|i| cs.col(i)).collect();
            col_idx += n;

            cols
        };

        let a = alloc(12);
        let b = alloc(12);
        let result = alloc(12);
        let lhs_result = alloc(layout.result_width);
        let lhs_carry = alloc(layout.carry_width);
        let rhs_result = alloc(layout.result_width);
        let rhs_carry = alloc(layout.carry_width);
        let flag_col = alloc(1);
        let range_result = alloc(layout.range_result_width);
        let range_borrow = alloc(layout.range_borrow_width);

        mod_add(
            &cs,
            &a,
            &b,
            &result,
            &ModAddWitness {
                lhs_result: &lhs_result,
                lhs_carry: &lhs_carry,
                rhs_result: &rhs_result,
                rhs_carry: &rhs_carry,
                flag: flag_col[0],
                range_result: &range_result,
                range_borrow: &range_borrow,
            },
            3329,
        );

        let ast = cs.build();

        // flag boolean: 1
        // LHS add (12-bit):
        // 4*12 + 1 = 49
        // RHS add (12-bit):
        // 4*12 + 1 = 49
        // Equality:
        // 12 result bits + 1 carry = 13
        // Range check (12-bit):
        // 4*12 + 2 = 50
        // Total:
        // 1 + 49 + 49 + 13 + 50 = 162
        assert_eq!(ast.roots.len(), 162);
    }

    #[test]
    fn bit_packing_constraint_count() {
        let cs = ConstraintSystem::<F>::new();
        let bus = cs.col(0);
        let bits: Vec<_> = (1..=12).map(|i| cs.col(i)).collect();

        bit_packing(&cs, bus, &bits);

        let ast = cs.build();

        // Single degree-1 constraint
        assert_eq!(ast.roots.len(), 1);
    }

    #[test]
    fn schoolbook_mul_layout_8x12() {
        let layout = schoolbook_mul_layout(8, 12);

        assert_eq!(layout.product_width, 20);
        assert_eq!(layout.pp0_width, 8);

        // 11 additions (b_width - 1)
        assert_eq!(layout.carry_widths.len(), 11);

        // 10 intermediate sums (b_width - 2)
        assert_eq!(layout.sum_widths.len(), 10);
    }
}
