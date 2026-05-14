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

//! HighBits/LowBits Chiplet for ML-DSA.
//!
//! Computes unsigned Euclidean division
//! by a constant divisor:
//!   r = r₁ * divisor + r₀
//!   0 ≤ r₀ < divisor
//!
//! For ML-DSA:
//! divisor = 2γ₂ = 523776 (ML-DSA-65),
//! r₁ = HighBits(r),
//! r₀ = LowBits(r) (unsigned, shifted).
//!
//! The control chiplet handles the
//! centered-mod adjustment:
//!   actual_r0 = r₀ - γ₂

use super::utils::{fill_add_carry_packed, fill_sub_borrow_packed, flush_bit_buffer, pack_bits};
use crate::atoms::int_arith::{
    add_carry_chain, mul_const, mul_const_scratch_widths, range_check, MulConstLayout,
};
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder, TraceCompatibleField};
use hekate_math::{Block32, TowerField};
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::constraint::ConstraintAst;
use hekate_program::expander::VirtualExpander;
use hekate_program::permutation::{PermutationCheckSpec, Source};
use hekate_program::Air;

// =================================================================
// Column Layout
// =================================================================

/// Column index map for HighBitsLowBits chiplet.
/// Computed at construction time from
/// `modulus`, `bit_width`, and `divisor`.
#[derive(Clone, Debug)]
pub struct HighBitsLayout {
    pub bit_width: usize,
    pub r1_width: usize,
    pub r0_width: usize,
    pub product_width: usize,

    // Arithmetic Bit columns
    pub r_bits: usize,
    pub r1_bits: usize,
    pub r0_bits: usize,

    // Constant multiply scratch (r₁ * divisor)
    pub quot_x_div: usize,
    pub mul_scratch_r: Vec<(usize, usize)>,
    pub mul_scratch_c: Vec<(usize, usize)>,

    // Addition carry chain (quot_x_div + r₀ = r)
    pub add_carry: usize,
    pub add_carry_width: usize,

    // Range check scratch (r₀ < divisor)
    pub range_result: usize,
    pub range_borrow: usize,

    pub num_bit_cols: usize,
    pub num_packed_b32_cols: usize,
    pub num_expanded_bits: usize,

    // Centering columns
    // (FIPS 204 Algorithm 35)
    pub is_neg: usize,
    pub r1c_bits: usize,
    pub neg_carry: usize,
    pub is_qm1: usize,
    pub neg_range_result: usize,
    pub neg_range_borrow: usize,

    // UseHint bit columns (virtual indices)
    pub w1_bits: usize,
    pub w1_width: usize,
    pub chain: usize,
    pub h_bit: usize,
    pub r0_nonzero: usize,
    pub uh_wrap: usize,

    /// Precomputed direction:
    /// r0>0 AND r0≤γ₂.
    pub s_dir: usize,

    // B32 columns (bus, virtual indices)
    pub bus_r: usize,
    pub bus_r1: usize,
    pub bus_r0: usize,
    pub bus_idx: usize,
    pub bus_h_bit: usize,
    pub bus_w1_prime: usize,

    // Control
    pub s_active: usize,

    pub num_columns: usize,
    pub num_physical_columns: usize,

    // Sub-layout for constraint building
    pub mul_layout: MulConstLayout,
}

impl HighBitsLayout {
    /// Compute the column layout for a
    /// given modulus and divisor.
    pub fn compute(modulus: u32, bit_width: usize, divisor: u32) -> Self {
        let max_r1 = (modulus - 1) / divisor;
        let r1_width = if max_r1 == 0 {
            1
        } else {
            32 - max_r1.leading_zeros() as usize
        };

        let r0_width = 32 - (divisor - 1).leading_zeros() as usize;

        let mul_layout = mul_const_scratch_widths(r1_width, divisor);
        let product_width = mul_layout.result_width;

        let mut offset = 0usize;

        let mut alloc = |n: usize| -> usize {
            let start = offset;
            offset += n;
            start
        };

        // Input value
        let r_bits = alloc(bit_width);

        // Quotient (high bits) and remainder (low bits)
        let r1_bits = alloc(r1_width);
        let r0_bits = alloc(r0_width);

        // Constant multiply:
        // r₁ * divisor
        let quot_x_div = alloc(product_width);

        let mut mul_scratch_r = Vec::with_capacity(mul_layout.scratch_result_widths.len());
        for &w in &mul_layout.scratch_result_widths {
            mul_scratch_r.push((alloc(w), w));
        }

        let mut mul_scratch_c = Vec::with_capacity(mul_layout.scratch_carry_widths.len());
        for &w in &mul_layout.scratch_carry_widths {
            mul_scratch_c.push((alloc(w), w));
        }

        // Addition carry chain:
        // r₁*divisor + r₀ = r
        let add_carry_width = product_width + 1;
        let add_carry = alloc(add_carry_width);

        // Range check:
        // r₀ < divisor
        let range_result = alloc(r0_width);
        let range_borrow = alloc(r0_width + 1);

        // Centering:
        // is_neg=1 when r0_u > gamma2
        let is_neg = alloc(1);
        let r1c_bits = alloc(r1_width);
        let neg_carry = alloc(r1_width.saturating_sub(1));
        let is_qm1 = alloc(1);
        let neg_range_result = alloc(r0_width);
        let neg_range_borrow = alloc(r0_width + 1);

        // UseHint columns
        let w1_width = r1_width;
        let w1_bits = alloc(w1_width);
        let chain = alloc(w1_width.saturating_sub(1));
        let h_bit = alloc(1);
        let r0_nonzero = alloc(1);
        let uh_wrap = alloc(1);
        let s_dir = alloc(1);

        let num_bit_cols = offset;
        let num_packed_b32_cols = num_bit_cols.div_ceil(32);
        let num_expanded_bits = num_packed_b32_cols * 32;

        let bus_r = num_expanded_bits;
        let bus_r1 = num_expanded_bits + 1;
        let bus_r0 = num_expanded_bits + 2;
        let bus_idx = num_expanded_bits + 3;
        let bus_h_bit = num_expanded_bits + 4;
        let bus_w1_prime = num_expanded_bits + 5;
        let s_active = num_expanded_bits + 6;

        let num_columns = num_expanded_bits + 7;
        let num_physical_columns = num_packed_b32_cols + 7;

        HighBitsLayout {
            bit_width,
            r1_width,
            r0_width,
            product_width,
            r_bits,
            r1_bits,
            r0_bits,
            quot_x_div,
            mul_scratch_r,
            mul_scratch_c,
            add_carry,
            add_carry_width,
            range_result,
            range_borrow,
            is_neg,
            r1c_bits,
            neg_carry,
            is_qm1,
            neg_range_result,
            neg_range_borrow,
            w1_bits,
            w1_width,
            chain,
            h_bit,
            r0_nonzero,
            uh_wrap,
            s_dir,
            num_bit_cols,
            num_packed_b32_cols,
            num_expanded_bits,
            bus_r,
            bus_r1,
            bus_r0,
            bus_idx,
            bus_h_bit,
            bus_w1_prime,
            s_active,
            num_columns,
            num_physical_columns,
            mul_layout,
        }
    }

    pub fn build_virtual_layout(&self) -> Vec<ColumnType> {
        let mut layout = Vec::with_capacity(self.num_columns);

        for _ in 0..self.num_expanded_bits {
            layout.push(ColumnType::Bit);
        }

        // bus:
        // r, r1, r0, idx, h_bit, w1_prime
        for _ in 0..6 {
            layout.push(ColumnType::B32);
        }

        layout.push(ColumnType::Bit);

        debug_assert_eq!(layout.len(), self.num_columns);

        layout
    }

    pub fn build_physical_layout(&self) -> Vec<ColumnType> {
        let mut layout = Vec::with_capacity(self.num_physical_columns);

        for _ in 0..self.num_packed_b32_cols {
            layout.push(ColumnType::B32);
        }

        for _ in 0..6 {
            layout.push(ColumnType::B32);
        }

        layout.push(ColumnType::Bit);

        debug_assert_eq!(layout.len(), self.num_physical_columns);

        layout
    }
}

// =================================================================
// HighBitsLowBits Chiplet
// =================================================================

/// HighBits/LowBits Chiplet.
///
/// Parameterized by modulus q and divisor:
/// - q=8380417, divisor=523776 for ML-DSA-65
///
/// Each row computes:
///   r₁ = floor(r / divisor)
///   r₀ = r mod divisor
#[derive(Clone, Debug)]
pub struct HighBitsChiplet {
    pub modulus: u32,
    pub bit_width: usize,
    pub divisor: u32,
    pub num_rows: usize,

    layout: HighBitsLayout,
    expander: VirtualExpander,
}

impl HighBitsChiplet {
    pub const BUS_ID: &'static str = "highbits";

    pub fn new(modulus: u32, divisor: u32, num_rows: usize) -> Self {
        assert!(num_rows.is_power_of_two());
        assert!(divisor > 0 && divisor < modulus);

        let bit_width = 32 - modulus.leading_zeros() as usize;
        let layout = HighBitsLayout::compute(modulus, bit_width, divisor);

        let expander = VirtualExpander::new()
            .expand_bits(layout.num_packed_b32_cols, ColumnType::B32)
            .pass_through(6, ColumnType::B32)
            .control_bits(1)
            .build()
            .expect("HighBitsChiplet expander");

        Self {
            modulus,
            bit_width,
            divisor,
            num_rows,
            layout,
            expander,
        }
    }

    pub fn layout(&self) -> &HighBitsLayout {
        &self.layout
    }

    /// Linking specification for the highbits bus.
    /// Carries (r, r1, r0, index).
    pub fn linking_spec(&self) -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (Source::Column(self.layout.bus_r), b"kappa_hb_r" as &[u8]),
                (Source::Column(self.layout.bus_r1), b"kappa_hb_r1" as &[u8]),
                (Source::Column(self.layout.bus_r0), b"kappa_hb_r0" as &[u8]),
                (
                    Source::Column(self.layout.bus_idx),
                    b"kappa_hb_idx" as &[u8],
                ),
                (
                    Source::Column(self.layout.bus_h_bit),
                    b"kappa_hb_h" as &[u8],
                ),
                (
                    Source::Column(self.layout.bus_w1_prime),
                    b"kappa_hb_w1" as &[u8],
                ),
            ],
            Some(self.layout.s_active),
        )
        .with_clock_waiver(
            "see pqc/high_bits.rs: bus_idx is positional, AIR forces one row per \
             (idx) value; partner mldsa ctrl side carries the matching idx clock",
        )
    }
}

// =================================================================
// Air Implementation
// =================================================================

impl<F: TowerField + TraceCompatibleField> Air<F> for HighBitsChiplet {
    fn name(&self) -> String {
        "HighBitsChiplet".to_string()
    }

    fn column_layout(&self) -> &[ColumnType] {
        let layout = self.layout.build_physical_layout();
        Box::leak(layout.into_boxed_slice())
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![(Self::BUS_ID.into(), self.linking_spec())]
    }

    fn virtual_expander(&self) -> Option<&VirtualExpander> {
        Some(&self.expander)
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        build_highbits_constraints(self.modulus, self.divisor, &self.layout)
    }
}

// =================================================================
// Constraint Generation
// =================================================================

/// Build the constraint AST for one
/// HighBits/LowBits row.
///
/// Proves:
/// r = r₁ * divisor + r₀,
/// with r₀ < divisor.
fn build_highbits_constraints<F: TowerField>(
    modulus: u32,
    divisor: u32,
    ly: &HighBitsLayout,
) -> ConstraintAst<F> {
    let cs = ConstraintSystem::<F>::new();
    let s_active = cs.col(ly.s_active);

    let bit_width = ly.bit_width;
    let r1_width = ly.r1_width;
    let r0_width = ly.r0_width;
    let product_width = ly.product_width;

    // Collect bit expressions
    let r: Vec<_> = (0..bit_width).map(|k| cs.col(ly.r_bits + k)).collect();
    let r1: Vec<_> = (0..r1_width).map(|k| cs.col(ly.r1_bits + k)).collect();
    let r0: Vec<_> = (0..r0_width).map(|k| cs.col(ly.r0_bits + k)).collect();

    // Booleanity (s_active gated)
    for &bit in r.iter().chain(r1.iter()).chain(r0.iter()) {
        cs.assert_zero_when(s_active, bit * (bit + cs.one()));
    }

    // 1. Constant multiply:
    // r₁ * divisor
    let qxd: Vec<_> = (0..product_width)
        .map(|k| cs.col(ly.quot_x_div + k))
        .collect();

    let mul_sr: Vec<Vec<_>> = ly
        .mul_scratch_r
        .iter()
        .map(|&(start, width)| (0..width).map(|k| cs.col(start + k)).collect())
        .collect();
    let mul_sr_refs: Vec<&[_]> = mul_sr.iter().map(|v| v.as_slice()).collect();

    let mul_sc: Vec<Vec<_>> = ly
        .mul_scratch_c
        .iter()
        .map(|&(start, width)| (0..width).map(|k| cs.col(start + k)).collect())
        .collect();
    let mul_sc_refs: Vec<&[_]> = mul_sc.iter().map(|v| v.as_slice()).collect();

    mul_const(&cs, &r1, &qxd, &mul_sr_refs, &mul_sc_refs, divisor);

    // 2. Decomposition:
    //    r₁*divisor + r₀ = r
    //    Both sides padded to product_width.
    let zero = cs.constant(F::ZERO);

    let r0_padded: Vec<_> = (0..product_width)
        .map(|k| if k < r0_width { r0[k] } else { zero })
        .collect();

    let r_padded: Vec<_> = (0..product_width)
        .map(|k| if k < bit_width { r[k] } else { zero })
        .collect();

    let add_c: Vec<_> = (0..ly.add_carry_width)
        .map(|k| cs.col(ly.add_carry + k))
        .collect();

    add_carry_chain(&cs, &qxd, &r0_padded, &r_padded, &add_c);

    // Carry-out must be 0
    cs.constrain(add_c[product_width]);

    // 3. Range check:
    // r₀ < divisor
    let rng_r: Vec<_> = (0..r0_width).map(|k| cs.col(ly.range_result + k)).collect();
    let rng_w: Vec<_> = (0..=r0_width)
        .map(|k| cs.col(ly.range_borrow + k))
        .collect();

    range_check(&cs, &r0, &rng_r, &rng_w, divisor);

    // 4. Centering (FIPS 204 Algorithm 35).
    // Subtraction gamma2 - r0_u:
    // borrow_out = is_neg.
    let gamma2 = divisor / 2;
    let is_neg = cs.col(ly.is_neg);
    let is_qm1 = cs.col(ly.is_qm1);

    cs.assert_zero_when(s_active, is_neg * (is_neg + cs.one()));
    cs.assert_zero_when(s_active, is_qm1 * (is_qm1 + cs.one()));

    let neg_r: Vec<_> = (0..r0_width)
        .map(|k| cs.col(ly.neg_range_result + k))
        .collect();
    let neg_w: Vec<_> = (0..=r0_width)
        .map(|k| cs.col(ly.neg_range_borrow + k))
        .collect();

    cs.constrain(neg_w[0]);

    for i in 0..r0_width {
        let gbit = (gamma2 >> i) & 1;
        let v = r0[i];
        let w = neg_w[i];
        let nr = neg_r[i];
        let w_next = neg_w[i + 1];

        cs.assert_boolean(nr);
        cs.assert_boolean(w_next);

        if gbit == 1 {
            cs.constrain(nr + cs.one() + v + w);
            cs.constrain(w_next + v * w);
        } else {
            cs.constrain(nr + v + w);
            cs.constrain(w_next + v + w + v * w);
        }
    }

    cs.constrain(neg_w[r0_width] + is_neg);

    // r1_centered = r1_u + is_neg (carry chain).
    // is_qm1=1 forces r1c = 0 (Q-1 corner case).
    let r1c: Vec<_> = (0..r1_width).map(|k| cs.col(ly.r1c_bits + k)).collect();
    let nc_w = r1_width.saturating_sub(1);
    let nc: Vec<_> = (0..nc_w).map(|k| cs.col(ly.neg_carry + k)).collect();

    for &bit in r1c.iter().chain(nc.iter()) {
        cs.assert_zero_when(s_active, bit * (bit + cs.one()));
    }

    let not_qm1 = cs.one() + is_qm1;
    cs.constrain(s_active * (r1c[0] + not_qm1 * (r1[0] + is_neg)));

    if r1_width > 1 {
        cs.constrain(s_active * (nc[0] + not_qm1 * r1[0] * is_neg));
        cs.constrain(s_active * (r1c[1] + not_qm1 * (r1[1] + nc[0])));
    }

    for i in 1..nc_w {
        cs.constrain(s_active * (nc[i] + not_qm1 * r1[i] * nc[i - 1]));
        cs.constrain(s_active * (r1c[i + 1] + not_qm1 * (r1[i + 1] + nc[i])));
    }

    // is_qm1=1 => r1+is_neg = m.
    let m = (modulus - 1) / divisor;
    for (k, &r1_k) in r1.iter().enumerate() {
        if (m >> k) & 1 == 1 {
            cs.constrain(is_qm1 * (cs.one() + is_neg) * (cs.one() + r1_k));
        } else {
            cs.constrain(is_qm1 * (cs.one() + is_neg) * r1_k);
        }

        if ((m - 1) >> k) & 1 == 1 {
            cs.constrain(is_qm1 * is_neg * (cs.one() + r1_k));
        } else {
            cs.constrain(is_qm1 * is_neg * r1_k);
        }
    }

    // 5. UseHint (FIPS 204 Algorithm 39).
    // h=0:
    //   w1 = r1c.
    // h=1:
    //   w1 = (r1c ± 1) mod m.
    //   s=1 (increment) when r0_u > 0 AND r0_u ≤ gamma2.
    //   s=0 (decrement) when r0_u = 0 OR r0_u > gamma2.
    //   wrap=1 at modular boundary (r1c=m-1 inc, r1c=0 dec).
    let m1 = m - 1;

    let w1: Vec<_> = (0..ly.w1_width).map(|k| cs.col(ly.w1_bits + k)).collect();
    let chain_width = ly.w1_width.saturating_sub(1);
    let ch: Vec<_> = (0..chain_width).map(|k| cs.col(ly.chain + k)).collect();

    let h = cs.col(ly.h_bit);
    let r0_nz = cs.col(ly.r0_nonzero);
    let wrap = cs.col(ly.uh_wrap);
    let s = cs.col(ly.s_dir);

    let nw = cs.one() + wrap;

    for &bit in w1.iter().chain(ch.iter()).chain([h, r0_nz, wrap, s].iter()) {
        cs.assert_zero_when(s_active, bit * (bit + cs.one()));
    }

    // s_dir = r0_nz AND NOT is_neg
    cs.constrain(s_active * (s + r0_nz * (cs.one() + is_neg)));
    cs.constrain(s_active * wrap * (cs.one() + h));

    // Non-wrap carry chain
    cs.constrain_named("uh_w0", s_active * nw * (w1[0] + r1c[0] + h));

    if ly.w1_width > 1 {
        cs.constrain_named("uh_c0", s_active * nw * h * (ch[0] + cs.one() + r1c[0] + s));
        cs.constrain_named("uh_w1", s_active * nw * (w1[1] + r1c[1] + h * ch[0]));
    }

    let last_ch = chain_width.saturating_sub(1);
    for i in 1..last_ch {
        cs.constrain_named(
            "uh_ch",
            s_active * nw * h * (ch[i] + ch[i - 1] * (cs.one() + r1c[i] + s)),
        );
        cs.constrain_named(
            "uh_wb",
            s_active * nw * (w1[i + 1] + r1c[i + 1] + h * ch[i]),
        );
    }

    if chain_width > 1 {
        cs.constrain_named(
            "uh_ch",
            s_active * nw * h * (ch[last_ch] + ch[last_ch - 1] * (cs.one() + r1c[last_ch] + s)),
        );
    }

    if ly.w1_width > 1 {
        cs.constrain_named(
            "uh_top",
            s_active * nw * (w1[ly.w1_width - 1] + r1c[ly.w1_width - 1] + h * ch[last_ch]),
        );
    }

    // Wrap branch:
    // w1=0 (inc) or m-1 (dec).
    // Soundness:
    // r1c=m-1 (inc) or r1c=0 (dec).
    for i in 0..ly.w1_width {
        let m1_bit = (m1 >> i) & 1;
        if m1_bit == 1 {
            cs.constrain(s_active * wrap * (w1[i] + cs.one() + s));
            cs.constrain(s_active * wrap * s * (r1c[i] + cs.one()));
        } else {
            cs.constrain(s_active * wrap * w1[i]);
            cs.constrain(s_active * wrap * s * r1c[i]);
        }

        cs.constrain(s_active * wrap * (cs.one() + s) * r1c[i]);
    }

    // 6. Bus column linkage.
    // Tower basis:
    // B32(x) promoted to F = Σ x_k · F(1 << k).
    // Forces bus pass-through columns to match
    // the constrained bit decomposition.
    let mut w1_link = cs.col(ly.bus_w1_prime);
    for (k, &w1_k) in w1.iter().enumerate() {
        w1_link = w1_link + w1_k * cs.constant(F::from(1u128 << k));
    }

    cs.constrain(s_active * w1_link);

    let mut h_link = cs.col(ly.bus_h_bit);
    h_link = h_link + h * cs.constant(F::from(1u128));

    cs.constrain(s_active * h_link);

    // Unused packed bits must be zero
    for k in ly.num_bit_cols..ly.num_expanded_bits {
        cs.constrain(cs.col(k));
    }

    cs.build()
}

// =================================================================
// Trace Generation
// =================================================================

/// A single HighBits/LowBits operation.
#[derive(Clone, Debug)]
pub struct HighBitsOp {
    /// Input value mod q.
    pub r: u32,

    /// Operation index
    /// (for bus linking).
    pub idx: u32,

    /// RAM address where
    /// w_approx was written.
    pub ram_addr: u32,

    /// Hint bit from signature.
    pub h_bit: bool,

    /// UseHint result.
    pub w1_prime: u32,
}

/// Generate the HighBits/LowBits trace.
///
/// `modulus`: prime q
/// `divisor`: the divisor (2γ₂)
/// `ops`: list of decompose operations
/// `num_rows`: trace height (power of 2)
pub fn generate_highbits_trace(
    modulus: u32,
    divisor: u32,
    ops: &[HighBitsOp],
    num_rows: usize,
) -> hekate_core::errors::Result<ColumnTrace> {
    assert!(num_rows.is_power_of_two());
    assert!(ops.len() <= num_rows);

    let bit_width = 32 - modulus.leading_zeros() as usize;
    let layout = HighBitsLayout::compute(modulus, bit_width, divisor);
    let physical = layout.build_physical_layout();

    let num_vars = num_rows.trailing_zeros() as usize;
    let num_packed = layout.num_packed_b32_cols;

    let mut tb = TraceBuilder::new(&physical, num_vars)?;

    let phy_bus_r = num_packed;
    let phy_bus_r1 = num_packed + 1;
    let phy_bus_r0 = num_packed + 2;
    let phy_bus_idx = num_packed + 3;
    let phy_bus_h_bit = num_packed + 4;
    let phy_bus_w1 = num_packed + 5;
    let phy_s_active = num_packed + 6;

    let mut bits = vec![0u32; num_packed];

    for (row, op) in ops.iter().enumerate() {
        debug_assert!(op.r < modulus);

        let r = op.r;
        let r1 = r / divisor;
        let r0 = r % divisor;

        bits.iter_mut().for_each(|w| *w = 0);

        pack_bits(&mut bits, layout.r_bits, r as u64, bit_width);
        pack_bits(&mut bits, layout.r1_bits, r1 as u64, layout.r1_width);
        pack_bits(&mut bits, layout.r0_bits, r0 as u64, layout.r0_width);

        // r1 * divisor
        let product = (r1 as u64) * (divisor as u64);
        fill_mul_const_packed(&mut bits, &layout, r1, divisor);

        // r1*divisor + r0 = r
        fill_add_carry_packed(
            &mut bits,
            layout.add_carry,
            layout.add_carry_width,
            product,
            r0 as u64,
        );

        // r0 < divisor
        fill_sub_borrow_packed(
            &mut bits,
            layout.range_result,
            layout.range_borrow,
            layout.r0_width,
            (divisor - 1) as u64,
            r0 as u64,
        );

        // Centering witness
        let gamma2 = divisor / 2;
        let neg = if r0 > gamma2 { 1u32 } else { 0u32 };

        let m = (modulus - 1) / divisor;
        let qm1 = if r1 + neg == m { 1u32 } else { 0u32 };

        pack_bits(&mut bits, layout.is_neg, neg as u64, 1);
        pack_bits(&mut bits, layout.is_qm1, qm1 as u64, 1);

        fill_sub_borrow_packed(
            &mut bits,
            layout.neg_range_result,
            layout.neg_range_borrow,
            layout.r0_width,
            gamma2 as u64,
            r0 as u64,
        );

        // r1_centered = r1 + neg, or 0 if qm1
        let r1c = if qm1 == 1 { 0u32 } else { r1 + neg };
        pack_bits(&mut bits, layout.r1c_bits, r1c as u64, layout.r1_width);

        if layout.r1_width > 1 {
            let mut carry = r1 & 1 & neg;
            for k in 0..(layout.r1_width - 1) {
                let c_val = if qm1 == 1 { 0u32 } else { carry };
                pack_bits(&mut bits, layout.neg_carry + k, c_val as u64, 1);

                if k + 1 < layout.r1_width - 1 {
                    carry &= (r1 >> (k + 1)) & 1;
                }
            }
        }

        // UseHint witness
        let h_val = op.h_bit as u64;
        let r0_nz = if r0 > 0 { 1u64 } else { 0u64 };
        let s_dir = if r0 > 0 && neg == 0 { 1u64 } else { 0u64 };

        let m = (modulus - 1) / divisor;
        let uh_wrap = if op.h_bit {
            let inc = s_dir == 1;
            if (inc && r1c == m - 1) || (!inc && r1c == 0) {
                1u64
            } else {
                0u64
            }
        } else {
            0u64
        };

        pack_bits(
            &mut bits,
            layout.w1_bits,
            op.w1_prime as u64,
            layout.w1_width,
        );
        pack_bits(&mut bits, layout.h_bit, h_val, 1);
        pack_bits(&mut bits, layout.r0_nonzero, r0_nz, 1);
        pack_bits(&mut bits, layout.uh_wrap, uh_wrap, 1);
        pack_bits(&mut bits, layout.s_dir, s_dir, 1);

        // Carry/borrow chain witness
        if layout.w1_width > 1 {
            let chain_width = layout.w1_width - 1;

            let mut prev = if s_dir == 1 { r1c & 1 } else { 1 ^ (r1c & 1) };
            for k in 0..chain_width {
                pack_bits(&mut bits, layout.chain + k, prev as u64, 1);

                if k + 1 < chain_width {
                    let next_bit = (r1c >> (k + 1)) & 1;
                    let sel = if s_dir == 1 { next_bit } else { 1 ^ next_bit };

                    prev &= sel;
                }
            }
        }

        flush_bit_buffer(&bits, &mut tb, row)?;

        tb.set_b32(phy_bus_r, row, Block32::from(r))?;
        tb.set_b32(phy_bus_r1, row, Block32::from(r1))?;
        tb.set_b32(phy_bus_r0, row, Block32::from(r0))?;
        tb.set_b32(phy_bus_idx, row, Block32::from(op.idx))?;
        tb.set_b32(phy_bus_h_bit, row, Block32::from(h_val as u32))?;
        tb.set_b32(phy_bus_w1, row, Block32::from(op.w1_prime))?;
    }

    // Padding rows:
    // fill range check and add-carry
    // witnesses so constraints with
    // constant bits evaluate to zero
    // on all-zero operands.
    for pad_row in ops.len()..num_rows {
        bits.iter_mut().for_each(|w| *w = 0);

        fill_padding_witnesses(&mut bits, &layout, divisor);
        flush_bit_buffer(&bits, &mut tb, pad_row)?;
    }

    tb.fill_selector(phy_s_active, ops.len())?;

    Ok(tb.build())
}

/// Fill the constant multiply witness
/// for r₁ * divisor.
///
/// Cascaded addition tree of shifted
/// partial products into packed buffer.
fn fill_mul_const_packed(bits: &mut [u32], ly: &HighBitsLayout, operand: u32, constant: u32) {
    let set_bits: Vec<usize> = (0..32).filter(|&i| (constant >> i) & 1 == 1).collect();
    let m = set_bits.len();

    if m <= 1 {
        let shift = if m == 1 { set_bits[0] } else { 0 };
        let shifted = (operand as u64) << shift;
        pack_bits(bits, ly.quot_x_div, shifted, ly.product_width);
        return;
    }

    let shifted_vals: Vec<u64> = set_bits.iter().map(|&s| (operand as u64) << s).collect();

    // shifted[0] + shifted[1]
    let mut acc = shifted_vals[0] + shifted_vals[1];

    if m == 2 {
        pack_bits(bits, ly.quot_x_div, acc, ly.product_width);
    } else {
        let (col, width) = ly.mul_scratch_r[0];
        pack_bits(bits, col, acc, width);
    }

    fill_add_carry_packed(
        bits,
        ly.mul_scratch_c[0].0,
        ly.mul_scratch_c[0].1,
        shifted_vals[0],
        shifted_vals[1],
    );

    for (j, &shifted_j) in shifted_vals.iter().enumerate().skip(2) {
        let prev_acc = acc;
        acc = prev_acc + shifted_j;

        if j == m - 1 {
            pack_bits(bits, ly.quot_x_div, acc, ly.product_width);
        } else {
            let (col, width) = ly.mul_scratch_r[j - 1];
            pack_bits(bits, col, acc, width);
        }

        fill_add_carry_packed(
            bits,
            ly.mul_scratch_c[j - 1].0,
            ly.mul_scratch_c[j - 1].1,
            prev_acc,
            shifted_j,
        );
    }
}

/// Fill range-check witness on padding rows.
///
/// With r=0, r1=0, r0=0:
/// range_check needs result = divisor-1
/// (the subtraction (divisor-1) - 0).
/// mul_const and add_carry are zero-on-zero.
fn fill_padding_witnesses(bits: &mut [u32], ly: &HighBitsLayout, divisor: u32) {
    let dm1 = (divisor - 1) as u64;
    pack_bits(bits, ly.range_result, dm1, ly.r0_width);

    // Centering:
    // gamma2 - 0 = gamma2 (no borrow).
    let gamma2 = (divisor / 2) as u64;
    pack_bits(bits, ly.neg_range_result, gamma2, ly.r0_width);
}

// =================================================================
// Tests
// =================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use hekate_core::trace::Trace;
    use hekate_math::{Bit, Block128};

    type F = Block128;

    const Q: u32 = 8380417;
    const BIT_WIDTH: usize = 23;

    // ML-DSA-65:
    // γ₂ = (q-1)/32 = 261888, 2γ₂ = 523776
    const DIVISOR: u32 = 523776;

    fn read_packed_bit(trace: &ColumnTrace, virt_col: usize, row: usize) -> bool {
        let packed_col = virt_col / 32;
        let bit_idx = virt_col % 32;
        let word = trace.columns[packed_col].as_b32_slice().unwrap()[row]
            .to_tower()
            .0;

        (word >> bit_idx) & 1 == 1
    }

    fn read_packed_value(trace: &ColumnTrace, virt_start: usize, width: usize, row: usize) -> u32 {
        let mut val = 0u32;
        for k in 0..width {
            if read_packed_bit(trace, virt_start + k, row) {
                val |= 1 << k;
            }
        }

        val
    }

    fn phy_sel(ly: &HighBitsLayout) -> usize {
        ly.num_packed_b32_cols + 6
    }

    fn test_op(r: u32, idx: u32) -> HighBitsOp {
        let r1 = r / DIVISOR;
        let r0 = r % DIVISOR;
        let gamma2 = DIVISOR / 2;
        let is_neg = r0 > gamma2;
        let m = (Q - 1) / DIVISOR;
        let r1c = if r1 + is_neg as u32 == m {
            0
        } else {
            r1 + is_neg as u32
        };

        HighBitsOp {
            r,
            idx,
            ram_addr: 0,
            h_bit: false,
            w1_prime: r1c,
        }
    }

    #[test]
    fn layout_dimensions() {
        let ly = HighBitsLayout::compute(Q, BIT_WIDTH, DIVISOR);

        // r1:
        // max_r1 = 8380416/523776 = 16 → 5 bits
        assert_eq!(ly.r1_width, 5);

        // r0:
        // divisor-1 = 523775 < 2^19 → 19 bits
        assert_eq!(ly.r0_width, 19);

        // product_width from mul_const (5-bit × 523776)
        assert!(ly.product_width >= BIT_WIDTH);

        assert!(ly.num_packed_b32_cols > 0);
        assert_eq!(ly.num_expanded_bits, ly.num_packed_b32_cols * 32);
        assert_eq!(ly.num_physical_columns, ly.num_packed_b32_cols + 7);
    }

    #[test]
    fn constraint_ast_builds() {
        let chiplet = HighBitsChiplet::new(Q, DIVISOR, 1024);
        let ast = Air::<F>::constraint_ast(&chiplet);

        // Non-trivial constraint count
        assert!(ast.roots.len() > 50);
    }

    #[test]
    fn air_declares_one_bus() {
        let chiplet = HighBitsChiplet::new(Q, DIVISOR, 1024);
        let checks = Air::<F>::permutation_checks(&chiplet);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].0, "highbits");
    }

    #[test]
    fn trace_simple_decomposition() {
        // r = 1000000, divisor = 523776
        // r1 = 1000000 / 523776 = 1
        // r0 = 1000000 % 523776 = 476224
        let ops = vec![test_op(1_000_000, 0)];

        let trace = generate_highbits_trace(Q, DIVISOR, &ops, 4).unwrap();
        let ly = HighBitsLayout::compute(Q, BIT_WIDTH, DIVISOR);

        assert_eq!(trace.num_rows().unwrap(), 4);

        // Verify r1 bits encode 1
        let r1_val = read_packed_value(&trace, ly.r1_bits, ly.r1_width, 0);
        assert_eq!(r1_val, 1);

        // Verify r0 bits encode 476224
        let r0_val = read_packed_value(&trace, ly.r0_bits, ly.r0_width, 0);
        assert_eq!(r0_val, 476224);

        // Verify decomposition
        assert_eq!(r1_val as u64 * 523776 + r0_val as u64, 1_000_000);
    }

    #[test]
    fn trace_zero_value() {
        // r = 0 → r1 = 0, r0 = 0
        let ops = vec![test_op(0, 0)];
        let trace = generate_highbits_trace(Q, DIVISOR, &ops, 4).unwrap();
        let ly = HighBitsLayout::compute(Q, BIT_WIDTH, DIVISOR);

        let sel = trace.columns[phy_sel(&ly)].as_bit_slice().unwrap();
        assert_eq!(sel[0], Bit::ONE);
    }

    #[test]
    fn trace_max_value() {
        // r = q-1 = 8380416
        // r1 = 8380416 / 523776 = 16
        // r0 = 8380416 % 523776 = 0
        // (16 * 523776 = 8380416)
        let ops = vec![test_op(Q - 1, 0)];
        let trace = generate_highbits_trace(Q, DIVISOR, &ops, 4).unwrap();
        let ly = HighBitsLayout::compute(Q, BIT_WIDTH, DIVISOR);

        // Verify r1 = 16
        let r1_val = read_packed_value(&trace, ly.r1_bits, ly.r1_width, 0);
        assert_eq!(r1_val, 16);

        // Verify r0 = 0
        let r0_val = read_packed_value(&trace, ly.r0_bits, ly.r0_width, 0);
        assert_eq!(r0_val, 0);
    }

    #[test]
    fn trace_divisor_minus_one() {
        // r = 523775 (divisor - 1)
        // r1 = 0, r0 = 523775
        let ops = vec![test_op(DIVISOR - 1, 0)];

        let trace = generate_highbits_trace(Q, DIVISOR, &ops, 4).unwrap();
        let ly = HighBitsLayout::compute(Q, BIT_WIDTH, DIVISOR);

        let r1_val = read_packed_value(&trace, ly.r1_bits, ly.r1_width, 0);
        assert_eq!(r1_val, 0);

        let r0_val = read_packed_value(&trace, ly.r0_bits, ly.r0_width, 0);
        assert_eq!(r0_val, DIVISOR - 1);
    }

    #[test]
    fn trace_with_padding() {
        let ops = vec![test_op(42, 0)];
        let trace = generate_highbits_trace(Q, DIVISOR, &ops, 8).unwrap();
        let ly = HighBitsLayout::compute(Q, BIT_WIDTH, DIVISOR);

        let sel = trace.columns[phy_sel(&ly)].as_bit_slice().unwrap();
        assert_eq!(sel[0], Bit::ONE);

        for &s in &sel[1..8] {
            assert_eq!(s, Bit::ZERO);
        }
    }

    #[test]
    fn bus_labels() {
        let chiplet = HighBitsChiplet::new(Q, DIVISOR, 1024);
        let spec = chiplet.linking_spec();

        assert_eq!(spec.sources.len(), 6);
        assert_eq!(spec.sources[0].1, b"kappa_hb_r");
        assert_eq!(spec.sources[1].1, b"kappa_hb_r1");
        assert_eq!(spec.sources[2].1, b"kappa_hb_r0");
        assert_eq!(spec.sources[3].1, b"kappa_hb_idx");
        assert_eq!(spec.sources[4].1, b"kappa_hb_h");
        assert_eq!(spec.sources[5].1, b"kappa_hb_w1");
    }

    #[test]
    fn multiple_operations() {
        let ops: Vec<HighBitsOp> = (0..8)
            .map(|i| test_op((i as u32) * 600_000, i as u32))
            .collect();

        let trace = generate_highbits_trace(Q, DIVISOR, &ops, 8).unwrap();
        let ly = HighBitsLayout::compute(Q, BIT_WIDTH, DIVISOR);

        // Verify all rows have correct decomposition
        for (row_idx, op) in ops.iter().enumerate() {
            let expected_r1 = op.r / DIVISOR;
            let expected_r0 = op.r % DIVISOR;

            let r1_val = read_packed_value(&trace, ly.r1_bits, ly.r1_width, row_idx);
            assert_eq!(r1_val, expected_r1, "row {} r1 mismatch", row_idx);

            let r0_val = read_packed_value(&trace, ly.r0_bits, ly.r0_width, row_idx);
            assert_eq!(r0_val, expected_r0, "row {} r0 mismatch", row_idx);
        }
    }

    #[test]
    fn packed_trace_roundtrip() {
        let ops = vec![
            test_op(1_000_000, 0),
            test_op(Q - 1, 1),
            test_op(DIVISOR - 1, 2),
        ];

        let trace = generate_highbits_trace(Q, DIVISOR, &ops, 4).unwrap();
        let ly = HighBitsLayout::compute(Q, BIT_WIDTH, DIVISOR);

        assert_eq!(trace.columns.len(), ly.num_physical_columns);

        let chiplet = HighBitsChiplet::new(Q, DIVISOR, 4);
        let variants = Air::<F>::virtual_expander(&chiplet)
            .unwrap()
            .expand_variants::<F, _>(&trace, 0)
            .unwrap();

        assert_eq!(variants.len(), ly.num_columns);

        // Tail padding bits must be zero
        for k in ly.num_bit_cols..ly.num_expanded_bits {
            for row in 0..4 {
                assert!(!read_packed_bit(&trace, k, row));
            }
        }
    }

    #[test]
    fn usehint_increment() {
        // r = 1_000_000, r1 = 1,
        // r0 = 476224 (> 0)
        // h=true, r0>0 -> w1 = (1+1)%16 = 2
        let ops = vec![HighBitsOp {
            r: 1_000_000,
            idx: 0,
            ram_addr: 0,
            h_bit: true,
            w1_prime: 2,
        }];
        let trace = generate_highbits_trace(Q, DIVISOR, &ops, 4).unwrap();
        let ly = HighBitsLayout::compute(Q, BIT_WIDTH, DIVISOR);

        let w1_val = read_packed_value(&trace, ly.w1_bits, ly.w1_width, 0);
        assert_eq!(w1_val, 2);

        let h_val = read_packed_bit(&trace, ly.h_bit, 0);
        assert!(h_val);

        let r0_nz = read_packed_bit(&trace, ly.r0_nonzero, 0);
        assert!(r0_nz);
    }

    #[test]
    fn usehint_decrement() {
        // r = Q-1 = 8380416, r1 = 16, r0 = 0
        // h=true, r0=0 -> w1 = (16-1)%16 = 15
        let ops = vec![HighBitsOp {
            r: Q - 1,
            idx: 0,
            ram_addr: 0,
            h_bit: true,
            w1_prime: 15,
        }];
        let trace = generate_highbits_trace(Q, DIVISOR, &ops, 4).unwrap();
        let ly = HighBitsLayout::compute(Q, BIT_WIDTH, DIVISOR);

        let w1_val = read_packed_value(&trace, ly.w1_bits, ly.w1_width, 0);
        assert_eq!(w1_val, 15);

        let r0_nz = read_packed_bit(&trace, ly.r0_nonzero, 0);
        assert!(!r0_nz);
    }
}
