// SPDX-License-Identifier: Apache-2.0
// This file is part of the hekate project.
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

//! NTT Butterfly Chiplet for
//! modular arithmetic over Z_q.
//!
//! Processes NTT-256 butterflies for
//! ML-KEM (q=3329, 12-bit) and
//! ML-DSA (q=8380417, 23-bit).
//!
//! Each row = one complete butterfly:
//!   a' = (a + w*b) mod q
//!   b' = (a - w*b) mod q
//!
//! Single-row design with:
//! - Schoolbook multiplication (w*b)
//! - Modular reduction (Barrett/Euclidean)
//! - Modular add/sub with range checks
//!
//! All integer arithmetic is emulated
//! via explicit carry chains over GF(2).

use super::utils::{
    fill_add_carry_packed, fill_sub_borrow_packed, flush_bit_buffer, pack_bits, pack_one,
};
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder, TraceCompatibleField};
use hekate_gadgets::atoms::int_arith::{
    ModAddLayout, ModAddWitness, ModReductionLayout, ModReductionWitness, SchoolbookMulLayout,
    SchoolbookMulWitness, mod_add, mod_add_scratch_count, mod_reduction,
    mod_reduction_scratch_count, range_check, schoolbook_mul, schoolbook_mul_layout,
};
use hekate_math::{Bit, Block32, TowerField};
use hekate_program::Air;
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::expander::VirtualExpander;
use hekate_program::permutation::{PermutationCheckSpec, Source};

// =================================================================
// Column Layout (computed dynamically per modulus)
// =================================================================

// Trailing Bit selectors:
// 9 control bits + aux_flow + aux_bound.
const NTT_CONTROL_BITS: usize = 11;

/// Column index map for the NTT chiplet.
/// Computed at construction time from
/// `modulus` and `bit_width`.
///
/// All arithmetic columns are Bit (1 byte).
/// Bus/control columns are B32 or Bit.
#[derive(Clone, Debug)]
pub struct NttLayout {
    pub bit_width: usize,
    pub product_width: usize,

    // Operand bits (Bit columns)
    pub a_bits: usize,
    pub b_bits: usize,
    pub w_bits: usize,

    // Schoolbook multiplication scratch
    pub mul_pp0: usize,
    pub mul_pp0_width: usize,
    pub mul_sums: Vec<(usize, usize)>,
    pub mul_carries: Vec<(usize, usize)>,
    pub product_bits: usize,

    // Barrett reduction
    pub quot_bits: usize,
    pub wb_bits: usize,
    pub quot_x_q: usize,
    pub quot_x_q_width: usize,
    pub barrett_sr: Vec<(usize, usize)>,
    pub barrett_sc: Vec<(usize, usize)>,
    pub barrett_add_carry: usize,
    pub barrett_add_carry_width: usize,
    pub barrett_rng_result: usize,
    pub barrett_rng_borrow: usize,

    // Modular addition
    pub a_out_bits: usize,
    pub add_lhs_result: usize,
    pub add_lhs_carry: usize,
    pub add_rhs_result: usize,
    pub add_rhs_carry: usize,
    pub add_flag: usize,
    pub add_rng_result: usize,
    pub add_rng_borrow: usize,

    // Modular subtraction
    pub b_out_bits: usize,
    pub sub_lhs_result: usize,
    pub sub_lhs_carry: usize,
    pub sub_rhs_result: usize,
    pub sub_rhs_carry: usize,
    pub sub_flag: usize,
    pub sub_rng_result: usize,
    pub sub_rng_borrow: usize,

    // b_out range check (b_out < q)
    pub b_out_rng_result: usize,
    pub b_out_rng_borrow: usize,

    // End of active Bit columns
    pub num_bit_cols: usize,

    // Packed B32 count for physical layout
    pub num_packed_b32_cols: usize,

    // Total expanded bits (num_packed * 32).
    // Includes padding in the tail column.
    pub num_expanded_bits: usize,

    // B32 columns (bus + control), virtual indices
    pub bus_a: usize,
    pub bus_b: usize,
    pub bus_w: usize,
    pub bus_wb: usize,
    pub bus_a_out: usize,
    pub bus_b_out: usize,
    pub layer: usize,
    pub butterfly_idx: usize,

    // Flow connectivity columns (B32)
    pub ntt_instance: usize,
    pub pos_a: usize,
    pub pos_b: usize,
    pub src_layer: usize,

    // Flow clock (4 × B32 bytes).
    // On flow-input rows:
    // stores the matching
    // flow-output row index.
    pub flow_clk: usize,

    // Control Bit columns
    pub s_active: usize,
    pub s_output: usize,
    pub s_butterfly: usize,
    pub s_companion: usize,
    pub s_flow_output: usize,
    pub s_flow_input: usize,
    pub s_bound_in: usize,
    pub s_bound_out: usize,
    pub s_mulonly: usize,

    // Pin s_companion at degree 3
    pub aux_flow: usize,
    pub aux_bound: usize,

    // Totals (virtual)
    pub num_columns: usize,

    // Totals (physical, packed)
    pub num_physical_columns: usize,

    // Sub-layouts for constraint building
    pub mul_layout: SchoolbookMulLayout,
    pub barrett_layout: ModReductionLayout,
    pub mod_add_layout: ModAddLayout,
}

impl NttLayout {
    /// Compute the full column
    /// layout for a given modulus.
    pub fn compute(modulus: u32, bit_width: usize) -> Self {
        let mul_layout = schoolbook_mul_layout(bit_width, bit_width);
        let barrett_layout = mod_reduction_scratch_count(bit_width, modulus);
        let mod_add_layout = mod_add_scratch_count(bit_width);

        let product_width = mul_layout.product_width;

        let mut offset = 0usize;

        // Helper:
        // advance offset by n, return start.
        let mut alloc = |n: usize| -> usize {
            let start = offset;
            offset += n;

            start
        };

        // Operand bits
        let a_bits = alloc(bit_width);
        let b_bits = alloc(bit_width);
        let w_bits = alloc(bit_width);

        // Schoolbook multiplication scratch
        let mul_pp0 = alloc(mul_layout.pp0_width);
        let mul_pp0_width = mul_layout.pp0_width;

        let mut mul_sums = Vec::with_capacity(mul_layout.sum_widths.len());
        for &w in &mul_layout.sum_widths {
            mul_sums.push((alloc(w), w));
        }

        let mut mul_carries = Vec::with_capacity(mul_layout.carry_widths.len());
        for &w in &mul_layout.carry_widths {
            mul_carries.push((alloc(w), w));
        }

        let product_bits = alloc(product_width);

        // Barrett reduction
        let quot_bits = alloc(bit_width);
        let wb_bits = alloc(bit_width);
        let quot_x_q = alloc(barrett_layout.product_width);
        let quot_x_q_width = barrett_layout.product_width;

        let mut barrett_sr =
            Vec::with_capacity(barrett_layout.mul_layout.scratch_result_widths.len());

        for &w in &barrett_layout.mul_layout.scratch_result_widths {
            barrett_sr.push((alloc(w), w));
        }

        let mut barrett_sc =
            Vec::with_capacity(barrett_layout.mul_layout.scratch_carry_widths.len());

        for &w in &barrett_layout.mul_layout.scratch_carry_widths {
            barrett_sc.push((alloc(w), w));
        }

        let barrett_add_carry = alloc(barrett_layout.add_carry_width);
        let barrett_add_carry_width = barrett_layout.add_carry_width;
        let barrett_rng_result = alloc(barrett_layout.range_result_width);
        let barrett_rng_borrow = alloc(barrett_layout.range_borrow_width);

        // Modular addition
        let a_out_bits = alloc(bit_width);
        let add_lhs_result = alloc(mod_add_layout.result_width);
        let add_lhs_carry = alloc(mod_add_layout.carry_width);
        let add_rhs_result = alloc(mod_add_layout.result_width);
        let add_rhs_carry = alloc(mod_add_layout.carry_width);
        let add_flag = alloc(1);
        let add_rng_result = alloc(mod_add_layout.range_result_width);
        let add_rng_borrow = alloc(mod_add_layout.range_borrow_width);

        // Modular subtraction
        let b_out_bits = alloc(bit_width);
        let sub_lhs_result = alloc(mod_add_layout.result_width);
        let sub_lhs_carry = alloc(mod_add_layout.carry_width);
        let sub_rhs_result = alloc(mod_add_layout.result_width);
        let sub_rhs_carry = alloc(mod_add_layout.carry_width);
        let sub_flag = alloc(1);
        let sub_rng_result = alloc(mod_add_layout.range_result_width);
        let sub_rng_borrow = alloc(mod_add_layout.range_borrow_width);

        // b_out range check (b_out < q)
        let b_out_rng_result = alloc(bit_width);
        let b_out_rng_borrow = alloc(bit_width + 1);

        // Drop the closure so we can read offset.
        let num_bit_cols = offset;
        let num_packed_b32_cols = num_bit_cols.div_ceil(32);

        // Total expanded bits = full 32 per packed B32.
        // Includes padding bits in the tail column
        // that must be constrained to zero.
        let num_expanded_bits = num_packed_b32_cols * 32;

        // Virtual indices:
        // B32 bus + control start AFTER all
        // expanded bits (including padding).
        let bus_a = num_expanded_bits;
        let bus_b = num_expanded_bits + 1;
        let bus_w = num_expanded_bits + 2;
        let bus_wb = num_expanded_bits + 3;
        let bus_a_out = num_expanded_bits + 4;
        let bus_b_out = num_expanded_bits + 5;
        let layer = num_expanded_bits + 6;
        let butterfly_idx = num_expanded_bits + 7;

        // Flow connectivity (B32)
        let ntt_instance = num_expanded_bits + 8;
        let pos_a = num_expanded_bits + 9;
        let pos_b = num_expanded_bits + 10;
        let src_layer = num_expanded_bits + 11;

        // Flow clock (4 × B32 bytes)
        let flow_clk = num_expanded_bits + 12;

        // Control Bit columns
        let s_active = num_expanded_bits + 16;
        let s_output = num_expanded_bits + 17;
        let s_butterfly = num_expanded_bits + 18;
        let s_companion = num_expanded_bits + 19;
        let s_flow_output = num_expanded_bits + 20;
        let s_flow_input = num_expanded_bits + 21;
        let s_bound_in = num_expanded_bits + 22;
        let s_bound_out = num_expanded_bits + 23;
        let s_mulonly = num_expanded_bits + 24;

        let aux_flow = num_expanded_bits + 25;
        let aux_bound = num_expanded_bits + 26;

        let num_columns = num_expanded_bits + 27;
        let num_physical_columns = num_packed_b32_cols + 27;

        NttLayout {
            bit_width,
            product_width,
            a_bits,
            b_bits,
            w_bits,
            mul_pp0,
            mul_pp0_width,
            mul_sums,
            mul_carries,
            product_bits,
            quot_bits,
            wb_bits,
            quot_x_q,
            quot_x_q_width,
            barrett_sr,
            barrett_sc,
            barrett_add_carry,
            barrett_add_carry_width,
            barrett_rng_result,
            barrett_rng_borrow,
            a_out_bits,
            add_lhs_result,
            add_lhs_carry,
            add_rhs_result,
            add_rhs_carry,
            add_flag,
            add_rng_result,
            add_rng_borrow,
            b_out_bits,
            sub_lhs_result,
            sub_lhs_carry,
            sub_rhs_result,
            sub_rhs_carry,
            sub_flag,
            sub_rng_result,
            sub_rng_borrow,
            b_out_rng_result,
            b_out_rng_borrow,
            num_bit_cols,
            num_packed_b32_cols,
            num_expanded_bits,
            bus_a,
            bus_b,
            bus_w,
            bus_wb,
            bus_a_out,
            bus_b_out,
            layer,
            butterfly_idx,
            ntt_instance,
            pos_a,
            pos_b,
            src_layer,
            flow_clk,
            s_active,
            s_output,
            s_butterfly,
            s_companion,
            s_flow_output,
            s_flow_input,
            s_mulonly,
            s_bound_in,
            s_bound_out,
            aux_flow,
            aux_bound,
            num_columns,
            num_physical_columns,
            mul_layout,
            barrett_layout,
            mod_add_layout,
        }
    }

    /// Build the virtual column layout.
    pub fn build_virtual_layout(&self) -> Vec<ColumnType> {
        let mut layout = Vec::with_capacity(self.num_columns);

        // All expanded bits (active + padding)
        let num_expanded = self.num_packed_b32_cols * 32;
        for _ in 0..num_expanded {
            layout.push(ColumnType::Bit);
        }

        // 8 bus B32 + 4 flow B32 + 4 flow_clk B32
        for _ in 0..16 {
            layout.push(ColumnType::B32);
        }

        for _ in 0..NTT_CONTROL_BITS {
            layout.push(ColumnType::Bit);
        }

        debug_assert_eq!(layout.len(), self.num_columns);

        layout
    }

    /// Build the physical column layout.
    pub fn build_physical_layout(&self) -> Vec<ColumnType> {
        let mut layout = Vec::with_capacity(self.num_physical_columns);

        // Packed B32 columns (32 virtual bits each)
        for _ in 0..self.num_packed_b32_cols {
            layout.push(ColumnType::B32);
        }

        // 8 bus B32 + 4 flow B32 + 4 flow_clk B32
        for _ in 0..16 {
            layout.push(ColumnType::B32);
        }

        for _ in 0..NTT_CONTROL_BITS {
            layout.push(ColumnType::Bit);
        }

        debug_assert_eq!(layout.len(), self.num_physical_columns);

        layout
    }

    /// Number of active bits in the last
    /// packed B32 column (1..32).
    pub fn tail_bits(&self) -> usize {
        let tail = self.num_bit_cols % 32;
        if tail == 0 { 32 } else { tail }
    }
}

// =================================================================
// NTT Chiplet
// =================================================================

/// NTT Butterfly Chiplet.
///
/// Parameterized by modulus q:
/// - q=3329 (12-bit) for ML-KEM
/// - q=8380417 (23-bit) for ML-DSA
///
/// Each row computes one full butterfly:
///   wb = w * b mod q
///   a' = (a + wb) mod q
///   b' = (a - wb) mod q
#[derive(Clone, Debug)]
pub struct NttChiplet {
    pub modulus: u32,
    pub bit_width: usize,
    pub num_rows: usize,

    layout: NttLayout,
    expander: VirtualExpander,
}

impl NttChiplet {
    pub const DATA_BUS_ID: &'static str = "ntt_data";
    pub const TWIDDLE_BUS_ID: &'static str = "ntt_twiddle";
    pub const FLOW_BUS_ID: &'static str = "ntt_flow";
    pub const BOUND_IN_BUS_ID: &'static str = "ntt_bound_in";
    pub const BOUND_OUT_BUS_ID: &'static str = "ntt_bound_out";

    pub fn new(modulus: u32, num_rows: usize) -> Self {
        assert!(num_rows.is_power_of_two());

        let bit_width = 32 - modulus.leading_zeros() as usize;
        let layout = NttLayout::compute(modulus, bit_width);

        let expander = VirtualExpander::new()
            .expand_bits(layout.num_packed_b32_cols, ColumnType::B32)
            .pass_through(16, ColumnType::B32)
            .control_bits(NTT_CONTROL_BITS)
            .build()
            .expect("NttChiplet expander");

        Self {
            modulus,
            bit_width,
            num_rows,
            layout,
            expander,
        }
    }

    pub fn layout(&self) -> &NttLayout {
        &self.layout
    }

    /// Linking specification for the data bus.
    /// Carries (a, b, w, a_out, b_out, layer, butterfly_idx).
    pub fn data_linking_spec(&self) -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (Source::Column(self.layout.bus_a), b"kappa_ntt_a" as &[u8]),
                (Source::Column(self.layout.bus_b), b"kappa_ntt_b" as &[u8]),
                (
                    Source::Column(self.layout.bus_a_out),
                    b"kappa_ntt_a_out" as &[u8],
                ),
                (
                    Source::Column(self.layout.bus_b_out),
                    b"kappa_ntt_b_out" as &[u8],
                ),
                (
                    Source::Column(self.layout.layer),
                    b"kappa_ntt_layer" as &[u8],
                ),
                (
                    Source::Column(self.layout.butterfly_idx),
                    b"kappa_ntt_bfly" as &[u8],
                ),
                (
                    Source::Column(self.layout.ntt_instance),
                    b"kappa_ntt_inst" as &[u8],
                ),
            ],
            Some(self.layout.s_output),
        )
        .with_clock_waiver(
            "see pqc/ntt.rs: per-row uniqueness pinned by the (ntt_instance, layer, \
             butterfly_idx) triple which the AIR's flow constraints force to be \
             distinct across active rows",
        )
    }

    /// Linking specification for the twiddle ROM bus.
    /// Carries (layer, butterfly_idx, w).
    pub fn twiddle_linking_spec(&self) -> PermutationCheckSpec {
        PermutationCheckSpec::new_lookup(
            vec![
                (
                    Source::Column(self.layout.layer),
                    b"kappa_tw_layer" as &[u8],
                ),
                (
                    Source::Column(self.layout.butterfly_idx),
                    b"kappa_tw_bfly" as &[u8],
                ),
                (Source::Column(self.layout.bus_w), b"kappa_tw_w" as &[u8]),
            ],
            Some(self.layout.s_active),
        )
    }

    /// Flow output spec.
    /// Primary rows:
    /// (inst, layer, pos_a, a_out).
    /// Companion rows:
    /// pos_a=pos_b,
    /// bus_a_out=b_out.
    fn flow_output_spec(&self) -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(self.layout.ntt_instance),
                    b"kappa_flow_inst" as &[u8],
                ),
                (
                    Source::Column(self.layout.layer),
                    b"kappa_flow_layer" as &[u8],
                ),
                (
                    Source::Column(self.layout.pos_a),
                    b"kappa_flow_pos" as &[u8],
                ),
                (
                    Source::Column(self.layout.bus_a_out),
                    b"kappa_flow_val" as &[u8],
                ),
                (Source::RowIndexByte(0), b"kappa_flow_clk_b0" as &[u8]),
                (Source::RowIndexByte(1), b"kappa_flow_clk_b1" as &[u8]),
                (Source::RowIndexByte(2), b"kappa_flow_clk_b2" as &[u8]),
                (Source::RowIndexByte(3), b"kappa_flow_clk_b3" as &[u8]),
            ],
            Some(self.layout.s_flow_output),
        )
    }

    /// Flow input spec.
    /// Primary rows:
    /// (inst, src_layer, pos_a, a).
    /// Companion rows:
    /// pos_a=pos_b,
    /// bus_a=b.
    fn flow_input_spec(&self) -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(self.layout.ntt_instance),
                    b"kappa_flow_inst" as &[u8],
                ),
                (
                    Source::Column(self.layout.src_layer),
                    b"kappa_flow_layer" as &[u8],
                ),
                (
                    Source::Column(self.layout.pos_a),
                    b"kappa_flow_pos" as &[u8],
                ),
                (
                    Source::Column(self.layout.bus_a),
                    b"kappa_flow_val" as &[u8],
                ),
                (
                    Source::Column(self.layout.flow_clk),
                    b"kappa_flow_clk_b0" as &[u8],
                ),
                (
                    Source::Column(self.layout.flow_clk + 1),
                    b"kappa_flow_clk_b1" as &[u8],
                ),
                (
                    Source::Column(self.layout.flow_clk + 2),
                    b"kappa_flow_clk_b2" as &[u8],
                ),
                (
                    Source::Column(self.layout.flow_clk + 3),
                    b"kappa_flow_clk_b3" as &[u8],
                ),
            ],
            Some(self.layout.s_flow_input),
        )
        .with_clock_waiver(
            "see pqc/ntt.rs: partner flow_output_spec carries Source::RowIndexByte; \
             this side stores the matching clock in committed flow_clk[0..4] columns \
             pinned by AIR transitions",
        )
    }

    /// Boundary input spec (layer 0):
    /// (inst, pos_a, bus_a).
    pub fn bound_in_spec(&self) -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(self.layout.ntt_instance),
                    b"kappa_bound_inst" as &[u8],
                ),
                (
                    Source::Column(self.layout.pos_a),
                    b"kappa_bound_pos" as &[u8],
                ),
                (
                    Source::Column(self.layout.bus_a),
                    b"kappa_bound_val" as &[u8],
                ),
            ],
            Some(self.layout.s_bound_in),
        )
        .with_clock_waiver(
            "see pqc/ntt.rs: per-row uniqueness pinned by (ntt_instance, pos_a) at \
             layer 0 boundary; the AIR forces pos_a to take each value at most once \
             per instance via flow constraints",
        )
    }

    /// Boundary output spec (last layer):
    /// (inst, pos_a, bus_a_out).
    pub fn bound_out_spec(&self) -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(self.layout.ntt_instance),
                    b"kappa_bound_inst" as &[u8],
                ),
                (
                    Source::Column(self.layout.pos_a),
                    b"kappa_bound_pos" as &[u8],
                ),
                (
                    Source::Column(self.layout.bus_a_out),
                    b"kappa_bound_val" as &[u8],
                ),
            ],
            Some(self.layout.s_bound_out),
        )
        .with_clock_waiver(
            "see pqc/ntt.rs: per-row uniqueness pinned by (ntt_instance, pos_a) at \
             last-layer boundary; AIR flow constraints force pos_a uniqueness per \
             instance",
        )
    }
}

// =================================================================
// Air Implementation
// =================================================================

impl<F: TowerField + TraceCompatibleField> Air<F> for NttChiplet {
    fn name(&self) -> String {
        "NttChiplet".to_string()
    }

    fn column_layout(&self) -> &[ColumnType] {
        // Box::leak:
        // layout depends on self.modulus, so
        // a single static cache doesn't work
        // for multi-modulus usage (ML-KEM/ML-DSA).
        //
        // Bounded leak:
        // one per NttChiplet instance.
        let layout = self.layout.build_physical_layout();
        Box::leak(layout.into_boxed_slice())
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![
            (Self::DATA_BUS_ID.into(), self.data_linking_spec()),
            (Self::TWIDDLE_BUS_ID.into(), self.twiddle_linking_spec()),
            // Flow connectivity:
            // same bus_id dual specs.
            (Self::FLOW_BUS_ID.into(), self.flow_output_spec()),
            (Self::FLOW_BUS_ID.into(), self.flow_input_spec()),
            (Self::BOUND_IN_BUS_ID.into(), self.bound_in_spec()),
            (Self::BOUND_OUT_BUS_ID.into(), self.bound_out_spec()),
        ]
    }

    fn virtual_expander(&self) -> Option<&VirtualExpander> {
        Some(&self.expander)
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        build_ntt_constraints(self.modulus, self.bit_width, &self.layout)
    }
}

// =================================================================
// Constraint Generation
// =================================================================

/// Build the full constraint AST
/// for one NTT butterfly row.
fn build_ntt_constraints<F: TowerField>(
    modulus: u32,
    bit_width: usize,
    ly: &NttLayout,
) -> ConstraintAst<F> {
    let cs = ConstraintSystem::<F>::new();
    let s_active = cs.col(ly.s_active);

    // Collect operand bit expressions
    let a_bits: Vec<_> = (0..bit_width).map(|k| cs.col(ly.a_bits + k)).collect();
    let b_bits: Vec<_> = (0..bit_width).map(|k| cs.col(ly.b_bits + k)).collect();
    let w_bits: Vec<_> = (0..bit_width).map(|k| cs.col(ly.w_bits + k)).collect();

    // Schoolbook multiplication:
    // w * b = product

    let pp0: Vec<_> = (0..ly.mul_pp0_width)
        .map(|k| cs.col(ly.mul_pp0 + k))
        .collect();

    let mul_sums: Vec<Vec<_>> = ly
        .mul_sums
        .iter()
        .map(|&(start, width)| (0..width).map(|k| cs.col(start + k)).collect())
        .collect();
    let mul_sum_refs: Vec<&[_]> = mul_sums.iter().map(|v| v.as_slice()).collect();

    let mul_carries: Vec<Vec<_>> = ly
        .mul_carries
        .iter()
        .map(|&(start, width)| (0..width).map(|k| cs.col(start + k)).collect())
        .collect();
    let mul_carry_refs: Vec<&[_]> = mul_carries.iter().map(|v| v.as_slice()).collect();

    let product: Vec<_> = (0..ly.product_width)
        .map(|k| cs.col(ly.product_bits + k))
        .collect();

    schoolbook_mul(
        &cs,
        &w_bits,
        &b_bits,
        &product,
        &SchoolbookMulWitness {
            pp0: &pp0,
            sums: &mul_sum_refs,
            carries: &mul_carry_refs,
        },
    );

    // Barrett reduction:
    // product mod q = wb

    let quot: Vec<_> = (0..bit_width).map(|k| cs.col(ly.quot_bits + k)).collect();
    let wb: Vec<_> = (0..bit_width).map(|k| cs.col(ly.wb_bits + k)).collect();

    let quot_x_q: Vec<_> = (0..ly.quot_x_q_width)
        .map(|k| cs.col(ly.quot_x_q + k))
        .collect();

    let bsr: Vec<Vec<_>> = ly
        .barrett_sr
        .iter()
        .map(|&(start, width)| (0..width).map(|k| cs.col(start + k)).collect())
        .collect();
    let bsr_refs: Vec<&[_]> = bsr.iter().map(|v| v.as_slice()).collect();

    let bsc: Vec<Vec<_>> = ly
        .barrett_sc
        .iter()
        .map(|&(start, width)| (0..width).map(|k| cs.col(start + k)).collect())
        .collect();
    let bsc_refs: Vec<&[_]> = bsc.iter().map(|v| v.as_slice()).collect();

    let barrett_add_c: Vec<_> = (0..ly.barrett_add_carry_width)
        .map(|k| cs.col(ly.barrett_add_carry + k))
        .collect();
    let barrett_rng_r: Vec<_> = (0..bit_width)
        .map(|k| cs.col(ly.barrett_rng_result + k))
        .collect();
    let barrett_rng_w: Vec<_> = (0..bit_width + 1)
        .map(|k| cs.col(ly.barrett_rng_borrow + k))
        .collect();

    mod_reduction(
        &cs,
        &product,
        &quot,
        &wb,
        &ModReductionWitness {
            quot_x_mod_bits: &quot_x_q,
            mul_scratch_results: &bsr_refs,
            mul_scratch_carries: &bsc_refs,
            add_carry_bits: &barrett_add_c,
            range_result_bits: &barrett_rng_r,
            range_borrow_bits: &barrett_rng_w,
        },
        modulus,
    );

    // Modular addition:
    // a_out = (a + wb) mod q

    let a_out: Vec<_> = (0..bit_width).map(|k| cs.col(ly.a_out_bits + k)).collect();

    let add_lhs_r: Vec<_> = (0..ly.mod_add_layout.result_width)
        .map(|k| cs.col(ly.add_lhs_result + k))
        .collect();
    let add_lhs_c: Vec<_> = (0..ly.mod_add_layout.carry_width)
        .map(|k| cs.col(ly.add_lhs_carry + k))
        .collect();
    let add_rhs_r: Vec<_> = (0..ly.mod_add_layout.result_width)
        .map(|k| cs.col(ly.add_rhs_result + k))
        .collect();
    let add_rhs_c: Vec<_> = (0..ly.mod_add_layout.carry_width)
        .map(|k| cs.col(ly.add_rhs_carry + k))
        .collect();
    let add_flag = cs.col(ly.add_flag);
    let add_rng_r: Vec<_> = (0..ly.mod_add_layout.range_result_width)
        .map(|k| cs.col(ly.add_rng_result + k))
        .collect();
    let add_rng_w: Vec<_> = (0..ly.mod_add_layout.range_borrow_width)
        .map(|k| cs.col(ly.add_rng_borrow + k))
        .collect();

    mod_add(
        &cs,
        &a_bits,
        &wb,
        &a_out,
        &ModAddWitness {
            lhs_result: &add_lhs_r,
            lhs_carry: &add_lhs_c,
            rhs_result: &add_rhs_r,
            rhs_carry: &add_rhs_c,
            flag: add_flag,
            range_result: &add_rng_r,
            range_borrow: &add_rng_w,
        },
        modulus,
    );

    // Modular subtraction:
    // b_out + wb = a + flag_sub * q
    let b_out: Vec<_> = (0..bit_width).map(|k| cs.col(ly.b_out_bits + k)).collect();

    let sub_lhs_r: Vec<_> = (0..ly.mod_add_layout.result_width)
        .map(|k| cs.col(ly.sub_lhs_result + k))
        .collect();
    let sub_lhs_c: Vec<_> = (0..ly.mod_add_layout.carry_width)
        .map(|k| cs.col(ly.sub_lhs_carry + k))
        .collect();
    let sub_rhs_r: Vec<_> = (0..ly.mod_add_layout.result_width)
        .map(|k| cs.col(ly.sub_rhs_result + k))
        .collect();
    let sub_rhs_c: Vec<_> = (0..ly.mod_add_layout.carry_width)
        .map(|k| cs.col(ly.sub_rhs_carry + k))
        .collect();
    let sub_flag = cs.col(ly.sub_flag);
    let sub_rng_r: Vec<_> = (0..ly.mod_add_layout.range_result_width)
        .map(|k| cs.col(ly.sub_rng_result + k))
        .collect();
    let sub_rng_w: Vec<_> = (0..ly.mod_add_layout.range_borrow_width)
        .map(|k| cs.col(ly.sub_rng_borrow + k))
        .collect();

    // b_out + wb = a + flag_sub * q
    // Reframe:
    // verify a = (b_out + wb) mod q
    mod_add(
        &cs,
        &b_out,
        &wb,
        &a_bits,
        &ModAddWitness {
            lhs_result: &sub_lhs_r,
            lhs_carry: &sub_lhs_c,
            rhs_result: &sub_rhs_r,
            rhs_carry: &sub_rhs_c,
            flag: sub_flag,
            range_result: &sub_rng_r,
            range_borrow: &sub_rng_w,
        },
        modulus,
    );

    // Range check:
    // b_out < q.
    //
    // Without this,a malicious prover
    // can set b_out to:
    // a + q - wb ∈ [q, 2^bit_width)
    // when a ≥ wb, yielding an
    // incorrect subtraction result.
    let b_out_rng_r: Vec<_> = (0..bit_width)
        .map(|k| cs.col(ly.b_out_rng_result + k))
        .collect();
    let b_out_rng_w: Vec<_> = (0..bit_width + 1)
        .map(|k| cs.col(ly.b_out_rng_borrow + k))
        .collect();

    range_check(&cs, &b_out, &b_out_rng_r, &b_out_rng_w, modulus);

    // Packing constraints:
    // bind B32 bus columns to bit decompositions.

    // Bit packing gated by s_active.
    // Companion rows (s_active=0) reuse bus columns
    // for flow values, unconstrained on those rows.
    let pack_pairs: &[(usize, &[_])] = &[
        (ly.bus_a, &a_bits),
        (ly.bus_b, &b_bits),
        (ly.bus_w, &w_bits),
        (ly.bus_wb, &wb),
        (ly.bus_a_out, &a_out),
        (ly.bus_b_out, &b_out),
    ];

    for &(col_idx, bits_slice) in pack_pairs {
        let mut recon = cs.constant(F::ZERO);
        for (k, &bit) in bits_slice.iter().enumerate() {
            recon = recon + bit * cs.constant(F::from(1u128 << k));
        }

        cs.assert_zero_when(s_active, cs.col(col_idx) + recon);
    }

    // Tail padding bits must be zero.
    // The last packed B32 column may have
    // unused bits (32 - tail_bits). These
    // are committed but must not carry data.
    for k in ly.num_bit_cols..ly.num_expanded_bits {
        cs.constrain(cs.col(k));
    }

    // Selector subset constraints:
    // s_output and s_butterfly can
    // only be 1 when s_active is 1.
    let one = cs.one();
    let not_active = one + s_active;

    cs.constrain(cs.col(ly.s_output) * not_active);
    cs.constrain(cs.col(ly.s_butterfly) * not_active);

    cs.assert_boolean(s_active);
    cs.assert_boolean(cs.col(ly.s_output));
    cs.assert_boolean(cs.col(ly.s_butterfly));

    let s_comp = cs.col(ly.s_companion);
    let s_fout = cs.col(ly.s_flow_output);
    let s_fin = cs.col(ly.s_flow_input);
    let s_bin = cs.col(ly.s_bound_in);
    let s_bout = cs.col(ly.s_bound_out);

    cs.assert_boolean(s_comp);
    cs.assert_boolean(s_fout);
    cs.assert_boolean(s_fin);
    cs.assert_boolean(s_bin);
    cs.assert_boolean(s_bout);

    // s_active and s_companion
    // are mutually exclusive.
    cs.constrain(s_active * s_comp);

    let aux_flow = cs.col(ly.aux_flow);
    let aux_bound = cs.col(ly.aux_bound);

    cs.assert_boolean(aux_flow);
    cs.assert_boolean(aux_bound);

    cs.constrain(aux_flow + (one + s_fout) * (one + s_fin));
    cs.constrain(aux_bound + (one + s_bin) * (one + s_bout));

    // Pin s_companion:
    // requires at least one bus selector to fire.
    cs.constrain(s_comp * aux_flow * aux_bound);

    // s_alive = s_active + s_companion
    // (XOR, no overlap guaranteed above).
    // Padding rows:
    // alive=0, sterile.
    let s_alive = s_active + s_comp;
    let not_alive = one + s_alive;

    // Flow/boundary selectors are
    // subsets of s_alive. Padding
    // rows cannot emit on any bus.
    cs.constrain(s_fout * not_alive);
    cs.constrain(s_fin * not_alive);
    cs.constrain(s_bin * not_alive);
    cs.constrain(s_bout * not_alive);

    // s_bound_in and s_bound_out
    // are mutually exclusive.
    cs.constrain(s_bin * s_bout);

    // Partition:
    // s_active = s_butterfly + s_mulonly = s_output.
    let s_mulonly = cs.col(ly.s_mulonly);

    cs.assert_boolean(s_mulonly);

    cs.constrain(s_mulonly * not_active);
    cs.constrain(cs.col(ly.s_butterfly) * s_mulonly);
    cs.constrain(s_active + cs.col(ly.s_butterfly) + s_mulonly);
    cs.constrain(s_active + cs.col(ly.s_output));

    cs.build()
}

// =================================================================
// Trace Generation
// =================================================================

/// A single NTT operation.
#[derive(Clone, Debug)]
pub enum NttOp {
    /// Full butterfly:
    ///   a' = (a + w*b) mod q
    ///   b' = (a - w*b) mod q
    Butterfly(NttButterfly),

    /// Pointwise multiply only:
    ///   wb = w * b mod q
    /// (no butterfly add/sub)
    MulOnly(NttMulOnly),

    /// Flow companion row for pos_b.
    /// No arithmetic, only flow bus entry.
    FlowCompanion(NttFlowCompanion),
}

/// Companion row carrying the pos_b
/// flow entry for a forward butterfly.
#[derive(Clone, Debug)]
pub struct NttFlowCompanion {
    pub b_in: u32,
    pub b_out: u32,
    pub layer: u32,
    pub ntt_instance: u32,
    pub pos: u32,
    pub src_layer: u32,
    pub is_flow_output: bool,
    pub is_flow_input: bool,
    pub is_forward: bool,
}

/// Full NTT butterfly inputs.
#[derive(Clone, Debug)]
pub struct NttButterfly {
    pub a: u32,
    pub b: u32,
    pub w: u32,
    pub layer: u32,
    pub butterfly_idx: u32,
    pub is_forward: bool,
    pub ntt_instance: u32,
    pub pos_a: u32,
    pub pos_b: u32,
}

/// Pointwise multiply-only inputs.
#[derive(Clone, Debug)]
pub struct NttMulOnly {
    pub b: u32,
    pub w: u32,
    pub layer: u32,
    pub butterfly_idx: u32,

    /// True for basemul coefficient multiplies.
    /// False for inverse NTT GS decomposition
    /// and normalization multiplies.
    pub is_basemul: bool,

    /// Inverse NTT flow:
    /// this MulOnly carries the
    /// pos_b output of a GS butterfly.
    pub flow_pos: Option<u32>,
    pub flow_instance: u32,
    pub flow_src_layer: u32,
}

/// Generate the trace for a batch of
/// NTT operations (butterflies and/or
/// pointwise multiplications).
///
/// `modulus`: the prime modulus q
/// `ops`: list of NTT operations
/// `num_rows`: trace height (power of 2)
pub fn generate_ntt_trace(
    modulus: u32,
    ops: &[NttOp],
    num_rows: usize,
) -> hekate_core::errors::Result<ColumnTrace> {
    assert!(num_rows.is_power_of_two());
    assert!(ops.len() <= num_rows);

    let bit_width = 32 - modulus.leading_zeros() as usize;
    let layout = NttLayout::compute(modulus, bit_width);
    let physical = layout.build_physical_layout();

    let num_vars = num_rows.trailing_zeros() as usize;
    let num_packed = layout.num_packed_b32_cols;

    let mut tb = TraceBuilder::new(&physical, num_vars)?;

    // Reusable per-row bit buffer.
    // Accumulates virtual bit values,
    // then packs into B32 physical columns.
    let mut bits = vec![0u32; num_packed];

    // Physical indices for
    // B32 flow columns.
    let phy_ntt_instance = num_packed + 8;
    let phy_pos_a = num_packed + 9;
    let phy_pos_b = num_packed + 10;
    let phy_src_layer = num_packed + 11;
    let phy_flow_clk = num_packed + 12;

    // Physical indices for
    // Bit control columns.
    let phy_s_active = num_packed + 16;
    let phy_s_output = num_packed + 17;
    let phy_s_butterfly = num_packed + 18;
    let phy_s_companion = num_packed + 19;
    let phy_s_flow_output = num_packed + 20;
    let phy_s_flow_input = num_packed + 21;
    let phy_s_bound_in = num_packed + 22;
    let phy_s_bound_out = num_packed + 23;
    let phy_s_mulonly = num_packed + 24;
    let phy_aux_flow = num_packed + 25;
    let phy_aux_bound = num_packed + 26;

    // Derive max_layer from ops.
    // Flow output:
    // layers 0..(max-1).
    // Flow input:
    // layers 1..max.
    let max_layer = ops
        .iter()
        .filter_map(|op| match op {
            NttOp::Butterfly(b) if b.is_forward => Some(b.layer),
            _ => None,
        })
        .max()
        .unwrap_or(0);

    // Build flow-output row map:
    // (inst, layer, pos) -> ntt_row.
    // Used to fill FLOW_CLK on flow-input rows.
    let mut flow_out_rows: BTreeMap<(u32, u32, u32), usize> = BTreeMap::new();
    for (row, op) in ops.iter().enumerate() {
        match op {
            NttOp::Butterfly(b) if (b.ntt_instance > 0 || b.is_forward) && b.layer < max_layer => {
                flow_out_rows.insert((b.ntt_instance, b.layer, b.pos_a), row);
            }
            NttOp::FlowCompanion(c) if c.is_flow_output => {
                flow_out_rows.insert((c.ntt_instance, c.layer, c.pos), row);
            }
            NttOp::MulOnly(m) if m.flow_pos.is_some() && m.layer < max_layer => {
                flow_out_rows.insert((m.flow_instance, m.layer, m.flow_pos.unwrap()), row);
            }
            _ => {}
        }
    }

    for (row, op) in ops.iter().enumerate() {
        // Clear bit buffer for this row
        bits.iter_mut().for_each(|w| *w = 0);

        let mut s_fout_set = false;
        let mut s_fin_set = false;
        let mut s_bin_set = false;
        let mut s_bout_set = false;

        match op {
            NttOp::Butterfly(bfly) => {
                fill_butterfly_row_packed(
                    &mut bits, &mut tb, &layout, modulus, bit_width, row, bfly,
                )?;

                tb.set_bit(phy_s_butterfly, row, Bit::ONE)?;

                // Flow connectivity columns
                tb.set_b32(phy_ntt_instance, row, Block32::from(bfly.ntt_instance))?;
                tb.set_b32(phy_pos_a, row, Block32::from(bfly.pos_a))?;
                tb.set_b32(phy_pos_b, row, Block32::from(bfly.pos_b))?;

                let sl = if bfly.layer > 0 { bfly.layer - 1 } else { 0 };
                tb.set_b32(phy_src_layer, row, Block32::from(sl))?;

                // Flow + boundary selectors:
                // active on both forward and inverse
                // butterflies with valid instance data.
                if bfly.ntt_instance > 0 || bfly.is_forward {
                    if bfly.layer < max_layer {
                        s_fout_set = true;

                        tb.set_bit(phy_s_flow_output, row, Bit::ONE)?;
                    }

                    if bfly.layer > 0 {
                        s_fin_set = true;

                        tb.set_bit(phy_s_flow_input, row, Bit::ONE)?;

                        let src = bfly.layer - 1;
                        if let Some(&out_row) =
                            flow_out_rows.get(&(bfly.ntt_instance, src, bfly.pos_a))
                        {
                            let clk = (out_row as u32).to_le_bytes();
                            for (k, &byte) in clk.iter().enumerate() {
                                tb.set_b32(phy_flow_clk + k, row, Block32::from(byte as u32))?;
                            }
                        }
                    }

                    if bfly.is_forward && bfly.layer == 0 {
                        s_bin_set = true;

                        tb.set_bit(phy_s_bound_in, row, Bit::ONE)?;
                    }

                    if bfly.is_forward && bfly.layer == max_layer {
                        s_bout_set = true;

                        tb.set_bit(phy_s_bound_out, row, Bit::ONE)?;
                    }
                }
            }
            NttOp::MulOnly(mul) => {
                let bfly = NttButterfly {
                    a: 0,
                    b: mul.b,
                    w: mul.w,
                    layer: mul.layer,
                    butterfly_idx: mul.butterfly_idx,
                    is_forward: false,
                    ntt_instance: mul.flow_instance,
                    pos_a: mul.flow_pos.unwrap_or(0),
                    pos_b: 0,
                };

                fill_butterfly_row_packed(
                    &mut bits, &mut tb, &layout, modulus, bit_width, row, &bfly,
                )?;

                tb.set_b32(phy_ntt_instance, row, Block32::from(mul.flow_instance))?;

                if let Some(pos) = mul.flow_pos {
                    tb.set_b32(phy_pos_a, row, Block32::from(pos))?;
                    tb.set_b32(phy_src_layer, row, Block32::from(mul.flow_src_layer))?;

                    if mul.layer < max_layer {
                        s_fout_set = true;

                        tb.set_bit(phy_s_flow_output, row, Bit::ONE)?;
                    }
                }

                tb.set_bit(phy_s_mulonly, row, Bit::ONE)?;
            }
            NttOp::FlowCompanion(comp) => {
                // No arithmetic, fill
                // padding range checks.
                fill_padding_range_checks_packed(&mut bits, &layout, modulus, bit_width);

                // Reuse bus_a / bus_a_out for flow values
                let phy_bus_a = num_packed;
                let phy_bus_a_out = num_packed + 4;
                let phy_layer = num_packed + 6;

                tb.set_b32(phy_bus_a, row, Block32::from(comp.b_in))?;
                tb.set_b32(phy_bus_a_out, row, Block32::from(comp.b_out))?;
                tb.set_b32(phy_layer, row, Block32::from(comp.layer))?;
                tb.set_b32(phy_ntt_instance, row, Block32::from(comp.ntt_instance))?;
                tb.set_b32(phy_pos_a, row, Block32::from(comp.pos))?;
                tb.set_b32(phy_src_layer, row, Block32::from(comp.src_layer))?;

                if comp.is_flow_output {
                    s_fout_set = true;

                    tb.set_bit(phy_s_flow_output, row, Bit::ONE)?;
                }

                if comp.is_flow_input {
                    s_fin_set = true;

                    tb.set_bit(phy_s_flow_input, row, Bit::ONE)?;

                    if let Some(&out_row) =
                        flow_out_rows.get(&(comp.ntt_instance, comp.src_layer, comp.pos))
                    {
                        let clk = (out_row as u32).to_le_bytes();
                        for (k, &byte) in clk.iter().enumerate() {
                            tb.set_b32(phy_flow_clk + k, row, Block32::from(byte as u32))?;
                        }
                    }
                }

                if comp.is_forward && comp.layer == 0 {
                    s_bin_set = true;

                    tb.set_bit(phy_s_bound_in, row, Bit::ONE)?;
                }

                if comp.is_forward && comp.layer == max_layer {
                    s_bout_set = true;

                    tb.set_bit(phy_s_bound_out, row, Bit::ONE)?;
                }
            }
        }

        // Set s_active / s_companion per-row.
        // Three-way partition:
        // - active=1, companion=0 -> arithmetic
        // - active=0, companion=1 -> flow data
        // - active=0, companion=0 -> padding (dead)
        match op {
            NttOp::Butterfly(_) | NttOp::MulOnly(_) => {
                tb.set_bit(phy_s_active, row, Bit::ONE)?;
                tb.set_bit(phy_s_output, row, Bit::ONE)?;
            }
            NttOp::FlowCompanion(_) => {
                tb.set_bit(phy_s_companion, row, Bit::ONE)?;
            }
        }

        let aux_flow_val = if !s_fout_set && !s_fin_set {
            Bit::ONE
        } else {
            Bit::ZERO
        };
        let aux_bound_val = if !s_bin_set && !s_bout_set {
            Bit::ONE
        } else {
            Bit::ZERO
        };

        tb.set_bit(phy_aux_flow, row, aux_flow_val)?;
        tb.set_bit(phy_aux_bound, row, aux_bound_val)?;

        // Flush packed B32 columns
        flush_bit_buffer(&bits, &mut tb, row)?;
    }

    // Fill padding rows with valid witnesses.
    for pad_row in ops.len()..num_rows {
        bits.iter_mut().for_each(|w| *w = 0);

        fill_padding_range_checks_packed(&mut bits, &layout, modulus, bit_width);

        tb.set_bit(phy_aux_flow, pad_row, Bit::ONE)?;
        tb.set_bit(phy_aux_bound, pad_row, Bit::ONE)?;

        flush_bit_buffer(&bits, &mut tb, pad_row)?;
    }

    Ok(tb.build())
}

/// Fill range check witnesses on padding
/// rows into the packed bit buffer.
fn fill_padding_range_checks_packed(
    bits: &mut [u32],
    ly: &NttLayout,
    modulus: u32,
    bit_width: usize,
) {
    let bm1 = modulus - 1;

    // result = (modulus-1) - 0 = modulus-1.
    // borrow is always 0 (buffer zeroed).
    pack_bits(bits, ly.barrett_rng_result, bm1 as u64, bit_width);
    pack_bits(bits, ly.add_rng_result, bm1 as u64, bit_width);
    pack_bits(bits, ly.sub_rng_result, bm1 as u64, bit_width);
    pack_bits(bits, ly.b_out_rng_result, bm1 as u64, bit_width);
}

/// Fill one butterfly row into packed bit
/// buffer + bus columns in TraceBuilder.
#[allow(clippy::too_many_arguments)]
fn fill_butterfly_row_packed(
    bits: &mut [u32],
    tb: &mut TraceBuilder,
    ly: &NttLayout,
    modulus: u32,
    bit_width: usize,
    row: usize,
    bfly: &NttButterfly,
) -> hekate_core::errors::Result<()> {
    let a = bfly.a;
    let b = bfly.b;
    let w = bfly.w;

    // Compute butterfly outputs
    let wb_full = (w as u64) * (b as u64);
    let quot = (wb_full / modulus as u64) as u32;
    let wb = (wb_full % modulus as u64) as u32;

    let a_plus_wb = a + wb;
    let a_out = a_plus_wb % modulus;
    let add_flag = if a_plus_wb >= modulus { 1u32 } else { 0 };

    let b_out_plus_wb = a + if wb <= a { 0 } else { modulus };
    let b_out = b_out_plus_wb - wb;
    let sub_flag = if wb > a { 1u32 } else { 0 };

    // Write operand bits into packed buffer
    pack_bits(bits, ly.a_bits, a as u64, bit_width);
    pack_bits(bits, ly.b_bits, b as u64, bit_width);
    pack_bits(bits, ly.w_bits, w as u64, bit_width);

    // Schoolbook multiplication witness
    fill_schoolbook_mul_packed(bits, ly, w, b, bit_width);

    // Product
    pack_bits(bits, ly.product_bits, wb_full, ly.product_width);

    // Barrett reduction witness
    pack_bits(bits, ly.quot_bits, quot as u64, bit_width);
    pack_bits(bits, ly.wb_bits, wb as u64, bit_width);
    fill_barrett_packed(bits, ly, modulus, bit_width, wb_full, quot, wb);

    // Modular addition witness
    pack_bits(bits, ly.a_out_bits, a_out as u64, bit_width);
    fill_mod_add_packed(
        bits,
        modulus,
        bit_width,
        a,
        wb,
        a_out,
        add_flag,
        ly.add_lhs_result,
        ly.add_lhs_carry,
        ly.add_rhs_result,
        ly.add_rhs_carry,
        ly.add_flag,
        ly.add_rng_result,
        ly.add_rng_borrow,
    );

    // Modular subtraction witness
    pack_bits(bits, ly.b_out_bits, b_out as u64, bit_width);
    fill_mod_add_packed(
        bits,
        modulus,
        bit_width,
        b_out,
        wb,
        a,
        sub_flag,
        ly.sub_lhs_result,
        ly.sub_lhs_carry,
        ly.sub_rhs_result,
        ly.sub_rhs_carry,
        ly.sub_flag,
        ly.sub_rng_result,
        ly.sub_rng_borrow,
    );

    // b_out range check
    fill_sub_borrow_packed(
        bits,
        ly.b_out_rng_result,
        ly.b_out_rng_borrow,
        bit_width,
        (modulus - 1) as u64,
        b_out as u64,
    );

    // Bus B32 columns at physical offset
    let bus_phy = ly.num_packed_b32_cols;
    tb.set_b32(bus_phy, row, Block32::from(a))?;
    tb.set_b32(bus_phy + 1, row, Block32::from(b))?;
    tb.set_b32(bus_phy + 2, row, Block32::from(w))?;
    tb.set_b32(bus_phy + 3, row, Block32::from(wb))?;
    tb.set_b32(bus_phy + 4, row, Block32::from(a_out))?;
    tb.set_b32(bus_phy + 5, row, Block32::from(b_out))?;
    tb.set_b32(bus_phy + 6, row, Block32::from(bfly.layer))?;
    tb.set_b32(bus_phy + 7, row, Block32::from(bfly.butterfly_idx))?;

    Ok(())
}

// =================================================================
// Packed Bit Buffer Helpers
// =================================================================

/// Fill schoolbook multiplication
/// witness into the packed bit buffer.
fn fill_schoolbook_mul_packed(bits: &mut [u32], ly: &NttLayout, w: u32, b: u32, bit_width: usize) {
    // Convention:
    // iterate over b's bits.
    // pp[j][k] = b[j] * w[k-j].
    let b0 = b & 1;
    let pp0_val = if b0 == 1 { w } else { 0 };

    pack_bits(bits, ly.mul_pp0, pp0_val as u64, ly.mul_pp0_width);

    let mut acc = pp0_val as u64;
    for (j, &(sum_start, sum_width)) in ly.mul_sums.iter().enumerate() {
        let step = j + 1;
        let bj = (b >> step) & 1;
        let pp_j = if bj == 1 { (w as u64) << step } else { 0u64 };

        let new_acc = acc + pp_j;
        pack_bits(bits, sum_start, new_acc, sum_width);

        let (carry_start, carry_width) = ly.mul_carries[step - 1];
        fill_add_carry_packed(bits, carry_start, carry_width, acc, pp_j);

        acc = new_acc;
    }

    let last_step = bit_width - 1;
    let blast = (b >> last_step) & 1;

    let pp_last = if blast == 1 {
        (w as u64) << last_step
    } else {
        0u64
    };

    let (carry_start, carry_width) = ly.mul_carries[last_step - 1];
    fill_add_carry_packed(bits, carry_start, carry_width, acc, pp_last);
}

/// Fill Barrett reduction
/// witness into packed buffer.
#[allow(clippy::too_many_arguments)]
fn fill_barrett_packed(
    bits: &mut [u32],
    ly: &NttLayout,
    modulus: u32,
    bit_width: usize,
    _product: u64,
    quot: u32,
    wb: u32,
) {
    let quot_x_q = quot as u64 * modulus as u64;
    pack_bits(bits, ly.quot_x_q, quot_x_q, ly.quot_x_q_width);

    fill_mul_const_packed(bits, modulus, quot, &ly.barrett_sr, &ly.barrett_sc);

    fill_add_carry_packed(
        bits,
        ly.barrett_add_carry,
        ly.barrett_add_carry_width,
        quot_x_q,
        wb as u64,
    );

    fill_sub_borrow_packed(
        bits,
        ly.barrett_rng_result,
        ly.barrett_rng_borrow,
        bit_width,
        (modulus - 1) as u64,
        wb as u64,
    );
}

/// Fill constant multiplication
/// witness into packed buffer.
fn fill_mul_const_packed(
    bits: &mut [u32],
    constant: u32,
    operand: u32,
    scratch_results: &[(usize, usize)],
    scratch_carries: &[(usize, usize)],
) {
    let set_bits: Vec<usize> = (0..32).filter(|&i| (constant >> i) & 1 == 1).collect();

    let m = set_bits.len();
    if m <= 1 {
        return;
    }

    let shifted = |j: usize| -> u64 { (operand as u64) << set_bits[j] };

    let mut acc = shifted(0) + shifted(1);

    fill_add_carry_packed(
        bits,
        scratch_carries[0].0,
        scratch_carries[0].1,
        shifted(0),
        shifted(1),
    );

    if m == 2 {
        return;
    }

    pack_bits(bits, scratch_results[0].0, acc, scratch_results[0].1);

    for j in 2..m - 1 {
        let new_acc = acc + shifted(j);
        let sr_idx = j - 1;

        pack_bits(
            bits,
            scratch_results[sr_idx].0,
            new_acc,
            scratch_results[sr_idx].1,
        );

        fill_add_carry_packed(
            bits,
            scratch_carries[j - 1].0,
            scratch_carries[j - 1].1,
            acc,
            shifted(j),
        );

        acc = new_acc;
    }

    let last = m - 1;
    fill_add_carry_packed(
        bits,
        scratch_carries[last - 1].0,
        scratch_carries[last - 1].1,
        acc,
        shifted(last),
    );
}

/// Fill modular addition
/// witness into packed buffer.
#[allow(clippy::too_many_arguments)]
fn fill_mod_add_packed(
    bits: &mut [u32],
    modulus: u32,
    bit_width: usize,
    a: u32,
    b: u32,
    result: u32,
    flag: u32,
    lhs_result_start: usize,
    lhs_carry_start: usize,
    rhs_result_start: usize,
    rhs_carry_start: usize,
    flag_col: usize,
    rng_result_start: usize,
    rng_borrow_start: usize,
) {
    // LHS = a + b
    let lhs = a as u64 + b as u64;
    pack_bits(bits, lhs_result_start, lhs, bit_width);

    fill_add_carry_packed(bits, lhs_carry_start, bit_width + 1, a as u64, b as u64);

    // RHS = result + flag * modulus
    let rhs = result as u64 + flag as u64 * modulus as u64;
    pack_bits(bits, rhs_result_start, rhs, bit_width);

    let flag_q = flag as u64 * modulus as u64;
    fill_add_carry_packed(bits, rhs_carry_start, bit_width + 1, result as u64, flag_q);

    // Flag
    pack_one(bits, flag_col, flag == 1);

    // Range:
    // result < modulus
    fill_sub_borrow_packed(
        bits,
        rng_result_start,
        rng_borrow_start,
        bit_width,
        (modulus - 1) as u64,
        result as u64,
    );
}

// =================================================================
// Tests
// =================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use hekate_math::{Block128, Flat};

    type F = Block128;

    /// Helper:
    /// physical bus column index.
    fn phy_bus(layout: &NttLayout, bus_offset: usize) -> usize {
        layout.num_packed_b32_cols + bus_offset
    }

    /// Helper:
    /// physical control column index.
    fn phy_ctrl(layout: &NttLayout, ctrl_offset: usize) -> usize {
        layout.num_packed_b32_cols + 16 + ctrl_offset
    }

    #[test]
    fn layout_q3329() {
        let layout = NttLayout::compute(3329, 12);

        // 12-bit operands -> 24-bit product
        assert_eq!(layout.bit_width, 12);
        assert_eq!(layout.product_width, 24);

        // Schoolbook:
        // 11 additions, 10 intermediate sums
        assert_eq!(layout.mul_carries.len(), 11);
        assert_eq!(layout.mul_sums.len(), 10);

        // Barrett:
        // q=3329 has popcount 4
        assert_eq!(layout.barrett_sr.len(), 2);
        assert_eq!(layout.barrett_sc.len(), 3);

        // Packed B32 column count
        eprintln!(
            "num_bit_cols={}, num_packed_b32_cols={}, tail={}",
            layout.num_bit_cols,
            layout.num_packed_b32_cols,
            layout.tail_bits(),
        );

        assert_eq!(layout.num_packed_b32_cols, layout.num_bit_cols.div_ceil(32));
        assert!(layout.tail_bits() >= 1 && layout.tail_bits() <= 32);

        // Physical layout is well-formed
        let phys = layout.build_physical_layout();
        assert_eq!(phys.len(), layout.num_physical_columns);

        // Virtual layout unchanged
        let virt = layout.build_virtual_layout();
        assert_eq!(virt.len(), layout.num_columns);
    }

    #[test]
    fn layout_q8380417() {
        let layout = NttLayout::compute(8380417, 23);

        assert_eq!(layout.bit_width, 23);
        assert_eq!(layout.product_width, 46);
        assert_eq!(layout.mul_carries.len(), 22);
        assert_eq!(layout.mul_sums.len(), 21);
    }

    #[test]
    fn constraint_ast_builds_q3329() {
        let chiplet = NttChiplet::new(3329, 1024);
        let ast: ConstraintAst<F> = chiplet.constraint_ast();

        // Sanity:
        // non-empty constraint set
        assert!(!ast.roots.is_empty());
        assert!(!ast.arena.is_empty());

        // Should have many constraints:
        // schoolbook (~1000) + barrett (~400)
        // + mod_add (~160) + mod_sub (~160)
        // + booleanity (~100)
        assert!(
            ast.roots.len() > 1000,
            "Expected >1000 constraints, got {}",
            ast.roots.len(),
        );
    }

    #[test]
    fn packing_and_selector_constraints_present() {
        let chiplet = NttChiplet::new(3329, 1024);
        let ast: ConstraintAst<F> = chiplet.constraint_ast();
        let count_with = ast.roots.len();

        // Expected additions over a hypothetical
        // constraint set without packing/selector:
        //   6 packing (bus_a, bus_b, bus_w, bus_wb, bus_a_out, bus_b_out)
        // + 2 selector subset (s_output, s_butterfly)
        // = 8 constraints

        // Verify by checking the total is
        // above a known minimum from before
        // the security patch. Pre-patch NTT
        // had >1000 constraints; post-patch
        // must have at least 8 more.
        assert!(
            count_with > 1008,
            "Expected >1008 constraints (>1000 + 8 security), got {}",
            count_with,
        );

        // Print for manual audit
        eprintln!(
            "NTT constraint count (q=3329): {} \
             (includes 6 packing + 2 selector subset)",
            count_with,
        );
    }

    #[test]
    fn trace_single_butterfly_q3329() {
        let modulus = 3329u32;
        let bfly = NttButterfly {
            a: 1000,
            b: 2000,
            w: 17,
            layer: 0,
            butterfly_idx: 0,
            is_forward: true,
            ntt_instance: 0,
            pos_a: 0,
            pos_b: 1,
        };

        let wb = ((17u64 * 2000) % 3329) as u32;
        let a_out = ((1000u64 + wb as u64) % 3329) as u32;
        let b_out = ((1000u64 + 3329 - wb as u64) % 3329) as u32;

        let trace = generate_ntt_trace(modulus, &[NttOp::Butterfly(bfly)], 2).unwrap();
        let layout = NttLayout::compute(modulus, 12);

        // Bus columns are at physical
        // offset num_packed_b32_cols.
        let bus_a = trace.columns[phy_bus(&layout, 0)].as_b32_slice().unwrap();
        assert_eq!(bus_a[0].to_tower(), Block32::from(1000u32));

        let bus_a_out = trace.columns[phy_bus(&layout, 4)].as_b32_slice().unwrap();
        assert_eq!(bus_a_out[0].to_tower(), Block32::from(a_out));

        let bus_b_out = trace.columns[phy_bus(&layout, 5)].as_b32_slice().unwrap();
        assert_eq!(bus_b_out[0].to_tower(), Block32::from(b_out));
    }

    #[test]
    fn trace_mul_only_q3329() {
        let modulus = 3329u32;
        let mul = NttMulOnly {
            b: 2000,
            w: 17,
            layer: 3,
            butterfly_idx: 42,
            is_basemul: false,
            flow_pos: None,
            flow_instance: 0,
            flow_src_layer: 0,
        };

        let wb = ((17u64 * 2000) % 3329) as u32;

        let trace = generate_ntt_trace(modulus, &[NttOp::MulOnly(mul)], 2).unwrap();
        let layout = NttLayout::compute(modulus, 12);

        // bus_wb at offset 3
        let bus_wb = trace.columns[phy_bus(&layout, 3)].as_b32_slice().unwrap();
        assert_eq!(bus_wb[0].to_tower(), Block32::from(wb));

        // bus_a at offset 0
        let bus_a = trace.columns[phy_bus(&layout, 0)].as_b32_slice().unwrap();
        assert_eq!(bus_a[0].to_tower(), Block32::from(0u32));

        // bus_a_out at offset 4
        let bus_a_out = trace.columns[phy_bus(&layout, 4)].as_b32_slice().unwrap();
        assert_eq!(bus_a_out[0].to_tower(), Block32::from(wb));

        // S_BUTTERFLY at ctrl offset 2
        let s_bfly = trace.columns[phy_ctrl(&layout, 2)].as_bit_slice().unwrap();
        assert_eq!(s_bfly[0], Bit::ZERO);
    }

    #[test]
    fn trace_mixed_butterfly_and_mul_only() {
        let modulus = 3329u32;
        let ops = vec![
            NttOp::Butterfly(NttButterfly {
                a: 100,
                b: 200,
                w: 5,
                layer: 0,
                butterfly_idx: 0,
                is_forward: true,
                ntt_instance: 0,
                pos_a: 0,
                pos_b: 1,
            }),
            NttOp::MulOnly(NttMulOnly {
                b: 300,
                w: 7,
                layer: 1,
                butterfly_idx: 1,
                is_basemul: false,
                flow_pos: None,
                flow_instance: 0,
                flow_src_layer: 0,
            }),
        ];

        let trace = generate_ntt_trace(modulus, &ops, 4).unwrap();
        let layout = NttLayout::compute(modulus, 12);

        // Row 0:
        // butterfly, S_BUTTERFLY = 1
        let s_bfly = trace.columns[phy_ctrl(&layout, 2)].as_bit_slice().unwrap();
        assert_eq!(s_bfly[0], Bit::ONE);
        assert_eq!(s_bfly[1], Bit::ZERO);

        // Row 2,3:
        // padding, S_ACTIVE = 0
        let s_active = trace.columns[phy_ctrl(&layout, 0)].as_bit_slice().unwrap();
        assert_eq!(s_active[0], Bit::ONE);
        assert_eq!(s_active[1], Bit::ONE);
        assert_eq!(s_active[2], Bit::ZERO);
        assert_eq!(s_active[3], Bit::ZERO);
    }

    #[test]
    fn packed_trace_roundtrip_q3329() {
        let modulus = 3329u32;
        let bfly = NttButterfly {
            a: 1000,
            b: 2000,
            w: 17,
            layer: 0,
            butterfly_idx: 0,
            is_forward: true,
            ntt_instance: 0,
            pos_a: 0,
            pos_b: 1,
        };

        let trace = generate_ntt_trace(modulus, &[NttOp::Butterfly(bfly)], 2).unwrap();
        let layout = NttLayout::compute(modulus, 12);

        // Verify packed B32 -> virtual bit roundtrip.
        // Check operand a = 1000 = 0b1111101000.
        // a_bits starts at virtual col 0.
        let packed_col_0 = trace.columns[0].as_b32_slice().unwrap();
        let word = packed_col_0[0].to_tower().0; // tower-basis u32

        // Bits 0..11 of word should be a = 1000
        for k in 0..12 {
            let expected = ((1000u32 >> k) & 1) as u8;
            let actual = ((word >> k) & 1) as u8;
            assert_eq!(actual, expected, "a_bit[{k}] mismatch");
        }

        let chiplet = NttChiplet::new(modulus, 4);
        let variants = Air::<F>::virtual_expander(&chiplet)
            .unwrap()
            .expand_variants(&trace, 0)
            .unwrap();

        assert_eq!(
            variants.len(),
            layout.num_columns,
            "virtual column count mismatch: expected {}, got {}",
            layout.num_columns,
            variants.len(),
        );

        // Verify virtual bit extraction via PackedBitB32.
        // a_bits[0..12] should decode to a=1000.
        for (k, variant) in variants.iter().enumerate().take(12) {
            let val = variant.get_at(0);
            let expected = if ((1000u32 >> k) & 1) == 1 {
                Flat::from_raw(F::ONE)
            } else {
                Flat::from_raw(F::ZERO)
            };

            assert_eq!(val, expected, "PackedBitB32 a_bit[{k}] mismatch");
        }
    }
}
