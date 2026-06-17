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

//! RAM Chiplet:
//! Memory with CPU Linking via Grand Product Argument.
//! The RAM chiplet maintains a sorted memory table
//! and links to the CPU's memory events.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use hekate_core::errors::Error;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate_math::{Bit, Block32, Block128, TowerField};
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::expander::VirtualExpander;
use hekate_program::permutation::{PermutationCheckSpec, Source};
use hekate_program::{Air, FixedColumn, define_columns};
use once_cell::race::OnceBox;

// Physical layout constants.
const NUM_PACKED_B32: usize = 2;
const NUM_B32_DATA: usize = 13; // addr(4) + clk(4) + val(4) + val_packed(1)
const NUM_B128: usize = 1; // AUX_INV
const NUM_CONTROL_BIT: usize = 5; // IS_WRITE, SELECTOR, Q_STEP, Q_FIRST, Q_LAST
const NUM_PHYSICAL_COLUMNS: usize = NUM_PACKED_B32 + NUM_B32_DATA + NUM_B128 + NUM_CONTROL_BIT; // = 21

// Physical column indices.
// Order:
// packed B32 (2)
// + data B32 (13)
// + B128 (1)
// + control Bit (5).
const PHY_PACK_SORT: usize = 0;
const PHY_PACK_VAL: usize = 1;
const PHY_ADDR_B0: usize = NUM_PACKED_B32;
const PHY_CLK_B0: usize = NUM_PACKED_B32 + 4;
const PHY_VAL_B0: usize = NUM_PACKED_B32 + 8;
const PHY_VAL_PACKED: usize = NUM_PACKED_B32 + 12;
const PHY_AUX_INV: usize = NUM_PACKED_B32 + NUM_B32_DATA;
const PHY_IS_WRITE: usize = NUM_PACKED_B32 + NUM_B32_DATA + NUM_B128;
const PHY_SELECTOR: usize = PHY_IS_WRITE + 1;
const PHY_Q_STEP: usize = PHY_IS_WRITE + 2;
const PHY_Q_FIRST: usize = PHY_IS_WRITE + 3;
const PHY_Q_LAST: usize = PHY_IS_WRITE + 4;

/// Default pack_sort for rows with no
/// comparison (last event row, padding rows).
/// DIFF_BYTE_IDX[0]=1,
/// DIFF_BIT_IDX[0]=1,
/// rest zero.
const PACK_SORT_DEFAULT: u32 = 1 | (1 << 8);

define_columns! {
    pub RamColumns {
        // Packed Bit columns.
        // Sorting helpers (32 Bit)
        DIFF_BYTE_IDX: [Bit; 8],
        DIFF_BIT_IDX: [Bit; 8],
        A_BITS: [Bit; 8],
        B_BITS: [Bit; 8],

        // Value bit decomposition
        VAL_BITS: [Bit; 32],

        // B32 data columns (13)
        ADDR_B0: B32,
        ADDR_B1: B32,
        ADDR_B2: B32,
        ADDR_B3: B32,
        CLK_B0: B32,
        CLK_B1: B32,
        CLK_B2: B32,
        CLK_B3: B32,
        VAL_B0: B32,
        VAL_B1: B32,
        VAL_B2: B32,
        VAL_B3: B32,
        VAL_PACKED: B32,

        // B128 auxiliary
        AUX_INV: B128,

        // Unpacked control Bit columns
        IS_WRITE: Bit,
        SELECTOR: Bit,
        Q_STEP: Bit,
        Q_FIRST: Bit,
        Q_LAST: Bit,
    }
}

define_columns! {
    pub CpuMemColumns {
        ADDR_B0: B32,
        ADDR_B1: B32,
        ADDR_B2: B32,
        ADDR_B3: B32,
        VAL_B0: B32,
        VAL_B1: B32,
        VAL_B2: B32,
        VAL_B3: B32,
        IS_WRITE: Bit,
        SELECTOR: Bit,
    }
}

/// RAM chiplet column layout.
#[derive(Clone, Debug)]
pub struct RamChiplet {
    /// Number of memory events (power of 2)
    pub num_rows: usize,
}

impl RamChiplet {
    pub const BUS_ID: &'static str = "ram_link";
    pub const VALUE_BUS_ID: &'static str = "ram_value_bind";

    /// Creates a new RAM chiplet
    /// with the given number of rows.
    pub fn new(num_rows: usize) -> Self {
        assert!(num_rows.is_power_of_two(), "RAM size must be power of 2");
        Self { num_rows }
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn num_columns(&self) -> usize {
        RamColumns::NUM_COLUMNS
    }

    /// Physical layout for Brakedown commitment.
    pub fn build_physical_layout() -> Vec<ColumnType> {
        let mut layout = Vec::with_capacity(NUM_PHYSICAL_COLUMNS);

        for _ in 0..NUM_PACKED_B32 {
            layout.push(ColumnType::B32);
        }

        for _ in 0..NUM_B32_DATA {
            layout.push(ColumnType::B32);
        }

        for _ in 0..NUM_B128 {
            layout.push(ColumnType::B128);
        }

        for _ in 0..NUM_CONTROL_BIT {
            layout.push(ColumnType::Bit);
        }

        debug_assert_eq!(layout.len(), NUM_PHYSICAL_COLUMNS);

        layout
    }

    /// Returns the permutation check
    /// specification for RAM-CPU linking.
    ///
    /// # Linking Key
    /// K = Σ κ_addr_i · addr_i + Σ κ_clk_j · clk_j + Σ κ_val_k · val_k + κ_write · is_write
    ///
    /// All 13 data fields are included in
    /// the key to ensure complete event matching.
    pub fn linking_spec() -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(RamColumns::ADDR_B0),
                    b"kappa_addr_b0" as &[u8],
                ),
                (
                    Source::Column(RamColumns::ADDR_B1),
                    b"kappa_addr_b1" as &[u8],
                ),
                (
                    Source::Column(RamColumns::ADDR_B2),
                    b"kappa_addr_b2" as &[u8],
                ),
                (
                    Source::Column(RamColumns::ADDR_B3),
                    b"kappa_addr_b3" as &[u8],
                ),
                (Source::Column(RamColumns::CLK_B0), b"kappa_clk_b0" as &[u8]),
                (Source::Column(RamColumns::CLK_B1), b"kappa_clk_b1" as &[u8]),
                (Source::Column(RamColumns::CLK_B2), b"kappa_clk_b2" as &[u8]),
                (Source::Column(RamColumns::CLK_B3), b"kappa_clk_b3" as &[u8]),
                (Source::Column(RamColumns::VAL_B0), b"kappa_val_b0" as &[u8]),
                (Source::Column(RamColumns::VAL_B1), b"kappa_val_b1" as &[u8]),
                (Source::Column(RamColumns::VAL_B2), b"kappa_val_b2" as &[u8]),
                (Source::Column(RamColumns::VAL_B3), b"kappa_val_b3" as &[u8]),
                (
                    Source::Column(RamColumns::IS_WRITE),
                    b"kappa_is_write" as &[u8],
                ),
            ],
            Some(RamColumns::SELECTOR),
        )
        .with_clock_waiver(
            "see ram.rs: partner CpuMemoryUnit::linking_spec carries Source::RowIndexByte; \
             this side stores that clock in committed CLK_B0..3 columns sorted by AIR",
        )
    }

    pub fn value_binding_spec() -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(RamColumns::VAL_PACKED),
                    b"kappa_val_packed" as &[u8],
                ),
                (
                    Source::Column(RamColumns::ADDR_B0),
                    b"kappa_vb_addr_b0" as &[u8],
                ),
                (
                    Source::Column(RamColumns::ADDR_B1),
                    b"kappa_vb_addr_b1" as &[u8],
                ),
                (
                    Source::Column(RamColumns::ADDR_B2),
                    b"kappa_vb_addr_b2" as &[u8],
                ),
                (
                    Source::Column(RamColumns::ADDR_B3),
                    b"kappa_vb_addr_b3" as &[u8],
                ),
                (
                    Source::Column(RamColumns::CLK_B0),
                    b"kappa_vb_clk_b0" as &[u8],
                ),
                (
                    Source::Column(RamColumns::CLK_B1),
                    b"kappa_vb_clk_b1" as &[u8],
                ),
                (
                    Source::Column(RamColumns::CLK_B2),
                    b"kappa_vb_clk_b2" as &[u8],
                ),
                (
                    Source::Column(RamColumns::CLK_B3),
                    b"kappa_vb_clk_b3" as &[u8],
                ),
            ],
            Some(RamColumns::SELECTOR),
        )
        .with_clock_waiver(
            "see hekate-gadgets/src/ram.rs: RAM-internal value binding pinned by \
             sorted-by-(addr,clk) AIR transitions plus CpuMemoryUnit linking_spec",
        )
    }
}

/// CPU memory event column
/// layout for RAM linking.
///
/// The CPU side emits memory events
/// in execution order (unsorted).
///
/// # Column Schema (14 columns total)
/// Same as RAM side, but:
/// - Clock bytes are generated virtually from RowIndexLeBytes (CPU execution time)
/// - Events are in execution order (not sorted)
#[derive(Clone, Debug)]
pub struct CpuMemoryUnit;

impl CpuMemoryUnit {
    /// Returns the permutation check
    /// specification for CPU memory side.
    ///
    /// This uses RowIndexLeBytes for clock (virtual time),
    /// unlike RAM which has committed columns.
    pub fn linking_spec() -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(CpuMemColumns::ADDR_B0),
                    b"kappa_addr_b0" as &[u8],
                ),
                (
                    Source::Column(CpuMemColumns::ADDR_B1),
                    b"kappa_addr_b1" as &[u8],
                ),
                (
                    Source::Column(CpuMemColumns::ADDR_B2),
                    b"kappa_addr_b2" as &[u8],
                ),
                (
                    Source::Column(CpuMemColumns::ADDR_B3),
                    b"kappa_addr_b3" as &[u8],
                ),
                (Source::RowIndexByte(0), b"kappa_clk_b0" as &[u8]),
                (Source::RowIndexByte(1), b"kappa_clk_b1" as &[u8]),
                (Source::RowIndexByte(2), b"kappa_clk_b2" as &[u8]),
                (Source::RowIndexByte(3), b"kappa_clk_b3" as &[u8]),
                (
                    Source::Column(CpuMemColumns::VAL_B0),
                    b"kappa_val_b0" as &[u8],
                ),
                (
                    Source::Column(CpuMemColumns::VAL_B1),
                    b"kappa_val_b1" as &[u8],
                ),
                (
                    Source::Column(CpuMemColumns::VAL_B2),
                    b"kappa_val_b2" as &[u8],
                ),
                (
                    Source::Column(CpuMemColumns::VAL_B3),
                    b"kappa_val_b3" as &[u8],
                ),
                (
                    Source::Column(CpuMemColumns::IS_WRITE),
                    b"kappa_is_write" as &[u8],
                ),
            ],
            Some(CpuMemColumns::SELECTOR),
        )
    }

    pub fn num_columns(&self) -> usize {
        CpuMemColumns::NUM_COLUMNS
    }
}

/// Memory event representation.
///
/// This is the prover's witness
/// format before trace generation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryEvent {
    /// Memory address (32-bit)
    pub addr: u32,

    /// Clock/timestamp (32-bit)
    pub clk: u32,

    /// Value (32-bit)
    pub val: u32,

    /// Write flag (true=write, false=read)
    pub is_write: bool,
}

impl MemoryEvent {
    /// Creates a new memory event.
    pub fn new(addr: u32, clk: u32, val: u32, is_write: bool) -> Self {
        Self {
            addr,
            clk,
            val,
            is_write,
        }
    }

    /// Creates a write event.
    pub fn write(addr: u32, clk: u32, val: u32) -> Self {
        Self::new(addr, clk, val, true)
    }

    /// Creates a read event.
    pub fn read(addr: u32, clk: u32, val: u32) -> Self {
        Self::new(addr, clk, val, false)
    }

    /// Extracts address bytes (little-endian).
    pub fn addr_bytes(&self) -> [u8; 4] {
        self.addr.to_le_bytes()
    }

    /// Extracts clock bytes (little-endian).
    pub fn clk_bytes(&self) -> [u8; 4] {
        self.clk.to_le_bytes()
    }

    /// Extracts value bytes (little-endian).
    pub fn val_bytes(&self) -> [u8; 4] {
        self.val.to_le_bytes()
    }

    /// Returns the sort key (addr || clk)
    /// for lexicographic ordering.
    ///
    /// This creates a 64-bit key where
    /// upper 32 bits = addr,
    /// lower 32 bits = clk.
    /// Sorting by this key ensures
    /// (addr, clk) lexicographic order.
    pub fn sort_key(&self) -> u64 {
        ((self.addr as u64) << 32) | (self.clk as u64)
    }
}

// =========================================================
// AIR TRAIT IMPLEMENTATION
// =========================================================

impl<F: TowerField> Air<F> for RamChiplet {
    fn name(&self) -> String {
        "RamChiplet".to_string()
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: OnceBox<Vec<ColumnType>> = OnceBox::new();
        LAYOUT.get_or_init(|| Box::new(Self::build_physical_layout()))
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![(Self::BUS_ID.into(), Self::linking_spec())]
    }

    /// Closes the uniform `q_last ≡ 0` forgery the
    /// q_last/q_first relational chain admitted alone.
    fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
        vec![
            FixedColumn::last_row(RamColumns::Q_STEP),
            FixedColumn::first_row(RamColumns::Q_FIRST),
        ]
    }

    fn virtual_expander(&self) -> Option<&VirtualExpander> {
        static E: OnceBox<VirtualExpander> = OnceBox::new();
        Some(E.get_or_init(|| {
            Box::new(
                VirtualExpander::new()
                    .expand_bits(NUM_PACKED_B32, ColumnType::B32)
                    .pass_through(NUM_B32_DATA, ColumnType::B32)
                    .pass_through(NUM_B128, ColumnType::B128)
                    .control_bits(NUM_CONTROL_BIT)
                    .build()
                    .expect("RamChiplet expander"),
            )
        }))
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        let one = cs.one();

        let selector = cs.col(RamColumns::SELECTOR);
        let q_step = cs.col(RamColumns::Q_STEP);
        let q_first = cs.col(RamColumns::Q_FIRST);
        let inv = cs.col(RamColumns::AUX_INV);
        let is_write = cs.col(RamColumns::IS_WRITE);
        let is_write_next = cs.next(RamColumns::IS_WRITE);
        let s_mem_next = cs.next(RamColumns::SELECTOR);

        cs.assert_boolean(selector);
        cs.assert_boolean(q_step);
        cs.assert_boolean(q_first);
        cs.assert_boolean(is_write);

        // Address/value column arrays
        let addr: [_; 4] = core::array::from_fn(|i| cs.col(RamColumns::ADDR_B0 + i));
        let addr_next: [_; 4] = core::array::from_fn(|i| cs.next(RamColumns::ADDR_B0 + i));
        let val: [_; 4] = core::array::from_fn(|i| cs.col(RamColumns::VAL_B0 + i));
        let val_next: [_; 4] = core::array::from_fn(|i| cs.next(RamColumns::VAL_B0 + i));

        // 1. Address difference:
        // D = Σ 256^i * (addr_next_i + addr_i)
        let coeffs = [1u128, 256, 65536, 16777216];
        let d = cs.sum(
            &(0..4)
                .map(|i| cs.scale(F::from(coeffs[i]), addr_next[i] + addr[i]))
                .collect::<Vec<_>>(),
        );

        // 2. Inverse validity:
        // q_step * (d + d*d*inv) = 0
        let d_inv = d * inv;
        cs.assert_zero_when(q_step, d + d * d_inv);

        // 3. Value difference:
        // V = Σ 256^i * (val_next_i + val_i)
        let v_diff = cs.sum(
            &(0..4)
                .map(|i| cs.scale(F::from(coeffs[i]), val_next[i] + val[i]))
                .collect::<Vec<_>>(),
        );

        // 4. Memory consistency:
        // q_step * (1+d*inv) * (1+is_write_next) * v_diff = 0
        let not_dinv = one + d_inv;
        let not_write_next = one + is_write_next;

        cs.assert_zero_when(q_step, not_dinv * not_write_next * v_diff);

        // 5. Initial read protection:
        // q_step * d*inv * (1+is_write_next) * val_next_byte = 0
        for &vn in &val_next {
            cs.assert_zero_when(q_step, d_inv * not_write_next * vn);
        }

        // 5.2. Absolute first row:
        // q_first * (1+is_write) * val_byte = 0
        let not_write = one + is_write;
        for &v in &val {
            cs.constrain(q_first * not_write * v);
        }

        // 6. Lexicographic sorting
        let byte_indices = [
            RamColumns::ADDR_B3,
            RamColumns::ADDR_B2,
            RamColumns::ADDR_B1,
            RamColumns::ADDR_B0,
            RamColumns::CLK_B3,
            RamColumns::CLK_B2,
            RamColumns::CLK_B1,
            RamColumns::CLK_B0,
        ];

        let diff_byte: [_; 8] = core::array::from_fn(|i| cs.col(RamColumns::DIFF_BYTE_IDX + i));
        let diff_bit: [_; 8] = core::array::from_fn(|i| cs.col(RamColumns::DIFF_BIT_IDX + i));
        let a_bits: [_; 8] = core::array::from_fn(|i| cs.col(RamColumns::A_BITS + i));
        let b_bits: [_; 8] = core::array::from_fn(|i| cs.col(RamColumns::B_BITS + i));

        // 6.1 One-hot and booleanity
        cs.assert_one_hot(&diff_byte);

        for i in 0..8 {
            cs.assert_boolean(diff_bit[i]);
            cs.assert_boolean(a_bits[i]);
            cs.assert_boolean(b_bits[i]);
            cs.assert_boolean(diff_byte[i]);
        }

        // 6.2 Byte comparison logic
        let bit_decomp = |bits: &[_; 8]| {
            cs.sum(
                &(0..8)
                    .map(|j| cs.scale(F::from(1u128 << j), bits[j]))
                    .collect::<Vec<_>>(),
            )
        };

        let a_decomp = bit_decomp(&a_bits);
        let b_decomp = bit_decomp(&b_bits);

        for i in 0..8 {
            let col = byte_indices[i];
            let is_diff = diff_byte[i];

            // Decompose differing byte
            // into a_bits / b_bits.
            cs.assert_zero_when(q_step * is_diff, cs.next(col) + a_decomp);
            cs.constrain(is_diff * (cs.col(col) + b_decomp));

            // Higher bytes must be equal
            for &higher_col in byte_indices.iter().take(i) {
                cs.assert_zero_when(q_step * is_diff, cs.next(higher_col) + cs.col(higher_col));
            }
        }

        // 6.3 Strictly greater-than check
        let gt_gate = q_step * s_mem_next;

        // One-hot for diff_bit
        cs.assert_zero_when(gt_gate, cs.sum(&diff_bit.map(|x| x)) + one);

        // Booleanity (gated)
        for i in 0..8 {
            for &cell in &[a_bits[i], b_bits[i], diff_bit[i]] {
                cs.assert_zero_when(gt_gate, cell * cell + cell);
            }
        }

        // Difference logic:
        // diff_bit[i] * (a[i]*(b[i]+1) + 1) = 0
        for i in 0..8 {
            cs.assert_zero_when(gt_gate, diff_bit[i] * (a_bits[i] * (b_bits[i] + one) + one));
        }

        // Equality for higher bits
        for (i, &db) in diff_bit.iter().enumerate() {
            for j in (i + 1)..8 {
                cs.assert_zero_when(gt_gate, db * (a_bits[j] + b_bits[j]));
            }
        }

        // 7. Value bit decomposition + packing
        let val_bits: [_; 32] = core::array::from_fn(|k| cs.col(RamColumns::VAL_BITS + k));
        let val_packed = cs.col(RamColumns::VAL_PACKED);

        // 7.1 Booleanity:
        // val_bits[k] * (val_bits[k] + 1) = 0
        for &vb in &val_bits {
            cs.assert_boolean(vb);
        }

        // 7.2 Byte-to-bit binding:
        // VAL_Bk + Σ_{j=0}^{7} val_bit_{8k+j} * F::from(1 << j) = 0
        for k in 0..4 {
            let byte_col = cs.col(RamColumns::VAL_B0 + k);
            let bit_sum = cs.sum(
                &(0..8)
                    .map(|j| cs.scale(F::from(1u128 << j), val_bits[8 * k + j]))
                    .collect::<Vec<_>>(),
            );

            cs.constrain(byte_col + bit_sum);
        }

        // 7.3 Packed value:
        // VAL_PACKED + Σ_{k=0}^{31} val_bit_k * F::from(1 << k) = 0
        let packed_sum = cs.sum(
            &(0..32)
                .map(|k| cs.scale(F::from(1u128 << k), val_bits[k]))
                .collect::<Vec<_>>(),
        );

        cs.constrain(val_packed + packed_sum);

        // IS_WRITE is in the bus key;
        // pin to 0 on padding.
        let not_selector = one + selector;
        cs.assert_zero_when(not_selector, is_write);

        let q_last = cs.col(RamColumns::Q_LAST);
        let q_last_next = cs.next(RamColumns::Q_LAST);

        cs.assert_boolean(q_last);

        cs.constrain(q_step + q_last + one);
        cs.constrain(q_last * q_last_next);

        cs.build()
    }
}

/// Generates a RAM trace from a list of memory events.
///
/// Produces a physical layout trace (20 columns)
/// with sorting and value bits packed into B32 columns.
pub fn generate_ram_trace(
    events: &[MemoryEvent],
    num_rows: usize,
) -> hekate_core::errors::Result<ColumnTrace> {
    assert!(
        num_rows.is_power_of_two(),
        "RAM trace size must be power of 2, got {num_rows}"
    );

    if events.len() > num_rows {
        return Err(Error::Protocol {
            protocol: "ram",
            message: "too many events for trace size",
        });
    }

    // Sort events by (addr || clk)
    let mut sorted_events = events.to_vec();
    sorted_events.sort_by_key(|e| e.sort_key());

    // Verify sorting and consistency
    if !comparator::verify_sorted(&sorted_events) {
        return Err(Error::Protocol {
            protocol: "ram",
            message: "events not sorted by (addr, clk)",
        });
    }
    if !comparator::verify_consistency(&sorted_events) {
        return Err(Error::Protocol {
            protocol: "ram",
            message: "memory consistency violated: read value differs from last write",
        });
    }

    let num_vars = num_rows.trailing_zeros() as usize;
    let mut tb = TraceBuilder::new(&RamChiplet::build_physical_layout(), num_vars)?;

    for (i, event) in sorted_events.iter().enumerate() {
        let addr_bytes = event.addr_bytes();
        let clk_bytes = event.clk_bytes();
        let val_bytes = event.val_bytes();

        // B32 data columns
        for j in 0..4 {
            tb.set_b32(PHY_ADDR_B0 + j, i, Block32::from(addr_bytes[j] as u32))?;
            tb.set_b32(PHY_CLK_B0 + j, i, Block32::from(clk_bytes[j] as u32))?;
            tb.set_b32(PHY_VAL_B0 + j, i, Block32::from(val_bytes[j] as u32))?;
        }

        tb.set_b32(PHY_VAL_PACKED, i, Block32::from(event.val))?;

        // Packed value bits:
        // val decomposed into 32 bits.
        tb.set_b32(PHY_PACK_VAL, i, Block32::from(event.val))?;

        // Control Bit columns
        tb.set_bit(
            PHY_IS_WRITE,
            i,
            if event.is_write { Bit::ONE } else { Bit::ZERO },
        )?;
        tb.set_bit(PHY_SELECTOR, i, Bit::ONE)?;
        tb.set_bit(
            PHY_Q_STEP,
            i,
            if i < num_rows - 1 {
                Bit::ONE
            } else {
                Bit::ZERO
            },
        )?;
        tb.set_bit(PHY_Q_FIRST, i, if i == 0 { Bit::ONE } else { Bit::ZERO })?;
        tb.set_bit(
            PHY_Q_LAST,
            i,
            if i == num_rows - 1 {
                Bit::ONE
            } else {
                Bit::ZERO
            },
        )?;

        // AUX_INV:
        // address difference inverse.
        // Tower multiplication by
        // 256 is NOT bit shift.
        let next_addr = if i + 1 < sorted_events.len() {
            sorted_events[i + 1].addr
        } else if num_rows > sorted_events.len() {
            0
        } else {
            sorted_events[0].addr
        };

        let coeffs = [1u128, 256, 65536, 16777216];

        let mut curr_addr_field = Block128::ZERO;
        let curr_addr_bytes = event.addr_bytes();

        for j in 0..4 {
            curr_addr_field +=
                Block128::from(curr_addr_bytes[j] as u128) * Block128::from(coeffs[j]);
        }

        let mut next_addr_field = Block128::ZERO;
        let next_addr_bytes = next_addr.to_le_bytes();

        for j in 0..4 {
            next_addr_field +=
                Block128::from(next_addr_bytes[j] as u128) * Block128::from(coeffs[j]);
        }

        let field_diff = next_addr_field + curr_addr_field;
        tb.set_b128(PHY_AUX_INV, i, field_diff.invert())?;

        // Packed sorting helpers
        if i + 1 < sorted_events.len() {
            let next = &sorted_events[i + 1];

            let curr_key = [
                event.addr_bytes()[3],
                event.addr_bytes()[2],
                event.addr_bytes()[1],
                event.addr_bytes()[0],
                event.clk_bytes()[3],
                event.clk_bytes()[2],
                event.clk_bytes()[1],
                event.clk_bytes()[0],
            ];
            let next_key = [
                next.addr_bytes()[3],
                next.addr_bytes()[2],
                next.addr_bytes()[1],
                next.addr_bytes()[0],
                next.clk_bytes()[3],
                next.clk_bytes()[2],
                next.clk_bytes()[1],
                next.clk_bytes()[0],
            ];

            let mut b_idx = 0;
            for k in 0..8 {
                if next_key[k] != curr_key[k] {
                    b_idx = k;
                    break;
                }
            }

            let byte_a = next_key[b_idx];
            let byte_b = curr_key[b_idx];

            let mut bit_idx = 0;
            for k in (0..8).rev() {
                if ((byte_a >> k) & 1) != ((byte_b >> k) & 1) {
                    bit_idx = k;
                    break;
                }
            }

            tb.set_b32(
                PHY_PACK_SORT,
                i,
                Block32::from(pack_sort_bits(b_idx, bit_idx, byte_a, byte_b)),
            )?;
        } else {
            tb.set_b32(PHY_PACK_SORT, i, Block32::from(PACK_SORT_DEFAULT))?;
        }
    }

    // Padding rows
    for row in sorted_events.len()..num_rows {
        tb.set_b32(PHY_PACK_SORT, row, Block32::from(PACK_SORT_DEFAULT))?;

        if row < num_rows - 1 {
            tb.set_bit(PHY_Q_STEP, row, Bit::ONE)?;
        } else {
            tb.set_bit(PHY_Q_LAST, row, Bit::ONE)?;
        }
    }

    Ok(tb.build())
}

/// Pack sorting helper bits into a u32.
///
/// Bit layout matches virtual column order:
/// [0..7]   = DIFF_BYTE_IDX (one-hot)
/// [8..15]  = DIFF_BIT_IDX (one-hot)
/// [16..23] = A_BITS (byte decomp)
/// [24..31] = B_BITS (byte decomp)
fn pack_sort_bits(b_idx: usize, bit_idx: usize, byte_a: u8, byte_b: u8) -> u32 {
    (1u32 << b_idx) | ((1u32 << bit_idx) << 8) | ((byte_a as u32) << 16) | ((byte_b as u32) << 24)
}

/// Comparator gadget for byte-wise
/// lexicographic comparison.
///
/// This module implements constraints
/// to enforce that RAM trace is sorted
/// by (addr || clk) in MSB-first
/// lexicographic order.
///
/// # Algorithm
/// To prove row(i) < row(i+1) lexicographically:
/// 1. Find the first differing byte position k
/// 2. Prove that byte_k(i) < byte_k(i+1)
/// 3. Prove all higher-significance bytes are equal
///
/// # Constraint Structure
/// For each consecutive pair of rows, we need:
/// - addr_b3(i) < addr_b3(i+1), OR
/// - addr_b3(i) == addr_b3(i+1) AND addr_b2(i) < addr_b2(i+1), OR
/// - addr_b3(i) == addr_b3(i+1) AND addr_b2(i) == addr_b2(i+1) AND addr_b1(i) < addr_b1(i+1), OR
/// - ... (continue for all address bytes, then clock bytes)
mod comparator {
    use super::*;

    /// Enforces lexicographic sorting
    /// constraint for (addr || clk).
    ///
    /// This is a simplified V1 implementation
    /// that checks sorting via witness generation.
    /// A full AIR implementation would encode
    /// byte-wise comparison constraints.
    ///
    /// # Arguments
    /// - `events`: List of memory events (must be pre-sorted by caller)
    ///
    /// # Returns
    /// True if events are properly sorted, false otherwise.
    pub fn verify_sorted(events: &[MemoryEvent]) -> bool {
        for i in 0..events.len().saturating_sub(1) {
            if events[i].sort_key() >= events[i + 1].sort_key() {
                return false;
            }
        }

        true
    }

    /// Enforces memory consistency constraint.
    ///
    /// For consecutive rows with the same address:
    /// if addr(i) == addr(i+1) AND is_write(i+1) == 0:
    ///     val(i) MUST equal val(i+1)
    ///
    /// This ensures read consistency:
    /// reads observe the most recent value. Writes
    /// (is_write=1) are allowed to change the value.
    pub fn verify_consistency(events: &[MemoryEvent]) -> bool {
        if events.is_empty() {
            return true;
        }

        // Rule 1:
        // The very first event in the sorted table
        // must be a Write OR a Read of value 0.
        if !events[0].is_write && events[0].val != 0 {
            return false;
        }

        for i in 0..events.len().saturating_sub(1) {
            if events[i].addr == events[i + 1].addr {
                // Rule 2:
                // Read-after-Write / Read-after-Read consistency
                if !events[i + 1].is_write && events[i].val != events[i + 1].val {
                    return false;
                }
            } else {
                // Rule 3:
                // First access to a NEW address
                // must be a Write OR a Read of value 0.
                if !events[i + 1].is_write && events[i + 1].val != 0 {
                    return false;
                }
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hekate_core::trace::Trace;

    #[test]
    fn ram_chiplet_column_count() {
        let ram = RamChiplet::new(16);
        assert_eq!(ram.num_columns(), 83);
        assert_eq!(ram.num_rows(), 16);
    }

    #[test]
    fn memory_event_sort_key() {
        let e1 = MemoryEvent::new(0x1000, 5, 42, true);
        let e2 = MemoryEvent::new(0x1000, 10, 42, false);
        let e3 = MemoryEvent::new(0x2000, 1, 99, true);

        assert!(e1.sort_key() < e2.sort_key()); // Same addr, e1.clk < e2.clk
        assert!(e2.sort_key() < e3.sort_key()); // e2.addr < e3.addr
    }

    #[test]
    fn comparator_verify_sorted_valid() {
        let events = vec![
            MemoryEvent::new(0x1000, 1, 42, true),
            MemoryEvent::new(0x1000, 5, 42, false),
            MemoryEvent::new(0x2000, 3, 99, true),
        ];

        assert!(comparator::verify_sorted(&events));
    }

    #[test]
    fn comparator_verify_sorted_invalid() {
        let events = vec![
            MemoryEvent::new(0x2000, 3, 99, true),
            MemoryEvent::new(0x1000, 1, 42, true), // Out of order!
        ];

        assert!(!comparator::verify_sorted(&events));
    }

    #[test]
    fn comparator_verify_consistency_valid() {
        let events = vec![
            MemoryEvent::new(0x1000, 1, 42, true),  // Write 42
            MemoryEvent::new(0x1000, 5, 42, false), // Read 42 (consistent)
            MemoryEvent::new(0x2000, 3, 99, true),  // Different addr
        ];

        assert!(comparator::verify_consistency(&events));
    }

    #[test]
    fn comparator_verify_consistency_invalid() {
        let events = vec![
            MemoryEvent::new(0x1000, 1, 42, true),  // Write 42
            MemoryEvent::new(0x1000, 5, 99, false), // Read 99 (INCONSISTENT!)
        ];

        assert!(!comparator::verify_consistency(&events));
    }

    #[test]
    fn generate_ram_trace_basic() {
        let events = vec![
            MemoryEvent::new(0x1000, 1, 42, true),
            MemoryEvent::new(0x1004, 2, 99, true),
        ];

        let trace = generate_ram_trace(&events, 8).unwrap();

        assert_eq!(trace.num_cols(), NUM_PHYSICAL_COLUMNS);
        assert_eq!(trace.num_rows().unwrap(), 8);

        // Physical layout:
        // use PHY_* indices.
        let addr_b0 = trace.columns[PHY_ADDR_B0].as_b32_slice().unwrap();
        let addr_b1 = trace.columns[PHY_ADDR_B0 + 1].as_b32_slice().unwrap();
        let val_b0 = trace.columns[PHY_VAL_B0].as_b32_slice().unwrap();
        let is_write = trace.columns[PHY_IS_WRITE].as_bit_slice().unwrap();
        let selector = trace.columns[PHY_SELECTOR].as_bit_slice().unwrap();

        // Check first event
        // NOTE:
        // B32 columns are stored in hardware/flat basis;
        // convert back to tower for semantic checks.
        assert_eq!(addr_b0[0].to_tower(), Block32::from(0x00u32));
        assert_eq!(addr_b1[0].to_tower(), Block32::from(0x10u32));
        assert_eq!(val_b0[0].to_tower(), Block32::from(42u32));
        assert_eq!(is_write[0], Bit::ONE);
        assert_eq!(selector[0], Bit::ONE);

        // Check padding row
        assert_eq!(selector[7], Bit::ZERO);
    }

    #[test]
    fn generate_ram_trace_consistency_violation() {
        let events = vec![
            MemoryEvent::new(0x1000, 1, 42, true),
            MemoryEvent::new(0x1000, 5, 99, false),
        ];

        let result = generate_ram_trace(&events, 8);
        assert!(result.is_err());
    }

    /// Test:
    /// Memory Consistency Enforcement (Trace Generation Logic).
    ///
    /// Ensures the `generate_ram_trace` helper
    /// enforces read-after-write consistency.
    #[test]
    fn memory_consistency() {
        let valid_events = vec![
            MemoryEvent::write(0x1000, 1, 42),
            MemoryEvent::read(0x1000, 2, 42),
        ];

        let trace = generate_ram_trace(&valid_events, 8).unwrap();
        assert_eq!(trace.num_cols(), NUM_PHYSICAL_COLUMNS);
    }

    /// Test:
    /// Sorting Verification.
    ///
    /// Ensures RAM trace events are
    /// sorted by (addr || clk).
    #[test]
    fn sorting_by_addr_clk() {
        let events = vec![
            MemoryEvent::write(0x2000, 1, 99),
            MemoryEvent::write(0x1000, 0, 42),
            MemoryEvent::write(0x1000, 5, 42),
            MemoryEvent::write(0x1004, 3, 123),
        ];

        let trace = generate_ram_trace(&events, 8).unwrap();

        let addr_b0 = trace.columns[PHY_ADDR_B0].as_b32_slice().unwrap();
        let addr_b1 = trace.columns[PHY_ADDR_B0 + 1].as_b32_slice().unwrap();
        let clk_b0 = trace.columns[PHY_CLK_B0].as_b32_slice().unwrap();

        // 1st: 0x1000, clk 0
        assert_eq!(addr_b0[0].to_tower(), Block32::from(0x00u32));
        assert_eq!(addr_b1[0].to_tower(), Block32::from(0x10u32));
        assert_eq!(clk_b0[0].to_tower(), Block32::from(0x00u32));

        // 2nd: 0x1000, clk 5
        assert_eq!(addr_b0[1].to_tower(), Block32::from(0x00u32));
        assert_eq!(addr_b1[1].to_tower(), Block32::from(0x10u32));
        assert_eq!(clk_b0[1].to_tower(), Block32::from(0x05u32));

        // 3rd: 0x1004, clk 3
        assert_eq!(addr_b0[2].to_tower(), Block32::from(0x04u32));
        assert_eq!(addr_b1[2].to_tower(), Block32::from(0x10u32));
        assert_eq!(clk_b0[2].to_tower(), Block32::from(0x03u32));

        // 4th: 0x2000, clk 1
        assert_eq!(addr_b0[3].to_tower(), Block32::from(0x00u32));
        assert_eq!(addr_b1[3].to_tower(), Block32::from(0x20u32));
        assert_eq!(clk_b0[3].to_tower(), Block32::from(0x01u32));
    }

    /// Test:
    /// Spec Verification, CPU uses Virtual Clock.
    #[test]
    fn cpu_uses_row_index_for_clock() {
        let cpu_spec = CpuMemoryUnit::linking_spec();

        // CPU spec should have 13 sources
        assert_eq!(cpu_spec.num_sources(), 13);

        // Verify sources 4-7 are
        // RowIndexByte (virtual bytes).
        for i in 4..8 {
            match &cpu_spec.sources[i].0 {
                Source::RowIndexByte(_) => {}
                _ => panic!("CPU spec should use RowIndexByte for clock at index {}", i),
            }
        }
    }

    /// Test:
    /// Spec Verification,
    /// RAM uses Committed Clock.
    #[test]
    fn ram_uses_committed_columns_for_clock() {
        let ram_spec = RamChiplet::linking_spec();

        assert_eq!(ram_spec.num_sources(), 13);

        // Verify sources 4-7 are Columns
        for i in 4..8 {
            match &ram_spec.sources[i].0 {
                Source::Column(_) => {}
                _ => panic!("RAM spec should use Column for clock at index {}", i),
            }
        }
    }

    #[test]
    fn comparator_verify_consistency_write_after_write() {
        let events = vec![
            MemoryEvent::new(0x1000, 1, 42, true),  // Write 42
            MemoryEvent::new(0x1000, 2, 99, true),  // Write 99 (Same Addr, New Val - OK)
            MemoryEvent::new(0x1000, 3, 99, false), // Read 99 (Consistent)
        ];

        // This ensures the Write-after-Write pattern is valid
        assert!(comparator::verify_consistency(&events));
    }

    #[test]
    fn comparator_verify_consistency_read_after_write_invalid() {
        let events = vec![
            MemoryEvent::new(0x1000, 1, 42, true), // Write 42
            // Next op is READ (is_write=false).
            // The value MUST remain 42.
            // Here we simulate corruption (99).
            MemoryEvent::new(0x1000, 2, 99, false),
        ];

        // This ensures that reading a different value
        // without an intermediate write is invalid.
        // Returns false = corruption detected.
        assert!(
            !comparator::verify_consistency(&events),
            "Memory corruption detected! Read different value after Write without intermediate Write."
        );
    }

    #[test]
    fn consistency_read_after_write_gap_invalid() {
        let mut events = vec![
            // Row 1:
            // Write 42 (Addr 0x10, Clk 1)
            MemoryEvent::new(0x10, 1, 42, true),
            // Row 2:
            // Read 0 (Addr 0x11, Clk 2)
            MemoryEvent::new(0x11, 2, 0, false),
            // Row 3:
            // Read 99  (Addr 0x10, Clk 3) - CORRUPTION!
            MemoryEvent::new(0x10, 3, 99, false),
        ];

        // Sort the events as the chiplet
        // does before generating the trace.
        events.sort_by_key(|e| e.sort_key());

        // After sorting by (Addr, Clk), the trace becomes:
        // 1. (0x10, 1, 42, Write)
        // 2. (0x10, 3, 99, Read)  <-- AIR will check this pair. D=0, is_write_next=0.
        // 3. (0x11, 2, 0, Read)

        assert!(
            !comparator::verify_consistency(&events),
            "Gap consistency failed to catch corruption!"
        );
    }

    #[test]
    fn time_travel_attack_invalid() {
        // Scenario 1: Duplicate (Addr, Clk)
        // This is the most dangerous attack.
        // Even after sorting, identical keys
        // MUST fail verify_sorted because of
        // the >= check. This targets the core
        // reason for strict monotonicity in the AIR.
        let events = vec![
            MemoryEvent::new(0x10, 100, 42, true),
            MemoryEvent::new(0x10, 100, 99, false), // Duplicate Clk 100!
        ];

        // 1. Auditor must reject duplicates
        let mut sorted = events.clone();
        sorted.sort_by_key(|e| e.sort_key());

        assert!(!comparator::verify_sorted(&sorted));

        // 2. Trace generation must return error
        let result = generate_ram_trace(&events, 8);
        assert!(result.is_err());
    }

    #[test]
    fn lexicographical_order_priority() {
        // Scenario 2:
        // Ensure Addr (MSB) has absolute priority
        // over Clk. This tests that our byte_indices
        // [Addr_B3..B0, Clk_B3..B0] correctly enforce
        // the sorting order used by the Range gadget.
        let e_low_addr = MemoryEvent::new(0x10, 0xFFFFFFFF, 0, false);
        let e_high_addr = MemoryEvent::new(0x11, 0, 0, false);

        assert!(e_low_addr.sort_key() < e_high_addr.sort_key());

        let mut events = vec![e_high_addr, e_low_addr];
        events.sort_by_key(|e| e.sort_key());

        // Addr 0x10 must be Row 0,
        // despite having a massive Clock value.
        assert_eq!(events[0].addr, 0x10);
        assert_eq!(events[1].addr, 0x11);
        assert!(comparator::verify_sorted(&events));
    }

    #[test]
    fn initial_read_vulnerability_test() {
        // Attack:
        // The first operation on address
        // 0x99 is a Read of 1000 BTC.
        let events = vec![MemoryEvent::new(0x99, 1, 1000, false)];

        // Auditor must detect that uninitialized
        // memory is read as non-zero.
        assert!(
            !comparator::verify_consistency(&events),
            "Initial read vulnerability not caught!"
        );
    }
}
