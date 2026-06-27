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

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::cmp;
use hekate_core::errors::Error;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder, TraceCompatibleField};
use hekate_math::{Bit, Block64, TowerField};
use hekate_program::Air;
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::define_columns;
use hekate_program::expander::VirtualExpander;
use hekate_program::permutation::{PermutationCheckSpec, REQUEST_IDX_LABEL, Source};
use once_cell::race::OnceBox;

// FIPS 202 sponge parameters.
const SHA3_DOMAIN_SEP: u8 = 0x06;
const SHAKE_DOMAIN_SEP: u8 = 0x1f;

const SHA3_256_RATE: usize = 136; // (1600 - 512) / 8
const SHA3_512_RATE: usize = 72; // (1600 - 1024) / 8
const SHAKE128_RATE: usize = 168; // (1600 - 256) / 8
const SHAKE256_RATE: usize = 136; // (1600 - 512) / 8

// Physical column indices for Keccak trace.
// Distinct from virtual KeccakColumns
// (1692 cols after bit-unpacking).
const PHYS_LANES: usize = 0; // 0..24: B64 state lanes
const PHYS_RC: usize = 25; // B64 round constant
const PHYS_REQUEST_IDX: usize = 26; // B32 partner-side row index
const PHYS_S_ROUND: usize = 27; // Bit: active round
const PHYS_S_IN_OUT: usize = 28; // Bit: input/output row
const PHYS_NUM_COLS: usize = 29;

/// Shared labels for Keccak bus linking.
pub const KECCAK_LANE_LABELS: [&[u8]; 25] = [
    b"keccak_lane_0",
    b"keccak_lane_1",
    b"keccak_lane_2",
    b"keccak_lane_3",
    b"keccak_lane_4",
    b"keccak_lane_5",
    b"keccak_lane_6",
    b"keccak_lane_7",
    b"keccak_lane_8",
    b"keccak_lane_9",
    b"keccak_lane_10",
    b"keccak_lane_11",
    b"keccak_lane_12",
    b"keccak_lane_13",
    b"keccak_lane_14",
    b"keccak_lane_15",
    b"keccak_lane_16",
    b"keccak_lane_17",
    b"keccak_lane_18",
    b"keccak_lane_19",
    b"keccak_lane_20",
    b"keccak_lane_21",
    b"keccak_lane_22",
    b"keccak_lane_23",
    b"keccak_lane_24",
];

define_columns! {
    pub KeccakColumns {
        STATE_BITS: [Bit; 1600],
        RC_BITS: [Bit; 64],
        LANES: [B64; 25],
        REQUEST_IDX: B32,
        S_ROUND: Bit,
        S_IN_OUT: Bit,
    }
}

define_columns! {
    pub CpuKeccakColumns {
        LANES: [B64; 25],
        SELECTOR: Bit,
    }
}

/// CPU Interface for Keccak.
///
/// Represents the CPU side of the Keccak bus.
/// The CPU provides/consumes 25 lanes (u64 packed)
/// when the selector is active.
#[derive(Clone, Debug)]
pub struct CpuKeccakUnit;

impl CpuKeccakUnit {
    /// Returns the permutation check
    /// specification for CPU Keccak side.
    pub fn linking_spec() -> PermutationCheckSpec {
        let mut sources = Vec::with_capacity(26);
        for (i, label) in KECCAK_LANE_LABELS.iter().enumerate() {
            let col_idx = CpuKeccakColumns::LANES + i;
            sources.push((Source::Column(col_idx), *label));
        }

        sources.push((Source::RowIndexLeBytes(4), REQUEST_IDX_LABEL));

        PermutationCheckSpec::new(sources, Some(CpuKeccakColumns::SELECTOR))
    }

    pub fn num_columns(&self) -> usize {
        CpuKeccakColumns::NUM_COLUMNS
    }
}

/// Keccak-f[1600] Permutation Chiplet.
///
/// # Layout
/// - 1 Row = 1 Round of Keccak-f.
/// - 24 Rows = 1 Full Permutation.
/// - State is stored as 1600 "Bit" columns (to allow native degree-2 algebraic constraints).
/// - I/O is stored as 25 "Block64" columns (to link with CPU via GPA).
#[derive(Clone, Debug)]
pub struct KeccakChiplet {
    pub num_rows: usize,
}

impl KeccakChiplet {
    pub const BUS_ID: &'static str = "keccak_link";

    /// The number of physical bytes occupied
    /// by the Keccak Chiplet in a single row.
    /// Layout:
    /// 25 Lanes (u64) + 1 RC (u64) + 2 Selectors (u8) = 200 + 8 + 2 = 210 bytes.
    pub const PHYSICAL_ROW_BYTES: usize = 25 * 8 + 8 + 2;

    /// Rotation offsets for Rho step (x, y) -> r
    pub const RHO_OFFSETS: [[usize; 5]; 5] = [
        [0, 36, 3, 41, 18],
        [1, 44, 10, 45, 2],
        [62, 6, 43, 15, 61],
        [28, 55, 25, 21, 56],
        [27, 20, 39, 8, 14],
    ];

    /// Round constants for Keccak-f[1600].
    pub const ROUND_CONSTANTS: [u64; 24] = [
        0x0000000000000001,
        0x0000000000008082,
        0x800000000000808a,
        0x8000000080008000,
        0x000000000000808b,
        0x0000000080000001,
        0x8000000080008081,
        0x8000000000008009,
        0x000000000000008a,
        0x0000000000000088,
        0x0000000080008009,
        0x000000008000000a,
        0x000000008000808b,
        0x800000000000008b,
        0x8000000000008089,
        0x8000000000008003,
        0x8000000000008002,
        0x8000000000000080,
        0x000000000000800a,
        0x800000008000000a,
        0x8000000080008081,
        0x8000000000008080,
        0x0000000080000001,
        0x8000000080008008,
    ];

    pub fn new(num_rows: usize) -> Self {
        assert!(num_rows.is_power_of_two());
        Self { num_rows }
    }

    /// Construct for constraint-only use
    /// (constraints/constraint_ast).
    /// No trace size needed, constraint
    /// generation is size-independent.
    pub fn for_constraints() -> Self {
        Self { num_rows: 0 }
    }

    #[inline(always)]
    pub fn get_bit_col(x: usize, y: usize, z: usize) -> usize {
        let lane_idx = y * 5 + x;
        KeccakColumns::STATE_BITS + lane_idx * 64 + z
    }

    #[inline(always)]
    pub fn get_rc_col(z: usize) -> usize {
        KeccakColumns::RC_BITS + z
    }

    #[inline(always)]
    pub fn get_lane_col(x: usize, y: usize) -> usize {
        KeccakColumns::LANES + (y * 5 + x)
    }

    /// Linking specification. Defines how the
    /// 25 "Lane" columns match the Bus.
    pub fn linking_spec() -> PermutationCheckSpec {
        let mut sources = Vec::with_capacity(26);
        for y in 0..5 {
            for x in 0..5 {
                let lane_idx = y * 5 + x;
                let col_idx = KeccakColumns::LANES + lane_idx;
                let label = KECCAK_LANE_LABELS[lane_idx];

                sources.push((Source::Column(col_idx), label));
            }
        }

        sources.push((
            Source::Column(KeccakColumns::REQUEST_IDX),
            REQUEST_IDX_LABEL,
        ));

        PermutationCheckSpec::new(sources, Some(KeccakColumns::S_IN_OUT))
    }
}

impl<F: TowerField + TraceCompatibleField> Air<F> for KeccakChiplet {
    fn name(&self) -> String {
        "KeccakChiplet".to_string()
    }

    fn column_layout(&self) -> &[ColumnType] {
        // Physical layout:
        // 25 B64 lanes
        //   + 1 B64 RC
        //   + 1 B32 request_idx
        //   + 2 Bit selectors
        static PHYSICAL_LAYOUT: OnceBox<Vec<ColumnType>> = OnceBox::new();
        PHYSICAL_LAYOUT.get_or_init(|| {
            let mut cols = Vec::with_capacity(PHYS_NUM_COLS);
            cols.extend(vec![ColumnType::B64; 26]); // 25 lanes + RC
            cols.push(ColumnType::B32); // request_idx
            cols.extend(vec![ColumnType::Bit; 2]); // s_round, s_in_out

            Box::new(cols)
        })
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![(Self::BUS_ID.into(), Self::linking_spec())]
    }

    fn virtual_expander(&self) -> Option<&VirtualExpander> {
        static E: OnceBox<VirtualExpander> = OnceBox::new();
        Some(E.get_or_init(|| {
            Box::new(
                VirtualExpander::new()
                    .expand_bits(25, ColumnType::B64) // 25 lanes -> 1600 bits
                    .expand_bits(1, ColumnType::B64) // RC -> 64 bits
                    .reuse_pass_through(0, 25) // 25 lanes as IO B64
                    .pass_through(1, ColumnType::B32) // request_idx
                    .control_bits(2) // s_round, s_in_out
                    .build()
                    .expect("KeccakChiplet expander"),
            )
        }))
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();

        let s_round = cs.col(KeccakColumns::S_ROUND);
        let s_in_out = cs.col(KeccakColumns::S_IN_OUT);

        // 1. Packing:
        // lane = Σ bit[z] * 2^z, gated by s_in_out
        for y in 0..5 {
            for x in 0..5 {
                let lane = cs.col(Self::get_lane_col(x, y));
                let bits: Vec<_> = (0..64)
                    .map(|z| cs.scale(F::from(1u128 << z), cs.col(Self::get_bit_col(x, y, z))))
                    .collect();

                cs.assert_zero_when(s_in_out, lane + cs.sum(&bits));
            }
        }

        // 2. Column parity (CSE level 1):
        // C[x,z] = Σ_{y=0..4} A[x,y,z]
        let col_parity: [[_; 64]; 5] = core::array::from_fn(|x| {
            core::array::from_fn(|z| {
                cs.sum(
                    &(0..5)
                        .map(|y| cs.col(Self::get_bit_col(x, y, z)))
                        .collect::<Vec<_>>(),
                )
            })
        });

        // 3. Theta (CSE level 2):
        // Theta(x,y,z) = A[x,y,z] + C[x-1,z] + C[x+1,z-1]
        let theta: [[[_; 64]; 5]; 5] = core::array::from_fn(|x| {
            core::array::from_fn(|y| {
                core::array::from_fn(|z| {
                    let self_bit = cs.col(Self::get_bit_col(x, y, z));
                    let c_prev = col_parity[(x + 4) % 5][z];
                    let c_next = col_parity[(x + 1) % 5][(z + 63) % 64];

                    cs.sum(&[self_bit, c_prev, c_next])
                })
            })
        });

        // Helper:
        // B[x,y,z] via inverse Pi + inverse Rho → Theta
        let get_b = |out_x: usize, out_y: usize, out_z: usize| {
            let in_x = (out_x + 3 * out_y) % 5;
            let in_y = out_x;
            let rot = Self::RHO_OFFSETS[in_x][in_y];
            let in_z = (out_z + 64 - rot) % 64;

            theta[in_x][in_y][in_z]
        };

        // 4. Round constraints:
        // Chi + Iota
        for x in 0..5 {
            for y in 0..5 {
                for z in 0..64 {
                    let b_curr = get_b(x, y, z);
                    let b_next1 = get_b((x + 1) % 5, y, z);
                    let b_next2 = get_b((x + 2) % 5, y, z);

                    // Chi = b_curr + b_next2 + b_next1 * b_next2
                    let chi = cs.sum(&[b_curr, b_next2, b_next1 * b_next2]);
                    let next_bit = cs.next(Self::get_bit_col(x, y, z));

                    if x == 0 && y == 0 {
                        let rc = cs.col(Self::get_rc_col(z));
                        cs.assert_zero_when(s_round, cs.sum(&[next_bit, chi, rc]));
                    } else {
                        cs.assert_zero_when(s_round, next_bit + chi);
                    }
                }
            }
        }

        // 5. Ghost Protocol:
        // Continuity, round row must be followed
        // by round or I/O. Prevents mid-chain
        // s_round deactivation.
        cs.assert_boolean(s_round);
        cs.assert_boolean(s_in_out);

        let next_s_round = cs.next(KeccakColumns::S_ROUND);
        let next_s_in_out = cs.next(KeccakColumns::S_IN_OUT);
        let one = cs.constant(F::ONE);

        cs.constrain(s_round * (one + next_s_round + next_s_in_out));

        // Output binding, row before any
        // output must be a round row.
        // MLE wrap (row 31->0) forces
        // s_round=1 on input row.
        cs.constrain(next_s_in_out * (one + next_s_round) * (one + s_round));

        cs.build()
    }
}

pub struct KeccakWitness;

impl KeccakWitness {
    /// Performs one Keccak-f[1600] round in pure Rust.
    #[inline(always)]
    pub fn keccak_f_round(mut a: [u64; 25], rc: u64) -> [u64; 25] {
        // Theta
        let mut c = [0u64; 5];
        for x in 0..5 {
            c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
        }

        let mut d = [0u64; 5];
        for x in 0..5 {
            d[x] = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
        }

        for i in 0..25 {
            a[i] ^= d[i % 5];
        }

        // Rho & Pi
        let mut b = [0u64; 25];
        for y in 0..5 {
            for x in 0..5 {
                let rot = KeccakChiplet::RHO_OFFSETS[x][y] as u32;
                b[((2 * x + 3 * y) % 5) * 5 + y] = a[y * 5 + x].rotate_left(rot);
            }
        }

        // Chi
        for y in 0..5 {
            for x in 0..5 {
                a[y * 5 + x] = b[y * 5 + x] ^ ((!b[y * 5 + (x + 1) % 5]) & b[y * 5 + (x + 2) % 5]);
            }
        }

        // Iota
        a[0] ^= rc;

        a
    }

    /// Assigns a full 24-round
    /// permutation to the trace (Row-by-Row).
    ///
    /// # Note
    /// This method is slower than
    /// `KeccakSpongeNative::generate_trace`
    /// and is intended primarily for testing
    /// and manual trace construction where
    /// batching is not applicable.
    pub fn assign_permutation<F: TowerField>(
        trace: &mut [Vec<F>],
        start_row: usize,
        mut state: [u64; 25],
    ) -> [u64; 25] {
        for round in 0..24 {
            let row = start_row + round;
            let rc = KeccakChiplet::ROUND_CONSTANTS[round];

            trace[KeccakColumns::S_ROUND][row] = F::ONE;

            if round == 0 {
                trace[KeccakColumns::S_IN_OUT][row] = F::ONE;
            }

            Self::assign_state_at_row(trace, row, state);
            Self::assign_rc_at_row(trace, row, rc);

            state = Self::keccak_f_round(state, rc);
        }

        let final_row = start_row + 24;
        trace[KeccakColumns::S_IN_OUT][final_row] = F::ONE;

        Self::assign_state_at_row(trace, final_row, state);

        state
    }

    fn assign_state_at_row<F: TowerField>(trace: &mut [Vec<F>], row: usize, state: [u64; 25]) {
        for y in 0..5 {
            for x in 0..5 {
                let lane_val = state[y * 5 + x];
                trace[KeccakChiplet::get_lane_col(x, y)][row] = F::from(lane_val as u128);

                for z in 0..64 {
                    let bit_val = (lane_val >> z) & 1;
                    let bit_col = KeccakChiplet::get_bit_col(x, y, z);
                    trace[bit_col][row] = if bit_val == 1 { F::ONE } else { F::ZERO };
                }
            }
        }
    }

    fn assign_rc_at_row<F: TowerField>(trace: &mut [Vec<F>], row: usize, rc: u64) {
        for z in 0..64 {
            let rc_bit = (rc >> z) & 1;
            trace[KeccakChiplet::get_rc_col(z)][row] = if rc_bit == 1 { F::ONE } else { F::ZERO };
        }
    }
}

/// Generates Keccak chiplet trace from
/// raw keccak-f[1600] permutation calls.
///
/// Each call processes one full permutation
/// (25 rows: 24 rounds + 1 output).
/// Returns a `ColumnTrace` with 28 physical
/// columns (same layout as sponge mode).
///
/// `request_idx_pairs = None` defaults to
/// `(25k, 25k+24)` to match CPU-side row layout
/// (input row 25k, output row 25k+24); pass `Some`
/// for non-default partner emit positions.
///
/// Use this instead of `KeccakSpongeNative::generate_trace`
/// when the caller controls the input states directly
/// (e.g., Merkle tree hashing).
pub fn generate_keccak_trace(
    calls: &[[Block64; 25]],
    request_idx_pairs: Option<&[(u32, u32)]>,
    num_rows: usize,
) -> hekate_core::errors::Result<ColumnTrace> {
    let default_pairs: Vec<(u32, u32)> = match request_idx_pairs {
        Some(_) => Vec::new(),
        None => (0..calls.len() as u32)
            .map(|k| (25 * k, 25 * k + 24))
            .collect(),
    };

    let pairs: &[(u32, u32)] = request_idx_pairs.unwrap_or(&default_pairs);

    if pairs.len() != calls.len() {
        return Err(Error::Protocol {
            protocol: "keccak",
            message: "request_idx_pairs length must match calls length",
        });
    }

    #[derive(Clone, Copy)]
    struct RowData {
        state: [u64; 25],
        rc: u64,
        s_round: bool,
        s_in_out: bool,
        request_idx: u32,
    }

    // 1. Compute Pass
    let mut rows = Vec::with_capacity(num_rows);

    for (call, &(in_idx, out_idx)) in calls.iter().zip(pairs.iter()) {
        if rows.len() + 25 > num_rows {
            return Err(Error::Protocol {
                protocol: "keccak",
                message: "trace overflow: too many calls for allocated rows",
            });
        }

        let mut state = [0u64; 25];
        for i in 0..25 {
            state[i] = call[i].0;
        }

        for round in 0..24 {
            let rc = KeccakChiplet::ROUND_CONSTANTS[round];
            rows.push(RowData {
                state,
                rc,
                s_round: true,
                s_in_out: round == 0,
                request_idx: if round == 0 { in_idx } else { 0 },
            });

            state = KeccakWitness::keccak_f_round(state, rc);
        }

        rows.push(RowData {
            state,
            rc: 0,
            s_round: false,
            s_in_out: true,
            request_idx: out_idx,
        });
    }

    // TraceBuilder zero-fills padding,
    // only iterate active rows.

    // 2. Fill Pass
    let num_vars = num_rows.trailing_zeros() as usize;
    let mut layout = Vec::with_capacity(PHYS_NUM_COLS);

    for _ in 0..PHYS_REQUEST_IDX {
        layout.push(ColumnType::B64);
    }

    layout.push(ColumnType::B32);
    layout.push(ColumnType::Bit);
    layout.push(ColumnType::Bit);

    let mut tb = TraceBuilder::new(&layout, num_vars)?;

    for (i, row) in rows.iter().enumerate() {
        for lane in 0..25 {
            tb.set_b64(PHYS_LANES + lane, i, Block64::from(row.state[lane]))?;
        }

        tb.set_b64(PHYS_RC, i, Block64::from(row.rc))?;
        tb.set_b32(
            PHYS_REQUEST_IDX,
            i,
            hekate_math::Block32::from(row.request_idx),
        )?;
        tb.set_bit(
            PHYS_S_ROUND,
            i,
            if row.s_round { Bit::ONE } else { Bit::ZERO },
        )?;
        tb.set_bit(
            PHYS_S_IN_OUT,
            i,
            if row.s_in_out { Bit::ONE } else { Bit::ZERO },
        )?;
    }

    Ok(tb.build())
}

// =================================================================
// Native Keccak Sponge
// =================================================================

/// (input_state, output_state) pair
/// per Keccak-f permutation call.
/// Used for chiplet trace generation.
pub type KeccakCall = ([u64; 25], [u64; 25]);

/// Native Keccak sponge with traced
/// permutation calls for chiplet witness.
///
/// Records every Keccak-f (input, output)
/// state pair so the Keccak chiplet can
/// reproduce the trace.
pub struct KeccakSpongeNative {
    state: [u64; 25],
    permutation_calls: Vec<KeccakCall>,
}

impl Default for KeccakSpongeNative {
    fn default() -> Self {
        Self::new()
    }
}

impl KeccakSpongeNative {
    pub fn new() -> Self {
        Self {
            state: [0u64; 25],
            permutation_calls: Vec::new(),
        }
    }

    /// Absorb one rate-sized block and permute.
    pub fn absorb_block(&mut self, block: &[u8], rate_bytes: usize) {
        let rate_lanes = rate_bytes / 8;
        for i in 0..rate_lanes {
            if i * 8 + 8 <= block.len() {
                let lane = u64::from_le_bytes(block[i * 8..i * 8 + 8].try_into().unwrap());
                self.state[i] ^= lane;
            } else if i * 8 < block.len() {
                let mut buf = [0u8; 8];
                let end = cmp::min(block.len(), i * 8 + 8);
                buf[..end - i * 8].copy_from_slice(&block[i * 8..end]);

                self.state[i] ^= u64::from_le_bytes(buf);
            }
        }

        let input = self.state;
        keccak_f(&mut self.state);

        self.permutation_calls.push((input, self.state));
    }

    /// Full absorb with domain-separated padding.
    pub fn absorb(&mut self, msg: &[u8], rate_bytes: usize, domain_sep: u8) {
        let mut offset = 0;
        while offset + rate_bytes <= msg.len() {
            self.absorb_block(&msg[offset..offset + rate_bytes], rate_bytes);
            offset += rate_bytes;
        }

        let mut last = vec![0u8; rate_bytes];
        let remaining = msg.len() - offset;

        last[..remaining].copy_from_slice(&msg[offset..]);
        last[remaining] = domain_sep;
        last[rate_bytes - 1] |= 0x80;

        self.absorb_block(&last, rate_bytes);
    }

    /// Squeeze `out_len` bytes from the state.
    pub fn squeeze(&mut self, out_len: usize, rate_bytes: usize) -> Vec<u8> {
        let mut output = Vec::with_capacity(out_len);
        let rate_lanes = rate_bytes / 8;

        loop {
            for i in 0..rate_lanes {
                let bytes = self.state[i].to_le_bytes();
                for &b in &bytes {
                    if output.len() < out_len {
                        output.push(b);
                    }
                }
            }

            if output.len() >= out_len {
                break;
            }

            let input = self.state;
            keccak_f(&mut self.state);

            self.permutation_calls.push((input, self.state));
        }

        output.truncate(out_len);

        output
    }

    /// Consume the sponge and return
    /// all recorded permutation calls.
    pub fn into_calls(self) -> Vec<KeccakCall> {
        self.permutation_calls
    }

    /// Consume the sponge and build trace
    /// from the recorded permutation calls.
    pub fn generate_trace(
        self,
        request_idx_pairs: Option<&[(u32, u32)]>,
        num_rows: usize,
    ) -> hekate_core::errors::Result<ColumnTrace> {
        let calls: Vec<[Block64; 25]> = self
            .into_calls()
            .iter()
            .map(|(input, _)| {
                let mut block = [Block64::ZERO; 25];
                for (i, &lane) in input.iter().enumerate() {
                    block[i] = Block64::from(lane);
                }

                block
            })
            .collect();

        generate_keccak_trace(&calls, request_idx_pairs, num_rows)
    }
}

/// SHA3-256
pub fn sha3_256(msg: &[u8]) -> ([u8; 32], Vec<KeccakCall>) {
    let mut sponge = KeccakSpongeNative::new();
    sponge.absorb(msg, SHA3_256_RATE, SHA3_DOMAIN_SEP);

    let out = sponge.squeeze(32, SHA3_256_RATE);

    let mut hash = [0u8; 32];
    hash.copy_from_slice(&out);

    (hash, sponge.into_calls())
}

/// SHA3-512
pub fn sha3_512(msg: &[u8]) -> ([u8; 64], Vec<KeccakCall>) {
    let mut sponge = KeccakSpongeNative::new();
    sponge.absorb(msg, SHA3_512_RATE, SHA3_DOMAIN_SEP);

    let out = sponge.squeeze(64, SHA3_512_RATE);

    let mut hash = [0u8; 64];
    hash.copy_from_slice(&out);

    (hash, sponge.into_calls())
}

/// SHAKE-128
pub fn shake128(msg: &[u8], out_len: usize) -> (Vec<u8>, Vec<KeccakCall>) {
    let mut sponge = KeccakSpongeNative::new();
    sponge.absorb(msg, SHAKE128_RATE, SHAKE_DOMAIN_SEP);

    let out = sponge.squeeze(out_len, SHAKE128_RATE);

    (out, sponge.into_calls())
}

/// SHAKE-256
pub fn shake256(msg: &[u8], out_len: usize) -> (Vec<u8>, Vec<KeccakCall>) {
    let mut sponge = KeccakSpongeNative::new();
    sponge.absorb(msg, SHAKE256_RATE, SHAKE_DOMAIN_SEP);

    let out = sponge.squeeze(out_len, SHAKE256_RATE);

    (out, sponge.into_calls())
}

/// Apply full Keccak-f (24 rounds).
fn keccak_f(state: &mut [u64; 25]) {
    for &rc in &KeccakChiplet::ROUND_CONSTANTS {
        *state = KeccakWitness::keccak_f_round(*state, rc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hekate_math::Block128;
    use hekate_program::constraint::{ConstraintExpr, ExprId};

    type F = Block128;

    fn keccak_trace(msg: &[u8], num_rows: usize) -> hekate_core::errors::Result<ColumnTrace> {
        let mut sponge = KeccakSpongeNative::new();
        sponge.absorb(msg, 136, 0x01);

        sponge.generate_trace(None, num_rows)
    }

    #[test]
    fn keccak_layout_from_schema() {
        let layout = KeccakColumns::build_layout();
        assert_eq!(KeccakColumns::NUM_COLUMNS, 1692);
        assert_eq!(layout.len(), 1692);
        assert_eq!(layout[0], ColumnType::Bit);
        assert_eq!(layout[KeccakColumns::LANES], ColumnType::B64);
    }

    #[test]
    fn keccak_chiplet_air_metadata() {
        let chiplet = KeccakChiplet::new(32);
        assert_eq!(Air::<F>::num_columns(&chiplet), 1692);
        assert_eq!(Air::<F>::name(&chiplet), "KeccakChiplet".to_string());
    }

    #[test]
    fn keccak_round_function_zero_input() {
        // Keccak-f round with zero state should
        // only be affected by RC in A[0,0].
        let state = [0u64; 25];
        let rc = 0x800000000000808au64;
        let next_state = KeccakWitness::keccak_f_round(state, rc);

        assert_eq!(next_state[0], rc);

        for &v in next_state.iter().skip(1) {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn witness_assignment_boundaries() {
        let num_rows = 32;
        let mut trace = vec![vec![F::ZERO; num_rows]; KeccakColumns::NUM_COLUMNS];

        let initial_state = [0xAAu64; 25];
        let final_state = KeccakWitness::assign_permutation(&mut trace, 0, initial_state);

        // Check Input boundary (Row 0)
        assert_eq!(trace[KeccakColumns::S_IN_OUT][0], F::ONE);
        assert_eq!(trace[KeccakColumns::S_ROUND][0], F::ONE);
        assert_eq!(
            trace[KeccakChiplet::get_lane_col(0, 0)][0],
            F::from(0xAAu128)
        );

        // Check Round Transition (Row 23)
        assert_eq!(trace[KeccakColumns::S_ROUND][23], F::ONE);
        assert_eq!(trace[KeccakColumns::S_IN_OUT][23], F::ZERO);

        // Check Output boundary (Row 24)
        assert_eq!(trace[KeccakColumns::S_IN_OUT][24], F::ONE);
        assert_eq!(trace[KeccakColumns::S_ROUND][24], F::ZERO);
        assert_eq!(
            trace[KeccakChiplet::get_lane_col(0, 0)][24],
            F::from(final_state[0] as u128)
        );
    }

    #[test]
    fn bit_decomposition_consistency() {
        let mut trace = vec![vec![F::ZERO; 32]; KeccakColumns::NUM_COLUMNS];
        let state = [0x123456789ABCDEF0u64; 25];

        KeccakWitness::assign_permutation(&mut trace, 0, state);

        // Check Row 0, Lane (0,0)
        let lane_val = 0x123456789ABCDEF0u64;
        for z in 0..64 {
            let bit = (lane_val >> z) & 1;
            let expected = if bit == 1 { F::ONE } else { F::ZERO };
            assert_eq!(trace[KeccakChiplet::get_bit_col(0, 0, z)][0], expected);
        }
    }

    #[test]
    fn keccak_f_round_all_ones() {
        // Edge case: All bits are 1
        let state = [u64::MAX; 25];
        let rc = KeccakChiplet::ROUND_CONSTANTS[0];

        let next_state = KeccakWitness::keccak_f_round(state, rc);

        // Verify that Chi and Theta produced deterministic
        // results for max values. After Theta, C should be
        // 0xFF..FF since XOR of 5 u64::MAX is u64::MAX
        // Chi step on all ones will result in specific
        // bit patterns.
        assert_ne!(next_state[0], 0);
        assert_ne!(next_state[24], 0);
    }

    #[test]
    fn bit_packing_max_values() {
        let mut trace = vec![vec![F::ZERO; 32]; KeccakColumns::NUM_COLUMNS];

        // Edge case: State with alternating bit
        // patterns to check packing integrity.
        let mut state = [0u64; 25];
        for (i, s) in state.iter_mut().enumerate() {
            *s = if i % 2 == 0 {
                0xAAAAAAAAAAAAAAAA
            } else {
                0x5555555555555555
            };
        }

        KeccakWitness::assign_permutation(&mut trace, 0, state);

        // Verify Lane (0,0) packing in the first row
        let lane_col = KeccakChiplet::get_lane_col(0, 0);
        assert_eq!(trace[lane_col][0], F::from(0xAAAAAAAAAAAAAAAAu128));

        // Verify specific bits are correctly
        // placed for the alternating pattern
        // 0xA is 1010, so even bits should be 0,
        // odd bits should be 1.
        assert_eq!(trace[KeccakChiplet::get_bit_col(0, 0, 0)][0], F::ZERO);
        assert_eq!(trace[KeccakChiplet::get_bit_col(0, 0, 1)][0], F::ONE);
    }

    #[test]
    fn round_constant_assignment() {
        let mut trace = vec![vec![F::ZERO; 32]; KeccakColumns::NUM_COLUMNS];
        let state = [0u64; 25];

        KeccakWitness::assign_permutation(&mut trace, 0, state);

        // Verify RC bits for round 0 (RC = 0x0000000000000001)
        assert_eq!(trace[KeccakChiplet::get_rc_col(0)][0], F::ONE);

        for z in 1..64 {
            assert_eq!(trace[KeccakChiplet::get_rc_col(z)][0], F::ZERO);
        }
    }

    #[test]
    fn sponge_single_block_trace() {
        let message = b"hekate";
        let num_rows = 32;

        let trace = keccak_trace(message, num_rows).unwrap();

        let s_in_out = trace.columns[PHYS_S_IN_OUT].as_bit_slice().unwrap();

        assert_eq!(s_in_out[0], Bit::ONE);
        assert_eq!(s_in_out[24], Bit::ONE);
        assert_eq!(s_in_out[25], Bit::ZERO);
    }

    #[test]
    fn sponge_multi_block_trace() {
        let message = vec![0u8; 136];
        let num_rows = 64;

        let trace = keccak_trace(&message, num_rows).unwrap();

        let s_in_out = trace.columns[PHYS_S_IN_OUT].as_bit_slice().unwrap();

        assert_eq!(s_in_out[0], Bit::ONE);
        assert_eq!(s_in_out[24], Bit::ONE);
        assert_eq!(s_in_out[25], Bit::ONE);
    }

    #[test]
    fn sponge_absorb_logic() {
        let mut message = vec![0u8; 136];
        message[0] = 0x12;
        message[7] = 0x34;

        let num_rows = 64;
        let trace = keccak_trace(&message, num_rows).unwrap();

        // Check Lane (Block64 type)
        let expected = 0x3400000000000012u64;

        // Lane(0,0) is at index 0 in physical trace
        let lane00 = trace.columns[0].as_b64_slice().unwrap();

        assert_eq!(lane00[0].to_tower(), Block64::from(expected));
    }

    #[test]
    fn ast_node_count() {
        let chiplet = KeccakChiplet::new(1024);
        let ast: ConstraintAst<F> = chiplet.constraint_ast();

        assert!(
            ast.arena.len() < 20_000,
            "Arena too large: {} nodes",
            ast.arena.len()
        );
        assert_eq!(
            ast.roots.len(),
            25 + 1600 + 2 + 2,
            "Expected 25 packing + 1600 round + 2 Ghost Protocol + 2 selector boolean"
        );
    }

    #[test]
    fn ast_matches_flat_constraints() {
        let chiplet = KeccakChiplet::new(1024);
        let ast: ConstraintAst<F> = chiplet.constraint_ast();
        let flat = ast.to_constraints();

        // Verify structural equivalence:
        // same number of constraints (roots)
        assert_eq!(ast.roots.len(), flat.len());

        // Verify packing constraints have correct structure
        // Each packing root:
        // Mul(s_in_out, Sum(lane, -2^0*bit0, ..., -2^63*bit63))
        for i in 0..25 {
            let root = ast.roots[i];
            match ast.arena.get(root) {
                ConstraintExpr::Mul(_, _) => {} // Gated by s_in_out
                other => panic!("Packing root {} should be Mul, got {:?}", i, other),
            }
        }

        // Verify round constraints have correct structure
        // Each round root:
        // Mul(s_round, Sum(next_bit, chi, ...))
        for i in 25..25 + 1600 {
            let root = ast.roots[i];
            match ast.arena.get(root) {
                ConstraintExpr::Mul(_, _) => {} // Gated by s_round
                other => panic!("Round root {} should be Mul, got {:?}", i, other),
            }
        }

        // Verify flat term count is massive while AST is compact
        let flat_term_count: usize = flat.iter().map(|c| c.terms.len()).sum();
        assert!(
            flat_term_count > 200_000,
            "Flat should have >200K terms, got {}",
            flat_term_count
        );
        assert!(
            ast.arena.len() < 20_000,
            "AST should have <20K nodes, got {}",
            ast.arena.len()
        );
    }

    #[test]
    fn parity_and_theta_nodes_are_shared() {
        let chiplet = KeccakChiplet::new(1024);
        let ast: ConstraintAst<F> = chiplet.constraint_ast();

        // Count column parity nodes:
        // Sum with exactly 5 children.
        // Should be exactly 320 (5 x-values * 64 z-values).
        let parity_count = (0..ast.arena.len())
            .filter(|&i| {
                matches!(
                    ast.arena.get(ExprId(i as u32)),
                    ConstraintExpr::Sum(children) if children.len() == 5
                )
            })
            .count();

        // Count Theta nodes:
        // Sum with exactly 3 children.
        // At least 1600 (5*5*64), but
        // some Chi/round Sum nodes
        // also have 3 children.
        let three_child_sum_count = (0..ast.arena.len())
            .filter(|&i| {
                matches!(
                    ast.arena.get(ExprId(i as u32)),
                    ConstraintExpr::Sum(children) if children.len() == 3
                )
            })
            .count();

        assert_eq!(
            parity_count, 320,
            "Expected exactly 320 parity nodes, got {}",
            parity_count
        );
        assert!(
            three_child_sum_count >= 1600,
            "Expected at least 1600 three-child Sum nodes (Theta), got {}",
            three_child_sum_count
        );
    }

    #[test]
    fn sha3_256_empty() {
        let (hash, _) = sha3_256(b"");

        // SHA3-256("") = a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a
        let expected = [
            0xa7, 0xff, 0xc6, 0xf8, 0xbf, 0x1e, 0xd7, 0x66, 0x51, 0xc1, 0x47, 0x56, 0xa0, 0x61,
            0xd6, 0x62, 0xf5, 0x80, 0xff, 0x4d, 0xe4, 0x3b, 0x49, 0xfa, 0x82, 0xd8, 0x0a, 0x4b,
            0x80, 0xf8, 0x43, 0x4a,
        ];
        assert_eq!(hash, expected, "SHA3-256 empty string mismatch");
    }

    #[test]
    fn sha3_512_empty() {
        let (hash, _) = sha3_512(b"");

        // SHA3-512("") first 8 bytes: a69f73cca23a9ac5
        assert_eq!(hash[0], 0xa6);
        assert_eq!(hash[1], 0x9f);
        assert_eq!(hash[2], 0x73);
        assert_eq!(hash[3], 0xcc);
    }

    #[test]
    fn shake256_known_vector() {
        let (out, _) = shake256(b"", 32);

        // SHAKE-256("", 32) = 46b9dd2b0ba88d13...
        assert_eq!(out[0], 0x46);
        assert_eq!(out[1], 0xb9);
    }
}
