// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! `Keccak-f[1600]` chiplet for Hekate, the Rust zero-knowledge proof engine.
//!
//! [`KeccakChiplet`] proves each output is the permutation of its input.
//! [`sha3_256`], [`sha3_512`], [`shake128`] and [`shake256`] hash a message
//! natively and return the permutation calls a host table requests,
//! and [`generate_keccak_trace`] builds the chiplet's columns from them.
//!
//! - [`Keccak-f[1600]`][keccak]: what the proof states,
//!   and binding a digest to a message
//! - [Cryptographic Chiplets][chiplets]
//!
//! [keccak]: https://oumuamua.dev/primitives/hashing/keccak
//! [chiplets]: https://oumuamua.dev/hekate/docs/basics/cryptographic-chiplets#inside-a-chiplet-keccak-f-1600

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
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::define_columns;
use hekate_program::expander::VirtualExpander;
use hekate_program::permutation::{BusKind, PermutationCheckSpec, Service, ServiceSlot};
use hekate_program::{Air, FixedColumn, FixedShape, fix};
use once_cell::race::OnceBox;
use zeroize::{Zeroize, Zeroizing};

// FIPS 202 §6.1-6.2:
// suffix bits folded into pad10*1.
const SHA3_DOMAIN_SEP: u8 = 0x06;
const SHAKE_DOMAIN_SEP: u8 = 0x1f;

const SHA3_256_RATE: usize = 136; // (1600 - 512) / 8
const SHA3_512_RATE: usize = 72; // (1600 - 1024) / 8
const SHAKE128_RATE: usize = 168; // (1600 - 256) / 8
const SHAKE256_RATE: usize = 136; // (1600 - 512) / 8

/// Both bus endpoints must emit these labels in this order.
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

// Physical layout. Column order must match VirtualExpander
// sequence exactly. LANES and ROUND are bit-decomposed by
// the expander; KeccakColumns indexes the virtual trace.
define_columns! {
    pub PhysKeccakColumns {
        P_LANES: [B64; 25],
        P_ROUND: B32,
        P_S_ROUND: Bit,
        P_S_IN_OUT: Bit,
    }
}

define_columns! {
    pub KeccakColumns {
        STATE_BITS: [Bit; 1600],
        ROUND_BITS: [Bit; 32],
        LANES: [B64; 25],
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

/// `Keccak-f[1600]` as 24 round rows plus one output row per block.
///
/// State lives in 1600 virtual bit columns; Chi stays degree 2.
/// The 25 B64 lanes carrying the bus key are the same physical
/// columns those bits expand from. The schedule columns are
/// fixed columns on a stride-25 cadence of `num_blocks` blocks.
#[derive(Clone, Debug)]
pub struct KeccakChiplet {
    pub num_rows: usize,
    pub num_blocks: usize,
}

impl KeccakChiplet {
    pub const BUS_ID: &'static str = "keccak_link";

    /// FIPS 202 §3.2.2, indexed `[x][y]`.
    pub const RHO_OFFSETS: [[usize; 5]; 5] = [
        [0, 36, 3, 41, 18],
        [1, 44, 10, 45, 2],
        [62, 6, 43, 15, 61],
        [28, 55, 25, 21, 56],
        [27, 20, 39, 8, 14],
    ];

    /// FIPS 202 §3.2.5.
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

    /// The 24 round rows plus the output row.
    pub const BLOCK_ROWS: usize = Self::ROUND_CONSTANTS.len() + 1;

    /// Blocks sit contiguously at rows `0..BLOCK_ROWS·num_blocks`.
    pub fn new(num_rows: usize, num_blocks: usize) -> Self {
        assert!(
            num_rows.is_power_of_two(),
            "num_rows must be a power of two"
        );
        assert!(
            num_blocks
                .checked_mul(Self::BLOCK_ROWS)
                .is_some_and(|span| span <= num_rows),
            "{num_blocks} blocks need {} rows, the table holds {num_rows}",
            num_blocks.saturating_mul(Self::BLOCK_ROWS),
        );

        Self {
            num_rows,
            num_blocks,
        }
    }

    #[inline(always)]
    pub fn get_bit_col(x: usize, y: usize, z: usize) -> usize {
        let lane_idx = y * 5 + x;
        KeccakColumns::STATE_BITS + lane_idx * 64 + z
    }

    #[inline(always)]
    pub fn get_round_col(round: usize) -> usize {
        KeccakColumns::ROUND_BITS + round
    }

    #[inline(always)]
    pub fn get_lane_col(x: usize, y: usize) -> usize {
        KeccakColumns::LANES + (y * 5 + x)
    }

    /// Both endpoints derive from this schema:
    /// 25 lanes, then the emit rank.
    pub fn service() -> Service {
        let mut slots = Vec::with_capacity(26);

        for label in KECCAK_LANE_LABELS {
            slots.push(ServiceSlot::Value(label));
        }

        slots.push(ServiceSlot::EmitRank);

        Service {
            bus_id: Self::BUS_ID,
            kind: BusKind::Permutation,
            slots,
        }
    }

    pub fn linking_spec() -> PermutationCheckSpec {
        let values: Vec<usize> = (0..25).map(|lane| KeccakColumns::LANES + lane).collect();

        Self::service()
            .respond(&values, KeccakColumns::S_IN_OUT)
            .expect("service slots match the responder columns")
    }

    /// Host emit schedule: `SELECTOR` fires
    /// at block offsets 0 and `stride - 1`.
    pub fn host_selector_shape<F: TowerField>(stride: usize, count: usize) -> FixedShape<F> {
        assert!(stride >= 2);

        FixedShape::Cadence {
            stride,
            count,
            origin: 0,
            values: (0..stride)
                .map(|off| {
                    if off == 0 || off == stride - 1 {
                        F::ONE
                    } else {
                        F::ZERO
                    }
                })
                .collect(),
        }
    }

    pub fn physical_layout() -> &'static [ColumnType] {
        static PHYSICAL_LAYOUT: OnceBox<Vec<ColumnType>> = OnceBox::new();
        PHYSICAL_LAYOUT.get_or_init(|| Box::new(PhysKeccakColumns::build_layout()))
    }

    /// `phys_offset` is absolute in the host program's
    /// physical layout, not relative to this chiplet.
    pub fn expand_into(expander: VirtualExpander, phys_offset: usize) -> VirtualExpander {
        expander
            .expand_bits(25, ColumnType::B64)
            .expand_bits(1, ColumnType::B32)
            .reuse_pass_through(phys_offset, 25)
            .control_bits(2)
    }
}

impl<F: TowerField + TraceCompatibleField> Air<F> for KeccakChiplet {
    fn name(&self) -> String {
        "KeccakChiplet".to_string()
    }

    fn column_layout(&self) -> &[ColumnType] {
        Self::physical_layout()
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![(Self::BUS_ID.into(), Self::linking_spec())]
    }

    fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
        const ROUNDS: usize = KeccakChiplet::ROUND_CONSTANTS.len();
        const OUTPUT_OFFSET: usize = KeccakChiplet::BLOCK_ROWS - 1;

        let block = |values: Vec<F>| FixedShape::Cadence {
            stride: Self::BLOCK_ROWS,
            count: self.num_blocks,
            origin: 0,
            values,
        };

        let shape = |pred: &dyn Fn(usize) -> bool| {
            block(
                (0..Self::BLOCK_ROWS)
                    .map(|off| if pred(off) { F::ONE } else { F::ZERO })
                    .collect(),
            )
        };

        let mut pins = Vec::with_capacity(34);

        pins.push(fix(KeccakColumns::S_ROUND, shape(&|off| off < ROUNDS)));
        pins.push(fix(
            KeccakColumns::S_IN_OUT,
            shape(&|off| off == 0 || off == OUTPUT_OFFSET),
        ));

        for r in 0..32 {
            pins.push(fix(
                KeccakColumns::ROUND_BITS + r,
                shape(&|off| r < ROUNDS && off == r),
            ));
        }

        pins
    }

    fn virtual_expander(&self) -> Option<&VirtualExpander> {
        static E: OnceBox<VirtualExpander> = OnceBox::new();
        Some(E.get_or_init(|| {
            Box::new(
                Self::expand_into(VirtualExpander::new(), PhysKeccakColumns::P_LANES)
                    .build()
                    .expect("KeccakChiplet expander"),
            )
        }))
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();

        let s_round = cs.col(KeccakColumns::S_ROUND);
        let s_in_out = cs.col(KeccakColumns::S_IN_OUT);

        // lane = Σ bit[z] · 2^z on emit rows
        for y in 0..5 {
            for x in 0..5 {
                let lane = cs.col(Self::get_lane_col(x, y));
                let bits: Vec<_> = (0..64)
                    .map(|z| cs.scale(F::from(1u128 << z), cs.col(Self::get_bit_col(x, y, z))))
                    .collect();

                cs.assert_zero_when(s_in_out, lane + cs.sum(&bits));
            }
        }

        // C[x,z] = Σ_y A[x,y,z]
        let col_parity: [[_; 64]; 5] = core::array::from_fn(|x| {
            core::array::from_fn(|z| {
                cs.sum(
                    &(0..5)
                        .map(|y| cs.col(Self::get_bit_col(x, y, z)))
                        .collect::<Vec<_>>(),
                )
            })
        });

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

        // B[x,y,z] read through inverse Pi and Rho
        let get_b = |out_x: usize, out_y: usize, out_z: usize| {
            let in_x = (out_x + 3 * out_y) % 5;
            let in_y = out_x;
            let rot = Self::RHO_OFFSETS[in_x][in_y];
            let in_z = (out_z + 64 - rot) % 64;

            theta[in_x][in_y][in_z]
        };

        for x in 0..5 {
            for y in 0..5 {
                for z in 0..64 {
                    let b_curr = get_b(x, y, z);
                    let b_next1 = get_b((x + 1) % 5, y, z);
                    let b_next2 = get_b((x + 2) % 5, y, z);

                    let chi = cs.sum(&[b_curr, b_next2, b_next1 * b_next2]);
                    let next_bit = cs.next(Self::get_bit_col(x, y, z));

                    // Iota's constant is a coefficient, never a witness column
                    let flips: Vec<_> = match (x, y) {
                        (0, 0) => (0..24)
                            .filter(|&round| (Self::ROUND_CONSTANTS[round] >> z) & 1 == 1)
                            .map(|round| cs.col(Self::get_round_col(round)))
                            .collect(),
                        _ => Vec::new(),
                    };

                    match flips.is_empty() {
                        true => cs.assert_zero_when(s_round, next_bit + chi),
                        false => {
                            cs.assert_zero_when(s_round, cs.sum(&[next_bit, chi, cs.sum(&flips)]))
                        }
                    }
                }
            }
        }

        cs.build()
    }
}

pub struct KeccakWitness;

impl KeccakWitness {
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

    /// Writes 25 rows from `start_row` and returns the permuted state.
    ///
    /// Row-at-a-time; use `KeccakSpongeNative::generate_trace`
    /// outside tests and hand-built traces.
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

            trace[KeccakChiplet::get_round_col(round)][row] = F::ONE;

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

                    trace[bit_col][row] = F::from(bit_val as u128);
                }
            }
        }
    }
}

#[derive(Clone, Copy, Zeroize)]
struct RowData {
    state: [u64; 25],
    round: u32,
    s_round: bool,
    s_in_out: bool,
}

/// Lays `calls[k]` out as block `k`, 24 round rows plus
/// one output row, answering the host's `k`-th call.
///
/// Preferred over `KeccakSpongeNative::generate_trace` when
/// the caller owns the input states, as in Merkle tree hashing.
///
/// # Errors
/// More calls than `num_rows` holds.
pub fn generate_keccak_trace(
    calls: &[[Block64; 25]],
    num_rows: usize,
) -> hekate_core::errors::Result<ColumnTrace> {
    let mut rows = Zeroizing::new(Vec::with_capacity(num_rows));

    for call in calls {
        if rows.len() + KeccakChiplet::BLOCK_ROWS > num_rows {
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
                round: 1u32 << round,
                s_round: true,
                s_in_out: round == 0,
            });

            state = KeccakWitness::keccak_f_round(state, rc);
        }

        rows.push(RowData {
            state,
            round: 0,
            s_round: false,
            s_in_out: true,
        });
    }

    // TraceBuilder zero-fills padding
    let num_vars = num_rows.trailing_zeros() as usize;

    let mut tb = TraceBuilder::new_secret(KeccakChiplet::physical_layout(), num_vars)?;

    for (i, row) in rows.iter().enumerate() {
        for lane in 0..25 {
            tb.set_b64(
                PhysKeccakColumns::P_LANES + lane,
                i,
                Block64::from(row.state[lane]),
            )?;
        }

        tb.set_b32(
            PhysKeccakColumns::P_ROUND,
            i,
            hekate_math::Block32::from(row.round),
        )?;
        tb.set_bit(
            PhysKeccakColumns::P_S_ROUND,
            i,
            if row.s_round { Bit::ONE } else { Bit::ZERO },
        )?;
        tb.set_bit(
            PhysKeccakColumns::P_S_IN_OUT,
            i,
            if row.s_in_out { Bit::ONE } else { Bit::ZERO },
        )?;
    }

    Ok(tb.build())
}

// =================================================================
// Native Keccak Sponge
// =================================================================

/// `(input_state, output_state)` of one Keccak-f call.
pub type KeccakCall = ([u64; 25], [u64; 25]);

/// Sponge that records every Keccak-f call;
/// the chiplet can reproduce the trace.
pub struct KeccakSpongeNative {
    state: [u64; 25],
    permutation_calls: Vec<KeccakCall>,
}

impl Drop for KeccakSpongeNative {
    fn drop(&mut self) {
        self.state.zeroize();
        self.permutation_calls.zeroize();
    }
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

    /// XORs `block` into the rate lanes, then permutes.
    /// A short `block` is zero-extended, not padded.
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

        self.record(input);
    }

    /// Applies FIPS 202 pad10*1 carrying `domain_sep`.
    pub fn absorb(&mut self, msg: &[u8], rate_bytes: usize, domain_sep: u8) {
        let mut offset = 0;
        while offset + rate_bytes <= msg.len() {
            self.absorb_block(&msg[offset..offset + rate_bytes], rate_bytes);
            offset += rate_bytes;
        }

        let mut last = Zeroizing::new(vec![0u8; rate_bytes]);
        let remaining = msg.len() - offset;

        last[..remaining].copy_from_slice(&msg[offset..]);
        last[remaining] = domain_sep;
        last[rate_bytes - 1] |= 0x80;

        self.absorb_block(&last, rate_bytes);
    }

    pub fn squeeze(&mut self, out_len: usize, rate_bytes: usize) -> Zeroizing<Vec<u8>> {
        let mut output = Zeroizing::new(Vec::with_capacity(out_len));
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

            self.record(input);
        }

        output.truncate(out_len);

        output
    }

    pub fn into_calls(mut self) -> Zeroizing<Vec<KeccakCall>> {
        Zeroizing::new(core::mem::take(&mut self.permutation_calls))
    }

    pub fn generate_trace(self, num_rows: usize) -> hekate_core::errors::Result<ColumnTrace> {
        let recorded = self.into_calls();
        let calls: Zeroizing<Vec<[Block64; 25]>> = Zeroizing::new(
            recorded
                .iter()
                .map(|(input, _)| input.map(Block64::from))
                .collect(),
        );

        generate_keccak_trace(&calls, num_rows)
    }

    /// Appends the call `(input, self.state)`,
    /// wiping the outgrown buffer on each growth.
    fn record(&mut self, input: [u64; 25]) {
        let calls = &mut self.permutation_calls;

        if calls.len() == calls.capacity() {
            let mut grown = Vec::with_capacity((2 * calls.capacity()).max(4));
            grown.extend_from_slice(calls);

            calls.zeroize();

            *calls = grown;
        }

        calls.push((input, self.state));
    }
}

pub fn sha3_256(msg: &[u8]) -> ([u8; 32], Zeroizing<Vec<KeccakCall>>) {
    let mut sponge = KeccakSpongeNative::new();
    sponge.absorb(msg, SHA3_256_RATE, SHA3_DOMAIN_SEP);

    let out = sponge.squeeze(32, SHA3_256_RATE);

    let mut hash = [0u8; 32];
    hash.copy_from_slice(&out);

    (hash, sponge.into_calls())
}

pub fn sha3_512(msg: &[u8]) -> ([u8; 64], Zeroizing<Vec<KeccakCall>>) {
    let mut sponge = KeccakSpongeNative::new();
    sponge.absorb(msg, SHA3_512_RATE, SHA3_DOMAIN_SEP);

    let out = sponge.squeeze(64, SHA3_512_RATE);

    let mut hash = [0u8; 64];
    hash.copy_from_slice(&out);

    (hash, sponge.into_calls())
}

pub fn shake128(msg: &[u8], out_len: usize) -> (Zeroizing<Vec<u8>>, Zeroizing<Vec<KeccakCall>>) {
    let mut sponge = KeccakSpongeNative::new();
    sponge.absorb(msg, SHAKE128_RATE, SHAKE_DOMAIN_SEP);

    let out = sponge.squeeze(out_len, SHAKE128_RATE);

    (out, sponge.into_calls())
}

pub fn shake256(msg: &[u8], out_len: usize) -> (Zeroizing<Vec<u8>>, Zeroizing<Vec<KeccakCall>>) {
    let mut sponge = KeccakSpongeNative::new();
    sponge.absorb(msg, SHAKE256_RATE, SHAKE_DOMAIN_SEP);

    let out = sponge.squeeze(out_len, SHAKE256_RATE);

    (out, sponge.into_calls())
}

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

        sponge.generate_trace(num_rows)
    }

    #[test]
    fn keccak_layout_from_schema() {
        let layout = KeccakColumns::build_layout();
        assert_eq!(KeccakColumns::NUM_COLUMNS, 1659);
        assert_eq!(layout.len(), 1659);
        assert_eq!(layout[0], ColumnType::Bit);
        assert_eq!(layout[KeccakColumns::LANES], ColumnType::B64);
    }

    #[test]
    fn keccak_chiplet_air_metadata() {
        let chiplet = KeccakChiplet::new(32, 1);
        assert_eq!(Air::<F>::num_columns(&chiplet), 1659);
        assert_eq!(KeccakChiplet::physical_layout().len(), 28);
        assert_eq!(Air::<F>::name(&chiplet), "KeccakChiplet".to_string());
    }

    #[test]
    fn keccak_round_function_zero_input() {
        // From a zero state only Iota writes, and only A[0,0]
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

        assert_eq!(trace[KeccakColumns::S_IN_OUT][0], F::ONE);
        assert_eq!(trace[KeccakColumns::S_ROUND][0], F::ONE);
        assert_eq!(
            trace[KeccakChiplet::get_lane_col(0, 0)][0],
            F::from(0xAAu128)
        );

        assert_eq!(trace[KeccakColumns::S_ROUND][23], F::ONE);
        assert_eq!(trace[KeccakColumns::S_IN_OUT][23], F::ZERO);

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

        let lane_val = 0x123456789ABCDEF0u64;
        for z in 0..64 {
            let bit = (lane_val >> z) & 1;
            let expected = if bit == 1 { F::ONE } else { F::ZERO };
            assert_eq!(trace[KeccakChiplet::get_bit_col(0, 0, z)][0], expected);
        }
    }

    #[test]
    fn keccak_f_round_all_ones() {
        let state = [u64::MAX; 25];
        let rc = KeccakChiplet::ROUND_CONSTANTS[0];

        let next_state = KeccakWitness::keccak_f_round(state, rc);

        assert_ne!(next_state[0], 0);
        assert_ne!(next_state[24], 0);
    }

    #[test]
    fn bit_packing_max_values() {
        let mut trace = vec![vec![F::ZERO; 32]; KeccakColumns::NUM_COLUMNS];
        let mut state = [0u64; 25];

        for (i, s) in state.iter_mut().enumerate() {
            *s = if i % 2 == 0 {
                0xAAAAAAAAAAAAAAAA
            } else {
                0x5555555555555555
            };
        }

        KeccakWitness::assign_permutation(&mut trace, 0, state);

        let lane_col = KeccakChiplet::get_lane_col(0, 0);
        assert_eq!(trace[lane_col][0], F::from(0xAAAAAAAAAAAAAAAAu128));

        // 0xA is 1010: even bits clear, odd bits set.
        assert_eq!(trace[KeccakChiplet::get_bit_col(0, 0, 0)][0], F::ZERO);
        assert_eq!(trace[KeccakChiplet::get_bit_col(0, 0, 1)][0], F::ONE);
    }

    #[test]
    fn round_index_assignment() {
        let mut trace = vec![vec![F::ZERO; 32]; KeccakColumns::NUM_COLUMNS];
        let state = [0u64; 25];

        KeccakWitness::assign_permutation(&mut trace, 0, state);

        for r in 0..32 {
            let column = &trace[KeccakChiplet::get_round_col(r)];
            for (row, value) in column.iter().take(25).enumerate() {
                let expected = if row == r && r < 24 { F::ONE } else { F::ZERO };
                assert_eq!(*value, expected, "round column {r} at row {row}");
            }
        }
    }

    #[test]
    fn sponge_single_block_trace() {
        let message = b"hekate";
        let num_rows = 32;

        let trace = keccak_trace(message, num_rows).unwrap();

        let s_in_out = trace.columns[PhysKeccakColumns::P_S_IN_OUT]
            .as_bit_slice()
            .unwrap();

        assert_eq!(s_in_out[0], Bit::ONE);
        assert_eq!(s_in_out[24], Bit::ONE);
        assert_eq!(s_in_out[25], Bit::ZERO);
    }

    #[test]
    fn sponge_multi_block_trace() {
        let message = vec![0u8; 136];
        let num_rows = 64;

        let trace = keccak_trace(&message, num_rows).unwrap();

        let s_in_out = trace.columns[PhysKeccakColumns::P_S_IN_OUT]
            .as_bit_slice()
            .unwrap();

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

        let expected = 0x3400000000000012u64;

        // Lane(0,0) sits at physical index 0
        let lane00 = trace.columns[0].as_b64_slice().unwrap();

        assert_eq!(lane00[0].to_tower(), Block64::from(expected));
    }

    #[test]
    fn sponge_growth_keeps_every_call() {
        let mut sponge = KeccakSpongeNative::new();
        sponge.absorb(&[0u8; 9 * 136], 136, 0x06);

        let calls = sponge.into_calls();

        assert_eq!(calls.len(), 10);
        assert_eq!(calls[0].0, [0u64; 25]);

        for (k, &(input, output)) in calls.iter().enumerate() {
            let mut expected = input;
            keccak::Keccak::new().with_f1600(|f| f(&mut expected));

            assert_eq!(output, expected, "call {k}");
        }

        for (k, pair) in calls.windows(2).take(8).enumerate() {
            assert_eq!(pair[1].0, pair[0].1, "chain {k}");
        }
    }

    #[test]
    fn ast_node_count() {
        let chiplet = KeccakChiplet::new(1024, 40);
        let ast: ConstraintAst<F> = chiplet.constraint_ast();

        assert!(
            ast.arena.len() < 20_000,
            "Arena too large: {} nodes",
            ast.arena.len()
        );
        assert_eq!(
            ast.roots.len(),
            25 + 1600,
            "Expected 25 packing + 1600 round; the schedule is cadence pins"
        );
    }

    #[test]
    fn ast_matches_flat_constraints() {
        let chiplet = KeccakChiplet::new(1024, 40);
        let ast: ConstraintAst<F> = chiplet.constraint_ast();
        let flat = ast.to_constraints();

        assert_eq!(ast.roots.len(), flat.len());

        for i in 0..25 {
            let root = ast.roots[i];
            match ast.arena.get(root) {
                ConstraintExpr::Mul(_, _) => {} // Gated by s_in_out
                other => panic!("Packing root {} should be Mul, got {:?}", i, other),
            }
        }

        for i in 25..25 + 1600 {
            let root = ast.roots[i];
            match ast.arena.get(root) {
                ConstraintExpr::Mul(_, _) => {} // Gated by s_round
                other => panic!("Round root {} should be Mul, got {:?}", i, other),
            }
        }

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
        let chiplet = KeccakChiplet::new(1024, 40);
        let ast: ConstraintAst<F> = chiplet.constraint_ast();

        // Column parity is the only 5-child Sum:
        // 5 x × 64 z.
        let parity_count = (0..ast.arena.len())
            .filter(|&i| {
                matches!(
                    ast.arena.get(ExprId(i as u32)),
                    ConstraintExpr::Sum(children) if children.len() == 5
                )
            })
            .count();

        // Theta contributes 5·5·64 three-child Sums,
        // and Chi adds more, hence the lower bound.
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

    #[test]
    fn cadence_pins_pin_every_schedule_column() {
        let num_vars = 10;
        let pins = Air::<F>::fixed_columns(&KeccakChiplet::new(1024, 40));

        assert_eq!(pins.len(), 34);

        let pinned: Vec<usize> = pins.iter().map(|p| p.col_idx).collect();

        assert!(pinned.contains(&KeccakColumns::S_ROUND));
        assert!(pinned.contains(&KeccakColumns::S_IN_OUT));

        for r in 0..32 {
            assert!(pinned.contains(&(KeccakColumns::ROUND_BITS + r)));
        }

        let one = hekate_math::Flat::from_raw(F::ONE);
        let zero = hekate_math::Flat::from_raw(F::ZERO);
        let at = |col: usize, row: usize| {
            pins.iter()
                .find(|p| p.col_idx == col)
                .unwrap()
                .shape
                .value_at_row(row, num_vars)
        };

        assert_eq!(at(KeccakColumns::S_ROUND, 0), one);
        assert_eq!(at(KeccakColumns::S_ROUND, 23), one);
        assert_eq!(at(KeccakColumns::S_ROUND, 24), zero);
        assert_eq!(at(KeccakColumns::S_IN_OUT, 25), one);
        assert_eq!(at(KeccakColumns::S_IN_OUT, 26), zero);
        assert_eq!(at(KeccakColumns::S_IN_OUT, 49), one);
        assert_eq!(at(KeccakColumns::ROUND_BITS + 7, 25 + 7), one);
        assert_eq!(at(KeccakColumns::ROUND_BITS + 7, 25 + 8), zero);
        assert_eq!(at(KeccakColumns::ROUND_BITS + 30, 30), zero);

        assert_eq!(at(KeccakColumns::S_ROUND, 40 * 25), zero);
        assert_eq!(at(KeccakColumns::S_IN_OUT, 40 * 25), zero);
    }

    #[test]
    fn generated_trace_matches_cadence_pins() {
        let num_rows = 64;
        let num_vars = 6;
        let message = vec![0u8; 136];

        let trace = keccak_trace(&message, num_rows).unwrap();

        let chiplet = KeccakChiplet::new(num_rows, 2);
        let pins = Air::<F>::fixed_columns(&chiplet);
        let def = hekate_program::chiplet::ChipletDef::from_air(&chiplet).unwrap();
        let variants = def.expand_variants(&trace).unwrap();

        for pin in &pins {
            for row in 0..num_rows {
                assert_eq!(
                    variants[pin.col_idx].get_at(row),
                    pin.shape.value_at_row(row, num_vars),
                    "col {} row {}",
                    pin.col_idx,
                    row
                );
            }
        }
    }
}
