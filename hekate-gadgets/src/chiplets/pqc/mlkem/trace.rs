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

use super::witness::{ml_kem_decaps_traced, KeccakCallTag, MlKemDecapsKey, MlKemDecapsResult};
use super::{MlKemChiplet, MlKemCtrlColumns, Phase, MLKEM_Q};
use crate::chiplets::pqc::{basemul, ntt, twiddle_rom};
use crate::chiplets::{keccak, ram};
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use hekate_core::trace::{ColumnTrace, TraceBuilder, TraceCompatibleField};
use hekate_core::{errors, trace};
use hekate_math::{
    Bit, Block128, Block32, Block64, Flat, HardwareField, PackableField, TowerField,
};

// Dispatch kind enum, tags each schedule
// entry with the data needed to fill the
// corresponding ctrl row.
enum CtrlDispatch<'a> {
    KeccakInput(&'a [u64; 25]),
    KeccakOutput(&'a [u64; 25]),
    Basemul(&'a basemul::BasemulOp, ram::MemoryEvent),
    Ram(&'a ram::MemoryEvent),
    NttRam(&'a ntt::NttOp, &'a ram::MemoryEvent),
    WBindRam(u32, &'a ram::MemoryEvent),
    NttBoundary(u32, u32, bool, &'a ram::MemoryEvent),

    /// Ground-truth RAM write for
    /// lane-packed delta value.
    /// Precedes the Keccak dispatch
    /// so bind reads see it.
    KecInputWrite(ram::MemoryEvent),

    /// Keccak input ref row:
    /// one-hot selects which rate
    /// lane to verify delta against.
    /// Carry chain propagates
    /// KECCAK_LANES from KeccakInput.
    KecInputRef {
        one_hot_idx: usize,
        delta: u64,
        input_state: &'a [u64; 25],
    },

    /// Keccak input bind lo:
    /// reads lo u32 from RAM.
    /// Packing constraint verifies
    /// delta = lo + hi * tau_32.
    KecInputBindLo {
        delta: u64,
        lane_idx: u32,
        event: ram::MemoryEvent,
    },

    /// Keccak input bind hi:
    /// reads hi u32 from RAM.
    KecInputBindHi {
        event: ram::MemoryEvent,
    },

    /// Public-ct or SHA3-padding deposit row.
    /// Single B32 chunk, RAM write at `addr`.
    Io {
        addr: u32,
        value: u32,
        pad_kind: Option<PadKind>,
    },

    /// Raw H(ct) byte read, lo half-lane.
    /// `delta` and `lane_idx` engage the
    /// existing kec_input_bind GPA bus.
    IoLaneBindLo {
        delta: u64,
        lane_idx: u32,
        event: ram::MemoryEvent,
    },

    /// Raw H(ct) byte read, hi half-lane.
    IoLaneBindHi {
        event: ram::MemoryEvent,
    },

    /// Snapshot RATE_REG into HASH_REF
    /// after the last H(ct) absorption row.
    HCtBindSel,

    /// Snapshot RATE_REG into KECCAK_LANES[0..4]
    /// after the last H(ct') absorption row.
    HCtPrimeBindSel,

    /// Snapshot RATE_REG into K_PRIME after G(m'||h).
    KPrimeBindSel,

    /// Snapshot RATE_REG into K_BAR after J(z||c).
    KBarBindSel,

    HashCompare,

    /// Mux:
    /// SS = ct_match ? K_PRIME : K_BAR.
    SsMuxSel,

    /// Bus emission row for shared secret.
    SsOutSel,
}

// SHA3-256 padding chunk kind.
// Per FIPS 202 §B.2 the rate-aligned
// padding is 0x06 || 0x00 ... || 0x80,
// so as 4-byte chunks one is `First`
// (0x00000006), one is `Last` (0x80000000),
// the rest are `Mid` (0).
#[derive(Copy, Clone)]
enum PadKind {
    First,
    Mid,
    Last,
}

// =================================================================
// Chiplet Trace-Building Methods
// =================================================================

impl<F> MlKemChiplet<F>
where
    F: TowerField + TraceCompatibleField + PackableField + HardwareField + 'static,
    <F as PackableField>::Packed: Copy + Send + Sync,
    Flat<F>: Send + Sync,
{
    /// Generate chiplet traces for
    /// ML-KEM decapsulation.
    ///
    /// Runs the full decapsulation pipeline
    /// internally and produces all sub-chiplet
    /// traces in composite order.
    ///
    /// Returns `(traces, shared_secret)`.
    pub fn generate_traces(
        &self,
        ct: &[u8],
        sk: &[u8],
    ) -> errors::Result<(Vec<ColumnTrace>, [u8; 32])> {
        let dk = MlKemDecapsKey::from_nist_bytes(self.level, sk);
        let (result, ntt_ops) = ml_kem_decaps_traced(&dk, ct);
        let traces = self.generate_traces_inner(&result, &ntt_ops, ct)?;

        Ok((traces, result.shared_secret))
    }

    /// Internal trace generation from
    /// pre-computed decapsulation result.
    fn generate_traces_inner(
        &self,
        result: &MlKemDecapsResult,
        ntt_ops: &[ntt::NttOp],
        ct: &[u8],
    ) -> errors::Result<Vec<ColumnTrace>> {
        // 1. Keccak trace from permutation
        // calls generate_keccak_trace takes
        // input states only.
        let keccak_inputs: Vec<[Block64; 25]> = result
            .keccak_calls
            .iter()
            .map(|(input, _output)| {
                let mut block = [Block64::default(); 25];
                for (i, &lane) in input.iter().enumerate() {
                    block[i] = Block64::from(lane);
                }

                block
            })
            .collect();

        // 2. NTT trace from butterfly operations
        let ntt_trace = ntt::generate_ntt_trace(MLKEM_Q, ntt_ops, self.params.ntt_rows)?;

        // 3. Twiddle ROM trace
        // The twiddle bus uses s_active,
        // so EVERY op (butterfly + mul-only)
        // generates a twiddle event. Build
        // the ROM table from the actual ops.
        let mut twiddle_entries = Vec::with_capacity(ntt_ops.len());
        for op in ntt_ops {
            match op {
                ntt::NttOp::Butterfly(b) => {
                    twiddle_entries.push(twiddle_rom::TwiddleEntry {
                        layer: b.layer,
                        is_mulonly: false,
                        butterfly_idx: b.butterfly_idx,
                        w: b.w,
                        active: true,
                        request_idx_tr: 0,
                    });
                }
                ntt::NttOp::MulOnly(m) => {
                    twiddle_entries.push(twiddle_rom::TwiddleEntry {
                        layer: m.layer,
                        is_mulonly: m.is_basemul,
                        butterfly_idx: m.butterfly_idx,
                        w: m.w,
                        active: true,
                        request_idx_tr: 0,
                    });
                }
                ntt::NttOp::FlowCompanion(_) => {
                    twiddle_entries.push(twiddle_rom::TwiddleEntry {
                        layer: 0,
                        butterfly_idx: 0,
                        w: 0,
                        is_mulonly: false,
                        active: false,
                        request_idx_tr: 0,
                    });
                }
            }
        }

        // 4. Ctrl trace:
        // Build a unified dispatch schedule
        // sorted by phase so the forward-only
        // phase constraint (phase[row+1] >= phase[row])
        // is satisfied. Within each phase
        // the operation order is preserved
        // from the witness generator.
        let ctrl_layout = MlKemCtrlColumns::build_layout();
        let ctrl_vars = self.params.ctrl_rows.trailing_zeros() as usize;

        let mut ctrl_tb = TraceBuilder::new(&ctrl_layout, ctrl_vars)?;

        // Required capacity per Keccak call:
        // 42 writes + 1 input + 21 ref
        // + 42 bind + 1 output = 107.
        let keccak_rows_needed = result.keccak_calls.len() * 107;

        let total_needed = keccak_rows_needed
            + result.basemul_ops.len()
            + result.ram_events.len()
            + result.ntt_boundary_bindings.len()
            + 1;

        if total_needed > self.params.ctrl_rows {
            return Err(errors::Error::Trace(trace::Error::InvalidParameters {
                message: "ctrl_rows too small for ML-KEM dispatch schedule",
            }));
        }

        // Helper:
        // set phase column for a row.
        let set_phase = |tb: &mut TraceBuilder, row: usize, phase: Phase| -> errors::Result<()> {
            let col = match phase {
                Phase::Io => MlKemCtrlColumns::PH_IO,
                Phase::Decrypt => MlKemCtrlColumns::PH_DECRYPT,
                Phase::GHash => MlKemCtrlColumns::PH_G_HASH,
                Phase::Encrypt => MlKemCtrlColumns::PH_ENCRYPT,
                Phase::CmpHash => MlKemCtrlColumns::PH_CMP_HASH,
                Phase::Compare => MlKemCtrlColumns::PH_COMPARE,
            };

            tb.set_bit(col, row, Bit::ONE)?;

            Ok(())
        };

        // Build the schedule from witness data
        let mut schedule: Vec<(Phase, CtrlDispatch)> = Vec::with_capacity(total_needed);

        // Public ciphertext + SHA3-256 padding.
        // Each row deposits one B32 chunk into
        // the IO RAM range at byte address k*4.
        // Padding bytes for SHA3-256:
        // 0x06 || 0x00 ... || 0x80 (FIPS 202 §B.2).
        let level = self.level;
        let ct_bytes = level.ct_bytes();
        let pad_bytes = {
            let r = ct_bytes % 136;
            if r == 0 {
                136
            } else {
                136 - r
            }
        };

        let io_bytes = ct_bytes + pad_bytes;

        let mut io_buf = vec![0u8; io_bytes];
        io_buf[..ct_bytes].copy_from_slice(ct);
        io_buf[ct_bytes] = 0x06;
        io_buf[io_bytes - 1] |= 0x80;

        let first_pad_chunk = ct_bytes / 4;
        let last_pad_chunk = io_bytes / 4 - 1;

        for (chunk_idx, chunk) in io_buf.chunks(4).enumerate() {
            let mut buf = [0u8; 4];
            buf[..chunk.len()].copy_from_slice(chunk);

            let value = u32::from_le_bytes(buf);
            let addr = (chunk_idx as u32) * 4;

            let pad_kind = if chunk_idx < first_pad_chunk {
                None
            } else if chunk_idx == first_pad_chunk {
                Some(PadKind::First)
            } else if chunk_idx == last_pad_chunk {
                Some(PadKind::Last)
            } else {
                Some(PadKind::Mid)
            };

            schedule.push((
                Phase::Io,
                CtrlDispatch::Io {
                    addr,
                    value,
                    pad_kind,
                },
            ));
        }

        // Keccak dispatch + input binding.
        // Track rate regs to compute lane
        // deltas (input_state XOR prev_output).
        let mut sched_rate_regs = [0u64; 25];
        let mut kec_bind_addr = result.next_free_addr;
        let mut h_ct_block: u32 = 0;

        let n_calls = result.keccak_calls.len();

        let is_last_h_ct = |i: usize| {
            result.call_tag[i] == KeccakCallTag::HashCt
                && (i + 1 == n_calls || result.call_tag[i + 1] != KeccakCallTag::HashCt)
        };
        let is_last_h_ct_prime = |i: usize| {
            result.call_tag[i] == KeccakCallTag::HashCtPrime
                && (i + 1 == n_calls || result.call_tag[i + 1] != KeccakCallTag::HashCtPrime)
        };
        let is_last_g_hash = |i: usize| {
            result.call_tag[i] == KeccakCallTag::GHash
                && (i + 1 == n_calls || result.call_tag[i + 1] != KeccakCallTag::GHash)
        };
        let is_last_j_call = |i: usize| {
            result.call_tag[i] == KeccakCallTag::JCall
                && (i + 1 == n_calls || result.call_tag[i + 1] != KeccakCallTag::JCall)
        };

        for (i, (input, output)) in result.keccak_calls.iter().enumerate() {
            let ph = result.keccak_phases[i];

            let (is_init, _is_sha3_512, _is_shake_128) = result.keccak_sponge_meta[i];
            if is_init {
                sched_rate_regs = [0u64; 25];
            }

            let is_h_ct = result.call_tag[i] == KeccakCallTag::HashCt;
            if is_h_ct {
                // SHA3-256 rate = 17 lanes × 8 bytes.
                // The 17 ref rows engage the existing
                // kec_input_bind GPA bus + carry chain;
                // each (lo, hi) pair reads from the IO
                // RAM range and produces the matching
                // bind tuple via KEC_BIND_LO_SEL/HI_SEL.
                const SHA3_256_RATE_LANES: usize = 17;

                schedule.push((ph, CtrlDispatch::KeccakInput(input)));

                for k in 0..SHA3_256_RATE_LANES {
                    let delta = input[k] ^ sched_rate_regs[k];

                    schedule.push((
                        ph,
                        CtrlDispatch::KecInputRef {
                            one_hot_idx: k,
                            delta,
                            input_state: input,
                        },
                    ));
                }

                for k in 0..SHA3_256_RATE_LANES {
                    let delta = input[k] ^ sched_rate_regs[k];

                    let lo = delta as u32;
                    let hi = (delta >> 32) as u32;

                    let lo_addr = h_ct_block * 136 + (k as u32) * 8;
                    let hi_addr = lo_addr + 4;

                    schedule.push((
                        ph,
                        CtrlDispatch::IoLaneBindLo {
                            delta,
                            lane_idx: k as u32,
                            event: ram::MemoryEvent::read(lo_addr, 0, lo),
                        },
                    ));
                    schedule.push((
                        ph,
                        CtrlDispatch::IoLaneBindHi {
                            event: ram::MemoryEvent::read(hi_addr, 0, hi),
                        },
                    ));
                }

                schedule.push((ph, CtrlDispatch::KeccakOutput(output)));
                sched_rate_regs.copy_from_slice(&output[..]);

                h_ct_block += 1;

                if is_last_h_ct(i) {
                    schedule.push((ph, CtrlDispatch::HCtBindSel));
                }

                continue;
            }

            let rate_lanes = 21usize;
            let call_base_addr = kec_bind_addr;

            // Ground-truth writes before dispatch
            for k in 0..rate_lanes {
                let delta = input[k] ^ sched_rate_regs[k];
                let lo_addr = call_base_addr + (k * 2) as u32;
                let hi_addr = lo_addr + 1;

                schedule.push((
                    ph,
                    CtrlDispatch::KecInputWrite(ram::MemoryEvent::write(lo_addr, 0, delta as u32)),
                ));
                schedule.push((
                    ph,
                    CtrlDispatch::KecInputWrite(ram::MemoryEvent::write(
                        hi_addr,
                        0,
                        (delta >> 32) as u32,
                    )),
                ));
            }

            kec_bind_addr += (rate_lanes * 2) as u32;

            schedule.push((ph, CtrlDispatch::KeccakInput(input)));

            // All 21 ref rows first
            // (consecutive for carry chain),
            // then all bind rows.
            for k in 0..rate_lanes {
                let delta = input[k] ^ sched_rate_regs[k];

                schedule.push((
                    ph,
                    CtrlDispatch::KecInputRef {
                        one_hot_idx: k,
                        delta,
                        input_state: input,
                    },
                ));
            }

            for k in 0..rate_lanes {
                let delta = input[k] ^ sched_rate_regs[k];
                let lo_addr = call_base_addr + (k * 2) as u32;
                let hi_addr = lo_addr + 1;

                schedule.push((
                    ph,
                    CtrlDispatch::KecInputBindLo {
                        delta,
                        lane_idx: k as u32,
                        event: ram::MemoryEvent::read(lo_addr, 0, delta as u32),
                    },
                ));
                schedule.push((
                    ph,
                    CtrlDispatch::KecInputBindHi {
                        event: ram::MemoryEvent::read(hi_addr, 0, (delta >> 32) as u32),
                    },
                ));
            }

            schedule.push((ph, CtrlDispatch::KeccakOutput(output)));

            // Update all 25 state regs from output
            sched_rate_regs.copy_from_slice(&output[..]);

            if is_last_h_ct_prime(i) {
                schedule.push((ph, CtrlDispatch::HCtPrimeBindSel));
            }
            if is_last_g_hash(i) {
                schedule.push((ph, CtrlDispatch::KPrimeBindSel));
            }
            if is_last_j_call(i) {
                schedule.push((ph, CtrlDispatch::KBarBindSel));
            }
        }

        for (i, op) in result.basemul_ops.iter().enumerate() {
            let ram_event = ram::MemoryEvent {
                addr: op.ram_addr,
                clk: 0,
                val: op.c,
                is_write: true,
            };

            schedule.push((
                result.basemul_phases[i],
                CtrlDispatch::Basemul(op, ram_event),
            ));
        }

        // RAM events in original order.
        // b-side bound -> NttRam,
        // w-side bound -> WBindRam,
        // unbound -> Ram.
        let mut ram_to_ntt: BTreeMap<usize, usize> = BTreeMap::new();
        let mut ram_to_w_bfly: BTreeMap<usize, u32> = BTreeMap::new();

        for &(ntt_idx, ram_idx) in &result.ntt_ram_bindings {
            ram_to_ntt.insert(ram_idx, ntt_idx);
        }

        for &(ram_idx, bfly_idx) in &result.w_side_bindings {
            ram_to_w_bfly.insert(ram_idx, bfly_idx);
        }

        for (i, event) in result.ram_events.iter().enumerate() {
            match (ram_to_ntt.get(&i), ram_to_w_bfly.get(&i)) {
                (Some(&ntt_idx), None) => {
                    schedule.push((
                        result.ram_phases[i],
                        CtrlDispatch::NttRam(&ntt_ops[ntt_idx], event),
                    ));
                }
                (None, Some(&bfly_idx)) => {
                    schedule.push((
                        result.ram_phases[i],
                        CtrlDispatch::WBindRam(bfly_idx, event),
                    ));
                }
                (None, None) => {
                    schedule.push((result.ram_phases[i], CtrlDispatch::Ram(event)));
                }
                _ => unreachable!("RAM event bound to both b-side and w-side"),
            }
        }

        // NTT boundary dispatch entries.
        for &(inst, pos, ram_idx, is_input) in &result.ntt_boundary_bindings {
            let event = &result.ram_events[ram_idx];
            schedule.push((
                result.ram_phases[ram_idx],
                CtrlDispatch::NttBoundary(inst, pos, is_input, event),
            ));
        }

        schedule.push((Phase::Compare, CtrlDispatch::HashCompare));
        schedule.push((Phase::Compare, CtrlDispatch::SsMuxSel));
        schedule.push((Phase::Compare, CtrlDispatch::SsOutSel));

        // Stable sort by phase ordinal,
        // preserves within-phase order.
        schedule.sort_by_key(|(ph, _)| *ph as u8);

        // Single iteration:
        // fill ctrl rows + collect RAM
        // events with correct clock values.
        let mut ctrl_row = 0usize;
        let mut ram_events_fixed = Vec::with_capacity(result.ram_events.len());

        // Sponge state tracking for
        // sticky rate-lane registers.
        let mut rate_regs = [0u64; 25];
        let mut keccak_call_idx = 0usize;
        let mut h_ct_active_state = false;
        let mut hash_ref_carry = [0u64; 4];
        let mut hash_ct_prime_carry = [0u64; 4];
        let mut h_ct_bind_seen = false;
        let mut h_ct_prime_bind_seen = false;

        let mut k_prime_lo_carry = [0u32; 4];
        let mut k_prime_hi_carry = [0u32; 4];
        let mut k_bar_lo_carry = [0u32; 4];
        let mut k_bar_hi_carry = [0u32; 4];
        let mut ct_match_carry = false;
        let mut ss_lo_carry = [0u32; 4];
        let mut ss_hi_carry = [0u32; 4];
        let mut k_prime_bind_seen = false;
        let mut k_bar_bind_seen = false;

        let mut io_data_counter: u32 = 0;
        let mut bm_dispatch_ctrl_rows: Vec<u32> = Vec::with_capacity(result.basemul_ops.len());
        let mut keccak_request_idx_pairs: Vec<(u32, u32)> =
            Vec::with_capacity(result.keccak_calls.len());
        let mut pending_keccak_input_ctrl_row: Option<u32> = None;

        let mut wbind_ctrl_rows: BTreeMap<(u32, u32), Vec<u32>> = BTreeMap::new();

        for (phase, dispatch) in &schedule {
            // Sponge init:
            // zero registers before
            // writing to this row.
            if let CtrlDispatch::KeccakInput(_) = dispatch {
                let (is_init, _, _) = result.keccak_sponge_meta[keccak_call_idx];
                if is_init {
                    rate_regs = [0u64; 25];
                }

                if result.call_tag[keccak_call_idx] == KeccakCallTag::HashCt {
                    h_ct_active_state = true;
                }
            }

            if let CtrlDispatch::HCtBindSel = dispatch {
                hash_ref_carry.copy_from_slice(&rate_regs[..4]);
            }

            if let CtrlDispatch::HCtPrimeBindSel = dispatch {
                hash_ct_prime_carry.copy_from_slice(&rate_regs[..4]);
            }

            if let CtrlDispatch::KPrimeBindSel = dispatch {
                k_prime_lo_carry = result.k_prime_lo;
                k_prime_hi_carry = result.k_prime_hi;
            }

            if let CtrlDispatch::KBarBindSel = dispatch {
                k_bar_lo_carry = result.k_bar_lo;
                k_bar_hi_carry = result.k_bar_hi;
            }

            if let CtrlDispatch::SsMuxSel = dispatch {
                let ct_match_bit = result.h_ct == result.h_ct_prime;
                ct_match_carry = ct_match_bit;

                if ct_match_bit {
                    ss_lo_carry = k_prime_lo_carry;
                    ss_hi_carry = k_prime_hi_carry;
                } else {
                    ss_lo_carry = k_bar_lo_carry;
                    ss_hi_carry = k_bar_hi_carry;
                }
            }

            // Write sticky rate registers
            // before the dispatch match so
            // that KeccakOutput rows carry
            // the pre-update values.
            for i in 0..25 {
                ctrl_tb.set_b64(
                    MlKemCtrlColumns::RATE_REG + i,
                    ctrl_row,
                    Block64::from(rate_regs[i]),
                )?;
            }

            for i in 0..4 {
                ctrl_tb.set_b64(
                    MlKemCtrlColumns::HASH_REF + i,
                    ctrl_row,
                    Block64::from(hash_ref_carry[i]),
                )?;
                ctrl_tb.set_b64(
                    MlKemCtrlColumns::HASH_CT_PRIME + i,
                    ctrl_row,
                    Block64::from(hash_ct_prime_carry[i]),
                )?;
            }

            for i in 0..4 {
                ctrl_tb.set_b32(
                    MlKemCtrlColumns::K_PRIME_LO + i,
                    ctrl_row,
                    Block32::from(k_prime_lo_carry[i]),
                )?;
                ctrl_tb.set_b32(
                    MlKemCtrlColumns::K_PRIME_HI + i,
                    ctrl_row,
                    Block32::from(k_prime_hi_carry[i]),
                )?;
                ctrl_tb.set_b32(
                    MlKemCtrlColumns::K_BAR_LO + i,
                    ctrl_row,
                    Block32::from(k_bar_lo_carry[i]),
                )?;
                ctrl_tb.set_b32(
                    MlKemCtrlColumns::K_BAR_HI + i,
                    ctrl_row,
                    Block32::from(k_bar_hi_carry[i]),
                )?;
                ctrl_tb.set_b32(
                    MlKemCtrlColumns::SS_LO + i,
                    ctrl_row,
                    Block32::from(ss_lo_carry[i]),
                )?;
                ctrl_tb.set_b32(
                    MlKemCtrlColumns::SS_HI + i,
                    ctrl_row,
                    Block32::from(ss_hi_carry[i]),
                )?;
            }

            if ct_match_carry {
                ctrl_tb.set_bit(MlKemCtrlColumns::CT_MATCH, ctrl_row, Bit::ONE)?;
            }
            if h_ct_active_state {
                ctrl_tb.set_bit(MlKemCtrlColumns::H_CT_ACTIVE, ctrl_row, Bit::ONE)?;
            }
            if h_ct_bind_seen {
                ctrl_tb.set_bit(MlKemCtrlColumns::H_CT_BIND_SEEN, ctrl_row, Bit::ONE)?;
            }
            if h_ct_prime_bind_seen {
                ctrl_tb.set_bit(MlKemCtrlColumns::H_CT_PRIME_BIND_SEEN, ctrl_row, Bit::ONE)?;
            }
            if k_prime_bind_seen {
                ctrl_tb.set_bit(MlKemCtrlColumns::K_PRIME_BIND_SEEN, ctrl_row, Bit::ONE)?;
            }
            if k_bar_bind_seen {
                ctrl_tb.set_bit(MlKemCtrlColumns::K_BAR_BIND_SEEN, ctrl_row, Bit::ONE)?;
            }

            match dispatch {
                CtrlDispatch::KeccakInput(input) => {
                    pending_keccak_input_ctrl_row = Some(ctrl_row as u32);

                    for (lane, &val) in input.iter().enumerate() {
                        ctrl_tb.set_b64(
                            MlKemCtrlColumns::KECCAK_LANES + lane,
                            ctrl_row,
                            Block64::from(val),
                        )?;
                    }

                    ctrl_tb.set_bit(MlKemCtrlColumns::KECCAK_SELECTOR, ctrl_row, Bit::ONE)?;

                    // Sponge selectors from metadata
                    let (is_init, is_sha3_512, is_shake_128) =
                        result.keccak_sponge_meta[keccak_call_idx];

                    if is_init {
                        ctrl_tb.set_bit(MlKemCtrlColumns::SPONGE_INIT, ctrl_row, Bit::ONE)?;
                    }
                    if is_sha3_512 {
                        ctrl_tb.set_bit(MlKemCtrlColumns::SHA3_512, ctrl_row, Bit::ONE)?;
                    }
                    if is_shake_128 {
                        ctrl_tb.set_bit(MlKemCtrlColumns::SHAKE_128, ctrl_row, Bit::ONE)?;
                    }
                    if result.call_tag[keccak_call_idx] == KeccakCallTag::HashCt {
                        ctrl_tb.set_bit(MlKemCtrlColumns::H_CT_INPUT_SEL, ctrl_row, Bit::ONE)?;
                    }
                }
                CtrlDispatch::KeccakOutput(output) => {
                    let in_row =
                        pending_keccak_input_ctrl_row
                            .take()
                            .ok_or(errors::Error::Protocol {
                                protocol: "mlkem_trace",
                                message: "KeccakOutput dispatched without preceding KeccakInput",
                            })?;

                    keccak_request_idx_pairs.push((in_row, ctrl_row as u32));

                    for (lane, &val) in output.iter().enumerate() {
                        ctrl_tb.set_b64(
                            MlKemCtrlColumns::KECCAK_LANES + lane,
                            ctrl_row,
                            Block64::from(val),
                        )?;
                    }

                    ctrl_tb.set_bit(MlKemCtrlColumns::KECCAK_SELECTOR, ctrl_row, Bit::ONE)?;
                    ctrl_tb.set_bit(MlKemCtrlColumns::KEC_IS_OUTPUT, ctrl_row, Bit::ONE)?;

                    // Update all 25 state registers
                    // from output for next row.
                    rate_regs.copy_from_slice(&output[..]);

                    if h_ct_active_state {
                        ctrl_tb.set_bit(MlKemCtrlColumns::H_CT_ACTIVE, ctrl_row, Bit::ZERO)?;
                        h_ct_active_state = false;
                    }

                    keccak_call_idx += 1;
                }
                CtrlDispatch::Basemul(bm_op, ram_event) => {
                    bm_dispatch_ctrl_rows.push(ctrl_row as u32);

                    ctrl_tb.set_b32(MlKemCtrlColumns::BM_A, ctrl_row, Block32::from(bm_op.a))?;
                    ctrl_tb.set_b32(MlKemCtrlColumns::BM_B, ctrl_row, Block32::from(bm_op.b))?;
                    ctrl_tb.set_b32(MlKemCtrlColumns::BM_C, ctrl_row, Block32::from(bm_op.c))?;
                    ctrl_tb.set_b32(
                        MlKemCtrlColumns::BM_IDX,
                        ctrl_row,
                        Block32::from(bm_op.idx),
                    )?;

                    ctrl_tb.set_bit(MlKemCtrlColumns::BM_SELECTOR, ctrl_row, Bit::ONE)?;

                    let mut fixed = ram_event.clone();
                    fixed.clk = ctrl_row as u32;

                    ram_events_fixed.push(fixed);

                    let addr = ram_event.addr_bytes();
                    let val = ram_event.val_bytes();

                    for b in 0..4 {
                        ctrl_tb.set_b32(
                            MlKemCtrlColumns::RAM_ADDR + b,
                            ctrl_row,
                            Block32::from(addr[b] as u32),
                        )?;
                        ctrl_tb.set_b32(
                            MlKemCtrlColumns::RAM_VAL + b,
                            ctrl_row,
                            Block32::from(val[b] as u32),
                        )?;
                    }

                    ctrl_tb.set_b32(
                        MlKemCtrlColumns::RAM_VAL_PACKED,
                        ctrl_row,
                        Block32::from(ram_event.val),
                    )?;
                    ctrl_tb.set_bit(
                        MlKemCtrlColumns::RAM_IS_WRITE,
                        ctrl_row,
                        if ram_event.is_write {
                            Bit::ONE
                        } else {
                            Bit::ZERO
                        },
                    )?;
                    ctrl_tb.set_bit(MlKemCtrlColumns::RAM_SELECTOR, ctrl_row, Bit::ONE)?;
                }
                CtrlDispatch::Ram(event) => {
                    let mut fixed = (*event).clone();
                    fixed.clk = ctrl_row as u32;

                    ram_events_fixed.push(fixed);

                    let addr = event.addr_bytes();
                    let val = event.val_bytes();

                    for b in 0..4 {
                        ctrl_tb.set_b32(
                            MlKemCtrlColumns::RAM_ADDR + b,
                            ctrl_row,
                            Block32::from(addr[b] as u32),
                        )?;
                        ctrl_tb.set_b32(
                            MlKemCtrlColumns::RAM_VAL + b,
                            ctrl_row,
                            Block32::from(val[b] as u32),
                        )?;
                    }

                    ctrl_tb.set_b32(
                        MlKemCtrlColumns::RAM_VAL_PACKED,
                        ctrl_row,
                        Block32::from(event.val),
                    )?;

                    ctrl_tb.set_bit(
                        MlKemCtrlColumns::RAM_IS_WRITE,
                        ctrl_row,
                        if event.is_write { Bit::ONE } else { Bit::ZERO },
                    )?;
                    ctrl_tb.set_bit(MlKemCtrlColumns::RAM_SELECTOR, ctrl_row, Bit::ONE)?;
                }
                CtrlDispatch::WBindRam(bfly_idx, event) => {
                    // RAM columns (same as Ram dispatch)
                    let mut fixed = (*event).clone();
                    fixed.clk = ctrl_row as u32;

                    ram_events_fixed.push(fixed);

                    let addr = event.addr_bytes();
                    let val = event.val_bytes();

                    for b in 0..4 {
                        ctrl_tb.set_b32(
                            MlKemCtrlColumns::RAM_ADDR + b,
                            ctrl_row,
                            Block32::from(addr[b] as u32),
                        )?;
                        ctrl_tb.set_b32(
                            MlKemCtrlColumns::RAM_VAL + b,
                            ctrl_row,
                            Block32::from(val[b] as u32),
                        )?;
                    }

                    ctrl_tb.set_b32(
                        MlKemCtrlColumns::RAM_VAL_PACKED,
                        ctrl_row,
                        Block32::from(event.val),
                    )?;

                    ctrl_tb.set_bit(
                        MlKemCtrlColumns::RAM_IS_WRITE,
                        ctrl_row,
                        if event.is_write { Bit::ONE } else { Bit::ZERO },
                    )?;
                    ctrl_tb.set_bit(MlKemCtrlColumns::RAM_SELECTOR, ctrl_row, Bit::ONE)?;

                    // W-side binding columns
                    ctrl_tb.set_b32(
                        MlKemCtrlColumns::W_BIND_BFLY_IDX,
                        ctrl_row,
                        Block32::from(*bfly_idx),
                    )?;
                    ctrl_tb.set_bit(MlKemCtrlColumns::W_BIND_SELECTOR, ctrl_row, Bit::ONE)?;

                    wbind_ctrl_rows
                        .entry((*bfly_idx, event.val))
                        .or_default()
                        .push(ctrl_row as u32);
                }
                CtrlDispatch::NttRam(op, event) => {
                    // NTT columns (MulOnly for binding)
                    match op {
                        ntt::NttOp::Butterfly(bfly) => {
                            let wb = ((bfly.w as u64 * bfly.b as u64) % MLKEM_Q as u64) as u32;
                            let a_out = (bfly.a + wb) % MLKEM_Q;
                            let b_out = (bfly.a + MLKEM_Q - wb) % MLKEM_Q;

                            ctrl_tb.set_b32(
                                MlKemCtrlColumns::NTT_A,
                                ctrl_row,
                                Block32::from(bfly.a),
                            )?;
                            ctrl_tb.set_b32(
                                MlKemCtrlColumns::NTT_B,
                                ctrl_row,
                                Block32::from(bfly.b),
                            )?;
                            ctrl_tb.set_b32(
                                MlKemCtrlColumns::NTT_A_OUT,
                                ctrl_row,
                                Block32::from(a_out),
                            )?;
                            ctrl_tb.set_b32(
                                MlKemCtrlColumns::NTT_B_OUT,
                                ctrl_row,
                                Block32::from(b_out),
                            )?;
                            ctrl_tb.set_b32(
                                MlKemCtrlColumns::NTT_LAYER,
                                ctrl_row,
                                Block32::from(bfly.layer),
                            )?;
                            ctrl_tb.set_b32(
                                MlKemCtrlColumns::NTT_BUTTERFLY,
                                ctrl_row,
                                Block32::from(bfly.butterfly_idx),
                            )?;
                            ctrl_tb.set_b32(
                                MlKemCtrlColumns::NTT_INSTANCE,
                                ctrl_row,
                                Block32::from(bfly.ntt_instance),
                            )?;
                        }
                        ntt::NttOp::MulOnly(mul) => {
                            let wb = ((mul.w as u64 * mul.b as u64) % MLKEM_Q as u64) as u32;

                            ctrl_tb.set_b32(
                                MlKemCtrlColumns::NTT_A,
                                ctrl_row,
                                Block32::from(0u32),
                            )?;
                            ctrl_tb.set_b32(
                                MlKemCtrlColumns::NTT_B,
                                ctrl_row,
                                Block32::from(mul.b),
                            )?;
                            ctrl_tb.set_b32(
                                MlKemCtrlColumns::NTT_A_OUT,
                                ctrl_row,
                                Block32::from(wb),
                            )?;
                            ctrl_tb.set_b32(
                                MlKemCtrlColumns::NTT_B_OUT,
                                ctrl_row,
                                Block32::from((MLKEM_Q - wb) % MLKEM_Q),
                            )?;
                            ctrl_tb.set_b32(
                                MlKemCtrlColumns::NTT_LAYER,
                                ctrl_row,
                                Block32::from(mul.layer),
                            )?;
                            ctrl_tb.set_b32(
                                MlKemCtrlColumns::NTT_BUTTERFLY,
                                ctrl_row,
                                Block32::from(mul.butterfly_idx),
                            )?;
                            ctrl_tb.set_b32(
                                MlKemCtrlColumns::NTT_INSTANCE,
                                ctrl_row,
                                Block32::from(mul.flow_instance),
                            )?;
                        }
                        ntt::NttOp::FlowCompanion(_) => {
                            unreachable!("FlowCompanion filtered from schedule");
                        }
                    }

                    ctrl_tb.set_bit(MlKemCtrlColumns::NTT_SELECTOR, ctrl_row, Bit::ONE)?;

                    // RAM columns
                    let mut fixed = (*event).clone();
                    fixed.clk = ctrl_row as u32;

                    ram_events_fixed.push(fixed);

                    let addr = event.addr_bytes();
                    let val = event.val_bytes();

                    for b in 0..4 {
                        ctrl_tb.set_b32(
                            MlKemCtrlColumns::RAM_ADDR + b,
                            ctrl_row,
                            Block32::from(addr[b] as u32),
                        )?;
                        ctrl_tb.set_b32(
                            MlKemCtrlColumns::RAM_VAL + b,
                            ctrl_row,
                            Block32::from(val[b] as u32),
                        )?;
                    }

                    ctrl_tb.set_b32(
                        MlKemCtrlColumns::RAM_VAL_PACKED,
                        ctrl_row,
                        Block32::from(event.val),
                    )?;

                    ctrl_tb.set_bit(
                        MlKemCtrlColumns::RAM_IS_WRITE,
                        ctrl_row,
                        if event.is_write { Bit::ONE } else { Bit::ZERO },
                    )?;
                    ctrl_tb.set_bit(MlKemCtrlColumns::RAM_SELECTOR, ctrl_row, Bit::ONE)?;
                }
                CtrlDispatch::NttBoundary(inst, pos, is_input, event) => {
                    // RAM columns
                    let mut fixed = (*event).clone();
                    fixed.clk = ctrl_row as u32;

                    ram_events_fixed.push(fixed);

                    let addr = event.addr_bytes();
                    let val = event.val_bytes();

                    for b in 0..4 {
                        ctrl_tb.set_b32(
                            MlKemCtrlColumns::RAM_ADDR + b,
                            ctrl_row,
                            Block32::from(addr[b] as u32),
                        )?;
                        ctrl_tb.set_b32(
                            MlKemCtrlColumns::RAM_VAL + b,
                            ctrl_row,
                            Block32::from(val[b] as u32),
                        )?;
                    }

                    ctrl_tb.set_b32(
                        MlKemCtrlColumns::RAM_VAL_PACKED,
                        ctrl_row,
                        Block32::from(event.val),
                    )?;
                    ctrl_tb.set_bit(
                        MlKemCtrlColumns::RAM_IS_WRITE,
                        ctrl_row,
                        if event.is_write { Bit::ONE } else { Bit::ZERO },
                    )?;
                    ctrl_tb.set_bit(MlKemCtrlColumns::RAM_SELECTOR, ctrl_row, Bit::ONE)?;

                    // Boundary columns
                    ctrl_tb.set_b32(
                        MlKemCtrlColumns::NTT_INSTANCE,
                        ctrl_row,
                        Block32::from(*inst),
                    )?;
                    ctrl_tb.set_b32(MlKemCtrlColumns::BOUND_POS, ctrl_row, Block32::from(*pos))?;

                    if *is_input {
                        ctrl_tb.set_bit(MlKemCtrlColumns::BOUND_IN_SEL, ctrl_row, Bit::ONE)?;
                    } else {
                        ctrl_tb.set_bit(MlKemCtrlColumns::BOUND_OUT_SEL, ctrl_row, Bit::ONE)?;
                    }
                }
                CtrlDispatch::HashCompare => {
                    for i in 0..4 {
                        let h_ct_lane = {
                            let mut b = [0u8; 8];
                            b.copy_from_slice(&result.h_ct[i * 8..(i + 1) * 8]);

                            Block64::from(u64::from_le_bytes(b))
                        };
                        let h_ct_prime_lane = {
                            let mut b = [0u8; 8];
                            b.copy_from_slice(&result.h_ct_prime[i * 8..(i + 1) * 8]);

                            Block64::from(u64::from_le_bytes(b))
                        };

                        ctrl_tb.set_b64(MlKemCtrlColumns::HASH_REF + i, ctrl_row, h_ct_lane)?;
                        ctrl_tb.set_b64(
                            MlKemCtrlColumns::KECCAK_LANES + i,
                            ctrl_row,
                            h_ct_prime_lane,
                        )?;
                        ctrl_tb.set_b64(
                            MlKemCtrlColumns::HASH_CT_PRIME + i,
                            ctrl_row,
                            h_ct_prime_lane,
                        )?;
                    }

                    let ct_match = result.h_ct == result.h_ct_prime;
                    ctrl_tb.set_bit(
                        MlKemCtrlColumns::CT_MATCH,
                        ctrl_row,
                        Bit::from(ct_match as u8),
                    )?;
                    ctrl_tb.set_bit(MlKemCtrlColumns::CMP_SELECTOR, ctrl_row, Bit::ONE)?;

                    // Reverse hash comparison witnesses.
                    // diff_lo = (lane[0]^hash[0]) + (lane[1]^hash[1])*TAU
                    // diff_hi = (lane[2]^hash[2]) + (lane[3]^hash[3])*TAU
                    let hash_lane = |idx: usize| -> u64 {
                        let mut b = [0u8; 8];
                        b.copy_from_slice(&result.h_ct[idx * 8..(idx + 1) * 8]);

                        u64::from_le_bytes(b)
                    };

                    let prime_lane = |idx: usize| -> u64 {
                        let mut b = [0u8; 8];
                        b.copy_from_slice(&result.h_ct_prime[idx * 8..(idx + 1) * 8]);

                        u64::from_le_bytes(b)
                    };

                    let d0 = Block128::from(Block64::from(hash_lane(0) ^ prime_lane(0)));
                    let d1 = Block128::from(Block64::from(hash_lane(1) ^ prime_lane(1)));
                    let d2 = Block128::from(Block64::from(hash_lane(2) ^ prime_lane(2)));
                    let d3 = Block128::from(Block64::from(hash_lane(3) ^ prime_lane(3)));

                    let tau = Block128::EXTENSION_TAU;
                    let diff_lo = d0 + d1 * tau;
                    let diff_hi = d2 + d3 * tau;

                    let lo_zero = diff_lo == Block128::ZERO;
                    let hi_zero = diff_hi == Block128::ZERO;

                    ctrl_tb.set_bit(
                        MlKemCtrlColumns::HASH_EQ_LO,
                        ctrl_row,
                        Bit::from(lo_zero as u8),
                    )?;
                    ctrl_tb.set_bit(
                        MlKemCtrlColumns::HASH_EQ_HI,
                        ctrl_row,
                        Bit::from(hi_zero as u8),
                    )?;

                    if !lo_zero {
                        ctrl_tb.set_b128(
                            MlKemCtrlColumns::HASH_DIFF_INV_LO,
                            ctrl_row,
                            diff_lo.invert(),
                        )?;
                    }
                    if !hi_zero {
                        ctrl_tb.set_b128(
                            MlKemCtrlColumns::HASH_DIFF_INV_HI,
                            ctrl_row,
                            diff_hi.invert(),
                        )?;
                    }
                }
                CtrlDispatch::KPrimeBindSel => {
                    ctrl_tb.set_bit(MlKemCtrlColumns::K_PRIME_BIND_SEL, ctrl_row, Bit::ONE)?;
                }
                CtrlDispatch::KBarBindSel => {
                    ctrl_tb.set_bit(MlKemCtrlColumns::K_BAR_BIND_SEL, ctrl_row, Bit::ONE)?;
                }
                CtrlDispatch::SsMuxSel => {
                    ctrl_tb.set_bit(MlKemCtrlColumns::SS_MUX_SEL, ctrl_row, Bit::ONE)?;
                }
                CtrlDispatch::SsOutSel => {
                    ctrl_tb.set_bit(MlKemCtrlColumns::SS_OUT_SEL, ctrl_row, Bit::ONE)?;
                    ctrl_tb.set_b32(
                        MlKemCtrlColumns::REQUEST_IDX_OUT,
                        ctrl_row,
                        Block32::from(io_data_counter),
                    )?;
                }
                CtrlDispatch::KecInputWrite(event) => {
                    let mut fixed = event.clone();
                    fixed.clk = ctrl_row as u32;

                    ram_events_fixed.push(fixed);

                    let addr = event.addr_bytes();
                    let val = event.val_bytes();

                    for b in 0..4 {
                        ctrl_tb.set_b32(
                            MlKemCtrlColumns::RAM_ADDR + b,
                            ctrl_row,
                            Block32::from(addr[b] as u32),
                        )?;
                        ctrl_tb.set_b32(
                            MlKemCtrlColumns::RAM_VAL + b,
                            ctrl_row,
                            Block32::from(val[b] as u32),
                        )?;
                    }

                    ctrl_tb.set_b32(
                        MlKemCtrlColumns::RAM_VAL_PACKED,
                        ctrl_row,
                        Block32::from(event.val),
                    )?;
                    ctrl_tb.set_bit(MlKemCtrlColumns::RAM_IS_WRITE, ctrl_row, Bit::ONE)?;
                    ctrl_tb.set_bit(MlKemCtrlColumns::RAM_SELECTOR, ctrl_row, Bit::ONE)?;
                }
                CtrlDispatch::KecInputRef {
                    one_hot_idx,
                    delta,
                    input_state,
                } => {
                    for j in 0..25 {
                        ctrl_tb.set_b64(
                            MlKemCtrlColumns::KECCAK_LANES + j,
                            ctrl_row,
                            Block64::from(input_state[j]),
                        )?;
                    }

                    ctrl_tb.set_b64(
                        MlKemCtrlColumns::KEC_LANE_DELTA,
                        ctrl_row,
                        Block64::from(*delta),
                    )?;
                    ctrl_tb.set_bit(
                        MlKemCtrlColumns::KEC_LANE_ONE_HOT + one_hot_idx,
                        ctrl_row,
                        Bit::ONE,
                    )?;
                    ctrl_tb.set_b32(
                        MlKemCtrlColumns::KEC_LANE_IDX,
                        ctrl_row,
                        Block32::from(*one_hot_idx as u32),
                    )?;
                    ctrl_tb.set_bit(MlKemCtrlColumns::KEC_INPUT_REF_SEL, ctrl_row, Bit::ONE)?;
                }
                CtrlDispatch::KecInputBindLo {
                    delta,
                    lane_idx,
                    event,
                } => {
                    ctrl_tb.set_b64(
                        MlKemCtrlColumns::KEC_LANE_DELTA,
                        ctrl_row,
                        Block64::from(*delta),
                    )?;

                    // RAM columns
                    let mut fixed = event.clone();
                    fixed.clk = ctrl_row as u32;

                    ram_events_fixed.push(fixed);

                    let addr = event.addr_bytes();
                    let val = event.val_bytes();

                    for b in 0..4 {
                        ctrl_tb.set_b32(
                            MlKemCtrlColumns::RAM_ADDR + b,
                            ctrl_row,
                            Block32::from(addr[b] as u32),
                        )?;
                        ctrl_tb.set_b32(
                            MlKemCtrlColumns::RAM_VAL + b,
                            ctrl_row,
                            Block32::from(val[b] as u32),
                        )?;
                    }

                    ctrl_tb.set_b32(
                        MlKemCtrlColumns::RAM_VAL_PACKED,
                        ctrl_row,
                        Block32::from(event.val),
                    )?;
                    ctrl_tb.set_bit(
                        MlKemCtrlColumns::RAM_IS_WRITE,
                        ctrl_row,
                        if event.is_write { Bit::ONE } else { Bit::ZERO },
                    )?;
                    ctrl_tb.set_bit(MlKemCtrlColumns::RAM_SELECTOR, ctrl_row, Bit::ONE)?;
                    ctrl_tb.set_bit(MlKemCtrlColumns::KEC_BIND_LO_SEL, ctrl_row, Bit::ONE)?;
                    ctrl_tb.set_b32(
                        MlKemCtrlColumns::KEC_LANE_IDX,
                        ctrl_row,
                        Block32::from(*lane_idx),
                    )?;
                }
                CtrlDispatch::KecInputBindHi { event } => {
                    // RAM columns only
                    let mut fixed = event.clone();
                    fixed.clk = ctrl_row as u32;

                    ram_events_fixed.push(fixed);

                    let addr = event.addr_bytes();
                    let val = event.val_bytes();

                    for b in 0..4 {
                        ctrl_tb.set_b32(
                            MlKemCtrlColumns::RAM_ADDR + b,
                            ctrl_row,
                            Block32::from(addr[b] as u32),
                        )?;
                        ctrl_tb.set_b32(
                            MlKemCtrlColumns::RAM_VAL + b,
                            ctrl_row,
                            Block32::from(val[b] as u32),
                        )?;
                    }

                    ctrl_tb.set_b32(
                        MlKemCtrlColumns::RAM_VAL_PACKED,
                        ctrl_row,
                        Block32::from(event.val),
                    )?;

                    ctrl_tb.set_bit(
                        MlKemCtrlColumns::RAM_IS_WRITE,
                        ctrl_row,
                        if event.is_write { Bit::ONE } else { Bit::ZERO },
                    )?;
                    ctrl_tb.set_bit(MlKemCtrlColumns::RAM_SELECTOR, ctrl_row, Bit::ONE)?;
                    ctrl_tb.set_bit(MlKemCtrlColumns::KEC_BIND_HI_SEL, ctrl_row, Bit::ONE)?;
                }
                CtrlDispatch::Io {
                    addr,
                    value,
                    pad_kind,
                } => {
                    let event = ram::MemoryEvent::write(*addr, ctrl_row as u32, *value);
                    ram_events_fixed.push(event.clone());

                    let addr_b = event.addr_bytes();
                    let val_b = event.val_bytes();

                    for b in 0..4 {
                        ctrl_tb.set_b32(
                            MlKemCtrlColumns::RAM_ADDR + b,
                            ctrl_row,
                            Block32::from(addr_b[b] as u32),
                        )?;
                        ctrl_tb.set_b32(
                            MlKemCtrlColumns::RAM_VAL + b,
                            ctrl_row,
                            Block32::from(val_b[b] as u32),
                        )?;
                    }

                    ctrl_tb.set_b32(
                        MlKemCtrlColumns::RAM_VAL_PACKED,
                        ctrl_row,
                        Block32::from(*value),
                    )?;
                    ctrl_tb.set_b32(MlKemCtrlColumns::IO_DATA, ctrl_row, Block32::from(*value))?;
                    ctrl_tb.set_bit(MlKemCtrlColumns::RAM_IS_WRITE, ctrl_row, Bit::ONE)?;
                    ctrl_tb.set_bit(MlKemCtrlColumns::RAM_SELECTOR, ctrl_row, Bit::ONE)?;

                    match pad_kind {
                        None => {
                            ctrl_tb.set_bit(MlKemCtrlColumns::IO_SELECTOR, ctrl_row, Bit::ONE)?;
                            ctrl_tb.set_b32(
                                MlKemCtrlColumns::REQUEST_IDX_OUT,
                                ctrl_row,
                                Block32::from(io_data_counter),
                            )?;

                            io_data_counter += 1;
                        }
                        Some(kind) => {
                            ctrl_tb.set_bit(MlKemCtrlColumns::PAD_SEL, ctrl_row, Bit::ONE)?;

                            match kind {
                                PadKind::First => {
                                    ctrl_tb.set_bit(
                                        MlKemCtrlColumns::PAD_FIRST,
                                        ctrl_row,
                                        Bit::ONE,
                                    )?;
                                }
                                PadKind::Last => {
                                    ctrl_tb.set_bit(
                                        MlKemCtrlColumns::PAD_LAST,
                                        ctrl_row,
                                        Bit::ONE,
                                    )?;
                                }
                                PadKind::Mid => {}
                            }
                        }
                    }
                }
                CtrlDispatch::IoLaneBindLo {
                    delta,
                    lane_idx,
                    event,
                } => {
                    let mut fixed = event.clone();
                    fixed.clk = ctrl_row as u32;

                    ram_events_fixed.push(fixed);

                    let addr_b = event.addr_bytes();
                    let val_b = event.val_bytes();

                    for b in 0..4 {
                        ctrl_tb.set_b32(
                            MlKemCtrlColumns::RAM_ADDR + b,
                            ctrl_row,
                            Block32::from(addr_b[b] as u32),
                        )?;
                        ctrl_tb.set_b32(
                            MlKemCtrlColumns::RAM_VAL + b,
                            ctrl_row,
                            Block32::from(val_b[b] as u32),
                        )?;
                    }

                    ctrl_tb.set_b32(
                        MlKemCtrlColumns::RAM_VAL_PACKED,
                        ctrl_row,
                        Block32::from(event.val),
                    )?;
                    ctrl_tb.set_b32(
                        MlKemCtrlColumns::IO_LANE_LO,
                        ctrl_row,
                        Block32::from(event.val),
                    )?;
                    ctrl_tb.set_b64(
                        MlKemCtrlColumns::KEC_LANE_DELTA,
                        ctrl_row,
                        Block64::from(*delta),
                    )?;
                    ctrl_tb.set_b32(
                        MlKemCtrlColumns::KEC_LANE_IDX,
                        ctrl_row,
                        Block32::from(*lane_idx),
                    )?;
                    ctrl_tb.set_bit(MlKemCtrlColumns::RAM_SELECTOR, ctrl_row, Bit::ONE)?;
                    ctrl_tb.set_bit(MlKemCtrlColumns::IO_LANE_BIND_SEL, ctrl_row, Bit::ONE)?;
                    ctrl_tb.set_bit(MlKemCtrlColumns::KEC_BIND_LO_SEL, ctrl_row, Bit::ONE)?;
                }
                CtrlDispatch::HCtBindSel => {
                    for i in 0..4 {
                        ctrl_tb.set_b64(
                            MlKemCtrlColumns::HASH_REF + i,
                            ctrl_row,
                            Block64::from(rate_regs[i]),
                        )?;
                    }

                    ctrl_tb.set_bit(MlKemCtrlColumns::H_CT_BIND_SEL, ctrl_row, Bit::ONE)?;
                }
                CtrlDispatch::HCtPrimeBindSel => {
                    for i in 0..4 {
                        let val = Block64::from(rate_regs[i]);
                        ctrl_tb.set_b64(MlKemCtrlColumns::KECCAK_LANES + i, ctrl_row, val)?;
                        ctrl_tb.set_b64(MlKemCtrlColumns::HASH_CT_PRIME + i, ctrl_row, val)?;
                    }

                    ctrl_tb.set_bit(MlKemCtrlColumns::H_CT_PRIME_BIND_SEL, ctrl_row, Bit::ONE)?;
                }
                CtrlDispatch::IoLaneBindHi { event } => {
                    let mut fixed = event.clone();
                    fixed.clk = ctrl_row as u32;

                    ram_events_fixed.push(fixed);

                    let addr_b = event.addr_bytes();
                    let val_b = event.val_bytes();

                    for b in 0..4 {
                        ctrl_tb.set_b32(
                            MlKemCtrlColumns::RAM_ADDR + b,
                            ctrl_row,
                            Block32::from(addr_b[b] as u32),
                        )?;
                        ctrl_tb.set_b32(
                            MlKemCtrlColumns::RAM_VAL + b,
                            ctrl_row,
                            Block32::from(val_b[b] as u32),
                        )?;
                    }

                    ctrl_tb.set_b32(
                        MlKemCtrlColumns::RAM_VAL_PACKED,
                        ctrl_row,
                        Block32::from(event.val),
                    )?;
                    ctrl_tb.set_b32(
                        MlKemCtrlColumns::IO_LANE_HI,
                        ctrl_row,
                        Block32::from(event.val),
                    )?;
                    ctrl_tb.set_bit(MlKemCtrlColumns::RAM_SELECTOR, ctrl_row, Bit::ONE)?;
                    ctrl_tb.set_bit(MlKemCtrlColumns::IO_LANE_BIND_SEL, ctrl_row, Bit::ONE)?;
                    ctrl_tb.set_bit(MlKemCtrlColumns::KEC_BIND_HI_SEL, ctrl_row, Bit::ONE)?;
                }
            }

            set_phase(&mut ctrl_tb, ctrl_row, *phase)?;
            ctrl_row += 1;

            if let CtrlDispatch::HCtBindSel = dispatch {
                h_ct_bind_seen = true;
            }
            if let CtrlDispatch::HCtPrimeBindSel = dispatch {
                h_ct_prime_bind_seen = true;
            }
            if let CtrlDispatch::KPrimeBindSel = dispatch {
                k_prime_bind_seen = true;
            }
            if let CtrlDispatch::KBarBindSel = dispatch {
                k_bar_bind_seen = true;
            }
        }

        // Write rate registers to the
        // first padding row so the
        // sticky constraint on the
        // last active row sees
        // reg_next = reg (cyclic boundary).
        if ctrl_row < self.params.ctrl_rows {
            for i in 0..25 {
                ctrl_tb.set_b64(
                    MlKemCtrlColumns::RATE_REG + i,
                    ctrl_row,
                    Block64::from(rate_regs[i]),
                )?;
            }

            for i in 0..4 {
                ctrl_tb.set_b64(
                    MlKemCtrlColumns::HASH_REF + i,
                    ctrl_row,
                    Block64::from(hash_ref_carry[i]),
                )?;
                ctrl_tb.set_b64(
                    MlKemCtrlColumns::HASH_CT_PRIME + i,
                    ctrl_row,
                    Block64::from(hash_ct_prime_carry[i]),
                )?;
            }

            for i in 0..4 {
                ctrl_tb.set_b32(
                    MlKemCtrlColumns::K_PRIME_LO + i,
                    ctrl_row,
                    Block32::from(k_prime_lo_carry[i]),
                )?;
                ctrl_tb.set_b32(
                    MlKemCtrlColumns::K_PRIME_HI + i,
                    ctrl_row,
                    Block32::from(k_prime_hi_carry[i]),
                )?;
                ctrl_tb.set_b32(
                    MlKemCtrlColumns::K_BAR_LO + i,
                    ctrl_row,
                    Block32::from(k_bar_lo_carry[i]),
                )?;
                ctrl_tb.set_b32(
                    MlKemCtrlColumns::K_BAR_HI + i,
                    ctrl_row,
                    Block32::from(k_bar_hi_carry[i]),
                )?;
                ctrl_tb.set_b32(
                    MlKemCtrlColumns::SS_LO + i,
                    ctrl_row,
                    Block32::from(ss_lo_carry[i]),
                )?;
                ctrl_tb.set_b32(
                    MlKemCtrlColumns::SS_HI + i,
                    ctrl_row,
                    Block32::from(ss_hi_carry[i]),
                )?;
            }

            if ct_match_carry {
                ctrl_tb.set_bit(MlKemCtrlColumns::CT_MATCH, ctrl_row, Bit::ONE)?;
            }
        }

        // Ghost Protocol
        ctrl_tb.fill_selector(MlKemCtrlColumns::S_ACTIVE, ctrl_row)?;

        let ctrl_trace = ctrl_tb.build();

        // 5. Twiddle ROW trace
        for entry in twiddle_entries.iter_mut() {
            if !entry.is_mulonly {
                continue;
            }

            let rows = wbind_ctrl_rows
                .get_mut(&(entry.butterfly_idx, entry.w))
                .ok_or(errors::Error::Protocol {
                    protocol: "mlkem_trace",
                    message: "mulonly twiddle entry has no matching W-bind ctrl row",
                })?;

            entry.request_idx_tr = rows.pop().ok_or(errors::Error::Protocol {
                protocol: "mlkem_trace",
                message: "W-bind ctrl rows exhausted before twiddle mulonly entries",
            })?;
        }

        let twiddle_trace =
            twiddle_rom::generate_twiddle_rom_trace(&twiddle_entries, self.params.twiddle_rows)?;

        // 6. Keccak trace
        let keccak_trace = keccak::generate_keccak_trace(
            &keccak_inputs,
            Some(&keccak_request_idx_pairs),
            self.params.keccak_rows,
        )?;

        // 7. Basemul trace
        let mut bm_ops_with_request_idx: Vec<basemul::BasemulOp> = result.basemul_ops.clone();
        for (i, ctrl_row) in bm_dispatch_ctrl_rows.iter().enumerate() {
            bm_ops_with_request_idx[i].request_idx = *ctrl_row;
        }

        let bm_trace = basemul::generate_basemul_trace(
            MLKEM_Q,
            &bm_ops_with_request_idx,
            self.params.basemul_rows,
        )?;

        // 8. RAM trace
        let ram_trace = ram::generate_ram_trace(&ram_events_fixed, self.params.ram_rows)?;

        Ok(vec![
            ctrl_trace,
            keccak_trace,
            ntt_trace,
            twiddle_trace,
            bm_trace,
            ram_trace,
        ])
    }
}
