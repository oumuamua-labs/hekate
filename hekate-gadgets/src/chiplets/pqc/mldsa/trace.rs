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

//! ML-DSA Trace Generation

use super::ctrl::MlDsaCtrlColumns;
use super::witness::{
    KeccakCallTag, MlDsaPublicKey, MlDsaSignature, MlDsaVerifyResult, ml_dsa_verify_traced,
};
use super::{MLDSA_Q, MlDsaChiplet, Phase};
use crate::chiplets::pqc::ntt;
use crate::chiplets::pqc::{high_bits, norm_check, twiddle_rom};
use crate::chiplets::{keccak, ram};
use alloc::collections::BTreeMap;
use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;
use hekate_core::errors;
use hekate_core::trace::{ColumnTrace, TraceBuilder, TraceCompatibleField};
use hekate_math::{
    Bit, Block32, Block64, Block128, Flat, HardwareField, PackableField, TowerField,
};
// =================================================================
// Ctrl Dispatch Tags
// =================================================================

/// Tags each ctrl row with the data
/// needed to fill its columns.
#[derive(Clone)]
pub(crate) enum CtrlDispatch<'a> {
    /// IO row:
    /// public input byte chunk.
    /// PAD_SEL binding.
    Io {
        data: u32,
        #[allow(dead_code)]
        is_pad: bool,
    },

    /// Keccak permutation input row.
    KeccakInput {
        lanes: &'a [u64; 25],
        is_output: bool,
        sponge_init: bool,
        is_shake128: bool,
    },

    /// NTT + RAM co-active row.
    NttRam {
        op: &'a ntt::NttOp,
        ram: &'a ram::MemoryEvent,
    },

    /// Standalone RAM row.
    Ram { event: &'a ram::MemoryEvent },

    /// W-side binding only (no RAM bus).
    WBind {
        bfly_idx: u32,
        w_value: u32,
        instance: u32,
    },

    /// NTT boundary RAM row.
    BoundaryRam {
        event: &'a ram::MemoryEvent,
        ntt_instance: u32,
        bound_pos: u32,
        is_input: bool,
    },

    /// NormCheck dispatch row.
    /// Co-activates RAM for
    /// NC-RAM value binding.
    NormCheck {
        value: u32,
        idx: u32,
        ram_event: ram::MemoryEvent,
    },

    /// HighBits dispatch row.
    /// Co-activates RAM for
    /// HB-RAM value binding.
    HighBits {
        r: u32,
        r1: u32,
        r0: u32,
        idx: u32,
        h_bit: bool,
        w1_prime: u32,
        ram_event: ram::MemoryEvent,
    },

    /// Hash comparison row (c̃ vs c̃').
    HashCompare,

    /// Empty active row to break
    /// NTT instance contiguity.
    Separator { instance: u32 },
}

impl<'a> CtrlDispatch<'a> {
    fn ntt_instance_key(&self) -> u32 {
        match self {
            Self::NttRam { op, .. } => match op {
                ntt::NttOp::Butterfly(b) => b.ntt_instance,
                ntt::NttOp::MulOnly(m) => m.flow_instance,
                ntt::NttOp::FlowCompanion(_) => u32::MAX,
            },
            Self::WBind { instance, .. } => *instance,
            Self::BoundaryRam { ntt_instance, .. } => *ntt_instance,
            Self::Separator { instance } => *instance,
            _ => u32::MAX,
        }
    }
}

// =================================================================
// Trace Generation
// =================================================================

impl<F> MlDsaChiplet<F>
where
    F: TowerField + TraceCompatibleField + PackableField + HardwareField + 'static,
    <F as PackableField>::Packed: Copy + Send + Sync,
    Flat<F>: Send + Sync,
{
    pub fn generate_traces(
        &self,
        pk: &MlDsaPublicKey,
        sig: &MlDsaSignature,
        msg: &[u8],
    ) -> errors::Result<Vec<ColumnTrace>> {
        let (result, ntt_ops) = ml_dsa_verify_traced(pk, sig, msg);
        self.generate_traces_inner(&result, &ntt_ops)
    }

    fn generate_traces_inner(
        &self,
        result: &MlDsaVerifyResult,
        ntt_ops: &[ntt::NttOp],
    ) -> errors::Result<Vec<ColumnTrace>> {
        let level = self.level;
        let gamma2 = level.gamma2;

        // ==========================================================
        // Sub-chiplet traces (independent of ctrl)
        // ==========================================================

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

        let ntt_trace = ntt::generate_ntt_trace(MLDSA_Q, ntt_ops, self.params.ntt_rows)?;

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

        let nc_trace = norm_check::generate_norm_check_trace(
            MLDSA_Q,
            level.z_bound(),
            &result.norm_check_ops,
            self.params.norm_rows,
        )?;

        let hb_trace = high_bits::generate_highbits_trace(
            MLDSA_Q,
            level.highbits_divisor(),
            &result.highbits_ops,
            self.params.highbits_rows,
        )?;

        // ==========================================================
        // Ctrl dispatch schedule
        // ==========================================================

        let ctrl_layout = MlDsaCtrlColumns::build_layout();
        let ctrl_vars = self.params.ctrl_rows.trailing_zeros() as usize;

        let mut ctrl_tb = TraceBuilder::new(&ctrl_layout, ctrl_vars)?;

        let mut schedule: Vec<(Phase, CtrlDispatch)> = Vec::new();

        // IO rows
        let mut io_buf = Vec::new();
        io_buf.extend_from_slice(&result.c_tilde);

        while io_buf.len() % 4 != 0 {
            io_buf.push(0);
        }

        for chunk in io_buf.chunks(4) {
            let val = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            schedule.push((
                Phase::Io,
                CtrlDispatch::Io {
                    data: val,
                    is_pad: false,
                },
            ));
        }

        // Keccak calls.
        // Input binding deferred to Phase 4.

        for (i, (input, output)) in result.keccak_calls.iter().enumerate() {
            let ph = result.keccak_phases[i];
            let (sponge_init, _, is_shake128) = result.keccak_sponge_meta[i];

            schedule.push((
                ph,
                CtrlDispatch::KeccakInput {
                    lanes: input,
                    is_output: false,
                    sponge_init,
                    is_shake128,
                },
            ));

            schedule.push((
                ph,
                CtrlDispatch::KeccakInput {
                    lanes: output,
                    is_output: true,
                    sponge_init: false,
                    is_shake128: false,
                },
            ));
        }

        // RAM events with binding classification
        let mut ram_to_ntt: BTreeMap<usize, usize> = BTreeMap::new();
        let mut ram_to_w_bfly: BTreeMap<usize, u32> = BTreeMap::new();
        let mut ram_is_boundary: BTreeSet<usize> = BTreeSet::new();

        for &(ntt_idx, ram_idx) in &result.ntt_ram_bindings {
            ram_to_ntt.insert(ram_idx, ntt_idx);
        }

        for &(ram_idx, bfly_idx) in &result.w_side_bindings {
            ram_to_w_bfly.insert(ram_idx, bfly_idx);
        }

        for &(_, _, ram_idx, _) in &result.ntt_boundary_bindings {
            ram_is_boundary.insert(ram_idx);
        }

        for (i, event) in result.ram_events.iter().enumerate() {
            if ram_is_boundary.contains(&i) {
                continue;
            }

            let ph = result.ram_phases[i];

            let has_ntt = ram_to_ntt.get(&i).copied();
            let has_w = ram_to_w_bfly.get(&i).copied();

            match (has_ntt, has_w) {
                (Some(ntt_idx), Some(bfly_idx)) => {
                    let (w_val, inst) = match &ntt_ops[ntt_idx] {
                        ntt::NttOp::MulOnly(m) => (m.w, m.flow_instance),
                        _ => unreachable!("co-bound NTT+W event must be MulOnly"),
                    };

                    schedule.push((
                        ph,
                        CtrlDispatch::NttRam {
                            op: &ntt_ops[ntt_idx],
                            ram: event,
                        },
                    ));
                    schedule.push((
                        ph,
                        CtrlDispatch::WBind {
                            bfly_idx,
                            w_value: w_val,
                            instance: inst,
                        },
                    ));
                }
                (Some(ntt_idx), None) => {
                    schedule.push((
                        ph,
                        CtrlDispatch::NttRam {
                            op: &ntt_ops[ntt_idx],
                            ram: event,
                        },
                    ));
                }
                (None, Some(_)) => {
                    unreachable!("ML-DSA w_side events are always co-bound with NTT");
                }
                (None, None) => {
                    schedule.push((ph, CtrlDispatch::Ram { event }));
                }
            }
        }

        // NTT boundary bindings
        for &(inst, pos, ram_idx, is_input) in &result.ntt_boundary_bindings {
            schedule.push((
                result.ram_phases[ram_idx],
                CtrlDispatch::BoundaryRam {
                    event: &result.ram_events[ram_idx],
                    ntt_instance: inst,
                    bound_pos: pos,
                    is_input,
                },
            ));
        }

        // NormCheck dispatch;
        // co-activates RAM for value binding.
        for (i, op) in result.norm_check_ops.iter().enumerate() {
            schedule.push((
                result.norm_phases[i],
                CtrlDispatch::NormCheck {
                    value: op.value,
                    idx: op.idx,
                    ram_event: ram::MemoryEvent {
                        addr: op.ram_addr,
                        clk: 0,
                        val: op.value,
                        is_write: false,
                    },
                },
            ));
        }

        // HighBits dispatch
        let divisor = 2 * gamma2;
        for (i, op) in result.highbits_ops.iter().enumerate() {
            let r1 = op.r / divisor;
            let r0 = op.r % divisor;

            schedule.push((
                result.highbits_phases[i],
                CtrlDispatch::HighBits {
                    r: op.r,
                    r1,
                    r0,
                    idx: op.idx,
                    h_bit: op.h_bit,
                    w1_prime: op.w1_prime,
                    ram_event: ram::MemoryEvent {
                        addr: op.ram_addr,
                        clk: 0,
                        val: op.r,
                        is_write: false,
                    },
                },
            ));
        }

        // HashCompare
        schedule.push((Phase::HashCompare, CtrlDispatch::HashCompare));

        // INTT instance separators.
        // Pushed AFTER all NttRam entries
        // so stable sort places them at the
        // END of each instance group.
        {
            let mut prev_inst: Option<u32> = None;
            for (i, op) in ntt_ops.iter().enumerate() {
                if result.ntt_phases[i] != Phase::NttInverse {
                    continue;
                }

                let inst = match op {
                    ntt::NttOp::Butterfly(b) => b.ntt_instance,
                    ntt::NttOp::MulOnly(m) => m.flow_instance,
                    ntt::NttOp::FlowCompanion(_) => continue,
                };

                if let Some(p) = prev_inst
                    && p != inst
                {
                    schedule.push((Phase::NttInverse, CtrlDispatch::Separator { instance: p }));
                }

                prev_inst = Some(inst);
            }
        }

        // Sort by (phase, instance),
        // groups NTT ops per-instance
        // so consecutive NTT_SELECTOR=1
        // rows share the same instance.
        schedule.sort_by_key(|(ph, d)| (*ph as u8, d.ntt_instance_key()));

        // ==========================================================
        // Fill ctrl rows
        // ==========================================================

        let mut ctrl_row = 0usize;
        let mut rate_regs = [0u64; 25];

        let mut keccak_call_idx = 0usize;
        let mut seen_tr = false;
        let mut seen_mu = false;
        let mut seen_ctilde_prime = false;
        let mut seen_ctilde_ref = false;

        let mut ram_events_fixed: Vec<ram::MemoryEvent> = Vec::new();

        let mut io_data_counter: u32 = 0;
        let mut keccak_request_idx_pairs: Vec<(u32, u32)> =
            Vec::with_capacity(result.keccak_calls.len());
        let mut pending_keccak_input_ctrl_row: Option<u32> = None;

        let mut wbind_ctrl_rows: BTreeMap<(u32, u32), Vec<u32>> = BTreeMap::new();

        for (phase, dispatch) in &schedule {
            if ctrl_row >= self.params.ctrl_rows {
                break;
            }

            // Sticky RATE_REG carry
            for i in 0..25 {
                ctrl_tb.set_b64(
                    MlDsaCtrlColumns::RATE_REG + i,
                    ctrl_row,
                    Block64::from(rate_regs[i]),
                )?;
            }

            ctrl_tb.set_bit(MlDsaCtrlColumns::S_ACTIVE, ctrl_row, Bit::ONE)?;

            if seen_tr {
                ctrl_tb.set_bit(MlDsaCtrlColumns::TR_BIND_SEEN, ctrl_row, Bit::ONE)?;
            }
            if seen_mu {
                ctrl_tb.set_bit(MlDsaCtrlColumns::MU_BIND_SEEN, ctrl_row, Bit::ONE)?;
            }
            if seen_ctilde_prime {
                ctrl_tb.set_bit(MlDsaCtrlColumns::CTILDE_PRIME_BIND_SEEN, ctrl_row, Bit::ONE)?;
            }
            if seen_ctilde_ref {
                ctrl_tb.set_bit(MlDsaCtrlColumns::CTILDE_REF_BIND_SEEN, ctrl_row, Bit::ONE)?;
            }

            match dispatch {
                CtrlDispatch::Io { data, .. } => {
                    ctrl_tb.set_b32(MlDsaCtrlColumns::IO_DATA, ctrl_row, Block32::from(*data))?;
                    ctrl_tb.set_b32(
                        MlDsaCtrlColumns::RAM_VAL_PACKED,
                        ctrl_row,
                        Block32::from(*data),
                    )?;
                    ctrl_tb.set_bit(MlDsaCtrlColumns::IO_SELECTOR, ctrl_row, Bit::ONE)?;
                    ctrl_tb.set_b32(
                        MlDsaCtrlColumns::REQUEST_IDX_OUT,
                        ctrl_row,
                        Block32::from(io_data_counter),
                    )?;

                    io_data_counter += 1;
                }
                CtrlDispatch::KeccakInput {
                    lanes,
                    is_output,
                    sponge_init,
                    is_shake128,
                } => {
                    if *is_output {
                        let in_row = pending_keccak_input_ctrl_row.take().ok_or(
                            errors::Error::Protocol {
                                protocol: "mldsa_trace",
                                message: "Keccak output dispatched without preceding input",
                            },
                        )?;

                        keccak_request_idx_pairs.push((in_row, ctrl_row as u32));
                    } else {
                        pending_keccak_input_ctrl_row = Some(ctrl_row as u32);
                    }

                    for (lane, &val) in lanes.iter().enumerate() {
                        ctrl_tb.set_b64(
                            MlDsaCtrlColumns::KECCAK_LANES + lane,
                            ctrl_row,
                            Block64::from(val),
                        )?;
                    }

                    ctrl_tb.set_bit(MlDsaCtrlColumns::KECCAK_SELECTOR, ctrl_row, Bit::ONE)?;

                    if *is_output {
                        ctrl_tb.set_bit(MlDsaCtrlColumns::KEC_IS_OUTPUT, ctrl_row, Bit::ONE)?;

                        rate_regs.copy_from_slice(lanes.as_slice());

                        assert!(
                            keccak_call_idx < result.call_tag.len(),
                            "keccak_call_idx {keccak_call_idx} >= call_tag.len() {}",
                            result.call_tag.len()
                        );

                        match result.call_tag[keccak_call_idx] {
                            KeccakCallTag::HashPk => {
                                seen_tr = true;

                                ctrl_tb.set_bit(
                                    MlDsaCtrlColumns::TR_BIND_SEL,
                                    ctrl_row,
                                    Bit::ONE,
                                )?;
                            }
                            KeccakCallTag::HashMu => {
                                seen_mu = true;

                                ctrl_tb.set_bit(
                                    MlDsaCtrlColumns::MU_BIND_SEL,
                                    ctrl_row,
                                    Bit::ONE,
                                )?;
                            }
                            KeccakCallTag::HashCompare => {
                                seen_ctilde_prime = true;

                                ctrl_tb.set_bit(
                                    MlDsaCtrlColumns::CTILDE_PRIME_BIND_SEL,
                                    ctrl_row,
                                    Bit::ONE,
                                )?;
                            }
                            _ => {}
                        }

                        keccak_call_idx += 1;
                    } else {
                        if *sponge_init {
                            rate_regs = [0u64; 25];

                            ctrl_tb.set_bit(MlDsaCtrlColumns::SPONGE_INIT, ctrl_row, Bit::ONE)?;

                            for i in 0..25 {
                                ctrl_tb.set_b64(
                                    MlDsaCtrlColumns::RATE_REG + i,
                                    ctrl_row,
                                    Block64::from(0u64),
                                )?;
                            }
                        }

                        if *is_shake128 {
                            ctrl_tb.set_bit(MlDsaCtrlColumns::SHAKE_128, ctrl_row, Bit::ONE)?;
                        }
                    }
                }
                CtrlDispatch::NttRam { op, ram } => {
                    set_ntt_columns(&mut ctrl_tb, ctrl_row, op)?;

                    let mut fixed = (*ram).clone();
                    fixed.clk = ctrl_row as u32;

                    ram_events_fixed.push(fixed);

                    set_ram_columns(&mut ctrl_tb, ctrl_row, ram)?;
                }
                CtrlDispatch::Ram { event } => {
                    let mut fixed = (*event).clone();
                    fixed.clk = ctrl_row as u32;

                    ram_events_fixed.push(fixed);

                    set_ram_columns(&mut ctrl_tb, ctrl_row, event)?;
                }
                CtrlDispatch::WBind {
                    bfly_idx, w_value, ..
                } => {
                    ctrl_tb.set_b32(
                        MlDsaCtrlColumns::W_BIND_BFLY_IDX,
                        ctrl_row,
                        Block32::from(*bfly_idx),
                    )?;
                    ctrl_tb.set_b32(
                        MlDsaCtrlColumns::RAM_VAL_PACKED,
                        ctrl_row,
                        Block32::from(*w_value),
                    )?;
                    ctrl_tb.set_bit(MlDsaCtrlColumns::W_BIND_SELECTOR, ctrl_row, Bit::ONE)?;

                    wbind_ctrl_rows
                        .entry((*bfly_idx, *w_value))
                        .or_default()
                        .push(ctrl_row as u32);
                }
                CtrlDispatch::BoundaryRam {
                    event,
                    ntt_instance,
                    bound_pos,
                    is_input,
                } => {
                    let mut fixed = (*event).clone();
                    fixed.clk = ctrl_row as u32;

                    ram_events_fixed.push(fixed);

                    set_ram_columns(&mut ctrl_tb, ctrl_row, event)?;

                    ctrl_tb.set_b32(
                        MlDsaCtrlColumns::NTT_INSTANCE,
                        ctrl_row,
                        Block32::from(*ntt_instance),
                    )?;
                    ctrl_tb.set_b32(
                        MlDsaCtrlColumns::BOUND_POS,
                        ctrl_row,
                        Block32::from(*bound_pos),
                    )?;

                    if *is_input {
                        ctrl_tb.set_bit(MlDsaCtrlColumns::BOUND_IN_SEL, ctrl_row, Bit::ONE)?;
                    } else {
                        ctrl_tb.set_bit(MlDsaCtrlColumns::BOUND_OUT_SEL, ctrl_row, Bit::ONE)?;
                    }
                }
                CtrlDispatch::NormCheck {
                    value,
                    idx,
                    ram_event,
                } => {
                    ctrl_tb.set_b32(MlDsaCtrlColumns::NC_VALUE, ctrl_row, Block32::from(*value))?;
                    ctrl_tb.set_b32(MlDsaCtrlColumns::NC_IDX, ctrl_row, Block32::from(*idx))?;
                    ctrl_tb.set_bit(MlDsaCtrlColumns::NC_SELECTOR, ctrl_row, Bit::ONE)?;

                    let mut fixed = ram_event.clone();
                    fixed.clk = ctrl_row as u32;

                    ram_events_fixed.push(fixed);

                    set_ram_columns(&mut ctrl_tb, ctrl_row, ram_event)?;
                }
                CtrlDispatch::HighBits {
                    r,
                    r1,
                    r0,
                    idx,
                    h_bit,
                    w1_prime,
                    ram_event,
                } => {
                    ctrl_tb.set_b32(MlDsaCtrlColumns::HB_R, ctrl_row, Block32::from(*r))?;
                    ctrl_tb.set_b32(MlDsaCtrlColumns::HB_R1, ctrl_row, Block32::from(*r1))?;
                    ctrl_tb.set_b32(MlDsaCtrlColumns::HB_R0, ctrl_row, Block32::from(*r0))?;
                    ctrl_tb.set_b32(MlDsaCtrlColumns::HB_IDX, ctrl_row, Block32::from(*idx))?;
                    ctrl_tb.set_bit(MlDsaCtrlColumns::HB_SELECTOR, ctrl_row, Bit::ONE)?;

                    if *h_bit {
                        ctrl_tb.set_bit(MlDsaCtrlColumns::HB_H_BIT, ctrl_row, Bit::ONE)?;
                    }

                    ctrl_tb.set_b32(
                        MlDsaCtrlColumns::HB_W1_PRIME,
                        ctrl_row,
                        Block32::from(*w1_prime),
                    )?;

                    let r0_val = *r0;
                    if r0_val > 0 {
                        ctrl_tb.set_bit(MlDsaCtrlColumns::HB_R0_NONZERO, ctrl_row, Bit::ONE)?;

                        let r0_field = Block128::from(r0_val as u128);
                        let r0_inv_field = r0_field.invert();

                        ctrl_tb.set_b128(MlDsaCtrlColumns::HB_R0_INV, ctrl_row, r0_inv_field)?;
                    }

                    let mut fixed = ram_event.clone();
                    fixed.clk = ctrl_row as u32;

                    ram_events_fixed.push(fixed);

                    set_ram_columns(&mut ctrl_tb, ctrl_row, ram_event)?;
                }
                CtrlDispatch::HashCompare => {
                    ctrl_tb.set_bit(MlDsaCtrlColumns::CMP_SELECTOR, ctrl_row, Bit::ONE)?;

                    // c̃ from signature -> CTILDE_REF
                    // (4 × B64 = 32 bytes).
                    for i in 0..4 {
                        let lo = u32::from_le_bytes([
                            result.c_tilde[i * 8],
                            result.c_tilde[i * 8 + 1],
                            result.c_tilde[i * 8 + 2],
                            result.c_tilde[i * 8 + 3],
                        ]);
                        let hi = u32::from_le_bytes([
                            result.c_tilde[i * 8 + 4],
                            result.c_tilde[i * 8 + 5],
                            result.c_tilde[i * 8 + 6],
                            result.c_tilde[i * 8 + 7],
                        ]);

                        let val = lo as u64 | ((hi as u64) << 32);

                        ctrl_tb.set_b64(
                            MlDsaCtrlColumns::CTILDE_REF + i,
                            ctrl_row,
                            Block64::from(val),
                        )?;
                    }

                    ctrl_tb.set_bit(MlDsaCtrlColumns::HASH_EQ_LO, ctrl_row, Bit::ONE)?;
                    ctrl_tb.set_bit(MlDsaCtrlColumns::HASH_EQ_HI, ctrl_row, Bit::ONE)?;
                    ctrl_tb.set_bit(MlDsaCtrlColumns::CTILDE_REF_BIND_SEEN, ctrl_row, Bit::ONE)?;

                    seen_ctilde_ref = true;
                }
                CtrlDispatch::Separator { .. } => {}
            }

            set_phase(&mut ctrl_tb, ctrl_row, *phase)?;

            ctrl_row += 1;
        }

        // Padding:
        // carry sticky RATE_REG + BIND_SEEN
        // to first padding row
        // for cyclic boundary.
        if ctrl_row < self.params.ctrl_rows {
            for i in 0..25 {
                ctrl_tb.set_b64(
                    MlDsaCtrlColumns::RATE_REG + i,
                    ctrl_row,
                    Block64::from(rate_regs[i]),
                )?;
            }

            if seen_tr {
                ctrl_tb.set_bit(MlDsaCtrlColumns::TR_BIND_SEEN, ctrl_row, Bit::ONE)?;
            }
            if seen_mu {
                ctrl_tb.set_bit(MlDsaCtrlColumns::MU_BIND_SEEN, ctrl_row, Bit::ONE)?;
            }
            if seen_ctilde_prime {
                ctrl_tb.set_bit(MlDsaCtrlColumns::CTILDE_PRIME_BIND_SEEN, ctrl_row, Bit::ONE)?;
            }
            if seen_ctilde_ref {
                ctrl_tb.set_bit(MlDsaCtrlColumns::CTILDE_REF_BIND_SEEN, ctrl_row, Bit::ONE)?;
            }
        }

        let ctrl_trace = ctrl_tb.build();

        // ==========================================================
        // Twiddle ROM trace
        // ==========================================================

        for entry in twiddle_entries.iter_mut() {
            if !entry.is_mulonly {
                continue;
            }

            let rows = wbind_ctrl_rows
                .get_mut(&(entry.butterfly_idx, entry.w))
                .ok_or(errors::Error::Protocol {
                    protocol: "mldsa_trace",
                    message: "mulonly twiddle entry has no matching W-bind ctrl row",
                })?;

            entry.request_idx_tr = rows.pop().ok_or(errors::Error::Protocol {
                protocol: "mldsa_trace",
                message: "W-bind ctrl rows exhausted before twiddle mulonly entries",
            })?;
        }

        let twiddle_trace =
            twiddle_rom::generate_twiddle_rom_trace(&twiddle_entries, self.params.twiddle_rows)?;

        // ==========================================================
        // Keccak trace
        // ==========================================================

        let keccak_trace = keccak::generate_keccak_trace(
            &keccak_inputs,
            Some(&keccak_request_idx_pairs),
            self.params.keccak_rows,
        )?;

        // ==========================================================
        // RAM trace from fixed-clock events
        // ==========================================================

        let ram_trace = ram::generate_ram_trace(&ram_events_fixed, self.params.ram_rows)?;

        // ==========================================================
        // Assemble in composite order:
        // ctrl, keccak, ntt, twiddle,
        // normcheck, highbits, ram
        // ==========================================================

        Ok(vec![
            ctrl_trace,
            keccak_trace,
            ntt_trace,
            twiddle_trace,
            nc_trace,
            hb_trace,
            ram_trace,
        ])
    }
}

// =================================================================
// Schedule Building
// =================================================================

fn set_phase(tb: &mut TraceBuilder, row: usize, phase: Phase) -> errors::Result<()> {
    let col = match phase {
        Phase::Io => MlDsaCtrlColumns::PH_IO,
        Phase::ExpandSample => MlDsaCtrlColumns::PH_EXPAND_SAMPLE,
        Phase::NttForward => MlDsaCtrlColumns::PH_NTT_FORWARD,
        Phase::PointwiseMul => MlDsaCtrlColumns::PH_POINTWISE_MUL,
        Phase::NttInverse => MlDsaCtrlColumns::PH_NTT_INVERSE,
        Phase::UseHint => MlDsaCtrlColumns::PH_USE_HINT,
        Phase::HashCompare => MlDsaCtrlColumns::PH_HASH_COMPARE,
        Phase::NormCheck => MlDsaCtrlColumns::PH_NORM_CHECK,
    };

    tb.set_bit(col, row, Bit::ONE)
}

fn set_ram_columns(
    tb: &mut TraceBuilder,
    row: usize,
    event: &ram::MemoryEvent,
) -> errors::Result<()> {
    let addr_bytes = event.addr.to_le_bytes();
    for b in 0..4 {
        tb.set_b32(
            MlDsaCtrlColumns::RAM_ADDR + b,
            row,
            Block32::from(addr_bytes[b] as u32),
        )?;
    }

    let val_bytes = event.val.to_le_bytes();
    for b in 0..4 {
        tb.set_b32(
            MlDsaCtrlColumns::RAM_VAL + b,
            row,
            Block32::from(val_bytes[b] as u32),
        )?;
    }

    tb.set_b32(
        MlDsaCtrlColumns::RAM_VAL_PACKED,
        row,
        Block32::from(event.val),
    )?;

    if event.is_write {
        tb.set_bit(MlDsaCtrlColumns::RAM_IS_WRITE, row, Bit::ONE)?;
    }

    tb.set_bit(MlDsaCtrlColumns::RAM_SELECTOR, row, Bit::ONE)?;

    Ok(())
}

fn set_ntt_columns(tb: &mut TraceBuilder, row: usize, op: &ntt::NttOp) -> errors::Result<()> {
    match op {
        ntt::NttOp::Butterfly(b) => {
            let wb = ((b.w as u64 * b.b as u64) % MLDSA_Q as u64) as u32;
            let a_out = (b.a as u64 + wb as u64) % MLDSA_Q as u64;
            let b_out = (b.a as u64 + MLDSA_Q as u64 - wb as u64) % MLDSA_Q as u64;

            tb.set_b32(MlDsaCtrlColumns::NTT_A, row, Block32::from(b.a))?;
            tb.set_b32(MlDsaCtrlColumns::NTT_B, row, Block32::from(b.b))?;
            tb.set_b32(
                MlDsaCtrlColumns::NTT_A_OUT,
                row,
                Block32::from(a_out as u32),
            )?;
            tb.set_b32(
                MlDsaCtrlColumns::NTT_B_OUT,
                row,
                Block32::from(b_out as u32),
            )?;
            tb.set_b32(MlDsaCtrlColumns::NTT_LAYER, row, Block32::from(b.layer))?;
            tb.set_b32(
                MlDsaCtrlColumns::NTT_BUTTERFLY,
                row,
                Block32::from(b.butterfly_idx),
            )?;
            tb.set_b32(
                MlDsaCtrlColumns::NTT_INSTANCE,
                row,
                Block32::from(b.ntt_instance),
            )?;
        }
        ntt::NttOp::MulOnly(m) => {
            let wb = ((m.w as u64 * m.b as u64) % MLDSA_Q as u64) as u32;

            tb.set_b32(MlDsaCtrlColumns::NTT_A, row, Block32::from(0u32))?;
            tb.set_b32(MlDsaCtrlColumns::NTT_B, row, Block32::from(m.b))?;
            tb.set_b32(MlDsaCtrlColumns::NTT_A_OUT, row, Block32::from(wb))?;
            tb.set_b32(
                MlDsaCtrlColumns::NTT_B_OUT,
                row,
                Block32::from((MLDSA_Q - wb) % MLDSA_Q),
            )?;
            tb.set_b32(MlDsaCtrlColumns::NTT_LAYER, row, Block32::from(m.layer))?;
            tb.set_b32(
                MlDsaCtrlColumns::NTT_BUTTERFLY,
                row,
                Block32::from(m.butterfly_idx),
            )?;
            tb.set_b32(
                MlDsaCtrlColumns::NTT_INSTANCE,
                row,
                Block32::from(m.flow_instance),
            )?;
        }
        ntt::NttOp::FlowCompanion(_) => {}
    }

    tb.set_bit(MlDsaCtrlColumns::NTT_SELECTOR, row, Bit::ONE)?;

    Ok(())
}
