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

use crate::chiplets::keccak::KECCAK_LANE_LABELS;
use crate::chiplets::pqc::mldsa::{KEC_INPUT_BIND_BUS_ID, MLDSA_DATA_BUS_ID};
use crate::chiplets::pqc::ntt::NttChiplet;
use crate::chiplets::pqc::twiddle_rom;
use crate::{KeccakChiplet, RamChiplet};
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use hekate_core::trace::ColumnType;
use hekate_math::TowerField;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::constraint::ConstraintAst;
use hekate_program::permutation::{PermutationCheckSpec, Source, REQUEST_IDX_LABEL};
use hekate_program::{define_columns, Air};

// =================================================================
// ML-DSA Control Chiplet
// =================================================================

define_columns! {
    pub MlDsaCtrlColumns {
        // I/O column for main trace bus.
        IO_DATA: B32,
        IO_SELECTOR: Bit,

        // SHA3 padding sub-selector.
        PAD_SEL: Bit,

        // Keccak dispatch columns.
        KECCAK_LANES: [B64; 25],
        KECCAK_SELECTOR: Bit,

        // Sponge state:
        // sticky rate-lane registers.
        RATE_REG: [B64; 25],

        // Sponge control.
        KEC_IS_OUTPUT: Bit,
        SPONGE_INIT: Bit,
        SHAKE_128: Bit,

        // NTT dispatch columns.
        NTT_A: B32,
        NTT_B: B32,
        NTT_A_OUT: B32,
        NTT_B_OUT: B32,
        NTT_LAYER: B32,
        NTT_BUTTERFLY: B32,
        NTT_INSTANCE: B32,
        NTT_SELECTOR: Bit,

        // RAM columns.
        RAM_ADDR: [B32; 4],
        RAM_VAL: [B32; 4],
        RAM_VAL_PACKED: B32,
        RAM_IS_WRITE: Bit,
        RAM_SELECTOR: Bit,

        // W-side binding columns.
        W_BIND_BFLY_IDX: B32,
        W_BIND_SELECTOR: Bit,

        // NTT boundary binding.
        BOUND_POS: B32,
        BOUND_IN_SEL: Bit,
        BOUND_OUT_SEL: Bit,

        // Keccak input binding columns.
        KEC_LANE_ONE_HOT: [Bit; 21],
        KEC_LANE_DELTA: B64,
        KEC_LANE_IDX: B32,
        KEC_INPUT_REF_SEL: Bit,
        KEC_BIND_LO_SEL: Bit,
        KEC_BIND_HI_SEL: Bit,

        // Raw-byte IO → Keccak binding.
        IO_LANE_LO: B32,
        IO_LANE_HI: B32,
        IO_LANE_BIND_SEL: Bit,
        H_INPUT_SEL: Bit,

        // Sticky H(pk) input active marker.
        H_PK_ACTIVE: Bit,

        // NormCheck dispatch columns.
        NC_VALUE: B32,
        NC_IDX: B32,
        NC_SELECTOR: Bit,

        // HighBits dispatch columns.
        HB_R: B32,
        HB_R1: B32,
        HB_R0: B32,
        HB_IDX: B32,
        HB_H_BIT: Bit,
        HB_W1_PRIME: B32,
        HB_R0_NONZERO: Bit,
        HB_R0_INV: B128,
        HB_SELECTOR: Bit,

        // RATE_REG snapshot selectors.
        // Fire after Keccak output rows.
        TR_BIND_SEL: Bit,
        MU_BIND_SEL: Bit,
        CTILDE_PRIME_BIND_SEL: Bit,

        // Monotonic 0->1 flags.
        TR_BIND_SEEN: Bit,
        MU_BIND_SEEN: Bit,
        CTILDE_PRIME_BIND_SEEN: Bit,
        CTILDE_REF_BIND_SEEN: Bit,

        // c̃ reference from signature bytes.
        // Carried sticky to CMP row.
        CTILDE_REF: [B64; 4],
        CTILDE_REF_BIND_SEL: Bit,

        // Hash comparison:
        // c̃ (CTILDE_REF) vs c̃' (RATE_REG).
        CMP_SELECTOR: Bit,
        HASH_EQ_LO: Bit,
        HASH_EQ_HI: Bit,
        HASH_DIFF_INV_LO: B128,
        HASH_DIFF_INV_HI: B128,

        // Partner-side row index for the
        // ml_dsa_data outward CPU bus.
        REQUEST_IDX_OUT: B32,

        // Control flow.
        S_ACTIVE: Bit,

        // Phase state machine.
        // 8 one-hot phases, forward-only.
        PH_IO: Bit,
        PH_EXPAND_SAMPLE: Bit,
        PH_NTT_FORWARD: Bit,
        PH_POINTWISE_MUL: Bit,
        PH_NTT_INVERSE: Bit,
        PH_USE_HINT: Bit,
        PH_HASH_COMPARE: Bit,
        PH_NORM_CHECK: Bit,
    }
}

/// ML-DSA Control Chiplet.
///
/// Internal "CPU" of the ML-DSA composite:
/// routes data between main trace and
/// sub-chiplets.
#[derive(Clone, Debug)]
pub struct MlDsaCtrlChiplet {
    #[allow(dead_code)]
    pub num_rows: usize,
}

impl MlDsaCtrlChiplet {
    pub fn new(num_rows: usize) -> Self {
        assert!(num_rows.is_power_of_two());
        Self { num_rows }
    }

    // =============================================================
    // Linking Specs
    // =============================================================

    /// External "ml_dsa_data" bus.
    pub fn main_linking_spec() -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(MlDsaCtrlColumns::IO_DATA),
                    b"kappa_mldsa_d0" as &[u8],
                ),
                (
                    Source::Column(MlDsaCtrlColumns::REQUEST_IDX_OUT),
                    REQUEST_IDX_LABEL,
                ),
            ],
            Some(MlDsaCtrlColumns::IO_SELECTOR),
        )
    }

    /// Internal "keccak_link" bus.
    fn keccak_linking_spec() -> PermutationCheckSpec {
        let mut sources = Vec::with_capacity(26);
        for (i, label) in KECCAK_LANE_LABELS.iter().enumerate() {
            sources.push((Source::Column(MlDsaCtrlColumns::KECCAK_LANES + i), *label));
        }

        sources.push((Source::RowIndexLeBytes(4), REQUEST_IDX_LABEL));

        PermutationCheckSpec::new(sources, Some(MlDsaCtrlColumns::KECCAK_SELECTOR))
    }

    /// Internal "ntt_data" bus.
    fn ntt_linking_spec() -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(MlDsaCtrlColumns::NTT_A),
                    b"kappa_ntt_a" as &[u8],
                ),
                (
                    Source::Column(MlDsaCtrlColumns::NTT_B),
                    b"kappa_ntt_b" as &[u8],
                ),
                (
                    Source::Column(MlDsaCtrlColumns::NTT_A_OUT),
                    b"kappa_ntt_a_out" as &[u8],
                ),
                (
                    Source::Column(MlDsaCtrlColumns::NTT_B_OUT),
                    b"kappa_ntt_b_out" as &[u8],
                ),
                (
                    Source::Column(MlDsaCtrlColumns::NTT_LAYER),
                    b"kappa_ntt_layer" as &[u8],
                ),
                (
                    Source::Column(MlDsaCtrlColumns::NTT_BUTTERFLY),
                    b"kappa_ntt_bfly" as &[u8],
                ),
                (
                    Source::Column(MlDsaCtrlColumns::NTT_INSTANCE),
                    b"kappa_ntt_inst" as &[u8],
                ),
            ],
            Some(MlDsaCtrlColumns::NTT_SELECTOR),
        )
        .with_clock_waiver(
            "see pqc/mldsa/ctrl.rs: paired with NttChiplet::data_linking_spec; \
             (ntt_instance, layer, butterfly_idx) triple is positional, AIR-forced \
             unique across active rows",
        )
    }

    /// Internal "ram_link" bus.
    fn ram_linking_spec() -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(MlDsaCtrlColumns::RAM_ADDR),
                    b"kappa_addr_b0" as &[u8],
                ),
                (
                    Source::Column(MlDsaCtrlColumns::RAM_ADDR + 1),
                    b"kappa_addr_b1" as &[u8],
                ),
                (
                    Source::Column(MlDsaCtrlColumns::RAM_ADDR + 2),
                    b"kappa_addr_b2" as &[u8],
                ),
                (
                    Source::Column(MlDsaCtrlColumns::RAM_ADDR + 3),
                    b"kappa_addr_b3" as &[u8],
                ),
                (Source::RowIndexByte(0), b"kappa_clk_b0" as &[u8]),
                (Source::RowIndexByte(1), b"kappa_clk_b1" as &[u8]),
                (Source::RowIndexByte(2), b"kappa_clk_b2" as &[u8]),
                (Source::RowIndexByte(3), b"kappa_clk_b3" as &[u8]),
                (
                    Source::Column(MlDsaCtrlColumns::RAM_VAL),
                    b"kappa_val_b0" as &[u8],
                ),
                (
                    Source::Column(MlDsaCtrlColumns::RAM_VAL + 1),
                    b"kappa_val_b1" as &[u8],
                ),
                (
                    Source::Column(MlDsaCtrlColumns::RAM_VAL + 2),
                    b"kappa_val_b2" as &[u8],
                ),
                (
                    Source::Column(MlDsaCtrlColumns::RAM_VAL + 3),
                    b"kappa_val_b3" as &[u8],
                ),
                (
                    Source::Column(MlDsaCtrlColumns::RAM_IS_WRITE),
                    b"kappa_is_write" as &[u8],
                ),
            ],
            Some(MlDsaCtrlColumns::RAM_SELECTOR),
        )
    }

    /// W-side binding bus.
    pub fn w_binding_linking_spec() -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(MlDsaCtrlColumns::W_BIND_BFLY_IDX),
                    b"kappa_wb_bfly" as &[u8],
                ),
                (
                    Source::Column(MlDsaCtrlColumns::RAM_VAL_PACKED),
                    b"kappa_wb_w" as &[u8],
                ),
                (Source::RowIndexLeBytes(4), REQUEST_IDX_LABEL),
            ],
            Some(MlDsaCtrlColumns::W_BIND_SELECTOR),
        )
    }

    /// NTT boundary input bus.
    pub fn bound_in_linking_spec() -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(MlDsaCtrlColumns::NTT_INSTANCE),
                    b"kappa_bound_inst" as &[u8],
                ),
                (
                    Source::Column(MlDsaCtrlColumns::BOUND_POS),
                    b"kappa_bound_pos" as &[u8],
                ),
                (
                    Source::Column(MlDsaCtrlColumns::RAM_VAL_PACKED),
                    b"kappa_bound_val" as &[u8],
                ),
            ],
            Some(MlDsaCtrlColumns::BOUND_IN_SEL),
        )
        .with_clock_waiver(
            "see pqc/mldsa/ctrl.rs: paired with NttChiplet::bound_in_spec; \
             (NTT_INSTANCE, BOUND_POS) is positional, AIR-forced unique per instance",
        )
    }

    /// NTT boundary output bus.
    pub fn bound_out_linking_spec() -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(MlDsaCtrlColumns::NTT_INSTANCE),
                    b"kappa_bound_inst" as &[u8],
                ),
                (
                    Source::Column(MlDsaCtrlColumns::BOUND_POS),
                    b"kappa_bound_pos" as &[u8],
                ),
                (
                    Source::Column(MlDsaCtrlColumns::RAM_VAL_PACKED),
                    b"kappa_bound_val" as &[u8],
                ),
            ],
            Some(MlDsaCtrlColumns::BOUND_OUT_SEL),
        )
        .with_clock_waiver(
            "see pqc/mldsa/ctrl.rs: paired with NttChiplet::bound_out_spec; \
             (NTT_INSTANCE, BOUND_POS) is positional, AIR-forced unique per instance",
        )
    }

    /// Keccak input ref (consume side).
    fn kec_input_ref_spec() -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(MlDsaCtrlColumns::KEC_LANE_DELTA),
                    b"kappa_kib_delta" as &[u8],
                ),
                (
                    Source::Column(MlDsaCtrlColumns::KEC_LANE_IDX),
                    b"kappa_kib_idx" as &[u8],
                ),
            ],
            Some(MlDsaCtrlColumns::KEC_INPUT_REF_SEL),
        )
        .with_clock_waiver(
            "see pqc/mldsa/ctrl.rs: KEC_LANE_IDX is positional; paired with \
             kec_input_bind_spec on the produce side",
        )
    }

    /// Keccak input bind (produce side).
    fn kec_input_bind_spec() -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(MlDsaCtrlColumns::KEC_LANE_DELTA),
                    b"kappa_kib_delta" as &[u8],
                ),
                (
                    Source::Column(MlDsaCtrlColumns::KEC_LANE_IDX),
                    b"kappa_kib_idx" as &[u8],
                ),
            ],
            Some(MlDsaCtrlColumns::KEC_BIND_LO_SEL),
        )
        .with_clock_waiver(
            "see pqc/mldsa/ctrl.rs: KEC_LANE_IDX is positional; paired with \
             kec_input_ref_spec on the consume side",
        )
    }

    /// NormCheck dispatch bus.
    fn norm_check_linking_spec() -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(MlDsaCtrlColumns::NC_VALUE),
                    b"kappa_nc_value" as &[u8],
                ),
                (
                    Source::Column(MlDsaCtrlColumns::NC_IDX),
                    b"kappa_nc_idx" as &[u8],
                ),
            ],
            Some(MlDsaCtrlColumns::NC_SELECTOR),
        )
        .with_clock_waiver(
            "see pqc/mldsa/ctrl.rs: paired with NormCheckChiplet::linking_spec; NC_IDX \
             is positional, AIR-forced unique per active row",
        )
    }

    /// HighBits dispatch bus.
    fn highbits_linking_spec() -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(MlDsaCtrlColumns::HB_R),
                    b"kappa_hb_r" as &[u8],
                ),
                (
                    Source::Column(MlDsaCtrlColumns::HB_R1),
                    b"kappa_hb_r1" as &[u8],
                ),
                (
                    Source::Column(MlDsaCtrlColumns::HB_R0),
                    b"kappa_hb_r0" as &[u8],
                ),
                (
                    Source::Column(MlDsaCtrlColumns::HB_IDX),
                    b"kappa_hb_idx" as &[u8],
                ),
                (
                    Source::Column(MlDsaCtrlColumns::HB_H_BIT),
                    b"kappa_hb_h" as &[u8],
                ),
                (
                    Source::Column(MlDsaCtrlColumns::HB_W1_PRIME),
                    b"kappa_hb_w1" as &[u8],
                ),
            ],
            Some(MlDsaCtrlColumns::HB_SELECTOR),
        )
        .with_clock_waiver(
            "see pqc/mldsa/ctrl.rs: paired with HighBitsChiplet::linking_spec; HB_IDX \
             is positional, AIR-forced unique per active row",
        )
    }
}

// =================================================================
// Air Implementation
// =================================================================

impl<F: TowerField> Air<F> for MlDsaCtrlChiplet {
    fn name(&self) -> String {
        "MlDsaCtrlChiplet".into()
    }

    fn num_columns(&self) -> usize {
        MlDsaCtrlColumns::NUM_COLUMNS
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: once_cell::race::OnceBox<Vec<ColumnType>> = once_cell::race::OnceBox::new();
        LAYOUT.get_or_init(|| Box::new(MlDsaCtrlColumns::build_layout()))
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![
            (MLDSA_DATA_BUS_ID.into(), Self::main_linking_spec()),
            (KeccakChiplet::BUS_ID.into(), Self::keccak_linking_spec()),
            (NttChiplet::DATA_BUS_ID.into(), Self::ntt_linking_spec()),
            (RamChiplet::BUS_ID.into(), Self::ram_linking_spec()),
            (
                twiddle_rom::TWIDDLE_W_BINDING_BUS_ID.into(),
                Self::w_binding_linking_spec(),
            ),
            (
                NttChiplet::BOUND_IN_BUS_ID.into(),
                Self::bound_in_linking_spec(),
            ),
            (
                NttChiplet::BOUND_OUT_BUS_ID.into(),
                Self::bound_out_linking_spec(),
            ),
            (KEC_INPUT_BIND_BUS_ID.into(), Self::kec_input_ref_spec()),
            (KEC_INPUT_BIND_BUS_ID.into(), Self::kec_input_bind_spec()),
            ("norm_check".into(), Self::norm_check_linking_spec()),
            ("highbits".into(), Self::highbits_linking_spec()),
        ]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();

        let s_active = cs.col(MlDsaCtrlColumns::S_ACTIVE);

        // =========================================================
        // Selector booleanity
        // =========================================================

        cs.assert_boolean(s_active);
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::IO_SELECTOR));
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::PAD_SEL));
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::KECCAK_SELECTOR));
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::KEC_IS_OUTPUT));
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::SPONGE_INIT));
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::SHAKE_128));
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::NTT_SELECTOR));
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::RAM_SELECTOR));
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::RAM_IS_WRITE));
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::W_BIND_SELECTOR));
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::BOUND_IN_SEL));
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::BOUND_OUT_SEL));
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::KEC_INPUT_REF_SEL));
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::KEC_BIND_LO_SEL));
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::KEC_BIND_HI_SEL));
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::IO_LANE_BIND_SEL));
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::H_INPUT_SEL));
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::H_PK_ACTIVE));
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::NC_SELECTOR));
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::HB_H_BIT));
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::HB_R0_NONZERO));
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::HB_SELECTOR));
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::TR_BIND_SEL));
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::MU_BIND_SEL));
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::CTILDE_PRIME_BIND_SEL));
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::TR_BIND_SEEN));
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::MU_BIND_SEEN));
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::CTILDE_PRIME_BIND_SEEN));
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::CTILDE_REF_BIND_SEEN));
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::CTILDE_REF_BIND_SEL));
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::CMP_SELECTOR));
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::HASH_EQ_LO));
        cs.assert_boolean(cs.col(MlDsaCtrlColumns::HASH_EQ_HI));

        // Phase booleanity
        let ph_io = cs.col(MlDsaCtrlColumns::PH_IO);
        let ph_expand = cs.col(MlDsaCtrlColumns::PH_EXPAND_SAMPLE);
        let ph_ntt_fwd = cs.col(MlDsaCtrlColumns::PH_NTT_FORWARD);
        let ph_pw_mul = cs.col(MlDsaCtrlColumns::PH_POINTWISE_MUL);
        let ph_ntt_inv = cs.col(MlDsaCtrlColumns::PH_NTT_INVERSE);
        let ph_hint = cs.col(MlDsaCtrlColumns::PH_USE_HINT);
        let ph_hash_cmp = cs.col(MlDsaCtrlColumns::PH_HASH_COMPARE);
        let ph_norm = cs.col(MlDsaCtrlColumns::PH_NORM_CHECK);

        cs.assert_boolean(ph_io);
        cs.assert_boolean(ph_expand);
        cs.assert_boolean(ph_ntt_fwd);
        cs.assert_boolean(ph_pw_mul);
        cs.assert_boolean(ph_ntt_inv);
        cs.assert_boolean(ph_hint);
        cs.assert_boolean(ph_hash_cmp);
        cs.assert_boolean(ph_norm);

        // KEC_LANE_ONE_HOT booleanity
        for i in 0..21 {
            cs.assert_boolean(cs.col(MlDsaCtrlColumns::KEC_LANE_ONE_HOT + i));
        }

        let one = cs.constant(F::ONE);
        let not_active = one + s_active;

        // =========================================================
        // One-hot phase:
        // exactly one phase bit active per active row
        // =========================================================

        let ph = [
            ph_io,
            ph_expand,
            ph_ntt_fwd,
            ph_pw_mul,
            ph_ntt_inv,
            ph_hint,
            ph_hash_cmp,
            ph_norm,
        ];

        // (a) At most one via
        // pairwise orthogonality.
        for i in 0..ph.len() {
            for j in (i + 1)..ph.len() {
                cs.constrain(ph[i] * ph[j]);
            }
        }

        // (b) At least one on active rows
        let phase_sum = ph_io
            + ph_expand
            + ph_ntt_fwd
            + ph_pw_mul
            + ph_ntt_inv
            + ph_hint
            + ph_hash_cmp
            + ph_norm;

        cs.constrain(s_active * (one + phase_sum));

        // Phases zero on padding rows
        for &p in &ph {
            cs.constrain(p * not_active);
        }

        // =========================================================
        // Forward-only phase transitions.
        // If current is phase i, next row
        // cannot be any earlier phase j < i.
        // =========================================================

        for i in 1..ph.len() {
            for j in 0..i {
                cs.constrain(s_active * ph[i] * cs.next(MlDsaCtrlColumns::PH_IO + j));
            }
        }

        // =========================================================
        // Selector mutual exclusivity
        // =========================================================

        let io_sel = cs.col(MlDsaCtrlColumns::IO_SELECTOR);
        let kec_sel = cs.col(MlDsaCtrlColumns::KECCAK_SELECTOR);
        let ntt_sel = cs.col(MlDsaCtrlColumns::NTT_SELECTOR);
        let ram_sel = cs.col(MlDsaCtrlColumns::RAM_SELECTOR);
        let nc_sel = cs.col(MlDsaCtrlColumns::NC_SELECTOR);
        let hb_sel = cs.col(MlDsaCtrlColumns::HB_SELECTOR);
        let cmp_sel = cs.col(MlDsaCtrlColumns::CMP_SELECTOR);

        // IO vs all dispatchers
        cs.constrain(io_sel * kec_sel);
        cs.constrain(io_sel * ntt_sel);
        cs.constrain(io_sel * nc_sel);
        cs.constrain(io_sel * hb_sel);
        cs.constrain(io_sel * cmp_sel);

        // Keccak vs NTT / NormCheck / HighBits / CMP
        cs.constrain(kec_sel * ntt_sel);
        cs.constrain(kec_sel * nc_sel);
        cs.constrain(kec_sel * hb_sel);
        cs.constrain(kec_sel * cmp_sel);

        // NTT vs NormCheck / HighBits / CMP
        cs.constrain(ntt_sel * nc_sel);
        cs.constrain(ntt_sel * hb_sel);
        cs.constrain(ntt_sel * cmp_sel);

        // NormCheck vs HighBits / CMP
        cs.constrain(nc_sel * hb_sel);
        cs.constrain(nc_sel * cmp_sel);

        // HighBits vs CMP
        cs.constrain(hb_sel * cmp_sel);

        // NTT + RAM, NC + RAM, HB + RAM
        // co-activation allowed (data binding).
        cs.constrain(io_sel * ram_sel);
        cs.constrain(kec_sel * ram_sel);
        cs.constrain(cmp_sel * ram_sel);

        // =========================================================
        // Phase-selector consistency
        // =========================================================

        // IO phase:
        // only IO_SELECTOR active
        cs.constrain(ph_io * ntt_sel);
        cs.constrain(ph_io * kec_sel);
        cs.constrain(ph_io * nc_sel);
        cs.constrain(ph_io * hb_sel);
        cs.constrain(ph_io * cmp_sel);

        // ExpandSample:
        // Keccak + RAM only
        cs.constrain(ph_expand * ntt_sel);
        cs.constrain(ph_expand * nc_sel);
        cs.constrain(ph_expand * hb_sel);
        cs.constrain(ph_expand * cmp_sel);
        cs.constrain(ph_expand * io_sel);

        // NTT forward:
        // NTT + RAM only
        cs.constrain(ph_ntt_fwd * kec_sel);
        cs.constrain(ph_ntt_fwd * nc_sel);
        cs.constrain(ph_ntt_fwd * hb_sel);
        cs.constrain(ph_ntt_fwd * cmp_sel);
        cs.constrain(ph_ntt_fwd * io_sel);

        // Pointwise mul:
        // NTT + RAM only
        cs.constrain(ph_pw_mul * kec_sel);
        cs.constrain(ph_pw_mul * nc_sel);
        cs.constrain(ph_pw_mul * hb_sel);
        cs.constrain(ph_pw_mul * cmp_sel);
        cs.constrain(ph_pw_mul * io_sel);

        // NTT inverse:
        // NTT + RAM only
        cs.constrain(ph_ntt_inv * kec_sel);
        cs.constrain(ph_ntt_inv * nc_sel);
        cs.constrain(ph_ntt_inv * hb_sel);
        cs.constrain(ph_ntt_inv * cmp_sel);
        cs.constrain(ph_ntt_inv * io_sel);

        // UseHint:
        // HighBits + RAM (value binding)
        cs.constrain(ph_hint * kec_sel);
        cs.constrain(ph_hint * ntt_sel);
        cs.constrain(ph_hint * nc_sel);
        cs.constrain(ph_hint * cmp_sel);
        cs.constrain(ph_hint * io_sel);

        // HashCompare:
        // Keccak + CMP + RAM
        cs.constrain(ph_hash_cmp * ntt_sel);
        cs.constrain(ph_hash_cmp * nc_sel);
        cs.constrain(ph_hash_cmp * hb_sel);
        cs.constrain(ph_hash_cmp * io_sel);

        // NormCheck:
        // NormCheck + RAM (value binding)
        cs.constrain(ph_norm * kec_sel);
        cs.constrain(ph_norm * ntt_sel);
        cs.constrain(ph_norm * hb_sel);
        cs.constrain(ph_norm * cmp_sel);
        cs.constrain(ph_norm * io_sel);

        // =========================================================
        // Ghost Protocol:
        // inactive rows are sterile.
        // All event selectors zero on padding.
        // =========================================================

        let ghost_sels = [
            MlDsaCtrlColumns::IO_SELECTOR,
            MlDsaCtrlColumns::PAD_SEL,
            MlDsaCtrlColumns::KECCAK_SELECTOR,
            MlDsaCtrlColumns::NTT_SELECTOR,
            MlDsaCtrlColumns::RAM_SELECTOR,
            MlDsaCtrlColumns::NC_SELECTOR,
            MlDsaCtrlColumns::HB_SELECTOR,
            MlDsaCtrlColumns::CMP_SELECTOR,
            MlDsaCtrlColumns::W_BIND_SELECTOR,
            MlDsaCtrlColumns::BOUND_IN_SEL,
            MlDsaCtrlColumns::BOUND_OUT_SEL,
            MlDsaCtrlColumns::KEC_INPUT_REF_SEL,
            MlDsaCtrlColumns::KEC_BIND_LO_SEL,
            MlDsaCtrlColumns::KEC_BIND_HI_SEL,
            MlDsaCtrlColumns::IO_LANE_BIND_SEL,
            MlDsaCtrlColumns::H_INPUT_SEL,
            MlDsaCtrlColumns::TR_BIND_SEL,
            MlDsaCtrlColumns::MU_BIND_SEL,
            MlDsaCtrlColumns::CTILDE_PRIME_BIND_SEL,
            MlDsaCtrlColumns::CTILDE_REF_BIND_SEL,
        ];

        for &sel in &ghost_sels {
            cs.constrain(not_active * cs.col(sel));
        }

        // =========================================================
        // Binding constraints
        // =========================================================

        let ntt_b = cs.col(MlDsaCtrlColumns::NTT_B);
        let ram_val_packed = cs.col(MlDsaCtrlColumns::RAM_VAL_PACKED);

        cs.constrain(ntt_sel * ram_sel * (ntt_b + ram_val_packed));

        // NC rows must co-activate RAM.
        let nc_val = cs.col(MlDsaCtrlColumns::NC_VALUE);
        cs.constrain(nc_sel * (one + ram_sel));
        cs.constrain(nc_sel * (nc_val + ram_val_packed));

        // HB rows must co-activate RAM.
        let hb_r = cs.col(MlDsaCtrlColumns::HB_R);
        cs.constrain(hb_sel * (one + ram_sel));
        cs.constrain(hb_sel * (hb_r + ram_val_packed));

        // r0_nonzero bidirectional check.
        // Forward:
        // r0_nonzero=0 forces r0=0.
        // Reverse:
        // r0_nonzero=1 forces r0 invertible.
        let hb_r0 = cs.col(MlDsaCtrlColumns::HB_R0);
        let r0_nz = cs.col(MlDsaCtrlColumns::HB_R0_NONZERO);
        let r0_inv = cs.col(MlDsaCtrlColumns::HB_R0_INV);

        cs.constrain(hb_sel * (one + r0_nz) * hb_r0);
        cs.constrain(hb_sel * r0_nz * (hb_r0 * r0_inv + one));

        // UseHint:
        // w1_prime = f(h_bit, r1, r0_nonzero).
        // h=0: w1 = r1
        // h=1: w1 = (r1 ± 1) mod m,
        // direction from r0_nonzero.
        //
        // Verified inside the HighBits chiplet
        // via bit decomposition. The bus carries
        // (h_bit, w1_prime) to link ctrl and chiplet.

        // Schedule guarantees NttRam row
        // immediately precedes its WBind row.
        let next_wb_sel = cs.next(MlDsaCtrlColumns::W_BIND_SELECTOR);
        cs.constrain(next_wb_sel * (one + ntt_sel));

        let ntt_bfly = cs.col(MlDsaCtrlColumns::NTT_BUTTERFLY);
        let next_wb_bfly = cs.next(MlDsaCtrlColumns::W_BIND_BFLY_IDX);

        cs.constrain(ntt_sel * next_wb_sel * (ntt_bfly + next_wb_bfly));

        // Relies on schedule grouping
        // by (phase, instance):
        // non-NTT rows (WBind, BoundaryRam)
        // separate instance blocks.
        let ntt_inst = cs.col(MlDsaCtrlColumns::NTT_INSTANCE);
        let next_ntt_sel = cs.next(MlDsaCtrlColumns::NTT_SELECTOR);
        let next_ntt_inst = cs.next(MlDsaCtrlColumns::NTT_INSTANCE);

        cs.constrain(ntt_sel * next_ntt_sel * (ntt_inst + next_ntt_inst));

        // =========================================================
        // Sponge state carry
        // =========================================================

        let kec_out = cs.col(MlDsaCtrlColumns::KEC_IS_OUTPUT);
        let sponge_init = cs.col(MlDsaCtrlColumns::SPONGE_INIT);
        let sponge_init_next = cs.next(MlDsaCtrlColumns::SPONGE_INIT);
        let shake_128 = cs.col(MlDsaCtrlColumns::SHAKE_128);
        let kec_input = kec_sel * (one + kec_out);

        // reg[next] = reg (carry) when kec_out=0, init_next=0
        // reg[next] = lane (update) when kec_out=1, init_next=0
        // reg[next] = 0 (reset) when init_next=1
        for i in 0..25 {
            let reg = cs.col(MlDsaCtrlColumns::RATE_REG + i);
            let reg_next = cs.next(MlDsaCtrlColumns::RATE_REG + i);
            let lane = cs.col(MlDsaCtrlColumns::KECCAK_LANES + i);

            cs.constrain(
                s_active * (one + sponge_init_next) * (reg_next + reg + kec_out * (reg + lane)),
            );
        }

        // SPONGE_INIT=1 -> all registers = 0
        for i in 0..25 {
            let reg = cs.col(MlDsaCtrlColumns::RATE_REG + i);
            cs.constrain(sponge_init * reg);
        }

        // Capacity lane continuity:
        // on Keccak input rows, capacity lanes
        // must equal RATE_REG (previous output).
        //
        // SHAKE-256 rate=17:
        //   lanes 17..24 are capacity.
        // SHAKE-128 rate=21:
        //   lanes 21..24 are capacity.
        // Lanes 21..24:
        //   always capacity.
        for i in 21..25 {
            let reg = cs.col(MlDsaCtrlColumns::RATE_REG + i);
            let lane = cs.col(MlDsaCtrlColumns::KECCAK_LANES + i);

            cs.constrain(kec_input * (lane + reg));
        }

        // Lanes 17..20:
        // capacity when SHAKE-256 only.
        for i in 17..21 {
            let reg = cs.col(MlDsaCtrlColumns::RATE_REG + i);
            let lane = cs.col(MlDsaCtrlColumns::KECCAK_LANES + i);

            cs.constrain(kec_input * (one + shake_128) * (lane + reg));
        }

        // =========================================================
        // IO / hash comparison binding
        // =========================================================

        let pad_sel = cs.col(MlDsaCtrlColumns::PAD_SEL);

        cs.constrain(io_sel * (cs.col(MlDsaCtrlColumns::IO_DATA) + ram_val_packed));
        cs.constrain(io_sel * pad_sel);

        // c̃ == c̃' hard equality on CMP row.
        // CTILDE_REF (c̃ from signature)
        // vs
        // RATE_REG (c̃' from hash).
        for i in 0..4 {
            cs.constrain(
                cmp_sel
                    * (cs.col(MlDsaCtrlColumns::CTILDE_REF + i)
                        + cs.col(MlDsaCtrlColumns::RATE_REG + i)),
            );
        }

        // Bidirectional HASH_EQ verification
        // (ML-KEM pattern). TAU-combined diffs
        // reduce 4 B64 lanes to 2 B128 checks.
        let tau = cs.constant(F::EXTENSION_TAU);

        let diff_lo = (cs.col(MlDsaCtrlColumns::CTILDE_REF) + cs.col(MlDsaCtrlColumns::RATE_REG))
            + (cs.col(MlDsaCtrlColumns::CTILDE_REF + 1) + cs.col(MlDsaCtrlColumns::RATE_REG + 1))
                * tau;

        let diff_hi = (cs.col(MlDsaCtrlColumns::CTILDE_REF + 2)
            + cs.col(MlDsaCtrlColumns::RATE_REG + 2))
            + (cs.col(MlDsaCtrlColumns::CTILDE_REF + 3) + cs.col(MlDsaCtrlColumns::RATE_REG + 3))
                * tau;

        let eq_lo = cs.col(MlDsaCtrlColumns::HASH_EQ_LO);
        let eq_hi = cs.col(MlDsaCtrlColumns::HASH_EQ_HI);
        let inv_lo = cs.col(MlDsaCtrlColumns::HASH_DIFF_INV_LO);
        let inv_hi = cs.col(MlDsaCtrlColumns::HASH_DIFF_INV_HI);

        // eq=1 -> diff=0
        cs.constrain(cmp_sel * eq_lo * diff_lo);
        cs.constrain(cmp_sel * eq_hi * diff_hi);

        // eq=0 -> diff has valid inverse
        cs.constrain(cmp_sel * (one + eq_lo) * (diff_lo * inv_lo + one));
        cs.constrain(cmp_sel * (one + eq_hi) * (diff_hi * inv_hi + one));

        // Proof existence = verdict:
        // both halves must match.
        cs.constrain(cmp_sel * (one + eq_lo));
        cs.constrain(cmp_sel * (one + eq_hi));

        // Bus multiset ensures correct
        // (value, idx) pairs. First-row
        // anchor prevents reordering.
        cs.constrain(
            (one + nc_sel)
                * cs.next(MlDsaCtrlColumns::NC_SELECTOR)
                * cs.next(MlDsaCtrlColumns::NC_IDX),
        );
        cs.constrain(
            (one + hb_sel)
                * cs.next(MlDsaCtrlColumns::HB_SELECTOR)
                * cs.next(MlDsaCtrlColumns::HB_IDX),
        );

        // =========================================================
        // BIND_SEEN monotonic flags
        // =========================================================

        let bind_seen_cols = [
            MlDsaCtrlColumns::TR_BIND_SEEN,
            MlDsaCtrlColumns::MU_BIND_SEEN,
            MlDsaCtrlColumns::CTILDE_PRIME_BIND_SEEN,
            MlDsaCtrlColumns::CTILDE_REF_BIND_SEEN,
        ];

        for &col in &bind_seen_cols {
            let seen = cs.col(col);
            let seen_next = cs.next(col);

            // Monotonic:
            // once set, stays set.
            cs.constrain(s_active * seen * (one + seen_next));

            // Last active row must have SEEN=1
            let s_active_next = cs.next(MlDsaCtrlColumns::S_ACTIVE);
            cs.constrain(s_active * (one + s_active_next) * (one + seen));
        }

        // CTILDE_REF_BIND_SEEN 0->1 transition
        // requires CMP_SELECTOR on that row.
        let cref_seen = cs.col(MlDsaCtrlColumns::CTILDE_REF_BIND_SEEN);
        let cref_seen_next = cs.next(MlDsaCtrlColumns::CTILDE_REF_BIND_SEEN);

        cs.constrain(
            s_active
                * (one + cref_seen)
                * cref_seen_next
                * (one + cs.next(MlDsaCtrlColumns::CMP_SELECTOR)),
        );

        // BIND_SEL anchored to Keccak output rows.
        // BIND_SEL can only fire when both
        // KECCAK_SELECTOR=1 and KEC_IS_OUTPUT=1.
        let tr_bind_sel = cs.col(MlDsaCtrlColumns::TR_BIND_SEL);
        let mu_bind_sel = cs.col(MlDsaCtrlColumns::MU_BIND_SEL);
        let cp_bind_sel = cs.col(MlDsaCtrlColumns::CTILDE_PRIME_BIND_SEL);

        let kec_output = kec_sel * kec_out;

        cs.constrain(tr_bind_sel * (one + kec_output));
        cs.constrain(mu_bind_sel * (one + kec_output));
        cs.constrain(cp_bind_sel * (one + kec_output));

        cs.constrain(tr_bind_sel * mu_bind_sel);
        cs.constrain(tr_bind_sel * cp_bind_sel);
        cs.constrain(mu_bind_sel * cp_bind_sel);

        // BIND_SEEN 0->1 requires BIND_SEL
        // on the Keccak output row (current).
        let bind_pairs = [
            (
                MlDsaCtrlColumns::TR_BIND_SEEN,
                MlDsaCtrlColumns::TR_BIND_SEL,
            ),
            (
                MlDsaCtrlColumns::MU_BIND_SEEN,
                MlDsaCtrlColumns::MU_BIND_SEL,
            ),
            (
                MlDsaCtrlColumns::CTILDE_PRIME_BIND_SEEN,
                MlDsaCtrlColumns::CTILDE_PRIME_BIND_SEL,
            ),
        ];

        for &(seen_col, sel_col) in &bind_pairs {
            let seen = cs.col(seen_col);
            let seen_next = cs.next(seen_col);
            let sel = cs.col(sel_col);

            cs.constrain(s_active * (one + seen) * seen_next * (one + sel));
        }

        cs.build()
    }
}
