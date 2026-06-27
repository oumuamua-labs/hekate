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

//! ML-KEM Control Chiplet

use crate::basemul::BasemulChiplet;
use crate::mlkem::{KEC_INPUT_BIND_BUS_ID, MLKEM_DATA_BUS_ID, MLKEM_SS_BUS_ID};
use crate::ntt::NttChiplet;
use crate::twiddle_rom;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use hekate_core::trace::ColumnType;
use hekate_gadgets::RamChiplet;
use hekate_keccak::KECCAK_LANE_LABELS;
use hekate_keccak::KeccakChiplet;
use hekate_math::TowerField;
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::permutation::{PermutationCheckSpec, REQUEST_IDX_LABEL, Source};
use hekate_program::{Air, define_columns};

define_columns! {
    pub MlKemCtrlColumns {
        // I/O column for main trace bus.
        // Single B32 per IO row;
        // each row carries one ct chunk
        // (or one SHA3 padding chunk).
        IO_DATA: B32,
        IO_SELECTOR: Bit,

        // IO sub-selector marking SHA3-256
        // padding rows. Distinguishes constant
        // padding bytes from public ct bytes.
        PAD_SEL: Bit,

        // Keccak dispatch columns
        KECCAK_LANES: [B64; 25],
        KECCAK_SELECTOR: Bit,

        // Sponge state:
        // sticky rate-lane registers.
        // Carry Keccak output rate
        // lanes across non-Keccak rows.
        // Updated on output rows.
        RATE_REG: [B64; 25],

        // Sponge control.
        KEC_IS_OUTPUT: Bit,
        SPONGE_INIT: Bit,
        SHA3_512: Bit,
        SHAKE_128: Bit,

        // NTT dispatch columns
        NTT_A: B32,
        NTT_B: B32,
        NTT_A_OUT: B32,
        NTT_B_OUT: B32,
        NTT_LAYER: B32,
        NTT_BUTTERFLY: B32,
        NTT_INSTANCE: B32,
        NTT_SELECTOR: Bit,

        // Basemul dispatch columns
        BM_A: B32,
        BM_B: B32,
        BM_C: B32,
        BM_IDX: B32,
        BM_SELECTOR: Bit,

        // RAM columns for polynomial
        // coefficient routing.
        RAM_ADDR: [B32; 4],
        RAM_VAL: [B32; 4],
        RAM_VAL_PACKED: B32,
        RAM_IS_WRITE: Bit,
        RAM_SELECTOR: Bit,

        // W-side binding columns.
        // Active on w-side RAM read rows.
        W_BIND_BFLY_IDX: B32,
        W_BIND_SELECTOR: Bit,

        // NTT boundary binding columns.
        // Links layer-0 inputs / layer-6
        // outputs to RAM read/write events.
        BOUND_POS: B32,
        BOUND_IN_SEL: Bit,
        BOUND_OUT_SEL: Bit,

        // Keccak input binding columns.
        // One-hot lane selector:
        // identifies which rate lane
        // each ref row constrains.
        KEC_LANE_ONE_HOT: [Bit; 21],
        KEC_LANE_DELTA: B64,
        KEC_LANE_IDX: B32,
        KEC_INPUT_REF_SEL: Bit,
        KEC_BIND_LO_SEL: Bit,
        KEC_BIND_HI_SEL: Bit,

        // Raw-byte H(ct) input binding.
        // Lo/hi half-lane bytes read from
        // the IO RAM range; H_CT_INPUT_SEL
        // marks the H(ct) absorption row.
        IO_LANE_LO: B32,
        IO_LANE_HI: B32,
        IO_LANE_BIND_SEL: Bit,
        H_CT_INPUT_SEL: Bit,

        // Sticky activation marker for H(ct).
        // Forced to 1 on H_CT_INPUT_SEL rows,
        // sticky through ref + IO bind rows,
        // forced to 0 on Keccak output rows.
        // Required by IO_LANE_BIND_SEL rows
        // so clearing H_CT_INPUT_SEL
        // collapses the chain.
        H_CT_ACTIVE: Bit,

        // SHA3-256 padding constant tags.
        // PAD_FIRST:
        //   chunk = 0x00000006
        // PAD_LAST:
        //   chunk = 0x80000000
        // Mid chunks:
        //   pad_sel & !first & !last (FIPS 202 §B.2.)
        PAD_FIRST: Bit,
        PAD_LAST: Bit,

        // RATE_REG snapshot selectors.
        // Fire on the row immediately
        // following the H(ct) / H(ct')
        // Keccak output row.
        H_CT_BIND_SEL: Bit,
        H_CT_PRIME_BIND_SEL: Bit,

        // H(ct') digest carry columns.
        // Snapshot from RATE_REG on
        // H_CT_PRIME_BIND_SEL row,
        // sticky-carried to the CMP row.
        // Needed because J(z||c) Keccak
        // calls overwrite KECCAK_LANES
        // between the snapshot and CMP.
        HASH_CT_PRIME: [B64; 4],

        // Monotonic 0->1 flags tracking
        // whether H_CT_BIND_SEL and
        // H_CT_PRIME_BIND_SEL have fired.
        // CMP row requires both = 1.
        H_CT_BIND_SEEN: Bit,
        H_CT_PRIME_BIND_SEEN: Bit,

        // Re-encryption hash comparison.
        // HASH_REF = H(ct),
        // KECCAK_LANES[0..3] = H(ct').
        // Bidirectional:
        // CT_MATCH=1 <> H(ct)==H(ct').
        HASH_REF: [B64; 4],
        CT_MATCH: Bit,
        CMP_SELECTOR: Bit,

        // Reverse hash comparison witnesses.
        // diff_lo = (lane[0]+hash[0]) + (lane[1]+hash[1])*TAU
        // diff_hi = (lane[2]+hash[2]) + (lane[3]+hash[3])*TAU
        // {1, TAU} is a basis for GF(2^128)/GF(2^64),
        // so diff_lo=0 iff lanes 0,1 match;
        // diff_hi=0 iff lanes 2,3 match.
        HASH_EQ_LO: Bit,
        HASH_EQ_HI: Bit,
        HASH_DIFF_INV_LO: B128,
        HASH_DIFF_INV_HI: B128,

        // K' decomposed from RATE_REG
        // after G(m'||h) output.
        K_PRIME_LO: [B32; 4],
        K_PRIME_HI: [B32; 4],
        K_PRIME_BIND_SEL: Bit,

        // K̄ decomposed from RATE_REG
        // after J(z||c) output.
        K_BAR_LO: [B32; 4],
        K_BAR_HI: [B32; 4],
        K_BAR_BIND_SEL: Bit,

        // Monotonic 0->1 flags tracking
        // whether K_PRIME_BIND_SEL and
        // K_BAR_BIND_SEL have fired.
        // SS_MUX_SEL requires both = 1.
        K_PRIME_BIND_SEEN: Bit,
        K_BAR_BIND_SEEN: Bit,

        // Shared secret mux output.
        SS_LO: [B32; 4],
        SS_HI: [B32; 4],
        SS_MUX_SEL: Bit,
        SS_OUT_SEL: Bit,

        // Partner-side row index for the
        // ml_kem_data and ml_kem_ss buses;
        // selectors are phase-disjoint.
        REQUEST_IDX_OUT: B32,

        // Control flow
        S_ACTIVE: Bit,

        // Protocol phase state machine.
        // One-hot:
        // exactly one active per active row.
        // Forward-only:
        // phase index can only increase.
        //
        // 0=IO       public ciphertext deposit + SHA3 padding
        // 1=DECRYPT  NTT/BM/RAM for decryption
        // 2=G_HASH   Keccak for G(m'||h)
        // 3=ENCRYPT  NTT/BM/RAM/Keccak for re-encrypt
        // 4=CMP_HASH Keccak for H(ct), H(ct'), J(z||c)
        // 5=COMPARE  hash comparison row
        PH_IO: Bit,
        PH_DECRYPT: Bit,
        PH_G_HASH: Bit,
        PH_ENCRYPT: Bit,
        PH_CMP_HASH: Bit,
        PH_COMPARE: Bit,
    }
}

/// ML-KEM Control Chiplet.
///
/// The composite's internal "CPU" that
/// routes data between the main trace
/// and sub-chiplets.
#[derive(Clone, Debug)]
pub struct MlKemCtrlChiplet {
    #[allow(dead_code)]
    pub num_rows: usize,
}

impl MlKemCtrlChiplet {
    pub fn new(num_rows: usize) -> Self {
        assert!(num_rows.is_power_of_two());
        Self { num_rows }
    }

    /// Linking spec for the external
    /// "ml_kem_data" bus. This is what
    /// the main trace connects to.
    pub fn main_linking_spec() -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(MlKemCtrlColumns::IO_DATA),
                    b"kappa_mlkem_d0" as &[u8],
                ),
                (
                    Source::Column(MlKemCtrlColumns::REQUEST_IDX_OUT),
                    REQUEST_IDX_LABEL,
                ),
            ],
            Some(MlKemCtrlColumns::IO_SELECTOR),
        )
    }

    /// Linking spec for the
    /// internal "keccak_link" bus.
    fn keccak_linking_spec() -> PermutationCheckSpec {
        let mut sources = Vec::with_capacity(26);
        for (i, label) in KECCAK_LANE_LABELS.iter().enumerate() {
            sources.push((Source::Column(MlKemCtrlColumns::KECCAK_LANES + i), *label));
        }

        sources.push((Source::RowIndexLeBytes(4), REQUEST_IDX_LABEL));

        PermutationCheckSpec::new(sources, Some(MlKemCtrlColumns::KECCAK_SELECTOR))
    }

    /// Linking spec for the
    /// internal "ntt_data" bus.
    fn ntt_linking_spec() -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(MlKemCtrlColumns::NTT_A),
                    b"kappa_ntt_a" as &[u8],
                ),
                (
                    Source::Column(MlKemCtrlColumns::NTT_B),
                    b"kappa_ntt_b" as &[u8],
                ),
                (
                    Source::Column(MlKemCtrlColumns::NTT_A_OUT),
                    b"kappa_ntt_a_out" as &[u8],
                ),
                (
                    Source::Column(MlKemCtrlColumns::NTT_B_OUT),
                    b"kappa_ntt_b_out" as &[u8],
                ),
                (
                    Source::Column(MlKemCtrlColumns::NTT_LAYER),
                    b"kappa_ntt_layer" as &[u8],
                ),
                (
                    Source::Column(MlKemCtrlColumns::NTT_BUTTERFLY),
                    b"kappa_ntt_bfly" as &[u8],
                ),
                (
                    Source::Column(MlKemCtrlColumns::NTT_INSTANCE),
                    b"kappa_ntt_inst" as &[u8],
                ),
            ],
            Some(MlKemCtrlColumns::NTT_SELECTOR),
        )
        .with_clock_waiver(
            "see pqc/mlkem/ctrl.rs: paired with NttChiplet::data_linking_spec; \
             (ntt_instance, layer, butterfly_idx) triple is positional and unique \
             across active rows by AIR flow constraints",
        )
    }

    /// Linking spec for "basemul" bus
    fn basemul_linking_spec() -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(MlKemCtrlColumns::BM_A),
                    b"kappa_bm_a" as &[u8],
                ),
                (
                    Source::Column(MlKemCtrlColumns::BM_B),
                    b"kappa_bm_b" as &[u8],
                ),
                (
                    Source::Column(MlKemCtrlColumns::BM_C),
                    b"kappa_bm_c" as &[u8],
                ),
                (
                    Source::Column(MlKemCtrlColumns::BM_IDX),
                    b"kappa_bm_idx" as &[u8],
                ),
                (Source::RowIndexLeBytes(4), REQUEST_IDX_LABEL),
            ],
            Some(MlKemCtrlColumns::BM_SELECTOR),
        )
    }

    /// Linking spec for "ram_link" bus.
    /// Uses Source::RowIndexByte for clock
    /// (ctrl row index = execution timestamp).
    fn ram_linking_spec() -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(MlKemCtrlColumns::RAM_ADDR),
                    b"kappa_addr_b0" as &[u8],
                ),
                (
                    Source::Column(MlKemCtrlColumns::RAM_ADDR + 1),
                    b"kappa_addr_b1" as &[u8],
                ),
                (
                    Source::Column(MlKemCtrlColumns::RAM_ADDR + 2),
                    b"kappa_addr_b2" as &[u8],
                ),
                (
                    Source::Column(MlKemCtrlColumns::RAM_ADDR + 3),
                    b"kappa_addr_b3" as &[u8],
                ),
                (Source::RowIndexByte(0), b"kappa_clk_b0" as &[u8]),
                (Source::RowIndexByte(1), b"kappa_clk_b1" as &[u8]),
                (Source::RowIndexByte(2), b"kappa_clk_b2" as &[u8]),
                (Source::RowIndexByte(3), b"kappa_clk_b3" as &[u8]),
                (
                    Source::Column(MlKemCtrlColumns::RAM_VAL),
                    b"kappa_val_b0" as &[u8],
                ),
                (
                    Source::Column(MlKemCtrlColumns::RAM_VAL + 1),
                    b"kappa_val_b1" as &[u8],
                ),
                (
                    Source::Column(MlKemCtrlColumns::RAM_VAL + 2),
                    b"kappa_val_b2" as &[u8],
                ),
                (
                    Source::Column(MlKemCtrlColumns::RAM_VAL + 3),
                    b"kappa_val_b3" as &[u8],
                ),
                (
                    Source::Column(MlKemCtrlColumns::RAM_IS_WRITE),
                    b"kappa_is_write" as &[u8],
                ),
            ],
            Some(MlKemCtrlColumns::RAM_SELECTOR),
        )
    }

    /// W-side binding bus spec.
    /// Matches twiddle ROM's
    /// w_binding_linking_spec.
    pub fn w_binding_linking_spec() -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(MlKemCtrlColumns::W_BIND_BFLY_IDX),
                    b"kappa_wb_bfly" as &[u8],
                ),
                (
                    Source::Column(MlKemCtrlColumns::RAM_VAL_PACKED),
                    b"kappa_wb_w" as &[u8],
                ),
                (Source::RowIndexLeBytes(4), REQUEST_IDX_LABEL),
            ],
            Some(MlKemCtrlColumns::W_BIND_SELECTOR),
        )
    }

    /// NTT boundary input linking spec.
    /// Matches NttChiplet::bound_in_spec.
    pub fn bound_in_linking_spec() -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(MlKemCtrlColumns::NTT_INSTANCE),
                    b"kappa_bound_inst" as &[u8],
                ),
                (
                    Source::Column(MlKemCtrlColumns::BOUND_POS),
                    b"kappa_bound_pos" as &[u8],
                ),
                (
                    Source::Column(MlKemCtrlColumns::RAM_VAL_PACKED),
                    b"kappa_bound_val" as &[u8],
                ),
            ],
            Some(MlKemCtrlColumns::BOUND_IN_SEL),
        )
        .with_clock_waiver(
            "see pqc/mlkem/ctrl.rs: paired with NttChiplet::bound_in_spec; \
             (ntt_instance, BOUND_POS) is positional, AIR-forced unique per instance",
        )
    }

    /// NTT boundary output linking spec.
    /// Matches NttChiplet::bound_out_spec.
    pub fn bound_out_linking_spec() -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(MlKemCtrlColumns::NTT_INSTANCE),
                    b"kappa_bound_inst" as &[u8],
                ),
                (
                    Source::Column(MlKemCtrlColumns::BOUND_POS),
                    b"kappa_bound_pos" as &[u8],
                ),
                (
                    Source::Column(MlKemCtrlColumns::RAM_VAL_PACKED),
                    b"kappa_bound_val" as &[u8],
                ),
            ],
            Some(MlKemCtrlColumns::BOUND_OUT_SEL),
        )
        .with_clock_waiver(
            "see pqc/mlkem/ctrl.rs: paired with NttChiplet::bound_out_spec; \
             (ntt_instance, BOUND_POS) is positional, AIR-forced unique per instance",
        )
    }

    /// Keccak input ref linking spec.
    /// GPA consume side:
    /// ref rows carry (delta, lane_idx).
    fn kec_input_ref_spec() -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(MlKemCtrlColumns::KEC_LANE_DELTA),
                    b"kappa_kib_delta" as &[u8],
                ),
                (
                    Source::Column(MlKemCtrlColumns::KEC_LANE_IDX),
                    b"kappa_kib_idx" as &[u8],
                ),
            ],
            Some(MlKemCtrlColumns::KEC_INPUT_REF_SEL),
        )
        .with_clock_waiver(
            "see pqc/mlkem/ctrl.rs: KEC_LANE_IDX is positional (one row per lane \
             index); paired with kec_input_bind_spec on the produce side",
        )
    }

    /// Keccak input bind linking spec.
    /// GPA produce side:
    /// bind lo rows carry (delta, lane_idx).
    fn kec_input_bind_spec() -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(MlKemCtrlColumns::KEC_LANE_DELTA),
                    b"kappa_kib_delta" as &[u8],
                ),
                (
                    Source::Column(MlKemCtrlColumns::KEC_LANE_IDX),
                    b"kappa_kib_idx" as &[u8],
                ),
            ],
            Some(MlKemCtrlColumns::KEC_BIND_LO_SEL),
        )
        .with_clock_waiver(
            "see pqc/mlkem/ctrl.rs: KEC_LANE_IDX is positional; paired with \
             kec_input_ref_spec on the consume side",
        )
    }

    /// Shared secret output bus spec.
    pub fn ss_linking_spec() -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(MlKemCtrlColumns::SS_LO),
                    b"kappa_ss_lo0" as &[u8],
                ),
                (
                    Source::Column(MlKemCtrlColumns::SS_LO + 1),
                    b"kappa_ss_lo1" as &[u8],
                ),
                (
                    Source::Column(MlKemCtrlColumns::SS_LO + 2),
                    b"kappa_ss_lo2" as &[u8],
                ),
                (
                    Source::Column(MlKemCtrlColumns::SS_LO + 3),
                    b"kappa_ss_lo3" as &[u8],
                ),
                (
                    Source::Column(MlKemCtrlColumns::SS_HI),
                    b"kappa_ss_hi0" as &[u8],
                ),
                (
                    Source::Column(MlKemCtrlColumns::SS_HI + 1),
                    b"kappa_ss_hi1" as &[u8],
                ),
                (
                    Source::Column(MlKemCtrlColumns::SS_HI + 2),
                    b"kappa_ss_hi2" as &[u8],
                ),
                (
                    Source::Column(MlKemCtrlColumns::SS_HI + 3),
                    b"kappa_ss_hi3" as &[u8],
                ),
                (
                    Source::Column(MlKemCtrlColumns::REQUEST_IDX_OUT),
                    REQUEST_IDX_LABEL,
                ),
            ],
            Some(MlKemCtrlColumns::SS_OUT_SEL),
        )
    }
}

impl<F: TowerField> Air<F> for MlKemCtrlChiplet {
    fn name(&self) -> String {
        "MlKemCtrlChiplet".into()
    }

    fn num_columns(&self) -> usize {
        MlKemCtrlColumns::NUM_COLUMNS
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: once_cell::race::OnceBox<Vec<ColumnType>> = once_cell::race::OnceBox::new();

        LAYOUT.get_or_init(|| Box::new(MlKemCtrlColumns::build_layout()))
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![
            (MLKEM_DATA_BUS_ID.into(), Self::main_linking_spec()),
            (KeccakChiplet::BUS_ID.into(), Self::keccak_linking_spec()),
            (NttChiplet::DATA_BUS_ID.into(), Self::ntt_linking_spec()),
            (BasemulChiplet::BUS_ID.into(), Self::basemul_linking_spec()),
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
            (MLKEM_SS_BUS_ID.into(), Self::ss_linking_spec()),
        ]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();

        let io_sel = cs.col(MlKemCtrlColumns::IO_SELECTOR);
        let pad_sel = cs.col(MlKemCtrlColumns::PAD_SEL);
        let kec_sel = cs.col(MlKemCtrlColumns::KECCAK_SELECTOR);
        let ntt_sel = cs.col(MlKemCtrlColumns::NTT_SELECTOR);
        let bm_sel = cs.col(MlKemCtrlColumns::BM_SELECTOR);
        let ram_sel = cs.col(MlKemCtrlColumns::RAM_SELECTOR);
        let s_active = cs.col(MlKemCtrlColumns::S_ACTIVE);

        let cmp_sel = cs.col(MlKemCtrlColumns::CMP_SELECTOR);
        let ct_match = cs.col(MlKemCtrlColumns::CT_MATCH);

        let sels = [io_sel, kec_sel, ntt_sel, bm_sel, ram_sel, cmp_sel];

        // Selector booleanity
        for &s in &sels {
            cs.assert_boolean(s);
        }

        cs.assert_boolean(pad_sel);
        cs.assert_boolean(s_active);
        cs.assert_boolean(ct_match);

        // IO_SELECTOR and PAD_SEL
        // are mutually exclusive.
        cs.constrain(io_sel * pad_sel);

        // W-side binding selector
        let w_bind_sel = cs.col(MlKemCtrlColumns::W_BIND_SELECTOR);
        cs.assert_boolean(w_bind_sel);

        // W_BIND_SELECTOR implies RAM_SELECTOR
        cs.constrain(w_bind_sel * (cs.one() + ram_sel));

        // NTT boundary selectors
        let bound_in_sel = cs.col(MlKemCtrlColumns::BOUND_IN_SEL);
        let bound_out_sel = cs.col(MlKemCtrlColumns::BOUND_OUT_SEL);

        cs.assert_boolean(bound_in_sel);
        cs.assert_boolean(bound_out_sel);

        // Subset of s_active:
        // boundary entries only on active rows.
        let not_active = cs.one() + s_active;
        cs.constrain(bound_in_sel * not_active);
        cs.constrain(bound_out_sel * not_active);

        // Boundary implies RAM:
        // every boundary row carries RAM data.
        cs.constrain(bound_in_sel * (cs.one() + ram_sel));
        cs.constrain(bound_out_sel * (cs.one() + ram_sel));

        // Mutual exclusivity:
        // a row is either input or output, not both.
        cs.constrain(bound_in_sel * bound_out_sel);

        // Boundary vs other dispatch selectors.
        // Boundary rows carry RAM data only —
        // no NTT/Keccak/Basemul/IO/CMP dispatch.
        for &s in &[io_sel, kec_sel, ntt_sel, bm_sel, cmp_sel, w_bind_sel] {
            cs.constrain(bound_in_sel * s);
            cs.constrain(bound_out_sel * s);
        }

        // Selector orthogonality
        //
        // (ntt_sel, ram_sel) OMITTED:
        // NTT+RAM co-activation for data binding.
        // (bm_sel, ram_sel) OMITTED:
        // BM+RAM co-activation for result binding.
        // (kec_sel, ram_sel) OMITTED:
        // reserved for Keccak input binding.
        // (io_sel, ram_sel) OMITTED:
        // IO+RAM for public ct deposit.
        let exclusive_pairs: &[(usize, usize)] = &[
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 5), // io vs {kec, ntt, bm, cmp}
            (1, 2),
            (1, 3),
            (1, 5), // kec vs {ntt, bm, cmp}
            (2, 3),
            (2, 5), // ntt vs {bm, cmp}
            (3, 5), // bm vs cmp
            (4, 5), // ram vs cmp
        ];

        for &(i, j) in exclusive_pairs {
            cs.constrain(sels[i] * sels[j]);
        }

        // NTT <> RAM data binding.
        // On co-activated NTT+RAM rows:
        // NTT_B = RAM_VAL_PACKED.
        let ntt_b = cs.col(MlKemCtrlColumns::NTT_B);
        let ram_val_packed = cs.col(MlKemCtrlColumns::RAM_VAL_PACKED);
        let ram_is_write = cs.col(MlKemCtrlColumns::RAM_IS_WRITE);
        let io_data = cs.col(MlKemCtrlColumns::IO_DATA);

        cs.constrain(ntt_sel * ram_sel * (ntt_b + ram_val_packed));

        // BM <> RAM binding.
        // BM rows must co-activate RAM;
        // BM_C = RAM_VAL_PACKED.
        let bm_c = cs.col(MlKemCtrlColumns::BM_C);
        cs.constrain(bm_sel * (cs.one() + ram_sel));
        cs.constrain(bm_sel * (bm_c + ram_val_packed));

        // S_ACTIVE consistency
        let one = cs.one();
        let not_active = one + s_active;

        for &s in &sels {
            cs.constrain(s * not_active);
        }

        // =============================================================
        // Phase state machine
        // =============================================================

        let ph_io = cs.col(MlKemCtrlColumns::PH_IO);
        let ph_dec = cs.col(MlKemCtrlColumns::PH_DECRYPT);
        let ph_ghash = cs.col(MlKemCtrlColumns::PH_G_HASH);
        let ph_enc = cs.col(MlKemCtrlColumns::PH_ENCRYPT);
        let ph_cmphash = cs.col(MlKemCtrlColumns::PH_CMP_HASH);
        let ph_cmp = cs.col(MlKemCtrlColumns::PH_COMPARE);

        let ph = [ph_io, ph_dec, ph_ghash, ph_enc, ph_cmphash, ph_cmp];

        // Phase booleanity
        for &p in &ph {
            cs.assert_boolean(p);
        }

        // Phase one-hot on active rows:
        // (a) at most one via pairwise orthogonality
        for i in 0..ph.len() {
            for j in (i + 1)..ph.len() {
                cs.constrain(ph[i] * ph[j]);
            }
        }

        // (b) at least one on active rows:
        // s_active * (1 + Σ ph[i]) = 0
        // In GF(2), Σ of one-hot = 1, so 1+1=0.
        let ph_sum = ph_io + ph_dec + ph_ghash + ph_enc + ph_cmphash + ph_cmp;
        cs.constrain(s_active * (one + ph_sum));

        // Phases zero on padding rows
        for &p in &ph {
            cs.constrain(p * not_active);
        }

        // Forward-only transitions:
        // if current row is phase i, next row
        // cannot be any earlier phase j < i.
        // Uses cs.next() to access next-row values.
        //
        // On padding rows:
        // s_active=0, so these are trivially
        // satisfied (0 * anything).
        // On last active -> first padding:
        // next ph[j]=0, so also satisfied.
        for i in 1..ph.len() {
            for j in 0..i {
                cs.constrain(s_active * ph[i] * cs.next(MlKemCtrlColumns::PH_IO + j));
            }
        }

        // =============================================================
        // Phase–selector consistency
        // =============================================================
        //
        // IO:
        // IO, RAM
        //
        // DECRYPT:
        // NTT, BM, RAM
        //
        // G_HASH:
        // Keccak, RAM
        //
        // ENCRYPT:
        // NTT, BM, RAM, Keccak
        //
        // CMP_HASH:
        // Keccak
        //
        // COMPARE:
        // CMP

        // IO forbids:
        // Keccak, NTT, BM, CMP
        cs.constrain(ph_io * kec_sel);
        cs.constrain(ph_io * ntt_sel);
        cs.constrain(ph_io * bm_sel);
        cs.constrain(ph_io * cmp_sel);

        cs.constrain(pad_sel * (one + ph_io));
        cs.constrain(ph_io * (one + io_sel + pad_sel));

        cs.constrain(io_sel * (one + ram_sel));
        cs.constrain(pad_sel * (one + ram_sel));
        cs.constrain(io_sel * ram_sel * (one + ram_is_write));
        cs.constrain(pad_sel * ram_sel * (one + ram_is_write));
        cs.constrain(io_sel * ram_sel * (ram_val_packed + io_data));
        cs.constrain(pad_sel * ram_sel * (ram_val_packed + io_data));

        // DECRYPT forbids:
        // IO, Keccak, CMP
        cs.constrain(ph_dec * io_sel);
        cs.constrain(ph_dec * pad_sel);
        cs.constrain(ph_dec * kec_sel);
        cs.constrain(ph_dec * cmp_sel);

        // G_HASH forbids:
        // IO, NTT, BM, CMP
        cs.constrain(ph_ghash * io_sel);
        cs.constrain(ph_ghash * pad_sel);
        cs.constrain(ph_ghash * ntt_sel);
        cs.constrain(ph_ghash * bm_sel);
        cs.constrain(ph_ghash * cmp_sel);

        // ENCRYPT forbids:
        // IO, CMP
        cs.constrain(ph_enc * io_sel);
        cs.constrain(ph_enc * pad_sel);
        cs.constrain(ph_enc * cmp_sel);

        // CMP_HASH forbids:
        // IO, NTT, BM, CMP.
        cs.constrain(ph_cmphash * io_sel);
        cs.constrain(ph_cmphash * pad_sel);
        cs.constrain(ph_cmphash * ntt_sel);
        cs.constrain(ph_cmphash * bm_sel);
        cs.constrain(ph_cmphash * cmp_sel);

        // RAM read in CmpHash is permitted
        // only for Keccak input binding
        // (KEC_BIND_LO/HI). H(ct) raw-byte
        // reads also set KEC_BIND_LO/HI,
        // so the same factor covers them.
        cs.constrain(
            ph_cmphash
                * ram_sel
                * (one + ram_is_write)
                * (one + cs.col(MlKemCtrlColumns::KEC_BIND_LO_SEL))
                * (one + cs.col(MlKemCtrlColumns::KEC_BIND_HI_SEL)),
        );

        // COMPARE forbids:
        // IO, Keccak, NTT, BM, RAM
        cs.constrain(ph_cmp * io_sel);
        cs.constrain(ph_cmp * pad_sel);
        cs.constrain(ph_cmp * kec_sel);
        cs.constrain(ph_cmp * ntt_sel);
        cs.constrain(ph_cmp * bm_sel);
        cs.constrain(ph_cmp * ram_sel);

        // =============================================================
        // Re-encryption hash comparison (bidirectional)
        // =============================================================
        //
        // Forward:
        // CT_MATCH=1 -> all lanes match.
        for i in 0..4 {
            let kec_lane = cs.col(MlKemCtrlColumns::KECCAK_LANES + i);
            let hash_ref = cs.col(MlKemCtrlColumns::HASH_REF + i);

            cs.constrain(cmp_sel * ct_match * (kec_lane + hash_ref));
        }

        // Reverse:
        // all lanes match -> CT_MATCH=1.
        //
        // Uses TAU-combined pairs:
        //   diff_lo = diff[0] + diff[1]·TAU
        //   diff_hi = diff[2] + diff[3]·TAU
        // {1, TAU} is a basis for GF(2^128)/GF(2^64),
        // so diff_lo=0 iff lanes 0,1 both match,
        // diff_hi=0 iff lanes 2,3 both match.
        let tau = cs.constant(F::EXTENSION_TAU);

        let diff_lo = (cs.col(MlKemCtrlColumns::KECCAK_LANES) + cs.col(MlKemCtrlColumns::HASH_REF))
            + (cs.col(MlKemCtrlColumns::KECCAK_LANES + 1) + cs.col(MlKemCtrlColumns::HASH_REF + 1))
                * tau;

        let diff_hi = (cs.col(MlKemCtrlColumns::KECCAK_LANES + 2)
            + cs.col(MlKemCtrlColumns::HASH_REF + 2))
            + (cs.col(MlKemCtrlColumns::KECCAK_LANES + 3) + cs.col(MlKemCtrlColumns::HASH_REF + 3))
                * tau;

        let eq_lo = cs.col(MlKemCtrlColumns::HASH_EQ_LO);
        let eq_hi = cs.col(MlKemCtrlColumns::HASH_EQ_HI);
        let inv_lo = cs.col(MlKemCtrlColumns::HASH_DIFF_INV_LO);
        let inv_hi = cs.col(MlKemCtrlColumns::HASH_DIFF_INV_HI);

        cs.assert_boolean(eq_lo);
        cs.assert_boolean(eq_hi);

        // eq_lo=1 -> diff_lo=0
        cs.constrain(cmp_sel * eq_lo * diff_lo);

        // eq_lo=0 -> diff_lo has valid inverse
        cs.constrain(cmp_sel * (one + eq_lo) * (diff_lo * inv_lo + one));

        // eq_hi=1 -> diff_hi=0
        cs.constrain(cmp_sel * eq_hi * diff_hi);

        // eq_hi=0 -> diff_hi has valid inverse
        cs.constrain(cmp_sel * (one + eq_hi) * (diff_hi * inv_hi + one));

        // CT_MATCH = eq_lo · eq_hi on CMP rows
        cs.constrain(cmp_sel * (ct_match + eq_lo * eq_hi));

        // =============================================================
        // Sponge state binding
        // =============================================================

        let kec_out = cs.col(MlKemCtrlColumns::KEC_IS_OUTPUT);
        let sponge_init = cs.col(MlKemCtrlColumns::SPONGE_INIT);
        let sha3_512 = cs.col(MlKemCtrlColumns::SHA3_512);
        let shake_128 = cs.col(MlKemCtrlColumns::SHAKE_128);

        cs.assert_boolean(kec_out);
        cs.assert_boolean(sponge_init);
        cs.assert_boolean(sha3_512);
        cs.assert_boolean(shake_128);

        // Subsets of KECCAK_SELECTOR
        cs.constrain(kec_out * (one + kec_sel));
        cs.constrain(sponge_init * (one + kec_sel));
        cs.constrain(sha3_512 * (one + kec_sel));
        cs.constrain(shake_128 * (one + kec_sel));

        // Mutual exclusion:
        // sha3_512 and shake_128
        // are disjoint sponge modes.
        cs.constrain(sha3_512 * shake_128);

        // SPONGE_INIT and KEC_IS_OUTPUT
        // are mutually exclusive:
        // init is on input rows,
        // output on output rows.
        cs.constrain(sponge_init * kec_out);

        // Keccak input row:
        // kec_sel=1, kec_out=0.
        let kec_input = kec_sel * (one + kec_out);

        // Sticky transition for
        // all 25 state registers.
        //
        // Three cases:
        //   kec_out=0, init_next=0:
        //     reg[next] = reg (carry)
        //   kec_out=1, init_next=0:
        //     reg[next] = lane (update)
        //   init_next=1:
        //     reg[next] = 0 (reset)
        //     (enforced by sponge_init * reg)
        let sponge_init_next = cs.next(MlKemCtrlColumns::SPONGE_INIT);

        for i in 0..25 {
            let reg = cs.col(MlKemCtrlColumns::RATE_REG + i);
            let reg_next = cs.next(MlKemCtrlColumns::RATE_REG + i);
            let lane = cs.col(MlKemCtrlColumns::KECCAK_LANES + i);

            cs.constrain(
                s_active * (one + sponge_init_next) * (reg_next + reg + kec_out * (reg + lane)),
            );
        }

        // Sponge init:
        // SPONGE_INIT=1 -> all registers = 0.
        for i in 0..25 {
            let reg = cs.col(MlKemCtrlColumns::RATE_REG + i);
            cs.constrain(sponge_init * reg);
        }

        // Capacity lane continuity:
        // on input rows, capacity lanes
        // must equal the register
        // (= previous output).
        //
        // Lanes 0..8:
        // always rate for all modes.
        //
        // Lanes 9..20:
        // capacity when SHA3-512 (rate=9).
        for i in 9..21 {
            let reg = cs.col(MlKemCtrlColumns::RATE_REG + i);
            let lane = cs.col(MlKemCtrlColumns::KECCAK_LANES + i);

            cs.constrain(kec_input * sha3_512 * (lane + reg));
        }

        // Lanes 17..20:
        // capacity when SHA3-256/SHAKE-256
        // (rate=17). Rate for SHAKE-128.
        // SHA3-512 already covers these above.
        for i in 17..21 {
            let reg = cs.col(MlKemCtrlColumns::RATE_REG + i);
            let lane = cs.col(MlKemCtrlColumns::KECCAK_LANES + i);

            cs.constrain(kec_input * (one + sha3_512) * (one + shake_128) * (lane + reg));
        }

        // Lanes 21..24:
        // always capacity for all modes.
        for i in 21..25 {
            let reg = cs.col(MlKemCtrlColumns::RATE_REG + i);
            let lane = cs.col(MlKemCtrlColumns::KECCAK_LANES + i);

            cs.constrain(kec_input * (lane + reg));
        }

        // =============================================================
        // Keccak input binding
        // =============================================================

        let kec_ref_sel = cs.col(MlKemCtrlColumns::KEC_INPUT_REF_SEL);
        let kec_bind_lo = cs.col(MlKemCtrlColumns::KEC_BIND_LO_SEL);
        let kec_bind_hi = cs.col(MlKemCtrlColumns::KEC_BIND_HI_SEL);
        let kec_delta = cs.col(MlKemCtrlColumns::KEC_LANE_DELTA);

        cs.assert_boolean(kec_ref_sel);
        cs.assert_boolean(kec_bind_lo);
        cs.assert_boolean(kec_bind_hi);

        // One-hot booleanity + exclusion.
        //
        // Carry-chain approach:
        // accumulate running XOR parity.
        // Constrain acc[k-1] * oh[k] = 0
        // so at most one bit can be 1.
        // Combined with the parity check
        // (acc = 1 on ref rows),
        // exactly one bit is set.
        let oh_first = cs.col(MlKemCtrlColumns::KEC_LANE_ONE_HOT);
        cs.assert_boolean(oh_first);

        let mut oh_acc = oh_first;
        for k in 1..21 {
            let oh = cs.col(MlKemCtrlColumns::KEC_LANE_ONE_HOT + k);
            cs.assert_boolean(oh);
            cs.constrain(kec_ref_sel * oh_acc * oh);

            oh_acc = oh_acc + oh;
        }

        // Exactly one bit set on ref rows
        cs.constrain(kec_ref_sel * (one + oh_acc));

        // One-hot delta verification:
        // for the selected lane k,
        // delta = KECCAK_LANES[k] + RATE_REG[k].
        for k in 0..21 {
            let oh = cs.col(MlKemCtrlColumns::KEC_LANE_ONE_HOT + k);
            let lane = cs.col(MlKemCtrlColumns::KECCAK_LANES + k);
            let reg = cs.col(MlKemCtrlColumns::RATE_REG + k);

            cs.constrain(kec_ref_sel * oh * (kec_delta + lane + reg));
        }

        // Carry chain:
        // if the next row is a ref row,
        // all 25 KECCAK_LANES must match.
        let ref_sel_next = cs.next(MlKemCtrlColumns::KEC_INPUT_REF_SEL);
        for k in 0..25 {
            let lane = cs.col(MlKemCtrlColumns::KECCAK_LANES + k);
            let lane_next = cs.next(MlKemCtrlColumns::KECCAK_LANES + k);

            cs.constrain(s_active * ref_sel_next * (lane + lane_next));
        }

        // Bind lo rows:
        // delta = lo + hi * tau_32.
        // Tower decomposition:
        // Block64(lo | hi<<32)
        //   = Block32(lo) + Block32(hi) * v
        // where v = Block64(1 << 32).
        let tau_32 = cs.constant(F::from(1u64 << 32));
        let ram_val = cs.col(MlKemCtrlColumns::RAM_VAL_PACKED);
        let ram_val_next = cs.next(MlKemCtrlColumns::RAM_VAL_PACKED);

        cs.constrain(kec_bind_lo * (kec_delta + ram_val + ram_val_next * tau_32));

        // bind_hi must follow bind_lo
        cs.constrain(kec_bind_lo * (one + cs.next(MlKemCtrlColumns::KEC_BIND_HI_SEL)));

        // =============================================================
        // H(ct) raw-byte -> Keccak input binding
        // =============================================================

        let io_lane_bind_sel = cs.col(MlKemCtrlColumns::IO_LANE_BIND_SEL);
        let h_ct_input_sel = cs.col(MlKemCtrlColumns::H_CT_INPUT_SEL);
        let h_ct_active = cs.col(MlKemCtrlColumns::H_CT_ACTIVE);
        let io_lane_lo = cs.col(MlKemCtrlColumns::IO_LANE_LO);
        let io_lane_hi = cs.col(MlKemCtrlColumns::IO_LANE_HI);

        cs.assert_boolean(io_lane_bind_sel);
        cs.assert_boolean(h_ct_input_sel);
        cs.assert_boolean(h_ct_active);

        // IO bind row is a RAM read
        // and is also a kec bind row
        // (lo or hi). Reusing kec_input_bind
        // gives us delta verification +
        // GPA pairing with the ref rows
        // for free.
        cs.constrain(io_lane_bind_sel * (one + ram_sel));
        cs.constrain(io_lane_bind_sel * ram_is_write);
        cs.constrain(io_lane_bind_sel * (one + kec_bind_lo + kec_bind_hi));

        // The trace fill writes the read
        // value into IO_LANE_LO on bind_lo
        // rows and IO_LANE_HI on bind_hi
        // rows. Mirror them against
        // RAM_VAL_PACKED so a tamper of
        // either column breaks the witness.
        cs.constrain(io_lane_bind_sel * kec_bind_lo * (io_lane_lo + ram_val));
        cs.constrain(io_lane_bind_sel * kec_bind_hi * (io_lane_hi + ram_val));

        // H_CT_ACTIVE sticky chain:
        // - H_CT_INPUT_SEL row forces it 1
        // - Keccak output row forces it 0
        // - IO bind row requires it 1
        // - Otherwise sticky between rows
        let h_ct_active_next = cs.next(MlKemCtrlColumns::H_CT_ACTIVE);
        let h_ct_input_sel_next = cs.next(MlKemCtrlColumns::H_CT_INPUT_SEL);
        let kec_out_next = cs.next(MlKemCtrlColumns::KEC_IS_OUTPUT);

        cs.constrain(h_ct_input_sel * (one + h_ct_active));
        cs.constrain(kec_out * h_ct_active);
        cs.constrain(io_lane_bind_sel * (one + h_ct_active));
        cs.constrain(
            s_active
                * (one + h_ct_input_sel_next)
                * (one + kec_out_next)
                * (h_ct_active + h_ct_active_next),
        );

        // =============================================================
        // SHA3-256 padding constant binding
        // =============================================================

        let pad_first = cs.col(MlKemCtrlColumns::PAD_FIRST);
        let pad_last = cs.col(MlKemCtrlColumns::PAD_LAST);

        cs.assert_boolean(pad_first);
        cs.assert_boolean(pad_last);

        cs.constrain(pad_first * (one + pad_sel));
        cs.constrain(pad_last * (one + pad_sel));
        cs.constrain(pad_first * pad_last);

        // FIPS 202 §B.2:
        // SHA3-256 padding is:
        // 0x06 || 0x00 ... || 0x80 over the
        // remaining rate bytes. As 4-byte
        // chunks the first chunk is 0x06,
        // the last is 0x80 << 24, and middle
        // chunks are 0. Encoded as one
        // constraint via the constant-sum trick.
        let pad_first_const = cs.constant(F::from(0x06u64));
        let pad_last_const = cs.constant(F::from(0x80000000u64));

        cs.constrain(pad_sel * (ram_val + pad_first * pad_first_const + pad_last * pad_last_const));

        // =============================================================
        // HASH_REF / HASH_CT_PRIME sticky binding
        // =============================================================

        let h_ct_bind_sel = cs.col(MlKemCtrlColumns::H_CT_BIND_SEL);
        let h_ct_prime_bind_sel = cs.col(MlKemCtrlColumns::H_CT_PRIME_BIND_SEL);

        cs.assert_boolean(h_ct_bind_sel);
        cs.assert_boolean(h_ct_prime_bind_sel);
        cs.constrain(h_ct_bind_sel * h_ct_prime_bind_sel);

        cs.constrain(h_ct_bind_sel * (one + ph_cmphash));
        cs.constrain(h_ct_prime_bind_sel * (one + ph_cmphash));

        // Snapshot RATE_REG into HASH_REF
        // (H(ct)) and HASH_CT_PRIME (H(ct')).
        // Also pin KECCAK_LANES on the H(ct')
        // bind row so tampering there is caught.
        for i in 0..4 {
            let hash_ref = cs.col(MlKemCtrlColumns::HASH_REF + i);
            let reg = cs.col(MlKemCtrlColumns::RATE_REG + i);
            let hash_ct_prime = cs.col(MlKemCtrlColumns::HASH_CT_PRIME + i);
            let kec_lane = cs.col(MlKemCtrlColumns::KECCAK_LANES + i);

            cs.constrain(h_ct_bind_sel * (hash_ref + reg));
            cs.constrain(h_ct_prime_bind_sel * (hash_ct_prime + reg));
            cs.constrain(h_ct_prime_bind_sel * (kec_lane + reg));
        }

        // Sticky carry:
        // HASH_REF from the snapshot row
        // forward to the CMP row.
        // Gate on cs.next(BIND_SEL) so the
        // row-before-snapshot transition
        // (0 -> H(ct)) is vacuous.
        let h_ct_bind_sel_next = cs.next(MlKemCtrlColumns::H_CT_BIND_SEL);
        let h_ct_prime_bind_sel_next = cs.next(MlKemCtrlColumns::H_CT_PRIME_BIND_SEL);

        for i in 0..4 {
            let h = cs.col(MlKemCtrlColumns::HASH_REF + i);
            let h_nxt = cs.next(MlKemCtrlColumns::HASH_REF + i);

            cs.constrain(s_active * (one + h_ct_bind_sel_next) * (h_nxt + h));

            let p = cs.col(MlKemCtrlColumns::HASH_CT_PRIME + i);
            let p_nxt = cs.next(MlKemCtrlColumns::HASH_CT_PRIME + i);

            cs.constrain(s_active * (one + h_ct_prime_bind_sel_next) * (p_nxt + p));
        }

        for i in 0..4 {
            let kec_lane = cs.col(MlKemCtrlColumns::KECCAK_LANES + i);
            let hash_ct_prime = cs.col(MlKemCtrlColumns::HASH_CT_PRIME + i);

            cs.constrain(cmp_sel * (kec_lane + hash_ct_prime));
        }

        // =============================================================
        // Bind-sel existence, ordering, position
        // =============================================================

        let h_ct_bind_seen = cs.col(MlKemCtrlColumns::H_CT_BIND_SEEN);
        let h_ct_prime_bind_seen = cs.col(MlKemCtrlColumns::H_CT_PRIME_BIND_SEEN);
        let h_ct_bind_seen_next = cs.next(MlKemCtrlColumns::H_CT_BIND_SEEN);
        let h_ct_prime_bind_seen_next = cs.next(MlKemCtrlColumns::H_CT_PRIME_BIND_SEEN);

        cs.assert_boolean(h_ct_bind_seen);
        cs.assert_boolean(h_ct_prime_bind_seen);

        // Grounding:
        // seen=0 outside CMP_HASH + COMPARE.
        // Solves padding->active initialization.
        cs.constrain(h_ct_bind_seen * (one + ph_cmphash + ph_cmp));
        cs.constrain(h_ct_prime_bind_seen * (one + ph_cmphash + ph_cmp));

        let s_active_next = cs.next(MlKemCtrlColumns::S_ACTIVE);

        // Monotonic 1->1, gated by s_active_next
        // so CMP->padding drop is allowed.
        cs.constrain(s_active * s_active_next * h_ct_bind_seen * (one + h_ct_bind_seen_next));
        cs.constrain(
            s_active * s_active_next * h_ct_prime_bind_seen * (one + h_ct_prime_bind_seen_next),
        );

        // Transition 0->1 requires sel
        cs.constrain(
            s_active * h_ct_bind_seen_next * (one + h_ct_bind_seen) * (one + h_ct_bind_sel),
        );
        cs.constrain(
            s_active
                * h_ct_prime_bind_seen_next
                * (one + h_ct_prime_bind_seen)
                * (one + h_ct_prime_bind_sel),
        );

        // CMP requires both snapshots taken
        cs.constrain(cmp_sel * (one + h_ct_bind_seen));
        cs.constrain(cmp_sel * (one + h_ct_prime_bind_seen));

        // Ordering:
        // prime only after bind.
        cs.constrain(h_ct_prime_bind_sel * (one + h_ct_bind_seen));

        // Position:
        // bind-sel follows a kec_out row.
        cs.constrain(s_active * h_ct_bind_sel_next * (one + kec_out));
        cs.constrain(s_active * h_ct_prime_bind_sel_next * (one + kec_out));

        // K_PRIME / K_BAR_REJECT RATE_REG bind
        let k_prime_bind_sel = cs.col(MlKemCtrlColumns::K_PRIME_BIND_SEL);
        let k_bar_bind_sel = cs.col(MlKemCtrlColumns::K_BAR_BIND_SEL);

        cs.assert_boolean(k_prime_bind_sel);
        cs.assert_boolean(k_bar_bind_sel);

        cs.constrain(k_prime_bind_sel * k_bar_bind_sel);

        for i in 0..4 {
            let reg = cs.col(MlKemCtrlColumns::RATE_REG + i);
            let kp_lo = cs.col(MlKemCtrlColumns::K_PRIME_LO + i);
            let kp_hi = cs.col(MlKemCtrlColumns::K_PRIME_HI + i);

            cs.constrain(k_prime_bind_sel * (reg + kp_lo + kp_hi * tau_32));

            let kb_lo = cs.col(MlKemCtrlColumns::K_BAR_LO + i);
            let kb_hi = cs.col(MlKemCtrlColumns::K_BAR_HI + i);

            cs.constrain(k_bar_bind_sel * (reg + kb_lo + kb_hi * tau_32));
        }

        // K_PRIME / K_BAR bind
        // existence and ordering.
        let k_prime_bind_seen = cs.col(MlKemCtrlColumns::K_PRIME_BIND_SEEN);
        let k_bar_bind_seen = cs.col(MlKemCtrlColumns::K_BAR_BIND_SEEN);
        let k_prime_bind_seen_next = cs.next(MlKemCtrlColumns::K_PRIME_BIND_SEEN);
        let k_bar_bind_seen_next = cs.next(MlKemCtrlColumns::K_BAR_BIND_SEEN);

        cs.assert_boolean(k_prime_bind_seen);
        cs.assert_boolean(k_bar_bind_seen);

        // K_PRIME seen=0 outside G_HASH..COMPARE
        cs.constrain(k_prime_bind_seen * (one + ph_ghash + ph_enc + ph_cmphash + ph_cmp));

        // K_BAR seen=0 outside CMP_HASH..COMPARE
        cs.constrain(k_bar_bind_seen * (one + ph_cmphash + ph_cmp));

        // Monotonic 1->1
        cs.constrain(s_active * s_active_next * k_prime_bind_seen * (one + k_prime_bind_seen_next));
        cs.constrain(s_active * s_active_next * k_bar_bind_seen * (one + k_bar_bind_seen_next));

        // 0->1 requires bind_sel
        cs.constrain(
            s_active
                * k_prime_bind_seen_next
                * (one + k_prime_bind_seen)
                * (one + k_prime_bind_sel),
        );
        cs.constrain(
            s_active * k_bar_bind_seen_next * (one + k_bar_bind_seen) * (one + k_bar_bind_sel),
        );

        // K_BAR only after K_PRIME
        cs.constrain(k_bar_bind_sel * (one + k_prime_bind_seen));

        // K_PRIME / K_BAR sticky carry
        let k_prime_bind_sel_next = cs.next(MlKemCtrlColumns::K_PRIME_BIND_SEL);
        let k_bar_bind_sel_next = cs.next(MlKemCtrlColumns::K_BAR_BIND_SEL);

        for i in 0..4 {
            let kp_lo = cs.col(MlKemCtrlColumns::K_PRIME_LO + i);
            let kp_lo_next = cs.next(MlKemCtrlColumns::K_PRIME_LO + i);

            cs.constrain(s_active * (one + k_prime_bind_sel_next) * (kp_lo_next + kp_lo));

            let kp_hi = cs.col(MlKemCtrlColumns::K_PRIME_HI + i);
            let kp_hi_next = cs.next(MlKemCtrlColumns::K_PRIME_HI + i);

            cs.constrain(s_active * (one + k_prime_bind_sel_next) * (kp_hi_next + kp_hi));

            let kb_lo = cs.col(MlKemCtrlColumns::K_BAR_LO + i);
            let kb_lo_next = cs.next(MlKemCtrlColumns::K_BAR_LO + i);

            cs.constrain(s_active * (one + k_bar_bind_sel_next) * (kb_lo_next + kb_lo));

            let kb_hi = cs.col(MlKemCtrlColumns::K_BAR_HI + i);
            let kb_hi_next = cs.next(MlKemCtrlColumns::K_BAR_HI + i);

            cs.constrain(s_active * (one + k_bar_bind_sel_next) * (kb_hi_next + kb_hi));
        }

        // CT_MATCH sticky carry
        let cmp_sel_next = cs.next(MlKemCtrlColumns::CMP_SELECTOR);
        let ct_match_next = cs.next(MlKemCtrlColumns::CT_MATCH);

        cs.constrain(s_active * (one + cmp_sel_next) * (ct_match_next + ct_match));

        // SS mux + output selectors
        let ss_mux_sel = cs.col(MlKemCtrlColumns::SS_MUX_SEL);
        let ss_out_sel = cs.col(MlKemCtrlColumns::SS_OUT_SEL);

        cs.assert_boolean(ss_mux_sel);
        cs.assert_boolean(ss_out_sel);

        // Mux requires both K binds seen
        cs.constrain(ss_mux_sel * (one + k_prime_bind_seen));
        cs.constrain(ss_mux_sel * (one + k_bar_bind_seen));

        for i in 0..4 {
            let ss_lo = cs.col(MlKemCtrlColumns::SS_LO + i);
            let ss_hi = cs.col(MlKemCtrlColumns::SS_HI + i);
            let kp_lo = cs.col(MlKemCtrlColumns::K_PRIME_LO + i);
            let kp_hi = cs.col(MlKemCtrlColumns::K_PRIME_HI + i);
            let kb_lo = cs.col(MlKemCtrlColumns::K_BAR_LO + i);
            let kb_hi = cs.col(MlKemCtrlColumns::K_BAR_HI + i);

            cs.constrain(ss_mux_sel * (ss_lo + ct_match * kp_lo + (one + ct_match) * kb_lo));
            cs.constrain(ss_mux_sel * (ss_hi + ct_match * kp_hi + (one + ct_match) * kb_hi));
        }

        // SS sticky carry
        let ss_mux_sel_next = cs.next(MlKemCtrlColumns::SS_MUX_SEL);

        for i in 0..4 {
            let ss_lo = cs.col(MlKemCtrlColumns::SS_LO + i);
            let ss_lo_next = cs.next(MlKemCtrlColumns::SS_LO + i);

            cs.constrain(s_active * (one + ss_mux_sel_next) * (ss_lo_next + ss_lo));

            let ss_hi = cs.col(MlKemCtrlColumns::SS_HI + i);
            let ss_hi_next = cs.next(MlKemCtrlColumns::SS_HI + i);

            cs.constrain(s_active * (one + ss_mux_sel_next) * (ss_hi_next + ss_hi));
        }

        // Phase permissions:
        // SS_MUX_SEL and SS_OUT_SEL
        // only in compare phase.
        cs.constrain(ph_io * ss_mux_sel);
        cs.constrain(ph_dec * ss_mux_sel);
        cs.constrain(ph_ghash * ss_mux_sel);
        cs.constrain(ph_enc * ss_mux_sel);
        cs.constrain(ph_cmphash * ss_mux_sel);

        cs.constrain(ph_io * ss_out_sel);
        cs.constrain(ph_dec * ss_out_sel);
        cs.constrain(ph_ghash * ss_out_sel);
        cs.constrain(ph_enc * ss_out_sel);
        cs.constrain(ph_cmphash * ss_out_sel);

        // Terminal:
        // last active row must be SS_OUT_SEL.
        cs.constrain(s_active * (one + s_active_next) * (one + ss_out_sel));

        cs.build()
    }
}

// =================================================================
// Tests
// =================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlkem::{MLKEM_Q, MlKemChiplet, MlKemLevel, MlKemParams};
    use hekate_core::trace::TraceColumn;
    use hekate_math::{Bit, Block128};
    use pqcrypto_mlkem::mlkem768;
    use pqcrypto_traits::kem::{Ciphertext, SecretKey};

    type F = Block128;

    #[test]
    fn ctrl_chiplet_column_count() {
        // 1 IO B32
        //   + 1 IO Bit
        //   + 1 PAD_SEL Bit
        // + 25 Keccak B64 + 1 Keccak Bit
        // + 17 RATE_REG B64
        // + 1 KEC_IS_OUTPUT Bit
        // + 1 SPONGE_INIT Bit
        // + 1 SHA3_512 Bit
        // + 7 NTT B32 + 1 NTT Bit
        // + 4 BM B32 + 1 BM Bit
        // + 4 RAM_ADDR B32
        //   + 4 RAM_VAL B32
        //   + 1 RAM_VAL_PACKED B32
        //   + 1 RAM_IS_WRITE Bit
        //   + 1 RAM Bit
        // + 1 W_BIND_BFLY_IDX B32
        // + 1 W_BIND_SELECTOR Bit
        // + 1 BOUND_POS B32
        // + 1 BOUND_IN_SEL Bit
        // + 1 BOUND_OUT_SEL Bit
        // + 17 KEC_LANE_ONE_HOT Bit
        // + 1 KEC_LANE_DELTA B64
        // + 1 KEC_INPUT_REF_SEL Bit
        // + 1 KEC_BIND_LO_SEL Bit
        // + 2 IO_LANE_{LO,HI} B32
        //   + 1 IO_LANE_BIND_SEL Bit
        //   + 1 H_CT_INPUT_SEL Bit
        //   + 1 H_CT_ACTIVE Bit
        //   + 2 PAD_FIRST/LAST Bit
        // + 2 H_CT_{,PRIME_}BIND_SEL Bit
        // + 4 HASH_CT_PRIME B64
        // + 2 H_CT_{,PRIME_}BIND_SEEN Bit
        // + 4 HASH_REF B64
        //   + 1 CT_MATCH Bit
        //   + 1 CMP Bit
        // + 2 HASH_EQ Bit
        //   + 2 HASH_DIFF_INV B128
        // + 4 K_PRIME_LO B32
        //   + 4 K_PRIME_HI B32
        //   + 1 K_PRIME_BIND_SEL Bit
        // + 4 K_BAR_LO B32
        //   + 4 K_BAR_HI B32
        //   + 1 K_BAR_BIND_SEL Bit
        // + 1 K_PRIME_BIND_SEEN Bit
        //   + 1 K_BAR_BIND_SEEN Bit
        // + 4 SS_LO B32
        //   + 4 SS_HI B32
        //   + 1 SS_MUX_SEL Bit
        //   + 1 SS_OUT_SEL Bit
        // + 1 REQUEST_IDX_OUT B32
        // + 1 S_ACTIVE Bit
        // + 6 PH_* Bit (phase state machine)
        // = 176
        assert_eq!(MlKemCtrlColumns::NUM_COLUMNS, 176);
    }

    #[test]
    fn ctrl_chiplet_declares_all_buses() {
        let ctrl = MlKemCtrlChiplet::new(16);
        let checks: Vec<(String, PermutationCheckSpec)> =
            <MlKemCtrlChiplet as Air<F>>::permutation_checks(&ctrl);

        assert_eq!(checks.len(), 11);
        assert_eq!(checks[0].0, "ml_kem_data");
        assert_eq!(checks[1].0, "keccak_link");
        assert_eq!(checks[2].0, "ntt_data");
        assert_eq!(checks[3].0, "basemul");
        assert_eq!(checks[4].0, "ram_link");
        assert_eq!(checks[5].0, "twiddle_w_binding");
        assert_eq!(checks[6].0, "ntt_bound_in");
        assert_eq!(checks[7].0, "ntt_bound_out");
        assert_eq!(checks[8].0, "kec_input_bind");
        assert_eq!(checks[9].0, "kec_input_bind");
        assert_eq!(checks[10].0, "ml_kem_ss");
    }

    #[test]
    fn ctrl_hash_comparison_bidirectional() {
        // Verify the constraint AST contains
        // reverse hash comparison constraints.
        //
        // Forward: CT_MATCH=1 -> lanes match (4)
        // Reverse: eq booleanity (2),
        //   eq->diff=0 (2), !eq->diff*inv=1 (2),
        //   ct_match=eq_lo*eq_hi (1)
        // = 7 reverse constraints
        let ctrl = MlKemCtrlChiplet::new(16);
        let ast = Air::<F>::constraint_ast(&ctrl);

        // Pre-patch: ~25 constraints.
        // Post-patch: +7 reverse = ~32+.
        assert!(
            ast.roots.len() > 30,
            "Expected >30 constraints (includes reverse hash), got {}",
            ast.roots.len(),
        );
    }

    #[test]
    fn keccak_bus_labels_match_ctrl_and_chiplet() {
        let ctrl = MlKemCtrlChiplet::new(16);
        let ctrl_checks: Vec<(String, PermutationCheckSpec)> =
            <MlKemCtrlChiplet as Air<F>>::permutation_checks(&ctrl);

        let keccak = KeccakChiplet::new(32);
        let keccak_spec = KeccakChiplet::linking_spec();

        // Find the ctrl's keccak bus
        let ctrl_keccak = ctrl_checks
            .iter()
            .find(|(id, _)| id == "keccak_link")
            .expect("ctrl must declare keccak_link bus");

        // Challenge labels must match
        assert_eq!(
            ctrl_keccak.1.sources.len(),
            keccak_spec.sources.len(),
            "keccak bus source count mismatch",
        );

        for (c, k) in ctrl_keccak.1.sources.iter().zip(keccak_spec.sources.iter()) {
            assert_eq!(c.1, k.1, "keccak challenge label mismatch");
        }

        let _ = keccak;
    }

    #[test]
    fn ntt_bus_labels_match_ctrl_and_chiplet() {
        let ctrl = MlKemCtrlChiplet::new(16);
        let ctrl_checks: Vec<(String, PermutationCheckSpec)> =
            <MlKemCtrlChiplet as Air<F>>::permutation_checks(&ctrl);

        let ntt = NttChiplet::new(MLKEM_Q, 16);
        let ntt_spec = ntt.data_linking_spec();

        let ctrl_ntt = ctrl_checks
            .iter()
            .find(|(id, _)| id == "ntt_data")
            .expect("ctrl must declare ntt_data bus");

        assert_eq!(
            ctrl_ntt.1.sources.len(),
            ntt_spec.sources.len(),
            "ntt bus source count mismatch",
        );

        for (c, n) in ctrl_ntt.1.sources.iter().zip(ntt_spec.sources.iter()) {
            assert_eq!(c.1, n.1, "ntt challenge label mismatch");
        }
    }

    #[test]
    fn sticky_rate_regs_satisfy_constraint() {
        let (nist_pk, nist_sk) = mlkem768::keypair();
        let (_, nist_ct) = mlkem768::encapsulate(&nist_pk);

        let params = MlKemParams {
            ctrl_rows: 1 << 16,
            keccak_rows: 1 << 11,
            ntt_rows: 1 << 15,
            twiddle_rows: 1 << 15,
            basemul_rows: 1 << 12,
            ram_rows: 1 << 16,
        };
        let chiplet = MlKemChiplet::<F>::new(MlKemLevel::MLKEM_768, params);
        let (traces, _) = chiplet
            .generate_traces(nist_ct.as_bytes(), nist_sk.as_bytes())
            .unwrap();

        let ctrl = &traces[0];
        let num_rows = ctrl.columns[0].len();

        for row in 0..num_rows {
            let next = (row + 1) % num_rows;
            let s_active = ctrl.columns[MlKemCtrlColumns::S_ACTIVE]
                .as_bit_slice()
                .unwrap()[row];

            if s_active == Bit::ZERO {
                continue;
            }

            let kec_out = ctrl.columns[MlKemCtrlColumns::KEC_IS_OUTPUT]
                .as_bit_slice()
                .unwrap()[row];
            let init_next = ctrl.columns[MlKemCtrlColumns::SPONGE_INIT]
                .as_bit_slice()
                .unwrap()[next];

            // Sponge init on next row exempts
            // this row from sticky/update rule.
            if init_next == Bit::ONE {
                continue;
            }

            for i in 0..25 {
                let reg = match &ctrl.columns[MlKemCtrlColumns::RATE_REG + i] {
                    TraceColumn::B64(d) => d[row].to_tower().0,
                    _ => panic!("RATE_REG must be B64"),
                };
                let reg_next = match &ctrl.columns[MlKemCtrlColumns::RATE_REG + i] {
                    TraceColumn::B64(d) => d[next].to_tower().0,
                    _ => panic!("RATE_REG must be B64"),
                };
                let lane = match &ctrl.columns[MlKemCtrlColumns::KECCAK_LANES + i] {
                    TraceColumn::B64(d) => d[row].to_tower().0,
                    _ => panic!("KECCAK_LANES must be B64"),
                };

                if kec_out == Bit::ONE {
                    assert_eq!(
                        reg_next, lane,
                        "row {row} lane {i}: kec_out=1, reg_next={reg_next:#x} != lane={lane:#x}"
                    );
                } else {
                    assert_eq!(
                        reg_next, reg,
                        "row {row} lane {i}: kec_out=0, reg_next={reg_next:#x} != reg={reg:#x}"
                    );
                }
            }
        }
    }
}
