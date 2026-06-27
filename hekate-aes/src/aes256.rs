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

//! AES-256 Round Chiplet.
//!
//! 15 rows per block:
//! 13 full rounds + 1 final + 1 output.
//!
//! KEY_AUX[16] carries the round key from
//! 2 positions back (Nk=8 = 2 round keys).
//! RCON column tracks key schedule Rcon
//! independently from ROUND_NUM (Rcon
//! doubles every 2 rows, not every row).
//! S_EVEN selects RotWord (even) vs
//! direct (odd) S-box input.

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use hekate_core::errors::Error;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceCompatibleField};
use hekate_math::TowerField;
use hekate_math::{Flat, HardwareField, PackableField};
use hekate_program::Air;
use hekate_program::chiplet::CompositeChiplet;
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::define_columns;
use hekate_program::expander::VirtualExpander;
use hekate_program::permutation::{PermutationCheckSpec, REQUEST_IDX_LABEL, Source};

use super::sbox_rom;
use super::{AES_BYTE_LABELS, ROT_MAP, SBOX_IN_LABELS, SBOX_OUT_LABELS};

#[rustfmt::skip]
const AES256_KEY_LABELS: [&[u8]; 32] = [
    b"aes_key_byte_0",  b"aes_key_byte_1",
    b"aes_key_byte_2",  b"aes_key_byte_3",
    b"aes_key_byte_4",  b"aes_key_byte_5",
    b"aes_key_byte_6",  b"aes_key_byte_7",
    b"aes_key_byte_8",  b"aes_key_byte_9",
    b"aes_key_byte_10", b"aes_key_byte_11",
    b"aes_key_byte_12", b"aes_key_byte_13",
    b"aes_key_byte_14", b"aes_key_byte_15",
    b"aes_key_byte_16", b"aes_key_byte_17",
    b"aes_key_byte_18", b"aes_key_byte_19",
    b"aes_key_byte_20", b"aes_key_byte_21",
    b"aes_key_byte_22", b"aes_key_byte_23",
    b"aes_key_byte_24", b"aes_key_byte_25",
    b"aes_key_byte_26", b"aes_key_byte_27",
    b"aes_key_byte_28", b"aes_key_byte_29",
    b"aes_key_byte_30", b"aes_key_byte_31",
];

// Physical layout. Column order must
// match VirtualExpander sequence exactly.
// KS_INV are B8 columns
// bit-decomposed by the expander.
define_columns! {
    pub PhysAes256Columns {
        P_STATE_IN: [B8; 16],
        P_SBOX_OUT: [B8; 16],
        P_ROUND_KEY: [B8; 16],

        // FIPS 197 §5.2, Nk=8:
        // key derives from w[i-8]
        // (2 round keys back).
        P_KEY_AUX: [B8; 16],
        P_ROUND_NUM: B8,

        // Rcon doubles every 2 rows,
        // not every row like ROUND_NUM.
        P_RCON: B8,
        P_S_ROUND: Bit,
        P_S_FINAL: Bit,
        P_S_IN_OUT: Bit,
        P_S_ACTIVE: Bit,
        P_S_INPUT: Bit,

        // Even:
        // SubWord(RotWord) + Rcon.
        // Odd:
        // SubWord only.
        P_S_EVEN: Bit,
        P_K0: [B8; 32],

        // S-box input:
        // RotWord(RK) on even,
        // RK[12..15] on odd.
        // Constrained to match source.
        P_KS_INPUT: [B8; 4],
        P_KS_SUB: [B8; 4],
        P_KS_INV: [B8; 4],
        P_KS_Z: [Bit; 4],
        P_REQUEST_IDX_LINK: B32,
        P_REQUEST_IDX_KEY: B32,
    }
}

// Virtual column indices.
// Constraints reference these.
define_columns! {
    pub Aes256Columns {
        STATE_IN: [B8; 16],
        SBOX_OUT: [B8; 16],
        ROUND_KEY: [B8; 16],
        KEY_AUX: [B8; 16],

        // GF(2^8) round counter.
        // Doubles each round:
        // 1, 2, 4, ..., 0xAB, 0x4D.
        // Enforces exactly 13 s_round
        // rows before s_final.
        ROUND_NUM: B8,

        // Key schedule Rcon.
        // Doubles every 2 rows
        // (odd->even transitions).
        // Decoupled from ROUND_NUM because
        // Nk=8 key schedule iterates at
        // half the round rate.
        RCON: B8,

        S_ROUND: Bit,
        S_FINAL: Bit,
        S_IN_OUT: Bit,

        // Gates S-box bus lookups.
        // 1 on all round rows
        // (s_round OR s_final).
        S_ACTIVE: Bit,

        // s_in_out ∧ s_round (input row only).
        S_INPUT: Bit,

        // Even/odd cascade selector.
        // Toggles each s_round row.
        S_EVEN: Bit,

        // Raw AES-256 key (32 bytes).
        // Populated on s_input rows.
        // Bound to consumer via "aes_key_in" bus.
        K0: [B8; 32],

        // S-box input:
        // RotWord(RK) on even,
        // RK[12..15] on odd.
        KS_INPUT: [B8; 4],

        // SubWord(KS_INPUT) witness.
        KS_SUB: [B8; 4],
        KS_INV_BITS: [Bit; 32],
        KS_Z: [Bit; 4],

        // Partner CPU row index for the
        // aes256_link bus (S_IN_OUT-gated:
        // input row + output row per block).
        REQUEST_IDX_LINK: B32,

        // Partner CPU row index for the
        // aes256_key_in bus (S_INPUT-gated:
        // input row only).
        REQUEST_IDX_KEY: B32,
    }
}

#[derive(Clone, Debug)]
pub struct AesRound256Air {
    pub num_rows: usize,
}

impl AesRound256Air {
    pub const LINK_BUS_ID: &'static str = "aes256_link";
    pub const KEY_BUS_ID: &'static str = "aes256_key_in";

    pub(crate) fn new(num_rows: usize) -> Self {
        Self { num_rows }
    }

    pub fn for_constraints() -> Self {
        Self { num_rows: 0 }
    }

    /// External bus:
    /// 16 state_in bytes gated by s_in_out.
    pub fn link_spec() -> PermutationCheckSpec {
        let mut sources: Vec<_> = (0..16)
            .map(|i| {
                (
                    Source::Column(Aes256Columns::STATE_IN + i),
                    AES_BYTE_LABELS[i],
                )
            })
            .collect();

        sources.push((
            Source::Column(Aes256Columns::REQUEST_IDX_LINK),
            REQUEST_IDX_LABEL,
        ));

        PermutationCheckSpec::new(sources, Some(Aes256Columns::S_IN_OUT))
    }

    /// External bus:
    /// 32 K0 bytes gated by s_input.
    pub fn key_spec() -> PermutationCheckSpec {
        let mut sources: Vec<_> = (0..32)
            .map(|i| (Source::Column(Aes256Columns::K0 + i), AES256_KEY_LABELS[i]))
            .collect();

        sources.push((
            Source::Column(Aes256Columns::REQUEST_IDX_KEY),
            REQUEST_IDX_LABEL,
        ));

        PermutationCheckSpec::new(sources, Some(Aes256Columns::S_INPUT))
    }

    /// Per-byte-position S-box bus specs.
    pub fn sbox_specs() -> Vec<(String, PermutationCheckSpec)> {
        let mut sources = Vec::with_capacity(32);
        for i in 0..16 {
            sources.push((
                Source::Column(Aes256Columns::STATE_IN + i),
                SBOX_IN_LABELS[i],
            ));
            sources.push((
                Source::Column(Aes256Columns::SBOX_OUT + i),
                SBOX_OUT_LABELS[i],
            ));
        }

        let spec = PermutationCheckSpec::new(sources, Some(Aes256Columns::S_ACTIVE))
            .with_clock_waiver(
                "see hekate-chiplets/src/aes/aes256.rs: AES<>SboxRom internal; \
                 phantom blocks caught at link+key v3",
            );

        vec![(sbox_rom::SboxRomChiplet::BUS_ID.into(), spec)]
    }
}

impl<F: TowerField> Air<F> for AesRound256Air {
    fn name(&self) -> String {
        "AesRound256Air".to_string()
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: once_cell::race::OnceBox<Vec<ColumnType>> = once_cell::race::OnceBox::new();
        LAYOUT.get_or_init(|| Box::new(PhysAes256Columns::build_layout()))
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        let mut checks = Vec::with_capacity(3);
        checks.push((Self::LINK_BUS_ID.into(), Self::link_spec()));
        checks.push((Self::KEY_BUS_ID.into(), Self::key_spec()));
        checks.extend(Self::sbox_specs());

        checks
    }

    fn virtual_expander(&self) -> Option<&VirtualExpander> {
        static E: once_cell::race::OnceBox<VirtualExpander> = once_cell::race::OnceBox::new();
        Some(E.get_or_init(|| {
            Box::new(
                VirtualExpander::new()
                    .pass_through(66, ColumnType::B8) // STATE_IN..RCON
                    .control_bits(6) // S_ROUND..S_EVEN
                    .pass_through(32, ColumnType::B8) // K0
                    .pass_through(4, ColumnType::B8) // KS_INPUT
                    .pass_through(4, ColumnType::B8) // KS_SUB
                    .expand_bits(4, ColumnType::B8) // KS_INV -> KS_INV_BITS
                    .control_bits(4) // KS_Z
                    .pass_through(2, ColumnType::B32) // REQUEST_IDX_LINK, REQUEST_IDX_KEY
                    .build()
                    .expect("AesRound256Air expander"),
            )
        }))
    }

    #[allow(clippy::needless_range_loop)]
    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();

        let s_round = cs.col(Aes256Columns::S_ROUND);
        let s_final = cs.col(Aes256Columns::S_FINAL);
        let s_in_out = cs.col(Aes256Columns::S_IN_OUT);
        let s_active = cs.col(Aes256Columns::S_ACTIVE);

        cs.assert_boolean(s_round);
        cs.assert_boolean(s_final);
        cs.assert_boolean(s_in_out);
        cs.assert_boolean(s_active);

        cs.constrain(s_round * s_final);

        // s_active = s_round | s_final
        // (in GF(2): a + b + a*b)
        cs.constrain(s_active + s_round + s_final + s_round * s_final);

        // Prevents Ghost Protocol:
        // zeroing s_round mid-block to skip a round.
        let next_s_active = cs.next(Aes256Columns::S_ACTIVE);
        let next_s_in_out = cs.next(Aes256Columns::S_IN_OUT);

        cs.constrain(s_active * (cs.one() + next_s_active + next_s_in_out));

        let round_num = cs.col(Aes256Columns::ROUND_NUM);
        let next_round_num = cs.next(Aes256Columns::ROUND_NUM);
        let s_input = cs.col(Aes256Columns::S_INPUT);
        let s_even = cs.col(Aes256Columns::S_EVEN);
        let rcon = cs.col(Aes256Columns::RCON);
        let next_rcon = cs.next(Aes256Columns::RCON);

        let one = cs.one();
        let two = cs.constant(F::from(2u8));

        cs.assert_boolean(s_input);
        cs.assert_boolean(s_even);

        cs.constrain(s_input + s_in_out * s_round);

        // Round counter:
        // 13 xtime steps, final = 0x4D
        cs.assert_zero_when(s_input, round_num + one);
        cs.assert_zero_when(s_round, next_round_num + two * round_num);
        cs.assert_zero_when(s_final, round_num + cs.constant(F::from(0x4Du8)));

        // S_EVEN:
        // init=1, toggles each s_round row
        cs.assert_zero_when(s_input, s_even + one);
        cs.assert_zero_when(s_round, cs.next(Aes256Columns::S_EVEN) + s_even + one);

        // RCON:
        // init=1. Stays on even->odd,
        // doubles on odd->even transitions.
        cs.assert_zero_when(s_input, rcon + one);
        cs.assert_zero_when(s_round * s_even, next_rcon + rcon);
        cs.assert_zero_when(s_round * (one + s_even), next_rcon + two * rcon);

        // =============================================================
        // Round function
        // =============================================================

        super::build_round_constraints(
            &cs,
            Aes256Columns::STATE_IN,
            Aes256Columns::SBOX_OUT,
            Aes256Columns::ROUND_KEY,
            Aes256Columns::S_ROUND,
            Aes256Columns::S_FINAL,
        );

        // =============================================================
        // Key schedule: inline FIPS 197 §5.2, Nk=8
        // =============================================================

        // KS_INPUT source binding.
        // Even:
        // KS_INPUT = RotWord(ROUND_KEY)
        // Odd:
        // KS_INPUT = ROUND_KEY[12..15]
        let s_round_even = s_round * s_even;
        let s_round_odd = s_round * (one + s_even);

        for j in 0..4usize {
            let ks_in = cs.col(Aes256Columns::KS_INPUT + j);
            let rot = cs.col(Aes256Columns::ROUND_KEY + ROT_MAP[j]);
            let direct = cs.col(Aes256Columns::ROUND_KEY + 12 + j);

            cs.assert_zero_when(s_round_even, ks_in + rot);
            cs.assert_zero_when(s_round_odd, ks_in + direct);
        }

        // S-box inversion on KS_INPUT (degree 3).
        super::build_sbox_inversion_constraints(
            &cs,
            core::array::from_fn(|j| Aes256Columns::KS_INPUT + j),
            Aes256Columns::KS_SUB,
            Aes256Columns::KS_INV_BITS,
            Aes256Columns::KS_Z,
            Aes256Columns::S_ROUND,
        );

        // Word cascade:
        // base = KEY_AUX,
        // Rcon = s_even * rcon (zero on odd rounds).
        for j in 0..16usize {
            let next_rk = cs.next(Aes256Columns::ROUND_KEY + j);
            let aux = cs.col(Aes256Columns::KEY_AUX + j);

            let body = match j {
                0 => next_rk + aux + cs.col(Aes256Columns::KS_SUB) + s_even * rcon,
                1..=3 => next_rk + aux + cs.col(Aes256Columns::KS_SUB + j),
                4..=15 => next_rk + aux + cs.next(Aes256Columns::ROUND_KEY + j - 4),
                _ => unreachable!(),
            };

            cs.assert_zero_when(s_round, body);
        }

        // KEY_AUX slides forward each round:
        // next row's KEY_AUX = current ROUND_KEY
        for j in 0..16usize {
            cs.assert_zero_when(
                s_round,
                cs.next(Aes256Columns::KEY_AUX + j) + cs.col(Aes256Columns::ROUND_KEY + j),
            );
        }

        // Init:
        // RK_1 = K0[16..31],
        // KEY_AUX = K0[0..15].
        // No key expansion, both halves
        // come directly from K0.
        for j in 0..16usize {
            cs.assert_zero_when(
                s_input,
                cs.col(Aes256Columns::ROUND_KEY + j) + cs.col(Aes256Columns::K0 + 16 + j),
            );
            cs.assert_zero_when(
                s_input,
                cs.col(Aes256Columns::KEY_AUX + j) + cs.col(Aes256Columns::K0 + j),
            );
        }

        // S_EVEN, KS_Z, KS_INV load-bearing only on s_round rows
        let not_s_round = one + s_round;
        cs.assert_zero_when(not_s_round, s_even);

        for i in 0..4 {
            cs.assert_zero_when(not_s_round, cs.col(Aes256Columns::KS_Z + i));

            let ks_inv_byte = cs.sum(
                &(0..8)
                    .map(|k| {
                        cs.scale(
                            F::from(1u8 << k),
                            cs.col(Aes256Columns::KS_INV_BITS + i * 8 + k),
                        )
                    })
                    .collect::<Vec<_>>(),
            );

            cs.assert_zero_when(not_s_round, ks_inv_byte);
        }

        cs.build()
    }
}

// =================================================================
// CPU-Side Interface
// =================================================================

define_columns! {
    pub CpuAes256Columns {
        KEY: [B8; 32],
        KEY_SELECTOR: Bit,
        DATA: [B8; 16],
        SELECTOR: Bit,
    }
}

pub struct CpuAes256Unit;

impl CpuAes256Unit {
    pub fn num_columns() -> usize {
        CpuAes256Columns::NUM_COLUMNS
    }

    pub fn linking_spec() -> PermutationCheckSpec {
        let data_cols: [usize; 16] = core::array::from_fn(|i| CpuAes256Columns::DATA + i);

        Self::linking_spec_at(&data_cols, CpuAes256Columns::SELECTOR)
    }

    pub fn key_linking_spec() -> PermutationCheckSpec {
        let key_cols: [usize; 32] = core::array::from_fn(|i| CpuAes256Columns::KEY + i);

        Self::key_linking_spec_at(&key_cols, CpuAes256Columns::KEY_SELECTOR)
    }

    pub fn linking_spec_at(data_cols: &[usize; 16], selector_col: usize) -> PermutationCheckSpec {
        let mut sources: Vec<_> = (0..16)
            .map(|i| (Source::Column(data_cols[i]), AES_BYTE_LABELS[i]))
            .collect();

        sources.push((Source::RowIndexLeBytes(4), REQUEST_IDX_LABEL));

        PermutationCheckSpec::new(sources, Some(selector_col))
    }

    pub fn key_linking_spec_at(
        key_cols: &[usize; 32],
        key_selector_col: usize,
    ) -> PermutationCheckSpec {
        let mut sources: Vec<_> = (0..32)
            .map(|i| (Source::Column(key_cols[i]), AES256_KEY_LABELS[i]))
            .collect();

        sources.push((Source::RowIndexLeBytes(4), REQUEST_IDX_LABEL));

        PermutationCheckSpec::new(sources, Some(key_selector_col))
    }
}

// =================================================================
// AES-256 Composite Chiplet
// =================================================================

#[derive(Clone)]
pub struct Aes256Chiplet<F: TraceCompatibleField> {
    composite: CompositeChiplet<F>,
    num_rows: usize,
    sbox_rom_rows: usize,
}

impl<F> Aes256Chiplet<F>
where
    F: TowerField + TraceCompatibleField + PackableField + HardwareField + 'static,
    <F as PackableField>::Packed: Copy + Send + Sync,
    Flat<F>: Send + Sync,
{
    pub fn new(num_rows: usize, sbox_rom_rows: usize) -> Result<Self, Error> {
        if !num_rows.is_power_of_two() {
            return Err(Error::Protocol {
                protocol: "aes256_chiplet",
                message: "num_rows must be power of 2",
            });
        }

        let round_air = AesRound256Air::new(num_rows);
        let sbox_rom = sbox_rom::SboxRomChiplet::new(sbox_rom_rows)?;

        let composite = CompositeChiplet::<F>::builder("aes256")
            .chiplet(round_air)
            .chiplet(sbox_rom)
            .external_bus(AesRound256Air::LINK_BUS_ID, AesRound256Air::link_spec())
            .external_bus(AesRound256Air::KEY_BUS_ID, AesRound256Air::key_spec())
            .build()?;

        Ok(Self {
            composite,
            num_rows,
            sbox_rom_rows,
        })
    }

    pub fn composite(&self) -> &CompositeChiplet<F> {
        &self.composite
    }

    pub fn generate_traces(
        &self,
        calls: &[super::trace::Aes256Call],
    ) -> Result<Vec<ColumnTrace>, Error> {
        self.generate_traces_with(calls, None)
    }

    pub fn generate_traces_at(
        &self,
        calls: &[super::trace::Aes256Call],
        triples: &[(u32, u32, u32)],
    ) -> Result<Vec<ColumnTrace>, Error> {
        self.generate_traces_with(calls, Some(triples))
    }

    fn generate_traces_with(
        &self,
        calls: &[super::trace::Aes256Call],
        triples: Option<&[(u32, u32, u32)]>,
    ) -> Result<Vec<ColumnTrace>, Error> {
        let aes_trace = super::trace::generate_aes_trace(calls, triples, self.num_rows)?;

        let mut sbox_rounds = Vec::new();
        let s_active = aes_trace.columns[PhysAes256Columns::P_S_ACTIVE]
            .as_bit_slice()
            .ok_or(Error::Protocol {
                protocol: "aes256_chiplet",
                message: "S_ACTIVE column type mismatch",
            })?;

        for (row, &active) in s_active.iter().enumerate() {
            if active != hekate_math::Bit::ONE {
                continue;
            }

            let mut inputs = [0u8; 16];
            let mut outputs = [0u8; 16];

            for j in 0..16 {
                inputs[j] = aes_trace.columns[PhysAes256Columns::P_STATE_IN + j]
                    .as_b8_slice()
                    .unwrap()[row]
                    .to_tower()
                    .0;
                outputs[j] = aes_trace.columns[PhysAes256Columns::P_SBOX_OUT + j]
                    .as_b8_slice()
                    .unwrap()[row]
                    .to_tower()
                    .0;
            }

            sbox_rounds.push(sbox_rom::SboxRound { inputs, outputs });
        }

        let sbox_trace = sbox_rom::generate_sbox_rom_trace(&sbox_rounds, self.sbox_rom_rows)?;

        Ok(vec![aes_trace, sbox_trace])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hekate_math::Block128;

    type F = Block128;

    #[test]
    fn physical_column_count() {
        let layout = PhysAes256Columns::build_layout();
        assert_eq!(layout.len(), PhysAes256Columns::NUM_COLUMNS);

        assert_eq!(PhysAes256Columns::P_STATE_IN, 0);
        assert_eq!(PhysAes256Columns::P_ROUND_KEY, 32);
        assert_eq!(PhysAes256Columns::P_KEY_AUX, 48);
    }

    #[test]
    fn virtual_column_count() {
        assert_eq!(Aes256Columns::STATE_IN, 0);
        assert_eq!(Aes256Columns::SBOX_OUT, 16);
        assert_eq!(Aes256Columns::ROUND_KEY, 32);
        assert_eq!(Aes256Columns::KEY_AUX, 48);
    }

    #[test]
    fn constraint_count() {
        let ast: ConstraintAst<F> = AesRound256Air::for_constraints().constraint_ast();

        assert_eq!(ast.roots.len(), 183);
    }

    #[test]
    fn link_spec_structure() {
        let spec = AesRound256Air::link_spec();
        assert_eq!(spec.num_sources(), 17);
        assert_eq!(spec.selector, Some(Aes256Columns::S_IN_OUT));
        assert_eq!(spec.sources[16].1, REQUEST_IDX_LABEL);
    }

    #[test]
    fn key_spec_structure() {
        let spec = AesRound256Air::key_spec();
        assert_eq!(spec.num_sources(), 33);
        assert_eq!(spec.selector, Some(Aes256Columns::S_INPUT));
        assert_eq!(spec.sources[32].1, REQUEST_IDX_LABEL);
    }

    #[test]
    fn sbox_specs_structure() {
        let specs = AesRound256Air::sbox_specs();
        assert_eq!(specs.len(), 1);

        let (bus_id, spec) = &specs[0];
        assert_eq!(bus_id, sbox_rom::SboxRomChiplet::BUS_ID);
        assert_eq!(spec.num_sources(), 32);
        assert_eq!(spec.selector, Some(Aes256Columns::S_ACTIVE));
    }

    #[test]
    fn virtual_expander_dimensions() {
        let air = AesRound256Air::for_constraints();
        let exp = Air::<F>::virtual_expander(&air).expect("expander must exist");

        assert_eq!(exp.num_physical_columns(), PhysAes256Columns::NUM_COLUMNS);
        assert_eq!(exp.num_virtual_columns(), Aes256Columns::NUM_COLUMNS);
    }

    #[test]
    fn composite_builds() {
        let aes = Aes256Chiplet::<F>::new(16, 256).unwrap();
        assert_eq!(aes.composite().flatten_defs().unwrap().len(), 2);
    }

    #[test]
    fn new_validates() {
        assert!(Aes256Chiplet::<F>::new(100, 256).is_err());
        assert!(Aes256Chiplet::<F>::new(16, 7).is_err());
        assert!(Aes256Chiplet::<F>::new(16, 16).is_ok());
    }
}
