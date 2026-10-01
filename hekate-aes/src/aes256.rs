// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! AES-256 Round Chiplet.
//!
//! 15 rows per block:
//! 13 full rounds + 1 final + 1 output.
//!
//! KEY_AUX[16] carries the round key from
//! 2 positions back (Nk=8 = 2 round keys).

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use hekate_core::errors::Error;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceCompatibleField};
use hekate_math::TowerField;
use hekate_math::{HardwareField, PackableField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::define_columns;
use hekate_program::expander::VirtualExpander;
use hekate_program::permutation::{BusKind, PermutationCheckSpec, Service, ServiceSlot};
use hekate_program::{Air, FixedColumn, FixedShape, fix};
use zeroize::Zeroizing;

use super::sbox_rom;
use super::{AES_BYTE_LABELS, ROT_MAP};

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

// Physical layout. Column order must match VirtualExpander sequence
// exactly. ROUND_IDX / KS_INV are bit-decomposed by the expander.
define_columns! {
    pub PhysAes256Columns {
        P_STATE_IN: [B8; 16],
        P_SBOX_OUT: [B8; 16],
        P_ROUND_KEY: [B8; 16],

        // FIPS 197 §5.2, Nk=8:
        // key derives from w[i-8]
        // (2 round keys back).
        P_KEY_AUX: [B8; 16],
        P_ROUND_IDX: B16,
        P_S_ROUND: Bit,
        P_S_FINAL: Bit,
        P_S_IN_OUT: Bit,
        P_S_ACTIVE: Bit,
        P_S_INPUT: Bit,
        P_K0: [B8; 32],

        // S-box input:
        // RotWord(RK) on even,
        // RK[12..15] on odd.
        // Constrained to match source.
        P_KS_INPUT: [B8; 4],
        P_KS_SUB: [B8; 4],
        P_KS_INV: [B8; 4],
        P_KS_Z: [Bit; 4],
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

        // One-hot over the block's active rows.
        // Even index selects RotWord and
        // contributes Rcon, odd selects direct.
        ROUND_BITS: [Bit; 16],

        S_ROUND: Bit,
        S_FINAL: Bit,
        S_IN_OUT: Bit,

        // Gates S-box bus lookups.
        // 1 on all round rows
        // (s_round OR s_final).
        S_ACTIVE: Bit,

        // s_in_out ∧ s_round (input row only).
        S_INPUT: Bit,

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
    }
}

/// AES-256 as 14 round rows plus one output row per block.
///
/// Blocks sit contiguously at rows `0..15·num_blocks`.
/// The schedule columns are fixed columns on
/// a stride-15 cadence of `num_blocks` blocks.
#[derive(Clone, Debug)]
pub struct AesRound256Air {
    num_blocks: usize,
}

impl AesRound256Air {
    pub const LINK_BUS_ID: &'static str = "aes256_link";
    pub const KEY_BUS_ID: &'static str = "aes256_key_in";

    /// 13 full rounds and the final round.
    pub const ACTIVE_ROWS: usize = 14;

    /// Rounds plus the output row.
    pub const BLOCK_ROWS: usize = Self::ACTIVE_ROWS + 1;

    /// FIPS 197 §5.2 round constants, Nk=8.
    const RCON: [u8; Self::ACTIVE_ROWS / 2] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40];

    pub(crate) fn new(num_blocks: usize) -> Self {
        Self { num_blocks }
    }

    /// Both endpoints derive from this schema:
    /// 16 state bytes, then the emit rank.
    pub fn link_service() -> Service {
        let mut slots = Vec::with_capacity(17);
        for label in AES_BYTE_LABELS {
            slots.push(ServiceSlot::Value(label));
        }

        slots.push(ServiceSlot::EmitRank);

        Service {
            bus_id: Self::LINK_BUS_ID,
            kind: BusKind::Permutation,
            slots,
        }
    }

    /// Both endpoints derive from this schema:
    /// 32 key bytes, then the emit rank.
    pub fn key_service() -> Service {
        let mut slots = Vec::with_capacity(33);
        for label in AES256_KEY_LABELS {
            slots.push(ServiceSlot::Value(label));
        }

        slots.push(ServiceSlot::EmitRank);

        Service {
            bus_id: Self::KEY_BUS_ID,
            kind: BusKind::Permutation,
            slots,
        }
    }

    pub fn link_spec() -> PermutationCheckSpec {
        let values: Vec<usize> = (0..16).map(|i| Aes256Columns::STATE_IN + i).collect();

        Self::link_service()
            .respond(&values, Aes256Columns::S_IN_OUT)
            .expect("service slots match the responder columns")
    }

    pub fn key_spec() -> PermutationCheckSpec {
        let values: Vec<usize> = (0..32).map(|i| Aes256Columns::K0 + i).collect();

        Self::key_service()
            .respond(&values, Aes256Columns::S_INPUT)
            .expect("service slots match the responder columns")
    }

    pub fn sbox_specs() -> Vec<(String, PermutationCheckSpec)> {
        let spec = sbox_rom::SboxRomChiplet::service()
            .request(
                &sbox_rom::SboxRomChiplet::byte_columns(
                    Aes256Columns::STATE_IN,
                    Aes256Columns::SBOX_OUT,
                ),
                Aes256Columns::S_ACTIVE,
            )
            .expect("service slots match the requester columns");

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
                    .pass_through(64, ColumnType::B8) // STATE_IN..KEY_AUX
                    .expand_bits(1, ColumnType::B16) // ROUND_IDX -> ROUND_BITS
                    .control_bits(5) // S_ROUND..S_INPUT
                    .pass_through(32, ColumnType::B8) // K0
                    .pass_through(4, ColumnType::B8) // KS_INPUT
                    .pass_through(4, ColumnType::B8) // KS_SUB
                    .expand_bits(4, ColumnType::B8) // KS_INV -> KS_INV_BITS
                    .control_bits(4) // KS_Z
                    .build()
                    .expect("AesRound256Air expander"),
            )
        }))
    }

    #[allow(clippy::needless_range_loop)]
    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();

        let s_round = cs.col(Aes256Columns::S_ROUND);
        let s_input = cs.col(Aes256Columns::S_INPUT);
        let one = cs.one();

        // The schedule is cadence pins; only the round
        // function and key schedule remain as roots.
        const ACTIVE: usize = AesRound256Air::ACTIVE_ROWS;

        let round = |k: usize| cs.col(Aes256Columns::ROUND_BITS + k);

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
        // Even: KS_INPUT = RotWord(ROUND_KEY)
        // Odd: KS_INPUT = ROUND_KEY[12..15]
        let s_round_even = cs.sum(&(0..ACTIVE - 1).step_by(2).map(round).collect::<Vec<_>>());
        let s_round_odd = cs.sum(&(1..ACTIVE - 1).step_by(2).map(round).collect::<Vec<_>>());

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

        // Rcon reads off the index, no round constant
        // is ever a witness. Odd rounds contribute none.
        let rcon = cs.sum(
            &(0..ACTIVE - 1)
                .step_by(2)
                .map(|k| cs.scale(F::from(Self::RCON[k / 2]), round(k)))
                .collect::<Vec<_>>(),
        );

        // Word cascade:
        // base = KEY_AUX
        for j in 0..16usize {
            let next_rk = cs.next(Aes256Columns::ROUND_KEY + j);
            let aux = cs.col(Aes256Columns::KEY_AUX + j);

            let body = match j {
                0 => next_rk + aux + cs.col(Aes256Columns::KS_SUB) + rcon,
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
        // RK_1 = K0[16..31], KEY_AUX = K0[0..15].
        // No key expansion, both halves come directly from K0.
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

        // KS_Z, KS_INV load-bearing only on s_round rows
        let not_s_round = one + s_round;

        for i in 0..4 {
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

            cs.assert_zero_when(not_s_round, cs.col(Aes256Columns::KS_Z + i));
            cs.assert_zero_when(not_s_round, ks_inv_byte);
        }

        cs.build()
    }

    fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
        const ACTIVE: usize = AesRound256Air::ACTIVE_ROWS;

        let stride = Self::BLOCK_ROWS;
        let block = |values: Vec<F>| FixedShape::Cadence {
            stride,
            count: self.num_blocks,
            origin: 0,
            values,
        };

        let shape = |pred: &dyn Fn(usize) -> bool| {
            block(
                (0..stride)
                    .map(|off| if pred(off) { F::ONE } else { F::ZERO })
                    .collect(),
            )
        };

        let mut pins = Vec::with_capacity(21);

        pins.push(fix(Aes256Columns::S_ROUND, shape(&|off| off < ACTIVE - 1)));
        pins.push(fix(Aes256Columns::S_FINAL, shape(&|off| off == ACTIVE - 1)));
        pins.push(fix(
            Aes256Columns::S_IN_OUT,
            shape(&|off| off == 0 || off == stride - 1),
        ));
        pins.push(fix(Aes256Columns::S_ACTIVE, shape(&|off| off < ACTIVE)));
        pins.push(fix(Aes256Columns::S_INPUT, shape(&|off| off == 0)));

        for k in 0..16 {
            pins.push(fix(
                Aes256Columns::ROUND_BITS + k,
                shape(&|off| k < ACTIVE && off == k),
            ));
        }

        pins
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

// =================================================================
// AES-256 Chiplet
// =================================================================

#[derive(Clone)]
pub struct Aes256Chiplet {
    round_air: AesRound256Air,
    sbox_rom: sbox_rom::SboxRomChiplet,
    num_rows: usize,
    sbox_rom_rows: usize,
}

impl Aes256Chiplet {
    pub const EXTERNAL_BUS_IDS: [&'static str; 2] =
        [AesRound256Air::LINK_BUS_ID, AesRound256Air::KEY_BUS_ID];

    pub fn new(num_rows: usize, sbox_rom_rows: usize, num_blocks: usize) -> Result<Self, Error> {
        if !num_rows.is_power_of_two() {
            return Err(Error::Protocol {
                protocol: "aes256_chiplet",
                message: "num_rows must be power of 2",
            });
        }

        let span = num_blocks
            .checked_mul(AesRound256Air::BLOCK_ROWS)
            .ok_or(Error::Protocol {
                protocol: "aes256_chiplet",
                message: "num_blocks exceeds the trace height",
            })?;

        if span > num_rows {
            return Err(Error::Protocol {
                protocol: "aes256_chiplet",
                message: "num_blocks exceeds the trace height",
            });
        }

        let round_air = AesRound256Air::new(num_blocks);
        let sbox_rom =
            sbox_rom::SboxRomChiplet::new(sbox_rom_rows, num_blocks * AesRound256Air::ACTIVE_ROWS)?;

        Ok(Self {
            round_air,
            sbox_rom,
            num_rows,
            sbox_rom_rows,
        })
    }

    /// The round table and the S-box ROM as chiplets
    /// for a host program to attach, in the order
    /// `generate_traces` returns their traces.
    pub fn defs<F>(&self) -> Result<Vec<ChipletDef<F>>, Error>
    where
        F: TraceCompatibleField + PackableField + HardwareField + 'static,
        <F as PackableField>::Packed: Copy + Send + Sync,
    {
        Ok(vec![
            ChipletDef::from_air(&self.round_air)?,
            ChipletDef::from_air(&self.sbox_rom)?,
        ])
    }

    pub fn generate_traces(
        &self,
        calls: &[super::trace::Aes256Call],
    ) -> Result<Vec<ColumnTrace>, Error> {
        let aes_trace = super::trace::generate_aes_trace(calls, self.num_rows)?;

        let s_active = aes_trace.columns[PhysAes256Columns::P_S_ACTIVE]
            .as_bit_slice()
            .ok_or(Error::Protocol {
                protocol: "aes256_chiplet",
                message: "S_ACTIVE column type mismatch",
            })?;

        let active_rows = s_active
            .iter()
            .filter(|&&active| active == hekate_math::Bit::ONE)
            .count();

        let mut sbox_rounds = Zeroizing::new(Vec::with_capacity(active_rows));
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
    use hekate_math::{Block128, Flat};
    use hekate_program::permutation::EMIT_RANK_LABEL;

    type F = Block128;

    #[test]
    fn physical_column_count() {
        let layout = PhysAes256Columns::build_layout();
        assert_eq!(layout.len(), PhysAes256Columns::NUM_COLUMNS);

        assert_eq!(PhysAes256Columns::NUM_COLUMNS, 118);
        assert_eq!(PhysAes256Columns::P_STATE_IN, 0);
        assert_eq!(PhysAes256Columns::P_ROUND_KEY, 32);
        assert_eq!(PhysAes256Columns::P_KEY_AUX, 48);
        assert_eq!(PhysAes256Columns::P_ROUND_IDX, 64);
        assert_eq!(PhysAes256Columns::P_S_ROUND, 65);
        assert_eq!(PhysAes256Columns::P_K0, 70);
        assert_eq!(PhysAes256Columns::P_KS_INPUT, 102);
        assert_eq!(PhysAes256Columns::P_KS_SUB, 106);
        assert_eq!(PhysAes256Columns::P_KS_INV, 110);
        assert_eq!(PhysAes256Columns::P_KS_Z, 114);
    }

    #[test]
    fn virtual_column_count() {
        assert_eq!(Aes256Columns::NUM_COLUMNS, 161);
        assert_eq!(Aes256Columns::STATE_IN, 0);
        assert_eq!(Aes256Columns::SBOX_OUT, 16);
        assert_eq!(Aes256Columns::ROUND_KEY, 32);
        assert_eq!(Aes256Columns::KEY_AUX, 48);
        assert_eq!(Aes256Columns::ROUND_BITS, 64);
        assert_eq!(Aes256Columns::S_ROUND, 80);
        assert_eq!(Aes256Columns::S_FINAL, 81);
        assert_eq!(Aes256Columns::S_IN_OUT, 82);
        assert_eq!(Aes256Columns::S_ACTIVE, 83);
        assert_eq!(Aes256Columns::S_INPUT, 84);
        assert_eq!(Aes256Columns::K0, 85);
        assert_eq!(Aes256Columns::KS_INPUT, 117);
        assert_eq!(Aes256Columns::KS_SUB, 121);
        assert_eq!(Aes256Columns::KS_INV_BITS, 125);
        assert_eq!(Aes256Columns::KS_Z, 157);
    }

    #[test]
    fn constraint_count() {
        let ast: ConstraintAst<F> = AesRound256Air::new(4).constraint_ast();
        assert_eq!(ast.roots.len(), 164);
    }

    #[test]
    fn cadence_pins_pin_every_schedule_column() {
        let air = AesRound256Air::new(4);
        let pins = Air::<F>::fixed_columns(&air);

        assert_eq!(pins.len(), 21);

        let at = |col: usize, row: usize| {
            pins.iter()
                .find(|p| p.col_idx == col)
                .unwrap()
                .shape
                .value_at_row(row, 10)
        };

        let one = Flat::from_raw(F::ONE);
        let zero = Flat::from_raw(F::ZERO);

        assert_eq!(at(Aes256Columns::S_INPUT, 0), one);
        assert_eq!(at(Aes256Columns::S_IN_OUT, 0), one);
        assert_eq!(at(Aes256Columns::S_ROUND, 12), one);
        assert_eq!(at(Aes256Columns::S_ROUND, 13), zero);
        assert_eq!(at(Aes256Columns::S_FINAL, 13), one);
        assert_eq!(at(Aes256Columns::S_ACTIVE, 13), one);
        assert_eq!(at(Aes256Columns::S_ACTIVE, 14), zero);
        assert_eq!(at(Aes256Columns::S_IN_OUT, 14), one);
        assert_eq!(at(Aes256Columns::ROUND_BITS + 5, 15 + 5), one);
        assert_eq!(at(Aes256Columns::ROUND_BITS + 5, 15 + 6), zero);
        assert_eq!(at(Aes256Columns::ROUND_BITS + 14, 14), zero);
        assert_eq!(at(Aes256Columns::S_IN_OUT, 4 * 15), zero);
    }

    #[test]
    fn link_spec_structure() {
        let spec = AesRound256Air::link_spec();

        assert_eq!(spec.num_sources(), 17);
        assert_eq!(spec.selector, Some(Aes256Columns::S_IN_OUT));
        assert_eq!(spec.sources[16].1, EMIT_RANK_LABEL);
    }

    #[test]
    fn link_endpoints_agree() {
        let chiplet = AesRound256Air::link_spec();

        let cpu_values: Vec<usize> = (0..16).map(|i| CpuAes256Columns::DATA + i).collect();

        let cpu = AesRound256Air::link_service()
            .request(&cpu_values, CpuAes256Columns::SELECTOR)
            .unwrap();

        assert_eq!(chiplet.num_sources(), cpu.num_sources());

        for (c, p) in chiplet.sources.iter().zip(cpu.sources.iter()) {
            assert_eq!(c.1, p.1);
        }
    }

    #[test]
    fn key_spec_structure() {
        let spec = AesRound256Air::key_spec();

        assert_eq!(spec.num_sources(), 33);
        assert_eq!(spec.selector, Some(Aes256Columns::S_INPUT));
        assert_eq!(spec.sources[32].1, EMIT_RANK_LABEL);
    }

    #[test]
    fn sbox_specs_structure() {
        let specs = AesRound256Air::sbox_specs();
        assert_eq!(specs.len(), 1);

        let (bus_id, spec) = &specs[0];

        assert_eq!(bus_id, sbox_rom::SboxRomChiplet::BUS_ID);
        assert_eq!(spec.num_sources(), 33);
        assert_eq!(spec.sources[32].1, EMIT_RANK_LABEL);
        assert_eq!(spec.selector, Some(Aes256Columns::S_ACTIVE));
        assert!(spec.clock_waiver.is_none());
    }

    #[test]
    fn virtual_expander_dimensions() {
        let air = AesRound256Air::new(4);
        let exp = Air::<F>::virtual_expander(&air).expect("expander must exist");

        assert_eq!(exp.num_physical_columns(), PhysAes256Columns::NUM_COLUMNS);
        assert_eq!(exp.num_virtual_columns(), Aes256Columns::NUM_COLUMNS);
    }

    #[test]
    fn defs_follow_trace_order() {
        let aes = Aes256Chiplet::new(16, 256, 1).unwrap();
        let names: Vec<String> = aes
            .defs::<F>()
            .unwrap()
            .iter()
            .map(Air::<F>::name)
            .collect();

        assert_eq!(names, ["AesRound256Air", "SboxRomChiplet"]);
    }

    #[test]
    fn new_validates() {
        assert!(Aes256Chiplet::new(100, 256, 1).is_err());
        assert!(Aes256Chiplet::new(16, 7, 1).is_err());
        assert!(Aes256Chiplet::new(16, 16, 2).is_err());
        assert!(Aes256Chiplet::new(16, 16, 1).is_ok());
    }
}
