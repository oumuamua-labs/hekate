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

//! AES S-box ROM Chiplet.
//!
//! Algebraic constraint:
//! S(x) = A(x^{-1}) + 0x63. INV = x^{-1} is
//! bit-decomposed; the FIPS 197 affine
//! transform A is applied via its column
//! vectors [0x1F, 0x3E, 0x7C, 0xF8, 0xF1, 0xE3, 0xC7, 0x8F].

use super::{SBOX, SBOX_IN_LABELS, SBOX_OUT_LABELS};
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use errors::Error;
use hekate_core::errors;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate_math::{Bit, Block128, Block64, Block8, TowerField};
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::constraint::ConstraintAst;
use hekate_program::define_columns;
use hekate_program::expander::VirtualExpander;
use hekate_program::permutation::{PermutationCheckSpec, Source};
use hekate_program::Air;
use once_cell::race::OnceBox;

/// FIPS 197 §5.1.1 affine transform columns.
/// Column k = Σ_j A[j][k] * 2^j where A is
/// the SubBytes affine matrix.
#[rustfmt::skip]
pub(crate) const AFFINE_COLS: [u8; 8] = [
    0x1F, 0x3E, 0x7C, 0xF8,
    0xF1, 0xE3, 0xC7, 0x8F,
];

// Physical column indices.
// Distinct from virtual SboxRomColumns
// (177 cols after bit-unpacking).
const PHYS_INV: usize = 0;
const PHYS_INPUT: usize = 2;
const PHYS_OUTPUT: usize = 18;
const PHYS_Z: usize = 34;
const PHYS_SELECTOR: usize = 50;
const PHYS_NUM_COLS: usize = 51;

// Virtual layout:
// constraints reference these.
define_columns! {
    pub SboxRomColumns {
        INV_BITS: [Bit; 128],
        INPUT: [B8; 16],
        OUTPUT: [B8; 16],
        Z: [Bit; 16],
        SELECTOR: Bit,
    }
}

#[derive(Clone, Debug)]
pub struct SboxRomChiplet {
    #[allow(dead_code)]
    pub num_rows: usize,
}

impl SboxRomChiplet {
    pub const BUS_ID: &'static str = "aes_sbox";

    pub fn new(num_rows: usize) -> errors::Result<Self> {
        if !num_rows.is_power_of_two() {
            return Err(Error::Protocol {
                protocol: "aes_sbox_rom",
                message: "ROM size must be power of 2",
            });
        }

        Ok(Self { num_rows })
    }

    pub fn linking_spec() -> PermutationCheckSpec {
        let mut sources = Vec::with_capacity(32);
        for i in 0..16 {
            sources.push((Source::Column(SboxRomColumns::INPUT + i), SBOX_IN_LABELS[i]));
            sources.push((
                Source::Column(SboxRomColumns::OUTPUT + i),
                SBOX_OUT_LABELS[i],
            ));
        }

        PermutationCheckSpec::new(sources, Some(SboxRomColumns::SELECTOR)).with_clock_waiver(
            "see hekate-chiplets/src/aes/sbox_rom.rs: AES<>SboxRom internal; \
             phantom blocks caught at link+key v3",
        )
    }
}

impl<F: TowerField> Air<F> for SboxRomChiplet {
    fn name(&self) -> String {
        "SboxRomChiplet".to_string()
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: OnceBox<Vec<ColumnType>> = OnceBox::new();
        LAYOUT.get_or_init(|| {
            let mut cols = Vec::with_capacity(PHYS_NUM_COLS);
            cols.extend(vec![ColumnType::B64; 2]);
            cols.extend(vec![ColumnType::B8; 32]);
            cols.extend(vec![ColumnType::Bit; 17]);

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
                    .expand_bits(2, ColumnType::B64)
                    .pass_through(16, ColumnType::B8)
                    .pass_through(16, ColumnType::B8)
                    .control_bits(17)
                    .build()
                    .expect("SboxRomChiplet expander"),
            )
        }))
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();

        let sel = cs.col(SboxRomColumns::SELECTOR);
        cs.assert_boolean(sel);

        let one = cs.one();
        let affine_const = cs.constant(F::from(0x63u8));

        for j in 0..16 {
            let input = cs.col(SboxRomColumns::INPUT + j);
            let output = cs.col(SboxRomColumns::OUTPUT + j);
            let z = cs.col(SboxRomColumns::Z + j);

            cs.assert_boolean(z);

            let bit_base = SboxRomColumns::INV_BITS + j * 8;
            let bits: [_; 8] = core::array::from_fn(|k| {
                let b = cs.col(bit_base + k);
                cs.assert_boolean(b);

                b
            });

            // INV = Σ bit_k * 2^k
            let inv_terms: Vec<_> = (0..8)
                .map(|k| cs.scale(F::from(1u8 << k), bits[k]))
                .collect();
            let inv_sum = cs.sum(&inv_terms);

            // INPUT * INV + Z = 1 (gated)
            cs.assert_zero_when(sel, input * inv_sum + z + one);

            // Z * INPUT = 0
            cs.constrain(z * input);

            // Z * INV = 0
            cs.constrain(z * inv_sum);

            // OUTPUT = AFFINE(INV_bits) + 0x63 (gated)
            let affine_terms: Vec<_> = (0..8)
                .map(|k| cs.scale(F::from(AFFINE_COLS[k]), bits[k]))
                .collect();
            let affine_sum = cs.sum(&affine_terms);

            cs.assert_zero_when(sel, output + affine_const + affine_sum);
        }

        // Z and INV bits load-bearing only on sel=1 rows
        let not_sel = one + sel;
        for j in 0..16 {
            cs.assert_zero_when(not_sel, cs.col(SboxRomColumns::Z + j));

            let inv_byte = cs.sum(
                &(0..8)
                    .map(|k| {
                        cs.scale(
                            F::from(1u8 << k),
                            cs.col(SboxRomColumns::INV_BITS + j * 8 + k),
                        )
                    })
                    .collect::<Vec<_>>(),
            );

            cs.assert_zero_when(not_sel, inv_byte);
        }

        cs.build()
    }
}

/// One round's 16 S-box evaluations.
pub struct SboxRound {
    pub inputs: [u8; 16],
    pub outputs: [u8; 16],
}

pub fn generate_sbox_rom_trace(
    rounds: &[SboxRound],
    num_rows: usize,
) -> errors::Result<ColumnTrace> {
    if !num_rows.is_power_of_two() {
        return Err(Error::Protocol {
            protocol: "aes_sbox_rom",
            message: "trace size must be power of 2",
        });
    }

    if rounds.len() > num_rows {
        return Err(Error::Protocol {
            protocol: "aes_sbox_rom",
            message: "too many rounds for trace size",
        });
    }

    for round in rounds {
        for j in 0..16 {
            if round.outputs[j] != SBOX[round.inputs[j] as usize] {
                return Err(Error::Protocol {
                    protocol: "aes_sbox_rom",
                    message: "entry does not match FIPS 197 S-box",
                });
            }
        }
    }

    let num_vars = num_rows.trailing_zeros() as usize;

    let chiplet = SboxRomChiplet { num_rows };
    let layout = Air::<Block128>::column_layout(&chiplet);

    let mut tb = TraceBuilder::new(layout, num_vars)?;

    for (row, round) in rounds.iter().enumerate() {
        tb.set_b8_array(PHYS_INPUT, row, &round.inputs.map(Block8))?;
        tb.set_b8_array(PHYS_OUTPUT, row, &round.outputs.map(Block8))?;

        let mut inv_bytes = [0u8; 16];
        for (j, inv) in inv_bytes.iter_mut().enumerate() {
            *inv = gf256_inv(round.inputs[j]);

            if round.inputs[j] == 0 {
                tb.set_bit(PHYS_Z + j, row, Bit::ONE)?;
            }
        }

        // Pack 8 inverse bytes per B64 column
        let lo = u64::from_le_bytes(inv_bytes[..8].try_into().unwrap());
        let hi = u64::from_le_bytes(inv_bytes[8..].try_into().unwrap());

        tb.set_b64(PHYS_INV, row, Block64(lo))?;
        tb.set_b64(PHYS_INV + 1, row, Block64(hi))?;
    }

    tb.fill_selector(PHYS_SELECTOR, rounds.len())?;

    Ok(tb.build())
}

/// x^{-1} in GF(2^8) via x^{254}.
/// Returns 0 for x = 0 (FIPS 197 convention).
pub(crate) fn gf256_inv(x: u8) -> u8 {
    if x == 0 {
        return 0;
    }

    let b = Block8(x);
    let b2 = b * b;
    let b4 = b2 * b2;
    let b8 = b4 * b4;
    let b16 = b8 * b8;
    let b32 = b16 * b16;
    let b64 = b32 * b32;
    let b128 = b64 * b64;

    // x^{254} = x^{2+4+8+16+32+64+128}
    (b2 * b4 * b8 * b16 * b32 * b64 * b128).0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chiplets::aes::AesRound128Air;
    use hekate_core::trace::Trace;
    use hekate_math::{Bit, Block128};

    fn identity_round() -> SboxRound {
        let inputs: [u8; 16] = core::array::from_fn(|i| i as u8);
        let outputs: [u8; 16] = core::array::from_fn(|i| SBOX[i]);

        SboxRound { inputs, outputs }
    }

    #[test]
    fn sbox_rom_column_count() {
        // Virtual layout
        assert_eq!(SboxRomColumns::NUM_COLUMNS, 177);
        assert_eq!(SboxRomColumns::INV_BITS, 0);
        assert_eq!(SboxRomColumns::INPUT, 128);
        assert_eq!(SboxRomColumns::OUTPUT, 144);
        assert_eq!(SboxRomColumns::Z, 160);
        assert_eq!(SboxRomColumns::SELECTOR, 176);

        // Physical layout
        assert_eq!(PHYS_NUM_COLS, 51);
    }

    #[test]
    fn sbox_rom_linking_spec_structure() {
        let spec = SboxRomChiplet::linking_spec();
        assert_eq!(spec.num_sources(), 32);
        assert!(spec.has_selector());
        assert_eq!(spec.selector, Some(SboxRomColumns::SELECTOR));
    }

    #[test]
    fn sbox_table_fips197() {
        // FIPS 197 Appendix B known values.
        assert_eq!(SBOX[0x00], 0x63);
        assert_eq!(SBOX[0x01], 0x7c);
        assert_eq!(SBOX[0x53], 0xed);
        assert_eq!(SBOX[0xFF], 0x16);

        // S(0x00) = 0x63
        // (affine of 0^{-1} = 0 by convention)
        assert_eq!(SBOX[0x00], 0x63);

        // S(0x01) = 0x7c
        // (1^{-1} = 1, then affine)
        assert_eq!(SBOX[0x01], 0x7c);
    }

    #[test]
    fn sbox_table_is_permutation() {
        let mut seen = [false; 256];
        for &out in &SBOX {
            assert!(!seen[out as usize], "duplicate output: 0x{out:02x}");
            seen[out as usize] = true;
        }
    }

    #[test]
    fn sbox_trace_single_round() {
        let round = identity_round();
        let trace = generate_sbox_rom_trace(&[round], 4).unwrap();

        assert_eq!(trace.num_cols(), PHYS_NUM_COLS);

        let sel = trace.columns[PHYS_SELECTOR].as_bit_slice().unwrap();
        assert_eq!(sel[0], Bit::ONE);
        assert_eq!(sel[1], Bit::ZERO);
    }

    #[test]
    fn sbox_trace_rejects_bad_entry() {
        let bad = SboxRound {
            inputs: [0u8; 16],
            outputs: [0u8; 16],
        };
        assert!(generate_sbox_rom_trace(&[bad], 4).is_err());
    }

    #[test]
    fn rom_bus_labels_match_aes_chiplet() {
        let rom = SboxRomChiplet::new(16).unwrap();
        let rom_checks: Vec<_> = Air::<Block128>::permutation_checks(&rom);
        let aes_checks = AesRound128Air::sbox_specs();

        assert_eq!(rom_checks.len(), 1);
        assert_eq!(aes_checks.len(), 1);

        assert_eq!(rom_checks[0].0, aes_checks[0].0, "bus ID mismatch");
        assert_eq!(
            rom_checks[0].1.sources.len(),
            aes_checks[0].1.sources.len(),
            "source count mismatch"
        );

        for (r, a) in rom_checks[0]
            .1
            .sources
            .iter()
            .zip(aes_checks[0].1.sources.iter())
        {
            assert_eq!(r.1, a.1, "challenge label mismatch");
        }
    }

    #[test]
    fn gf256_inv_all_entries() {
        assert_eq!(gf256_inv(0), 0);
        assert_eq!(gf256_inv(1), 1);

        for x in 1..=255u8 {
            let inv = gf256_inv(x);

            assert_ne!(inv, 0, "inverse of 0x{x:02X} must be nonzero");
            assert_eq!(
                Block8(x) * Block8(inv),
                Block8(1),
                "0x{x:02X} * 0x{inv:02X} != 1"
            );
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn affine_cols_reproduce_sbox() {
        for x in 0..=255u8 {
            let inv = gf256_inv(x);

            let mut affine_val = 0x63u8;
            for k in 0..8 {
                if (inv >> k) & 1 == 1 {
                    affine_val ^= AFFINE_COLS[k];
                }
            }

            assert_eq!(
                affine_val, SBOX[x as usize],
                "algebraic S-box mismatch at 0x{x:02X}"
            );
        }
    }

    #[test]
    fn trace_fills_inv_and_z() {
        let round = identity_round();
        let trace = generate_sbox_rom_trace(&[round], 4).unwrap();

        for j in 0..16 {
            let input = j as u8;
            let expected_inv = gf256_inv(input);
            let expected_z = if input == 0 { Bit::ONE } else { Bit::ZERO };

            let z = trace.columns[PHYS_Z + j].as_bit_slice().unwrap()[0];
            assert_eq!(z, expected_z, "Z mismatch at byte {j}");

            let b64_col = j / 8;
            let byte_pos = j % 8;
            let packed = trace.columns[PHYS_INV + b64_col].as_b64_slice().unwrap()[0];
            let inv_byte = (packed.to_tower().0 >> (byte_pos * 8)) as u8;

            assert_eq!(inv_byte, expected_inv, "INV mismatch at byte {j}");
        }
    }
}
