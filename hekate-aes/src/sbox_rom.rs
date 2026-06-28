// SPDX-License-Identifier: Apache-2.0
// This file is part of the hekate project.
// Copyright (C) 2026 Andrei Kochergin <andrei@oumuamua.dev>
// Copyright (C) 2026 Oumuamua Labs <info@oumuamua.dev>.
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

use super::{SBOX_IN_LABELS, SBOX_OUT_LABELS};
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use errors::Error;
use hekate_core::errors;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate_math::{Bit, Block8, Block64, Block128, TowerField};
use hekate_program::Air;
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::define_columns;
use hekate_program::expander::VirtualExpander;
use hekate_program::permutation::{PermutationCheckSpec, Source};
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
            if round.outputs[j] != ct_sbox(round.inputs[j]) {
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

            let z = if round.inputs[j] == 0 {
                Bit::ONE
            } else {
                Bit::ZERO
            };
            tb.set_bit(PHYS_Z + j, row, z)?;
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

/// x^{-1} in GF(2^8) via x^{254}, constant-time.
/// 0^{254} = 0 yields the FIPS 197 convention
/// with no data-dependent special case.
pub(crate) fn gf256_inv(x: u8) -> u8 {
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

/// Constant-time AES S-box:
/// a `SBOX[x]` table lookup would leak
/// the key via cache timing during proving.
pub(crate) fn ct_sbox(x: u8) -> u8 {
    aes_affine(gf256_inv(x))
}

/// AES SubBytes affine map (FIPS 197 §5.1.1).
pub(crate) fn aes_affine(b: u8) -> u8 {
    b ^ b.rotate_left(1) ^ b.rotate_left(2) ^ b.rotate_left(3) ^ b.rotate_left(4) ^ 0x63
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aes128::AesRound128Air;
    use hekate_core::trace::Trace;
    use hekate_math::{Bit, Block128};

    // FIPS 197 Table 4, oracle to cross-check ct_sbox
    #[rustfmt::skip]
    const SBOX: [u8; 256] = [
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
        0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
        0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc,
        0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a,
        0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
        0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b,
        0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85,
        0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
        0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17,
        0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88,
        0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
        0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9,
        0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6,
        0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
        0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94,
        0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68,
        0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
    ];

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
    fn ct_sbox_matches_fips197_table() {
        for x in 0..=255u8 {
            assert_eq!(
                ct_sbox(x),
                SBOX[x as usize],
                "ct_sbox mismatch at 0x{x:02X}"
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
