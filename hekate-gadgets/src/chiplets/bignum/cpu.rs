// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec::Vec;
use hekate_core::errors;
use hekate_core::trace::{ColumnType, TraceBuilder};
use hekate_math::{Block32, Block128};
use hekate_program::circuit::{Circuit, Col, ColRange};

use crate::chiplets::bignum::modexp::{self, LIMBS32, Modexp};

type F = Block128;

/// One modexp request on the host table:
/// the modulus, the base and the result,
/// in that physical order from `phys_base`.
pub struct CpuModexpBlock {
    pub modulus: ColRange,
    pub base: ColRange,
    pub result: ColRange,

    phys_base: usize,
}

impl CpuModexpBlock {
    pub const COLUMNS: usize = 3 * LIMBS32;

    pub const MODULUS: usize = 0;
    pub const BASE: usize = Self::MODULUS + LIMBS32;
    pub const RESULT: usize = Self::BASE + LIMBS32;

    pub fn layout() -> [ColumnType; Self::COLUMNS] {
        [ColumnType::B32; Self::COLUMNS]
    }

    /// `phys_base` is the block's first physical
    /// column in the host layout; the caller's
    /// declarations before this call determine it.
    pub fn declare(cx: &mut Circuit<F>, phys_base: usize) -> Self {
        let modulus = cx.columns(LIMBS32, ColumnType::B32);
        let base = cx.columns(LIMBS32, ColumnType::B32);
        let result = cx.columns(LIMBS32, ColumnType::B32);

        Self {
            modulus,
            base,
            result,
            phys_base,
        }
    }

    /// # Errors
    /// The service schema does not match the declared columns.
    pub fn connect(&self, cx: &mut Circuit<F>, selector: Col) -> errors::Result<()> {
        let values: Vec<Col> = self
            .modulus
            .iter()
            .chain(self.base.iter())
            .chain(self.result.iter())
            .collect();

        cx.call(&modexp::service(), &values, selector)
    }

    /// # Errors
    /// `row` outside the trace or a column type mismatch.
    pub fn write(&self, tb: &mut TraceBuilder, row: usize, modexp: &Modexp) -> errors::Result<()> {
        let groups = [
            (Self::MODULUS, modexp.modulus()),
            (Self::BASE, modexp.base()),
            (Self::RESULT, modexp.result()),
        ];

        for (offset, limbs) in groups {
            for (j, &limb) in limbs.iter().enumerate() {
                tb.set_b32(self.phys_base + offset + j, row, Block32::from(limb))?;
            }
        }

        Ok(())
    }
}
