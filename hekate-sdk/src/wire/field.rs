// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::errors::Result;
use hekate_math::TowerField;

use super::wire_err;

pub(crate) fn field_to_lo_hi<F: TowerField>(f: &F) -> (u64, u64) {
    const { assert!(F::BITS <= 128, "wire scalar holds at most GF(2^128)") };

    let bytes = f.to_bytes();

    let mut buf = [0u8; 16];
    buf[..bytes.len()].copy_from_slice(&bytes);

    (
        u64::from_le_bytes(buf[0..8].try_into().unwrap()),
        u64::from_le_bytes(buf[8..16].try_into().unwrap()),
    )
}

pub(crate) fn lo_hi_to_field<F: TowerField>(lo: u64, hi: u64) -> Result<F> {
    let mut buf = [0u8; 16];
    buf[0..8].copy_from_slice(&lo.to_le_bytes());
    buf[8..16].copy_from_slice(&hi.to_le_bytes());

    F::deserialize(&buf).map_err(|_| wire_err("invalid field element bytes"))
}
