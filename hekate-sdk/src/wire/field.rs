// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::errors::Result;
use hekate_math::TowerField;

use super::wire_err;

const WORDS: usize = 4;

pub(crate) fn field_to_words<F: TowerField>(f: &F) -> [u64; WORDS] {
    const { assert!(F::BITS <= 64 * WORDS, "wire scalar holds at most GF(2^256)") };

    let bytes = f.to_bytes();

    let mut buf = [0u8; 8 * WORDS];
    buf[..bytes.len()].copy_from_slice(&bytes);

    let (chunks, _) = buf.as_chunks::<8>();

    core::array::from_fn(|w| u64::from_le_bytes(chunks[w]))
}

pub(crate) fn words_to_field<F: TowerField>(words: [u64; WORDS]) -> Result<F> {
    let mut buf = [0u8; 8 * WORDS];
    for (w, word) in words.iter().enumerate() {
        buf[8 * w..8 * (w + 1)].copy_from_slice(&word.to_le_bytes());
    }

    let width = F::ZERO.serialized_size();

    if buf[width..].iter().any(|&b| b != 0) {
        return Err(wire_err("field element wider than the field"));
    }

    F::deserialize(&buf[..width]).map_err(|_| wire_err("invalid field element bytes"))
}
