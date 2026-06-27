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

//! Twiddle Factor ROM Chiplet.
//!
//! Validates NTT twiddle factors (roots
//! of unity mod q) via Grand Product Argument.
//! Each entry maps (layer, butterfly_idx)
//! to the correct twiddle factor w.
//!
//! Links to NTT chiplet's `ntt_twiddle` bus.
//! The NTT chiplet uses w as a witness column;
//! this ROM proves the witness matches the
//! precomputed twiddle table.
//!
//! # Twiddle Factor Generation
//!
//! For NTT-256 over Z_q:
//! - Primitive 256th root of unity g mod q
//! - Layer l (0..7), butterfly b (0..127):
//!   w = g^(bit_reverse(b, 7-l) * 2^l) mod q
//!
//! The twiddle table has 1024 entries
//! (8 layers × 128 butterflies per layer).

use super::ntt::NttChiplet;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate_math::{Bit, Block32, TowerField};
use hekate_program::Air;
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::define_columns;
use hekate_program::permutation::{PermutationCheckSpec, REQUEST_IDX_LABEL, Source};

// Sentinel layer for basemul MulOnly entries;
// must exceed any real NTT layer (0..=7)
// and normalization (8).
pub const BASEMUL_LAYER_MARKER: u32 = 9;

/// Bus ID for w-side binding:
/// twiddle ROM MulOnly <> ctrl w-side RAM reads.
pub const TWIDDLE_W_BINDING_BUS_ID: &str = "twiddle_w_binding";

define_columns! {
    pub TwiddleRomColumns {
        LAYER: B32,
        BUTTERFLY_IDX: B32,
        W_VALUE: B32,
        SELECTOR: Bit,
        MULONLY_SELECTOR: Bit,

        // Partner ctrl row index for
        // the twiddle_w_binding bus,
        // gated by MULONLY_SELECTOR.
        REQUEST_IDX_TR: B32,

        LAYER_INV: B32,
    }
}

/// Twiddle Factor ROM Chiplet.
///
/// Stores precomputed (layer, butterfly_idx, w)
/// tuples for NTT twiddle validation.
/// Bus ID matches NTT chiplet's `ntt_twiddle`.
#[derive(Clone, Debug)]
pub struct TwiddleRomChiplet {
    pub modulus: u32,
    pub num_rows: usize,
}

impl TwiddleRomChiplet {
    pub fn new(modulus: u32, num_rows: usize) -> Self {
        assert!(num_rows.is_power_of_two());
        Self { modulus, num_rows }
    }

    /// Linking specification matching
    /// NTT chiplet's `ntt_twiddle` bus.
    pub fn linking_spec() -> PermutationCheckSpec {
        PermutationCheckSpec::new_lookup(
            vec![
                (
                    Source::Column(TwiddleRomColumns::LAYER),
                    b"kappa_tw_layer" as &[u8],
                ),
                (
                    Source::Column(TwiddleRomColumns::BUTTERFLY_IDX),
                    b"kappa_tw_bfly" as &[u8],
                ),
                (
                    Source::Column(TwiddleRomColumns::W_VALUE),
                    b"kappa_tw_w" as &[u8],
                ),
            ],
            Some(TwiddleRomColumns::SELECTOR),
        )
    }

    /// W-side binding bus:
    /// MulOnly entries matched
    /// against ctrl w-side RAM reads.
    pub fn w_binding_linking_spec() -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(TwiddleRomColumns::BUTTERFLY_IDX),
                    b"kappa_wb_bfly" as &[u8],
                ),
                (
                    Source::Column(TwiddleRomColumns::W_VALUE),
                    b"kappa_wb_w" as &[u8],
                ),
                (
                    Source::Column(TwiddleRomColumns::REQUEST_IDX_TR),
                    REQUEST_IDX_LABEL,
                ),
            ],
            Some(TwiddleRomColumns::MULONLY_SELECTOR),
        )
    }
}

impl<F: TowerField> Air<F> for TwiddleRomChiplet {
    fn name(&self) -> String {
        "TwiddleRomChiplet".to_string()
    }

    fn num_columns(&self) -> usize {
        TwiddleRomColumns::NUM_COLUMNS
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: once_cell::race::OnceBox<Vec<ColumnType>> = once_cell::race::OnceBox::new();

        LAYOUT.get_or_init(|| Box::new(TwiddleRomColumns::build_layout()))
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![
            (NttChiplet::TWIDDLE_BUS_ID.into(), Self::linking_spec()),
            (
                TWIDDLE_W_BINDING_BUS_ID.into(),
                Self::w_binding_linking_spec(),
            ),
        ]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        let one = cs.one();

        cs.assert_boolean(cs.col(TwiddleRomColumns::SELECTOR));
        cs.assert_boolean(cs.col(TwiddleRomColumns::MULONLY_SELECTOR));

        // MULONLY_SELECTOR implies SELECTOR
        let sel = cs.col(TwiddleRomColumns::SELECTOR);
        let mulonly = cs.col(TwiddleRomColumns::MULONLY_SELECTOR);
        let layer = cs.col(TwiddleRomColumns::LAYER);
        let layer_inv = cs.col(TwiddleRomColumns::LAYER_INV);

        let marker = cs.constant(F::from(BASEMUL_LAYER_MARKER));

        cs.constrain(mulonly * (one + sel));

        // Pin MULONLY_SELECTOR <> (LAYER == BASEMUL_LAYER_MARKER);
        // LAYER_INV pinned to 0 when mulonly=1.
        cs.constrain(mulonly * (layer + marker));
        cs.constrain((one + mulonly) * ((layer + marker) * layer_inv + one));
        cs.constrain(mulonly * layer_inv);

        cs.build()
    }
}

// =================================================================
// Twiddle Factor Computation
// =================================================================

/// A twiddle table entry.
///
/// `active = false` produces a sel=0 padding row
/// to preserve positional alignment with NTT
/// chiplet's interleaved FlowCompanion rows.
#[derive(Clone, Debug)]
pub struct TwiddleEntry {
    pub layer: u32,
    pub butterfly_idx: u32,
    pub w: u32,
    pub is_mulonly: bool,
    pub active: bool,

    /// Ctrl row index of the
    /// partnered W-bind emit.
    pub request_idx_tr: u32,
}

/// Compute the twiddle factor table for
/// NTT-256 over Z_q.
///
/// Returns 8*128 = 1024 entries:
/// one per (layer, butterfly) pair.
///
/// `modulus`: the prime q (3329 or 8380417)
/// `root`: primitive 256th root of unity mod q
///
/// Twiddle for layer l, butterfly b:
///   w = root^(bit_reverse(b, 7-l) << l) mod q
///
/// For Cooley-Tukey bit-reversed input,
/// natural output (FIPS 203/204 ordering).
pub fn compute_twiddle_table(modulus: u32, root: u32) -> Vec<TwiddleEntry> {
    let n = 256usize;
    let log_n = 8usize;

    // Precompute powers of root
    let mut powers = Vec::with_capacity(n);
    let mut p = 1u64;

    for _ in 0..n {
        powers.push(p as u32);
        p = (p * root as u64) % modulus as u64;
    }

    let mut entries = Vec::with_capacity(log_n * (n / 2));

    for layer in 0..log_n {
        let half_size = 1 << layer;
        let num_groups = n / (2 * half_size);

        for group in 0..num_groups {
            for j in 0..half_size {
                let butterfly_idx = group * half_size + j;

                // Twiddle index for CT bit-reversed:
                // w = root^(j * n / (2 * half_size))
                let exp = j * (n / (2 * half_size));
                entries.push(TwiddleEntry {
                    layer: layer as u32,
                    butterfly_idx: butterfly_idx as u32,
                    w: powers[exp],
                    is_mulonly: false,
                    active: true,
                    request_idx_tr: 0,
                });
            }
        }
    }

    entries
}

/// Generate the twiddle ROM trace.
///
/// `entries`: precomputed twiddle table
/// `num_rows`: trace height (power of 2, ≥ entries.len())
pub fn generate_twiddle_rom_trace(
    entries: &[TwiddleEntry],
    num_rows: usize,
) -> hekate_core::errors::Result<ColumnTrace> {
    assert!(num_rows.is_power_of_two());
    assert!(entries.len() <= num_rows);

    let num_vars = num_rows.trailing_zeros() as usize;

    let mut tb = TraceBuilder::new(&TwiddleRomColumns::build_layout(), num_vars)?;

    let marker = Block32::from(BASEMUL_LAYER_MARKER);
    let marker_inv = marker.invert();

    for (i, entry) in entries.iter().enumerate() {
        if !entry.active {
            tb.set_b32(TwiddleRomColumns::LAYER_INV, i, marker_inv)?;
            continue;
        }

        tb.set_b32(TwiddleRomColumns::LAYER, i, Block32::from(entry.layer))?;
        tb.set_b32(
            TwiddleRomColumns::BUTTERFLY_IDX,
            i,
            Block32::from(entry.butterfly_idx),
        )?;
        tb.set_b32(TwiddleRomColumns::W_VALUE, i, Block32::from(entry.w))?;
        tb.set_bit(TwiddleRomColumns::SELECTOR, i, Bit::ONE)?;

        if entry.is_mulonly {
            tb.set_bit(TwiddleRomColumns::MULONLY_SELECTOR, i, Bit::ONE)?;
            tb.set_b32(
                TwiddleRomColumns::REQUEST_IDX_TR,
                i,
                Block32::from(entry.request_idx_tr),
            )?;
            tb.set_b32(TwiddleRomColumns::LAYER_INV, i, Block32::ZERO)?;
        } else {
            assert_ne!(
                entry.layer, BASEMUL_LAYER_MARKER,
                "non-mulonly TwiddleEntry must not collide with BASEMUL_LAYER_MARKER",
            );

            let layer_plus_marker = Block32::from(entry.layer) + marker;
            tb.set_b32(TwiddleRomColumns::LAYER_INV, i, layer_plus_marker.invert())?;
        }
    }

    for i in entries.len()..num_rows {
        tb.set_b32(TwiddleRomColumns::LAYER_INV, i, marker_inv)?;
    }

    Ok(tb.build())
}

/// Compute modular exponentiation:
/// base^exp mod modulus.
pub fn mod_pow(base: u32, exp: u32, modulus: u32) -> u32 {
    let mut result = 1u64;
    let mut b = base as u64 % modulus as u64;
    let mut e = exp;

    let m = modulus as u64;

    while e > 0 {
        if e & 1 == 1 {
            result = result * b % m;
        }

        b = b * b % m;
        e >>= 1;
    }

    result as u32
}

/// Find a primitive n-th root of unity mod q.
///
/// For q prime with q ≡ 1 (mod n):
/// g = generator^((q-1)/n) mod q
/// where generator is a primitive root of q.
///
/// Returns None if q ≢ 1 (mod n).
pub fn find_primitive_root(modulus: u32, n: u32) -> Option<u32> {
    if !(modulus - 1).is_multiple_of(n) {
        return None;
    }

    let exp = (modulus - 1) / n;

    // Try small generators until we find
    // one whose n-th power ≠ 1 for all
    // proper divisors of n.
    for g in 2..modulus {
        let root = mod_pow(g, exp, modulus);
        if root == 1 {
            continue;
        }

        // Verify root^n = 1
        if mod_pow(root, n, modulus) != 1 {
            continue;
        }

        // Verify root^(n/p) ≠ 1
        // for each prime factor p of n.
        // For n=256=2^8, the only prime
        // factor is 2.
        if mod_pow(root, n / 2, modulus) == 1 {
            continue;
        }

        return Some(root);
    }

    None
}

// =================================================================
// Tests
// =================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use hekate_math::Bit;

    #[test]
    fn twiddle_rom_column_count() {
        assert_eq!(TwiddleRomColumns::NUM_COLUMNS, 7);
    }

    #[test]
    fn twiddle_rom_bus_labels_match_ntt() {
        // The challenge labels in
        // TwiddleRomChiplet::linking_spec must
        // match NttChiplet::twiddle_linking_spec.
        let ntt = NttChiplet::new(3329, 1024);
        let ntt_spec = ntt.twiddle_linking_spec();
        let rom_spec = TwiddleRomChiplet::linking_spec();

        assert_eq!(ntt_spec.sources.len(), rom_spec.sources.len(),);

        for (n, r) in ntt_spec.sources.iter().zip(rom_spec.sources.iter()) {
            assert_eq!(n.1, r.1, "challenge label mismatch");
        }
    }

    #[test]
    fn find_root_q3329() {
        // q=3329:
        // 3329-1 = 3328 = 13*256,
        // so 256 | (q-1).
        let root = find_primitive_root(3329, 256);
        assert!(root.is_some());

        let g = root.unwrap();
        // g^256 = 1 mod q
        assert_eq!(mod_pow(g, 256, 3329), 1);
        // g^128 ≠ 1 mod q (primitive)
        assert_ne!(mod_pow(g, 128, 3329), 1);
    }

    #[test]
    fn find_root_q8380417() {
        // q=8380417:
        // 8380417-1 = 8380416 = 2^23 * 998.
        // 256 = 2^8 divides 2^23, so 256 | (q-1).
        let root = find_primitive_root(8380417, 256);
        assert!(root.is_some());

        let g = root.unwrap();
        assert_eq!(mod_pow(g, 256, 8380417), 1);
        assert_ne!(mod_pow(g, 128, 8380417), 1);
    }

    #[test]
    fn twiddle_table_q3329() {
        let root = find_primitive_root(3329, 256).unwrap();
        let table = compute_twiddle_table(3329, root);

        // 8 layers × 128 butterflies = 1024 entries
        assert_eq!(table.len(), 1024);

        // Layer 0:
        // 128 butterflies, twiddle = root^0 = 1 for all
        // (because half_size=1, exp = j * 128, and j=0 only)
        assert_eq!(table[0].layer, 0);
        assert_eq!(table[0].w, 1);

        // All twiddle values must be in [1, q-1]
        for entry in &table {
            assert!(entry.w > 0 && entry.w < 3329);
        }
    }

    #[test]
    fn twiddle_trace_generation() {
        let root = find_primitive_root(3329, 256).unwrap();
        let table = compute_twiddle_table(3329, root);

        let trace = generate_twiddle_rom_trace(&table, 1024).unwrap();

        // Check first entry
        let layer = trace.columns[TwiddleRomColumns::LAYER]
            .as_b32_slice()
            .unwrap();
        assert_eq!(layer[0].to_tower(), Block32::from(0u32),);

        // Last active row
        let sel = trace.columns[TwiddleRomColumns::SELECTOR]
            .as_bit_slice()
            .unwrap();
        assert_eq!(sel[0], Bit::ONE);
        assert_eq!(sel[1023], Bit::ONE);
        // No padding since 1024 entries = 1024 rows
    }

    #[test]
    fn twiddle_trace_with_padding() {
        let entries = vec![
            TwiddleEntry {
                layer: 0,
                butterfly_idx: 0,
                w: 1,
                is_mulonly: false,
                active: true,
                request_idx_tr: 0,
            },
            TwiddleEntry {
                layer: BASEMUL_LAYER_MARKER,
                butterfly_idx: 1,
                w: 17,
                is_mulonly: true,
                active: true,
                request_idx_tr: 0,
            },
        ];

        let trace = generate_twiddle_rom_trace(&entries, 4).unwrap();

        let sel = trace.columns[TwiddleRomColumns::SELECTOR]
            .as_bit_slice()
            .unwrap();

        assert_eq!(sel[0], Bit::ONE);
        assert_eq!(sel[1], Bit::ONE);
        assert_eq!(sel[2], Bit::ZERO); // padding
        assert_eq!(sel[3], Bit::ZERO);
    }
}
