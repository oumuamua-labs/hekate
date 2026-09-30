// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod codec;
pub mod gadgets;
pub mod high_bits;
pub mod kem_select;
pub mod mldsa;
pub mod mlkem;
pub mod ntt;
pub mod poly_arith;
pub mod sampler;
pub mod utils;
pub mod wiring;

pub(crate) mod ctrl;

#[cfg(test)]
mod census;
