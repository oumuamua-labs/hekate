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

//! Hekate protocol bindings layered on top of
//! the generic Fiat-Shamir transcript.

use crate::errors;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use hekate_crypto::Hasher;
use hekate_crypto::transcript::Transcript;
use hekate_math::TowerField;

/// Absorbs a chiplet header (name, dims, Merkle root).
/// Prover and verifier must call this with identical
/// arguments in identical order or chiplet-phase
/// Fiat-Shamir challenges diverge.
pub fn absorb_chiplet_header<H: Hasher>(
    transcript: &mut Transcript<H>,
    name: &str,
    num_rows: usize,
    num_cols: usize,
    root: &[u8; 32],
) {
    transcript.append_message(b"chiplet_name", name.as_bytes());
    transcript.append_u64(b"chiplet_num_rows", num_rows as u64);
    transcript.append_u64(b"chiplet_num_cols", num_cols as u64);
    transcript.append_message(b"chiplet_root", root);
}

/// Binds LogUp `claimed_sums` into the
/// transcript before `alpha` / `r_zerocheck`
/// are drawn, so the ZeroCheck challenges
/// depend on the bus-sum target.
pub fn absorb_logup_claimed_sums<F, H>(transcript: &mut Transcript<H>, claimed_sums: &[(String, F)])
where
    F: TowerField,
    H: Hasher,
{
    transcript.append_u64(b"logup_claim_count", claimed_sums.len() as u64);

    for (bus_id, claim) in claimed_sums {
        transcript.append_message(b"logup_claim_bus_id", bus_id.as_bytes());
        transcript.append_field(b"logup_claim_sum", *claim);
    }
}

/// Binds LogUp `h_evals` into the
/// transcript after ZeroCheck completes;
/// downstream challenges (eta, rho, LDT queries)
/// must depend on these values.
pub fn absorb_logup_h_evals<F, H>(transcript: &mut Transcript<H>, h_evals: &[(String, F)])
where
    F: TowerField,
    H: Hasher,
{
    transcript.append_u64(b"logup_h_count", h_evals.len() as u64);

    for (bus_id, h_eval) in h_evals {
        transcript.append_message(b"logup_h_bus_id", bus_id.as_bytes());
        transcript.append_field(b"logup_h_eval", *h_eval);
    }
}

/// Binds lookup-bus heights before
/// any `r_bus` draw. Entries must be
/// in the same order prover and verifier
/// agree on (sorted by `bus_id`).
pub fn absorb_lookup_bus_heights<H: Hasher>(
    transcript: &mut Transcript<H>,
    entries: &[(String, u64)],
) {
    transcript.append_u64(b"lookup_bus_count", entries.len() as u64);

    for (bus_id, n_max) in entries {
        transcript.append_message(b"lookup_bus_id", bus_id.as_bytes());
        transcript.append_u64(b"lookup_bus_n_max", *n_max);
    }
}

/// Draws `num_vars` field challenges
/// forming one lookup bus's `r_bus` point.
pub fn challenge_r_bus<F, H>(
    transcript: &mut Transcript<H>,
    bus_id: &str,
    num_vars: usize,
) -> errors::Result<Vec<F>>
where
    F: TowerField,
    H: Hasher,
{
    transcript.append_message(b"lookup_bus_id", bus_id.as_bytes());

    let mut coords = Vec::with_capacity(num_vars);
    for _ in 0..num_vars {
        coords.push(transcript.challenge_field::<F>(b"r_bus")?);
    }

    Ok(coords)
}

/// Absorbs all lookup-bus heights
/// and draws one `r_bus` per bus_id.
pub fn draw_lookup_bus_points<F, H>(
    transcript: &mut Transcript<H>,
    entries: &[(String, u64)],
) -> errors::Result<BTreeMap<String, Vec<F>>>
where
    F: TowerField,
    H: Hasher,
{
    for (_, n_max) in entries {
        if *n_max == 0 || !n_max.is_power_of_two() {
            return Err(errors::Error::Protocol {
                protocol: "transcript",
                message: "lookup bus N_max must be a non-zero power of two",
            });
        }
    }

    absorb_lookup_bus_heights(transcript, entries);

    let mut points = BTreeMap::new();
    for (bus_id, n_max) in entries {
        let num_vars = n_max.trailing_zeros() as usize;
        points.insert(
            bus_id.clone(),
            challenge_r_bus::<F, H>(transcript, bus_id, num_vars)?,
        );
    }

    Ok(points)
}
