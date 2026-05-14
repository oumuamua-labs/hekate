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

use crate::flat_matches_canonical;
use alloc::vec::Vec;
use hekate_core::errors;
use hekate_core::poly::UnivariatePoly;
use hekate_core::proofs::SumcheckProof;
use hekate_crypto::Hasher;
use hekate_crypto::transcript::Transcript;
use hekate_math::{Flat, HardwareField};

pub type SumcheckVerifyResult<F> = errors::Result<Option<(Vec<Flat<F>>, Flat<F>)>>;

/// Verifies a Sumcheck protocol
/// execution over a Boolean hypercube.
///
/// The Sumcheck protocol reduces the claim
/// that a multi-variate polynomial sums to
/// a specific value over the Boolean hypercube
/// $\{0, 1\}^v$ into a single claim about the
/// polynomial's evaluation at a randomly
/// chosen point $r \in \mathbb{F}^v$.
///
/// # Security & Protocol Mechanics
/// 1. **Degree Enforcement:** Strictly validates
///    that each round polynomial provides exactly
///    `degree + 1` evaluations. This is a critical
///    security boundary to prevent degree-inflation
///    attacks where a malicious prover might attempt
///    to hide forged claims in higher-degree terms.
/// 2. **Round Consistency:** For each round $i$,
///    verifies the consistency equation:
///    $g_i(0) + g_i(1) = g_{i-1}(r_{i-1})$.
/// 3. **Fiat-Shamir Binding:** Absorbs the round
///    polynomial evaluations into the transcript
///    to securely sample the next random challenge $r_i$.
///
/// # Returns
/// If successful, returns `Some((challenges, final_eval))`
/// containing the drawn random point $r = (r_0, \dots, r_{v-1})$
/// and the final expected evaluation at that point.
/// Returns `Err` or `None` if any consistency check
/// or degree validation fails.
pub fn verify<F: HardwareField, H: Hasher>(
    num_vars: usize,
    degree: usize,
    initial_claim: Flat<F>,
    proof: &SumcheckProof<F>,
    transcript: &mut Transcript<H>,
) -> SumcheckVerifyResult<F> {
    if proof.round_polys.len() != num_vars {
        return Ok(None);
    }

    if num_vars == 0 {
        transcript.append_field(b"final_val", proof.claimed_evaluation);

        if !flat_matches_canonical(initial_claim, proof.claimed_evaluation) {
            return Ok(None);
        }

        return Ok(Some((Vec::new(), proof.claimed_evaluation.to_hardware())));
    }

    let mut challenges = Vec::with_capacity(num_vars);
    let mut current_claim = initial_claim;

    for poly in proof.round_polys.iter() {
        // SECURITY:
        // Enforce polynomial degree to prevent degree
        // overflow attacks. A malicious prover could
        // send higher-degree polynomials. With Degree
        // Stripping, P(0) is omitted, so strictly
        // expect exactly `degree` evaluations.
        let expected_len = degree;
        if poly.evals.len() != expected_len {
            return Err(errors::Error::Protocol {
                protocol: "sumcheck",
                message: "polynomial degree mismatch - potential forgery attempt",
            });
        }

        // Degree stripping reconstruction:
        // P(0) + P(1) = current_claim => P(0) = current_claim - P(1)
        let p_1 = poly.evals[0].to_hardware();
        let p_0 = current_claim - p_1;

        let mut full_evals = Vec::with_capacity(degree + 1);
        full_evals.push(p_0.to_tower());
        full_evals.extend_from_slice(&poly.evals);

        // Replay Transcript interaction
        // with full polynomial.
        transcript.append_field_list(b"round_poly", &full_evals);

        // Generate Challenge
        let r_tower: F = transcript.challenge_field(b"challenge_r")?;
        let r_hw = r_tower.to_hardware();
        challenges.push(r_hw);

        // Update Claim:
        // expected_next = g_i(r)
        let full_poly = UnivariatePoly::new(full_evals);
        current_claim = full_poly.evaluate_hw(r_hw);
    }

    // Final Consistency Check
    if !flat_matches_canonical(current_claim, proof.claimed_evaluation) {
        return Ok(None);
    }

    // Commit final value to transcript
    transcript.append_field(b"final_val", proof.claimed_evaluation);

    Ok(Some((challenges, current_claim)))
}
