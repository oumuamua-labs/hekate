// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec::Vec;
use hekate_core::poly::UnivariatePoly;
use hekate_core::proofs::SumcheckProof;
use hekate_core::skip::SkipDomain;
use hekate_crypto::Hasher;
use hekate_crypto::transcript::Transcript;
use hekate_math::{Flat, HardwareField};

use crate::flat_matches_canonical;
use crate::rejection::SumcheckRejection;

/// Verifies a Sumcheck protocol execution over a Boolean hypercube.
///
/// The Sumcheck protocol reduces the claim that a multi-variate
/// polynomial sums to a specific value over the Boolean hypercube
/// $\{0, 1\}^v$ into a single claim about the polynomial's evaluation
/// at a randomly chosen point $r \in \mathbb{F}^v$.
///
/// # Security & Protocol Mechanics
/// 1. **Degree Enforcement:** Strictly validates
///    that each round polynomial provides exactly
///    `degree` evaluations. This is a critical
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
/// If successful, returns `(challenges, final_eval)`
/// containing the drawn random point $r = (r_0, \dots, r_{v-1})$
/// and the final expected evaluation at that point.
/// Returns the [`SumcheckRejection`] of the first
/// failing count, degree or consistency check.
pub fn verify<F: HardwareField, H: Hasher>(
    num_vars: usize,
    degree: usize,
    initial_claim: Flat<F>,
    proof: &SumcheckProof<F>,
    transcript: &mut Transcript<H>,
) -> Result<(Vec<Flat<F>>, Flat<F>), SumcheckRejection> {
    if proof.round_polys.len() != num_vars {
        return Err(SumcheckRejection::Rounds);
    }

    if num_vars == 0 {
        transcript.append_field(b"final_val", proof.claimed_evaluation);

        if !flat_matches_canonical(initial_claim, proof.claimed_evaluation) {
            return Err(SumcheckRejection::Claim);
        }

        return Ok((Vec::new(), proof.claimed_evaluation.to_hardware()));
    }

    let mut full_poly = UnivariatePoly::new(Vec::with_capacity(degree + 1));
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
            return Err(SumcheckRejection::Degree);
        }

        // Degree stripping reconstruction:
        // P(0) + P(1) = current_claim => P(0) = current_claim - P(1)
        let p_1 = poly.evals[0].to_hardware();
        let p_0 = current_claim - p_1;

        full_poly.evals.clear();
        full_poly.evals.push(p_0.to_tower());
        full_poly.evals.extend_from_slice(&poly.evals);

        // Replay Transcript interaction
        // with full polynomial.
        transcript.append_field_list(b"round_poly", &full_poly.evals);

        // Generate Challenge
        let r_tower: F = transcript.challenge_field(b"challenge_r")?;
        let r_hw = r_tower.to_hardware();
        challenges.push(r_hw);

        // Update Claim:
        // expected_next = g_i(r)
        current_claim = full_poly.evaluate_hw(r_hw);
    }

    // Final Consistency Check
    if !flat_matches_canonical(current_claim, proof.claimed_evaluation) {
        return Err(SumcheckRejection::Claim);
    }

    // Commit final value to transcript
    transcript.append_field(b"final_val", proof.claimed_evaluation);

    Ok((challenges, current_claim))
}

/// Replays a skip round: rebuilds the first domain value
/// from the claim, absorbs the univariate, draws the challenge
/// and returns it with the claim the tail continues from.
pub fn verify_skip<F: HardwareField, H: Hasher>(
    domain: &SkipDomain<F>,
    initial_claim: Flat<F>,
    sent: &[F],
    transcript: &mut Transcript<H>,
) -> Result<(Flat<F>, Flat<F>), SumcheckRejection> {
    let plan = domain.plan();

    if sent.len() != plan.sent_scalars() {
        return Err(SumcheckRejection::Degree);
    }

    let rest_of_block = sent[..plan.block_rows() - 1]
        .iter()
        .fold(Flat::from_raw(F::ZERO), |acc, v| acc + v.to_hardware());

    let mut values = Vec::with_capacity(plan.domain_points());
    values.push((initial_claim - rest_of_block).to_tower());
    values.extend_from_slice(sent);

    transcript.append_field_list(b"skip_poly", &values);

    let r_skip = transcript.challenge_field::<F>(b"skip_r")?.to_hardware();

    let flat: Vec<Flat<F>> = values.iter().map(|v| v.to_hardware()).collect();
    let claim = domain.interpolate(&flat, r_skip)?;

    Ok((r_skip, claim))
}
