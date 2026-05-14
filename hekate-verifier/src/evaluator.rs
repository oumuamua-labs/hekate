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

use crate::brakedown::BrakedownVerifier;
use crate::sumcheck::verify;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::proofs::{BrakedownCommitment, EvalBatchProof};
use hekate_core::tensor::TensorProduct;
use hekate_core::trace::TraceCompatibleField;
use hekate_core::utils;
use hekate_crypto::transcript::Transcript;
use hekate_crypto::Hasher;
use hekate_math::matrix::ByteSparseMatrix;
use hekate_math::{Flat, HardwareField, PackableField};
use tracing::{debug, instrument, warn};

pub struct EvaluatorVerifier<F, H: Hasher> {
    _marker: PhantomData<(F, H)>,
}

pub struct EvalVerifyContext<'a, F, P: FnMut(&[u8], &mut Vec<Flat<F>>)> {
    pub points: Vec<&'a [Flat<F>]>,
    pub claimed_values_per_point: Vec<&'a [Flat<F>]>,
    pub num_vars: usize,
    pub row_parser: P,
}

impl<F, H: Hasher> EvaluatorVerifier<F, H>
where
    F: HardwareField + PackableField + TraceCompatibleField,
{
    /// Verifies the Batch Evaluation Argument (TensorPCS).
    ///
    /// This is the cryptographic cornerstone that links all
    /// polynomial evaluations (AIR constraints, GPA bus,
    /// GKR gadgets) to the physical Brakedown commitment.
    /// It prevents "Floating Proof" attacks by mathematically
    /// proving that the claimed evaluations strictly belong
    /// to the committed Merkle tree.
    ///
    /// # Cryptographic Protocol
    /// 1. **Random Linear Combination (RLC):** Folds all
    ///    requested columns using the `eta` challenge, and
    ///    all requested evaluation points using the `rho`
    ///    challenge, into a single target value.
    /// 2. **Evaluation Sumcheck:** Verifies a Sumcheck
    ///    protocol that reduces the 2D trace evaluation
    ///    to a single 1D vector `q`.
    /// 3. **ZK Codeword Check:** Since the Brakedown LDT
    ///    commits only to the encoded Parity/Code (without
    ///    exposing original data), the Verifier explicitly
    ///    encodes the `q` vector using the exact same
    ///    Expander Matrix.
    /// 4. **LDT Verification:** Verifies the Brakedown
    ///    proofs to ensure the committed matrix is close
    ///    to a valid codeword.
    /// 5. **Virtual Unpacking & Proximity Check:**
    ///    Parses the raw physical bytes from the LDT leaves
    ///    using the layout-aware `row_parser`. The physical
    ///    bytes are expanded into virtual field elements,
    ///    linearly combined using interleaved `eta`
    ///    coefficients, and checked against `q_encoded`.
    ///
    /// # Architecture: Column Consistency Check
    ///
    /// The Verifier receives the folded vector `q` and
    /// explicitly encodes it into `q_encoded`. Then, for
    /// each randomly opened LDT column, the Verifier applies
    /// the exact same folding logic (using powers of `r`)
    /// and asserts that the result matches the corresponding
    /// element in `q_encoded`.
    ///
    /// ```text
    /// Given a random challenge opening Column 2:
    ///
    /// Prover provides LDT column: [C2, C9, C16]
    /// Verifier computes:          (C2*r^0 + C9*r^1 + C16*r^2)
    /// Verifier asserts:           Result == q_encoded[2]
    /// ```
    #[instrument(skip_all, name = "Evaluator::verify")]
    pub fn verify<P>(
        commitment: &BrakedownCommitment,
        proof: &EvalBatchProof<F>,
        transcript: &mut Transcript<H>,
        mut ctx: EvalVerifyContext<'_, F, P>,
        config: &Config,
    ) -> errors::Result<bool>
    where
        P: FnMut(&[u8], &mut Vec<Flat<F>>),
    {
        let points = ctx.points;
        let claimed_values_per_point = ctx.claimed_values_per_point;
        let num_vars = ctx.num_vars;

        let num_points = points.len();
        if num_points == 0 || claimed_values_per_point.len() != num_points {
            debug!(
                "Input mismatch: points={}, values={}",
                num_points,
                claimed_values_per_point.len()
            );
            return Err(errors::Error::Protocol {
                protocol: "evaluator_verifier",
                message: "input length mismatch",
            });
        }

        let num_cols = claimed_values_per_point[0].len();

        transcript.append_message(b"eval_batch_start", b"");

        // Sync η (eta) and ρ (rho)
        for point_vals in &claimed_values_per_point {
            for val in *point_vals {
                transcript.append_field(b"claimed_val", val.to_tower());
            }
        }

        let eta_tower = transcript.challenge_field::<F>(b"eval_eta")?;
        let rho_tower = transcript.challenge_field::<F>(b"eval_rho")?;
        let eta = eta_tower.to_hardware();
        let rho = rho_tower.to_hardware();

        // Combined Target V(r_0)
        let mut target_value = Flat::from_raw(F::ZERO);
        let mut rho_pow = Flat::from_raw(F::ONE);

        for point_vals in &claimed_values_per_point {
            if point_vals.len() != num_cols {
                warn!(
                    "Input mismatch: expected {} columns, got {}",
                    num_cols,
                    point_vals.len()
                );
                return Err(errors::Error::Protocol {
                    protocol: "evaluator_verifier",
                    message: "claimed_values_per_point column count mismatch",
                });
            }

            let mut col_rlc = Flat::from_raw(F::ZERO);
            let mut eta_pow = Flat::from_raw(F::ONE);

            for &val in *point_vals {
                col_rlc += eta_pow * val;
                eta_pow *= eta;
            }

            target_value += rho_pow * col_rlc;
            rho_pow *= rho;
        }

        // Verify Sumcheck
        let sc_res = verify(num_vars, 2, target_value, &proof.sumcheck_proof, transcript)?;
        let (r_row, sumcheck_final_eval) = match sc_res {
            Some(res) => res,
            None => {
                warn!("Sumcheck failed");
                return Ok(false);
            }
        };

        // Verification of the Sumcheck evaluation against q
        let q = &proof.tensor_vec;
        transcript.append_field_list(b"tensor_q", q);

        let split_vars = utils::compute_split_vars(num_vars, config.num_queries);
        let grid_cols = 1 << split_vars;
        let grid_rows = 1 << (num_vars - split_vars);
        let encoded_width = grid_cols + config.ldt_blinding_factor;

        // Codeword consistency check
        //
        // The Prover's Sumcheck yielded a vector
        // q (folded data + ZK noise). Since the
        // Brakedown LDT only commits to the encoded
        // matrix (Parity/Code), the Verifier must
        // explicitly encode q using the exact same
        // Expander Matrix. Because the matrix is
        // binary, this algebraically mirrors the
        // Prover's actions natively.
        let matrix = ByteSparseMatrix::generate_random(
            encoded_width,
            encoded_width,
            config.expansion_degree,
            config.matrix_seed,
        );

        let q_flat = q
            .iter()
            .copied()
            .map(|value| value.to_hardware())
            .collect::<Vec<_>>();
        let q_encoded = matrix.spmv(q_flat.as_slice());

        if q.len() != encoded_width || q_encoded.len() != encoded_width {
            warn!("tensor_q length mismatch");
            return Ok(false);
        }

        let r_col_low = &r_row[..split_vars];
        let tensor_col = TensorProduct::<F>::new(r_col_low.to_vec());

        let mut master_eval = Flat::from_raw(F::ZERO);

        // Verify Sumcheck vs Tensor Fold
        //
        // The master_eval connects the 2D tensor sum back
        // to the 1D Sumcheck claim. The parity and ZK noise
        // portions are only meant for the LDT proximity check
        // and must not participate in the core AIR polynomial
        // evaluation.
        for (i, &val) in q_flat.iter().take(grid_cols).enumerate() {
            master_eval += val * tensor_col.evaluate_at_index(i);
        }

        if sumcheck_final_eval != master_eval {
            warn!("Dot product failed: sumcheck_final_eval != master_eval");
            return Ok(false);
        }

        // Fork transcript to reproduce
        // exact random queries generated by LDT
        transcript.append_message(b"eval_batch_ldt", b"");

        let mut ldt_transcript = transcript.clone();

        let opened_columns = BrakedownVerifier::<F, H>::verify(
            commitment,
            &proof.ldt_proof,
            transcript, // advances the real transcript
            config,
        )?;

        // Replay randomness generation
        let mut random_indices = Vec::with_capacity(config.num_queries);
        for _ in 0..config.num_queries {
            let bytes = ldt_transcript
                .challenge_field::<F>(b"idx_query")?
                .to_bytes();

            let mut rng_val: u64 = 0;
            for (k, &b) in bytes.iter().take(8).enumerate() {
                rng_val |= (b as u64) << (8 * k);
            }

            random_indices.push((rng_val % (encoded_width as u64)) as usize);
        }

        // Proximity Test
        let r_row_high = &r_row[split_vars..];
        let tensor_row = TensorProduct::<F>::new(r_row_high.to_vec());

        // Pre-expand TensorPCS row evaluations
        let mut tensor_row_evals = Vec::with_capacity(grid_rows);
        for r in 0..grid_rows {
            tensor_row_evals.push(tensor_row.evaluate_at_index(r));
        }

        let combo_factor = {
            let mut sum = Flat::from_raw(F::ZERO);
            let mut r_pow = Flat::from_raw(F::ONE);

            for point in &points {
                sum += r_pow * TensorProduct::evaluate_eq_slice(point, &r_row);
                r_pow *= rho;
            }

            sum
        };

        let mut final_col_coeffs = vec![Flat::from_raw(F::ZERO); num_cols];

        // Proximity Test (TensorPCS Folding Check)
        //
        // Set up coefficients for the folded evaluation
        // based on the eta/rho challenges. Since the data
        // is physically interleaved as [Base0, Shift0, Base1, Shift1...],
        // the coefficients must perfectly align with this layout:
        // [eta^0, eta^N * eta^0,  eta^1, eta^N * eta^1, ...]
        // (where N is base_width).
        let base_width = num_cols / 2;
        let mut eta_pow = Flat::from_raw(F::ONE);

        // Calculate eta^N
        let mut eta_shift_coeff = Flat::from_raw(F::ONE);
        for _ in 0..base_width {
            eta_shift_coeff *= eta;
        }

        for i in 0..base_width {
            // Coefficient for Base column i
            final_col_coeffs[2 * i] = eta_pow * combo_factor;

            // Coefficient for Shifted column i,
            // must include eta^N shift.
            final_col_coeffs[2 * i + 1] = (eta_pow * eta_shift_coeff) * combo_factor;
            eta_pow *= eta;
        }

        let mut virtual_row = Vec::with_capacity(num_cols);

        for (q_idx, &col_idx) in random_indices.iter().enumerate() {
            // Process LDT Openings:
            // Because Base and Shifted columns are
            // interleaved during matrix encoding, one
            // queried leaf natively contains all the
            // data needed for AIR transitions.
            let col_bytes = &opened_columns[q_idx];
            let row_bytes_len = col_bytes.len() / grid_rows;

            let mut calculated_q_val = Flat::from_raw(F::ZERO);

            for r in 0..grid_rows {
                let row_data = &col_bytes[r * row_bytes_len..(r + 1) * row_bytes_len];
                let mut row_lin_comb = Flat::from_raw(F::ZERO);

                virtual_row.clear();

                // Virtual Unpacking:
                // The row_parser will slice the physical bytes
                // using the strict physical layout and then
                // invoke the parse_virtual_row to expand them
                // into field elements.
                (ctx.row_parser)(row_data, &mut virtual_row);

                // SAFETY
                if virtual_row.len() != num_cols {
                    warn!(
                        "Row parser produced {} columns, but expected {}",
                        virtual_row.len(),
                        num_cols
                    );
                    return Err(errors::Error::Protocol {
                        protocol: "evaluator",
                        message: "row parser column count mismatch",
                    });
                }

                // Elements are interleaved:
                // [Col0_Base, Col0_Shifted, Col1_Base, ...]
                for c_idx in 0..num_cols {
                    row_lin_comb += virtual_row[c_idx] * final_col_coeffs[c_idx];
                }

                let v_row_r = tensor_row_evals[r];
                calculated_q_val += row_lin_comb * v_row_r;
            }

            // Compare against the encoded q vector
            if calculated_q_val != q_encoded[col_idx] {
                warn!("TensorPCS proximity check mismatch for column {}", col_idx);
                return Ok(false);
            }
        }

        Ok(true)
    }
}
