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

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod brakedown;
mod evaluator;
mod logup;
mod sumcheck;

pub use sumcheck::verify;

use crate::evaluator::{EvalVerifyContext, EvaluatorVerifier};
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::proofs::InnerProof;
use hekate_core::protocol;
use hekate_core::tensor::TensorProduct;
use hekate_core::trace::{ColumnType, TraceCompatibleField};
use hekate_crypto::Hasher;
use hekate_crypto::transcript::Transcript;
use hekate_math::{Flat, HardwareField, PackableField, TowerField};
use hekate_program::permutation::{self, BusKind};
use hekate_program::{Air, LagrangePin, Program, ProgramInstance, chiplet, validate_lagrange_pins};
use tracing::{debug, info, instrument, warn};

/// The main Hekate Verifier for AIR circuits.
pub struct HekateVerifier<F, H> {
    _marker: PhantomData<(F, H)>,
}

impl<F, H: Hasher> HekateVerifier<F, H>
where
    F: HardwareField + PackableField + TraceCompatibleField,
{
    /// Verifies an `InnerProof` produced
    /// by `HekateProver::prove`.
    ///
    /// Replays the prover's phase ordering:
    /// 1. Bind public inputs, config,
    ///    and trace root into the transcript.
    /// 2. Absorb each chiplet header
    ///    (name, rows, cols, root).
    /// 3. Draw global LogUp challenges γ, β.
    /// 4. Per chiplet:
    ///    verify ZeroCheck (with LogUp)
    ///    and the eval at `r_final`.
    /// 5. Verify the main AIR ZeroCheck (with LogUp).
    /// 6. Verify the main eval at `r_final`.
    /// 7. Check that LogUp `claimed_sum` totals cancel per `bus_id`.
    #[instrument(skip_all, name = "Hekate::verify")]
    pub fn verify<P: Program<F>>(
        program: &P,
        instance: &ProgramInstance<F>,
        proof: &InnerProof<F>,
        transcript: &mut Transcript<H>,
        config: &Config,
    ) -> errors::Result<bool> {
        let num_rows = instance.num_rows();
        let num_vars = num_rows.trailing_zeros() as usize;
        let num_cols = program.num_columns();
        let trace_width = num_cols;

        // =========================================================
        // 1. CRITICAL SECURITY VALIDATIONS
        // =========================================================

        if config.num_queries == 0 {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "num_queries cannot be zero",
            });
        }

        if num_rows == 0 || !num_rows.is_power_of_two() {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "num_rows must be a non-zero power of two",
            });
        }

        if proof.trace_commitment.num_rows != num_rows {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "trace_commitment.num_rows does not match instance.num_rows",
            });
        }

        let field_bits = size_of::<F>() * 8;
        let metrics = config.security_metrics(field_bits);

        info!(
            "System Security: ~{} bits (LDT: {}, Field: {}, Distance: {:.4}, d={})",
            metrics.security_bits,
            metrics.ldt_bits,
            field_bits,
            metrics.relative_distance,
            metrics.expansion_degree
        );

        config.check_security(num_vars, field_bits)?;

        if proof.eval_proof.point_evaluations.is_empty() {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "eval_proof is missing point evaluations",
            });
        }

        if proof.eval_proof.point_evaluations.len() != 1 {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "main eval_proof must carry exactly one point (r_final)",
            });
        }

        let expected_trace_len = trace_width + config.sumcheck_blinding_factor;
        let combined_vals = &proof.eval_proof.point_evaluations[0].1;

        if combined_vals.len() != expected_trace_len * 2 {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "combined trace values length mismatch with physical trace",
            });
        }

        let trace_values = canonical_slice_to_flat(&combined_vals[0..expected_trace_len]);
        let trace_values_next = canonical_slice_to_flat(&combined_vals[expected_trace_len..]);

        let main_perm = program.permutation_checks();
        let chiplet_defs_for_check = program.chiplet_defs()?;

        for (bus_id, spec) in &main_perm {
            spec.validate_clock_stitching(bus_id)?;
        }

        for def in &chiplet_defs_for_check {
            for (bus_id, spec) in &def.permutation_checks {
                spec.validate_clock_stitching(bus_id)?;
            }

            chiplet::validate_paired_bus_mutex(&def.permutation_checks, &def.constraint_ast())?;
        }

        chiplet::validate_paired_bus_mutex(&main_perm, &program.constraint_ast())?;

        let all_endpoints = main_perm.iter().map(|(id, s)| (id.as_str(), s)).chain(
            chiplet_defs_for_check
                .iter()
                .flat_map(|d| d.permutation_checks.iter().map(|(id, s)| (id.as_str(), s))),
        );

        permutation::validate_bus_set(all_endpoints)?;

        validate_lagrange_pins(
            &program.lagrange_pinned_columns(),
            program.num_columns(),
            Some(num_vars),
        )?;

        for (c_idx, def) in chiplet_defs_for_check.iter().enumerate() {
            let c_num_rows = proof
                .chiplet_commitments
                .get(c_idx)
                .ok_or(errors::Error::Protocol {
                    protocol: "verifier",
                    message: "chiplet commitment count mismatch",
                })?
                .num_rows;

            if c_num_rows == 0 || !c_num_rows.is_power_of_two() {
                return Err(errors::Error::Protocol {
                    protocol: "verifier",
                    message: "chiplet_commitments[c].num_rows must be a non-zero power of two",
                });
            }

            let c_num_vars = c_num_rows.trailing_zeros() as usize;

            validate_lagrange_pins(
                &Air::<F>::lagrange_pinned_columns(def),
                def.num_columns(),
                Some(c_num_vars),
            )?;
        }

        // =========================================================
        // PHASE 1: TRACE COMMITMENT & FIAT-SHAMIR BINDING
        // =========================================================
        Self::verify_trace_commitment(program, instance, proof, transcript, config)?;

        // =========================================================
        // PHASE 2: COMMIT EACH CHIPLET (absorb roots)
        // =========================================================
        Self::verify_chiplet_commitments_only(program, proof, transcript)?;

        // =========================================================
        // PHASE 3: DRAW GLOBAL γ, β,
        // AND r_bus PER LOOKUP BUS
        // =========================================================
        let gamma = transcript.challenge_field::<F>(b"bus_gamma")?.to_hardware();
        let beta = transcript.challenge_field::<F>(b"bus_beta")?.to_hardware();

        let lookup_bus_points = Self::draw_lookup_bus_points(program, proof, transcript)?;

        // =========================================================
        // PHASE 4: PER-CHIPLET FUSED (ZC + LogUp + eval)
        // =========================================================
        Self::verify_chiplet_fused(
            program,
            proof,
            transcript,
            config,
            gamma,
            beta,
            &lookup_bus_points,
        )?;

        // =========================================================
        // PHASE 5: MAIN AIR ZEROCHECK + LogUp
        // =========================================================
        let main_bus_specs = program.permutation_checks();
        let r_final = match Self::verify_zerocheck(
            program,
            instance,
            &proof.zerocheck_proof,
            transcript,
            config,
            trace_width,
            &trace_values,
            &trace_values_next,
            &main_bus_specs,
            &proof.main_logup_aux,
            gamma,
            beta,
            &lookup_bus_points,
        )? {
            Some(r) => r,
            None => return Ok(false),
        };

        // The eval_proof's stored r_final
        // must match the Sumcheck-derived
        // r_final exactly.
        let stored_pt = canonical_slice_to_flat(&proof.eval_proof.point_evaluations[0].0);
        if stored_pt != r_final {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "eval_proof point 0 mismatch with r_final",
            });
        }

        // =========================================================
        // PHASE 6: MAIN EVAL AT r_final (single point)
        // =========================================================
        if !Self::verify_eval_at_r_final(
            program,
            &proof.trace_commitment,
            &proof.eval_proof,
            transcript,
            config,
            num_vars,
            &r_final,
            &combined_vals
                .iter()
                .copied()
                .map(|v| v.to_hardware())
                .collect::<Vec<_>>(),
        )? {
            warn!("Main trace evaluation verification failed");
            return Ok(false);
        }

        // =========================================================
        // PHASE 7: CROSS-BUS MATCHING
        // (Σ claimed_sum = 0 per bus_id)
        // =========================================================
        let mut endpoints: Vec<(String, F)> = Vec::new();
        for (bus_id, claim) in &proof.main_logup_aux.claimed_sums {
            endpoints.push((bus_id.clone(), *claim));
        }
        for aux in &proof.chiplet_logup_aux {
            for (bus_id, claim) in &aux.claimed_sums {
                endpoints.push((bus_id.clone(), *claim));
            }
        }

        logup::check_bus_sum_matching(&endpoints)?;

        Ok(true)
    }

    /// PHASE 2:
    /// Absorb each chiplet's structure +
    /// root into the transcript (no ZeroCheck).
    /// Mirrors prover's commit_chiplets_only.
    #[instrument(skip_all, name = "verify_chiplet_commitments_only")]
    fn verify_chiplet_commitments_only<P: Program<F>>(
        program: &P,
        proof: &InnerProof<F>,
        transcript: &mut Transcript<H>,
    ) -> errors::Result<()> {
        let chiplet_defs = program.chiplet_defs()?;

        if proof.chiplet_commitments.len() != chiplet_defs.len() {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "chiplet commitment count mismatch",
            });
        }

        for (c_idx, def) in chiplet_defs.iter().enumerate() {
            let c_comm = &proof.chiplet_commitments[c_idx];
            let c_num_rows = c_comm.num_rows;

            protocol::absorb_chiplet_header(
                transcript,
                &def.name(),
                c_num_rows,
                def.num_columns(),
                &c_comm.root,
            );

            for bc in &Air::<F>::boundary_constraints(def) {
                bc.absorb_into(transcript);
            }
        }

        Ok(())
    }

    /// PHASE 4:
    /// Per-chiplet fused verification.
    /// Mirrors prover's fused_chiplet_loop.
    #[instrument(skip_all, name = "verify_chiplet_fused")]
    fn verify_chiplet_fused<P: Program<F>>(
        program: &P,
        proof: &InnerProof<F>,
        transcript: &mut Transcript<H>,
        config: &Config,
        gamma: Flat<F>,
        beta: Flat<F>,
        lookup_bus_points: &BTreeMap<String, Vec<Flat<F>>>,
    ) -> errors::Result<()> {
        let chiplet_defs = program.chiplet_defs()?;

        if proof.chiplet_zerocheck_proofs.len() != chiplet_defs.len() {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "chiplet zerocheck proof count mismatch",
            });
        }

        if proof.chiplet_logup_aux.len() != chiplet_defs.len() {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "chiplet logup_aux count mismatch",
            });
        }

        if proof.chiplet_eval_proofs.len() != chiplet_defs.len() {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "chiplet eval_proofs count mismatch",
            });
        }

        for (c_idx, def) in chiplet_defs.iter().enumerate() {
            let c_comm = &proof.chiplet_commitments[c_idx];
            let c_sc_proof = &proof.chiplet_zerocheck_proofs[c_idx];
            let c_eval_proof = &proof.chiplet_eval_proofs[c_idx];
            let c_logup_aux = &proof.chiplet_logup_aux[c_idx];

            let c_num_rows = c_comm.num_rows;
            let c_num_vars = c_num_rows.trailing_zeros() as usize;
            let c_num_cols = def.num_columns();
            let c_trace_width = c_num_cols + config.sumcheck_blinding_factor;

            if c_eval_proof.point_evaluations.len() != 1 {
                return Err(errors::Error::Protocol {
                    protocol: "verifier",
                    message: "chiplet eval_proof must carry exactly one point",
                });
            }

            let c_combined = &c_eval_proof.point_evaluations[0].1;
            if c_combined.len() != c_trace_width * 2 {
                return Err(errors::Error::Protocol {
                    protocol: "verifier",
                    message: "chiplet trace values length mismatch",
                });
            }

            let c_trace_values = canonical_slice_to_flat(&c_combined[0..c_trace_width]);
            let c_trace_values_next = canonical_slice_to_flat(&c_combined[c_trace_width..]);

            let c_instance = ProgramInstance::new(c_num_rows, vec![]);
            let c_bus_specs = def.permutation_checks.clone();

            let r_final = match Self::verify_zerocheck(
                def,
                &c_instance,
                c_sc_proof,
                transcript,
                config,
                c_num_cols,
                &c_trace_values,
                &c_trace_values_next,
                &c_bus_specs,
                c_logup_aux,
                gamma,
                beta,
                lookup_bus_points,
            )? {
                Some(r) => r,
                None => {
                    warn!(chiplet_idx = c_idx, "Chiplet ZeroCheck failed");
                    return Err(errors::Error::Protocol {
                        protocol: "verifier",
                        message: "chiplet ZeroCheck failed",
                    });
                }
            };

            let stored_pt = canonical_slice_to_flat(&c_eval_proof.point_evaluations[0].0);
            if stored_pt != r_final {
                return Err(errors::Error::Protocol {
                    protocol: "verifier",
                    message: "chiplet eval_proof point mismatch with r_final",
                });
            }

            let combined_hw: Vec<Flat<F>> = c_combined
                .iter()
                .copied()
                .map(|v| v.to_hardware())
                .collect();

            if !Self::verify_eval_at_r_final(
                def,
                c_comm,
                c_eval_proof,
                transcript,
                config,
                c_num_vars,
                &r_final,
                &combined_hw,
            )? {
                warn!(
                    chiplet_idx = c_idx,
                    "Chiplet evaluation verification failed"
                );
                return Err(errors::Error::Protocol {
                    protocol: "verifier",
                    message: "chiplet evaluation verification failed",
                });
            }

            info!(
                chiplet_idx = c_idx,
                chiplet_name = def.name(),
                "Chiplet verified"
            );
        }

        Ok(())
    }

    /// PHASE 6 helper:
    /// single-point eval verify. Mirrors
    /// prover's prove_eval_at_r_final.
    #[instrument(skip_all, name = "verify_eval_at_r_final")]
    #[allow(clippy::too_many_arguments)]
    fn verify_eval_at_r_final<A: Air<F>>(
        air: &A,
        commitment: &hekate_core::proofs::BrakedownCommitment,
        eval_proof: &hekate_core::proofs::EvalBatchProof<F>,
        transcript: &mut Transcript<H>,
        config: &Config,
        num_vars: usize,
        r_final: &[Flat<F>],
        claimed_values: &[Flat<F>],
    ) -> errors::Result<bool> {
        let blinding_factor = config.sumcheck_blinding_factor;
        let layout: Vec<ColumnType> = air.column_layout().to_vec();

        let mut expected_row_bytes: usize = 0;
        for ct in &layout {
            expected_row_bytes += ct.byte_size() * 2;
        }

        // ZK Noise:
        // B128 base + shift per blinding column.
        expected_row_bytes += blinding_factor * 16 * 2;

        let mut base_bytes: Vec<u8> = Vec::with_capacity(512);
        let mut shift_bytes: Vec<u8> = Vec::with_capacity(512);
        let mut virt_base: Vec<Flat<F>> = Vec::with_capacity(layout.len());
        let mut virt_shift: Vec<Flat<F>> = Vec::with_capacity(layout.len());

        let mut row_parser = |row_bytes: &[u8], buf: &mut Vec<Flat<F>>| {
            if row_bytes.len() < expected_row_bytes {
                return;
            }

            base_bytes.clear();
            shift_bytes.clear();
            virt_base.clear();
            virt_shift.clear();

            let mut ptr = 0;

            // De-interleave Base + Shifted physical bytes.
            for ct in &layout {
                let size = ct.byte_size();
                base_bytes.extend_from_slice(&row_bytes[ptr..ptr + size]);

                ptr += size;

                shift_bytes.extend_from_slice(&row_bytes[ptr..ptr + size]);

                ptr += size;
            }

            // Virtual unpack via the AIR's row parser.
            air.parse_virtual_row(&base_bytes, &mut virt_base);
            air.parse_virtual_row(&shift_bytes, &mut virt_shift);

            // Re-interleave virtual columns:
            // [V0_base, V0_shift, V1_base, V1_shift, ...].
            for i in 0..virt_base.len() {
                buf.push(virt_base[i]);
                buf.push(virt_shift[i]);
            }

            // Append ZK Noise (always B128).
            for _ in 0..blinding_factor {
                let v_base = ColumnType::B128.parse_from_bytes::<F>(&row_bytes[ptr..ptr + 16]);
                buf.push(v_base);

                ptr += 16;

                let v_shift = ColumnType::B128.parse_from_bytes::<F>(&row_bytes[ptr..ptr + 16]);
                buf.push(v_shift);

                ptr += 16;
            }
        };

        let ctx = EvalVerifyContext {
            points: vec![r_final],
            claimed_values_per_point: vec![claimed_values],
            num_vars,
            row_parser: &mut row_parser,
        };

        EvaluatorVerifier::<F, H>::verify(commitment, eval_proof, transcript, ctx, config)
    }

    /// Mirrors `commit_main_trace` on the verifier side:
    /// absorbs every public parameter and the
    /// trace root before the first challenge.
    #[instrument(skip_all, name = "verify_trace_commitment")]
    fn verify_trace_commitment<P: Program<F>>(
        program: &P,
        instance: &ProgramInstance<F>,
        proof: &InnerProof<F>,
        transcript: &mut Transcript<H>,
        config: &Config,
    ) -> errors::Result<()> {
        let num_rows = instance.num_rows();
        let num_cols = program.num_columns();

        transcript.append_u64(b"num_columns", num_cols as u64);
        transcript.append_u64(b"num_rows", num_rows as u64);
        transcript.append_u64(b"ldt_blinding_factor", config.ldt_blinding_factor as u64);
        transcript.append_u64(
            b"sumcheck_blinding_factor",
            config.sumcheck_blinding_factor as u64,
        );
        transcript.append_u64(b"num_queries", config.num_queries as u64);

        for val in instance.public_inputs() {
            transcript.append_field(b"public_input", *val);
        }

        transcript.append_message(b"trace_root", &proof.trace_commitment.root);

        for bc in &program.boundary_constraints() {
            bc.absorb_into(transcript);
        }

        Ok(())
    }

    /// Verifies AIR + LogUp ZeroCheck.
    ///
    /// The initial sumcheck claim is `Σ_k α^k · claimed_sum_k`
    /// (LogUp bus-sum total), not zero. The consistency
    /// check at `r_final` covers AIR, boundary,
    /// ZK blinding, and LogUp contributions.
    #[instrument(skip_all, name = "verify_zerocheck")]
    #[allow(clippy::too_many_arguments)]
    fn verify_zerocheck<A: Air<F>>(
        air: &A,
        instance: &ProgramInstance<F>,
        sc_proof: &hekate_core::proofs::SumcheckProof<F>,
        transcript: &mut Transcript<H>,
        config: &Config,
        trace_width: usize,
        trace_values: &[Flat<F>],
        trace_values_next: &[Flat<F>],
        bus_specs: &[(String, permutation::PermutationCheckSpec)],
        logup_aux: &hekate_core::proofs::LogUpAux<F>,
        gamma: Flat<F>,
        beta: Flat<F>,
        lookup_bus_points: &BTreeMap<String, Vec<Flat<F>>>,
    ) -> errors::Result<Option<Vec<Flat<F>>>> {
        let num_rows = instance.num_rows();
        let num_vars = num_rows.trailing_zeros() as usize;

        // =========================================================
        // Constraint System Setup
        // =========================================================

        // Validate LogUp aux structure and bind
        // claimed_sums into the transcript before
        // drawing alpha / r_zerocheck so the
        // ZeroCheck challenges depend on the
        // bus-sum target.
        if !bus_specs.is_empty() {
            if logup_aux.claimed_sums.len() != bus_specs.len() {
                return Err(errors::Error::Protocol {
                    protocol: "verifier",
                    message: "logup_aux claimed_sums length mismatch with bus_specs",
                });
            }

            if logup_aux.h_evals.len() != bus_specs.len() {
                return Err(errors::Error::Protocol {
                    protocol: "verifier",
                    message: "logup_aux h_evals length mismatch with bus_specs",
                });
            }

            for (i, ((h_bus, _), (claim_bus, _))) in logup_aux
                .h_evals
                .iter()
                .zip(logup_aux.claimed_sums.iter())
                .enumerate()
            {
                if h_bus != claim_bus {
                    return Err(errors::Error::Protocol {
                        protocol: "verifier",
                        message: "logup_aux bus_id order diverges between h_evals and claimed_sums",
                    });
                }

                if h_bus.as_str() != bus_specs[i].0.as_str() {
                    return Err(errors::Error::Protocol {
                        protocol: "verifier",
                        message: "logup_aux bus_id does not match bus_specs ordering",
                    });
                }
            }
        }

        protocol::absorb_logup_claimed_sums(transcript, &logup_aux.claimed_sums);

        let alpha_tower = transcript.challenge_field::<F>(b"alpha")?;
        let alpha = alpha_tower.to_hardware();
        let r_zerocheck = (0..num_vars)
            .map(|_| {
                transcript
                    .challenge_field::<F>(b"r_zerocheck")
                    .map(|v| v.to_hardware())
            })
            .collect::<Result<Vec<_>, _>>()?;

        let ast = air.constraint_ast();
        let boundary_constraints = air.boundary_constraints();

        let mut sumcheck_degree = ast.max_degree() + 1;
        if !boundary_constraints.is_empty() {
            sumcheck_degree = sumcheck_degree.max(2);
        }

        // LogUp consistency `h · key · Eq` is degree 3.
        if !bus_specs.is_empty() {
            sumcheck_degree = sumcheck_degree.max(3);
        }

        // LogUp α_pow offset must match
        // the prover's continuation past
        // AIR + boundary + blinding.
        let logup_alpha_offset =
            ast.roots.len() + boundary_constraints.len() + config.sumcheck_blinding_factor;

        let mut alpha_logup_start = Flat::from_raw(F::ONE);
        for _ in 0..logup_alpha_offset {
            alpha_logup_start *= alpha;
        }

        let mut initial_claim = Flat::from_raw(F::ZERO);
        if !bus_specs.is_empty() {
            let mut alpha_pow = alpha_logup_start;
            for (_, claim) in &logup_aux.claimed_sums {
                initial_claim += alpha_pow * claim.to_hardware();
                alpha_pow *= alpha;
            }
        }

        let sc_res = verify(
            num_vars,
            sumcheck_degree,
            initial_claim,
            sc_proof,
            transcript,
        )?;

        protocol::absorb_logup_h_evals(transcript, &logup_aux.h_evals);

        let (r_final, val_final) = match sc_res {
            Some(res) => res,
            None => {
                warn!("Main constraint sumcheck failed");
                return Ok(None);
            }
        };

        // =========================================================
        // GLOBAL CONSTRAINT CONSISTENCY CHECK
        // =========================================================

        debug!("Verifying AIR constraint consistency at r_final");

        let eq_zc_eval = TensorProduct::evaluate_eq_slice(&r_zerocheck, &r_final);

        // Separate physical trace from blinding
        // to match AIR constraints index layout.
        let current_row = &trace_values[0..trace_width];
        let next_row = &trace_values_next[0..trace_width];

        // A. Enforce Lagrange-pinned columns:
        // each pin's committed evaluation must equal
        // the MLE eval of its Lagrange point at r_final.
        // Builds an override row passed to ast.evaluate
        // so constraints see the verified value,
        // not the raw commit.
        let mut current_row_subst: Vec<Flat<F>> = current_row.to_vec();

        let pins = air.lagrange_pinned_columns();
        if !pins.is_empty() {
            for pin in &pins {
                let LagrangePin { col_idx, point } = pin;

                if *col_idx >= trace_width {
                    return Err(errors::Error::Protocol {
                        protocol: "verifier",
                        message: "lagrange pin col_idx out of trace_width",
                    });
                }

                let expected = point.evaluate::<F>(&r_final);

                if current_row[*col_idx] != expected {
                    warn!(
                        "Lagrange-pin forgery detected.\nCol: {}\nClaimed: {:?}\nExpected: {:?}",
                        col_idx, current_row[*col_idx], expected
                    );
                    return Ok(None);
                }

                current_row_subst[*col_idx] = expected;
            }
        }

        let mut expected_val = Flat::from_raw(F::ZERO);
        let mut alpha_pow = Flat::from_raw(F::ONE);
        let mut main_constraints_sum = Flat::from_raw(F::ZERO);

        // B. Main Constraints
        let constraint_evals = ast.evaluate(&current_row_subst, next_row);

        for eval in constraint_evals {
            main_constraints_sum += eval * alpha_pow;
            alpha_pow *= alpha;
        }

        // Factor out eq_zc_eval
        expected_val += main_constraints_sum * eq_zc_eval;

        // C. Boundary Constraints
        let eq_row_checker = TensorProduct::new(r_final.clone());

        for bc in boundary_constraints {
            if bc.col_idx >= current_row.len() {
                return Err(errors::Error::Protocol {
                    protocol: "verifier",
                    message: "boundary constraint col_idx out of bounds",
                });
            }

            if bc.row_idx >= 1 << num_vars {
                return Err(errors::Error::Protocol {
                    protocol: "verifier",
                    message: "boundary constraint row_idx exceeds trace height",
                });
            }

            let pub_val = bc.resolve_target(instance)?.to_hardware();
            let trace_val = current_row[bc.col_idx];
            let eq_eval = eq_row_checker.evaluate_at_index(bc.row_idx);

            expected_val += (trace_val - pub_val) * eq_eval * alpha_pow;
            alpha_pow *= alpha;
        }

        // D. Blinding Polynomial (Telescopic Sum)
        if config.sumcheck_blinding_factor > 0 {
            for k in 0..config.sumcheck_blinding_factor {
                // Extract blinding values directly
                // from the verified proof.
                let b_k = trace_values[trace_width + k];
                let b_k_next = trace_values_next[trace_width + k];

                // b_k * alpha - b_k_next * alpha = (b_k - b_k_next) * alpha
                expected_val += (b_k - b_k_next) * alpha_pow;
                alpha_pow *= alpha;
            }
        }

        // E. LogUp consistency + bus-sum at r_final
        if !bus_specs.is_empty() {
            let mut source_evals: Vec<Flat<F>> = Vec::new();

            for (spec_idx, (bus_id, spec)) in bus_specs.iter().enumerate() {
                let h_eval = logup_aux.h_evals[spec_idx].1.to_hardware();

                let s_eval = match spec.selector {
                    Some(idx) => current_row[idx],
                    None => Flat::from_raw(F::ONE),
                };

                let s_recv_eval = match spec.recv_selector {
                    Some(idx) => current_row[idx],
                    None => Flat::from_raw(F::ZERO),
                };

                // Source values flattened in spec order.
                // Const folds in at its own β-position
                // so the helper's stitching loop produces
                // the same key as the prover.
                source_evals.clear();

                for (source, _) in &spec.sources {
                    match source {
                        permutation::Source::Column(col_idx) => {
                            source_evals.push(current_row[*col_idx]);
                        }
                        permutation::Source::Columns(indices) => {
                            for &col_idx in indices {
                                source_evals.push(current_row[col_idx]);
                            }
                        }
                        permutation::Source::Const(val) => {
                            source_evals.push(F::from(*val).to_hardware());
                        }
                        permutation::Source::RowIndexLeBytes(n) => {
                            source_evals.push(eval_row_idx_le_mle::<F>(*n, &r_final));
                        }
                        permutation::Source::RowIndexByte(n) => {
                            source_evals.push(eval_row_idx_byte_mle::<F>(*n, &r_final));
                        }
                    }
                }

                let eq_lookup = match spec.kind {
                    BusKind::Permutation => Flat::from_raw(F::ONE),
                    BusKind::Lookup => {
                        let r_bus =
                            lookup_bus_points
                                .get(bus_id)
                                .ok_or(errors::Error::Protocol {
                                    protocol: "verifier",
                                    message: "lookup bus spec missing r_bus point",
                                })?;

                        if r_bus.len() < num_vars {
                            return Err(errors::Error::Protocol {
                                protocol: "verifier",
                                message: "r_bus shorter than table num_vars",
                            });
                        }

                        let r_lo = &r_bus[0..num_vars];
                        let r_hi = &r_bus[num_vars..];
                        let eq_r_lo_at_r_final = TensorProduct::evaluate_eq_slice(r_lo, &r_final);

                        let one = Flat::from_raw(F::ONE);

                        let mut eq_r_hi_at_0 = one;
                        for r_j in r_hi {
                            eq_r_hi_at_0 *= one - *r_j;
                        }

                        eq_r_hi_at_0 * eq_r_lo_at_r_final
                    }
                };

                let bus_eval = logup::BusSpecEvaluation {
                    h_eval,
                    s_eval,
                    s_recv_eval,
                    source_evals: &source_evals,
                    alpha_bus: alpha_pow,
                    eq_lookup,
                };

                expected_val +=
                    logup::expected_bus_contribution(&bus_eval, gamma, beta, eq_zc_eval);
                alpha_pow *= alpha;
            }
        }

        // Strict Check
        let masking_bias = val_final - expected_val;
        if masking_bias != Flat::from_raw(F::ZERO) {
            warn!(
                "Constraint logic mismatch.\nClaimed (Sumcheck): {:?}\nCalculated (Constraints): {:?}\nDiff: {:?}",
                val_final, expected_val, masking_bias
            );
            return Ok(None);
        }

        Ok(Some(r_final))
    }

    /// Aggregates lookup-bus heights from
    /// main + chiplet commitments and draws
    /// one `r_bus` per bus_id in sorted order.
    fn draw_lookup_bus_points<P: Program<F>>(
        program: &P,
        proof: &InnerProof<F>,
        transcript: &mut Transcript<H>,
    ) -> errors::Result<BTreeMap<String, Vec<Flat<F>>>> {
        let mut heights: BTreeMap<String, u64> = BTreeMap::new();
        permutation::accumulate_lookup_heights(
            &program.permutation_checks(),
            proof.trace_commitment.num_rows as u64,
            &mut heights,
        );

        let defs = program.chiplet_defs()?;
        for (def, c_comm) in defs.iter().zip(proof.chiplet_commitments.iter()) {
            permutation::accumulate_lookup_heights(
                &def.permutation_checks,
                c_comm.num_rows as u64,
                &mut heights,
            );
        }

        let entries: Vec<(String, u64)> = heights.into_iter().collect();
        let tower: BTreeMap<String, Vec<F>> =
            protocol::draw_lookup_bus_points(transcript, &entries)?;

        Ok(tower
            .into_iter()
            .map(|(id, v)| (id, v.into_iter().map(|x| x.to_hardware()).collect()))
            .collect())
    }
}

/// MLE of `Source::RowIndexLeBytes` at `r_final`.
/// Linear in `r_final` because `F::from`
/// is XOR-additive over char-2.
fn eval_row_idx_le_mle<F>(num_bytes: usize, r_final: &[Flat<F>]) -> Flat<F>
where
    F: TowerField + HardwareField + From<u128>,
{
    let total_bits = (num_bytes.min(8) * 8).min(r_final.len());

    let mut acc = Flat::from_raw(F::ZERO);
    for (i, r) in r_final.iter().enumerate().take(total_bits) {
        acc += F::from(1u128 << i).to_hardware() * *r;
    }

    acc
}

/// MLE of `Source::RowIndexByte` at `r_final`.
/// Same char-2 shortcut as `eval_row_idx_le_mle`,
/// restricted to one byte.
fn eval_row_idx_byte_mle<F>(byte_idx: usize, r_final: &[Flat<F>]) -> Flat<F>
where
    F: TowerField + HardwareField + From<u128>,
{
    let bit_start = byte_idx.saturating_mul(8);
    if bit_start >= r_final.len() {
        return Flat::from_raw(F::ZERO);
    }

    let end = (bit_start + 8).min(r_final.len());

    let mut acc = Flat::from_raw(F::ZERO);
    for (j, i) in (bit_start..end).enumerate() {
        acc += F::from(1u128 << j).to_hardware() * r_final[i];
    }

    acc
}

fn canonical_slice_to_flat<F: HardwareField>(values: &[F]) -> Vec<Flat<F>> {
    values
        .iter()
        .copied()
        .map(|value| value.to_hardware())
        .collect()
}

pub(crate) fn flat_matches_canonical<F: HardwareField>(value: Flat<F>, canonical: F) -> bool {
    value.to_tower() == canonical
}
