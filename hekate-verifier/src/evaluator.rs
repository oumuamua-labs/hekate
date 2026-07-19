// SPDX-License-Identifier: Apache-2.0
// This file is part of the hekate project.
// Copyright (C) 2026 Andrei Kochergin <andrei@oumuamua.dev>
// Copyright (C) 2026 Oumuamua Labs <info@oumuamua.dev>.
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
use hekate_core::trace::{ColumnType, TraceCompatibleField};
use hekate_core::utils;
use hekate_crypto::Hasher;
use hekate_crypto::transcript::Transcript;
use hekate_math::{
    AdditiveFft, BinaryFieldExtras, Block128, Flat, HardwareField, PackableField, TowerField,
};
use hekate_program::expander::RingSwitchPlan;
use tracing::{instrument, warn};

#[cfg(feature = "parallel")]
const PARALLEL_PROXIMITY_THRESHOLD: usize = 1 << 18;

const NBITS: usize = 128;

pub struct EvaluatorVerifier<F, H: Hasher> {
    _marker: PhantomData<(F, H)>,
}

pub struct EvalVerifyContext<'a, F: HardwareField> {
    pub points: Vec<&'a [Flat<F>]>,
    pub claimed_values_per_point: Vec<&'a [Flat<F>]>,
    pub num_vars: usize,
    pub row_bytes: usize,
    pub ring_plan: &'a RingSwitchPlan,

    /// `false` opens base columns only (no next-row shift).
    pub next_row: bool,
}

impl<F, H: Hasher> EvaluatorVerifier<F, H>
where
    F: HardwareField + PackableField + TraceCompatibleField,
{
    /// Verifies the ring-switch TensorPCS evaluation argument,
    /// binding the claimed virtual evals to the Brakedown commitment.
    /// A degree-2 sumcheck reduces `A·master_bit + master_whole·Eq(P)`
    /// to `r_final`; the final check pairs `A(r')` and `Eq(P,r')`
    /// with two whole-column openings that the proximity test
    /// binds to the committed codewords.
    #[instrument(skip_all, name = "Evaluator::verify")]
    pub fn verify(
        commitment: &BrakedownCommitment,
        proof: &EvalBatchProof<F>,
        transcript: &mut Transcript<H>,
        ctx: EvalVerifyContext<'_, F>,
        config: &Config,
    ) -> errors::Result<bool>
    where
        F: BinaryFieldExtras + Into<Block128> + From<u128>,
    {
        let points = ctx.points;
        let claimed = ctx.claimed_values_per_point;
        let num_vars = ctx.num_vars;
        let row_bytes = ctx.row_bytes;
        let plan = ctx.ring_plan;
        let next_row = ctx.next_row;
        let variants = if next_row { 2 } else { 1 };

        if points.len() != 1 || claimed.len() != 1 {
            return Err(errors::Error::Protocol {
                protocol: "evaluator_verifier",
                message: "ring-switch evaluation expects a single point",
            });
        }

        let point = points[0];
        let claims = claimed[0];

        if claims.len() != plan.total_claims() * variants {
            return Err(errors::Error::Protocol {
                protocol: "evaluator_verifier",
                message: "ring-switch plan claim count does not match the claimed evaluations",
            });
        }

        transcript.append_message(b"eval_batch_start", b"");

        for &val in claims {
            transcript.append_field(b"claimed_val", val.to_tower());
        }

        let eta_tower = transcript.challenge_field::<F>(b"eval_eta")?;

        // Drawn for transcript parity with the prover's multi-point path;
        // production is single-point, rho is unused.
        let _rho = transcript.challenge_field::<F>(b"eval_rho")?;

        let eta = eta_tower.to_hardware();

        let has_ring = plan.has_ring();

        if has_ring && F::BITS != NBITS {
            return Err(errors::Error::Protocol {
                protocol: "evaluator_verifier",
                message: "ring-switch evaluation requires a 128-bit field",
            });
        }

        // Order is load-bearing:
        // r'' follows the claimed evals, precedes the sumcheck.
        let kappa = F::BITS.ilog2() as usize;

        let r_mix: Vec<Block128> = if has_ring {
            let mut m = Vec::with_capacity(kappa);
            for _ in 0..kappa {
                m.push(transcript.challenge_field::<F>(b"eval_rmix")?.into());
            }

            m
        } else {
            Vec::new()
        };

        let target = ring_target::<F>(plan, claims, eta_tower, &r_mix, next_row);
        let target_flat = F::from(target.0).to_hardware();

        let sc_res = verify(num_vars, 2, target_flat, &proof.sumcheck_proof, transcript)?;
        let (r_row, sumcheck_final_eval) = match sc_res {
            Some(res) => res,
            None => {
                warn!("Sumcheck failed");
                return Ok(false);
            }
        };

        let q_whole = &proof.tensor_vec;
        let q_ring = &proof.tensor_vec_ring;

        transcript.append_field_list(b"tensor_q", q_whole);

        if has_ring {
            transcript.append_field_list(b"tensor_q_ring", q_ring);
        }

        let split_vars = utils::compute_split_vars(
            num_vars,
            config.num_queries,
            config.ldt_support_size,
            row_bytes,
        );

        let grid_cols = 1 << split_vars;
        let grid_rows = 1 << (num_vars - split_vars);
        let geom = config.table_geom(grid_cols);
        let encoded_width = geom.encoded_width;

        if grid_cols + geom.support_size > encoded_width {
            warn!("support + data message exceeds the codeword width");
            return Ok(false);
        }

        config.check_security(size_of::<F>() * 8, grid_cols)?;

        let expected_len = grid_cols + geom.support_size;

        if q_whole.len() != expected_len || (has_ring && q_ring.len() != expected_len) {
            warn!("tensor_q length mismatch");
            return Ok(false);
        }

        let q_whole_flat: Vec<Flat<F>> = q_whole.iter().map(|v| v.to_hardware()).collect();
        let q_ring_flat: Vec<Flat<F>> = if has_ring {
            q_ring.iter().map(|v| v.to_hardware()).collect()
        } else {
            Vec::new()
        };

        // Two independent encodes
        #[cfg(feature = "parallel")]
        let (q_whole_res, q_ring_res) = rayon::join(
            || rs_encode_row::<F>(&q_whole_flat, grid_cols, config),
            || {
                if has_ring {
                    rs_encode_row::<F>(&q_ring_flat, grid_cols, config)
                } else {
                    Ok(Vec::new())
                }
            },
        );

        #[cfg(feature = "parallel")]
        let (q_whole_encoded, q_ring_encoded) = (q_whole_res?, q_ring_res?);

        #[cfg(not(feature = "parallel"))]
        let q_whole_encoded = rs_encode_row::<F>(&q_whole_flat, grid_cols, config)?;

        #[cfg(not(feature = "parallel"))]
        let q_ring_encoded = if has_ring {
            rs_encode_row::<F>(&q_ring_flat, grid_cols, config)?
        } else {
            Vec::new()
        };

        let r_col_low = &r_row[..split_vars];
        let tensor_col = build_tensor_table::<F>(r_col_low);

        let master_eval = |q: &[Flat<F>]| {
            let mut acc = Flat::from_raw(F::ZERO);
            for (&val, &t) in q.iter().take(grid_cols).zip(&tensor_col) {
                acc += val * t;
            }

            acc
        };

        let master_whole_eval = master_eval(&q_whole_flat);
        let master_bit_eval = if has_ring {
            master_eval(&q_ring_flat)
        } else {
            Flat::from_raw(F::ZERO)
        };

        // A(r') and Eq(P,r') are transparent; the two master evals
        // are bound by the whole-column proximity check below.
        let eq_at_r = TensorProduct::evaluate_eq_slice(point, &r_row);
        let a_r = if has_ring {
            let point_b: Vec<Block128> = point.iter().map(|f| f.to_tower().into()).collect();
            let r_row_b: Vec<Block128> = r_row.iter().map(|f| f.to_tower().into()).collect();

            F::from(ring_switch_a_at(&point_b, &r_row_b, &r_mix).0).to_hardware()
        } else {
            Flat::from_raw(F::ZERO)
        };

        if sumcheck_final_eval != a_r * master_bit_eval + eq_at_r * master_whole_eval {
            warn!("ring-switch final check failed");
            return Ok(false);
        }

        // Fork transcript to reproduce
        // exact random queries generated by LDT
        transcript.append_message(b"eval_batch_ldt", b"");

        let mut ldt_transcript = transcript.clone();

        let openings = BrakedownVerifier::<F, H>::verify(
            commitment,
            &proof.ldt_proof,
            transcript, // advances the real transcript
            config,
            row_bytes,
        )?;

        let opened_columns = openings.columns;
        let slot_map = &openings.slot_map;

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

        let r_row_high = &r_row[split_vars..];
        let tensor_row = TensorProduct::<F>::new(r_row_high.to_vec());

        let mut tensor_row_evals = Vec::with_capacity(grid_rows);
        for r in 0..grid_rows {
            tensor_row_evals.push(tensor_row.evaluate_at_index(r));
        }

        let num_phys = plan.phys_rs.len();
        let (coeff_bit, coeff_whole, eta_shift) = plan.column_coeffs::<F>(eta);
        let phys_row_bytes: usize = plan
            .phys_rs
            .iter()
            .map(|ct| variants * ct.byte_size())
            .sum();

        // eta^U is row-invariant:
        // fold it into the shift coefficients once.
        let coeff_whole_shift: Vec<Flat<F>> = coeff_whole.iter().map(|&c| c * eta_shift).collect();
        let coeff_bit_shift: Vec<Flat<F>> = if has_ring {
            coeff_bit.iter().map(|&c| c * eta_shift).collect()
        } else {
            Vec::new()
        };

        // Re-derive both folded openings from the physical columns
        // in the opened leaf; RS commutes with a whole-column fold,
        // this must match the RS re-encodings of the prover's
        // committed q vectors.
        let check_query =
            |q_idx: usize, col_idx: usize, phys_row: &mut Vec<Flat<F>>| -> errors::Result<bool> {
                let col_bytes = &opened_columns[slot_map[q_idx]];

                if col_bytes.len() != grid_rows * phys_row_bytes {
                    warn!("opened column length does not match the physical row layout");
                    return Ok(false);
                }

                let mut q_whole_val = Flat::from_raw(F::ZERO);
                let mut q_ring_val = Flat::from_raw(F::ZERO);

                for r in 0..grid_rows {
                    let row_data = &col_bytes[r * phys_row_bytes..(r + 1) * phys_row_bytes];

                    phys_row.clear();

                    parse_physical_row::<F>(row_data, &plan.phys_rs, phys_row, next_row);

                    let mut fold_whole = Flat::from_raw(F::ZERO);
                    let mut fold_bit = Flat::from_raw(F::ZERO);

                    for p in 0..num_phys {
                        let base = phys_row[variants * p];
                        fold_whole += base * coeff_whole[p];

                        if has_ring {
                            fold_bit += base * coeff_bit[p];
                        }

                        if next_row {
                            let shift = phys_row[variants * p + 1];
                            fold_whole += shift * coeff_whole_shift[p];

                            if has_ring {
                                fold_bit += shift * coeff_bit_shift[p];
                            }
                        }
                    }

                    let tr = tensor_row_evals[r];
                    q_whole_val += fold_whole * tr;
                    q_ring_val += fold_bit * tr;
                }

                let ok = q_whole_val == q_whole_encoded[col_idx]
                    && (!has_ring || q_ring_val == q_ring_encoded[col_idx]);

                if !ok {
                    warn!("TensorPCS proximity mismatch for column {}", col_idx);
                }

                Ok(ok)
            };

        let run_sequential = |indices: &[usize]| -> errors::Result<bool> {
            let mut phys_row = Vec::with_capacity(2 * num_phys);
            for (q_idx, &col_idx) in indices.iter().enumerate() {
                if !check_query(q_idx, col_idx, &mut phys_row)? {
                    return Ok(false);
                }
            }

            Ok(true)
        };

        #[cfg(feature = "parallel")]
        let all_matched = {
            let per_row_cols = num_phys * if has_ring { 3 } else { 2 };
            let proximity_work = config.num_queries * grid_rows * per_row_cols;

            if proximity_work >= PARALLEL_PROXIMITY_THRESHOLD {
                use rayon::prelude::*;

                random_indices
                    .par_iter()
                    .enumerate()
                    .map_init(
                        || Vec::<Flat<F>>::with_capacity(2 * num_phys),
                        |phys_row, (q_idx, &col_idx)| check_query(q_idx, col_idx, phys_row),
                    )
                    .try_reduce(|| true, |a, b| Ok(a && b))?
            } else {
                run_sequential(&random_indices)?
            }
        };

        #[cfg(not(feature = "parallel"))]
        let all_matched = run_sequential(&random_indices)?;

        if !all_matched {
            return Ok(false);
        }

        Ok(true)
    }
}

/// `q_flat = [q_data(grid_cols), q_support(ldt)]`. Layout must
/// match the prover's `rs_encode_grid`; `master_eval` reads `q_data`
/// alone, the support masks openings without entering the claim.
fn rs_encode_row<F: HardwareField + BinaryFieldExtras>(
    q_flat: &[Flat<F>],
    grid_cols: usize,
    config: &Config,
) -> errors::Result<Vec<Flat<F>>> {
    let geom = config.table_geom(grid_cols);
    let ldt = geom.support_size;
    let code_width = geom.encoded_width;

    let mut buf = vec![Flat::from_raw(F::ZERO); code_width];
    buf[..ldt].copy_from_slice(&q_flat[grid_cols..grid_cols + ldt]);
    buf[ldt..ldt + grid_cols].copy_from_slice(&q_flat[..grid_cols]);

    let fft = AdditiveFft::<F>::new(code_width.trailing_zeros());

    fft.forward_scalar(&mut buf)
        .map_err(|_| errors::Error::Protocol {
            protocol: "evaluator",
            message: "additive-FFT row encode failed",
        })?;

    Ok(buf)
}

fn build_tensor_table<F: HardwareField>(r: &[Flat<F>]) -> Vec<Flat<F>> {
    let one = Flat::from_raw(F::ONE);
    let mut table = vec![one];

    for &ri in r {
        let one_minus = one - ri;
        let n = table.len();

        let mut next = Vec::with_capacity(2 * n);

        for &v in &table {
            next.push(v * one_minus);
        }

        for &v in &table {
            next.push(v * ri);
        }

        table = next;
    }

    table
}

fn eq_tensor_b(r: &[Block128]) -> Vec<Block128> {
    let mut t = vec![Block128::ONE];
    for &ri in r {
        let len = t.len();
        let mut nt = Vec::with_capacity(len * 2);

        for &v in &t {
            nt.push(v * (Block128::ONE + ri));
        }

        for &v in &t {
            nt.push(v * ri);
        }

        t = nt;
    }

    t
}

fn transpose128(cols: &[Block128; NBITS]) -> [Block128; NBITS] {
    let mut rows = [Block128::ZERO; NBITS];
    for (v, cv) in cols.iter().enumerate() {
        for (u, ru) in rows.iter_mut().enumerate() {
            ru.0 |= ((cv.0 >> u) & 1) << v;
        }
    }

    rows
}

/// A(r') via the tensor algebra, without a materialized Dense A.
/// `e := eq~(phi0(P), phi1(r'))`;
/// `A(r') = Σ_u eq(r'',u)·e_row[u]`.
fn ring_switch_a_at(point: &[Block128], r_final: &[Block128], r_mix: &[Block128]) -> Block128 {
    let mut e = [Block128::ZERO; NBITS];
    e[0] = Block128::ONE;

    for (a, b) in point.iter().zip(r_final) {
        let mut col_scaled = e;
        for cv in col_scaled.iter_mut() {
            *cv *= *a;
        }

        let mut row_scaled = transpose128(&e);
        for ru in row_scaled.iter_mut() {
            *ru *= *b;
        }

        let row_scaled = transpose128(&row_scaled);

        for i in 0..NBITS {
            e[i] += col_scaled[i] + row_scaled[i];
        }
    }

    let e_rows = transpose128(&e);
    let eq_mix = eq_tensor_b(r_mix);

    let mut acc = Block128::ZERO;
    for u in 0..NBITS {
        acc += eq_mix[u] * e_rows[u];
    }

    acc
}

/// Σ_u eq(r'',u)·ŝ_u for one ring unit, ŝ_u = Σ_v bit_u(c_v)·2^v.
fn ring_batch_b(bit_claims: &[Block128], eq_mix: &[Block128]) -> Block128 {
    let mut acc = Block128::ZERO;
    for (u, &m) in eq_mix.iter().enumerate() {
        let mut shat = 0u128;
        for (v, cv) in bit_claims.iter().enumerate() {
            shat |= ((cv.0 >> u) & 1) << v;
        }

        acc += m * Block128(shat);
    }

    acc
}

/// Reconstructs the sumcheck's initial claim from the claimed
/// virtual evals, in the tower basis. Ring units contribute
/// `eta·Σ_u eq(r'',u) ŝ_u`; whole units contribute `eta·c'`.
fn ring_target<F>(
    plan: &RingSwitchPlan,
    claims: &[Flat<F>],
    eta_tower: F,
    r_mix: &[Block128],
    next_row: bool,
) -> Block128
where
    F: HardwareField + Into<Block128>,
{
    let variants = if next_row { 2 } else { 1 };
    let half = claims.len() / variants;
    let eq_mix = eq_tensor_b(r_mix);
    let eta: Block128 = eta_tower.into();

    let mut eta_pows = Vec::with_capacity(plan.num_units + 1);
    let mut e = Block128::ONE;

    for _ in 0..=plan.num_units {
        eta_pows.push(e);
        e *= eta;
    }

    let eta_shift = eta_pows[plan.num_units];

    let base = [(0usize, Block128::ONE)];
    let base_and_shift = [(0usize, Block128::ONE), (half, eta_shift)];
    let offsets: &[(usize, Block128)] = if next_row { &base_and_shift } else { &base };

    let mut target = Block128::ZERO;
    for &(offset, shift_mul) in offsets {
        let half_claims = &claims[offset..offset + half];

        let mut ci = 0usize;
        for (unit_idx, &(is_ring, num_claims)) in plan.units.iter().enumerate() {
            let weight = eta_pows[unit_idx] * shift_mul;
            if is_ring {
                let bits: Vec<Block128> = half_claims[ci..ci + num_claims]
                    .iter()
                    .map(|f| f.to_tower().into())
                    .collect();

                target += weight * ring_batch_b(&bits, &eq_mix);
            } else {
                let c: Block128 = half_claims[ci].to_tower().into();
                target += weight * c;
            }

            ci += num_claims;
        }
    }

    target
}

/// Parses one opened grid-row:
/// `base` per committed column, plus `shift` when `next_row`,
/// each at its `rs_field` width (sub-B32 columns are B32-wide).
fn parse_physical_row<F: TraceCompatibleField>(
    row_data: &[u8],
    phys_rs: &[ColumnType],
    out: &mut Vec<Flat<F>>,
    next_row: bool,
) {
    let mut ptr = 0;
    for ct in phys_rs {
        let sz = ct.byte_size();

        out.push(ct.parse_from_bytes(&row_data[ptr..ptr + sz]));
        ptr += sz;

        if next_row {
            out.push(ct.parse_from_bytes(&row_data[ptr..ptr + sz]));
            ptr += sz;
        }
    }
}
