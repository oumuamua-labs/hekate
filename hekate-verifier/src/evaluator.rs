// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use crate::brakedown::BrakedownVerifier;
use crate::sumcheck::verify;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;
use hekate_core::config::{Config, INV_RATE};
use hekate_core::errors;
use hekate_core::proofs::{BrakedownCommitment, EvalBatchProof};
use hekate_core::tensor::TensorProduct;
use hekate_core::trace::{ColumnType, TraceCompatibleField};
use hekate_crypto::Hasher;
use hekate_crypto::transcript::Transcript;
use hekate_math::{
    AdditiveFft, BinaryFieldExtras, Block128, Flat, HardwareField, PackableField, TowerField,
};
use hekate_program::expander::{RingSwitchPlan, eq_tensor_b, ring_target};
use tracing::{debug, instrument, warn};

#[cfg(feature = "parallel")]
const PARALLEL_PROXIMITY_THRESHOLD: usize = 1 << 18;

const NBITS: usize = 128;

pub struct EvaluatorVerifier<F, H: Hasher> {
    _marker: PhantomData<(F, H)>,
}

pub struct EvalVerifyContext<'a, F: HardwareField> {
    pub point: &'a [Flat<F>],
    pub claimed_values: &'a [Flat<F>],
    pub num_vars: usize,
    pub ring_plan: &'a RingSwitchPlan,

    /// `true` when the claims carry a next-row half,
    /// proven through the K_P / A_next weights.
    pub shifted_claims: bool,

    /// Masked claims: the ring-switch final
    /// check is left to the outer argument.
    pub masked: bool,

    /// The table's `h` tree, opened alongside the trace.
    pub h_commitment: Option<&'a BrakedownCommitment>,
}

/// What the outer argument needs from an accepted
/// eval sumcheck: `claim_n = fin` on plaintext values.
pub struct EvalOutcome<F: HardwareField> {
    pub eta: F,
    pub r_mix: Vec<Block128>,
    pub challenges: Vec<Flat<F>>,
    pub claim_masked: Flat<F>,
    pub fin: Flat<F>,
}

impl<F, H: Hasher> EvaluatorVerifier<F, H>
where
    F: HardwareField + PackableField + TraceCompatibleField,
{
    /// Verifies the ring-switch TensorPCS evaluation argument,
    /// binding the claimed virtual evals to the base-only
    /// Brakedown commitment. A degree-2 sumcheck reduces
    /// `[A + η^U·A_next]·master_bit + [Eq + η^U·K_P]·master_whole`
    /// to `r_final`; the final check pairs the transparent weight
    /// evals at `r'` with the two master evaluations, which the
    /// λ-line fold and the proximity test bind to the codewords.
    #[instrument(skip_all, name = "Evaluator::verify")]
    pub fn verify(
        commitment: &BrakedownCommitment,
        proof: &EvalBatchProof<F>,
        transcript: &mut Transcript<H>,
        ctx: EvalVerifyContext<'_, F>,
        config: &Config,
    ) -> errors::Result<Option<EvalOutcome<F>>>
    where
        F: BinaryFieldExtras + Into<Block128> + From<u128>,
    {
        let point = ctx.point;
        let claims = ctx.claimed_values;
        let num_vars = ctx.num_vars;
        let plan = ctx.ring_plan;
        let shifted_claims = ctx.shifted_claims;
        let claim_halves = if shifted_claims { 2 } else { 1 };
        let zero = Flat::from_raw(F::ZERO);

        if claims.len() != plan.total_claims() * claim_halves {
            return Err(errors::Error::Protocol {
                protocol: "evaluator_verifier",
                message: "ring-switch plan claim count does not match the claimed evaluations",
            });
        }

        let h_tree = match (
            ctx.h_commitment,
            proof.h_ldt_proof.as_ref(),
            plan.h_cols > 0,
        ) {
            (Some(h_comm), Some(h_opening), true) => Some((h_comm, h_opening)),
            (None, None, false) => None,
            _ => {
                return Err(errors::Error::Protocol {
                    protocol: "evaluator_verifier",
                    message: "h opening present iff the plan carries h columns",
                });
            }
        };

        transcript.append_message(b"eval_batch_start", b"");

        for &val in claims {
            transcript.append_field(b"claimed_val", val.to_tower());
        }

        let eta_tower = transcript.challenge_field::<F>(b"eval_eta")?;
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

        let target = ring_target::<F>(plan, claims, eta_tower, &r_mix, shifted_claims);
        let target_flat = F::from(target.0).to_hardware();

        let sc_res = verify(num_vars, 2, target_flat, &proof.sumcheck_proof, transcript)?;
        let (r_row, sumcheck_final_eval) = match sc_res {
            Some(res) => res,
            None => {
                warn!("Sumcheck failed");
                return Ok(None);
            }
        };

        // Both master evaluations precede λ
        let line = match (proof.master_evals.as_ref(), has_ring) {
            (Some(evals), true) => {
                transcript.append_field(b"eval_master_whole", evals.whole);
                transcript.append_field(b"eval_master_ring", evals.ring);

                let lambda = transcript
                    .challenge_field::<F>(b"eval_lambda")?
                    .to_hardware();

                Some((evals.whole.to_hardware(), evals.ring.to_hardware(), lambda))
            }
            (None, false) => None,
            _ => {
                return Err(errors::Error::Protocol {
                    protocol: "evaluator_verifier",
                    message: "master evaluations present iff the plan carries a ring unit",
                });
            }
        };

        let q = &proof.tensor_vec;

        transcript.append_field_list(b"tensor_q", q);

        let split_vars = plan.split_vars(num_vars, config);

        let grid_cols = 1 << split_vars;
        let grid_rows = 1 << (num_vars - split_vars);
        let geom = config.table_geom(grid_cols);
        let encoded_width = geom.encoded_width;
        let phys_row_bytes = plan.leaf_row_bytes();
        let h_row_bytes = plan.h_leaf_row_bytes();

        debug!(
            num_vars,
            split_vars,
            grid_cols,
            encoded_width,
            support = geom.support_size,
            row_bytes = phys_row_bytes,
            h_row_bytes,
            fractional = encoded_width == grid_cols * INV_RATE,
            "table geometry"
        );

        if grid_cols + geom.support_size > encoded_width {
            warn!("support + data message exceeds the codeword width");
            return Ok(None);
        }

        config.check_security(size_of::<F>() * 8, grid_cols)?;

        let expected_len = grid_cols + geom.support_size;

        if q.len() != expected_len {
            warn!("tensor_q length mismatch");
            return Ok(None);
        }

        let q_flat: Vec<Flat<F>> = q.iter().map(|v| v.to_hardware()).collect();
        let q_encoded = rs_encode_row::<F>(&q_flat, grid_cols, config)?;

        let r_col_low = &r_row[..split_vars];
        let tensor_col = build_tensor_table::<F>(r_col_low);

        let mut q_eval = zero;
        for (&val, &t) in q_flat.iter().take(grid_cols).zip(&tensor_col) {
            q_eval += val * t;
        }

        let (master_whole_eval, master_bit_eval) = match line {
            Some((whole, ring, lambda)) => {
                if q_eval != ring + lambda * whole {
                    warn!("line fold disagrees with the master evaluations");
                    return Ok(None);
                }

                (whole, ring)
            }
            None => (q_eval, zero),
        };

        let (coeff_bit, coeff_whole, eta_shift) = plan.column_coeffs::<F>(eta);

        let coeff_line = match line {
            Some((_, _, lambda)) => coeff_bit
                .iter()
                .zip(&coeff_whole)
                .map(|(&bit, &whole)| bit + lambda * whole)
                .collect(),
            None => coeff_whole,
        };

        // The weight evals at r' are transparent; the two
        // master evals are bound by the proximity check below.
        let (whole_weight_at_r, ring_weight_at_r) =
            master_weights_at::<F>(point, &r_row, &r_mix, eta_shift, has_ring, shifted_claims);

        let fin = ring_weight_at_r * master_bit_eval + whole_weight_at_r * master_whole_eval;

        if !ctx.masked && sumcheck_final_eval != fin {
            warn!("ring-switch final check failed");
            return Ok(None);
        }

        transcript.append_message(b"eval_batch_ldt", b"");

        let queries = BrakedownVerifier::<F, H>::draw_queries(transcript, config, split_vars)?;
        let opened_columns =
            BrakedownVerifier::<F, H>::verify_opening(commitment, &proof.ldt_proof, &queries)?;

        let h_opened = match h_tree {
            Some((h_commitment, h_opening)) => Some(BrakedownVerifier::<F, H>::verify_opening(
                h_commitment,
                h_opening,
                &queries,
            )?),
            None => None,
        };

        let slot_map = &queries.slot_map;
        let random_indices = &queries.indices;
        let h_rs = vec![ColumnType::B128; plan.h_cols];

        let r_row_high = &r_row[split_vars..];
        let tensor_row = TensorProduct::<F>::new(r_row_high.to_vec());

        let mut tensor_row_evals = Vec::with_capacity(grid_rows);
        for r in 0..grid_rows {
            tensor_row_evals.push(tensor_row.evaluate_at_index(r));
        }

        let num_fold_cols = plan.phys_rs.len() + plan.h_cols;

        // Re-derive the folded opening from the physical columns in the
        // opened leaf; RS commutes with a whole-column fold, this must
        // match the RS re-encoding of the prover's committed q vector.
        let check_query =
            |q_idx: usize, col_idx: usize, phys_row: &mut Vec<Flat<F>>| -> errors::Result<bool> {
                let col_bytes = &opened_columns[slot_map[q_idx]];

                if col_bytes.len() != grid_rows * phys_row_bytes {
                    warn!("opened column length does not match the physical row layout");
                    return Ok(false);
                }

                let h_bytes = h_opened.map(|h| &h[slot_map[q_idx]]);

                if h_bytes.is_some_and(|h| h.len() != grid_rows * h_row_bytes) {
                    warn!("opened h column length does not match the h row layout");
                    return Ok(false);
                }

                let mut q_val = zero;

                for r in 0..grid_rows {
                    let row_data = &col_bytes[r * phys_row_bytes..(r + 1) * phys_row_bytes];

                    phys_row.clear();

                    parse_physical_row::<F>(row_data, &plan.phys_rs, phys_row);

                    if let Some(h) = h_bytes {
                        let h_row = &h[r * h_row_bytes..(r + 1) * h_row_bytes];
                        parse_physical_row::<F>(h_row, &h_rs, phys_row);
                    }

                    let mut fold = zero;
                    for (&base, &coeff) in phys_row.iter().zip(&coeff_line) {
                        fold += base * coeff;
                    }

                    q_val += fold * tensor_row_evals[r];
                }

                let ok = q_val == q_encoded[col_idx];

                if !ok {
                    warn!("TensorPCS proximity mismatch for column {}", col_idx);
                }

                Ok(ok)
            };

        let run_sequential = |indices: &[usize]| -> errors::Result<bool> {
            let mut phys_row = Vec::with_capacity(num_fold_cols);
            for (q_idx, &col_idx) in indices.iter().enumerate() {
                if !check_query(q_idx, col_idx, &mut phys_row)? {
                    return Ok(false);
                }
            }

            Ok(true)
        };

        #[cfg(feature = "parallel")]
        let all_matched = {
            let proximity_work = config.num_queries * grid_rows * 2 * num_fold_cols;

            if proximity_work >= PARALLEL_PROXIMITY_THRESHOLD {
                use rayon::prelude::*;

                random_indices
                    .par_iter()
                    .enumerate()
                    .map_init(
                        || Vec::<Flat<F>>::with_capacity(num_fold_cols),
                        |phys_row, (q_idx, &col_idx)| check_query(q_idx, col_idx, phys_row),
                    )
                    .try_reduce(|| true, |a, b| Ok(a && b))?
            } else {
                run_sequential(random_indices)?
            }
        };

        #[cfg(not(feature = "parallel"))]
        let all_matched = run_sequential(random_indices)?;

        if !all_matched {
            return Ok(None);
        }

        Ok(Some(EvalOutcome {
            eta: eta_tower,
            r_mix,
            challenges: r_row,
            claim_masked: sumcheck_final_eval,
            fin,
        }))
    }
}

/// `q_flat = [q_data(grid_cols), q_support(ldt)]`. Layout must
/// match the prover's `rs_encode_grid`; `q_eval` reads `q_data`
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

fn transpose128(cols: &[Block128; NBITS]) -> [Block128; NBITS] {
    let mut m = *cols;
    let mut j = NBITS / 2;
    let mut mask = (1u128 << (NBITS / 2)) - 1;

    while j != 0 {
        let mut k = 0;

        while k < NBITS {
            for i in k..k + j {
                let a = m[i].0;
                let b = m[i + j].0;
                let t = ((a >> j) ^ b) & mask;

                m[i] = Block128(a ^ (t << j));
                m[i + j] = Block128(b ^ t);
            }

            k += j << 1;
        }

        j >>= 1;
        mask ^= mask << j;
    }

    m
}

/// `(whole, ring)` master weights at `r'`, pairing with
/// `master_whole` and `master_bit` in the final check.
fn master_weights_at<F>(
    point: &[Flat<F>],
    r_row: &[Flat<F>],
    r_mix: &[Block128],
    eta_shift: Flat<F>,
    has_ring: bool,
    shifted_claims: bool,
) -> (Flat<F>, Flat<F>)
where
    F: HardwareField + Into<Block128> + From<u128>,
{
    let zero = Flat::from_raw(F::ZERO);
    let eq_at_r = TensorProduct::evaluate_eq_slice(point, r_row);

    if !has_ring && !shifted_claims {
        return (eq_at_r, zero);
    }

    let point_b: Vec<Block128> = point.iter().map(|f| f.to_tower().into()).collect();
    let r_row_b: Vec<Block128> = r_row.iter().map(|f| f.to_tower().into()).collect();

    let (a, a_next) = match has_ring {
        true => ring_switch_a_pair(&point_b, &r_row_b, r_mix, shifted_claims),
        false => (Block128::ZERO, Block128::ZERO),
    };

    let a_r = F::from(a.0).to_hardware();

    match shifted_claims {
        true => {
            let k_p_r = F::from(k_p_at(&point_b, &r_row_b).0).to_hardware();
            let a_next_r = F::from(a_next.0).to_hardware();

            (eq_at_r + eta_shift * k_p_r, a_r + eta_shift * a_next_r)
        }
        false => (eq_at_r, a_r),
    }
}

/// `Ã(r')` and the `K_P` carry chain `Ã_next(r')`.
/// Reverse iteration is safe: the per-variable
/// operators `I + L_{P_k} + R_{r'_k}` commute pairwise.
fn ring_switch_a_pair(
    point: &[Block128],
    r_final: &[Block128],
    r_mix: &[Block128],
    with_next: bool,
) -> (Block128, Block128) {
    let n = point.len();
    let one = Block128::ONE;

    let mut p_pref = vec![one; n + 1];
    let mut q_pref = vec![one; n + 1];

    for i in 0..n {
        p_pref[i + 1] = p_pref[i] * point[i];
        q_pref[i + 1] = q_pref[i] * (one + r_final[i]);
    }

    let mut e = [Block128::ZERO; NBITS];
    let mut h = [Block128::ZERO; NBITS];

    e[0] = one;

    for k in (0..n).rev() {
        if with_next {
            let alpha = p_pref[k] * (one + point[k]);
            let beta = q_pref[k] * r_final[k];

            let mut m = e;
            for cv in m.iter_mut() {
                *cv *= alpha;
            }

            let mut m_rows = transpose128(&m);
            for ru in m_rows.iter_mut() {
                *ru *= beta;
            }

            for (hv, mv) in h.iter_mut().zip(transpose128(&m_rows).iter()) {
                *hv += *mv;
            }
        }

        let mut col_scaled = e;
        for cv in col_scaled.iter_mut() {
            *cv *= point[k];
        }

        let mut row_scaled = transpose128(&e);
        for ru in row_scaled.iter_mut() {
            *ru *= r_final[k];
        }

        let row_scaled = transpose128(&row_scaled);

        for i in 0..NBITS {
            e[i] += col_scaled[i] + row_scaled[i];
        }
    }

    if with_next {
        let wrap_col = p_pref[n];
        let wrap_row = q_pref[n];

        for (v, hv) in h.iter_mut().enumerate() {
            if (wrap_row.0 >> v) & 1 == 1 {
                *hv += wrap_col;
            }
        }
    }

    let eq_mix = eq_tensor_b(r_mix);
    let e_rows = transpose128(&e);
    let h_rows = transpose128(&h);

    let mut a = Block128::ZERO;
    let mut a_next = Block128::ZERO;

    for u in 0..NBITS {
        a += eq_mix[u] * e_rows[u];
        a_next += eq_mix[u] * h_rows[u];
    }

    (a, a_next)
}

/// Parses one opened grid-row: one symbol per committed column
/// at its `rs_field` width (sub-B32 columns are B32-wide).
fn parse_physical_row<F: TraceCompatibleField>(
    row_data: &[u8],
    phys_rs: &[ColumnType],
    out: &mut Vec<Flat<F>>,
) {
    let mut ptr = 0;
    for ct in phys_rs {
        let sz = ct.byte_size();

        out.push(ct.parse_from_bytes(&row_data[ptr..ptr + sz]));
        ptr += sz;
    }
}

/// K̃_P(r') in O(n): the K_P carry chain
/// `Σ_k Π_{i<k}(1+r'_i)P_i · r'_k(1+P_k) · Π_{i>k}(1+r'_i+P_i)`
/// plus the cyclic wrap `Π_i (1+r'_i)P_i`.
fn k_p_at(point: &[Block128], r_final: &[Block128]) -> Block128 {
    let n = point.len();
    let one = Block128::ONE;

    let mut suffix = vec![one; n + 1];
    for i in (0..n).rev() {
        suffix[i] = suffix[i + 1] * (one + r_final[i] + point[i]);
    }

    let mut acc = Block128::ZERO;
    let mut prefix = one;

    for k in 0..n {
        acc += prefix * r_final[k] * (one + point[k]) * suffix[k + 1];
        prefix *= (one + r_final[k]) * point[k];
    }

    acc + prefix
}

#[cfg(test)]
mod tests {
    use super::*;
    use hekate_core::poly::PolyVariant;
    use hekate_math::TowerField;

    const GRID_COLS: usize = 1024;

    fn elems(seed: u128, n: usize) -> Vec<Block128> {
        let g = Block128(0x2545F4914F6CDD1D_517CC1B727220A95);

        let mut x = Block128(seed | 1);
        let mut out = Vec::with_capacity(n);

        for _ in 0..n {
            x = x * g + Block128::ONE;
            out.push(x);
        }

        out
    }

    fn rs_row(seed: u128, len: usize) -> Vec<Flat<Block128>> {
        elems(seed, len).iter().map(|v| v.to_hardware()).collect()
    }

    fn k_p_on_cube(point: &[Block128]) -> Vec<Block128> {
        let eq_p = eq_tensor_b(point);
        let n = eq_p.len();

        (0..n).map(|i| eq_p[(i + n - 1) & (n - 1)]).collect()
    }

    fn mle_at(values: &[Block128], r: &[Block128]) -> Block128 {
        let weights = eq_tensor_b(r);

        values
            .iter()
            .zip(weights.iter())
            .fold(Block128::ZERO, |acc, (&v, &w)| acc + v * w)
    }

    fn contract_bits(values: &[Block128], eq_mix: &[Block128]) -> Vec<Block128> {
        values
            .iter()
            .map(|v| {
                let mut acc = Block128::ZERO;
                for (u, &m) in eq_mix.iter().enumerate() {
                    if (v.0 >> u) & 1 == 1 {
                        acc += m;
                    }
                }

                acc
            })
            .collect()
    }

    /// `K_P` is only the right weight if its wrap agrees
    /// with `PolyVariant::Shifted`, which is what the AIR
    /// and the prover's fold read as the next row.
    #[test]
    fn k_p_contracts_to_the_polyvariant_next_row() {
        for num_vars in 1..=5 {
            let n = 1usize << num_vars;

            let point = elems(1289 + num_vars as u128, num_vars);
            let col_tower = elems(53 + num_vars as u128, n);
            let col: Vec<Flat<Block128>> = col_tower.iter().map(|v| v.to_hardware()).collect();

            let shifted = PolyVariant::Shifted(&col);
            let eq_p = eq_tensor_b(&point);
            let k_p = k_p_on_cube(&point);

            let mut via_variant = Block128::ZERO;
            let mut via_k_p = Block128::ZERO;

            for i in 0..n {
                via_variant += shifted.get_at(i).to_tower() * eq_p[i];
                via_k_p += col_tower[i] * k_p[i];
            }

            assert_eq!(via_variant, via_k_p, "n={num_vars}");
        }
    }

    #[test]
    fn k_p_closed_form_matches_materialized() {
        for num_vars in 1..=5 {
            let point = elems(3 + num_vars as u128, num_vars);
            let r_final = elems(101 + num_vars as u128, num_vars);

            let direct = mle_at(&k_p_on_cube(&point), &r_final);

            assert_eq!(k_p_at(&point, &r_final), direct, "n={num_vars}");
        }
    }

    #[test]
    fn a_closed_forms_match_materialized() {
        let kappa = NBITS.ilog2() as usize;

        for num_vars in 1..=5 {
            let point = elems(7 + num_vars as u128, num_vars);
            let r_final = elems(211 + num_vars as u128, num_vars);
            let r_mix = elems(919 + num_vars as u128, kappa);

            let eq_mix = eq_tensor_b(&r_mix);
            let a_cube = contract_bits(&eq_tensor_b(&point), &eq_mix);
            let a_next_cube = contract_bits(&k_p_on_cube(&point), &eq_mix);

            let (a, a_next) = ring_switch_a_pair(&point, &r_final, &r_mix, true);

            assert_eq!(a, mle_at(&a_cube, &r_final), "A n={num_vars}");
            assert_eq!(
                a_next,
                mle_at(&a_next_cube, &r_final),
                "A_next n={num_vars}"
            );

            let (a_only, skipped) = ring_switch_a_pair(&point, &r_final, &r_mix, false);

            assert_eq!(a_only, a, "A without the carry chain n={num_vars}");
            assert_eq!(skipped, Block128::ZERO, "n={num_vars}");
        }
    }

    #[test]
    fn transpose128_swaps_bit_indices() {
        let mut cols = [Block128::ZERO; NBITS];
        for (c, v) in cols.iter_mut().zip(elems(0x51, NBITS)) {
            *c = v;
        }

        let mut expected = [Block128::ZERO; NBITS];
        for (v, cv) in cols.iter().enumerate() {
            for (u, ru) in expected.iter_mut().enumerate() {
                ru.0 |= ((cv.0 >> u) & 1) << v;
            }
        }

        let rows = transpose128(&cols);

        assert_eq!(rows, expected);
        assert_eq!(transpose128(&rows), cols);
    }

    /// The proximity fold commutes with the encode
    /// only while `rs_encode_row` stays linear
    /// over its own support / data split.
    #[test]
    fn rs_encode_row_is_linear() {
        let config = Config::prod();
        let len = GRID_COLS + config.table_geom(GRID_COLS).support_size;

        let a = rs_row(0x11, len);
        let b = rs_row(0x22, len);
        let sum: Vec<Flat<Block128>> = a.iter().zip(&b).map(|(x, y)| *x + *y).collect();

        let ea = rs_encode_row::<Block128>(&a, GRID_COLS, &config).unwrap();
        let eb = rs_encode_row::<Block128>(&b, GRID_COLS, &config).unwrap();
        let es = rs_encode_row::<Block128>(&sum, GRID_COLS, &config).unwrap();

        for ((x, y), s) in ea.iter().zip(&eb).zip(&es) {
            assert_eq!(*x + *y, *s);
        }
    }

    #[test]
    fn rs_encode_row_meets_singleton_bound() {
        let config = Config::prod();
        let len = GRID_COLS + config.table_geom(GRID_COLS).support_size;

        let code = rs_encode_row::<Block128>(&rs_row(0x33, len), GRID_COLS, &config).unwrap();

        let zeros = code
            .iter()
            .filter(|v| **v == Flat::from_raw(Block128::ZERO))
            .count();

        assert!(zeros < len);
    }

    /// A drift in `CantorBasis` or `AdditiveFft` silently
    /// changes the code the proximity bound is stated over.
    #[test]
    fn rs_encode_row_realises_cantor_subspace_chain() {
        let config = Config::prod();
        let geom = config.table_geom(GRID_COLS);
        let ldt = geom.support_size;
        let len = GRID_COLS + ldt;

        let zero = Flat::from_raw(Block128::ZERO);

        for j in 0..=len.ilog2() as usize {
            let at = 1usize << j;
            let slot = match at < ldt {
                true => GRID_COLS + at,
                false => at - ldt,
            };

            let mut row = vec![zero; len];
            row[slot] = Flat::from_raw(Block128::ONE);

            let code = rs_encode_row::<Block128>(&row, GRID_COLS, &config).unwrap();

            let zeros: Vec<usize> = (0..geom.encoded_width)
                .filter(|&x| code[x] == zero)
                .collect();

            assert_eq!(zeros, (0..at).collect::<Vec<usize>>(), "s_{j}");
        }
    }
}
