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
use hekate_core::proofs::EvalBatchProof;
use hekate_core::tensor::TensorProduct;
use hekate_core::trace::{ColumnType, TraceCompatibleField};
use hekate_crypto::Hasher;
use hekate_crypto::transcript::Transcript;
use hekate_math::{
    BinaryFieldExtras, Block128, CantorBasis, Flat, HardwareField, PackableField, TowerField,
};
use hekate_program::expander::{PoolLayout, eq_tensor_b, ring_target};
use tracing::{debug, instrument, trace_span, warn};

#[cfg(feature = "parallel")]
const PARALLEL_PROXIMITY_THRESHOLD: usize = 1 << 18;

const NBITS: usize = 128;

pub struct EvaluatorVerifier<F, H: Hasher> {
    _marker: PhantomData<(F, H)>,
}

/// Claimed evaluations as the proof carries them, and their
/// flat form derived from them in `new`: the transcript
/// absorbs exactly the values the checks read.
pub struct Claims<'a, F> {
    canonical: &'a [F],
    flat: Vec<Flat<F>>,
}

impl<'a, F: HardwareField> Claims<'a, F> {
    pub fn new(canonical: &'a [F]) -> Self {
        Self {
            canonical,
            flat: canonical.iter().map(|c| c.to_hardware()).collect(),
        }
    }

    pub fn canonical(&self) -> &'a [F] {
        self.canonical
    }

    pub fn flat(&self) -> &[Flat<F>] {
        &self.flat
    }
}

pub struct TableClaims<'a, F> {
    pub point: &'a [Flat<F>],
    pub claims: &'a Claims<'a, F>,
}

pub struct EvalVerifyContext<'a, F: HardwareField> {
    pub pool: &'a PoolLayout,
    pub tables: &'a [TableClaims<'a, F>],

    /// `true` when the claims carry a next-row half,
    /// proven through the K_P / A_next weights.
    pub shifted_claims: bool,

    /// Masked claims: the ring-switch final
    /// check is left to the outer argument.
    pub masked: bool,
}

/// What the outer argument needs from an accepted
/// eval sumcheck: `claim_n = fin` on plaintext values.
pub struct EvalOutcome<F: HardwareField> {
    pub eta: F,
    pub rho: Vec<F>,
    pub r_mix: Vec<Block128>,
    pub challenges: Vec<Flat<F>>,
    pub claim_masked: Flat<F>,
    pub fin: Flat<F>,
}

impl<F, H: Hasher> EvaluatorVerifier<F, H>
where
    F: HardwareField + PackableField + TraceCompatibleField,
{
    /// Binds every pooled table's claimed virtual evals to
    /// the two roots: one degree-2 sumcheck to `r`, each table's
    /// masters pinned by the λ line and the proximity test.
    #[instrument(skip_all, level = "trace", name = "Evaluator::verify")]
    pub fn verify(
        trace_root: &[u8; 32],
        h_root: Option<&[u8; 32]>,
        proof: &EvalBatchProof<F>,
        transcript: &mut Transcript<H>,
        ctx: EvalVerifyContext<'_, F>,
        config: &Config,
    ) -> errors::Result<Option<EvalOutcome<F>>>
    where
        F: BinaryFieldExtras + Into<Block128> + From<u128>,
    {
        let pool = ctx.pool;
        let shifted_claims = ctx.shifted_claims;
        let claim_halves = if shifted_claims { 2 } else { 1 };

        let zero = Flat::from_raw(F::ZERO);
        let one = Flat::from_raw(F::ONE);

        if ctx.tables.len() != pool.tables.len() {
            return Err(errors::Error::Protocol {
                protocol: "evaluator_verifier",
                message: "one claim set per pooled table is required",
            });
        }

        for (table, layout) in ctx.tables.iter().zip(&pool.tables) {
            if table.claims.flat().len() != layout.plan.total_claims() * claim_halves {
                return Err(errors::Error::Protocol {
                    protocol: "evaluator_verifier",
                    message: "ring-switch plan claim count does not match the claimed evaluations",
                });
            }
        }

        let has_h = pool.tables.iter().any(|t| t.plan.h_cols > 0);

        let h_tree = match (h_root, proof.h_ldt_proof.as_ref(), has_h) {
            (Some(root), Some(h_opening), true) => Some((root, h_opening)),
            (None, None, false) => None,
            _ => {
                return Err(errors::Error::Protocol {
                    protocol: "evaluator_verifier",
                    message: "h opening present iff the plan carries h columns",
                });
            }
        };

        transcript.append_message(b"eval_batch_start", b"");

        let eta_tower = transcript.challenge_field::<F>(b"eval_eta")?;
        let eta = eta_tower.to_hardware();

        let mut rho = Vec::with_capacity(pool.rho_vars());
        for _ in 0..pool.rho_vars() {
            rho.push(transcript.challenge_field::<F>(b"eval_rho")?);
        }

        let has_ring = pool.tables.iter().any(|t| t.plan.has_ring());

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

        let mut target = Block128::ZERO;
        for (table, layout) in ctx.tables.iter().zip(&pool.tables) {
            let rho = &rho[..layout.pack_vars()];

            target += ring_target::<F>(
                layout,
                table.claims.flat(),
                eta_tower,
                rho,
                &r_mix,
                shifted_claims,
            )?;
        }

        let target_flat = F::from(target.0).to_hardware();
        let num_vars = pool.num_vars();

        let sc_res = verify(num_vars, 2, target_flat, &proof.sumcheck_proof, transcript)?;

        let (r, sumcheck_final_eval) = match sc_res {
            Some(res) => res,
            None => {
                warn!("Sumcheck failed");
                return Ok(None);
            }
        };

        if proof.masters.len() != pool.num_masters() {
            return Err(errors::Error::Protocol {
                protocol: "evaluator_verifier",
                message: "one master evaluation per pooled master is required",
            });
        }

        // Every master evaluation precedes λ
        transcript.append_field_list(b"eval_masters", &proof.masters);

        let lambda = match proof.masters.len() > 1 {
            true => transcript
                .challenge_field::<F>(b"eval_lambda")?
                .to_hardware(),
            false => one,
        };

        let masters: Vec<Flat<F>> = proof.masters.iter().map(|m| m.to_hardware()).collect();

        let q = &proof.tensor_vec;

        transcript.append_field_list(b"tensor_q", q);

        let field_bits = F::BITS;
        let split_vars = pool.split_vars;
        let grid_cols = pool.grid_cols();
        let geom = config.table_geom(grid_cols);
        let encoded_width = geom.encoded_width;
        let shape = pool.fold_shape();

        debug!(
            num_vars,
            split_vars,
            tables = pool.tables.len(),
            grid_cols,
            encoded_width,
            support = geom.support_size,
            fractional = encoded_width == grid_cols * INV_RATE,
            "pool geometry"
        );

        debug!(
            bits = config.estimated_security_bits(field_bits, shape),
            ldt_query = config.security_metrics(field_bits, shape).ldt_bits,
            fold_gap = config.proximity_gap_bits(field_bits, shape),
            units = shape.units,
            "pool security"
        );

        if grid_cols + geom.support_size > encoded_width {
            warn!("support + data message exceeds the codeword width");
            return Ok(None);
        }

        config.check_security(field_bits, shape)?;

        let expected_len = grid_cols + geom.support_size;

        if q.len() != expected_len {
            warn!("tensor_q length mismatch");
            return Ok(None);
        }

        let q_flat: Vec<Flat<F>> = q.iter().map(|v| v.to_hardware()).collect();

        let tensor_col = build_tensor_table::<F>(&r[..split_vars]);

        let mut q_eval = zero;
        for (&val, &t) in q_flat.iter().take(grid_cols).zip(&tensor_col) {
            q_eval += val * t;
        }

        let mut lambda_pows = Vec::with_capacity(masters.len());
        let mut line = zero;
        let mut lambda_pow = one;

        for &master in &masters {
            lambda_pows.push(lambda_pow);
            line += lambda_pow * master;
            lambda_pow *= lambda;
        }

        if q_eval != line {
            warn!("line fold disagrees with the master evaluations");
            return Ok(None);
        }

        let mut coeff_lines: Vec<Vec<Flat<F>>> = Vec::with_capacity(pool.tables.len());
        let mut fin = zero;
        let mut k = 0;

        for (table, layout) in ctx.tables.iter().zip(&pool.tables) {
            let ring = layout.plan.has_ring();
            let n = layout.num_vars;

            let (lambda_bit, master_bit) = match ring {
                true => (lambda_pows[k], masters[k]),
                false => (zero, zero),
            };

            k += usize::from(ring);

            let (lambda_whole, master_whole) = (lambda_pows[k], masters[k]);

            k += 1;

            let (coeff_bit, coeff_whole, eta_shift) = layout.slot_coeffs::<F>(eta);

            coeff_lines.push(
                coeff_bit
                    .iter()
                    .zip(&coeff_whole)
                    .map(|(&bit, &whole)| lambda_bit * bit + lambda_whole * whole)
                    .collect(),
            );

            // The weight evals at r' are transparent; the two
            // master evals are bound by the proximity check below.
            let (whole_weight, ring_weight) = master_weights_at::<F>(
                table.point,
                &r[..n],
                &r_mix,
                eta_shift,
                ring,
                shifted_claims,
            );

            let rho_table: Vec<Flat<F>> = rho[..layout.pack_vars()]
                .iter()
                .map(|x| x.to_hardware())
                .collect();

            let blocks =
                TensorProduct::evaluate_eq_slice(&rho_table, &r[n..n + layout.pack_vars()]);

            fin += blocks * (ring_weight * master_bit + whole_weight * master_whole);
        }

        if !ctx.masked && sumcheck_final_eval != fin {
            warn!("ring-switch final check failed");
            return Ok(None);
        }

        transcript.append_message(b"eval_batch_ldt", b"");

        let queries = BrakedownVerifier::<F, H>::draw_queries(transcript, config, split_vars)?;

        let trace_parts: Vec<usize> = pool
            .tables
            .iter()
            .map(|t| t.grid_rows() * t.leaf_row_bytes())
            .collect();

        let opened_columns = BrakedownVerifier::<F, H>::verify_opening(
            trace_root,
            &proof.ldt_proof,
            &queries,
            &trace_parts,
        )?;

        let h_parts: Vec<usize> = pool
            .tables
            .iter()
            .filter(|t| t.h_slots > 0)
            .map(|t| t.grid_rows() * t.h_leaf_row_bytes())
            .collect();

        let h_opened = match h_tree {
            Some((root, h_opening)) => Some(BrakedownVerifier::<F, H>::verify_opening(
                root, h_opening, &queries, &h_parts,
            )?),
            None => None,
        };

        let _proximity = trace_span!("proximity").entered();

        let q_encoded = rs_encode_at::<F>(&q_flat, grid_cols, config, &queries.distinct)?;

        let slot_map = &queries.slot_map;
        let random_indices = &queries.indices;

        let h_rs: Vec<Vec<ColumnType>> = pool
            .tables
            .iter()
            .map(|t| vec![ColumnType::B128; t.h_slots])
            .collect();

        let tensor_row_evals = build_tensor_table::<F>(&r[split_vars..]);

        let num_fold_cols = pool
            .tables
            .iter()
            .map(|t| t.slot_rs.len() + t.h_slots)
            .max()
            .unwrap_or(0);

        // Re-derive the folded opening from the slots in the
        // opened leaf; RS commutes with a whole-column fold, this must
        // match the RS re-encoding of the prover's committed q vector.
        let check_query =
            |q_idx: usize, col_idx: usize, phys_row: &mut Vec<Flat<F>>| -> errors::Result<bool> {
                let col_bytes = opened_columns[slot_map[q_idx]].as_slice();
                let h_bytes = h_opened.map(|h| h[slot_map[q_idx]].as_slice());

                let mut q_val = zero;
                let mut at = 0;
                let mut h_at = 0;

                for ((layout, coeff_line), h_rs) in pool.tables.iter().zip(&coeff_lines).zip(&h_rs)
                {
                    let row_bytes = layout.leaf_row_bytes();
                    let h_row_bytes = layout.h_leaf_row_bytes();
                    let rows = layout.grid_rows();

                    for (g, &eq_g) in tensor_row_evals.iter().take(rows).enumerate() {
                        let row = at + g * row_bytes;

                        phys_row.clear();

                        parse_physical_row::<F>(
                            &col_bytes[row..row + row_bytes],
                            &layout.slot_rs,
                            phys_row,
                        );

                        if let Some(h) = h_bytes {
                            let h_row = h_at + g * h_row_bytes;
                            parse_physical_row::<F>(&h[h_row..h_row + h_row_bytes], h_rs, phys_row);
                        }

                        let mut fold = zero;
                        for (&base, &coeff) in phys_row.iter().zip(coeff_line) {
                            fold += base * coeff;
                        }

                        q_val += fold * eq_g;
                    }

                    at += rows * row_bytes;
                    h_at += rows * h_row_bytes;
                }

                let ok = q_val == q_encoded[slot_map[q_idx]];

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
            let fold_cells: usize = pool
                .tables
                .iter()
                .map(|t| t.grid_rows() * (t.slot_rs.len() + t.h_slots))
                .sum();

            let proximity_work = config.num_queries * 2 * fold_cells;

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
            rho,
            r_mix,
            challenges: r,
            claim_masked: sumcheck_final_eval,
            fin,
        }))
    }
}

/// `q_flat = [q_data(grid_cols), q_support(ldt)]`. Layout must
/// match the prover's `rs_encode_grid`; `q_eval` reads `q_data`
/// alone, the support masks openings without entering the claim.
fn rs_encode_at<F: HardwareField + BinaryFieldExtras>(
    q_flat: &[Flat<F>],
    grid_cols: usize,
    config: &Config,
    columns: &[usize],
) -> errors::Result<Vec<Flat<F>>> {
    let geom = config.table_geom(grid_cols);
    let ldt = geom.support_size;
    let zero = Flat::from_raw(F::ZERO);

    let len = (ldt + grid_cols).next_power_of_two();

    let mut coeffs = vec![zero; len];
    coeffs[..ldt].copy_from_slice(&q_flat[grid_cols..grid_cols + ldt]);
    coeffs[ldt..ldt + grid_cols].copy_from_slice(&q_flat[..grid_cols]);

    let encode_failed = |_| errors::Error::Protocol {
        protocol: "evaluator",
        message: "additive-FFT row encode failed",
    };

    let basis = CantorBasis::<F>::new(geom.encoded_width.trailing_zeros() as usize)
        .map_err(encode_failed)?;

    let mut scratch = vec![zero; len - 1];
    let mut out = vec![zero; columns.len()];

    basis
        .evaluate_at(&coeffs, zero, columns, &mut scratch, &mut out)
        .map_err(encode_failed)?;

    Ok(out)
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
    use hekate_math::{AdditiveFft, TowerField};

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

    fn all_columns(config: &Config, grid_cols: usize) -> Vec<usize> {
        (0..config.table_geom(grid_cols).encoded_width).collect()
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

    /// The proximity fold commutes with the encode only while
    /// `rs_encode_at` stays linear over its own support / data split.
    #[test]
    fn rs_encode_at_is_linear() {
        let config = Config::prod();
        let len = GRID_COLS + config.table_geom(GRID_COLS).support_size;
        let all = all_columns(&config, GRID_COLS);

        let a = rs_row(0x11, len);
        let b = rs_row(0x22, len);
        let sum: Vec<Flat<Block128>> = a.iter().zip(&b).map(|(x, y)| *x + *y).collect();

        let ea = rs_encode_at::<Block128>(&a, GRID_COLS, &config, &all).unwrap();
        let eb = rs_encode_at::<Block128>(&b, GRID_COLS, &config, &all).unwrap();
        let es = rs_encode_at::<Block128>(&sum, GRID_COLS, &config, &all).unwrap();

        for ((x, y), s) in ea.iter().zip(&eb).zip(&es) {
            assert_eq!(*x + *y, *s);
        }
    }

    #[test]
    fn rs_encode_at_meets_singleton_bound() {
        let config = Config::prod();
        let len = GRID_COLS + config.table_geom(GRID_COLS).support_size;
        let all = all_columns(&config, GRID_COLS);

        let code = rs_encode_at::<Block128>(&rs_row(0x33, len), GRID_COLS, &config, &all).unwrap();

        let zeros = code
            .iter()
            .filter(|v| **v == Flat::from_raw(Block128::ZERO))
            .count();

        assert!(zeros < len);
    }

    /// A drift in `CantorBasis` or `AdditiveFft` silently
    /// changes the code the proximity bound is stated over.
    #[test]
    fn rs_encode_at_realises_cantor_subspace_chain() {
        let config = Config::prod();
        let geom = config.table_geom(GRID_COLS);
        let ldt = geom.support_size;
        let len = GRID_COLS + ldt;
        let all = all_columns(&config, GRID_COLS);

        let zero = Flat::from_raw(Block128::ZERO);

        let mut at = 1usize;
        while at < len {
            let slot = match at < ldt {
                true => GRID_COLS + at,
                false => at - ldt,
            };

            let mut row = vec![zero; len];
            row[slot] = Flat::from_raw(Block128::ONE);

            let code = rs_encode_at::<Block128>(&row, GRID_COLS, &config, &all).unwrap();

            let zeros: Vec<usize> = (0..geom.encoded_width)
                .filter(|&x| code[x] == zero)
                .collect();

            assert_eq!(zeros, (0..at).collect::<Vec<usize>>(), "s_{}", at.ilog2());

            at <<= 1;
        }
    }

    #[test]
    fn rs_encode_at_matches_full_transform() {
        let config = Config::prod();

        for log_cols in 1..=12 {
            let grid_cols = 1usize << log_cols;
            let geom = config.table_geom(grid_cols);
            let len = grid_cols + geom.support_size;
            let row = rs_row(0x44 + log_cols as u128, len);

            let mut full = vec![Flat::from_raw(Block128::ZERO); geom.encoded_width];
            full[..geom.support_size].copy_from_slice(&row[grid_cols..]);
            full[geom.support_size..len].copy_from_slice(&row[..grid_cols]);

            AdditiveFft::<Block128>::new(geom.encoded_width.trailing_zeros())
                .unwrap()
                .forward_scalar(&mut full)
                .unwrap();

            let sparse: Vec<usize> = (0..geom.encoded_width).step_by(7).collect();
            let edges = vec![0, geom.encoded_width - 1];

            for columns in [sparse, edges, all_columns(&config, grid_cols)] {
                let want: Vec<Flat<Block128>> = columns.iter().map(|&c| full[c]).collect();

                assert_eq!(
                    rs_encode_at::<Block128>(&row, grid_cols, &config, &columns).unwrap(),
                    want,
                    "grid_cols {grid_cols}"
                );
            }
        }
    }
}
