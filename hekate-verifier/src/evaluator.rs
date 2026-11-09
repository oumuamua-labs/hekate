// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;
use hekate_core::errors;
use hekate_core::fold::PolyKind;
use hekate_core::proofs::FoldEvalProof;
use hekate_core::skip::SkipPoint;
use hekate_core::tensor::TensorProduct;
use hekate_core::trace::{ColumnType, TraceCompatibleField};
use hekate_crypto::Hasher;
use hekate_crypto::transcript::Transcript;
use hekate_math::{BinaryFieldExtras, Flat, HardwareField, PackableField};
use hekate_program::expander::{
    EVAL_SUMCHECK_DEGREE, RingMaps, field_bits, field_coeffs, ring_target,
};
use hekate_program::fold_pool::{FoldPool, UnitClass};
use hekate_program::linearized::linearized_coeffs;
use tracing::{instrument, trace_span, warn};

use crate::fold::{self, FoldClaims};
use crate::rejection::{PoolStage, Rejection};
use crate::sumcheck::verify;

struct PolyAt<'a, F> {
    r: &'a [Flat<F>],
    blocks: Vec<(UnitClass, Flat<F>)>,
}

struct RowTerm<'a, F> {
    r: &'a [Flat<F>],
    eq_low: Vec<Flat<F>>,
    step: usize,
    eq: Flat<F>,
    shifted: Flat<F>,
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

pub struct TableClaims<'a, F: HardwareField> {
    pub point: &'a SkipPoint<F>,
    pub claims: &'a Claims<'a, F>,
}

pub struct EvalVerifyContext<'a, F: HardwareField> {
    pub pool: &'a FoldPool,
    pub tables: &'a [TableClaims<'a, F>],

    /// Masked claims: the ring-switch final
    /// check is left to the outer argument.
    pub masked: bool,
}

/// What the outer argument needs from an accepted
/// eval sumcheck: `claim_n = fin` on plaintext values.
pub struct EvalOutcome<F: HardwareField> {
    pub eta: F,
    pub rho: Vec<F>,
    pub r_mix: Vec<F>,
    pub challenges: Vec<Flat<F>>,
    pub claim_masked: Flat<F>,
    pub fin: Flat<F>,
}

pub struct EvaluatorVerifier<F, H: Hasher> {
    _marker: PhantomData<(F, H)>,
}

impl<F, H: Hasher> EvaluatorVerifier<F, H>
where
    F: HardwareField + PackableField + TraceCompatibleField,
{
    /// Binds every pooled table's claimed virtual evals to the two roots:
    /// one degree-2 sumcheck to `r`, each polynomial's master opened by the fold.
    #[instrument(skip_all, level = "trace", name = "Evaluator::verify")]
    pub fn verify(
        trace_root: &H::Digest,
        h_root: Option<&H::Digest>,
        proof: &FoldEvalProof<F>,
        transcript: &mut Transcript<H>,
        ctx: EvalVerifyContext<'_, F>,
    ) -> Result<EvalOutcome<F>, Rejection>
    where
        F: BinaryFieldExtras + From<u128>,
    {
        let pool = ctx.pool;
        let layout = pool.layout();

        let zero = Flat::from_raw(F::ZERO);
        let one = Flat::from_raw(F::ONE);

        if ctx.tables.len() != pool.tables().len() {
            return Err(Rejection::Malformed(errors::Error::Protocol {
                protocol: "evaluator_verifier",
                message: "one claim set per pooled table is required",
            }));
        }

        for (table, pooled) in ctx.tables.iter().zip(pool.tables()) {
            let plan = &pooled.plan;

            if table.claims.flat().len() != plan.total_claims() + plan.next_claims() {
                return Err(Rejection::Malformed(errors::Error::Protocol {
                    protocol: "evaluator_verifier",
                    message: "ring-switch plan claim count does not match the claimed evaluations",
                }));
            }
        }

        transcript.append_message(b"eval_batch_start", b"");

        let eta_tower = transcript.challenge_field::<F>(b"eval_eta")?;

        let mut rho = Vec::with_capacity(pool.rho_vars());
        for _ in 0..pool.rho_vars() {
            rho.push(transcript.challenge_field::<F>(b"eval_rho")?);
        }

        let subfield = pool.has_subfield_units();

        // Order is load-bearing:
        // r'' follows the claimed evals, precedes the sumcheck.
        let kappa = F::BITS.ilog2() as usize;

        let r_mix: Vec<F> = if subfield {
            let mut m = Vec::with_capacity(kappa);
            for _ in 0..kappa {
                m.push(transcript.challenge_field::<F>(b"eval_rmix")?);
            }

            m
        } else {
            Vec::new()
        };

        let rho_hw: Vec<Flat<F>> = rho.iter().map(|r| r.to_hardware()).collect();

        let ring_span = trace_span!("ring_target").entered();

        let mut maps = RingMaps::new(&r_mix);
        let mut weights = Vec::with_capacity(ctx.tables.len());
        let mut target = F::ZERO;

        for (i, (table, pooled)) in ctx.tables.iter().zip(pool.tables()).enumerate() {
            let unit_weights = pool.unit_weights(i, eta_tower.to_hardware(), &rho_hw)?;

            target +=
                ring_target::<F>(&pooled.plan, &unit_weights, table.claims.flat(), &mut maps)?;

            weights.push(unit_weights);
        }

        drop(maps);
        drop(ring_span);

        let target_flat = target.to_hardware();
        let num_vars = layout.num_vars();

        if !proof.sumcheck_proof.skip.is_empty() {
            return Err(Rejection::PoolEval(PoolStage::Degree));
        }

        let (r, sumcheck_final_eval) = trace_span!("eval_sumcheck")
            .in_scope(|| {
                verify(
                    num_vars,
                    EVAL_SUMCHECK_DEGREE,
                    target_flat,
                    &proof.sumcheck_proof,
                    transcript,
                )
            })
            .map_err(|rejection| {
                warn!("eval sumcheck failed");

                rejection.pool_eval()
            })?;

        let masks = layout
            .polys()
            .iter()
            .filter(|fold| fold.shape.kind == PolyKind::Mask)
            .count();

        if proof.masters.len() + masks != layout.polys().len() || proof.mask_masters.len() != masks
        {
            return Err(Rejection::PoolEval(PoolStage::Masters));
        }

        // Every master evaluation precedes λ
        transcript.append_field_list(b"eval_masters", &proof.masters);
        transcript.append_field_list(b"eval_masks", &proof.mask_masters);

        let masters: Vec<Flat<F>> = proof
            .masters
            .iter()
            .chain(&proof.mask_masters)
            .map(|m| m.to_hardware())
            .collect();

        let weights_span = trace_span!("master_weights").entered();

        let twists = twist_coeffs::<F>(
            pool.tables()
                .iter()
                .flat_map(|t| &t.plan.units)
                .map(|unit| unit.field),
            &r_mix,
        )?;

        let mut fin = zero;
        for (i, ((table, pooled), unit_weights)) in ctx
            .tables
            .iter()
            .zip(pool.tables())
            .zip(&weights)
            .enumerate()
        {
            let mut lifted = Vec::new();
            let mut polys = Vec::new();

            for poly in pooled.polys() {
                let lift = num_vars - layout.polys()[poly].vars;
                let (r_pre, r_suf) = r.split_at(lift);
                let (r_x, r_b) = r_suf.split_at(pooled.num_vars);

                lifted.push((poly, r_pre));
                polys.push(PolyAt {
                    r: r_x,
                    blocks: pool.block_weights_at(i, poly, unit_weights, r_b),
                });
            }

            // The weight evals at r' are transparent;
            // the masters are bound by the fold below.
            let field_weights =
                master_weights_at::<F>(table.point, &polys, &twists, unit_weights.shift)?;

            let lifted_polys = lifted.iter().zip(&polys).zip(&field_weights);

            for (((poly, r_pre), at), field_weights) in lifted_polys {
                let prefix = r_pre.iter().fold(one, |acc, &x| acc * (one + x));

                let mut weight = zero;
                for (&(_, block_weight), &field_weight) in at.blocks.iter().zip(field_weights) {
                    weight += field_weight * block_weight;
                }

                fin += prefix * weight * masters[*poly];
            }
        }

        drop(weights_span);

        if !ctx.masked && sumcheck_final_eval != fin {
            warn!("ring-switch final check failed");
            return Err(Rejection::PoolEval(PoolStage::RingSwitch));
        }

        let claims = FoldClaims::<'_, F, H> {
            trace_root,
            helper_root: h_root,
            point: &r,
            masters: &masters,
        };

        fold::verify(layout, claims, proof, transcript)?;

        Ok(EvalOutcome {
            eta: eta_tower,
            rho,
            r_mix,
            challenges: r,
            claim_masked: sumcheck_final_eval,
            fin,
        })
    }
}

fn twist_coeffs<F>(
    fields: impl IntoIterator<Item = ColumnType>,
    r_mix: &[F],
) -> errors::Result<Vec<(ColumnType, Vec<Flat<F>>)>>
where
    F: TraceCompatibleField + From<u128>,
{
    let mut identity = vec![Flat::from_raw(F::ZERO); F::BITS / field_bits(ColumnType::B128)];
    identity[0] = Flat::from_raw(F::ONE);

    let mut twists = vec![(ColumnType::B128, identity)];
    let mut subfields: Vec<ColumnType> = Vec::new();

    for field in fields {
        if field != ColumnType::B128 && !subfields.contains(&field) {
            subfields.push(field);
        }
    }

    if subfields.is_empty() {
        return Ok(twists);
    }

    let maps: Vec<(Vec<F>, usize)> = subfields
        .iter()
        .map(|&field| (field_coeffs(field, r_mix), field_bits(field)))
        .collect();

    for (&field, mu) in subfields.iter().zip(linearized_coeffs(&maps)?) {
        let twist = mu.iter().step_by(field_bits(field)).copied().collect();

        twists.push((field, twist));
    }

    Ok(twists)
}

/// Master weight `Σ_j μ_j·(w^(2^j) + η·K̃_w^(2^j))(r')` per unit class
/// at each polynomial's `r'`, the `η` term for next-row classes only:
/// squaring fixes cube points, hence `φ_K(w(x)) = Σ_j μ_j·w(x)^(2^j)`.
fn master_weights_at<F: HardwareField>(
    point: &SkipPoint<F>,
    polys: &[PolyAt<'_, F>],
    twists: &[(ColumnType, Vec<Flat<F>>)],
    eta_shift: Flat<F>,
) -> errors::Result<Vec<Vec<Flat<F>>>> {
    let zero = Flat::from_raw(F::ZERO);
    let k = point.skip_vars();

    let mut rows: Vec<RowTerm<'_, F>> = Vec::new();
    let mut members = Vec::with_capacity(polys.len());

    for poly in polys {
        if point.num_vars() != poly.r.len() {
            return Err(errors::Error::Protocol {
                protocol: "evaluator_verifier",
                message: "claim point and row challenges differ in length",
            });
        }

        let mut mus = Vec::with_capacity(poly.blocks.len());
        for &(class, _) in &poly.blocks {
            let (_, mu) =
                twists
                    .iter()
                    .find(|(f, _)| *f == class.field)
                    .ok_or(errors::Error::Protocol {
                        protocol: "evaluator_verifier",
                        message: "unit field has no twist coefficients",
                    })?;

            mus.push((field_bits(class.field), mu, class.next));
        }

        let width = mus
            .iter()
            .map(|&(width, _, _)| width)
            .min()
            .unwrap_or(F::BITS);

        let at = match rows.iter().position(|row| row.r == poly.r) {
            Some(at) => {
                rows[at].step = rows[at].step.min(width);

                at
            }
            None => {
                rows.push(RowTerm {
                    r: poly.r,
                    eq_low: TensorProduct::eq_table(&poly.r[..k]),
                    step: width,
                    eq: zero,
                    shifted: zero,
                });

                rows.len() - 1
            }
        };

        members.push((at, mus));
    }

    let step = rows.iter().map(|row| row.step).min().unwrap_or(F::BITS);

    let mut low = point.low().to_vec();
    let mut sigma = point.tail().to_vec();
    let mut weights: Vec<Vec<Flat<F>>> = polys
        .iter()
        .map(|poly| vec![zero; poly.blocks.len()])
        .collect();

    for j in (0..F::BITS).step_by(step) {
        if j > 0 {
            for p in low.iter_mut().chain(sigma.iter_mut()) {
                for _ in 0..step {
                    *p = *p * *p;
                }
            }
        }

        for row in rows.iter_mut().filter(|row| j % row.step == 0) {
            let (eq, k_p) = eq_and_k_p_at(&low, &row.eq_low, &sigma, &row.r[k..]);

            row.eq = eq;
            row.shifted = eq + eta_shift * k_p;
        }

        for (poly_weights, (at, mus)) in weights.iter_mut().zip(&members) {
            let row = &rows[*at];

            for (weight, &(width, mu, next)) in poly_weights.iter_mut().zip(mus) {
                if j % width == 0 {
                    let term = match next {
                        true => row.shifted,
                        false => row.eq,
                    };

                    *weight += mu[j / width] * term;
                }
            }
        }
    }

    Ok(weights)
}

/// Evaluates the weights' MLE and next-row shift `K̃` at `r'`,
/// the block part by its eq table and the tail part by
/// `eq_and_k_p_tail`, a block's last row carrying into the next.
fn eq_and_k_p_at<F: HardwareField>(
    low: &[Flat<F>],
    eq_low: &[Flat<F>],
    tail: &[Flat<F>],
    r_high: &[Flat<F>],
) -> (Flat<F>, Flat<F>) {
    let zero = Flat::from_raw(F::ZERO);
    let last = low.len() - 1;

    let block = low
        .iter()
        .zip(eq_low)
        .fold(zero, |acc, (&w, &e)| acc + w * e);

    let shift = low[..last]
        .iter()
        .zip(&eq_low[1..])
        .fold(zero, |acc, (&w, &e)| acc + w * e);

    let (eq_high, k_high) = eq_and_k_p_tail(tail, r_high);

    (
        block * eq_high,
        shift * eq_high + low[last] * eq_low[0] * k_high,
    )
}

/// `(eq(P, r'), K̃_P(r'))` in O(n): the K_P carry chain
/// `Σ_k Π_{i<k}(1+r'_i)P_i · r'_k(1+P_k) · Π_{i>k}(1+r'_i+P_i)`
/// plus the cyclic wrap `Π_i (1+r'_i)P_i`.
fn eq_and_k_p_tail<F: HardwareField>(point: &[Flat<F>], r_final: &[Flat<F>]) -> (Flat<F>, Flat<F>) {
    let one = Flat::from_raw(F::ONE);

    let mut eq = one;
    let mut carry = Flat::from_raw(F::ZERO);
    let mut wrap = one;

    for (&p, &r) in point.iter().zip(r_final).rev() {
        let ripple = (one + r) * p;

        carry = r * (one + p) * eq + ripple * carry;
        eq *= one + r + p;
        wrap *= ripple;
    }

    (eq, carry + wrap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hekate_core::poly::PolyVariant;
    use hekate_math::{Block128, TowerField};
    use hekate_program::expander::eq_tensor_b;

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

    fn contract_bits(values: &[Block128], coeffs: &[Block128]) -> Vec<Block128> {
        values
            .iter()
            .map(|v| {
                let mut acc = Block128::ZERO;
                for (u, &m) in coeffs.iter().enumerate() {
                    if (v.0 >> u) & 1 == 1 {
                        acc += m;
                    }
                }

                acc
            })
            .collect()
    }

    fn flat(values: &[Block128]) -> Vec<Flat<Block128>> {
        values.iter().map(|v| v.to_hardware()).collect()
    }

    /// `K_P` is only the right weight if its wrap agrees
    /// with `PolyVariant::Shifted`, which is what the AIR
    /// and the prover's fold read as the next row.
    #[test]
    fn k_p_contracts_to_polyvariant_next_row() {
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
            let (eq, k_p) = eq_and_k_p_tail(&flat(&point), &flat(&r_final));

            assert_eq!(k_p.to_tower(), direct, "n={num_vars}");
            assert_eq!(
                eq.to_tower(),
                mle_at(&eq_tensor_b(&point), &r_final),
                "eq n={num_vars}"
            );
        }
    }

    #[test]
    fn skip_point_closed_form_matches_materialized() {
        for tail_vars in 0..=3 {
            let low = flat(&elems(17 + tail_vars as u128, 4));
            let tail = flat(&elems(29 + tail_vars as u128, tail_vars));
            let point = SkipPoint::new(low.clone(), tail.clone()).unwrap();

            let r_final = elems(401 + tail_vars as u128, tail_vars + 2);
            let r_flat = flat(&r_final);

            let weights: Vec<Block128> = point.expand().iter().map(|w| w.to_tower()).collect();
            let n = weights.len();
            let shifted: Vec<Block128> = (0..n).map(|i| weights[(i + n - 1) & (n - 1)]).collect();

            let (eq, k_p) = eq_and_k_p_at(
                &low,
                &TensorProduct::eq_table(&r_flat[..2]),
                &tail,
                &r_flat[2..],
            );

            assert_eq!(
                eq.to_tower(),
                mle_at(&weights, &r_final),
                "eq t={tail_vars}"
            );
            assert_eq!(
                k_p.to_tower(),
                mle_at(&shifted, &r_final),
                "k_p t={tail_vars}"
            );
        }
    }

    #[test]
    fn closed_forms_match_materialized() {
        let kappa = Block128::BITS.ilog2() as usize;

        let fields = [
            ColumnType::Bit,
            ColumnType::B8,
            ColumnType::B16,
            ColumnType::B32,
            ColumnType::B64,
            ColumnType::B128,
        ];

        for num_vars in 1..=5 {
            let point = elems(7 + num_vars as u128, num_vars);
            let r_final = elems(211 + num_vars as u128, num_vars);
            let r_mix = elems(919 + num_vars as u128, kappa);
            let eta = elems(503 + num_vars as u128, 1)[0];

            let twists = twist_coeffs::<Block128>(fields, &r_mix).unwrap();
            let blocks: Vec<(UnitClass, Flat<Block128>)> = fields
                .iter()
                .flat_map(|&field| [false, true].map(|next| UnitClass { field, next }))
                .map(|class| (class, Flat::from_raw(Block128::ZERO)))
                .collect();

            let r_flat = flat(&r_final);

            let weights = |blocks: &[(UnitClass, Flat<Block128>)]| {
                master_weights_at(
                    &SkipPoint::multilinear(flat(&point)),
                    &[PolyAt {
                        r: &r_flat,
                        blocks: blocks.to_vec(),
                    }],
                    &twists,
                    eta.to_hardware(),
                )
                .unwrap()
                .remove(0)
            };

            let joint = weights(&blocks);

            for (k, &field) in fields.iter().enumerate() {
                let coeffs = field_coeffs(field, &r_mix);
                let a = mle_at(&contract_bits(&eq_tensor_b(&point), &coeffs), &r_final);
                let a_next = mle_at(&contract_bits(&k_p_on_cube(&point), &coeffs), &r_final);

                let current = weights(&blocks[2 * k..=2 * k])[0];
                let shifted = weights(&blocks[2 * k + 1..=2 * k + 1])[0];

                assert_eq!(current.to_tower(), a, "A {field:?} n={num_vars}");
                assert_eq!(
                    shifted.to_tower(),
                    a + eta * a_next,
                    "A_next {field:?} n={num_vars}"
                );
                assert_eq!(
                    joint[2 * k..2 * k + 2],
                    [current, shifted],
                    "joint {field:?} n={num_vars}"
                );
            }
        }
    }

    #[test]
    fn shared_rows_match_separate_calls() {
        let kappa = Block128::BITS.ilog2() as usize;

        let point = SkipPoint::new(flat(&elems(41, 4)), flat(&elems(43, 3))).unwrap();
        let r_mix = elems(47, kappa);
        let eta = elems(53, 1)[0].to_hardware();

        let r_a = flat(&elems(59, 5));
        let r_b = flat(&elems(61, 5));

        let fields = [
            ColumnType::Bit,
            ColumnType::B8,
            ColumnType::B32,
            ColumnType::B64,
            ColumnType::B128,
        ];

        let twists = twist_coeffs::<Block128>(fields, &r_mix).unwrap();
        let blocks: Vec<(UnitClass, Flat<Block128>)> = fields
            .iter()
            .flat_map(|&field| [false, true].map(|next| UnitClass { field, next }))
            .map(|class| (class, Flat::from_raw(Block128::ZERO)))
            .collect();

        let polys =
            [(&r_a, 4..8), (&r_b, 8..10), (&r_a, 0..4), (&r_b, 3..6)].map(|(r, classes)| PolyAt {
                r: &r[..],
                blocks: blocks[classes].to_vec(),
            });

        let shared = master_weights_at(&point, &polys, &twists, eta).unwrap();

        for (poly, weights) in polys.iter().zip(&shared) {
            let alone = master_weights_at(
                &point,
                &[PolyAt {
                    r: poly.r,
                    blocks: poly.blocks.clone(),
                }],
                &twists,
                eta,
            )
            .unwrap()
            .remove(0);

            assert_eq!(*weights, alone);
        }
    }

    #[test]
    fn skip_point_weights_match_materialized() {
        let kappa = Block128::BITS.ilog2() as usize;

        let fields = [
            ColumnType::Bit,
            ColumnType::B8,
            ColumnType::B16,
            ColumnType::B32,
            ColumnType::B64,
            ColumnType::B128,
        ];

        for (block, tail_vars) in [(4usize, 0usize), (4, 3), (32, 1), (32, 2)] {
            let seed = (block + tail_vars) as u128;
            let point = SkipPoint::new(
                flat(&elems(67 + seed, block)),
                flat(&elems(71 + seed, tail_vars)),
            )
            .unwrap();

            let r_final = elems(613 + seed, point.num_vars());
            let r_flat = flat(&r_final);
            let r_mix = elems(827 + seed, kappa);
            let eta = elems(89 + seed, 1)[0];

            let twists = twist_coeffs::<Block128>(fields, &r_mix).unwrap();

            let weights: Vec<Block128> = point.expand().iter().map(|w| w.to_tower()).collect();
            let n = weights.len();
            let shifted: Vec<Block128> = (0..n).map(|i| weights[(i + n - 1) & (n - 1)]).collect();

            for field in fields {
                let coeffs = field_coeffs(field, &r_mix);
                let a = mle_at(&contract_bits(&weights, &coeffs), &r_final);
                let a_next = mle_at(&contract_bits(&shifted, &coeffs), &r_final);

                let blocks = [false, true]
                    .map(|next| (UnitClass { field, next }, Flat::from_raw(Block128::ZERO)));

                let at_point = master_weights_at(
                    &point,
                    &[PolyAt {
                        r: &r_flat,
                        blocks: blocks.to_vec(),
                    }],
                    &twists,
                    eta.to_hardware(),
                )
                .unwrap()
                .remove(0);

                assert_eq!(
                    at_point[0].to_tower(),
                    a,
                    "A {field:?} block {block} tail {tail_vars}"
                );
                assert_eq!(
                    at_point[1].to_tower(),
                    a + eta * a_next,
                    "A_next {field:?} block {block} tail {tail_vars}"
                );
            }
        }
    }
}
