// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use crate::fold::{self, FoldClaims};
use crate::sumcheck::verify;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;
use hekate_core::errors;
use hekate_core::fold::PolyKind;
use hekate_core::proofs::FoldEvalProof;
use hekate_core::trace::{ColumnType, TraceCompatibleField};
use hekate_crypto::Hasher;
use hekate_crypto::transcript::Transcript;
use hekate_math::{BinaryFieldExtras, Block128, Flat, HardwareField, PackableField, TowerField};
use hekate_program::expander::{
    EVAL_SUMCHECK_DEGREE, RingMaps, field_bits, field_coeffs, ring_target,
};
use hekate_program::fold_pool::FoldPool;
use hekate_program::linearized::linearized_coeffs;
use tracing::{instrument, warn};

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
    pub pool: &'a FoldPool,
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
    /// Binds every pooled table's claimed virtual evals to the two roots:
    /// one degree-2 sumcheck to `r`, each polynomial's master opened by the fold.
    #[instrument(skip_all, level = "trace", name = "Evaluator::verify")]
    pub fn verify(
        trace_root: &[u8; 32],
        h_root: Option<&[u8; 32]>,
        proof: &FoldEvalProof<F>,
        transcript: &mut Transcript<H>,
        ctx: EvalVerifyContext<'_, F>,
    ) -> errors::Result<Option<EvalOutcome<F>>>
    where
        F: BinaryFieldExtras + Into<Block128> + From<u128>,
    {
        let pool = ctx.pool;
        let layout = pool.layout();
        let shifted_claims = ctx.shifted_claims;
        let claim_halves = if shifted_claims { 2 } else { 1 };

        let zero = Flat::from_raw(F::ZERO);
        let one = Flat::from_raw(F::ONE);

        if ctx.tables.len() != pool.tables().len() {
            return Err(errors::Error::Protocol {
                protocol: "evaluator_verifier",
                message: "one claim set per pooled table is required",
            });
        }

        for (table, pooled) in ctx.tables.iter().zip(pool.tables()) {
            if table.claims.flat().len() != pooled.plan.total_claims() * claim_halves {
                return Err(errors::Error::Protocol {
                    protocol: "evaluator_verifier",
                    message: "ring-switch plan claim count does not match the claimed evaluations",
                });
            }
        }

        transcript.append_message(b"eval_batch_start", b"");

        let eta_tower = transcript.challenge_field::<F>(b"eval_eta")?;

        let mut rho = Vec::with_capacity(pool.rho_vars());
        for _ in 0..pool.rho_vars() {
            rho.push(transcript.challenge_field::<F>(b"eval_rho")?);
        }

        let subfield = pool.has_subfield_units();

        if subfield && F::BITS != NBITS {
            return Err(errors::Error::Protocol {
                protocol: "evaluator_verifier",
                message: "ring-switch evaluation requires a 128-bit field",
            });
        }

        // Order is load-bearing:
        // r'' follows the claimed evals, precedes the sumcheck.
        let kappa = F::BITS.ilog2() as usize;

        let r_mix: Vec<Block128> = if subfield {
            let mut m = Vec::with_capacity(kappa);
            for _ in 0..kappa {
                m.push(transcript.challenge_field::<F>(b"eval_rmix")?.into());
            }

            m
        } else {
            Vec::new()
        };

        let rho_b: Vec<Flat<Block128>> = rho.iter().map(|&x| to_block(x)).collect();

        let mut maps = RingMaps::new(&r_mix);
        let mut weights = Vec::with_capacity(ctx.tables.len());
        let mut target = Block128::ZERO;

        for (i, (table, pooled)) in ctx.tables.iter().zip(pool.tables()).enumerate() {
            let unit_weights = pool.unit_weights(i, to_block(eta_tower), &rho_b)?;

            target += ring_target::<F>(
                &pooled.plan,
                &unit_weights,
                table.claims.flat(),
                &mut maps,
                shifted_claims,
            );

            weights.push(unit_weights);
        }

        drop(maps);

        let target_flat = F::from(target.0).to_hardware();
        let num_vars = layout.num_vars();

        let sc_res = verify(
            num_vars,
            EVAL_SUMCHECK_DEGREE,
            target_flat,
            &proof.sumcheck_proof,
            transcript,
        )?;

        let (r, sumcheck_final_eval) = match sc_res {
            Some(res) => res,
            None => {
                warn!("Sumcheck failed");
                return Ok(None);
            }
        };

        let masks = layout
            .polys()
            .iter()
            .filter(|fold| fold.shape.kind == PolyKind::Mask)
            .count();

        if proof.masters.len() + masks != layout.polys().len() || proof.mask_masters.len() != masks
        {
            return Err(errors::Error::Protocol {
                protocol: "evaluator_verifier",
                message: "one master evaluation per committed polynomial is required",
            });
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

        let twists = twist_coeffs::<F>(
            pool.tables()
                .iter()
                .flat_map(|t| &t.plan.units)
                .map(|unit| unit.field),
            &r_mix,
        );

        let mut fin = zero;
        for (i, ((table, pooled), unit_weights)) in ctx
            .tables
            .iter()
            .zip(pool.tables())
            .zip(&weights)
            .enumerate()
        {
            let eta_shift = from_block::<F>(unit_weights.shift);
            let polys = [Some(pooled.trace), pooled.helper].into_iter().flatten();

            for poly in polys {
                let lift = num_vars - layout.polys()[poly].vars;
                let (r_pre, r_suf) = r.split_at(lift);
                let (r_x, r_b) = r_suf.split_at(pooled.num_vars);

                let r_b: Vec<Flat<Block128>> = r_b.iter().map(|x| to_block(x.to_tower())).collect();
                let blocks = pool.block_weights_at(i, poly, unit_weights, &r_b);

                // The weight evals at r' are transparent;
                // the masters are bound by the fold below.
                let field_weights = master_weights_at::<F>(
                    table.point,
                    r_x,
                    &twists,
                    eta_shift,
                    &blocks,
                    shifted_claims,
                )?;

                let prefix = r_pre.iter().fold(one, |acc, &x| acc * (one + x));

                let mut weight = zero;
                for (&(_, block_weight), &field_weight) in blocks.iter().zip(&field_weights) {
                    weight += field_weight * from_block(block_weight);
                }

                fin += prefix * weight * masters[poly];
            }
        }

        if !ctx.masked && sumcheck_final_eval != fin {
            warn!("ring-switch final check failed");
            return Ok(None);
        }

        let claims = FoldClaims {
            trace_root,
            helper_root: h_root,
            point: &r,
            masters: &masters,
        };

        if !fold::verify(layout, claims, proof, transcript)? {
            warn!("fold opening of the masters failed");
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

fn twist_coeffs<F>(
    fields: impl IntoIterator<Item = ColumnType>,
    r_mix: &[Block128],
) -> Vec<(ColumnType, Vec<Flat<F>>)>
where
    F: HardwareField + From<u128>,
{
    let mut twists = vec![(ColumnType::B128, vec![Flat::from_raw(F::ONE)])];
    let mut subfields: Vec<ColumnType> = Vec::new();

    for field in fields {
        if field != ColumnType::B128 && !subfields.contains(&field) {
            subfields.push(field);
        }
    }

    if subfields.is_empty() {
        return twists;
    }

    let maps: Vec<(Vec<Block128>, usize)> = subfields
        .iter()
        .map(|&field| (field_coeffs(field, r_mix), field_bits(field)))
        .collect();

    for (&field, mu) in subfields.iter().zip(linearized_coeffs(&maps)) {
        let twist = mu
            .iter()
            .step_by(field_bits(field))
            .map(|&m| from_block(m))
            .collect();

        twists.push((field, twist));
    }

    twists
}

/// One master weight `Σ_j μ_j·(eq + η·K̃)(σ^j(P), r')`
/// per unit field of `fields`: squaring fixes every cube
/// point, hence `φ_K(eq(P, x)) = Σ_j μ_j·eq(σ^j(P), x)`.
fn master_weights_at<F: HardwareField>(
    point: &[Flat<F>],
    r_row: &[Flat<F>],
    twists: &[(ColumnType, Vec<Flat<F>>)],
    eta_shift: Flat<F>,
    fields: &[(ColumnType, Flat<Block128>)],
    shifted_claims: bool,
) -> errors::Result<Vec<Flat<F>>> {
    if point.len() != r_row.len() {
        return Err(errors::Error::Protocol {
            protocol: "evaluator_verifier",
            message: "claim point and row challenges differ in length",
        });
    }

    let mut mus = Vec::with_capacity(fields.len());
    for &(field, _) in fields {
        let (_, mu) = twists
            .iter()
            .find(|(f, _)| *f == field)
            .ok_or(errors::Error::Protocol {
                protocol: "evaluator_verifier",
                message: "unit field has no twist coefficients",
            })?;

        mus.push((field_bits(field), mu));
    }

    let step = mus.iter().map(|&(width, _)| width).min().unwrap_or(NBITS);

    let mut sigma = point.to_vec();
    let mut weights = vec![Flat::from_raw(F::ZERO); fields.len()];

    for j in (0..NBITS).step_by(step) {
        if j > 0 {
            for p in sigma.iter_mut() {
                for _ in 0..step {
                    *p = *p * *p;
                }
            }
        }

        let (eq, k_p) = eq_and_k_p_at(&sigma, r_row);
        let term = match shifted_claims {
            true => eq + eta_shift * k_p,
            false => eq,
        };

        for (weight, &(width, mu)) in weights.iter_mut().zip(&mus) {
            if j % width == 0 {
                *weight += mu[j / width] * term;
            }
        }
    }

    Ok(weights)
}

/// `(eq(P, r'), K̃_P(r'))` in O(n): the K_P carry chain
/// `Σ_k Π_{i<k}(1+r'_i)P_i · r'_k(1+P_k) · Π_{i>k}(1+r'_i+P_i)`
/// plus the cyclic wrap `Π_i (1+r'_i)P_i`.
fn eq_and_k_p_at<F: HardwareField>(point: &[Flat<F>], r_final: &[Flat<F>]) -> (Flat<F>, Flat<F>) {
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

fn to_block<F: Into<Block128>>(x: F) -> Flat<Block128> {
    x.into().to_hardware()
}

fn from_block<F: HardwareField + From<u128>>(x: Flat<Block128>) -> Flat<F> {
    F::from(x.to_tower().0).to_hardware()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hekate_core::poly::PolyVariant;
    use hekate_math::TowerField;
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
            let (eq, k_p) = eq_and_k_p_at(&flat(&point), &flat(&r_final));

            assert_eq!(k_p.to_tower(), direct, "n={num_vars}");
            assert_eq!(
                eq.to_tower(),
                mle_at(&eq_tensor_b(&point), &r_final),
                "eq n={num_vars}"
            );
        }
    }

    #[test]
    fn closed_forms_match_materialized() {
        let kappa = NBITS.ilog2() as usize;

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

            let twists = twist_coeffs::<Block128>(fields, &r_mix);
            let blocks: Vec<(ColumnType, Flat<Block128>)> = fields
                .iter()
                .map(|&field| (field, Flat::from_raw(Block128::ZERO)))
                .collect();

            let weights = |blocks: &[(ColumnType, Flat<Block128>)], shifted_claims| {
                master_weights_at(
                    &flat(&point),
                    &flat(&r_final),
                    &twists,
                    eta.to_hardware(),
                    blocks,
                    shifted_claims,
                )
                .unwrap()
            };

            let joint = weights(&blocks, true);

            for (k, &field) in fields.iter().enumerate() {
                let coeffs = field_coeffs(field, &r_mix);
                let a = mle_at(&contract_bits(&eq_tensor_b(&point), &coeffs), &r_final);
                let a_next = mle_at(&contract_bits(&k_p_on_cube(&point), &coeffs), &r_final);

                let alone = &blocks[k..=k];

                assert_eq!(
                    weights(alone, false)[0].to_tower(),
                    a,
                    "A {field:?} n={num_vars}"
                );
                assert_eq!(
                    weights(alone, true)[0].to_tower(),
                    a + eta * a_next,
                    "A_next {field:?} n={num_vars}"
                );
                assert_eq!(
                    joint[k],
                    weights(alone, true)[0],
                    "joint {field:?} n={num_vars}"
                );
            }
        }
    }
}
