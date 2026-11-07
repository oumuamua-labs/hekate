// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use super::wire_err;
use alloc::vec::Vec;
use flatbuffers::FlatBufferBuilder;
use hekate_core::errors::Result;
use hekate_core::poly::UnivariatePoly;
use hekate_core::proofs::{
    FoldEvalProof, InnerProof, LogUpAux, OuterOpening, OuterProof, SumcheckProof, TreeOpening,
};
use hekate_crypto::merkle::Multiproof;
use hekate_math::TowerField;

use crate::generated::proof as fb;

const WIRE_PROOF_VERSION: u32 = 9;

pub fn serialize_proof<'a, F: TowerField>(
    fbb: &mut FlatBufferBuilder<'a>,
    proof: &InnerProof<F>,
) -> flatbuffers::WIPOffset<fb::Proof<'a>> {
    let trace_root = fbb.create_vector(&proof.trace_root);
    let h_root = proof.h_root.as_ref().map(|r| fbb.create_vector(r));
    let zc = serialize_sumcheck(fbb, &proof.zerocheck_proof);
    let mla = serialize_logup_aux(fbb, &proof.main_logup_aux);
    let mpe = serialize_point_evaluation(fbb, &proof.main_point_evaluation);
    let ep = serialize_fold_eval(fbb, &proof.eval_proof);

    let rows: Vec<u64> = proof.chiplet_rows.iter().map(|&r| r as u64).collect();
    let cr = fbb.create_vector(&rows);

    let czc_offsets: Vec<_> = proof
        .chiplet_zerocheck_proofs
        .iter()
        .map(|p| serialize_sumcheck(fbb, p))
        .collect();
    let czc = fbb.create_vector(&czc_offsets);

    let cla_offsets: Vec<_> = proof
        .chiplet_logup_aux
        .iter()
        .map(|a| serialize_logup_aux(fbb, a))
        .collect();
    let cla = fbb.create_vector(&cla_offsets);

    let cpe_offsets: Vec<_> = proof
        .chiplet_point_evaluations
        .iter()
        .map(|p| serialize_point_evaluation(fbb, p))
        .collect();
    let cpe = fbb.create_vector(&cpe_offsets);

    let pad_root = proof.pad_root.as_ref().map(|r| fbb.create_vector(r));
    let outer = proof.outer.as_ref().map(|o| serialize_outer(fbb, o));

    fb::Proof::create(
        fbb,
        &fb::ProofArgs {
            version: WIRE_PROOF_VERSION,
            trace_root: Some(trace_root),
            h_root,
            zerocheck_proof: Some(zc),
            main_logup_aux: Some(mla),
            main_point_evaluation: Some(mpe),
            eval_proof: Some(ep),
            chiplet_rows: Some(cr),
            chiplet_zerocheck_proofs: Some(czc),
            chiplet_logup_aux: Some(cla),
            chiplet_point_evaluations: Some(cpe),
            pad_root,
            outer,
        },
    )
}

fn serialize_outer_opening<'a, F: TowerField>(
    fbb: &mut FlatBufferBuilder<'a>,
    opening: &OuterOpening<F>,
) -> flatbuffers::WIPOffset<fb::OuterOpening<'a>> {
    let columns = fbb.create_vector(&opening.columns);

    let values: Vec<fb::Block256> = opening.values.iter().map(|f| block_from_field(f)).collect();
    let values = fbb.create_vector(&values);
    let multiproof = serialize_multiproof(fbb, &opening.multiproof);

    fb::OuterOpening::create(
        fbb,
        &fb::OuterOpeningArgs {
            columns: Some(columns),
            values: Some(values),
            multiproof: Some(multiproof),
        },
    )
}

fn serialize_multiproof<'a>(
    fbb: &mut FlatBufferBuilder<'a>,
    multiproof: &Multiproof,
) -> flatbuffers::WIPOffset<fb::Multiproof<'a>> {
    let siblings = fbb.create_vector(&multiproof.siblings);
    let unreached = fbb.create_vector(&multiproof.unreached);

    fb::Multiproof::create(
        fbb,
        &fb::MultiproofArgs {
            siblings: Some(siblings),
            unreached: Some(unreached),
        },
    )
}

fn serialize_outer<'a, F: TowerField>(
    fbb: &mut FlatBufferBuilder<'a>,
    outer: &OuterProof<F>,
) -> flatbuffers::WIPOffset<fb::OuterProof<'a>> {
    let aux_root = fbb.create_vector(&outer.aux_root);

    let field_vec = |fbb: &mut FlatBufferBuilder<'a>, values: &[F]| {
        let blocks: Vec<fb::Block256> = values.iter().map(|f| block_from_field(f)).collect();

        fbb.create_vector(&blocks)
    };

    let interleaved = field_vec(fbb, &outer.interleaved);
    let linear = field_vec(fbb, &outer.linear);
    let quadratic = field_vec(fbb, &outer.quadratic);

    let pad_opening = serialize_outer_opening(fbb, &outer.pad_opening);
    let aux_opening = serialize_outer_opening(fbb, &outer.aux_opening);

    fb::OuterProof::create(
        fbb,
        &fb::OuterProofArgs {
            aux_root: Some(aux_root),
            interleaved: Some(interleaved),
            linear: Some(linear),
            quadratic: Some(quadratic),
            pad_opening: Some(pad_opening),
            aux_opening: Some(aux_opening),
        },
    )
}

pub fn serialize_proof_bytes<F: TowerField>(proof: &InnerProof<F>) -> Vec<u8> {
    let mut fbb = FlatBufferBuilder::with_capacity(512 * 1024);

    let offset = serialize_proof(&mut fbb, proof);
    fbb.finish(offset, None);

    fbb.finished_data().to_vec()
}

pub fn deserialize_proof<F: TowerField>(bytes: &[u8]) -> Result<InnerProof<F>> {
    let fb_proof =
        flatbuffers::root::<fb::Proof>(bytes).map_err(|_| wire_err("invalid proof FlatBuffer"))?;

    if fb_proof.version() != WIRE_PROOF_VERSION {
        return Err(wire_err("proof wire format version mismatch"));
    }

    let trace_root =
        deserialize_bytes(fb_proof.trace_root()).ok_or(wire_err("missing trace_root"))?;
    let h_root = deserialize_bytes(fb_proof.h_root());

    let zerocheck_proof = fb_proof
        .zerocheck_proof()
        .map(|p| deserialize_sumcheck::<F>(p))
        .transpose()?
        .ok_or(wire_err("missing zerocheck_proof"))?;

    let main_logup_aux = fb_proof
        .main_logup_aux()
        .map(|a| deserialize_logup_aux::<F>(a))
        .transpose()?
        .ok_or(wire_err("missing main_logup_aux"))?;

    let main_point_evaluation = fb_proof
        .main_point_evaluation()
        .map(|p| deserialize_point_evaluation::<F>(p))
        .transpose()?
        .ok_or(wire_err("missing main_point_evaluation"))?;

    let eval_proof = fb_proof
        .eval_proof()
        .map(|p| deserialize_fold_eval::<F>(p))
        .transpose()?
        .ok_or(wire_err("missing eval_proof"))?;

    let chiplet_rows = match fb_proof.chiplet_rows() {
        Some(v) => v
            .iter()
            .map(|r| usize::try_from(r).map_err(|_| wire_err("chiplet_rows exceeds usize")))
            .collect::<Result<Vec<_>>>()?,
        None => Vec::new(),
    };

    let chiplet_zerocheck_proofs = match fb_proof.chiplet_zerocheck_proofs() {
        Some(v) => {
            let mut proofs = Vec::with_capacity(v.len());
            for i in 0..v.len() {
                proofs.push(deserialize_sumcheck::<F>(v.get(i))?);
            }

            proofs
        }
        None => Vec::new(),
    };

    let chiplet_logup_aux = match fb_proof.chiplet_logup_aux() {
        Some(v) => {
            let mut auxs = Vec::with_capacity(v.len());
            for i in 0..v.len() {
                auxs.push(deserialize_logup_aux::<F>(v.get(i))?);
            }

            auxs
        }
        None => Vec::new(),
    };

    let chiplet_point_evaluations = match fb_proof.chiplet_point_evaluations() {
        Some(v) => {
            let mut evaluations = Vec::with_capacity(v.len());
            for i in 0..v.len() {
                evaluations.push(deserialize_point_evaluation::<F>(v.get(i))?);
            }

            evaluations
        }
        None => Vec::new(),
    };

    let pad_root = deserialize_bytes(fb_proof.pad_root());

    let outer = fb_proof
        .outer()
        .map(|o| deserialize_outer::<F>(o))
        .transpose()?;

    Ok(InnerProof {
        trace_root,
        h_root,
        zerocheck_proof,
        main_logup_aux,
        main_point_evaluation,
        eval_proof,
        chiplet_rows,
        chiplet_zerocheck_proofs,
        chiplet_logup_aux,
        chiplet_point_evaluations,
        pad_root,
        outer,
    })
}

fn deserialize_field_vec<F: TowerField>(
    v: Option<flatbuffers::Vector<'_, fb::Block256>>,
) -> Result<Vec<F>> {
    match v {
        Some(v) => {
            let mut out = Vec::with_capacity(v.len());
            for i in 0..v.len() {
                out.push(field_from_block::<F>(*v.get(i))?);
            }

            Ok(out)
        }
        None => Ok(Vec::new()),
    }
}

fn deserialize_bytes(data: Option<flatbuffers::Vector<'_, u8>>) -> Option<Vec<u8>> {
    data.map(|data| data.bytes().to_vec())
}

fn deserialize_outer_opening<F: TowerField>(fb: fb::OuterOpening<'_>) -> Result<OuterOpening<F>> {
    let columns = match fb.columns() {
        Some(v) => v.iter().collect(),
        None => Vec::new(),
    };

    Ok(OuterOpening {
        columns,
        values: deserialize_field_vec::<F>(fb.values())?,
        multiproof: deserialize_multiproof(fb.multiproof())?,
    })
}

fn deserialize_multiproof(fb: Option<fb::Multiproof<'_>>) -> Result<Multiproof> {
    let fb = fb.ok_or(wire_err("missing multiproof"))?;

    Ok(Multiproof {
        siblings: deserialize_bytes(fb.siblings()).unwrap_or_default(),
        unreached: deserialize_bytes(fb.unreached()).unwrap_or_default(),
    })
}

fn deserialize_outer<F: TowerField>(fb: fb::OuterProof<'_>) -> Result<OuterProof<F>> {
    let aux_root = deserialize_bytes(fb.aux_root()).ok_or(wire_err("missing outer aux_root"))?;

    let pad_opening = fb
        .pad_opening()
        .map(|o| deserialize_outer_opening::<F>(o))
        .transpose()?
        .ok_or(wire_err("missing outer pad_opening"))?;

    let aux_opening = fb
        .aux_opening()
        .map(|o| deserialize_outer_opening::<F>(o))
        .transpose()?
        .ok_or(wire_err("missing outer aux_opening"))?;

    Ok(OuterProof {
        aux_root,
        interleaved: deserialize_field_vec::<F>(fb.interleaved())?,
        linear: deserialize_field_vec::<F>(fb.linear())?,
        quadratic: deserialize_field_vec::<F>(fb.quadratic())?,
        pad_opening,
        aux_opening,
    })
}

fn block_from_field<F: TowerField>(f: &F) -> fb::Block256 {
    let [w0, w1, w2, w3] = super::field::field_to_words(f);

    fb::Block256::new(w0, w1, w2, w3)
}

fn field_from_block<F: TowerField>(block: fb::Block256) -> Result<F> {
    super::field::words_to_field([block.w0(), block.w1(), block.w2(), block.w3()])
}

fn serialize_univariate<'a, F: TowerField>(
    fbb: &mut FlatBufferBuilder<'a>,
    poly: &UnivariatePoly<F>,
) -> flatbuffers::WIPOffset<fb::UnivariatePoly<'a>> {
    let coeffs: Vec<fb::Block256> = poly.evals.iter().map(|c| block_from_field(c)).collect();
    let vec = fbb.create_vector(&coeffs);

    fb::UnivariatePoly::create(
        fbb,
        &fb::UnivariatePolyArgs {
            coefficients: Some(vec),
        },
    )
}

fn serialize_sumcheck<'a, F: TowerField>(
    fbb: &mut FlatBufferBuilder<'a>,
    proof: &SumcheckProof<F>,
) -> flatbuffers::WIPOffset<fb::SumcheckProof<'a>> {
    let round_offsets: Vec<_> = proof
        .round_polys
        .iter()
        .map(|rp| serialize_univariate(fbb, rp))
        .collect();
    let rounds = fbb.create_vector(&round_offsets);
    let eval = block_from_field(&proof.claimed_evaluation);

    let skip_values: Vec<fb::Block256> = proof.skip.iter().map(block_from_field).collect();
    let skip = fbb.create_vector(&skip_values);

    fb::SumcheckProof::create(
        fbb,
        &fb::SumcheckProofArgs {
            round_polys: Some(rounds),
            claimed_evaluation: Some(&eval),
            skip: Some(skip),
        },
    )
}

fn serialize_tree_opening<'a>(
    fbb: &mut FlatBufferBuilder<'a>,
    opening: &TreeOpening,
) -> flatbuffers::WIPOffset<fb::TreeOpening<'a>> {
    let leaves = fbb.create_vector(&opening.leaves);
    let multiproof = serialize_multiproof(fbb, &opening.multiproof);

    fb::TreeOpening::create(
        fbb,
        &fb::TreeOpeningArgs {
            leaves: Some(leaves),
            multiproof: Some(multiproof),
        },
    )
}

fn serialize_fold_eval<'a, F: TowerField>(
    fbb: &mut FlatBufferBuilder<'a>,
    proof: &FoldEvalProof<F>,
) -> flatbuffers::WIPOffset<fb::FoldEvalProof<'a>> {
    let sc = serialize_sumcheck(fbb, &proof.sumcheck_proof);
    let masters = serialize_field_vec(fbb, &proof.masters);
    let mask_masters = serialize_field_vec(fbb, &proof.mask_masters);
    let fold_rounds = serialize_field_vec(fbb, &proof.fold_rounds);
    let round_roots = fbb.create_vector(&proof.round_roots);
    let final_layer = serialize_field_vec(fbb, &proof.final_layer);

    let opening_offsets: Vec<_> = proof
        .openings
        .iter()
        .map(|o| serialize_tree_opening(fbb, o))
        .collect();

    let openings = fbb.create_vector(&opening_offsets);

    fb::FoldEvalProof::create(
        fbb,
        &fb::FoldEvalProofArgs {
            sumcheck_proof: Some(sc),
            masters: Some(masters),
            mask_masters: Some(mask_masters),
            fold_rounds: Some(fold_rounds),
            round_roots: Some(round_roots),
            final_layer: Some(final_layer),
            openings: Some(openings),
        },
    )
}

fn serialize_field_vec<'a, F: TowerField>(
    fbb: &mut FlatBufferBuilder<'a>,
    values: &[F],
) -> flatbuffers::WIPOffset<flatbuffers::Vector<'a, fb::Block256>> {
    let blocks: Vec<fb::Block256> = values.iter().map(block_from_field).collect();

    fbb.create_vector(&blocks)
}

fn serialize_point_evaluation<'a, F: TowerField>(
    fbb: &mut FlatBufferBuilder<'a>,
    (point, vals): &(Vec<F>, Vec<F>),
) -> flatbuffers::WIPOffset<fb::PointEvaluation<'a>> {
    let pt: Vec<fb::Block256> = point.iter().map(|f| block_from_field(f)).collect();
    let pt_vec = fbb.create_vector(&pt);

    let cv: Vec<fb::Block256> = vals.iter().map(|f| block_from_field(f)).collect();
    let cv_vec = fbb.create_vector(&cv);

    fb::PointEvaluation::create(
        fbb,
        &fb::PointEvaluationArgs {
            point: Some(pt_vec),
            column_values: Some(cv_vec),
        },
    )
}

fn serialize_logup_aux<'a, F: TowerField>(
    fbb: &mut FlatBufferBuilder<'a>,
    aux: &LogUpAux<F>,
) -> flatbuffers::WIPOffset<fb::LogUpAux<'a>> {
    let hv: Vec<fb::Block256> = aux.h_evals.iter().map(|f| block_from_field(f)).collect();

    let h_evals = fbb.create_vector(&hv);

    let cv: Vec<fb::Block256> = aux
        .claimed_sums
        .iter()
        .map(|f| block_from_field(f))
        .collect();

    let claimed_sums = fbb.create_vector(&cv);

    fb::LogUpAux::create(
        fbb,
        &fb::LogUpAuxArgs {
            h_evals: Some(h_evals),
            claimed_sums: Some(claimed_sums),
        },
    )
}

fn deserialize_sumcheck<F: TowerField>(fb: fb::SumcheckProof<'_>) -> Result<SumcheckProof<F>> {
    let round_polys = match fb.round_polys() {
        Some(rps) => {
            let mut polys = Vec::with_capacity(rps.len());
            for i in 0..rps.len() {
                polys.push(deserialize_univariate::<F>(rps.get(i))?);
            }

            polys
        }
        None => Vec::new(),
    };

    let claimed_evaluation = match fb.claimed_evaluation() {
        Some(b) => field_from_block::<F>(*b)?,
        None => F::ZERO,
    };

    let skip = match fb.skip() {
        Some(values) => values
            .iter()
            .map(|b| field_from_block::<F>(*b))
            .collect::<Result<Vec<F>>>()?,
        None => Vec::new(),
    };

    Ok(SumcheckProof {
        skip,
        round_polys,
        claimed_evaluation,
    })
}

fn deserialize_univariate<F: TowerField>(fb: fb::UnivariatePoly<'_>) -> Result<UnivariatePoly<F>> {
    let coeffs = match fb.coefficients() {
        Some(v) => {
            let mut c = Vec::with_capacity(v.len());
            for i in 0..v.len() {
                c.push(field_from_block::<F>(*v.get(i))?);
            }

            c
        }
        None => Vec::new(),
    };

    Ok(UnivariatePoly::new(coeffs))
}

fn deserialize_fold_eval<F: TowerField>(fb: fb::FoldEvalProof<'_>) -> Result<FoldEvalProof<F>> {
    let sumcheck_proof = fb
        .sumcheck_proof()
        .map(|p| deserialize_sumcheck::<F>(p))
        .transpose()?
        .ok_or(wire_err("missing eval sumcheck_proof"))?;

    let openings = match fb.openings() {
        Some(v) => {
            let mut openings = Vec::with_capacity(v.len());
            for i in 0..v.len() {
                openings.push(deserialize_tree_opening(v.get(i))?);
            }

            openings
        }
        None => Vec::new(),
    };

    Ok(FoldEvalProof {
        sumcheck_proof,
        masters: deserialize_field_vec::<F>(fb.masters())?,
        mask_masters: deserialize_field_vec::<F>(fb.mask_masters())?,
        fold_rounds: deserialize_field_vec::<F>(fb.fold_rounds())?,
        round_roots: deserialize_bytes(fb.round_roots()).unwrap_or_default(),
        final_layer: deserialize_field_vec::<F>(fb.final_layer())?,
        openings,
    })
}

fn deserialize_point_evaluation<F: TowerField>(
    fb: fb::PointEvaluation<'_>,
) -> Result<(Vec<F>, Vec<F>)> {
    Ok((
        deserialize_field_vec::<F>(fb.point())?,
        deserialize_field_vec::<F>(fb.column_values())?,
    ))
}

fn deserialize_tree_opening(fb: fb::TreeOpening<'_>) -> Result<TreeOpening> {
    let leaves = match fb.leaves() {
        Some(v) => v.bytes().to_vec(),
        None => Vec::new(),
    };

    Ok(TreeOpening {
        leaves,
        multiproof: deserialize_multiproof(fb.multiproof())?,
    })
}

fn deserialize_logup_aux<F: TowerField>(fb: fb::LogUpAux<'_>) -> Result<LogUpAux<F>> {
    Ok(LogUpAux {
        h_evals: deserialize_field_vec(fb.h_evals())?,
        claimed_sums: deserialize_field_vec(fb.claimed_sums())?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hekate_core::errors::Error;
    use hekate_math::Block128;

    fn proof_with_trace_root(root: &[u8]) -> Result<InnerProof<Block128>> {
        let mut fbb = FlatBufferBuilder::new();
        let trace_root = fbb.create_vector(root);

        let proof = fb::Proof::create(
            &mut fbb,
            &fb::ProofArgs {
                version: WIRE_PROOF_VERSION,
                trace_root: Some(trace_root),
                ..Default::default()
            },
        );

        fbb.finish_minimal(proof);

        deserialize_proof(fbb.finished_data())
    }

    #[test]
    fn trace_root_of_any_width_passes_the_codec() {
        for len in [32usize, 64] {
            assert!(matches!(
                proof_with_trace_root(&vec![7u8; len]),
                Err(Error::Protocol {
                    message: "missing zerocheck_proof",
                    ..
                })
            ));
        }
    }

    #[test]
    fn missing_trace_root_rejected() {
        let mut fbb = FlatBufferBuilder::new();
        let proof = fb::Proof::create(
            &mut fbb,
            &fb::ProofArgs {
                version: WIRE_PROOF_VERSION,
                ..Default::default()
            },
        );

        fbb.finish_minimal(proof);

        assert!(matches!(
            deserialize_proof::<Block128>(fbb.finished_data()),
            Err(Error::Protocol {
                message: "missing trace_root",
                ..
            })
        ));
    }
}
