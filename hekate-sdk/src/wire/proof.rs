// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use super::wire_err;
use alloc::string::ToString;
use alloc::vec::Vec;
use flatbuffers::FlatBufferBuilder;
use hekate_core::errors::Result;
use hekate_core::poly::UnivariatePoly;
use hekate_core::proofs::{
    BrakedownCommitment, BrakedownProof, EvalBatchProof, InnerProof, LogUpAux, MasterEvals,
    OuterOpening, OuterProof, SumcheckProof,
};
use hekate_math::TowerField;

use crate::generated::proof as fb;

const WIRE_PROOF_VERSION: u32 = 6;

pub fn serialize_proof<'a, F: TowerField>(
    fbb: &mut FlatBufferBuilder<'a>,
    proof: &InnerProof<F>,
) -> flatbuffers::WIPOffset<fb::Proof<'a>> {
    let tc = serialize_brakedown_commitment(fbb, &proof.trace_commitment);
    let zc = serialize_sumcheck(fbb, &proof.zerocheck_proof);
    let mla = serialize_logup_aux(fbb, &proof.main_logup_aux);
    let ep = serialize_eval_batch(fbb, &proof.eval_proof);

    let cc_offsets: Vec<_> = proof
        .chiplet_commitments
        .iter()
        .map(|c| serialize_brakedown_commitment(fbb, c))
        .collect();
    let cc = fbb.create_vector(&cc_offsets);

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

    let cep_offsets: Vec<_> = proof
        .chiplet_eval_proofs
        .iter()
        .map(|p| serialize_eval_batch(fbb, p))
        .collect();
    let cep = fbb.create_vector(&cep_offsets);

    let pad_root = proof.pad_root.map(|r| fbb.create_vector(&r));
    let outer = proof.outer.as_ref().map(|o| serialize_outer(fbb, o));

    fb::Proof::create(
        fbb,
        &fb::ProofArgs {
            version: WIRE_PROOF_VERSION,
            trace_commitment: Some(tc),
            zerocheck_proof: Some(zc),
            main_logup_aux: Some(mla),
            eval_proof: Some(ep),
            chiplet_commitments: Some(cc),
            chiplet_zerocheck_proofs: Some(czc),
            chiplet_logup_aux: Some(cla),
            chiplet_eval_proofs: Some(cep),
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

    let values: Vec<fb::Block128> = opening
        .values
        .iter()
        .map(|f| block128_from_field(f))
        .collect();
    let values = fbb.create_vector(&values);

    let mut flat_siblings = Vec::with_capacity(opening.siblings.len() * 32);
    for hash in &opening.siblings {
        flat_siblings.extend_from_slice(hash);
    }

    let siblings = fbb.create_vector(&flat_siblings);

    fb::OuterOpening::create(
        fbb,
        &fb::OuterOpeningArgs {
            columns: Some(columns),
            values: Some(values),
            siblings: Some(siblings),
        },
    )
}

fn serialize_outer<'a, F: TowerField>(
    fbb: &mut FlatBufferBuilder<'a>,
    outer: &OuterProof<F>,
) -> flatbuffers::WIPOffset<fb::OuterProof<'a>> {
    let aux_root = fbb.create_vector(&outer.aux_root);

    let field_vec = |fbb: &mut FlatBufferBuilder<'a>, values: &[F]| {
        let blocks: Vec<fb::Block128> = values.iter().map(|f| block128_from_field(f)).collect();

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

    let trace_commitment = fb_proof
        .trace_commitment()
        .map(|c| deserialize_commitment(c))
        .ok_or(wire_err("missing trace_commitment"))?;

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

    let eval_proof = fb_proof
        .eval_proof()
        .map(|p| deserialize_eval_batch::<F>(p))
        .transpose()?
        .ok_or(wire_err("missing eval_proof"))?;

    let chiplet_commitments = match fb_proof.chiplet_commitments() {
        Some(v) => (0..v.len())
            .map(|i| deserialize_commitment(v.get(i)))
            .collect(),
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

    let chiplet_eval_proofs = match fb_proof.chiplet_eval_proofs() {
        Some(v) => {
            let mut proofs = Vec::with_capacity(v.len());
            for i in 0..v.len() {
                proofs.push(deserialize_eval_batch::<F>(v.get(i))?);
            }

            proofs
        }
        None => Vec::new(),
    };

    let pad_root = match fb_proof.pad_root() {
        None => None,
        Some(bytes) => {
            let raw = bytes.bytes();
            if raw.len() != 32 {
                return Err(wire_err("pad_root must be 32 bytes"));
            }

            let mut root = [0u8; 32];
            root.copy_from_slice(raw);

            Some(root)
        }
    };

    let outer = fb_proof
        .outer()
        .map(|o| deserialize_outer::<F>(o))
        .transpose()?;

    Ok(InnerProof {
        trace_commitment,
        zerocheck_proof,
        main_logup_aux,
        eval_proof,
        chiplet_commitments,
        chiplet_zerocheck_proofs,
        chiplet_logup_aux,
        chiplet_eval_proofs,
        pad_root,
        outer,
    })
}

fn deserialize_field_vec<F: TowerField>(
    v: Option<flatbuffers::Vector<'_, fb::Block128>>,
) -> Result<Vec<F>> {
    match v {
        Some(v) => {
            let mut out = Vec::with_capacity(v.len());
            for i in 0..v.len() {
                out.push(field_from_block128::<F>(*v.get(i))?);
            }

            Ok(out)
        }
        None => Ok(Vec::new()),
    }
}

fn deserialize_hashes(
    data: Option<flatbuffers::Vector<'_, u8>>,
    what: &'static str,
) -> Result<Vec<[u8; 32]>> {
    match data {
        Some(data) => {
            let bytes = data.bytes();
            if !bytes.len().is_multiple_of(32) {
                return Err(wire_err(what));
            }

            Ok(bytes.as_chunks::<32>().0.to_vec())
        }
        None => Ok(Vec::new()),
    }
}

fn deserialize_outer_opening<F: TowerField>(fb: fb::OuterOpening<'_>) -> Result<OuterOpening<F>> {
    let columns = match fb.columns() {
        Some(v) => v.iter().collect(),
        None => Vec::new(),
    };

    Ok(OuterOpening {
        columns,
        values: deserialize_field_vec::<F>(fb.values())?,
        siblings: deserialize_hashes(fb.siblings(), "outer siblings length not a multiple of 32")?,
    })
}

fn deserialize_outer<F: TowerField>(fb: fb::OuterProof<'_>) -> Result<OuterProof<F>> {
    let aux_root = match fb.aux_root() {
        Some(bytes) if bytes.len() == 32 => {
            let mut root = [0u8; 32];
            root.copy_from_slice(bytes.bytes());

            root
        }
        _ => return Err(wire_err("outer aux_root must be 32 bytes")),
    };

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

fn block128_from_field<F: TowerField>(f: &F) -> fb::Block128 {
    let (lo, hi) = super::field::field_to_lo_hi(f);
    fb::Block128::new(lo, hi)
}

fn field_from_block128<F: TowerField>(block: fb::Block128) -> Result<F> {
    super::field::lo_hi_to_field(block.lo(), block.hi())
}

fn serialize_univariate<'a, F: TowerField>(
    fbb: &mut FlatBufferBuilder<'a>,
    poly: &UnivariatePoly<F>,
) -> flatbuffers::WIPOffset<fb::UnivariatePoly<'a>> {
    let coeffs: Vec<fb::Block128> = poly.evals.iter().map(|c| block128_from_field(c)).collect();
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
    let eval = block128_from_field(&proof.claimed_evaluation);

    fb::SumcheckProof::create(
        fbb,
        &fb::SumcheckProofArgs {
            round_polys: Some(rounds),
            claimed_evaluation: Some(&eval),
        },
    )
}

fn serialize_brakedown_commitment<'a>(
    fbb: &mut FlatBufferBuilder<'a>,
    commit: &BrakedownCommitment,
) -> flatbuffers::WIPOffset<fb::BrakedownCommitment<'a>> {
    let root = fbb.create_vector(&commit.root);
    fb::BrakedownCommitment::create(
        fbb,
        &fb::BrakedownCommitmentArgs {
            root: Some(root),
            num_rows: commit.num_rows as u64,
            num_cols: commit.num_cols as u64,
        },
    )
}

fn serialize_brakedown_proof<'a, F: TowerField>(
    fbb: &mut FlatBufferBuilder<'a>,
    proof: &BrakedownProof<F>,
) -> flatbuffers::WIPOffset<fb::BrakedownProof<'a>> {
    let mut flat_cols = Vec::new();
    for col in &proof.opened_columns {
        let len_bytes = (col.len() as u32).to_le_bytes();
        flat_cols.extend_from_slice(&len_bytes);
        flat_cols.extend_from_slice(col);
    }

    let cols = fbb.create_vector(&flat_cols);

    let mut flat_path = Vec::with_capacity(proof.batch_path.len() * 32);
    for hash in &proof.batch_path {
        flat_path.extend_from_slice(hash);
    }

    let path = fbb.create_vector(&flat_path);

    fb::BrakedownProof::create(
        fbb,
        &fb::BrakedownProofArgs {
            opened_columns: Some(cols),
            batch_path: Some(path),
        },
    )
}

fn serialize_eval_batch<'a, F: TowerField>(
    fbb: &mut FlatBufferBuilder<'a>,
    proof: &EvalBatchProof<F>,
) -> flatbuffers::WIPOffset<fb::EvalBatchProof<'a>> {
    let sc = serialize_sumcheck(fbb, &proof.sumcheck_proof);
    let ldt = serialize_brakedown_proof(fbb, &proof.ldt_proof);

    let (point, vals) = &proof.point_evaluation;

    let pt: Vec<fb::Block128> = point.iter().map(|f| block128_from_field(f)).collect();
    let pt_vec = fbb.create_vector(&pt);

    let cv: Vec<fb::Block128> = vals.iter().map(|f| block128_from_field(f)).collect();
    let cv_vec = fbb.create_vector(&cv);

    let pt_offset = fb::PointEvaluation::create(
        fbb,
        &fb::PointEvaluationArgs {
            point: Some(pt_vec),
            column_values: Some(cv_vec),
        },
    );

    let tv: Vec<fb::Block128> = proof
        .tensor_vec
        .iter()
        .map(|f| block128_from_field(f))
        .collect();

    let tensor = fbb.create_vector(&tv);

    let master_evals = proof.master_evals.as_ref().map(|evals| {
        let whole = block128_from_field(&evals.whole);
        let ring = block128_from_field(&evals.ring);

        fb::MasterEvals::create(
            fbb,
            &fb::MasterEvalsArgs {
                whole: Some(&whole),
                ring: Some(&ring),
            },
        )
    });

    let h_ldt_proof = proof
        .h_ldt_proof
        .as_ref()
        .map(|p| serialize_brakedown_proof(fbb, p));

    fb::EvalBatchProof::create(
        fbb,
        &fb::EvalBatchProofArgs {
            sumcheck_proof: Some(sc),
            ldt_proof: Some(ldt),
            point_evaluation: Some(pt_offset),
            tensor_vec: Some(tensor),
            master_evals,
            h_ldt_proof,
        },
    )
}

fn serialize_logup_aux<'a, F: TowerField>(
    fbb: &mut FlatBufferBuilder<'a>,
    aux: &LogUpAux<F>,
) -> flatbuffers::WIPOffset<fb::LogUpAux<'a>> {
    let h_commitment = aux
        .h_commitment
        .as_ref()
        .map(|c| serialize_brakedown_commitment(fbb, c));

    let h_offsets: Vec<_> = aux
        .h_evals
        .iter()
        .map(|(bus_id, val)| {
            let id = fbb.create_string(bus_id);
            let block = block128_from_field(val);

            fb::LogUpEntry::create(
                fbb,
                &fb::LogUpEntryArgs {
                    bus_id: Some(id),
                    value: Some(&block),
                },
            )
        })
        .collect();
    let h_evals = fbb.create_vector(&h_offsets);

    let cs_offsets: Vec<_> = aux
        .claimed_sums
        .iter()
        .map(|(bus_id, val)| {
            let id = fbb.create_string(bus_id);
            let block = block128_from_field(val);

            fb::LogUpEntry::create(
                fbb,
                &fb::LogUpEntryArgs {
                    bus_id: Some(id),
                    value: Some(&block),
                },
            )
        })
        .collect();
    let claimed_sums = fbb.create_vector(&cs_offsets);

    fb::LogUpAux::create(
        fbb,
        &fb::LogUpAuxArgs {
            h_evals: Some(h_evals),
            claimed_sums: Some(claimed_sums),
            h_commitment,
        },
    )
}

fn deserialize_commitment(fb: fb::BrakedownCommitment<'_>) -> BrakedownCommitment {
    let mut root = [0u8; 32];
    if let Some(r) = fb.root() {
        let len = r.len().min(32);
        root[..len].copy_from_slice(&r.bytes()[..len]);
    }

    BrakedownCommitment {
        root,
        num_rows: fb.num_rows() as usize,
        num_cols: fb.num_cols() as usize,
    }
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
        Some(b) => field_from_block128::<F>(*b)?,
        None => F::ZERO,
    };

    Ok(SumcheckProof {
        round_polys,
        claimed_evaluation,
    })
}

fn deserialize_univariate<F: TowerField>(fb: fb::UnivariatePoly<'_>) -> Result<UnivariatePoly<F>> {
    let coeffs = match fb.coefficients() {
        Some(v) => {
            let mut c = Vec::with_capacity(v.len());
            for i in 0..v.len() {
                c.push(field_from_block128::<F>(*v.get(i))?);
            }

            c
        }
        None => Vec::new(),
    };

    Ok(UnivariatePoly::new(coeffs))
}

fn deserialize_eval_batch<F: TowerField>(fb: fb::EvalBatchProof<'_>) -> Result<EvalBatchProof<F>> {
    let sumcheck_proof = fb
        .sumcheck_proof()
        .map(|p| deserialize_sumcheck::<F>(p))
        .transpose()?
        .ok_or(wire_err("missing eval sumcheck_proof"))?;

    let ldt_proof = fb
        .ldt_proof()
        .map(|p| deserialize_brakedown_proof::<F>(p))
        .transpose()?
        .ok_or(wire_err("missing eval ldt_proof"))?;

    let pt = fb
        .point_evaluation()
        .ok_or(wire_err("missing eval point_evaluation"))?;

    let point: Vec<F> = match pt.point() {
        Some(v) => {
            let mut p = Vec::with_capacity(v.len());
            for j in 0..v.len() {
                p.push(field_from_block128::<F>(*v.get(j))?);
            }

            p
        }
        None => Vec::new(),
    };

    let vals: Vec<F> = match pt.column_values() {
        Some(v) => {
            let mut cv = Vec::with_capacity(v.len());
            for j in 0..v.len() {
                cv.push(field_from_block128::<F>(*v.get(j))?);
            }

            cv
        }
        None => Vec::new(),
    };

    let point_evaluation = (point, vals);

    let tensor_vec = match fb.tensor_vec() {
        Some(v) => {
            let mut tv = Vec::with_capacity(v.len());
            for i in 0..v.len() {
                tv.push(field_from_block128::<F>(*v.get(i))?);
            }

            tv
        }
        None => Vec::new(),
    };

    let master_evals = match fb.master_evals() {
        Some(evals) => {
            let whole = evals
                .whole()
                .ok_or(wire_err("missing master_evals.whole"))?;
            let ring = evals.ring().ok_or(wire_err("missing master_evals.ring"))?;

            Some(MasterEvals {
                whole: field_from_block128::<F>(*whole)?,
                ring: field_from_block128::<F>(*ring)?,
            })
        }
        None => None,
    };

    let h_ldt_proof = fb
        .h_ldt_proof()
        .map(|p| deserialize_brakedown_proof::<F>(p))
        .transpose()?;

    Ok(EvalBatchProof {
        sumcheck_proof,
        ldt_proof,
        point_evaluation,
        tensor_vec,
        master_evals,
        h_ldt_proof,
    })
}

fn deserialize_brakedown_proof<F: TowerField>(
    fb: fb::BrakedownProof<'_>,
) -> Result<BrakedownProof<F>> {
    let opened_columns = match fb.opened_columns() {
        Some(data) => {
            let bytes = data.bytes();

            let mut cols = Vec::new();
            let mut offset = 0;

            while offset + 4 <= bytes.len() {
                let len =
                    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
                offset += 4;

                if len > bytes.len() - offset {
                    return Err(wire_err("truncated opened_columns data"));
                }

                cols.push(bytes[offset..offset + len].to_vec());
                offset += len;
            }

            cols
        }
        None => Vec::new(),
    };

    let batch_path = match fb.batch_path() {
        Some(data) => {
            let bytes = data.bytes();

            if !bytes.len().is_multiple_of(32) {
                return Err(wire_err("batch_path length not a multiple of 32"));
            }

            bytes.as_chunks::<32>().0.to_vec()
        }
        None => Vec::new(),
    };

    Ok(BrakedownProof::new(opened_columns, batch_path))
}

fn deserialize_logup_aux<F: TowerField>(fb: fb::LogUpAux<'_>) -> Result<LogUpAux<F>> {
    let h_evals = match fb.h_evals() {
        Some(v) => {
            let mut entries = Vec::with_capacity(v.len());
            for i in 0..v.len() {
                let entry = v.get(i);
                let bus_id = entry.bus_id().unwrap_or("").to_string();

                let val = match entry.value() {
                    Some(b) => field_from_block128::<F>(*b)?,
                    None => F::ZERO,
                };

                entries.push((bus_id, val));
            }

            entries
        }
        None => Vec::new(),
    };

    let claimed_sums = match fb.claimed_sums() {
        Some(v) => {
            let mut entries = Vec::with_capacity(v.len());
            for i in 0..v.len() {
                let entry = v.get(i);
                let bus_id = entry.bus_id().unwrap_or("").to_string();

                let val = match entry.value() {
                    Some(b) => field_from_block128::<F>(*b)?,
                    None => F::ZERO,
                };

                entries.push((bus_id, val));
            }

            entries
        }
        None => Vec::new(),
    };

    let h_commitment = fb.h_commitment().map(deserialize_commitment);

    Ok(LogUpAux {
        h_evals,
        claimed_sums,
        h_commitment,
    })
}
