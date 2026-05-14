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

use super::wire_err;
use alloc::string::ToString;
use alloc::vec::Vec;
use flatbuffers::FlatBufferBuilder;
use hekate_core::errors::Result;
use hekate_core::poly::UnivariatePoly;
use hekate_core::proofs::{
    BrakedownCommitment, BrakedownProof, EvalBatchProof, InnerProof, LogUpAux, SumcheckProof,
};
use hekate_math::TowerField;

use crate::generated::proof as fb;

const WIRE_PROOF_VERSION: u32 = 1;

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

    fb::Proof::create(
        fbb,
        &fb::ProofArgs {
            version: 1,
            trace_commitment: Some(tc),
            zerocheck_proof: Some(zc),
            main_logup_aux: Some(mla),
            eval_proof: Some(ep),
            chiplet_commitments: Some(cc),
            chiplet_zerocheck_proofs: Some(czc),
            chiplet_logup_aux: Some(cla),
            chiplet_eval_proofs: Some(cep),
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

    Ok(InnerProof {
        trace_commitment,
        zerocheck_proof,
        main_logup_aux,
        eval_proof,
        chiplet_commitments,
        chiplet_zerocheck_proofs,
        chiplet_logup_aux,
        chiplet_eval_proofs,
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
    let ldt_offsets: Vec<_> = proof
        .ldt_proofs
        .iter()
        .map(|path| {
            let mut flat = Vec::with_capacity(path.len() * 32);
            for hash in path {
                flat.extend_from_slice(hash);
            }

            let data = fbb.create_vector(&flat);

            fb::MerklePath::create(fbb, &fb::MerklePathArgs { hashes: Some(data) })
        })
        .collect();
    let ldt = fbb.create_vector(&ldt_offsets);

    let mut flat_cols = Vec::new();
    for col in &proof.opened_columns {
        let len_bytes = (col.len() as u32).to_le_bytes();
        flat_cols.extend_from_slice(&len_bytes);
        flat_cols.extend_from_slice(col);
    }

    let cols = fbb.create_vector(&flat_cols);

    fb::BrakedownProof::create(
        fbb,
        &fb::BrakedownProofArgs {
            ldt_proofs: Some(ldt),
            opened_columns: Some(cols),
        },
    )
}

fn serialize_eval_batch<'a, F: TowerField>(
    fbb: &mut FlatBufferBuilder<'a>,
    proof: &EvalBatchProof<F>,
) -> flatbuffers::WIPOffset<fb::EvalBatchProof<'a>> {
    let sc = serialize_sumcheck(fbb, &proof.sumcheck_proof);
    let ldt = serialize_brakedown_proof(fbb, &proof.ldt_proof);

    let pt_offsets: Vec<_> = proof
        .point_evaluations
        .iter()
        .map(|(point, vals)| {
            let pt: Vec<fb::Block128> = point.iter().map(|f| block128_from_field(f)).collect();
            let pt_vec = fbb.create_vector(&pt);
            let cv: Vec<fb::Block128> = vals.iter().map(|f| block128_from_field(f)).collect();
            let cv_vec = fbb.create_vector(&cv);

            fb::PointEvaluation::create(
                fbb,
                &fb::PointEvaluationArgs {
                    point: Some(pt_vec),
                    column_values: Some(cv_vec),
                },
            )
        })
        .collect();
    let pts = fbb.create_vector(&pt_offsets);

    let tv: Vec<fb::Block128> = proof
        .tensor_vec
        .iter()
        .map(|f| block128_from_field(f))
        .collect();
    let tensor = fbb.create_vector(&tv);

    fb::EvalBatchProof::create(
        fbb,
        &fb::EvalBatchProofArgs {
            sumcheck_proof: Some(sc),
            ldt_proof: Some(ldt),
            point_evaluations: Some(pts),
            tensor_vec: Some(tensor),
        },
    )
}

fn serialize_logup_aux<'a, F: TowerField>(
    fbb: &mut FlatBufferBuilder<'a>,
    aux: &LogUpAux<F>,
) -> flatbuffers::WIPOffset<fb::LogUpAux<'a>> {
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

    let point_evaluations = match fb.point_evaluations() {
        Some(pts) => {
            let mut evals = Vec::with_capacity(pts.len());
            for i in 0..pts.len() {
                let pt = pts.get(i);

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

                evals.push((point, vals));
            }

            evals
        }
        None => Vec::new(),
    };

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

    Ok(EvalBatchProof {
        sumcheck_proof,
        ldt_proof,
        point_evaluations,
        tensor_vec,
    })
}

fn deserialize_brakedown_proof<F: TowerField>(
    fb: fb::BrakedownProof<'_>,
) -> Result<BrakedownProof<F>> {
    let ldt_proofs = match fb.ldt_proofs() {
        Some(paths) => {
            let mut proofs = Vec::with_capacity(paths.len());
            for i in 0..paths.len() {
                let path = paths.get(i);
                let hashes = match path.hashes() {
                    Some(data) => data
                        .bytes()
                        .chunks_exact(32)
                        .map(|c| {
                            let mut h = [0u8; 32];
                            h.copy_from_slice(c);

                            h
                        })
                        .collect(),
                    None => Vec::new(),
                };

                proofs.push(hashes);
            }

            proofs
        }
        None => Vec::new(),
    };

    let opened_columns = match fb.opened_columns() {
        Some(data) => {
            let bytes = data.bytes();

            let mut cols = Vec::new();
            let mut offset = 0;

            while offset + 4 <= bytes.len() {
                let len =
                    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
                offset += 4;

                if offset + len > bytes.len() {
                    return Err(wire_err("truncated opened_columns data"));
                }

                cols.push(bytes[offset..offset + len].to_vec());
                offset += len;
            }

            cols
        }
        None => Vec::new(),
    };

    Ok(BrakedownProof::new(ldt_proofs, opened_columns))
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

    Ok(LogUpAux {
        h_evals,
        claimed_sums,
    })
}
