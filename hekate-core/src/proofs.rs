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

use crate::poly::univariate::UnivariatePoly;
use alloc::string::String;
use alloc::vec::Vec;
use core::marker::PhantomData;
use hekate_math::TowerField;
use serde::{Deserialize, Serialize};

// ===================================
// PROGRAM INNER PROOF
// ===================================

/// The prover's full transcript-independent
/// output. Main-trace and chiplet-trace
/// sub-vectors are parallel:
/// the k-th chiplet contributes
/// `(chiplet_commitments[k], chiplet_zerocheck_proofs[k],
/// chiplet_logup_aux[k], chiplet_eval_proofs[k])`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InnerProof<F: TowerField> {
    pub trace_commitment: BrakedownCommitment,

    /// Sumcheck proving
    /// `Σ_x ( Σ α_i · C_i(x) ) · eq(r, x) = 0`
    /// for the main AIR.
    pub zerocheck_proof: SumcheckProof<F>,

    /// `h_k(r_final)` and `Σ h_k[i]` for main-trace
    /// bus endpoints. `h_k` is not Merkle-committed
    /// so its evaluation must travel with the proof.
    pub main_logup_aux: LogUpAux<F>,

    /// Pins trace evaluations to `trace_commitment`.
    /// Blocks disconnected-witness / floating-proof
    /// forgeries.
    pub eval_proof: EvalBatchProof<F>,

    pub chiplet_commitments: Vec<BrakedownCommitment>,
    pub chiplet_zerocheck_proofs: Vec<SumcheckProof<F>>,
    pub chiplet_logup_aux: Vec<LogUpAux<F>>,
    pub chiplet_eval_proofs: Vec<EvalBatchProof<F>>,
}

impl<F: TowerField> InnerProof<F> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        trace_commitment: BrakedownCommitment,
        zerocheck_proof: SumcheckProof<F>,
        main_logup_aux: LogUpAux<F>,
        eval_proof: EvalBatchProof<F>,
        chiplet_commitments: Vec<BrakedownCommitment>,
        chiplet_zerocheck_proofs: Vec<SumcheckProof<F>>,
        chiplet_logup_aux: Vec<LogUpAux<F>>,
        chiplet_eval_proofs: Vec<EvalBatchProof<F>>,
    ) -> Self {
        Self {
            trace_commitment,
            zerocheck_proof,
            main_logup_aux,
            eval_proof,
            chiplet_commitments,
            chiplet_zerocheck_proofs,
            chiplet_logup_aux,
            chiplet_eval_proofs,
        }
    }
}

// ===================================
// BRAKEDOWN PROOF
// ===================================

/// LDT opening payload for one Brakedown commitment:
/// the opened encoded columns and one octopus multiproof
/// that a verifier replays against the commitment root.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrakedownProof<F: TowerField> {
    /// Raw bytes of the opened 2D columns, one per
    /// distinct queried index in ascending order.
    /// Only encoded code bytes, data stays private.
    pub opened_columns: Vec<Vec<u8>>,

    /// Octopus multiproof: pruned Merkle sibling
    /// set covering all queried columns.
    pub batch_path: Vec<[u8; 32]>,

    _marker: PhantomData<F>,
}

impl<F: TowerField> BrakedownProof<F> {
    pub fn new(opened_columns: Vec<Vec<u8>>, batch_path: Vec<[u8; 32]>) -> Self {
        Self {
            opened_columns,
            batch_path,
            _marker: PhantomData,
        }
    }
}

/// Merkle root plus the dimensions it was
/// taken over. `num_rows` and `num_cols`
/// must be absorbed into the transcript
/// before any challenge is drawn.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BrakedownCommitment {
    pub root: [u8; 32],
    pub num_rows: usize,
    pub num_cols: usize,
}

// ===================================
// SUMCHECK PROOF
// ===================================

/// Per-round univariates `g_j(X)` plus
/// the prover's terminal evaluation at
/// the random challenge point `r`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SumcheckProof<F: TowerField> {
    /// Per-round univariate `g_j(X)`.
    pub round_polys: Vec<UnivariatePoly<F>>,

    /// `C(r_0, ..., r_{k-1})` claimed by the prover.
    pub claimed_evaluation: F,
}

// ===================================
// EVALUATION BATCH PROOF
// ===================================

/// Multi-point evaluation argument binding
/// trace-column evaluations at several challenge
/// points to one Brakedown commitment.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvalBatchProof<F: TowerField> {
    /// Sumcheck reducing the RLC-batched multi-point
    /// claim to a single-point evaluation.
    pub sumcheck_proof: SumcheckProof<F>,

    /// Brakedown opening for the rows selected
    /// by the evaluation sumcheck's challenge.
    pub ldt_proof: BrakedownProof<F>,

    /// `(point, claimed_column_evals)` per
    /// batched query. Currently, always length 1:
    /// `(r_final, AIR column evals)`.
    pub point_evaluations: Vec<(Vec<F>, Vec<F>)>,

    /// TensorPCS intermediate fold
    /// `q = M · r_col`, length `sqrt(N)`.
    #[serde(default)]
    pub tensor_vec: Vec<F>,
}

impl<F: TowerField> EvalBatchProof<F> {
    pub fn new(
        sumcheck_proof: SumcheckProof<F>,
        ldt_proof: BrakedownProof<F>,
        point_evaluations: Vec<(Vec<F>, Vec<F>)>,
        tensor_vec: Vec<F>,
    ) -> Self {
        Self {
            sumcheck_proof,
            ldt_proof,
            point_evaluations,
            tensor_vec,
        }
    }
}

// ===================================
// LOGUP AUXILIARY
// ===================================

/// Per-table LogUp auxiliary payload keyed
/// by `bus_id`. `claimed_sums[i]` is absorbed
/// pre-`α`/`r_zerocheck`; `h_evals[i]` is
/// absorbed post-sumcheck.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogUpAux<F: TowerField> {
    pub h_evals: Vec<(String, F)>,
    pub claimed_sums: Vec<(String, F)>,
}

impl<F: TowerField> LogUpAux<F> {
    pub fn new(h_evals: Vec<(String, F)>, claimed_sums: Vec<(String, F)>) -> Self {
        Self {
            h_evals,
            claimed_sums,
        }
    }
}
