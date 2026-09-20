// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use crate::poly::univariate::UnivariatePoly;
use alloc::string::String;
use alloc::vec::Vec;
use core::marker::PhantomData;
use hekate_math::TowerField;
use serde::{Deserialize, Serialize};

// ===================================
// PROGRAM INNER PROOF
// ===================================

/// The prover's full transcript-independent output.
/// Main-trace and chiplet-trace sub-vectors are parallel:
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
    /// bus endpoints, pinned to the committed `h`.
    pub main_logup_aux: LogUpAux<F>,

    /// Pins trace evaluations to `trace_commitment`.
    /// Blocks disconnected-witness / floating-proof
    /// forgeries.
    pub eval_proof: EvalBatchProof<F>,

    pub chiplet_commitments: Vec<BrakedownCommitment>,
    pub chiplet_zerocheck_proofs: Vec<SumcheckProof<F>>,
    pub chiplet_logup_aux: Vec<LogUpAux<F>>,
    pub chiplet_eval_proofs: Vec<EvalBatchProof<F>>,

    /// Absorbed before the first challenge.
    pub pad_root: Option<[u8; 32]>,

    /// Present iff `pad_root` is.
    pub outer: Option<OuterProof<F>>,
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
        pad_root: Option<[u8; 32]>,
        outer: Option<OuterProof<F>>,
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
            pad_root,
            outer,
        }
    }
}

// ===================================
// OUTER ARGUMENT
// ===================================

/// One oracle's opened columns: `values` holds
/// every column's rows back to back, in `columns`
/// order; its length is `columns.len() * rows`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OuterOpening<F: TowerField> {
    pub columns: Vec<u32>,
    pub values: Vec<F>,
    pub siblings: Vec<[u8; 32]>,
}

/// The zk-Ligero segment after the base protocol:
/// AUX root, the three test responses, and both
/// oracles opened at the same query columns.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OuterProof<F: TowerField> {
    pub aux_root: [u8; 32],
    pub interleaved: Vec<F>,
    pub linear: Vec<F>,
    pub quadratic: Vec<F>,
    pub pad_opening: OuterOpening<F>,
    pub aux_opening: OuterOpening<F>,
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

/// Merkle root plus the dimensions it was taken over.
/// `num_rows` and `num_cols` must be absorbed into
/// the transcript before any challenge is drawn.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BrakedownCommitment {
    pub root: [u8; 32],
    pub num_rows: usize,
    pub num_cols: usize,
}

// ===================================
// SUMCHECK PROOF
// ===================================

/// Per-round univariates `g_j(X)` plus the prover's
/// terminal evaluation at the random challenge point `r`.
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

/// The two master evaluations at `r'`, fixed before
/// the line challenge `λ` binds them to one vector.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MasterEvals<F: TowerField> {
    pub whole: F,
    pub ring: F,
}

/// Single-point evaluation argument binding
/// trace-column evaluations at one challenge
/// point to the table's Brakedown commitments.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvalBatchProof<F: TowerField> {
    /// Sumcheck reducing the batched claim
    /// to a single-point evaluation.
    pub sumcheck_proof: SumcheckProof<F>,

    /// Brakedown opening for the rows selected
    /// by the evaluation sumcheck's challenge.
    pub ldt_proof: BrakedownProof<F>,

    /// `(r_final, claimed_column_evals)`.
    pub point_evaluation: (Vec<F>, Vec<F>),

    /// `q_whole = M_whole · r_col`, or the line
    /// `q_ring + λ · q_whole` with a ring unit.
    /// Length `grid_cols + support_size`.
    pub tensor_vec: Vec<F>,

    /// Present iff the ring-switch plan carries a ring unit.
    pub master_evals: Option<MasterEvals<F>>,

    /// The table's `h` tree opened at the same query
    /// columns; present iff the table carries a bus.
    pub h_ldt_proof: Option<BrakedownProof<F>>,
}

impl<F: TowerField> EvalBatchProof<F> {
    pub fn new(
        sumcheck_proof: SumcheckProof<F>,
        ldt_proof: BrakedownProof<F>,
        point_evaluation: (Vec<F>, Vec<F>),
        tensor_vec: Vec<F>,
        master_evals: Option<MasterEvals<F>>,
        h_ldt_proof: Option<BrakedownProof<F>>,
    ) -> Self {
        Self {
            sumcheck_proof,
            ldt_proof,
            point_evaluation,
            tensor_vec,
            master_evals,
            h_ldt_proof,
        }
    }
}

// ===================================
// LOGUP AUXILIARY
// ===================================

/// Per-table LogUp auxiliary payload keyed by `bus_id`.
/// `claimed_sums[i]` and `h_commitment` are absorbed
/// pre-`α`/`r_zerocheck`; `h_evals[i]` post-sumcheck.
/// `h_commitment` is `None` iff the table carries no bus;
/// the table's eval argument opens it at `r_final`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogUpAux<F: TowerField> {
    pub h_evals: Vec<(String, F)>,
    pub claimed_sums: Vec<(String, F)>,
    pub h_commitment: Option<BrakedownCommitment>,
}

impl<F: TowerField> LogUpAux<F> {
    pub fn new(h_evals: Vec<(String, F)>, claimed_sums: Vec<(String, F)>) -> Self {
        Self {
            h_evals,
            claimed_sums,
            h_commitment: None,
        }
    }
}
