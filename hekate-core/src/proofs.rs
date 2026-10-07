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

/// The prover's full transcript-independent output; the
/// k-th chiplet contributes index k of every `chiplet_*` vector.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InnerProof<F: TowerField> {
    /// Root over every table's trace codewords.
    pub trace_root: [u8; 32],

    /// Root over every bus-carrying table's
    /// `h` codewords, absorbed before any `α`.
    pub h_root: Option<[u8; 32]>,

    /// Sumcheck proving
    /// `Σ_x ( Σ α_i · C_i(x) ) · eq(r, x) = 0`
    /// for the main AIR.
    pub zerocheck_proof: SumcheckProof<F>,

    /// `h_k(r_final)` and `Σ h_k[i]` for main-trace
    /// bus endpoints, pinned to the committed `h`.
    pub main_logup_aux: LogUpAux<F>,

    /// `(r_final, claimed_column_evals)` of main.
    pub main_point_evaluation: (Vec<F>, Vec<F>),

    /// Pins every table's trace evaluations to the two roots.
    pub eval_proof: EvalBatchProof<F>,

    pub chiplet_rows: Vec<usize>,
    pub chiplet_zerocheck_proofs: Vec<SumcheckProof<F>>,
    pub chiplet_logup_aux: Vec<LogUpAux<F>>,
    pub chiplet_point_evaluations: Vec<(Vec<F>, Vec<F>)>,

    /// Absorbed before the first challenge.
    pub pad_root: Option<[u8; 32]>,

    /// Present iff `pad_root` is.
    pub outer: Option<OuterProof<F>>,
}

impl<F: TowerField> InnerProof<F> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        trace_root: [u8; 32],
        h_root: Option<[u8; 32]>,
        zerocheck_proof: SumcheckProof<F>,
        main_logup_aux: LogUpAux<F>,
        main_point_evaluation: (Vec<F>, Vec<F>),
        eval_proof: EvalBatchProof<F>,
        chiplet_rows: Vec<usize>,
        chiplet_zerocheck_proofs: Vec<SumcheckProof<F>>,
        chiplet_logup_aux: Vec<LogUpAux<F>>,
        chiplet_point_evaluations: Vec<(Vec<F>, Vec<F>)>,
        pad_root: Option<[u8; 32]>,
        outer: Option<OuterProof<F>>,
    ) -> Self {
        Self {
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

/// Evaluation argument binding every table's trace-column
/// evaluations to the pool's two Brakedown commitments.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvalBatchProof<F: TowerField> {
    /// Sumcheck reducing the batched claim
    /// to a single-point evaluation.
    pub sumcheck_proof: SumcheckProof<F>,

    /// The trace tree opened at the query columns.
    pub ldt_proof: BrakedownProof<F>,

    /// `Σ_k λ^k · q_k` over the masters' folds.
    /// Length `grid_cols + support_size`.
    pub tensor_vec: Vec<F>,

    /// Every table's master evaluations at `r`, pool order,
    /// ring before whole, fixed before `λ` binds them.
    pub masters: Vec<F>,

    /// The `h` tree opened at the same query columns;
    /// present iff some table carries a bus.
    pub h_ldt_proof: Option<BrakedownProof<F>>,
}

impl<F: TowerField> EvalBatchProof<F> {
    pub fn new(
        sumcheck_proof: SumcheckProof<F>,
        ldt_proof: BrakedownProof<F>,
        tensor_vec: Vec<F>,
        masters: Vec<F>,
        h_ldt_proof: Option<BrakedownProof<F>>,
    ) -> Self {
        Self {
            sumcheck_proof,
            ldt_proof,
            tensor_vec,
            masters,
            h_ldt_proof,
        }
    }
}

// ===================================
// LOGUP AUXILIARY
// ===================================

/// Per-table LogUp auxiliary payload keyed by `bus_id`.
/// `claimed_sums[i]` is absorbed pre-`α`/`r_zerocheck`;
/// `h_evals[i]` post-sumcheck.
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
