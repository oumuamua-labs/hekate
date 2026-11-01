// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use crate::poly::univariate::UnivariatePoly;
use alloc::vec::Vec;
use hekate_crypto::merkle::Multiproof;
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

    /// Root of the helper tree: every table's `h` and, with
    /// zero knowledge, the two masks; absorbed before any `α`.
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
    pub eval_proof: FoldEvalProof<F>,

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
        eval_proof: FoldEvalProof<F>,
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
    pub multiproof: Multiproof,
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

/// One fold tree opened at the queries: its distinct
/// leaves in ascending index, then their paths.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TreeOpening {
    /// Untagged leaf preimages back to back: each symbol as flat LE
    /// bytes of its code field, a fold-field one as `lo` then `hi`.
    pub leaves: Vec<u8>,

    /// Opens every leaf against the tree's root.
    pub multiproof: Multiproof,
}

/// The fold's evaluation argument: batching sumcheck,
/// masters, fold rounds, final layer and openings.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FoldEvalProof<F: TowerField> {
    /// Reduces every claim to the masters at one point `r`.
    pub sumcheck_proof: SumcheckProof<F>,

    /// `M_k`, polynomial `k` at the suffix of `r`, derived order.
    pub masters: Vec<F>,

    /// The masks' masters at the suffix `r[a..]`:
    /// two in zero-knowledge mode, none otherwise.
    pub mask_masters: Vec<F>,

    /// Each fold round's `g(1)` and `X²` coefficient over the
    /// fold field, as `F` limbs; `g(0)` comes from the claim.
    pub fold_rounds: Vec<F>,

    /// Roots of the round codewords `H_1 .. H_{R−1}`.
    pub round_roots: Vec<[u8; 32]>,

    /// `H_R`'s message, `2^s` fold-field symbols as `F` limbs.
    pub final_layer: Vec<F>,

    /// One per tree of the fold layout, in its order.
    pub openings: Vec<TreeOpening>,
}

// ===================================
// LOGUP AUXILIARY
// ===================================

/// Per-table LogUp auxiliary payload
/// in the table's limb-column order.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogUpAux<F: TowerField> {
    /// Absorbed after the ZeroCheck sumcheck.
    pub h_evals: Vec<F>,

    /// Absorbed before `α` and `r_zerocheck`.
    pub claimed_sums: Vec<F>,
}

impl<F: TowerField> LogUpAux<F> {
    pub fn new(h_evals: Vec<F>, claimed_sums: Vec<F>) -> Self {
        Self {
            h_evals,
            claimed_sums,
        }
    }
}
