// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Why the verifier rejected a proof: the first check
//! that failed, by its argument, table and stage.

use core::fmt;
use hekate_core::errors;
use hekate_crypto::transcript;

/// The first check that failed on a proof; payloads
/// carry indices only, never field values or proof bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rejection {
    /// The program, `Config`, pinned id or hasher is
    /// invalid: the same rejection for every proof.
    Program(errors::Error),

    /// The instance or the proof does not have
    /// the shape the program and `Config` require.
    Malformed(errors::Error),

    /// A table's ZeroCheck failed.
    Zerocheck {
        /// The table whose ZeroCheck failed.
        table: Table,

        /// The first check of it that failed.
        stage: ZerocheckStage,
    },

    /// In the clear, a table's reported `h` evaluation
    /// differs from its committed `h` claim.
    HEval {
        /// The table whose `h` evaluation differs.
        table: Table,
    },

    /// In the clear, the tables' claimed bus sums do not cancel.
    BusSum,

    /// The batched eval sumcheck over every table's claims failed.
    PoolEval(PoolStage),

    /// The fold opening of the masters failed.
    Fold(FoldStage),

    /// Under zero knowledge, the outer argument failed;
    /// its rows carry every check of the clear mode.
    Outer(OuterStage),
}

impl fmt::Display for Rejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Program(e) => write!(f, "invalid program or config: {e}"),
            Self::Malformed(e) => write!(f, "malformed instance or proof: {e}"),
            Self::Zerocheck { table, stage } => {
                write!(f, "{table:?} zerocheck rejected at {stage:?}")
            }
            Self::HEval { table } => {
                write!(f, "{table:?} h evaluation differs from its committed claim")
            }
            Self::BusSum => write!(f, "bus sums do not cancel"),
            Self::PoolEval(stage) => write!(f, "pool eval rejected at {stage:?}"),
            Self::Fold(stage) => write!(f, "fold opening rejected at {stage:?}"),
            Self::Outer(stage) => write!(f, "outer argument rejected at {stage:?}"),
        }
    }
}

impl core::error::Error for Rejection {}

impl From<errors::Error> for Rejection {
    fn from(value: errors::Error) -> Self {
        Self::Malformed(value)
    }
}

impl From<transcript::Error> for Rejection {
    fn from(value: transcript::Error) -> Self {
        Self::Malformed(value.into())
    }
}

/// A table of the program: the main trace or one of its chiplets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Table {
    /// The main trace.
    Main,

    /// The chiplet at this index of `chiplet_defs`.
    Chiplet(usize),
}

/// A check of a table's ZeroCheck, in
/// the order the verifier runs them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZerocheckStage {
    /// The proof carries other than one round
    /// polynomial per variable past the skip.
    Rounds,

    /// A round polynomial or the skip round carries
    /// other than its planned number of values.
    Degree,

    /// The ZeroCheck rounds reduce to a claim
    /// other than the proof's final value.
    Sumcheck,

    /// In the clear, the column's claim differs
    /// from its fixed shape at the ZeroCheck point.
    FixedColumn(usize),

    /// In the clear, the constraints, boundaries and
    /// bus terms disagree with the ZeroCheck claim.
    Constraints,
}

/// A check of the pool's eval argument,
/// in the order the verifier runs them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolStage {
    /// The eval sumcheck carries other than
    /// one round polynomial per pool variable.
    Rounds,

    /// A round polynomial carries other than its
    /// planned values, or a skip round is sent.
    Degree,

    /// The eval sumcheck's rounds reduce to
    /// a claim other than its final value.
    Sumcheck,

    /// The proof carries other than one master
    /// evaluation per committed polynomial.
    Masters,

    /// In the clear, the masters' weighted sum differs
    /// from the eval sumcheck's final claim.
    RingSwitch,
}

/// A check of the fold opening of the masters,
/// in the order the verifier runs them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FoldStage {
    /// The rounds, round roots, final layer or number
    /// of tree openings do not match the fold layout.
    Layout,

    /// The fold sumcheck's final claim disagrees with the final layer.
    FinalClaim,

    /// A tree's Merkle multiproof does not
    /// open the queried leaves to its root.
    Multiproof,

    /// A query's folded value differs from
    /// the committed word of this fold round.
    Round(usize),

    /// A query's folded value differs from the final layer's codeword.
    FinalLayer,
}

/// A check of the outer zk-Ligero argument,
/// in the order the verifier runs them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OuterStage {
    /// The masked scalars the replay consumed differ
    /// from the count the statement derives.
    PadCount,

    /// The pad or aux opening covers
    /// other than the queried columns.
    Coverage,

    /// The pad or aux opening does not
    /// authenticate against its root.
    Opening,

    /// The interleaved proximity test failed.
    Interleaved,

    /// The linear test failed: an affine row of the
    /// statement, a check of the clear mode, is false.
    Linear,

    /// The quadratic test failed:
    /// a product wire of the statement is false.
    Quadratic,
}

/// Why a sumcheck replay rejected; the caller
/// names the argument the sumcheck belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SumcheckRejection {
    /// The proof carries other than one
    /// round polynomial per variable.
    Rounds,

    /// A round polynomial or the skip round carries
    /// other than its planned number of values.
    Degree,

    /// The rounds reduce to a claim other
    /// than the proof's final value.
    Claim,

    /// The skip domain or the transcript failed.
    Malformed(errors::Error),
}

impl SumcheckRejection {
    pub(crate) fn zerocheck(self, table: Table) -> Rejection {
        let stage = match self {
            Self::Rounds => ZerocheckStage::Rounds,
            Self::Degree => ZerocheckStage::Degree,
            Self::Claim => ZerocheckStage::Sumcheck,
            Self::Malformed(e) => return Rejection::Malformed(e),
        };

        Rejection::Zerocheck { table, stage }
    }

    pub(crate) fn pool_eval(self) -> Rejection {
        let stage = match self {
            Self::Rounds => PoolStage::Rounds,
            Self::Degree => PoolStage::Degree,
            Self::Claim => PoolStage::Sumcheck,
            Self::Malformed(e) => return Rejection::Malformed(e),
        };

        Rejection::PoolEval(stage)
    }
}

impl From<errors::Error> for SumcheckRejection {
    fn from(value: errors::Error) -> Self {
        Self::Malformed(value)
    }
}

impl From<transcript::Error> for SumcheckRejection {
    fn from(value: transcript::Error) -> Self {
        Self::Malformed(value.into())
    }
}
