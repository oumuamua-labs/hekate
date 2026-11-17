// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Verifier of Hekate, the Rust zero-knowledge proof engine, in pure Rust.
//!
//! [`HekateVerifier::verify`] checks one proof against the program, its instance,
//! the pinned program id, and the prover's `Config` and transcript label.
//!
//! [`HekateVerifier::verify_prepared`] and [`HekateVerifier::verify_batch`]
//! serve many proofs of one [`prepared::PreparedProgram`]. Only `Ok(())`
//! accepts a proof; a [`rejection::Rejection`] names the check that failed.
//!
//! - [Verifier Logic][verifier]: the checks in order and reading the result
//! - [Zero Knowledge][zk]
//!
//! [verifier]: https://oumuamua.dev/hekate/docs/basics/verifier-logic
//! [zk]: https://oumuamua.dev/hekate/docs/advanced/zero-knowledge

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod fold;
mod outer;
mod sumcheck;

pub mod evaluator;
pub mod logup;
pub mod prepared;
pub mod rejection;

pub use sumcheck::{verify, verify_skip};

use crate::evaluator::{Claims, EvalOutcome, EvalVerifyContext, EvaluatorVerifier, TableClaims};
use crate::outer::PadCursor;
use crate::prepared::{PreparedProgram, VerifierScratch};
use crate::rejection::{Rejection, Table, ZerocheckStage};

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::marker::PhantomData;
use hekate_core::config::{Config, MAX_TABLE_VARS};
use hekate_core::errors;
use hekate_core::fold::{PolyKind, TreeKind};
use hekate_core::proofs::{InnerProof, LogUpAux, SumcheckProof};
use hekate_core::protocol;
use hekate_core::skip::{SkipDomain, SkipPoint};
use hekate_core::trace::TraceCompatibleField;
use hekate_crypto::transcript::Transcript;
use hekate_crypto::{Digest, Hasher};
use hekate_math::{BinaryFieldExtras, Flat, HardwareField, PackableField, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::digest::ProgramId;
use hekate_program::expander::RingSwitchPlan;
use hekate_program::fold_pool::FoldPool;
use hekate_program::outer::{
    EvalInputs, EvalRecord, OuterStatement, TableInputs, TablePads, TableRecord, TableShape,
    TableStatics, check_draws, eval_record, table_record,
};
use hekate_program::permutation::{
    self, BusFraction, PermutationCheckSpec, RankClock, RankTable, TableHeight, bus_alpha_weights,
    bus_columns, plane_basis, rank_clocks,
};
use hekate_program::{
    Air, FixedColumn, Program, ProgramInstance, ShapeEvaluator, validate_fixed_columns,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use tracing::{debug, info, instrument, trace_span, warn};

struct ZerocheckMasked<F: HardwareField> {
    claimed_sums_first: u32,
    zerocheck_first: u32,
    h_evals_first: u32,
    alpha: Flat<F>,
    r_zerocheck: Vec<Flat<F>>,
    val_final: Flat<F>,
}

/// `r_final` is the challenge list the proof carries, the skip
/// challenge first when the table skips; `point` weighs the rows.
struct ZerocheckOutcome<F: HardwareField> {
    r_final: Vec<Flat<F>>,
    point: SkipPoint<F>,
    masked: Option<ZerocheckMasked<F>>,
}

/// One chiplet's static data, built once
/// before the replay and read by every phase.
struct ChipletTable<'a, F: TowerField> {
    def: &'a ChipletDef<F>,
    shape: TableShape,
    plan: &'a RingSwitchPlan,
}

/// One table's static shape shared by its
/// ZeroCheck, eval opening and outer record.
#[derive(Clone, Copy)]
struct TableView<'a, 's, F: TowerField> {
    table: Table,
    instance: &'a ProgramInstance<F>,
    plan: &'a RingSwitchPlan,
    statics: TableStatics<'s, F>,
    shape: &'a TableShape,
    clocks: &'a [Option<RankClock>],
}

/// One table's proof parts.
#[derive(Clone, Copy)]
struct TableProof<'a, F: TowerField> {
    sc_proof: &'a SumcheckProof<F>,
    logup_aux: &'a LogUpAux<F>,
    point_evaluation: &'a (Vec<F>, Vec<F>),
}

struct CheckedTable<'a, 's, F: TowerField + HardwareField> {
    view: TableView<'a, 's, F>,
    logup_aux: &'a LogUpAux<F>,
    claims: Claims<'a, F>,
    zerocheck: ZerocheckOutcome<F>,
    claims_first: Option<u32>,
}

/// Phase-3 LogUp challenges that
/// flow into every table's ZeroCheck.
#[derive(Clone, Copy)]
struct LogUpContext<'a, F> {
    gamma: F,
    beta: F,
    lookup_bus_points: &'a BTreeMap<String, Vec<F>>,
}

/// The main Hekate Verifier for AIR circuits.
pub struct HekateVerifier<F, H> {
    _marker: PhantomData<(F, H)>,
}

impl<F, H: Hasher> HekateVerifier<F, H>
where
    F: HardwareField + PackableField + TraceCompatibleField + BinaryFieldExtras + From<u128>,
{
    /// Verifies an `InnerProof` produced by `HekateProver::prove`.
    ///
    /// Replays the prover's phase ordering:
    /// 1. Bind public inputs and config into the transcript.
    /// 2. Absorb each chiplet header, then the trace root.
    /// 3. Draw global LogUp challenges γ, β; absorb the `h` root.
    /// 4. Verify each chiplet's ZeroCheck (with LogUp).
    /// 5. Verify the main AIR ZeroCheck (with LogUp).
    /// 6. Verify every table's eval in one pool sumcheck.
    /// 7. Check that LogUp `claimed_sum` totals cancel per `bus_id`.
    #[instrument(skip_all, level = "trace", name = "Hekate::verify")]
    pub fn verify<P: Program<F> + Sync>(
        program_id: &ProgramId,
        program: &P,
        instance: &ProgramInstance<F>,
        proof: &InnerProof<F>,
        transcript: &mut Transcript<H>,
        config: &Config,
    ) -> Result<(), Rejection> {
        let prepared = trace_span!("prepare")
            .in_scope(|| PreparedProgram::new(program, config))
            .map_err(Rejection::Program)?;

        Self::verify_prepared(
            program_id,
            &prepared,
            instance,
            proof,
            transcript,
            &mut VerifierScratch::new(),
        )
    }

    /// [`Self::verify_prepared`] on each `(instance, proof)`, one
    /// proof per pool thread with its own scratch and a transcript
    /// opened with `label`; results come back in input order.
    pub fn verify_batch(
        program_id: &ProgramId,
        prepared: &PreparedProgram<F>,
        label: &'static [u8],
        proofs: &[(&ProgramInstance<F>, &InnerProof<F>)],
    ) -> Vec<Result<(), Rejection>> {
        let verify =
            |scratch: &mut VerifierScratch<F>,
             &(instance, proof): &(&ProgramInstance<F>, &InnerProof<F>)| {
                Self::verify_prepared(
                    program_id,
                    prepared,
                    instance,
                    proof,
                    &mut Transcript::new(label),
                    scratch,
                )
            };

        #[cfg(feature = "parallel")]
        let results = {
            proofs
                .par_iter()
                .map_init(VerifierScratch::new, verify)
                .collect()
        };

        #[cfg(not(feature = "parallel"))]
        let results = {
            let mut scratch = VerifierScratch::new();

            proofs
                .iter()
                .map(|item| verify(&mut scratch, item))
                .collect()
        };

        results
    }

    /// [`Self::verify`] against a program prepared once, under
    /// the `Config` it was built with; `program_id` is checked
    /// against the id computed when `prepared` was built.
    #[instrument(skip_all, level = "trace", name = "verify_prepared")]
    pub fn verify_prepared(
        program_id: &ProgramId,
        prepared: &PreparedProgram<F>,
        instance: &ProgramInstance<F>,
        proof: &InnerProof<F>,
        transcript: &mut Transcript<H>,
        scratch: &mut VerifierScratch<F>,
    ) -> Result<(), Rejection> {
        let config = &prepared.config;
        let actual = prepared.program_id;

        if actual != *program_id {
            return Err(Rejection::Program(errors::Error::ProgramIdMismatch {
                actual,
            }));
        }

        config
            .check_digest(H::Digest::BYTES.min(<ProgramId as Digest>::BYTES))
            .map_err(Rejection::Program)?;

        let shapes_span = trace_span!("table_shapes").entered();

        let num_rows = instance.num_rows();
        let num_vars = num_rows.trailing_zeros() as usize;

        // =========================================================
        // 1. CRITICAL SECURITY VALIDATIONS
        // =========================================================

        if num_rows == 0 || !num_rows.is_power_of_two() {
            return Err(Rejection::Malformed(errors::Error::Protocol {
                protocol: "verifier",
                message: "num_rows must be a non-zero power of two",
            }));
        }

        if num_vars > MAX_TABLE_VARS {
            return Err(Rejection::Malformed(errors::Error::Protocol {
                protocol: "verifier",
                message: "num_rows exceeds the table height bound",
            }));
        }

        if instance.public_inputs().len() != prepared.num_public_inputs {
            return Err(Rejection::Malformed(errors::Error::Protocol {
                protocol: "verifier",
                message: "instance public input count does not match the program",
            }));
        }

        let main_statics = prepared.main.statics();
        let main_perm = main_statics.specs;
        let main_fixed = main_statics.fixed;
        let main_plan = &prepared.main_plan;

        let field_bits = F::BITS;
        let chiplet_defs = &prepared.chiplets;

        if proof.chiplet_rows.len() != chiplet_defs.len() {
            return Err(Rejection::Malformed(errors::Error::Protocol {
                protocol: "verifier",
                message: "chiplet height count mismatch",
            }));
        }

        if proof.chiplet_point_evaluations.len() != chiplet_defs.len() {
            return Err(Rejection::Malformed(errors::Error::Protocol {
                protocol: "verifier",
                message: "chiplet point evaluation count mismatch",
            }));
        }

        validate_fixed_columns(
            main_fixed,
            Air::<F>::virtual_column_layout(&prepared.main),
            Some(num_vars),
        )?;

        let mut chiplet_tables = Vec::with_capacity(chiplet_defs.len());

        let prepared_chiplets = chiplet_defs
            .iter()
            .zip(&prepared.chiplet_plans)
            .zip(&prepared.chiplet_shapes);

        for ((def, plan), shape) in prepared_chiplets {
            let c_num_rows = proof.chiplet_rows[chiplet_tables.len()];

            if c_num_rows == 0 || !c_num_rows.is_power_of_two() {
                return Err(Rejection::Malformed(errors::Error::Protocol {
                    protocol: "verifier",
                    message: "chiplet_rows[c] must be a non-zero power of two",
                }));
            }

            let c_num_vars = c_num_rows.trailing_zeros() as usize;

            if c_num_vars > MAX_TABLE_VARS {
                return Err(Rejection::Malformed(errors::Error::Protocol {
                    protocol: "verifier",
                    message: "chiplet_rows[c] exceeds the table height bound",
                }));
            }

            validate_fixed_columns(
                def.pins(),
                Air::<F>::virtual_column_layout(def),
                Some(c_num_vars),
            )?;

            chiplet_tables.push(ChipletTable {
                def,
                shape: TableShape {
                    num_vars: c_num_vars,
                    ..*shape
                },
                plan,
            });
        }

        drop(shapes_span);

        let heights: Vec<(&RingSwitchPlan, usize)> = chiplet_tables
            .iter()
            .map(|t| (t.plan, t.shape.num_vars))
            .chain([(main_plan, num_vars)])
            .collect();

        let pool = trace_span!("fold_pool").in_scope(|| FoldPool::new(&heights, config))?;

        for (table, pooled) in chiplet_tables.iter_mut().zip(pool.tables()) {
            table.shape = table.shape.at(pooled);
        }

        let main_layout = &pool.tables()[chiplet_tables.len()];
        let main_shape = prepared.main_shape.at(main_layout);

        let mut rank_tables = Vec::with_capacity(1 + chiplet_tables.len());

        rank_tables.push(RankTable {
            specs: main_perm,
            fixed: main_fixed,
            height: TableHeight::Main(num_vars),
        });

        for table in &chiplet_tables {
            rank_tables.push(RankTable {
                specs: &table.def.permutation_checks,
                fixed: table.def.pins(),
                height: TableHeight::Chiplet(Some(table.shape.num_vars)),
            });
        }

        let clocks = trace_span!("rank_clocks").in_scope(|| rank_clocks(&rank_tables))?;

        let main_view = TableView {
            table: Table::Main,
            instance,
            plan: main_plan,
            statics: main_statics,
            shape: &main_shape,
            clocks: &clocks[0],
        };

        // =========================================================
        // PHASE 1: TRACE COMMITMENT & FIAT-SHAMIR BINDING
        // =========================================================
        let trace_root = root::<H>(&proof.trace_root)?;
        let h_root = proof.h_root.as_deref().map(root::<H>).transpose()?;
        let pad_root = proof.pad_root.as_deref().map(root::<H>).transpose()?;

        Self::bind_statement(actual, &main_view, transcript, config);

        if config.zero_knowledge != pad_root.is_some()
            || config.zero_knowledge != proof.outer.is_some()
        {
            return Err(Rejection::Malformed(errors::Error::Protocol {
                protocol: "verifier",
                message: "pad_root and outer presence must match zero_knowledge",
            }));
        }

        let mut pad_cursor = config.zero_knowledge.then(PadCursor::default);

        // =========================================================
        // PHASE 2: CHIPLET HEADERS, THEN THE POOL ROOT
        // =========================================================
        Self::absorb_chiplet_headers(&chiplet_tables, transcript);

        transcript.append_message(b"trace_root", &proof.trace_root);

        // Derived after every root,
        // bound before any challenge.
        if let Some(root) = &pad_root {
            transcript.append_message(b"pad_root", root.as_ref());
        }

        // =========================================================
        // PHASE 3: DRAW GLOBAL γ, β, AND r_bus PER LOOKUP BUS
        // =========================================================
        let security_span = trace_span!("security").entered();

        let draws = check_draws(
            config,
            &pool,
            chiplet_tables
                .iter()
                .map(|t| (t.def.statics(), t.shape))
                .chain([(main_statics, main_shape)]),
        )?;

        let fold_bits = pool.layout().estimated_security_bits();

        info!(
            bits = fold_bits.min(draws.weakest()),
            fold = fold_bits,
            radius = ?pool.layout().radius(),
            logup_bus = draws.bus,
            zerocheck = draws.zerocheck,
            skip = ?draws.skip,
            sumcheck_round = draws.sumcheck,
            eval_batch = draws.eval_batch,
            eval_round = draws.eval_round,
            outer = ?draws.outer,
            field_bits,
            "pool security"
        );

        drop(security_span);

        let gamma = transcript.challenge_field::<F>(b"bus_gamma")?;
        let beta = transcript.challenge_field::<F>(b"bus_beta")?;

        let lookup_bus_points =
            Self::draw_lookup_bus_points(main_perm, num_rows, chiplet_defs, proof, transcript)?;

        let logup = LogUpContext {
            gamma,
            beta,
            lookup_bus_points: &lookup_bus_points,
        };

        let has_helper = pool
            .layout()
            .trees()
            .iter()
            .any(|t| t.kind == TreeKind::Polys(PolyKind::Helper));

        if proof.h_root.is_some() != has_helper {
            return Err(Rejection::Malformed(errors::Error::Protocol {
                protocol: "verifier",
                message: "h_root presence must match the helper tree",
            }));
        }

        // Before the first `alpha`:
        // the `h` root binds every ZeroCheck's challenges.
        if let Some(root) = proof.h_root.as_ref() {
            transcript.append_message(b"logup_h_root", root);
        }

        // =========================================================
        // PHASE 4: PER-CHIPLET ZEROCHECK + LogUp
        // =========================================================
        let chiplet_instances: Vec<ProgramInstance<F>> = chiplet_tables
            .iter()
            .map(|t| ProgramInstance::new(1 << t.shape.num_vars, Vec::new()))
            .collect();

        let mut checked = Self::verify_chiplet_zerochecks(
            &chiplet_tables,
            &clocks[1..],
            &chiplet_instances,
            &pool,
            proof,
            transcript,
            config,
            &logup,
            pad_cursor.as_mut(),
        )?;

        // =========================================================
        // PHASE 5: MAIN AIR ZEROCHECK + LogUp
        // =========================================================
        let main_proof = TableProof {
            sc_proof: &proof.zerocheck_proof,
            logup_aux: &proof.main_logup_aux,
            point_evaluation: &proof.main_point_evaluation,
        };

        checked.push(Self::verify_table(
            main_view,
            main_proof,
            &main_layout.plan,
            transcript,
            config,
            &logup,
            pad_cursor.as_mut(),
        )?);

        // =========================================================
        // PHASE 6: POOL EVAL AT EVERY TABLE'S r_final
        // =========================================================
        let table_claims: Vec<TableClaims<F>> = checked
            .iter()
            .map(|t| TableClaims {
                point: &t.zerocheck.point,
                claims: &t.claims,
            })
            .collect();

        let ctx = EvalVerifyContext {
            pool: &pool,
            tables: &table_claims,
            masked: pad_cursor.is_some(),
        };

        let outcome = EvaluatorVerifier::<F, H>::verify(
            &trace_root,
            h_root.as_ref(),
            &proof.eval_proof,
            transcript,
            ctx,
        )?;

        // =========================================================
        // PHASE 7: CROSS-BUS MATCHING (Σ claimed_sum = 0 per group)
        // =========================================================
        if let Some(mut cursor) = pad_cursor {
            let shapes: Vec<TableShape> = checked.iter().map(|t| *t.view.shape).collect();
            let statement = trace_span!("outer_statement")
                .in_scope(|| OuterStatement::new::<F>(&shapes, pool.layout().num_vars()));
            let first_wire = shapes.iter().map(TableShape::mul_wires).sum::<usize>() as u32;

            let eval = Self::pool_eval_record(&pool, &checked, &outcome, first_wire, &mut cursor)?;
            let records = Self::table_records(&pool, &checked, &logup, config)?;

            return outer::verify_outer(
                proof,
                pad_root.as_ref(),
                transcript,
                config,
                statement,
                &records,
                &eval,
                cursor,
                scratch,
            );
        }

        let _bus_sum = trace_span!("bus_sum").entered();

        let mut columns = Vec::new();
        for table in &checked {
            let specs = table.view.statics.specs;
            let sums = &table.logup_aux.claimed_sums;
            let degree = table.view.shape.sumcheck_degree;

            for (c, column) in bus_columns(specs, degree).iter().enumerate() {
                columns.push((column.group(specs), sums[c]));
            }
        }

        logup::check_bus_sum_matching(&columns)?;

        Ok(())
    }

    /// PHASE 2:
    /// Absorb each chiplet's structure
    /// into the transcript (no ZeroCheck).
    #[instrument(skip_all, level = "trace", name = "absorb_chiplet_headers")]
    fn absorb_chiplet_headers(
        chiplet_tables: &[ChipletTable<'_, F>],
        transcript: &mut Transcript<H>,
    ) {
        for table in chiplet_tables {
            let def = table.def;

            protocol::absorb_chiplet_header(
                transcript,
                &def.name(),
                1 << table.shape.num_vars,
                def.num_columns(),
                table.plan.row_bytes(),
            );

            for bc in def.boundaries() {
                bc.absorb_into(transcript);
            }
        }
    }

    /// PHASE 4:
    /// Per-chiplet ZeroCheck, LogUp and claims.
    /// Mirrors prover's chiplet_zerochecks.
    #[instrument(skip_all, level = "trace", name = "verify_chiplet_zerochecks")]
    #[allow(clippy::too_many_arguments)]
    fn verify_chiplet_zerochecks<'a, 's>(
        chiplet_tables: &'a [ChipletTable<'s, F>],
        chiplet_clocks: &'a [Vec<Option<RankClock>>],
        chiplet_instances: &'a [ProgramInstance<F>],
        pool: &FoldPool,
        proof: &'a InnerProof<F>,
        transcript: &mut Transcript<H>,
        config: &Config,
        logup: &LogUpContext<'_, F>,
        mut cursor: Option<&mut PadCursor>,
    ) -> Result<Vec<CheckedTable<'a, 's, F>>, Rejection> {
        if proof.chiplet_zerocheck_proofs.len() != chiplet_tables.len() {
            return Err(Rejection::Malformed(errors::Error::Protocol {
                protocol: "verifier",
                message: "chiplet zerocheck proof count mismatch",
            }));
        }

        if proof.chiplet_logup_aux.len() != chiplet_tables.len() {
            return Err(Rejection::Malformed(errors::Error::Protocol {
                protocol: "verifier",
                message: "chiplet logup_aux count mismatch",
            }));
        }

        let mut checked = Vec::with_capacity(chiplet_tables.len() + 1);
        for (c_idx, table) in chiplet_tables.iter().enumerate() {
            let def = table.def;

            let view = TableView {
                table: Table::Chiplet(c_idx),
                instance: &chiplet_instances[c_idx],
                plan: table.plan,
                statics: def.statics(),
                shape: &table.shape,
                clocks: &chiplet_clocks[c_idx],
            };

            let table_proof = TableProof {
                sc_proof: &proof.chiplet_zerocheck_proofs[c_idx],
                logup_aux: &proof.chiplet_logup_aux[c_idx],
                point_evaluation: &proof.chiplet_point_evaluations[c_idx],
            };

            checked.push(Self::verify_table(
                view,
                table_proof,
                &pool.tables()[c_idx].plan,
                transcript,
                config,
                logup,
                cursor.as_deref_mut(),
            )?);

            debug!(
                chiplet_idx = c_idx,
                chiplet_name = def.name(),
                "Chiplet ZeroCheck verified"
            );
        }

        Ok(checked)
    }

    fn verify_table<'a, 's>(
        view: TableView<'a, 's, F>,
        proof: TableProof<'a, F>,
        plan: &RingSwitchPlan,
        transcript: &mut Transcript<H>,
        config: &Config,
        logup: &LogUpContext<'_, F>,
        mut cursor: Option<&mut PadCursor>,
    ) -> Result<CheckedTable<'a, 's, F>, Rejection> {
        let claims = Claims::new(&proof.point_evaluation.1);
        let half = plan.total_claims();

        if claims.flat().len() != half + plan.next_claims() {
            return Err(Rejection::Malformed(errors::Error::Protocol {
                protocol: "verifier",
                message: "combined trace values length mismatch with physical trace",
            }));
        }

        let (values, values_next) = claims.flat().split_at(half);

        let zerocheck = Self::verify_zerocheck(
            &view,
            &proof,
            transcript,
            config,
            values,
            (values_next, plan.next_slots()),
            logup,
            cursor.as_deref_mut(),
        )?;

        if canonical_slice_to_flat(&proof.point_evaluation.0) != zerocheck.r_final {
            return Err(Rejection::Malformed(errors::Error::Protocol {
                protocol: "verifier",
                message: "point evaluation mismatch with r_final",
            }));
        }

        let claims_first = cursor.map(|c| c.take(claims.flat().len()));

        transcript.append_field_list(b"claimed_val", claims.canonical());

        Ok(CheckedTable {
            view,
            logup_aux: proof.logup_aux,
            claims,
            zerocheck,
            claims_first,
        })
    }

    /// The pool eval's outer record; its rounds take
    /// the pads after every table's claim pads.
    #[instrument(skip_all, level = "trace", name = "pool_eval_record")]
    fn pool_eval_record(
        pool: &FoldPool,
        checked: &[CheckedTable<'_, '_, F>],
        outcome: &EvalOutcome<F>,
        first_wire: u32,
        cursor: &mut PadCursor,
    ) -> errors::Result<EvalRecord<F>> {
        let round_first = cursor.take(2 * pool.layout().num_vars());

        let mut claim_pads = Vec::new();
        for table in checked {
            let first = table.claims_first.ok_or(errors::Error::Protocol {
                protocol: "verifier",
                message: "masked claims carry no pads",
            })?;

            claim_pads.extend((0..table.claims.flat().len() as u32).map(|c| first + c));
        }

        eval_record(&EvalInputs {
            pool,
            eta: outcome.eta,
            rho: &outcome.rho,
            r_mix: &outcome.r_mix,
            challenges: &outcome.challenges,
            claim_masked: outcome.claim_masked,
            fin: outcome.fin,
            claim_pads: &claim_pads,
            round_first,
            first_wire,
        })
    }

    /// Each checked table's outer record.
    #[instrument(skip_all, level = "trace", name = "table_records")]
    fn table_records<'s>(
        pool: &FoldPool,
        checked: &[CheckedTable<'_, 's, F>],
        logup: &LogUpContext<'_, F>,
        config: &Config,
    ) -> errors::Result<Vec<TableRecord<'s, F>>> {
        let unpadded = || errors::Error::Protocol {
            protocol: "verifier",
            message: "masked claims carry no pads",
        };

        checked
            .iter()
            .zip(pool.tables())
            .map(|(table, pooled)| {
                let masked = table.zerocheck.masked.as_ref().ok_or_else(unpadded)?;
                let claims_first = table.claims_first.ok_or_else(unpadded)?;
                let view = &table.view;

                let _record = trace_span!("table_record", table = ?view.table).entered();

                table_record(TableInputs {
                    statics: view.statics,
                    shape: *view.shape,
                    instance: view.instance,
                    blinding_columns: config.blind_units(),
                    alpha: masked.alpha,
                    gamma: logup.gamma,
                    beta: logup.beta,
                    r_zerocheck: &masked.r_zerocheck,
                    r_final: &table.zerocheck.r_final,
                    lookup_bus_points: logup.lookup_bus_points,
                    clocks: view.clocks,
                    claimed_sums_masked: &table.logup_aux.claimed_sums,
                    h_evals_masked: &table.logup_aux.h_evals,
                    val_final_masked: masked.val_final,
                    claims_masked: table.claims.flat().to_vec(),
                    bindings: pooled.plan.binding_claims(),
                    next_slots: pooled.plan.next_slots().to_vec(),
                    pads: TablePads {
                        claimed_sums: masked.claimed_sums_first,
                        zerocheck: masked.zerocheck_first,
                        h_evals: masked.h_evals_first,
                        claims: claims_first,
                    },
                })
            })
            .collect()
    }

    /// Absorbs every public parameter before the first challenge.
    #[instrument(skip_all, level = "trace", name = "bind_statement")]
    fn bind_statement(
        program_id: ProgramId,
        main: &TableView<'_, '_, F>,
        transcript: &mut Transcript<H>,
        config: &Config,
    ) {
        let num_rows = main.instance.num_rows();
        let num_cols = main.shape.num_columns;

        transcript.append_message(b"program_id", &program_id);
        transcript.append_u64(b"num_columns", num_cols as u64);
        transcript.append_u64(b"num_rows", num_rows as u64);
        transcript.append_u64(b"zero_knowledge", u64::from(config.zero_knowledge));
        transcript.append_u64(b"num_queries", config.num_queries as u64);
        transcript.append_u64(b"log_inv_rate", u64::from(config.code_rate.log_inv_rate()));
        transcript.append_u64(b"fold_arity_log", config.fold_arity_log as u64);
        transcript.append_u64(b"early_stop_vars", config.early_stop_vars as u64);
        transcript.append_u64(b"outer_queries", config.outer_queries as u64);
        transcript.append_u64(b"main_row_bytes", main.plan.row_bytes() as u64);
        transcript.append_field_list(b"public_input", main.instance.public_inputs());

        for bc in main.statics.boundary {
            bc.absorb_into(transcript);
        }
    }

    /// Verifies AIR + LogUp ZeroCheck.
    ///
    /// The initial sumcheck claim, the bus sums at `bus_alpha_weights`, is
    /// not zero. The consistency check at `r_final` covers AIR, boundary,
    /// ZK blinding, and LogUp contributions; with a pad cursor the
    /// values are masked and the check becomes the table's record.
    #[instrument(skip_all, level = "trace", name = "verify_zerocheck")]
    #[allow(clippy::too_many_arguments)]
    fn verify_zerocheck(
        view: &TableView<'_, '_, F>,
        proof: &TableProof<'_, F>,
        transcript: &mut Transcript<H>,
        config: &Config,
        trace_values: &[Flat<F>],
        (trace_values_next, next_slots): (&[Flat<F>], &[Option<usize>]),
        logup: &LogUpContext<'_, F>,
        mut cursor: Option<&mut PadCursor>,
    ) -> Result<ZerocheckOutcome<F>, Rejection> {
        let TableView {
            table,
            instance,
            statics,
            shape,
            clocks,
            ..
        } = *view;
        let TableProof {
            sc_proof,
            logup_aux,
            ..
        } = *proof;
        let LogUpContext {
            gamma,
            beta,
            lookup_bus_points,
        } = *logup;

        let ast = statics.ast;
        let bus_specs = statics.specs;
        let trace_width = shape.num_columns;

        let num_rows = instance.num_rows();
        let num_vars = num_rows.trailing_zeros() as usize;
        let h_columns = shape.h_columns();

        let claimed_sums_first = cursor.as_deref_mut().map(|c| c.take(shape.bus_columns));

        // =========================================================
        // Constraint System Setup
        // =========================================================

        // Validate LogUp aux structure and bind claimed_sums
        // into the transcript before drawing alpha / r_zerocheck,
        // the ZeroCheck challenges depend on the bus-sum target.
        if logup_aux.claimed_sums.len() != shape.bus_columns {
            return Err(Rejection::Malformed(errors::Error::Protocol {
                protocol: "verifier",
                message: "logup_aux claimed_sums length mismatch with the h columns",
            }));
        }

        if logup_aux.h_evals.len() != h_columns {
            return Err(Rejection::Malformed(errors::Error::Protocol {
                protocol: "verifier",
                message: "logup_aux h_evals length mismatch with the h planes",
            }));
        }

        protocol::absorb_logup_claimed_sums(transcript, &logup_aux.claimed_sums);

        let alpha_tower = transcript.challenge_field::<F>(b"alpha")?;
        let alpha = alpha_tower.to_hardware();

        let r_zerocheck = (0..num_vars)
            .map(|_| {
                transcript
                    .challenge_field::<F>(b"r_zerocheck")
                    .map(|v| v.to_hardware())
            })
            .collect::<Result<Vec<_>, _>>()?;

        let boundary_constraints = statics.boundary;
        let sumcheck_degree = shape.sumcheck_degree;

        // LogUp α_pow offset must match the prover's
        // continuation past AIR + boundary + blinding.
        let logup_alpha_offset =
            ast.roots.len() + boundary_constraints.len() + config.blind_units();

        let mut alpha_logup_start = Flat::from_raw(F::ONE);
        for _ in 0..logup_alpha_offset {
            alpha_logup_start *= alpha;
        }

        let mut initial_claim = Flat::from_raw(F::ZERO);
        for (claim, w) in logup_aux
            .claimed_sums
            .iter()
            .zip(bus_alpha_weights(alpha_logup_start, alpha))
        {
            initial_claim += w.bus_sum * claim.to_hardware();
        }

        let skip = shape.skip();

        let zc_first = cursor
            .as_deref_mut()
            .map(|c| c.take(skip.sent_scalars() + (num_vars - skip.vars) * sumcheck_degree));

        let (skipped, tail_claim) = match skip.is_skip() {
            true => {
                let domain = SkipDomain::<F>::new(skip)?;
                let (r_skip, claim) =
                    verify_skip(&domain, initial_claim, &sc_proof.skip, transcript)
                        .map_err(|rejection| rejection.zerocheck(table))?;

                (Some((domain, r_skip)), claim)
            }
            false if sc_proof.skip.is_empty() => (None, initial_claim),
            false => {
                return Err(Rejection::Zerocheck {
                    table,
                    stage: ZerocheckStage::Degree,
                });
            }
        };

        let (tail, val_final) = verify(
            num_vars - skip.vars,
            sumcheck_degree,
            tail_claim,
            sc_proof,
            transcript,
        )
        .map_err(|rejection| {
            warn!(?table, "zerocheck sumcheck failed");

            rejection.zerocheck(table)
        })?;

        let h_first = cursor.map(|c| c.take(h_columns));

        protocol::absorb_logup_h_evals(transcript, &logup_aux.h_evals);

        let (r_final, point) = match skipped {
            Some((domain, r_skip)) => {
                let mut r_final = Vec::with_capacity(tail.len() + 1);
                r_final.push(r_skip);
                r_final.extend_from_slice(&tail);

                (
                    r_final,
                    SkipPoint::new(domain.lagrange_block(r_skip), tail)?,
                )
            }
            None => (tail.clone(), SkipPoint::multilinear(tail)),
        };

        // =========================================================
        // GLOBAL CONSTRAINT CONSISTENCY CHECK
        // =========================================================

        debug!("Verifying AIR constraint consistency at r_final");

        let _final = trace_span!("zerocheck_final").entered();

        if let (Some(claimed_sums_first), Some(zerocheck_first), Some(h_evals_first)) =
            (claimed_sums_first, zc_first, h_first)
        {
            return Ok(ZerocheckOutcome {
                r_final,
                point,
                masked: Some(ZerocheckMasked {
                    claimed_sums_first,
                    zerocheck_first,
                    h_evals_first,
                    alpha,
                    r_zerocheck,
                    val_final,
                }),
            });
        }

        // Every reported h_eval is the base h claim
        // the eval argument binds to the committed h.
        let first_h = trace_values.len() - h_columns - shape.h_blinds();
        let h_claims = &trace_values[first_h..first_h + h_columns];

        for (h_eval, &claim) in logup_aux.h_evals.iter().zip(h_claims) {
            if h_eval.to_hardware() != claim {
                warn!(?table, "h_eval does not match the committed h claim");
                return Err(Rejection::HEval { table });
            }
        }

        let eq_zc_eval = point.eq_with(&r_zerocheck)?;

        // Separate physical trace from blinding
        // to match AIR constraints index layout.
        let current_row = &trace_values[0..trace_width];

        let next_value = |claim: usize| next_slots[claim].map(|slot| trace_values_next[slot]);

        // Base mode verifies against the prepared plan, and preparation
        // rejects one that ships no slot for a next-row cell the AST reads.
        let next_row: Vec<Flat<F>> = (0..trace_width)
            .map(|c| next_value(c).unwrap_or(Flat::from_raw(F::ZERO)))
            .collect();

        // A. Enforce fixed columns:
        // each committed column's eval must equal its shape
        // MLE at r_final, substituted into the row ast.evaluate
        // sees so constraints use the verified value.
        let mut current_row_subst: Vec<Flat<F>> = current_row.to_vec();
        let mut shapes = ShapeEvaluator::new(&point);

        for fc in statics.fixed {
            let FixedColumn { col_idx, shape } = fc;

            if *col_idx >= trace_width {
                return Err(Rejection::Program(errors::Error::Protocol {
                    protocol: "verifier",
                    message: "fixed column col_idx out of trace_width",
                }));
            }

            let expected = shapes.evaluate(shape);

            if current_row[*col_idx] != expected {
                warn!(
                    ?table,
                    col = col_idx,
                    claimed = ?current_row[*col_idx],
                    expected = ?expected,
                    "fixed column differs from its shape"
                );

                return Err(Rejection::Zerocheck {
                    table,
                    stage: ZerocheckStage::FixedColumn(*col_idx),
                });
            }

            current_row_subst[*col_idx] = expected;
        }

        let mut expected_val = Flat::from_raw(F::ZERO);
        let mut alpha_pow = Flat::from_raw(F::ONE);
        let mut main_constraints_sum = Flat::from_raw(F::ZERO);

        // B. Main Constraints
        let constraint_evals = ast.evaluate(&current_row_subst, &next_row);

        for eval in constraint_evals {
            main_constraints_sum += eval * alpha_pow;
            alpha_pow *= alpha;
        }

        // Factor out eq_zc_eval
        expected_val += main_constraints_sum * eq_zc_eval;

        // C. Boundary Constraints
        for bc in boundary_constraints {
            if bc.col_idx >= current_row.len() {
                return Err(Rejection::Program(errors::Error::Protocol {
                    protocol: "verifier",
                    message: "boundary constraint col_idx out of bounds",
                }));
            }

            if bc.row_idx >= 1 << num_vars {
                return Err(Rejection::Malformed(errors::Error::Protocol {
                    protocol: "verifier",
                    message: "boundary constraint row_idx exceeds trace height",
                }));
            }

            let pub_val = bc.resolve_target(instance)?.to_hardware();
            let trace_val = current_row[bc.col_idx];
            let eq_eval = point.weight(bc.row_idx);

            expected_val += (trace_val - pub_val) * eq_eval * alpha_pow;
            alpha_pow *= alpha;
        }

        // D. Blinding Polynomial (Telescopic Sum)
        for k in 0..config.blind_units() {
            // Extract blinding values directly
            // from the verified proof.
            let b_k = trace_values[trace_width + k];
            let b_k_next =
                next_value(trace_width + k).ok_or(Rejection::Program(errors::Error::Protocol {
                    protocol: "verifier",
                    message: "blind unit ships no next-row claim",
                }))?;

            // b_k * alpha - b_k_next * alpha = (b_k - b_k_next) * alpha
            expected_val += (b_k - b_k_next) * alpha_pow;
            alpha_pow *= alpha;
        }

        // E. LogUp consistency + bus-sum at r_final
        if !bus_specs.is_empty() {
            let challenges = [gamma.to_hardware(), beta.to_hardware()];
            let planes = plane_basis::<F>()?;

            let mut weights = bus_alpha_weights(alpha_pow, alpha);
            for (c, column) in bus_columns(bus_specs, sumcheck_degree).iter().enumerate() {
                let fraction = |idx: usize| {
                    BusFraction::new(
                        idx,
                        &bus_specs[idx],
                        trace_width,
                        challenges,
                        &point,
                        clocks,
                    )
                };

                let first = fraction(column.first)?;
                let second = column.second.map(fraction).transpose()?;

                let g =
                    planes
                        .iter()
                        .enumerate()
                        .fold(Flat::from_raw(F::ZERO), |acc, (k, &basis)| {
                            acc + logup_aux.h_evals[planes.len() * c + k].to_hardware() * basis
                        });

                let weight = column
                    .group(bus_specs)
                    .row_weight(lookup_bus_points, &point)?;

                let terms = logup::ColumnTerms::at(&first, second.as_ref(), g, weight, current_row);

                let w = weights.next().ok_or(errors::Error::Protocol {
                    protocol: "LogUp",
                    message: "bus weights end before the columns",
                })?;

                expected_val += terms.weighted(w, eq_zc_eval);
            }
        }

        // Strict Check
        let masking_bias = val_final - expected_val;
        if masking_bias != Flat::from_raw(F::ZERO) {
            warn!(
                ?table,
                claimed = ?val_final,
                calculated = ?expected_val,
                diff = ?masking_bias,
                "constraints disagree with the zerocheck claim"
            );

            return Err(Rejection::Zerocheck {
                table,
                stage: ZerocheckStage::Constraints,
            });
        }

        Ok(ZerocheckOutcome {
            r_final,
            point,
            masked: None,
        })
    }

    /// Aggregates lookup-bus heights from main + chiplet rows
    /// and draws one `r_bus` per bus_id in sorted order.
    #[instrument(skip_all, level = "trace", name = "lookup_bus_points")]
    fn draw_lookup_bus_points(
        main_specs: &[(String, PermutationCheckSpec)],
        main_rows: usize,
        chiplet_defs: &[ChipletDef<F>],
        proof: &InnerProof<F>,
        transcript: &mut Transcript<H>,
    ) -> errors::Result<BTreeMap<String, Vec<F>>> {
        let mut heights: BTreeMap<String, u64> = BTreeMap::new();
        permutation::accumulate_lookup_heights(main_specs, main_rows as u64, &mut heights);

        for (def, &rows) in chiplet_defs.iter().zip(&proof.chiplet_rows) {
            permutation::accumulate_lookup_heights(
                &def.permutation_checks,
                rows as u64,
                &mut heights,
            );
        }

        let entries: Vec<(String, u64)> = heights.into_iter().collect();

        protocol::draw_lookup_bus_points(transcript, &entries)
    }
}

pub(crate) fn flat_matches_canonical<F: HardwareField>(value: Flat<F>, canonical: F) -> bool {
    value.to_tower() == canonical
}

fn canonical_slice_to_flat<F: HardwareField>(values: &[F]) -> Vec<Flat<F>> {
    values
        .iter()
        .copied()
        .map(|value| value.to_hardware())
        .collect()
}

pub(crate) fn root<H: Hasher>(bytes: &[u8]) -> errors::Result<H::Digest> {
    H::Digest::from_slice(bytes).ok_or(errors::Error::Protocol {
        protocol: "verifier",
        message: "root width does not match the hash",
    })
}
