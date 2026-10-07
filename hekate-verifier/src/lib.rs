// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod brakedown;
mod outer;
mod sumcheck;

pub mod evaluator;
pub mod logup;
pub mod prepared;

pub use sumcheck::verify;

use crate::evaluator::{Claims, EvalOutcome, EvalVerifyContext, EvaluatorVerifier, TableClaims};
use crate::outer::PadCursor;
use crate::prepared::{PreparedProgram, VerifierScratch};

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::marker::PhantomData;
use hekate_core::config::{Config, MAX_TABLE_VARS};
use hekate_core::errors;
use hekate_core::proofs::{InnerProof, LogUpAux, SumcheckProof};
use hekate_core::protocol;
use hekate_core::tensor::TensorProduct;
use hekate_core::trace::TraceCompatibleField;
use hekate_crypto::Hasher;
use hekate_crypto::transcript::Transcript;
use hekate_math::{BinaryFieldExtras, Block128, Flat, HardwareField, PackableField, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::expander::{PoolLayout, RingSwitchPlan};
use hekate_program::outer::{
    EvalInputs, EvalRecord, OuterStatement, TableInputs, TablePads, TableRecord, TableShape,
    TableStatics, eval_record, table_record,
};
use hekate_program::permutation::{
    self, BusKind, PermutationCheckSpec, RankClock, RankTable, TableHeight, eval_row_idx_byte_mle,
    eval_row_idx_le_mle, rank_clocks,
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

struct ZerocheckOutcome<F: HardwareField> {
    r_final: Vec<Flat<F>>,
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
struct LogUpContext<'a, F: HardwareField> {
    gamma: Flat<F>,
    beta: Flat<F>,
    lookup_bus_points: &'a BTreeMap<String, Vec<Flat<F>>>,
}

/// The main Hekate Verifier for AIR circuits.
pub struct HekateVerifier<F, H> {
    _marker: PhantomData<(F, H)>,
}

impl<F, H: Hasher> HekateVerifier<F, H>
where
    F: HardwareField
        + PackableField
        + TraceCompatibleField
        + BinaryFieldExtras
        + Into<Block128>
        + From<u128>,
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
        program_id: &[u8; 32],
        program: &P,
        instance: &ProgramInstance<F>,
        proof: &InnerProof<F>,
        transcript: &mut Transcript<H>,
        config: &Config,
    ) -> errors::Result<bool> {
        let prepared = PreparedProgram::new(program, config)?;

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
        program_id: &[u8; 32],
        prepared: &PreparedProgram<F>,
        label: &'static [u8],
        proofs: &[(&ProgramInstance<F>, &InnerProof<F>)],
    ) -> Vec<errors::Result<bool>> {
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
        program_id: &[u8; 32],
        prepared: &PreparedProgram<F>,
        instance: &ProgramInstance<F>,
        proof: &InnerProof<F>,
        transcript: &mut Transcript<H>,
        scratch: &mut VerifierScratch<F>,
    ) -> errors::Result<bool> {
        let config = &prepared.config;

        let num_rows = instance.num_rows();
        let num_vars = num_rows.trailing_zeros() as usize;

        // =========================================================
        // 1. CRITICAL SECURITY VALIDATIONS
        // =========================================================

        if config.num_queries == 0 {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "num_queries cannot be zero",
            });
        }

        if num_rows == 0 || !num_rows.is_power_of_two() {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "num_rows must be a non-zero power of two",
            });
        }

        if num_vars > MAX_TABLE_VARS {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "num_rows exceeds the table height bound",
            });
        }

        if instance.public_inputs().len() != prepared.num_public_inputs {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "instance public input count does not match the program",
            });
        }

        let main_statics = prepared.main.statics();
        let main_perm = main_statics.specs;
        let main_fixed = main_statics.fixed;
        let main_plan = &prepared.main_plan;

        let field_bits = F::BITS;
        let chiplet_defs = &prepared.chiplets;

        if proof.chiplet_rows.len() != chiplet_defs.len() {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "chiplet height count mismatch",
            });
        }

        if proof.chiplet_point_evaluations.len() != chiplet_defs.len() {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "chiplet point evaluation count mismatch",
            });
        }

        validate_fixed_columns(main_fixed, &prepared.virtual_column_layout, Some(num_vars))?;

        let mut chiplet_tables = Vec::with_capacity(chiplet_defs.len());

        let prepared_chiplets = chiplet_defs
            .iter()
            .zip(&prepared.chiplet_plans)
            .zip(&prepared.chiplet_shapes);

        for ((def, plan), shape) in prepared_chiplets {
            let c_num_rows = proof.chiplet_rows[chiplet_tables.len()];

            if c_num_rows == 0 || !c_num_rows.is_power_of_two() {
                return Err(errors::Error::Protocol {
                    protocol: "verifier",
                    message: "chiplet_rows[c] must be a non-zero power of two",
                });
            }

            let c_num_vars = c_num_rows.trailing_zeros() as usize;

            if c_num_vars > MAX_TABLE_VARS {
                return Err(errors::Error::Protocol {
                    protocol: "verifier",
                    message: "chiplet_rows[c] exceeds the table height bound",
                });
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

        let heights: Vec<(&RingSwitchPlan, usize)> = chiplet_tables
            .iter()
            .map(|t| (t.plan, t.shape.num_vars))
            .chain([(main_plan, num_vars)])
            .collect();

        let split_vars = PoolLayout::split_vars_for(&heights, field_bits, config);

        let point_evaluations = proof
            .chiplet_point_evaluations
            .iter()
            .chain([&proof.main_point_evaluation]);

        // Before at_split: its blind columns stay
        // bounded by the claims the proof carries.
        for (&(plan, table_vars), (_, claims)) in heights.iter().zip(point_evaluations) {
            if claims.len() != 2 * plan.total_claims_at(table_vars, split_vars) {
                return Err(errors::Error::Protocol {
                    protocol: "verifier",
                    message: "claimed evaluation count does not match the pool layout",
                });
            }
        }

        let pool = PoolLayout::at_split(&heights, split_vars);
        let pool_shape = pool.fold_shape();

        let metrics = config.security_metrics(field_bits, pool_shape);

        config.check_security(field_bits, pool_shape)?;

        for (table, layout) in chiplet_tables.iter_mut().zip(&pool.tables) {
            table.shape = table.shape.at(layout);
        }

        let main_layout = &pool.tables[chiplet_tables.len()];
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

        let clocks = rank_clocks(&rank_tables)?;

        let main_view = TableView {
            instance,
            plan: main_plan,
            statics: main_statics,
            shape: &main_shape,
            clocks: &clocks[0],
        };

        // =========================================================
        // PHASE 1: TRACE COMMITMENT & FIAT-SHAMIR BINDING
        // =========================================================
        let actual = prepared.program_id;

        if actual != *program_id {
            return Err(errors::Error::ProgramIdMismatch { actual });
        }

        Self::verify_trace_commitment(actual, &main_view, transcript, config)?;

        if config.zero_knowledge != proof.pad_root.is_some()
            || config.zero_knowledge != proof.outer.is_some()
        {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "pad_root and outer presence must match zero_knowledge",
            });
        }

        let mut pad_cursor = config.zero_knowledge.then(PadCursor::default);

        // =========================================================
        // PHASE 2: CHIPLET HEADERS, THEN THE POOL ROOT
        // =========================================================
        Self::absorb_chiplet_headers(&chiplet_tables, transcript);

        transcript.append_message(b"trace_root", &proof.trace_root);

        // Derived after every root,
        // bound before any challenge.
        if let Some(root) = proof.pad_root.as_ref() {
            transcript.append_message(b"pad_root", root);
        }

        // =========================================================
        // PHASE 3: DRAW GLOBAL γ, β, AND r_bus PER LOOKUP BUS
        // =========================================================
        let bus_rows = main_perm.len() as u64 * num_rows as u64
            + chiplet_tables
                .iter()
                .zip(&proof.chiplet_rows)
                .map(|(t, &rows)| t.def.permutation_checks.len() as u64 * rows as u64)
                .sum::<u64>();

        let logup_bits = config.logup_gamma_bits(field_bits, bus_rows);

        info!(
            bits = metrics.security_bits.min(logup_bits),
            ldt_query = metrics.ldt_bits,
            fold_gap = metrics.proximity_bits,
            logup_gamma = logup_bits,
            field_bits,
            code_distance = metrics.relative_distance,
            "pool security"
        );

        config.check_logup_security(field_bits, bus_rows)?;

        let gamma = transcript.challenge_field::<F>(b"bus_gamma")?.to_hardware();
        let beta = transcript.challenge_field::<F>(b"bus_beta")?.to_hardware();

        let lookup_bus_points =
            Self::draw_lookup_bus_points(main_perm, num_rows, chiplet_defs, proof, transcript)?;

        let logup = LogUpContext {
            gamma,
            beta,
            lookup_bus_points: &lookup_bus_points,
        };

        let has_bus = !main_perm.is_empty()
            || chiplet_tables
                .iter()
                .any(|t| !t.def.permutation_checks.is_empty());

        if proof.h_root.is_some() != has_bus {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "h_root presence must match bus presence",
            });
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

        match Self::verify_table(
            main_view,
            main_proof,
            main_layout.plan.total_claims(),
            transcript,
            config,
            &logup,
            pad_cursor.as_mut(),
        )? {
            Some(table) => checked.push(table),
            None => return Ok(false),
        }

        // =========================================================
        // PHASE 6: POOL EVAL AT EVERY TABLE'S r_final
        // =========================================================
        let table_claims: Vec<TableClaims<F>> = checked
            .iter()
            .map(|t| TableClaims {
                point: &t.zerocheck.r_final,
                claims: &t.claims,
            })
            .collect();

        let ctx = EvalVerifyContext {
            pool: &pool,
            tables: &table_claims,
            shifted_claims: true,
            masked: pad_cursor.is_some(),
        };

        let outcome = match EvaluatorVerifier::<F, H>::verify(
            &proof.trace_root,
            proof.h_root.as_ref(),
            &proof.eval_proof,
            transcript,
            ctx,
            config,
        )? {
            Some(outcome) => outcome,
            None => {
                warn!("Pool evaluation verification failed");
                return Ok(false);
            }
        };

        // =========================================================
        // PHASE 7: CROSS-BUS MATCHING (Σ claimed_sum = 0 per bus_id)
        // =========================================================
        if let Some(mut cursor) = pad_cursor {
            let shapes: Vec<TableShape> = checked.iter().map(|t| *t.view.shape).collect();
            let statement = OuterStatement::new(&shapes, pool.num_vars());
            let first_wire = shapes.iter().map(TableShape::mul_wires).sum::<usize>() as u32;

            let eval = Self::pool_eval_record(&pool, &checked, &outcome, first_wire, &mut cursor)?;
            let records = Self::table_records(&checked, &logup, config)?;

            return outer::verify_outer(
                proof, transcript, config, statement, &records, &eval, cursor, scratch,
            );
        }

        let mut endpoints: Vec<(String, F)> = Vec::new();
        for (bus_id, claim) in &proof.main_logup_aux.claimed_sums {
            endpoints.push((bus_id.clone(), *claim));
        }

        for aux in &proof.chiplet_logup_aux {
            for (bus_id, claim) in &aux.claimed_sums {
                endpoints.push((bus_id.clone(), *claim));
            }
        }

        logup::check_bus_sum_matching(&endpoints)?;

        Ok(true)
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
                table.plan.opened_row_bytes(),
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
        pool: &PoolLayout,
        proof: &'a InnerProof<F>,
        transcript: &mut Transcript<H>,
        config: &Config,
        logup: &LogUpContext<'_, F>,
        mut cursor: Option<&mut PadCursor>,
    ) -> errors::Result<Vec<CheckedTable<'a, 's, F>>> {
        if proof.chiplet_zerocheck_proofs.len() != chiplet_tables.len() {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "chiplet zerocheck proof count mismatch",
            });
        }

        if proof.chiplet_logup_aux.len() != chiplet_tables.len() {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "chiplet logup_aux count mismatch",
            });
        }

        let mut checked = Vec::with_capacity(chiplet_tables.len() + 1);
        for (c_idx, table) in chiplet_tables.iter().enumerate() {
            let def = table.def;

            let view = TableView {
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

            let half = pool.tables[c_idx].plan.total_claims();

            match Self::verify_table(
                view,
                table_proof,
                half,
                transcript,
                config,
                logup,
                cursor.as_deref_mut(),
            )? {
                Some(table) => checked.push(table),
                None => {
                    warn!(chiplet_idx = c_idx, "Chiplet ZeroCheck failed");
                    return Err(errors::Error::Protocol {
                        protocol: "verifier",
                        message: "chiplet ZeroCheck failed",
                    });
                }
            }

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
        half: usize,
        transcript: &mut Transcript<H>,
        config: &Config,
        logup: &LogUpContext<'_, F>,
        mut cursor: Option<&mut PadCursor>,
    ) -> errors::Result<Option<CheckedTable<'a, 's, F>>> {
        let claims = Claims::new(&proof.point_evaluation.1);

        if claims.flat().len() != 2 * half {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "combined trace values length mismatch with physical trace",
            });
        }

        let (values, values_next) = claims.flat().split_at(half);

        let zerocheck = match Self::verify_zerocheck(
            &view,
            &proof,
            transcript,
            config,
            values,
            values_next,
            logup,
            cursor.as_deref_mut(),
        )? {
            Some(outcome) => outcome,
            None => return Ok(None),
        };

        if canonical_slice_to_flat(&proof.point_evaluation.0) != zerocheck.r_final {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "point evaluation mismatch with r_final",
            });
        }

        let claims_first = cursor.map(|c| c.take(claims.flat().len()));

        transcript.append_field_each(b"claimed_val", claims.canonical());

        Ok(Some(CheckedTable {
            view,
            logup_aux: proof.logup_aux,
            claims,
            zerocheck,
            claims_first,
        }))
    }

    /// The pool eval's outer record; its rounds take
    /// the pads after every table's claim pads.
    fn pool_eval_record(
        pool: &PoolLayout,
        checked: &[CheckedTable<'_, '_, F>],
        outcome: &EvalOutcome<F>,
        first_wire: u32,
        cursor: &mut PadCursor,
    ) -> errors::Result<EvalRecord<F>> {
        let round_first = cursor.take(2 * pool.num_vars());

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
            shifted_claims: true,
            challenges: &outcome.challenges,
            claim_masked: outcome.claim_masked,
            fin: outcome.fin,
            claim_pads: &claim_pads,
            round_first,
            first_wire,
        })
    }

    /// Each checked table's outer record.
    fn table_records<'s>(
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
            .map(|table| {
                let masked = table.zerocheck.masked.as_ref().ok_or_else(unpadded)?;
                let claims_first = table.claims_first.ok_or_else(unpadded)?;
                let view = &table.view;

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

    /// Mirrors `commit_main_trace` on the verifier side:
    /// absorbs every public parameter before the first challenge.
    #[instrument(skip_all, level = "trace", name = "verify_trace_commitment")]
    fn verify_trace_commitment(
        program_id: [u8; 32],
        main: &TableView<'_, '_, F>,
        transcript: &mut Transcript<H>,
        config: &Config,
    ) -> errors::Result<()> {
        let num_rows = main.instance.num_rows();
        let num_cols = main.shape.num_columns;

        transcript.append_message(b"program_id", &program_id);
        transcript.append_u64(b"num_columns", num_cols as u64);
        transcript.append_u64(b"num_rows", num_rows as u64);
        transcript.append_u64(b"ldt_support_size", config.ldt_support_size as u64);
        transcript.append_u64(b"zero_knowledge", u64::from(config.zero_knowledge));
        transcript.append_u64(b"num_queries", config.num_queries as u64);
        transcript.append_u64(b"outer_queries", config.outer_queries as u64);
        transcript.append_u64(b"main_row_bytes", main.plan.opened_row_bytes() as u64);
        transcript.append_field_each(b"public_input", main.instance.public_inputs());

        for bc in main.statics.boundary {
            bc.absorb_into(transcript);
        }

        Ok(())
    }

    /// Verifies AIR + LogUp ZeroCheck.
    ///
    /// The initial sumcheck claim is `Σ_k α^k · claimed_sum_k` (LogUp bus-sum total),
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
        trace_values_next: &[Flat<F>],
        logup: &LogUpContext<'_, F>,
        mut cursor: Option<&mut PadCursor>,
    ) -> errors::Result<Option<ZerocheckOutcome<F>>> {
        let TableView {
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
        let num_buses = bus_specs.len();

        let claimed_sums_first = cursor.as_deref_mut().map(|c| c.take(num_buses));

        // =========================================================
        // Constraint System Setup
        // =========================================================

        // Validate LogUp aux structure and bind claimed_sums
        // into the transcript before drawing alpha / r_zerocheck,
        // the ZeroCheck challenges depend on the bus-sum target.
        if logup_aux.claimed_sums.len() != bus_specs.len() {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "logup_aux claimed_sums length mismatch with bus_specs",
            });
        }

        if logup_aux.h_evals.len() != bus_specs.len() {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "logup_aux h_evals length mismatch with bus_specs",
            });
        }

        for (i, ((h_bus, _), (claim_bus, _))) in logup_aux
            .h_evals
            .iter()
            .zip(logup_aux.claimed_sums.iter())
            .enumerate()
        {
            if h_bus != claim_bus {
                return Err(errors::Error::Protocol {
                    protocol: "verifier",
                    message: "logup_aux bus_id order diverges between h_evals and claimed_sums",
                });
            }

            if h_bus.as_str() != bus_specs[i].0.as_str() {
                return Err(errors::Error::Protocol {
                    protocol: "verifier",
                    message: "logup_aux bus_id does not match bus_specs ordering",
                });
            }
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
        if !bus_specs.is_empty() {
            let mut alpha_pow = alpha_logup_start;
            for (_, claim) in &logup_aux.claimed_sums {
                initial_claim += alpha_pow * claim.to_hardware();
                alpha_pow *= alpha;
            }
        }

        let zc_first = cursor
            .as_deref_mut()
            .map(|c| c.take(num_vars * sumcheck_degree));

        let sc_res = verify(
            num_vars,
            sumcheck_degree,
            initial_claim,
            sc_proof,
            transcript,
        )?;

        let h_first = cursor.map(|c| c.take(num_buses));

        protocol::absorb_logup_h_evals(transcript, &logup_aux.h_evals);

        let (r_final, val_final) = match sc_res {
            Some(res) => res,
            None => {
                warn!("Main constraint sumcheck failed");
                return Ok(None);
            }
        };

        // =========================================================
        // GLOBAL CONSTRAINT CONSISTENCY CHECK
        // =========================================================

        debug!("Verifying AIR constraint consistency at r_final");

        let _final = trace_span!("zerocheck_final").entered();

        if let (Some(claimed_sums_first), Some(zerocheck_first), Some(h_evals_first)) =
            (claimed_sums_first, zc_first, h_first)
        {
            return Ok(Some(ZerocheckOutcome {
                r_final,
                masked: Some(ZerocheckMasked {
                    claimed_sums_first,
                    zerocheck_first,
                    h_evals_first,
                    alpha,
                    r_zerocheck,
                    val_final,
                }),
            }));
        }

        // Every reported h_eval is the base h claim
        // the eval argument binds to the committed h.
        let h_claims = &trace_values[trace_values.len() - num_buses..];

        for ((_, h_eval), &claim) in logup_aux.h_evals.iter().zip(h_claims) {
            if h_eval.to_hardware() != claim {
                warn!("Reported h_eval does not match the committed h claim");
                return Ok(None);
            }
        }

        let eq_zc_eval = TensorProduct::evaluate_eq_slice(&r_zerocheck, &r_final);

        // Separate physical trace from blinding
        // to match AIR constraints index layout.
        let current_row = &trace_values[0..trace_width];
        let next_row = &trace_values_next[0..trace_width];

        // A. Enforce fixed columns:
        // each committed column's eval must equal its shape
        // MLE at r_final, substituted into the row ast.evaluate
        // sees so constraints use the verified value.
        let mut current_row_subst: Vec<Flat<F>> = current_row.to_vec();
        let mut shapes = ShapeEvaluator::new(&r_final);

        for fc in statics.fixed {
            let FixedColumn { col_idx, shape } = fc;

            if *col_idx >= trace_width {
                return Err(errors::Error::Protocol {
                    protocol: "verifier",
                    message: "fixed column col_idx out of trace_width",
                });
            }

            let expected = shapes.evaluate(shape);

            if current_row[*col_idx] != expected {
                warn!(
                    "Fixed-column forgery detected.\nCol: {}\nClaimed: {:?}\nExpected: {:?}",
                    col_idx, current_row[*col_idx], expected
                );
                return Ok(None);
            }

            current_row_subst[*col_idx] = expected;
        }

        let mut expected_val = Flat::from_raw(F::ZERO);
        let mut alpha_pow = Flat::from_raw(F::ONE);
        let mut main_constraints_sum = Flat::from_raw(F::ZERO);

        // B. Main Constraints
        let constraint_evals = ast.evaluate(&current_row_subst, next_row);

        for eval in constraint_evals {
            main_constraints_sum += eval * alpha_pow;
            alpha_pow *= alpha;
        }

        // Factor out eq_zc_eval
        expected_val += main_constraints_sum * eq_zc_eval;

        // C. Boundary Constraints
        let eq_row_checker = TensorProduct::new(r_final.clone());

        for bc in boundary_constraints {
            if bc.col_idx >= current_row.len() {
                return Err(errors::Error::Protocol {
                    protocol: "verifier",
                    message: "boundary constraint col_idx out of bounds",
                });
            }

            if bc.row_idx >= 1 << num_vars {
                return Err(errors::Error::Protocol {
                    protocol: "verifier",
                    message: "boundary constraint row_idx exceeds trace height",
                });
            }

            let pub_val = bc.resolve_target(instance)?.to_hardware();
            let trace_val = current_row[bc.col_idx];
            let eq_eval = eq_row_checker.evaluate_at_index(bc.row_idx);

            expected_val += (trace_val - pub_val) * eq_eval * alpha_pow;
            alpha_pow *= alpha;
        }

        // D. Blinding Polynomial (Telescopic Sum)
        for k in 0..config.blind_units() {
            // Extract blinding values directly
            // from the verified proof.
            let b_k = trace_values[trace_width + k];
            let b_k_next = trace_values_next[trace_width + k];

            // b_k * alpha - b_k_next * alpha = (b_k - b_k_next) * alpha
            expected_val += (b_k - b_k_next) * alpha_pow;
            alpha_pow *= alpha;
        }

        // E. LogUp consistency + bus-sum at r_final
        if !bus_specs.is_empty() {
            let mut source_evals: Vec<Flat<F>> = Vec::new();

            for (spec_idx, (bus_id, spec)) in bus_specs.iter().enumerate() {
                let h_eval = logup_aux.h_evals[spec_idx].1.to_hardware();

                let s_eval = match spec.selector {
                    Some(idx) => current_row[idx],
                    None => Flat::from_raw(F::ONE),
                };

                let s_recv_eval = match spec.recv_selector {
                    Some(idx) => current_row[idx],
                    None => Flat::from_raw(F::ZERO),
                };

                // Source values flattened in spec order.
                // Const folds in at its own β-position
                // so the helper's stitching loop produces
                // the same key as the prover.
                source_evals.clear();

                for (source, _) in &spec.sources {
                    match source {
                        permutation::Source::Column(col_idx)
                        | permutation::Source::PhaseColumn(col_idx) => {
                            source_evals.push(current_row[*col_idx]);
                        }
                        permutation::Source::Columns(indices) => {
                            for &col_idx in indices {
                                source_evals.push(current_row[col_idx]);
                            }
                        }
                        permutation::Source::Const(val) => {
                            source_evals.push(F::from(*val).to_hardware());
                        }
                        permutation::Source::RowIndexLeBytes(n) => {
                            source_evals.push(eval_row_idx_le_mle::<F>(*n, &r_final));
                        }
                        permutation::Source::RowIndexByte(n) => {
                            source_evals.push(eval_row_idx_byte_mle::<F>(*n, &r_final));
                        }
                        permutation::Source::EmitRank(_) => {
                            let clock = RankClock::for_spec(clocks, spec_idx)?;
                            source_evals.push(clock.evaluate(&r_final)?);
                        }
                    }
                }

                let eq_lookup = match spec.kind {
                    BusKind::Permutation => Flat::from_raw(F::ONE),
                    BusKind::Lookup => {
                        let r_bus =
                            lookup_bus_points
                                .get(bus_id)
                                .ok_or(errors::Error::Protocol {
                                    protocol: "verifier",
                                    message: "lookup bus spec missing r_bus point",
                                })?;

                        if r_bus.len() < num_vars {
                            return Err(errors::Error::Protocol {
                                protocol: "verifier",
                                message: "r_bus shorter than table num_vars",
                            });
                        }

                        let r_lo = &r_bus[0..num_vars];
                        let r_hi = &r_bus[num_vars..];
                        let eq_r_lo_at_r_final = TensorProduct::evaluate_eq_slice(r_lo, &r_final);

                        let one = Flat::from_raw(F::ONE);

                        let mut eq_r_hi_at_0 = one;
                        for r_j in r_hi {
                            eq_r_hi_at_0 *= one - *r_j;
                        }

                        eq_r_hi_at_0 * eq_r_lo_at_r_final
                    }
                };

                let bus_eval = logup::BusSpecEvaluation {
                    h_eval,
                    s_eval,
                    s_recv_eval,
                    source_evals: &source_evals,
                    alpha_bus: alpha_pow,
                    eq_lookup,
                };

                expected_val +=
                    logup::expected_bus_contribution(&bus_eval, gamma, beta, eq_zc_eval);
                alpha_pow *= alpha;
            }
        }

        // Strict Check
        let masking_bias = val_final - expected_val;
        if masking_bias != Flat::from_raw(F::ZERO) {
            warn!(
                "Constraint logic mismatch.\nClaimed (Sumcheck): {:?}\nCalculated (Constraints): {:?}\nDiff: {:?}",
                val_final, expected_val, masking_bias
            );
            return Ok(None);
        }

        Ok(Some(ZerocheckOutcome {
            r_final,
            masked: None,
        }))
    }

    /// Aggregates lookup-bus heights from main + chiplet rows
    /// and draws one `r_bus` per bus_id in sorted order.
    fn draw_lookup_bus_points(
        main_specs: &[(String, PermutationCheckSpec)],
        main_rows: usize,
        chiplet_defs: &[ChipletDef<F>],
        proof: &InnerProof<F>,
        transcript: &mut Transcript<H>,
    ) -> errors::Result<BTreeMap<String, Vec<Flat<F>>>> {
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
        let tower: BTreeMap<String, Vec<F>> =
            protocol::draw_lookup_bus_points(transcript, &entries)?;

        Ok(tower
            .into_iter()
            .map(|(id, v)| (id, v.into_iter().map(|x| x.to_hardware()).collect()))
            .collect())
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
