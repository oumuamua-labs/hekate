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

pub use sumcheck::verify;

use crate::evaluator::{EvalVerifyContext, EvaluatorVerifier};
use crate::outer::PadCursor;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;
use hekate_core::config::{Config, FoldShape};
use hekate_core::errors;
use hekate_core::proofs::{
    BrakedownCommitment, EvalBatchProof, InnerProof, LogUpAux, SumcheckProof,
};
use hekate_core::protocol;
use hekate_core::tensor::TensorProduct;
use hekate_core::trace::TraceCompatibleField;
use hekate_crypto::Hasher;
use hekate_crypto::transcript::Transcript;
use hekate_math::{BinaryFieldExtras, Block128, Flat, HardwareField, PackableField, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::ConstraintAst;
use hekate_program::expander::RingSwitchPlan;
use hekate_program::outer::{
    EvalInputs, OuterStatement, TableInputs, TablePads, TableRecord, TableShape, eval_record,
    table_record,
};
use hekate_program::permutation::{
    self, BusKind, eval_row_idx_byte_mle, eval_row_idx_le_mle, validate_fixed_selectors,
};
use hekate_program::{Air, FixedColumn, Program, ProgramInstance, digest, validate_fixed_columns};
use tracing::{debug, info, instrument, warn};

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
    ast: ConstraintAst<F>,
    shape: TableShape,
    plan: RingSwitchPlan,
}

/// One table's static shape shared by its
/// ZeroCheck, eval opening and outer record.
#[derive(Clone, Copy)]
struct TableView<'a, F: TowerField, A> {
    air: &'a A,
    instance: &'a ProgramInstance<F>,
    plan: &'a RingSwitchPlan,
    ast: &'a ConstraintAst<F>,
    shape: &'a TableShape,
}

/// One table's proof parts.
#[derive(Clone, Copy)]
struct TableProof<'a, F: TowerField> {
    commitment: &'a BrakedownCommitment,
    sc_proof: &'a SumcheckProof<F>,
    eval_proof: &'a EvalBatchProof<F>,
    logup_aux: &'a LogUpAux<F>,
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
    /// 1. Bind public inputs, config, and trace root into the transcript.
    /// 2. Absorb each chiplet header (name, rows, cols, row_bytes, root).
    /// 3. Draw global LogUp challenges γ, β.
    /// 4. Per chiplet:
    ///    verify ZeroCheck (with LogUp) and the trace eval at
    ///    `r_final`, which opens the committed `h` alongside.
    /// 5. Verify the main AIR ZeroCheck (with LogUp).
    /// 6. Verify the main eval at `r_final`.
    /// 7. Check that LogUp `claimed_sum` totals cancel per `bus_id`.
    #[instrument(skip_all, name = "Hekate::verify")]
    pub fn verify<P: Program<F> + Sync>(
        program_id: &[u8; 32],
        program: &P,
        instance: &ProgramInstance<F>,
        proof: &InnerProof<F>,
        transcript: &mut Transcript<H>,
        config: &Config,
    ) -> errors::Result<bool> {
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

        if proof.trace_commitment.num_rows != num_rows {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "trace_commitment.num_rows does not match instance.num_rows",
            });
        }

        if instance.public_inputs().len() != program.num_public_inputs() {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "instance public input count does not match the program",
            });
        }

        let main_entries = program.virtual_expander().map(|e| e.expansion_entries());
        let main_plan = RingSwitchPlan::new(
            program.column_layout(),
            main_entries.as_deref(),
            config.blind_units(),
            program.permutation_checks().len(),
        )?;

        let field_bits = F::BITS;
        let main_split = main_plan.split_vars(num_vars, field_bits, config);

        let main_shape = FoldShape {
            grid_cols: 1 << main_split,
            grid_rows: 1 << (num_vars - main_split),
            units: main_plan.num_units,
        };

        let metrics = config.security_metrics(field_bits, main_shape);

        config.check_security(field_bits, main_shape)?;

        let expected_trace_len = main_plan.total_claims();
        let combined_vals = &proof.eval_proof.point_evaluation.1;

        if combined_vals.len() != expected_trace_len * 2 {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "combined trace values length mismatch with physical trace",
            });
        }

        let combined_hw = canonical_slice_to_flat(combined_vals);
        let trace_values = &combined_hw[0..expected_trace_len];
        let trace_values_next = &combined_hw[expected_trace_len..];

        let main_perm = program.permutation_checks();
        let chiplet_defs = program.chiplet_defs()?;

        if proof.chiplet_commitments.len() != chiplet_defs.len() {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "chiplet commitment count mismatch",
            });
        }

        let mut chiplet_asts = Vec::with_capacity(chiplet_defs.len());

        for def in &chiplet_defs {
            validate_fixed_selectors(&def.permutation_checks, def.pins())?;

            chiplet_asts.push(def.constraint_ast());
        }

        let main_ast = program.constraint_ast();
        let main_fixed = program.fixed_columns();

        validate_fixed_selectors(&main_perm, &main_fixed)?;
        validate_fixed_columns(&main_fixed, program.virtual_column_layout(), Some(num_vars))?;

        let main_shape = TableShape::from_air(program, num_vars, &main_ast)?;
        let mut chiplet_tables = Vec::with_capacity(chiplet_defs.len());

        for (def, ast) in chiplet_defs.iter().zip(chiplet_asts) {
            let c_num_rows = proof.chiplet_commitments[chiplet_tables.len()].num_rows;

            if c_num_rows == 0 || !c_num_rows.is_power_of_two() {
                return Err(errors::Error::Protocol {
                    protocol: "verifier",
                    message: "chiplet_commitments[c].num_rows must be a non-zero power of two",
                });
            }

            let c_num_vars = c_num_rows.trailing_zeros() as usize;

            validate_fixed_columns(
                &Air::<F>::fixed_columns(def),
                Air::<F>::virtual_column_layout(def),
                Some(c_num_vars),
            )?;

            let c_entries = Air::<F>::virtual_expander(def).map(|e| e.expansion_entries());
            let plan = RingSwitchPlan::new(
                Air::<F>::column_layout(def),
                c_entries.as_deref(),
                config.blind_units(),
                def.permutation_checks.len(),
            )?;

            chiplet_tables.push(ChipletTable {
                def,
                shape: TableShape::from_air(def, c_num_vars, &ast)?,
                ast,
                plan,
            });
        }

        // =========================================================
        // PHASE 1: TRACE COMMITMENT & FIAT-SHAMIR BINDING
        // =========================================================
        let actual = digest::program_id_of(program, &chiplet_defs, &program.inline_chiplets()?);

        if actual != *program_id {
            return Err(errors::Error::ProgramIdMismatch { actual });
        }

        Self::verify_trace_commitment(
            actual, program, instance, proof, transcript, config, &main_plan,
        )?;

        if config.zero_knowledge != proof.pad_root.is_some()
            || config.zero_knowledge != proof.outer.is_some()
        {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "pad_root and outer presence must match zero_knowledge",
            });
        }

        let mut pad_cursor = config.zero_knowledge.then(PadCursor::default);
        let mut records: Vec<TableRecord<F>> = Vec::new();

        // =========================================================
        // PHASE 2: COMMIT EACH CHIPLET (absorb roots)
        // =========================================================
        Self::verify_chiplet_commitments_only(&chiplet_tables, proof, transcript);

        // Derived after every root,
        // bound before any challenge.
        if let Some(root) = proof.pad_root.as_ref() {
            transcript.append_message(b"pad_root", root);
        }

        // =========================================================
        // PHASE 3: DRAW GLOBAL γ, β, AND r_bus PER LOOKUP BUS
        // =========================================================
        let bus_rows = main_perm.len() as u64 * proof.trace_commitment.num_rows as u64
            + chiplet_tables
                .iter()
                .zip(proof.chiplet_commitments.iter())
                .map(|(t, c)| t.def.permutation_checks.len() as u64 * c.num_rows as u64)
                .sum::<u64>();

        let logup_bits = config.logup_gamma_bits(field_bits, bus_rows);

        info!(
            bits = metrics.security_bits.min(logup_bits),
            ldt_query = metrics.ldt_bits,
            fold_gap = metrics.proximity_bits,
            logup_gamma = logup_bits,
            field_bits,
            code_distance = metrics.relative_distance,
            "main table security"
        );

        config.check_logup_security(field_bits, bus_rows)?;

        let gamma = transcript.challenge_field::<F>(b"bus_gamma")?.to_hardware();
        let beta = transcript.challenge_field::<F>(b"bus_beta")?.to_hardware();

        let lookup_bus_points =
            Self::draw_lookup_bus_points(program, &chiplet_defs, proof, transcript)?;

        let logup = LogUpContext {
            gamma,
            beta,
            lookup_bus_points: &lookup_bus_points,
        };

        // =========================================================
        // PHASE 4: PER-CHIPLET FUSED (ZC + LogUp + eval)
        // =========================================================
        Self::verify_chiplet_fused(
            &chiplet_tables,
            proof,
            transcript,
            config,
            &logup,
            pad_cursor.as_mut(),
            &mut records,
        )?;

        // =========================================================
        // PHASE 5: MAIN AIR ZEROCHECK + LogUp
        // =========================================================
        let main_view = TableView {
            air: program,
            instance,
            plan: &main_plan,
            ast: &main_ast,
            shape: &main_shape,
        };

        let main_proof = TableProof {
            commitment: &proof.trace_commitment,
            sc_proof: &proof.zerocheck_proof,
            eval_proof: &proof.eval_proof,
            logup_aux: &proof.main_logup_aux,
        };

        let zerocheck = match Self::verify_zerocheck(
            &main_view,
            &main_proof,
            transcript,
            config,
            trace_values,
            trace_values_next,
            &logup,
            pad_cursor.as_mut(),
        )? {
            Some(outcome) => outcome,
            None => return Ok(false),
        };

        // =========================================================
        // PHASE 6: MAIN EVAL AT r_final (single point)
        // =========================================================
        if !Self::verify_table_eval(
            &main_view,
            &main_proof,
            transcript,
            config,
            &combined_hw,
            zerocheck,
            &logup,
            pad_cursor.as_mut(),
            &mut records,
        )? {
            warn!("Main trace evaluation verification failed");
            return Ok(false);
        }

        // =========================================================
        // PHASE 7: CROSS-BUS MATCHING (Σ claimed_sum = 0 per bus_id)
        // =========================================================
        if let Some(cursor) = pad_cursor {
            let mut shapes = Vec::with_capacity(1 + chiplet_tables.len());
            shapes.push(main_shape);
            shapes.extend(chiplet_tables.iter().map(|t| t.shape));

            let statement = OuterStatement::new(&shapes, config.blind_units());

            return outer::verify_outer(proof, transcript, config, statement, records, cursor);
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
    /// Absorb each chiplet's structure +
    /// root into the transcript (no ZeroCheck).
    /// Mirrors prover's commit_chiplets_only.
    #[instrument(skip_all, name = "verify_chiplet_commitments_only")]
    fn verify_chiplet_commitments_only(
        chiplet_tables: &[ChipletTable<'_, F>],
        proof: &InnerProof<F>,
        transcript: &mut Transcript<H>,
    ) {
        for (table, c_comm) in chiplet_tables.iter().zip(&proof.chiplet_commitments) {
            let def = table.def;

            protocol::absorb_chiplet_header(
                transcript,
                &def.name(),
                c_comm.num_rows,
                def.num_columns(),
                table.plan.opened_row_bytes(),
                &c_comm.root,
            );

            for bc in &Air::<F>::boundary_constraints(def) {
                bc.absorb_into(transcript);
            }
        }
    }

    /// PHASE 4:
    /// Per-chiplet fused verification.
    /// Mirrors prover's fused_chiplet_loop.
    #[instrument(skip_all, name = "verify_chiplet_fused")]
    #[allow(clippy::too_many_arguments)]
    fn verify_chiplet_fused(
        chiplet_tables: &[ChipletTable<'_, F>],
        proof: &InnerProof<F>,
        transcript: &mut Transcript<H>,
        config: &Config,
        logup: &LogUpContext<'_, F>,
        mut cursor: Option<&mut PadCursor>,
        records: &mut Vec<TableRecord<F>>,
    ) -> errors::Result<()> {
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

        if proof.chiplet_eval_proofs.len() != chiplet_tables.len() {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "chiplet eval_proofs count mismatch",
            });
        }

        for (c_idx, table) in chiplet_tables.iter().enumerate() {
            let def = table.def;

            let c_comm = &proof.chiplet_commitments[c_idx];
            let c_sc_proof = &proof.chiplet_zerocheck_proofs[c_idx];
            let c_eval_proof = &proof.chiplet_eval_proofs[c_idx];
            let c_logup_aux = &proof.chiplet_logup_aux[c_idx];

            let c_num_rows = c_comm.num_rows;
            let c_trace_width = table.plan.total_claims();

            let c_combined = &c_eval_proof.point_evaluation.1;
            if c_combined.len() != c_trace_width * 2 {
                return Err(errors::Error::Protocol {
                    protocol: "verifier",
                    message: "chiplet trace values length mismatch",
                });
            }

            let c_combined_hw = canonical_slice_to_flat(c_combined);
            let c_trace_values = &c_combined_hw[0..c_trace_width];
            let c_trace_values_next = &c_combined_hw[c_trace_width..];

            let c_instance = ProgramInstance::new(c_num_rows, vec![]);

            let view = TableView {
                air: def,
                instance: &c_instance,
                plan: &table.plan,
                ast: &table.ast,
                shape: &table.shape,
            };

            let table_proof = TableProof {
                commitment: c_comm,
                sc_proof: c_sc_proof,
                eval_proof: c_eval_proof,
                logup_aux: c_logup_aux,
            };

            let zerocheck = match Self::verify_zerocheck(
                &view,
                &table_proof,
                transcript,
                config,
                c_trace_values,
                c_trace_values_next,
                logup,
                cursor.as_deref_mut(),
            )? {
                Some(outcome) => outcome,
                None => {
                    warn!(chiplet_idx = c_idx, "Chiplet ZeroCheck failed");
                    return Err(errors::Error::Protocol {
                        protocol: "verifier",
                        message: "chiplet ZeroCheck failed",
                    });
                }
            };

            if !Self::verify_table_eval(
                &view,
                &table_proof,
                transcript,
                config,
                &c_combined_hw,
                zerocheck,
                logup,
                cursor.as_deref_mut(),
                records,
            )? {
                warn!(
                    chiplet_idx = c_idx,
                    "Chiplet evaluation verification failed"
                );
                return Err(errors::Error::Protocol {
                    protocol: "verifier",
                    message: "chiplet evaluation verification failed",
                });
            }

            debug!(
                chiplet_idx = c_idx,
                chiplet_name = def.name(),
                "Chiplet verified"
            );
        }

        Ok(())
    }

    /// A table's eval at `r_final` after its ZeroCheck.
    /// Mirrors the prover's `prove_eval_at_r_final`; with
    /// a pad cursor it also completes the table's record.
    #[instrument(skip_all, name = "verify_table_eval")]
    #[allow(clippy::too_many_arguments)]
    fn verify_table_eval<A: Air<F>>(
        view: &TableView<'_, F, A>,
        proof: &TableProof<'_, F>,
        transcript: &mut Transcript<H>,
        config: &Config,
        claimed_values: &[Flat<F>],
        zerocheck: ZerocheckOutcome<F>,
        logup: &LogUpContext<'_, F>,
        mut cursor: Option<&mut PadCursor>,
        records: &mut Vec<TableRecord<F>>,
    ) -> errors::Result<bool> {
        let TableView {
            air,
            instance,
            plan: ring_plan,
            ast,
            shape,
        } = *view;
        let TableProof {
            commitment,
            eval_proof,
            logup_aux,
            ..
        } = *proof;

        let num_vars = instance.num_rows().trailing_zeros() as usize;
        let r_final = zerocheck.r_final;

        if canonical_slice_to_flat(&eval_proof.point_evaluation.0) != r_final {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "eval_proof point mismatch with r_final",
            });
        }

        let claims_first = cursor.as_deref_mut().map(|c| c.take(claimed_values.len()));

        let ctx = EvalVerifyContext {
            point: &r_final,
            claimed_values,
            num_vars,
            ring_plan,
            shifted_claims: true,
            masked: cursor.is_some(),
            h_commitment: logup_aux.h_commitment.as_ref(),
        };

        let outcome = match EvaluatorVerifier::<F, H>::verify(
            commitment, eval_proof, transcript, ctx, config,
        )? {
            Some(outcome) => outcome,
            None => return Ok(false),
        };

        let (Some(cursor), Some(claims_first), Some(masked)) =
            (cursor, claims_first, zerocheck.masked)
        else {
            return Ok(true);
        };

        let eval_rounds_first = cursor.take(2 * num_vars);

        let claim_pads: Vec<u32> = (0..claimed_values.len() as u32)
            .map(|c| claims_first + c)
            .collect();

        let eval = eval_record(&EvalInputs {
            plan: ring_plan,
            eta: outcome.eta,
            r_mix: &outcome.r_mix,
            shifted_claims: true,
            challenges: &outcome.challenges,
            claim_masked: outcome.claim_masked,
            fin: outcome.fin,
            claim_pads: &claim_pads,
            round_first: eval_rounds_first,
            first_wire: (shape.mul_nodes + shape.num_buses) as u32,
        })?;

        records.push(table_record(TableInputs {
            air,
            ast,
            instance,
            num_vars,
            trace_width: air.num_columns(),
            blinding_columns: config.blind_units(),
            alpha: masked.alpha,
            gamma: logup.gamma,
            beta: logup.beta,
            r_zerocheck: &masked.r_zerocheck,
            r_final: &r_final,
            lookup_bus_points: logup.lookup_bus_points,
            claimed_sums_masked: &logup_aux.claimed_sums,
            h_evals_masked: &logup_aux.h_evals,
            val_final_masked: masked.val_final,
            claims_masked: claimed_values.to_vec(),
            pads: TablePads {
                claimed_sums: masked.claimed_sums_first,
                zerocheck: masked.zerocheck_first,
                h_evals: masked.h_evals_first,
                claims: claims_first,
            },
            trace_eval: eval.record,
            gadget: eval.gadget,
        })?);

        Ok(true)
    }

    /// Mirrors `commit_main_trace` on the verifier side:
    /// absorbs every public parameter and the
    /// trace root before the first challenge.
    #[instrument(skip_all, name = "verify_trace_commitment")]
    fn verify_trace_commitment<A: Air<F>>(
        program_id: [u8; 32],
        main: &A,
        instance: &ProgramInstance<F>,
        proof: &InnerProof<F>,
        transcript: &mut Transcript<H>,
        config: &Config,
        main_plan: &RingSwitchPlan,
    ) -> errors::Result<()> {
        let num_rows = instance.num_rows();
        let num_cols = main.num_columns();

        transcript.append_message(b"program_id", &program_id);
        transcript.append_u64(b"num_columns", num_cols as u64);
        transcript.append_u64(b"num_rows", num_rows as u64);
        transcript.append_u64(b"ldt_support_size", config.ldt_support_size as u64);
        transcript.append_u64(b"zero_knowledge", u64::from(config.zero_knowledge));
        transcript.append_u64(b"num_queries", config.num_queries as u64);
        transcript.append_u64(b"outer_queries", config.outer_queries as u64);
        transcript.append_u64(b"main_row_bytes", main_plan.opened_row_bytes() as u64);

        for val in instance.public_inputs() {
            transcript.append_field(b"public_input", *val);
        }

        transcript.append_message(b"trace_root", &proof.trace_commitment.root);

        for bc in &main.boundary_constraints() {
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
    #[instrument(skip_all, name = "verify_zerocheck")]
    #[allow(clippy::too_many_arguments)]
    fn verify_zerocheck<A: Air<F>>(
        view: &TableView<'_, F, A>,
        proof: &TableProof<'_, F>,
        transcript: &mut Transcript<H>,
        config: &Config,
        trace_values: &[Flat<F>],
        trace_values_next: &[Flat<F>],
        logup: &LogUpContext<'_, F>,
        mut cursor: Option<&mut PadCursor>,
    ) -> errors::Result<Option<ZerocheckOutcome<F>>> {
        let TableView {
            air,
            instance,
            ast,
            shape,
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

        let bus_specs = air.permutation_checks();
        let bus_specs = bus_specs.as_slice();
        let trace_width = air.num_columns();

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

        // The `h` commitment is present exactly
        // when the table carries a bus.
        let has_bus = !bus_specs.is_empty();
        if logup_aux.h_commitment.is_some() != has_bus {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "logup_aux h binding presence must match bus presence",
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

        // Absorb the `h` commitment root before alpha;
        // it must bind the ZeroCheck challenges.
        if let Some(h_comm) = logup_aux.h_commitment.as_ref() {
            if h_comm.num_rows != num_rows || h_comm.num_cols != bus_specs.len() {
                return Err(errors::Error::Protocol {
                    protocol: "verifier",
                    message: "logup h_commitment dimensions do not match the table",
                });
            }

            transcript.append_message(b"logup_h_root", &h_comm.root);
        }

        let alpha_tower = transcript.challenge_field::<F>(b"alpha")?;
        let alpha = alpha_tower.to_hardware();

        let r_zerocheck = (0..num_vars)
            .map(|_| {
                transcript
                    .challenge_field::<F>(b"r_zerocheck")
                    .map(|v| v.to_hardware())
            })
            .collect::<Result<Vec<_>, _>>()?;

        let boundary_constraints = air.boundary_constraints();
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

        let fixed = air.fixed_columns();
        for fc in &fixed {
            let FixedColumn { col_idx, shape } = fc;

            if *col_idx >= trace_width {
                return Err(errors::Error::Protocol {
                    protocol: "verifier",
                    message: "fixed column col_idx out of trace_width",
                });
            }

            let expected = shape.evaluate(&r_final);

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

    /// Aggregates lookup-bus heights from main + chiplet commitments
    /// and draws one `r_bus` per bus_id in sorted order.
    fn draw_lookup_bus_points<A: Air<F>>(
        main: &A,
        chiplet_defs: &[ChipletDef<F>],
        proof: &InnerProof<F>,
        transcript: &mut Transcript<H>,
    ) -> errors::Result<BTreeMap<String, Vec<Flat<F>>>> {
        let mut heights: BTreeMap<String, u64> = BTreeMap::new();
        permutation::accumulate_lookup_heights(
            &main.permutation_checks(),
            proof.trace_commitment.num_rows as u64,
            &mut heights,
        );

        for (def, c_comm) in chiplet_defs.iter().zip(proof.chiplet_commitments.iter()) {
            permutation::accumulate_lookup_heights(
                &def.permutation_checks,
                c_comm.num_rows as u64,
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

fn canonical_slice_to_flat<F: HardwareField>(values: &[F]) -> Vec<Flat<F>> {
    values
        .iter()
        .copied()
        .map(|value| value.to_hardware())
        .collect()
}

pub(crate) fn flat_matches_canonical<F: HardwareField>(value: Flat<F>, canonical: F) -> bool {
    value.to_tower() == canonical
}
