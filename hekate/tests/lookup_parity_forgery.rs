// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate::core::config::Config;
use hekate::core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::Block128;
use hekate_math::{Bit, Block32, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::digest::program_id;
use hekate_program::permutation::{EMIT_RANK_LABEL, PermutationCheckSpec, Side, Source};
use hekate_program::{Air, FixedColumn, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;

type F = Block128;
type H = DefaultHasher;

const BUS_ID: &str = "parity_forgery_bus";

// =====================
// Lookup-kind endpoint
// =====================

const LK_KEY: usize = 0;
const LK_SELECTOR: usize = 1;

#[derive(Clone)]
struct LookupEndpoint;

impl Air<F> for LookupEndpoint {
    fn num_columns(&self) -> usize {
        2
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(|| vec![ColumnType::B32, ColumnType::Bit])
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![(
            BUS_ID.into(),
            PermutationCheckSpec::new_lookup(
                vec![(Source::Column(LK_KEY), b"kappa_key" as &[u8])],
                Some(LK_SELECTOR),
            ),
        )]
    }

    fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
        vec![FixedColumn::prefix(LK_SELECTOR, 4)]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
    }
}

#[derive(Clone)]
struct LookupForgeryProgram;

impl Air<F> for LookupForgeryProgram {
    fn num_columns(&self) -> usize {
        2
    }

    fn column_layout(&self) -> &[ColumnType] {
        LookupEndpoint.column_layout()
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        LookupEndpoint.permutation_checks()
    }

    fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
        Air::<F>::fixed_columns(&LookupEndpoint)
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        LookupEndpoint.constraint_ast()
    }
}

impl Program<F> for LookupForgeryProgram {
    fn chiplet_defs(&self) -> hekate_core::errors::Result<Vec<ChipletDef<F>>> {
        Ok(vec![ChipletDef::from_air(&LookupEndpoint)?])
    }
}

// ==================
// EmitRank endpoints
// ==================

const RDR_KEY: usize = 0;
const RDR_SELECTOR: usize = 1;

const TBL_KEY: usize = 0;
const TBL_SELECTOR: usize = 1;

#[derive(Clone)]
struct RankReader;

impl Air<F> for RankReader {
    fn num_columns(&self) -> usize {
        2
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(|| vec![ColumnType::B32, ColumnType::Bit])
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![(
            BUS_ID.into(),
            PermutationCheckSpec::new(
                vec![
                    (Source::Column(RDR_KEY), b"kappa_key" as &[u8]),
                    (Source::EmitRank(Side::Request), EMIT_RANK_LABEL),
                ],
                Some(RDR_SELECTOR),
            ),
        )]
    }

    fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
        vec![FixedColumn::prefix(RDR_SELECTOR, 4)]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
    }
}

#[derive(Clone)]
struct RankResponder;

impl Air<F> for RankResponder {
    fn num_columns(&self) -> usize {
        2
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(|| vec![ColumnType::B32, ColumnType::Bit])
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![(
            BUS_ID.into(),
            PermutationCheckSpec::new(
                vec![
                    (Source::Column(TBL_KEY), b"kappa_key" as &[u8]),
                    (Source::EmitRank(Side::Response), EMIT_RANK_LABEL),
                ],
                Some(TBL_SELECTOR),
            ),
        )]
    }

    fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
        vec![FixedColumn::prefix(TBL_SELECTOR, 4)]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
    }
}

#[derive(Clone)]
struct RankForgeryProgram;

impl Air<F> for RankForgeryProgram {
    fn num_columns(&self) -> usize {
        RankReader.num_columns()
    }

    fn column_layout(&self) -> &[ColumnType] {
        RankReader.column_layout()
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        RankReader.permutation_checks()
    }

    fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
        Air::<F>::fixed_columns(&RankReader)
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        RankReader.constraint_ast()
    }
}

impl Program<F> for RankForgeryProgram {
    fn chiplet_defs(&self) -> hekate_core::errors::Result<Vec<ChipletDef<F>>> {
        Ok(vec![ChipletDef::from_air(&RankResponder)?])
    }
}

// ===============
// Trace builders
// ===============

fn build_lookup_trace(rows: &[(Block32, Bit)], num_rows: usize) -> ColumnTrace {
    let num_vars = num_rows.trailing_zeros() as usize;
    let layout = vec![ColumnType::B32, ColumnType::Bit];

    let mut tb = TraceBuilder::new(&layout, num_vars).unwrap();

    for (i, (key, sel)) in rows.iter().enumerate() {
        tb.set_b32(LK_KEY, i, *key).unwrap();
        tb.set_bit(LK_SELECTOR, i, *sel).unwrap();
    }

    tb.build()
}

fn build_reader_trace(rows: &[(Block32, Bit)], num_rows: usize) -> ColumnTrace {
    let num_vars = num_rows.trailing_zeros() as usize;
    let layout = vec![ColumnType::B32, ColumnType::Bit];

    let mut tb = TraceBuilder::new(&layout, num_vars).unwrap();

    for (i, (key, sel)) in rows.iter().enumerate() {
        tb.set_b32(RDR_KEY, i, *key).unwrap();
        tb.set_bit(RDR_SELECTOR, i, *sel).unwrap();
    }

    tb.build()
}

fn build_responder_trace(rows: &[(Block32, Bit)], num_rows: usize) -> ColumnTrace {
    let num_vars = num_rows.trailing_zeros() as usize;
    let layout = vec![ColumnType::B32, ColumnType::Bit];

    let mut tb = TraceBuilder::new(&layout, num_vars).unwrap();

    for (i, (key, sel)) in rows.iter().enumerate() {
        tb.set_b32(TBL_KEY, i, *key).unwrap();
        tb.set_bit(TBL_SELECTOR, i, *sel).unwrap();
    }

    tb.build()
}

// =======
// Runners
// =======

fn run_lookup(reader: &[(Block32, Bit)], table: &[(Block32, Bit)]) -> bool {
    let num_rows = 4;
    let seed = [0xAAu8; 32];

    let program = LookupForgeryProgram;

    let reader_trace = build_lookup_trace(reader, num_rows);
    let table_trace = build_lookup_trace(table, num_rows);

    let witness = ProgramWitness::new(reader_trace).with_chiplets(vec![table_trace]);
    let instance = ProgramInstance::new(num_rows, vec![]);

    let config = Config {
        num_queries: 4,
        min_security_bits: 0,
        zero_knowledge: false,
        ldt_support_size: 4,
        ..Config::default()
    };

    let proof = match prove(
        b"FORGERY", &program, &instance, &witness, &config, seed, None,
    ) {
        Ok(p) => p,
        Err(_) => return false,
    };

    let mut verifier_ts = Transcript::<H>::new(b"FORGERY");
    let pinned_id = program_id(&program).unwrap();

    HekateVerifier::<F, H>::verify(
        &pinned_id,
        &program,
        &instance,
        &proof,
        &mut verifier_ts,
        &config,
    )
    .unwrap_or(false)
}

fn run_rank(reader: &[(Block32, Bit)], table: &[(Block32, Bit)]) -> bool {
    let num_rows = 4;
    let seed = [0xAAu8; 32];

    let program = RankForgeryProgram;

    let reader_trace = build_reader_trace(reader, num_rows);
    let table_trace = build_responder_trace(table, num_rows);

    let witness = ProgramWitness::new(reader_trace).with_chiplets(vec![table_trace]);
    let instance = ProgramInstance::new(num_rows, vec![]);

    let config = Config {
        num_queries: 4,
        min_security_bits: 0,
        zero_knowledge: false,
        ldt_support_size: 4,
        ..Config::default()
    };

    let proof = match prove(
        b"FORGERY", &program, &instance, &witness, &config, seed, None,
    ) {
        Ok(p) => p,
        Err(_) => return false,
    };

    let mut verifier_ts = Transcript::<H>::new(b"FORGERY");
    let pinned_id = program_id(&program).unwrap();

    HekateVerifier::<F, H>::verify(
        &pinned_id,
        &program,
        &instance,
        &proof,
        &mut verifier_ts,
        &config,
    )
    .unwrap_or(false)
}

// =========
// Test data
// =========

fn forged_reader() -> Vec<(Block32, Bit)> {
    vec![
        (Block32::from(0xA1A1A1A1u32), Bit::ONE),
        (Block32::from(0xB2B2B2B2u32), Bit::ONE),
        (Block32::from(0xDEADBEEFu32), Bit::ONE),
        (Block32::from(0xDEADBEEFu32), Bit::ONE),
    ]
}

fn honest_lookup_table() -> Vec<(Block32, Bit)> {
    vec![
        (Block32::from(0xA1A1A1A1u32), Bit::ONE),
        (Block32::from(0xB2B2B2B2u32), Bit::ONE),
        (Block32::from(0xC3C3C3C3u32), Bit::ONE),
        (Block32::from(0xD4D4D4D4u32), Bit::ONE),
    ]
}

fn honest_rank_table() -> Vec<(Block32, Bit)> {
    vec![
        (Block32::from(0xA1A1A1A1u32), Bit::ONE),
        (Block32::from(0xB2B2B2B2u32), Bit::ONE),
        (Block32::from(0xC3C3C3C3u32), Bit::ONE),
        (Block32::from(0xD4D4D4D4u32), Bit::ONE),
    ]
}

fn matched_reader() -> Vec<(Block32, Bit)> {
    vec![
        (Block32::from(0xA1A1A1A1u32), Bit::ONE),
        (Block32::from(0xB2B2B2B2u32), Bit::ONE),
        (Block32::from(0xC3C3C3C3u32), Bit::ONE),
        (Block32::from(0xD4D4D4D4u32), Bit::ONE),
    ]
}

fn cancelling_lookup_table() -> Vec<(Block32, Bit)> {
    vec![
        (Block32::from(0xA1A1A1A1u32), Bit::ONE),
        (Block32::from(0xB2B2B2B2u32), Bit::ONE),
        (Block32::from(0xFEEDFACEu32), Bit::ONE),
        (Block32::from(0xFEEDFACEu32), Bit::ONE),
    ]
}

fn cancelling_rank_table() -> Vec<(Block32, Bit)> {
    vec![
        (Block32::from(0xA1A1A1A1u32), Bit::ONE),
        (Block32::from(0xB2B2B2B2u32), Bit::ONE),
        (Block32::from(0xFEEDFACEu32), Bit::ONE),
        (Block32::from(0xFEEDFACEu32), Bit::ONE),
    ]
}

// =====
// Tests
// =====

#[test]
fn permutation_with_emit_rank_rejects_forged_pair() {
    let accepted = run_rank(&forged_reader(), &cancelling_rank_table());
    assert!(!accepted);
}

#[test]
fn permutation_with_emit_rank_accepts_honest_match() {
    let accepted = run_rank(&matched_reader(), &honest_rank_table());
    assert!(accepted);
}

#[test]
fn lookup_bus_rejects_forged_pair() {
    let accepted = run_lookup(&forged_reader(), &cancelling_lookup_table());
    assert!(!accepted);
}

#[test]
fn lookup_bus_accepts_honest_match() {
    let accepted = run_lookup(&matched_reader(), &honest_lookup_table());
    assert!(accepted);
}
