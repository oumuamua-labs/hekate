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

use hekate::core::config::Config;
use hekate::core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::Block128;
use hekate_math::{Bit, Block32, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::permutation::{PermutationCheckSpec, REQUEST_IDX_LABEL, Source};
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness};
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

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(LK_SELECTOR));

        cs.build()
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

    fn constraint_ast(&self) -> ConstraintAst<F> {
        LookupEndpoint.constraint_ast()
    }
}

impl Program<F> for LookupForgeryProgram {
    fn chiplet_defs(&self) -> hekate_core::errors::Result<Vec<ChipletDef<F>>> {
        Ok(vec![ChipletDef::from_air(&LookupEndpoint)?])
    }
}

// =====================
// REQUEST_IDX endpoints
// =====================

const RDR_KEY: usize = 0;
const RDR_SELECTOR: usize = 1;

const TBL_KEY: usize = 0;
const TBL_REQUEST_IDX: usize = 1;
const TBL_SELECTOR: usize = 2;

#[derive(Clone)]
struct ReqIdxReader;

impl Air<F> for ReqIdxReader {
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
                    (Source::RowIndexLeBytes(4), REQUEST_IDX_LABEL),
                ],
                Some(RDR_SELECTOR),
            ),
        )]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(RDR_SELECTOR));

        cs.build()
    }
}

#[derive(Clone)]
struct ReqIdxTable;

impl Air<F> for ReqIdxTable {
    fn num_columns(&self) -> usize {
        3
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(|| vec![ColumnType::B32, ColumnType::B32, ColumnType::Bit])
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![(
            BUS_ID.into(),
            PermutationCheckSpec::new(
                vec![
                    (Source::Column(TBL_KEY), b"kappa_key" as &[u8]),
                    (Source::Column(TBL_REQUEST_IDX), REQUEST_IDX_LABEL),
                ],
                Some(TBL_SELECTOR),
            ),
        )]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(TBL_SELECTOR));

        cs.build()
    }
}

#[derive(Clone)]
struct ReqIdxForgeryProgram;

impl Air<F> for ReqIdxForgeryProgram {
    fn num_columns(&self) -> usize {
        ReqIdxReader.num_columns()
    }

    fn column_layout(&self) -> &[ColumnType] {
        ReqIdxReader.column_layout()
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        ReqIdxReader.permutation_checks()
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ReqIdxReader.constraint_ast()
    }
}

impl Program<F> for ReqIdxForgeryProgram {
    fn chiplet_defs(&self) -> hekate_core::errors::Result<Vec<ChipletDef<F>>> {
        Ok(vec![ChipletDef::from_air(&ReqIdxTable)?])
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

fn build_table_trace(rows: &[(Block32, u32, Bit)], num_rows: usize) -> ColumnTrace {
    let num_vars = num_rows.trailing_zeros() as usize;
    let layout = vec![ColumnType::B32, ColumnType::B32, ColumnType::Bit];

    let mut tb = TraceBuilder::new(&layout, num_vars).unwrap();

    for (i, (key, req_idx, sel)) in rows.iter().enumerate() {
        tb.set_b32(TBL_KEY, i, *key).unwrap();
        tb.set_b32(TBL_REQUEST_IDX, i, Block32::from(*req_idx))
            .unwrap();
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
        sumcheck_blinding_factor: 0,
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
    HekateVerifier::<F, H>::verify(&program, &instance, &proof, &mut verifier_ts, &config)
        .unwrap_or(false)
}

fn run_req_idx(reader: &[(Block32, Bit)], table: &[(Block32, u32, Bit)]) -> bool {
    let num_rows = 4;
    let seed = [0xAAu8; 32];

    let program = ReqIdxForgeryProgram;

    let reader_trace = build_reader_trace(reader, num_rows);
    let table_trace = build_table_trace(table, num_rows);

    let witness = ProgramWitness::new(reader_trace).with_chiplets(vec![table_trace]);
    let instance = ProgramInstance::new(num_rows, vec![]);

    let config = Config {
        num_queries: 4,
        min_security_bits: 0,
        sumcheck_blinding_factor: 0,
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
    HekateVerifier::<F, H>::verify(&program, &instance, &proof, &mut verifier_ts, &config)
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
    ]
}

fn honest_req_idx_table() -> Vec<(Block32, u32, Bit)> {
    vec![
        (Block32::from(0xA1A1A1A1u32), 0, Bit::ONE),
        (Block32::from(0xB2B2B2B2u32), 1, Bit::ONE),
    ]
}

fn matched_reader() -> Vec<(Block32, Bit)> {
    vec![
        (Block32::from(0xA1A1A1A1u32), Bit::ONE),
        (Block32::from(0xB2B2B2B2u32), Bit::ONE),
    ]
}

// =====
// Tests
// =====

#[test]
fn permutation_with_request_idx_rejects_forged_pair() {
    let accepted = run_req_idx(&forged_reader(), &honest_req_idx_table());
    assert!(!accepted);
}

#[test]
fn permutation_with_request_idx_accepts_honest_match() {
    let accepted = run_req_idx(&matched_reader(), &honest_req_idx_table());
    assert!(accepted);
}

#[test]
fn lookup_bus_rejects_forged_pair() {
    let accepted = run_lookup(&forged_reader(), &honest_lookup_table());
    assert!(!accepted);
}

#[test]
fn lookup_bus_accepts_honest_match() {
    let accepted = run_lookup(&matched_reader(), &honest_lookup_table());
    assert!(accepted);
}
