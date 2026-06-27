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
use hekate_program::permutation::{
    ChallengeLabel, PermutationCheckSpec, REQUEST_IDX_LABEL, Source,
};
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_sdk::preflight;
use hekate_verifier::HekateVerifier;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

type F = Block128;
type H = DefaultHasher;

const BUS_ID: &str = "ghost_activation_bus";

const COL_KEY: usize = 0;
const COL_SELECTOR: usize = 1;
const COL_REQUEST_IDX: usize = 2;

#[derive(Clone)]
struct Endpoint {
    lookup: bool,
}

fn layout() -> Vec<ColumnType> {
    vec![ColumnType::B32, ColumnType::Bit, ColumnType::B32]
}

impl Endpoint {
    fn spec(&self) -> PermutationCheckSpec {
        let sources = vec![
            (Source::Column(COL_KEY), b"kappa_key" as &[u8]),
            (Source::RowIndexLeBytes(4), REQUEST_IDX_LABEL),
            (Source::Column(COL_REQUEST_IDX), REQUEST_IDX_LABEL),
        ];

        if self.lookup {
            PermutationCheckSpec::new_lookup(sources, Some(COL_SELECTOR))
        } else {
            PermutationCheckSpec::new(sources, Some(COL_SELECTOR))
        }
    }
}

impl Air<F> for Endpoint {
    fn num_columns(&self) -> usize {
        3
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(layout)
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![(BUS_ID.into(), self.spec())]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(COL_SELECTOR));

        cs.build()
    }
}

#[derive(Clone)]
struct GhostProgram {
    cpu: Endpoint,
    table: Endpoint,
}

impl Air<F> for GhostProgram {
    fn num_columns(&self) -> usize {
        self.cpu.num_columns()
    }

    fn column_layout(&self) -> &[ColumnType] {
        self.cpu.column_layout()
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        self.cpu.permutation_checks()
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        self.cpu.constraint_ast()
    }
}

impl Program<F> for GhostProgram {
    fn chiplet_defs(&self) -> hekate_core::errors::Result<Vec<ChipletDef<F>>> {
        Ok(vec![ChipletDef::from_air(&self.table)?])
    }
}

fn build_trace(rows: &[(Block32, Bit)], num_rows: usize) -> ColumnTrace {
    let num_vars = num_rows.trailing_zeros() as usize;
    let layout: Vec<ColumnType> = vec![ColumnType::B32, ColumnType::Bit, ColumnType::B32];

    let mut tb = TraceBuilder::new(&layout, num_vars).unwrap();

    for (i, (key, sel)) in rows.iter().enumerate() {
        tb.set_b32(COL_KEY, i, *key).unwrap();
        tb.set_bit(COL_SELECTOR, i, *sel).unwrap();
        tb.set_b32(COL_REQUEST_IDX, i, Block32::from(i as u32))
            .unwrap();
    }

    tb.build()
}

fn run(program: &GhostProgram, cpu: &[(Block32, Bit)], table: &[(Block32, Bit)]) -> bool {
    let num_rows = 4;
    let seed = [0xC3u8; 32];

    let cpu_trace = build_trace(cpu, num_rows);
    let table_trace = build_trace(table, num_rows);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![table_trace]);
    let instance = ProgramInstance::new(num_rows, vec![]);

    let config = Config {
        num_queries: 4,
        min_security_bits: 0,
        sumcheck_blinding_factor: 0,
        ..Config::default()
    };

    let proof = match prove(
        b"GHOST_ACTIVATION",
        program,
        &instance,
        &witness,
        &config,
        seed,
        None,
    ) {
        Ok(p) => p,
        Err(_) => return false,
    };

    let mut verifier_ts = Transcript::<H>::new(b"GHOST_ACTIVATION");
    HekateVerifier::<F, H>::verify(program, &instance, &proof, &mut verifier_ts, &config)
        .unwrap_or(false)
}

const ZERO_KEY: Block32 = Block32(0);

fn honest_cpu() -> Vec<(Block32, Bit)> {
    vec![
        (Block32::from(0xA1A1A1A1u32), Bit::ONE),
        (Block32::from(0xB2B2B2B2u32), Bit::ONE),
        (ZERO_KEY, Bit::ZERO),
        (ZERO_KEY, Bit::ZERO),
    ]
}

fn honest_table() -> Vec<(Block32, Bit)> {
    vec![
        (Block32::from(0xA1A1A1A1u32), Bit::ONE),
        (Block32::from(0xB2B2B2B2u32), Bit::ONE),
        (ZERO_KEY, Bit::ZERO),
        (ZERO_KEY, Bit::ZERO),
    ]
}

#[test]
fn ghost_activation_honest_accepts() {
    let program = GhostProgram {
        cpu: Endpoint { lookup: false },
        table: Endpoint { lookup: false },
    };

    let accepted = run(&program, &honest_cpu(), &honest_table());
    assert!(
        accepted,
        "honest aligned witness with zero-keyed padding must verify"
    );
}

#[test]
fn exploit_ghost_activation_partial_zero_key_permutation() {
    let cpu = vec![
        (Block32::from(0xA1A1A1A1u32), Bit::ONE),
        (Block32::from(0xB2B2B2B2u32), Bit::ONE),
        (ZERO_KEY, Bit::ONE),
        (ZERO_KEY, Bit::ONE),
    ];
    let table = vec![
        (Block32::from(0xA1A1A1A1u32), Bit::ONE),
        (Block32::from(0xB2B2B2B2u32), Bit::ONE),
        (ZERO_KEY, Bit::ZERO),
        (ZERO_KEY, Bit::ZERO),
    ];

    let program = GhostProgram {
        cpu: Endpoint { lookup: false },
        table: Endpoint { lookup: false },
    };

    let accepted = run(&program, &cpu, &table);
    assert!(
        !accepted,
        "Permutation-kind bus must reject double ghost activation \
         on zero-key rows (CPU forges two reads with no chiplet match)"
    );
}

#[test]
fn preflight_flags_partial_zero_key_permutation() {
    let cpu = vec![
        (Block32::from(0xA1A1A1A1u32), Bit::ONE),
        (Block32::from(0xB2B2B2B2u32), Bit::ONE),
        (ZERO_KEY, Bit::ONE),
        (ZERO_KEY, Bit::ONE),
    ];
    let table = vec![
        (Block32::from(0xA1A1A1A1u32), Bit::ONE),
        (Block32::from(0xB2B2B2B2u32), Bit::ONE),
        (ZERO_KEY, Bit::ZERO),
        (ZERO_KEY, Bit::ZERO),
    ];

    let program = GhostProgram {
        cpu: Endpoint { lookup: false },
        table: Endpoint { lookup: false },
    };

    let num_rows = 4;
    let cpu_trace = build_trace(&cpu, num_rows);
    let table_trace = build_trace(&table, num_rows);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![table_trace]);
    let instance = ProgramInstance::new(num_rows, vec![]);

    let report = preflight(&program, &instance, &witness).unwrap();

    let bus = report
        .bus_diagnostics
        .iter()
        .find(|d| d.bus_id == BUS_ID)
        .expect(
            "preflight must surface a bus_diagnostic for the ghost-activated bus; \
             a clean report on this witness would mean preflight uses a weaker \
             oracle than the runtime LogUp algebra",
        );

    let cpu_product = bus
        .endpoints
        .iter()
        .find(|e| matches!(e.source, preflight::TableId::Main))
        .map(|e| e.product)
        .expect("CPU endpoint missing");
    let chiplet_product = bus
        .endpoints
        .iter()
        .find(|e| matches!(e.source, preflight::TableId::Chiplet(_)))
        .map(|e| e.product)
        .expect("chiplet endpoint missing");

    assert_ne!(
        cpu_product, chiplet_product,
        "preflight Π(γ + key) products must differ between ghost-activated \
         CPU and honest chiplet (CPU has two extra γ factors)"
    );
}

// =================================================================
// Multi-table Lookup
// =================================================================

const MT_BUS: &str = "audit_multi_table_bus";

#[derive(Clone)]
struct MultiTableLookupAir;

impl Air<F> for MultiTableLookupAir {
    fn num_columns(&self) -> usize {
        2
    }

    fn column_layout(&self) -> &[ColumnType] {
        &[ColumnType::B32, ColumnType::Bit]
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![(
            MT_BUS.into(),
            PermutationCheckSpec::new_lookup(vec![(Source::Column(0), b"mt_k" as &[u8])], Some(1)),
        )]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(1));

        cs.build()
    }
}

impl Program<F> for MultiTableLookupAir {
    fn chiplet_defs(&self) -> hekate_core::errors::Result<Vec<ChipletDef<F>>> {
        Ok(vec![
            ChipletDef::from_air(&MultiTableLookupAir)?,
            ChipletDef::from_air(&MultiTableLookupAir)?,
        ])
    }
}

fn mt_lookup_trace(num_vars: usize, key: u32) -> ColumnTrace {
    let mut tb = TraceBuilder::new(&[ColumnType::B32, ColumnType::Bit], num_vars).unwrap();
    tb.set_b32(0, 0, Block32::from(key)).unwrap();
    tb.set_bit(1, 0, Bit::ONE).unwrap();

    tb.build()
}

#[test]
fn lookup_bus_two_tables_mismatched_content_rejected() {
    let num_vars = 2;
    let num_rows = 1 << num_vars;

    let air = MultiTableLookupAir;

    let reader = mt_lookup_trace(num_vars, 0xA1A1_A1A1);
    let table_a = mt_lookup_trace(num_vars, 0xA1A1_A1A1);
    let table_b = mt_lookup_trace(num_vars, 0xB2B2_B2B2);

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(reader).with_chiplets(vec![table_a, table_b]);

    let config = Config {
        num_queries: 4,
        min_security_bits: 0,
        sumcheck_blinding_factor: 0,
        ..Config::default()
    };

    let proof_res = prove(
        b"BUS_MT", &air, &instance, &witness, &config, [0x5A; 32], None,
    );

    let accepted = match proof_res {
        Ok(proof) => {
            let mut vt = Transcript::<H>::new(b"BUS_MT");
            HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut vt, &config)
                .unwrap_or(false)
        }
        Err(_) => false,
    };

    assert!(
        !accepted,
        "SECURITY FAILURE: mismatched two-table Lookup accepted"
    );
}

// =====================================================
// Multi-bus chiplet fixtures (h_evals tests)
// =====================================================

const MULTI_BUS_0: &str = "audit_bus_0";
const MULTI_BUS_1: &str = "audit_bus_1";
const MULTI_BUS_2: &str = "audit_bus_2";

#[derive(Clone)]
struct MultiBusChiplet;

impl Air<F> for MultiBusChiplet {
    fn name(&self) -> String {
        "audit_multibus".to_string()
    }

    fn num_columns(&self) -> usize {
        2
    }

    fn column_layout(&self) -> &[ColumnType] {
        &[ColumnType::B32, ColumnType::Bit]
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        let spec = || {
            PermutationCheckSpec::new(
                vec![
                    (Source::Column(0), b"audit_v" as ChallengeLabel),
                    (Source::RowIndexLeBytes(4), REQUEST_IDX_LABEL),
                ],
                Some(1),
            )
        };

        vec![
            (MULTI_BUS_0.into(), spec()),
            (MULTI_BUS_1.into(), spec()),
            (MULTI_BUS_2.into(), spec()),
        ]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(1));

        cs.build()
    }
}

#[derive(Clone)]
struct MultiBusProgram;

impl Air<F> for MultiBusProgram {
    fn name(&self) -> String {
        "audit_multibus_main".to_string()
    }

    fn num_columns(&self) -> usize {
        1
    }

    fn column_layout(&self) -> &[ColumnType] {
        &[ColumnType::Bit]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(0));

        cs.build()
    }
}

impl Program<F> for MultiBusProgram {
    fn chiplet_defs(&self) -> hekate_core::errors::Result<Vec<ChipletDef<F>>> {
        Ok(vec![
            ChipletDef::from_air(&MultiBusChiplet)?,
            ChipletDef::from_air(&MultiBusChiplet)?,
        ])
    }
}

fn multibus_trace(num_vars: usize) -> ColumnTrace {
    let mut tb = TraceBuilder::new(&[ColumnType::B32, ColumnType::Bit], num_vars).unwrap();
    let num_rows = tb.num_rows();

    tb.set_b32(0, 0, Block32::from(0xCAFE_F00Du32)).unwrap();
    tb.fill_selector(1, num_rows - 1).unwrap();

    tb.build()
}

fn cfg() -> Config {
    Config {
        num_queries: 4,
        min_security_bits: 0,
        sumcheck_blinding_factor: 2,
        ldt_blinding_factor: 4,
        ..Config::default()
    }
}

fn multibus_proof() -> (
    MultiBusProgram,
    ProgramInstance<F>,
    Config,
    hekate_core::proofs::InnerProof<F>,
) {
    let num_vars = 4;
    let num_rows = 1 << num_vars;

    let air = MultiBusProgram;
    let main_trace = TraceBuilder::new(&[ColumnType::Bit], num_vars)
        .unwrap()
        .build();
    let chiplet_a = multibus_trace(num_vars);
    let chiplet_b = multibus_trace(num_vars);

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(main_trace).with_chiplets(vec![chiplet_a, chiplet_b]);
    let config = cfg();

    let proof = prove(
        b"AuditP0", &air, &instance, &witness, &config, [0xA5; 32], None,
    )
    .expect("multibus baseline prove");

    let mut vt = Transcript::<H>::new(b"AuditP0");
    assert!(
        HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut vt, &config).unwrap(),
        "multibus baseline must verify"
    );

    (air, instance, config, proof)
}

fn verify_rejects(
    air: &impl Program<F>,
    instance: &ProgramInstance<F>,
    config: &Config,
    proof: &hekate_core::proofs::InnerProof<F>,
) -> bool {
    let mut vt = Transcript::<H>::new(b"AuditP0");
    match HekateVerifier::<F, H>::verify(air, instance, proof, &mut vt, config) {
        Ok(true) => false,
        Ok(false) | Err(_) => true,
    }
}

// =====================================================
// h_evals ordering bound to claimed_sums
// =====================================================

#[test]
fn logup_h_evals_reordered_rejected() {
    let (air, instance, config, mut proof) = multibus_proof();

    let aux = &mut proof.chiplet_logup_aux[0];
    assert!(aux.h_evals.len() >= 2);

    aux.h_evals.swap(0, 1);

    assert!(
        verify_rejects(&air, &instance, &config, &proof),
        "SECURITY FAILURE: reordered h_evals accepted"
    );
}

// =====================================================
// Strict val_final == expected_val
// =====================================================

#[test]
fn h_eval_corruption_after_sumcheck_rejected() {
    let (air, instance, config, mut proof) = multibus_proof();

    let aux = &mut proof.chiplet_logup_aux[0];
    assert!(!aux.h_evals.is_empty());

    aux.h_evals[0].1 += F::ONE;

    assert!(
        verify_rejects(&air, &instance, &config, &proof),
        "SECURITY FAILURE: corrupted h_evals accepted"
    );
}

// =====================================================
// bus_id label decoupled from bus_specs ordering
// =====================================================

const FAKE_BUS_ID: &str = "fake_bus_z";

#[derive(Clone)]
struct LabelFlipChiplet {
    honest_bus_id: String,
    mode_honest: Arc<AtomicBool>,
}

impl Air<F> for LabelFlipChiplet {
    fn name(&self) -> String {
        format!("flip_{}", self.honest_bus_id)
    }

    fn num_columns(&self) -> usize {
        2
    }

    fn column_layout(&self) -> &[ColumnType] {
        &[ColumnType::B32, ColumnType::Bit]
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        let bus_id = if self.mode_honest.load(Ordering::SeqCst) {
            self.honest_bus_id.clone()
        } else {
            FAKE_BUS_ID.into()
        };

        vec![(
            bus_id,
            PermutationCheckSpec::new(
                vec![
                    (Source::Column(0), b"n9_key" as ChallengeLabel),
                    (Source::RowIndexLeBytes(4), REQUEST_IDX_LABEL),
                ],
                Some(1),
            ),
        )]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(1));

        cs.build()
    }
}

#[derive(Clone)]
struct LabelFlipHost {
    mode_honest: Arc<AtomicBool>,
}

impl Air<F> for LabelFlipHost {
    fn num_columns(&self) -> usize {
        1
    }

    fn column_layout(&self) -> &[ColumnType] {
        &[ColumnType::Bit]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(0));

        cs.build()
    }
}

impl Program<F> for LabelFlipHost {
    fn chiplet_defs(&self) -> hekate_core::errors::Result<Vec<ChipletDef<F>>> {
        Ok(vec![
            ChipletDef::from_air(&LabelFlipChiplet {
                honest_bus_id: "honest_bus_a".to_string(),
                mode_honest: self.mode_honest.clone(),
            })?,
            ChipletDef::from_air(&LabelFlipChiplet {
                honest_bus_id: "honest_bus_b".to_string(),
                mode_honest: self.mode_honest.clone(),
            })?,
        ])
    }
}

fn n9_chiplet_trace(num_vars: usize, key: u32) -> ColumnTrace {
    let mut tb = TraceBuilder::new(&[ColumnType::B32, ColumnType::Bit], num_vars).unwrap();

    tb.set_b32(0, 0, Block32::from(key)).unwrap();
    tb.set_bit(1, 0, Bit::ONE).unwrap();

    tb.build()
}

#[test]
fn logup_bus_id_proof_label_diverged_from_program_specs_rejected() {
    let num_vars = 4;
    let num_rows = 1 << num_vars;

    let mode_honest = Arc::new(AtomicBool::new(false));
    let air = LabelFlipHost {
        mode_honest: mode_honest.clone(),
    };

    let main_trace = TraceBuilder::new(&[ColumnType::Bit], num_vars)
        .unwrap()
        .build();

    let chiplet_a = n9_chiplet_trace(num_vars, 0xCAFEF00D);
    let chiplet_b = n9_chiplet_trace(num_vars, 0xCAFEF00D);

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(main_trace).with_chiplets(vec![chiplet_a, chiplet_b]);
    let config = cfg();

    let proof = prove(
        b"BusLabelFlip",
        &air,
        &instance,
        &witness,
        &config,
        [0xC7; 32],
        None,
    )
    .expect("prove under fake-label mode");

    for aux in &proof.chiplet_logup_aux {
        assert_eq!(aux.claimed_sums[0].0, FAKE_BUS_ID);
    }

    mode_honest.store(true, Ordering::SeqCst);

    let mut vt = Transcript::<H>::new(b"BusLabelFlip");
    let res = HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut vt, &config);

    assert!(
        res.is_err() || !res.unwrap(),
        "SECURITY FAILURE: proof bus_id labels diverge from program bus_specs"
    );
}
