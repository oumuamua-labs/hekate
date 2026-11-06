// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

mod multibus;

use hekate::core::config::Config;
use hekate::core::trace::{ColumnTrace, ColumnType, IntoTraceColumn, TraceBuilder, TraceColumn};
use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::{Block256, Flat};
use hekate_core::errors::{self, Error};
use hekate_math::{Bit, Block32, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::digest::program_id;
use hekate_program::permutation::{
    ChallengeLabel, PermutationCheckSpec, Source, bus_columns, bus_tag,
};
use hekate_program::{Air, FixedColumn, FixedShape, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_sdk::preflight;
use hekate_testkit::site::{
    BUS_SUM, CHIPLET_ZEROCHECK, FIXED_COLUMN, H_EVAL_BINDING, MULTIPROOF, OUTER_LINEAR,
};
use hekate_testkit::verdict::{rejected_at, rejected_with, verify_logged};
use hekate_verifier::HekateVerifier;
use multibus::{LABEL, cfg, multibus_proof};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

type F = Block256;
type H = DefaultHasher;

const BUS_ID: &str = "ghost_activation_bus";

const COL_KEY: usize = 0;
const COL_SELECTOR: usize = 1;
const COL_REQUEST_IDX: usize = 2;

const ZERO_KEY: Block32 = Block32(0);

#[derive(Clone)]
struct Endpoint {
    lookup: bool,
    active: usize,
}

fn layout() -> Vec<ColumnType> {
    vec![ColumnType::B32, ColumnType::Bit, ColumnType::B32]
}

impl Endpoint {
    fn spec(&self) -> PermutationCheckSpec {
        let sources = vec![
            (Source::Column(COL_KEY), b"kappa_key" as &[u8]),
            (Source::RowIndexLeBytes(4), b"kappa_clk" as &[u8]),
            (Source::Column(COL_REQUEST_IDX), b"kappa_row_idx" as &[u8]),
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

    fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
        vec![FixedColumn::prefix(COL_SELECTOR, self.active)]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
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

    fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
        self.cpu.fixed_columns()
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        self.cpu.constraint_ast()
    }
}

impl Program<F> for GhostProgram {
    fn chiplet_defs(&self) -> errors::Result<Vec<ChipletDef<F>>> {
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

fn run(
    program: &GhostProgram,
    cpu: &[(Block32, Bit)],
    table: &[(Block32, Bit)],
) -> (errors::Result<bool>, String) {
    let num_rows = 4;
    let seed = [0xC3u8; 32];

    let cpu_trace = build_trace(cpu, num_rows);
    let table_trace = build_trace(table, num_rows);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![table_trace]);
    let instance = ProgramInstance::new(num_rows, vec![]);

    let config = Config {
        num_queries: 4,
        min_security_bits: 0,
        zero_knowledge: false,
        fold_arity_log: 1,
        early_stop_vars: 1,
        ..Config::default()
    };

    let proof = prove(
        b"GHOST_ACTIVATION",
        program,
        &instance,
        &witness,
        &config,
        seed,
        None,
    )
    .expect("ghost activation prove");

    verify_logged(program, &instance, &config, &proof, b"GHOST_ACTIVATION")
}

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
        cpu: Endpoint {
            lookup: false,
            active: 2,
        },
        table: Endpoint {
            lookup: false,
            active: 2,
        },
    };

    let (outcome, _) = run(&program, &honest_cpu(), &honest_table());

    assert!(matches!(outcome, Ok(true)), "got {outcome:?}");
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

    let pinned = GhostProgram {
        cpu: Endpoint {
            lookup: false,
            active: 2,
        },
        table: Endpoint {
            lookup: false,
            active: 2,
        },
    };

    let (outcome, log) = run(&pinned, &cpu, &table);

    assert!(
        rejected_at(&outcome, &log, FIXED_COLUMN),
        "got {outcome:?}\n{log}"
    );

    let admitted = GhostProgram {
        cpu: Endpoint {
            lookup: false,
            active: 4,
        },
        table: Endpoint {
            lookup: false,
            active: 2,
        },
    };

    let (outcome, _) = run(&admitted, &cpu, &table);

    assert!(rejected_with(&outcome, BUS_SUM), "got {outcome:?}");
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

    let pinned = GhostProgram {
        cpu: Endpoint {
            lookup: false,
            active: 2,
        },
        table: Endpoint {
            lookup: false,
            active: 2,
        },
    };

    let program = GhostProgram {
        cpu: Endpoint {
            lookup: false,
            active: 4,
        },
        ..pinned.clone()
    };

    let num_rows = 4;
    let instance = ProgramInstance::new(num_rows, vec![]);

    let honest = ProgramWitness::new(build_trace(&honest_cpu(), num_rows))
        .with_chiplets(vec![build_trace(&honest_table(), num_rows)]);
    let control = preflight(&pinned, &instance, &honest).unwrap();

    assert!(control.is_clean(), "{control}");

    let cpu_trace = build_trace(&cpu, num_rows);
    let table_trace = build_trace(&table, num_rows);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![table_trace]);

    let report = preflight(&program, &instance, &witness).unwrap();

    assert!(
        report.constraint_violations.is_empty() && report.fixed_column_violations.is_empty(),
        "{report}"
    );

    let bus = report
        .bus_diagnostics
        .iter()
        .find(|d| d.bus_id == BUS_ID)
        .expect(
            "preflight must surface a bus_diagnostic for the ghost-activated bus; \
             a clean report on this witness would mean preflight uses a weaker \
             oracle than the runtime LogUp algebra",
        );

    assert!(bus.bus_imbalance, "{report}");

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
struct MultiTableLookupAir {
    origin: usize,
    count: usize,
}

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

    fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
        vec![FixedColumn {
            col_idx: 1,
            shape: FixedShape::Cadence {
                stride: 1,
                count: self.count,
                origin: self.origin,
                values: vec![F::ONE],
            },
        }]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
    }
}

impl Program<F> for MultiTableLookupAir {
    fn chiplet_defs(&self) -> errors::Result<Vec<ChipletDef<F>>> {
        Ok(vec![
            ChipletDef::from_air(&MultiTableLookupAir {
                origin: 0,
                count: 1,
            })?,
            ChipletDef::from_air(&MultiTableLookupAir {
                origin: 1,
                count: 1,
            })?,
        ])
    }
}

fn mt_lookup_trace(num_vars: usize, rows: &[(usize, u32)]) -> ColumnTrace {
    let mut tb = TraceBuilder::new(&[ColumnType::B32, ColumnType::Bit], num_vars).unwrap();
    for &(row, key) in rows {
        tb.set_b32(0, row, Block32::from(key)).unwrap();
        tb.set_bit(1, row, Bit::ONE).unwrap();
    }

    tb.build()
}

#[test]
fn lookup_bus_two_tables_mismatched_content_rejected() {
    let num_vars = 2;
    let num_rows = 1 << num_vars;

    let air = MultiTableLookupAir {
        origin: 0,
        count: 2,
    };

    let instance = ProgramInstance::new(num_rows, vec![]);

    let config = Config {
        num_queries: 4,
        min_security_bits: 0,
        fold_arity_log: 1,
        early_stop_vars: 1,
        zero_knowledge: false,
        ..Config::default()
    };

    let run = |table_b_key: u32| {
        let reader = mt_lookup_trace(num_vars, &[(0, 0xA1A1_A1A1), (1, 0xB2B2_B2B2)]);
        let table_a = mt_lookup_trace(num_vars, &[(0, 0xA1A1_A1A1)]);
        let table_b = mt_lookup_trace(num_vars, &[(1, table_b_key)]);
        let witness = ProgramWitness::new(reader).with_chiplets(vec![table_a, table_b]);

        let proof = prove(
            b"BUS_MT", &air, &instance, &witness, &config, [0x5A; 32], None,
        )
        .expect("two-table lookup prove");

        verify_logged(&air, &instance, &config, &proof, b"BUS_MT")
    };

    let (matched, _) = run(0xB2B2_B2B2);
    assert!(matches!(matched, Ok(true)), "got {matched:?}");

    let (outcome, _) = run(0xC3C3_C3C3);
    assert!(rejected_with(&outcome, BUS_SUM), "got {outcome:?}");
}

// =====================================================
// Bus-free fixture (h_evals tests)
// =====================================================

#[derive(Clone)]
struct BusFreeProgram;

impl Air<F> for BusFreeProgram {
    fn name(&self) -> String {
        "audit_bus_free_main".to_string()
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

impl Program<F> for BusFreeProgram {}

fn bus_free_proof() -> (
    BusFreeProgram,
    ProgramInstance<F>,
    Config,
    hekate_core::proofs::InnerProof<F>,
) {
    let num_vars = 5;

    let air = BusFreeProgram;
    let trace = TraceBuilder::new(&[ColumnType::Bit], num_vars)
        .unwrap()
        .build();

    let instance = ProgramInstance::new(1 << num_vars, vec![]);
    let witness = ProgramWitness::new(trace);

    let config = Config {
        zero_knowledge: false,
        ..cfg()
    };

    let proof = prove(LABEL, &air, &instance, &witness, &config, [0x5A; 32], None)
        .expect("bus-free baseline prove");

    assert_eq!(
        verify_logged(&air, &instance, &config, &proof, LABEL).0,
        Ok(true)
    );

    (air, instance, config, proof)
}

fn h_eval_rejected(zero_knowledge: bool, outcome: &errors::Result<bool>, log: &str) -> bool {
    rejected_with(outcome, CHIPLET_ZEROCHECK) && (zero_knowledge || log.contains(H_EVAL_BINDING))
}

// =====================================================
// h_evals ordering bound to claimed_sums
// =====================================================

#[test]
fn logup_h_evals_reordered_rejected() {
    for zero_knowledge in [true, false] {
        let (air, instance, config, mut proof) = multibus_proof(Config {
            zero_knowledge,
            ..cfg()
        });

        let aux = &mut proof.chiplet_logup_aux[0];
        assert!(aux.h_evals.len() >= 2);

        aux.h_evals.swap(0, 1);

        let (outcome, log) = verify_logged(&air, &instance, &config, &proof, LABEL);

        assert!(
            h_eval_rejected(zero_knowledge, &outcome, &log),
            "zero_knowledge={zero_knowledge}: got {outcome:?}\n{log}"
        );
    }
}

// =====================================================
// Strict val_final == expected_val
// =====================================================

#[test]
fn h_eval_corruption_after_sumcheck_rejected() {
    for zero_knowledge in [true, false] {
        let (air, instance, config, mut proof) = multibus_proof(Config {
            zero_knowledge,
            ..cfg()
        });

        let aux = &mut proof.chiplet_logup_aux[0];
        assert!(!aux.h_evals.is_empty());

        aux.h_evals[0] += F::ONE;

        let (outcome, log) = verify_logged(&air, &instance, &config, &proof, LABEL);

        assert!(
            h_eval_rejected(zero_knowledge, &outcome, &log),
            "zero_knowledge={zero_knowledge}: got {outcome:?}\n{log}"
        );
    }
}

// =====================================================
// h binding omitted on a bused table (presence check)
// =====================================================

#[test]
fn h_opening_omitted_on_bused_table_rejected() {
    let (air, instance, config, mut proof) = multibus_proof(cfg());

    assert!(proof.h_root.is_some());

    // Keep h_root, the transcript stays synced
    proof.eval_proof.openings.remove(1);

    let result = verify_logged(&air, &instance, &config, &proof, LABEL).0;

    assert!(
        matches!(
            result,
            Err(Error::Protocol {
                protocol: "fold",
                message: "fold proof does not match its layout",
            })
        ),
        "{result:?}"
    );
}

#[test]
fn h_root_omitted_on_bused_proof_rejected() {
    let (air, instance, config, mut proof) = multibus_proof(cfg());

    assert!(proof.h_root.is_some());

    proof.h_root = None;

    let result = verify_logged(&air, &instance, &config, &proof, LABEL).0;

    assert!(
        matches!(
            result,
            Err(Error::Protocol {
                protocol: "verifier",
                message: "h_root presence must match the helper tree",
            })
        ),
        "{result:?}"
    );
}

// =====================================================
// Proof made under other bus ids
// (program_id binds the transcript to the bus layout)
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
                    (Source::RowIndexLeBytes(4), b"n9_clk" as ChallengeLabel),
                ],
                Some(1),
            ),
        )]
    }

    fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
        vec![FixedColumn::prefix(1, 1)]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
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
    fn chiplet_defs(&self) -> errors::Result<Vec<ChipletDef<F>>> {
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
fn proof_under_other_bus_ids_rejected() {
    let num_vars = 5;
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

    assert!(preflight(&air, &instance, &witness).unwrap().is_clean());

    let (fake, _) = verify_logged(&air, &instance, &config, &proof, b"BusLabelFlip");
    assert!(matches!(fake, Ok(true)), "got {fake:?}");

    mode_honest.store(true, Ordering::SeqCst);

    let (outcome, _) = verify_logged(&air, &instance, &config, &proof, b"BusLabelFlip");

    assert!(
        rejected_with(&outcome, CHIPLET_ZEROCHECK),
        "got {outcome:?}"
    );
}

// =====================================================
// Zero-bus table must present empty LogUp aux
// =====================================================

#[test]
fn zero_bus_main_emits_empty_logup_aux() {
    let (_air, _instance, _config, proof) = multibus_proof(cfg());

    assert!(
        proof.main_logup_aux.claimed_sums.is_empty(),
        "MultiBusProgram main declares no permutation_checks; \
         the honest prover must emit empty claimed_sums"
    );

    assert!(
        proof.main_logup_aux.h_evals.is_empty(),
        "zero-bus main must emit empty h_evals"
    );
}

#[test]
fn zero_bus_main_injected_claimed_sum_rejected() {
    let (air, instance, config, mut proof) = multibus_proof(cfg());

    assert!(
        proof.main_logup_aux.claimed_sums.is_empty(),
        "precondition: MultiBusProgram main is a zero-bus table"
    );

    proof.main_logup_aux.claimed_sums.push(F::ONE);

    assert_eq!(
        verify_logged(&air, &instance, &config, &proof, LABEL).0,
        Err(Error::Protocol {
            protocol: "verifier",
            message: "logup_aux claimed_sums length mismatch with the h columns",
        })
    );
}

// =====================================================
// h binding present on a bus-free proof
// (presence check, opposite arm)
// =====================================================

#[test]
fn h_binding_on_bus_free_proof_rejected() {
    let (air, instance, config, proof) = bus_free_proof();
    let (_, _, _, bused) = multibus_proof(cfg());

    assert!(proof.h_root.is_none());

    let mut with_root = proof.clone();
    with_root.h_root = bused.h_root;

    let (result, _) = verify_logged(&air, &instance, &config, &with_root, LABEL);

    assert!(
        matches!(
            result,
            Err(Error::Protocol {
                protocol: "verifier",
                message: "h_root presence must match the helper tree",
            })
        ),
        "{result:?}"
    );

    let mut with_opening = proof;
    with_opening
        .eval_proof
        .openings
        .insert(1, bused.eval_proof.openings[1].clone());

    let (result, _) = verify_logged(&air, &instance, &config, &with_opening, LABEL);

    assert!(
        matches!(
            result,
            Err(Error::Protocol {
                protocol: "fold",
                message: "fold proof does not match its layout",
            })
        ),
        "{result:?}"
    );
}

// =====================================================
// Corrupted Merkle path on the h opening
// =====================================================

#[test]
fn h_opening_sibling_tamper_rejected() {
    let (air, instance, config, mut proof) = multibus_proof(cfg());

    let opening = &mut proof.eval_proof.openings[1];

    assert!(!opening.multiproof.siblings.is_empty());

    opening.multiproof.siblings[0] ^= 1;

    let (outcome, log) = verify_logged(&air, &instance, &config, &proof, LABEL);

    assert!(rejected_with(&outcome, MULTIPROOF), "{outcome:?}\n{log}");
}

// =====================================================
// Bus tag keeps buses of one tuple shape apart
// =====================================================

const TAG_BUS_A: &str = "tag_bus_a";
const TAG_BUS_B: &str = "tag_bus_b";
const TAG_VARS: usize = 5;

const TAG_WAIVER: &str = "see hekate/tests/logup.rs: keys are distinct row values by construction";

#[derive(Clone)]
struct TwoBuses {
    specs: Vec<(String, PermutationCheckSpec)>,
    cubic: bool,
}

impl Air<F> for TwoBuses {
    fn name(&self) -> String {
        "tag_two_buses".to_string()
    }

    fn num_columns(&self) -> usize {
        3
    }

    fn column_layout(&self) -> &[ColumnType] {
        &[ColumnType::B32, ColumnType::B32, ColumnType::B32]
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        self.specs.clone()
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();

        if self.cubic {
            let zero = cs.col(2);
            cs.constrain(zero * zero * zero);
        }

        cs.build()
    }
}

impl Program<F> for TwoBuses {}

fn tag_endpoint(
    bus_id: &str,
    sources: Vec<(Source, ChallengeLabel)>,
) -> (String, PermutationCheckSpec) {
    let spec = PermutationCheckSpec::new(sources, None).with_clock_waiver(TAG_WAIVER);

    (bus_id.into(), spec)
}

fn tag_trace() -> ColumnTrace {
    let mut tb = TraceBuilder::new(
        &[ColumnType::B32, ColumnType::B32, ColumnType::B32],
        TAG_VARS,
    )
    .unwrap();

    for row in 0..tb.num_rows() {
        tb.set_b32(0, row, Block32::from(row as u32 + 1)).unwrap();
        tb.set_b32(1, row, Block32::from(row as u32 + 0x100))
            .unwrap();
    }

    tb.build()
}

fn tag_outcome(air: &TwoBuses, zero_knowledge: bool) -> errors::Result<bool> {
    let instance = ProgramInstance::new(1 << TAG_VARS, vec![]);
    let witness = ProgramWitness::new(tag_trace());

    let config = Config {
        zero_knowledge,
        ..cfg()
    };

    let proof = prove(LABEL, air, &instance, &witness, &config, [0x7A; 32], None)
        .expect("tag fixture prove");

    verify_logged(air, &instance, &config, &proof, LABEL).0
}

#[test]
fn crossed_buses_of_one_tuple_shape_rejected() {
    let column = |c| vec![(Source::Column(c), b"v" as ChallengeLabel)];

    let air = TwoBuses {
        specs: vec![
            tag_endpoint(TAG_BUS_A, column(0)),
            tag_endpoint(TAG_BUS_A, column(1)),
            tag_endpoint(TAG_BUS_B, column(1)),
            tag_endpoint(TAG_BUS_B, column(0)),
        ],
        cubic: false,
    };

    let instance = ProgramInstance::new(1 << TAG_VARS, vec![]);
    let report = preflight(&air, &instance, &ProgramWitness::new(tag_trace())).unwrap();

    assert_eq!(report.bus_diagnostics.len(), 2);
    assert!(report.bus_diagnostics.iter().all(|d| d.bus_imbalance));

    let raw = tag_outcome(&air, false);
    assert!(rejected_with(&raw, BUS_SUM), "{raw:?}");

    let zk = tag_outcome(&air, true);
    assert!(matches!(zk, Ok(false)), "{zk:?}");
}

#[test]
fn slot_zero_const_does_not_merge_two_buses() {
    let lo = |bus_id: &str| bus_tag::<F>(bus_id).split().0;
    let offset = (lo(TAG_BUS_A) + lo(TAG_BUS_B)).0;

    let keyed = |constant: u128| {
        vec![
            (Source::Const(constant), b"c" as ChallengeLabel),
            (Source::Column(0), b"v" as ChallengeLabel),
        ]
    };

    let air = TwoBuses {
        specs: vec![
            tag_endpoint(TAG_BUS_A, keyed(offset)),
            tag_endpoint(TAG_BUS_B, keyed(0)),
        ],
        cubic: false,
    };

    let raw = tag_outcome(&air, false);
    assert!(rejected_with(&raw, BUS_SUM), "{raw:?}");

    let zk = tag_outcome(&air, true);
    assert!(matches!(zk, Ok(false)), "{zk:?}");
}

#[test]
fn buses_crossed_inside_pair_columns_rejected() {
    let column = |c| vec![(Source::Column(c), b"v" as ChallengeLabel)];

    let air = TwoBuses {
        specs: vec![
            tag_endpoint(TAG_BUS_A, column(0)),
            tag_endpoint(TAG_BUS_B, column(1)),
            tag_endpoint(TAG_BUS_A, column(1)),
            tag_endpoint(TAG_BUS_B, column(0)),
        ],
        cubic: true,
    };

    let columns = bus_columns(&air.specs, air.constraint_ast().max_degree() + 1);

    assert_eq!(columns.len(), 2);
    assert!(columns.iter().all(|c| {
        let second = c.second.expect("degree 4 pairs every Permutation column");
        air.specs[c.first].0 != air.specs[second].0
    }));

    let raw = tag_outcome(&air, false);
    assert!(rejected_with(&raw, BUS_SUM), "{raw:?}");

    let zk = tag_outcome(&air, true);
    assert!(matches!(zk, Ok(false)), "{zk:?}");
}

#[test]
fn balanced_twins_of_tag_fixtures_verify() {
    let column = |c| vec![(Source::Column(c), b"v" as ChallengeLabel)];
    let keyed = vec![
        (Source::Const(5), b"c" as ChallengeLabel),
        (Source::Column(0), b"v" as ChallengeLabel),
    ];

    let twins = [
        (
            vec![
                tag_endpoint(TAG_BUS_A, column(0)),
                tag_endpoint(TAG_BUS_A, column(0)),
                tag_endpoint(TAG_BUS_B, column(1)),
                tag_endpoint(TAG_BUS_B, column(1)),
            ],
            false,
        ),
        (
            vec![
                tag_endpoint(TAG_BUS_A, keyed.clone()),
                tag_endpoint(TAG_BUS_A, keyed),
            ],
            false,
        ),
        (
            vec![
                tag_endpoint(TAG_BUS_A, column(0)),
                tag_endpoint(TAG_BUS_B, column(1)),
                tag_endpoint(TAG_BUS_A, column(0)),
                tag_endpoint(TAG_BUS_B, column(1)),
            ],
            true,
        ),
    ];

    for (specs, cubic) in twins {
        let air = TwoBuses { specs, cubic };
        for zero_knowledge in [false, true] {
            let outcome = tag_outcome(&air, zero_knowledge);

            assert!(
                matches!(outcome, Ok(true)),
                "cubic={cubic} zero_knowledge={zero_knowledge}: {outcome:?}"
            );
        }
    }
}

// =====================================================
// Intra-table permutation bus
// =====================================================

/// Intra-table LogUp bus:
/// two specs on the same `bus_id`, one reading column 0,
/// the other reading column 1. Endpoint sums cancel iff the
/// two columns carry the same multiset under the shared selector.
fn same_bus_permutation_air(num_rows: usize) -> CircuitProgram<F> {
    let waiver = "see hekate/tests/logup.rs: synthetic intra-table reverse-permutation \
                  test, both endpoints positional on same trace; not a production \
                  bus shape, exercises LogUp algebra only";

    let mut cx = Circuit::<F>::new("SameBusPermutation", num_rows).unwrap();

    let data = cx.columns(2, ColumnType::B32);
    let selector = cx.column(ColumnType::Bit);

    cx.fix(
        selector,
        FixedShape::Cadence {
            stride: 1,
            count: num_rows,
            origin: 0,
            values: vec![F::ONE],
        },
    );

    for col in data.iter() {
        let spec = PermutationCheckSpec::new(
            vec![(Source::Column(col.index()), b"kappa_data" as &[u8])],
            Some(selector.index()),
        )
        .with_clock_waiver(waiver);

        cx.bus("same_bus", spec);
    }

    cx.compile().unwrap()
}

fn generate_permutation_trace(num_vars: usize) -> ColumnTrace {
    let num_rows = 1 << num_vars;

    let mut col_a: Vec<Block32> = Vec::with_capacity(num_rows);
    let mut col_b: Vec<Block32> = Vec::with_capacity(num_rows);
    let mut selector: Vec<Bit> = Vec::with_capacity(num_rows);

    // Col A:
    // [0, 1, ..., N-1]
    // Col B:
    // [N-1, ..., 1, 0]  (permutation of A)
    for i in 0..num_rows {
        col_a.push(Block32::from(i as u32));
        col_b.push(Block32::from((num_rows - 1 - i) as u32));
        selector.push(Bit::ONE);
    }

    let mut trace = ColumnTrace::new(num_vars).unwrap();
    trace.add_column(col_a.into_trace_column()).unwrap();
    trace.add_column(col_b.into_trace_column()).unwrap();
    trace.add_column(TraceColumn::Bit(selector)).unwrap();

    trace
}

/// Happy path:
/// identical multisets -> paired logUp sums cancel -> verify passes.
#[test]
fn logup_bus_happy_path() {
    let num_vars = 8;
    let num_rows = 1 << num_vars;
    let seed = [0xAAu8; 32];

    let air = same_bus_permutation_air(num_rows);
    let trace = generate_permutation_trace(num_vars);
    let witness = ProgramWitness::new(trace);
    let instance = ProgramInstance::new(num_rows, vec![]);

    let config = Config {
        num_queries: 4,
        min_security_bits: 0,
        zero_knowledge: true,
        ..Config::default()
    };

    let proof = prove(
        b"LogUp_Happy",
        &air,
        &instance,
        &witness,
        &config,
        seed,
        None,
    )
    .expect("Proving failed");

    assert_eq!(proof.main_logup_aux.claimed_sums.len(), 2);

    let mut verifier_transcript = Transcript::<H>::new(b"LogUp_Happy");
    let pinned_id = program_id(&air).unwrap();

    let ok = HekateVerifier::<F, H>::verify(
        &pinned_id,
        &air,
        &instance,
        &proof,
        &mut verifier_transcript,
        &config,
    )
    .expect("Verifier returned error");

    assert!(ok, "Verifier rejected a valid permutation proof");
}

/// Divergent multisets -> paired logUp
/// sums do NOT cancel -> verifier rejects.
#[test]
fn logup_bus_divergence_rejected() {
    let num_vars = 4;
    let num_rows = 1 << num_vars;
    let seed = [0xAAu8; 32];

    let air = same_bus_permutation_air(num_rows);

    let instance = ProgramInstance::new(num_rows, vec![]);
    let config = Config {
        num_queries: 2,
        min_security_bits: 0,
        fold_arity_log: 1,
        early_stop_vars: 1,
        ..Config::default()
    };

    let run = |corrupt: bool| {
        let mut trace = generate_permutation_trace(num_vars);

        // Corrupt column B: its multiset differs from column A
        if corrupt && let TraceColumn::B32(ref mut vals) = trace.columns[1] {
            vals[num_rows - 1] = Flat::from_raw(Block32::from(999u32));
        }

        let witness = ProgramWitness::new(trace);

        let proof = prove(
            b"LogUp_Divergence",
            &air,
            &instance,
            &witness,
            &config,
            seed,
            None,
        )
        .expect("Prover generates an honest proof of a non-permutation");

        verify_logged(&air, &instance, &config, &proof, b"LogUp_Divergence")
    };

    let (honest, _) = run(false);
    assert!(matches!(honest, Ok(true)), "got {honest:?}");

    let (outcome, log) = run(true);
    assert!(
        rejected_at(&outcome, &log, OUTER_LINEAR),
        "got {outcome:?}\n{log}"
    );
}
