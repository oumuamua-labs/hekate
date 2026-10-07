// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate::core::config::Config;
use hekate::core::trace::ColumnType;
use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::{Block128, TowerField};
use hekate_core::trace::TraceBuilder;
use hekate_math::{Bit, Block32, HardwareField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::digest::program_id;
use hekate_program::expander::VirtualExpander;
use hekate_program::{Air, FixedColumn, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;

type F = Block128;
type H = DefaultHasher;

const MAGIC: u32 = 0xDEAD_BEEF;

const NUM_VARS: usize = 4;
const NUM_ROWS: usize = 1 << NUM_VARS;

// =========================================================
// Synthetic Chiplet:
// 1×B32 -> 32 virtual bits + 1 control
// =========================================================

#[derive(Clone)]
struct PackedBitChiplet {
    expander: VirtualExpander,
}

impl PackedBitChiplet {
    fn new() -> Self {
        Self {
            expander: VirtualExpander::new()
                .expand_bits(1, ColumnType::B32)
                .control_bits(1)
                .build()
                .expect("PackedBitChiplet expander"),
        }
    }
}

impl Air<F> for PackedBitChiplet {
    fn name(&self) -> String {
        "packed_bit".to_string()
    }

    fn column_layout(&self) -> &'static [ColumnType] {
        &[ColumnType::B32, ColumnType::Bit]
    }

    fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
        vec![FixedColumn::last_row(32)]
    }

    fn virtual_expander(&self) -> Option<&VirtualExpander> {
        Some(&self.expander)
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        let bit0 = cs.col(0);
        let sel = cs.col(32);

        cs.constrain(sel * bit0 * (bit0 + cs.one()));

        cs.build()
    }
}

fn packed_bit_host() -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("packed_bit_host", NUM_ROWS).unwrap();

    let flag = cx.column(ColumnType::Bit);

    let cs = cx.cs();
    cs.assert_boolean(cs.col(flag.index()));

    cx.attach(ChipletDef::from_air(&PackedBitChiplet::new()).unwrap());

    cx.compile().unwrap()
}

// =========================================================
// Helpers
// =========================================================

fn make_chiplet_trace(magic: u32) -> hekate_core::trace::ColumnTrace {
    let layout = [ColumnType::B32, ColumnType::Bit];
    let mut tb = TraceBuilder::new(&layout, NUM_VARS).unwrap();

    tb.set_b32(0, 0, Block32(magic)).unwrap();

    for i in 0..NUM_ROWS {
        tb.set_bit(
            1,
            i,
            if i < NUM_ROWS - 1 {
                Bit::ONE
            } else {
                Bit::ZERO
            },
        )
        .unwrap();
    }

    tb.build()
}

fn make_main_trace() -> hekate_core::trace::ColumnTrace {
    let layout = [ColumnType::Bit];

    let mut tb = TraceBuilder::new(&layout, NUM_VARS).unwrap();

    for i in 0..NUM_ROWS {
        tb.set_bit(
            0,
            i,
            if i < NUM_ROWS - 1 {
                Bit::ONE
            } else {
                Bit::ZERO
            },
        )
        .unwrap();
    }

    tb.build()
}

fn make_test_system() -> (
    CircuitProgram<F>,
    ProgramInstance<F>,
    ProgramWitness<F, hekate_core::trace::ColumnTrace>,
) {
    let air = packed_bit_host();
    let instance = ProgramInstance::new(NUM_ROWS, vec![]);
    let witness =
        ProgramWitness::new(make_main_trace()).with_chiplets(vec![make_chiplet_trace(MAGIC)]);

    (air, instance, witness)
}

fn zk_config() -> Config {
    Config {
        num_queries: 4,
        min_security_bits: 0,
        zero_knowledge: true,
        ldt_support_size: 4,
        ..Config::default()
    }
}

// =========================================================
// TEST 1:
// Virtual Packing Eval Forgery
//
// Corrupt a virtual bit column evaluation in
// the chiplet's point_evaluation. The TensorPCS
// proximity check must catch the mismatch between
// claimed virtual values and the physically
// committed trace data.
// =========================================================

#[test]
fn virtual_packing_eval_forgery_rejected() {
    let (air, instance, witness) = make_test_system();
    let config = zk_config();
    let seed = [0xAAu8; 32];

    let mut proof = prove(
        b"VirtualPackEvalForgery",
        &air,
        &instance,
        &witness,
        &config,
        seed,
        None,
    )
    .expect("honest proof must succeed");

    let mut vt = Transcript::<H>::new(b"VirtualPackEvalForgery");
    let pinned_id = program_id(&air).unwrap();

    let ok = HekateVerifier::<F, H>::verify(&pinned_id, &air, &instance, &proof, &mut vt, &config)
        .expect("verification must not error");

    assert!(ok, "baseline must verify");

    // Corrupt virtual bit column 17
    let evals = &mut proof.chiplet_point_evaluations[0].1;
    assert!(evals.len() > 17);
    evals[17] += F::ONE;

    let mut at = Transcript::<H>::new(b"VirtualPackEvalForgery");
    let attack =
        HekateVerifier::<F, H>::verify(&pinned_id, &air, &instance, &proof, &mut at, &config);

    assert!(
        attack.is_err() || !attack.unwrap(),
        "forged virtual bit column 17 accepted",
    );
}

// =========================================================
// TEST 2:
// Virtual Expansion Witness Isolation
//
// Stage A:
// With ZK enabled, the magic witness bytes must
// NOT appear in chiplet LDT openings. The noise
// generation path through parse_virtual_row must
// mask the data.
// =========================================================

#[test]
fn virtual_expansion_witness_isolation() {
    let (air, instance, witness) = make_test_system();
    let seed = [0xBBu8; 32];
    let needle = Block32(MAGIC).to_hardware().into_raw().0.to_le_bytes();

    let scan_openings = |proof: &hekate_core::proofs::InnerProof<F>| -> bool {
        proof
            .eval_proof
            .ldt_proof
            .opened_columns
            .iter()
            .any(|col| col.windows(needle.len()).any(|w| w == needle))
    };

    let config_zk = zk_config();
    let proof_zk = prove(
        b"VirtualPackWitnessIsolation",
        &air,
        &instance,
        &witness,
        &config_zk,
        seed,
        None,
    )
    .expect("ZK proof must succeed");

    let mut vt_zk = Transcript::<H>::new(b"VirtualPackWitnessIsolation");
    let pinned_id = program_id(&air).unwrap();

    assert!(
        HekateVerifier::<F, H>::verify(
            &pinned_id, &air, &instance, &proof_zk, &mut vt_zk, &config_zk
        )
        .unwrap(),
        "ZK proof must verify"
    );

    assert!(
        !scan_openings(&proof_zk),
        "ZK enabled but witness bytes leaked in chiplet LDT openings",
    );

    // The non-systematic row code emits no verbatim
    // column, no real leak exists to detect.
    let mut planted = proof_zk.clone();

    let target = planted
        .eval_proof
        .ldt_proof
        .opened_columns
        .iter_mut()
        .find(|col| col.len() >= needle.len())
        .expect("opening wide enough to hold the needle");

    target[..needle.len()].copy_from_slice(&needle);

    assert!(
        scan_openings(&planted),
        "positive control: a planted needle must be detected by the scan",
    );
}
