// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use std::ops::Range;

use hekate::core::config::Config;
use hekate::core::trace::{ColumnTrace, ColumnType};
use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::{Bit, Block64, Block128, Flat, HardwareField, TowerField};
use hekate_core::poly::PolyVariant;
use hekate_core::trace::{Trace, TraceBuilder};
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::digest::program_id;
use hekate_program::{ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;

type F = Block128;
type H = DefaultHasher;

const SECRET: usize = 0;
const ACTIVE: usize = 1;
const NUM_VARS: usize = 10;
const BLOCK: Range<usize> = 128..742;

const SLOT_LAYOUT: [ColumnType; 2] = [ColumnType::B64, ColumnType::Bit];

fn slot_constant_air() -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("SlotConstant", 1 << NUM_VARS).unwrap();
    cx.schema(&SLOT_LAYOUT);

    let cs = cx.cs();

    let [secret, active] = [cs.col(SECRET), cs.col(ACTIVE)];
    let [next_secret, next_active] = [cs.next(SECRET), cs.next(ACTIVE)];

    cs.assert_boolean(active);
    cs.constrain(active * next_active * (next_secret + secret));
    cs.assert_zero_when(cs.one() + active, secret);

    cx.compile().unwrap()
}

fn slot_constant_trace(secret: Block64) -> ColumnTrace {
    let mut tb = TraceBuilder::new(&SLOT_LAYOUT, NUM_VARS).unwrap();
    for row in BLOCK {
        tb.set_b64(SECRET, row, secret).unwrap();
        tb.set_bit(ACTIVE, row, Bit::ONE).unwrap();
    }

    tb.build()
}

// A column constant on a row block and zero elsewhere
// has MLE(r) = secret * sum_{i in block} eq(r, i).
fn recover_slot_constant(
    point_evaluation: &(Vec<F>, Vec<F>),
    col: usize,
    block: Range<usize>,
) -> F {
    let (r_final, claims) = point_evaluation;
    let r_hw: Vec<Flat<F>> = r_final.iter().map(|x| x.to_hardware()).collect();
    let weights = PolyVariant::<F>::expand_mle_weights(&r_hw);

    let block_weight = block
        .map(|row| weights[row])
        .fold(Flat::from_raw(F::ZERO), |acc, w| acc + w);

    claims[col] * block_weight.to_tower().invert()
}

fn recover_from_proof(zero_knowledge: bool) -> (F, F) {
    let secret = Block64::from(0x0BAD_C0FF_EE15_F00Du64);
    let trace = slot_constant_trace(secret);
    let secret_f = trace
        .get_element::<F>(SECRET, BLOCK.start)
        .unwrap()
        .to_tower();

    let air = slot_constant_air();
    let instance = ProgramInstance::new(1 << NUM_VARS, vec![]);
    let witness = ProgramWitness::new(trace);

    let config = Config {
        zero_knowledge,
        ldt_support_size: 4,
        num_queries: 4,
        min_security_bits: 0,
        ..Config::default()
    };

    let mut seed = [0u8; 32];
    seed[0] = 3;

    let proof = prove(
        b"SlotConstant",
        &air,
        &instance,
        &witness,
        &config,
        seed,
        None,
    )
    .unwrap();

    let mut vt = Transcript::<H>::new(b"SlotConstant");
    let pinned_id = program_id(&air).unwrap();

    assert!(
        HekateVerifier::<F, H>::verify(&pinned_id, &air, &instance, &proof, &mut vt, &config)
            .unwrap()
    );

    let recovered = recover_slot_constant(&proof.main_point_evaluation, SECRET, BLOCK);

    (recovered, secret_f)
}

#[test]
fn point_evaluation_hides_slot_constant_secret() {
    let (recovered, secret_f) = recover_from_proof(false);
    assert_eq!(
        recovered, secret_f,
        "raw claims must recover the slot constant, else the probe is broken"
    );

    let (recovered, secret_f) = recover_from_proof(true);
    assert_ne!(
        recovered, secret_f,
        "secret recovered exactly from point_evaluation and the public row layout"
    );
}
