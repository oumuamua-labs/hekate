// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! End-to-end LogUp bus tests against
//! the high-level prover/verifier.

use hekate::core::config::Config;
use hekate::core::trace::{ColumnTrace, ColumnType, IntoTraceColumn, TraceColumn};
use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::{Bit, Block32, Block128, Flat};
use hekate_math::TowerField;
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::digest::program_id;
use hekate_program::permutation::{BUS_LIMBS, PermutationCheckSpec, Source};
use hekate_program::{FixedShape, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;

type F = Block128;
type H = DefaultHasher;

/// Intra-table LogUp bus:
/// two specs on the same `bus_id`, one reading
/// column 0, the other reading column 1. Endpoint
/// sums cancel iff the two columns carry the same
/// multiset under the shared selector.
fn same_bus_permutation_air(num_rows: usize) -> CircuitProgram<F> {
    let waiver = "see hekate/tests/gpa.rs: synthetic intra-table reverse-permutation \
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
/// identical multisets -> paired
/// LogUp sums cancel -> verify passes.
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

    assert_eq!(proof.main_logup_aux.claimed_sums.len(), 2 * BUS_LIMBS);

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

/// Divergent multisets -> paired LogUp
/// sums do NOT cancel → verifier rejects.
#[test]
fn logup_bus_divergence_rejected() {
    let num_vars = 4;
    let num_rows = 1 << num_vars;
    let seed = [0xAAu8; 32];

    let air = same_bus_permutation_air(num_rows);

    // Corrupt column B so its
    // multiset differs from column A.
    let mut trace = generate_permutation_trace(num_vars);
    if let TraceColumn::B32(ref mut vals) = trace.columns[1] {
        vals[num_rows - 1] = Flat::from_raw(Block32::from(999u32));
    }

    let witness = ProgramWitness::new(trace);
    let instance = ProgramInstance::new(num_rows, vec![]);
    let config = Config {
        ldt_support_size: 4,
        min_security_bits: 0,
        ..Config::default()
    };

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

    let mut verifier_transcript = Transcript::<H>::new(b"LogUp_Divergence");
    let pinned_id = program_id(&air).unwrap();

    let result = HekateVerifier::<F, H>::verify(
        &pinned_id,
        &air,
        &instance,
        &proof,
        &mut verifier_transcript,
        &config,
    );

    assert!(
        matches!(result, Ok(false)),
        "outer linear test must reject a non-permutation (bus sums do not cancel)",
    );
}
