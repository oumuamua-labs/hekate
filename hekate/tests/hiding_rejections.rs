// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! The `bound` tests mutate one proof field: transcript
//! or Merkle binding rejects them, no outer row is reached.
//! `violating_witnesses_are_rejected` runs an honest prover
//! on a broken witness, which only an outer row can reject.

use hekate::core::config::Config;
use hekate::core::proofs::{EvalBatchProof, InnerProof};
use hekate::core::trace::{ColumnTrace, ColumnType, TraceColumn};
use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::{Bit, Block32, Block128, TowerField};
use hekate_core::trace::{IntoTraceColumn, TraceBuilder};
use hekate_gadgets::{CpuMemColumns, MemoryEvent, RamChiplet, generate_ram_trace};
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::digest::program_id;
use hekate_program::{FixedShape, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;

type F = Block128;
type H = DefaultHasher;

fn pinned_air(num_rows: usize) -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("Pinned", num_rows).unwrap();

    let val_col = cx.column(ColumnType::B32);
    let q = cx.column(ColumnType::Bit);

    let cs = cx.cs();

    let val = cs.col(val_col.index());
    cs.constrain(cs.col(q.index()) * (cs.next(val_col.index()) + val));

    cx.fix(q, FixedShape::LastRow);
    cx.publish(val_col, 0);

    cx.compile().unwrap()
}

fn ram_air(num_rows: usize, num_events: usize) -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("RamHiding", num_rows).unwrap();
    let cpu = cx.schema(&CpuMemColumns::build_layout());

    cx.fix(
        cpu.at(CpuMemColumns::SELECTOR),
        FixedShape::Cadence {
            stride: 1,
            count: num_events,
            origin: 0,
            values: vec![F::ONE],
        },
    );

    cx.bus(RamChiplet::BUS_ID, RamChiplet::cpu_linking_spec());

    let cs = cx.cs();
    cs.assert_boolean(cs.col(CpuMemColumns::IS_WRITE));

    cx.attach(ChipletDef::from_air(&RamChiplet::new(num_rows, num_events)).unwrap());

    cx.compile().unwrap()
}

fn config() -> Config {
    Config {
        num_queries: 4,
        min_security_bits: 0,
        zero_knowledge: true,
        ldt_support_size: 4,
        ..Config::default()
    }
}

/// Honest witness with `public_input = 10`; `mutate` may
/// break the value column or the fixed selector.
fn pinned_case_with(
    public_input: u32,
    mutate: impl FnOnce(&mut Vec<Block32>, &mut Vec<Bit>),
) -> (CircuitProgram<F>, ProgramInstance<F>, InnerProof<F>) {
    let num_vars = 4;
    let num_rows = 1 << num_vars;

    let mut values = vec![Block32::from(10u32); num_rows];
    let mut q = vec![Bit::ONE; num_rows];
    q[num_rows - 1] = Bit::ZERO;

    mutate(&mut values, &mut q);

    let mut trace = ColumnTrace::new(num_vars).unwrap();
    trace.add_column(values.into_trace_column()).unwrap();
    trace.add_column(TraceColumn::Bit(q)).unwrap();

    let instance = ProgramInstance::new(num_rows, vec![F::from(public_input)]);
    let witness = ProgramWitness::new(trace);
    let air = pinned_air(num_rows);

    let proof = prove(
        b"Tamper_Pinned",
        &air,
        &instance,
        &witness,
        &config(),
        [0x51u8; 32],
        None,
    )
    .unwrap();

    (air, instance, proof)
}

fn pinned_case() -> (CircuitProgram<F>, ProgramInstance<F>, InnerProof<F>) {
    pinned_case_with(10, |_, _| {})
}

/// CPU side from `cpu_events`, RAM chiplet from `ram_events`.
fn ram_case_with(
    cpu_events: &[MemoryEvent],
    ram_events: &[MemoryEvent],
) -> (CircuitProgram<F>, ProgramInstance<F>, InnerProof<F>) {
    let num_vars = 4;
    let num_rows = 1 << num_vars;

    let mut tb = TraceBuilder::new(&CpuMemColumns::build_layout(), num_vars).unwrap();
    for (i, event) in cpu_events.iter().enumerate() {
        let addr = event.addr_bytes();
        let val = event.val_bytes();

        for j in 0..4 {
            tb.set_b32(CpuMemColumns::ADDR_B0 + j, i, Block32::from(addr[j] as u32))
                .unwrap();
            tb.set_b32(CpuMemColumns::VAL_B0 + j, i, Block32::from(val[j] as u32))
                .unwrap();
        }

        let is_write = if event.is_write { Bit::ONE } else { Bit::ZERO };

        tb.set_bit(CpuMemColumns::IS_WRITE, i, is_write).unwrap();
        tb.set_bit(CpuMemColumns::SELECTOR, i, Bit::ONE).unwrap();
    }

    let air = ram_air(num_rows, ram_events.len());
    let witness = ProgramWitness::new(tb.build())
        .with_chiplets(vec![generate_ram_trace(ram_events, num_rows).unwrap()]);
    let instance = ProgramInstance::new(num_rows, vec![]);

    let proof = prove(
        b"Tamper_Ram",
        &air,
        &instance,
        &witness,
        &config(),
        [0x52u8; 32],
        None,
    )
    .unwrap();

    (air, instance, proof)
}

fn ram_events() -> Vec<MemoryEvent> {
    vec![
        MemoryEvent::write(0x1000, 0, 42),
        MemoryEvent::write(0x2000, 1, 99),
        MemoryEvent::read(0x1000, 2, 42),
        MemoryEvent::read(0x2000, 3, 99),
    ]
}

fn ram_case() -> (CircuitProgram<F>, ProgramInstance<F>, InnerProof<F>) {
    let events = ram_events();
    ram_case_with(&events, &events)
}

fn accepted<P: Program<F> + Sync>(
    label: &'static [u8],
    program: &P,
    instance: &ProgramInstance<F>,
    proof: &InnerProof<F>,
) -> bool {
    let mut transcript = Transcript::<H>::new(label);
    let pinned_id = program_id(program).unwrap();

    matches!(
        HekateVerifier::<F, H>::verify(
            &pinned_id,
            program,
            instance,
            proof,
            &mut transcript,
            &config()
        ),
        Ok(true)
    )
}

fn bump(value: &mut F) {
    *value += F::ONE;
}

fn first_h_claim(eval: &mut EvalBatchProof<F>, num_buses: usize) -> &mut F {
    let half = eval.point_evaluation.1.len() / 2;

    &mut eval.point_evaluation.1[half - num_buses]
}

#[test]
fn honest_proofs_verify() {
    let (air, instance, proof) = pinned_case();
    assert!(accepted(b"Tamper_Pinned", &air, &instance, &proof));

    let (air, instance, proof) = ram_case();
    assert!(accepted(b"Tamper_Ram", &air, &instance, &proof));
}

#[test]
fn absorbed_claims_are_transcript_bound() {
    let (air, instance, proof) = pinned_case();
    let claims = proof.eval_proof.point_evaluation.1.len();

    for idx in 0..claims {
        let mut mutant = proof.clone();
        bump(&mut mutant.eval_proof.point_evaluation.1[idx]);

        assert!(
            !accepted(b"Tamper_Pinned", &air, &instance, &mutant),
            "claim {idx}"
        );
    }
}

#[test]
fn absorbed_sumcheck_stream_is_transcript_bound() {
    let (air, instance, proof) = pinned_case();

    let mut mutant = proof.clone();
    bump(&mut mutant.zerocheck_proof.round_polys[0].evals[0]);

    assert!(!accepted(b"Tamper_Pinned", &air, &instance, &mutant));

    let mut mutant = proof.clone();
    bump(&mut mutant.zerocheck_proof.claimed_evaluation);

    assert!(!accepted(b"Tamper_Pinned", &air, &instance, &mutant));

    let mut mutant = proof.clone();
    bump(&mut mutant.eval_proof.sumcheck_proof.round_polys[0].evals[0]);

    assert!(!accepted(b"Tamper_Pinned", &air, &instance, &mutant));

    let mut mutant = proof.clone();
    bump(&mut mutant.eval_proof.sumcheck_proof.claimed_evaluation);

    assert!(!accepted(b"Tamper_Pinned", &air, &instance, &mutant));
}

#[test]
fn absorbed_bus_values_are_transcript_bound() {
    let (air, instance, proof) = ram_case();

    let mut mutant = proof.clone();
    bump(&mut mutant.main_logup_aux.claimed_sums[0].1);

    assert!(
        !accepted(b"Tamper_Ram", &air, &instance, &mutant),
        "main claimed_sum"
    );

    let mut mutant = proof.clone();
    bump(&mut mutant.chiplet_logup_aux[0].claimed_sums[0].1);

    assert!(
        !accepted(b"Tamper_Ram", &air, &instance, &mutant),
        "chiplet claimed_sum"
    );

    // Moving the h_eval and its base h claim together
    // keeps the pin row, and the h opening rejects.
    let mut mutant = proof.clone();
    let num_buses = mutant.main_logup_aux.h_evals.len();

    bump(&mut mutant.main_logup_aux.h_evals[0].1);
    bump(first_h_claim(&mut mutant.eval_proof, num_buses));

    assert!(
        !accepted(b"Tamper_Ram", &air, &instance, &mutant),
        "main h_eval"
    );

    let mut mutant = proof.clone();
    let num_buses = mutant.chiplet_logup_aux[0].h_evals.len();

    bump(&mut mutant.chiplet_logup_aux[0].h_evals[0].1);
    bump(first_h_claim(&mut mutant.chiplet_eval_proofs[0], num_buses));

    assert!(
        !accepted(b"Tamper_Ram", &air, &instance, &mutant),
        "chiplet h_eval"
    );

    let mut mutant = proof.clone();
    bump(&mut mutant.chiplet_eval_proofs[0].point_evaluation.1[0]);

    assert!(
        !accepted(b"Tamper_Ram", &air, &instance, &mutant),
        "chiplet ring claim"
    );
}

#[test]
fn outer_segment_fields_are_bound() {
    let (air, instance, proof) = ram_case();

    let mut mutant = proof.clone();
    bump(&mut mutant.outer.as_mut().unwrap().interleaved[0]);

    assert!(
        !accepted(b"Tamper_Ram", &air, &instance, &mutant),
        "interleaved"
    );

    let mut mutant = proof.clone();
    bump(&mut mutant.outer.as_mut().unwrap().linear[0]);

    assert!(!accepted(b"Tamper_Ram", &air, &instance, &mutant), "linear");

    let mut mutant = proof.clone();
    bump(&mut mutant.outer.as_mut().unwrap().quadratic[0]);

    assert!(
        !accepted(b"Tamper_Ram", &air, &instance, &mutant),
        "quadratic"
    );

    let mut mutant = proof.clone();
    bump(&mut mutant.outer.as_mut().unwrap().pad_opening.values[0]);

    assert!(
        !accepted(b"Tamper_Ram", &air, &instance, &mutant),
        "pad opening"
    );

    let mut mutant = proof.clone();
    bump(&mut mutant.outer.as_mut().unwrap().aux_opening.values[0]);

    assert!(
        !accepted(b"Tamper_Ram", &air, &instance, &mutant),
        "aux opening"
    );

    let mut mutant = proof.clone();
    mutant.outer.as_mut().unwrap().aux_root[0] ^= 1;

    assert!(
        !accepted(b"Tamper_Ram", &air, &instance, &mutant),
        "aux root"
    );

    let mut mutant = proof.clone();
    mutant.pad_root.as_mut().unwrap()[0] ^= 1;

    assert!(
        !accepted(b"Tamper_Ram", &air, &instance, &mutant),
        "pad root"
    );

    let mut mutant = proof.clone();
    mutant.outer = None;

    assert!(
        !accepted(b"Tamper_Ram", &air, &instance, &mutant),
        "missing outer"
    );
}

/// Under hiding only the outer rows can
/// reject an honest run on a violating witness.
#[test]
fn violating_witnesses_are_rejected() {
    let (air, instance, proof) = pinned_case_with(10, |values, _| {
        values[5] = Block32::from(11u32);
    });

    assert!(
        !accepted(b"Tamper_Pinned", &air, &instance, &proof),
        "broken transition"
    );

    let (air, instance, proof) = pinned_case_with(11, |_, _| {});

    assert!(
        !accepted(b"Tamper_Pinned", &air, &instance, &proof),
        "wrong boundary value"
    );

    let (air, instance, proof) = pinned_case_with(10, |_, q| {
        let last = q.len() - 1;
        q[last] = Bit::ONE;
    });

    assert!(
        !accepted(b"Tamper_Pinned", &air, &instance, &proof),
        "fixed selector violated"
    );

    let cpu = ram_events();

    let mut ram = cpu.clone();
    ram[2] = MemoryEvent::read(0x1000, 2, 43);
    ram[0] = MemoryEvent::write(0x1000, 0, 43);

    let (air, instance, proof) = ram_case_with(&cpu, &ram);

    assert!(
        !accepted(b"Tamper_Ram", &air, &instance, &proof),
        "bus multisets diverge"
    );
}
