// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! The `bound` tests mutate one proof field:
//! no outer row is reached. `violating_witnesses_are_rejected` runs an
//! honest prover on a broken witness, which only an outer row can reject.

use hekate::core::config::Config;
use hekate::core::proofs::InnerProof;
use hekate::core::trace::{ColumnTrace, ColumnType, TraceColumn};
use hekate::math::{Bit, Block32, Block256, TowerField};
use hekate_core::errors::Error;
use hekate_core::trace::{IntoTraceColumn, TraceBuilder};
use hekate_gadgets::{CpuMemColumns, MemoryEvent, RamChiplet, generate_ram_trace};
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::outer::{TableShape, TableStatics, table_plan};
use hekate_program::{Air, FixedShape, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_testkit::verdict::verify;
use hekate_verifier::rejection::{OuterStage, PoolStage, Rejection, Table, ZerocheckStage};

type F = Block256;

const MAIN_SUMCHECK: Result<(), Rejection> = Err(Rejection::Zerocheck {
    table: Table::Main,
    stage: ZerocheckStage::Sumcheck,
});

const CHIPLET_SUMCHECK: Result<(), Rejection> = Err(Rejection::Zerocheck {
    table: Table::Chiplet(0),
    stage: ZerocheckStage::Sumcheck,
});

const EVAL_SUMCHECK: Result<(), Rejection> = Err(Rejection::PoolEval(PoolStage::Sumcheck));

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
        num_queries: 2,
        fold_arity_log: 1,
        early_stop_vars: 1,
        min_security_bits: 0,
        zero_knowledge: true,
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

fn verdict<P: Program<F> + Sync>(
    label: &'static [u8],
    program: &P,
    instance: &ProgramInstance<F>,
    proof: &InnerProof<F>,
) -> Result<(), Rejection> {
    verify(program, instance, &config(), proof, label)
}

fn bump(value: &mut F) {
    *value += F::ONE;
}

fn h_claim(evaluation: &mut (Vec<F>, Vec<F>), first_h: usize, plane: usize) -> &mut F {
    &mut evaluation.1[first_h + plane]
}

fn first_h_claim(air: &impl Air<F>, statics: &TableStatics<'_, F>) -> usize {
    let shape = TableShape::from_air(air, statics).unwrap();
    let plan = table_plan(air, statics, &shape, &config()).unwrap();

    plan.total_claims() - plan.h_cols - plan.h_blinds
}

#[test]
fn honest_proofs_verify() {
    let (air, instance, proof) = pinned_case();
    assert_eq!(verdict(b"Tamper_Pinned", &air, &instance, &proof), Ok(()));

    let (air, instance, proof) = ram_case();
    assert_eq!(verdict(b"Tamper_Ram", &air, &instance, &proof), Ok(()));
}

#[test]
fn absorbed_claims_are_transcript_bound() {
    let (air, instance, proof) = pinned_case();
    let claims = proof.main_point_evaluation.1.len();

    for idx in 0..claims {
        let mut mutant = proof.clone();
        bump(&mut mutant.main_point_evaluation.1[idx]);

        assert_eq!(
            verdict(b"Tamper_Pinned", &air, &instance, &mutant),
            EVAL_SUMCHECK,
            "claim {idx}"
        );
    }
}

#[test]
fn absorbed_sumcheck_stream_is_transcript_bound() {
    let (air, instance, proof) = pinned_case();

    let mut mutant = proof.clone();
    bump(&mut mutant.zerocheck_proof.round_polys[0].evals[0]);

    assert_eq!(
        verdict(b"Tamper_Pinned", &air, &instance, &mutant),
        MAIN_SUMCHECK
    );

    let mut mutant = proof.clone();
    bump(&mut mutant.zerocheck_proof.claimed_evaluation);

    assert_eq!(
        verdict(b"Tamper_Pinned", &air, &instance, &mutant),
        MAIN_SUMCHECK
    );

    let mut mutant = proof.clone();
    bump(&mut mutant.eval_proof.sumcheck_proof.round_polys[0].evals[0]);

    assert_eq!(
        verdict(b"Tamper_Pinned", &air, &instance, &mutant),
        EVAL_SUMCHECK
    );

    let mut mutant = proof.clone();
    bump(&mut mutant.eval_proof.sumcheck_proof.claimed_evaluation);

    assert_eq!(
        verdict(b"Tamper_Pinned", &air, &instance, &mutant),
        EVAL_SUMCHECK
    );
}

#[test]
fn absorbed_bus_values_are_transcript_bound() {
    let (air, instance, proof) = ram_case();
    let chiplets = air.chiplet_defs().unwrap();
    let main_first_h = first_h_claim(&air, &air.statics());
    let chiplet_first_h = first_h_claim(&chiplets[0], &chiplets[0].statics());

    for column in 0..proof.main_logup_aux.claimed_sums.len() {
        let mut mutant = proof.clone();
        bump(&mut mutant.main_logup_aux.claimed_sums[column]);

        assert_eq!(
            verdict(b"Tamper_Ram", &air, &instance, &mutant),
            MAIN_SUMCHECK,
            "main claimed_sum column {column}"
        );
    }

    for column in 0..proof.chiplet_logup_aux[0].claimed_sums.len() {
        let mut mutant = proof.clone();
        bump(&mut mutant.chiplet_logup_aux[0].claimed_sums[column]);

        assert_eq!(
            verdict(b"Tamper_Ram", &air, &instance, &mutant),
            CHIPLET_SUMCHECK,
            "chiplet claimed_sum column {column}"
        );
    }

    let main_planes = proof.main_logup_aux.h_evals.len();

    // The h_eval and its base h claim move together, as their pin row ties them
    for plane in 0..main_planes {
        let mut mutant = proof.clone();

        bump(&mut mutant.main_logup_aux.h_evals[plane]);
        bump(h_claim(
            &mut mutant.main_point_evaluation,
            main_first_h,
            plane,
        ));

        assert_eq!(
            verdict(b"Tamper_Ram", &air, &instance, &mutant),
            EVAL_SUMCHECK,
            "main h_eval plane {plane}"
        );
    }

    let chiplet_planes = proof.chiplet_logup_aux[0].h_evals.len();

    for plane in 0..chiplet_planes {
        let mut mutant = proof.clone();

        bump(&mut mutant.chiplet_logup_aux[0].h_evals[plane]);
        bump(h_claim(
            &mut mutant.chiplet_point_evaluations[0],
            chiplet_first_h,
            plane,
        ));

        assert_eq!(
            verdict(b"Tamper_Ram", &air, &instance, &mutant),
            MAIN_SUMCHECK,
            "chiplet h_eval plane {plane}"
        );
    }

    let mut mutant = proof.clone();
    bump(&mut mutant.chiplet_point_evaluations[0].1[0]);

    assert_eq!(
        verdict(b"Tamper_Ram", &air, &instance, &mutant),
        MAIN_SUMCHECK,
        "chiplet ring claim"
    );
}

#[test]
fn outer_segment_fields_are_bound() {
    let (air, instance, proof) = ram_case();

    let coverage = Err(Rejection::Outer(OuterStage::Coverage));
    let opening = Err(Rejection::Outer(OuterStage::Opening));

    let mut mutant = proof.clone();
    bump(&mut mutant.outer.as_mut().unwrap().interleaved[0]);

    assert_eq!(
        verdict(b"Tamper_Ram", &air, &instance, &mutant),
        coverage,
        "interleaved"
    );

    let mut mutant = proof.clone();
    bump(&mut mutant.outer.as_mut().unwrap().linear[0]);

    assert_eq!(
        verdict(b"Tamper_Ram", &air, &instance, &mutant),
        coverage,
        "linear"
    );

    let mut mutant = proof.clone();
    bump(&mut mutant.outer.as_mut().unwrap().quadratic[0]);

    assert_eq!(
        verdict(b"Tamper_Ram", &air, &instance, &mutant),
        coverage,
        "quadratic"
    );

    let mut mutant = proof.clone();
    bump(&mut mutant.outer.as_mut().unwrap().pad_opening.values[0]);

    assert_eq!(
        verdict(b"Tamper_Ram", &air, &instance, &mutant),
        opening,
        "pad opening"
    );

    let mut mutant = proof.clone();
    bump(&mut mutant.outer.as_mut().unwrap().aux_opening.values[0]);

    assert_eq!(
        verdict(b"Tamper_Ram", &air, &instance, &mutant),
        opening,
        "aux opening"
    );

    let mut mutant = proof.clone();
    mutant.outer.as_mut().unwrap().aux_root[0] ^= 1;

    assert_eq!(
        verdict(b"Tamper_Ram", &air, &instance, &mutant),
        coverage,
        "aux root"
    );

    let mut mutant = proof.clone();
    mutant.pad_root.as_mut().unwrap()[0] ^= 1;

    assert_eq!(
        verdict(b"Tamper_Ram", &air, &instance, &mutant),
        CHIPLET_SUMCHECK,
        "pad root"
    );

    let mut mutant = proof.clone();
    mutant.outer = None;

    assert_eq!(
        verdict(b"Tamper_Ram", &air, &instance, &mutant),
        Err(Rejection::Malformed(Error::Protocol {
            protocol: "verifier",
            message: "pad_root and outer presence must match zero_knowledge",
        })),
        "missing outer"
    );
}

/// Under hiding only the outer rows can
/// reject an honest run on a violating witness.
#[test]
fn violating_witnesses_are_rejected() {
    let linear = Err(Rejection::Outer(OuterStage::Linear));

    let (air, instance, proof) = pinned_case_with(10, |values, _| {
        values[5] = Block32::from(11u32);
    });

    assert_eq!(
        verdict(b"Tamper_Pinned", &air, &instance, &proof),
        linear,
        "broken transition"
    );

    let (air, instance, proof) = pinned_case_with(11, |_, _| {});

    assert_eq!(
        verdict(b"Tamper_Pinned", &air, &instance, &proof),
        linear,
        "wrong boundary value"
    );

    let (air, instance, proof) = pinned_case_with(10, |_, q| {
        let last = q.len() - 1;
        q[last] = Bit::ONE;
    });

    assert_eq!(
        verdict(b"Tamper_Pinned", &air, &instance, &proof),
        linear,
        "fixed selector violated"
    );

    let cpu = ram_events();

    let mut ram = cpu.clone();
    ram[2] = MemoryEvent::read(0x1000, 2, 43);
    ram[0] = MemoryEvent::write(0x1000, 0, 43);

    let (air, instance, proof) = ram_case_with(&cpu, &ram);

    assert_eq!(
        verdict(b"Tamper_Ram", &air, &instance, &proof),
        linear,
        "bus multisets diverge"
    );
}

#[test]
fn outer_multiproof_sections_are_bound() {
    let (air, instance, proof) = ram_case();
    let opening = Err(Rejection::Outer(OuterStage::Opening));

    let mut mutant = proof.clone();
    mutant
        .outer
        .as_mut()
        .unwrap()
        .pad_opening
        .multiproof
        .unreached[0] ^= 1;

    assert_eq!(
        verdict(b"Tamper_Ram", &air, &instance, &mutant),
        opening,
        "pad unreached node"
    );

    let mut mutant = proof.clone();
    mutant
        .outer
        .as_mut()
        .unwrap()
        .pad_opening
        .multiproof
        .siblings[0] ^= 1;

    assert_eq!(
        verdict(b"Tamper_Ram", &air, &instance, &mutant),
        opening,
        "pad sibling"
    );

    let mut mutant = proof.clone();
    mutant
        .outer
        .as_mut()
        .unwrap()
        .aux_opening
        .multiproof
        .siblings[0] ^= 1;

    assert_eq!(
        verdict(b"Tamper_Ram", &air, &instance, &mutant),
        opening,
        "aux sibling"
    );

    let mut mutant = proof;
    mutant
        .outer
        .as_mut()
        .unwrap()
        .aux_opening
        .multiproof
        .unreached
        .pop();

    assert_eq!(
        verdict(b"Tamper_Ram", &air, &instance, &mutant),
        opening,
        "aux unreached count"
    );
}
