// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::config::Config;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate_math::{Bit, Block32, Block128, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::digest::program_id;
use hekate_program::permutation::{
    BusKind, PermutationCheckSpec, Service, ServiceSlot, Side, Source,
};
use hekate_program::{FixedShape, ProgramInstance, ProgramWitness};
use hekate_sdk::{
    BundleProgram, DeserializedBundle, deserialize_bundle, preflight, serialize_bundle_header,
};

type F = Block128;

const BUS: &str = "ordered_bus";
const ROWS: usize = 16;

fn service() -> Service {
    Service {
        bus_id: BUS,
        kind: BusKind::Permutation,
        slots: vec![ServiceSlot::Value(b"kappa_value"), ServiceSlot::EmitRank],
        clock_waiver: None,
    }
}

fn two_calls() -> FixedShape<F> {
    FixedShape::Cadence {
        stride: 4,
        count: 2,
        origin: 0,
        values: vec![F::ONE, F::ZERO, F::ZERO, F::ONE],
    }
}

fn responder() -> ChipletDef<F> {
    let mut cx = Circuit::<F>::new("OrderedResponder", ROWS).unwrap();

    let value = cx.column(ColumnType::B32);
    let sel = cx.column(ColumnType::Bit);

    cx.fix(sel, two_calls());

    let respond = service()
        .respond(&[value.index()], &[], sel.index())
        .unwrap();

    cx.bus(BUS, respond);

    ChipletDef::from_air(&cx.compile().unwrap()).unwrap()
}

fn host() -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("OrderedHost", ROWS).unwrap();

    let value = cx.column(ColumnType::B32);
    let sel = cx.column(ColumnType::Bit);

    cx.fix(sel, two_calls());
    cx.call(&service(), &[value], sel).unwrap();
    cx.attach(responder());

    cx.compile().unwrap()
}

fn trace(values: [u32; 4]) -> ColumnTrace {
    let layout = [ColumnType::B32, ColumnType::Bit];
    let mut tb = TraceBuilder::new(&layout, ROWS.trailing_zeros() as usize).unwrap();

    for (row, value) in [0, 3, 4, 7].into_iter().zip(values) {
        tb.set_b32(0, row, Block32(value)).unwrap();
        tb.set_bit(1, row, Bit::ONE).unwrap();
    }

    tb.build()
}

fn rank_source(specs: &[(String, PermutationCheckSpec)]) -> Source {
    specs[0].1.sources[1].0.clone()
}

#[test]
fn emit_rank_sides_survive_wire() {
    let program = host();
    let instance = ProgramInstance::new(ROWS, vec![]);

    let bytes = serialize_bundle_header(&program, &instance, &Config::default()).unwrap();
    let restored: DeserializedBundle<F> = deserialize_bundle(&bytes).unwrap();

    assert_eq!(
        rank_source(&restored.permutation_checks),
        Source::EmitRank(Side::Request)
    );
    assert_eq!(
        rank_source(&restored.chiplet_defs[0].permutation_checks),
        Source::EmitRank(Side::Response)
    );
    assert_eq!(
        program_id::<F, _>(&program).unwrap(),
        program_id::<F, _>(&BundleProgram::from_bundle(&restored)).unwrap()
    );
}

#[test]
fn preflight_pairs_calls_by_rank() {
    let program = host();
    let served = trace([11, 12, 21, 22]);

    let report = |requested: [u32; 4]| {
        let instance = ProgramInstance::new(ROWS, vec![]);
        let witness =
            ProgramWitness::<F>::new(trace(requested)).with_chiplets(vec![served.clone()]);

        preflight(&program, &instance, &witness).unwrap()
    };

    assert!(report([11, 12, 21, 22]).is_clean());

    let traded = report([11, 22, 21, 12]);

    assert_eq!(traded.bus_diagnostics.len(), 1);
    assert!(traded.bus_diagnostics[0].bus_imbalance);
    assert!(traded.bus_diagnostics[0].clock_collisions.is_empty());
}
