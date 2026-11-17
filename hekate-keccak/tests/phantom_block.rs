// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! A block that starts on a bare round row never
//! announces its input, and its state is free. No round
//! constraint fires; the cadence pins reject it first.

mod common;

use common::keccak_f;
use hekate_core::config::Config;
use hekate_core::trace::{ColumnTrace, TraceBuilder};
use hekate_keccak::{
    CpuKeccakColumns, KeccakChiplet, KeccakColumns, KeccakWitness, PhysKeccakColumns,
};
use hekate_math::{Bit, Block32, Block64, Block256, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram, Col};
use hekate_program::{Air, ProgramInstance, ProgramWitness};
use hekate_sdk::preflight::TableId;
use hekate_testkit::fault::{Check, Fault};
use hekate_testkit::verdict::{assert_rejected, assert_verifies};

type F = Block256;

const ROUNDS: usize = 24;
const CPU_ROWS: usize = 32;
const KECCAK_ROWS: usize = 64;

const KECCAK: TableId = TableId::Chiplet(0);

struct Row {
    state: [u64; 25],
    round_word: u32,
    s_round: bool,
    s_in_out: bool,
}

fn build_program() -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("KeccakPhantom", CPU_ROWS).unwrap();
    let cpu = cx.schema(&CpuKeccakColumns::build_layout());

    let selector = cpu.at(CpuKeccakColumns::SELECTOR);

    let call_values: Vec<Col> = (0..25)
        .map(|lane| cpu.at(CpuKeccakColumns::LANES + lane))
        .collect();

    cx.call(&KeccakChiplet::service(), &call_values, selector)
        .unwrap();

    cx.fix(
        selector,
        KeccakChiplet::host_selector_shape(KeccakChiplet::BLOCK_ROWS, 1),
    );

    cx.attach(ChipletDef::from_air(&KeccakChiplet::new(KECCAK_ROWS, 1)).unwrap());

    cx.compile().unwrap()
}

fn test_input() -> [u64; 25] {
    core::array::from_fn(|i| (i as u64).wrapping_mul(0x9E37_79B9) | 1)
}

fn cpu_trace(input: [u64; 25], output: [u64; 25]) -> ColumnTrace {
    let num_vars = CPU_ROWS.trailing_zeros() as usize;
    let mut tb = TraceBuilder::new(&CpuKeccakColumns::build_layout(), num_vars).unwrap();

    for i in 0..25 {
        tb.set_b64(CpuKeccakColumns::LANES + i, 0, Block64(input[i]))
            .unwrap();
        tb.set_b64(CpuKeccakColumns::LANES + i, ROUNDS, Block64(output[i]))
            .unwrap();
    }

    tb.set_bit(CpuKeccakColumns::SELECTOR, 0, Bit::ONE).unwrap();
    tb.set_bit(CpuKeccakColumns::SELECTOR, ROUNDS, Bit::ONE)
        .unwrap();

    tb.build()
}

fn write_rows(rows: &[Row]) -> ColumnTrace {
    let layout = Air::<F>::column_layout(&KeccakChiplet::new(KECCAK_ROWS, 1)).to_vec();
    let num_vars = KECCAK_ROWS.trailing_zeros() as usize;

    let mut tb = TraceBuilder::new(&layout, num_vars).unwrap();

    for (i, row) in rows.iter().enumerate() {
        for (lane, &value) in row.state.iter().enumerate() {
            tb.set_b64(lane, i, Block64(value)).unwrap();
        }

        tb.set_b32(PhysKeccakColumns::P_ROUND, i, Block32::from(row.round_word))
            .unwrap();
        tb.set_bit(
            PhysKeccakColumns::P_S_ROUND,
            i,
            if row.s_round { Bit::ONE } else { Bit::ZERO },
        )
        .unwrap();
        tb.set_bit(
            PhysKeccakColumns::P_S_IN_OUT,
            i,
            if row.s_in_out { Bit::ONE } else { Bit::ZERO },
        )
        .unwrap();
    }

    tb.build()
}

/// A 24-round block. `announce_input` false makes the
/// head a bare round row, only the tail reaches the bus.
fn block(input: [u64; 25], announce_input: bool, out: &mut Vec<Row>) {
    let mut state = input;

    for (round, rc) in KeccakChiplet::ROUND_CONSTANTS.iter().enumerate() {
        out.push(Row {
            state,
            round_word: 1u32 << round,
            s_round: true,
            s_in_out: announce_input && round == 0,
        });

        state = KeccakWitness::keccak_f_round(state, *rc);
    }

    out.push(Row {
        state,
        round_word: 0,
        s_round: false,
        s_in_out: true,
    });
}

/// Two rows:
/// a lone round-23 row plus its output row.
fn stub(seed: [u64; 25], out: &mut Vec<Row>) -> [u64; 25] {
    let rc = KeccakChiplet::ROUND_CONSTANTS[23];
    let image = KeccakWitness::keccak_f_round(seed, rc);

    out.push(Row {
        state: seed,
        round_word: 1u32 << 23,
        s_round: true,
        s_in_out: false,
    });

    out.push(Row {
        state: image,
        round_word: 0,
        s_round: false,
        s_in_out: true,
    });

    image
}

fn witness(cpu_input: [u64; 25], cpu_output: [u64; 25], rows: &[Row]) -> ProgramWitness<F> {
    ProgramWitness::new(cpu_trace(cpu_input, cpu_output)).with_chiplets(vec![write_rows(rows)])
}

fn honest_witness() -> ProgramWitness<F> {
    let input = test_input();

    let mut rows = Vec::new();
    block(input, true, &mut rows);

    witness(input, keccak_f(input), &rows)
}

fn config() -> Config {
    Config {
        num_queries: 2,
        fold_arity_log: 1,
        early_stop_vars: 1,
        ..Config::dev()
    }
}

fn schedule_pins() -> Vec<(TableId, usize)> {
    let selectors = [KeccakColumns::S_ROUND, KeccakColumns::S_IN_OUT];

    (0..ROUNDS)
        .map(|round| (KECCAK, KeccakColumns::ROUND_BITS + round))
        .chain(selectors.map(|col| (KECCAK, col)))
        .collect()
}

fn assert_phantom_rejected(forged: &ProgramWitness<F>) {
    let instance = ProgramInstance::new(CPU_ROWS, vec![]);
    let pins = schedule_pins();
    let fault = Fault {
        pin_columns: &pins,
        buses: &[KeccakChiplet::BUS_ID],
        first_check: Some(Check::FixedColumn),
        ..Fault::NONE
    };

    assert_rejected(
        &build_program(),
        (&instance, &honest_witness()),
        (&instance, forged),
        fault,
        &config(),
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn announced_block_verifies() {
    let instance = ProgramInstance::new(CPU_ROWS, vec![]);

    assert_verifies(&build_program(), (&instance, &honest_witness()), &config());
}

/// Emits the honest output twice. The second block
/// sits past the declared cadence, on rows pinned idle.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn unannounced_block_rejected() {
    let input = test_input();
    let mut rows = Vec::new();

    block(input, true, &mut rows);
    block(input, false, &mut rows);

    assert_phantom_rejected(&witness(input, keccak_f(input), &rows));
}

/// A two-row block reaching round 23 from a free
/// state answers the CPU's output row with any value.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn phantom_output_rejected() {
    let input = test_input();
    let honest = keccak_f(input);

    let mut rows = Vec::new();

    block(input, true, &mut rows);
    block(input, false, &mut rows);

    let forged = stub([0x11u64; 25], &mut rows);

    assert_ne!(forged, honest);
    assert!(rows.len() <= KECCAK_ROWS);

    assert_phantom_rejected(&witness(input, forged, &rows));
}
