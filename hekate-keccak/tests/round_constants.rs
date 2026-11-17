// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::config::Config;
use hekate_core::trace::{ColumnTrace, ColumnType, Trace, TraceBuilder};
use hekate_keccak::{
    CpuKeccakColumns, KeccakChiplet, KeccakColumns, KeccakWitness, PhysKeccakColumns,
    generate_keccak_trace,
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

const ROWS_PER_CALL: usize = ROUNDS + 1;
const ROWS: usize = 32;

const KECCAK: TableId = TableId::Chiplet(0);

/// Which row sends the state and which receives
/// it is the statement, not a witness value.
fn build_program() -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("KeccakRoundConstants", ROWS).unwrap();
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

    cx.attach(ChipletDef::from_air(&KeccakChiplet::new(ROWS, 1)).unwrap());

    cx.compile().unwrap()
}

fn test_input() -> [u64; 25] {
    core::array::from_fn(|i| (i as u64).wrapping_mul(0x9E37_79B9) | 1)
}

fn constant_schedule(constants: &[u64; ROUNDS]) -> Vec<(u32, u64)> {
    constants
        .iter()
        .enumerate()
        .map(|(round, &rc)| (1u32 << round, rc))
        .collect()
}

fn subset_schedule(offsets: &[usize], steps: usize) -> Vec<(u32, u64)> {
    (0..steps)
        .map(|step| {
            offsets.iter().fold((0u32, 0u64), |(word, rc), &offset| {
                (
                    word | (1u32 << (offset + step)),
                    rc ^ KeccakChiplet::ROUND_CONSTANTS[offset + step],
                )
            })
        })
        .collect()
}

fn cpu_trace(input: [u64; 25], output: [u64; 25]) -> ColumnTrace {
    let num_vars = ROWS.trailing_zeros() as usize;
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

/// One block run under `schedule`, emitting
/// on its first row and on its last.
fn chiplet_trace(input: [u64; 25], schedule: &[(u32, u64)]) -> (ColumnTrace, [u64; 25]) {
    let layout = Air::<F>::column_layout(&KeccakChiplet::new(ROWS, 1)).to_vec();
    let num_vars = ROWS.trailing_zeros() as usize;

    let mut tb = TraceBuilder::new(&layout, num_vars).unwrap();
    let mut state = input;

    for (row, &(word, rc)) in schedule.iter().enumerate() {
        for (lane, &value) in state.iter().enumerate() {
            tb.set_b64(lane, row, Block64(value)).unwrap();
        }

        tb.set_b32(PhysKeccakColumns::P_ROUND, row, Block32::from(word))
            .unwrap();
        tb.set_bit(PhysKeccakColumns::P_S_ROUND, row, Bit::ONE)
            .unwrap();

        if row == 0 {
            tb.set_bit(PhysKeccakColumns::P_S_IN_OUT, row, Bit::ONE)
                .unwrap();
        }

        state = KeccakWitness::keccak_f_round(state, rc);
    }

    let out_row = schedule.len();

    for (lane, &value) in state.iter().enumerate() {
        tb.set_b64(lane, out_row, Block64(value)).unwrap();
    }

    tb.set_bit(PhysKeccakColumns::P_S_IN_OUT, out_row, Bit::ONE)
        .unwrap();

    (tb.build(), state)
}

fn witness(cpu_input: [u64; 25], cpu_output: [u64; 25], chiplet: ColumnTrace) -> ProgramWitness<F> {
    ProgramWitness::new(cpu_trace(cpu_input, cpu_output)).with_chiplets(vec![chiplet])
}

fn scheduled(schedule: &[(u32, u64)]) -> ProgramWitness<F> {
    let input = test_input();
    let (chiplet, output) = chiplet_trace(input, schedule);

    witness(input, output, chiplet)
}

fn canonical() -> Vec<(u32, u64)> {
    constant_schedule(&KeccakChiplet::ROUND_CONSTANTS)
}

fn config() -> Config {
    Config {
        num_queries: 2,
        fold_arity_log: 1,
        early_stop_vars: 1,
        ..Config::dev()
    }
}

fn assert_schedule_rejected(forged: &ProgramWitness<F>, fault: Fault<'_>) {
    let instance = ProgramInstance::new(ROWS, vec![]);

    assert_rejected(
        &build_program(),
        (&instance, &scheduled(&canonical())),
        (&instance, forged),
        fault,
        &config(),
    );
}

fn assert_short_schedule_rejected(schedule: &[(u32, u64)]) {
    let selectors = [KeccakColumns::S_ROUND, KeccakColumns::S_IN_OUT];
    let pins: Vec<(TableId, usize)> = (5..ROUNDS)
        .map(|round| (KECCAK, KeccakColumns::ROUND_BITS + round))
        .chain(selectors.map(|col| (KECCAK, col)))
        .collect();

    let fault = Fault {
        constraint_tables: &[KECCAK],
        pin_columns: &pins,
        buses: &[KeccakChiplet::BUS_ID],
        first_check: Some(Check::FixedColumn),
        ..Fault::NONE
    };

    assert_schedule_rejected(&scheduled(schedule), fault);
}

#[test]
fn layout_matches_crate() {
    let chiplet = KeccakChiplet::new(ROWS, 1);
    let layout = Air::<F>::column_layout(&chiplet);

    assert_eq!(layout.len(), PhysKeccakColumns::P_S_IN_OUT + 1);
    assert!(
        layout[..PhysKeccakColumns::P_ROUND]
            .iter()
            .all(|c| *c == ColumnType::B64)
    );
    assert_eq!(layout[PhysKeccakColumns::P_ROUND], ColumnType::B32);
    assert_eq!(layout[PhysKeccakColumns::P_S_ROUND], ColumnType::Bit);
    assert_eq!(layout[PhysKeccakColumns::P_S_IN_OUT], ColumnType::Bit);
}

#[test]
fn harness_matches_generate_keccak_trace() {
    let input = test_input();
    let block: [Block64; 25] = core::array::from_fn(|i| Block64(input[i]));

    let reference = generate_keccak_trace(&[block], ROWS).unwrap();
    let (mine, _) = chiplet_trace(input, &constant_schedule(&KeccakChiplet::ROUND_CONSTANTS));

    for col in 0..=PhysKeccakColumns::P_S_IN_OUT {
        for row in 0..ROWS_PER_CALL {
            assert_eq!(
                reference.get_element::<F>(col, row).unwrap(),
                mine.get_element::<F>(col, row).unwrap(),
                "column {col} row {row} diverges from generate_keccak_trace"
            );
        }
    }
}

#[test]
fn canonical_schedule_verifies() {
    let instance = ProgramInstance::new(ROWS, vec![]);

    assert_verifies(
        &build_program(),
        (&instance, &scheduled(&canonical())),
        &config(),
    );
}

/// The state chain is honest under a zero schedule,
/// the trace is consistent while the permutation
/// is not Keccak-f[1600].
#[test]
fn zero_schedule_rejected() {
    let fault = Fault {
        constraint_tables: &[KECCAK],
        ..Fault::NONE
    };

    assert_schedule_rejected(&scheduled(&constant_schedule(&[0u64; ROUNDS])), fault);
}

#[test]
fn flipped_constant_rejected() {
    let mut constants = KeccakChiplet::ROUND_CONSTANTS;
    constants[7] ^= 1;

    let fault = Fault {
        constraint_tables: &[KECCAK],
        ..Fault::NONE
    };

    assert_schedule_rejected(&scheduled(&constant_schedule(&constants)), fault);
}

/// An odd-weight set of round bits shifts in lockstep
/// and feeds iota the XOR of its constants; the
/// round-bit pins reject it.
#[test]
fn odd_weight_schedule_rejected() {
    assert_short_schedule_rejected(&subset_schedule(&[0, 5, 9], 15));
}

/// The round-bit and selector pins force
/// the chain to reach round 23.
#[test]
fn truncated_schedule_rejected() {
    assert_short_schedule_rejected(&subset_schedule(&[0], 5));
}

/// The host reads the block backwards, `f(x)` in
/// and `x` out; rank 0 holds `x` and both emits miss.
#[test]
fn backwards_block_rejected() {
    let x = test_input();
    let (chiplet, y) = chiplet_trace(x, &canonical());

    let fault = Fault {
        buses: &[KeccakChiplet::BUS_ID],
        ..Fault::NONE
    };

    assert_schedule_rejected(&witness(y, x, chiplet), fault);
}
