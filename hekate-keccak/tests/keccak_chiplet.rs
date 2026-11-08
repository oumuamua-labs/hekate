// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

mod common;

use common::keccak_f;
use hekate_core::config::Config;
use hekate_core::trace::{ColumnTrace, TraceBuilder};
use hekate_keccak::{
    CpuKeccakColumns, KeccakChiplet, KeccakColumns, PhysKeccakColumns, generate_keccak_trace,
};
use hekate_math::{Bit, Block64, Block256, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram, Col};
use hekate_program::{ProgramInstance, ProgramWitness};
use hekate_scribble::{MutationKind, ScribbleConfig, assert_all_caught_all_targets};
use hekate_sdk::preflight::TableId;
use hekate_testkit::cell::{copy_row, set_b64, set_bit, swap_b64_cells, xor_b32, xor_b64};
use hekate_testkit::fault::{Check, Fault};
use hekate_testkit::verdict::{assert_rejected, assert_verifies};

type F = Block256;

const CPU_ROWS: usize = 32;
const KECCAK_ROWS: usize = 32;

const IN_ROW: usize = 0;
const OUT_ROW: usize = KeccakChiplet::BLOCK_ROWS - 1;

const KECCAK: TableId = TableId::Chiplet(0);

const BUS_ONLY: Fault<'static> = Fault {
    buses: &[KeccakChiplet::BUS_ID],
    ..Fault::NONE
};

fn test_input_state() -> [u64; 25] {
    let mut state = [0u64; 25];
    state[0] = 0xDEADBEEFCAFEBABE;
    state[1] = 0x0123456789ABCDEF;
    state[2] = 0xFEDCBA9876543210;

    state
}

// =================================================================
// Test Program
// =================================================================

fn build_program(cpu_rows: usize, keccak_rows: usize, num_blocks: usize) -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("KeccakTest", cpu_rows).unwrap();
    let cpu = cx.schema(&CpuKeccakColumns::build_layout());

    let selector = cpu.at(CpuKeccakColumns::SELECTOR);

    let call_values: Vec<Col> = (0..25)
        .map(|lane| cpu.at(CpuKeccakColumns::LANES + lane))
        .collect();

    cx.call(&KeccakChiplet::service(), &call_values, selector)
        .unwrap();

    cx.fix(
        selector,
        KeccakChiplet::host_selector_shape(KeccakChiplet::BLOCK_ROWS, num_blocks),
    );

    cx.attach(ChipletDef::from_air(&KeccakChiplet::new(keccak_rows, num_blocks)).unwrap());

    cx.compile().unwrap()
}

// =================================================================
// Trace Builders
// =================================================================

fn build_cpu_trace(calls: &[([u64; 25], [u64; 25])], cpu_rows: usize) -> ColumnTrace {
    let num_vars = cpu_rows.trailing_zeros() as usize;
    let mut tb = TraceBuilder::new(&CpuKeccakColumns::build_layout(), num_vars).unwrap();

    for (k, (input, output)) in calls.iter().enumerate() {
        let in_row = 25 * k;
        let out_row = 25 * k + 24;

        for i in 0..25 {
            tb.set_b64(CpuKeccakColumns::LANES + i, in_row, Block64(input[i]))
                .unwrap();
            tb.set_b64(CpuKeccakColumns::LANES + i, out_row, Block64(output[i]))
                .unwrap();
        }

        tb.set_bit(CpuKeccakColumns::SELECTOR, in_row, Bit::ONE)
            .unwrap();
        tb.set_bit(CpuKeccakColumns::SELECTOR, out_row, Bit::ONE)
            .unwrap();
    }

    tb.build()
}

fn build_chiplet_trace(inputs: &[[u64; 25]], keccak_rows: usize) -> ColumnTrace {
    let blocks: Vec<[Block64; 25]> = inputs
        .iter()
        .map(|s| core::array::from_fn(|i| Block64(s[i])))
        .collect();

    generate_keccak_trace(&blocks, keccak_rows).unwrap()
}

// =================================================================
// Adversarial Harness
// =================================================================

fn config() -> Config {
    Config {
        num_queries: 2,
        fold_arity_log: 1,
        early_stop_vars: 1,
        ..Config::dev()
    }
}

fn assert_keccak_rejected<T>(fault: Fault<'_>, tamper: T)
where
    T: FnOnce(&mut ColumnTrace, &mut ColumnTrace),
{
    let input = test_input_state();

    let air = build_program(CPU_ROWS, KECCAK_ROWS, 1);
    let instance = ProgramInstance::new(CPU_ROWS, vec![]);

    let chiplet = build_chiplet_trace(&[input], KECCAK_ROWS);
    let cpu = build_cpu_trace(&[(input, keccak_f(input))], CPU_ROWS);
    let honest = ProgramWitness::new(cpu.clone()).with_chiplets(vec![chiplet.clone()]);

    let (mut forged_chiplet, mut forged_cpu) = (chiplet, cpu);
    tamper(&mut forged_chiplet, &mut forged_cpu);

    let forged = ProgramWitness::new(forged_cpu).with_chiplets(vec![forged_chiplet]);

    assert_rejected(
        &air,
        (&instance, &honest),
        (&instance, &forged),
        fault,
        &config(),
    );
}

// =================================================================
// Honest Path
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn keccak_e2e() {
    let input = test_input_state();

    let air = build_program(CPU_ROWS, KECCAK_ROWS, 1);
    let instance = ProgramInstance::new(CPU_ROWS, vec![]);
    let witness = ProgramWitness::new(build_cpu_trace(&[(input, keccak_f(input))], CPU_ROWS))
        .with_chiplets(vec![build_chiplet_trace(&[input], KECCAK_ROWS)]);

    assert_verifies(&air, (&instance, &witness), &config());
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn keccak_e2e_multi_call() {
    const ROWS: usize = 64;

    let in_a = test_input_state();
    let mut in_b = test_input_state();
    in_b[0] ^= u64::MAX;

    let calls = [(in_a, keccak_f(in_a)), (in_b, keccak_f(in_b))];

    let air = build_program(ROWS, ROWS, 2);
    let instance = ProgramInstance::new(ROWS, vec![]);
    let witness = ProgramWitness::new(build_cpu_trace(&calls, ROWS))
        .with_chiplets(vec![build_chiplet_trace(&[in_a, in_b], ROWS)]);

    assert_verifies(&air, (&instance, &witness), &config());
}

// =================================================================
// Round Constraint Exploits (Chi/Iota)
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_round_state_tamper() {
    let fault = Fault {
        constraint_tables: &[KECCAK],
        ..Fault::NONE
    };

    assert_keccak_rejected(fault, |keccak, _| {
        xor_b64(keccak, PhysKeccakColumns::P_LANES, 5, 0x01);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_round_index_tamper() {
    let fault = Fault {
        pin_columns: &[(KECCAK, KeccakColumns::ROUND_BITS)],
        first_check: Some(Check::Sumcheck),
        ..Fault::NONE
    };

    assert_keccak_rejected(fault, |keccak, _| {
        xor_b32(keccak, PhysKeccakColumns::P_ROUND, 3, 0x01);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_lane_swap() {
    let fault = Fault {
        constraint_tables: &[KECCAK],
        ..Fault::NONE
    };

    assert_keccak_rejected(fault, |keccak, _| {
        swap_b64_cells(
            keccak,
            PhysKeccakColumns::P_LANES,
            PhysKeccakColumns::P_LANES + 1,
            5,
        );
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_multi_round_tamper() {
    let fault = Fault {
        constraint_tables: &[KECCAK],
        ..Fault::NONE
    };

    assert_keccak_rejected(fault, |keccak, _| {
        xor_b64(keccak, PhysKeccakColumns::P_LANES + 3, 10, 0xFF);
        xor_b64(keccak, PhysKeccakColumns::P_LANES + 7, 15, 0xFF);
    });
}

// =================================================================
// Bus Integrity Exploits
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_chiplet_output_tamper() {
    let fault = Fault {
        constraint_tables: &[KECCAK],
        buses: &[KeccakChiplet::BUS_ID],
        ..Fault::NONE
    };

    assert_keccak_rejected(fault, |keccak, _| {
        xor_b64(keccak, PhysKeccakColumns::P_LANES, OUT_ROW, 0x01);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_cpu_input_tamper() {
    assert_keccak_rejected(BUS_ONLY, |_, cpu| {
        xor_b64(cpu, CpuKeccakColumns::LANES, IN_ROW, 0x01);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_keccak_link_input_replayed_as_output_rejected() {
    assert_keccak_rejected(BUS_ONLY, |_, cpu| {
        let lanes = CpuKeccakColumns::LANES;
        copy_row(cpu, lanes..lanes + 25, IN_ROW, OUT_ROW);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_cpu_output_tamper() {
    assert_keccak_rejected(BUS_ONLY, |_, cpu| {
        xor_b64(cpu, CpuKeccakColumns::LANES + 12, OUT_ROW, 0x01);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_consistent_output_forgery() {
    let fault = Fault {
        constraint_tables: &[KECCAK],
        ..Fault::NONE
    };

    assert_keccak_rejected(fault, |keccak, cpu| {
        // Tamper output on BOTH sides; the bus passes.
        // Round chain must still catch it:
        // next_bit[24] ≠ chi(state[23]).
        xor_b64(keccak, PhysKeccakColumns::P_LANES, OUT_ROW, 0x01);
        xor_b64(cpu, CpuKeccakColumns::LANES, OUT_ROW, 0x01);
    });
}

// =================================================================
// Selector Integrity Exploits
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_s_in_out_injection() {
    let fault = Fault {
        pin_columns: &[(KECCAK, KeccakColumns::S_IN_OUT)],
        buses: &[KeccakChiplet::BUS_ID],
        first_check: Some(Check::FixedColumn),
        ..Fault::NONE
    };

    assert_keccak_rejected(fault, |keccak, _| {
        set_bit(keccak, PhysKeccakColumns::P_S_IN_OUT, 5, Bit::ONE);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_s_in_out_deactivation() {
    let fault = Fault {
        pin_columns: &[(KECCAK, KeccakColumns::S_IN_OUT)],
        buses: &[KeccakChiplet::BUS_ID],
        first_check: Some(Check::FixedColumn),
        ..Fault::NONE
    };

    assert_keccak_rejected(fault, |keccak, _| {
        set_bit(keccak, PhysKeccakColumns::P_S_IN_OUT, OUT_ROW, Bit::ZERO);
    });
}

// =================================================================
// Ghost Protocol Exploits
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_ghost_state_injection() {
    let fault = Fault {
        pin_columns: &[(KECCAK, KeccakColumns::S_ROUND)],
        first_check: Some(Check::Sumcheck),
        ..Fault::NONE
    };

    assert_keccak_rejected(fault, |keccak, _| {
        // Inject non-zero state + s_round on padding row.
        // Row 26 is all-zero padding → chi(non-zero state)
        // produces non-zero output ≠ zero next_bit.
        set_bit(keccak, PhysKeccakColumns::P_S_ROUND, 25, Bit::ONE);
        set_b64(keccak, PhysKeccakColumns::P_LANES, 25, 0xDEADBEEF);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_s_round_deactivation() {
    let fault = Fault {
        pin_columns: &[(KECCAK, KeccakColumns::S_ROUND)],
        first_check: Some(Check::FixedColumn),
        ..Fault::NONE
    };

    assert_keccak_rejected(fault, |keccak, _| {
        set_bit(keccak, PhysKeccakColumns::P_S_ROUND, 23, Bit::ZERO);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_s_round_input_deactivation() {
    let fault = Fault {
        pin_columns: &[(KECCAK, KeccakColumns::S_ROUND)],
        first_check: Some(Check::FixedColumn),
        ..Fault::NONE
    };

    assert_keccak_rejected(fault, |keccak, _| {
        set_bit(keccak, PhysKeccakColumns::P_S_ROUND, IN_ROW, Bit::ZERO);
    });
}

#[test]
fn scribble_keccak_flip_selector_caught() {
    let input = test_input_state();

    let air = build_program(CPU_ROWS, KECCAK_ROWS, 1);
    let instance = ProgramInstance::new(CPU_ROWS, vec![]);
    let witness = ProgramWitness::new(build_cpu_trace(&[(input, keccak_f(input))], CPU_ROWS))
        .with_chiplets(vec![build_chiplet_trace(&[input], KECCAK_ROWS)]);

    assert_all_caught_all_targets(
        &air,
        &instance,
        &witness,
        ScribbleConfig::default()
            .mutations([MutationKind::FlipSelector])
            .cases(64),
    );
}
