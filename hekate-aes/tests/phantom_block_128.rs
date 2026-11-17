// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

mod common;

use common::{
    AES_ROWS, Aes128Program, FIPS128_CIPHER, FIPS128_KEY, FREE_CIPHER, HOST, IN_ROW, OUT_ROW, ROM,
    ROUND, assert_rejected, assert_verifies, build_cpu_trace_128, deactivate_rom, fips_call_128,
    make_program_128, whitened_128,
};
use hekate_aes::{
    Aes128Columns, AesRound128Air, CpuAes128Columns, PhysAes128Columns, SboxRomColumns,
    trace::{Aes128Call, expand_key, generate_aes_trace},
};
use hekate_core::trace::ColumnTrace;
use hekate_math::{Bit, TowerField};
use hekate_sdk::preflight::TableId;
use hekate_testkit::cell::{b8_at, copy_row, set_b8, set_b16, set_bit};
use hekate_testkit::fault::{Check, Fault};

const ACTIVE_ROWS: usize = 10;
const OUTPUT_ROW: usize = 10;

/// Oracle over the crate's own generator,
/// not a second AES implementation.
fn aes_rounds(state: &[u8; 16]) -> [u8; 16] {
    let round_keys = expand_key(&FIPS128_KEY);
    let plaintext: [u8; 16] = core::array::from_fn(|j| state[j] ^ round_keys[0][j]);

    let trace = generate_aes_trace(
        &[Aes128Call {
            key: FIPS128_KEY,
            plaintext,
            round_keys,
        }],
        AES_ROWS,
    )
    .unwrap();

    core::array::from_fn(|j| b8_at(&trace, PhysAes128Columns::P_STATE_IN + j, OUTPUT_ROW))
}

/// Clears the selectors and every cell pinned off its gate.
fn make_idle(aes: &mut ColumnTrace, row: usize) {
    for col in [
        PhysAes128Columns::P_S_ROUND,
        PhysAes128Columns::P_S_FINAL,
        PhysAes128Columns::P_S_IN_OUT,
        PhysAes128Columns::P_S_ACTIVE,
        PhysAes128Columns::P_S_INPUT,
    ] {
        set_bit(aes, col, row, Bit::ZERO);
    }

    for j in 0..4 {
        set_b8(aes, PhysAes128Columns::P_KS_INV + j, row, 0);
        set_b8(aes, PhysAes128Columns::P_K0_INV + j, row, 0);
        set_bit(aes, PhysAes128Columns::P_KS_Z + j, row, Bit::ZERO);
        set_bit(aes, PhysAes128Columns::P_K0_Z + j, row, Bit::ZERO);
    }

    set_b16(aes, PhysAes128Columns::P_ROUND_IDX, row, 0);
}

fn make_bare_emit(aes: &mut ColumnTrace, row: usize) {
    make_idle(aes, row);
    set_bit(aes, PhysAes128Columns::P_S_IN_OUT, row, Bit::ONE);
}

/// Moves the key bytes and their emit to row `to`.
fn move_key_row(cpu: &mut ColumnTrace, from: usize, to: usize) {
    copy_row(
        cpu,
        CpuAes128Columns::KEY..CpuAes128Columns::KEY + 16,
        from,
        to,
    );

    for j in 0..16 {
        set_b8(cpu, CpuAes128Columns::KEY + j, from, 0);
    }

    set_bit(cpu, CpuAes128Columns::KEY_SELECTOR, from, Bit::ZERO);
    set_bit(cpu, CpuAes128Columns::KEY_SELECTOR, to, Bit::ONE);
}

fn honest(air: &Aes128Program) -> (ColumnTrace, Vec<ColumnTrace>) {
    (
        build_cpu_trace_128(&[(whitened_128(), FIPS128_CIPHER)]),
        air.aes.generate_traces(&[fips_call_128()]).unwrap(),
    )
}

fn schedule_pins() -> Vec<(TableId, usize)> {
    let selectors = [
        Aes128Columns::S_ROUND,
        Aes128Columns::S_FINAL,
        Aes128Columns::S_IN_OUT,
        Aes128Columns::S_ACTIVE,
    ];

    (1..ACTIVE_ROWS)
        .map(|k| (ROUND, Aes128Columns::ROUND_BITS + k))
        .chain(selectors.map(|col| (ROUND, col)))
        .chain([(ROM, SboxRomColumns::SELECTOR)])
        .collect()
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn honest_block_verifies() {
    let air = make_program_128(AES_ROWS, 1);
    let (cpu_trace, chiplet_traces) = honest(&air);

    assert_verifies(&air.program, cpu_trace, chiplet_traces);
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn free_ciphertext_rejected() {
    let air = make_program_128(AES_ROWS, 1);
    let mut traces = air.aes.generate_traces(&[fips_call_128()]).unwrap();

    {
        let (head, tail) = traces.split_at_mut(1);
        let (aes, rom) = (&mut head[0], &mut tail[0]);

        make_bare_emit(aes, 1);
        make_bare_emit(aes, 2);

        let state = PhysAes128Columns::P_STATE_IN;
        copy_row(aes, state..state + 16, 1, 2);

        make_bare_emit(aes, 3);

        for (j, &byte) in FREE_CIPHER.iter().enumerate() {
            set_b8(aes, PhysAes128Columns::P_STATE_IN + j, 3, byte);
        }

        for row in 4..=OUTPUT_ROW {
            make_idle(aes, row);
        }

        deactivate_rom(rom, 1..ACTIVE_ROWS);
    }

    assert_ne!(FREE_CIPHER, FIPS128_CIPHER);

    let cpu_trace = build_cpu_trace_128(&[(whitened_128(), FREE_CIPHER)]);
    let pins = schedule_pins();
    let fault = Fault {
        constraint_tables: &[ROUND, ROM],
        pin_columns: &pins,
        buses: &[AesRound128Air::LINK_BUS_ID],
        first_check: Some(Check::FixedColumn),
        ..Fault::NONE
    };

    assert_rejected(&air.program, honest(&air), (cpu_trace, traces), fault);
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn single_round_block_rejected() {
    let air = make_program_128(AES_ROWS, 1);
    let mut traces = air.aes.generate_traces(&[fips_call_128()]).unwrap();

    let after_round_one: [u8; 16] =
        core::array::from_fn(|j| b8_at(&traces[0], PhysAes128Columns::P_STATE_IN + j, 1));

    {
        let (head, tail) = traces.split_at_mut(1);
        let (aes, rom) = (&mut head[0], &mut tail[0]);

        make_bare_emit(aes, 1);

        for row in 2..=OUTPUT_ROW {
            make_idle(aes, row);
        }

        deactivate_rom(rom, 1..ACTIVE_ROWS);
    }

    assert_ne!(after_round_one, FIPS128_CIPHER);

    let cpu_trace = build_cpu_trace_128(&[(whitened_128(), after_round_one)]);
    let pins = schedule_pins();
    let fault = Fault {
        constraint_tables: &[ROUND, ROM],
        pin_columns: &pins,
        buses: &[AesRound128Air::LINK_BUS_ID],
        first_check: Some(Check::FixedColumn),
        ..Fault::NONE
    };

    assert_rejected(&air.program, honest(&air), (cpu_trace, traces), fault);
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn backwards_block_rejected() {
    let air = make_program_128(AES_ROWS, 1);
    let traces = air.aes.generate_traces(&[fips_call_128()]).unwrap();

    let whitened = whitened_128();
    assert_ne!(aes_rounds(&FIPS128_CIPHER), whitened);

    let cpu_trace = build_cpu_trace_128(&[(FIPS128_CIPHER, whitened)]);
    let fault = Fault {
        buses: &[AesRound128Air::LINK_BUS_ID],
        ..Fault::NONE
    };

    assert_rejected(&air.program, honest(&air), (cpu_trace, traces), fault);
}

/// The key emit follows the reversed block to
/// the output row, where its pin forbids it.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn backwards_block_with_moved_key_rejected() {
    let air = make_program_128(AES_ROWS, 1);
    let traces = air.aes.generate_traces(&[fips_call_128()]).unwrap();

    let whitened = whitened_128();
    let mut cpu_trace = build_cpu_trace_128(&[(FIPS128_CIPHER, whitened)]);

    move_key_row(&mut cpu_trace, IN_ROW, OUT_ROW);

    let fault = Fault {
        pin_columns: &[(HOST, CpuAes128Columns::KEY_SELECTOR)],
        buses: &[AesRound128Air::KEY_BUS_ID, AesRound128Air::LINK_BUS_ID],
        first_check: Some(Check::FixedColumn),
        ..Fault::NONE
    };

    assert_rejected(&air.program, honest(&air), (cpu_trace, traces), fault);
}
