// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

mod common;

use common::{
    AES_ROWS, Aes256Program, FIPS256_CIPHER, FIPS256_KEY, FREE_CIPHER, HOST, IN_ROW, OUT_ROW, ROM,
    ROUND, assert_rejected, assert_verifies, build_cpu_trace_256, deactivate_rom, fips_call_256,
    make_program_256, whitened_256,
};
use hekate_aes::{
    Aes256Columns, AesRound256Air, CpuAes256Columns, PhysAes256Columns, SboxRomColumns,
    trace::{aes256_encrypt_block, expand_key_256},
};
use hekate_core::trace::ColumnTrace;
use hekate_math::{Bit, TowerField};
use hekate_sdk::preflight::TableId;
use hekate_testkit::cell::{copy_row, set_b8, set_b16, set_bit};
use hekate_testkit::fault::{Check, Fault};

const ACTIVE_ROWS: usize = 14;
const OUTPUT_ROW: usize = 14;

/// Fourteen rounds applied to `state`, which the block
/// enters already whitened.
fn aes_rounds(state: &[u8; 16]) -> [u8; 16] {
    let round_keys = expand_key_256(&FIPS256_KEY);
    let plaintext: [u8; 16] = core::array::from_fn(|j| state[j] ^ round_keys[0][j]);

    aes256_encrypt_block(&round_keys, &plaintext)
}

/// Clears the selectors and every cell pinned off its gate.
fn make_idle(aes: &mut ColumnTrace, row: usize) {
    for col in [
        PhysAes256Columns::P_S_ROUND,
        PhysAes256Columns::P_S_FINAL,
        PhysAes256Columns::P_S_IN_OUT,
        PhysAes256Columns::P_S_ACTIVE,
        PhysAes256Columns::P_S_INPUT,
    ] {
        set_bit(aes, col, row, Bit::ZERO);
    }

    for j in 0..4 {
        set_b8(aes, PhysAes256Columns::P_KS_INV + j, row, 0);
        set_bit(aes, PhysAes256Columns::P_KS_Z + j, row, Bit::ZERO);
    }

    set_b16(aes, PhysAes256Columns::P_ROUND_IDX, row, 0);
}

fn make_bare_emit(aes: &mut ColumnTrace, row: usize) {
    make_idle(aes, row);
    set_bit(aes, PhysAes256Columns::P_S_IN_OUT, row, Bit::ONE);
}

/// Moves the key bytes and their emit to row `to`.
fn move_key_row(cpu: &mut ColumnTrace, from: usize, to: usize) {
    copy_row(
        cpu,
        CpuAes256Columns::KEY..CpuAes256Columns::KEY + 32,
        from,
        to,
    );

    for j in 0..32 {
        set_b8(cpu, CpuAes256Columns::KEY + j, from, 0);
    }

    set_bit(cpu, CpuAes256Columns::KEY_SELECTOR, from, Bit::ZERO);
    set_bit(cpu, CpuAes256Columns::KEY_SELECTOR, to, Bit::ONE);
}

fn honest(air: &Aes256Program) -> (ColumnTrace, Vec<ColumnTrace>) {
    (
        build_cpu_trace_256(&whitened_256(), &FIPS256_CIPHER),
        air.aes.generate_traces(&[fips_call_256()]).unwrap(),
    )
}

fn schedule_pins() -> Vec<(TableId, usize)> {
    let selectors = [
        Aes256Columns::S_ROUND,
        Aes256Columns::S_FINAL,
        Aes256Columns::S_IN_OUT,
        Aes256Columns::S_ACTIVE,
    ];

    (1..ACTIVE_ROWS)
        .map(|k| (ROUND, Aes256Columns::ROUND_BITS + k))
        .chain(selectors.map(|col| (ROUND, col)))
        .chain([(ROM, SboxRomColumns::SELECTOR)])
        .collect()
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn honest_block_verifies() {
    let air = make_program_256(AES_ROWS, 1);
    let (cpu_trace, chiplet_traces) = honest(&air);

    assert_verifies(&air.program, cpu_trace, chiplet_traces);
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn free_ciphertext_rejected() {
    let air = make_program_256(AES_ROWS, 1);
    let mut traces = air.aes.generate_traces(&[fips_call_256()]).unwrap();

    {
        let (head, tail) = traces.split_at_mut(1);
        let (aes, rom) = (&mut head[0], &mut tail[0]);

        make_bare_emit(aes, 1);
        make_bare_emit(aes, 2);

        let state = PhysAes256Columns::P_STATE_IN;
        copy_row(aes, state..state + 16, 1, 2);

        make_bare_emit(aes, 3);

        for (j, &byte) in FREE_CIPHER.iter().enumerate() {
            set_b8(aes, PhysAes256Columns::P_STATE_IN + j, 3, byte);
        }

        for row in 4..=OUTPUT_ROW {
            make_idle(aes, row);
        }

        deactivate_rom(rom, 1..ACTIVE_ROWS);
    }

    assert_ne!(FREE_CIPHER, FIPS256_CIPHER);

    let cpu_trace = build_cpu_trace_256(&whitened_256(), &FREE_CIPHER);
    let pins = schedule_pins();
    let fault = Fault {
        constraint_tables: &[ROUND, ROM],
        pin_columns: &pins,
        buses: &[AesRound256Air::LINK_BUS_ID],
        first_check: Some(Check::FixedColumn),
        ..Fault::NONE
    };

    assert_rejected(&air.program, honest(&air), (cpu_trace, traces), fault);
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn backwards_block_rejected() {
    let air = make_program_256(AES_ROWS, 1);
    let traces = air.aes.generate_traces(&[fips_call_256()]).unwrap();

    let whitened = whitened_256();
    assert_ne!(aes_rounds(&FIPS256_CIPHER), whitened);

    let cpu_trace = build_cpu_trace_256(&FIPS256_CIPHER, &whitened);
    let fault = Fault {
        buses: &[AesRound256Air::LINK_BUS_ID],
        ..Fault::NONE
    };

    assert_rejected(&air.program, honest(&air), (cpu_trace, traces), fault);
}

/// The key emit follows the reversed block to
/// the output row, where its pin forbids it.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn backwards_block_with_moved_key_rejected() {
    let air = make_program_256(AES_ROWS, 1);
    let traces = air.aes.generate_traces(&[fips_call_256()]).unwrap();

    let whitened = whitened_256();
    let mut cpu_trace = build_cpu_trace_256(&FIPS256_CIPHER, &whitened);

    move_key_row(&mut cpu_trace, IN_ROW, OUT_ROW);

    let fault = Fault {
        pin_columns: &[(HOST, CpuAes256Columns::KEY_SELECTOR)],
        buses: &[AesRound256Air::KEY_BUS_ID, AesRound256Air::LINK_BUS_ID],
        first_check: Some(Check::FixedColumn),
        ..Fault::NONE
    };

    assert_rejected(&air.program, honest(&air), (cpu_trace, traces), fault);
}
