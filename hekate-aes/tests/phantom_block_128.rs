// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

mod common;

use common::{
    AES_ROWS, FIPS128_CIPHER, FIPS128_KEY, FREE_CIPHER, IN_ROW, OUT_ROW, assert_air_clean,
    assert_air_violated, b8_at, build_cpu_trace_128, copy_b8_block, deactivate_rom, fips_call_128,
    make_program_128, prove_and_verify, set_b8, set_b16, set_bit, whitened_128,
};
use hekate_aes::{
    CpuAes128Columns, PhysAes128Columns,
    trace::{Aes128Call, expand_key, generate_aes_trace},
};
use hekate_core::trace::ColumnTrace;
use hekate_math::{Bit, TowerField};

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
    copy_b8_block(cpu, CpuAes128Columns::KEY, 16, from, to);

    for j in 0..16 {
        set_b8(cpu, CpuAes128Columns::KEY + j, from, 0);
    }

    set_bit(cpu, CpuAes128Columns::KEY_SELECTOR, from, Bit::ZERO);
    set_bit(cpu, CpuAes128Columns::KEY_SELECTOR, to, Bit::ONE);
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn honest_block_verifies() {
    let air = make_program_128(AES_ROWS, 1);
    let chiplet_traces = air.aes.generate_traces(&[fips_call_128()]).unwrap();
    let cpu_trace = build_cpu_trace_128(&[(whitened_128(), FIPS128_CIPHER)]);

    match prove_and_verify(&air.program, cpu_trace, chiplet_traces) {
        Ok(true) => {}
        Ok(false) => panic!("rejected"),
        Err(e) => panic!("error: {e}"),
    }
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

        copy_b8_block(aes, PhysAes128Columns::P_STATE_IN, 16, 1, 2);

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
    assert_air_violated(&air.program, &cpu_trace, &traces);

    match prove_and_verify(&air.program, cpu_trace, traces) {
        Ok(false) | Err(_) => {}
        Ok(true) => panic!("accepted"),
    }
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
    assert_air_violated(&air.program, &cpu_trace, &traces);

    match prove_and_verify(&air.program, cpu_trace, traces) {
        Ok(false) | Err(_) => {}
        Ok(true) => panic!("accepted"),
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn backwards_block_rejected() {
    let air = make_program_128(AES_ROWS, 1);
    let traces = air.aes.generate_traces(&[fips_call_128()]).unwrap();

    let whitened = whitened_128();
    assert_ne!(aes_rounds(&FIPS128_CIPHER), whitened);

    let cpu_trace = build_cpu_trace_128(&[(FIPS128_CIPHER, whitened)]);
    assert_air_clean(&air.program, &cpu_trace, &traces);

    match prove_and_verify(&air.program, cpu_trace, traces) {
        Ok(false) | Err(_) => {}
        Ok(true) => panic!("accepted"),
    }
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

    assert_air_violated(&air.program, &cpu_trace, &traces);

    match prove_and_verify(&air.program, cpu_trace, traces) {
        Ok(false) | Err(_) => {}
        Ok(true) => panic!("accepted"),
    }
}
