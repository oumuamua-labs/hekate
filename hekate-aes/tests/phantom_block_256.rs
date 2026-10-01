// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

mod common;

use common::{
    AES_ROWS, FIPS256_CIPHER, FIPS256_KEY, FREE_CIPHER, IN_ROW, OUT_ROW, assert_air_clean,
    assert_air_violated, build_cpu_trace_256, copy_b8_block, deactivate_rom, fips_call_256,
    make_program_256, prove_and_verify, set_b8, set_b16, set_bit, whitened_256,
};
use hekate_aes::{
    CpuAes256Columns, PhysAes256Columns,
    trace::{aes256_encrypt_block, expand_key_256},
};
use hekate_core::trace::ColumnTrace;
use hekate_math::{Bit, TowerField};

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
    copy_b8_block(cpu, CpuAes256Columns::KEY, 32, from, to);

    for j in 0..32 {
        set_b8(cpu, CpuAes256Columns::KEY + j, from, 0);
    }

    set_bit(cpu, CpuAes256Columns::KEY_SELECTOR, from, Bit::ZERO);
    set_bit(cpu, CpuAes256Columns::KEY_SELECTOR, to, Bit::ONE);
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn honest_block_verifies() {
    let air = make_program_256(AES_ROWS, 1);
    let chiplet_traces = air.aes.generate_traces(&[fips_call_256()]).unwrap();
    let cpu_trace = build_cpu_trace_256(&whitened_256(), &FIPS256_CIPHER);

    match prove_and_verify(&air.program, cpu_trace, chiplet_traces) {
        Ok(true) => {}
        Ok(false) => panic!("rejected"),
        Err(e) => panic!("error: {e}"),
    }
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

        copy_b8_block(aes, PhysAes256Columns::P_STATE_IN, 16, 1, 2);

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
    assert_air_violated(&air.program, &cpu_trace, &traces);

    match prove_and_verify(&air.program, cpu_trace, traces) {
        Ok(false) | Err(_) => {}
        Ok(true) => panic!("accepted"),
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn backwards_block_rejected() {
    let air = make_program_256(AES_ROWS, 1);
    let traces = air.aes.generate_traces(&[fips_call_256()]).unwrap();

    let whitened = whitened_256();
    assert_ne!(aes_rounds(&FIPS256_CIPHER), whitened);

    let cpu_trace = build_cpu_trace_256(&FIPS256_CIPHER, &whitened);
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
    let air = make_program_256(AES_ROWS, 1);
    let traces = air.aes.generate_traces(&[fips_call_256()]).unwrap();

    let whitened = whitened_256();
    let mut cpu_trace = build_cpu_trace_256(&FIPS256_CIPHER, &whitened);

    move_key_row(&mut cpu_trace, IN_ROW, OUT_ROW);

    assert_air_violated(&air.program, &cpu_trace, &traces);

    match prove_and_verify(&air.program, cpu_trace, traces) {
        Ok(false) | Err(_) => {}
        Ok(true) => panic!("accepted"),
    }
}
