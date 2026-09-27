// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Even-multiplicity forgery on the AES<>SboxRom bus.
//!
//! Two identical AES-128 calls emit identical S-box keys.
//! The emit rank in the key is what keeps the pair from
//! cancelling in char-2, and that cancellation is the
//! only thing between `SBOX_OUT` and a free witness:
//! the round AIR consumes it as an input.

mod common;

use common::{
    FIPS128_CIPHER, SBOX_ROM_ROWS, assert_air_clean, build_cpu_trace_128, fips_call_128,
    make_program_128, prove_and_verify, whitened_128, xor_b8,
};
use hekate_aes::PhysAes128Columns;
use hekate_core::trace::ColumnTrace;

const AES_ROWS: usize = 32;

/// 9 full rounds, one final round, one output row.
const ROWS_PER_CALL: usize = 11;
const FINAL_ROUND_ROW: usize = 9;
const OUTPUT_ROW: usize = 10;
const CALLS: usize = 2;

/// ShiftRows fixes byte 0 (`SHIFT_MAP[0] = 0`), so the
/// final round propagates a delta on `SBOX_OUT[0]`
/// into ciphertext byte 0 unchanged.
const FORGED_BYTE: usize = 0;
const FORGED_DELTA: u8 = 0x5a;

const _: () = assert!(CALLS * ROWS_PER_CALL <= AES_ROWS);
const _: () = assert!(SBOX_ROM_ROWS >= 256);

/// Both blocks take the same delta; the two forged
/// S-box emissions differ only in the emit rank.
fn forge_final_round_sbox(aes: &mut ColumnTrace) {
    for block in 0..CALLS {
        let base = block * ROWS_PER_CALL;

        xor_b8(
            aes,
            PhysAes128Columns::P_SBOX_OUT + FORGED_BYTE,
            base + FINAL_ROUND_ROW,
            FORGED_DELTA,
        );

        xor_b8(
            aes,
            PhysAes128Columns::P_STATE_IN + FORGED_BYTE,
            base + OUTPUT_ROW,
            FORGED_DELTA,
        );
    }
}

fn two_call_traces() -> Vec<ColumnTrace> {
    let call = fips_call_128();
    make_program_128(AES_ROWS, 2)
        .aes
        .generate_traces(&[call.clone(), call])
        .unwrap()
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn two_identical_calls_e2e() {
    let air = make_program_128(AES_ROWS, 2);
    let whitened = whitened_128();

    let cpu_trace = build_cpu_trace_128(&[(whitened, FIPS128_CIPHER), (whitened, FIPS128_CIPHER)]);

    match prove_and_verify(&air.program, cpu_trace, two_call_traces()) {
        Ok(true) => {}
        Ok(false) => panic!("rejected two honest identical calls"),
        Err(e) => panic!("error: {e}"),
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes128_sbox_even_multiplicity_rejected() {
    let air = make_program_128(AES_ROWS, 2);
    let mut traces = two_call_traces();

    forge_final_round_sbox(&mut traces[0]);

    let mut forged = FIPS128_CIPHER;
    forged[FORGED_BYTE] ^= FORGED_DELTA;

    assert_ne!(forged, FIPS128_CIPHER);

    let whitened = whitened_128();
    let cpu_trace = build_cpu_trace_128(&[(whitened, forged), (whitened, forged)]);

    assert_air_clean(&air.program, &cpu_trace, &traces);

    match prove_and_verify(&air.program, cpu_trace, traces) {
        Ok(false) | Err(_) => {}
        Ok(true) => {
            panic!("accepted a ciphertext with SBOX_OUT[{FORGED_BYTE}] off by {FORGED_DELTA:#04x}")
        }
    }
}
