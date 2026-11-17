// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Two AES calls trading ciphertexts or keys,
//! rejected in every block order at both key sizes.

mod common;

use common::{
    CPU_ROWS, F, FIPS128_KEY, FIPS128_PLAIN, FIPS256_KEY, FIPS256_PLAIN, assert_rejected,
    assert_verifies, make_program_128, make_program_256,
};
use hekate_aes::trace::{Aes128Call, Aes256Call, expand_key, expand_key_256};
use hekate_aes::{
    AesRound128Air, AesRound256Air, CpuAes128Columns, CpuAes256Columns, PhysAes128Columns,
};
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate_math::{Bit, Block8, TowerField};
use hekate_program::circuit::CircuitProgram;
use hekate_testkit::cell::b8_at;
use hekate_testkit::fault::Fault;

const AES_ROWS: usize = 32;

struct Level {
    program: CircuitProgram<F>,
    host: Host,
    link: &'static str,
    key: &'static str,
    keys: [Vec<u8>; 2],
    whitened: [[u8; 16]; 2],
    ciphers: [[u8; 16]; 2],
    blocks: Box<dyn Fn([usize; 2]) -> Vec<ColumnTrace>>,
}

struct Host {
    layout: Vec<ColumnType>,
    key: usize,
    key_selector: usize,
    data: usize,
    selector: usize,
}

#[derive(Clone, Copy)]
struct Claim {
    input: usize,
    output: usize,
    key: usize,
}

fn aes128() -> Level {
    let air = make_program_128(AES_ROWS, 2);

    let calls = [
        (FIPS128_KEY, FIPS128_PLAIN),
        (flip_first(FIPS128_KEY), flip_first(FIPS128_PLAIN)),
    ]
    .map(|(key, plaintext)| Aes128Call {
        key,
        plaintext,
        round_keys: expand_key(&key),
    });

    let honest = air.aes.generate_traces(&calls).unwrap();

    let whitened = calls
        .clone()
        .map(|call| core::array::from_fn(|j| call.plaintext[j] ^ call.round_keys[0][j]));

    let ciphers = [0, 1].map(|k| output_state(&honest[0], k, AesRound128Air::BLOCK_ROWS));

    let aes = air.aes.clone();
    let served = calls.clone();

    Level {
        program: air.program,
        host: Host {
            layout: CpuAes128Columns::build_layout(),
            key: CpuAes128Columns::KEY,
            key_selector: CpuAes128Columns::KEY_SELECTOR,
            data: CpuAes128Columns::DATA,
            selector: CpuAes128Columns::SELECTOR,
        },
        link: AesRound128Air::LINK_BUS_ID,
        key: AesRound128Air::KEY_BUS_ID,
        keys: calls.map(|call| call.key.to_vec()),
        whitened,
        ciphers,
        blocks: Box::new(move |order| {
            aes.generate_traces(&order.map(|k| served[k].clone()))
                .unwrap()
        }),
    }
}

fn aes256() -> Level {
    let air = make_program_256(AES_ROWS, 2);

    let calls = [
        (FIPS256_KEY, FIPS256_PLAIN),
        (flip_first(FIPS256_KEY), flip_first(FIPS256_PLAIN)),
    ]
    .map(|(key, plaintext)| Aes256Call {
        key,
        plaintext,
        round_keys: expand_key_256(&key),
    });

    let honest = air.aes.generate_traces(&calls).unwrap();

    let whitened = calls
        .clone()
        .map(|call| core::array::from_fn(|j| call.plaintext[j] ^ call.round_keys[0][j]));

    let ciphers = [0, 1].map(|k| output_state(&honest[0], k, AesRound256Air::BLOCK_ROWS));

    let aes = air.aes.clone();
    let served = calls.clone();

    Level {
        program: air.program,
        host: Host {
            layout: CpuAes256Columns::build_layout(),
            key: CpuAes256Columns::KEY,
            key_selector: CpuAes256Columns::KEY_SELECTOR,
            data: CpuAes256Columns::DATA,
            selector: CpuAes256Columns::SELECTOR,
        },
        link: AesRound256Air::LINK_BUS_ID,
        key: AesRound256Air::KEY_BUS_ID,
        keys: calls.map(|call| call.key.to_vec()),
        whitened,
        ciphers,
        blocks: Box::new(move |order| {
            aes.generate_traces(&order.map(|k| served[k].clone()))
                .unwrap()
        }),
    }
}

fn flip_first<const N: usize>(mut bytes: [u8; N]) -> [u8; N] {
    bytes[0] ^= 0x80;

    bytes
}

fn output_state(aes: &ColumnTrace, block: usize, block_rows: usize) -> [u8; 16] {
    let row = block * block_rows + block_rows - 1;

    core::array::from_fn(|j| b8_at(aes, PhysAes128Columns::P_STATE_IN + j, row))
}

fn host_trace(level: &Level, claims: [Claim; 2]) -> ColumnTrace {
    let host = &level.host;
    let num_vars = CPU_ROWS.trailing_zeros() as usize;

    let mut tb = TraceBuilder::new(&host.layout, num_vars).unwrap();

    for (k, claim) in claims.iter().enumerate() {
        let (in_row, out_row) = (2 * k, 2 * k + 1);

        for j in 0..16 {
            tb.set_b8(
                host.data + j,
                in_row,
                Block8(level.whitened[claim.input][j]),
            )
            .unwrap();
            tb.set_b8(
                host.data + j,
                out_row,
                Block8(level.ciphers[claim.output][j]),
            )
            .unwrap();
        }

        for (j, &byte) in level.keys[claim.key].iter().enumerate() {
            tb.set_b8(host.key + j, in_row, Block8(byte)).unwrap();
        }

        tb.set_bit(host.selector, in_row, Bit::ONE).unwrap();
        tb.set_bit(host.selector, out_row, Bit::ONE).unwrap();
        tb.set_bit(host.key_selector, in_row, Bit::ONE).unwrap();
    }

    tb.build()
}

fn honest(level: &Level) -> (ColumnTrace, Vec<ColumnTrace>) {
    (
        host_trace(level, claims([0, 1], [0, 1], [0, 1])),
        (level.blocks)([0, 1]),
    )
}

fn claims(inputs: [usize; 2], outputs: [usize; 2], keys: [usize; 2]) -> [Claim; 2] {
    [0, 1].map(|k| Claim {
        input: inputs[k],
        output: outputs[k],
        key: keys[k],
    })
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn honest_calls_verify() {
    for level in [aes128(), aes256()] {
        let (cpu, blocks) = honest(&level);

        assert_verifies(&level.program, cpu, blocks);
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn traded_ciphertexts_rejected() {
    for level in [aes128(), aes256()] {
        assert_ne!(level.ciphers[0], level.ciphers[1]);

        assert_eq!(level.whitened[0], level.whitened[1]);

        for (block_order, bus) in [([0, 1], level.link), ([1, 0], level.key)] {
            let forged = (
                host_trace(&level, claims([0, 1], [1, 0], [0, 1])),
                (level.blocks)(block_order),
            );
            let fault = Fault {
                buses: &[bus],
                ..Fault::NONE
            };

            assert_rejected(&level.program, honest(&level), forged, fault);
        }
    }
}

/// The host's key emits hold the blocks' multiset;
/// rank 0 carries the other call's key, or in the
/// reversed block order the other call's data.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn traded_keys_rejected() {
    for level in [aes128(), aes256()] {
        assert_ne!(level.keys[0], level.keys[1]);

        for (block_order, bus) in [([0, 1], level.key), ([1, 0], level.link)] {
            let forged = (
                host_trace(&level, claims([0, 1], [0, 1], [1, 0])),
                (level.blocks)(block_order),
            );
            let fault = Fault {
                buses: &[bus],
                ..Fault::NONE
            };

            assert_rejected(&level.program, honest(&level), forged, fault);
        }
    }
}
