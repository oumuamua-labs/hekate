// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Two AES calls trading ciphertexts or keys,
//! rejected in every block order at both key sizes.

mod common;

use common::{
    CPU_ROWS, F, FIPS128_KEY, FIPS128_PLAIN, FIPS256_KEY, FIPS256_PLAIN, H, b8_at,
    make_program_128, make_program_256,
};
use hekate_aes::trace::{Aes128Call, Aes256Call, expand_key, expand_key_256};
use hekate_aes::{
    AesRound128Air, AesRound256Air, CpuAes128Columns, CpuAes256Columns, PhysAes128Columns,
};
use hekate_core::config::Config;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate_crypto::transcript::Transcript;
use hekate_math::{Bit, Block8, TowerField};
use hekate_program::circuit::CircuitProgram;
use hekate_program::digest::program_id;
use hekate_program::{ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;

const LABEL: &[u8] = b"AES_Call_Pairing";
const AES_ROWS: usize = 32;
const BLOCK_ORDERS: [[usize; 2]; 2] = [[0, 1], [1, 0]];

struct Level {
    name: &'static str,
    program: CircuitProgram<F>,
    host: Host,
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
        name: "aes128",
        program: air.program,
        host: Host {
            layout: CpuAes128Columns::build_layout(),
            key: CpuAes128Columns::KEY,
            key_selector: CpuAes128Columns::KEY_SELECTOR,
            data: CpuAes128Columns::DATA,
            selector: CpuAes128Columns::SELECTOR,
        },
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
        name: "aes256",
        program: air.program,
        host: Host {
            layout: CpuAes256Columns::build_layout(),
            key: CpuAes256Columns::KEY,
            key_selector: CpuAes256Columns::KEY_SELECTOR,
            data: CpuAes256Columns::DATA,
            selector: CpuAes256Columns::SELECTOR,
        },
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

fn verdicts(level: &Level, claims: [Claim; 2], block_order: [usize; 2]) -> [bool; 2] {
    let instance = ProgramInstance::new(CPU_ROWS, vec![]);
    let witness =
        ProgramWitness::new(host_trace(level, claims)).with_chiplets((level.blocks)(block_order));

    [false, true].map(|zero_knowledge| {
        let config = Config {
            zero_knowledge,
            ..Config::dev()
        };

        let proof = prove(
            LABEL,
            &level.program,
            &instance,
            &witness,
            &config,
            [0x3Cu8; 32],
            None,
        )
        .expect("the prover proves the witness it is handed");

        let mut vt = Transcript::<H>::new(LABEL);
        let pinned_id = program_id(&level.program).unwrap();

        HekateVerifier::<F, H>::verify(
            &pinned_id,
            &level.program,
            &instance,
            &proof,
            &mut vt,
            &config,
        )
        .unwrap_or(false)
    })
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
        assert_eq!(
            verdicts(&level, claims([0, 1], [0, 1], [0, 1]), [0, 1]),
            [true, true],
            "{}: the harness itself is broken, the rejections below are unattributable",
            level.name
        );
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn traded_ciphertexts_rejected() {
    for level in [aes128(), aes256()] {
        assert_ne!(level.ciphers[0], level.ciphers[1]);

        for block_order in BLOCK_ORDERS {
            assert_eq!(
                verdicts(&level, claims([0, 1], [1, 0], [0, 1]), block_order),
                [false, false],
                "{} block order {block_order:?}",
                level.name
            );
        }
    }
}

/// The host's key emits hold the blocks' multiset;
/// rank 0 carries the other call's key.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn traded_keys_rejected() {
    for level in [aes128(), aes256()] {
        assert_ne!(level.keys[0], level.keys[1]);

        for block_order in BLOCK_ORDERS {
            assert_eq!(
                verdicts(&level, claims([0, 1], [0, 1], [1, 0]), block_order),
                [false, false],
                "{} block order {block_order:?}",
                level.name
            );
        }
    }
}
