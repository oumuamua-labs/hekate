// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

#[path = "common/mod.rs"]
mod common;

use std::time::Instant;

use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::Block128;
use hekate_core::config::Config;
use hekate_gadgets::chiplets::bignum::modexp::LIMBS32;
use hekate_program::Air;
use hekate_prover_sys::prove;
use hekate_rsa::statement::Pkcs1Statement;
use hekate_sdk::preflight::preflight;
use hekate_sha2::pad_message;
use hekate_verifier::HekateVerifier;
use rand::{TryRngCore, rngs::OsRng};

type F = Block128;
type H = DefaultHasher;

const SHA_ROWS: usize = 512;
const CPU_ROWS: usize = 512;

/// A real RSA-2048 key and PKCS#1 v1.5 signature over
/// `(0..200u8)`, from CPython (`genkey.py`, seed 81030311).
#[rustfmt::skip]
const N: [u32; LIMBS32] = [
    0x3a4cd3af, 0x66d45f4b, 0x3f3be64e, 0xb2e418a0,
    0x742ea128, 0x27e345b9, 0xb108f0a3, 0xa6f8429a,
    0xc72689b4, 0x4f8f7e38, 0xce80a527, 0x87411f36,
    0xcd98948c, 0x519ea53b, 0x22687894, 0x0d4d92f2,
    0xf08075d6, 0xcaf97b9b, 0xdce5fffe, 0x4acd311e,
    0x99604ca5, 0xf033bd1e, 0x7bd5b173, 0x62685cef,
    0x7a14f624, 0x8124d7ad, 0xd86c706c, 0x3c5cf329,
    0x0326fe04, 0xa25d9739, 0xb32f3618, 0x62616eef,
    0x0be6d55d, 0x798e6118, 0x7c955161, 0x96e68b02,
    0x9c992aa1, 0xe26db0b1, 0xabcb78c9, 0x8902c9c6,
    0xc69ebade, 0x34836040, 0xb993f3f6, 0x61c9132d,
    0x36f6db51, 0x513061eb, 0xd14ee03a, 0x730ec5b8,
    0x2416eb11, 0x6103a696, 0xf50eccc4, 0x52d8f47f,
    0xbd54d737, 0x8933ec2a, 0x052eeb80, 0x86caea5c,
    0x042752d0, 0x75c552d8, 0x5646de60, 0xf0c63af4,
    0xde43815a, 0x4823c34e, 0x7e65c266, 0xd116244d,
];

#[rustfmt::skip]
const S: [u32; LIMBS32] = [
    0x7897deff, 0x54c0389d, 0x6369e349, 0x1d7ff0df,
    0x41b31683, 0x993e97c4, 0x5fe7fabc, 0x2ae01b48,
    0xbcce59d5, 0xe7e1adcb, 0xd5bd2f3a, 0xf1b9a348,
    0xd4919224, 0xe507d801, 0x60fcca30, 0x459ba412,
    0x206c7dcd, 0xae3fb65d, 0x607ae10d, 0x0ef5293c,
    0x1b9f6745, 0x3a114648, 0x671f96db, 0xb341eab5,
    0x98cd1f12, 0x8960b317, 0x0c67d850, 0x0dfc05c8,
    0x531c86be, 0xe8ccea73, 0xff2f2e3a, 0x5aef901c,
    0xd99089a1, 0xd2ffe09c, 0x78aaf7aa, 0x40a51646,
    0x9cd6c6e1, 0x5a8cd9a3, 0xc2b15955, 0xcbb79a47,
    0x1837ca53, 0xaf04db30, 0xf580b968, 0xfc839ed3,
    0x3061abc2, 0x84475224, 0x7218c6fb, 0x1fabca0e,
    0x58467859, 0x726b3d7f, 0xc33b246f, 0xc660be0d,
    0x230dd49c, 0x25e755a6, 0xca114abf, 0x5cdc2d7e,
    0x0e97d198, 0x77101fcc, 0xcf5a2b1f, 0x53e60c09,
    0x5e0682b7, 0x1ed8bd9b, 0x6c7f6532, 0x9f1e9450,
];

fn rounds_per_row() -> usize {
    std::env::var("HEKATE_ROUNDS_PER_ROW")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2)
}

fn main() {
    common::init("RSA-2048 PKCS#1 v1.5 + SHA-256");

    let config = Config {
        zero_knowledge: common::zero_knowledge(),
        ..Config::default()
    };

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    let message: Vec<u8> = (0..200u8).collect();
    let num_blocks = pad_message(&message).len();
    let rounds_per_row = rounds_per_row();

    let statement = common::phase("Statement build", || {
        Pkcs1Statement::new(num_blocks, rounds_per_row, SHA_ROWS, CPU_ROWS)
            .expect("statement build")
    });

    let program = statement.program();

    println!(
        "Message: {} bytes ({} SHA-256 blocks, {} rounds per row)",
        message.len(),
        num_blocks,
        rounds_per_row
    );
    println!("Zero-knowledge: {}", config.zero_knowledge);
    println!(
        "Host table: 2^{} x {} columns, {} public inputs",
        CPU_ROWS.trailing_zeros(),
        program.num_columns(),
        LIMBS32
    );

    let witness = common::phase("Trace Generation", || {
        statement.witness(&message, &N, &S).expect("witness")
    });

    let instance = statement.instance(&N);

    let report = common::phase("Preflight", || {
        preflight(program, &instance, &witness).expect("preflight")
    });

    assert!(
        report.is_clean(),
        "preflight dirty: {} constraint, {} boundary violations",
        report.constraint_violations.len(),
        report.boundary_violations.len()
    );

    let prove_start = Instant::now();
    let proof = common::phase("Proving", || {
        prove(
            b"RsaPkcs1",
            program,
            &instance,
            &witness,
            &config,
            blinding_seed,
            None,
        )
        .expect("Prover failed")
    });

    println!(
        "RSA-2048 PKCS#1 v1.5 + SHA-256: {:.0} ms prove",
        prove_start.elapsed().as_secs_f64() * 1e3
    );

    common::proof_breakdown(&proof);

    let mut verifier_transcript = Transcript::<H>::new(b"RsaPkcs1");
    let pinned_id = common::audited_id::<F, _>(program);

    let is_valid = common::phase_with_mem("Verifying", || {
        HekateVerifier::<F, H>::verify(
            &pinned_id,
            program,
            &instance,
            &proof,
            &mut verifier_transcript,
            &config,
        )
        .expect("Verifier failed")
    });

    common::result(is_valid);
}
