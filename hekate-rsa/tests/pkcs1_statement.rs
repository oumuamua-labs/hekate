// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::config::Config;
use hekate_core::trace::ColumnTrace;
use hekate_crypto::DefaultHasher;
use hekate_crypto::transcript::Transcript;
use hekate_gadgets::Modexp;
use hekate_gadgets::chiplets::bignum::modexp::LIMBS32;
use hekate_math::Block128;
use hekate_program::ProgramWitness;
use hekate_program::digest::program_id;
use hekate_prover_sys::prove;
use hekate_rsa::statement::Pkcs1Statement;
use hekate_rsa::{encode_sha256, padding_limbs};
use hekate_scribble::{MutationKind, ScribbleConfig, assert_all_caught_all_targets};
use hekate_sdk::preflight;
use hekate_sha2::sha256_words;
use hekate_verifier::HekateVerifier;

type F = Block128;
type H = DefaultHasher;

const NUM_BLOCKS: usize = 4;
const ROUNDS_PER_ROW: usize = 2;
const SHA_ROWS: usize = 128;
const CPU_ROWS: usize = 256;

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

fn message() -> Vec<u8> {
    (0..200u8).collect()
}

fn statement() -> Pkcs1Statement {
    Pkcs1Statement::new(NUM_BLOCKS, ROUNDS_PER_ROW, SHA_ROWS, CPU_ROWS).unwrap()
}

fn prove_and_verify(
    st: &Pkcs1Statement,
    witness: ProgramWitness<F, ColumnTrace>,
) -> Result<bool, String> {
    let instance = st.instance(&N);

    let report =
        preflight(st.program(), &instance, &witness).map_err(|e| format!("preflight: {e:?}"))?;

    if !report.is_clean() {
        for v in &report.constraint_violations {
            eprintln!("constraint={} row={}", v.constraint_idx, v.row_idx);
        }

        return Err("preflight violations".into());
    }

    verdict_without_preflight(st, witness)
}

fn verdict_without_preflight(
    st: &Pkcs1Statement,
    witness: ProgramWitness<F, ColumnTrace>,
) -> Result<bool, String> {
    let instance = st.instance(&N);

    let config = Config {
        zero_knowledge: true,
        ..Config::dev()
    };

    let proof = prove(
        b"RsaPkcs1_E2E",
        st.program(),
        &instance,
        &witness,
        &config,
        [11u8; 32],
        None,
    )
    .map_err(|e| format!("prover: {e:?}"))?;

    let mut vt = Transcript::<H>::new(b"RsaPkcs1_E2E");
    let pinned_id = program_id(st.program()).unwrap();

    HekateVerifier::<F, H>::verify(
        &pinned_id,
        st.program(),
        &instance,
        &proof,
        &mut vt,
        &config,
    )
    .map_err(|e| format!("verifier: {e:?}"))
}

fn assert_rejected_at_compare(st: &Pkcs1Statement, witness: ProgramWitness<F, ColumnTrace>) {
    let instance = st.instance(&N);
    let report = preflight(st.program(), &instance, &witness).expect("preflight");
    let verify_row = NUM_BLOCKS - 1;

    assert!(report.boundary_violations.is_empty());
    assert!(report.bus_diagnostics.iter().all(|d| !d.has_failures()));
    assert!(!report.constraint_violations.is_empty());
    assert!(
        report
            .constraint_violations
            .iter()
            .all(|v| v.row_idx == verify_row)
    );

    assert_eq!(verdict_without_preflight(st, witness), Ok(false));
}

#[test]
fn modexp_reproduces_signed_encoding() {
    let em = Modexp::new(&N, &S).unwrap();
    let expected = encode_sha256(&sha256_words(&message()));

    assert_eq!(em.result(), &expected);
}

#[test]
fn encoding_padding_matches_signature() {
    let em = encode_sha256(&sha256_words(&message()));
    let padding = padding_limbs();

    assert_eq!(em[LIMBS32 - 1], 0x0001_ffff);

    for j in 8..LIMBS32 {
        assert_eq!(em[j], padding[j], "limb {j}");
    }
}

#[test]
fn honest_proof_verifies() {
    let st = statement();
    let witness = st.witness(&message(), &N, &S).unwrap();

    match prove_and_verify(&st, witness) {
        Ok(true) => {}
        Ok(false) => panic!("verifier rejected an honest proof"),
        Err(e) => panic!("{e}"),
    }
}

#[test]
fn forged_signature_rejected() {
    let st = statement();

    let mut forged = S;
    forged[0] ^= 1;

    let witness = st.witness(&message(), &N, &forged).unwrap();

    assert_rejected_at_compare(&st, witness);
}

#[test]
fn altered_message_rejected() {
    let st = statement();

    let mut message = message();
    message[7] ^= 1;

    let witness = st.witness(&message, &N, &S).unwrap();

    assert_rejected_at_compare(&st, witness);
}

#[test]
fn scribble_flip_selector_caught() {
    let st = statement();

    assert_all_caught_all_targets(
        st.program(),
        &st.instance(&N),
        &st.witness(&message(), &N, &S).unwrap(),
        ScribbleConfig::default()
            .cases(48)
            .mutations([MutationKind::FlipSelector]),
    );
}
