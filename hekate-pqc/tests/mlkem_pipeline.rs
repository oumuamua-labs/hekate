// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::config::Config;
use hekate_core::errors::Error;
use hekate_core::trace::{ColumnType, TraceBuilder};
use hekate_math::{Bit, Block32, Block256, TowerField};
use hekate_pqc::mlkem::{
    self, Forgery, MLKEM_DATA_BUS_ID, MlKemCall, MlKemChiplet, MlKemInput, MlKemOutput,
    MlKemParams, MlKemWitness,
};
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::{FixedShape, ProgramInstance, ProgramWitness};
use hekate_sdk::preflight::TableId;
use hekate_testkit::fault::Fault;
use hekate_testkit::verdict::{assert_rejected, assert_verifies};
use ml_kem::{B32, Decapsulate, DecapsulationKey, KeyExport, MlKem512, MlKem768, MlKem1024, Seed};
use sha3::{Digest, Sha3_256};

type F = Block256;

const CODEC: TableId = TableId::Chiplet(3);

const CALLS: [MlKemCall; 4] = [
    MlKemCall::KeyGen,
    MlKemCall::Encaps,
    MlKemCall::Decaps,
    MlKemCall::Decaps,
];

struct Vectors {
    d: [u8; 32],
    z: [u8; 32],
    m: [u8; 32],
    ek: Vec<u8>,
    c: Vec<u8>,
    key: [u8; 32],
    tampered: Vec<u8>,
    rejected: [u8; 32],
}

struct Proof {
    program: CircuitProgram<F>,
    instance: ProgramInstance<F>,
    witness: ProgramWitness<F>,
}

macro_rules! vectors {
    ($kem:ty, $seed:expr) => {{
        let (d, z, m) = ([$seed; 32], [$seed ^ 0x3c; 32], [$seed ^ 0xa5; 32]);

        let mut seed = [0u8; 64];
        seed[..32].copy_from_slice(&d);
        seed[32..].copy_from_slice(&z);

        let dk = DecapsulationKey::<$kem>::from_seed(Seed::from(seed));
        let ek = dk.encapsulation_key();

        let (c, key) = ek.encapsulate_deterministic(&B32::from(m));

        let mut tampered = c.clone();
        tampered[0] ^= 1;

        Vectors {
            d,
            z,
            m,
            ek: ek.to_bytes().to_vec(),
            c: c.to_vec(),
            key: key.into(),
            rejected: dk.decapsulate(&tampered).into(),
            tampered: tampered.to_vec(),
        }
    }};
}

fn honest(params: MlKemParams, v: &Vectors) {
    let dk = expanded_dk(params, v);

    let inputs = [
        MlKemInput::KeyGen { d: &v.d },
        MlKemInput::Encaps { ek: &v.ek, m: &v.m },
        MlKemInput::Decaps { dk: &dk, c: &v.c },
        MlKemInput::Decaps {
            dk: &dk,
            c: &v.tampered,
        },
    ];

    let pipeline = MlKemChiplet::<F>::new(params, &CALLS).unwrap();
    let traced = pipeline.trace(&inputs).unwrap();

    let MlKemOutput::KeyGen { ek, h, .. } = &traced.outputs[0] else {
        unreachable!()
    };

    assert_eq!(*ek, v.ek);
    assert_eq!(h.as_slice(), Sha3_256::digest(ek).as_slice());

    let MlKemOutput::Encaps { key, c } = &traced.outputs[1] else {
        unreachable!()
    };

    assert_eq!(*key, v.key);
    assert_eq!(*c, v.c);

    let MlKemOutput::Decaps { key, valid } = &traced.outputs[2] else {
        unreachable!()
    };

    assert_eq!(*key, v.key);
    assert!(*valid);

    let MlKemOutput::Decaps { key, valid } = &traced.outputs[3] else {
        unreachable!()
    };

    assert_eq!(*key, v.rejected);
    assert!(!*valid);

    let proof = host(&pipeline, traced);

    assert_verifies(
        &proof.program,
        (&proof.instance, &proof.witness),
        &Config::prod(),
    );
}

fn honest_call(params: MlKemParams, input: MlKemInput<'_>) -> Proof {
    let pipeline = MlKemChiplet::<F>::new(params, &[kind(&input)]).unwrap();
    let traced = pipeline.trace(&[input]).unwrap();

    host(&pipeline, traced)
}

fn forged(params: MlKemParams, input: MlKemInput<'_>, forgeries: &[Forgery]) -> Proof {
    let pipeline = MlKemChiplet::<F>::new(params, &[kind(&input)]).unwrap();
    let traced = pipeline.trace_forged(&[input], forgeries).unwrap();

    host(&pipeline, traced)
}

fn assert_breaks_only_bus(honest: &Proof, forged: &Proof, bus: &str) {
    assert_fault(
        honest,
        forged,
        Fault {
            buses: &[bus],
            ..Fault::NONE
        },
    );
}

fn assert_breaks_only_constraint(honest: &Proof, forged: &Proof, label: &str) {
    assert_fault(
        honest,
        forged,
        Fault {
            constraint_labels: &[(CODEC, label)],
            ..Fault::NONE
        },
    );
}

fn assert_fault(honest: &Proof, forged: &Proof, fault: Fault<'_>) {
    assert_rejected(
        &forged.program,
        (&honest.instance, &honest.witness),
        (&forged.instance, &forged.witness),
        fault,
        &Config::prod(),
    );
}

fn kind(input: &MlKemInput<'_>) -> MlKemCall {
    match input {
        MlKemInput::KeyGen { .. } => MlKemCall::KeyGen,
        MlKemInput::Encaps { .. } => MlKemCall::Encaps,
        MlKemInput::Decaps { .. } => MlKemCall::Decaps,
    }
}

fn expanded_dk(params: MlKemParams, v: &Vectors) -> Vec<u8> {
    let pipeline = MlKemChiplet::<F>::new(params, &[MlKemCall::KeyGen]).unwrap();
    let traced = pipeline.trace(&[MlKemInput::KeyGen { d: &v.d }]).unwrap();

    let MlKemOutput::KeyGen { ek, dk_pke, h } = &traced.outputs[0] else {
        unreachable!()
    };

    [dk_pke.as_slice(), ek, h, &v.z].concat()
}

fn host(pipeline: &MlKemChiplet<F>, traced: MlKemWitness) -> Proof {
    let words = &traced.words;
    let rows = words.len().next_power_of_two();

    let mut cx = Circuit::<F>::new("MlKemHost", rows).unwrap();

    let word = cx.column(ColumnType::B32);
    let sel = cx.column(ColumnType::Bit);

    cx.fix(
        sel,
        FixedShape::Cadence {
            stride: 1,
            count: words.len(),
            origin: 0,
            values: vec![F::ONE],
        },
    );

    cx.call(&mlkem::service(), &[word], sel).unwrap();

    cx.attach_namespaced("mlkem", pipeline.defs().unwrap(), &[MLKEM_DATA_BUS_ID])
        .unwrap();

    let mut tb = TraceBuilder::new(
        &[ColumnType::B32, ColumnType::Bit],
        rows.trailing_zeros() as usize,
    )
    .unwrap();

    for (r, &w) in words.iter().enumerate() {
        tb.set_b32(0, r, Block32::from(w)).unwrap();
        tb.set_bit(1, r, Bit::from(1u8)).unwrap();
    }

    Proof {
        program: cx.compile().unwrap(),
        instance: ProgramInstance::new(rows, Vec::new()),
        witness: ProgramWitness::new(tb.build()).with_chiplets(traced.traces),
    }
}

#[test]
fn ml_kem_512_matches_fips_203_and_proves() {
    honest(MlKemParams::ML_KEM_512, &vectors!(MlKem512, 0x51));
}

#[test]
fn ml_kem_768_matches_fips_203_and_proves() {
    honest(MlKemParams::ML_KEM_768, &vectors!(MlKem768, 0x76));
}

#[test]
fn ml_kem_1024_matches_fips_203_and_proves() {
    honest(MlKemParams::ML_KEM_1024, &vectors!(MlKem1024, 0x10));
}

#[test]
fn non_canonical_encapsulation_key_is_refused() {
    let params = MlKemParams::ML_KEM_512;
    let mut v = vectors!(MlKem512, 0x21);

    v.ek[1] |= 0x0f;
    v.ek[2] = 0xff;

    let pipeline = MlKemChiplet::<F>::new(params, &[MlKemCall::Encaps]).unwrap();

    assert_eq!(
        pipeline
            .trace(&[MlKemInput::Encaps { ek: &v.ek, m: &v.m }])
            .err(),
        Some(Error::Protocol {
            protocol: "mlkem_chiplet",
            message: "encapsulation key fails the FIPS 203 modulus check",
        })
    );
}

#[test]
fn decapsulation_key_with_foreign_hash_is_refused() {
    let params = MlKemParams::ML_KEM_512;
    let v = vectors!(MlKem512, 0x22);

    let mut dk = expanded_dk(params, &v);
    let at = dk.len() - 64;
    dk[at] ^= 1;

    let pipeline = MlKemChiplet::<F>::new(params, &[MlKemCall::Decaps]).unwrap();

    assert_eq!(
        pipeline
            .trace(&[MlKemInput::Decaps { dk: &dk, c: &v.c }])
            .err(),
        Some(Error::Protocol {
            protocol: "mlkem_chiplet",
            message: "decapsulation key fails the FIPS 203 hash check",
        })
    );
}

#[test]
fn key_from_another_message_breaks_word_bus() {
    let v = vectors!(MlKem512, 0x31);

    let mut m = v.m;
    m[0] ^= 1;

    let input = || MlKemInput::Encaps { ek: &v.ek, m: &v.m };

    let proof = forged(
        MlKemParams::ML_KEM_512,
        input(),
        &[Forgery::Message { call: 0, m }],
    );

    assert_breaks_only_bus(
        &honest_call(MlKemParams::ML_KEM_512, input()),
        &proof,
        "mlkem::word",
    );
}

#[test]
fn key_hash_other_than_h_of_ek_breaks_word_bus() {
    let v = vectors!(MlKem512, 0x32);

    let mut h: [u8; 32] = Sha3_256::digest(&v.ek).into();
    h[5] ^= 0x10;

    let input = || MlKemInput::Encaps { ek: &v.ek, m: &v.m };

    let proof = forged(
        MlKemParams::ML_KEM_512,
        input(),
        &[Forgery::KeyHash { call: 0, h }],
    );

    assert_breaks_only_bus(
        &honest_call(MlKemParams::ML_KEM_512, input()),
        &proof,
        "mlkem::word",
    );
}

#[test]
fn non_canonical_encapsulation_key_breaks_codec_modulus() {
    let v = vectors!(MlKem512, 0x33);

    let mut ek = v.ek.clone();
    ek[1] |= 0x0f;
    ek[2] = 0xff;

    let proof = forged(
        MlKemParams::ML_KEM_512,
        MlKemInput::Encaps { ek: &ek, m: &v.m },
        &[],
    );

    let honest = honest_call(
        MlKemParams::ML_KEM_512,
        MlKemInput::Encaps { ek: &v.ek, m: &v.m },
    );

    assert_breaks_only_constraint(&honest, &proof, "codec_kem_modulus");
}

#[test]
fn decapsulation_key_with_foreign_hash_breaks_word_bus() {
    let params = MlKemParams::ML_KEM_512;
    let v = vectors!(MlKem512, 0x34);

    let honest_dk = expanded_dk(params, &v);
    let mut dk = honest_dk.clone();

    let at = dk.len() - 64;
    dk[at] ^= 1;

    let proof = forged(params, MlKemInput::Decaps { dk: &dk, c: &v.c }, &[]);

    let honest = honest_call(
        params,
        MlKemInput::Decaps {
            dk: &honest_dk,
            c: &v.c,
        },
    );

    assert_breaks_only_bus(&honest, &proof, "mlkem::word");
}

#[test]
fn codec_decoding_another_dk_pke_breaks_word_bus() {
    let params = MlKemParams::ML_KEM_512;
    let v = vectors!(MlKem512, 0x35);
    let dk = expanded_dk(params, &v);

    let input = || MlKemInput::Decaps { dk: &dk, c: &v.c };

    let proof = forged(
        params,
        input(),
        &[Forgery::DkPke {
            call: 0,
            word: 3,
            mask: 1,
        }],
    );

    assert_breaks_only_bus(&honest_call(params, input()), &proof, "mlkem::word");
}

#[test]
fn sampler_expanding_another_rho_breaks_lane_bus() {
    let params = MlKemParams::ML_KEM_512;
    let v = vectors!(MlKem512, 0x36);
    let dk = expanded_dk(params, &v);

    let mut rho: [u8; 32] = v.ek[v.ek.len() - 32..].try_into().unwrap();
    rho[0] ^= 1;

    let input = || MlKemInput::Decaps { dk: &dk, c: &v.c };

    let proof = forged(params, input(), &[Forgery::Rho { call: 0, rho }]);

    assert_breaks_only_bus(&honest_call(params, input()), &proof, "mlkem::lane");
}

#[test]
fn keygen_sampler_expanding_another_rho_breaks_lane_bus() {
    let v = vectors!(MlKem512, 0x37);

    let mut rho: [u8; 32] = v.ek[v.ek.len() - 32..].try_into().unwrap();
    rho[0] ^= 1;

    let input = || MlKemInput::KeyGen { d: &v.d };

    let proof = forged(
        MlKemParams::ML_KEM_512,
        input(),
        &[Forgery::Rho { call: 0, rho }],
    );

    assert_breaks_only_bus(
        &honest_call(MlKemParams::ML_KEM_512, input()),
        &proof,
        "mlkem::lane",
    );
}

#[test]
fn call_counts_outside_one_to_256_are_refused() {
    let refused = Some(Error::Protocol {
        protocol: "mlkem_chiplet",
        message: "mlkem pipeline serves 1 to 256 calls",
    });

    assert_eq!(
        MlKemChiplet::<F>::new(MlKemParams::ML_KEM_512, &[]).err(),
        refused
    );
    assert_eq!(
        MlKemChiplet::<F>::new(MlKemParams::ML_KEM_512, &[MlKemCall::Encaps; 257]).err(),
        refused
    );
}
