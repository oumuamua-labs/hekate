// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::config::Config;
use hekate_core::errors::Error;
use hekate_core::trace::{ColumnType, TraceBuilder};
use hekate_crypto::DefaultHasher;
use hekate_crypto::transcript::Transcript;
use hekate_math::{Bit, Block32, Block128, TowerField};
use hekate_pqc::mlkem::{
    self, Forgery, MLKEM_DATA_BUS_ID, MlKemCall, MlKemChiplet, MlKemInput, MlKemOutput,
    MlKemParams, MlKemWitness,
};
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::digest::program_id;
use hekate_program::{FixedShape, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_sdk::preflight::{PreflightReport, preflight};
use hekate_verifier::HekateVerifier;
use ml_kem::{B32, Decapsulate, DecapsulationKey, KeyExport, MlKem512, MlKem768, MlKem1024, Seed};
use sha3::{Digest, Sha3_256};

type F = Block128;
type H = DefaultHasher;

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

impl Proof {
    fn report(&self) -> PreflightReport<F> {
        preflight(&self.program, &self.instance, &self.witness).unwrap()
    }

    fn accepted(&self, zero_knowledge: bool) -> bool {
        let config = Config {
            zero_knowledge,
            ..Config::prod()
        };

        let proof = prove(
            b"MlKemPipeline",
            &self.program,
            &self.instance,
            &self.witness,
            &config,
            [7; 32],
            None,
        )
        .unwrap();

        let mut transcript = Transcript::<H>::new(b"MlKemPipeline");

        HekateVerifier::<F, H>::verify(
            &program_id(&self.program).unwrap(),
            &self.program,
            &self.instance,
            &proof,
            &mut transcript,
            &config,
        )
        .unwrap_or(false)
    }
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

    assert!(proof.report().is_clean());
    assert!(proof.accepted(false));
    assert!(proof.accepted(true));
}

fn forged(params: MlKemParams, input: MlKemInput<'_>, forgeries: &[Forgery]) -> Proof {
    let call = match input {
        MlKemInput::KeyGen { .. } => MlKemCall::KeyGen,
        MlKemInput::Encaps { .. } => MlKemCall::Encaps,
        MlKemInput::Decaps { .. } => MlKemCall::Decaps,
    };

    let pipeline = MlKemChiplet::<F>::new(params, &[call]).unwrap();
    let traced = pipeline.trace_forged(&[input], forgeries).unwrap();

    host(&pipeline, traced)
}

fn assert_breaks_only_bus(proof: &Proof, bus: &str) {
    let report = proof.report();

    assert!(report.constraint_violations.is_empty());
    assert!(report.boundary_violations.is_empty());
    assert!(report.fixed_column_violations.is_empty());
    assert!(!report.bus_diagnostics.is_empty());

    for d in &report.bus_diagnostics {
        assert_eq!(d.bus_id, bus);
    }

    assert!(!proof.accepted(false));
    assert!(!proof.accepted(true));
}

fn assert_breaks_only_constraint(proof: &Proof, label: &str) {
    let report = proof.report();

    assert!(report.boundary_violations.is_empty());
    assert!(report.fixed_column_violations.is_empty());
    assert!(report.bus_diagnostics.is_empty());
    assert!(!report.constraint_violations.is_empty());

    for v in &report.constraint_violations {
        assert_eq!(v.label, Some(label));
    }

    assert!(!proof.accepted(false));
    assert!(!proof.accepted(true));
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

    let proof = forged(
        MlKemParams::ML_KEM_512,
        MlKemInput::Encaps { ek: &v.ek, m: &v.m },
        &[Forgery::Message { call: 0, m }],
    );

    assert_breaks_only_bus(&proof, "mlkem::word");
}

#[test]
fn key_hash_other_than_h_of_ek_breaks_word_bus() {
    let v = vectors!(MlKem512, 0x32);

    let mut h: [u8; 32] = Sha3_256::digest(&v.ek).into();
    h[5] ^= 0x10;

    let proof = forged(
        MlKemParams::ML_KEM_512,
        MlKemInput::Encaps { ek: &v.ek, m: &v.m },
        &[Forgery::KeyHash { call: 0, h }],
    );

    assert_breaks_only_bus(&proof, "mlkem::word");
}

#[test]
fn non_canonical_encapsulation_key_breaks_codec_modulus() {
    let mut v = vectors!(MlKem512, 0x33);

    v.ek[1] |= 0x0f;
    v.ek[2] = 0xff;

    let proof = forged(
        MlKemParams::ML_KEM_512,
        MlKemInput::Encaps { ek: &v.ek, m: &v.m },
        &[],
    );

    assert_breaks_only_constraint(&proof, "codec_kem_modulus");
}

#[test]
fn decapsulation_key_with_foreign_hash_breaks_word_bus() {
    let params = MlKemParams::ML_KEM_512;
    let v = vectors!(MlKem512, 0x34);

    let mut dk = expanded_dk(params, &v);
    let at = dk.len() - 64;
    dk[at] ^= 1;

    let proof = forged(params, MlKemInput::Decaps { dk: &dk, c: &v.c }, &[]);

    assert_breaks_only_bus(&proof, "mlkem::word");
}

#[test]
fn codec_decoding_another_dk_pke_breaks_word_bus() {
    let params = MlKemParams::ML_KEM_512;
    let v = vectors!(MlKem512, 0x35);
    let dk = expanded_dk(params, &v);

    let proof = forged(
        params,
        MlKemInput::Decaps { dk: &dk, c: &v.c },
        &[Forgery::DkPke {
            call: 0,
            word: 3,
            mask: 1,
        }],
    );

    assert_breaks_only_bus(&proof, "mlkem::word");
}

#[test]
fn sampler_expanding_another_rho_breaks_lane_bus() {
    let params = MlKemParams::ML_KEM_512;
    let v = vectors!(MlKem512, 0x36);
    let dk = expanded_dk(params, &v);

    let mut rho: [u8; 32] = v.ek[v.ek.len() - 32..].try_into().unwrap();
    rho[0] ^= 1;

    let proof = forged(
        params,
        MlKemInput::Decaps { dk: &dk, c: &v.c },
        &[Forgery::Rho { call: 0, rho }],
    );

    assert_breaks_only_bus(&proof, "mlkem::lane");
}

#[test]
fn keygen_sampler_expanding_another_rho_breaks_lane_bus() {
    let v = vectors!(MlKem512, 0x37);

    let mut rho: [u8; 32] = v.ek[v.ek.len() - 32..].try_into().unwrap();
    rho[0] ^= 1;

    let proof = forged(
        MlKemParams::ML_KEM_512,
        MlKemInput::KeyGen { d: &v.d },
        &[Forgery::Rho { call: 0, rho }],
    );

    assert_breaks_only_bus(&proof, "mlkem::lane");
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
