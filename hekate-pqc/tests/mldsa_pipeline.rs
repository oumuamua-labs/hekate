// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::config::Config;
use hekate_core::errors::{self, Error};
use hekate_core::trace::{ColumnType, TraceBuilder};
use hekate_crypto::DefaultHasher;
use hekate_crypto::transcript::Transcript;
use hekate_keccak::shake256;
use hekate_math::{Bit, Block32, Block128, TowerField};
use hekate_pqc::mldsa::{
    self, Forgery, MLDSA_DATA_BUS_ID, MlDsaChiplet, MlDsaInput, MlDsaOutput, MlDsaParams,
    MlDsaWitness,
};
use hekate_pqc::ntt::NttParams;
use hekate_pqc::wiring::N;
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::digest::program_id;
use hekate_program::{FixedShape, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_sdk::preflight::{PreflightReport, preflight};
use hekate_verifier::HekateVerifier;
use ml_dsa::{B32, MlDsa44, MlDsa65, MlDsa87, SigningKey};

type F = Block128;
type H = DefaultHasher;

struct Signed {
    pk: Vec<u8>,
    message: Vec<u8>,
    signature: Vec<u8>,
}

impl Signed {
    fn input(&self) -> MlDsaInput<'_> {
        MlDsaInput {
            pk: &self.pk,
            message: &self.message,
            signature: &self.signature,
        }
    }
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
            b"MlDsaPipeline",
            &self.program,
            &self.instance,
            &self.witness,
            &config,
            [7; 32],
            None,
        )
        .unwrap();

        let mut transcript = Transcript::<H>::new(b"MlDsaPipeline");

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

fn honest<P: ml_dsa::MlDsaParams>(params: MlDsaParams, messages: &[&[u8]]) {
    let calls: Vec<Signed> = messages
        .iter()
        .zip(1u8..)
        .map(|(message, seed)| sign::<P>(seed, message))
        .collect();

    let (proof, outputs) = run(params, &calls);

    for (signed, output) in calls.iter().zip(&outputs) {
        assert!(*output == reference(signed));
    }

    assert!(proof.report().is_clean());
    assert!(proof.accepted(false));
    assert!(proof.accepted(true));
}

fn run(params: MlDsaParams, calls: &[Signed]) -> (Proof, Vec<MlDsaOutput>) {
    let pipeline = pipeline(params, calls);

    let inputs: Vec<MlDsaInput<'_>> = calls.iter().map(Signed::input).collect();
    let traced = pipeline.trace(&inputs).unwrap();
    let outputs = traced.outputs.to_vec();

    (host(&pipeline, traced), outputs)
}

fn forged(
    params: MlDsaParams,
    calls: &[Signed],
    forge: impl FnOnce(&mut Forgery<'_>) -> errors::Result<()>,
) -> Proof {
    let pipeline = pipeline(params, calls);

    let inputs: Vec<MlDsaInput<'_>> = calls.iter().map(Signed::input).collect();
    let traced = pipeline.trace_forged(&inputs, forge).unwrap();

    host(&pipeline, traced)
}

fn assert_rejected_at(proof: &Proof, bus: &str) {
    let report = proof.report();

    assert!(
        report
            .bus_diagnostics
            .iter()
            .any(|d| d.bus_id == bus && d.has_failures())
    );
    assert!(!proof.accepted(false));
    assert!(!proof.accepted(true));
}

fn assert_rejected_only_at(proof: &Proof, bus: &str) {
    let report = proof.report();

    assert!(report.constraint_violations.is_empty());
    assert!(report.fixed_column_violations.is_empty());
    assert!(report.boundary_violations.is_empty());

    let failing: Vec<&str> = report
        .bus_diagnostics
        .iter()
        .filter(|d| d.has_failures())
        .map(|d| d.bus_id.as_str())
        .collect();

    assert!(
        !failing.is_empty() && failing.iter().all(|&b| b == bus),
        "{failing:?}"
    );
    assert!(!proof.accepted(false));
    assert!(!proof.accepted(true));
}

fn assert_breaks_only(proof: &Proof, label: &str) {
    let report = proof.report();

    assert!(report.fixed_column_violations.is_empty());
    assert!(report.boundary_violations.is_empty());
    assert!(report.bus_diagnostics.iter().all(|d| !d.has_failures()));
    assert!(!report.constraint_violations.is_empty());
    assert!(
        report
            .constraint_violations
            .iter()
            .all(|v| v.label == Some(label))
    );
    assert!(!proof.accepted(false));
    assert!(!proof.accepted(true));
}

fn z_hat(params: MlDsaParams, signature: &[u8], j: usize) -> [u32; N] {
    let bits = params.z_bits();
    let start = params.lambda() / 4 + j * 32 * bits;
    let bytes = &signature[start..start + 32 * bits];

    let z: [u32; N] = core::array::from_fn(|m| {
        let y = (0..bits).fold(0u32, |acc, b| {
            let bit = m * bits + b;

            acc | (((bytes[bit / 8] >> (bit % 8)) & 1) as u32) << b
        });

        (params.gamma1() + mldsa::Q - y) % mldsa::Q
    });

    NttParams::ML_DSA.ntt(&z)
}

fn pipeline(params: MlDsaParams, calls: &[Signed]) -> MlDsaChiplet<F> {
    let lens: Vec<usize> = calls.iter().map(|s| s.message.len()).collect();

    MlDsaChiplet::<F>::new(params, &lens).unwrap()
}

fn host(pipeline: &MlDsaChiplet<F>, traced: MlDsaWitness) -> Proof {
    let words = &traced.words;
    let rows = words.len().next_power_of_two();

    let mut cx = Circuit::<F>::new("MlDsaHost", rows).unwrap();

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
    cx.call(&mldsa::service(), &[word], sel).unwrap();

    cx.attach_namespaced("mldsa", pipeline.defs().unwrap(), &[MLDSA_DATA_BUS_ID])
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

fn sign<P: ml_dsa::MlDsaParams>(seed: u8, message: &[u8]) -> Signed {
    let signing = SigningKey::<P>::from_seed(&B32::from([seed; 32]));
    let key = signing.expanded_key();
    let signature = key.sign_internal(&[message], &B32::from([seed ^ 0x5a; 32]));

    Signed {
        pk: key.verifying_key().encode().as_slice().to_vec(),
        message: message.to_vec(),
        signature: signature.encode().as_slice().to_vec(),
    }
}

fn reference(signed: &Signed) -> MlDsaOutput {
    let (tr, _) = shake256(&signed.pk, 64);
    let (mu, _) = shake256(&[tr.as_slice(), &signed.message].concat(), 64);

    MlDsaOutput {
        tr: tr.try_into().unwrap(),
        mu: mu.try_into().unwrap(),
    }
}

#[test]
fn ml_dsa_44_verifies_through_pipeline() {
    honest::<MlDsa44>(
        MlDsaParams::ML_DSA_44,
        &[b"hekate ml-dsa-44 pipeline, tail of 1."],
    );
}

#[test]
fn ml_dsa_65_verifies_through_pipeline() {
    honest::<MlDsa65>(
        MlDsaParams::ML_DSA_65,
        &[b"hekate ml-dsa-65 pipeline, tail of 2.."],
    );
}

#[test]
fn ml_dsa_87_verifies_through_pipeline() {
    honest::<MlDsa87>(
        MlDsaParams::ML_DSA_87,
        &[b"hekate ml-dsa-87 pipeline, tail of 3..."],
    );
}

#[test]
fn one_pipeline_serves_two_calls() {
    honest::<MlDsa44>(
        MlDsaParams::ML_DSA_44,
        &[
            b"four",
            b"a second signer and a longer message of fifty-one b",
        ],
    );
}

#[test]
fn empty_and_hundred_byte_messages_verify_through_pipeline() {
    honest::<MlDsa44>(MlDsaParams::ML_DSA_44, &[b"", &[0x5c; 100]]);
}

#[test]
fn call_counts_past_label_space_are_refused() {
    let limit = Some(Error::Protocol {
        protocol: "mldsa_chiplet",
        message: "mldsa pipeline serves 1 to 256 calls",
    });

    assert!(MlDsaChiplet::<F>::new(MlDsaParams::ML_DSA_87, &[32; 227]).is_ok());
    assert_eq!(
        MlDsaChiplet::<F>::new(MlDsaParams::ML_DSA_87, &[32; 228]).err(),
        Some(Error::Protocol {
            protocol: "pqc_wiring",
            message: "coef bus labels exhausted: 65535 polynomials per composite",
        })
    );
    assert_eq!(
        MlDsaChiplet::<F>::new(MlDsaParams::ML_DSA_44, &[]).err(),
        limit
    );
    assert_eq!(
        MlDsaChiplet::<F>::new(MlDsaParams::ML_DSA_44, &[32; 257]).err(),
        limit
    );
}

#[test]
fn invalid_signature_has_no_witness() {
    let mut signed = sign::<MlDsa65>(9, b"signed message");
    signed.message[0] ^= 1;

    let pipeline = MlDsaChiplet::<F>::new(MlDsaParams::ML_DSA_65, &[signed.message.len()]).unwrap();

    assert_eq!(
        pipeline.trace(&[signed.input()]).err(),
        Some(Error::Protocol {
            protocol: "mldsa_chiplet",
            message: "signature does not verify: c̃' differs from c̃",
        })
    );
}

#[test]
fn matrix_entries_forged_with_w_hat_kept_break_only_coef_bus() {
    let params = MlDsaParams::ML_DSA_44;
    let calls = [sign::<MlDsa44>(3, b"forged matrix entry")];

    let (z0, z1) = (
        z_hat(params, &calls[0].signature, 0),
        z_hat(params, &calls[0].signature, 1),
    );

    let m = (0..N).find(|&m| z0[m] != 0 && z1[m] != 0).unwrap();

    let proof = forged(params, &calls, |f| {
        let a = f.a_hat(0, 1, 0)?;
        a[m] = (a[m] + z1[m]) % mldsa::Q;

        let a = f.a_hat(0, 1, 1)?;
        a[m] = (a[m] + mldsa::Q - z0[m]) % mldsa::Q;

        Ok(())
    });

    assert_rejected_only_at(&proof, "mldsa::coef");
}

#[test]
fn forged_hint_bit_breaks_hint_bus() {
    let calls = [sign::<MlDsa44>(4, b"forged hint bit")];

    let proof = forged(MlDsaParams::ML_DSA_44, &calls, |f| {
        let h = f.hint(0, 2)?;
        h[40] = !h[40];

        Ok(())
    });

    assert_rejected_at(&proof, "mldsa::hint");
}

#[test]
fn unsigned_message_breaks_ctilde_comparison() {
    let mut signed = sign::<MlDsa44>(5, b"a message the signer never signed");
    signed.message[3] ^= 0x20;

    let proof = forged(MlDsaParams::ML_DSA_44, &[signed], |_| Ok(()));

    assert_rejected_at(&proof, "mldsa::word");
}

#[test]
fn tail_bytes_of_longer_signed_message_break_only_ctrl_tail() {
    let params = MlDsaParams::ML_DSA_44;
    let extra = 0x41u8;

    let mut signed = sign::<MlDsa44>(6, b"fourteen bytes\x41");
    signed.message.pop();

    let last = params.pk_bytes() / 4 + signed.message.len().div_ceil(4) - 1;

    let proof = forged(params, &[signed], |f| {
        f.host(0)?[last] |= ((extra ^ 0x1f) as u32) << 16 | 0x1f << 24;

        Ok(())
    });

    assert_breaks_only(&proof, "ctrl_tail");
}

#[test]
fn host_words_altered_alone_break_only_word_bus() {
    let params = MlDsaParams::ML_DSA_44;
    let signed = sign::<MlDsa44>(7, b"host words");

    let ct = params.pk_bytes() / 4 + signed.message.len().div_ceil(4);
    let z = ct + params.lambda() / 16;
    let h = z + 8 * params.l() * params.z_bits();

    for at in [ct, z, h] {
        let proof = forged(params, std::slice::from_ref(&signed), |f| {
            f.host(0)?[at] ^= 1 << 3;

            Ok(())
        });

        assert_rejected_only_at(&proof, "mldsa::word");
    }
}

#[test]
fn host_t1_word_altered_alone_breaks_word_bus() {
    let signed = sign::<MlDsa44>(10, b"host t1 word");

    let proof = forged(MlDsaParams::ML_DSA_44, &[signed], |f| {
        f.host(0)?[8] ^= 1 << 3;

        Ok(())
    });

    assert_rejected_at(&proof, "mldsa::word");
}

#[test]
fn host_rho_word_altered_alone_breaks_lane_bus() {
    let signed = sign::<MlDsa44>(11, b"host rho word");

    let proof = forged(MlDsaParams::ML_DSA_44, &[signed], |f| {
        f.host(0)?[0] ^= 1 << 3;

        Ok(())
    });

    assert_rejected_at(&proof, "mldsa::lane");
}

#[test]
fn foreign_hint_decoded_while_high_bits_keeps_valid_one_breaks_only_hint_bus() {
    let params = MlDsaParams::ML_DSA_44;
    let mut signed = sign::<MlDsa44>(8, b"foreign hint");

    let (omega, k) = (params.omega(), params.k());
    let hint = signed.signature.len() - omega - k;

    let (i, last) = (0..k)
        .find_map(|i| {
            let start = match i {
                0 => 0,
                _ => signed.signature[hint + omega + i - 1] as usize,
            };

            let end = signed.signature[hint + omega + i] as usize;

            (end > start && signed.signature[hint + end - 1] < 255).then_some((i, end - 1))
        })
        .unwrap();

    let moved = signed.signature[hint + last] as usize;
    signed.signature[hint + last] += 1;

    let proof = forged(params, &[signed], |f| {
        let h = f.hint(0, i)?;
        h[moved] = true;
        h[moved + 1] = false;

        Ok(())
    });

    assert_rejected_only_at(&proof, "mldsa::hint");
}
