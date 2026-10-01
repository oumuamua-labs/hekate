// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! ML-DSA verification (FIPS 204) with the public key and
//! the signature in the witness: the statement publishes M'
//! and tr = H(pk). Usage: HEKATE_LEVEL=[44|65|87] mldsa

#[path = "common/mod.rs"]
mod common;

use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::{Bit, Block32, Block128, TowerField};
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::trace::{ColumnTrace, TraceBuilder};
use hekate_pqc::mldsa::{self, MLDSA_DATA_BUS_ID, MlDsaChiplet, MlDsaInput, MlDsaParams};
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::define_columns;
use hekate_program::{FixedShape, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;
use ml_dsa::{B32, MlDsa44, MlDsa65, MlDsa87, SigningKey};
use rand::{TryRngCore, rngs::OsRng};
use zeroize::Zeroizing;

type F = Block128;
type H = DefaultHasher;

const DIGEST_WORDS: usize = 16;

define_columns! {
    VerifyColumns {
        WORD: B32,
        SEL: Bit,
    }
}

struct Signed {
    pk: Zeroizing<Vec<u8>>,
    signature: Zeroizing<Vec<u8>>,
}

/// The host requests one word a row, in the order the pipeline
/// serves them: pk, M' and σ in, then tr and μ out.
fn build(
    pipeline: &MlDsaChiplet<F>,
    rows: usize,
    words: usize,
    published: &[usize],
) -> errors::Result<CircuitProgram<F>> {
    let mut cx = Circuit::<F>::new("MlDsaVerify", rows)?;

    let cols = cx.schema(&VerifyColumns::build_layout());

    let word = cols.at(VerifyColumns::WORD);
    let sel = cols.at(VerifyColumns::SEL);

    cx.fix(
        sel,
        FixedShape::Cadence {
            stride: 1,
            count: words,
            origin: 0,
            values: vec![F::ONE],
        },
    );

    let cs = cx.cs();

    // WORD is zero past the service rows: no free cell on padding
    cs.constrain((cs.one() + cs.col(sel.index())) * cs.col(word.index()));

    cx.call(&mldsa::service(), &[word], sel)?;

    for &row in published {
        cx.publish(word, row);
    }

    cx.attach_namespaced("mldsa", pipeline.defs()?, &[MLDSA_DATA_BUS_ID])?;

    cx.compile()
}

fn host_trace(words: &[u32], rows: usize) -> errors::Result<ColumnTrace> {
    let mut tb = TraceBuilder::new_secret(
        &VerifyColumns::build_layout(),
        rows.trailing_zeros() as usize,
    )?;

    for (r, &word) in words.iter().enumerate() {
        tb.set_b32(VerifyColumns::WORD, r, Block32::from(word))?;
        tb.set_bit(VerifyColumns::SEL, r, Bit::ONE)?;
    }

    Ok(tb.build())
}

/// Signs M' with ml-dsa's sign_internal (FIPS 204 Algorithm 7).
/// The context-taking sign would wrap M' a second time, and
/// the pipeline, which verifies M' as given, would reject it.
fn sign<P: ml_dsa::MlDsaParams>(m_prime: &[u8]) -> Signed {
    let mut seed = Zeroizing::new([0u8; 32]);
    let mut rnd = Zeroizing::new([0u8; 32]);

    OsRng.try_fill_bytes(&mut *seed).unwrap();
    OsRng.try_fill_bytes(&mut *rnd).unwrap();

    let key = SigningKey::<P>::from_seed(&B32::from(*seed));
    let expanded = key.expanded_key();
    let signature = expanded.sign_internal(&[m_prime], &B32::from(*rnd));

    Signed {
        pk: Zeroizing::new(expanded.verifying_key().encode().to_vec()),
        signature: Zeroizing::new(signature.encode().to_vec()),
    }
}

fn run(label: &str, params: MlDsaParams, m_prime: &[u8], signed: &Signed) {
    common::init(label);

    println!("  Public key:     {} bytes, private", signed.pk.len());
    println!(
        "  Signature:      {} bytes, private",
        signed.signature.len()
    );
    println!("  M':             {} bytes, public", m_prime.len());

    let pipeline = MlDsaChiplet::<F>::new(params, &[m_prime.len()]).expect("pipeline build");

    let traced = common::phase("Trace Generation", || {
        let input = MlDsaInput {
            pk: &signed.pk,
            message: m_prime,
            signature: &signed.signature,
        };

        pipeline.trace(&[input]).expect("the signature verifies")
    });

    let words = &traced.words;
    let rows = words.len().next_power_of_two();

    // Public: M', right after the pk words, and tr, the first
    // of the two digests the host receives (tr, then μ).
    let pk_words = params.pk_bytes() / 4;
    let tr_row = words.len() - 2 * DIGEST_WORDS;

    let published: Vec<usize> = (pk_words..pk_words + m_prime.len().div_ceil(4))
        .chain(tr_row..tr_row + DIGEST_WORDS)
        .collect();

    let public_inputs: Vec<F> = published
        .iter()
        .map(|&row| F::from(u128::from(words[row])))
        .collect();

    println!(
        "  Public inputs:  {} words, M' then tr",
        public_inputs.len()
    );

    let program = build(&pipeline, rows, words.len(), &published).expect("program build");
    let host = host_trace(words, rows).expect("host trace");

    let instance = ProgramInstance::new(rows, public_inputs);
    let witness = ProgramWitness::new(host).with_chiplets(traced.traces);

    let config = Config {
        zero_knowledge: common::zero_knowledge(),
        ..Config::default()
    };

    let mut blinding_seed = Zeroizing::new([0u8; 32]);
    OsRng.try_fill_bytes(&mut *blinding_seed).unwrap();

    let proof = common::phase("Proving", || {
        prove(
            b"MlDsa_E2E",
            &program,
            &instance,
            &witness,
            &config,
            *blinding_seed,
            None,
        )
        .expect("Prover failed")
    });

    common::proof_breakdown(&proof);

    let mut verifier_transcript = Transcript::<H>::new(b"MlDsa_E2E");
    let pinned_id = common::audited_id(&program);

    let is_valid = common::phase_with_mem("Verifying", || {
        HekateVerifier::<F, H>::verify(
            &pinned_id,
            &program,
            &instance,
            &proof,
            &mut verifier_transcript,
            &config,
        )
        .expect("Verifier failed")
    });

    common::result(is_valid);
}

fn main() {
    let level = common::level("65");
    let message = format!("Hekate ML-DSA-{level} verification example");

    // FIPS 204 Algorithm 2:
    // M' = 0 ∥ |ctx| ∥ ctx ∥ M, here with the empty context.
    let m_prime = [&[0u8, 0][..], message.as_bytes()].concat();

    let (params, signed) = match level.as_str() {
        "44" => (MlDsaParams::ML_DSA_44, sign::<MlDsa44>(&m_prime)),
        "65" => (MlDsaParams::ML_DSA_65, sign::<MlDsa65>(&m_prime)),
        "87" => (MlDsaParams::ML_DSA_87, sign::<MlDsa87>(&m_prime)),
        other => {
            eprintln!("Usage: HEKATE_LEVEL=[44|65|87] mldsa (got {other:?})");
            std::process::exit(1);
        }
    };

    run(
        &format!("ML-DSA-{level} Signature Verification"),
        params,
        &m_prime,
        &signed,
    );
}
