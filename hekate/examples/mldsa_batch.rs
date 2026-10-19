// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Batched ML-DSA verification (FIPS 204): one proof over
//! `--count` signatures under as many keys, every key and
//! signature private, each call's M' and tr = H(pk) public.

#[path = "common/mod.rs"]
mod common;

use std::time::Instant;

use clap::error::ErrorKind;
use clap::{CommandFactory, Parser, ValueEnum};
use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::{Bit, Block32, Block128, TowerField};
use hekate_core::errors;
use hekate_core::trace::{ColumnTrace, Trace, TraceBuilder};
use hekate_pqc::mldsa::{self, MLDSA_DATA_BUS_ID, MlDsaChiplet, MlDsaInput, MlDsaParams};
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::define_columns;
use hekate_program::{Air, FixedShape, ProgramInstance, ProgramWitness};
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
    m_prime: Vec<u8>,
    pk: Zeroizing<Vec<u8>>,
    signature: Zeroizing<Vec<u8>>,
}

#[derive(Clone, Copy, ValueEnum)]
enum Level {
    #[value(name = "44")]
    MlDsa44,
    #[value(name = "65")]
    MlDsa65,
    #[value(name = "87")]
    MlDsa87,
}

#[derive(Parser)]
struct Args {
    #[command(flatten)]
    run: common::RunArgs,

    /// ML-DSA parameter set.
    #[arg(long, value_enum, default_value = "65")]
    level: Level,

    /// Signatures verified in the one proof.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u16).range(1..))]
    count: u16,
}

/// The host requests one word a row, in the order the pipeline
/// serves them: per call pk, M' and σ in, then tr and μ out.
fn build(
    pipeline: &MlDsaChiplet<F>,
    rows: usize,
    words: usize,
    published: &[usize],
) -> errors::Result<CircuitProgram<F>> {
    let mut cx = Circuit::<F>::new("MlDsaBatchVerify", rows)?;

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

/// Signs M' under a fresh key with sign_internal (FIPS 204
/// Algorithm 7): the context-taking sign would wrap M' a
/// second time, and the pipeline verifies M' as given.
fn sign<P: ml_dsa::MlDsaParams>(m_prime: Vec<u8>) -> Signed {
    let mut seed = Zeroizing::new([0u8; 32]);
    let mut rnd = Zeroizing::new([0u8; 32]);

    OsRng.try_fill_bytes(&mut *seed).unwrap();
    OsRng.try_fill_bytes(&mut *rnd).unwrap();

    let key = SigningKey::<P>::from_seed(&B32::from(*seed));
    let expanded = key.expanded_key();
    let signature = expanded.sign_internal(&[&m_prime], &B32::from(*rnd));

    Signed {
        m_prime,
        pk: Zeroizing::new(expanded.verifying_key().encode().to_vec()),
        signature: Zeroizing::new(signature.encode().to_vec()),
    }
}

fn published_rows(params: MlDsaParams, batch: &[Signed]) -> Vec<usize> {
    let pk_words = params.pk_bytes() / 4;
    let sig_words = params.sig_bytes().div_ceil(4);

    let mut rows = Vec::new();
    let mut start = 0;

    for signed in batch {
        let msg_words = signed.m_prime.len().div_ceil(4);
        let tr_row = start + pk_words + msg_words + sig_words;

        rows.extend(start + pk_words..start + pk_words + msg_words);
        rows.extend(tr_row..tr_row + DIGEST_WORDS);

        start = tr_row + 2 * DIGEST_WORDS;
    }

    rows
}

fn print_tables(pipeline: &MlDsaChiplet<F>, traces: &[ColumnTrace], host_rows: usize) {
    println!("--------------------------------------------------");
    println!("  TABLE HEIGHTS");
    println!("--------------------------------------------------");
    println!(
        "  {:<14} {:>12} {:>5} {:>6} B/row",
        "host", host_rows, "", 5
    );

    let defs = pipeline.defs().expect("pipeline defs");

    for (def, trace) in defs.iter().zip(traces) {
        let row_bytes: usize = trace.column_layout().iter().map(|ct| ct.byte_size()).sum();

        println!(
            "  {:<14} {:>12} {:>5} {:>6} B/row",
            def.name(),
            1usize << trace.num_vars(),
            format!("2^{}", trace.num_vars()),
            row_bytes,
        );
    }

    println!("--------------------------------------------------");
}

fn run(args: &common::RunArgs, label: &str, params: MlDsaParams, batch: &[Signed]) {
    common::init(label);

    let count = batch.len();

    println!("  Signatures:     {count}, one key each, private");
    println!("  Public key:     {} bytes each", params.pk_bytes());
    println!("  Signature:      {} bytes each", params.sig_bytes());

    let msg_lens: Vec<usize> = batch.iter().map(|s| s.m_prime.len()).collect();

    let pipeline = MlDsaChiplet::<F>::new(params, &msg_lens).expect("pipeline build");

    let inputs: Vec<MlDsaInput<'_>> = batch
        .iter()
        .map(|s| MlDsaInput {
            pk: &s.pk,
            message: &s.m_prime,
            signature: &s.signature,
        })
        .collect();

    let traced = common::phase("Trace Generation", || {
        pipeline.trace(&inputs).expect("every signature verifies")
    });

    let words = &traced.words;
    let rows = words.len().next_power_of_two();

    let published = published_rows(params, batch);

    assert_eq!(
        published.last().map(|&r| r + DIGEST_WORDS + 1),
        Some(words.len())
    );

    let public_inputs: Vec<F> = published
        .iter()
        .map(|&row| F::from(u128::from(words[row])))
        .collect();

    println!(
        "  Public inputs:  {} words, M' then tr per call",
        public_inputs.len()
    );

    print_tables(&pipeline, &traced.traces, rows);

    let program = build(&pipeline, rows, words.len(), &published).expect("program build");
    let host = host_trace(words, rows).expect("host trace");

    let instance = ProgramInstance::new(rows, public_inputs);
    let witness = ProgramWitness::new(host).with_chiplets(traced.traces);

    let config = args.config();

    let mut blinding_seed = Zeroizing::new([0u8; 32]);
    OsRng.try_fill_bytes(&mut *blinding_seed).unwrap();

    let prove_start = Instant::now();
    let proof = common::phase("Proving", || {
        prove(
            b"MlDsa_Batch_E2E",
            &program,
            &instance,
            &witness,
            &config,
            *blinding_seed,
            None,
        )
        .expect("Prover failed")
    });
    let prove_time = prove_start.elapsed();

    common::proof_breakdown(&proof);

    let proof_bytes = hekate_sdk::serialize_proof_bytes(&proof).len();

    let mut verifier_transcript = Transcript::<H>::new(b"MlDsa_Batch_E2E");

    let pinned_id = args.audited_id(&program);

    let verify_start = Instant::now();
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
    let verify_time = verify_start.elapsed();

    let per = count as f64;

    println!("--------------------------------------------------");
    println!("  PER SIGNATURE ({count} in the batch)");
    println!("--------------------------------------------------");
    println!(
        "  Prove:          {:>10.2} ms   total {:.2?}",
        prove_time.as_secs_f64() * 1e3 / per,
        prove_time
    );
    println!(
        "  Verify:         {:>10.3} ms   total {:.2?}",
        verify_time.as_secs_f64() * 1e3 / per,
        verify_time
    );
    println!(
        "  Proof:          {:>10.2} KB   total {:.2} KB",
        proof_bytes as f64 / 1024.0 / per,
        proof_bytes as f64 / 1024.0
    );

    common::result(is_valid);
}

fn main() {
    let args = Args::parse();

    let (level, params, max_count) = match args.level {
        Level::MlDsa44 => ("44", MlDsaParams::ML_DSA_44, 128),
        Level::MlDsa65 => ("65", MlDsaParams::ML_DSA_65, 128),
        Level::MlDsa87 => ("87", MlDsaParams::ML_DSA_87, 64),
    };

    if args.count > max_count {
        Args::command()
            .error(
                ErrorKind::ValueValidation,
                format!("ML-DSA-{level} proves at most {max_count} signatures in one batch"),
            )
            .exit();
    }

    let batch: Vec<Signed> = (0..args.count)
        .map(|i| {
            let message =
                format!("Hekate ML-DSA-{level} batch verification example, signature {i}");

            // FIPS 204 Algorithm 2:
            // M' = 0 ∥ |ctx| ∥ ctx ∥ M, here with the empty context.
            let m_prime = [&[0u8, 0][..], message.as_bytes()].concat();

            match args.level {
                Level::MlDsa44 => sign::<MlDsa44>(m_prime),
                Level::MlDsa65 => sign::<MlDsa65>(m_prime),
                Level::MlDsa87 => sign::<MlDsa87>(m_prime),
            }
        })
        .collect();

    run(
        &args.run,
        &format!("ML-DSA-{level} Batch Signature Verification"),
        params,
        &batch,
    );
}
