// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! ML-KEM receiver (FIPS 203): one proof chains KeyGen from
//! the seed d into Decaps of a sender's ciphertext c. d, z
//! and dk stay private; H(ek), c and valid are public.
//!
//! Usage: HEKATE_LEVEL=[512|768|1024] mlkem_receiver

#[path = "common/mod.rs"]
mod common;

use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::{Bit, Block16, Block32, Block128, TowerField};
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::trace::{ColumnTrace, TraceBuilder};
use hekate_pqc::mlkem::{
    self, MLKEM_DATA_BUS_ID, MlKemCall, MlKemChiplet, MlKemInput, MlKemOutput, MlKemParams,
};
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::define_columns;
use hekate_program::permutation::{PermutationCheckSpec, Source};
use hekate_program::{FixedShape, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;
use ml_kem::{Encapsulate, Kem, MlKem512, MlKem768, MlKem1024, TryKeyInit};
use rand::{TryRngCore, rngs::OsRng};
use zeroize::Zeroizing;

type F = Block128;
type H = DefaultHasher;

const SEED_WORDS: usize = 8;

const LINK_BUS_ID: &str = "dk_link";
const LINK_WAIVER: &str = "see hekate/examples/mlkem_receiver.rs: \
     link_idx is a fixed column; every index is emitted once \
     on a KeyGen output row and once on a Decaps input row";

define_columns! {
    ReceiverColumns {
        // ML-KEM service
        WORD: B32,
        SEL: Bit,

        // KeyGen-to-Decaps link
        LINK_IDX: B16,
        LINK_SEL: Bit,
    }
}

/// Host rows of KeyGen, then Decaps, in service order, and
/// the pairs the dk link ties: each KeyGen output row with
/// the Decaps input row carrying the same word of dk.
struct Layout {
    words: usize,
    h: usize,
    c: usize,
    c_words: usize,
    valid: usize,
    links: Vec<(usize, u16)>,
}

impl Layout {
    fn new(params: MlKemParams) -> Self {
        let w = 96 * params.k();
        let c_words = 8 * (params.du() as usize * params.k() + params.dv() as usize);

        let ek_out = SEED_WORDS;
        let rho_out = ek_out + w;
        let h_out = rho_out + SEED_WORDS;
        let dk_out = h_out + SEED_WORDS;

        let dk_in = dk_out + w;
        let ek_in = dk_in + w;
        let rho_in = ek_in + w;
        let h_in = rho_in + SEED_WORDS;
        let c = h_in + 2 * SEED_WORDS;
        let valid = c + c_words + SEED_WORDS;

        let mut links = Vec::with_capacity(4 * w + 4 * SEED_WORDS);

        // Link indices are word positions in FIPS 203's
        // dk = dk_PKE ‖ ek ‖ h ‖ z (Algorithm 16).
        for (keygen, decaps, first, len) in [
            (dk_out, dk_in, 0, w),
            (ek_out, ek_in, w, w),
            (rho_out, rho_in, 2 * w, SEED_WORDS),
            (h_out, h_in, 2 * w + SEED_WORDS, SEED_WORDS),
        ] {
            for j in 0..len {
                let idx = (first + j) as u16;

                links.push((keygen + j, idx));
                links.push((decaps + j, idx));
            }
        }

        links.sort_unstable();

        Self {
            words: valid + 1,
            h: h_out,
            c,
            c_words,
            valid,
            links,
        }
    }

    fn published(&self) -> Vec<usize> {
        (self.h..self.h + SEED_WORDS)
            .chain(self.c..self.c + self.c_words)
            .chain([self.valid])
            .collect()
    }
}

fn build(
    pipeline: &MlKemChiplet<F>,
    layout: &Layout,
    rows: usize,
) -> errors::Result<CircuitProgram<F>> {
    let mut cx = Circuit::<F>::new("MlKemReceiver", rows)?;

    let cols = cx.schema(&ReceiverColumns::build_layout());

    let word = cols.at(ReceiverColumns::WORD);
    let sel = cols.at(ReceiverColumns::SEL);
    let link_idx = cols.at(ReceiverColumns::LINK_IDX);
    let link_sel = cols.at(ReceiverColumns::LINK_SEL);

    cx.fix(
        sel,
        FixedShape::Cadence {
            stride: 1,
            count: layout.words,
            origin: 0,
            values: vec![F::ONE],
        },
    );
    cx.fix(
        link_idx,
        FixedShape::Sparse(
            layout
                .links
                .iter()
                .map(|&(row, idx)| (row, F::from(idx as u32)))
                .filter(|&(_, v)| v != F::ZERO)
                .collect(),
        ),
    );
    cx.fix(
        link_sel,
        FixedShape::Sparse(layout.links.iter().map(|&(row, _)| (row, F::ONE)).collect()),
    );

    let cs = cx.cs();

    // WORD is zero past the service rows: no free cell on padding
    cs.constrain((cs.one() + cs.col(sel.index())) * cs.col(word.index()));

    cx.call(&mlkem::service(), &[word], sel)?;

    // Each dk index is emitted on a KeyGen output row and a Decaps
    // input row; the pair cancels only if both rows hold the same word.
    cx.bus(
        LINK_BUS_ID,
        PermutationCheckSpec::new(
            vec![
                (Source::Column(link_idx.index()), b"kappa_link_idx" as &[u8]),
                (Source::Column(word.index()), b"kappa_link_word" as &[u8]),
            ],
            Some(link_sel.index()),
        )
        .with_clock_waiver(LINK_WAIVER),
    );

    for row in layout.published() {
        cx.publish(word, row);
    }

    cx.attach_namespaced("mlkem", pipeline.defs()?, &[MLKEM_DATA_BUS_ID])?;

    cx.compile()
}

fn host_trace(words: &[u32], layout: &Layout, rows: usize) -> errors::Result<ColumnTrace> {
    let mut tb = TraceBuilder::new_secret(
        &ReceiverColumns::build_layout(),
        rows.trailing_zeros() as usize,
    )?;

    for (r, &word) in words.iter().enumerate() {
        tb.set_b32(ReceiverColumns::WORD, r, Block32::from(word))?;
        tb.set_bit(ReceiverColumns::SEL, r, Bit::ONE)?;
    }

    for &(row, idx) in &layout.links {
        tb.set_b16(ReceiverColumns::LINK_IDX, row, Block16(idx))?;
        tb.set_bit(ReceiverColumns::LINK_SEL, row, Bit::ONE)?;
    }

    Ok(tb.build())
}

/// A sender encapsulating to `ek` with the ml-kem crate,
/// an implementation independent of the pipeline: (c, K).
fn encapsulate<K: Kem>(ek: &[u8]) -> (Vec<u8>, Zeroizing<[u8; 32]>) {
    let ek = K::EncapsulationKey::new_from_slice(ek).expect("FIPS 203 encapsulation key");
    let (c, key) = ek.encapsulate();

    let mut shared = Zeroizing::new([0u8; 32]);
    shared.copy_from_slice(&key);

    (c.to_vec(), shared)
}

fn run<K: Kem>(label: &str, params: MlKemParams) {
    common::init(label);

    let mut d = Zeroizing::new([0u8; 32]);
    let mut z = Zeroizing::new([0u8; 32]);

    OsRng.try_fill_bytes(&mut *d).unwrap();
    OsRng.try_fill_bytes(&mut *z).unwrap();

    // Key generation: the pipeline's witness generator expands d
    // into (ek, dk) with no proof; the chained proof re-derives them.
    let keygen = MlKemChiplet::<F>::new(params, &[MlKemCall::KeyGen]).expect("pipeline build");

    let (ek, dk) = common::phase("Key Generation", || {
        let generated = keygen
            .trace(&[MlKemInput::KeyGen { d: &d }])
            .expect("key generation");

        let MlKemOutput::KeyGen { ek, dk_pke, h } = &generated.outputs[0] else {
            unreachable!()
        };

        let dk = Zeroizing::new([dk_pke.as_slice(), ek, h, &z[..]].concat());

        (ek.clone(), dk)
    });

    let (c, sent) = encapsulate::<K>(&ek);

    println!("  Encapsulation key: {} bytes, hashed into H(ek)", ek.len());
    println!("  Decapsulation key: {} bytes, private", dk.len());
    println!("  Ciphertext:        {} bytes, public", c.len());

    // KeyGen and Decaps share one proof: the dk link
    // binds Decaps to the key KeyGen derives from d.
    let pipeline = MlKemChiplet::<F>::new(params, &[MlKemCall::KeyGen, MlKemCall::Decaps])
        .expect("pipeline build");

    let traced = common::phase("Trace Generation", || {
        let inputs = [
            MlKemInput::KeyGen { d: &d },
            MlKemInput::Decaps { dk: &dk, c: &c },
        ];

        pipeline.trace(&inputs).expect("well-formed inputs")
    });

    let MlKemOutput::Decaps { key, valid } = &traced.outputs[1] else {
        unreachable!()
    };

    assert!(*key == *sent, "decapsulated key differs from the sender's");

    println!("  Valid:             {valid}");

    let layout = Layout::new(params);
    let words = &traced.words;
    let rows = words.len().next_power_of_two();

    let public_inputs: Vec<F> = layout
        .published()
        .iter()
        .map(|&row| F::from(u128::from(words[row])))
        .collect();

    println!(
        "  Public inputs:     {} words, H(ek), c then valid",
        public_inputs.len()
    );

    let program = build(&pipeline, &layout, rows).expect("program build");
    let host = host_trace(words, &layout, rows).expect("host trace");

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
            b"MlKemReceiver_E2E",
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

    let mut verifier_transcript = Transcript::<H>::new(b"MlKemReceiver_E2E");
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
    let level = common::level("768");

    match level.as_str() {
        "512" => run::<MlKem512>("ML-KEM-512 Receiver", MlKemParams::ML_KEM_512),
        "768" => run::<MlKem768>("ML-KEM-768 Receiver", MlKemParams::ML_KEM_768),
        "1024" => run::<MlKem1024>("ML-KEM-1024 Receiver", MlKemParams::ML_KEM_1024),
        other => {
            eprintln!("Usage: HEKATE_LEVEL=[512|768|1024] mlkem_receiver (got {other:?})");
            std::process::exit(1);
        }
    }
}
