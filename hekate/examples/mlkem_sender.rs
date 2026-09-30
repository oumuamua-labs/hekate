// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! ML-KEM sender (FIPS 203) with an AES-256-CTR payload.
//! The proof shows c encapsulates K to the key behind H(ek)
//! and K encrypts the public message; ek, m and K stay private.
//!
//! Usage: HEKATE_LEVEL=[512|768|1024] HEKATE_MESSAGE=<text> mlkem_sender

#[path = "common/mod.rs"]
mod common;

use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::{Bit, Block8, Block16, Block32, Block128, TowerField};
use hekate_aes::trace::{Aes256Call, aes256_encrypt_block, expand_key_256};
use hekate_aes::{Aes256Chiplet, AesRound256Air, host_key_selector_shape, host_selector_shape};
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::trace::{ColumnTrace, TraceBuilder};
use hekate_pqc::mlkem::{
    self, MLKEM_DATA_BUS_ID, MlKemCall, MlKemChiplet, MlKemInput, MlKemOutput, MlKemParams,
};
use hekate_program::circuit::{Circuit, CircuitProgram, Col};
use hekate_program::define_columns;
use hekate_program::permutation::{PermutationCheckSpec, Source};
use hekate_program::{FixedShape, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;
use ml_kem::kem::Decapsulator;
use ml_kem::{Ciphertext, Decapsulate, Generate, Kem, KeyExport, MlKem512, MlKem768, MlKem1024};
use rand::{TryRngCore, rngs::OsRng};
use zeroize::Zeroizing;

type F = Block128;
type H = DefaultHasher;

const SEED_WORDS: usize = 8;
const BLOCK: usize = 16;
const NONCE_BYTES: usize = 12;
const AES_ROWS: usize = 15;
const SBOX_ROUNDS: usize = 14;

const DEFAULT_MESSAGE: &str =
    "Hekate ML-KEM sender: this message travels under AES-256-CTR with the encapsulated key.";

const KEY_LINK_BUS_ID: &str = "key_link";
const KEY_LINK_WAIVER: &str = "see hekate/examples/mlkem_sender.rs: \
     link_idx is a fixed column over K's host rows and a constant \
     on the first AES row; each index is emitted once on each side";

define_columns! {
    /// Row r carries the r-th ML-KEM word;
    /// rows 2b and 2b + 1 also carry AES block b,
    /// the counter block in and its keystream out.
    SenderColumns {
        // ML-KEM service
        WORD: B32,
        SEL: Bit,

        // K's end of the key link
        LINK_IDX: B16,
        LINK_SEL: Bit,

        // AES-256 service
        KEY: [B8; 32],
        KEY_SELECTOR: Bit,
        DATA: [B8; 16],
        SELECTOR: Bit,

        // AES end of the key link
        PACKED: [B32; 8],
        FIRST: Bit,

        // One key on every AES row
        CARRY: Bit,

        // CTR mode
        COUNTER: [B8; 16],
        PLAINTEXT: [B8; 16],
        CIPHERTEXT: [B8; 16],
    }
}

/// Rows of the ML-KEM host words in service order:
/// ek, m and h in, then K and c out.
struct Layout {
    words: usize,
    h: usize,
    key: usize,
    c: usize,
    c_words: usize,
}

impl Layout {
    fn new(params: MlKemParams) -> Self {
        let m = 96 * params.k() + SEED_WORDS;
        let h = m + SEED_WORDS;
        let key = h + SEED_WORDS;
        let c = key + SEED_WORDS;
        let c_words = 8 * (params.du() as usize * params.k() + params.dv() as usize);

        Self {
            words: c + c_words,
            h,
            key,
            c,
            c_words,
        }
    }
}

/// AES-256-CTR (NIST SP 800-38A): keystream block b is
/// AES_K(nonce ‖ b) and the ciphertext is message ⊕ keystream.
struct Ctr {
    counters: Vec<[u8; BLOCK]>,
    keystream: Zeroizing<Vec<[u8; BLOCK]>>,
    plaintext: Zeroizing<Vec<u8>>,
    ciphertext: Vec<u8>,
}

impl Ctr {
    fn encrypt(key: &[u8; 32], nonce: &[u8; NONCE_BYTES], message: &[u8]) -> Self {
        let round_keys = Zeroizing::new(expand_key_256(key));
        let blocks = message.len().div_ceil(BLOCK);

        let counters: Vec<[u8; BLOCK]> = (0..blocks as u32)
            .map(|b| {
                let mut block = [0u8; BLOCK];
                block[..NONCE_BYTES].copy_from_slice(nonce);
                block[NONCE_BYTES..].copy_from_slice(&b.to_be_bytes());

                block
            })
            .collect();

        let keystream = Zeroizing::new(
            counters
                .iter()
                .map(|ctr| aes256_encrypt_block(&round_keys, ctr))
                .collect::<Vec<_>>(),
        );

        let mut plaintext = Zeroizing::new(vec![0u8; blocks * BLOCK]);
        plaintext[..message.len()].copy_from_slice(message);

        let ciphertext = plaintext
            .iter()
            .zip(keystream.iter().flatten())
            .map(|(p, k)| p ^ k)
            .collect();

        Self {
            counters,
            keystream,
            plaintext,
            ciphertext,
        }
    }

    fn blocks(&self) -> usize {
        self.counters.len()
    }
}

fn build(
    pipeline: &MlKemChiplet<F>,
    aes: &Aes256Chiplet<F>,
    layout: &Layout,
    message_len: usize,
    rows: usize,
    statement: &[(usize, usize, F)],
) -> errors::Result<CircuitProgram<F>> {
    let blocks = message_len.div_ceil(BLOCK);

    let mut cx = Circuit::<F>::new("MlKemSender", rows)?;

    let cols = cx.schema(&SenderColumns::build_layout());

    let at = |col: usize| cols.at(col);

    let (word, sel) = (at(SenderColumns::WORD), at(SenderColumns::SEL));
    let (link_idx, link_sel) = (at(SenderColumns::LINK_IDX), at(SenderColumns::LINK_SEL));
    let (key_selector, selector) = (at(SenderColumns::KEY_SELECTOR), at(SenderColumns::SELECTOR));
    let (first, carry) = (at(SenderColumns::FIRST), at(SenderColumns::CARRY));

    let key: Vec<Col> = (0..32).map(|j| at(SenderColumns::KEY + j)).collect();
    let data: Vec<Col> = (0..BLOCK).map(|j| at(SenderColumns::DATA + j)).collect();

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
            (1..SEED_WORDS)
                .map(|j| (layout.key + j, F::from(j as u32)))
                .collect(),
        ),
    );
    cx.fix(
        link_sel,
        FixedShape::Sparse((0..SEED_WORDS).map(|j| (layout.key + j, F::ONE)).collect()),
    );
    cx.fix(selector, host_selector_shape(2, blocks));
    cx.fix(key_selector, host_key_selector_shape(2, blocks));

    // The first AES row answers the key link, and CARRY holds
    // the key equal on every AES row, binding all blocks to K.
    cx.fix(first, FixedShape::Sparse(vec![(0, F::ONE)]));
    cx.fix(
        carry,
        FixedShape::Cadence {
            stride: 1,
            count: 2 * blocks - 1,
            origin: 0,
            values: vec![F::ONE],
        },
    );

    for i in 0..BLOCK - NONCE_BYTES {
        cx.fix(
            at(SenderColumns::COUNTER + NONCE_BYTES + i),
            FixedShape::Sparse(
                (0..blocks)
                    .map(|b| (2 * b, F::from(u32::from((b as u32).to_be_bytes()[i]))))
                    .collect(),
            ),
        );
    }

    cx.call(&mlkem::service(), &[word], sel)?;
    cx.call(&AesRound256Air::link_service(), &data, selector)?;
    cx.call(&AesRound256Air::key_service(), &key, key_selector)?;

    // K's words and the packed AES key emit the same (index,
    // word) tokens: the bus balances only if the keys agree.
    cx.bus(
        KEY_LINK_BUS_ID,
        key_link(Source::Column(link_idx.index()), word, link_sel),
    );

    for j in 0..SEED_WORDS {
        cx.bus(
            KEY_LINK_BUS_ID,
            key_link(
                Source::Const(j as u128),
                at(SenderColumns::PACKED + j),
                first,
            ),
        );
    }

    let cs = cx.cs();

    let col = |c: Col| cs.col(c.index());

    // WORD is zero past the service rows: no free cell on padding
    cs.constrain((cs.one() + col(sel)) * col(word));

    // PACKED_j is key bytes 4j..4j+3 as a little-endian word.
    // In the tower basis, a byte times 2^(8i) is the byte shifted by 8i.
    for j in 0..SEED_WORDS {
        let packed = (0..4).fold(col(at(SenderColumns::PACKED + j)), |acc, i| {
            acc + cs.constant(F::from(1u128 << (8 * i))) * col(key[4 * j + i])
        });

        cs.constrain(packed);
    }

    for &k in &key {
        cs.constrain(col(carry) * (cs.next(k.index()) + col(k)));
        cs.constrain((cs.one() + col(selector)) * col(k));
    }

    // Key rows feed AES the counter block XOR round key 0,
    // the first 16 key bytes; output rows XOR the keystream into
    // the message. selector + key_selector is 1 on output rows only.
    let output = col(selector) + col(key_selector);

    for j in 0..BLOCK {
        let whitened = col(data[j]) + col(at(SenderColumns::COUNTER + j)) + col(key[j]);
        let encrypted = col(at(SenderColumns::CIPHERTEXT + j))
            + col(at(SenderColumns::PLAINTEXT + j))
            + col(data[j]);

        cs.constrain(col(key_selector) * whitened);
        cs.constrain(output * encrypted);

        cs.constrain((cs.one() + col(selector)) * col(data[j]));
        cs.constrain((cs.one() + col(key_selector)) * col(at(SenderColumns::COUNTER + j)));
        cs.constrain((cs.one() + output) * col(at(SenderColumns::PLAINTEXT + j)));
        cs.constrain((cs.one() + output) * col(at(SenderColumns::CIPHERTEXT + j)));
    }

    for &(c, row, _) in statement {
        cx.publish(at(c), row);
    }

    let pad = (BLOCK - message_len % BLOCK) % BLOCK;

    for j in BLOCK - pad..BLOCK {
        cx.boundary(at(SenderColumns::PLAINTEXT + j), 2 * blocks - 1, F::ZERO);
    }

    cx.attach_namespaced("mlkem", pipeline.defs()?, &[MLKEM_DATA_BUS_ID])?;

    for def in aes.composite().flatten_defs()? {
        cx.attach(def);
    }

    cx.compile()
}

/// The host trace: the ML-KEM words, K's link rows, and two
/// rows per AES block, each AES row holding the whole key.
fn host_trace(
    words: &[u32],
    layout: &Layout,
    key: &[u8; 32],
    ctr: &Ctr,
    rows: usize,
) -> errors::Result<ColumnTrace> {
    let mut tb = TraceBuilder::new_secret(
        &SenderColumns::build_layout(),
        rows.trailing_zeros() as usize,
    )?;

    for (r, &word) in words.iter().enumerate() {
        tb.set_b32(SenderColumns::WORD, r, Block32::from(word))?;
        tb.set_bit(SenderColumns::SEL, r, Bit::ONE)?;
    }

    for j in 0..SEED_WORDS {
        tb.set_b16(SenderColumns::LINK_IDX, layout.key + j, Block16(j as u16))?;
        tb.set_bit(SenderColumns::LINK_SEL, layout.key + j, Bit::ONE)?;
    }

    let aes_rows = 2 * ctr.blocks();

    for r in 0..aes_rows {
        for (j, &byte) in key.iter().enumerate() {
            tb.set_b8(SenderColumns::KEY + j, r, Block8(byte))?;
        }

        for j in 0..SEED_WORDS {
            let packed = u32::from_le_bytes(core::array::from_fn(|i| key[4 * j + i]));

            tb.set_b32(SenderColumns::PACKED + j, r, Block32::from(packed))?;
        }

        tb.set_bit(SenderColumns::SELECTOR, r, Bit::ONE)?;
        tb.set_bit(SenderColumns::CARRY, r, Bit::from((r + 1 < aes_rows) as u8))?;
    }

    tb.set_bit(SenderColumns::FIRST, 0, Bit::ONE)?;

    for (b, counter) in ctr.counters.iter().enumerate() {
        let (input, output) = (2 * b, 2 * b + 1);

        tb.set_bit(SenderColumns::KEY_SELECTOR, input, Bit::ONE)?;

        for j in 0..BLOCK {
            let at = b * BLOCK + j;

            tb.set_b8(SenderColumns::COUNTER + j, input, Block8(counter[j]))?;
            tb.set_b8(SenderColumns::DATA + j, input, Block8(counter[j] ^ key[j]))?;

            tb.set_b8(SenderColumns::DATA + j, output, Block8(ctr.keystream[b][j]))?;
            tb.set_b8(
                SenderColumns::PLAINTEXT + j,
                output,
                Block8(ctr.plaintext[at]),
            )?;
            tb.set_b8(
                SenderColumns::CIPHERTEXT + j,
                output,
                Block8(ctr.ciphertext[at]),
            )?;
        }
    }

    Ok(tb.build())
}

fn key_link(idx: Source, word: Col, selector: Col) -> PermutationCheckSpec {
    PermutationCheckSpec::new(
        vec![
            (idx, b"kappa_key_idx" as &[u8]),
            (Source::Column(word.index()), b"kappa_key_word" as &[u8]),
        ],
        Some(selector.index()),
    )
    .with_clock_waiver(KEY_LINK_WAIVER)
}

fn run<K: Kem>(label: &str, params: MlKemParams, message: &[u8])
where
    K::DecapsulationKey: Decapsulate,
{
    common::init(label);

    // The recipient's ml-kem key pair;
    // the sender sees only ek.
    let recipient = K::DecapsulationKey::generate();
    let ek = recipient.encapsulation_key().to_bytes().to_vec();

    let mut m = Zeroizing::new([0u8; 32]);
    let mut nonce = [0u8; NONCE_BYTES];

    OsRng.try_fill_bytes(&mut *m).unwrap();
    OsRng.try_fill_bytes(&mut nonce).unwrap();

    let blocks = message.len().div_ceil(BLOCK);

    println!("  Encapsulation key: {} bytes, hashed into H(ek)", ek.len());
    println!(
        "  Message:           {} bytes in {blocks} AES-256-CTR blocks, public",
        message.len()
    );

    let floor = Config::default().min_table_rows();

    let pipeline = MlKemChiplet::<F>::new(params, &[MlKemCall::Encaps]).expect("pipeline build");
    let aes = Aes256Chiplet::<F>::new(
        (blocks * AES_ROWS).next_power_of_two().max(floor),
        (blocks * SBOX_ROUNDS).next_power_of_two().max(floor),
        blocks,
    )
    .expect("aes build");

    let (traced, key, ctr, aes_traces) = common::phase("Trace Generation", || {
        let traced = pipeline
            .trace(&[MlKemInput::Encaps { ek: &ek, m: &m }])
            .expect("a well-formed encapsulation key");

        let MlKemOutput::Encaps { key, .. } = &traced.outputs[0] else {
            unreachable!()
        };

        let key = Zeroizing::new(*key);
        let ctr = Ctr::encrypt(&key, &nonce, message);
        let round_keys = Zeroizing::new(expand_key_256(&key));

        let calls: Vec<Aes256Call> = ctr
            .counters
            .iter()
            .map(|counter| Aes256Call {
                key: *key,
                plaintext: *counter,
                round_keys: *round_keys,
            })
            .collect();

        let aes_traces = aes.generate_traces(&calls).expect("aes trace");

        (traced, key, ctr, aes_traces)
    });

    let MlKemOutput::Encaps { c, .. } = &traced.outputs[0] else {
        unreachable!()
    };

    // The recipient decapsulates c with ml-kem and
    // decrypts the ciphertext with the K it recovers.
    let shared = recipient.decapsulate(&Ciphertext::<K>::try_from(c.as_slice()).unwrap());

    let mut received = Zeroizing::new([0u8; 32]);
    received.copy_from_slice(&shared);

    let decrypted = Ctr::encrypt(&received, &nonce, &ctr.ciphertext[..message.len()]);

    println!("  Ciphertext c:      {} bytes, public", c.len());
    println!(
        "  Recipient:         {} K and decrypts {} message",
        match *received == *key {
            true => "recovers",
            false => "misses",
        },
        match &decrypted.ciphertext[..message.len()] == message {
            true => "the",
            false => "a different",
        }
    );

    let layout = Layout::new(params);
    let words = &traced.words;
    let rows = words.len().max(2 * blocks).next_power_of_two();

    // Public: H(ek) and c from the host words, the counter
    // nonces, then each message byte with its ciphertext byte.
    let mut statement: Vec<(usize, usize, F)> = (layout.h..layout.h + SEED_WORDS)
        .chain(layout.c..layout.c + layout.c_words)
        .map(|row| (SenderColumns::WORD, row, F::from(u128::from(words[row]))))
        .collect();

    for (b, counter) in ctr.counters.iter().enumerate() {
        for (j, &byte) in counter[..NONCE_BYTES].iter().enumerate() {
            statement.push((SenderColumns::COUNTER + j, 2 * b, F::from(byte as u32)));
        }
    }

    for i in 0..message.len() {
        let (row, j) = (2 * (i / BLOCK) + 1, i % BLOCK);

        statement.push((
            SenderColumns::PLAINTEXT + j,
            row,
            F::from(ctr.plaintext[i] as u32),
        ));
        statement.push((
            SenderColumns::CIPHERTEXT + j,
            row,
            F::from(ctr.ciphertext[i] as u32),
        ));
    }

    println!(
        "  Public inputs:     {}, H(ek), c, the counter nonces, the message and its ciphertext",
        statement.len()
    );

    let program =
        build(&pipeline, &aes, &layout, message.len(), rows, &statement).expect("program build");
    let host = host_trace(words, &layout, &key, &ctr, rows).expect("host trace");

    let chiplets = traced.traces.into_iter().chain(aes_traces).collect();
    let public_inputs = statement.iter().map(|&(_, _, value)| value).collect();

    let instance = ProgramInstance::new(rows, public_inputs);
    let witness = ProgramWitness::new(host).with_chiplets(chiplets);

    let config = Config {
        zero_knowledge: common::zero_knowledge(),
        ..Config::default()
    };

    let mut blinding_seed = Zeroizing::new([0u8; 32]);
    OsRng.try_fill_bytes(&mut *blinding_seed).unwrap();

    let proof = common::phase("Proving", || {
        prove(
            b"MlKemSender_E2E",
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

    let mut verifier_transcript = Transcript::<H>::new(b"MlKemSender_E2E");
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
    let message = Zeroizing::new(match std::env::var("HEKATE_MESSAGE") {
        Ok(message) => message,
        Err(std::env::VarError::NotPresent) => DEFAULT_MESSAGE.to_owned(),
        Err(std::env::VarError::NotUnicode(_)) => {
            eprintln!("HEKATE_MESSAGE is not UTF-8 text");
            std::process::exit(1);
        }
    });

    if message.is_empty() {
        eprintln!("HEKATE_MESSAGE is empty: the sender needs at least one byte to encrypt");
        std::process::exit(1);
    }

    let message = message.as_bytes();

    match level.as_str() {
        "512" => run::<MlKem512>("ML-KEM-512 Sender", MlKemParams::ML_KEM_512, message),
        "768" => run::<MlKem768>("ML-KEM-768 Sender", MlKemParams::ML_KEM_768, message),
        "1024" => run::<MlKem1024>("ML-KEM-1024 Sender", MlKemParams::ML_KEM_1024, message),
        other => {
            eprintln!(
                "Usage: HEKATE_LEVEL=[512|768|1024] HEKATE_MESSAGE=<text> mlkem_sender \
                 (got {other:?})"
            );
            std::process::exit(1);
        }
    }
}
