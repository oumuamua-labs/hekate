// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::config::Config;
use hekate_core::errors::Error;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder, TraceColumn};
use hekate_crypto::DefaultHasher;
use hekate_crypto::transcript::Transcript;
use hekate_math::{Bit, Block16, Block32, Block128, HardwareField, TowerField};
use hekate_pqc::kem_select::{CipherPart, KemSelectChiplet, KemSelectStep};
use hekate_pqc::wiring::{PolyLabels, Stream, WORD_BUS_ID, WordValues, word_spec};
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::digest::program_id;
use hekate_program::{FixedShape, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_scribble::{MutationKind, ScribbleConfig, assert_all_caught_all_targets};
use hekate_sdk::preflight::{PreflightReport, TableId, preflight};
use hekate_verifier::HekateVerifier;

type F = Block128;
type H = DefaultHasher;

const WORDS: usize = 272;
const C1_WORDS: usize = 240;
const KEY_WORDS: usize = 8;

const HOST_LAYOUT: [ColumnType; 4] = [
    ColumnType::B16,
    ColumnType::B16,
    ColumnType::B32,
    ColumnType::Bit,
];

struct Case {
    chiplet: KemSelectChiplet<F>,
    step: KemSelectStep,
    c: Vec<u32>,
    c_prime: Vec<u32>,
    k_prime: Vec<u32>,
    k_bar: Vec<u32>,
}

struct Proof {
    program: CircuitProgram<F>,
    instance: ProgramInstance<F>,
    witness: ProgramWitness<F>,
}

impl Case {
    fn new(differ: Option<usize>) -> Self {
        let mut labels = PolyLabels::new();
        let mut stream = || labels.stream().unwrap();

        let parts = [C1_WORDS, WORDS - C1_WORDS]
            .map(|words| CipherPart {
                c: stream(),
                c_prime: stream(),
                relay: stream(),
                words,
            })
            .to_vec();

        let [k_prime, k_bar, k, valid] = [(); 4].map(|_| stream());

        let step = KemSelectStep {
            parts,
            k_prime,
            k_bar,
            k,
            valid,
        };

        let ct: Vec<u32> = (0..WORDS).map(junk).collect();
        let mut ct_prime = ct.clone();

        if let Some(j) = differ {
            ct_prime[j] ^= 1 << 7;
        }

        Case {
            chiplet: KemSelectChiplet::new(vec![step.clone()], height(step.rows())).unwrap(),
            step,
            c: ct,
            c_prime: ct_prime,
            k_prime: (0..KEY_WORDS).map(|i| junk(1000 + i)).collect(),
            k_bar: (0..KEY_WORDS).map(|i| junk(2000 + i)).collect(),
        }
    }

    fn parted<'a>(&'a self, list: &'a [u32]) -> impl Iterator<Item = (&'a CipherPart, &'a [u32])> {
        self.step.parts.iter().scan(0, move |at, part| {
            let chunk = &list[*at..*at + part.words];
            *at += part.words;

            Some((part, chunk))
        })
    }

    fn run(&self) -> (ColumnTrace, WordValues) {
        let mut words = WordValues::default();

        for ((part, c), (_, c_prime)) in self.parted(&self.c).zip(self.parted(&self.c_prime)) {
            words.insert(part.c, c.to_vec()).unwrap();
            words.insert(part.c_prime, c_prime.to_vec()).unwrap();
        }

        words
            .insert(self.step.k_prime, self.k_prime.clone())
            .unwrap();
        words.insert(self.step.k_bar, self.k_bar.clone()).unwrap();

        let trace = self.chiplet.trace(&mut words).unwrap();

        (trace, words)
    }

    fn tokens(&self, key: &[u32], valid: u32) -> Vec<(u16, u16, u32)> {
        let s = &self.step;
        let valid = [valid];

        let mut lists: Vec<(Stream, &[u32])> = Vec::new();

        for ((part, c), (_, c_prime)) in self.parted(&self.c).zip(self.parted(&self.c_prime)) {
            lists.extend([(part.c, c), (part.c_prime, c_prime), (part.relay, c)]);
        }

        lists.extend([
            (s.k_prime, self.k_prime.as_slice()),
            (s.k_bar, &self.k_bar),
            (s.k, key),
            (s.valid, &valid),
        ]);

        lists
            .into_iter()
            .flat_map(|(stream, list)| {
                list.iter()
                    .enumerate()
                    .map(move |(j, &w)| (stream.id(), j as u16, w))
            })
            .collect()
    }

    fn proof(&self, trace: ColumnTrace, tokens: &[(u16, u16, u32)]) -> Proof {
        let rows = height(tokens.len());
        let mut cx = Circuit::<F>::new("KemSelectHost", rows).unwrap();

        let cols = cx.schema(&HOST_LAYOUT);

        let pin = |values: Vec<u64>| {
            FixedShape::Sparse(
                values
                    .into_iter()
                    .enumerate()
                    .filter(|&(_, v)| v != 0)
                    .map(|(row, v)| (row, F::from(v as u128)))
                    .collect(),
            )
        };

        cx.fix(cols.at(0), pin(tokens.iter().map(|t| t.0 as u64).collect()));
        cx.fix(cols.at(1), pin(tokens.iter().map(|t| t.1 as u64).collect()));
        cx.fix(cols.at(3), pin(vec![1; tokens.len()]));

        cx.bus(WORD_BUS_ID, word_spec(0, 1, 2, 3));

        cx.attach(self.chiplet.def().unwrap());

        let mut tb = TraceBuilder::new(&HOST_LAYOUT, rows.trailing_zeros() as usize).unwrap();

        for (r, &(stream, j, w)) in tokens.iter().enumerate() {
            tb.set_b16(0, r, Block16(stream)).unwrap();
            tb.set_b16(1, r, Block16(j)).unwrap();
            tb.set_b32(2, r, Block32::from(w)).unwrap();
            tb.set_bit(3, r, Bit::ONE).unwrap();
        }

        Proof {
            program: cx.compile().unwrap(),
            instance: ProgramInstance::new(rows, Vec::new()),
            witness: ProgramWitness::new(tb.build()).with_chiplets(vec![trace]),
        }
    }
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
            b"KemSelectChiplet",
            &self.program,
            &self.instance,
            &self.witness,
            &config,
            [7; 32],
            None,
        )
        .unwrap();

        let mut transcript = Transcript::<H>::new(b"KemSelectChiplet");

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

    fn rejected(&self) -> bool {
        !self.accepted(false) && !self.accepted(true)
    }
}

fn assert_breaks_only(proof: &Proof, label: &str) {
    let report = proof.report();

    assert!(report.boundary_violations.is_empty());
    assert!(report.fixed_column_violations.is_empty());
    assert!(report.bus_diagnostics.is_empty());
    assert!(!report.constraint_violations.is_empty());

    for v in &report.constraint_violations {
        assert!(v.table == TableId::Chiplet(0));
        assert_eq!(v.label, Some(label));
    }

    assert!(proof.rejected());
}

fn set_word(trace: &mut ColumnTrace, col: usize, row: usize, value: u32) {
    let TraceColumn::B32(cells) = &mut trace.columns[col] else {
        panic!("forged words live in B32 columns");
    };

    cells[row] = Block32::from(value).to_hardware();
}

fn set_bit(trace: &mut ColumnTrace, col: usize, row: usize, on: bool) {
    let TraceColumn::Bit(cells) = &mut trace.columns[col] else {
        panic!("forged flags live in Bit columns");
    };

    cells[row] = Bit::from(on as u8);
}

fn height(rows: usize) -> usize {
    rows.next_power_of_two()
        .max(Config::prod().min_table_rows())
}

fn junk(i: usize) -> u32 {
    (i as u32).wrapping_mul(0x9e37_79b9) ^ 0x5a5a
}

#[test]
fn equal_ciphertexts_select_k_prime_and_prove() {
    let case = Case::new(None);
    let (trace, words) = case.run();

    assert_eq!(words.get(case.step.k).unwrap(), case.k_prime);
    assert_eq!(words.get(case.step.valid).unwrap(), [1]);

    let proof = case.proof(trace, &case.tokens(&case.k_prime, 1));

    assert!(proof.report().is_clean());
    assert!(proof.accepted(false));
    assert!(proof.accepted(true));
}

#[test]
fn differing_ciphertexts_select_k_bar_and_prove() {
    let case = Case::new(Some(WORDS - 1));
    let (trace, words) = case.run();

    assert_eq!(words.get(case.step.k).unwrap(), case.k_bar);
    assert_eq!(words.get(case.step.valid).unwrap(), [0]);

    let proof = case.proof(trace, &case.tokens(&case.k_bar, 0));

    assert!(proof.report().is_clean());
    assert!(proof.accepted(false));
    assert!(proof.accepted(true));
}

#[test]
fn hidden_difference_breaks_only_kem_select_zero() {
    let case = Case::new(Some(WORDS - 1));
    let (mut trace, _) = case.run();
    let ly = case.chiplet.layout();

    set_bit(&mut trace, ly.physical(ly.nz), WORDS - 1, false);
    set_word(&mut trace, ly.physical(ly.inv), WORDS - 1, 0);

    for r in WORDS..WORDS + KEY_WORDS + 1 {
        set_bit(&mut trace, ly.physical(ly.flag), r, false);
    }

    for (j, &k) in case.k_prime.iter().enumerate() {
        set_word(&mut trace, ly.physical(ly.word_c), WORDS + j, k);
    }

    set_word(&mut trace, ly.physical(ly.word_c), WORDS + KEY_WORDS, 1);

    let proof = case.proof(trace, &case.tokens(&case.k_prime, 1));

    assert_breaks_only(&proof, "kem_select_zero");
}

#[test]
fn ignored_difference_breaks_only_kem_select_flag() {
    let case = Case::new(Some(WORDS - 1));
    let (mut trace, _) = case.run();
    let ly = case.chiplet.layout();

    for r in WORDS..WORDS + KEY_WORDS + 1 {
        set_bit(&mut trace, ly.physical(ly.flag), r, false);
    }

    for (j, &k) in case.k_prime.iter().enumerate() {
        set_word(&mut trace, ly.physical(ly.word_c), WORDS + j, k);
    }

    set_word(&mut trace, ly.physical(ly.word_c), WORDS + KEY_WORDS, 1);

    let proof = case.proof(trace, &case.tokens(&case.k_prime, 1));

    assert_breaks_only(&proof, "kem_select_flag");
}

#[test]
fn k_prime_released_on_reported_difference_breaks_only_kem_select_out() {
    let case = Case::new(Some(WORDS - 1));
    let (mut trace, _) = case.run();

    let ly = case.chiplet.layout();

    for (j, &k) in case.k_prime.iter().enumerate() {
        set_word(&mut trace, ly.physical(ly.word_c), WORDS + j, k);
    }

    let proof = case.proof(trace, &case.tokens(&case.k_prime, 0));

    assert_breaks_only(&proof, "kem_select_out");
}

#[test]
fn phantom_difference_breaks_only_kem_select_nonzero() {
    let case = Case::new(None);
    let (mut trace, _) = case.run();

    let ly = case.chiplet.layout();

    let row = 5;

    set_bit(&mut trace, ly.physical(ly.nz), row, true);

    for r in row + 1..WORDS + KEY_WORDS + 1 {
        set_bit(&mut trace, ly.physical(ly.flag), r, true);
    }

    for (j, &k) in case.k_bar.iter().enumerate() {
        set_word(&mut trace, ly.physical(ly.word_c), WORDS + j, k);
    }

    set_word(&mut trace, ly.physical(ly.word_c), WORDS + KEY_WORDS, 0);

    let proof = case.proof(trace, &case.tokens(&case.k_bar, 0));

    assert_breaks_only(&proof, "kem_select_nonzero");
}

#[test]
fn relay_other_than_compared_c_breaks_only_kem_select_out() {
    let case = Case::new(None);
    let (mut trace, _) = case.run();

    let ly = case.chiplet.layout();

    let row = 5;
    let forged = case.c[row] ^ 1;

    set_word(&mut trace, ly.physical(ly.word_c), row, forged);

    let relay = case.step.parts[0].relay.id();

    let mut tokens = case.tokens(&case.k_prime, 1);
    let token = tokens
        .iter_mut()
        .find(|t| t.0 == relay && t.1 == row as u16)
        .unwrap();

    token.2 = forged;

    let proof = case.proof(trace, &tokens);

    assert_breaks_only(&proof, "kem_select_out");
}

#[test]
fn flag_set_on_first_row_breaks_only_kem_select_first() {
    let case = Case::new(None);
    let (mut trace, _) = case.run();

    let ly = case.chiplet.layout();

    for r in 0..WORDS + KEY_WORDS + 1 {
        set_bit(&mut trace, ly.physical(ly.flag), r, true);
    }

    for (j, &k) in case.k_bar.iter().enumerate() {
        set_word(&mut trace, ly.physical(ly.word_c), WORDS + j, k);
    }

    set_word(&mut trace, ly.physical(ly.word_c), WORDS + KEY_WORDS, 0);

    let proof = case.proof(trace, &case.tokens(&case.k_bar, 0));

    assert_breaks_only(&proof, "kem_select_first");
}

#[test]
fn key_split_on_equal_ciphertexts_breaks_only_kem_select_nz_gate() {
    let case = Case::new(None);
    let (mut trace, _) = case.run();

    let ly = case.chiplet.layout();

    let row = 3;

    set_bit(&mut trace, ly.physical(ly.nz), WORDS + row, true);

    for r in WORDS + row + 1..WORDS + KEY_WORDS + 1 {
        set_bit(&mut trace, ly.physical(ly.flag), r, true);
    }

    let key: Vec<u32> = case.k_prime[..=row]
        .iter()
        .chain(&case.k_bar[row + 1..])
        .copied()
        .collect();

    for (j, &k) in key.iter().enumerate() {
        set_word(&mut trace, ly.physical(ly.word_c), WORDS + j, k);
    }

    set_word(&mut trace, ly.physical(ly.word_c), WORDS + KEY_WORDS, 0);

    let proof = case.proof(trace, &case.tokens(&key, 0));

    assert_breaks_only(&proof, "kem_select_nz_gate");
}

#[test]
fn part_past_sixteen_bit_index_is_refused() {
    let mut labels = PolyLabels::new();
    let mut stream = || labels.stream().unwrap();

    let parts = vec![CipherPart {
        c: stream(),
        c_prime: stream(),
        relay: stream(),
        words: (1 << 16) + 1,
    }];

    let [k_prime, k_bar, k, valid] = [(); 4].map(|_| stream());

    let step = KemSelectStep {
        parts,
        k_prime,
        k_bar,
        k,
        valid,
    };

    assert_eq!(
        KemSelectChiplet::<F>::new(vec![step], 1 << 18).err(),
        Some(Error::Protocol {
            protocol: "kem_select_chiplet",
            message: "ciphertext part holds more words than a 16-bit index counts",
        })
    );
}

#[test]
fn scribble_kem_select_row_mutations_caught() {
    let case = Case::new(Some(WORDS / 2));
    let (trace, words) = case.run();

    let key = words.get(case.step.k).unwrap().to_vec();
    let proof = case.proof(trace, &case.tokens(&key, 0));

    assert_all_caught_all_targets(
        &proof.program,
        &proof.instance,
        &proof.witness,
        ScribbleConfig::default()
            .mutations([
                MutationKind::FlipSelector,
                MutationKind::SwapRows,
                MutationKind::DuplicateRow,
            ])
            .cases(64),
    );
}
