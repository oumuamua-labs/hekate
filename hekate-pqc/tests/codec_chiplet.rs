// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::config::Config;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder, TraceColumn};
use hekate_crypto::DefaultHasher;
use hekate_crypto::transcript::Transcript;
use hekate_math::{Bit, Block16, Block32, Block128, HardwareField, TowerField};
use hekate_pqc::codec::{CodecChiplet, CodecForgery, CodecStep};
use hekate_pqc::mldsa::{self, MlDsaParams};
use hekate_pqc::mlkem::{self, MlKemParams};
use hekate_pqc::wiring::{
    COEF_BUS_ID, HINT_BUS_ID, N, Poly, PolyLabels, PolyValues, Stream, WORD_BUS_ID, WordValues,
    coef_spec, hint_spec, word_spec,
};
use hekate_program::circuit::{Circuit, CircuitProgram, Col};
use hekate_program::digest::program_id;
use hekate_program::{FixedShape, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_scribble::{MutationKind, ScribbleConfig, assert_all_caught_all_targets};
use hekate_sdk::preflight::{PreflightReport, TableId, preflight};
use hekate_verifier::HekateVerifier;

type F = Block128;
type H = DefaultHasher;

const T1_BITS: usize = 10;
const KEM_BITS: usize = 12;

const HOST_LAYOUT: [ColumnType; 11] = [
    ColumnType::B16,
    ColumnType::B16,
    ColumnType::B32,
    ColumnType::Bit,
    ColumnType::B16,
    ColumnType::B16,
    ColumnType::Bit,
    ColumnType::B16,
    ColumnType::B16,
    ColumnType::B32,
    ColumnType::Bit,
];

#[derive(Clone, Copy)]
enum KemKind {
    Ek,
    Dk,
    Encode12,
    Decompress(u32),
    Compress(u32),
}

#[derive(Clone, Copy)]
enum Cell {
    Coef(usize),
    Word,
}

#[derive(Default)]
struct Statement {
    coefs: Vec<(u16, u16, u32)>,
    hints: Vec<(u16, u16)>,
    words: Vec<(u16, u16, u32)>,
}

#[derive(Clone)]
struct DsaEncoding {
    t1: Vec<[u32; N]>,
    y: Vec<[u32; N]>,
    hint: Vec<u8>,
}

impl DsaEncoding {
    fn words(&self, case: &DsaCase) -> WordValues {
        let width = z_width(&case.params);
        let mut words = WordValues::default();

        words
            .insert(case.streams[0], pack(self.t1.iter().flatten(), T1_BITS))
            .unwrap();
        words
            .insert(case.streams[1], pack(self.y.iter().flatten(), width))
            .unwrap();
        words
            .insert(
                case.streams[2],
                self.hint
                    .chunks(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            )
            .unwrap();

        words
    }
}

struct DsaCase {
    params: MlDsaParams,
    t1: Vec<Poly>,
    z: Vec<Poly>,
    streams: [Stream; 3],
    encoding: DsaEncoding,
    hints: Vec<[bool; N]>,
    chiplet: CodecChiplet<F>,
}

impl DsaCase {
    fn new(params: MlDsaParams) -> Self {
        let mut labels = PolyLabels::new();

        let t1: Vec<Poly> = (0..params.k()).map(|_| labels.fresh().unwrap()).collect();
        let z: Vec<Poly> = (0..params.l()).map(|_| labels.fresh().unwrap()).collect();

        let streams = [
            labels.stream().unwrap(),
            labels.stream().unwrap(),
            labels.stream().unwrap(),
        ];

        let (g1, beta) = (params.gamma1(), params.beta());

        let mut seed = 0xc0de_u64 ^ g1 as u64;

        let t1_values = (0..params.k())
            .map(|_| {
                core::array::from_fn(|j| match j {
                    0 => 0,
                    1 => (1 << T1_BITS) - 1,
                    _ => (next(&mut seed) % (1 << T1_BITS)) as u32,
                })
            })
            .collect();

        let corners = [beta + 1, 2 * g1 - beta - 1, g1, g1 + 1];
        let span = (2 * g1 - 2 * beta - 1) as u64;

        let y = (0..params.l())
            .map(|_| {
                core::array::from_fn(|j| match corners.get(j) {
                    Some(&c) => c,
                    None => beta + 1 + (next(&mut seed) % span) as u32,
                })
            })
            .collect();

        let hints = hint_pattern(&params, &mut seed);

        let steps = vec![
            CodecStep::t1(streams[0], t1.clone()),
            CodecStep::z(&params, streams[1], z.clone()),
            CodecStep::hint(&params, streams[2], 0),
        ];

        Self {
            params,
            t1,
            z,
            streams,
            encoding: DsaEncoding {
                t1: t1_values,
                y,
                hint: hint_bit_pack(&params, &hints),
            },
            hints,
            chiplet: CodecChiplet::new(steps.clone(), table_rows(&steps)).unwrap(),
        }
    }

    fn expected(&self) -> Vec<(Poly, [u32; N])> {
        let g1 = self.params.gamma1() as i64;

        let t1 = self
            .t1
            .iter()
            .zip(&self.encoding.t1)
            .map(|(&poly, v)| (poly, v.map(|c| c << 13)));

        let z = self.z.iter().zip(&self.encoding.y).map(|(&poly, y)| {
            (
                poly,
                y.map(|y| (g1 - y as i64).rem_euclid(mldsa::Q as i64) as u32),
            )
        });

        t1.chain(z).collect()
    }

    fn statement(&self) -> Statement {
        let mut st = Statement {
            words: word_tokens(&self.streams, &self.encoding.words(self)),
            ..Statement::default()
        };

        for (poly, coeffs) in self.expected() {
            for (pos, &value) in coeffs.iter().enumerate() {
                st.coefs.push((poly.id(), pos as u16, value));
            }
        }

        for (i, h) in self.hints.iter().enumerate() {
            for (pos, _) in h.iter().enumerate().filter(|&(_, &bit)| bit) {
                st.hints.push((i as u16 + 1, pos as u16));
            }
        }

        if st.hints.len() % 2 == 1 {
            st.hints.push((0, 0));
        }

        st
    }

    fn accomplice(
        &self,
        words: &WordValues,
        values: &PolyValues,
        trace: &ColumnTrace,
    ) -> Statement {
        let mut st = Statement {
            words: word_tokens(&self.streams, words),
            hints: emitted(&self.chiplet, trace),
            ..Statement::default()
        };

        for &poly in self.t1.iter().chain(&self.z) {
            for (pos, &value) in values.get(poly).unwrap().iter().enumerate() {
                st.coefs.push((poly.id(), pos as u16, value));
            }
        }

        st
    }
}

#[derive(Clone)]
struct KemPart {
    kind: KemKind,
    stream: Stream,
    poly: Poly,
    values: [u32; N],
}

impl KemPart {
    fn step(&self) -> CodecStep {
        let (stream, polys) = (self.stream, vec![self.poly]);

        match self.kind {
            KemKind::Ek => CodecStep::decode12_canonical(stream, polys),
            KemKind::Dk => CodecStep::decode12(stream, polys),
            KemKind::Encode12 => CodecStep::encode12(polys, stream),
            KemKind::Decompress(d) => CodecStep::decompress(d, stream, polys).unwrap(),
            KemKind::Compress(d) => CodecStep::compress(d, polys, stream).unwrap(),
        }
    }

    fn decodes(&self) -> bool {
        matches!(
            self.kind,
            KemKind::Ek | KemKind::Dk | KemKind::Decompress(_)
        )
    }

    fn coefficients(&self) -> [u32; N] {
        match self.kind {
            KemKind::Ek | KemKind::Encode12 | KemKind::Compress(_) => self.values,
            KemKind::Dk => self.values.map(|b| b % mlkem::Q),
            KemKind::Decompress(d) => self.values.map(|y| fips_decompress(d, y)),
        }
    }

    fn words(&self) -> Vec<u32> {
        let (fields, width) = match self.kind {
            KemKind::Compress(d) => (self.values.map(|x| fips_compress(d, x)), d as usize),
            KemKind::Decompress(d) => (self.values, d as usize),
            _ => (self.values, KEM_BITS),
        };

        pack(fields.iter(), width)
    }
}

struct KemCase {
    parts: Vec<KemPart>,
    chiplet: CodecChiplet<F>,
}

impl KemCase {
    fn new(params: MlKemParams) -> Self {
        let mut labels = PolyLabels::new();
        let mut seed = 0x5eed_u64 ^ params.du() as u64;

        let (q, du, dv) = (mlkem::Q, params.du(), params.dv());

        let half = [0, q - 1, q / 2, q.div_ceil(2)];
        let quarter = [q / 4, q.div_ceil(4), 3 * q / 4, 3 * q / 4 + 1];

        let specs: [(KemKind, u32, &[u32]); 9] = [
            (KemKind::Ek, q, &[0, q - 1]),
            (
                KemKind::Dk,
                1 << KEM_BITS,
                &[0, q - 1, q, (1 << KEM_BITS) - 1],
            ),
            (KemKind::Encode12, q, &[0, q - 1]),
            (KemKind::Decompress(du), 1 << du, &[0, (1 << du) - 1]),
            (KemKind::Decompress(dv), 1 << dv, &[0, (1 << dv) - 1]),
            (KemKind::Decompress(1), 2, &[0, 1]),
            (KemKind::Compress(du), q, &half),
            (KemKind::Compress(dv), q, &half),
            (KemKind::Compress(1), q, &quarter),
        ];

        let parts: Vec<KemPart> = specs
            .iter()
            .map(|&(kind, bound, corners)| KemPart {
                kind,
                stream: labels.stream().unwrap(),
                poly: labels.fresh().unwrap(),
                values: core::array::from_fn(|i| match corners.get(i) {
                    Some(&c) => c,
                    None => (next(&mut seed) % bound as u64) as u32,
                }),
            })
            .collect();

        let steps: Vec<CodecStep> = parts.iter().map(KemPart::step).collect();

        Self {
            chiplet: CodecChiplet::new(steps.clone(), table_rows(&steps)).unwrap(),
            parts,
        }
    }

    fn origin(&self, index: usize) -> usize {
        self.parts[..index].iter().map(|p| p.step().rows()).sum()
    }
}

struct Proof {
    program: CircuitProgram<F>,
    instance: ProgramInstance<F>,
    witness: ProgramWitness<F>,
}

impl Proof {
    fn new(chiplet: &CodecChiplet<F>, trace: ColumnTrace, st: &Statement) -> Self {
        let rows = height(st.coefs.len().max(st.hints.len()).max(st.words.len()));

        Self {
            program: host_program(chiplet, st, rows),
            instance: ProgramInstance::new(rows, Vec::new()),
            witness: ProgramWitness::new(host_trace(st, rows)).with_chiplets(vec![trace]),
        }
    }

    fn report(&self) -> PreflightReport<F> {
        preflight(&self.program, &self.instance, &self.witness).unwrap()
    }

    fn accepted(&self, zero_knowledge: bool) -> bool {
        let config = Config {
            zero_knowledge,
            ..Config::prod()
        };

        let proof = prove(
            b"CodecChiplet",
            &self.program,
            &self.instance,
            &self.witness,
            &config,
            [7; 32],
            None,
        )
        .unwrap();

        let mut transcript = Transcript::<H>::new(b"CodecChiplet");

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

fn dsa_honest_case(params: MlDsaParams) {
    let case = DsaCase::new(params);

    let mut words = case.encoding.words(&case);
    let mut values = PolyValues::default();

    let (trace, hints) = case.chiplet.trace(&mut words, &mut values).unwrap();

    for (poly, coeffs) in case.expected() {
        assert_eq!(values.get(poly).unwrap(), &coeffs);
    }

    assert_eq!(*hints, case.hints);

    let proof = Proof::new(&case.chiplet, trace, &case.statement());

    assert!(proof.report().is_clean());
    assert!(proof.accepted(false));
    assert!(proof.accepted(true));
}

fn kem_honest_case(params: MlKemParams) {
    let case = KemCase::new(params);

    let (mut words, mut values) = kem_inputs(&case.parts);

    let (trace, hints) = case.chiplet.trace(&mut words, &mut values).unwrap();

    assert!(hints.is_empty());

    for part in &case.parts {
        assert_eq!(values.get(part.poly).unwrap(), &part.coefficients());
        assert_eq!(words.get(part.stream).unwrap(), part.words().as_slice());
    }

    let proof = Proof::new(&case.chiplet, trace, &kem_statement(&case.parts));

    assert!(proof.report().is_clean());
    assert!(proof.accepted(false));
    assert!(proof.accepted(true));
}

fn assert_malformed_breaks_only(case: &DsaCase, enc: &DsaEncoding, label: &str) {
    let mut words = enc.words(case);

    assert!(
        case.chiplet
            .trace(&mut words, &mut PolyValues::default())
            .is_err()
    );

    let mut values = PolyValues::default();

    let (trace, _) = case
        .chiplet
        .trace_forged(&mut words, &mut values, &[])
        .unwrap();

    let st = case.accomplice(&words, &values, &trace);

    assert_breaks_only(&case.chiplet, trace, &st, label);
}

fn assert_breaks_only(chiplet: &CodecChiplet<F>, trace: ColumnTrace, st: &Statement, label: &str) {
    let proof = Proof::new(chiplet, trace, st);
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

fn assert_bus_alone_rejects(chiplet: &CodecChiplet<F>, trace: ColumnTrace, st: &Statement) {
    let proof = Proof::new(chiplet, trace, st);
    let report = proof.report();

    assert!(report.constraint_violations.is_empty());
    assert!(report.fixed_column_violations.is_empty());
    assert!(!report.bus_diagnostics.is_empty());
    assert!(proof.rejected());
}

fn host_program(chiplet: &CodecChiplet<F>, st: &Statement, rows: usize) -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("CodecHost", rows).unwrap();
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

    cx.fix(
        cols.at(0),
        pin(st.coefs.iter().map(|c| c.0 as u64).collect()),
    );
    cx.fix(
        cols.at(1),
        pin(st.coefs.iter().map(|c| c.1 as u64).collect()),
    );
    cx.fix(cols.at(3), pin(vec![1; st.coefs.len()]));

    cx.fix(
        cols.at(4),
        pin(st.hints.iter().map(|h| h.0 as u64).collect()),
    );
    cx.fix(
        cols.at(5),
        pin(st.hints.iter().map(|h| h.1 as u64).collect()),
    );
    cx.fix(cols.at(6), pin(vec![1; st.hints.len()]));

    cx.fix(
        cols.at(7),
        pin(st.words.iter().map(|w| w.0 as u64).collect()),
    );
    cx.fix(
        cols.at(8),
        pin(st.words.iter().map(|w| w.1 as u64).collect()),
    );
    cx.fix(cols.at(10), pin(vec![1; st.words.len()]));

    cx.bus(
        COEF_BUS_ID,
        coef_spec(
            cols.at(0).index(),
            cols.at(1).index(),
            cols.at(2).index(),
            cols.at(3).index(),
        ),
    );

    if chiplet.layout().hint_cols.is_some() {
        cx.bus(
            HINT_BUS_ID,
            hint_spec(cols.at(4).index(), cols.at(5).index(), cols.at(6).index()),
        );
    }

    cx.bus(
        WORD_BUS_ID,
        word_spec(
            cols.at(7).index(),
            cols.at(8).index(),
            cols.at(9).index(),
            cols.at(10).index(),
        ),
    );

    cx.attach(chiplet.def().unwrap());

    cx.compile().unwrap()
}

fn host_trace(st: &Statement, rows: usize) -> ColumnTrace {
    let mut tb = TraceBuilder::new(&HOST_LAYOUT, rows.trailing_zeros() as usize).unwrap();

    for (r, &(poly, pos, value)) in st.coefs.iter().enumerate() {
        tb.set_b16(0, r, Block16(poly)).unwrap();
        tb.set_b16(1, r, Block16(pos)).unwrap();
        tb.set_b32(2, r, Block32::from(value)).unwrap();
        tb.set_bit(3, r, Bit::ONE).unwrap();
    }

    for (r, &(poly, pos)) in st.hints.iter().enumerate() {
        tb.set_b16(4, r, Block16(poly)).unwrap();
        tb.set_b16(5, r, Block16(pos)).unwrap();
        tb.set_bit(6, r, Bit::ONE).unwrap();
    }

    for (r, &(stream, index, word)) in st.words.iter().enumerate() {
        tb.set_b16(7, r, Block16(stream)).unwrap();
        tb.set_b16(8, r, Block16(index)).unwrap();
        tb.set_b32(9, r, Block32::from(word)).unwrap();
        tb.set_bit(10, r, Bit::ONE).unwrap();
    }

    tb.build()
}

fn emitted(chiplet: &CodecChiplet<F>, trace: &ColumnTrace) -> Vec<(u16, u16)> {
    let ly = chiplet.layout();
    let hc = ly.hint_cols.as_ref().unwrap();

    let TraceColumn::Bit(kind) = &trace.columns[ly.physical(hc.kind).unwrap()] else {
        panic!("the hint kind flag is a Bit column");
    };

    let label = |col: Col, row: usize| match &trace.columns[ly.physical(col).unwrap()] {
        TraceColumn::B16(cells) => cells[row].to_tower().0,
        _ => panic!("hint labels are B16 columns"),
    };

    let mut tokens = Vec::new();
    for (row, _) in kind.iter().enumerate().filter(|&(_, &on)| on == Bit::ONE) {
        for t in 0..hc.hp.len() {
            tokens.push((label(hc.hp.at(t), row), label(hc.hm.at(t), row)));
        }
    }

    tokens
}

fn forge(trace: &mut ColumnTrace, col: usize, row: usize) -> u32 {
    let TraceColumn::B32(cells) = &mut trace.columns[col] else {
        panic!("coefficients and words are B32 columns");
    };

    let forged = cells[row].to_tower().0 ^ 1;
    cells[row] = Block32::from(forged).to_hardware();

    forged
}

fn kem_inputs(parts: &[KemPart]) -> (WordValues, PolyValues) {
    let mut words = WordValues::default();
    let mut values = PolyValues::default();

    for part in parts {
        match part.decodes() {
            true => words.insert(part.stream, part.words()).unwrap(),
            false => values.insert(part.poly, part.values).unwrap(),
        }
    }

    (words, values)
}

fn kem_statement(parts: &[KemPart]) -> Statement {
    let mut st = Statement::default();
    for part in parts {
        for (pos, &value) in part.coefficients().iter().enumerate() {
            st.coefs.push((part.poly.id(), pos as u16, value));
        }

        for (i, &word) in part.words().iter().enumerate() {
            st.words.push((part.stream.id(), i as u16, word));
        }
    }

    st
}

fn kem_accomplice(parts: &[KemPart], words: &WordValues, values: &PolyValues) -> Statement {
    let streams: Vec<Stream> = parts.iter().map(|p| p.stream).collect();

    let mut st = Statement {
        words: word_tokens(&streams, words),
        ..Statement::default()
    };

    for part in parts {
        for (pos, &value) in values.get(part.poly).unwrap().iter().enumerate() {
            st.coefs.push((part.poly.id(), pos as u16, value));
        }
    }

    st
}

fn word_tokens(streams: &[Stream], words: &WordValues) -> Vec<(u16, u16, u32)> {
    streams
        .iter()
        .flat_map(|&stream| {
            words
                .get(stream)
                .unwrap()
                .iter()
                .enumerate()
                .map(move |(i, &word)| (stream.id(), i as u16, word))
        })
        .collect()
}

fn table_rows(steps: &[CodecStep]) -> usize {
    height(steps.iter().map(CodecStep::rows).sum())
}

fn height(rows: usize) -> usize {
    rows.next_power_of_two()
        .max(Config::prod().min_table_rows())
}

fn hint_pattern(params: &MlDsaParams, seed: &mut u64) -> Vec<[bool; N]> {
    let mut h = vec![[false; N]; params.k()];

    h[0][0] = true;
    h[0][N - 1] = true;

    let mut set = 2;
    while set < params.omega() - 3 {
        let i = (next(seed) % params.k() as u64) as usize;
        let pos = (next(seed) % N as u64) as usize;

        if i != 1 && !h[i][pos] {
            h[i][pos] = true;
            set += 1;
        }
    }

    h
}

fn hint_bit_pack(params: &MlDsaParams, h: &[[bool; N]]) -> Vec<u8> {
    let omega = params.omega();

    let mut y = vec![0u8; (omega + params.k()).next_multiple_of(4)];
    let mut index = 0;

    for (i, poly) in h.iter().enumerate() {
        for (pos, _) in poly.iter().enumerate().filter(|&(_, &bit)| bit) {
            y[index] = pos as u8;
            index += 1;
        }

        y[omega + i] = index as u8;
    }

    y
}

fn fips_compress(d: u32, x: u32) -> u32 {
    let q = mlkem::Q as u64;

    ((((x as u64) << (d + 1)) + q) / (2 * q) % (1 << d)) as u32
}

fn fips_decompress(d: u32, y: u32) -> u32 {
    ((2 * mlkem::Q as u64 * y as u64 + (1 << d)) >> (d + 1)) as u32
}

fn pack<'a>(values: impl Iterator<Item = &'a u32>, width: usize) -> Vec<u32> {
    let mut words = Vec::new();
    let mut at = 0;

    for &v in values {
        for t in 0..width {
            if at % 32 == 0 {
                words.push(0);
            }

            words[at / 32] |= ((v >> t) & 1) << (at % 32);
            at += 1;
        }
    }

    words
}

fn z_width(params: &MlDsaParams) -> usize {
    32 - (2 * params.gamma1() - 1).leading_zeros() as usize
}

fn next(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);

    *seed >> 33
}

#[test]
fn ml_dsa_44_decoding_matches_fips_and_proves() {
    dsa_honest_case(MlDsaParams::ML_DSA_44);
}

#[test]
fn ml_dsa_65_decoding_matches_fips_and_proves() {
    dsa_honest_case(MlDsaParams::ML_DSA_65);
}

#[test]
fn ml_dsa_87_decoding_matches_fips_and_proves() {
    dsa_honest_case(MlDsaParams::ML_DSA_87);
}

#[test]
fn ml_kem_768_coding_matches_fips_and_proves() {
    kem_honest_case(MlKemParams::ML_KEM_768);
}

#[test]
fn ml_kem_1024_coding_matches_fips_and_proves() {
    kem_honest_case(MlKemParams::ML_KEM_1024);
}

#[test]
fn z_outside_norm_bound_breaks_only_its_constraint() {
    let case = DsaCase::new(MlDsaParams::ML_DSA_44);
    let (g1, beta) = (case.params.gamma1(), case.params.beta());

    for (poly, pos, y, label) in [
        (0, 5, beta, "codec_z_low"),
        (1, 7, 2 * g1 - beta, "codec_z_high"),
    ] {
        let mut enc = case.encoding.clone();
        enc.y[poly][pos] = y;

        assert_malformed_breaks_only(&case, &enc, label);
    }
}

#[test]
fn z_lifted_by_q_breaks_only_z_range() {
    let case = DsaCase::new(MlDsaParams::ML_DSA_44);

    let mut words = case.encoding.words(&case);
    let mut values = PolyValues::default();

    let row = CodecStep::t1(case.streams[0], case.t1.clone()).rows();

    let (trace, _) = case
        .chiplet
        .trace_forged(
            &mut words,
            &mut values,
            &[CodecForgery::Lifted { row, field: 2 }],
        )
        .unwrap();

    let st = case.accomplice(&words, &values, &trace);

    assert_breaks_only(&case.chiplet, trace, &st, "codec_z_range");
}

#[test]
fn malformed_hint_breaks_only_its_constraint() {
    let case = DsaCase::new(MlDsaParams::ML_DSA_65);
    let (omega, k) = (case.params.omega(), case.params.k());

    let increasing = |counts: &[usize]| {
        let mut enc = case.encoding.clone();
        enc.hint.fill(0);

        for s in 0..counts[k - 1].min(omega) {
            enc.hint[s] = 10 + s as u8;
        }

        for (i, &count) in counts.iter().enumerate() {
            enc.hint[omega + i] = count as u8;
        }

        enc
    };

    let decreasing: Vec<usize> = (0..k)
        .map(|i| match i {
            0 => 4,
            1 => 2,
            _ => 2 * (i + 1),
        })
        .collect();

    let beyond: Vec<usize> = (0..k)
        .map(|i| match i + 1 < k {
            true => 2 * (i + 1),
            false => omega + 1,
        })
        .collect();

    let mut repeated = case.encoding.clone();
    repeated.hint[1] = repeated.hint[0];

    assert!(case.encoding.hint[omega] >= 5);

    let mut across = case.encoding.clone();
    across.hint[4] = across.hint[3];

    let mut unused = case.encoding.clone();
    unused.hint[omega - 3] = 7;

    let mut padded = case.encoding.clone();
    padded.hint[omega + k + 1] = 1;

    for (enc, label) in [
        (repeated, "codec_hint_order"),
        (across, "codec_hint_order"),
        (increasing(&decreasing), "codec_hint_counts"),
        (increasing(&beyond), "codec_hint_bound"),
        (unused, "codec_hint_unused"),
        (padded, "codec_hint_pad"),
    ] {
        assert_malformed_breaks_only(&case, &enc, label);
    }
}

#[test]
fn count_moved_on_index_row_breaks_only_codec_hint_cont() {
    let case = DsaCase::new(MlDsaParams::ML_DSA_44);
    let (omega, k) = (case.params.omega(), case.params.k());

    let mut enc = case.encoding.clone();
    enc.hint.fill(0);
    enc.hint[..8].copy_from_slice(&[10, 20, 30, 40, 50, 60, 70, 80]);
    enc.hint[omega..omega + k].copy_from_slice(&[4, 8, 8, 8]);

    let origin = CodecStep::t1(case.streams[0], case.t1.clone()).rows()
        + CodecStep::z(&case.params, case.streams[1], case.z.clone()).rows();

    let forgery = CodecForgery::Count {
        row: origin + 1,
        poly: 0,
        value: 8,
    };

    let mut words = enc.words(&case);
    let mut values = PolyValues::default();

    let (trace, _) = case
        .chiplet
        .trace_forged(&mut words, &mut values, &[forgery])
        .unwrap();

    let st = case.accomplice(&words, &values, &trace);

    assert_breaks_only(&case.chiplet, trace, &st, "codec_hint_cont");
}

#[test]
fn count_held_off_its_byte_breaks_only_codec_hint_count() {
    let case = DsaCase::new(MlDsaParams::ML_DSA_44);
    let (omega, k) = (case.params.omega(), case.params.k());

    let mut enc = case.encoding.clone();
    enc.hint.fill(0);
    enc.hint[..8].copy_from_slice(&[10, 20, 30, 40, 50, 60, 70, 80]);
    enc.hint[omega..omega + k].copy_from_slice(&[4, 8, 8, 8]);

    let origin = CodecStep::t1(case.streams[0], case.t1.clone()).rows()
        + CodecStep::z(&case.params, case.streams[1], case.z.clone()).rows();
    let rows = CodecStep::hint(&case.params, case.streams[2], 0).rows();

    let forgeries: Vec<CodecForgery> = (origin..origin + rows)
        .map(|row| CodecForgery::Count {
            row,
            poly: 0,
            value: 5,
        })
        .collect();

    let mut words = enc.words(&case);
    let mut values = PolyValues::default();

    let (trace, _) = case
        .chiplet
        .trace_forged(&mut words, &mut values, &forgeries)
        .unwrap();

    let st = case.accomplice(&words, &values, &trace);

    assert_breaks_only(&case.chiplet, trace, &st, "codec_hint_count");
}

#[test]
fn forged_coefficient_breaks_only_its_link() {
    let case = DsaCase::new(MlDsaParams::ML_DSA_44);
    let mut words = case.encoding.words(&case);

    let ly = case.chiplet.layout();

    let t1_rows = CodecStep::t1(case.streams[0], case.t1.clone()).rows();

    for (poly, row, j, label) in [
        (case.t1[0], 0, 3, "codec_t1"),
        (case.z[0], t1_rows, 2, "codec_z_coef"),
    ] {
        let mut values = PolyValues::default();

        let (mut trace, _) = case.chiplet.trace(&mut words, &mut values).unwrap();

        let forged = forge(&mut trace, ly.physical(ly.coef.at(j)).unwrap(), row);

        let mut st = case.accomplice(&words, &values, &trace);
        let read = st
            .coefs
            .iter_mut()
            .find(|c| c.0 == poly.id() && c.1 == j as u16)
            .unwrap();

        read.2 = forged;

        assert_breaks_only(&case.chiplet, trace, &st, label);
    }
}

#[test]
fn host_supplying_different_word_is_caught_by_bus_alone() {
    let case = DsaCase::new(MlDsaParams::ML_DSA_87);
    let mut words = case.encoding.words(&case);

    let (trace, _) = case
        .chiplet
        .trace(&mut words, &mut PolyValues::default())
        .unwrap();

    let mut st = case.statement();
    st.words[3].2 ^= 1 << 9;

    assert_bus_alone_rejects(&case.chiplet, trace, &st);
}

#[test]
fn hint_key_without_its_bit_is_caught_by_bus_alone() {
    let case = DsaCase::new(MlDsaParams::ML_DSA_65);
    let mut words = case.encoding.words(&case);

    let (trace, _) = case
        .chiplet
        .trace(&mut words, &mut PolyValues::default())
        .unwrap();

    let unset = case.hints[0].iter().position(|&h| !h).unwrap();

    let mut st = case.statement();
    st.hints[0] = (1, unset as u16);

    assert_bus_alone_rejects(&case.chiplet, trace, &st);
}

#[test]
fn non_canonical_kem_coefficients_are_refused() {
    let case = KemCase::new(MlKemParams::ML_KEM_768);

    for index in [0, 2, 6] {
        let mut parts = case.parts.clone();
        parts[index].values[5] = mlkem::Q;

        let (mut words, mut values) = kem_inputs(&parts);

        assert!(case.chiplet.trace(&mut words, &mut values).is_err());
    }
}

#[test]
fn ek_coefficient_at_or_above_q_breaks_only_modulus_check() {
    let case = KemCase::new(MlKemParams::ML_KEM_768);

    for b in [mlkem::Q, (1 << KEM_BITS) - 1] {
        let mut parts = case.parts.clone();
        parts[0].values[5] = b;

        let (mut words, mut values) = kem_inputs(&parts);

        let (trace, _) = case
            .chiplet
            .trace_forged(&mut words, &mut values, &[])
            .unwrap();

        let st = kem_accomplice(&parts, &words, &values);

        assert_breaks_only(&case.chiplet, trace, &st, "codec_kem_modulus");
    }
}

#[test]
fn unreduced_12_bit_field_breaks_only_kem_range() {
    let case = KemCase::new(MlKemParams::ML_KEM_768);

    let mut ek = case.parts.clone();
    ek[0].values[5] = mlkem::Q;

    for (parts, row, field) in [(&ek, case.origin(0), 5), (&case.parts, case.origin(1), 2)] {
        let (mut words, mut values) = kem_inputs(parts);

        let (trace, _) = case
            .chiplet
            .trace_forged(
                &mut words,
                &mut values,
                &[CodecForgery::Unreduced { row, field }],
            )
            .unwrap();

        let st = kem_accomplice(parts, &words, &values);

        assert_breaks_only(&case.chiplet, trace, &st, "codec_kem_range");
    }
}

#[test]
fn compress_one_quotient_short_breaks_only_compress_range() {
    let case = KemCase::new(MlKemParams::ML_KEM_768);

    let (mut words, mut values) = kem_inputs(&case.parts);

    let (trace, _) = case
        .chiplet
        .trace_forged(
            &mut words,
            &mut values,
            &[CodecForgery::Borrowed {
                row: case.origin(6),
                field: 1,
            }],
        )
        .unwrap();

    let st = kem_accomplice(&case.parts, &words, &values);

    assert_breaks_only(&case.chiplet, trace, &st, "codec_compress_range");
}

#[test]
fn forged_kem_output_breaks_only_its_link() {
    let case = KemCase::new(MlKemParams::ML_KEM_768);
    let ly = case.chiplet.layout();

    for (index, cell, label) in [
        (1, Cell::Coef(3), "codec_kem_byte12"),
        (3, Cell::Coef(2), "codec_kem_decompress"),
        (5, Cell::Coef(7), "codec_kem_decompress"),
        (6, Cell::Word, "codec_kem_compress"),
        (8, Cell::Word, "codec_kem_compress"),
    ] {
        let (mut words, mut values) = kem_inputs(&case.parts);

        let (mut trace, _) = case.chiplet.trace(&mut words, &mut values).unwrap();

        let part = &case.parts[index];
        let row = case.origin(index);

        let mut st = kem_accomplice(&case.parts, &words, &values);

        match cell {
            Cell::Coef(j) => {
                let forged = forge(&mut trace, ly.physical(ly.coef.at(j)).unwrap(), row);
                let read = st
                    .coefs
                    .iter_mut()
                    .find(|c| c.0 == part.poly.id() && c.1 == j as u16)
                    .unwrap();

                read.2 = forged;
            }
            Cell::Word => {
                let forged = forge(&mut trace, ly.physical(ly.word.at(0)).unwrap(), row);
                let read = st
                    .words
                    .iter_mut()
                    .find(|w| w.0 == part.stream.id() && w.1 == 0)
                    .unwrap();

                read.2 = forged;
            }
        }

        assert_breaks_only(&case.chiplet, trace, &st, label);
    }
}

#[test]
fn host_disagreeing_on_kem_tokens_is_caught_by_bus_alone() {
    let case = KemCase::new(MlKemParams::ML_KEM_768);

    for (index, cell) in [(6, Cell::Coef(0)), (3, Cell::Word)] {
        let (mut words, mut values) = kem_inputs(&case.parts);

        let (trace, _) = case.chiplet.trace(&mut words, &mut values).unwrap();

        let part = &case.parts[index];

        let mut st = kem_statement(&case.parts);

        match cell {
            Cell::Coef(j) => {
                let read = st
                    .coefs
                    .iter_mut()
                    .find(|c| c.0 == part.poly.id() && c.1 == j as u16)
                    .unwrap();

                read.2 ^= 1;
            }
            Cell::Word => {
                let read = st
                    .words
                    .iter_mut()
                    .find(|w| w.0 == part.stream.id() && w.1 == 0)
                    .unwrap();

                read.2 ^= 1 << 5;
            }
        }

        assert_bus_alone_rejects(&case.chiplet, trace, &st);
    }
}

#[test]
fn twinned_compress1_feeds_decompress1_and_proves() {
    let mut labels = PolyLabels::new();

    let (w, mu) = (labels.fresh().unwrap(), labels.fresh().unwrap());
    let (m, twin) = (labels.stream().unwrap(), labels.stream().unwrap());

    let steps = vec![
        CodecStep::compress(1, vec![w], m).unwrap().with_twin(twin),
        CodecStep::decompress(1, twin, vec![mu]).unwrap(),
    ];

    let chiplet = CodecChiplet::<F>::new(steps.clone(), table_rows(&steps)).unwrap();

    let mut seed = 0x7717;
    let coeffs: [u32; N] = core::array::from_fn(|_| (next(&mut seed) % mlkem::Q as u64) as u32);

    let mut values = PolyValues::default();
    values.insert(w, coeffs).unwrap();

    let mut words = WordValues::default();
    let (trace, _) = chiplet.trace(&mut words, &mut values).unwrap();

    let expected: [u32; N] =
        core::array::from_fn(|i| fips_decompress(1, fips_compress(1, coeffs[i])));

    assert_eq!(values.get(mu).unwrap(), &expected);
    assert_eq!(words.get(twin).unwrap(), words.get(m).unwrap());

    let st = Statement {
        coefs: [(w, coeffs), (mu, expected)]
            .iter()
            .flat_map(|(poly, c)| {
                c.iter()
                    .enumerate()
                    .map(move |(i, &v)| (poly.id(), i as u16, v))
            })
            .collect(),
        hints: Vec::new(),
        words: word_tokens(&[m], &words),
    };

    let proof = Proof::new(&chiplet, trace, &st);

    assert!(proof.report().is_clean());
    assert!(proof.accepted(false));
    assert!(proof.accepted(true));
}

#[test]
fn scribble_codec_row_mutations_caught() {
    let dsa = DsaCase::new(MlDsaParams::ML_DSA_44);
    let mut words = dsa.encoding.words(&dsa);

    let (trace, _) = dsa
        .chiplet
        .trace(&mut words, &mut PolyValues::default())
        .unwrap();

    let kem = KemCase::new(MlKemParams::ML_KEM_768);

    let (mut kem_words, mut kem_values) = kem_inputs(&kem.parts);

    let (kem_trace, _) = kem.chiplet.trace(&mut kem_words, &mut kem_values).unwrap();

    for proof in [
        Proof::new(&dsa.chiplet, trace, &dsa.statement()),
        Proof::new(&kem.chiplet, kem_trace, &kem_statement(&kem.parts)),
    ] {
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
}
