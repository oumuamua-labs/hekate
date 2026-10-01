// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::config::Config;
use hekate_core::errors::Error;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder, TraceColumn};
use hekate_crypto::DefaultHasher;
use hekate_crypto::transcript::Transcript;
use hekate_keccak::{KeccakChiplet, generate_keccak_trace, shake128, shake256};
use hekate_math::{Bit, Block16, Block32, Block64, Block128, HardwareField, TowerField};
use hekate_pqc::mldsa::MlDsaParams;
use hekate_pqc::sampler::{SamplerChiplet, SamplerForgery, SamplerStep};
use hekate_pqc::wiring::{
    COEF_BUS_ID, LANE_BUS_ID, LaneValues, N, Poly, PolyLabels, PolyValues, Stream, coef_spec,
    lane_spec,
};
use hekate_pqc::{mldsa, mlkem};
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::digest::program_id;
use hekate_program::{FixedShape, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_scribble::{MutationKind, ScribbleConfig, assert_all_caught_all_targets};
use hekate_sdk::preflight::{PreflightReport, TableId, preflight};
use hekate_verifier::HekateVerifier;
use zeroize::Zeroizing;

type F = Block128;
type H = DefaultHasher;

const RATE: usize = 168;
const SIGN_BYTES: usize = 8;

const HOST_LAYOUT: [ColumnType; 8] = [
    ColumnType::B16,
    ColumnType::B16,
    ColumnType::B32,
    ColumnType::Bit,
    ColumnType::B16,
    ColumnType::B16,
    ColumnType::B64,
    ColumnType::Bit,
];

#[derive(Clone, Copy)]
enum Draw {
    ExpandA { r: u8, s: u8 },
    SampleNtt { i: u8, j: u8 },
    Cbd { n: u8, eta: u32 },
    Ball(MlDsaParams),
}

impl Draw {
    fn seed_bytes(self) -> usize {
        match self {
            Draw::Ball(params) => params.lambda() / 4,
            Draw::ExpandA { .. } | Draw::SampleNtt { .. } | Draw::Cbd { .. } => 32,
        }
    }

    fn squeeze(self, seed: &[u8]) -> Zeroizing<Vec<u8>> {
        let msg = |suffix: &[u8]| [seed, suffix].concat();

        match self {
            Draw::ExpandA { r, s } => shake128(&msg(&[s, r]), 5 * RATE).0,
            Draw::SampleNtt { i, j } => shake128(&msg(&[j, i]), 4 * RATE).0,
            Draw::Cbd { n, eta } => shake256(&msg(&[n]), 64 * eta as usize).0,
            Draw::Ball(_) => shake256(seed, SIGN_BYTES + 2 * N).0,
        }
    }

    fn expected(self, seed: &[u8]) -> [u32; N] {
        let bytes = self.squeeze(seed);

        match self {
            Draw::ExpandA { .. } => rej_ntt_poly(&bytes),
            Draw::SampleNtt { .. } => sample_ntt(&bytes),
            Draw::Cbd { eta, .. } => sample_poly_cbd(&bytes, eta as usize),
            Draw::Ball(params) => sample_in_ball(&bytes, params.tau()),
        }
    }
}

struct Entry {
    draw: Draw,
    seed: usize,
    out: Poly,
}

struct Case {
    streams: Vec<(Stream, Vec<u8>)>,
    entries: Vec<Entry>,
    chiplet: SamplerChiplet<F>,
}

#[derive(Default)]
struct Statement {
    coefs: Vec<(u16, u16, u32)>,
    lanes: Vec<(u16, u16, u64)>,
}

struct Proof {
    program: CircuitProgram<F>,
    instance: ProgramInstance<F>,
    witness: ProgramWitness<F>,
}

impl Case {
    fn new(draws: &[(usize, Draw)]) -> Self {
        let mut labels = PolyLabels::new();
        let mut state = 0x5a3c_u64;

        let num_streams = draws.iter().map(|&(seed, _)| seed + 1).max().unwrap();
        let streams: Vec<(Stream, Vec<u8>)> = (0..num_streams)
            .map(|index| {
                let (_, draw) = draws.iter().find(|&&(seed, _)| seed == index).unwrap();
                let seed = (0..draw.seed_bytes())
                    .map(|_| next(&mut state) as u8)
                    .collect();

                (labels.stream().unwrap(), seed)
            })
            .collect();

        let entries: Vec<Entry> = draws
            .iter()
            .map(|&(seed, draw)| Entry {
                draw,
                seed,
                out: labels.fresh().unwrap(),
            })
            .collect();

        let steps: Vec<SamplerStep> = entries
            .iter()
            .map(|e| {
                let stream = streams[e.seed].0;

                match e.draw {
                    Draw::ExpandA { r, s } => SamplerStep::expand_a(stream, r, s, e.out),
                    Draw::SampleNtt { i, j } => SamplerStep::sample_ntt(stream, i, j, e.out),
                    Draw::Cbd { n, eta } => SamplerStep::prf(stream, n, eta, e.out),
                    Draw::Ball(params) => {
                        SamplerStep::sample_in_ball(&params, stream, e.out).unwrap()
                    }
                }
            })
            .collect();

        let rows = height(steps.iter().map(SamplerStep::rows).sum());

        Self {
            streams,
            entries,
            chiplet: SamplerChiplet::new(steps, rows).unwrap(),
        }
    }

    fn seeds(&self) -> LaneValues {
        let mut seeds = LaneValues::default();
        for (stream, seed) in &self.streams {
            let lanes = seed
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                .collect();

            seeds.insert(*stream, lanes).unwrap();
        }

        seeds
    }

    fn statement(&self, values: &PolyValues) -> Statement {
        let mut st = Statement::default();
        let seeds = self.seeds();

        for &(stream, _) in &self.streams {
            for (i, &lane) in seeds.get(stream).unwrap().iter().enumerate() {
                st.lanes.push((stream.id(), i as u16, lane));
            }
        }

        for entry in &self.entries {
            for (pos, &value) in values.get(entry.out).unwrap().iter().enumerate() {
                st.coefs.push((entry.out.id(), pos as u16, value));
            }
        }

        st
    }

    fn proof(&self, trace: ColumnTrace, calls: &[[u64; 25]], st: &Statement) -> Proof {
        let host_rows = height(st.coefs.len().max(st.lanes.len()));
        let keccak_rows = height(KeccakChiplet::BLOCK_ROWS * calls.len());

        let inputs: Vec<[Block64; 25]> = calls.iter().map(|c| c.map(Block64)).collect();
        let keccak = generate_keccak_trace(&inputs, keccak_rows).unwrap();

        Proof {
            program: self.host_program(st, host_rows, keccak_rows, calls.len()),
            instance: ProgramInstance::new(host_rows, Vec::new()),
            witness: ProgramWitness::new(host_trace(st, host_rows))
                .with_chiplets(vec![trace, keccak]),
        }
    }

    fn host_program(
        &self,
        st: &Statement,
        rows: usize,
        keccak_rows: usize,
        calls: usize,
    ) -> CircuitProgram<F> {
        let mut cx = Circuit::<F>::new("SamplerHost", rows).unwrap();
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
            pin(st.lanes.iter().map(|l| l.0 as u64).collect()),
        );
        cx.fix(
            cols.at(5),
            pin(st.lanes.iter().map(|l| l.1 as u64).collect()),
        );
        cx.fix(cols.at(7), pin(vec![1; st.lanes.len()]));

        cx.bus(
            COEF_BUS_ID,
            coef_spec(
                cols.at(0).index(),
                cols.at(1).index(),
                cols.at(2).index(),
                cols.at(3).index(),
            ),
        );
        cx.bus(
            LANE_BUS_ID,
            lane_spec(
                cols.at(4).index(),
                cols.at(5).index(),
                cols.at(6).index(),
                cols.at(7).index(),
            ),
        );

        cx.attach(self.chiplet.def().unwrap());
        cx.attach(ChipletDef::from_air(&KeccakChiplet::new(keccak_rows, calls)).unwrap());

        cx.compile().unwrap()
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
            b"SamplerChiplet",
            &self.program,
            &self.instance,
            &self.witness,
            &config,
            [7; 32],
            None,
        )
        .unwrap();

        let mut transcript = Transcript::<H>::new(b"SamplerChiplet");

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

fn dsa_case() -> Case {
    Case::new(&[
        (0, Draw::ExpandA { r: 0, s: 0 }),
        (0, Draw::ExpandA { r: 1, s: 2 }),
        (0, Draw::ExpandA { r: 3, s: 1 }),
    ])
}

fn kem_case() -> Case {
    Case::new(&[
        (0, Draw::SampleNtt { i: 0, j: 0 }),
        (0, Draw::SampleNtt { i: 1, j: 2 }),
        (1, Draw::Cbd { n: 0, eta: 2 }),
        (1, Draw::Cbd { n: 1, eta: 2 }),
    ])
}

fn kem_512_case() -> Case {
    Case::new(&[
        (0, Draw::SampleNtt { i: 0, j: 1 }),
        (0, Draw::SampleNtt { i: 1, j: 0 }),
        (1, Draw::Cbd { n: 0, eta: 3 }),
        (1, Draw::Cbd { n: 1, eta: 2 }),
    ])
}

fn cbd_case() -> Case {
    Case::new(&[
        (0, Draw::Cbd { n: 0, eta: 3 }),
        (0, Draw::Cbd { n: 1, eta: 3 }),
    ])
}

fn ball_case(params: MlDsaParams) -> Case {
    Case::new(&[(0, Draw::ExpandA { r: 0, s: 1 }), (1, Draw::Ball(params))])
}

fn two_ball_case(params: MlDsaParams) -> Case {
    Case::new(&[(0, Draw::Ball(params)), (1, Draw::Ball(params))])
}

fn honest_case(case: Case) {
    let mut values = PolyValues::default();

    let (trace, calls) = case.chiplet.trace(&case.seeds(), &mut values).unwrap();

    for entry in &case.entries {
        let seed = &case.streams[entry.seed].1;

        assert_eq!(values.get(entry.out).unwrap(), &entry.draw.expected(seed));
    }

    let proof = case.proof(trace, &calls, &case.statement(&values));

    assert!(proof.report().is_clean());
    assert!(proof.accepted(false));
    assert!(proof.accepted(true));
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

fn forge_word(trace: &mut ColumnTrace, col: usize, row: usize) -> u32 {
    let TraceColumn::B32(cells) = &mut trace.columns[col] else {
        panic!("forged cells live in B32 columns");
    };

    let forged = cells[row].to_tower().0 ^ 1;
    cells[row] = Block32::from(forged).to_hardware();

    forged
}

fn set_word(trace: &mut ColumnTrace, col: usize, row: usize, value: u32) {
    let TraceColumn::B32(cells) = &mut trace.columns[col] else {
        panic!("forged cells live in B32 columns");
    };

    cells[row] = Block32::from(value).to_hardware();
}

fn set_packed_bit(trace: &mut ColumnTrace, (col, bit): (usize, usize), row: usize) {
    let TraceColumn::B32(cells) = &mut trace.columns[col] else {
        panic!("packed bits live in B32 columns");
    };

    let word = cells[row].to_tower().0 | 1 << bit;
    cells[row] = Block32::from(word).to_hardware();
}

fn row_where(trace: &ColumnTrace, cells: &[(usize, u64)]) -> usize {
    let at = |col: usize, row: usize| match &trace.columns[col] {
        TraceColumn::Bit(cells) => cells[row].get() as u64,
        TraceColumn::B16(cells) => cells[row].to_tower().0 as u64,
        _ => panic!("row lookups read Bit and B16 columns"),
    };

    (0..1 << trace.num_vars)
        .find(|&row| cells.iter().all(|&(col, value)| at(col, row) == value))
        .unwrap()
}

fn host_trace(st: &Statement, rows: usize) -> ColumnTrace {
    let mut tb = TraceBuilder::new(&HOST_LAYOUT, rows.trailing_zeros() as usize).unwrap();

    for (r, &(poly, pos, value)) in st.coefs.iter().enumerate() {
        tb.set_b16(0, r, Block16(poly)).unwrap();
        tb.set_b16(1, r, Block16(pos)).unwrap();
        tb.set_b32(2, r, Block32::from(value)).unwrap();
        tb.set_bit(3, r, Bit::ONE).unwrap();
    }

    for (r, &(stream, index, lane)) in st.lanes.iter().enumerate() {
        tb.set_b16(4, r, Block16(stream)).unwrap();
        tb.set_b16(5, r, Block16(index)).unwrap();
        tb.set_b64(6, r, Block64(lane)).unwrap();
        tb.set_bit(7, r, Bit::ONE).unwrap();
    }

    tb.build()
}

fn rej_ntt_poly(bytes: &[u8]) -> [u32; N] {
    let mut out = [0u32; N];
    let (mut j, mut at) = (0, 0);

    while j < N {
        let z =
            bytes[at] as u32 | (bytes[at + 1] as u32) << 8 | ((bytes[at + 2] & 127) as u32) << 16;

        if z < mldsa::Q {
            out[j] = z;
            j += 1;
        }

        at += 3;
    }

    out
}

fn sample_ntt(bytes: &[u8]) -> [u32; N] {
    let mut out = [0u32; N];
    let (mut j, mut at) = (0, 0);

    while j < N {
        let (c0, c1, c2) = (bytes[at] as u32, bytes[at + 1] as u32, bytes[at + 2] as u32);
        let d1 = c0 + 256 * (c1 % 16);
        let d2 = c1 / 16 + 16 * c2;

        if d1 < mlkem::Q {
            out[j] = d1;
            j += 1;
        }

        if d2 < mlkem::Q && j < N {
            out[j] = d2;
            j += 1;
        }

        at += 3;
    }

    out
}

fn sample_in_ball(bytes: &[u8], tau: usize) -> [u32; N] {
    let signs = u64::from_le_bytes(bytes[..SIGN_BYTES].try_into().unwrap());

    let mut candidates = bytes[SIGN_BYTES..].iter().map(|&b| b as usize);
    let mut c = [0u32; N];

    for i in N - tau..N {
        let j = candidates.find(|&j| j <= i).unwrap();

        c[i] = c[j];
        c[j] = match (signs >> (i + tau - N)) & 1 {
            1 => mldsa::Q - 1,
            _ => 1,
        };
    }

    c
}

fn ball_decisions(bytes: &[u8], tau: usize) -> Vec<bool> {
    let mut picked = 0;

    bytes[SIGN_BYTES..]
        .iter()
        .map(|&j| {
            let take = picked < tau && j as usize <= N - tau + picked;

            picked += take as usize;

            take
        })
        .collect()
}

fn sample_poly_cbd(bytes: &[u8], eta: usize) -> [u32; N] {
    let bit = |k: usize| ((bytes[k / 8] >> (k % 8)) & 1) as u32;

    core::array::from_fn(|i| {
        let x: u32 = (0..eta).map(|j| bit(2 * i * eta + j)).sum();
        let y: u32 = (0..eta).map(|j| bit(2 * i * eta + eta + j)).sum();

        (x + mlkem::Q - y) % mlkem::Q
    })
}

fn next(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);

    *seed >> 33
}

fn height(rows: usize) -> usize {
    rows.next_power_of_two()
        .max(Config::prod().min_table_rows())
}

#[test]
fn ml_dsa_expand_matches_fips_and_proves() {
    honest_case(dsa_case());
}

#[test]
fn ml_kem_768_table_matches_fips_and_proves() {
    honest_case(kem_case());
}

#[test]
fn ml_kem_512_table_matches_fips_and_proves() {
    honest_case(kem_512_case());
}

#[test]
fn ml_kem_cbd_table_matches_fips_and_proves() {
    honest_case(cbd_case());
}

#[test]
fn ml_dsa_sample_in_ball_matches_fips_and_proves() {
    for params in [
        MlDsaParams::ML_DSA_44,
        MlDsaParams::ML_DSA_65,
        MlDsaParams::ML_DSA_87,
    ] {
        honest_case(ball_case(params));
    }
}

#[test]
fn flipped_acceptance_breaks_only_its_constraint() {
    let case = kem_case();
    let bytes = case.entries[0].draw.squeeze(&case.streams[0].1);

    let value = |c: usize| {
        let bit = 12 * c;
        let word = u32::from_le_bytes([bytes[bit / 8], bytes[bit / 8 + 1], bytes[bit / 8 + 2], 0]);

        (word >> (bit % 8)) & 0xfff
    };

    let rejected = (0..100).find(|&c| value(c) >= mlkem::Q).unwrap();
    let accepted = (0..100).find(|&c| value(c) < mlkem::Q).unwrap();

    for c in [rejected, accepted] {
        let mut values = PolyValues::default();

        let (trace, calls) = case
            .chiplet
            .trace_forged(
                &case.seeds(),
                &mut values,
                &[SamplerForgery::Flip {
                    step: 0,
                    candidate: c,
                }],
            )
            .unwrap();

        let proof = case.proof(trace, &calls, &case.statement(&values));

        assert_breaks_only(&proof, "sampler_accept");
    }
}

#[test]
fn swapped_positions_break_only_sampler_count() {
    let case = kem_case();
    let bytes = case.entries[0].draw.squeeze(&case.streams[0].1);

    let taken = |c: usize| {
        let bit = 12 * c;
        let word = u32::from_le_bytes([bytes[bit / 8], bytes[bit / 8 + 1], bytes[bit / 8 + 2], 0]);

        ((word >> (bit % 8)) & 0xfff) < mlkem::Q
    };

    let slots = 16;
    let candidate = (0..100)
        .find(|&c| (1..slots - 1).contains(&(c % slots)) && taken(c) && taken(c + 1))
        .unwrap();

    let mut values = PolyValues::default();

    let (trace, calls) = case
        .chiplet
        .trace_forged(
            &case.seeds(),
            &mut values,
            &[SamplerForgery::Swap { step: 0, candidate }],
        )
        .unwrap();

    let proof = case.proof(trace, &calls, &case.statement(&values));

    assert_breaks_only(&proof, "sampler_count");
}

#[test]
fn reversed_suffix_breaks_only_sampler_absorb() {
    let case = dsa_case();
    let mut values = PolyValues::default();

    let (trace, calls) = case
        .chiplet
        .trace_forged(
            &case.seeds(),
            &mut values,
            &[SamplerForgery::Suffix { step: 1 }],
        )
        .unwrap();

    let proof = case.proof(trace, &calls, &case.statement(&values));

    assert_breaks_only(&proof, "sampler_absorb");
}

#[test]
fn window_read_one_lane_on_breaks_only_sampler_window() {
    let case = dsa_case();
    let mut values = PolyValues::default();

    let (trace, calls) = case
        .chiplet
        .trace_forged(
            &case.seeds(),
            &mut values,
            &[SamplerForgery::Window { step: 0, window: 0 }],
        )
        .unwrap();

    let proof = case.proof(trace, &calls, &case.statement(&values));

    assert_breaks_only(&proof, "sampler_window");
}

#[test]
fn ball_accesses_out_of_time_order_break_only_sampler_sib_order() {
    let case = ball_case(MlDsaParams::ML_DSA_44);
    let mut values = PolyValues::default();

    let (trace, calls) = case
        .chiplet
        .trace_forged(
            &case.seeds(),
            &mut values,
            &[SamplerForgery::Unsorted { step: 1 }],
        )
        .unwrap();

    let proof = case.proof(trace, &calls, &case.statement(&values));

    assert_breaks_only(&proof, "sampler_sib_order");
}

#[test]
fn forged_coefficient_breaks_only_sampler_value() {
    let case = dsa_case();
    let mut values = PolyValues::default();

    let (mut trace, calls) = case.chiplet.trace(&case.seeds(), &mut values).unwrap();

    let ly = case.chiplet.layout();
    let rc = ly.rej_cols.as_ref().unwrap();

    let forged = forge_word(&mut trace, ly.physical(rc.key_value.at(0)).unwrap(), 2);

    let mut st = case.statement(&values);
    st.coefs[0].2 = forged;

    assert_breaks_only(&case.proof(trace, &calls, &st), "sampler_value");
}

#[test]
fn forged_cbd_value_breaks_only_sampler_cbd() {
    let case = cbd_case();
    let mut values = PolyValues::default();

    let (mut trace, calls) = case.chiplet.trace(&case.seeds(), &mut values).unwrap();

    let ly = case.chiplet.layout();
    let forged = forge_word(
        &mut trace,
        ly.physical(ly.cbd_cols[0].value.at(0)).unwrap(),
        2,
    );

    let mut st = case.statement(&values);
    st.coefs[0].2 = forged;

    assert_breaks_only(&case.proof(trace, &calls, &st), "sampler_cbd");
}

#[test]
fn schedules_outside_table_are_refused() {
    let mut labels = PolyLabels::new();

    let (a, b) = (labels.stream().unwrap(), labels.stream().unwrap());
    let outs: Vec<Poly> = (0..3).map(|_| labels.fresh().unwrap()).collect();

    let (p44, p65) = (MlDsaParams::ML_DSA_44, MlDsaParams::ML_DSA_65);

    let ball = |params, seed, out| SamplerStep::sample_in_ball(&params, seed, out).unwrap();

    let refused = |steps: Vec<SamplerStep>, rows: usize| {
        SamplerChiplet::<F>::new(steps, rows)
            .err()
            .map(|e| match e {
                Error::Protocol { message, .. } => message,
                _ => "",
            })
    };

    let rows = 1 << 16;

    for (steps, message) in [
        (vec![], "sampler table needs at least one step"),
        (
            vec![
                SamplerStep::expand_a(a, 0, 0, outs[0]),
                SamplerStep::sample_ntt(a, 0, 1, outs[1]),
            ],
            "one sampler table holds one rejection shape",
        ),
        (
            vec![ball(p44, a, outs[0]), ball(p65, b, outs[1])],
            "one sampler table holds one SampleInBall shape",
        ),
        (
            vec![SamplerStep::prf(a, 0, 4, outs[0])],
            "CBD takes eta 2 or 3",
        ),
        (
            vec![
                SamplerStep::expand_a(a, 0, 0, outs[0]),
                ball(p65, a, outs[1]),
            ],
            "adjacent steps on one seed stream disagree on the seed length",
        ),
        (
            vec![
                SamplerStep::expand_a(a, 0, 0, outs[0]),
                SamplerStep::expand_a(a, 0, 1, outs[0]),
            ],
            "two steps sample into one polynomial label",
        ),
        (
            vec![
                SamplerStep::expand_a(a, 0, 0, outs[0]),
                SamplerStep::expand_a(b, 0, 0, outs[1]),
                SamplerStep::expand_a(a, 0, 1, outs[2]),
            ],
            "seed stream loads in two separate runs of steps",
        ),
    ] {
        assert_eq!(refused(steps, rows), Some(message));
    }

    assert_eq!(
        refused(vec![SamplerStep::expand_a(a, 0, 0, outs[0])], 16),
        Some("steps need more rows than the table holds")
    );
}

#[test]
fn tail_from_second_out_breaks_only_sampler_tail() {
    let case = cbd_case();
    let mut values = PolyValues::default();

    let (trace, calls) = case
        .chiplet
        .trace_forged(
            &case.seeds(),
            &mut values,
            &[SamplerForgery::Tail { step: 0 }],
        )
        .unwrap();

    let proof = case.proof(trace, &calls, &case.statement(&values));

    assert_breaks_only(&proof, "sampler_tail");
}

#[test]
fn flipped_ball_candidate_breaks_only_its_constraint() {
    let params = MlDsaParams::ML_DSA_65;
    let case = ball_case(params);

    let bytes = case.entries[1].draw.squeeze(&case.streams[1].1);
    let decisions = ball_decisions(&bytes, params.tau());

    let rejected = decisions.iter().position(|&taken| !taken).unwrap();
    let accepted = decisions.iter().position(|&taken| taken).unwrap();

    for c in [rejected, accepted] {
        let mut values = PolyValues::default();

        let (trace, calls) = case
            .chiplet
            .trace_forged(
                &case.seeds(),
                &mut values,
                &[SamplerForgery::Flip {
                    step: 1,
                    candidate: c,
                }],
            )
            .unwrap();

        let proof = case.proof(trace, &calls, &case.statement(&values));

        assert_breaks_only(&proof, "sampler_ball_accept");
    }
}

#[test]
fn flipped_sign_breaks_only_sampler_sign() {
    let params = MlDsaParams::ML_DSA_44;
    let case = ball_case(params);

    let mut values = PolyValues::default();

    let (mut trace, calls) = case.chiplet.trace(&case.seeds(), &mut values).unwrap();

    let ly = case.chiplet.layout();
    let bc = ly.ball_cols.as_ref().unwrap();
    let col = |c| ly.physical(c).unwrap();

    let tau = params.tau();
    let bytes = case.entries[1].draw.squeeze(&case.streams[1].1);
    let last = ball_decisions(&bytes, tau)
        .iter()
        .rposition(|&taken| taken)
        .unwrap();
    let j = bytes[SIGN_BYTES + last] as usize;

    let flipped = match values.get(case.entries[1].out).unwrap()[j] {
        1 => mldsa::Q - 1,
        _ => 1,
    };

    let step = row_where(
        &trace,
        &[(col(bc.mem.step), 1), (col(bc.mem.pos), (2 * N - 1) as u64)],
    );

    let readout = row_where(
        &trace,
        &[(col(bc.mem.readout), 1), (col(bc.mem.pos), j as u64)],
    );

    let sorted = |time: usize| {
        row_where(
            &trace,
            &[
                (col(bc.sort.sorted), 1),
                (col(bc.sort.addr), j as u64),
                (col(bc.sort.time), time as u64),
            ],
        )
    };

    let (write, read) = (sorted(3 * tau - 1), sorted(3 * tau + j));

    set_word(&mut trace, col(bc.sign.value), step, flipped);
    set_word(&mut trace, col(bc.sort.value), write, flipped);
    set_word(&mut trace, col(bc.sort.carried), write, flipped);
    set_word(&mut trace, col(bc.mem.value), readout, flipped);
    set_word(&mut trace, col(bc.sort.value), read, flipped);

    let mut st = case.statement(&values);
    st.coefs[N + j].2 = flipped;

    assert_breaks_only(&case.proof(trace, &calls, &st), "sampler_sign");
}

#[test]
fn forged_ball_readout_breaks_only_the_memory_check() {
    let params = MlDsaParams::ML_DSA_44;
    let case = ball_case(params);
    let mut values = PolyValues::default();

    let (mut trace, calls) = case.chiplet.trace(&case.seeds(), &mut values).unwrap();

    let ly = case.chiplet.layout();
    let bc = ly.ball_cols.as_ref().unwrap();
    let cell = N - 1;

    let readout = row_where(
        &trace,
        &[
            (ly.physical(bc.mem.readout).unwrap(), 1),
            (ly.physical(bc.mem.pos).unwrap(), cell as u64),
        ],
    );

    let sorted = row_where(
        &trace,
        &[
            (ly.physical(bc.sort.sorted).unwrap(), 1),
            (ly.physical(bc.sort.addr).unwrap(), cell as u64),
            (
                ly.physical(bc.sort.time).unwrap(),
                (3 * params.tau() + cell) as u64,
            ),
        ],
    );

    let forged = forge_word(&mut trace, ly.physical(bc.mem.value).unwrap(), readout);
    forge_word(&mut trace, ly.physical(bc.sort.value).unwrap(), sorted);

    let mut st = case.statement(&values);
    st.coefs[N + cell].2 = forged;

    assert_breaks_only(&case.proof(trace, &calls, &st), "sampler_sib_read");
}

#[test]
fn ball_steps_cannot_trade_memory_accesses() {
    let params = MlDsaParams::ML_DSA_44;
    let case = two_ball_case(params);

    let mut values = PolyValues::default();

    let (mut trace, calls) = case.chiplet.trace(&case.seeds(), &mut values).unwrap();

    let ly = case.chiplet.layout();
    let bc = ly.ball_cols.as_ref().unwrap();
    let write = ly.sort_write_cell().unwrap();

    let first = case.entries[0].out;
    let cell = (0..N)
        .find(|&p| values.get(first).unwrap()[p] != 0)
        .unwrap();

    let time = (3 * params.tau() + cell) as u64;

    let mut st = case.statement(&values);

    for (e, entry) in case.entries.iter().enumerate() {
        let poly = (ly.physical(ly.poly).unwrap(), entry.out.id() as u64);

        let readout = row_where(
            &trace,
            &[
                (ly.physical(bc.mem.readout).unwrap(), 1),
                (ly.physical(bc.mem.pos).unwrap(), cell as u64),
                poly,
            ],
        );

        let sorted = row_where(
            &trace,
            &[
                (ly.physical(bc.sort.sorted).unwrap(), 1),
                (ly.physical(bc.sort.addr).unwrap(), cell as u64),
                (ly.physical(bc.sort.time).unwrap(), time),
                poly,
            ],
        );

        set_word(&mut trace, ly.physical(bc.mem.value).unwrap(), readout, 0);
        set_word(&mut trace, ly.physical(bc.sort.value).unwrap(), sorted, 0);
        set_packed_bit(&mut trace, write, sorted);

        st.coefs[e * N + cell].2 = 0;
    }

    let proof = case.proof(trace, &calls, &st);
    let report = proof.report();

    assert!(report.constraint_violations.is_empty());
    assert!(report.fixed_column_violations.is_empty());
    assert!(report.boundary_violations.is_empty());
    assert!(!report.bus_diagnostics.is_empty());
    assert!(
        report
            .bus_diagnostics
            .iter()
            .all(|d| d.bus_id == "sampler_sib")
    );
    assert!(proof.rejected());
}

#[test]
fn host_supplying_different_seed_is_caught_by_bus_alone() {
    let case = kem_case();
    let mut values = PolyValues::default();

    let (trace, calls) = case.chiplet.trace(&case.seeds(), &mut values).unwrap();

    let mut st = case.statement(&values);
    st.lanes[1].2 ^= 1 << 13;

    let proof = case.proof(trace, &calls, &st);
    let report = proof.report();

    assert!(report.constraint_violations.is_empty());
    assert!(!report.bus_diagnostics.is_empty());
    assert!(proof.rejected());
}

#[test]
fn scribble_sampler_row_mutations_caught() {
    let case = kem_512_case();
    let mut values = PolyValues::default();

    let (trace, calls) = case.chiplet.trace(&case.seeds(), &mut values).unwrap();
    let proof = case.proof(trace, &calls, &case.statement(&values));

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

#[test]
fn scribble_sampler_ball_rows_caught() {
    let case = ball_case(MlDsaParams::ML_DSA_44);
    let mut values = PolyValues::default();

    let (trace, calls) = case.chiplet.trace(&case.seeds(), &mut values).unwrap();
    let proof = case.proof(trace, &calls, &case.statement(&values));

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
