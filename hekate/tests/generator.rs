// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Fixed-seed randomized circuit corpus. Every sample draws
//! a random table geometry, gate shape, bus topology and
//! witness, then proves and verifies end to end, asserting
//! the published value is bound to the committed witness
//! cell: the honest instance verifies, a forged public
//! input is rejected, and a drifted program id is rejected.

use std::collections::BTreeSet;

use hekate::core::config::Config;
use hekate::core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::{Bit, Block32, Block128, TowerField};
use hekate_core::errors::Error;
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::digest::program_id;
use hekate_program::permutation::{BusKind, PermutationCheckSpec, Service, ServiceSlot};
use hekate_program::{
    Air, CadenceSegment, FixedColumn, FixedShape, ProgramInstance, ProgramWitness, fix,
};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;

type F = Block128;
type H = DefaultHasher;

const DOMAIN: &[u8] = b"generator_corpus";
const CORPUS: u64 = 64;
const GEN_BUS_ID: &str = "gen_service";
const KAPPA_GEN_X: &[u8] = b"kappa_gen_x";

const CHIP_X: usize = 0;
const CHIP_SEL: usize = 1;

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);

        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);

        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn coin(&mut self) -> bool {
        self.next() & 1 == 1
    }
}

#[derive(Clone)]
struct GenChiplet {
    live: usize,
}

impl Air<F> for GenChiplet {
    fn num_columns(&self) -> usize {
        2
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(|| vec![ColumnType::B32, ColumnType::Bit])
    }

    fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
        vec![fix(
            CHIP_SEL,
            FixedShape::Cadence {
                stride: 1,
                count: self.live,
                origin: 0,
                values: vec![F::ONE],
            },
        )]
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![(
            GEN_BUS_ID.into(),
            gen_service().respond(&[CHIP_X], CHIP_SEL).unwrap(),
        )]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        let not_sel = cs.one() + cs.col(CHIP_SEL);

        cs.assert_zero_when(not_sel, cs.col(CHIP_X));

        cs.build()
    }
}

struct Sample {
    num_vars: usize,
    triples: usize,
    gate: FixedShape<F>,
    active: Vec<usize>,
    publish_triple: usize,
    publish_row: usize,
    with_bus: bool,
    zk: bool,
}

fn build_air(s: &Sample) -> CircuitProgram<F> {
    let num_rows = 1 << s.num_vars;

    let mut cx = Circuit::<F>::new("GeneratorHost", num_rows).unwrap();
    let mut cols = Vec::with_capacity(s.triples);

    for _ in 0..s.triples {
        let a = cx.column(ColumnType::B32);
        let b = cx.column(ColumnType::B32);
        let d = cx.column(ColumnType::B32);

        cols.push((a, b, d));
    }

    let gate_col = cx.column(ColumnType::Bit);
    cx.fix(gate_col, s.gate.clone());

    {
        let cs = cx.cs();
        let sel = cs.col(gate_col.index());
        let not_sel = cs.one() + sel;

        for &(a, b, d) in &cols {
            let ae = cs.col(a.index());
            let be = cs.col(b.index());
            let de = cs.col(d.index());

            cs.constrain(sel * (de + ae * be + ae + be));
            cs.assert_zero_when(not_sel, ae);
            cs.assert_zero_when(not_sel, be);
            cs.assert_zero_when(not_sel, de);
        }
    }

    cx.publish(cols[s.publish_triple].2, s.publish_row);

    if s.with_bus {
        cx.call(&gen_service(), &[cols[0].0], gate_col).unwrap();

        cx.attach(
            ChipletDef::from_air(&GenChiplet {
                live: s.active.len(),
            })
            .unwrap(),
        );
    }

    cx.compile().unwrap()
}

fn build_witness(
    s: &Sample,
    vals: &[Vec<(Block32, Block32, Block32)>],
) -> (ColumnTrace, Option<ColumnTrace>) {
    let mut layout: Vec<ColumnType> = Vec::new();
    for _ in 0..s.triples {
        layout.extend([ColumnType::B32; 3]);
    }

    layout.push(ColumnType::Bit);

    let gate_idx = s.triples * 3;
    let mut tb = TraceBuilder::new(&layout, s.num_vars).unwrap();

    for (t, tvals) in vals.iter().enumerate() {
        for (j, &row) in s.active.iter().enumerate() {
            let (a, b, d) = tvals[j];
            tb.set_b32(t * 3, row, a).unwrap();
            tb.set_b32(t * 3 + 1, row, b).unwrap();
            tb.set_b32(t * 3 + 2, row, d).unwrap();
        }
    }

    for &row in &s.active {
        tb.set_bit(gate_idx, row, Bit::ONE).unwrap();
    }

    let main = tb.build();

    let chip = s.with_bus.then(|| {
        let chip_layout = [ColumnType::B32, ColumnType::Bit];
        let mut tb = TraceBuilder::new(&chip_layout, s.num_vars).unwrap();

        for (j, &(a, _, _)) in vals[0].iter().enumerate() {
            tb.set_b32(CHIP_X, j, a).unwrap();
            tb.set_bit(CHIP_SEL, j, Bit::ONE).unwrap();
        }

        tb.build()
    });

    (main, chip)
}

fn run_seed(seed: u64) {
    let mut rng = Rng::new(seed);

    let num_vars = 4 + rng.below(3);
    let num_rows = 1usize << num_vars;
    let triples = 1 + rng.below(2);
    let (gate, active) = sample_gate(&mut rng, num_rows);

    let publish_row = active[rng.below(active.len())];

    let s = Sample {
        num_vars,
        triples,
        gate,
        publish_triple: rng.below(triples),
        publish_row,
        with_bus: rng.coin(),
        zk: rng.below(3) == 0,
        active,
    };

    let vals: Vec<Vec<(Block32, Block32, Block32)>> = (0..s.triples)
        .map(|_| {
            s.active
                .iter()
                .map(|_| {
                    let a = Block32(rng.next() as u32);
                    let b = Block32(rng.next() as u32);
                    let d = a * b + a + b;

                    (a, b, d)
                })
                .collect()
        })
        .collect();

    let publish_idx = s.active.iter().position(|&r| r == s.publish_row).unwrap();
    let public_value = F::from(vals[s.publish_triple][publish_idx].2.0 as u128);

    let air = build_air(&s);
    let (main, chip) = build_witness(&s, &vals);

    let witness = match chip {
        Some(c) => ProgramWitness::new(main).with_chiplets(vec![c]),
        None => ProgramWitness::new(main),
    };

    let instance = ProgramInstance::new(num_rows, vec![public_value]);

    let config = if s.zk {
        Config {
            num_queries: 4,
            min_security_bits: 0,
            zero_knowledge: true,
            ..Config::default()
        }
    } else {
        Config {
            num_queries: 8,
            min_security_bits: 0,
            ldt_support_size: 4,
            ..Config::default()
        }
    };

    let mut seed_bytes = [0u8; 32];
    for chunk in seed_bytes.chunks_mut(8) {
        chunk.copy_from_slice(&rng.next().to_le_bytes());
    }

    let proof = prove(DOMAIN, &air, &instance, &witness, &config, seed_bytes, None)
        .unwrap_or_else(|e| panic!("seed {seed}: prove failed: {e:?}"));

    let pinned = program_id(&air).unwrap();

    let mut t = Transcript::<H>::new(DOMAIN);
    let honest = HekateVerifier::<F, H>::verify(&pinned, &air, &instance, &proof, &mut t, &config)
        .unwrap_or_else(|e| panic!("seed {seed}: honest verify errored: {e:?}"));

    assert!(honest, "seed {seed}: honest proof rejected");

    let delta = F::from(((rng.next() as u32) | 1) as u128);
    let forged = ProgramInstance::new(num_rows, vec![public_value + delta]);

    let mut t = Transcript::<H>::new(DOMAIN);
    let outcome = HekateVerifier::<F, H>::verify(&pinned, &air, &forged, &proof, &mut t, &config);

    assert!(
        !matches!(outcome, Ok(true)),
        "seed {seed}: forged public input accepted",
    );

    let mut drifted = pinned;
    drifted[0] ^= 1;

    let mut t = Transcript::<H>::new(DOMAIN);
    let outcome =
        HekateVerifier::<F, H>::verify(&drifted, &air, &instance, &proof, &mut t, &config);

    assert!(
        matches!(outcome, Err(Error::ProgramIdMismatch { .. })),
        "seed {seed}: drifted program id not rejected",
    );
}

fn sample_gate(rng: &mut Rng, num_rows: usize) -> (FixedShape<F>, Vec<usize>) {
    match rng.below(4) {
        0 => {
            let count = 1 + rng.below(num_rows - 1);
            let shape = FixedShape::Cadence {
                stride: 1,
                count,
                origin: 0,
                values: vec![F::ONE],
            };

            (shape, (0..count).collect())
        }
        1 => {
            let stride = 2 + rng.below(2);
            let origin = rng.below(stride);
            let count = 1 + rng.below((num_rows - origin) / stride);

            let mut values = vec![F::ZERO; stride];
            values[0] = F::ONE;

            let rows = (0..count).map(|j| origin + j * stride).collect();
            let shape = FixedShape::Cadence {
                stride,
                count,
                origin,
                values,
            };

            (shape, rows)
        }
        2 => {
            let m = 1 + rng.below(4);

            let mut rows = BTreeSet::new();
            while rows.len() < m {
                rows.insert(rng.below(num_rows));
            }

            let rows: Vec<usize> = rows.into_iter().collect();
            let shape = FixedShape::Sparse(rows.iter().map(|&r| (r, F::ONE)).collect());

            (shape, rows)
        }
        _ => {
            let half = num_rows / 2;
            let o1 = rng.below(half / 2);
            let c1 = 1 + rng.below(half - o1);
            let o2 = half + rng.below(half / 2);
            let c2 = 1 + rng.below(num_rows - o2);

            let seg = |origin: usize, count: usize| CadenceSegment {
                stride: 1,
                count,
                origin,
                values: vec![F::ONE],
            };

            let rows = (o1..o1 + c1).chain(o2..o2 + c2).collect();

            (FixedShape::Segments(vec![seg(o1, c1), seg(o2, c2)]), rows)
        }
    }
}

fn gen_service() -> Service {
    Service {
        bus_id: GEN_BUS_ID,
        kind: BusKind::Permutation,
        slots: vec![ServiceSlot::Value(KAPPA_GEN_X), ServiceSlot::EmitRank],
    }
}

#[test]
fn corpus_binds_published_values() {
    for seed in 0..CORPUS {
        run_seed(seed);
    }
}
