// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::config::Config;
use hekate_core::errors::Error;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate_crypto::DefaultHasher;
use hekate_crypto::transcript::Transcript;
use hekate_math::{Bit, Block16, Block32, Block128, TowerField};
use hekate_pqc::ntt::NttParams;
use hekate_pqc::poly_arith::{BaseCaseMac, PolyArithChiplet, PolyArithForgery};
use hekate_pqc::wiring::{COEF_BUS_ID, N, Poly, PolyLabels, PolyValues, coef_spec};
use hekate_program::circuit::{Circuit, CircuitProgram, Col};
use hekate_program::digest::program_id;
use hekate_program::{CadenceSegment, FixedShape, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_scribble::{MutationKind, ScribbleConfig, assert_all_caught_all_targets};
use hekate_sdk::preflight::{PreflightReport, TableId, preflight};
use hekate_verifier::HekateVerifier;

type F = Block128;
type H = DefaultHasher;

const Q: u32 = 3329;
const ZETA: u64 = 17;

const HOST_LAYOUT: [ColumnType; 4] = [
    ColumnType::B16,
    ColumnType::B16,
    ColumnType::B32,
    ColumnType::Bit,
];

#[derive(Clone, Copy)]
struct Product {
    outputs: usize,
    terms: usize,
    seeded: bool,
}

#[derive(Clone)]
struct Step {
    m: Vec<Vec<Poly>>,
    v: Vec<Poly>,
    seed: Option<Vec<Poly>>,
    out: Vec<Poly>,
    copy: Option<Vec<Poly>>,
}

struct Case {
    steps: Vec<Step>,
    chiplet: PolyArithChiplet<F>,
}

impl Case {
    fn new(products: &[Product]) -> Self {
        Self::build(products, false)
    }

    fn build(products: &[Product], copied: bool) -> Self {
        let mut labels = PolyLabels::new();
        let mut fresh = |n: usize| {
            (0..n)
                .map(|_| labels.fresh().unwrap())
                .collect::<Vec<Poly>>()
        };

        let steps = products
            .iter()
            .map(|p| Step {
                m: (0..p.outputs).map(|_| fresh(p.terms)).collect(),
                v: fresh(p.terms),
                seed: p.seeded.then(|| fresh(p.outputs)),
                out: fresh(p.outputs),
                copy: copied.then(|| fresh(p.terms)),
            })
            .collect();

        Self::from_steps(steps)
    }

    fn from_steps(steps: Vec<Step>) -> Self {
        let macs: Vec<BaseCaseMac> = steps
            .iter()
            .map(|s| {
                let mac = BaseCaseMac::new(s.m.clone(), s.v.clone(), s.seed.clone(), s.out.clone());

                match &s.copy {
                    Some(copy) => mac.and_then(|mac| mac.with_copy(copy.clone())),
                    None => mac,
                }
            })
            .collect::<Result<_, _>>()
            .unwrap();

        let rows = height(macs.iter().map(BaseCaseMac::rows).sum());

        Self {
            steps,
            chiplet: PolyArithChiplet::new(macs, rows).unwrap(),
        }
    }

    fn produced(&self) -> Vec<Poly> {
        let mut polys = Vec::new();
        for s in &self.steps {
            polys.extend(s.m.iter().flatten());
            polys.extend(&s.v);
            polys.extend(s.seed.iter().flatten());
        }

        polys
    }

    fn consumed(&self) -> Vec<Poly> {
        self.steps
            .iter()
            .flat_map(|s| s.out.iter().chain(s.copy.iter().flatten()).copied())
            .collect()
    }

    fn tokens(&self) -> Vec<Poly> {
        let mut tokens = self.produced();
        tokens.extend(self.consumed());

        tokens
    }

    fn inputs(&self) -> PolyValues {
        let mut values = PolyValues::default();
        let mut seed = 0x9a1_u64;

        for poly in self.produced() {
            values
                .insert(poly, core::array::from_fn(|_| next(&mut seed) % Q))
                .unwrap();
        }

        values
    }

    fn run(&self) -> (ColumnTrace, PolyValues) {
        let mut values = self.inputs();
        let trace = self.chiplet.trace(&mut values).unwrap();

        (trace, values)
    }

    fn check_outputs(&self, values: &PolyValues) {
        let get = |poly: Poly| *values.get(poly).unwrap();

        for s in &self.steps {
            for (i, &out) in s.out.iter().enumerate() {
                let mut expected = match &s.seed {
                    Some(seeds) => get(seeds[i]),
                    None => [0; N],
                };

                for (j, &v) in s.v.iter().enumerate() {
                    let product = multiply_ntts(&get(s.m[i][j]), &get(v));

                    for (e, p) in expected.iter_mut().zip(product) {
                        *e = (*e + p) % Q;
                    }
                }

                assert_eq!(get(out), expected);
            }

            for (&copy, &v) in s.copy.iter().flatten().zip(&s.v) {
                assert_eq!(get(copy), get(v));
            }
        }
    }

    fn proof(&self, trace: ColumnTrace, values: &PolyValues) -> Proof {
        let tokens = self.tokens();
        let rows = height(tokens.len() * N);

        Proof {
            program: self.host_program(&tokens, rows),
            instance: ProgramInstance::new(rows, Vec::new()),
            witness: ProgramWitness::new(host_trace(&tokens, values, rows))
                .with_chiplets(vec![trace]),
        }
    }

    fn host_program(&self, tokens: &[Poly], rows: usize) -> CircuitProgram<F> {
        let mut cx = Circuit::<F>::new("PolyArithHost", rows).unwrap();

        let cols = cx.schema(&HOST_LAYOUT);

        let (poly, pos, value, sel) = (cols.at(0), cols.at(1), cols.at(2), cols.at(3));

        let runs = tokens
            .iter()
            .enumerate()
            .map(|(t, p)| CadenceSegment {
                stride: 1,
                count: N,
                origin: t * N,
                values: vec![F::from(p.id() as u32)],
            })
            .collect();

        cx.fix(poly, FixedShape::Segments(runs));
        cx.fix(
            pos,
            FixedShape::Cadence {
                stride: N,
                count: tokens.len(),
                origin: 0,
                values: (0..N as u32).map(F::from).collect(),
            },
        );
        cx.fix(
            sel,
            FixedShape::Cadence {
                stride: 1,
                count: tokens.len() * N,
                origin: 0,
                values: vec![F::ONE],
            },
        );

        cx.bus(
            COEF_BUS_ID,
            coef_spec(poly.index(), pos.index(), value.index(), sel.index()),
        );

        cx.attach(self.chiplet.def().unwrap());

        cx.compile().unwrap()
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
            b"PolyArithChiplet",
            &self.program,
            &self.instance,
            &self.witness,
            &config,
            [7; 32],
            None,
        )
        .unwrap();

        let mut transcript = Transcript::<H>::new(b"PolyArithChiplet");

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

fn honest(products: &[Product]) {
    let case = Case::new(products);
    let (trace, values) = case.run();

    case.check_outputs(&values);

    let proof = case.proof(trace, &values);

    assert!(proof.report().is_clean());
    assert!(proof.accepted(false));
    assert!(proof.accepted(true));
}

fn assert_forgery_breaks_only(honest: &Case, forged: &Case, col: Col) {
    let (trace, values) = forged.run();

    assert!(forged.proof(trace.clone(), &values).report().is_clean());

    let proof = honest.proof(trace, &values);
    let report = proof.report();

    assert!(report.boundary_violations.is_empty());
    assert!(report.bus_diagnostics.is_empty());
    assert!(!report.fixed_column_violations.is_empty());

    for v in &report.fixed_column_violations {
        assert!(v.table == TableId::Chiplet(0));
        assert_eq!(v.col_idx, col.index());
    }

    assert!(proof.rejected());
}

fn assert_breaks_only(forgery: PolyArithForgery, label: &str) {
    let case = Case::new(&[Product {
        outputs: 3,
        terms: 3,
        seeded: false,
    }]);

    let mut values = case.inputs();
    let trace = case.chiplet.trace_forged(&mut values, &[forgery]).unwrap();

    let proof = case.proof(trace, &values);
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

fn host_trace(tokens: &[Poly], values: &PolyValues, rows: usize) -> ColumnTrace {
    let mut tb = TraceBuilder::new(&HOST_LAYOUT, rows.trailing_zeros() as usize).unwrap();

    for (t, &poly) in tokens.iter().enumerate() {
        for (pos, &c) in values.get(poly).unwrap().iter().enumerate() {
            let row = t * N + pos;

            tb.set_b16(0, row, Block16(poly.id())).unwrap();
            tb.set_b16(1, row, Block16(pos as u16)).unwrap();
            tb.set_b32(2, row, Block32::from(c)).unwrap();
            tb.set_bit(3, row, Bit::ONE).unwrap();
        }
    }

    tb.build()
}

fn multiply_ntts(f: &[u32; N], g: &[u32; N]) -> [u32; N] {
    let mul = |a: u32, b: u32| (a as u64 * b as u64 % Q as u64) as u32;

    let mut h = [0u32; N];
    for i in 0..N / 2 {
        let rev = (i as u32).reverse_bits() >> 25;
        let gamma = pow(ZETA, 2 * rev as u64 + 1);

        let (a0, a1, b0, b1) = (f[2 * i], f[2 * i + 1], g[2 * i], g[2 * i + 1]);

        h[2 * i] = (mul(a0, b0) + mul(mul(a1, b1), gamma)) % Q;
        h[2 * i + 1] = (mul(a0, b1) + mul(a1, b0)) % Q;
    }

    h
}

fn pow(base: u64, exp: u64) -> u32 {
    (0..exp).fold(1u64, |acc, _| acc * base % Q as u64) as u32
}

fn height(rows: usize) -> usize {
    rows.next_power_of_two()
        .max(Config::prod().min_table_rows())
}

fn next(seed: &mut u64) -> u32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);

    (*seed >> 33) as u32
}

#[test]
fn keygen_product_matches_fips_and_proves() {
    honest(&[Product {
        outputs: 3,
        terms: 3,
        seeded: true,
    }]);
}

#[test]
fn keygen_product_copies_its_terms_and_proves() {
    let case = Case::build(
        &[Product {
            outputs: 3,
            terms: 3,
            seeded: true,
        }],
        true,
    );

    let (trace, values) = case.run();

    case.check_outputs(&values);

    let proof = case.proof(trace, &values);

    assert!(proof.report().is_clean());
    assert!(proof.accepted(false));
    assert!(proof.accepted(true));
}

#[test]
fn encaps_products_match_fips_and_prove() {
    honest(&[
        Product {
            outputs: 3,
            terms: 3,
            seeded: false,
        },
        Product {
            outputs: 1,
            terms: 3,
            seeded: false,
        },
    ]);
}

#[test]
fn decaps_product_matches_fips_and_proves() {
    honest(&[Product {
        outputs: 1,
        terms: 4,
        seeded: false,
    }]);
}

#[test]
fn transposed_matrix_breaks_only_its_labels() {
    let honest = Case::new(&[Product {
        outputs: 2,
        terms: 2,
        seeded: false,
    }]);

    let mut steps = honest.steps.clone();
    let m = &mut steps[0].m;

    let upper = m[0][1];
    m[0][1] = m[1][0];
    m[1][0] = upper;

    let forged = Case::from_steps(steps);

    assert_forgery_breaks_only(&honest, &forged, honest.chiplet.layout().poly_a);
}

#[test]
fn shifted_copy_of_operand_breaks_only_poly_arith_copy() {
    assert_breaks_only(
        PolyArithForgery::Operand {
            step: 0,
            row: 1,
            delta: 1,
        },
        "poly_arith_copy",
    );
}

#[test]
fn shifted_register_breaks_only_poly_arith_reg() {
    assert_breaks_only(
        PolyArithForgery::Register {
            step: 0,
            row: 3,
            delta: 1,
        },
        "poly_arith_reg",
    );
}

#[test]
fn unseeded_product_starting_off_zero_breaks_only_poly_arith_acc() {
    assert_breaks_only(
        PolyArithForgery::Start {
            step: 0,
            row: 0,
            delta: 1,
        },
        "poly_arith_acc",
    );
}

#[test]
fn products_table_cannot_hold_are_refused() {
    let mut labels = PolyLabels::new();
    let [a, b, c, d, e] = [(); 5].map(|_| labels.fresh().unwrap());

    let message = |r: Option<Error>| match r {
        Some(Error::Protocol { message, .. }) => message,
        _ => "",
    };

    let product = |m: Poly, v: Poly, out: Poly| {
        BaseCaseMac::new(vec![vec![m]], vec![v], None, vec![out]).unwrap()
    };

    assert_eq!(
        message(BaseCaseMac::new(vec![], vec![a], None, vec![]).err()),
        "product needs at least one output and one term"
    );
    assert_eq!(
        message(BaseCaseMac::new(vec![vec![a, b]], vec![c], None, vec![d]).err()),
        "product needs m[output][term] and one seed per output"
    );
    assert_eq!(
        message(product(a, b, c).with_copy(vec![]).err()),
        "product copies each of its terms once"
    );

    for (steps, rows, expected) in [
        (
            vec![product(a, b, c)],
            64,
            "products need more rows than the table holds",
        ),
        (
            vec![product(a, b, c), product(a, d, e)],
            512,
            "products read a polynomial label twice",
        ),
        (
            vec![product(a, b, c), product(d, e, c)],
            512,
            "products write a polynomial label twice",
        ),
        (
            vec![product(a, b, a)],
            512,
            "product reads its own output label",
        ),
    ] {
        assert_eq!(
            message(PolyArithChiplet::<F>::new(steps, rows).err()),
            expected
        );
    }
}

#[test]
fn forgery_delta_past_q_is_refused() {
    let case = Case::new(&[Product {
        outputs: 2,
        terms: 2,
        seeded: false,
    }]);

    let mut values = case.inputs();

    let forgery = PolyArithForgery::Start {
        step: 0,
        row: 0,
        delta: NttParams::ML_KEM.q(),
    };

    assert!(case.chiplet.trace_forged(&mut values, &[forgery]).is_err());
}

#[test]
fn host_reading_different_output_is_caught_by_bus_alone() {
    let case = Case::new(&[Product {
        outputs: 2,
        terms: 3,
        seeded: true,
    }]);

    let (trace, values) = case.run();

    let target = case.consumed()[1];
    let mut tampered = PolyValues::default();

    for poly in case.tokens() {
        let mut coeffs = *values.get(poly).unwrap();
        if poly == target {
            coeffs[33] = (coeffs[33] + 1) % Q;
        }

        tampered.insert(poly, coeffs).unwrap();
    }

    let proof = case.proof(trace, &tampered);
    let report = proof.report();

    assert!(report.constraint_violations.is_empty());
    assert!(report.fixed_column_violations.is_empty());
    assert!(!report.bus_diagnostics.is_empty());
    assert!(proof.rejected());
}

#[test]
fn non_canonical_input_is_refused() {
    let case = Case::new(&[Product {
        outputs: 2,
        terms: 2,
        seeded: true,
    }]);

    let honest = case.inputs();

    for target in case.produced() {
        for bad in [Q, u32::MAX] {
            let mut values = PolyValues::default();
            for poly in case.produced() {
                let mut coeffs = *honest.get(poly).unwrap();
                if poly == target {
                    coeffs[7] = bad;
                }

                values.insert(poly, coeffs).unwrap();
            }

            assert!(case.chiplet.trace(&mut values).is_err());
        }
    }
}

#[test]
fn scribble_poly_arith_row_mutations_caught() {
    let case = Case::new(&[Product {
        outputs: 2,
        terms: 2,
        seeded: true,
    }]);

    let (trace, values) = case.run();
    let proof = case.proof(trace, &values);

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
