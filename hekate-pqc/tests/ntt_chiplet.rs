// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::config::Config;
use hekate_core::errors::Error;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate_crypto::DefaultHasher;
use hekate_crypto::transcript::Transcript;
use hekate_math::{Bit, Block16, Block32, Block128, TowerField};
use hekate_pqc::ntt::{Mac, NttChiplet, NttForgery, NttParams, NttSchedule, NttStep, Transform};
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

const HOST_LAYOUT: [ColumnType; 4] = [
    ColumnType::B16,
    ColumnType::B16,
    ColumnType::B32,
    ColumnType::Bit,
];

#[derive(Clone, Copy)]
struct Shape {
    params: NttParams,
    mac: bool,
    subtract: bool,
    mac_negate: [bool; 2],
}

struct Case {
    params: NttParams,
    produced: Vec<Poly>,
    consumed: Vec<Poly>,
    chiplet: NttChiplet<F>,
}

struct Proof {
    program: CircuitProgram<F>,
    instance: ProgramInstance<F>,
    witness: ProgramWitness<F>,
}

impl Shape {
    const ML_KEM: Self = Self {
        params: NttParams::ML_KEM,
        mac: false,
        subtract: true,
        mac_negate: [false, false],
    };

    const ML_DSA: Self = Self {
        params: NttParams::ML_DSA,
        mac: true,
        subtract: true,
        mac_negate: [false, true],
    };

    fn build(self) -> Case {
        let mut labels = PolyLabels::new();
        let mut fresh = || labels.fresh().unwrap();

        let [f, f_hat, g_hat, g, h_hat, e, h, k_hat, v, k] = [(); 10].map(|_| fresh());

        let last = match self.subtract {
            true => Transform::inverse_minus(k_hat, v, k),
            false => Transform::inverse_plus(k_hat, v, k),
        };

        let mut steps = vec![
            NttStep::Transform(Transform::forward(f, f_hat)),
            NttStep::Transform(Transform::inverse(g_hat, g)),
            NttStep::Transform(Transform::inverse_plus(h_hat, e, h)),
            NttStep::Transform(last),
        ];

        let mut produced = vec![f, g_hat, h_hat, e, k_hat, v];
        let mut consumed = vec![g, h, k];

        match self.mac {
            true => {
                let [a00, a01, a10, a11, b1, out0, out1] = [(); 7].map(|_| fresh());

                let mac = Mac::new(
                    vec![vec![a00, a01], vec![a10, a11]],
                    vec![f_hat, b1],
                    self.mac_negate.to_vec(),
                    vec![out0, out1],
                )
                .unwrap();

                steps.push(NttStep::Mac(mac));
                produced.extend([a00, a01, a10, a11, b1]);
                consumed.extend([out0, out1]);
            }
            false => consumed.push(f_hat),
        }

        let schedule = NttSchedule::new(self.params, steps, &mut labels).unwrap();
        let num_rows = schedule.rows().next_power_of_two();

        Case {
            params: self.params,
            produced,
            consumed,
            chiplet: NttChiplet::new(schedule, num_rows).unwrap(),
        }
    }
}

impl Case {
    fn inputs(&self) -> PolyValues {
        let q = self.params.q();

        let mut values = PolyValues::default();
        let mut seed = 0x5eed_u64 ^ q as u64;

        for &poly in &self.produced {
            values
                .insert(poly, core::array::from_fn(|_| next(&mut seed) % q))
                .unwrap();
        }

        values
    }

    fn tokens(&self) -> Vec<Poly> {
        self.produced
            .iter()
            .chain(&self.consumed)
            .copied()
            .collect()
    }

    fn run(&self) -> (ColumnTrace, PolyValues) {
        let mut values = self.inputs();
        let trace = self.chiplet.trace(&mut values).unwrap();

        (trace, values)
    }

    fn proof(&self, ntt_trace: ColumnTrace, values: &PolyValues) -> Proof {
        let tokens = self.tokens();
        let rows = (tokens.len() * N).next_power_of_two();

        Proof {
            program: self.host_program(&tokens, rows),
            instance: ProgramInstance::new(rows, Vec::new()),
            witness: ProgramWitness::new(host_trace(&tokens, values, rows))
                .with_chiplets(vec![ntt_trace]),
        }
    }

    fn host_program(&self, tokens: &[Poly], rows: usize) -> CircuitProgram<F> {
        let mut cx = Circuit::<F>::new("NttHost", rows).unwrap();

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

    fn check_outputs(&self, values: &PolyValues) {
        let p = &self.params;
        let q = p.q();
        let get = |set: &[Poly], i: usize| *values.get(set[i]).unwrap();

        let add = |a: u32, b: u32| (a + b) % q;
        let sub = |a: u32, b: u32| (a + q - b) % q;
        let mul = |a: u32, b: u32| ((a as u64 * b as u64) % q as u64) as u32;

        let [f, g_hat, h_hat, e, k_hat, v] = core::array::from_fn(|i| get(&self.produced, i));

        let intt_h = p.intt(&h_hat);
        let intt_k = p.intt(&k_hat);

        assert_eq!(get(&self.consumed, 0), p.intt(&g_hat));
        assert_eq!(
            get(&self.consumed, 1),
            core::array::from_fn(|i| add(e[i], intt_h[i]))
        );
        assert_eq!(
            get(&self.consumed, 2),
            core::array::from_fn(|i| sub(v[i], intt_k[i]))
        );

        let f_hat = p.ntt(&f);

        if self.consumed.len() == 4 {
            assert_eq!(get(&self.consumed, 3), f_hat);
            return;
        }

        let b1 = get(&self.produced, 10);

        let a: [[u32; N]; 4] = core::array::from_fn(|i| get(&self.produced, 6 + i));

        for i in 0..2 {
            let expected: [u32; N] = core::array::from_fn(|m| {
                sub(mul(a[2 * i][m], f_hat[m]), mul(a[2 * i + 1][m], b1[m]))
            });

            assert_eq!(get(&self.consumed, 3 + i), expected);
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
            b"NttChiplet",
            &self.program,
            &self.instance,
            &self.witness,
            &config,
            [7; 32],
            None,
        )
        .unwrap();

        let mut transcript = Transcript::<H>::new(b"NttChiplet");

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

fn k_pke_v_case() -> Case {
    let params = NttParams::ML_KEM;

    let mut labels = PolyLabels::new();
    let mut fresh = || labels.fresh().unwrap();

    let [e2, mu, t_y, s, v] = [(); 5].map(|_| fresh());

    let steps = vec![
        NttStep::Add {
            a: e2,
            b: mu,
            out: s,
        },
        NttStep::Transform(Transform::inverse_plus(t_y, s, v)),
    ];

    let schedule = NttSchedule::new(params, steps, &mut labels).unwrap();
    let num_rows = schedule.rows().next_power_of_two();

    Case {
        params,
        produced: vec![e2, mu, t_y],
        consumed: vec![v],
        chiplet: NttChiplet::new(schedule, num_rows).unwrap(),
    }
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

fn next(seed: &mut u64) -> u32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);

    (*seed >> 33) as u32
}

#[test]
fn ml_kem_transforms_match_fips_and_prove() {
    let case = Shape::ML_KEM.build();
    let (trace, values) = case.run();

    case.check_outputs(&values);

    let proof = case.proof(trace, &values);

    assert!(proof.report().is_clean());
    assert!(proof.accepted(false));
    assert!(proof.accepted(true));
}

#[test]
fn ml_dsa_transforms_and_mac_match_fips_and_prove() {
    let case = Shape::ML_DSA.build();
    let (trace, values) = case.run();

    case.check_outputs(&values);

    let proof = case.proof(trace, &values);

    assert!(proof.report().is_clean());
    assert!(proof.accepted(false));
    assert!(proof.accepted(true));
}

#[test]
fn add_step_computes_k_pke_v_and_proves() {
    let case = k_pke_v_case();
    let (trace, values) = case.run();

    let q = case.params.q();
    let get = |poly: Poly| *values.get(poly).unwrap();

    let [e2, mu, t_y] = [0, 1, 2].map(|i| get(case.produced[i]));
    let intt = case.params.intt(&t_y);

    let v: [u32; N] = core::array::from_fn(|i| (e2[i] + mu[i] + intt[i]) % q);

    assert_eq!(get(case.consumed[0]), v);

    let proof = case.proof(trace, &values);

    assert!(proof.report().is_clean());
    assert!(proof.accepted(false));
    assert!(proof.accepted(true));
}

#[test]
fn transform_under_another_root_breaks_only_twiddle_pin() {
    let q = NttParams::ML_DSA.q();
    let zeta_cubed = ((1753u64 * 1753 % q as u64) * 1753 % q as u64) as u32;
    let params = NttParams::new(q, 8, zeta_cubed, NttParams::ML_DSA.n_inv()).unwrap();

    let honest = Shape::ML_DSA.build();
    let forged = Shape {
        params,
        ..Shape::ML_DSA
    }
    .build();

    assert_forgery_breaks_only(&honest, &forged, honest.chiplet.layout().tw);
}

#[test]
fn scale_sign_flip_breaks_only_its_pin() {
    let honest = Shape::ML_KEM.build();
    let forged = Shape {
        subtract: false,
        ..Shape::ML_KEM
    }
    .build();

    assert_forgery_breaks_only(&honest, &forged, honest.chiplet.layout().neg);
}

#[test]
fn mac_term_sign_flip_breaks_only_its_pin() {
    let honest = Shape::ML_DSA.build();
    let forged = Shape {
        mac_negate: [false, false],
        ..Shape::ML_DSA
    }
    .build();

    assert_forgery_breaks_only(&honest, &forged, honest.chiplet.layout().neg);
}

#[test]
fn forged_accumulator_breaks_only_ntt_register() {
    let case = Shape::ML_DSA.build();
    let mut values = case.inputs();

    let forgery = NttForgery::Register {
        step: 4,
        row: 2,
        delta: 1,
    };

    let trace = case.chiplet.trace_forged(&mut values, &[forgery]).unwrap();

    assert_breaks_only(&case.proof(trace, &values), "ntt_register");
}

#[test]
fn forged_copied_operand_breaks_only_ntt_copy() {
    let case = Shape::ML_DSA.build();
    let mut values = case.inputs();

    let forgery = NttForgery::Operand {
        step: 4,
        row: 1,
        delta: 1,
    };

    let trace = case.chiplet.trace_forged(&mut values, &[forgery]).unwrap();

    assert_breaks_only(&case.proof(trace, &values), "ntt_copy");
}

#[test]
fn accumulator_started_off_zero_breaks_only_ntt_acc() {
    let case = Shape::ML_DSA.build();
    let mut values = case.inputs();

    let forgery = NttForgery::Start {
        step: 4,
        row: 1,
        delta: 1,
    };

    let trace = case.chiplet.trace_forged(&mut values, &[forgery]).unwrap();

    assert_breaks_only(&case.proof(trace, &values), "ntt_acc");
}

#[test]
fn addend_on_inverse_without_one_breaks_only_ntt_scale_zero() {
    let case = Shape::ML_DSA.build();
    let mut values = case.inputs();

    let forgery = NttForgery::Addend {
        step: 1,
        row: case.params.layers() * N / 2 + 5,
        delta: 1,
    };

    let trace = case.chiplet.trace_forged(&mut values, &[forgery]).unwrap();

    assert_breaks_only(&case.proof(trace, &values), "ntt_scale_zero");
}

#[test]
fn schedules_table_cannot_hold_are_refused() {
    let mut labels = PolyLabels::new();

    let [a, b, c] = [(); 3].map(|_| labels.fresh().unwrap());

    let message = |e: Option<Error>| match e {
        Some(Error::Protocol { message, .. }) => message,
        _ => "",
    };

    let forward = |x: Poly, y: Poly| NttStep::Transform(Transform::forward(x, y));
    let schedule = |steps: Vec<NttStep>, labels: &mut PolyLabels| {
        NttSchedule::new(NttParams::ML_KEM, steps, labels).err()
    };

    for (steps, expected) in [
        (
            vec![forward(a, b), forward(a, c)],
            "schedule reads a polynomial label twice",
        ),
        (
            vec![forward(a, c), forward(b, c)],
            "schedule writes a polynomial label twice",
        ),
        (vec![forward(a, a)], "step reads its own output label"),
    ] {
        assert_eq!(message(schedule(steps, &mut labels)), expected);
    }

    assert_eq!(
        message(Mac::new(vec![], vec![a], vec![false], vec![]).err()),
        "MAC needs at least one output and one term"
    );
    assert_eq!(
        message(Mac::new(vec![vec![a]], vec![b, c], vec![false, false], vec![a]).err()),
        "MAC needs a[output][term], one sign per term and one output per row of a"
    );

    let long = NttSchedule::new(NttParams::ML_KEM, vec![forward(a, b)], &mut labels).unwrap();

    assert_eq!(
        message(NttChiplet::<F>::new(long, 512).err()),
        "schedule needs more rows than the table holds"
    );
}

#[test]
fn forgery_delta_past_q_is_refused() {
    let case = Shape::ML_DSA.build();
    let mut values = case.inputs();

    let forgery = NttForgery::Start {
        step: 4,
        row: 1,
        delta: case.params.q(),
    };

    assert!(case.chiplet.trace_forged(&mut values, &[forgery]).is_err());
}

#[test]
fn host_reading_different_output_is_caught_by_bus_alone() {
    let case = Shape::ML_DSA.build();
    let (trace, values) = case.run();

    let target = case.consumed[3];
    let mut tampered = PolyValues::default();

    for poly in case.tokens() {
        let mut coeffs = *values.get(poly).unwrap();

        if poly == target {
            coeffs[17] = (coeffs[17] + 1) % case.params.q();
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
    for case in [Shape::ML_DSA.build(), k_pke_v_case()] {
        let honest = case.inputs();

        for &target in &case.produced {
            for bad in [case.params.q(), u32::MAX] {
                let mut values = PolyValues::default();
                for &poly in &case.produced {
                    let mut coeffs = *honest.get(poly).unwrap();
                    if poly == target {
                        coeffs[5] = bad;
                    }

                    values.insert(poly, coeffs).unwrap();
                }

                assert!(case.chiplet.trace(&mut values).is_err());
            }
        }
    }
}

#[test]
fn scribble_ntt_row_mutations_caught() {
    for case in [Shape::ML_DSA.build(), k_pke_v_case()] {
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
}
