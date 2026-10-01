// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::config::Config;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder, TraceColumn};
use hekate_crypto::DefaultHasher;
use hekate_crypto::transcript::Transcript;
use hekate_math::{Bit, Block16, Block32, Block64, Block128, HardwareField, TowerField};
use hekate_pqc::high_bits::{HighBitsChiplet, HighBitsRow};
use hekate_pqc::mldsa::{MlDsaParams, Q};
use hekate_pqc::wiring::{
    COEF_BUS_ID, HINT_BUS_ID, LANE_BUS_ID, N, Poly, PolyLabels, PolyValues, Stream, coef_spec,
    hint_spec, lane_spec,
};
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::digest::program_id;
use hekate_program::{FixedShape, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_scribble::{MutationKind, ScribbleConfig, assert_all_caught_all_targets};
use hekate_sdk::preflight::{PreflightReport, TableId, preflight};
use hekate_verifier::HekateVerifier;

type F = Block128;
type H = DefaultHasher;

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
    ColumnType::B64,
    ColumnType::Bit,
];

struct Case {
    params: MlDsaParams,
    inputs: Vec<Poly>,
    lanes: Stream,
    values: PolyValues,
    hints: Vec<[bool; N]>,
    rows: Vec<HighBitsRow>,
    chiplet: HighBitsChiplet<F>,
}

#[derive(Default)]
struct Statement {
    coefs: Vec<(u16, u16, u32)>,
    hints: Vec<(u16, u16)>,
    lanes: Vec<(u16, u16, u64)>,
}

struct Proof {
    program: CircuitProgram<F>,
    instance: ProgramInstance<F>,
    witness: ProgramWitness<F>,
}

impl Case {
    fn new(params: MlDsaParams, corner_hint: bool) -> Self {
        let mut labels = PolyLabels::new();

        let inputs: Vec<Poly> = (0..params.k()).map(|_| labels.fresh().unwrap()).collect();
        let lanes = labels.stream().unwrap();

        let g2 = params.gamma2();
        let corners = [0, g2, g2 + 1, Q - 1, Q - g2, 10 * g2];

        let mut seed = 0x4b1d_u64 ^ g2 as u64;
        let mut values = PolyValues::default();
        let mut hints = Vec::with_capacity(inputs.len());
        let mut rows = Vec::with_capacity(inputs.len() * N);

        for &poly in &inputs {
            let coeffs: [u32; N] = core::array::from_fn(|pos| match corners.get(pos) {
                Some(&c) => c,
                None => (next(&mut seed) % Q as u64) as u32,
            });

            let h: [bool; N] = core::array::from_fn(|pos| match pos < corners.len() {
                true => corner_hint,
                false => next(&mut seed).is_multiple_of(5),
            });

            for (&w, &bit) in coeffs.iter().zip(&h) {
                rows.push(HighBitsRow::new(&params, w, bit).unwrap());
            }

            values.insert(poly, coeffs).unwrap();
            hints.push(h);
        }

        let num_rows = (inputs.len() * N).next_power_of_two();

        Self {
            params,
            chiplet: HighBitsChiplet::new(params, inputs.clone(), lanes, num_rows).unwrap(),
            inputs,
            lanes,
            values,
            hints,
            rows,
        }
    }

    fn statement(&self, rows: &[HighBitsRow]) -> Statement {
        let mut st = Statement::default();
        for (i, (&poly, h)) in self.inputs.iter().zip(&self.hints).enumerate() {
            let w = self.values.get(poly).unwrap();

            for pos in 0..N {
                st.coefs.push((poly.id(), pos as u16, w[pos]));

                if h[pos] {
                    st.hints.push((i as u16 + 1, pos as u16));
                }
            }

            let w1: [u32; N] = core::array::from_fn(|pos| rows[i * N + pos].w1());

            let mut lanes = vec![0u64; N * self.params.w1_bits() / 64];
            self.params.w1_lanes(&w1, &mut lanes).unwrap();

            for &lane in &lanes {
                st.lanes
                    .push((self.lanes.id(), st.lanes.len() as u16, lane));
            }
        }

        if st.hints.len() % 2 == 1 {
            st.hints.push((0, 0));
        }

        st
    }

    fn proof(&self, trace: ColumnTrace, st: &Statement) -> Proof {
        let count = st.coefs.len().max(st.hints.len()).max(st.lanes.len());
        let host_rows = count.next_power_of_two();

        Proof {
            program: self.host_program(st, host_rows),
            instance: ProgramInstance::new(host_rows, Vec::new()),
            witness: ProgramWitness::new(host_trace(st, host_rows)).with_chiplets(vec![trace]),
        }
    }

    fn host_program(&self, st: &Statement, rows: usize) -> CircuitProgram<F> {
        let mut cx = Circuit::<F>::new("HighBitsHost", rows).unwrap();
        let cols = cx.schema(&HOST_LAYOUT);

        let pin = |values: Vec<(usize, u64)>| {
            FixedShape::Sparse(
                values
                    .into_iter()
                    .filter(|&(_, v)| v != 0)
                    .map(|(row, v)| (row, F::from(v as u128)))
                    .collect(),
            )
        };

        let rows_of = |n: usize| (0..n).map(|r| (r, 1)).collect::<Vec<_>>();

        cx.fix(
            cols.at(0),
            pin(st
                .coefs
                .iter()
                .enumerate()
                .map(|(r, c)| (r, c.0 as u64))
                .collect()),
        );
        cx.fix(
            cols.at(1),
            pin(st
                .coefs
                .iter()
                .enumerate()
                .map(|(r, c)| (r, c.1 as u64))
                .collect()),
        );
        cx.fix(cols.at(3), pin(rows_of(st.coefs.len())));

        cx.fix(
            cols.at(4),
            pin(st
                .hints
                .iter()
                .enumerate()
                .map(|(r, h)| (r, h.0 as u64))
                .collect()),
        );
        cx.fix(
            cols.at(5),
            pin(st
                .hints
                .iter()
                .enumerate()
                .map(|(r, h)| (r, h.1 as u64))
                .collect()),
        );
        cx.fix(cols.at(6), pin(rows_of(st.hints.len())));

        cx.fix(
            cols.at(7),
            pin(st
                .lanes
                .iter()
                .enumerate()
                .map(|(r, l)| (r, l.0 as u64))
                .collect()),
        );
        cx.fix(
            cols.at(8),
            pin(st
                .lanes
                .iter()
                .enumerate()
                .map(|(r, l)| (r, l.1 as u64))
                .collect()),
        );
        cx.fix(cols.at(10), pin(rows_of(st.lanes.len())));

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
            HINT_BUS_ID,
            hint_spec(cols.at(4).index(), cols.at(5).index(), cols.at(6).index()),
        );
        cx.bus(
            LANE_BUS_ID,
            lane_spec(
                cols.at(7).index(),
                cols.at(8).index(),
                cols.at(9).index(),
                cols.at(10).index(),
            ),
        );

        cx.attach(self.chiplet.def().unwrap());

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
            b"HighBitsChiplet",
            &self.program,
            &self.instance,
            &self.witness,
            &config,
            [7; 32],
            None,
        )
        .unwrap();

        let mut transcript = Transcript::<H>::new(b"HighBitsChiplet");

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

fn honest_case(case: Case) {
    for (i, (&poly, h)) in case.inputs.iter().zip(&case.hints).enumerate() {
        let w = case.values.get(poly).unwrap();
        for pos in 0..N {
            assert_eq!(
                case.rows[i * N + pos].w1(),
                case.params.use_hint(h[pos], w[pos])
            );
        }
    }

    let trace = case.chiplet.trace_rows(&case.rows).unwrap();
    let proof = case.proof(trace, &case.statement(&case.rows));

    assert!(proof.report().is_clean());
    assert!(proof.accepted(false));
    assert!(proof.accepted(true));
}

fn assert_forgery_breaks_only(case: &Case, forged: &[HighBitsRow], label: &str) {
    let trace = case.chiplet.trace_rows(forged).unwrap();

    assert_breaks_only(&case.proof(trace, &case.statement(forged)), label);
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

    for (r, &(stream, index, lane)) in st.lanes.iter().enumerate() {
        tb.set_b16(7, r, Block16(stream)).unwrap();
        tb.set_b16(8, r, Block16(index)).unwrap();
        tb.set_b64(9, r, Block64(lane)).unwrap();
        tb.set_bit(10, r, Bit::ONE).unwrap();
    }

    tb.build()
}

fn next(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);

    *seed >> 33
}

#[test]
fn ml_dsa_44_use_hint_matches_fips_and_proves() {
    honest_case(Case::new(MlDsaParams::ML_DSA_44, true));
}

#[test]
fn ml_dsa_65_use_hint_matches_fips_and_proves() {
    honest_case(Case::new(MlDsaParams::ML_DSA_65, true));
}

#[test]
fn corners_without_hint_match_fips_and_prove() {
    for params in [MlDsaParams::ML_DSA_44, MlDsaParams::ML_DSA_65] {
        honest_case(Case::new(params, false));
    }
}

#[test]
fn forged_nonzero_bit_breaks_only_its_constraint() {
    let case = Case::new(MlDsaParams::ML_DSA_44, true);
    let p = case.params;

    for (pos, nz, label) in [(1, false, "hb_nonzero_set"), (5, true, "hb_nonzero_unset")] {
        let mut forged = case.rows.clone();
        forged[pos] = forged[pos].with_nonzero(&p, nz);

        assert_ne!(forged[pos].w1(), case.rows[pos].w1());
        assert_forgery_breaks_only(&case, &forged, label);
    }
}

#[test]
fn hint_applied_downward_breaks_only_hb_dir() {
    let case = Case::new(MlDsaParams::ML_DSA_44, true);

    let mut forged = case.rows.clone();
    forged[1] = forged[1].with_direction(&case.params, false);

    assert_ne!(forged[1].w1(), case.rows[1].w1());
    assert_forgery_breaks_only(&case, &forged, "hb_dir");
}

#[test]
fn coefficient_at_q_breaks_only_hb_w_range() {
    let case = Case::new(MlDsaParams::ML_DSA_44, true);

    let at = case.hints.iter().flatten().position(|&h| h).unwrap();

    let mut forged = case.rows.clone();
    forged[at] = HighBitsRow::unchecked(&case.params, Q, true);

    let fips = HighBitsRow::new(&case.params, 0, true).unwrap();

    assert_ne!(forged[at].w1(), fips.w1());

    let trace = case.chiplet.trace_rows(&forged).unwrap();

    let mut st = case.statement(&forged);
    st.coefs[at].2 = Q;

    assert_breaks_only(&case.proof(trace, &st), "hb_w_range");
}

#[test]
fn lane_other_than_its_bits_breaks_only_hb_lane() {
    let case = Case::new(MlDsaParams::ML_DSA_65, true);
    let ly = case.chiplet.layout();

    let mut trace = case.chiplet.trace_rows(&case.rows).unwrap();

    let TraceColumn::Bit(emit) = &trace.columns[ly.physical(ly.emit).unwrap()] else {
        panic!("the emit flag is a Bit column");
    };

    let row = emit.iter().position(|&on| on == Bit::ONE).unwrap();

    let TraceColumn::B64(lanes) = &mut trace.columns[ly.physical(ly.lane).unwrap()] else {
        panic!("lanes live in B64 columns");
    };

    let forged = lanes[row].to_tower().0.rotate_left(4);
    lanes[row] = Block64(forged).to_hardware();

    let mut st = case.statement(&case.rows);

    assert_ne!(st.lanes[0].2, forged);

    st.lanes[0].2 = forged;

    assert_breaks_only(&case.proof(trace, &st), "hb_lane");
}

#[test]
fn hint_bit_flipped_at_high_bits_is_caught_by_bus_alone() {
    let case = Case::new(MlDsaParams::ML_DSA_44, true);

    let pos = case.hints[0].iter().position(|&h| !h).unwrap();
    let w = case.values.get(case.inputs[0]).unwrap()[pos];

    let mut forged = case.rows.clone();
    forged[pos] = HighBitsRow::new(&case.params, w, true).unwrap();

    let trace = case.chiplet.trace_rows(&forged).unwrap();
    let proof = case.proof(trace, &case.statement(&forged));
    let report = proof.report();

    assert!(report.constraint_violations.is_empty());
    assert!(report.fixed_column_violations.is_empty());
    assert!(!report.bus_diagnostics.is_empty());
    assert!(
        report
            .bus_diagnostics
            .iter()
            .all(|d| d.bus_id == HINT_BUS_ID)
    );
    assert!(proof.rejected());
}

#[test]
fn host_reading_different_lane_is_caught_by_bus_alone() {
    let case = Case::new(MlDsaParams::ML_DSA_65, true);

    let mut st = case.statement(&case.rows);
    st.lanes[3].2 ^= 1 << 17;

    let trace = case.chiplet.trace_rows(&case.rows).unwrap();
    let proof = case.proof(trace, &st);
    let report = proof.report();

    assert!(report.constraint_violations.is_empty());
    assert!(report.fixed_column_violations.is_empty());
    assert!(!report.bus_diagnostics.is_empty());
    assert!(proof.rejected());
}

#[test]
fn hint_key_without_its_bit_is_caught_by_bus_alone() {
    let case = Case::new(MlDsaParams::ML_DSA_44, true);

    let mut st = case.statement(&case.rows);
    let unset = case.hints[0].iter().position(|&h| !h).unwrap();

    st.hints[0] = (1, unset as u16);

    let trace = case.chiplet.trace_rows(&case.rows).unwrap();
    let proof = case.proof(trace, &st);
    let report = proof.report();

    assert!(report.constraint_violations.is_empty());
    assert!(!report.bus_diagnostics.is_empty());
    assert!(proof.rejected());
}

#[test]
fn non_canonical_input_is_refused() {
    let p = MlDsaParams::ML_DSA_87;

    assert!(HighBitsRow::new(&p, Q, true).is_err());
    assert!(HighBitsRow::new(&p, u32::MAX, false).is_err());
}

#[test]
fn scribble_high_bits_row_mutations_caught() {
    let case = Case::new(MlDsaParams::ML_DSA_44, true);
    let trace = case.chiplet.trace_rows(&case.rows).unwrap();
    let proof = case.proof(trace, &case.statement(&case.rows));

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
