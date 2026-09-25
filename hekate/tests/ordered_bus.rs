// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate_crypto::DefaultHasher;
use hekate_crypto::transcript::Transcript;
use hekate_math::{Bit, Block32, Block128, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::constraint::{BoundaryConstraint, ConstraintAst};
use hekate_program::define_columns;
use hekate_program::digest::program_id;
use hekate_program::expander::VirtualExpander;
use hekate_program::permutation::{BusKind, PermutationCheckSpec, Service, ServiceSlot, Source};
use hekate_program::{
    Air, CadenceSegment, FixedColumn, FixedShape, InlineKernelHint, Program, ProgramInstance,
    ProgramWitness,
};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;

type F = Block128;
type H = DefaultHasher;

const BUS: &str = "square_link";
const LABEL: &[u8] = b"ordered_bus";
const ROWS: usize = 32;
const STEPS: usize = 4;

define_columns! {
    SquarerColumns {
        VAL: B32,
        S_EMIT: Bit,
        S_STEP: Bit,
    }
}

define_columns! {
    HostColumns {
        VAL: B32,
        SEL: Bit,
    }
}

#[derive(Clone, Copy)]
enum Defect {
    None,
    DenseSelector,
    WitnessClock,
    LookupKind,
}

#[derive(Clone)]
struct Twin {
    honest: CircuitProgram<F>,
    defect: Defect,
}

impl Air<F> for Twin {
    fn name(&self) -> String {
        self.honest.name()
    }

    fn num_columns(&self) -> usize {
        self.honest.num_columns()
    }

    fn boundary_constraints(&self) -> Vec<BoundaryConstraint<F>> {
        self.honest.boundary_constraints()
    }

    fn column_layout(&self) -> &[ColumnType] {
        self.honest.column_layout()
    }

    fn virtual_column_layout(&self) -> &[ColumnType] {
        self.honest.virtual_column_layout()
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        let mut checks = self.honest.permutation_checks();
        let spec = &mut checks[0].1;

        match self.defect {
            Defect::WitnessClock => spec.sources[1].0 = Source::Column(HostColumns::VAL),
            Defect::LookupKind => spec.kind = BusKind::Lookup,
            Defect::None | Defect::DenseSelector => {}
        }

        checks
    }

    fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
        let mut pins = self.honest.fixed_columns();
        if let Defect::DenseSelector = self.defect {
            let num_vars = ROWS.trailing_zeros() as usize;
            let rows = (0..ROWS)
                .map(|row| pins[0].shape.value_at_row(row, num_vars).to_tower())
                .collect();

            pins[0].shape = FixedShape::Dense(rows);
        }

        pins
    }

    fn virtual_expander(&self) -> Option<&VirtualExpander> {
        self.honest.virtual_expander()
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        self.honest.constraint_ast()
    }

    fn inline_chiplets(&self) -> errors::Result<Vec<ChipletDef<F>>> {
        self.honest.inline_chiplets()
    }

    fn inline_chiplet_kernels(&self) -> Vec<InlineKernelHint> {
        self.honest.inline_chiplet_kernels()
    }
}

impl Program<F> for Twin {
    fn num_public_inputs(&self) -> usize {
        self.honest.num_public_inputs()
    }

    fn chiplet_defs(&self) -> errors::Result<Vec<ChipletDef<F>>> {
        self.honest.chiplet_defs()
    }
}

fn service() -> Service {
    Service {
        bus_id: BUS,
        kind: BusKind::Permutation,
        slots: vec![
            ServiceSlot::Value(b"kappa_square_value"),
            ServiceSlot::EmitRank,
        ],
        clock_waiver: None,
    }
}

fn squarer(calls: usize) -> ChipletDef<F> {
    pinned_squarer(
        call_rows(calls),
        FixedShape::Cadence {
            stride: STEPS,
            count: calls,
            origin: 0,
            values: indicator(STEPS, &[0, 1, 2]),
        },
    )
}

fn tiled_squarer() -> ChipletDef<F> {
    pinned_squarer(
        FixedShape::Periodic {
            period: STEPS,
            values: indicator(STEPS, &[0, STEPS - 1]),
        },
        FixedShape::Periodic {
            period: STEPS,
            values: indicator(STEPS, &[0, 1, 2]),
        },
    )
}

fn pinned_squarer(emit_rows: FixedShape<F>, step_rows: FixedShape<F>) -> ChipletDef<F> {
    let mut cx = Circuit::<F>::new("Squarer", ROWS).unwrap();

    let cols = cx.schema(&SquarerColumns::build_layout());

    let value = cols.at(SquarerColumns::VAL);
    let emit = cols.at(SquarerColumns::S_EMIT);
    let step = cols.at(SquarerColumns::S_STEP);

    cx.fix(emit, emit_rows);
    cx.fix(step, step_rows);

    let cs = cx.cs();
    let v = cs.col(value.index());

    cs.assert_zero_when(cs.col(step.index()), cs.next(value.index()) + v * v);

    let respond = service()
        .respond(&[value.index()], &[], emit.index())
        .unwrap();

    cx.bus(BUS, respond);

    ChipletDef::from_air(&cx.compile().unwrap()).unwrap()
}

fn asker() -> ChipletDef<F> {
    let mut cx = Circuit::<F>::new("Asker", ROWS).unwrap();

    let cols = cx.schema(&HostColumns::build_layout());

    let value = cols.at(HostColumns::VAL);
    let sel = cols.at(HostColumns::SEL);

    cx.fix(sel, call_rows(1));
    cx.call(&service(), &[value], sel).unwrap();

    ChipletDef::from_air(&cx.compile().unwrap()).unwrap()
}

fn dense_host(calls: usize, server: ChipletDef<F>) -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("DenseSquareHost", ROWS).unwrap();

    let cols = cx.schema(&HostColumns::build_layout());

    let value = cols.at(HostColumns::VAL);
    let sel = cols.at(HostColumns::SEL);

    cx.fix(sel, call_rows(calls));
    cx.call(&service(), &[value], sel).unwrap();

    for call in 0..calls {
        cx.publish(value, call * STEPS);
        cx.publish(value, call * STEPS + STEPS - 1);
    }

    cx.attach(server);

    cx.compile().unwrap()
}

fn sparse_layout() -> [(usize, usize); 2] {
    [(2, 5), (11, 9)]
}

fn sparse_host() -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("SparseSquareHost", ROWS).unwrap();

    let cols = cx.schema(&HostColumns::build_layout());

    let value = cols.at(HostColumns::VAL);
    let sel = cols.at(HostColumns::SEL);

    let segments = sparse_layout()
        .iter()
        .map(|&(origin, gap)| CadenceSegment {
            stride: gap + 1,
            count: 1,
            origin,
            values: indicator(gap + 1, &[0, gap]),
        })
        .collect();

    cx.fix(sel, FixedShape::Segments(segments));
    cx.call(&service(), &[value], sel).unwrap();

    for (origin, gap) in sparse_layout() {
        cx.publish(value, origin);
        cx.publish(value, origin + gap);
    }

    cx.attach(squarer(2));

    cx.compile().unwrap()
}

fn chain_host() -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("SquareChainHost", ROWS).unwrap();

    let cols = cx.schema(&HostColumns::build_layout());

    let value = cols.at(HostColumns::VAL);
    let sel = cols.at(HostColumns::SEL);
    let chain = cx.column(ColumnType::Bit);

    cx.fix(sel, call_rows(3));
    cx.fix(
        chain,
        FixedShape::Sparse(vec![(STEPS - 1, F::ONE), (2 * STEPS - 1, F::ONE)]),
    );

    let cs = cx.cs();

    cs.assert_zero_when(
        cs.col(chain.index()),
        cs.next(value.index()) + cs.col(value.index()),
    );

    cx.call(&service(), &[value], sel).unwrap();

    cx.publish(value, 0);

    for call in 0..3 {
        cx.publish(value, call * STEPS + STEPS - 1);
    }

    cx.attach(squarer(3));

    cx.compile().unwrap()
}

fn collision_host() -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("CollisionHost", ROWS).unwrap();

    let cols = cx.schema(&HostColumns::build_layout());

    let value = cols.at(HostColumns::VAL);
    let sel = cols.at(HostColumns::SEL);

    cx.fix(sel, call_rows(1));
    cx.call(&service(), &[value], sel).unwrap();

    cx.publish(value, 0);
    cx.publish(value, STEPS - 1);

    cx.attach(asker());
    cx.attach(squarer(2));

    cx.compile().unwrap()
}

fn indicator(len: usize, ones: &[usize]) -> Vec<F> {
    (0..len)
        .map(|i| match ones.contains(&i) {
            true => F::ONE,
            false => F::ZERO,
        })
        .collect()
}

fn call_rows(calls: usize) -> FixedShape<F> {
    FixedShape::Cadence {
        stride: STEPS,
        count: calls,
        origin: 0,
        values: indicator(STEPS, &[0, STEPS - 1]),
    }
}

fn squarings(x: Block32) -> [Block32; STEPS] {
    let mut chain = [x; STEPS];
    for step in 1..STEPS {
        chain[step] = chain[step - 1] * chain[step - 1];
    }

    chain
}

fn f(x: Block32) -> Block32 {
    squarings(x)[STEPS - 1]
}

fn inputs() -> [Block32; 3] {
    [
        Block32(0x1234_5679),
        Block32(0x9ABC_DEF1),
        Block32(0x0F1E_2D3C),
    ]
}

fn squarer_trace(rows: usize, inputs: &[Block32]) -> ColumnTrace {
    let mut tb = TraceBuilder::new(
        &SquarerColumns::build_layout(),
        rows.trailing_zeros() as usize,
    )
    .unwrap();

    for (call, &x) in inputs.iter().enumerate() {
        let origin = call * STEPS;

        for (step, value) in squarings(x).into_iter().enumerate() {
            tb.set_b32(SquarerColumns::VAL, origin + step, value)
                .unwrap();

            if step + 1 < STEPS {
                tb.set_bit(SquarerColumns::S_STEP, origin + step, Bit::ONE)
                    .unwrap();
            }
        }

        tb.set_bit(SquarerColumns::S_EMIT, origin, Bit::ONE)
            .unwrap();
        tb.set_bit(SquarerColumns::S_EMIT, origin + STEPS - 1, Bit::ONE)
            .unwrap();
    }

    tb.build()
}

fn host_trace(width: usize, emits: &[(usize, Block32)], chain: &[usize]) -> ColumnTrace {
    let mut layout = HostColumns::build_layout();
    layout.resize(width, ColumnType::Bit);

    let mut tb = TraceBuilder::new(&layout, ROWS.trailing_zeros() as usize).unwrap();

    for &(row, value) in emits {
        tb.set_b32(HostColumns::VAL, row, value).unwrap();
        tb.set_bit(HostColumns::SEL, row, Bit::ONE).unwrap();
    }

    for &row in chain {
        tb.set_bit(HostColumns::NUM_COLUMNS, row, Bit::ONE).unwrap();
    }

    tb.build()
}

fn dense_emits(calls: &[(Block32, Block32)]) -> Vec<(usize, Block32)> {
    calls
        .iter()
        .enumerate()
        .flat_map(|(call, &(x, y))| [(call * STEPS, x), (call * STEPS + STEPS - 1, y)])
        .collect()
}

fn public(emits: &[(usize, Block32)]) -> Vec<F> {
    emits.iter().map(|&(_, value)| F::from(value)).collect()
}

fn verdicts<P: Program<F> + Sync>(
    program: &P,
    public: &[F],
    main: &ColumnTrace,
    chiplets: &[ColumnTrace],
) -> [bool; 2] {
    [false, true].map(|zero_knowledge| {
        let config = Config {
            zero_knowledge,
            ..Config::dev()
        };

        let instance = ProgramInstance::new(ROWS, public.to_vec());
        let witness = ProgramWitness::new(main.clone()).with_chiplets(chiplets.to_vec());

        let proof = prove(
            LABEL, program, &instance, &witness, &config, [0x5A; 32], None,
        )
        .expect("the prover proves the witness it is handed");

        let mut transcript = Transcript::<H>::new(LABEL);

        HekateVerifier::<F, H>::verify(
            &program_id::<F, _>(program).unwrap(),
            program,
            &instance,
            &proof,
            &mut transcript,
            &config,
        )
        .unwrap_or(false)
    })
}

fn dense_verdicts(claims: &[(Block32, Block32)], served: &[Block32]) -> [bool; 2] {
    let emits = dense_emits(claims);

    verdicts(
        &dense_host(claims.len(), squarer(claims.len())),
        &public(&emits),
        &host_trace(HostColumns::NUM_COLUMNS, &emits, &[]),
        &[squarer_trace(ROWS, served)],
    )
}

fn tiled_verdicts(claims: &[(Block32, Block32)], served: &[Block32]) -> [bool; 2] {
    let emits = dense_emits(claims);

    verdicts(
        &dense_host(claims.len(), tiled_squarer()),
        &public(&emits),
        &host_trace(HostColumns::NUM_COLUMNS, &emits, &[]),
        &[squarer_trace(STEPS * served.len(), served)],
    )
}

fn sparse_verdicts(claims: &[(Block32, Block32)], served: &[Block32]) -> [bool; 2] {
    let emits: Vec<(usize, Block32)> = sparse_layout()
        .iter()
        .zip(claims)
        .flat_map(|(&(origin, gap), &(x, y))| [(origin, x), (origin + gap, y)])
        .collect();

    verdicts(
        &sparse_host(),
        &public(&emits),
        &host_trace(HostColumns::NUM_COLUMNS, &emits, &[]),
        &[squarer_trace(ROWS, served)],
    )
}

fn chain_verdicts(outputs: [Block32; 3], served: &[Block32]) -> [bool; 2] {
    let seed = inputs()[0];
    let claims = [
        (seed, outputs[0]),
        (outputs[0], outputs[1]),
        (outputs[1], outputs[2]),
    ];

    let emits = dense_emits(&claims);

    let mut published = vec![F::from(seed)];
    published.extend(outputs.map(F::from));

    verdicts(
        &chain_host(),
        &published,
        &host_trace(
            HostColumns::NUM_COLUMNS + 1,
            &emits,
            &[STEPS - 1, 2 * STEPS - 1],
        ),
        &[squarer_trace(ROWS, served)],
    )
}

fn collision_verdicts(
    host: (Block32, Block32),
    asked: (Block32, Block32),
    served: &[Block32],
) -> [bool; 2] {
    let host_emits = dense_emits(&[host]);
    let asked_emits = dense_emits(&[asked]);

    verdicts(
        &collision_host(),
        &public(&host_emits),
        &host_trace(HostColumns::NUM_COLUMNS, &host_emits, &[]),
        &[
            host_trace(HostColumns::NUM_COLUMNS, &asked_emits, &[]),
            squarer_trace(ROWS, served),
        ],
    )
}

fn verify_twin(defect: Defect) -> errors::Result<bool> {
    let [x0, x1, _] = inputs();
    let emits = dense_emits(&[(x0, f(x0)), (x1, f(x1))]);

    let honest = dense_host(2, squarer(2));
    let config = Config::dev();

    let instance = ProgramInstance::new(ROWS, public(&emits));
    let witness = ProgramWitness::new(host_trace(HostColumns::NUM_COLUMNS, &emits, &[]))
        .with_chiplets(vec![squarer_trace(ROWS, &[x0, x1])]);

    let proof = prove(
        LABEL, &honest, &instance, &witness, &config, [0x5A; 32], None,
    )
    .unwrap();

    let twin = Twin { honest, defect };

    let mut transcript = Transcript::<H>::new(LABEL);

    HekateVerifier::<F, H>::verify(
        &program_id::<F, _>(&twin)?,
        &twin,
        &instance,
        &proof,
        &mut transcript,
        &config,
    )
}

fn rejected(message: &'static str) -> errors::Result<bool> {
    Err(errors::Error::Protocol {
        protocol: "logup_bus",
        message,
    })
}

#[test]
fn dense_host_honest_calls_verify() {
    let [x0, x1, _] = inputs();

    assert_eq!(
        dense_verdicts(&[(x0, f(x0)), (x1, f(x1))], &[x0, x1]),
        [true, true]
    );
}

#[test]
fn dense_host_traded_responses_rejected() {
    let [x0, x1, _] = inputs();

    assert_ne!(f(x0), f(x1));
    assert_eq!(
        dense_verdicts(&[(x0, f(x1)), (x1, f(x0))], &[x0, x1]),
        [false, false]
    );
}

#[test]
fn sparse_host_honest_calls_verify() {
    let [x0, x1, _] = inputs();

    assert_eq!(
        sparse_verdicts(&[(x0, f(x0)), (x1, f(x1))], &[x0, x1]),
        [true, true]
    );
}

#[test]
fn sparse_host_traded_responses_rejected() {
    let [x0, x1, _] = inputs();

    assert_eq!(
        sparse_verdicts(&[(x0, f(x1)), (x1, f(x0))], &[x0, x1]),
        [false, false]
    );
}

#[test]
fn honest_chain_verifies() {
    let s = inputs()[0];

    assert_eq!(
        chain_verdicts([f(s), f(f(s)), f(f(f(s)))], &[s, f(s), f(f(s))]),
        [true, true]
    );
}

#[test]
fn rotated_chain_rejected() {
    let s = inputs()[0];

    assert_ne!(f(s), f(f(s)));
    assert_eq!(
        chain_verdicts([f(f(s)), f(s), f(f(f(s)))], &[s, f(s), f(f(s))]),
        [false, false]
    );
}

#[test]
fn two_requester_tables_verify() {
    let [x0, x1, _] = inputs();

    assert_eq!(
        collision_verdicts((x0, f(x0)), (x1, f(x1)), &[x0, x1]),
        [true, true]
    );
}

#[test]
fn two_table_request_collision_rejected() {
    let [x, _, u] = inputs();
    let forged = f(x) + Block32(1);

    assert_eq!(
        collision_verdicts((x, forged), (x, forged), &[u, u]),
        [false, false]
    );
}

#[test]
fn tiled_responder_honest_calls_verify() {
    let [x0, x1, _] = inputs();

    assert_eq!(
        tiled_verdicts(&[(x0, f(x0)), (x1, f(x1))], &[x0, x1]),
        [true, true]
    );
}

#[test]
fn tiled_responder_traded_responses_rejected() {
    let [x0, x1, _] = inputs();

    assert_eq!(
        tiled_verdicts(&[(x0, f(x1)), (x1, f(x0))], &[x0, x1]),
        [false, false]
    );
}

#[test]
fn tiled_responder_taller_than_its_calls_rejected() {
    let [x0, x1, u] = inputs();

    assert_eq!(
        tiled_verdicts(&[(x0, f(x0)), (x1, f(x1))], &[x0, x1, u, u]),
        [false, false]
    );
}

#[test]
fn faithful_twin_verifies() {
    assert_eq!(verify_twin(Defect::None), Ok(true));
}

#[test]
fn dense_rank_selector_rejected_at_verify_entry() {
    assert_eq!(
        verify_twin(Defect::DenseSelector),
        rejected(
            "ordered bus selector must be None or pinned to Cadence, \
             Segments, Periodic or Sparse"
        )
    );
}

#[test]
fn witness_clock_slot_rejected_at_verify_entry() {
    assert_eq!(
        verify_twin(Defect::WitnessClock),
        rejected(
            "ordered bus endpoint has no EmitRank source; a witness column \
             in the clock slot mimics any rank"
        )
    );
}

#[test]
fn lookup_rank_endpoint_rejected_at_verify_entry() {
    assert_eq!(
        verify_twin(Defect::LookupKind),
        rejected("ordered bus endpoint is Lookup kind; ranks order Permutation buses only")
    );
}
