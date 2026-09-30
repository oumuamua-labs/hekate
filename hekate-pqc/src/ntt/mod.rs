// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! NTTs, NTT-domain multiply-accumulate and sums for ML-KEM
//! and ML-DSA, reading and writing polynomials on the `coef` bus.

mod air;
mod layout;
mod params;
mod trace;

pub use layout::NttLayout;
pub use params::NttParams;

use alloc::vec;
use alloc::vec::Vec;
use hekate_core::errors::{self, Error};
use hekate_core::trace::{ColumnTrace, TraceCompatibleField};
use hekate_math::{Flat, HardwareField, PackableField, TowerField};
use hekate_program::FixedShape;
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram, Col};

use crate::wiring::{
    COEF_BUS_ID, N, Pins, Poly, PolyLabels, PolyValues, bit, coef_spec, distinct, label,
};
use params::Butterfly;

const HALF: usize = N / 2;

/// Direction of a transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NttKind {
    /// NTT, FIPS 203 Algorithm 9 and FIPS 204 Algorithm 41.
    Forward,

    /// NTT⁻¹ with its final multiply by n⁻¹, FIPS 203
    /// Algorithm 10 and FIPS 204 Algorithm 42.
    Inverse,
}

/// One entry of an NTT schedule.
#[derive(Clone, Debug)]
pub enum NttStep {
    /// NTT or NTT⁻¹ of one polynomial, in 896 rows for
    /// ML-KEM and 1024 for ML-DSA. An inverse adds
    /// 256 rows that scale by n⁻¹ and apply its addend.
    Transform(Transform),

    /// Sums of pointwise NTT-domain products, 256 rows per term
    /// per output. Pointwise is the ring product only under the
    /// full ML-DSA NTT; ML-KEM uses [`crate::poly_arith`].
    Mac(Mac),

    /// `out = a + b` mod q, coefficient by coefficient.
    Add { a: Poly, b: Poly, out: Poly },
}

impl NttStep {
    fn reads(&self) -> Vec<Poly> {
        match self {
            NttStep::Transform(t) => t.addend.into_iter().chain([t.input]).collect(),
            NttStep::Mac(m) => m.a.iter().flatten().chain(&m.b).copied().collect(),
            NttStep::Add { a, b, .. } => vec![*a, *b],
        }
    }

    fn writes(&self) -> Vec<Poly> {
        match self {
            NttStep::Transform(t) => vec![t.output],
            NttStep::Mac(m) => m.out.clone(),
            NttStep::Add { out, .. } => vec![*out],
        }
    }
}

/// A row that reads a shifted value: row `row` of schedule
/// step `step` adds `delta`, below q, to it mod q.
#[derive(Clone, Copy, Debug)]
pub enum NttForgery {
    Register { step: usize, row: usize, delta: u32 },
    Operand { step: usize, row: usize, delta: u32 },
    Start { step: usize, row: usize, delta: u32 },
    Addend { step: usize, row: usize, delta: u32 },
}

#[cfg(feature = "forgery")]
impl NttForgery {
    fn delta(self) -> u32 {
        match self {
            Self::Register { delta, .. }
            | Self::Operand { delta, .. }
            | Self::Start { delta, .. }
            | Self::Addend { delta, .. } => delta,
        }
    }
}

/// One 256-coefficient transform between labelled polynomials.
#[derive(Clone, Debug)]
pub struct Transform {
    kind: NttKind,
    input: Poly,
    output: Poly,
    addend: Option<Poly>,
    subtract: bool,
}

impl Transform {
    /// `output = NTT(input)`.
    pub fn forward(input: Poly, output: Poly) -> Self {
        Self {
            kind: NttKind::Forward,
            input,
            output,
            addend: None,
            subtract: false,
        }
    }

    /// `output = NTT⁻¹(input)`.
    pub fn inverse(input: Poly, output: Poly) -> Self {
        Self {
            kind: NttKind::Inverse,
            input,
            output,
            addend: None,
            subtract: false,
        }
    }

    /// `output = addend + NTT⁻¹(input)`.
    pub fn inverse_plus(input: Poly, addend: Poly, output: Poly) -> Self {
        Self {
            kind: NttKind::Inverse,
            input,
            output,
            addend: Some(addend),
            subtract: false,
        }
    }

    /// `output = addend − NTT⁻¹(input)`.
    pub fn inverse_minus(input: Poly, addend: Poly, output: Poly) -> Self {
        Self {
            kind: NttKind::Inverse,
            input,
            output,
            addend: Some(addend),
            subtract: true,
        }
    }

    fn rows(&self, layers: usize) -> usize {
        match self.kind {
            NttKind::Forward => layers * HALF,
            NttKind::Inverse => layers * HALF + N,
        }
    }

    fn pattern_key(&self) -> (NttKind, bool, bool) {
        (self.kind, self.addend.is_some(), self.subtract)
    }
}

/// `out[i] = Σ_j ±a[i][j] ∘ b[j]`,
/// coefficient by coefficient in the NTT domain.
#[derive(Clone, Debug)]
pub struct Mac {
    a: Vec<Vec<Poly>>,
    b: Vec<Poly>,
    negate: Vec<bool>,
    out: Vec<Poly>,
}

impl Mac {
    /// Builds the sum, subtracting term `j` where `negate[j]`.
    ///
    /// # Errors
    /// When `out` or `b` is empty, or the shapes are not
    /// `a[out.len()][b.len()]` and `negate[b.len()]`.
    pub fn new(
        a: Vec<Vec<Poly>>,
        b: Vec<Poly>,
        negate: Vec<bool>,
        out: Vec<Poly>,
    ) -> errors::Result<Self> {
        if out.is_empty() || b.is_empty() {
            return Err(Error::Protocol {
                protocol: "ntt_chiplet",
                message: "MAC needs at least one output and one term",
            });
        }

        if a.len() != out.len()
            || negate.len() != b.len()
            || a.iter().any(|row| row.len() != b.len())
        {
            return Err(Error::Protocol {
                protocol: "ntt_chiplet",
                message: "MAC needs a[output][term], one sign per term and one output per row of a",
            });
        }

        Ok(Self { a, b, negate, out })
    }

    fn outputs(&self) -> usize {
        self.out.len()
    }

    fn terms(&self) -> usize {
        self.b.len()
    }

    fn rows(&self) -> usize {
        N * self.terms() * self.outputs()
    }
}

/// Steps of one NTT table in row order, with
/// the labels its transforms pass between layers.
#[derive(Clone, Debug)]
pub struct NttSchedule {
    params: NttParams,
    steps: Vec<NttStep>,
    stages: Vec<Vec<Poly>>,
}

impl NttSchedule {
    /// Lays out `steps` and draws the between-layer labels from `labels`.
    pub fn new(
        params: NttParams,
        steps: Vec<NttStep>,
        labels: &mut PolyLabels,
    ) -> errors::Result<Self> {
        distinct(
            steps.iter().flat_map(NttStep::reads),
            "ntt_chiplet",
            "schedule reads a polynomial label twice",
        )?;

        distinct(
            steps.iter().flat_map(NttStep::writes),
            "ntt_chiplet",
            "schedule writes a polynomial label twice",
        )?;

        if steps.iter().any(|step| {
            let writes = step.writes();

            step.reads().iter().any(|p| writes.contains(p))
        }) {
            return Err(Error::Protocol {
                protocol: "ntt_chiplet",
                message: "step reads its own output label",
            });
        }

        let mut stages = Vec::with_capacity(steps.len());
        for step in &steps {
            let internal = match step {
                NttStep::Transform(t) if t.kind == NttKind::Forward => params.layers - 1,
                NttStep::Transform(_) => params.layers,
                NttStep::Mac(_) | NttStep::Add { .. } => 0,
            };

            stages.push(
                (0..internal)
                    .map(|_| labels.fresh())
                    .collect::<errors::Result<Vec<Poly>>>()?,
            );
        }

        Ok(Self {
            params,
            steps,
            stages,
        })
    }

    /// Modulus and root of unity every step uses.
    pub fn params(&self) -> NttParams {
        self.params
    }

    /// Rows the steps occupy, before padding.
    pub fn rows(&self) -> usize {
        self.steps
            .iter()
            .map(|step| match step {
                NttStep::Transform(t) => t.rows(self.params.layers),
                NttStep::Mac(m) => m.rows(),
                NttStep::Add { .. } => N,
            })
            .sum()
    }

    /// Accumulators the widest MAC step needs.
    pub fn registers(&self) -> usize {
        self.steps
            .iter()
            .map(|step| match step {
                NttStep::Transform(_) | NttStep::Add { .. } => 0,
                NttStep::Mac(m) => m.outputs(),
            })
            .max()
            .unwrap_or(0)
    }

    fn layer_polys(&self, step: usize, t: &Transform, layer: usize) -> (Poly, Poly) {
        let stages = &self.stages[step];
        let last = self.params.layers - 1;

        let input = match layer {
            0 => t.input,
            _ => stages[layer - 1],
        };

        let output = match (t.kind, layer == last) {
            (NttKind::Forward, true) => t.output,
            _ => stages[layer],
        };

        (input, output)
    }

    fn scale_input(&self, step: usize) -> Poly {
        self.stages[step][self.params.layers - 1]
    }
}

/// Table proving every step of a schedule.
/// It reads inputs and writes outputs on the `coef` bus.
#[derive(Clone)]
pub struct NttChiplet<F: TowerField> {
    program: CircuitProgram<F>,
    schedule: NttSchedule,
    layout: NttLayout,
    num_rows: usize,
}

impl<F> NttChiplet<F>
where
    F: TowerField + TraceCompatibleField + PackableField + HardwareField + Send + 'static,
    <F as PackableField>::Packed: Copy + Send + Sync,
    Flat<F>: Send + Sync,
{
    /// A table holds any schedule; each row carries
    /// one accumulator per output of the widest MAC.
    pub fn new(schedule: NttSchedule, num_rows: usize) -> errors::Result<Self> {
        if schedule.rows() > num_rows {
            return Err(Error::Protocol {
                protocol: "ntt_chiplet",
                message: "schedule needs more rows than the table holds",
            });
        }

        let params = schedule.params();

        let mut cx = Circuit::<F>::new("NttChiplet", num_rows)?;

        let layout =
            NttLayout::declare(&mut cx, params.q, params.bit_width(), schedule.registers());

        for (col, shape) in NttPins::build(&schedule, &layout) {
            cx.fix(col, shape);
        }

        let ly = &layout;

        cx.bus(
            COEF_BUS_ID,
            coef_spec(
                ly.poly1.index(),
                ly.pos1.index(),
                ly.v_in1.index(),
                ly.c1.index(),
            ),
        );
        cx.bus(
            COEF_BUS_ID,
            coef_spec(
                ly.poly2.index(),
                ly.pos2.index(),
                ly.v_in2.index(),
                ly.c2.index(),
            ),
        );
        cx.bus(
            COEF_BUS_ID,
            coef_spec(
                ly.poly3.index(),
                ly.pos1.index(),
                ly.v_out1.index(),
                ly.p1.index(),
            ),
        );
        cx.bus(
            COEF_BUS_ID,
            coef_spec(
                ly.poly3.index(),
                ly.pos2.index(),
                ly.v_out2.index(),
                ly.p2.index(),
            ),
        );

        air::constrain(cx.cs(), params.q, &layout);

        let program = cx.compile()?;

        Ok(Self {
            program,
            schedule,
            layout,
            num_rows,
        })
    }

    /// The table as a chiplet for a host program to attach.
    pub fn def(&self) -> errors::Result<ChipletDef<F>> {
        ChipletDef::from_air(&self.program)
    }

    /// The compiled table: constraints, bus endpoints and fixed columns.
    pub fn program(&self) -> &CircuitProgram<F> {
        &self.program
    }

    /// Column positions, for tests that forge cells of a trace.
    pub fn layout(&self) -> &NttLayout {
        &self.layout
    }

    pub fn params(&self) -> NttParams {
        self.schedule.params()
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Computes every step's outputs into
    /// `values` and returns the table's trace.
    pub fn trace(&self, values: &mut PolyValues) -> errors::Result<ColumnTrace> {
        trace::generate(
            &self.program,
            &self.schedule,
            &self.layout,
            self.num_rows,
            values,
            &[],
        )
    }

    /// `trace` with the rows `forgeries` names
    /// perturbed and every later row following from them.
    #[cfg(feature = "forgery")]
    pub fn trace_forged(
        &self,
        values: &mut PolyValues,
        forgeries: &[NttForgery],
    ) -> errors::Result<ColumnTrace> {
        if forgeries.iter().any(|f| f.delta() >= self.params().q) {
            return Err(Error::Protocol {
                protocol: "ntt_chiplet",
                message: "forgery delta is not below q",
            });
        }

        trace::generate(
            &self.program,
            &self.schedule,
            &self.layout,
            self.num_rows,
            values,
            forgeries,
        )
    }

    #[cfg(test)]
    pub(crate) fn produced(&self) -> Vec<(&'static str, u16)> {
        self.schedule
            .steps
            .iter()
            .flat_map(NttStep::writes)
            .chain(self.schedule.stages.iter().flatten().copied())
            .map(|p| (COEF_BUS_ID, p.id()))
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RowPlan {
    pos1: usize,
    pos2: usize,
    tw: u32,
    gs: bool,
    za: bool,
    neg: bool,
    c1: bool,
    c2: bool,
    p1: bool,
    p2: bool,
}

struct NttPins<F> {
    tw: Pins<F>,
    poly1: Pins<F>,
    poly2: Pins<F>,
    poly3: Pins<F>,
    pos1: Pins<F>,
    pos2: Pins<F>,
    gs: Pins<F>,
    mac: Pins<F>,
    neg: Pins<F>,
    za: Pins<F>,
    c1: Pins<F>,
    c2: Pins<F>,
    p1: Pins<F>,
    p2: Pins<F>,
    bcont: Pins<F>,
    rcont: Pins<F>,
    sel: Vec<Pins<F>>,
    selin: Vec<Pins<F>>,
}

impl<F: TowerField> NttPins<F> {
    fn build(schedule: &NttSchedule, layout: &NttLayout) -> Vec<(Col, FixedShape<F>)> {
        let params = schedule.params();
        let steps = &schedule.steps;

        let mut pins = Self::new(layout.registers);
        let mut origin = 0;
        let mut i = 0;

        while i < steps.len() {
            match &steps[i] {
                NttStep::Transform(t) => {
                    let key = t.pattern_key();

                    let mut end = i + 1;
                    while let Some(NttStep::Transform(next)) = steps.get(end)
                        && next.pattern_key() == key
                    {
                        end += 1;
                    }

                    let plan = transform_plan(&params, t);
                    let rows = plan.len();

                    pins.pattern(origin, end - i, &plan);

                    for (k, step) in steps[i..end].iter().enumerate() {
                        if let NttStep::Transform(t) = step {
                            pins.transform_labels(origin + k * rows, schedule, i + k, t);
                        }
                    }

                    origin += (end - i) * rows;
                    i = end;
                }
                NttStep::Mac(m) => {
                    pins.mac(origin, m);

                    origin += m.rows();
                    i += 1;
                }
                NttStep::Add { .. } => {
                    let mut end = i + 1;
                    while let Some(NttStep::Add { .. }) = steps.get(end) {
                        end += 1;
                    }

                    pins.pattern(origin, end - i, &add_plan().collect::<Vec<RowPlan>>());

                    for (k, step) in steps[i..end].iter().enumerate() {
                        if let NttStep::Add { a, b, out } = step {
                            let row = origin + k * N;

                            pins.poly1.run(row, N, label(*a));
                            pins.poly2.run(row, N, label(*b));
                            pins.poly3.run(row, N, label(*out));
                        }
                    }

                    origin += (end - i) * N;
                    i = end;
                }
            }
        }

        pins.into_columns(layout)
    }

    fn new(registers: usize) -> Self {
        Self {
            tw: Pins::new(),
            poly1: Pins::new(),
            poly2: Pins::new(),
            poly3: Pins::new(),
            pos1: Pins::new(),
            pos2: Pins::new(),
            gs: Pins::new(),
            mac: Pins::new(),
            neg: Pins::new(),
            za: Pins::new(),
            c1: Pins::new(),
            c2: Pins::new(),
            p1: Pins::new(),
            p2: Pins::new(),
            bcont: Pins::new(),
            rcont: Pins::new(),
            sel: (0..registers).map(|_| Pins::new()).collect(),
            selin: (0..registers).map(|_| Pins::new()).collect(),
        }
    }

    fn pattern(&mut self, origin: usize, count: usize, plan: &[RowPlan]) {
        let column = |f: &dyn Fn(&RowPlan) -> F| plan.iter().map(f).collect::<Vec<F>>();

        self.pos1
            .cadence(origin, count, column(&|r| F::from(r.pos1 as u32)));
        self.pos2
            .cadence(origin, count, column(&|r| F::from(r.pos2 as u32)));

        self.tw.cadence(origin, count, column(&|r| F::from(r.tw)));
        self.gs.cadence(origin, count, column(&|r| bit(r.gs)));
        self.za.cadence(origin, count, column(&|r| bit(r.za)));
        self.neg.cadence(origin, count, column(&|r| bit(r.neg)));
        self.c1.cadence(origin, count, column(&|r| bit(r.c1)));
        self.c2.cadence(origin, count, column(&|r| bit(r.c2)));
        self.p1.cadence(origin, count, column(&|r| bit(r.p1)));
        self.p2.cadence(origin, count, column(&|r| bit(r.p2)));
    }

    fn transform_labels(
        &mut self,
        origin: usize,
        schedule: &NttSchedule,
        step: usize,
        t: &Transform,
    ) {
        for layer in 0..schedule.params.layers {
            let (input, output) = schedule.layer_polys(step, t, layer);
            let row = origin + layer * HALF;

            self.poly1.run(row, HALF, label(input));
            self.poly2.run(row, HALF, label(input));
            self.poly3.run(row, HALF, label(output));
        }

        if t.kind == NttKind::Inverse {
            let row = origin + schedule.params.layers * HALF;

            if let Some(addend) = t.addend {
                self.poly1.run(row, N, label(addend));
            }

            self.poly2.run(row, N, label(schedule.scale_input(step)));
            self.poly3.run(row, N, label(t.output));
        }
    }

    fn mac(&mut self, origin: usize, m: &Mac) {
        let (terms, outputs) = (m.terms(), m.outputs());
        let block = terms * outputs;
        let rows = m.rows();

        let pattern = |f: &dyn Fn(usize, usize) -> F| {
            (0..block)
                .map(|r| f(r / outputs, r % outputs))
                .collect::<Vec<F>>()
        };

        for pos in 0..N {
            let value = F::from(pos as u32);

            self.pos1.run(origin + pos * block, block, value);
            self.pos2.run(origin + pos * block, block, value);
        }

        self.poly1
            .cadence(origin, N, pattern(&|j, i| label(m.a[i][j])));
        self.poly2
            .cadence(origin, N, pattern(&|j, _| label(m.b[j])));
        self.poly3
            .cadence(origin, N, pattern(&|_, i| label(m.out[i])));

        self.mac.run(origin, rows, F::ONE);
        self.c1.run(origin, rows, F::ONE);
        self.rcont.run(origin, rows - 1, F::ONE);

        self.neg
            .cadence(origin, N, pattern(&|j, _| bit(m.negate[j])));
        self.c2.cadence(origin, N, pattern(&|_, i| bit(i == 0)));
        self.p1
            .cadence(origin, N, pattern(&|j, _| bit(j == terms - 1)));
        self.bcont
            .cadence(origin, N, pattern(&|_, i| bit(i + 1 < outputs)));

        for r in 0..outputs {
            self.sel[r].cadence(origin, N, pattern(&|_, i| bit(i == r)));
            self.selin[r].cadence(origin, N, pattern(&|j, i| bit(i == r && j > 0)));
        }
    }

    fn into_columns(self, ly: &NttLayout) -> Vec<(Col, FixedShape<F>)> {
        let mut columns = Vec::with_capacity(16 + 2 * ly.registers);

        columns.push((ly.tw, self.tw.shape()));
        columns.push((ly.poly1, self.poly1.shape()));
        columns.push((ly.poly2, self.poly2.shape()));
        columns.push((ly.poly3, self.poly3.shape()));
        columns.push((ly.pos1, self.pos1.shape()));
        columns.push((ly.pos2, self.pos2.shape()));
        columns.push((ly.gs, self.gs.shape()));
        columns.push((ly.mac, self.mac.shape()));
        columns.push((ly.neg, self.neg.shape()));
        columns.push((ly.za, self.za.shape()));
        columns.push((ly.c1, self.c1.shape()));
        columns.push((ly.c2, self.c2.shape()));
        columns.push((ly.p1, self.p1.shape()));
        columns.push((ly.p2, self.p2.shape()));
        columns.push((ly.bcont, self.bcont.shape()));
        columns.push((ly.rcont, self.rcont.shape()));

        for (r, pins) in self.sel.into_iter().enumerate() {
            columns.push((ly.sel.at(r), pins.shape()));
        }

        for (r, pins) in self.selin.into_iter().enumerate() {
            columns.push((ly.selin.at(r), pins.shape()));
        }

        columns
    }
}

fn transform_plan(params: &NttParams, t: &Transform) -> Vec<RowPlan> {
    let butterfly = |bf: &Butterfly, gs: bool| RowPlan {
        pos1: bf.pos_a,
        pos2: bf.pos_b,
        tw: bf.w,
        gs,
        za: false,
        neg: false,
        c1: true,
        c2: true,
        p1: true,
        p2: true,
    };

    match t.kind {
        NttKind::Forward => params
            .forward_plan()
            .iter()
            .map(|bf| butterfly(bf, false))
            .collect(),
        NttKind::Inverse => {
            let mut plan: Vec<RowPlan> = params
                .inverse_plan()
                .iter()
                .map(|bf| butterfly(bf, true))
                .collect();

            plan.extend(scale_plan(params.n_inv(), t.addend.is_some(), t.subtract));

            plan
        }
    }
}

fn add_plan() -> impl Iterator<Item = RowPlan> {
    scale_plan(1, true, false)
}

fn scale_plan(tw: u32, addend: bool, subtract: bool) -> impl Iterator<Item = RowPlan> {
    (0..N).map(move |pos| RowPlan {
        pos1: pos,
        pos2: pos,
        tw,
        gs: false,
        za: !addend,
        neg: subtract,
        c1: addend,
        c2: true,
        p1: true,
        p2: false,
    })
}
