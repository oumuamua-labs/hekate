// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! ML-KEM matrix-vector products in the NTT domain, as one table
//! on the `coef` bus. `∘` is MultiplyNTTs (FIPS 203 Algorithm 11):
//! 128 pairs, each multiplied mod X² − γ_i with γ_i = ζ^(2·BitRev7(i)+1).

mod air;
mod layout;
mod trace;

pub use layout::PolyArithLayout;

use alloc::vec;
use alloc::vec::Vec;
use hekate_core::errors::{self, Error};
use hekate_core::trace::{ColumnTrace, TraceCompatibleField};
use hekate_math::{Flat, HardwareField, PackableField, TowerField};
use hekate_program::FixedShape;
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram, Col};

use crate::ntt::NttParams;
use crate::wiring::{COEF_BUS_ID, N, Pins, Poly, PolyValues, bit, coef_spec, distinct, label};

const PAIRS: usize = N / 2;

/// `out[i] = seed[i] + Σ_j m[i][j] ∘ v[j]` over `coef` labels.
/// KeyGen: `Â∘ŝ + ê`. Encaps: `Â^T∘ŷ`, with `m[i][j] = Â[j][i]`.
/// Decaps: `ŝ^T∘û`, one output.
#[derive(Clone, Debug)]
pub struct BaseCaseMac {
    m: Vec<Vec<Poly>>,
    v: Vec<Poly>,
    seed: Option<Vec<Poly>>,
    out: Vec<Poly>,
    copy: Option<Vec<Poly>>,
}

impl BaseCaseMac {
    /// Builds the product of matrix `m` and vector `v` into `out`,
    /// starting each output from its `seed` polynomial when given.
    pub fn new(
        m: Vec<Vec<Poly>>,
        v: Vec<Poly>,
        seed: Option<Vec<Poly>>,
        out: Vec<Poly>,
    ) -> errors::Result<Self> {
        if out.is_empty() || v.is_empty() {
            return Err(Error::Protocol {
                protocol: "poly_arith_chiplet",
                message: "product needs at least one output and one term",
            });
        }

        if m.len() != out.len()
            || m.iter().any(|row| row.len() != v.len())
            || seed.as_ref().is_some_and(|s| s.len() != out.len())
        {
            return Err(Error::Protocol {
                protocol: "poly_arith_chiplet",
                message: "product needs m[output][term] and one seed per output",
            });
        }

        Ok(Self {
            m,
            v,
            seed,
            out,
            copy: None,
        })
    }

    /// Re-emits each term of `v` under the matching
    /// `copy` label, for a second consumer of the terms.
    pub fn with_copy(mut self, copy: Vec<Poly>) -> errors::Result<Self> {
        if copy.len() != self.v.len() {
            return Err(Error::Protocol {
                protocol: "poly_arith_chiplet",
                message: "product copies each of its terms once",
            });
        }

        self.copy = Some(copy);

        Ok(self)
    }

    /// Rows the product takes in the table.
    pub fn rows(&self) -> usize {
        PAIRS * self.terms() * self.outputs()
    }

    fn outputs(&self) -> usize {
        self.out.len()
    }

    fn terms(&self) -> usize {
        self.v.len()
    }

    fn reads(&self) -> Vec<Poly> {
        self.m
            .iter()
            .flatten()
            .chain(&self.v)
            .chain(self.seed.iter().flatten())
            .copied()
            .collect()
    }

    fn writes(&self) -> Vec<Poly> {
        self.out
            .iter()
            .chain(self.copy.iter().flatten())
            .copied()
            .collect()
    }
}

/// A product row that reads a shifted input: row `row` of step
/// `step` adds `delta`, below q, to its even coefficient mod q.
#[derive(Clone, Copy, Debug)]
pub enum PolyArithForgery {
    Register { step: usize, row: usize, delta: u32 },
    Operand { step: usize, row: usize, delta: u32 },
    Start { step: usize, row: usize, delta: u32 },
}

#[cfg(feature = "forgery")]
impl PolyArithForgery {
    fn delta(self) -> u32 {
        match self {
            Self::Register { delta, .. }
            | Self::Operand { delta, .. }
            | Self::Start { delta, .. } => delta,
        }
    }
}

/// Table proving a list of products. It takes every `m`, `v`
/// and `seed` coefficient from `coef` and puts every `out`
/// and `copy` coefficient there, each once, for the host to match.
#[derive(Clone)]
pub struct PolyArithChiplet<F: TowerField> {
    program: CircuitProgram<F>,
    steps: Vec<BaseCaseMac>,
    layout: PolyArithLayout,
    num_rows: usize,
}

impl<F> PolyArithChiplet<F>
where
    F: TowerField + TraceCompatibleField + PackableField + HardwareField + Send + 'static,
    <F as PackableField>::Packed: Copy + Send + Sync,
    Flat<F>: Send + Sync,
{
    /// A table holds any list of products; each row carries
    /// an accumulator pair per output of the widest product.
    pub fn new(steps: Vec<BaseCaseMac>, num_rows: usize) -> errors::Result<Self> {
        if steps.iter().map(BaseCaseMac::rows).sum::<usize>() > num_rows {
            return Err(Error::Protocol {
                protocol: "poly_arith_chiplet",
                message: "products need more rows than the table holds",
            });
        }

        distinct(
            steps.iter().flat_map(BaseCaseMac::reads),
            "poly_arith_chiplet",
            "products read a polynomial label twice",
        )?;

        distinct(
            steps.iter().flat_map(BaseCaseMac::writes),
            "poly_arith_chiplet",
            "products write a polynomial label twice",
        )?;

        if steps.iter().any(|step| {
            let writes = step.writes();

            step.reads().iter().any(|p| writes.contains(p))
        }) {
            return Err(Error::Protocol {
                protocol: "poly_arith_chiplet",
                message: "product reads its own output label",
            });
        }

        let params = NttParams::ML_KEM;
        let registers = steps.iter().map(BaseCaseMac::outputs).max().unwrap_or(0);

        let mut cx = Circuit::<F>::new("PolyArithChiplet", num_rows)?;

        let copies = steps.iter().any(|s| s.copy.is_some());

        let layout =
            PolyArithLayout::declare(&mut cx, params.q(), params.bit_width(), registers, copies);

        for (col, shape) in pins(&steps, &layout, &params.gammas()) {
            cx.fix(col, shape);
        }

        let ly = &layout;

        for (poly, pos, value, selector) in [
            (ly.poly_a, ly.pos0, ly.v_a0, ly.active),
            (ly.poly_a, ly.pos1, ly.v_a1, ly.active),
            (ly.poly_b, ly.pos0, ly.v_b0, ly.btake),
            (ly.poly_b, ly.pos1, ly.v_b1, ly.btake),
            (ly.poly_s, ly.pos0, ly.v_in0, ly.seed),
            (ly.poly_s, ly.pos1, ly.v_in1, ly.seed),
            (ly.poly_o, ly.pos0, ly.v_out0, ly.emit),
            (ly.poly_o, ly.pos1, ly.v_out1, ly.emit),
        ] {
            cx.bus(
                COEF_BUS_ID,
                coef_spec(poly.index(), pos.index(), value.index(), selector.index()),
            );
        }

        if let Some((label, sel)) = ly.copy {
            for (pos, value) in [(ly.pos0, ly.v_b0), (ly.pos1, ly.v_b1)] {
                cx.bus(
                    COEF_BUS_ID,
                    coef_spec(label.index(), pos.index(), value.index(), sel.index()),
                );
            }
        }

        air::constrain(cx.cs(), params.q(), &layout);

        let program = cx.compile()?;

        Ok(Self {
            program,
            steps,
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
    pub fn layout(&self) -> &PolyArithLayout {
        &self.layout
    }

    /// Computes every product's `out` into `values` from the inputs
    /// there, each coefficient below q, and returns the table's trace.
    pub fn trace(&self, values: &mut PolyValues) -> errors::Result<ColumnTrace> {
        trace::generate(
            &self.program,
            &self.steps,
            &self.layout,
            self.num_rows,
            values,
            &[],
        )
    }

    /// `trace` with the rows `forgeries` names perturbed
    /// and every later row following from them.
    #[cfg(feature = "forgery")]
    pub fn trace_forged(
        &self,
        values: &mut PolyValues,
        forgeries: &[PolyArithForgery],
    ) -> errors::Result<ColumnTrace> {
        if forgeries.iter().any(|f| f.delta() >= NttParams::ML_KEM.q) {
            return Err(Error::Protocol {
                protocol: "poly_arith_chiplet",
                message: "forgery delta is not below q",
            });
        }

        trace::generate(
            &self.program,
            &self.steps,
            &self.layout,
            self.num_rows,
            values,
            forgeries,
        )
    }

    #[cfg(test)]
    pub(crate) fn produced(&self) -> Vec<(&'static str, u16)> {
        self.steps
            .iter()
            .flat_map(BaseCaseMac::writes)
            .map(|p| (COEF_BUS_ID, p.id()))
            .collect()
    }
}

fn pins<F: TowerField>(
    steps: &[BaseCaseMac],
    ly: &PolyArithLayout,
    gammas: &[u32],
) -> Vec<(Col, FixedShape<F>)> {
    let mut gamma = Pins::new();
    let mut poly_a = Pins::new();
    let mut poly_b = Pins::new();
    let mut poly_s = Pins::new();
    let mut poly_o = Pins::new();
    let mut pos0 = Pins::new();
    let mut pos1 = Pins::new();
    let mut active = Pins::new();
    let mut btake = Pins::new();
    let mut seed = Pins::new();
    let mut emit = Pins::new();
    let mut bcont = Pins::new();
    let mut rcont = Pins::new();
    let mut copy_label = Pins::new();
    let mut copy_sel = Pins::new();

    let mut sel: Vec<Pins<F>> = (0..ly.registers).map(|_| Pins::new()).collect();
    let mut selin: Vec<Pins<F>> = (0..ly.registers).map(|_| Pins::new()).collect();

    let mut origin = 0;

    for s in steps {
        let (terms, outputs) = (s.terms(), s.outputs());

        let block = terms * outputs;
        let rows = s.rows();

        let pattern = |f: &dyn Fn(usize, usize) -> F| {
            (0..block)
                .map(|r| f(r / outputs, r % outputs))
                .collect::<Vec<F>>()
        };

        for (p, &g) in gammas.iter().enumerate().take(PAIRS) {
            let row = origin + p * block;

            gamma.run(row, block, F::from(g));
            pos0.run(row, block, F::from(2 * p as u32));
            pos1.run(row, block, F::from(2 * p as u32 + 1));
        }

        poly_a.cadence(origin, PAIRS, pattern(&|j, i| label(s.m[i][j])));
        poly_b.cadence(origin, PAIRS, pattern(&|j, _| label(s.v[j])));
        poly_o.cadence(origin, PAIRS, pattern(&|_, i| label(s.out[i])));

        if let Some(seeds) = &s.seed {
            poly_s.cadence(origin, PAIRS, pattern(&|_, i| label(seeds[i])));
            seed.cadence(origin, PAIRS, pattern(&|j, _| bit(j == 0)));
        }

        if let Some(copies) = &s.copy {
            copy_label.cadence(
                origin,
                PAIRS,
                pattern(&|j, i| if i == 0 { label(copies[j]) } else { F::ZERO }),
            );
            copy_sel.cadence(origin, PAIRS, pattern(&|_, i| bit(i == 0)));
        }

        btake.cadence(origin, PAIRS, pattern(&|_, i| bit(i == 0)));
        bcont.cadence(origin, PAIRS, pattern(&|_, i| bit(i + 1 < outputs)));
        emit.cadence(origin, PAIRS, pattern(&|j, _| bit(j + 1 == terms)));

        active.run(origin, rows, F::ONE);
        rcont.run(origin, rows - 1, F::ONE);

        for (r, (sel, selin)) in sel.iter_mut().zip(&mut selin).enumerate().take(outputs) {
            sel.cadence(origin, PAIRS, pattern(&|_, i| bit(i == r)));
            selin.cadence(origin, PAIRS, pattern(&|j, i| bit(i == r && j > 0)));
        }

        origin += rows;
    }

    let mut columns = vec![
        (ly.v_gamma, gamma.shape()),
        (ly.poly_a, poly_a.shape()),
        (ly.poly_b, poly_b.shape()),
        (ly.poly_s, poly_s.shape()),
        (ly.poly_o, poly_o.shape()),
        (ly.pos0, pos0.shape()),
        (ly.pos1, pos1.shape()),
        (ly.active, active.shape()),
        (ly.btake, btake.shape()),
        (ly.seed, seed.shape()),
        (ly.emit, emit.shape()),
        (ly.bcont, bcont.shape()),
        (ly.rcont, rcont.shape()),
    ];

    for (r, (sel, selin)) in sel.into_iter().zip(selin).enumerate() {
        columns.push((ly.sel.at(r), sel.shape()));
        columns.push((ly.selin.at(r), selin.shape()));
    }

    if let Some((label_col, sel_col)) = ly.copy {
        columns.push((label_col, copy_label.shape()));
        columns.push((sel_col, copy_sel.shape()));
    }

    columns
}
