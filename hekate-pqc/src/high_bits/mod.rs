// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! UseHint of ML-DSA over coefficients on the `coef` bus,
//! emitting hint keys and w1Encode lanes.

mod air;
mod layout;
mod trace;

pub use layout::HighBitsLayout;

use alloc::vec;
use alloc::vec::Vec;
use hekate_core::errors::{self, Error};
use hekate_core::trace::{ColumnTrace, TraceCompatibleField};
use hekate_math::{Block32, Flat, HardwareField, PackableField, TowerField};
use hekate_program::FixedShape;
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram, Col};
use subtle::{
    Choice, ConditionallySelectable, ConstantTimeEq, ConstantTimeGreater, ConstantTimeLess,
};
use zeroize::{Zeroize, Zeroizing};

use crate::mldsa::{MlDsaParams, Q};
use crate::utils::gcd;
use crate::wiring::{
    COEF_BUS_ID, HINT_BUS_ID, LANE_BUS_ID, N, Pins, Poly, PolyValues, Stream, bit, coef_spec,
    distinct, hint_poly, hint_spec, label, lane_spec,
};

const CALLS: usize = 256;

/// The values one table row holds for
/// coefficient `w` and hint bit `h`.
#[derive(Clone, Copy, Zeroize)]
pub struct HighBitsRow {
    w: u32,
    h: bool,
    r1u: u32,
    r0u: u32,
    neg: bool,
    r1p: u32,
    nz: bool,
    inv0: u32,
    dir: bool,
    down: bool,
    delta: u32,
    wrap: bool,
    w1: u32,
}

impl HighBitsRow {
    /// Derives the row for coefficient `w` and hint bit `h`.
    pub fn new(params: &MlDsaParams, w: u32, h: bool) -> errors::Result<Self> {
        if w >= Q {
            return Err(Error::Protocol {
                protocol: "high_bits_chiplet",
                message: "input coefficient is not below q",
            });
        }

        Ok(Self::decomposed(params, w, h))
    }

    #[cfg(feature = "forgery")]
    pub fn unchecked(params: &MlDsaParams, w: u32, h: bool) -> Self {
        Self::decomposed(params, w, h)
    }

    /// Sets the `r0 ≠ 0` witness bit and re-derives what follows.
    #[cfg(feature = "forgery")]
    pub fn with_nonzero(self, params: &MlDsaParams, nz: bool) -> Self {
        let upstream = (self.r1u, self.r0u, self.neg);

        Self::derive(params, self.w, self.h, upstream, nz)
    }

    /// Sets the UseHint direction bit and re-derives what follows.
    #[cfg(feature = "forgery")]
    pub fn with_direction(self, params: &MlDsaParams, dir: bool) -> Self {
        let upstream = (self.r1u, self.r0u, self.neg);

        Self::directed(params, (self.w, self.h), upstream, self.nz, dir)
    }

    /// UseHint(h, w) for this row.
    pub fn w1(&self) -> u32 {
        self.w1
    }

    fn decomposed(params: &MlDsaParams, w: u32, h: bool) -> Self {
        let (r1u, r0u) = params.split(w);

        let neg = bool::from(r0u.ct_gt(&params.gamma2()));
        let nz = bool::from(!r0u.ct_eq(&0));

        Self::derive(params, w, h, (r1u, r0u, neg), nz)
    }

    fn derive(
        params: &MlDsaParams,
        w: u32,
        h: bool,
        (r1u, r0u, neg): (u32, u32, bool),
        nz: bool,
    ) -> Self {
        Self::directed(params, (w, h), (r1u, r0u, neg), nz, !neg & nz)
    }

    fn directed(
        params: &MlDsaParams,
        (w, h): (u32, bool),
        (r1u, r0u, neg): (u32, u32, bool),
        nz: bool,
        dir: bool,
    ) -> Self {
        let m = params.high_bits_range();
        let r1p = r1u + neg as u32;
        let inv0 = u32::conditional_select(&0, &invert(r0u), Choice::from(nz as u8));

        let down = h & !dir;

        let step = u32::conditional_select(&(m - 1), &1, Choice::from(dir as u8));
        let delta = u32::conditional_select(&0, &step, Choice::from(h as u8));

        let sum = r1p + delta;
        let wrap = bool::from(!sum.ct_lt(&m));

        Self {
            w,
            h,
            r1u,
            r0u,
            neg,
            r1p,
            nz,
            inv0,
            dir,
            down,
            delta,
            wrap,
            w1: u32::conditional_select(&sum, &sum.wrapping_sub(m), Choice::from(wrap as u8)),
        }
    }
}

/// Table proving UseHint(h, w) for every coefficient of k
/// polynomials per call. The w1Encode lanes go out on one stream.
#[derive(Clone)]
pub struct HighBitsChiplet<F: TowerField> {
    program: CircuitProgram<F>,
    params: MlDsaParams,
    inputs: Vec<Poly>,
    lanes: Stream,
    layout: HighBitsLayout,
    num_rows: usize,
}

impl<F> HighBitsChiplet<F>
where
    F: TowerField + TraceCompatibleField + PackableField + HardwareField + Send + 'static,
    <F as PackableField>::Packed: Copy + Send + Sync,
    Flat<F>: Send + Sync,
{
    /// A table holds the k polynomials w_i of 1 to 256
    /// calls under one parameter set; other inputs fail.
    pub fn new(
        params: MlDsaParams,
        inputs: Vec<Poly>,
        lanes: Stream,
        num_rows: usize,
    ) -> errors::Result<Self> {
        if inputs.is_empty() || !inputs.len().is_multiple_of(params.k()) {
            return Err(Error::Protocol {
                protocol: "high_bits_chiplet",
                message: "HighBits takes one input polynomial per row of A for every call",
            });
        }

        if inputs.len() / params.k() > CALLS {
            return Err(Error::Protocol {
                protocol: "high_bits_chiplet",
                message: "HighBits serves at most 256 calls",
            });
        }

        if inputs.len() * N > num_rows {
            return Err(Error::Protocol {
                protocol: "high_bits_chiplet",
                message: "inputs need more rows than the table holds",
            });
        }

        distinct(
            inputs.iter(),
            "high_bits_chiplet",
            "inputs name a polynomial label twice",
        )?;

        let mut cx = Circuit::<F>::new("HighBitsChiplet", num_rows)?;

        let layout = HighBitsLayout::declare(&mut cx, &params);

        for (col, shape) in pins(&inputs, params.k(), lanes, &layout) {
            cx.fix(col, shape);
        }

        let ly = &layout;

        cx.bus(
            COEF_BUS_ID,
            coef_spec(
                ly.poly.index(),
                ly.pos.index(),
                ly.word.index(),
                ly.active.index(),
            ),
        );
        cx.bus(
            HINT_BUS_ID,
            hint_spec(ly.hp.index(), ly.hm.index(), ly.active.index()),
        );
        cx.bus(
            LANE_BUS_ID,
            lane_spec(
                ly.lstream.index(),
                ly.lidx.index(),
                ly.lane.index(),
                ly.emit.index(),
            ),
        );

        air::constrain(cx.cs(), &params, &layout);

        let program = cx.compile()?;

        Ok(Self {
            program,
            params,
            inputs,
            lanes,
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
    pub fn layout(&self) -> &HighBitsLayout {
        &self.layout
    }

    /// Traces the table from the values of
    /// the inputs and their hint bits.
    pub fn trace(&self, values: &PolyValues, hints: &[[bool; N]]) -> errors::Result<ColumnTrace> {
        if hints.len() != self.inputs.len() {
            return Err(Error::Protocol {
                protocol: "high_bits_chiplet",
                message: "one hint vector per input polynomial",
            });
        }

        let mut rows = Zeroizing::new(Vec::with_capacity(self.inputs.len() * N));
        for (&poly, h) in self.inputs.iter().zip(hints) {
            let w = values.get(poly)?;

            for (&coeff, &bit) in w.iter().zip(h) {
                rows.push(HighBitsRow::new(&self.params, coeff, bit)?);
            }
        }

        self.trace_rows(&rows)
    }

    /// Traces the table from one row per
    /// coefficient, inputs in order.
    pub fn trace_rows(&self, rows: &[HighBitsRow]) -> errors::Result<ColumnTrace> {
        if rows.len() != self.inputs.len() * N {
            return Err(Error::Protocol {
                protocol: "high_bits_chiplet",
                message: "one row per coefficient of every input polynomial",
            });
        }

        trace::generate(
            &self.program,
            &self.params,
            (&self.inputs, self.lanes),
            &self.layout,
            self.num_rows,
            rows,
        )
    }

    #[cfg(test)]
    pub(crate) fn produced(&self) -> Vec<(&'static str, u16)> {
        alloc::vec![(LANE_BUS_ID, self.lanes.id())]
    }
}

pub(crate) fn slots(b: usize) -> usize {
    64 / gcd(b, 64)
}

pub(crate) fn emits(slot: usize, b: usize) -> bool {
    (slot + 1) * b / 64 > slot * b / 64
}

pub(crate) fn lane_index(row: usize, b: usize) -> usize {
    row * b / 64
}

pub(crate) fn bit_position(slot: usize, t: usize, b: usize, carried: bool) -> Option<u32> {
    let p = slot * b + t;

    match (p / 64 != slot * b / 64) == carried {
        true => Some((p % 64) as u32),
        false => None,
    }
}

fn pins<F: TowerField>(
    inputs: &[Poly],
    k: usize,
    lanes: Stream,
    ly: &HighBitsLayout,
) -> Vec<(Col, FixedShape<F>)> {
    let rows = inputs.len() * N;
    let b = ly.b;
    let groups = rows / slots(b);

    let mut poly = Pins::new();
    let mut hpoly = Pins::new();

    for (i, &input) in inputs.iter().enumerate() {
        let key = hint_poly((i / k) as u8, i % k);

        poly.run(i * N, N, label(input));
        hpoly.run(i * N, N, F::from(key as u32));
    }

    let mut pos = Pins::new();
    pos.cadence(0, inputs.len(), (0..N as u32).map(F::from).collect());

    let lidx = FixedShape::Sparse(
        (0..rows)
            .filter(|&r| emits(r % slots(b), b))
            .map(|r| (r, F::from(lane_index(r, b) as u32)))
            .filter(|&(_, v)| v != F::ZERO)
            .collect(),
    );

    let mut lstream = Pins::new();
    lstream.run(0, rows, F::from(lanes.id() as u32));

    let mut active = Pins::new();
    let mut start = Pins::new();
    let mut cont = Pins::new();
    let mut emit = Pins::new();

    active.run(0, rows, F::ONE);
    start.run(0, 1, F::ONE);
    cont.run(0, rows - 1, F::ONE);
    emit.cadence(0, groups, (0..slots(b)).map(|j| bit(emits(j, b))).collect());

    let mut columns = vec![
        (ly.poly, poly.shape()),
        (ly.hpoly, hpoly.shape()),
        (ly.pos, pos.shape()),
        (ly.lidx, lidx),
        (ly.lstream, lstream.shape()),
        (ly.active, active.shape()),
        (ly.start, start.shape()),
        (ly.cont, cont.shape()),
        (ly.emit, emit.shape()),
    ];

    for t in 0..b {
        let pattern = |carried: bool| -> Vec<F> {
            (0..slots(b))
                .map(|j| match bit_position(j, t, b, carried) {
                    Some(p) => F::from(1u128 << p),
                    None => F::ZERO,
                })
                .collect()
        };

        let mut current = Pins::new();
        let mut carried = Pins::new();

        current.cadence(0, groups, pattern(false));
        carried.cadence(0, groups, pattern(true));

        columns.push((ly.pw.at(t), current.shape()));
        columns.push((ly.pn.at(t), carried.shape()));
    }

    columns
}

fn invert(x: u32) -> u32 {
    let nonzero = !x.ct_eq(&0);
    let inverse = Block32::from(x | u32::conditional_select(&1, &0, nonzero))
        .invert()
        .0;

    u32::conditional_select(&0, &inverse, nonzero)
}
