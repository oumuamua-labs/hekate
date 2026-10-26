// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use crate::chiplet::ChipletDef;
use crate::constraint::{BoundaryConstraint, ConstraintAst, ConstraintExpr, ExprId};
use crate::expander::{
    EVAL_SUMCHECK_DEGREE, PoolLayout, RING_BLIND_BITS, RingSwitchPlan, SlotLayout, claim_weights,
    eq_tensor_b,
};
use crate::linearized::{self, RingGadget, linearized_coeffs};
use crate::permutation::{
    BUS_LIMBS, BusColumn, BusElement, BusField, BusFraction, BusGroup, BusValue,
    PermutationCheckSpec, RankClock, bus_alpha_weights, bus_columns, bus_draw_events,
};
use crate::predicate::{AffineRow, ClaimLayout, Form, Unknown, WireRole, compile};
use crate::{Air, FixedColumn, ProgramInstance, ShapeEvaluator};
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::mem;
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::outer::{OUTER_MASK_ROWS, OuterGeometry};
use hekate_core::poly::univariate::{MAX_POINTS, UnivariatePoly};
use hekate_core::tensor::TensorProduct;
use hekate_math::{Block128, Flat, HardwareField, TowerField};

/// Hadamard wires of one GF(2^256) product.
const KARATSUBA_WIRES: usize = 3;

pub struct LinearBatch<F> {
    pub weights: Vec<Vec<Flat<F>>>,
    pub rows: Vec<usize>,
    pub target: Flat<F>,
}

/// Claimed-sum endpoints `(pad, masked limb)` per group and limb.
type BusSums<'r, F> = BTreeMap<(&'r BusGroup, usize), Vec<(u32, Flat<F>)>>;

/// One `h` column in the outer statement: its limb pads and masked
/// limbs, its fractions, its row weight and its table-local first wire.
pub struct BusColumnRow<F> {
    pub h_pads: [u32; BUS_LIMBS],
    pub h_masked: [Flat<F>; BUS_LIMBS],
    pub first: BusFraction<F>,
    pub second: Option<BusFraction<F>>,
    pub eq_lookup: BusValue<F>,
    pub first_wire: u32,
}

pub struct ConsistencyInputs<F> {
    /// Pad index of claim 0, matching `ClaimLayout`.
    pub pad_first: u32,

    pub eq_zc: Flat<F>,
    pub alpha: Flat<F>,

    /// `τ` of the bus field, `y² = y + τ`.
    pub tau: Flat<F>,

    /// `(claim index, public value, eq(r_final, row))`.
    pub boundary: Vec<(u32, Flat<F>, Flat<F>)>,

    /// `(b_k claim index, b_k_next claim index)`.
    pub telescope: Vec<(u32, u32)>,

    pub buses: Vec<BusColumnRow<F>>,

    /// Mask form of the ZeroCheck's final claim,
    /// and the masked value the replay reaches.
    pub val_final_form: Vec<(Unknown, Flat<F>)>,
    pub val_final_masked: Flat<F>,
}

impl<F: HardwareField> ConsistencyInputs<F> {
    /// `α^k · eq_zc` for `k` in `0..roots`, the
    /// weight root `k` carries in the consistency row.
    fn root_weights(&self, roots: usize) -> impl Iterator<Item = Flat<F>> + '_ {
        core::iter::successors(Some(self.eq_zc), |w| Some(*w * self.alpha)).take(roots)
    }
}

/// One table's AST, bus specs, fixed pins and boundary
/// constraints, borrowed where the `Air` getters clone.
#[derive(Clone, Copy)]
pub struct TableStatics<'a, F: TowerField> {
    pub ast: &'a ConstraintAst<F>,
    pub specs: &'a [(String, PermutationCheckSpec)],
    pub fixed: &'a [FixedColumn<F>],
    pub boundary: &'a [BoundaryConstraint<F>],
}

/// The parts behind [`TableStatics`], read once from an
/// `Air` that does not hold them, such as a `Program`.
pub struct TableParts<F: TowerField> {
    pub ast: ConstraintAst<F>,
    pub specs: Vec<(String, PermutationCheckSpec)>,
    pub fixed: Vec<FixedColumn<F>>,
    pub boundary: Vec<BoundaryConstraint<F>>,
}

impl<F: TowerField> TableParts<F> {
    pub fn of(air: &impl Air<F>) -> Self {
        Self {
            ast: air.constraint_ast(),
            specs: air.permutation_checks(),
            fixed: air.fixed_columns(),
            boundary: air.boundary_constraints(),
        }
    }

    pub fn statics(&self) -> TableStatics<'_, F> {
        TableStatics {
            ast: &self.ast,
            specs: &self.specs,
            fixed: &self.fixed,
            boundary: &self.boundary,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TableShape {
    pub num_vars: usize,

    /// Virtual columns, after expansion.
    pub num_columns: usize,

    pub sumcheck_degree: usize,

    /// Packed `h` columns, [`BUS_LIMBS`] limbs each.
    pub bus_columns: usize,

    /// Hadamard wires of the LogUp terms.
    pub bus_wires: usize,

    pub mul_nodes: usize,
    pub ring_units: bool,
    pub whole_blinds: usize,
}

impl TableShape {
    pub fn from_air<F: TowerField>(
        air: &impl Air<F>,
        statics: &TableStatics<'_, F>,
    ) -> errors::Result<Self> {
        let ast = statics.ast;

        ast.validate_order()?;

        let mut sumcheck_degree = ast.max_degree() + 1;

        if !statics.boundary.is_empty() {
            sumcheck_degree = sumcheck_degree.max(2);
        }

        if !statics.specs.is_empty() {
            sumcheck_degree = sumcheck_degree.max(3);
        }

        let columns = bus_columns(statics.specs, sumcheck_degree);
        let bus_wires = columns
            .iter()
            .map(|column| column_wires(column, statics.specs))
            .sum();

        if sumcheck_degree + 1 > MAX_POINTS {
            return Err(errors::Error::Protocol {
                protocol: "air",
                message: "sumcheck degree exceeds the interpolation stack bound",
            });
        }

        if air.num_columns() != air.virtual_column_layout().len() {
            return Err(errors::Error::Protocol {
                protocol: "air",
                message: "num_columns does not match the virtual column layout",
            });
        }

        let entries = air.virtual_expander().map(|e| e.expansion_entries());
        let plan = RingSwitchPlan::new(air.column_layout(), entries.as_deref(), 0, 0)?;

        Ok(Self {
            num_vars: 0,
            num_columns: air.num_columns(),
            sumcheck_degree,
            bus_columns: columns.len(),
            bus_wires,
            mul_nodes: mul_node_count(ast),
            ring_units: plan.has_ring(),
            whole_blinds: 0,
        })
    }

    /// Counts the committed `h` limb columns.
    pub fn h_columns(&self) -> usize {
        BUS_LIMBS * self.bus_columns
    }

    pub fn at(self, layout: &SlotLayout) -> Self {
        Self {
            num_vars: layout.num_vars,
            whole_blinds: layout.plan.whole_blinds(),
            ..self
        }
    }

    /// Eval claims per half: virtual columns, whole blind
    /// units, the ring blind unit's planes, every `h` limb.
    pub fn eval_claims(&self) -> usize {
        let ring_blind = match self.ring_units && self.whole_blinds > 0 {
            true => RING_BLIND_BITS,
            false => 0,
        };

        self.num_columns + self.whole_blinds + ring_blind + self.h_columns()
    }

    /// Final sumcheck values are the masked running
    /// claims themselves and consume no pad entry.
    pub fn masked_scalars(&self) -> usize {
        let zerocheck = self.num_vars * self.sumcheck_degree;
        let claims = 2 * self.eval_claims();

        2 * self.h_columns() + zerocheck + claims
    }

    /// Counts the table's Hadamard wires, AST and bus products.
    pub fn mul_wires(&self) -> usize {
        self.mul_nodes + self.bus_wires
    }
}

/// Bits of the rounds past the pool term,
/// a table's rounds at their weakest table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DrawBits {
    pub bus: usize,
    pub zerocheck: usize,
    pub sumcheck: usize,
    pub eval_batch: usize,
    pub eval_round: usize,
    pub outer: Option<usize>,
}

impl DrawBits {
    pub fn weakest(&self) -> usize {
        [
            self.bus,
            self.zerocheck,
            self.sumcheck,
            self.eval_batch,
            self.eval_round,
        ]
        .into_iter()
        .fold(self.outer.unwrap_or(usize::MAX), usize::min)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OuterStatement {
    pub masked_scalars: usize,
    pub mul_wires: usize,
}

impl OuterStatement {
    pub fn new(shapes: &[TableShape], pool_vars: usize) -> Self {
        let tables: usize = shapes.iter().map(TableShape::masked_scalars).sum();
        let wires: usize = shapes.iter().map(TableShape::mul_wires).sum();

        let gadget = match shapes.iter().any(|s| s.ring_units) {
            true => linearized::BITS - 1,
            false => 0,
        };

        Self {
            masked_scalars: tables + 2 * pool_vars,
            mul_wires: wires + gadget,
        }
    }

    /// Statement size for a whole proof.
    pub fn for_tables<F: TowerField, A: Air<F>>(
        main: &A,
        main_num_vars: usize,
        chiplets: &[ChipletDef<F>],
        chiplet_num_vars: &[usize],
        config: &Config,
    ) -> errors::Result<Self> {
        if chiplets.len() != chiplet_num_vars.len() {
            return Err(errors::Error::Protocol {
                protocol: "outer",
                message: "one height per chiplet is required",
            });
        }

        let main_parts = TableParts::of(main);

        let mut shapes = Vec::with_capacity(chiplets.len() + 1);
        let mut plans = Vec::with_capacity(chiplets.len() + 1);

        for def in chiplets {
            let shape = TableShape::from_air(def, &def.statics())?;

            plans.push(table_plan(def, &shape, config)?);
            shapes.push(shape);
        }

        let main_shape = TableShape::from_air(main, &main_parts.statics())?;

        plans.push(table_plan(main, &main_shape, config)?);
        shapes.push(main_shape);

        let heights: Vec<(&RingSwitchPlan, usize)> = plans
            .iter()
            .zip(chiplet_num_vars.iter().chain([&main_num_vars]))
            .map(|(plan, &num_vars)| (plan, num_vars))
            .collect();

        let pool = PoolLayout::new(&heights, F::BITS, config);

        let shapes: Vec<TableShape> = shapes
            .into_iter()
            .zip(&pool.tables)
            .map(|(shape, layout)| shape.at(layout))
            .collect();

        Ok(Self::new(&shapes, pool.num_vars()))
    }
}

/// Oracle row order: pad, then the `Lhs`, `Rhs` and
/// `Product` wire blocks, then the five mask rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OuterLayout {
    pub message_len: usize,
    pub pad_rows: usize,
    pub aux_rows: usize,
    pub domain_len: usize,
}

impl OuterLayout {
    pub fn new(
        geom: &OuterGeometry,
        masked_scalars: usize,
        mul_wires: usize,
    ) -> errors::Result<Self> {
        if geom.message_len == 0 {
            return Err(errors::Error::Protocol {
                protocol: "outer",
                message: "message length must be positive",
            });
        }

        Ok(Self {
            message_len: geom.message_len,
            pad_rows: masked_scalars.div_ceil(geom.message_len),
            aux_rows: mul_wires.div_ceil(geom.message_len).max(1),
            domain_len: geom.domain_len,
        })
    }

    pub fn slot(&self, unknown: Unknown) -> (usize, usize) {
        match unknown {
            Unknown::Pad(i) => {
                let i = i as usize;

                (i / self.message_len, i % self.message_len)
            }
            Unknown::Wire { mul, role } => {
                let mul = mul as usize;
                let block = role as usize;

                (
                    self.pad_rows + block * self.aux_rows + mul / self.message_len,
                    mul % self.message_len,
                )
            }
        }
    }

    pub fn interleaved_mask(&self) -> usize {
        self.pad_rows + 3 * self.aux_rows
    }

    pub fn linear_mask(&self) -> usize {
        self.interleaved_mask() + 1
    }

    pub fn linear_mask_hi(&self) -> usize {
        self.interleaved_mask() + 2
    }

    pub fn quadratic_mask(&self) -> usize {
        self.interleaved_mask() + 3
    }

    pub fn quadratic_mask_hi(&self) -> usize {
        self.interleaved_mask() + 4
    }

    pub fn total_rows(&self) -> usize {
        self.interleaved_mask() + OUTER_MASK_ROWS
    }

    pub fn hadamard_triples(&self) -> Vec<[usize; 3]> {
        (0..self.aux_rows)
            .map(|i| {
                [
                    self.pad_rows + i,
                    self.pad_rows + self.aux_rows + i,
                    self.pad_rows + 2 * self.aux_rows + i,
                ]
            })
            .collect()
    }
}

pub struct EvalRecord<F> {
    pub mask_form: Vec<(Unknown, Flat<F>)>,
    pub claim_masked: Flat<F>,
    pub fin: Flat<F>,
    pub gadget: Option<RingGadget<F>>,
}

/// One table's share of the statement, in transcript order.
pub struct TableRecord<'s, F> {
    pub claims_masked: Vec<Flat<F>>,
    pub ast: &'s ConstraintAst<F>,
    pub consistency: ConsistencyInputs<F>,

    /// `(pad, claim, public value)` per fixed column.
    pub fixed: Vec<(u32, u32, Flat<F>)>,

    /// `(group, limb, pad, masked limb)` per claimed-sum limb.
    pub claimed_sums: Vec<(BusGroup, usize, u32, Flat<F>)>,

    pub shape: TableShape,
}

impl<F: TowerField + HardwareField> TableRecord<'_, F> {
    fn affine_rows(&self) -> usize {
        2 * self.shape.mul_wires() + 1 + self.shape.h_columns() + self.fixed.len()
    }

    /// `f` on each row after the predicate rows, in statement order;
    /// the consistency row comes without its root terms.
    fn for_each_tail_row(&self, mut f: impl FnMut(AffineRow<F>)) {
        let consistency = &self.consistency;
        let pad_first = consistency.pad_first;

        f(consistency_row(self.ast.roots.len(), consistency));

        for column in &consistency.buses {
            column_terms(column, pad_first, consistency.tau, &mut f);
        }

        for &(pad, claim, public) in &self.fixed {
            f(fixed_column_row(pad, claim, public));
        }

        let half = (self.claims_masked.len() / 2) as u32;
        let first_h = half - self.shape.h_columns() as u32;

        for (c, column) in consistency.buses.iter().enumerate() {
            for j in 0..BUS_LIMBS {
                let claim = first_h + (BUS_LIMBS * c + j) as u32;

                f(h_claim_pin_row(
                    column.h_pads[j],
                    column.h_masked[j],
                    pad_first + claim,
                    claim,
                ));
            }
        }
    }
}

/// The assembled statement: affine rows, and each ring
/// gadget with the statement row its chain starts at.
pub struct OuterRows<'r, F> {
    pub affine: Vec<AffineRow<F>>,
    pub gadgets: Vec<(usize, &'r RingGadget<F>)>,
}

/// `H_0` of an eval sumcheck over the pad.
pub struct InitialForm<F> {
    pub form: Vec<(Unknown, Flat<F>)>,
    pub gadget: Option<RingGadget<F>>,
}

/// First pad index of each masked group of a table,
/// in the prover's masking order.
#[derive(Clone, Copy, Debug)]
pub struct TablePads {
    pub claimed_sums: u32,
    pub zerocheck: u32,
    pub h_evals: u32,
    pub claims: u32,
}

/// Public data of the pool's eval sumcheck on masked claims.
pub struct EvalInputs<'a, F> {
    pub pool: &'a PoolLayout,
    pub eta: F,
    pub rho: &'a [F],
    pub r_mix: &'a [Block128],
    pub shifted_claims: bool,
    pub challenges: &'a [Flat<F>],
    pub claim_masked: Flat<F>,
    pub fin: Flat<F>,
    pub claim_pads: &'a [u32],
    pub round_first: u32,
    pub first_wire: u32,
}

/// Everything a table's record is built from; both
/// sides hold all of it after the table's eval phase.
pub struct TableInputs<'a, 's, F: TowerField> {
    pub statics: TableStatics<'s, F>,
    pub shape: TableShape,
    pub instance: &'a ProgramInstance<F>,
    pub blinding_columns: usize,

    pub alpha: Flat<F>,
    pub gamma: BusElement,
    pub beta: BusElement,
    pub r_zerocheck: &'a [Flat<F>],
    pub r_final: &'a [Flat<F>],
    pub lookup_bus_points: &'a BTreeMap<String, Vec<BusElement>>,
    pub clocks: &'a [Option<RankClock>],

    pub claimed_sums_masked: &'a [F],
    pub h_evals_masked: &'a [F],
    pub val_final_masked: Flat<F>,
    pub claims_masked: Vec<Flat<F>>,

    pub pads: TablePads,
}

/// Weight rows and target of the linear
/// test, summed over scaled affine rows.
struct Accumulator<'l, F> {
    layout: &'l OuterLayout,
    weights: Vec<Vec<Flat<F>>>,
    target: Flat<F>,
}

impl<'l, F: TowerField + HardwareField> Accumulator<'l, F> {
    fn new(layout: &'l OuterLayout) -> Self {
        Self {
            layout,
            weights: vec![vec![Flat::from_raw(F::ZERO); layout.message_len]; layout.total_rows()],
            target: Flat::from_raw(F::ZERO),
        }
    }

    fn add(&mut self, unknown: Unknown, value: Flat<F>) {
        let (row, col) = self.layout.slot(unknown);
        self.weights[row][col] += value;
    }

    /// Adds `scale` times one table's `row`: claims read
    /// from `claims`, wires shifted by `mul_offset`.
    fn row(&mut self, row: &AffineRow<F>, scale: Flat<F>, claims: &[Flat<F>], mul_offset: u32) {
        for &(unknown, coeff) in &row.unknowns {
            self.add(unknown.shifted(mul_offset), scale * coeff);
        }

        let mut constant = row.constant;
        for &(idx, coeff) in &row.claims {
            constant += coeff * claims[idx as usize];
        }

        self.target += scale * constant;
    }

    /// The weight rows that hold a nonzero
    /// entry, with their oracle row indices.
    fn batch(mut self) -> LinearBatch<F> {
        let zero = Flat::from_raw(F::ZERO);
        let rows: Vec<usize> = (0..self.weights.len())
            .filter(|&r| self.weights[r].iter().any(|v| *v != zero))
            .collect();

        let weights = rows
            .iter()
            .map(|&r| mem::take(&mut self.weights[r]))
            .collect();

        LinearBatch {
            weights,
            rows,
            target: self.target,
        }
    }
}

struct Sweep<F> {
    live: Vec<bool>,
    adjoint: Vec<Flat<F>>,
}

impl<F: TowerField + HardwareField> Sweep<F> {
    fn new() -> Self {
        Self {
            live: Vec::new(),
            adjoint: Vec::new(),
        }
    }

    /// Adds the table's predicate rows at `scales` and its
    /// root terms at `consistency` in one pass from the last
    /// node to the first, each node handing down its adjoint.
    fn run(
        &mut self,
        acc: &mut Accumulator<'_, F>,
        record: &TableRecord<'_, F>,
        scales: &[Flat<F>],
        consistency: Flat<F>,
        mul_offset: u32,
    ) -> errors::Result<()> {
        let ast = record.ast;
        let claims = &record.claims_masked;

        let layout = ClaimLayout {
            pad_first: record.consistency.pad_first,
            half: (claims.len() / 2) as u32,
        };

        let n = ast.arena.len();

        self.live.clear();
        self.live.resize(n, false);
        self.adjoint.clear();
        self.adjoint.resize(n, Flat::from_raw(F::ZERO));

        let roots = record.consistency.root_weights(ast.roots.len());
        for (root, weight) in ast.roots.iter().zip(roots) {
            self.pass(*root, consistency * weight);
        }

        let mut muls = scales.len() / 2;

        // Children sit at lower indices than their parent
        for i in (0..n).rev() {
            if !self.live[i] {
                continue;
            }

            let adjoint = self.adjoint[i];

            match ast.arena.get(ExprId(i as u32)) {
                ConstraintExpr::Cell(cell) => {
                    let idx = layout.index(*cell);

                    acc.add(Unknown::Pad(layout.pad_first + idx), adjoint);

                    acc.target += adjoint * claims[idx as usize];
                }
                ConstraintExpr::Const(v) => acc.target += adjoint * v.to_hardware(),
                ConstraintExpr::Add(a, b) => {
                    self.pass(*a, adjoint);
                    self.pass(*b, adjoint);
                }
                ConstraintExpr::Scale(coeff, a) => self.pass(*a, adjoint * coeff.to_hardware()),
                ConstraintExpr::Sum(children) => {
                    for c in children {
                        self.pass(*c, adjoint);
                    }
                }
                ConstraintExpr::Mul(a, b) => {
                    muls = muls.checked_sub(1).ok_or(errors::Error::Protocol {
                        protocol: "outer",
                        message: "table shape undercounts the AST's Mul nodes",
                    })?;

                    let lhs = scales[2 * muls];
                    let rhs = scales[2 * muls + 1];
                    let mul = mul_offset + muls as u32;

                    acc.add(
                        Unknown::Wire {
                            mul,
                            role: WireRole::Lhs,
                        },
                        lhs,
                    );
                    acc.add(
                        Unknown::Wire {
                            mul,
                            role: WireRole::Rhs,
                        },
                        rhs,
                    );
                    acc.add(
                        Unknown::Wire {
                            mul,
                            role: WireRole::Product,
                        },
                        adjoint,
                    );

                    self.pass(*a, lhs);
                    self.pass(*b, rhs);
                }
            }
        }

        match muls {
            0 => Ok(()),
            _ => Err(errors::Error::Protocol {
                protocol: "outer",
                message: "table shape overcounts the AST's Mul nodes",
            }),
        }
    }

    fn pass(&mut self, child: ExprId, value: Flat<F>) {
        let c = child.0 as usize;

        self.live[c] = true;
        self.adjoint[c] += value;
    }
}

pub fn eval_initial_form<F: HardwareField + Into<Block128> + From<u128>>(
    pool: &PoolLayout,
    eta: F,
    rho: &[F],
    r_mix: &[Block128],
    shifted_claims: bool,
    claim_pads: &[u32],
    first_wire: u32,
) -> errors::Result<InitialForm<F>> {
    let mut weights = Vec::with_capacity(claim_pads.len());
    for layout in &pool.tables {
        let rho = rho
            .get(..layout.pack_vars())
            .ok_or(errors::Error::Protocol {
                protocol: "outer",
                message: "fewer block challenges than a table packs",
            })?;

        weights.extend(claim_weights(layout, eta, rho, shifted_claims)?);
    }

    if weights.len() != claim_pads.len() {
        return Err(errors::Error::Protocol {
            protocol: "outer",
            message: "one pad entry per eval claim is required",
        });
    }

    let mut form = Vec::new();
    let mut ring = Vec::new();

    for ((is_ring, weight), &pad) in weights.iter().zip(claim_pads) {
        let a: Flat<F> = F::from(weight.0).to_hardware();
        match is_ring {
            true => ring.push((pad, a)),
            false => form.push((Unknown::Pad(pad), a)),
        }
    }

    if ring.is_empty() {
        return Ok(InitialForm { form, gadget: None });
    }

    let mu: Vec<Flat<F>> = linearized_coeffs(&eq_tensor_b(r_mix))
        .iter()
        .map(|m| F::from(m.to_tower().0).to_hardware())
        .collect();

    let gadget = RingGadget::new(ring, mu, first_wire);

    form.extend(gadget.delta_form());

    Ok(InitialForm {
        form,
        gadget: Some(gadget),
    })
}

/// The eval row's mask form propagated
/// through the rounds, plus the ring gadget.
pub fn eval_record<F>(inputs: &EvalInputs<'_, F>) -> errors::Result<EvalRecord<F>>
where
    F: HardwareField + Into<Block128> + From<u128>,
{
    let initial = eval_initial_form(
        inputs.pool,
        inputs.eta,
        inputs.rho,
        inputs.r_mix,
        inputs.shifted_claims,
        inputs.claim_pads,
        inputs.first_wire,
    )?;

    let round_pads: Vec<[u32; 2]> = (0..inputs.challenges.len() as u32)
        .map(|i| [inputs.round_first + 2 * i, inputs.round_first + 2 * i + 1])
        .collect();
    let round_refs: Vec<&[u32]> = round_pads.iter().map(|p| p.as_slice()).collect();

    Ok(EvalRecord {
        mask_form: sumcheck_mask_form(&initial.form, &round_refs, inputs.challenges, 2),
        claim_masked: inputs.claim_masked,
        fin: inputs.fin,
        gadget: initial.gadget,
    })
}

pub fn table_record<'s, F>(inputs: TableInputs<'_, 's, F>) -> errors::Result<TableRecord<'s, F>>
where
    F: TowerField + HardwareField,
{
    let TableStatics {
        ast,
        specs: bus_specs,
        fixed: fixed_columns,
        boundary: boundary_constraints,
    } = inputs.statics;

    let shape = inputs.shape;
    let num_vars = shape.num_vars;
    let trace_width = shape.num_columns;
    let r_final = inputs.r_final;
    let pad_first = inputs.pads.claims;
    let half = shape.eval_claims() as u32;

    if inputs.claims_masked.len() != 2 * half as usize {
        return Err(errors::Error::Protocol {
            protocol: "outer",
            message: "eval claim count does not match the table shape",
        });
    }

    let eq_row_checker = TensorProduct::new(r_final.to_vec());

    let mut shapes = ShapeEvaluator::new(r_final);
    let mut fixed = Vec::new();

    for fc in fixed_columns {
        if fc.col_idx >= trace_width {
            return Err(errors::Error::Protocol {
                protocol: "outer",
                message: "fixed column col_idx out of trace_width",
            });
        }

        fixed.push((
            pad_first + fc.col_idx as u32,
            fc.col_idx as u32,
            shapes.evaluate(&fc.shape),
        ));
    }

    let mut boundary = Vec::with_capacity(boundary_constraints.len());
    for bc in boundary_constraints {
        if bc.col_idx >= trace_width || bc.row_idx >= 1 << num_vars {
            return Err(errors::Error::Protocol {
                protocol: "outer",
                message: "boundary constraint out of the table",
            });
        }

        boundary.push((
            bc.col_idx as u32,
            bc.resolve_target(inputs.instance)?.to_hardware(),
            eq_row_checker.evaluate_at_index(bc.row_idx),
        ));
    }

    let h_columns = shape.h_columns();

    if inputs.claimed_sums_masked.len() != h_columns || inputs.h_evals_masked.len() != h_columns {
        return Err(errors::Error::Protocol {
            protocol: "outer",
            message: "LogUp limb count does not match the table shape",
        });
    }

    let columns = bus_columns(bus_specs, shape.sumcheck_degree);

    let mut buses = Vec::with_capacity(columns.len());
    let mut claimed_sums = Vec::with_capacity(h_columns);

    if !columns.is_empty() {
        let field = BusField::<F>::new()?;
        let challenges = [field.import(inputs.gamma), field.import(inputs.beta)];

        let fraction = |idx: usize| {
            BusFraction::new(
                &field,
                idx,
                &bus_specs[idx],
                trace_width,
                challenges,
                r_final,
                inputs.clocks,
            )
        };

        let mut first_wire = shape.mul_nodes as u32;
        for (c, column) in columns.iter().enumerate() {
            let group = column.group(bus_specs);
            let eq_lookup = group.row_weight(&field, inputs.lookup_bus_points, r_final)?;

            let limb = |j: usize| BUS_LIMBS * c + j;

            for j in 0..BUS_LIMBS {
                claimed_sums.push((
                    group.clone(),
                    j,
                    inputs.pads.claimed_sums + limb(j) as u32,
                    inputs.claimed_sums_masked[limb(j)].to_hardware(),
                ));
            }

            buses.push(BusColumnRow {
                h_pads: core::array::from_fn(|j| inputs.pads.h_evals + limb(j) as u32),
                h_masked: core::array::from_fn(|j| inputs.h_evals_masked[limb(j)].to_hardware()),
                first: fraction(column.first)?,
                second: column.second.map(fraction).transpose()?,
                eq_lookup,
                first_wire,
            });

            first_wire += column_wires(column, bus_specs) as u32;
        }
    }

    let logup_alpha_offset = ast.roots.len() + boundary_constraints.len() + inputs.blinding_columns;

    let mut alpha_pow = Flat::from_raw(F::ONE);
    for _ in 0..logup_alpha_offset {
        alpha_pow *= inputs.alpha;
    }

    let initial: Vec<(Unknown, Flat<F>)> = (0..h_columns as u32)
        .zip(bus_alpha_weights(alpha_pow, inputs.alpha))
        .map(|(k, w)| (Unknown::Pad(inputs.pads.claimed_sums + k), w.bus_sum))
        .collect();

    let degree = shape.sumcheck_degree;
    let round_pads: Vec<Vec<u32>> = (0..num_vars)
        .map(|i| {
            let first = inputs.pads.zerocheck + (i * degree) as u32;

            (0..degree as u32).map(|j| first + j).collect()
        })
        .collect();

    let round_refs: Vec<&[u32]> = round_pads.iter().map(|p| p.as_slice()).collect();

    let consistency = ConsistencyInputs {
        pad_first,
        eq_zc: TensorProduct::evaluate_eq_slice(inputs.r_zerocheck, r_final),
        alpha: inputs.alpha,
        tau: F::EXTENSION_TAU.to_hardware(),
        boundary,
        telescope: (0..inputs.blinding_columns as u32)
            .map(|k| {
                let b = trace_width as u32 + k;

                (b, half + b)
            })
            .collect(),
        buses,
        val_final_form: sumcheck_mask_form(&initial, &round_refs, r_final, degree),
        val_final_masked: inputs.val_final_masked,
    };

    Ok(TableRecord {
        claims_masked: inputs.claims_masked,
        ast,
        consistency,
        fixed,
        claimed_sums,
        shape,
    })
}

/// Pushes an `h` column's honest `(lhs, rhs, product)` wires,
/// in wire order, from the true claims and `h` limbs.
pub fn bus_wire_values<F: TowerField + HardwareField>(
    field: &BusField<F>,
    first: &BusFraction<F>,
    second: Option<&BusFraction<F>>,
    claims: &[Flat<F>],
    h: BusValue<F>,
    out: &mut Vec<[Flat<F>; 3]>,
) {
    let d1 = first.denominator(claims);

    karatsuba_values(h, d1, out);

    let Some(second) = second else {
        return;
    };

    let u = field.mul(h, d1) + BusValue::from_base(first.selector_value(claims));

    karatsuba_values(u, second.denominator(claims), out);

    if second.selector_claimed() {
        let s2 = second.selector_value(claims);

        for d in d1.limbs() {
            out.push([s2, d, s2 * d]);
        }
    }
}

/// Every affine row of the statement, claims folded
/// into constants and wire indices offset per table.
pub fn assemble<'r, F: TowerField + HardwareField>(
    records: &'r [TableRecord<'_, F>],
    eval: &'r EvalRecord<F>,
    statement: &OuterStatement,
) -> errors::Result<OuterRows<'r, F>> {
    let mut affine = Vec::new();
    let mut gadgets = Vec::new();
    let mut offset = 0u32;

    for record in records {
        let first = affine.len();

        let layout = ClaimLayout {
            pad_first: record.consistency.pad_first,
            half: (record.claims_masked.len() / 2) as u32,
        };

        let predicate = compile(record.ast, layout);
        let consistency = first + predicate.affine.len();

        affine.extend(predicate.affine);

        record.for_each_tail_row(|row| affine.push(row));

        let weights = record.consistency.root_weights(predicate.roots.len());
        for (form, weight) in predicate.roots.iter().zip(weights) {
            scale_into(&mut affine[consistency], form, weight);
        }

        for row in &mut affine[first..] {
            resolve_table_row(row, &record.claims_masked, offset);
        }

        offset += record.shape.mul_wires() as u32;
    }

    affine.push(eval_final_row(&eval.mask_form, eval.claim_masked, eval.fin));

    if let Some(gadget) = &eval.gadget {
        gadgets.push((affine.len(), gadget));
        affine.extend(gadget.wire_rows());
    }

    if offset as usize + eval_wires(eval) != statement.mul_wires {
        return Err(errors::Error::Protocol {
            protocol: "outer",
            message: "table wire counts do not sum to the statement",
        });
    }

    for endpoints in bus_endpoints(records).values() {
        affine.push(bus_sum_row(endpoints));
    }

    Ok(OuterRows { affine, gadgets })
}

/// Affine rows of the statement [`assemble`] builds:
/// every table's, the pool's eval rows,
/// one bus-sum row per bus group and limb.
pub fn statement_rows<F: TowerField + HardwareField>(
    records: &[TableRecord<'_, F>],
    eval: &EvalRecord<F>,
    statement: &OuterStatement,
) -> errors::Result<usize> {
    let tables_wires: usize = records.iter().map(|r| r.shape.mul_wires()).sum();

    if tables_wires + eval_wires(eval) != statement.mul_wires {
        return Err(errors::Error::Protocol {
            protocol: "outer",
            message: "table wire counts do not sum to the statement",
        });
    }

    let tables: usize = records.iter().map(TableRecord::affine_rows).sum();

    Ok(tables + eval_rows(eval) + bus_endpoints(records).len())
}

pub fn reachable<F: TowerField>(ast: &ConstraintAst<F>) -> Vec<bool> {
    let n = ast.arena.len();
    let mut live = vec![false; n];

    for root in &ast.roots {
        live[root.0 as usize] = true;
    }

    // Children sit at lower indices than their parent
    for i in (0..n).rev() {
        if !live[i] {
            continue;
        }

        let mut mark = |id: &ExprId| live[id.0 as usize] = true;
        match ast.arena.get(ExprId(i as u32)) {
            ConstraintExpr::Cell(_) | ConstraintExpr::Const(_) => {}
            ConstraintExpr::Add(a, b) | ConstraintExpr::Mul(a, b) => {
                mark(a);
                mark(b);
            }
            ConstraintExpr::Scale(_, a) => mark(a),
            ConstraintExpr::Sum(children) => children.iter().for_each(mark),
        }
    }

    live
}

pub fn mul_node_count<F: TowerField>(ast: &ConstraintAst<F>) -> usize {
    reachable(ast)
        .iter()
        .enumerate()
        .filter(|&(i, &live)| {
            live && matches!(ast.arena.get(ExprId(i as u32)), ConstraintExpr::Mul(..))
        })
        .count()
}

/// The consistency row without its root terms:
/// root `k` enters at `α^k · eq_zc`.
pub fn consistency_row<F: TowerField + HardwareField>(
    roots: usize,
    inputs: &ConsistencyInputs<F>,
) -> AffineRow<F> {
    let mut row = AffineRow {
        unknowns: inputs.val_final_form.to_vec(),
        claims: Vec::new(),
        constant: inputs.val_final_masked,
    };

    let mut alpha_pow = Flat::from_raw(F::ONE);

    for _ in 0..roots {
        alpha_pow *= inputs.alpha;
    }

    for &(claim, public, eq_row) in &inputs.boundary {
        let weight = eq_row * alpha_pow;

        push_claim(&mut row, inputs.pad_first, claim, weight);
        row.constant += public * weight;

        alpha_pow *= inputs.alpha;
    }

    for &(b, b_next) in &inputs.telescope {
        push_claim(&mut row, inputs.pad_first, b, alpha_pow);
        push_claim(&mut row, inputs.pad_first, b_next, alpha_pow);

        alpha_pow *= inputs.alpha;
    }

    let mut weights = bus_alpha_weights(alpha_pow, inputs.alpha);

    for column in &inputs.buses {
        let consistency = column_terms(column, inputs.pad_first, inputs.tau, &mut |_| {});
        let bus_sum = bus_sum_forms(column, inputs.tau);

        for ((c, s), w) in consistency.iter().zip(&bus_sum).zip(weights.by_ref()) {
            add_form(&mut row, &c.scaled(w.consistency * inputs.eq_zc));
            add_form(&mut row, &s.scaled(w.bus_sum));
        }
    }

    row
}

/// Pad form of `claim'_n - claim_n` for a sumcheck whose
/// replay reconstructs `p'_0 = claim' - p'_1` each round:
/// entries of round `i` carry `Π_{j>i} L_0(r_j)`.
pub fn sumcheck_mask_form<F: TowerField + HardwareField>(
    initial: &[(Unknown, Flat<F>)],
    round_pads: &[&[u32]],
    challenges: &[Flat<F>],
    degree: usize,
) -> Vec<(Unknown, Flat<F>)> {
    let rounds: Vec<Vec<Flat<F>>> = challenges
        .iter()
        .map(|&r| lagrange_weights::<F>(degree, r))
        .collect();

    let mut suffix = vec![Flat::from_raw(F::ONE); rounds.len() + 1];
    for i in (0..rounds.len()).rev() {
        suffix[i] = suffix[i + 1] * rounds[i][0];
    }

    let mut form: Vec<(Unknown, Flat<F>)> = initial
        .iter()
        .map(|&(u, coeff)| (u, coeff * suffix[0]))
        .collect();

    for (i, (pads, weights)) in round_pads.iter().zip(&rounds).enumerate() {
        let after = suffix[i + 1];

        form.push((Unknown::Pad(pads[0]), weights[0] * after));

        for (j, &pad) in pads.iter().enumerate() {
            form.push((Unknown::Pad(pad), weights[j + 1] * after));
        }
    }

    form
}

/// `claim_n = fin` for an eval sumcheck: `fin` is public,
/// `claim_n` is the masked running claim minus its form.
pub fn eval_final_row<F: TowerField + HardwareField>(
    mask_form: &[(Unknown, Flat<F>)],
    claim_masked: Flat<F>,
    fin: Flat<F>,
) -> AffineRow<F> {
    AffineRow {
        unknowns: mask_form.to_vec(),
        claims: Vec::new(),
        constant: claim_masked + fin,
    }
}

/// `c[claim] = public`, i.e. `h[pad] = c'[claim] + public`.
pub fn fixed_column_row<F: TowerField + HardwareField>(
    pad: u32,
    claim: u32,
    public: Flat<F>,
) -> AffineRow<F> {
    let one = Flat::from_raw(F::ONE);

    AffineRow {
        unknowns: vec![(Unknown::Pad(pad), one)],
        claims: vec![(claim, one)],
        constant: public,
    }
}

/// `c[claim] = h_eval`: the eval argument's base `h`
/// claim and the reported `h_eval` unmask to one value.
pub fn h_claim_pin_row<F: TowerField + HardwareField>(
    h_pad: u32,
    h_masked: Flat<F>,
    claim_pad: u32,
    claim: u32,
) -> AffineRow<F> {
    let one = Flat::from_raw(F::ONE);

    AffineRow {
        unknowns: vec![(Unknown::Pad(h_pad), one), (Unknown::Pad(claim_pad), one)],
        claims: vec![(claim, one)],
        constant: h_masked,
    }
}

/// `Σ claimed_sum = 0` over one bus group and limb:
/// `(pad, masked sum)` per column.
pub fn bus_sum_row<F: TowerField + HardwareField>(endpoints: &[(u32, Flat<F>)]) -> AffineRow<F> {
    let mut constant = Flat::from_raw(F::ZERO);
    for &(_, masked) in endpoints {
        constant += masked;
    }

    AffineRow {
        unknowns: endpoints
            .iter()
            .map(|&(pad, _)| (Unknown::Pad(pad), Flat::from_raw(F::ONE)))
            .collect(),
        claims: Vec::new(),
        constant,
    }
}

/// Folds a table's public masked claims into the constant
/// and shifts its table-local wire indices by `mul_offset`.
pub fn resolve_table_row<F: TowerField + HardwareField>(
    row: &mut AffineRow<F>,
    claims: &[Flat<F>],
    mul_offset: u32,
) {
    for &(idx, coeff) in &row.claims {
        row.constant += coeff * claims[idx as usize];
    }

    row.claims.clear();

    for (unknown, _) in row.unknowns.iter_mut() {
        *unknown = unknown.shifted(mul_offset);
    }
}

/// PAD oracle rows: the pad in [`OuterLayout::slot`] order,
/// `filler` (exactly `pad_rows · (code_len - message_len)`
/// entries) on the message tail.
pub fn build_pad_rows<F: TowerField + HardwareField>(
    layout: &OuterLayout,
    pad: &[Flat<F>],
    code_len: usize,
    filler: &[Flat<F>],
) -> errors::Result<Vec<Vec<Flat<F>>>> {
    let tail = code_len - layout.message_len;

    if pad.len() > layout.pad_rows * layout.message_len || filler.len() != layout.pad_rows * tail {
        return Err(errors::Error::Protocol {
            protocol: "outer",
            message: "pad rows do not hold the pad and its filler",
        });
    }

    let mut rows = vec![vec![Flat::from_raw(F::ZERO); layout.domain_len]; layout.pad_rows];
    for (i, value) in pad.iter().enumerate() {
        let (r, c) = layout.slot(Unknown::Pad(i as u32));
        rows[r][c] = *value;
    }

    for (row, fill) in rows.iter_mut().zip(filler.chunks_exact(tail)) {
        row[layout.message_len..code_len].copy_from_slice(fill);
    }

    Ok(rows)
}

/// AUX oracle rows: the wire blocks and the five mask
/// rows, `filler` on every message tail, then the
/// zero-sum mask row's first `code_len - 1` cells and
/// the messages of the interleaved and both high masks.
pub fn build_aux_rows<F: TowerField + HardwareField>(
    layout: &OuterLayout,
    wires: &[[Flat<F>; 3]],
    code_len: usize,
    filler: &[Flat<F>],
) -> errors::Result<Vec<Vec<Flat<F>>>> {
    let count = layout.total_rows() - layout.pad_rows;
    let tail = code_len - layout.message_len;

    if wires.len() > layout.aux_rows * layout.message_len
        || filler.len() != aux_filler_len(layout, code_len)
    {
        return Err(errors::Error::Protocol {
            protocol: "outer",
            message: "aux rows do not hold the wires and their filler",
        });
    }

    let mut rows = vec![vec![Flat::from_raw(F::ZERO); layout.domain_len]; count];
    for (mul, wire) in wires.iter().enumerate() {
        for role in [WireRole::Lhs, WireRole::Rhs, WireRole::Product] {
            let (r, c) = layout.slot(Unknown::Wire {
                mul: mul as u32,
                role,
            });

            rows[r - layout.pad_rows][c] = wire[role as usize];
        }
    }

    let (tails, rest) = filler.split_at(count * tail);
    for (row, fill) in rows.iter_mut().zip(tails.chunks_exact(tail)) {
        row[layout.message_len..code_len].copy_from_slice(fill);
    }

    let (zero_sum_fill, rest) = rest.split_at(code_len - 1);

    let zero_sum = layout.linear_mask() - layout.pad_rows;

    let mut acc = Flat::from_raw(F::ZERO);
    for (slot, &v) in rows[zero_sum][..code_len - 1].iter_mut().zip(zero_sum_fill) {
        *slot = v;
        acc += v;
    }

    rows[zero_sum][code_len - 1] = acc;

    // The quadratic mask keeps its zero message; every
    // other mask row is uniform over the whole message.
    let uniform = [
        layout.interleaved_mask(),
        layout.linear_mask_hi(),
        layout.quadratic_mask_hi(),
    ];

    for (row, fill) in uniform
        .into_iter()
        .zip(rest.chunks_exact(layout.message_len))
    {
        rows[row - layout.pad_rows][..layout.message_len].copy_from_slice(fill);
    }

    Ok(rows)
}

pub fn aux_filler_len(layout: &OuterLayout, code_len: usize) -> usize {
    let count = layout.total_rows() - layout.pad_rows;

    count * (code_len - layout.message_len) + code_len - 1 + 3 * layout.message_len
}

/// Challenge count for `outer_r_lin`. The tensor
/// batches the affine rows at `k / |F|`, not the
/// `1 / |F|` of one challenge per row.
pub fn linear_tensor_vars(rows: usize) -> usize {
    rows.next_power_of_two().ilog2() as usize
}

/// Weights and target of the linear test on
/// the rows [`assemble`] builds, computed
/// from the records without building them.
pub fn linear_weights<F: TowerField + HardwareField + Into<Block128>>(
    layout: &OuterLayout,
    statement: &OuterStatement,
    records: &[TableRecord<'_, F>],
    eval: &EvalRecord<F>,
    tensor: &[F],
) -> errors::Result<LinearBatch<F>> {
    let rows = statement_rows(records, eval, statement)?;

    if tensor.len() != linear_tensor_vars(rows) {
        return Err(errors::Error::Protocol {
            protocol: "outer",
            message: "linear batch takes ceil(log2(affine rows)) challenges",
        });
    }

    weigh_records(layout, records, eval, &expand_batch_tensor(tensor, rows))
}

/// Weights and target of the linear test on built rows:
/// the reference [`linear_weights`] is tested against.
pub fn linear_weights_of_rows<F: TowerField + HardwareField + Into<Block128>>(
    layout: &OuterLayout,
    rows: &OuterRows<'_, F>,
    tensor: &[F],
) -> errors::Result<LinearBatch<F>> {
    if tensor.len() != linear_tensor_vars(rows.affine.len()) {
        return Err(errors::Error::Protocol {
            protocol: "outer",
            message: "linear batch takes ceil(log2(affine rows)) challenges",
        });
    }

    if rows.affine.iter().any(|row| !row.claims.is_empty()) {
        return Err(errors::Error::Protocol {
            protocol: "outer",
            message: "affine rows must be resolved before weighting",
        });
    }

    Ok(weigh_rows(
        layout,
        rows,
        &expand_batch_tensor(tensor, rows.affine.len()),
    ))
}

/// Builds a table's ring-switch plan with the `h` limb columns of `shape`.
pub fn table_plan<F: TowerField>(
    air: &impl Air<F>,
    shape: &TableShape,
    config: &Config,
) -> errors::Result<RingSwitchPlan> {
    let entries = air.virtual_expander().map(|e| e.expansion_entries());

    RingSwitchPlan::new(
        air.column_layout(),
        entries.as_deref(),
        config.blind_units(),
        shape.h_columns(),
    )
}

/// Gates every draw past the pool term against `config`: the bus
/// draw, each table's `α`/`r_zerocheck` draw and sumcheck rounds,
/// the eval's batch draw and rounds, the outer argument in ZK mode.
pub fn check_draws<'s, F, I>(
    config: &Config,
    pool: &PoolLayout,
    tables: I,
) -> errors::Result<DrawBits>
where
    F: TowerField + 's,
    I: IntoIterator<Item = (TableStatics<'s, F>, TableShape)>,
    I::IntoIter: Clone,
{
    let tables = tables.into_iter();

    let bus_events = bus_draw_events(
        tables
            .clone()
            .map(|(statics, shape)| (statics.specs, 1u64 << shape.num_vars)),
    );

    let shapes: Vec<TableShape> = tables.clone().map(|(_, shape)| shape).collect();

    let mut zerocheck_events: u128 = 0;
    let mut degree: u128 = 0;

    for (statics, shape) in tables {
        let terms = statics.ast.roots.len()
            + statics.boundary.len()
            + config.blind_units()
            + 2 * shape.h_columns();

        zerocheck_events = zerocheck_events.max((terms + shape.num_vars) as u128);
        degree = degree.max(shape.sumcheck_degree as u128);
    }

    let mix = match pool.tables.iter().any(|t| t.plan.has_ring()) {
        true => F::BITS.ilog2() as usize,
        false => 0,
    };

    // The eval batch identity's degree: `2Σ - 1` in η over both
    // claim halves, multilinear in ρ and, on a ring, in r_mix.
    let batch_events = ((2 * pool.num_sigma()).saturating_sub(1) + pool.rho_vars() + mix) as u128;
    let round_events = EVAL_SUMCHECK_DEGREE as u128;

    config.check_draw("bus draw", BusElement::BITS, bus_events)?;
    config.check_draw("zerocheck draw", F::BITS, zerocheck_events)?;
    config.check_draw("sumcheck round", F::BITS, degree)?;
    config.check_draw("eval batch draw", F::BITS, batch_events)?;
    config.check_draw("eval sumcheck round", F::BITS, round_events)?;

    let outer = match config.zero_knowledge {
        true => {
            let statement = OuterStatement::new(&shapes, pool.num_vars());
            let geom = config.outer_geom(statement.masked_scalars, statement.mul_wires, F::BITS)?;

            Some(
                config
                    .outer_security_bits(F::BITS, &geom)
                    .min(config.outer_field_term_bits(F::BITS, &geom)),
            )
        }
        false => None,
    };

    Ok(DrawBits {
        bus: config.draw_bits(BusElement::BITS, bus_events),
        zerocheck: config.draw_bits(F::BITS, zerocheck_events),
        sumcheck: config.draw_bits(F::BITS, degree),
        eval_batch: config.draw_bits(F::BITS, batch_events),
        eval_round: config.draw_bits(F::BITS, round_events),
        outer,
    })
}

fn push_claim<F: TowerField>(row: &mut AffineRow<F>, pad_first: u32, idx: u32, coeff: Flat<F>) {
    row.unknowns.push((Unknown::Pad(pad_first + idx), coeff));
    row.claims.push((idx, coeff));
}

fn scale_into<F: TowerField + HardwareField>(dst: &mut AffineRow<F>, src: &Form<F>, c: Flat<F>) {
    for &(u, k) in &src.unknowns {
        dst.unknowns.push((u, k * c));
    }

    for &(i, k) in &src.claims {
        dst.claims.push((i, k * c));
    }

    dst.constant += src.constant * c;
}

fn expand_batch_tensor<F: TowerField + HardwareField>(tensor: &[F], len: usize) -> Vec<Flat<F>> {
    let mut scales = vec![Flat::from_raw(F::ZERO); len];

    if len == 0 {
        return scales;
    }

    scales[0] = Flat::from_raw(F::ONE);

    for (j, &r) in tensor.iter().enumerate() {
        let half = 1usize << j;
        let scale = r.to_hardware();

        for i in half..len.min(half << 1) {
            scales[i] = scales[i - half] * scale;
        }
    }

    scales
}

fn lagrange_weights<F: TowerField + HardwareField>(degree: usize, r: Flat<F>) -> Vec<Flat<F>> {
    (0..=degree)
        .map(|j| {
            let mut evals = vec![F::ZERO; degree + 1];
            evals[j] = F::ONE;

            UnivariatePoly::new(evals).evaluate_hw(r)
        })
        .collect()
}

fn weigh_records<F: TowerField + HardwareField + Into<Block128>>(
    layout: &OuterLayout,
    records: &[TableRecord<'_, F>],
    eval: &EvalRecord<F>,
    scales: &[Flat<F>],
) -> errors::Result<LinearBatch<F>> {
    let buses = bus_endpoints(records);
    let tables: usize = records.iter().map(TableRecord::affine_rows).sum();

    if scales.len() != tables + eval_rows(eval) + buses.len() {
        return Err(errors::Error::Protocol {
            protocol: "outer",
            message: "scale count does not match the statement rows",
        });
    }

    let mut acc = Accumulator::new(layout);
    let mut sweep = Sweep::new();

    let mut first = 0;
    let mut mul_offset = 0u32;

    for record in records {
        let end = first + record.affine_rows();

        weigh_table(
            &mut acc,
            &mut sweep,
            record,
            &scales[first..end],
            mul_offset,
        )?;

        first = end;
        mul_offset += record.shape.mul_wires() as u32;
    }

    let final_row = eval_final_row(&eval.mask_form, eval.claim_masked, eval.fin);
    acc.row(&final_row, scales[first], &[], 0);

    first += 1;

    if let Some(gadget) = &eval.gadget {
        let chain = &scales[first..first + linearized::CHAIN_ROWS];
        for (row, &scale) in gadget.wire_rows().iter().zip(chain) {
            acc.row(row, scale, &[], 0);
        }

        gadget.for_each_tie_weight(chain, |pad, weight| acc.add(Unknown::Pad(pad), weight));

        first += linearized::CHAIN_ROWS;
    }

    for (endpoints, &scale) in buses.values().zip(&scales[first..]) {
        acc.row(&bus_sum_row(endpoints), scale, &[], 0);
    }

    Ok(acc.batch())
}

fn weigh_table<F: TowerField + HardwareField + Into<Block128>>(
    acc: &mut Accumulator<'_, F>,
    sweep: &mut Sweep<F>,
    record: &TableRecord<'_, F>,
    scales: &[Flat<F>],
    mul_offset: u32,
) -> errors::Result<()> {
    let (predicate, tail) = scales.split_at(2 * record.shape.mul_nodes);

    sweep.run(acc, record, predicate, tail[0], mul_offset)?;

    let mut next = 0;
    record.for_each_tail_row(|row| {
        acc.row(&row, tail[next], &record.claims_masked, mul_offset);

        next += 1;
    });

    Ok(())
}

fn weigh_rows<F: TowerField + HardwareField + Into<Block128>>(
    layout: &OuterLayout,
    rows: &OuterRows<'_, F>,
    scales: &[Flat<F>],
) -> LinearBatch<F> {
    let mut acc = Accumulator::new(layout);
    for (row, &scale) in rows.affine.iter().zip(scales) {
        acc.row(row, scale, &[], 0);
    }

    for &(first, gadget) in &rows.gadgets {
        let chain = &scales[first..first + linearized::CHAIN_ROWS];
        gadget.for_each_tie_weight(chain, |pad, weight| acc.add(Unknown::Pad(pad), weight));
    }

    acc.batch()
}

fn bus_endpoints<'r, F: TowerField>(records: &'r [TableRecord<'_, F>]) -> BusSums<'r, F> {
    let mut buses = BusSums::new();
    for record in records {
        for (group, limb, pad, masked) in &record.claimed_sums {
            buses
                .entry((group, *limb))
                .or_default()
                .push((*pad, *masked));
        }
    }

    buses
}

fn eval_rows<F>(eval: &EvalRecord<F>) -> usize {
    match eval.gadget {
        Some(_) => 1 + linearized::CHAIN_ROWS,
        None => 1,
    }
}

fn eval_wires<F>(eval: &EvalRecord<F>) -> usize {
    match eval.gadget {
        Some(_) => linearized::BITS - 1,
        None => 0,
    }
}

/// Counts the Hadamard wires one bus column adds to the outer statement.
fn column_wires(column: &BusColumn, specs: &[(String, PermutationCheckSpec)]) -> usize {
    let Some(second) = column.second else {
        return KARATSUBA_WIRES;
    };

    let spec = &specs[second].1;
    let selector_wires = match spec.selector.is_some() || spec.recv_selector.is_some() {
        true => BUS_LIMBS,
        false => 0,
    };

    2 * KARATSUBA_WIRES + selector_wires
}

/// Emits a column's operand rows into `f` and
/// returns its consistency `C` as limb forms:
/// `g·D + s`, or `(g·D₁ + s₁)·D₂ + s₂·D₁` for a pair.
fn column_terms<F: TowerField + HardwareField>(
    column: &BusColumnRow<F>,
    pad_first: u32,
    tau: Flat<F>,
    f: &mut impl FnMut(AffineRow<F>),
) -> [Form<F>; BUS_LIMBS] {
    let one = Flat::from_raw(F::ONE);
    let g = h_forms(column);
    let d1 = denominator_forms(&column.first, pad_first);
    let wire = column.first_wire;

    karatsuba_rows(&g, &d1, wire, f);

    let [mut lo, hi] = product_forms(wire, tau);
    lo.extend(&selector_form(&column.first, pad_first));

    let Some(second) = &column.second else {
        return [lo, hi];
    };

    let d2 = denominator_forms(second, pad_first);
    let next = wire + KARATSUBA_WIRES as u32;

    karatsuba_rows(&[lo, hi], &d2, next, f);

    let [mut lo, mut hi] = product_forms(next, tau);

    match second.selector_claimed() {
        true => {
            let s2 = selector_form(second, pad_first);
            let first = next + KARATSUBA_WIRES as u32;

            for (j, d) in d1.iter().enumerate() {
                let mul = first + j as u32;

                f(operand_row(&s2, mul, WireRole::Lhs));
                f(operand_row(d, mul, WireRole::Rhs));
            }

            lo.unknowns.push((product_wire(first), one));
            hi.unknowns.push((product_wire(first + 1), one));
        }
        false => {
            lo.extend(&d1[0]);
            hi.extend(&d1[1]);
        }
    }

    [lo, hi]
}

/// Returns a column's bus-sum `S = eq_lookup · g` as limb forms.
fn bus_sum_forms<F: TowerField + HardwareField>(
    column: &BusColumnRow<F>,
    tau: Flat<F>,
) -> [Form<F>; BUS_LIMBS] {
    let [g_lo, g_hi] = h_forms(column);
    let e = column.eq_lookup;

    let mut lo = g_lo.scaled(e.lo);
    lo.extend(&g_hi.scaled(tau * e.hi));

    let mut hi = g_hi.scaled(e.lo + e.hi);
    hi.extend(&g_lo.scaled(e.hi));

    [lo, hi]
}

fn h_forms<F: TowerField + HardwareField>(column: &BusColumnRow<F>) -> [Form<F>; BUS_LIMBS] {
    core::array::from_fn(|j| Form {
        unknowns: vec![(Unknown::Pad(column.h_pads[j]), Flat::from_raw(F::ONE))],
        claims: Vec::new(),
        constant: column.h_masked[j],
    })
}

fn denominator_forms<F: TowerField + HardwareField>(
    fraction: &BusFraction<F>,
    pad_first: u32,
) -> [Form<F>; BUS_LIMBS] {
    let constant = fraction.constant.limbs();

    core::array::from_fn(|j| {
        let mut form = Form {
            unknowns: Vec::with_capacity(fraction.slots.len()),
            claims: Vec::with_capacity(fraction.slots.len()),
            constant: constant[j],
        };

        for &(claim, coeff) in &fraction.slots {
            let c = coeff.limbs()[j];

            form.unknowns.push((Unknown::Pad(pad_first + claim), c));
            form.claims.push((claim, c));
        }

        form
    })
}

fn selector_form<F: TowerField + HardwareField>(
    fraction: &BusFraction<F>,
    pad_first: u32,
) -> Form<F> {
    let one = Flat::from_raw(F::ONE);
    let mut form = Form::default();

    match fraction.selector {
        Some(s) => {
            form.unknowns.push((Unknown::Pad(pad_first + s), one));
            form.claims.push((s, one));
        }
        None => form.constant = one,
    }

    if let Some(s) = fraction.recv_selector {
        form.unknowns.push((Unknown::Pad(pad_first + s), one));
        form.claims.push((s, one));
    }

    form
}

fn product_forms<F: TowerField + HardwareField>(wire: u32, tau: Flat<F>) -> [Form<F>; BUS_LIMBS] {
    let one = Flat::from_raw(F::ONE);
    let zero = Flat::from_raw(F::ZERO);

    [
        Form {
            unknowns: vec![(product_wire(wire), one), (product_wire(wire + 1), tau)],
            claims: Vec::new(),
            constant: zero,
        },
        Form {
            unknowns: vec![(product_wire(wire), one), (product_wire(wire + 2), one)],
            claims: Vec::new(),
            constant: zero,
        },
    ]
}

fn karatsuba_rows<F: TowerField + HardwareField>(
    a: &[Form<F>; BUS_LIMBS],
    b: &[Form<F>; BUS_LIMBS],
    wire: u32,
    f: &mut impl FnMut(AffineRow<F>),
) {
    let mut a_sum = a[0].clone();
    a_sum.extend(&a[1]);

    let mut b_sum = b[0].clone();
    b_sum.extend(&b[1]);

    let operands = [(&a[0], &b[0]), (&a[1], &b[1]), (&a_sum, &b_sum)];

    for (k, (lhs, rhs)) in operands.into_iter().enumerate() {
        let mul = wire + k as u32;

        f(operand_row(lhs, mul, WireRole::Lhs));
        f(operand_row(rhs, mul, WireRole::Rhs));
    }
}

fn karatsuba_values<F: TowerField + HardwareField>(
    a: BusValue<F>,
    b: BusValue<F>,
    out: &mut Vec<[Flat<F>; 3]>,
) {
    let a_sum = a.lo + a.hi;
    let b_sum = b.lo + b.hi;

    out.push([a.lo, b.lo, a.lo * b.lo]);
    out.push([a.hi, b.hi, a.hi * b.hi]);
    out.push([a_sum, b_sum, a_sum * b_sum]);
}

fn operand_row<F: TowerField + HardwareField>(
    form: &Form<F>,
    mul: u32,
    role: WireRole,
) -> AffineRow<F> {
    let mut unknowns = form.unknowns.clone();
    unknowns.push((Unknown::Wire { mul, role }, Flat::from_raw(F::ONE)));

    AffineRow {
        unknowns,
        claims: form.claims.clone(),
        constant: form.constant,
    }
}

fn product_wire(mul: u32) -> Unknown {
    Unknown::Wire {
        mul,
        role: WireRole::Product,
    }
}

fn add_form<F: TowerField + HardwareField>(row: &mut AffineRow<F>, form: &Form<F>) {
    row.unknowns.extend_from_slice(&form.unknowns);
    row.claims.extend_from_slice(&form.claims);

    row.constant += form.constant;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProgramCell;
    use crate::constraint::ConstraintArena;
    use crate::permutation::Source;
    use core::ops::Range;
    use hekate_core::config;
    use hekate_core::trace::ColumnType;
    use hekate_math::Block128;

    type F = Block128;

    const BUS_PAD_FIRST: u32 = 10;
    const BUS_H_PAD: u32 = 20;
    const BUS_PUBLIC: u128 = 77;

    /// `half - h_columns` at `bus_record`'s `half = 4` and one bus column.
    const H_CLAIM: usize = 2;

    fn shape(num_vars: usize, num_columns: usize, degree: usize, buses: usize) -> TableShape {
        TableShape {
            num_vars,
            num_columns,
            sumcheck_degree: degree,
            bus_columns: buses,
            bus_wires: KARATSUBA_WIRES * buses,
            mul_nodes: 0,
            ring_units: false,
            whole_blinds: 0,
        }
    }

    fn field() -> BusField<F> {
        BusField::new().unwrap()
    }

    fn value(seed: u128) -> BusValue<F> {
        BusValue {
            lo: mix(seed),
            hi: mix(seed + 1000),
        }
    }

    fn bus_fraction(seed: u128, selector: Option<u32>) -> BusFraction<F> {
        BusFraction {
            slots: vec![(0, value(seed))],
            constant: value(seed + BUS_PUBLIC),
            selector,
            recv_selector: None,
        }
    }

    fn masked(plain: &[Flat<F>], pad: &[Flat<F>]) -> Vec<Flat<F>> {
        plain
            .iter()
            .enumerate()
            .map(|(i, &c)| c + pad[BUS_PAD_FIRST as usize + i])
            .collect()
    }

    fn holds(
        row: &AffineRow<F>,
        claims: &[Flat<F>],
        pad: &[Flat<F>],
        wires: &[[Flat<F>; 3]],
    ) -> bool {
        let form = Form {
            unknowns: row.unknowns.clone(),
            claims: row.claims.clone(),
            constant: row.constant,
        };

        form.evaluate(claims, pad, wires) == Flat::from_raw(F::ZERO)
    }

    fn mix(seed: u128) -> Flat<F> {
        F::from(
            seed.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .wrapping_add(0x51ed_2701),
        )
        .to_hardware()
    }

    fn bus_ast() -> ConstraintAst<F> {
        let mut arena = ConstraintArena::<F>::new();

        let a = arena.cell(ProgramCell::current(0));
        let b = arena.cell(ProgramCell::current(1));
        let root = arena.mul(a, b);

        ConstraintAst {
            arena,
            roots: vec![root],
            labels: vec![None],
        }
    }

    /// One table over [`bus_ast`]: `c0 · c1` as the only
    /// constraint, two columns, one Permutation bus column
    /// keying `c0` against a public constant, selector `c1`.
    fn bus_record(
        ast: &ConstraintAst<F>,
        claims_masked: Vec<Flat<F>>,
        h_masked: [Flat<F>; BUS_LIMBS],
    ) -> TableRecord<'_, F> {
        let mul_nodes = mul_node_count(ast);

        let column = BusColumnRow {
            h_pads: [BUS_H_PAD, BUS_H_PAD + 1],
            h_masked,
            first: bus_fraction(3, Some(1)),
            second: None,
            eq_lookup: BusValue::one(),
            first_wire: mul_nodes as u32,
        };

        TableRecord {
            claims_masked,
            ast,
            consistency: ConsistencyInputs {
                pad_first: BUS_PAD_FIRST,
                eq_zc: mix(1),
                alpha: mix(2),
                tau: field().tau(),
                boundary: Vec::new(),
                telescope: Vec::new(),
                buses: vec![column],
                val_final_form: Vec::new(),
                val_final_masked: mix(4),
            },
            fixed: Vec::new(),
            claimed_sums: (0..BUS_LIMBS)
                .map(|j| (BusGroup::Permutation, j, j as u32, mix(7 + j as u128)))
                .collect(),
            shape: TableShape {
                mul_nodes,
                ..shape(1, 2, 3, 1)
            },
        }
    }

    fn eval_without_gadget() -> EvalRecord<F> {
        EvalRecord {
            mask_form: Vec::new(),
            claim_masked: mix(5),
            fin: mix(6),
            gadget: None,
        }
    }

    fn mentions(rows: &[AffineRow<F>], mul: u32, role: WireRole) -> usize {
        rows.iter()
            .filter(|r| {
                r.unknowns
                    .iter()
                    .any(|&(u, _)| u == Unknown::Wire { mul, role })
            })
            .count()
    }

    fn rich_ast() -> ConstraintAst<F> {
        let mut arena = ConstraintArena::<F>::new();

        let c0 = arena.cell(ProgramCell::current(0));
        let c1 = arena.cell(ProgramCell::current(1));
        let c2 = arena.cell(ProgramCell::current(2));
        let n0 = arena.cell(ProgramCell::next(0));

        let k = arena.constant(F::from(0x55u128));

        let prod = arena.mul(c0, c1);
        let square = arena.mul(c2, c2);

        arena.mul(c1, n0);

        let scaled = arena.scale(F::from(0x1234u128), prod);
        let sum = arena.sum(vec![scaled, n0, k, n0, square]);
        let nested = arena.mul(sum, prod);
        let root = arena.add(nested, c2);

        ConstraintAst {
            arena,
            roots: vec![root, prod, root, square],
            labels: vec![None; 4],
        }
    }

    fn linear_ast() -> ConstraintAst<F> {
        let mut arena = ConstraintArena::<F>::new();

        let c0 = arena.cell(ProgramCell::current(0));
        let n1 = arena.cell(ProgramCell::next(1));
        let k = arena.constant(F::from(7u128));
        let scaled = arena.scale(F::from(3u128), n1);
        let root = arena.sum(vec![c0, scaled, k]);

        ConstraintAst {
            arena,
            roots: vec![root],
            labels: vec![None],
        }
    }

    fn empty_ast() -> ConstraintAst<F> {
        ConstraintAst {
            arena: ConstraintArena::new(),
            roots: Vec::new(),
            labels: Vec::new(),
        }
    }

    fn statics<'a>(
        ast: &'a ConstraintAst<F>,
        specs: &'a [(String, PermutationCheckSpec)],
    ) -> TableStatics<'a, F> {
        TableStatics {
            ast,
            specs,
            fixed: &[],
            boundary: &[],
        }
    }

    fn one_bus() -> Vec<(String, PermutationCheckSpec)> {
        vec![(
            "bus".into(),
            PermutationCheckSpec::new(vec![(Source::Column(0), b"v".as_slice())], None),
        )]
    }

    fn raw_at(min_security_bits: usize) -> Config {
        Config {
            min_security_bits,
            zero_knowledge: false,
            ..Config::prod()
        }
    }

    fn empty_pool() -> PoolLayout {
        PoolLayout {
            split_vars: 0,
            tables: Vec::new(),
        }
    }

    fn weighted_record(
        ast: &ConstraintAst<F>,
        base: u32,
        columns: u32,
        buses: u32,
        fixed: bool,
    ) -> TableRecord<'_, F> {
        let seed = base as u128;
        let limbs = BUS_LIMBS as u32;
        let half = columns + limbs * buses;
        let mul_nodes = mul_node_count(ast);

        let free = base + 2 * half;
        let sums_pad = free + limbs * buses;
        let mask_pad = sums_pad + limbs * buses;

        let wires = |k: u32| match k {
            1 => 2 * KARATSUBA_WIRES + BUS_LIMBS,
            _ => KARATSUBA_WIRES,
        };

        let mut first_wire = mul_nodes as u32;
        let mut bus_rows = Vec::with_capacity(buses as usize);

        for k in 0..buses {
            let s = seed + 100 * k as u128;

            bus_rows.push(BusColumnRow {
                h_pads: [free + limbs * k, free + limbs * k + 1],
                h_masked: [mix(s + 10), mix(s + 11)],
                first: BusFraction {
                    slots: vec![(0, value(s + 12)), (columns - 1, value(s + 13))],
                    constant: value(s + 14),
                    selector: (k % 2 == 0).then_some(1),
                    recv_selector: (k == 1).then_some(0),
                },
                second: (k == 1).then(|| bus_fraction(s + 15, Some(1))),
                eq_lookup: match k % 2 {
                    0 => BusValue::one(),
                    _ => value(s + 16),
                },
                first_wire,
            });

            first_wire += wires(k) as u32;
        }

        TableRecord {
            claims_masked: (0..2 * half as u128).map(|i| mix(seed + 300 + i)).collect(),
            ast,
            consistency: ConsistencyInputs {
                pad_first: base,
                eq_zc: mix(seed + 1),
                alpha: mix(seed + 2),
                tau: field().tau(),
                boundary: vec![
                    (1, mix(seed + 5), mix(seed + 6)),
                    (0, mix(seed + 7), mix(seed + 8)),
                ],
                telescope: vec![(columns - 1, half + columns - 1)],
                buses: bus_rows,
                val_final_form: (0..3)
                    .map(|j| (Unknown::Pad(mask_pad + j), mix(seed + 60 + j as u128)))
                    .collect(),
                val_final_masked: mix(seed + 9),
            },
            fixed: match fixed {
                true => vec![(base, 0, mix(seed + 70)), (base + 1, 1, mix(seed + 71))],
                false => Vec::new(),
            },
            claimed_sums: (0..buses)
                .flat_map(|k| {
                    (0..BUS_LIMBS).map(move |j| {
                        let group = match k {
                            0 => BusGroup::Permutation,
                            _ => BusGroup::Lookup(alloc::format!("table {base} bus {k}")),
                        };

                        let limb = limbs * k + j as u32;

                        (group, j, sums_pad + limb, mix(seed + 90 + limb as u128))
                    })
                })
                .collect(),
            shape: TableShape {
                num_vars: 3,
                num_columns: columns as usize,
                sumcheck_degree: 3,
                bus_columns: buses as usize,
                bus_wires: (0..buses).map(wires).sum(),
                mul_nodes,
                ring_units: false,
                whole_blinds: 0,
            },
        }
    }

    fn weighted_eval(base: u32, first_wire: u32) -> EvalRecord<F> {
        let seed = base as u128;

        let mut mask_form: Vec<(Unknown, Flat<F>)> = (0..3)
            .map(|j| (Unknown::Pad(base + j), mix(seed + 40 + j as u128)))
            .collect();

        let entries = (0..4)
            .map(|c| (base + 6 + c, mix(seed + 50 + c as u128)))
            .collect();

        let mu = (0..linearized::BITS as u128)
            .map(|j| mix(seed + 100 + j))
            .collect();

        let gadget = RingGadget::new(entries, mu, first_wire);
        mask_form.extend(gadget.delta_form());

        EvalRecord {
            mask_form,
            claim_masked: mix(seed + 80),
            fin: mix(seed + 81),
            gadget: Some(gadget),
        }
    }

    /// Three tables over `asts`: the first carries every
    /// row class, the Permutation group spans two tables.
    fn weighted_statement(
        asts: &[ConstraintAst<F>; 3],
    ) -> (
        Vec<TableRecord<'_, F>>,
        EvalRecord<F>,
        OuterStatement,
        OuterLayout,
    ) {
        let records = vec![
            weighted_record(&asts[0], 0, 4, 2, true),
            weighted_record(&asts[1], 600, 3, 1, false),
            weighted_record(&asts[2], 1200, 2, 0, true),
        ];

        let table_wires: usize = records.iter().map(|r| r.shape.mul_wires()).sum();
        let eval = weighted_eval(1800, table_wires as u32);

        let statement = OuterStatement {
            masked_scalars: 2048,
            mul_wires: table_wires + linearized::BITS - 1,
        };

        let geom = hekate_core::config::Config::prod()
            .outer_geom(statement.masked_scalars, statement.mul_wires, 128)
            .unwrap();

        let layout =
            OuterLayout::new(&geom, statement.masked_scalars, statement.mul_wires).unwrap();

        (records, eval, statement, layout)
    }

    /// Statement rows of each nonempty row class, in [`assemble`] order.
    fn row_classes(
        records: &[TableRecord<'_, F>],
        eval: &EvalRecord<F>,
    ) -> Vec<(&'static str, Range<usize>)> {
        let mut classes = Vec::new();
        let mut at = 0;

        for record in records {
            for (class, len) in [
                ("predicate", 2 * record.shape.mul_nodes),
                ("consistency", 1),
                ("bus operands", 2 * record.shape.bus_wires),
                ("fixed", record.fixed.len()),
                ("h pins", record.shape.h_columns()),
            ] {
                if len > 0 {
                    classes.push((class, at..at + len));
                }

                at += len;
            }
        }

        let chain = eval_rows(eval) - 1;

        classes.push(("eval final", at..at + 1));

        at += 1;

        if chain > 0 {
            classes.push(("chain", at..at + chain));

            at += chain;
        }

        classes.push(("bus sums", at..at + bus_endpoints(records).len()));

        classes
    }

    #[test]
    fn ring_units_add_one_squaring_chain() {
        let plain = shape(12, 100, 3, 0);
        let ring = TableShape {
            ring_units: true,
            ..plain
        };

        let without = OuterStatement::new(&[plain, plain], 12);
        let with = OuterStatement::new(&[ring, ring], 12);

        assert_eq!(ring.mul_wires(), plain.mul_wires());
        assert_eq!(with.mul_wires - without.mul_wires, 127);
    }

    #[test]
    fn ring_blind_planes_are_masked_only_under_blinding() {
        let plain = shape(12, 100, 3, 0);
        let ring = TableShape {
            ring_units: true,
            ..plain
        };

        assert_eq!(ring.masked_scalars(), plain.masked_scalars());

        for whole_blinds in [1, 4] {
            let blinded = |s: TableShape| TableShape { whole_blinds, ..s };

            assert_eq!(
                blinded(ring).masked_scalars() - blinded(plain).masked_scalars(),
                2 * RING_BLIND_BITS
            );
        }
    }

    #[test]
    fn empty_arena_has_no_wires() {
        let ast = ConstraintAst::<F> {
            arena: ConstraintArena::new(),
            roots: vec![],
            labels: vec![],
        };

        assert_eq!(mul_node_count(&ast), 0);
    }

    #[test]
    fn unreachable_mul_nodes_carry_no_wire() {
        let mut arena = ConstraintArena::<F>::new();

        let a = arena.cell(ProgramCell::current(0));
        let b = arena.cell(ProgramCell::current(1));
        let live = arena.mul(a, b);
        let orphan = arena.cell(ProgramCell::current(2));

        arena.mul(orphan, b);

        let ast = ConstraintAst {
            arena,
            roots: vec![live],
            labels: vec![None],
        };

        assert_eq!(mul_node_count(&ast), 1);
    }

    #[test]
    fn shared_mul_node_counts_once() {
        let mut arena = ConstraintArena::<F>::new();

        let a = arena.cell(ProgramCell::current(0));
        let b = arena.cell(ProgramCell::current(1));

        let shared = arena.mul(a, b);
        let left = arena.add(shared, a);
        let right = arena.add(shared, b);

        let ast = ConstraintAst {
            arena,
            roots: vec![left, right],
            labels: vec![None, None],
        };

        assert_eq!(mul_node_count(&ast), 1);
    }

    #[test]
    fn nested_mul_nodes_each_carry_wire() {
        let mut arena = ConstraintArena::<F>::new();

        let a = arena.cell(ProgramCell::current(0));
        let b = arena.cell(ProgramCell::current(1));
        let c = arena.cell(ProgramCell::current(2));

        let inner = arena.mul(a, b);
        let outer = arena.mul(inner, c);

        let ast = ConstraintAst {
            arena,
            roots: vec![outer],
            labels: vec![None],
        };

        assert_eq!(mul_node_count(&ast), 2);
    }

    #[test]
    fn mul_under_sum_and_scale_is_reachable() {
        let mut arena = ConstraintArena::<F>::new();

        let a = arena.cell(ProgramCell::current(0));
        let b = arena.cell(ProgramCell::current(1));
        let product = arena.mul(a, b);
        let scaled = arena.scale(F::ONE, product);
        let summed = arena.sum(vec![scaled, a]);

        let ast = ConstraintAst {
            arena,
            roots: vec![summed],
            labels: vec![None],
        };

        assert_eq!(mul_node_count(&ast), 1);
    }

    #[test]
    fn busless_table_pays_for_no_h_claims() {
        let with_bus = TableShape {
            whole_blinds: 1,
            ..shape(12, 100, 3, 1)
        };

        let without = TableShape {
            bus_columns: 0,
            bus_wires: 0,
            ..with_bus
        };

        let claimed_sum_and_h_eval = 2 * BUS_LIMBS;
        let h_claim_halves = 2 * BUS_LIMBS;

        assert_eq!(
            with_bus.masked_scalars() - without.masked_scalars(),
            claimed_sum_and_h_eval + h_claim_halves
        );
    }

    #[test]
    fn masked_scalars_match_transcript_tally() {
        for whole_blinds in [1, 8] {
            let s = TableShape {
                whole_blinds,
                ..shape(12, 1977, 5, 3)
            };

            let claimed_sums_and_h_evals = 2 * BUS_LIMBS * 3;
            let zerocheck_rounds = 12 * 5;
            let claims = 2 * (1977 + whole_blinds + BUS_LIMBS * 3);

            assert_eq!(
                s.masked_scalars(),
                claimed_sums_and_h_evals + zerocheck_rounds + claims
            );
        }
    }

    #[test]
    fn every_bus_column_adds_karatsuba_wires() {
        let s = TableShape {
            mul_nodes: 40,
            ..shape(12, 100, 3, 3)
        };

        assert_eq!(s.mul_wires(), 40 + 3 * KARATSUBA_WIRES);
    }

    #[test]
    fn statement_sums_over_every_table() {
        let main = TableShape {
            mul_nodes: 40,
            whole_blinds: 2,
            ..shape(12, 100, 3, 2)
        };

        let chiplet = TableShape {
            mul_nodes: 7,
            whole_blinds: 8,
            ..shape(10, 30, 4, 1)
        };

        let statement = OuterStatement::new(&[main, chiplet], 12);

        assert_eq!(
            statement.masked_scalars,
            main.masked_scalars() + chiplet.masked_scalars() + 2 * 12
        );
        assert_eq!(statement.mul_wires, 40 + 7 + 3 * KARATSUBA_WIRES);
    }

    #[test]
    fn every_hadamard_wire_operand_is_pinned_by_row() {
        let claims: Vec<Flat<F>> = (0..8).map(|i| mix(100 + i)).collect();
        let ast = bus_ast();
        let record = bus_record(&ast, claims, [mix(8), mix(9)]);
        let statement = OuterStatement {
            masked_scalars: 32,
            mul_wires: record.shape.mul_wires(),
        };

        let records = [record];
        let eval = eval_without_gadget();
        let rows = assemble(&records, &eval, &statement).unwrap();

        for mul in 0..statement.mul_wires as u32 {
            for role in [WireRole::Lhs, WireRole::Rhs, WireRole::Product] {
                assert!(
                    mentions(&rows.affine, mul, role) >= 1,
                    "wire {mul} {role:?}"
                );
            }
        }
    }

    #[test]
    fn bus_column_rows_hold_on_honest_wires() {
        let field = field();
        let plain: Vec<Flat<F>> = (0..8).map(|i| mix(200 + i)).collect();
        let pad: Vec<Flat<F>> = (0..32).map(|i| mix(300 + i)).collect();
        let claims_masked = masked(&plain, &pad);

        let h = value(400);
        let h_masked = [
            h.lo + pad[BUS_H_PAD as usize],
            h.hi + pad[BUS_H_PAD as usize + 1],
        ];

        for second in [
            None,
            Some(bus_fraction(5, None)),
            Some(bus_fraction(5, Some(1))),
        ] {
            let column = BusColumnRow {
                h_pads: [BUS_H_PAD, BUS_H_PAD + 1],
                h_masked,
                first: bus_fraction(3, Some(1)),
                second,
                eq_lookup: value(500),
                first_wire: 0,
            };

            let mut wires = Vec::new();
            bus_wire_values(
                &field,
                &column.first,
                column.second.as_ref(),
                &plain,
                h,
                &mut wires,
            );

            let mut rows = Vec::new();
            let consistency = column_terms(&column, BUS_PAD_FIRST, field.tau(), &mut |row| {
                rows.push(row)
            });

            assert_eq!(rows.len(), 2 * wires.len());
            assert!(
                rows.iter()
                    .all(|row| holds(row, &claims_masked, &pad, &wires))
            );

            for (mul, [lhs, rhs, product]) in wires.iter().enumerate() {
                assert_eq!(*lhs * *rhs, *product);
                assert_eq!(mentions(&rows, mul as u32, WireRole::Lhs), 1);
                assert_eq!(mentions(&rows, mul as u32, WireRole::Rhs), 1);
            }

            let d1 = column.first.denominator(&plain);
            let s1 = BusValue::from_base(column.first.selector_value(&plain));

            let expected = match &column.second {
                None => field.mul(h, d1) + s1,
                Some(second) => {
                    let d2 = second.denominator(&plain);
                    let s2 = BusValue::from_base(second.selector_value(&plain));

                    field.mul(field.mul(h, d1), d2) + field.mul(s1, d2) + field.mul(s2, d1)
                }
            };

            let got = consistency.map(|form| form.evaluate(&claims_masked, &pad, &wires));

            assert_eq!(got, expected.limbs());

            let bus_sum = bus_sum_forms(&column, field.tau())
                .map(|form| form.evaluate(&claims_masked, &pad, &wires));

            assert_eq!(bus_sum, field.mul(column.eq_lookup, h).limbs());
        }
    }

    #[test]
    fn bus_column_rows_reject_free_operand() {
        let field = field();
        let plain: Vec<Flat<F>> = (0..8).map(|i| mix(200 + i)).collect();
        let pad: Vec<Flat<F>> = (0..32).map(|i| mix(300 + i)).collect();
        let claims_masked = masked(&plain, &pad);

        let h = value(400);
        let column = BusColumnRow {
            h_pads: [BUS_H_PAD, BUS_H_PAD + 1],
            h_masked: [
                h.lo + pad[BUS_H_PAD as usize],
                h.hi + pad[BUS_H_PAD as usize + 1],
            ],
            first: bus_fraction(3, Some(1)),
            second: None,
            eq_lookup: BusValue::one(),
            first_wire: 0,
        };

        let mut wires = Vec::new();
        bus_wire_values(
            &field,
            &column.first,
            column.second.as_ref(),
            &plain,
            h,
            &mut wires,
        );

        let mut rows = Vec::new();
        column_terms(&column, BUS_PAD_FIRST, field.tau(), &mut |row| {
            rows.push(row)
        });

        let one = Flat::from_raw(F::ONE);

        let mut forged = wires.clone();
        forged[0][0] = one;
        forged[0][2] = one * forged[0][1];

        assert!(!holds(&rows[0], &claims_masked, &pad, &forged));
        assert!(holds(&rows[1], &claims_masked, &pad, &forged));
    }

    #[test]
    fn layout_holds_whole_statement() {
        let cfg = Config::prod();

        for (scalars, wires) in [(1usize, 0usize), (4_075, 2_384), (12_000, 20_000)] {
            let geom = cfg.outer_geom(scalars, wires, 128).unwrap();
            let layout = OuterLayout::new(&geom, scalars, wires).unwrap();

            assert!(layout.pad_rows * layout.message_len >= scalars);
            assert!(layout.aux_rows * layout.message_len >= wires);

            let (r, _) = layout.slot(Unknown::Pad(scalars as u32 - 1));
            assert!(r < layout.pad_rows);

            if wires > 0 {
                let (r, _) = layout.slot(Unknown::Wire {
                    mul: wires as u32 - 1,
                    role: WireRole::Product,
                });

                assert!(r < layout.interleaved_mask());
            }

            assert_eq!(layout.total_rows(), layout.quadratic_mask_hi() + 1);
        }
    }

    #[test]
    fn every_mask_row_interleaved_test_reveals_is_uniform() {
        let cfg = Config::prod();
        let (scalars, wires) = (4_075usize, 2_384usize);

        let geom = cfg.outer_geom(scalars, wires, 128).unwrap();
        let layout = OuterLayout::new(&geom, scalars, wires).unwrap();

        let triples: Vec<[Flat<F>; 3]> = (0..wires as u128)
            .map(|i| [mix(i), mix(i + (1 << 20)), mix(i + (1 << 40))])
            .collect();

        let filler: Vec<Flat<F>> = (0..aux_filler_len(&layout, geom.code_len) as u128)
            .map(|i| mix(i + (1 << 60)))
            .collect();

        let rows = build_aux_rows(&layout, &triples, geom.code_len, &filler).unwrap();

        let zero = Flat::from_raw(F::ZERO);

        for mask in [
            layout.interleaved_mask(),
            layout.linear_mask_hi(),
            layout.quadratic_mask_hi(),
        ] {
            assert!(
                rows[mask - layout.pad_rows][..layout.message_len]
                    .iter()
                    .all(|v| *v != zero)
            );
        }

        let quadratic = layout.quadratic_mask() - layout.pad_rows;
        assert!(
            rows[quadratic][..layout.message_len]
                .iter()
                .all(|v| *v == zero)
        );

        assert!(
            build_aux_rows(&layout, &triples, geom.code_len, &filler[1..]).is_err(),
            "filler length is exact"
        );
    }

    #[test]
    fn mask_form_matches_round_by_round_recursion() {
        let degree = 3;

        let initial: Vec<(Unknown, Flat<F>)> = (0..4u32)
            .map(|i| (Unknown::Pad(i), mix(500 + i as u128)))
            .collect();

        let round_pads: Vec<Vec<u32>> = (0..5u32)
            .map(|i| {
                (0..degree as u32)
                    .map(|j| 100 + i * degree as u32 + j)
                    .collect()
            })
            .collect();

        let round_refs: Vec<&[u32]> = round_pads.iter().map(|p| p.as_slice()).collect();
        let challenges: Vec<Flat<F>> = (0..5).map(|i| mix(600 + i)).collect();

        let mut expected: Vec<(Unknown, Flat<F>)> = initial.clone();
        for (pads, &r) in round_refs.iter().zip(&challenges) {
            let weights = lagrange_weights::<F>(degree, r);
            for (_, coeff) in expected.iter_mut() {
                *coeff *= weights[0];
            }

            expected.push((Unknown::Pad(pads[0]), weights[0]));

            for (j, &pad) in pads.iter().enumerate() {
                expected.push((Unknown::Pad(pad), weights[j + 1]));
            }
        }

        assert_eq!(
            sumcheck_mask_form(&initial, &round_refs, &challenges, degree),
            expected
        );
    }

    #[test]
    fn h_claim_pin_row_ties_h_eval_to_committed_claim() {
        let pad: Vec<Flat<F>> = (0..32).map(|i| mix(300 + i)).collect();
        let one = Flat::from_raw(F::ONE);

        let h = [mix(400), mix(401)];
        let h_masked = [
            h[0] + pad[BUS_H_PAD as usize],
            h[1] + pad[BUS_H_PAD as usize + 1],
        ];

        for j in 0..BUS_LIMBS {
            let claim = H_CLAIM + j;
            let h_pad = BUS_H_PAD + j as u32;

            let pin_form = [
                (Unknown::Pad(h_pad), one),
                (Unknown::Pad(BUS_PAD_FIRST + claim as u32), one),
            ];

            for (h_claim, honest) in [(h[j], true), (h[j] + one, false)] {
                let mut plain: Vec<Flat<F>> = (0..8).map(|i| mix(200 + i)).collect();
                plain[claim] = h_claim;

                let claims_masked = masked(&plain, &pad);

                let ast = bus_ast();
                let record = bus_record(&ast, claims_masked.clone(), h_masked);
                let statement = OuterStatement {
                    masked_scalars: 32,
                    mul_wires: record.shape.mul_wires(),
                };

                let records = [record];
                let eval = eval_without_gadget();
                let rows = assemble(&records, &eval, &statement).unwrap();
                let pins: Vec<&AffineRow<F>> = rows
                    .affine
                    .iter()
                    .filter(|r| r.unknowns == pin_form)
                    .collect();

                assert_eq!(pins.len(), 1, "one pin row per limb");
                assert_eq!(pins[0].constant, h_masked[j] + claims_masked[claim]);

                let left = pad[h_pad as usize] + pad[BUS_PAD_FIRST as usize + claim];

                assert_eq!(left == pins[0].constant, honest);
            }
        }
    }

    #[test]
    fn batch_tensor_is_product_over_set_bits() {
        let a = F::from(0x1234_5678_9abc_def0u128);
        let b = F::from(0x0fed_cba9_8765_4321u128);
        let c = F::from(0xdead_beef_cafe_babeu128);

        let (fa, fb, fc) = (a.to_hardware(), b.to_hardware(), c.to_hardware());
        let one = Flat::from_raw(F::ONE);

        assert_eq!(expand_batch_tensor::<F>(&[], 1), vec![one]);
        assert_eq!(expand_batch_tensor(&[a], 2), vec![one, fa]);
        assert_eq!(expand_batch_tensor(&[a, b], 4), vec![one, fa, fb, fa * fb]);

        assert_eq!(
            expand_batch_tensor(&[a, b, c], 8),
            vec![one, fa, fb, fa * fb, fc, fa * fc, fb * fc, fa * fb * fc,],
        );
    }

    #[test]
    fn batch_tensor_truncates_below_power_of_two() {
        let a = F::from(0x1234_5678_9abc_def0u128);
        let b = F::from(0x0fed_cba9_8765_4321u128);

        let full = expand_batch_tensor(&[a, b], 4);

        for len in 1..=4 {
            assert_eq!(expand_batch_tensor(&[a, b], len), full[..len]);
        }
    }

    #[test]
    fn reverse_pass_reproduces_assembled_weights() {
        let asts = [rich_ast(), linear_ast(), empty_ast()];
        let (records, eval, statement, layout) = weighted_statement(&asts);

        let rows = statement_rows(&records, &eval, &statement).unwrap();
        let assembled = assemble(&records, &eval, &statement).unwrap();

        assert_eq!(assembled.affine.len(), rows);

        let tensor: Vec<F> = (0..linear_tensor_vars(rows) as u128)
            .map(|i| mix(9000 + i).to_tower())
            .collect();

        let got = linear_weights(&layout, &statement, &records, &eval, &tensor).unwrap();
        let want = linear_weights_of_rows(&layout, &assembled, &tensor).unwrap();

        assert_eq!(got.rows, want.rows);
        assert_eq!(got.weights, want.weights);
        assert_eq!(got.target, want.target);
    }

    #[test]
    fn dropping_row_class_breaks_equality() {
        let asts = [rich_ast(), linear_ast(), empty_ast()];
        let (records, eval, statement, layout) = weighted_statement(&asts);

        let assembled = assemble(&records, &eval, &statement).unwrap();
        let scales: Vec<Flat<F>> = (0..assembled.affine.len() as u128)
            .map(|i| mix(7000 + i))
            .collect();

        let want = weigh_rows(&layout, &assembled, &scales);
        let full = weigh_records(&layout, &records, &eval, &scales).unwrap();

        assert_eq!(full.weights, want.weights);
        assert_eq!(full.target, want.target);

        for (class, rows) in row_classes(&records, &eval) {
            let mut cut = scales.clone();
            cut[rows].fill(Flat::from_raw(F::ZERO));

            let got = weigh_records(&layout, &records, &eval, &cut).unwrap();

            assert!(
                got.weights != want.weights || got.target != want.target,
                "{class}"
            );
        }
    }

    #[test]
    fn miscounted_mul_nodes_are_rejected() {
        let ast = rich_ast();

        for (delta, message) in [
            (-1, "table shape undercounts the AST's Mul nodes"),
            (1, "table shape overcounts the AST's Mul nodes"),
        ] {
            let mut record = weighted_record(&ast, 0, 4, 2, true);
            record.shape.mul_nodes = record.shape.mul_nodes.saturating_add_signed(delta);

            let statement = OuterStatement {
                masked_scalars: 2048,
                mul_wires: record.shape.mul_wires(),
            };

            let geom = Config::prod()
                .outer_geom(statement.masked_scalars, statement.mul_wires, 128)
                .unwrap();

            let layout =
                OuterLayout::new(&geom, statement.masked_scalars, statement.mul_wires).unwrap();

            let records = [record];
            let eval = eval_without_gadget();
            let rows = statement_rows(&records, &eval, &statement).unwrap();
            let tensor: Vec<F> = (0..linear_tensor_vars(rows) as u128)
                .map(|i| mix(8000 + i).to_tower())
                .collect();

            match linear_weights(&layout, &statement, &records, &eval, &tensor) {
                Err(errors::Error::Protocol { message: got, .. }) => assert_eq!(got, message),
                _ => panic!("{message}"),
            }
        }
    }

    #[test]
    fn mismatched_scale_count_is_rejected() {
        let asts = [rich_ast(), linear_ast(), empty_ast()];
        let (records, eval, statement, layout) = weighted_statement(&asts);

        let rows = statement_rows(&records, &eval, &statement).unwrap();
        let scales: Vec<Flat<F>> = (0..=rows as u128).map(|i| mix(6000 + i)).collect();

        assert!(weigh_records(&layout, &records, &eval, &scales[..rows]).is_ok());

        for len in [0, rows - 1, rows + 1] {
            match weigh_records(&layout, &records, &eval, &scales[..len]) {
                Err(errors::Error::Protocol { message, .. }) => {
                    assert_eq!(
                        message, "scale count does not match the statement rows",
                        "{len}"
                    )
                }
                _ => panic!("{len}"),
            }
        }
    }

    #[test]
    fn check_draws_rejects_each_round_under_its_label() {
        let roots = rich_ast();
        let none = empty_ast();
        let specs = one_bus();

        let empty = empty_pool();
        let plan = RingSwitchPlan::new(&[ColumnType::B32], None, 0, 0).unwrap();
        let packed = PoolLayout::at_split(&[(&plan, 4)], 6);

        assert_eq!((packed.num_sigma(), packed.rho_vars()), (1, 2));

        let cases = [
            (
                statics(&none, &specs),
                shape(4, 2, 3, 1),
                &empty,
                252,
                "bus draw",
                251,
            ),
            (
                statics(&roots, &[]),
                shape(4, 3, 2, 0),
                &empty,
                126,
                "zerocheck draw",
                125,
            ),
            (
                statics(&none, &[]),
                shape(1, 1, 9, 0),
                &empty,
                125,
                "sumcheck round",
                124,
            ),
            (
                statics(&none, &[]),
                shape(1, 1, 1, 0),
                &packed,
                127,
                "eval batch draw",
                126,
            ),
            (
                statics(&none, &[]),
                shape(1, 1, 1, 0),
                &empty,
                128,
                "eval sumcheck round",
                127,
            ),
        ];

        for (table, shape, pool, floor, label, bits) in cases {
            match check_draws(&raw_at(floor), pool, [(table, shape)]) {
                Err(errors::Error::Config(config::Error::SecurityTooLow {
                    round,
                    estimated_bits,
                    min_bits,
                })) => assert_eq!((round, estimated_bits, min_bits), (label, bits, floor)),
                other => panic!("{label}: {other:?}"),
            }
        }
    }

    #[test]
    fn check_draws_reports_weakest_table_per_round() {
        let roots = rich_ast();
        let none = empty_ast();
        let specs = one_bus();

        let draws = check_draws(
            &raw_at(100),
            &empty_pool(),
            [
                (statics(&roots, &specs), shape(4, 3, 3, 1)),
                (statics(&none, &[]), shape(1, 1, 5, 0)),
            ],
        )
        .unwrap();

        assert_eq!(
            draws,
            DrawBits {
                bus: 251,
                zerocheck: 124,
                sumcheck: 125,
                eval_batch: 128,
                eval_round: 127,
                outer: None,
            }
        );
    }

    #[test]
    fn weakest_draw_counts_outer_round() {
        let none = empty_ast();
        let table = [(statics(&none, &[]), shape(4, 2, 3, 0))];

        let zk = Config {
            zero_knowledge: true,
            ..raw_at(0)
        };

        let draws = check_draws(&zk, &empty_pool(), table).unwrap();

        assert!(draws.outer.is_some());
        assert!(
            check_draws(&raw_at(0), &empty_pool(), table)
                .unwrap()
                .outer
                .is_none()
        );

        let below = DrawBits {
            outer: Some(draws.weakest() - 1),
            ..draws
        };

        assert_eq!(below.weakest(), draws.weakest() - 1);
    }
}
