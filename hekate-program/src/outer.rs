// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use crate::chiplet::ChipletDef;
use crate::constraint::{BoundaryConstraint, ConstraintAst, ConstraintExpr, ExprId};
use crate::expander::{
    ClaimBinding, EVAL_SUMCHECK_DEGREE, RING_BLIND_BITS, RingSwitchPlan, claim_weights, field_bits,
    field_coeffs,
};
use crate::fold_pool::{FoldPool, PoolTable};
use crate::linearized::{self, RingGadget, linearized_coeffs};
use crate::permutation::{
    BusColumn, BusFraction, BusGroup, PermutationCheckSpec, RankClock, bus_alpha_weights,
    bus_columns, bus_draw_events, h_planes, plane_basis,
};
use crate::predicate::{
    AffineRow, ClaimLayout, Form, Unknown, WireRole, compile, unshipped_next_cell,
};
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
use hekate_core::skip::{SkipDomain, SkipLanes, SkipPlan, SkipPoint};
use hekate_core::trace::{ColumnType, TraceCompatibleField};
use hekate_math::{Block8, Block16, Flat, FlatPromote, HardwareField, TowerField};

const SKIP_MIN_VARS: usize = 9;
const SKIP_MIN_VARS_HIGH_DEGREE: usize = 12;
const SKIP_HIGH_DEGREE: usize = 5;
const SKIP_BLOCK8_VARS: usize = 6;
const SKIP_BLOCK16_VARS: usize = 7;

pub struct LinearBatch<F> {
    pub weights: Vec<Vec<Flat<F>>>,
    pub rows: Vec<usize>,
    pub target: Flat<F>,
}

/// Claimed-sum endpoints `(pad, masked sum)` per group.
type BusSums<'r, F> = BTreeMap<&'r BusGroup, Vec<(u32, Flat<F>)>>;

/// One `h` column in the outer statement: its plane pads and masked
/// planes, its fractions, its row weight and its table-local first wire.
pub struct BusColumnRow<F> {
    pub h_pads: Vec<u32>,
    pub h_masked: Vec<Flat<F>>,
    pub first: BusFraction<F>,
    pub second: Option<BusFraction<F>>,
    pub eq_lookup: Flat<F>,
    pub first_wire: u32,
}

pub struct ConsistencyInputs<F> {
    /// Pad index of claim 0, matching `ClaimLayout`.
    pub pad_first: u32,

    pub eq_zc: Flat<F>,
    pub alpha: Flat<F>,

    /// The element behind each B128 plane of an `h` value.
    pub planes: Vec<Flat<F>>,

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

    /// Packed `h` columns, `h_planes` B128 planes each.
    pub bus_columns: usize,

    pub h_planes: usize,

    /// Hadamard wires of the LogUp terms.
    pub bus_wires: usize,

    pub mul_nodes: usize,
    pub expands_bits: bool,
    pub subfield_units: bool,
    pub whole_blinds: usize,

    /// Claims of the units that ship a next-row half.
    pub next_claims: usize,

    /// The lane field a skip may run the constraints in:
    /// GF(2^8) when the AST reads bit columns and
    /// byte constants only, GF(2^16) otherwise.
    pub skip_lanes: SkipLanes,
}

impl TableShape {
    pub fn from_air<F: TowerField>(
        air: &impl Air<F>,
        statics: &TableStatics<'_, F>,
    ) -> errors::Result<Self> {
        let ast = statics.ast;

        ast.validate_order()?;

        let layout = air.virtual_column_layout();

        let byte_lanes = (0..ast.arena.len() as u32).all(|id| match ast.arena.get(ExprId(id)) {
            ConstraintExpr::Cell(cell) => layout.get(cell.col_idx) == Some(&ColumnType::Bit),
            ConstraintExpr::Const(c) | ConstraintExpr::Scale(c, _) => in_byte_field(c),
            _ => true,
        });

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

        let plan = RingSwitchPlan::new(
            air.column_layout(),
            entries.as_deref(),
            0,
            0,
            &next_row_reads(ast, air.num_columns())?,
        )?;

        Ok(Self {
            num_vars: 0,
            num_columns: air.num_columns(),
            sumcheck_degree,
            bus_columns: columns.len(),
            h_planes: h_planes::<F>(),
            bus_wires,
            mul_nodes: mul_node_count(ast),
            expands_bits: plan.expands_bits(),
            subfield_units: plan.has_subfield_units(),
            whole_blinds: 0,
            next_claims: 0,
            skip_lanes: match byte_lanes {
                true => SkipLanes::Block8,
                false => SkipLanes::Block16,
            },
        })
    }

    /// Counts the committed `h` planes.
    pub fn h_columns(&self) -> usize {
        self.h_planes * self.bus_columns
    }

    /// Derives the table's skip over GF(2^8) lanes for a bit
    /// table whose degree fits the field, GF(2^16) otherwise.
    pub fn skip(&self) -> SkipPlan {
        let min_vars = match self.sumcheck_degree >= SKIP_HIGH_DEGREE {
            true => SKIP_MIN_VARS_HIGH_DEGREE,
            false => SKIP_MIN_VARS,
        };

        if self.num_vars < min_vars {
            return SkipPlan::NONE;
        }

        let byte = SkipPlan {
            vars: SKIP_BLOCK8_VARS.min(self.num_vars / 2),
            lanes: SkipLanes::Block8,
            cosets: self.sumcheck_degree,
        };

        if self.skip_lanes == SkipLanes::Block8 && byte.check().is_ok() {
            return byte;
        }

        let wide = SkipPlan {
            vars: SKIP_BLOCK16_VARS.min(self.num_vars / 2),
            lanes: SkipLanes::Block16,
            cosets: self.sumcheck_degree,
        };

        match wide.check() {
            Ok(()) => wide,
            Err(_) => SkipPlan::NONE,
        }
    }

    pub fn at(self, table: &PoolTable) -> Self {
        Self {
            num_vars: table.num_vars,
            expands_bits: table.plan.expands_bits(),
            subfield_units: table.plan.has_subfield_units(),
            whole_blinds: table.plan.whole_blinds(),
            next_claims: table.plan.next_claims(),
            ..self
        }
    }

    /// Whole blinds of the helper polynomial, one per
    /// trace whole blind when the table carries a bus.
    pub fn h_blinds(&self) -> usize {
        match self.bus_columns > 0 {
            true => self.whole_blinds,
            false => 0,
        }
    }

    /// Current-row eval claims: virtual columns, whole blind units,
    /// the ring blind unit's planes, every `h` plane, `h`'s blinds.
    pub fn eval_claims(&self) -> usize {
        let ring_blind = match self.expands_bits && self.whole_blinds > 0 {
            true => RING_BLIND_BITS,
            false => 0,
        };

        self.num_columns + self.whole_blinds + ring_blind + self.h_columns() + self.h_blinds()
    }

    /// Final sumcheck values are the masked running
    /// claims themselves and consume no pad entry.
    pub fn masked_scalars(&self) -> usize {
        let skip = self.skip();
        let zerocheck = skip.sent_scalars() + (self.num_vars - skip.vars) * self.sumcheck_degree;
        let claims = self.eval_claims() + self.next_claims;

        self.bus_columns + self.h_columns() + zerocheck + claims
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
    pub skip: Option<usize>,
    pub sumcheck: usize,
    pub eval_batch: usize,
    pub eval_round: usize,
    pub outer: Option<usize>,
}

impl DrawBits {
    pub fn weakest(&self) -> usize {
        [
            Some(self.bus),
            Some(self.zerocheck),
            self.skip,
            Some(self.sumcheck),
            Some(self.eval_batch),
            Some(self.eval_round),
            self.outer,
        ]
        .into_iter()
        .flatten()
        .fold(usize::MAX, usize::min)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OuterStatement {
    pub masked_scalars: usize,
    pub mul_wires: usize,
}

impl OuterStatement {
    pub fn new<F: TowerField>(shapes: &[TableShape], pool_vars: usize) -> Self {
        let tables: usize = shapes.iter().map(TableShape::masked_scalars).sum();
        let wires: usize = shapes.iter().map(TableShape::mul_wires).sum();

        let gadget = match shapes.iter().any(|s| s.subfield_units) {
            true => F::BITS - 1,
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
            let statics = def.statics();
            let shape = TableShape::from_air(def, &statics)?;

            plans.push(table_plan(def, &statics, &shape, config)?);
            shapes.push(shape);
        }

        let main_statics = main_parts.statics();
        let main_shape = TableShape::from_air(main, &main_statics)?;

        plans.push(table_plan(main, &main_statics, &main_shape, config)?);
        shapes.push(main_shape);

        let heights: Vec<(&RingSwitchPlan, usize)> = plans
            .iter()
            .zip(chiplet_num_vars.iter().chain([&main_num_vars]))
            .map(|(plan, &num_vars)| (plan, num_vars))
            .collect();

        let pool = FoldPool::new(&heights, config)?;

        let shapes: Vec<TableShape> = shapes
            .into_iter()
            .zip(pool.tables())
            .map(|(shape, table)| shape.at(table))
            .collect();

        Ok(Self::new::<F>(&shapes, pool.layout().num_vars()))
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

/// A sumcheck round as the outer statement masks it: the pads
/// of its sent values, its Lagrange weights at the challenge
/// and the leading pads that rebuild the derived point.
pub struct RoundForm<F> {
    pub pads: Vec<u32>,
    pub weights: Vec<Flat<F>>,
    pub derived_from: usize,
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

    /// `(group, pad, masked sum)` per `h` column.
    pub claimed_sums: Vec<(BusGroup, u32, Flat<F>)>,

    /// Plane bindings of both halves, in claim indices.
    pub bindings: Vec<ClaimBinding>,

    pub next_slots: Vec<Option<usize>>,
    pub shape: TableShape,
}

impl<F: TowerField + HardwareField> TableRecord<'_, F> {
    fn affine_rows(&self) -> usize {
        2 * self.shape.mul_wires()
            + 1
            + self.shape.h_columns()
            + self.fixed.len()
            + self.bindings.len()
    }

    fn claim_layout(&self) -> ClaimLayout<'_> {
        ClaimLayout {
            pad_first: self.consistency.pad_first,
            half: self.shape.eval_claims() as u32,
            next: &self.next_slots,
        }
    }

    /// `f` on each row after the predicate rows, in statement order;
    /// the consistency row comes without its root terms.
    fn for_each_tail_row(&self, mut f: impl FnMut(AffineRow<F>)) {
        let consistency = &self.consistency;
        let pad_first = consistency.pad_first;

        f(consistency_row(self.ast.roots.len(), consistency));

        for column in &consistency.buses {
            column_terms(column, pad_first, &consistency.planes, &mut f);
        }

        for &(pad, claim, public) in &self.fixed {
            f(fixed_column_row(pad, claim, public));
        }

        let half = self.shape.eval_claims() as u32;
        let first_h = half - (self.shape.h_columns() + self.shape.h_blinds()) as u32;
        let planes = self.shape.h_planes;

        for (c, column) in consistency.buses.iter().enumerate() {
            for j in 0..planes {
                let claim = first_h + (planes * c + j) as u32;

                f(h_claim_pin_row(
                    column.h_pads[j],
                    column.h_masked[j],
                    pad_first + claim,
                    claim,
                ));
            }
        }

        for binding in &self.bindings {
            f(plane_binding_row(
                pad_first,
                binding.whole as u32,
                binding.first as u32,
                binding.width,
            ));
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
    pub pool: &'a FoldPool,
    pub eta: F,
    pub rho: &'a [F],
    pub r_mix: &'a [F],
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
    pub gamma: F,
    pub beta: F,
    pub r_zerocheck: &'a [Flat<F>],
    pub r_final: &'a [Flat<F>],
    pub lookup_bus_points: &'a BTreeMap<String, Vec<F>>,
    pub clocks: &'a [Option<RankClock>],

    pub claimed_sums_masked: &'a [F],
    pub h_evals_masked: &'a [F],
    pub val_final_masked: Flat<F>,
    pub claims_masked: Vec<Flat<F>>,
    pub bindings: Vec<ClaimBinding>,
    pub next_slots: Vec<Option<usize>>,

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
        let layout = record.claim_layout();

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
                    let idx = layout.index(*cell).ok_or(unshipped_next_cell())?;

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

pub fn eval_initial_form<F: TraceCompatibleField + From<u128>>(
    pool: &FoldPool,
    eta: F,
    rho: &[F],
    r_mix: &[F],
    claim_pads: &[u32],
    first_wire: u32,
) -> errors::Result<InitialForm<F>> {
    let rho: Vec<Flat<F>> = rho.iter().map(|r| r.to_hardware()).collect();

    let mut weights = Vec::with_capacity(claim_pads.len());
    for (i, table) in pool.tables().iter().enumerate() {
        let unit_weights = pool.unit_weights(i, eta.to_hardware(), &rho)?;

        weights.extend(claim_weights(&table.plan, &unit_weights));
    }

    if weights.len() != claim_pads.len() {
        return Err(errors::Error::Protocol {
            protocol: "outer",
            message: "one pad entry per eval claim is required",
        });
    }

    let mut form = Vec::new();
    let mut ring = Vec::new();
    let mut fields: Vec<ColumnType> = Vec::new();

    for (&(field, a), &pad) in weights.iter().zip(claim_pads) {
        if field == ColumnType::B128 {
            form.push((Unknown::Pad(pad), a));
            continue;
        }

        let k = match fields.iter().position(|&f| f == field) {
            Some(k) => k,
            None => {
                fields.push(field);
                fields.len() - 1
            }
        };

        ring.push((pad, a, k));
    }

    if ring.is_empty() {
        return Ok(InitialForm { form, gadget: None });
    }

    let maps: Vec<(Vec<F>, usize)> = fields
        .iter()
        .map(|&field| (field_coeffs(field, r_mix), field_bits(field)))
        .collect();

    let gadget = RingGadget::new(ring, linearized_coeffs(&maps)?, first_wire)?;

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
    F: TraceCompatibleField + From<u128>,
{
    let initial = eval_initial_form(
        inputs.pool,
        inputs.eta,
        inputs.rho,
        inputs.r_mix,
        inputs.claim_pads,
        inputs.first_wire,
    )?;

    let rounds = multilinear_rounds(inputs.challenges, 2, inputs.round_first);

    Ok(EvalRecord {
        mask_form: sumcheck_mask_form(&initial.form, &rounds),
        claim_masked: inputs.claim_masked,
        fin: inputs.fin,
        gadget: initial.gadget,
    })
}

pub fn table_record<'s, F>(inputs: TableInputs<'_, 's, F>) -> errors::Result<TableRecord<'s, F>>
where
    F: TowerField + HardwareField + FlatPromote<Block8> + FlatPromote<Block16>,
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
    let pad_first = inputs.pads.claims;

    let skip = shape.skip();
    let point = SkipPoint::from_challenges(skip, inputs.r_final)?;
    let half = shape.eval_claims();
    let next_slots = inputs.next_slots;

    if inputs.claims_masked.len() != half + shape.next_claims || next_slots.len() != half {
        return Err(errors::Error::Protocol {
            protocol: "outer",
            message: "eval claim count does not match the table shape",
        });
    }

    let next_index = |claim: usize| -> errors::Result<usize> {
        let slot = next_slots[claim].ok_or(errors::Error::Protocol {
            protocol: "outer",
            message: "claim read at the next row ships no next-row claim",
        })?;

        Ok(half + slot)
    };

    let mut bindings = Vec::with_capacity(2 * inputs.bindings.len());
    for binding in &inputs.bindings {
        bindings.push(*binding);

        if next_slots[binding.whole].is_some() {
            bindings.push(ClaimBinding {
                whole: next_index(binding.whole)?,
                first: next_index(binding.first)?,
                width: binding.width,
            });
        }
    }

    let telescope = (0..inputs.blinding_columns)
        .map(|k| {
            let b = trace_width + k;

            Ok((b as u32, next_index(b)? as u32))
        })
        .collect::<errors::Result<Vec<_>>>()?;

    let mut shapes = ShapeEvaluator::new(&point);
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
            point.weight(bc.row_idx),
        ));
    }

    let h_columns = shape.h_columns();
    let planes = shape.h_planes;

    if inputs.claimed_sums_masked.len() != shape.bus_columns
        || inputs.h_evals_masked.len() != h_columns
    {
        return Err(errors::Error::Protocol {
            protocol: "outer",
            message: "LogUp claim count does not match the table shape",
        });
    }

    let columns = bus_columns(bus_specs, shape.sumcheck_degree);

    let mut buses = Vec::with_capacity(columns.len());
    let mut claimed_sums = Vec::with_capacity(columns.len());

    if !columns.is_empty() {
        let challenges = [inputs.gamma.to_hardware(), inputs.beta.to_hardware()];

        let fraction = |idx: usize| {
            BusFraction::new(
                idx,
                &bus_specs[idx],
                trace_width,
                challenges,
                &point,
                inputs.clocks,
            )
        };

        let mut first_wire = shape.mul_nodes as u32;
        for (c, column) in columns.iter().enumerate() {
            let group = column.group(bus_specs);
            let eq_lookup = group.row_weight(inputs.lookup_bus_points, &point)?;

            claimed_sums.push((
                group,
                inputs.pads.claimed_sums + c as u32,
                inputs.claimed_sums_masked[c].to_hardware(),
            ));

            buses.push(BusColumnRow {
                h_pads: (0..planes)
                    .map(|j| inputs.pads.h_evals + (planes * c + j) as u32)
                    .collect(),
                h_masked: (0..planes)
                    .map(|j| inputs.h_evals_masked[planes * c + j].to_hardware())
                    .collect(),
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

    let initial: Vec<(Unknown, Flat<F>)> = (0..columns.len() as u32)
        .zip(bus_alpha_weights(alpha_pow, inputs.alpha))
        .map(|(c, w)| (Unknown::Pad(inputs.pads.claimed_sums + c), w.bus_sum))
        .collect();

    let rounds = zerocheck_rounds(
        skip,
        inputs.r_final,
        shape.sumcheck_degree,
        inputs.pads.zerocheck,
    )?;

    let consistency = ConsistencyInputs {
        pad_first,
        eq_zc: point.eq_with(inputs.r_zerocheck)?,
        alpha: inputs.alpha,
        planes: plane_basis::<F>()?,
        boundary,
        telescope,
        buses,
        val_final_form: sumcheck_mask_form(&initial, &rounds),
        val_final_masked: inputs.val_final_masked,
    };

    Ok(TableRecord {
        claims_masked: inputs.claims_masked,
        ast,
        consistency,
        fixed,
        claimed_sums,
        bindings,
        next_slots,
        shape,
    })
}

/// Pushes an `h` column's honest `(lhs, rhs, product)` wires,
/// in wire order, from the true claims and `h`.
pub fn bus_wire_values<F: TowerField + HardwareField>(
    first: &BusFraction<F>,
    second: Option<&BusFraction<F>>,
    claims: &[Flat<F>],
    h: Flat<F>,
    out: &mut Vec<[Flat<F>; 3]>,
) {
    let d1 = first.denominator(claims);

    out.push([h, d1, h * d1]);

    let Some(second) = second else {
        return;
    };

    let u = h * d1 + first.selector_value(claims);
    let d2 = second.denominator(claims);

    out.push([u, d2, u * d2]);

    if second.selector_claimed() {
        let s2 = second.selector_value(claims);
        out.push([s2, d1, s2 * d1]);
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

        let predicate = compile(record.ast, record.claim_layout())?;
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
/// one bus-sum row per bus group.
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

    let weights = bus_alpha_weights(alpha_pow, inputs.alpha);

    for (column, w) in inputs.buses.iter().zip(weights) {
        let consistency = column_terms(column, inputs.pad_first, &inputs.planes, &mut |_| {});
        let bus_sum = h_form(column, &inputs.planes).scaled(column.eq_lookup);

        add_form(&mut row, &consistency.scaled(w.consistency * inputs.eq_zc));
        add_form(&mut row, &bus_sum.scaled(w.bus_sum));
    }

    row
}

/// Pad form of `claim'_n - claim_n` for a sumcheck whose
/// replay reconstructs `p'_0 = claim' - p'_1` each round:
/// entries of round `i` carry `Π_{j>i} L_0(r_j)`.
pub fn sumcheck_mask_form<F: TowerField + HardwareField>(
    initial: &[(Unknown, Flat<F>)],
    rounds: &[RoundForm<F>],
) -> Vec<(Unknown, Flat<F>)> {
    let mut suffix = vec![Flat::from_raw(F::ONE); rounds.len() + 1];
    for i in (0..rounds.len()).rev() {
        suffix[i] = suffix[i + 1] * rounds[i].weights[0];
    }

    let mut form: Vec<(Unknown, Flat<F>)> = initial
        .iter()
        .map(|&(u, coeff)| (u, coeff * suffix[0]))
        .collect();

    for (i, round) in rounds.iter().enumerate() {
        let after = suffix[i + 1];

        for (j, &pad) in round.pads.iter().enumerate() {
            let mut weight = round.weights[j + 1];

            if j < round.derived_from {
                weight += round.weights[0];
            }

            form.push((Unknown::Pad(pad), weight * after));
        }
    }

    form
}

/// Builds a zerocheck's rounds for the mask form: the skip
/// round first when the table skips, then the multilinear ones.
pub fn zerocheck_rounds<F>(
    skip: SkipPlan,
    challenges: &[Flat<F>],
    degree: usize,
    first_pad: u32,
) -> errors::Result<Vec<RoundForm<F>>>
where
    F: TowerField + HardwareField + FlatPromote<Block8> + FlatPromote<Block16>,
{
    if !skip.is_skip() {
        return Ok(multilinear_rounds(challenges, degree, first_pad));
    }

    let (&r_skip, tail) = challenges.split_first().ok_or(errors::Error::Protocol {
        protocol: "outer",
        message: "skipped zerocheck ended without its skip challenge",
    })?;

    let sent = skip.sent_scalars() as u32;

    let mut rounds = Vec::with_capacity(challenges.len());

    rounds.push(RoundForm {
        pads: (first_pad..first_pad + sent).collect(),
        weights: SkipDomain::<F>::new(skip)?.lagrange(r_skip),
        derived_from: skip.block_rows() - 1,
    });

    rounds.extend(multilinear_rounds(tail, degree, first_pad + sent));

    Ok(rounds)
}

/// Builds `degree` multilinear rounds for the mask form, their
/// pads consecutive from `first_pad`, `P(0)` rebuilt from `P(1)`.
pub fn multilinear_rounds<F: TowerField + HardwareField>(
    challenges: &[Flat<F>],
    degree: usize,
    first_pad: u32,
) -> Vec<RoundForm<F>> {
    challenges
        .iter()
        .enumerate()
        .map(|(i, &r)| {
            let first = first_pad + (i * degree) as u32;

            RoundForm {
                pads: (first..first + degree as u32).collect(),
                weights: lagrange_weights::<F>(degree, r),
                derived_from: 1,
            }
        })
        .collect()
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

pub fn plane_binding_row<F: TowerField + HardwareField>(
    pad_first: u32,
    whole: u32,
    first: u32,
    width: usize,
) -> AffineRow<F> {
    let mut row = AffineRow {
        unknowns: Vec::with_capacity(width + 1),
        claims: Vec::with_capacity(width + 1),
        constant: Flat::from_raw(F::ZERO),
    };

    push_claim(&mut row, pad_first, whole, Flat::from_raw(F::ONE));

    for v in 0..width {
        push_claim(
            &mut row,
            pad_first,
            first + v as u32,
            F::from(1u128 << v).to_hardware(),
        );
    }

    row
}

/// `Σ claimed_sum = 0` over one bus group:
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
pub fn linear_weights<F: TraceCompatibleField>(
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
pub fn linear_weights_of_rows<F: TraceCompatibleField>(
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

    weigh_rows(
        layout,
        rows,
        &expand_batch_tensor(tensor, rows.affine.len()),
    )
}

/// Builds a table's ring-switch plan with the `h` planes of `shape`
/// and a next-row half for each unit the AST reads at the next row.
pub fn table_plan<F: TowerField>(
    air: &impl Air<F>,
    statics: &TableStatics<'_, F>,
    shape: &TableShape,
    config: &Config,
) -> errors::Result<RingSwitchPlan> {
    let entries = air.virtual_expander().map(|e| e.expansion_entries());

    RingSwitchPlan::new(
        air.column_layout(),
        entries.as_deref(),
        config.blind_units(),
        shape.h_columns(),
        &next_row_reads(statics.ast, shape.num_columns)?,
    )
}

/// Gates every draw past the pool term against `config`: the bus
/// draw, each table's `α`/`r_zerocheck` draw and sumcheck rounds,
/// the eval's batch draw and rounds, the outer argument in ZK mode.
pub fn check_draws<'s, F, I>(
    config: &Config,
    pool: &FoldPool,
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
    let mut skip_events: u128 = 0;
    let mut sumcheck_events: u128 = 0;

    for (statics, shape) in tables {
        let terms = statics.ast.roots.len()
            + statics.boundary.len()
            + config.blind_units()
            + 2 * shape.bus_columns;

        let round = shape.sumcheck_degree + usize::from(shape.subfield_units);

        zerocheck_events = zerocheck_events.max((terms + shape.num_vars) as u128);
        skip_events = skip_events.max(shape.skip().domain_points() as u128);
        sumcheck_events = sumcheck_events.max(round as u128);
    }

    let skip = match skip_events {
        0 => None,
        events => {
            config.check_draw("skip round", F::BITS, events)?;
            Some(config.draw_bits(F::BITS, events))
        }
    };

    let mix = match pool.has_subfield_units() {
        true => F::BITS.ilog2() as usize,
        false => 0,
    };

    // The eval batch identity's degree: `2Σ - 1` in η over both
    // claim halves, multilinear in ρ and, on a ring, in r_mix.
    let batch_events = ((2 * pool.num_sigma()).saturating_sub(1) + pool.rho_vars() + mix) as u128;
    let round_events = EVAL_SUMCHECK_DEGREE as u128;

    config.check_draw("bus draw", F::BITS, bus_events)?;
    config.check_draw("zerocheck draw", F::BITS, zerocheck_events)?;
    config.check_draw("sumcheck round", F::BITS, sumcheck_events)?;
    config.check_draw("eval batch draw", F::BITS, batch_events)?;
    config.check_draw("eval sumcheck round", F::BITS, round_events)?;

    let outer = match config.zero_knowledge {
        true => {
            let statement = OuterStatement::new::<F>(&shapes, pool.layout().num_vars());
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
        bus: config.draw_bits(F::BITS, bus_events),
        zerocheck: config.draw_bits(F::BITS, zerocheck_events),
        skip,
        sumcheck: config.draw_bits(F::BITS, sumcheck_events),
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

/// Tells whether `c` lies in GF(2^8), the tower elements
/// with every coordinate past the first byte zero.
fn in_byte_field<F: TowerField>(c: &F) -> bool {
    let mut raw = [0u8; linearized::MAX_BYTES];
    let bytes = &mut raw[..c.serialized_size()];

    c.serialize(bytes).is_ok() && bytes[1..].iter().all(|&b| b == 0)
}

/// Marks each virtual column the AST reads at the next row;
/// errs on a cell outside the `num_columns` virtual columns.
fn next_row_reads<F: TowerField>(
    ast: &ConstraintAst<F>,
    num_columns: usize,
) -> errors::Result<Vec<bool>> {
    let mut reads = vec![false; num_columns];
    for id in 0..ast.arena.len() as u32 {
        let ConstraintExpr::Cell(cell) = ast.arena.get(ExprId(id)) else {
            continue;
        };

        if cell.col_idx >= num_columns {
            return Err(errors::Error::Protocol {
                protocol: "air",
                message: "AST cell outside the virtual columns",
            });
        }

        reads[cell.col_idx] |= cell.next_row;
    }

    Ok(reads)
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

fn weigh_records<F: TraceCompatibleField>(
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
        let chain = &scales[first..first + linearized::chain_rows::<F>()];
        for (row, &scale) in gadget.wire_rows().iter().zip(chain) {
            acc.row(row, scale, &[], 0);
        }

        gadget.for_each_tie_weight(chain, |pad, weight| acc.add(Unknown::Pad(pad), weight))?;

        first += linearized::chain_rows::<F>();
    }

    for (endpoints, &scale) in buses.values().zip(&scales[first..]) {
        acc.row(&bus_sum_row(endpoints), scale, &[], 0);
    }

    Ok(acc.batch())
}

fn weigh_table<F: TowerField + HardwareField>(
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

fn weigh_rows<F: TraceCompatibleField>(
    layout: &OuterLayout,
    rows: &OuterRows<'_, F>,
    scales: &[Flat<F>],
) -> errors::Result<LinearBatch<F>> {
    let mut acc = Accumulator::new(layout);
    for (row, &scale) in rows.affine.iter().zip(scales) {
        acc.row(row, scale, &[], 0);
    }

    for &(first, gadget) in &rows.gadgets {
        let chain = &scales[first..first + linearized::chain_rows::<F>()];
        gadget.for_each_tie_weight(chain, |pad, weight| acc.add(Unknown::Pad(pad), weight))?;
    }

    Ok(acc.batch())
}

fn bus_endpoints<'r, F: TowerField>(records: &'r [TableRecord<'_, F>]) -> BusSums<'r, F> {
    let mut buses = BusSums::new();
    for record in records {
        for (group, pad, masked) in &record.claimed_sums {
            buses.entry(group).or_default().push((*pad, *masked));
        }
    }

    buses
}

fn eval_rows<F: TowerField>(eval: &EvalRecord<F>) -> usize {
    match eval.gadget {
        Some(_) => 1 + linearized::chain_rows::<F>(),
        None => 1,
    }
}

fn eval_wires<F: TowerField>(eval: &EvalRecord<F>) -> usize {
    match eval.gadget {
        Some(_) => F::BITS - 1,
        None => 0,
    }
}

/// Counts the Hadamard wires one bus column adds to the outer statement.
fn column_wires(column: &BusColumn, specs: &[(String, PermutationCheckSpec)]) -> usize {
    let Some(second) = column.second else {
        return 1;
    };

    let spec = &specs[second].1;
    let selector_wires = usize::from(spec.selector.is_some() || spec.recv_selector.is_some());

    2 + selector_wires
}

/// Emits a column's operand rows into `f` and returns its consistency `C`:
/// `g·D + s`, or `(g·D₁ + s₁)·D₂ + s₂·D₁` for a pair.
fn column_terms<F: TowerField + HardwareField>(
    column: &BusColumnRow<F>,
    pad_first: u32,
    planes: &[Flat<F>],
    f: &mut impl FnMut(AffineRow<F>),
) -> Form<F> {
    let one = Flat::from_raw(F::ONE);
    let g = h_form(column, planes);
    let d1 = denominator_form(&column.first, pad_first);
    let wire = column.first_wire;

    f(operand_row(&g, wire, WireRole::Lhs));
    f(operand_row(&d1, wire, WireRole::Rhs));

    let mut c = product_form(wire);
    c.extend(&selector_form(&column.first, pad_first));

    let Some(second) = &column.second else {
        return c;
    };

    let d2 = denominator_form(second, pad_first);

    f(operand_row(&c, wire + 1, WireRole::Lhs));
    f(operand_row(&d2, wire + 1, WireRole::Rhs));

    let mut c = product_form(wire + 1);

    match second.selector_claimed() {
        true => {
            let s2 = selector_form(second, pad_first);

            f(operand_row(&s2, wire + 2, WireRole::Lhs));
            f(operand_row(&d1, wire + 2, WireRole::Rhs));

            c.unknowns.push((product_wire(wire + 2), one));
        }
        false => c.extend(&d1),
    }

    c
}

/// Returns a column's `h` as a form over its plane pads,
/// `Σ_k plane_k · basis_k` with the masked planes as the constant.
fn h_form<F: TowerField + HardwareField>(column: &BusColumnRow<F>, planes: &[Flat<F>]) -> Form<F> {
    let mut form = Form::default();
    for ((&pad, &masked), &basis) in column.h_pads.iter().zip(&column.h_masked).zip(planes) {
        form.unknowns.push((Unknown::Pad(pad), basis));

        form.constant += masked * basis;
    }

    form
}

fn denominator_form<F: TowerField + HardwareField>(
    fraction: &BusFraction<F>,
    pad_first: u32,
) -> Form<F> {
    let mut form = Form {
        unknowns: Vec::with_capacity(fraction.slots.len()),
        claims: Vec::with_capacity(fraction.slots.len()),
        constant: fraction.constant,
    };

    for &(claim, coeff) in &fraction.slots {
        form.unknowns.push((Unknown::Pad(pad_first + claim), coeff));
        form.claims.push((claim, coeff));
    }

    form
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

fn product_form<F: TowerField + HardwareField>(wire: u32) -> Form<F> {
    Form {
        unknowns: vec![(product_wire(wire), Flat::from_raw(F::ONE))],
        claims: Vec::new(),
        constant: Flat::from_raw(F::ZERO),
    }
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

    /// `half - h_columns` at `bus_record`'s `half = 4` and one bus column of one plane.
    const H_CLAIM: usize = 3;

    fn shape(num_vars: usize, num_columns: usize, degree: usize, buses: usize) -> TableShape {
        TableShape {
            num_vars,
            num_columns,
            sumcheck_degree: degree,
            bus_columns: buses,
            h_planes: 1,
            bus_wires: buses,
            mul_nodes: 0,
            expands_bits: false,
            subfield_units: false,
            whole_blinds: 0,
            next_claims: 0,
            skip_lanes: SkipLanes::Block16,
        }
    }

    fn every_half(shape: TableShape) -> TableShape {
        TableShape {
            next_claims: shape.eval_claims(),
            ..shape
        }
    }

    fn value(seed: u128) -> Flat<F> {
        mix(seed + 1000)
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
    /// constraint, three columns, one Permutation bus column
    /// keying `c0` against a public constant, selector `c1`.
    fn bus_record(
        ast: &ConstraintAst<F>,
        claims_masked: Vec<Flat<F>>,
        h_masked: Flat<F>,
    ) -> TableRecord<'_, F> {
        let mul_nodes = mul_node_count(ast);
        let half = claims_masked.len() / 2;

        let column = BusColumnRow {
            h_pads: vec![BUS_H_PAD],
            h_masked: vec![h_masked],
            first: bus_fraction(3, Some(1)),
            second: None,
            eq_lookup: Flat::from_raw(F::ONE),
            first_wire: mul_nodes as u32,
        };

        TableRecord {
            claims_masked,
            ast,
            consistency: ConsistencyInputs {
                pad_first: BUS_PAD_FIRST,
                eq_zc: mix(1),
                alpha: mix(2),
                planes: plane_basis().unwrap(),
                boundary: Vec::new(),
                telescope: Vec::new(),
                buses: vec![column],
                val_final_form: Vec::new(),
                val_final_masked: mix(4),
            },
            fixed: Vec::new(),
            claimed_sums: vec![(BusGroup::Permutation, 0, mix(7))],
            bindings: Vec::new(),
            next_slots: (0..half).map(Some).collect(),
            shape: TableShape {
                mul_nodes,
                next_claims: half,
                ..shape(1, 3, 3, 1)
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

    fn pool_of(columns: usize, field: ColumnType) -> FoldPool {
        let plan =
            RingSwitchPlan::new(&vec![field; columns], None, 0, 0, &vec![true; columns]).unwrap();

        FoldPool::new(&[(&plan, 4)], &raw_at(0)).unwrap()
    }

    fn weighted_record(
        ast: &ConstraintAst<F>,
        base: u32,
        columns: u32,
        buses: u32,
        fixed: bool,
    ) -> TableRecord<'_, F> {
        let seed = base as u128;
        let half = columns + buses;
        let mul_nodes = mul_node_count(ast);

        let free = base + 2 * half;
        let sums_pad = free + buses;
        let mask_pad = sums_pad + buses;

        let wires = |k: u32| match k {
            1 => 3,
            _ => 1,
        };

        let mut first_wire = mul_nodes as u32;
        let mut bus_rows = Vec::with_capacity(buses as usize);

        for k in 0..buses {
            let s = seed + 100 * k as u128;

            bus_rows.push(BusColumnRow {
                h_pads: vec![free + k],
                h_masked: vec![mix(s + 10)],
                first: BusFraction {
                    slots: vec![(0, value(s + 12)), (columns - 1, value(s + 13))],
                    constant: value(s + 14),
                    selector: (k % 2 == 0).then_some(1),
                    recv_selector: (k == 1).then_some(0),
                },
                second: (k == 1).then(|| bus_fraction(s + 15, Some(1))),
                eq_lookup: match k % 2 {
                    0 => Flat::from_raw(F::ONE),
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
                planes: plane_basis().unwrap(),
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
                .map(|k| {
                    let group = match k {
                        0 => BusGroup::Permutation,
                        _ => BusGroup::Lookup(format!("table {base} bus {k}")),
                    };

                    (group, sums_pad + k, mix(seed + 90 + k as u128))
                })
                .collect(),
            bindings: Vec::new(),
            next_slots: (0..half as usize).map(Some).collect(),
            shape: TableShape {
                num_vars: 3,
                num_columns: columns as usize,
                sumcheck_degree: 3,
                bus_columns: buses as usize,
                h_planes: 1,
                bus_wires: (0..buses).map(wires).sum(),
                mul_nodes,
                expands_bits: false,
                subfield_units: false,
                whole_blinds: 0,
                next_claims: half as usize,
                skip_lanes: SkipLanes::Block16,
            },
        }
    }

    fn weighted_eval(base: u32, first_wire: u32) -> EvalRecord<F> {
        let seed = base as u128;

        let mut mask_form: Vec<(Unknown, Flat<F>)> = (0..3)
            .map(|j| (Unknown::Pad(base + j), mix(seed + 40 + j as u128)))
            .collect();

        let entries = (0..4)
            .map(|c| (base + 6 + c, mix(seed + 50 + c as u128), c as usize % 2))
            .collect();

        let mu = (0..2u128)
            .map(|k| {
                (0..F::BITS as u128)
                    .map(|j| mix(seed + 100 + 200 * k + j))
                    .collect()
            })
            .collect();

        let gadget = RingGadget::new(entries, mu, first_wire).unwrap();
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
            mul_wires: table_wires + F::BITS - 1,
        };

        let geom = Config::prod()
            .outer_geom(statement.masked_scalars, statement.mul_wires, 256)
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
    fn subfield_units_add_one_squaring_chain() {
        let plain = shape(12, 100, 3, 0);
        let ring = TableShape {
            subfield_units: true,
            ..plain
        };

        let without = OuterStatement::new::<F>(&[plain, plain], 12);
        let with = OuterStatement::new::<F>(&[ring, ring], 12);

        assert_eq!(ring.mul_wires(), plain.mul_wires());
        assert_eq!(with.mul_wires - without.mul_wires, 127);
    }

    #[test]
    fn ring_blind_planes_are_masked_only_under_blinding() {
        let plain = shape(12, 100, 3, 0);
        let ring = TableShape {
            expands_bits: true,
            ..plain
        };

        assert_eq!(ring.masked_scalars(), plain.masked_scalars());

        for whole_blinds in [1, 4] {
            let blinded = |s: TableShape| every_half(TableShape { whole_blinds, ..s });

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
        let with_bus = every_half(TableShape {
            whole_blinds: 1,
            ..shape(12, 100, 3, 1)
        });

        let without = every_half(TableShape {
            bus_columns: 0,
            bus_wires: 0,
            ..with_bus
        });

        let claimed_sum_and_h_eval = 2;

        let h_claim_halves = 2;
        let h_blind_halves = 2 * with_bus.whole_blinds;

        assert_eq!(
            with_bus.masked_scalars() - without.masked_scalars(),
            claimed_sum_and_h_eval + h_claim_halves + h_blind_halves
        );
    }

    #[test]
    fn masked_scalars_match_transcript_tally() {
        for whole_blinds in [1, 8] {
            let s = every_half(TableShape {
                whole_blinds,
                ..shape(12, 1977, 5, 3)
            });

            let claimed_sums_and_h_evals = 3 + 3;
            let skip_round = (5 << 6) - 1;
            let zerocheck_rounds = skip_round + (12 - 6) * 5;
            let h_blinds = whole_blinds;
            let claims = 2 * (1977 + whole_blinds + 3 + h_blinds);

            assert_eq!(
                s.masked_scalars(),
                claimed_sums_and_h_evals + zerocheck_rounds + claims
            );
        }
    }

    #[test]
    fn every_bus_column_adds_one_wire() {
        let s = TableShape {
            mul_nodes: 40,
            ..shape(12, 100, 3, 3)
        };

        assert_eq!(s.mul_wires(), 40 + 3);
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

        let statement = OuterStatement::new::<F>(&[main, chiplet], 12);

        assert_eq!(
            statement.masked_scalars,
            main.masked_scalars() + chiplet.masked_scalars() + 2 * 12
        );
        assert_eq!(statement.mul_wires, 40 + 7 + 3);
    }

    #[test]
    fn every_hadamard_wire_operand_is_pinned_by_row() {
        let claims: Vec<Flat<F>> = (0..8).map(|i| mix(100 + i)).collect();
        let ast = bus_ast();
        let record = bus_record(&ast, claims, mix(8));
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
        let planes = plane_basis::<F>().unwrap();
        let plain: Vec<Flat<F>> = (0..8).map(|i| mix(200 + i)).collect();
        let pad: Vec<Flat<F>> = (0..32).map(|i| mix(300 + i)).collect();
        let claims_masked = masked(&plain, &pad);

        let h = value(400);
        let h_masked = vec![h + pad[BUS_H_PAD as usize]];

        for second in [
            None,
            Some(bus_fraction(5, None)),
            Some(bus_fraction(5, Some(1))),
        ] {
            let column = BusColumnRow {
                h_pads: vec![BUS_H_PAD],
                h_masked: h_masked.clone(),
                first: bus_fraction(3, Some(1)),
                second,
                eq_lookup: value(500),
                first_wire: 0,
            };

            let mut wires = Vec::new();
            bus_wire_values(&column.first, column.second.as_ref(), &plain, h, &mut wires);

            let mut rows = Vec::new();
            let consistency =
                column_terms(&column, BUS_PAD_FIRST, &planes, &mut |row| rows.push(row));

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
            let s1 = column.first.selector_value(&plain);

            let expected = match &column.second {
                None => h * d1 + s1,
                Some(second) => {
                    let d2 = second.denominator(&plain);
                    let s2 = second.selector_value(&plain);

                    h * d1 * d2 + s1 * d2 + s2 * d1
                }
            };

            assert_eq!(consistency.evaluate(&claims_masked, &pad, &wires), expected);

            let bus_sum = h_form(&column, &planes).scaled(column.eq_lookup).evaluate(
                &claims_masked,
                &pad,
                &wires,
            );

            assert_eq!(bus_sum, column.eq_lookup * h);
        }
    }

    #[test]
    fn bus_column_rows_reject_free_operand() {
        let planes = plane_basis::<F>().unwrap();
        let plain: Vec<Flat<F>> = (0..8).map(|i| mix(200 + i)).collect();
        let pad: Vec<Flat<F>> = (0..32).map(|i| mix(300 + i)).collect();
        let claims_masked = masked(&plain, &pad);

        let h = value(400);
        let column = BusColumnRow {
            h_pads: vec![BUS_H_PAD],
            h_masked: vec![h + pad[BUS_H_PAD as usize]],
            first: bus_fraction(3, Some(1)),
            second: None,
            eq_lookup: Flat::from_raw(F::ONE),
            first_wire: 0,
        };

        let mut wires = Vec::new();
        bus_wire_values(&column.first, column.second.as_ref(), &plain, h, &mut wires);

        let mut rows = Vec::new();
        column_terms(&column, BUS_PAD_FIRST, &planes, &mut |row| rows.push(row));

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
            let geom = cfg.outer_geom(scalars, wires, 256).unwrap();
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

        let geom = cfg.outer_geom(scalars, wires, 256).unwrap();
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

        let challenges: Vec<Flat<F>> = (0..5).map(|i| mix(600 + i)).collect();
        let rounds = multilinear_rounds(&challenges, degree, 100);

        let mut expected: Vec<(Unknown, Flat<F>)> = initial.clone();
        for (round, &r) in rounds.iter().zip(&challenges) {
            let weights = lagrange_weights::<F>(degree, r);
            for (_, coeff) in expected.iter_mut() {
                *coeff *= weights[0];
            }

            assert_eq!(round.pads[0], 100 + 3 * (round.pads[0] - 100) / 3);

            for (j, &pad) in round.pads.iter().enumerate() {
                let derived = match j {
                    0 => weights[0],
                    _ => Flat::from_raw(F::ZERO),
                };

                expected.push((Unknown::Pad(pad), weights[j + 1] + derived));
            }
        }

        assert_eq!(sumcheck_mask_form(&initial, &rounds), expected);
    }

    #[test]
    fn skip_round_form_matches_recursion_on_pads() {
        let zero = Flat::from_raw(F::ZERO);
        let degree = 3;

        let skip = SkipPlan {
            vars: 2,
            lanes: SkipLanes::Block8,
            cosets: 3,
        };

        let domain = SkipDomain::<F>::new(skip).unwrap();
        let challenges: Vec<Flat<F>> = (0..4).map(|i| mix(700 + i)).collect();
        let rounds = zerocheck_rounds(skip, &challenges, degree, 40).unwrap();

        assert_eq!(rounds[0].pads.len(), skip.sent_scalars());
        assert_eq!(rounds[0].derived_from, skip.block_rows() - 1);
        assert_eq!(rounds.len(), challenges.len());

        let pad: Vec<Flat<F>> = (0..80).map(|i| mix(800 + i as u128)).collect();
        let initial = vec![(Unknown::Pad(7), mix(900))];
        let form = sumcheck_mask_form(&initial, &rounds);

        let via_form = form.iter().fold(zero, |acc, &(u, coeff)| match u {
            Unknown::Pad(p) => acc + coeff * pad[p as usize],
            _ => acc,
        });

        let mut claim = mix(900) * pad[7];
        let mut next_pad = 40usize;

        let sent: Vec<Flat<F>> = (0..skip.sent_scalars())
            .map(|_| {
                let v = pad[next_pad];
                next_pad += 1;

                v
            })
            .collect();

        let rest: Flat<F> = sent[..skip.block_rows() - 1]
            .iter()
            .fold(zero, |a, &v| a + v);

        let mut values = vec![claim - rest];
        values.extend_from_slice(&sent);

        claim = domain.interpolate(&values, challenges[0]).unwrap();

        for &r in &challenges[1..] {
            let p1 = pad[next_pad];
            let weights = lagrange_weights::<F>(degree, r);

            let mut round = vec![claim - p1];
            round.extend_from_slice(&pad[next_pad..next_pad + degree]);

            next_pad += degree;

            claim = round
                .iter()
                .zip(&weights)
                .fold(zero, |acc, (&v, &w)| acc + v * w);
        }

        assert_eq!(via_form, claim);
    }

    #[test]
    fn h_claim_pin_row_ties_h_eval_to_committed_claim() {
        let pad: Vec<Flat<F>> = (0..32).map(|i| mix(300 + i)).collect();
        let one = Flat::from_raw(F::ONE);

        let h = mix(400);
        let h_masked = h + pad[BUS_H_PAD as usize];

        for whole_blinds in [0, 1] {
            let half = 4 + 2 * whole_blinds;
            let claim = H_CLAIM + whole_blinds;

            let pin_form = [
                (Unknown::Pad(BUS_H_PAD), one),
                (Unknown::Pad(BUS_PAD_FIRST + claim as u32), one),
            ];

            for (h_claim, honest) in [(h, true), (h + one, false)] {
                let mut plain: Vec<Flat<F>> = (0..2 * half).map(|i| mix(200 + i as u128)).collect();

                plain[claim] = h_claim;

                let claims_masked = masked(&plain, &pad);
                let ast = bus_ast();

                let mut record = bus_record(&ast, claims_masked.clone(), h_masked);
                record.shape.whole_blinds = whole_blinds;

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

                assert_eq!(pins.len(), 1, "{whole_blinds} whole blinds");
                assert_eq!(pins[0].constant, h_masked + claims_masked[claim]);

                let left = pad[BUS_H_PAD as usize] + pad[BUS_PAD_FIRST as usize + claim];

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

        let want = weigh_rows(&layout, &assembled, &scales).unwrap();
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
                .outer_geom(statement.masked_scalars, statement.mul_wires, 256)
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

        let empty = pool_of(1, ColumnType::B128);
        let packed = pool_of(4, ColumnType::B128);
        let narrow = pool_of(1, ColumnType::B32);

        assert_eq!((empty.num_sigma(), empty.rho_vars()), (1, 0));
        assert_eq!((packed.num_sigma(), packed.rho_vars()), (1, 2));

        let cases = [
            (
                statics(&none, &specs),
                shape(4, 2, 3, 1),
                &empty,
                124,
                "bus draw",
                123,
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
                TableShape {
                    subfield_units: true,
                    ..shape(1, 1, 1, 0)
                },
                &empty,
                128,
                "sumcheck round",
                127,
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
                &narrow,
                126,
                "eval batch draw",
                125,
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
            &pool_of(1, ColumnType::B128),
            [
                (statics(&roots, &specs), shape(4, 3, 3, 1)),
                (statics(&none, &[]), shape(1, 1, 5, 0)),
            ],
        )
        .unwrap();

        assert_eq!(
            draws,
            DrawBits {
                bus: 123,
                zerocheck: 124,
                skip: None,
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

        let draws = check_draws(&zk, &pool_of(1, ColumnType::B128), table).unwrap();

        assert!(draws.outer.is_some());
        assert!(
            check_draws(&raw_at(0), &pool_of(1, ColumnType::B128), table)
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

    #[test]
    fn skip_plan_follows_lanes_degree_and_height() {
        let with = |num_vars, degree, lanes| TableShape {
            skip_lanes: lanes,
            ..shape(num_vars, 8, degree, 0)
        };

        let plan = |vars, lanes, cosets| SkipPlan {
            vars,
            lanes,
            cosets,
        };

        assert_eq!(
            with(15, 4, SkipLanes::Block8).skip(),
            plan(6, SkipLanes::Block8, 4)
        );
        assert_eq!(
            with(15, 5, SkipLanes::Block8).skip(),
            plan(7, SkipLanes::Block16, 5)
        );
        assert_eq!(
            with(20, 3, SkipLanes::Block16).skip(),
            plan(7, SkipLanes::Block16, 3)
        );
        assert_eq!(
            with(12, 3, SkipLanes::Block16).skip(),
            plan(6, SkipLanes::Block16, 3)
        );
        assert_eq!(
            with(9, 3, SkipLanes::Block16).skip(),
            plan(4, SkipLanes::Block16, 3)
        );
        assert_eq!(
            with(9, 4, SkipLanes::Block16).skip(),
            plan(4, SkipLanes::Block16, 4)
        );
        assert_eq!(with(8, 3, SkipLanes::Block16).skip(), SkipPlan::NONE);
        assert_eq!(with(11, 5, SkipLanes::Block16).skip(), SkipPlan::NONE);
        assert_eq!(
            with(12, 5, SkipLanes::Block16).skip(),
            plan(6, SkipLanes::Block16, 5)
        );
    }

    #[test]
    fn next_claims_enter_masked_scalars_once() {
        let current_only = shape(12, 100, 3, 1);
        let shipped = TableShape {
            next_claims: 7,
            ..current_only
        };

        assert_eq!(shipped.masked_scalars() - current_only.masked_scalars(), 7);
    }

    #[test]
    fn next_row_reads_mark_next_row_cells_only() {
        let reads = next_row_reads(&rich_ast(), 4).unwrap();

        assert_eq!(reads, [true, false, false, false]);
    }

    #[test]
    fn ast_cell_outside_virtual_columns_rejected() {
        assert!(next_row_reads(&rich_ast(), 2).is_err());
    }
}
