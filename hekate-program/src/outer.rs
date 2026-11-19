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
    AffineRow, ClaimLayout, Form, LevelEntry, PredicateRows, Unknown, WireRole, compile,
    fixed_only, fixed_values, level_rounds, product_levels, unshipped_next_cell, wired_mul,
};
use crate::{Air, FixedColumn, ProgramInstance, ShapeEvaluator};
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::mem;
use core::ops::Range;
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::outer::{OUTER_MASK_ROWS, OuterGeometry};
use hekate_core::poly::univariate::{MAX_POINTS, UnivariatePoly};
use hekate_core::skip::{SkipDomain, SkipLanes, SkipPlan, SkipPoint};
use hekate_core::tensor::TensorProduct;
use hekate_core::trace::{ColumnType, TraceCompatibleField};
use hekate_math::{Block8, Block16, Flat, FlatPromote, HardwareField, TowerField};

/// Degree of each product reduction sumcheck,
/// whose terms multiply one weight by two operands.
pub const PRODUCT_SUMCHECK_DEGREE: usize = 3;

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
    /// Index of claim 0, matching `ClaimLayout`.
    pub claims_first: u32,

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

impl<F: TowerField> TableStatics<'_, F> {
    /// Lists the table's product levels, from the roots down.
    pub fn levels(&self) -> Vec<Vec<LevelEntry>> {
        let columns = self.fixed.iter().map(|column| column.col_idx);

        product_levels(self.ast, &fixed_only(self.ast, columns))
    }
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

    /// Levels of the product reduction, and
    /// the sumcheck rounds they run in total.
    pub product_levels: usize,

    pub product_rounds: usize,

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

        let levels = statics.levels();

        Ok(Self {
            num_vars: 0,
            num_columns: air.num_columns(),
            sumcheck_degree,
            bus_columns: columns.len(),
            h_planes: h_planes::<F>(),
            bus_wires,
            product_levels: levels.len(),
            product_rounds: levels.iter().map(|level| level_rounds(level.len())).sum(),
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

        let level_sums = 2 * self.product_levels.saturating_sub(1);
        let products = PRODUCT_SUMCHECK_DEGREE * self.product_rounds + level_sums;

        self.bus_columns + self.h_columns() + zerocheck + products
    }

    /// Counts the eval claims of both rows,
    /// the table's share of the claims oracle.
    pub fn claims(&self) -> usize {
        self.eval_claims() + self.next_claims
    }

    /// Counts the table's Hadamard wires, one per
    /// product level and those of the bus products.
    pub fn mul_wires(&self) -> usize {
        self.product_levels + self.bus_wires
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
    pub claims: usize,
    pub mul_wires: usize,
}

impl OuterStatement {
    pub fn new<F: TowerField>(shapes: &[TableShape], pool_vars: usize) -> Self {
        let tables: usize = shapes.iter().map(TableShape::masked_scalars).sum();
        let claims: usize = shapes.iter().map(TableShape::claims).sum();
        let wires: usize = shapes.iter().map(TableShape::mul_wires).sum();

        let gadget = match shapes.iter().any(|s| s.subfield_units) {
            true => F::BITS - 1,
            false => 0,
        };

        Self {
            masked_scalars: tables + 2 * pool_vars,
            claims,
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

/// Oracle row order: pad, claims, then the `Lhs`, `Rhs`
/// and `Product` wire blocks, then the five mask rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OuterLayout {
    pub message_len: usize,
    pub pad_rows: usize,
    pub claim_rows: usize,
    pub aux_rows: usize,
    pub domain_len: usize,
}

impl OuterLayout {
    pub fn new(geom: &OuterGeometry, statement: &OuterStatement) -> errors::Result<Self> {
        if geom.message_len == 0 {
            return Err(errors::Error::Protocol {
                protocol: "outer",
                message: "message length must be positive",
            });
        }

        Ok(Self {
            message_len: geom.message_len,
            pad_rows: statement.masked_scalars.div_ceil(geom.message_len),
            claim_rows: statement.claims.div_ceil(geom.message_len),
            aux_rows: statement.mul_wires.div_ceil(geom.message_len).max(1),
            domain_len: geom.domain_len,
        })
    }

    pub fn slot(&self, unknown: Unknown) -> (usize, usize) {
        match unknown {
            Unknown::Pad(i) => {
                let i = i as usize;

                (i / self.message_len, i % self.message_len)
            }
            Unknown::Claim(i) => {
                let i = i as usize;

                (self.pad_rows + i / self.message_len, i % self.message_len)
            }
            Unknown::Wire { mul, role } => {
                let mul = mul as usize;
                let block = role as usize;

                (
                    self.aux_first() + block * self.aux_rows + mul / self.message_len,
                    mul % self.message_len,
                )
            }
        }
    }

    pub fn aux_first(&self) -> usize {
        self.pad_rows + self.claim_rows
    }

    pub fn interleaved_mask(&self) -> usize {
        self.aux_first() + 3 * self.aux_rows
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
        let first = self.aux_first();

        (0..self.aux_rows)
            .map(|i| {
                [
                    first + i,
                    first + self.aux_rows + i,
                    first + 2 * self.aux_rows + i,
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

/// Two padded sums that open one product level past
/// the first, with their first pad and their challenge.
pub struct LevelSums<F> {
    pub pad: u32,
    pub masked: [Flat<F>; 2],
    pub tau: Flat<F>,
}

impl<F: TowerField + HardwareField> LevelSums<F> {
    fn initial(&self) -> [(Unknown, Flat<F>); 2] {
        [
            (Unknown::Pad(self.pad), Flat::from_raw(F::ONE)),
            (Unknown::Pad(self.pad + 1), self.tau),
        ]
    }
}

/// One level of one table's product reduction, after its replay.
pub struct LevelRecord<'s, F> {
    pub entries: &'s [LevelEntry],
    pub challenges: Vec<Flat<F>>,

    /// First pad of the rounds' sent values.
    pub round_first: u32,

    /// Masked claim the replay ends on.
    pub claim_masked: Flat<F>,

    /// `None` at the first level.
    pub sums: Option<LevelSums<F>>,
}

impl<F: TowerField + HardwareField> LevelRecord<'_, F> {
    fn rounds(&self) -> Vec<RoundForm<F>> {
        multilinear_rounds(&self.challenges, PRODUCT_SUMCHECK_DEGREE, self.round_first)
    }

    fn opening_sums(&self) -> errors::Result<&LevelSums<F>> {
        self.sums.as_ref().ok_or(errors::Error::Protocol {
            protocol: "outer",
            message: "product level past the first carries no sums",
        })
    }

    /// Builds `mask form + weight · Product = masked claim`,
    /// the level's last value against its one triple.
    fn final_row(
        &self,
        initial: &[(Unknown, Flat<F>)],
        rounds: &[RoundForm<F>],
        level: u32,
        weight: Flat<F>,
    ) -> AffineRow<F> {
        let mut unknowns = sumcheck_mask_form(initial, rounds);
        unknowns.push((
            Unknown::Wire {
                mul: level,
                role: WireRole::Product,
            },
            weight,
        ));

        AffineRow {
            unknowns,
            constant: self.claim_masked,
        }
    }
}

/// One table's share of the statement, in transcript order.
pub struct TableRecord<'s, F> {
    pub ast: &'s ConstraintAst<F>,
    pub consistency: ConsistencyInputs<F>,

    /// Product reduction, from the roots down;
    /// empty until the product phase has run.
    pub products: Vec<LevelRecord<'s, F>>,

    /// `(column, public value)` per fixed column;
    /// its current-row claim sits at the column's index.
    pub fixed: Vec<(u32, Flat<F>)>,

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
            + self.shape.product_levels.max(1)
            + self.shape.h_columns()
            + self.fixed.len()
            + self.bindings.len()
    }

    fn claim_layout(&self) -> ClaimLayout<'_> {
        ClaimLayout {
            first: self.consistency.claims_first,
            half: self.shape.eval_claims() as u32,
            next: &self.next_slots,
        }
    }

    fn fixed_nodes(&self, out: &mut Vec<Option<Flat<F>>>) -> errors::Result<()> {
        let columns = self.fixed.iter().map(|&(column, _)| column as usize);
        let marks = fixed_only(self.ast, columns);

        let public = |column: usize| {
            self.fixed
                .iter()
                .find(|&&(fixed, _)| fixed as usize == column)
                .map(|&(_, value)| value)
        };

        fixed_values(self.ast, &marks, public, out)
    }

    /// `f` on each row after the product rows, in statement order.
    fn for_each_tail_row(&self, mut f: impl FnMut(AffineRow<F>)) {
        let consistency = &self.consistency;
        let first = consistency.claims_first;

        for column in &consistency.buses {
            column_terms(column, first, &consistency.planes, &mut f);
        }

        for &(column, public) in &self.fixed {
            f(fixed_column_row(first + column, public));
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
                    first + claim,
                ));
            }
        }

        for binding in &self.bindings {
            f(plane_binding_row(
                first,
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

/// `H_0` of an eval sumcheck over the claims.
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
}

/// Public data of the pool's eval sumcheck on masked rounds.
pub struct EvalInputs<'a, F> {
    pub pool: &'a FoldPool,
    pub eta: F,
    pub rho: &'a [F],
    pub r_mix: &'a [F],
    pub challenges: &'a [Flat<F>],
    pub claim_masked: Flat<F>,
    pub fin: Flat<F>,
    pub claims: usize,
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
    pub bindings: Vec<ClaimBinding>,
    pub next_slots: Vec<Option<usize>>,

    pub pads: TablePads,

    /// Index of the table's claim 0 among every table's claims.
    pub claims_first: u32,
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

    /// Adds `scale` times one table's `row`,
    /// its wires shifted by `mul_offset`.
    fn row(&mut self, row: &AffineRow<F>, scale: Flat<F>, mul_offset: u32) {
        for &(unknown, coeff) in &row.unknowns {
            self.add(unknown.shifted(mul_offset), scale * coeff);
        }

        self.target += scale * row.constant;
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
    wired: usize,
}

impl<F: TowerField + HardwareField> Sweep<F> {
    fn new() -> Self {
        Self {
            live: Vec::new(),
            adjoint: Vec::new(),
            wired: 0,
        }
    }

    fn reset(&mut self, nodes: usize) {
        self.live.clear();
        self.live.resize(nodes, false);
        self.adjoint.clear();
        self.adjoint.resize(nodes, Flat::from_raw(F::ZERO));

        self.wired = 0;
    }

    /// Hands every seeded adjoint down through linear nodes, last
    /// node first. `on_claim` takes each claim's weight, each wired
    /// product keeps its own, and reached constants sum into the result.
    fn run(
        &mut self,
        ast: &ConstraintAst<F>,
        layout: ClaimLayout<'_>,
        fixed: &[Option<Flat<F>>],
        mut on_claim: impl FnMut(u32, Flat<F>),
    ) -> errors::Result<Flat<F>> {
        let mut constant = Flat::from_raw(F::ZERO);

        // Children sit at lower indices than their parent
        for i in (0..ast.arena.len()).rev() {
            if !self.live[i] {
                continue;
            }

            let adjoint = self.adjoint[i];

            match ast.arena.get(ExprId(i as u32)) {
                ConstraintExpr::Cell(cell) => {
                    let idx = layout.index(*cell).ok_or(unshipped_next_cell())?;
                    on_claim(layout.first + idx, adjoint);
                }
                ConstraintExpr::Const(v) => {
                    constant += adjoint * fixed[i].unwrap_or_else(|| v.to_hardware());
                }
                ConstraintExpr::Add(a, b) => {
                    self.seed(*a, adjoint);
                    self.seed(*b, adjoint);
                }
                ConstraintExpr::Scale(coeff, a) => self.seed(*a, adjoint * coeff.to_hardware()),
                ConstraintExpr::Sum(children) => {
                    for c in children {
                        self.seed(*c, adjoint);
                    }
                }
                ConstraintExpr::Mul(a, b) => match (fixed[a.0 as usize], fixed[b.0 as usize]) {
                    (Some(lhs), Some(rhs)) => constant += adjoint * lhs * rhs,
                    (Some(scale), None) => self.seed(*b, adjoint * scale),
                    (None, Some(scale)) => self.seed(*a, adjoint * scale),
                    (None, None) => self.wired += 1,
                },
            }
        }

        Ok(constant)
    }

    fn seed(&mut self, node: ExprId, value: Flat<F>) {
        let n = node.0 as usize;
        self.live[n] = true;
        self.adjoint[n] += value;
    }

    fn roots(
        &mut self,
        record: &TableRecord<'_, F>,
        fixed: &[Option<Flat<F>>],
        on_claim: impl FnMut(u32, Flat<F>),
    ) -> errors::Result<Flat<F>> {
        let ast = record.ast;

        self.reset(ast.arena.len());

        let weights = record.consistency.root_weights(ast.roots.len());
        for (root, weight) in ast.roots.iter().zip(weights) {
            self.seed(*root, weight);
        }

        self.run(ast, record.claim_layout(), fixed, on_claim)
    }

    /// Sweeps one operand of every entry, entry `j` at `eq[j]`.
    fn side(
        &mut self,
        record: &TableRecord<'_, F>,
        fixed: &[Option<Flat<F>>],
        entries: &[LevelEntry],
        eq: &[Flat<F>],
        side: usize,
        on_claim: impl FnMut(u32, Flat<F>),
    ) -> errors::Result<Flat<F>> {
        let ast = record.ast;

        self.reset(ast.arena.len());

        for (entry, &weight) in entries.iter().zip(eq) {
            match ast.arena.get(entry.node) {
                ConstraintExpr::Mul(a, b) => self.seed([*a, *b][side], weight),
                _ => {
                    return Err(errors::Error::Protocol {
                        protocol: "outer",
                        message: "product level lists a node that multiplies nothing",
                    });
                }
            }
        }

        self.run(ast, record.claim_layout(), fixed, on_claim)
    }

    /// Sums the adjoints the last sweep left on `entries` under `eq`.
    fn weight(&self, entries: &[LevelEntry], eq: &[Flat<F>]) -> Flat<F> {
        entries
            .iter()
            .zip(eq)
            .fold(Flat::from_raw(F::ZERO), |acc, (entry, &e)| {
                acc + e * self.adjoint[entry.node.0 as usize]
            })
    }

    fn adjoints(&self, entries: &[LevelEntry]) -> Vec<Flat<F>> {
        entries
            .iter()
            .map(|entry| self.adjoint[entry.node.0 as usize])
            .collect()
    }

    /// Errs when the last sweep stopped at
    /// some wired product outside `entries`.
    fn covered(&self, entries: &[LevelEntry]) -> errors::Result<()> {
        let listed = entries
            .iter()
            .filter(|entry| self.live[entry.node.0 as usize])
            .count();

        match listed == self.wired {
            true => Ok(()),
            false => Err(stray_product()),
        }
    }
}

/// Public weights of one table's product reduction,
/// computed alike by the prover and the verifier.
pub struct ProductWeights<'r, 's, F> {
    record: &'r TableRecord<'s, F>,
    sweep: Sweep<F>,
    fixed: Vec<Option<Flat<F>>>,
}

impl<'r, 's, F: TowerField + HardwareField> ProductWeights<'r, 's, F> {
    pub fn new(record: &'r TableRecord<'s, F>) -> errors::Result<Self> {
        let mut fixed = Vec::new();
        record.fixed_nodes(&mut fixed)?;

        Ok(Self {
            record,
            sweep: Sweep::new(),
            fixed,
        })
    }

    /// Returns the public side of the claim the first level's products sum to.
    pub fn claim(&mut self) -> errors::Result<Flat<F>> {
        let record = self.record;
        let constant = self.sweep.roots(record, &self.fixed, |_, _| {})?;
        let row = consistency_row(record.ast.roots.len(), &record.consistency);

        Ok(row.constant + constant)
    }

    /// Weighs the first level's products and returns, with
    /// their weights, the public side of the claim they sum to.
    pub fn first(&mut self, entries: &[LevelEntry]) -> errors::Result<(Vec<Flat<F>>, Flat<F>)> {
        let claim = self.claim()?;

        Ok((self.sweep.adjoints(entries), claim))
    }

    /// Weighs `next`'s products under each operand
    /// side of `entries`, entry `j` at `eq[j]`.
    pub fn below(
        &mut self,
        entries: &[LevelEntry],
        eq: &[Flat<F>],
        next: &[LevelEntry],
    ) -> errors::Result<[Vec<Flat<F>>; 2]> {
        let mut sides = [Vec::new(), Vec::new()];
        for (side, weights) in sides.iter_mut().enumerate() {
            self.sweep
                .side(self.record, &self.fixed, entries, eq, side, |_, _| {})?;

            *weights = self.sweep.adjoints(next);
        }

        Ok(sides)
    }
}

pub fn eval_initial_form<F: TraceCompatibleField + From<u128>>(
    pool: &FoldPool,
    eta: F,
    rho: &[F],
    r_mix: &[F],
    claims: usize,
    first_wire: u32,
) -> errors::Result<InitialForm<F>> {
    let rho: Vec<Flat<F>> = rho.iter().map(|r| r.to_hardware()).collect();

    let mut weights = Vec::with_capacity(claims);
    for (i, table) in pool.tables().iter().enumerate() {
        let unit_weights = pool.unit_weights(i, eta.to_hardware(), &rho)?;

        weights.extend(claim_weights(&table.plan, &unit_weights));
    }

    if weights.len() != claims {
        return Err(errors::Error::Protocol {
            protocol: "outer",
            message: "one weight per eval claim is required",
        });
    }

    let mut form = Vec::new();
    let mut ring = Vec::new();
    let mut fields: Vec<ColumnType> = Vec::new();

    for (claim, &(field, a)) in weights.iter().enumerate() {
        let claim = claim as u32;

        if field == ColumnType::B128 {
            form.push((Unknown::Claim(claim), a));
            continue;
        }

        let k = match fields.iter().position(|&f| f == field) {
            Some(k) => k,
            None => {
                fields.push(field);
                fields.len() - 1
            }
        };

        ring.push((claim, a, k));
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
        inputs.claims,
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

    let skip = shape.skip();
    let point = SkipPoint::from_challenges(skip, inputs.r_final)?;
    let half = shape.eval_claims();
    let next_slots = inputs.next_slots;
    let shipped = next_slots.iter().flatten();

    if next_slots.len() != half || shipped.clone().any(|&slot| slot >= shape.next_claims) {
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

        fixed.push((fc.col_idx as u32, shapes.evaluate(&fc.shape)));
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

        let mut first_wire = shape.product_levels as u32;
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
        claims_first: inputs.claims_first,
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
        ast,
        consistency,
        products: Vec::new(),
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

/// Every affine row of the statement, wire indices offset per table.
pub fn assemble<'r, F: TowerField + HardwareField>(
    records: &'r [TableRecord<'_, F>],
    eval: &'r EvalRecord<F>,
    statement: &OuterStatement,
) -> errors::Result<OuterRows<'r, F>> {
    let mut affine = Vec::new();
    let mut gadgets = Vec::new();
    let mut fixed = Vec::new();
    let mut offset = 0u32;

    for record in records {
        let first = affine.len();

        record.fixed_nodes(&mut fixed)?;

        let predicate = compile(record.ast, record.claim_layout(), &fixed)?;

        affine.extend(product_rows(record, &predicate)?);

        record.for_each_tail_row(|row| affine.push(row));

        for row in &mut affine[first..] {
            for (unknown, _) in row.unknowns.iter_mut() {
                *unknown = unknown.shifted(offset);
            }
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

pub fn mul_node_count<F: TowerField>(ast: &ConstraintAst<F>, fixed_only: &[bool]) -> usize {
    reachable(ast)
        .iter()
        .enumerate()
        .filter(|&(i, &live)| live && wired_mul(ast, fixed_only, i).is_some())
        .count()
}

/// The consistency row without its root terms:
/// root `k` enters at `α^k · eq_zc`.
pub fn consistency_row<F: TowerField + HardwareField>(
    roots: usize,
    inputs: &ConsistencyInputs<F>,
) -> AffineRow<F> {
    let first = inputs.claims_first;

    let mut row = AffineRow {
        unknowns: inputs.val_final_form.to_vec(),
        constant: inputs.val_final_masked,
    };

    let mut alpha_pow = Flat::from_raw(F::ONE);

    for _ in 0..roots {
        alpha_pow *= inputs.alpha;
    }

    for &(claim, public, eq_row) in &inputs.boundary {
        let weight = eq_row * alpha_pow;

        row.unknowns.push((Unknown::Claim(first + claim), weight));

        row.constant += public * weight;
        alpha_pow *= inputs.alpha;
    }

    for &(b, b_next) in &inputs.telescope {
        row.unknowns.push((Unknown::Claim(first + b), alpha_pow));
        row.unknowns
            .push((Unknown::Claim(first + b_next), alpha_pow));

        alpha_pow *= inputs.alpha;
    }

    let weights = bus_alpha_weights(alpha_pow, inputs.alpha);

    for (column, w) in inputs.buses.iter().zip(weights) {
        let consistency = column_terms(column, first, &inputs.planes, &mut |_| {});
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
        constant: claim_masked + fin,
    }
}

/// `c[claim] = public`.
pub fn fixed_column_row<F: TowerField + HardwareField>(
    claim: u32,
    public: Flat<F>,
) -> AffineRow<F> {
    AffineRow {
        unknowns: vec![(Unknown::Claim(claim), Flat::from_raw(F::ONE))],
        constant: public,
    }
}

/// `c[claim] = h_eval`: the eval argument's base `h`
/// claim is the value the reported `h_eval` unmasks to.
pub fn h_claim_pin_row<F: TowerField + HardwareField>(
    h_pad: u32,
    h_masked: Flat<F>,
    claim: u32,
) -> AffineRow<F> {
    let one = Flat::from_raw(F::ONE);

    AffineRow {
        unknowns: vec![(Unknown::Pad(h_pad), one), (Unknown::Claim(claim), one)],
        constant: h_masked,
    }
}

pub fn plane_binding_row<F: TowerField + HardwareField>(
    claims_first: u32,
    whole: u32,
    first: u32,
    width: usize,
) -> AffineRow<F> {
    let mut unknowns = Vec::with_capacity(width + 1);

    unknowns.push((Unknown::Claim(claims_first + whole), Flat::from_raw(F::ONE)));

    for v in 0..width {
        unknowns.push((
            Unknown::Claim(claims_first + first + v as u32),
            F::from(1u128 << v).to_hardware(),
        ));
    }

    AffineRow {
        unknowns,
        constant: Flat::from_raw(F::ZERO),
    }
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
        constant,
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
    let rows = 0..layout.pad_rows;

    message_rows(layout, rows, pad, Unknown::Pad, code_len, filler).ok_or(errors::Error::Protocol {
        protocol: "outer",
        message: "pad rows do not hold the pad and its filler",
    })
}

/// CLAIMS oracle rows: the claims in [`OuterLayout::slot`] order, `filler`
/// (exactly `claim_rows · (code_len - message_len)` entries) on the message tail.
pub fn build_claim_rows<F: TowerField + HardwareField>(
    layout: &OuterLayout,
    claims: &[Flat<F>],
    code_len: usize,
    filler: &[Flat<F>],
) -> errors::Result<Vec<Vec<Flat<F>>>> {
    let rows = layout.pad_rows..layout.aux_first();

    message_rows(layout, rows, claims, Unknown::Claim, code_len, filler).ok_or(
        errors::Error::Protocol {
            protocol: "outer",
            message: "claim rows do not hold the claims and their filler",
        },
    )
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
    let first = layout.aux_first();
    let count = layout.total_rows() - first;
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

            rows[r - first][c] = wire[role as usize];
        }
    }

    let (tails, rest) = filler.split_at(count * tail);
    for (row, fill) in rows.iter_mut().zip(tails.chunks_exact(tail)) {
        row[layout.message_len..code_len].copy_from_slice(fill);
    }

    let (zero_sum_fill, rest) = rest.split_at(code_len - 1);

    let zero_sum = layout.linear_mask() - first;

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
        rows[row - first][..layout.message_len].copy_from_slice(fill);
    }

    Ok(rows)
}

pub fn aux_filler_len(layout: &OuterLayout, code_len: usize) -> usize {
    let count = layout.total_rows() - layout.aux_first();

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
    let list = pool.layout().list_bound();

    let bus_events = bus_draw_events(
        tables
            .clone()
            .map(|(statics, shape)| (statics.specs, 1u64 << shape.num_vars)),
    )
    .saturating_mul(list);

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

    let zerocheck_events = zerocheck_events.saturating_mul(list);
    let skip_events = skip_events.saturating_mul(list);
    let sumcheck_events = sumcheck_events.saturating_mul(list);

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
    let batch_events = (((2 * pool.num_sigma()).saturating_sub(1) + pool.rho_vars() + mix) as u128)
        .saturating_mul(list);
    let round_events = (EVAL_SUMCHECK_DEGREE as u128).saturating_mul(list);

    config.check_draw("bus draw", F::BITS, bus_events)?;
    config.check_draw("zerocheck draw", F::BITS, zerocheck_events)?;
    config.check_draw("sumcheck round", F::BITS, sumcheck_events)?;
    config.check_draw("eval batch draw", F::BITS, batch_events)?;
    config.check_draw("eval sumcheck round", F::BITS, round_events)?;

    let outer = match config.zero_knowledge {
        true => {
            let statement = OuterStatement::new::<F>(&shapes, pool.layout().num_vars());

            let geom = config.outer_geom(
                statement.masked_scalars,
                statement.claims,
                statement.mul_wires,
                F::BITS,
            )?;

            let bits = config
                .outer_security_bits(F::BITS, &geom)
                .min(config.outer_field_term_bits(F::BITS, &geom));

            match shapes.iter().any(|shape| shape.product_levels > 0) {
                true => {
                    let events = (PRODUCT_SUMCHECK_DEGREE as u128).saturating_mul(list);

                    config.check_draw("product sumcheck round", F::BITS, events)?;

                    Some(bits.min(config.draw_bits(F::BITS, events)))
                }
                false => Some(bits),
            }
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

fn message_rows<F: TowerField + HardwareField>(
    layout: &OuterLayout,
    oracle_rows: Range<usize>,
    values: &[Flat<F>],
    unknown: fn(u32) -> Unknown,
    code_len: usize,
    filler: &[Flat<F>],
) -> Option<Vec<Vec<Flat<F>>>> {
    let count = oracle_rows.len();
    let tail = code_len - layout.message_len;

    if values.len() > count * layout.message_len || filler.len() != count * tail {
        return None;
    }

    let mut rows = vec![vec![Flat::from_raw(F::ZERO); layout.domain_len]; count];
    for (i, value) in values.iter().enumerate() {
        let (r, c) = layout.slot(unknown(i as u32));
        rows[r - oracle_rows.start][c] = *value;
    }

    for (row, fill) in rows.iter_mut().zip(filler.chunks_exact(tail)) {
        row[layout.message_len..code_len].copy_from_slice(fill);
    }

    Some(rows)
}

/// Builds one table's product rows from its compiled predicate,
/// two operand rows per level and then every level's final row.
/// A table that wires no product gets its consistency row alone.
fn product_rows<F: TowerField + HardwareField>(
    record: &TableRecord<'_, F>,
    predicate: &PredicateRows<F>,
) -> errors::Result<Vec<AffineRow<F>>> {
    let levels = &record.products;

    if levels.len() != record.shape.product_levels {
        return Err(errors::Error::Protocol {
            protocol: "outer",
            message: "product reduction does not match the table shape",
        });
    }

    let zero = Flat::from_raw(F::ZERO);
    let one = Flat::from_raw(F::ONE);

    let mut tail = consistency_row(record.ast.roots.len(), &record.consistency);
    let mut below: Vec<(u32, Flat<F>)> = Vec::new();

    let weights = record.consistency.root_weights(predicate.roots.len());
    for (form, weight) in predicate.roots.iter().zip(weights) {
        split_products(&form.unknowns, weight, &mut tail.unknowns, &mut below);

        tail.constant += weight * form.constant;
    }

    if levels.is_empty() {
        return match below.is_empty() {
            true => Ok(vec![tail]),
            false => Err(stray_product()),
        };
    }

    let mut operands = Vec::with_capacity(3 * levels.len());
    let mut finals = Vec::with_capacity(levels.len());
    let mut initial = tail.unknowns;
    let mut slots: Vec<Option<usize>> = Vec::new();

    for (l, level) in levels.iter().enumerate() {
        let eq = TensorProduct::eq_table(&level.challenges);

        slots.clear();
        slots.resize(predicate.affine.len() / 2, None);

        for (j, entry) in level.entries.iter().enumerate() {
            let slot = slots.get_mut(entry.wire as usize);

            *slot.ok_or_else(stray_product)? = Some(j);
        }

        let mut weight = zero;
        for &(wire, coeff) in &below {
            let j = slots
                .get(wire as usize)
                .copied()
                .flatten()
                .ok_or_else(stray_product)?;

            weight += eq[j] * coeff;
        }

        finals.push(level.final_row(&initial, &level.rounds(), l as u32, weight));

        let next = match levels.get(l + 1) {
            Some(next) => Some(next.opening_sums()?),
            None => None,
        };

        let mut sides: [Vec<(u32, Flat<F>)>; 2] = [Vec::new(), Vec::new()];
        for (side, role) in [WireRole::Lhs, WireRole::Rhs].into_iter().enumerate() {
            let mul = l as u32;

            let mut row = AffineRow {
                unknowns: vec![(Unknown::Wire { mul, role }, one)],
                constant: zero,
            };

            for (entry, &e) in level.entries.iter().zip(&eq) {
                let operand = predicate
                    .affine
                    .get(2 * entry.wire as usize + side)
                    .ok_or_else(stray_product)?;

                // The product's own operand wire closes its row
                let (_, form) = operand.unknowns.split_last().ok_or_else(stray_product)?;

                split_products(form, e, &mut row.unknowns, &mut sides[side]);

                row.constant += e * operand.constant;
            }

            match (next, sides[side].is_empty()) {
                (Some(sums), _) => {
                    row.unknowns
                        .push((Unknown::Pad(sums.pad + side as u32), one));

                    row.constant += sums.masked[side];
                }
                (None, true) => {}
                (None, false) => return Err(stray_product()),
            }

            operands.push(row);
        }

        if let Some(sums) = next {
            let [lhs, rhs] = sides;

            initial = sums.initial().to_vec();
            below = lhs;

            below.extend(
                rhs.into_iter()
                    .map(|(wire, coeff)| (wire, coeff * sums.tau)),
            );
        }
    }

    operands.extend(finals);

    Ok(operands)
}

fn split_products<F: TowerField + HardwareField>(
    form: &[(Unknown, Flat<F>)],
    scale: Flat<F>,
    linear: &mut Vec<(Unknown, Flat<F>)>,
    products: &mut Vec<(u32, Flat<F>)>,
) {
    for &(unknown, coeff) in form {
        match unknown {
            Unknown::Wire { mul, .. } => products.push((mul, scale * coeff)),
            _ => linear.push((unknown, scale * coeff)),
        }
    }
}

fn stray_product() -> errors::Error {
    errors::Error::Protocol {
        protocol: "outer",
        message: "wired product sits outside the level that reads it",
    }
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
    let mut fixed = Vec::new();

    let mut first = 0;
    let mut mul_offset = 0u32;

    for record in records {
        let end = first + record.affine_rows();

        weigh_table(
            &mut acc,
            &mut sweep,
            &mut fixed,
            record,
            &scales[first..end],
            mul_offset,
        )?;

        first = end;
        mul_offset += record.shape.mul_wires() as u32;
    }

    let final_row = eval_final_row(&eval.mask_form, eval.claim_masked, eval.fin);
    acc.row(&final_row, scales[first], 0);

    first += 1;

    if let Some(gadget) = &eval.gadget {
        let chain = &scales[first..first + linearized::chain_rows::<F>()];
        for (row, &scale) in gadget.wire_rows().iter().zip(chain) {
            acc.row(row, scale, 0);
        }

        gadget.for_each_tie_weight(chain, |claim, weight| {
            acc.add(Unknown::Claim(claim), weight)
        })?;

        first += linearized::chain_rows::<F>();
    }

    for (endpoints, &scale) in buses.values().zip(&scales[first..]) {
        acc.row(&bus_sum_row(endpoints), scale, 0);
    }

    Ok(acc.batch())
}

fn weigh_table<F: TowerField + HardwareField>(
    acc: &mut Accumulator<'_, F>,
    sweep: &mut Sweep<F>,
    fixed: &mut Vec<Option<Flat<F>>>,
    record: &TableRecord<'_, F>,
    scales: &[Flat<F>],
    mul_offset: u32,
) -> errors::Result<()> {
    let levels = &record.products;
    let depth = levels.len();

    if depth != record.shape.product_levels {
        return Err(errors::Error::Protocol {
            protocol: "outer",
            message: "product reduction does not match the table shape",
        });
    }

    record.fixed_nodes(fixed)?;

    let zero = Flat::from_raw(F::ZERO);
    let one = Flat::from_raw(F::ONE);

    let (operands, tail) = scales.split_at(2 * depth);
    let (finals, pins) = tail.split_at(depth.max(1));

    let rounds: Vec<Vec<RoundForm<F>>> = levels.iter().map(LevelRecord::rounds).collect();
    let eqs: Vec<Vec<Flat<F>>> = levels
        .iter()
        .map(|level| TensorProduct::eq_table(&level.challenges))
        .collect();

    let lead = match rounds.first() {
        Some(first) => first.iter().fold(one, |acc, round| acc * round.weights[0]),
        None => one,
    };

    let root_scale = finals[0] * lead;

    let constant = sweep.roots(record, fixed, |claim, weight| {
        acc.add(Unknown::Claim(claim), weight * root_scale)
    })?;

    sweep.covered(
        levels
            .first()
            .map(|level| level.entries)
            .unwrap_or_default(),
    )?;

    let mut row = consistency_row(record.ast.roots.len(), &record.consistency);

    match levels.first() {
        None => row.constant += constant,
        Some(level) => {
            let weight = sweep.weight(level.entries, &eqs[0]);
            row = level.final_row(&row.unknowns, &rounds[0], 0, weight);
        }
    }

    acc.row(&row, finals[0], mul_offset);

    for (l, level) in levels.iter().enumerate() {
        let next = match levels.get(l + 1) {
            Some(next) => Some((next, next.opening_sums()?)),
            None => None,
        };

        let listed = next.map(|(next, _)| next.entries).unwrap_or_default();

        let mut below = [zero; 2];
        for (side, role) in [WireRole::Lhs, WireRole::Rhs].into_iter().enumerate() {
            let scale = operands[2 * l + side];

            let constant =
                sweep.side(record, fixed, level.entries, &eqs[l], side, |claim, w| {
                    acc.add(Unknown::Claim(claim), w * scale)
                })?;

            sweep.covered(listed)?;

            let mul = mul_offset + l as u32;

            acc.add(Unknown::Wire { mul, role }, scale);

            acc.target += scale * constant;

            if let Some((next, sums)) = next {
                acc.add(Unknown::Pad(sums.pad + side as u32), scale);

                acc.target += scale * sums.masked[side];
                below[side] = sweep.weight(next.entries, &eqs[l + 1]);
            }
        }

        if let Some((next, sums)) = next {
            let weight = below[0] + sums.tau * below[1];
            let row = next.final_row(&sums.initial(), &rounds[l + 1], (l + 1) as u32, weight);

            acc.row(&row, finals[l + 1], mul_offset);
        }
    }

    let mut next = 0;
    record.for_each_tail_row(|row| {
        acc.row(&row, pins[next], mul_offset);

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
        acc.row(row, scale, 0);
    }

    for &(first, gadget) in &rows.gadgets {
        let chain = &scales[first..first + linearized::chain_rows::<F>()];

        gadget.for_each_tie_weight(chain, |claim, weight| {
            acc.add(Unknown::Claim(claim), weight)
        })?;
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
    claims_first: u32,
    planes: &[Flat<F>],
    f: &mut impl FnMut(AffineRow<F>),
) -> Form<F> {
    let one = Flat::from_raw(F::ONE);
    let g = h_form(column, planes);
    let d1 = denominator_form(&column.first, claims_first);
    let wire = column.first_wire;

    f(operand_row(&g, wire, WireRole::Lhs));
    f(operand_row(&d1, wire, WireRole::Rhs));

    let mut c = product_form(wire);
    c.extend(&selector_form(&column.first, claims_first));

    let Some(second) = &column.second else {
        return c;
    };

    let d2 = denominator_form(second, claims_first);

    f(operand_row(&c, wire + 1, WireRole::Lhs));
    f(operand_row(&d2, wire + 1, WireRole::Rhs));

    let mut c = product_form(wire + 1);

    match second.selector_claimed() {
        true => {
            let s2 = selector_form(second, claims_first);

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
    claims_first: u32,
) -> Form<F> {
    let mut form = Form {
        unknowns: Vec::with_capacity(fraction.slots.len()),
        constant: fraction.constant,
    };

    for &(claim, coeff) in &fraction.slots {
        form.unknowns
            .push((Unknown::Claim(claims_first + claim), coeff));
    }

    form
}

fn selector_form<F: TowerField + HardwareField>(
    fraction: &BusFraction<F>,
    claims_first: u32,
) -> Form<F> {
    let one = Flat::from_raw(F::ONE);
    let mut form = Form::default();

    match fraction.selector {
        Some(s) => form.unknowns.push((Unknown::Claim(claims_first + s), one)),
        None => form.constant = one,
    }

    if let Some(s) = fraction.recv_selector {
        form.unknowns.push((Unknown::Claim(claims_first + s), one));
    }

    form
}

fn product_form<F: TowerField + HardwareField>(wire: u32) -> Form<F> {
    Form {
        unknowns: vec![(product_wire(wire), Flat::from_raw(F::ONE))],
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

    row.constant += form.constant;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProgramCell;
    use crate::constraint::ConstraintArena;
    use crate::permutation::Source;
    use crate::predicate::wire_values;
    use core::ops::Range;
    use hekate_core::config;
    use hekate_core::trace::ColumnType;
    use hekate_math::{Block128, Block256};

    type F = Block256;

    const BUS_H_PAD: u32 = 30;
    const BUS_PUBLIC: u128 = 77;

    /// `half - h_columns` at `bus_record`'s three
    /// columns and one bus column, any plane count.
    const H_CLAIM: usize = 3;

    /// `(base, columns, buses, fixed)` per table of `weighted_statement`.
    const WEIGHTED_TABLES: [(u32, u32, u32, bool); 3] =
        [(0, 4, 2, true), (600, 3, 1, false), (1200, 2, 0, true)];

    fn shape(num_vars: usize, num_columns: usize, degree: usize, buses: usize) -> TableShape {
        TableShape {
            num_vars,
            num_columns,
            sumcheck_degree: degree,
            bus_columns: buses,
            h_planes: h_planes::<F>(),
            bus_wires: buses,
            product_levels: 0,
            product_rounds: 0,
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

    fn holds(
        row: &AffineRow<F>,
        claims: &[Flat<F>],
        pad: &[Flat<F>],
        wires: &[[Flat<F>; 3]],
    ) -> bool {
        let form = Form {
            unknowns: row.unknowns.clone(),
            constant: row.constant,
        };

        form.evaluate(claims, pad, wires) == Flat::from_raw(F::ZERO)
    }

    fn mix(seed: u128) -> Flat<F> {
        let lo = seed
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(0x51ed_2701);

        F::new(Block128(lo), Block128(!lo)).to_hardware()
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

    fn h_pads() -> Vec<u32> {
        (0..h_planes::<F>() as u32).map(|j| BUS_H_PAD + j).collect()
    }

    /// Plane values `h_j` at full width, `h = Σ_j h_j · basis_j`,
    /// and each plane masked by its pad at `BUS_H_PAD + j`.
    fn h_split(seed: u128, pad: &[Flat<F>]) -> (Flat<F>, Vec<Flat<F>>, Vec<Flat<F>>) {
        let basis = plane_basis::<F>().unwrap();
        let parts: Vec<Flat<F>> = (0..basis.len() as u128).map(|j| mix(seed + j)).collect();

        let h = parts
            .iter()
            .zip(&basis)
            .fold(Flat::from_raw(F::ZERO), |acc, (&part, &b)| acc + part * b);

        let h_masked = parts
            .iter()
            .zip(h_pads())
            .map(|(&part, p)| part + pad[p as usize])
            .collect();

        (h, parts, h_masked)
    }

    /// One table over [`bus_ast`]: `c0 · c1` as the only
    /// constraint, three columns, one Permutation bus column
    /// keying `c0` against a public constant, selector `c1`.
    fn bus_record<'s>(
        ast: &'s ConstraintAst<F>,
        levels: &'s [Vec<LevelEntry>],
        half: usize,
        h_masked: Vec<Flat<F>>,
    ) -> TableRecord<'s, F> {
        let products = level_records(levels, 500, 40);

        let column = BusColumnRow {
            h_pads: h_pads(),
            h_masked,
            first: bus_fraction(3, Some(1)),
            second: None,
            eq_lookup: Flat::from_raw(F::ONE),
            first_wire: products.len() as u32,
        };

        let shape = with_products(
            TableShape {
                next_claims: half,
                ..shape(1, 3, 3, 1)
            },
            &products,
        );

        TableRecord {
            ast,
            products,
            consistency: ConsistencyInputs {
                claims_first: 0,
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
            shape,
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

    fn wired(ast: &ConstraintAst<F>, fixed_columns: &[usize]) -> usize {
        mul_node_count(ast, &fixed_only(ast, fixed_columns.iter().copied()))
    }

    fn levels_of(ast: &ConstraintAst<F>, fixed_columns: &[usize]) -> Vec<Vec<LevelEntry>> {
        product_levels(ast, &fixed_only(ast, fixed_columns.iter().copied()))
    }

    /// Level records of `levels` at mixed challenges and
    /// sums, their pads consecutive from `first_pad`.
    fn level_records(
        levels: &[Vec<LevelEntry>],
        seed: u128,
        first_pad: u32,
    ) -> Vec<LevelRecord<'_, F>> {
        let mut pad = first_pad;

        levels
            .iter()
            .enumerate()
            .map(|(l, entries)| {
                let s = seed + 1000 * l as u128;

                let sums = (l > 0).then(|| {
                    let first = pad;

                    pad += 2;

                    LevelSums {
                        pad: first,
                        masked: [mix(s + 1), mix(s + 2)],
                        tau: mix(s + 3),
                    }
                });

                let rounds = level_rounds(entries.len());
                let round_first = pad;

                pad += (PRODUCT_SUMCHECK_DEGREE * rounds) as u32;

                LevelRecord {
                    entries,
                    challenges: (0..rounds as u128).map(|i| mix(s + 10 + i)).collect(),
                    round_first,
                    claim_masked: mix(s + 4),
                    sums,
                }
            })
            .collect()
    }

    fn with_products(shape: TableShape, products: &[LevelRecord<'_, F>]) -> TableShape {
        TableShape {
            product_levels: products.len(),
            product_rounds: products.iter().map(|level| level.challenges.len()).sum(),
            ..shape
        }
    }

    fn layout_of(statement: &OuterStatement) -> OuterLayout {
        let geom = Config::prod()
            .outer_geom(
                statement.masked_scalars,
                statement.claims,
                statement.mul_wires,
                256,
            )
            .unwrap();

        OuterLayout::new(&geom, statement).unwrap()
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

    fn weighted_levels(ast: &ConstraintAst<F>, fixed: bool) -> Vec<Vec<LevelEntry>> {
        match fixed {
            true => levels_of(ast, &[0, 1]),
            false => levels_of(ast, &[]),
        }
    }

    fn weighted_record<'s>(
        ast: &'s ConstraintAst<F>,
        levels: &'s [Vec<LevelEntry>],
        base: u32,
        columns: u32,
        buses: u32,
        fixed: bool,
    ) -> TableRecord<'s, F> {
        let seed = base as u128;
        let planes = h_planes::<F>() as u32;
        let half = columns + planes * buses;

        let free = base + 2 * half;
        let sums_pad = free + planes * buses;
        let mask_pad = sums_pad + buses;

        let products = level_records(levels, seed + 300, mask_pad + 3);

        let wires = |k: u32| match k {
            1 => 3,
            _ => 1,
        };

        let mut first_wire = products.len() as u32;
        let mut bus_rows = Vec::with_capacity(buses as usize);

        for k in 0..buses {
            let s = seed + 100 * k as u128;

            bus_rows.push(BusColumnRow {
                h_pads: (0..planes).map(|j| free + planes * k + j).collect(),
                h_masked: (0..planes).map(|j| mix(s + 10 + 7 * j as u128)).collect(),
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

        let shape = with_products(
            TableShape {
                bus_wires: (0..buses).map(wires).sum(),
                next_claims: half as usize,
                ..shape(3, columns as usize, 3, buses as usize)
            },
            &products,
        );

        TableRecord {
            ast,
            products,
            consistency: ConsistencyInputs {
                claims_first: base,
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
                true => vec![(0, mix(seed + 70)), (1, mix(seed + 71))],
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
            shape,
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

    fn statement_levels(asts: &[ConstraintAst<F>; 3]) -> Vec<Vec<Vec<LevelEntry>>> {
        asts.iter()
            .zip(WEIGHTED_TABLES)
            .map(|(ast, (.., fixed))| weighted_levels(ast, fixed))
            .collect()
    }

    /// Three tables over `asts`: the first carries every
    /// row class, the Permutation group spans two tables.
    fn weighted_statement<'s>(
        asts: &'s [ConstraintAst<F>; 3],
        levels: &'s [Vec<Vec<LevelEntry>>],
    ) -> (
        Vec<TableRecord<'s, F>>,
        EvalRecord<F>,
        OuterStatement,
        OuterLayout,
    ) {
        let records: Vec<_> = asts
            .iter()
            .zip(levels)
            .zip(WEIGHTED_TABLES)
            .map(|((ast, levels), (base, columns, buses, fixed))| {
                weighted_record(ast, levels, base, columns, buses, fixed)
            })
            .collect();

        let table_wires: usize = records.iter().map(|r| r.shape.mul_wires()).sum();
        let eval = weighted_eval(1800, table_wires as u32);

        let statement = OuterStatement {
            masked_scalars: 2048,
            claims: 2048,
            mul_wires: table_wires + F::BITS - 1,
        };

        let layout = layout_of(&statement);

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
                ("level operands", 2 * record.shape.product_levels),
                ("level finals", record.shape.product_levels.max(1)),
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
        assert_eq!(with.mul_wires - without.mul_wires, 255);
    }

    #[test]
    fn ring_blind_planes_are_claimed_only_under_blinding() {
        let plain = shape(12, 100, 3, 0);
        let ring = TableShape {
            expands_bits: true,
            ..plain
        };

        assert_eq!(ring.claims(), plain.claims());

        for whole_blinds in [1, 4] {
            let blinded = |s: TableShape| every_half(TableShape { whole_blinds, ..s });

            assert_eq!(
                blinded(ring).claims() - blinded(plain).claims(),
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

        assert_eq!(wired(&ast, &[]), 0);
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

        assert_eq!(wired(&ast, &[]), 1);
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

        assert_eq!(wired(&ast, &[]), 1);
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

        assert_eq!(wired(&ast, &[]), 2);
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

        assert_eq!(wired(&ast, &[]), 1);
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

        let planes = h_planes::<F>();
        let claimed_sum_and_h_evals = 1 + planes;

        let h_claim_halves = 2 * planes;
        let h_blind_halves = 2 * with_bus.whole_blinds;

        assert_eq!(
            with_bus.masked_scalars() - without.masked_scalars(),
            claimed_sum_and_h_evals
        );
        assert_eq!(
            with_bus.claims() - without.claims(),
            h_claim_halves + h_blind_halves
        );
    }

    #[test]
    fn masked_scalars_match_transcript_tally() {
        let planes = h_planes::<F>();

        for whole_blinds in [1, 8] {
            let s = every_half(TableShape {
                whole_blinds,
                ..shape(12, 1977, 5, 3)
            });

            let claimed_sums_and_h_evals = 3 + 3 * planes;
            let skip_round = (5 << 6) - 1;
            let zerocheck_rounds = skip_round + (12 - 6) * 5;
            let h_blinds = whole_blinds;
            let claims = 2 * (1977 + whole_blinds + 3 * planes + h_blinds);

            assert_eq!(
                s.masked_scalars(),
                claimed_sums_and_h_evals + zerocheck_rounds
            );
            assert_eq!(s.claims(), claims);
        }
    }

    #[test]
    fn every_bus_column_adds_one_wire() {
        let s = TableShape {
            product_levels: 40,
            ..shape(12, 100, 3, 3)
        };

        assert_eq!(s.mul_wires(), 40 + 3);
    }

    #[test]
    fn statement_sums_over_every_table() {
        let main = TableShape {
            product_levels: 40,
            whole_blinds: 2,
            ..shape(12, 100, 3, 2)
        };

        let chiplet = TableShape {
            product_levels: 7,
            whole_blinds: 8,
            ..shape(10, 30, 4, 1)
        };

        let statement = OuterStatement::new::<F>(&[main, chiplet], 12);

        assert_eq!(
            statement.masked_scalars,
            main.masked_scalars() + chiplet.masked_scalars() + 2 * 12
        );
        assert_eq!(statement.claims, main.claims() + chiplet.claims());
        assert_eq!(statement.mul_wires, 40 + 7 + 3);
    }

    #[test]
    fn every_hadamard_wire_operand_is_pinned_by_row() {
        let planes = h_planes::<F>();
        let h_masked = (0..planes as u128).map(|j| mix(8 + 50 * j)).collect();
        let ast = bus_ast();
        let levels = levels_of(&ast, &[]);
        let record = bus_record(&ast, &levels, 3 + planes, h_masked);

        let statement = OuterStatement {
            masked_scalars: 32,
            claims: record.shape.claims(),
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

        let (h, _, h_masked) = h_split(1400, &pad);

        for second in [
            None,
            Some(bus_fraction(5, None)),
            Some(bus_fraction(5, Some(1))),
        ] {
            let column = BusColumnRow {
                h_pads: h_pads(),
                h_masked: h_masked.clone(),
                first: bus_fraction(3, Some(1)),
                second,
                eq_lookup: value(500),
                first_wire: 0,
            };

            let mut wires = Vec::new();
            bus_wire_values(&column.first, column.second.as_ref(), &plain, h, &mut wires);

            let mut rows = Vec::new();
            let consistency = column_terms(&column, 0, &planes, &mut |row| rows.push(row));

            assert_eq!(rows.len(), 2 * wires.len());
            assert!(rows.iter().all(|row| holds(row, &plain, &pad, &wires)));

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

            assert_eq!(consistency.evaluate(&plain, &pad, &wires), expected);

            let bus_sum = h_form(&column, &planes)
                .scaled(column.eq_lookup)
                .evaluate(&plain, &pad, &wires);

            assert_eq!(bus_sum, column.eq_lookup * h);
        }
    }

    #[test]
    fn bus_column_rows_reject_free_operand() {
        let planes = plane_basis::<F>().unwrap();
        let plain: Vec<Flat<F>> = (0..8).map(|i| mix(200 + i)).collect();
        let pad: Vec<Flat<F>> = (0..32).map(|i| mix(300 + i)).collect();

        let (h, _, h_masked) = h_split(1400, &pad);
        let column = BusColumnRow {
            h_pads: h_pads(),
            h_masked,
            first: bus_fraction(3, Some(1)),
            second: None,
            eq_lookup: Flat::from_raw(F::ONE),
            first_wire: 0,
        };

        let mut wires = Vec::new();
        bus_wire_values(&column.first, column.second.as_ref(), &plain, h, &mut wires);

        let mut rows = Vec::new();
        column_terms(&column, 0, &planes, &mut |row| rows.push(row));

        let one = Flat::from_raw(F::ONE);

        let mut forged = wires.clone();
        forged[0][0] = one;
        forged[0][2] = one * forged[0][1];

        assert!(!holds(&rows[0], &plain, &pad, &forged));
        assert!(holds(&rows[1], &plain, &pad, &forged));
    }

    #[test]
    fn layout_holds_whole_statement() {
        // The least statement, then two whose claims and wires span several rows
        for (scalars, claims, wires) in [
            (1usize, 1usize, 0usize),
            (75, 4_000, 2_384),
            (2_000, 10_000, 20_000),
        ] {
            let layout = layout_of(&OuterStatement {
                masked_scalars: scalars,
                claims,
                mul_wires: wires,
            });

            assert!(layout.pad_rows * layout.message_len >= scalars);
            assert!(layout.claim_rows * layout.message_len >= claims);
            assert!(layout.aux_rows * layout.message_len >= wires);

            let (r, _) = layout.slot(Unknown::Pad(scalars as u32 - 1));
            assert!(r < layout.pad_rows);

            let (r, _) = layout.slot(Unknown::Claim(claims as u32 - 1));
            assert!((layout.pad_rows..layout.aux_first()).contains(&r));

            if wires > 0 {
                let (r, _) = layout.slot(Unknown::Wire {
                    mul: wires as u32 - 1,
                    role: WireRole::Product,
                });

                assert!((layout.aux_first()..layout.interleaved_mask()).contains(&r));
            }

            assert_eq!(layout.total_rows(), layout.quadratic_mask_hi() + 1);
        }
    }

    #[test]
    fn every_mask_row_interleaved_test_reveals_is_uniform() {
        let statement = OuterStatement {
            masked_scalars: 75,
            claims: 4_000,
            mul_wires: 2_384,
        };

        let wires = statement.mul_wires;

        let geom = Config::prod()
            .outer_geom(statement.masked_scalars, statement.claims, wires, 256)
            .unwrap();

        let layout = OuterLayout::new(&geom, &statement).unwrap();

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
                rows[mask - layout.aux_first()][..layout.message_len]
                    .iter()
                    .all(|v| *v != zero)
            );
        }

        let quadratic = layout.quadratic_mask() - layout.aux_first();
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
        let zero = Flat::from_raw(F::ZERO);
        let one = Flat::from_raw(F::ONE);

        let (_, parts, h_masked) = h_split(400, &pad);
        let pads = h_pads();

        for whole_blinds in [0, 1] {
            let half = 3 + parts.len() + 2 * whole_blinds;
            let first = H_CLAIM + whole_blinds;

            for (offset, honest) in [(zero, true), (one, false)] {
                let mut plain: Vec<Flat<F>> = (0..2 * half).map(|i| mix(200 + i as u128)).collect();
                for (j, &part) in parts.iter().enumerate() {
                    plain[first + j] = part + offset;
                }

                let ast = bus_ast();
                let levels = levels_of(&ast, &[]);

                let mut record = bus_record(&ast, &levels, half, h_masked.clone());
                record.shape.whole_blinds = whole_blinds;

                let statement = OuterStatement {
                    masked_scalars: 32,
                    claims: record.shape.claims(),
                    mul_wires: record.shape.mul_wires(),
                };

                let records = [record];
                let eval = eval_without_gadget();
                let rows = assemble(&records, &eval, &statement).unwrap();

                for (j, (&h_pad, &plane_masked)) in pads.iter().zip(&h_masked).enumerate() {
                    let claim = first + j;
                    let pin_form = [
                        (Unknown::Pad(h_pad), one),
                        (Unknown::Claim(claim as u32), one),
                    ];

                    let pins: Vec<&AffineRow<F>> = rows
                        .affine
                        .iter()
                        .filter(|r| r.unknowns == pin_form)
                        .collect();

                    assert_eq!(pins.len(), 1, "{whole_blinds} whole blinds, plane {j}");
                    assert_eq!(pins[0].constant, plane_masked);

                    let left = pad[h_pad as usize] + plain[claim];

                    assert_eq!(left == pins[0].constant, honest);
                }
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
        let levels = statement_levels(&asts);
        let (records, eval, statement, layout) = weighted_statement(&asts, &levels);

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
        let levels = statement_levels(&asts);
        let (records, eval, statement, layout) = weighted_statement(&asts, &levels);

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
    fn miscounted_product_levels_are_rejected() {
        let ast = rich_ast();
        let levels = weighted_levels(&ast, true);
        let message = "product reduction does not match the table shape";

        for delta in [-1, 1] {
            let mut record = weighted_record(&ast, &levels, 0, 4, 2, true);

            assert!(record.shape.product_levels > 0);

            record.shape.product_levels = record.shape.product_levels.saturating_add_signed(delta);

            let statement = OuterStatement {
                masked_scalars: 2048,
                claims: 2048,
                mul_wires: record.shape.mul_wires(),
            };

            let layout = layout_of(&statement);

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
        let levels = statement_levels(&asts);
        let (records, eval, statement, layout) = weighted_statement(&asts, &levels);

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
                252,
                "bus draw",
                251,
            ),
            (
                statics(&roots, &[]),
                shape(4, 3, 2, 0),
                &empty,
                254,
                "zerocheck draw",
                253,
            ),
            (
                statics(&none, &[]),
                shape(1, 1, 9, 0),
                &empty,
                253,
                "sumcheck round",
                252,
            ),
            (
                statics(&none, &[]),
                TableShape {
                    subfield_units: true,
                    ..shape(1, 1, 1, 0)
                },
                &empty,
                256,
                "sumcheck round",
                255,
            ),
            (
                statics(&none, &[]),
                shape(1, 1, 1, 0),
                &packed,
                255,
                "eval batch draw",
                254,
            ),
            (
                statics(&none, &[]),
                shape(1, 1, 1, 0),
                &narrow,
                253,
                "eval batch draw",
                252,
            ),
            (
                statics(&none, &[]),
                shape(1, 1, 1, 0),
                &empty,
                256,
                "eval sumcheck round",
                255,
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
                bus: 251,
                zerocheck: 252,
                skip: None,
                sumcheck: 253,
                eval_batch: 256,
                eval_round: 255,
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
    fn next_claims_enter_claims_once() {
        let current_only = shape(12, 100, 3, 1);
        let shipped = TableShape {
            next_claims: 7,
            ..current_only
        };

        assert_eq!(shipped.claims() - current_only.claims(), 7);
        assert_eq!(shipped.masked_scalars(), current_only.masked_scalars());
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

    #[test]
    fn johnson_pool_charges_list_bound_on_every_draw() {
        let roots = rich_ast();
        let specs = one_bus();
        let table = [(statics(&roots, &specs), shape(10, 3, 3, 1))];

        let plan = RingSwitchPlan::new(&[ColumnType::B128], None, 0, 0, &[true]).unwrap();
        let pool_at = |num_queries| {
            let config = Config {
                num_queries,
                ..raw_at(config::MIN_PRODUCTION_BITS)
            };

            FoldPool::new(&[(&plan, 10)], &config).unwrap()
        };

        let unique = pool_at(309);
        let johnson = pool_at(262);

        assert_eq!(
            (unique.layout().list_bound(), johnson.layout().list_bound()),
            (1, 92)
        );

        let unique = check_draws(&raw_at(0), &unique, table).unwrap();
        let johnson = check_draws(&raw_at(0), &johnson, table).unwrap();

        for (unique_bits, johnson_bits) in [
            (unique.bus, johnson.bus),
            (unique.zerocheck, johnson.zerocheck),
            (unique.skip.unwrap(), johnson.skip.unwrap()),
            (unique.sumcheck, johnson.sumcheck),
            (unique.eval_batch, johnson.eval_batch),
            (unique.eval_round, johnson.eval_round),
        ] {
            assert!(
                matches!(unique_bits - johnson_bits, 6 | 7),
                "{unique_bits} against {johnson_bits}"
            );
        }
    }

    #[test]
    fn layout_of_chosen_geometry_carries_fewest_elements() {
        let cfg = Config::prod();
        let queries = cfg.outer_queries;

        let elements = |geom: &OuterGeometry, statement: &OuterStatement| {
            let layout = OuterLayout::new(geom, statement).unwrap();
            let responses = geom.code_len + 2 * (2 * geom.code_len);

            layout.total_rows() * queries + responses
        };

        // Seven statements proven at 0.39 (fibonacci, AES-128, Keccak, SHA-256,
        // ML-DSA-65, RSA-2048, ML-KEM-768) as masked scalars without their claims,
        // claims and wires, each with the code length and rows the rule gives it.
        for (masked_scalars, claims, mul_wires, code_len, rows) in [
            (466usize, 521usize, 289usize, 512usize, 12usize),
            (1_400, 952, 743, 512, 21),
            (63, 3_581, 3_458, 1_024, 23),
            (151, 4_498, 1_337, 512, 31),
            (1_739, 10_960, 8_801, 1_024, 54),
            (236, 18_414, 13_297, 2_048, 40),
            (1_279, 17_719, 12_823, 2_048, 37),
        ] {
            let statement = OuterStatement {
                masked_scalars,
                claims,
                mul_wires,
            };

            let chosen = cfg
                .outer_geom(masked_scalars, claims, mul_wires, 256)
                .unwrap();

            let layout = OuterLayout::new(&chosen, &statement).unwrap();

            assert_eq!((chosen.code_len, layout.total_rows()), (code_len, rows));

            let fewest = elements(&chosen, &statement);

            let mut other_len = (queries + 2).next_power_of_two();
            while other_len <= 1 << 20 {
                let other = OuterGeometry {
                    message_len: other_len - queries - 1,
                    code_len: other_len,
                    domain_len: other_len << config::OUTER_RATE_LOG2,
                    queries,
                };

                assert!(
                    elements(&other, &statement) >= fewest,
                    "{statement:?} {other_len}"
                );

                other_len *= 2;
            }
        }
    }

    #[test]
    fn reverse_pass_folds_fixed_operand_on_either_side() {
        let mut arena = ConstraintArena::<F>::new();

        let gate = arena.cell(ProgramCell::current(0));
        let other = arena.cell(ProgramCell::current(1));
        let x = arena.cell(ProgramCell::current(2));
        let y = arena.cell(ProgramCell::current(3));
        let gate_next = arena.cell(ProgramCell::next(0));

        let body = arena.add(x, y);
        let left = arena.mul(gate, body);
        let right = arena.mul(body, other);
        let both = arena.mul(gate, other);
        let plain = arena.mul(x, y);
        let next_read = arena.mul(gate_next, x);
        let under = arena.mul(left, y);

        let ast = ConstraintAst {
            arena,
            roots: vec![left, right, both, plain, next_read, under],
            labels: vec![None; 6],
        };

        let levels = weighted_levels(&ast, true);
        let record = weighted_record(&ast, &levels, 0, 4, 2, true);

        assert_eq!(wired(&ast, &[0, 1]), 3);
        assert_eq!(record.shape.product_levels, 1);

        let statement = OuterStatement {
            masked_scalars: 2048,
            claims: 2048,
            mul_wires: record.shape.mul_wires(),
        };

        let layout = layout_of(&statement);

        let records = [record];
        let eval = eval_without_gadget();

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
    fn product_rows_hold_on_honest_reduction() {
        let ast = rich_ast();
        let zero = Flat::from_raw(F::ZERO);
        let half = 3;
        let claims: Vec<Flat<F>> = (0..2 * half as u128).map(value).collect();

        let mut node_values = Vec::new();
        ast.evaluate_into(
            &ast.precompute_hardware_consts(),
            &claims[..half],
            &claims[half..],
            &mut node_values,
        );

        let marks = fixed_only(&ast, core::iter::empty());
        let factors = wire_values(&ast, &marks, &node_values);
        let levels = product_levels(&ast, &marks);

        assert!(levels.len() > 1);

        let mut record = TableRecord {
            ast: &ast,
            consistency: ConsistencyInputs {
                claims_first: 0,
                eq_zc: mix(1),
                alpha: mix(2),
                planes: plane_basis().unwrap(),
                boundary: Vec::new(),
                telescope: Vec::new(),
                buses: Vec::new(),
                val_final_form: Vec::new(),
                val_final_masked: zero,
            },
            products: Vec::new(),
            fixed: Vec::new(),
            claimed_sums: Vec::new(),
            bindings: Vec::new(),
            next_slots: (0..half).map(Some).collect(),
            shape: TableShape {
                next_claims: half,
                product_levels: levels.len(),
                product_rounds: levels.iter().map(|level| level_rounds(level.len())).sum(),
                ..shape(1, 3, 3, 0)
            },
        };

        record.consistency.val_final_masked = ast
            .roots
            .iter()
            .zip(record.consistency.root_weights(ast.roots.len()))
            .fold(zero, |acc, (root, weight)| {
                acc + weight * node_values[root.0 as usize]
            });

        let pad: Vec<Flat<F>> = (0..64).map(|i| mix(4000 + i)).collect();

        let mut next_pad = 0;
        let mut wires = Vec::new();
        let mut products = Vec::new();

        {
            let mut weights = ProductWeights::new(&record).unwrap();
            let (mut omega, mut masked) = weights.first(&levels[0]).unwrap();
            let mut opening = None;

            for (l, entries) in levels.iter().enumerate() {
                let rounds = level_rounds(entries.len());

                let mut columns = [(); 3].map(|_| vec![zero; 1 << rounds]);
                for (j, (entry, &weight)) in entries.iter().zip(&omega).enumerate() {
                    let wire = factors[entry.wire as usize];

                    columns[0][j] = weight;
                    columns[1][j] = wire[0];
                    columns[2][j] = wire[1];
                }

                let truth = (0..1 << rounds).fold(zero, |acc, j| {
                    acc + columns[0][j] * columns[1][j] * columns[2][j]
                });

                let round_first = next_pad as u32;
                let challenges: Vec<Flat<F>> = (0..rounds as u128)
                    .map(|i| mix(5000 + 100 * l as u128 + i))
                    .collect();

                let mut delta = masked + truth;
                for &r in &challenges {
                    let w = lagrange_weights::<F>(PRODUCT_SUMCHECK_DEGREE, r);
                    let p = &pad[next_pad..next_pad + PRODUCT_SUMCHECK_DEGREE];

                    delta = w[0] * delta + (w[0] + w[1]) * p[0] + w[2] * p[1] + w[3] * p[2];
                    next_pad += PRODUCT_SUMCHECK_DEGREE;

                    for column in &mut columns {
                        *column = column
                            .chunks(2)
                            .map(|pair| pair[0] + r * (pair[0] + pair[1]))
                            .collect();
                    }
                }

                let [weight, lhs, rhs] = [columns[0][0], columns[1][0], columns[2][0]];

                wires.push([lhs, rhs, lhs * rhs]);

                let eq = TensorProduct::eq_table(&challenges);
                let sums = opening.take();

                if let Some(next) = levels.get(l + 1) {
                    let below = weights.below(entries, &eq, next).unwrap();
                    let first = next_pad as u32;

                    let mut sent = [zero; 2];
                    for (side, weights) in below.iter().enumerate() {
                        let sum = next.iter().zip(weights).fold(zero, |acc, (entry, &w)| {
                            acc + w * factors[entry.wire as usize][2]
                        });

                        sent[side] = sum + pad[next_pad];
                        next_pad += 1;
                    }

                    let tau = mix(6000 + l as u128);

                    omega = below[0]
                        .iter()
                        .zip(&below[1])
                        .map(|(&a, &b)| a + tau * b)
                        .collect();

                    masked = sent[0] + tau * sent[1];

                    opening = Some(LevelSums {
                        pad: first,
                        masked: sent,
                        tau,
                    });
                }

                products.push(LevelRecord {
                    entries,
                    challenges,
                    round_first,
                    claim_masked: weight * lhs * rhs + delta,
                    sums,
                });
            }
        }

        record.products = products;

        let mut fixed = Vec::new();
        record.fixed_nodes(&mut fixed).unwrap();

        let predicate = compile(&ast, record.claim_layout(), &fixed).unwrap();
        let rows = product_rows(&record, &predicate).unwrap();

        assert_eq!(rows.len(), 3 * levels.len());

        for (i, row) in rows.iter().enumerate() {
            assert!(holds(row, &claims, &pad, &wires), "row {i}");
        }
    }

    #[test]
    fn unlisted_product_is_rejected_on_both_paths() {
        let ast = rich_ast();
        let levels = weighted_levels(&ast, true);
        let message = "wired product sits outside the level that reads it";

        assert!(!levels.is_empty());

        for cut in 0..levels.len() {
            let mut record = weighted_record(&ast, &levels, 0, 4, 2, true);
            record.products[cut].entries = &levels[cut][1..];

            let statement = OuterStatement {
                masked_scalars: 2048,
                claims: 2048,
                mul_wires: record.shape.mul_wires(),
            };

            let layout = layout_of(&statement);

            let records = [record];
            let eval = eval_without_gadget();
            let rows = statement_rows(&records, &eval, &statement).unwrap();
            let tensor: Vec<F> = (0..linear_tensor_vars(rows) as u128)
                .map(|i| mix(8000 + i).to_tower())
                .collect();

            let fast = linear_weights(&layout, &statement, &records, &eval, &tensor).map(|_| ());
            let reference = assemble(&records, &eval, &statement).map(|_| ());

            for got in [fast, reference] {
                match got {
                    Err(errors::Error::Protocol { message: got, .. }) => assert_eq!(got, message),
                    _ => panic!("{message}"),
                }
            }
        }
    }
}
