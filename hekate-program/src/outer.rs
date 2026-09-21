// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use crate::chiplet::ChipletDef;
use crate::constraint::{ConstraintAst, ConstraintExpr, ExprId};
use crate::expander::{RING_BLIND_BITS, RingSwitchPlan, claim_weights, eq_tensor_b};
use crate::linearized::{self, RingGadget, linearized_coeffs};
use crate::permutation::{BusKind, Source, eval_row_idx_byte_mle, eval_row_idx_le_mle};
use crate::predicate::{AffineRow, ClaimLayout, Form, PredicateRows, Unknown, WireRole, compile};
use crate::{Air, ProgramInstance};
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::mem;
use hekate_core::errors;
use hekate_core::outer::{OUTER_MASK_ROWS, OuterGeometry};
use hekate_core::poly::univariate::{MAX_POINTS, UnivariatePoly};
use hekate_core::tensor::TensorProduct;
use hekate_math::{Block128, Flat, HardwareField, TowerField};

pub struct LinearBatch<F> {
    pub weights: Vec<Vec<Flat<F>>>,
    pub rows: Vec<usize>,
    pub target: Flat<F>,
}

pub enum BusSource<F> {
    Claim(u32),
    Public(Flat<F>),
}

pub struct BusRow<F> {
    pub h_pad: u32,
    pub h_masked: Flat<F>,
    pub h_wire: u32,
    pub sources: Vec<BusSource<F>>,
    pub selector: Option<u32>,
    pub recv_selector: Option<u32>,
    pub eq_lookup: Flat<F>,
}

pub struct ConsistencyInputs<F> {
    /// Pad index of claim 0, matching `ClaimLayout`.
    pub pad_first: u32,

    pub eq_zc: Flat<F>,
    pub alpha: Flat<F>,
    pub gamma: Flat<F>,
    pub beta: Flat<F>,

    /// `(claim index, public value, eq(r_final, row))`.
    pub boundary: Vec<(u32, Flat<F>, Flat<F>)>,

    /// `(b_k claim index, b_k_next claim index)`.
    pub telescope: Vec<(u32, u32)>,

    pub buses: Vec<BusRow<F>>,

    /// Mask form of the ZeroCheck's final claim,
    /// and the masked value the replay reaches.
    pub val_final_form: Vec<(Unknown, Flat<F>)>,
    pub val_final_masked: Flat<F>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TableShape {
    pub num_vars: usize,

    /// Virtual columns, after expansion.
    pub num_columns: usize,

    pub sumcheck_degree: usize,
    pub num_buses: usize,
    pub mul_nodes: usize,
    pub ring_units: bool,
}

impl TableShape {
    pub fn from_air<F: TowerField>(
        air: &impl Air<F>,
        num_vars: usize,
        ast: &ConstraintAst<F>,
    ) -> errors::Result<Self> {
        let num_buses = air.permutation_checks().len();

        let mut sumcheck_degree = ast.max_degree() + 1;

        if !air.boundary_constraints().is_empty() {
            sumcheck_degree = sumcheck_degree.max(2);
        }

        if num_buses > 0 {
            sumcheck_degree = sumcheck_degree.max(3);
        }

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
            num_vars,
            num_columns: air.num_columns(),
            sumcheck_degree,
            num_buses,
            mul_nodes: mul_node_count(ast),
            ring_units: plan.has_ring(),
        })
    }

    /// Eval claims per half: virtual columns, whole blind
    /// units, the ring blind unit's planes, one `h` per bus.
    pub fn eval_claims(&self, blinding_columns: usize) -> usize {
        let ring_blind = match self.ring_units && blinding_columns > 0 {
            true => RING_BLIND_BITS,
            false => 0,
        };

        self.num_columns + blinding_columns + ring_blind + self.num_buses
    }

    /// Final sumcheck values are the masked running
    /// claims themselves and consume no pad entry.
    pub fn masked_scalars(&self, blinding_columns: usize) -> usize {
        let zerocheck = self.num_vars * self.sumcheck_degree;
        let claims = 2 * self.eval_claims(blinding_columns);
        let trace_eval = 2 * self.num_vars;

        2 * self.num_buses + zerocheck + claims + trace_eval
    }

    /// Each bus contributes one `h · key` product;
    /// ring units add the gadget's squaring chain.
    pub fn mul_wires(&self) -> usize {
        let gadget = match self.ring_units {
            true => linearized::BITS - 1,
            false => 0,
        };

        self.mul_nodes + self.num_buses + gadget
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OuterStatement {
    pub masked_scalars: usize,
    pub mul_wires: usize,
}

impl OuterStatement {
    pub fn new(shapes: &[TableShape], blinding_columns: usize) -> Self {
        Self {
            masked_scalars: shapes
                .iter()
                .map(|s| s.masked_scalars(blinding_columns))
                .sum(),
            mul_wires: shapes.iter().map(TableShape::mul_wires).sum(),
        }
    }

    /// Statement size for a whole proof.
    pub fn for_tables<F: TowerField, A: Air<F>>(
        main: &A,
        main_num_vars: usize,
        chiplets: &[ChipletDef<F>],
        chiplet_num_vars: &[usize],
        blinding_columns: usize,
    ) -> errors::Result<Self> {
        if chiplets.len() != chiplet_num_vars.len() {
            return Err(errors::Error::Protocol {
                protocol: "outer",
                message: "one height per chiplet is required",
            });
        }

        let mut shapes = vec![TableShape::from_air(
            main,
            main_num_vars,
            &main.constraint_ast(),
        )?];

        for (def, &num_vars) in chiplets.iter().zip(chiplet_num_vars) {
            shapes.push(TableShape::from_air(def, num_vars, &def.constraint_ast())?);
        }

        Ok(Self::new(&shapes, blinding_columns))
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
}

/// One table's share of the statement, in transcript order.
pub struct TableRecord<F> {
    pub claims_masked: Vec<Flat<F>>,
    pub predicate: PredicateRows<F>,
    pub consistency: ConsistencyInputs<F>,

    /// `(pad, claim, public value)` per fixed column.
    pub fixed: Vec<(u32, u32, Flat<F>)>,

    pub trace_eval: EvalRecord<F>,
    pub gadget: Option<RingGadget<F>>,

    /// `(bus id, pad, masked claimed sum)` per bus.
    pub claimed_sums: Vec<(String, u32, Flat<F>)>,

    pub mul_wires: u32,
}

/// The assembled statement: affine rows, and the ring
/// gadgets whose tie coefficients [`linear_weights`]
/// streams onto their chain rows.
pub struct OuterRows<F> {
    pub affine: Vec<AffineRow<F>>,
    pub gadgets: Vec<RingGadget<F>>,
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

/// Public data of one eval sumcheck on masked claims.
pub struct EvalInputs<'a, F> {
    pub plan: &'a RingSwitchPlan,
    pub eta: F,
    pub r_mix: &'a [Block128],
    pub shifted_claims: bool,
    pub challenges: &'a [Flat<F>],
    pub claim_masked: Flat<F>,
    pub fin: Flat<F>,
    pub claim_pads: &'a [u32],
    pub round_first: u32,
    pub first_wire: u32,
}

pub struct EvalOutput<F> {
    pub record: EvalRecord<F>,
    pub gadget: Option<RingGadget<F>>,
}

/// Everything a table's record is built from; both
/// sides hold all of it after the table's eval phase.
pub struct TableInputs<'a, F: TowerField, A> {
    pub air: &'a A,
    pub ast: &'a ConstraintAst<F>,
    pub instance: &'a ProgramInstance<F>,
    pub num_vars: usize,
    pub trace_width: usize,
    pub blinding_columns: usize,

    pub alpha: Flat<F>,
    pub gamma: Flat<F>,
    pub beta: Flat<F>,
    pub r_zerocheck: &'a [Flat<F>],
    pub r_final: &'a [Flat<F>],
    pub lookup_bus_points: &'a BTreeMap<String, Vec<Flat<F>>>,

    pub claimed_sums_masked: &'a [(String, F)],
    pub h_evals_masked: &'a [(String, F)],
    pub val_final_masked: Flat<F>,
    pub claims_masked: Vec<Flat<F>>,

    pub pads: TablePads,
    pub trace_eval: EvalRecord<F>,
    pub gadget: Option<RingGadget<F>>,
}

pub fn eval_initial_form<F: HardwareField + Into<Block128> + From<u128>>(
    plan: &RingSwitchPlan,
    eta: F,
    r_mix: &[Block128],
    shifted_claims: bool,
    claim_pads: &[u32],
    first_wire: u32,
) -> errors::Result<InitialForm<F>> {
    let weights = claim_weights(plan, eta, shifted_claims);
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
pub fn eval_record<F>(inputs: &EvalInputs<'_, F>) -> errors::Result<EvalOutput<F>>
where
    F: HardwareField + Into<Block128> + From<u128>,
{
    let initial = eval_initial_form(
        inputs.plan,
        inputs.eta,
        inputs.r_mix,
        inputs.shifted_claims,
        inputs.claim_pads,
        inputs.first_wire,
    )?;

    let round_pads: Vec<[u32; 2]> = (0..inputs.challenges.len() as u32)
        .map(|i| [inputs.round_first + 2 * i, inputs.round_first + 2 * i + 1])
        .collect();
    let round_refs: Vec<&[u32]> = round_pads.iter().map(|p| p.as_slice()).collect();

    Ok(EvalOutput {
        record: EvalRecord {
            mask_form: sumcheck_mask_form(&initial.form, &round_refs, inputs.challenges, 2),
            claim_masked: inputs.claim_masked,
            fin: inputs.fin,
        },
        gadget: initial.gadget,
    })
}

pub fn table_record<F, A>(inputs: TableInputs<'_, F, A>) -> errors::Result<TableRecord<F>>
where
    F: TowerField + HardwareField,
    A: Air<F>,
{
    let air = inputs.air;
    let ast = inputs.ast;

    let boundary_constraints = air.boundary_constraints();
    let bus_specs = air.permutation_checks();
    let shape = TableShape::from_air(air, inputs.num_vars, ast)?;

    let num_vars = inputs.num_vars;
    let trace_width = inputs.trace_width;
    let r_final = inputs.r_final;
    let pad_first = inputs.pads.claims;
    let half = shape.eval_claims(inputs.blinding_columns) as u32;

    if inputs.claims_masked.len() != 2 * half as usize {
        return Err(errors::Error::Protocol {
            protocol: "outer",
            message: "eval claim count does not match the table shape",
        });
    }

    let eq_row_checker = TensorProduct::new(r_final.to_vec());

    let mut fixed = Vec::new();
    for fc in &air.fixed_columns() {
        if fc.col_idx >= trace_width {
            return Err(errors::Error::Protocol {
                protocol: "outer",
                message: "fixed column col_idx out of trace_width",
            });
        }

        fixed.push((
            pad_first + fc.col_idx as u32,
            fc.col_idx as u32,
            fc.shape.evaluate(r_final),
        ));
    }

    let mut boundary = Vec::with_capacity(boundary_constraints.len());
    for bc in &boundary_constraints {
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

    let mut buses = Vec::with_capacity(bus_specs.len());
    for (spec_idx, (bus_id, spec)) in bus_specs.iter().enumerate() {
        let eq_lookup = match spec.kind {
            BusKind::Permutation => Flat::from_raw(F::ONE),
            BusKind::Lookup => {
                let r_bus =
                    inputs
                        .lookup_bus_points
                        .get(bus_id)
                        .ok_or(errors::Error::Protocol {
                            protocol: "outer",
                            message: "lookup bus spec missing r_bus point",
                        })?;

                if r_bus.len() < num_vars {
                    return Err(errors::Error::Protocol {
                        protocol: "outer",
                        message: "r_bus shorter than table num_vars",
                    });
                }

                let one = Flat::from_raw(F::ONE);
                let mut eq_r_hi_at_0 = one;
                for r_j in &r_bus[num_vars..] {
                    eq_r_hi_at_0 *= one - *r_j;
                }

                eq_r_hi_at_0 * TensorProduct::evaluate_eq_slice(&r_bus[0..num_vars], r_final)
            }
        };

        let mut sources = Vec::new();
        for (source, _) in &spec.sources {
            match source {
                Source::Column(col) | Source::PhaseColumn(col) => {
                    sources.push(BusSource::Claim(*col as u32))
                }
                Source::Columns(indices) => {
                    sources.extend(indices.iter().map(|&col| BusSource::Claim(col as u32)));
                }
                Source::Const(v) => sources.push(BusSource::Public(F::from(*v).to_hardware())),
                Source::RowIndexLeBytes(n) => {
                    sources.push(BusSource::Public(eval_row_idx_le_mle::<F>(*n, r_final)));
                }
                Source::RowIndexByte(n) => {
                    sources.push(BusSource::Public(eval_row_idx_byte_mle::<F>(*n, r_final)));
                }
            }
        }

        buses.push(BusRow {
            h_pad: inputs.pads.h_evals + spec_idx as u32,
            h_masked: inputs.h_evals_masked[spec_idx].1.to_hardware(),
            h_wire: (shape.mul_nodes + spec_idx) as u32,
            sources,
            selector: spec.selector.map(|i| i as u32),
            recv_selector: spec.recv_selector.map(|i| i as u32),
            eq_lookup,
        });
    }

    let logup_alpha_offset = ast.roots.len() + boundary_constraints.len() + inputs.blinding_columns;
    let mut alpha_pow = Flat::from_raw(F::ONE);

    for _ in 0..logup_alpha_offset {
        alpha_pow *= inputs.alpha;
    }

    let mut initial = Vec::with_capacity(bus_specs.len());
    for k in 0..bus_specs.len() as u32 {
        initial.push((Unknown::Pad(inputs.pads.claimed_sums + k), alpha_pow));
        alpha_pow *= inputs.alpha;
    }

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
        gamma: inputs.gamma,
        beta: inputs.beta,
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
        predicate: compile(ast, ClaimLayout { pad_first, half }),
        consistency,
        fixed,
        trace_eval: inputs.trace_eval,
        gadget: inputs.gadget,
        claimed_sums: inputs
            .claimed_sums_masked
            .iter()
            .enumerate()
            .map(|(k, (bus_id, sum))| {
                (
                    bus_id.clone(),
                    inputs.pads.claimed_sums + k as u32,
                    sum.to_hardware(),
                )
            })
            .collect(),
        mul_wires: shape.mul_wires() as u32,
    })
}

/// Every affine row of the statement, claims folded
/// into constants and wire indices offset per table.
pub fn assemble<F: TowerField + HardwareField>(
    records: Vec<TableRecord<F>>,
    statement: &OuterStatement,
) -> errors::Result<OuterRows<F>> {
    let mut affine = Vec::new();
    let mut gadgets = Vec::new();
    let mut offset = 0u32;
    let mut buses: BTreeMap<String, Vec<(u32, Flat<F>)>> = BTreeMap::new();

    for mut record in records {
        let first = affine.len();

        affine.append(&mut record.predicate.affine);
        affine.push(consistency_row(
            &record.predicate.roots,
            &record.consistency,
        ));

        for bus in &record.consistency.buses {
            let [lhs, rhs] =
                bus_operand_rows(bus, record.consistency.pad_first, record.consistency.beta);

            affine.push(lhs);
            affine.push(rhs);
        }

        for &(pad, claim, public) in &record.fixed {
            affine.push(fixed_column_row(pad, claim, public));
        }

        affine.push(eval_final_row(
            &record.trace_eval.mask_form,
            record.trace_eval.claim_masked,
            record.trace_eval.fin,
        ));

        let half = (record.claims_masked.len() / 2) as u32;
        let num_buses = record.consistency.buses.len() as u32;
        let pad_first = record.consistency.pad_first;

        for (k, bus) in record.consistency.buses.iter().enumerate() {
            let claim = half - num_buses + k as u32;

            affine.push(h_claim_pin_row(
                bus.h_pad,
                bus.h_masked,
                pad_first + claim,
                claim,
            ));
        }

        if let Some(mut gadget) = record.gadget {
            gadget.first_row = affine.len();
            affine.extend(gadget.wire_rows());
            gadgets.push(gadget);
        }

        for row in &mut affine[first..] {
            resolve_table_row(row, &record.claims_masked, offset);
        }

        offset += record.mul_wires;

        for (bus_id, pad, masked) in record.claimed_sums {
            buses.entry(bus_id).or_default().push((pad, masked));
        }
    }

    if offset as usize != statement.mul_wires {
        return Err(errors::Error::Protocol {
            protocol: "outer",
            message: "table wire counts do not sum to the statement",
        });
    }

    for endpoints in buses.values() {
        affine.push(bus_sum_row(endpoints));
    }

    Ok(OuterRows { affine, gadgets })
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

pub fn consistency_row<F: TowerField + HardwareField>(
    roots: &[Form<F>],
    inputs: &ConsistencyInputs<F>,
) -> AffineRow<F> {
    let mut row = AffineRow {
        unknowns: inputs.val_final_form.to_vec(),
        claims: Vec::new(),
        constant: inputs.val_final_masked,
    };

    let mut alpha_pow = Flat::from_raw(F::ONE);

    for form in roots {
        scale_into(&mut row, form, alpha_pow * inputs.eq_zc);
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

    for bus in &inputs.buses {
        let consistency = alpha_pow * inputs.eq_zc;
        let gamma_weight = consistency * inputs.gamma;

        row.unknowns.push((Unknown::Pad(bus.h_pad), gamma_weight));

        row.constant += gamma_weight * bus.h_masked;

        row.unknowns.push((
            Unknown::Wire {
                mul: bus.h_wire,
                role: WireRole::Product,
            },
            consistency,
        ));

        match bus.selector {
            Some(s) => push_claim(&mut row, inputs.pad_first, s, consistency),
            None => row.constant += consistency,
        }

        if let Some(s) = bus.recv_selector {
            push_claim(&mut row, inputs.pad_first, s, consistency);
        }

        let lookup_weight = alpha_pow * bus.eq_lookup;

        row.unknowns.push((Unknown::Pad(bus.h_pad), lookup_weight));

        row.constant += lookup_weight * bus.h_masked;

        alpha_pow *= inputs.alpha;
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

/// `Σ claimed_sum = 0` over one bus id:
/// `(pad, masked sum)` per endpoint.
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
        if let Unknown::Wire { mul, .. } = unknown {
            *mul += mul_offset;
        }
    }
}

pub fn bus_key_form<F: TowerField + HardwareField>(
    bus: &BusRow<F>,
    pad_first: u32,
    beta: Flat<F>,
) -> Form<F> {
    let mut form = Form::default();
    let mut beta_pow = Flat::from_raw(F::ONE);

    for source in &bus.sources {
        match source {
            BusSource::Claim(idx) => {
                form.unknowns
                    .push((Unknown::Pad(pad_first + idx), beta_pow));
                form.claims.push((*idx, beta_pow));
            }
            BusSource::Public(v) => form.constant += *v * beta_pow,
        }

        beta_pow *= beta;
    }

    form
}

/// Pins the bus Hadamard wire `h · key`:
/// `Lhs = h` and `Rhs = key` over the pad.
pub fn bus_operand_rows<F: TowerField + HardwareField>(
    bus: &BusRow<F>,
    pad_first: u32,
    beta: Flat<F>,
) -> [AffineRow<F>; 2] {
    let one = Flat::from_raw(F::ONE);

    let lhs = AffineRow {
        unknowns: vec![
            (
                Unknown::Wire {
                    mul: bus.h_wire,
                    role: WireRole::Lhs,
                },
                one,
            ),
            (Unknown::Pad(bus.h_pad), one),
        ],
        claims: Vec::new(),
        constant: bus.h_masked,
    };

    let key = bus_key_form(bus, pad_first, beta);

    let mut unknowns = key.unknowns;
    unknowns.push((
        Unknown::Wire {
            mul: bus.h_wire,
            role: WireRole::Rhs,
        },
        one,
    ));

    let rhs = AffineRow {
        unknowns,
        claims: key.claims,
        constant: key.constant,
    };

    [lhs, rhs]
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
pub fn linear_tensor_vars<F>(rows: &OuterRows<F>) -> usize {
    rows.affine.len().next_power_of_two().ilog2() as usize
}

pub fn linear_weights<F: TowerField + HardwareField>(
    layout: &OuterLayout,
    rows: &OuterRows<F>,
    tensor: &[F],
) -> errors::Result<LinearBatch<F>> {
    if tensor.len() != linear_tensor_vars(rows) {
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

    let scales = expand_batch_tensor(tensor, rows.affine.len());

    let mut weights = vec![vec![Flat::from_raw(F::ZERO); layout.message_len]; layout.total_rows()];
    let mut target = Flat::from_raw(F::ZERO);

    for (row, &scale) in rows.affine.iter().zip(&scales) {
        for &(unknown, coeff) in &row.unknowns {
            let (orow, ocol) = layout.slot(unknown);
            weights[orow][ocol] += scale * coeff;
        }

        target += scale * row.constant;
    }

    for gadget in &rows.gadgets {
        gadget.for_each_tie_row(|j, tie| {
            let scale = scales[gadget.row_of(j)];
            for (&(pad, _), &t) in gadget.ring.iter().zip(tie) {
                let (orow, ocol) = layout.slot(Unknown::Pad(pad));
                weights[orow][ocol] += scale * t;
            }
        });
    }

    let used: Vec<usize> = (0..layout.total_rows())
        .filter(|&r| weights[r].iter().any(|v| *v != Flat::from_raw(F::ZERO)))
        .collect();

    let selected = used.iter().map(|&r| mem::take(&mut weights[r])).collect();

    Ok(LinearBatch {
        weights: selected,
        rows: used,
        target,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProgramCell;
    use crate::constraint::ConstraintArena;
    use hekate_math::Block128;

    type F = Block128;

    const BUS_PAD_FIRST: u32 = 10;
    const BUS_H_PAD: u32 = 20;
    const BUS_PUBLIC: u128 = 77;

    /// `half - num_buses` at `bus_record`'s `half = 2` and one bus.
    const H_CLAIM: usize = 1;

    fn shape(num_vars: usize, num_columns: usize, degree: usize, buses: usize) -> TableShape {
        TableShape {
            num_vars,
            num_columns,
            sumcheck_degree: degree,
            num_buses: buses,
            mul_nodes: 0,
            ring_units: false,
        }
    }

    fn mix(seed: u128) -> Flat<F> {
        F::from(
            seed.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .wrapping_add(0x51ed_2701),
        )
        .to_hardware()
    }

    /// One table:
    /// `c0 · c1` as the only constraint, two columns,
    /// one Permutation bus over `[Claim(0), Public(77)]`
    /// with selector `c1`.
    fn bus_record(claims_masked: Vec<Flat<F>>, h_masked: Flat<F>, beta: Flat<F>) -> TableRecord<F> {
        let mut arena = ConstraintArena::<F>::new();

        let a = arena.cell(ProgramCell::current(0));
        let b = arena.cell(ProgramCell::current(1));
        let root = arena.mul(a, b);

        let ast = ConstraintAst {
            arena,
            roots: vec![root],
            labels: vec![None],
        };

        let predicate = compile(
            &ast,
            ClaimLayout {
                pad_first: BUS_PAD_FIRST,
                half: 2,
            },
        );

        let bus = BusRow {
            h_pad: BUS_H_PAD,
            h_masked,
            h_wire: predicate.mul_nodes,
            sources: vec![BusSource::Claim(0), BusSource::Public(mix(BUS_PUBLIC))],
            selector: Some(1),
            recv_selector: None,
            eq_lookup: Flat::from_raw(F::ONE),
        };

        let mul_wires = predicate.mul_nodes + 1;

        TableRecord {
            claims_masked,
            predicate,
            consistency: ConsistencyInputs {
                pad_first: BUS_PAD_FIRST,
                eq_zc: mix(1),
                alpha: mix(2),
                gamma: mix(3),
                beta,
                boundary: Vec::new(),
                telescope: Vec::new(),
                buses: vec![bus],
                val_final_form: Vec::new(),
                val_final_masked: mix(4),
            },
            fixed: Vec::new(),
            trace_eval: EvalRecord {
                mask_form: Vec::new(),
                claim_masked: mix(5),
                fin: mix(6),
            },
            gadget: None,
            claimed_sums: vec![(String::from("bus"), 0, mix(7))],
            mul_wires,
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

    #[test]
    fn ring_units_add_squaring_chain() {
        let plain = shape(12, 100, 3, 0);
        let ring = TableShape {
            ring_units: true,
            ..plain
        };

        assert_eq!(ring.mul_wires() - plain.mul_wires(), 127);
    }

    #[test]
    fn ring_blind_planes_are_masked_only_under_blinding() {
        let plain = shape(12, 100, 3, 0);
        let ring = TableShape {
            ring_units: true,
            ..plain
        };

        assert_eq!(ring.masked_scalars(0), plain.masked_scalars(0));
        assert_eq!(
            ring.masked_scalars(1) - plain.masked_scalars(1),
            2 * RING_BLIND_BITS
        );
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
        let with_bus = shape(12, 100, 3, 1);
        let without = TableShape {
            num_buses: 0,
            ..with_bus
        };

        let blind_units = 1;
        let claimed_sum_and_h_eval = 2;
        let h_claim_halves = 2;

        assert_eq!(
            with_bus.masked_scalars(blind_units) - without.masked_scalars(blind_units),
            claimed_sum_and_h_eval + h_claim_halves
        );
    }

    #[test]
    fn masked_scalars_match_transcript_tally() {
        let s = shape(12, 1977, 5, 3);

        let blind_units = 1;
        let claimed_sums_and_h_evals = 2 * 3;
        let zerocheck_rounds = 12 * 5;
        let claims = 2 * (1977 + blind_units + 3);
        let trace_eval_rounds = 2 * 12;

        assert_eq!(
            s.masked_scalars(blind_units),
            claimed_sums_and_h_evals + zerocheck_rounds + claims + trace_eval_rounds
        );
    }

    #[test]
    fn every_bus_adds_hadamard_row() {
        let s = TableShape {
            mul_nodes: 40,
            ..shape(12, 100, 3, 3)
        };

        assert_eq!(s.mul_wires(), 43);
    }

    #[test]
    fn statement_sums_over_every_table() {
        let main = TableShape {
            mul_nodes: 40,
            ..shape(12, 100, 3, 2)
        };
        let chiplet = TableShape {
            mul_nodes: 7,
            ..shape(10, 30, 4, 1)
        };

        let statement = OuterStatement::new(&[main, chiplet], 2);

        assert_eq!(
            statement.masked_scalars,
            main.masked_scalars(2) + chiplet.masked_scalars(2)
        );
        assert_eq!(statement.mul_wires, 50);
    }

    #[test]
    fn every_hadamard_wire_operand_is_pinned_by_row() {
        let claims: Vec<Flat<F>> = (0..4).map(|i| mix(100 + i)).collect();
        let record = bus_record(claims, mix(8), mix(9));
        let statement = OuterStatement {
            masked_scalars: 32,
            mul_wires: record.mul_wires as usize,
        };

        let rows = assemble(vec![record], &statement).unwrap();

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
    fn bus_operand_rows_hold_on_honest_wires() {
        let plain: Vec<Flat<F>> = (0..4).map(|i| mix(200 + i)).collect();
        let pad: Vec<Flat<F>> = (0..32).map(|i| mix(300 + i)).collect();
        let claims_masked: Vec<Flat<F>> = plain
            .iter()
            .enumerate()
            .map(|(i, &c)| c + pad[BUS_PAD_FIRST as usize + i])
            .collect();

        let beta = mix(9);
        let h = mix(400);
        let h_masked = h + pad[BUS_H_PAD as usize];
        let key = plain[0] + beta * mix(BUS_PUBLIC);

        let record = bus_record(claims_masked.clone(), h_masked, beta);
        let bus = &record.consistency.buses[0];
        let [lhs, rhs] = bus_operand_rows(bus, BUS_PAD_FIRST, beta);

        let wire = |mul: u32, role: WireRole| -> Flat<F> {
            assert_eq!(mul, bus.h_wire);
            match role {
                WireRole::Lhs => h,
                WireRole::Rhs => key,
                WireRole::Product => h * key,
            }
        };

        for row in [&lhs, &rhs] {
            let mut left = Flat::from_raw(F::ZERO);
            for &(u, coeff) in &row.unknowns {
                let value = match u {
                    Unknown::Pad(i) => pad[i as usize],
                    Unknown::Wire { mul, role } => wire(mul, role),
                };

                left += coeff * value;
            }

            let mut right = row.constant;
            for &(idx, coeff) in &row.claims {
                right += coeff * claims_masked[idx as usize];
            }

            assert_eq!(left, right);
        }

        assert_eq!(mentions(&[lhs], bus.h_wire, WireRole::Lhs), 1);
        assert_eq!(mentions(&[rhs], bus.h_wire, WireRole::Rhs), 1);
    }

    #[test]
    fn bus_operand_rows_reject_free_operand() {
        let plain: Vec<Flat<F>> = (0..4).map(|i| mix(200 + i)).collect();
        let pad: Vec<Flat<F>> = (0..32).map(|i| mix(300 + i)).collect();
        let claims_masked: Vec<Flat<F>> = plain
            .iter()
            .enumerate()
            .map(|(i, &c)| c + pad[BUS_PAD_FIRST as usize + i])
            .collect();

        let beta = mix(9);
        let h = mix(400);
        let h_masked = h + pad[BUS_H_PAD as usize];
        let key = plain[0] + beta * mix(BUS_PUBLIC);

        let record = bus_record(claims_masked.clone(), h_masked, beta);
        let bus = &record.consistency.buses[0];
        let [lhs, _] = bus_operand_rows(bus, BUS_PAD_FIRST, beta);

        let one = Flat::from_raw(F::ONE);
        let forged = |role: WireRole| match role {
            WireRole::Lhs => one,
            WireRole::Rhs | WireRole::Product => h * key,
        };

        assert_eq!(
            forged(WireRole::Lhs) * forged(WireRole::Rhs),
            forged(WireRole::Product)
        );

        let mut left = Flat::from_raw(F::ZERO);
        for &(u, coeff) in &lhs.unknowns {
            let value = match u {
                Unknown::Pad(i) => pad[i as usize],
                Unknown::Wire { role, .. } => forged(role),
            };

            left += coeff * value;
        }

        let mut right = lhs.constant;
        for &(idx, coeff) in &lhs.claims {
            right += coeff * claims_masked[idx as usize];
        }

        assert_ne!(left, right);
    }

    #[test]
    fn layout_holds_whole_statement() {
        let cfg = hekate_core::config::Config::prod();

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
        let cfg = hekate_core::config::Config::prod();
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
        let h = mix(400);
        let h_masked = h + pad[BUS_H_PAD as usize];

        let pin_form = [
            (Unknown::Pad(BUS_H_PAD), one),
            (Unknown::Pad(BUS_PAD_FIRST + H_CLAIM as u32), one),
        ];

        for (h_claim, holds) in [(h, true), (h + one, false)] {
            let mut plain: Vec<Flat<F>> = (0..4).map(|i| mix(200 + i)).collect();
            plain[H_CLAIM] = h_claim;

            let claims_masked: Vec<Flat<F>> = plain
                .iter()
                .enumerate()
                .map(|(i, &c)| c + pad[BUS_PAD_FIRST as usize + i])
                .collect();

            let record = bus_record(claims_masked.clone(), h_masked, mix(9));
            let statement = OuterStatement {
                masked_scalars: 32,
                mul_wires: record.mul_wires as usize,
            };

            let rows = assemble(vec![record], &statement).unwrap();
            let pins: Vec<&AffineRow<F>> = rows
                .affine
                .iter()
                .filter(|r| r.unknowns == pin_form)
                .collect();

            assert_eq!(pins.len(), 1, "one pin row per bus");
            assert_eq!(pins[0].constant, h_masked + claims_masked[H_CLAIM]);

            let left = pad[BUS_H_PAD as usize] + pad[BUS_PAD_FIRST as usize + H_CLAIM];

            assert_eq!(left == pins[0].constant, holds);
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
}
