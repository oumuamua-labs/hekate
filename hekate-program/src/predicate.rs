// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use crate::ProgramCell;
use crate::constraint::{ConstraintAst, ConstraintExpr, ExprId};
use alloc::vec;
use alloc::vec::Vec;
use hekate_core::errors;
use hekate_math::{Flat, HardwareField, TowerField};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WireRole {
    Lhs = 0,
    Rhs = 1,
    Product = 2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Unknown {
    Pad(u32),
    Claim(u32),
    Wire { mul: u32, role: WireRole },
}

impl Unknown {
    pub fn shifted(self, mul_offset: u32) -> Self {
        match self {
            Unknown::Pad(_) | Unknown::Claim(_) => self,
            Unknown::Wire { mul, role } => Unknown::Wire {
                mul: mul + mul_offset,
                role,
            },
        }
    }
}

/// Places a table's AST cells among every table's eval claims: its
/// first at `first`, its next-row claims from `half` in `next`'s slots.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClaimLayout<'a> {
    pub first: u32,
    pub half: u32,
    pub next: &'a [Option<usize>],
}

impl ClaimLayout<'_> {
    /// Gives a cell's claim index, `None` for a next-row
    /// cell whose unit ships no next-row claim.
    pub fn index(&self, cell: ProgramCell) -> Option<u32> {
        match cell.next_row {
            true => {
                let slot = self.next.get(cell.col_idx).copied().flatten()?;

                Some(self.half + slot as u32)
            }
            false => Some(cell.col_idx as u32),
        }
    }
}

/// `Σ unknowns = constant`.
#[derive(Clone, Debug)]
pub struct AffineRow<F> {
    pub unknowns: Vec<(Unknown, Flat<F>)>,
    pub constant: Flat<F>,
}

/// Hadamard rows are implicit:
/// `Lhs · Rhs = Product` for each `mul` in `0..mul_nodes`.
#[derive(Clone, Debug)]
pub struct PredicateRows<F> {
    pub affine: Vec<AffineRow<F>>,
    pub mul_nodes: u32,
    pub roots: Vec<Form<F>>,
}

#[derive(Clone, Debug)]
pub struct Form<F> {
    pub unknowns: Vec<(Unknown, Flat<F>)>,
    pub constant: Flat<F>,
}

impl<F: TowerField> Default for Form<F> {
    fn default() -> Self {
        Self {
            unknowns: Vec::new(),
            constant: Flat::from_raw(F::ZERO),
        }
    }
}

impl<F: HardwareField> Form<F> {
    pub(crate) fn extend(&mut self, other: &Self) {
        self.unknowns.extend_from_slice(&other.unknowns);

        self.constant += other.constant;
    }

    pub(crate) fn scaled(&self, coeff: Flat<F>) -> Self {
        Self {
            unknowns: self.unknowns.iter().map(|&(v, c)| (v, c * coeff)).collect(),
            constant: self.constant * coeff,
        }
    }

    pub fn evaluate(&self, claims: &[Flat<F>], pad: &[Flat<F>], wires: &[[Flat<F>; 3]]) -> Flat<F> {
        let mut acc = self.constant;
        for &(unknown, coeff) in &self.unknowns {
            let v = match unknown {
                Unknown::Pad(i) => pad[i as usize],
                Unknown::Claim(i) => claims[i as usize],
                Unknown::Wire { mul, role } => wires[mul as usize][role as usize],
            };

            acc += coeff * v;
        }

        acc
    }
}

/// Wired product of one level, located by its node and by
/// the number [`compile`] and [`wire_values`] give its wire.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LevelEntry {
    pub node: ExprId,
    pub wire: u32,
}

/// Marks the nodes whose subtree reads constants
/// and current-row cells of `fixed_columns` alone.
pub fn fixed_only<F: TowerField>(
    ast: &ConstraintAst<F>,
    fixed_columns: impl IntoIterator<Item = usize>,
) -> Vec<bool> {
    let mut columns: Vec<usize> = fixed_columns.into_iter().collect();
    columns.sort_unstable();

    let n = ast.arena.len();

    let mut only: Vec<bool> = Vec::with_capacity(n);
    for i in 0..n {
        let fixed = match ast.arena.get(ExprId(i as u32)) {
            ConstraintExpr::Cell(cell) => {
                !cell.next_row && columns.binary_search(&cell.col_idx).is_ok()
            }
            ConstraintExpr::Const(_) => true,
            ConstraintExpr::Add(a, b) | ConstraintExpr::Mul(a, b) => {
                only[a.0 as usize] && only[b.0 as usize]
            }
            ConstraintExpr::Scale(_, a) => only[a.0 as usize],
            ConstraintExpr::Sum(children) => children.iter().all(|c| only[c.0 as usize]),
        };

        only.push(fixed);
    }

    only
}

/// Fills `out` with the value of each node `fixed_only` marks,
/// `None` elsewhere, reading each fixed column's value at the table's
/// point from `column`; errs on any marked node left without one.
pub fn fixed_values<F: HardwareField>(
    ast: &ConstraintAst<F>,
    fixed_only: &[bool],
    column: impl Fn(usize) -> Option<Flat<F>>,
    out: &mut Vec<Option<Flat<F>>>,
) -> errors::Result<()> {
    let missing = || errors::Error::Protocol {
        protocol: "outer",
        message: "fixed-only node reads a value it was not given",
    };

    if fixed_only.len() != ast.arena.len() {
        return Err(missing());
    }

    out.clear();

    for (i, &fixed) in fixed_only.iter().enumerate() {
        if !fixed {
            out.push(None);
            continue;
        }

        let child = |id: &ExprId| {
            out.get(id.0 as usize)
                .copied()
                .flatten()
                .ok_or_else(missing)
        };

        let value = match ast.arena.get(ExprId(i as u32)) {
            ConstraintExpr::Cell(cell) => match cell.next_row {
                true => return Err(missing()),
                false => column(cell.col_idx).ok_or_else(missing)?,
            },
            ConstraintExpr::Const(c) => c.to_hardware(),
            ConstraintExpr::Add(a, b) => child(a)? + child(b)?,
            ConstraintExpr::Mul(a, b) => child(a)? * child(b)?,
            ConstraintExpr::Scale(coeff, a) => coeff.to_hardware() * child(a)?,
            ConstraintExpr::Sum(children) => {
                let mut acc = Flat::from_raw(F::ZERO);
                for c in children {
                    acc += child(c)?;
                }

                acc
            }
        };

        out.push(Some(value));
    }

    Ok(())
}

/// Gives the operands of node `i` when it multiplies two nodes
/// outside `fixed_only`, the one kind of product that takes wires.
pub(crate) fn wired_mul<F: TowerField>(
    ast: &ConstraintAst<F>,
    fixed_only: &[bool],
    i: usize,
) -> Option<(ExprId, ExprId)> {
    match ast.arena.get(ExprId(i as u32)) {
        ConstraintExpr::Mul(a, b) if !fixed_only[a.0 as usize] && !fixed_only[b.0 as usize] => {
            Some((*a, *b))
        }
        _ => None,
    }
}

/// Groups wired products by nesting depth, each level in arena order.
/// Level 0 holds those roots read through linear nodes, and level
/// `l + 1` those read that way by operands of level `l`.
pub fn product_levels<F: TowerField>(
    ast: &ConstraintAst<F>,
    fixed_only: &[bool],
) -> Vec<Vec<LevelEntry>> {
    let n = ast.arena.len();

    let mut wire = vec![0u32; n];
    let mut wires = 0;

    for (i, &alive) in crate::outer::reachable(ast).iter().enumerate() {
        if alive && wired_mul(ast, fixed_only, i).is_some() {
            wire[i] = wires;
            wires += 1;
        }
    }

    let mut levels = Vec::new();
    let mut seeds: Vec<ExprId> = ast.roots.clone();

    while !seeds.is_empty() {
        let mut live = vec![false; n];
        for seed in &seeds {
            live[seed.0 as usize] = true;
        }

        let mut entries = Vec::new();

        // Children sit at lower indices than their parent
        for i in (0..n).rev() {
            if !live[i] {
                continue;
            }

            let mut mark = |id: &ExprId| live[id.0 as usize] = true;

            match ast.arena.get(ExprId(i as u32)) {
                ConstraintExpr::Cell(_) | ConstraintExpr::Const(_) => {}
                ConstraintExpr::Add(a, b) => {
                    mark(a);
                    mark(b);
                }
                ConstraintExpr::Scale(_, a) => mark(a),
                ConstraintExpr::Sum(children) => children.iter().for_each(mark),
                ConstraintExpr::Mul(a, b) => {
                    match (fixed_only[a.0 as usize], fixed_only[b.0 as usize]) {
                        (true, true) => {}
                        (true, false) => mark(b),
                        (false, true) => mark(a),
                        (false, false) => entries.push(LevelEntry {
                            node: ExprId(i as u32),
                            wire: wire[i],
                        }),
                    }
                }
            }
        }

        entries.reverse();

        seeds = entries
            .iter()
            .filter_map(|entry| wired_mul(ast, fixed_only, entry.node.0 as usize))
            .flat_map(|(a, b)| [a, b])
            .collect();

        if !entries.is_empty() {
            levels.push(entries);
        }
    }

    levels
}

/// Counts the sumcheck rounds of one level of `entries` products.
pub fn level_rounds(entries: usize) -> usize {
    entries.next_power_of_two().ilog2() as usize
}

/// Honest `(lhs, rhs, product)` per reachable `Mul` that takes
/// wires, numbered exactly as [`compile`] numbers its wires.
pub fn wire_values<F: TowerField + HardwareField>(
    ast: &ConstraintAst<F>,
    fixed_only: &[bool],
    node_values: &[Flat<F>],
) -> Vec<[Flat<F>; 3]> {
    let live = crate::outer::reachable(ast);
    let mut wires = Vec::new();

    for (i, &alive) in live.iter().enumerate() {
        if !alive {
            continue;
        }

        if let Some((a, b)) = wired_mul(ast, fixed_only, i) {
            wires.push([
                node_values[a.0 as usize],
                node_values[b.0 as usize],
                node_values[i],
            ]);
        }
    }

    wires
}

pub fn compile<F: HardwareField>(
    ast: &ConstraintAst<F>,
    layout: ClaimLayout,
    fixed: &[Option<Flat<F>>],
) -> errors::Result<PredicateRows<F>> {
    let n = ast.arena.len();

    if fixed.len() != n {
        return Err(errors::Error::Protocol {
            protocol: "outer",
            message: "one fixed value slot per AST node is required",
        });
    }

    let live = crate::outer::reachable(ast);
    let one = Flat::from_raw(F::ONE);

    let mut forms: Vec<Form<F>> = Vec::with_capacity(n);
    let mut affine: Vec<AffineRow<F>> = Vec::new();
    let mut mul_nodes: u32 = 0;

    for (i, &alive) in live.iter().enumerate() {
        if !alive {
            forms.push(Form::default());
            continue;
        }

        let form = match ast.arena.get(ExprId(i as u32)) {
            ConstraintExpr::Cell(cell) => {
                let idx = layout.index(*cell).ok_or(unshipped_next_cell())?;

                Form {
                    unknowns: vec![(Unknown::Claim(layout.first + idx), one)],
                    constant: Flat::from_raw(F::ZERO),
                }
            }
            ConstraintExpr::Const(v) => Form {
                unknowns: Vec::new(),
                constant: v.to_hardware(),
            },
            ConstraintExpr::Add(a, b) => {
                let mut acc = forms[a.0 as usize].clone();
                acc.extend(&forms[b.0 as usize]);

                acc
            }
            ConstraintExpr::Scale(coeff, a) => forms[a.0 as usize].scaled(coeff.to_hardware()),
            ConstraintExpr::Sum(children) => {
                let mut acc = Form::default();
                for c in children {
                    acc.extend(&forms[c.0 as usize]);
                }

                acc
            }
            ConstraintExpr::Mul(a, b) => match (fixed[a.0 as usize], fixed[b.0 as usize]) {
                (Some(lhs), Some(rhs)) => Form {
                    unknowns: Vec::new(),
                    constant: lhs * rhs,
                },
                (Some(scale), None) => forms[b.0 as usize].scaled(scale),
                (None, Some(scale)) => forms[a.0 as usize].scaled(scale),
                (None, None) => {
                    let mul = mul_nodes;

                    mul_nodes += 1;

                    for (role, side) in [(WireRole::Lhs, a), (WireRole::Rhs, b)] {
                        let operand = &forms[side.0 as usize];

                        let mut unknowns = operand.unknowns.clone();
                        unknowns.push((Unknown::Wire { mul, role }, one));

                        affine.push(AffineRow {
                            unknowns,
                            constant: operand.constant,
                        });
                    }

                    Form {
                        unknowns: vec![(
                            Unknown::Wire {
                                mul,
                                role: WireRole::Product,
                            },
                            one,
                        )],
                        constant: Flat::from_raw(F::ZERO),
                    }
                }
            },
        };

        forms.push(form);
    }

    let roots = ast
        .roots
        .iter()
        .map(|r| forms[r.0 as usize].clone())
        .collect();

    Ok(PredicateRows {
        affine,
        mul_nodes,
        roots,
    })
}

pub(crate) fn unshipped_next_cell() -> errors::Error {
    errors::Error::Protocol {
        protocol: "outer",
        message: "AST reads a next-row cell its table ships no claim for",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::ConstraintArena;
    use hekate_math::{Block128, Flat, HardwareField};

    type F = Block128;

    const SLOTS: [Option<usize>; 8] = [
        Some(0),
        Some(1),
        Some(2),
        Some(3),
        Some(4),
        Some(5),
        Some(6),
        Some(7),
    ];

    const LAYOUT: ClaimLayout = ClaimLayout {
        first: 100,
        half: 8,
        next: &SLOTS,
    };

    const COLS: usize = 6;

    const HONEST: ClaimLayout = ClaimLayout {
        first: 0,
        half: COLS as u32,
        next: &SLOTS,
    };

    fn mix(seed: u128) -> F {
        F::from(
            seed.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .wrapping_add(0x51ed_2701),
        )
    }

    fn one() -> Flat<F> {
        Flat::from_raw(F::ONE)
    }

    fn honest_claims() -> Vec<Flat<F>> {
        (0..2 * COLS)
            .map(|i| mix(i as u128 + 1).to_hardware())
            .collect()
    }

    fn sum_of(
        unknowns: &[(Unknown, Flat<F>)],
        claims: &[Flat<F>],
        wires: &[[Flat<F>; 3]],
    ) -> Flat<F> {
        let form = Form {
            unknowns: unknowns.to_vec(),
            constant: Flat::from_raw(F::ZERO),
        };

        form.evaluate(claims, &[], wires)
    }

    fn fixed_at(ast: &ConstraintAst<F>, columns: &[(usize, Flat<F>)]) -> Vec<Option<Flat<F>>> {
        let marks = fixed_only(ast, columns.iter().map(|&(column, _)| column));
        let value = |column: usize| {
            columns
                .iter()
                .find(|&&(fixed, _)| fixed == column)
                .map(|&(_, value)| value)
        };

        let mut values = Vec::new();
        fixed_values(ast, &marks, value, &mut values).unwrap();

        values
    }

    fn no_fixed_columns(ast: &ConstraintAst<F>) -> Vec<Option<Flat<F>>> {
        fixed_at(ast, &[])
    }

    fn ast_of(build: impl FnOnce(&mut ConstraintArena<F>) -> Vec<ExprId>) -> ConstraintAst<F> {
        let mut arena = ConstraintArena::<F>::new();
        let roots = build(&mut arena);
        let labels = roots.iter().map(|_| None).collect();

        ConstraintAst {
            arena,
            roots,
            labels,
        }
    }

    fn rich_ast() -> ConstraintAst<F> {
        ast_of(|a| {
            let cols: Vec<ExprId> = (0..COLS).map(|i| a.cell(ProgramCell::current(i))).collect();
            let nexts: Vec<ExprId> = (0..COLS).map(|i| a.cell(ProgramCell::next(i))).collect();

            let shared = a.mul(cols[0], cols[1]);
            let k = a.constant(mix(7));
            let scaled = a.scale(mix(11), shared);
            let summed = a.sum(vec![cols[2], nexts[3], k, scaled]);
            let deep = a.mul(summed, nexts[4]);
            let nested = a.mul(deep, cols[5]);
            let tail = a.add(nested, shared);

            vec![tail, a.add(shared, nexts[0]), deep]
        })
    }

    #[test]
    fn honest_values_satisfy_every_affine_row() {
        let ast = rich_ast();
        let rows = compile(&ast, HONEST, &no_fixed_columns(&ast)).unwrap();

        let claims = honest_claims();
        let (current, next) = claims.split_at(COLS);

        let consts = ast.precompute_hardware_consts();

        let mut node_values = Vec::new();
        ast.evaluate_into(&consts, current, next, &mut node_values);

        let wires = wire_values(&ast, &fixed_only(&ast, []), &node_values);

        assert_eq!(wires.len(), rows.mul_nodes as usize);

        for row in &rows.affine {
            assert_eq!(sum_of(&row.unknowns, &claims, &wires), row.constant);
        }

        for wire in &wires {
            assert_eq!(wire[0] * wire[1], wire[2]);
        }
    }

    #[test]
    fn compiled_roots_reproduce_ast_evaluation() {
        let ast = rich_ast();
        let rows = compile(&ast, HONEST, &no_fixed_columns(&ast)).unwrap();

        let claims = honest_claims();
        let (current, next) = claims.split_at(COLS);

        let consts = ast.precompute_hardware_consts();

        let mut node_values = Vec::new();
        ast.evaluate_into(&consts, current, next, &mut node_values);

        let wires = wire_values(&ast, &fixed_only(&ast, []), &node_values);
        let expected = ast.evaluate(current, next);

        for (k, form) in rows.roots.iter().enumerate() {
            assert_eq!(form.evaluate(&claims, &[], &wires), expected[k]);
        }
    }

    #[test]
    fn flipped_claim_breaks_roots_that_read_it() {
        let ast = rich_ast();
        let rows = compile(&ast, HONEST, &no_fixed_columns(&ast)).unwrap();

        let mut claims = honest_claims();
        let (current, next) = claims.split_at(COLS);

        let consts = ast.precompute_hardware_consts();

        let mut node_values = Vec::new();
        ast.evaluate_into(&consts, current, next, &mut node_values);

        let wires = wire_values(&ast, &fixed_only(&ast, []), &node_values);

        let honest: Vec<Flat<F>> = rows
            .roots
            .iter()
            .map(|form| form.evaluate(&claims, &[], &wires))
            .collect();

        const FLIPPED: u32 = 2;
        claims[FLIPPED as usize] += one();

        for (k, form) in rows.roots.iter().enumerate() {
            let got = form.evaluate(&claims, &[], &wires);
            let reads_it = form
                .unknowns
                .iter()
                .any(|&(unknown, _)| unknown == Unknown::Claim(FLIPPED));

            assert_eq!(got != honest[k], reads_it);
        }
    }

    #[test]
    fn cell_reads_its_claim() {
        let ast = ast_of(|a| vec![a.cell(ProgramCell::current(3))]);
        let rows = compile(&ast, LAYOUT, &no_fixed_columns(&ast)).unwrap();

        assert_eq!(rows.mul_nodes, 0);
        assert!(rows.affine.is_empty());
        assert_eq!(rows.roots[0].unknowns, vec![(Unknown::Claim(103), one())]);
    }

    #[test]
    fn next_row_cell_lands_in_second_half() {
        let ast = ast_of(|a| vec![a.cell(ProgramCell::next(3))]);
        let rows = compile(&ast, LAYOUT, &no_fixed_columns(&ast)).unwrap();

        assert_eq!(rows.roots[0].unknowns, vec![(Unknown::Claim(111), one())]);
    }

    #[test]
    fn mul_emits_two_affine_rows_and_yields_its_product_wire() {
        let ast = ast_of(|a| {
            let x = a.cell(ProgramCell::current(0));
            let y = a.cell(ProgramCell::current(1));

            vec![a.mul(x, y)]
        });

        let rows = compile(&ast, LAYOUT, &no_fixed_columns(&ast)).unwrap();

        assert_eq!(rows.mul_nodes, 1);
        assert_eq!(rows.affine.len(), 2);
        assert_eq!(
            rows.roots[0].unknowns,
            vec![(
                Unknown::Wire {
                    mul: 0,
                    role: WireRole::Product
                },
                one()
            )]
        );
    }

    #[test]
    fn wire_row_pins_its_operand() {
        let ast = ast_of(|a| {
            let x = a.cell(ProgramCell::current(0));
            let y = a.cell(ProgramCell::current(1));

            vec![a.mul(x, y)]
        });

        let rows = compile(&ast, LAYOUT, &no_fixed_columns(&ast)).unwrap();
        let lhs = &rows.affine[0];

        assert!(lhs.unknowns.contains(&(Unknown::Claim(100), one())));
        assert!(lhs.unknowns.contains(&(
            Unknown::Wire {
                mul: 0,
                role: WireRole::Lhs
            },
            one()
        )));
    }

    #[test]
    fn nested_muls_number_their_wires_in_arena_order() {
        let ast = ast_of(|a| {
            let x = a.cell(ProgramCell::current(0));
            let y = a.cell(ProgramCell::current(1));
            let z = a.cell(ProgramCell::current(2));
            let inner = a.mul(x, y);

            vec![a.mul(inner, z)]
        });

        let rows = compile(&ast, LAYOUT, &no_fixed_columns(&ast)).unwrap();

        assert_eq!(rows.mul_nodes, 2);
        assert_eq!(rows.affine.len(), 4);
        assert_eq!(
            rows.affine[2].unknowns,
            vec![
                (
                    Unknown::Wire {
                        mul: 0,
                        role: WireRole::Product
                    },
                    one()
                ),
                (
                    Unknown::Wire {
                        mul: 1,
                        role: WireRole::Lhs
                    },
                    one()
                )
            ]
        );
    }

    #[test]
    fn shared_mul_is_compiled_once() {
        let ast = ast_of(|a| {
            let x = a.cell(ProgramCell::current(0));
            let y = a.cell(ProgramCell::current(1));
            let shared = a.mul(x, y);

            vec![a.add(shared, x), a.add(shared, y)]
        });

        let rows = compile(&ast, LAYOUT, &no_fixed_columns(&ast)).unwrap();

        assert_eq!(rows.mul_nodes, 1);
        assert_eq!(rows.affine.len(), 2);
    }

    #[test]
    fn constant_lands_on_public_side() {
        let ast = ast_of(|a| {
            let k = a.constant(F::ONE);
            let x = a.cell(ProgramCell::current(0));

            vec![a.add(k, x)]
        });

        let rows = compile(&ast, LAYOUT, &no_fixed_columns(&ast)).unwrap();

        assert_eq!(rows.roots[0].constant, one());
        assert_eq!(rows.roots[0].unknowns, vec![(Unknown::Claim(100), one())]);
    }

    #[test]
    fn scale_multiplies_every_coefficient() {
        let two = F::from(2u8);

        let ast = ast_of(|a| {
            let x = a.cell(ProgramCell::current(0));
            let k = a.constant(F::ONE);
            let s = a.add(x, k);

            vec![a.scale(two, s)]
        });

        let rows = compile(&ast, LAYOUT, &no_fixed_columns(&ast)).unwrap();

        assert_eq!(
            rows.roots[0].unknowns,
            vec![(Unknown::Claim(100), two.to_hardware())]
        );
        assert_eq!(rows.roots[0].constant, two.to_hardware());
    }

    #[test]
    fn sum_collects_every_child() {
        let ast = ast_of(|a| {
            let cells: Vec<ExprId> = (0..4).map(|i| a.cell(ProgramCell::current(i))).collect();

            vec![a.sum(cells)]
        });

        let rows = compile(&ast, LAYOUT, &no_fixed_columns(&ast)).unwrap();

        assert_eq!(rows.roots[0].unknowns.len(), 4);
    }

    #[test]
    fn mul_count_agrees_with_shape_pass() {
        let ast = ast_of(|a| {
            let x = a.cell(ProgramCell::current(0));
            let y = a.cell(ProgramCell::current(1));
            let z = a.cell(ProgramCell::current(2));
            let p = a.mul(x, y);
            let q = a.mul(p, z);

            vec![a.add(p, q)]
        });

        let rows = compile(&ast, LAYOUT, &no_fixed_columns(&ast)).unwrap();

        let marks = fixed_only(&ast, []);

        assert_eq!(
            rows.mul_nodes as usize,
            crate::outer::mul_node_count(&ast, &marks)
        );
    }

    #[test]
    fn next_row_cell_lands_at_its_slot() {
        let ast = ast_of(|a| vec![a.cell(ProgramCell::next(3))]);
        let slots = [None, None, None, Some(0)];
        let layout = ClaimLayout {
            first: 100,
            half: 8,
            next: &slots,
        };

        let rows = compile(&ast, layout, &no_fixed_columns(&ast)).unwrap();

        assert_eq!(rows.roots[0].unknowns, vec![(Unknown::Claim(108), one())]);
    }

    #[test]
    fn unshipped_next_row_cell_rejected() {
        let ast = ast_of(|a| vec![a.cell(ProgramCell::next(2))]);
        let slots = [None, None, None, Some(0)];
        let layout = ClaimLayout {
            first: 0,
            half: 8,
            next: &slots,
        };

        assert!(compile(&ast, layout, &no_fixed_columns(&ast)).is_err());
    }

    #[test]
    fn fixed_marks_follow_subtree_reads() {
        let ast = ast_of(|a| {
            let gate = a.cell(ProgramCell::current(0));
            let witness = a.cell(ProgramCell::current(1));
            let gate_next = a.cell(ProgramCell::next(0));
            let k = a.constant(F::ONE);
            let off = a.add(gate, k);
            let scaled = a.scale(mix(3), off);
            let both = a.mul(gate, scaled);
            let mixed = a.sum(vec![both, witness]);
            let empty = a.sum(Vec::new());

            vec![mixed, gate_next, empty]
        });

        assert_eq!(
            fixed_only(&ast, [0]),
            [true, false, false, true, true, true, true, false, true]
        );
        assert_eq!(
            fixed_only(&ast, []),
            [false, false, false, true, false, false, false, false, true]
        );
    }

    #[test]
    fn fixed_gate_scales_body_without_wire() {
        let ast = ast_of(|a| {
            let gate = a.cell(ProgramCell::current(0));
            let x = a.cell(ProgramCell::current(1));
            let y = a.cell(ProgramCell::next(2));
            let body = a.add(x, y);

            vec![a.mul(gate, body), a.mul(body, gate)]
        });

        let scale = mix(31).to_hardware();
        let rows = compile(&ast, LAYOUT, &fixed_at(&ast, &[(0, scale)])).unwrap();

        assert_eq!(rows.mul_nodes, 0);
        assert!(rows.affine.is_empty());

        for root in &rows.roots {
            assert_eq!(
                root.unknowns,
                vec![(Unknown::Claim(101), scale), (Unknown::Claim(110), scale)]
            );
        }
    }

    #[test]
    fn nested_gates_multiply_their_scales() {
        let ast = ast_of(|a| {
            let outer = a.cell(ProgramCell::current(0));
            let inner = a.cell(ProgramCell::current(1));
            let x = a.cell(ProgramCell::current(2));
            let gated = a.mul(inner, x);

            vec![a.mul(outer, gated)]
        });

        let (u, v) = (mix(41).to_hardware(), mix(43).to_hardware());
        let rows = compile(&ast, LAYOUT, &fixed_at(&ast, &[(0, u), (1, v)])).unwrap();

        assert_eq!(rows.mul_nodes, 0);
        assert_eq!(rows.roots[0].unknowns, vec![(Unknown::Claim(102), u * v)]);
    }

    #[test]
    fn product_of_two_fixed_columns_is_constant() {
        let ast = ast_of(|a| {
            let f = a.cell(ProgramCell::current(0));
            let g = a.cell(ProgramCell::current(1));

            vec![a.mul(f, g)]
        });

        let (u, v) = (mix(41).to_hardware(), mix(43).to_hardware());
        let rows = compile(&ast, LAYOUT, &fixed_at(&ast, &[(0, u), (1, v)])).unwrap();

        assert_eq!(rows.mul_nodes, 0);
        assert!(rows.roots[0].unknowns.is_empty());
        assert_eq!(rows.roots[0].constant, u * v);
    }

    #[test]
    fn fixed_column_read_at_next_row_keeps_its_wire() {
        let ast = ast_of(|a| {
            let gate_next = a.cell(ProgramCell::next(0));
            let x = a.cell(ProgramCell::current(1));

            vec![a.mul(gate_next, x)]
        });

        let fixed = fixed_at(&ast, &[(0, mix(31).to_hardware())]);
        let rows = compile(&ast, LAYOUT, &fixed).unwrap();

        assert_eq!(rows.mul_nodes, 1);
        assert_eq!(rows.affine.len(), 2);
    }

    #[test]
    fn constant_operand_scales_without_wire() {
        let ast = ast_of(|a| {
            let k = a.constant(mix(7));
            let x = a.cell(ProgramCell::current(0));

            vec![a.mul(k, x)]
        });

        let rows = compile(&ast, LAYOUT, &no_fixed_columns(&ast)).unwrap();

        assert_eq!(rows.mul_nodes, 0);
        assert_eq!(
            rows.roots[0].unknowns,
            vec![(Unknown::Claim(100), mix(7).to_hardware())]
        );
    }

    #[test]
    fn gate_under_wired_product_scales_operand_row() {
        let ast = ast_of(|a| {
            let gate = a.cell(ProgramCell::current(0));
            let x = a.cell(ProgramCell::current(1));
            let y = a.cell(ProgramCell::current(2));
            let gated = a.mul(gate, x);

            vec![a.mul(gated, y)]
        });

        let scale = mix(31).to_hardware();
        let rows = compile(&ast, LAYOUT, &fixed_at(&ast, &[(0, scale)])).unwrap();

        assert_eq!(rows.mul_nodes, 1);
        assert_eq!(rows.affine.len(), 2);
        assert_eq!(rows.affine[0].unknowns[0], (Unknown::Claim(101), scale));
    }

    #[test]
    fn folded_roots_reproduce_ast_evaluation() {
        let ast = rich_ast();

        let claims = honest_claims();
        let (current, next) = claims.split_at(COLS);

        let fixed = fixed_at(&ast, &[(0, current[0]), (5, current[5])]);
        let rows = compile(&ast, HONEST, &fixed).unwrap();

        let consts = ast.precompute_hardware_consts();

        let mut node_values = Vec::new();
        ast.evaluate_into(&consts, current, next, &mut node_values);

        let wires = wire_values(&ast, &fixed_only(&ast, [0, 5]), &node_values);

        assert_eq!(rows.mul_nodes, 1);
        assert_eq!(wires.len(), 1);

        for row in &rows.affine {
            assert_eq!(sum_of(&row.unknowns, &claims, &wires), row.constant);
        }

        let expected = ast.evaluate(current, next);

        for (k, form) in rows.roots.iter().enumerate() {
            assert_eq!(form.evaluate(&claims, &[], &wires), expected[k]);
        }
    }

    #[test]
    fn marked_node_without_value_is_rejected() {
        let ast = ast_of(|a| vec![a.cell(ProgramCell::current(0))]);
        let marks = fixed_only(&ast, [0]);

        let mut values = Vec::new();

        assert!(fixed_values(&ast, &marks, |_| Some(one()), &mut values).is_ok());
        assert!(fixed_values(&ast, &marks, |_| None, &mut values).is_err());
        assert!(fixed_values(&ast, &[], |_| Some(one()), &mut values).is_err());
        assert!(compile(&ast, LAYOUT, &[]).is_err());
    }

    #[test]
    fn product_levels_follow_nesting_depth() {
        let ast = rich_ast();
        let levels = product_levels(&ast, &fixed_only(&ast, []));
        let wires: Vec<Vec<u32>> = levels
            .iter()
            .map(|level| level.iter().map(|entry| entry.wire).collect())
            .collect();

        // `shared`, `deep`, `nested` take wires 0, 1, 2; `nested`
        // reads `deep`, and `deep` reads `shared` under its scale.
        assert_eq!(wires, [vec![0, 1, 2], vec![0, 1], vec![0]]);

        for level in &levels {
            assert!(level.windows(2).all(|pair| pair[0].node.0 < pair[1].node.0));
        }
    }

    #[test]
    fn product_under_fixed_scale_stays_at_its_reader_level() {
        let ast = ast_of(|a| {
            let gate = a.cell(ProgramCell::current(0));
            let x = a.cell(ProgramCell::current(1));
            let y = a.cell(ProgramCell::current(2));
            let inner = a.mul(x, y);

            vec![a.mul(gate, inner)]
        });

        let gated = product_levels(&ast, &fixed_only(&ast, [0]));

        assert_eq!(gated.len(), 1);
        assert_eq!(gated[0].len(), 1);
        assert_eq!(gated[0][0].wire, 0);

        let free = product_levels(&ast, &fixed_only(&ast, []));

        assert_eq!(free.len(), 2);
        assert_eq!((free[0][0].wire, free[1][0].wire), (1, 0));
    }

    #[test]
    fn product_free_ast_has_no_level() {
        let ast = ast_of(|a| {
            let x = a.cell(ProgramCell::current(0));
            let y = a.cell(ProgramCell::current(1));

            vec![a.add(x, y)]
        });

        assert!(product_levels(&ast, &fixed_only(&ast, [])).is_empty());
    }

    #[test]
    fn level_rounds_cover_every_entry() {
        for (entries, rounds) in [
            (1, 0),
            (2, 1),
            (3, 2),
            (4, 2),
            (5, 3),
            (1024, 10),
            (1025, 11),
        ] {
            assert_eq!(level_rounds(entries), rounds);
        }
    }
}
