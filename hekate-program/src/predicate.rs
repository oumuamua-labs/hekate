// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use crate::ProgramCell;
use crate::constraint::{ConstraintAst, ConstraintExpr, ExprId};
use alloc::vec;
use alloc::vec::Vec;
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
    Wire { mul: u32, role: WireRole },
}

impl Unknown {
    pub fn shifted(self, mul_offset: u32) -> Self {
        match self {
            Unknown::Pad(_) => self,
            Unknown::Wire { mul, role } => Unknown::Wire {
                mul: mul + mul_offset,
                role,
            },
        }
    }
}

/// `half` is `V + K`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClaimLayout {
    pub pad_first: u32,
    pub half: u32,
}

impl ClaimLayout {
    pub fn index(&self, cell: ProgramCell) -> u32 {
        let base = cell.col_idx as u32;

        match cell.next_row {
            true => base + self.half,
            false => base,
        }
    }
}

/// `Σ unknowns = Σ claims + constant`.
#[derive(Clone, Debug)]
pub struct AffineRow<F> {
    pub unknowns: Vec<(Unknown, Flat<F>)>,
    pub claims: Vec<(u32, Flat<F>)>,
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
    pub claims: Vec<(u32, Flat<F>)>,
    pub constant: Flat<F>,
}

impl<F: TowerField> Default for Form<F> {
    fn default() -> Self {
        Self {
            unknowns: Vec::new(),
            claims: Vec::new(),
            constant: Flat::from_raw(F::ZERO),
        }
    }
}

impl<F: HardwareField> Form<F> {
    fn extend(&mut self, other: &Self) {
        self.unknowns.extend_from_slice(&other.unknowns);
        self.claims.extend_from_slice(&other.claims);

        self.constant += other.constant;
    }

    fn scaled(&self, coeff: Flat<F>) -> Self {
        Self {
            unknowns: self.unknowns.iter().map(|&(v, c)| (v, c * coeff)).collect(),
            claims: self.claims.iter().map(|&(v, c)| (v, c * coeff)).collect(),
            constant: self.constant * coeff,
        }
    }

    pub fn evaluate(&self, claims: &[Flat<F>], pad: &[Flat<F>], wires: &[[Flat<F>; 3]]) -> Flat<F> {
        let mut acc = self.constant;
        for &(idx, coeff) in &self.claims {
            acc += coeff * claims[idx as usize];
        }

        for &(unknown, coeff) in &self.unknowns {
            let v = match unknown {
                Unknown::Pad(i) => pad[i as usize],
                Unknown::Wire { mul, role } => wires[mul as usize][role as usize],
            };

            acc += coeff * v;
        }

        acc
    }
}

/// Honest `(lhs, rhs, product)` per reachable `Mul`,
/// numbered exactly as [`compile`] numbers its wires.
pub fn wire_values<F: TowerField + HardwareField>(
    ast: &ConstraintAst<F>,
    node_values: &[Flat<F>],
) -> Vec<[Flat<F>; 3]> {
    let live = crate::outer::reachable(ast);
    let mut wires = Vec::new();

    for (i, &alive) in live.iter().enumerate() {
        if !alive {
            continue;
        }

        if let ConstraintExpr::Mul(a, b) = ast.arena.get(ExprId(i as u32)) {
            wires.push([
                node_values[a.0 as usize],
                node_values[b.0 as usize],
                node_values[i],
            ]);
        }
    }

    wires
}

pub fn compile<F: HardwareField>(ast: &ConstraintAst<F>, layout: ClaimLayout) -> PredicateRows<F> {
    let n = ast.arena.len();
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
                let idx = layout.index(*cell);

                Form {
                    unknowns: vec![(Unknown::Pad(layout.pad_first + idx), one)],
                    claims: vec![(idx, one)],
                    constant: Flat::from_raw(F::ZERO),
                }
            }
            ConstraintExpr::Const(v) => Form {
                unknowns: Vec::new(),
                claims: Vec::new(),
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
            ConstraintExpr::Mul(a, b) => {
                let mul = mul_nodes;
                mul_nodes += 1;

                for (role, side) in [(WireRole::Lhs, a), (WireRole::Rhs, b)] {
                    let operand = &forms[side.0 as usize];

                    let mut unknowns = operand.unknowns.clone();
                    unknowns.push((Unknown::Wire { mul, role }, one));

                    affine.push(AffineRow {
                        unknowns,
                        claims: operand.claims.clone(),
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
                    claims: Vec::new(),
                    constant: Flat::from_raw(F::ZERO),
                }
            }
        };

        forms.push(form);
    }

    let roots = ast
        .roots
        .iter()
        .map(|r| forms[r.0 as usize].clone())
        .collect();

    PredicateRows {
        affine,
        mul_nodes,
        roots,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::ConstraintArena;
    use hekate_math::{Block128, Flat, HardwareField};

    type F = Block128;

    const LAYOUT: ClaimLayout = ClaimLayout {
        pad_first: 100,
        half: 8,
    };

    const COLS: usize = 6;

    const HONEST: ClaimLayout = ClaimLayout {
        pad_first: 0,
        half: COLS as u32,
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

    fn value_of(
        unknowns: &[(Unknown, Flat<F>)],
        claims: &[(u32, Flat<F>)],
        constant: Flat<F>,
        masked: &[F],
        pad: &[F],
        wires: &[[Flat<F>; 3]],
    ) -> Flat<F> {
        let mut acc = constant;
        for &(idx, coeff) in claims {
            acc += coeff * masked[idx as usize].to_hardware();
        }

        for &(unknown, coeff) in unknowns {
            let v = match unknown {
                Unknown::Pad(i) => pad[i as usize].to_hardware(),
                Unknown::Wire { mul, role } => {
                    let slot = match role {
                        WireRole::Lhs => 0,
                        WireRole::Rhs => 1,
                        WireRole::Product => 2,
                    };

                    wires[mul as usize][slot]
                }
            };

            acc += coeff * v;
        }

        acc
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
        let rows = compile(&ast, HONEST);

        let plain: Vec<F> = (0..2 * COLS).map(|i| mix(i as u128 + 1)).collect();
        let pad: Vec<F> = (0..2 * COLS).map(|i| mix(i as u128 + 977)).collect();
        let masked: Vec<F> = plain.iter().zip(&pad).map(|(c, h)| *c + *h).collect();

        let current: Vec<Flat<F>> = plain[..COLS].iter().map(|v| v.to_hardware()).collect();
        let next: Vec<Flat<F>> = plain[COLS..].iter().map(|v| v.to_hardware()).collect();

        let consts = ast.precompute_hardware_consts();

        let mut node_values = Vec::new();
        ast.evaluate_into(&consts, &current, &next, &mut node_values);

        let wires = wire_values(&ast, &node_values);

        assert_eq!(wires.len(), rows.mul_nodes as usize);

        for row in &rows.affine {
            let zero = Flat::from_raw(F::ZERO);
            let lhs = value_of(&row.unknowns, &[], zero, &masked, &pad, &wires);
            let rhs = value_of(&[], &row.claims, row.constant, &masked, &pad, &wires);

            assert_eq!(lhs, rhs);
        }

        for wire in &wires {
            assert_eq!(wire[0] * wire[1], wire[2]);
        }
    }

    #[test]
    fn compiled_roots_reproduce_ast_evaluation() {
        let ast = rich_ast();
        let rows = compile(&ast, HONEST);

        let plain: Vec<F> = (0..2 * COLS).map(|i| mix(i as u128 + 1)).collect();
        let pad: Vec<F> = (0..2 * COLS).map(|i| mix(i as u128 + 977)).collect();
        let masked: Vec<F> = plain.iter().zip(&pad).map(|(c, h)| *c + *h).collect();

        let current: Vec<Flat<F>> = plain[..COLS].iter().map(|v| v.to_hardware()).collect();
        let next: Vec<Flat<F>> = plain[COLS..].iter().map(|v| v.to_hardware()).collect();

        let consts = ast.precompute_hardware_consts();

        let mut node_values = Vec::new();
        ast.evaluate_into(&consts, &current, &next, &mut node_values);

        let wires = wire_values(&ast, &node_values);
        let expected = ast.evaluate(&current, &next);

        for (k, form) in rows.roots.iter().enumerate() {
            let got = value_of(
                &form.unknowns,
                &form.claims,
                form.constant,
                &masked,
                &pad,
                &wires,
            );

            assert_eq!(got, expected[k]);
        }
    }

    #[test]
    fn flipped_claim_breaks_roots_that_read_it() {
        let ast = rich_ast();
        let rows = compile(&ast, HONEST);

        let plain: Vec<F> = (0..2 * COLS).map(|i| mix(i as u128 + 1)).collect();
        let pad: Vec<F> = (0..2 * COLS).map(|i| mix(i as u128 + 977)).collect();

        let mut masked: Vec<F> = plain.iter().zip(&pad).map(|(c, h)| *c + *h).collect();

        let current: Vec<Flat<F>> = plain[..COLS].iter().map(|v| v.to_hardware()).collect();
        let next: Vec<Flat<F>> = plain[COLS..].iter().map(|v| v.to_hardware()).collect();

        let consts = ast.precompute_hardware_consts();

        let mut node_values = Vec::new();
        ast.evaluate_into(&consts, &current, &next, &mut node_values);

        let wires = wire_values(&ast, &node_values);

        let honest: Vec<Flat<F>> = rows
            .roots
            .iter()
            .map(|f| value_of(&f.unknowns, &f.claims, f.constant, &masked, &pad, &wires))
            .collect();

        const FLIPPED: u32 = 2;
        masked[FLIPPED as usize] += F::ONE;

        for (k, form) in rows.roots.iter().enumerate() {
            let got = value_of(
                &form.unknowns,
                &form.claims,
                form.constant,
                &masked,
                &pad,
                &wires,
            );

            let reads_it = form.claims.iter().any(|&(idx, _)| idx == FLIPPED);

            assert_eq!(got != honest[k], reads_it);
        }
    }

    #[test]
    fn cell_binds_one_pad_entry_and_one_claim() {
        let ast = ast_of(|a| vec![a.cell(ProgramCell::current(3))]);
        let rows = compile(&ast, LAYOUT);

        assert_eq!(rows.mul_nodes, 0);
        assert!(rows.affine.is_empty());
        assert_eq!(rows.roots[0].unknowns, vec![(Unknown::Pad(103), one())]);
        assert_eq!(rows.roots[0].claims, vec![(3, one())]);
    }

    #[test]
    fn next_row_cell_lands_in_second_half() {
        let ast = ast_of(|a| vec![a.cell(ProgramCell::next(3))]);
        let rows = compile(&ast, LAYOUT);

        assert_eq!(rows.roots[0].claims, vec![(11, one())]);
        assert_eq!(rows.roots[0].unknowns, vec![(Unknown::Pad(111), one())]);
    }

    #[test]
    fn mul_emits_two_affine_rows_and_yields_its_product_wire() {
        let ast = ast_of(|a| {
            let x = a.cell(ProgramCell::current(0));
            let y = a.cell(ProgramCell::current(1));

            vec![a.mul(x, y)]
        });

        let rows = compile(&ast, LAYOUT);

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

        let rows = compile(&ast, LAYOUT);
        let lhs = &rows.affine[0];

        assert_eq!(lhs.claims, vec![(0, one())]);
        assert!(lhs.unknowns.contains(&(Unknown::Pad(100), one())));
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

        let rows = compile(&ast, LAYOUT);

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

        let rows = compile(&ast, LAYOUT);

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

        let rows = compile(&ast, LAYOUT);

        assert_eq!(rows.roots[0].constant, one());
        assert_eq!(rows.roots[0].claims, vec![(0, one())]);
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

        let rows = compile(&ast, LAYOUT);

        assert_eq!(rows.roots[0].claims, vec![(0, two.to_hardware())]);
        assert_eq!(rows.roots[0].constant, two.to_hardware());
    }

    #[test]
    fn sum_collects_every_child() {
        let ast = ast_of(|a| {
            let cells: Vec<ExprId> = (0..4).map(|i| a.cell(ProgramCell::current(i))).collect();

            vec![a.sum(cells)]
        });

        let rows = compile(&ast, LAYOUT);

        assert_eq!(rows.roots[0].claims.len(), 4);
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

        let rows = compile(&ast, LAYOUT);

        assert_eq!(rows.mul_nodes as usize, crate::outer::mul_node_count(&ast));
    }
}
