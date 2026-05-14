// SPDX-License-Identifier: Apache-2.0
// This file is part of the hekate-math project.
// Copyright (C) 2026 Andrei Kochergin <andrei@oumuamua.dev>
// Copyright (C) 2026 Oumuamua Labs <info@oumuamua.dev>. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! SYMBOLIC CONSTRAINTS DEFINITION

use crate::ProgramCell;
use alloc::vec;
use alloc::vec::Vec;
use hashbrown::HashMap;
use hekate_core::errors::Error;
use hekate_math::{Flat, HardwareField, TowerField};

pub mod builder;

/// Represents a single term in a polynomial constraint.
/// Form: `coeff * product(cells)`
/// Example: `5 * x_curr * y_next`
#[derive(Clone, Debug)]
pub struct ConstraintTerm<F> {
    pub coeff: F,
    pub poly_ind: Vec<ProgramCell>, // Multiplicands (variables)
}

impl<F: TowerField> ConstraintTerm<F> {
    pub fn new(coeff: F, cells: Vec<ProgramCell>) -> Self {
        Self {
            coeff,
            poly_ind: cells,
        }
    }
}

/// Represents a full constraint equation:
/// sum(terms) == 0.
#[derive(Clone, Debug)]
pub struct Constraint<F> {
    pub terms: Vec<ConstraintTerm<F>>,
}

impl<F: TowerField> Constraint<F> {
    pub fn new(terms: Vec<ConstraintTerm<F>>) -> Self {
        Self { terms }
    }
}

/// Source of the value a boundary constraint
/// pins `Trace(row_idx, col_idx)` to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BoundaryTarget<F> {
    /// `instance.public_inputs[idx]`. Main-program
    /// use only; `ChipletDef::from_air` rejects
    /// this variant on chiplets.
    PublicInput(usize),

    /// Literal field value. Required for
    /// chiplets, optional for main programs.
    Constant(F),
}

/// Pins a single trace cell to a
/// target value at a specific row.
#[derive(Clone, Debug)]
pub struct BoundaryConstraint<F> {
    pub col_idx: usize,
    pub row_idx: usize,
    pub target: BoundaryTarget<F>,
}

impl<F> BoundaryConstraint<F> {
    pub fn with_public_input(col_idx: usize, row_idx: usize, public_input_idx: usize) -> Self {
        Self {
            col_idx,
            row_idx,
            target: BoundaryTarget::PublicInput(public_input_idx),
        }
    }

    pub fn with_constant(col_idx: usize, row_idx: usize, val: F) -> Self {
        Self {
            col_idx,
            row_idx,
            target: BoundaryTarget::Constant(val),
        }
    }
}

impl<F: TowerField> BoundaryConstraint<F> {
    /// Read the pin value out of `target`.
    /// `Constant` returns the literal;
    /// `PublicInput(idx)` reads
    /// `instance.public_inputs[idx]`.
    pub fn resolve_target(
        &self,
        instance: &crate::ProgramInstance<F>,
    ) -> hekate_core::errors::Result<F> {
        match &self.target {
            BoundaryTarget::Constant(v) => Ok(*v),
            BoundaryTarget::PublicInput(idx) => {
                instance.public_input(*idx).ok_or(Error::Protocol {
                    protocol: "boundary",
                    message: "public_input_idx out of bounds",
                })
            }
        }
    }

    /// Constant values are load-bearing:
    /// a malicious prover could swap them between
    /// sessions while keeping the chiplet root valid
    /// unless the transcript binds them.
    pub fn absorb_into<H: hekate_crypto::Hasher>(
        &self,
        transcript: &mut hekate_crypto::transcript::Transcript<H>,
    ) {
        transcript.append_u64(b"chiplet_bnd_col", self.col_idx as u64);
        transcript.append_u64(b"chiplet_bnd_row", self.row_idx as u64);

        match &self.target {
            BoundaryTarget::PublicInput(idx) => {
                transcript.append_u64(b"chiplet_bnd_kind", 0);
                transcript.append_u64(b"chiplet_bnd_pub", *idx as u64);
            }
            BoundaryTarget::Constant(v) => {
                transcript.append_u64(b"chiplet_bnd_kind", 1);
                transcript.append_field(b"chiplet_bnd_const", *v);
            }
        }
    }
}

// =================================================================
// CONSTRAINT IR
// =================================================================

/// Index into a `ConstraintArena`. Cheap to copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExprId(pub u32);

/// A single node in the constraint AST-DAG.
/// All variants are algebraic operations over GF(2^128).
#[derive(Clone, Debug)]
pub enum ConstraintExpr<F> {
    /// A trace cell reference (column + current/next row).
    Cell(ProgramCell),

    /// A field constant.
    Const(F),

    /// Addition of two sub-expressions.
    /// In GF(2^N) this is XOR.
    Add(ExprId, ExprId),

    /// Multiplication of two sub-expressions.
    Mul(ExprId, ExprId),

    /// Scalar multiplication:
    /// coeff * expr.
    ///
    /// Avoids creating a Const node + Mul
    /// pair for the common case.
    Scale(F, ExprId),

    /// Sum of N sub-expressions.
    /// Exists specifically for Theta-style linear
    /// combinations to avoid deep Add chains.
    /// Evaluates to sum of children[i].
    Sum(Vec<ExprId>),
}

/// Arena-allocated constraint DAG.
/// Nodes reference each other by
/// `ExprId` (index into `nodes`).
///
/// Cell nodes are automatically deduplicated:
/// calling `cell()` twice with the same `ProgramCell`
/// returns the same `ExprId`. This is mandatory, the
/// downstream compiler maps ExprId to poly index,
/// so duplicate cells would create duplicate polys
/// in VirtualPoly and bloat Sumcheck evaluation.
pub struct ConstraintArena<F> {
    nodes: Vec<ConstraintExpr<F>>,

    /// Dedup cache for Cell nodes.
    /// Same ProgramCell -> same ExprId.
    cell_cache: HashMap<ProgramCell, ExprId>,
}

impl<F: TowerField> Default for ConstraintArena<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: TowerField> ConstraintArena<F> {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            cell_cache: HashMap::new(),
        }
    }

    /// Allocate a new expression node.
    /// Returns its ID.
    pub fn alloc(&mut self, expr: ConstraintExpr<F>) -> ExprId {
        let id = ExprId(self.nodes.len() as u32);
        self.nodes.push(expr);

        id
    }

    /// Read a node by ID.
    pub fn get(&self, id: ExprId) -> &ConstraintExpr<F> {
        &self.nodes[id.0 as usize]
    }

    /// Total number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Shift all Cell node col_idx by `offset`.
    /// Used when embedding a chiplet's AST into
    /// a combined program where column indices
    /// are offset.
    pub fn shift_cells(&mut self, offset: usize) {
        for node in &mut self.nodes {
            if let ConstraintExpr::Cell(cell) = node {
                cell.col_idx += offset;
            }
        }

        let old_cache = core::mem::take(&mut self.cell_cache);
        for (mut cell, id) in old_cache {
            cell.col_idx += offset;
            self.cell_cache.insert(cell, id);
        }
    }

    /// Create or reuse a cell reference.
    /// Automatically deduplicates:
    /// same ProgramCell -> same ExprId.
    pub fn cell(&mut self, cell: ProgramCell) -> ExprId {
        if let Some(&id) = self.cell_cache.get(&cell) {
            return id;
        }

        let id = self.alloc(ConstraintExpr::Cell(cell));
        self.cell_cache.insert(cell, id);

        id
    }

    /// Create a constant.
    pub fn constant(&mut self, val: F) -> ExprId {
        self.alloc(ConstraintExpr::Const(val))
    }

    /// a + b
    pub fn add(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.alloc(ConstraintExpr::Add(a, b))
    }

    /// a * b
    pub fn mul(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.alloc(ConstraintExpr::Mul(a, b))
    }

    /// coeff * a
    pub fn scale(&mut self, coeff: F, a: ExprId) -> ExprId {
        self.alloc(ConstraintExpr::Scale(coeff, a))
    }

    /// Sum of multiple expressions.
    pub fn sum(&mut self, children: Vec<ExprId>) -> ExprId {
        self.alloc(ConstraintExpr::Sum(children))
    }
}

/// A full constraint system in AST form.
/// `roots` are the top-level constraint expressions.
/// Each root must evaluate to 0 for a valid trace.
pub struct ConstraintAst<F> {
    pub arena: ConstraintArena<F>,
    pub roots: Vec<ExprId>,
    pub labels: Vec<Option<&'static str>>,
}

impl<F: TowerField> Clone for ConstraintArena<F> {
    fn clone(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
            cell_cache: self.cell_cache.clone(),
        }
    }
}

impl<F: TowerField> Clone for ConstraintAst<F> {
    fn clone(&self) -> Self {
        Self {
            arena: self.arena.clone(),
            roots: self.roots.clone(),
            labels: self.labels.clone(),
        }
    }
}

impl<F: TowerField> ConstraintAst<F> {
    /// Maximum polynomial degree
    /// across all constraint roots.
    pub fn max_degree(&self) -> usize {
        if self.arena.is_empty() {
            return 0;
        }

        let n = self.arena.len();
        let mut deg: Vec<usize> = Vec::with_capacity(n);

        for i in 0..n {
            let d = match self.arena.get(ExprId(i as u32)) {
                ConstraintExpr::Cell(_) => 1,
                ConstraintExpr::Const(_) => 0,
                ConstraintExpr::Add(a, b) => deg[a.0 as usize].max(deg[b.0 as usize]),
                ConstraintExpr::Mul(a, b) => deg[a.0 as usize] + deg[b.0 as usize],
                ConstraintExpr::Scale(_, a) => deg[a.0 as usize],
                ConstraintExpr::Sum(children) => children
                    .iter()
                    .map(|c| deg[c.0 as usize])
                    .max()
                    .unwrap_or(0),
            };

            deg.push(d);
        }

        self.roots
            .iter()
            .map(|r| deg[r.0 as usize])
            .max()
            .unwrap_or(0)
    }

    /// Evaluate each constraint
    /// root at a single point.
    pub fn evaluate(&self, current_row: &[Flat<F>], next_row: &[Flat<F>]) -> Vec<Flat<F>>
    where
        F: HardwareField,
    {
        let n = self.arena.len();
        let mut val: Vec<Flat<F>> = Vec::with_capacity(n);

        for i in 0..n {
            let v = match self.arena.get(ExprId(i as u32)) {
                ConstraintExpr::Cell(cell) => {
                    if cell.next_row {
                        next_row[cell.col_idx]
                    } else {
                        current_row[cell.col_idx]
                    }
                }
                ConstraintExpr::Const(k) => k.to_hardware(),
                ConstraintExpr::Add(a, b) => val[a.0 as usize] + val[b.0 as usize],
                ConstraintExpr::Mul(a, b) => val[a.0 as usize] * val[b.0 as usize],
                ConstraintExpr::Scale(k, a) => k.to_hardware() * val[a.0 as usize],
                ConstraintExpr::Sum(children) => {
                    let mut s = Flat::from_raw(F::ZERO);
                    for c in children {
                        s += val[c.0 as usize];
                    }

                    s
                }
            };

            val.push(v);
        }

        self.roots.iter().map(|r| val[r.0 as usize]).collect()
    }

    /// Buffer-reuse variant of `evaluate()`.
    /// Caller owns `buf`; reused across
    /// rows to avoid per-row allocation.
    pub fn evaluate_into(
        &self,
        current_row: &[Flat<F>],
        next_row: &[Flat<F>],
        buf: &mut Vec<Flat<F>>,
    ) where
        F: HardwareField,
    {
        buf.clear();

        let n = self.arena.len();
        for i in 0..n {
            let v = match self.arena.get(ExprId(i as u32)) {
                ConstraintExpr::Cell(cell) => {
                    if cell.next_row {
                        next_row[cell.col_idx]
                    } else {
                        current_row[cell.col_idx]
                    }
                }
                ConstraintExpr::Const(k) => k.to_hardware(),
                ConstraintExpr::Add(a, b) => buf[a.0 as usize] + buf[b.0 as usize],
                ConstraintExpr::Mul(a, b) => buf[a.0 as usize] * buf[b.0 as usize],
                ConstraintExpr::Scale(k, a) => k.to_hardware() * buf[a.0 as usize],
                ConstraintExpr::Sum(children) => {
                    let mut s = Flat::from_raw(F::ZERO);
                    for c in children {
                        s += buf[c.0 as usize];
                    }

                    s
                }
            };

            buf.push(v);
        }
    }

    /// Merge another constraint AST into this one.
    pub fn merge(&mut self, other: ConstraintAst<F>) {
        let mut id_map: Vec<ExprId> = Vec::with_capacity(other.arena.len());
        for node in other.arena.nodes {
            let new_id = match node {
                ConstraintExpr::Cell(cell) => self.arena.cell(cell),
                ConstraintExpr::Const(val) => self.arena.constant(val),
                ConstraintExpr::Add(a, b) => {
                    self.arena.add(id_map[a.0 as usize], id_map[b.0 as usize])
                }
                ConstraintExpr::Mul(a, b) => {
                    self.arena.mul(id_map[a.0 as usize], id_map[b.0 as usize])
                }
                ConstraintExpr::Scale(coeff, inner) => {
                    self.arena.scale(coeff, id_map[inner.0 as usize])
                }
                ConstraintExpr::Sum(children) => {
                    let remapped: Vec<ExprId> =
                        children.into_iter().map(|c| id_map[c.0 as usize]).collect();
                    self.arena.sum(remapped)
                }
            };

            id_map.push(new_id);
        }

        for (root, label) in other.roots.into_iter().zip(other.labels) {
            self.roots.push(id_map[root.0 as usize]);
            self.labels.push(label);
        }
    }

    /// Convert AST to flat `Vec<Constraint<F>>`.
    /// Expands the DAG into sum-of-products form.
    pub fn to_constraints(&self) -> Vec<Constraint<F>> {
        /// A flat term:
        /// coefficient × product of cells.
        type FlatTerm<F> = (F, Vec<ProgramCell>);

        fn expand<F: TowerField>(
            arena: &ConstraintArena<F>,
            id: ExprId,
            cache: &mut Vec<Option<Vec<FlatTerm<F>>>>,
        ) -> Vec<FlatTerm<F>> {
            if let Some(ref cached) = cache[id.0 as usize] {
                return cached.clone();
            }

            let result = match arena.get(id) {
                ConstraintExpr::Cell(cell) => {
                    vec![(F::ONE, vec![*cell])]
                }
                ConstraintExpr::Const(k) => {
                    vec![(*k, vec![])]
                }
                ConstraintExpr::Add(a, b) => {
                    let mut terms = expand(arena, *a, cache);
                    terms.extend(expand(arena, *b, cache));

                    terms
                }
                ConstraintExpr::Mul(a, b) => {
                    let left = expand(arena, *a, cache);
                    let right = expand(arena, *b, cache);

                    let mut terms = Vec::with_capacity(left.len() * right.len());
                    for (lc, lp) in &left {
                        for (rc, rp) in &right {
                            let coeff = *lc * *rc;
                            let mut cells = lp.clone();

                            cells.extend_from_slice(rp);
                            terms.push((coeff, cells));
                        }
                    }

                    terms
                }
                ConstraintExpr::Scale(k, a) => {
                    let inner = expand(arena, *a, cache);
                    inner
                        .into_iter()
                        .map(|(c, cells)| (*k * c, cells))
                        .collect()
                }
                ConstraintExpr::Sum(children) => {
                    let mut terms = Vec::new();
                    for child in children {
                        terms.extend(expand(arena, *child, cache));
                    }

                    terms
                }
            };

            cache[id.0 as usize] = Some(result.clone());

            result
        }

        let n = self.arena.len();
        let mut cache: Vec<Option<Vec<FlatTerm<F>>>> = vec![None; n];

        self.roots
            .iter()
            .map(|root| {
                let flat_terms = expand(&self.arena, *root, &mut cache);
                Constraint::new(
                    flat_terms
                        .into_iter()
                        .map(|(coeff, cells)| ConstraintTerm::new(coeff, cells))
                        .collect(),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::builder::ConstraintSystem;
    use crate::constraint::ConstraintExpr;
    use crate::{Air, Program};
    use hekate_core::trace::ColumnType;
    use hekate_math::{Block128, Flat};

    type F = Block128;

    #[derive(Clone)]
    struct TestFibProgram;

    impl Air<F> for TestFibProgram {
        fn num_columns(&self) -> usize {
            3
        }

        fn column_layout(&self) -> &[ColumnType] {
            &[ColumnType::B32, ColumnType::B32, ColumnType::Bit]
        }

        fn constraint_ast(&self) -> ConstraintAst<F> {
            let cs = ConstraintSystem::<F>::new();

            let [a, b, q] = [cs.col(0), cs.col(1), cs.col(2)];
            let [na, nb] = [cs.next(0), cs.next(1)];

            cs.constrain(q * (na + b));
            cs.constrain(q * (nb + a + b));

            cs.build()
        }
    }

    impl Program<F> for TestFibProgram {}

    #[test]
    fn default_constraint_ast_produces_correct_roots() {
        let program = TestFibProgram;
        let ast = program.constraint_ast();

        // 2 constraints -> 2 roots
        assert_eq!(ast.roots.len(), 2);

        // c1 has 2 terms, c2 has 3 terms.
        // Each term: 1 Const + 2 Cell + 2 Mul = 5 nodes
        // c1: 2 terms * 5 + 1 Sum = 11 nodes (but cells are deduped)
        // Verify non-empty and structurally sound.
        assert!(!ast.arena.is_empty());

        // Both roots should be Sum or product nodes
        for &root in &ast.roots {
            let node = ast.arena.get(root);
            match node {
                ConstraintExpr::Sum(children) => {
                    assert!(!children.is_empty());
                }
                ConstraintExpr::Mul(_, _) => {
                    // single-term constraint would be a Mul
                }
                _ => panic!("Root should be Sum or Mul, got {:?}", node),
            }
        }
    }

    #[test]
    fn cell_dedup_works() {
        let mut arena = ConstraintArena::<F>::new();

        let cell_a = ProgramCell::current(0);
        let cell_b = ProgramCell::current(0);
        let cell_c = ProgramCell::next(0);

        let id_a = arena.cell(cell_a);
        let id_b = arena.cell(cell_b);
        let id_c = arena.cell(cell_c);

        // Same cell -> same ExprId
        assert_eq!(id_a, id_b);
        // Different cell -> different ExprId
        assert_ne!(id_a, id_c);
        // Only 2 nodes allocated, not 3
        assert_eq!(arena.len(), 2);
    }

    #[test]
    fn dag_sharing_reduces_node_count() {
        let mut arena = ConstraintArena::<F>::new();

        // Build a shared sub-expression: theta = a + b + c
        let a = arena.cell(ProgramCell::current(0));
        let b = arena.cell(ProgramCell::current(1));
        let c = arena.cell(ProgramCell::current(2));
        let theta = arena.sum(vec![a, b, c]);

        // Use theta in two different constraints (DAG sharing)
        let d = arena.cell(ProgramCell::current(3));
        let expr1 = arena.mul(theta, d);
        let expr2 = arena.mul(theta, a); // reuses both theta and a

        let dag_node_count = arena.len();
        // 4 cells + 1 sum + 2 muls = 7 nodes
        assert_eq!(dag_node_count, 7);

        // Without sharing (tree), theta would be duplicated:
        // 4 cells + 2 sums + 2 muls = 8 nodes minimum
        // (and the cells inside the second theta would also duplicate)
        // Actually: 3 cells * 2 + 1 extra cell + 2 sums + 2 muls = 10 nodes
        // DAG: 7 < 10 tree nodes
        assert!(dag_node_count < 10);

        // Verify both expressions reference the same theta
        match arena.get(expr1) {
            ConstraintExpr::Mul(lhs, _) => assert_eq!(*lhs, theta),
            _ => panic!("Expected Mul"),
        }

        match arena.get(expr2) {
            ConstraintExpr::Mul(lhs, rhs) => {
                assert_eq!(*lhs, theta);
                assert_eq!(*rhs, a);
            }
            _ => panic!("Expected Mul"),
        }
    }

    #[test]
    fn default_constraint_ast_node_count_matches_flat() {
        let program = TestFibProgram;
        let flat = program.constraints();
        let ast = program.constraint_ast();

        // Count total cells across flat constraints
        let mut flat_cell_count = 0;
        for c in &flat {
            for t in &c.terms {
                flat_cell_count += t.poly_ind.len();
            }
        }

        // With dedup, AST cell nodes <= flat cell references
        let ast_cell_count = ast
            .arena
            .nodes
            .iter()
            .filter(|n| matches!(n, ConstraintExpr::Cell(_)))
            .count();

        assert!(ast_cell_count <= flat_cell_count);

        // Fib shares ProgramCell::current(2) across all terms.
        // 5 total term-cells reference current(2), but dedup
        // means only 1 Cell node for it.
        // Unique cells: current(0), current(1), current(2), next(0), next(1) = 5
        assert_eq!(ast_cell_count, 5);
    }

    #[test]
    fn empty_constraint_produces_empty_ast() {
        #[derive(Clone)]
        struct EmptyProgram;

        impl Air<F> for EmptyProgram {
            fn num_columns(&self) -> usize {
                0
            }

            fn column_layout(&self) -> &[ColumnType] {
                &[]
            }

            fn constraint_ast(&self) -> ConstraintAst<F> {
                ConstraintSystem::<F>::new().build()
            }
        }

        impl Program<F> for EmptyProgram {}

        let ast = EmptyProgram.constraint_ast();
        assert!(ast.roots.is_empty());
        assert!(ast.arena.is_empty());
    }

    #[test]
    fn single_term_constraint_no_sum_wrapper() {
        #[derive(Clone)]
        struct SingleTermProgram;

        impl Air<F> for SingleTermProgram {
            fn num_columns(&self) -> usize {
                2
            }

            fn column_layout(&self) -> &[ColumnType] {
                &[ColumnType::B128, ColumnType::B128]
            }

            fn constraint_ast(&self) -> ConstraintAst<F> {
                let cs = ConstraintSystem::<F>::new();
                cs.constrain(cs.col(0) * cs.col(1));

                cs.build()
            }
        }

        impl Program<F> for SingleTermProgram {}

        let ast = SingleTermProgram.constraint_ast();
        assert_eq!(ast.roots.len(), 1);

        // Single term: Const * cell0 * cell1 -> chain of Mul, no Sum
        match ast.arena.get(ast.roots[0]) {
            ConstraintExpr::Mul(_, _) => {} // correct
            other => panic!("Expected Mul for single-term, got {:?}", other),
        }
    }

    // =========================================================
    // ConstraintAst method tests
    // =========================================================

    #[test]
    fn max_degree_fibonacci() {
        let program = TestFibProgram;
        let ast = program.constraint_ast();

        // Fib constraints: q * next_a (degree 2), q * curr_b (degree 2)
        // Default AST adds Const(ONE) * cell * cell per term → degree 2 + Const
        // Const has degree 0, so each term is Mul chain: 0 + 1 + 1 = 2
        // The AST max degree should match the flat form.
        let flat = program.constraints();
        let flat_max = flat
            .iter()
            .flat_map(|c| c.terms.iter())
            .map(|t| t.poly_ind.len())
            .max()
            .unwrap_or(0);

        assert_eq!(ast.max_degree(), flat_max);
    }

    #[test]
    fn max_degree_empty() {
        let ast = ConstraintAst::<F> {
            arena: ConstraintArena::new(),
            roots: Vec::new(),
            labels: Vec::new(),
        };
        assert_eq!(ast.max_degree(), 0);
    }

    #[test]
    fn max_degree_builder_mul_chain() {
        use crate::constraint::builder::ConstraintSystem;

        let cs = ConstraintSystem::<F>::new();
        let a = cs.col(0);
        let b = cs.col(1);
        let c = cs.col(2);

        // a * b * c = degree 3
        cs.constrain(a * b * c);
        // a + b = degree 1
        cs.constrain(a + b);

        let ast = cs.build();
        assert_eq!(ast.max_degree(), 3);
    }

    #[test]
    fn to_constraints_roundtrip_fibonacci() {
        let program = TestFibProgram;
        let ast = program.constraint_ast();
        let flat_from_ast = ast.to_constraints();
        let flat_direct = program.constraints();

        // Same number of constraints
        assert_eq!(flat_from_ast.len(), flat_direct.len());

        // Same number of terms per constraint
        for (a, d) in flat_from_ast.iter().zip(flat_direct.iter()) {
            assert_eq!(a.terms.len(), d.terms.len());
        }
    }

    #[test]
    fn to_constraints_from_builder() {
        use crate::constraint::builder::ConstraintSystem;

        let cs = ConstraintSystem::<F>::new();
        let a = cs.col(0);
        let b = cs.col(1);

        // a + b = 0  →  should produce 2 flat terms: (ONE, [a]) + (ONE, [b])
        cs.constrain(a + b);

        let ast = cs.build();
        let flat = ast.to_constraints();

        assert_eq!(flat.len(), 1);
        assert_eq!(flat[0].terms.len(), 2);

        // Both terms should have exactly 1 cell each
        for term in &flat[0].terms {
            assert_eq!(term.coeff, F::ONE);
            assert_eq!(term.poly_ind.len(), 1);
        }
    }

    #[test]
    fn evaluate_simple_constraint() {
        use crate::constraint::builder::ConstraintSystem;
        use hekate_math::Flat;

        let cs = ConstraintSystem::<F>::new();
        let a = cs.col(0);
        let b = cs.col(1);

        // a + b = 0
        cs.constrain(a + b);

        let ast = cs.build();

        // Evaluate at a=3, b=3 -> 3 + 3 = 0 in GF(2^k) (XOR)
        let current = vec![
            Flat::from_raw(F::from(3u128)),
            Flat::from_raw(F::from(3u128)),
        ];
        let next = vec![Flat::from_raw(F::ZERO); 2];

        let evals = ast.evaluate(&current, &next);
        assert_eq!(evals.len(), 1);
        assert_eq!(evals[0], Flat::from_raw(F::ZERO)); // 3 XOR 3 = 0

        // Evaluate at a=3, b=5 -> 3 XOR 5 = 6 ≠ 0
        let current2 = vec![
            Flat::from_raw(F::from(3u128)),
            Flat::from_raw(F::from(5u128)),
        ];
        let evals2 = ast.evaluate(&current2, &next);
        assert_ne!(evals2[0], Flat::from_raw(F::ZERO));
    }

    #[test]
    fn evaluate_into_matches_evaluate() {
        let cs = ConstraintSystem::<F>::new();

        let a = cs.col(0);
        let b = cs.col(1);
        let na = cs.next(0);

        cs.constrain(a + b);
        cs.constrain(a * b);
        cs.constrain(na + a);

        let ast = cs.build();
        let zero = Flat::from_raw(F::ZERO);

        let current = vec![
            Flat::from_raw(F::from(3u128)),
            Flat::from_raw(F::from(5u128)),
        ];
        let next = vec![Flat::from_raw(F::from(7u128)), zero];

        let expected = ast.evaluate(&current, &next);

        let mut buf = Vec::new();
        ast.evaluate_into(&current, &next, &mut buf);

        for (i, root) in ast.roots.iter().enumerate() {
            assert_eq!(buf[root.0 as usize], expected[i]);
        }
    }
}
