// SPDX-License-Identifier: Apache-2.0
// This file is part of the hekate project.
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

//! Constraint Builder DSL.
//!
//! Provides `ConstraintSystem` and `Expr` for writing
//! algebraic constraints with operator overloading.
//!
//! # Example
//!
//! ```ignore
//! let cs = ConstraintSystem::<F>::new();
//! let [a, b, q] = [cs.col(0), cs.col(1), cs.col(2)];
//! let [na, nb] = [cs.next(0), cs.next(1)];
//!
//! cs.constrain(q * (na + b));       // next_a = b
//! cs.constrain(q * (nb + a + b));   // next_b = a + b
//!
//! let ast = cs.build();
//! ```
//!
//! `Expr` is `Copy` — same expression reused in
//! multiple constraints shares the underlying DAG node.
//! Cell references are auto-deduplicated.
//!
//! `Sub` delegates to `Add` because in GF(2^k),
//! subtraction is addition (XOR). This is correct
//! for binary tower fields only.

use crate::ProgramCell;
use crate::constraint::{ConstraintArena, ConstraintAst, ExprId};
use alloc::vec::Vec;
use core::cell::RefCell;
use core::ops::{Add, Mul, Sub};
use hekate_math::TowerField;

// =================================================================
// CONSTRAINT SYSTEM
// =================================================================

/// Builder context for algebraic constraints.
pub struct ConstraintSystem<F: TowerField> {
    inner: RefCell<Inner<F>>,
}

struct Inner<F: TowerField> {
    arena: ConstraintArena<F>,
    roots: Vec<ExprId>,
    labels: Vec<Option<&'static str>>,
}

impl<F: TowerField> ConstraintSystem<F> {
    /// Create a new constraint system.
    pub fn new() -> Self {
        Self {
            inner: RefCell::new(Inner {
                arena: ConstraintArena::new(),
                roots: Vec::new(),
                labels: Vec::new(),
            }),
        }
    }

    /// Create a builder from an existing AST.
    pub fn from_ast(ast: ConstraintAst<F>) -> Self {
        Self {
            inner: RefCell::new(Inner {
                arena: ast.arena,
                roots: ast.roots,
                labels: ast.labels,
            }),
        }
    }

    // ===========================================================
    // Cell References
    // ===========================================================

    /// Reference to a column in the current row.
    pub fn col(&self, idx: usize) -> Expr<'_, F> {
        let id = self
            .inner
            .borrow_mut()
            .arena
            .cell(ProgramCell::current(idx));
        Expr { id, cs: self }
    }

    /// Reference to a column in the next row.
    pub fn next(&self, idx: usize) -> Expr<'_, F> {
        let id = self.inner.borrow_mut().arena.cell(ProgramCell::next(idx));
        Expr { id, cs: self }
    }

    // ===========================================================
    // Constants
    // ===========================================================

    /// Field constant.
    pub fn constant(&self, val: F) -> Expr<'_, F> {
        let id = self.inner.borrow_mut().arena.constant(val);
        Expr { id, cs: self }
    }

    /// The multiplicative identity.
    pub fn one(&self) -> Expr<'_, F> {
        self.constant(F::ONE)
    }

    // ===========================================================
    // Arithmetic
    // ===========================================================

    /// Scalar multiplication:
    /// `coeff * expr`.
    ///
    /// Use this for powers of 2, coefficients, etc.
    /// Orphan rules prevent implementing `F * Expr`
    /// via operator overloading.
    pub fn scale(&self, coeff: F, expr: Expr<'_, F>) -> Expr<'_, F> {
        let id = self.inner.borrow_mut().arena.scale(coeff, expr.id);
        Expr { id, cs: self }
    }

    /// N-ary sum. More efficient than chaining binary `+`
    /// for linear combinations (avoids deep Add chains).
    pub fn sum(&self, children: &[Expr<'_, F>]) -> Expr<'_, F> {
        let ids: Vec<ExprId> = children.iter().map(|e| e.id).collect();
        let id = self.inner.borrow_mut().arena.sum(ids);

        Expr { id, cs: self }
    }

    // ===========================================================
    // Constraint Registration
    // ===========================================================

    /// Register a constraint:
    /// `expr = 0`.
    ///
    /// The expression must evaluate to zero
    /// for every valid row in the execution trace.
    pub fn constrain(&self, expr: Expr<'_, F>) {
        let mut inner = self.inner.borrow_mut();
        inner.roots.push(expr.id);
        inner.labels.push(None);
    }

    pub fn constrain_named(&self, label: &'static str, expr: Expr<'_, F>) {
        let mut inner = self.inner.borrow_mut();
        inner.roots.push(expr.id);
        inner.labels.push(Some(label));
    }

    // ===========================================================
    // Built-in Gadgets
    // ===========================================================

    /// Assert that `s` is boolean:
    /// `s * (s + 1) = 0`.
    ///
    /// In GF(2^k), `s + 1` equals `s - 1`.
    /// Enforces `s ∈ {0, 1}`.
    pub fn assert_boolean(&self, s: Expr<'_, F>) {
        // s * s + s = 0  (expanded form of s*(s+1)=0)
        let sq = s * s;
        let expr = sq + s;

        self.constrain_named("boolean", expr);
    }

    /// Assert that `body = 0` whenever `sel = 1`.
    ///
    /// Registers `sel * body = 0`.
    /// When `sel = 0`, the constraint
    /// is trivially satisfied.
    pub fn assert_zero_when(&self, sel: Expr<'_, F>, body: Expr<'_, F>) {
        self.constrain_named("zero_when", sel * body);
    }

    /// Assert that exactly
    /// one selector is active.
    /// Enforces:
    /// sum(selectors) = 1.
    ///
    /// In GF(2^k):
    /// sum(s_i) + 1 = 0.
    pub fn assert_one_hot(&self, selectors: &[Expr<'_, F>]) {
        let s = self.sum(selectors);
        let one = self.one();

        self.constrain_named("one_hot", s + one);
    }

    /// Emit the `s_send · s_recv = 0` mutex root
    /// plus boolean checks on both selectors.
    pub fn assert_paired_bus_mutex(&self, s_send: usize, s_recv: usize) {
        let send = self.col(s_send);
        let recv = self.col(s_recv);

        self.assert_boolean(send);
        self.assert_boolean(recv);

        self.constrain_named("paired_bus_mutex", send * recv);
    }

    // ===========================================================
    // Compile
    // ===========================================================

    /// Consume the builder and
    /// produce a `ConstraintAst`.
    ///
    /// All `Expr` handles must be
    /// dropped before calling this
    /// (enforced by the borrow checker,
    /// `build` takes `self`).
    pub fn build(self) -> ConstraintAst<F> {
        let inner = self.inner.into_inner();
        ConstraintAst {
            arena: inner.arena,
            roots: inner.roots,
            labels: inner.labels,
        }
    }
}

impl<F: TowerField> Default for ConstraintSystem<F> {
    fn default() -> Self {
        Self::new()
    }
}

// =================================================================
// EXPRESSION HANDLE
// =================================================================

/// Lightweight handle to a DAG node in a `ConstraintSystem`.
///
/// Supports `+`, `*`, `-` via operator overloading.
/// `Sub` delegates to `Add` (correct for GF(2^k) only).
#[derive(Clone, Copy)]
pub struct Expr<'a, F: TowerField> {
    pub(crate) id: ExprId,
    pub(crate) cs: &'a ConstraintSystem<F>,
}

// a + b
impl<'a, F: TowerField> Add for Expr<'a, F> {
    type Output = Expr<'a, F>;

    fn add(self, rhs: Self) -> Self::Output {
        let id = self.cs.inner.borrow_mut().arena.add(self.id, rhs.id);
        Expr { id, cs: self.cs }
    }
}

// a * b
impl<'a, F: TowerField> Mul for Expr<'a, F> {
    type Output = Expr<'a, F>;

    fn mul(self, rhs: Self) -> Self::Output {
        let id = self.cs.inner.borrow_mut().arena.mul(self.id, rhs.id);
        Expr { id, cs: self.cs }
    }
}

// a - b  (in GF(2^k), subtraction = addition)
impl<'a, F: TowerField> Sub for Expr<'a, F> {
    type Output = Expr<'a, F>;

    fn sub(self, rhs: Self) -> Self::Output {
        // In characteristic 2:
        // -1 = 1, so a - b = a + b = a XOR b.
        // This is correct for binary tower fields only.
        let id = self.cs.inner.borrow_mut().arena.add(self.id, rhs.id);
        Expr { id, cs: self.cs }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::ConstraintExpr;
    use hekate_math::Block128;

    type F = Block128;

    #[test]
    fn basic_fibonacci_builder() {
        let cs = ConstraintSystem::<F>::new();

        let a = cs.col(0);
        let b = cs.col(1);
        let q = cs.col(2);
        let na = cs.next(0);
        let nb = cs.next(1);

        // q * (next_a + b) = 0
        cs.constrain(q * (na + b));
        // q * (next_b + a + b) = 0
        cs.constrain(q * (nb + a + b));

        let ast = cs.build();

        assert_eq!(ast.roots.len(), 2);
        assert!(!ast.arena.is_empty());

        // Both roots should be Mul (selector * sum)
        for &root in &ast.roots {
            match ast.arena.get(root) {
                ConstraintExpr::Mul(_, _) => {}
                other => panic!("Expected Mul root, got {:?}", other),
            }
        }
    }

    #[test]
    fn cell_dedup_through_builder() {
        let cs = ConstraintSystem::<F>::new();

        let a1 = cs.col(0);
        let a2 = cs.col(0);
        let b = cs.col(1);

        // Same column → same ExprId
        assert_eq!(a1.id, a2.id);
        // Different column → different ExprId
        assert_ne!(a1.id, b.id);
    }

    #[test]
    fn sub_equals_add_in_char2() {
        let cs = ConstraintSystem::<F>::new();

        let a = cs.col(0);
        let b = cs.col(1);

        let sum = a + b;
        let diff = a - b;

        // In GF(2^k), a + b == a - b
        // Both should produce Add nodes with same children
        let ast_sum = cs.inner.borrow();
        match (ast_sum.arena.get(sum.id), ast_sum.arena.get(diff.id)) {
            (ConstraintExpr::Add(la, ra), ConstraintExpr::Add(lb, rb)) => {
                assert_eq!(la, lb);
                assert_eq!(ra, rb);
            }
            _ => panic!("Expected Add nodes for both + and -"),
        }
    }

    #[test]
    fn assert_boolean_structure() {
        let cs = ConstraintSystem::<F>::new();
        let s = cs.col(5);

        cs.assert_boolean(s);

        let ast = cs.build();
        assert_eq!(ast.roots.len(), 1);

        // Root should be Add(Mul(s,s), s) = s² + s
        match ast.arena.get(ast.roots[0]) {
            ConstraintExpr::Add(lhs, rhs) => {
                // lhs = s * s
                match ast.arena.get(*lhs) {
                    ConstraintExpr::Mul(a, b) => {
                        assert_eq!(a, b); // same cell
                    }
                    other => panic!("Expected Mul for s², got {:?}", other),
                }

                // rhs = s
                match ast.arena.get(*rhs) {
                    ConstraintExpr::Cell(cell) => {
                        assert_eq!(cell.col_idx, 5);
                        assert!(!cell.next_row);
                    }
                    other => panic!("Expected Cell for s, got {:?}", other),
                }
            }
            other => panic!("Expected Add for s²+s, got {:?}", other),
        }
    }

    #[test]
    fn assert_zero_when_structure() {
        let cs = ConstraintSystem::<F>::new();
        let sel = cs.col(0);
        let body = cs.col(1) + cs.col(2);

        cs.assert_zero_when(sel, body);

        let ast = cs.build();
        assert_eq!(ast.roots.len(), 1);

        // Root = Mul(sel, Add(col1, col2))
        match ast.arena.get(ast.roots[0]) {
            ConstraintExpr::Mul(_, _) => {}
            other => panic!("Expected Mul, got {:?}", other),
        }
    }

    #[test]
    fn scale_produces_scale_node() {
        let cs = ConstraintSystem::<F>::new();
        let a = cs.col(0);
        let scaled = cs.scale(F::from(8u128), a);

        // Save ids before build() consumes cs
        let a_id = a.id;
        let scaled_id = scaled.id;

        let ast = cs.build();
        match ast.arena.get(scaled_id) {
            ConstraintExpr::Scale(coeff, inner) => {
                assert_eq!(*coeff, F::from(8u128));
                assert_eq!(*inner, a_id);
            }
            other => panic!("Expected Scale, got {:?}", other),
        }
    }

    #[test]
    fn sum_produces_sum_node() {
        let cs = ConstraintSystem::<F>::new();
        let a = cs.col(0);
        let b = cs.col(1);
        let c = cs.col(2);
        let s = cs.sum(&[a, b, c]);

        // Save ids before build() consumes cs
        let (a_id, b_id, c_id) = (a.id, b.id, c.id);

        let s_id = s.id;
        let ast = cs.build();

        match ast.arena.get(s_id) {
            ConstraintExpr::Sum(children) => {
                assert_eq!(children.len(), 3);
                assert_eq!(children[0], a_id);
                assert_eq!(children[1], b_id);
                assert_eq!(children[2], c_id);
            }
            other => panic!("Expected Sum, got {:?}", other),
        }
    }

    #[test]
    fn dag_sharing_via_expr_reuse() {
        let cs = ConstraintSystem::<F>::new();

        let a = cs.col(0);
        let b = cs.col(1);
        let c = cs.col(2);

        // Shared sub-expression
        let theta = cs.sum(&[a, b, c]);

        // Use theta in two constraints
        let d = cs.col(3);
        cs.constrain(theta * d);
        cs.constrain(theta * a);

        let ast = cs.build();
        assert_eq!(ast.roots.len(), 2);

        // Both roots reference the same theta node
        match (ast.arena.get(ast.roots[0]), ast.arena.get(ast.roots[1])) {
            (ConstraintExpr::Mul(lhs0, _), ConstraintExpr::Mul(lhs1, _)) => {
                assert_eq!(lhs0, lhs1); // same theta ExprId
            }
            _ => panic!("Expected Mul roots"),
        }
    }

    #[test]
    fn empty_system_produces_empty_ast() {
        let cs = ConstraintSystem::<F>::new();
        let ast = cs.build();
        assert!(ast.roots.is_empty());
        assert!(ast.arena.is_empty());
    }

    #[test]
    fn builder_matches_manual_structure() {
        // Build via builder: q * (next_a + b)
        let cs = ConstraintSystem::<F>::new();
        let _a = cs.col(0);
        let b = cs.col(1);
        let q = cs.col(2);
        let na = cs.next(0);

        cs.constrain(q * (na + b));

        let ast = cs.build();

        // Verify: Mul(Cell(2,curr), Add(Cell(0,next), Cell(1,curr)))
        assert_eq!(ast.roots.len(), 1);

        match ast.arena.get(ast.roots[0]) {
            ConstraintExpr::Mul(lhs, rhs) => {
                match ast.arena.get(*lhs) {
                    ConstraintExpr::Cell(cell) => {
                        assert_eq!(cell.col_idx, 2);
                        assert!(!cell.next_row);
                    }
                    other => panic!("Expected Cell for q, got {:?}", other),
                }
                match ast.arena.get(*rhs) {
                    ConstraintExpr::Add(a, b) => {
                        match ast.arena.get(*a) {
                            ConstraintExpr::Cell(cell) => {
                                assert_eq!(cell.col_idx, 0);
                                assert!(cell.next_row);
                            }
                            other => panic!("Expected Cell for next_a, got {:?}", other),
                        }
                        match ast.arena.get(*b) {
                            ConstraintExpr::Cell(cell) => {
                                assert_eq!(cell.col_idx, 1);
                                assert!(!cell.next_row);
                            }
                            other => panic!("Expected Cell for b, got {:?}", other),
                        }
                    }
                    other => panic!("Expected Add, got {:?}", other),
                }
            }
            other => panic!("Expected Mul root, got {:?}", other),
        }
    }

    #[test]
    fn labels_round_trip_through_build() {
        let cs = ConstraintSystem::<F>::new();
        let a = cs.col(0);
        let b = cs.col(1);

        cs.constrain(a + b);
        cs.constrain_named("transition", a * b);
        cs.assert_boolean(a);

        let ast = cs.build();

        assert_eq!(ast.roots.len(), 3);
        assert_eq!(ast.labels.len(), 3);
        assert_eq!(ast.labels[0], None);
        assert_eq!(ast.labels[1], Some("transition"));
        assert_eq!(ast.labels[2], Some("boolean"));
    }

    #[test]
    fn labels_preserved_through_merge() {
        let cs1 = ConstraintSystem::<F>::new();
        cs1.constrain_named("first", cs1.col(0));

        let mut ast1 = cs1.build();

        let cs2 = ConstraintSystem::<F>::new();

        cs2.constrain(cs2.col(0));
        cs2.constrain_named("second", cs2.col(1));

        let ast2 = cs2.build();

        ast1.merge(ast2);

        assert_eq!(ast1.roots.len(), 3);
        assert_eq!(ast1.labels.len(), 3);
        assert_eq!(ast1.labels[0], Some("first"));
        assert_eq!(ast1.labels[1], None);
        assert_eq!(ast1.labels[2], Some("second"));
    }

    #[test]
    fn labels_preserved_through_from_ast() {
        let cs = ConstraintSystem::<F>::new();
        cs.constrain_named("original", cs.col(0));

        let ast = cs.build();

        let cs2 = ConstraintSystem::from_ast(ast);
        cs2.constrain_named("added", cs2.col(1));

        let ast2 = cs2.build();

        assert_eq!(ast2.labels.len(), 2);
        assert_eq!(ast2.labels[0], Some("original"));
        assert_eq!(ast2.labels[1], Some("added"));
    }

    #[test]
    fn builtin_gadgets_have_labels() {
        let cs = ConstraintSystem::<F>::new();

        let a = cs.col(0);
        let b = cs.col(1);

        cs.assert_boolean(a);
        cs.assert_zero_when(a, b);
        cs.assert_one_hot(&[a, b]);

        let ast = cs.build();

        assert_eq!(ast.labels.len(), 3);
        assert_eq!(ast.labels[0], Some("boolean"));
        assert_eq!(ast.labels[1], Some("zero_when"));
        assert_eq!(ast.labels[2], Some("one_hot"));
    }
}
