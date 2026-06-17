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

use hekate_core::trace::ColumnType;
use hekate_math::{Block128, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::constraint::{BoundaryConstraint, ConstraintAst};
use hekate_program::expander::VirtualExpander;
use hekate_program::permutation::{
    BusKind, ChallengeLabel, PermutationCheckSpec, REQUEST_IDX_LABEL, Source,
};
use hekate_program::{Air, FixedColumn, FixedShape, InlineKernelHint, Program};
use hekate_sdk::program_id;

type F = Block128;

#[derive(Clone)]
struct TestChiplet {
    name: String,
    column_layout: Vec<ColumnType>,
    constraint_ast: ConstraintAst<F>,
    boundary_constraints: Vec<BoundaryConstraint<F>>,
    permutation_checks: Vec<(String, PermutationCheckSpec)>,
    fixed_columns: Vec<FixedColumn<F>>,
    expander: Option<VirtualExpander>,
}

impl TestChiplet {
    fn baseline() -> Self {
        let cs = ConstraintSystem::<F>::new();
        let a = cs.col(0);
        let b = cs.col(1);
        cs.constrain(a * b);

        Self {
            name: "test_chiplet".to_string(),
            column_layout: vec![ColumnType::B32, ColumnType::B32],
            constraint_ast: cs.build(),
            boundary_constraints: Vec::new(),
            permutation_checks: Vec::new(),
            fixed_columns: Vec::new(),
            expander: None,
        }
    }
}

impl Air<F> for TestChiplet {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn num_columns(&self) -> usize {
        self.column_layout.len()
    }

    fn boundary_constraints(&self) -> Vec<BoundaryConstraint<F>> {
        self.boundary_constraints.clone()
    }

    fn column_layout(&self) -> &[ColumnType] {
        &self.column_layout
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        self.permutation_checks.clone()
    }

    fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
        self.fixed_columns.clone()
    }

    fn virtual_expander(&self) -> Option<&VirtualExpander> {
        self.expander.as_ref()
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        self.constraint_ast.clone()
    }
}

#[derive(Clone)]
struct TestProgram {
    name: String,
    column_layout: Vec<ColumnType>,
    constraint_ast: ConstraintAst<F>,
    boundary_constraints: Vec<BoundaryConstraint<F>>,
    permutation_checks: Vec<(String, PermutationCheckSpec)>,
    fixed_columns: Vec<FixedColumn<F>>,
    expander: Option<VirtualExpander>,
    chiplets: Vec<TestChiplet>,
    inline_chiplets: Vec<TestChiplet>,
    inline_kernel_hints: Vec<InlineKernelHint>,
    num_public_inputs: usize,
    num_columns_override: Option<usize>,
}

impl TestProgram {
    fn baseline() -> Self {
        let cs = ConstraintSystem::<F>::new();
        let a = cs.col(0);
        let b = cs.col(1);
        let next_a = cs.next(0);
        cs.constrain(next_a + a + b);

        Self {
            name: "test_program".to_string(),
            column_layout: vec![ColumnType::B32, ColumnType::B32, ColumnType::Bit],
            constraint_ast: cs.build(),
            boundary_constraints: vec![BoundaryConstraint::with_public_input(1, 0, 0)],
            permutation_checks: vec![("test_bus".to_string(), paired_perm_spec(Some(2), 0, 1))],
            fixed_columns: vec![FixedColumn::last_row(0)],
            expander: None,
            chiplets: vec![TestChiplet::baseline()],
            inline_chiplets: Vec::new(),
            inline_kernel_hints: Vec::new(),
            num_public_inputs: 1,
            num_columns_override: None,
        }
    }
}

impl Air<F> for TestProgram {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn num_columns(&self) -> usize {
        self.num_columns_override
            .unwrap_or(self.column_layout.len())
    }

    fn boundary_constraints(&self) -> Vec<BoundaryConstraint<F>> {
        self.boundary_constraints.clone()
    }

    fn column_layout(&self) -> &[ColumnType] {
        &self.column_layout
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        self.permutation_checks.clone()
    }

    fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
        self.fixed_columns.clone()
    }

    fn virtual_expander(&self) -> Option<&VirtualExpander> {
        self.expander.as_ref()
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        self.constraint_ast.clone()
    }

    fn inline_chiplets(&self) -> hekate_core::errors::Result<Vec<ChipletDef<F>>> {
        self.inline_chiplets
            .iter()
            .map(ChipletDef::from_air)
            .collect()
    }

    fn inline_chiplet_kernels(&self) -> Vec<InlineKernelHint> {
        self.inline_kernel_hints.clone()
    }
}

impl Program<F> for TestProgram {
    fn num_public_inputs(&self) -> usize {
        self.num_public_inputs
    }

    fn chiplet_defs(&self) -> hekate_core::errors::Result<Vec<ChipletDef<F>>> {
        self.chiplets.iter().map(ChipletDef::from_air).collect()
    }
}

fn paired_perm_spec(
    recv_selector: Option<usize>,
    send_sel_col: usize,
    _recv_sel_col: usize,
) -> PermutationCheckSpec {
    PermutationCheckSpec {
        kind: BusKind::Permutation,
        sources: vec![
            (Source::Column(0), b"col_0" as ChallengeLabel),
            (Source::RowIndexLeBytes(4), REQUEST_IDX_LABEL),
        ],
        selector: Some(send_sel_col),
        recv_selector,
        clock_waiver: None,
    }
}

fn id(p: &TestProgram) -> [u8; 32] {
    program_id::<F, _>(p).expect("program_id")
}

#[test]
fn program_id_deterministic_across_calls() {
    let p = TestProgram::baseline();
    assert_eq!(id(&p), id(&p));
}

#[test]
fn program_id_equal_for_two_baseline_clones() {
    let p1 = TestProgram::baseline();
    let p2 = TestProgram::baseline();

    assert_eq!(id(&p1), id(&p2));
}

#[test]
fn mutate_main_layout_changes_hash() {
    let mut p = TestProgram::baseline();
    let h1 = id(&p);

    p.column_layout[0] = ColumnType::B64;

    assert_ne!(h1, id(&p));
}

#[test]
fn mutate_main_constraint_ast_changes_hash() {
    let mut p = TestProgram::baseline();
    let h1 = id(&p);

    let cs = ConstraintSystem::<F>::new();

    let a = cs.col(0);
    cs.constrain(a + a);

    p.constraint_ast = cs.build();

    assert_ne!(h1, id(&p));
}

#[test]
fn mutate_main_boundary_col_idx_changes_hash() {
    let mut p = TestProgram::baseline();
    let h1 = id(&p);

    p.boundary_constraints = vec![BoundaryConstraint::with_public_input(2, 0, 0)];

    assert_ne!(h1, id(&p));
}

#[test]
fn mutate_main_boundary_target_changes_hash() {
    let mut p = TestProgram::baseline();
    let h1 = id(&p);

    p.boundary_constraints = vec![BoundaryConstraint::with_constant(1, 0, F::ONE)];

    assert_ne!(h1, id(&p));
}

#[test]
fn mutate_main_perm_kind_changes_hash() {
    let mut p = TestProgram::baseline();
    let h1 = id(&p);

    p.permutation_checks[0].1.kind = BusKind::Lookup;

    assert_ne!(h1, id(&p));
}

#[test]
fn mutate_main_perm_selector_changes_hash() {
    let mut p = TestProgram::baseline();
    let h1 = id(&p);

    p.permutation_checks[0].1.selector = Some(7);

    assert_ne!(h1, id(&p));
}

#[test]
fn mutate_main_perm_clock_waiver_changes_hash() {
    let mut p = TestProgram::baseline();
    let h1 = id(&p);

    p.permutation_checks[0].1.clock_waiver =
        Some("see hekate-sdk/tests/program_structural_hash.rs: synthetic test waiver".to_string());

    assert_ne!(h1, id(&p));
}

#[test]
fn mutate_main_perm_bus_id_changes_hash() {
    let mut p = TestProgram::baseline();
    let h1 = id(&p);

    p.permutation_checks[0].0 = "different_bus".to_string();

    assert_ne!(h1, id(&p));
}

#[test]
fn mutate_main_perm_source_changes_hash() {
    let mut p = TestProgram::baseline();
    let h1 = id(&p);

    p.permutation_checks[0].1.sources[0].0 = Source::Column(99);

    assert_ne!(h1, id(&p));
}

#[test]
fn mutate_main_fixed_column_col_changes_hash() {
    let mut p = TestProgram::baseline();
    let h1 = id(&p);

    p.fixed_columns = vec![FixedColumn::last_row(2)];

    assert_ne!(h1, id(&p));
}

#[test]
fn mutate_main_fixed_column_shape_changes_hash() {
    let mut p = TestProgram::baseline();
    let h1 = id(&p);

    p.fixed_columns = vec![FixedColumn {
        col_idx: 0,
        shape: FixedShape::FirstRow,
    }];

    assert_ne!(h1, id(&p));
}

#[test]
fn mutate_main_num_columns_changes_hash() {
    let mut p = TestProgram::baseline();
    let h1 = id(&p);

    p.num_columns_override = Some(99);

    assert_ne!(h1, id(&p));
}

#[test]
fn mutate_main_num_public_inputs_changes_hash() {
    let mut p = TestProgram::baseline();
    let h1 = id(&p);

    p.num_public_inputs = 42;
    assert_ne!(h1, id(&p));
}

#[test]
fn mutate_main_virtual_expander_toggle_changes_hash() {
    let mut p = TestProgram::baseline();
    p.column_layout = vec![ColumnType::B64];

    let h1 = id(&p);

    let exp = VirtualExpander::new()
        .expand_bits(1, ColumnType::B64)
        .build()
        .expect("expander build");
    p.expander = Some(exp);

    assert_ne!(h1, id(&p));
}

#[test]
fn mutate_chiplet_count_changes_hash() {
    let mut p = TestProgram::baseline();
    let h1 = id(&p);

    p.chiplets.push(TestChiplet::baseline());

    assert_ne!(h1, id(&p));
}

#[test]
fn mutate_chiplet_name_changes_hash() {
    let mut p = TestProgram::baseline();
    let h1 = id(&p);

    p.chiplets[0].name = "renamed_chiplet".to_string();

    assert_ne!(h1, id(&p));
}

#[test]
fn mutate_chiplet_layout_changes_hash() {
    let mut p = TestProgram::baseline();
    let h1 = id(&p);

    p.chiplets[0].column_layout[0] = ColumnType::B128;

    assert_ne!(h1, id(&p));
}

#[test]
fn mutate_main_perm_recv_selector_changes_hash() {
    let mut p = TestProgram::baseline();
    let h1 = id(&p);

    p.permutation_checks[0].1.recv_selector = Some(99);

    assert_ne!(h1, id(&p));
}

#[test]
fn mutate_main_perm_recv_selector_some_to_none_changes_hash() {
    let mut p = TestProgram::baseline();
    let h1 = id(&p);

    p.permutation_checks[0].1.recv_selector = None;

    assert_ne!(h1, id(&p));
}

#[test]
fn mutate_main_inline_chiplets_changes_hash() {
    let mut p = TestProgram::baseline();
    let h1 = id(&p);

    p.inline_chiplets.push(TestChiplet::baseline());

    assert_ne!(h1, id(&p));
}

#[test]
fn mutate_main_inline_chiplet_layout_changes_hash() {
    let mut p = TestProgram::baseline();
    p.inline_chiplets.push(TestChiplet::baseline());

    let h1 = id(&p);
    p.inline_chiplets[0].column_layout[0] = ColumnType::B128;

    assert_ne!(h1, id(&p));
}

#[test]
fn mutate_main_inline_kernels_changes_hash() {
    let mut p = TestProgram::baseline();
    let h1 = id(&p);
    p.inline_kernel_hints.push(InlineKernelHint {
        chiplet_idx: 0,
        root_offset: 0,
        column_offset: 0,
    });

    assert_ne!(h1, id(&p));
}

#[test]
fn mutate_main_inline_kernel_hint_field_changes_hash() {
    let mut p = TestProgram::baseline();
    p.inline_kernel_hints.push(InlineKernelHint {
        chiplet_idx: 0,
        root_offset: 0,
        column_offset: 0,
    });

    let h1 = id(&p);
    p.inline_kernel_hints[0].chiplet_idx = 7;

    assert_ne!(h1, id(&p));
}

#[test]
fn mutate_main_name_changes_hash() {
    let mut p = TestProgram::baseline();
    let h1 = id(&p);

    p.name = "renamed_program".to_string();

    assert_ne!(h1, id(&p));
}

#[test]
fn mutate_chiplet_constraint_ast_changes_hash() {
    let mut p = TestProgram::baseline();
    let h1 = id(&p);

    let cs = ConstraintSystem::<F>::new();

    let a = cs.col(0);
    cs.constrain(a + a);

    p.chiplets[0].constraint_ast = cs.build();

    assert_ne!(h1, id(&p));
}

#[test]
fn mutate_chiplet_boundary_changes_hash() {
    let mut p = TestProgram::baseline();
    let h1 = id(&p);

    p.chiplets[0].boundary_constraints = vec![BoundaryConstraint::with_constant(0, 0, F::ONE)];

    assert_ne!(h1, id(&p));
}

#[test]
fn mutate_chiplet_perm_changes_hash() {
    let mut p = TestProgram::baseline();
    let h1 = id(&p);

    p.chiplets[0]
        .permutation_checks
        .push(("chiplet_bus".to_string(), paired_perm_spec(None, 0, 1)));

    assert_ne!(h1, id(&p));
}

#[test]
fn mutate_chiplet_fixed_column_changes_hash() {
    let mut p = TestProgram::baseline();
    let h1 = id(&p);

    p.chiplets[0].fixed_columns.push(FixedColumn::last_row(0));

    assert_ne!(h1, id(&p));
}

#[test]
fn mutate_chiplet_virtual_expander_toggle_changes_hash() {
    let mut p = TestProgram::baseline();
    p.chiplets[0].column_layout = vec![ColumnType::B64];

    let h1 = id(&p);

    let exp = VirtualExpander::new()
        .expand_bits(1, ColumnType::B64)
        .build()
        .expect("expander build");
    p.chiplets[0].expander = Some(exp);

    assert_ne!(h1, id(&p));
}

#[test]
fn mutate_inline_kernel_hint_root_offset_changes_hash() {
    let mut p = TestProgram::baseline();
    p.inline_kernel_hints.push(InlineKernelHint {
        chiplet_idx: 0,
        root_offset: 0,
        column_offset: 0,
    });

    let h1 = id(&p);
    p.inline_kernel_hints[0].root_offset = 11;

    assert_ne!(h1, id(&p));
}

#[test]
fn mutate_inline_kernel_hint_column_offset_changes_hash() {
    let mut p = TestProgram::baseline();
    p.inline_kernel_hints.push(InlineKernelHint {
        chiplet_idx: 0,
        root_offset: 0,
        column_offset: 0,
    });

    let h1 = id(&p);
    p.inline_kernel_hints[0].column_offset = 23;

    assert_ne!(h1, id(&p));
}
