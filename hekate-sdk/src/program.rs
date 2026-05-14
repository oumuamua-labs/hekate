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

use alloc::string::String;
use alloc::vec::Vec;
use hekate_core::errors;
use hekate_core::trace::ColumnType;
use hekate_math::TowerField;
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::{BoundaryConstraint, ConstraintAst};
use hekate_program::expander::VirtualExpander;
use hekate_program::permutation::PermutationCheckSpec;
use hekate_program::{Air, InlineKernelHint, LagrangePin, Program};

use crate::wire::bundle::DeserializedBundle;

/// Program reconstructed from a
/// deserialized `ProgramBundle`.
///
/// Implements `Air<F>` + `Program<F>` so the
/// prover and verifier can consume it directly.
#[derive(Clone)]
pub struct BundleProgram<F: TowerField> {
    constraint_ast: ConstraintAst<F>,
    column_layout: Vec<ColumnType>,
    virtual_column_layout: Vec<ColumnType>,
    boundary_constraints: Vec<BoundaryConstraint<F>>,
    permutation_checks: Vec<(String, PermutationCheckSpec)>,
    virtual_expander: Option<VirtualExpander>,
    chiplet_defs: Vec<ChipletDef<F>>,
    inline_chiplets: Vec<ChipletDef<F>>,
    inline_chiplet_kernels: Vec<InlineKernelHint>,
    num_columns: usize,
    num_public_inputs: usize,
    lagrange_pins: Vec<LagrangePin>,
}

impl<F: TowerField> BundleProgram<F> {
    pub fn from_bundle(bundle: &DeserializedBundle<F>) -> Self {
        Self {
            constraint_ast: bundle.constraint_ast.clone(),
            column_layout: bundle.column_layout.clone(),
            virtual_column_layout: bundle.virtual_column_layout.clone(),
            boundary_constraints: bundle.boundary_constraints.clone(),
            permutation_checks: bundle.permutation_checks.clone(),
            virtual_expander: bundle.virtual_expander.clone(),
            chiplet_defs: bundle.chiplet_defs.clone(),
            inline_chiplets: bundle.inline_chiplets.clone(),
            inline_chiplet_kernels: bundle.inline_chiplet_kernels.clone(),
            num_columns: bundle.num_columns,
            num_public_inputs: bundle.num_public_inputs,
            lagrange_pins: bundle.lagrange_pins.clone(),
        }
    }
}

impl<F: TowerField> Air<F> for BundleProgram<F> {
    fn num_columns(&self) -> usize {
        self.num_columns
    }

    fn boundary_constraints(&self) -> Vec<BoundaryConstraint<F>> {
        self.boundary_constraints.clone()
    }

    fn column_layout(&self) -> &[ColumnType] {
        &self.column_layout
    }

    fn virtual_column_layout(&self) -> &[ColumnType] {
        &self.virtual_column_layout
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        self.permutation_checks.clone()
    }

    fn lagrange_pinned_columns(&self) -> Vec<LagrangePin> {
        self.lagrange_pins.clone()
    }

    fn virtual_expander(&self) -> Option<&VirtualExpander> {
        self.virtual_expander.as_ref()
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        self.constraint_ast.clone()
    }

    fn inline_chiplets(&self) -> errors::Result<Vec<ChipletDef<F>>> {
        Ok(self.inline_chiplets.clone())
    }

    fn inline_chiplet_kernels(&self) -> Vec<InlineKernelHint> {
        self.inline_chiplet_kernels.clone()
    }
}

impl<F: TowerField> Program<F> for BundleProgram<F> {
    fn num_public_inputs(&self) -> usize {
        self.num_public_inputs
    }

    fn chiplet_defs(&self) -> errors::Result<Vec<ChipletDef<F>>> {
        Ok(self.chiplet_defs.clone())
    }
}
