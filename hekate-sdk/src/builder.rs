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
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::trace::{ColumnType, Trace};
use hekate_crypto::{DefaultHasher, Hasher};
use hekate_math::TowerField;
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::{BoundaryTarget, ConstraintAst, ConstraintExpr, ExprId};
use hekate_program::expander::ExpansionEntry;
use hekate_program::permutation::{BusKind, PermutationCheckSpec, Source};
use hekate_program::{
    Air, InlineKernelHint, LagrangePin, LagrangePoint, Program, ProgramInstance, ProgramWitness,
};

use crate::wire::bundle;

/// Serialize a program bundle with
/// deterministic matrix seed.
///
/// Derives `config.matrix_seed` from
/// program structure so prover and verifier
/// agree without manual coordination. All
/// other config fields are used as-is.
pub fn build_bundle<F, P, T>(
    program: &P,
    instance: &ProgramInstance<F>,
    witness: &ProgramWitness<F, T>,
    cfg: &Config,
) -> errors::Result<Vec<u8>>
where
    F: TowerField,
    P: Program<F>,
    T: Trace,
{
    let mut cfg = cfg.clone();
    cfg.matrix_seed = derive_matrix_seed::<F, P>(program, instance.num_rows())?;

    bundle::serialize_bundle(program, instance, witness, &cfg)
}

/// Deterministic 32-byte structural ID. Witness-independent,
/// num_rows-independent. Same program shape -> same ID.
pub fn program_id<F: TowerField, P: Program<F>>(program: &P) -> errors::Result<[u8; 32]> {
    program_structural_hash::<F, P>(program)
}

/// Hex-encoded `program_id` (64 lowercase chars, no prefix).
pub fn program_id_hex<F: TowerField, P: Program<F>>(program: &P) -> errors::Result<String> {
    const HEX: &[u8; 16] = b"0123456789abcdef";

    let bytes = program_id::<F, P>(program)?;

    let mut s = String::with_capacity(64);
    for &b in bytes.iter() {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xf) as usize] as char);
    }

    Ok(s)
}

/// Deterministic matrix seed from
/// program structure + `num_rows`.
///
/// Same program + same num_rows = same seed = same
/// expander graph on prover and verifier.
pub(crate) fn derive_matrix_seed<F: TowerField, P: Program<F>>(
    program: &P,
    num_rows: usize,
) -> errors::Result<[u8; 32]> {
    let h_struct = program_structural_hash::<F, P>(program)?;

    let mut h = DefaultHasher::new();
    h.update(b"hekate-matrix-seed-v1");
    h.update(&h_struct);
    h.update(&(num_rows as u64).to_le_bytes());

    Ok(h.finalize())
}

/// Pure shape-only hash.
pub(crate) fn program_structural_hash<F: TowerField, P: Program<F>>(
    program: &P,
) -> errors::Result<[u8; 32]> {
    let mut h = DefaultHasher::new();
    h.update(b"hekate-program-id-v1");

    let main_name = program.name();
    h.update(&(main_name.len() as u64).to_le_bytes());
    h.update(main_name.as_bytes());

    absorb_layout(&mut h, program.column_layout());

    let ast = program.constraint_ast();
    absorb_ast::<F>(&mut h, &ast);

    absorb_boundaries(&mut h, &program.boundary_constraints());
    absorb_permutation_checks(&mut h, &program.permutation_checks());

    let chiplet_defs: Vec<ChipletDef<F>> = program.chiplet_defs()?;
    h.update(&(chiplet_defs.len() as u64).to_le_bytes());

    for cd in &chiplet_defs {
        absorb_chiplet_def::<F>(&mut h, cd);
    }

    if let Some(exp) = program.virtual_expander() {
        h.update(&[1]);

        absorb_expander(&mut h, exp);
    } else {
        h.update(&[0]);
    }

    absorb_lagrange_pins(&mut h, &program.lagrange_pinned_columns());

    h.update(&(program.num_columns() as u64).to_le_bytes());
    h.update(&(program.num_public_inputs() as u64).to_le_bytes());

    let inline_chiplets = program.inline_chiplets()?;
    absorb_inline_chiplets::<F>(&mut h, &inline_chiplets, &program.inline_chiplet_kernels());

    Ok(h.finalize())
}

fn absorb_chiplet_def<F: TowerField>(h: &mut DefaultHasher, cd: &ChipletDef<F>) {
    let name = Air::<F>::name(cd);

    h.update(&(name.len() as u64).to_le_bytes());
    h.update(name.as_bytes());

    absorb_layout(h, cd.column_layout());

    let cd_ast = cd.constraint_ast();
    absorb_ast::<F>(h, &cd_ast);

    absorb_boundaries(h, &cd.boundary_constraints());
    absorb_permutation_checks(h, &cd.permutation_checks());
    absorb_lagrange_pins(h, &Air::<F>::lagrange_pinned_columns(cd));

    if let Some(exp) = cd.virtual_expander() {
        h.update(&[1]);

        absorb_expander(h, exp);
    } else {
        h.update(&[0]);
    }
}

fn absorb_inline_chiplets<F: TowerField>(
    h: &mut DefaultHasher,
    inline_chiplets: &[ChipletDef<F>],
    inline_kernel_hints: &[InlineKernelHint],
) {
    h.update(&(inline_chiplets.len() as u64).to_le_bytes());

    for cd in inline_chiplets {
        absorb_chiplet_def::<F>(h, cd);
    }

    h.update(&(inline_kernel_hints.len() as u64).to_le_bytes());

    for hint in inline_kernel_hints {
        h.update(&(hint.chiplet_idx as u64).to_le_bytes());
        h.update(&(hint.root_offset as u64).to_le_bytes());
        h.update(&(hint.column_offset as u64).to_le_bytes());
    }
}

fn absorb_layout(h: &mut DefaultHasher, layout: &[ColumnType]) {
    h.update(&(layout.len() as u64).to_le_bytes());

    for ct in layout {
        h.update(&[column_type_tag(*ct)]);
    }
}

fn absorb_ast<F: TowerField>(h: &mut DefaultHasher, ast: &ConstraintAst<F>) {
    h.update(&(ast.arena.len() as u64).to_le_bytes());

    for i in 0..ast.arena.len() {
        absorb_expr(h, ast.arena.get(ExprId(i as u32)));
    }

    h.update(&(ast.roots.len() as u64).to_le_bytes());

    for root in &ast.roots {
        h.update(&root.0.to_le_bytes());
    }
}

fn absorb_expr<F: TowerField>(h: &mut DefaultHasher, expr: &ConstraintExpr<F>) {
    match expr {
        ConstraintExpr::Cell(cell) => {
            h.update(&[0]);
            h.update(&(cell.col_idx as u32).to_le_bytes());
            h.update(&[cell.next_row as u8]);
        }
        ConstraintExpr::Const(val) => {
            h.update(&[1]);
            h.update(&val.to_bytes());
        }
        ConstraintExpr::Add(l, r) => {
            h.update(&[2]);
            h.update(&l.0.to_le_bytes());
            h.update(&r.0.to_le_bytes());
        }
        ConstraintExpr::Mul(l, r) => {
            h.update(&[3]);
            h.update(&l.0.to_le_bytes());
            h.update(&r.0.to_le_bytes());
        }
        ConstraintExpr::Scale(scalar, child) => {
            h.update(&[4]);
            h.update(&scalar.to_bytes());
            h.update(&child.0.to_le_bytes());
        }
        ConstraintExpr::Sum(children) => {
            h.update(&[5]);
            h.update(&(children.len() as u64).to_le_bytes());

            for c in children {
                h.update(&c.0.to_le_bytes());
            }
        }
    }
}

fn absorb_boundaries<F: TowerField>(
    h: &mut DefaultHasher,
    boundaries: &[hekate_program::constraint::BoundaryConstraint<F>],
) {
    h.update(&(boundaries.len() as u64).to_le_bytes());

    for bc in boundaries {
        h.update(&(bc.col_idx as u64).to_le_bytes());
        h.update(&(bc.row_idx as u64).to_le_bytes());

        match &bc.target {
            BoundaryTarget::PublicInput(idx) => {
                h.update(&[0]);
                h.update(&(*idx as u64).to_le_bytes());
            }
            BoundaryTarget::Constant(v) => {
                h.update(&[1]);
                h.update(&v.to_bytes());
            }
        }
    }
}

fn absorb_lagrange_pins(h: &mut DefaultHasher, pins: &[LagrangePin]) {
    h.update(&(pins.len() as u64).to_le_bytes());

    for pin in pins {
        h.update(&(pin.col_idx as u64).to_le_bytes());

        match &pin.point {
            LagrangePoint::LastRow => h.update(&[0]),
            LagrangePoint::FirstRow => h.update(&[1]),
            LagrangePoint::Custom(bits) => {
                h.update(&[2]);
                h.update(&(bits.len() as u64).to_le_bytes());

                for &b in bits {
                    h.update(&[b as u8]);
                }
            }
        }
    }
}

fn absorb_permutation_checks(h: &mut DefaultHasher, checks: &[(String, PermutationCheckSpec)]) {
    h.update(&(checks.len() as u64).to_le_bytes());

    for (bus_id, spec) in checks {
        h.update(&(bus_id.len() as u64).to_le_bytes());
        h.update(bus_id.as_bytes());

        absorb_perm_spec(h, spec);
    }
}

fn absorb_perm_spec(h: &mut DefaultHasher, spec: &PermutationCheckSpec) {
    h.update(&[match spec.kind {
        BusKind::Permutation => 0,
        BusKind::Lookup => 1,
    }]);

    h.update(&(spec.sources.len() as u64).to_le_bytes());

    for (source, label) in &spec.sources {
        absorb_source(h, source);

        h.update(&(label.len() as u64).to_le_bytes());
        h.update(label);
    }

    match spec.selector {
        Some(sel) => {
            h.update(&[1]);
            h.update(&(sel as u64).to_le_bytes());
        }
        None => h.update(&[0]),
    }

    match spec.recv_selector {
        Some(sel) => {
            h.update(&[1]);
            h.update(&(sel as u64).to_le_bytes());
        }
        None => h.update(&[0]),
    }

    match spec.clock_waiver.as_deref() {
        Some(reason) => {
            h.update(&[1]);
            h.update(&(reason.len() as u64).to_le_bytes());
            h.update(reason.as_bytes());
        }
        None => h.update(&[0]),
    }
}

fn absorb_source(h: &mut DefaultHasher, source: &Source) {
    match source {
        Source::Column(idx) => {
            h.update(&[0]);
            h.update(&(*idx as u64).to_le_bytes());
        }
        Source::Columns(indices) => {
            h.update(&[1]);
            h.update(&(indices.len() as u64).to_le_bytes());

            for idx in indices {
                h.update(&(*idx as u64).to_le_bytes());
            }
        }
        Source::RowIndexLeBytes(n) => {
            h.update(&[2]);
            h.update(&(*n as u64).to_le_bytes());
        }
        Source::Const(val) => {
            h.update(&[3]);
            h.update(&val.to_le_bytes());
        }
        Source::RowIndexByte(n) => {
            h.update(&[4]);
            h.update(&(*n as u64).to_le_bytes());
        }
    }
}

fn absorb_expander(h: &mut DefaultHasher, exp: &hekate_program::expander::VirtualExpander) {
    let entries = exp.expansion_entries();
    h.update(&(entries.len() as u64).to_le_bytes());

    for entry in &entries {
        match *entry {
            ExpansionEntry::ExpandBits { count, storage } => {
                h.update(&[0]);
                h.update(&(count as u64).to_le_bytes());
                h.update(&[column_type_tag(storage)]);
            }
            ExpansionEntry::PassThrough { count, storage } => {
                h.update(&[1]);
                h.update(&(count as u64).to_le_bytes());
                h.update(&[column_type_tag(storage)]);
            }
            ExpansionEntry::ControlBits { count } => {
                h.update(&[2]);
                h.update(&(count as u64).to_le_bytes());
            }
            ExpansionEntry::ReusePassThrough {
                phy_col_start,
                count,
                storage,
            } => {
                h.update(&[3]);
                h.update(&(phy_col_start as u64).to_le_bytes());
                h.update(&(count as u64).to_le_bytes());
                h.update(&[column_type_tag(storage)]);
            }
        }
    }
}

fn column_type_tag(ct: ColumnType) -> u8 {
    match ct {
        ColumnType::Bit => 0,
        ColumnType::B8 => 1,
        ColumnType::B16 => 2,
        ColumnType::B32 => 3,
        ColumnType::B64 => 4,
        ColumnType::B128 => 5,
    }
}
