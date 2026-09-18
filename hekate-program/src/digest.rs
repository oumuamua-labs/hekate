// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use crate::chiplet::ChipletDef;
use crate::constraint::{BoundaryTarget, ConstraintAst, ConstraintExpr, ExprId};
use crate::expander::ExpansionEntry;
use crate::permutation::{BusKind, PermutationCheckSpec, Source};
use crate::{Air, FixedColumn, FixedShape, InlineKernelHint, Program};
use alloc::string::String;
use alloc::vec::Vec;
use hekate_core::errors;
use hekate_core::trace::ColumnType;
use hekate_crypto::{DefaultHasher, Hasher};
use hekate_math::TowerField;

/// Chunk size only:
/// the byte stream, hence the digest, is independent of it.
const FLUSH_BYTES: usize = 64 * 1024;

struct Absorb {
    hasher: DefaultHasher,
    buf: Vec<u8>,
}

impl Absorb {
    fn new() -> Self {
        Self {
            hasher: DefaultHasher::new(),
            buf: Vec::with_capacity(FLUSH_BYTES + 64),
        }
    }

    fn update(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);

        if self.buf.len() >= FLUSH_BYTES {
            self.hasher.update(&self.buf);
            self.buf.clear();
        }
    }

    fn finalize(mut self) -> [u8; 32] {
        self.hasher.update(&self.buf);

        self.hasher.finalize()
    }
}

/// Deterministic 32-byte structural ID. Witness-independent,
/// num_rows-independent. Same program shape -> same ID.
pub fn program_id<F: TowerField, P: Program<F>>(program: &P) -> errors::Result<[u8; 32]> {
    Ok(program_id_of(
        program,
        &program.chiplet_defs()?,
        &program.inline_chiplets()?,
    ))
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

pub fn program_id_of<F: TowerField, P: Program<F>>(
    program: &P,
    chiplet_defs: &[ChipletDef<F>],
    inline_chiplets: &[ChipletDef<F>],
) -> [u8; 32] {
    let mut h = Absorb::new();
    h.update(b"hekate-program-id-v2");

    let main_name = program.name();
    h.update(&(main_name.len() as u64).to_le_bytes());
    h.update(main_name.as_bytes());

    absorb_layout(&mut h, program.column_layout());
    absorb_ast::<F>(&mut h, &program.constraint_ast());
    absorb_boundaries(&mut h, &program.boundary_constraints());
    absorb_permutation_checks(&mut h, &program.permutation_checks());

    h.update(&(chiplet_defs.len() as u64).to_le_bytes());

    for cd in chiplet_defs {
        absorb_chiplet_def::<F>(&mut h, cd);
    }

    if let Some(exp) = program.virtual_expander() {
        h.update(&[1]);

        absorb_expander(&mut h, exp);
    } else {
        h.update(&[0]);
    }

    absorb_fixed_columns(&mut h, &program.fixed_columns());

    h.update(&(program.num_columns() as u64).to_le_bytes());
    h.update(&(program.num_public_inputs() as u64).to_le_bytes());

    absorb_inline_chiplets::<F>(&mut h, inline_chiplets, &program.inline_chiplet_kernels());

    h.finalize()
}

fn absorb_chiplet_def<F: TowerField>(h: &mut Absorb, cd: &ChipletDef<F>) {
    let name = Air::<F>::name(cd);

    h.update(&(name.len() as u64).to_le_bytes());
    h.update(name.as_bytes());

    absorb_layout(h, cd.column_layout());

    let cd_ast = cd.constraint_ast();
    absorb_ast::<F>(h, &cd_ast);

    absorb_boundaries(h, &cd.boundary_constraints());
    absorb_permutation_checks(h, &cd.permutation_checks());
    absorb_fixed_columns(h, &Air::<F>::fixed_columns(cd));

    if let Some(exp) = cd.virtual_expander() {
        h.update(&[1]);

        absorb_expander(h, exp);
    } else {
        h.update(&[0]);
    }
}

fn absorb_inline_chiplets<F: TowerField>(
    h: &mut Absorb,
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

fn absorb_layout(h: &mut Absorb, layout: &[ColumnType]) {
    h.update(&(layout.len() as u64).to_le_bytes());

    for ct in layout {
        h.update(&[column_type_tag(*ct)]);
    }
}

fn absorb_ast<F: TowerField>(h: &mut Absorb, ast: &ConstraintAst<F>) {
    h.update(&(ast.arena.len() as u64).to_le_bytes());

    for i in 0..ast.arena.len() {
        absorb_expr(h, ast.arena.get(ExprId(i as u32)));
    }

    h.update(&(ast.roots.len() as u64).to_le_bytes());

    for root in &ast.roots {
        h.update(&root.0.to_le_bytes());
    }
}

fn absorb_expr<F: TowerField>(h: &mut Absorb, expr: &ConstraintExpr<F>) {
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
    h: &mut Absorb,
    boundaries: &[crate::constraint::BoundaryConstraint<F>],
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

fn absorb_fixed_columns<F: TowerField>(h: &mut Absorb, fixed: &[FixedColumn<F>]) {
    h.update(&(fixed.len() as u64).to_le_bytes());

    for fc in fixed {
        h.update(&(fc.col_idx as u64).to_le_bytes());

        match &fc.shape {
            FixedShape::LastRow => h.update(&[0]),
            FixedShape::FirstRow => h.update(&[1]),
            FixedShape::Custom(bits) => {
                h.update(&[2]);
                h.update(&(bits.len() as u64).to_le_bytes());

                for &b in bits {
                    h.update(&[b as u8]);
                }
            }
            FixedShape::Periodic { period, values } => {
                h.update(&[3]);
                h.update(&(*period as u64).to_le_bytes());
                h.update(&(values.len() as u64).to_le_bytes());

                for v in values {
                    h.update(&v.to_bytes());
                }
            }
            FixedShape::Sparse(entries) => {
                h.update(&[4]);
                h.update(&(entries.len() as u64).to_le_bytes());

                for (row, v) in entries {
                    h.update(&(*row as u64).to_le_bytes());
                    h.update(&v.to_bytes());
                }
            }
            FixedShape::Dense(values) => {
                h.update(&[5]);
                h.update(&(values.len() as u64).to_le_bytes());

                for v in values {
                    h.update(&v.to_bytes());
                }
            }
            FixedShape::Cadence {
                stride,
                count,
                origin,
                values,
            } => {
                h.update(&[6]);
                h.update(&(*stride as u64).to_le_bytes());
                h.update(&(*count as u64).to_le_bytes());
                h.update(&(*origin as u64).to_le_bytes());
                h.update(&(values.len() as u64).to_le_bytes());

                for v in values {
                    h.update(&v.to_bytes());
                }
            }
            FixedShape::Segments(segments) => {
                h.update(&[7]);
                h.update(&(segments.len() as u64).to_le_bytes());

                for seg in segments {
                    h.update(&(seg.stride as u64).to_le_bytes());
                    h.update(&(seg.count as u64).to_le_bytes());
                    h.update(&(seg.origin as u64).to_le_bytes());
                    h.update(&(seg.values.len() as u64).to_le_bytes());

                    for v in &seg.values {
                        h.update(&v.to_bytes());
                    }
                }
            }
        }
    }
}

fn absorb_permutation_checks(h: &mut Absorb, checks: &[(String, PermutationCheckSpec)]) {
    h.update(&(checks.len() as u64).to_le_bytes());

    for (bus_id, spec) in checks {
        h.update(&(bus_id.len() as u64).to_le_bytes());
        h.update(bus_id.as_bytes());

        absorb_perm_spec(h, spec);
    }
}

fn absorb_perm_spec(h: &mut Absorb, spec: &PermutationCheckSpec) {
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

fn absorb_source(h: &mut Absorb, source: &Source) {
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
        Source::PhaseColumn(idx) => {
            h.update(&[5]);
            h.update(&(*idx as u64).to_le_bytes());
        }
    }
}

fn absorb_expander(h: &mut Absorb, exp: &crate::expander::VirtualExpander) {
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
            ExpansionEntry::ReuseExpandBits {
                phy_col_start,
                count,
                storage,
            } => {
                h.update(&[4]);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffering_does_not_move_digest() {
        let fields: Vec<Vec<u8>> = (0..FLUSH_BYTES / 4 + 17)
            .map(|i| (i as u32).to_le_bytes().to_vec())
            .collect();

        let mut buffered = Absorb::new();
        let mut direct = DefaultHasher::new();

        for field in &fields {
            buffered.update(field);
            direct.update(field);
        }

        assert_eq!(buffered.finalize(), direct.finalize());
    }
}
