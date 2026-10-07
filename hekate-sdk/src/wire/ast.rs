// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec::Vec;
use flatbuffers::{Allocator, FlatBufferBuilder};
use hekate_core::errors::{Error, Result};
use hekate_math::TowerField;
use hekate_program::ProgramCell;
use hekate_program::constraint::{ConstraintArena, ConstraintAst, ConstraintExpr, ExprId};

use super::Interner;
use crate::generated::program as fb;

pub fn serialize_ast<'a, F: TowerField, A: Allocator + 'a>(
    fbb: &mut FlatBufferBuilder<'a, A>,
    ast: &ConstraintAst<F>,
) -> flatbuffers::WIPOffset<fb::ConstraintAst<'a>> {
    let num_nodes = ast.arena.len();

    let mut node_offsets = Vec::with_capacity(num_nodes);
    for i in 0..num_nodes {
        let expr = ast.arena.get(ExprId(i as u32));
        let node = serialize_expr(fbb, expr);
        node_offsets.push(node);
    }

    let nodes_vec = fbb.create_vector(&node_offsets);

    let roots: Vec<u32> = ast.roots.iter().map(|id| id.0).collect();
    let roots_vec = fbb.create_vector(&roots);

    let label_offsets: Vec<_> = ast
        .labels
        .iter()
        .map(|opt| match opt {
            Some(s) => fbb.create_string(s),
            None => fbb.create_string(""),
        })
        .collect();
    let labels_vec = fbb.create_vector(&label_offsets);

    fb::ConstraintAst::create(
        fbb,
        &fb::ConstraintAstArgs {
            nodes: Some(nodes_vec),
            roots: Some(roots_vec),
            labels: Some(labels_vec),
        },
    )
}

pub fn deserialize_ast<F: TowerField>(
    fb_ast: fb::ConstraintAst<'_>,
    interner: &mut Interner,
) -> Result<ConstraintAst<F>> {
    let fb_nodes = fb_ast.nodes().ok_or(Error::Protocol {
        protocol: "wire",
        message: "missing AST nodes",
    })?;

    let mut arena = ConstraintArena::new();

    for i in 0..fb_nodes.len() {
        let node = fb_nodes.get(i);
        let expr = deserialize_expr::<F>(&node)?;
        let id = arena.alloc(expr);

        debug_assert_eq!(id.0, i as u32);
    }

    let fb_roots = fb_ast.roots().ok_or(Error::Protocol {
        protocol: "wire",
        message: "missing AST roots",
    })?;
    let roots: Vec<ExprId> = (0..fb_roots.len())
        .map(|i| ExprId(fb_roots.get(i)))
        .collect();

    let labels = match fb_ast.labels() {
        Some(fb_labels) => {
            let mut l = Vec::with_capacity(fb_labels.len());
            for i in 0..fb_labels.len() {
                let s = fb_labels.get(i);
                if s.is_empty() {
                    l.push(None);
                } else {
                    l.push(Some(interner.intern(s)?));
                }
            }

            l
        }
        None => Vec::new(),
    };

    let ast = ConstraintAst {
        arena,
        roots,
        labels,
    };

    ast.validate_order()?;

    Ok(ast)
}

fn serialize_expr<'a, F: TowerField, A: Allocator + 'a>(
    fbb: &mut FlatBufferBuilder<'a, A>,
    expr: &ConstraintExpr<F>,
) -> flatbuffers::WIPOffset<fb::AstNode<'a>> {
    match expr {
        ConstraintExpr::Cell(cell) => fb::AstNode::create(
            fbb,
            &fb::AstNodeArgs {
                tag: fb::NodeTag::Cell,
                col_index: cell.col_idx as u32,
                next_row: cell.next_row,
                ..Default::default()
            },
        ),
        ConstraintExpr::Const(val) => {
            let bytes = val.to_bytes();
            let block = block128_from_bytes(&bytes);

            fb::AstNode::create(
                fbb,
                &fb::AstNodeArgs {
                    tag: fb::NodeTag::Const,
                    const_value: Some(&block),
                    ..Default::default()
                },
            )
        }
        ConstraintExpr::Add(l, r) => fb::AstNode::create(
            fbb,
            &fb::AstNodeArgs {
                tag: fb::NodeTag::Add,
                left: l.0,
                right: r.0,
                ..Default::default()
            },
        ),
        ConstraintExpr::Mul(l, r) => fb::AstNode::create(
            fbb,
            &fb::AstNodeArgs {
                tag: fb::NodeTag::Mul,
                left: l.0,
                right: r.0,
                ..Default::default()
            },
        ),
        ConstraintExpr::Scale(scalar, child) => {
            let bytes = scalar.to_bytes();
            let block = block128_from_bytes(&bytes);

            fb::AstNode::create(
                fbb,
                &fb::AstNodeArgs {
                    tag: fb::NodeTag::Scale,
                    scalar: Some(&block),
                    left: child.0,
                    ..Default::default()
                },
            )
        }
        ConstraintExpr::Sum(children) => {
            let ids: Vec<u32> = children.iter().map(|id| id.0).collect();
            let vec = fbb.create_vector(&ids);

            fb::AstNode::create(
                fbb,
                &fb::AstNodeArgs {
                    tag: fb::NodeTag::Sum,
                    sum_children: Some(vec),
                    ..Default::default()
                },
            )
        }
    }
}

fn deserialize_expr<F: TowerField>(node: &fb::AstNode<'_>) -> Result<ConstraintExpr<F>> {
    match node.tag() {
        fb::NodeTag::Cell => Ok(ConstraintExpr::Cell(ProgramCell {
            col_idx: node.col_index() as usize,
            next_row: node.next_row(),
        })),
        fb::NodeTag::Const => {
            let block = node.const_value().ok_or(Error::Protocol {
                protocol: "wire",
                message: "Const node missing value",
            })?;

            Ok(ConstraintExpr::Const(field_from_block128::<F>(block)?))
        }
        fb::NodeTag::Add => Ok(ConstraintExpr::Add(
            ExprId(node.left()),
            ExprId(node.right()),
        )),
        fb::NodeTag::Mul => Ok(ConstraintExpr::Mul(
            ExprId(node.left()),
            ExprId(node.right()),
        )),
        fb::NodeTag::Scale => {
            let block = node.scalar().ok_or(Error::Protocol {
                protocol: "wire",
                message: "Scale node missing scalar",
            })?;

            Ok(ConstraintExpr::Scale(
                field_from_block128::<F>(block)?,
                ExprId(node.left()),
            ))
        }
        fb::NodeTag::Sum => {
            let children = node.sum_children().ok_or(Error::Protocol {
                protocol: "wire",
                message: "Sum node missing children",
            })?;

            let ids: Vec<ExprId> = (0..children.len())
                .map(|i| ExprId(children.get(i)))
                .collect();

            Ok(ConstraintExpr::Sum(ids))
        }
        _ => Err(Error::Protocol {
            protocol: "wire",
            message: "unknown AST node tag",
        }),
    }
}

fn block128_from_bytes(bytes: &[u8]) -> fb::Block128 {
    let (lo, hi) = super::field::bytes_to_lo_hi(bytes);
    fb::Block128::new(lo, hi)
}

fn field_from_block128<F: TowerField>(block: &fb::Block128) -> Result<F> {
    super::field::lo_hi_to_field(block.lo(), block.hi())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hekate_math::Block128;

    type F = Block128;

    #[test]
    fn decode_rejects_misordered_asts() {
        let cell = |col| ConstraintExpr::Cell(ProgramCell::current(col));

        let cases: [(&str, &str, [ConstraintExpr<F>; 3], u32); 3] = [
            (
                "forward child",
                "child does not precede its node",
                [
                    cell(0),
                    ConstraintExpr::Sum(vec![ExprId(0), ExprId(2)]),
                    cell(1),
                ],
                1,
            ),
            (
                "self reference",
                "child does not precede its node",
                [cell(0), ConstraintExpr::Mul(ExprId(0), ExprId(1)), cell(1)],
                1,
            ),
            (
                "root past the arena",
                "root outside the arena",
                [cell(0), cell(1), ConstraintExpr::Add(ExprId(0), ExprId(1))],
                3,
            ),
        ];

        for (case, expected, nodes, root) in cases {
            let mut arena = ConstraintArena::new();
            for node in nodes {
                arena.alloc(node);
            }

            let ast = ConstraintAst {
                arena,
                roots: vec![ExprId(root)],
                labels: vec![None],
            };

            let mut fbb = FlatBufferBuilder::new();
            let offset = serialize_ast(&mut fbb, &ast);

            fbb.finish(offset, None);

            let fb_ast =
                flatbuffers::root::<fb::ConstraintAst>(fbb.finished_data()).expect("flatbuffer");

            match deserialize_ast::<F>(fb_ast, &mut Interner::new()) {
                Err(Error::Protocol { message, .. }) => assert_eq!(message, expected, "{case}"),
                other => panic!("{case}: {:?}", other.map(|_| ())),
            }
        }
    }
}
