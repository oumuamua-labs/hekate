// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Eval-claim hiding on the verifier side:
//! the pad cursor that mirrors the prover's masking order,
//! and the zk-Ligero segment over the assembled statement.

use alloc::vec::Vec;
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::ligero::{
    Opening, ProductMask, RowEncoder, verify_interleaved, verify_linear, verify_opening,
    verify_quadratic, weights_at_columns,
};
use hekate_core::proofs::{InnerProof, OuterOpening};
use hekate_core::protocol;
use hekate_crypto::Hasher;
use hekate_crypto::transcript::Transcript;
use hekate_math::{BinaryFieldExtras, Flat, HardwareField, TowerField};
use hekate_program::outer::{
    OuterLayout, OuterStatement, TableRecord, assemble, linear_tensor_vars, linear_weights,
};
use tracing::{debug_span, instrument, warn};

/// Next unconsumed pad index;
/// advanced in the prover's masking order.
#[derive(Clone, Copy, Debug, Default)]
pub struct PadCursor(pub u32);

impl PadCursor {
    pub fn take(&mut self, count: usize) -> u32 {
        let first = self.0;
        self.0 += count as u32;

        first
    }
}

/// The zk-Ligero segment: geometry from the statement,
/// FS challenges after `aux_root`, both oracles opened
/// at the same columns, and the three Ligero tests
/// over the assembled rows.
#[instrument(skip_all, name = "verify_outer")]
#[allow(clippy::too_many_arguments)]
pub fn verify_outer<F, H>(
    proof: &InnerProof<F>,
    transcript: &mut Transcript<H>,
    config: &Config,
    statement: OuterStatement,
    records: Vec<TableRecord<F>>,
    cursor: PadCursor,
) -> errors::Result<bool>
where
    F: HardwareField + BinaryFieldExtras + TowerField,
    H: Hasher,
{
    if cursor.0 as usize != statement.masked_scalars {
        return Err(errors::Error::Protocol {
            protocol: "outer",
            message: "masked scalar count does not match the statement",
        });
    }

    let (pad_root, outer) = match (proof.pad_root.as_ref(), proof.outer.as_ref()) {
        (Some(root), Some(outer)) => (root, outer),
        _ => {
            return Err(errors::Error::Protocol {
                protocol: "outer",
                message: "eval-claim hiding needs both pad_root and the outer segment",
            });
        }
    };

    let field_bits = F::BITS;
    let geom = config.outer_geom(statement.masked_scalars, statement.mul_wires, field_bits)?;
    let layout = OuterLayout::new(&geom, statement.masked_scalars, statement.mul_wires)?;

    let encoder = RowEncoder::<F>::new(&geom)?;
    let vanisher = encoder.message_vanisher()?;

    transcript.append_message(b"aux_root", &outer.aux_root);

    let to_flat = |v: &[F]| -> Vec<Flat<F>> { v.iter().map(|x| x.to_hardware()).collect() };

    let r_int: Vec<F> = (0..layout.total_rows() - 1)
        .map(|_| transcript.challenge_field::<F>(b"outer_r"))
        .collect::<Result<_, _>>()?;

    transcript.append_field_list(b"outer_w", &outer.interleaved);

    let rows = assemble(records, &statement)?;
    let lin_vars = linear_tensor_vars(&rows);

    let r_lin: Vec<F> =
        debug_span!("r_lin", rows = rows.affine.len(), lin_vars).in_scope(|| {
            (0..lin_vars)
                .map(|_| transcript.challenge_field::<F>(b"outer_r_lin"))
                .collect::<Result<_, _>>()
        })?;

    transcript.append_field_list(b"outer_q", &outer.linear);

    let triples = layout.hadamard_triples();
    let r_quad: Vec<F> = (0..triples.len())
        .map(|_| transcript.challenge_field::<F>(b"outer_r_quad"))
        .collect::<Result<_, _>>()?;

    transcript.append_field_list(b"outer_p0", &outer.quadratic);

    let columns =
        protocol::challenge_outer_queries::<F, H>(transcript, geom.queries, geom.domain_len)?;

    let aux_rows = layout.total_rows() - layout.pad_rows;

    let opened = |opening: &OuterOpening<F>| -> bool {
        opening.columns.len() == columns.len()
            && opening
                .columns
                .iter()
                .zip(&columns)
                .all(|(&col, &want)| col as usize == want)
    };

    if !opened(&outer.pad_opening) || !opened(&outer.aux_opening) {
        warn!("outer openings do not cover the queried columns");
        return Ok(false);
    }

    let openings_hold = debug_span!("openings").in_scope(|| {
        verify_opening::<F, H>(
            pad_root,
            geom.domain_len,
            layout.pad_rows,
            &outer.pad_opening,
        ) && verify_opening::<F, H>(
            &outer.aux_root,
            geom.domain_len,
            aux_rows,
            &outer.aux_opening,
        )
    });

    if !openings_hold {
        warn!("outer opening rejected");
        return Ok(false);
    }

    let pad_opening = Opening::from_wire(&outer.pad_opening, layout.pad_rows)?;
    let aux_opening = Opening::from_wire(&outer.aux_opening, aux_rows)?;

    let stacked = Opening::stack(&[&pad_opening, &aux_opening])?;

    let interleaved_holds = debug_span!("interleaved").in_scope(|| {
        verify_interleaved(
            &encoder,
            &to_flat(&outer.interleaved),
            &to_flat(&r_int),
            layout.interleaved_mask(),
            &stacked,
        )
    });

    if !interleaved_holds {
        warn!("outer interleaved test failed");
        return Ok(false);
    }

    let batch =
        debug_span!("linear_weights").in_scope(|| linear_weights(&layout, &rows, &r_lin))?;
    let encoded = debug_span!("encode_weights", rows = batch.rows.len())
        .in_scope(|| weights_at_columns(&encoder, batch.weights, &columns))?;

    let linear_mask = ProductMask {
        low: layout.linear_mask(),
        high: layout.linear_mask_hi(),
        vanisher: &vanisher,
    };

    let linear_holds = debug_span!("linear").in_scope(|| {
        verify_linear(
            &encoder,
            &to_flat(&outer.linear),
            &encoded,
            &batch.rows,
            &linear_mask,
            batch.target,
            &stacked,
        )
    });

    if !linear_holds {
        warn!("outer linear test failed");
        return Ok(false);
    }

    let quadratic_mask = ProductMask {
        low: layout.quadratic_mask(),
        high: layout.quadratic_mask_hi(),
        vanisher: &vanisher,
    };

    let quadratic_holds = debug_span!("quadratic").in_scope(|| {
        verify_quadratic(
            &encoder,
            &to_flat(&outer.quadratic),
            &triples,
            &to_flat(&r_quad),
            &quadratic_mask,
            layout.message_len,
            &stacked,
        )
    });

    if !quadratic_holds {
        warn!("outer quadratic test failed");
        return Ok(false);
    }

    Ok(true)
}
