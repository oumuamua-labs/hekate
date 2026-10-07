// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Eval-claim hiding on the verifier side:
//! the pad cursor that mirrors the prover's masking order,
//! and the zk-Ligero segment over the assembled statement.

use crate::prepared::VerifierScratch;
use alloc::vec::Vec;
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::ligero::{
    Opening, ProductMask, verify_interleaved, verify_linear, verify_opening, verify_quadratic,
    weights_at_columns,
};
use hekate_core::proofs::{InnerProof, OuterOpening};
use hekate_core::protocol;
use hekate_crypto::Hasher;
use hekate_crypto::transcript::Transcript;
use hekate_math::{BinaryFieldExtras, Block128, Flat, HardwareField, TowerField};
use hekate_program::outer::{
    OuterLayout, OuterStatement, TableRecord, linear_tensor_vars, linear_weights, statement_rows,
};
use tracing::{instrument, trace_span, warn};

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
#[instrument(skip_all, level = "trace", name = "verify_outer")]
#[allow(clippy::too_many_arguments)]
pub fn verify_outer<F, H>(
    proof: &InnerProof<F>,
    transcript: &mut Transcript<H>,
    config: &Config,
    statement: OuterStatement,
    records: &[TableRecord<'_, F>],
    cursor: PadCursor,
    scratch: &mut VerifierScratch<F>,
) -> errors::Result<bool>
where
    F: HardwareField + BinaryFieldExtras + TowerField + Into<Block128>,
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

    let encoder = scratch.encoder(&geom)?;

    transcript.append_message(b"aux_root", &outer.aux_root);

    let to_flat = |v: &[F]| -> Vec<Flat<F>> { v.iter().map(|x| x.to_hardware()).collect() };

    let r_int: Vec<F> = (0..layout.total_rows() - 1)
        .map(|_| transcript.challenge_field::<F>(b"outer_r"))
        .collect::<Result<_, _>>()?;

    transcript.append_field_list(b"outer_w", &outer.interleaved);

    let rows = statement_rows(records, &statement)?;
    let lin_vars = linear_tensor_vars(rows);

    let r_lin: Vec<F> = trace_span!("r_lin", rows, lin_vars).in_scope(|| {
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

    let openings = || {
        trace_span!("openings").in_scope(|| {
            let (pad, aux) = join(
                || {
                    verify_opening::<F, H>(
                        pad_root,
                        geom.domain_len,
                        layout.pad_rows,
                        &outer.pad_opening,
                    )
                },
                || {
                    verify_opening::<F, H>(
                        &outer.aux_root,
                        geom.domain_len,
                        aux_rows,
                        &outer.aux_opening,
                    )
                },
            );

            pad && aux
        })
    };

    let tests = || -> errors::Result<[(errors::Result<bool>, &'static str); 3]> {
        let pad_opening = Opening::from_wire(&outer.pad_opening, layout.pad_rows)?;
        let aux_opening = Opening::from_wire(&outer.aux_opening, aux_rows)?;

        let stacked = Opening::stack(&[&pad_opening, &aux_opening])?;
        let vanisher = encoder.vanisher_at(&columns)?;

        let interleaved = || {
            trace_span!("interleaved").in_scope(|| {
                verify_interleaved(
                    encoder,
                    &to_flat(&outer.interleaved),
                    &to_flat(&r_int),
                    layout.interleaved_mask(),
                    &stacked,
                )
            })
        };

        let linear = || -> errors::Result<bool> {
            let batch = trace_span!("linear_weights")
                .in_scope(|| linear_weights(&layout, &statement, records, &r_lin))?;
            let encoded = trace_span!("encode_weights", rows = batch.rows.len())
                .in_scope(|| weights_at_columns(encoder, &batch.weights, &columns))?;

            let mask = ProductMask {
                low: layout.linear_mask(),
                high: layout.linear_mask_hi(),
                vanisher: &vanisher,
            };

            Ok(trace_span!("linear").in_scope(|| {
                verify_linear(
                    encoder,
                    &to_flat(&outer.linear),
                    &encoded,
                    &batch.rows,
                    &mask,
                    batch.target,
                    &stacked,
                )
            }))
        };

        let quadratic = || {
            let mask = ProductMask {
                low: layout.quadratic_mask(),
                high: layout.quadratic_mask_hi(),
                vanisher: &vanisher,
            };

            trace_span!("quadratic").in_scope(|| {
                verify_quadratic(
                    encoder,
                    &to_flat(&outer.quadratic),
                    &triples,
                    &to_flat(&r_quad),
                    &mask,
                    layout.message_len,
                    &stacked,
                )
            })
        };

        let (interleaved, (linear, quadratic)) = join(interleaved, || join(linear, quadratic));

        Ok([
            (Ok(interleaved), "outer interleaved test failed"),
            (linear, "outer linear test failed"),
            (Ok(quadratic), "outer quadratic test failed"),
        ])
    };

    let (openings_hold, tests) = join(openings, tests);

    if !openings_hold {
        warn!("outer opening rejected");
        return Ok(false);
    }

    for (holds, failure) in tests? {
        if !holds? {
            warn!("{failure}");
            return Ok(false);
        }
    }

    Ok(true)
}

/// Both closures always run: on the pool with
/// `parallel`, one after the other without it.
fn join<A, B, RA, RB>(a: A, b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    #[cfg(feature = "parallel")]
    let pair = rayon::join(a, b);

    #[cfg(not(feature = "parallel"))]
    let pair = (a(), b());

    pair
}
