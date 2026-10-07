// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Each outer response is uniform on its constrained
//! set, hence independent of the committed rows.

use hekate_core::config::Config;
use hekate_core::ligero::{
    Opening, ProductMask, RowEncoder, column_leaf, verify_interleaved, verify_linear,
    verify_opening, verify_quadratic, weights_at_columns,
};
use hekate_core::outer::OuterGeometry;
use hekate_core::proofs::OuterOpening;
use hekate_crypto::DefaultHasher;
use hekate_crypto::merkle::MerkleTree;
use hekate_math::{Block128, Flat, HardwareField, TowerField};
use hekate_program::outer::{OuterLayout, aux_filler_len, build_aux_rows, build_pad_rows};

type K = Block128;
type H = DefaultHasher;

const FIELD_BITS: usize = 128;
const MASKED_SCALARS: usize = 16;
const WIRE_COUNTS: [usize; 2] = [6, 40];
const SEED: u128 = 0x0f1e_2d3c_4b5a_6978_8796_a5b4_c3d2_e1f0;

struct Outer {
    geom: OuterGeometry,
    layout: OuterLayout,
    encoder: RowEncoder<K>,
    vanisher: Vec<Flat<K>>,
}

/// Everything the verifier holds before it
/// reads the outer segment, coins included.
struct Coins {
    columns: Vec<usize>,
    interleaved: Vec<Flat<K>>,
    messages: Vec<Vec<Flat<K>>>,
    weights: Vec<Vec<Flat<K>>>,
    rows_used: Vec<usize>,
    target: Flat<K>,
    triples: Vec<[usize; 3]>,
    quadratic: Vec<Flat<K>>,
}

struct View {
    pad_root: [u8; 32],
    aux_root: [u8; 32],
    pad: OuterOpening<K>,
    aux: OuterOpening<K>,
    interleaved: Vec<Flat<K>>,
    linear: Vec<Flat<K>>,
    quadratic: Vec<Flat<K>>,
}

fn zero() -> Flat<K> {
    Flat::from_raw(K::ZERO)
}

fn rank(mut rows: Vec<Vec<Flat<K>>>) -> usize {
    let cols = rows.first().map_or(0, Vec::len);
    let mut rank = 0;

    for col in 0..cols {
        let Some(pivot) = (rank..rows.len()).find(|&r| rows[r][col] != zero()) else {
            continue;
        };

        rows.swap(rank, pivot);

        let inv = rows[rank][col].to_tower().invert().to_hardware();
        for v in rows[rank][col..].iter_mut() {
            *v *= inv;
        }

        let pivot_row = rows[rank].clone();
        for (r, row) in rows.iter_mut().enumerate() {
            if r == rank || row[col] == zero() {
                continue;
            }

            let factor = row[col];
            for (v, p) in row[col..].iter_mut().zip(&pivot_row[col..]) {
                *v += *p * factor;
            }
        }

        rank += 1;
    }

    rank
}

fn outer(mul_wires: usize) -> Outer {
    let geom = Config::dev()
        .outer_geom(MASKED_SCALARS, mul_wires, FIELD_BITS)
        .unwrap();
    let layout = OuterLayout::new(&geom, MASKED_SCALARS, mul_wires).unwrap();
    let encoder = RowEncoder::<K>::new(&geom).unwrap();
    let vanisher = encoder.message_vanisher().unwrap();

    Outer {
        geom,
        layout,
        encoder,
        vanisher,
    }
}

/// Filler entries that feed no mask row
/// contribute zero and leave the rank alone.
fn mask_images(
    outer: &Outer,
    mask: impl Fn(&[Vec<Flat<K>>], usize) -> Flat<K>,
) -> Vec<Vec<Flat<K>>> {
    let Outer {
        geom,
        layout,
        encoder,
        ..
    } = outer;

    let len = aux_filler_len(layout, geom.code_len);

    (0..len)
        .map(|j| {
            let mut filler = vec![zero(); len];
            filler[j] = Flat::from_raw(K::ONE);

            let mut rows = build_aux_rows(layout, &[], geom.code_len, &filler).unwrap();
            for row in rows.iter_mut() {
                encoder.encode(row).unwrap();
            }

            (0..geom.domain_len).map(|c| mask(&rows, c)).collect()
        })
        .collect()
}

/// The space every response lives in.
fn product_code_basis(outer: &Outer) -> Vec<Vec<Flat<K>>> {
    let Outer {
        geom,
        encoder,
        vanisher,
        ..
    } = outer;

    let unit = |i: usize| {
        let mut row = vec![zero(); geom.domain_len];
        row[i] = Flat::from_raw(K::ONE);
        encoder.encode(&mut row).unwrap();

        row
    };

    let low = (0..geom.code_len).map(unit);
    let high = (0..geom.code_len).map(|i| {
        let row = unit(i);

        (0..geom.domain_len).map(|c| vanisher[c] * row[c]).collect()
    });

    low.chain(high).collect()
}

fn rnd(state: &mut u128) -> Flat<K> {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;

    K::from(*state).to_hardware()
}

fn inv(v: Flat<K>) -> Flat<K> {
    v.to_tower().invert().to_hardware()
}

fn wire_form(outer: &Outer, response: &[Flat<K>], len: usize) -> Vec<Flat<K>> {
    outer.encoder.coefficients(response).unwrap()[..len].to_vec()
}

fn is_codeword(outer: &Outer, values: &[Flat<K>]) -> bool {
    outer.encoder.coefficients(values).unwrap()[outer.geom.code_len..]
        .iter()
        .all(|c| *c == zero())
}

fn sum_on_message(outer: &Outer, values: &[Flat<K>]) -> Flat<K> {
    outer.encoder.message_evaluations(values).unwrap().unwrap()[..outer.geom.code_len]
        .iter()
        .fold(zero(), |a, b| a + *b)
}

fn vanishes_on_message(outer: &Outer, values: &[Flat<K>]) -> bool {
    outer.encoder.message_evaluations(values).unwrap().unwrap()[..outer.geom.message_len]
        .iter()
        .all(|v| *v == zero())
}

fn coins(outer: &Outer, state: &mut u128) -> Coins {
    let Outer { geom, layout, .. } = outer;

    let mut columns = Vec::with_capacity(geom.queries);
    let mut c = 0usize;

    // Hull-Dobell: `37x + 11` has full
    // period on a power-of-two modulus.
    while columns.len() < geom.queries {
        c = (c * 37 + 11) % geom.domain_len;

        if !columns.contains(&c) {
            columns.push(c);
        }
    }

    columns.sort_unstable();

    let rows_used = vec![0, layout.pad_rows, layout.pad_rows + layout.aux_rows];
    let messages: Vec<Vec<Flat<K>>> = rows_used
        .iter()
        .map(|_| (0..geom.message_len).map(|_| rnd(state)).collect())
        .collect();

    let weights: Vec<Vec<Flat<K>>> = messages.iter().map(|m| codeword(outer, m)).collect();

    let triples = layout.hadamard_triples();
    let interleaved: Vec<Flat<K>> = (0..layout.total_rows() - 1).map(|_| rnd(state)).collect();
    let target = rnd(state);
    let quadratic: Vec<Flat<K>> = triples.iter().map(|_| rnd(state)).collect();

    assert!(interleaved.iter().chain(&quadratic).any(|v| *v != zero()));

    Coins {
        columns,
        interleaved,
        messages,
        weights,
        rows_used,
        target,
        triples,
        quadratic,
    }
}

fn codeword(outer: &Outer, message: &[Flat<K>]) -> Vec<Flat<K>> {
    let mut row = vec![zero(); outer.geom.domain_len];
    row[..message.len()].copy_from_slice(message);

    outer.encoder.encode(&mut row).unwrap();

    row
}

fn uniform_codeword(outer: &Outer, state: &mut u128) -> Vec<Flat<K>> {
    let message: Vec<Flat<K>> = (0..outer.geom.code_len).map(|_| rnd(state)).collect();

    codeword(outer, &message)
}

fn product(outer: &Outer, low: &[Flat<K>], high: &[Flat<K>]) -> Vec<Flat<K>> {
    (0..outer.geom.domain_len)
        .map(|c| low[c] + outer.vanisher[c] * high[c])
        .collect()
}

/// Uniform on `{deg < 2k, sum on message = target}`.
fn response_with_sum(outer: &Outer, target: Flat<K>, state: &mut u128) -> Vec<Flat<K>> {
    let mut message: Vec<Flat<K>> = (0..outer.geom.code_len).map(|_| rnd(state)).collect();
    let last = message[..outer.geom.code_len - 1]
        .iter()
        .fold(target, |acc, v| acc + *v);

    message[outer.geom.code_len - 1] = last;

    let low = codeword(outer, &message);
    let high = uniform_codeword(outer, state);

    product(outer, &low, &high)
}

/// Uniform on `{deg < 2k, zero on the message}`.
fn response_vanishing(outer: &Outer, state: &mut u128) -> Vec<Flat<K>> {
    let mut message = vec![zero(); outer.geom.code_len];
    for v in message[outer.geom.message_len..].iter_mut() {
        *v = rnd(state);
    }

    let low = codeword(outer, &message);
    let high = uniform_codeword(outer, state);

    product(outer, &low, &high)
}

fn commit(
    outer: &Outer,
    coins: &Coins,
    opened: &[Vec<Flat<K>>],
    rows: core::ops::Range<usize>,
) -> ([u8; 32], OuterOpening<K>) {
    let mut leaves: Vec<[u8; 32]> = vec![[0u8; 32]; outer.geom.domain_len];
    for (i, &c) in coins.columns.iter().enumerate() {
        leaves[c] = column_leaf::<K, H>(opened[i][rows.clone()].iter().map(|v| v.to_tower()));
    }

    let tree = MerkleTree::<K, H>::new(&leaves);

    let opening = Opening {
        columns: coins
            .columns
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, opened[i][rows.clone()].to_vec()))
            .collect(),
        siblings: tree.prove_batch(&coins.columns).unwrap(),
    };

    (tree.root(), opening.to_wire())
}

fn simulate(outer: &Outer, coins: &Coins, state: &mut u128) -> View {
    let Outer { layout, .. } = outer;

    let total = layout.total_rows();
    let interleaved = uniform_codeword(outer, state);
    let linear = response_with_sum(outer, coins.target, state);
    let quadratic = response_vanishing(outer, state);

    let mut opened: Vec<Vec<Flat<K>>> = coins
        .columns
        .iter()
        .map(|_| (0..total).map(|_| rnd(state)).collect())
        .collect();

    for (i, &c) in coins.columns.iter().enumerate() {
        let values = &mut opened[i];
        let mut acc = quadratic[c];

        for (t, [x, y, z]) in coins.triples.iter().enumerate() {
            acc += coins.quadratic[t] * (values[*x] * values[*y] + values[*z]);
        }

        values[layout.quadratic_mask_hi()] =
            (acc + values[layout.quadratic_mask()]) * inv(outer.vanisher[c]);

        let mut acc = linear[c];
        for (w, &r) in coins.weights.iter().zip(&coins.rows_used) {
            acc += w[c] * values[r];
        }

        values[layout.linear_mask_hi()] =
            (acc + values[layout.linear_mask()]) * inv(outer.vanisher[c]);

        let mut acc = interleaved[c];
        let mut next = 0;

        for (r, v) in values.iter().enumerate() {
            if r == layout.interleaved_mask() {
                continue;
            }

            acc += coins.interleaved[next] * *v;
            next += 1;
        }

        values[layout.interleaved_mask()] = acc;
    }

    let (pad_root, pad) = commit(outer, coins, &opened, 0..layout.pad_rows);
    let (aux_root, aux) = commit(outer, coins, &opened, layout.pad_rows..total);

    View {
        pad_root,
        aux_root,
        pad,
        aux,
        interleaved: wire_form(outer, &interleaved, outer.geom.code_len),
        linear: wire_form(outer, &linear, 2 * outer.geom.code_len),
        quadratic: wire_form(outer, &quadratic, 2 * outer.geom.code_len),
    }
}

/// The check block of `verify_outer` at fixed coins:
/// simulating Fiat-Shamir needs a programmable oracle.
fn accepts(outer: &Outer, coins: &Coins, view: &View) -> bool {
    let Outer {
        geom,
        layout,
        encoder,
        ..
    } = outer;

    let aux_rows = layout.total_rows() - layout.pad_rows;

    if !verify_opening::<K, H>(&view.pad_root, geom.domain_len, layout.pad_rows, &view.pad)
        || !verify_opening::<K, H>(&view.aux_root, geom.domain_len, aux_rows, &view.aux)
    {
        return false;
    }

    let pad = Opening::from_wire(&view.pad, layout.pad_rows).unwrap();
    let aux = Opening::from_wire(&view.aux, aux_rows).unwrap();
    let stacked = Opening::stack(&[&pad, &aux]).unwrap();

    let opened_weights = weights_at_columns(encoder, &coins.messages, &coins.columns).unwrap();
    let opened_vanisher = encoder.vanisher_at(&coins.columns).unwrap();

    let linear_mask = ProductMask {
        low: layout.linear_mask(),
        high: layout.linear_mask_hi(),
        vanisher: &opened_vanisher,
    };

    let quadratic_mask = ProductMask {
        low: layout.quadratic_mask(),
        high: layout.quadratic_mask_hi(),
        vanisher: &opened_vanisher,
    };

    verify_interleaved(
        encoder,
        &view.interleaved,
        &coins.interleaved,
        layout.interleaved_mask(),
        &stacked,
    ) && verify_linear(
        encoder,
        &view.linear,
        &opened_weights,
        &coins.rows_used,
        &linear_mask,
        coins.target,
        &stacked,
    ) && verify_quadratic(
        encoder,
        &view.quadratic,
        &coins.triples,
        &coins.quadratic,
        &quadratic_mask,
        geom.message_len,
        &stacked,
    )
}

fn honest_rows(outer: &Outer, coins: &Coins, state: &mut u128) -> Vec<Vec<Flat<K>>> {
    let Outer { geom, layout, .. } = outer;

    let pad: Vec<Flat<K>> = (0..MASKED_SCALARS).map(|_| rnd(state)).collect();
    let pad_filler: Vec<Flat<K>> = (0..layout.pad_rows * (geom.code_len - geom.message_len))
        .map(|_| rnd(state))
        .collect();

    let wires: Vec<[Flat<K>; 3]> = coins
        .triples
        .iter()
        .map(|_| {
            let lhs = rnd(state);
            let rhs = rnd(state);

            [lhs, rhs, lhs * rhs]
        })
        .collect();

    let aux_filler: Vec<Flat<K>> = (0..aux_filler_len(layout, geom.code_len))
        .map(|_| rnd(state))
        .collect();

    let mut rows = build_pad_rows(layout, &pad, geom.code_len, &pad_filler).unwrap();
    rows.extend(build_aux_rows(layout, &wires, geom.code_len, &aux_filler).unwrap());

    rows
}

/// The linear form the rows carry, before encoding.
fn message_sum(coins: &Coins, rows: &[Vec<Flat<K>>]) -> Flat<K> {
    let mut target = zero();
    for (message, &r) in coins.messages.iter().zip(&coins.rows_used) {
        for (w, v) in message.iter().zip(&rows[r]) {
            target += *w * *v;
        }
    }

    target
}

fn honest_view(outer: &Outer, coins: &Coins, rows: &[Vec<Flat<K>>]) -> View {
    let Outer {
        geom,
        layout,
        vanisher,
        ..
    } = outer;

    let interleaved: Vec<Flat<K>> = (0..geom.domain_len)
        .map(|c| {
            let mut acc = rows[layout.interleaved_mask()][c];
            let mut next = 0;

            for (r, row) in rows.iter().enumerate() {
                if r == layout.interleaved_mask() {
                    continue;
                }

                acc += coins.interleaved[next] * row[c];
                next += 1;
            }

            acc
        })
        .collect();

    let linear: Vec<Flat<K>> = (0..geom.domain_len)
        .map(|c| {
            let mut acc =
                rows[layout.linear_mask()][c] + vanisher[c] * rows[layout.linear_mask_hi()][c];
            for (w, &r) in coins.weights.iter().zip(&coins.rows_used) {
                acc += w[c] * rows[r][c];
            }

            acc
        })
        .collect();

    let quadratic: Vec<Flat<K>> = (0..geom.domain_len)
        .map(|c| {
            let mut acc = rows[layout.quadratic_mask()][c]
                + vanisher[c] * rows[layout.quadratic_mask_hi()][c];

            for (t, [x, y, z]) in coins.triples.iter().enumerate() {
                acc += coins.quadratic[t] * (rows[*x][c] * rows[*y][c] + rows[*z][c]);
            }

            acc
        })
        .collect();

    let opened: Vec<Vec<Flat<K>>> = coins
        .columns
        .iter()
        .map(|&c| rows.iter().map(|row| row[c]).collect())
        .collect();

    let (pad_root, pad) = commit(outer, coins, &opened, 0..layout.pad_rows);
    let (aux_root, aux) = commit(outer, coins, &opened, layout.pad_rows..layout.total_rows());

    View {
        pad_root,
        aux_root,
        pad,
        aux,
        interleaved: wire_form(outer, &interleaved, geom.code_len),
        linear: wire_form(outer, &linear, 2 * geom.code_len),
        quadratic: wire_form(outer, &quadratic, 2 * geom.code_len),
    }
}

#[test]
fn product_code_basis_is_independent() {
    for wires in WIRE_COUNTS {
        let outer = outer(wires);

        assert_eq!(
            rank(product_code_basis(&outer)),
            2 * outer.geom.code_len,
            "{wires} wires"
        );
    }
}

#[test]
fn interleaved_mask_spans_row_code() {
    for wires in WIRE_COUNTS {
        let outer = outer(wires);
        let row = outer.layout.interleaved_mask() - outer.layout.pad_rows;

        let images = mask_images(&outer, |rows, c| rows[row][c]);

        for image in &images {
            assert!(is_codeword(&outer, image), "{wires} wires");
        }

        assert_eq!(rank(images), outer.geom.code_len, "{wires} wires");
    }
}

/// The target fixes the coset,
/// the mask covers the rest.
#[test]
fn linear_mask_spans_zero_sum_product_code() {
    for wires in WIRE_COUNTS {
        let outer = outer(wires);
        let low = outer.layout.linear_mask() - outer.layout.pad_rows;
        let high = outer.layout.linear_mask_hi() - outer.layout.pad_rows;

        let images = mask_images(&outer, |rows, c| {
            rows[low][c] + outer.vanisher[c] * rows[high][c]
        });

        for image in &images {
            assert_eq!(sum_on_message(&outer, image), zero(), "{wires} wires");
        }

        let sums: Vec<Vec<Flat<K>>> = product_code_basis(&outer)
            .iter()
            .map(|v| vec![sum_on_message(&outer, v)])
            .collect();

        let constrained = 2 * outer.geom.code_len - rank(sums);

        assert_eq!(constrained, 2 * outer.geom.code_len - 1, "{wires} wires");
        assert_eq!(rank(images), constrained, "{wires} wires");

        let alone = mask_images(&outer, |rows, c| rows[low][c]);

        assert_eq!(rank(alone), outer.geom.code_len - 1, "{wires} wires");
    }
}

#[test]
fn quadratic_mask_spans_vanishing_product_code() {
    for wires in WIRE_COUNTS {
        let outer = outer(wires);
        let low = outer.layout.quadratic_mask() - outer.layout.pad_rows;
        let high = outer.layout.quadratic_mask_hi() - outer.layout.pad_rows;

        let images = mask_images(&outer, |rows, c| {
            rows[low][c] + outer.vanisher[c] * rows[high][c]
        });

        for image in &images {
            assert!(vanishes_on_message(&outer, image), "{wires} wires");
        }

        let evals: Vec<Vec<Flat<K>>> = product_code_basis(&outer)
            .iter()
            .map(|v| {
                outer.encoder.message_evaluations(v).unwrap().unwrap()[..outer.geom.message_len]
                    .to_vec()
            })
            .collect();

        let constrained = 2 * outer.geom.code_len - rank(evals);

        assert_eq!(
            constrained,
            2 * outer.geom.code_len - outer.geom.message_len,
            "{wires} wires"
        );
        assert_eq!(rank(images), constrained, "{wires} wires");

        let alone = mask_images(&outer, |rows, c| rows[low][c]);

        assert_eq!(
            rank(alone),
            outer.geom.code_len - outer.geom.message_len,
            "{wires} wires"
        );
    }
}

#[test]
fn honest_view_verifies() {
    for wires in WIRE_COUNTS {
        let outer = outer(wires);
        let mut state = SEED;

        let mut coins = coins(&outer, &mut state);
        let mut rows = honest_rows(&outer, &coins, &mut state);

        coins.target = message_sum(&coins, &rows);

        for row in rows.iter_mut() {
            outer.encoder.encode(row).unwrap();
        }

        let view = honest_view(&outer, &coins, &rows);

        assert!(accepts(&outer, &coins, &view), "{wires} wires");
    }
}

#[test]
fn simulated_view_verifies() {
    for wires in WIRE_COUNTS {
        let outer = outer(wires);
        let mut state = SEED;

        let coins = coins(&outer, &mut state);
        let view = simulate(&outer, &coins, &mut state);

        assert!(accepts(&outer, &coins, &view), "{wires} wires");
    }
}
