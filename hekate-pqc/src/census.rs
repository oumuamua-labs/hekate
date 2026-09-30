// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! The bus keys a pipeline's tables pin, read
//! from the fixed columns the verifier checks.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use hekate_math::Block128;
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::Col;
use hekate_program::permutation::Source;
use hekate_program::{Air, FixedShape};

use crate::sampler::{SIB_BUS_ID, SamplerChiplet};
use crate::wiring::{COEF_BUS_ID, LANE_BUS_ID, N, WORD_BUS_ID};

type F = Block128;

pub(crate) type Key = (String, u128, u128);

pub(crate) type Producers = [(String, Vec<(&'static str, u16)>)];

/// Emitters of every key the `coef`, `word`, `lane` and sib endpoints
/// of `defs` pin, by table and whether it produces the key, with the keys
/// `sampler`'s AIR forces. An unpinned key column or an idle producer fails.
pub(crate) fn census(
    defs: &[ChipletDef<F>],
    sampler: &SamplerChiplet<F>,
    producers: &Producers,
) -> Result<BTreeMap<Key, Vec<(String, bool)>>, String> {
    let (witness, forced) = sampler_keys(sampler);

    let mut keys = pinned_keys(defs, &witness)?;
    for key in forced {
        keys.entry(key)
            .or_default()
            .push("SamplerChiplet".to_string());
    }

    let made: BTreeSet<(&str, &str, u128)> = producers
        .iter()
        .flat_map(|(table, list)| {
            list.iter()
                .map(move |&(bus, label)| (table.as_str(), bus, u128::from(label)))
        })
        .collect();

    let emitted: BTreeSet<(&str, &str, u128)> = keys
        .iter()
        .flat_map(|((bus, label, _), emitters)| {
            emitters
                .iter()
                .map(move |table| (table.as_str(), bus.as_str(), *label))
        })
        .collect();

    if let Some((table, bus, label)) = made.difference(&emitted).next() {
        return Err(format!(
            "{table}: produces {bus} label {label} it never emits"
        ));
    }

    Ok(keys
        .iter()
        .map(|((bus, label, index), emitters)| {
            let tagged = emitters
                .iter()
                .map(|table| {
                    let produces = made.contains(&(table.as_str(), bus.as_str(), *label));

                    (table.clone(), produces)
                })
                .collect();

            ((bus.clone(), *label, *index), tagged)
        })
        .collect())
}

/// Keys that break the one-producer-one-consumer rule: two emitters
/// per `coef`, `word` and `lane` key, a producer among them, one per
/// program-order sib access. Label 0 is the JUNK slot and cancels in pairs.
pub(crate) fn unbalanced(keys: &BTreeMap<Key, Vec<(String, bool)>>) -> Vec<&Key> {
    keys.iter()
        .filter(|((bus, label, _), emitters)| {
            let (expected, directed) = match bus.as_str() {
                SIB_BUS_ID => (1, false),
                _ => (2, true),
            };

            let produced = !directed || emitters.iter().any(|&(_, produces)| produces);

            *label != 0 && (emitters.len() != expected || !produced)
        })
        .map(|(key, _)| key)
        .collect()
}

fn pinned_keys(
    defs: &[ChipletDef<F>],
    witness: &[(&str, usize)],
) -> Result<BTreeMap<Key, Vec<String>>, String> {
    let mut keys: BTreeMap<Key, Vec<String>> = BTreeMap::new();
    for def in defs {
        let name = def.name();

        let pins: BTreeMap<usize, &FixedShape<F>> =
            def.pins().iter().map(|p| (p.col_idx, &p.shape)).collect();

        for (bus, spec) in &def.permutation_checks {
            if ![COEF_BUS_ID, WORD_BUS_ID, LANE_BUS_ID, SIB_BUS_ID].contains(&bus.as_str()) {
                continue;
            }

            let column = |slot: usize| match spec.sources.get(slot) {
                Some((Source::Column(col), _)) => Some(*col),
                _ => None,
            };

            let (label, index) = (column(0), column(1));

            let pinned = |col: Option<usize>| col.and_then(|c| pins.get(&c).copied());

            let (Some(label), Some(index)) = (pinned(label), pinned(index)) else {
                let listed = [label, index].into_iter().all(|col| {
                    col.is_some_and(|c| {
                        pins.contains_key(&c) || witness.contains(&(name.as_str(), c))
                    })
                });

                match listed {
                    true => continue,
                    false => {
                        return Err(format!("{name}: {bus} key has an unpinned column"));
                    }
                }
            };

            let selector = spec
                .selector
                .and_then(|col| pins.get(&col).copied())
                .ok_or_else(|| format!("{name}: {bus} selector is not pinned"))?;

            for row in 0..extent(selector)? {
                if value(selector, row) == 0 {
                    continue;
                }

                keys.entry((bus.clone(), value(label, row), value(index, row)))
                    .or_default()
                    .push(name.clone());
            }
        }
    }

    Ok(keys)
}

/// The Sampler's witness-keyed columns, and the keys its
/// AIR forces them to carry: positions 0..N of each rejection
/// polynomial and N + i, i ≥ 256 − τ, of each challenge link.
fn sampler_keys(sampler: &SamplerChiplet<F>) -> (Vec<(&'static str, usize)>, Vec<Key>) {
    let ly = sampler.layout();
    let def = sampler.def().unwrap();

    let pins: BTreeMap<usize, &FixedShape<F>> =
        def.pins().iter().map(|p| (p.col_idx, &p.shape)).collect();

    let labels = |work: Col| -> Vec<u128> {
        let (Some(work), Some(poly)) = (pins.get(&work.index()), pins.get(&ly.poly.index())) else {
            return Vec::new();
        };

        let mut labels: Vec<u128> = (0..extent(work).unwrap())
            .filter(|&row| value(work, row) != 0)
            .map(|row| value(poly, row))
            .collect();

        labels.sort_unstable();
        labels.dedup();

        labels
    };

    let (mut witness, mut keys) = (Vec::new(), Vec::new());

    let coef = |label: u128, pos: usize| (COEF_BUS_ID.to_string(), label, pos as u128);

    if let Some(rc) = &ly.rej_cols {
        witness.extend(
            rc.key_poly
                .iter()
                .chain(rc.key_pos.iter())
                .map(|col| ("SamplerChiplet", col.index())),
        );

        for label in labels(rc.work) {
            keys.extend((0..N).map(|pos| coef(label, pos)));
        }
    }

    if let (Some(bb), Some(bc)) = (&ly.ball, &ly.ball_cols) {
        witness.push(("SamplerChiplet", bc.key_poly.index()));
        witness.push(("SamplerChiplet", bc.key_pos.index()));
        witness.push(("SamplerChiplet", bc.sort.time.index()));

        let first = bb.shape.first_cell();

        for label in labels(bc.work) {
            keys.extend((first..N).map(|i| coef(label, N + i)));
        }
    }

    (witness, keys)
}

fn extent(shape: &FixedShape<F>) -> Result<usize, String> {
    match shape {
        FixedShape::Segments(segments) => Ok(segments
            .iter()
            .map(|s| s.origin + s.stride * s.count)
            .max()
            .unwrap_or(0)),
        FixedShape::Sparse(entries) => {
            Ok(entries.iter().map(|&(row, _)| row + 1).max().unwrap_or(0))
        }
        FixedShape::Cadence {
            stride,
            count,
            origin,
            ..
        } => Ok(origin + stride * count),
        other => Err(format!("selector has an unbounded shape: {other:?}")),
    }
}

fn value(shape: &FixedShape<F>, row: usize) -> u128 {
    shape.value_at_row(row, 0).to_tower().0
}
