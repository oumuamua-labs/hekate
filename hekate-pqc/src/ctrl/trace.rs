// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use hekate_core::errors::{self, Error};
use hekate_core::trace::TraceBuilder;
use hekate_core::trace::TraceCompatibleField;
use hekate_keccak::{KeccakChiplet, KeccakWitness};
use hekate_math::{Flat, HardwareField, PackableField, TowerField};
use hekate_program::Air;
use zeroize::{Zeroize, Zeroizing};

use super::{CtrlChiplet, CtrlTrace, Effect, Io, LANES, PREFIX, RATE, Token, pinned};
use crate::utils::{Writer, write_column};
use crate::wiring::{LaneValues, WordValues};

#[derive(Zeroize)]
struct Produced {
    #[zeroize(skip)]
    token: Token,
    word: u32,
}

pub(super) fn generate<F>(
    chiplet: &CtrlChiplet<F>,
    inputs: &[u32],
    lanes: &LaneValues,
    words: &mut WordValues,
) -> errors::Result<CtrlTrace>
where
    F: TowerField + TraceCompatibleField + PackableField + HardwareField + Send + 'static,
    <F as PackableField>::Packed: Copy + Send + Sync,
    Flat<F>: Send + Sync,
{
    let (ly, num_rows) = (&chiplet.layout, chiplet.num_rows);
    let types = chiplet.program.column_layout();

    let mut tb = TraceBuilder::new_secret(types, num_rows.trailing_zeros() as usize)?;

    pinned(&chiplet.rows, ly, num_rows, |col, value| {
        let at = ly.physical(col)?;

        write_column(&mut tb, types[at], at, (0..num_rows).map(value))
    })?;

    let rows = &chiplet.rows;

    let calls = rows
        .iter()
        .filter(|row| row.kec && matches!(row.effect, Effect::Copy))
        .count();

    let tokens = rows
        .iter()
        .filter(|row| (row.split || row.high) && row.word.is_some())
        .count();

    let mut state = Zeroizing::new([0u64; LANES]);
    let mut keccak = Zeroizing::new(Vec::with_capacity(calls));
    let mut produced: Zeroizing<Vec<Produced>> = Zeroizing::new(Vec::with_capacity(tokens));
    let mut position: BTreeMap<Token, usize> = BTreeMap::new();
    let mut hosted = inputs.iter();
    let mut high = Zeroizing::new(0u32);

    for r in 0..num_rows {
        let row = rows.get(r).copied().unwrap_or_default();

        let word = match (row.io, row.word) {
            (Io::Input, _) => *hosted
                .next()
                .ok_or(missing("fewer host words than input rows"))?,
            (_, Some(_)) if row.split => state[0] as u32,
            (_, Some(_)) if row.high => *high,
            (_, Some((stream, idx))) => match position.get(&(stream, idx)) {
                Some(&at) => produced[at].word,
                None => *words
                    .get(stream)?
                    .get(idx as usize)
                    .ok_or(missing("consumed word no row or table produced"))?,
            },
            _ => 0,
        };

        if row.split {
            *high = (state[0] >> 32) as u32;
        }

        if let (true, Some(token)) = (row.split || row.high, row.word) {
            position.entry(token).or_insert(produced.len());
            produced.push(Produced { token, word });
        }

        let absorbed = match (row.effect, row.lane) {
            (Effect::Lane, Some((stream, idx))) => *lanes
                .get(stream)?
                .get(idx as usize)
                .ok_or(missing("a lane token past the end of its stream"))?,
            _ => 0,
        };

        match row.effect {
            Effect::Copy if row.kec => keccak.push(*state),
            Effect::Copy => {}
            Effect::Reset => *state = [0; LANES],
            Effect::Prefix => {
                for i in (0..=PREFIX).chain(RATE..LANES) {
                    state[i] = 0;
                }
            }
            Effect::Lo => state[0] ^= word as u64,
            Effect::Hi => rotate(&mut state, ((word as u64) << 32) ^ row.pad),
            Effect::Lane => rotate(&mut state, absorbed ^ row.pad),
            Effect::Rotate => rotate(&mut state, row.pad),
            Effect::Out => *state = keccak_f(*state),
        }

        let lane = match row.emit {
            true => state[RATE - 1],
            false => absorbed,
        };

        let mut w = Writer {
            tb: &mut tb,
            physical: |col| ly.physical(col),
            row: r,
        };

        for (i, &value) in state.iter().enumerate() {
            w.lane(ly.state.at(i), value)?;
        }

        w.word(ly.word, word)?;
        w.lane(ly.lane, lane)?;
    }

    if hosted.next().is_some() {
        return Err(missing("more host words than input rows"));
    }

    produced.sort_unstable_by_key(|p| p.token);

    for run in produced.chunk_by(|a, b| a.token.0 == b.token.0) {
        if let Some(first) = run.first() {
            let mut list = Vec::with_capacity(run.len());

            list.extend(run.iter().map(|p| p.word));
            words.insert(first.token.0, list)?;
        }
    }

    Ok(CtrlTrace {
        trace: tb.build(),
        keccak,
    })
}

fn rotate(state: &mut [u64; LANES], add: u64) {
    let front = state[0];

    state.copy_within(1..RATE, 0);
    state[RATE - 1] = front ^ add;
}

fn keccak_f(mut state: [u64; LANES]) -> [u64; LANES] {
    for &rc in KeccakChiplet::ROUND_CONSTANTS.iter() {
        state = KeccakWitness::keccak_f_round(state, rc);
    }

    state
}

fn missing(message: &'static str) -> Error {
    Error::Protocol {
        protocol: "ctrl_chiplet",
        message,
    }
}
