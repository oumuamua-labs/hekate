// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec;
use alloc::vec::Vec;
use hekate_core::errors::{self, Error};
use hekate_core::trace::{ColumnTrace, TraceBuilder};
use hekate_math::{Block32, TowerField};
use hekate_program::Air;
use hekate_program::circuit::CircuitProgram;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};
use zeroize::Zeroizing;

use super::layout::KemSelectLayout;
use super::{KEY_WORDS, KemSelectStep};
use crate::utils::Writer;
use crate::wiring::{Stream, WordValues};

pub(super) fn generate<F: TowerField>(
    program: &CircuitProgram<F>,
    steps: &[KemSelectStep],
    ly: &KemSelectLayout,
    num_rows: usize,
    words: &mut WordValues,
) -> errors::Result<ColumnTrace> {
    let num_vars = num_rows.trailing_zeros() as usize;
    let mut tb = TraceBuilder::new_secret(program.column_layout(), num_vars)?;

    let mut row = 0;
    for s in steps {
        let k_prime = Zeroizing::new(read(words, s.k_prime, KEY_WORDS)?);
        let k_bar = Zeroizing::new(read(words, s.k_bar, KEY_WORDS)?);

        let mut flag = Choice::from(0);

        let origin = row;

        for part in &s.parts {
            let mut c = Zeroizing::new(read(words, part.c, part.words)?);
            let c_prime = Zeroizing::new(read(words, part.c_prime, part.words)?);

            for j in 0..part.words {
                let d = c[j] ^ c_prime[j];
                let nz = !d.ct_eq(&0);

                let mut w = Writer {
                    tb: &mut tb,
                    physical: |col| Ok(ly.physical(col)),
                    row,
                };

                w.word(ly.word_a, c[j])?;
                w.word(ly.word_b, c_prime[j])?;
                w.word(ly.word_c, c[j])?;
                w.word(ly.inv, Block32(d).invert().0)?;

                w.label(ly.stream_a, part.c.id())?;
                w.label(ly.stream_b, part.c_prime.id())?;
                w.label(ly.stream_c, part.relay.id())?;
                w.label(ly.idx, j as u16)?;

                w.flag(ly.sel_ab, true)?;
                w.flag(ly.sel_c, true)?;
                w.flag(ly.compare, true)?;
                w.flag(ly.first, row == origin)?;
                w.flag(ly.chain, true)?;
                w.flag(ly.nz, bool::from(nz))?;
                w.flag(ly.flag, bool::from(flag))?;

                flag |= nz;
                row += 1;
            }

            words.insert(part.relay, core::mem::take(&mut *c))?;
        }

        let mut key = Zeroizing::new(Vec::with_capacity(KEY_WORDS));

        for j in 0..KEY_WORDS {
            let k = u32::conditional_select(&k_prime[j], &k_bar[j], flag);

            let mut w = Writer {
                tb: &mut tb,
                physical: |col| Ok(ly.physical(col)),
                row,
            };

            w.word(ly.word_a, k_prime[j])?;
            w.word(ly.word_b, k_bar[j])?;
            w.word(ly.word_c, k)?;

            w.label(ly.stream_a, s.k_prime.id())?;
            w.label(ly.stream_b, s.k_bar.id())?;
            w.label(ly.stream_c, s.k.id())?;
            w.label(ly.idx, j as u16)?;

            w.flag(ly.sel_ab, true)?;
            w.flag(ly.sel_c, true)?;
            w.flag(ly.select, true)?;
            w.flag(ly.chain, true)?;
            w.flag(ly.flag, bool::from(flag))?;

            key.push(k);

            row += 1;
        }

        let valid = u32::from(bool::from(!flag));

        let mut w = Writer {
            tb: &mut tb,
            physical: |col| Ok(ly.physical(col)),
            row,
        };

        w.word(ly.word_c, valid)?;
        w.label(ly.stream_c, s.valid.id())?;
        w.flag(ly.sel_c, true)?;
        w.flag(ly.report, true)?;
        w.flag(ly.flag, bool::from(flag))?;

        row += 1;

        words.insert(s.k, core::mem::take(&mut *key))?;
        words.insert(s.valid, vec![valid])?;
    }

    Ok(tb.build())
}

fn read(words: &WordValues, stream: Stream, len: usize) -> errors::Result<Vec<u32>> {
    let list = words.get(stream)?;

    if list.len() != len {
        return Err(Error::Protocol {
            protocol: "kem_select_chiplet",
            message: "word stream length differs from its step",
        });
    }

    Ok(list.to_vec())
}
