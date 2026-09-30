// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::trace::ColumnType;
use hekate_math::{HardwareField, TowerField};
use hekate_program::circuit::{Circuit, Col};

/// Column positions of the KemSelect table.
#[derive(Clone, Debug)]
pub struct KemSelectLayout {
    pub word_a: Col,
    pub word_b: Col,
    pub word_c: Col,
    pub inv: Col,

    pub stream_a: Col,
    pub stream_b: Col,
    pub stream_c: Col,
    pub idx: Col,

    pub sel_ab: Col,
    pub sel_c: Col,
    pub compare: Col,
    pub select: Col,
    pub report: Col,
    pub first: Col,
    pub chain: Col,
    pub nz: Col,
    pub flag: Col,
}

impl KemSelectLayout {
    pub(crate) fn declare<F: TowerField + HardwareField>(cx: &mut Circuit<F>) -> Self {
        let [word_a, word_b, word_c, inv] = [(); 4].map(|_| cx.column(ColumnType::B32));
        let [stream_a, stream_b, stream_c, idx] = [(); 4].map(|_| cx.column(ColumnType::B16));

        let [
            sel_ab,
            sel_c,
            compare,
            select,
            report,
            first,
            chain,
            nz,
            flag,
        ] = [(); 9].map(|_| cx.column(ColumnType::Bit));

        Self {
            word_a,
            word_b,
            word_c,
            inv,
            stream_a,
            stream_b,
            stream_c,
            idx,
            sel_ab,
            sel_c,
            compare,
            select,
            report,
            first,
            chain,
            nz,
            flag,
        }
    }

    pub fn physical(&self, col: Col) -> usize {
        col.index()
    }
}
