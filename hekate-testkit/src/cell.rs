// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use core::ops::Range;

use hekate_core::trace::{ColumnTrace, TraceColumn};
use hekate_math::{Bit, Block8, Block16, Block32, Block64, HardwareField, TowerField};

macro_rules! cell_access {
    ($variant:ident, $block:ident, $int:ty, $at:ident, $set:ident, $xor:ident, $swap:ident) => {
        pub fn $at(trace: &ColumnTrace, col: usize, row: usize) -> $int {
            match &trace.columns[col] {
                TraceColumn::$variant(cells) => cells[row].to_tower().0,
                _ => panic!(concat!("column {} is not ", stringify!($variant)), col),
            }
        }

        pub fn $set(trace: &mut ColumnTrace, col: usize, row: usize, value: $int) {
            match &mut trace.columns[col] {
                TraceColumn::$variant(cells) => cells[row] = $block(value).to_hardware(),
                _ => panic!(concat!("column {} is not ", stringify!($variant)), col),
            }
        }

        pub fn $xor(trace: &mut ColumnTrace, col: usize, row: usize, mask: $int) {
            let value = $at(trace, col, row);
            $set(trace, col, row, value ^ mask);
        }

        pub fn $swap(trace: &mut ColumnTrace, a: usize, b: usize, row: usize) {
            let (va, vb) = ($at(trace, a, row), $at(trace, b, row));

            $set(trace, a, row, vb);
            $set(trace, b, row, va);
        }
    };
}

macro_rules! each_column {
    ($column:expr, $cells:ident => $body:expr) => {
        match $column {
            TraceColumn::Bit($cells) => $body,
            TraceColumn::B8($cells) => $body,
            TraceColumn::B16($cells) => $body,
            TraceColumn::B32($cells) => $body,
            TraceColumn::B64($cells) => $body,
            TraceColumn::B128($cells) => $body,
        }
    };
}

cell_access!(B8, Block8, u8, b8_at, set_b8, xor_b8, swap_b8_cells);
cell_access!(B16, Block16, u16, b16_at, set_b16, xor_b16, swap_b16_cells);
cell_access!(B32, Block32, u32, b32_at, set_b32, xor_b32, swap_b32_cells);
cell_access!(B64, Block64, u64, b64_at, set_b64, xor_b64, swap_b64_cells);

pub fn set_bit(trace: &mut ColumnTrace, col: usize, row: usize, value: Bit) {
    match &mut trace.columns[col] {
        TraceColumn::Bit(cells) => cells[row] = value,
        _ => panic!("column {col} is not Bit"),
    }
}

pub fn bit_rows(trace: &ColumnTrace, col: usize) -> Vec<usize> {
    match &trace.columns[col] {
        TraceColumn::Bit(cells) => (0..cells.len())
            .filter(|&row| cells[row] == Bit::ONE)
            .collect(),
        _ => panic!("column {col} is not Bit"),
    }
}

pub fn swap_rows(trace: &mut ColumnTrace, col: usize, a: usize, b: usize) {
    each_column!(&mut trace.columns[col], cells => cells.swap(a, b));
}

pub fn copy_row(trace: &mut ColumnTrace, cols: Range<usize>, src: usize, dst: usize) {
    for col in cols {
        each_column!(&mut trace.columns[col], cells => cells[dst] = cells[src]);
    }
}
