use crate::mutation::Mutation;
use hekate_core::trace::{ColumnTrace, TraceColumn};
use hekate_math::{Bit, Block8, Block16, Block32, Block64, Block128, Flat, TowerField};
use hekate_program::ProgramWitness;

/// Mutate `witness` in place per `mutation`.
pub fn apply_mutation<F: TowerField>(
    witness: &mut ProgramWitness<F, ColumnTrace>,
    mutation: &Mutation,
) {
    match mutation {
        Mutation::BitFlip {
            target,
            col,
            row,
            mask,
        } => xor_cell(target.resolve_mut(witness), *col, *row, *mask),
        Mutation::OutOfBounds {
            target,
            col,
            row,
            value,
        } => write_cell(target.resolve_mut(witness), *col, *row, *value),
        Mutation::FlipSelector { target, col, row } => {
            flip_bit_cell(target.resolve_mut(witness), *col, *row)
        }
        Mutation::SwapRows {
            target,
            row_a,
            row_b,
        } => {
            let trace = target.resolve_mut(witness);
            for column in &mut trace.columns {
                swap_cells(column, *row_a, *row_b);
            }
        }
        Mutation::SwapColumns {
            target,
            cols,
            row_a,
            row_b,
        } => {
            let trace = target.resolve_mut(witness);
            for col in cols {
                swap_cells(column_at_mut(trace, *col), *row_a, *row_b);
            }
        }
        Mutation::DuplicateRow {
            target,
            src_row,
            dst_row,
        } => {
            let trace = target.resolve_mut(witness);
            for column in &mut trace.columns {
                copy_cell(column, *src_row, *dst_row);
            }
        }
        Mutation::CopyColumns {
            target,
            cols,
            src_row,
            dst_row,
        } => {
            let trace = target.resolve_mut(witness);
            for col in cols {
                copy_cell(column_at_mut(trace, *col), *src_row, *dst_row);
            }
        }
        Mutation::ColumnUniformWrite { target, col, value } => {
            let trace = target.resolve_mut(witness);
            let column = column_at_mut(trace, *col);
            let n = column_len(column);

            for row in 0..n {
                write_value(column, *col, row, *value);
            }
        }
        Mutation::RowSegmentZero { target, rows, cols } => {
            let trace = target.resolve_mut(witness);
            for &col in cols {
                for &row in rows {
                    write_value(column_at_mut(trace, col), col, row, 0);
                }
            }
        }
        Mutation::MonotonicReplace {
            target,
            col,
            base,
            step,
        } => {
            let trace = target.resolve_mut(witness);
            let column = column_at_mut(trace, *col);
            let n = column_len(column);

            for row in 0..n {
                let value = base.wrapping_add((row as u128).wrapping_mul(*step));
                write_value(column, *col, row, value);
            }
        }
        Mutation::Compound(mutations) => {
            for m in mutations {
                apply_mutation(witness, m);
            }
        }
    }
}

fn column_len(column: &TraceColumn) -> usize {
    match column {
        TraceColumn::Bit(v) => v.len(),
        TraceColumn::B8(v) => v.len(),
        TraceColumn::B16(v) => v.len(),
        TraceColumn::B32(v) => v.len(),
        TraceColumn::B64(v) => v.len(),
        TraceColumn::B128(v) => v.len(),
    }
}

fn write_value(column: &mut TraceColumn, col: usize, row: usize, value: u128) {
    match column {
        TraceColumn::Bit(v) => *row_mut(v, col, row) = Bit::from((value & 1) as u8),
        TraceColumn::B8(v) => *row_mut(v, col, row) = Flat::from_raw(Block8((value & 0xFF) as u8)),
        TraceColumn::B16(v) => {
            *row_mut(v, col, row) = Flat::from_raw(Block16((value & 0xFFFF) as u16))
        }
        TraceColumn::B32(v) => {
            *row_mut(v, col, row) = Flat::from_raw(Block32((value & 0xFFFF_FFFF) as u32))
        }
        TraceColumn::B64(v) => {
            *row_mut(v, col, row) = Flat::from_raw(Block64((value & 0xFFFF_FFFF_FFFF_FFFF) as u64))
        }
        TraceColumn::B128(v) => *row_mut(v, col, row) = Flat::from_raw(Block128(value)),
    }
}

fn column_at_mut(trace: &mut ColumnTrace, col: usize) -> &mut TraceColumn {
    let n = trace.columns.len();
    trace
        .columns
        .get_mut(col)
        .unwrap_or_else(|| panic!("apply_mutation: column {col} out of bounds (trace has {n})"))
}

fn xor_cell(trace: &mut ColumnTrace, col: usize, row: usize, mask: u128) {
    match column_at_mut(trace, col) {
        TraceColumn::Bit(v) => {
            let cell = row_mut(v, col, row);
            *cell = Bit::from(cell.0 ^ ((mask & 1) as u8));
        }
        TraceColumn::B8(v) => {
            let cell = row_mut(v, col, row);
            *cell = Flat::from_raw(Block8(cell.into_raw().0 ^ ((mask & 0xFF) as u8)));
        }
        TraceColumn::B16(v) => {
            let cell = row_mut(v, col, row);
            *cell = Flat::from_raw(Block16(cell.into_raw().0 ^ ((mask & 0xFFFF) as u16)));
        }
        TraceColumn::B32(v) => {
            let cell = row_mut(v, col, row);
            *cell = Flat::from_raw(Block32(cell.into_raw().0 ^ ((mask & 0xFFFF_FFFF) as u32)));
        }
        TraceColumn::B64(v) => {
            let cell = row_mut(v, col, row);
            *cell = Flat::from_raw(Block64(
                cell.into_raw().0 ^ ((mask & 0xFFFF_FFFF_FFFF_FFFF) as u64),
            ));
        }
        TraceColumn::B128(v) => {
            let cell = row_mut(v, col, row);
            *cell = Flat::from_raw(Block128(cell.into_raw().0 ^ mask));
        }
    }
}

fn write_cell(trace: &mut ColumnTrace, col: usize, row: usize, value: u128) {
    match column_at_mut(trace, col) {
        TraceColumn::Bit(v) => *row_mut(v, col, row) = Bit::from((value & 1) as u8),
        TraceColumn::B8(v) => *row_mut(v, col, row) = Flat::from_raw(Block8((value & 0xFF) as u8)),
        TraceColumn::B16(v) => {
            *row_mut(v, col, row) = Flat::from_raw(Block16((value & 0xFFFF) as u16))
        }
        TraceColumn::B32(v) => {
            *row_mut(v, col, row) = Flat::from_raw(Block32((value & 0xFFFF_FFFF) as u32))
        }
        TraceColumn::B64(v) => {
            *row_mut(v, col, row) = Flat::from_raw(Block64((value & 0xFFFF_FFFF_FFFF_FFFF) as u64))
        }
        TraceColumn::B128(v) => *row_mut(v, col, row) = Flat::from_raw(Block128(value)),
    }
}

fn flip_bit_cell(trace: &mut ColumnTrace, col: usize, row: usize) {
    match column_at_mut(trace, col) {
        TraceColumn::Bit(v) => {
            let cell = row_mut(v, col, row);
            *cell = Bit::from(cell.0 ^ 1);
        }
        other => panic!(
            "FlipSelector requires a Bit column; column {col} is {:?}",
            other.column_type()
        ),
    }
}

fn swap_cells(column: &mut TraceColumn, a: usize, b: usize) {
    match column {
        TraceColumn::Bit(v) => v.swap(a, b),
        TraceColumn::B8(v) => v.swap(a, b),
        TraceColumn::B16(v) => v.swap(a, b),
        TraceColumn::B32(v) => v.swap(a, b),
        TraceColumn::B64(v) => v.swap(a, b),
        TraceColumn::B128(v) => v.swap(a, b),
    }
}

fn copy_cell(column: &mut TraceColumn, src: usize, dst: usize) {
    match column {
        TraceColumn::Bit(v) => v[dst] = v[src],
        TraceColumn::B8(v) => v[dst] = v[src],
        TraceColumn::B16(v) => v[dst] = v[src],
        TraceColumn::B32(v) => v[dst] = v[src],
        TraceColumn::B64(v) => v[dst] = v[src],
        TraceColumn::B128(v) => v[dst] = v[src],
    }
}

fn row_mut<T>(v: &mut [T], col: usize, row: usize) -> &mut T {
    let n = v.len();
    v.get_mut(row).unwrap_or_else(|| {
        panic!("apply_mutation: row {row} out of bounds in column {col} (column has {n})")
    })
}
