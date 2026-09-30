// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use super::wire_err;
use alloc::vec::Vec;
use flatbuffers::{Allocator, FlatBufferBuilder};
use hekate_core::errors::{Error, Result};
use hekate_core::trace::{ColumnTrace, ColumnType, Trace, TraceColumn};
use hekate_math::{Bit, Block8, Block16, Block32, Block64, Block128, Flat};

use crate::generated::program as fb;

#[cfg(target_endian = "big")]
compile_error!("wire serialization requires little-endian byte order");

const _: () = assert!(size_of::<Bit>() == 1);
const _: () = assert!(size_of::<Flat<Block8>>() == 1);
const _: () = assert!(size_of::<Flat<Block16>>() == 2);
const _: () = assert!(size_of::<Flat<Block32>>() == 4);
const _: () = assert!(size_of::<Flat<Block64>>() == 8);
const _: () = assert!(size_of::<Flat<Block128>>() == 16);

const COLUMN_OVERHEAD: usize = 64;
const TRACE_OVERHEAD: usize = 64;

pub fn serialize_trace<'a, A: Allocator + 'a>(
    fbb: &mut FlatBufferBuilder<'a, A>,
    trace: &impl Trace,
) -> flatbuffers::WIPOffset<fb::ColumnTrace<'a>> {
    let mut col_offsets = Vec::with_capacity(trace.columns().len());
    for col in trace.columns() {
        let col_type = column_type_to_fb(col.column_type());
        let bytes = column_as_bytes(col);
        let data = fbb.create_vector(bytes);

        let fb_col = fb::TraceColumn::create(
            fbb,
            &fb::TraceColumnArgs {
                col_type,
                data: Some(data),
            },
        );

        col_offsets.push(fb_col);
    }

    let columns = fbb.create_vector(&col_offsets);
    let num_rows = trace.num_rows().unwrap_or(0) as u64;

    fb::ColumnTrace::create(
        fbb,
        &fb::ColumnTraceArgs {
            columns: Some(columns),
            num_rows,
        },
    )
}

pub fn deserialize_trace(fb_trace: fb::ColumnTrace<'_>) -> Result<ColumnTrace> {
    let fb_columns = fb_trace
        .columns()
        .ok_or(wire_err("missing trace columns"))?;

    let num_rows = fb_trace.num_rows() as usize;
    if num_rows == 0 || !num_rows.is_power_of_two() {
        return Err(wire_err("trace num_rows must be a non-zero power of two"));
    }

    let num_vars = num_rows.trailing_zeros() as usize;

    let mut trace = ColumnTrace::new_secret(num_vars)?;

    for i in 0..fb_columns.len() {
        let fb_col = fb_columns.get(i);
        let col_type = column_type_from_fb(fb_col.col_type())?;

        let data = fb_col.data().ok_or(wire_err("missing column data"))?;

        let col = column_from_bytes(col_type, data.bytes(), num_rows)?;
        trace.add_column(col)?;
    }

    Ok(trace)
}

pub fn column_type_to_fb(ct: ColumnType) -> fb::ColumnType {
    match ct {
        ColumnType::Bit => fb::ColumnType::Bit,
        ColumnType::B8 => fb::ColumnType::B8,
        ColumnType::B16 => fb::ColumnType::B16,
        ColumnType::B32 => fb::ColumnType::B32,
        ColumnType::B64 => fb::ColumnType::B64,
        ColumnType::B128 => fb::ColumnType::B128,
    }
}

pub fn column_type_from_fb(ct: fb::ColumnType) -> Result<ColumnType> {
    match ct {
        fb::ColumnType::Bit => Ok(ColumnType::Bit),
        fb::ColumnType::B8 => Ok(ColumnType::B8),
        fb::ColumnType::B16 => Ok(ColumnType::B16),
        fb::ColumnType::B32 => Ok(ColumnType::B32),
        fb::ColumnType::B64 => Ok(ColumnType::B64),
        fb::ColumnType::B128 => Ok(ColumnType::B128),
        _ => Err(Error::Protocol {
            protocol: "wire",
            message: "unknown ColumnType",
        }),
    }
}

pub fn serialize_column_layout<'a, A: Allocator + 'a>(
    fbb: &mut FlatBufferBuilder<'a, A>,
    layout: &[ColumnType],
) -> flatbuffers::WIPOffset<flatbuffers::Vector<'a, fb::ColumnType>> {
    let fb_types: Vec<fb::ColumnType> = layout.iter().map(|ct| column_type_to_fb(*ct)).collect();
    fbb.create_vector(&fb_types)
}

pub fn deserialize_column_layout(
    vec: flatbuffers::Vector<'_, fb::ColumnType>,
) -> Result<Vec<ColumnType>> {
    let mut layout = Vec::with_capacity(vec.len());
    for i in 0..vec.len() {
        layout.push(column_type_from_fb(vec.get(i))?);
    }

    Ok(layout)
}

pub(crate) fn serialized_size_bound(trace: &impl Trace) -> usize {
    let columns: usize = trace
        .columns()
        .iter()
        .map(|col| column_as_bytes(col).len() + COLUMN_OVERHEAD)
        .sum();

    columns + TRACE_OVERHEAD
}

/// # Safety
///
/// All scalar types are `#[repr(transparent)]`
/// over LE integers. Valid only
/// on little-endian platforms.
fn column_as_bytes(col: &TraceColumn) -> &[u8] {
    match col {
        TraceColumn::Bit(v) => unsafe { core::slice::from_raw_parts(v.as_ptr().cast(), v.len()) },
        TraceColumn::B8(v) => unsafe { core::slice::from_raw_parts(v.as_ptr().cast(), v.len()) },
        TraceColumn::B16(v) => unsafe {
            core::slice::from_raw_parts(v.as_ptr().cast(), v.len() * 2)
        },
        TraceColumn::B32(v) => unsafe {
            core::slice::from_raw_parts(v.as_ptr().cast(), v.len() * 4)
        },
        TraceColumn::B64(v) => unsafe {
            core::slice::from_raw_parts(v.as_ptr().cast(), v.len() * 8)
        },
        TraceColumn::B128(v) => unsafe {
            core::slice::from_raw_parts(v.as_ptr().cast(), v.len() * 16)
        },
    }
}

fn column_from_bytes(col_type: ColumnType, data: &[u8], num_rows: usize) -> Result<TraceColumn> {
    let elem_size = col_type.byte_size();
    if data.len() != num_rows * elem_size {
        return Err(wire_err("column byte length mismatch"));
    }

    match col_type {
        ColumnType::Bit => {
            if data.iter().any(|&b| b > 1) {
                return Err(wire_err("Bit column contains non-binary value"));
            }

            let mut v = Vec::<Bit>::with_capacity(num_rows);

            // Safety:
            // Bit is #[repr(transparent)] over u8.
            // All bytes validated as 0 or 1 above.
            unsafe {
                core::ptr::copy_nonoverlapping(data.as_ptr(), v.as_mut_ptr().cast(), num_rows);
                v.set_len(num_rows);
            }

            Ok(TraceColumn::Bit(v))
        }
        ColumnType::B8 => {
            let mut v = Vec::<Flat<Block8>>::with_capacity(num_rows);
            unsafe {
                core::ptr::copy_nonoverlapping(data.as_ptr(), v.as_mut_ptr().cast(), num_rows);
                v.set_len(num_rows);
            }

            Ok(TraceColumn::B8(v))
        }
        ColumnType::B16 => {
            let mut v = Vec::<Flat<Block16>>::with_capacity(num_rows);
            unsafe {
                core::ptr::copy_nonoverlapping(data.as_ptr(), v.as_mut_ptr().cast(), num_rows * 2);
                v.set_len(num_rows);
            }

            Ok(TraceColumn::B16(v))
        }
        ColumnType::B32 => {
            let mut v = Vec::<Flat<Block32>>::with_capacity(num_rows);
            unsafe {
                core::ptr::copy_nonoverlapping(data.as_ptr(), v.as_mut_ptr().cast(), num_rows * 4);
                v.set_len(num_rows);
            }

            Ok(TraceColumn::B32(v))
        }
        ColumnType::B64 => {
            let mut v = Vec::<Flat<Block64>>::with_capacity(num_rows);
            unsafe {
                core::ptr::copy_nonoverlapping(data.as_ptr(), v.as_mut_ptr().cast(), num_rows * 8);
                v.set_len(num_rows);
            }

            Ok(TraceColumn::B64(v))
        }
        ColumnType::B128 => {
            let mut v = Vec::<Flat<Block128>>::with_capacity(num_rows);
            unsafe {
                core::ptr::copy_nonoverlapping(data.as_ptr(), v.as_mut_ptr().cast(), num_rows * 16);
                v.set_len(num_rows);
            }

            Ok(TraceColumn::B128(v))
        }
    }
}
