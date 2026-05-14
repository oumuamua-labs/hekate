// SPDX-License-Identifier: Apache-2.0
// This file is part of the hekate project.
// Copyright (C) 2026 Andrei Kochergin <andrei@oumuamua.dev>
// Copyright (C) 2026 Oumuamua Labs <info@oumuamua.dev>. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::errors;
use crate::poly::variant::PolyVariant;
use alloc::vec;
use alloc::vec::Vec;
use core::any::TypeId;
use core::fmt;
use core::mem::transmute;
use hekate_math::{
    Bit, Block8, Block16, Block32, Block64, Block128, CanonicalSerialize, Flat, FlatPromote,
    HardwareField, PackableField, TowerField,
};
use zeroize::Zeroize;
#[cfg(feature = "secure-memory")]
use zeroize::ZeroizeOnDrop;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    InvalidParameters {
        message: &'static str,
    },
    ColumnLengthMismatch {
        expected_len: usize,
        got_len: usize,
    },
    ColumnIndexOutOfBounds {
        col_idx: usize,
        num_cols: usize,
    },
    RowIndexOutOfBounds {
        row_idx: usize,
        num_rows: usize,
    },
    PointDimensionMismatch {
        expected_len: usize,
        got_len: usize,
    },
    ColumnTypeMismatch {
        col_idx: usize,
        expected: &'static str,
        got: &'static str,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameters { message } => {
                write!(f, "Trace invalid parameters: {message}")
            }
            Self::ColumnLengthMismatch {
                expected_len,
                got_len,
            } => write!(
                f,
                "Trace column length mismatch: expected {expected_len}, got {got_len}",
            ),
            Self::ColumnIndexOutOfBounds { col_idx, num_cols } => write!(
                f,
                "Trace column index out of bounds: col_idx={col_idx}, num_cols={num_cols}",
            ),
            Self::RowIndexOutOfBounds { row_idx, num_rows } => write!(
                f,
                "Trace row index out of bounds: row_idx={row_idx}, num_rows={num_rows}",
            ),
            Self::PointDimensionMismatch {
                expected_len,
                got_len,
            } => write!(
                f,
                "Trace evaluation point dimension mismatch: expected {expected_len}, got {got_len}",
            ),
            Self::ColumnTypeMismatch {
                col_idx,
                expected,
                got,
            } => write!(
                f,
                "Trace column type mismatch at col_idx={col_idx}: expected {expected}, got {got}",
            ),
        }
    }
}

/// Bound for the proving field `F`:
/// it must losslessly represent every
/// `ColumnType` used by the trace
/// (Bit, B8, B16, B32, B64, B128).
pub trait TraceCompatibleField:
    TowerField
    + HardwareField
    + PackableField
    + FlatPromote<Block8>
    + FlatPromote<Block16>
    + FlatPromote<Block32>
    + FlatPromote<Block64>
    + FlatPromote<Block128>
    + From<Bit>
    + From<Block8>
    + From<Block16>
    + From<Block32>
    + From<Block64>
    + From<Block128>
    + Send
    + Sync
{
}

impl<T> TraceCompatibleField for T where
    T: TowerField
        + HardwareField
        + PackableField
        + FlatPromote<Block8>
        + FlatPromote<Block16>
        + FlatPromote<Block32>
        + FlatPromote<Block64>
        + FlatPromote<Block128>
        + From<Bit>
        + From<Block8>
        + From<Block16>
        + From<Block32>
        + From<Block64>
        + From<Block128>
        + Send
        + Sync
{
}

// =========================================================
// TRACE TRAIT DEFINITION
// =========================================================

/// Execution-trace interface. Separates physical
/// storage (`TraceColumn`) from the virtual
/// polynomial view consumed by Sumcheck.
pub trait Trace: Send + Sync {
    /// `log2` of the trace height.
    fn num_vars(&self) -> usize;

    fn columns(&self) -> &[TraceColumn];

    /// `2^num_vars`.
    fn num_rows(&self) -> errors::Result<usize> {
        num_rows_from_num_vars(self.num_vars())
    }

    fn num_cols(&self) -> usize {
        self.columns().len()
    }

    fn column_layout(&self) -> Vec<ColumnType> {
        self.columns().iter().map(|col| col.column_type()).collect()
    }

    /// Read a single trace cell and lift
    /// it into `F` in the flat/hardware basis.
    fn get_element<F: TraceCompatibleField>(
        &self,
        col_idx: usize,
        row_idx: usize,
    ) -> errors::Result<Flat<F>> {
        let cols = self.columns();
        let num_cols = self.num_cols();

        if col_idx >= num_cols {
            return Err(Error::ColumnIndexOutOfBounds { col_idx, num_cols }.into());
        }

        let num_rows = self.num_rows()?;
        if row_idx >= num_rows {
            return Err(Error::RowIndexOutOfBounds { row_idx, num_rows }.into());
        }

        match &cols[col_idx] {
            TraceColumn::Bit(v) => Ok(Flat::from_raw(F::from(v[row_idx]))),
            TraceColumn::B8(v) => Ok(F::promote_flat(v[row_idx])),
            TraceColumn::B16(v) => Ok(F::promote_flat(v[row_idx])),
            TraceColumn::B32(v) => Ok(F::promote_flat(v[row_idx])),
            TraceColumn::B64(v) => Ok(F::promote_flat(v[row_idx])),
            TraceColumn::B128(v) => Ok(F::promote_flat(v[row_idx])),
        }
    }

    /// Zero-copy typed slice over a column.
    /// Fails if `F` does not match
    /// the column's storage type.
    fn get_column_slice<F: 'static>(&self, col_idx: usize) -> errors::Result<&[F]> {
        let cols = self.columns();
        let num_cols = self.num_cols();

        if col_idx >= num_cols {
            return Err(Error::ColumnIndexOutOfBounds { col_idx, num_cols }.into());
        }

        let got = core::any::type_name::<F>();

        match &cols[col_idx] {
            TraceColumn::Bit(vec) => {
                if TypeId::of::<F>() != TypeId::of::<Bit>() {
                    return Err(Error::ColumnTypeMismatch {
                        col_idx,
                        expected: "Bit",
                        got,
                    }
                    .into());
                }

                // SAFETY:
                // The TypeId check guarantees F == Bit.
                Ok(unsafe { transmute::<&[Bit], &[F]>(vec.as_slice()) })
            }
            TraceColumn::B8(vec) => {
                if TypeId::of::<F>() != TypeId::of::<Flat<Block8>>() {
                    return Err(Error::ColumnTypeMismatch {
                        col_idx,
                        expected: "Flat<Block8>",
                        got,
                    }
                    .into());
                }
                Ok(unsafe { transmute::<&[Flat<Block8>], &[F]>(vec.as_slice()) })
            }
            TraceColumn::B16(vec) => {
                if TypeId::of::<F>() != TypeId::of::<Flat<Block16>>() {
                    return Err(Error::ColumnTypeMismatch {
                        col_idx,
                        expected: "Flat<Block16>",
                        got,
                    }
                    .into());
                }
                Ok(unsafe { transmute::<&[Flat<Block16>], &[F]>(vec.as_slice()) })
            }
            TraceColumn::B32(vec) => {
                if TypeId::of::<F>() != TypeId::of::<Flat<Block32>>() {
                    return Err(Error::ColumnTypeMismatch {
                        col_idx,
                        expected: "Flat<Block32>",
                        got,
                    }
                    .into());
                }
                Ok(unsafe { transmute::<&[Flat<Block32>], &[F]>(vec.as_slice()) })
            }
            TraceColumn::B64(vec) => {
                if TypeId::of::<F>() != TypeId::of::<Flat<Block64>>() {
                    return Err(Error::ColumnTypeMismatch {
                        col_idx,
                        expected: "Flat<Block64>",
                        got,
                    }
                    .into());
                }
                Ok(unsafe { transmute::<&[Flat<Block64>], &[F]>(vec.as_slice()) })
            }
            TraceColumn::B128(vec) => {
                if TypeId::of::<F>() != TypeId::of::<Flat<Block128>>() {
                    return Err(Error::ColumnTypeMismatch {
                        col_idx,
                        expected: "Flat<Block128>",
                        got,
                    }
                    .into());
                }
                Ok(unsafe { transmute::<&[Flat<Block128>], &[F]>(vec.as_slice()) })
            }
        }
    }

    /// Map physical columns to the `PolyVariant`s
    /// consumed by Sumcheck. Default is a 1:1
    /// `BitSlice` / `B{N}Slice` mapping; chiplets
    /// that pack data (e.g. Keccak) override this
    /// to expose virtual bit-columns.
    fn get_poly_variants<F>(&'_ self) -> errors::Result<Vec<PolyVariant<'_, F>>>
    where
        F: TraceCompatibleField + 'static,
    {
        let cols = self.columns();
        let mut variants = Vec::with_capacity(cols.len());

        for (i, col) in cols.iter().enumerate() {
            if i >= self.num_cols() {
                return Err(errors::Error::Protocol {
                    protocol: "air",
                    message: "trace has fewer columns than required by AIR",
                });
            }

            let variant = if let Some(s) = col.as_bit_slice() {
                PolyVariant::BitSlice(s)
            } else if let Some(s) = col.as_b8_slice() {
                PolyVariant::B8Slice(s)
            } else if let Some(s) = col.as_b16_slice() {
                PolyVariant::B16Slice(s)
            } else if let Some(s) = col.as_b32_slice() {
                PolyVariant::B32Slice(s)
            } else if let Some(s) = col.as_b64_slice() {
                PolyVariant::B64Slice(s)
            } else if let Some(s) = col.as_b128_slice() {
                PolyVariant::B128Slice(s)
            } else {
                return Err(errors::Error::Protocol {
                    protocol: "air",
                    message: "unsupported trace column variant",
                });
            };

            variants.push(variant);
        }

        Ok(variants)
    }
}

// =========================================================
// COLUMN LAYOUT METADATA
// =========================================================

/// Column storage type, without the data itself.
/// Mixed-field traces require the verifier to know
/// the exact byte width of every opened column in
/// order to parse the raw LDT bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnType {
    /// AIR authors MUST `cs.assert_boolean(cs.col(idx))`
    /// on every `Bit` used as selector, constraint operand,
    /// or LogUp source, parse is byte-preserving, the
    /// verifier accepts any byte and lifts it to `F`.
    Bit,
    B8,
    B16,
    B32,
    B64,
    B128,
}

impl ColumnType {
    #[inline]
    pub const fn byte_size(&self) -> usize {
        match self {
            Self::Bit => 1,
            Self::B8 => 1,
            Self::B16 => 2,
            Self::B32 => 4,
            Self::B64 => 8,
            Self::B128 => 16,
        }
    }

    /// Parse a field element from its on-wire
    /// bytes (little-endian, hardware basis)
    /// without intermediate allocation.
    pub fn parse_from_bytes<F>(&self, bytes: &[u8]) -> Flat<F>
    where
        F: TraceCompatibleField,
    {
        match self {
            Self::Bit => Flat::from_raw(F::from(Bit(bytes[0]))),
            Self::B8 => F::promote_flat(Flat::from_raw(Block8(bytes[0]))),
            Self::B16 => {
                let mut buf = [0u8; 2];
                buf.copy_from_slice(&bytes[0..2]);

                F::promote_flat(Flat::from_raw(Block16(u16::from_le_bytes(buf))))
            }
            Self::B32 => {
                let mut buf = [0u8; 4];
                buf.copy_from_slice(&bytes[0..4]);

                F::promote_flat(Flat::from_raw(Block32(u32::from_le_bytes(buf))))
            }
            Self::B64 => {
                let mut buf = [0u8; 8];
                buf.copy_from_slice(&bytes[0..8]);

                F::promote_flat(Flat::from_raw(Block64(u64::from_le_bytes(buf))))
            }
            Self::B128 => {
                let mut buf = [0u8; 16];
                buf.copy_from_slice(&bytes[0..16]);

                F::promote_flat(Flat::from_raw(Block128(u128::from_le_bytes(buf))))
            }
        }
    }
}

/// One typed column of the execution trace.
/// Stored in hardware (flat) basis for all
/// non-`Bit` variants.
#[derive(Clone, Debug, Zeroize)]
#[cfg_attr(feature = "secure-memory", derive(ZeroizeOnDrop))]
pub enum TraceColumn {
    Bit(Vec<Bit>),
    B8(Vec<Flat<Block8>>),
    B16(Vec<Flat<Block16>>),
    B32(Vec<Flat<Block32>>),
    B64(Vec<Flat<Block64>>),
    B128(Vec<Flat<Block128>>),
}

impl TraceColumn {
    /// Narrow a vector of `Block128` into a column
    /// of `target_type` by truncating to the low bytes.
    pub fn from_data(data: Vec<Block128>, target_type: ColumnType) -> Self {
        match target_type {
            ColumnType::Bit => {
                let converted: Vec<Bit> = data
                    .iter()
                    .map(|val| {
                        let bytes = val.to_bytes();
                        Bit::from(bytes[0] & 1)
                    })
                    .collect();
                TraceColumn::Bit(converted)
            }
            ColumnType::B8 => {
                let converted: Vec<Flat<Block8>> = data
                    .iter()
                    .map(|val| {
                        let bytes = val.to_bytes();
                        Block8::from(bytes[0]).to_hardware()
                    })
                    .collect();
                TraceColumn::B8(converted)
            }
            ColumnType::B16 => {
                let converted: Vec<Flat<Block16>> = data
                    .iter()
                    .map(|val| {
                        let bytes = val.to_bytes();
                        let mut chunk = [0u8; 2];
                        chunk.copy_from_slice(&bytes[0..2]);

                        Block16::from(u16::from_le_bytes(chunk)).to_hardware()
                    })
                    .collect();
                TraceColumn::B16(converted)
            }
            ColumnType::B32 => {
                let converted: Vec<Flat<Block32>> = data
                    .iter()
                    .map(|val| {
                        let bytes = val.to_bytes();
                        let mut chunk = [0u8; 4];
                        chunk.copy_from_slice(&bytes[0..4]);

                        Block32::from(u32::from_le_bytes(chunk)).to_hardware()
                    })
                    .collect();
                TraceColumn::B32(converted)
            }
            ColumnType::B64 => {
                let converted: Vec<Flat<Block64>> = data
                    .iter()
                    .map(|val| {
                        let bytes = val.to_bytes();
                        let mut chunk = [0u8; 8];
                        chunk.copy_from_slice(&bytes[0..8]);

                        Block64::from(u64::from_le_bytes(chunk)).to_hardware()
                    })
                    .collect();
                TraceColumn::B64(converted)
            }
            ColumnType::B128 => {
                TraceColumn::B128(data.into_iter().map(|value| value.to_hardware()).collect())
            }
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Bit(v) => v.len(),
            Self::B8(v) => v.len(),
            Self::B16(v) => v.len(),
            Self::B32(v) => v.len(),
            Self::B64(v) => v.len(),
            Self::B128(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_all_zeros(&self) -> bool {
        match self {
            Self::Bit(v) => v.iter().all(|x| x.0 == 0),
            Self::B8(v) => v.iter().all(|x| x.into_raw().0 == 0),
            Self::B16(v) => v.iter().all(|x| x.into_raw().0 == 0),
            Self::B32(v) => v.iter().all(|x| x.into_raw().0 == 0),
            Self::B64(v) => v.iter().all(|x| x.into_raw().0 == 0),
            Self::B128(v) => v.iter().all(|x| x.into_raw() == Block128::ZERO),
        }
    }

    /// The `ColumnType` tag matching this variant.
    pub fn column_type(&self) -> ColumnType {
        match self {
            Self::Bit(_) => ColumnType::Bit,
            Self::B8(_) => ColumnType::B8,
            Self::B16(_) => ColumnType::B16,
            Self::B32(_) => ColumnType::B32,
            Self::B64(_) => ColumnType::B64,
            Self::B128(_) => ColumnType::B128,
        }
    }

    /// Append this row's little-endian
    /// serialization to `buf`. Used for
    /// Merkle leaf hashing and LDT opening bytes.
    pub fn append_bytes_at(&self, row_idx: usize, buf: &mut Vec<u8>) {
        match self {
            Self::Bit(v) => {
                buf.push(v[row_idx].0);
            }
            Self::B8(v) => {
                buf.push(v[row_idx].into_raw().0);
            }
            Self::B16(v) => {
                buf.extend_from_slice(&v[row_idx].into_raw().0.to_le_bytes());
            }
            Self::B32(v) => {
                buf.extend_from_slice(&v[row_idx].into_raw().0.to_le_bytes());
            }
            Self::B64(v) => {
                buf.extend_from_slice(&v[row_idx].into_raw().0.to_le_bytes());
            }
            Self::B128(v) => {
                buf.extend_from_slice(&v[row_idx].into_raw().0.to_le_bytes());
            }
        }
    }

    // ===========================================
    // Typed slice accessors
    // ===========================================

    pub fn as_bit_slice(&self) -> Option<&[Bit]> {
        if let Self::Bit(v) = self {
            Some(v)
        } else {
            None
        }
    }

    pub fn as_b8_slice(&self) -> Option<&[Flat<Block8>]> {
        if let Self::B8(v) = self {
            Some(v)
        } else {
            None
        }
    }
    pub fn as_b16_slice(&self) -> Option<&[Flat<Block16>]> {
        if let Self::B16(v) = self {
            Some(v)
        } else {
            None
        }
    }
    pub fn as_b32_slice(&self) -> Option<&[Flat<Block32>]> {
        if let Self::B32(v) = self {
            Some(v)
        } else {
            None
        }
    }
    pub fn as_b64_slice(&self) -> Option<&[Flat<Block64>]> {
        if let Self::B64(v) = self {
            Some(v)
        } else {
            None
        }
    }
    pub fn as_b128_slice(&self) -> Option<&[Flat<Block128>]> {
        if let Self::B128(v) = self {
            Some(v)
        } else {
            None
        }
    }
}

/// A concrete implementation of
/// Trace using column-major storage.
#[derive(Clone, Debug, Zeroize)]
#[cfg_attr(feature = "secure-memory", derive(ZeroizeOnDrop))]
pub struct ColumnTrace {
    pub columns: Vec<TraceColumn>,
    pub num_vars: usize,
}

impl Trace for ColumnTrace {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn columns(&self) -> &[TraceColumn] {
        &self.columns
    }
}

impl ColumnTrace {
    pub fn new(num_vars: usize) -> errors::Result<Self> {
        let num_rows = num_rows_from_num_vars(num_vars)?;
        if num_rows == 0 {
            return Err(Error::InvalidParameters {
                message: "trace height is zero",
            }
            .into());
        }

        Ok(Self {
            num_vars,
            columns: Vec::new(),
        })
    }

    /// Consume the trace and return
    /// its owned column storage.
    pub fn into_columns(mut self) -> Vec<TraceColumn> {
        core::mem::take(&mut self.columns)
    }

    pub fn add_column(&mut self, col: TraceColumn) -> errors::Result<()> {
        let expected_len = self.num_rows()?;
        let got_len = col.len();

        if got_len != expected_len {
            return Err(Error::ColumnLengthMismatch {
                expected_len,
                got_len,
            }
            .into());
        }

        self.columns.push(col);

        Ok(())
    }
}

pub trait IntoTraceColumn {
    fn into_trace_column(self) -> TraceColumn;
}

impl IntoTraceColumn for Vec<Bit> {
    fn into_trace_column(self) -> TraceColumn {
        TraceColumn::Bit(self)
    }
}

impl IntoTraceColumn for Vec<Block8> {
    fn into_trace_column(self) -> TraceColumn {
        TraceColumn::B8(self.into_iter().map(|value| value.to_hardware()).collect())
    }
}

impl IntoTraceColumn for Vec<Block16> {
    fn into_trace_column(self) -> TraceColumn {
        TraceColumn::B16(self.into_iter().map(|value| value.to_hardware()).collect())
    }
}

impl IntoTraceColumn for Vec<Block32> {
    fn into_trace_column(self) -> TraceColumn {
        TraceColumn::B32(self.into_iter().map(|value| value.to_hardware()).collect())
    }
}

impl IntoTraceColumn for Vec<Block64> {
    fn into_trace_column(self) -> TraceColumn {
        TraceColumn::B64(self.into_iter().map(|value| value.to_hardware()).collect())
    }
}

impl IntoTraceColumn for Vec<Block128> {
    fn into_trace_column(self) -> TraceColumn {
        TraceColumn::B128(self.into_iter().map(|value| value.to_hardware()).collect())
    }
}

impl IntoTraceColumn for Vec<Flat<Block8>> {
    fn into_trace_column(self) -> TraceColumn {
        TraceColumn::B8(self)
    }
}

impl IntoTraceColumn for Vec<Flat<Block16>> {
    fn into_trace_column(self) -> TraceColumn {
        TraceColumn::B16(self)
    }
}

impl IntoTraceColumn for Vec<Flat<Block32>> {
    fn into_trace_column(self) -> TraceColumn {
        TraceColumn::B32(self)
    }
}

impl IntoTraceColumn for Vec<Flat<Block64>> {
    fn into_trace_column(self) -> TraceColumn {
        TraceColumn::B64(self)
    }
}

impl IntoTraceColumn for Vec<Flat<Block128>> {
    fn into_trace_column(self) -> TraceColumn {
        TraceColumn::B128(self)
    }
}

/// Zero-copy byte views `(ptr, elem_width)` for
/// every column in a trace. Centralizes the
/// `#[repr(transparent)]`-dependent pointer casts.
pub fn get_col_views(columns: &[TraceColumn]) -> Vec<(&[u8], usize)> {
    columns
        .iter()
        .map(|col| match col {
            TraceColumn::Bit(v) => (
                unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) },
                1,
            ),
            TraceColumn::B8(v) => (
                unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) },
                1,
            ),
            TraceColumn::B16(v) => (
                unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) },
                2,
            ),
            TraceColumn::B32(v) => (
                unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) },
                4,
            ),
            TraceColumn::B64(v) => (
                unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 8) },
                8,
            ),
            TraceColumn::B128(v) => (
                unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 16) },
                16,
            ),
        })
        .collect()
}

// =========================================================
// TRACE BUILDER
// =========================================================

/// Schema-driven builder. Every column
/// is allocated zero-filled from the layout;
/// unfilled rows stay zero, so padding is implicit.
pub struct TraceBuilder {
    columns: Vec<TraceColumn>,
    num_vars: usize,
    num_rows: usize,
    cursors: Vec<usize>,
}

impl TraceBuilder {
    pub fn new(layout: &[ColumnType], num_vars: usize) -> errors::Result<Self> {
        let num_rows = num_rows_from_num_vars(num_vars)?;
        let columns = layout
            .iter()
            .map(|ct| match ct {
                ColumnType::Bit => TraceColumn::Bit(vec![Bit::ZERO; num_rows]),
                ColumnType::B8 => TraceColumn::B8(vec![Block8::ZERO.to_hardware(); num_rows]),
                ColumnType::B16 => TraceColumn::B16(vec![Block16::ZERO.to_hardware(); num_rows]),
                ColumnType::B32 => TraceColumn::B32(vec![Block32::ZERO.to_hardware(); num_rows]),
                ColumnType::B64 => TraceColumn::B64(vec![Block64::ZERO.to_hardware(); num_rows]),
                ColumnType::B128 => TraceColumn::B128(vec![Block128::ZERO.to_hardware(); num_rows]),
            })
            .collect();

        Ok(Self {
            columns,
            num_vars,
            num_rows,
            cursors: vec![0; layout.len()],
        })
    }

    /// `2^num_vars`.
    #[inline]
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    // =========================================================
    // Indexed write (random access)
    // =========================================================

    #[inline]
    pub fn set_bit(&mut self, col: usize, row: usize, val: Bit) -> errors::Result<()> {
        let num_rows = self.num_rows;
        let data = self.expect_bit_col(col)?;
        let slot = data.get_mut(row).ok_or(Error::RowIndexOutOfBounds {
            row_idx: row,
            num_rows,
        })?;

        *slot = val;

        Ok(())
    }

    #[inline]
    pub fn set_b8(&mut self, col: usize, row: usize, val: Block8) -> errors::Result<()> {
        let num_rows = self.num_rows;
        let data = self.expect_b8_col(col)?;
        let slot = data.get_mut(row).ok_or(Error::RowIndexOutOfBounds {
            row_idx: row,
            num_rows,
        })?;

        *slot = val.to_hardware();

        Ok(())
    }

    #[inline]
    pub fn set_b16(&mut self, col: usize, row: usize, val: Block16) -> errors::Result<()> {
        let num_rows = self.num_rows;
        let data = self.expect_b16_col(col)?;
        let slot = data.get_mut(row).ok_or(Error::RowIndexOutOfBounds {
            row_idx: row,
            num_rows,
        })?;

        *slot = val.to_hardware();

        Ok(())
    }

    #[inline]
    pub fn set_b32(&mut self, col: usize, row: usize, val: Block32) -> errors::Result<()> {
        let num_rows = self.num_rows;
        let data = self.expect_b32_col(col)?;
        let slot = data.get_mut(row).ok_or(Error::RowIndexOutOfBounds {
            row_idx: row,
            num_rows,
        })?;

        *slot = val.to_hardware();

        Ok(())
    }

    #[inline]
    pub fn set_b64(&mut self, col: usize, row: usize, val: Block64) -> errors::Result<()> {
        let num_rows = self.num_rows;
        let data = self.expect_b64_col(col)?;
        let slot = data.get_mut(row).ok_or(Error::RowIndexOutOfBounds {
            row_idx: row,
            num_rows,
        })?;

        *slot = val.to_hardware();

        Ok(())
    }

    #[inline]
    pub fn set_b128(&mut self, col: usize, row: usize, val: Block128) -> errors::Result<()> {
        let num_rows = self.num_rows;
        let data = self.expect_b128_col(col)?;
        let slot = data.get_mut(row).ok_or(Error::RowIndexOutOfBounds {
            row_idx: row,
            num_rows,
        })?;

        *slot = val.to_hardware();

        Ok(())
    }

    // =========================================================
    // Push write (sequential overwrite-at-cursor)
    // =========================================================

    #[inline]
    pub fn push_bit(&mut self, col: usize, val: Bit) -> errors::Result<()> {
        let row = self.cursor(col)?;
        self.set_bit(col, row, val)?;

        self.cursors[col] = row + 1;

        Ok(())
    }

    #[inline]
    pub fn push_b8(&mut self, col: usize, val: Block8) -> errors::Result<()> {
        let row = self.cursor(col)?;
        self.set_b8(col, row, val)?;

        self.cursors[col] = row + 1;

        Ok(())
    }

    #[inline]
    pub fn push_b16(&mut self, col: usize, val: Block16) -> errors::Result<()> {
        let row = self.cursor(col)?;
        self.set_b16(col, row, val)?;

        self.cursors[col] = row + 1;

        Ok(())
    }

    #[inline]
    pub fn push_b32(&mut self, col: usize, val: Block32) -> errors::Result<()> {
        let row = self.cursor(col)?;
        self.set_b32(col, row, val)?;

        self.cursors[col] = row + 1;

        Ok(())
    }

    #[inline]
    pub fn push_b64(&mut self, col: usize, val: Block64) -> errors::Result<()> {
        let row = self.cursor(col)?;
        self.set_b64(col, row, val)?;

        self.cursors[col] = row + 1;

        Ok(())
    }

    #[inline]
    pub fn push_b128(&mut self, col: usize, val: Block128) -> errors::Result<()> {
        let row = self.cursor(col)?;
        self.set_b128(col, row, val)?;

        self.cursors[col] = row + 1;

        Ok(())
    }

    // =========================================================
    // Array column helpers
    // =========================================================

    pub fn set_bit_array(&mut self, base: usize, row: usize, values: &[Bit]) -> errors::Result<()> {
        for (i, &val) in values.iter().enumerate() {
            self.set_bit(base + i, row, val)?;
        }

        Ok(())
    }

    pub fn set_b8_array(
        &mut self,
        base: usize,
        row: usize,
        values: &[Block8],
    ) -> errors::Result<()> {
        for (i, &val) in values.iter().enumerate() {
            self.set_b8(base + i, row, val)?;
        }

        Ok(())
    }

    pub fn set_b16_array(
        &mut self,
        base: usize,
        row: usize,
        values: &[Block16],
    ) -> errors::Result<()> {
        for (i, &val) in values.iter().enumerate() {
            self.set_b16(base + i, row, val)?;
        }

        Ok(())
    }

    pub fn set_b32_array(
        &mut self,
        base: usize,
        row: usize,
        values: &[Block32],
    ) -> errors::Result<()> {
        for (i, &val) in values.iter().enumerate() {
            self.set_b32(base + i, row, val)?;
        }

        Ok(())
    }

    pub fn set_b64_array(
        &mut self,
        base: usize,
        row: usize,
        values: &[Block64],
    ) -> errors::Result<()> {
        for (i, &val) in values.iter().enumerate() {
            self.set_b64(base + i, row, val)?;
        }

        Ok(())
    }

    pub fn set_b128_array(
        &mut self,
        base: usize,
        row: usize,
        values: &[Block128],
    ) -> errors::Result<()> {
        for (i, &val) in values.iter().enumerate() {
            self.set_b128(base + i, row, val)?;
        }

        Ok(())
    }

    // =========================================================
    // Selector helpers
    // =========================================================

    /// Write `ONE` into rows `[0, active_rows)`
    /// of a `Bit` column. The tail stays zero from allocation.
    pub fn fill_selector(&mut self, col: usize, active_rows: usize) -> errors::Result<()> {
        let limit = active_rows.min(self.num_rows);
        let data = self.expect_bit_col(col)?;

        for slot in data.iter_mut().take(limit) {
            *slot = Bit::ONE;
        }

        Ok(())
    }

    // =========================================================
    // Finalization
    // =========================================================

    /// Consume the builder and return a `ColumnTrace`.
    /// Column order matches the schema passed to `new`.
    pub fn build(self) -> ColumnTrace {
        ColumnTrace {
            columns: self.columns,
            num_vars: self.num_vars,
        }
    }

    // =========================================================
    // Internal helpers
    // =========================================================

    #[inline]
    fn cursor(&self, col: usize) -> errors::Result<usize> {
        self.cursors.get(col).copied().ok_or_else(|| {
            Error::ColumnIndexOutOfBounds {
                col_idx: col,
                num_cols: self.columns.len(),
            }
            .into()
        })
    }

    #[inline]
    fn expect_bit_col(&mut self, col: usize) -> errors::Result<&mut Vec<Bit>> {
        let num_cols = self.columns.len();
        let tc = self
            .columns
            .get_mut(col)
            .ok_or(Error::ColumnIndexOutOfBounds {
                col_idx: col,
                num_cols,
            })?;

        match tc {
            TraceColumn::Bit(data) => Ok(data),
            other => Err(Error::ColumnTypeMismatch {
                col_idx: col,
                expected: "Bit",
                got: other.column_type_name(),
            }
            .into()),
        }
    }

    #[inline]
    fn expect_b8_col(&mut self, col: usize) -> errors::Result<&mut Vec<Flat<Block8>>> {
        let num_cols = self.columns.len();
        let tc = self
            .columns
            .get_mut(col)
            .ok_or(Error::ColumnIndexOutOfBounds {
                col_idx: col,
                num_cols,
            })?;

        match tc {
            TraceColumn::B8(data) => Ok(data),
            other => Err(Error::ColumnTypeMismatch {
                col_idx: col,
                expected: "B8",
                got: other.column_type_name(),
            }
            .into()),
        }
    }

    #[inline]
    fn expect_b16_col(&mut self, col: usize) -> errors::Result<&mut Vec<Flat<Block16>>> {
        let num_cols = self.columns.len();
        let tc = self
            .columns
            .get_mut(col)
            .ok_or(Error::ColumnIndexOutOfBounds {
                col_idx: col,
                num_cols,
            })?;

        match tc {
            TraceColumn::B16(data) => Ok(data),
            other => Err(Error::ColumnTypeMismatch {
                col_idx: col,
                expected: "B16",
                got: other.column_type_name(),
            }
            .into()),
        }
    }

    #[inline]
    fn expect_b32_col(&mut self, col: usize) -> errors::Result<&mut Vec<Flat<Block32>>> {
        let num_cols = self.columns.len();
        let tc = self
            .columns
            .get_mut(col)
            .ok_or(Error::ColumnIndexOutOfBounds {
                col_idx: col,
                num_cols,
            })?;

        match tc {
            TraceColumn::B32(data) => Ok(data),
            other => Err(Error::ColumnTypeMismatch {
                col_idx: col,
                expected: "B32",
                got: other.column_type_name(),
            }
            .into()),
        }
    }

    #[inline]
    fn expect_b64_col(&mut self, col: usize) -> errors::Result<&mut Vec<Flat<Block64>>> {
        let num_cols = self.columns.len();
        let tc = self
            .columns
            .get_mut(col)
            .ok_or(Error::ColumnIndexOutOfBounds {
                col_idx: col,
                num_cols,
            })?;

        match tc {
            TraceColumn::B64(data) => Ok(data),
            other => Err(Error::ColumnTypeMismatch {
                col_idx: col,
                expected: "B64",
                got: other.column_type_name(),
            }
            .into()),
        }
    }

    #[inline]
    fn expect_b128_col(&mut self, col: usize) -> errors::Result<&mut Vec<Flat<Block128>>> {
        let num_cols = self.columns.len();
        let tc = self
            .columns
            .get_mut(col)
            .ok_or(Error::ColumnIndexOutOfBounds {
                col_idx: col,
                num_cols,
            })?;

        match tc {
            TraceColumn::B128(data) => Ok(data),
            other => Err(Error::ColumnTypeMismatch {
                col_idx: col,
                expected: "B128",
                got: other.column_type_name(),
            }
            .into()),
        }
    }
}

impl TraceColumn {
    fn column_type_name(&self) -> &'static str {
        match self {
            Self::Bit(_) => "Bit",
            Self::B8(_) => "B8",
            Self::B16(_) => "B16",
            Self::B32(_) => "B32",
            Self::B64(_) => "B64",
            Self::B128(_) => "B128",
        }
    }
}

fn num_rows_from_num_vars(num_vars: usize) -> errors::Result<usize> {
    let num_vars_u32 = match u32::try_from(num_vars) {
        Ok(v) => v,
        Err(_) => {
            return Err(Error::InvalidParameters {
                message: "num_vars too large",
            }
            .into());
        }
    };

    let Some(num_rows) = 1usize.checked_shl(num_vars_u32) else {
        return Err(Error::InvalidParameters {
            message: "num_rows overflow",
        }
        .into());
    };

    if num_rows == 0 {
        return Err(Error::InvalidParameters {
            message: "num_rows is zero",
        }
        .into());
    }

    Ok(num_rows)
}

// =================================================================
// UNIT TESTS
// =================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors;
    use hekate_math::HardwareField;

    fn create_mock_trace(num_vars: usize) -> ColumnTrace {
        ColumnTrace::new(num_vars).unwrap()
    }

    #[test]
    fn trace_construction_basic() {
        let num_vars = 3;
        let mut trace = create_mock_trace(num_vars);

        let col_data = vec![Block128::from(1u8); 8];
        trace.add_column(col_data.into_trace_column()).unwrap();

        assert_eq!(trace.num_rows().unwrap(), 8);
        assert_eq!(trace.num_cols(), 1);
        assert_eq!(trace.num_vars, 3);
    }

    #[test]
    fn trace_add_column_wrong_len() {
        let num_vars = 2;
        let mut trace = create_mock_trace(num_vars);

        let col_data = vec![Block128::ZERO; 5];
        let err = trace
            .add_column(col_data.into_trace_column())
            .expect_err("Expected length mismatch error");

        assert!(matches!(
            err,
            errors::Error::Trace(Error::ColumnLengthMismatch { .. })
        ));
    }

    #[test]
    fn trace_get_element_mixed_types() {
        let num_vars = 1;
        let mut trace = create_mock_trace(num_vars);

        trace
            .add_column(TraceColumn::Bit(vec![Bit::new(0), Bit::new(1)]))
            .unwrap();
        trace
            .add_column(vec![Block32::from(10u32), Block32::from(20u32)].into_trace_column())
            .unwrap();

        let val0_r0: Flat<Block128> = trace.get_element(0, 0).unwrap();
        let val0_r1: Flat<Block128> = trace.get_element(0, 1).unwrap();
        let val1_r0: Flat<Block128> = trace.get_element(1, 0).unwrap();
        let val1_r1: Flat<Block128> = trace.get_element(1, 1).unwrap();

        assert_eq!(val0_r0.into_raw(), Block128::ZERO);
        assert_eq!(val0_r1.into_raw(), Block128::ONE);

        let expected_10 = Block128::promote_flat(Block32::from(10u32).to_hardware()).into_raw();
        let expected_20 = Block128::promote_flat(Block32::from(20u32).to_hardware()).into_raw();

        assert_eq!(val1_r0.into_raw(), expected_10);
        assert_eq!(val1_r1.into_raw(), expected_20);
    }

    #[test]
    fn get_element_oob_row() {
        let mut trace = create_mock_trace(1);
        trace
            .add_column(TraceColumn::Bit(vec![Bit::ZERO; 2]))
            .unwrap();
        trace
            .get_element::<Block128>(0, 2)
            .expect_err("Expected out-of-bounds row error");
    }

    #[test]
    fn get_element_oob_col() {
        let trace = create_mock_trace(1);
        trace
            .get_element::<Block128>(0, 0)
            .expect_err("Expected out-of-bounds column error");
    }

    // ===========================================
    // SAFETY & SLICE TESTS
    // ===========================================

    #[test]
    fn get_column_slice_correct_type() {
        let mut trace = create_mock_trace(2);
        let data = vec![
            Block32::from(1u32),
            Block32::from(2u32),
            Block32::from(3u32),
            Block32::from(4u32),
        ];
        trace.add_column(data.clone().into_trace_column()).unwrap();

        let expected_hw: Vec<Flat<Block32>> = data.into_iter().map(|x| x.to_hardware()).collect();

        let slice: &[Flat<Block32>] = trace.get_column_slice(0).unwrap();
        assert_eq!(slice, expected_hw.as_slice());
    }

    #[test]
    fn get_column_slice_wrong_type() {
        let mut trace = create_mock_trace(1);
        trace
            .add_column(vec![Block128::ZERO; 2].into_trace_column())
            .unwrap();

        trace
            .get_column_slice::<Flat<Block32>>(0)
            .expect_err("Expected column type mismatch error");
    }

    #[test]
    fn trace_stores_hardware_basis() {
        let mut trace = create_mock_trace(2);

        let tower_data = vec![
            Block32::from(42u32),
            Block32::from(13u32),
            Block32::from(255u32),
            Block32::from(1u32),
        ];

        let expected_hardware: Vec<Block32> = tower_data
            .iter()
            .map(|x| x.to_hardware().into_raw())
            .collect();

        trace.add_column(tower_data.into_trace_column()).unwrap();

        let stored: &[Flat<Block32>] = trace.get_column_slice(0).unwrap();

        for (i, (&stored_val, &expected_val)) in
            stored.iter().zip(expected_hardware.iter()).enumerate()
        {
            assert_eq!(
                stored_val.into_raw(),
                expected_val,
                "Row {}: stored value {:?} != expected hardware {:?}",
                i,
                stored_val,
                expected_val
            );
        }
    }

    #[test]
    fn trace_hardware_basis_homomorphism() {
        let mut trace = create_mock_trace(3);

        let a_tower = vec![Block32::from(5u32); 8];
        let b_tower = vec![Block32::from(7u32); 8];

        trace
            .add_column(a_tower.clone().into_trace_column())
            .unwrap();
        trace
            .add_column(b_tower.clone().into_trace_column())
            .unwrap();

        let a_stored: &[Flat<Block32>] = trace.get_column_slice(0).unwrap();
        let b_stored: &[Flat<Block32>] = trace.get_column_slice(1).unwrap();

        let a_hw_expected = a_tower[0].to_hardware().into_raw();
        let b_hw_expected = b_tower[0].to_hardware().into_raw();

        assert_eq!(a_stored[0].into_raw(), a_hw_expected);
        assert_eq!(b_stored[0].into_raw(), b_hw_expected);

        let product_hw = a_stored[0] * b_stored[0];
        let product_expected = (a_tower[0] * b_tower[0]).to_hardware().into_raw();

        assert_eq!(product_hw.into_raw(), product_expected);
    }

    // =========================================================
    // TRACE BUILDER TESTS
    // =========================================================

    #[test]
    fn trace_builder_construction_and_auto_padding() {
        let layout = &[ColumnType::B32, ColumnType::Bit];
        let tb = TraceBuilder::new(layout, 2).unwrap(); // 4 rows
        assert_eq!(tb.num_rows(), 4);

        let trace = tb.build();
        assert_eq!(trace.num_cols(), 2);
        assert_eq!(trace.num_rows().unwrap(), 4);

        // All values should be zero (auto-padding)
        assert!(trace.columns[0].is_all_zeros());
        assert!(trace.columns[1].is_all_zeros());
    }

    #[test]
    fn trace_builder_set_b32_stores_hardware_basis() {
        let layout = &[ColumnType::B32];
        let mut tb = TraceBuilder::new(layout, 1).unwrap(); // 2 rows

        tb.set_b32(0, 0, Block32::from(42u32)).unwrap();
        tb.set_b32(0, 1, Block32::from(13u32)).unwrap();

        let trace = tb.build();
        let stored: &[Flat<Block32>] = trace.get_column_slice(0).unwrap();

        assert_eq!(stored[0], Block32::from(42u32).to_hardware());
        assert_eq!(stored[1], Block32::from(13u32).to_hardware());
    }

    #[test]
    fn trace_builder_column_ordering_matches_schema() {
        let layout = &[ColumnType::Bit, ColumnType::B32, ColumnType::B128];
        let mut tb = TraceBuilder::new(layout, 1).unwrap();

        tb.set_bit(0, 0, Bit::ONE).unwrap();
        tb.set_b32(1, 0, Block32::from(99u32)).unwrap();
        tb.set_b128(2, 0, Block128::from(7u8)).unwrap();

        let trace = tb.build();
        assert_eq!(trace.columns[0].column_type(), ColumnType::Bit);
        assert_eq!(trace.columns[1].column_type(), ColumnType::B32);
        assert_eq!(trace.columns[2].column_type(), ColumnType::B128);
    }

    #[test]
    fn trace_builder_type_mismatch_returns_error() {
        let layout = &[ColumnType::Bit];
        let mut tb = TraceBuilder::new(layout, 1).unwrap();

        let err = tb.set_b32(0, 0, Block32::ZERO);
        assert!(err.is_err());
    }

    #[test]
    fn trace_builder_row_out_of_bounds_returns_error() {
        let layout = &[ColumnType::B32];
        let mut tb = TraceBuilder::new(layout, 1).unwrap(); // 2 rows

        let err = tb.set_b32(0, 2, Block32::ZERO);
        assert!(err.is_err());
    }

    #[test]
    fn trace_builder_col_out_of_bounds_returns_error() {
        let layout = &[ColumnType::B32];
        let mut tb = TraceBuilder::new(layout, 1).unwrap();

        let err = tb.set_b32(1, 0, Block32::ZERO);
        assert!(err.is_err());
    }

    #[test]
    fn trace_builder_fill_selector() {
        let layout = &[ColumnType::Bit];
        let mut tb = TraceBuilder::new(layout, 2).unwrap(); // 4 rows

        tb.fill_selector(0, 3).unwrap(); // rows 0,1,2 = ONE

        let trace = tb.build();
        let bits = trace.columns[0].as_bit_slice().unwrap();
        assert_eq!(bits[0], Bit::ONE);
        assert_eq!(bits[1], Bit::ONE);
        assert_eq!(bits[2], Bit::ONE);
        assert_eq!(bits[3], Bit::ZERO); // padding
    }

    #[test]
    fn trace_builder_push_mode() {
        let layout = &[ColumnType::B32, ColumnType::Bit];
        let mut tb = TraceBuilder::new(layout, 1).unwrap(); // 2 rows

        tb.push_b32(0, Block32::from(10u32)).unwrap();
        tb.push_b32(0, Block32::from(20u32)).unwrap();
        tb.push_bit(1, Bit::ONE).unwrap();
        // row 1 of col 1 stays zero (auto-pad)

        let trace = tb.build();
        let b32s: &[Flat<Block32>] = trace.get_column_slice(0).unwrap();
        assert_eq!(b32s[0], Block32::from(10u32).to_hardware());
        assert_eq!(b32s[1], Block32::from(20u32).to_hardware());

        let bits = trace.columns[1].as_bit_slice().unwrap();
        assert_eq!(bits[0], Bit::ONE);
        assert_eq!(bits[1], Bit::ZERO);
    }

    #[test]
    fn trace_builder_set_b32_array() {
        let layout = &[ColumnType::B32, ColumnType::B32, ColumnType::B32];
        let mut tb = TraceBuilder::new(layout, 1).unwrap(); // 2 rows

        let vals = [
            Block32::from(1u32),
            Block32::from(2u32),
            Block32::from(3u32),
        ];
        tb.set_b32_array(0, 0, &vals).unwrap();

        let trace = tb.build();
        for (i, &expected) in vals.iter().enumerate() {
            let stored: &[Flat<Block32>] = trace.get_column_slice(i).unwrap();
            assert_eq!(stored[0], expected.to_hardware());
        }
    }

    #[test]
    fn trace_builder_set_b8() {
        let layout = &[ColumnType::B8];
        let mut tb = TraceBuilder::new(layout, 1).unwrap();

        tb.set_b8(0, 0, Block8(0xAB)).unwrap();
        tb.set_b8(0, 1, Block8(0xCD)).unwrap();

        let trace = tb.build();
        let stored: &[Flat<Block8>] = trace.get_column_slice(0).unwrap();
        assert_eq!(stored[0], Block8(0xAB).to_hardware());
        assert_eq!(stored[1], Block8(0xCD).to_hardware());
    }

    #[test]
    fn trace_builder_set_b16() {
        let layout = &[ColumnType::B16];
        let mut tb = TraceBuilder::new(layout, 1).unwrap();

        tb.set_b16(0, 0, Block16(1000)).unwrap();
        tb.set_b16(0, 1, Block16(2000)).unwrap();

        let trace = tb.build();
        let stored: &[Flat<Block16>] = trace.get_column_slice(0).unwrap();
        assert_eq!(stored[0], Block16(1000).to_hardware());
        assert_eq!(stored[1], Block16(2000).to_hardware());
    }

    #[test]
    fn trace_builder_set_b64() {
        let layout = &[ColumnType::B64];
        let mut tb = TraceBuilder::new(layout, 1).unwrap();

        tb.set_b64(0, 0, Block64(0xDEADBEEF_CAFEBABE)).unwrap();

        let trace = tb.build();
        let stored: &[Flat<Block64>] = trace.get_column_slice(0).unwrap();
        assert_eq!(stored[0], Block64(0xDEADBEEF_CAFEBABE).to_hardware());
        assert_eq!(stored[1], Block64::ZERO.to_hardware()); // auto-padding
    }

    #[test]
    fn trace_builder_set_b128() {
        let layout = &[ColumnType::B128];
        let mut tb = TraceBuilder::new(layout, 1).unwrap();

        let val = Block128::from(0xFFu8);
        tb.set_b128(0, 0, val).unwrap();

        let trace = tb.build();
        let stored: &[Flat<Block128>] = trace.get_column_slice(0).unwrap();
        assert_eq!(stored[0], val.to_hardware());
    }

    #[test]
    fn trace_builder_push_all_types() {
        let layout = &[
            ColumnType::Bit,
            ColumnType::B8,
            ColumnType::B16,
            ColumnType::B32,
            ColumnType::B64,
            ColumnType::B128,
        ];
        let mut tb = TraceBuilder::new(layout, 1).unwrap(); // 2 rows

        tb.push_bit(0, Bit::ONE).unwrap();
        tb.push_b8(1, Block8(0x42)).unwrap();
        tb.push_b16(2, Block16(1234)).unwrap();
        tb.push_b32(3, Block32::from(5678u32)).unwrap();
        tb.push_b64(4, Block64(9999)).unwrap();
        tb.push_b128(5, Block128::from(77u8)).unwrap();

        let trace = tb.build();
        assert_eq!(trace.num_cols(), 6);

        let bits = trace.columns[0].as_bit_slice().unwrap();
        assert_eq!(bits[0], Bit::ONE);
        assert_eq!(bits[1], Bit::ZERO); // auto-pad

        let b8s: &[Flat<Block8>] = trace.get_column_slice(1).unwrap();
        assert_eq!(b8s[0], Block8(0x42).to_hardware());

        let b16s: &[Flat<Block16>] = trace.get_column_slice(2).unwrap();
        assert_eq!(b16s[0], Block16(1234).to_hardware());

        let b32s: &[Flat<Block32>] = trace.get_column_slice(3).unwrap();
        assert_eq!(b32s[0], Block32::from(5678u32).to_hardware());

        let b64s: &[Flat<Block64>] = trace.get_column_slice(4).unwrap();
        assert_eq!(b64s[0], Block64(9999).to_hardware());

        let b128s: &[Flat<Block128>] = trace.get_column_slice(5).unwrap();
        assert_eq!(b128s[0], Block128::from(77u8).to_hardware());
    }

    #[test]
    fn trace_builder_type_mismatch_all_setters() {
        // B32 column, every non-B32 setter should fail
        let layout = &[ColumnType::B32];
        let mut tb = TraceBuilder::new(layout, 1).unwrap();

        assert!(tb.set_bit(0, 0, Bit::ONE).is_err());
        assert!(tb.set_b8(0, 0, Block8(1)).is_err());
        assert!(tb.set_b16(0, 0, Block16(1)).is_err());
        assert!(tb.set_b64(0, 0, Block64(1)).is_err());
        assert!(tb.set_b128(0, 0, Block128::ONE).is_err());

        // Correct type succeeds
        assert!(tb.set_b32(0, 0, Block32::ONE).is_ok());
    }

    #[test]
    fn trace_builder_array_setters_all_types() {
        let layout = &[
            ColumnType::Bit,
            ColumnType::Bit,
            ColumnType::B8,
            ColumnType::B8,
            ColumnType::B64,
            ColumnType::B64,
            ColumnType::B128,
            ColumnType::B128,
        ];
        let mut tb = TraceBuilder::new(layout, 1).unwrap();

        tb.set_bit_array(0, 0, &[Bit::ONE, Bit::ZERO]).unwrap();
        tb.set_b8_array(2, 0, &[Block8(10), Block8(20)]).unwrap();
        tb.set_b64_array(4, 0, &[Block64(100), Block64(200)])
            .unwrap();
        tb.set_b128_array(6, 0, &[Block128::ONE, Block128::from(2u8)])
            .unwrap();

        let trace = tb.build();

        let bits = trace.columns[0].as_bit_slice().unwrap();
        assert_eq!(bits[0], Bit::ONE);

        let bits1 = trace.columns[1].as_bit_slice().unwrap();
        assert_eq!(bits1[0], Bit::ZERO);

        let b8s: &[Flat<Block8>] = trace.get_column_slice(2).unwrap();
        assert_eq!(b8s[0], Block8(10).to_hardware());

        let b8s1: &[Flat<Block8>] = trace.get_column_slice(3).unwrap();
        assert_eq!(b8s1[0], Block8(20).to_hardware());
    }

    #[test]
    fn trace_builder_invalid_num_vars() {
        let layout = &[ColumnType::B32];
        // num_vars too large to shift
        assert!(TraceBuilder::new(layout, 128).is_err());
    }
}
