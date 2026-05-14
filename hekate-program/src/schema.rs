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

//! Typed Column Schema.
//!
//! `define_columns!` generates index constants,
//! `NUM_COLUMNS`, and `build_layout()` from a
//! single definition. One source of truth —
//! indices and layout can't drift from each other.
//!
//! # Scalar columns
//!
//! `NAME: Type` → `const NAME: usize = <offset>`.
//!
//! # Array columns
//!
//! `NAME: [Type; N]` → `const NAME: usize = <base>`.
//! Offset advances by N.
//!
//! # Example
//!
//! ```ignore
//! define_columns! {
//!     pub FibColumns {
//!         A: B32,
//!         B: B32,
//!         Q: Bit,
//!     }
//! }
//!
//! // FibColumns::A == 0
//! // FibColumns::B == 1
//! // FibColumns::Q == 2
//! // FibColumns::NUM_COLUMNS == 3
//! // FibColumns::build_layout() → [B32, B32, Bit]
//! ```

use alloc::vec::Vec;

pub use alloc::vec::Vec as SchemaVec;
pub use hekate_core::trace::ColumnType;

/// Define a typed column schema.
///
/// Generates index constants, `NUM_COLUMNS`,
/// and `build_layout() -> Vec<ColumnType>`.
#[macro_export]
macro_rules! define_columns {
    (
        $(#[$meta:meta])*
        $vis:vis $name:ident {
            $( $field:ident : $kind:tt ),* $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug)]
        $vis struct $name;

        impl $name {
            $crate::define_columns!(@consts 0usize; $( $field : $kind, )*);

            /// Column layout derived
            /// from the schema definition.
            #[allow(unused_mut)]
            pub fn build_layout() -> $crate::schema::SchemaVec<$crate::schema::ColumnType> {
                let mut v = $crate::schema::SchemaVec::with_capacity(Self::NUM_COLUMNS);
                $( $crate::define_columns!(@push v, $kind); )*

                v
            }
        }
    };

    (@consts $offset:expr;) => {
        pub const NUM_COLUMNS: usize = $offset;
    };

    // Array:
    // NAME: [Type; N] — base index, offset += N
    (@consts $offset:expr; $field:ident : [$ty:ident; $n:expr], $( $rest:tt )*) => {
        #[allow(dead_code)]
        pub const $field: usize = $offset;
        $crate::define_columns!(@consts $offset + $n; $( $rest )*);
    };

    // Scalar:
    // NAME: Type — index, offset += 1
    (@consts $offset:expr; $field:ident : $ty:ident, $( $rest:tt )*) => {
        #[allow(dead_code)]
        pub const $field: usize = $offset;
        $crate::define_columns!(@consts $offset + 1usize; $( $rest )*);
    };

    // Layout push:
    // scalar
    (@push $v:ident, $ty:ident) => {
        $v.push($crate::schema::ColumnType::$ty);
    };

    // Layout push: array
    (@push $v:ident, [$ty:ident; $n:expr]) => {
        $v.extend(::core::iter::repeat($crate::schema::ColumnType::$ty).take($n));
    };
}

/// Cumulative column offsets from chiplet widths.
///
/// `offsets[i]` = starting column index for chiplet i.
pub fn chiplet_offsets(column_counts: &[usize]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(column_counts.len());
    let mut acc = 0;

    for &count in column_counts {
        offsets.push(acc);
        acc += count;
    }

    offsets
}

#[cfg(test)]
mod tests {
    use super::*;

    define_columns! {
        pub TestScalar {
            A: B32,
            B: B32,
            Q: Bit,
        }
    }

    define_columns! {
        pub TestMixed {
            ADDR_B0: B32,
            ADDR_B1: B32,
            ADDR_B2: B32,
            ADDR_B3: B32,
            IS_WRITE: Bit,
            SELECTOR: Bit,
            AUX_INV: B128,
            DIFF_BYTE_IDX: [Bit; 8],
            DIFF_BIT_IDX: [Bit; 8],
            A_BITS: [Bit; 8],
        }
    }

    define_columns! {
        pub TestLargeArrays {
            STATE_BITS: [Bit; 1600],
            RC_BITS: [Bit; 64],
            LANES: [B64; 25],
            S_ROUND: Bit,
            S_IN_OUT: Bit,
        }
    }

    define_columns! {
        pub TestEmpty {}
    }

    #[test]
    fn scalar_indices() {
        assert_eq!(TestScalar::A, 0);
        assert_eq!(TestScalar::B, 1);
        assert_eq!(TestScalar::Q, 2);
        assert_eq!(TestScalar::NUM_COLUMNS, 3);
    }

    #[test]
    fn mixed_scalar_and_array() {
        assert_eq!(TestMixed::ADDR_B0, 0);
        assert_eq!(TestMixed::ADDR_B1, 1);
        assert_eq!(TestMixed::ADDR_B2, 2);
        assert_eq!(TestMixed::ADDR_B3, 3);
        assert_eq!(TestMixed::IS_WRITE, 4);
        assert_eq!(TestMixed::SELECTOR, 5);
        assert_eq!(TestMixed::AUX_INV, 6);
        assert_eq!(TestMixed::DIFF_BYTE_IDX, 7);
        assert_eq!(TestMixed::DIFF_BIT_IDX, 15);
        assert_eq!(TestMixed::A_BITS, 23);
        assert_eq!(TestMixed::NUM_COLUMNS, 31);
    }

    #[test]
    fn large_arrays() {
        assert_eq!(TestLargeArrays::STATE_BITS, 0);
        assert_eq!(TestLargeArrays::RC_BITS, 1600);
        assert_eq!(TestLargeArrays::LANES, 1664);
        assert_eq!(TestLargeArrays::S_ROUND, 1689);
        assert_eq!(TestLargeArrays::S_IN_OUT, 1690);
        assert_eq!(TestLargeArrays::NUM_COLUMNS, 1691);
    }

    #[test]
    fn empty_schema() {
        assert_eq!(TestEmpty::NUM_COLUMNS, 0);
        assert!(TestEmpty::build_layout().is_empty());
    }

    #[test]
    fn scalar_layout() {
        let layout = TestScalar::build_layout();
        assert_eq!(layout.len(), 3);
        assert_eq!(layout[0], ColumnType::B32);
        assert_eq!(layout[1], ColumnType::B32);
        assert_eq!(layout[2], ColumnType::Bit);
    }

    #[test]
    fn mixed_layout() {
        let layout = TestMixed::build_layout();
        assert_eq!(layout.len(), TestMixed::NUM_COLUMNS);
        assert_eq!(layout[0], ColumnType::B32);
        assert_eq!(layout[4], ColumnType::Bit);
        assert_eq!(layout[6], ColumnType::B128);

        for i in 0..8 {
            assert_eq!(layout[TestMixed::DIFF_BYTE_IDX + i], ColumnType::Bit);
        }
    }

    #[test]
    fn large_array_layout() {
        let layout = TestLargeArrays::build_layout();
        assert_eq!(layout.len(), 1691);
        assert_eq!(layout[0], ColumnType::Bit);
        assert_eq!(layout[1599], ColumnType::Bit);
        assert_eq!(layout[1664], ColumnType::B64);
        assert_eq!(layout[1688], ColumnType::B64);
        assert_eq!(layout[1689], ColumnType::Bit);
    }

    #[test]
    fn offsets_basic() {
        let o = chiplet_offsets(&[9, 5, 10, 9, 17, 49]);
        assert_eq!(o, vec![0, 9, 14, 24, 33, 50]);
    }

    #[test]
    fn offsets_empty() {
        assert!(chiplet_offsets(&[]).is_empty());
    }

    #[test]
    fn offsets_single() {
        assert_eq!(chiplet_offsets(&[42]), vec![0]);
    }
}
