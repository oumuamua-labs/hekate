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

use hekate_core::trace::{ColumnTrace, TraceColumn};

use crate::ffi::HekateColumnView;

const COL_BIT: u8 = 0;
const COL_B8: u8 = 1;
const COL_B16: u8 = 2;
const COL_B32: u8 = 3;
const COL_B64: u8 = 4;
const COL_B128: u8 = 5;

pub fn views_for(trace: &ColumnTrace) -> Vec<HekateColumnView> {
    trace
        .columns
        .iter()
        .map(|col| match col {
            TraceColumn::Bit(v) => HekateColumnView {
                col_type: COL_BIT,
                num_rows: v.len() as u64,
                data: v.as_ptr() as *const u8,
            },
            TraceColumn::B8(v) => HekateColumnView {
                col_type: COL_B8,
                num_rows: v.len() as u64,
                data: v.as_ptr() as *const u8,
            },
            TraceColumn::B16(v) => HekateColumnView {
                col_type: COL_B16,
                num_rows: v.len() as u64,
                data: v.as_ptr() as *const u8,
            },
            TraceColumn::B32(v) => HekateColumnView {
                col_type: COL_B32,
                num_rows: v.len() as u64,
                data: v.as_ptr() as *const u8,
            },
            TraceColumn::B64(v) => HekateColumnView {
                col_type: COL_B64,
                num_rows: v.len() as u64,
                data: v.as_ptr() as *const u8,
            },
            TraceColumn::B128(v) => HekateColumnView {
                col_type: COL_B128,
                num_rows: v.len() as u64,
                data: v.as_ptr() as *const u8,
            },
        })
        .collect()
}
