// SPDX-License-Identifier: Apache-2.0
// This file is part of the hekate-math project.
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

use hekate_core::errors::Result;
use hekate_math::TowerField;

use super::wire_err;

pub(crate) fn field_to_lo_hi<F: TowerField>(f: &F) -> (u64, u64) {
    let bytes = f.to_bytes();
    let len = bytes.len().min(16);

    let mut buf = [0u8; 16];
    buf[..len].copy_from_slice(&bytes[..len]);

    (
        u64::from_le_bytes(buf[0..8].try_into().unwrap()),
        u64::from_le_bytes(buf[8..16].try_into().unwrap()),
    )
}

pub(crate) fn bytes_to_lo_hi(bytes: &[u8]) -> (u64, u64) {
    let len = bytes.len().min(16);

    let mut buf = [0u8; 16];
    buf[..len].copy_from_slice(&bytes[..len]);

    (
        u64::from_le_bytes(buf[0..8].try_into().unwrap()),
        u64::from_le_bytes(buf[8..16].try_into().unwrap()),
    )
}

pub(crate) fn lo_hi_to_field<F: TowerField>(lo: u64, hi: u64) -> Result<F> {
    let mut buf = [0u8; 16];
    buf[0..8].copy_from_slice(&lo.to_le_bytes());
    buf[8..16].copy_from_slice(&hi.to_le_bytes());

    F::deserialize(&buf).map_err(|_| wire_err("invalid field element bytes"))
}
