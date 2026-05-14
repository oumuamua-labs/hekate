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

use core::ffi::c_char;

#[repr(C)]
pub struct HekateError {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct HekateCancelToken {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct HekateColumnView {
    pub col_type: u8,
    pub num_rows: u64,
    pub data: *const u8,
}

#[repr(C)]
pub struct HekateChipletWitness {
    pub columns: *const HekateColumnView,
    pub num_columns: usize,
    pub num_rows: u64,
}

unsafe extern "C" {
    #[allow(clippy::too_many_arguments)]
    pub fn hekate_prove_b128_default(
        bundle_bytes: *const u8,
        bundle_len: usize,
        main_witness: *const HekateColumnView,
        main_witness_len: usize,
        chiplet_witnesses: *const HekateChipletWitness,
        num_chiplet_witnesses: usize,
        transcript_label: *const u8,
        transcript_label_len: usize,
        seed: *const u8,
        cancel: *const HekateCancelToken,
        out_proof_bytes: *mut *mut u8,
        out_proof_len: *mut usize,
        out_error: *mut *mut HekateError,
    ) -> i32;

    pub fn hekate_free_proof(proof_bytes: *mut u8, proof_len: usize);

    pub fn hekate_cancel_new() -> *mut HekateCancelToken;

    pub fn hekate_cancel_request(token: *mut HekateCancelToken);

    pub fn hekate_cancel_free(token: *mut HekateCancelToken);

    pub fn hekate_error_message(err: *const HekateError) -> *const c_char;

    pub fn hekate_free_error(err: *mut HekateError);

    pub fn hekate_version() -> *const c_char;

    pub fn hekate_build_id() -> *const c_char;

    pub fn hekate_init_tracing();
}
