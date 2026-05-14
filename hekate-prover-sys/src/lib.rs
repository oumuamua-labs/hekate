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

mod cancel;
mod error;
mod ffi;
mod witness;

pub use cancel::CancelToken;
pub use error::{Error, ErrorCode};

use core::ffi::CStr;
use core::ptr;

use hekate_core::config::Config;
use hekate_core::proofs::InnerProof;
use hekate_math::Block128;
use hekate_program::{Program, ProgramInstance, ProgramWitness};
use hekate_sdk::{deserialize_proof, serialize_bundle_header};

pub fn version() -> &'static str {
    // Safety:
    // hekate_version returns a
    // NUL-terminated static C string.
    unsafe {
        CStr::from_ptr(ffi::hekate_version())
            .to_str()
            .unwrap_or("unknown")
    }
}

pub fn build_id() -> &'static str {
    // Safety:
    // hekate_build_id returns a NUL-terminated
    // static C string.
    unsafe {
        CStr::from_ptr(ffi::hekate_build_id())
            .to_str()
            .unwrap_or("unknown")
    }
}

pub fn init_tracing() {
    // Safety:
    // no inputs; cdylib mutates only
    // its own tracing-core statics.
    unsafe { ffi::hekate_init_tracing() };
}

pub fn prove<P: Program<Block128>>(
    transcript_label: &[u8],
    program: &P,
    instance: &ProgramInstance<Block128>,
    witness: &ProgramWitness<Block128>,
    config: &Config,
    seed: [u8; 32],
    cancel: Option<&CancelToken>,
) -> Result<InnerProof<Block128>, Error> {
    let bundle = serialize_bundle_header(program, instance, config)
        .map_err(|e| Error::bundle(format!("serialize_bundle_header: {e}")))?;

    let main_views = witness::views_for(&witness.trace);

    let chiplet_view_arrays: Vec<Vec<ffi::HekateColumnView>> = witness
        .chiplet_traces
        .iter()
        .map(witness::views_for)
        .collect();

    let chiplet_witnesses: Vec<ffi::HekateChipletWitness> = chiplet_view_arrays
        .iter()
        .zip(witness.chiplet_traces.iter())
        .enumerate()
        .map(|(i, (views, trace))| {
            if trace.num_vars >= 64 {
                return Err(Error::witness(format!(
                    "chiplet[{i}] num_vars={} >= 64",
                    trace.num_vars
                )));
            }

            Ok(ffi::HekateChipletWitness {
                columns: views.as_ptr(),
                num_columns: views.len(),
                num_rows: 1u64 << trace.num_vars,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let cancel_ptr = cancel.map(CancelToken::as_ptr).unwrap_or(ptr::null());

    let mut out_proof_bytes: *mut u8 = ptr::null_mut();
    let mut out_proof_len: usize = 0;
    let mut out_error: *mut ffi::HekateError = ptr::null_mut();

    // Safety:
    // all input pointers are valid for the
    // call duration; the four out parameters
    // are caller-owned stack slots.
    let rc = unsafe {
        ffi::hekate_prove_b128_default(
            bundle.as_ptr(),
            bundle.len(),
            main_views.as_ptr(),
            main_views.len(),
            chiplet_witnesses.as_ptr(),
            chiplet_witnesses.len(),
            transcript_label.as_ptr(),
            transcript_label.len(),
            seed.as_ptr(),
            cancel_ptr,
            &mut out_proof_bytes,
            &mut out_proof_len,
            &mut out_error,
        )
    };

    if rc != 0 {
        return Err(Error::from_ffi(rc, out_error));
    }

    if out_proof_bytes.is_null() || out_proof_len == 0 {
        return Err(Error::deserialize(format!(
            "ffi rc=0 but proof buffer is empty (ptr={:p}, len={out_proof_len})",
            out_proof_bytes
        )));
    }

    // Safety:
    // cdylib contract:
    //   rc=0 with non-null bytes yields a
    //   valid (ptr, len) buffer released
    //   by hekate_free_proof.
    let proof_bytes = unsafe { core::slice::from_raw_parts(out_proof_bytes, out_proof_len) };
    let proof = deserialize_proof::<Block128>(proof_bytes)
        .map_err(|e| Error::deserialize(format!("deserialize_proof: {e}")));

    // Safety:
    // matches the rc=0 acquisition;
    // deserialize_proof returns owned data.
    unsafe { ffi::hekate_free_proof(out_proof_bytes, out_proof_len) };

    proof
}
