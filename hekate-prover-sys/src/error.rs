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

use core::ffi::CStr;

use crate::ffi;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ErrorCode {
    InvalidInput = 1,
    InvalidWitnessAlignment = 2,
    InvalidWitnessLength = 3,
    UnsupportedColumnType = 4,
    Reentrant = 5,
    Cancelled = 6,
    Panicked = 7,
    DeserializeBundle = 8,
    Prove = 9,
    SerializeProof = 10,
    Unknown = 0,
}

impl ErrorCode {
    fn from_raw(code: i32) -> Self {
        match code {
            1 => Self::InvalidInput,
            2 => Self::InvalidWitnessAlignment,
            3 => Self::InvalidWitnessLength,
            4 => Self::UnsupportedColumnType,
            5 => Self::Reentrant,
            6 => Self::Cancelled,
            7 => Self::Panicked,
            8 => Self::DeserializeBundle,
            9 => Self::Prove,
            10 => Self::SerializeProof,
            _ => Self::Unknown,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("hekate prover {code:?}: {message}")]
    Ffi { code: ErrorCode, message: String },

    #[error("bundle build: {0}")]
    Bundle(String),

    #[error("witness conversion: {0}")]
    Witness(String),

    #[error("proof deserialization: {0}")]
    Deserialize(String),
}

impl Error {
    pub(crate) fn bundle(msg: impl Into<String>) -> Self {
        Self::Bundle(msg.into())
    }

    pub(crate) fn witness(msg: impl Into<String>) -> Self {
        Self::Witness(msg.into())
    }

    pub(crate) fn deserialize(msg: impl Into<String>) -> Self {
        Self::Deserialize(msg.into())
    }

    pub(crate) fn from_ffi(rc: i32, err_ptr: *mut ffi::HekateError) -> Self {
        let code = ErrorCode::from_raw(rc);

        let body = if err_ptr.is_null() {
            None
        } else {
            // Safety:
            // err_ptr non-null per branch;
            // cdylib owns the underlying error object.
            let msg_ptr = unsafe { ffi::hekate_error_message(err_ptr) };

            let owned = if msg_ptr.is_null() {
                None
            } else {
                // Safety:
                // msg_ptr is a NUL-terminated
                // C string owned by err_ptr.
                Some(
                    unsafe { CStr::from_ptr(msg_ptr) }
                        .to_string_lossy()
                        .into_owned(),
                )
            };

            // Safety:
            // matches hekate_error_message;
            // releases the cdylib-owned error.
            unsafe { ffi::hekate_free_error(err_ptr) };

            owned
        };

        let message = match body {
            Some(b) if !b.is_empty() => format!("rc={rc}: {b}"),
            Some(_) => format!("rc={rc} (empty error message)"),
            None => format!("rc={rc} (no error object)"),
        };

        Self::Ffi { code, message }
    }

    pub fn code(&self) -> Option<ErrorCode> {
        match self {
            Self::Ffi { code, .. } => Some(*code),
            _ => None,
        }
    }
}
