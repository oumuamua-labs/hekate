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

use crate::ffi;

pub struct CancelToken {
    raw: *mut ffi::HekateCancelToken,
}

// Safety:
// hekate_cancel_request and hekate_cancel_free
// are documented thread-safe by the cdylib.
unsafe impl Send for CancelToken {}
unsafe impl Sync for CancelToken {}

impl CancelToken {
    pub fn new() -> Self {
        // Safety:
        // hekate_cancel_new returns either a valid
        // opaque handle or null on alloc failure.
        let raw = unsafe { ffi::hekate_cancel_new() };
        assert!(!raw.is_null(), "hekate_cancel_new returned null");

        Self { raw }
    }

    pub fn request(&self) {
        // Safety:
        // self.raw non-null since construction;
        // cdylib guarantees thread-safety.
        unsafe { ffi::hekate_cancel_request(self.raw) };
    }

    pub(crate) fn as_ptr(&self) -> *const ffi::HekateCancelToken {
        self.raw as *const _
    }
}

impl Default for CancelToken {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for CancelToken {
    fn drop(&mut self) {
        // Safety:
        // matches hekate_cancel_new;
        // called exactly once on drop.
        unsafe { ffi::hekate_cancel_free(self.raw) };
    }
}
