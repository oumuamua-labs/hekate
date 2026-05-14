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

use alloc::boxed::Box;
use alloc::string::String;
use core::sync::atomic::{AtomicUsize, Ordering};
use hekate_core::errors::Error;

mod field;

pub mod ast;
pub mod boundary;
pub mod bundle;
pub mod chiplet;
pub mod config;
pub mod expander;
pub mod lagrange;
pub mod permutation;
pub mod proof;
pub mod trace;

const MAX_LABEL_LEN: usize = 256;
const MAX_TOTAL_LEAKED: usize = 64 * 1024;

static LEAKED_BYTES: AtomicUsize = AtomicUsize::new(0);

pub(crate) fn wire_err(message: &'static str) -> Error {
    Error::Protocol {
        protocol: "wire",
        message,
    }
}

/// Leak a string to obtain `&'static str`.
pub(crate) fn leak_str(s: &str) -> Result<&'static str, Error> {
    if s.len() > MAX_LABEL_LEN {
        return Err(wire_err("label exceeds 256 bytes"));
    }

    let prev = LEAKED_BYTES.fetch_add(s.len(), Ordering::Relaxed);
    if prev + s.len() > MAX_TOTAL_LEAKED {
        LEAKED_BYTES.fetch_sub(s.len(), Ordering::Relaxed);
        return Err(wire_err("total leaked label bytes exceeds 64 KB"));
    }

    Ok(Box::leak(String::from(s).into_boxed_str()))
}

/// Resets the per-decode leak budget.
pub(crate) fn reset_leak_budget() {
    LEAKED_BYTES.store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn leaked_bytes() -> usize {
        LEAKED_BYTES.load(Ordering::Relaxed)
    }

    #[test]
    fn leak_str_basic() {
        let before = leaked_bytes();
        let s = leak_str("hello").unwrap();

        assert_eq!(s, "hello");
        assert_eq!(leaked_bytes(), before + 5);
    }

    #[test]
    fn leak_str_rejects_oversized_label() {
        let big = "x".repeat(MAX_LABEL_LEN + 1);
        let before = leaked_bytes();

        assert!(leak_str(&big).is_err());
        assert_eq!(leaked_bytes(), before);
    }

    #[test]
    fn leak_str_rejects_when_global_cap_exceeded() {
        let before = leaked_bytes();
        let remaining = MAX_TOTAL_LEAKED.saturating_sub(before);

        if remaining < MAX_LABEL_LEN + 1 {
            return;
        }

        let label = "a".repeat(MAX_LABEL_LEN);
        let fills_needed = remaining / MAX_LABEL_LEN;

        for _ in 0..fills_needed {
            let _ = leak_str(&label);
        }

        let result = leak_str(&label);
        assert!(result.is_err());
    }

    #[test]
    fn leak_str_empty_succeeds() {
        let s = leak_str("").unwrap();
        assert_eq!(s, "");
    }

    #[test]
    fn leak_str_exactly_max_label() {
        let label = "b".repeat(MAX_LABEL_LEN);
        let result = leak_str(&label);

        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), MAX_LABEL_LEN);
    }
}
