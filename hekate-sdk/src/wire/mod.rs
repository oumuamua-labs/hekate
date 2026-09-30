// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::boxed::Box;
use alloc::collections::BTreeSet;
use alloc::string::String;
use hekate_core::errors::Error;

mod field;

pub mod ast;
pub mod boundary;
pub mod bundle;
pub mod chiplet;
pub mod config;
pub mod expander;
pub mod fixed_column;
pub mod permutation;
pub mod proof;
pub mod trace;

const MAX_LABEL_LEN: usize = 256;
const MAX_TOTAL_LEAKED: usize = 64 * 1024;

/// Per-decode leak budget, charged once per distinct
/// label. Never shared across decodes: one client could
/// fill a shared table and fail every later decode.
pub(crate) struct Interner {
    table: BTreeSet<&'static str>,
    leaked: usize,
}

impl Interner {
    pub(crate) fn new() -> Self {
        Self {
            table: BTreeSet::new(),
            leaked: 0,
        }
    }

    /// Leaks `s` to obtain `&'static str`, once per
    /// distinct string: a repeat returns the first copy.
    pub(crate) fn intern(&mut self, s: &str) -> Result<&'static str, Error> {
        if s.len() > MAX_LABEL_LEN {
            return Err(wire_err("label exceeds 256 bytes"));
        }

        if let Some(&label) = self.table.get(s) {
            return Ok(label);
        }

        if self.leaked + s.len() > MAX_TOTAL_LEAKED {
            return Err(wire_err("distinct label bytes exceed 64 KB"));
        }

        let label: &'static str = Box::leak(String::from(s).into_boxed_str());

        self.leaked += s.len();

        self.table.insert(label);

        Ok(label)
    }
}

pub(crate) fn wire_err(message: &'static str) -> Error {
    Error::Protocol {
        protocol: "wire",
        message,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_label_is_leaked_once() {
        let mut interner = Interner::new();

        let first = interner.intern("boolean").unwrap();
        let again = interner.intern(&String::from("boolean")).unwrap();

        assert_eq!(first, "boolean");
        assert!(core::ptr::eq(first, again));
        assert_eq!(interner.leaked, 7);
    }

    #[test]
    fn label_length_is_capped() {
        let mut interner = Interner::new();

        assert_eq!(
            interner.intern(&"b".repeat(MAX_LABEL_LEN)).unwrap().len(),
            MAX_LABEL_LEN
        );
        assert!(interner.intern(&"x".repeat(MAX_LABEL_LEN + 1)).is_err());
        assert_eq!(interner.leaked, MAX_LABEL_LEN);
    }

    #[test]
    fn budget_bounds_distinct_bytes_only() {
        let mut interner = Interner::new();

        for i in 0..MAX_TOTAL_LEAKED / MAX_LABEL_LEN {
            interner.intern(&format!("{i:0>MAX_LABEL_LEN$}")).unwrap();
        }

        assert!(interner.intern("fresh").is_err());
        assert!(interner.intern(&format!("{:0>MAX_LABEL_LEN$}", 0)).is_ok());
        assert!(interner.intern("").is_ok());
        assert_eq!(interner.leaked, MAX_TOTAL_LEAKED);
    }
}
