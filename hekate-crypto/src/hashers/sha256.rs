// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! SHA2-256 (NIST Standard)
use crate::Hasher;
use sha2::digest::Update;
use sha2::{Digest, Sha256};

#[derive(Clone, Debug)]
pub struct Sha256Hasher {
    inner: Sha256,
}

impl Hasher for Sha256Hasher {
    type Digest = [u8; 32];

    fn new() -> Self {
        Self {
            inner: Sha256::new(),
        }
    }

    fn update(&mut self, data: &[u8]) {
        Update::update(&mut self.inner, data);
    }

    fn finalize(self) -> [u8; 32] {
        self.inner.finalize().into()
    }

    fn finalize_reset(&mut self) -> [u8; 32] {
        let out = self.inner.finalize_reset();
        out.into()
    }
}
