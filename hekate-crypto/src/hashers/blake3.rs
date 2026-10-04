// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! BLAKE3 (Maximum Performance)
use crate::Hasher;
#[cfg(feature = "blake3")]
use blake3;
use core::fmt::Debug;

#[cfg(feature = "blake3")]
#[derive(Clone, Debug)]
pub struct Blake3Hasher {
    inner: blake3::Hasher,
}

#[cfg(feature = "blake3")]
impl Hasher for Blake3Hasher {
    const OUTPUT_SIZE: usize = 32;

    fn new() -> Self {
        Self {
            inner: blake3::Hasher::new(),
        }
    }

    fn update(&mut self, data: &[u8]) {
        self.inner.update(data);
    }

    fn finalize(self) -> [u8; 32] {
        self.inner.finalize().into()
    }

    fn finalize_reset(&mut self) -> [u8; 32] {
        let res = self.inner.finalize();
        self.inner.reset();

        res.into()
    }

    fn digest(data: &[u8]) -> [u8; 32] {
        blake3::hash(data).into()
    }
}
