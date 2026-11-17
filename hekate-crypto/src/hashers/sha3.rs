// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! SHA-3-256 (FIPS-202, RustCrypto `sha3` crate).
use crate::Hasher;
#[cfg(target_arch = "aarch64")]
use crate::hashers::sponge::{Batch, SHA3_PAD};
use sha3::{Digest, Sha3_256};

#[derive(Clone, Debug)]
pub struct Sha3_256Hasher {
    inner: Sha3_256,
}

impl Hasher for Sha3_256Hasher {
    type Digest = [u8; 32];

    fn new() -> Self {
        Self {
            inner: Sha3_256::new(),
        }
    }

    fn update(&mut self, data: &[u8]) {
        Digest::update(&mut self.inner, data);
    }

    fn finalize(self) -> [u8; 32] {
        self.inner.finalize().into()
    }

    fn finalize_reset(&mut self) -> [u8; 32] {
        let out = self.inner.finalize_reset();
        out.into()
    }

    #[cfg(target_arch = "aarch64")]
    fn digest_many(data: &[u8], len: usize, out: &mut [[u8; 32]]) {
        Batch {
            data,
            len,
            out,
            pad: SHA3_PAD,
        }
        .run();
    }
}
