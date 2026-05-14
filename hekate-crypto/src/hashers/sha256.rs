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

//! SHA2-256 (NIST Standard)
use crate::Hasher;
use sha2::digest::Update;
use sha2::{Digest, Sha256};

#[derive(Clone, Debug)]
pub struct Sha256Hasher {
    inner: Sha256,
}

impl Hasher for Sha256Hasher {
    const OUTPUT_SIZE: usize = 32;

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
