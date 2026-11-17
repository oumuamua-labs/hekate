// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

pub fn keccak_f(mut state: [u64; 25]) -> [u64; 25] {
    keccak::Keccak::new().with_f1600(|f| f(&mut state));

    state
}
