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

//! FIPS 202 known-answer vectors
//! + external-crate permutation parity.
//!
//! Two independent anchors to the spec:
//! - Hardcoded NIST vectors pin `KeccakSpongeNative` end-to-end
//!   (padding, domain-sep, absorb chain, squeeze).
//! - The audited `keccak` crate's `with_f1600` pins the in-house
//!   `KeccakWitness::keccak_f_round` loop on canonical states.

use hekate_keccak::{KeccakChiplet, KeccakWitness, sha3_256, sha3_512, shake256};

fn from_hex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn fips202_sha3_256_empty() {
    let (out, _) = sha3_256(b"");
    assert_eq!(
        out.as_slice(),
        from_hex("a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a").as_slice(),
    );
}

#[test]
fn fips202_sha3_256_abc() {
    let (out, _) = sha3_256(b"abc");
    assert_eq!(
        out.as_slice(),
        from_hex("3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532").as_slice(),
    );
}

#[test]
fn fips202_sha3_512_empty() {
    let (out, _) = sha3_512(b"");
    assert_eq!(
        out.as_slice(),
        from_hex(
            "a69f73cca23a9ac5c8b567dc185a756e97c982164fe25859e0d1dcc1475c80a6\
             15b2123af1f5f94c11e3e9402c3ac558f500199d95b6d3e301758586281dcd26",
        )
        .as_slice(),
    );
}

#[test]
fn fips202_sha3_512_abc() {
    let (out, _) = sha3_512(b"abc");
    assert_eq!(
        out.as_slice(),
        from_hex(
            "b751850b1a57168a5693cd924b6b096e08f621827444f70d884f5d0240d2712e\
             10e116e9192af3c91a7ec57647e3934057340b4cf408d5a56592f8274eec53f0",
        )
        .as_slice(),
    );
}

#[test]
fn fips202_shake256_empty_32() {
    let (out, _) = shake256(b"", 32);
    assert_eq!(
        out.as_slice(),
        from_hex("46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762f").as_slice(),
    );
}

#[test]
fn keccak_f_matches_external_on_canonical_states() {
    let alternating: [u64; 25] = core::array::from_fn(|i| {
        if i % 2 == 0 {
            0x5555_5555_5555_5555
        } else {
            0xaaaa_aaaa_aaaa_aaaa
        }
    });

    let cases: [[u64; 25]; 3] = [[0u64; 25], [u64::MAX; 25], alternating];

    for state in cases {
        let mut ours = state;
        for &rc in &KeccakChiplet::ROUND_CONSTANTS {
            ours = KeccakWitness::keccak_f_round(ours, rc);
        }

        let mut theirs = state;
        keccak::Keccak::new().with_f1600(|f| f(&mut theirs));

        assert_eq!(ours, theirs);
    }
}
