# hekate-crypto

[![Crates.io](https://img.shields.io/crates/v/hekate-crypto.svg)](https://crates.io/crates/hekate-crypto)
[![Docs.rs](https://docs.rs/hekate-crypto/badge.svg)](https://docs.rs/hekate-crypto)
[![CI](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml/badge.svg)](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml)
[![License: AGPL-3.0-only](https://img.shields.io/badge/License-AGPL--3.0--only-blue.svg)](https://github.com/oumuamua-labs/hekate/blob/main/hekate-crypto/LICENSE)

*Copyright (c) 2026 Andrei Kochergin and Oumuamua Labs.*

Hash functions, Merkle trees and the Fiat-Shamir transcript of [Hekate](https://oumuamua.dev/hekate), the Rust
zero-knowledge proof engine. A proof commits to its tables with Merkle trees and draws every challenge from the
transcript, which makes the hash part of the proof: the prover and the verifier must use the same one.

## ⚠️ Security Warning

This crate has not been independently audited and may contain bugs and security flaws.

USE AT YOUR OWN RISK!

## Usage

```bash
cargo add hekate-crypto hekate-math
```

`DefaultHasher` hashes in parts, or in one call with `DefaultHasher::digest`:

```rust
use hekate_crypto::{DefaultHasher, Hasher};

fn row_digest(parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = DefaultHasher::new();
    for part in parts {
        hasher.update(part);
    }

    hasher.finalize()
}
```

A Merkle tree commits to leaf digests, and a path opens one leaf against the root:

```rust
use hekate_crypto::merkle::{self, MerkleTree};
use hekate_crypto::{DefaultHasher, Hasher};
use hekate_math::Block128;

fn commit_and_open(rows: &[&[u8]], index: usize) -> merkle::Result<([u8; 32], Vec<[u8; 32]>)> {
    let leaves: Vec<[u8; 32]> = rows.iter().map(|row| DefaultHasher::digest(row)).collect();
    let tree = MerkleTree::<Block128>::new(&leaves);

    Ok((tree.root(), tree.prove_batch(&[index])?))
}

fn opens(root: &[u8; 32], num_rows: usize, row: &[u8], index: usize, path: &[[u8; 32]]) -> bool {
    let leaf = (index, DefaultHasher::digest(row));

    MerkleTree::<Block128>::verify_batch(root, num_rows.next_power_of_two(), &[leaf], path)
}
```

A leaf count that is not a power of two pads with zero leaves. The padded count fixes the path length, and several
leaves open with one shared set of siblings.

A transcript turns what both sides have absorbed into the same challenge, here drawn after the root:

```rust
use hekate_crypto::DefaultHasher;
use hekate_crypto::transcript::{self, Transcript};
use hekate_math::Block128;

fn challenge(root: &[u8; 32]) -> transcript::Result<Block128> {
    let mut transcript = Transcript::<DefaultHasher>::new(b"my-app");

    transcript.append_message(b"root", root);
    transcript.challenge_field(b"alpha")
}
```

The label keeps your application's transcripts apart from every other one, and the verifier opens its transcript with
the same bytes.

## Choosing the hash

| Feature  | Default | `DefaultHasher` |
|:---------|:--------|:----------------|
| `blake3` | yes     | BLAKE3          |
| `sha2`   | no      | SHA-256         |
| `sha3`   | no      | SHA3-256        |

Cargo enables a feature for the whole build as soon as one crate asks for it, and SHA3-256 then wins over SHA-256,
which wins over BLAKE3. The prover binary and the verifier must use the same hash, or no proof verifies.

`parallel` (default) builds Merkle trees on a Rayon pool, and `transcript-trace` records every transcript operation
to locate a prover/verifier divergence.

## Documentation

- [Prover Engine: commit](https://oumuamua.dev/hekate/docs/basics/prover-engine#step-1-commit): how a proof commits
  to its tables
- [Prover Engine: challenges](https://oumuamua.dev/hekate/docs/basics/prover-engine#step-2-challenges): the
  Fiat-Shamir transcript, and how the verifier replays it
- [API reference](https://docs.rs/hekate-crypto)

## License

AGPL-3.0-only. See [LICENSE](https://github.com/oumuamua-labs/hekate/blob/main/hekate-crypto/LICENSE) and
[NOTICE](https://github.com/oumuamua-labs/hekate/blob/main/hekate-crypto/NOTICE).
Commercial licenses are available from Oumuamua Labs <info@oumuamua.dev>.
