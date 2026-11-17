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

fn row_digest(parts: &[&[u8]]) -> <DefaultHasher as Hasher>::Digest {
    // The hash the features pick, BLAKE3 by default
    let mut hasher = DefaultHasher::new();
    for part in parts {
        hasher.update(part);
    }

    hasher.finalize()
}
```

A Merkle tree commits to leaf digests, and a path opens one leaf against the root:

```rust
use hekate_crypto::merkle::{self, MerkleTree, Multiproof};
use hekate_crypto::{DefaultHasher, Hasher};
use hekate_math::Block128;

type Digest = <DefaultHasher as Hasher>::Digest;

fn commit_and_open(rows: &[&[u8]], index: usize) -> merkle::Result<(Digest, Multiproof)> {
    // One leaf per row, the digest of its bytes
    let leaves: Vec<Digest> = rows.iter().map(|row| DefaultHasher::digest(row)).collect();
    let tree = MerkleTree::<Block128>::new(&leaves);

    // The root, and the opening of leaf index for one query draw
    Ok((tree.root(), tree.prove_batch(&[index], 1)?))
}

fn opens(root: &Digest, num_rows: usize, row: &[u8], index: usize, proof: &Multiproof) -> bool {
    let leaf = (index, DefaultHasher::digest(row));

    // The same padded leaf count and draw count as the prover
    MerkleTree::<Block128>::verify_batch(root, num_rows.next_power_of_two(), 1, &[leaf], proof)
}
```

A leaf count that is not a power of two pads with zero leaves, and several leaves open under one multiproof.

A transcript turns what both sides have absorbed into the same challenge, here drawn after the root:

```rust
use hekate_crypto::DefaultHasher;
use hekate_crypto::transcript::{self, Transcript};
use hekate_math::Block256;

fn challenge(root: &[u8]) -> transcript::Result<Block256> {
    // The label keeps this application's transcripts apart; the verifier uses it too
    let mut transcript = Transcript::<DefaultHasher>::new(b"my-app");

    // Absorb the commitment before any challenge
    transcript.append_message(b"root", root);

    // The verifier draws the same value from the same bytes
    transcript.challenge_field(b"alpha")
}
```

## Choosing the hash

| Feature    | Default | `DefaultHasher`      |
|:-----------|:--------|:---------------------|
| `blake3`   | yes     | BLAKE3               |
| `sha2`     | no      | SHA-256              |
| `sha3`     | no      | SHA3-256             |
| `shake256` | no      | SHAKE256 at 512 bits |

Cargo enables a feature for the whole build as soon as one crate asks for it, and the enabled hash lowest in the table
then wins. The prover binary and the verifier must use the same hash, or no proof verifies.

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
