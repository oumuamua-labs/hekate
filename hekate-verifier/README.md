# hekate-verifier

[![Crates.io](https://img.shields.io/crates/v/hekate-verifier.svg)](https://crates.io/crates/hekate-verifier)
[![Docs.rs](https://docs.rs/hekate-verifier/badge.svg)](https://docs.rs/hekate-verifier)
[![CI](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml/badge.svg)](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml)
[![License: AGPL-3.0-only](https://img.shields.io/badge/License-AGPL--3.0--only-blue.svg)](https://github.com/oumuamua-labs/hekate/blob/main/hekate-verifier/LICENSE)

*Copyright (c) 2026 Andrei Kochergin and Oumuamua Labs.*

Verifier of [Hekate](https://oumuamua.dev/hekate), the Rust zero-knowledge proof engine, in pure Rust. It checks a
proof that a program's constraints, pinned cells and bus relations hold over a committed trace and its chiplet tables,
given the program, its public inputs and the proof bytes. Proofs are zero-knowledge by default and can be made in the
clear; this crate verifies both. It runs no prover code and takes nothing from the prover on trust.

## ⚠️ Security Warning

This crate has not been independently audited and may contain bugs and security flaws.

USE AT YOUR OWN RISK!

## What you pin

Only proof bytes cross from the prover's side. Your build fixes four inputs before any proof arrives:

- **The program**, built from the same audited source the prover used.
- **The program id**, printed once with `hekate_program::digest::program_id_hex` and compiled in as a constant. The
  verifier recomputes it from the program and returns `Err(ProgramIdMismatch)` on a difference.
- **The `Config` and the transcript label**, equal to the prover's. `Config::prod()` is the default; `Config::dev()`
  accepts weak geometries and exists for tests.
- **The instance**: the trace height and the public inputs the statement is about.

`Ok(true)` accepts. `Ok(false)` means the main table's ZeroCheck, an evaluation check or the outer argument failed.
`Err` means any other check failed, the proof or input is malformed, the program id differs, or the configuration is
below its security floor. A forged proof can produce either: treat everything except `Ok(true)` as a rejection.

## Usage

```bash
cargo add hekate-verifier hekate-core hekate-crypto hekate-math hekate-program hekate-sdk
```

```rust
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_crypto::DefaultHasher;
use hekate_crypto::transcript::Transcript;
use hekate_math::Block128;
use hekate_program::{Program, ProgramInstance};
use hekate_sdk::deserialize_proof;
use hekate_verifier::HekateVerifier;

type F = Block128;

/// `audited_id` is a constant of this build, printed
/// once with `digest::program_id_hex` at audit time.
fn verify_one<P: Program<F> + Sync>(
    audited_id: &[u8; 32],
    program: &P,
    instance: &ProgramInstance<F>,
    proof_bytes: &[u8],
) -> errors::Result<bool> {
    let proof = deserialize_proof::<F>(proof_bytes)?;

    HekateVerifier::<F, DefaultHasher>::verify(
        audited_id,
        program,
        instance,
        &proof,
        &mut Transcript::new(b"my-app"),
        &Config::prod(),
    )
}
```

| Entry point       | For                                  | Each call                                    |
|:------------------|:-------------------------------------|:---------------------------------------------|
| `verify`          | one proof                            | prepares the program, recomputing its id     |
| `verify_prepared` | a long-lived verifier of one program | replays the proof against a prepared program |
| `verify_batch`    | many proofs of one program at once   | one proof per pool thread, results in order  |

`PreparedProgram` holds what verifying one program needs before any proof and serves every thread read-only.
`VerifierScratch` caches state between proofs: one per thread.

## Features

| Feature            | Default | Effect                                                                             |
|:-------------------|:--------|:-----------------------------------------------------------------------------------|
| `std`              | yes     | Standard library. Without it the crate is `no_std` with `alloc`.                   |
| `blake3`           | yes     | BLAKE3 transcript and Merkle hashing.                                              |
| `sha2`, `sha3`     | no      | SHA-256 or SHA3-256 hashing. The prover must use the same hash.                    |
| `parallel`         | yes     | Rayon pool for leaf hashing, proximity checks, the outer tests and `verify_batch`. |
| `table-math`       | no      | Variable-time basis conversion, faster.                                            |
| `transcript-trace` | no      | Records every transcript operation, to locate a prover/verifier divergence.        |

Every value the verifier touches is public, which makes `table-math` safe for verification. Cargo applies the feature
to every crate in the build that uses `hekate-math`, however: leave it off in a binary that also generates witnesses.

## Documentation

- [Verifier Logic](https://oumuamua.dev/hekate/docs/basics/verifier-logic): the checks in order, reading the result,
  and the prepared and batch entry points with code
- [Zero Knowledge](https://oumuamua.dev/hekate/docs/advanced/zero-knowledge): what the masked replay and the outer
  argument hide
- [System Architecture: open crates, closed prover](https://oumuamua.dev/hekate/docs/basics/system-architecture#open-crates-closed-prover):
  the trust boundary between your build and the prover
- [API reference](https://docs.rs/hekate-verifier)

## License

AGPL-3.0-only. See [LICENSE](https://github.com/oumuamua-labs/hekate/blob/main/hekate-verifier/LICENSE) and
[NOTICE](https://github.com/oumuamua-labs/hekate/blob/main/hekate-verifier/NOTICE).
Commercial licenses are available from Oumuamua Labs <info@oumuamua.dev>.
