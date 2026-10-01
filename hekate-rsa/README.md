# hekate-rsa

[![Crates.io](https://img.shields.io/crates/v/hekate-rsa.svg)](https://crates.io/crates/hekate-rsa)
[![Docs.rs](https://docs.rs/hekate-rsa/badge.svg)](https://docs.rs/hekate-rsa)
[![CI](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml/badge.svg)](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml)
[![License: AGPL-3.0-only](https://img.shields.io/badge/License-AGPL--3.0--only-blue.svg)](LICENSE)

*Copyright (c) 2026 Andrei Kochergin and Oumuamua Labs.*

RSA PKCS#1 v1.5 signature statements for the [Hekate ZK](https://github.com/oumuamua-labs/hekate) proving system.

`Pkcs1Statement` proves `s^65537 mod N == PKCS1-v1_5(H)` for a 2048-bit modulus, `H` chained over committed block
words with FIPS 180-4 §5.1.1 padding left to the caller. `N` is public, the signature and the message are witness.

The host table chains SHA-256 one block per row and requests two chiplets over the LogUp bus: modexp from
`hekate-gadgets`, compression from `hekate-sha2`. On the last block row it compares the modexp output against the
RFC 8017 §9.2 encoding: limbs `0..8` against the digest words, limbs `8..64` against the encoding constants. All 56
constant limbs are pinned.

```
Apple M3 Max, ZK, best of three runs (hekate/examples/rsa_pkcs1.rs):
  200-byte message, 4 blocks at 2 rounds/row, host 2^9, SHA chiplet 2^9, modexp chiplet 2^10:
  355 ms prove, 31.2 ms verify, 14,585 KiB proof, 503 MiB
```

## Examples

- [RSA-2048 PKCS#1 v1.5 + SHA-256 end to end](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/rsa_pkcs1.rs)

### Usage

A PKCS#1 v1.5 statement for one RSA-2048 signature over `message`:

```rust
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::trace::ColumnTrace;
use hekate_math::Block128;
use hekate_program::{ProgramInstance, ProgramWitness};
use hekate_rsa::Pkcs1Statement;
use hekate_sha2::{ROUNDS, pad_message};

type F = Block128;

fn pkcs1(
    message: &[u8],
    modulus: &[u32; 64],
    signature: &[u32; 64],
) -> errors::Result<(
    Pkcs1Statement,
    ProgramInstance<F>,
    ProgramWitness<F, ColumnTrace>,
)> {
    let blocks = pad_message(message).len();
    let rounds_per_row = 2;
    let floor = Config::default().min_table_rows();

    let statement = Pkcs1Statement::new(
        blocks,
        rounds_per_row,
        (blocks * ROUNDS / rounds_per_row)
            .next_power_of_two()
            .max(floor),
        blocks.next_power_of_two().max(floor),
    )?;

    let instance = statement.instance(modulus);
    let witness = statement.witness(message, modulus, signature)?;

    Ok((statement, instance, witness))
}
```

`modulus` and `signature` are 64 little-endian 32-bit limbs, and `instance` publishes the modulus. The prover
takes `statement.program()`, `instance` and `witness`; the verifier takes the program and `instance`.

---

## ⚠️ Security Warning

This crate has not been audited and may contain bugs and security flaws.