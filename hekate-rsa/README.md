# hekate-rsa

[![Crates.io](https://img.shields.io/crates/v/hekate-rsa.svg)](https://crates.io/crates/hekate-rsa)
[![Docs.rs](https://docs.rs/hekate-rsa/badge.svg)](https://docs.rs/hekate-rsa)
[![CI](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml/badge.svg)](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml)
[![License: AGPL-3.0-only](https://img.shields.io/badge/License-AGPL--3.0--only-blue.svg)](https://github.com/oumuamua-labs/hekate/blob/main/hekate-rsa/LICENSE)

*Copyright (c) 2026 Andrei Kochergin and Oumuamua Labs.*

RSA-2048 PKCS#1 v1.5 signature verification in zero knowledge for [Hekate](https://oumuamua.dev/hekate), the Rust
zero-knowledge proof engine. `Pkcs1Statement` proves `s^65537 mod N == PKCS1-v1_5(H)` for a 2048-bit modulus, `H`
chained over committed block words with FIPS 180-4 §5.1.1 padding left to the caller. `N` is public, the signature
and the message are witness.

The host table chains SHA-256 one block per row and requests two chiplets over the LogUp bus: modexp from
`hekate-gadgets` and the compression function from `hekate-sha2`.

## ⚠️ Security Warning

This crate has not been independently audited and may contain bugs and security flaws.

## Usage

```bash
cargo add hekate-rsa hekate-sha2 hekate-core hekate-math hekate-program
```

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

`modulus` and `signature` are 64 little-endian 32-bit limbs, and `instance` publishes the modulus. The prover takes
`statement.program()`, `instance` and `witness`; the verifier takes the program and `instance`.

[`rsa_pkcs1.rs`](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/rsa_pkcs1.rs) runs RSA-2048 with
SHA-256 end to end.

## Documentation

- [RSA-2048 PKCS#1 v1.5 Verification](https://oumuamua.dev/primitives/signatures/rsa): what the proof states, how the
  tables split the work, and proving time, proof size and memory
- [SHA-256](https://oumuamua.dev/primitives/hashing/sha2): the chiplet that hashes the message
- [API reference](https://docs.rs/hekate-rsa)

## License

AGPL-3.0-only. See [LICENSE](https://github.com/oumuamua-labs/hekate/blob/main/hekate-rsa/LICENSE) and
[NOTICE](https://github.com/oumuamua-labs/hekate/blob/main/hekate-rsa/NOTICE).
Commercial licenses are available from Oumuamua Labs <info@oumuamua.dev>.
