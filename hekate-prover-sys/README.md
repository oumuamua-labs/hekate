# hekate-prover-sys

[![Crates.io](https://img.shields.io/crates/v/hekate-prover-sys.svg)](https://crates.io/crates/hekate-prover-sys)
[![Docs.rs](https://docs.rs/hekate-prover-sys/badge.svg)](https://docs.rs/hekate-prover-sys)
[![CI](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml/badge.svg)](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml)
[![License: AGPL-3.0-only](https://img.shields.io/badge/License-AGPL--3.0--only-blue.svg)](https://github.com/oumuamua-labs/hekate/blob/main/hekate-prover-sys/LICENSE)

*Copyright (c) 2026 Andrei Kochergin and Oumuamua Labs.*

The prover of [Hekate](https://oumuamua.dev/hekate), the Rust zero-knowledge proof engine, as one function: `prove`.
The prover ships as a signed binary. This crate's build script fetches the one for your target, checks its SHA-256,
an Ed25519 signature and an ML-DSA-65 signature, and links it. It is the only crate that calls the prover; a service
that only verifies proofs leaves it out.

## Usage

```bash
cargo add hekate-prover-sys --features ct
cargo add hekate-core hekate-math hekate-program rand
```

```rust
use hekate_core::config::Config;
use hekate_core::proofs::InnerProof;
use hekate_math::Block128;
use hekate_program::{Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use rand::{TryRng, rngs::SysRng};
use std::error::Error;

fn prove_once<P: Program<Block128>>(
    program: &P,
    instance: &ProgramInstance<Block128>,
    witness: &ProgramWitness<Block128>,
    config: &Config,
) -> Result<InnerProof<Block128>, Box<dyn Error>> {
    let mut seed = [0; 32];
    SysRng.try_fill_bytes(&mut seed)?;

    Ok(prove(b"my-program", program, instance, witness, config, seed, None)?)
}
```

The label keeps your program's transcript apart from every other program's, and the verifier must use the same bytes.
Draw the seed fresh for every proof: one seed used for two different witnesses discloses them. To stop a proof early,
pass `Some(&token)` with a `CancelToken` and call `token.request()` from another thread.

## Variants

| Feature  | Arithmetic                   | Use it for                                              |
|:---------|:-----------------------------|:--------------------------------------------------------|
| `ct`     | Constant-time                | Any witness that holds private data                     |
| `public` | Variable-time tables, faster | Public data only: the timing of a run leaks the witness |

The build needs exactly one of them and stops with an error that names both otherwise. The hash features, `blake3` by
default, `sha2` and `sha3`, must match the hash the linked binary was built with and the verifier's, or no proof verifies.

## The binary

The build script looks in `HEKATE_PROVER_DYLIB_DIR` when it is set, and nowhere else. Otherwise it reuses the cache
under `~/.cache/hekate-prover-sys/` or downloads the binary pinned in `artifacts/manifest.toml`. A file that fails the
pinned SHA-256 or the publisher signatures fails the build. `version()` and `build_id()` report which binary your
program linked.

Public binaries exist for ARM64 macOS, Linux with glibc, iOS, the iOS Simulator and Android.
No public x86_64 prover build exists yet.

## Documentation

- [Installation](https://oumuamua.dev/hekate/docs/getting-started/installation#how-the-prover-binary-reaches-your-build):
  variants, targets, and how the binary reaches your build
- [Prover Releases](https://oumuamua.dev/hekate/releases/prover): every prover version with its files and their SHA-256
- [Prover Engine](https://oumuamua.dev/hekate/docs/basics/prover-engine): what the prover does with your tables, its
  errors and cancellation
- [API reference](https://docs.rs/hekate-prover-sys)

## License

AGPL-3.0-only. See [LICENSE](https://github.com/oumuamua-labs/hekate/blob/main/hekate-prover-sys/LICENSE) and
[NOTICE](https://github.com/oumuamua-labs/hekate/blob/main/hekate-prover-sys/NOTICE).
Commercial licenses are available from Oumuamua Labs <info@oumuamua.dev>.

The prover shared library this crate links is not part of this crate and is distributed under its own terms. Linking
it into an AGPL work is covered by
[LICENSE-EXCEPTION](https://github.com/oumuamua-labs/hekate/blob/main/hekate-prover-sys/LICENSE-EXCEPTION), an
additional permission under AGPL section 7.
