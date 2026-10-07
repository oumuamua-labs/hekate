# hekate-sdk

[![Crates.io](https://img.shields.io/crates/v/hekate-sdk.svg)](https://crates.io/crates/hekate-sdk)
[![Docs.rs](https://docs.rs/hekate-sdk/badge.svg)](https://docs.rs/hekate-sdk)
[![CI](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml/badge.svg)](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml)
[![License: AGPL-3.0-only](https://img.shields.io/badge/License-AGPL--3.0--only-blue.svg)](https://github.com/oumuamua-labs/hekate/blob/main/hekate-sdk/LICENSE)

*Copyright (c) 2026 Andrei Kochergin and Oumuamua Labs.*

Preflight diagnostics and the proof wire format of [Hekate](https://oumuamua.dev/hekate), the Rust zero-knowledge
proof engine. Preflight evaluates every constraint, pinned cell and fixed column of a program on the concrete trace,
checks the multisets of every bus, and names what fails before you spend a proof on it.

## Usage

```bash
cargo add hekate-sdk hekate-core hekate-math hekate-program
```

```rust
use hekate_core::errors;
use hekate_math::Block128;
use hekate_program::{Program, ProgramInstance, ProgramWitness};
use hekate_sdk::preflight;

type F = Block128;

fn assert_preflight_clean<P: Program<F>>(
    program: &P,
    instance: &ProgramInstance<F>,
    witness: &ProgramWitness<F>,
) -> errors::Result<()> {
    let report = preflight(program, instance, witness)?;

    assert!(report.is_clean(), "{report}");

    Ok(())
}
```

Call it from a test with your generator's trace. A failing report names each failing check and where it failed: the
table, the constraint label and the row, or the bus and its endpoints. `constraint_violations`, `boundary_violations`,
`fixed_column_violations` and `bus_diagnostics` hold them for assertions. A clean report says the trace meets the
constraints you wrote; a constraint you left out never fails.

| Function                                     | Does                                                                     |
|:---------------------------------------------|:-------------------------------------------------------------------------|
| `preflight`                                  | Checks a witness without proving                                         |
| `serialize_proof_bytes`, `deserialize_proof` | A proof to bytes for the wire, and back                                  |
| `serialize_bundle`, `deserialize_bundle`     | The program, instance, witness and configuration in one buffer, and back |
| `serialize_bundle_header`                    | The same without the witness: the header the prover binary reads         |

## Documentation

- [Preflight](https://oumuamua.dev/hekate/docs/basics/preflight): reading the report, bus failures, tests that break a
  cell on purpose, and what a clean report does not prove
- [Installation: prove](https://oumuamua.dev/hekate/docs/getting-started/installation#prove): a proof and its bytes
- [API reference](https://docs.rs/hekate-sdk)

## License

AGPL-3.0-only. See [LICENSE](https://github.com/oumuamua-labs/hekate/blob/main/hekate-sdk/LICENSE) and
[NOTICE](https://github.com/oumuamua-labs/hekate/blob/main/hekate-sdk/NOTICE).
Commercial licenses are available from Oumuamua Labs <info@oumuamua.dev>.
