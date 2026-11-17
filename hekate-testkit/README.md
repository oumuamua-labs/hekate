# hekate-testkit

[![Crates.io](https://img.shields.io/crates/v/hekate-testkit.svg)](https://crates.io/crates/hekate-testkit)
[![Docs.rs](https://docs.rs/hekate-testkit/badge.svg)](https://docs.rs/hekate-testkit)
[![CI](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml/badge.svg)](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml)
[![License: AGPL-3.0-only](https://img.shields.io/badge/License-AGPL--3.0--only-blue.svg)](https://github.com/oumuamua-labs/hekate/blob/main/hekate-testkit/LICENSE)

*Copyright (c) 2026 Andrei Kochergin and Oumuamua Labs.*

Test oracles for [Hekate](https://oumuamua.dev/hekate) programs, the Rust zero-knowledge proof engine. An honest
statement must pass preflight and verify. A forged trace must show preflight exactly the violations you name, and
the verifier must reject its proof with the `Rejection` they predict. Every check runs with and without zero
knowledge.

## Usage

```bash
cargo add --dev hekate-testkit --features public
cargo add --dev hekate-core hekate-math hekate-program hekate-sdk
```

A table of two 32-bit columns that must be equal on every row, its trace, and a test that forges one cell:

```rust
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::trace::{ColumnTrace, TraceBuilder};
use hekate_math::{Block32, Block256};
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::{ProgramInstance, ProgramWitness, define_columns};
use hekate_sdk::preflight::TableId;
use hekate_testkit::cell::xor_b32;
use hekate_testkit::fault::Fault;
use hekate_testkit::verdict::{assert_rejected, assert_verifies};

type F = Block256;

define_columns! {
    Pair {
        A: B32,
        B: B32,
    }
}

fn program(rows: usize) -> errors::Result<CircuitProgram<F>> {
    let mut cx = Circuit::<F>::new("Pair", rows)?;

    let cols = cx.schema(&Pair::build_layout());
    let (a, b) = (cols.at(Pair::A), cols.at(Pair::B));

    let cs = cx.cs();

    // A + B = 0, and + is XOR: A equals B on every row
    cs.constrain_named("equal", cs.col(a.index()) + cs.col(b.index()));

    cx.compile()
}

fn trace(rows: usize) -> errors::Result<ColumnTrace> {
    let mut tb = TraceBuilder::new(&Pair::build_layout(), rows.trailing_zeros() as usize)?;

    for row in 0..rows {
        let value = Block32::from(row as u32);

        tb.set_b32(Pair::A, row, value)?;
        tb.set_b32(Pair::B, row, value)?;
    }

    Ok(tb.build())
}

#[test]
fn flipped_cell_fails_equal() -> errors::Result<()> {
    let rows = 8;
    let program = program(rows)?;

    // The table height, and no public inputs
    let instance = ProgramInstance::new(rows, vec![]);

    // Few queries and no security floor: fast, for tests only
    let config = Config {
        num_queries: 4,
        fold_arity_log: 1,
        early_stop_vars: 1,
        min_security_bits: 0,
        ..Config::default()
    };

    // Preflight is clean, and the proof verifies with and without zero knowledge
    let honest = ProgramWitness::new(trace(rows)?);
    assert_verifies(&program, (&instance, &honest), &config);

    // Flip bit 0 of B on row 5
    let mut tampered = trace(rows)?;
    xor_b32(&mut tampered, Pair::B, 5, 1);

    let forged = ProgramWitness::new(tampered);

    // Preflight must find exactly equal failing on row 5 of the main table
    let fault = Fault {
        constraint_rows: &[(TableId::Main, 5, Some("equal"))],
        ..Fault::NONE
    };

    // Honest verifies and forged fails as the fault predicts, in both modes
    assert_rejected(
        &program,
        (&instance, &honest),
        (&instance, &forged),
        fault,
        &config,
    );

    Ok(())
}
```

Under zero knowledge the forged proof fails the outer linear test, and without it the main table's zerocheck. A
`Fault` also names boundaries, pinned cells and buses, and `..Fault::NONE` leaves the rest empty.

## The prover mode and the hash

testkit proves through `hekate-prover-sys`, which needs exactly one mode and one hash in the whole build. testkit
sets no mode by default. When your crate already depends on `hekate-prover-sys`, its `ct` or `public` applies, and
testkit goes in without `--features public`. Its hash features, `blake3` by default, `sha2`, `sha3` and `shake256`,
set the hash of the prover, the verifier and the transcript alike. A build on any hash but BLAKE3 adds testkit with
`default-features = false` and that hash's feature.

## Documentation

- [Preflight: keep it in your tests](https://oumuamua.dev/hekate/docs/basics/preflight#keep-it-in-your-tests): tests
  that break one cell on purpose and assert the check that catches it
- [API reference](https://docs.rs/hekate-testkit)

## License

AGPL-3.0-only. See [LICENSE](https://github.com/oumuamua-labs/hekate/blob/main/hekate-testkit/LICENSE) and
[NOTICE](https://github.com/oumuamua-labs/hekate/blob/main/hekate-testkit/NOTICE).
Commercial licenses are available from Oumuamua Labs <info@oumuamua.dev>.

Linking the prover shared library through `hekate-prover-sys` is covered by
[LICENSE-EXCEPTION](https://github.com/oumuamua-labs/hekate/blob/main/hekate-testkit/LICENSE-EXCEPTION), an
additional permission under AGPL section 7.