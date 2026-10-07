# hekate-program

[![Crates.io](https://img.shields.io/crates/v/hekate-program.svg)](https://crates.io/crates/hekate-program)
[![Docs.rs](https://docs.rs/hekate-program/badge.svg)](https://docs.rs/hekate-program)
[![CI](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml/badge.svg)](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml)
[![License: AGPL-3.0-only](https://img.shields.io/badge/License-AGPL--3.0--only-blue.svg)](https://github.com/oumuamua-labs/hekate/blob/main/hekate-program/LICENSE)

*Copyright (c) 2026 Andrei Kochergin and Oumuamua Labs.*

The circuit DSL of [Hekate](https://oumuamua.dev/hekate), the Rust zero-knowledge proof engine: columns,
constraints, fixed columns, public inputs, chiplets, LogUp buses and program ids. A program is a table whose columns
you declare and whose rows your constraints relate, and a proof shows that a private trace satisfies all of them.

## Usage

```bash
cargo add hekate-program hekate-core hekate-math
```

A circuit that proves some private 32-bit words have a public XOR checksum:

```rust
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_math::{Block128, TowerField};
use hekate_program::circuit::{Circuit, CircuitProgram, Col};
use hekate_program::{FixedShape, define_columns};

type F = Block128;

define_columns! {
    Checksum {
        WORD: B32,
        ACC: B32,
        ACTIVE: Bit,
    }
}

fn table_rows(config: &Config, words: usize) -> usize {
    (words + 1).next_power_of_two().max(config.min_table_rows())
}

fn build_program(rows: usize, words: usize) -> errors::Result<CircuitProgram<F>> {
    let mut cx = Circuit::<F>::new("XorChecksum", rows)?;

    let cols = cx.schema(&Checksum::build_layout());
    let word = cols.at(Checksum::WORD);
    let acc = cols.at(Checksum::ACC);
    let active = cols.at(Checksum::ACTIVE);

    let cs = cx.cs();
    let col = |c: Col| cs.col(c.index());
    let next = |c: Col| cs.next(c.index());

    cs.constrain_named("xor_step", col(active) * (next(acc) + col(acc) + col(word)));

    cx.fix(
        active,
        FixedShape::Cadence {
            stride: 1,
            count: words,
            origin: 0,
            values: vec![F::ONE],
        },
    );

    cx.boundary(acc, 0, F::ZERO);
    cx.publish(acc, words);

    cx.compile()
}
```

Where `ACTIVE` is 1, `xor_step` makes the next accumulator this one XOR this word: addition in a binary field is XOR.
The fixed column sets `ACTIVE` on the first `words` rows, the boundary starts the accumulator at 0, and `publish` pins
the row after the last word to the public checksum. The words stay free: a real program binds them to something
public. The `hekate-core` README fills this table.

`digest::program_id` hashes the compiled program. A verifier compiles the id in as a constant and rejects every proof
made for another circuit. The id covers the pinned rows and the fixed column's count, which gives this circuit one id
per value of `words`.

## Documentation

- [Your First ZK Program](https://oumuamua.dev/hekate/docs/getting-started/your-first-zk-program): a private payment,
  proved and verified end to end
- [AIR Constraints](https://oumuamua.dev/hekate/docs/basics/air-constraints): this circuit line by line, constraint
  helpers, degree and the program id
- [LogUp Buses](https://oumuamua.dev/hekate/docs/basics/logup-buses): how tables exchange values, and the traps
- [Cryptographic Chiplets](https://oumuamua.dev/hekate/docs/basics/cryptographic-chiplets): calling ready-made tables
  for hashes, ciphers and signatures
- [API reference](https://docs.rs/hekate-program)

## License

AGPL-3.0-only. See [LICENSE](https://github.com/oumuamua-labs/hekate/blob/main/hekate-program/LICENSE) and
[NOTICE](https://github.com/oumuamua-labs/hekate/blob/main/hekate-program/NOTICE).
Commercial licenses are available from Oumuamua Labs <info@oumuamua.dev>.
