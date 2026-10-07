# hekate-core

[![Crates.io](https://img.shields.io/crates/v/hekate-core.svg)](https://crates.io/crates/hekate-core)
[![Docs.rs](https://docs.rs/hekate-core/badge.svg)](https://docs.rs/hekate-core)
[![CI](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml/badge.svg)](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml)
[![License: AGPL-3.0-only](https://img.shields.io/badge/License-AGPL--3.0--only-blue.svg)](https://github.com/oumuamua-labs/hekate/blob/main/hekate-core/LICENSE)

*Copyright (c) 2026 Andrei Kochergin and Oumuamua Labs.*

Configuration, execution traces and proof types of [Hekate](https://oumuamua.dev/hekate), the Rust zero-knowledge
proof engine. The witness of a Hekate program is a table of typed columns: `TraceBuilder` fills it, `Config` sets the
proof parameters, and `InnerProof` is what the prover returns.

## Usage

```bash
cargo add hekate-core hekate-math
```

The trace of a XOR checksum over private words. Row `i` holds word `i` and the checksum of the words before it, and
the row after the last word holds the result:

```rust
use hekate_core::errors;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate_math::Block32;

const WORD: usize = 0;
const ACC: usize = 1;
const ACTIVE: usize = 2;

const LAYOUT: [ColumnType; 3] = [ColumnType::B32, ColumnType::B32, ColumnType::Bit];

fn generate_trace(words: &[u32], rows: usize) -> errors::Result<ColumnTrace> {
    let num_vars = rows.trailing_zeros() as usize;
    let mut tb = TraceBuilder::new_secret(&LAYOUT, num_vars)?;
    let mut acc = 0;

    for (row, &word) in words.iter().enumerate() {
        tb.set_b32(WORD, row, Block32::from(word))?;
        tb.set_b32(ACC, row, Block32::from(acc))?;

        acc ^= word;
    }

    tb.set_b32(ACC, words.len(), Block32::from(acc))?;
    tb.fill_selector(ACTIVE, words.len())?;

    Ok(tb.build())
}
```

`rows` is a power of two above `words.len()` and at least `Config::default().min_table_rows()`. `new_secret` zeroizes
the columns when the trace drops. The `hekate-program` README builds the circuit this trace satisfies.

## Features

| Feature         | Default | Effect                                                                   |
|:----------------|:--------|:-------------------------------------------------------------------------|
| `std`           | yes     | The standard library                                                     |
| `parallel`      | yes     | Rayon threads, here and in `hekate-crypto` and `hekate-math`             |
| `blake3`        | yes     | BLAKE3 as `DefaultHasher`                                                |
| `sha2`, `sha3`  | no      | SHA-256 or SHA3-256 as `DefaultHasher`; the prover must hash the same way |
| `secure-memory` | no      | Zeroize every trace on drop, also those built with `TraceBuilder::new`   |

## Documentation

- [Execution Trace](https://oumuamua.dev/hekate/docs/basics/execution-trace): columns and their types, padding, the
  last row and secrets in memory
- [Prover Engine: configuration](https://oumuamua.dev/hekate/docs/basics/prover-engine#configuration): what `Config`
  sets and what it costs
- [API reference](https://docs.rs/hekate-core)

## License

AGPL-3.0-only. See [LICENSE](https://github.com/oumuamua-labs/hekate/blob/main/hekate-core/LICENSE) and
[NOTICE](https://github.com/oumuamua-labs/hekate/blob/main/hekate-core/NOTICE).
Commercial licenses are available from Oumuamua Labs <info@oumuamua.dev>.
