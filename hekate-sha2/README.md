# hekate-sha2

[![Crates.io](https://img.shields.io/crates/v/hekate-sha2.svg)](https://crates.io/crates/hekate-sha2)
[![Docs.rs](https://docs.rs/hekate-sha2/badge.svg)](https://docs.rs/hekate-sha2)
[![CI](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml/badge.svg)](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml)
[![License: AGPL-3.0-only](https://img.shields.io/badge/License-AGPL--3.0--only-blue.svg)](LICENSE)

*Copyright (c) 2026 Andrei Kochergin and Oumuamua Labs.*

SHA-256 AIR chiplet for the [Hekate ZK](https://github.com/oumuamua-labs/hekate) proving system.

The chiplet proves the 64 rounds of FIPS 180-4 §6.2.2 over bit-expanded B32 columns: one committed carry word per
32-bit add, `Ch` and `Maj` committed, every round root ungated (composition degree 2). A block is `64 / rounds_per_row`
rows; the CPU table requests `(state_in, block) -> state_out` once per block over the `sha256_rounds` bus and applies
the feed-forward add `h_out = h_in + state_out` itself through `CpuSha256Block`, the same split as the Keccak chiplet
and its sponge. Padding and chaining are the CPU table's.

```
Apple M3 Max, ZK, chiplet mounted into the CPU table, best of three runs (hekate/examples/sha256.rs):
  DSC SOD, 40 compressions (2.5 KB), 2 rounds/row, 2^11 rows: 70 ms prove, 9.8 ms verify, 538 KiB proof, 63 MiB
  131,072 compressions (8.4 MB), 4 rounds/row, 2^21 rows:    11.57 s prove, 22.5 ms verify, 5,479 KiB proof, 5,216 MiB
```

## Examples

- [SHA-256 scaling, mounted inline](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/sha256.rs)

### Usage

SHA-256 of one message, one compression per host row, with the chiplet attached as its own table:

```rust
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::trace::ColumnType;
use hekate_math::{Block128, TowerField};
use hekate_program::FixedShape;
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_sha2::{CpuSha256Block, IV, ROUNDS, STATE_WORDS, Sha256Chiplet, pad_message};

type F = Block128;

fn sha256(message: &[u8]) -> errors::Result<(CircuitProgram<F>, CpuSha256Block, Sha256Chiplet<F>)> {
    let blocks = pad_message(message).len();
    let rounds_per_row = 4;
    let floor = Config::default().min_table_rows();

    let sha = Sha256Chiplet::<F>::new(
        (blocks * ROUNDS / rounds_per_row)
            .next_power_of_two()
            .max(floor),
        blocks,
        rounds_per_row,
    )?;

    let mut cx = Circuit::<F>::new("Host", blocks.next_power_of_two().max(floor))?;

    let block = CpuSha256Block::declare(&mut cx, 0);

    let active = cx.column(ColumnType::Bit);
    let chain = cx.column(ColumnType::Bit);

    let prefix = |count: usize| FixedShape::Cadence {
        stride: 1,
        count,
        origin: 0,
        values: vec![F::ONE],
    };

    cx.fix(active, prefix(blocks));
    cx.fix(chain, prefix(blocks - 1));

    block.connect(&mut cx, active)?;

    for (i, &iv) in IV.iter().enumerate() {
        cx.boundary(block.h_in_words.at(i), 0, F::from(u128::from(iv)));
    }

    let cs = cx.cs();
    let chain = cs.col(chain.index());

    // Each block's h_in is the previous block's h_out
    for i in 0..STATE_WORDS {
        let h_out = cs.col(block.h_out_words.at(i).index());
        cs.assert_zero_when(chain, cs.next(block.h_in_words.at(i).index()) + h_out);
    }

    // Public: the digest, h_out of the last block
    for i in 0..STATE_WORDS {
        cx.publish(block.h_out_words.at(i), blocks - 1);
    }

    cx.attach(sha.def()?);

    Ok((cx.compile()?, block, sha))
}
```

Host row `b` holds `block.write(&mut tb, b, &calls[b])`, and the two bits after the block's columns, `active` and
`chain`, are one on rows `0..blocks` and `0..blocks - 1`. `calls[b]` pairs block `b` of `pad_message(message)` with
`h_in` from `calls[b - 1].h_out()`, `IV` for the first, and `sha.trace(&calls)` is the chiplet trace.

---

## ⚠️ Security Warning

This crate has not been audited and may contain bugs and security flaws.