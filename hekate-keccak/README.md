# hekate-keccak

[![Crates.io](https://img.shields.io/crates/v/hekate-keccak.svg)](https://crates.io/crates/hekate-keccak)
[![Docs.rs](https://docs.rs/hekate-keccak/badge.svg)](https://docs.rs/hekate-keccak)
[![CI](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml/badge.svg)](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml)
[![License: AGPL-3.0-only](https://img.shields.io/badge/License-AGPL--3.0--only-blue.svg)](https://github.com/oumuamua-labs/hekate/blob/main/hekate-keccak/LICENSE)

*Copyright (c) 2026 Andrei Kochergin and Oumuamua Labs.*

Keccak-f[1600] in zero knowledge for [Hekate](https://oumuamua.dev/hekate), the Rust zero-knowledge proof engine: a
chiplet that proves each output is the permutation of its input, and SHA3-256, SHA3-512, SHAKE128 and SHAKE256 host
helpers that return the permutation calls of one hash. The 1600 state bits sit in 25 64-bit columns and expand to bits
only during evaluation.

## ⚠️ Security Warning

This crate has not been audited and may contain bugs and security flaws.

USE AT YOUR OWN RISK!

## Usage

```bash
cargo add hekate-keccak hekate-core hekate-math hekate-program zeroize
```

The Keccak-f[1600] calls of one SHA3-256 hash, with the chiplet mounted in the host table:

```rust
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_keccak::{KeccakCall, KeccakChiplet, sha3_256};
use hekate_math::Block128;
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram, Col};
use hekate_program::define_columns;
use zeroize::Zeroizing;

type F = Block128;

define_columns! {
    HostColumns {
        LANES: [B64; 25],
        SELECTOR: Bit,
    }
}

fn sha3(message: &[u8]) -> errors::Result<(CircuitProgram<F>, Zeroizing<Vec<KeccakCall>>)> {
    let (_, calls) = sha3_256(message);

    let blocks = calls.len();
    let rows = (blocks * KeccakChiplet::BLOCK_ROWS)
        .next_power_of_two()
        .max(Config::default().min_table_rows());

    let mut cx = Circuit::<F>::new("Host", rows)?;

    let host = cx.schema(&HostColumns::build_layout());

    let sel = host.at(HostColumns::SELECTOR);
    let lanes: Vec<Col> = (0..25).map(|i| host.at(HostColumns::LANES + i)).collect();

    cx.call(&KeccakChiplet::service(), &lanes, sel)?;

    cx.fix(
        sel,
        KeccakChiplet::host_selector_shape(KeccakChiplet::BLOCK_ROWS, blocks),
    );

    cx.mount(ChipletDef::from_air(&KeccakChiplet::new(rows, blocks))?);

    // Public: the digest, the first 4 lanes of the last output
    let last_output = blocks * KeccakChiplet::BLOCK_ROWS - 1;

    for &lane in &lanes[..4] {
        cx.publish(lane, last_output);
    }

    Ok((cx.compile()?, calls))
}
```

The host trace holds call `k`'s input lanes on row `25k` and its output lanes on row `25k + 24`, `SELECTOR` set on
both, and the columns `generate_keccak_trace` builds from the call inputs are appended to the same trace. Binding the
digest to a message is the host's job: each call's input is the previous output with the next block XORed into its
rate lanes.

[`keccak.rs`](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/keccak.rs) is a complete program, and
`cargo bench --bench keccak` runs the Criterion suite at 2^12, 2^15 and 2^20 rows.

## Documentation

- [Keccak-f[1600]](https://oumuamua.dev/primitives/hashing/keccak): what the proof states, binding a digest to a
  message, and proving time, proof size and memory
- [Cryptographic Chiplets](https://oumuamua.dev/hekate/docs/basics/cryptographic-chiplets#inside-a-chiplet-keccak-f-1600):
  inside this chiplet, and the ways a host table calls one
- [API reference](https://docs.rs/hekate-keccak)

## License

AGPL-3.0-only. See [LICENSE](https://github.com/oumuamua-labs/hekate/blob/main/hekate-keccak/LICENSE) and
[NOTICE](https://github.com/oumuamua-labs/hekate/blob/main/hekate-keccak/NOTICE).
Commercial licenses are available from Oumuamua Labs <info@oumuamua.dev>.
