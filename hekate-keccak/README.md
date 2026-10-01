# hekate-keccak

[![Crates.io](https://img.shields.io/crates/v/hekate-keccak.svg)](https://crates.io/crates/hekate-keccak)
[![Docs.rs](https://docs.rs/hekate-keccak/badge.svg)](https://docs.rs/hekate-keccak)
[![CI](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml/badge.svg)](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml)
[![License: AGPL-3.0-only](https://img.shields.io/badge/License-AGPL--3.0--only-blue.svg)](LICENSE)

*Copyright (c) 2026 Andrei Kochergin and Oumuamua Labs.*

Keccak-f[1600] AIR chiplet for the [Hekate](https://github.com/oumuamua-labs/hekate) ZK proving system. Includes
SHA-3-256, SHA-3-512, SHAKE128, and SHAKE256 sponge constructions.

Virtual packing: 1600 state bits stored in 25 physical B64 columns instead of 1600 bit columns. Bits expand JIT in
registers during evaluation. ~16x memory savings vs. naive bit-column layout.

```
Scaling (Apple M3 Max, zero-knowledge):
  2^15 trace rows (1,310 permutations): 191 ms, 166 MiB peak, 851 KiB proof, 12.4 ms verify
  2^20 trace rows (41,943 permutations): 3.92 s, 2,418 MiB peak, 3,545 KiB proof, 17.5 ms verify
```

Conditions and the base-protocol column are in the
[workspace README](https://github.com/oumuamua-labs/hekate#performance).

---

## ⚠️ Security Warning

This crate has not been audited and may contain bugs and security flaws.

USE AT YOUR OWN RISK!

---

## Examples

- [Keccak kernel (CPU AIR with embedded permutation)](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/keccak.rs)

### Usage

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

The host trace holds call `k`'s input lanes on row `25k` and its output lanes on row `25k + 24`, `SELECTOR` set
on both, and the columns `generate_keccak_trace` builds from the call inputs are appended to the same trace. The
chiplet proves each output is Keccak-f[1600] of its input. Binding the digest to a message is the host's job: each
call's input is the previous output with the next block XORed into its rate lanes.

## Benchmarks

Run the Criterion suite natively. Sizes 2^12, 2^15, and 2^20 are baked in. Throughput is reported in MB/s of hashed
input (SHA-3 rate 136 B/permutation, 25 trace rows per permutation).

```bash
# Run the full sweep
cargo bench --bench keccak

# Run a specific trace size (e.g., 2^15 rows)
cargo bench --bench keccak -- Prove/15
```

---

## License

AGPL-3.0-only. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
Commercial licenses are available from Oumuamua Labs <info@oumuamua.dev>.