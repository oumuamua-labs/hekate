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

- [Keccak isolated chiplet (standalone AIR)](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/keccak.rs)
- [Keccak inline kernel (CPU AIR with embedded permutation)](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/keccak_inline.rs)

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