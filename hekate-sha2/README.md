# hekate-sha2

[![License: AGPL-3.0-only](https://img.shields.io/badge/License-AGPL--3.0--only-blue.svg)](LICENSE)

*Copyright (c) 2026 Andrei Kochergin and Oumuamua Labs.*

SHA-256 AIR chiplet for the [Hekate ZK](https://github.com/oumuamua-labs/hekate) proving system.

The chiplet proves the 64 rounds of FIPS 180-4 §6.2.2 over bit-expanded B32 columns: one committed carry word per
32-bit add, `Ch` and `Maj` committed, every round root ungated (composition degree 2). A block is `64 / rounds_per_row`
rows; the CPU table requests `(state_in, block) -> state_out` once per block over the `sha256_rounds` bus and applies
the feed-forward add `h_out = h_in + state_out` itself through `CpuSha256Block`, the same split as the Keccak chiplet
and its sponge. Padding and chaining are the CPU table's.

```
Apple M3 Max, ZK, chiplet mounted into the CPU table, mean of 5 runs (hekate/examples/sha256.rs):
  DSC SOD, 40 compressions (2.5 KB), 2 rounds/row, 2^11 rows: 75 ms prove, 11.2 ms verify, 591 KiB proof, 67 MiB
  131,072 compressions (8.4 MB), 4 rounds/row, 2^21 rows:    11.6 s prove, 29 ms verify, 7,144 KiB proof, 5,079 MiB
```

## Examples

- [SHA-256 scaling, mounted inline](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/sha256.rs)

---

## ⚠️ Security Warning

This crate has not been audited and may contain bugs and security flaws.