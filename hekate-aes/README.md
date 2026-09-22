# hekate-aes

[![Crates.io](https://img.shields.io/crates/v/hekate-aes.svg)](https://crates.io/crates/hekate-aes)
[![Docs.rs](https://docs.rs/hekate-aes/badge.svg)](https://docs.rs/hekate-aes)
[![CI](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml/badge.svg)](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml)
[![License: AGPL-3.0-only](https://img.shields.io/badge/License-AGPL--3.0--only-blue.svg)](LICENSE)

*Copyright (c) 2026 Andrei Kochergin and Oumuamua Labs.*

AES-128 / AES-256 AIR chiplet for the [Hekate ZK](https://github.com/oumuamua-labs/hekate) proving system.

Implements FIPS 197 round function (SubBytes, ShiftRows, MixColumns, AddRoundKey) as a binary-field AIR with an
S-box ROM chiplet for the GF(2^8) inversion. Round-AIR trace is wired to the CPU AIR via LogUp bus.

```
Per-block proving cost (Apple M3 Max, zero-knowledge, 31,250 blocks per run):
  AES-128: ~42 µs/block, 1,217 MiB peak, 4,692 KiB proof, 21.0 ms verify
  AES-256: ~52 µs/block, 1,508 MiB peak, 4,982 KiB proof, 20.6 ms verify
```

Conditions and the base-protocol column are in the
[workspace README](https://github.com/oumuamua-labs/hekate#performance).

## Examples

- [AES-128 / AES-256 proving and verification](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/aes.rs)

---

## ⚠️ Security Warning

This crate has not been audited and may contain bugs and security flaws.

USE AT YOUR OWN RISK!

### Proof soundness vs. AES

The proof binds at **100 bits**, which is below both AES parameter sets, and the proof
is the weaker link for AES-128 and AES-256 alike. The ciphertext is still full AES.

### Constant-time trace generation

No secret-indexed `SBOX[x]` table, the key cannot leak via cache timing. The S-box is
field arithmetic, the GF(2⁸) inverse (`x²⁵⁴`) plus the FIPS 197 affine map, with no
key-dependent index, branch, or memory access.

---

## License

AGPL-3.0-only. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
Commercial licenses are available from Oumuamua Labs <info@oumuamua.dev>.