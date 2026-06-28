# hekate-aes

[![Crates.io](https://img.shields.io/crates/v/hekate-aes.svg)](https://crates.io/crates/hekate-aes)
[![Docs.rs](https://docs.rs/hekate-aes/badge.svg)](https://docs.rs/hekate-aes)
[![CI](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml/badge.svg)](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache2-yellow.svg)](LICENSE)

AES-128 / AES-256 AIR chiplet for the [Hekate ZK](https://github.com/oumuamua-labs/hekate) proving system.

Implements FIPS 197 round function (SubBytes, ShiftRows, MixColumns, AddRoundKey) as a binary-field AIR with an
S-box ROM chiplet for the GF(2^8) inversion. Round-AIR trace is wired to the CPU AIR via LogUp bus.

```
Per-block proving cost (Apple M3 Max, 31,250 blocks per run):
  AES-128: ~69 µs/block, 772 MB peak, 3,405 KiB proof
  AES-256: ~73 µs/block, 1,005 MB peak, 3,706 KiB proof
```

## Examples

- [AES-128 / AES-256 proving and verification](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/aes.rs)

---

## ⚠️ Security Warning

This crate has not been audited and may contain bugs and security flaws.

USE AT YOUR OWN RISK!

### Proof soundness vs. AES-256

Soundness is field-capped at **≈128 bits**: an AES-256 proof binds at ~2⁻¹²⁸, not 2⁻²⁵⁶,
the ZK layer is the weaker link (AES-128 is matched). The ciphertext is still full AES-256.

### Constant-time trace generation

No secret-indexed `SBOX[x]` table, the key cannot leak via cache timing. The S-box is
field arithmetic, the GF(2⁸) inverse (`x²⁵⁴`) plus the FIPS 197 affine map, with no
key-dependent index, branch, or memory access.

---

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).