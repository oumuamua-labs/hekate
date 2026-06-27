# hekate-pqc

[![Crates.io](https://img.shields.io/crates/v/hekate-pqc.svg)](https://crates.io/crates/hekate-pqc)
[![Docs.rs](https://docs.rs/hekate-pqc/badge.svg)](https://docs.rs/hekate-pqc)
[![CI](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml/badge.svg)](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache2-yellow.svg)](LICENSE)

Post-quantum AIR chiplets for the [Hekate](https://github.com/oumuamua-labs/hekate) ZK proving system. Implements
ML-KEM (Kyber) decapsulation and ML-DSA (Dilithium) signature verification natively in binary fields, with supporting
NTT, basemul, high-bits, norm-check, and twiddle-ROM chiplets.

```
Proving on Apple M3 Max:
  ML-KEM-768  : 1.40 s,  331 MB peak, 4,232 KiB proof, 30.6 ms verify
  ML-DSA-44   : 2.43 s,  294 MB peak, 5,139 KiB proof, 69.0 ms verify
  ML-DSA-65   : 2.54 s,  294 MB peak, 5,156 KiB proof, 70.7 ms verify
  ML-DSA-87   : 3.98 s,  580 MB peak, 8,620 KiB proof, 115.6 ms verify
```

## Examples

- [ML-KEM-768 decapsulation proof](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/mlkem.rs)
- [ML-DSA signature verification proof](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/mldsa.rs)

## ⚠️ Security Warning

This crate has not been audited and may contain bugs and security flaws.

USE AT YOUR OWN RISK!

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).