# hekate-pqc

[![Crates.io](https://img.shields.io/crates/v/hekate-pqc.svg)](https://crates.io/crates/hekate-pqc)
[![Docs.rs](https://docs.rs/hekate-pqc/badge.svg)](https://docs.rs/hekate-pqc)
[![CI](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml/badge.svg)](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml)
[![License: AGPL-3.0-only](https://img.shields.io/badge/License-AGPL--3.0--only-blue.svg)](LICENSE)

*Copyright (c) 2026 Andrei Kochergin and Oumuamua Labs.*

Post-quantum AIR chiplets for the [Hekate](https://github.com/oumuamua-labs/hekate) ZK proving system. Implements
ML-KEM (Kyber) decapsulation and ML-DSA (Dilithium) signature verification natively in binary fields, with supporting
NTT, basemul, high-bits, norm-check, and twiddle-ROM chiplets.

> **Experimental.** This crate exists to demonstrate that Hekate can prove
> lattice-based cryptography natively in binary fields. The circuits have no
> external audit and the statements they prove are fully public, which is the
> case where verifying the signature directly is cheaper. Treat it as a
> working example, not a production dependency.

```
Proving on Apple M3 Max (zero-knowledge):
  ML-KEM-768  : 629 ms, 464 MiB peak, 3,483 KiB proof, 33.7 ms verify
  ML-DSA-44   : 883 ms, 477 MiB peak, 4,184 KiB proof, 44.8 ms verify
  ML-DSA-65   : 946 ms, 512 MiB peak, 4,193 KiB proof, 51.1 ms verify
  ML-DSA-87   : 1.34 s, 811 MiB peak, 5,484 KiB proof, 48.1 ms verify
```

Conditions and the base-protocol column are in the
[workspace README](https://github.com/oumuamua-labs/hekate#performance).

---

## ⚠️ Security Warning

This crate has not been audited and may contain bugs and security flaws.

USE AT YOUR OWN RISK!

### Proof soundness vs. PQC security level

The proof binds at **100 bits** regardless of parameter set, which is below every level here,
and the proof is the weaker link rather than the lattice scheme. The proven decapsulation and
verification are still the full FIPS 203 / 204 parameter sets.

---

## Examples

- [ML-KEM-768 decapsulation proof](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/mlkem.rs)
- [ML-DSA signature verification proof](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/mldsa.rs)

---

## License

AGPL-3.0-only. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
Commercial licenses are available from Oumuamua Labs <info@oumuamua.dev>.