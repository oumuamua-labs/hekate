# Hekate ZK Engine

[![CI](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml/badge.svg)](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml)
[![License: AGPL-3.0-only](https://img.shields.io/badge/License-AGPL--3.0--only-blue.svg)](./LICENSE)

*Copyright (c) 2026 Andrei Kochergin and Oumuamua Labs.*

Zero-knowledge proof system over binary tower fields. Streaming architecture. Bounded memory. Edge-native.
Hekate proves computations in GF(2^128) using Sumcheck + Brakedown PCS with O(N) prover time and O(N) memory.
100 proven bits of soundness.

Documentation: [oumuamua.dev/hekate/docs](https://oumuamua.dev/hekate/docs)

> [!WARNING]  
> This workspace is under aggressive development. APIs, ABIs, and cryptographic signatures will break
> without notice. Do not deploy to mainnet.

> [!NOTE]  
> The verifier, core SDK, and cryptographic chiplets are open-source under AGPL-3.0-only. The prover
> and compression engine stay proprietary, shipped as free binaries for macOS (Apple Silicon),
> Linux (ARM64, glibc), iOS (ARM64, device and simulator), and Android (ARM64).

> [!IMPORTANT]  
> [`hekate-mobile`](https://github.com/oumuamua-labs/hekate-mobile) compiles a Rust prover into a signed
> iOS `.xcframework` and Android `.aar` behind a typed Swift / Kotlin API, one `await` per proof, zero ZK
> terminology across the boundary. Shipping ZK to edge devices? Start there, with the Swift and Kotlin
> guides at [oumuamua.dev/mobile](https://oumuamua.dev/mobile).

---

## ⚠️ Security Warning

This workspace has not been independently audited and may contain bugs and security flaws.

The failures fixed in 0.31.0 to 0.37.0 are written up as postmortems:
[0.32](docs/postmortem-0.32.md), [0.33](docs/postmortem-0.33.md), [0.34](docs/postmortem-0.34.md),
[0.35](docs/postmortem-0.35.md), [0.36](docs/postmortem-0.36.md) and [0.37](docs/postmortem-0.37.md). Each states
what broke, how it was found, what fixed it and what the release still does not promise.
[Soundness and Security](https://oumuamua.dev/hekate/docs/advanced/soundness-and-security) collects the
guarantees a proof gives today and the checks that stay with you.

USE AT YOUR OWN RISK!

---

## The Hekate Ecosystem

Hekate is not one crate. The prover is the engine; the math core, hardware chiplets, mobile toolchain,
and fuzzer ship as independent crates you compose as needed.

| Crate                                                                                      | Role                                                                                                        |
|:-------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------|
| [`hekate-math`](https://github.com/oumuamua-labs/hekate-math)                              | Binary tower field arithmetic, constant-time, PMULL on aarch64. The mathematical core.                      |
| [`hekate-prover-sys`](https://github.com/oumuamua-labs/hekate/tree/main/hekate-prover-sys) | Open FFI shim. Links the signed prover cdylib over a stable C ABI; the only crate that can call the prover. |
| [`hekate-gadgets`](https://github.com/oumuamua-labs/hekate/tree/main/hekate-gadgets)       | Base AIR chiplets: 32- and 64-bit integer arithmetic, RAM, ROM and modular exponentiation.                  |
| [`hekate-keccak`](https://github.com/oumuamua-labs/hekate/tree/main/hekate-keccak)         | Keccak-f[1600] chiplet plus SHA-3 / SHAKE. Virtual packing, ~16x memory savings.                            |
| [`hekate-aes`](https://github.com/oumuamua-labs/hekate/tree/main/hekate-aes)               | AES-128 / AES-256 round-function chiplet (FIPS 197) with an S-box ROM.                                      |
| [`hekate-sha2`](https://github.com/oumuamua-labs/hekate/tree/main/hekate-sha2)             | SHA-256 compression chiplet (FIPS 180-4). Bit-expanded B32 columns, degree 2, 1-16 rounds per row.          |
| [`hekate-rsa`](https://github.com/oumuamua-labs/hekate/tree/main/hekate-rsa)               | RSA-2048 PKCS#1 v1.5 signature statements (RFC 8017) over the modexp and SHA-256 chiplets.                  |
| [`hekate-pqc`](https://github.com/oumuamua-labs/hekate/tree/main/hekate-pqc)               | ML-KEM KeyGen, Encaps and Decaps and ML-DSA verification (FIPS 203 / 204) over NTT, Sampler, Codec tables.  |
| [`hekate-mobile`](https://github.com/oumuamua-labs/hekate-mobile)                          | Wraps a Rust prover into a signed iOS `.xcframework` / Android `.aar` with a typed Swift / Kotlin API.      |
| [`hekate-scribble`](https://github.com/oumuamua-labs/hekate/tree/main/hekate-scribble)     | Trace-mutation fuzzer. Tampers a valid trace, panics if your constraints miss the tamper.                   |

The workspace also holds `hekate-core`, `hekate-crypto`, `hekate-program`, `hekate-verifier` and
`hekate-sdk`. [System Architecture](https://oumuamua.dev/hekate/docs/basics/system-architecture#the-crates)
draws all of them as one stack, and each chiplet has a page at
[oumuamua.dev/primitives](https://oumuamua.dev/primitives) stating what it proves, which values a program can
publish and what stays outside the proof.

---

## What It Does

**Binary tower field arithmetic**, GF(2^8) through GF(2^128), recursive tower extension, hardware-accelerated via
PMULL on aarch64. Constant-time by default.

**Chiplet architecture**, Independent AIR tables (Keccak, AES, RAM, NTT, ML-KEM, ML-DSA) with own traces. No column
waste, no forced padding. Tables linked by LogUp bus.

**Virtual packing**, Keccak stores 1600 bits in 25 physical B64 columns instead of 1600 bit columns. Bits expand JIT in
registers. 16x memory savings.

**Linear-code commitments**, Brakedown PCS: O(N) prover, O(N) memory. MDS Reed-Solomon row code via additive
binary-field FFT, exact distance δ = 1 − rate. Merkle tree over encoded columns only.

**Zero knowledge**, on by default. A one-time pad covers every round, claim and bus sum a proof sends, the commitment's
opened columns and leaves look random, and a zk-Ligero argument proves the checks the verifier cannot run on padded
values. [Zero Knowledge](https://oumuamua.dev/hekate/docs/advanced/zero-knowledge) specifies all three.

**Post-quantum crypto suite**, ML-DSA (Dilithium) signature verification, ML-KEM (Kyber) key generation,
encapsulation and decapsulation, AES-128/256. AES and Keccak are native to binary fields; lattice arithmetic
mod q runs on bit-decomposed carry chains.

### Hardware Support

| Architecture | Status     | Instructions                          |
|:-------------|:-----------|:--------------------------------------|
| aarch64      | Production | PMULL, NEON                           |
| x86_64       | Fallback   | Software fallback (PCLMULQDQ roadmap) |
| WASM         | Planned    | Software multiply                     |

---

## Examples

End-to-end programs that prove and verify with `hekate-prover-sys` and `hekate-verifier`. Each file is a self-contained
binary you can run with `cargo run --release --example <name>`.

- [ML-DSA signature verification](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/mldsa.rs) (FIPS 204;
  44 / 65 / 87 levels)
- [RSA-2048 PKCS#1 v1.5 verification](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/rsa_pkcs1.rs)
  (RFC 8017, over the modexp and SHA-256 chiplets)
- [ML-KEM sender](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/mlkem_sender.rs) (FIPS 203
  Encaps with an AES-256-CTR payload; 512 / 768 / 1024 levels)
- [ML-KEM receiver](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/mlkem_receiver.rs) (FIPS 203
  KeyGen chained into Decaps; 512 / 768 / 1024 levels)
- [AES-128 / AES-256 block proving](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/aes.rs) (FIPS 197)
- [Keccak kernel](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/keccak.rs) (CPU AIR
  with embedded f1600 permutation)
- [SHA-256 compression](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/sha256.rs) (FIPS 180-4;
  1, 2, 4, 8 or 16 rounds per row)
- [32-bit integer arithmetic](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/arith.rs) (add / sub /
  and / xor / not / lt via `IntArithmeticChiplet`)
- [Fibonacci on a chiplet table](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/fibonacci.rs)
  (`IntArithmeticChiplet` mounted as the program's own table, no bus)
- [Raw 32-bit Fibonacci](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/fibonacci_raw.rs)
  (a bit-sliced carry chain in the CPU AIR, no chiplet; the integer-arithmetic benchmark)
- [RAM read/write proof](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/ram.rs) (offline-memory
  consistency via `RamChiplet`)
- [ROM instruction fetch](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/rom.rs) (fetches
  matched against a ROM table via `RomChiplet`)
- [Three chiplets on one CPU](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/many_chiplets.rs)
  (ROM, integer arithmetic and RAM, each on its own bus)

---

## Getting Started

The guides at [oumuamua.dev/hekate/docs](https://oumuamua.dev/hekate/docs), in reading order:

- [Installation](https://oumuamua.dev/hekate/docs/getting-started/installation), the crates, the signed prover binary,
  supported targets
- [Your First ZK Program](https://oumuamua.dev/hekate/docs/getting-started/your-first-zk-program), a private payment
  proved end-to-end
- [System Architecture](https://oumuamua.dev/hekate/docs/basics/system-architecture), binary tower fields, Sumcheck,
  Brakedown, what the verifier trusts
- [Prover Engine](https://oumuamua.dev/hekate/docs/basics/prover-engine), commit every table, prove the constraints
  with Sumcheck, open the commitments
- [AIR Constraints](https://oumuamua.dev/hekate/docs/basics/air-constraints), constraints over two rows, boundaries,
  gadgets, degree
- [Execution Trace](https://oumuamua.dev/hekate/docs/basics/execution-trace), column types, padding rows, wiping
  secrets
- [Verifier Logic](https://oumuamua.dev/hekate/docs/basics/verifier-logic), pinning the program id, one proof, a
  prepared program, a batch
- [LogUp Buses](https://oumuamua.dev/hekate/docs/basics/logup-buses), a multiset check as a sum of fractions, a bus of
  your own
- [Cryptographic Chiplets](https://oumuamua.dev/hekate/docs/basics/cryptographic-chiplets), the chiplet set, calling a
  chiplet, a chiplet of your own
- [PIOP Protocol](https://oumuamua.dev/hekate/docs/advanced/piop-protocol), multilinear columns, Sumcheck down to one
  point, the commitment
- [Zero Knowledge](https://oumuamua.dev/hekate/docs/advanced/zero-knowledge), how a proof hides the witness, and the
  [ring-switching paper](https://oumuamua.dev/blog/zk-ring-switching) behind packed bit columns
- [Soundness and Security](https://oumuamua.dev/hekate/docs/advanced/soundness-and-security), threat model, adversarial
  test suite, Fiat-Shamir binding

---

## Performance

Proving time, verification time, proof size and peak memory for the hashing, signature, encryption and
integer-arithmetic examples are at [oumuamua.dev/hekate/benchmarks](https://oumuamua.dev/hekate/benchmarks),
in zero-knowledge and base mode (`HEKATE_ZK=0`), with the machine, the build features and the command behind
each run.

---

## License

AGPL-3.0-only. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

Linking against the proprietary prover shared library is covered by the
[prover linking exception](LICENSE-EXCEPTION), an additional permission under AGPL section 7.

Commercial licenses are available from Oumuamua Labs <info@oumuamua.dev>. The Free, Pro and Enterprise tiers
are compared at [oumuamua.dev/hekate/pricing](https://oumuamua.dev/hekate/pricing).

Hekate does not accept external code contributions. See [CONTRIBUTING](CONTRIBUTING.md).
