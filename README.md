# Hekate ZK Engine

[![CI](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml/badge.svg)](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml)
[![License: AGPL-3.0-only](https://img.shields.io/badge/License-AGPL--3.0--only-blue.svg)](./LICENSE)

*Copyright (c) 2026 Andrei Kochergin and Oumuamua Labs.*

Zero-knowledge proof system over binary tower fields. Streaming architecture. Bounded memory. Edge-native.
Hekate proves computations in GF(2^128) using Sumcheck + Brakedown PCS with O(N) prover time and O(N) memory.

> [!WARNING]  
> This workspace is under aggressive development. APIs, ABIs, and cryptographic signatures will break
> without notice. Do not deploy to mainnet.

> [!NOTE]  
> The verifier, core SDK, and cryptographic chiplets are open-source under AGPL-3.0-only. The prover
> and compression engine stay proprietary, shipped as free binaries for macOS (Apple Silicon),
> Linux (ARM64, glibc), and Android (ARM64). Linking the two is covered by the
> [prover linking exception](LICENSE-EXCEPTION).

> [!IMPORTANT]  
> [`hekate-mobile`](https://github.com/oumuamua-labs/hekate-mobile) compiles a Rust prover into a signed
> iOS `.xcframework` and Android `.aar` behind a typed Swift / Kotlin API, one `await` per proof, zero ZK
> terminology across the boundary. Shipping ZK to edge devices? Start there.

---

## ⚠️ Security Warning

This workspace has not been independently audited and may contain bugs and security flaws.

USE AT YOUR OWN RISK!

---

## The Hekate Ecosystem

Hekate is not one crate. The prover is the engine; the math core, hardware chiplets, mobile toolchain,
and fuzzer ship as independent crates you compose as needed.

| Crate                                                                                      | Role                                                                                                        |
|:-------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------|
| [`hekate-math`](https://github.com/oumuamua-labs/hekate-math)                              | Binary tower field arithmetic, constant-time, PMULL / PCLMULQDQ. The mathematical core.                     |
| [`hekate-prover-sys`](https://github.com/oumuamua-labs/hekate/tree/main/hekate-prover-sys) | Open FFI shim. Links the signed prover cdylib over a stable C ABI; the only crate that can call the prover. |
| [`hekate-keccak`](https://github.com/oumuamua-labs/hekate/tree/main/hekate-keccak)         | Keccak-f[1600] chiplet plus SHA-3 / SHAKE. Virtual packing, ~16x memory savings.                            |
| [`hekate-aes`](https://github.com/oumuamua-labs/hekate/tree/main/hekate-aes)               | AES-128 / AES-256 round-function chiplet (FIPS 197) with an S-box ROM.                                      |
| [`hekate-sha2`](https://github.com/oumuamua-labs/hekate/tree/main/hekate-sha2)             | SHA-256 compression chiplet (FIPS 180-4). Bit-expanded B32 columns, degree 2, 1-16 rounds per row.          |
| [`hekate-rsa`](https://github.com/oumuamua-labs/hekate/tree/main/hekate-rsa)               | RSA-2048 PKCS#1 v1.5 signature statements (RFC 8017) over the modexp and SHA-256 chiplets.                  |
| [`hekate-pqc`](https://github.com/oumuamua-labs/hekate/tree/main/hekate-pqc)               | ML-KEM decapsulation and ML-DSA verification (FIPS 203 / 204), with NTT, basemul, norm-check.               |
| [`hekate-mobile`](https://github.com/oumuamua-labs/hekate-mobile)                          | Wraps a Rust prover into a signed iOS `.xcframework` / Android `.aar` with a typed Swift / Kotlin API.      |
| [`hekate-scribble`](https://github.com/oumuamua-labs/hekate/tree/main/hekate-scribble)     | Trace-mutation fuzzer. Tampers a valid trace, panics if your constraints miss the tamper.                   |

The in-workspace crates: `hekate-core`, `hekate-crypto`, `hekate-program`, `hekate-verifier`,
`hekate-sdk` are shown in the stack above.

---

## What It Does

**Binary tower field arithmetic**, GF(2^8) through GF(2^128), recursive tower extension, hardware-accelerated via
PMULL/CLMUL. Constant-time by default.

**Chiplet architecture**, Independent AIR tables (Keccak, AES, RAM, NTT, ML-KEM, ML-DSA) with own traces and
commitments. No column waste, no forced padding. Tables linked by LogUp bus.

**Virtual packing**, Keccak stores 1600 bits in 25 physical B64 columns instead of 1600 bit columns. Bits expand JIT in
registers. 16x memory savings.

**Linear-code commitments**, Brakedown PCS: O(N) prover, O(N) memory. MDS Reed-Solomon row code via additive
binary-field FFT, exact distance δ = 1 − rate. Merkle tree over encoded columns only (raw trace never hashed, true ZK).

**Post-quantum crypto suite**, ML-DSA (Dilithium) signature verification, ML-KEM (Kyber) decapsulation, AES-128/256,
all proven natively in binary fields without bit-decomposition overhead.

### Hardware Support

| Architecture | Status     | Instructions                          |
|:-------------|:-----------|:--------------------------------------|
| aarch64      | Production | PMULL, NEON                           |
| x86_64       | Fallback   | Software fallback (PCLMULQDQ roadmap) |
| WASM         | Planned    | Software multiply                     |

---

## Quick Example

Real 32-bit-integer Fibonacci. The CPU side holds five columns and the two Fibonacci transition
constraints. Every `u32` ADD is offloaded to the `IntArithmeticChiplet`, its own trace, own
commitment, own ZeroCheck, own evaluation argument, and is wired in by a LogUp bus
(`(val_a, val_b, val_res, opcode, request_idx)` keys with a row-index clock). The result reaches
the verdict through `publish`: a boundary pin to the public input on a row whose schedule a fixed
column forces, with the transition chain determined from the pinned origins.

```rust
use hekate::core::errors;
use hekate::math::{Block128, TowerField};
use hekate_gadgets::{CpuArithColumns, IntArithmeticChiplet};
use hekate_program::FixedShape;
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram};

type F = Block128;

fn build_program(num_rows: usize) -> errors::Result<CircuitProgram<F>> {
    let mut cx = Circuit::<F>::new("Fibonacci", num_rows)?;

    let cpu = cx.schema(&CpuArithColumns::build_layout());
    let selector = cpu.at(CpuArithColumns::SELECTOR);
    let a = cpu.at(CpuArithColumns::VAL_A);
    let b = cpu.at(CpuArithColumns::VAL_B);
    let res = cpu.at(CpuArithColumns::VAL_RES);

    let cs = cx.cs();

    let s = cs.col(selector.index());
    let val_b = cs.col(b.index());
    let val_res = cs.col(res.index());
    let next_a = cs.next(a.index());
    let next_b = cs.next(b.index());

    cs.constrain(s * (next_a + val_b));     // next_a = b
    cs.constrain(s * (next_b + val_res));   // next_b = a + b (chiplet provides val_res)

    cx.bus(
        IntArithmeticChiplet::BUS_ID,
        IntArithmeticChiplet::cpu_linking_spec(),
    );

    cx.attach(ChipletDef::from_air(&IntArithmeticChiplet::new(
        32,
        num_rows,
        num_rows - 1,
    )?)?);

    cx.fix(
        selector,
        FixedShape::Cadence {
            stride: 1,
            count: num_rows - 1,
            origin: 0,
            values: vec![F::ONE],
        },
    );

    cx.boundary(a, 0, F::ZERO);
    cx.boundary(b, 0, F::ONE);

    cx.publish(b, num_rows - 1);

    cx.compile()
}
```

Trace generation builds the CPU columns and the chiplet trace independently; they meet on the bus.
`CpuArithColumns` is the shipped CPU-side schema for that bus.

```rust
use hekate::core::errors;
use hekate::core::trace::{ColumnTrace, TraceBuilder};
use hekate::math::{Bit, Block32, TowerField};
use hekate_gadgets::{
    ArithmeticOpcode, CpuArithColumns, IntArithmeticLayout, IntArithmeticOp,
    generate_arithmetic_trace,
};

fn generate_traces(num_rows: usize) -> errors::Result<(ColumnTrace, ColumnTrace, u32)> {
    let num_vars = num_rows.trailing_zeros() as usize;

    let mut tb = TraceBuilder::new(&CpuArithColumns::build_layout(), num_vars)?;
    let mut ops: Vec<IntArithmeticOp> = Vec::with_capacity(num_rows - 1);

    let mut a: u32 = 0;
    let mut b: u32 = 1;

    for i in 0..num_rows - 1 {
        let res = a.wrapping_add(b);

        tb.set_b32(CpuArithColumns::VAL_A, i, Block32::from(a))?;
        tb.set_b32(CpuArithColumns::VAL_B, i, Block32::from(b))?;
        tb.set_b32(CpuArithColumns::VAL_RES, i, Block32::from(res))?;
        tb.set_b32(CpuArithColumns::OPCODE, i, Block32::from(ArithmeticOpcode::ADD as u32))?;
        tb.set_bit(CpuArithColumns::SELECTOR, i, Bit::ONE)?;

        ops.push(IntArithmeticOp::U32 {
            op: ArithmeticOpcode::ADD,
            a,
            b,
            request_idx: i as u32,
        });

        a = b;
        b = res;
    }

    // Padding row: selector = 0, val_b carries fib[N-1] for the boundary pin.
    tb.set_b32(CpuArithColumns::VAL_A, num_rows - 1, Block32::from(a))?;
    tb.set_b32(CpuArithColumns::VAL_B, num_rows - 1, Block32::from(b))?;

    let cpu_trace = tb.build();

    let arith_layout = IntArithmeticLayout::compute(32);
    let arith_trace = generate_arithmetic_trace(&ops, &arith_layout, num_rows)?;

    Ok((cpu_trace, arith_trace, b))
}
```

The chiplet enforces 32-bit ADD with carry, boolean-checks its own selectors, and zero-pins shadow
columns when its row is idle. The CPU AIR only needs the two transition constraints above, the
LogUp bus guarantees `val_res = a + b` for every row where `s = 1`.

Wire up the program, instance, and witness, then prove with `hekate-prover-sys` and verify with
`hekate-verifier`. The transcript label and `Config` must match across both sides, the driver builds
one `config` and reuses it. `verify` returns `true` only if every Sumcheck round, the LogUp bus sums,
and the evaluation openings hold.

```rust
use hekate::core::config::Config;
use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate_program::{ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;
use rand::{TryRngCore, rngs::OsRng};

fn run(num_rows: usize, audited_id: &[u8; 32]) -> Result<bool, Box<dyn core::error::Error>> {
    let (cpu, arith, fib_n) = generate_traces(num_rows)?;

    let program = build_program(num_rows)?;
    let instance = ProgramInstance::new(num_rows, vec![F::from(fib_n as u128)]);
    let witness = ProgramWitness::new(cpu).with_chiplets(vec![arith]);

    let config = Config::default();

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed)?;

    let proof = prove(
        b"Fibonacci",
        &program,
        &instance,
        &witness,
        &config,
        blinding_seed,
        None,
    )?;

    let mut transcript = Transcript::<DefaultHasher>::new(b"Fibonacci");

    // `audited_id` is a constant of the verifying build,
    // printed once by `digest::program_id_hex`.
    Ok(HekateVerifier::<F, DefaultHasher>::verify(
        audited_id,
        &program,
        &instance,
        &proof,
        &mut transcript,
        &config,
    )?)
}
```

---

## Examples

End-to-end programs that prove and verify with `hekate-prover-sys` and `hekate-verifier`. Each file is a self-contained
binary you can run with `cargo run --release --example <name>`.

- [ML-DSA signature verification](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/mldsa.rs) (FIPS 204;
  44 / 65 / 87 levels)
- [ML-KEM-768 decapsulation](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/mlkem.rs) (FIPS 203)
- [AES-128 / AES-256 block proving](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/aes.rs) (FIPS 197)
- [Keccak inline kernel](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/keccak_inline.rs) (CPU AIR
  with embedded f1600 permutation)
- [32-bit integer arithmetic](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/arith.rs) (add / sub /
  and / xor / not / lt via `IntArithmeticChiplet`)
- [RAM read/write proof](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/ram.rs) (offline-memory
  consistency via `RamChiplet`)

---

## Performance

All numbers on Apple M3 Max (16 cores, 48 GB RAM), `--release`, features
`std parallel blake3 table-math`, `Config::prod()`. Cells read zero-knowledge / base,
the second value being the same run under `HEKATE_ZK=0`. Measured with the example
binaries in `hekate/examples/` on an otherwise idle machine; every figure is the
best of three runs. Peak memory is the larger of the process peak physical footprint
and its peak resident set size.

Reproduce:

```bash
just example mlkem public
HEKATE_LEVEL=65 just example mldsa public         # 44 | 65 | 87
HEKATE_LEVEL=256 just example aes public          # 128 | 256
HEKATE_NUM_VARS=20 just example keccak_inline public
HEKATE_NUM_VARS=26 just example fibonacci_raw public

HEKATE_ZK=0 just example keccak_inline public     # base protocol, no hiding
```

### Post-Quantum Crypto and AES

Cells: ZK / base.

|              | ML-KEM-768        | ML-DSA-44         | ML-DSA-65         | ML-DSA-87         | AES-128           | AES-256           |
|:-------------|:------------------|:------------------|:------------------|:------------------|:------------------|:------------------|
| Proving      | 629 / 581 ms      | 883 / 809 ms      | 946 / 849 ms      | 1.34 / 1.24 s     | 1.24 / 1.20 s     | 1.38 / 1.31 s     |
| Verification | 33.7 / 18.8 ms    | 44.8 / 23.6 ms    | 51.1 / 24.2 ms    | 48.1 / 25.3 ms    | 21.0 / 16.4 ms    | 20.6 / 16.0 ms    |
| Proof Size   | 3,483 / 3,103 KiB | 4,184 / 3,689 KiB | 4,193 / 3,706 KiB | 5,484 / 4,909 KiB | 4,692 / 4,390 KiB | 4,982 / 4,677 KiB |
| Peak memory  | 464 / 458 MiB     | 477 / 469 MiB     | 512 / 466 MiB     | 811 / 787 MiB     | 1,217 / 1,185 MiB | 1,508 / 1,465 MiB |
| Chiplets     | 6                 | 7                 | 7                 | 7                 | 2                 | 2                 |

AES note: both AES-128 and AES-256 prove **31,250 blocks** (~500 KB plaintext) per run.
CPU trace 2^16 rows; Round-AIR and S-box ROM chiplets at 2^19. Per-block proving cost:
~42 µs (AES-128) / ~52 µs (AES-256).

### Keccak-f[1600], scaling

`hekate/examples/keccak_inline.rs`, `HEKATE_NUM_VARS` default 20. Cells: ZK / base.

| Scale (rows) | Permutations | Hashed  | Proving       | Verify         | Proof Size        | Peak memory       |
|:-------------|:-------------|:--------|:--------------|:---------------|:------------------|:------------------|
| 2^15         | 1,310        | ~178 KB | 194 / 172 ms  | 11.8 / 4.9 ms  | 864 / 682 KiB     | 156 / 145 MiB     |
| 2^20         | 41,943       | ~5.4 MB | 3.82 / 3.66 s | 20.2 / 9.6 ms  | 3,536 / 3,285 KiB | 2,431 / 2,454 MiB |

### Fibonacci (32-bit integer add), scaling

`hekate/examples/fibonacci_raw.rs`, `HEKATE_NUM_VARS` default 24. Each row: bit-sliced 32-bit add with
explicit carry chain, virtual-expanded into 32 bit + 32 sum + 32 carry columns. Cells: ZK / base.

| Scale (rows) | Proving         | Verify         | Proof Size        | Peak memory        |
|:-------------|:----------------|:---------------|:------------------|:-------------------|
| 2^20         | 329 / 285 ms    | 6.77 / 3.12 ms | 1,145 / 677 KiB   | 263 / 169 MiB      |
| 2^24         | 5.49 / 4.40 s   | 12.8 / 6.93 ms | 4,100 / 2,537 KiB | 3,567 / 2,228 MiB  |
| 2^26         | 25.19 / 20.23 s | 20.5 / 11.7 ms | 8,023 / 5,001 KiB | 14,336 / 8,478 MiB |

---

## Getting Started

- [Installation](https://oumuamua.dev/hekate/docs/getting-started/installation), build from source, configure features
- [Your First ZK Program](https://oumuamua.dev/hekate/docs/getting-started/your-first-zk-program), first proof
  end-to-end
- [Architecture](https://oumuamua.dev/hekate/docs/basics/system-architecture), binary tower fields, Sumcheck, Brakedown,
  LogUp
- [Writing AIR Constraints](https://oumuamua.dev/hekate/docs/basics/air-constraints), constraint DSL, boundary
  conditions
- [Chiplets](https://oumuamua.dev/hekate/docs/basics/cryptographic-chiplets), independent tables, virtual packing, bus
  integration
- [Security](https://oumuamua.dev/hekate/docs/advanced/soundness-and-security), threat model, adversarial test suite,
  Fiat-Shamir binding

---

## License

AGPL-3.0-only. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

Linking against the proprietary prover shared library is covered by the
[prover linking exception](LICENSE-EXCEPTION), an additional permission under AGPL section 7.

Releases up to and including 0.33.0 are Apache-2.0. AGPL-3.0-only applies from 0.34.0 onward.
Commercial licenses are available from Oumuamua Labs <info@oumuamua.dev>.

Hekate does not accept external code contributions. See [CONTRIBUTING](CONTRIBUTING.md).
