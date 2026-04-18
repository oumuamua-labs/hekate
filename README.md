# Hekate Engine

Zero-knowledge proof system over binary tower fields. Streaming architecture. Bounded memory. Edge-native.

Hekate proves computations in GF(2^128) using Sumcheck + Brakedown PCS with O(N) prover time and O(N) memory. No FFTs,
no trace materialization, no server-grade RAM requirements. Proves ML-KEM decapsulation in 1.6s and ML-DSA signature
verification in 4.7s on a laptop.

---

## Why Hekate Exists

Current ZK provers, RISC Zero, Plonky2, Plonky3, Binius, Stwo, Winterfell, materialize the full execution trace in RAM
before proving. Most then run FFT-based commitments (FRI, Circle FRI) that blow up memory by 2x–8x on top of the trace
with O(N log N) prover time. This "monolithic trace + FFT blowup" architecture imposes a hard floor on memory:
128GB+ for real workloads, 76GB just for Keccak at 2^20 scale (Binius), swap death at 2^24 (Plonky3).

That floor kills client-side proving. No mobile device, no browser, no edge node can run these provers.

Hekate eliminates the floor. The prover streams through the trace, folds in-place, and discards intermediate state. Peak
memory is bounded per-table, not per-computation. A 2^24 Keccak proof runs in 29.7 GB on a consumer laptop where Binius
and Plonky3 crash or thrash.

---

## What It Does

**Binary tower field arithmetic**, GF(2^8) through GF(2^128), recursive tower extension, hardware-accelerated via
PMULL/CLMUL. Constant-time by default.

**Chiplet architecture**, Independent AIR tables (Keccak, AES, RAM, NTT, ML-KEM, ML-DSA) with own traces and
commitments. No column waste, no forced padding. Tables linked by LogUp bus.

**Virtual packing**, Keccak stores 1600 bits in 25 physical B64 columns instead of 1600 bit columns. Bits expand JIT in
registers. 16x memory savings.

**Linear-code commitments**, Brakedown PCS: O(N) prover, O(N) memory. No FFT blowup. Merkle tree over encoded columns
only (raw trace never hashed, true ZK).

**Post-quantum crypto suite**, ML-DSA (Dilithium) signature verification, ML-KEM (Kyber) decapsulation, AES-128/256,
all proven natively in binary fields without bit-decomposition overhead.

---

## Performance

All benchmarks on Apple M3 Max (16 cores, 48 GB RAM), `--release` with LTO.

### Post-Quantum Crypto

|              | ML-KEM-768 | ML-DSA-65 | AES-128   |
|:-------------|:-----------|:----------|:----------|
| Proving      | 1.61 s     | 4.68 s    | 1.94 s    |
| Verification | 25.4 ms    | 60.3 ms   | 23.1 ms   |
| Proof Size   | 4,089 KiB  | 5,019 KiB | 3,325 KiB |
| Peak Memory  | 338 MB     | 291 MB    | 717 MB    |
| Chiplets     | 6          | 7         | 2         |

### Keccak-f[1600], Memory Scaling

| Scale             | Binius64         | Plonky3             | Hekate                |
|:------------------|:-----------------|:--------------------|:----------------------|
| 2^15 (1.3K perms) | 253 ms / 4.33 GB | 500 ms / 753 MB     | 1.48 s / **139 MB**   |
| 2^20 (41K perms)  | CRASH (76+ GB)   | 13.0 s / 22.8 GB    | 21.2 s / **2.64 GB**  |
| 2^24 (671K perms) | CRASH            | SWAP DEATH (164 GB) | 364.6 s / **29.7 GB** |

### Fibonacci, Pure Algebra

| Scale | Stwo                | Winterfell     | Hekate               |
|:------|:--------------------|:---------------|:---------------------|
| 2^20  | 480 ms / 1.2 GB     | 792 ms / ~1 GB | **144 ms / 167 MB**  |
| 2^24  | 6.9 s / 16.6 GB     | 18.3 s / 24 GB | **1.85 s / 2.6 GB**  |
| 2^26  | 46.3 s / 20 GB swap | CRASH          | **8.23 s / 10.4 GB** |

---

## Architecture at a Glance

```
hekate-math          Binary tower field arithmetic (external, sealed)
  ↓
hekate-crypto        Hash primitives (Blake3, Groestl, SHA-256)
  ↓
hekate-core          Merkle trees, Fiat-Shamir transcript, polynomial representations, trace memory
  ↓
hekate-program       AIR definition API, constraint DSL, chiplet composition
  ↓
hekate-prover        Streaming prover: Sumcheck, Brakedown, LogUp
  ↓
hekate-verifier      Analytical verifier: Fiat-Shamir replay, Sumcheck checks
  ↓
hekate-chiplets      Keccak, AES, RAM, ROM, NTT, ML-KEM, ML-DSA chiplets
```

---

## Quick Example

Minimal Fibonacci program, column schema, AIR constraints, and trace generation:

```rust
type F = Block128;

define_columns! {
    FibColumns {
        A: B32,
        B: B32,
        Q: Bit,
    }
}

#[derive(Clone)]
struct FibProgram {
    num_rows: usize,
}

impl Air<F> for FibProgram {
    fn num_columns(&self) -> usize {
        FibColumns::NUM_COLUMNS
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(FibColumns::build_layout)
    }

    fn boundary_constraints(&self) -> Vec<BoundaryConstraint> {
        vec![BoundaryConstraint::new(FibColumns::B, self.num_rows - 1, 0)]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::new();

        let [a, b, q] = [
            cs.col(FibColumns::A),
            cs.col(FibColumns::B),
            cs.col(FibColumns::Q),
        ];
        let [na, nb] = [cs.next(FibColumns::A), cs.next(FibColumns::B)];

        cs.constrain(q * (na + b));
        cs.constrain(q * (nb + a + b));

        cs.build()
    }
}

impl Program<F> for FibProgram {
    fn num_public_inputs(&self) -> usize { 1 }
}
```

Trace generation, fill columns row by row via `TraceBuilder`:

```rust
fn generate_trace(num_vars: usize) -> ColumnTrace {
    let num_rows = 1 << num_vars;

    let mut tb = TraceBuilder::new(&FibColumns::build_layout(), num_vars).unwrap();

    let (mut a, mut b) = (Block32::ZERO, Block32::ONE);
    for i in 0..num_rows {
        tb.set_b32(FibColumns::A, i, a).unwrap();
        tb.set_b32(FibColumns::B, i, b).unwrap();

        let temp = a + b;
        a = b;
        b = temp;
    }

    tb.fill_selector(FibColumns::Q, num_rows - 1).unwrap();

    tb.build()
}
```

See [Getting Started](getting-started.md) for the full walkthrough including prover and verifier calls.

---

## Hardware Support

| Architecture | Status      | Instructions                          |
|:-------------|:------------|:--------------------------------------|
| aarch64      | Production  | PMULL, NEON                           |
| x86_64       | Development | Software fallback (PCLMULQDQ roadmap) |
| WASM         | Fallback    | Software multiply                     |

---

## Next Steps

- [Installation](installation.md), build from source, configure features
- [Getting Started](getting-started.md), first proof end-to-end
- [Architecture](architecture.md), binary tower fields, Sumcheck, Brakedown, LogUp
- [Writing AIR Constraints](air-constraints.md), constraint DSL, boundary conditions
- [Chiplets](chiplets.md), independent tables, virtual packing, bus integration
- [Security](security.md), threat model, adversarial test suite, Fiat-Shamir binding

---

## Status

Hekate verifier, core SDK, and chiplets are being open-sourced. The prover and recursive engine remain closed-source,
licensed as proprietary binaries.

## Contact

[info@oumuamua.dev](mailto:info@oumuamua.dev)
