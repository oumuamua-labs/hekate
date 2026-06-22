# Hekate ZK Engine

[![CI](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml/badge.svg)](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache2-yellow.svg)](./LICENSE)

Zero-knowledge proof system over binary tower fields. Streaming architecture. Bounded memory. Edge-native.

Hekate proves computations in GF(2^128) using Sumcheck + Brakedown PCS with O(N) prover time and O(N) memory. No FFTs,
no trace materialization, no server-grade RAM requirements. Proves ML-KEM decapsulation and ML-DSA signature
verification on a laptop and mobile.

> [!WARNING]  
> This workspace is under aggressive development. APIs, ABIs, and cryptographic signatures will break
> without notice. Do not deploy to mainnet.

> [!NOTE]  
> The verifier, core SDK, and cryptographic chiplets are open-source. The prover and
> compression engine stay proprietary, shipped as free, unrestricted binaries for macOS (Apple Silicon),
> Linux (ARM64, glibc), and Android (ARM64).

> [!IMPORTANT]  
> [`hekate-mobile`](https://github.com/oumuamua-labs/hekate-mobile) compiles a Rust prover into a signed
> iOS `.xcframework` and Android `.aar` behind a typed Swift / Kotlin API, one `await` per proof, zero ZK
> terminology across the boundary. Shipping ZK to edge devices? Start there.

---

## ⚠️ Security Warning

This crate has not been audited and may contain bugs and security flaws.

USE AT YOUR OWN RISK!

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

## The Hekate Ecosystem

Hekate is not one crate. The prover is the engine; the math core, hardware chiplets, mobile toolchain,
and fuzzer ship as independent crates you compose as needed.

| Crate                                                                                      | Role                                                                                                        |
|:-------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------|
| [`hekate-math`](https://github.com/oumuamua-labs/hekate-math)                              | Binary tower field arithmetic, constant-time, PMULL / PCLMULQDQ. The mathematical core.                     |
| [`hekate-prover-sys`](https://github.com/oumuamua-labs/hekate/tree/main/hekate-prover-sys) | Open FFI shim. Links the signed prover cdylib over a stable C ABI; the only crate that can call the prover. |
| [`hekate-keccak`](https://github.com/oumuamua-labs/hekate-keccak)                          | Keccak-f[1600] chiplet plus SHA-3 / SHAKE. Virtual packing, ~16x memory savings.                            |
| [`hekate-aes`](https://github.com/oumuamua-labs/hekate-aes)                                | AES-128 / AES-256 round-function chiplet (FIPS 197) with an S-box ROM.                                      |
| [`hekate-pqc`](https://github.com/oumuamua-labs/hekate-pqc)                                | ML-KEM decapsulation and ML-DSA verification (FIPS 203 / 204), with NTT, basemul, norm-check.               |
| [`hekate-mobile`](https://github.com/oumuamua-labs/hekate-mobile)                          | Wraps a Rust prover into a signed iOS `.xcframework` / Android `.aar` with a typed Swift / Kotlin API.      |
| [`zk-scribble`](https://github.com/yoozzeek/zk-scribble)                                   | Community trace-mutation fuzzer. Tampers a valid trace, panics if your constraints miss the tamper.         |

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

**Linear-code commitments**, Brakedown PCS: O(N) prover, O(N) memory. No FFT blowup. Merkle tree over encoded columns
only (raw trace never hashed, true ZK).

**Post-quantum crypto suite**, ML-DSA (Dilithium) signature verification, ML-KEM (Kyber) decapsulation, AES-128/256,
all proven natively in binary fields without bit-decomposition overhead.

---

## Quick Example

Real 32-bit-integer Fibonacci. The CPU side holds five columns and the two Fibonacci transition
constraints. Every `u32` ADD is offloaded to the `IntArithmeticChiplet`, its own trace, own
commitment, own ZeroCheck, own evaluation argument, and is wired in by a LogUp bus
(`(val_a, val_b, val_res, opcode, request_idx)` keys with a row-index clock).

```rust
type F = Block128;

#[derive(Clone)]
struct FibProgram {
    num_rows: usize,
}

impl Air<F> for FibProgram {
    fn num_columns(&self) -> usize {
        CpuArithColumns::NUM_COLUMNS
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(CpuArithColumns::build_layout)
    }

    fn boundary_constraints(&self) -> Vec<BoundaryConstraint<F>> {
        vec![BoundaryConstraint::with_public_input(
            CpuArithColumns::VAL_B,
            self.num_rows - 1,
            0,
        )]
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![(
            IntArithmeticChiplet::BUS_ID.into(),
            CpuIntArithmeticUnit::linking_spec(),
        )]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();

        let s = cs.col(CpuArithColumns::SELECTOR);
        let val_b = cs.col(CpuArithColumns::VAL_B);
        let val_res = cs.col(CpuArithColumns::VAL_RES);
        let next_a = cs.next(CpuArithColumns::VAL_A);
        let next_b = cs.next(CpuArithColumns::VAL_B);

        cs.assert_boolean(s);
        cs.constrain(s * (next_a + val_b));     // next_a = b
        cs.constrain(s * (next_b + val_res));   // next_b = a + b (chiplet provides val_res)

        cs.build()
    }
}

impl Program<F> for FibProgram {
    fn num_public_inputs(&self) -> usize { 1 }

    fn chiplet_defs(&self) -> errors::Result<Vec<ChipletDef<F>>> {
        let arith = IntArithmeticChiplet::new(32, self.num_rows)?;
        Ok(vec![ChipletDef::from_air(&arith)?])
    }
}
```

Trace generation builds the CPU columns and the chiplet trace independently; they meet on the bus.

```rust
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

    // Padding row: selector = 0, val_b carries fib[N-1] for the boundary check.
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
`hekate-verifier`. The transcript label, `Config`, and matrix seed must match across both sides, the
driver builds one `config` and reuses it. `verify` returns `true` only if every Sumcheck round, the
LogUp bus sums, and the evaluation openings hold.

```rust
fn run(num_rows: usize) -> errors::Result<bool> {
    let (cpu, arith, fib_n) = generate_traces(num_rows)?;

    let program = FibProgram { num_rows };
    let instance = ProgramInstance::new(num_rows, vec![F::from(fib_n as u128)]);
    let witness = ProgramWitness::new(cpu).with_chiplets(vec![arith]);

    let mut config = Config::default();
    OsRng.try_fill_bytes(&mut config.matrix_seed).unwrap();

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

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

    HekateVerifier::<F, DefaultHasher>::verify(
        &program,
        &instance,
        &proof,
        &mut transcript,
        &config,
    )
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
  mul via `IntArithmeticChiplet`)
- [RAM read/write proof](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/ram.rs) (offline-memory
  consistency via `RamChiplet`)

---

## Performance

All numbers on Apple M3 Max (16 cores, 48 GB RAM), `--release` with `-C target-cpu=native`,
features `std parallel blake3 table-math`. Measured on commit `master` with the example binaries
in `hekate/examples/`. Peak / total heap via `dhat-heap`.

Reproduce:

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release \
  --no-default-features --features "std parallel blake3 table-math" \
  --example <name> [-- <arg>]
```

### Post-Quantum Crypto and AES

|              | ML-KEM-768 | ML-DSA-44 | ML-DSA-65 | ML-DSA-87 | AES-128   | AES-256   |
|:-------------|:-----------|:----------|:----------|:----------|:----------|:----------|
| Proving      | 1.40 s     | 2.43 s    | 2.54 s    | 3.98 s    | 2.15 s    | 2.27 s    |
| Verification | 30.6 ms    | 69.0 ms   | 70.7 ms   | 115.6 ms  | 24.5 ms   | 25.9 ms   |
| Proof Size   | 4,232 KiB  | 5,139 KiB | 5,156 KiB | 8,620 KiB | 3,405 KiB | 3,706 KiB |
| Peak Heap    | 331 MB     | 294 MB    | 294 MB    | 580 MB    | 772 MB    | 1,005 MB  |
| Total Alloc  | 1.58 GB    | 3.75 GB   | 3.76 GB   | 7.28 GB   | 2.05 GB   | 2.40 GB   |
| Chiplets     | 6          | 7         | 7         | 7         | 2         | 2         |

Chiplet trace sizes:

- ML-KEM-768: Ctrl 2^16, Keccak 2^11, NTT 2^15, TwiddleROM 2^15, Basemul 2^12, RAM 2^16.
- ML-DSA-44 / ML-DSA-65: Ctrl 2^16, Keccak 2^13, NTT 2^16, TwiddleROM 2^16, NormCheck 2^11, HighBits 2^11, RAM 2^16.
- ML-DSA-87 doubles Ctrl and Keccak: 2^17 / 2^14.

AES note: both AES-128 and AES-256 prove **31,250 blocks** (~500 KB plaintext) per run.
CPU trace 2^16 rows; Round-AIR and S-box ROM chiplets at 2^19. Per-block proving cost: ~69 µs (AES-128) / ~73 µs (
AES-256).

### Keccak-f[1600], scaling

`hekate/examples/keccak_inline.rs <num_vars>`, default 20.

| Scale (rows) | Permutations | Hashed  | Proving  | Verify   | Proof Size | Peak Heap | Total Alloc |
|:-------------|:-------------|:--------|:---------|:---------|:-----------|:----------|:------------|
| 2^15         | 1,310        | ~178 KB | 919 ms   | 23.3 ms  | 1,312 KiB  | 92 MB     | 255 MB      |
| 2^20         | 41,943       | ~5.4 MB | 14.16 s  | 87.0 ms  | 5,156 KiB  | 2,278 MB  | 3,747 MB    |
| 2^24         | 671,088      | ~91 MB  | 268.08 s | 333.9 ms | 20,209 KiB | 31,088 MB | 51,535 MB   |

### Fibonacci (32-bit integer add), scaling

`hekate/examples/fibonacci_raw.rs <num_vars>`, default 24. Each row: bit-sliced 32-bit add with
explicit carry chain, virtual-expanded into 32 bit + 32 sum + 32 carry columns.

| Scale (rows) | Proving | Verify  | Proof Size | Peak Heap | Total Alloc |
|:-------------|:--------|:--------|:-----------|:----------|:------------|
| 2^20         | 745 ms  | 10.1 ms | 1,125 KiB  | 209 MB    | 361 MB      |
| 2^24         | 11.30 s | 36.9 ms | 4,237 KiB  | 3,077 MB  | 5,210 MB    |
| 2^26         | 47.20 s | 76.1 ms | 8,378 KiB  | 12,072 MB | 20,486 MB   |

---

## Hardware Support

| Architecture | Status      | Instructions                          |
|:-------------|:------------|:--------------------------------------|
| aarch64      | Production  | PMULL, NEON                           |
| x86_64       | Development | Software fallback (PCLMULQDQ roadmap) |
| WASM         | Fallback    | Software multiply                     |

---

## Next Steps

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

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).