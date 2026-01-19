<div align="center">
  <h1>Hekate Engine</h1>
  <h3>The Executioner of Monolithic ZK Architectures.</h3>
</div>

---

### THE MANIFESTO

The Zero-Knowledge industry is built on a lie. It sold scalable trust but delivered infrastructure debt.
It builds server-grade dinosaurs in an era that demands edge-native agility.

Current STARK and Binary-Field provers (Stone/Stwo, Winterfell, Miden, Binius) are architecturally obsolete. They suffer from the "Monolithic Trace Fallacy", the naive belief that the entire history of a computation must be materialized in RAM to be proven.

This architectural failure imposes a heavy "RAM Tax" on every rollup:
* It demands 128GB+ server nodes for real-world workloads.
* It centralizes proving into expensive SaaS silos.
* It makes client-side privacy a physics impossibility.

Hekate Engine is the end of that era. The mistake was not optimized. It was eradicated.

---

### THE WEAPON: EPHEMERAL STATE STREAMING

Hekate rejects history. It proves only the transition.

The proprietary Boundary-State Architecture fundamentally decouples proof capacity from memory capacity. The Engine does not store the execution trace. It generates segments JIT, proves the state transition, and instantly discards the ephemeral data.

The result is absolute linearity. Infinite execution streams proved with flat, O(1) RAM usage.

---

### UNDER THE HOOD (ARCHITECTURE)

Hekate is not a fork. It is a ground-up rewrite of the ZK stack tailored for the "Memory Wall" era.

#### 1. The Physics (Binary Tower Fields)
Instead of large prime fields (which waste bandwidth) or naive binary fields (which lack expressiveness), Hekate operates on **Canonical Binary Tower Fields** (GF(2^128)).
* **Base:** GF(2^8) optimized for AVX/Neon lookups.
* **Tower:** Recursive extension up to 128-bits.
* **Hardware Isomorphism:** On-the-fly basis conversion to utilize native carry-less multiplication (`PMULL` / `clmul`) instructions.

#### 2. The Engine (Streaming Sumcheck & GKR)
Traditional provers (Halo2/Plonky2) materialize the full execution trace ($N \times M$ matrix) to evaluate constraints.
Hekate defines the trace as a **Virtual Polynomial**.
* **JIT Evaluation:** Constraints are evaluated Just-In-Time. The "Trace" exists only ephemerally in CPU registers during the Sumcheck protocol.
* **Lazy Folding:** The memory footprint for the prover is $O(\log N)$ relative to the trace length during the folding phase.

#### 3. The Commitment (Brakedown PCS)
The architecture discards FFT-based commitments (FRI/KZG) in favor of **Linear Codes (Brakedown)**.
* **Complexity:** Strictly linear $O(N)$ prover time.
* **Zero-Copy Merkle:** The commitment phase streams data directly from the generator to the Hasher, bypassing RAM buffers.

#### 4. The Bus (GKR-based GPA)
Chiplets (CPU, RAM, Keccak) are interconnected via a **Grand Product Argument (GPA)** based on GKR layer reduction, avoiding the high degree constraints of standard permutation arguments.

---

### THE KILL SHOT (BENCHMARKS)

Benchmarks conducted on a consumer-grade M3 Max Laptop.

### Binius64

> [!CAUTION]
> **The Workload:**
> Unlike the synthetic Fibonacci test below, this benchmark runs Keccak-f[1600] (Ethereum-native hashing). This is a real-world, heavy-duty cryptographic workload.

![Benchmark Chart](https://github.com/oumuamua-corp/hekate/blob/main/hekate_vs_binius64_keccak_f1600.png?raw=true)

| Metric (Keccak) | Binius64 (Bit-Level Optimization) | **Hekate Engine** (Zero-Copy) | Kill Stats |
| :--- | :--- | :--- | :--- |
| **$2^{15}$ (1.3k Permutations)** | **147 ms** (~400 MB RAM) | 202 ms (**44 MB RAM**) | **~9x Less RAM Overhead** |
| **$2^{20}$ (41k Permutations)** | SWAP HELL (**72 GB RAM**) | **4.74 s** (1.4 GB RAM) | **50x Less RAM** |
| **$2^{24}$ (671k Permutations)** | **CRASH (Out of Memory)** | **88 s** (21.5 GB Peak) | **Operational vs Dead** |

> **Why Hekate Wins (The "Virtual Unpacking" Technique):**
> A standard Keccak trace requires ~1600 bit-columns. Storing this for 2^24 rows consumes massive RAM.
> Hekate stores the state in a packed format (25 `u64` columns) and **virtually unpacks** bits only at the precise moment they are needed by the AIR constraints. The raw bit-trace is never fully materialized in RAM.

### Winterfell

> [!WARNING]
> **The Handicap:**
> * Winterfell: Configured in "Packed" mode (2 terms per row) to hide architectural latency.
> * Hekate: Configured in "Pure" mode (1 term per row). Hekate performs 2x the logical work per row and still dominates.

![Benchmark Chart](https://github.com/oumuamua-corp/hekate/blob/main/hekate_vs_winterfell_fibonacci.png?raw=true)

| Metric | Winterfell (Legacy / Miden Core) | **Hekate Engine** (Gen-5) | Kill Stats |
| :--- | :--- | :--- | :--- |
| **Security Level** | ~99 Bits (Proven <60) | **~166 Bits** (Standard) | **Safer** |
| **$2^{20}$ (1M Rows)** | 792 ms (~1GB RAM) | **397 ms** (154 MB RAM) | **2.0x Faster / 6x Less RAM** |
| **$2^{24}$ (16M Rows)** | 18.34 s (**24 GB RAM**) | **6.16 s** (2.4 GB RAM) | **2.9x Faster / ~9x Less RAM** |
| **$2^{26}$ (67M Rows)** | **CRASH (Out of Memory)** | **26.7 s** (9.8 GB RAM) | **Infinite Scale vs Failure** |
| **$2^{28}$ (268M Rows)** | **CRASH (Impossible)** | **116.9 s** (39.4 GB RAM) | **No comments** |

*Note: Winterfell crashes because it attempts to materialize the entire execution trace + LDE in RAM (O(N log N)). Hekate streams the computation (O(N)), keeping memory usage flat.*

### STATUS: PROPRIETARY / PILOT PROGRAM

Hekate Engine is currently a closed-source, proprietary technology.
Hekate is solving the "Cloud Cost Crisis" for protocols that require heavy client-side proving or massive zkML workloads.

**Looking for pilot partners:**
If you are running ZK bridges, Rollups, or ML inference and are bleeding money on RAM-heavy cloud instances, Hekate can reduce your infrastructure costs by an order of magnitude.

[Email](mailto:zeek@tuta.com) | [LinkedIn](https://www.linkedin.com/in/andrei-kochergin-0966002a3)
