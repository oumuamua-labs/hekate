<div align="center">
  <h1>Hekate Engine</h1>
  <h3>The Executioner of Monolithic ZK Architectures.</h3>
</div>

---

### THE MANIFESTO

The Zero-Knowledge industry is built on a lie. It sold scalable trust but delivered infrastructure debt. It builds
server-grade dinosaurs in an era that demands edge-native agility.

Current STARK and Binary-Field provers (Stone/Stwo, Winterfell, Plonky3, Binius) are architecturally obsolete. They
suffer from the "Monolithic Trace Fallacy", the naive belief that the entire history of a computation must be
materialized in RAM to be proven. They aggressively trade memory for micro-scale speed.

This architectural failure imposes a heavy "RAM Tax" on every rollup:

* It demands 128GB+ server nodes for real-world workloads.
* It centralizes proving into expensive SaaS silos.
* It makes client-side Edge Proving (mobile/browser) a physics impossibility.

Hekate Engine is the end of that era. We didn't optimize the mistake. We eradicated it.

---

### THE WEAPON: VIRTUAL PACKING & EPHEMERAL STATE

Hekate rejects history. It proves only the transition.

The Boundary-State Architecture decouples proof capacity from memory capacity. The engine stores no full execution
trace. Virtual Packing compresses the physical footprint, segments are generated JIT, the state transition is proved,
and the ephemeral data is discarded.

Absolute linearity. Infinite execution streams proved with flat, strictly bounded RAM.

---

### UNDER THE HOOD (ARCHITECTURE)

Hekate is not a fork. Ground-up rewrite of the ZK stack built to shatter the Memory Wall.

#### 1. The Physics (Binary Tower Fields)

Hekate operates on **Canonical Binary Tower Fields** (GF(2^128)), not large prime fields (bandwidth waste, memory
thrash) or naive binary fields (lack expressiveness).

* **Base:** $GF(2^8)$ optimized for AVX/Neon lookups.
* **Tower:** Recursive extension up to 128-bits.
* **Hardware Isomorphism:** On-the-fly basis conversion to utilize native carry-less multiplication (`PMULL` / `clmul`)
  instructions.

#### 2. The Protocol (Sumcheck + LogUp)

Hekate's proving protocol is built on two primitives that make monolithic trace materialization unnecessary.

**Sumcheck** reduces a multivariate polynomial claim over $2^n$ points to a single evaluation, one variable at a time,
$n$ rounds, $O(N)$ total work. No FFTs. No $O(N \log N)$ blowup. The prover streams through the trace once per round
and folds in-place. Memory stays flat regardless of circuit size.

**LogUp** connects independent tables through a char-2 fractional sumcheck. Each chiplet (Keccak, AES, RAM, NTT) runs as
an isolated sub-proof with its own trace and commitment. No shared column layout, no forced padding to the tallest
table's height. A Keccak chiplet at $2^{17}$ rows doesn't force a RAM chiplet at $2^{14}$ rows to waste $8\times$ its
memory on zero-padding. Tables are linked by a cryptographic bus, matched multisets cancel pairwise in $GF(2^{128})$.

This is why Hekate scales linearly while monolithic provers hit the Memory Wall: the work is $O(N)$, the memory is
bounded per-table, and nothing is materialized that isn't immediately consumed.

#### 3. Virtual Unpacking

A standard Keccak trace requires ~2633 materialized physical columns in Plonky3 or Binius. Hekate stores the state in
**54 physical columns** and **virtually unpacks** bits only when the AIR constraints need them. The full bit-trace never
exists in RAM.

#### 4. The Commitment (Univariate Brakedown PCS)

**Linear Codes (Brakedown)** replace $O(N \log N)$ FFT-based commitments (Circle FRI/KZG) and heavy Multilinear
Hypercubes.

* **Complexity:** Strictly linear $O(N)$ prover time and memory.
* **Zero-Copy Merkle:** Commitment streams data directly from generator to hasher, bypassing RAM buffers.

---

### THE POST-QUANTUM ZK CRYPTO SUITE

Most ZK frameworks are built for server farms. Hekate is built for the edge. The composite chiplet architecture proves
this with the industry's first Post-Quantum ZK Crypto Suite. Native binary tower field execution eliminates the
bit-decomposition overhead that plagues prime field frameworks.

* **ML-DSA (Dilithium):** Verifying post-quantum digital signatures (pk, sig, msg) natively inside the ZK circuit. The
  exact primitive L2s need to build fully quantum-resistant Rollups.
* **ML-KEM (Kyber):** Proving a secure payload was successfully decrypted without ever exposing the private key to the
  public network.
* **AES-128/256:** Processing payloads with pure mathematical throughput on binary fields, ready to be wired directly
  into the ML-KEM decapsulation loop.

**Benchmark Results**
| Metric | ML-KEM-768 | ML-DSA-65 | AES-128 |
| :--- | :--- | :--- | :--- |
| **Trace Gen** | 92.8 ms | 228.4 ms | 309.2 ms |
| **Proving** | **1.61 s** | **4.68 s** | **1.94 s** |
| **Verification** | 25.4 ms | 60.3 ms | 23.1 ms |
| **Proof Size** | 4,089 KiB | 5,019 KiB | 3,325 KiB |
| **Max RSS** | 497 MB | 574 MB | 1.01 GB |
| **Peak Footprint** | **338 MB** | **291 MB** | **717 MB** |
| **Main Trace Rows** | 65,536 | 65,536 | 65,536 |
| **Chiplets** | 6 | 7 | 2 |
| **Largest Chiplet** | MlKemCtrl (64K) | NttChiplet (64K) | AesRound128 (512K) |

> *Note: ML-DSA is the heaviest: 7 chiplets, NTT at 64K rows + Keccak at 8K rows driving the 4.68s proving time. AES has
the highest raw row count (524K total) but only 2 chiplets, keeping proving under 2s. ML-KEM is the leanest at 1.61s.*

**AES-128 vs AES-256 (31,250 Blocks Proved)**
| Metric | AES-128 | AES-256 |
| :--- | :--- | :--- |
| **Trace Gen** | 309.2 ms | 476.6 ms |
| **Proving** | 1.94 s | 2.12 s |
| **Verification** | 23.1 ms | 23.5 ms |
| **Proof Size** | 3,325 KiB | 3,626 KiB |
| **Max RSS** | 1.01 GB | 1.28 GB |
| **Chiplet Rows** | 524K | 524K |

> *Note: Both use the exact same trace dimensions. AES-256 adds only ~9% proving overhead and ~9% proof size from the
extra key schedule columns. Verification remains nearly identical. Clean scaling.*

---

### BENCHMARKS

Benchmarks conducted on a consumer-grade Apple M3 Max Laptop (16 cores, 48 GB RAM).

#### 1. Keccak-f[1600] (Heavy-Duty Cryptography)

*Real-world Ethereum-native hashing. This tests trace width, bitwise operations, and physical memory scaling.*

> [!CAUTION]
> **The Server-Side Illusion**
> Binius and Plonky3 demonstrate impressive speed on micro-domains (2^15). However, scaling to real-world workloads
> shatters the illusion. Binius crashes at 76GB of RAM as early as the 2^20 scale, and Plonky3 thrashes the OS with 164GB
> of Swap Virtual Memory at the 2^24 scale, dropping CPU utilization to 10%. Hekate remains fully compute-bound (98% CPU)
> and survives both workloads.

**Hekate vs Binius64 (Multilinear Binary)**
| Metric (Keccak) | Binius64 (Multilinear Tower) | **Hekate Engine** (Univariate Binary) | Edge Proving Verdict |
| :--- | :--- | :--- | :--- |
| $2^{15}$ (1.3k Perms) | **253 ms** (4.33 GB RAM) | 1.48 s (**139 MB RAM**) | Hekate uses ~31x Less RAM |
| $2^{20}$ (41k Perms) | CRASH (**76.0+ GB RAM**) | **21.2 s** (2.64 GB RAM) | Hekate survives Swap Death |
| $2^{24}$ (671k Perms) | **CRASH (Impossible)** | **364.6 s** (29.7 GB RAM) | Operational vs Dead |

**Hekate vs Plonky3 (Circle M31)**
| Metric (Keccak) | Plonky3 (Circle FRI / M31) | **Hekate Engine** (Univariate Binary) | Edge Proving Verdict |
| :--- | :--- | :--- | :--- |
| $2^{15}$ (1.3k Perms) | **500 ms** (753 MB RAM) | 1.48 s (**139 MB RAM**) | Hekate uses 5.4x Less RAM |
| $2^{20}$ (41k Perms) | 13.0 s (22.8 GB RAM) | 21.2 s (**2.64 GB RAM**) | Hekate uses 8.6x Less RAM |
| $2^{24}$ (671k Perms) | **SWAP DEATH (164.6 GB VRAM)** | **364.6 s** (**29.7 GB RAM**) | Server vs Edge Paradigm |

#### 2. Fibonacci (Pure Algebra & Scaling Limits)

*A pure prime-field home ground test to isolate the memory overhead of the polynomial commitment schemes. Evaluates raw
scaling capability without bitwise penalties.*

**Hekate vs Stwo (Modern Circle STARK)**
> Both engines compute a 2-column pure Fibonacci sequence. This exposes the raw memory overhead of Circle FRI (O(N log
> N)) against Hekate's Tower Brakedown (O(N)).

| Metric              | Stwo (Circle M31)                | **Hekate Engine** (Binary Brakedown) | Stats                      |
|:--------------------|:---------------------------------|:-------------------------------------|:--------------------------------|
| 2^20 (1M Rows)  | 480 ms (1.2 GB RAM)              | **144 ms** (167 MB RAM)              | 3.3x Faster / 7.1x Less RAM |
| 2^24 (16M Rows) | 6.90 s (16.6 GB RAM)             | **1.85 s** (2.6 GB RAM)              | 3.7x Faster / 6.3x Less RAM |
| 2^26 (67M Rows) | 46.3 s (**20.0 GB VRAM / Swap**) | **8.23 s** (10.4 GB RAM)             | 5.6x Faster / ~2x Less RAM  |

*Note: At 2^26 rows, Stwo hits the physical memory limits of consumer hardware, severely bottlenecking its performance
due to swap pressure, while Hekate's linear commitment easily fits into standard RAM.*

**Hekate vs Winterfell (Legacy STARK)**
> [!WARNING]
> **The Architectural Handicap:**
> * Winterfell: Configured in "Packed" mode (2 terms per row) to artificially halve the trace length and hide
    architectural latency.
> * Hekate: Configured in "Pure" mode (1 term per row). Hekate performs 2x the logical work per row and still
    dominates.

| Metric               | Winterfell (Legacy / Prime Field) | **Hekate Engine** (Gen-5 Binary) | Stats                      |
|:---------------------|:----------------------------------|:---------------------------------|:--------------------------------|
| **Security Level**   | ~99 Bits (Proven <60)             | **~128 Bits** (Standard)         | **Safer**                       |
| 2^20 (1M Rows)   | 792 ms (~1GB RAM)                 | **144 ms** (167 MB RAM)          | 5.5x Faster / 6x Less RAM   |
| 2^24 (16M Rows)  | 18.34 s (**24 GB RAM**)           | **1.85 s** (2.6 GB RAM)          | 9.9x Faster / 9.2x Less RAM |
| 2^26 (67M Rows)  | **CRASH (Out of Memory)**         | **8.23 s** (10.4 GB RAM)         | Infinite Scale vs Failure   |
| 2^28 (268M Rows) | **CRASH (Impossible)**            | **45.6 s** (41.8 GB RAM)         | No comments                  |

---

### THE FUTURE: RECURSION & OPEN SOURCE

**Next up: The Recursive Verifier**

The recursion layer compresses 3-5MB proofs down to < 800KB. Built on the same chiplet architecture. In progress.

**Open Source Roadmap**

We will be open-sourcing the Hekate Verifier, Core SDK, and Chiplets soon.

*The Prover and Recursive engine remain closed-source, licensed as proprietary binaries.*

---

### STATUS: PROPRIETARY / PILOT PROGRAM

Hekate solves the Cloud Cost Crisis for protocols that require heavy client-side proving or massive zkML workloads.

End-game: ePassport verification, ZK-KYC, Dark Forest DEX transactions, running natively on mobile devices without OOM
crashes or battery drain.

**Pilot partners wanted.**

ZK bridges, Rollups, ML inference, if you're bleeding money on RAM-heavy cloud instances, Hekate cuts infrastructure
costs by an order of magnitude.

[Email](mailto:zeek@tuta.com) | [LinkedIn](https://www.linkedin.com/in/andrei-kochergin-0966002a3)
