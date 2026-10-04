# hekate-verifier

[![Crates.io](https://img.shields.io/crates/v/hekate-verifier.svg)](https://crates.io/crates/hekate-verifier)
[![Docs.rs](https://docs.rs/hekate-verifier/badge.svg)](https://docs.rs/hekate-verifier)
[![CI](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml/badge.svg)](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml)
[![License: AGPL-3.0-only](https://img.shields.io/badge/License-AGPL--3.0--only-blue.svg)](LICENSE)

*Copyright (c) 2026 Andrei Kochergin and Oumuamua Labs.*

Verifier for the [Hekate ZK Engine](https://github.com/oumuamua-labs/hekate). It checks a proof that a
program's constraints, pinned cells and bus relations hold over a committed trace and its chiplet
tables, given the program, its public inputs and the proof bytes. Proofs are zero-knowledge by default
and can be made in the clear; this crate verifies both. It runs no prover code and takes nothing from
the prover on trust.

---

## ⚠️ Security Warning

This crate has not been independently audited and may contain bugs and security flaws.

USE AT YOUR OWN RISK!

---

## Trust model

The prover is an untrusted party, and this crate is where acceptance is decided. A proof is accepted
only if every check below passes, and no check depends on who produced the proof or how.

```
 your build (trusted)                          prover (untrusted)
 ──────────────────────────────────            ──────────────────
 Program, built from audited source            witness
 program_id, pinned at audit time                 │
 Config and transcript label                      ▼
 instance: trace height, public inputs         proof bytes
                  │                               │
                  └──────►  HekateVerifier  ◄─────┘
                                  │
                   Ok(true) · Ok(false) · Err(_)
```

Only bytes cross from the prover's side. The shipped prover is a closed-source binary. A modified build
of it, another implementation or bytes written by hand meet the same checks, and soundness holds
against each of them. Every check a valid proof must pass lives in this crate and the open crates it
builds on. To trust a Hekate proof, audit this side.

**What you pin.** Your code supplies the four inputs on the left and fixes them before any proof arrives.

- `Program` is the statement. Build it from the same audited source the prover used.
- `program_id` is the audited program's id, printed once with
  `hekate_program::digest::program_id_hex` and compiled in as a constant. The verifier recomputes the
  id from the program object it receives and returns `Err(ProgramIdMismatch)` on a difference, which
  catches a verifying build whose program drifted from the audited one. The id never travels in the
  proof.
- `Config` and the transcript label must equal the prover's. The label and the config's proof
  parameters enter the transcript, and a mismatch rejects. `Config::prod()` is the default;
  `Config::dev()` accepts weak geometries and exists for tests.
- The instance carries the trace height and the public inputs the statement is about.

**Reading the result.** `Ok(true)` accepts. `Ok(false)` means a check failed. `Err` means a
malformed proof or input, a program id mismatch, or a configuration below its security floor. Treat
everything except `Ok(true)` as a rejection.

---

## How a proof is checked

The verifier replays the prover's Fiat-Shamir transcript from the proof bytes. Each challenge is
drawn after the values it depends on are absorbed, in the order below, and a proof computed against
any other order meets different challenges and fails.

```
stage         the verifier
────────────  ──────────────────────────────────────────────────────────────────
shape         trace heights, counts, the security floor of the configuration
identity      prepared program id against the pinned id
bind          absorb program id, configuration, sizes, public inputs, trace root,
              boundary pins and every chiplet header; in ZK mode the pad root
buses         draw γ and β, and one r_bus per lookup bus
each table    absorb the bus helper root and the claimed bus sums, draw α
(chiplets,    replay the ZeroCheck rounds, degree-checked, down to r_final
then main)    check constraints, boundary pins, fixed columns and bus terms
              at r_final; in ZK mode record them for the outer argument
              absorb the claimed column values, draw η, replay the
              evaluation sumcheck
              draw query columns, check the Merkle openings and the
              proximity of the opened columns to the code
close         base mode: the claimed bus sums cancel for every bus id
              ZK mode: the outer argument over the masked scalars
```

ZeroCheck turns "every constraint vanishes on every row" into one claim at a random point
`r_final`. The evaluation argument ties the column values claimed at `r_final` to the committed
trace: each table is committed under a Reed-Solomon row code and a Merkle tree, and the verifier
opens the queried columns and checks them against the claims. The LogUp helper column `h` of a table
with buses gets the same commitment and opening, which binds the bus sums to the trace.

With `zero_knowledge` on, the default, every scalar the proof exposes is one-time padded: round
evaluations, claimed bus sums and column claims. The replay runs on the masked values, and each
check that base mode runs in the clear becomes a row of an outer statement over them, proven at the
end by a zk-Ligero argument.

---

## Usage

Proof bytes decode with `hekate_sdk::deserialize_proof`. The transcript label and the `Config` must be
the ones the prover used. An end-to-end example that proves with `hekate-prover-sys` and verifies with
this crate is the [Quick Example](https://github.com/oumuamua-labs/hekate#quick-example) of the
workspace README.

| Entry point       | For                                  | Each call                                    |
|:------------------|:-------------------------------------|:---------------------------------------------|
| `verify`          | one proof                            | prepares the program, recomputing its id     |
| `verify_prepared` | a long-lived verifier of one program | replays the proof against a prepared program |
| `verify_batch`    | many proofs of one program at once   | one proof per pool thread, results in order  |

### One proof

```rust
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_crypto::DefaultHasher;
use hekate_crypto::transcript::Transcript;
use hekate_math::Block128;
use hekate_program::{Program, ProgramInstance};
use hekate_sdk::deserialize_proof;
use hekate_verifier::HekateVerifier;

type F = Block128;

/// `audited_id` is a constant of this build, printed
/// once with `digest::program_id_hex` at audit time.
fn verify_one<P: Program<F> + Sync>(
    audited_id: &[u8; 32],
    program: &P,
    instance: &ProgramInstance<F>,
    proof_bytes: &[u8],
) -> errors::Result<bool> {
    let proof = deserialize_proof::<F>(proof_bytes)?;

    HekateVerifier::<F, DefaultHasher>::verify(
        audited_id,
        program,
        instance,
        &proof,
        &mut Transcript::new(b"my-app"),
        &Config::prod(),
    )
}
```

### Many proofs of one program

`PreparedProgram` holds what verifying one program needs before any proof: its tables, ring-switch
plans, shapes and id. It owns that data and is read-only, which lets one copy serve every thread.
`VerifierScratch` caches state across verifies and is mutable: one per thread.

```rust
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_crypto::DefaultHasher;
use hekate_crypto::transcript::Transcript;
use hekate_math::Block128;
use hekate_program::{Program, ProgramInstance};
use hekate_sdk::deserialize_proof;
use hekate_verifier::HekateVerifier;
use hekate_verifier::prepared::{PreparedProgram, VerifierScratch};

type F = Block128;

struct Checker {
    audited_id: [u8; 32],
    prepared: PreparedProgram<F>,
    scratch: VerifierScratch<F>,
}

impl Checker {
    fn new<P: Program<F>>(audited_id: [u8; 32], program: &P) -> errors::Result<Self> {
        Ok(Self {
            audited_id,
            prepared: PreparedProgram::new(program, &Config::prod())?,
            scratch: VerifierScratch::new(),
        })
    }

    fn verify(
        &mut self,
        instance: &ProgramInstance<F>,
        proof_bytes: &[u8],
    ) -> errors::Result<bool> {
        let proof = deserialize_proof::<F>(proof_bytes)?;

        HekateVerifier::<F, DefaultHasher>::verify_prepared(
            &self.audited_id,
            &self.prepared,
            instance,
            &proof,
            &mut Transcript::new(b"my-app"),
            &mut self.scratch,
        )
    }
}
```

### A batch

`verify_batch` opens a transcript with the given label for each proof. With `parallel` it runs one
proof per pool thread, each with its own scratch, and returns the results in input order.

```rust
use hekate_core::errors;
use hekate_core::proofs::InnerProof;
use hekate_crypto::DefaultHasher;
use hekate_math::Block128;
use hekate_program::ProgramInstance;
use hekate_verifier::HekateVerifier;
use hekate_verifier::prepared::PreparedProgram;

type F = Block128;

fn verify_all(
    audited_id: &[u8; 32],
    prepared: &PreparedProgram<F>,
    proofs: &[(ProgramInstance<F>, InnerProof<F>)],
) -> Vec<errors::Result<bool>> {
    let items: Vec<_> = proofs
        .iter()
        .map(|(instance, proof)| (instance, proof))
        .collect();

    HekateVerifier::<F, DefaultHasher>::verify_batch(audited_id, prepared, b"my-app", &items)
}
```

---

## Features

| Feature            | Default | Effect                                                                             |
|:-------------------|:--------|:-----------------------------------------------------------------------------------|
| `std`              | yes     | Standard library. Without it the crate is `no_std` with `alloc`.                   |
| `blake3`           | yes     | BLAKE3 transcript and Merkle hashing.                                              |
| `sha2`, `sha3`     | no      | SHA-256 or SHA3-256 hashing. The prover must use the same hash.                    |
| `parallel`         | yes     | Rayon pool for leaf hashing, proximity checks, the outer tests and `verify_batch`. |
| `table-math`       | no      | Variable-time basis conversion, faster. See below.                                 |
| `transcript-trace` | no      | Records every transcript operation, to locate a prover/verifier divergence.        |

Every value the verifier touches is public, which makes `table-math` safe for verification. Cargo
applies the feature to every crate in the build that uses `hekate-math`, however: leave it off in a
binary that also generates witnesses.

---

## License

AGPL-3.0-only. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
Commercial licenses are available from Oumuamua Labs <info@oumuamua.dev>.
