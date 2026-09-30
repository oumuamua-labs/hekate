# hekate-pqc

[![Crates.io](https://img.shields.io/crates/v/hekate-pqc.svg)](https://crates.io/crates/hekate-pqc)
[![Docs.rs](https://docs.rs/hekate-pqc/badge.svg)](https://docs.rs/hekate-pqc)
[![CI](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml/badge.svg)](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml)
[![License: AGPL-3.0-only](https://img.shields.io/badge/License-AGPL--3.0--only-blue.svg)](LICENSE)

*Copyright (c) 2026 Andrei Kochergin and Oumuamua Labs.*

Post-quantum chiplets for the [Hekate](https://github.com/oumuamua-labs/hekate) ZK proving system:
ML-KEM key generation, encapsulation and decapsulation (FIPS 203) and ML-DSA signature
verification (FIPS 204), at every parameter set, proven natively over binary tower fields.

```
Proving on Apple M3 Max (zero-knowledge):
  ML-KEM-768 sender   : 573 ms, 248 MiB peak, 3,512 KiB proof, 72.6 ms verify
  ML-KEM-768 receiver : 746 ms, 359 MiB peak, 3,895 KiB proof, 90.3 ms verify
  ML-DSA-44           : 558 ms, 280 MiB peak, 2,877 KiB proof, 51.6 ms verify
  ML-DSA-65           : 617 ms, 274 MiB peak, 2,958 KiB proof, 53.8 ms verify
  ML-DSA-87           : 776 ms, 361 MiB peak, 3,326 KiB proof, 58.5 ms verify
```

The sender proves Encaps and AES-256-CTR over an 87-byte message; the receiver
proves KeyGen chained into Decaps. Conditions and the base-protocol column are in the
[workspace README](https://github.com/oumuamua-labs/hekate#performance).

---

## ⚠️ Security Warning

This crate has not been audited and may contain bugs and security flaws.

USE AT YOUR OWN RISK!

---

## How a call is proven

Each scheme is a pipeline of tables that a host program attaches. The host trades 32-bit words with the
ctrl table over one service bus. The tables pass coefficients, words and Keccak lanes to each other over
internal buses, where every value is emitted once by its producer and once by its consumer. ML-KEM
Encaps runs through the tables like this:

```
FIPS 203 Encaps_internal(ek, m)               table
────────────────────────────────────────────  ───────────────
h ← H(ek); (K, r) ← G(m ‖ h)                  ctrl, Keccak
t̂ ← ByteDecode12(ek); μ ← Decompress1(m)      Codec
Â ← SampleNTT(ρ); y, e1, e2 ← CBD(r)          Sampler, Keccak
ŷ ← NTT(y); s ← e2 + μ                        NTT
û ← Âᵀ∘ŷ; v̂ ← t̂ᵀ∘ŷ                            PolyArith
u ← NTT⁻¹(û) + e1; v ← NTT⁻¹(v̂) + s           NTT
c1 ← Compress_du(u); c2 ← Compress_dv(v)      Codec
```

Decaps decrypts c on the same tables and re-encrypts, and a KemSelect table compares c with c′ and
returns K′ or the implicit-rejection key K̄. KeyGen samples from G(d ‖ k) and encodes t̂ and ŝ in the
Codec. ML-DSA verification runs on the ctrl, Sampler, Keccak, Codec and NTT tables plus HighBits.

---

## Examples

- [ML-KEM sender](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/mlkem_sender.rs):
  Encaps and AES-256-CTR, publishing H(ek), c, the message and its ciphertext
- [ML-KEM receiver](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/mlkem_receiver.rs):
  KeyGen chained into Decaps, publishing H(ek), c and valid
- [ML-DSA verification](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/mldsa.rs):
  the public key and signature in the witness, publishing M′ and tr = H(pk)

### Usage

ML-DSA-65 verification of one signature over M′:

```rust
use hekate_core::errors;
use hekate_math::{Block128, TowerField};
use hekate_pqc::mldsa::{
    self, MLDSA_DATA_BUS_ID, MlDsaChiplet, MlDsaInput, MlDsaParams, MlDsaWitness,
};
use hekate_program::FixedShape;
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::define_columns;

type F = Block128;

define_columns! {
    HostColumns {
        WORD: B32,
        SEL: Bit,
    }
}

fn verify(
    pk: &[u8],
    m_prime: &[u8],
    signature: &[u8],
) -> errors::Result<(CircuitProgram<F>, MlDsaWitness)> {
    let pipeline = MlDsaChiplet::<F>::new(MlDsaParams::ML_DSA_65, &[m_prime.len()])?;
    let witness = pipeline.trace(&[MlDsaInput {
        pk,
        message: m_prime,
        signature,
    }])?;

    let rows = witness.words.len().next_power_of_two();

    let mut cx = Circuit::<F>::new("Host", rows)?;

    let host = cx.schema(&HostColumns::build_layout());

    let word = host.at(HostColumns::WORD);
    let sel = host.at(HostColumns::SEL);

    cx.fix(
        sel,
        FixedShape::Cadence {
            stride: 1,
            count: witness.words.len(),
            origin: 0,
            values: vec![F::ONE],
        },
    );

    cx.call(&mldsa::service(), &[word], sel)?;

    let cs = cx.cs();

    // WORD is zero past the service rows: no free cell on padding
    cs.constrain((cs.one() + cs.col(sel.index())) * cs.col(word.index()));

    // Public: M′ after the pk words, and tr, the first returned digest
    let pk_words = pk.len() / 4;
    let tr_row = witness.words.len() - 32;

    for row in (pk_words..pk_words + m_prime.len().div_ceil(4)).chain(tr_row..tr_row + 16) {
        cx.publish(word, row);
    }

    cx.attach_namespaced("mldsa", pipeline.defs()?, &[MLDSA_DATA_BUS_ID])?;

    Ok((cx.compile()?, witness))
}
```

ML-KEM-768 encapsulation to `ek`:

```rust
use hekate_core::errors;
use hekate_math::{Block128, TowerField};
use hekate_pqc::mlkem::{
    self, MLKEM_DATA_BUS_ID, MlKemCall, MlKemChiplet, MlKemInput, MlKemParams, MlKemWitness,
};
use hekate_program::FixedShape;
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::define_columns;

type F = Block128;

define_columns! {
    HostColumns {
        WORD: B32,
        SEL: Bit,
    }
}

fn encaps(ek: &[u8], m: &[u8; 32]) -> errors::Result<(CircuitProgram<F>, MlKemWitness)> {
    let pipeline = MlKemChiplet::<F>::new(MlKemParams::ML_KEM_768, &[MlKemCall::Encaps])?;
    let witness = pipeline.trace(&[MlKemInput::Encaps { ek, m }])?;

    let rows = witness.words.len().next_power_of_two();

    let mut cx = Circuit::<F>::new("Host", rows)?;

    let host = cx.schema(&HostColumns::build_layout());

    let word = host.at(HostColumns::WORD);
    let sel = host.at(HostColumns::SEL);

    cx.fix(
        sel,
        FixedShape::Cadence {
            stride: 1,
            count: witness.words.len(),
            origin: 0,
            values: vec![F::ONE],
        },
    );

    cx.call(&mlkem::service(), &[word], sel)?;

    let cs = cx.cs();

    // WORD is zero past the service rows: no free cell on padding
    cs.constrain((cs.one() + cs.col(sel.index())) * cs.col(word.index()));

    // Public: H(ek) after ek and m, and c after K, at the end
    let h_row = ek.len() / 4 + 8;

    for row in (h_row..h_row + 8).chain(h_row + 16..witness.words.len()) {
        cx.publish(word, row);
    }

    cx.attach_namespaced("mlkem", pipeline.defs()?, &[MLKEM_DATA_BUS_ID])?;

    Ok((cx.compile()?, witness))
}
```

In both, the host trace fills `word` with `witness.words` and `sel` with ones, `witness.traces` are
the chiplet traces, and `cx.publish` exposes the rows the statement needs.

---

### Proof soundness vs. PQC security level

The proof binds at **100 bits** regardless of parameter set, which is below every level here,
and the proof is the weaker link rather than the lattice scheme. The proven operations still
run the full FIPS 203 / 204 parameter sets.

---

## License

AGPL-3.0-only. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
Commercial licenses are available from Oumuamua Labs <info@oumuamua.dev>.