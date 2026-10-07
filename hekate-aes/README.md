# hekate-aes

[![Crates.io](https://img.shields.io/crates/v/hekate-aes.svg)](https://crates.io/crates/hekate-aes)
[![Docs.rs](https://docs.rs/hekate-aes/badge.svg)](https://docs.rs/hekate-aes)
[![CI](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml/badge.svg)](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml)
[![License: AGPL-3.0-only](https://img.shields.io/badge/License-AGPL--3.0--only-blue.svg)](https://github.com/oumuamua-labs/hekate/blob/main/hekate-aes/LICENSE)

*Copyright (c) 2026 Andrei Kochergin and Oumuamua Labs.*

AES-128 and AES-256 encryption in zero knowledge for [Hekate](https://oumuamua.dev/hekate), the Rust zero-knowledge
proof engine. A round table proves the FIPS 197 round function (SubBytes, ShiftRows, MixColumns, AddRoundKey) as a
binary-field AIR with an S-box ROM chiplet for the GF(2^8) inversion, and a LogUp bus wires the round table to your
host table.

## ⚠️ Security Warning

This crate has not been independently audited and may contain bugs and security flaws.

USE AT YOUR OWN RISK!

The proof binds at **100 bits**, below both AES parameter sets: the proof is the weaker link for AES-128 and AES-256
alike, and the ciphertext is still full AES. Trace generation is constant-time: the S-box is field arithmetic, the
GF(2^8) inverse plus the FIPS 197 affine map, with no key-dependent index, branch or memory access.

## Usage

```bash
cargo add hekate-aes hekate-core hekate-math hekate-program
```

AES-256 over a run of plaintext blocks under one key, with both chiplets attached as their own tables:

```rust
use hekate_aes::trace::{Aes256Call, expand_key_256};
use hekate_aes::{Aes256Chiplet, AesRound256Air, host_key_selector_shape, host_selector_shape};
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::trace::ColumnTrace;
use hekate_math::Block128;
use hekate_program::circuit::{Circuit, CircuitProgram, Col};
use hekate_program::define_columns;

type F = Block128;

define_columns! {
    HostColumns {
        KEY: [B8; 32],
        KEY_SELECTOR: Bit,
        DATA: [B8; 16],
        SELECTOR: Bit,
    }
}

fn encrypt(
    key: &[u8; 32],
    plaintexts: &[[u8; 16]],
) -> errors::Result<(CircuitProgram<F>, Vec<ColumnTrace>)> {
    let blocks = plaintexts.len();
    let floor = Config::default().min_table_rows();

    let rows = |per_block: usize| (per_block * blocks).next_power_of_two().max(floor);

    let aes = Aes256Chiplet::new(
        rows(AesRound256Air::BLOCK_ROWS),
        rows(AesRound256Air::ACTIVE_ROWS),
        blocks,
    )?;

    let round_keys = expand_key_256(key);
    let calls: Vec<Aes256Call> = plaintexts
        .iter()
        .map(|&plaintext| Aes256Call {
            key: *key,
            plaintext,
            round_keys,
        })
        .collect();

    let traces = aes.generate_traces(&calls)?;

    let mut cx = Circuit::<F>::new("Host", rows(2))?;

    let host = cx.schema(&HostColumns::build_layout());

    let sel = host.at(HostColumns::SELECTOR);
    let key_sel = host.at(HostColumns::KEY_SELECTOR);

    let data: Vec<Col> = (0..16).map(|j| host.at(HostColumns::DATA + j)).collect();

    let key_bytes: Vec<Col> = (0..32).map(|j| host.at(HostColumns::KEY + j)).collect();

    cx.call(&AesRound256Air::link_service(), &data, sel)?;
    cx.call(&AesRound256Air::key_service(), &key_bytes, key_sel)?;

    cx.fix(sel, host_selector_shape(2, blocks));
    cx.fix(key_sel, host_key_selector_shape(2, blocks));

    cx.attach_namespaced("aes256", aes.defs()?, &Aes256Chiplet::EXTERNAL_BUS_IDS)?;

    Ok((cx.compile()?, traces))
}
```

The host trace holds block `b`'s input `plaintext ⊕ round_keys[0]` and the key on row `2b` and its ciphertext
`aes256_encrypt_block(&round_keys, &plaintext)` on row `2b + 1`, with `SELECTOR` set on both and `KEY_SELECTOR` on the
first. `traces` holds the round table and the S-box ROM, the witness's chiplet traces in that order. The host publishes
nothing, and its instance carries no public inputs. AES-128 runs on `Aes128Chiplet`, `AesRound128Air` and `Aes128Call`
with a 16-byte key and round keys from `trace::expand_key`; the crate has no AES-128 block function, and `aes.rs` reads
each ciphertext from the round table's trace.

[`aes.rs`](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/aes.rs) is a complete program for both
key sizes.

## Documentation

- [AES-128 and AES-256 Encryption](https://oumuamua.dev/primitives/encryption/aes): what the proof states, what stays
  outside it, and proving time, proof size and memory
- [ML-KEM Sender / Receiver](https://oumuamua.dev/primitives/encryption/mlkem): AES-256-CTR under a key from ML-KEM,
  joined in one proof
- [API reference](https://docs.rs/hekate-aes)

## License

AGPL-3.0-only. See [LICENSE](https://github.com/oumuamua-labs/hekate/blob/main/hekate-aes/LICENSE) and
[NOTICE](https://github.com/oumuamua-labs/hekate/blob/main/hekate-aes/NOTICE).
Commercial licenses are available from Oumuamua Labs <info@oumuamua.dev>.
