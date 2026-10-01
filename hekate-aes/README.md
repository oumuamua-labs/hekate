# hekate-aes

[![Crates.io](https://img.shields.io/crates/v/hekate-aes.svg)](https://crates.io/crates/hekate-aes)
[![Docs.rs](https://docs.rs/hekate-aes/badge.svg)](https://docs.rs/hekate-aes)
[![CI](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml/badge.svg)](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml)
[![License: AGPL-3.0-only](https://img.shields.io/badge/License-AGPL--3.0--only-blue.svg)](LICENSE)

*Copyright (c) 2026 Andrei Kochergin and Oumuamua Labs.*

AES-128 / AES-256 AIR chiplet for the [Hekate ZK](https://github.com/oumuamua-labs/hekate) proving system.

Implements FIPS 197 round function (SubBytes, ShiftRows, MixColumns, AddRoundKey) as a binary-field AIR with an
S-box ROM chiplet for the GF(2^8) inversion. Round-AIR trace is wired to the CPU AIR via LogUp bus.

```
Per-block proving cost (Apple M3 Max, zero-knowledge, 31,250 blocks per run):
  AES-128: ~42 µs/block, 1,204 MiB peak, 4,674 KiB proof, 20.6 ms verify
  AES-256: ~45 µs/block, 1,496 MiB peak, 4,959 KiB proof, 20.2 ms verify
```

Conditions and the base-protocol column are in the
[workspace README](https://github.com/oumuamua-labs/hekate#performance).

## Examples

- [AES-128 / AES-256 proving and verification](https://github.com/oumuamua-labs/hekate/blob/main/hekate/examples/aes.rs)

### Usage

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
nothing, and its instance carries no public inputs.

---

## ⚠️ Security Warning

This crate has not been audited and may contain bugs and security flaws.

USE AT YOUR OWN RISK!

### Proof soundness vs. AES

The proof binds at **100 bits**, which is below both AES parameter sets, and the proof
is the weaker link for AES-128 and AES-256 alike. The ciphertext is still full AES.

### Constant-time trace generation

No secret-indexed `SBOX[x]` table, the key cannot leak via cache timing. The S-box is
field arithmetic, the GF(2⁸) inverse (`x²⁵⁴`) plus the FIPS 197 affine map, with no
key-dependent index, branch, or memory access.

---

## License

AGPL-3.0-only. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
Commercial licenses are available from Oumuamua Labs <info@oumuamua.dev>.