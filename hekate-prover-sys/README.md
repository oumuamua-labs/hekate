# hekate-prover-sys

Open-source FFI shim that links the closed-source Hekate prover cdylib
over a stable C ABI. The cdylib is fetched at build time, verified
against a pinned manifest, and dynamically linked. This is the only
crate in the workspace that can call the prover.

## Public surface

- `prove(label, program, instance, witness, config, seed, cancel)` —
  the single prove entry point. Builds a wire bundle, hands it to the
  cdylib, deserializes the returned `InnerProof<Block128>`.
- `CancelToken` — cooperative cancellation; pass `Some(&token)` to
  `prove` and call `token.cancel()` from another thread.
- `Error`, `ErrorCode` — typed FFI / bundle / witness / deserialize
  failure modes.
- `version()`, `build_id()` — strings reported by the linked cdylib.

## Features

Pick exactly one runtime profile, plus one hash backend.

**Profile** (mutually exclusive, no default — `build.rs` rejects the
build if neither or both are set):

- `ct` — constant-time arithmetic. Required for any prover that
  handles a private witness.
- `public` — variable-time `table-math`. Public-data only (rollups,
  recursive verifiers). Faster, leaks witness via timing.

**Hash backend** (default: `blake3`):

- `blake3`, `sha2`, `sha3` — must match the verifier-side
  `DefaultHasher` and the hash backend the cdylib was built with.
  Mismatch silently fails verification.

## Building

`build.rs` resolves the cdylib in this order:

1. `HEKATE_PROVER_DYLIB_DIR=/abs/path/to/dist/<version>/<variant>/<triple>` —
   point at a local cdylib directory (development).
2. `~/.cache/hekate-prover-sys/<version>/<variant>/<triple>/<filename>` —
   reuse a previously downloaded copy.
3. Download from the URL in `artifacts/manifest.toml`.

Every resolved cdylib is verified before linking:

- SHA-256 digest
- Ed25519 signature
- ML-DSA-65 (post-quantum) signature

Manifest signatures use Anthropic-pinned publisher keys; any tamper
fails the build.

## License

Licensed under Apache 2.0. See the [LICENSE](LICENSE) and [NOTICE](NOTICE)
files for details.