# hekate-sdk

Bundling, wire-format, and preflight diagnostics for the Hekate ZK proving system.

Proving is driven through `hekate-prover-sys` (which links the signed cdylib).
Verification is `hekate_verifier::HekateVerifier::verify`. This crate owns the
serialization, identity, and preflight layers — not the prove/verify call sites.

## Preflight diagnostics

Evaluate constraints row-by-row on the concrete trace before proving. Catches
constraint, boundary, and bus violations without running the prover.

```rust
use hekate_sdk::preflight;

let report = preflight(&program, &instance, &witness)?;
assert!(report.is_clean());
```

## Bundle wire format

`build_bundle` / `serialize_bundle` / `deserialize_bundle` produce and parse the
internal program + instance + config + chiplet-defs payload. `serialize_bundle_header`
emits a witness-free bundle for the `hekate-prover-sys` ↔ cdylib boundary.

## Program identity

`program_id(program)` / `program_id_hex(program)` derive a stable 32-byte hash
over the program's structure (layout, constraints, chiplet defs, bus topology).

## License

Licensed under Apache 2.0. See the [LICENSE](LICENSE) and [NOTICE](NOTICE) files
for details.