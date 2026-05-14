# hekate-core

Core primitives for the Hekate ZK proving system.

## Modules

| Module       | Description                                                  |
|--------------|--------------------------------------------------------------|
| `merkle`     | Binary Merkle tree with subtree openings                     |
| `transcript` | Fiat-Shamir transcript with wide-pipe squeeze                |
| `poly`       | Zero-copy multilinear polynomial views and univariate rounds |
| `tensor`     | Lazy `Eq(x, r)` with constant-time fold                      |
| `trace`      | Typed trace-column storage and builder                       |
| `proofs`     | Wire-level proof and commitment types                        |
| `config`     | LDT security parameters and relative-distance estimate       |

## Features

| Feature            | Default | Effect                                                |
|--------------------|---------|-------------------------------------------------------|
| `std`              | yes     | Enable `std` (transitively through dependencies)      |
| `parallel`         | yes     | Rayon-backed Merkle build                             |
| `blake3`           | yes     | Blake3 as `DefaultHasher`                             |
| `sha2`             | no      | SHA-256 as `DefaultHasher`                            |
| `sha3`             | no      | SHA-3-256 as `DefaultHasher`                          |
| `secure-memory`    | no      | `ZeroizeOnDrop` on `TraceColumn`                      |

Exactly one of `blake3` / `sha2` / `sha3` must be enabled.

## Usage

```toml
[dependencies]
hekate-core = "0.23"
```

## License

Licensed under Apache 2.0. See the [LICENSE](LICENSE) and [NOTICE](NOTICE) files for details.
