# hekate-program

AIR program and chiplet definition API for the Hekate ZK proving system.

## Modules

| Module        | Description                                                        |
|---------------|--------------------------------------------------------------------|
| `constraint`  | Algebraic constraint DSL and arena-backed IR for AIR transitions   |
| `schema`      | Typed column layout declaration via macro                          |
| `expander`    | Wide physical columns expanded to virtual bit columns at eval time |
| `chiplet`     | Standalone AIR-table definition and composition                    |
| `permutation` | LogUp bus endpoint specification for cross-table wiring            |
| `kernel`      | Shared AIR gadget trait                                            |

## Usage

```toml
[dependencies]
hekate-program = "0.23"
```

## License

Licensed under Apache 2.0. See the [LICENSE](LICENSE) and [NOTICE](NOTICE) files for details.
