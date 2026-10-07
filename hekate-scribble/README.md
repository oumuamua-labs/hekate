# hekate-scribble

[![Crates.io](https://img.shields.io/crates/v/hekate-scribble.svg)](https://crates.io/crates/hekate-scribble)
[![Docs.rs](https://docs.rs/hekate-scribble/badge.svg)](https://docs.rs/hekate-scribble)
[![CI](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml/badge.svg)](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml)
[![License: AGPL-3.0-only](https://img.shields.io/badge/License-AGPL--3.0--only-blue.svg)](https://github.com/oumuamua-labs/hekate/blob/main/hekate-scribble/LICENSE)

*Copyright (c) 2026 Andrei Kochergin and Oumuamua Labs.*

A trace mutation fuzzer for [Hekate](https://oumuamua.dev/hekate) programs and chiplets. It tampers with a valid
trace, runs preflight on each tamper, and panics with the smallest one preflight missed. A tamper that passes is a
hole in your constraints.

Scribble never calls the prover or the verifier, which keeps a run to seconds in a debug build. It tests constraints
on the concrete trace: transcript binding and the evaluation argument need prove-and-verify tests. Catching every
tamper shows the constraints you wrote are wired up; a constraint missing where no tamper lands goes unnoticed.

## Usage

```bash
cargo add --dev hekate-scribble hekate-math hekate-program
```

```rust
use hekate_math::Block128;
use hekate_program::{Program, ProgramInstance, ProgramWitness};
use hekate_scribble::{
    Mutation, ScribbleConfig, Target, assert_all_caught_all_targets, check_single_mutation,
};

type F = Block128;

fn fuzz<P: Program<F>>(program: &P, instance: &ProgramInstance<F>, witness: &ProgramWitness<F>) {
    let config = ScribbleConfig::default().cases(512);

    assert_all_caught_all_targets(program, instance, witness, config);
}

fn swapped_rows_are_caught<P: Program<F>>(
    program: &P,
    instance: &ProgramInstance<F>,
    witness: &ProgramWitness<F>,
) -> bool {
    let swap = Mutation::SwapRows {
        target: Target::Main,
        row_a: 0,
        row_b: 1,
    };

    check_single_mutation(program, instance, witness, &swap).is_ok()
}
```

Call both from tests with a witness your generator built. `fuzz` runs 512 random tampers on the main trace and 512
on each chiplet trace. `check_single_mutation` runs one tamper you choose and returns `Ok` with the report of the
checks that caught it, or `Err` with the clean report of a tamper that escaped.

| Tampers                | Kinds                                                                                                                            |
|:-----------------------|:---------------------------------------------------------------------------------------------------------------------------------|
| Random                 | `BitFlip`, `OutOfBounds`, `FlipSelector`, `SwapRows`, `DuplicateRow`, `ColumnUniformWrite`, `RowSegmentZero`, `MonotonicReplace` |
| Hand-made, in addition | `SwapColumns`, `CopyColumns`, and `Compound`: several tampers at once, such as a chiplet trace and the main trace changed together |

`ScribbleConfig` narrows a random run with `target`, `mutations`, `include_cols`, `exclude_cols`, `include_rows` and
`exclude_rows`.

## Documentation

- [Preflight: keep it in your tests](https://oumuamua.dev/hekate/docs/basics/preflight#keep-it-in-your-tests): tests
  that break one cell on purpose and assert the check that catches it
- [Preflight](https://oumuamua.dev/hekate/docs/basics/preflight): what each check covers, and what a clean report does
  not prove
- [API reference](https://docs.rs/hekate-scribble)

## License

AGPL-3.0-only. See [LICENSE](https://github.com/oumuamua-labs/hekate/blob/main/hekate-scribble/LICENSE) and
[NOTICE](https://github.com/oumuamua-labs/hekate/blob/main/hekate-scribble/NOTICE).
Commercial licenses are available from Oumuamua Labs <info@oumuamua.dev>.
