# hekate-scribble

[![Crates.io](https://img.shields.io/crates/v/hekate-scribble.svg)](https://crates.io/crates/hekate-scribble)
[![Docs.rs](https://docs.rs/hekate-scribble/badge.svg)](https://docs.rs/hekate-scribble)
[![CI](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml/badge.svg)](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache2-yellow.svg)](./LICENSE)

Trace mutation fuzzer for [Hekate](https://github.com/oumuamua-labs/hekate) ZK programs and chiplets.

Tampers your valid trace, runs preflight checks,
panics if the tamper goes undetected. If scribble
panics, your constraints have a hole.

## What it does

- Flips bits, swaps rows, toggles selectors, injects out-of-range values
- Validates using preflight (row-by-row constraint eval + bus multiset check)
- Proptest integration, shrinks failures to the smallest mutation that escapes your AIR
- Runs in debug mode in seconds, not minutes

## What it doesn't do

- **No proofs.** Scribble never calls the prover or verifier. It checks your constraints on the concrete trace, not
  through the ZK pipeline. This is why it's fast.
- **No protocol-level testing.** Transcript binding, Fiat-Shamir, evaluation arguments, Brakedown, those need the real
  prover/verifier e2e tests.
- **No soundness guarantees.** Passing scribble means every random mutation was caught. It doesn't prove your
  constraints are complete, only that the ones you wrote are wired up.

## Two layers

**Layer 1: random coverage.** `assert_all_caught` with shallow mutations (BitFlip, FlipSelector, SwapRows,
OutOfBounds). Finds unconstrained columns, missing booleans, selector isolation gaps. Proptest shrinks any escape
to a minimal reproduction. Run during chiplet development.

**Layer 2: deterministic emulation.** `check_single_mutation` with hand-crafted structural or compound mutations
(SwapColumns, CopyColumns, Compound). Replaces the 80-line boilerplate of each e2e exploit test with a 15-line
call. Same attack logic, 100x faster (preflight vs prove+verify). Run as regression suite.

## Quick start

```rust
use hekate_scribble::{ScribbleConfig, assert_all_caught};

#[test]
fn my_chiplet_survives_chaos() {
    let (air, instance, witness) = setup_my_chiplet();
    assert_all_caught(&air, &instance, &witness, ScribbleConfig::default());
}
```

## Target a specific chiplet

```rust
use hekate_scribble::{ScribbleConfig, Target, assert_all_caught};

#[test]
fn mlkem_ntt_chiplet_survives_chaos() {
    let (program, instance, witness) = setup_mlkem_fixture();

    let config = ScribbleConfig::default()
        .target(Target::Chiplet(1))
        .cases(512);

    assert_all_caught(&program, &instance, &witness, config);
}
```

## Restrict mutation kinds

```rust
use hekate_scribble::{MutationKind, ScribbleConfig, assert_all_caught};

#[test]
fn ram_selector_fuzzing() {
    let (air, instance, witness) = setup_ram_fixture();

    let config = ScribbleConfig::default()
        .mutations([MutationKind::FlipSelector, MutationKind::SwapRows])
        .cases(1024);

    assert_all_caught(&air, &instance, &witness, config);
}
```

## Dispatch swap (Layer 2)

Swap a subset of columns between two rows. Selectors and RAM columns stay intact,
emulates an attacker who rearranges data while preserving dispatch structure.

```rust
use hekate_scribble::{Mutation, Target, check_single_mutation};

#[test]
fn ntt_dispatch_swap_caught() {
    let (air, instance, witness) = setup_ntt_fixture();

    let ntt_data_cols = vec![
        NTT_A, NTT_B, NTT_A_OUT, NTT_B_OUT,
        NTT_LAYER, NTT_BFLY, NTT_INSTANCE,
    ];

    let mutation = Mutation::SwapColumns {
        target: Target::Chiplet(0),
        cols: ntt_data_cols,
        row_a: 5,
        row_b: 12,
    };

    let result = check_single_mutation(&air, &instance, &witness, &mutation);
    assert!(result.is_ok(), "dispatch swap must be caught by RAM binding");
}
```

## Coordinated cross-trace attack

`Compound` applies multiple mutations atomically. Modify chiplet AND main trace
in sync to bypass per-table checks, the cross-table bus is the only backstop.

```rust
use hekate_scribble::{Mutation, Target, check_single_mutation};

#[test]
fn out_of_range_cross_trace_caught() {
    let (program, instance, witness) = setup_ntt_with_cpu_fixture();

    let mutation = Mutation::Compound(vec![
        Mutation::OutOfBounds {
            target: Target::Chiplet(0),
            col: BUS_B_OUT_PHY,
            row: 0,
            value: Q as u128,
        },
        Mutation::OutOfBounds {
            target: Target::Main,
            col: CPU_B_OUT,
            row: 0,
            value: Q as u128,
        },
    ]);

    let result = check_single_mutation(&program, &instance, &witness, &mutation);
    assert!(result.is_ok(), "out-of-range b_out must be caught");
}
```

## Reading the preflight report

When preflight catches a mutation, the report names the failing invariant, table, and row:

```
  PREFLIGHT: 1 constraint violations, 0 boundary violations, 0 bus issues

    [Chiplet 0] Constraint 32 "bits_01_equal" failed at row 0
```

Boundary violations show the concrete mismatch:

```
  PREFLIGHT: 0 constraint violations, 2 boundary violations, 0 bus issues

    Boundary #0: col=0 row=0 actual=Flat(Block128(0)) expected=Flat(Block128(22067681354706156661646625971774519825))
    Boundary #1: col=1 row=0 actual=Flat(Block128(1)) expected=Flat(Block128(265498766201044366875656389800751278795))
```

Bus diagnostics list every endpoint with its row count, active rows,
and multiset product, so you can see which side of the bus diverged:

```
  PREFLIGHT: 0 constraint violations, 0 boundary violations, 1 bus issues

    Bus "test_bus" (2 endpoints):
      Main: 8 rows, 4 active, product=Flat(Block128(16200159481073039905153729824056248498))
      Chiplet 0: 8 rows, 4 active, product=Flat(Block128(179105452667142969769264969604701840821))
```

If a mutation escapes (report is clean), scribble panics with the proptest-shrunk `Mutation`. That tamper
is the soundness gap, add a constraint that catches it.

## Requirements

Depends on `hekate-math`, `hekate-sdk`, `hekate-core`, `hekate-program`.

## License

Apache-2.0