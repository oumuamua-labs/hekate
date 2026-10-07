# hekate-gadgets

[![Crates.io](https://img.shields.io/crates/v/hekate-gadgets.svg)](https://crates.io/crates/hekate-gadgets)
[![Docs.rs](https://docs.rs/hekate-gadgets/badge.svg)](https://docs.rs/hekate-gadgets)
[![CI](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml/badge.svg)](https://github.com/oumuamua-labs/hekate/actions/workflows/ci.yml)
[![License: AGPL-3.0-only](https://img.shields.io/badge/License-AGPL--3.0--only-blue.svg)](https://github.com/oumuamua-labs/hekate/blob/main/hekate-gadgets/LICENSE)

*Copyright (c) 2026 Andrei Kochergin and Oumuamua Labs.*

The base parts of [Hekate](https://oumuamua.dev/hekate), the Rust zero-knowledge proof engine, in two forms.
Chiplets are tables your program calls over a LogUp bus: integer arithmetic, RAM, ROM and the modular exponentiation
behind RSA. Atoms are constraint helpers that run inside your own table: carry and borrow chains, range checks,
multiplication and arithmetic modulo a constant.

## Usage

```bash
cargo add hekate-gadgets hekate-core hekate-math hekate-program
```

Fibonacci over 32-bit integers, every addition offloaded to the arithmetic chiplet:

```rust
use hekate_core::errors;
use hekate_gadgets::IntArithmeticChiplet;
use hekate_math::{Block128, TowerField};
use hekate_program::FixedShape;
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram, Col};
use hekate_program::define_columns;

type F = Block128;

define_columns! {
    ProgColumns {
        VAL_A: B32,
        VAL_B: B32,
        VAL_RES: B32,
        OPCODE: B32,
        SELECTOR: Bit,
    }
}

fn build_program(num_rows: usize) -> errors::Result<CircuitProgram<F>> {
    let mut cx = Circuit::<F>::new("Fibonacci", num_rows)?;

    let cpu = cx.schema(&ProgColumns::build_layout());
    let selector = cpu.at(ProgColumns::SELECTOR);
    let a = cpu.at(ProgColumns::VAL_A);
    let b = cpu.at(ProgColumns::VAL_B);
    let res = cpu.at(ProgColumns::VAL_RES);
    let opcode = cpu.at(ProgColumns::OPCODE);

    let cs = cx.cs();

    let col = |c: Col| cs.col(c.index());
    let next = |c: Col| cs.next(c.index());

    cs.constrain(col(selector) * (next(a) + col(b))); // next(a) = b
    cs.constrain(col(selector) * (next(b) + col(res))); // next(b) = res = a + b
    cs.constrain(col(opcode)); // opcode = ADD

    cx.call(
        &IntArithmeticChiplet::service(),
        &[a, b, res, opcode],
        selector,
    )?;

    cx.attach(ChipletDef::from_air(&IntArithmeticChiplet::new(
        32,
        num_rows,
        num_rows - 1,
    )?)?);

    cx.fix(
        selector,
        FixedShape::Cadence {
            stride: 1,
            count: num_rows - 1,
            origin: 0,
            values: vec![F::ONE],
        },
    );

    cx.boundary(a, 0, F::ZERO);
    cx.boundary(b, 0, F::ONE);

    cx.publish(b, num_rows - 1);

    cx.compile()
}
```

The chiplet computes the operation its opcode names, and the opcode is part of the request: the host pins it to
`ADD`, which is 0, or a prover could request any operation. With the pin, the bus guarantees `res = a + b` modulo
2^32 on every row where the selector is 1. `generate_arithmetic_trace` builds the chiplet's trace from the list of
operations.

An atom runs inside your own table instead. `range_check` proves a 12-bit value is below q = 3329, the ML-KEM
modulus, on every row:

```rust
use hekate_core::errors;
use hekate_gadgets::atoms::int_arith::range_check;
use hekate_math::Block128;
use hekate_program::circuit::{Circuit, CircuitProgram, Col};
use hekate_program::define_columns;

type F = Block128;

const Q: u32 = 3329;

define_columns! {
    Coefficients {
        VALUE: [Bit; 12],
        RESULT: [Bit; 12],
        BORROW: [Bit; 13],
    }
}

fn below_q(rows: usize) -> errors::Result<CircuitProgram<F>> {
    let mut cx = Circuit::<F>::new("BelowQ", rows)?;

    let cols = cx.schema(&Coefficients::build_layout());

    let cs = cx.cs();

    let col = |c: Col| cs.col(c.index());
    let bits = |first: usize, width: usize| -> Vec<_> {
        (0..width).map(|i| col(cols.at(first + i))).collect()
    };

    let value = bits(Coefficients::VALUE, 12);

    for &bit in &value {
        cs.assert_boolean(bit);
    }

    range_check(
        cs,
        &value,
        &bits(Coefficients::RESULT, 12),
        &bits(Coefficients::BORROW, 13),
        Q,
    );

    cx.compile()
}
```

`range_check` subtracts the value from q − 1 bit by bit and pins the last borrow to zero, which holds exactly when the
value is below q. It boolean-checks the result and borrow bits; the value's bits are yours to check, as the loop does.
The trace fills `RESULT` and `BORROW` with that subtraction on every row.

## Documentation

- [Quick Example](https://oumuamua.dev/hekate/docs#quick-example): this program with its traces, proved and verified
- [Cryptographic Chiplets: chiplets and atoms](https://oumuamua.dev/hekate/docs/basics/cryptographic-chiplets#gadgets-chiplets-and-atoms):
  which form to use, and what each costs
- [LogUp Buses](https://oumuamua.dev/hekate/docs/basics/logup-buses): what the bus between a host and a chiplet checks
- [RSA-2048 PKCS#1 v1.5 Verification](https://oumuamua.dev/primitives/signatures/rsa): the modexp chiplet at work
- [API reference](https://docs.rs/hekate-gadgets)

## License

AGPL-3.0-only. See [LICENSE](https://github.com/oumuamua-labs/hekate/blob/main/hekate-gadgets/LICENSE) and
[NOTICE](https://github.com/oumuamua-labs/hekate/blob/main/hekate-gadgets/NOTICE).
Commercial licenses are available from Oumuamua Labs <info@oumuamua.dev>.
