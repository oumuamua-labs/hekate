// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Base chiplets and constraint atoms of Hekate, the
//! Rust zero-knowledge proof engine.
//!
//! The chiplets are tables a program calls over a
//! LogUp bus: [`IntArithmeticChiplet`],
//! [`RamChiplet`], [`RomChiplet`] and
//! [`ModexpChiplet`], the modular exponentiation
//! behind RSA. The [`atoms`] are constraint helpers
//! that run inside the caller's own table: carry and
//! borrow chains, range checks, multiplication and
//! arithmetic modulo a constant.
//!
//! - [Chiplets and atoms][gadgets]: which form to
//!   use, and what each costs
//! - [Quick Example][quick]: Fibonacci on the
//!   arithmetic chiplet, proved and verified
//!
//! [gadgets]: https://oumuamua.dev/hekate/docs/basics/cryptographic-chiplets#gadgets-chiplets-and-atoms
//! [quick]: https://oumuamua.dev/hekate/docs#quick-example

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
extern crate core;

pub mod atoms;
pub mod chiplets;

pub use chiplets::bignum::cpu::CpuModexpBlock;
pub use chiplets::bignum::modexp::{Modexp, ModexpChiplet};
pub use chiplets::int::arith::{
    ArithmeticOpcode, CpuArithColumns, IntArithmeticChiplet, IntArithmeticLayout, IntArithmeticOp,
    generate_arithmetic_trace,
};
pub use chiplets::ram::{CpuMemColumns, MemoryEvent, RamChiplet, RamColumns, generate_ram_trace};
pub use chiplets::rom::{CpuFetchColumns, Instruction, RomChiplet, RomColumns, generate_rom_trace};
