// SPDX-License-Identifier: Apache-2.0
// This file is part of the hekate-math project.
// Copyright (C) 2026 Andrei Kochergin <andrei@oumuamua.dev>
// Copyright (C) 2026 Oumuamua Labs <info@oumuamua.dev>. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Chiplet modules for the Hekate.
//!
//! Chiplets are specialized execution units that
//! handle specific operations and link to the main
//! CPU via Grand Product Arguments.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
extern crate core;

pub mod atoms;
pub mod chiplets;

pub use chiplets::int::arith::{
    generate_arithmetic_trace, ArithmeticOpcode, CpuArithColumns, CpuIntArithmeticUnit,
    IntArithmeticChiplet, IntArithmeticLayout, IntArithmeticOp,
};
pub use chiplets::keccak::{
    generate_keccak_trace, sha3_256, sha3_512, shake128, shake256, CpuKeccakColumns, CpuKeccakUnit,
    KeccakCall, KeccakChiplet, KeccakColumns, KeccakSpongeNative, KeccakWitness,
};
pub use chiplets::ram::{
    generate_ram_trace, CpuMemColumns, CpuMemoryUnit, MemoryEvent, RamChiplet, RamColumns,
};
pub use chiplets::rom::{
    generate_rom_trace, CpuFetchColumns, CpuFetchUnit, Instruction, RomChiplet, RomColumns,
};
