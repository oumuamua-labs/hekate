// SPDX-License-Identifier: Apache-2.0
// This file is part of the hekate project.
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

#[path = "common/mod.rs"]
mod common;

use hekate::core::trace::{ColumnTrace, ColumnType};
use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::Block128;
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_gadgets::{
    ArithmeticOpcode, IntArithmeticChiplet, IntArithmeticLayout, IntArithmeticOp,
    generate_arithmetic_trace,
};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::ConstraintAst;
use hekate_program::expander::VirtualExpander;
use hekate_program::{Air, InlineKernelHint, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;
use rand::{TryRngCore, rngs::OsRng};

type F = Block128;
type H = DefaultHasher;

// =================================================================
// 1. INLINED ARITHMETIC CHIPLET - SINGLE TRACE
//
// 1:1 op:row makes a separate chiplet trace wasteful
// (2× commit, 2× ZeroCheck, 2× eval, plus a LogUp bus).
// Reuse the arithmetic chiplet's columns + AIR directly
// as the program's trace and rely on the registered
// IntArith kernel via `inline_chiplet_kernels`.
//
// Trace columns = IntArithmeticChiplet physical layout.
// Workload     = cycle through ADD, SUB, AND, XOR, NOT, LT.
// =================================================================

#[derive(Clone)]
struct ArithProgram {
    chiplet: IntArithmeticChiplet,
}

impl ArithProgram {
    fn new(num_rows: usize) -> Self {
        let chiplet = IntArithmeticChiplet::new(32, num_rows)
            .expect("IntArithmeticChiplet::new(32, num_rows)");
        Self { chiplet }
    }
}

impl Air<F> for ArithProgram {
    fn column_layout(&self) -> &[ColumnType] {
        <IntArithmeticChiplet as Air<F>>::column_layout(&self.chiplet)
    }

    fn virtual_expander(&self) -> Option<&VirtualExpander> {
        <IntArithmeticChiplet as Air<F>>::virtual_expander(&self.chiplet)
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        <IntArithmeticChiplet as Air<F>>::constraint_ast(&self.chiplet)
    }

    fn inline_chiplets(&self) -> errors::Result<Vec<ChipletDef<F>>> {
        Ok(vec![ChipletDef::from_air(&self.chiplet)?])
    }

    fn inline_chiplet_kernels(&self) -> Vec<InlineKernelHint> {
        vec![InlineKernelHint {
            chiplet_idx: 0,
            root_offset: 0,
            column_offset: 0,
        }]
    }
}

impl Program<F> for ArithProgram {}

// =================================================================
// 2. WORKLOAD + TRACE GENERATION
// =================================================================
fn generate_all_ops_workload(num_ops: usize) -> Vec<IntArithmeticOp> {
    let mut ops = Vec::with_capacity(num_ops);
    for i in 0..num_ops {
        let a = (i * 12345) as u32;
        let b = (i * 67890) as u32;

        let op = match i % 6 {
            0 => ArithmeticOpcode::ADD,
            1 => ArithmeticOpcode::SUB,
            2 => ArithmeticOpcode::AND,
            3 => ArithmeticOpcode::XOR,
            4 => ArithmeticOpcode::NOT,
            5 => ArithmeticOpcode::LT,
            _ => unreachable!(),
        };

        let b = if matches!(op, ArithmeticOpcode::NOT) {
            0
        } else {
            b
        };

        ops.push(IntArithmeticOp::U32 {
            op,
            a,
            b,
            request_idx: i as u32,
        });
    }

    ops
}

fn generate_arith_trace(num_ops: usize, num_rows: usize) -> errors::Result<ColumnTrace> {
    let layout = IntArithmeticLayout::compute(32);
    let ops = generate_all_ops_workload(num_ops);

    generate_arithmetic_trace(&ops, &layout, num_rows)
}

// =================================================================
// 3. MAIN EXECUTION
// =================================================================
fn main() {
    common::init("Arithmetic Chiplet (inline + IntArith kernel)");

    let num_vars: usize = 20;
    let num_rows: usize = 1 << num_vars;
    let num_ops: usize = num_rows;

    let mut config = Config {
        sumcheck_blinding_factor: 2,
        ..Config::default()
    };

    OsRng.try_fill_bytes(&mut config.matrix_seed).unwrap();

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    println!(
        "Total Ops: {} (cycle through ADD, SUB, AND, XOR, NOT, LT)",
        num_ops
    );
    println!("Trace: 2^{} rows (single AIR)", num_vars);
    println!(
        "ZK Blinding: {} virtual columns",
        config.sumcheck_blinding_factor
    );

    let trace = common::phase("Trace Generation", || {
        generate_arith_trace(num_ops, num_rows).expect("arith trace")
    });

    let program = ArithProgram::new(num_rows);
    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(trace);

    let proof = common::phase("Proving", || {
        prove(
            b"Arith_Example",
            &program,
            &instance,
            &witness,
            &config,
            blinding_seed,
            None,
        )
        .expect("Prover failed")
    });

    common::proof_breakdown(&proof);

    let mut verifier_transcript = Transcript::<H>::new(b"Arith_Example");

    let is_valid = common::phase("Verifying", || {
        HekateVerifier::<F, H>::verify(
            &program,
            &instance,
            &proof,
            &mut verifier_transcript,
            &config,
        )
        .expect("Verifier failed")
    });

    common::result(is_valid);
}
