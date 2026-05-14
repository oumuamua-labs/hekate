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

use hekate::core::trace::{ColumnTrace, ColumnType, TraceColumn};
use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::{Block32, Block128, HardwareField};
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_gadgets::{
    ArithmeticOpcode, IntArithmeticChiplet, IntArithmeticLayout, IntArithmeticOp,
    generate_arithmetic_trace,
};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::constraint::{BoundaryConstraint, ConstraintAst};
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
// as the program's trace, then add Fibonacci transition
// constraints on top.
//
// Trace columns = IntArithmeticChiplet physical layout (12 cols, 27 B/row).
// Trace rows    = Fibonacci length N (power of two). Rows 0..N-2 are
// active ADD ops; row N-1 is padding holding fib[N-1] in val_b.
// =================================================================

const PHY_VAL_A: usize = 0;
const PHY_VAL_B: usize = 1;

#[derive(Clone)]
struct FibChipletProgram {
    num_rows: usize,
    chiplet: IntArithmeticChiplet,
}

impl FibChipletProgram {
    fn new(num_rows: usize) -> Self {
        let chiplet =
            IntArithmeticChiplet::new(32, num_rows).expect("ArithmeticChiplet::new(32, num_rows)");
        Self { num_rows, chiplet }
    }
}

impl Air<F> for FibChipletProgram {
    fn boundary_constraints(&self) -> Vec<BoundaryConstraint<F>> {
        vec![BoundaryConstraint::with_public_input(
            self.chiplet.layout().val_b,
            self.num_rows - 1,
            0,
        )]
    }

    fn column_layout(&self) -> &[ColumnType] {
        <IntArithmeticChiplet as Air<F>>::column_layout(&self.chiplet)
    }

    fn virtual_expander(&self) -> Option<&VirtualExpander> {
        <IntArithmeticChiplet as Air<F>>::virtual_expander(&self.chiplet)
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let mut ast = <IntArithmeticChiplet as Air<F>>::constraint_ast(&self.chiplet);

        let layout = self.chiplet.layout();
        let cs = ConstraintSystem::<F>::new();

        let s_add = cs.col(layout.s_add);
        let val_b = cs.col(layout.val_b);
        let val_res = cs.col(layout.val_res);
        let next_val_a = cs.next(layout.val_a);
        let next_val_b = cs.next(layout.val_b);

        cs.constrain(s_add * (next_val_a + val_b));
        cs.constrain(s_add * (next_val_b + val_res));

        let fib_ast = cs.build();

        ast.merge(fib_ast);

        ast
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

impl Program<F> for FibChipletProgram {
    fn num_public_inputs(&self) -> usize {
        1
    }
}

// =================================================================
// 2. TRACE GENERATION
//
// Build the chiplet trace via `generate_arithmetic_trace` with N-1
// ADD ops. Patch the padding row's val_a / val_b so the Fibonacci
// transition `next_a = b, next_b = res` from row N-2 to N-1 is
// satisfied (gated by s_add[N-2] = 1).
// =================================================================

fn generate_fib_trace(num_rows: usize) -> errors::Result<(ColumnTrace, u32)> {
    let layout = IntArithmeticLayout::compute(32);

    let mut a: u32 = 0;
    let mut b: u32 = 1;
    let mut prev_b: u32 = 0;

    let mut ops: Vec<IntArithmeticOp> = Vec::with_capacity(num_rows - 1);

    for i in 0..num_rows - 1 {
        let sum = a.wrapping_add(b);
        ops.push(IntArithmeticOp::U32 {
            op: ArithmeticOpcode::ADD,
            a,
            b,
            request_idx: i as u32,
        });

        prev_b = b;
        a = b;
        b = sum;
    }

    let final_b = b;
    let final_a = prev_b;

    let mut trace = generate_arithmetic_trace(&ops, &layout, num_rows)?;

    if let TraceColumn::B32(col) = &mut trace.columns[PHY_VAL_A] {
        col[num_rows - 1] = Block32::from(final_a).to_hardware();
    }

    if let TraceColumn::B32(col) = &mut trace.columns[PHY_VAL_B] {
        col[num_rows - 1] = Block32::from(final_b).to_hardware();
    }

    Ok((trace, final_b))
}

// =================================================================
// 3. MAIN EXECUTION
// =================================================================
fn main() {
    common::init("Fibonacci (integer, inlined arithmetic chiplet)");

    let num_vars: usize = 24;
    let num_rows: usize = 1 << num_vars;

    let mut config = Config {
        sumcheck_blinding_factor: 2,
        ..Config::default()
    };

    OsRng.try_fill_bytes(&mut config.matrix_seed).unwrap();

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    println!(
        "Trace: 2^{} rows ({} Fibonacci steps, single AIR)",
        num_vars,
        num_rows - 1
    );
    println!(
        "ZK Blinding: {} virtual columns",
        config.sumcheck_blinding_factor
    );

    let (trace, final_b) = common::phase("Trace Generation", || {
        generate_fib_trace(num_rows).expect("trace gen")
    });

    println!("   Public Input (Fib #{} mod 2^32): {}", num_rows, final_b);

    let instance = ProgramInstance::new(num_rows, vec![F::from(final_b as u128)]);
    let witness = ProgramWitness::new(trace);
    let program = FibChipletProgram::new(num_rows);

    let proof = common::phase("Proving", || {
        prove(
            b"FibonacciIntChiplet",
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

    let mut verifier_transcript = Transcript::<H>::new(b"FibonacciIntChiplet");

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
