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
use hekate::math::{Block32, Block128, TowerField};
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::trace::{Trace, TraceBuilder};
use hekate_crypto::transcript::Transcript;
use hekate_gadgets::atoms::int_arith::add_carry_chain_with_carry_in;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::constraint::{BoundaryConstraint, ConstraintAst};
use hekate_program::define_columns;
use hekate_program::expander::VirtualExpander;
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;
use rand::{TryRngCore, rngs::OsRng};
use std::sync::OnceLock;

// =================================================================
// 1. CONFIGURATION
// =================================================================
type F = Block128;
type H = DefaultHasher;

// =================================================================
// 2. INTEGER FIBONACCI AIR DEFINITION
//
// Real 32-bit integer Fibonacci over GF(2^k):
// the carry chain is emulated bit-by-bit via
// `add_carry_chain_with_carry_in`. Four B32
// physical columns (A, B, SUM, CARRY) expand
// virtually into 128 bit slots the adder operates
// on; the same four expose a packed view for
// transition equalities.
//
// Total:
// 17 bytes/row, 133 virtual cols.
// =================================================================
define_columns! {
    FibIntPhys {
        A: B32,
        B: B32,
        SUM: B32,
        CARRY: B32,
        Q: Bit,
    }
}

define_columns! {
    FibIntVirt {
        A_BITS: [Bit; 32],
        B_BITS: [Bit; 32],
        SUM_BITS: [Bit; 32],
        CARRY_BITS: [Bit; 32],
        A_PACKED: B32,
        B_PACKED: B32,
        SUM_PACKED: B32,
        CARRY_PACKED: B32,
        Q: Bit,
    }
}

#[derive(Clone)]
struct FibIntProgram {
    num_rows: usize,
}

impl Air<F> for FibIntProgram {
    fn boundary_constraints(&self) -> Vec<BoundaryConstraint<F>> {
        vec![BoundaryConstraint::with_public_input(
            FibIntVirt::B_PACKED,
            self.num_rows - 1,
            0,
        )]
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: OnceLock<Vec<ColumnType>> = OnceLock::new();
        LAYOUT.get_or_init(FibIntPhys::build_layout)
    }

    fn virtual_expander(&self) -> Option<&VirtualExpander> {
        static E: OnceLock<VirtualExpander> = OnceLock::new();
        Some(E.get_or_init(|| {
            VirtualExpander::new()
                .expand_bits(4, ColumnType::B32)
                .reuse_pass_through(0, 4)
                .control_bits(1)
                .build()
                .expect("FibIntProgram expander")
        }))
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();

        let a_bits: Vec<_> = (0..32).map(|i| cs.col(FibIntVirt::A_BITS + i)).collect();
        let b_bits: Vec<_> = (0..32).map(|i| cs.col(FibIntVirt::B_BITS + i)).collect();
        let sum_bits: Vec<_> = (0..32).map(|i| cs.col(FibIntVirt::SUM_BITS + i)).collect();
        let carry_v: Vec<_> = (0..32)
            .map(|i| cs.col(FibIntVirt::CARRY_BITS + i))
            .collect();

        let zero = cs.constant(F::ZERO);

        let mut carry = Vec::with_capacity(33);
        carry.push(zero);
        carry.extend(carry_v.iter().copied());

        add_carry_chain_with_carry_in(&cs, &a_bits, &b_bits, &sum_bits, &carry);

        let q = cs.col(FibIntVirt::Q);
        let b_packed = cs.col(FibIntVirt::B_PACKED);
        let sum_packed = cs.col(FibIntVirt::SUM_PACKED);
        let next_a = cs.next(FibIntVirt::A_PACKED);
        let next_b = cs.next(FibIntVirt::B_PACKED);

        cs.constrain(q * (next_a + b_packed));
        cs.constrain(q * (next_b + sum_packed));

        cs.build()
    }
}

impl Program<F> for FibIntProgram {
    fn num_public_inputs(&self) -> usize {
        1
    }
}

// =================================================================
// 3. TRACE GENERATION
//
// carry_word layout:
// bit k = adder's carry[k+1]. Bit 31 is the
// overflow (adder's carry[32]), constrained
// by the adder but otherwise unused.
// =================================================================
fn generate_fib_trace(num_vars: usize) -> errors::Result<ColumnTrace> {
    let num_rows = 1 << num_vars;
    let mut tb = TraceBuilder::new(&FibIntPhys::build_layout(), num_vars)?;

    let mut a: u32 = 0;
    let mut b: u32 = 1;

    for i in 0..num_rows {
        let mut c: u32 = 0;
        let mut sum: u32 = 0;
        let mut carry_word: u32 = 0;

        for k in 0..32 {
            let a_k = (a >> k) & 1;
            let b_k = (b >> k) & 1;
            let s_k = a_k ^ b_k ^ c;
            let c_next = (a_k & b_k) | (c & (a_k ^ b_k));

            sum |= s_k << k;
            carry_word |= c_next << k;
            c = c_next;
        }

        tb.set_b32(FibIntPhys::A, i, Block32::from(a))?;
        tb.set_b32(FibIntPhys::B, i, Block32::from(b))?;
        tb.set_b32(FibIntPhys::SUM, i, Block32::from(sum))?;
        tb.set_b32(FibIntPhys::CARRY, i, Block32::from(carry_word))?;

        a = b;
        b = sum;
    }

    tb.fill_selector(FibIntPhys::Q, num_rows - 1)?;

    Ok(tb.build())
}

// =================================================================
// 4. MAIN EXECUTION
// =================================================================
fn main() {
    common::init("Fibonacci (integer, 32-bit)");

    let num_vars: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);

    let num_rows = 1 << num_vars;

    let mut config = Config {
        sumcheck_blinding_factor: 0,
        ..Config::default()
    };

    OsRng.try_fill_bytes(&mut config.matrix_seed).unwrap();

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    println!("Rows: 2^{} (~{} million)", num_vars, num_rows / 1_000_000);
    println!(
        "ZK Blinding: {} virtual columns",
        config.sumcheck_blinding_factor
    );

    let trace = common::phase("Trace Generation", || generate_fib_trace(num_vars).unwrap());

    let expected_result = trace
        .get_element(FibIntPhys::B, num_rows - 1)
        .unwrap()
        .to_tower();
    println!(
        "   Public Input (Fib #{} mod 2^32): {:?}",
        num_rows, expected_result
    );

    let instance = ProgramInstance::new(num_rows, vec![expected_result]);
    let witness = ProgramWitness::new(trace);
    let program = FibIntProgram { num_rows };

    let proof = common::phase("Proving", || {
        prove(
            b"FibonacciInt",
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

    let mut verifier_transcript = Transcript::<H>::new(b"FibonacciInt");

    let is_valid = common::phase_with_mem("Verifying", || {
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
