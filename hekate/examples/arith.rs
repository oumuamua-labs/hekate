// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

#[path = "common/mod.rs"]
mod common;

use hekate::core::trace::ColumnTrace;
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
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::{ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;
use rand::{TryRngCore, rngs::OsRng};
use zeroize::Zeroizing;

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

fn build_program(num_rows: usize, num_ops: usize) -> errors::Result<CircuitProgram<F>> {
    let chiplet = IntArithmeticChiplet::new(32, num_rows, num_ops)?;

    let mut cx = Circuit::<F>::new("ArithInline", num_rows)?;
    cx.mount_unlinked(ChipletDef::from_air(&chiplet)?);

    cx.compile()
}

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

        ops.push(IntArithmeticOp::U32 { op, a, b });
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

    let config = Config {
        zero_knowledge: common::zero_knowledge(),
        ..Config::default()
    };

    let mut blinding_seed = Zeroizing::new([0u8; 32]);
    OsRng.try_fill_bytes(&mut *blinding_seed).unwrap();

    println!(
        "Total Ops: {} (cycle through ADD, SUB, AND, XOR, NOT, LT)",
        num_ops
    );
    println!("Trace: 2^{} rows (single AIR)", num_vars);
    println!("Zero-knowledge: {}", config.zero_knowledge);

    let trace = common::phase("Trace Generation", || {
        generate_arith_trace(num_ops, num_rows).expect("arith trace")
    });

    let program = build_program(num_rows, num_ops).expect("program build");
    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(trace);

    let proof = common::phase("Proving", || {
        prove(
            b"Arith_Example",
            &program,
            &instance,
            &witness,
            &config,
            *blinding_seed,
            None,
        )
        .expect("Prover failed")
    });

    common::proof_breakdown(&proof);

    let mut verifier_transcript = Transcript::<H>::new(b"Arith_Example");

    let pinned_id = common::audited_id(&program);
    let is_valid = common::phase_with_mem("Verifying", || {
        HekateVerifier::<F, H>::verify(
            &pinned_id,
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
