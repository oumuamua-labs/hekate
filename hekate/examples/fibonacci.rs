// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

#[path = "common/mod.rs"]
mod common;

use hekate::core::trace::{ColumnTrace, TraceColumn};
use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::{Block32, Block128, HardwareField, TowerField};
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_gadgets::{
    ArithmeticOpcode, IntArithmeticChiplet, IntArithmeticLayout, IntArithmeticOp,
    generate_arithmetic_trace,
};
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::{FixedShape, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;
use rand::{TryRngCore, rngs::OsRng};
use zeroize::Zeroizing;

type F = Block128;
type H = DefaultHasher;

// =================================================================
// 1. INLINED ARITHMETIC CHIPLET - SINGLE TRACE
//
// 1:1 op:row makes a separate chiplet trace wasteful.
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

fn build_program(num_rows: usize) -> errors::Result<CircuitProgram<F>> {
    let chiplet = IntArithmeticChiplet::new(32, num_rows, num_rows - 1)?;
    let layout = chiplet.layout().clone();

    let mut cx = Circuit::<F>::new("FibonacciInt", num_rows)?;

    let arith = cx.mount_unlinked(ChipletDef::from_air(&chiplet)?);

    let cs = cx.cs();

    let s_add = cs.col(layout.s_add);
    let val_b = cs.col(layout.val_b);
    let val_res = cs.col(layout.val_res);
    let next_val_a = cs.next(layout.val_a);
    let next_val_b = cs.next(layout.val_b);

    cs.constrain(s_add * (next_val_a + val_b));
    cs.constrain(s_add * (next_val_b + val_res));

    cs.assert_zero_when(cs.one() + s_add, val_res);

    cx.boundary(arith.col(layout.val_a), 0, F::ZERO);
    cx.boundary(arith.col(layout.val_b), 0, F::ONE);

    cx.fix(arith.col(layout.s_add), FixedShape::LastRow);

    cx.publish(arith.col(layout.val_b), num_rows - 1);

    cx.compile()
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

    for _ in 0..num_rows - 1 {
        let sum = a.wrapping_add(b);
        ops.push(IntArithmeticOp::U32 {
            op: ArithmeticOpcode::ADD,
            a,
            b,
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

    let config = Config {
        zero_knowledge: common::zero_knowledge(),
        ..Config::default()
    };

    let mut blinding_seed = Zeroizing::new([0u8; 32]);
    OsRng.try_fill_bytes(&mut *blinding_seed).unwrap();

    println!(
        "Trace: 2^{} rows ({} Fibonacci steps, single AIR)",
        num_vars,
        num_rows - 1
    );
    println!("Zero-knowledge: {}", config.zero_knowledge);

    let (trace, final_b) = common::phase("Trace Generation", || {
        generate_fib_trace(num_rows).expect("trace gen")
    });

    println!("   Public Input (Fib #{} mod 2^32): {}", num_rows, final_b);

    let instance = ProgramInstance::new(num_rows, vec![F::from(final_b as u128)]);
    let witness = ProgramWitness::new(trace);
    let program = build_program(num_rows).expect("program build");

    let proof = common::phase("Proving", || {
        prove(
            b"FibonacciIntChiplet",
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

    let mut verifier_transcript = Transcript::<H>::new(b"FibonacciIntChiplet");

    let pinned_id = common::audited_id::<F, _>(&program);

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
