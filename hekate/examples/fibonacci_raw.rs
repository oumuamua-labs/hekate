// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

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
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::define_columns;
use hekate_program::{FixedShape, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;
use rand::{TryRngCore, rngs::OsRng};
use zeroize::Zeroizing;

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

fn build_program(num_rows: usize) -> errors::Result<CircuitProgram<F>> {
    let mut cx = Circuit::<F>::new("FibonacciRaw", num_rows)?;

    let words = cx.expand_bits(4, ColumnType::B32);
    let packed = cx.reuse_pass_through(&words);

    let q = cx.column(ColumnType::Bit);

    let [a_packed, b_packed_col, sum_packed_col] = [packed.at(0), packed.at(1), packed.at(2)];

    let cs = cx.cs();

    let a_bits: Vec<_> = words.bits(0).iter().map(|c| cs.col(c.index())).collect();
    let b_bits: Vec<_> = words.bits(1).iter().map(|c| cs.col(c.index())).collect();
    let sum_bits: Vec<_> = words.bits(2).iter().map(|c| cs.col(c.index())).collect();
    let carry_v: Vec<_> = words.bits(3).iter().map(|c| cs.col(c.index())).collect();

    let zero = cs.constant(F::ZERO);

    let mut carry = Vec::with_capacity(33);
    carry.push(zero);
    carry.extend(carry_v.iter().copied());

    add_carry_chain_with_carry_in(cs, &a_bits, &b_bits, &sum_bits, &carry);

    let q_cell = cs.col(q.index());
    let b_packed = cs.col(b_packed_col.index());
    let sum_packed = cs.col(sum_packed_col.index());
    let next_a = cs.next(a_packed.index());
    let next_b = cs.next(b_packed_col.index());

    cs.constrain(q_cell * (next_a + b_packed));
    cs.constrain(q_cell * (next_b + sum_packed));

    cx.fix(q, FixedShape::LastRow);

    cx.boundary(a_packed, 0, F::ZERO);
    cx.boundary(b_packed_col, 0, F::ONE);

    cx.publish(b_packed_col, num_rows - 1);

    cx.compile()
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

    let num_vars = common::num_vars(24);
    let num_rows = 1 << num_vars;

    let config = Config {
        zero_knowledge: common::zero_knowledge(),
        ..Config::default()
    };

    let mut blinding_seed = Zeroizing::new([0u8; 32]);
    OsRng.try_fill_bytes(&mut *blinding_seed).unwrap();

    println!("Rows: 2^{} (~{} million)", num_vars, num_rows / 1_000_000);
    println!("Zero-knowledge: {}", config.zero_knowledge);

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
    let program = build_program(num_rows).expect("program build");

    let proof = common::phase("Proving", || {
        prove(
            b"FibonacciInt",
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

    let mut verifier_transcript = Transcript::<H>::new(b"FibonacciInt");

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
