// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Five witnesses that leave every gate low and still carry
//! a chosen public value into the cell the host reads. The
//! value pin accepts each one; the schedule pins reject them.

use hekate_core::config::Config;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate_gadgets::atoms::int_arith::add_carry_chain;
use hekate_gadgets::{
    ArithmeticOpcode, IntArithmeticChiplet, IntArithmeticLayout, IntArithmeticOp,
    generate_arithmetic_trace,
};
use hekate_keccak::{CpuKeccakColumns, KeccakChiplet, KeccakWitness, generate_keccak_trace};
use hekate_math::{Bit, Block32, Block64, Block256, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram, Col, ColRange};
use hekate_program::{FixedShape, ProgramInstance, ProgramWitness, define_columns};
use hekate_sdk::preflight::TableId;
use hekate_testkit::cell::set_b32;
use hekate_testkit::fault::{Check, Fault};
use hekate_testkit::verdict::assert_rejected;

type F = Block256;

const KECCAK_ROWS: usize = 256;
const FULL_BLOCKS: usize = KECCAK_ROWS / KeccakChiplet::BLOCK_ROWS;
const SHORT_BLOCKS: usize = 4;

const FIB_ROWS: usize = 32;
const FORGED_FIB: u32 = 0xDEAD_BEEF;

/// `build_physical_layout` lays the operands
/// out as `val_a, val_b, val_res, opcode`.
const PHY_VAL_A: usize = 0;
const PHY_VAL_B: usize = 1;

// ===============================================
// Keccak hosts
// ===============================================

fn inline_host(num_blocks: usize) -> CircuitProgram<F> {
    let (mut cx, cpu) = keccak_cpu_side("KeccakInlineProbe", num_blocks);

    cx.mount(ChipletDef::from_air(&KeccakChiplet::new(KECCAK_ROWS, num_blocks)).unwrap());

    publish_digest(&mut cx, cpu, num_blocks);

    cx.compile().unwrap()
}

fn independent_host(num_blocks: usize) -> CircuitProgram<F> {
    let (mut cx, cpu) = keccak_cpu_side("KeccakIsolatedProbe", num_blocks);

    cx.attach(ChipletDef::from_air(&KeccakChiplet::new(KECCAK_ROWS, num_blocks)).unwrap());

    publish_digest(&mut cx, cpu, num_blocks);

    cx.compile().unwrap()
}

fn keccak_cpu_side(name: &str, num_blocks: usize) -> (Circuit<F>, ColRange) {
    let mut cx = Circuit::<F>::new(name, KECCAK_ROWS).unwrap();
    let cpu = cx.schema(&CpuKeccakColumns::build_layout());

    let selector = cpu.at(CpuKeccakColumns::SELECTOR);

    let values: Vec<Col> = (0..25)
        .map(|lane| cpu.at(CpuKeccakColumns::LANES + lane))
        .collect();

    cx.call(&KeccakChiplet::service(), &values, selector)
        .unwrap();

    cx.fix(
        selector,
        KeccakChiplet::host_selector_shape(KeccakChiplet::BLOCK_ROWS, num_blocks),
    );

    (cx, cpu)
}

fn publish_digest(cx: &mut Circuit<F>, cpu: ColRange, num_blocks: usize) {
    for i in 0..4 {
        cx.publish(
            cpu.at(CpuKeccakColumns::LANES + i),
            last_output_row(num_blocks),
        );
    }
}

// ===============================================
// Keccak witnesses
// (no block, and one block where four are declared)
// ===============================================

fn keccak_chain(num_blocks: usize) -> (ColumnTrace, ColumnTrace, [u64; 4]) {
    let mut inputs = Vec::with_capacity(num_blocks);
    let mut state = probe_input();

    for _ in 0..num_blocks {
        inputs.push(state);

        state = keccak_f(state);
    }

    let digest = [state[0], state[1], state[2], state[3]];

    (cpu_blocks(&inputs).build(), chiplet_blocks(&inputs), digest)
}

fn idle_cpu_trace(digest: [u64; 4], num_blocks: usize) -> ColumnTrace {
    let mut tb = keccak_cpu_builder();
    write_lanes(&mut tb, &digest, last_output_row(num_blocks));

    tb.build()
}

fn idle_chiplet_trace() -> ColumnTrace {
    let num_vars = KECCAK_ROWS.trailing_zeros() as usize;

    TraceBuilder::new(KeccakChiplet::physical_layout(), num_vars)
        .unwrap()
        .build()
}

fn short_chain_cpu_trace(input: [u64; 25], digest: [u64; 4]) -> ColumnTrace {
    let mut tb = cpu_blocks(&[input]);
    write_lanes(&mut tb, &digest, last_output_row(SHORT_BLOCKS));

    tb.build()
}

fn cpu_blocks(inputs: &[[u64; 25]]) -> TraceBuilder {
    let mut tb = keccak_cpu_builder();
    for (block, &input) in inputs.iter().enumerate() {
        let first = block * KeccakChiplet::BLOCK_ROWS;
        let last = first + KeccakChiplet::BLOCK_ROWS - 1;

        write_lanes(&mut tb, &input, first);
        write_lanes(&mut tb, &keccak_f(input), last);

        tb.set_bit(CpuKeccakColumns::SELECTOR, first, Bit::ONE)
            .unwrap();
        tb.set_bit(CpuKeccakColumns::SELECTOR, last, Bit::ONE)
            .unwrap();
    }

    tb
}

fn chiplet_blocks(inputs: &[[u64; 25]]) -> ColumnTrace {
    let calls: Vec<[Block64; 25]> = inputs.iter().map(|input| input.map(Block64)).collect();

    generate_keccak_trace(&calls, KECCAK_ROWS).unwrap()
}

// ===============================================
// Fibonacci hosts
// (fibonacci and fibonacci_raw, at probe height)
// ===============================================

define_columns! {
    FibRawPhys {
        A: B32,
        B: B32,
        SUM: B32,
        CARRY: B32,
        Q: Bit,
    }
}

struct FibRawHost {
    program: CircuitProgram<F>,
    b: usize,
    q: usize,
}

fn fib_host() -> CircuitProgram<F> {
    let chiplet = IntArithmeticChiplet::new(32, FIB_ROWS, FIB_ROWS - 1).unwrap();
    let layout = chiplet.layout().clone();

    let mut cx = Circuit::<F>::new("FibonacciProbe", FIB_ROWS).unwrap();
    let arith = cx.mount_unlinked(ChipletDef::from_air(&chiplet).unwrap());

    let cs = cx.cs();

    let s_add = cs.col(layout.s_add);
    let val_b = cs.col(layout.val_b);
    let val_res = cs.col(layout.val_res);
    let next_val_a = cs.next(layout.val_a);
    let next_val_b = cs.next(layout.val_b);

    cs.constrain(s_add * (next_val_a + val_b));
    cs.constrain(s_add * (next_val_b + val_res));
    cs.assert_zero_when(cs.one() + s_add, val_res);

    cx.fix(arith.col(layout.s_add), FixedShape::LastRow);

    cx.boundary(arith.col(layout.val_a), 0, F::ZERO);
    cx.boundary(arith.col(layout.val_b), 0, F::ONE);

    cx.publish(arith.col(layout.val_b), FIB_ROWS - 1);

    cx.compile().unwrap()
}

fn fib_raw_host() -> FibRawHost {
    let mut cx = Circuit::<F>::new("FibonacciRawProbe", FIB_ROWS).unwrap();

    let words = cx.expand_bits(4, ColumnType::B32);
    let packed = cx.reuse_pass_through(&words);
    let q = cx.column(ColumnType::Bit);

    let cs = cx.cs();

    let a_bits: Vec<_> = words.bits(0).iter().map(|c| cs.col(c.index())).collect();
    let b_bits: Vec<_> = words.bits(1).iter().map(|c| cs.col(c.index())).collect();
    let sum_bits: Vec<_> = words.bits(2).iter().map(|c| cs.col(c.index())).collect();
    let carry_v: Vec<_> = words.bits(3).iter().map(|c| cs.col(c.index())).collect();

    add_carry_chain(cs, &a_bits, &b_bits, &sum_bits, &carry_v);

    let q_cell = cs.col(q.index());
    let b_packed = cs.col(packed.at(1).index());
    let sum_packed = cs.col(packed.at(2).index());
    let next_a = cs.next(packed.at(0).index());
    let next_b = cs.next(packed.at(1).index());

    cs.constrain(q_cell * (next_a + b_packed));
    cs.constrain(q_cell * (next_b + sum_packed));

    cx.fix(q, FixedShape::LastRow);
    cx.boundary(packed.at(0), 0, F::ZERO);
    cx.boundary(packed.at(1), 0, F::ONE);
    cx.publish(packed.at(1), FIB_ROWS - 1);

    FibRawHost {
        b: packed.at(1).index(),
        q: q.index(),
        program: cx.compile().unwrap(),
    }
}

// ===============================================
// Fibonacci witnesses
// (no step taken, the answer written into the tail)
// ===============================================

fn fib_arith_trace() -> (ColumnTrace, u32) {
    let layout = IntArithmeticLayout::compute(32);

    let mut ops = Vec::with_capacity(FIB_ROWS - 1);
    let (mut a, mut b) = (0u32, 1u32);

    for _ in 0..FIB_ROWS - 1 {
        ops.push(IntArithmeticOp::U32 {
            op: ArithmeticOpcode::ADD,
            a,
            b,
        });

        (a, b) = (b, a.wrapping_add(b));
    }

    let mut trace = generate_arithmetic_trace(&ops, &layout, FIB_ROWS).unwrap();
    set_b32(&mut trace, PHY_VAL_A, FIB_ROWS - 1, a);
    set_b32(&mut trace, PHY_VAL_B, FIB_ROWS - 1, b);

    (trace, b)
}

fn idle_arith_trace() -> ColumnTrace {
    let layout = IntArithmeticLayout::compute(32);
    let mut trace = generate_arithmetic_trace(&[], &layout, FIB_ROWS).unwrap();
    set_b32(&mut trace, PHY_VAL_B, FIB_ROWS - 1, FORGED_FIB);

    trace
}

fn fib_raw_trace() -> (ColumnTrace, u32) {
    let num_vars = FIB_ROWS.trailing_zeros() as usize;

    let mut tb = TraceBuilder::new(&FibRawPhys::build_layout(), num_vars).unwrap();
    let (mut a, mut b) = (0u32, 1u32);

    for row in 0..FIB_ROWS {
        let sum = a.wrapping_add(b);
        let carry = (a & b) | ((a ^ b) & !sum);

        tb.set_b32(FibRawPhys::A, row, Block32::from(a)).unwrap();
        tb.set_b32(FibRawPhys::B, row, Block32::from(b)).unwrap();
        tb.set_b32(FibRawPhys::SUM, row, Block32::from(sum))
            .unwrap();
        tb.set_b32(FibRawPhys::CARRY, row, Block32::from(carry))
            .unwrap();

        (a, b) = (b, sum);
    }

    tb.fill_selector(FibRawPhys::Q, FIB_ROWS - 1).unwrap();

    (tb.build(), a)
}

/// The carry chain is ungated, and with `A` zero
/// it forces `SUM = B` on the read row as well.
fn idle_fib_raw_trace() -> ColumnTrace {
    let num_vars = FIB_ROWS.trailing_zeros() as usize;
    let mut tb = TraceBuilder::new(&FibRawPhys::build_layout(), num_vars).unwrap();

    tb.set_b32(FibRawPhys::B, FIB_ROWS - 1, Block32::from(FORGED_FIB))
        .unwrap();
    tb.set_b32(FibRawPhys::SUM, FIB_ROWS - 1, Block32::from(FORGED_FIB))
        .unwrap();

    tb.build()
}

// ===============================================
// Verdict oracles
// ===============================================

fn config() -> Config {
    Config {
        num_queries: 2,
        fold_arity_log: 1,
        early_stop_vars: 1,
        ..Config::dev()
    }
}

fn keccak_cpu_builder() -> TraceBuilder {
    let num_vars = KECCAK_ROWS.trailing_zeros() as usize;

    TraceBuilder::new(&CpuKeccakColumns::build_layout(), num_vars).unwrap()
}

fn merge_inline(cpu: ColumnTrace, chiplet: ColumnTrace) -> ColumnTrace {
    let mut trace = cpu;
    for col in chiplet.into_columns() {
        trace.add_column(col).unwrap();
    }

    trace
}

fn write_lanes(tb: &mut TraceBuilder, lanes: &[u64], row: usize) {
    for (i, &lane) in lanes.iter().enumerate() {
        tb.set_b64(CpuKeccakColumns::LANES + i, row, Block64(lane))
            .unwrap();
    }
}

fn last_output_row(num_blocks: usize) -> usize {
    KeccakChiplet::BLOCK_ROWS * num_blocks - 1
}

fn chosen_digest() -> [u64; 4] {
    core::array::from_fn(|i| 0xDEAD_BEEF_0000_0000u64 | i as u64)
}

fn digest_public_inputs(digest: [u64; 4]) -> Vec<F> {
    digest.iter().map(|&lane| F::from(Block64(lane))).collect()
}

fn probe_input() -> [u64; 25] {
    core::array::from_fn(|i| (i as u64).wrapping_mul(0x9E37_79B9) | 1)
}

fn keccak_f(mut state: [u64; 25]) -> [u64; 25] {
    for rc in KeccakChiplet::ROUND_CONSTANTS {
        state = KeccakWitness::keccak_f_round(state, rc);
    }

    state
}

// ===============================================
// Probes
// ===============================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn keccak_inline_rejects_empty_trace() {
    let digest = chosen_digest();
    let air = inline_host(FULL_BLOCKS);
    let instance = ProgramInstance::new(KECCAK_ROWS, digest_public_inputs(digest));
    let witness = ProgramWitness::new(merge_inline(
        idle_cpu_trace(digest, FULL_BLOCKS),
        idle_chiplet_trace(),
    ));

    let (cpu, chiplet, true_digest) = keccak_chain(FULL_BLOCKS);
    let honest_instance = ProgramInstance::new(KECCAK_ROWS, digest_public_inputs(true_digest));
    let honest = ProgramWitness::new(merge_inline(cpu, chiplet));

    let fault = Fault {
        constraint_tables: &[TableId::Main],
        pin_tables: &[TableId::Main],
        buses: &[KeccakChiplet::BUS_ID],
        first_check: Some(Check::FixedColumn),
        ..Fault::NONE
    };

    assert_rejected(
        &air,
        (&honest_instance, &honest),
        (&instance, &witness),
        fault,
        &config(),
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn keccak_independent_rejects_empty_trace() {
    let digest = chosen_digest();
    let air = independent_host(FULL_BLOCKS);
    let instance = ProgramInstance::new(KECCAK_ROWS, digest_public_inputs(digest));
    let witness = ProgramWitness::new(idle_cpu_trace(digest, FULL_BLOCKS))
        .with_chiplets(vec![idle_chiplet_trace()]);

    let (cpu, chiplet, true_digest) = keccak_chain(FULL_BLOCKS);
    let honest_instance = ProgramInstance::new(KECCAK_ROWS, digest_public_inputs(true_digest));
    let honest = ProgramWitness::new(cpu).with_chiplets(vec![chiplet]);

    let fault = Fault {
        constraint_tables: &[TableId::Chiplet(0)],
        pin_tables: &[TableId::Main, TableId::Chiplet(0)],
        buses: &[KeccakChiplet::BUS_ID],
        first_check: Some(Check::FixedColumn),
        ..Fault::NONE
    };

    assert_rejected(
        &air,
        (&honest_instance, &honest),
        (&instance, &witness),
        fault,
        &config(),
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn keccak_inline_rejects_short_chain() {
    let input = probe_input();
    let output = keccak_f(input);
    let digest = chosen_digest();

    assert_ne!(digest, [output[0], output[1], output[2], output[3]]);

    let air = inline_host(SHORT_BLOCKS);
    let instance = ProgramInstance::new(KECCAK_ROWS, digest_public_inputs(digest));
    let witness = ProgramWitness::new(merge_inline(
        short_chain_cpu_trace(input, digest),
        chiplet_blocks(&[input]),
    ));

    let (cpu, chiplet, true_digest) = keccak_chain(SHORT_BLOCKS);
    let honest_instance = ProgramInstance::new(KECCAK_ROWS, digest_public_inputs(true_digest));
    let honest = ProgramWitness::new(merge_inline(cpu, chiplet));

    let fault = Fault {
        constraint_tables: &[TableId::Main],
        pin_tables: &[TableId::Main],
        first_check: Some(Check::FixedColumn),
        ..Fault::NONE
    };

    assert_rejected(
        &air,
        (&honest_instance, &honest),
        (&instance, &witness),
        fault,
        &config(),
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn fibonacci_rejects_idle_trace() {
    let air = fib_host();
    let layout = IntArithmeticLayout::compute(32);
    let instance = ProgramInstance::new(FIB_ROWS, vec![F::from(FORGED_FIB as u128)]);
    let witness = ProgramWitness::new(idle_arith_trace());

    let (trace, fib) = fib_arith_trace();
    let honest_instance = ProgramInstance::new(FIB_ROWS, vec![F::from(fib as u128)]);
    let honest = ProgramWitness::new(trace);

    let fault = Fault {
        constraint_tables: &[TableId::Main],
        boundary_columns: &[(TableId::Main, layout.val_b)],
        pin_columns: &[
            (TableId::Main, layout.s_add),
            (TableId::Main, layout.s_output),
        ],
        first_check: Some(Check::Sumcheck),
        ..Fault::NONE
    };

    assert_rejected(
        &air,
        (&honest_instance, &honest),
        (&instance, &witness),
        fault,
        &config(),
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn fibonacci_raw_rejects_idle_trace() {
    let host = fib_raw_host();
    let instance = ProgramInstance::new(FIB_ROWS, vec![F::from(FORGED_FIB as u128)]);
    let witness = ProgramWitness::new(idle_fib_raw_trace());

    let (trace, fib) = fib_raw_trace();
    let honest_instance = ProgramInstance::new(FIB_ROWS, vec![F::from(fib as u128)]);
    let honest = ProgramWitness::new(trace);

    let fault = Fault {
        constraint_tables: &[TableId::Main],
        boundary_columns: &[(TableId::Main, host.b)],
        pin_columns: &[(TableId::Main, host.q)],
        first_check: Some(Check::Sumcheck),
        ..Fault::NONE
    };

    assert_rejected(
        &host.program,
        (&honest_instance, &honest),
        (&instance, &witness),
        fault,
        &config(),
    );
}
