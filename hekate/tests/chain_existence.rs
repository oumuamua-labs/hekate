// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Five witnesses that leave every gate low and still carry
//! a chosen public value into the cell the host reads. The
//! value pin accepts each one; the schedule pins reject them.

use hekate_core::config::Config;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder, TraceColumn};
use hekate_crypto::DefaultHasher;
use hekate_crypto::transcript::Transcript;
use hekate_gadgets::atoms::int_arith::add_carry_chain_with_carry_in;
use hekate_gadgets::{IntArithmeticChiplet, IntArithmeticLayout, generate_arithmetic_trace};
use hekate_keccak::{CpuKeccakColumns, KeccakChiplet, KeccakWitness, generate_keccak_trace};
use hekate_math::{Bit, Block32, Block64, Block128, HardwareField, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram, Col, ColRange};
use hekate_program::digest::program_id;
use hekate_program::{FixedShape, Program, ProgramInstance, ProgramWitness, define_columns};
use hekate_prover_sys::prove;
use hekate_sdk::preflight;
use hekate_verifier::HekateVerifier;

type F = Block128;
type H = DefaultHasher;

const DOMAIN: &[u8] = b"chain_existence";

const KECCAK_ROWS: usize = 256;
const FULL_BLOCKS: usize = KECCAK_ROWS / KeccakChiplet::BLOCK_ROWS;
const SHORT_BLOCKS: usize = 4;

const FIB_ROWS: usize = 32;
const FORGED_FIB: u32 = 0xDEAD_BEEF;

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

fn idle_cpu_trace(digest: [u64; 4], num_blocks: usize) -> ColumnTrace {
    let mut tb = keccak_cpu_builder();
    write_digest(&mut tb, digest, last_output_row(num_blocks));

    tb.build()
}

fn idle_chiplet_trace() -> ColumnTrace {
    let num_vars = KECCAK_ROWS.trailing_zeros() as usize;

    TraceBuilder::new(KeccakChiplet::physical_layout(), num_vars)
        .unwrap()
        .build()
}

fn short_chain_cpu_trace(input: [u64; 25], output: [u64; 25], digest: [u64; 4]) -> ColumnTrace {
    let mut tb = keccak_cpu_builder();

    for i in 0..25 {
        tb.set_b64(CpuKeccakColumns::LANES + i, 0, Block64(input[i]))
            .unwrap();
        tb.set_b64(
            CpuKeccakColumns::LANES + i,
            KeccakChiplet::BLOCK_ROWS - 1,
            Block64(output[i]),
        )
        .unwrap();
    }

    tb.set_bit(CpuKeccakColumns::SELECTOR, 0, Bit::ONE).unwrap();
    tb.set_bit(
        CpuKeccakColumns::SELECTOR,
        KeccakChiplet::BLOCK_ROWS - 1,
        Bit::ONE,
    )
    .unwrap();

    write_digest(&mut tb, digest, last_output_row(SHORT_BLOCKS));

    tb.build()
}

fn short_chain_chiplet_trace(input: [u64; 25]) -> ColumnTrace {
    let lanes: [Block64; 25] = core::array::from_fn(|i| Block64(input[i]));

    generate_keccak_trace(&[lanes], KECCAK_ROWS).unwrap()
}

// ===============================================
// Fibonacci hosts
// (fibonacci and fibonacci_raw, at probe height)
// ===============================================

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

define_columns! {
    FibRawPhys {
        A: B32,
        B: B32,
        SUM: B32,
        CARRY: B32,
        Q: Bit,
    }
}

fn fib_raw_host() -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("FibonacciRawProbe", FIB_ROWS).unwrap();

    let words = cx.expand_bits(4, ColumnType::B32);
    let packed = cx.reuse_pass_through(&words);
    let q = cx.column(ColumnType::Bit);

    let cs = cx.cs();

    let a_bits: Vec<_> = words.bits(0).iter().map(|c| cs.col(c.index())).collect();
    let b_bits: Vec<_> = words.bits(1).iter().map(|c| cs.col(c.index())).collect();
    let sum_bits: Vec<_> = words.bits(2).iter().map(|c| cs.col(c.index())).collect();
    let carry_v: Vec<_> = words.bits(3).iter().map(|c| cs.col(c.index())).collect();

    let mut carry = Vec::with_capacity(33);
    carry.push(cs.constant(F::ZERO));
    carry.extend(carry_v.iter().copied());

    add_carry_chain_with_carry_in(cs, &a_bits, &b_bits, &sum_bits, &carry);

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

    cx.compile().unwrap()
}

// ===============================================
// Fibonacci witnesses
// (no step taken, the answer written into the tail)
// ===============================================

/// `build_physical_layout` lays the operands
/// out as `val_a, val_b, val_res, opcode`.
const PHY_VAL_B: usize = 1;

fn idle_arith_trace() -> ColumnTrace {
    let layout = IntArithmeticLayout::compute(32);
    let mut trace = generate_arithmetic_trace(&[], &layout, FIB_ROWS).unwrap();

    match &mut trace.columns[PHY_VAL_B] {
        TraceColumn::B32(col) => col[FIB_ROWS - 1] = Block32::from(FORGED_FIB).to_hardware(),
        _ => panic!("val_b is not a B32 column"),
    }

    trace
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

/// The read row holds the chosen value and the origin
/// pin plus the schedule pins are what stand against it.
fn assert_read_row_holds_the_forgery<P: Program<F>>(
    air: &P,
    instance: &ProgramInstance<F>,
    witness: &ProgramWitness<F>,
) {
    let report = preflight(air, instance, witness).unwrap();

    assert!(
        report.boundary_violations.iter().all(|v| v.row_idx == 0),
        "{report}"
    );
    assert!(!report.fixed_column_violations.is_empty(), "{report}");
}

fn assert_unproven<P: Program<F>>(
    air: &P,
    instance: &ProgramInstance<F>,
    witness: &ProgramWitness<F>,
) {
    let config = Config {
        zero_knowledge: true,
        ..Config::dev()
    };

    let proof = prove(DOMAIN, air, instance, witness, &config, [0x5Au8; 32], None)
        .expect("a forger runs a prover that does not refuse");

    let mut vt = Transcript::<H>::new(DOMAIN);
    let pinned_id = program_id(air).unwrap();

    let verdict =
        HekateVerifier::<F, H>::verify(&pinned_id, air, instance, &proof, &mut vt, &config);

    assert!(!matches!(verdict, Ok(true)));
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

fn write_digest(tb: &mut TraceBuilder, digest: [u64; 4], row: usize) {
    for (i, &lane) in digest.iter().enumerate() {
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

    let report = preflight(&air, &instance, &witness).unwrap();

    assert!(report.boundary_violations.is_empty(), "{report}");
    assert!(!report.fixed_column_violations.is_empty(), "{report}");

    assert_unproven(&air, &instance, &witness);
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn keccak_independent_rejects_empty_trace() {
    let digest = chosen_digest();
    let air = independent_host(FULL_BLOCKS);
    let instance = ProgramInstance::new(KECCAK_ROWS, digest_public_inputs(digest));
    let witness = ProgramWitness::new(idle_cpu_trace(digest, FULL_BLOCKS))
        .with_chiplets(vec![idle_chiplet_trace()]);

    let report = preflight(&air, &instance, &witness).unwrap();

    assert!(report.boundary_violations.is_empty(), "{report}");
    assert!(!report.fixed_column_violations.is_empty(), "{report}");

    assert_unproven(&air, &instance, &witness);
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
        short_chain_cpu_trace(input, output, digest),
        short_chain_chiplet_trace(input),
    ));

    let report = preflight(&air, &instance, &witness).unwrap();

    assert!(report.boundary_violations.is_empty(), "{report}");
    assert!(!report.fixed_column_violations.is_empty(), "{report}");

    assert_unproven(&air, &instance, &witness);
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn fibonacci_rejects_idle_trace() {
    let air = fib_host();
    let instance = ProgramInstance::new(FIB_ROWS, vec![F::from(FORGED_FIB as u128)]);
    let witness = ProgramWitness::new(idle_arith_trace());

    assert_read_row_holds_the_forgery(&air, &instance, &witness);
    assert_unproven(&air, &instance, &witness);
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn fibonacci_raw_rejects_idle_trace() {
    let air = fib_raw_host();
    let instance = ProgramInstance::new(FIB_ROWS, vec![F::from(FORGED_FIB as u128)]);
    let witness = ProgramWitness::new(idle_fib_raw_trace());

    assert_read_row_holds_the_forgery(&air, &instance, &witness);
    assert_unproven(&air, &instance, &witness);
}
