// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::config::Config;
use hekate_core::trace::{ColumnTrace, ColumnType, Trace, TraceBuilder, TraceColumn};
use hekate_gadgets::{
    ArithmeticOpcode, CpuArithColumns, IntArithmeticChiplet, IntArithmeticLayout, IntArithmeticOp,
    generate_arithmetic_trace,
};
use hekate_math::{Bit, Block32, Block64, Block256, HardwareField, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram, Col};
use hekate_program::{FixedShape, ProgramInstance, ProgramWitness, define_columns};
use hekate_scribble::{MutationKind, ScribbleConfig, assert_all_caught_all_targets};
use hekate_sdk::preflight::TableId;
use hekate_testkit::cell::{set_b32, set_bit};
use hekate_testkit::fault::Fault;
use hekate_testkit::verdict::{assert_rejected, assert_verifies};

type F = Block256;

// =================================================================
// 0. PHYSICAL COLUMN INDICES
// =================================================================

const PHY_VAL_RES: usize = 2;
const PHY_CARRY: usize = 3;
const PHY_OPCODE: usize = 4;
const PHY_S_SUB: usize = 7;
const PHY_S_XOR: usize = 9;

const BUS_ONLY: Fault<'static> = Fault {
    buses: &[IntArithmeticChiplet::BUS_ID],
    ..Fault::NONE
};

// =================================================================
// 1. SINGLE-CHIPLET PROGRAM (u32 OR u64)
// =================================================================

fn arith_cpu_layout(bit_width: usize) -> Vec<ColumnType> {
    let operand = if bit_width == 64 {
        ColumnType::B64
    } else {
        ColumnType::B32
    };

    vec![operand, operand, operand, ColumnType::B32, ColumnType::Bit]
}

fn arith_cpu_program(bit_width: usize, arith_num_rows: usize, num_ops: usize) -> CircuitProgram<F> {
    let chiplet = IntArithmeticChiplet::new(bit_width, arith_num_rows, num_ops)
        .expect("IntArithmeticChiplet::new in test program");

    arith_cpu_program_attaching(
        bit_width,
        arith_num_rows,
        num_ops,
        ChipletDef::from_air(&chiplet).unwrap(),
    )
}

fn arith_cpu_program_attaching(
    bit_width: usize,
    arith_num_rows: usize,
    num_ops: usize,
    arith: ChipletDef<F>,
) -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("ArithCpu", arith_num_rows).unwrap();
    let cpu = cx.schema(&arith_cpu_layout(bit_width));

    cx.fix(
        cpu.at(CpuArithColumns::SELECTOR),
        FixedShape::Cadence {
            stride: 1,
            count: num_ops,
            origin: 0,
            values: vec![F::ONE],
        },
    );

    cx.bus(
        IntArithmeticChiplet::BUS_ID,
        IntArithmeticChiplet::cpu_linking_spec(),
    );

    let cs = cx.cs();

    let selector = cs.col(CpuArithColumns::SELECTOR);
    let not_active = cs.one() + selector;

    cs.assert_zero_when(not_active, cs.col(CpuArithColumns::VAL_A));
    cs.assert_zero_when(not_active, cs.col(CpuArithColumns::VAL_B));
    cs.assert_zero_when(not_active, cs.col(CpuArithColumns::VAL_RES));
    cs.assert_zero_when(not_active, cs.col(CpuArithColumns::OPCODE));

    cx.attach(arith);
    cx.compile().unwrap()
}

fn composite_arith_def(arith_num_rows: usize, num_ops: usize, extra: bool) -> ChipletDef<F> {
    let chiplet = IntArithmeticChiplet::new(32, arith_num_rows, num_ops).unwrap();

    let mut cx = Circuit::<F>::new("ArithComposite", arith_num_rows).unwrap();
    cx.mount(ChipletDef::from_air(&chiplet).unwrap());

    if extra {
        let col = cx.column(ColumnType::B32);

        let cs = cx.cs();
        cs.constrain(cs.col(col.index()) * cs.col(col.index()));
    }

    ChipletDef::from_air(&cx.compile().unwrap()).unwrap()
}

// =================================================================
// 2. MIXED-WIDTH PROGRAM
// =================================================================

const MIXED_BUS_U32: &str = "arith_link_u32";
const MIXED_BUS_U64: &str = "arith_link_u64";

define_columns! {
    MixedArithColumns {
        U32_VAL_A: B32,
        U32_VAL_B: B32,
        U32_VAL_RES: B32,
        U32_OPCODE: B32,
        U32_SELECTOR: Bit,
        U64_VAL_A: B64,
        U64_VAL_B: B64,
        U64_VAL_RES: B64,
        U64_OPCODE: B32,
        U64_SELECTOR: Bit,
    }
}

fn mixed_arith_program(
    cpu_rows: usize,
    chip32_rows: usize,
    chip64_rows: usize,
    num_ops32: usize,
    num_ops64: usize,
) -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("MixedArith", cpu_rows).unwrap();
    let cpu = cx.schema(&MixedArithColumns::build_layout());

    let ops_prefix = |count: usize| FixedShape::Cadence {
        stride: 1,
        count,
        origin: 0,
        values: vec![F::ONE],
    };

    cx.fix(
        cpu.at(MixedArithColumns::U32_SELECTOR),
        ops_prefix(num_ops32),
    );
    cx.fix(
        cpu.at(MixedArithColumns::U64_SELECTOR),
        ops_prefix(num_ops64),
    );

    let u32_values: Vec<Col> = [
        MixedArithColumns::U32_VAL_A,
        MixedArithColumns::U32_VAL_B,
        MixedArithColumns::U32_VAL_RES,
        MixedArithColumns::U32_OPCODE,
    ]
    .map(|c| cpu.at(c))
    .to_vec();

    let u64_values: Vec<Col> = [
        MixedArithColumns::U64_VAL_A,
        MixedArithColumns::U64_VAL_B,
        MixedArithColumns::U64_VAL_RES,
        MixedArithColumns::U64_OPCODE,
    ]
    .map(|c| cpu.at(c))
    .to_vec();

    let mut u32_service = IntArithmeticChiplet::service();
    u32_service.bus_id = MIXED_BUS_U32;

    let mut u64_service = IntArithmeticChiplet::service();
    u64_service.bus_id = MIXED_BUS_U64;

    cx.call(
        &u32_service,
        &u32_values,
        cpu.at(MixedArithColumns::U32_SELECTOR),
    )
    .unwrap();
    cx.call(
        &u64_service,
        &u64_values,
        cpu.at(MixedArithColumns::U64_SELECTOR),
    )
    .unwrap();

    let chip32 = IntArithmeticChiplet::new(32, chip32_rows, num_ops32)
        .expect("u32 chiplet")
        .with_bus_id(MIXED_BUS_U32);
    let chip64 = IntArithmeticChiplet::new(64, chip64_rows, num_ops64)
        .expect("u64 chiplet")
        .with_bus_id(MIXED_BUS_U64);

    cx.attach(ChipletDef::from_air(&chip32).unwrap());
    cx.attach(ChipletDef::from_air(&chip64).unwrap());

    cx.compile().unwrap()
}

// =================================================================
// 3. HELPERS
// =================================================================

fn compute_u32(op: ArithmeticOpcode, a: u32, b: u32) -> u32 {
    match op {
        ArithmeticOpcode::ADD => a.wrapping_add(b),
        ArithmeticOpcode::SUB => a.wrapping_sub(b),
        ArithmeticOpcode::AND => a & b,
        ArithmeticOpcode::XOR => a ^ b,
        ArithmeticOpcode::NOT => !a,
        ArithmeticOpcode::LT => (a < b) as u32,
    }
}

fn compute_u64(op: ArithmeticOpcode, a: u64, b: u64) -> u64 {
    match op {
        ArithmeticOpcode::ADD => a.wrapping_add(b),
        ArithmeticOpcode::SUB => a.wrapping_sub(b),
        ArithmeticOpcode::AND => a & b,
        ArithmeticOpcode::XOR => a ^ b,
        ArithmeticOpcode::NOT => !a,
        ArithmeticOpcode::LT => (a < b) as u64,
    }
}

fn generate_cpu_trace(ops: &[IntArithmeticOp], num_rows: usize, bit_width: usize) -> ColumnTrace {
    let num_vars = num_rows.trailing_zeros() as usize;
    let mut tb = TraceBuilder::new(&arith_cpu_layout(bit_width), num_vars).unwrap();

    for (i, call) in ops.iter().enumerate() {
        match call {
            IntArithmeticOp::U32 { op, a, b, .. } => {
                assert_eq!(bit_width, 32, "U32 op in non-u32 cpu trace");

                let res = compute_u32(*op, *a, *b);
                tb.set_b32(CpuArithColumns::VAL_A, i, Block32::from(*a))
                    .unwrap();
                tb.set_b32(CpuArithColumns::VAL_B, i, Block32::from(*b))
                    .unwrap();
                tb.set_b32(CpuArithColumns::VAL_RES, i, Block32::from(res))
                    .unwrap();
                tb.set_b32(CpuArithColumns::OPCODE, i, Block32::from(*op as u8 as u32))
                    .unwrap();
            }
            IntArithmeticOp::U64 { op, a, b, .. } => {
                assert_eq!(bit_width, 64, "U64 op in non-u64 cpu trace");

                let res = compute_u64(*op, *a, *b);
                tb.set_b64(CpuArithColumns::VAL_A, i, Block64::from(*a))
                    .unwrap();
                tb.set_b64(CpuArithColumns::VAL_B, i, Block64::from(*b))
                    .unwrap();
                tb.set_b64(CpuArithColumns::VAL_RES, i, Block64::from(res))
                    .unwrap();
                tb.set_b32(CpuArithColumns::OPCODE, i, Block32::from(*op as u8 as u32))
                    .unwrap();
            }
        }

        tb.set_bit(CpuArithColumns::SELECTOR, i, Bit::ONE).unwrap();
    }

    tb.build()
}

fn config() -> Config {
    Config {
        num_queries: 4,
        min_security_bits: 0,
        fold_arity_log: 1,
        early_stop_vars: 1,
        ..Config::default()
    }
}

fn assert_arith_verifies(ops: &[IntArithmeticOp], bit_width: usize) {
    let (program, instance, witness) = arith_case(ops, bit_width, |_, _| {});

    assert_verifies(&program, (&instance, &witness), &config());
}

fn arith_case(
    ops: &[IntArithmeticOp],
    bit_width: usize,
    tamper: impl FnOnce(&mut ColumnTrace, &mut ColumnTrace),
) -> (CircuitProgram<F>, ProgramInstance<F>, ProgramWitness<F>) {
    let num_rows = ops.len().next_power_of_two().max(2);
    let layout = IntArithmeticLayout::compute(bit_width);

    let mut cpu_trace = generate_cpu_trace(ops, num_rows, bit_width);
    let mut arith_trace = generate_arithmetic_trace(ops, &layout, num_rows).unwrap();

    tamper(&mut cpu_trace, &mut arith_trace);

    (
        arith_cpu_program(bit_width, num_rows, ops.len()),
        ProgramInstance::new(num_rows, vec![]),
        ProgramWitness::new(cpu_trace).with_chiplets(vec![arith_trace]),
    )
}

fn assert_tamper_rejected(
    ops: &[IntArithmeticOp],
    bit_width: usize,
    fault: Fault<'_>,
    tamper: impl FnOnce(&mut ColumnTrace, &mut ColumnTrace),
) {
    let (program, instance, honest) = arith_case(ops, bit_width, |_, _| {});
    let (_, _, forged) = arith_case(ops, bit_width, tamper);

    assert_rejected(
        &program,
        (&instance, &honest),
        (&instance, &forged),
        fault,
        &config(),
    );
}

fn assert_result_forgery_rejected(op: IntArithmeticOp, claimed: u32) {
    let (code, a, b) = match op {
        IntArithmeticOp::U32 { op, a, b } => (op, a, b),
        IntArithmeticOp::U64 { .. } => panic!("u32 forgeries only"),
    };

    let wrong_bits = (claimed ^ compute_u32(code, a, b)).count_ones() as usize;
    let rows = chiplet_rows(wrong_bits, Some("zero_when"));
    let fault = Fault {
        constraint_rows: &rows,
        ..Fault::NONE
    };

    assert_tamper_rejected(&[op], 32, fault, |cpu, arith| {
        set_b32(cpu, CpuArithColumns::VAL_RES, 0, claimed);
        set_b32(arith, PHY_VAL_RES, 0, claimed);
    });
}

fn assert_fitted_chain_rejected(op: IntArithmeticOp, forged: u32, carries: u32, broken: usize) {
    let rows = chiplet_rows(broken, Some("zero_when"));
    let fault = Fault {
        constraint_rows: &rows,
        ..Fault::NONE
    };

    assert_tamper_rejected(&[op], 32, fault, |cpu, arith| {
        set_b32(cpu, CpuArithColumns::VAL_RES, 0, forged);
        set_b32(arith, PHY_VAL_RES, 0, forged);
        set_b32(arith, PHY_CARRY, 0, carries);
    });
}

fn assert_padding_garbage_rejected(column: usize, garbage: u32) {
    let ops = padding_test_ops();
    let row = ops.len();
    let fault = Fault {
        constraint_rows: &[(TableId::Main, row, Some("zero_when"))],
        ..Fault::NONE
    };

    assert_tamper_rejected(&ops, 32, fault, |cpu, _| {
        set_b32(cpu, column, row, garbage);
    });
}

fn chiplet_rows(
    count: usize,
    label: Option<&'static str>,
) -> Vec<(TableId, usize, Option<&'static str>)> {
    vec![(TableId::Chiplet(0), 0, label); count]
}

fn chiplet_answers(arith: &mut ColumnTrace, ops: &[IntArithmeticOp]) {
    let num_rows = arith.num_rows().unwrap();

    *arith = generate_arithmetic_trace(ops, &IntArithmeticLayout::compute(32), num_rows).unwrap();
}

fn fitted_carries(a: u32, b: u32, res: u32, out: fn(u32, u32, u32) -> u32) -> (u32, usize) {
    let fitted = (res ^ a ^ b) >> 1;

    let mut carries = 0;
    let mut broken = 0;
    let mut carry_in = 0;

    for k in 0..32 {
        let want = out((a >> k) & 1, (b >> k) & 1, carry_in);
        let have = match k {
            31 => want,
            _ => (fitted >> k) & 1,
        };

        broken += usize::from(have != want);
        carries |= have << k;
        carry_in = have;
    }

    (carries, broken)
}

fn op_add(a: u32, b: u32) -> IntArithmeticOp {
    IntArithmeticOp::U32 {
        op: ArithmeticOpcode::ADD,
        a,
        b,
    }
}

fn op_sub(a: u32, b: u32) -> IntArithmeticOp {
    IntArithmeticOp::U32 {
        op: ArithmeticOpcode::SUB,
        a,
        b,
    }
}

fn op_and(a: u32, b: u32) -> IntArithmeticOp {
    IntArithmeticOp::U32 {
        op: ArithmeticOpcode::AND,
        a,
        b,
    }
}

fn op_xor(a: u32, b: u32) -> IntArithmeticOp {
    IntArithmeticOp::U32 {
        op: ArithmeticOpcode::XOR,
        a,
        b,
    }
}

fn op_not(a: u32) -> IntArithmeticOp {
    IntArithmeticOp::U32 {
        op: ArithmeticOpcode::NOT,
        a,
        b: 0,
    }
}

fn op_lt(a: u32, b: u32) -> IntArithmeticOp {
    IntArithmeticOp::U32 {
        op: ArithmeticOpcode::LT,
        a,
        b,
    }
}

fn op_add64(a: u64, b: u64) -> IntArithmeticOp {
    IntArithmeticOp::U64 {
        op: ArithmeticOpcode::ADD,
        a,
        b,
    }
}

fn op_sub64(a: u64, b: u64) -> IntArithmeticOp {
    IntArithmeticOp::U64 {
        op: ArithmeticOpcode::SUB,
        a,
        b,
    }
}

fn bus_forgery_ops() -> Vec<IntArithmeticOp> {
    vec![
        op_add(10, 20),
        op_add(5, 7),
        op_sub(100, 50),
        op_and(0xFF, 0x0F),
    ]
}

fn padding_test_ops() -> Vec<IntArithmeticOp> {
    vec![op_add(10, 20), op_add(5, 7), op_add(1, 2)]
}

// =================================================================
// 4. FUNCTIONAL TESTS - u32
// =================================================================

#[test]
fn arithmetic_chiplet_all_opcodes_basic() {
    let ops = vec![
        op_add(10, 20),
        op_sub(100, 50),
        op_and(0xFF, 0x0F),
        op_xor(0xAA, 0x55),
        op_not(0),
        op_lt(10, 20),
    ];

    assert_arith_verifies(&ops, 32);
}

#[test]
fn arithmetic_chiplet_add_boundary_values() {
    let ops = vec![
        op_add(0, 0),
        op_add(u32::MAX, 1),
        op_add(u32::MAX, u32::MAX),
        op_add(0x80000000, 0x80000000),
        op_add(0xFFFFFFFE, 1),
    ];

    assert_arith_verifies(&ops, 32);
}

#[test]
fn arithmetic_chiplet_sub_boundary_values() {
    let ops = vec![
        op_sub(0, 0),
        op_sub(0, 1),
        op_sub(1, 2),
        op_sub(u32::MAX, u32::MAX),
        op_sub(0x80000000, 0x7FFFFFFF),
    ];

    assert_arith_verifies(&ops, 32);
}

#[test]
fn arithmetic_chiplet_lt_boundary_values() {
    let ops = vec![
        op_lt(0, 0),
        op_lt(1, 0),
        op_lt(u32::MAX, u32::MAX),
        op_lt(u32::MAX - 1, u32::MAX),
        op_lt(u32::MAX, u32::MAX - 1),
        op_lt(0x80000000, 0x7FFFFFFF),
        op_lt(0, u32::MAX),
    ];

    assert_arith_verifies(&ops, 32);
}

#[test]
fn arithmetic_chiplet_bitwise_boundary_values() {
    let ops = vec![
        op_and(0, 0xFFFFFFFF),
        op_and(0xFFFFFFFF, 0xFFFFFFFF),
        op_xor(0xFFFFFFFF, 0xFFFFFFFF),
        op_xor(0, 0xFFFFFFFF),
        op_not(0xFFFFFFFF),
        op_not(0x5A5A5A5A),
    ];

    assert_arith_verifies(&ops, 32);
}

#[test]
fn arithmetic_chiplet_fibonacci_batch_matches_local_u32() {
    let num_ops: usize = 32;

    let mut a: u32 = 0;
    let mut b: u32 = 1;
    let mut ops = Vec::with_capacity(num_ops);
    let mut expected_last_res: u32 = 0;

    for i in 0..num_ops {
        let sum = a.wrapping_add(b);
        ops.push(op_add(a, b));

        if i == num_ops - 1 {
            expected_last_res = sum;
        }

        a = b;
        b = sum;
    }

    assert_arith_verifies(&ops, 32);

    let mut ra: u32 = 0;
    let mut rb: u32 = 1;

    for _ in 0..num_ops {
        let s = ra.wrapping_add(rb);
        ra = rb;
        rb = s;
    }

    assert_eq!(rb, expected_last_res);
}

// =================================================================
// 5. FUNCTIONAL TESTS - u64
// =================================================================

#[test]
fn arithmetic_chiplet_u64_add() {
    let ops = vec![
        op_add64(10, 20),
        op_add64(u64::MAX, 1),
        op_add64(u64::MAX, u64::MAX),
        op_add64(1u64 << 32, 1u64 << 32),
    ];

    assert_arith_verifies(&ops, 64);
}

#[test]
fn arithmetic_chiplet_u64_sub() {
    let ops = vec![
        op_sub64(0, 1),
        op_sub64(1, 2),
        op_sub64(u64::MAX, u64::MAX),
        op_sub64(1u64 << 63, (1u64 << 63) - 1),
    ];

    assert_arith_verifies(&ops, 64);
}

// =================================================================
// 6. MIXED-WIDTH COMPOSITION
// =================================================================

#[test]
fn arithmetic_chiplet_mixed_widths_isolated() {
    let ops32 = vec![op_add(10, 20), op_sub(100, 50)];
    let ops64 = vec![op_add64(u64::MAX, 1), op_add64(1u64 << 40, 1u64 << 40)];

    let cpu_rows: usize = 4;
    let chip32_rows: usize = ops32.len().next_power_of_two().max(2);
    let chip64_rows: usize = ops64.len().next_power_of_two().max(2);
    let num_vars = cpu_rows.trailing_zeros() as usize;

    let mut tb = TraceBuilder::new(&MixedArithColumns::build_layout(), num_vars).unwrap();

    for (i, op) in ops32.iter().enumerate() {
        let IntArithmeticOp::U32 { op, a, b, .. } = op else {
            unreachable!()
        };

        let res = compute_u32(*op, *a, *b);

        tb.set_b32(MixedArithColumns::U32_VAL_A, i, Block32::from(*a))
            .unwrap();
        tb.set_b32(MixedArithColumns::U32_VAL_B, i, Block32::from(*b))
            .unwrap();
        tb.set_b32(MixedArithColumns::U32_VAL_RES, i, Block32::from(res))
            .unwrap();
        tb.set_b32(
            MixedArithColumns::U32_OPCODE,
            i,
            Block32::from(*op as u8 as u32),
        )
        .unwrap();
        tb.set_bit(MixedArithColumns::U32_SELECTOR, i, Bit::ONE)
            .unwrap();
    }

    for (i, op) in ops64.iter().enumerate() {
        let IntArithmeticOp::U64 { op, a, b, .. } = op else {
            unreachable!()
        };

        let res = compute_u64(*op, *a, *b);

        tb.set_b64(MixedArithColumns::U64_VAL_A, i, Block64::from(*a))
            .unwrap();
        tb.set_b64(MixedArithColumns::U64_VAL_B, i, Block64::from(*b))
            .unwrap();
        tb.set_b64(MixedArithColumns::U64_VAL_RES, i, Block64::from(res))
            .unwrap();
        tb.set_b32(
            MixedArithColumns::U64_OPCODE,
            i,
            Block32::from(*op as u8 as u32),
        )
        .unwrap();
        tb.set_bit(MixedArithColumns::U64_SELECTOR, i, Bit::ONE)
            .unwrap();
    }

    let cpu_trace = tb.build();

    let layout32 = IntArithmeticLayout::compute(32);
    let layout64 = IntArithmeticLayout::compute(64);

    let arith32 = generate_arithmetic_trace(&ops32, &layout32, chip32_rows).unwrap();
    let arith64 = generate_arithmetic_trace(&ops64, &layout64, chip64_rows).unwrap();

    let program = mixed_arith_program(cpu_rows, chip32_rows, chip64_rows, ops32.len(), ops64.len());

    let instance = ProgramInstance::new(cpu_rows, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![arith32, arith64]);

    assert_verifies(&program, (&instance, &witness), &config());
}

// =================================================================
// 7. ADVERSARIAL — CPU-SIDE BUS TAMPERING
// =================================================================

#[test]
fn untampered_bus_forgery_ops_verify() {
    assert_arith_verifies(&bus_forgery_ops(), 32);
}

#[test]
fn reject_cpu_val_a_forgery() {
    assert_tamper_rejected(&bus_forgery_ops(), 32, BUS_ONLY, |cpu, _| {
        set_b32(cpu, CpuArithColumns::VAL_A, 0, 11);
    });
}

#[test]
fn reject_cpu_val_b_forgery() {
    assert_tamper_rejected(&bus_forgery_ops(), 32, BUS_ONLY, |cpu, _| {
        set_b32(cpu, CpuArithColumns::VAL_B, 1, 8);
    });
}

#[test]
fn reject_cpu_val_res_forgery() {
    assert_tamper_rejected(&bus_forgery_ops(), 32, BUS_ONLY, |cpu, _| {
        set_b32(cpu, CpuArithColumns::VAL_RES, 2, 51);
    });
}

#[test]
fn reject_cpu_opcode_forgery() {
    assert_tamper_rejected(&bus_forgery_ops(), 32, BUS_ONLY, |cpu, _| {
        set_b32(
            cpu,
            CpuArithColumns::OPCODE,
            0,
            ArithmeticOpcode::SUB as u32,
        );
    });
}

#[test]
fn reject_cpu_midrow_forgery() {
    let ops = bus_forgery_ops();
    let mid = ops.len() / 2;

    assert_tamper_rejected(&ops, 32, BUS_ONLY, move |cpu, _| {
        set_b32(cpu, CpuArithColumns::VAL_A, mid, 0xDEADBEEF);
    });
}

#[test]
fn exploit_int_arith_duplicate_cpu_request_rejected() {
    assert_tamper_rejected(&bus_forgery_ops(), 32, BUS_ONLY, |cpu, _| {
        set_b32(cpu, CpuArithColumns::VAL_A, 1, 10);
        set_b32(cpu, CpuArithColumns::VAL_B, 1, 20);
        set_b32(cpu, CpuArithColumns::VAL_RES, 1, 30);
        set_b32(
            cpu,
            CpuArithColumns::OPCODE,
            1,
            ArithmeticOpcode::ADD as u32,
        );
    });
}

#[test]
fn reject_cpu_traded_results() {
    assert_tamper_rejected(&bus_forgery_ops(), 32, BUS_ONLY, |cpu, _| {
        set_b32(cpu, CpuArithColumns::VAL_RES, 0, 12);
        set_b32(cpu, CpuArithColumns::VAL_RES, 1, 30);
    });
}

/// Two identical calls with the same forged result
/// cancel in char 2 unless ranks tell them apart.
#[test]
fn reject_cpu_identical_forged_results() {
    let ops = [op_add(1, 1), op_add(1, 1)];

    assert_tamper_rejected(&ops, 32, BUS_ONLY, |cpu, _| {
        set_b32(cpu, CpuArithColumns::VAL_RES, 0, 3);
        set_b32(cpu, CpuArithColumns::VAL_RES, 1, 3);
    });
}

// =================================================================
// 8. ADVERSARIAL - CHIPLET-SIDE BUS TAMPERING
// =================================================================

#[test]
fn reject_chiplet_val_a_forgery() {
    let mut answered = bus_forgery_ops();
    answered[0] = op_add(11, 20);

    assert_tamper_rejected(&bus_forgery_ops(), 32, BUS_ONLY, |_, arith| {
        chiplet_answers(arith, &answered);
    });
}

#[test]
fn reject_chiplet_val_b_forgery() {
    let mut answered = bus_forgery_ops();
    answered[0] = op_add(10, 21);

    assert_tamper_rejected(&bus_forgery_ops(), 32, BUS_ONLY, |_, arith| {
        chiplet_answers(arith, &answered);
    });
}

#[test]
fn reject_chiplet_val_res_forgery() {
    let rows = chiplet_rows((31u32 ^ 30).count_ones() as usize, Some("zero_when"));
    let fault = Fault {
        constraint_rows: &rows,
        buses: &[IntArithmeticChiplet::BUS_ID],
        ..Fault::NONE
    };

    assert_tamper_rejected(&bus_forgery_ops(), 32, fault, |_, arith| {
        set_b32(arith, PHY_VAL_RES, 0, 31);
    });
}

#[test]
fn reject_chiplet_opcode_forgery() {
    let mut answered = bus_forgery_ops();
    answered[0] = op_xor(10, 20);

    assert_eq!(10 ^ 20, 10 + 20);

    assert_tamper_rejected(&bus_forgery_ops(), 32, BUS_ONLY, |_, arith| {
        chiplet_answers(arith, &answered);
    });
}

#[test]
fn reject_chiplet_second_op_forgery() {
    let mut answered = bus_forgery_ops();
    answered[1] = op_add(99, 7);

    assert_tamper_rejected(&bus_forgery_ops(), 32, BUS_ONLY, |_, arith| {
        chiplet_answers(arith, &answered);
    });
}

// =================================================================
// 9. ADVERSARIAL - CHIPLET LOCAL CONSTRAINTS
// =================================================================

#[test]
fn reject_wrong_result_via_carry_chain() {
    assert_result_forgery_rejected(op_add(10, 20), 31);
}

#[test]
fn reject_wrong_borrow_via_sub_chain() {
    assert_result_forgery_rejected(op_sub(100, 50), 51);
}

#[test]
fn reject_wrong_result_via_and_constraint() {
    assert_result_forgery_rejected(op_and(0xFF, 0x0F), 0x10);
}

#[test]
fn reject_wrong_result_via_xor_constraint() {
    assert_result_forgery_rejected(op_xor(0xAAAA, 0x5555), 0xFFFE);
}

#[test]
fn reject_wrong_result_via_not_constraint() {
    assert_result_forgery_rejected(op_not(0), 0xFFFFFFFE);
}

#[test]
fn reject_wrong_result_via_lt_constraint() {
    assert_result_forgery_rejected(op_lt(10, 20), 0);
}

#[test]
fn reject_wrong_sum_with_fitted_carries() {
    let (a, b) = (10u32, 20u32);
    let forged = a.wrapping_add(b) ^ 2;
    let (carries, broken) = fitted_carries(a, b, forged, |a, b, c| (a & b) | (c & (a ^ b)));

    assert_fitted_chain_rejected(op_add(a, b), forged, carries, broken);
}

#[test]
fn reject_wrong_difference_with_fitted_borrows() {
    let (a, b) = (100u32, 50u32);
    let forged = a.wrapping_sub(b) ^ 2;
    let (borrows, broken) =
        fitted_carries(a, b, forged, |a, b, w| ((a ^ 1) & b) | (w & (a ^ b ^ 1)));

    assert_fitted_chain_rejected(op_sub(a, b), forged, borrows, broken);
}

#[test]
fn reject_three_selectors_set() {
    let opcode = ArithmeticOpcode::AND as u32;
    let rows = chiplet_rows(3, None);
    let fault = Fault {
        constraint_rows: &rows,
        ..Fault::NONE
    };

    assert_tamper_rejected(&[op_add(0, 0)], 32, fault, |cpu, arith| {
        set_bit(arith, PHY_S_SUB, 0, Bit::ONE);
        set_bit(arith, PHY_S_XOR, 0, Bit::ONE);
        set_b32(arith, PHY_OPCODE, 0, opcode);
        set_b32(cpu, CpuArithColumns::OPCODE, 0, opcode);
    });
}

#[test]
fn reject_opcode_mismatch() {
    let ops = vec![op_add(10, 20)];
    let rows = chiplet_rows(1, None);
    let fault = Fault {
        constraint_rows: &rows,
        ..Fault::NONE
    };

    assert_tamper_rejected(&ops, 32, fault, |cpu, arith| {
        set_b32(
            cpu,
            CpuArithColumns::OPCODE,
            0,
            ArithmeticOpcode::SUB as u32,
        );
        set_b32(arith, PHY_OPCODE, 0, ArithmeticOpcode::SUB as u32);
    });
}

// =================================================================
// 10. PADDING ROWS - BUS GATE ALLOWS GARBAGE
// =================================================================

#[test]
fn padding_val_a_garbage_rejected() {
    assert_padding_garbage_rejected(CpuArithColumns::VAL_A, 0xDEADBEEF);
}

#[test]
fn padding_val_b_garbage_rejected() {
    assert_padding_garbage_rejected(CpuArithColumns::VAL_B, 0xCAFEBABE);
}

#[test]
fn padding_val_res_garbage_rejected() {
    assert_padding_garbage_rejected(CpuArithColumns::VAL_RES, 0xBADF00D);
}

#[test]
fn padding_opcode_garbage_rejected() {
    assert_padding_garbage_rejected(CpuArithColumns::OPCODE, 0xFFFFFFFF);
}

#[test]
fn scribble_arithmetic_flip_selector_caught() {
    let bit_width = 32;
    let ops = [op_add(10, 20), op_sub(100, 50), op_and(0xFF, 0x0F)];

    let num_rows = ops.len().next_power_of_two().max(2);
    let arith_num_rows = num_rows;

    let cpu_trace = generate_cpu_trace(&ops, num_rows, bit_width);
    let layout = IntArithmeticLayout::compute(bit_width);
    let arith_trace =
        generate_arithmetic_trace(&ops, &layout, arith_num_rows).expect("arith trace");

    let air = arith_cpu_program(bit_width, arith_num_rows, ops.len());
    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![arith_trace]);

    assert_all_caught_all_targets(
        &air,
        &instance,
        &witness,
        ScribbleConfig::default()
            .mutations([MutationKind::FlipSelector])
            .cases(64),
    );
}

// =================================================================
// 11. ATTACHED COMPOSITE THROUGH THE WIRE
// =================================================================

#[test]
fn attached_composite_proves_through_wire() {
    let ops = [
        op_add(10, 20),
        op_sub(100, 50),
        op_and(0xFF, 0x0F),
        op_xor(0xAA, 0x55),
        op_not(0),
        op_lt(10, 20),
    ];

    let num_rows = ops.len().next_power_of_two();
    let layout = IntArithmeticLayout::compute(32);

    for extra in [false, true] {
        let mut arith_trace = generate_arithmetic_trace(&ops, &layout, num_rows).unwrap();

        if extra {
            arith_trace
                .add_column(TraceColumn::B32(vec![
                    Block32::ZERO.to_hardware();
                    num_rows
                ]))
                .unwrap();
        }

        let program = arith_cpu_program_attaching(
            32,
            num_rows,
            ops.len(),
            composite_arith_def(num_rows, ops.len(), extra),
        );

        let instance = ProgramInstance::new(num_rows, vec![]);
        let witness = ProgramWitness::new(generate_cpu_trace(&ops, num_rows, 32))
            .with_chiplets(vec![arith_trace]);

        assert_verifies(&program, (&instance, &witness), &config());
    }
}
