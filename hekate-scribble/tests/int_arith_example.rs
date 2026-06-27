//! Layer-1 + Layer-2 scribble coverage for the
//! IntArithmeticChiplet (u32 mode) from
//! `hekate-gadgets`. Random shallow tampers
//! plus hand-crafted exploits that target the
//! chiplet's load-bearing invariants: opcode
//! gating, result equality with bit-decomposed
//! operands, and padding-row shadow columns.

use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate_gadgets::{
    ArithmeticOpcode, CpuArithColumns, CpuIntArithmeticUnit, IntArithmeticChiplet,
    IntArithmeticLayout, IntArithmeticOp, generate_arithmetic_trace,
};
use hekate_math::{Bit, Block32, Block128, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::permutation::PermutationCheckSpec;
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness};
use hekate_scribble::{
    Mutation, MutationKind, ScribbleConfig, Target, assert_all_caught, check_single_mutation,
};

type F = Block128;

const BIT_WIDTH: usize = 32;

#[derive(Clone)]
struct ArithTestProgram {
    arith_num_rows: usize,
    cpu_layout: Vec<ColumnType>,
}

impl ArithTestProgram {
    fn new(arith_num_rows: usize) -> Self {
        let cpu_layout = vec![
            ColumnType::B32,
            ColumnType::B32,
            ColumnType::B32,
            ColumnType::B32,
            ColumnType::Bit,
        ];

        Self {
            arith_num_rows,
            cpu_layout,
        }
    }
}

impl Air<F> for ArithTestProgram {
    fn column_layout(&self) -> &[ColumnType] {
        &self.cpu_layout
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![(
            IntArithmeticChiplet::BUS_ID.into(),
            CpuIntArithmeticUnit::linking_spec(),
        )]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();

        let selector = cs.col(CpuArithColumns::SELECTOR);
        cs.assert_boolean(selector);

        let not_active = cs.one() + selector;
        cs.assert_zero_when(not_active, cs.col(CpuArithColumns::VAL_A));
        cs.assert_zero_when(not_active, cs.col(CpuArithColumns::VAL_B));
        cs.assert_zero_when(not_active, cs.col(CpuArithColumns::VAL_RES));
        cs.assert_zero_when(not_active, cs.col(CpuArithColumns::OPCODE));

        cs.build()
    }
}

impl Program<F> for ArithTestProgram {
    fn chiplet_defs(&self) -> hekate_core::errors::Result<Vec<ChipletDef<F>>> {
        let arith = IntArithmeticChiplet::new(BIT_WIDTH, self.arith_num_rows)
            .expect("IntArithmeticChiplet::new");
        Ok(vec![ChipletDef::from_air(&arith)?])
    }
}

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

fn with_request_idx(ops: &[(ArithmeticOpcode, u32, u32)]) -> Vec<IntArithmeticOp> {
    ops.iter()
        .enumerate()
        .map(|(i, &(op, a, b))| IntArithmeticOp::U32 {
            op,
            a,
            b,
            request_idx: i as u32,
        })
        .collect()
}

fn generate_cpu_trace(
    ops: &[IntArithmeticOp],
    num_rows: usize,
    layout: &[ColumnType],
) -> ColumnTrace {
    let num_vars = num_rows.trailing_zeros() as usize;
    let mut tb = TraceBuilder::new(layout, num_vars).unwrap();

    for (i, call) in ops.iter().enumerate() {
        let IntArithmeticOp::U32 { op, a, b, .. } = *call else {
            panic!("non-u32 op in u32 cpu trace");
        };

        let res = compute_u32(op, a, b);

        tb.set_b32(CpuArithColumns::VAL_A, i, Block32::from(a))
            .unwrap();
        tb.set_b32(CpuArithColumns::VAL_B, i, Block32::from(b))
            .unwrap();
        tb.set_b32(CpuArithColumns::VAL_RES, i, Block32::from(res))
            .unwrap();
        tb.set_b32(CpuArithColumns::OPCODE, i, Block32::from(op as u8 as u32))
            .unwrap();
        tb.set_bit(CpuArithColumns::SELECTOR, i, Bit::ONE).unwrap();
    }

    tb.build()
}

fn build_fixture(
    raw_ops: &[(ArithmeticOpcode, u32, u32)],
    num_rows: usize,
) -> (
    ArithTestProgram,
    ProgramInstance<F>,
    ProgramWitness<F, ColumnTrace>,
) {
    let ops = with_request_idx(raw_ops);

    let air = ArithTestProgram::new(num_rows);
    let cpu_trace = generate_cpu_trace(&ops, num_rows, &air.cpu_layout);

    let layout = IntArithmeticLayout::compute(BIT_WIDTH);
    let arith_trace = generate_arithmetic_trace(&ops, &layout, num_rows).expect("arith trace gen");

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![arith_trace]);

    (air, instance, witness)
}

fn setup_dense_fixture() -> (
    ArithTestProgram,
    ProgramInstance<F>,
    ProgramWitness<F, ColumnTrace>,
) {
    let ops = [
        (ArithmeticOpcode::ADD, 10, 20),
        (ArithmeticOpcode::SUB, 100, 50),
        (ArithmeticOpcode::AND, 0xFF, 0x0F),
        (ArithmeticOpcode::XOR, 0xAA, 0x55),
    ];
    build_fixture(&ops, 4)
}

fn setup_padding_fixture() -> (
    ArithTestProgram,
    ProgramInstance<F>,
    ProgramWitness<F, ColumnTrace>,
) {
    let ops = [
        (ArithmeticOpcode::ADD, 10, 20),
        (ArithmeticOpcode::SUB, 100, 50),
        (ArithmeticOpcode::AND, 0xFF, 0x0F),
    ];

    // 4-row trace with 1 padding row (s_active = 0).
    build_fixture(&ops, 4)
}

fn setup_overflow_fixture() -> (
    ArithTestProgram,
    ProgramInstance<F>,
    ProgramWitness<F, ColumnTrace>,
) {
    // ADD overflow + SUB underflow + LT boundary.
    // Exercises the carry/borrow chains the body
    // constraints must enforce.
    let ops = [
        (ArithmeticOpcode::ADD, u32::MAX, 1),
        (ArithmeticOpcode::SUB, 0, 1),
        (ArithmeticOpcode::LT, u32::MAX, u32::MAX),
        (ArithmeticOpcode::LT, 0, u32::MAX),
    ];

    build_fixture(&ops, 4)
}

#[test]
fn arith_chiplet_survives_chaos() {
    let (air, instance, witness) = setup_dense_fixture();

    let config = ScribbleConfig::default()
        .target(Target::Chiplet(0))
        .mutations([
            MutationKind::BitFlip,
            MutationKind::OutOfBounds,
            MutationKind::FlipSelector,
            MutationKind::DuplicateRow,
        ])
        .cases(256);

    assert_all_caught(&air, &instance, &witness, config);
}

#[test]
fn arith_padding_fixture_survives_chaos() {
    let (air, instance, witness) = setup_padding_fixture();

    let config = ScribbleConfig::default()
        .target(Target::Main)
        .mutations([
            MutationKind::BitFlip,
            MutationKind::OutOfBounds,
            MutationKind::FlipSelector,
            MutationKind::DuplicateRow,
        ])
        .cases(512);

    assert_all_caught(&air, &instance, &witness, config);
}

#[test]
fn arith_overflow_fixture_survives_chaos() {
    let (air, instance, witness) = setup_overflow_fixture();

    let config = ScribbleConfig::default()
        .target(Target::Chiplet(0))
        .mutations([
            MutationKind::BitFlip,
            MutationKind::OutOfBounds,
            MutationKind::FlipSelector,
            MutationKind::DuplicateRow,
        ])
        .cases(256);

    assert_all_caught(&air, &instance, &witness, config);
}

#[test]
fn arith_result_tamper_on_chiplet_caught() {
    // Flip a low bit of the chiplet's first
    // physical column on row 0. That's a bit
    // of the result decomposition; the body
    // constraint binding result_bits to the
    // operand-derived value must catch it.
    let (air, instance, witness) = setup_dense_fixture();

    let mutation = Mutation::BitFlip {
        target: Target::Chiplet(0),
        col: 0,
        row: 0,
        mask: 1,
    };

    let result = check_single_mutation(&air, &instance, &witness, &mutation);
    assert!(result.is_ok(), "result-bit tamper escaped preflight");
}

#[test]
fn arith_opcode_tamper_on_chiplet_caught() {
    // OPCODE column rerouting flips which
    // body equation is gated; the new gate
    // doesn't match the witnessed operands,
    // so at least one constraint must fail.
    let (air, instance, witness) = setup_dense_fixture();

    let mutation = Mutation::OutOfBounds {
        target: Target::Chiplet(0),
        col: 4,
        row: 0,
        value: ArithmeticOpcode::XOR as u8 as u128,
    };

    let result = check_single_mutation(&air, &instance, &witness, &mutation);
    assert!(result.is_ok(), "opcode tamper escaped preflight");
}
