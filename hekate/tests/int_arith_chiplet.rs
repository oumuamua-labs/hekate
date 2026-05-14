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

use hekate::core::config::Config;
use hekate::core::trace::{ColumnTrace, ColumnType, TraceColumn};
use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::{Block128, Flat, HardwareField, TowerField};
use hekate_core::trace::TraceBuilder;
use hekate_gadgets::{
    ArithmeticOpcode, CpuArithColumns, CpuIntArithmeticUnit, IntArithmeticChiplet,
    IntArithmeticLayout, IntArithmeticOp, generate_arithmetic_trace,
};
use hekate_math::{Bit, Block32, Block64};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::permutation::{PermutationCheckSpec, REQUEST_IDX_LABEL, Source};
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_sdk::preflight;
use hekate_verifier::HekateVerifier;
use zk_scribble::{MutationKind, ScribbleConfig, assert_all_caught_all_targets};

type F = Block128;
type H = DefaultHasher;

// =================================================================
// 0. PHYSICAL COLUMN INDICES
// =================================================================

const PHY_VAL_A: usize = 0;
const PHY_VAL_B: usize = 1;
const PHY_VAL_RES: usize = 2;
const PHY_OPCODE: usize = 4;
const PHY_S_SUB: usize = 8;

// =================================================================
// 1. SINGLE-CHIPLET PROGRAM (u32 OR u64)
// =================================================================

#[derive(Clone)]
struct ArithCpuProgram {
    bit_width: usize,
    arith_num_rows: usize,
    cpu_layout: Vec<ColumnType>,
}

impl ArithCpuProgram {
    fn new(bit_width: usize, arith_num_rows: usize) -> Self {
        let operand = if bit_width == 64 {
            ColumnType::B64
        } else {
            ColumnType::B32
        };

        let cpu_layout = vec![operand, operand, operand, ColumnType::B32, ColumnType::Bit];

        Self {
            bit_width,
            arith_num_rows,
            cpu_layout,
        }
    }
}

impl Air<F> for ArithCpuProgram {
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
        cs.assert_boolean(cs.col(CpuArithColumns::SELECTOR));

        cs.build()
    }
}

impl Program<F> for ArithCpuProgram {
    fn chiplet_defs(&self) -> hekate_core::errors::Result<Vec<ChipletDef<F>>> {
        let arith = IntArithmeticChiplet::new(self.bit_width, self.arith_num_rows)
            .expect("IntArithmeticChiplet::new in test program");
        Ok(vec![ChipletDef::from_air(&arith)?])
    }
}

// =================================================================
// 2. MIXED-WIDTH PROGRAM
// =================================================================

const MIXED_BUS_U32: &str = "arith_link_u32";
const MIXED_BUS_U64: &str = "arith_link_u64";

const MIXED_U32_VAL_A: usize = 0;
const MIXED_U32_VAL_B: usize = 1;
const MIXED_U32_VAL_RES: usize = 2;
const MIXED_U32_OPCODE: usize = 3;
const MIXED_U32_SELECTOR: usize = 4;
const MIXED_U64_VAL_A: usize = 5;
const MIXED_U64_VAL_B: usize = 6;
const MIXED_U64_VAL_RES: usize = 7;
const MIXED_U64_OPCODE: usize = 8;
const MIXED_U64_SELECTOR: usize = 9;

#[derive(Clone)]
struct MixedArithProgram {
    chip32_rows: usize,
    chip64_rows: usize,
}

impl MixedArithProgram {
    fn cpu_layout() -> Vec<ColumnType> {
        vec![
            ColumnType::B32,
            ColumnType::B32,
            ColumnType::B32,
            ColumnType::B32,
            ColumnType::Bit,
            ColumnType::B64,
            ColumnType::B64,
            ColumnType::B64,
            ColumnType::B32,
            ColumnType::Bit,
        ]
    }
}

impl Air<F> for MixedArithProgram {
    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(MixedArithProgram::cpu_layout)
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        let cpu32 = PermutationCheckSpec::new(
            vec![
                (Source::Column(MIXED_U32_VAL_A), b"kappa_val_a" as &[u8]),
                (Source::Column(MIXED_U32_VAL_B), b"kappa_val_b" as &[u8]),
                (Source::Column(MIXED_U32_VAL_RES), b"kappa_val_res" as &[u8]),
                (Source::Column(MIXED_U32_OPCODE), b"kappa_opcode" as &[u8]),
                (Source::RowIndexLeBytes(4), REQUEST_IDX_LABEL),
            ],
            Some(MIXED_U32_SELECTOR),
        );

        let cpu64 = PermutationCheckSpec::new(
            vec![
                (Source::Column(MIXED_U64_VAL_A), b"kappa_val_a" as &[u8]),
                (Source::Column(MIXED_U64_VAL_B), b"kappa_val_b" as &[u8]),
                (Source::Column(MIXED_U64_VAL_RES), b"kappa_val_res" as &[u8]),
                (Source::Column(MIXED_U64_OPCODE), b"kappa_opcode" as &[u8]),
                (Source::RowIndexLeBytes(4), REQUEST_IDX_LABEL),
            ],
            Some(MIXED_U64_SELECTOR),
        );

        vec![(MIXED_BUS_U32.into(), cpu32), (MIXED_BUS_U64.into(), cpu64)]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(MIXED_U32_SELECTOR));
        cs.assert_boolean(cs.col(MIXED_U64_SELECTOR));

        cs.build()
    }
}

impl Program<F> for MixedArithProgram {
    fn chiplet_defs(&self) -> hekate_core::errors::Result<Vec<ChipletDef<F>>> {
        let chip32 = IntArithmeticChiplet::new(32, self.chip32_rows)
            .expect("u32 chiplet")
            .with_bus_id(MIXED_BUS_U32);
        let chip64 = IntArithmeticChiplet::new(64, self.chip64_rows)
            .expect("u64 chiplet")
            .with_bus_id(MIXED_BUS_U64);

        Ok(vec![
            ChipletDef::from_air(&chip32)?,
            ChipletDef::from_air(&chip64)?,
        ])
    }
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
    let prog = ArithCpuProgram::new(bit_width, num_rows);

    let mut tb = TraceBuilder::new(&prog.cpu_layout, num_vars).unwrap();

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

fn test_config() -> Config {
    Config {
        num_queries: 4,
        min_security_bits: 0,
        sumcheck_blinding_factor: 2,
        ..Config::default()
    }
}

fn run_prover_verifier(
    program: &impl Program<F>,
    instance: &ProgramInstance<F>,
    witness: &ProgramWitness<F>,
    domain: &'static [u8],
) -> Result<bool, String> {
    let config = test_config();

    let proof = prove(
        domain, program, instance, witness, &config, [0xAA; 32], None,
    )
    .map_err(|e| format!("prover: {e:?}"))?;

    let mut vt = Transcript::<H>::new(domain);
    HekateVerifier::<F, H>::verify(program, instance, &proof, &mut vt, &config)
        .map_err(|e| format!("verifier: {e:?}"))
}

fn with_cpu_row_idx(ops: &[IntArithmeticOp]) -> Vec<IntArithmeticOp> {
    ops.iter()
        .enumerate()
        .map(|(i, op)| match *op {
            IntArithmeticOp::U32 { op, a, b, .. } => IntArithmeticOp::U32 {
                op,
                a,
                b,
                request_idx: i as u32,
            },
            IntArithmeticOp::U64 { op, a, b, .. } => IntArithmeticOp::U64 {
                op,
                a,
                b,
                request_idx: i as u32,
            },
        })
        .collect()
}

fn prove_and_verify_arithmetic(ops: &[IntArithmeticOp], bit_width: usize) -> Result<bool, String> {
    let num_rows = ops.len().next_power_of_two().max(2);
    let arith_num_rows = num_rows;

    let ops = with_cpu_row_idx(ops);

    let cpu_trace = generate_cpu_trace(&ops, num_rows, bit_width);
    let layout = IntArithmeticLayout::compute(bit_width);
    let arith_trace = generate_arithmetic_trace(&ops, &layout, arith_num_rows)
        .map_err(|e| format!("arith trace: {e:?}"))?;

    let program = ArithCpuProgram::new(bit_width, arith_num_rows);
    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![arith_trace]);

    let report =
        preflight(&program, &instance, &witness).map_err(|e| format!("preflight: {e:?}"))?;

    if !report.is_clean() {
        return Err(format!(
            "preflight dirty: {} constraint, {} boundary, {} bus diagnostics",
            report.constraint_violations.len(),
            report.boundary_violations.len(),
            report.bus_diagnostics.len(),
        ));
    }

    run_prover_verifier(&program, &instance, &witness, b"ARITH_E2E")
}

fn prove_and_verify_with_tamper<M>(
    ops: &[IntArithmeticOp],
    bit_width: usize,
    tamper: M,
) -> Result<bool, String>
where
    M: FnOnce(&mut ColumnTrace, &mut ColumnTrace),
{
    let num_rows = ops.len().next_power_of_two().max(2);
    let arith_num_rows = num_rows;

    let ops = with_cpu_row_idx(ops);

    let mut cpu_trace = generate_cpu_trace(&ops, num_rows, bit_width);

    let layout = IntArithmeticLayout::compute(bit_width);

    let mut arith_trace = generate_arithmetic_trace(&ops, &layout, arith_num_rows)
        .map_err(|e| format!("arith trace: {e:?}"))?;

    tamper(&mut cpu_trace, &mut arith_trace);

    let program = ArithCpuProgram::new(bit_width, arith_num_rows);
    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![arith_trace]);

    run_prover_verifier(&program, &instance, &witness, b"ARITH_TAMPER")
}

fn assert_tamper_rejected(
    ops: &[IntArithmeticOp],
    bit_width: usize,
    label: &str,
    tamper: impl FnOnce(&mut ColumnTrace, &mut ColumnTrace),
) {
    if let Ok(true) = prove_and_verify_with_tamper(ops, bit_width, tamper) {
        panic!("{label}: tampered proof accepted")
    }
}

fn b32_cell(v: u32) -> Flat<Block32> {
    Block32::from(v).to_hardware()
}

fn b64_cell(v: u64) -> Flat<Block64> {
    Block64::from(v).to_hardware()
}

fn tamper_b32(trace: &mut ColumnTrace, col: usize, row: usize, value: u32) {
    if let TraceColumn::B32(c) = &mut trace.columns[col] {
        c[row] = b32_cell(value);
    } else {
        panic!("tamper_b32: column {col} is not B32");
    }
}

fn tamper_b64(trace: &mut ColumnTrace, col: usize, row: usize, value: u64) {
    if let TraceColumn::B64(c) = &mut trace.columns[col] {
        c[row] = b64_cell(value);
    } else {
        panic!("tamper_b64: column {col} is not B64");
    }
}

fn tamper_bit(trace: &mut ColumnTrace, col: usize, row: usize, value: u8) {
    if let TraceColumn::Bit(c) = &mut trace.columns[col] {
        c[row] = Bit(value);
    } else {
        panic!("tamper_bit: column {col} is not Bit");
    }
}

fn op_add(a: u32, b: u32) -> IntArithmeticOp {
    IntArithmeticOp::U32 {
        op: ArithmeticOpcode::ADD,
        a,
        b,
        request_idx: 0,
    }
}

fn op_sub(a: u32, b: u32) -> IntArithmeticOp {
    IntArithmeticOp::U32 {
        op: ArithmeticOpcode::SUB,
        a,
        b,
        request_idx: 0,
    }
}

fn op_and(a: u32, b: u32) -> IntArithmeticOp {
    IntArithmeticOp::U32 {
        op: ArithmeticOpcode::AND,
        a,
        b,
        request_idx: 0,
    }
}

fn op_xor(a: u32, b: u32) -> IntArithmeticOp {
    IntArithmeticOp::U32 {
        op: ArithmeticOpcode::XOR,
        a,
        b,
        request_idx: 0,
    }
}

fn op_not(a: u32) -> IntArithmeticOp {
    IntArithmeticOp::U32 {
        op: ArithmeticOpcode::NOT,
        a,
        b: 0,
        request_idx: 0,
    }
}

fn op_lt(a: u32, b: u32) -> IntArithmeticOp {
    IntArithmeticOp::U32 {
        op: ArithmeticOpcode::LT,
        a,
        b,
        request_idx: 0,
    }
}

fn op_add64(a: u64, b: u64) -> IntArithmeticOp {
    IntArithmeticOp::U64 {
        op: ArithmeticOpcode::ADD,
        a,
        b,
        request_idx: 0,
    }
}

fn op_sub64(a: u64, b: u64) -> IntArithmeticOp {
    IntArithmeticOp::U64 {
        op: ArithmeticOpcode::SUB,
        a,
        b,
        request_idx: 0,
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

    assert_eq!(prove_and_verify_arithmetic(&ops, 32), Ok(true));
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

    assert_eq!(prove_and_verify_arithmetic(&ops, 32), Ok(true));
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

    assert_eq!(prove_and_verify_arithmetic(&ops, 32), Ok(true));
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

    assert_eq!(prove_and_verify_arithmetic(&ops, 32), Ok(true));
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

    assert_eq!(prove_and_verify_arithmetic(&ops, 32), Ok(true));
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

    assert_eq!(prove_and_verify_arithmetic(&ops, 32), Ok(true));

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

    assert_eq!(prove_and_verify_arithmetic(&ops, 64), Ok(true));
}

#[test]
fn arithmetic_chiplet_u64_sub() {
    let ops = vec![
        op_sub64(0, 1),
        op_sub64(1, 2),
        op_sub64(u64::MAX, u64::MAX),
        op_sub64(1u64 << 63, (1u64 << 63) - 1),
    ];

    assert_eq!(prove_and_verify_arithmetic(&ops, 64), Ok(true));
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

    let layout = MixedArithProgram::cpu_layout();

    let mut tb = TraceBuilder::new(&layout, num_vars).unwrap();

    for (i, op) in ops32.iter().enumerate() {
        let IntArithmeticOp::U32 { op, a, b, .. } = op else {
            unreachable!()
        };

        let res = compute_u32(*op, *a, *b);

        tb.set_b32(MIXED_U32_VAL_A, i, Block32::from(*a)).unwrap();
        tb.set_b32(MIXED_U32_VAL_B, i, Block32::from(*b)).unwrap();
        tb.set_b32(MIXED_U32_VAL_RES, i, Block32::from(res))
            .unwrap();
        tb.set_b32(MIXED_U32_OPCODE, i, Block32::from(*op as u8 as u32))
            .unwrap();
        tb.set_bit(MIXED_U32_SELECTOR, i, Bit::ONE).unwrap();
    }

    for (i, op) in ops64.iter().enumerate() {
        let IntArithmeticOp::U64 { op, a, b, .. } = op else {
            unreachable!()
        };

        let res = compute_u64(*op, *a, *b);

        tb.set_b64(MIXED_U64_VAL_A, i, Block64::from(*a)).unwrap();
        tb.set_b64(MIXED_U64_VAL_B, i, Block64::from(*b)).unwrap();
        tb.set_b64(MIXED_U64_VAL_RES, i, Block64::from(res))
            .unwrap();
        tb.set_b32(MIXED_U64_OPCODE, i, Block32::from(*op as u8 as u32))
            .unwrap();
        tb.set_bit(MIXED_U64_SELECTOR, i, Bit::ONE).unwrap();
    }

    let cpu_trace = tb.build();

    let layout32 = IntArithmeticLayout::compute(32);
    let layout64 = IntArithmeticLayout::compute(64);

    let ops32_idx = with_cpu_row_idx(&ops32);
    let ops64_idx = with_cpu_row_idx(&ops64);

    let arith32 = generate_arithmetic_trace(&ops32_idx, &layout32, chip32_rows).unwrap();
    let arith64 = generate_arithmetic_trace(&ops64_idx, &layout64, chip64_rows).unwrap();

    let program = MixedArithProgram {
        chip32_rows,
        chip64_rows,
    };
    let instance = ProgramInstance::new(cpu_rows, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![arith32, arith64]);

    let report = preflight(&program, &instance, &witness).expect("preflight mixed_widths");

    assert!(
        report.is_clean(),
        "preflight dirty: {} constraint, {} boundary, {} bus",
        report.constraint_violations.len(),
        report.boundary_violations.len(),
        report.bus_diagnostics.len(),
    );

    let result = run_prover_verifier(&program, &instance, &witness, b"ARITH_MIXED").unwrap();
    assert!(result, "mixed-width prove/verify failed");
}

// =================================================================
// 7. ADVERSARIAL — CPU-SIDE BUS TAMPERING
// =================================================================

#[test]
fn reject_cpu_val_a_forgery() {
    assert_tamper_rejected(&bus_forgery_ops(), 32, "cpu VAL_A", |cpu, _| {
        tamper_b32(cpu, CpuArithColumns::VAL_A, 0, 11);
    });
}

#[test]
fn reject_cpu_val_b_forgery() {
    assert_tamper_rejected(&bus_forgery_ops(), 32, "cpu VAL_B", |cpu, _| {
        tamper_b32(cpu, CpuArithColumns::VAL_B, 1, 8);
    });
}

#[test]
fn reject_cpu_val_res_forgery() {
    assert_tamper_rejected(&bus_forgery_ops(), 32, "cpu VAL_RES", |cpu, _| {
        tamper_b32(cpu, CpuArithColumns::VAL_RES, 2, 51);
    });
}

#[test]
fn reject_cpu_opcode_forgery() {
    assert_tamper_rejected(&bus_forgery_ops(), 32, "cpu OPCODE", |cpu, _| {
        tamper_b32(
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

    assert_tamper_rejected(&ops, 32, "cpu VAL_A mid-row", move |cpu, _| {
        tamper_b32(cpu, CpuArithColumns::VAL_A, mid, 0xDEADBEEF);
    });
}

#[test]
fn exploit_int_arith_duplicate_cpu_request_rejected() {
    assert_tamper_rejected(
        &bus_forgery_ops(),
        32,
        "duplicate cpu request without chiplet partner",
        |cpu, _| {
            tamper_b32(cpu, CpuArithColumns::VAL_A, 1, 10);
            tamper_b32(cpu, CpuArithColumns::VAL_B, 1, 20);
            tamper_b32(cpu, CpuArithColumns::VAL_RES, 1, 30);
            tamper_b32(
                cpu,
                CpuArithColumns::OPCODE,
                1,
                ArithmeticOpcode::ADD as u32,
            );
        },
    );
}

// =================================================================
// 8. ADVERSARIAL - CHIPLET-SIDE BUS TAMPERING
// =================================================================

#[test]
fn reject_chiplet_val_a_forgery() {
    assert_tamper_rejected(&bus_forgery_ops(), 32, "chiplet VAL_A", |_, arith| {
        tamper_b32(arith, PHY_VAL_A, 0, 11);
    });
}

#[test]
fn reject_chiplet_val_b_forgery() {
    assert_tamper_rejected(&bus_forgery_ops(), 32, "chiplet VAL_B", |_, arith| {
        tamper_b32(arith, PHY_VAL_B, 0, 21);
    });
}

#[test]
fn reject_chiplet_val_res_forgery() {
    assert_tamper_rejected(&bus_forgery_ops(), 32, "chiplet VAL_RES", |_, arith| {
        tamper_b32(arith, PHY_VAL_RES, 0, 31);
    });
}

#[test]
fn reject_chiplet_opcode_forgery() {
    assert_tamper_rejected(&bus_forgery_ops(), 32, "chiplet OPCODE", |_, arith| {
        tamper_b32(arith, PHY_OPCODE, 0, ArithmeticOpcode::SUB as u32);
    });
}

#[test]
fn reject_chiplet_second_op_forgery() {
    let ops = bus_forgery_ops();
    assert_tamper_rejected(&ops, 32, "chiplet VAL_A second op", |_, arith| {
        tamper_b32(arith, PHY_VAL_A, 1, 99);
    });
}

// =================================================================
// 9. ADVERSARIAL - CHIPLET LOCAL CONSTRAINTS
// =================================================================

#[test]
fn reject_wrong_result_via_carry_chain() {
    let ops = vec![op_add(10, 20)];
    assert_tamper_rejected(
        &ops,
        32,
        "ADD: both sides claim VAL_RES = 31 (correct = 30); carry chain catches",
        |cpu, arith| {
            tamper_b32(cpu, CpuArithColumns::VAL_RES, 0, 31);
            tamper_b32(arith, PHY_VAL_RES, 0, 31);
        },
    );
}

#[test]
fn reject_wrong_borrow_via_sub_chain() {
    let ops = vec![op_sub(100, 50)];
    assert_tamper_rejected(
        &ops,
        32,
        "SUB: both sides claim VAL_RES = 51 (correct = 50); borrow chain catches",
        |cpu, arith| {
            tamper_b32(cpu, CpuArithColumns::VAL_RES, 0, 51);
            tamper_b32(arith, PHY_VAL_RES, 0, 51);
        },
    );
}

#[test]
fn reject_wrong_result_via_and_constraint() {
    let ops = vec![op_and(0xFF, 0x0F)];
    assert_tamper_rejected(
        &ops,
        32,
        "AND: both sides claim VAL_RES = 0x10 (correct = 0x0F); AND constraint catches",
        |cpu, arith| {
            tamper_b32(cpu, CpuArithColumns::VAL_RES, 0, 0x10);
            tamper_b32(arith, PHY_VAL_RES, 0, 0x10);
        },
    );
}

#[test]
fn reject_wrong_result_via_xor_constraint() {
    let ops = vec![op_xor(0xAAAA, 0x5555)];
    assert_tamper_rejected(
        &ops,
        32,
        "XOR: both sides claim VAL_RES = 0xFFFE (correct = 0xFFFF); XOR constraint catches",
        |cpu, arith| {
            tamper_b32(cpu, CpuArithColumns::VAL_RES, 0, 0xFFFE);
            tamper_b32(arith, PHY_VAL_RES, 0, 0xFFFE);
        },
    );
}

#[test]
fn reject_wrong_result_via_not_constraint() {
    let ops = vec![op_not(0)];
    assert_tamper_rejected(
        &ops,
        32,
        "NOT: both sides claim VAL_RES = 0xFFFFFFFE (correct = 0xFFFFFFFF); NOT constraint catches",
        |cpu, arith| {
            tamper_b32(cpu, CpuArithColumns::VAL_RES, 0, 0xFFFFFFFE);
            tamper_b32(arith, PHY_VAL_RES, 0, 0xFFFFFFFE);
        },
    );
}

#[test]
fn reject_wrong_result_via_lt_constraint() {
    let ops = vec![op_lt(10, 20)];
    assert_tamper_rejected(
        &ops,
        32,
        "LT: both sides claim VAL_RES = 0 (correct = 1); LT borrow chain catches",
        |cpu, arith| {
            tamper_b32(cpu, CpuArithColumns::VAL_RES, 0, 0);
            tamper_b32(arith, PHY_VAL_RES, 0, 0);
        },
    );
}

#[test]
fn reject_two_selectors_set() {
    let ops = vec![op_add(10, 20)];
    assert_tamper_rejected(
        &ops,
        32,
        "two selectors set: s_add=1 AND s_sub=1; pairwise mutex catches",
        |_, arith| {
            tamper_bit(arith, PHY_S_SUB, 0, 1);
        },
    );
}

#[test]
fn reject_opcode_mismatch() {
    let ops = vec![op_add(10, 20)];
    assert_tamper_rejected(
        &ops,
        32,
        "OPCODE bound: chiplet OPCODE=SUB but s_add=1 (both sides forge); opcode-bind catches",
        |cpu, arith| {
            tamper_b32(
                cpu,
                CpuArithColumns::OPCODE,
                0,
                ArithmeticOpcode::SUB as u32,
            );
            tamper_b32(arith, PHY_OPCODE, 0, ArithmeticOpcode::SUB as u32);
        },
    );
}

// =================================================================
// 10. PADDING ROWS - BUS GATE ALLOWS GARBAGE
// =================================================================

fn prove_and_verify_with_padding_tamper<T>(
    ops: &[IntArithmeticOp],
    tamper: T,
) -> Result<bool, String>
where
    T: FnOnce(&mut ColumnTrace, usize),
{
    let num_rows = ops.len().next_power_of_two().max(2);
    let arith_num_rows = num_rows;

    let ops = with_cpu_row_idx(ops);

    let mut cpu_trace = generate_cpu_trace(&ops, num_rows, 32);

    let layout = IntArithmeticLayout::compute(32);
    let arith_trace = generate_arithmetic_trace(&ops, &layout, arith_num_rows)
        .map_err(|e| format!("arith trace: {e:?}"))?;

    let padding_row = ops.len();
    assert!(padding_row < num_rows);

    tamper(&mut cpu_trace, padding_row);

    let program = ArithCpuProgram::new(32, arith_num_rows);
    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![arith_trace]);

    run_prover_verifier(&program, &instance, &witness, b"ARITH_PADDING")
}

#[test]
fn padding_val_a_garbage_still_verifies() {
    let r = prove_and_verify_with_padding_tamper(&padding_test_ops(), |cpu, row| {
        tamper_b32(cpu, CpuArithColumns::VAL_A, row, 0xDEADBEEF);
    });
    assert_eq!(r, Ok(true));
}

#[test]
fn padding_val_b_garbage_still_verifies() {
    let r = prove_and_verify_with_padding_tamper(&padding_test_ops(), |cpu, row| {
        tamper_b32(cpu, CpuArithColumns::VAL_B, row, 0xCAFEBABE);
    });
    assert_eq!(r, Ok(true));
}

#[test]
fn padding_val_res_garbage_still_verifies() {
    let r = prove_and_verify_with_padding_tamper(&padding_test_ops(), |cpu, row| {
        tamper_b32(cpu, CpuArithColumns::VAL_RES, row, 0xBADF00D);
    });
    assert_eq!(r, Ok(true));
}

#[test]
fn padding_opcode_garbage_still_verifies() {
    let r = prove_and_verify_with_padding_tamper(&padding_test_ops(), |cpu, row| {
        tamper_b32(cpu, CpuArithColumns::OPCODE, row, 0xFFFFFFFF);
    });
    assert_eq!(r, Ok(true));
}

#[test]
fn scribble_arithmetic_flip_selector_caught() {
    let bit_width = 32;
    let ops = with_cpu_row_idx(&[op_add(10, 20), op_sub(100, 50), op_and(0xFF, 0x0F)]);

    let num_rows = ops.len().next_power_of_two().max(2);
    let arith_num_rows = num_rows;

    let cpu_trace = generate_cpu_trace(&ops, num_rows, bit_width);
    let layout = IntArithmeticLayout::compute(bit_width);
    let arith_trace =
        generate_arithmetic_trace(&ops, &layout, arith_num_rows).expect("arith trace");

    let air = ArithCpuProgram::new(bit_width, arith_num_rows);
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
