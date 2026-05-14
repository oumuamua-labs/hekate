// SPDX-License-Identifier: Apache-2.0
// This file is part of the hekate-math project.
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

use hekate::core::trace::ColumnTrace;
use hekate::core::trace::ColumnType;
use hekate::crypto::transcript::Transcript;
use hekate::crypto::DefaultHasher;
use hekate::math::{Block128, TowerField};
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::trace::TraceBuilder;
use hekate_gadgets::{
    generate_arithmetic_trace, generate_ram_trace, generate_rom_trace, ArithmeticOpcode,
    CpuArithColumns, CpuFetchColumns, CpuFetchUnit, CpuIntArithmeticUnit, CpuMemColumns,
    CpuMemoryUnit, Instruction, IntArithmeticChiplet, IntArithmeticLayout, IntArithmeticOp,
    MemoryEvent, RamChiplet, RomChiplet,
};
use hekate_math::{Bit, Block32};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::constraint::ConstraintAst;
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;
use rand::{rngs::OsRng, TryRngCore};

type F = Block128;
type H = DefaultHasher;

// =================================================================
// 1. COLUMN LAYOUT — CPU COLUMNS ONLY (MAIN TRACE)
// =================================================================
//
// ROM, Arithmetic, RAM chiplets now have independent traces.
// Main trace = CPU Fetch (9) + CPU Arith (5) + CPU Mem (10) = 24 columns.

const CPU_FETCH: usize = 0;
const CPU_ARITH: usize = CpuFetchColumns::NUM_COLUMNS;
const CPU_MEM: usize = CPU_ARITH + CpuArithColumns::NUM_COLUMNS;
const NUM_CPU_COLS: usize = CPU_MEM + CpuMemColumns::NUM_COLUMNS;

// =================================================================
// 2. PROGRAM DEFINITION
// =================================================================
#[derive(Clone)]
struct ManyInlineChipletsProgram {
    rom_num_rows: usize,
    arith_num_rows: usize,
    ram_num_rows: usize,
}

impl Air<F> for ManyInlineChipletsProgram {
    fn name(&self) -> String {
        "AluRamRomWithChiplets".to_string()
    }

    fn num_columns(&self) -> usize {
        NUM_CPU_COLS
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();

        LAYOUT.get_or_init(|| {
            let mut cols = CpuFetchColumns::build_layout();
            cols.extend(CpuArithColumns::build_layout());
            cols.extend(CpuMemColumns::build_layout());

            cols
        })
    }

    fn permutation_checks(
        &self,
    ) -> Vec<(String, hekate_program::permutation::PermutationCheckSpec)> {
        // CPU-side bus endpoints only.
        // bus_ids match the chiplet supply-side IDs.
        let cpu_fetch = CpuFetchUnit::linking_spec();

        let mut cpu_arith = CpuIntArithmeticUnit::linking_spec();
        cpu_arith.shift_column_indices(CPU_ARITH);

        let mut cpu_mem = CpuMemoryUnit::linking_spec();
        cpu_mem.shift_column_indices(CPU_MEM);

        vec![
            (RomChiplet::BUS_ID.into(), cpu_fetch),
            (IntArithmeticChiplet::BUS_ID.into(), cpu_arith),
            (RamChiplet::BUS_ID.into(), cpu_mem),
        ]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        // CPU selector booleanity,
        // the only main-trace constraints.
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(CPU_FETCH + CpuFetchColumns::SELECTOR));
        cs.assert_boolean(cs.col(CPU_ARITH + CpuArithColumns::SELECTOR));
        cs.assert_boolean(cs.col(CPU_MEM + CpuMemColumns::SELECTOR));

        // ROM, Arithmetic, RAM constraints
        // live on their own chiplet traces.
        cs.build()
    }
}

impl Program<F> for ManyInlineChipletsProgram {
    fn chiplet_defs(&self) -> errors::Result<Vec<ChipletDef<F>>> {
        let rom = RomChiplet::new(self.rom_num_rows);
        let arith = IntArithmeticChiplet::new(32, self.arith_num_rows)
            .expect("IntArithmeticChiplet::new(32, arith_num_rows)");
        let ram = RamChiplet::new(self.ram_num_rows);

        Ok(vec![
            ChipletDef::from_air(&rom)?,
            ChipletDef::from_air(&arith)?,
            ChipletDef::from_air(&ram)?,
        ])
    }
}

// =================================================================
// 3. WORKLOAD & TRACE GENERATION
// =================================================================
fn generate_workload(num_ops: usize) -> (Vec<Instruction>, Vec<IntArithmeticOp>, Vec<MemoryEvent>) {
    let mut instrs = Vec::new();
    let mut ariths = Vec::new();
    let mut mems = Vec::new();

    for i in 0..num_ops {
        instrs.push(Instruction::new((i * 4) as u32, 0x01, [0, 0, 0]));

        let val_a = (i * 100) as u32;
        let val_b = 0xAAAA_BBBB;

        let (opcode, b_val, result) = match i % 6 {
            0 => (ArithmeticOpcode::ADD, val_b, val_a.wrapping_add(val_b)),
            1 => (ArithmeticOpcode::SUB, val_b, val_a.wrapping_sub(val_b)),
            2 => (ArithmeticOpcode::AND, val_b, val_a & val_b),
            3 => (ArithmeticOpcode::XOR, val_b, val_a ^ val_b),
            4 => (ArithmeticOpcode::NOT, 0, !val_a),
            5 => (ArithmeticOpcode::LT, val_b, (val_a < val_b) as u32),
            _ => unreachable!(),
        };

        ariths.push(IntArithmeticOp::U32 {
            op: opcode,
            a: val_a,
            b: b_val,
            request_idx: i as u32,
        });

        mems.push(MemoryEvent::write((i * 4) as u32, i as u32, result));
    }

    (instrs, ariths, mems)
}

fn generate_cpu_trace(
    instrs: &[Instruction],
    ariths: &[IntArithmeticOp],
    mems: &[MemoryEvent],
    num_rows: usize,
) -> errors::Result<ColumnTrace> {
    let num_vars = num_rows.trailing_zeros() as usize;

    let mut layout = CpuFetchColumns::build_layout();
    layout.extend(CpuArithColumns::build_layout());
    layout.extend(CpuMemColumns::build_layout());

    let mut tb = TraceBuilder::new(&layout, num_vars)?;

    for (i, instr) in instrs.iter().enumerate() {
        let r = i;
        if r >= num_rows {
            break;
        }

        let pc = instr.pc_bytes();
        let args = instr.args();

        // CPU Fetch columns (offset CPU_FETCH = 0)
        tb.set_b32(
            CPU_FETCH + CpuFetchColumns::PC_B0,
            r,
            Block32::from(pc[0] as u32),
        )?;
        tb.set_b32(
            CPU_FETCH + CpuFetchColumns::PC_B1,
            r,
            Block32::from(pc[1] as u32),
        )?;
        tb.set_b32(
            CPU_FETCH + CpuFetchColumns::PC_B2,
            r,
            Block32::from(pc[2] as u32),
        )?;
        tb.set_b32(
            CPU_FETCH + CpuFetchColumns::PC_B3,
            r,
            Block32::from(pc[3] as u32),
        )?;
        tb.set_b32(
            CPU_FETCH + CpuFetchColumns::OPCODE,
            r,
            Block32::from(instr.opcode as u32),
        )?;

        tb.set_b32(
            CPU_FETCH + CpuFetchColumns::ARG0,
            r,
            Block32::from(args[0] as u32),
        )?;
        tb.set_b32(
            CPU_FETCH + CpuFetchColumns::ARG1,
            r,
            Block32::from(args[1] as u32),
        )?;
        tb.set_b32(
            CPU_FETCH + CpuFetchColumns::ARG2,
            r,
            Block32::from(args[2] as u32),
        )?;
        tb.set_bit(CPU_FETCH + CpuFetchColumns::SELECTOR, r, Bit::ONE)?;

        // CPU Arith columns (offset CPU_ARITH = 9)
        let IntArithmeticOp::U32 {
            op,
            a,
            b,
            request_idx: _,
        } = ariths[i]
        else {
            unreachable!("many_chiplets is u32-only");
        };

        let res = match op {
            ArithmeticOpcode::ADD => a.wrapping_add(b),
            ArithmeticOpcode::SUB => a.wrapping_sub(b),
            ArithmeticOpcode::AND => a & b,
            ArithmeticOpcode::XOR => a ^ b,
            ArithmeticOpcode::NOT => !a,
            ArithmeticOpcode::LT => (a < b) as u32,
        };

        tb.set_b32(CPU_ARITH + CpuArithColumns::VAL_A, r, Block32::from(a))?;
        tb.set_b32(CPU_ARITH + CpuArithColumns::VAL_B, r, Block32::from(b))?;
        tb.set_b32(CPU_ARITH + CpuArithColumns::VAL_RES, r, Block32::from(res))?;
        tb.set_b32(
            CPU_ARITH + CpuArithColumns::OPCODE,
            r,
            Block32::from(op as u8 as u32),
        )?;
        tb.set_bit(CPU_ARITH + CpuArithColumns::SELECTOR, r, Bit::ONE)?;

        // CPU Mem columns (offset CPU_MEM = 14)
        let mem = &mems[i];
        let addr = mem.addr_bytes();
        let val = mem.val_bytes();

        tb.set_b32(
            CPU_MEM + CpuMemColumns::ADDR_B0,
            r,
            Block32::from(addr[0] as u32),
        )?;
        tb.set_b32(
            CPU_MEM + CpuMemColumns::ADDR_B1,
            r,
            Block32::from(addr[1] as u32),
        )?;
        tb.set_b32(
            CPU_MEM + CpuMemColumns::ADDR_B2,
            r,
            Block32::from(addr[2] as u32),
        )?;
        tb.set_b32(
            CPU_MEM + CpuMemColumns::ADDR_B3,
            r,
            Block32::from(addr[3] as u32),
        )?;
        tb.set_b32(
            CPU_MEM + CpuMemColumns::VAL_B0,
            r,
            Block32::from(val[0] as u32),
        )?;
        tb.set_b32(
            CPU_MEM + CpuMemColumns::VAL_B1,
            r,
            Block32::from(val[1] as u32),
        )?;
        tb.set_b32(
            CPU_MEM + CpuMemColumns::VAL_B2,
            r,
            Block32::from(val[2] as u32),
        )?;
        tb.set_b32(
            CPU_MEM + CpuMemColumns::VAL_B3,
            r,
            Block32::from(val[3] as u32),
        )?;

        tb.set_bit(
            CPU_MEM + CpuMemColumns::IS_WRITE,
            r,
            if mem.is_write { Bit::ONE } else { Bit::ZERO },
        )?;
        tb.set_bit(CPU_MEM + CpuMemColumns::SELECTOR, r, Bit::ONE)?;
    }

    Ok(tb.build())
}

fn main() {
    common::init("ALU + RAM + ROM Chiplets");

    // Workload parameters
    let num_ops: usize = 1 << 20;

    // Derive trace heights per table
    let main_num_rows = num_ops.next_power_of_two();
    let main_num_vars = main_num_rows.trailing_zeros() as usize;

    // ROM:
    // 1 row per instruction
    let rom_num_rows = num_ops.next_power_of_two();
    let rom_num_vars = rom_num_rows.trailing_zeros() as usize;

    // Arithmetic:
    // 1 row per operation
    let arith_num_rows = num_ops.next_power_of_two();
    let arith_num_vars = arith_num_rows.trailing_zeros() as usize;

    // RAM:
    // 1 row per memory event
    let ram_num_rows = num_ops.next_power_of_two();
    let ram_num_vars = ram_num_rows.trailing_zeros() as usize;

    let mut config = Config {
        sumcheck_blinding_factor: 2,
        ..Config::default()
    };

    OsRng.try_fill_bytes(&mut config.matrix_seed).unwrap();

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    println!("  Operations:     {}", num_ops);
    println!(
        "  Main trace:     2^{} ({} rows)",
        main_num_vars, main_num_rows
    );
    println!(
        "  ROM chiplet:    2^{} ({} rows)",
        rom_num_vars, rom_num_rows
    );
    println!(
        "  Arith chiplet:  2^{} ({} rows)",
        arith_num_vars, arith_num_rows
    );
    println!(
        "  RAM chiplet:    2^{} ({} rows)",
        ram_num_vars, ram_num_rows
    );

    let (cpu_trace, rom_trace, arith_trace, ram_trace) = common::phase("Trace Generation", || {
        let (instrs, ariths, mems) = generate_workload(num_ops);

        // Main trace:
        // CPU columns only
        let cpu_trace = generate_cpu_trace(&instrs, &ariths, &mems, main_num_rows).unwrap();

        // Independent chiplet traces
        let rom_trace = generate_rom_trace(&instrs, rom_num_rows).unwrap();
        let ram_trace = generate_ram_trace(&mems, ram_num_rows).unwrap();

        let arith_layout = IntArithmeticLayout::compute(32);
        let arith_trace =
            generate_arithmetic_trace(&ariths, &arith_layout, arith_num_rows).unwrap();

        (cpu_trace, rom_trace, arith_trace, ram_trace)
    });

    println!(
        "   CPU cols: {}  |  ROM: {}  |  Arith: {}  |  RAM: {}",
        cpu_trace.columns.len(),
        rom_trace.columns.len(),
        arith_trace.columns.len(),
        ram_trace.columns.len(),
    );

    let air = ManyInlineChipletsProgram {
        rom_num_rows,
        arith_num_rows,
        ram_num_rows,
    };

    let instance = ProgramInstance::new(main_num_rows, vec![]);
    let witness =
        ProgramWitness::new(cpu_trace).with_chiplets(vec![rom_trace, arith_trace, ram_trace]);

    let proof = common::phase("Proving", || {
        prove(
            b"Unified_Example",
            &air,
            &instance,
            &witness,
            &config,
            blinding_seed,
            None,
        )
        .expect("Prover failed")
    });

    common::proof_breakdown(&proof);

    let mut verifier_transcript = Transcript::<H>::new(b"Unified_Example");

    let is_valid = common::phase("Verifying", || {
        HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut verifier_transcript, &config)
            .expect("Verifier failed")
    });

    common::result(is_valid);
}
