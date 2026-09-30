// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

#[path = "common/mod.rs"]
mod common;

use hekate::core::trace::ColumnTrace;
use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::{Block128, TowerField};
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::trace::TraceBuilder;
use hekate_gadgets::{
    ArithmeticOpcode, CpuArithColumns, CpuFetchColumns, CpuMemColumns, Instruction,
    IntArithmeticChiplet, IntArithmeticLayout, IntArithmeticOp, MemoryEvent, RamChiplet,
    RomChiplet, generate_arithmetic_trace, generate_ram_trace, generate_rom_trace,
};
use hekate_math::{Bit, Block32};
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
// 1. COLUMN LAYOUT: CPU COLUMNS ONLY (MAIN TRACE)
// =================================================================
//
// ROM, Arithmetic, RAM chiplets now have independent traces.
// Main trace = CPU Fetch (9) + CPU Arith (5) + CPU Mem (10) = 24 columns.

const CPU_FETCH: usize = 0;
const CPU_ARITH: usize = CpuFetchColumns::NUM_COLUMNS;
const CPU_MEM: usize = CPU_ARITH + CpuArithColumns::NUM_COLUMNS;

// =================================================================
// 2. PROGRAM DEFINITION
// =================================================================

/// Three CPU schemas back to back; the chiplet
/// constraints live on their own traces.
fn build_program(
    num_rows: usize,
    rom_num_rows: usize,
    arith_num_rows: usize,
    ram_num_rows: usize,
    num_ops: usize,
) -> errors::Result<CircuitProgram<F>> {
    let mut cx = Circuit::<F>::new("AluRamRomWithChiplets", num_rows)?;

    let fetch = cx.schema(&CpuFetchColumns::build_layout());
    let arith = cx.schema(&CpuArithColumns::build_layout());
    let mem = cx.schema(&CpuMemColumns::build_layout());

    let ops_prefix = || FixedShape::Cadence {
        stride: 1,
        count: num_ops,
        origin: 0,
        values: vec![F::ONE],
    };

    cx.fix(fetch.at(CpuFetchColumns::SELECTOR), ops_prefix());
    cx.fix(arith.at(CpuArithColumns::SELECTOR), ops_prefix());
    cx.fix(mem.at(CpuMemColumns::SELECTOR), ops_prefix());

    let mut cpu_arith = IntArithmeticChiplet::cpu_linking_spec();
    cpu_arith.shift_column_indices(arith.start());

    let mut cpu_mem = RamChiplet::cpu_linking_spec();
    cpu_mem.shift_column_indices(mem.start());

    cx.bus(RomChiplet::BUS_ID, RomChiplet::cpu_linking_spec());
    cx.bus(IntArithmeticChiplet::BUS_ID, cpu_arith);
    cx.bus(RamChiplet::BUS_ID, cpu_mem);

    cx.attach(ChipletDef::from_air(&RomChiplet::new(
        rom_num_rows,
        num_ops,
    ))?);
    cx.attach(ChipletDef::from_air(&IntArithmeticChiplet::new(
        32,
        arith_num_rows,
        num_ops,
    )?)?);
    cx.attach(ChipletDef::from_air(&RamChiplet::new(
        ram_num_rows,
        num_ops,
    ))?);

    cx.compile()
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
        let IntArithmeticOp::U32 { op, a, b } = ariths[i] else {
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

    let config = Config {
        zero_knowledge: common::zero_knowledge(),
        ..Config::default()
    };

    let mut blinding_seed = Zeroizing::new([0u8; 32]);
    OsRng.try_fill_bytes(&mut *blinding_seed).unwrap();

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

    let air = build_program(
        main_num_rows,
        rom_num_rows,
        arith_num_rows,
        ram_num_rows,
        num_ops,
    )
    .unwrap();

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
            *blinding_seed,
            None,
        )
        .expect("Prover failed")
    });

    common::proof_breakdown(&proof);

    let mut verifier_transcript = Transcript::<H>::new(b"Unified_Example");

    let pinned_id = common::audited_id(&air);
    let is_valid = common::phase_with_mem("Verifying", || {
        HekateVerifier::<F, H>::verify(
            &pinned_id,
            &air,
            &instance,
            &proof,
            &mut verifier_transcript,
            &config,
        )
        .expect("Verifier failed")
    });

    common::result(is_valid);
}
