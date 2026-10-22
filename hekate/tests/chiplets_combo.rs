// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate::core::config::Config;
use hekate::core::trace::{ColumnTrace, Trace, TraceColumn};
use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::{Block128, TowerField};
use hekate_core::trace::IntoTraceColumn;
use hekate_gadgets::{CpuFetchColumns, Instruction, RomChiplet, RomColumns, generate_rom_trace};
use hekate_math::{Bit, Block32};
use hekate_program::circuit::{Circuit, CircuitProgram, Col};
use hekate_program::digest::program_id;
use hekate_program::permutation::BUS_LIMBS;
use hekate_program::{Air, FixedShape, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;

type F = Block128;
type H = DefaultHasher;

// --- 1. Define Combined AIR using Library Specs ---

// Both endpoints sit on the main trace for this
// combined AIR, they share the ROM's canonical
// bus_id and their paired LogUp sums must cancel.
fn cpu_rom_air(num_rows: usize) -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("CpuRom", num_rows).unwrap();

    let cpu = cx.schema(&CpuFetchColumns::build_layout());
    let rom = cx.schema(&RomColumns::build_layout());

    let full_prefix = || FixedShape::Cadence {
        stride: 1,
        count: num_rows,
        origin: 0,
        values: vec![F::ONE],
    };

    cx.fix(cpu.at(CpuFetchColumns::SELECTOR), full_prefix());
    cx.fix(rom.at(RomColumns::SELECTOR), full_prefix());

    let values: Vec<Col> = [
        CpuFetchColumns::PC_B0,
        CpuFetchColumns::PC_B1,
        CpuFetchColumns::PC_B2,
        CpuFetchColumns::PC_B3,
        CpuFetchColumns::OPCODE,
        CpuFetchColumns::ARG0,
        CpuFetchColumns::ARG1,
        CpuFetchColumns::ARG2,
    ]
    .map(|c| cpu.at(c))
    .to_vec();

    cx.call(
        &RomChiplet::service(),
        &values,
        cpu.at(CpuFetchColumns::SELECTOR),
    )
    .unwrap();

    let mut rom_spec = RomChiplet::linking_spec();
    rom_spec.shift_column_indices(rom.start());

    cx.bus(RomChiplet::BUS_ID, rom_spec);

    cx.compile().unwrap()
}

// --- 2. Trace Generation ---

fn generate_combined_trace(num_vars: usize) -> ColumnTrace {
    let num_rows = 1 << num_vars;

    // A. Generate ROM Instructions
    let mut instructions = Vec::new();
    for i in 0..num_rows {
        instructions.push(Instruction::new(i as u32, 1, [0, 0, 0]));
    }

    // Initialize Trace
    let mut trace = ColumnTrace::new(num_vars).unwrap();

    // B. Generate CPU Data manually (Simulation)
    // CPU Fetch Unit has 6 columns: PC(4) + Opcode(1) + Selector(1)

    // PC Bytes (0-3)
    let mut pc_cols = (0..4)
        .map(|_| Vec::with_capacity(num_rows))
        .collect::<Vec<_>>();

    // Opcode (4)
    let mut op_col = Vec::with_capacity(num_rows);

    // CPU side argument buffers
    let mut arg_cols = (0..3)
        .map(|_| Vec::with_capacity(num_rows))
        .collect::<Vec<_>>();

    // Selector (5)
    let mut sel_col = Vec::with_capacity(num_rows);

    for instr in &instructions {
        let bytes = instr.pc_bytes();
        for b in 0..4 {
            pc_cols[b].push(Block32::from(bytes[b] as u32));
        }

        op_col.push(Block32::from(instr.opcode as u32));

        let args = instr.args();
        for a in 0..3 {
            arg_cols[a].push(Block32::from(args[a] as u32));
        }

        sel_col.push(Bit::ONE);
    }

    // Add CPU Columns to Trace (Indices 0..5)
    for col in pc_cols {
        trace.add_column(col.into_trace_column()).unwrap();
    }

    trace.add_column(op_col.into_trace_column()).unwrap();

    for col in arg_cols {
        trace.add_column(col.into_trace_column()).unwrap();
    }

    trace.add_column(TraceColumn::Bit(sel_col)).unwrap();

    // C. Generate ROM Data using Library Helper
    let rom = generate_rom_trace(&instructions, num_rows).unwrap();
    for col in rom.into_columns() {
        trace.add_column(col).unwrap();
    }

    trace
}

#[test]
fn chiplets_integration() {
    let num_vars = 8; // 256 rows
    let num_rows = 1 << num_vars;
    let seed = [0u8; 32];

    let air = cpu_rom_air(num_rows);
    let trace = generate_combined_trace(num_vars);

    // Verify trace dimensions match AIR expectation
    assert_eq!(trace.num_cols(), air.num_columns());

    let witness = ProgramWitness::new(trace);
    let instance = ProgramInstance::new(num_rows, vec![]);

    let config = Config {
        num_queries: 4,
        min_security_bits: 0,
        zero_knowledge: true,
        ..Config::default()
    };

    // 1. Prove
    println!("-> Proving...");
    let proof = prove(
        b"ChipletTestRefactored",
        &air,
        &instance,
        &witness,
        &config,
        seed,
        None,
    )
    .expect("Proving failed");

    // 2. Verify
    println!("-> Verifying...");

    let mut verifier_transcript = Transcript::<H>::new(b"ChipletTestRefactored");
    let pinned_id = program_id(&air).unwrap();

    let result = HekateVerifier::<F, H>::verify(
        &pinned_id,
        &air,
        &instance,
        &proof,
        &mut verifier_transcript,
        &config,
    );

    assert!(result.unwrap(), "Verification failed");

    // 3. Bus Consistency
    // Both spec endpoints share a single bus_id
    // on the main trace; `verify()` rejects if
    // their LogUp sums do not cancel.
    assert_eq!(proof.main_logup_aux.claimed_sums.len(), 2 * BUS_LIMBS);
    assert!(proof.chiplet_logup_aux.is_empty());

    println!("BUS SECURE: Chiplets are cryptographically linked.");
}
