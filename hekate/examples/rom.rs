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
use hekate_gadgets::{CpuFetchColumns, Instruction, RomChiplet, generate_rom_trace};
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
// 1. ROM TEST PROGRAM DEFINITION
// =================================================================

/// CPU columns in `CpuFetchColumns` order;
/// the circuit handles match the trace generator's schema.
fn build_program(num_rows: usize) -> errors::Result<CircuitProgram<F>> {
    let mut cx = Circuit::<F>::new("RomInline", num_rows)?;

    let cpu = cx.schema(&CpuFetchColumns::build_layout());
    let selector = cpu.at(CpuFetchColumns::SELECTOR);

    cx.fix(
        selector,
        FixedShape::Cadence {
            stride: 1,
            count: num_rows,
            origin: 0,
            values: vec![F::ONE],
        },
    );

    cx.bus(RomChiplet::BUS_ID, RomChiplet::cpu_linking_spec());

    cx.mount(ChipletDef::from_air(&RomChiplet::new(num_rows, num_rows))?);

    cx.compile()
}

// =================================================================
// 2. TRACE GENERATION HELPERS
// =================================================================
fn generate_rom_instructions(num_rows: usize) -> Vec<Instruction> {
    (0..num_rows)
        .map(|i| {
            Instruction::new(
                (i * 4) as u32,
                ((i % 256) as u8).wrapping_add(1),
                [(i & 0xFF) as u8, ((i >> 8) & 0xFF) as u8, 0],
            )
        })
        .collect()
}

/// Generates a combined trace containing
/// both CPU fetch events and ROM data.
fn generate_combined_trace(
    instructions: &[Instruction],
    num_rows: usize,
) -> errors::Result<ColumnTrace> {
    let num_vars = num_rows.trailing_zeros() as usize;
    let mut tb = TraceBuilder::new_secret(&CpuFetchColumns::build_layout(), num_vars)?;

    for (i, instr) in instructions.iter().enumerate() {
        let pc_bytes = instr.pc_bytes();
        tb.set_b32(CpuFetchColumns::PC_B0, i, Block32::from(pc_bytes[0] as u32))?;
        tb.set_b32(CpuFetchColumns::PC_B1, i, Block32::from(pc_bytes[1] as u32))?;
        tb.set_b32(CpuFetchColumns::PC_B2, i, Block32::from(pc_bytes[2] as u32))?;
        tb.set_b32(CpuFetchColumns::PC_B3, i, Block32::from(pc_bytes[3] as u32))?;

        tb.set_b32(
            CpuFetchColumns::OPCODE,
            i,
            Block32::from(instr.opcode() as u32),
        )?;

        let args = instr.args();
        tb.set_b32(CpuFetchColumns::ARG0, i, Block32::from(args[0] as u32))?;
        tb.set_b32(CpuFetchColumns::ARG1, i, Block32::from(args[1] as u32))?;
        tb.set_b32(CpuFetchColumns::ARG2, i, Block32::from(args[2] as u32))?;
        tb.set_bit(CpuFetchColumns::SELECTOR, i, Bit::ONE)?;
    }

    let mut trace = tb.build();

    // Append ROM chiplet columns
    let rom = generate_rom_trace(instructions, num_rows)?;
    for col in rom.into_columns() {
        trace.add_column(col)?;
    }

    Ok(trace)
}

fn main() {
    common::init("ROM Chiplet");

    let num_vars = 20;
    let num_rows = 1 << num_vars;

    let config = Config {
        zero_knowledge: common::zero_knowledge(),
        ..Config::default()
    };

    let mut blinding_seed = Zeroizing::new([0u8; 32]);
    OsRng.try_fill_bytes(&mut *blinding_seed).unwrap();

    println!(
        "Rows: 2^{} ({} million)",
        num_vars,
        num_rows as f64 / 1_000_000.0
    );

    let trace = common::phase("Trace Generation", || {
        let instructions = generate_rom_instructions(num_rows);
        generate_combined_trace(&instructions, num_rows).unwrap()
    });

    let air = build_program(num_rows).unwrap();
    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(trace);

    let proof = common::phase("Proving", || {
        prove(
            b"ROM_Example",
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

    let mut verifier_transcript = Transcript::<H>::new(b"ROM_Example");
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
