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
use hekate_gadgets::{CpuMemColumns, MemoryEvent, RamChiplet, generate_ram_trace};
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
// 1. PROGRAM DEFINITION
// =================================================================

/// CPU columns in `CpuMemColumns` order;
/// the circuit handles match the trace generator's schema.
fn build_program(num_rows: usize) -> errors::Result<CircuitProgram<F>> {
    let mut cx = Circuit::<F>::new("RamInline", num_rows)?;

    let cpu = cx.schema(&CpuMemColumns::build_layout());
    let selector = cpu.at(CpuMemColumns::SELECTOR);

    cx.fix(
        selector,
        FixedShape::Cadence {
            stride: 1,
            count: num_rows / 2,
            origin: 0,
            values: vec![F::ONE],
        },
    );

    cx.bus(RamChiplet::BUS_ID, RamChiplet::cpu_linking_spec());

    cx.mount(ChipletDef::from_air(&RamChiplet::new(
        num_rows,
        num_rows / 2,
    ))?);

    cx.compile()
}

// =================================================================
// 3. WORKLOAD & TRACE GENERATION
// =================================================================

fn generate_memory_workload(num_rows: usize) -> Vec<MemoryEvent> {
    let mut events = Vec::new();
    let num_ops = num_rows / 2;

    for i in 0..num_ops {
        let addr = (i * 4) as u32;
        let val = 0xDEADBEEF ^ (i as u32);

        if i % 2 == 0 {
            events.push(MemoryEvent::write(addr, i as u32, val));
        } else {
            let prev_addr = ((i - 1) * 4) as u32;
            let prev_val = 0xDEADBEEF ^ ((i - 1) as u32);

            events.push(MemoryEvent::read(prev_addr, i as u32, prev_val));
        }
    }

    events
}

fn generate_combined_trace(events: &[MemoryEvent], num_rows: usize) -> errors::Result<ColumnTrace> {
    let num_vars = num_rows.trailing_zeros() as usize;

    // CPU trace
    let mut tb = TraceBuilder::new_secret(&CpuMemColumns::build_layout(), num_vars)?;

    for (i, event) in events.iter().enumerate() {
        let addr_bytes = event.addr_bytes();
        let val_bytes = event.val_bytes();

        for j in 0..4 {
            tb.set_b32(
                CpuMemColumns::ADDR_B0 + j,
                i,
                Block32::from(addr_bytes[j] as u32),
            )?;
            tb.set_b32(
                CpuMemColumns::VAL_B0 + j,
                i,
                Block32::from(val_bytes[j] as u32),
            )?;
        }

        tb.set_bit(
            CpuMemColumns::IS_WRITE,
            i,
            if event.is_write { Bit::ONE } else { Bit::ZERO },
        )?;
        tb.set_bit(CpuMemColumns::SELECTOR, i, Bit::ONE)?;
    }

    let mut trace = tb.build();

    // Append RAM physical columns (20 columns)
    let ram = generate_ram_trace(events, num_rows)?;
    for col in ram.into_columns() {
        trace.add_column(col)?;
    }

    Ok(trace)
}

// =================================================================
// 4. MAIN
// =================================================================

fn main() {
    common::init("RAM Chiplet");

    let num_vars = 20; // 1M rows
    let num_rows = 1 << num_vars;

    let config = Config {
        zero_knowledge: common::zero_knowledge(),
        ..Config::default()
    };

    let mut blinding_seed = Zeroizing::new([0u8; 32]);
    OsRng.try_fill_bytes(&mut *blinding_seed).unwrap();

    println!("Rows: 2^{} ({} million)", num_vars, num_rows as f64 / 1e6);

    let (trace, air) = common::phase("Trace Generation", || {
        let events = generate_memory_workload(num_rows);
        let trace = generate_combined_trace(&events, num_rows).unwrap();
        let air = build_program(num_rows).unwrap();

        (trace, air)
    });

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(trace);

    let proof = common::phase("Proving", || {
        prove(
            b"RAM_Example",
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

    let mut verifier_transcript = Transcript::<H>::new(b"RAM_Example");
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
