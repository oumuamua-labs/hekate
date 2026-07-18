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

#[path = "common/mod.rs"]
mod common;

use hekate::core::trace::{ColumnTrace, ColumnType};
use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::{Block128, TowerField};
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::trace::{TraceBuilder, TraceCompatibleField};
use hekate_gadgets::{
    CpuMemColumns, CpuMemoryUnit, MemoryEvent, RamChiplet, RamColumns, generate_ram_trace,
};
use hekate_math::{Bit, Block32, Flat, FlatPromote};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::expander::VirtualExpander;
use hekate_program::permutation::PermutationCheckSpec;
use hekate_program::{Air, InlineKernelHint, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;
use rand::{TryRngCore, rngs::OsRng};

type F = Block128;
type H = DefaultHasher;

const RAM_OFFSET: usize = CpuMemColumns::NUM_COLUMNS;

// =================================================================
// 1. PROGRAM DEFINITION
// =================================================================

#[derive(Clone)]
struct RamInlineProgram {
    num_rows: usize,
}

impl Air<F> for RamInlineProgram {
    fn num_columns(&self) -> usize {
        CpuMemColumns::NUM_COLUMNS + RamColumns::NUM_COLUMNS
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();

        LAYOUT.get_or_init(|| {
            let mut cols = CpuMemColumns::build_layout();
            cols.extend(RamChiplet::build_physical_layout());

            cols
        })
    }

    fn virtual_column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();

        LAYOUT.get_or_init(|| {
            let mut cols = CpuMemColumns::build_layout();
            cols.extend(RamColumns::build_layout());

            cols
        })
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        let cpu_spec = CpuMemoryUnit::linking_spec();

        let mut ram_spec = RamChiplet::linking_spec();
        ram_spec.shift_column_indices(RAM_OFFSET);

        vec![
            (RamChiplet::BUS_ID.to_string(), cpu_spec),
            (RamChiplet::BUS_ID.to_string(), ram_spec),
        ]
    }

    fn virtual_expander(&self) -> Option<&VirtualExpander> {
        static E: std::sync::OnceLock<VirtualExpander> = std::sync::OnceLock::new();
        Some(E.get_or_init(|| {
            // CPU:
            // 8 B32 (addr_b0..3 + val_b0..3) + 2 Bit (IS_WRITE, SELECTOR).
            // RAM physical:
            // 2 packed B32 + 13 B32 data + 1 B128 + 5 Bit selectors.
            VirtualExpander::new()
                .pass_through(8, ColumnType::B32)
                .control_bits(2)
                .expand_bits(2, ColumnType::B32)
                .pass_through(13, ColumnType::B32)
                .pass_through(1, ColumnType::B128)
                .control_bits(5)
                .build()
                .expect("ram inline expander")
        }))
    }

    fn parse_virtual_row(&self, bytes: &[u8], res: &mut Vec<Flat<F>>)
    where
        F: TraceCompatibleField,
    {
        res.clear();

        // CPU columns:
        // 4 addr B32 + 4 val B32 + 2 Bit = 34 bytes
        let cpu_size = 4 * 4 + 4 * 4 + 2;
        let (cpu_bytes, ram_bytes) = bytes.split_at(cpu_size);

        // CPU:
        // 1:1 (B32 columns then Bit columns)
        let mut offset = 0;
        for _ in 0..8 {
            let mut arr = [0u8; 4];
            arr.copy_from_slice(&cpu_bytes[offset..offset + 4]);

            let val = Block32(u32::from_le_bytes(arr));
            res.push(F::promote_flat(Flat::from_raw(val)));

            offset += 4;
        }

        for _ in 0..2 {
            let val = cpu_bytes[offset] & 1;
            res.push(Flat::from_raw(F::from(Bit::from(val))));

            offset += 1;
        }

        // RAM:
        // virtual expansion
        let ram = RamChiplet::new(1);
        Air::<F>::virtual_expander(&ram)
            .unwrap()
            .parse_row(ram_bytes, res)
            .expect("ram row parse");
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(CpuMemColumns::SELECTOR));

        let mut ast = cs.build();

        let mut ram_ast = RamChiplet::new(self.num_rows).constraint_ast();
        ram_ast.arena.shift_cells(RAM_OFFSET);

        ast.merge(ram_ast);

        ast
    }

    fn inline_chiplets(&self) -> errors::Result<Vec<ChipletDef<F>>> {
        Ok(vec![ChipletDef::from_air(&RamChiplet::new(self.num_rows))?])
    }

    fn inline_chiplet_kernels(&self) -> Vec<InlineKernelHint> {
        vec![InlineKernelHint {
            chiplet_idx: 0,
            root_offset: 1,
            column_offset: RAM_OFFSET,
        }]
    }
}

impl Program<F> for RamInlineProgram {}

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
    let mut tb = TraceBuilder::new(&CpuMemColumns::build_layout(), num_vars)?;

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
        sumcheck_blinding_factor: 2,
        ..Config::default()
    };

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    println!("Rows: 2^{} ({} million)", num_vars, num_rows as f64 / 1e6);

    let (trace, air) = common::phase("Trace Generation", || {
        let events = generate_memory_workload(num_rows);
        let trace = generate_combined_trace(&events, num_rows).unwrap();
        let air = RamInlineProgram { num_rows };

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
            blinding_seed,
            None,
        )
        .expect("Prover failed")
    });

    common::proof_breakdown(&proof);

    let mut verifier_transcript = Transcript::<H>::new(b"RAM_Example");

    let is_valid = common::phase_with_mem("Verifying", || {
        HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut verifier_transcript, &config)
            .expect("Verifier failed")
    });

    common::result(is_valid);
}
