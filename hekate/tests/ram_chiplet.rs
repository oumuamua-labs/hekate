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
use hekate::math::{Block128, TowerField};
use hekate_core::trace::TraceBuilder;
use hekate_gadgets::{
    CpuMemColumns, CpuMemoryUnit, MemoryEvent, RamChiplet, RamColumns, generate_ram_trace,
};
use hekate_math::{Bit, Block32};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::permutation::PermutationCheckSpec;
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_scribble::{MutationKind, ScribbleConfig, assert_all_caught_all_targets};
use hekate_verifier::HekateVerifier;

type F = Block128;
type H = DefaultHasher;

// =================================================================
// RAM Program:
// CPU-only main trace, RAM as independent chiplet.
// =================================================================

#[derive(Clone)]
struct RamTestAir {
    num_rows: usize,
}

impl Air<F> for RamTestAir {
    fn num_columns(&self) -> usize {
        CpuMemColumns::NUM_COLUMNS
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(CpuMemColumns::build_layout)
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![(RamChiplet::BUS_ID.into(), CpuMemoryUnit::linking_spec())]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();

        cs.assert_boolean(cs.col(CpuMemColumns::SELECTOR));
        cs.assert_boolean(cs.col(CpuMemColumns::IS_WRITE));

        let one = cs.one();
        let s = cs.col(CpuMemColumns::SELECTOR);

        cs.assert_zero_when(one - s, cs.col(CpuMemColumns::IS_WRITE));

        cs.build()
    }
}

impl Program<F> for RamTestAir {
    fn chiplet_defs(&self) -> hekate_core::errors::Result<Vec<ChipletDef<F>>> {
        let ram = RamChiplet::new(self.num_rows);
        Ok(vec![ChipletDef::from_air(&ram)?])
    }
}

// =================================================================
// Exploit-only AIR:
// inline merge (no chiplet path)
// =================================================================

/// This AIR merges RAM columns into the main trace
/// for exploit testing. Uses virtual layout directly.
#[derive(Clone)]
struct RamExploitAir {
    num_rows: usize,
}

impl Air<F> for RamExploitAir {
    fn num_columns(&self) -> usize {
        CpuMemColumns::NUM_COLUMNS + RamColumns::NUM_COLUMNS
    }

    fn column_layout(&self) -> &[ColumnType] {
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
        ram_spec.shift_column_indices(CpuMemColumns::NUM_COLUMNS);

        vec![
            ("cpu_mem".to_string(), cpu_spec),
            ("ram_mem".to_string(), ram_spec),
        ]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(CpuMemColumns::SELECTOR));

        let mut ast = cs.build();

        let mut ram_ast = RamChiplet::new(self.num_rows).constraint_ast();
        ram_ast.arena.shift_cells(CpuMemColumns::NUM_COLUMNS);

        ast.merge(ram_ast);

        ast
    }
}

impl Program<F> for RamExploitAir {}

// =================================================================
// Trace Generation
// =================================================================

/// Generate CPU-only main
/// trace from memory events.
fn generate_cpu_trace(events: &[MemoryEvent], num_rows: usize) -> ColumnTrace {
    let num_vars = num_rows.trailing_zeros() as usize;

    let mut tb = TraceBuilder::new(&CpuMemColumns::build_layout(), num_vars).unwrap();

    for (i, event) in events.iter().enumerate() {
        let addr_bytes = event.addr_bytes();
        let val_bytes = event.val_bytes();

        for j in 0..4 {
            tb.set_b32(
                CpuMemColumns::ADDR_B0 + j,
                i,
                Block32::from(addr_bytes[j] as u32),
            )
            .unwrap();
            tb.set_b32(
                CpuMemColumns::VAL_B0 + j,
                i,
                Block32::from(val_bytes[j] as u32),
            )
            .unwrap();
        }

        tb.set_bit(
            CpuMemColumns::IS_WRITE,
            i,
            if event.is_write { Bit::ONE } else { Bit::ZERO },
        )
        .unwrap();
        tb.set_bit(CpuMemColumns::SELECTOR, i, Bit::ONE).unwrap();
    }

    tb.build()
}

// =================================================================
// E2E Tests
// =================================================================

/// E2E RAM-CPU Linking Test via HekateProver.
///
/// RAM runs as an independent chiplet with its own
/// trace, commitment, and ZeroCheck. Connected to
/// the CPU via Grand Product Argument.
#[test]
fn ram_cpu_linking() {
    let num_vars = 4; // 16 rows
    let num_rows = 1 << num_vars;
    let seed = [0xAAu8; 32];

    // Define memory operations (execution order)
    let events = vec![
        MemoryEvent::write(0x1000, 0, 42),  // clk=0: Write 42 to 0x1000
        MemoryEvent::write(0x2000, 1, 99),  // clk=1: Write 99 to 0x2000
        MemoryEvent::read(0x1000, 2, 42),   // clk=2: Read 0x1000 (should be 42)
        MemoryEvent::write(0x1004, 3, 123), // clk=3: Write 123 to 0x1004
        MemoryEvent::read(0x2000, 4, 99),   // clk=4: Read 0x2000 (should be 99)
        MemoryEvent::read(0x1000, 5, 42),   // clk=5: Read 0x1000 again (should be 42)
    ];

    // 1. Setup
    let air = RamTestAir { num_rows };

    let cpu_trace = generate_cpu_trace(&events, num_rows);
    let ram_trace = generate_ram_trace(&events, num_rows).unwrap();

    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![ram_trace]);
    let instance = ProgramInstance::new(num_rows, vec![]);

    let config = Config {
        num_queries: 4,
        min_security_bits: 0,
        sumcheck_blinding_factor: 2, // Enable ZK
        ..Config::default()
    };

    // 2. Prove
    println!("-> Proving...");

    let proof =
        prove(b"RAM_E2E", &air, &instance, &witness, &config, seed, None).expect("Proving failed");

    // 3. Verify
    println!("-> Verifying...");

    let mut verifier_transcript = Transcript::<H>::new(b"RAM_E2E");
    let result =
        HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut verifier_transcript, &config);

    assert!(result.unwrap(), "Verification failed");

    // 4. Check Bus Consistency
    // `verify()` above has already run
    // `check_bus_sum_matching`on main
    // + chiplet LogUp endpoints. Here we
    // just assert the expected layout:
    // one bus on main, one on the RAM chiplet.
    assert_eq!(proof.main_logup_aux.claimed_sums.len(), 1);
    assert_eq!(proof.chiplet_logup_aux.len(), 1);
    assert_eq!(proof.chiplet_logup_aux[0].claimed_sums.len(), 1);
}

// =================================================================
// SECURITY EXPLOIT TESTS
// =================================================================

/// EXPLOIT TEST:
/// RAM Consistency Bypass
///
/// Scenario:
/// We manually construct a malicious trace that violates Memory Consistency.
/// 1. WRITE: Addr 0x10 -> Val 100
/// 2. READ:  Addr 0x10 -> Val 200 (INCONSISTENCY!)
///
/// Expected Behavior:
/// Verification FAILS, the RAM chiplet
/// constraints catch the inconsistency.
#[test]
fn exploit_ram_consistency_bypass() {
    let num_vars = 3; // 8 rows
    let num_rows = 1 << num_vars;
    let seed = [0xAAu8; 32];

    // Define Malicious Events (Impossible sequence)
    let events = [
        MemoryEvent::write(0x10, 0, 100),
        MemoryEvent::read(0x10, 1, 200),
    ];

    // A. CPU Trace (Requests the malicious values)
    let mut cpu_cols: Vec<Vec<F>> = (0..CpuMemColumns::NUM_COLUMNS)
        .map(|_| vec![F::ZERO; num_rows])
        .collect();

    for (i, event) in events.iter().enumerate() {
        let addr_b = event.addr_bytes();
        let val_b = event.val_bytes();

        // Addr
        cpu_cols[CpuMemColumns::ADDR_B0][i] = F::from(addr_b[0]);
        cpu_cols[CpuMemColumns::ADDR_B1][i] = F::from(addr_b[1]);
        cpu_cols[CpuMemColumns::ADDR_B2][i] = F::from(addr_b[2]);
        cpu_cols[CpuMemColumns::ADDR_B3][i] = F::from(addr_b[3]);

        // Val
        cpu_cols[CpuMemColumns::VAL_B0][i] = F::from(val_b[0]);
        cpu_cols[CpuMemColumns::VAL_B1][i] = F::from(val_b[1]);
        cpu_cols[CpuMemColumns::VAL_B2][i] = F::from(val_b[2]);
        cpu_cols[CpuMemColumns::VAL_B3][i] = F::from(val_b[3]);

        // Flags
        cpu_cols[CpuMemColumns::IS_WRITE][i] = if event.is_write { F::ONE } else { F::ZERO };
        cpu_cols[CpuMemColumns::SELECTOR][i] = F::ONE;
    }

    let mut cpu_trace = ColumnTrace::new(num_vars).unwrap();
    for (i, col) in cpu_cols.into_iter().enumerate() {
        let t = if i == CpuMemColumns::IS_WRITE || i == CpuMemColumns::SELECTOR {
            ColumnType::Bit
        } else {
            ColumnType::B32
        };

        cpu_trace
            .add_column(TraceColumn::from_data(col, t))
            .unwrap();
    }

    // B. RAM Trace (Malicious, records inconsistent values)
    let mut ram_cols: Vec<Vec<F>> = (0..RamColumns::NUM_COLUMNS)
        .map(|_| vec![F::ZERO; num_rows])
        .collect();

    for (i, event) in events.iter().enumerate() {
        let addr_b = event.addr_bytes();
        let clk_b = event.clk_bytes();
        let val_b = event.val_bytes();

        // Addr
        ram_cols[RamColumns::ADDR_B0][i] = F::from(addr_b[0]);
        ram_cols[RamColumns::ADDR_B1][i] = F::from(addr_b[1]);
        ram_cols[RamColumns::ADDR_B2][i] = F::from(addr_b[2]);
        ram_cols[RamColumns::ADDR_B3][i] = F::from(addr_b[3]);

        // Clk
        ram_cols[RamColumns::CLK_B0][i] = F::from(clk_b[0]);
        ram_cols[RamColumns::CLK_B1][i] = F::from(clk_b[1]);
        ram_cols[RamColumns::CLK_B2][i] = F::from(clk_b[2]);
        ram_cols[RamColumns::CLK_B3][i] = F::from(clk_b[3]);

        // Val
        ram_cols[RamColumns::VAL_B0][i] = F::from(val_b[0]);
        ram_cols[RamColumns::VAL_B1][i] = F::from(val_b[1]);
        ram_cols[RamColumns::VAL_B2][i] = F::from(val_b[2]);
        ram_cols[RamColumns::VAL_B3][i] = F::from(val_b[3]);

        // Flags
        ram_cols[RamColumns::IS_WRITE][i] = if event.is_write { F::ONE } else { F::ZERO };
        ram_cols[RamColumns::SELECTOR][i] = F::ONE;
    }

    // Build RAM trace using virtual layout types
    let virtual_layout = RamColumns::build_layout();
    let mut ram_trace = ColumnTrace::new(num_vars).unwrap();

    for (i, col) in ram_cols.into_iter().enumerate() {
        ram_trace
            .add_column(TraceColumn::from_data(col, virtual_layout[i]))
            .unwrap();
    }

    // 3. Prove & Verify (RAM as inline, no
    // chiplet path, uses virtual layout directly).
    // This exploit test uses the old inline pattern
    // to inject malicious data. The RamTestAir for
    // this test merges RAM into main trace.
    let air = RamExploitAir { num_rows };

    let mut combined = cpu_trace;
    for col in ram_trace.into_columns() {
        combined.add_column(col).unwrap();
    }

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(combined);

    let config = Config {
        num_queries: 4,
        min_security_bits: 0,
        sumcheck_blinding_factor: 0, // No ZK to simplify debugging
        ..Config::default()
    };

    let proof = prove(
        b"ExploitTest",
        &air,
        &instance,
        &witness,
        &config,
        seed,
        None,
    )
    .expect("Prover should succeed (it simply proves the trace it was given)");

    let mut verifier_transcript = Transcript::<H>::new(b"ExploitTest");
    let result =
        HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut verifier_transcript, &config);

    // SECURITY CHECK:
    // Verification must FAIL, RAM
    // constraints catch the inconsistency.
    assert!(
        result.is_err() || !result.unwrap(),
        "CRITICAL: Verifier accepted a trace with inconsistent memory reads"
    );
}

// =================================================================
// Q_LAST ANCHOR EXPLOIT
// =================================================================

// Mirror of RamChiplet::build_physical_layout indices.
const PHY_PACK_SORT: usize = 0;
const PHY_PACK_VAL: usize = 1;
const PHY_ADDR_B0: usize = 2;
const PHY_CLK_B0: usize = 6;
const PHY_VAL_B0: usize = 10;
const PHY_VAL_PACKED: usize = 14;
const PHY_AUX_INV: usize = 15;
const PHY_IS_WRITE: usize = 16;
const PHY_SELECTOR: usize = 17;
const PHY_Q_STEP: usize = 18;
const PHY_Q_FIRST: usize = 19;
const PHY_Q_LAST: usize = 20;

fn malicious_uninitialised_read_ram_trace(
    addr: u32,
    clk_real: u32,
    val: u32,
    num_rows: usize,
) -> ColumnTrace {
    assert!(num_rows.is_power_of_two() && num_rows >= 4);
    assert!(clk_real >= 1, "real clk must exceed padding clk = 0");

    let num_vars = num_rows.trailing_zeros() as usize;
    let mut tb = TraceBuilder::new(&RamChiplet::build_physical_layout(), num_vars).unwrap();

    let addr_bytes = addr.to_le_bytes();
    let val_bytes = val.to_le_bytes();

    for i in 0..num_rows {
        let is_real = i == 0;

        let row_clk: u32 = if is_real { clk_real } else { 0 };
        let row_clk_bytes = row_clk.to_le_bytes();
        let selector = if is_real { Bit::ONE } else { Bit::ZERO };

        for j in 0..4 {
            tb.set_b32(PHY_ADDR_B0 + j, i, Block32::from(addr_bytes[j] as u32))
                .unwrap();
            tb.set_b32(PHY_CLK_B0 + j, i, Block32::from(row_clk_bytes[j] as u32))
                .unwrap();
            tb.set_b32(PHY_VAL_B0 + j, i, Block32::from(val_bytes[j] as u32))
                .unwrap();
        }

        tb.set_b32(PHY_VAL_PACKED, i, Block32::from(val)).unwrap();
        tb.set_b32(PHY_PACK_VAL, i, Block32::from(val)).unwrap();

        tb.set_bit(PHY_IS_WRITE, i, Bit::ZERO).unwrap();
        tb.set_bit(PHY_SELECTOR, i, selector).unwrap();
        tb.set_bit(PHY_Q_STEP, i, Bit::ONE).unwrap();
        tb.set_bit(PHY_Q_FIRST, i, Bit::ZERO).unwrap();
        tb.set_bit(PHY_Q_LAST, i, Bit::ZERO).unwrap();

        tb.set_b128(PHY_AUX_INV, i, Block128::ZERO).unwrap();

        // pack_sort:
        // DIFF_BYTE_IDX[7]=1 (CLK_B0 slot),
        // b_decomp = curr CLK_B0.
        let (a_byte, b_byte): (u8, u8) = if i == 0 {
            (0, (clk_real & 0xFF) as u8)
        } else if i == num_rows - 1 {
            ((clk_real & 0xFF) as u8, 0)
        } else {
            (0, 0)
        };

        let bit_idx = if a_byte != b_byte {
            let xor = a_byte ^ b_byte;
            (0..8).rev().find(|&k| (xor >> k) & 1 == 1).unwrap_or(0)
        } else {
            0
        };

        let pack_sort = (1u32 << 7)
            | ((1u32 << bit_idx) << 8)
            | ((a_byte as u32) << 16)
            | ((b_byte as u32) << 24);

        tb.set_b32(PHY_PACK_SORT, i, Block32::from(pack_sort))
            .unwrap();
    }

    tb.build()
}

fn cpu_trace_one_read(addr: u32, val: u32, read_row: usize, num_rows: usize) -> ColumnTrace {
    let num_vars = num_rows.trailing_zeros() as usize;
    let mut tb = TraceBuilder::new(&CpuMemColumns::build_layout(), num_vars).unwrap();

    let addr_bytes = addr.to_le_bytes();
    let val_bytes = val.to_le_bytes();

    for j in 0..4 {
        tb.set_b32(
            CpuMemColumns::ADDR_B0 + j,
            read_row,
            Block32::from(addr_bytes[j] as u32),
        )
        .unwrap();
        tb.set_b32(
            CpuMemColumns::VAL_B0 + j,
            read_row,
            Block32::from(val_bytes[j] as u32),
        )
        .unwrap();
    }

    tb.set_bit(CpuMemColumns::IS_WRITE, read_row, Bit::ZERO)
        .unwrap();
    tb.set_bit(CpuMemColumns::SELECTOR, read_row, Bit::ONE)
        .unwrap();

    tb.build()
}

#[test]
fn exploit_ram_uninitialised_read_via_q_last_chain() {
    let num_vars = 3;
    let num_rows = 1 << num_vars;
    let seed = [0xAAu8; 32];

    let addr: u32 = 0x42;
    let clk_real: u32 = 1;
    let val: u32 = 0xDEAD_BEEF;

    let air = RamTestAir { num_rows };
    let cpu_trace = cpu_trace_one_read(addr, val, clk_real as usize, num_rows);
    let ram_trace = malicious_uninitialised_read_ram_trace(addr, clk_real, val, num_rows);

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![ram_trace]);

    let config = Config {
        num_queries: 4,
        min_security_bits: 0,
        sumcheck_blinding_factor: 0,
        ..Config::default()
    };

    let proof = prove(
        b"QLastExploit",
        &air,
        &instance,
        &witness,
        &config,
        seed,
        None,
    )
    .expect("Prover accepts the malicious trace (all AIR identities pass under q_last ≡ 0)");

    let mut verifier_transcript = Transcript::<H>::new(b"QLastExploit");
    let result =
        HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut verifier_transcript, &config);

    assert!(
        result.is_err() || !result.unwrap(),
        "CRITICAL: verifier accepted a RAM trace with q_last ≡ 0, exfiltrating \
         val=0x{val:08X} from uninitialised addr=0x{addr:X}"
    );
}

#[test]
fn scribble_ram_flip_selector_caught() {
    let num_vars = 3;
    let num_rows = 1 << num_vars;

    let events = vec![
        MemoryEvent::write(0x1000, 0, 42),
        MemoryEvent::write(0x2000, 1, 99),
        MemoryEvent::read(0x1000, 2, 42),
        MemoryEvent::read(0x2000, 3, 99),
    ];

    let air = RamTestAir { num_rows };
    let cpu_trace = generate_cpu_trace(&events, num_rows);
    let ram_trace = generate_ram_trace(&events, num_rows).unwrap();

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![ram_trace]);

    assert_all_caught_all_targets(
        &air,
        &instance,
        &witness,
        ScribbleConfig::default()
            .mutations([MutationKind::FlipSelector])
            .cases(64),
    );
}
