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

use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate_crypto::DefaultHasher;
use hekate_crypto::transcript::Transcript;
use hekate_gadgets::{
    ArithmeticOpcode, CpuArithColumns, CpuIntArithmeticUnit, CpuKeccakColumns, CpuKeccakUnit,
    CpuMemColumns, CpuMemoryUnit, IntArithmeticChiplet, IntArithmeticLayout, IntArithmeticOp,
    KeccakChiplet, KeccakWitness, MemoryEvent, RamChiplet, generate_arithmetic_trace,
    generate_keccak_trace, generate_ram_trace,
};
use hekate_math::{Bit, Block32, Block64, Block128, Flat, HardwareField, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::constraint::{BoundaryConstraint, ConstraintAst};
use hekate_program::permutation::PermutationCheckSpec;
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;
use rand::{TryRngCore, rngs::OsRng};
// =================================================================
// 0. SETUP
// =================================================================

type F = Block128;
type H = DefaultHasher;

// =================================================================
// 1. COLUMN LAYOUT — MAIN TRACE (CPU ONLY + RESPONSE)
// =================================================================
//
// ALL chiplets (RAM, Arithmetic, Keccak) have independent traces.
// Main trace = CPU columns + Response.
//
// Section              | Virt Start | Virt Count
// ---------------------|------------|----------
// CPU Memory Unit      | 0          | 10
// CPU ALU Unit         | 10         | 5
// CPU Keccak Unit      | 15         | 26
// Response             | 41         | 2

const CPU_MEM: usize = 0;
const CPU_ALU: usize = CpuMemColumns::NUM_COLUMNS;
const CPU_KECCAK: usize = CPU_ALU + CpuArithColumns::NUM_COLUMNS;
const RESP_LO: usize = CPU_KECCAK + CpuKeccakColumns::NUM_COLUMNS;
const RESP_HI: usize = RESP_LO + 1;
const NUM_CPU_COLS: usize = RESP_HI + 1; // 43

// =================================================================
// 2. ROLLUP PROGRAM DEFINITION
// =================================================================

#[derive(Clone)]
struct RollupProgram {
    ram_num_rows: usize,
    arith_num_rows: usize,
    keccak_num_rows: usize,
    genesis_output_row: usize,
    final_output_row: usize,
}

impl Air<F> for RollupProgram {
    fn name(&self) -> String {
        "Rollup".to_string()
    }

    fn num_columns(&self) -> usize {
        NUM_CPU_COLS
    }

    fn boundary_constraints(&self) -> Vec<BoundaryConstraint<F>> {
        vec![
            BoundaryConstraint::with_public_input(RESP_LO, self.genesis_output_row, 0),
            BoundaryConstraint::with_public_input(RESP_HI, self.genesis_output_row, 1),
            BoundaryConstraint::with_public_input(RESP_LO, self.final_output_row, 2),
            BoundaryConstraint::with_public_input(RESP_HI, self.final_output_row, 3),
        ]
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(cpu_layout)
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        // CPU-side bus endpoints only.
        // bus_ids match the chiplet supply-side IDs.

        let mut cpu_mem = CpuMemoryUnit::linking_spec();
        cpu_mem.shift_column_indices(CPU_MEM);

        let mut cpu_alu = CpuIntArithmeticUnit::linking_spec();
        cpu_alu.shift_column_indices(CPU_ALU);

        let mut cpu_keccak = CpuKeccakUnit::linking_spec();
        cpu_keccak.shift_column_indices(CPU_KECCAK);

        vec![
            (RamChiplet::BUS_ID.into(), cpu_mem),
            (IntArithmeticChiplet::BUS_ID.into(), cpu_alu),
            (KeccakChiplet::BUS_ID.into(), cpu_keccak),
        ]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        // CPU selector booleanity,
        // the only main-trace constraints.
        // All chiplet constraints live
        // on their independent traces.
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(CPU_MEM + CpuMemColumns::SELECTOR));
        cs.assert_boolean(cs.col(CPU_ALU + CpuArithColumns::SELECTOR));
        cs.assert_boolean(cs.col(CPU_KECCAK + CpuKeccakColumns::SELECTOR));

        cs.build()
    }
}

impl Program<F> for RollupProgram {
    fn num_public_inputs(&self) -> usize {
        4 // genesis_lo, genesis_hi, final_lo, final_hi
    }

    fn chiplet_defs(&self) -> errors::Result<Vec<ChipletDef<F>>> {
        let ram = RamChiplet::new(self.ram_num_rows);
        let arith = IntArithmeticChiplet::new(32, self.arith_num_rows)
            .expect("IntArithmeticChiplet::new(32, arith_num_rows)");
        let keccak = KeccakChiplet::new(self.keccak_num_rows);

        Ok(vec![
            ChipletDef::from_air(&ram)?,
            ChipletDef::from_air(&arith)?,
            ChipletDef::from_air(&keccak)?,
        ])
    }
}

// =================================================================
// 3. MERKLE TREE BUILDER
// =================================================================

fn compute_state_tree(balances: &[u32]) -> (Vec<[Block64; 25]>, [u128; 2]) {
    let mut keccak_calls = Vec::new();

    let mut padded_len = 4;
    while padded_len < balances.len() {
        padded_len *= 4;
    }

    let mut current_level = vec![[0u64; 4]; padded_len];
    for i in 0..balances.len() {
        current_level[i][0] = balances[i] as u64;
    }

    while current_level.len() > 1 {
        let mut next_level = Vec::new();
        for chunk in current_level.chunks(4) {
            let mut state = [0u64; 25];
            for (i, child) in chunk.iter().enumerate() {
                state[i * 4] = child[0];
                state[i * 4 + 1] = child[1];
                state[i * 4 + 2] = child[2];
                state[i * 4 + 3] = child[3];
            }

            let mut block = [Block64::ZERO; 25];
            for i in 0..25 {
                block[i] = Block64::from(state[i]);
            }

            keccak_calls.push(block);

            // Compute hash natively
            let mut hw_state = [0u64; 25];
            for i in 0..25 {
                hw_state[i] = block[i].to_hardware().into_raw().0;
            }

            keccak::Keccak::new().with_f1600(|f| f(&mut hw_state));

            next_level.push([hw_state[0], hw_state[1], hw_state[2], hw_state[3]]);
        }

        current_level = next_level;
    }

    let root = current_level[0];
    let val1 = (root[1] as u128) << 64 | (root[0] as u128);
    let val2 = (root[3] as u128) << 64 | (root[2] as u128);

    (keccak_calls, [val1, val2])
}

// =================================================================
// 4. MAIN TRACE BUILDER
// =================================================================

fn cpu_layout() -> Vec<ColumnType> {
    let mut cols = Vec::with_capacity(NUM_CPU_COLS);
    cols.extend(CpuMemColumns::build_layout());
    cols.extend(CpuArithColumns::build_layout());
    cols.extend(CpuKeccakColumns::build_layout());
    cols.push(ColumnType::B128);
    cols.push(ColumnType::B128);

    cols
}

fn build_main_trace(
    mem_events: &[MemoryEvent],
    alu_ops: &[IntArithmeticOp],
    all_keccak_calls: &[[Block64; 25]],
    main_num_rows: usize,
    main_num_vars: usize,
    genesis_output_row: usize,
    final_output_row: usize,
) -> errors::Result<ColumnTrace> {
    let mut tb = TraceBuilder::new(&cpu_layout(), main_num_vars)?;

    // CPU Memory Unit (10 cols)
    for (i, ev) in mem_events.iter().enumerate() {
        if i >= main_num_rows {
            break;
        }

        let addr_b = ev.addr_bytes();
        let val_b = ev.val_bytes();

        for b in 0..4 {
            tb.set_b32(
                CPU_MEM + CpuMemColumns::ADDR_B0 + b,
                i,
                Block32::from(addr_b[b] as u32),
            )?;
            tb.set_b32(
                CPU_MEM + CpuMemColumns::VAL_B0 + b,
                i,
                Block32::from(val_b[b] as u32),
            )?;
        }

        tb.set_bit(
            CPU_MEM + CpuMemColumns::IS_WRITE,
            i,
            if ev.is_write { Bit::ONE } else { Bit::ZERO },
        )?;
        tb.set_bit(CPU_MEM + CpuMemColumns::SELECTOR, i, Bit::ONE)?;
    }

    // CPU ALU Unit (5 cols)
    for (i, call) in alu_ops.iter().enumerate() {
        if i >= main_num_rows {
            break;
        }

        let IntArithmeticOp::U32 {
            op,
            a,
            b,
            request_idx: _,
        } = *call
        else {
            unreachable!("rollup is u32-only");
        };

        let res = match op {
            ArithmeticOpcode::ADD => a.wrapping_add(b),
            ArithmeticOpcode::SUB => a.wrapping_sub(b),
            ArithmeticOpcode::AND => a & b,
            ArithmeticOpcode::XOR => a ^ b,
            ArithmeticOpcode::NOT => !a,
            ArithmeticOpcode::LT => (a < b) as u32,
        };

        tb.set_b32(CPU_ALU + CpuArithColumns::VAL_A, i, Block32::from(a))?;
        tb.set_b32(CPU_ALU + CpuArithColumns::VAL_B, i, Block32::from(b))?;
        tb.set_b32(CPU_ALU + CpuArithColumns::VAL_RES, i, Block32::from(res))?;
        tb.set_b32(
            CPU_ALU + CpuArithColumns::OPCODE,
            i,
            Block32::from(op as u8 as u32),
        )?;
        tb.set_bit(CPU_ALU + CpuArithColumns::SELECTOR, i, Bit::ONE)?;
    }

    // CPU Keccak Unit (26 cols)
    // Must emit 2 entries per call to match the chiplet's
    // s_in_out=1 at both input (row 0) and output (row 24).
    // Row 2*i = input lanes,
    // Row 2*i+1 = output lanes.
    for (call_idx, call) in all_keccak_calls.iter().enumerate() {
        let input_row = 2 * call_idx;
        let output_row = 2 * call_idx + 1;

        if output_row >= main_num_rows {
            break;
        }

        // Input lanes (matches chiplet row 0 of call block)
        for (lane, &val) in call.iter().enumerate() {
            tb.set_b64(CPU_KECCAK + CpuKeccakColumns::LANES + lane, input_row, val)?;
        }

        tb.set_bit(CPU_KECCAK + CpuKeccakColumns::SELECTOR, input_row, Bit::ONE)?;

        // Output lanes (matches chiplet row 24 of call block).
        // Must use tower-basis keccak_f_round,
        // same as generate_keccak_trace.
        let mut state = [0u64; 25];
        for i in 0..25 {
            state[i] = call[i].0; // tower-basis u64
        }

        for round in 0..24 {
            let rc = KeccakChiplet::ROUND_CONSTANTS[round];
            state = KeccakWitness::keccak_f_round(state, rc);
        }

        for (lane, &val) in state.iter().enumerate() {
            tb.set_b64(
                CPU_KECCAK + CpuKeccakColumns::LANES + lane,
                output_row,
                Block64::from(val),
            )?;
        }

        tb.set_bit(
            CPU_KECCAK + CpuKeccakColumns::SELECTOR,
            output_row,
            Bit::ONE,
        )?;
    }

    // Response columns (B128),
    // boundary constraint targets.
    for (call_idx, &output_row) in [genesis_output_row, final_output_row].iter().enumerate() {
        let call = &all_keccak_calls[call_idx];
        let mut hw_state = [0u64; 25];

        for i in 0..25 {
            hw_state[i] = call[i].to_hardware().into_raw().0;
        }

        keccak::Keccak::new().with_f1600(|f| f(&mut hw_state));

        let val_lo = (hw_state[1] as u128) << 64 | (hw_state[0] as u128);
        let val_hi = (hw_state[3] as u128) << 64 | (hw_state[2] as u128);

        tb.set_b128(
            RESP_LO,
            output_row,
            Flat::from_raw(Block128(val_lo)).to_tower(),
        )?;
        tb.set_b128(
            RESP_HI,
            output_row,
            Flat::from_raw(Block128(val_hi)).to_tower(),
        )?;
    }

    Ok(tb.build())
}

// =================================================================
// 5. MAIN
// =================================================================

fn main() {
    common::init("Rollup");

    // =========================================================
    // PARAMETERS
    // =========================================================
    let num_accounts = 4000;
    let num_txs = 2000;

    // Derive trace heights per table
    let num_mem_events = 2 * num_accounts + 4 * num_txs;
    let num_alu_ops = 2 * num_txs;

    // Keccak:
    // two 4-ary Merkle trees over num_accounts leaves
    let mut merkle_padded = 4usize;
    while merkle_padded < num_accounts {
        merkle_padded *= 4;
    }

    let calls_per_tree = (merkle_padded - 1) / 3;
    let total_keccak_calls = 2 * calls_per_tree;
    let keccak_calls_padded = total_keccak_calls.next_power_of_two();
    let keccak_rows = keccak_calls_padded * 25;

    // Each table gets its own optimal height.
    // Main trace must fit:
    // mem events + 2 rows per keccak call (input + output)
    let cpu_keccak_rows = 2 * keccak_calls_padded;
    let main_min = num_mem_events.max(cpu_keccak_rows);
    let main_num_rows = main_min.next_power_of_two();
    let main_num_vars = main_num_rows.trailing_zeros() as usize;

    // Independent chiplet heights:
    // sized to their actual workload.
    let ram_num_rows = num_mem_events.next_power_of_two();
    let ram_num_vars = ram_num_rows.trailing_zeros() as usize;

    // Arithmetic:
    // 1 row per operation
    let arith_num_rows = num_alu_ops.next_power_of_two();
    let arith_num_vars = arith_num_rows.trailing_zeros() as usize;

    let keccak_num_rows = keccak_rows.next_power_of_two();
    let keccak_num_vars = keccak_num_rows.trailing_zeros() as usize;

    let mut config = Config {
        sumcheck_blinding_factor: 2,
        ..Config::default()
    };

    OsRng.try_fill_bytes(&mut config.matrix_seed).unwrap();

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    println!("  Accounts:       {}", num_accounts);
    println!("  Transactions:   {}", num_txs);
    println!(
        "  Main trace:     2^{} ({} rows)",
        main_num_vars, main_num_rows
    );
    println!(
        "  RAM chiplet:    2^{} ({} rows)",
        ram_num_vars, ram_num_rows
    );
    println!(
        "  Arith chiplet:  2^{} ({} rows)",
        arith_num_vars, arith_num_rows
    );
    println!(
        "  Keccak chiplet: 2^{} ({} rows)  [{} calls]",
        keccak_num_vars, keccak_num_rows, total_keccak_calls
    );

    let (main_trace, ram_trace, arith_trace, keccak_trace, genesis_root, final_root) =
        common::phase("Trace Generation", || {
            // ==================================================
            // A. ROLLUP STATE SIMULATION
            // ==================================================
            let mut balances = vec![0u32; num_accounts];
            balances[0] = 100000;
            balances[1] = 50000;

            // Genesis Merkle Tree
            let (mut genesis_calls, genesis_root) = compute_state_tree(&balances);

            // Memory events + ALU ops
            // from transaction processing.
            let mut mem_events = Vec::new();
            let mut alu_ops = Vec::new();
            let mut clk = 0;

            // Init memory (Genesis)
            for (i, &balance) in balances.iter().take(num_accounts).enumerate() {
                mem_events.push(MemoryEvent::write((i * 4) as u32, clk, balance));
                clk += 1;
            }

            // Process transactions
            for i in 0..num_txs {
                let from = i % num_accounts;
                let to = (i + 1) % num_accounts;
                let amt = 10;
                let fee = 1;

                let from_bal = balances[from];
                mem_events.push(MemoryEvent::read((from * 4) as u32, clk, from_bal));

                clk += 1;

                alu_ops.push(IntArithmeticOp::U32 {
                    op: ArithmeticOpcode::SUB,
                    a: from_bal,
                    b: amt + fee,
                    request_idx: alu_ops.len() as u32,
                });

                let new_from = from_bal.saturating_sub(amt + fee);
                balances[from] = new_from;

                mem_events.push(MemoryEvent::write((from * 4) as u32, clk, new_from));

                clk += 1;

                let to_bal = balances[to];
                mem_events.push(MemoryEvent::read((to * 4) as u32, clk, to_bal));

                clk += 1;

                alu_ops.push(IntArithmeticOp::U32 {
                    op: ArithmeticOpcode::ADD,
                    a: to_bal,
                    b: amt,
                    request_idx: alu_ops.len() as u32,
                });

                let new_to = to_bal + amt;
                balances[to] = new_to;

                mem_events.push(MemoryEvent::write((to * 4) as u32, clk, new_to));

                clk += 1;
            }

            // Read final state
            for (i, &balance) in balances.iter().take(num_accounts).enumerate() {
                mem_events.push(MemoryEvent::read((i * 4) as u32, clk, balance));
                clk += 1;
            }

            // Final Merkle Tree
            let (mut final_calls, final_root) = compute_state_tree(&balances);

            // Assemble keccak calls:
            // root nodes first (for boundary constraints)
            let genesis_root_call = genesis_calls.pop().unwrap();
            let final_root_call = final_calls.pop().unwrap();

            let mut all_keccak_calls = Vec::new();

            all_keccak_calls.push(genesis_root_call);
            all_keccak_calls.push(final_root_call);
            all_keccak_calls.extend(genesis_calls);
            all_keccak_calls.extend(final_calls);

            let keccak_num_calls = all_keccak_calls.len().next_power_of_two();
            while all_keccak_calls.len() < keccak_num_calls {
                all_keccak_calls.push([Block64::ZERO; 25]);
            }

            // Output rows for boundary constraints (main trace).
            // CPU keccak layout:
            // row 2*i = input,
            // row 2*i+1 = output.
            // Genesis root is call 0,
            // final root is call 1.
            let genesis_output_row = 1; // call 0 output
            let final_output_row = 3; // call 1 output

            // ==================================================
            // B. BUILD MAIN TRACE (CPU + RESPONSE)
            // ==================================================
            let main_trace = build_main_trace(
                &mem_events,
                &alu_ops,
                &all_keccak_calls,
                main_num_rows,
                main_num_vars,
                genesis_output_row,
                final_output_row,
            )
            .unwrap();

            // ==================================================
            // C. BUILD ALL INDEPENDENT CHIPLET TRACES
            // ==================================================

            // RAM Chiplet (49 cols)
            let ram_trace = generate_ram_trace(&mem_events, ram_num_rows).unwrap();

            // Arithmetic Chiplet (12 phys cols, 140 virtual)
            let arith_layout = IntArithmeticLayout::compute(32);
            let arith_trace =
                generate_arithmetic_trace(&alu_ops, &arith_layout, arith_num_rows).unwrap();

            // Keccak Chiplet (28 physical cols, 1691 virtual)
            let keccak_trace =
                generate_keccak_trace(&all_keccak_calls, None, keccak_num_rows).unwrap();

            (
                main_trace,
                ram_trace,
                arith_trace,
                keccak_trace,
                genesis_root,
                final_root,
            )
        });

    println!(
        "   Main: {}  |  RAM: {}  |  Arith: {}  |  Keccak: {}",
        main_trace.columns.len(),
        ram_trace.columns.len(),
        arith_trace.columns.len(),
        keccak_trace.columns.len(),
    );

    // ==================================================
    // D. PROVING & VERIFYING
    // ==================================================
    let genesis_output_row = 1;
    let final_output_row = 3;

    let air = RollupProgram {
        ram_num_rows,
        arith_num_rows,
        keccak_num_rows,
        genesis_output_row,
        final_output_row,
    };

    println!("   Constraint count: {}", air.constraint_ast().roots.len());

    let public_inputs = vec![
        Flat::from_raw(Block128(genesis_root[0])).to_tower(),
        Flat::from_raw(Block128(genesis_root[1])).to_tower(),
        Flat::from_raw(Block128(final_root[0])).to_tower(),
        Flat::from_raw(Block128(final_root[1])).to_tower(),
    ];

    let instance = ProgramInstance::new(main_num_rows, public_inputs);
    let witness =
        ProgramWitness::new(main_trace).with_chiplets(vec![ram_trace, arith_trace, keccak_trace]);

    let proof = common::phase("Proving", || {
        prove(
            b"Hekate_Rollup",
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

    let mut verifier_transcript = Transcript::<H>::new(b"Hekate_Rollup");

    let is_valid = common::phase("Verifying", || {
        HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut verifier_transcript, &config)
            .expect("Verifier failed")
    });

    common::result(is_valid);
}
