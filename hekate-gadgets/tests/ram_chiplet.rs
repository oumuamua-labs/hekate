// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::config::Config;
use hekate_core::trace::{ColumnTrace, TraceBuilder};
use hekate_crypto::DefaultHasher;
use hekate_crypto::transcript::Transcript;
use hekate_gadgets::{CpuMemColumns, MemoryEvent, RamChiplet, RamColumns, generate_ram_trace};
use hekate_math::{Bit, Block32, Block128, Block256, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::digest::program_id;
use hekate_program::{FixedShape, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_scribble::{MutationKind, ScribbleConfig, assert_all_caught_all_targets};
use hekate_sdk::preflight::TableId;
use hekate_testkit::cell::{set_b32, swap_rows};
use hekate_testkit::fault::{Check, Fault};
use hekate_testkit::verdict::assert_rejected;
use hekate_verifier::HekateVerifier;

type F = Block256;
type H = DefaultHasher;

// =================================================================
// RAM Program:
// CPU-only main trace, RAM as independent chiplet.
// =================================================================

fn ram_test_air(num_rows: usize, first_event: usize, num_events: usize) -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("RamTest", num_rows).unwrap();
    let cpu = cx.schema(&CpuMemColumns::build_layout());

    cx.fix(
        cpu.at(CpuMemColumns::SELECTOR),
        FixedShape::Cadence {
            stride: 1,
            count: num_events,
            origin: first_event,
            values: vec![F::ONE],
        },
    );

    cx.bus(RamChiplet::BUS_ID, RamChiplet::cpu_linking_spec());

    let cs = cx.cs();

    cs.assert_boolean(cs.col(CpuMemColumns::IS_WRITE));

    let one = cs.one();
    let s = cs.col(CpuMemColumns::SELECTOR);

    cs.assert_zero_when(one - s, cs.col(CpuMemColumns::IS_WRITE));

    cx.attach(ChipletDef::from_air(&RamChiplet::new(num_rows, num_events)).unwrap());

    cx.compile().unwrap()
}

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
    let air = ram_test_air(num_rows, 0, events.len());

    let cpu_trace = generate_cpu_trace(&events, num_rows);
    let ram_trace = generate_ram_trace(&events, num_rows).unwrap();

    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![ram_trace]);
    let instance = ProgramInstance::new(num_rows, vec![]);

    let config = Config {
        num_queries: 2,
        fold_arity_log: 1,
        early_stop_vars: 1,
        min_security_bits: 0,
        zero_knowledge: true,
        ..Config::default()
    };

    // 2. Prove
    println!("-> Proving...");

    let proof =
        prove(b"RAM_E2E", &air, &instance, &witness, &config, seed, None).expect("Proving failed");

    // 3. Verify
    println!("-> Verifying...");

    let mut verifier_transcript = Transcript::<H>::new(b"RAM_E2E");
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
/// We construct a malicious trace that violates Memory Consistency.
/// 1. WRITE: Addr 0x10 -> Val 100
/// 2. READ:  Addr 0x10 -> Val 200 (INCONSISTENCY!)
///
/// Expected Behavior:
/// Verification FAILS, the RAM chiplet
/// constraints catch the inconsistency.
#[test]
fn exploit_ram_consistency_bypass() {
    let num_rows = 8;

    let written = [
        MemoryEvent::write(0x10, 0, 100),
        MemoryEvent::read(0x10, 1, 100),
    ];

    let forged_events = [
        MemoryEvent::write(0x10, 0, 100),
        MemoryEvent::read(0x10, 1, 200),
    ];

    let air = ram_test_air(num_rows, 0, written.len());
    let instance = ProgramInstance::new(num_rows, vec![]);

    let honest = ProgramWitness::new(generate_cpu_trace(&written, num_rows))
        .with_chiplets(vec![generate_ram_trace(&written, num_rows).unwrap()]);

    let mut ram_trace = generate_ram_trace(&written, num_rows).unwrap();
    write_ram_value(&mut ram_trace, 1, 200);

    let forged = ProgramWitness::new(generate_cpu_trace(&forged_events, num_rows))
        .with_chiplets(vec![ram_trace]);

    let fault = Fault {
        constraint_rows: &[(TableId::Chiplet(0), 0, Some("zero_when"))],
        ..Fault::NONE
    };

    assert_rejected(
        &air,
        (&instance, &honest),
        (&instance, &forged),
        fault,
        &config(),
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

fn write_ram_value(trace: &mut ColumnTrace, row: usize, val: u32) {
    for (j, byte) in val.to_le_bytes().into_iter().enumerate() {
        set_b32(trace, PHY_VAL_B0 + j, row, byte as u32);
    }

    set_b32(trace, PHY_VAL_PACKED, row, val);
    set_b32(trace, PHY_PACK_VAL, row, val);
}

fn config() -> Config {
    Config {
        num_queries: 2,
        fold_arity_log: 1,
        early_stop_vars: 1,
        min_security_bits: 0,
        ..Config::default()
    }
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

    let addr: u32 = 0x42;
    let clk_real: u32 = 1;
    let val: u32 = 0xDEAD_BEEF;
    let read_row = clk_real as usize;

    let air = ram_test_air(num_rows, read_row, 1);
    let instance = ProgramInstance::new(num_rows, vec![]);

    let fresh_read = [MemoryEvent::read(addr, clk_real, 0)];
    let honest = ProgramWitness::new(cpu_trace_one_read(addr, 0, read_row, num_rows))
        .with_chiplets(vec![generate_ram_trace(&fresh_read, num_rows).unwrap()]);

    let forged = ProgramWitness::new(cpu_trace_one_read(addr, val, read_row, num_rows))
        .with_chiplets(vec![malicious_uninitialised_read_ram_trace(
            addr, clk_real, val, num_rows,
        )]);

    let chiplet = TableId::Chiplet(0);
    let last = num_rows - 1;

    let fault = Fault {
        constraint_rows: &[
            (chiplet, 0, None),
            (chiplet, 0, None),
            (chiplet, 0, None),
            (chiplet, 0, None),
            (chiplet, last, None),
        ],
        pin_cells: &[
            (chiplet, RamColumns::Q_FIRST, 0),
            (chiplet, RamColumns::Q_STEP, last),
        ],
        first_check: Some(Check::Zerocheck),
        ..Fault::NONE
    };

    assert_rejected(
        &air,
        (&instance, &honest),
        (&instance, &forged),
        fault,
        &config(),
    );
}

#[test]
fn exploit_ram_clock_out_of_order() {
    let num_vars = 3;
    let num_rows = 1 << num_vars;

    let events = vec![
        MemoryEvent::write(0x1000, 0, 42),
        MemoryEvent::write(0x1000, 1, 7),
    ];

    let air = ram_test_air(num_rows, 0, events.len());
    let cpu_trace = generate_cpu_trace(&events, num_rows);

    let honest = ProgramWitness::new(cpu_trace.clone())
        .with_chiplets(vec![generate_ram_trace(&events, num_rows).unwrap()]);

    let mut ram_trace = generate_ram_trace(&events, num_rows).unwrap();

    let traded = (0..4)
        .flat_map(|j| [PHY_CLK_B0 + j, PHY_VAL_B0 + j])
        .chain([PHY_VAL_PACKED, PHY_PACK_VAL]);

    for col in traded {
        swap_rows(&mut ram_trace, col, 0, 1);
    }

    // pack_sort for the descending pair:
    // DIFF_BYTE_IDX[7]=1 (CLK_B0 slot), DIFF_BIT_IDX[0]=1,
    // a_decomp = next CLK_B0 = 0, b_decomp = curr CLK_B0 = 1.
    let pack_sort = (1u32 << 7) | (1 << 8) | (1 << 24);
    set_b32(&mut ram_trace, PHY_PACK_SORT, 0, pack_sort);

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![ram_trace]);

    let fault = Fault {
        constraint_rows: &[(TableId::Chiplet(0), 0, Some("zero_when"))],
        ..Fault::NONE
    };

    assert_rejected(
        &air,
        (&instance, &honest),
        (&instance, &witness),
        fault,
        &config(),
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

    let air = ram_test_air(num_rows, 0, events.len());
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
