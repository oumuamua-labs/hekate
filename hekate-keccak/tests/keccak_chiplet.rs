// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::config::Config;
use hekate_core::trace::{ColumnTrace, TraceBuilder, TraceColumn};
use hekate_crypto::DefaultHasher;
use hekate_crypto::transcript::Transcript;
use hekate_keccak::{
    CpuKeccakColumns, KeccakChiplet, KeccakWitness, PhysKeccakColumns, generate_keccak_trace,
};
use hekate_math::{Bit, Block32, Block64, Block128, Flat, HardwareField, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram, Col};
use hekate_program::digest::program_id;
use hekate_program::{ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_scribble::{MutationKind, ScribbleConfig, assert_all_caught_all_targets};
use hekate_sdk::preflight;
use hekate_verifier::HekateVerifier;
use rand::{TryRngCore, rngs::OsRng};

type F = Block128;
type H = DefaultHasher;

const CPU_ROWS: usize = 32;
const KECCAK_ROWS: usize = 32;

fn test_input_state() -> [u64; 25] {
    let mut state = [0u64; 25];
    state[0] = 0xDEADBEEFCAFEBABE;
    state[1] = 0x0123456789ABCDEF;
    state[2] = 0xFEDCBA9876543210;

    state
}

fn compute_keccak_f(input: [u64; 25]) -> [u64; 25] {
    let mut state = input;
    for &rc in &KeccakChiplet::ROUND_CONSTANTS {
        state = KeccakWitness::keccak_f_round(state, rc);
    }

    state
}

// =================================================================
// Test Program
// =================================================================

fn build_program(cpu_rows: usize, keccak_rows: usize, num_blocks: usize) -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("KeccakTest", cpu_rows).unwrap();
    let cpu = cx.schema(&CpuKeccakColumns::build_layout());

    let selector = cpu.at(CpuKeccakColumns::SELECTOR);

    let call_values: Vec<Col> = (0..25)
        .map(|lane| cpu.at(CpuKeccakColumns::LANES + lane))
        .collect();

    cx.call(&KeccakChiplet::service(), &call_values, selector)
        .unwrap();

    cx.fix(
        selector,
        KeccakChiplet::host_selector_shape(KeccakChiplet::BLOCK_ROWS, num_blocks),
    );

    cx.attach(ChipletDef::from_air(&KeccakChiplet::new(keccak_rows, num_blocks)).unwrap());

    cx.compile().unwrap()
}

// =================================================================
// Trace Builders
// =================================================================

fn build_cpu_trace(calls: &[([u64; 25], [u64; 25])], cpu_rows: usize) -> ColumnTrace {
    let num_vars = cpu_rows.trailing_zeros() as usize;
    let mut tb = TraceBuilder::new(&CpuKeccakColumns::build_layout(), num_vars).unwrap();

    for (k, (input, output)) in calls.iter().enumerate() {
        let in_row = 25 * k;
        let out_row = 25 * k + 24;

        for i in 0..25 {
            tb.set_b64(CpuKeccakColumns::LANES + i, in_row, Block64(input[i]))
                .unwrap();
            tb.set_b64(CpuKeccakColumns::LANES + i, out_row, Block64(output[i]))
                .unwrap();
        }

        tb.set_bit(CpuKeccakColumns::SELECTOR, in_row, Bit::ONE)
            .unwrap();
        tb.set_bit(CpuKeccakColumns::SELECTOR, out_row, Bit::ONE)
            .unwrap();
    }

    tb.build()
}

fn build_chiplet_trace(inputs: &[[u64; 25]], keccak_rows: usize) -> ColumnTrace {
    let blocks: Vec<[Block64; 25]> = inputs
        .iter()
        .map(|s| core::array::from_fn(|i| Block64(s[i])))
        .collect();

    generate_keccak_trace(&blocks, keccak_rows).unwrap()
}

// =================================================================
// Prove + Verify
// =================================================================

fn prove_and_verify(
    cpu_trace: ColumnTrace,
    chiplet_trace: ColumnTrace,
    cpu_rows: usize,
    keccak_rows: usize,
    num_blocks: usize,
) -> Result<bool, String> {
    let air = build_program(cpu_rows, keccak_rows, num_blocks);

    let instance = ProgramInstance::new(cpu_rows, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![chiplet_trace]);

    let report = preflight(&air, &instance, &witness).map_err(|e| format!("preflight: {e:?}"))?;

    if !report.is_clean() {
        for v in &report.constraint_violations {
            eprintln!(
                "constraint={} label={:?} row={}",
                v.constraint_idx, v.label, v.row_idx,
            );
        }

        for d in &report.bus_diagnostics {
            for ep in &d.endpoints {
                eprintln!("bus \"{}\": active={}", d.bus_id, ep.active_rows);
            }
        }

        return Err("preflight violations".into());
    }

    let config = Config {
        zero_knowledge: true,
        ..Config::dev()
    };

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    let proof = prove(
        b"Keccak_E2E",
        &air,
        &instance,
        &witness,
        &config,
        blinding_seed,
        None,
    )
    .map_err(|e| format!("prover: {e:?}"))?;

    let mut vt = Transcript::<H>::new(b"Keccak_E2E");
    let pinned_id = program_id(&air).unwrap();

    HekateVerifier::<F, H>::verify(&pinned_id, &air, &instance, &proof, &mut vt, &config)
        .map_err(|e| format!("verifier: {e:?}"))
}

// =================================================================
// Adversarial Harness
// =================================================================

fn run_tampered_keccak<T>(tamper: T) -> bool
where
    T: FnOnce(&mut ColumnTrace, &mut ColumnTrace),
{
    let input = test_input_state();
    let output = compute_keccak_f(input);

    let mut chiplet_trace = build_chiplet_trace(&[input], KECCAK_ROWS);
    let mut cpu_trace = build_cpu_trace(&[(input, output)], CPU_ROWS);

    tamper(&mut chiplet_trace, &mut cpu_trace);

    let air = build_program(CPU_ROWS, KECCAK_ROWS, 1);

    let instance = ProgramInstance::new(CPU_ROWS, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![chiplet_trace]);

    let config = Config {
        zero_knowledge: true,
        ..Config::dev()
    };

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    let proof_result = prove(
        b"Keccak_Adversarial",
        &air,
        &instance,
        &witness,
        &config,
        blinding_seed,
        None,
    );

    match proof_result {
        Err(_) => true,
        Ok(proof) => {
            let mut vt = Transcript::<H>::new(b"Keccak_Adversarial");
            let pinned_id = program_id(&air).unwrap();

            let result = HekateVerifier::<F, H>::verify(
                &pinned_id, &air, &instance, &proof, &mut vt, &config,
            );

            result.is_err() || !result.unwrap()
        }
    }
}

// =================================================================
// Helpers
// =================================================================

fn flip_b32(trace: &mut ColumnTrace, col: usize, row: usize, mask: u32) {
    match &mut trace.columns[col] {
        TraceColumn::B32(data) => {
            let original = data[row].to_tower().0;
            data[row] = Block32(original ^ mask).to_hardware();
        }
        _ => panic!("expected B32 column at {col}"),
    }
}

fn flip_b64(trace: &mut ColumnTrace, col: usize, row: usize, mask: u64) {
    match &mut trace.columns[col] {
        TraceColumn::B64(data) => {
            let original = data[row].to_tower().0;
            data[row] = Block64(original ^ mask).to_hardware();
        }
        _ => panic!("expected B64 column at {col}"),
    }
}

fn set_bit_val(trace: &mut ColumnTrace, col: usize, row: usize, val: Bit) {
    match &mut trace.columns[col] {
        TraceColumn::Bit(data) => data[row] = val,
        _ => panic!("expected Bit column at {col}"),
    }
}

// =================================================================
// Honest Path
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn keccak_e2e() {
    let input = test_input_state();
    let output = compute_keccak_f(input);

    let cpu_trace = build_cpu_trace(&[(input, output)], CPU_ROWS);
    let chiplet_trace = build_chiplet_trace(&[input], KECCAK_ROWS);

    match prove_and_verify(cpu_trace, chiplet_trace, CPU_ROWS, KECCAK_ROWS, 1) {
        Ok(true) => {}
        Ok(false) => panic!("verifier rejected honest proof"),
        Err(e) => panic!("error: {e}"),
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn keccak_e2e_multi_call() {
    const ROWS: usize = 64;

    let in_a = test_input_state();
    let mut in_b = test_input_state();
    in_b[0] ^= u64::MAX;

    let calls = [
        (in_a, compute_keccak_f(in_a)),
        (in_b, compute_keccak_f(in_b)),
    ];
    let inputs: Vec<[u64; 25]> = calls.iter().map(|(i, _)| *i).collect();

    let cpu_trace = build_cpu_trace(&calls, ROWS);
    let chiplet_trace = build_chiplet_trace(&inputs, ROWS);

    match prove_and_verify(cpu_trace, chiplet_trace, ROWS, ROWS, 2) {
        Ok(true) => {}
        Ok(false) => panic!("verifier rejected honest multi-call proof"),
        Err(e) => panic!("error: {e}"),
    }
}

// =================================================================
// Round Constraint Exploits (Chi/Iota)
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_round_state_tamper() {
    let detected = run_tampered_keccak(|keccak, _| {
        flip_b64(keccak, PhysKeccakColumns::P_LANES, 5, 0x01);
    });

    assert!(
        detected,
        "lane tamper on round row must be caught by Chi/Iota constraint"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_round_index_tamper() {
    let detected = run_tampered_keccak(|keccak, _| {
        flip_b32(keccak, PhysKeccakColumns::P_ROUND, 3, 0x01);
    });

    assert!(detected);
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_lane_swap() {
    let detected = run_tampered_keccak(|keccak, _| {
        let (col0, col1) = (PhysKeccakColumns::P_LANES, PhysKeccakColumns::P_LANES + 1);

        let v0 = match &keccak.columns[col0] {
            TraceColumn::B64(d) => d[5],
            _ => panic!("expected B64"),
        };
        let v1 = match &keccak.columns[col1] {
            TraceColumn::B64(d) => d[5],
            _ => panic!("expected B64"),
        };

        if let TraceColumn::B64(d) = &mut keccak.columns[col0] {
            d[5] = v1;
        }
        if let TraceColumn::B64(d) = &mut keccak.columns[col1] {
            d[5] = v0;
        }
    });

    assert!(
        detected,
        "lane swap on round row must be caught by Chi/Iota constraint"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_multi_round_tamper() {
    let detected = run_tampered_keccak(|keccak, _| {
        flip_b64(keccak, PhysKeccakColumns::P_LANES + 3, 10, 0xFF);
        flip_b64(keccak, PhysKeccakColumns::P_LANES + 7, 15, 0xFF);
    });

    assert!(
        detected,
        "multi-round state tamper must be caught by Chi/Iota constraints"
    );
}

// =================================================================
// Bus Integrity Exploits
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_chiplet_output_tamper() {
    let detected = run_tampered_keccak(|keccak, _| {
        flip_b64(keccak, PhysKeccakColumns::P_LANES, 24, 0x01);
    });

    assert!(
        detected,
        "chiplet output lane tamper must be caught by bus + round chain"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_cpu_input_tamper() {
    let detected = run_tampered_keccak(|_, cpu| {
        flip_b64(cpu, CpuKeccakColumns::LANES, 0, 0x01);
    });

    assert!(
        detected,
        "CPU-side input lane tamper must be caught by keccak_link bus"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_keccak_link_duplicate_cpu_request_rejected() {
    let detected = run_tampered_keccak(|_, cpu| {
        for i in 0..25 {
            let value = match &cpu.columns[CpuKeccakColumns::LANES + i] {
                TraceColumn::B64(data) => data[0],
                _ => panic!("expected B64 lane column"),
            };

            match &mut cpu.columns[CpuKeccakColumns::LANES + i] {
                TraceColumn::B64(data) => data[2] = value,
                _ => unreachable!(),
            }
        }

        set_bit_val(cpu, CpuKeccakColumns::SELECTOR, 2, Bit::ONE);
    });

    assert!(
        detected,
        "duplicate CPU input request without chiplet partner must be caught by keccak_link bus"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_cpu_output_tamper() {
    let detected = run_tampered_keccak(|_, cpu| {
        flip_b64(cpu, CpuKeccakColumns::LANES + 12, 24, 0x01);
    });

    assert!(
        detected,
        "CPU-side output lane tamper must be caught by keccak_link bus"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_consistent_output_forgery() {
    let detected = run_tampered_keccak(|keccak, cpu| {
        // Tamper output on BOTH sides so bus passes.
        // Round chain must still catch it:
        // next_bit[24] ≠ chi(state[23]).
        flip_b64(keccak, PhysKeccakColumns::P_LANES, 24, 0x01);
        flip_b64(cpu, CpuKeccakColumns::LANES, 24, 0x01);
    });

    assert!(
        detected,
        "bus-consistent output forgery must be caught by round chain constraint"
    );
}

// =================================================================
// Selector Integrity Exploits
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_s_in_out_injection() {
    let detected = run_tampered_keccak(|keccak, _| {
        set_bit_val(keccak, PhysKeccakColumns::P_S_IN_OUT, 5, Bit::ONE);
    });

    assert!(
        detected,
        "s_in_out=1 on mid-round row must be caught by bus event mismatch"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_s_in_out_deactivation() {
    let detected = run_tampered_keccak(|keccak, _| {
        set_bit_val(keccak, PhysKeccakColumns::P_S_IN_OUT, 24, Bit::ZERO);
    });

    assert!(
        detected,
        "s_in_out=0 on output row must be caught by bus event mismatch"
    );
}

// =================================================================
// Ghost Protocol Exploits
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_ghost_state_injection() {
    let detected = run_tampered_keccak(|keccak, _| {
        // Inject non-zero state + s_round on padding row.
        // Row 26 is all-zero padding → chi(non-zero state)
        // produces non-zero output ≠ zero next_bit.
        set_bit_val(keccak, PhysKeccakColumns::P_S_ROUND, 25, Bit::ONE);
        match &mut keccak.columns[PhysKeccakColumns::P_LANES] {
            TraceColumn::B64(data) => {
                data[25] = Flat::from_raw(Block64(0xDEADBEEF));
            }
            _ => panic!("expected B64"),
        }
    });

    assert!(
        detected,
        "ghost round with non-zero state must be caught by Chi/Iota constraint"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_s_round_deactivation() {
    let detected = run_tampered_keccak(|keccak, _| {
        set_bit_val(keccak, PhysKeccakColumns::P_S_ROUND, 23, Bit::ZERO);
    });

    assert!(
        detected,
        "s_round=0 on last round row must be caught by continuity constraint"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_s_round_input_deactivation() {
    let detected = run_tampered_keccak(|keccak, _| {
        set_bit_val(keccak, PhysKeccakColumns::P_S_ROUND, 0, Bit::ZERO);
    });

    assert!(
        detected,
        "s_round=0 on input row must be caught by output binding (MLE wrap)"
    );
}

#[test]
fn scribble_keccak_flip_selector_caught() {
    let input = test_input_state();
    let output = compute_keccak_f(input);

    let cpu_trace = build_cpu_trace(&[(input, output)], CPU_ROWS);
    let chiplet_trace = build_chiplet_trace(&[input], KECCAK_ROWS);

    let air = build_program(CPU_ROWS, KECCAK_ROWS, 1);

    let instance = ProgramInstance::new(CPU_ROWS, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![chiplet_trace]);

    assert_all_caught_all_targets(
        &air,
        &instance,
        &witness,
        ScribbleConfig::default()
            .mutations([MutationKind::FlipSelector])
            .cases(64),
    );
}
