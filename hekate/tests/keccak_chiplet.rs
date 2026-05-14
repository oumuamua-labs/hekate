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

use hekate::crypto::transcript::Transcript;
use hekate::crypto::DefaultHasher;
use hekate::math::{Bit, Block128, Block64, Flat, TowerField};
use hekate_core::config::Config;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder, TraceColumn};
use hekate_gadgets::{
    generate_keccak_trace, CpuKeccakColumns, CpuKeccakUnit, KeccakChiplet, KeccakWitness,
};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::constraint::ConstraintAst;
use hekate_program::permutation::PermutationCheckSpec;
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_sdk::preflight;
use hekate_verifier::HekateVerifier;
use rand::{rngs::OsRng, TryRngCore};
use zk_scribble::{assert_all_caught_all_targets, MutationKind, ScribbleConfig};

type F = Block128;
type H = DefaultHasher;

const CPU_ROWS: usize = 32;
const KECCAK_ROWS: usize = 32;

const PHYS_LANES: usize = 0;
const PHYS_RC: usize = 25;
const PHYS_REQUEST_IDX: usize = 26;
const PHYS_S_ROUND: usize = 27;
const PHYS_S_IN_OUT: usize = 28;

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

#[derive(Clone)]
struct KeccakTestProgram;

impl Air<F> for KeccakTestProgram {
    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(CpuKeccakColumns::build_layout)
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![(KeccakChiplet::BUS_ID.into(), CpuKeccakUnit::linking_spec())]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(CpuKeccakColumns::SELECTOR));

        cs.build()
    }
}

impl Program<F> for KeccakTestProgram {
    fn num_public_inputs(&self) -> usize {
        0
    }

    fn chiplet_defs(&self) -> hekate_core::errors::Result<Vec<ChipletDef<F>>> {
        Ok(vec![ChipletDef::from_air(&KeccakChiplet::new(
            KECCAK_ROWS,
        ))?])
    }
}

// =================================================================
// Trace Builders
// =================================================================

fn build_cpu_trace(input: &[u64; 25], output: &[u64; 25]) -> ColumnTrace {
    let num_vars = CPU_ROWS.trailing_zeros() as usize;
    let mut tb = TraceBuilder::new(&CpuKeccakColumns::build_layout(), num_vars).unwrap();

    for i in 0..25 {
        tb.set_b64(CpuKeccakColumns::LANES + i, 0, Block64(input[i]))
            .unwrap();
        tb.set_b64(CpuKeccakColumns::LANES + i, 1, Block64(output[i]))
            .unwrap();
    }

    tb.set_bit(CpuKeccakColumns::SELECTOR, 0, Bit::ONE).unwrap();
    tb.set_bit(CpuKeccakColumns::SELECTOR, 1, Bit::ONE).unwrap();

    tb.build()
}

fn build_chiplet_trace(input: &[u64; 25]) -> ColumnTrace {
    let input_block: [Block64; 25] = core::array::from_fn(|i| Block64(input[i]));
    generate_keccak_trace(&[input_block], None, KECCAK_ROWS).unwrap()
}

// =================================================================
// Prove + Verify
// =================================================================

fn prove_and_verify(cpu_trace: ColumnTrace, chiplet_trace: ColumnTrace) -> Result<bool, String> {
    let air = KeccakTestProgram;
    let instance = ProgramInstance::new(CPU_ROWS, vec![]);
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

    let mut config = Config {
        sumcheck_blinding_factor: 2,
        ..Config::default()
    };

    OsRng.try_fill_bytes(&mut config.matrix_seed).unwrap();

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
    HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut vt, &config)
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

    let mut chiplet_trace = build_chiplet_trace(&input);
    let mut cpu_trace = build_cpu_trace(&input, &output);

    tamper(&mut chiplet_trace, &mut cpu_trace);

    let air = KeccakTestProgram;
    let instance = ProgramInstance::new(CPU_ROWS, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![chiplet_trace]);

    let mut config = Config {
        sumcheck_blinding_factor: 2,
        ..Config::default()
    };

    OsRng.try_fill_bytes(&mut config.matrix_seed).unwrap();

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
            let result = HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut vt, &config);

            result.is_err() || !result.unwrap()
        }
    }
}

// =================================================================
// Helpers
// =================================================================

fn flip_b64(trace: &mut ColumnTrace, col: usize, row: usize, mask: u64) {
    match &mut trace.columns[col] {
        TraceColumn::B64(data) => {
            let original = data[row];
            data[row] = Flat::from_raw(Block64(original.to_tower().0 ^ mask));
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

    let cpu_trace = build_cpu_trace(&input, &output);
    let chiplet_trace = build_chiplet_trace(&input);

    match prove_and_verify(cpu_trace, chiplet_trace) {
        Ok(true) => {}
        Ok(false) => panic!("verifier rejected honest proof"),
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
        flip_b64(keccak, PHYS_LANES, 5, 0x01);
    });

    assert!(
        detected,
        "lane tamper on round row must be caught by Chi/Iota constraint"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_rc_tamper() {
    let detected = run_tampered_keccak(|keccak, _| {
        flip_b64(keccak, PHYS_RC, 3, 0x01);
    });

    assert!(
        detected,
        "RC tamper must be caught by Iota constraint on lane (0,0)"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_lane_swap() {
    let detected = run_tampered_keccak(|keccak, _| {
        let (col0, col1) = (PHYS_LANES, PHYS_LANES + 1);

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
        flip_b64(keccak, PHYS_LANES + 3, 10, 0xFF);
        flip_b64(keccak, PHYS_LANES + 7, 15, 0xFF);
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
        flip_b64(keccak, PHYS_LANES, 24, 0x01);
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
        flip_b64(cpu, CpuKeccakColumns::LANES + 12, 1, 0x01);
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
        flip_b64(keccak, PHYS_LANES, 24, 0x01);
        flip_b64(cpu, CpuKeccakColumns::LANES, 1, 0x01);
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
        set_bit_val(keccak, PHYS_S_IN_OUT, 5, Bit::ONE);
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
        set_bit_val(keccak, PHYS_S_IN_OUT, 24, Bit::ZERO);
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
        set_bit_val(keccak, PHYS_S_ROUND, 25, Bit::ONE);
        match &mut keccak.columns[PHYS_LANES] {
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
        set_bit_val(keccak, PHYS_S_ROUND, 23, Bit::ZERO);
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
        set_bit_val(keccak, PHYS_S_ROUND, 0, Bit::ZERO);
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

    let cpu_trace = build_cpu_trace(&input, &output);
    let chiplet_trace = build_chiplet_trace(&input);

    let air = KeccakTestProgram;
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
