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

//! End-to-end prove/verify test for the
//! NTT chiplet (isolated).
//!
//! Exercises the NTT butterfly constraint
//! system through the full prover/verifier
//! pipeline with a handful of operations.

use hekate_core::config::Config;
use hekate_core::trace::TraceColumn;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate_crypto::DefaultHasher;
use hekate_crypto::transcript::Transcript;
use hekate_math::Block128;
use hekate_math::{Bit, Block32, Flat, TowerField};
use hekate_pqc::ntt::{
    NttButterfly, NttChiplet, NttFlowCompanion, NttMulOnly, NttOp, generate_ntt_trace,
};
use hekate_pqc::twiddle_rom::{
    TWIDDLE_W_BINDING_BUS_ID, TwiddleEntry, TwiddleRomChiplet, generate_twiddle_rom_trace,
};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::permutation::{PermutationCheckSpec, Source};
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_scribble::{MutationKind, ScribbleConfig, assert_all_caught_all_targets};
use hekate_verifier::HekateVerifier;
use rand::{TryRngCore, rngs::OsRng};

type F = Block128;
type H = DefaultHasher;

const Q: u32 = 3329;

// =================================================================
// CPU-side columns for NTT data bus
// =================================================================

const CPU_A: usize = 0;
const CPU_B: usize = 1;
const CPU_A_OUT: usize = 2;
const CPU_B_OUT: usize = 3;
const CPU_LAYER: usize = 4;
const CPU_BFLY: usize = 5;
const CPU_INST: usize = 6;
const CPU_SEL: usize = 7;
const CPU_NUM_COLS: usize = 8;

fn cpu_layout() -> Vec<ColumnType> {
    vec![
        ColumnType::B32, // a
        ColumnType::B32, // b
        ColumnType::B32, // a_out
        ColumnType::B32, // b_out
        ColumnType::B32, // layer
        ColumnType::B32, // butterfly_idx
        ColumnType::B32, // ntt_instance
        ColumnType::Bit, // selector
    ]
}

fn cpu_linking_spec() -> PermutationCheckSpec {
    PermutationCheckSpec::new(
        vec![
            (Source::Column(CPU_A), b"kappa_ntt_a" as &[u8]),
            (Source::Column(CPU_B), b"kappa_ntt_b" as &[u8]),
            (Source::Column(CPU_A_OUT), b"kappa_ntt_a_out" as &[u8]),
            (Source::Column(CPU_B_OUT), b"kappa_ntt_b_out" as &[u8]),
            (Source::Column(CPU_LAYER), b"kappa_ntt_layer" as &[u8]),
            (Source::Column(CPU_BFLY), b"kappa_ntt_bfly" as &[u8]),
            (Source::Column(CPU_INST), b"kappa_ntt_inst" as &[u8]),
        ],
        Some(CPU_SEL),
    )
    .with_clock_waiver(
        "see hekate-gadgets/src/pqc/ntt.rs: NTT data bus, partner chiplet's \
         data_linking_spec carries the same waiver citing (ntt_instance, layer, \
         butterfly_idx) per-row uniqueness via flow constraints",
    )
}

// =================================================================
// NTT Isolated Program
// =================================================================

#[derive(Clone)]
struct NttIsolatedProgram {
    ntt_rows: usize,
    twiddle_rows: usize,
}

impl Air<F> for NttIsolatedProgram {
    fn num_columns(&self) -> usize {
        CPU_NUM_COLS
    }

    fn column_layout(&self) -> &[ColumnType] {
        let layout = cpu_layout();
        Box::leak(layout.into_boxed_slice())
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![(NttChiplet::DATA_BUS_ID.into(), cpu_linking_spec())]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(CPU_SEL));

        cs.build()
    }
}

impl Program<F> for NttIsolatedProgram {
    fn chiplet_defs(&self) -> hekate_core::errors::Result<Vec<ChipletDef<F>>> {
        let ntt = NttChiplet::new(Q, self.ntt_rows);
        let twiddle = TwiddleRomChiplet::new(Q, self.twiddle_rows);

        let mut ntt_def = ChipletDef::from_air(&ntt)?;

        // Strip boundary specs:
        // no ctrl chiplet in standalone test.
        ntt_def.permutation_checks.retain(|(id, _)| {
            id != NttChiplet::BOUND_IN_BUS_ID && id != NttChiplet::BOUND_OUT_BUS_ID
        });

        let mut twiddle_def = ChipletDef::from_air(&twiddle)?;

        // Strip w_binding spec, no
        // ctrl in standalone test.
        twiddle_def
            .permutation_checks
            .retain(|(id, _)| id != TWIDDLE_W_BINDING_BUS_ID);

        Ok(vec![ntt_def, twiddle_def])
    }
}

// =================================================================
// Trace helpers
// =================================================================

fn flip_b32(trace: &mut ColumnTrace, col: usize, row: usize, mask: u32) {
    match &mut trace.columns[col] {
        TraceColumn::B32(data) => {
            let original = data[row];
            data[row] = Flat::from_raw(Block32(original.to_tower().0 ^ mask));
        }
        _ => panic!("expected B32 column at {col}"),
    }
}

fn set_b32(trace: &mut ColumnTrace, col: usize, row: usize, val: u32) {
    match &mut trace.columns[col] {
        TraceColumn::B32(data) => {
            data[row] = Flat::from_raw(Block32::from(val));
        }
        _ => panic!("expected B32 column at {col}"),
    }
}

fn read_b32(trace: &ColumnTrace, col: usize, row: usize) -> u32 {
    match &trace.columns[col] {
        TraceColumn::B32(data) => data[row].to_tower().0,
        _ => panic!("expected B32 column at {col}"),
    }
}

fn set_bit_val(trace: &mut ColumnTrace, col: usize, row: usize, val: Bit) {
    match &mut trace.columns[col] {
        TraceColumn::Bit(data) => data[row] = val,
        _ => panic!("expected Bit column at {col}"),
    }
}

#[allow(dead_code)]
fn read_bit(trace: &ColumnTrace, col: usize, row: usize) -> Bit {
    match &trace.columns[col] {
        TraceColumn::Bit(data) => data[row],
        _ => panic!("expected Bit column at {col}"),
    }
}

// =================================================================
// CPU trace generation
// =================================================================

fn generate_cpu_trace(ops: &[NttOp], num_rows: usize) -> ColumnTrace {
    let num_vars = num_rows.trailing_zeros() as usize;

    let mut tb = TraceBuilder::new(&cpu_layout(), num_vars).unwrap();

    for (row, op) in ops.iter().enumerate() {
        let (a, b, w, layer, bfly_idx, inst) = match op {
            NttOp::Butterfly(bf) => (
                bf.a,
                bf.b,
                bf.w,
                bf.layer,
                bf.butterfly_idx,
                bf.ntt_instance,
            ),
            NttOp::MulOnly(m) => (0u32, m.b, m.w, m.layer, m.butterfly_idx, m.flow_instance),
            NttOp::FlowCompanion(_) => continue,
        };

        let wb = ((w as u64 * b as u64) % Q as u64) as u32;
        let a_out = (a + wb) % Q;
        let b_out = (a + Q - wb) % Q;

        tb.set_b32(CPU_A, row, Block32::from(a)).unwrap();
        tb.set_b32(CPU_B, row, Block32::from(b)).unwrap();
        tb.set_b32(CPU_A_OUT, row, Block32::from(a_out)).unwrap();
        tb.set_b32(CPU_B_OUT, row, Block32::from(b_out)).unwrap();
        tb.set_b32(CPU_LAYER, row, Block32::from(layer)).unwrap();
        tb.set_b32(CPU_BFLY, row, Block32::from(bfly_idx)).unwrap();
        tb.set_b32(CPU_INST, row, Block32::from(inst)).unwrap();
        tb.set_bit(CPU_SEL, row, Bit::ONE).unwrap();
    }

    tb.build()
}

// =================================================================
// Prove/verify harnesses
// =================================================================

fn prove_and_verify(ops: &[NttOp], label: &str) -> bool {
    let num_ops = ops.len();
    let cpu_rows = (num_ops + 1).next_power_of_two().max(4);
    let ntt_rows = (num_ops + 1).next_power_of_two().max(4);

    let twiddle_entries: Vec<TwiddleEntry> = ops
        .iter()
        .map(|op| match op {
            NttOp::Butterfly(b) => TwiddleEntry {
                layer: b.layer,
                butterfly_idx: b.butterfly_idx,
                w: b.w,
                is_mulonly: false,
                active: true,
                request_idx_tr: 0,
            },
            NttOp::MulOnly(m) => TwiddleEntry {
                layer: m.layer,
                butterfly_idx: m.butterfly_idx,
                w: m.w,
                is_mulonly: m.is_basemul,
                active: true,
                request_idx_tr: 0,
            },
            NttOp::FlowCompanion(_) => TwiddleEntry {
                layer: 0,
                butterfly_idx: 0,
                w: 0,
                is_mulonly: false,
                active: false,
                request_idx_tr: 0,
            },
        })
        .collect();

    let twiddle_rows = (twiddle_entries.len() + 1).next_power_of_two().max(4);

    let air = NttIsolatedProgram {
        ntt_rows,
        twiddle_rows,
    };

    let cpu_trace = generate_cpu_trace(ops, cpu_rows);
    let ntt_trace = generate_ntt_trace(Q, ops, ntt_rows).expect("NTT trace gen failed");
    let twiddle_trace = generate_twiddle_rom_trace(&twiddle_entries, twiddle_rows)
        .expect("Twiddle trace gen failed");

    let instance = ProgramInstance::new(cpu_rows, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![ntt_trace, twiddle_trace]);

    let config = Config {
        sumcheck_blinding_factor: 2,
        ..Config::dev()
    };

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    let proof = prove(
        b"NTT_E2E_Test",
        &air,
        &instance,
        &witness,
        &config,
        blinding_seed,
        None,
    );

    let proof = match proof {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[{label}] Prover failed: {e:?}");
            return false;
        }
    };

    let mut vt = Transcript::<H>::new(b"NTT_E2E_Test");
    HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut vt, &config).unwrap_or_else(|e| {
        eprintln!("[{label}] Verifier error: {e:?}");
        false
    })
}

fn run_tampered_ntt<T>(ops: &[NttOp], tamper: T) -> bool
where
    T: FnOnce(&mut [ColumnTrace]),
{
    run_tampered_ntt_with_cpu(ops, |chiplet_traces, _cpu| tamper(chiplet_traces))
}

fn run_tampered_ntt_with_cpu<T>(ops: &[NttOp], tamper: T) -> bool
where
    T: FnOnce(&mut [ColumnTrace], &mut ColumnTrace),
{
    let num_ops = ops.len();
    let cpu_rows = (num_ops + 1).next_power_of_two().max(4);
    let ntt_rows = (num_ops + 1).next_power_of_two().max(4);

    let twiddle_entries: Vec<TwiddleEntry> = ops
        .iter()
        .map(|op| match op {
            NttOp::Butterfly(b) => TwiddleEntry {
                layer: b.layer,
                butterfly_idx: b.butterfly_idx,
                w: b.w,
                is_mulonly: false,
                active: true,
                request_idx_tr: 0,
            },
            NttOp::MulOnly(m) => TwiddleEntry {
                layer: m.layer,
                butterfly_idx: m.butterfly_idx,
                w: m.w,
                is_mulonly: m.is_basemul,
                active: true,
                request_idx_tr: 0,
            },
            NttOp::FlowCompanion(_) => TwiddleEntry {
                layer: 0,
                butterfly_idx: 0,
                w: 0,
                is_mulonly: false,
                active: false,
                request_idx_tr: 0,
            },
        })
        .collect();

    let twiddle_rows = (twiddle_entries.len() + 1).next_power_of_two().max(4);

    let air = NttIsolatedProgram {
        ntt_rows,
        twiddle_rows,
    };

    let mut cpu_trace = generate_cpu_trace(ops, cpu_rows);

    let ntt_trace = generate_ntt_trace(Q, ops, ntt_rows).expect("NTT trace gen failed");
    let twiddle_trace = generate_twiddle_rom_trace(&twiddle_entries, twiddle_rows)
        .expect("Twiddle trace gen failed");

    let mut chiplet_traces = vec![ntt_trace, twiddle_trace];

    tamper(&mut chiplet_traces, &mut cpu_trace);

    let instance = ProgramInstance::new(cpu_rows, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(chiplet_traces);

    let config = Config {
        sumcheck_blinding_factor: 2,
        ..Config::dev()
    };

    let mut seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut seed).unwrap();

    let result = prove(
        b"NTT_E2E_Test",
        &air,
        &instance,
        &witness,
        &config,
        seed,
        None,
    );

    match result {
        Err(_) => true,
        Ok(proof) => {
            let mut vt = Transcript::<H>::new(b"NTT_E2E_Test");
            let result = HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut vt, &config);
            result.is_err() || !result.unwrap()
        }
    }
}

// =================================================================
// Happy path tests
// =================================================================

#[test]
fn single_butterfly_w17() {
    let ops = vec![NttOp::Butterfly(NttButterfly {
        a: 1000,
        b: 2000,
        w: 17,
        layer: 0,
        butterfly_idx: 0,
        is_forward: false,
        ntt_instance: 0,
        pos_a: 0,
        pos_b: 0,
    })];

    assert!(prove_and_verify(&ops, "single_butterfly_w17"));
}

#[test]
fn single_butterfly_w1() {
    let ops = vec![NttOp::Butterfly(NttButterfly {
        a: 1500,
        b: 2000,
        w: 1,
        layer: 0,
        butterfly_idx: 0,
        is_forward: false,
        ntt_instance: 0,
        pos_a: 0,
        pos_b: 0,
    })];

    assert!(prove_and_verify(&ops, "single_butterfly_w1"));
}

#[test]
fn single_mul_only() {
    let ops = vec![NttOp::MulOnly(NttMulOnly {
        b: 1234,
        w: 567,
        layer: 0,
        butterfly_idx: 0,
        is_basemul: false,
        flow_pos: None,
        flow_instance: 0,
        flow_src_layer: 0,
    })];

    assert!(prove_and_verify(&ops, "single_mul_only"));
}

#[test]
fn butterfly_near_q_boundary() {
    let ops = vec![NttOp::Butterfly(NttButterfly {
        a: 3328,
        b: 3328,
        w: 3328,
        layer: 0,
        butterfly_idx: 0,
        is_forward: false,
        ntt_instance: 0,
        pos_a: 0,
        pos_b: 0,
    })];

    assert!(prove_and_verify(&ops, "butterfly_near_q"));
}

#[test]
fn butterfly_a_zero() {
    let ops = vec![NttOp::Butterfly(NttButterfly {
        a: 0,
        b: 1000,
        w: 17,
        layer: 0,
        butterfly_idx: 0,
        is_forward: false,
        ntt_instance: 0,
        pos_a: 0,
        pos_b: 0,
    })];

    assert!(prove_and_verify(&ops, "butterfly_a_zero"));
}

#[test]
fn mixed_butterfly_and_mul() {
    let ops = vec![
        NttOp::Butterfly(NttButterfly {
            a: 100,
            b: 200,
            w: 17,
            layer: 0,
            butterfly_idx: 0,
            is_forward: false,
            ntt_instance: 0,
            pos_a: 0,
            pos_b: 0,
        }),
        NttOp::MulOnly(NttMulOnly {
            b: 300,
            w: 42,
            layer: 0,
            butterfly_idx: 1,
            is_basemul: false,
            flow_pos: None,
            flow_instance: 0,
            flow_src_layer: 0,
        }),
        NttOp::Butterfly(NttButterfly {
            a: 3000,
            b: 1000,
            w: 1,
            layer: 1,
            butterfly_idx: 0,
            is_forward: false,
            ntt_instance: 0,
            pos_a: 0,
            pos_b: 0,
        }),
    ];

    assert!(prove_and_verify(&ops, "mixed_ops"));
}

#[test]
fn empty_ntt_trace() {
    assert!(prove_and_verify(&[], "empty_ntt"));
}

// =================================================================
// Flow connectivity tests
// =================================================================

fn two_layer_forward_ops() -> Vec<NttOp> {
    let q = Q as u64;

    let a0 = 100u32;
    let b0 = 200u32;
    let w0 = 17u32;
    let wb0 = ((w0 as u64 * b0 as u64) % q) as u32;
    let a0_out = ((a0 as u64 + wb0 as u64) % q) as u32;
    let b0_out = ((a0 as u64 + q - wb0 as u64) % q) as u32;

    let a1 = a0_out;
    let b1 = b0_out;
    let w1 = 42u32;

    vec![
        NttOp::Butterfly(NttButterfly {
            a: a0,
            b: b0,
            w: w0,
            layer: 0,
            butterfly_idx: 0,
            is_forward: true,
            ntt_instance: 1,
            pos_a: 0,
            pos_b: 1,
        }),
        NttOp::FlowCompanion(NttFlowCompanion {
            b_in: b0,
            b_out: b0_out,
            layer: 0,
            ntt_instance: 1,
            pos: 1,
            src_layer: 0,
            is_flow_output: true,
            is_flow_input: false,
            is_forward: true,
        }),
        NttOp::Butterfly(NttButterfly {
            a: a1,
            b: b1,
            w: w1,
            layer: 1,
            butterfly_idx: 0,
            is_forward: true,
            ntt_instance: 1,
            pos_a: 0,
            pos_b: 1,
        }),
        NttOp::FlowCompanion(NttFlowCompanion {
            b_in: b1,
            b_out: ((a1 as u64 + q - ((w1 as u64 * b1 as u64) % q)) % q) as u32,
            layer: 1,
            ntt_instance: 1,
            pos: 1,
            src_layer: 0,
            is_flow_output: false,
            is_flow_input: true,
            is_forward: true,
        }),
    ]
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn forward_butterfly_flow_connectivity() {
    assert!(prove_and_verify(&two_layer_forward_ops(), "forward_flow"));
}

// =================================================================
// Adversarial tests
// =================================================================

// Corrupt a_out in NTT chiplet.
// Butterfly constraint a + wb = a_out + flag*q fails.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_corrupted_a_out() {
    let ops = vec![NttOp::Butterfly(NttButterfly {
        a: 1000,
        b: 2000,
        w: 17,
        layer: 0,
        butterfly_idx: 0,
        is_forward: false,
        ntt_instance: 0,
        pos_a: 0,
        pos_b: 0,
    })];

    let layout = hekate_pqc::ntt::NttLayout::compute(Q, 12);
    let bus_a_out_phy = layout.num_packed_b32_cols + 4;

    let detected = run_tampered_ntt(&ops, |traces| {
        set_b32(&mut traces[0], bus_a_out_phy, 0, 9999);
    });

    assert!(detected, "corrupted a_out must be detected");
}

// Wrong twiddle factor vs TwiddleROM.
// GPA twiddle bus rejects the mismatch.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_wrong_twiddle() {
    let correct_ops = vec![NttOp::Butterfly(NttButterfly {
        a: 1000,
        b: 2000,
        w: 42,
        layer: 0,
        butterfly_idx: 0,
        is_forward: false,
        ntt_instance: 0,
        pos_a: 0,
        pos_b: 0,
    })];

    // Twiddle ROM expects w=17 at (layer=0, bfly=0).
    // NTT trace has w=42.
    // Keys differ -> bus rejects.
    let cpu_rows = 4;
    let ntt_rows = 4;
    let twiddle_rows = 4;

    let twiddle_entries = vec![TwiddleEntry {
        layer: 0,
        butterfly_idx: 0,
        w: 17,
        is_mulonly: false,
        active: true,
        request_idx_tr: 0,
    }];

    let air = NttIsolatedProgram {
        ntt_rows,
        twiddle_rows,
    };

    let cpu_trace = generate_cpu_trace(&correct_ops, cpu_rows);
    let ntt_trace = generate_ntt_trace(Q, &correct_ops, ntt_rows).expect("trace gen");
    let twiddle_trace =
        generate_twiddle_rom_trace(&twiddle_entries, twiddle_rows).expect("twiddle gen");

    let instance = ProgramInstance::new(cpu_rows, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![ntt_trace, twiddle_trace]);

    let config = Config {
        sumcheck_blinding_factor: 2,
        ..Config::dev()
    };

    let mut seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut seed).unwrap();

    let result = prove(
        b"NTT_E2E_Test",
        &air,
        &instance,
        &witness,
        &config,
        seed,
        None,
    );

    match result {
        Err(_) => {}
        Ok(proof) => {
            let mut vt = Transcript::<H>::new(b"NTT_E2E_Test");
            let valid = HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut vt, &config);

            assert!(
                valid.is_err() || !valid.unwrap(),
                "wrong twiddle must be detected"
            );
        }
    }
}

// b_out >= q violates the range check.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_b_out_out_of_range() {
    let ops = vec![NttOp::Butterfly(NttButterfly {
        a: 1000,
        b: 500,
        w: 1,
        layer: 0,
        butterfly_idx: 0,
        is_forward: false,
        ntt_instance: 0,
        pos_a: 0,
        pos_b: 0,
    })];

    let layout = hekate_pqc::ntt::NttLayout::compute(Q, 12);
    let bus_b_out_phy = layout.num_packed_b32_cols + 5;

    let detected = run_tampered_ntt_with_cpu(&ops, |traces, cpu| {
        set_b32(&mut traces[0], bus_b_out_phy, 0, Q);
        set_b32(cpu, CPU_B_OUT, 0, Q);
    });

    assert!(detected, "out-of-range b_out must be detected");
}

// Corrupt flow output value on a companion row.
// Flow bus clock stitching detects the mismatch.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_flow_companion_value_tamper() {
    let ops = two_layer_forward_ops();

    let layout = hekate_pqc::ntt::NttLayout::compute(Q, 12);
    let phy_bus_a_out = layout.num_packed_b32_cols + 4;

    let detected = run_tampered_ntt(&ops, |traces| {
        // Companion at row 1 carries pos_b output.
        // Corrupt bus_a_out (= b_out for companion).
        flip_b32(&mut traces[0], phy_bus_a_out, 1, 0xFF);
    });

    assert!(detected, "flow companion value tamper must be detected");
}

// Duplicate a companion row's flow data onto a
// padding row. Without flow clock, char-2
// cancellation hides the duplicate. With clock,
// the duplicate has a different RowIndex -> rejected.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_flow_companion_duplication() {
    let ops = two_layer_forward_ops();

    let layout = hekate_pqc::ntt::NttLayout::compute(Q, 12);
    let phy_bus_a = layout.num_packed_b32_cols;
    let phy_bus_a_out = layout.num_packed_b32_cols + 4;
    let phy_layer = layout.num_packed_b32_cols + 6;
    let phy_ntt_instance = layout.num_packed_b32_cols + 8;
    let phy_pos_a = layout.num_packed_b32_cols + 9;
    let phy_src_layer = layout.num_packed_b32_cols + 11;
    let phy_s_companion = layout.num_packed_b32_cols + 19;
    let phy_s_flow_output = layout.num_packed_b32_cols + 20;

    let detected = run_tampered_ntt(&ops, |traces| {
        let ntt = &mut traces[0];

        // Row 1 is the honest companion (layer 0, pos 1).
        // Copy its flow data to a padding row.
        let pad_row = ntt.columns[0].len() - 1;

        let inst = read_b32(ntt, phy_ntt_instance, 1);
        let layer = read_b32(ntt, phy_layer, 1);
        let pos = read_b32(ntt, phy_pos_a, 1);
        let a_out = read_b32(ntt, phy_bus_a_out, 1);
        let b_in = read_b32(ntt, phy_bus_a, 1);
        let src_layer = read_b32(ntt, phy_src_layer, 1);

        set_b32(ntt, phy_ntt_instance, pad_row, inst);
        set_b32(ntt, phy_layer, pad_row, layer);
        set_b32(ntt, phy_pos_a, pad_row, pos);
        set_b32(ntt, phy_bus_a_out, pad_row, a_out);
        set_b32(ntt, phy_bus_a, pad_row, b_in);
        set_b32(ntt, phy_src_layer, pad_row, src_layer);

        set_bit_val(ntt, phy_s_companion, pad_row, Bit::ONE);
        set_bit_val(ntt, phy_s_flow_output, pad_row, Bit::ONE);
    });

    assert!(
        detected,
        "companion row duplication must be detected by flow clock"
    );
}

#[test]
fn scribble_ntt_flip_selector_caught() {
    let ops = vec![
        NttOp::Butterfly(NttButterfly {
            a: 100,
            b: 200,
            w: 17,
            layer: 0,
            butterfly_idx: 0,
            is_forward: false,
            ntt_instance: 0,
            pos_a: 0,
            pos_b: 0,
        }),
        NttOp::Butterfly(NttButterfly {
            a: 50,
            b: 70,
            w: 23,
            layer: 0,
            butterfly_idx: 1,
            is_forward: false,
            ntt_instance: 0,
            pos_a: 0,
            pos_b: 0,
        }),
    ];

    let cpu_rows = (ops.len() + 1).next_power_of_two().max(4);
    let ntt_rows = cpu_rows;

    let twiddle_entries: Vec<TwiddleEntry> = ops
        .iter()
        .map(|op| match op {
            NttOp::Butterfly(b) => TwiddleEntry {
                layer: b.layer,
                butterfly_idx: b.butterfly_idx,
                w: b.w,
                is_mulonly: false,
                active: true,
                request_idx_tr: 0,
            },
            NttOp::MulOnly(m) => TwiddleEntry {
                layer: m.layer,
                butterfly_idx: m.butterfly_idx,
                w: m.w,
                is_mulonly: m.is_basemul,
                active: true,
                request_idx_tr: 0,
            },
            NttOp::FlowCompanion(_) => TwiddleEntry {
                layer: 0,
                butterfly_idx: 0,
                w: 0,
                is_mulonly: false,
                active: false,
                request_idx_tr: 0,
            },
        })
        .collect();

    let twiddle_rows = (twiddle_entries.len() + 1).next_power_of_two().max(4);

    let air = NttIsolatedProgram {
        ntt_rows,
        twiddle_rows,
    };

    let cpu_trace = generate_cpu_trace(&ops, cpu_rows);
    let ntt_trace = generate_ntt_trace(Q, &ops, ntt_rows).expect("NTT trace gen");
    let twiddle_trace =
        generate_twiddle_rom_trace(&twiddle_entries, twiddle_rows).expect("Twiddle trace gen");

    let instance = ProgramInstance::new(cpu_rows, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![ntt_trace, twiddle_trace]);

    assert_all_caught_all_targets(
        &air,
        &instance,
        &witness,
        ScribbleConfig::default()
            .mutations([MutationKind::FlipSelector])
            .cases(256),
    );
}
