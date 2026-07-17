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

use hekate_core::config::Config;
use hekate_core::trace::ColumnTrace;
use hekate_core::trace::{ColumnType, TraceBuilder, TraceColumn};
use hekate_crypto::DefaultHasher;
use hekate_crypto::transcript::Transcript;
use hekate_keccak::{KeccakChiplet, KeccakWitness};
use hekate_math::{Bit, Block32, Block64, Block128, Flat, HardwareField, TowerField};
use hekate_pqc::high_bits::HighBitsLayout;
use hekate_pqc::mldsa::{
    self, CpuMlDsaColumns, CpuMlDsaUnit, MlDsaChiplet, MlDsaCtrlColumns, MlDsaLevel,
    MlDsaPublicKey, MlDsaSignature,
};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::constraint::{BoundaryConstraint, ConstraintAst};
use hekate_program::permutation::PermutationCheckSpec;
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_sdk::preflight;
use hekate_verifier::HekateVerifier;
use ml_dsa::signature::{Keypair, Signer};
use ml_dsa::{B32, MlDsa44, MlDsa65, MlDsa87, SigningKey};
use rand::{TryRngCore, rngs::OsRng};

type F = Block128;
type H = DefaultHasher;

macro_rules! nist_mldsa {
    ($fn:ident, $level:ty) => {
        fn $fn(msg: &[u8]) -> (Vec<u8>, Vec<u8>) {
            let mut xi = [0u8; 32];
            OsRng.try_fill_bytes(&mut xi).unwrap();

            let key = SigningKey::<$level>::from_seed(&B32::from(xi));
            let pk = key.verifying_key().encode();
            let sig = key.sign(msg).encode();

            (pk.as_slice().to_vec(), sig.as_slice().to_vec())
        }
    };
}

nist_mldsa!(nist_mldsa_44, MlDsa44);
nist_mldsa!(nist_mldsa_65, MlDsa65);
nist_mldsa!(nist_mldsa_87, MlDsa87);

#[derive(Clone)]
struct MlDsaTestProgram {
    mldsa: MlDsaChiplet<F>,
    num_public: usize,
}

impl Air<F> for MlDsaTestProgram {
    fn num_columns(&self) -> usize {
        CpuMlDsaUnit::num_columns()
    }

    fn boundary_constraints(&self) -> Vec<BoundaryConstraint<F>> {
        (0..self.num_public)
            .map(|k| BoundaryConstraint::with_public_input(CpuMlDsaColumns::DATA, k, k))
            .collect()
    }

    fn column_layout(&self) -> &[ColumnType] {
        Box::leak(CpuMlDsaColumns::build_layout().into_boxed_slice())
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![(
            mldsa::MLDSA_DATA_BUS_ID.into(),
            CpuMlDsaUnit::linking_spec(),
        )]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(CpuMlDsaColumns::SELECTOR));

        cs.build()
    }
}

impl Program<F> for MlDsaTestProgram {
    fn num_public_inputs(&self) -> usize {
        self.num_public
    }

    fn chiplet_defs(&self) -> hekate_core::errors::Result<Vec<ChipletDef<F>>> {
        self.mldsa.composite().flatten_defs()
    }
}

fn prove_and_verify_mldsa(
    level: MlDsaLevel,
    pk_bytes: &[u8],
    sig_bytes: &[u8],
    msg: &[u8],
) -> Result<bool, String> {
    let mldsa_chiplet = MlDsaChiplet::<F>::new(level);

    let pk = MlDsaPublicKey::from_bytes(level, pk_bytes);
    let sig = MlDsaSignature::from_bytes(level, sig_bytes).expect("NIST signature should parse");

    let chiplet_traces = mldsa_chiplet
        .generate_traces(&pk, &sig, msg)
        .map_err(|e| format!("trace gen: {e:?}"))?;

    let cpu_rows: usize = 1 << 10;
    let layout = CpuMlDsaColumns::build_layout();

    let mut cpu_tb = TraceBuilder::new(&layout, cpu_rows.trailing_zeros() as usize).unwrap();

    // Deposit c̃ from signature as IO data.
    // Pad to B32 alignment.
    let mut io_buf = sig.c_tilde.clone();
    while !io_buf.len().is_multiple_of(4) {
        io_buf.push(0);
    }

    for (i, chunk) in io_buf.chunks(4).enumerate() {
        let val = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        cpu_tb
            .set_b32(CpuMlDsaColumns::DATA, i, Block32::from(val))
            .unwrap();
        cpu_tb
            .set_bit(CpuMlDsaColumns::SELECTOR, i, Bit::ONE)
            .unwrap();
    }

    let cpu_trace = cpu_tb.build();

    let public_inputs: Vec<F> = io_buf
        .chunks(4)
        .map(|chunk| Block128(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as u128))
        .collect();

    let air = MlDsaTestProgram {
        mldsa: mldsa_chiplet,
        num_public: public_inputs.len(),
    };

    let instance = ProgramInstance::new(cpu_rows, public_inputs);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(chiplet_traces);

    let defs = air.chiplet_defs().expect("chiplet_defs");
    for (i, (def, ct)) in defs.iter().zip(witness.chiplet_traces.iter()).enumerate() {
        let mut report = preflight::PreflightReport::new();
        let result = preflight::check_chiplet_constraints(
            std::slice::from_ref(def),
            std::slice::from_ref(ct),
            &mut report,
        );

        match result {
            Ok(()) if report.is_clean() => eprintln!("chiplet[{i}] {}: OK", def.name()),
            Ok(()) => {
                eprintln!(
                    "chiplet[{i}] {}: {} violations",
                    def.name(),
                    report.constraint_violations.len()
                );

                for v in report.constraint_violations.iter().take(10) {
                    eprintln!(
                        "  constraint={} label={:?} row={} val={:?}",
                        v.constraint_idx, v.label, v.row_idx, v.value
                    );
                }
            }
            Err(e) => eprintln!("chiplet[{i}] {}: error {e:?}", def.name()),
        }
    }

    let mut bus_report = preflight::PreflightReport::new();
    preflight::check_bus_multisets(&air, &witness, &mut bus_report)
        .map_err(|e| format!("bus check: {e:?}"))?;

    for d in &bus_report.bus_diagnostics {
        eprintln!("bus \"{}\":", d.bus_id);

        for ep in &d.endpoints {
            let table = match ep.source {
                preflight::TableId::Main => "Main".into(),
                preflight::TableId::Chiplet(i) => format!("Chiplet({i})"),
            };

            eprintln!("  {table}: active={}", ep.active_rows);
        }
    }

    let config = Config {
        sumcheck_blinding_factor: 2,
        ..Config::default()
    };

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    let proof = prove(
        b"MLDSA_E2E",
        &air,
        &instance,
        &witness,
        &config,
        blinding_seed,
        None,
    )
    .map_err(|e| format!("prover: {e:?}"))?;

    let mut vt = Transcript::<H>::new(b"MLDSA_E2E");
    HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut vt, &config)
        .map_err(|e| format!("verifier: {e:?}"))
}

// m=44, exercises uh_wrap
// for non-power-of-2 modulus.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn mldsa_44_e2e() {
    let msg = b"Hekate ML-DSA-44 e2e test";
    let (nist_pk, nist_sig) = nist_mldsa_44(msg);

    let result = prove_and_verify_mldsa(MlDsaLevel::MLDSA_44, &nist_pk, &nist_sig, msg);

    match result {
        Ok(true) => {}
        Ok(false) => panic!("verifier rejected honest proof"),
        Err(e) => panic!("error: {e}"),
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn mldsa_65_e2e() {
    let msg = b"Hekate ML-DSA e2e test";
    let (nist_pk, nist_sig) = nist_mldsa_65(msg);

    let result = prove_and_verify_mldsa(MlDsaLevel::MLDSA_65, &nist_pk, &nist_sig, msg);

    match result {
        Ok(true) => {}
        Ok(false) => panic!("verifier rejected honest proof"),
        Err(e) => panic!("error: {e}"),
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn mldsa_87_e2e() {
    let msg = b"Hekate ML-DSA-87 e2e test";
    let (nist_pk, nist_sig) = nist_mldsa_87(msg);

    let result = prove_and_verify_mldsa(MlDsaLevel::MLDSA_87, &nist_pk, &nist_sig, msg);

    match result {
        Ok(true) => {}
        Ok(false) => panic!("verifier rejected honest proof"),
        Err(e) => panic!("error: {e}"),
    }
}

// =================================================================
// Adversarial Test Harness
// =================================================================

fn run_tampered_mldsa_65<T>(tamper: T) -> bool
where
    T: FnOnce(&mut [ColumnTrace]),
{
    run_tampered_mldsa_65_with_cpu(|chiplet_traces, _cpu_trace| tamper(chiplet_traces))
}

fn run_tampered_mldsa_65_with_cpu<T>(tamper: T) -> bool
where
    T: FnOnce(&mut [ColumnTrace], &mut ColumnTrace),
{
    let msg = b"adversarial test";
    let (nist_pk, nist_sig) = nist_mldsa_65(msg);

    let level = MlDsaLevel::MLDSA_65;
    let mldsa_chiplet = MlDsaChiplet::<F>::new(level);

    let pk = MlDsaPublicKey::from_bytes(level, &nist_pk);
    let sig = MlDsaSignature::from_bytes(level, &nist_sig).expect("NIST signature must parse");

    let mut chiplet_traces = mldsa_chiplet
        .generate_traces(&pk, &sig, msg)
        .expect("trace gen failed");

    let cpu_rows: usize = 1 << 10;
    let layout = CpuMlDsaColumns::build_layout();

    let mut cpu_tb = TraceBuilder::new(&layout, cpu_rows.trailing_zeros() as usize).unwrap();

    let mut io_buf = sig.c_tilde.clone();
    while !io_buf.len().is_multiple_of(4) {
        io_buf.push(0);
    }

    for (i, chunk) in io_buf.chunks(4).enumerate() {
        let val = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        cpu_tb
            .set_b32(CpuMlDsaColumns::DATA, i, Block32::from(val))
            .unwrap();
        cpu_tb
            .set_bit(CpuMlDsaColumns::SELECTOR, i, Bit::ONE)
            .unwrap();
    }

    let mut cpu_trace = cpu_tb.build();

    tamper(&mut chiplet_traces, &mut cpu_trace);

    let public_inputs: Vec<F> = io_buf
        .chunks(4)
        .map(|chunk| Block128(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as u128))
        .collect();

    let air = MlDsaTestProgram {
        mldsa: mldsa_chiplet,
        num_public: public_inputs.len(),
    };

    let instance = ProgramInstance::new(cpu_rows, public_inputs);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(chiplet_traces);

    let config = Config {
        sumcheck_blinding_factor: 2,
        ..Config::default()
    };

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    let proof_result = prove(
        b"MLDSA_Adversarial",
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
            let mut vt = Transcript::<H>::new(b"MLDSA_Adversarial");
            let result = HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut vt, &config);
            result.is_err() || !result.unwrap()
        }
    }
}

// =================================================================
// Helpers
// =================================================================

fn first_row_with_bit(trace: &ColumnTrace, col: usize) -> usize {
    let bits = trace.columns[col].as_bit_slice().unwrap();
    (0..bits.len())
        .find(|&r| bits[r] == Bit::ONE)
        .expect("no row with bit set")
}

fn rows_with_bit(trace: &ColumnTrace, col: usize) -> Vec<usize> {
    let bits = trace.columns[col].as_bit_slice().unwrap();
    (0..bits.len()).filter(|&r| bits[r] == Bit::ONE).collect()
}

fn coactivated_row(trace: &ColumnTrace, col_a: usize, col_b: usize) -> usize {
    let bits_a = trace.columns[col_a].as_bit_slice().unwrap();
    let bits_b = trace.columns[col_b].as_bit_slice().unwrap();

    (0..bits_a.len())
        .find(|&r| bits_a[r] == Bit::ONE && bits_b[r] == Bit::ONE)
        .expect("no co-activated row found")
}

fn flip_b32(trace: &mut ColumnTrace, col: usize, row: usize, mask: u32) {
    match &mut trace.columns[col] {
        TraceColumn::B32(data) => {
            let original = data[row];
            data[row] = Flat::from_raw(Block32(original.to_tower().0 ^ mask));
        }
        _ => panic!("expected B32 column at {col}"),
    }
}

fn flip_b64(trace: &mut ColumnTrace, col: usize, row: usize, mask: u64) {
    match &mut trace.columns[col] {
        TraceColumn::B64(data) => {
            let original = data[row];
            data[row] = Flat::from_raw(Block64(original.to_tower().0 ^ mask));
        }
        _ => panic!("expected B64 column at {col}"),
    }
}

fn swap_b32(trace: &mut ColumnTrace, col: usize, r0: usize, r1: usize) {
    match &mut trace.columns[col] {
        TraceColumn::B32(data) => data.swap(r0, r1),
        _ => panic!("expected B32 column at {col}"),
    }
}

#[allow(dead_code)]
fn swap_b64(trace: &mut ColumnTrace, col: usize, r0: usize, r1: usize) {
    match &mut trace.columns[col] {
        TraceColumn::B64(data) => data.swap(r0, r1),
        _ => panic!("expected B64 column at {col}"),
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

// =================================================================
// Adversarial Exploit Tests
// =================================================================

// RAM_VAL_PACKED is not on ntt_data
// or ram_link bus for NttRam rows.
// Binding constraint:
// ntt_sel * ram_sel * (ntt_b + ram_val_packed) = 0
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_ntt_ram_mismatch() {
    let detected = run_tampered_mldsa_65(|traces| {
        let ctrl = &mut traces[0];
        let row = coactivated_row(
            ctrl,
            MlDsaCtrlColumns::NTT_SELECTOR,
            MlDsaCtrlColumns::RAM_SELECTOR,
        );

        flip_b32(ctrl, MlDsaCtrlColumns::RAM_VAL_PACKED, row, 0x1);
    });

    assert!(
        detected,
        "NTT_B != RAM_VAL_PACKED accepted without binding constraint"
    );
}

// ml_dsa_data bus links CPU DATA to
// ctrl IO_DATA. Flipping one side
// breaks the GPA permutation product.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_bus_io_corruption() {
    let detected = run_tampered_mldsa_65_with_cpu(|_chiplet_traces, cpu_trace| {
        let row = first_row_with_bit(cpu_trace, CpuMlDsaColumns::SELECTOR);
        flip_b32(cpu_trace, CpuMlDsaColumns::DATA, row, 0x1);
    });

    assert!(
        detected,
        "CPU IO data tamper must be caught by ml_dsa_data bus"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_mldsa_data_duplicate_cpu_request_rejected() {
    let detected = run_tampered_mldsa_65_with_cpu(|_chiplet_traces, cpu_trace| {
        let value = match &cpu_trace.columns[CpuMlDsaColumns::DATA] {
            TraceColumn::B32(data) => data[0],
            _ => panic!("expected B32 column at DATA"),
        };

        let bits = cpu_trace.columns[CpuMlDsaColumns::SELECTOR]
            .as_bit_slice()
            .unwrap();
        let target = (0..bits.len())
            .find(|&r| bits[r] == Bit::ZERO)
            .expect("no free CPU row to inject duplicate");

        match &mut cpu_trace.columns[CpuMlDsaColumns::DATA] {
            TraceColumn::B32(data) => data[target] = value,
            _ => unreachable!(),
        }

        match &mut cpu_trace.columns[CpuMlDsaColumns::SELECTOR] {
            TraceColumn::Bit(data) => data[target] = Bit::ONE,
            _ => unreachable!(),
        }
    });

    assert!(
        detected,
        "duplicate CPU IO request without chiplet partner must be caught by ml_dsa_data bus"
    );
}

// twiddle_w_binding bus is a multiset check
// on (bfly_idx, w). Swapping two W_BIND rows
// preserves the multiset but breaks the next
// row butterfly index continuity constraint.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_w_side_unbind() {
    let detected = run_tampered_mldsa_65(|traces| {
        let ctrl = &mut traces[0];
        let wb_rows = rows_with_bit(ctrl, MlDsaCtrlColumns::W_BIND_SELECTOR);

        assert!(wb_rows.len() >= 2);

        let (r0, r1) = (wb_rows[0], wb_rows[1]);
        swap_b32(ctrl, MlDsaCtrlColumns::W_BIND_BFLY_IDX, r0, r1);
        swap_b32(ctrl, MlDsaCtrlColumns::RAM_VAL_PACKED, r0, r1);
    });

    assert!(
        detected,
        "W_BIND row swap must be caught by butterfly index continuity constraint"
    );
}

// Schedule groups NTT ops by (phase, instance).
// Consecutive NTT_SELECTOR=1 rows share
// NTT_INSTANCE. Swapping across instances
// breaks the contiguity constraint.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_ntt_flow_scramble() {
    let detected = run_tampered_mldsa_65(|traces| {
        let ctrl = &mut traces[0];
        let ntt_rows = rows_with_bit(ctrl, MlDsaCtrlColumns::NTT_SELECTOR);
        let bound_in = rows_with_bit(ctrl, MlDsaCtrlColumns::BOUND_IN_SEL);
        let bound_out = rows_with_bit(ctrl, MlDsaCtrlColumns::BOUND_OUT_SEL);

        let non_bound: Vec<usize> = ntt_rows
            .into_iter()
            .filter(|r| !bound_in.contains(r) && !bound_out.contains(r))
            .collect();
        assert!(non_bound.len() >= 2);

        let inst_0 = read_b32(ctrl, MlDsaCtrlColumns::NTT_INSTANCE, non_bound[0]);
        let other = non_bound
            .iter()
            .find(|&&r| read_b32(ctrl, MlDsaCtrlColumns::NTT_INSTANCE, r) != inst_0)
            .expect("need two NTT rows from different instances");

        let (r0, r1) = (non_bound[0], *other);

        // Swap all ntt_data bus
        // columns, multiset unchanged.
        swap_b32(ctrl, MlDsaCtrlColumns::NTT_A, r0, r1);
        swap_b32(ctrl, MlDsaCtrlColumns::NTT_B, r0, r1);
        swap_b32(ctrl, MlDsaCtrlColumns::NTT_A_OUT, r0, r1);
        swap_b32(ctrl, MlDsaCtrlColumns::NTT_B_OUT, r0, r1);
        swap_b32(ctrl, MlDsaCtrlColumns::NTT_LAYER, r0, r1);
        swap_b32(ctrl, MlDsaCtrlColumns::NTT_BUTTERFLY, r0, r1);
        swap_b32(ctrl, MlDsaCtrlColumns::NTT_INSTANCE, r0, r1);
    });

    assert!(
        detected,
        "NTT row swap must be caught by instance contiguity constraint"
    );
}

// RATE_REG is sticky sponge state, never on
// any bus, never constrained. Corrupting it
// on any row (including boundary) poisons
// downstream absorptions.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_ntt_boundary_break() {
    let detected = run_tampered_mldsa_65(|traces| {
        let ctrl = &mut traces[0];
        let bound_rows = rows_with_bit(ctrl, MlDsaCtrlColumns::BOUND_IN_SEL);

        flip_b64(ctrl, MlDsaCtrlColumns::RATE_REG, bound_rows[0], 0x1);
    });

    assert!(
        detected,
        "boundary RATE_REG corruption must be caught by sponge carry constraint"
    );
}

// Fabricate Keccak input, recompute
// keccak_f, update ctrl+chiplet traces
// consistently. Bus balanced, but
// capacity lane continuity breaks:
// tampered output changes capacity lanes
// that the next input row must match.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_keccak_input_unbind() {
    #[allow(clippy::needless_range_loop)]
    let detected = run_tampered_mldsa_65(|traces| {
        let keccak_trace = &mut traces[1];

        let mut original_input = [0u64; 25];
        for lane in 0..25 {
            match &keccak_trace.columns[lane] {
                TraceColumn::B64(data) => original_input[lane] = data[0].to_tower().0,
                _ => panic!("Keccak lane must be B64"),
            }
        }

        let mut tampered_input = original_input;
        tampered_input[0] ^= 1;

        let mut state = tampered_input;
        let mut round_states = Vec::with_capacity(25);

        for round in 0..24 {
            round_states.push(state);
            state = KeccakWitness::keccak_f_round(state, KeccakChiplet::ROUND_CONSTANTS[round]);
        }

        round_states.push(state);

        let tampered_output = state;

        for row in 0..25 {
            for lane in 0..25 {
                match &mut keccak_trace.columns[lane] {
                    TraceColumn::B64(data) => {
                        data[row] = Block64::from(round_states[row][lane]).to_hardware();
                    }
                    _ => panic!("Keccak lane must be B64"),
                }
            }
        }

        let ctrl = &mut traces[0];
        let kec_rows = rows_with_bit(ctrl, MlDsaCtrlColumns::KECCAK_SELECTOR);

        assert!(kec_rows.len() >= 2);

        let (in_row, out_row) = (kec_rows[0], kec_rows[1]);

        for lane in 0..25 {
            match &mut ctrl.columns[MlDsaCtrlColumns::KECCAK_LANES + lane] {
                TraceColumn::B64(data) => {
                    data[in_row] = Block64::from(tampered_input[lane]).to_hardware();
                    data[out_row] = Block64::from(tampered_output[lane]).to_hardware();
                }
                _ => panic!("KECCAK_LANES must be B64"),
            }
        }

        // Propagate tampered sponge carry
        let sponge_init_bits = ctrl.columns[MlDsaCtrlColumns::SPONGE_INIT]
            .as_bit_slice()
            .unwrap();
        let active_bits = ctrl.columns[MlDsaCtrlColumns::S_ACTIVE]
            .as_bit_slice()
            .unwrap();

        let mut end = out_row + 1;
        while end < active_bits.len() {
            if active_bits[end] == Bit::ZERO || sponge_init_bits[end] == Bit::ONE {
                break;
            }

            end += 1;
        }

        for row in (out_row + 1)..end {
            for lane in 0..25 {
                match &mut ctrl.columns[MlDsaCtrlColumns::RATE_REG + lane] {
                    TraceColumn::B64(data) => {
                        data[row] = Block64::from(tampered_output[lane]).to_hardware();
                    }
                    _ => panic!("RATE_REG must be B64"),
                }
            }
        }
    });

    assert!(
        detected,
        "fabricated Keccak input must be caught by capacity lane continuity"
    );
}

// RATE_REG flip on output row
// breaks the sponge carry constraint:
// next row expects the unmodified value.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_sponge_rate_skip() {
    let detected = run_tampered_mldsa_65(|traces| {
        let ctrl = &mut traces[0];
        let row = first_row_with_bit(ctrl, MlDsaCtrlColumns::KEC_IS_OUTPUT);

        flip_b64(ctrl, MlDsaCtrlColumns::RATE_REG, row, 0x1);
    });

    assert!(
        detected,
        "RATE_REG corruption must be caught by sponge carry constraint"
    );
}

// IO_DATA is bus-protected.
// RAM_VAL_PACKED must equal IO_DATA on IO rows
// (io_sel * (IO_DATA + RAM_VAL_PACKED) = 0).
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_io_data_mismatch() {
    let detected = run_tampered_mldsa_65(|traces| {
        let ctrl = &mut traces[0];
        let row = first_row_with_bit(ctrl, MlDsaCtrlColumns::IO_SELECTOR);

        flip_b32(ctrl, MlDsaCtrlColumns::RAM_VAL_PACKED, row, 0xFF);
    });

    assert!(
        detected,
        "IO_DATA != RAM_VAL_PACKED must be caught by IO packing constraint"
    );
}

// c̃ bytes transit IO bus but nothing
// constrains their flow into the
// Keccak absorption. Corrupt RATE_REG
// on an IO-phase row to corrupt the
// sponge state before c̃ absorption.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_ct_substitution() {
    let detected = run_tampered_mldsa_65(|traces| {
        let ctrl = &mut traces[0];
        let io_rows = rows_with_bit(ctrl, MlDsaCtrlColumns::IO_SELECTOR);

        assert!(!io_rows.is_empty());

        flip_b64(ctrl, MlDsaCtrlColumns::RATE_REG, io_rows[0], 0xFF);
    });

    assert!(
        detected,
        "c̃ sponge corruption must be caught by sponge carry constraint"
    );
}

// PAD_SEL is mutually exclusive with
// IO_SELECTOR. Setting PAD_SEL on an
// IO row violates io_sel * pad_sel = 0.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_pad_byte_tamper() {
    let detected = run_tampered_mldsa_65(|traces| {
        let ctrl = &mut traces[0];
        let pad_rows = rows_with_bit(ctrl, MlDsaCtrlColumns::PAD_SEL);

        if !pad_rows.is_empty() {
            set_bit_val(ctrl, MlDsaCtrlColumns::PAD_SEL, pad_rows[0], Bit::ZERO);
        } else {
            let io_row = first_row_with_bit(ctrl, MlDsaCtrlColumns::IO_SELECTOR);
            set_bit_val(ctrl, MlDsaCtrlColumns::PAD_SEL, io_row, Bit::ONE);
        }
    });

    assert!(
        detected,
        "PAD_SEL tampering must be caught by IO/PAD mutual exclusivity"
    );
}

// pk data flows through Keccak (H(pk) -> tr)
// in ExpandSample phase. RATE_REG carries
// sponge state but is unconstrained.
// Corrupting it substitutes arbitrary
// pk data into the verification pipeline.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_pk_substitution() {
    let detected = run_tampered_mldsa_65(|traces| {
        let ctrl = &mut traces[0];
        let expand_rows = rows_with_bit(ctrl, MlDsaCtrlColumns::PH_EXPAND_SAMPLE);
        let active_rows = rows_with_bit(ctrl, MlDsaCtrlColumns::S_ACTIVE);

        let target = expand_rows
            .iter()
            .find(|r| active_rows.contains(r))
            .expect("need active row in ExpandSample phase");

        flip_b64(ctrl, MlDsaCtrlColumns::RATE_REG, *target, 0x1);
    });

    assert!(
        detected,
        "pk sponge corruption must be caught by sponge carry constraint"
    );
}

// Signature coefficients enter the NTT
// pipeline via RAM. RATE_REG in
// NttForward phase is unconstrained,
// corrupting it substitutes arbitrary
// data without detection.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_sig_substitution() {
    let detected = run_tampered_mldsa_65(|traces| {
        let ctrl = &mut traces[0];
        let ntt_fwd = rows_with_bit(ctrl, MlDsaCtrlColumns::PH_NTT_FORWARD);
        let active = rows_with_bit(ctrl, MlDsaCtrlColumns::S_ACTIVE);

        let target = ntt_fwd
            .iter()
            .find(|r| active.contains(r))
            .expect("need active row in NttForward phase");

        flip_b64(ctrl, MlDsaCtrlColumns::RATE_REG, *target, 0x1);
    });

    assert!(
        detected,
        "sig RATE_REG corruption must be caught by sponge carry constraint"
    );
}

// CTILDE_REF carries c̃ from the signature.
// Hard equality with RATE_REG (c̃') on
// the CMP row:
// CMP * (REF[i] + REG[i]) = 0.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_ctilde_ref_tamper() {
    let detected = run_tampered_mldsa_65(|traces| {
        let ctrl = &mut traces[0];
        let cmp_row = first_row_with_bit(ctrl, MlDsaCtrlColumns::CMP_SELECTOR);

        flip_b64(ctrl, MlDsaCtrlColumns::CTILDE_REF, cmp_row, 0xDEAD);
    });

    assert!(
        detected,
        "CTILDE_REF tampering must be caught by c̃==c̃' hard equality"
    );
}

// HASH_EQ bits are constrained via
// bidirectional equality (ML-KEM pattern).
//
// Corrupt CTILDE_REF so c̃ ≠ c̃', then
// claim equality via HASH_EQ=1. Forward
// constraint CMP * eq * diff = 0 catches
// the false claim (eq=1, diff≠0).
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_ctilde_hash_mismatch() {
    let detected = run_tampered_mldsa_65(|traces| {
        let ctrl = &mut traces[0];
        let cmp_row = first_row_with_bit(ctrl, MlDsaCtrlColumns::CMP_SELECTOR);

        flip_b64(ctrl, MlDsaCtrlColumns::CTILDE_REF, cmp_row, 0x1);
        set_bit_val(ctrl, MlDsaCtrlColumns::HASH_EQ_LO, cmp_row, Bit::ONE);
        set_bit_val(ctrl, MlDsaCtrlColumns::HASH_EQ_HI, cmp_row, Bit::ONE);
    });

    assert!(
        detected,
        "false HASH_EQ claim must be caught by bidirectional equality constraint"
    );
}

// norm_check bus is a multiset check.
// First-row constraint anchors NC_IDX=0;
// swapping the first two rows puts idx≠0
// on the first NC row.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_norm_check_bypass() {
    let detected = run_tampered_mldsa_65(|traces| {
        let ctrl = &mut traces[0];
        let nc_rows = rows_with_bit(ctrl, MlDsaCtrlColumns::NC_SELECTOR);

        assert!(nc_rows.len() >= 2);

        let (r0, r1) = (nc_rows[0], nc_rows[1]);
        swap_b32(ctrl, MlDsaCtrlColumns::NC_VALUE, r0, r1);
        swap_b32(ctrl, MlDsaCtrlColumns::NC_IDX, r0, r1);
    });

    assert!(
        detected,
        "NormCheck row swap must be caught by first-row NC_IDX=0 constraint"
    );
}

// highbits bus is a multiset check.
// First-row constraint anchors HB_IDX=0;
// swapping the first two rows puts idx≠0
// on the first HB row.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_highbits_tamper() {
    let detected = run_tampered_mldsa_65(|traces| {
        let ctrl = &mut traces[0];
        let hb_rows = rows_with_bit(ctrl, MlDsaCtrlColumns::HB_SELECTOR);

        assert!(hb_rows.len() >= 2);

        let (r0, r1) = (hb_rows[0], hb_rows[1]);
        swap_b32(ctrl, MlDsaCtrlColumns::HB_R, r0, r1);
        swap_b32(ctrl, MlDsaCtrlColumns::HB_R1, r0, r1);
        swap_b32(ctrl, MlDsaCtrlColumns::HB_R0, r0, r1);
        swap_b32(ctrl, MlDsaCtrlColumns::HB_IDX, r0, r1);
    });

    assert!(
        detected,
        "HighBits row swap must be caught by first-row HB_IDX=0 constraint"
    );
}

// Clear all IO_SELECTOR bits on ctrl.
// CPU-side ml_dsa_data bus still has
// entries, product mismatch.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_io_phase_skip() {
    let detected = run_tampered_mldsa_65(|traces| {
        let ctrl = &mut traces[0];
        let io_rows = rows_with_bit(ctrl, MlDsaCtrlColumns::IO_SELECTOR);

        for &r in &io_rows {
            set_bit_val(ctrl, MlDsaCtrlColumns::IO_SELECTOR, r, Bit::ZERO);
        }
    });

    assert!(
        detected,
        "IO phase skip must be caught by ml_dsa_data bus mismatch"
    );
}

// Clear CMP_SELECTOR, all hash comparison
// constraints become vacuous (gated on CMP).
// No constraint forces CMP to fire.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_cmp_row_removal() {
    let detected = run_tampered_mldsa_65(|traces| {
        let ctrl = &mut traces[0];
        let cmp_row = first_row_with_bit(ctrl, MlDsaCtrlColumns::CMP_SELECTOR);
        set_bit_val(ctrl, MlDsaCtrlColumns::CMP_SELECTOR, cmp_row, Bit::ZERO);
    });

    assert!(
        detected,
        "CMP row removal must be caught by CTILDE_REF_BIND_SEEN end-of-trace constraint"
    );
}

// BIND_SEEN flags are monotonic 0->1.
// Clearing one after it was set means
// a Keccak snapshot was skipped.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_bind_seen_bypass() {
    let detected = run_tampered_mldsa_65(|traces| {
        let ctrl = &mut traces[0];
        let seen_rows = rows_with_bit(ctrl, MlDsaCtrlColumns::CTILDE_PRIME_BIND_SEEN);

        if !seen_rows.is_empty() {
            set_bit_val(
                ctrl,
                MlDsaCtrlColumns::CTILDE_PRIME_BIND_SEEN,
                seen_rows[0],
                Bit::ZERO,
            );
        } else {
            // Never populated,
            // gap confirmed.
        }
    });

    assert!(
        detected,
        "BIND_SEEN bypass must be caught by monotonicity + end-of-trace constraint"
    );
}

// Swap preserves the (NC_VALUE, NC_IDX) multiset.
// No NC_IDX continuity constraint on ctrl.
// NormCheck chiplet untouched, its multiset unchanged.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_z_coefficient_swap() {
    let detected = run_tampered_mldsa_65(|traces| {
        let ctrl = &mut traces[0];
        let nc_rows = rows_with_bit(ctrl, MlDsaCtrlColumns::NC_SELECTOR);

        assert!(nc_rows.len() >= 4);

        let (r0, r1) = (nc_rows[2], nc_rows[3]);
        swap_b32(ctrl, MlDsaCtrlColumns::NC_VALUE, r0, r1);
        swap_b32(ctrl, MlDsaCtrlColumns::NC_IDX, r0, r1);
    });

    assert!(
        detected,
        "z coefficient swap must be caught by NC-RAM binding"
    );
}

// Same multiset-preserving swap on HighBits dispatch.
// No HB_IDX continuity constraint on ctrl.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_hb_decomposition_swap() {
    let detected = run_tampered_mldsa_65(|traces| {
        let ctrl = &mut traces[0];
        let hb_rows = rows_with_bit(ctrl, MlDsaCtrlColumns::HB_SELECTOR);

        assert!(hb_rows.len() >= 4);

        let (r0, r1) = (hb_rows[2], hb_rows[3]);
        swap_b32(ctrl, MlDsaCtrlColumns::HB_R, r0, r1);
        swap_b32(ctrl, MlDsaCtrlColumns::HB_R1, r0, r1);
        swap_b32(ctrl, MlDsaCtrlColumns::HB_R0, r0, r1);
        swap_b32(ctrl, MlDsaCtrlColumns::HB_IDX, r0, r1);
    });

    assert!(
        detected,
        "HB decomposition swap must be caught by HB-RAM binding"
    );
}

// No hint weight accumulator column.
// Per-row h_bit binding only.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_hint_weight_unconstrained() {
    let detected = run_tampered_mldsa_65(|traces| {
        let ctrl = &mut traces[0];
        let hb_rows = rows_with_bit(ctrl, MlDsaCtrlColumns::HB_SELECTOR);
        assert!(hb_rows.len() >= 2);

        set_bit_val(ctrl, MlDsaCtrlColumns::HB_H_BIT, hb_rows[0], Bit::ONE);
    });

    assert!(detected, "h_bit flip must be caught by UseHint bus");
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_h_hint_flip() {
    let detected = run_tampered_mldsa_65(|traces| {
        let ctrl = &mut traces[0];
        let hb_rows = rows_with_bit(ctrl, MlDsaCtrlColumns::HB_SELECTOR);

        assert!(!hb_rows.is_empty());

        let row = hb_rows[0];
        let current = ctrl.columns[MlDsaCtrlColumns::HB_H_BIT]
            .as_bit_slice()
            .unwrap()[row];

        let flipped = if current == Bit::ONE {
            Bit::ZERO
        } else {
            Bit::ONE
        };

        set_bit_val(ctrl, MlDsaCtrlColumns::HB_H_BIT, row, flipped);
    });

    assert!(detected, "h_bit flip must be caught by HB bus mismatch");
}

// NTT boundary binding
// catches t1 substitution.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_t1_ram_substitution() {
    let detected = run_tampered_mldsa_65(|traces| {
        let ctrl = &mut traces[0];

        let bound_in = rows_with_bit(ctrl, MlDsaCtrlColumns::BOUND_IN_SEL);
        assert!(!bound_in.is_empty());

        // Flip a t1 coefficient at a boundary
        // input row. NTT boundary binding
        // pins RAM value to NTT input.
        flip_b32(ctrl, MlDsaCtrlColumns::RAM_VAL_PACKED, bound_in[0], 0x1);
    });

    assert!(
        detected,
        "t1 RAM substitution must be caught by NTT boundary binding"
    );
}

// Fully consistent tamper:
// flip ω+1 h_bits from 0->1 on BOTH ctrl
// and chiplet with correct carry chain
// witness. Caught by bus_w1_prime linkage:
// chiplet constraint ties bus B32 to
// packed w1_bits via tower recomposition.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_hint_weight_overflow() {
    const MLDSA_Q: u32 = 8380417;
    const GAMMA2: u32 = 261888;
    const DIVISOR: u32 = 2 * GAMMA2;
    const M: u32 = (MLDSA_Q - 1) / DIVISOR;
    const OMEGA: usize = 55;

    let ly = HighBitsLayout::compute(MLDSA_Q, 23, DIVISOR);

    let hb_chiplet_idx = 5;
    let phy_bus_h = ly.num_packed_b32_cols + 4;
    let phy_bus_w1 = ly.num_packed_b32_cols + 5;
    let chain_width = ly.w1_width.saturating_sub(1);

    let detected = run_tampered_mldsa_65(|traces| {
        let mut flipped = 0usize;

        let hb_rows: Vec<usize> = {
            let ctrl = &traces[0];
            rows_with_bit(ctrl, MlDsaCtrlColumns::HB_SELECTOR)
        };

        for (chip_row, &ctrl_row) in hb_rows.iter().enumerate() {
            if flipped > OMEGA {
                break;
            }

            let h_current = traces[0].columns[MlDsaCtrlColumns::HB_H_BIT]
                .as_bit_slice()
                .unwrap()[ctrl_row];

            if h_current == Bit::ONE {
                continue;
            }

            let r = read_b32(&traces[0], MlDsaCtrlColumns::HB_R, ctrl_row);
            let r1_u = r / DIVISOR;
            let r0_u = r % DIVISOR;
            let is_neg = r0_u > GAMMA2;
            let is_qm1 = r1_u + is_neg as u32 == M;
            let r1c = if is_qm1 { 0 } else { r1_u + is_neg as u32 };
            let s_dir = r0_u > 0 && !is_neg;

            let new_w1 = if s_dir {
                (r1c + 1) % M
            } else {
                (r1c + M - 1) % M
            };

            // Ctrl:
            // flip h_bit, set new w1_prime.
            {
                let ctrl = &mut traces[0];
                set_bit_val(ctrl, MlDsaCtrlColumns::HB_H_BIT, ctrl_row, Bit::ONE);

                match &mut ctrl.columns[MlDsaCtrlColumns::HB_W1_PRIME] {
                    TraceColumn::B32(data) => {
                        data[ctrl_row] = Flat::from_raw(Block32(new_w1));
                    }
                    _ => panic!("expected B32"),
                }
            }

            // Chiplet:
            // flip bus_h_bit, set bus_w1_prime,
            // update packed h_bit and w1_bits.
            {
                let chip = &mut traces[hb_chiplet_idx];

                match &mut chip.columns[phy_bus_h] {
                    TraceColumn::B32(data) => {
                        data[chip_row] = Flat::from_raw(Block32(1));
                    }
                    _ => panic!("expected B32"),
                }

                match &mut chip.columns[phy_bus_w1] {
                    TraceColumn::B32(data) => {
                        data[chip_row] = Flat::from_raw(Block32(new_w1));
                    }
                    _ => panic!("expected B32"),
                }

                // Packed bits:
                // flip h_bit, update w1_bits.
                let h_word = ly.h_bit / 32;
                let h_bit_pos = ly.h_bit % 32;

                match &mut chip.columns[h_word] {
                    TraceColumn::B32(data) => {
                        let mut w = data[chip_row].to_tower().0;
                        w |= 1 << h_bit_pos;

                        data[chip_row] = Flat::from_raw(Block32(w));
                    }
                    _ => panic!("expected B32"),
                }

                for k in 0..ly.w1_width {
                    let bit = (new_w1 >> k) & 1;
                    let w_idx = (ly.w1_bits + k) / 32;
                    let b_idx = (ly.w1_bits + k) % 32;

                    match &mut chip.columns[w_idx] {
                        TraceColumn::B32(data) => {
                            let mut w = data[chip_row].to_tower().0;
                            w = (w & !(1 << b_idx)) | (bit << b_idx);

                            data[chip_row] = Flat::from_raw(Block32(w));
                        }
                        _ => panic!("expected B32"),
                    }
                }

                // Carry chain:
                // recompute for h=1. Matches
                // constraint uh_c0/uh_ch.
                if chain_width > 0 {
                    let mut prev = if s_dir { r1c & 1 } else { 1 ^ (r1c & 1) };
                    for k in 0..chain_width {
                        let w_idx = (ly.chain + k) / 32;
                        let b_idx = (ly.chain + k) % 32;

                        match &mut chip.columns[w_idx] {
                            TraceColumn::B32(data) => {
                                let mut w = data[chip_row].to_tower().0;
                                w = (w & !(1 << b_idx)) | (prev << b_idx);

                                data[chip_row] = Flat::from_raw(Block32(w));
                            }
                            _ => panic!("expected B32"),
                        }

                        if k + 1 < chain_width {
                            let next_bit = (r1c >> (k + 1)) & 1;
                            let sel = if s_dir { next_bit } else { 1 ^ next_bit };

                            prev &= sel;
                        }
                    }
                }
            }

            flipped += 1;
        }

        assert!(flipped > OMEGA, "not enough h=0 rows to flip: {flipped}");
    });

    assert!(
        detected,
        "hint weight overflow must be caught by bus_w1_prime linkage"
    );
}

// NTT dispatch swap with distinct B values.
// Detected via ntt_b + ram_val_packed binding.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_ntt_dispatch_swap() {
    let detected = run_tampered_mldsa_65(|traces| {
        let ctrl = &mut traces[0];
        let ntt_rows = rows_with_bit(ctrl, MlDsaCtrlColumns::NTT_SELECTOR);
        let bound_in = rows_with_bit(ctrl, MlDsaCtrlColumns::BOUND_IN_SEL);

        let non_bound: Vec<usize> = ntt_rows
            .into_iter()
            .filter(|r| !bound_in.contains(r))
            .collect();

        assert!(non_bound.len() >= 200);

        let get_inst = |trace: &ColumnTrace, row: usize| -> u32 {
            match &trace.columns[MlDsaCtrlColumns::NTT_INSTANCE] {
                TraceColumn::B32(data) => {
                    let v: Block32 = data[row].to_tower();
                    v.0
                }
                _ => panic!("expected B32"),
            }
        };

        let mut r0 = 0usize;
        let mut r1 = 0usize;
        let mut found = false;

        'outer: for i in 0..non_bound.len() {
            let bi = read_b32(ctrl, MlDsaCtrlColumns::NTT_B, non_bound[i]);
            let ii = get_inst(ctrl, non_bound[i]);

            for j in (i + 1)..non_bound.len() {
                let bj = read_b32(ctrl, MlDsaCtrlColumns::NTT_B, non_bound[j]);
                let ij = get_inst(ctrl, non_bound[j]);

                if ii == ij && bi != bj {
                    r0 = non_bound[i];
                    r1 = non_bound[j];

                    found = true;

                    break 'outer;
                }
            }
        }

        assert!(found, "must find same-instance rows with different B");

        let b0 = read_b32(ctrl, MlDsaCtrlColumns::NTT_B, r0);
        let b1 = read_b32(ctrl, MlDsaCtrlColumns::NTT_B, r1);

        assert_ne!(b0, b1, "pre-swap: B values must differ to violate binding");

        swap_b32(ctrl, MlDsaCtrlColumns::NTT_A, r0, r1);
        swap_b32(ctrl, MlDsaCtrlColumns::NTT_B, r0, r1);
        swap_b32(ctrl, MlDsaCtrlColumns::NTT_A_OUT, r0, r1);
        swap_b32(ctrl, MlDsaCtrlColumns::NTT_B_OUT, r0, r1);
        swap_b32(ctrl, MlDsaCtrlColumns::NTT_LAYER, r0, r1);
        swap_b32(ctrl, MlDsaCtrlColumns::NTT_BUTTERFLY, r0, r1);
        swap_b32(ctrl, MlDsaCtrlColumns::NTT_INSTANCE, r0, r1);
    });

    assert!(
        detected,
        "NTT dispatch swap with different B values must be detected"
    );
}
