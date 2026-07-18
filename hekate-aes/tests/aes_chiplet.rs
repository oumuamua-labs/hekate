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

use hekate_aes::{
    Aes128Chiplet, Aes128Columns, Aes256Chiplet, Aes256Columns, AesRound128Air, AesRound256Air,
    CpuAes128Columns, CpuAes128Unit, CpuAes256Columns, CpuAes256Unit, PhysAes128Columns,
    PhysAes256Columns,
    trace::{Aes128Call, Aes256Call, expand_key, expand_key_256},
};
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder, TraceColumn};
use hekate_crypto::DefaultHasher;
use hekate_crypto::transcript::Transcript;
use hekate_math::{Bit, Block8, Block64, Block128, Flat, TowerField};
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::permutation::PermutationCheckSpec;
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_scribble::{MutationKind, ScribbleConfig, assert_all_caught_all_targets};
use hekate_sdk::preflight;
use hekate_verifier::HekateVerifier;
use rand::{TryRngCore, rngs::OsRng};

type F = Block128;
type H = DefaultHasher;

/// FIPS 197 Appendix B test vector.
#[rustfmt::skip]
const FIPS_KEY: [u8; 16] = [
    0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
    0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
];

#[rustfmt::skip]
const FIPS_PLAIN: [u8; 16] = [
    0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d,
    0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34,
];

#[rustfmt::skip]
const FIPS_CIPHER: [u8; 16] = [
    0x39, 0x25, 0x84, 0x1d, 0x02, 0xdc, 0x09, 0xfb,
    0xdc, 0x11, 0x85, 0x97, 0x19, 0x6a, 0x0b, 0x32,
];

fn fips128_call() -> Aes128Call {
    Aes128Call {
        key: FIPS_KEY,
        plaintext: FIPS_PLAIN,
        round_keys: expand_key(&FIPS_KEY),
    }
}

// =================================================================
// AES-128 Test Program
// =================================================================

const CPU128_ROWS: usize = 4;

#[derive(Clone)]
struct Aes128TestProgram {
    aes: Aes128Chiplet<F>,
}

impl Air<F> for Aes128TestProgram {
    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(CpuAes128Columns::build_layout)
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![
            (
                AesRound128Air::LINK_BUS_ID.into(),
                CpuAes128Unit::linking_spec(),
            ),
            (
                AesRound128Air::KEY_BUS_ID.into(),
                CpuAes128Unit::key_linking_spec(),
            ),
        ]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(CpuAes128Columns::SELECTOR));
        cs.assert_boolean(cs.col(CpuAes128Columns::KEY_SELECTOR));

        cs.build()
    }
}

impl Program<F> for Aes128TestProgram {
    fn num_public_inputs(&self) -> usize {
        0
    }

    fn chiplet_defs(&self) -> errors::Result<Vec<hekate_program::chiplet::ChipletDef<F>>> {
        self.aes.composite().flatten_defs()
    }
}

/// CPU trace:
/// row 0 = plaintext XOR K0 + key (input)
/// row 1 = ciphertext (output)
fn build_cpu128_trace(call: &Aes128Call, ciphertext: &[u8; 16]) -> ColumnTrace {
    let num_vars = CPU128_ROWS.trailing_zeros() as usize;

    let mut tb = TraceBuilder::new(&CpuAes128Columns::build_layout(), num_vars).unwrap();

    for (j, &ct_byte) in ciphertext.iter().enumerate() {
        let whitened = call.plaintext[j] ^ call.round_keys[0][j];
        tb.set_b8(CpuAes128Columns::DATA + j, 0, Block8(whitened))
            .unwrap();
        tb.set_b8(CpuAes128Columns::DATA + j, 1, Block8(ct_byte))
            .unwrap();
    }

    tb.set_bit(CpuAes128Columns::SELECTOR, 0, Bit::ONE).unwrap();
    tb.set_bit(CpuAes128Columns::SELECTOR, 1, Bit::ONE).unwrap();

    for (j, &kb) in call.key.iter().enumerate() {
        tb.set_b8(CpuAes128Columns::KEY + j, 0, Block8(kb)).unwrap();
    }

    tb.set_bit(CpuAes128Columns::KEY_SELECTOR, 0, Bit::ONE)
        .unwrap();

    tb.build()
}

fn make_128_program() -> Aes128TestProgram {
    Aes128TestProgram {
        aes: Aes128Chiplet::new(16, 256).unwrap(),
    }
}

// =================================================================
// AES-128 Prove + Verify
// =================================================================

fn prove_and_verify_128(
    air: &Aes128TestProgram,
    cpu_trace: ColumnTrace,
    chiplet_traces: Vec<ColumnTrace>,
) -> Result<bool, String> {
    let instance = ProgramInstance::new(CPU128_ROWS, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(chiplet_traces);

    let report = preflight(air, &instance, &witness).map_err(|e| format!("preflight: {e:?}"))?;

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
        sumcheck_blinding_factor: 2,
        ..Config::dev()
    };

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    let proof = prove(
        b"AES_E2E",
        air,
        &instance,
        &witness,
        &config,
        blinding_seed,
        None,
    )
    .map_err(|e| format!("prover: {e:?}"))?;

    let mut vt = Transcript::<H>::new(b"AES_E2E");
    HekateVerifier::<F, H>::verify(air, &instance, &proof, &mut vt, &config)
        .map_err(|e| format!("verifier: {e:?}"))
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn aes_128_e2e() {
    let air = make_128_program();
    let call = fips128_call();

    let chiplet_traces = air
        .aes
        .generate_traces(core::slice::from_ref(&call))
        .unwrap();
    let cpu_trace = build_cpu128_trace(&call, &FIPS_CIPHER);

    match prove_and_verify_128(&air, cpu_trace, chiplet_traces) {
        Ok(true) => {}
        Ok(false) => panic!("verifier rejected honest proof"),
        Err(e) => panic!("error: {e}"),
    }
}

// =================================================================
// AES-128 Adversarial Harness
// =================================================================

fn run_tampered_aes128<T>(tamper: T) -> bool
where
    T: FnOnce(&mut [ColumnTrace], &mut ColumnTrace),
{
    let air = make_128_program();
    let call = fips128_call();

    let mut chiplet_traces = air
        .aes
        .generate_traces(core::slice::from_ref(&call))
        .unwrap();
    let mut cpu_trace = build_cpu128_trace(&call, &FIPS_CIPHER);

    tamper(&mut chiplet_traces, &mut cpu_trace);

    let instance = ProgramInstance::new(CPU128_ROWS, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(chiplet_traces);

    let config = Config {
        sumcheck_blinding_factor: 2,
        ..Config::dev()
    };

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    let proof_result = prove(
        b"AES_Adversarial",
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
            let mut vt = Transcript::<H>::new(b"AES_Adversarial");
            let result = HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut vt, &config);
            result.is_err() || !result.unwrap()
        }
    }
}

// =================================================================
// Helpers
// =================================================================

// S-box ROM physical column indices
const ROM_PHYS_INV: usize = 0;
const ROM_PHYS_INPUT: usize = 2;
const ROM_PHYS_OUTPUT: usize = 18;
const ROM_PHYS_Z: usize = 34;
const ROM_PHYS_SELECTOR: usize = 50;

fn flip_b8(trace: &mut ColumnTrace, col: usize, row: usize, mask: u8) {
    match &mut trace.columns[col] {
        TraceColumn::B8(data) => {
            let original = data[row];
            data[row] = Flat::from_raw(Block8(original.to_tower().0 ^ mask));
        }
        _ => panic!("expected B8 column at {col}"),
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

fn set_bit_val(trace: &mut ColumnTrace, col: usize, row: usize, val: Bit) {
    match &mut trace.columns[col] {
        TraceColumn::Bit(data) => data[row] = val,
        _ => panic!("expected Bit column at {col}"),
    }
}

fn rows_with_bit(trace: &ColumnTrace, col: usize) -> Vec<usize> {
    let bits = trace.columns[col].as_bit_slice().unwrap();
    (0..bits.len()).filter(|&r| bits[r] == Bit::ONE).collect()
}

fn copy_b8_block(trace: &mut ColumnTrace, base: usize, len: usize, src: usize, dst: usize) {
    for j in 0..len {
        let value = match &trace.columns[base + j] {
            TraceColumn::B8(data) => data[src],
            _ => panic!("expected B8 column at {}", base + j),
        };

        match &mut trace.columns[base + j] {
            TraceColumn::B8(data) => data[dst] = value,
            _ => unreachable!(),
        }
    }
}

// =================================================================
// AES-128 Exploit Tests
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_sbox_substitution() {
    let detected = run_tampered_aes128(|traces, _| {
        flip_b8(&mut traces[0], Aes128Columns::SBOX_OUT, 0, 0x01);
    });

    assert!(detected, "wrong S-box output must be caught by sbox bus");
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_round_key_swap() {
    let detected = run_tampered_aes128(|traces, _| {
        let aes = &mut traces[0];
        for j in 0..16 {
            match &mut aes.columns[Aes128Columns::ROUND_KEY + j] {
                TraceColumn::B8(data) => data.swap(0, 1),
                _ => panic!("expected B8"),
            }
        }
    });

    assert!(
        detected,
        "round key swap must be caught by round transition constraint"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_mixcol_bypass() {
    let detected = run_tampered_aes128(|traces, _| {
        flip_b8(&mut traces[0], Aes128Columns::STATE_IN, 1, 0x01);
    });

    assert!(detected, "MixColumns output tamper must be caught");
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_shiftrows_bypass() {
    let detected = run_tampered_aes128(|traces, _| {
        let aes = &mut traces[0];
        let (col1, col5) = (Aes128Columns::SBOX_OUT + 1, Aes128Columns::SBOX_OUT + 5);

        let v1 = match &aes.columns[col1] {
            TraceColumn::B8(d) => d[0],
            _ => panic!("expected B8"),
        };
        let v5 = match &aes.columns[col5] {
            TraceColumn::B8(d) => d[0],
            _ => panic!("expected B8"),
        };

        if let TraceColumn::B8(d) = &mut aes.columns[col1] {
            d[0] = v5;
        }
        if let TraceColumn::B8(d) = &mut aes.columns[col5] {
            d[0] = v1;
        }
    });

    assert!(detected, "ShiftRows byte swap must be caught");
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_round_skip() {
    let detected = run_tampered_aes128(|traces, _| {
        set_bit_val(&mut traces[0], Aes128Columns::S_ROUND, 4, Bit::ZERO);
    });

    assert!(detected, "deactivating s_round must be caught");
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_ghost_injection() {
    let detected = run_tampered_aes128(|traces, _| {
        set_bit_val(&mut traces[0], Aes128Columns::S_ROUND, 12, Bit::ONE);
    });

    assert!(detected, "ghost row activation must be caught");
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_io_bus_unbind() {
    let detected = run_tampered_aes128(|traces, _| {
        flip_b8(&mut traces[0], Aes128Columns::STATE_IN, 10, 0x01);
    });

    assert!(
        detected,
        "ciphertext tamper on IO row must be caught by aes_link bus"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_cpu_io_corruption() {
    let detected = run_tampered_aes128(|_, cpu| {
        flip_b8(cpu, CpuAes128Columns::DATA, 0, 0x01);
    });

    assert!(
        detected,
        "CPU-side plaintext tamper must be caught by aes_link bus"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes128_link_duplicate_cpu_request_rejected() {
    let detected = run_tampered_aes128(|_, cpu| {
        copy_b8_block(cpu, CpuAes128Columns::DATA, 16, 0, 2);
        set_bit_val(cpu, CpuAes128Columns::SELECTOR, 2, Bit::ONE);
    });

    assert!(
        detected,
        "duplicate CPU link request without chiplet partner must be caught by aes128_link bus"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes128_key_in_duplicate_cpu_request_rejected() {
    let detected = run_tampered_aes128(|_, cpu| {
        copy_b8_block(cpu, CpuAes128Columns::KEY, 16, 0, 2);
        set_bit_val(cpu, CpuAes128Columns::KEY_SELECTOR, 2, Bit::ONE);
    });

    assert!(
        detected,
        "duplicate CPU key request without chiplet partner must be caught by aes128_key_in bus"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes128_sbox_phantom_active_without_link_rejected() {
    let detected = run_tampered_aes128(|traces, _| {
        let aes = &mut traces[0];
        set_bit_val(aes, Aes128Columns::S_ACTIVE, 15, Bit::ONE);
        set_bit_val(aes, Aes128Columns::S_INPUT, 15, Bit::ZERO);
        set_bit_val(aes, Aes128Columns::S_IN_OUT, 15, Bit::ZERO);
    });

    assert!(
        detected,
        "phantom S_ACTIVE=1 with S_INPUT=0 and S_IN_OUT=0 fires the AES<>SboxRom \
         internal bus on a row that does NOT fire link or key; the SboxRom waiver \
         (\"phantom blocks caught at link+key v3\") must reject this shape"
    );
}

// =================================================================
// Round Counter Exploits
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_round_counter_tamper() {
    let detected = run_tampered_aes128(|traces, _| {
        flip_b8(&mut traces[0], Aes128Columns::ROUND_NUM, 3, 0x01);
    });

    assert!(
        detected,
        "round_num tamper must be caught by doubling constraint"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_round_counter_init() {
    let detected = run_tampered_aes128(|traces, _| {
        flip_b8(&mut traces[0], Aes128Columns::ROUND_NUM, 0, 0x02);
    });

    assert!(
        detected,
        "round_num init tamper must be caught by s_input constraint"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_round_counter_final() {
    let detected = run_tampered_aes128(|traces, _| {
        flip_b8(&mut traces[0], Aes128Columns::ROUND_NUM, 9, 0x01);
    });

    assert!(
        detected,
        "round_num final tamper must be caught by s_final constraint"
    );
}

// =================================================================
// Selector Integrity Exploits
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_s_active_desync() {
    let detected = run_tampered_aes128(|traces, _| {
        set_bit_val(&mut traces[0], Aes128Columns::S_ACTIVE, 3, Bit::ZERO);
    });

    assert!(
        detected,
        "s_active=0 on s_round=1 row must be caught by s_active tie constraint"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_s_input_forgery() {
    let detected = run_tampered_aes128(|traces, _| {
        set_bit_val(&mut traces[0], Aes128Columns::S_INPUT, 3, Bit::ONE);
    });

    assert!(
        detected,
        "s_input=1 on non-input row must be caught by s_input tie constraint"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_mutual_exclusivity() {
    let detected = run_tampered_aes128(|traces, _| {
        set_bit_val(&mut traces[0], Aes128Columns::S_FINAL, 3, Bit::ONE);
    });

    assert!(
        detected,
        "s_round=1 AND s_final=1 must be caught by mutual exclusivity"
    );
}

// =================================================================
// S-box ROM Algebraic Exploits
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_sbox_inv_tamper() {
    let detected = run_tampered_aes128(|traces, _| {
        flip_b64(&mut traces[1], ROM_PHYS_INV, 0, 0x01);
    });

    assert!(
        detected,
        "flipped INV bit must be caught by INPUT*INV+Z=1 constraint"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_sbox_z_forgery() {
    let detected = run_tampered_aes128(|traces, _| {
        let rom = &mut traces[1];
        let active = rows_with_bit(rom, ROM_PHYS_SELECTOR);
        let target = active.iter().find(|&&r| {
            rom.columns[ROM_PHYS_INPUT].as_b8_slice().unwrap()[r]
                .to_tower()
                .0
                != 0
        });

        if let Some(&row) = target {
            set_bit_val(rom, ROM_PHYS_Z, row, Bit::ONE);
        }
    });

    assert!(
        detected,
        "Z=1 on nonzero input must be caught by Z*INPUT=0 constraint"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_sbox_rom_deactivation() {
    let detected = run_tampered_aes128(|traces, _| {
        let rom = &mut traces[1];
        let active = rows_with_bit(rom, ROM_PHYS_SELECTOR);

        set_bit_val(rom, ROM_PHYS_SELECTOR, active[0], Bit::ZERO);
    });

    assert!(
        detected,
        "ROM selector deactivation must be caught by sbox bus mismatch"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_sbox_output_forgery() {
    let detected = run_tampered_aes128(|traces, _| {
        let rom = &mut traces[1];
        let active = rows_with_bit(rom, ROM_PHYS_SELECTOR);

        flip_b8(rom, ROM_PHYS_OUTPUT, active[0], 0xFF);
    });

    assert!(
        detected,
        "ROM output tamper must be caught by affine constraint"
    );
}

// =================================================================
// Ghost Protocol Exploits
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_ghost_protocol_mid_block() {
    let detected = run_tampered_aes128(|traces, _| {
        set_bit_val(&mut traces[0], Aes128Columns::S_ROUND, 4, Bit::ZERO);
        set_bit_val(&mut traces[0], Aes128Columns::S_ACTIVE, 4, Bit::ZERO);
        set_bit_val(&mut traces[1], ROM_PHYS_SELECTOR, 4, Bit::ZERO);
    });

    assert!(
        detected,
        "deactivating mid-block row must be caught by continuity constraint"
    );
}

// =================================================================
// Key Schedule Exploits
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_k0_init_cascade() {
    let detected = run_tampered_aes128(|traces, cpu| {
        // Tamper K0 on BOTH sides so bus passes.
        // Init cascade must catch K1 ≠ expand(tampered_K0).
        flip_b8(&mut traces[0], PhysAes128Columns::P_K0, 0, 0x01);
        flip_b8(cpu, CpuAes128Columns::KEY, 0, 0x01);
    });

    assert!(
        detected,
        "K0 tamper (bus-consistent) must be caught by init cascade: K1 ≠ expand(K0')"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_k0_bus_unbind() {
    let detected = run_tampered_aes128(|_traces, cpu| {
        flip_b8(cpu, CpuAes128Columns::KEY, 0, 0x01);
    });

    assert!(
        detected,
        "CPU-side key tamper must be caught by aes_key_in bus mismatch"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_ks_sub_tamper() {
    let detected = run_tampered_aes128(|traces, _| {
        flip_b8(&mut traces[0], PhysAes128Columns::P_KS_SUB, 0, 0x01);
    });

    assert!(
        detected,
        "KS_SUB tamper must be caught by S-box inversion or cascade constraint"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_ks_inv_tamper() {
    let detected = run_tampered_aes128(|traces, _| {
        flip_b8(&mut traces[0], PhysAes128Columns::P_KS_INV, 0, 0x01);
    });

    assert!(
        detected,
        "KS_INV tamper must be caught by inversion constraint"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_k0_sub_tamper() {
    let detected = run_tampered_aes128(|traces, _| {
        flip_b8(&mut traces[0], PhysAes128Columns::P_K0_SUB, 0, 0x01);
    });

    assert!(
        detected,
        "K0_SUB tamper must be caught by init S-box or init cascade constraint"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_k0_inv_tamper() {
    let detected = run_tampered_aes128(|traces, _| {
        flip_b8(&mut traces[0], PhysAes128Columns::P_K0_INV, 0, 0x01);
    });

    assert!(
        detected,
        "K0_INV tamper must be caught by init inversion constraint"
    );
}

// =================================================================
// AES-256 Test Program
// =================================================================

/// FIPS 197 Appendix C.3 test vector.
#[rustfmt::skip]
const FIPS256_KEY: [u8; 32] = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
];

#[rustfmt::skip]
const FIPS256_PLAIN: [u8; 16] = [
    0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
    0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
];

#[rustfmt::skip]
const FIPS256_CIPHER: [u8; 16] = [
    0x8e, 0xa2, 0xb7, 0xca, 0x51, 0x67, 0x45, 0xbf,
    0xea, 0xfc, 0x49, 0x90, 0x4b, 0x49, 0x60, 0x89,
];

const CPU256_ROWS: usize = 4;

fn fips256_call() -> Aes256Call {
    Aes256Call {
        key: FIPS256_KEY,
        plaintext: FIPS256_PLAIN,
        round_keys: expand_key_256(&FIPS256_KEY),
    }
}

#[derive(Clone)]
struct Aes256TestProgram {
    aes: Aes256Chiplet<F>,
}

impl Air<F> for Aes256TestProgram {
    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(CpuAes256Columns::build_layout)
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![
            (
                AesRound256Air::LINK_BUS_ID.into(),
                CpuAes256Unit::linking_spec(),
            ),
            (
                AesRound256Air::KEY_BUS_ID.into(),
                CpuAes256Unit::key_linking_spec(),
            ),
        ]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(CpuAes256Columns::SELECTOR));
        cs.assert_boolean(cs.col(CpuAes256Columns::KEY_SELECTOR));

        cs.build()
    }
}

impl Program<F> for Aes256TestProgram {
    fn num_public_inputs(&self) -> usize {
        0
    }

    fn chiplet_defs(&self) -> errors::Result<Vec<hekate_program::chiplet::ChipletDef<F>>> {
        self.aes.composite().flatten_defs()
    }
}

fn build_cpu256_trace(call: &Aes256Call, ciphertext: &[u8; 16]) -> ColumnTrace {
    let num_vars = CPU256_ROWS.trailing_zeros() as usize;

    let mut tb = TraceBuilder::new(&CpuAes256Columns::build_layout(), num_vars).unwrap();

    for (j, &ct_byte) in ciphertext.iter().enumerate() {
        let whitened = call.plaintext[j] ^ call.round_keys[0][j];
        tb.set_b8(CpuAes256Columns::DATA + j, 0, Block8(whitened))
            .unwrap();
        tb.set_b8(CpuAes256Columns::DATA + j, 1, Block8(ct_byte))
            .unwrap();
    }

    tb.set_bit(CpuAes256Columns::SELECTOR, 0, Bit::ONE).unwrap();
    tb.set_bit(CpuAes256Columns::SELECTOR, 1, Bit::ONE).unwrap();

    for (j, &kb) in call.key.iter().enumerate() {
        tb.set_b8(CpuAes256Columns::KEY + j, 0, Block8(kb)).unwrap();
    }

    tb.set_bit(CpuAes256Columns::KEY_SELECTOR, 0, Bit::ONE)
        .unwrap();

    tb.build()
}

fn make_256_program() -> Aes256TestProgram {
    Aes256TestProgram {
        aes: Aes256Chiplet::new(16, 256).unwrap(),
    }
}

fn prove_and_verify_256(
    air: &Aes256TestProgram,
    cpu_trace: ColumnTrace,
    chiplet_traces: Vec<ColumnTrace>,
) -> Result<bool, String> {
    let instance = ProgramInstance::new(CPU256_ROWS, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(chiplet_traces);

    let report = preflight(air, &instance, &witness).map_err(|e| format!("preflight: {e:?}"))?;

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
        sumcheck_blinding_factor: 2,
        ..Config::dev()
    };

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    let proof = prove(
        b"AES256_E2E",
        air,
        &instance,
        &witness,
        &config,
        blinding_seed,
        None,
    )
    .map_err(|e| format!("prover: {e:?}"))?;

    let mut vt = Transcript::<H>::new(b"AES256_E2E");
    HekateVerifier::<F, H>::verify(air, &instance, &proof, &mut vt, &config)
        .map_err(|e| format!("verifier: {e:?}"))
}

// =================================================================
// AES-256 Prove + Verify
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn aes_256_e2e() {
    let air = make_256_program();
    let call = fips256_call();

    let chiplet_traces = air
        .aes
        .generate_traces(core::slice::from_ref(&call))
        .unwrap();
    let cpu_trace = build_cpu256_trace(&call, &FIPS256_CIPHER);

    match prove_and_verify_256(&air, cpu_trace, chiplet_traces) {
        Ok(true) => {}
        Ok(false) => panic!("verifier rejected honest proof"),
        Err(e) => panic!("error: {e}"),
    }
}

// =================================================================
// AES-256 Adversarial Harness
// =================================================================

fn run_tampered_aes256<T>(tamper: T) -> bool
where
    T: FnOnce(&mut [ColumnTrace], &mut ColumnTrace),
{
    let air = make_256_program();
    let call = fips256_call();

    let mut chiplet_traces = air
        .aes
        .generate_traces(core::slice::from_ref(&call))
        .unwrap();
    let mut cpu_trace = build_cpu256_trace(&call, &FIPS256_CIPHER);

    tamper(&mut chiplet_traces, &mut cpu_trace);

    let instance = ProgramInstance::new(CPU256_ROWS, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(chiplet_traces);

    let config = Config {
        sumcheck_blinding_factor: 2,
        ..Config::dev()
    };

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    let proof_result = prove(
        b"AES256_Adversarial",
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
            let mut vt = Transcript::<H>::new(b"AES256_Adversarial");
            let result = HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut vt, &config);
            result.is_err() || !result.unwrap()
        }
    }
}

// =================================================================
// AES-256 Exploit Tests
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_sbox_substitution() {
    let detected = run_tampered_aes256(|traces, _| {
        flip_b8(&mut traces[0], Aes256Columns::SBOX_OUT, 0, 0x01);
    });

    assert!(detected, "wrong S-box output must be caught by sbox bus");
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_round_key_swap() {
    let detected = run_tampered_aes256(|traces, _| {
        let aes = &mut traces[0];
        for j in 0..16 {
            match &mut aes.columns[Aes256Columns::ROUND_KEY + j] {
                TraceColumn::B8(data) => data.swap(0, 1),
                _ => panic!("expected B8"),
            }
        }
    });

    assert!(
        detected,
        "round key swap must be caught by round transition constraint"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_mixcol_bypass() {
    let detected = run_tampered_aes256(|traces, _| {
        flip_b8(&mut traces[0], Aes256Columns::STATE_IN, 1, 0x01);
    });

    assert!(detected, "MixColumns output tamper must be caught");
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_round_skip() {
    let detected = run_tampered_aes256(|traces, _| {
        set_bit_val(&mut traces[0], Aes256Columns::S_ROUND, 4, Bit::ZERO);
    });

    assert!(detected, "deactivating s_round must be caught");
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_ghost_injection() {
    let detected = run_tampered_aes256(|traces, _| {
        set_bit_val(&mut traces[0], Aes256Columns::S_ROUND, 15, Bit::ONE);
    });

    assert!(detected, "ghost row activation must be caught");
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_io_bus_unbind() {
    let detected = run_tampered_aes256(|traces, _| {
        flip_b8(&mut traces[0], Aes256Columns::STATE_IN, 14, 0x01);
    });

    assert!(
        detected,
        "ciphertext tamper on IO row must be caught by aes_link bus"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_cpu_io_corruption() {
    let detected = run_tampered_aes256(|_, cpu| {
        flip_b8(cpu, CpuAes256Columns::DATA, 0, 0x01);
    });

    assert!(
        detected,
        "CPU-side plaintext tamper must be caught by aes_link bus"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_link_duplicate_cpu_request_rejected() {
    let detected = run_tampered_aes256(|_, cpu| {
        copy_b8_block(cpu, CpuAes256Columns::DATA, 16, 0, 2);
        set_bit_val(cpu, CpuAes256Columns::SELECTOR, 2, Bit::ONE);
    });

    assert!(
        detected,
        "duplicate CPU link request without chiplet partner must be caught by aes256_link bus"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_key_in_duplicate_cpu_request_rejected() {
    let detected = run_tampered_aes256(|_, cpu| {
        copy_b8_block(cpu, CpuAes256Columns::KEY, 32, 0, 2);
        set_bit_val(cpu, CpuAes256Columns::KEY_SELECTOR, 2, Bit::ONE);
    });

    assert!(
        detected,
        "duplicate CPU key request without chiplet partner must be caught by aes256_key_in bus"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_sbox_phantom_active_without_link_rejected() {
    let detected = run_tampered_aes256(|traces, _| {
        let aes = &mut traces[0];
        set_bit_val(aes, Aes256Columns::S_ACTIVE, 15, Bit::ONE);
        set_bit_val(aes, Aes256Columns::S_INPUT, 15, Bit::ZERO);
        set_bit_val(aes, Aes256Columns::S_IN_OUT, 15, Bit::ZERO);
    });

    assert!(
        detected,
        "phantom S_ACTIVE=1 with S_INPUT=0 and S_IN_OUT=0 fires the AES<>SboxRom \
         internal bus on a row that does NOT fire link or key; the SboxRom waiver \
         (\"phantom blocks caught at link+key v3\") must reject this shape"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_round_counter_tamper() {
    let detected = run_tampered_aes256(|traces, _| {
        flip_b8(&mut traces[0], Aes256Columns::ROUND_NUM, 3, 0x01);
    });

    assert!(
        detected,
        "round_num tamper must be caught by doubling constraint"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_round_counter_init() {
    let detected = run_tampered_aes256(|traces, _| {
        flip_b8(&mut traces[0], Aes256Columns::ROUND_NUM, 0, 0x02);
    });

    assert!(detected, "round_num init tamper must be caught");
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_round_counter_final() {
    let detected = run_tampered_aes256(|traces, _| {
        flip_b8(&mut traces[0], Aes256Columns::ROUND_NUM, 13, 0x01);
    });

    assert!(
        detected,
        "round_num final tamper must be caught by s_final constraint"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_s_active_desync() {
    let detected = run_tampered_aes256(|traces, _| {
        set_bit_val(&mut traces[0], Aes256Columns::S_ACTIVE, 3, Bit::ZERO);
    });

    assert!(detected, "s_active=0 on s_round=1 row must be caught");
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_s_input_forgery() {
    let detected = run_tampered_aes256(|traces, _| {
        set_bit_val(&mut traces[0], Aes256Columns::S_INPUT, 3, Bit::ONE);
    });

    assert!(detected, "s_input=1 on non-input row must be caught");
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_mutual_exclusivity() {
    let detected = run_tampered_aes256(|traces, _| {
        set_bit_val(&mut traces[0], Aes256Columns::S_FINAL, 3, Bit::ONE);
    });

    assert!(detected, "s_round=1 AND s_final=1 must be caught");
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_ks_sub_tamper() {
    let detected = run_tampered_aes256(|traces, _| {
        flip_b8(&mut traces[0], PhysAes256Columns::P_KS_SUB, 0, 0x01);
    });

    assert!(
        detected,
        "KS_SUB tamper must be caught by S-box inversion or cascade"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_ks_inv_tamper() {
    let detected = run_tampered_aes256(|traces, _| {
        flip_b8(&mut traces[0], PhysAes256Columns::P_KS_INV, 0, 0x01);
    });

    assert!(
        detected,
        "KS_INV tamper must be caught by inversion constraint"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_k0_bus_unbind() {
    let detected = run_tampered_aes256(|_traces, cpu| {
        flip_b8(cpu, CpuAes256Columns::KEY, 0, 0x01);
    });

    assert!(
        detected,
        "CPU-side key tamper must be caught by aes_key_in bus"
    );
}

// =================================================================
// AES-256 S-box ROM / ShiftRows Exploits
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_shiftrows_bypass() {
    let detected = run_tampered_aes256(|traces, _| {
        let aes = &mut traces[0];
        let (col1, col5) = (Aes256Columns::SBOX_OUT + 1, Aes256Columns::SBOX_OUT + 5);

        let v1 = match &aes.columns[col1] {
            TraceColumn::B8(d) => d[0],
            _ => panic!("expected B8"),
        };
        let v5 = match &aes.columns[col5] {
            TraceColumn::B8(d) => d[0],
            _ => panic!("expected B8"),
        };

        if let TraceColumn::B8(d) = &mut aes.columns[col1] {
            d[0] = v5;
        }
        if let TraceColumn::B8(d) = &mut aes.columns[col5] {
            d[0] = v1;
        }
    });

    assert!(detected, "ShiftRows byte swap must be caught");
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_sbox_inv_tamper() {
    let detected = run_tampered_aes256(|traces, _| {
        flip_b64(&mut traces[1], ROM_PHYS_INV, 0, 0x01);
    });

    assert!(
        detected,
        "flipped INV bit must be caught by INPUT*INV+Z=1 constraint"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_sbox_z_forgery() {
    let detected = run_tampered_aes256(|traces, _| {
        let rom = &mut traces[1];
        let active = rows_with_bit(rom, ROM_PHYS_SELECTOR);
        let target = active.iter().find(|&&r| {
            rom.columns[ROM_PHYS_INPUT].as_b8_slice().unwrap()[r]
                .to_tower()
                .0
                != 0
        });

        if let Some(&row) = target {
            set_bit_val(rom, ROM_PHYS_Z, row, Bit::ONE);
        }
    });

    assert!(
        detected,
        "Z=1 on nonzero input must be caught by Z*INPUT=0 constraint"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_sbox_rom_deactivation() {
    let detected = run_tampered_aes256(|traces, _| {
        let rom = &mut traces[1];
        let active = rows_with_bit(rom, ROM_PHYS_SELECTOR);

        set_bit_val(rom, ROM_PHYS_SELECTOR, active[0], Bit::ZERO);
    });

    assert!(
        detected,
        "ROM selector deactivation must be caught by sbox bus mismatch"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_sbox_output_forgery() {
    let detected = run_tampered_aes256(|traces, _| {
        let rom = &mut traces[1];
        let active = rows_with_bit(rom, ROM_PHYS_SELECTOR);

        flip_b8(rom, ROM_PHYS_OUTPUT, active[0], 0xFF);
    });

    assert!(
        detected,
        "ROM output tamper must be caught by affine constraint"
    );
}

// =================================================================
// AES-256–Specific Exploits
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_key_aux_tamper() {
    let detected = run_tampered_aes256(|traces, _| {
        flip_b8(&mut traces[0], PhysAes256Columns::P_KEY_AUX, 1, 0x01);
    });

    assert!(
        detected,
        "KEY_AUX tamper must be caught by word cascade or slide constraint"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_rcon_tamper() {
    let detected = run_tampered_aes256(|traces, _| {
        flip_b8(&mut traces[0], PhysAes256Columns::P_RCON, 0, 0x02);
    });

    assert!(
        detected,
        "RCON tamper must be caught by init or doubling constraint"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_s_even_tamper() {
    let detected = run_tampered_aes256(|traces, _| {
        set_bit_val(&mut traces[0], Aes256Columns::S_EVEN, 1, Bit::ONE);
    });

    assert!(
        detected,
        "S_EVEN tamper must be caught by toggle constraint"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_ks_input_tamper() {
    let detected = run_tampered_aes256(|traces, _| {
        flip_b8(&mut traces[0], PhysAes256Columns::P_KS_INPUT, 1, 0x01);
    });

    assert!(
        detected,
        "KS_INPUT tamper must be caught by source binding constraint"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_init_rk_tamper() {
    let detected = run_tampered_aes256(|traces, _| {
        flip_b8(&mut traces[0], PhysAes256Columns::P_ROUND_KEY, 0, 0x01);
    });

    assert!(
        detected,
        "init RK tamper must be caught by RK=K0[16..31] constraint"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_init_key_aux_tamper() {
    let detected = run_tampered_aes256(|traces, _| {
        flip_b8(&mut traces[0], PhysAes256Columns::P_KEY_AUX, 0, 0x01);
    });

    assert!(
        detected,
        "init KEY_AUX tamper must be caught by KEY_AUX=K0[0..15] constraint"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_ghost_protocol_mid_block() {
    let detected = run_tampered_aes256(|traces, _| {
        set_bit_val(&mut traces[0], Aes256Columns::S_ROUND, 4, Bit::ZERO);
        set_bit_val(&mut traces[0], Aes256Columns::S_ACTIVE, 4, Bit::ZERO);
        set_bit_val(&mut traces[1], ROM_PHYS_SELECTOR, 4, Bit::ZERO);
    });

    assert!(
        detected,
        "deactivating mid-block row must be caught by continuity constraint"
    );
}

#[test]
fn scribble_aes128_flip_selector_caught() {
    let air = make_128_program();
    let call = fips128_call();

    let chiplet_traces = air
        .aes
        .generate_traces(core::slice::from_ref(&call))
        .unwrap();
    let cpu_trace = build_cpu128_trace(&call, &FIPS_CIPHER);

    let instance = ProgramInstance::new(CPU128_ROWS, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(chiplet_traces);

    assert_all_caught_all_targets(
        &air,
        &instance,
        &witness,
        ScribbleConfig::default()
            .mutations([MutationKind::FlipSelector])
            .cases(64),
    );
}

#[test]
fn scribble_aes256_flip_selector_caught() {
    let air = make_256_program();
    let call = fips256_call();

    let chiplet_traces = air
        .aes
        .generate_traces(core::slice::from_ref(&call))
        .unwrap();
    let cpu_trace = build_cpu256_trace(&call, &FIPS256_CIPHER);

    let instance = ProgramInstance::new(CPU256_ROWS, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(chiplet_traces);

    assert_all_caught_all_targets(
        &air,
        &instance,
        &witness,
        ScribbleConfig::default()
            .mutations([MutationKind::FlipSelector])
            .cases(64),
    );
}
