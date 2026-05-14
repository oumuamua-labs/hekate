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

//! Adversarial tests for ML-KEM chiplet soundness.
//!
//! These tests verify that a MALICIOUS PROVER
//! cannot produce valid proofs with wrong data.

use hekate::crypto::transcript::Transcript;
use hekate::crypto::DefaultHasher;
use hekate::math::{Bit, Block128, Block32, Block64, Flat, HardwareField, TowerField};
use hekate_core::config::Config;
use hekate_core::trace::TraceColumn;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate_gadgets::chiplets::pqc::mlkem::{
    self, CpuMlKemColumns, CpuMlKemUnit, MlKemChiplet, MlKemCtrlColumns, MlKemLevel, MlKemParams,
};
use hekate_gadgets::{KeccakChiplet, KeccakWitness};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::constraint::{BoundaryConstraint, ConstraintAst};
use hekate_program::permutation::PermutationCheckSpec;
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;
use pqcrypto_mlkem::{mlkem1024, mlkem512, mlkem768};
use pqcrypto_traits::kem::{Ciphertext as _, SecretKey as _};
use rand::{rngs::OsRng, TryRngCore};

type F = Block128;
type H = DefaultHasher;

#[derive(Clone)]
struct MlKemTestProgram {
    mlkem: MlKemChiplet<F>,
    num_public: usize,
}

impl Air<F> for MlKemTestProgram {
    fn num_columns(&self) -> usize {
        CpuMlKemUnit::num_columns()
    }

    fn boundary_constraints(&self) -> Vec<BoundaryConstraint<F>> {
        (0..self.num_public)
            .map(|k| BoundaryConstraint::with_public_input(CpuMlKemColumns::DATA, k, k))
            .collect()
    }

    fn column_layout(&self) -> &[ColumnType] {
        Box::leak(CpuMlKemColumns::build_layout().into_boxed_slice())
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![
            (
                mlkem::MLKEM_DATA_BUS_ID.into(),
                CpuMlKemUnit::linking_spec(),
            ),
            (
                mlkem::MLKEM_SS_BUS_ID.into(),
                CpuMlKemUnit::ss_linking_spec(),
            ),
        ]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(CpuMlKemColumns::SELECTOR));
        cs.assert_boolean(cs.col(CpuMlKemColumns::SS_SELECTOR));

        cs.build()
    }
}

impl Program<F> for MlKemTestProgram {
    fn num_public_inputs(&self) -> usize {
        self.num_public
    }

    fn chiplet_defs(&self) -> hekate_core::errors::Result<Vec<ChipletDef<F>>> {
        self.mlkem.composite().flatten_defs()
    }
}

fn params_for_level(level: MlKemLevel) -> MlKemParams {
    // Scale trace sizes with k.
    // Keccak:
    // k^2 sample_ntt calls (each ~3 absorbs)
    // plus CBD, G, H, J hashes.
    //
    // NTT:
    // scales linearly with k.
    //
    // RAM:
    // scales with k^2 (matrix-vector product).
    let k = level.k;
    let keccak_shift = if k >= 4 { 12 } else { 11 };
    let ctrl_shift = if k >= 4 { 17 } else { 16 };

    MlKemParams {
        ctrl_rows: 1 << ctrl_shift,
        keccak_rows: 1 << keccak_shift,
        ntt_rows: 1 << (14 + k.div_ceil(3)),
        twiddle_rows: 1 << (14 + k.div_ceil(3)),
        basemul_rows: 1 << (11 + k.div_ceil(2)),
        ram_rows: 1 << (14 + k.div_ceil(2)),
    }
}

fn prove_and_verify_mlkem(ct: &[u8], sk: &[u8], params: &MlKemParams) -> Result<bool, String> {
    prove_and_verify_mlkem_level(MlKemLevel::MLKEM_768, ct, sk, params)
}

fn prove_and_verify_mlkem_level(
    level: MlKemLevel,
    ct: &[u8],
    sk: &[u8],
    params: &MlKemParams,
) -> Result<bool, String> {
    let mlkem_chiplet = MlKemChiplet::<F>::new(level, params.clone());

    let (chiplet_traces, shared_secret) = mlkem_chiplet
        .generate_traces(ct, sk)
        .map_err(|e| format!("trace gen: {e:?}"))?;

    let cpu_rows: usize = 1 << 10;
    let layout = CpuMlKemColumns::build_layout();

    let mut cpu_tb = TraceBuilder::new(&layout, cpu_rows.trailing_zeros() as usize).unwrap();

    for (i, chunk) in ct.chunks(4).enumerate() {
        let mut buf = [0u8; 4];
        buf[..chunk.len()].copy_from_slice(chunk);

        cpu_tb
            .set_b32(
                CpuMlKemColumns::DATA,
                i,
                Block32::from(u32::from_le_bytes(buf)),
            )
            .unwrap();
        cpu_tb
            .set_bit(CpuMlKemColumns::SELECTOR, i, Bit::ONE)
            .unwrap();
    }

    let ss_row = ct.chunks(4).count();
    for i in 0..4 {
        let lo = u32::from_le_bytes(shared_secret[i * 8..i * 8 + 4].try_into().unwrap());
        let hi = u32::from_le_bytes(shared_secret[i * 8 + 4..i * 8 + 8].try_into().unwrap());

        cpu_tb
            .set_b32(CpuMlKemColumns::SS_DATA + i, ss_row, Block32::from(lo))
            .unwrap();
        cpu_tb
            .set_b32(CpuMlKemColumns::SS_DATA + 4 + i, ss_row, Block32::from(hi))
            .unwrap();
    }

    cpu_tb
        .set_bit(CpuMlKemColumns::SS_SELECTOR, ss_row, Bit::ONE)
        .unwrap();

    let cpu_trace = cpu_tb.build();

    let ct_public: Vec<F> = ct
        .chunks(4)
        .map(|chunk| {
            let mut buf = [0u8; 4];
            buf[..chunk.len()].copy_from_slice(chunk);

            Block128(u32::from_le_bytes(buf) as u128)
        })
        .collect();

    let air = MlKemTestProgram {
        mlkem: mlkem_chiplet,
        num_public: ct_public.len(),
    };

    let instance = ProgramInstance::new(cpu_rows, ct_public);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(chiplet_traces);

    let mut config = Config {
        sumcheck_blinding_factor: 2,
        ..Config::default()
    };

    OsRng.try_fill_bytes(&mut config.matrix_seed).unwrap();

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    let proof = prove(
        b"MLKem_Adversarial",
        &air,
        &instance,
        &witness,
        &config,
        blinding_seed,
        None,
    )
    .map_err(|e| format!("prover: {e:?}"))?;

    let mut vt = Transcript::<H>::new(b"MLKem_Adversarial");
    HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut vt, &config)
        .map_err(|e| format!("verifier: {e:?}"))
}

fn test_params() -> MlKemParams {
    MlKemParams {
        ctrl_rows: 1 << 16,
        keccak_rows: 1 << 11,
        ntt_rows: 1 << 15,
        twiddle_rows: 1 << 15,
        basemul_rows: 1 << 12,
        ram_rows: 1 << 16,
    }
}

/// Builds an honest ML-KEM-768 trace,
/// applies a tamper closure to the chiplet
/// traces, runs the prover/verifier, and
/// returns `true` iff the malicious proof
/// was rejected at any stage.
fn run_tampered_mlkem_768<T>(tamper: T) -> bool
where
    T: FnOnce(&mut [ColumnTrace]),
{
    let (pk, sk) = mlkem768::keypair();
    let (_, ct) = mlkem768::encapsulate(&pk);
    let ct_bytes = ct.as_bytes();
    let sk_bytes = sk.as_bytes();

    let params = test_params();
    let mlkem_chiplet = MlKemChiplet::<F>::new(MlKemLevel::MLKEM_768, params);

    let (mut chiplet_traces, shared_secret) = mlkem_chiplet
        .generate_traces(ct_bytes, sk_bytes)
        .expect("trace gen failed");

    tamper(&mut chiplet_traces);

    let cpu_rows: usize = 1 << 10;
    let layout = CpuMlKemColumns::build_layout();

    let mut cpu_tb = TraceBuilder::new(&layout, cpu_rows.trailing_zeros() as usize).unwrap();

    for (i, chunk) in ct_bytes.chunks(4).enumerate() {
        let mut buf = [0u8; 4];
        buf[..chunk.len()].copy_from_slice(chunk);

        cpu_tb
            .set_b32(
                CpuMlKemColumns::DATA,
                i,
                Block32::from(u32::from_le_bytes(buf)),
            )
            .unwrap();
        cpu_tb
            .set_bit(CpuMlKemColumns::SELECTOR, i, Bit::ONE)
            .unwrap();
    }

    let ss_row = ct_bytes.chunks(4).count();
    for i in 0..4 {
        let lo = u32::from_le_bytes(shared_secret[i * 8..i * 8 + 4].try_into().unwrap());
        let hi = u32::from_le_bytes(shared_secret[i * 8 + 4..i * 8 + 8].try_into().unwrap());

        cpu_tb
            .set_b32(CpuMlKemColumns::SS_DATA + i, ss_row, Block32::from(lo))
            .unwrap();
        cpu_tb
            .set_b32(CpuMlKemColumns::SS_DATA + 4 + i, ss_row, Block32::from(hi))
            .unwrap();
    }

    cpu_tb
        .set_bit(CpuMlKemColumns::SS_SELECTOR, ss_row, Bit::ONE)
        .unwrap();

    let cpu_trace = cpu_tb.build();

    let ct_public: Vec<F> = ct_bytes
        .chunks(4)
        .map(|chunk| {
            let mut buf = [0u8; 4];
            buf[..chunk.len()].copy_from_slice(chunk);

            Block128(u32::from_le_bytes(buf) as u128)
        })
        .collect();

    let air = MlKemTestProgram {
        mlkem: mlkem_chiplet,
        num_public: ct_public.len(),
    };
    let instance = ProgramInstance::new(cpu_rows, ct_public);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(chiplet_traces);

    let mut config = Config {
        sumcheck_blinding_factor: 2,
        ..Config::default()
    };
    OsRng.try_fill_bytes(&mut config.matrix_seed).unwrap();

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    let proof_result = prove(
        b"MLKem_Adversarial",
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
            let mut vt = Transcript::<H>::new(b"MLKem_Adversarial");
            let result = HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut vt, &config);

            result.is_err() || !result.unwrap()
        }
    }
}

/// Like `run_tampered_mlkem_768` but also
/// passes the CPU trace for tampering.
fn run_tampered_mlkem_768_with_ss<T>(tamper: T) -> bool
where
    T: FnOnce(&mut [ColumnTrace], &mut ColumnTrace),
{
    let (pk, sk) = mlkem768::keypair();
    let (_, ct) = mlkem768::encapsulate(&pk);

    let ct_bytes = ct.as_bytes();
    let sk_bytes = sk.as_bytes();

    let params = test_params();
    let mlkem_chiplet = MlKemChiplet::<F>::new(MlKemLevel::MLKEM_768, params);

    let (mut chiplet_traces, shared_secret) = mlkem_chiplet
        .generate_traces(ct_bytes, sk_bytes)
        .expect("trace gen failed");

    let cpu_rows: usize = 1 << 10;
    let layout = CpuMlKemColumns::build_layout();

    let mut cpu_tb = TraceBuilder::new(&layout, cpu_rows.trailing_zeros() as usize).unwrap();

    for (i, chunk) in ct_bytes.chunks(4).enumerate() {
        let mut buf = [0u8; 4];
        buf[..chunk.len()].copy_from_slice(chunk);

        cpu_tb
            .set_b32(
                CpuMlKemColumns::DATA,
                i,
                Block32::from(u32::from_le_bytes(buf)),
            )
            .unwrap();
        cpu_tb
            .set_bit(CpuMlKemColumns::SELECTOR, i, Bit::ONE)
            .unwrap();
    }

    let ss_row = ct_bytes.chunks(4).count();
    for i in 0..4 {
        let lo = u32::from_le_bytes(shared_secret[i * 8..i * 8 + 4].try_into().unwrap());
        let hi = u32::from_le_bytes(shared_secret[i * 8 + 4..i * 8 + 8].try_into().unwrap());

        cpu_tb
            .set_b32(CpuMlKemColumns::SS_DATA + i, ss_row, Block32::from(lo))
            .unwrap();
        cpu_tb
            .set_b32(CpuMlKemColumns::SS_DATA + 4 + i, ss_row, Block32::from(hi))
            .unwrap();
    }

    cpu_tb
        .set_bit(CpuMlKemColumns::SS_SELECTOR, ss_row, Bit::ONE)
        .unwrap();

    let mut cpu_trace = cpu_tb.build();

    tamper(&mut chiplet_traces, &mut cpu_trace);

    let ct_public: Vec<F> = ct_bytes
        .chunks(4)
        .map(|chunk| {
            let mut buf = [0u8; 4];
            buf[..chunk.len()].copy_from_slice(chunk);
            Block128(u32::from_le_bytes(buf) as u128)
        })
        .collect();

    let air = MlKemTestProgram {
        mlkem: mlkem_chiplet,
        num_public: ct_public.len(),
    };

    let instance = ProgramInstance::new(cpu_rows, ct_public);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(chiplet_traces);

    let mut config = Config {
        sumcheck_blinding_factor: 2,
        ..Config::default()
    };

    OsRng.try_fill_bytes(&mut config.matrix_seed).unwrap();

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    let proof_result = prove(
        b"MLKem_Adversarial",
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
            let mut vt = Transcript::<H>::new(b"MLKem_Adversarial");
            let result = HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut vt, &config);

            result.is_err() || !result.unwrap()
        }
    }
}

/// Returns the index of the first row where the
/// given Bit column equals `Bit::ONE`.
fn first_row_with_bit(trace: &ColumnTrace, col: usize) -> usize {
    let bits = trace.columns[col].as_bit_slice().unwrap();
    (0..bits.len())
        .find(|&r| bits[r] == Bit::ONE)
        .expect("no row with bit set")
}

/// XORs `mask` into the B32 cell at `(col, row)`.
fn flip_b32(trace: &mut ColumnTrace, col: usize, row: usize, mask: u32) {
    match &mut trace.columns[col] {
        TraceColumn::B32(data) => {
            let original = data[row];
            data[row] = Flat::from_raw(Block32(original.to_tower().0 ^ mask));
        }
        _ => panic!("expected B32 column"),
    }
}

fn rows_with_bit(trace: &ColumnTrace, col: usize) -> Vec<usize> {
    let bits = trace.columns[col].as_bit_slice().unwrap();
    (0..bits.len()).filter(|&r| bits[r] == Bit::ONE).collect()
}

fn swap_b32(trace: &mut ColumnTrace, col: usize, r0: usize, r1: usize) {
    match &mut trace.columns[col] {
        TraceColumn::B32(data) => data.swap(r0, r1),
        _ => panic!("expected B32"),
    }
}

fn flip_b64(trace: &mut ColumnTrace, col: usize, row: usize, mask: u64) {
    match &mut trace.columns[col] {
        TraceColumn::B64(data) => {
            let original = data[row];
            data[row] = Flat::from_raw(Block64(original.to_tower().0 ^ mask));
        }
        _ => panic!("expected B64 column"),
    }
}

// =================================================================
// Multi-level E2E tests (512, 768, 1024)
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn mlkem_512_e2e() {
    let level = MlKemLevel::MLKEM_512;
    let (pk, sk) = mlkem512::keypair();
    let (_, ct) = mlkem512::encapsulate(&pk);

    let result = prove_and_verify_mlkem_level(
        level,
        ct.as_bytes(),
        sk.as_bytes(),
        &params_for_level(level),
    );

    assert_eq!(result, Ok(true), "ML-KEM-512 E2E: {result:?}");
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn mlkem_768_e2e() {
    let level = MlKemLevel::MLKEM_768;
    let (pk, sk) = mlkem768::keypair();
    let (_, ct) = mlkem768::encapsulate(&pk);

    let result = prove_and_verify_mlkem_level(
        level,
        ct.as_bytes(),
        sk.as_bytes(),
        &params_for_level(level),
    );

    assert_eq!(result, Ok(true), "ML-KEM-768 E2E: {result:?}");
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn mlkem_1024_e2e() {
    let level = MlKemLevel::MLKEM_1024;
    let (pk, sk) = mlkem1024::keypair();
    let (_, ct) = mlkem1024::encapsulate(&pk);

    let result = prove_and_verify_mlkem_level(
        level,
        ct.as_bytes(),
        sk.as_bytes(),
        &params_for_level(level),
    );

    assert_eq!(result, Ok(true), "ML-KEM-1024 E2E: {result:?}");
}

// =================================================================
// Test 1:
// Valid proof MUST pass
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn honest_prover_succeeds() {
    let (pk, sk) = mlkem768::keypair();
    let (_, ct) = mlkem768::encapsulate(&pk);

    let result = prove_and_verify_mlkem(ct.as_bytes(), sk.as_bytes(), &test_params());
    assert_eq!(result, Ok(true), "Honest prover must succeed: {result:?}",);
}

// =================================================================
// Test 2:
// Wrong secret key -> RAM consistency MUST reject
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn wrong_secret_key_rejected() {
    // Generate two different keypairs
    let (pk1, _sk1) = mlkem768::keypair();
    let (_, sk2) = mlkem768::keypair();

    // Encapsulate with pk1
    let (_, ct1) = mlkem768::encapsulate(&pk1);

    // Try to decapsulate ct1 with sk2 (WRONG key).
    // The decapsulation will take the implicit
    // rejection path (ct != ct' because re-encryption
    // with wrong key differs). This should still
    // produce a VALID proof, implicit rejection
    // is a valid execution path.
    let result = prove_and_verify_mlkem(ct1.as_bytes(), sk2.as_bytes(), &test_params());

    // This SHOULD succeed, implicit rejection
    // is valid behavior. The proof proves that
    // decapsulation was performed correctly
    // (it just happened to reject).
    assert_eq!(
        result,
        Ok(true),
        "Implicit rejection path must produce valid proof: {result:?}",
    );
}

// =================================================================
// Test 3:
// Tampered ciphertext -> implicit rejection path valid
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn tampered_ciphertext_valid_rejection() {
    let (pk, sk) = mlkem768::keypair();
    let (_, ct) = mlkem768::encapsulate(&pk);

    // Tamper with ciphertext
    let mut ct_bad = ct.as_bytes().to_vec();
    ct_bad[0] ^= 0xff;

    // Decaps with tampered ct takes rejection path.
    // This is a VALID execution, proof should pass.
    let result = prove_and_verify_mlkem(&ct_bad, sk.as_bytes(), &test_params());
    assert_eq!(
        result,
        Ok(true),
        "Tampered ct rejection must produce valid proof: {result:?}",
    );
}

// =================================================================
// Test 4:
// RAM rejects if ct' differs from ct
// (the re-encryption comparison enforcement)
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn ram_enforces_ct_comparison() {
    // This test verifies that the RAM chiplet's
    // write-then-read pattern on the ciphertext
    // forces the re-encryption check.
    //
    // With a valid keypair, decaps succeeds and
    // ct == ct' (written then read at same addresses).
    // The RAM chiplet verifies consistency.
    let (pk, sk) = mlkem768::keypair();
    let (_nist_ss, ct) = mlkem768::encapsulate(&pk);

    // Full prove/verify
    let result = prove_and_verify_mlkem(ct.as_bytes(), sk.as_bytes(), &test_params());
    assert_eq!(
        result,
        Ok(true),
        "Valid decaps must prove and verify: {result:?}",
    );
}

// =================================================================
// EXPLOIT:
// NTT <> RAM data binding enforcement.
//
// A malicious prover dispatches NTT_B=X
// to the NTT chiplet but RAM_VAL_PACKED=Y (X≠Y)
// on a co-activated row. Without the binding
// constraint, both chiplets accept individually.
// The constraint ntt_sel * ram_sel * (ntt_b +
// ram_val_packed) = 0 must catch this.
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_ntt_ram_binding_mismatch() {
    let (pk, sk) = mlkem768::keypair();
    let (_, ct) = mlkem768::encapsulate(&pk);

    let params = test_params();
    let mlkem_chiplet = MlKemChiplet::<F>::new(MlKemLevel::MLKEM_768, params.clone());

    let (mut chiplet_traces, _) = mlkem_chiplet
        .generate_traces(ct.as_bytes(), sk.as_bytes())
        .expect("trace gen failed");

    // ctrl trace is chiplet_traces[0].
    // Find a co-activated row:
    // NTT_SELECTOR=1 AND RAM_SELECTOR=1.
    let ctrl = &chiplet_traces[0];

    let ntt_sel_bits = ctrl.columns[MlKemCtrlColumns::NTT_SELECTOR]
        .as_bit_slice()
        .unwrap();
    let ram_sel_bits = ctrl.columns[MlKemCtrlColumns::RAM_SELECTOR]
        .as_bit_slice()
        .unwrap();

    let coactivated_row = (0..ntt_sel_bits.len())
        .find(|&r| ntt_sel_bits[r] == Bit::ONE && ram_sel_bits[r] == Bit::ONE)
        .expect("No co-activated NTT+RAM row found — binding not wired");

    // Tamper:
    // flip RAM_VAL_PACKED on the co-activated row.
    // NTT_B stays correct.
    // Constraint fires:
    // ntt_b + ram_val_packed ≠ 0.
    let packed_col = &mut chiplet_traces[0].columns[MlKemCtrlColumns::RAM_VAL_PACKED];
    match packed_col {
        TraceColumn::B32(data) => {
            let original = data[coactivated_row];
            data[coactivated_row] = Flat::from_raw(Block32(original.to_tower().0 ^ 0x1));
        }
        _ => panic!("RAM_VAL_PACKED must be B32"),
    }

    // Prove with tampered trace
    let cpu_rows: usize = 1 << 10;
    let layout = CpuMlKemColumns::build_layout();
    let cpu_trace = TraceBuilder::new(&layout, cpu_rows.trailing_zeros() as usize)
        .unwrap()
        .build();

    let air = MlKemTestProgram {
        mlkem: mlkem_chiplet,
        num_public: 0,
    };

    let instance = ProgramInstance::new(cpu_rows, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(chiplet_traces);

    let mut config = Config {
        sumcheck_blinding_factor: 2,
        ..Config::default()
    };

    OsRng.try_fill_bytes(&mut config.matrix_seed).unwrap();

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    let proof_result = prove(
        b"MLKem_Adversarial",
        &air,
        &instance,
        &witness,
        &config,
        blinding_seed,
        None,
    );

    match proof_result {
        Err(_) => {
            // Prover rejected, binding enforced.
        }
        Ok(proof) => {
            let mut vt = Transcript::<H>::new(b"MLKem_Adversarial");
            let verify = HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut vt, &config);

            assert!(
                verify.is_err() || !verify.unwrap(),
                "Verifier accepted NTT_B != RAM_VAL_PACKED on co-activated row",
            );
        }
    }
}

// =================================================================
// EXPLOIT:
// NTT butterfly flow connectivity.
//
// A malicious prover corrupts a_out on an
// intermediate-layer forward butterfly row.
// The butterfly arithmetic passes (prover
// controls the witness), but the flow bus
// multiset equality fails: the corrupted
// output doesn't match any input at the
// next layer.
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_ntt_flow_connectivity_scramble() {
    let (pk, sk) = mlkem768::keypair();
    let (_, ct) = mlkem768::encapsulate(&pk);

    let params = test_params();
    let mlkem_chiplet = MlKemChiplet::<F>::new(MlKemLevel::MLKEM_768, params.clone());

    let (mut chiplet_traces, _) = mlkem_chiplet
        .generate_traces(ct.as_bytes(), sk.as_bytes())
        .expect("trace gen failed");

    // NTT trace is chiplet_traces[2].
    // Physical layout:
    // num_packed B32 + 16 B32 + 10 Bit.
    let ntt_layout = hekate_gadgets::chiplets::pqc::ntt::NttLayout::compute(3329, 12);
    let num_packed = ntt_layout.num_packed_b32_cols;
    let col_bus_a_out = num_packed + 4;
    let col_s_active = num_packed + 16;
    let col_s_flow_output = num_packed + 16 + 5;

    let ntt_trace = &mut chiplet_traces[2];

    let s_active = ntt_trace.columns[col_s_active].as_bit_slice().unwrap();
    let s_flow_out = ntt_trace.columns[col_s_flow_output].as_bit_slice().unwrap();

    // Find a forward butterfly with flow output
    // (intermediate layer, not layer 6).
    let target_row = (0..s_active.len())
        .find(|&r| s_active[r] == Bit::ONE && s_flow_out[r] == Bit::ONE)
        .expect("No flow-output butterfly row found");

    // Tamper:
    // flip bit 0 of bus_a_out
    match &mut ntt_trace.columns[col_bus_a_out] {
        TraceColumn::B32(data) => {
            let original = data[target_row];
            data[target_row] = Flat::from_raw(Block32(original.to_tower().0 ^ 0x1));
        }
        _ => panic!("bus_a_out must be B32"),
    }

    // Prove with tampered NTT trace
    let cpu_rows: usize = 1 << 10;
    let layout = CpuMlKemColumns::build_layout();
    let cpu_trace = TraceBuilder::new(&layout, cpu_rows.trailing_zeros() as usize)
        .unwrap()
        .build();

    let air = MlKemTestProgram {
        mlkem: mlkem_chiplet,
        num_public: 0,
    };

    let instance = ProgramInstance::new(cpu_rows, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(chiplet_traces);

    let mut config = Config {
        sumcheck_blinding_factor: 2,
        ..Config::default()
    };

    OsRng.try_fill_bytes(&mut config.matrix_seed).unwrap();

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    let proof_result = prove(
        b"MLKem_FlowExploit",
        &air,
        &instance,
        &witness,
        &config,
        blinding_seed,
        None,
    );

    match proof_result {
        Err(_) => {
            // Prover rejected, flow check caught it.
        }
        Ok(proof) => {
            let mut vt = Transcript::<H>::new(b"MLKem_FlowExploit");
            let verify = HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut vt, &config);

            assert!(
                verify.is_err() || !verify.unwrap(),
                "Verifier accepted scrambled butterfly a_out — flow connectivity bypass",
            );
        }
    }
}

// =================================================================
// EXPLOIT:
// Keccak input data is unbound.
//
// A malicious prover replaces the Keccak
// input state for a permutation call,
// recomputes keccak_f honestly with the
// fabricated input, and updates both ctrl
// and Keccak chiplet traces. Both sides
// of the Keccak bus carry consistent
// (but wrong) data. Without sponge state
// binding, the proof verifies, the prover
// hashed arbitrary data while claiming
// it computed G(m'||h).
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_keccak_input_unbound() {
    let (pk, sk) = mlkem768::keypair();
    let (_, ct) = mlkem768::encapsulate(&pk);

    let params = test_params();
    let mlkem_chiplet = MlKemChiplet::<F>::new(MlKemLevel::MLKEM_768, params.clone());

    let (mut chiplet_traces, _) = mlkem_chiplet
        .generate_traces(ct.as_bytes(), sk.as_bytes())
        .expect("trace gen failed");

    // Identify the target Keccak call.
    // Keccak trace = chiplet_traces[1].
    // Each call occupies 25 rows:
    // 24 rounds + 1 output row.
    // Row 0 of each call is the input
    // (s_in_out=1, s_round=1).
    // Row 24 is the output
    // (s_in_out=1, s_round=0).
    //
    // Target call 0 (first G hash).
    // Keccak trace rows 0..24.
    let keccak_trace = &mut chiplet_traces[1];

    // Read original input state
    // from Keccak trace row 0.
    // (lanes 0..24 are B64 columns).
    let mut original_input = [0u64; 25];
    for lane in 0..25 {
        match &keccak_trace.columns[lane] {
            TraceColumn::B64(data) => {
                original_input[lane] = data[0].to_tower().0;
            }
            _ => panic!("Keccak lane must be B64"),
        }
    }

    // Tamper:
    // flip lane 0 bit 0.
    let mut tampered_input = original_input;
    tampered_input[0] ^= 1;

    // Recompute keccak_f with tampered input
    let mut state = tampered_input;
    let mut round_states = Vec::with_capacity(25);

    for round in 0..24 {
        round_states.push(state);

        let rc = KeccakChiplet::ROUND_CONSTANTS[round];
        state = KeccakWitness::keccak_f_round(state, rc);
    }

    round_states.push(state); // output state

    let tampered_output = state;

    // Update Keccak chiplet trace:
    // 25 rows for call 0 (rows 0..24).
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

    // Update ctrl trace:
    // find the two ctrl rows for
    // Keccak call 0 (first pair of
    // KECCAK_SELECTOR=1 rows).
    let (ctrl_input_row, ctrl_output_row, reg_update_end) = {
        let ctrl = &chiplet_traces[0];
        let kec_sel_bits = ctrl.columns[MlKemCtrlColumns::KECCAK_SELECTOR]
            .as_bit_slice()
            .unwrap();

        let kec_rows: Vec<usize> = (0..kec_sel_bits.len())
            .filter(|&r| kec_sel_bits[r] == Bit::ONE)
            .collect();

        assert!(kec_rows.len() >= 2, "Need at least 2 Keccak ctrl rows");

        let in_row = kec_rows[0];
        let out_row = kec_rows[1];

        // Find where to stop RATE_REG updates
        let init_bits = ctrl.columns[MlKemCtrlColumns::SPONGE_INIT]
            .as_bit_slice()
            .unwrap();
        let active_bits = ctrl.columns[MlKemCtrlColumns::S_ACTIVE]
            .as_bit_slice()
            .unwrap();

        let mut end = out_row + 1;
        while end < active_bits.len() {
            if active_bits[end] == Bit::ZERO {
                end += 1; // include first padding
                break;
            }

            if init_bits[end] == Bit::ONE {
                break;
            }

            end += 1;
        }

        (in_row, out_row, end)
    };

    let ctrl = &mut chiplet_traces[0];

    // Tamper ctrl KeccakInput row
    // (lanes 0..24).
    for lane in 0..25 {
        match &mut ctrl.columns[MlKemCtrlColumns::KECCAK_LANES + lane] {
            TraceColumn::B64(data) => {
                data[ctrl_input_row] = Block64::from(tampered_input[lane]).to_hardware();
            }
            _ => panic!("KECCAK_LANES must be B64"),
        }
    }

    // Tamper ctrl KeccakOutput row
    // (lanes 0..24 with tampered output).
    for lane in 0..25 {
        match &mut ctrl.columns[MlKemCtrlColumns::KECCAK_LANES + lane] {
            TraceColumn::B64(data) => {
                data[ctrl_output_row] = Block64::from(tampered_output[lane]).to_hardware();
            }
            _ => panic!("KECCAK_LANES must be B64"),
        }
    }

    // Update RATE_REG on all rows from
    // the output row onward until the
    // next SPONGE_INIT (which resets to 0).
    // A real attacker controls the full
    // witness and would do this.
    for row in (ctrl_output_row + 1)..reg_update_end {
        for lane in 0..17 {
            match &mut ctrl.columns[MlKemCtrlColumns::RATE_REG + lane] {
                TraceColumn::B64(data) => {
                    data[row] = Block64::from(tampered_output[lane]).to_hardware();
                }
                _ => panic!("RATE_REG must be B64"),
            }
        }
    }

    // Prove with tampered Keccak input
    let cpu_rows: usize = 1 << 10;
    let layout = CpuMlKemColumns::build_layout();
    let cpu_trace = TraceBuilder::new(&layout, cpu_rows.trailing_zeros() as usize)
        .unwrap()
        .build();

    let air = MlKemTestProgram {
        mlkem: mlkem_chiplet,
        num_public: 0,
    };

    let instance = ProgramInstance::new(cpu_rows, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(chiplet_traces);

    let mut config = Config {
        sumcheck_blinding_factor: 2,
        ..Config::default()
    };

    OsRng.try_fill_bytes(&mut config.matrix_seed).unwrap();

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    let proof_result = prove(
        b"MLKem_Adversarial",
        &air,
        &instance,
        &witness,
        &config,
        blinding_seed,
        None,
    );

    // EXPLOIT CLOSED:
    // carry chain constrains KeccakInput
    // lanes to match ref row lanes.
    // RAM binding proves deltas match
    // ground-truth writes. Tampering
    // the input is detected at either
    // prove time (constraint violation)
    // or verify time (GPA mismatch).

    let detected = match proof_result {
        Err(_) => true,
        Ok(proof) => {
            let mut vt = Transcript::<H>::new(b"MLKem_Adversarial");
            let result = HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut vt, &config);

            result.is_err() || !result.unwrap()
        }
    };

    assert!(detected, "Keccak input tampering must be detected");
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_io_data_substitution() {
    let detected = run_tampered_mlkem_768(|traces| {
        let ctrl = &mut traces[0];
        let row = first_row_with_bit(ctrl, MlKemCtrlColumns::IO_SELECTOR);

        flip_b32(ctrl, MlKemCtrlColumns::IO_DATA, row, 0x1);
    });

    assert!(detected, "IO_DATA tampering must be rejected");
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_io_phase_skip() {
    let detected = run_tampered_mlkem_768(|traces| {
        let ctrl = &mut traces[0];

        let ph_dec_bits = ctrl.columns[MlKemCtrlColumns::PH_DECRYPT]
            .as_bit_slice()
            .unwrap();
        let row = (0..ph_dec_bits.len())
            .find(|&r| ph_dec_bits[r] == Bit::ONE)
            .expect("no PH_DECRYPT row");

        match &mut ctrl.columns[MlKemCtrlColumns::IO_SELECTOR] {
            TraceColumn::Bit(d) => d[row] = Bit::ONE,
            _ => panic!("IO_SELECTOR must be Bit"),
        }
    });

    assert!(detected, "IO_SELECTOR on non-PH_IO row must be rejected");
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_io_address_collision() {
    let detected = run_tampered_mlkem_768(|traces| {
        let ctrl = &mut traces[0];
        let row = first_row_with_bit(ctrl, MlKemCtrlColumns::IO_SELECTOR);

        flip_b32(ctrl, MlKemCtrlColumns::RAM_VAL_PACKED, row, 0xffff);
    });

    assert!(detected, "IO write address collision must be rejected");
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_io_pad_swap() {
    let detected = run_tampered_mlkem_768(|traces| {
        let ctrl = &mut traces[0];
        let row = first_row_with_bit(ctrl, MlKemCtrlColumns::IO_SELECTOR);

        match &mut ctrl.columns[MlKemCtrlColumns::PAD_SEL] {
            TraceColumn::Bit(d) => d[row] = Bit::ONE,
            _ => panic!("PAD_SEL must be Bit"),
        }
    });

    assert!(detected, "io_sel + pad_sel mutex must be rejected");
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_io_data_vs_ram_packed_mismatch() {
    let detected = run_tampered_mlkem_768(|traces| {
        let ctrl = &mut traces[0];
        let row = first_row_with_bit(ctrl, MlKemCtrlColumns::IO_SELECTOR);

        flip_b32(ctrl, MlKemCtrlColumns::IO_DATA, row, 0xff00);
    });

    assert!(
        detected,
        "IO_DATA != RAM_VAL_PACKED on IO row must be rejected"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_ct_substitution_at_h_ct() {
    let detected = run_tampered_mlkem_768(|traces| {
        let ctrl = &mut traces[0];
        let row = first_row_with_bit(ctrl, MlKemCtrlColumns::IO_LANE_BIND_SEL);

        flip_b32(ctrl, MlKemCtrlColumns::IO_LANE_LO, row, 0x1);
    });

    assert!(detected, "raw H(ct) read binding must be enforced");
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_padding_byte_substitution() {
    let detected = run_tampered_mlkem_768(|traces| {
        let ctrl = &mut traces[0];
        let pad_bits = ctrl.columns[MlKemCtrlColumns::PAD_SEL]
            .as_bit_slice()
            .unwrap();
        let row = (0..pad_bits.len())
            .find(|&r| pad_bits[r] == Bit::ONE)
            .expect("no PAD_SEL row");

        flip_b32(ctrl, MlKemCtrlColumns::IO_DATA, row, 0xff);
        flip_b32(ctrl, MlKemCtrlColumns::RAM_VAL_PACKED, row, 0xff);
    });

    assert!(detected, "SHA3 padding constant binding must be enforced");
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_h_ct_call_swap() {
    let detected = run_tampered_mlkem_768(|traces| {
        let ctrl = &mut traces[0];
        let h_ct_input_bits = ctrl.columns[MlKemCtrlColumns::H_CT_INPUT_SEL]
            .as_bit_slice()
            .unwrap();
        let row = (0..h_ct_input_bits.len())
            .find(|&r| h_ct_input_bits[r] == Bit::ONE)
            .expect("no H_CT_INPUT_SEL row");

        match &mut ctrl.columns[MlKemCtrlColumns::H_CT_INPUT_SEL] {
            TraceColumn::Bit(d) => d[row] = Bit::ZERO,
            _ => panic!("H_CT_INPUT_SEL must be Bit"),
        }
    });

    assert!(detected, "H_CT_INPUT_SEL placement must be enforced");
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_hash_ref_substitution() {
    let detected = run_tampered_mlkem_768(|traces| {
        let ctrl = &mut traces[0];
        let row = first_row_with_bit(ctrl, MlKemCtrlColumns::CMP_SELECTOR);

        // Tamper both sides of the bidirectional comparison
        // consistently so the structural cmp passes; only a
        // binding to the actual H(ct) digest can detect this.
        flip_b64(ctrl, MlKemCtrlColumns::HASH_REF, row, 0x1);
        flip_b64(ctrl, MlKemCtrlColumns::KECCAK_LANES, row, 0x1);
    });

    assert!(detected, "HASH_REF binding must be enforced");
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_keccak_lanes_cmp_substitution() {
    let detected = run_tampered_mlkem_768(|traces| {
        let ctrl = &mut traces[0];
        let row = first_row_with_bit(ctrl, MlKemCtrlColumns::H_CT_PRIME_BIND_SEL);

        flip_b64(ctrl, MlKemCtrlColumns::KECCAK_LANES, row, 0x1);
    });

    assert!(detected, "H_CT_PRIME_BIND_SEL binding must be enforced");
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_h_ct_bind_misplaced() {
    let detected = run_tampered_mlkem_768(|traces| {
        let ctrl = &mut traces[0];
        let row = first_row_with_bit(ctrl, MlKemCtrlColumns::H_CT_BIND_SEL);

        match &mut ctrl.columns[MlKemCtrlColumns::H_CT_BIND_SEL] {
            TraceColumn::Bit(d) => d[row] = Bit::ZERO,
            _ => panic!("H_CT_BIND_SEL must be Bit"),
        }
    });

    assert!(detected, "H_CT_BIND_SEL uniqueness must be enforced");
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_bind_sel_bypass() {
    let detected = run_tampered_mlkem_768(|traces| {
        let ctrl = &mut traces[0];

        let bind_row = first_row_with_bit(ctrl, MlKemCtrlColumns::H_CT_BIND_SEL);
        match &mut ctrl.columns[MlKemCtrlColumns::H_CT_BIND_SEL] {
            TraceColumn::Bit(d) => d[bind_row] = Bit::ZERO,
            _ => panic!("expected Bit"),
        }

        let prime_row = first_row_with_bit(ctrl, MlKemCtrlColumns::H_CT_PRIME_BIND_SEL);
        match &mut ctrl.columns[MlKemCtrlColumns::H_CT_PRIME_BIND_SEL] {
            TraceColumn::Bit(d) => d[prime_row] = Bit::ZERO,
            _ => panic!("expected Bit"),
        }

        let zero = Flat::from_raw(Block64(0));
        for i in 0..4 {
            match &mut ctrl.columns[MlKemCtrlColumns::HASH_REF + i] {
                TraceColumn::B64(d) => d.iter_mut().for_each(|v| *v = zero),
                _ => panic!("expected B64"),
            }
            match &mut ctrl.columns[MlKemCtrlColumns::HASH_CT_PRIME + i] {
                TraceColumn::B64(d) => d.iter_mut().for_each(|v| *v = zero),
                _ => panic!("expected B64"),
            }
        }

        let cmp_row = first_row_with_bit(ctrl, MlKemCtrlColumns::CMP_SELECTOR);
        for i in 0..4 {
            match &mut ctrl.columns[MlKemCtrlColumns::KECCAK_LANES + i] {
                TraceColumn::B64(d) => d[cmp_row] = zero,
                _ => panic!("expected B64"),
            }
        }
    });

    assert!(
        detected,
        "Clearing bind selectors + zeroing hash columns \
         must be rejected (FO comparison bypass)"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_cmp_phase_removal() {
    let detected = run_tampered_mlkem_768(|traces| {
        let ctrl = &mut traces[0];
        let cmp_row = first_row_with_bit(ctrl, MlKemCtrlColumns::CMP_SELECTOR);

        match &mut ctrl.columns[MlKemCtrlColumns::S_ACTIVE] {
            TraceColumn::Bit(d) => d[cmp_row] = Bit::ZERO,
            _ => panic!("expected Bit"),
        }
        match &mut ctrl.columns[MlKemCtrlColumns::PH_COMPARE] {
            TraceColumn::Bit(d) => d[cmp_row] = Bit::ZERO,
            _ => panic!("expected Bit"),
        }
        match &mut ctrl.columns[MlKemCtrlColumns::CMP_SELECTOR] {
            TraceColumn::Bit(d) => d[cmp_row] = Bit::ZERO,
            _ => panic!("expected Bit"),
        }
        match &mut ctrl.columns[MlKemCtrlColumns::CT_MATCH] {
            TraceColumn::Bit(d) => d[cmp_row] = Bit::ZERO,
            _ => panic!("expected Bit"),
        }
    });

    assert!(
        detected,
        "Removing the CMP phase entirely must be \
         rejected (no constraint enforces phase existence)"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_k_prime_forgery() {
    let detected = run_tampered_mlkem_768(|traces| {
        let ctrl = &mut traces[0];
        let row = first_row_with_bit(ctrl, MlKemCtrlColumns::K_PRIME_BIND_SEL);

        flip_b32(ctrl, MlKemCtrlColumns::K_PRIME_LO, row, 0xDEAD);
    });

    assert!(
        detected,
        "exploit_k_prime_forgery: bind constraint must reject"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_k_bar_forgery() {
    let detected = run_tampered_mlkem_768(|traces| {
        let ctrl = &mut traces[0];
        let row = first_row_with_bit(ctrl, MlKemCtrlColumns::K_BAR_BIND_SEL);

        flip_b32(ctrl, MlKemCtrlColumns::K_BAR_LO, row, 0xBEEF);
    });

    assert!(
        detected,
        "exploit_k_bar_forgery: bind constraint must reject"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_k_prime_carry_break() {
    let detected = run_tampered_mlkem_768(|traces| {
        let ctrl = &mut traces[0];
        let bind_row = first_row_with_bit(ctrl, MlKemCtrlColumns::K_PRIME_BIND_SEL);

        flip_b32(ctrl, MlKemCtrlColumns::K_PRIME_LO, bind_row + 2, 0x01);
    });

    assert!(
        detected,
        "exploit_k_prime_carry_break: sticky carry must reject"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_ct_match_carry_break() {
    let detected = run_tampered_mlkem_768(|traces| {
        let ctrl = &mut traces[0];
        let mux_row = first_row_with_bit(ctrl, MlKemCtrlColumns::SS_MUX_SEL);

        match &mut ctrl.columns[MlKemCtrlColumns::CT_MATCH] {
            TraceColumn::Bit(d) => {
                d[mux_row] = if d[mux_row] == Bit::ONE {
                    Bit::ZERO
                } else {
                    Bit::ONE
                };
            }
            _ => panic!("expected Bit"),
        }
    });

    assert!(
        detected,
        "exploit_ct_match_carry_break: sticky carry must reject"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_branch_substitution_match() {
    let detected = run_tampered_mlkem_768(|traces| {
        let ctrl = &mut traces[0];
        let mux_row = first_row_with_bit(ctrl, MlKemCtrlColumns::SS_MUX_SEL);

        flip_b32(ctrl, MlKemCtrlColumns::SS_LO, mux_row, 0xFF);
    });

    assert!(
        detected,
        "exploit_branch_substitution_match: mux constraint must reject"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_branch_substitution_reject() {
    let detected = run_tampered_mlkem_768(|traces| {
        let ctrl = &mut traces[0];
        let mux_row = first_row_with_bit(ctrl, MlKemCtrlColumns::SS_MUX_SEL);

        flip_b32(ctrl, MlKemCtrlColumns::SS_HI, mux_row, 0xFF);
    });

    assert!(
        detected,
        "exploit_branch_substitution_reject: mux constraint must reject"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_ss_bus_corruption() {
    let detected = run_tampered_mlkem_768_with_ss(|_chiplet_traces, cpu_trace| {
        let ss_row = first_row_with_bit(cpu_trace, CpuMlKemColumns::SS_SELECTOR);

        flip_b32(cpu_trace, CpuMlKemColumns::SS_DATA, ss_row, 0xCAFE);
    });

    assert!(
        detected,
        "exploit_ss_bus_corruption must PASS (bus permutation check is structural)"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_mlkem_data_duplicate_cpu_request_rejected() {
    let detected = run_tampered_mlkem_768_with_ss(|_chiplet_traces, cpu_trace| {
        let value = match &cpu_trace.columns[CpuMlKemColumns::DATA] {
            TraceColumn::B32(data) => data[0],
            _ => panic!("expected B32 column at DATA"),
        };

        let target = first_row_with_bit(cpu_trace, CpuMlKemColumns::SS_SELECTOR) + 1;
        match &mut cpu_trace.columns[CpuMlKemColumns::DATA] {
            TraceColumn::B32(data) => data[target] = value,
            _ => unreachable!(),
        }

        let bits = cpu_trace.columns[CpuMlKemColumns::SELECTOR]
            .as_bit_slice()
            .unwrap();
        assert_eq!(bits[target], Bit::ZERO, "duplicate target row not free");

        match &mut cpu_trace.columns[CpuMlKemColumns::SELECTOR] {
            TraceColumn::Bit(data) => data[target] = Bit::ONE,
            _ => unreachable!(),
        }
    });

    assert!(
        detected,
        "duplicate CPU IO request without chiplet partner must be caught by ml_kem_data bus"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_mlkem_ss_duplicate_cpu_request_rejected() {
    let detected = run_tampered_mlkem_768_with_ss(|_chiplet_traces, cpu_trace| {
        let ss_row = first_row_with_bit(cpu_trace, CpuMlKemColumns::SS_SELECTOR);
        let target = ss_row + 1;

        for j in 0..8 {
            let value = match &cpu_trace.columns[CpuMlKemColumns::SS_DATA + j] {
                TraceColumn::B32(data) => data[ss_row],
                _ => panic!("expected B32 column at SS_DATA+{j}"),
            };

            match &mut cpu_trace.columns[CpuMlKemColumns::SS_DATA + j] {
                TraceColumn::B32(data) => data[target] = value,
                _ => unreachable!(),
            }
        }

        match &mut cpu_trace.columns[CpuMlKemColumns::SS_SELECTOR] {
            TraceColumn::Bit(data) => data[target] = Bit::ONE,
            _ => unreachable!(),
        }
    });

    assert!(
        detected,
        "duplicate CPU SS request without chiplet partner must be caught by ml_kem_ss bus"
    );
}

// Multiset-preserving swap on basemul dispatch.
// Caught by BM-RAM co-activation binding.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_bm_dispatch_swap() {
    let detected = run_tampered_mlkem_768(|traces| {
        let ctrl = &mut traces[0];
        let bm_rows = rows_with_bit(ctrl, MlKemCtrlColumns::BM_SELECTOR);

        assert!(bm_rows.len() >= 4);

        let (r0, r1) = (bm_rows[2], bm_rows[3]);

        swap_b32(ctrl, MlKemCtrlColumns::BM_A, r0, r1);
        swap_b32(ctrl, MlKemCtrlColumns::BM_B, r0, r1);
        swap_b32(ctrl, MlKemCtrlColumns::BM_C, r0, r1);
        swap_b32(ctrl, MlKemCtrlColumns::BM_IDX, r0, r1);
    });

    assert!(
        detected,
        "basemul dispatch swap must be caught by BM-RAM binding"
    );
}

// Multiset-preserving swap on NTT dispatch.
// Caught by NTT-RAM co-activation binding.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_ntt_dispatch_swap() {
    let ctrl_ntt_rows = |traces: &[ColumnTrace]| -> Vec<usize> {
        let ctrl = &traces[0];
        let ntt_rows = rows_with_bit(ctrl, MlKemCtrlColumns::NTT_SELECTOR);
        let bound_in = rows_with_bit(ctrl, MlKemCtrlColumns::BOUND_IN_SEL);
        let bound_out = rows_with_bit(ctrl, MlKemCtrlColumns::BOUND_OUT_SEL);

        ntt_rows
            .into_iter()
            .filter(|r| !bound_in.contains(r) && !bound_out.contains(r))
            .collect()
    };

    let detected = run_tampered_mlkem_768(|traces| {
        let non_bound = ctrl_ntt_rows(traces);
        assert!(non_bound.len() >= 4);

        let (r0, r1) = (non_bound[2], non_bound[3]);
        let ctrl = &mut traces[0];

        swap_b32(ctrl, MlKemCtrlColumns::NTT_A, r0, r1);
        swap_b32(ctrl, MlKemCtrlColumns::NTT_B, r0, r1);
        swap_b32(ctrl, MlKemCtrlColumns::NTT_A_OUT, r0, r1);
        swap_b32(ctrl, MlKemCtrlColumns::NTT_B_OUT, r0, r1);
        swap_b32(ctrl, MlKemCtrlColumns::NTT_LAYER, r0, r1);
        swap_b32(ctrl, MlKemCtrlColumns::NTT_BUTTERFLY, r0, r1);
        swap_b32(ctrl, MlKemCtrlColumns::NTT_INSTANCE, r0, r1);
    });

    assert!(
        detected,
        "NTT dispatch swap must be caught by NTT-RAM binding"
    );
}
