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

//! ML-KEM-768 Decapsulation Proof Example.
//!
//! Proves ML-KEM-768 decapsulation (FIPS 203)
//! using the composite chiplet architecture.
//!
//! Pipeline:
//! 1. Generate keypair
//! 2. Encapsulate (create ciphertext + shared secret)
//! 3. Decapsulate (recover shared secret) with traced ops
//! 4. Generate chiplet traces
//! 5. Prove and verify

#[path = "common/mod.rs"]
mod common;

use hekate::core::trace::{ColumnType, TraceBuilder};
use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::{Bit, Block32, Block128};
use hekate_core::config::Config;
use hekate_gadgets::chiplets::pqc::mlkem::{
    self, CpuMlKemColumns, CpuMlKemUnit, MlKemChiplet, MlKemLevel, MlKemParams,
};
use hekate_math::TowerField;
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::constraint::{BoundaryConstraint, ConstraintAst};
use hekate_program::permutation::PermutationCheckSpec;
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;
use pqcrypto_mlkem::mlkem768;
use pqcrypto_traits::kem::{Ciphertext as _, SecretKey as _, SharedSecret as _};
use rand::TryRngCore;
use rand::rngs::OsRng;

type F = Block128;
type H = DefaultHasher;

// =================================================================
// ML-KEM Decapsulation Program
// =================================================================

#[derive(Clone)]
struct MlKemDecapsProgram {
    mlkem: MlKemChiplet<F>,
    num_public: usize,
}

impl Air<F> for MlKemDecapsProgram {
    fn name(&self) -> String {
        "MlKemDecapsProgram".into()
    }

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

impl Program<F> for MlKemDecapsProgram {
    fn num_public_inputs(&self) -> usize {
        self.num_public
    }

    fn chiplet_defs(&self) -> hekate_core::errors::Result<Vec<ChipletDef<F>>> {
        self.mlkem.composite().flatten_defs()
    }
}

// =================================================================
// Main
// =================================================================

fn main() {
    common::init("ML-KEM-768 Decapsulation");

    // Trace sizes for ML-KEM-768 decapsulation:
    //
    // Forward NTTs:
    // 6 × 1024 = 6144 butterfly ops
    //
    // INTT (GS→CT decomp):
    // 5 × (2048+256) = 11520 ops
    //
    // Basemul:
    // 15 × 256 = 3840 mul-only ops
    //
    // Total NTT ops:
    // ~21504 → 32768
    //
    // Keccak:
    // ~63 perms × 25 rows = 1575 → 2048
    //
    // Ctrl:
    // NTT + Keccak dispatches → 32768
    //
    // Twiddle:
    // 1 entry per NTT op → 32768
    //
    // Basemul:
    // 15 poly muls × 256 ops = 3840 → 4096
    //
    // RAM:
    // ~10 polys × 256 × 2 (write+read) = ~5120 → 8192
    let params = MlKemParams {
        ctrl_rows: 1 << 16,    // 65536 (NTT + Keccak + basemul + RAM dispatch)
        keccak_rows: 1 << 11,  // 2048
        ntt_rows: 1 << 15,     // 32768
        twiddle_rows: 1 << 15, // 32768
        basemul_rows: 1 << 12, // 4096
        ram_rows: 1 << 16,     // 65536 (decrypt + encrypt + ct comparison)
    };

    let cpu_num_rows: usize = 1 << 10; // 1024

    // Phase 1:
    // NIST reference keygen
    let (nist_pk, nist_sk) = common::phase("Key Generation (NIST)", mlkem768::keypair);

    // Phase 2:
    // NIST reference encapsulation
    let (nist_ss, nist_ct) =
        common::phase("Encapsulation (NIST)", || mlkem768::encapsulate(&nist_pk));

    let ct = nist_ct.as_bytes();
    let sk = nist_sk.as_bytes();

    let expected_ss = nist_ss.as_bytes();

    println!("  Ciphertext:     {} bytes", ct.len());

    // Phase 3:
    // Generate traces.
    let mlkem_chiplet = MlKemChiplet::<F>::new(MlKemLevel::MLKEM_768, params);

    let (cpu_trace, chiplet_traces, shared_secret) = common::phase("Trace Generation", || {
        let (chiplet_traces, shared_secret) = mlkem_chiplet
            .generate_traces(ct, sk)
            .expect("Trace generation failed");

        let layout = CpuMlKemColumns::build_layout();
        let cpu_vars = cpu_num_rows.trailing_zeros() as usize;

        let mut cpu_tb = TraceBuilder::new(&layout, cpu_vars).expect("CPU trace build failed");

        for (i, chunk) in ct.chunks(4).enumerate() {
            let mut buf = [0u8; 4];
            buf[..chunk.len()].copy_from_slice(chunk);

            cpu_tb
                .set_b32(
                    CpuMlKemColumns::DATA,
                    i,
                    Block32::from(u32::from_le_bytes(buf)),
                )
                .expect("CPU DATA set");
            cpu_tb
                .set_bit(CpuMlKemColumns::SELECTOR, i, Bit::ONE)
                .expect("CPU SELECTOR set");
        }

        let ss_row = ct.chunks(4).count();
        for i in 0..4 {
            let lo = u32::from_le_bytes(shared_secret[i * 8..i * 8 + 4].try_into().unwrap());
            let hi = u32::from_le_bytes(shared_secret[i * 8 + 4..i * 8 + 8].try_into().unwrap());

            cpu_tb
                .set_b32(CpuMlKemColumns::SS_DATA + i, ss_row, Block32::from(lo))
                .expect("CPU SS_DATA lo set");
            cpu_tb
                .set_b32(CpuMlKemColumns::SS_DATA + 4 + i, ss_row, Block32::from(hi))
                .expect("CPU SS_DATA hi set");
        }

        cpu_tb
            .set_bit(CpuMlKemColumns::SS_SELECTOR, ss_row, Bit::ONE)
            .expect("CPU SS_SELECTOR set");

        let cpu_trace = cpu_tb.build();

        (cpu_trace, chiplet_traces, shared_secret)
    });

    assert_eq!(
        &shared_secret, expected_ss,
        "Shared secret mismatch vs NIST reference"
    );

    println!("  Shared secret:  matches encapsulation");
    println!("  Chiplet traces: {}", chiplet_traces.len());

    let ct_public: Vec<F> = ct
        .chunks(4)
        .map(|chunk| {
            let mut buf = [0u8; 4];
            buf[..chunk.len()].copy_from_slice(chunk);

            Block128(u32::from_le_bytes(buf) as u128)
        })
        .collect();

    println!(
        "  Public inputs:  {} (ct as {} × B32)",
        ct_public.len(),
        ct_public.len()
    );

    // Phase 5:
    // Prove
    let air = MlKemDecapsProgram {
        mlkem: mlkem_chiplet,
        num_public: ct_public.len(),
    };

    let instance = ProgramInstance::new(cpu_num_rows, ct_public);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(chiplet_traces);

    let mut config = Config {
        sumcheck_blinding_factor: 2,
        ..Config::default()
    };

    OsRng.try_fill_bytes(&mut config.matrix_seed).unwrap();

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    let proof = common::phase("Proving", || {
        prove(
            b"ML-KEM-768_Decaps",
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

    // Phase 6:
    // Verify
    let mut verifier_transcript = Transcript::<H>::new(b"ML-KEM-768_Decaps");

    let is_valid = common::phase("Verifying", || {
        HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut verifier_transcript, &config)
            .expect("Verifier failed")
    });

    common::result(is_valid);
}
