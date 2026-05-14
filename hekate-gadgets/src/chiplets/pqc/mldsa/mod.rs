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

//! ML-DSA Composite Chiplet.
//!
//! Supports all three security levels (44, 65, 87)
//! via [`MlDsaLevel`] runtime parameterization.

mod arithmetic;
mod ctrl;
mod trace;
mod witness;

pub use ctrl::MlDsaCtrlColumns;
pub use witness::{MlDsaPublicKey, MlDsaSignature};

use super::high_bits::HighBitsChiplet;
use super::norm_check::NormCheckChiplet;
use super::ntt::NttChiplet;
use super::twiddle_rom::TwiddleRomChiplet;
use crate::chiplets::keccak::KeccakChiplet;
use crate::chiplets::ram::RamChiplet;
use alloc::vec;
use ctrl::MlDsaCtrlChiplet;
use hekate_core::trace::TraceCompatibleField;
use hekate_math::{Flat, HardwareField, PackableField, TowerField};
use hekate_program::chiplet::CompositeChiplet;
use hekate_program::define_columns;
use hekate_program::permutation::{PermutationCheckSpec, Source, REQUEST_IDX_LABEL};

// =================================================================
// Constants
// =================================================================

/// ML-DSA modulus (FIPS 204).
pub const MLDSA_Q: u32 = 8380417;

/// Bit width of MLDSA_Q.
pub const MLDSA_BIT_WIDTH: usize = 23;

/// Polynomial ring dimension.
pub const N: usize = 256;

/// External bus ID for ML-DSA I/O.
pub const MLDSA_DATA_BUS_ID: &str = "ml_dsa_data";

/// Bus ID for Keccak input binding.
const KEC_INPUT_BIND_BUS_ID: &str = "kec_input_bind";

// =================================================================
// Level Parameters
// =================================================================

/// ML-DSA security level
/// parameters (FIPS 204 Table 1).
#[derive(Clone, Copy, Debug)]
pub struct MlDsaLevel {
    pub(crate) k: usize,
    pub(crate) l: usize,
    #[allow(dead_code)]
    pub(crate) eta: u32,
    pub(crate) tau: usize,
    pub(crate) gamma1: u32,
    pub(crate) gamma2: u32,
    pub(crate) beta: u32,
    pub(crate) omega: usize,

    /// Dropped bits from t (FIPS 204 §5.2).
    pub(crate) d: usize,
}

impl MlDsaLevel {
    pub const MLDSA_44: Self = Self {
        k: 4,
        l: 4,
        eta: 2,
        tau: 39,
        gamma1: 1 << 17, // 2^17 = 131072
        gamma2: 95232,   // (q-1)/88
        beta: 78,        // tau * eta
        omega: 80,
        d: 13,
    };

    pub const MLDSA_65: Self = Self {
        k: 6,
        l: 5,
        eta: 4,
        tau: 49,
        gamma1: 1 << 19, // 2^19 = 524288
        gamma2: 261888,  // (q-1)/32
        beta: 196,       // tau * eta
        omega: 55,
        d: 13,
    };

    pub const MLDSA_87: Self = Self {
        k: 8,
        l: 7,
        eta: 2,
        tau: 60,
        gamma1: 1 << 19,
        gamma2: 261888, // (q-1)/32
        beta: 120,      // tau * eta
        omega: 75,
        d: 13,
    };

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn l(&self) -> usize {
        self.l
    }

    pub fn gamma2(&self) -> u32 {
        self.gamma2
    }

    pub fn omega(&self) -> usize {
        self.omega
    }

    /// Norm bound for z rejection:
    /// γ₁ - β.
    pub fn z_bound(&self) -> u32 {
        self.gamma1 - self.beta
    }

    /// HighBits divisor:
    /// 2γ₂.
    pub fn highbits_divisor(&self) -> u32 {
        2 * self.gamma2
    }

    /// Public key byte length (FIPS 204 §5.2).
    pub fn pk_bytes(&self) -> usize {
        // ρ (32) + t1 (k × bitlen(⌈(q-1)/(2^d)⌉) × N / 8)
        // t1 coefficients are 10 bits for d=13
        32 + self.k * 320
    }

    /// Signature byte length (FIPS 204 §5.2).
    pub fn sig_bytes(&self) -> usize {
        // c̃ (λ/4 bytes) + z (l × bitlen(γ₁-1) × N / 8) + h (ω + k)
        let lambda_bytes = match self.k {
            4 => 32, // λ=128 -> 32 bytes
            6 => 48, // λ=192 -> 48 bytes
            8 => 64, // λ=256 -> 64 bytes
            _ => unreachable!(),
        };

        let gamma1_bits = if self.gamma1 == (1 << 17) { 18 } else { 20 };
        let z_bytes = self.l * gamma1_bits * N / 8;
        let h_bytes = self.omega + self.k;

        lambda_bytes + z_bytes + h_bytes
    }
}

// =================================================================
// ML-DSA Composite Wrapper
// =================================================================

/// ML-DSA chiplet trace sizing.
#[derive(Clone, Debug)]
pub struct MlDsaParams {
    pub ctrl_rows: usize,
    pub keccak_rows: usize,
    pub ntt_rows: usize,
    pub twiddle_rows: usize,
    pub norm_rows: usize,
    pub highbits_rows: usize,
    pub ram_rows: usize,
}

impl Default for MlDsaParams {
    /// Default sizing for ML-DSA-65
    /// single verification.
    fn default() -> Self {
        Self {
            ctrl_rows: 1 << 15,     // 32K
            keccak_rows: 1 << 11,   // 2K
            ntt_rows: 1 << 16,      // 64K
            twiddle_rows: 1 << 10,  // 1K
            norm_rows: 1 << 11,     // 2K
            highbits_rows: 1 << 11, // 2K
            ram_rows: 1 << 15,      // 32K
        }
    }
}

/// ML-DSA Chiplet.
///
/// Composite wrapping the full
/// ML-DSA verification pipeline.
#[derive(Clone)]
pub struct MlDsaChiplet<F: TraceCompatibleField> {
    composite: CompositeChiplet<F>,
    level: MlDsaLevel,
    params: MlDsaParams,
}

impl<F> MlDsaChiplet<F>
where
    F: TowerField + TraceCompatibleField + PackableField + HardwareField + 'static,
    <F as PackableField>::Packed: Copy + Send + Sync,
    Flat<F>: Send + Sync,
{
    pub fn new(level: MlDsaLevel, params: MlDsaParams) -> Self {
        let ctrl = MlDsaCtrlChiplet::new(params.ctrl_rows);
        let keccak = KeccakChiplet::new(params.keccak_rows);
        let ntt = NttChiplet::new(MLDSA_Q, params.ntt_rows);
        let twiddle = TwiddleRomChiplet::new(MLDSA_Q, params.twiddle_rows);
        let norm = NormCheckChiplet::new(MLDSA_Q, level.z_bound(), params.norm_rows);
        let highbits =
            HighBitsChiplet::new(MLDSA_Q, level.highbits_divisor(), params.highbits_rows);
        let ram = RamChiplet::new(params.ram_rows);

        let composite = CompositeChiplet::<F>::builder("mldsa")
            .chiplet(ctrl)
            .chiplet(keccak)
            .chiplet(ntt)
            .chiplet(twiddle)
            .chiplet(norm)
            .chiplet(highbits)
            .chiplet(ram)
            .external_bus(MLDSA_DATA_BUS_ID, MlDsaCtrlChiplet::main_linking_spec())
            .build()
            .expect("ML-DSA composite build must succeed");

        Self {
            composite,
            level,
            params,
        }
    }

    pub fn composite(&self) -> &CompositeChiplet<F> {
        &self.composite
    }

    pub fn level(&self) -> MlDsaLevel {
        self.level
    }

    pub fn params(&self) -> &MlDsaParams {
        &self.params
    }
}

// =================================================================
// CPU-Side Interface
// =================================================================

define_columns! {
    pub CpuMlDsaColumns {
        DATA: B32,
        SELECTOR: Bit,
    }
}

/// CPU-side unit for ML-DSA chiplet.
///
/// Main trace connects to the ML-DSA
/// composite's "ml_dsa_data" external bus.
pub struct CpuMlDsaUnit;

impl CpuMlDsaUnit {
    pub fn num_columns() -> usize {
        CpuMlDsaColumns::NUM_COLUMNS
    }

    pub fn linking_spec() -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(CpuMlDsaColumns::DATA),
                    b"kappa_mldsa_d0" as &[u8],
                ),
                (Source::RowIndexLeBytes(4), REQUEST_IDX_LABEL),
            ],
            Some(CpuMlDsaColumns::SELECTOR),
        )
    }
}

// =================================================================
// Protocol Phase
// =================================================================

/// Protocol execution phase for the
/// ML-DSA control chiplet state machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub(crate) enum Phase {
    /// Public input deposit (pk, sig, M).
    Io = 0,

    /// ExpandA + SampleInBall.
    ExpandSample = 1,

    /// NTT forward:
    /// c, z[0..l].
    NttForward = 2,

    /// Pointwise multiply + accumulate.
    PointwiseMul = 3,

    /// Inverse NTT:
    /// w_approx.
    NttInverse = 4,

    /// UseHint:
    /// w'_1.
    UseHint = 5,

    /// Hash compare:
    /// c̃ vs c̃'.
    HashCompare = 6,

    /// Norm check:
    /// ‖z‖_∞ < γ₁ - β.
    NormCheck = 7,
}
