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

//! ML-KEM Composite Chiplet.
//!
//! Supports all three security levels (512, 768, 1024)
//! via [`MlKemLevel`] runtime parameterization.
//!
//! Encapsulates the full ML-KEM decapsulation
//! pipeline as a single CompositeChiplet.

mod arithmetic;
mod ctrl;
mod trace;
mod witness;

pub use ctrl::MlKemCtrlColumns;

use super::basemul::BasemulChiplet;
use super::ntt::NttChiplet;
use super::twiddle_rom::TwiddleRomChiplet;
use crate::chiplets::keccak::KeccakChiplet;
use crate::chiplets::ram::RamChiplet;
use alloc::vec;
use ctrl::MlKemCtrlChiplet;
use hekate_core::trace::TraceCompatibleField;
use hekate_math::{Flat, HardwareField, PackableField, TowerField};
use hekate_program::chiplet::CompositeChiplet;
use hekate_program::define_columns;
use hekate_program::permutation::{PermutationCheckSpec, Source, REQUEST_IDX_LABEL};

// =================================================================
// Constants
// =================================================================

/// ML-KEM-768 modulus.
pub const MLKEM_Q: u32 = 3329;

/// ML-KEM-768 bit width.
pub const MLKEM_BIT_WIDTH: usize = 12;

/// External bus ID for ML-KEM I/O.
pub const MLKEM_DATA_BUS_ID: &str = "ml_kem_data";

/// External bus ID for shared secret output.
pub const MLKEM_SS_BUS_ID: &str = "ml_kem_ss";

/// Bus ID for Keccak input binding.
const KEC_INPUT_BIND_BUS_ID: &str = "kec_input_bind";

/// Polynomial ring dimension.
const N: usize = 256;

/// ML-KEM security level parameters (FIPS 203 Table 2).
#[derive(Clone, Copy, Debug)]
pub struct MlKemLevel {
    pub k: usize,
    pub eta1: usize,
    pub eta2: usize,
    pub du: usize,
    pub dv: usize,
}

impl MlKemLevel {
    pub const MLKEM_512: Self = Self {
        k: 2,
        eta1: 3,
        eta2: 2,
        du: 10,
        dv: 4,
    };

    pub const MLKEM_768: Self = Self {
        k: 3,
        eta1: 2,
        eta2: 2,
        du: 10,
        dv: 4,
    };

    pub const MLKEM_1024: Self = Self {
        k: 4,
        eta1: 2,
        eta2: 2,
        du: 11,
        dv: 5,
    };

    /// Secret key byte length.
    pub fn sk_bytes(&self) -> usize {
        let dk_pke = self.k * 12 * N / 8;
        let ek = self.ek_bytes();

        dk_pke + ek + 32 + 32
    }

    /// Public (encapsulation) key byte length.
    pub fn ek_bytes(&self) -> usize {
        self.k * 12 * N / 8 + 32
    }

    /// Ciphertext byte length.
    pub fn ct_bytes(&self) -> usize {
        self.k * N * self.du / 8 + N * self.dv / 8
    }
}

// =================================================================
// ML-KEM Composite Wrapper
// =================================================================

/// ML-KEM chiplet trace sizing.
/// Internal sub-chiplet row counts.
#[derive(Clone, Debug)]
pub struct MlKemParams {
    pub ctrl_rows: usize,
    pub keccak_rows: usize,
    pub ntt_rows: usize,
    pub twiddle_rows: usize,
    pub basemul_rows: usize,
    pub ram_rows: usize,
}

impl Default for MlKemParams {
    /// Use `Default` for ML-KEM-768 single-decaps.
    /// Override for batched proofs or custom sizing.
    fn default() -> Self {
        Self {
            ctrl_rows: 1 << 14,    // 16K
            keccak_rows: 1 << 9,   // 512
            ntt_rows: 1 << 15,     // 32K
            twiddle_rows: 1 << 10, // 1K
            basemul_rows: 1 << 12, // 4K
            ram_rows: 1 << 15,     // 32K
        }
    }
}

/// ML-KEM Chiplet.
///
/// High-level wrapper around CompositeChiplet
/// that encapsulates the full ML-KEM pipeline.
///
/// # Usage
///
/// ```ignore
/// let mlkem = MlKemChiplet::new(
///     MlKemLevel::MLKEM_768,
///     MlKemParams::default(),
/// );
///
/// // In Program impl:
/// fn chiplet_defs(&self) -> Vec<ChipletDef<F>> {
///     self.mlkem.composite().flatten_defs()
/// }
///
/// fn permutation_checks(&self) -> ... {
///     self.mlkem.composite().external_buses()
/// }
/// ```
#[derive(Clone)]
pub struct MlKemChiplet<F: TraceCompatibleField> {
    composite: CompositeChiplet<F>,
    level: MlKemLevel,
    params: MlKemParams,
}

impl<F> MlKemChiplet<F>
where
    F: TowerField + TraceCompatibleField + PackableField + HardwareField + 'static,
    <F as PackableField>::Packed: Copy + Send + Sync,
    Flat<F>: Send + Sync,
{
    pub fn new(level: MlKemLevel, params: MlKemParams) -> Self {
        let ctrl = MlKemCtrlChiplet::new(params.ctrl_rows);
        let keccak = KeccakChiplet::new(params.keccak_rows);
        let ntt = NttChiplet::new(MLKEM_Q, params.ntt_rows);
        let twiddle = TwiddleRomChiplet::new(MLKEM_Q, params.twiddle_rows);
        let basemul = BasemulChiplet::new(MLKEM_Q, params.basemul_rows);
        let ram = RamChiplet::new(params.ram_rows);

        let composite = CompositeChiplet::<F>::builder("mlkem")
            .chiplet(ctrl)
            .chiplet(keccak)
            .chiplet(ntt)
            .chiplet(twiddle)
            .chiplet(basemul)
            .chiplet(ram)
            .external_bus(MLKEM_DATA_BUS_ID, MlKemCtrlChiplet::main_linking_spec())
            .external_bus(MLKEM_SS_BUS_ID, MlKemCtrlChiplet::ss_linking_spec())
            .build()
            .expect("ML-KEM composite build must succeed");

        Self {
            composite,
            level,
            params,
        }
    }

    pub fn composite(&self) -> &CompositeChiplet<F> {
        &self.composite
    }

    pub fn level(&self) -> MlKemLevel {
        self.level
    }

    pub fn params(&self) -> &MlKemParams {
        &self.params
    }
}

// =================================================================
// CPU-Side Interface
// =================================================================

define_columns! {
    pub CpuMlKemColumns {
        DATA: B32,
        SELECTOR: Bit,
        SS_DATA: [B32; 8],
        SS_SELECTOR: Bit,
    }
}

/// CPU-side unit for ML-KEM chiplet.
///
/// Provides the column schema and linking spec
/// that the main trace uses to connect to the
/// ML-KEM composite's external bus.
pub struct CpuMlKemUnit;

impl CpuMlKemUnit {
    pub fn num_columns() -> usize {
        CpuMlKemColumns::NUM_COLUMNS
    }

    pub fn linking_spec() -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(CpuMlKemColumns::DATA),
                    b"kappa_mlkem_d0" as &[u8],
                ),
                (Source::RowIndexLeBytes(4), REQUEST_IDX_LABEL),
            ],
            Some(CpuMlKemColumns::SELECTOR),
        )
    }

    pub fn ss_linking_spec() -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (
                    Source::Column(CpuMlKemColumns::SS_DATA),
                    b"kappa_ss_lo0" as &[u8],
                ),
                (
                    Source::Column(CpuMlKemColumns::SS_DATA + 1),
                    b"kappa_ss_lo1" as &[u8],
                ),
                (
                    Source::Column(CpuMlKemColumns::SS_DATA + 2),
                    b"kappa_ss_lo2" as &[u8],
                ),
                (
                    Source::Column(CpuMlKemColumns::SS_DATA + 3),
                    b"kappa_ss_lo3" as &[u8],
                ),
                (
                    Source::Column(CpuMlKemColumns::SS_DATA + 4),
                    b"kappa_ss_hi0" as &[u8],
                ),
                (
                    Source::Column(CpuMlKemColumns::SS_DATA + 5),
                    b"kappa_ss_hi1" as &[u8],
                ),
                (
                    Source::Column(CpuMlKemColumns::SS_DATA + 6),
                    b"kappa_ss_hi2" as &[u8],
                ),
                (
                    Source::Column(CpuMlKemColumns::SS_DATA + 7),
                    b"kappa_ss_hi3" as &[u8],
                ),
                (Source::RowIndexLeBytes(4), REQUEST_IDX_LABEL),
            ],
            Some(CpuMlKemColumns::SS_SELECTOR),
        )
    }
}

/// Protocol execution phase for the
/// ML-KEM control chiplet state machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub(crate) enum Phase {
    /// Public ciphertext deposit
    /// + SHA3-256 padding bytes.
    Io = 0,

    /// NTT forward, basemul, INTT, RAM.
    Decrypt = 1,

    /// G(m'||h) hash.
    GHash = 2,

    /// NTT, basemul, RAM, Keccak.
    Encrypt = 3,

    /// H(ct), H(ct'), J(z||c).
    CmpHash = 4,

    /// Re-encryption hash comparison.
    Compare = 5,
}

// =================================================================
// Tests
// =================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use hekate_math::Block128;
    use hekate_program::chiplet::ChipletDef;

    type F = Block128;

    #[test]
    fn composite_builds_six_chiplets() {
        let mlkem = MlKemChiplet::<F>::new(
            MlKemLevel::MLKEM_768,
            MlKemParams {
                ctrl_rows: 16,
                keccak_rows: 32,
                ntt_rows: 16,
                twiddle_rows: 16,
                basemul_rows: 16,
                ram_rows: 16,
            },
        );

        assert_eq!(mlkem.composite().len(), 6);
        assert_eq!(mlkem.composite().name(), "mlkem");
    }

    #[test]
    fn flatten_defs_produces_six_defs() {
        let mlkem = MlKemChiplet::<F>::new(
            MlKemLevel::MLKEM_768,
            MlKemParams {
                ctrl_rows: 16,
                keccak_rows: 32,
                ntt_rows: 16,
                twiddle_rows: 16,
                basemul_rows: 16,
                ram_rows: 16,
            },
        );

        let defs: Vec<ChipletDef<F>> = mlkem.composite().flatten_defs().unwrap();
        assert_eq!(defs.len(), 6);
    }

    #[test]
    fn internal_buses_namespaced() {
        let mlkem = MlKemChiplet::<F>::new(
            MlKemLevel::MLKEM_768,
            MlKemParams {
                ctrl_rows: 16,
                keccak_rows: 32,
                ntt_rows: 16,
                twiddle_rows: 16,
                basemul_rows: 16,
                ram_rows: 16,
            },
        );

        let defs: Vec<ChipletDef<F>> = mlkem.composite().flatten_defs().unwrap();

        // Collect all bus_ids across all defs
        let mut bus_ids: Vec<String> = Vec::new();
        for def in &defs {
            for (id, _) in &def.permutation_checks {
                bus_ids.push(id.clone());
            }
        }

        // External bus NOT namespaced
        assert!(
            bus_ids.contains(&"ml_kem_data".to_string()),
            "external bus must not be namespaced, got: {bus_ids:?}",
        );

        // Internal buses namespaced with "mlkem::"
        assert!(
            bus_ids.contains(&"mlkem::keccak_link".to_string()),
            "keccak bus must be namespaced, got: {bus_ids:?}",
        );
        assert!(
            bus_ids.contains(&"mlkem::ntt_data".to_string()),
            "ntt_data bus must be namespaced, got: {bus_ids:?}",
        );
        assert!(
            bus_ids.contains(&"mlkem::ntt_twiddle".to_string()),
            "ntt_twiddle bus must be namespaced, got: {bus_ids:?}",
        );
        assert!(
            bus_ids.contains(&"mlkem::basemul".to_string()),
            "basemul bus must be namespaced, got: {bus_ids:?}",
        );
        assert!(
            bus_ids.contains(&"mlkem::ram_link".to_string()),
            "ram bus must be namespaced, got: {bus_ids:?}",
        );
        assert!(
            bus_ids.contains(&"mlkem::ntt_bound_in".to_string()),
            "ntt_bound_in bus must be namespaced, got: {bus_ids:?}",
        );
        assert!(
            bus_ids.contains(&"mlkem::ntt_bound_out".to_string()),
            "ntt_bound_out bus must be namespaced, got: {bus_ids:?}",
        );
    }

    #[test]
    fn external_bus_spec_returned() {
        let mlkem = MlKemChiplet::<F>::new(
            MlKemLevel::MLKEM_768,
            MlKemParams {
                ctrl_rows: 16,
                keccak_rows: 32,
                ntt_rows: 16,
                twiddle_rows: 16,
                basemul_rows: 16,
                ram_rows: 16,
            },
        );

        let ext = mlkem.composite().external_buses();
        assert_eq!(ext.len(), 2);
        assert_eq!(ext[0].0, "ml_kem_data");
        assert_eq!(ext[1].0, "ml_kem_ss");
    }
}
