// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! ML-KEM Composite Chiplet.
//!
//! Supports all three security levels (512, 768, 1024)
//! via [`MlKemLevel`] runtime parameterization.
//!
//! Encapsulates the full ML-KEM decapsulation
//! pipeline as a single CompositeChiplet.

mod arithmetic;
mod ctrl;
mod schedule;
mod trace;
mod witness;

pub use ctrl::MlKemCtrlColumns;
pub use witness::{ntt_forward_traced, ntt_inverse_traced};

use super::basemul::BasemulChiplet;
use super::ntt::{NttChiplet, NttRun, NttSchedule};
use super::twiddle_rom::TwiddleRomChiplet;
use alloc::vec;
use alloc::vec::Vec;
use ctrl::MlKemCtrlChiplet;
use hekate_core::trace::TraceCompatibleField;
use hekate_gadgets::chiplets::ram::RamChiplet;
use hekate_keccak::KeccakChiplet;
use hekate_math::{Flat, HardwareField, PackableField, TowerField};
use hekate_program::chiplet::CompositeChiplet;
use hekate_program::define_columns;
use hekate_program::permutation::{BusKind, PermutationCheckSpec, Service, ServiceSlot};
use schedule::MlKemCtrlSchedule;

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

#[rustfmt::skip]
const MLKEM_SS_LABELS: [&[u8]; 8] = [
    b"kappa_ss_lo0", b"kappa_ss_lo1",
    b"kappa_ss_lo2", b"kappa_ss_lo3",
    b"kappa_ss_hi0", b"kappa_ss_hi1",
    b"kappa_ss_hi2", b"kappa_ss_hi3",
];

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
pub(crate) struct MlKemParams {
    pub ctrl_rows: usize,
    pub keccak_rows: usize,
    pub keccak_blocks: usize,
    pub ntt_rows: usize,
    pub twiddle_rows: usize,
    pub basemul_rows: usize,
    pub ram_rows: usize,
}

impl MlKemParams {
    /// Shifts are the FIPS 203 KAT-validated sizes.
    pub(crate) fn for_level(level: MlKemLevel, schedule: &MlKemCtrlSchedule) -> Self {
        let k = level.k;
        let keccak_shift = if k >= 4 { 12 } else { 11 };
        let ctrl_shift = if k >= 4 { 17 } else { 16 };

        Self {
            ctrl_rows: 1 << ctrl_shift,
            keccak_rows: 1 << keccak_shift,
            keccak_blocks: schedule.keccak_calls(),
            ntt_rows: 1 << (14 + k.div_ceil(3)),
            twiddle_rows: 1 << (14 + k.div_ceil(3)),
            basemul_rows: 1 << (11 + k.div_ceil(2)),
            ram_rows: 1 << (14 + k.div_ceil(2)),
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
/// let mlkem = MlKemChiplet::new(MlKemLevel::MLKEM_768);
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
    pub fn new(level: MlKemLevel) -> Self {
        let ctrl_schedule = MlKemCtrlSchedule::for_level(level);
        let params = MlKemParams::for_level(level, &ctrl_schedule);

        let ntt_schedule = ml_kem_ntt_schedule(level.k);

        let basemul =
            BasemulChiplet::new(MLKEM_Q, params.basemul_rows, ctrl_schedule.basemul_ops());
        let ram = RamChiplet::new(params.ram_rows, ctrl_schedule.ram_events());
        let ctrl = MlKemCtrlChiplet::new(params.ctrl_rows, ctrl_schedule);
        let keccak = KeccakChiplet::new(params.keccak_rows, params.keccak_blocks);
        let ntt = NttChiplet::new(MLKEM_Q, params.ntt_rows, ntt_schedule.clone());
        let twiddle = TwiddleRomChiplet::new(MLKEM_Q, params.twiddle_rows, ntt_schedule);

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

/// Both endpoints derive from this schema:
/// one ciphertext word, then the emit rank.
pub fn data_service() -> Service {
    Service {
        bus_id: MLKEM_DATA_BUS_ID,
        kind: BusKind::Permutation,
        slots: vec![ServiceSlot::Value(b"kappa_mlkem_d0"), ServiceSlot::EmitRank],
    }
}

/// Requester endpoint over `CpuMlKemColumns`.
pub fn cpu_data_spec() -> PermutationCheckSpec {
    data_service()
        .request(&[CpuMlKemColumns::DATA], CpuMlKemColumns::SELECTOR)
        .expect("data_service slots match the requester columns")
}

/// Both endpoints derive from this schema: the shared
/// secret as 4 low and 4 high words, then the emit rank.
pub fn ss_service() -> Service {
    let mut slots: Vec<ServiceSlot> = MLKEM_SS_LABELS
        .iter()
        .map(|label| ServiceSlot::Value(label))
        .collect();

    slots.push(ServiceSlot::EmitRank);

    Service {
        bus_id: MLKEM_SS_BUS_ID,
        kind: BusKind::Permutation,
        slots,
    }
}

/// Requester endpoint over `CpuMlKemColumns`.
pub fn cpu_ss_spec() -> PermutationCheckSpec {
    let values: Vec<usize> = (0..8).map(|i| CpuMlKemColumns::SS_DATA + i).collect();

    ss_service()
        .request(&values, CpuMlKemColumns::SS_SELECTOR)
        .expect("ss_service slots match the requester columns")
}

fn ml_kem_ntt_schedule(k: usize) -> NttSchedule {
    let mut runs = vec![
        NttRun::Forward { instances: k },
        NttRun::Pointwise { calls: k },
        NttRun::Inverse { instances: 1 },
        NttRun::Forward { instances: k },
    ];

    for _ in 0..=k {
        runs.push(NttRun::Pointwise { calls: k });
        runs.push(NttRun::Inverse { instances: 1 });
    }

    NttSchedule::new(7, runs)
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
        let mlkem = MlKemChiplet::<F>::new(MlKemLevel::MLKEM_768);

        assert_eq!(mlkem.composite().len(), 6);
        assert_eq!(mlkem.composite().name(), "mlkem");
    }

    #[test]
    fn flatten_defs_produces_six_defs() {
        let mlkem = MlKemChiplet::<F>::new(MlKemLevel::MLKEM_768);

        let defs: Vec<ChipletDef<F>> = mlkem.composite().flatten_defs().unwrap();
        assert_eq!(defs.len(), 6);
    }

    #[test]
    fn internal_buses_namespaced() {
        let mlkem = MlKemChiplet::<F>::new(MlKemLevel::MLKEM_768);

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
        let mlkem = MlKemChiplet::<F>::new(MlKemLevel::MLKEM_768);

        let ext = mlkem.composite().external_buses();
        assert_eq!(ext.len(), 2);
        assert_eq!(ext[0].0, "ml_kem_data");
        assert_eq!(ext[1].0, "ml_kem_ss");
    }
}
