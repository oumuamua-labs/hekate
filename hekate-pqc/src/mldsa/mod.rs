// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! ML-DSA Composite Chiplet.
//!
//! Supports all three security levels (44, 65, 87)
//! via [`MlDsaLevel`] runtime parameterization.

mod arithmetic;
mod ctrl;
mod schedule;
mod trace;
mod witness;

pub use ctrl::MlDsaCtrlColumns;
pub use witness::{MlDsaPublicKey, MlDsaSignature};

use super::high_bits::HighBitsChiplet;
use super::norm_check::NormCheckChiplet;
use super::ntt::{NttChiplet, NttRun, NttSchedule};
use super::twiddle_rom::TwiddleRomChiplet;
use alloc::vec;
use ctrl::MlDsaCtrlChiplet;
use hekate_core::trace::TraceCompatibleField;
use hekate_gadgets::chiplets::ram::RamChiplet;
use hekate_keccak::KeccakChiplet;
use hekate_math::{Flat, HardwareField, PackableField, TowerField};
use hekate_program::chiplet::CompositeChiplet;
use hekate_program::define_columns;
use hekate_program::permutation::{BusKind, PermutationCheckSpec, Service, ServiceSlot};
use schedule::MlDsaCtrlSchedule;

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
pub(crate) struct MlDsaParams {
    pub ctrl_rows: usize,
    pub keccak_rows: usize,
    pub keccak_blocks: usize,
    pub ntt_rows: usize,
    pub twiddle_rows: usize,
    pub norm_rows: usize,
    pub highbits_rows: usize,
    pub ram_rows: usize,
}

impl MlDsaParams {
    /// Shifts are the FIPS 204 KAT-validated floors; ctrl and
    /// Keccak are raised above theirs to fit the message hash.
    pub(crate) fn for_level(level: MlDsaLevel, schedule: &MlDsaCtrlSchedule) -> Self {
        let (table_shift, keccak_shift, aux_shift) = match level.k() {
            8 => (17, 14, 12),
            _ => (16, 13, 11),
        };

        let keccak_blocks = schedule.keccak_calls();

        Self {
            ctrl_rows: (schedule.active_rows() + 1)
                .next_power_of_two()
                .max(1 << table_shift),
            keccak_rows: (keccak_blocks * KeccakChiplet::BLOCK_ROWS)
                .next_power_of_two()
                .max(1 << keccak_shift),
            keccak_blocks,
            ntt_rows: 1 << table_shift,
            twiddle_rows: 1 << table_shift,
            norm_rows: 1 << aux_shift,
            highbits_rows: 1 << aux_shift,
            ram_rows: 1 << table_shift,
        }
    }
}

/// ML-DSA Chiplet.
///
/// Composite wrapping the full ML-DSA verification pipeline.
#[derive(Clone)]
pub struct MlDsaChiplet<F: TraceCompatibleField> {
    composite: CompositeChiplet<F>,
    level: MlDsaLevel,
    msg_len: usize,
    params: MlDsaParams,
}

impl<F> MlDsaChiplet<F>
where
    F: TowerField + TraceCompatibleField + PackableField + HardwareField + 'static,
    <F as PackableField>::Packed: Copy + Send + Sync,
    Flat<F>: Send + Sync,
{
    pub fn new(level: MlDsaLevel, msg_len: usize) -> Self {
        let ctrl_schedule = MlDsaCtrlSchedule::for_level(level, msg_len);
        let params = MlDsaParams::for_level(level, &ctrl_schedule);

        let ntt_schedule = ml_dsa_ntt_schedule(level.k, level.l);

        let norm = NormCheckChiplet::new(
            MLDSA_Q,
            level.z_bound(),
            params.norm_rows,
            ctrl_schedule.norm_check_ops(),
        );
        let highbits = HighBitsChiplet::new(
            MLDSA_Q,
            level.highbits_divisor(),
            params.highbits_rows,
            ctrl_schedule.highbits_ops(),
        );
        let ram = RamChiplet::new(params.ram_rows, ctrl_schedule.ram_events());
        let ctrl = MlDsaCtrlChiplet::new(params.ctrl_rows, ctrl_schedule);
        let keccak = KeccakChiplet::new(params.keccak_rows, params.keccak_blocks);
        let ntt = NttChiplet::new(MLDSA_Q, params.ntt_rows, ntt_schedule.clone());
        let twiddle = TwiddleRomChiplet::new(MLDSA_Q, params.twiddle_rows, ntt_schedule);

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
            msg_len,
            params,
        }
    }

    pub fn composite(&self) -> &CompositeChiplet<F> {
        &self.composite
    }

    pub fn level(&self) -> MlDsaLevel {
        self.level
    }

    pub fn msg_len(&self) -> usize {
        self.msg_len
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

/// Both endpoints derive from this schema:
/// one commitment word, then the emit rank.
pub fn data_service() -> Service {
    Service {
        bus_id: MLDSA_DATA_BUS_ID,
        kind: BusKind::Permutation,
        slots: vec![ServiceSlot::Value(b"kappa_mldsa_d0"), ServiceSlot::EmitRank],
    }
}

/// Requester endpoint over `CpuMlDsaColumns`.
pub fn cpu_data_spec() -> PermutationCheckSpec {
    data_service()
        .request(&[CpuMlDsaColumns::DATA], CpuMlDsaColumns::SELECTOR)
        .expect("data_service slots match the requester columns")
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

fn ml_dsa_ntt_schedule(k: usize, l: usize) -> NttSchedule {
    let mut runs = vec![
        NttRun::Forward { instances: l + 2 },
        NttRun::Pointwise { calls: l + 1 },
    ];

    for _ in 0..k - 1 {
        runs.push(NttRun::Forward { instances: 1 });
        runs.push(NttRun::Pointwise { calls: l + 1 });
    }

    runs.push(NttRun::Inverse { instances: k });

    NttSchedule::new(8, runs)
}
