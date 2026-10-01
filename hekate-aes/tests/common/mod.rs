// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

#![allow(dead_code)]

use hekate_aes::{
    Aes128Chiplet, Aes256Chiplet, AesRound128Air, AesRound256Air, CpuAes128Columns,
    CpuAes256Columns, PhysSboxRomColumns, host_key_selector_shape, host_selector_shape,
    trace::{Aes128Call, Aes256Call, expand_key, expand_key_256},
};
use hekate_core::config::Config;
use hekate_core::trace::{ColumnTrace, TraceBuilder, TraceColumn};
use hekate_crypto::DefaultHasher;
use hekate_crypto::transcript::Transcript;
use hekate_math::{Bit, Block8, Block16, Block64, Block128, HardwareField, TowerField};
use hekate_program::circuit::{Circuit, CircuitProgram, Col};
use hekate_program::digest::program_id;
use hekate_program::{Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_sdk::preflight;
use hekate_verifier::HekateVerifier;
use rand::{TryRngCore, rngs::OsRng};

pub type F = Block128;
pub type H = DefaultHasher;

pub const CPU_ROWS: usize = 4;
pub const AES_ROWS: usize = 16;
pub const SBOX_ROM_ROWS: usize = 256;

pub const IN_ROW: usize = 0;
pub const OUT_ROW: usize = 1;

#[rustfmt::skip]
pub const FREE_CIPHER: [u8; 16] = [
    0xde, 0xad, 0xbe, 0xef, 0xde, 0xad, 0xbe, 0xef,
    0xde, 0xad, 0xbe, 0xef, 0xde, 0xad, 0xbe, 0xef,
];

const TRANSCRIPT_LABEL: &[u8] = b"AES_Forgery";

// ===============================================
// FIPS 197 test vectors
// (Appendix B for AES-128, C.3 for AES-256)
// ===============================================

#[rustfmt::skip]
pub const FIPS128_KEY: [u8; 16] = [
    0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
    0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
];

#[rustfmt::skip]
pub const FIPS128_PLAIN: [u8; 16] = [
    0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d,
    0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34,
];

#[rustfmt::skip]
pub const FIPS128_CIPHER: [u8; 16] = [
    0x39, 0x25, 0x84, 0x1d, 0x02, 0xdc, 0x09, 0xfb,
    0xdc, 0x11, 0x85, 0x97, 0x19, 0x6a, 0x0b, 0x32,
];

#[rustfmt::skip]
pub const FIPS256_KEY: [u8; 32] = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
];

#[rustfmt::skip]
pub const FIPS256_PLAIN: [u8; 16] = [
    0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
    0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
];

#[rustfmt::skip]
pub const FIPS256_CIPHER: [u8; 16] = [
    0x8e, 0xa2, 0xb7, 0xca, 0x51, 0x67, 0x45, 0xbf,
    0xea, 0xfc, 0x49, 0x90, 0x4b, 0x49, 0x60, 0x89,
];

pub fn fips_call_128() -> Aes128Call {
    Aes128Call {
        key: FIPS128_KEY,
        plaintext: FIPS128_PLAIN,
        round_keys: expand_key(&FIPS128_KEY),
    }
}

pub fn fips_call_256() -> Aes256Call {
    Aes256Call {
        key: FIPS256_KEY,
        plaintext: FIPS256_PLAIN,
        round_keys: expand_key_256(&FIPS256_KEY),
    }
}

pub fn whitened_128() -> [u8; 16] {
    let rk = expand_key(&FIPS128_KEY);

    core::array::from_fn(|j| FIPS128_PLAIN[j] ^ rk[0][j])
}

pub fn whitened_256() -> [u8; 16] {
    let rk = expand_key_256(&FIPS256_KEY);

    core::array::from_fn(|j| FIPS256_PLAIN[j] ^ rk[0][j])
}

// ===============================================
// CPU-side host programs
// ===============================================

#[derive(Clone)]
pub struct Aes128Program {
    pub program: CircuitProgram<F>,
    pub aes: Aes128Chiplet,
}

#[derive(Clone)]
pub struct Aes256Program {
    pub program: CircuitProgram<F>,
    pub aes: Aes256Chiplet,
}

pub fn make_program_128(aes_rows: usize, num_blocks: usize) -> Aes128Program {
    let aes = Aes128Chiplet::new(aes_rows, SBOX_ROM_ROWS, num_blocks).unwrap();

    let mut cx = Circuit::<F>::new("Aes128Host", CPU_ROWS).unwrap();
    let cpu = cx.schema(&CpuAes128Columns::build_layout());

    let selector = cpu.at(CpuAes128Columns::SELECTOR);
    let key_selector = cpu.at(CpuAes128Columns::KEY_SELECTOR);

    let link_values: Vec<Col> = (0..16)
        .map(|j| cpu.at(CpuAes128Columns::DATA + j))
        .collect();

    cx.call(&AesRound128Air::link_service(), &link_values, selector)
        .unwrap();

    let key_values: Vec<Col> = (0..16).map(|j| cpu.at(CpuAes128Columns::KEY + j)).collect();

    cx.call(&AesRound128Air::key_service(), &key_values, key_selector)
        .unwrap();

    cx.fix(selector, host_selector_shape(2, num_blocks));
    cx.fix(key_selector, host_key_selector_shape(2, num_blocks));

    cx.attach_namespaced(
        "aes128",
        aes.defs().unwrap(),
        &Aes128Chiplet::EXTERNAL_BUS_IDS,
    )
    .unwrap();

    Aes128Program {
        program: cx.compile().unwrap(),
        aes,
    }
}

pub fn make_program_256(aes_rows: usize, num_blocks: usize) -> Aes256Program {
    let aes = Aes256Chiplet::new(aes_rows, SBOX_ROM_ROWS, num_blocks).unwrap();

    let mut cx = Circuit::<F>::new("Aes256Host", CPU_ROWS).unwrap();
    let cpu = cx.schema(&CpuAes256Columns::build_layout());

    let selector = cpu.at(CpuAes256Columns::SELECTOR);
    let key_selector = cpu.at(CpuAes256Columns::KEY_SELECTOR);

    let link_values: Vec<Col> = (0..16)
        .map(|j| cpu.at(CpuAes256Columns::DATA + j))
        .collect();

    cx.call(&AesRound256Air::link_service(), &link_values, selector)
        .unwrap();

    let key_values: Vec<Col> = (0..32).map(|j| cpu.at(CpuAes256Columns::KEY + j)).collect();

    cx.call(&AesRound256Air::key_service(), &key_values, key_selector)
        .unwrap();

    cx.fix(selector, host_selector_shape(2, num_blocks));
    cx.fix(key_selector, host_key_selector_shape(2, num_blocks));

    cx.attach_namespaced(
        "aes256",
        aes.defs().unwrap(),
        &Aes256Chiplet::EXTERNAL_BUS_IDS,
    )
    .unwrap();

    Aes256Program {
        program: cx.compile().unwrap(),
        aes,
    }
}

/// Call `k` emits on rows `2k` and `2k + 1`,
/// the ranks chiplet block `k` answers.
pub fn build_cpu_trace_128(blocks: &[([u8; 16], [u8; 16])]) -> ColumnTrace {
    let num_vars = CPU_ROWS.trailing_zeros() as usize;
    let mut tb = TraceBuilder::new(&CpuAes128Columns::build_layout(), num_vars).unwrap();

    for (k, (data_in, data_out)) in blocks.iter().enumerate() {
        let in_row = 2 * k;
        let out_row = 2 * k + 1;

        for j in 0..16 {
            tb.set_b8(CpuAes128Columns::DATA + j, in_row, Block8(data_in[j]))
                .unwrap();
            tb.set_b8(CpuAes128Columns::DATA + j, out_row, Block8(data_out[j]))
                .unwrap();
            tb.set_b8(CpuAes128Columns::KEY + j, in_row, Block8(FIPS128_KEY[j]))
                .unwrap();
        }

        tb.set_bit(CpuAes128Columns::SELECTOR, in_row, Bit::ONE)
            .unwrap();
        tb.set_bit(CpuAes128Columns::SELECTOR, out_row, Bit::ONE)
            .unwrap();
        tb.set_bit(CpuAes128Columns::KEY_SELECTOR, in_row, Bit::ONE)
            .unwrap();
    }

    tb.build()
}

pub fn build_cpu_trace_256(data_in: &[u8; 16], data_out: &[u8; 16]) -> ColumnTrace {
    let num_vars = CPU_ROWS.trailing_zeros() as usize;
    let mut tb = TraceBuilder::new(&CpuAes256Columns::build_layout(), num_vars).unwrap();

    for j in 0..16 {
        tb.set_b8(CpuAes256Columns::DATA + j, IN_ROW, Block8(data_in[j]))
            .unwrap();
        tb.set_b8(CpuAes256Columns::DATA + j, OUT_ROW, Block8(data_out[j]))
            .unwrap();
    }

    for (j, &byte) in FIPS256_KEY.iter().enumerate() {
        tb.set_b8(CpuAes256Columns::KEY + j, IN_ROW, Block8(byte))
            .unwrap();
    }

    tb.set_bit(CpuAes256Columns::SELECTOR, IN_ROW, Bit::ONE)
        .unwrap();
    tb.set_bit(CpuAes256Columns::SELECTOR, OUT_ROW, Bit::ONE)
        .unwrap();
    tb.set_bit(CpuAes256Columns::KEY_SELECTOR, IN_ROW, Bit::ONE)
        .unwrap();

    tb.build()
}

// ===============================================
// Trace cell access
// ===============================================

pub fn b8_at(trace: &ColumnTrace, col: usize, row: usize) -> u8 {
    match &trace.columns[col] {
        TraceColumn::B8(data) => data[row].to_tower().0,
        _ => panic!("expected B8 column at {col}"),
    }
}

pub fn set_b8(trace: &mut ColumnTrace, col: usize, row: usize, val: u8) {
    match &mut trace.columns[col] {
        TraceColumn::B8(data) => data[row] = Block8(val).to_hardware(),
        _ => panic!("expected B8 column at {col}"),
    }
}

pub fn xor_b8(trace: &mut ColumnTrace, col: usize, row: usize, mask: u8) {
    let original = b8_at(trace, col, row);
    set_b8(trace, col, row, original ^ mask);
}

pub fn set_b16(trace: &mut ColumnTrace, col: usize, row: usize, val: u16) {
    match &mut trace.columns[col] {
        TraceColumn::B16(data) => data[row] = Block16(val).to_hardware(),
        _ => panic!("expected B16 column at {col}"),
    }
}

pub fn set_bit(trace: &mut ColumnTrace, col: usize, row: usize, val: Bit) {
    match &mut trace.columns[col] {
        TraceColumn::Bit(data) => data[row] = val,
        _ => panic!("expected Bit column at {col}"),
    }
}

fn set_b64(trace: &mut ColumnTrace, col: usize, row: usize, val: u64) {
    match &mut trace.columns[col] {
        TraceColumn::B64(data) => data[row] = Block64(val).to_hardware(),
        _ => panic!("expected B64 column at {col}"),
    }
}

pub fn copy_b8_block(trace: &mut ColumnTrace, base: usize, len: usize, src: usize, dst: usize) {
    for j in 0..len {
        let value = b8_at(trace, base + j, src);
        set_b8(trace, base + j, dst, value);
    }
}

pub fn deactivate_rom(rom: &mut ColumnTrace, rows: core::ops::Range<usize>) {
    for row in rows {
        set_bit(rom, PhysSboxRomColumns::P_SELECTOR, row, Bit::ZERO);

        for j in 0..2 {
            set_b64(rom, PhysSboxRomColumns::P_INV + j, row, 0);
        }

        for j in 0..16 {
            set_bit(rom, PhysSboxRomColumns::P_Z + j, row, Bit::ZERO);
        }
    }
}

// ===============================================
// Verdict oracles
// ===============================================

pub fn prove_and_verify<P: Program<F>>(
    air: &P,
    cpu_trace: ColumnTrace,
    chiplet_traces: Vec<ColumnTrace>,
) -> Result<bool, String> {
    let instance = ProgramInstance::new(CPU_ROWS, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(chiplet_traces);

    let config = Config {
        zero_knowledge: true,
        ..Config::dev()
    };

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    let proof = prove(
        TRANSCRIPT_LABEL,
        air,
        &instance,
        &witness,
        &config,
        blinding_seed,
        None,
    )
    .map_err(|e| format!("prover: {e:?}"))?;

    let mut vt = Transcript::<H>::new(TRANSCRIPT_LABEL);
    let pinned_id = program_id(air).unwrap();

    HekateVerifier::<F, H>::verify(&pinned_id, air, &instance, &proof, &mut vt, &config)
        .map_err(|e| format!("verifier: {e:?}"))
}

/// A violation here would make the verdict
/// attributable to the AIR rather than the bus.
pub fn assert_air_clean<P: Program<F>>(air: &P, cpu: &ColumnTrace, chiplets: &[ColumnTrace]) {
    let instance = ProgramInstance::new(CPU_ROWS, vec![]);
    let witness = ProgramWitness::new(cpu.clone()).with_chiplets(chiplets.to_vec());
    let report = preflight(air, &instance, &witness).unwrap();

    assert!(
        report.constraint_violations.is_empty() && report.boundary_violations.is_empty(),
        "{report}"
    );
}

/// Pins the rejection to a constraint root or schedule pin;
/// a prover that failed for an unrelated reason cannot read green.
pub fn assert_air_violated<P: Program<F>>(air: &P, cpu: &ColumnTrace, chiplets: &[ColumnTrace]) {
    let instance = ProgramInstance::new(CPU_ROWS, vec![]);
    let witness = ProgramWitness::new(cpu.clone()).with_chiplets(chiplets.to_vec());
    let report = preflight(air, &instance, &witness).unwrap();

    assert!(
        !report.constraint_violations.is_empty() || !report.fixed_column_violations.is_empty(),
        "{report}"
    );
}
