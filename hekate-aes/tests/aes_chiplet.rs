// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

mod common;

use common::{
    AES_ROWS, CPU_ROWS, FIPS128_CIPHER, FIPS256_CIPHER, HOST, IN_ROW, OUT_ROW, ROM, ROUND,
    SBOX_128, SBOX_256, assert_rejected, assert_verifies, build_cpu_trace_128, build_cpu_trace_256,
    fips_call_128, fips_call_256, make_program_128, make_program_256, whitened_128, whitened_256,
};
use hekate_aes::{
    Aes128Columns, Aes256Columns, AesRound128Air, AesRound256Air, CpuAes128Columns,
    CpuAes256Columns, PhysAes128Columns, PhysAes256Columns, PhysSboxRomColumns, SboxRomColumns,
};
use hekate_core::trace::ColumnTrace;
use hekate_math::{Bit, TowerField};
use hekate_program::{ProgramInstance, ProgramWitness};
use hekate_scribble::{MutationKind, ScribbleConfig, assert_all_caught_all_targets};
use hekate_testkit::cell::{
    b8_at, bit_rows, copy_row, set_bit, swap_b8_cells, swap_rows, xor_b8, xor_b16, xor_b64,
};
use hekate_testkit::fault::{Check, Fault};

// =================================================================
// AES-128 Prove + Verify
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn aes_128_e2e() {
    let air = make_program_128(AES_ROWS, 1);
    let traces = air.aes.generate_traces(&[fips_call_128()]).unwrap();
    let cpu = build_cpu_trace_128(&[(whitened_128(), FIPS128_CIPHER)]);

    assert_verifies(&air.program, cpu, traces);
}

// =================================================================
// AES-128 Adversarial Harness
// =================================================================

fn assert_128_rejected<T>(fault: Fault<'_>, tamper: T)
where
    T: FnOnce(&mut [ColumnTrace], &mut ColumnTrace),
{
    let air = make_program_128(AES_ROWS, 1);
    let traces = air.aes.generate_traces(&[fips_call_128()]).unwrap();
    let cpu = build_cpu_trace_128(&[(whitened_128(), FIPS128_CIPHER)]);

    let mut forged_traces = traces.clone();
    let mut forged_cpu = cpu.clone();

    tamper(&mut forged_traces, &mut forged_cpu);

    assert_rejected(
        &air.program,
        (cpu, traces),
        (forged_cpu, forged_traces),
        fault,
    );
}

// =================================================================
// Helpers
// =================================================================

fn first_nonzero_rom_input(rom: &ColumnTrace) -> usize {
    bit_rows(rom, PhysSboxRomColumns::P_SELECTOR)
        .into_iter()
        .find(|&r| b8_at(rom, PhysSboxRomColumns::P_INPUT, r) != 0)
        .unwrap()
}

// =================================================================
// AES-128 Exploit Tests
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_sbox_substitution() {
    let fault = Fault {
        constraint_tables: &[ROUND],
        buses: &[SBOX_128],
        ..Fault::NONE
    };

    assert_128_rejected(fault, |traces, _| {
        xor_b8(&mut traces[0], PhysAes128Columns::P_SBOX_OUT, 0, 0x01);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_round_key_swap() {
    let fault = Fault {
        constraint_tables: &[ROUND],
        ..Fault::NONE
    };

    assert_128_rejected(fault, |traces, _| {
        for j in 0..16 {
            swap_rows(&mut traces[0], PhysAes128Columns::P_ROUND_KEY + j, 0, 1);
        }
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_mixcol_bypass() {
    let fault = Fault {
        constraint_tables: &[ROUND],
        buses: &[SBOX_128],
        ..Fault::NONE
    };

    assert_128_rejected(fault, |traces, _| {
        xor_b8(&mut traces[0], PhysAes128Columns::P_STATE_IN, 1, 0x01);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_shiftrows_bypass() {
    let fault = Fault {
        constraint_tables: &[ROUND],
        buses: &[SBOX_128],
        ..Fault::NONE
    };

    assert_128_rejected(fault, |traces, _| {
        swap_b8_cells(
            &mut traces[0],
            PhysAes128Columns::P_SBOX_OUT + 1,
            PhysAes128Columns::P_SBOX_OUT + 5,
            0,
        );
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_round_skip() {
    let fault = Fault {
        pin_columns: &[(ROUND, Aes128Columns::S_ROUND)],
        first_check: Some(Check::Zerocheck),
        ..Fault::NONE
    };

    assert_128_rejected(fault, |traces, _| {
        set_bit(&mut traces[0], PhysAes128Columns::P_S_ROUND, 4, Bit::ZERO);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_ghost_injection() {
    let fault = Fault {
        pin_columns: &[(ROUND, Aes128Columns::S_ROUND)],
        first_check: Some(Check::Zerocheck),
        ..Fault::NONE
    };

    assert_128_rejected(fault, |traces, _| {
        set_bit(&mut traces[0], PhysAes128Columns::P_S_ROUND, 12, Bit::ONE);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_io_bus_unbind() {
    let fault = Fault {
        constraint_tables: &[ROUND],
        buses: &[AesRound128Air::LINK_BUS_ID],
        ..Fault::NONE
    };

    assert_128_rejected(fault, |traces, _| {
        xor_b8(&mut traces[0], PhysAes128Columns::P_STATE_IN, 10, 0x01);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_cpu_io_corruption() {
    let fault = Fault {
        buses: &[AesRound128Air::LINK_BUS_ID],
        ..Fault::NONE
    };

    assert_128_rejected(fault, |_, cpu| {
        xor_b8(cpu, CpuAes128Columns::DATA, IN_ROW, 0x01);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes128_link_input_replayed_as_output_rejected() {
    let fault = Fault {
        buses: &[AesRound128Air::LINK_BUS_ID],
        ..Fault::NONE
    };

    assert_128_rejected(fault, |_, cpu| {
        copy_row(
            cpu,
            CpuAes128Columns::DATA..CpuAes128Columns::DATA + 16,
            IN_ROW,
            OUT_ROW,
        );
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes128_key_in_duplicate_cpu_request_rejected() {
    let fault = Fault {
        pin_columns: &[(HOST, CpuAes128Columns::KEY_SELECTOR)],
        buses: &[AesRound128Air::KEY_BUS_ID],
        first_check: Some(Check::FixedColumn),
        ..Fault::NONE
    };

    assert_128_rejected(fault, |_, cpu| {
        copy_row(
            cpu,
            CpuAes128Columns::KEY..CpuAes128Columns::KEY + 16,
            IN_ROW,
            2,
        );
        set_bit(cpu, CpuAes128Columns::KEY_SELECTOR, 2, Bit::ONE);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes128_sbox_phantom_active_without_link_rejected() {
    let fault = Fault {
        pin_columns: &[(ROUND, Aes128Columns::S_ACTIVE)],
        buses: &[SBOX_128],
        first_check: Some(Check::FixedColumn),
        ..Fault::NONE
    };

    assert_128_rejected(fault, |traces, _| {
        let aes = &mut traces[0];

        set_bit(aes, PhysAes128Columns::P_S_ACTIVE, 15, Bit::ONE);
        set_bit(aes, PhysAes128Columns::P_S_INPUT, 15, Bit::ZERO);
        set_bit(aes, PhysAes128Columns::P_S_IN_OUT, 15, Bit::ZERO);
    });
}

// =================================================================
// Round Index Exploits
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_round_index_shift() {
    let fault = Fault {
        pin_columns: &[
            (ROUND, Aes128Columns::ROUND_BITS + 2),
            (ROUND, Aes128Columns::ROUND_BITS + 3),
        ],
        first_check: Some(Check::Zerocheck),
        ..Fault::NONE
    };

    assert_128_rejected(fault, |traces, _| {
        xor_b16(&mut traces[0], PhysAes128Columns::P_ROUND_IDX, 3, 0x000C);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_round_index_init() {
    let fault = Fault {
        pin_columns: &[
            (ROUND, Aes128Columns::ROUND_BITS),
            (ROUND, Aes128Columns::ROUND_BITS + 1),
        ],
        first_check: Some(Check::Zerocheck),
        ..Fault::NONE
    };

    assert_128_rejected(fault, |traces, _| {
        xor_b16(&mut traces[0], PhysAes128Columns::P_ROUND_IDX, 0, 0x0003);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_round_index_final() {
    let fault = Fault {
        pin_columns: &[
            (ROUND, Aes128Columns::ROUND_BITS + 8),
            (ROUND, Aes128Columns::ROUND_BITS + 9),
        ],
        first_check: Some(Check::FixedColumn),
        ..Fault::NONE
    };

    assert_128_rejected(fault, |traces, _| {
        xor_b16(&mut traces[0], PhysAes128Columns::P_ROUND_IDX, 9, 0x0300);
    });
}

/// A bit above the active range is pinned to zero,
/// an out-of-range index cannot hide the row's activity.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_round_index_out_of_range() {
    let fault = Fault {
        pin_columns: &[(ROUND, Aes128Columns::ROUND_BITS + 15)],
        first_check: Some(Check::FixedColumn),
        ..Fault::NONE
    };

    assert_128_rejected(fault, |traces, _| {
        xor_b16(&mut traces[0], PhysAes128Columns::P_ROUND_IDX, 5, 0x8000);
    });
}

// =================================================================
// Selector Integrity Exploits
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_s_active_desync() {
    let fault = Fault {
        pin_columns: &[(ROUND, Aes128Columns::S_ACTIVE)],
        buses: &[SBOX_128],
        first_check: Some(Check::FixedColumn),
        ..Fault::NONE
    };

    assert_128_rejected(fault, |traces, _| {
        set_bit(&mut traces[0], PhysAes128Columns::P_S_ACTIVE, 3, Bit::ZERO);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_s_input_forgery() {
    let fault = Fault {
        pin_columns: &[(ROUND, Aes128Columns::S_INPUT)],
        buses: &[AesRound128Air::KEY_BUS_ID],
        first_check: Some(Check::Zerocheck),
        ..Fault::NONE
    };

    assert_128_rejected(fault, |traces, _| {
        set_bit(&mut traces[0], PhysAes128Columns::P_S_INPUT, 3, Bit::ONE);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_s_final_on_round_row() {
    let fault = Fault {
        pin_columns: &[(ROUND, Aes128Columns::S_FINAL)],
        first_check: Some(Check::Zerocheck),
        ..Fault::NONE
    };

    assert_128_rejected(fault, |traces, _| {
        set_bit(&mut traces[0], PhysAes128Columns::P_S_FINAL, 3, Bit::ONE);
    });
}

// =================================================================
// S-box ROM Algebraic Exploits
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_sbox_inv_tamper() {
    let fault = Fault {
        constraint_tables: &[ROM],
        ..Fault::NONE
    };

    assert_128_rejected(fault, |traces, _| {
        xor_b64(&mut traces[1], PhysSboxRomColumns::P_INV, 0, 0x01);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_sbox_z_forgery() {
    let fault = Fault {
        constraint_tables: &[ROM],
        ..Fault::NONE
    };

    assert_128_rejected(fault, |traces, _| {
        let rom = &mut traces[1];
        let row = first_nonzero_rom_input(rom);

        set_bit(rom, PhysSboxRomColumns::P_Z, row, Bit::ONE);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_sbox_rom_deactivation() {
    let fault = Fault {
        pin_columns: &[(ROM, SboxRomColumns::SELECTOR)],
        buses: &[SBOX_128],
        first_check: Some(Check::Zerocheck),
        ..Fault::NONE
    };

    assert_128_rejected(fault, |traces, _| {
        let rom = &mut traces[1];
        let active = bit_rows(rom, PhysSboxRomColumns::P_SELECTOR);

        set_bit(rom, PhysSboxRomColumns::P_SELECTOR, active[0], Bit::ZERO);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_sbox_output_forgery() {
    let fault = Fault {
        constraint_tables: &[ROM],
        buses: &[SBOX_128],
        ..Fault::NONE
    };

    assert_128_rejected(fault, |traces, _| {
        let rom = &mut traces[1];
        let active = bit_rows(rom, PhysSboxRomColumns::P_SELECTOR);

        xor_b8(rom, PhysSboxRomColumns::P_OUTPUT, active[0], 0xFF);
    });
}

// =================================================================
// Ghost Protocol Exploits
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_ghost_protocol_mid_block() {
    let fault = Fault {
        pin_columns: &[
            (ROUND, Aes128Columns::S_ROUND),
            (ROUND, Aes128Columns::S_ACTIVE),
            (ROM, SboxRomColumns::SELECTOR),
        ],
        first_check: Some(Check::Zerocheck),
        ..Fault::NONE
    };

    assert_128_rejected(fault, |traces, _| {
        set_bit(&mut traces[0], PhysAes128Columns::P_S_ROUND, 4, Bit::ZERO);
        set_bit(&mut traces[0], PhysAes128Columns::P_S_ACTIVE, 4, Bit::ZERO);
        set_bit(&mut traces[1], PhysSboxRomColumns::P_SELECTOR, 4, Bit::ZERO);
    });
}

// =================================================================
// Key Schedule Exploits
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_k0_init_cascade() {
    let fault = Fault {
        constraint_tables: &[ROUND],
        ..Fault::NONE
    };

    assert_128_rejected(fault, |traces, cpu| {
        xor_b8(&mut traces[0], PhysAes128Columns::P_K0, 0, 0x01);
        xor_b8(cpu, CpuAes128Columns::KEY, IN_ROW, 0x01);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_k0_bus_unbind() {
    let fault = Fault {
        buses: &[AesRound128Air::KEY_BUS_ID],
        ..Fault::NONE
    };

    assert_128_rejected(fault, |_, cpu| {
        xor_b8(cpu, CpuAes128Columns::KEY, IN_ROW, 0x01);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_ks_sub_tamper() {
    let fault = Fault {
        constraint_tables: &[ROUND],
        ..Fault::NONE
    };

    assert_128_rejected(fault, |traces, _| {
        xor_b8(&mut traces[0], PhysAes128Columns::P_KS_SUB, 0, 0x01);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_ks_inv_tamper() {
    let fault = Fault {
        constraint_tables: &[ROUND],
        ..Fault::NONE
    };

    assert_128_rejected(fault, |traces, _| {
        xor_b8(&mut traces[0], PhysAes128Columns::P_KS_INV, 0, 0x01);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_k0_sub_tamper() {
    let fault = Fault {
        constraint_tables: &[ROUND],
        ..Fault::NONE
    };

    assert_128_rejected(fault, |traces, _| {
        xor_b8(&mut traces[0], PhysAes128Columns::P_K0_SUB, 0, 0x01);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_128_k0_inv_tamper() {
    let fault = Fault {
        constraint_tables: &[ROUND],
        ..Fault::NONE
    };

    assert_128_rejected(fault, |traces, _| {
        xor_b8(&mut traces[0], PhysAes128Columns::P_K0_INV, 0, 0x01);
    });
}

// =================================================================
// AES-256 Prove + Verify
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn aes_256_e2e() {
    let air = make_program_256(AES_ROWS, 1);
    let traces = air.aes.generate_traces(&[fips_call_256()]).unwrap();
    let cpu = build_cpu_trace_256(&whitened_256(), &FIPS256_CIPHER);

    assert_verifies(&air.program, cpu, traces);
}

// =================================================================
// AES-256 Adversarial Harness
// =================================================================

fn assert_256_rejected<T>(fault: Fault<'_>, tamper: T)
where
    T: FnOnce(&mut [ColumnTrace], &mut ColumnTrace),
{
    let air = make_program_256(AES_ROWS, 1);
    let traces = air.aes.generate_traces(&[fips_call_256()]).unwrap();
    let cpu = build_cpu_trace_256(&whitened_256(), &FIPS256_CIPHER);

    let mut forged_traces = traces.clone();
    let mut forged_cpu = cpu.clone();

    tamper(&mut forged_traces, &mut forged_cpu);

    assert_rejected(
        &air.program,
        (cpu, traces),
        (forged_cpu, forged_traces),
        fault,
    );
}

// =================================================================
// AES-256 Exploit Tests
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_sbox_substitution() {
    let fault = Fault {
        constraint_tables: &[ROUND],
        buses: &[SBOX_256],
        ..Fault::NONE
    };

    assert_256_rejected(fault, |traces, _| {
        xor_b8(&mut traces[0], PhysAes256Columns::P_SBOX_OUT, 0, 0x01);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_round_key_swap() {
    let fault = Fault {
        constraint_tables: &[ROUND],
        ..Fault::NONE
    };

    assert_256_rejected(fault, |traces, _| {
        for j in 0..16 {
            swap_rows(&mut traces[0], PhysAes256Columns::P_ROUND_KEY + j, 0, 1);
        }
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_mixcol_bypass() {
    let fault = Fault {
        constraint_tables: &[ROUND],
        buses: &[SBOX_256],
        ..Fault::NONE
    };

    assert_256_rejected(fault, |traces, _| {
        xor_b8(&mut traces[0], PhysAes256Columns::P_STATE_IN, 1, 0x01);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_round_skip() {
    let fault = Fault {
        pin_columns: &[(ROUND, Aes256Columns::S_ROUND)],
        first_check: Some(Check::Zerocheck),
        ..Fault::NONE
    };

    assert_256_rejected(fault, |traces, _| {
        set_bit(&mut traces[0], PhysAes256Columns::P_S_ROUND, 4, Bit::ZERO);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_ghost_injection() {
    let fault = Fault {
        pin_columns: &[(ROUND, Aes256Columns::S_ROUND)],
        first_check: Some(Check::Zerocheck),
        ..Fault::NONE
    };

    assert_256_rejected(fault, |traces, _| {
        set_bit(&mut traces[0], PhysAes256Columns::P_S_ROUND, 15, Bit::ONE);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_io_bus_unbind() {
    let fault = Fault {
        constraint_tables: &[ROUND],
        buses: &[AesRound256Air::LINK_BUS_ID],
        ..Fault::NONE
    };

    assert_256_rejected(fault, |traces, _| {
        xor_b8(&mut traces[0], PhysAes256Columns::P_STATE_IN, 14, 0x01);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_cpu_io_corruption() {
    let fault = Fault {
        buses: &[AesRound256Air::LINK_BUS_ID],
        ..Fault::NONE
    };

    assert_256_rejected(fault, |_, cpu| {
        xor_b8(cpu, CpuAes256Columns::DATA, IN_ROW, 0x01);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_link_input_replayed_as_output_rejected() {
    let fault = Fault {
        buses: &[AesRound256Air::LINK_BUS_ID],
        ..Fault::NONE
    };

    assert_256_rejected(fault, |_, cpu| {
        copy_row(
            cpu,
            CpuAes256Columns::DATA..CpuAes256Columns::DATA + 16,
            IN_ROW,
            OUT_ROW,
        );
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_key_in_duplicate_cpu_request_rejected() {
    let fault = Fault {
        pin_columns: &[(HOST, CpuAes256Columns::KEY_SELECTOR)],
        buses: &[AesRound256Air::KEY_BUS_ID],
        first_check: Some(Check::FixedColumn),
        ..Fault::NONE
    };

    assert_256_rejected(fault, |_, cpu| {
        copy_row(
            cpu,
            CpuAes256Columns::KEY..CpuAes256Columns::KEY + 32,
            IN_ROW,
            2,
        );
        set_bit(cpu, CpuAes256Columns::KEY_SELECTOR, 2, Bit::ONE);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_sbox_phantom_active_without_link_rejected() {
    let fault = Fault {
        pin_columns: &[(ROUND, Aes256Columns::S_ACTIVE)],
        buses: &[SBOX_256],
        first_check: Some(Check::FixedColumn),
        ..Fault::NONE
    };

    assert_256_rejected(fault, |traces, _| {
        let aes = &mut traces[0];

        set_bit(aes, PhysAes256Columns::P_S_ACTIVE, 15, Bit::ONE);
        set_bit(aes, PhysAes256Columns::P_S_INPUT, 15, Bit::ZERO);
        set_bit(aes, PhysAes256Columns::P_S_IN_OUT, 15, Bit::ZERO);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_round_index_shift() {
    let fault = Fault {
        pin_columns: &[
            (ROUND, Aes256Columns::ROUND_BITS + 2),
            (ROUND, Aes256Columns::ROUND_BITS + 3),
        ],
        first_check: Some(Check::Zerocheck),
        ..Fault::NONE
    };

    assert_256_rejected(fault, |traces, _| {
        xor_b16(&mut traces[0], PhysAes256Columns::P_ROUND_IDX, 3, 0x000C);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_round_index_init() {
    let fault = Fault {
        pin_columns: &[
            (ROUND, Aes256Columns::ROUND_BITS),
            (ROUND, Aes256Columns::ROUND_BITS + 1),
        ],
        first_check: Some(Check::Zerocheck),
        ..Fault::NONE
    };

    assert_256_rejected(fault, |traces, _| {
        xor_b16(&mut traces[0], PhysAes256Columns::P_ROUND_IDX, 0, 0x0003);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_round_index_final() {
    let fault = Fault {
        pin_columns: &[
            (ROUND, Aes256Columns::ROUND_BITS + 12),
            (ROUND, Aes256Columns::ROUND_BITS + 13),
        ],
        first_check: Some(Check::Zerocheck),
        ..Fault::NONE
    };

    assert_256_rejected(fault, |traces, _| {
        xor_b16(&mut traces[0], PhysAes256Columns::P_ROUND_IDX, 13, 0x3000);
    });
}

/// A bit above the active range is pinned to zero;
/// an out-of-range index cannot hide the row's activity.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_round_index_out_of_range() {
    let fault = Fault {
        pin_columns: &[(ROUND, Aes256Columns::ROUND_BITS + 15)],
        first_check: Some(Check::FixedColumn),
        ..Fault::NONE
    };

    assert_256_rejected(fault, |traces, _| {
        xor_b16(&mut traces[0], PhysAes256Columns::P_ROUND_IDX, 5, 0x8000);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_s_active_desync() {
    let fault = Fault {
        pin_columns: &[(ROUND, Aes256Columns::S_ACTIVE)],
        buses: &[SBOX_256],
        first_check: Some(Check::FixedColumn),
        ..Fault::NONE
    };

    assert_256_rejected(fault, |traces, _| {
        set_bit(&mut traces[0], PhysAes256Columns::P_S_ACTIVE, 3, Bit::ZERO);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_s_input_forgery() {
    let fault = Fault {
        pin_columns: &[(ROUND, Aes256Columns::S_INPUT)],
        buses: &[AesRound256Air::KEY_BUS_ID],
        first_check: Some(Check::Zerocheck),
        ..Fault::NONE
    };

    assert_256_rejected(fault, |traces, _| {
        set_bit(&mut traces[0], PhysAes256Columns::P_S_INPUT, 3, Bit::ONE);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_s_final_on_round_row() {
    let fault = Fault {
        pin_columns: &[(ROUND, Aes256Columns::S_FINAL)],
        first_check: Some(Check::Zerocheck),
        ..Fault::NONE
    };

    assert_256_rejected(fault, |traces, _| {
        set_bit(&mut traces[0], PhysAes256Columns::P_S_FINAL, 3, Bit::ONE);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_ks_sub_tamper() {
    let fault = Fault {
        constraint_tables: &[ROUND],
        ..Fault::NONE
    };

    assert_256_rejected(fault, |traces, _| {
        xor_b8(&mut traces[0], PhysAes256Columns::P_KS_SUB, 0, 0x01);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_ks_inv_tamper() {
    let fault = Fault {
        constraint_tables: &[ROUND],
        ..Fault::NONE
    };

    assert_256_rejected(fault, |traces, _| {
        xor_b8(&mut traces[0], PhysAes256Columns::P_KS_INV, 0, 0x01);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_k0_bus_unbind() {
    let fault = Fault {
        buses: &[AesRound256Air::KEY_BUS_ID],
        ..Fault::NONE
    };

    assert_256_rejected(fault, |_, cpu| {
        xor_b8(cpu, CpuAes256Columns::KEY, IN_ROW, 0x01);
    });
}

// =================================================================
// AES-256 S-box ROM / ShiftRows Exploits
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_shiftrows_bypass() {
    let fault = Fault {
        constraint_tables: &[ROUND],
        buses: &[SBOX_256],
        ..Fault::NONE
    };

    assert_256_rejected(fault, |traces, _| {
        swap_b8_cells(
            &mut traces[0],
            PhysAes256Columns::P_SBOX_OUT + 1,
            PhysAes256Columns::P_SBOX_OUT + 5,
            0,
        );
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_sbox_inv_tamper() {
    let fault = Fault {
        constraint_tables: &[ROM],
        ..Fault::NONE
    };

    assert_256_rejected(fault, |traces, _| {
        xor_b64(&mut traces[1], PhysSboxRomColumns::P_INV, 0, 0x01);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_sbox_z_forgery() {
    let fault = Fault {
        constraint_tables: &[ROM],
        ..Fault::NONE
    };

    assert_256_rejected(fault, |traces, _| {
        let rom = &mut traces[1];
        let row = first_nonzero_rom_input(rom);

        set_bit(rom, PhysSboxRomColumns::P_Z, row, Bit::ONE);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_sbox_rom_deactivation() {
    let fault = Fault {
        pin_columns: &[(ROM, SboxRomColumns::SELECTOR)],
        buses: &[SBOX_256],
        first_check: Some(Check::Zerocheck),
        ..Fault::NONE
    };

    assert_256_rejected(fault, |traces, _| {
        let rom = &mut traces[1];
        let active = bit_rows(rom, PhysSboxRomColumns::P_SELECTOR);

        set_bit(rom, PhysSboxRomColumns::P_SELECTOR, active[0], Bit::ZERO);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_sbox_output_forgery() {
    let fault = Fault {
        constraint_tables: &[ROM],
        buses: &[SBOX_256],
        ..Fault::NONE
    };

    assert_256_rejected(fault, |traces, _| {
        let rom = &mut traces[1];
        let active = bit_rows(rom, PhysSboxRomColumns::P_SELECTOR);

        xor_b8(rom, PhysSboxRomColumns::P_OUTPUT, active[0], 0xFF);
    });
}

// =================================================================
// AES-256–Specific Exploits
// =================================================================

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_key_aux_tamper() {
    let fault = Fault {
        constraint_tables: &[ROUND],
        ..Fault::NONE
    };

    assert_256_rejected(fault, |traces, _| {
        xor_b8(&mut traces[0], PhysAes256Columns::P_KEY_AUX, 1, 0x01);
    });
}

/// Moving the index two places keeps the
/// RotWord/direct parity and changes only Rcon.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_round_index_parity_preserved() {
    let fault = Fault {
        pin_columns: &[
            (ROUND, Aes256Columns::ROUND_BITS + 2),
            (ROUND, Aes256Columns::ROUND_BITS + 4),
        ],
        first_check: Some(Check::Zerocheck),
        ..Fault::NONE
    };

    assert_256_rejected(fault, |traces, _| {
        xor_b16(&mut traces[0], PhysAes256Columns::P_ROUND_IDX, 2, 0x0014);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_ks_input_tamper() {
    let fault = Fault {
        constraint_tables: &[ROUND],
        ..Fault::NONE
    };

    assert_256_rejected(fault, |traces, _| {
        xor_b8(&mut traces[0], PhysAes256Columns::P_KS_INPUT, 1, 0x01);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_init_rk_tamper() {
    let fault = Fault {
        constraint_tables: &[ROUND],
        ..Fault::NONE
    };

    assert_256_rejected(fault, |traces, _| {
        xor_b8(&mut traces[0], PhysAes256Columns::P_ROUND_KEY, 0, 0x01);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_init_key_aux_tamper() {
    let fault = Fault {
        constraint_tables: &[ROUND],
        ..Fault::NONE
    };

    assert_256_rejected(fault, |traces, _| {
        xor_b8(&mut traces[0], PhysAes256Columns::P_KEY_AUX, 0, 0x01);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_aes256_ghost_protocol_mid_block() {
    let fault = Fault {
        pin_columns: &[
            (ROUND, Aes256Columns::S_ROUND),
            (ROUND, Aes256Columns::S_ACTIVE),
            (ROM, SboxRomColumns::SELECTOR),
        ],
        first_check: Some(Check::Zerocheck),
        ..Fault::NONE
    };

    assert_256_rejected(fault, |traces, _| {
        set_bit(&mut traces[0], PhysAes256Columns::P_S_ROUND, 4, Bit::ZERO);
        set_bit(&mut traces[0], PhysAes256Columns::P_S_ACTIVE, 4, Bit::ZERO);
        set_bit(&mut traces[1], PhysSboxRomColumns::P_SELECTOR, 4, Bit::ZERO);
    });
}

#[test]
fn scribble_aes128_flip_selector_caught() {
    let air = make_program_128(AES_ROWS, 1);
    let traces = air.aes.generate_traces(&[fips_call_128()]).unwrap();
    let cpu = build_cpu_trace_128(&[(whitened_128(), FIPS128_CIPHER)]);

    let instance = ProgramInstance::new(CPU_ROWS, vec![]);
    let witness = ProgramWitness::new(cpu).with_chiplets(traces);

    assert_all_caught_all_targets(
        &air.program,
        &instance,
        &witness,
        ScribbleConfig::default()
            .mutations([MutationKind::FlipSelector])
            .cases(64),
    );
}

#[test]
fn scribble_aes256_flip_selector_caught() {
    let air = make_program_256(AES_ROWS, 1);
    let traces = air.aes.generate_traces(&[fips_call_256()]).unwrap();
    let cpu = build_cpu_trace_256(&whitened_256(), &FIPS256_CIPHER);

    let instance = ProgramInstance::new(CPU_ROWS, vec![]);
    let witness = ProgramWitness::new(cpu).with_chiplets(traces);

    assert_all_caught_all_targets(
        &air.program,
        &instance,
        &witness,
        ScribbleConfig::default()
            .mutations([MutationKind::FlipSelector])
            .cases(64),
    );
}
