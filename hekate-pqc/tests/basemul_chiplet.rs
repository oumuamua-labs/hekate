// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! End-to-end prove/verify test for the
//! Basemul chiplet (isolated).

use hekate_core::config::Config;
use hekate_core::trace::TraceBuilder;
use hekate_core::trace::TraceColumn;
use hekate_crypto::DefaultHasher;
use hekate_crypto::transcript::Transcript;
use hekate_math::Block128;
use hekate_math::{Bit, Block32, TowerField};
use hekate_pqc::basemul::{BasemulChiplet, BasemulOp, CpuBasemulColumns, generate_basemul_trace};
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::digest::program_id;
use hekate_program::{FixedShape, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_scribble::{MutationKind, ScribbleConfig, assert_all_caught_all_targets};
use hekate_verifier::HekateVerifier;
use rand::{TryRngCore, rngs::OsRng};

type F = Block128;
type H = DefaultHasher;

const Q: u32 = 3329;

fn basemul_test_program(bm_rows: usize, num_ops: usize) -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("BasemulTest", bm_rows).unwrap();
    let cpu = cx.schema(&CpuBasemulColumns::build_layout());

    cx.fix(
        cpu.at(CpuBasemulColumns::SELECTOR),
        FixedShape::Cadence {
            stride: 1,
            count: num_ops,
            origin: 0,
            values: vec![F::ONE],
        },
    );

    cx.bus(BasemulChiplet::BUS_ID, BasemulChiplet::cpu_linking_spec());

    let cs = cx.cs();

    let s = cs.col(CpuBasemulColumns::SELECTOR);
    let not_active = cs.one() - s;

    cs.assert_zero_when(not_active, cs.col(CpuBasemulColumns::BM_A));
    cs.assert_zero_when(not_active, cs.col(CpuBasemulColumns::BM_B));
    cs.assert_zero_when(not_active, cs.col(CpuBasemulColumns::BM_C));
    cs.assert_zero_when(not_active, cs.col(CpuBasemulColumns::BM_IDX));

    cx.attach(ChipletDef::from_air(&BasemulChiplet::new(Q, bm_rows, num_ops)).unwrap());

    cx.compile().unwrap()
}

fn prove_and_verify(ops: &[BasemulOp], label: &str) -> bool {
    let cpu_rows = (ops.len() + 1).next_power_of_two().max(4);
    let bm_rows = cpu_rows;

    let air = basemul_test_program(bm_rows, ops.len());

    let layout = CpuBasemulColumns::build_layout();

    let mut tb = TraceBuilder::new(&layout, cpu_rows.trailing_zeros() as usize).unwrap();

    for (row, op) in ops.iter().enumerate() {
        tb.set_b32(CpuBasemulColumns::BM_A, row, Block32::from(op.a))
            .unwrap();
        tb.set_b32(CpuBasemulColumns::BM_B, row, Block32::from(op.b))
            .unwrap();
        tb.set_b32(CpuBasemulColumns::BM_C, row, Block32::from(op.c))
            .unwrap();
        tb.set_b32(CpuBasemulColumns::BM_IDX, row, Block32::from(op.idx))
            .unwrap();
        tb.set_bit(CpuBasemulColumns::SELECTOR, row, Bit::ONE)
            .unwrap();
    }

    let cpu_trace = tb.build();

    let bm_trace = generate_basemul_trace(Q, ops, bm_rows).expect("trace gen failed");

    let instance = ProgramInstance::new(cpu_rows, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![bm_trace]);

    let config = Config {
        zero_knowledge: true,
        ..Config::dev()
    };

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    let proof = match prove(
        b"Basemul_E2E",
        &air,
        &instance,
        &witness,
        &config,
        blinding_seed,
        None,
    ) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[{label}] Prover failed: {e:?}");
            return false;
        }
    };

    let mut vt = Transcript::<H>::new(b"Basemul_E2E");
    let pinned_id = program_id(&air).unwrap();

    match HekateVerifier::<F, H>::verify(&pinned_id, &air, &instance, &proof, &mut vt, &config) {
        Ok(valid) => {
            if valid {
                eprintln!("[{label}] PASS");
            } else {
                eprintln!("[{label}] FAIL: verifier returned false");
            }

            valid
        }
        Err(e) => {
            eprintln!("[{label}] FAIL: {e:?}");
            false
        }
    }
}

#[test]
fn empty_basemul() {
    assert!(prove_and_verify(&[], "empty"));
}

#[test]
fn single_addition() {
    let ops = vec![BasemulOp {
        a: 1000,
        b: 2000,
        c: 3000,
        idx: 0,
        ram_addr: 0,
    }];
    assert!(prove_and_verify(&ops, "single_add"));
}

#[test]
fn single_overflow() {
    let ops = vec![BasemulOp {
        a: 2000,
        b: 2000,
        c: 4000 % Q,
        idx: 0,
        ram_addr: 0,
    }];
    assert!(prove_and_verify(&ops, "single_overflow"));
}

#[test]
fn subtraction_encoding() {
    // r = 2500 - 1000 = 1500 mod q
    // Encode as: a=1500, b=1000, c=2500
    let ops = vec![BasemulOp {
        a: 1500,
        b: 1000,
        c: 2500,
        idx: 0,
        ram_addr: 0,
    }];
    assert!(prove_and_verify(&ops, "subtraction"));
}

#[test]
fn full_basemul_unit() {
    let a0 = 100u32;
    let a1 = 200;
    let b0 = 300;
    let b1 = 400;
    let zeta = 17u32;

    let p00 = (a0 * b0) % Q;
    let p11z = ((a1 * b1) % Q * zeta) % Q;
    let p01 = (a0 * b1) % Q;
    let p10 = (a1 * b0) % Q;

    let r0 = (p00 + p11z) % Q;
    let r1 = (p01 + p10) % Q;

    let a2 = 500u32;
    let a3 = 600;
    let b2 = 700;
    let b3 = 800;

    let p22 = (a2 * b2) % Q;
    let p33z = ((a3 * b3) % Q * zeta) % Q;
    let r2 = (p22 + Q - p33z) % Q;
    let p23 = (a2 * b3) % Q;
    let p32 = (a3 * b2) % Q;
    let r3 = (p23 + p32) % Q;

    let ops = vec![
        BasemulOp {
            a: p00,
            b: p11z,
            c: r0,
            idx: 0,
            ram_addr: 0,
        },
        BasemulOp {
            a: p01,
            b: p10,
            c: r1,
            idx: 1,
            ram_addr: 0,
        },
        BasemulOp {
            a: r2,
            b: p33z,
            c: p22,
            idx: 2,
            ram_addr: 0,
        },
        BasemulOp {
            a: p23,
            b: p32,
            c: r3,
            idx: 3,
            ram_addr: 0,
        },
    ];
    assert!(prove_and_verify(&ops, "full_basemul_unit"));
}

#[test]
fn boundary_values() {
    let ops = vec![
        BasemulOp {
            a: Q - 1,
            b: Q - 1,
            c: (2 * (Q - 1)) % Q,
            idx: 0,
            ram_addr: 0,
        },
        BasemulOp {
            a: 0,
            b: 0,
            c: 0,
            idx: 1,
            ram_addr: 0,
        },
        BasemulOp {
            a: Q - 1,
            b: 1,
            c: 0,
            idx: 2,
            ram_addr: 0,
        },
    ];
    assert!(prove_and_verify(&ops, "boundary"));
}

// =================================================================
// Adversarial tests
// =================================================================

/// Corrupt c (the modular sum) in the basemul
/// chiplet trace. The mod-add constraint must
/// reject: a + b ≠ c_wrong + flag*q.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn adversarial_corrupted_sum_rejected() {
    let ops = vec![BasemulOp {
        a: 1000,
        b: 2000,
        c: 3000,
        idx: 0,
        ram_addr: 0,
    }];

    let cpu_rows = 4usize;
    let bm_rows = 4;

    let air = basemul_test_program(bm_rows, ops.len());

    let layout = CpuBasemulColumns::build_layout();

    let mut tb = TraceBuilder::new(&layout, cpu_rows.trailing_zeros() as usize).unwrap();

    tb.set_b32(CpuBasemulColumns::BM_A, 0, Block32::from(1000u32))
        .unwrap();
    tb.set_b32(CpuBasemulColumns::BM_B, 0, Block32::from(2000u32))
        .unwrap();

    // CPU side also uses wrong
    // c to avoid GPA mismatch.
    tb.set_b32(CpuBasemulColumns::BM_C, 0, Block32::from(42u32))
        .unwrap();
    tb.set_b32(CpuBasemulColumns::BM_IDX, 0, Block32::from(0u32))
        .unwrap();
    tb.set_bit(CpuBasemulColumns::SELECTOR, 0, Bit::ONE)
        .unwrap();

    let cpu_trace = tb.build();

    let mut bm_trace = generate_basemul_trace(Q, &ops, bm_rows).expect("trace gen");

    // Corrupt bus_c to 42 (wrong sum)
    let bm_layout = hekate_pqc::basemul::BasemulLayout::compute(12);
    if let TraceColumn::B32(ref mut vals) = bm_trace.columns[bm_layout.num_packed_b32_cols + 2] {
        vals[0] = hekate_math::Flat::from_raw(Block32::from(42u32));
    }

    let instance = ProgramInstance::new(cpu_rows, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![bm_trace]);

    let config = Config {
        zero_knowledge: true,
        ..Config::dev()
    };

    let mut seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut seed).unwrap();

    let result = prove(
        b"Basemul_E2E",
        &air,
        &instance,
        &witness,
        &config,
        seed,
        None,
    );

    match result {
        Err(_) => {} // prover caught the corruption
        Ok(proof) => {
            let mut vt = Transcript::<H>::new(b"Basemul_E2E");
            let pinned_id = program_id(&air).unwrap();

            let valid = HekateVerifier::<F, H>::verify(
                &pinned_id, &air, &instance, &proof, &mut vt, &config,
            );

            if let Ok(true) = valid {
                panic!("corrupted sum accepted — mod-add soundness break")
            }
        }
    }
}

/// Set c ≥ q in the basemul trace. The range
/// check constraint (c < q) must reject.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn adversarial_c_out_of_range_rejected() {
    // Honest op:
    // 1000 + 2000 = 3000 mod 3329
    let ops = vec![BasemulOp {
        a: 1000,
        b: 2000,
        c: 3000,
        idx: 0,
        ram_addr: 0,
    }];

    let cpu_rows = 4usize;
    let bm_rows = 4;

    let air = basemul_test_program(bm_rows, ops.len());
    let layout = CpuBasemulColumns::build_layout();

    let mut tb = TraceBuilder::new(&layout, cpu_rows.trailing_zeros() as usize).unwrap();

    // CPU uses c=Q (out of range)
    tb.set_b32(CpuBasemulColumns::BM_A, 0, Block32::from(1000u32))
        .unwrap();
    tb.set_b32(CpuBasemulColumns::BM_B, 0, Block32::from(2000u32))
        .unwrap();
    tb.set_b32(CpuBasemulColumns::BM_C, 0, Block32::from(Q))
        .unwrap();
    tb.set_b32(CpuBasemulColumns::BM_IDX, 0, Block32::from(0u32))
        .unwrap();
    tb.set_bit(CpuBasemulColumns::SELECTOR, 0, Bit::ONE)
        .unwrap();

    let cpu_trace = tb.build();

    let mut bm_trace = generate_basemul_trace(Q, &ops, bm_rows).expect("trace gen");

    // Corrupt bus_c to Q (out of range)
    let bm_layout = hekate_pqc::basemul::BasemulLayout::compute(12);
    if let TraceColumn::B32(ref mut vals) = bm_trace.columns[bm_layout.num_packed_b32_cols + 2] {
        vals[0] = hekate_math::Flat::from_raw(Block32::from(Q));
    }

    let instance = ProgramInstance::new(cpu_rows, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![bm_trace]);

    let config = Config {
        zero_knowledge: true,
        ..Config::dev()
    };

    let mut seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut seed).unwrap();

    let result = prove(
        b"Basemul_E2E",
        &air,
        &instance,
        &witness,
        &config,
        seed,
        None,
    );

    match result {
        Err(_) => {}
        Ok(proof) => {
            let mut vt = Transcript::<H>::new(b"Basemul_E2E");
            let pinned_id = program_id(&air).unwrap();

            let valid = HekateVerifier::<F, H>::verify(
                &pinned_id, &air, &instance, &proof, &mut vt, &config,
            );

            if let Ok(true) = valid {
                panic!("c >= q accepted — range check soundness break")
            }
        }
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn exploit_basemul_duplicate_cpu_request_rejected() {
    let ops = vec![
        BasemulOp {
            a: 100,
            b: 200,
            c: 300,
            idx: 0,
            ram_addr: 0,
        },
        BasemulOp {
            a: 50,
            b: 70,
            c: 120,
            idx: 1,
            ram_addr: 0,
        },
    ];

    let cpu_rows = 4usize;
    let bm_rows = 4;

    let air = basemul_test_program(bm_rows, ops.len());
    let layout = CpuBasemulColumns::build_layout();

    let mut tb = TraceBuilder::new(&layout, cpu_rows.trailing_zeros() as usize).unwrap();

    for (row, op) in ops.iter().enumerate() {
        let (a, b, c, idx) = if row == 1 {
            (ops[0].a, ops[0].b, ops[0].c, ops[0].idx)
        } else {
            (op.a, op.b, op.c, op.idx)
        };

        tb.set_b32(CpuBasemulColumns::BM_A, row, Block32::from(a))
            .unwrap();
        tb.set_b32(CpuBasemulColumns::BM_B, row, Block32::from(b))
            .unwrap();
        tb.set_b32(CpuBasemulColumns::BM_C, row, Block32::from(c))
            .unwrap();
        tb.set_b32(CpuBasemulColumns::BM_IDX, row, Block32::from(idx))
            .unwrap();
        tb.set_bit(CpuBasemulColumns::SELECTOR, row, Bit::ONE)
            .unwrap();
    }

    let cpu_trace = tb.build();
    let bm_trace = generate_basemul_trace(Q, &ops, bm_rows).expect("trace gen");

    let instance = ProgramInstance::new(cpu_rows, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![bm_trace]);

    let config = Config {
        zero_knowledge: true,
        ..Config::dev()
    };

    let mut seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut seed).unwrap();

    let result = prove(
        b"Basemul_E2E",
        &air,
        &instance,
        &witness,
        &config,
        seed,
        None,
    );

    match result {
        Err(_) => {}
        Ok(proof) => {
            let mut vt = Transcript::<H>::new(b"Basemul_E2E");
            let pinned_id = program_id(&air).unwrap();

            let valid = HekateVerifier::<F, H>::verify(
                &pinned_id, &air, &instance, &proof, &mut vt, &config,
            );

            if let Ok(true) = valid {
                panic!("duplicate cpu request accepted — emit rank soundness break")
            }
        }
    }
}

#[test]
fn scribble_basemul_flip_selector_caught() {
    let ops = vec![
        BasemulOp {
            a: 100,
            b: 200,
            c: 300,
            idx: 0,
            ram_addr: 0,
        },
        BasemulOp {
            a: 50,
            b: 70,
            c: 120,
            idx: 1,
            ram_addr: 0,
        },
    ];

    let cpu_rows = (ops.len() + 1).next_power_of_two().max(4);
    let bm_rows = cpu_rows;

    let air = basemul_test_program(bm_rows, ops.len());

    let layout = CpuBasemulColumns::build_layout();
    let mut tb = TraceBuilder::new(&layout, cpu_rows.trailing_zeros() as usize).unwrap();

    for (row, op) in ops.iter().enumerate() {
        tb.set_b32(CpuBasemulColumns::BM_A, row, Block32::from(op.a))
            .unwrap();
        tb.set_b32(CpuBasemulColumns::BM_B, row, Block32::from(op.b))
            .unwrap();
        tb.set_b32(CpuBasemulColumns::BM_C, row, Block32::from(op.c))
            .unwrap();
        tb.set_b32(CpuBasemulColumns::BM_IDX, row, Block32::from(op.idx))
            .unwrap();
        tb.set_bit(CpuBasemulColumns::SELECTOR, row, Bit::ONE)
            .unwrap();
    }

    let cpu_trace = tb.build();
    let bm_trace = generate_basemul_trace(Q, &ops, bm_rows).expect("trace gen failed");

    let instance = ProgramInstance::new(cpu_rows, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![bm_trace]);

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
fn scribble_basemul_padding_row_attacks_caught() {
    let ops = vec![
        BasemulOp {
            a: 100,
            b: 200,
            c: 300,
            idx: 0,
            ram_addr: 0,
        },
        BasemulOp {
            a: 50,
            b: 70,
            c: 120,
            idx: 1,
            ram_addr: 1,
        },
        BasemulOp {
            a: 1500,
            b: 1500,
            c: (1500 + 1500) % Q,
            idx: 2,
            ram_addr: 2,
        },
    ];

    let cpu_rows = (ops.len() + 1).next_power_of_two().max(4);
    let bm_rows = cpu_rows;

    let air = basemul_test_program(bm_rows, ops.len());
    let layout = CpuBasemulColumns::build_layout();

    let mut tb = TraceBuilder::new(&layout, cpu_rows.trailing_zeros() as usize).unwrap();

    for (row, op) in ops.iter().enumerate() {
        tb.set_b32(CpuBasemulColumns::BM_A, row, Block32::from(op.a))
            .unwrap();
        tb.set_b32(CpuBasemulColumns::BM_B, row, Block32::from(op.b))
            .unwrap();
        tb.set_b32(CpuBasemulColumns::BM_C, row, Block32::from(op.c))
            .unwrap();
        tb.set_b32(CpuBasemulColumns::BM_IDX, row, Block32::from(op.idx))
            .unwrap();
        tb.set_bit(CpuBasemulColumns::SELECTOR, row, Bit::ONE)
            .unwrap();
    }

    let cpu_trace = tb.build();
    let bm_trace = generate_basemul_trace(Q, &ops, bm_rows).expect("trace gen failed");

    let instance = ProgramInstance::new(cpu_rows, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![bm_trace]);

    assert_all_caught_all_targets(
        &air,
        &instance,
        &witness,
        ScribbleConfig::default()
            .mutations([
                MutationKind::BitFlip,
                MutationKind::OutOfBounds,
                MutationKind::FlipSelector,
                MutationKind::DuplicateRow,
            ])
            .cases(256),
    );
}
