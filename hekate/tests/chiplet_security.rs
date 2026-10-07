// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Adversarial security tests for the
//! independent chiplet pipeline.
//!
//! Exercises forgery vectors specific
//! to chiplet isolation, transcript binding,
//! and evaluation argument integrity.

use hekate::core::config::{Config, MAX_TABLE_VARS};
use hekate::core::trace::{ColumnTrace, TraceColumn};
use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::{Block128, TowerField};
use hekate_core::errors::Error;
use hekate_core::trace::IntoTraceColumn;
use hekate_gadgets::{CpuFetchColumns, Instruction, RomChiplet, generate_rom_trace};
use hekate_math::{Bit, Block32};
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::digest::program_id;
use hekate_program::{FixedShape, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;

type F = Block128;
type H = DefaultHasher;

// ==========================================================
// Minimal independent-chiplet AIR for security tests.
// CPU has 6 columns (CpuFetchColumns).
// ROM is an independent chiplet via chiplet_defs().
// ==========================================================

fn chiplet_test_air(num_rows: usize) -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("ChipletTest", num_rows).unwrap();
    let cpu = cx.schema(&CpuFetchColumns::build_layout());

    cx.fix(
        cpu.at(CpuFetchColumns::SELECTOR),
        FixedShape::Cadence {
            stride: 1,
            count: num_rows,
            origin: 0,
            values: vec![F::ONE],
        },
    );

    cx.bus(RomChiplet::BUS_ID, RomChiplet::cpu_linking_spec());

    cx.attach(ChipletDef::from_air(&RomChiplet::new(num_rows, num_rows)).unwrap());

    cx.compile().unwrap()
}

// ==========================================================
// Helpers
// ==========================================================

fn build_test_system(
    num_vars: usize,
) -> (
    CircuitProgram<F>,
    ProgramInstance<F>,
    ProgramWitness<F, ColumnTrace>,
    Config,
) {
    let num_rows = 1 << num_vars;

    let instructions: Vec<Instruction> = (0..num_rows)
        .map(|i| Instruction::new(i as u32, 1, [0, 0, 0]))
        .collect();

    // CPU trace
    let mut cpu_trace = ColumnTrace::new(num_vars).unwrap();
    let mut pc_cols: Vec<Vec<Block32>> = (0..4).map(|_| Vec::with_capacity(num_rows)).collect();
    let mut op_col = Vec::with_capacity(num_rows);
    let mut arg_cols: Vec<Vec<Block32>> = (0..3).map(|_| Vec::with_capacity(num_rows)).collect();
    let mut sel_col = Vec::with_capacity(num_rows);

    for instr in &instructions {
        let bytes = instr.pc_bytes();
        for b in 0..4 {
            pc_cols[b].push(Block32::from(bytes[b] as u32));
        }

        op_col.push(Block32::from(instr.opcode as u32));

        let args = instr.args();
        for a in 0..3 {
            arg_cols[a].push(Block32::from(args[a] as u32));
        }

        sel_col.push(Bit::ONE);
    }

    for col in pc_cols {
        cpu_trace.add_column(col.into_trace_column()).unwrap();
    }

    cpu_trace.add_column(op_col.into_trace_column()).unwrap();

    for col in arg_cols {
        cpu_trace.add_column(col.into_trace_column()).unwrap();
    }

    cpu_trace.add_column(TraceColumn::Bit(sel_col)).unwrap();

    // ROM chiplet trace
    let rom_trace = generate_rom_trace(&instructions, num_rows).unwrap();

    let air = chiplet_test_air(num_rows);
    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![rom_trace]);

    let config = Config {
        num_queries: 4,
        min_security_bits: 0,
        zero_knowledge: true,
        ldt_support_size: 4,
        ..Config::default()
    };

    (air, instance, witness, config)
}

fn prove_and_verify(
    air: &CircuitProgram<F>,
    instance: &ProgramInstance<F>,
    witness: &ProgramWitness<F, ColumnTrace>,
    config: &Config,
) -> (hekate_core::proofs::InnerProof<F>, bool) {
    let seed = [0xBBu8; 32];
    let proof = prove(
        b"ChipletSecurity",
        air,
        instance,
        witness,
        config,
        seed,
        None,
    )
    .expect("proving failed");

    let mut vt = Transcript::<H>::new(b"ChipletSecurity");
    let pinned_id = program_id(air).unwrap();

    let ok = HekateVerifier::<F, H>::verify(&pinned_id, air, instance, &proof, &mut vt, config)
        .unwrap_or(false);

    (proof, ok)
}

// ==========================================================
// EXPLOIT: Extra chiplet heights in proof
//
// A malicious prover adds extra chiplet_rows
// that don't correspond to any chiplet_defs().
// The verifier must reject this immediately.
// ==========================================================

#[test]
fn extra_chiplet_rows_rejected() {
    let (air, instance, witness, config) = build_test_system(6);
    let (mut proof, ok) = prove_and_verify(&air, &instance, &witness, &config);
    assert!(ok, "Baseline proof must verify");

    // ATTACK:
    // Duplicate the first chiplet height
    let extra_rows = proof.chiplet_rows[0];
    proof.chiplet_rows.push(extra_rows);

    let mut vt = Transcript::<H>::new(b"ChipletSecurity");
    let pinned_id = program_id(&air).unwrap();

    let result =
        HekateVerifier::<F, H>::verify(&pinned_id, &air, &instance, &proof, &mut vt, &config);

    assert!(
        result.is_err() || !result.unwrap(),
        "SECURITY FAILURE: Extra chiplet height accepted"
    );
}

// ==========================================================
// EXPLOIT: Missing chiplet heights in proof
//
// A malicious prover strips chiplet_rows
// to bypass chiplet ZeroCheck verification.
// ==========================================================

#[test]
fn missing_chiplet_rows_rejected() {
    let (air, instance, witness, config) = build_test_system(6);
    let (mut proof, ok) = prove_and_verify(&air, &instance, &witness, &config);
    assert!(ok, "Baseline proof must verify");

    // ATTACK:
    // Remove all chiplet heights
    proof.chiplet_rows.clear();

    let mut vt = Transcript::<H>::new(b"ChipletSecurity");
    let pinned_id = program_id(&air).unwrap();

    let result =
        HekateVerifier::<F, H>::verify(&pinned_id, &air, &instance, &proof, &mut vt, &config);

    assert!(
        result.is_err(),
        "SECURITY FAILURE: Missing chiplet heights accepted"
    );
}

// ==========================================================
// EXPLOIT: Corrupted chiplet evaluation values
//
// A malicious prover modifies the claimed evaluation
// values for a chiplet. The TensorPCS proximity check
// must detect the mismatch.
// ==========================================================

#[test]
fn chiplet_eval_values_forgery() {
    let (air, instance, witness, config) = build_test_system(6);
    let (mut proof, ok) = prove_and_verify(&air, &instance, &witness, &config);
    assert!(ok, "Baseline proof must verify");

    // ATTACK:
    // Corrupt first chiplet's
    // claimed evaluation at r_final.
    let c_eval = &mut proof.chiplet_point_evaluations[0];
    assert!(!c_eval.1.is_empty());

    c_eval.1[0] += F::ONE;

    let mut vt = Transcript::<H>::new(b"ChipletSecurity");
    let pinned_id = program_id(&air).unwrap();

    let result =
        HekateVerifier::<F, H>::verify(&pinned_id, &air, &instance, &proof, &mut vt, &config);

    assert!(
        result.is_err() || !result.unwrap(),
        "SECURITY FAILURE: Chiplet eval value forgery accepted"
    );
}

// ==========================================================
// EXPLOIT: Swap the pool root
//
// A malicious prover replaces the pool's Merkle
// root with an all-zero root. This desynchronizes the
// Fiat-Shamir transcript (because the root is absorbed
// into the transcript before challenges are drawn).
// ==========================================================

#[test]
fn pool_root_swap_rejected() {
    let (air, instance, witness, config) = build_test_system(6);
    let (mut proof, ok) = prove_and_verify(&air, &instance, &witness, &config);
    assert!(ok, "Baseline proof must verify");

    // ATTACK:
    // Replace the pool root with zeros
    proof.trace_root = [0u8; 32];

    let mut vt = Transcript::<H>::new(b"ChipletSecurity");
    let pinned_id = program_id(&air).unwrap();

    let result =
        HekateVerifier::<F, H>::verify(&pinned_id, &air, &instance, &proof, &mut vt, &config);

    assert!(
        result.is_err() || !result.unwrap(),
        "SECURITY FAILURE: Forged pool Merkle root accepted"
    );
}

// ==========================================================
// EXPLOIT: Truncate chiplet claimed values
//
// A malicious prover strips entries from the
// chiplet's combined evaluation vector. The
// verifier must reject with an error, not panic.
// ==========================================================

#[test]
fn chiplet_eval_values_truncated() {
    let (air, instance, witness, config) = build_test_system(6);
    let (mut proof, ok) = prove_and_verify(&air, &instance, &witness, &config);
    assert!(ok, "Baseline proof must verify");

    // ATTACK:
    // Truncate chiplet's combined trace values
    // to 1 entry; the verifier's length check
    // rejects without panicking.
    proof.chiplet_point_evaluations[0].1.truncate(1);

    let mut vt = Transcript::<H>::new(b"ChipletSecurity");
    let pinned_id = program_id(&air).unwrap();

    let result =
        HekateVerifier::<F, H>::verify(&pinned_id, &air, &instance, &proof, &mut vt, &config);

    assert!(
        result.is_err(),
        "SECURITY FAILURE: Truncated chiplet eval values accepted (should reject)"
    );
}

// ==========================================================
// EXPLOIT: Forge chiplet LogUp claimed_sum
//
// A malicious prover modifies the chiplet's LogUp
// claimed_sum so the paired (main, chiplet) endpoints
// no longer cancel. `check_bus_sum_matching` must reject.
// ==========================================================

#[test]
fn chiplet_logup_sum_mismatch() {
    let (air, instance, witness, config) = build_test_system(6);
    let (mut proof, ok) = prove_and_verify(&air, &instance, &witness, &config);
    assert!(ok, "Baseline proof must verify");

    assert_eq!(proof.chiplet_logup_aux.len(), 1);
    assert!(!proof.chiplet_logup_aux[0].claimed_sums.is_empty());

    // ATTACK:
    // Corrupt the chiplet's claimed_sum
    // so the bus no longer cancels.
    proof.chiplet_logup_aux[0].claimed_sums[0].1 += F::ONE;

    let mut vt = Transcript::<H>::new(b"ChipletSecurity");
    let pinned_id = program_id(&air).unwrap();

    let result =
        HekateVerifier::<F, H>::verify(&pinned_id, &air, &instance, &proof, &mut vt, &config);

    assert!(
        result.is_err() || !result.unwrap(),
        "SECURITY FAILURE: Chiplet LogUp claimed_sum forgery accepted"
    );
}

// ==========================================================
// EXPLOIT: Unmatched main-trace bus_id
//
// A program declares a main-trace bus endpoint
// ("phantom_bus") that no chiplet or gadget supplies.
// The verifier's exhaustiveness check must reject
// this because the bus has no counterpart.
// ==========================================================

/// AIR with an extra phantom bus_id that
/// no chiplet provides. Used only for
/// verification (not proving).
fn phantom_bus_air(num_rows: usize) -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("PhantomBus", num_rows).unwrap();
    let cpu = cx.schema(&CpuFetchColumns::build_layout());

    cx.fix(
        cpu.at(CpuFetchColumns::SELECTOR),
        FixedShape::Cadence {
            stride: 1,
            count: num_rows,
            origin: 0,
            values: vec![F::ONE],
        },
    );

    cx.bus(RomChiplet::BUS_ID, RomChiplet::cpu_linking_spec());

    // Phantom bus:
    // no chiplet supplies this.
    cx.bus("phantom_bus", RomChiplet::cpu_linking_spec());

    cx.attach(ChipletDef::from_air(&RomChiplet::new(num_rows, num_rows)).unwrap());

    cx.compile().unwrap()
}

#[test]
fn unmatched_main_bus_rejected() {
    let num_vars = 6;
    let num_rows = 1 << num_vars;

    // Prove with the normal AIR (1 main bus + 1 chiplet).
    let (air, instance, witness, config) = build_test_system(num_vars);
    let seed = [0xBBu8; 32];

    let proof = prove(
        b"ChipletSecurity",
        &air,
        &instance,
        &witness,
        &config,
        seed,
        None,
    )
    .expect("proving failed");

    // Verify with the phantom AIR that declares
    // an extra bus_id no chiplet supplies.
    // The verifier must reject: "phantom_bus"
    // has no chiplet/gadget counterpart.
    let phantom_air = phantom_bus_air(num_rows);

    let mut vt = Transcript::<H>::new(b"ChipletSecurity");
    let pinned_id = program_id(&phantom_air).unwrap();

    let result = HekateVerifier::<F, H>::verify(
        &pinned_id,
        &phantom_air,
        &instance,
        &proof,
        &mut vt,
        &config,
    );

    assert!(
        result.is_err(),
        "SECURITY FAILURE: Unmatched main-trace bus_id accepted"
    );
}

// =====================================================
// Chiplet sumcheck round-poly degree is strict
// =====================================================

#[test]
fn chiplet_sumcheck_degree_inflation_rejected() {
    let (air, instance, witness, config) = build_test_system(6);
    let (mut proof, ok) = prove_and_verify(&air, &instance, &witness, &config);

    assert!(ok, "Baseline proof must verify");

    let pad = proof.chiplet_zerocheck_proofs[0].round_polys[0].evals[0];
    proof.chiplet_zerocheck_proofs[0].round_polys[0]
        .evals
        .push(pad);

    let mut vt = Transcript::<H>::new(b"ChipletSecurity");
    let pinned_id = program_id(&air).unwrap();

    let result =
        HekateVerifier::<F, H>::verify(&pinned_id, &air, &instance, &proof, &mut vt, &config);

    assert!(
        result.is_err() || !result.unwrap(),
        "SECURITY FAILURE: degree-inflated chiplet round poly accepted"
    );
}

// =====================================================
// chiplet_rows[i]
// must be non-zero power of two.
// =====================================================

#[test]
fn chiplet_rows_non_power_of_two_rejected() {
    let (air, instance, witness, config) = build_test_system(6);
    let (mut proof, ok) = prove_and_verify(&air, &instance, &witness, &config);

    assert!(ok, "Baseline proof must verify");

    proof.chiplet_rows[0] = 5;

    let mut vt = Transcript::<H>::new(b"ChipletSecurity");
    let pinned_id = program_id(&air).unwrap();

    let result =
        HekateVerifier::<F, H>::verify(&pinned_id, &air, &instance, &proof, &mut vt, &config);

    assert!(
        matches!(
            result,
            Err(Error::Protocol {
                protocol: "verifier",
                message: "chiplet_rows[c] must be a non-zero power of two",
            })
        ),
        "{result:?}"
    );
}

#[test]
fn chiplet_rows_zero_rejected() {
    let (air, instance, witness, config) = build_test_system(6);
    let (mut proof, ok) = prove_and_verify(&air, &instance, &witness, &config);

    assert!(ok, "Baseline proof must verify");

    proof.chiplet_rows[0] = 0;

    let mut vt = Transcript::<H>::new(b"ChipletSecurity");
    let pinned_id = program_id(&air).unwrap();

    let result =
        HekateVerifier::<F, H>::verify(&pinned_id, &air, &instance, &proof, &mut vt, &config);

    assert!(
        matches!(
            result,
            Err(Error::Protocol {
                protocol: "verifier",
                message: "chiplet_rows[c] must be a non-zero power of two",
            })
        ),
        "{result:?}"
    );
}

// ==========================================================
// EXPLOIT: Tamper one table's part of a pool leaf
//
// A pool leaf hashes every table's part. One flipped
// byte in the ROM or the CPU part, of the trace tree
// or the `h` tree, must fail the Merkle check.
// ==========================================================

#[test]
fn tampered_leaf_part_rejected_at_merkle_check() {
    let (air, instance, witness, config) = build_test_system(6);
    let (proof, ok) = prove_and_verify(&air, &instance, &witness, &config);

    assert!(ok, "Baseline proof must verify");

    let pinned_id = program_id(&air).unwrap();

    for h_tree in [false, true] {
        for chiplet_part in [true, false] {
            let mut forged = proof.clone();

            let opening = match h_tree {
                true => forged.eval_proof.h_ldt_proof.as_mut().unwrap(),
                false => &mut forged.eval_proof.ldt_proof,
            };

            let column = &mut opening.opened_columns[0];
            let byte = match chiplet_part {
                true => 0,
                false => column.len() - 1,
            };

            column[byte] ^= 1;

            let mut vt = Transcript::<H>::new(b"ChipletSecurity");
            let result = HekateVerifier::<F, H>::verify(
                &pinned_id, &air, &instance, &forged, &mut vt, &config,
            );

            assert!(
                matches!(
                    result,
                    Err(Error::Protocol {
                        protocol: "brakedown",
                        message: "batch merkle proof verification failed",
                    })
                ),
                "h_tree {h_tree}, chiplet_part {chiplet_part}: {result:?}"
            );
        }
    }
}

#[test]
fn chiplet_rows_above_height_bound_rejected() {
    let (air, instance, witness, config) = build_test_system(6);
    let (mut proof, ok) = prove_and_verify(&air, &instance, &witness, &config);

    assert!(ok, "Baseline proof must verify");

    proof.chiplet_rows[0] = 1 << (MAX_TABLE_VARS + 1);

    let mut vt = Transcript::<H>::new(b"ChipletSecurity");
    let pinned_id = program_id(&air).unwrap();

    let result =
        HekateVerifier::<F, H>::verify(&pinned_id, &air, &instance, &proof, &mut vt, &config);

    assert!(
        matches!(
            result,
            Err(Error::Protocol {
                protocol: "verifier",
                message: "chiplet_rows[c] exceeds the table height bound",
            })
        ),
        "{result:?}"
    );
}

#[test]
fn main_rows_above_height_bound_rejected() {
    let (air, instance, witness, config) = build_test_system(6);
    let (proof, ok) = prove_and_verify(&air, &instance, &witness, &config);

    assert!(ok, "Baseline proof must verify");

    let tall = ProgramInstance::new(1 << (MAX_TABLE_VARS + 1), vec![]);

    let mut vt = Transcript::<H>::new(b"ChipletSecurity");
    let pinned_id = program_id(&air).unwrap();

    let result = HekateVerifier::<F, H>::verify(&pinned_id, &air, &tall, &proof, &mut vt, &config);

    assert!(
        matches!(
            result,
            Err(Error::Protocol {
                protocol: "verifier",
                message: "num_rows exceeds the table height bound",
            })
        ),
        "{result:?}"
    );
}

#[test]
fn height_spread_rejected_before_pool_layout() {
    let (air, instance, witness, config) = build_test_system(6);
    let (mut proof, ok) = prove_and_verify(&air, &instance, &witness, &config);

    assert!(ok, "Baseline proof must verify");

    proof.chiplet_rows[0] = 1 << 20;

    let mut vt = Transcript::<H>::new(b"ChipletSecurity");
    let pinned_id = program_id(&air).unwrap();

    let result =
        HekateVerifier::<F, H>::verify(&pinned_id, &air, &instance, &proof, &mut vt, &config);

    assert!(
        matches!(
            result,
            Err(Error::Protocol {
                protocol: "verifier",
                message: "claimed evaluation count does not match the pool layout",
            })
        ),
        "{result:?}"
    );
}
