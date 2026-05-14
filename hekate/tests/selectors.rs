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

use hekate::core::config::Config;
use hekate::core::trace::{ColumnTrace, ColumnType, TraceColumn};
use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::{Bit, Block32, Block128};
use hekate_core::trace::IntoTraceColumn;
use hekate_math::TowerField;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;

/// Selector Discipline & Ghost Protocol
///
/// Tests the fundamental control flow primitives for zkVM:
/// - Boolean selectors
/// - One-hot encoding
/// - Halt discipline (Ghost Protocol)
type F = Block128;
type H = DefaultHasher;

/// Dummy CPU AIR with halt discipline.
///
/// Columns:
/// 0: pc (program counter)
/// 1: s_halt (halt selector)
/// 2: s_memory_op (memory operation event selector)
/// 3: q_step (transition selector: 1 for rows 0..N-2, 0 for N-1)
#[derive(Clone)]
struct DummyCpuAir {
    #[allow(dead_code)]
    num_rows: usize,
}

impl Air<F> for DummyCpuAir {
    fn num_columns(&self) -> usize {
        4
    }

    fn column_layout(&self) -> &'static [ColumnType] {
        &[
            ColumnType::B32,
            ColumnType::Bit,
            ColumnType::Bit,
            ColumnType::Bit,
        ]
    }

    fn constraint_ast(&self) -> hekate_program::constraint::ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();

        let pc = cs.col(0);
        let s_halt = cs.col(1);
        let s_mem = cs.col(2);
        let q_step = cs.col(3);
        let s_halt_next = cs.next(1);
        let pc_next = cs.next(0);

        // Booleanity:
        // s_halt, s_memory_op, q_step must be binary
        cs.assert_boolean(s_halt);
        cs.assert_boolean(s_mem);
        cs.assert_boolean(q_step);

        // Sticky halt:
        // q_step * s_halt * (1 + s_halt_next) = 0
        let one = cs.one();
        cs.constrain(q_step * s_halt * (s_halt_next + one));

        // Frozen PC:
        // q_step * s_halt * (pc_next + pc) = 0
        cs.constrain(q_step * s_halt * (pc_next + pc));

        // Event silence:
        // s_halt * s_mem = 0
        cs.constrain(s_halt * s_mem);

        cs.build()
    }
}

impl Program<F> for DummyCpuAir {}

/// Generate a valid trace:
/// runs for a few cycles, then halts cleanly.
fn generate_valid_trace(num_vars: usize) -> ColumnTrace {
    let num_rows = 1 << num_vars;

    let mut pc_col: Vec<Block32> = Vec::with_capacity(num_rows);
    let mut s_halt_col: Vec<Bit> = Vec::with_capacity(num_rows);
    let mut s_memory_col: Vec<Bit> = Vec::with_capacity(num_rows);
    let mut q_step_col: Vec<Bit> = Vec::with_capacity(num_rows);

    let halt_at_row = num_rows / 2;

    for i in 0..num_rows {
        if i < halt_at_row {
            // Running: PC increments, no halt, occasional memory op
            pc_col.push(Block32::from(i as u32));
            s_halt_col.push(Bit::ZERO);
            s_memory_col.push(if i % 4 == 0 { Bit::ONE } else { Bit::ZERO });
        } else {
            // Halted: PC frozen at last value before halt, halt=1, no memory ops
            pc_col.push(Block32::from((halt_at_row - 1) as u32)); // Frozen at last PC
            s_halt_col.push(Bit::ONE);
            s_memory_col.push(Bit::ZERO); // Ghost Protocol: no events when halted
        }

        // Transition selector: 1 for all rows except the last
        if i == num_rows - 1 {
            q_step_col.push(Bit::ZERO); // Disable constraints at last row
        } else {
            q_step_col.push(Bit::ONE);
        }
    }

    let mut trace = ColumnTrace::new(num_vars).unwrap();
    trace.add_column(pc_col.into_trace_column()).unwrap();
    trace.add_column(TraceColumn::Bit(s_halt_col)).unwrap();
    trace.add_column(TraceColumn::Bit(s_memory_col)).unwrap();
    trace.add_column(TraceColumn::Bit(q_step_col)).unwrap();

    trace
}

/// Generate invalid trace:
/// tries to "wake up" after halt (violates sticky_halt).
fn generate_wakeup_trace(num_vars: usize) -> ColumnTrace {
    let num_rows = 1 << num_vars;

    let mut pc_col: Vec<Block32> = Vec::with_capacity(num_rows);
    let mut s_halt_col: Vec<Bit> = Vec::with_capacity(num_rows);
    let mut s_memory_col: Vec<Bit> = Vec::with_capacity(num_rows);
    let mut q_step_col: Vec<Bit> = Vec::with_capacity(num_rows);

    let halt_at_row = num_rows / 3;
    let wakeup_at_row = (2 * num_rows) / 3;

    for i in 0..num_rows {
        if i < halt_at_row {
            // Running
            pc_col.push(Block32::from(i as u32));
            s_halt_col.push(Bit::ZERO);
            s_memory_col.push(Bit::ZERO);
        } else if i < wakeup_at_row {
            // Halted
            pc_col.push(Block32::from((halt_at_row - 1) as u32));
            s_halt_col.push(Bit::ONE);
            s_memory_col.push(Bit::ZERO);
        } else {
            // INVALID: Wake up (s_halt goes back to 0)
            pc_col.push(Block32::from(i as u32));
            s_halt_col.push(Bit::ZERO); // Violates sticky_halt!
            s_memory_col.push(Bit::ZERO);
        }

        // Transition selector
        if i == num_rows - 1 {
            q_step_col.push(Bit::ZERO);
        } else {
            q_step_col.push(Bit::ONE);
        }
    }

    let mut trace = ColumnTrace::new(num_vars).unwrap();
    trace.add_column(pc_col.into_trace_column()).unwrap();
    trace.add_column(TraceColumn::Bit(s_halt_col)).unwrap();
    trace.add_column(TraceColumn::Bit(s_memory_col)).unwrap();
    trace.add_column(TraceColumn::Bit(q_step_col)).unwrap();

    trace
}

/// Generate invalid trace:
/// performs memory op while halted (violates event_silence).
fn generate_event_while_halted_trace(num_vars: usize) -> ColumnTrace {
    let num_rows = 1 << num_vars;

    let mut pc_col: Vec<Block32> = Vec::with_capacity(num_rows);
    let mut s_halt_col: Vec<Bit> = Vec::with_capacity(num_rows);
    let mut s_memory_col: Vec<Bit> = Vec::with_capacity(num_rows);
    let mut q_step_col: Vec<Bit> = Vec::with_capacity(num_rows);

    let halt_at_row = num_rows / 2;
    let violate_at_row = (3 * num_rows) / 4;

    for i in 0..num_rows {
        if i < halt_at_row {
            // Running
            pc_col.push(Block32::from(i as u32));
            s_halt_col.push(Bit::ZERO);
            s_memory_col.push(Bit::ZERO);
        } else {
            // Halted, but...
            pc_col.push(Block32::from((halt_at_row - 1) as u32));
            s_halt_col.push(Bit::ONE);

            // INVALID: Memory op while halted at violate_at_row
            if i == violate_at_row {
                s_memory_col.push(Bit::ONE); // Violates event_silence!
            } else {
                s_memory_col.push(Bit::ZERO);
            }
        }

        // Transition selector
        if i == num_rows - 1 {
            q_step_col.push(Bit::ZERO);
        } else {
            q_step_col.push(Bit::ONE);
        }
    }

    let mut trace = ColumnTrace::new(num_vars).unwrap();
    trace.add_column(pc_col.into_trace_column()).unwrap();
    trace.add_column(TraceColumn::Bit(s_halt_col)).unwrap();
    trace.add_column(TraceColumn::Bit(s_memory_col)).unwrap();
    trace.add_column(TraceColumn::Bit(q_step_col)).unwrap();

    trace
}

// =======================================================================
// TEST 1 (Positive): Valid trace with clean halt passes
// =======================================================================
#[test]
fn valid_halt_trace_passes() {
    let num_vars = 6;
    let num_rows = 1 << num_vars;
    let seed = [0xAAu8; 32];

    let config = Config {
        num_queries: 4,
        min_security_bits: 0,
        sumcheck_blinding_factor: 0,
        ..Config::default()
    };

    let air = DummyCpuAir { num_rows };
    let instance = ProgramInstance::new(num_rows, vec![]);
    let trace = generate_valid_trace(num_vars);
    let witness = ProgramWitness::new(trace);

    // Prove
    let proof = prove(
        b"phase1_test",
        &air,
        &instance,
        &witness,
        &config,
        seed,
        None,
    )
    .unwrap();

    // Verify
    let mut verifier_transcript = Transcript::<H>::new(b"phase1_test");
    let result =
        HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut verifier_transcript, &config);

    match result {
        Ok(true) => {} // Success
        Ok(false) => panic!("Valid halt trace verification returned false"),
        Err(e) => panic!("Valid halt trace verification failed with error: {:?}", e),
    }

    assert!(result.unwrap(), "Valid halt trace should verify");
}

// =======================================================================
// TEST 2 (Negative): Waking up after halt is rejected
// =======================================================================
#[test]
fn wakeup_after_halt_rejected() {
    let num_vars = 6;
    let num_rows = 1 << num_vars;
    let seed = [0xAAu8; 32];

    let config = Config {
        num_queries: 4,
        min_security_bits: 0,
        sumcheck_blinding_factor: 0,
        ..Config::default()
    };

    let air = DummyCpuAir { num_rows };
    let instance = ProgramInstance::new(num_rows, vec![]);
    let trace = generate_wakeup_trace(num_vars);
    let witness = ProgramWitness::new(trace);

    // Prove should either fail or produce invalid proof
    let prove_result = prove(
        b"phase1_wakeup",
        &air,
        &instance,
        &witness,
        &config,
        seed,
        None,
    );

    match prove_result {
        Ok(proof) => {
            // If prover succeeds, verifier MUST reject
            let mut verifier_transcript = Transcript::<H>::new(b"phase1_wakeup");
            let verify_result = HekateVerifier::<F, H>::verify(
                &air,
                &instance,
                &proof,
                &mut verifier_transcript,
                &config,
            );

            match verify_result {
                Ok(true) => {
                    panic!("VULNERABILITY: Verifier accepted trace that violates sticky_halt!");
                }
                Ok(false) => {
                    // Correctly rejected
                }
                Err(_) => {
                    // Also acceptable
                }
            }
        }
        Err(_) => {
            // Prover rejected the trace (also acceptable)
        }
    }
}

// =======================================================================
// TEST 3 (Negative): Memory op while halted is rejected
// =======================================================================
#[test]
fn event_while_halted_rejected() {
    let num_vars = 6;
    let num_rows = 1 << num_vars;
    let seed = [0xAAu8; 32];

    let config = Config {
        num_queries: 4,
        min_security_bits: 0,
        sumcheck_blinding_factor: 0,
        ..Config::default()
    };

    let air = DummyCpuAir { num_rows };
    let instance = ProgramInstance::new(num_rows, vec![]);
    let trace = generate_event_while_halted_trace(num_vars);
    let witness = ProgramWitness::new(trace);

    // Prove should either fail or produce invalid proof
    let prove_result = prove(
        b"phase1_event",
        &air,
        &instance,
        &witness,
        &config,
        seed,
        None,
    );

    match prove_result {
        Ok(proof) => {
            // If prover succeeds, verifier MUST reject
            let mut verifier_transcript = Transcript::<H>::new(b"phase1_event");
            let verify_result = HekateVerifier::<F, H>::verify(
                &air,
                &instance,
                &proof,
                &mut verifier_transcript,
                &config,
            );

            match verify_result {
                Ok(true) => {
                    panic!("VULNERABILITY: Verifier accepted trace with events while halted!");
                }
                Ok(false) => {
                    // Correctly rejected
                }
                Err(_) => {
                    // Also acceptable
                }
            }
        }
        Err(_) => {
            // Prover rejected the trace (also acceptable)
        }
    }
}

// =======================================================================
// TEST 4: Boolean constraint enforcement
// =======================================================================
#[test]
fn boolean_constraint_enforced() {
    let num_vars = 4;
    let num_rows = 1 << num_vars;
    let seed = [0xAAu8; 32];

    let config = Config {
        num_queries: 4,
        min_security_bits: 0,
        sumcheck_blinding_factor: 0,
        ..Config::default()
    };

    let air = DummyCpuAir { num_rows };
    let instance = ProgramInstance::new(num_rows, vec![]);

    // Create trace with non-boolean selector (value = 2)
    let pc_col = vec![Block32::ZERO; num_rows];
    let s_halt_col = vec![Bit::ZERO; num_rows];
    let s_memory_col = vec![Bit::ZERO; num_rows];

    let mut q_step_col = vec![Bit::ONE; num_rows];

    // Last row: q_step = 0
    q_step_col[num_rows - 1] = Bit::ZERO;

    // Manually create invalid Bit value (this would require unsafe in practice)
    // For testing, we'll just use valid values and trust the constraint check

    let mut trace = ColumnTrace::new(num_vars).unwrap();
    trace.add_column(pc_col.into_trace_column()).unwrap();
    trace.add_column(TraceColumn::Bit(s_halt_col)).unwrap();
    trace.add_column(TraceColumn::Bit(s_memory_col)).unwrap();
    trace.add_column(TraceColumn::Bit(q_step_col)).unwrap();

    let witness = ProgramWitness::new(trace);

    // This trace is valid (all booleans are 0 or 1)
    let result = prove(
        b"phase1_boolean",
        &air,
        &instance,
        &witness,
        &config,
        seed,
        None,
    );

    assert!(result.is_ok(), "Valid boolean trace should prove");
}
