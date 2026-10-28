// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate::core::config::Config;
use hekate::core::trace::{ColumnTrace, ColumnType, TraceColumn};
use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::{Block128, TowerField};
use hekate_core::proofs::InnerProof;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::constraint::{BoundaryConstraint, ConstraintAst};
use hekate_program::digest::program_id;
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;
use std::io::Write;
use std::sync::{Arc, Mutex};

type F = Block128;
type H = DefaultHasher;

const VARS: usize = 6;
const LABEL: &[u8] = b"TypeBinding";
const VALUE: usize = 0;
const NARROW: usize = 1;

const NARROW_TYPES: [(ColumnType, u32); 5] = [
    (ColumnType::Bit, 1),
    (ColumnType::B8, 8),
    (ColumnType::B16, 16),
    (ColumnType::B32, 32),
    (ColumnType::B64, 64),
];

const WIDE_TYPES: [(ColumnType, u32); 3] = [
    (ColumnType::B32, 32),
    (ColumnType::B64, 64),
    (ColumnType::B128, 128),
];

/// Proves the public input fits the narrow column's type.
/// A constraint ranging that column would void the guard.
#[derive(Clone)]
struct RangeByType {
    layout: [ColumnType; 2],
}

impl Air<F> for RangeByType {
    fn boundary_constraints(&self) -> Vec<BoundaryConstraint<F>> {
        vec![BoundaryConstraint::with_public_input(VALUE, 0, 0)]
    }

    fn column_layout(&self) -> &[ColumnType] {
        &self.layout
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.constrain(cs.col(VALUE) + cs.col(NARROW));

        cs.build()
    }
}

impl Program<F> for RangeByType {
    fn num_public_inputs(&self) -> usize {
        1
    }
}

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);

impl Write for Capture {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .map_err(|_| std::io::Error::other("log capture poisoned"))?
            .extend_from_slice(buf);

        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn config(zero_knowledge: bool) -> Config {
    Config {
        fold_arity_log: 2,
        early_stop_vars: 2,
        zero_knowledge,
        ..Config::dev()
    }
}

fn instance(value: u128) -> ProgramInstance<F> {
    ProgramInstance::new(1 << VARS, vec![F::from(value)])
}

fn prove_with(
    air: &RangeByType,
    config: &Config,
    value: u128,
    storage: ColumnType,
) -> InnerProof<F> {
    let mut cells = vec![F::ZERO; 1 << VARS];
    cells[0] = F::from(value);

    let mut trace = ColumnTrace::new(VARS).unwrap();
    trace
        .add_column(TraceColumn::from_data(cells.clone(), air.layout[VALUE]))
        .unwrap();
    trace
        .add_column(TraceColumn::from_data(cells, storage))
        .unwrap();

    prove(
        LABEL,
        air,
        &instance(value),
        &ProgramWitness::new(trace),
        config,
        [0x94; 32],
        None,
    )
    .unwrap()
}

fn accepts(air: &RangeByType, config: &Config, value: u128, proof: &InnerProof<F>) -> bool {
    let mut transcript = Transcript::<H>::new(LABEL);

    matches!(
        HekateVerifier::<F, H>::verify(
            &program_id(air).unwrap(),
            air,
            &instance(value),
            proof,
            &mut transcript,
            config,
        ),
        Ok(true)
    )
}

fn verify_logged(
    air: &RangeByType,
    config: &Config,
    value: u128,
    proof: &InnerProof<F>,
) -> (bool, String) {
    let log = Capture::default();
    let writer = log.clone();

    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .with_writer(move || writer.clone())
        .finish();

    let accepted =
        tracing::subscriber::with_default(subscriber, || accepts(air, config, value, proof));

    let text = String::from_utf8_lossy(&log.0.lock().unwrap()).into_owned();

    (accepted, text)
}

#[test]
fn whole_read_cell_outside_type_rejected() {
    let mut escaped = Vec::new();
    for zero_knowledge in [false, true] {
        let config = config(zero_knowledge);
        let site = match zero_knowledge {
            false => "Sumcheck failed",
            true => "outer linear test failed",
        };

        for (wide, wide_bits) in WIDE_TYPES {
            for (narrow, bits) in NARROW_TYPES {
                if bits >= wide_bits {
                    continue;
                }

                let air = RangeByType {
                    layout: [wide, narrow],
                };

                let fits = (1u128 << bits) - 1;
                let honest = prove_with(&air, &config, fits, narrow);

                assert!(
                    accepts(&air, &config, fits, &honest),
                    "{wide:?}, {narrow:?}, zero knowledge {zero_knowledge}"
                );

                let outside = 1u128 << bits;
                let forged = prove_with(&air, &config, outside, wide);

                let (accepted, log) = verify_logged(&air, &config, outside, &forged);

                if accepted || !log.contains(site) {
                    escaped.push((wide, narrow, zero_knowledge, log));
                }
            }
        }
    }

    assert!(escaped.is_empty(), "{escaped:?}");
}
