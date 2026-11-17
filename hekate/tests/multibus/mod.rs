// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate::core::config::Config;
use hekate::core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate::math::Block256;
use hekate_core::errors;
use hekate_core::proofs::InnerProof;
use hekate_math::Block32;
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::permutation::{ChallengeLabel, PermutationCheckSpec, Source};
use hekate_program::{Air, FixedColumn, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_testkit::verdict::verify;

type F = Block256;

pub const LABEL: &[u8] = b"AuditP0";

const MULTI_BUS_0: &str = "audit_bus_0";
const MULTI_BUS_1: &str = "audit_bus_1";
const MULTI_BUS_2: &str = "audit_bus_2";
const MULTI_BUS_VARS: usize = 5;

#[derive(Clone)]
struct MultiBusChiplet {
    num_rows: usize,
}

impl Air<F> for MultiBusChiplet {
    fn name(&self) -> String {
        "audit_multibus".to_string()
    }

    fn num_columns(&self) -> usize {
        2
    }

    fn column_layout(&self) -> &[ColumnType] {
        &[ColumnType::B32, ColumnType::Bit]
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        let spec = || {
            PermutationCheckSpec::new(
                vec![
                    (Source::Column(0), b"audit_v" as ChallengeLabel),
                    (Source::RowIndexLeBytes(4), b"audit_clk" as ChallengeLabel),
                ],
                Some(1),
            )
        };

        vec![
            (MULTI_BUS_0.into(), spec()),
            (MULTI_BUS_1.into(), spec()),
            (MULTI_BUS_2.into(), spec()),
        ]
    }

    fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
        vec![FixedColumn::prefix(1, self.num_rows - 1)]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
    }
}

#[derive(Clone)]
pub struct MultiBusProgram;

impl Air<F> for MultiBusProgram {
    fn name(&self) -> String {
        "audit_multibus_main".to_string()
    }

    fn num_columns(&self) -> usize {
        1
    }

    fn column_layout(&self) -> &[ColumnType] {
        &[ColumnType::Bit]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(0));

        cs.build()
    }
}

impl Program<F> for MultiBusProgram {
    fn chiplet_defs(&self) -> errors::Result<Vec<ChipletDef<F>>> {
        Ok(vec![
            ChipletDef::from_air(&MultiBusChiplet {
                num_rows: 1 << MULTI_BUS_VARS,
            })?,
            ChipletDef::from_air(&MultiBusChiplet {
                num_rows: 1 << MULTI_BUS_VARS,
            })?,
        ])
    }
}

pub fn cfg() -> Config {
    Config {
        num_queries: 4,
        min_security_bits: 0,
        zero_knowledge: true,
        fold_arity_log: 1,
        early_stop_vars: 1,
        ..Config::default()
    }
}

pub fn multibus_proof(
    config: Config,
) -> (MultiBusProgram, ProgramInstance<F>, Config, InnerProof<F>) {
    let num_vars = MULTI_BUS_VARS;
    let num_rows = 1 << num_vars;

    let air = MultiBusProgram;
    let main_trace = TraceBuilder::new(&[ColumnType::Bit], num_vars)
        .unwrap()
        .build();
    let chiplet_a = multibus_trace(num_vars);
    let chiplet_b = multibus_trace(num_vars);

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(main_trace).with_chiplets(vec![chiplet_a, chiplet_b]);

    let proof = prove(LABEL, &air, &instance, &witness, &config, [0xA5; 32], None)
        .expect("multibus baseline prove");

    let outcome = verify(&air, &instance, &config, &proof, LABEL);

    assert_eq!(outcome, Ok(()));

    (air, instance, config, proof)
}

fn multibus_trace(num_vars: usize) -> ColumnTrace {
    let mut tb = TraceBuilder::new(&[ColumnType::B32, ColumnType::Bit], num_vars).unwrap();
    let num_rows = tb.num_rows();

    tb.set_b32(0, 0, Block32::from(0xCAFE_F00Du32)).unwrap();
    tb.fill_selector(1, num_rows - 1).unwrap();

    tb.build()
}
