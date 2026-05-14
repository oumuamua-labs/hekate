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
use hekate::core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::{Bit, Block32, Block128, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::permutation::{
    BusKind, ChallengeLabel, PermutationCheckSpec, REQUEST_IDX_LABEL, Source,
};
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness, define_columns};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

type F = Block128;
type H = DefaultHasher;

const BUS_ID: &str = "forgery_bus";

define_columns! {
    PairedCpuCols {
        KEY: B32,
        S_SEND: Bit,
        S_RECV: Bit,
    }
}

define_columns! {
    PartnerCols {
        KEY: B32,
        SEL: Bit,
    }
}

#[derive(Clone)]
struct PairedCpu {
    kind: BusKind,
    with_mutex: bool,
}

impl Air<F> for PairedCpu {
    fn num_columns(&self) -> usize {
        PairedCpuCols::NUM_COLUMNS
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(PairedCpuCols::build_layout)
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        let sources = vec![
            (Source::Column(PairedCpuCols::KEY), b"kappa_key" as &[u8]),
            (Source::RowIndexLeBytes(4), b"kappa_clk" as &[u8]),
        ];

        vec![(
            BUS_ID.into(),
            PermutationCheckSpec::new_paired(
                sources,
                PairedCpuCols::S_SEND,
                PairedCpuCols::S_RECV,
                self.kind,
            ),
        )]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();

        cs.assert_boolean(cs.col(PairedCpuCols::S_SEND));
        cs.assert_boolean(cs.col(PairedCpuCols::S_RECV));

        if self.with_mutex {
            cs.constrain_named(
                "paired_bus_mutex",
                cs.col(PairedCpuCols::S_SEND) * cs.col(PairedCpuCols::S_RECV),
            );
        }

        cs.build()
    }
}

#[derive(Clone)]
struct PartnerChiplet {
    kind: BusKind,
}

impl Air<F> for PartnerChiplet {
    fn num_columns(&self) -> usize {
        PartnerCols::NUM_COLUMNS
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(PartnerCols::build_layout)
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        let sources = vec![
            (Source::Column(PartnerCols::KEY), b"kappa_key" as &[u8]),
            (Source::RowIndexLeBytes(4), b"kappa_clk" as &[u8]),
        ];

        let spec = match self.kind {
            BusKind::Permutation => PermutationCheckSpec::new(sources, Some(PartnerCols::SEL)),
            BusKind::Lookup => PermutationCheckSpec::new_lookup(sources, Some(PartnerCols::SEL)),
        };

        vec![(BUS_ID.into(), spec)]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(PartnerCols::SEL));

        cs.build()
    }
}

#[derive(Clone)]
struct ForgeryHost {
    cpu: PairedCpu,
    partner: PartnerChiplet,
}

impl Air<F> for ForgeryHost {
    fn num_columns(&self) -> usize {
        self.cpu.num_columns()
    }

    fn column_layout(&self) -> &[ColumnType] {
        self.cpu.column_layout()
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        self.cpu.permutation_checks()
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        self.cpu.constraint_ast()
    }
}

impl Program<F> for ForgeryHost {
    fn chiplet_defs(&self) -> hekate_core::errors::Result<Vec<ChipletDef<F>>> {
        Ok(vec![ChipletDef::from_air(&self.partner)?])
    }
}

fn build_cpu_trace(rows: &[(Block32, Bit, Bit)], num_rows: usize) -> ColumnTrace {
    let num_vars = num_rows.trailing_zeros() as usize;
    let layout = PairedCpuCols::build_layout();
    let mut tb = TraceBuilder::new(&layout, num_vars).unwrap();

    for (i, (key, send, recv)) in rows.iter().enumerate() {
        tb.set_b32(PairedCpuCols::KEY, i, *key).unwrap();
        tb.set_bit(PairedCpuCols::S_SEND, i, *send).unwrap();
        tb.set_bit(PairedCpuCols::S_RECV, i, *recv).unwrap();
    }

    tb.build()
}

fn build_partner_trace(rows: &[(Block32, Bit)], num_rows: usize) -> ColumnTrace {
    let num_vars = num_rows.trailing_zeros() as usize;
    let layout = PartnerCols::build_layout();
    let mut tb = TraceBuilder::new(&layout, num_vars).unwrap();

    for (i, (key, sel)) in rows.iter().enumerate() {
        tb.set_b32(PartnerCols::KEY, i, *key).unwrap();
        tb.set_bit(PartnerCols::SEL, i, *sel).unwrap();
    }

    tb.build()
}

fn run(
    program: &ForgeryHost,
    cpu_rows: &[(Block32, Bit, Bit)],
    partner_rows: &[(Block32, Bit)],
) -> bool {
    let num_rows: usize = 4;
    let seed = [0xC3u8; 32];

    let cpu = build_cpu_trace(cpu_rows, num_rows);
    let partner = build_partner_trace(partner_rows, num_rows);

    let witness = ProgramWitness::new(cpu).with_chiplets(vec![partner]);
    let instance = ProgramInstance::new(num_rows, vec![]);

    let config = Config {
        num_queries: 4,
        min_security_bits: 0,
        sumcheck_blinding_factor: 0,
        ..Config::default()
    };

    let proof = match prove(
        b"BIDIR_FORGERY",
        program,
        &instance,
        &witness,
        &config,
        seed,
        None,
    ) {
        Ok(p) => p,
        Err(_) => return false,
    };

    let mut verifier_t = Transcript::<H>::new(b"BIDIR_FORGERY");
    HekateVerifier::<F, H>::verify(program, &instance, &proof, &mut verifier_t, &config)
        .unwrap_or(false)
}

#[test]
fn paired_bus_disjoint_selectors_honest_accepts() {
    let program = ForgeryHost {
        cpu: PairedCpu {
            kind: BusKind::Permutation,
            with_mutex: true,
        },
        partner: PartnerChiplet {
            kind: BusKind::Permutation,
        },
    };

    let cpu_rows = [
        (Block32::from(0xA1A1_A1A1u32), Bit::ONE, Bit::ZERO),
        (Block32::ZERO, Bit::ZERO, Bit::ZERO),
        (Block32::ZERO, Bit::ZERO, Bit::ZERO),
        (Block32::ZERO, Bit::ZERO, Bit::ZERO),
    ];

    let partner_rows = [
        (Block32::from(0xA1A1_A1A1u32), Bit::ONE),
        (Block32::ZERO, Bit::ZERO),
        (Block32::ZERO, Bit::ZERO),
        (Block32::ZERO, Bit::ZERO),
    ];

    assert!(
        run(&program, &cpu_rows, &partner_rows),
        "honest disjoint selectors with mutex constraint must verify"
    );
}

#[test]
fn exploit_paired_bus_both_selectors_high_permutation() {
    let program = ForgeryHost {
        cpu: PairedCpu {
            kind: BusKind::Permutation,
            with_mutex: false,
        },
        partner: PartnerChiplet {
            kind: BusKind::Permutation,
        },
    };

    let cpu_rows = [
        (Block32::from(0xA1A1_A1A1u32), Bit::ONE, Bit::ZERO),
        (Block32::from(0xDEAD_BEEFu32), Bit::ONE, Bit::ONE),
        (Block32::ZERO, Bit::ZERO, Bit::ZERO),
        (Block32::ZERO, Bit::ZERO, Bit::ZERO),
    ];

    let partner_rows = [
        (Block32::from(0xA1A1_A1A1u32), Bit::ONE),
        (Block32::ZERO, Bit::ZERO),
        (Block32::ZERO, Bit::ZERO),
        (Block32::ZERO, Bit::ZERO),
    ];

    assert!(
        !run(&program, &cpu_rows, &partner_rows),
        "paired permutation bus must reject row with both selectors high"
    );
}

#[test]
fn exploit_paired_bus_both_selectors_high_lookup() {
    let program = ForgeryHost {
        cpu: PairedCpu {
            kind: BusKind::Lookup,
            with_mutex: false,
        },
        partner: PartnerChiplet {
            kind: BusKind::Lookup,
        },
    };

    let cpu_rows = [
        (Block32::from(0xA1A1_A1A1u32), Bit::ONE, Bit::ZERO),
        (Block32::from(0xDEAD_BEEFu32), Bit::ONE, Bit::ONE),
        (Block32::ZERO, Bit::ZERO, Bit::ZERO),
        (Block32::ZERO, Bit::ZERO, Bit::ZERO),
    ];

    let partner_rows = [
        (Block32::from(0xA1A1_A1A1u32), Bit::ONE),
        (Block32::ZERO, Bit::ZERO),
        (Block32::ZERO, Bit::ZERO),
        (Block32::ZERO, Bit::ZERO),
    ];

    assert!(
        !run(&program, &cpu_rows, &partner_rows),
        "paired lookup bus must reject row with both selectors high"
    );
}

#[test]
fn exploit_paired_bus_missing_mutex_constraint() {
    #[derive(Clone)]
    struct PairedNoMutexChiplet;

    impl Air<F> for PairedNoMutexChiplet {
        fn num_columns(&self) -> usize {
            PairedCpuCols::NUM_COLUMNS
        }

        fn column_layout(&self) -> &[ColumnType] {
            static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
            LAYOUT.get_or_init(PairedCpuCols::build_layout)
        }

        fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
            let sources = vec![
                (Source::Column(PairedCpuCols::KEY), b"kappa_key" as &[u8]),
                (Source::RowIndexLeBytes(4), b"kappa_clk" as &[u8]),
            ];

            vec![(
                BUS_ID.into(),
                PermutationCheckSpec::new_paired(
                    sources,
                    PairedCpuCols::S_SEND,
                    PairedCpuCols::S_RECV,
                    BusKind::Permutation,
                ),
            )]
        }

        fn constraint_ast(&self) -> ConstraintAst<F> {
            let cs = ConstraintSystem::<F>::new();
            cs.assert_boolean(cs.col(PairedCpuCols::S_SEND));
            cs.assert_boolean(cs.col(PairedCpuCols::S_RECV));

            cs.build()
        }
    }

    let result = ChipletDef::<F>::from_air(&PairedNoMutexChiplet);
    assert!(
        result.is_err(),
        "ChipletDef::from_air must reject paired chiplet without mutex constraint"
    );
}

// =====================================================
// Paired AIR: constraint_ast mutated post-prove
// =====================================================

const PAIRED_BUS: &str = "audit_paired_bus";

#[derive(Clone)]
struct PairedAir {
    include_mutex: Arc<AtomicBool>,
}

impl Air<F> for PairedAir {
    fn name(&self) -> String {
        "audit_paired".to_string()
    }

    fn num_columns(&self) -> usize {
        3
    }

    fn column_layout(&self) -> &[ColumnType] {
        &[ColumnType::B32, ColumnType::Bit, ColumnType::Bit]
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![(
            PAIRED_BUS.into(),
            PermutationCheckSpec::new_paired(
                vec![
                    (Source::Column(0), b"audit_paired_v" as ChallengeLabel),
                    (Source::RowIndexLeBytes(4), REQUEST_IDX_LABEL),
                ],
                1,
                2,
                BusKind::Permutation,
            ),
        )]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(1));
        cs.assert_boolean(cs.col(2));

        if self.include_mutex.load(Ordering::SeqCst) {
            cs.constrain(cs.col(1) * cs.col(2));
        }

        cs.build()
    }
}

impl Program<F> for PairedAir {}

#[test]
fn paired_air_constraint_ast_mutated_post_prove_rejected() {
    let num_vars = 4;
    let num_rows = 1 << num_vars;

    let air = PairedAir {
        include_mutex: Arc::new(AtomicBool::new(true)),
    };

    let trace = TraceBuilder::new(
        &[ColumnType::B32, ColumnType::Bit, ColumnType::Bit],
        num_vars,
    )
    .unwrap()
    .build();

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(trace);
    let config = Config {
        num_queries: 4,
        min_security_bits: 0,
        sumcheck_blinding_factor: 2,
        ldt_blinding_factor: 4,
        ..Config::default()
    };

    let proof = prove(
        b"AuditP0", &air, &instance, &witness, &config, [0xC3; 32], None,
    )
    .expect("paired baseline prove");

    let mut vt = Transcript::<H>::new(b"AuditP0");
    assert!(
        HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut vt, &config).unwrap(),
        "paired baseline must verify"
    );

    air.include_mutex.store(false, Ordering::SeqCst);

    let mut vt = Transcript::<H>::new(b"AuditP0");
    let result = HekateVerifier::<F, H>::verify(&air, &instance, &proof, &mut vt, &config);

    assert!(
        result.is_err(),
        "SECURITY FAILURE: proof accepted under mutated constraint_ast"
    );
}
