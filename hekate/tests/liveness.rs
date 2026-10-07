// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate::math::Block128;
use hekate_aes::{Aes128Chiplet, Aes256Chiplet};
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_gadgets::{IntArithmeticChiplet, ModexpChiplet, RamChiplet, RomChiplet};
use hekate_keccak::KeccakChiplet;
use hekate_pqc::mldsa::{MlDsaChiplet, MlDsaParams};
use hekate_pqc::mlkem::{MlKemCall, MlKemChiplet, MlKemParams};
use hekate_program::chiplet::ChipletDef;
use hekate_program::expander::{PoolLayout, RingSwitchPlan};
use hekate_program::outer::{OuterStatement, TableShape, table_plan};
use hekate_program::{Air, FixedColumn};
use hekate_sha2::Sha256Chiplet;

const NUM_ROWS: usize = 256;
const KEM_CALLS: [MlKemCall; 3] = [MlKemCall::KeyGen, MlKemCall::Encaps, MlKemCall::Decaps];

type F = Block128;
type Snapshot = (&'static str, errors::Result<Vec<ChipletDef<F>>>);

fn witness_selectors<A: Air<F>>(air: &A) -> Vec<String> {
    let fixed: Vec<FixedColumn<F>> = air.fixed_columns();

    let mut hits = Vec::new();
    for (bus_id, spec) in air.permutation_checks() {
        for sel in [spec.selector, spec.recv_selector].into_iter().flatten() {
            let verdict = match fixed.iter().find(|fc| fc.col_idx == sel) {
                None => "witness column",
                Some(fc) if !fc.shape.is_overlay() => "substituted shape",
                Some(_) => continue,
            };

            hits.push(format!("{}::{} col {sel}: {verdict}", air.name(), bus_id));
        }
    }

    hits
}

fn shipped_tables() -> Vec<Snapshot> {
    let num_rows = NUM_ROWS;

    vec![
        (
            "KeccakChiplet",
            ChipletDef::from_air(&KeccakChiplet::new(num_rows, 4)).map(|d| vec![d]),
        ),
        (
            "RamChiplet",
            ChipletDef::from_air(&RamChiplet::new(num_rows, num_rows)).map(|d| vec![d]),
        ),
        (
            "RomChiplet",
            ChipletDef::from_air(&RomChiplet::new(num_rows, num_rows)).map(|d| vec![d]),
        ),
        (
            "IntArithmeticChiplet",
            ChipletDef::from_air(&IntArithmeticChiplet::new(32, num_rows, num_rows).unwrap())
                .map(|d| vec![d]),
        ),
        (
            "Aes128Chiplet",
            Aes128Chiplet::new(num_rows, num_rows, 4).unwrap().defs(),
        ),
        (
            "Aes256Chiplet",
            Aes256Chiplet::new(num_rows, num_rows, 4).unwrap().defs(),
        ),
        (
            "Sha256Chiplet",
            Sha256Chiplet::<F>::new(num_rows, 4, 4)
                .and_then(|c| c.def())
                .map(|d| vec![d]),
        ),
        (
            "ModexpChiplet",
            ModexpChiplet::new().and_then(|c| c.def()).map(|d| vec![d]),
        ),
        ("ML-DSA-44", dsa(MlDsaParams::ML_DSA_44)),
        ("ML-DSA-65", dsa(MlDsaParams::ML_DSA_65)),
        ("ML-DSA-87", dsa(MlDsaParams::ML_DSA_87)),
        ("ML-KEM-512", kem(MlKemParams::ML_KEM_512)),
        ("ML-KEM-768", kem(MlKemParams::ML_KEM_768)),
        ("ML-KEM-1024", kem(MlKemParams::ML_KEM_1024)),
    ]
}

fn dsa(params: MlDsaParams) -> errors::Result<Vec<ChipletDef<F>>> {
    MlDsaChiplet::<F>::new(params, &[32])?.defs()
}

fn kem(params: MlKemParams) -> errors::Result<Vec<ChipletDef<F>>> {
    MlKemChiplet::<F>::new(params, &KEM_CALLS)?.defs()
}

#[test]
fn every_bus_selector_is_fixed_or_absent() {
    let mut hits = Vec::new();
    for (label, snapshot) in shipped_tables() {
        match snapshot {
            Ok(defs) => hits.extend(defs.iter().flat_map(witness_selectors)),
            Err(e) => hits.push(format!("{label}: snapshot rejected: {e}")),
        }
    }

    assert!(
        hits.is_empty(),
        "witness bus selectors:\n{}",
        hits.join("\n")
    );
}

#[test]
fn shipped_tables_fit_outer_statement() {
    let config = Config::prod();
    let num_vars = NUM_ROWS.trailing_zeros() as usize;
    let field_bits = size_of::<F>() * 8;

    let mut labels = Vec::new();
    let mut defs = Vec::new();

    for (label, snapshot) in shipped_tables() {
        match snapshot {
            Ok(snapshot) => {
                labels.extend(snapshot.iter().map(|_| label));
                defs.extend(snapshot);
            }
            Err(e) => panic!("{label}: snapshot rejected: {e}"),
        }
    }

    let plans: Vec<RingSwitchPlan> = defs
        .iter()
        .map(|def| table_plan(def, def.permutation_checks.len(), &config).unwrap())
        .collect();

    let heights: Vec<(&RingSwitchPlan, usize)> = plans.iter().map(|p| (p, num_vars)).collect();
    let pool = PoolLayout::new(&heights, field_bits, &config);

    let shapes: Vec<TableShape> = defs
        .iter()
        .zip(&pool.tables)
        .map(|(def, layout)| {
            TableShape::from_air(def, &def.statics())
                .unwrap()
                .at(layout)
        })
        .collect();

    let statement = OuterStatement::new(&shapes, pool.num_vars());

    let per_table: Vec<String> = labels
        .iter()
        .zip(&shapes)
        .map(|(label, shape)| format!("{label}: {}", shape.mul_wires()))
        .collect();

    assert!(
        config
            .outer_geom(statement.masked_scalars, statement.mul_wires, field_bits)
            .is_ok(),
        "{} mul wires exceed the outer statement:\n{}",
        statement.mul_wires,
        per_table.join("\n"),
    );
}
