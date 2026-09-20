// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! What each shipped table raises against an all-zero trace.

use hekate_aes::{Aes128Chiplet, Aes256Chiplet};
use hekate_gadgets::chiplets::bignum::modexp;
use hekate_gadgets::{IntArithmeticChiplet, ModexpChiplet, RamChiplet, RomChiplet};
use hekate_keccak::KeccakChiplet;
use hekate_math::{Block128, Flat, TowerField};
use hekate_pqc::mldsa::{MlDsaChiplet, MlDsaLevel};
use hekate_pqc::mlkem::{MlKemChiplet, MlKemLevel};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::BoundaryTarget;
use hekate_program::{Air, FixedShape};
use hekate_sha2::Sha256Chiplet;

type F = Block128;

const PROBE_ROWS: usize = 256;
const PROBE_BLOCKS: usize = 4;
const PROBE_ROUNDS: usize = 4;
const SBOX_ROM_ROWS: usize = 256;

/// `(owner, table)`; owner is empty for a standalone
/// chiplet. Measured at `PROBE_BLOCKS` blocks and,
/// bar the fixed-height modexp, `PROBE_ROWS` rows.
#[rustfmt::skip]
const EXPECTED: &[(&str, &str, Floor)] = &[
    ("",       "KeccakChiplet",     Floor { roots: 1625, roots_violated: 0,  pins_firing: 27, boundaries_nonzero: 0 }),
    ("",       "RamChiplet",        Floor { roots: 190,  roots_violated: 2,  pins_firing: 3,  boundaries_nonzero: 0 }),
    ("",       "RomChiplet",        Floor { roots: 0,    roots_violated: 0,  pins_firing: 1,  boundaries_nonzero: 0 }),
    ("u32",    "ArithmeticChiplet", Floor { roots: 446,  roots_violated: 0,  pins_firing: 1,  boundaries_nonzero: 0 }),
    ("u64",    "ArithmeticChiplet", Floor { roots: 862,  roots_violated: 0,  pins_firing: 1,  boundaries_nonzero: 0 }),
    ("",       "Sha256Chiplet",     Floor { roots: 1960, roots_violated: 0,  pins_firing: 7,  boundaries_nonzero: 0 }),
    ("",       "ModexpChiplet",     Floor { roots: 3684, roots_violated: 152, pins_firing: 37, boundaries_nonzero: 0 }),
    ("aes128", "AesRound128Air",    Floor { roots: 184,  roots_violated: 0,  pins_firing: 15, boundaries_nonzero: 0 }),
    ("aes128", "SboxRomChiplet",    Floor { roots: 241,  roots_violated: 0,  pins_firing: 1,  boundaries_nonzero: 0 }),
    ("aes256", "AesRound256Air",    Floor { roots: 164,  roots_violated: 0,  pins_firing: 19, boundaries_nonzero: 0 }),
    ("aes256", "SboxRomChiplet",    Floor { roots: 241,  roots_violated: 0,  pins_firing: 1,  boundaries_nonzero: 0 }),
    ("mlkem",  "MlKemCtrlChiplet",  Floor { roots: 442,  roots_violated: 0,  pins_firing: 2,  boundaries_nonzero: 0 }),
    ("mlkem",  "KeccakChiplet",     Floor { roots: 1625, roots_violated: 0,  pins_firing: 27, boundaries_nonzero: 0 }),
    ("mlkem",  "NttChiplet",        Floor { roots: 1705, roots_violated: 14, pins_firing: 4,  boundaries_nonzero: 0 }),
    ("mlkem",  "TwiddleRomChiplet", Floor { roots: 6,    roots_violated: 1,  pins_firing: 1,  boundaries_nonzero: 0 }),
    ("mlkem",  "BasemulChiplet",    Floor { roots: 219,  roots_violated: 3,  pins_firing: 1,  boundaries_nonzero: 0 }),
    ("mlkem",  "RamChiplet",        Floor { roots: 190,  roots_violated: 2,  pins_firing: 3,  boundaries_nonzero: 0 }),
    ("mldsa",  "MlDsaCtrlChiplet",  Floor { roots: 269,  roots_violated: 0,  pins_firing: 2,  boundaries_nonzero: 0 }),
    ("mldsa",  "KeccakChiplet",     Floor { roots: 1625, roots_violated: 0,  pins_firing: 27, boundaries_nonzero: 0 }),
    ("mldsa",  "NttChiplet",        Floor { roots: 5885, roots_violated: 42, pins_firing: 4,  boundaries_nonzero: 0 }),
    ("mldsa",  "TwiddleRomChiplet", Floor { roots: 6,    roots_violated: 1,  pins_firing: 1,  boundaries_nonzero: 0 }),
    ("mldsa",  "NormCheckChiplet",  Floor { roots: 300,  roots_violated: 27, pins_firing: 1,  boundaries_nonzero: 0 }),
    ("mldsa",  "HighBitsChiplet",   Floor { roots: 1102, roots_violated: 28, pins_firing: 1,  boundaries_nonzero: 0 }),
    ("mldsa",  "RamChiplet",        Floor { roots: 190,  roots_violated: 2,  pins_firing: 3,  boundaries_nonzero: 0 }),
];

#[derive(Debug, PartialEq, Eq)]
struct Floor {
    roots: usize,
    roots_violated: usize,
    pins_firing: usize,
    boundaries_nonzero: usize,
}

fn measure(def: &ChipletDef<F>, rows: usize) -> Floor {
    let num_vars = rows.trailing_zeros() as usize;
    let zero = Flat::from_raw(F::ZERO);

    let ast = Air::<F>::constraint_ast(def);
    let consts = ast.precompute_hardware_consts();
    let row = vec![zero; Air::<F>::num_columns(def)];

    let mut buf = Vec::new();
    ast.evaluate_into(&consts, &row, &row, &mut buf);

    let roots_violated = ast
        .roots
        .iter()
        .filter(|root| buf[root.0 as usize] != zero)
        .count();

    let pins_firing = Air::<F>::fixed_columns(def)
        .iter()
        .filter(|pin| fires(&pin.shape, num_vars))
        .count();

    let boundaries_nonzero = def
        .boundary_constraints()
        .iter()
        .filter(|bc| matches!(bc.target, BoundaryTarget::Constant(v) if v != F::ZERO))
        .count();

    Floor {
        roots: ast.roots.len(),
        roots_violated,
        pins_firing,
        boundaries_nonzero,
    }
}

fn fires(shape: &FixedShape<F>, num_vars: usize) -> bool {
    let zero = Flat::from_raw(F::ZERO);

    (0..1usize << num_vars).any(|row| shape.value_at_row(row, num_vars) != zero)
}

fn all_tables() -> Vec<(&'static str, usize, ChipletDef<F>)> {
    let aes128 = Aes128Chiplet::<F>::new(PROBE_ROWS, SBOX_ROM_ROWS, PROBE_BLOCKS).unwrap();
    let aes256 = Aes256Chiplet::<F>::new(PROBE_ROWS, SBOX_ROM_ROWS, PROBE_BLOCKS).unwrap();
    let mlkem = MlKemChiplet::<F>::new(MlKemLevel::MLKEM_768);
    let mldsa = MlDsaChiplet::<F>::new(MlDsaLevel::MLDSA_65, 32);

    let mut defs = vec![
        (
            "",
            PROBE_ROWS,
            ChipletDef::from_air(&KeccakChiplet::new(PROBE_ROWS, PROBE_BLOCKS)).unwrap(),
        ),
        (
            "",
            PROBE_ROWS,
            ChipletDef::from_air(&RamChiplet::new(PROBE_ROWS, PROBE_ROWS)).unwrap(),
        ),
        (
            "",
            PROBE_ROWS,
            ChipletDef::from_air(&RomChiplet::new(PROBE_ROWS, PROBE_ROWS)).unwrap(),
        ),
        (
            "u32",
            PROBE_ROWS,
            ChipletDef::from_air(&IntArithmeticChiplet::new(32, PROBE_ROWS, PROBE_ROWS).unwrap())
                .unwrap(),
        ),
        (
            "u64",
            PROBE_ROWS,
            ChipletDef::from_air(&IntArithmeticChiplet::new(64, PROBE_ROWS, PROBE_ROWS).unwrap())
                .unwrap(),
        ),
        (
            "",
            PROBE_ROWS,
            Sha256Chiplet::<F>::new(PROBE_ROWS, PROBE_BLOCKS, PROBE_ROUNDS)
                .unwrap()
                .def()
                .unwrap(),
        ),
        (
            "",
            modexp::NUM_ROWS,
            ModexpChiplet::new().unwrap().def().unwrap(),
        ),
    ];

    for (owner, composite) in [
        ("aes128", aes128.composite()),
        ("aes256", aes256.composite()),
        ("mlkem", mlkem.composite()),
        ("mldsa", mldsa.composite()),
    ] {
        defs.extend(
            composite
                .flatten_defs()
                .unwrap()
                .into_iter()
                .map(|d| (owner, PROBE_ROWS, d)),
        );
    }

    defs
}

#[test]
fn shipped_tables_match_checked_in_floor() {
    for (owner, rows, def) in all_tables() {
        let name = def.name();

        let row = EXPECTED
            .iter()
            .find(|(o, n, _)| *o == owner && *n == name)
            .unwrap_or_else(|| panic!("{owner}/{name} is not in the checked-in floor"));

        assert_eq!(measure(&def, rows), row.2, "{owner}/{name}");
    }
}

#[test]
fn floor_names_no_table_that_left_tree() {
    let live: Vec<(&str, String)> = all_tables()
        .into_iter()
        .map(|(owner, _, def)| (owner, def.name()))
        .collect();

    for (owner, name, _) in EXPECTED {
        assert!(
            live.iter().any(|(o, n)| o == owner && n == name),
            "{owner}/{name} is no longer shipped"
        );
    }
}
