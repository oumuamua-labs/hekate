// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! What each shipped table raises against an all-zero trace.

use hekate_aes::{Aes128Chiplet, Aes256Chiplet};
use hekate_gadgets::{IntArithmeticChiplet, ModexpChiplet, RamChiplet, RomChiplet};
use hekate_keccak::KeccakChiplet;
use hekate_math::{Block128, Flat, TowerField};
use hekate_pqc::mldsa::{MlDsaChiplet, MlDsaParams};
use hekate_pqc::mlkem::{MlKemCall, MlKemChiplet, MlKemParams};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::BoundaryTarget;
use hekate_program::{Air, FixedShape};
use hekate_sha2::Sha256Chiplet;

type F = Block128;

const PROBE_ROWS: usize = 256;
const PROBE_BLOCKS: usize = 4;
const PROBE_ROUNDS: usize = 4;
const SBOX_ROM_ROWS: usize = 256;

const KEM_CALLS: [MlKemCall; 3] = [MlKemCall::KeyGen, MlKemCall::Encaps, MlKemCall::Decaps];

/// `(owner, table)`; owner is empty for a standalone chiplet.
/// Measured at `PROBE_BLOCKS` blocks and `PROBE_ROWS` rows,
/// bar the fixed-height modexp and the one-call PQC pipelines.
#[rustfmt::skip]
const EXPECTED: &[(&str, &str, Floor)] = &[
    ("",       "KeccakChiplet",     Floor { roots: 1625, roots_violated: 0,  pins_firing: 26, boundaries_nonzero: 0 }),
    ("",       "RamChiplet",        Floor { roots: 190,  roots_violated: 2,  pins_firing: 3,  boundaries_nonzero: 0 }),
    ("",       "RomChiplet",        Floor { roots: 0,    roots_violated: 0,  pins_firing: 1,  boundaries_nonzero: 0 }),
    ("u32",    "ArithmeticChiplet", Floor { roots: 445,  roots_violated: 0,  pins_firing: 1,  boundaries_nonzero: 0 }),
    ("u64",    "ArithmeticChiplet", Floor { roots: 861,  roots_violated: 0,  pins_firing: 1,  boundaries_nonzero: 0 }),
    ("",       "Sha256Chiplet",     Floor { roots: 1960, roots_violated: 0,  pins_firing: 7,  boundaries_nonzero: 0 }),
    ("",       "ModexpChiplet",     Floor { roots: 3684, roots_violated: 152, pins_firing: 37, boundaries_nonzero: 0 }),
    ("dsa44",  "MlDsaCtrl",         Floor { roots: 32,   roots_violated: 0,  pins_firing: 18, boundaries_nonzero: 0 }),
    ("dsa44",  "SamplerChiplet",    Floor { roots: 757,  roots_violated: 81, pins_firing: 60, boundaries_nonzero: 0 }),
    ("dsa44",  "KeccakChiplet",     Floor { roots: 1625, roots_violated: 0,  pins_firing: 26, boundaries_nonzero: 0 }),
    ("dsa44",  "CodecChiplet",      Floor { roots: 3345, roots_violated: 578, pins_firing: 72, boundaries_nonzero: 0 }),
    ("dsa44",  "NttChiplet",        Floor { roots: 5928, roots_violated: 40, pins_firing: 24, boundaries_nonzero: 0 }),
    ("dsa44",  "HighBitsChiplet",   Floor { roots: 735,  roots_violated: 34, pins_firing: 19, boundaries_nonzero: 0 }),
    ("dsa65",  "MlDsaCtrl",         Floor { roots: 32,   roots_violated: 0,  pins_firing: 18, boundaries_nonzero: 0 }),
    ("dsa65",  "SamplerChiplet",    Floor { roots: 761,  roots_violated: 81, pins_firing: 71, boundaries_nonzero: 0 }),
    ("dsa65",  "KeccakChiplet",     Floor { roots: 1625, roots_violated: 0,  pins_firing: 26, boundaries_nonzero: 0 }),
    ("dsa65",  "CodecChiplet",      Floor { roots: 2165, roots_violated: 301, pins_firing: 69, boundaries_nonzero: 0 }),
    ("dsa65",  "NttChiplet",        Floor { roots: 5930, roots_violated: 40, pins_firing: 28, boundaries_nonzero: 0 }),
    ("dsa65",  "HighBitsChiplet",   Floor { roots: 1109, roots_violated: 42, pins_firing: 13, boundaries_nonzero: 0 }),
    ("dsa87",  "MlDsaCtrl",         Floor { roots: 32,   roots_violated: 0,  pins_firing: 18, boundaries_nonzero: 0 }),
    ("dsa87",  "SamplerChiplet",    Floor { roots: 765,  roots_violated: 81, pins_firing: 80, boundaries_nonzero: 0 }),
    ("dsa87",  "KeccakChiplet",     Floor { roots: 1625, roots_violated: 0,  pins_firing: 26, boundaries_nonzero: 0 }),
    ("dsa87",  "CodecChiplet",      Floor { roots: 2349, roots_violated: 308, pins_firing: 69, boundaries_nonzero: 0 }),
    ("dsa87",  "NttChiplet",        Floor { roots: 5932, roots_violated: 40, pins_firing: 32, boundaries_nonzero: 0 }),
    ("dsa87",  "HighBitsChiplet",   Floor { roots: 1109, roots_violated: 42, pins_firing: 13, boundaries_nonzero: 0 }),
    ("kem512", "MlKemCtrl",         Floor { roots: 32,   roots_violated: 0,  pins_firing: 16, boundaries_nonzero: 0 }),
    ("kem512", "SamplerChiplet",    Floor { roots: 1340, roots_violated: 48, pins_firing: 92, boundaries_nonzero: 0 }),
    ("kem512", "KeccakChiplet",     Floor { roots: 1625, roots_violated: 0,  pins_firing: 26, boundaries_nonzero: 0 }),
    ("kem512", "CodecChiplet",      Floor { roots: 10813, roots_violated: 576, pins_firing: 86, boundaries_nonzero: 0 }),
    ("kem512", "NttChiplet",        Floor { roots: 1712, roots_violated: 12, pins_firing: 12, boundaries_nonzero: 0 }),
    ("kem512", "PolyArithChiplet",  Floor { roots: 7153, roots_violated: 27, pins_firing: 21, boundaries_nonzero: 0 }),
    ("kem512", "KemSelectChiplet",  Floor { roots: 10,   roots_violated: 0,  pins_firing: 11, boundaries_nonzero: 0 }),
    ("kem768", "MlKemCtrl",         Floor { roots: 32,   roots_violated: 0,  pins_firing: 16, boundaries_nonzero: 0 }),
    ("kem768", "SamplerChiplet",    Floor { roots: 1014, roots_violated: 48, pins_firing: 56, boundaries_nonzero: 0 }),
    ("kem768", "KeccakChiplet",     Floor { roots: 1625, roots_violated: 0,  pins_firing: 26, boundaries_nonzero: 0 }),
    ("kem768", "CodecChiplet",      Floor { roots: 10813, roots_violated: 576, pins_firing: 86, boundaries_nonzero: 0 }),
    ("kem768", "NttChiplet",        Floor { roots: 1712, roots_violated: 12, pins_firing: 12, boundaries_nonzero: 0 }),
    ("kem768", "PolyArithChiplet",  Floor { roots: 7157, roots_violated: 27, pins_firing: 23, boundaries_nonzero: 0 }),
    ("kem768", "KemSelectChiplet",  Floor { roots: 10,   roots_violated: 0,  pins_firing: 11, boundaries_nonzero: 0 }),
    ("kem1024", "MlKemCtrl",        Floor { roots: 32,   roots_violated: 0,  pins_firing: 16, boundaries_nonzero: 0 }),
    ("kem1024", "SamplerChiplet",   Floor { roots: 1014, roots_violated: 48, pins_firing: 56, boundaries_nonzero: 0 }),
    ("kem1024", "KeccakChiplet",    Floor { roots: 1625, roots_violated: 0,  pins_firing: 26, boundaries_nonzero: 0 }),
    ("kem1024", "CodecChiplet",     Floor { roots: 24331, roots_violated: 856, pins_firing: 98, boundaries_nonzero: 0 }),
    ("kem1024", "NttChiplet",       Floor { roots: 1712, roots_violated: 12, pins_firing: 12, boundaries_nonzero: 0 }),
    ("kem1024", "PolyArithChiplet", Floor { roots: 7161, roots_violated: 27, pins_firing: 25, boundaries_nonzero: 0 }),
    ("kem1024", "KemSelectChiplet", Floor { roots: 10,   roots_violated: 0,  pins_firing: 11, boundaries_nonzero: 0 }),
    ("aes128", "AesRound128Air",    Floor { roots: 184,  roots_violated: 0,  pins_firing: 15, boundaries_nonzero: 0 }),
    ("aes128", "SboxRomChiplet",    Floor { roots: 241,  roots_violated: 0,  pins_firing: 1,  boundaries_nonzero: 0 }),
    ("aes256", "AesRound256Air",    Floor { roots: 164,  roots_violated: 0,  pins_firing: 19, boundaries_nonzero: 0 }),
    ("aes256", "SboxRomChiplet",    Floor { roots: 241,  roots_violated: 0,  pins_firing: 1,  boundaries_nonzero: 0 }),
];

#[derive(Debug, PartialEq, Eq)]
struct Floor {
    roots: usize,
    roots_violated: usize,
    pins_firing: usize,
    boundaries_nonzero: usize,
}

fn measure(def: &ChipletDef<F>) -> Floor {
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
        .filter(|pin| fires(&pin.shape))
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

fn fires(shape: &FixedShape<F>) -> bool {
    let nonzero = |values: &[F]| values.iter().any(|&v| v != F::ZERO);

    match shape {
        FixedShape::FirstRow | FixedShape::LastRow | FixedShape::Custom(_) => true,
        FixedShape::Periodic { values, .. } | FixedShape::Dense(values) => nonzero(values),
        FixedShape::Sparse(entries) => entries.iter().any(|&(_, v)| v != F::ZERO),
        FixedShape::Cadence { count, values, .. } => *count > 0 && nonzero(values),
        FixedShape::Segments(segments) => {
            segments.iter().any(|s| s.count > 0 && nonzero(&s.values))
        }
    }
}

fn all_tables() -> Vec<(&'static str, ChipletDef<F>)> {
    let aes128 = Aes128Chiplet::new(PROBE_ROWS, SBOX_ROM_ROWS, PROBE_BLOCKS).unwrap();
    let aes256 = Aes256Chiplet::new(PROBE_ROWS, SBOX_ROM_ROWS, PROBE_BLOCKS).unwrap();

    let mut defs = vec![
        (
            "",
            ChipletDef::from_air(&KeccakChiplet::new(PROBE_ROWS, PROBE_BLOCKS)).unwrap(),
        ),
        (
            "",
            ChipletDef::from_air(&RamChiplet::new(PROBE_ROWS, PROBE_ROWS)).unwrap(),
        ),
        (
            "",
            ChipletDef::from_air(&RomChiplet::new(PROBE_ROWS, PROBE_ROWS)).unwrap(),
        ),
        (
            "u32",
            ChipletDef::from_air(&IntArithmeticChiplet::new(32, PROBE_ROWS, PROBE_ROWS).unwrap())
                .unwrap(),
        ),
        (
            "u64",
            ChipletDef::from_air(&IntArithmeticChiplet::new(64, PROBE_ROWS, PROBE_ROWS).unwrap())
                .unwrap(),
        ),
        (
            "",
            Sha256Chiplet::<F>::new(PROBE_ROWS, PROBE_BLOCKS, PROBE_ROUNDS)
                .unwrap()
                .def()
                .unwrap(),
        ),
        ("", ModexpChiplet::new().unwrap().def().unwrap()),
    ];

    for (owner, params) in [
        ("dsa44", MlDsaParams::ML_DSA_44),
        ("dsa65", MlDsaParams::ML_DSA_65),
        ("dsa87", MlDsaParams::ML_DSA_87),
    ] {
        let pipeline = MlDsaChiplet::<F>::new(params, &[32]).unwrap();
        defs.extend(pipeline.defs().unwrap().into_iter().map(|d| (owner, d)));
    }

    for (owner, params) in [
        ("kem512", MlKemParams::ML_KEM_512),
        ("kem768", MlKemParams::ML_KEM_768),
        ("kem1024", MlKemParams::ML_KEM_1024),
    ] {
        let pipeline = MlKemChiplet::<F>::new(params, &KEM_CALLS).unwrap();
        defs.extend(pipeline.defs().unwrap().into_iter().map(|d| (owner, d)));
    }

    for (owner, tables) in [
        ("aes128", aes128.defs().unwrap()),
        ("aes256", aes256.defs().unwrap()),
    ] {
        defs.extend(tables.into_iter().map(|d| (owner, d)));
    }

    defs
}

#[test]
fn shipped_tables_match_checked_in_floor() {
    for (owner, def) in all_tables() {
        let name = def.name();

        let row = EXPECTED
            .iter()
            .find(|(o, n, _)| *o == owner && *n == name)
            .unwrap_or_else(|| panic!("{owner}/{name} is not in the checked-in floor"));

        assert_eq!(measure(&def), row.2, "{owner}/{name}");
    }
}

#[test]
fn floor_names_no_table_that_left_tree() {
    let live: Vec<(&str, String)> = all_tables()
        .into_iter()
        .map(|(owner, def)| (owner, def.name()))
        .collect();

    for (owner, name, _) in EXPECTED {
        assert!(
            live.iter().any(|(o, n)| o == owner && n == name),
            "{owner}/{name} is no longer shipped"
        );
    }
}
