// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_gadgets::{IntArithmeticChiplet, RamChiplet};
use hekate_keccak::KeccakChiplet;
use hekate_math::{Block128, Flat, HardwareField, TowerField};
use hekate_pqc::mldsa::{MlDsaChiplet, MlDsaParams};
use hekate_pqc::mlkem::{MlKemCall, MlKemChiplet, MlKemParams};
use hekate_program::Air;
use hekate_program::predicate::{ClaimLayout, compile, wire_values};

const KEM_CALLS: [MlKemCall; 3] = [MlKemCall::KeyGen, MlKemCall::Encaps, MlKemCall::Decaps];

type F = Block128;

fn mix(seed: u128) -> Flat<F> {
    F::from(
        seed.wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(0x51ed_2701),
    )
    .to_hardware()
}

fn check<A: Air<F>>(name: &str, air: &A) {
    let ast = air.constraint_ast();
    let width = air.num_columns();

    let current: Vec<Flat<F>> = (0..width).map(|i| mix(i as u128 + 1)).collect();
    let next: Vec<Flat<F>> = (0..width).map(|i| mix(i as u128 + 900_001)).collect();

    let consts = ast.precompute_hardware_consts();
    let mut node_values = Vec::new();
    ast.evaluate_into(&consts, &current, &next, &mut node_values);

    let wires = wire_values(&ast, &node_values);

    let layout = ClaimLayout {
        pad_first: 0,
        half: width as u32,
    };
    let rows = compile(&ast, layout);

    assert_eq!(wires.len(), rows.mul_nodes as usize, "{name}: wire count");

    let claims: Vec<Flat<F>> = current.iter().chain(next.iter()).copied().collect();
    let pad = vec![Flat::from_raw(F::ZERO); claims.len()];

    let expected = ast.evaluate(&current, &next);

    for (k, form) in rows.roots.iter().enumerate() {
        assert_eq!(
            form.evaluate(&claims, &pad, &wires),
            expected[k],
            "{name}: root {k}"
        );
    }

    for row in &rows.affine {
        let lhs: Flat<F> = row
            .unknowns
            .iter()
            .map(|&(u, c)| {
                c * match u {
                    hekate_program::predicate::Unknown::Pad(i) => pad[i as usize],
                    hekate_program::predicate::Unknown::Wire { mul, role } => {
                        wires[mul as usize][role as usize]
                    }
                }
            })
            .fold(Flat::from_raw(F::ZERO), |a, b| a + b);

        let rhs = row
            .claims
            .iter()
            .map(|&(i, c)| c * claims[i as usize])
            .fold(row.constant, |a, b| a + b);

        assert_eq!(lhs, rhs, "{name}: affine row");
    }

    for w in &wires {
        assert_eq!(w[0] * w[1], w[2], "{name}: hadamard row");
    }
}

#[test]
fn compiler_reproduces_evaluation_on_real_chiplets() {
    check("keccak", &KeccakChiplet::new(1 << 15, (1 << 15) / 25));

    for (level, params) in [
        ("ML-DSA-44", MlDsaParams::ML_DSA_44),
        ("ML-DSA-65", MlDsaParams::ML_DSA_65),
        ("ML-DSA-87", MlDsaParams::ML_DSA_87),
    ] {
        for def in MlDsaChiplet::<F>::new(params, &[32])
            .unwrap()
            .defs()
            .unwrap()
        {
            check(&format!("{level} {}", def.name()), &def);
        }
    }

    for (level, params) in [
        ("ML-KEM-512", MlKemParams::ML_KEM_512),
        ("ML-KEM-768", MlKemParams::ML_KEM_768),
        ("ML-KEM-1024", MlKemParams::ML_KEM_1024),
    ] {
        for def in MlKemChiplet::<F>::new(params, &KEM_CALLS)
            .unwrap()
            .defs()
            .unwrap()
        {
            check(&format!("{level} {}", def.name()), &def);
        }
    }

    check("ram", &RamChiplet::new(1 << 12, 1 << 12));
    check(
        "int_arith",
        &IntArithmeticChiplet::new(32, 1 << 12, 1 << 12).unwrap(),
    );
}
