// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_aes::{Aes128Chiplet, Aes256Chiplet};
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_gadgets::{IntArithmeticChiplet, ModexpChiplet, RamChiplet, RomChiplet};
use hekate_keccak::KeccakChiplet;
use hekate_math::{Block128, Flat, HardwareField, TowerField};
use hekate_pqc::mldsa::{MlDsaChiplet, MlDsaParams};
use hekate_pqc::mlkem::{MlKemCall, MlKemChiplet, MlKemParams};
use hekate_program::chiplet::ChipletDef;
use hekate_program::expander::RingSwitchPlan;
use hekate_program::fold_pool::FoldPool;
use hekate_program::linearized::{self, RingGadget};
use hekate_program::outer::{
    BusColumnRow, ConsistencyInputs, EvalRecord, OuterLayout, OuterStatement, TableRecord,
    TableShape, assemble, linear_tensor_vars, linear_weights, linear_weights_of_rows,
    statement_rows, table_plan,
};
use hekate_program::permutation::{BUS_LIMBS, BusFraction, BusKind, BusValue, Slot, bus_columns};
use hekate_program::predicate::Unknown;
use hekate_sha2::Sha256Chiplet;

const NUM_ROWS: usize = 256;
const KEM_CALLS: [MlKemCall; 3] = [MlKemCall::KeyGen, MlKemCall::Encaps, MlKemCall::Decaps];
const KARATSUBA_WIRES: u32 = 3;

type F = Block128;
type Snapshot = (&'static str, errors::Result<Vec<ChipletDef<F>>>);

fn mix(seed: u128) -> Flat<F> {
    F::from(
        seed.wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(0x51ed_2701),
    )
    .to_hardware()
}

fn shipped_tables() -> Vec<Snapshot> {
    vec![
        (
            "KeccakChiplet",
            ChipletDef::from_air(&KeccakChiplet::new(NUM_ROWS, 4)).map(|d| vec![d]),
        ),
        (
            "RamChiplet",
            ChipletDef::from_air(&RamChiplet::new(NUM_ROWS, NUM_ROWS)).map(|d| vec![d]),
        ),
        (
            "RomChiplet",
            ChipletDef::from_air(&RomChiplet::new(NUM_ROWS, NUM_ROWS)).map(|d| vec![d]),
        ),
        (
            "IntArithmeticChiplet",
            ChipletDef::from_air(&IntArithmeticChiplet::new(32, NUM_ROWS, NUM_ROWS).unwrap())
                .map(|d| vec![d]),
        ),
        (
            "Aes128Chiplet",
            Aes128Chiplet::new(NUM_ROWS, NUM_ROWS, 4).unwrap().defs(),
        ),
        (
            "Aes256Chiplet",
            Aes256Chiplet::new(NUM_ROWS, NUM_ROWS, 4).unwrap().defs(),
        ),
        (
            "Sha256Chiplet",
            Sha256Chiplet::<F>::new(NUM_ROWS, 4, 4)
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

/// A record over `def`'s AST, bus specs and fixed
/// pins with random masked values, its pads taken
/// from `base` on; returns the next free pad.
fn record(
    def: &ChipletDef<F>,
    shape: TableShape,
    blind_units: usize,
    base: u32,
) -> (TableRecord<'_, F>, u32) {
    let statics = def.statics();
    let seed = base as u128;

    let half = shape.eval_claims() as u32;
    let columns = shape.num_columns as u32;
    let limbs = shape.h_columns() as u32;

    let h_pad = base + 2 * half;
    let sums_pad = h_pad + limbs;
    let mask_pad = sums_pad + limbs;

    let wide = |s: u128| BusValue {
        lo: mix(s),
        hi: mix(s + 7),
    };

    let fraction = |k: usize| {
        let spec = &statics.specs[k].1;

        BusFraction {
            slots: spec
                .slots()
                .enumerate()
                .filter_map(|(j, slot)| match slot {
                    Slot::Column(c) => Some((c as u32, wide(seed + 1000 * k as u128 + j as u128))),
                    _ => None,
                })
                .collect(),
            constant: wide(seed + 30 + k as u128),
            selector: spec.selector.map(|s| s as u32),
            recv_selector: spec.recv_selector.map(|s| s as u32),
        }
    };

    let mut buses = Vec::new();
    let mut claimed_sums = Vec::new();
    let mut first_wire = shape.mul_nodes as u32;

    for (c, column) in bus_columns(statics.specs, shape.sumcheck_degree)
        .iter()
        .enumerate()
    {
        let group = column.group(statics.specs);
        let limb = |j: usize| (BUS_LIMBS * c + j) as u32;

        for j in 0..BUS_LIMBS {
            claimed_sums.push((
                group.clone(),
                j,
                sums_pad + limb(j),
                mix(seed + 90 + limb(j) as u128),
            ));
        }

        let second = column.second.map(fraction);

        let wires = match &second {
            None => KARATSUBA_WIRES,
            Some(f) if f.selector_claimed() => 2 * KARATSUBA_WIRES + BUS_LIMBS as u32,
            Some(_) => 2 * KARATSUBA_WIRES,
        };

        buses.push(BusColumnRow {
            h_pads: core::array::from_fn(|j| h_pad + limb(j)),
            h_masked: core::array::from_fn(|j| mix(seed + 10 + limb(j) as u128)),
            first: fraction(column.first),
            second,
            eq_lookup: match column.kind {
                BusKind::Permutation => BusValue::one(),
                BusKind::Lookup => wide(seed + 20 + c as u128),
            },
            first_wire,
        });

        first_wire += wires;
    }

    assert_eq!(first_wire as usize, shape.mul_wires());

    let record = TableRecord {
        claims_masked: (0..2 * half as u128).map(|i| mix(seed + 500 + i)).collect(),
        ast: statics.ast,
        consistency: ConsistencyInputs {
            pad_first: base,
            eq_zc: mix(seed + 1),
            alpha: mix(seed + 2),
            tau: F::EXTENSION_TAU.to_hardware(),
            boundary: statics
                .boundary
                .iter()
                .enumerate()
                .map(|(i, bc)| {
                    (
                        bc.col_idx as u32,
                        mix(seed + 50 + i as u128),
                        mix(seed + 60),
                    )
                })
                .collect(),
            telescope: (0..blind_units as u32)
                .map(|k| (columns + k, half + columns + k))
                .collect(),
            buses,
            val_final_form: (0..3)
                .map(|j| (Unknown::Pad(mask_pad + j), mix(seed + 70 + j as u128)))
                .collect(),
            val_final_masked: mix(seed + 5),
        },
        fixed: statics
            .fixed
            .iter()
            .map(|fc| {
                let col = fc.col_idx as u32;

                (base + col, col, mix(seed + 80 + col as u128))
            })
            .collect(),
        claimed_sums,
        bindings: Vec::new(),
        shape,
    };

    (record, mask_pad + 3)
}

/// The pool's eval record with random masked values,
/// its pads from `base` on and its gadget's wires
/// from `first_wire` on; returns the next free pad.
fn eval(subfield_units: bool, base: u32, first_wire: u32) -> (EvalRecord<F>, u32) {
    let seed = base as u128;
    let ring_pad = base + 3;

    let mut mask_form: Vec<(Unknown, Flat<F>)> = (0..3)
        .map(|j| (Unknown::Pad(base + j), mix(seed + 30 + j as u128)))
        .collect();

    let gadget = subfield_units.then(|| {
        let entries = (0..4)
            .map(|c| (ring_pad + c, mix(seed + 40 + c as u128), c as usize % 2))
            .collect();

        let mu = (0..2u128)
            .map(|k| {
                (0..linearized::BITS as u128)
                    .map(|j| mix(seed + 100 + 200 * k + j))
                    .collect()
            })
            .collect();

        RingGadget::new(entries, mu, first_wire)
    });

    if let Some(gadget) = &gadget {
        mask_form.extend(gadget.delta_form());
    }

    let record = EvalRecord {
        mask_form,
        claim_masked: mix(seed + 6),
        fin: mix(seed + 7),
        gadget,
    };

    (record, ring_pad + 4)
}

#[test]
fn reverse_pass_reproduces_assembled_weights_on_shipped_tables() {
    let config = Config::prod();
    let blind_units = config.blind_units();
    let num_vars = NUM_ROWS.trailing_zeros() as usize;
    let field_bits = size_of::<F>() * 8;

    for (label, snapshot) in shipped_tables() {
        let defs = match snapshot {
            Ok(defs) => defs,
            Err(e) => panic!("{label}: snapshot rejected: {e}"),
        };

        let shapes: Vec<TableShape> = defs
            .iter()
            .map(|def| TableShape::from_air(def, &def.statics()).unwrap())
            .collect();

        let plans: Vec<RingSwitchPlan> = defs
            .iter()
            .zip(&shapes)
            .map(|(def, shape)| table_plan(def, shape, &config).unwrap())
            .collect();

        let heights: Vec<(&RingSwitchPlan, usize)> = plans.iter().map(|p| (p, num_vars)).collect();
        let pool = FoldPool::new(&heights, &config).unwrap();

        let mut records = Vec::with_capacity(defs.len());
        let mut base = 0;

        for ((def, shape), layout) in defs.iter().zip(&shapes).zip(pool.tables()) {
            let (table, next) = record(def, shape.at(layout), blind_units, base);
            records.push(table);

            base = next;
        }

        let subfield_units = records.iter().any(|r| r.shape.subfield_units);
        let table_wires: usize = records.iter().map(|r| r.shape.mul_wires()).sum();

        let (eval, next) = eval(subfield_units, base, table_wires as u32);

        let gadget_wires = match subfield_units {
            true => linearized::BITS - 1,
            false => 0,
        };

        let statement = OuterStatement {
            masked_scalars: next as usize,
            mul_wires: table_wires + gadget_wires,
        };

        let geom = config
            .outer_geom(statement.masked_scalars, statement.mul_wires, field_bits)
            .unwrap();

        let layout =
            OuterLayout::new(&geom, statement.masked_scalars, statement.mul_wires).unwrap();

        let rows = statement_rows(&records, &eval, &statement).unwrap();
        let assembled = assemble(&records, &eval, &statement).unwrap();

        assert_eq!(assembled.affine.len(), rows, "{label}");

        let tensor: Vec<F> = (0..linear_tensor_vars(rows) as u128)
            .map(|i| mix(i).to_tower())
            .collect();

        let got = linear_weights(&layout, &statement, &records, &eval, &tensor).unwrap();
        let want = linear_weights_of_rows(&layout, &assembled, &tensor).unwrap();

        assert_eq!(got.rows, want.rows, "{label}");
        assert_eq!(got.weights, want.weights, "{label}");
        assert_eq!(got.target, want.target, "{label}");
    }
}
