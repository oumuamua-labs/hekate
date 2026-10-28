// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::config::Config;
use hekate_core::trace::ColumnType;
use hekate_math::{Block128, Flat, HardwareField, TowerField};
use hekate_program::expander::{
    RingSwitchPlan, VirtualExpander, claim_weights, field_bits, field_coeffs, ring_target,
};
use hekate_program::fold_pool::{FoldPool, UnitWeights};
use hekate_program::linearized::{self, RingGadget, linearized_coeffs};
use hekate_program::outer::{
    EvalInputs, LinearBatch, OuterLayout, OuterStatement, eval_record, linear_tensor_vars,
    linear_weights, statement_rows,
};
use hekate_program::predicate::{Unknown, WireRole};

type K = Block128;

fn next(state: &mut u128) -> K {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;

    Block128(*state)
}

fn linear_residual(
    layout: &OuterLayout,
    batch: &LinearBatch<K>,
    pad: &[Flat<K>],
    wires: &[[Flat<K>; 3]],
) -> Flat<K> {
    let zero = Flat::from_raw(K::ZERO);
    let mut message = vec![vec![zero; layout.message_len]; layout.total_rows()];

    for (i, &value) in pad.iter().enumerate() {
        let (row, col) = layout.slot(Unknown::Pad(i as u32));
        message[row][col] = value;
    }

    let roles = [WireRole::Lhs, WireRole::Rhs, WireRole::Product];

    for (mul, triple) in wires.iter().enumerate() {
        for (&role, &value) in roles.iter().zip(triple) {
            let (row, col) = layout.slot(Unknown::Wire {
                mul: mul as u32,
                role,
            });

            message[row][col] = value;
        }
    }

    let mut residual = batch.target;
    for (weights, &row) in batch.weights.iter().zip(&batch.rows) {
        for (&w, &v) in weights.iter().zip(&message[row]) {
            residual += w * v;
        }
    }

    residual
}

/// `ring_target(c + h) - ring_target(c)` equals the
/// K-linear whole-unit part plus the ring part through
/// the Frobenius-Horner chain, on a plan with ring,
/// whole and blind units and shifted claims.
#[test]
fn pad_contribution_to_ring_target_is_horner_delta() {
    let layout = [
        ColumnType::B32,
        ColumnType::B32,
        ColumnType::B64,
        ColumnType::Bit,
    ];
    let expander = VirtualExpander::new()
        .expand_bits(2, ColumnType::B32)
        .pass_through(1, ColumnType::B64)
        .control_bits(1)
        .build()
        .unwrap();

    let entries = expander.expansion_entries();

    let mut state = 0x0101_0202_0303_0404_0505_0606_0707_0808u128;

    for num_h in [0, 2] {
        let plan = RingSwitchPlan::new(&layout, Some(&entries), 2, num_h).unwrap();
        let total = 2 * plan.total_claims();

        let claims: Vec<K> = (0..total).map(|_| next(&mut state)).collect();
        let pad: Vec<K> = (0..total).map(|_| next(&mut state)).collect();
        let masked: Vec<K> = claims.iter().zip(&pad).map(|(c, h)| *c + *h).collect();

        let unit_weights = UnitWeights {
            units: (0..plan.num_units)
                .map(|_| next(&mut state).to_hardware())
                .collect(),
            shift: next(&mut state).to_hardware(),
        };

        let r_mix: Vec<K> = (0..7).map(|_| next(&mut state)).collect();

        let to_flat = |v: &[K]| -> Vec<Flat<K>> { v.iter().map(|x| x.to_hardware()).collect() };

        let t_masked = ring_target(&plan, &unit_weights, &to_flat(&masked), &r_mix, true);
        let t_plain = ring_target(&plan, &unit_weights, &to_flat(&claims), &r_mix, true);

        let expected = (t_masked - t_plain).to_hardware();

        let mut whole_part = Flat::from_raw(K::ZERO);
        let mut ring: Vec<(u32, Flat<K>, usize)> = Vec::new();
        let mut ring_pad: Vec<Flat<K>> = Vec::new();
        let mut fields: Vec<ColumnType> = Vec::new();

        let weights = claim_weights(&plan, &unit_weights, true);

        for (c, (field, weight)) in weights.into_iter().enumerate() {
            let h = pad[c].to_hardware();

            if field == ColumnType::B128 {
                whole_part += weight.to_hardware() * h;
                continue;
            }

            let k = match fields.iter().position(|&f| f == field) {
                Some(k) => k,
                None => {
                    fields.push(field);
                    fields.len() - 1
                }
            };

            ring.push((c as u32, weight.to_hardware(), k));
            ring_pad.push(h);
        }

        let maps: Vec<(Vec<K>, usize)> = fields
            .iter()
            .map(|&field| (field_coeffs(field, &r_mix), field_bits(field)))
            .collect();

        let mu = linearized_coeffs(&maps);

        let gadget = RingGadget::new(ring, mu, 0);

        assert_eq!(
            whole_part + gadget.delta(&ring_pad),
            expected,
            "{num_h} h columns"
        );
    }
}

/// A pad swapped after `r''` closes the eval row on
/// a false claim over its own oracle row, yet the
/// linear batch rejects it against the committed pad.
#[test]
fn pad_swapped_after_ring_challenge_is_rejected() {
    let config = Config {
        zero_knowledge: true,
        ..Config::dev()
    };

    let expander = VirtualExpander::new()
        .expand_bits(1, ColumnType::B32)
        .pass_through(1, ColumnType::B64)
        .build()
        .unwrap();

    let entries = expander.expansion_entries();
    let plan = RingSwitchPlan::new(
        &[ColumnType::B32, ColumnType::B64],
        Some(&entries),
        config.blind_units(),
        0,
    )
    .unwrap();

    let pool = FoldPool::new(&[(&plan, 6)], &config).unwrap();
    let plan = &pool.tables()[0].plan;
    let total = 2 * plan.total_claims();

    let ring_claim = plan
        .units
        .iter()
        .find(|u| u.field != ColumnType::B128)
        .unwrap()
        .claims[0];

    let mut state = 0x0909_0a0a_0b0b_0c0c_0d0d_0e0e_0f0f_1010u128;
    let mut draw = || next(&mut state).to_hardware();

    let truth: Vec<Flat<K>> = (0..total).map(|_| draw()).collect();
    let pad: Vec<Flat<K>> = (0..total).map(|_| draw()).collect();

    let mut forged = truth.clone();
    forged[ring_claim] += draw();

    let masked: Vec<Flat<K>> = forged.iter().zip(&pad).map(|(&c, &h)| c + h).collect();

    let eta = draw();
    let rho: Vec<Flat<K>> = (0..pool.rho_vars()).map(|_| draw()).collect();
    let r_mix: Vec<K> = (0..7).map(|_| draw().to_tower()).collect();

    let unit_weights = pool.unit_weights(0, eta, &rho).unwrap();

    let target =
        |claims: &[Flat<K>]| ring_target(plan, &unit_weights, claims, &r_mix, true).to_hardware();

    let swapped: Vec<Flat<K>> = masked.iter().zip(&truth).map(|(&m, &c)| m + c).collect();

    let claim_pads: Vec<u32> = (0..total as u32).collect();
    let rho_tower: Vec<K> = rho.iter().map(|r| r.to_tower()).collect();

    let eval = eval_record(&EvalInputs {
        pool: &pool,
        eta: eta.to_tower(),
        rho: &rho_tower,
        r_mix: &r_mix,
        shifted_claims: true,
        challenges: &[],
        claim_masked: target(&masked),
        fin: target(&truth),
        claim_pads: &claim_pads,
        round_first: total as u32,
        first_wire: 0,
    })
    .unwrap();

    let gadget = eval.gadget.as_ref().unwrap();

    let statement = OuterStatement {
        masked_scalars: total,
        mul_wires: linearized::BITS - 1,
    };

    let geom = config
        .outer_geom(
            statement.masked_scalars,
            statement.mul_wires,
            size_of::<K>() * 8,
        )
        .unwrap();

    let layout = OuterLayout::new(&geom, statement.masked_scalars, statement.mul_wires).unwrap();

    let rows = statement_rows(&[], &eval, &statement).unwrap();
    let tensor: Vec<K> = (0..linear_tensor_vars(rows))
        .map(|_| draw().to_tower())
        .collect();

    let batch = linear_weights(&layout, &statement, &[], &eval, &tensor).unwrap();

    let chain = |values: &[Flat<K>]| {
        let ring: Vec<Flat<K>> = gadget
            .ring
            .iter()
            .map(|&(c, _, _)| values[c as usize])
            .collect();

        gadget.wires(&ring)
    };

    let zero = Flat::from_raw(K::ZERO);

    assert_eq!(
        linear_residual(&layout, &batch, &swapped, &chain(&swapped)),
        zero
    );
    assert_ne!(
        linear_residual(&layout, &batch, &pad, &chain(&swapped)),
        zero
    );
    assert_ne!(linear_residual(&layout, &batch, &pad, &chain(&pad)), zero);
}
