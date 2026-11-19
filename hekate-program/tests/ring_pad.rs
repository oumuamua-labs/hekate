// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::config::Config;
use hekate_core::trace::{ColumnType, TraceCompatibleField};
use hekate_math::{Block128, Block256, Flat, HardwareField, TowerField};
use hekate_program::expander::{
    RingMaps, RingSwitchPlan, VirtualExpander, claim_weights, field_bits, field_coeffs, ring_target,
};
use hekate_program::fold_pool::{FoldPool, UnitWeights};
use hekate_program::linearized::{RingGadget, linearized_coeffs};
use hekate_program::outer::{
    EvalInputs, LinearBatch, OuterLayout, OuterStatement, eval_record, linear_tensor_vars,
    linear_weights, statement_rows,
};
use hekate_program::predicate::{Unknown, WireRole};

trait Draw: TraceCompatibleField + From<u128> {
    fn next(state: &mut u128) -> Self;
}

type K = Block256;

impl Draw for Block128 {
    fn next(state: &mut u128) -> Self {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;

        Block128(*state)
    }
}

impl Draw for Block256 {
    fn next(state: &mut u128) -> Self {
        let lo = Block128::next(state);
        let hi = Block128::next(state);

        Block256::new(lo, hi)
    }
}

fn linear_residual(
    layout: &OuterLayout,
    batch: &LinearBatch<K>,
    claims: &[Flat<K>],
    wires: &[[Flat<K>; 3]],
) -> Flat<K> {
    let zero = Flat::from_raw(K::ZERO);
    let mut message = vec![vec![zero; layout.message_len]; layout.total_rows()];

    for (i, &value) in claims.iter().enumerate() {
        let (row, col) = layout.slot(Unknown::Claim(i as u32));
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

fn horner_delta<F: Draw>(seed: u128) {
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
    let reads: Vec<bool> = (0..expander.num_virtual_columns())
        .map(|c| c % 3 == 0)
        .collect();

    let mut state = seed;
    for num_h in [0, 2] {
        let plan = RingSwitchPlan::new(&layout, Some(&entries), 2, num_h, &reads).unwrap();
        let total = plan.total_claims() + plan.next_claims();

        let claims: Vec<Flat<F>> = (0..total)
            .map(|_| F::next(&mut state).to_hardware())
            .collect();

        let unit_weights = UnitWeights {
            units: (0..plan.num_units)
                .map(|_| F::next(&mut state).to_hardware())
                .collect(),
            shift: F::next(&mut state).to_hardware(),
        };

        let r_mix: Vec<F> = (0..F::BITS.ilog2()).map(|_| F::next(&mut state)).collect();

        let expected = ring_target(&plan, &unit_weights, &claims, &mut RingMaps::new(&r_mix))
            .unwrap()
            .to_hardware();

        let mut whole_part = Flat::from_raw(F::ZERO);
        let mut ring: Vec<(u32, Flat<F>, usize)> = Vec::new();
        let mut ring_claims: Vec<Flat<F>> = Vec::new();
        let mut fields: Vec<ColumnType> = Vec::new();

        let weights = claim_weights(&plan, &unit_weights);

        for (c, (field, weight)) in weights.into_iter().enumerate() {
            if field == ColumnType::B128 {
                whole_part += weight * claims[c];
                continue;
            }

            let k = match fields.iter().position(|&f| f == field) {
                Some(k) => k,
                None => {
                    fields.push(field);
                    fields.len() - 1
                }
            };

            ring.push((c as u32, weight, k));
            ring_claims.push(claims[c]);
        }

        let maps: Vec<(Vec<F>, usize)> = fields
            .iter()
            .map(|&field| (field_coeffs(field, &r_mix), field_bits(field)))
            .collect();

        let mu = linearized_coeffs(&maps).unwrap();

        let gadget = RingGadget::new(ring, mu, 0).unwrap();

        assert_eq!(
            whole_part + gadget.delta(&ring_claims).unwrap(),
            expected,
            "{} bits, {num_h} h columns",
            F::BITS
        );
    }
}

/// `ring_target` of the claims equals the linear whole-unit part
/// plus the ring part through the Frobenius-Horner chain, on
/// a plan with ring, whole and blind units and shifted claims.
#[test]
fn ring_target_of_claims_is_horner_delta() {
    horner_delta::<Block128>(0x0101_0202_0303_0404_0505_0606_0707_0808);
    horner_delta::<Block256>(0x0101_0202_0303_0404_0505_0606_0707_0808);
}

/// Claims swapped after `r''` close the eval row over
/// their own oracle row, yet the linear batch rejects
/// their gadget wires against the committed claims.
#[test]
fn claims_swapped_after_ring_challenge_are_rejected() {
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
        &vec![true; expander.num_virtual_columns()],
    )
    .unwrap();

    let pool = FoldPool::new(&[(&plan, 6)], &config).unwrap();
    let plan = &pool.tables()[0].plan;
    let total = plan.total_claims() + plan.next_claims();

    let ring_claim = plan
        .units
        .iter()
        .find(|u| u.field != ColumnType::B128)
        .unwrap()
        .claims[0];

    let mut state = 0x0909_0a0a_0b0b_0c0c_0d0d_0e0e_0f0f_1010u128;
    let mut draw = || K::next(&mut state).to_hardware();

    let truth: Vec<Flat<K>> = (0..total).map(|_| draw()).collect();

    let mut committed = truth.clone();
    committed[ring_claim] += draw();

    let eta = draw();
    let rho: Vec<Flat<K>> = (0..pool.rho_vars()).map(|_| draw()).collect();
    let r_mix: Vec<K> = (0..K::BITS.ilog2()).map(|_| draw().to_tower()).collect();

    let unit_weights = pool.unit_weights(0, eta, &rho).unwrap();

    let fin = ring_target(plan, &unit_weights, &truth, &mut RingMaps::new(&r_mix))
        .unwrap()
        .to_hardware();

    let zero = Flat::from_raw(K::ZERO);
    let rho_tower: Vec<K> = rho.iter().map(|r| r.to_tower()).collect();

    let eval = eval_record(&EvalInputs {
        pool: &pool,
        eta: eta.to_tower(),
        rho: &rho_tower,
        r_mix: &r_mix,
        challenges: &[],
        claim_masked: zero,
        fin,
        claims: total,
        round_first: 0,
        first_wire: 0,
    })
    .unwrap();

    let gadget = eval.gadget.as_ref().unwrap();

    // `outer_geom` rejects empty pads;
    // with no round, no row reads this one scalar.
    let statement = OuterStatement {
        masked_scalars: 1,
        claims: total,
        mul_wires: K::BITS - 1,
    };

    let geom = config
        .outer_geom(
            statement.masked_scalars,
            statement.claims,
            statement.mul_wires,
            K::BITS,
        )
        .unwrap();

    let layout = OuterLayout::new(&geom, &statement).unwrap();

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

        gadget.wires(&ring).unwrap()
    };

    assert_eq!(
        linear_residual(&layout, &batch, &truth, &chain(&truth)),
        zero
    );
    assert_ne!(
        linear_residual(&layout, &batch, &committed, &chain(&truth)),
        zero
    );
    assert_ne!(
        linear_residual(&layout, &batch, &committed, &chain(&committed)),
        zero
    );
}
