// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::trace::ColumnType;
use hekate_math::{Block128, Flat, HardwareField, TowerField};
use hekate_program::expander::{
    RingSwitchPlan, VirtualExpander, claim_weights, eq_tensor_b, ring_target,
};
use hekate_program::linearized::{RingGadget, linearized_coeffs};

type K = Block128;

fn next(state: &mut u128) -> K {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;

    Block128(*state)
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
    let plan = RingSwitchPlan::new(&layout, Some(&entries), 2, 0).unwrap();
    let total = 2 * plan.total_claims();

    let mut state = 0x0101_0202_0303_0404_0505_0606_0707_0808u128;

    let claims: Vec<K> = (0..total).map(|_| next(&mut state)).collect();
    let pad: Vec<K> = (0..total).map(|_| next(&mut state)).collect();
    let masked: Vec<K> = claims.iter().zip(&pad).map(|(c, h)| *c + *h).collect();

    let eta = next(&mut state);
    let r_mix: Vec<K> = (0..7).map(|_| next(&mut state)).collect();

    let to_flat = |v: &[K]| -> Vec<Flat<K>> { v.iter().map(|x| x.to_hardware()).collect() };

    let t_masked = ring_target::<K>(&plan, &to_flat(&masked), eta, &r_mix, true);
    let t_plain = ring_target::<K>(&plan, &to_flat(&claims), eta, &r_mix, true);
    let expected = (t_masked - t_plain).to_hardware();

    let mut whole_part = Flat::from_raw(K::ZERO);
    let mut ring: Vec<(u32, Flat<K>)> = Vec::new();
    let mut ring_pad: Vec<Flat<K>> = Vec::new();

    for (c, (is_ring, weight)) in claim_weights::<K>(&plan, eta, true).into_iter().enumerate() {
        let h = pad[c].to_hardware();

        match is_ring {
            true => {
                ring.push((c as u32, weight.to_hardware()));
                ring_pad.push(h);
            }
            false => whole_part += weight.to_hardware() * h,
        }
    }

    let mu = linearized_coeffs(&eq_tensor_b(&r_mix));
    let gadget = RingGadget::new(ring, mu, 0);

    assert_eq!(whole_part + gadget.delta(&ring_pad), expected);
}
