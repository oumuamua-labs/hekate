// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::collections::BTreeMap;
use hekate_core::errors::{self, Error};
use hekate_math::{Flat, HardwareField, TowerField};
use hekate_program::permutation::{
    BUS_LIMBS, BusField, BusFraction, BusGroup, BusValue, BusWeights,
};

/// An `h` column's LogUp terms at the claims: its consistency,
/// `g·D + s` or for a pair `(g·D₁ + s₁)·D₂ + s₂·D₁`, and its
/// bus-sum `weight · g`, read out only through [`Self::weighted`].
pub struct ColumnTerms<F: HardwareField> {
    consistency: BusValue<F>,
    bus_sum: BusValue<F>,
}

impl<F: TowerField + HardwareField> ColumnTerms<F> {
    pub fn at(
        field: &BusField<F>,
        first: &BusFraction<F>,
        second: Option<&BusFraction<F>>,
        g: BusValue<F>,
        weight: BusValue<F>,
        claims: &[Flat<F>],
    ) -> Self {
        let d1 = first.denominator(claims);
        let s1 = BusValue::from_base(first.selector_value(claims));

        let consistency = match second {
            None => field.mul(g, d1) + s1,
            Some(second) => {
                let d2 = second.denominator(claims);
                let s2 = BusValue::from_base(second.selector_value(claims));

                field.mul(field.mul(g, d1) + s1, d2) + field.mul(s2, d1)
            }
        };

        Self {
            consistency,
            bus_sum: field.mul(weight, g),
        }
    }

    /// Sums limb `k` of the consistency times `w_k.consistency · eq_zc`
    /// and of the bus-sum times `w_k.bus_sum`, `w_k` the next weight
    /// of the table's `bus_alpha_weights`; errors when they run out.
    pub fn weighted(
        &self,
        weights: &mut impl Iterator<Item = BusWeights<F>>,
        eq_zc: Flat<F>,
    ) -> errors::Result<Flat<F>> {
        let mut acc = Flat::from_raw(F::ZERO);
        for (c, s) in self
            .consistency
            .limbs()
            .into_iter()
            .zip(self.bus_sum.limbs())
        {
            let w = weights.next().ok_or(Error::Protocol {
                protocol: "LogUp",
                message: "bus weights end before the column's limbs",
            })?;

            acc += w.consistency * c * eq_zc + w.bus_sum * s;
        }

        Ok(acc)
    }
}

/// Rejects unless every group's claimed sums cancel limb
/// by limb, one `(group, limbs)` entry per `h` column.
pub fn check_bus_sum_matching<F: TowerField>(
    columns: &[(BusGroup, [F; BUS_LIMBS])],
) -> errors::Result<()> {
    let mut by_group: BTreeMap<&BusGroup, [F; BUS_LIMBS]> = BTreeMap::new();
    for (group, limbs) in columns {
        let acc = by_group.entry(group).or_insert([F::ZERO; BUS_LIMBS]);

        for (total, &limb) in acc.iter_mut().zip(limbs) {
            *total += limb;
        }
    }

    if by_group.values().flatten().any(|&total| total != F::ZERO) {
        return Err(Error::Protocol {
            protocol: "LogUp",
            message: "bus endpoint sums do not cancel to zero",
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;
    use alloc::vec::Vec;
    use hekate_math::{Block128, Block256};
    use hekate_program::permutation::bus_alpha_weights;

    type F = Block128;

    fn field() -> BusField<F> {
        BusField::new().unwrap()
    }

    fn value(lo: u128, hi: u128) -> BusValue<F> {
        field().import(Block256::new(Block128(lo), Block128(hi)))
    }

    fn invert(x: BusValue<F>) -> BusValue<F> {
        field().import(Block256::new(x.lo.to_tower(), x.hi.to_tower()).invert())
    }

    fn fraction(seed: u128, selector: Option<u32>) -> BusFraction<F> {
        BusFraction {
            slots: vec![(0, value(seed, seed + 1)), (1, value(seed + 2, seed + 3))],
            constant: value(seed + 4, seed + 5),
            selector,
            recv_selector: None,
        }
    }

    fn claims() -> Vec<Flat<F>> {
        [0x11u128, 0x22, 0x33]
            .map(|v| F::from(v).to_hardware())
            .to_vec()
    }

    fn quotient(fraction: &BusFraction<F>, claims: &[Flat<F>]) -> BusValue<F> {
        let s = BusValue::from_base(fraction.selector_value(claims));

        field().mul(s, invert(fraction.denominator(claims)))
    }

    #[test]
    fn honest_h_zeroes_consistency_of_one_fraction() {
        let claims = claims();
        let first = fraction(100, Some(2));
        let g = quotient(&first, &claims);

        let terms = ColumnTerms::at(&field(), &first, None, g, BusValue::one(), &claims);

        assert!(terms.consistency.is_zero());
    }

    #[test]
    fn honest_h_zeroes_consistency_of_packed_pair() {
        let claims = claims();
        let first = fraction(100, Some(2));
        let second = fraction(200, None);
        let g = quotient(&first, &claims) + quotient(&second, &claims);

        let terms = ColumnTerms::at(&field(), &first, Some(&second), g, BusValue::one(), &claims);

        assert!(terms.consistency.is_zero());
    }

    #[test]
    fn shifted_h_breaks_consistency() {
        let claims = claims();
        let first = fraction(100, Some(2));
        let second = fraction(200, Some(2));

        for pair in [None, Some(&second)] {
            let honest = match pair {
                None => quotient(&first, &claims),
                Some(second) => quotient(&first, &claims) + quotient(second, &claims),
            };

            for shift in [value(1, 0), value(0, 1)] {
                let terms = ColumnTerms::at(
                    &field(),
                    &first,
                    pair,
                    honest + shift,
                    BusValue::one(),
                    &claims,
                );

                assert!(!terms.consistency.is_zero());
            }
        }
    }

    #[test]
    fn bus_sum_is_row_weight_times_h() {
        let field = field();
        let claims = claims();
        let first = fraction(100, Some(2));
        let g = value(7, 8);
        let weight = value(9, 10);

        let terms = ColumnTerms::at(&field, &first, None, g, weight, &claims);

        assert_eq!(terms.bus_sum, field.mul(weight, g));
    }

    #[test]
    fn bus_matching_zero_sum_honest_pair() {
        let limbs = [F::from(0x9999u128), F::from(0x7777u128)];
        let columns = vec![
            (BusGroup::Lookup("rom_link".to_string()), limbs),
            (BusGroup::Lookup("rom_link".to_string()), limbs),
        ];

        check_bus_sum_matching(&columns).unwrap();
    }

    #[test]
    fn bus_matching_rejects_mismatch_in_either_limb() {
        let base = [F::from(0x9999u128), F::from(0x7777u128)];
        for limb in 0..BUS_LIMBS {
            let mut other = base;
            other[limb] += F::ONE;

            let columns = vec![
                (BusGroup::Lookup("rom_link".to_string()), base),
                (BusGroup::Lookup("rom_link".to_string()), other),
            ];

            assert!(check_bus_sum_matching(&columns).is_err());
        }
    }

    #[test]
    fn bus_matching_rejects_unpaired_column() {
        let columns = vec![(BusGroup::Permutation, [F::from(0x9999u128), F::ZERO])];

        assert!(check_bus_sum_matching(&columns).is_err());
    }

    #[test]
    fn permutation_columns_cancel_as_one_group() {
        let a = [F::from(0x1u128), F::from(0x2u128)];
        let b = [F::from(0x4u128), F::from(0x8u128)];
        let sum = [a[0] + b[0], a[1] + b[1]];

        let columns = vec![
            (BusGroup::Permutation, a),
            (BusGroup::Permutation, b),
            (BusGroup::Permutation, sum),
        ];

        check_bus_sum_matching(&columns).unwrap();
    }

    #[test]
    fn lookup_buses_cancel_apart() {
        let a = [F::from(0x1u128), F::ZERO];
        let b = [F::from(0x2u128), F::ZERO];

        let balanced = vec![
            (BusGroup::Lookup("ram".to_string()), a),
            (BusGroup::Lookup("ram".to_string()), a),
            (BusGroup::Lookup("rom".to_string()), b),
            (BusGroup::Lookup("rom".to_string()), b),
        ];

        let crossed = vec![
            (BusGroup::Lookup("ram".to_string()), a),
            (BusGroup::Lookup("rom".to_string()), a),
            (BusGroup::Lookup("ram".to_string()), b),
            (BusGroup::Lookup("rom".to_string()), b),
        ];

        check_bus_sum_matching(&balanced).unwrap();

        assert!(check_bus_sum_matching(&crossed).is_err());
    }

    #[test]
    fn weighted_takes_one_weight_per_limb() {
        let claims = claims();
        let first = fraction(100, Some(2));
        let terms = ColumnTerms::at(&field(), &first, None, value(7, 8), value(9, 10), &claims);

        let start = F::from(3u128).to_hardware();
        let alpha = F::from(5u128).to_hardware();
        let eq_zc = F::from(11u128).to_hardware();

        let mut weights = bus_alpha_weights(start, alpha);
        let next = bus_alpha_weights(start, alpha).nth(BUS_LIMBS).unwrap();

        terms.weighted(&mut weights, eq_zc).unwrap();

        assert_eq!(weights.next().unwrap().consistency, next.consistency);

        let mut short = bus_alpha_weights(start, alpha).take(BUS_LIMBS - 1);

        assert!(terms.weighted(&mut short, eq_zc).is_err());
    }
}
