// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::collections::BTreeMap;
use hekate_math::{Flat, HardwareField, TowerField};
use hekate_program::permutation::{BusFraction, BusGroup, BusWeights};

use crate::rejection::Rejection;

/// An `h` column's LogUp terms at the claims: its consistency,
/// `g·D + s` or for a pair `(g·D₁ + s₁)·D₂ + s₂·D₁`, and its
/// bus-sum `weight · g`, read out only through [`Self::weighted`].
pub struct ColumnTerms<F: HardwareField> {
    consistency: Flat<F>,
    bus_sum: Flat<F>,
}

impl<F: TowerField + HardwareField> ColumnTerms<F> {
    pub fn at(
        first: &BusFraction<F>,
        second: Option<&BusFraction<F>>,
        g: Flat<F>,
        weight: Flat<F>,
        claims: &[Flat<F>],
    ) -> Self {
        let d1 = first.denominator(claims);
        let s1 = first.selector_value(claims);

        let consistency = match second {
            None => g * d1 + s1,
            Some(second) => {
                let d2 = second.denominator(claims);
                let s2 = second.selector_value(claims);

                (g * d1 + s1) * d2 + s2 * d1
            }
        };

        Self {
            consistency,
            bus_sum: weight * g,
        }
    }

    /// Sums the consistency times `w.consistency · eq_zc`
    /// and the bus-sum times `w.bus_sum`.
    pub fn weighted(&self, w: BusWeights<F>, eq_zc: Flat<F>) -> Flat<F> {
        w.consistency * self.consistency * eq_zc + w.bus_sum * self.bus_sum
    }
}

/// Rejects unless every group's claimed sums cancel,
/// one `(group, sum)` entry per `h` column.
pub fn check_bus_sum_matching<F: TowerField>(columns: &[(BusGroup, F)]) -> Result<(), Rejection> {
    let mut by_group: BTreeMap<&BusGroup, F> = BTreeMap::new();
    for (group, sum) in columns {
        *by_group.entry(group).or_insert(F::ZERO) += *sum;
    }

    if by_group.values().any(|&total| total != F::ZERO) {
        return Err(Rejection::BusSum);
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

    type F = Block256;

    fn value(lo: u128, hi: u128) -> Flat<F> {
        Block256::new(Block128(lo), Block128(hi)).to_hardware()
    }

    fn invert(x: Flat<F>) -> Flat<F> {
        x.to_tower().invert().to_hardware()
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

    fn quotient(fraction: &BusFraction<F>, claims: &[Flat<F>]) -> Flat<F> {
        fraction.selector_value(claims) * invert(fraction.denominator(claims))
    }

    #[test]
    fn honest_h_zeroes_consistency_of_one_fraction() {
        let claims = claims();
        let first = fraction(100, Some(2));
        let g = quotient(&first, &claims);

        let terms = ColumnTerms::at(&first, None, g, Flat::from_raw(F::ONE), &claims);

        assert_eq!(terms.consistency, Flat::from_raw(F::ZERO));
    }

    #[test]
    fn honest_h_zeroes_consistency_of_packed_pair() {
        let claims = claims();
        let first = fraction(100, Some(2));
        let second = fraction(200, None);
        let g = quotient(&first, &claims) + quotient(&second, &claims);

        let terms = ColumnTerms::at(&first, Some(&second), g, Flat::from_raw(F::ONE), &claims);

        assert_eq!(terms.consistency, Flat::from_raw(F::ZERO));
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
                    &first,
                    pair,
                    honest + shift,
                    Flat::from_raw(F::ONE),
                    &claims,
                );

                assert_ne!(terms.consistency, Flat::from_raw(F::ZERO));
            }
        }
    }

    #[test]
    fn bus_sum_is_row_weight_times_h() {
        let claims = claims();
        let first = fraction(100, Some(2));
        let g = value(7, 8);
        let weight = value(9, 10);

        let terms = ColumnTerms::at(&first, None, g, weight, &claims);

        assert_eq!(terms.bus_sum, weight * g);
    }

    #[test]
    fn bus_matching_zero_sum_honest_pair() {
        let sum = F::from(0x9999u128);
        let columns = vec![
            (BusGroup::Lookup("rom_link".to_string()), sum),
            (BusGroup::Lookup("rom_link".to_string()), sum),
        ];

        check_bus_sum_matching(&columns).unwrap();
    }

    #[test]
    fn bus_matching_rejects_mismatch() {
        let base = F::from(0x9999u128);
        let columns = vec![
            (BusGroup::Lookup("rom_link".to_string()), base),
            (BusGroup::Lookup("rom_link".to_string()), base + F::ONE),
        ];

        assert!(check_bus_sum_matching(&columns).is_err());
    }

    #[test]
    fn bus_matching_rejects_unpaired_column() {
        let columns = vec![(BusGroup::Permutation, F::from(0x9999u128))];

        assert!(check_bus_sum_matching(&columns).is_err());
    }

    #[test]
    fn permutation_columns_cancel_as_one_group() {
        let a = F::from(0x1u128);
        let b = F::from(0x4u128);

        let columns = vec![
            (BusGroup::Permutation, a),
            (BusGroup::Permutation, b),
            (BusGroup::Permutation, a + b),
        ];

        check_bus_sum_matching(&columns).unwrap();
    }

    #[test]
    fn lookup_buses_cancel_apart() {
        let a = F::from(0x1u128);
        let b = F::from(0x2u128);

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
    fn weighted_applies_the_column_weights() {
        let claims = claims();
        let first = fraction(100, Some(2));
        let terms = ColumnTerms::at(&first, None, value(7, 8), value(9, 10), &claims);

        let start = F::from(3u128).to_hardware();
        let alpha = F::from(5u128).to_hardware();
        let eq_zc = F::from(11u128).to_hardware();

        let w = bus_alpha_weights(start, alpha).next().unwrap();

        assert_eq!(
            terms.weighted(w, eq_zc),
            start * terms.consistency * eq_zc + start * alpha * terms.bus_sum
        );
    }
}
