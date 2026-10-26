// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! A table's `h` columns and their keys. Every Permutation
//! column of every table cancels in one sum, and the bus tag
//! leading each Permutation key keeps the buses apart in it.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use hekate_core::errors;
use hekate_crypto::{DefaultHasher, Hasher};
use hekate_math::{Flat, HardwareField, TowerField};

use super::{
    BusElement, BusField, BusKind, BusValue, PermutationCheckSpec, RankClock, Slot,
    eval_row_idx_byte_mle, eval_row_idx_le_mle,
};

/// Fractions one Permutation column carries at most.
const MAX_FRACTIONS: usize = 2;

/// One `h` column of a table: the specs whose fractions
/// it sums, as indices into the table's spec list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BusColumn {
    pub kind: BusKind,
    pub first: usize,
    pub second: Option<usize>,
}

impl BusColumn {
    /// Returns the cancellation the column's claimed sums enter.
    pub fn group(&self, specs: &[(String, PermutationCheckSpec)]) -> BusGroup {
        match self.kind {
            BusKind::Permutation => BusGroup::Permutation,
            BusKind::Lookup => BusGroup::Lookup(specs[self.first].0.clone()),
        }
    }
}

/// The cancellation a claimed-sum limb enters:
/// every Permutation column at once, or one Lookup bus.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum BusGroup {
    Permutation,
    Lookup(String),
}

impl BusGroup {
    /// Returns the group's bus-sum row weight:
    /// one for Permutation, `eq(r_bus, (r_final, 0))` for a Lookup bus;
    /// errors when `points` holds no `r_bus` for the bus or a short one.
    pub fn row_weight<F: TowerField + HardwareField>(
        &self,
        field: &BusField<F>,
        points: &BTreeMap<String, Vec<BusElement>>,
        r_final: &[Flat<F>],
    ) -> errors::Result<BusValue<F>> {
        match self {
            Self::Permutation => Ok(BusValue::one()),
            Self::Lookup(bus_id) => {
                let r_bus = points.get(bus_id).ok_or(errors::Error::Protocol {
                    protocol: "logup",
                    message: "lookup bus spec missing r_bus point",
                })?;

                lookup_weight(field, r_bus, r_final)
            }
        }
    }
}

/// One fraction `s / D` of an `h` column at a point:
/// `D = constant + Σ coeff · claim` over the table's claims.
#[derive(Clone, Debug)]
pub struct BusFraction<F> {
    pub slots: Vec<(u32, BusValue<F>)>,
    pub constant: BusValue<F>,
    pub selector: Option<u32>,
    pub recv_selector: Option<u32>,
}

impl<F: TowerField + HardwareField> BusFraction<F> {
    /// Builds spec `spec_idx`'s fraction at `r_final`; errors on
    /// a column outside the table's `width` or a missing rank clock.
    pub fn new(
        field: &BusField<F>,
        spec_idx: usize,
        (bus_id, spec): &(String, PermutationCheckSpec),
        width: usize,
        challenges: [BusValue<F>; 2],
        r_final: &[Flat<F>],
        clocks: &[Option<RankClock>],
    ) -> errors::Result<Self> {
        let [gamma, beta] = challenges;

        let column = |idx: usize| match idx < width {
            true => Ok(idx as u32),
            false => Err(errors::Error::Protocol {
                protocol: "logup",
                message: "bus spec column outside the table",
            }),
        };

        let (schedule, tag) = key_schedule(field, (bus_id, spec), beta);

        let mut slots = Vec::new();
        let mut constant = gamma + tag;

        for (slot, coeff) in schedule {
            match slot {
                Slot::Column(c) => slots.push((column(c)?, coeff)),
                Slot::Const(v) => constant += coeff.mul_base(F::from(v).to_hardware()),
                Slot::RowIndexLeBytes(n) => {
                    constant += coeff.mul_base(eval_row_idx_le_mle::<F>(n, r_final));
                }
                Slot::RowIndexByte(n) => {
                    constant += coeff.mul_base(eval_row_idx_byte_mle::<F>(n, r_final));
                }
                Slot::EmitRank => {
                    let clock = RankClock::for_spec(clocks, spec_idx)?;
                    constant += coeff.mul_base(clock.evaluate(r_final)?);
                }
            }
        }

        Ok(Self {
            slots,
            constant,
            selector: spec.selector.map(column).transpose()?,
            recv_selector: spec.recv_selector.map(column).transpose()?,
        })
    }

    /// Evaluates `D` at the table's claims.
    pub fn denominator(&self, claims: &[Flat<F>]) -> BusValue<F> {
        self.slots.iter().fold(self.constant, |acc, &(c, coeff)| {
            acc + coeff.mul_base(claims[c as usize])
        })
    }

    /// Evaluates `s` at the table's claims.
    pub fn selector_value(&self, claims: &[Flat<F>]) -> Flat<F> {
        let send = self
            .selector
            .map_or(Flat::from_raw(F::ONE), |i| claims[i as usize]);

        let recv = self
            .recv_selector
            .map_or(Flat::from_raw(F::ZERO), |i| claims[i as usize]);

        send + recv
    }

    /// Tests whether `s` reads a claim.
    pub fn selector_claimed(&self) -> bool {
        self.selector.is_some() || self.recv_selector.is_some()
    }
}

/// Packs a table's bus specs into `h` columns:
/// Permutation specs pair in declaration order when
/// the degree has room, Lookup specs stay alone.
pub fn bus_columns(
    specs: &[(String, PermutationCheckSpec)],
    sumcheck_degree: usize,
) -> Vec<BusColumn> {
    let per_column = MAX_FRACTIONS.min(sumcheck_degree.saturating_sub(2)).max(1);

    let mut columns = Vec::with_capacity(specs.len());
    let mut open: Option<usize> = None;

    for (idx, (_, spec)) in specs.iter().enumerate() {
        match (spec.kind, open) {
            (BusKind::Lookup, _) => columns.push(BusColumn {
                kind: BusKind::Lookup,
                first: idx,
                second: None,
            }),
            (BusKind::Permutation, Some(c)) => {
                columns[c].second = Some(idx);
                open = None;
            }
            (BusKind::Permutation, None) => {
                columns.push(BusColumn {
                    kind: BusKind::Permutation,
                    first: idx,
                    second: None,
                });

                if per_column > 1 {
                    open = Some(columns.len() - 1);
                }
            }
        }
    }

    columns
}

/// Pairs every key slot with its `β` power and returns
/// the tag term, `β^k · t` one slot past the last of
/// a Permutation key's `k` slots, zero on a Lookup key.
pub fn key_schedule<F: TowerField + HardwareField>(
    field: &BusField<F>,
    (bus_id, spec): (&str, &PermutationCheckSpec),
    beta: BusValue<F>,
) -> (Vec<(Slot, BusValue<F>)>, BusValue<F>) {
    let mut beta_pow = BusValue::one();
    let mut schedule = Vec::new();

    for slot in spec.slots() {
        schedule.push((slot, beta_pow));

        beta_pow = field.mul(beta_pow, beta);
    }

    let tag = match spec.kind {
        BusKind::Permutation => field.mul(beta_pow, field.import(bus_tag(bus_id))),
        BusKind::Lookup => BusValue::zero(),
    };

    (schedule, tag)
}

/// Derives the tag a bus's Permutation keys carry in their top `β` slot.
pub fn bus_tag(bus_id: &str) -> BusElement {
    let mut h = DefaultHasher::new();
    h.update(b"hekate-bus-tag-v1");
    h.update(&(bus_id.len() as u64).to_le_bytes());
    h.update(bus_id.as_bytes());

    BusElement::from_uniform_bytes(&h.finalize())
}

/// Evaluates a Lookup bus's row weight `eq(r_bus, (r_final, 0))`;
/// errors when `r_bus` is shorter than `r_final`.
pub fn lookup_weight<F: TowerField + HardwareField>(
    field: &BusField<F>,
    r_bus: &[BusElement],
    r_final: &[Flat<F>],
) -> errors::Result<BusValue<F>> {
    if r_bus.len() < r_final.len() {
        return Err(errors::Error::Protocol {
            protocol: "logup",
            message: "r_bus shorter than table num_vars",
        });
    }

    let (lo, hi) = r_bus.split_at(r_final.len());
    let r_lo: Vec<BusValue<F>> = lo.iter().map(|&r| field.import(r)).collect();

    let mut weight = field.eq_at(&r_lo, r_final);
    for &r_j in hi {
        weight = field.mul(weight, field.import(r_j) + BusValue::one());
    }

    Ok(weight)
}

/// Counts the values that defeat the bus draw `(γ, β, every r_bus)`:
/// a spec of key width `k` on `N` rows adds `N` poles and its balance
/// identity's degree `N · max(1, k − 1)`, a Lookup bus `log₂ N_max`.
pub fn bus_draw_events<'a>(
    tables: impl IntoIterator<Item = (&'a [(String, PermutationCheckSpec)], u64)>,
) -> u128 {
    let mut events: u128 = 0;
    let mut lookup_rows: BTreeMap<&str, u64> = BTreeMap::new();

    for (specs, rows) in tables {
        for (bus_id, spec) in specs {
            events += rows as u128 * spec.key_width().max(2) as u128;

            if spec.kind == BusKind::Lookup {
                let n_max = lookup_rows.entry(bus_id.as_str()).or_insert(0);
                *n_max = (*n_max).max(rows);
            }
        }
    }

    let r_bus_steps: u128 = lookup_rows
        .values()
        .map(|&n| n.max(1).next_power_of_two().ilog2() as u128)
        .sum();

    events + r_bus_steps
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::permutation::Source;
    use alloc::string::ToString;
    use alloc::vec;
    use hekate_math::{Block128, Block256};

    fn spec(kind: BusKind) -> (String, PermutationCheckSpec) {
        let sources = vec![(Source::Column(0), b"v".as_slice())];

        let spec = match kind {
            BusKind::Permutation => PermutationCheckSpec::new(sources, None),
            BusKind::Lookup => PermutationCheckSpec::new_lookup(sources, None),
        };

        ("bus".to_string(), spec)
    }

    fn single(kind: BusKind, first: usize) -> BusColumn {
        BusColumn {
            kind,
            first,
            second: None,
        }
    }

    fn denominator(
        field: &BusField<Block128>,
        bus_id: &str,
        spec: &PermutationCheckSpec,
        value: u128,
    ) -> BusValue<Block128> {
        let gamma = field.import(Block256::new(Block128(0x9e37), Block128(0x79b9)));
        let beta = field.import(Block256::new(Block128(0x7f4a), Block128(0x7c15)));

        let (schedule, tag) = key_schedule(field, (bus_id, spec), beta);

        schedule
            .iter()
            .fold(gamma + tag, |acc, &(slot, coeff)| match slot {
                Slot::Const(c) => acc + coeff.mul_base(Block128::from(c).to_hardware()),
                _ => acc + coeff.mul_base(Block128::from(value).to_hardware()),
            })
    }

    #[test]
    fn degree_three_keeps_one_fraction_per_column() {
        let specs = vec![spec(BusKind::Permutation), spec(BusKind::Permutation)];

        assert_eq!(
            bus_columns(&specs, 3),
            vec![
                single(BusKind::Permutation, 0),
                single(BusKind::Permutation, 1),
            ]
        );
    }

    #[test]
    fn degree_four_pairs_permutation_specs_in_declaration_order() {
        let specs = vec![
            spec(BusKind::Permutation),
            spec(BusKind::Lookup),
            spec(BusKind::Permutation),
            spec(BusKind::Permutation),
        ];

        let paired = BusColumn {
            kind: BusKind::Permutation,
            first: 0,
            second: Some(2),
        };

        assert_eq!(
            bus_columns(&specs, 4),
            vec![
                paired,
                single(BusKind::Lookup, 1),
                single(BusKind::Permutation, 3),
            ]
        );
    }

    #[test]
    fn bus_tag_spends_full_digest() {
        let a = bus_tag("link");
        let b = bus_tag("link2");

        assert_ne!(a, b);
        assert_eq!(a, bus_tag("link"));
        assert_ne!(a.split().1, Block128::ZERO);
        assert_ne!(b.split().1, Block128::ZERO);
    }

    #[test]
    fn bus_draw_events_charge_tag_floor_and_r_bus_steps() {
        let wide = PermutationCheckSpec::new(
            vec![
                (Source::Columns(vec![0, 1]), b"v".as_slice()),
                (Source::Const(5), b"c".as_slice()),
            ],
            None,
        );

        let permutation = vec![("bus".to_string(), wide)];
        let lookup = vec![spec(BusKind::Lookup)];

        let events = bus_draw_events([
            (permutation.as_slice(), 8),
            (lookup.as_slice(), 16),
            (lookup.as_slice(), 4),
        ]);

        let tagged = 8 * 4;
        let floored = 16 * 2 + 4 * 2;
        let r_bus = 4;

        assert_eq!(events, tagged + floored + r_bus);
    }

    #[test]
    fn tag_keys_one_tuple_apart_on_two_buses() {
        let field = BusField::<Block128>::new().unwrap();
        let (_, spec) = spec(BusKind::Permutation);

        assert_ne!(
            denominator(&field, "bus_a", &spec, 0xCAFE),
            denominator(&field, "bus_b", &spec, 0xCAFE)
        );
    }

    #[test]
    fn slot_zero_const_cannot_cancel_tag_difference() {
        let field = BusField::<Block128>::new().unwrap();

        let lo = |bus_id: &str| bus_tag(bus_id).split().0;
        let offset = (lo("bus_a") + lo("bus_b")).0;

        let keyed = |constant: u128| {
            PermutationCheckSpec::new(
                vec![
                    (Source::Const(constant), b"c".as_slice()),
                    (Source::Column(0), b"v".as_slice()),
                ],
                None,
            )
        };

        assert_ne!(
            denominator(&field, "bus_a", &keyed(offset), 0xCAFE),
            denominator(&field, "bus_b", &keyed(0), 0xCAFE)
        );
    }
}
