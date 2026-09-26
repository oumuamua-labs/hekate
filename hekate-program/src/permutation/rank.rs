// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Clocks of ordered service buses: the `t`-th emit of an endpoint
//! carries `ω^(base + t)`, `t` counted over its pinned selector's rows.

use alloc::collections::btree_map::Entry;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use hekate_core::errors;
use hekate_core::trace::ColumnType;
use hekate_math::{Block128, Flat, HardwareField, TowerField};

use crate::permutation::{BusKind, PermutationCheckSpec, Side, Source};
use crate::{FixedColumn, FixedShape, eq_index, residue_sums, validate_shape};

/// `v^((2^128 - 1) / p)` for the tower generator
/// `v = Block128(1 << 64)`; order exactly `p`, `p` prime.
pub const OMEGA: Block128 = Block128(0xf90a_c8aa_f4e3_ef21_c3a6_4356_fb12_7046);

/// `p`, the prime cofactor of `274177` in `2^64 + 1`;
/// labels stay injective while `base + rank < p`.
pub const OMEGA_ORDER: u64 = 67_280_421_310_721;

const OMEGA_ORDER_BITS: usize = (u64::BITS - OMEGA_ORDER.leading_zeros()) as usize;

/// One table's bus endpoints and
/// the pins their selectors read.
pub struct RankTable<'a, F> {
    pub specs: &'a [(String, PermutationCheckSpec)],
    pub fixed: &'a [FixedColumn<F>],
    pub height: TableHeight,
}

/// Who picks a table's `num_vars`: the instance fixes
/// the main table's, the prover picks a chiplet's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableHeight {
    Main(usize),
    Chiplet(Option<usize>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Schedule {
    Rows(Vec<usize>),
    Runs(Vec<Run>),
}

impl Schedule {
    fn emits(&self) -> u64 {
        match self {
            Schedule::Rows(rows) => rows.len() as u64,
            Schedule::Runs(runs) => runs.iter().map(Run::emits).sum(),
        }
    }

    fn fits(&self, height: usize) -> bool {
        match self {
            Schedule::Rows(rows) => rows.last().is_none_or(|&row| row < height),
            Schedule::Runs(runs) => runs
                .iter()
                .all(|run| run.origin + run.stride * run.count <= height),
        }
    }

    fn for_each_row(&self, mut f: impl FnMut(usize)) {
        match self {
            Schedule::Rows(rows) => rows.iter().for_each(|&row| f(row)),
            Schedule::Runs(runs) => {
                for run in runs {
                    for k in 0..run.count {
                        let start = run.origin + k * run.stride;
                        for &j in &run.offsets {
                            f(start + j);
                        }
                    }
                }
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Run {
    stride: usize,
    count: usize,
    origin: usize,
    offsets: Vec<usize>,
}

impl Run {
    fn emits(&self) -> u64 {
        self.count as u64 * self.offsets.len() as u64
    }

    fn evaluate<F>(&self, r: &[Flat<F>], base: u64, powers: &OmegaPowers<F>) -> Flat<F>
    where
        F: TowerField + HardwareField,
    {
        let zero = Flat::from_raw(F::ZERO);
        let calls = self.offsets.len();

        if calls == 0 || self.count == 0 {
            return zero;
        }

        let stride = self.stride;
        let call_step = powers.pow(calls as u64);

        let mut steps = Vec::with_capacity(r.len());
        let mut step = call_step;

        for _ in 0..r.len() {
            steps.push(step);

            step *= step;
        }

        let end = self.origin + stride * self.count;

        let upper = residue_sums(r, end, stride, &steps);
        let lower = residue_sums(r, self.origin, stride, &steps);

        let (q_origin, r_origin) = (self.origin / stride, self.origin % stride);

        let order = OMEGA_ORDER as u128;

        let mut acc = zero;
        for (i, &j) in self.offsets.iter().enumerate() {
            let wrap = u128::from(j + r_origin >= stride);
            let residue = (j + r_origin) % stride;

            let shift = calls as u128 * (q_origin as u128 + wrap) % order;
            let exponent = (base as u128 + i as u128 + order - shift) % order;

            acc += (upper[residue] + lower[residue]) * powers.pow(exponent as u64);
        }

        acc
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Selector {
    Fixed(Schedule),
    Tiled { period: usize, offsets: Vec<usize> },
}

impl Selector {
    fn parse<F: TowerField>(shape: Option<&FixedShape<F>>) -> errors::Result<Self> {
        if let Some(shape) = shape {
            validate_shape(shape, ColumnType::B128, None)?;
        }

        match shape {
            None => Ok(Selector::Tiled {
                period: 1,
                offsets: vec![0],
            }),
            Some(FixedShape::Periodic { period, values }) => Ok(Selector::Tiled {
                period: *period,
                offsets: indicator_offsets(values)?,
            }),
            Some(FixedShape::Cadence {
                stride,
                count,
                origin,
                values,
            }) => Ok(Selector::Fixed(Schedule::Runs(vec![Run {
                stride: *stride,
                count: *count,
                origin: *origin,
                offsets: indicator_offsets(values)?,
            }]))),
            Some(FixedShape::Segments(segments)) => {
                let mut runs = Vec::with_capacity(segments.len());
                for seg in segments {
                    runs.push(Run {
                        stride: seg.stride,
                        count: seg.count,
                        origin: seg.origin,
                        offsets: indicator_offsets(&seg.values)?,
                    });
                }

                Ok(Selector::Fixed(Schedule::Runs(runs)))
            }
            Some(FixedShape::Sparse(entries)) => {
                let mut rows = Vec::with_capacity(entries.len());
                for &(row, value) in entries {
                    if indicator(value)? {
                        rows.push(row);
                    }
                }

                rows.sort_unstable();

                Ok(Selector::Fixed(Schedule::Rows(rows)))
            }
            Some(
                FixedShape::Dense(_)
                | FixedShape::LastRow
                | FixedShape::FirstRow
                | FixedShape::Custom(_),
            ) => Err(errors::Error::Protocol {
                protocol: "logup_bus",
                message: "ordered bus selector must be None or pinned to Cadence, \
                          Segments, Periodic or Sparse",
            }),
        }
    }

    fn emits(&self, num_vars: Option<usize>) -> errors::Result<Option<u64>> {
        match (self, num_vars) {
            (_, Some(num_vars)) => self.resolve(num_vars).map(|s| Some(s.emits())),
            (Selector::Fixed(schedule), None) => Ok(Some(schedule.emits())),
            (Selector::Tiled { .. }, None) => Ok(None),
        }
    }

    fn resolve(&self, num_vars: usize) -> errors::Result<Schedule> {
        let height = table_height(num_vars)?;

        match self {
            Selector::Fixed(schedule) if schedule.fits(height) => Ok(schedule.clone()),
            Selector::Tiled { period, offsets } if *period <= height => {
                Ok(Schedule::Runs(vec![Run {
                    stride: *period,
                    count: height / *period,
                    origin: 0,
                    offsets: offsets.clone(),
                }]))
            }
            Selector::Fixed(_) | Selector::Tiled { .. } => Err(errors::Error::Protocol {
                protocol: "logup_bus",
                message: "ordered bus selector reaches past the table height",
            }),
        }
    }
}

struct Ordered<'a> {
    table: usize,
    spec: usize,
    bus_id: &'a str,
    side: Side,
    selector: Selector,
    num_vars: Option<usize>,
}

struct OmegaPowers<F: HardwareField> {
    squares: [Flat<F>; OMEGA_ORDER_BITS],
}

impl<F: TowerField + HardwareField> OmegaPowers<F> {
    fn new() -> errors::Result<Self> {
        if F::BITS != 128 {
            return Err(errors::Error::Protocol {
                protocol: "emit_rank",
                message: "ordered buses need GF(2^128); omega has order p nowhere below",
            });
        }

        let mut squares = [Flat::from_raw(F::ZERO); OMEGA_ORDER_BITS];
        let mut power = F::from(OMEGA.0).to_hardware();

        for slot in squares.iter_mut() {
            *slot = power;
            power *= power;
        }

        Ok(Self { squares })
    }

    fn omega(&self) -> Flat<F> {
        self.squares[0]
    }

    fn pow(&self, exponent: u64) -> Flat<F> {
        let exponent = exponent % OMEGA_ORDER;

        let mut acc = Flat::from_raw(F::ONE);
        for (bit, &square) in self.squares.iter().enumerate() {
            if (exponent >> bit) & 1 == 1 {
                acc *= square;
            }
        }

        acc
    }
}

/// Public clock column of one ordered endpoint:
/// `sel(x) · ω^(base + rank(x))`, zero off the selector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankClock {
    base: u64,
    num_vars: usize,
    schedule: Schedule,
}

impl RankClock {
    /// The clock of spec `spec_idx` among one table's
    /// `rank_clocks` entries; errors when it has none.
    pub fn for_spec(clocks: &[Option<Self>], spec_idx: usize) -> errors::Result<&Self> {
        clocks
            .get(spec_idx)
            .and_then(Option::as_ref)
            .ok_or(errors::Error::Protocol {
                protocol: "emit_rank",
                message: "ordered bus spec has no rank clock",
            })
    }

    /// MLE of the clock column at `r`.
    /// Errors unless `r.len() == num_vars`.
    pub fn evaluate<F>(&self, r: &[Flat<F>]) -> errors::Result<Flat<F>>
    where
        F: TowerField + HardwareField,
    {
        if r.len() != self.num_vars {
            return Err(errors::Error::Protocol {
                protocol: "emit_rank",
                message: "evaluation point length differs from the table's num_vars",
            });
        }

        let powers = OmegaPowers::<F>::new()?;

        let mut acc = Flat::from_raw(F::ZERO);

        match &self.schedule {
            Schedule::Rows(rows) => {
                let omega = powers.omega();

                let mut label = powers.pow(self.base);
                for &row in rows {
                    acc += eq_index(r, row) * label;
                    label *= omega;
                }
            }
            Schedule::Runs(runs) => {
                let mut base = self.base;
                for run in runs {
                    acc += run.evaluate(r, base, &powers);
                    base += run.emits();
                }
            }
        }

        Ok(acc)
    }

    /// Adds `coeff · ω^(base + t)` to the row of the
    /// `t`-th emit. Errors unless `out.len() == 2^num_vars`.
    pub fn accumulate<F>(&self, out: &mut [Flat<F>], coeff: Flat<F>) -> errors::Result<()>
    where
        F: TowerField + HardwareField,
    {
        if out.len() != 1usize << self.num_vars {
            return Err(errors::Error::Protocol {
                protocol: "emit_rank",
                message: "clock buffer length differs from the table height",
            });
        }

        let powers = OmegaPowers::<F>::new()?;
        let omega = powers.omega();

        let mut term = coeff * powers.pow(self.base);

        self.schedule.for_each_row(|row| {
            out[row] += term;
            term *= omega;
        });

        Ok(())
    }
}

/// The verify-entry predicate over a whole program, main table
/// first, then chiplets in `chiplet_defs` order; the clock of
/// every ordered endpoint, indexed by table and spec.
pub fn rank_clocks<F: TowerField>(
    tables: &[RankTable<'_, F>],
) -> errors::Result<Vec<Vec<Option<RankClock>>>> {
    let mut totals: BTreeMap<(&str, Side), u64> = BTreeMap::new();
    let mut clocks: Vec<Vec<Option<RankClock>>> =
        tables.iter().map(|t| vec![None; t.specs.len()]).collect();

    for ep in ordered_endpoints(tables)? {
        let num_vars = ep.num_vars.ok_or(errors::Error::Protocol {
            protocol: "emit_rank",
            message: "rank clocks need the height of every table",
        })?;

        let schedule = ep.selector.resolve(num_vars)?;

        let total = totals.entry((ep.bus_id, ep.side)).or_insert(0);
        let base = *total;

        *total = side_total(base, schedule.emits())?;

        clocks[ep.table][ep.spec] = Some(RankClock {
            base,
            num_vars,
            schedule,
        });
    }

    Ok(clocks)
}

/// The height-free half of the predicate;
/// sound on any subset of a program's tables.
pub fn validate_ordered_buses<F: TowerField>(tables: &[RankTable<'_, F>]) -> errors::Result<()> {
    emit_totals(tables).map(|_| ())
}

/// `validate_ordered_buses`, plus equal request and response
/// totals on every bus that shows both sides with known counts.
pub fn validate_ordered_emits<F: TowerField>(tables: &[RankTable<'_, F>]) -> errors::Result<()> {
    let totals = emit_totals(tables)?;

    for (&(bus_id, side), &total) in &totals {
        let responses = totals.get(&(bus_id, Side::Response));

        if let (Side::Request, Some(requests), Some(&Some(responses))) = (side, total, responses)
            && requests != responses
        {
            return Err(errors::Error::Protocol {
                protocol: "logup_bus",
                message: "ordered bus sides emit different counts; the bus cannot balance",
            });
        }
    }

    Ok(())
}

fn ordered_endpoints<'a, F: TowerField>(
    tables: &[RankTable<'a, F>],
) -> errors::Result<Vec<Ordered<'a>>> {
    let ordered = ordered_bus_ids(tables);

    if !ordered.is_empty() && F::BITS != 128 {
        return Err(errors::Error::Protocol {
            protocol: "logup_bus",
            message: "ordered buses need GF(2^128); omega has order p nowhere below",
        });
    }

    let mut layouts: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    let mut tiled_chiplet_sides: BTreeSet<(&str, Side)> = BTreeSet::new();

    let mut out = Vec::new();

    for (table, t) in tables.iter().enumerate() {
        let num_vars = match t.height {
            TableHeight::Main(num_vars) => Some(num_vars),
            TableHeight::Chiplet(num_vars) => num_vars,
        };

        for (spec_idx, (bus_id, spec)) in t.specs.iter().enumerate() {
            let bus_id = bus_id.as_str();

            if !ordered.contains(bus_id) {
                continue;
            }

            let (side, layout) = rank_slot(spec)?;

            match layouts.entry(bus_id) {
                Entry::Vacant(slot) => {
                    slot.insert(layout);
                }
                Entry::Occupied(slot) if *slot.get() != layout => {
                    return Err(errors::Error::Protocol {
                        protocol: "logup_bus",
                        message: "ordered bus endpoints disagree on the EmitRank slot \
                                  position or on the key width",
                    });
                }
                Entry::Occupied(_) => {}
            }

            let selector = endpoint_selector(spec, t.fixed)?;

            if tiled_chiplet_sides.contains(&(bus_id, side)) {
                return Err(errors::Error::Protocol {
                    protocol: "logup_bus",
                    message: "ordered bus chiplet endpoint with a None or Periodic selector \
                              must be last on its side; the prover picks the chiplet's height",
                });
            }

            if matches!(selector, Selector::Tiled { .. })
                && matches!(t.height, TableHeight::Chiplet(_))
            {
                tiled_chiplet_sides.insert((bus_id, side));
            }

            out.push(Ordered {
                table,
                spec: spec_idx,
                bus_id,
                side,
                selector,
                num_vars,
            });
        }
    }

    Ok(out)
}

fn ordered_bus_ids<'a, F>(tables: &[RankTable<'a, F>]) -> BTreeSet<&'a str> {
    tables
        .iter()
        .flat_map(|t| t.specs.iter())
        .filter(|(_, spec)| {
            spec.sources
                .iter()
                .any(|(source, _)| matches!(source, Source::EmitRank(_)))
        })
        .map(|(bus_id, _)| bus_id.as_str())
        .collect()
}

fn rank_slot(spec: &PermutationCheckSpec) -> errors::Result<(Side, (usize, usize))> {
    let mut rank = None;
    let mut width = 0usize;

    for (source, _) in &spec.sources {
        if let Source::EmitRank(side) = source
            && rank.replace((*side, width)).is_some()
        {
            return Err(errors::Error::Protocol {
                protocol: "logup_bus",
                message: "ordered bus endpoint carries more than one EmitRank source",
            });
        }

        width += slot_width(source);
    }

    match rank {
        Some((side, position)) => Ok((side, (position, width))),
        None => Err(errors::Error::Protocol {
            protocol: "logup_bus",
            message: "ordered bus endpoint has no EmitRank source; a witness column \
                      in the clock slot mimics any rank",
        }),
    }
}

fn slot_width(source: &Source) -> usize {
    match source {
        Source::Columns(cols) => cols.len(),
        Source::Column(_)
        | Source::RowIndexLeBytes(_)
        | Source::Const(_)
        | Source::RowIndexByte(_)
        | Source::PhaseColumn(_)
        | Source::EmitRank(_) => 1,
    }
}

fn endpoint_selector<F: TowerField>(
    spec: &PermutationCheckSpec,
    fixed: &[FixedColumn<F>],
) -> errors::Result<Selector> {
    if spec.kind != BusKind::Permutation {
        return Err(errors::Error::Protocol {
            protocol: "logup_bus",
            message: "ordered bus endpoint is Lookup kind; \
                      ranks order Permutation buses only",
        });
    }

    if spec.recv_selector.is_some() {
        return Err(errors::Error::Protocol {
            protocol: "logup_bus",
            message: "ordered bus endpoint has a recv_selector; \
                      a paired row has no single rank",
        });
    }

    let shape = match spec.selector {
        None => None,
        Some(col) => {
            let pin = fixed
                .iter()
                .find(|fc| fc.col_idx == col)
                .ok_or(errors::Error::Protocol {
                    protocol: "logup_bus",
                    message: "ordered bus selector is a witness column; \
                              the verifier counts ranks over the pinned shape",
                })?;

            Some(&pin.shape)
        }
    };

    Selector::parse(shape)
}

fn emit_totals<'a, F: TowerField>(
    tables: &[RankTable<'a, F>],
) -> errors::Result<BTreeMap<(&'a str, Side), Option<u64>>> {
    let mut totals: BTreeMap<(&str, Side), Option<u64>> = BTreeMap::new();
    for ep in ordered_endpoints(tables)? {
        let emits = ep.selector.emits(ep.num_vars)?;
        let total = totals.entry((ep.bus_id, ep.side)).or_insert(Some(0));

        *total = match (*total, emits) {
            (Some(sum), Some(emits)) => Some(side_total(sum, emits)?),
            _ => None,
        };
    }

    Ok(totals)
}

fn side_total(sum: u64, emits: u64) -> errors::Result<u64> {
    sum.checked_add(emits)
        .filter(|&total| total < OMEGA_ORDER)
        .ok_or(errors::Error::Protocol {
            protocol: "logup_bus",
            message: "an ordered bus side emits p = 67280421310721 times or more; \
                      rank labels would repeat",
        })
}

fn indicator_offsets<F: TowerField>(values: &[F]) -> errors::Result<Vec<usize>> {
    let mut offsets = Vec::new();
    for (offset, &value) in values.iter().enumerate() {
        if indicator(value)? {
            offsets.push(offset);
        }
    }

    Ok(offsets)
}

fn indicator<F: TowerField>(value: F) -> errors::Result<bool> {
    match value {
        v if v == F::ONE => Ok(true),
        v if v == F::ZERO => Ok(false),
        _ => Err(errors::Error::Protocol {
            protocol: "logup_bus",
            message: "ordered bus selector takes a value outside {0, 1}; \
                      a rank counts rows where it is one",
        }),
    }
}

fn table_height(num_vars: usize) -> errors::Result<usize> {
    u32::try_from(num_vars)
        .ok()
        .and_then(|n| 1usize.checked_shl(n))
        .ok_or(errors::Error::Protocol {
            protocol: "emit_rank",
            message: "table num_vars exceeds the address width",
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CadenceSegment;
    use crate::permutation::{ChallengeLabel, EMIT_RANK_LABEL};

    use alloc::string::ToString;
    use hekate_math::Block64;

    type F = Block128;

    const BUS: &str = "svc";

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;

            self.0
        }

        fn below(&mut self, bound: usize) -> usize {
            (self.next() % bound as u64) as usize
        }

        fn bits(&mut self, len: usize) -> Vec<F> {
            (0..len)
                .map(|_| match self.next() & 1 {
                    1 => F::ONE,
                    _ => F::ZERO,
                })
                .collect()
        }

        fn point(&mut self, len: usize) -> Vec<Flat<F>> {
            (0..len)
                .map(|_| F::from(((self.next() as u128) << 64) | self.next() as u128).to_hardware())
                .collect()
        }
    }

    fn assert_matches_brute_force(shape: Option<&FixedShape<F>>, num_vars: usize, rng: &mut Rng) {
        let base = rng.next() % (1 << 40);

        let clock = RankClock {
            base,
            num_vars,
            schedule: Selector::parse(shape).unwrap().resolve(num_vars).unwrap(),
        };

        let expected = brute_column(shape, num_vars, base);

        let coeff = rng.point(1)[0];
        let mut column = rng.point(1 << num_vars);

        let accumulated: Vec<Flat<F>> = column
            .iter()
            .zip(&expected)
            .map(|(&prior, &label)| prior + coeff * label)
            .collect();

        clock.accumulate(&mut column, coeff).unwrap();

        assert_eq!(column, accumulated);

        for _ in 0..3 {
            let r = rng.point(num_vars);

            let mut mle = Flat::from_raw(F::ZERO);
            for (row, &value) in expected.iter().enumerate() {
                mle += eq_index(&r, row) * value;
            }

            assert_eq!(clock.evaluate(&r).unwrap(), mle);
        }
    }

    fn brute_column(shape: Option<&FixedShape<F>>, num_vars: usize, base: u64) -> Vec<Flat<F>> {
        let one = Flat::from_raw(F::ONE);
        let omega = OMEGA.to_hardware();

        let mut label = pow(omega, base as u128);

        (0..1usize << num_vars)
            .map(|row| {
                let live = match shape {
                    None => true,
                    Some(shape) => shape.value_at_row(row, num_vars) == one,
                };

                match live {
                    true => {
                        let value = label;
                        label *= omega;

                        value
                    }
                    false => Flat::from_raw(F::ZERO),
                }
            })
            .collect()
    }

    fn pow(base: Flat<F>, mut exponent: u128) -> Flat<F> {
        let mut square = base;
        let mut acc = Flat::from_raw(F::ONE);

        while exponent > 0 {
            if exponent & 1 == 1 {
                acc *= square;
            }

            square *= square;
            exponent >>= 1;
        }

        acc
    }

    fn assert_rejected_against_host(
        server: &[(String, PermutationCheckSpec)],
        pins: &[FixedColumn<F>],
    ) {
        let host = vec![endpoint(ordered(Side::Request), Some(1))];
        let host_pins = calls(2);

        assert_rejected(&[table(&host, &host_pins), table(server, pins)]);
    }

    fn assert_rejected(tables: &[RankTable<'_, F>]) {
        let unsized_: Vec<RankTable<'_, F>> = tables
            .iter()
            .map(|t| unsized_table(t.specs, t.fixed))
            .collect();

        assert!(rank_clocks(tables).is_err());
        assert!(validate_ordered_buses(&unsized_).is_err());
    }

    fn main_table<'a>(
        specs: &'a [(String, PermutationCheckSpec)],
        fixed: &'a [FixedColumn<F>],
    ) -> RankTable<'a, F> {
        RankTable {
            specs,
            fixed,
            height: TableHeight::Main(4),
        }
    }

    fn table<'a>(
        specs: &'a [(String, PermutationCheckSpec)],
        fixed: &'a [FixedColumn<F>],
    ) -> RankTable<'a, F> {
        RankTable {
            specs,
            fixed,
            height: TableHeight::Chiplet(Some(4)),
        }
    }

    fn unsized_table<'a>(
        specs: &'a [(String, PermutationCheckSpec)],
        fixed: &'a [FixedColumn<F>],
    ) -> RankTable<'a, F> {
        RankTable {
            specs,
            fixed,
            height: TableHeight::Chiplet(None),
        }
    }

    fn endpoint(
        sources: Vec<(Source, ChallengeLabel)>,
        selector: Option<usize>,
    ) -> (String, PermutationCheckSpec) {
        (
            BUS.to_string(),
            PermutationCheckSpec::new(sources, selector),
        )
    }

    fn ordered(side: Side) -> Vec<(Source, ChallengeLabel)> {
        vec![
            (Source::Column(0), b"k_v"),
            (Source::EmitRank(side), EMIT_RANK_LABEL),
        ]
    }

    fn calls(count: usize) -> Vec<FixedColumn<F>> {
        vec![FixedColumn {
            col_idx: 1,
            shape: FixedShape::Cadence {
                stride: 4,
                count,
                origin: 0,
                values: vec![F::ONE, F::ZERO, F::ZERO, F::ONE],
            },
        }]
    }

    #[test]
    fn omega_has_prime_order() {
        let p = OMEGA_ORDER;

        assert_eq!(((1u128 << 64) | 1) % p as u128, 0);
        assert!(
            (3u64..)
                .step_by(2)
                .take_while(|d| d * d <= p)
                .all(|d| !p.is_multiple_of(d))
        );

        let omega = OMEGA.to_hardware();
        let one = Flat::from_raw(F::ONE);
        let generator = Block128(1 << 64).to_hardware();

        assert_ne!(omega, one);
        assert_eq!(pow(omega, p as u128), one);
        assert_eq!(pow(generator, u128::MAX / p as u128), omega);
    }

    #[test]
    fn every_row_clock_matches_brute_force() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for num_vars in 0..=10 {
            assert_matches_brute_force(None, num_vars, &mut rng);
        }
    }

    #[test]
    fn periodic_clock_matches_brute_force() {
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        for num_vars in 0..=10 {
            for log_period in 0..=num_vars {
                let shape = FixedShape::Periodic {
                    period: 1 << log_period,
                    values: rng.bits(1 << log_period),
                };

                assert_matches_brute_force(Some(&shape), num_vars, &mut rng);
            }
        }
    }

    #[test]
    fn cadence_clock_matches_brute_force() {
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);

        for num_vars in [6, 8, 10] {
            let height = 1usize << num_vars;

            for stride in 1..=64 {
                let count = rng.below(height / stride + 1);
                let origin = rng.below(height - stride * count + 1);

                let shape = FixedShape::Cadence {
                    stride,
                    count,
                    origin,
                    values: rng.bits(stride),
                };

                assert_matches_brute_force(Some(&shape), num_vars, &mut rng);
            }
        }
    }

    #[test]
    fn segments_clock_matches_brute_force() {
        let mut rng = Rng(0x6A09_E667_F3BC_C909);
        let num_vars = 10;
        let height = 1usize << num_vars;

        for _ in 0..64 {
            let mut segments = Vec::new();
            let mut cursor = rng.below(64);

            for _ in 0..1 + rng.below(4) {
                let stride = 1 + rng.below(16);
                let count = 1 + rng.below(8);

                if cursor + stride * count > height {
                    break;
                }

                segments.push(CadenceSegment {
                    stride,
                    count,
                    origin: cursor,
                    values: rng.bits(stride),
                });

                cursor += stride * count + rng.below(32);
            }

            if segments.is_empty() {
                continue;
            }

            let shape = FixedShape::Segments(segments);

            assert_matches_brute_force(Some(&shape), num_vars, &mut rng);
        }
    }

    #[test]
    fn sparse_clock_matches_brute_force() {
        let mut rng = Rng(0xBB67_AE85_84CA_A73B);
        let num_vars = 10;

        for _ in 0..32 {
            let mut entries = Vec::new();
            for row in 0..1usize << num_vars {
                if rng.next().is_multiple_of(8) {
                    entries.push((row, rng.bits(1)[0]));
                }
            }

            entries.reverse();

            let shape = FixedShape::Sparse(entries);

            assert_matches_brute_force(Some(&shape), num_vars, &mut rng);
        }
    }

    #[test]
    fn requester_tables_tile_one_responder() {
        let host = vec![endpoint(ordered(Side::Request), Some(1))];
        let server = vec![endpoint(ordered(Side::Response), Some(1))];

        let (pins_a, pins_b, pins_server) = (calls(2), calls(1), calls(3));

        let tables = [
            table(&host, &pins_a),
            table(&host, &pins_b),
            table(&server, &pins_server),
        ];

        let clocks = rank_clocks(&tables).unwrap();
        let base = |t: usize| clocks[t][0].as_ref().unwrap().base;

        assert_eq!((base(0), base(1), base(2)), (0, 4, 0));

        validate_ordered_emits(&tables).unwrap();

        let short = calls(2);
        let unbalanced = [
            table(&host, &pins_a),
            table(&host, &pins_b),
            table(&server, &short),
        ];

        rank_clocks(&unbalanced).unwrap();

        assert!(validate_ordered_emits(&unbalanced).is_err());
    }

    #[test]
    fn tiled_chiplet_endpoint_must_be_last_on_its_side() {
        let every_row = vec![endpoint(ordered(Side::Request), None)];
        let host = vec![endpoint(ordered(Side::Request), Some(1))];
        let server = vec![endpoint(ordered(Side::Response), Some(1))];

        let periodic = vec![FixedColumn {
            col_idx: 1,
            shape: FixedShape::Periodic {
                period: 2,
                values: vec![F::ONE, F::ZERO],
            },
        }];

        let (pins, server_pins) = (calls(1), calls(3));

        for tiled in [table(&every_row, &[]), table(&host, &periodic)] {
            assert_rejected(&[tiled, table(&host, &pins), table(&server, &server_pins)]);
        }

        let last = [
            table(&host, &pins),
            table(&host, &periodic),
            table(&server, &server_pins),
        ];

        rank_clocks(&last).unwrap();
        validate_ordered_buses(&last).unwrap();

        let on_main = [
            main_table(&host, &periodic),
            table(&host, &pins),
            table(&server, &server_pins),
        ];

        rank_clocks(&on_main).unwrap();
    }

    #[test]
    fn lone_side_leaves_balance_to_its_partner() {
        let server = vec![endpoint(ordered(Side::Response), Some(1))];
        let pins = calls(3);

        validate_ordered_emits(&[table(&server, &pins)]).unwrap();
    }

    #[test]
    fn honest_pair_is_accepted_both_ways() {
        let host = vec![endpoint(ordered(Side::Request), Some(1))];
        let server = vec![endpoint(ordered(Side::Response), Some(1))];
        let pins = calls(2);

        rank_clocks(&[table(&host, &pins), table(&server, &pins)]).unwrap();
        validate_ordered_buses(&[unsized_table(&host, &pins), unsized_table(&server, &pins)])
            .unwrap();
    }

    #[test]
    fn witness_column_in_the_clock_slot_is_rejected() {
        let server = vec![endpoint(
            vec![
                (Source::Column(0), b"k_v"),
                (Source::Column(2), EMIT_RANK_LABEL),
            ],
            Some(1),
        )];

        assert_rejected_against_host(&server, &calls(2));
    }

    #[test]
    fn misaligned_rank_slot_is_rejected() {
        let server = vec![endpoint(
            vec![
                (Source::EmitRank(Side::Response), EMIT_RANK_LABEL),
                (Source::Column(0), b"k_v"),
            ],
            Some(1),
        )];

        assert_rejected_against_host(&server, &calls(2));
    }

    #[test]
    fn differing_key_width_is_rejected() {
        let mut sources = ordered(Side::Response);
        sources.push((Source::Const(7), b"k_dom"));

        assert_rejected_against_host(&[endpoint(sources, Some(1))], &calls(2));
    }

    #[test]
    fn rank_slot_is_measured_in_beta_positions() {
        let server = vec![endpoint(
            vec![
                (Source::Columns(vec![0, 2]), b"k_v"),
                (Source::EmitRank(Side::Response), EMIT_RANK_LABEL),
            ],
            Some(1),
        )];

        assert_rejected_against_host(&server, &calls(2));
    }

    #[test]
    fn second_rank_source_is_rejected() {
        let mut sources = ordered(Side::Response);
        sources.push((Source::EmitRank(Side::Response), EMIT_RANK_LABEL));

        assert_rejected(&[table(&[endpoint(sources, Some(1))], &calls(2))]);
    }

    #[test]
    fn dense_selector_is_rejected() {
        let pins = vec![FixedColumn {
            col_idx: 1,
            shape: FixedShape::Dense(vec![F::ONE; 16]),
        }];

        assert_rejected_against_host(&[endpoint(ordered(Side::Response), Some(1))], &pins);
    }

    #[test]
    fn witness_selector_is_rejected() {
        assert_rejected_against_host(&[endpoint(ordered(Side::Response), Some(1))], &[]);
    }

    #[test]
    fn recv_selector_is_rejected() {
        let paired =
            PermutationCheckSpec::new_paired(ordered(Side::Response), 1, 2, BusKind::Permutation);

        let mut pins = calls(1);
        pins.push(FixedColumn {
            col_idx: 2,
            shape: FixedShape::Sparse(vec![(9, F::ONE)]),
        });

        assert_rejected_against_host(&[(BUS.to_string(), paired)], &pins);
    }

    #[test]
    fn lookup_kind_is_rejected() {
        let lookup = PermutationCheckSpec::new_lookup(ordered(Side::Response), Some(1));

        assert_rejected_against_host(&[(BUS.to_string(), lookup)], &calls(2));
    }

    #[test]
    fn non_indicator_selector_is_rejected() {
        let pins = vec![FixedColumn {
            col_idx: 1,
            shape: FixedShape::Cadence {
                stride: 4,
                count: 2,
                origin: 0,
                values: vec![F::from(2u128), F::ZERO, F::ZERO, F::ONE],
            },
        }];

        assert_rejected_against_host(&[endpoint(ordered(Side::Response), Some(1))], &pins);
    }

    #[test]
    fn side_total_stays_below_p() {
        let host = vec![endpoint(ordered(Side::Request), None)];

        let tall = RankTable::<F> {
            specs: &host,
            fixed: &[],
            height: TableHeight::Main(46),
        };

        assert!(rank_clocks(&[tall]).is_err());

        let host = vec![endpoint(ordered(Side::Request), Some(1))];
        let pins = vec![FixedColumn {
            col_idx: 1,
            shape: FixedShape::Cadence {
                stride: 1,
                count: OMEGA_ORDER as usize,
                origin: 0,
                values: vec![F::ONE],
            },
        }];

        assert!(validate_ordered_buses(&[unsized_table(&host, &pins)]).is_err());
    }

    #[test]
    fn subfields_carry_no_rank_clock() {
        let host = vec![endpoint(ordered(Side::Request), None)];

        let narrow = RankTable::<Block64> {
            specs: &host,
            fixed: &[],
            height: TableHeight::Main(4),
        };

        assert!(rank_clocks(&[narrow]).is_err());
    }
}
