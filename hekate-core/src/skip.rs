// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec;
use alloc::vec::Vec;
use hekate_math::{
    BinaryFieldExtras, Block8, Block16, CantorBasis, Flat, FlatPromote, HardwareField,
};

use crate::errors;
use crate::tensor::TensorProduct;

pub trait SkipLaneField: BinaryFieldExtras + HardwareField {
    const LANES: SkipLanes;
}

impl SkipLaneField for Block8 {
    const LANES: SkipLanes = SkipLanes::Block8;
}

impl SkipLaneField for Block16 {
    const LANES: SkipLanes = SkipLanes::Block16;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SkipLanes {
    Block8,
    Block16,
}

impl SkipLanes {
    pub fn bits(self) -> usize {
        match self {
            Self::Block8 => 8,
            Self::Block16 => 16,
        }
    }
}

/// A table's univariate skip: its `vars` low variables
/// bound at once over `cosets` cosets of `W_vars` in
/// the lane field. `vars = 0` skips nothing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SkipPlan {
    pub vars: usize,
    pub lanes: SkipLanes,
    pub cosets: usize,
}

impl SkipPlan {
    pub const NONE: Self = Self {
        vars: 0,
        lanes: SkipLanes::Block8,
        cosets: 0,
    };

    pub fn is_skip(&self) -> bool {
        self.vars > 0
    }

    pub fn block_rows(&self) -> usize {
        1 << self.vars
    }

    pub fn domain_points(&self) -> usize {
        self.cosets << self.vars
    }

    pub fn extra_points(&self) -> usize {
        self.cosets.saturating_sub(1) << self.vars
    }

    /// Counts the scalars the round sends: every domain point but the first.
    pub fn sent_scalars(&self) -> usize {
        self.domain_points().saturating_sub(1)
    }

    pub fn basis_dim(&self) -> usize {
        self.vars + ceil_log2(self.cosets)
    }

    pub fn check(&self) -> errors::Result<()> {
        let fits = self.cosets >= 1
            && self.basis_dim() <= self.lanes.bits()
            && self.domain_points() <= 1 << self.lanes.bits();

        match self.is_skip() && !fits {
            true => Err(errors::Error::Protocol {
                protocol: "skip",
                message: "skip domain does not fit its lane field",
            }),
            false => Ok(()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SkipDomain<G: HardwareField> {
    plan: SkipPlan,
    points: Vec<Flat<G>>,
    offsets: Vec<Flat<G>>,
    subspace_weight: Flat<G>,
    coset_weights: Vec<Flat<G>>,
}

impl<G: HardwareField> SkipDomain<G> {
    pub fn plan(&self) -> SkipPlan {
        self.plan
    }

    /// Returns every domain point, coset-major.
    pub fn points(&self) -> &[Flat<G>] {
        &self.points
    }

    /// Returns `H`, the first coset.
    pub fn subspace(&self) -> &[Flat<G>] {
        &self.points[..self.plan.block_rows()]
    }

    /// Returns the Lagrange basis of the domain at `r`, one weight per point.
    pub fn lagrange(&self, r: Flat<G>) -> Vec<Flat<G>> {
        let one = Flat::from_raw(G::ONE);
        let block = self.plan.block_rows();

        let vanishing = self
            .offsets
            .iter()
            .fold(one, |acc, &o| acc * subspace_poly(self.subspace(), r + o));

        self.basis_at(&self.points, r, |i| {
            vanishing * self.coset_weights[i / block]
        })
    }

    /// Returns the Lagrange basis of `H` alone at `r`: the weights
    /// a column's interpolant over its block takes at the challenge.
    pub fn lagrange_block(&self, r: Flat<G>) -> Vec<Flat<G>> {
        let scale = subspace_poly(self.subspace(), r) * self.subspace_weight;

        self.basis_at(self.subspace(), r, |_| scale)
    }

    /// Evaluates at `r` the polynomial taking `values` on the domain.
    pub fn interpolate(&self, values: &[Flat<G>], r: Flat<G>) -> errors::Result<Flat<G>> {
        if values.len() != self.points.len() {
            return Err(errors::Error::Protocol {
                protocol: "skip",
                message: "skip values do not cover the domain",
            });
        }

        let zero = Flat::from_raw(G::ZERO);

        Ok(values
            .iter()
            .zip(self.lagrange(r))
            .fold(zero, |acc, (&v, l)| acc + v * l))
    }

    fn from_betas(plan: SkipPlan, betas: &[Flat<G>]) -> errors::Result<Self> {
        plan.check()?;

        if !plan.is_skip() || betas.len() < plan.basis_dim() {
            return Err(errors::Error::InvariantViolation {
                message: "skip domain needs a skip and a basis spanning it",
            });
        }

        let one = Flat::from_raw(G::ONE);
        let (low, high) = betas[..plan.basis_dim()].split_at(plan.vars);

        let subspace: Vec<Flat<G>> = (0..plan.block_rows()).map(|u| span(low, u)).collect();
        let offsets: Vec<Flat<G>> = (0..plan.cosets).map(|i| span(high, i)).collect();

        let points: Vec<Flat<G>> = offsets
            .iter()
            .flat_map(|&o| subspace.iter().map(move |&h| o + h))
            .collect();

        let subspace_weight = invert(subspace[1..].iter().fold(one, |acc, &h| acc * h));

        let coset_weights = (0..plan.cosets)
            .map(|i| {
                let mut denominator = one;
                for j in (0..plan.cosets).filter(|&j| j != i) {
                    denominator *= subspace_poly(&subspace, offsets[i] + offsets[j]);
                }

                subspace_weight * invert(denominator)
            })
            .collect();

        Ok(Self {
            plan,
            points,
            offsets,
            subspace_weight,
            coset_weights,
        })
    }

    /// Evaluates `weight(i) / (r + points[i])` at every point,
    /// or the indicator vector when `r` is one of them.
    fn basis_at(
        &self,
        points: &[Flat<G>],
        r: Flat<G>,
        weight: impl Fn(usize) -> Flat<G>,
    ) -> Vec<Flat<G>> {
        let zero = Flat::from_raw(G::ZERO);
        let one = Flat::from_raw(G::ONE);

        let diffs: Vec<Flat<G>> = points.iter().map(|&x| r + x).collect();

        if let Some(at) = diffs.iter().position(|&t| t == zero) {
            let mut indicator = vec![zero; points.len()];
            indicator[at] = one;

            return indicator;
        }

        batch_invert(&diffs)
            .into_iter()
            .enumerate()
            .map(|(i, inv)| weight(i) * inv)
            .collect()
    }
}

impl<F> SkipDomain<F>
where
    F: HardwareField + FlatPromote<Block8> + FlatPromote<Block16>,
{
    /// Builds the domain in the challenge field from
    /// the lane field's Cantor basis, promoted.
    pub fn new(plan: SkipPlan) -> errors::Result<Self> {
        let betas: Vec<Flat<F>> = match plan.lanes {
            SkipLanes::Block8 => cantor_betas::<Block8>(plan)?
                .iter()
                .map(|&b| F::promote_flat(b))
                .collect(),
            SkipLanes::Block16 => cantor_betas::<Block16>(plan)?
                .iter()
                .map(|&b| F::promote_flat(b))
                .collect(),
        };

        Self::from_betas(plan, &betas)
    }
}

impl<S: SkipLaneField> SkipDomain<S> {
    /// Builds the domain in its own lane field;
    /// errors when `plan` names another.
    pub fn in_lanes(plan: SkipPlan) -> errors::Result<Self> {
        if plan.lanes != S::LANES {
            return Err(errors::Error::Protocol {
                protocol: "skip",
                message: "skip domain requested in a field other than its lanes",
            });
        }

        Self::from_betas(plan, &cantor_betas::<S>(plan)?)
    }
}

/// The row weights split in two: `w(x) = low[x_lo] · high[x_hi]`.
#[derive(Clone, Debug)]
pub struct WeightHalves<F: HardwareField> {
    pub low: Vec<Flat<F>>,
    pub high: Vec<Flat<F>>,
}

/// Where a skipped sumcheck ends: row `x_lo + 2^k · x_hi` weighs
/// `low[x_lo] · eq(tail, x_hi)`, `low` the Lagrange weights over
/// `H` at the skip challenge; `low = [1]` is a multilinear point.
#[derive(Clone, Debug)]
pub struct SkipPoint<F: HardwareField> {
    low: Vec<Flat<F>>,
    tail: Vec<Flat<F>>,
    total: Flat<F>,
}

impl<F: HardwareField> SkipPoint<F> {
    /// Wraps a multilinear point: no variable skipped.
    pub fn multilinear(r: Vec<Flat<F>>) -> Self {
        Self {
            low: vec![Flat::from_raw(F::ONE)],
            tail: r,
            total: Flat::from_raw(F::ONE),
        }
    }

    /// Builds the point from the block weights and the tail;
    /// errors unless the weights span a power-of-two block.
    pub fn new(low: Vec<Flat<F>>, tail: Vec<Flat<F>>) -> errors::Result<Self> {
        if !low.len().is_power_of_two() {
            return Err(errors::Error::Protocol {
                protocol: "skip",
                message: "skip point weights do not cover a block",
            });
        }

        let total = low.iter().fold(Flat::from_raw(F::ZERO), |acc, &w| acc + w);

        Ok(Self { low, tail, total })
    }

    pub fn low(&self) -> &[Flat<F>] {
        &self.low
    }

    pub fn tail(&self) -> &[Flat<F>] {
        &self.tail
    }

    pub fn skip_vars(&self) -> usize {
        self.low.len().trailing_zeros() as usize
    }

    pub fn num_vars(&self) -> usize {
        self.skip_vars() + self.tail.len()
    }

    /// Returns the weight of `row`; its bits past `num_vars` are ignored.
    pub fn weight(&self, row: usize) -> Flat<F> {
        let low = self.low[row & (self.low.len() - 1)];
        let high = row >> self.skip_vars();

        let one = Flat::from_raw(F::ONE);

        self.tail.iter().enumerate().fold(low, |acc, (i, &t)| {
            acc * match (high >> i) & 1 {
                1 => t,
                _ => one + t,
            }
        })
    }

    /// Weighs a multilinear `r` over `num_vars`: `Σ_x w(x) · eq(r, x)`.
    pub fn eq_with(&self, r: &[Flat<F>]) -> errors::Result<Flat<F>> {
        if r.len() != self.num_vars() {
            return Err(errors::Error::Protocol {
                protocol: "skip",
                message: "eq point and skip point differ in variables",
            });
        }

        let (r_low, r_high) = r.split_at(self.skip_vars());
        let zero = Flat::from_raw(F::ZERO);

        let low = self
            .low
            .iter()
            .zip(TensorProduct::eq_table(r_low))
            .fold(zero, |acc, (&w, e)| acc + w * e);

        Ok(low * eq_slice(r_high, &self.tail))
    }

    /// Sums the weights of every row: one for a Lagrange block.
    pub fn total(&self) -> Flat<F> {
        self.total
    }

    /// Sums the weights of the rows whose bit `bit` is set.
    pub fn bit_marginal(&self, bit: usize) -> Flat<F> {
        let zero = Flat::from_raw(F::ZERO);
        let k = self.skip_vars();

        match bit < k {
            true => self
                .low
                .iter()
                .enumerate()
                .filter(|(x, _)| (x >> bit) & 1 == 1)
                .fold(zero, |acc, (_, &w)| acc + w),
            false => self.total * self.tail[bit - k],
        }
    }

    /// Returns the weight of every row, `2^num_vars` entries in row order.
    pub fn expand(&self) -> Vec<Flat<F>> {
        let high = TensorProduct::eq_table(&self.tail);

        high.iter()
            .flat_map(|&e| self.low.iter().map(move |&w| w * e))
            .collect()
    }

    /// Splits the weights at `low_vars`:
    /// `w(x) = low[x_lo] · high[x_hi]` with `x_lo` the low `low_vars` bits.
    /// Errors when the split falls inside the skip block.
    pub fn halves(&self, low_vars: usize) -> errors::Result<WeightHalves<F>> {
        let k = self.skip_vars();

        if low_vars < k || low_vars > self.num_vars() {
            return Err(errors::Error::Protocol {
                protocol: "skip",
                message: "weight split falls inside the skip block",
            });
        }

        let (tail_low, tail_high) = self.tail.split_at(low_vars - k);
        let mid = TensorProduct::eq_table(tail_low);

        let low = mid
            .iter()
            .flat_map(|&e| self.low.iter().map(move |&w| w * e))
            .collect();

        Ok(WeightHalves {
            low,
            high: TensorProduct::eq_table(tail_high),
        })
    }

    /// Sums the weights of the rows whose low `bits` bits equal `value`.
    pub fn low_bits_marginal(&self, bits: usize, value: usize) -> Flat<F> {
        let zero = Flat::from_raw(F::ZERO);
        let k = self.skip_vars();

        match bits <= k {
            true => self
                .low
                .iter()
                .enumerate()
                .filter(|(x, _)| x & ((1 << bits) - 1) == value)
                .fold(zero, |acc, (_, &w)| acc + w),
            false => {
                let one = Flat::from_raw(F::ONE);
                let high = value >> k;

                self.tail[..bits - k].iter().enumerate().fold(
                    self.low[value & (self.low.len() - 1)],
                    |acc, (i, &t)| {
                        acc * match (high >> i) & 1 {
                            1 => t,
                            _ => one + t,
                        }
                    },
                )
            }
        }
    }
}

impl<F> SkipPoint<F>
where
    F: HardwareField + FlatPromote<Block8> + FlatPromote<Block16>,
{
    /// Builds the point a skipped sumcheck ends at from
    /// its challenge list, the skip challenge first; a plan
    /// without a skip reads the list as a multilinear point.
    pub fn from_challenges(plan: SkipPlan, challenges: &[Flat<F>]) -> errors::Result<Self> {
        if !plan.is_skip() {
            return Ok(Self::multilinear(challenges.to_vec()));
        }

        let (&skip, tail) = challenges.split_first().ok_or(errors::Error::Protocol {
            protocol: "skip",
            message: "skipped sumcheck ended without its skip challenge",
        })?;

        let low = SkipDomain::<F>::new(plan)?.lagrange_block(skip);

        Self::new(low, tail.to_vec())
    }
}

fn cantor_betas<S: BinaryFieldExtras + HardwareField>(
    plan: SkipPlan,
) -> errors::Result<Vec<Flat<S>>> {
    CantorBasis::<S>::new(plan.basis_dim())
        .map(|basis| basis.betas().to_vec())
        .map_err(|_| errors::Error::Protocol {
            protocol: "skip",
            message: "lane field has no Cantor basis of the skip's dimension",
        })
}

fn span<G: HardwareField>(betas: &[Flat<G>], index: usize) -> Flat<G> {
    betas
        .iter()
        .enumerate()
        .filter(|(b, _)| (index >> b) & 1 == 1)
        .fold(Flat::from_raw(G::ZERO), |acc, (_, &beta)| acc + beta)
}

fn invert<G: HardwareField>(x: Flat<G>) -> Flat<G> {
    x.to_tower().invert().to_hardware()
}

/// Inverts every value by one field inversion;
/// a zero among them poisons every result.
fn batch_invert<G: HardwareField>(values: &[Flat<G>]) -> Vec<Flat<G>> {
    let one = Flat::from_raw(G::ONE);

    let mut prefix = Vec::with_capacity(values.len());
    let mut acc = one;

    for &v in values {
        prefix.push(acc);
        acc *= v;
    }

    let mut acc_inv = invert(acc);
    let mut out = vec![one; values.len()];

    for i in (0..values.len()).rev() {
        out[i] = prefix[i] * acc_inv;
        acc_inv *= values[i];
    }

    out
}

fn subspace_poly<G: HardwareField>(subspace: &[Flat<G>], x: Flat<G>) -> Flat<G> {
    let one = Flat::from_raw(G::ONE);

    subspace.iter().fold(one, |acc, &h| acc * (x + h))
}

fn eq_slice<F: HardwareField>(a: &[Flat<F>], b: &[Flat<F>]) -> Flat<F> {
    let one = Flat::from_raw(F::ONE);

    a.iter()
        .zip(b)
        .fold(one, |acc, (&x, &y)| acc * (one + x + y))
}

fn ceil_log2(n: usize) -> usize {
    match n <= 1 {
        true => 0,
        false => (usize::BITS - (n - 1).leading_zeros()) as usize,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hekate_math::{Block128, Block256, TowerField};

    type F = Block128;

    const PLAN: SkipPlan = SkipPlan {
        vars: 6,
        lanes: SkipLanes::Block8,
        cosets: 3,
    };

    fn element(seed: u64) -> Flat<F> {
        let x = (seed as u128).wrapping_mul(0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C835)
            ^ (seed as u128) << 64;

        F::from(x | 1).to_hardware()
    }

    fn horner(coeffs: &[Flat<F>], x: Flat<F>) -> Flat<F> {
        coeffs
            .iter()
            .rev()
            .fold(Flat::from_raw(F::ZERO), |acc, &c| acc * x + c)
    }

    #[test]
    fn plan_counts_points_and_sent_scalars() {
        assert_eq!(PLAN.block_rows(), 64);
        assert_eq!(PLAN.domain_points(), 192);
        assert_eq!(PLAN.extra_points(), 128);
        assert_eq!(PLAN.sent_scalars(), 191);
        assert_eq!(PLAN.basis_dim(), 8);
        assert!(PLAN.check().is_ok());

        assert!(!SkipPlan::NONE.is_skip());
        assert_eq!(SkipPlan::NONE.sent_scalars(), 0);
        assert!(SkipPlan::NONE.check().is_ok());
    }

    #[test]
    fn plan_past_lane_field_rejected() {
        let wide = SkipPlan { cosets: 5, ..PLAN };

        assert!(wide.check().is_err());
        assert!(SkipDomain::<F>::new(wide).is_err());

        let fits = SkipPlan {
            lanes: SkipLanes::Block16,
            ..wide
        };

        assert!(SkipDomain::<F>::new(fits).is_ok());
    }

    #[test]
    fn lagrange_sums_to_one_and_picks_domain_points() {
        let domain = SkipDomain::<F>::new(PLAN).unwrap();
        let one = Flat::from_raw(F::ONE);

        let weights = domain.lagrange(element(7));
        let sum = weights.iter().fold(Flat::from_raw(F::ZERO), |a, &w| a + w);

        assert_eq!(sum, one);

        for (i, &x) in domain.points().iter().enumerate().step_by(37) {
            let at = domain.lagrange(x);

            for (j, &w) in at.iter().enumerate() {
                assert_eq!(w == one, i == j);
            }
        }
    }

    #[test]
    fn interpolation_recovers_polynomial_below_domain_degree() {
        let domain = SkipDomain::<F>::new(PLAN).unwrap();
        let coeffs: Vec<Flat<F>> = (0..PLAN.domain_points() as u64).map(element).collect();

        let values: Vec<Flat<F>> = domain
            .points()
            .iter()
            .map(|&x| horner(&coeffs, x))
            .collect();

        for seed in [3u64, 1001, 65_537] {
            let r = element(seed);

            assert_eq!(domain.interpolate(&values, r).unwrap(), horner(&coeffs, r));
        }
    }

    #[test]
    fn block_weights_match_direct_lagrange_over_subspace() {
        let domain = SkipDomain::<F>::new(PLAN).unwrap();
        let one = Flat::from_raw(F::ONE);
        let r = element(11);

        let subspace = domain.subspace();
        let block = domain.lagrange_block(r);

        for (i, &x) in subspace.iter().enumerate() {
            let direct = subspace
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .fold(one, |acc, (_, &y)| acc * (r + y) * invert(x + y));

            assert_eq!(block[i], direct);
        }
    }

    #[test]
    fn lane_domain_promotes_to_challenge_domain() {
        let lanes = SkipDomain::<Block8>::in_lanes(PLAN).unwrap();
        let wide = SkipDomain::<F>::new(PLAN).unwrap();

        let promoted: Vec<Flat<F>> = lanes.points().iter().map(|&p| F::promote_flat(p)).collect();

        assert_eq!(promoted, wide.points());
        assert!(SkipDomain::<Block16>::in_lanes(PLAN).is_err());
    }

    #[test]
    fn multilinear_point_weighs_as_eq() {
        let r: Vec<Flat<F>> = (0..5u64).map(element).collect();
        let point = SkipPoint::multilinear(r.clone());
        let eq = TensorProduct::new(r.clone());

        assert_eq!(point.skip_vars(), 0);
        assert_eq!(point.num_vars(), 5);

        for row in 0..32 {
            assert_eq!(point.weight(row), eq.evaluate_at_index(row));
        }

        let other: Vec<Flat<F>> = (10..15u64).map(element).collect();

        assert_eq!(
            point.eq_with(&other).unwrap(),
            TensorProduct::evaluate_eq_slice(&r, &other)
        );
    }

    #[test]
    fn skip_point_functionals_match_expanded_weights() {
        let low: Vec<Flat<F>> = (20..24u64).map(element).collect();
        let tail: Vec<Flat<F>> = (30..33u64).map(element).collect();
        let point = SkipPoint::new(low, tail).unwrap();
        let zero = Flat::from_raw(F::ZERO);

        let weights = point.expand();

        assert_eq!(weights.len(), 32);

        for (row, &w) in weights.iter().enumerate() {
            assert_eq!(point.weight(row), w);
        }

        let r: Vec<Flat<F>> = (40..45u64).map(element).collect();
        let eq = TensorProduct::new(r.clone());
        let direct = (0..32).fold(zero, |acc, row| {
            acc + weights[row] * eq.evaluate_at_index(row)
        });

        assert_eq!(point.eq_with(&r).unwrap(), direct);

        for bit in 0..5 {
            let marginal = (0..32)
                .filter(|row| (row >> bit) & 1 == 1)
                .fold(zero, |acc, row| acc + weights[row]);

            assert_eq!(point.bit_marginal(bit), marginal);
        }

        let halves = point.halves(3).unwrap();

        for (row, &w) in weights.iter().enumerate() {
            assert_eq!(halves.low[row & 7] * halves.high[row >> 3], w);
        }

        for bits in 0..=5 {
            for value in 0..1usize << bits {
                let marginal = (0..32)
                    .filter(|row| row & ((1 << bits) - 1) == value)
                    .fold(zero, |acc, row| acc + weights[row]);

                assert_eq!(point.low_bits_marginal(bits, value), marginal);
            }
        }

        assert!(point.halves(1).is_err());
        assert!(SkipPoint::<F>::new(vec![zero; 3], Vec::new()).is_err());
    }

    #[test]
    fn block16_lanes_at_seven_vars_match_challenge_domain() {
        let plan = SkipPlan {
            vars: 7,
            lanes: SkipLanes::Block16,
            cosets: 4,
        };

        let lanes = SkipDomain::<Block16>::in_lanes(plan).unwrap();
        let wide = SkipDomain::<Block256>::new(plan).unwrap();

        let promoted: Vec<Flat<Block256>> = lanes
            .points()
            .iter()
            .map(|&p| Block256::promote_flat(p))
            .collect();

        assert_eq!(promoted, wide.points());

        let domain = SkipDomain::<F>::new(plan).unwrap();
        let coeffs: Vec<Flat<F>> = (0..plan.domain_points() as u64).map(element).collect();

        let values: Vec<Flat<F>> = domain
            .points()
            .iter()
            .map(|&x| horner(&coeffs, x))
            .collect();

        let r = element(17);

        assert_eq!(domain.interpolate(&values, r).unwrap(), horner(&coeffs, r));
    }
}
