// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! F₂-linear maps on `F` as linearized polynomials
//! `Σ_j μ_j x^{2^j}`, and the outer-statement gadget that
//! proves `Σ_c a_c φ(h_c)` on committed pad entries with
//! a Frobenius-Horner chain of `F::BITS − 1` squarings.

use crate::predicate::{AffineRow, Unknown, WireRole};
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;
use hekate_core::errors;
use hekate_math::{Flat, HardwareField, TowerField};

/// Trace-dual of the GF(2^8) tower basis.
const DUAL_KERNEL: [u8; 8] = [0x29, 0xb0, 0x58, 0x05, 0xa6, 0x53, 0xa4, 0x52];

/// Bytes of the widest tower field.
pub(crate) const MAX_BYTES: usize = 32;

/// F₂-linear map tabulated per byte of its argument's
/// raw bits; `new` reads `image` on basis vectors only.
/// `apply` indexes the tables by its argument: variable-time.
pub(crate) struct LinearMap<I, O> {
    tables: Vec<[Flat<O>; 256]>,
    input: PhantomData<I>,
}

impl<I, O> LinearMap<I, O>
where
    I: HardwareField,
    O: HardwareField,
{
    pub(crate) fn new(image: impl Fn(Flat<I>) -> errors::Result<Flat<O>>) -> errors::Result<Self> {
        const { assert!(I::BITS / 8 <= MAX_BYTES) };

        let mut tables = vec![[Flat::from_raw(O::ZERO); 256]; I::BITS / 8];
        for (w, table) in tables.iter_mut().enumerate() {
            let mut images = [Flat::from_raw(O::ZERO); 8];
            for (k, slot) in images.iter_mut().enumerate() {
                *slot = image(Flat::from_raw(unit_vector::<I>(8 * w + k)?))?;
            }

            for v in 1usize..256 {
                table[v] = table[v & (v - 1)] + images[v.trailing_zeros() as usize];
            }
        }

        Ok(Self {
            tables,
            input: PhantomData,
        })
    }

    pub(crate) fn apply(&self, x: Flat<I>) -> errors::Result<Flat<O>> {
        let mut acc = Flat::from_raw(O::ZERO);
        let mut raw = [0u8; MAX_BYTES];

        let raw = &mut raw[..self.tables.len()];

        x.into_raw().serialize(raw).map_err(|()| width_error())?;

        for (table, &byte) in self.tables.iter().zip(raw.iter()) {
            acc += table[byte as usize];
        }

        Ok(acc)
    }
}

struct TieMap<F> {
    lambda: Vec<Flat<F>>,
    table: Option<LinearMap<F, F>>,
}

impl<F: HardwareField> TieMap<F> {
    fn apply(&self, a: Flat<F>) -> errors::Result<Flat<F>> {
        match &self.table {
            Some(table) => table.apply(a),
            None => Ok(chain(&self.lambda, a)),
        }
    }
}

/// `Δ = Σ_c a_c φ_k(h_c)` over pad entries `(pad, a_c, k)`, `φ_k`
/// of `mu[k]`: `X_j = T'_j + X_{j+1}²` on wire `first_wire + j - 1`,
/// `T'_j = Σ_c σ^{-j}(μ_{k,j} a_c) h_c`, `X_BITS = 0`, `Δ = X_0`.
pub struct RingGadget<F> {
    pub ring: Vec<(u32, Flat<F>, usize)>,
    pub first_wire: u32,

    mu0: Vec<Flat<F>>,

    /// `σ^i(μ_{k,BITS−i})` at `[k][i]`, `i ≥ 1`:
    /// the coefficient every chain row reads.
    conj: Vec<Vec<Flat<F>>>,
}

impl<F: HardwareField> RingGadget<F> {
    pub fn new(ring: Vec<(u32, Flat<F>, usize)>, mu: Vec<Vec<Flat<F>>>, first_wire: u32) -> Self {
        let zero = Flat::from_raw(F::ZERO);

        let conj = mu
            .iter()
            .map(|mu| {
                let mut conj = vec![zero; F::BITS];
                for (i, c) in conj.iter_mut().enumerate().skip(1) {
                    if mu[F::BITS - i] != zero {
                        *c = frobenius(mu[F::BITS - i], i);
                    }
                }

                conj
            })
            .collect();

        Self {
            ring,
            first_wire,
            mu0: mu.iter().map(|mu| mu[0]).collect(),
            conj,
        }
    }

    fn wire(&self, j: usize) -> u32 {
        self.first_wire + j as u32 - 1
    }

    /// Row of `Lhs_j` among the gadget's
    /// [`chain_rows`]; `Rhs_j` follows it.
    pub fn row_of(&self, j: usize) -> usize {
        2 * (j - 1)
    }

    /// `f(pad, w_c)` per pad entry, with
    /// `w_c = Σ_j scales[row_of(j)] · tie_j[c]`
    /// folded into one F₂-linear map of `a_c`.
    pub fn for_each_tie_weight(
        &self,
        scales: &[Flat<F>],
        mut f: impl FnMut(u32, Flat<F>),
    ) -> errors::Result<()> {
        let maps = self
            .conj
            .iter()
            .enumerate()
            .map(|(k, conj)| {
                let mut lambda = vec![Flat::from_raw(F::ZERO); F::BITS];
                for (i, l) in lambda.iter_mut().enumerate().skip(1) {
                    *l = scales[self.row_of(F::BITS - i)] * conj[i];
                }

                // A table pays the chain on all BITS basis vectors
                let entries = self.ring.iter().filter(|&&(_, _, f)| f == k).count();
                let table = match entries >= F::BITS {
                    true => Some(LinearMap::new(|x| Ok(chain(&lambda, x)))?),
                    false => None,
                };

                Ok(TieMap { lambda, table })
            })
            .collect::<errors::Result<Vec<TieMap<F>>>>()?;

        for &(pad, a, k) in &self.ring {
            f(pad, maps[k].apply(a)?);
        }

        Ok(())
    }

    /// Wire terms of the chain rows, in [`Self::row_of`] order:
    /// `Lhs_j + Product_{j+1}` and `Rhs_j + Lhs_j`.
    pub fn wire_rows(&self) -> Vec<AffineRow<F>> {
        let one = Flat::from_raw(F::ONE);
        let zero = Flat::from_raw(F::ZERO);

        let mut rows = Vec::with_capacity(chain_rows::<F>());
        for j in 1..F::BITS {
            let mut unknowns = vec![(
                Unknown::Wire {
                    mul: self.wire(j),
                    role: WireRole::Lhs,
                },
                one,
            )];

            if j + 1 < F::BITS {
                unknowns.push((
                    Unknown::Wire {
                        mul: self.wire(j + 1),
                        role: WireRole::Product,
                    },
                    one,
                ));
            }

            rows.push(AffineRow {
                unknowns,
                claims: Vec::new(),
                constant: zero,
            });

            rows.push(AffineRow {
                unknowns: vec![
                    (
                        Unknown::Wire {
                            mul: self.wire(j),
                            role: WireRole::Rhs,
                        },
                        one,
                    ),
                    (
                        Unknown::Wire {
                            mul: self.wire(j),
                            role: WireRole::Lhs,
                        },
                        one,
                    ),
                ],
                claims: Vec::new(),
                constant: zero,
            });
        }

        rows
    }

    /// `Δ` over the pad entries and the wire `Product_1`.
    pub fn delta_form(&self) -> Vec<(Unknown, Flat<F>)> {
        let mut form: Vec<(Unknown, Flat<F>)> = self
            .ring
            .iter()
            .map(|&(pad, a, k)| (Unknown::Pad(pad), a * self.mu0[k]))
            .collect();

        form.push((
            Unknown::Wire {
                mul: self.wire(1),
                role: WireRole::Product,
            },
            Flat::from_raw(F::ONE),
        ));

        form
    }

    /// Honest `[X_j, X_j, X_j²]` for `j = 1..BITS`
    /// from the pad values, indexed by wire.
    pub fn wires(&self, pad_values: &[Flat<F>]) -> errors::Result<Vec<[Flat<F>; 3]>> {
        let mut wires = vec![[Flat::from_raw(F::ZERO); 3]; F::BITS - 1];
        let mut x = Flat::from_raw(F::ZERO);

        for (j, tie) in self.tie_dots(pad_values)?.into_iter().enumerate().rev() {
            x = tie + x * x;
            wires[j] = [x, x, x * x];
        }

        Ok(wires)
    }

    /// `Δ = Σ_c a_c φ(h_c)` on plaintext pad values.
    pub fn delta(&self, pad_values: &[Flat<F>]) -> errors::Result<Flat<F>> {
        let x = self
            .tie_dots(pad_values)?
            .into_iter()
            .rev()
            .fold(Flat::from_raw(F::ZERO), |x, tie| tie + x * x);

        let mut t0 = Flat::from_raw(F::ZERO);
        for (&h, &(_, a, k)) in pad_values.iter().zip(&self.ring) {
            t0 += a * self.mu0[k] * h;
        }

        Ok(t0 + x * x)
    }

    fn tie_dots(&self, pad_values: &[Flat<F>]) -> errors::Result<Vec<Flat<F>>> {
        let zero = Flat::from_raw(F::ZERO);
        let bytes = F::BITS / 8;
        let fields = self.conj.len();

        let mut buckets = vec![[zero; 256]; fields * bytes];
        let mut present = vec![false; fields];
        let mut raw = [0u8; MAX_BYTES];

        for (&(_, a, k), &h) in self.ring.iter().zip(pad_values) {
            a.into_raw().serialize(&mut raw[..bytes]).map_err(|()| {
                errors::Error::InvariantViolation {
                    message: "ring coefficient does not serialize",
                }
            })?;

            for (bucket, &byte) in buckets[k * bytes..(k + 1) * bytes].iter_mut().zip(&raw) {
                bucket[byte as usize] += h;
            }

            present[k] = true;
        }

        let mut sums = vec![zero; fields * F::BITS];
        for (b, bucket) in buckets.iter().enumerate() {
            let first = (b / bytes) * F::BITS + 8 * (b % bytes);
            for (v, &value) in bucket.iter().enumerate().skip(1) {
                for bit in (0..8).filter(|bit| (v >> bit) & 1 == 1) {
                    sums[first + bit] += value;
                }
            }
        }

        let mut images = (0..F::BITS)
            .map(|u| unit_vector::<F>(u).map(Flat::from_raw))
            .collect::<errors::Result<Vec<Flat<F>>>>()?;

        let mut ties = vec![zero; F::BITS - 1];
        for m in 1..F::BITS {
            for image in images.iter_mut() {
                *image = *image * *image;
            }

            let fields = self
                .conj
                .iter()
                .zip(sums.chunks_exact(F::BITS))
                .zip(&present);

            for ((conj, sums), _) in fields.filter(|&((conj, _), &live)| live && conj[m] != zero) {
                ties[F::BITS - m - 1] += conj[m] * dot(&images, sums);
            }
        }

        Ok(ties)
    }
}

pub fn chain_rows<F: TowerField>() -> usize {
    2 * (F::BITS - 1)
}

/// Trace-dual of the tower basis:
/// `Tr(dual_basis(u) · e_w) = [u == w]`.
pub fn dual_basis<F: TowerField>(u: usize) -> errors::Result<F> {
    let byte = DUAL_KERNEL[u & 7];
    let select = u >> 3;

    let mut bytes = [0u8; MAX_BYTES];
    for (p, slot) in bytes.iter_mut().enumerate().take(F::BITS / 8) {
        if p & select == 0 {
            *slot = byte;
        }
    }

    F::deserialize(&bytes[..F::BITS / 8]).map_err(|()| errors::Error::InvariantViolation {
        message: "dual basis element does not deserialize",
    })
}

/// `μ_{k,j}` with `Σ_u m_{k,u}·bit_u(x) = Σ_j μ_{k,j} x^{2^j}` per
/// `(m_k, w_k)`, bits in the tower basis; a map linear over the field
/// of `2^{w_k}` elements has `μ_{k,j} = 0` unless `w_k` divides `j`.
pub fn linearized_coeffs<F: TowerField + HardwareField>(
    maps: &[(Vec<F>, usize)],
) -> errors::Result<Vec<Vec<Flat<F>>>> {
    let zero = Flat::from_raw(F::ZERO);

    let coeffs: Vec<Vec<Flat<F>>> = maps
        .iter()
        .map(|(m, _)| m.iter().map(|x| x.to_hardware()).collect())
        .collect();

    let mut gamma = Vec::with_capacity(F::BITS);
    for u in 0..F::BITS {
        gamma.push(dual_basis::<F>(u)?.to_hardware());
    }

    let mut mu = vec![vec![zero; F::BITS]; maps.len()];
    for j in 0..F::BITS {
        for ((mu, coeffs), &(_, width)) in mu.iter_mut().zip(&coeffs).zip(maps) {
            if j % width == 0 {
                mu[j] = dot(&gamma, coeffs);
            }
        }

        for g in gamma.iter_mut() {
            *g = *g * *g;
        }
    }

    Ok(mu)
}

pub(crate) fn width_error() -> errors::Error {
    errors::Error::Protocol {
        protocol: "linearized",
        message: "field element does not fit its byte width",
    }
}

fn unit_vector<F: TowerField>(u: usize) -> errors::Result<F> {
    const { assert!(F::BITS / 8 <= MAX_BYTES) };

    let mut bytes = [0u8; MAX_BYTES];
    bytes[u / 8] = 1 << (u % 8);

    F::deserialize(&bytes[..F::BITS / 8]).map_err(|()| width_error())
}

fn chain<F: HardwareField>(lambda: &[Flat<F>], x: Flat<F>) -> Flat<F> {
    let zero = Flat::from_raw(F::ZERO);

    let mut power = x;
    let mut acc = zero;

    for &l in &lambda[1..] {
        power = power * power;

        if l != zero {
            acc += l * power;
        }
    }

    acc
}

fn dot<F: HardwareField>(a: &[Flat<F>], b: &[Flat<F>]) -> Flat<F> {
    let mut acc = Flat::from_raw(F::ZERO);
    for (&x, &y) in a.iter().zip(b) {
        acc += x * y;
    }

    acc
}

fn frobenius<F: HardwareField>(x: Flat<F>, k: usize) -> Flat<F> {
    let mut acc = x;
    for _ in 0..k {
        acc = acc * acc;
    }

    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expander::field_coeffs;
    use hekate_core::trace::ColumnType;
    use hekate_math::{BinaryFieldExtras, Block128, Block256};

    const FIELDS: [(ColumnType, usize); 5] = [
        (ColumnType::Bit, 1),
        (ColumnType::B8, 8),
        (ColumnType::B16, 16),
        (ColumnType::B32, 32),
        (ColumnType::B64, 64),
    ];

    trait Draw: TowerField + HardwareField + BinaryFieldExtras {
        fn next(state: &mut u128) -> Self;
    }

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

    fn next(state: &mut u128) -> Block128 {
        Block128::next(state)
    }

    fn r_mix<F: Draw>(state: &mut u128) -> Vec<F> {
        (0..F::BITS.ilog2()).map(|_| F::next(state)).collect()
    }

    fn bit<F: TowerField>(x: F, u: usize) -> bool {
        let mut bytes = [0u8; MAX_BYTES];
        x.serialize(&mut bytes[..F::BITS / 8]).unwrap();

        (bytes[u / 8] >> (u % 8)) & 1 == 1
    }

    fn maps<F: TowerField>(r_mix: &[F]) -> Vec<(Vec<F>, usize)> {
        FIELDS
            .iter()
            .map(|&(field, width)| (field_coeffs(field, r_mix), width))
            .collect()
    }

    fn phi<F: TowerField>(field: ColumnType, x: F, r_mix: &[F]) -> F {
        let mut acc = F::ZERO;
        for (u, &m) in field_coeffs(field, r_mix).iter().enumerate() {
            if bit(x, u) {
                acc += m;
            }
        }

        acc
    }

    /// `f(j, tie_j)` for `j = BITS − 1` down to `1`,
    /// `tie_j[c] = σ^{-j}(μ_j a_c)`; the coefficients
    /// of row `Lhs_j` on the pad entries.
    fn tie_rows<F: HardwareField>(gadget: &RingGadget<F>, mut f: impl FnMut(usize, &[Flat<F>])) {
        let mut powers: Vec<Flat<F>> = gadget.ring.iter().map(|&(_, a, _)| a).collect();
        let mut tie = vec![Flat::from_raw(F::ZERO); powers.len()];

        for j in (1..F::BITS).rev() {
            for a in powers.iter_mut() {
                *a = *a * *a;
            }

            for ((t, &a), &(_, _, k)) in tie.iter_mut().zip(&powers).zip(&gadget.ring) {
                *t = a * gadget.conj[k][F::BITS - j];
            }

            f(j, &tie);
        }
    }

    /// Every chain row on the given values: wire terms
    /// from `wire_rows()`, tie terms streamed onto `row_of(j)`.
    fn chain_rows_at<F: HardwareField>(
        gadget: &RingGadget<F>,
        pad: &[Flat<F>],
        wires: &[[Flat<F>; 3]],
    ) -> Vec<Flat<F>> {
        let value = |u: Unknown| -> Flat<F> {
            match u {
                Unknown::Pad(i) => pad[i as usize],
                Unknown::Wire { mul, role } => wires[mul as usize][role as usize],
            }
        };

        let mut acc: Vec<Flat<F>> = gadget
            .wire_rows()
            .iter()
            .map(|row| {
                let mut acc = row.constant;
                for &(u, coeff) in &row.unknowns {
                    acc += coeff * value(u);
                }

                acc
            })
            .collect();

        tie_rows(gadget, |j, tie| {
            for (&(pad_idx, _, _), &t) in gadget.ring.iter().zip(tie) {
                acc[gadget.row_of(j)] += t * value(Unknown::Pad(pad_idx));
            }
        });

        acc
    }

    fn trace_dual<F: TowerField + BinaryFieldExtras>() {
        for u in 0..F::BITS {
            for w in 0..F::BITS {
                let tr = (dual_basis::<F>(u).unwrap() * unit_vector::<F>(w).unwrap())
                    .trace()
                    .get();

                assert_eq!(tr, (u == w) as u8, "{} bits: u={u} w={w}", F::BITS);
            }
        }
    }

    fn field_maps<F: Draw>(seed: u128) {
        let mut state = seed;

        let r_mix = r_mix::<F>(&mut state);

        for ((field, _), mu) in FIELDS
            .into_iter()
            .zip(linearized_coeffs(&maps(&r_mix)).unwrap())
        {
            for _ in 0..8 {
                let x = F::next(&mut state);

                let mut lin = Flat::from_raw(F::ZERO);
                let mut power = x.to_hardware();

                for m in &mu {
                    lin += *m * power;
                    power = power * power;
                }

                assert_eq!(lin.to_tower(), phi(field, x, &r_mix), "{field:?}");
            }
        }
    }

    fn delta_of_wires<F: Draw>(seed: u128) {
        let mut state = seed;

        let r_mix = r_mix::<F>(&mut state);
        let mu = linearized_coeffs(&maps(&r_mix)).unwrap();

        let ring: Vec<(u32, Flat<F>, usize)> = (0..7u32)
            .map(|c| {
                (
                    c,
                    F::next(&mut state).to_hardware(),
                    c as usize % FIELDS.len(),
                )
            })
            .collect();

        let pad: Vec<Flat<F>> = (0..7).map(|_| F::next(&mut state).to_hardware()).collect();

        let mut direct = Flat::from_raw(F::ZERO);
        for (&(_, a, k), &h) in ring.iter().zip(&pad) {
            direct += a * phi(FIELDS[k].0, h.to_tower(), &r_mix).to_hardware();
        }

        let gadget = RingGadget::new(ring, mu, 0);
        assert_eq!(gadget.delta(&pad).unwrap(), direct);

        let wires = gadget.wires(&pad).unwrap();

        for (i, v) in chain_rows_at(&gadget, &pad, &wires).iter().enumerate() {
            assert_eq!(*v, Flat::from_raw(F::ZERO), "row {i}");
        }

        for wire in &wires {
            assert_eq!(wire[0], wire[1]);
            assert_eq!(wire[0] * wire[1], wire[2]);
        }

        let mut delta = Flat::from_raw(F::ZERO);
        for &(u, coeff) in &gadget.delta_form() {
            delta += coeff
                * match u {
                    Unknown::Pad(i) => pad[i as usize],
                    Unknown::Wire { mul, role } => wires[mul as usize][role as usize],
                };
        }

        assert_eq!(delta, direct);
    }

    fn tie_dots_of_rows<F: Draw>(seed: u128) {
        let mut state = seed;
        for len in [1u32, 9, 40, 700] {
            let r_mix = r_mix::<F>(&mut state);
            let mu = linearized_coeffs(&maps(&r_mix)).unwrap();

            let ring: Vec<(u32, Flat<F>, usize)> = (0..len)
                .map(|c| {
                    (
                        c,
                        F::next(&mut state).to_hardware(),
                        c as usize % FIELDS.len(),
                    )
                })
                .collect();

            let pad: Vec<Flat<F>> = (0..len)
                .map(|_| F::next(&mut state).to_hardware())
                .collect();

            let gadget = RingGadget::new(ring, mu, 0);

            let mut expected = vec![Flat::from_raw(F::ZERO); F::BITS - 1];
            tie_rows(&gadget, |j, tie| expected[j - 1] = dot(&pad, tie));

            assert_eq!(
                gadget.tie_dots(&pad).unwrap(),
                expected,
                "{} bits, {len} entries",
                F::BITS
            );
        }
    }

    #[test]
    fn dual_basis_is_trace_dual() {
        trace_dual::<Block128>();
        trace_dual::<Block256>();
    }

    #[test]
    fn coefficients_reproduce_field_maps() {
        field_maps::<Block128>(0x0f0f_1234_5678_9abc_def0_1122_3344_5566);
        field_maps::<Block256>(0x0f0f_1234_5678_9abc_def0_1122_3344_5577);
    }

    #[test]
    fn wires_and_rows_evaluate_to_delta() {
        delta_of_wires::<Block128>(0x7777_8888_9999_aaaa_bbbb_cccc_dddd_eeee);
        delta_of_wires::<Block256>(0x7777_8888_9999_aaaa_bbbb_cccc_dddd_eeef);
    }

    /// A floating wire would let a prover pick its value freely;
    /// every wire must be pinned by a row or a Hadamard triple.
    #[test]
    fn chain_is_rigid_under_single_wire_perturbation() {
        let mut state = 0xdead_beef_0bad_f00d_1357_9bdf_2468_ace0u128;

        let r_mix: Vec<Block128> = r_mix(&mut state);
        let mu = linearized_coeffs(&[(field_coeffs(ColumnType::Bit, &r_mix), 1)]).unwrap();

        let ring: Vec<(u32, Flat<Block128>, usize)> = (0..7u32)
            .map(|c| (c, next(&mut state).to_hardware(), 0))
            .collect();

        let pad: Vec<Flat<Block128>> = (0..7).map(|_| next(&mut state).to_hardware()).collect();

        let gadget = RingGadget::new(ring, mu, 0);
        let honest = gadget.wires(&pad).unwrap();
        let delta = next(&mut state).to_hardware();

        let zero = Flat::from_raw(Block128::ZERO);

        for target in 0..honest.len() {
            for role in 0..3 {
                let mut wires = honest.clone();
                wires[target][role] += delta;

                let row_broken = chain_rows_at(&gadget, &pad, &wires)
                    .iter()
                    .any(|acc| *acc != zero);
                let hadamard_broken = wires.iter().any(|w| w[0] * w[1] != w[2]);

                assert!(
                    row_broken || hadamard_broken,
                    "wire {target} role {role} perturbation went undetected"
                );
            }
        }
    }

    #[test]
    fn tie_weights_equal_scaled_tie_rows() {
        let mut state = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210u128;
        for len in [1u32, 9, 40, 700] {
            let r_mix: Vec<Block128> = r_mix(&mut state);
            let mu = linearized_coeffs(&maps(&r_mix)).unwrap();

            let ring: Vec<(u32, Flat<Block128>, usize)> = (0..len)
                .map(|c| {
                    let k = c as usize % FIELDS.len();

                    (3 * c + 1, next(&mut state).to_hardware(), k)
                })
                .collect();

            let gadget = RingGadget::new(ring, mu, 0);

            let scales: Vec<Flat<Block128>> = (0..chain_rows::<Block128>())
                .map(|_| next(&mut state).to_hardware())
                .collect();

            let mut expected: Vec<(u32, Flat<Block128>)> = gadget
                .ring
                .iter()
                .map(|&(pad, _, _)| (pad, Flat::from_raw(Block128::ZERO)))
                .collect();

            tie_rows(&gadget, |j, tie| {
                for ((_, w), &t) in expected.iter_mut().zip(tie) {
                    *w += scales[gadget.row_of(j)] * t;
                }
            });

            let mut folded = Vec::new();
            gadget
                .for_each_tie_weight(&scales, |pad, w| folded.push((pad, w)))
                .unwrap();

            assert_eq!(folded, expected);
        }
    }

    #[test]
    fn tie_dots_match_tie_rows() {
        tie_dots_of_rows::<Block128>(0x2468_ace0_1357_9bdf_0f1e_2d3c_4b5a_6978);
        tie_dots_of_rows::<Block256>(0x2468_ace0_1357_9bdf_0f1e_2d3c_4b5a_6979);
    }
}
