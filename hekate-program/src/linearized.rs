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
use hekate_core::trace::TraceCompatibleField;
use hekate_math::{Block128, CanonicalDeserialize, Flat, HardwareField, TowerField};

/// Trace-dual of the GF(2^8) tower basis.
const DUAL_KERNEL: [u8; 8] = [0x29, 0xb0, 0x58, 0x05, 0xa6, 0x53, 0xa4, 0x52];

const CODE_BITS: usize = Block128::BITS;

/// Ring entries per field at which its tie map switches
/// from a squaring chain per entry to a table of images.
const TABLE_ENTRIES: usize = 20;

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
        let images = (0..I::BITS)
            .map(|v| image(Flat::from_raw(unit_vector::<I>(v)?)))
            .collect::<errors::Result<Vec<Flat<O>>>>()?;

        Ok(Self::from_images(&images))
    }

    /// Tabulates the map from the image of each unit vector.
    pub(crate) fn from_images(images: &[Flat<O>]) -> Self {
        const { assert!(I::BITS / 8 <= MAX_BYTES) };

        let mut tables = vec![[Flat::from_raw(O::ZERO); 256]; I::BITS / 8];
        for (table, images) in tables.iter_mut().zip(images.as_chunks::<8>().0) {
            for v in 1usize..256 {
                table[v] = table[v & (v - 1)] + images[v.trailing_zeros() as usize];
            }
        }

        Self {
            tables,
            input: PhantomData,
        }
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

/// Raises flat elements to their `2^j`-th powers from a table
/// of GF(2^128)'s unit-vector powers, without squaring chains.
pub struct Frobenius<F> {
    orbits: Vec<[Flat<Block128>; CODE_BITS]>,
    c: [Flat<Block128>; CODE_BITS],
    y: Option<Flat<F>>,
}

impl<F: TraceCompatibleField> Frobenius<F> {
    pub fn new() -> errors::Result<Self> {
        let zero = Flat::from_raw(Block128::ZERO);

        let y = match F::BITS / CODE_BITS {
            1 => None,
            2 => Some(Flat::from_raw(unit_vector::<F>(CODE_BITS)?)),
            _ => return Err(width_error()),
        };

        let mut orbits = vec![[zero; CODE_BITS]; CODE_BITS];
        for (w, orbit) in orbits.iter_mut().enumerate() {
            let mut power = Flat::from_raw(unit_vector::<Block128>(w)?);
            for slot in orbit.iter_mut() {
                *slot = power;
                power = power * power;
            }
        }

        let mut c = [zero; CODE_BITS];
        let mut tau = Block128::EXTENSION_TAU.to_hardware();

        for j in 1..CODE_BITS {
            c[j] = c[j - 1] + tau;
            tau = tau * tau;
        }

        Ok(Self { orbits, c, y })
    }

    /// Raises `x` to its `2^j`-th power, `j < F::BITS`.
    pub fn conjugate(&self, x: Flat<F>, j: usize) -> errors::Result<Flat<F>> {
        let mut raw = [0u8; MAX_BYTES];

        x.into_raw()
            .serialize(&mut raw[..F::BITS / 8])
            .map_err(|()| width_error())?;

        let (lo, hi) = raw[..F::BITS / 8].split_at(CODE_BITS / 8);
        let lo = F::promote_flat(self.bit_sum(lo, j % CODE_BITS));

        Ok(match self.y {
            Some(y) => lo + self.y_shift(y, j) * F::promote_flat(self.bit_sum(hi, j % CODE_BITS)),
            None => lo,
        })
    }

    /// Sums weights split by [`Self::halves`] against the flat
    /// unit vectors raised to their `2^j`-th powers, `j < F::BITS`.
    fn conjugate_dot(&self, w: &[[Flat<Block128>; 2]], j: usize) -> Flat<F> {
        let (lo, hi) = w.split_at(CODE_BITS);
        let powers = || self.orbits.iter().map(move |orbit| orbit[j % CODE_BITS]);
        let lo = self.code_dot(lo, powers());

        match self.y {
            Some(y) => lo + self.y_shift(y, j) * self.code_dot(hi, powers()),
            None => lo,
        }
    }

    /// Sums `w` against the flat unit vectors raised to their
    /// `2^j`-th powers at every `stride`-th `j`, `j` and `j + 128`
    /// from shared GF(2^128) dots; `stride` divides `F::BITS`.
    pub fn conjugate_dots(&self, w: &[Flat<F>], stride: usize) -> errors::Result<Vec<Flat<F>>> {
        let w = Self::halves(w)?;
        let (lo, hi) = w.split_at(CODE_BITS);

        let mut out = vec![Flat::from_raw(F::ZERO); F::BITS / stride];
        for j in (0..CODE_BITS).step_by(stride) {
            let powers = || self.orbits.iter().map(move |orbit| orbit[j]);
            let lo_dot = self.code_dot(lo, powers());

            match self.y {
                Some(y) => {
                    let hi_dot = self.code_dot(hi, powers());

                    out[j / stride] = lo_dot + self.y_shift(y, j) * hi_dot;

                    if (j + CODE_BITS).is_multiple_of(stride) {
                        out[(j + CODE_BITS) / stride] =
                            lo_dot + self.y_shift(y, j + CODE_BITS) * hi_dot;
                    }
                }
                None => out[j / stride] = lo_dot,
            }
        }

        Ok(out)
    }

    /// Maps every flat unit vector through
    /// `Λ(x) = Σ_{j≥1} λ_j x^(2^j)`, in unit vector order.
    pub fn images(&self, lambda: &[Flat<F>]) -> errors::Result<Vec<Flat<F>>> {
        let zero = Flat::from_raw(F::ZERO);

        let mut a = vec![zero; CODE_BITS];
        for (j, &l) in lambda.iter().enumerate().skip(1) {
            a[j % CODE_BITS] += l;
        }

        let image = |coeffs: &[[Flat<Block128>; 2]], w: usize| {
            self.code_dot(coeffs, self.orbits[w].iter().copied())
        };

        let a = Self::halves(&a)?;

        let mut images: Vec<Flat<F>> = (0..CODE_BITS).map(|w| image(&a, w)).collect();

        if let Some(y) = self.y {
            let mut b = vec![zero; CODE_BITS];
            for (j, &l) in lambda.iter().enumerate().skip(1) {
                b[j % CODE_BITS] += l * self.y_shift(y, j);
            }

            let b = Self::halves(&b)?;

            images.extend((0..CODE_BITS).map(|w| image(&b, w)));
        }

        Ok(images)
    }

    fn y_shift(&self, y: Flat<F>, j: usize) -> Flat<F> {
        let c = match j < CODE_BITS {
            true => self.c[j],
            false => self.c[j - CODE_BITS] + Flat::from_raw(Block128::ONE),
        };

        y + F::promote_flat(c)
    }

    fn bit_sum(&self, bytes: &[u8], j: usize) -> Flat<Block128> {
        let mut acc = Flat::from_raw(Block128::ZERO);
        for (b, &byte) in bytes.iter().enumerate() {
            let mut bits = byte;
            while bits != 0 {
                acc += self.orbits[8 * b + bits.trailing_zeros() as usize][j];
                bits &= bits - 1;
            }
        }

        acc
    }

    /// Sums `w`, given as GF(2^128) halves, against the GF(2^128) values `k`.
    fn code_dot(
        &self,
        w: &[[Flat<Block128>; 2]],
        k: impl Iterator<Item = Flat<Block128>>,
    ) -> Flat<F> {
        let zero = Flat::from_raw(Block128::ZERO);
        let terms = w.iter().zip(k);

        match self.y {
            Some(y) => {
                let [lo, hi] =
                    terms.fold([zero; 2], |[lo, hi], (h, k)| [lo + h[0] * k, hi + h[1] * k]);

                F::promote_flat(lo) + y * F::promote_flat(hi)
            }
            None => F::promote_flat(terms.fold(zero, |acc, (h, k)| acc + h[0] * k)),
        }
    }

    /// Splits each flat element of `w` into its two GF(2^128)
    /// coordinates, the second zero when `F` is GF(2^128).
    fn halves(w: &[Flat<F>]) -> errors::Result<Vec<[Flat<Block128>; 2]>> {
        let half = CODE_BITS / 8;
        let bytes = F::BITS / 8;

        let mut raw = [0u8; MAX_BYTES];

        w.iter()
            .map(|x| {
                x.into_raw()
                    .serialize(&mut raw[..bytes])
                    .map_err(|()| width_error())?;

                let read = |at: usize| match at < bytes {
                    true => Block128::deserialize(&raw[at..at + half]).map_err(|()| width_error()),
                    false => Ok(Block128::ZERO),
                };

                Ok([Flat::from_raw(read(0)?), Flat::from_raw(read(half)?)])
            })
            .collect()
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

    frobenius: Frobenius<F>,
}

impl<F: HardwareField> RingGadget<F> {
    pub fn new(
        ring: Vec<(u32, Flat<F>, usize)>,
        mu: Vec<Vec<Flat<F>>>,
        first_wire: u32,
    ) -> errors::Result<Self>
    where
        F: TraceCompatibleField,
    {
        let zero = Flat::from_raw(F::ZERO);
        let frobenius = Frobenius::new()?;

        let mut conj = Vec::with_capacity(mu.len());
        for mu in &mu {
            let mut row = vec![zero; F::BITS];
            for (i, c) in row.iter_mut().enumerate().skip(1) {
                if mu[F::BITS - i] != zero {
                    *c = frobenius.conjugate(mu[F::BITS - i], i)?;
                }
            }

            conj.push(row);
        }

        Ok(Self {
            ring,
            first_wire,
            mu0: mu.iter().map(|mu| mu[0]).collect(),
            conj,
            frobenius,
        })
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
    ) -> errors::Result<()>
    where
        F: TraceCompatibleField,
    {
        let lambdas: Vec<Vec<Flat<F>>> = self
            .conj
            .iter()
            .map(|conj| {
                let mut lambda = vec![Flat::from_raw(F::ZERO); F::BITS];
                for (i, l) in lambda.iter_mut().enumerate().skip(1) {
                    *l = scales[self.row_of(F::BITS - i)] * conj[i];
                }

                lambda
            })
            .collect();

        let mut entries = vec![0usize; lambdas.len()];
        for &(_, _, k) in &self.ring {
            entries[k] += 1;
        }

        let maps = lambdas
            .into_iter()
            .zip(entries)
            .map(|(lambda, n)| {
                let table = match n >= TABLE_ENTRIES {
                    true => Some(LinearMap::from_images(&self.frobenius.images(&lambda)?)),
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
    pub fn wires(&self, pad_values: &[Flat<F>]) -> errors::Result<Vec<[Flat<F>; 3]>>
    where
        F: TraceCompatibleField,
    {
        let mut wires = vec![[Flat::from_raw(F::ZERO); 3]; F::BITS - 1];
        let mut x = Flat::from_raw(F::ZERO);

        for (j, tie) in self.tie_dots(pad_values)?.into_iter().enumerate().rev() {
            x = tie + x * x;
            wires[j] = [x, x, x * x];
        }

        Ok(wires)
    }

    /// `Δ = Σ_c a_c φ(h_c)` on plaintext pad values.
    pub fn delta(&self, pad_values: &[Flat<F>]) -> errors::Result<Flat<F>>
    where
        F: TraceCompatibleField,
    {
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

    fn tie_dots(&self, pad_values: &[Flat<F>]) -> errors::Result<Vec<Flat<F>>>
    where
        F: TraceCompatibleField,
    {
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

        let halves = sums
            .chunks_exact(F::BITS)
            .map(Frobenius::<F>::halves)
            .collect::<errors::Result<Vec<_>>>()?;

        let mut ties = vec![zero; F::BITS - 1];
        for m in 1..F::BITS {
            let fields = self.conj.iter().zip(&halves).zip(&present);

            for ((conj, sums), _) in fields.filter(|&((conj, _), &live)| live && conj[m] != zero) {
                ties[F::BITS - m - 1] += conj[m] * self.frobenius.conjugate_dot(sums, m);
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
pub fn linearized_coeffs<F: TraceCompatibleField>(
    maps: &[(Vec<F>, usize)],
) -> errors::Result<Vec<Vec<Flat<F>>>> {
    let zero = Flat::from_raw(F::ZERO);
    let frobenius = Frobenius::<F>::new()?;

    let gamma = (0..F::BITS)
        .map(|u| dual_basis::<F>(u).map(|g| g.to_hardware()))
        .collect::<errors::Result<Vec<Flat<F>>>>()?;

    let mut mu = Vec::with_capacity(maps.len());
    for (m, width) in maps {
        let coeffs: Vec<Flat<F>> = m.iter().map(|x| x.to_hardware()).collect();
        let sums = bit_sums(&gamma, &coeffs)?;

        let mut row = vec![zero; F::BITS];
        for (i, value) in frobenius
            .conjugate_dots(&sums, *width)?
            .into_iter()
            .enumerate()
        {
            row[i * width] = value;
        }

        mu.push(row);
    }

    Ok(mu)
}

/// Sums `weights` per flat bit of `values`: entry `v`
/// collects the weights whose value has bit `v` set, which
/// turns any F₂-linear sum into one over unit vectors.
pub fn bit_sums<F: TowerField + HardwareField>(
    values: &[Flat<F>],
    weights: &[Flat<F>],
) -> errors::Result<Vec<Flat<F>>> {
    let mut sums = vec![Flat::from_raw(F::ZERO); F::BITS];
    let mut raw = [0u8; MAX_BYTES];

    for (&value, &weight) in values.iter().zip(weights) {
        value
            .into_raw()
            .serialize(&mut raw[..F::BITS / 8])
            .map_err(|()| width_error())?;

        for (b, &byte) in raw[..F::BITS / 8].iter().enumerate() {
            let mut bits = byte;
            while bits != 0 {
                sums[8 * b + bits.trailing_zeros() as usize] += weight;
                bits &= bits - 1;
            }
        }
    }

    Ok(sums)
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

    trait Draw: TraceCompatibleField {
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

    fn dot<F: HardwareField>(a: &[Flat<F>], b: &[Flat<F>]) -> Flat<F> {
        let mut acc = Flat::from_raw(F::ZERO);
        for (&x, &y) in a.iter().zip(b) {
            acc += x * y;
        }

        acc
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

        let gadget = RingGadget::new(ring, mu, 0).unwrap();
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

            let gadget = RingGadget::new(ring, mu, 0).unwrap();

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

    fn frobenius_squarings<F: Draw>(seed: u128) {
        let mut state = seed;
        let frobenius = Frobenius::<F>::new().unwrap();

        let unit: Vec<Flat<F>> = (0..F::BITS)
            .map(|v| Flat::from_raw(unit_vector::<F>(v).unwrap()))
            .collect();

        for _ in 0..4 {
            let x = F::next(&mut state).to_hardware();
            let w: Vec<Flat<F>> = (0..F::BITS)
                .map(|_| F::next(&mut state).to_hardware())
                .collect();

            let mut power = x;
            let mut unit_powers = unit.clone();
            let mut dots = Vec::with_capacity(F::BITS);

            let halves = Frobenius::<F>::halves(&w).unwrap();

            for j in 0..F::BITS {
                dots.push(dot(&w, &unit_powers));

                assert_eq!(frobenius.conjugate(x, j).unwrap(), power, "j={j}");
                assert_eq!(frobenius.conjugate_dot(&halves, j), dots[j], "j={j}");

                power = power * power;

                for e in unit_powers.iter_mut() {
                    *e = *e * *e;
                }
            }

            for stride in [1, 8, 64, 128, F::BITS] {
                let strided: Vec<Flat<F>> = dots.iter().step_by(stride).copied().collect();

                assert_eq!(
                    frobenius.conjugate_dots(&w, stride).unwrap(),
                    strided,
                    "stride {stride}"
                );
            }

            let lambda: Vec<Flat<F>> = (0..F::BITS)
                .map(|_| F::next(&mut state).to_hardware())
                .collect();

            let images: Vec<Flat<F>> = unit.iter().map(|&e| chain(&lambda, e)).collect();

            assert_eq!(frobenius.images(&lambda).unwrap(), images);
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

        let gadget = RingGadget::new(ring, mu, 0).unwrap();
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

            let gadget = RingGadget::new(ring, mu, 0).unwrap();

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

    #[test]
    fn frobenius_matches_squarings() {
        frobenius_squarings::<Block128>(0x1f1e_1d1c_1b1a_1918_1716_1514_1312_1110);
        frobenius_squarings::<Block256>(0x1f1e_1d1c_1b1a_1918_1716_1514_1312_1111);
    }
}
