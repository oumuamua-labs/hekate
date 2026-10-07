// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! F₂-linear maps on GF(2^128) as linearized polynomials
//! `Σ_j μ_j x^{2^j}`, and the outer-statement gadget that
//! proves `Σ_c a_c φ(h_c)` on committed pad entries with
//! a Frobenius-Horner chain of 127 squarings.

use crate::predicate::{AffineRow, Unknown, WireRole};
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;
use hekate_math::{Block128, Flat, HardwareField, TowerField};

pub const BITS: usize = 128;

pub const CHAIN_ROWS: usize = 2 * (BITS - 1);

const DUAL_KERNEL: [u8; 8] = [0x29, 0xb0, 0x58, 0x05, 0xa6, 0x53, 0xa4, 0x52];

/// F₂-linear map tabulated per byte of its argument's
/// raw bits; `new` reads `image` on basis vectors only.
/// `apply` indexes the tables by its argument: variable-time.
pub(crate) struct LinearMap<I, O> {
    tables: Vec<[Flat<O>; 256]>,
    input: PhantomData<I>,
}

impl<I, O> LinearMap<I, O>
where
    I: HardwareField + Into<Block128>,
    O: HardwareField,
{
    pub(crate) fn new(image: impl Fn(Flat<I>) -> Flat<O>) -> Self {
        let mut tables = vec![[Flat::from_raw(O::ZERO); 256]; BITS / 8];
        for (w, table) in tables.iter_mut().enumerate() {
            let images: [Flat<O>; 8] =
                core::array::from_fn(|k| image(Flat::from_raw(I::from(1u128 << (8 * w + k)))));

            for v in 1usize..256 {
                table[v] = table[v & (v - 1)] + images[v.trailing_zeros() as usize];
            }
        }

        Self {
            tables,
            input: PhantomData,
        }
    }

    pub(crate) fn apply(&self, x: Flat<I>) -> Flat<O> {
        let raw: Block128 = x.into_raw().into();

        let mut acc = Flat::from_raw(O::ZERO);
        for (table, byte) in self.tables.iter().zip(raw.0.to_le_bytes()) {
            acc += table[byte as usize];
        }

        acc
    }
}

/// `Δ = Σ_c a_c φ(h_c)` over the pad entries `(pad, a_c)`
/// as the chain `X_127 = T'_127`, `X_j = T'_j + X_{j+1}²`,
/// `T'_j = Σ_c σ^{-j}(μ_j a_c) h_c`, on squaring wires
/// `first_wire + j - 1` for `j = 1..128`; `Δ = X_0`.
pub struct RingGadget<F> {
    pub ring: Vec<(u32, Flat<F>)>,
    pub mu: Vec<Flat<F>>,
    pub first_wire: u32,
}

impl<F: HardwareField> RingGadget<F> {
    pub fn new(ring: Vec<(u32, Flat<F>)>, mu: Vec<Flat<F>>, first_wire: u32) -> Self {
        Self {
            ring,
            mu,
            first_wire,
        }
    }

    fn wire(&self, j: usize) -> u32 {
        self.first_wire + j as u32 - 1
    }

    /// Row of `Lhs_j` among the gadget's
    /// [`CHAIN_ROWS`]; `Rhs_j` follows it.
    pub fn row_of(&self, j: usize) -> usize {
        2 * (j - 1)
    }

    /// `f(j, tie_j)` for `j = 127` down to `1`,
    /// `tie_j[c] = σ^{-j}(μ_j a_c)`; the coefficients
    /// of row `Lhs_j` on the pad entries.
    pub fn for_each_tie_row(&self, mut f: impl FnMut(usize, &[Flat<F>])) {
        let mut conj: Vec<Flat<F>> = self.ring.iter().map(|&(_, a)| a).collect();
        let mut tie = vec![Flat::from_raw(F::ZERO); conj.len()];

        for j in (1..BITS).rev() {
            for a in conj.iter_mut() {
                *a = *a * *a;
            }

            let mu_conj = frobenius(self.mu[j], BITS - j);
            for (t, &a) in tie.iter_mut().zip(&conj) {
                *t = a * mu_conj;
            }

            f(j, &tie);
        }
    }

    /// `f(pad, w_c)` per pad entry, with
    /// `w_c = Σ_j scales[row_of(j)] · tie_j[c]`
    /// folded into one F₂-linear map of `a_c`.
    pub fn for_each_tie_weight(&self, scales: &[Flat<F>], mut f: impl FnMut(u32, Flat<F>))
    where
        F: Into<Block128>,
    {
        let mut lambda = [Flat::from_raw(F::ZERO); BITS];
        for (i, l) in lambda.iter_mut().enumerate().skip(1) {
            let j = BITS - i;
            *l = scales[self.row_of(j)] * frobenius(self.mu[j], i);
        }

        let map = LinearMap::new(|x: Flat<F>| {
            let mut power = x;
            let mut acc = Flat::from_raw(F::ZERO);

            for &l in &lambda[1..] {
                power = power * power;
                acc += l * power;
            }

            acc
        });

        for &(pad, a) in &self.ring {
            f(pad, map.apply(a));
        }
    }

    /// Wire terms of the chain rows, in [`Self::row_of`] order:
    /// `Lhs_j + Product_{j+1}` and `Rhs_j + Lhs_j`.
    pub fn wire_rows(&self) -> Vec<AffineRow<F>> {
        let one = Flat::from_raw(F::ONE);
        let zero = Flat::from_raw(F::ZERO);

        let mut rows = Vec::with_capacity(CHAIN_ROWS);
        for j in 1..BITS {
            let mut unknowns = vec![(
                Unknown::Wire {
                    mul: self.wire(j),
                    role: WireRole::Lhs,
                },
                one,
            )];

            if j + 1 < BITS {
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
            .map(|&(pad, a)| (Unknown::Pad(pad), a * self.mu[0]))
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

    /// Honest `[X_j, X_j, X_j²]` for `j = 1..128`
    /// from the pad values, indexed by wire.
    pub fn wires(&self, pad_values: &[Flat<F>]) -> Vec<[Flat<F>; 3]> {
        let mut wires = vec![[Flat::from_raw(F::ZERO); 3]; BITS - 1];
        let mut x = Flat::from_raw(F::ZERO);

        self.for_each_tie_row(|j, tie| {
            x = dot(pad_values, tie) + x * x;
            wires[j - 1] = [x, x, x * x];
        });

        wires
    }

    /// `Δ = Σ_c a_c φ(h_c)` on plaintext pad values.
    pub fn delta(&self, pad_values: &[Flat<F>]) -> Flat<F> {
        let mut x = Flat::from_raw(F::ZERO);
        self.for_each_tie_row(|_, tie| x = dot(pad_values, tie) + x * x);

        let mut t0 = Flat::from_raw(F::ZERO);
        for (&h, &(_, a)) in pad_values.iter().zip(&self.ring) {
            t0 += a * self.mu[0] * h;
        }

        t0 + x * x
    }
}

/// Trace-dual of the tower basis:
/// `Tr(dual_basis(u) · 2^w) = [u == w]`.
pub fn dual_basis(u: usize) -> Block128 {
    let byte = DUAL_KERNEL[u & 7] as u128;
    let select = u >> 3;

    let mut acc = 0u128;
    for p in 0..16 {
        if p & select == 0 {
            acc |= byte << (8 * p);
        }
    }

    Block128(acc)
}

/// `μ_j` with `Σ_u eq(r'',u)·bit_u(x) = Σ_j μ_j x^{2^j}`
/// for every `x`, bits in the tower basis.
pub fn linearized_coeffs(eq_mix: &[Block128]) -> Vec<Flat<Block128>> {
    let eq: Vec<Flat<Block128>> = eq_mix.iter().map(|m| m.to_hardware()).collect();

    let mut gamma: Vec<Flat<Block128>> = (0..BITS).map(|u| dual_basis(u).to_hardware()).collect();
    let mut mu = Vec::with_capacity(BITS);

    for _ in 0..BITS {
        let mut acc = Flat::from_raw(Block128::ZERO);
        for (g, m) in gamma.iter().zip(&eq) {
            acc += *g * *m;
        }

        mu.push(acc);

        for g in gamma.iter_mut() {
            *g = *g * *g;
        }
    }

    mu
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
    use crate::expander::{eq_tensor_b, ring_batch_b};
    use hekate_math::BinaryFieldExtras;

    fn next(state: &mut u128) -> Block128 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;

        Block128(*state)
    }

    /// Every chain row on the given values: wire terms
    /// from `wire_rows()`, tie terms streamed onto `row_of(j)`.
    fn chain_rows(
        gadget: &RingGadget<Block128>,
        pad: &[Flat<Block128>],
        wires: &[[Flat<Block128>; 3]],
    ) -> Vec<Flat<Block128>> {
        let value = |u: Unknown| -> Flat<Block128> {
            match u {
                Unknown::Pad(i) => pad[i as usize],
                Unknown::Wire { mul, role } => wires[mul as usize][role as usize],
            }
        };

        let mut acc: Vec<Flat<Block128>> = gadget
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

        gadget.for_each_tie_row(|j, tie| {
            for (&(pad_idx, _), &t) in gadget.ring.iter().zip(tie) {
                acc[gadget.row_of(j)] += t * value(Unknown::Pad(pad_idx));
            }
        });

        acc
    }

    #[test]
    fn dual_basis_is_trace_dual() {
        for u in 0..BITS {
            for w in 0..BITS {
                let tr = (dual_basis(u) * Block128(1u128 << w)).trace().get();
                assert_eq!(tr == 1, u == w, "u={u} w={w}");
            }
        }
    }

    #[test]
    fn coefficients_reproduce_bit_transposition() {
        let mut state = 0x0f0f_1234_5678_9abc_def0_1122_3344_5566u128;

        let r_mix: Vec<Block128> = (0..7).map(|_| next(&mut state)).collect();
        let eq_mix = eq_tensor_b(&r_mix);
        let mu = linearized_coeffs(&eq_mix);

        for _ in 0..8 {
            let x = next(&mut state);

            let mut lin = Flat::from_raw(Block128::ZERO);
            let mut power = x.to_hardware();

            for m in &mu {
                lin += *m * power;
                power = power * power;
            }

            assert_eq!(lin.to_tower(), ring_batch_b(&[x], &eq_mix));
        }
    }

    #[test]
    fn wires_and_rows_evaluate_to_delta() {
        let mut state = 0x7777_8888_9999_aaaa_bbbb_cccc_dddd_eeeeu128;

        let r_mix: Vec<Block128> = (0..7).map(|_| next(&mut state)).collect();
        let eq_mix = eq_tensor_b(&r_mix);
        let mu = linearized_coeffs(&eq_mix);

        let ring: Vec<(u32, Flat<Block128>)> = (0..5u32)
            .map(|c| (c, next(&mut state).to_hardware()))
            .collect();
        let pad: Vec<Flat<Block128>> = (0..5).map(|_| next(&mut state).to_hardware()).collect();

        let mut direct = Flat::from_raw(Block128::ZERO);
        for (&(_, a), &h) in ring.iter().zip(&pad) {
            direct += a * ring_batch_b(&[h.to_tower()], &eq_mix).to_hardware();
        }

        let gadget = RingGadget::new(ring, mu, 0);
        assert_eq!(gadget.delta(&pad), direct);

        let wires = gadget.wires(&pad);

        for (i, v) in chain_rows(&gadget, &pad, &wires).iter().enumerate() {
            assert_eq!(*v, Flat::from_raw(Block128::ZERO), "row {i}");
        }

        for wire in &wires {
            assert_eq!(wire[0], wire[1]);
            assert_eq!(wire[0] * wire[1], wire[2]);
        }

        let mut delta = Flat::from_raw(Block128::ZERO);
        for &(u, coeff) in &gadget.delta_form() {
            delta += coeff
                * match u {
                    Unknown::Pad(i) => pad[i as usize],
                    Unknown::Wire { mul, role } => wires[mul as usize][role as usize],
                };
        }

        assert_eq!(delta, direct);
    }

    /// A floating wire would let a prover pick its value freely;
    /// every wire must be pinned by a row or a Hadamard triple.
    #[test]
    fn chain_is_rigid_under_single_wire_perturbation() {
        let mut state = 0xdead_beef_0bad_f00d_1357_9bdf_2468_ace0u128;

        let r_mix: Vec<Block128> = (0..7).map(|_| next(&mut state)).collect();
        let mu = linearized_coeffs(&eq_tensor_b(&r_mix));

        let ring: Vec<(u32, Flat<Block128>)> = (0..7u32)
            .map(|c| (c, next(&mut state).to_hardware()))
            .collect();
        let pad: Vec<Flat<Block128>> = (0..7).map(|_| next(&mut state).to_hardware()).collect();

        let gadget = RingGadget::new(ring, mu, 0);
        let honest = gadget.wires(&pad);
        let delta = next(&mut state).to_hardware();

        let zero = Flat::from_raw(Block128::ZERO);

        for target in 0..honest.len() {
            for role in 0..3 {
                let mut wires = honest.clone();
                wires[target][role] += delta;

                let row_broken = chain_rows(&gadget, &pad, &wires)
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
        for len in [1u32, 9, 40] {
            let r_mix: Vec<Block128> = (0..7).map(|_| next(&mut state)).collect();
            let mu = linearized_coeffs(&eq_tensor_b(&r_mix));

            let ring: Vec<(u32, Flat<Block128>)> = (0..len)
                .map(|c| (3 * c + 1, next(&mut state).to_hardware()))
                .collect();

            let gadget = RingGadget::new(ring, mu, 0);

            let scales: Vec<Flat<Block128>> = (0..CHAIN_ROWS)
                .map(|_| next(&mut state).to_hardware())
                .collect();

            let mut expected: Vec<(u32, Flat<Block128>)> = gadget
                .ring
                .iter()
                .map(|&(pad, _)| (pad, Flat::from_raw(Block128::ZERO)))
                .collect();

            gadget.for_each_tie_row(|j, tie| {
                for ((_, w), &t) in expected.iter_mut().zip(tie) {
                    *w += scales[gadget.row_of(j)] * t;
                }
            });

            let mut folded = Vec::new();
            gadget.for_each_tie_weight(&scales, |pad, w| folded.push((pad, w)));

            assert_eq!(folded, expected);
        }
    }
}
