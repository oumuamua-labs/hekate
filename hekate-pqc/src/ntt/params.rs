// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec::Vec;
use hekate_core::errors::{self, Error};
use subtle::{ConditionallySelectable, ConstantTimeLess};

use crate::wiring::N;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Butterfly {
    pub(crate) pos_a: usize,
    pub(crate) pos_b: usize,
    pub(crate) w: u32,
}

/// Modulus, depth and root of unity of one NTT.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NttParams {
    pub(crate) q: u32,
    pub(crate) layers: usize,
    pub(crate) zeta: u32,
    pub(crate) n_inv: u32,

    barrett: u64,
}

impl NttParams {
    /// FIPS 203: q = 3329, 7 layers, ζ = 17.
    pub const ML_KEM: Self = Self {
        q: 3329,
        layers: 7,
        zeta: 17,
        n_inv: 3303,
        barrett: u64::MAX / 3329,
    };

    /// FIPS 204: q = 8380417, 8 layers, ζ = 1753.
    pub const ML_DSA: Self = Self {
        q: 8380417,
        layers: 8,
        zeta: 1753,
        n_inv: 8347681,
        barrett: u64::MAX / 8380417,
    };

    /// Checks and builds a custom parameter set.
    ///
    /// # Errors
    /// When q is not an odd prime below 2^31, `layers` is
    /// outside 1..=8, ζ does not have order 2^(layers + 1),
    /// or `n_inv` does not invert 2^layers modulo q.
    pub fn new(q: u32, layers: usize, zeta: u32, n_inv: u32) -> errors::Result<Self> {
        let invalid = |message| Error::Protocol {
            protocol: "ntt_params",
            message,
        };

        if !(1..=8).contains(&layers) {
            return Err(invalid("layers must be 1..=8 for 256 coefficients"));
        }

        if !(3..1 << 31).contains(&q) || !is_prime(q) {
            return Err(invalid("q must be an odd prime below 2^31"));
        }

        let params = Self {
            q,
            layers,
            zeta: zeta % q,
            n_inv: n_inv % q,
            barrett: u64::MAX / q as u64,
        };

        if params.pow(params.zeta, 1 << layers) != q - 1 {
            return Err(invalid(
                "zeta must have multiplicative order 2^(layers + 1)",
            ));
        }

        if params.mul(params.n_inv, 1 << layers) != 1 {
            return Err(invalid("n_inv must invert 2^layers modulo q"));
        }

        Ok(params)
    }

    pub fn q(&self) -> u32 {
        self.q
    }

    pub fn layers(&self) -> usize {
        self.layers
    }

    /// Bits of q.
    pub fn bit_width(&self) -> usize {
        32 - self.q.leading_zeros() as usize
    }

    /// Multiplier that scales the inverse transform.
    pub fn n_inv(&self) -> u32 {
        self.n_inv
    }

    /// The FIPS twiddle table: ζ^BitRev(k) for k < 2^layers.
    pub fn zetas(&self) -> Vec<u32> {
        let count = 1usize << self.layers;

        (0..count)
            .map(|k| {
                let rev = (k as u32).reverse_bits() >> (32 - self.layers);
                self.pow(self.zeta, rev)
            })
            .collect()
    }

    pub(crate) fn gammas(&self) -> Vec<u32> {
        (0..1u32 << self.layers)
            .map(|i| {
                let rev = i.reverse_bits() >> (32 - self.layers);

                self.pow(self.zeta, 2 * rev + 1)
            })
            .collect()
    }

    pub(crate) fn forward_plan(&self) -> Vec<Butterfly> {
        let zetas = self.zetas();

        let mut plan = Vec::with_capacity(self.layers * N / 2);
        let mut k = 1usize;
        let mut len = N / 2;

        for _ in 0..self.layers {
            for start in (0..N).step_by(2 * len) {
                let w = zetas[k];

                k += 1;

                for j in start..start + len {
                    plan.push(Butterfly {
                        pos_a: j,
                        pos_b: j + len,
                        w,
                    });
                }
            }

            len /= 2;
        }

        plan
    }

    pub(crate) fn inverse_plan(&self) -> Vec<Butterfly> {
        let zetas = self.zetas();

        let mut plan = Vec::with_capacity(self.layers * N / 2);
        let mut k = 1usize << self.layers;
        let mut len = N >> self.layers;

        for _ in 0..self.layers {
            for start in (0..N).step_by(2 * len) {
                k -= 1;

                let w = (self.q - zetas[k]) % self.q;

                for j in start..start + len {
                    plan.push(Butterfly {
                        pos_a: j,
                        pos_b: j + len,
                        w,
                    });
                }
            }

            len *= 2;
        }

        plan
    }

    /// Reference forward transform, FIPS 203
    /// Algorithm 9 or FIPS 204 Algorithm 41.
    pub fn ntt(&self, f: &[u32; N]) -> [u32; N] {
        let mut out = *f;
        for bf in self.forward_plan() {
            let (a, b) = (out[bf.pos_a], out[bf.pos_b]);
            let t = self.mul(bf.w, b);

            out[bf.pos_a] = self.add(a, t);
            out[bf.pos_b] = self.sub(a, t);
        }

        out
    }

    /// Reference inverse transform with its scaling,
    /// FIPS 203 Algorithm 10 or FIPS 204 Algorithm 42.
    pub fn intt(&self, f: &[u32; N]) -> [u32; N] {
        let mut out = *f;
        for bf in self.inverse_plan() {
            let (a, b) = (out[bf.pos_a], out[bf.pos_b]);

            out[bf.pos_a] = self.add(a, b);
            out[bf.pos_b] = self.mul(bf.w, self.sub(a, b));
        }

        for c in out.iter_mut() {
            *c = self.mul(self.n_inv, *c);
        }

        out
    }

    pub(crate) fn add(&self, a: u32, b: u32) -> u32 {
        self.reduce_once(a + b)
    }

    pub(crate) fn sub(&self, a: u32, b: u32) -> u32 {
        self.reduce_once(a + self.q - b)
    }

    pub(crate) fn mul(&self, a: u32, b: u32) -> u32 {
        self.divmod(a as u64 * b as u64).1
    }

    /// ⌊x/q⌋ and x mod q, for x below q², by
    /// Barrett reduction: no division instruction.
    pub(crate) fn divmod(&self, x: u64) -> (u32, u32) {
        let q = self.q as u64;
        let estimate = ((x as u128 * self.barrett as u128) >> 64) as u64;
        let rem = x - estimate * q;
        let over = !rem.ct_lt(&q);

        (
            (estimate + u64::conditional_select(&0, &1, over)) as u32,
            u64::conditional_select(&rem, &rem.wrapping_sub(q), over) as u32,
        )
    }

    fn pow(&self, base: u32, mut exp: u32) -> u32 {
        let mut acc = 1u32;
        let mut sq = base % self.q;

        while exp > 0 {
            if exp & 1 == 1 {
                acc = self.mul(acc, sq);
            }

            sq = self.mul(sq, sq);
            exp >>= 1;
        }

        acc
    }

    fn reduce_once(&self, v: u32) -> u32 {
        u32::conditional_select(&v, &v.wrapping_sub(self.q), !v.ct_lt(&self.q))
    }
}

fn is_prime(q: u32) -> bool {
    !q.is_multiple_of(2)
        && (3..)
            .step_by(2)
            .take_while(|d| d * d <= q)
            .all(|d| !q.is_multiple_of(d))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);

        *seed >> 33
    }

    fn random_poly(p: &NttParams, seed: &mut u64) -> [u32; N] {
        core::array::from_fn(|_| (lcg(seed) % p.q as u64) as u32)
    }

    fn negacyclic(p: &NttParams, f: &[u32; N], g: &[u32; N]) -> [u32; N] {
        let mut h = [0u32; N];
        for (i, &fi) in f.iter().enumerate() {
            for (j, &gj) in g.iter().enumerate() {
                let prod = p.mul(fi, gj);
                let k = i + j;

                if k < N {
                    h[k] = p.add(h[k], prod);
                } else {
                    h[k - N] = p.sub(h[k - N], prod);
                }
            }
        }

        h
    }

    fn base_case_product(p: &NttParams, f: &[u32; N], g: &[u32; N]) -> [u32; N] {
        let zetas = p.zetas();

        let mut h = [0u32; N];
        for i in 0..N / 2 {
            let gamma = if i % 2 == 0 {
                zetas[64 + i / 2]
            } else {
                p.sub(0, zetas[64 + i / 2])
            };

            let (a0, a1, b0, b1) = (f[2 * i], f[2 * i + 1], g[2 * i], g[2 * i + 1]);

            h[2 * i] = p.add(p.mul(a0, b0), p.mul(p.mul(a1, b1), gamma));
            h[2 * i + 1] = p.add(p.mul(a0, b1), p.mul(a1, b0));
        }

        h
    }

    #[test]
    fn fips_constants_pass_validation() {
        assert_eq!(
            NttParams::new(3329, 7, 17, 3303).unwrap(),
            NttParams::ML_KEM
        );
        assert_eq!(
            NttParams::new(8380417, 8, 1753, 8347681).unwrap(),
            NttParams::ML_DSA
        );
    }

    #[test]
    fn invalid_parameters_are_rejected() {
        assert!(NttParams::new(3329, 0, 17, 3303).is_err());
        assert!(NttParams::new(3329, 9, 17, 3303).is_err());
        assert!(NttParams::new(3327, 7, 17, 3303).is_err());
        assert!(NttParams::new(3329, 7, 289, 3303).is_err());
        assert!(NttParams::new(3329, 7, 17, 3302).is_err());
    }

    #[test]
    fn zeta_tables_match_fips() {
        let kem = NttParams::ML_KEM.zetas();
        let dsa = NttParams::ML_DSA.zetas();

        assert_eq!(&kem[..4], &[1, 1729, 2580, 3289]);
        assert_eq!(kem[127], 2154);
        assert_eq!(&dsa[..4], &[1, 4808194, 3765607, 3761513]);
        assert_eq!(dsa[255], 7648983);
    }

    #[test]
    fn divmod_matches_exact_division() {
        let mut seed = 5;
        for p in [NttParams::ML_KEM, NttParams::ML_DSA] {
            let q = p.q as u64;
            let top = (q - 1) * (q - 1);

            let edges = [0, 1, q - 1, q, q + 1, 2 * q - 1, top - q, top - 1, top];
            let random = (0..4096).map(|_| (lcg(&mut seed) << 31 | lcg(&mut seed)) % (top + 1));

            for x in edges.into_iter().chain(random) {
                assert_eq!(p.divmod(x), ((x / q) as u32, (x % q) as u32));
            }
        }
    }

    #[test]
    fn inverse_undoes_forward() {
        let mut seed = 7;
        for p in [NttParams::ML_KEM, NttParams::ML_DSA] {
            let f = random_poly(&p, &mut seed);
            assert_eq!(p.intt(&p.ntt(&f)), f);
        }
    }

    #[test]
    fn ml_dsa_pointwise_product_is_negacyclic_convolution() {
        let p = NttParams::ML_DSA;

        let mut seed = 11;

        let f = random_poly(&p, &mut seed);
        let g = random_poly(&p, &mut seed);

        let (fh, gh) = (p.ntt(&f), p.ntt(&g));
        let hh: [u32; N] = core::array::from_fn(|i| p.mul(fh[i], gh[i]));

        assert_eq!(p.intt(&hh), negacyclic(&p, &f, &g));
    }

    #[test]
    fn ml_kem_base_case_product_is_negacyclic_convolution() {
        let p = NttParams::ML_KEM;

        let mut seed = 13;

        let f = random_poly(&p, &mut seed);
        let g = random_poly(&p, &mut seed);

        let hh = base_case_product(&p, &p.ntt(&f), &p.ntt(&g));

        assert_eq!(p.intt(&hh), negacyclic(&p, &f, &g));
    }
}
