// SPDX-License-Identifier: Apache-2.0
// This file is part of the hekate-math project.
// Copyright (C) 2026 Andrei Kochergin <andrei@oumuamua.dev>
// Copyright (C) 2026 Oumuamua Labs <info@oumuamua.dev>. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::errors;
use crate::poly::variant::Error;
use alloc::vec::Vec;
use hekate_math::{Flat, HardwareField};
use zeroize::Zeroize;

/// Lazy MLE representation of `Eq(x, r)`.
/// `O(num_vars)` memory, `O(1)` fold.
#[derive(Clone, Debug, Zeroize)]
pub struct TensorProduct<F: HardwareField> {
    /// Remaining coordinates of `r`; fold
    /// pops the LSB coordinate each round.
    pub r_coords: Vec<Flat<F>>,

    /// Accumulated fold factor:
    /// `new_eq(x) = current_scale · eq_rest(x)`.
    pub current_scale: Flat<F>,
}

impl<F: HardwareField> TensorProduct<F> {
    pub fn new(r: Vec<Flat<F>>) -> Self {
        Self {
            r_coords: r,
            current_scale: Flat::from_raw(F::ONE),
        }
    }

    pub fn num_vars(&self) -> usize {
        self.r_coords.len()
    }

    /// Evaluate `Eq(x, r) = ∏_i ((1-x_i)(1-r_i) + x_i·r_i)`
    /// at an arbitrary point `x`.
    pub fn evaluate_extension(&self, x: &[Flat<F>]) -> errors::Result<Flat<F>> {
        let expected_len = self.r_coords.len();
        let got_len = x.len();

        if got_len != expected_len {
            return Err(Error::PointDimensionMismatch {
                expected_len,
                got_len,
            }
            .into());
        }

        let mut res = self.current_scale;
        let one = Flat::from_raw(F::ONE);

        for (&xi, &ri) in x.iter().zip(self.r_coords.iter()) {
            let term_0 = (one - xi) * (one - ri);
            let term_1 = xi * ri;
            let term = term_0 + term_1;

            res *= term;
        }

        Ok(res)
    }

    /// Allocation-free `Eq(x, r)` evaluation
    /// when both points are already available
    /// as slices.
    #[inline(always)]
    pub fn evaluate_eq_slice(r: &[Flat<F>], x: &[Flat<F>]) -> Flat<F> {
        debug_assert_eq!(r.len(), x.len());

        let mut res = Flat::from_raw(F::ONE);
        let one = Flat::from_raw(F::ONE);

        for (&ri, &xi) in r.iter().zip(x.iter()) {
            let term_0 = (one - ri) * (one - xi);
            let term_1 = ri * xi;

            res *= term_0 + term_1;
        }

        res
    }

    /// JIT `Eq(index, r)` for a hypercube index
    /// (`O(num_vars)`). Hot path in Sumcheck.
    ///
    /// Bit 0 of `index` is variable 0 (LSB-first fold).
    #[inline(always)]
    pub fn evaluate_at_index(&self, index: usize) -> Flat<F> {
        let mut val = self.current_scale;

        for (i, &r_val) in self.r_coords.iter().enumerate() {
            let bit_is_set = (index >> i) & 1 == 1;
            if bit_is_set {
                val *= r_val;
            } else {
                val *= Flat::from_raw(F::ONE) - r_val;
            }
        }

        val
    }

    /// Fold by challenge `u`, dropping the LSB
    /// coordinate `r_0` and multiplying the scale
    /// by `(1-u)(1-r_0) + u·r_0`.
    pub fn fold(&self, u: Flat<F>) -> Self {
        if self.r_coords.is_empty() {
            return self.clone();
        }

        let r_0 = self.r_coords[0];
        let one = F::ONE.to_hardware();

        let term_0 = (one - u) * (one - r_0);
        let term_1 = u * r_0;
        let factor = term_0 + term_1;

        let mut new_r = self.r_coords.clone();
        new_r.remove(0);

        Self {
            r_coords: new_r,
            current_scale: self.current_scale * factor,
        }
    }
}
