// SPDX-License-Identifier: Apache-2.0
// This file is part of the hekate project.
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

use alloc::vec::Vec;
use hekate_math::{Flat, HardwareField, TowerField};
use serde::{Deserialize, Serialize};

/// Round polynomial `g(X)` sent by the prover
/// in Sumcheck, carried as evaluations
/// at `{0, 1, …, degree}`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnivariatePoly<F: TowerField> {
    pub evals: Vec<F>,
}

impl<F: TowerField> UnivariatePoly<F> {
    pub fn new(evals: Vec<F>) -> Self {
        Self { evals }
    }

    /// Evaluate at `z` using barycentric Lagrange
    /// interpolation with Montgomery batch inversion.
    /// Stack-limited to degree 15.
    pub fn evaluate(&self, z: F) -> F {
        let k = self.evals.len();
        if k == 0 {
            return F::ZERO;
        }
        if k == 1 {
            return self.evals[0];
        }

        const MAX_POINTS: usize = 16;

        assert!(k <= MAX_POINTS, "Degree too large for stack buffer");

        let mut z_prod = F::ONE;
        let mut v_vals = [F::ONE; MAX_POINTS];

        for (i, v_val) in v_vals.iter_mut().take(k).enumerate() {
            let xi = F::from(i as u8);
            if z == xi {
                return self.evals[i];
            }

            z_prod *= z - xi;

            let mut den = F::ONE;
            for j in 0..k {
                if i == j {
                    continue;
                }

                let xj = F::from(j as u8);
                den *= xi - xj;
            }

            *v_val = (z - xi) * den;
        }

        let mut prefixes = [F::ONE; MAX_POINTS];
        let mut acc = F::ONE;

        for i in 0..k {
            prefixes[i] = acc;
            acc *= v_vals[i];
        }

        let mut acc_inv = acc.invert();
        for i in (0..k).rev() {
            let inv = prefixes[i] * acc_inv;
            acc_inv *= v_vals[i];
            v_vals[i] = inv;
        }

        let mut result = F::ZERO;
        for (eval_i, v_i) in self.evals.iter().zip(v_vals.iter()).take(k) {
            result += *eval_i * z_prod * *v_i;
        }

        result
    }
}

impl<F: HardwareField> UnivariatePoly<F> {
    /// Same as `evaluate` but takes the challenge
    /// already in the hardware (flat) basis.
    pub fn evaluate_hw(&self, z_hw: Flat<F>) -> Flat<F> {
        let k = self.evals.len();
        debug_assert!(k > 0);

        if k == 1 {
            return self.evals[0].to_hardware();
        }

        const MAX_POINTS: usize = 16;

        assert!(k <= MAX_POINTS, "Degree too large for stack buffer");

        let one = Flat::from_raw(F::ONE);

        let mut z_prod = one;
        let mut v_vals = [one; MAX_POINTS];

        for (i, v_val) in v_vals.iter_mut().take(k).enumerate() {
            let xi = F::from(i as u8).to_hardware();
            if z_hw == xi {
                return self.evals[i].to_hardware();
            }

            z_prod *= z_hw - xi;

            let mut den = one;
            for j in 0..k {
                if i == j {
                    continue;
                }

                let xj = F::from(j as u8).to_hardware();
                den *= xi - xj;
            }

            *v_val = (z_hw - xi) * den;
        }

        let mut prefixes = [one; MAX_POINTS];
        let mut acc = one;

        for i in 0..k {
            prefixes[i] = acc;
            acc *= v_vals[i];
        }

        let mut acc_inv = acc.to_tower().invert().to_hardware();
        for i in (0..k).rev() {
            let inv = prefixes[i] * acc_inv;
            acc_inv *= v_vals[i];
            v_vals[i] = inv;
        }

        let mut result = Flat::from_raw(F::ZERO);
        for (eval_i, v_i) in self.evals.iter().zip(v_vals.iter()).take(k) {
            let term = eval_i.to_hardware() * z_prod * *v_i;
            result += term;
        }

        result
    }
}
