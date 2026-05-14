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

use alloc::collections::BTreeMap;
use alloc::string::String;
use hekate_core::errors::{self, Error};
use hekate_math::{Flat, HardwareField, TowerField};

pub struct BusSpecEvaluation<'a, F: HardwareField> {
    pub h_eval: Flat<F>,
    pub s_eval: Flat<F>,
    pub s_recv_eval: Flat<F>,
    pub source_evals: &'a [Flat<F>],
    pub alpha_bus: Flat<F>,
    pub eq_lookup: Flat<F>,
}

pub fn expected_bus_contribution<F: HardwareField>(
    spec: &BusSpecEvaluation<'_, F>,
    gamma: Flat<F>,
    beta: Flat<F>,
    eq_zc: Flat<F>,
) -> Flat<F> {
    let mut key = Flat::from_raw(F::ZERO);
    let mut beta_pow = Flat::from_raw(F::ONE);

    for src in spec.source_evals {
        key += *src * beta_pow;
        beta_pow *= beta;
    }

    let s_eff = spec.s_eval + spec.s_recv_eval;
    let consistency = spec.alpha_bus * (gamma * spec.h_eval + spec.h_eval * key + s_eff) * eq_zc;
    let bus_sum = spec.alpha_bus * spec.h_eval * spec.eq_lookup;

    consistency + bus_sum
}

pub fn check_bus_sum_matching<F: TowerField>(endpoints: &[(String, F)]) -> errors::Result<()> {
    let mut by_bus: BTreeMap<&str, F> = BTreeMap::new();
    for (bus_id, claim) in endpoints {
        let acc = by_bus.entry(bus_id.as_str()).or_insert(F::ZERO);
        *acc += *claim;
    }

    for (_bus_id, total) in by_bus {
        if total != F::ZERO {
            return Err(Error::Protocol {
                protocol: "LogUp",
                message: "bus endpoint sums do not cancel to zero",
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;
    use alloc::vec::Vec;
    use hekate_math::{Block128, HardwareField};

    type F = Block128;

    fn f(v: u128) -> F {
        F::from(v)
    }

    fn honest_h_and_sum(
        gamma: F,
        beta: F,
        sources_per_row: &[Vec<F>],
        s_per_row: &[F],
    ) -> (Vec<F>, F) {
        let mut h = Vec::with_capacity(sources_per_row.len());
        let mut sum = F::ZERO;

        for (srcs, s) in sources_per_row.iter().zip(s_per_row) {
            let mut key = F::ZERO;
            let mut bp = F::ONE;

            for src in srcs {
                key += *src * bp;
                bp *= beta;
            }

            let denom = gamma + key;
            let inv = denom.to_hardware().to_tower().invert();
            let h_i = *s * inv;

            h.push(h_i);

            sum += h_i;
        }

        (h, sum)
    }

    #[test]
    fn consistency_identity_holds_on_hypercube() {
        let gamma = f(0x1234);
        let beta = f(0x5678);
        let sources = vec![vec![f(11), f(22)], vec![f(33), f(44)], vec![f(55), f(66)]];
        let s = vec![f(7), f(13), f(19)];

        let (h, _) = honest_h_and_sum(gamma, beta, &sources, &s);

        for ((srcs, s_i), h_i) in sources.iter().zip(&s).zip(&h) {
            let mut key = F::ZERO;
            let mut bp = F::ONE;

            for src in srcs {
                key += *src * bp;
                bp *= beta;
            }

            let lhs = *h_i * (gamma + key) + *s_i;
            assert_eq!(lhs, F::ZERO, "char-2 identity h·(γ+key) + s must be 0");
        }
    }

    #[test]
    fn expected_contribution_matches_formula() {
        let gamma = f(0xAA).to_hardware();
        let beta = f(0xBB).to_hardware();
        let eq_zc = f(0xCC).to_hardware();
        let alpha_bus = f(0xDD).to_hardware();

        let h_eval = f(0x11).to_hardware();
        let s_eval = f(0x22).to_hardware();

        let source_evals = vec![
            f(0x33).to_hardware(),
            f(0x44).to_hardware(),
            f(0x55).to_hardware(),
        ];

        let spec = BusSpecEvaluation {
            h_eval,
            s_eval,
            s_recv_eval: Flat::from_raw(F::ZERO),
            source_evals: &source_evals,
            alpha_bus,
            eq_lookup: Flat::from_raw(F::ONE),
        };

        let got = expected_bus_contribution(&spec, gamma, beta, eq_zc);

        let mut key = Flat::from_raw(F::ZERO);
        let mut bp = Flat::from_raw(F::ONE);

        for src in &source_evals {
            key += *src * bp;
            bp *= beta;
        }

        let expected =
            alpha_bus * (gamma * h_eval + h_eval * key + s_eval) * eq_zc + alpha_bus * h_eval;

        assert_eq!(got, expected);
    }

    #[test]
    fn bus_matching_zero_sum_honest_pair() {
        let bus = "ram_link".to_string();
        let claim = f(0x9999);

        let endpoints = vec![(bus.clone(), claim), (bus, claim)];

        check_bus_sum_matching(&endpoints).unwrap();
    }

    #[test]
    fn bus_matching_rejects_mismatched_endpoints() {
        let bus = "ram_link".to_string();
        let endpoints = vec![(bus.clone(), f(0x9999)), (bus, f(0xAAAA))];

        let res = check_bus_sum_matching(&endpoints);
        assert!(res.is_err(), "mismatched endpoint sums must fail");
    }

    #[test]
    fn bus_matching_rejects_unpaired_endpoint() {
        let endpoints = vec![("ram_link".to_string(), f(0x9999))];

        let res = check_bus_sum_matching(&endpoints);
        assert!(res.is_err(), "unpaired endpoint must fail");
    }

    #[test]
    fn bus_matching_multiple_bus_ids_independent() {
        let endpoints = vec![
            ("ram_link".to_string(), f(0x1)),
            ("ram_link".to_string(), f(0x1)),
            ("rom_link".to_string(), f(0x2)),
            ("rom_link".to_string(), f(0x2)),
        ];

        check_bus_sum_matching(&endpoints).unwrap();
    }

    #[test]
    fn bus_matching_one_bus_fails_independently() {
        let endpoints = vec![
            ("ram_link".to_string(), f(0x1)),
            ("ram_link".to_string(), f(0x1)),
            ("rom_link".to_string(), f(0x2)),
            ("rom_link".to_string(), f(0x3)),
        ];

        assert!(check_bus_sum_matching(&endpoints).is_err());
    }
}
