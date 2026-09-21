// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! A pre-committed mixture passes at `rho + delta`, not `rho`,
//! which is why `ldt_bits` prices Brakedown App. B Thm 7.
//! Proximity layer only; a false statement needs the other checks.

use hekate::core::config::Config;
use hekate_math::{AdditiveFft, Block128, CantorBasis, Flat, HardwareField, TowerField};

type F = Block128;

const T: usize = 8;
const QUERY_SETS: usize = 200_000;

struct SplitMix(u64);

impl SplitMix {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);

        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);

        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

struct Shape {
    label: &'static str,
    grid_cols: usize,
    support: usize,
    width: usize,
}

impl Shape {
    fn msg_len(&self) -> usize {
        self.grid_cols + self.support
    }

    fn rho(&self) -> f64 {
        self.msg_len() as f64 / self.width as f64
    }

    /// Crossing point of `1 - delta`
    /// and `rho + delta`, Brakedown Thm 7.
    fn balanced_corruption(&self) -> usize {
        (((1.0 - self.rho()) / 2.0) * self.width as f64) as usize
    }
}

fn shapes() -> [Shape; 3] {
    let at = |label, grid_cols| {
        let geom = Config::prod().table_geom(grid_cols);

        Shape {
            label,
            grid_cols,
            support: geom.support_size,
            width: geom.encoded_width,
        }
    };

    [
        at("grid 2^13, fractional", 8192),
        at("grid 2^11, fractional edge", 2048),
        at("grid 2^9, full-half", 512),
    ]
}

/// Mirrors `hekate-verifier` `rs_encode_row`;
/// layout drift there voids this probe.
fn encode(msg: &[Flat<F>], width: usize) -> Vec<Flat<F>> {
    let mut buf = vec![Flat::from_raw(F::ZERO); width];
    buf[..msg.len()].copy_from_slice(msg);

    AdditiveFft::<F>::new(width.trailing_zeros())
        .forward_scalar(&mut buf)
        .unwrap();

    buf
}

fn domain(width: usize) -> Vec<Flat<F>> {
    let betas: Vec<Flat<F>> = (0..width.trailing_zeros() as usize)
        .map(|j| F::from(CantorBasis::beta_tower(j).0 as u128).to_hardware())
        .collect();

    (0..width)
        .map(|x| {
            let mut point = Flat::from_raw(F::ZERO);
            for (j, beta) in betas.iter().enumerate() {
                if (x >> j) & 1 == 1 {
                    point += *beta;
                }
            }

            point
        })
        .collect()
}

/// Degree `msg_len - 1` keeps it inside the code.
fn min_weight_codeword(shape: &Shape, dom: &[Flat<F>]) -> Vec<Flat<F>> {
    let roots = shape.msg_len() - 1;

    (0..shape.width)
        .map(|x| {
            let mut acc = Flat::from_raw(F::ONE);
            for root in dom.iter().take(roots) {
                acc *= dom[x] + *root;
            }

            acc
        })
        .collect()
}

fn empirical_accept(mask: &[bool], rng: &mut SplitMix) -> f64 {
    let mut accepted = 0;
    for _ in 0..QUERY_SETS {
        let mut pass = true;
        for _ in 0..T {
            if !mask[rng.below(mask.len())] {
                pass = false;
                break;
            }
        }

        if pass {
            accepted += 1;
        }
    }

    accepted as f64 / QUERY_SETS as f64
}

#[test]
fn reconstructed_domain_matches_the_shipped_encode() {
    for shape in shapes() {
        let dim = shape.width.trailing_zeros() as usize;

        for j in 0..dim.min(shape.msg_len().trailing_zeros() as usize) {
            let mut msg = vec![Flat::from_raw(F::ZERO); shape.msg_len()];
            msg[1 << j] = Flat::from_raw(F::ONE);

            let code = encode(&msg, shape.width);
            let zeros: Vec<usize> = (0..shape.width)
                .filter(|&x| code[x] == Flat::from_raw(F::ZERO))
                .collect();

            assert_eq!(
                zeros,
                (0..1 << j).collect::<Vec<usize>>(),
                "{}: subspace vanishing polynomial s_{j} has the wrong zero set",
                shape.label
            );
        }
    }
}

#[test]
fn mixture_commitment_beats_the_priced_per_query_rate() {
    println!("\n=== per-query proximity pass rate, shipped encode ===");
    println!("chosen:  committed word is a codeword, sent fold is wrong");
    println!("mixture: committed word is delta-far, opened as the far codeword");
    println!("accept at t = {T}, {QUERY_SETS} query sets\n");

    println!(
        "{:<24} {:>7} {:>9} {:>10} {:>10} {:>10}",
        "table", "rho", "strategy", "per-query", "bound", "accept"
    );

    for shape in shapes() {
        let dom = domain(shape.width);
        let deviation = min_weight_codeword(&shape, &dom);

        let zero = Flat::from_raw(F::ZERO);
        let coincide: Vec<bool> = deviation.iter().map(|d| *d == zero).collect();

        assert_eq!(
            coincide.iter().filter(|&&c| c).count(),
            shape.msg_len() - 1,
            "{}: deviation is not a minimum-weight codeword",
            shape.label
        );

        let rho = shape.rho();
        let mut rng = SplitMix(0x5A5A_1234);

        for (strategy, corrupt, bound) in [
            ("chosen", 0, rho),
            ("mixture", shape.balanced_corruption(), (1.0 + rho) / 2.0),
        ] {
            let mut mask = coincide.clone();
            let mut placed = 0;

            while placed < corrupt {
                let j = rng.below(shape.width);

                if !mask[j] {
                    mask[j] = true;
                    placed += 1;
                }
            }

            let per_query = mask.iter().filter(|&&ok| ok).count() as f64 / shape.width as f64;
            let accept = empirical_accept(&mask, &mut rng);

            println!(
                "{:<24} {:>7.4} {:>9} {:>10.4} {:>10.4} {:>9.2}%",
                shape.label,
                rho,
                strategy,
                per_query,
                bound,
                100.0 * accept
            );

            assert!(
                (per_query - bound).abs() < 0.01,
                "{}: {strategy} per-query rate {per_query:.4} deviates from {bound:.4}",
                shape.label
            );

            let expected = per_query.powi(T as i32);
            assert!(
                (accept - expected).abs() < 0.02,
                "{}: {strategy} accept {accept:.4} deviates from {expected:.4}",
                shape.label
            );
        }
    }

    println!();
}
