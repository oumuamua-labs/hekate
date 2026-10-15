// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! A pre-committed mixture passes at `rho + delta`, not `rho`,
//! which is why `ldt_bits` prices Brakedown App. B Thm 7.
//! Proximity layer only; a false statement needs the other checks.

use hekate::core::config::{CodeRate, Config};
use hekate_math::{AdditiveFft, Block128, CantorBasis, Flat, TowerField};
use hekate_verifier::evaluator::rs_encode_at;

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
    config: Config,
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

fn shapes() -> [Shape; 6] {
    let at = |label: &'static str, config: &Config, grid_cols: usize| {
        let geom = config.table_geom(grid_cols);

        Shape {
            label,
            config: config.clone(),
            grid_cols,
            support: geom.support_size,
            width: geom.encoded_width,
        }
    };

    let half = Config::prod();
    let quarter = Config::prod_at(CodeRate::Quarter);

    [
        at("1/2, grid 2^13, fractional", &half, 8192),
        at("1/2, grid 2^11, fractional edge", &half, 2048),
        at("1/2, grid 2^9, full-half", &half, 512),
        at("1/4, grid 2^13, fractional", &quarter, 8192),
        at("1/4, grid 2^11, fractional edge", &quarter, 2048),
        at("1/4, grid 2^9, full-half", &quarter, 512),
    ]
}

fn domain(width: usize) -> Vec<Flat<F>> {
    let basis = CantorBasis::<F>::new(width.trailing_zeros() as usize).unwrap();

    (0..width).map(|x| basis.point(x).unwrap()).collect()
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

fn shipped_deviation(shape: &Shape) -> Vec<Flat<F>> {
    let zero = Flat::from_raw(F::ZERO);
    let model = min_weight_codeword(shape, &domain(shape.width));

    let mut coeffs = model.clone();
    AdditiveFft::<F>::new(shape.width.trailing_zeros())
        .unwrap()
        .inverse_scalar(&mut coeffs)
        .unwrap();

    assert!(
        coeffs[shape.msg_len()..].iter().all(|c| *c == zero),
        "{}: deviation leaves the message span",
        shape.label
    );

    let message = [
        &coeffs[shape.support..shape.msg_len()],
        &coeffs[..shape.support],
    ]
    .concat();

    let columns: Vec<usize> = (0..shape.width).collect();
    let shipped = rs_encode_at::<F>(&message, shape.grid_cols, &shape.config, &columns).unwrap();

    assert_eq!(
        shipped, model,
        "{}: verifier encode departs from the modelled code",
        shape.label
    );

    shipped
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
fn shipped_encoder_realises_minimum_weight_deviation() {
    let zero = Flat::from_raw(F::ZERO);

    for shape in shapes() {
        let zeros: Vec<usize> = shipped_deviation(&shape)
            .iter()
            .enumerate()
            .filter_map(|(x, v)| (*v == zero).then_some(x))
            .collect();

        assert_eq!(
            zeros,
            (0..shape.msg_len() - 1).collect::<Vec<usize>>(),
            "{}",
            shape.label
        );
    }
}

#[test]
fn mixture_commitment_beats_the_priced_per_query_rate() {
    println!("\n=== per-query proximity pass rate, shipped encode ===");
    println!("chosen:  committed word is a codeword, sent fold is wrong");
    println!("mixture: committed word is delta-far, opened as the far codeword");
    println!("accept at t = {T}, {QUERY_SETS} query sets\n");

    println!(
        "{:<32} {:>7} {:>9} {:>10} {:>10} {:>10}",
        "table", "rho", "strategy", "per-query", "bound", "accept"
    );

    for shape in shapes() {
        let deviation = shipped_deviation(&shape);

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
                "{:<32} {:>7.4} {:>9} {:>10.4} {:>10.4} {:>9.2}%",
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
