// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

/// Bincode varint size of one `Block128`:
/// 1 tag byte + 16.
const Q_VECTOR_ELEM_BYTES: usize = 17;

#[cfg(feature = "std")]
pub use std::time::Instant;

#[cfg(not(feature = "std"))]
#[derive(Clone, Copy, Debug)]
pub struct Instant;

#[cfg(not(feature = "std"))]
impl Instant {
    pub fn now() -> Self {
        Self
    }

    pub fn elapsed(&self) -> core::time::Duration {
        core::time::Duration::from_secs(0)
    }
}

/// Splitting variable `c` minimising proof bytes,
/// the tensor vector against opened rows at `rs_field`
/// widths. `table_geom`'s commit width is not priced.
#[inline(always)]
pub fn compute_split_vars(
    num_vars: usize,
    num_queries: usize,
    support_size: usize,
    row_bytes: usize,
) -> usize {
    if num_vars == 0 {
        return 0;
    }

    let vector_cost = Q_VECTOR_ELEM_BYTES as u128;
    let opened_cost = ((num_queries * row_bytes).max(1)) as u128;

    let factor = (opened_cost / vector_cost).max(1);
    let floor_c = ((num_vars + factor.ilog2() as usize) / 2).min(num_vars);

    let cost = |c: usize| vector_cost * (1u128 << c) + opened_cost * (1u128 << (num_vars - c));

    // floor_c never overshoots the argmin
    let mut optimal_c = floor_c;
    while optimal_c < num_vars && cost(optimal_c + 1) < cost(optimal_c) {
        optimal_c += 1;
    }

    let support_floor = if support_size > 1 {
        (support_size - 1).ilog2() as usize + 1
    } else {
        1
    };

    optimal_c.max(support_floor).clamp(1, num_vars)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    const QUERIES: [usize; 4] = [8, 32, 128, 176];
    const SUPPORTS: [usize; 4] = [4, 32, 128, 512];
    const ROW_BYTES: [usize; 7] = [4, 12, 44, 244, 1024, 4096, 27056];

    fn scan_argmin(
        num_vars: usize,
        num_queries: usize,
        support_size: usize,
        row_bytes: usize,
    ) -> usize {
        let vector_cost = Q_VECTOR_ELEM_BYTES as u128;
        let opened_cost = ((num_queries * row_bytes).max(1)) as u128;

        let support_floor = if support_size > 1 {
            (support_size - 1).ilog2() as usize + 1
        } else {
            1
        };

        let lo = support_floor.clamp(1, num_vars);

        (lo..=num_vars)
            .min_by_key(|&c| vector_cost * (1u128 << c) + opened_cost * (1u128 << (num_vars - c)))
            .unwrap()
    }

    fn cases() -> Vec<(usize, usize, usize, usize)> {
        let mut out = Vec::new();
        for num_vars in 1..=24 {
            for &q in &QUERIES {
                for &s in &SUPPORTS {
                    out.extend(ROW_BYTES.iter().map(|&rb| (num_vars, q, s, rb)));
                }
            }
        }

        out
    }

    #[test]
    fn split_vars_hits_the_discrete_argmin() {
        for (num_vars, q, s, rb) in cases() {
            assert_eq!(
                compute_split_vars(num_vars, q, s, rb),
                scan_argmin(num_vars, q, s, rb),
                "n={num_vars} q={q} s={s} rb={rb}"
            );
        }
    }

    #[test]
    fn split_vars_respects_the_support_floor() {
        for (num_vars, q, s, rb) in cases() {
            let c = compute_split_vars(num_vars, q, s, rb);
            let floor = if s > 1 {
                (s - 1).ilog2() as usize + 1
            } else {
                1
            };

            assert!(
                c >= floor.min(num_vars),
                "n={num_vars} q={q} s={s} rb={rb} gave c={c}"
            );
        }
    }

    #[test]
    fn split_vars_collapses_for_a_single_row() {
        assert_eq!(compute_split_vars(0, 176, 128, 244), 0);
    }
}
