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

/// The split in `lo..=hi` that `admissible` accepts with
/// the fewest tensor-vector, opened-row and claim bytes;
/// ties to the narrower, `None` when none qualifies.
pub fn cheapest_split_vars(
    lo: usize,
    hi: usize,
    num_queries: usize,
    row_bytes: impl Fn(usize) -> usize,
    claim_bytes: impl Fn(usize) -> usize,
    admissible: impl Fn(usize) -> bool,
) -> Option<usize> {
    let vector_cost = Q_VECTOR_ELEM_BYTES as u128;
    let cost = |c: usize| {
        vector_cost * (1u128 << c)
            + (num_queries * row_bytes(c)).max(1) as u128
            + claim_bytes(c) as u128
    };

    (lo..=hi)
        .filter(|&c| admissible(c))
        .min_by_key(|&c| cost(c))
}

/// Fewest split variables whose grid holds the support block.
/// Narrower, the grid falls to full-half and `check_security`
/// rejects `support_size = grid_cols` below `num_queries`.
pub fn support_floor_vars(support_size: usize) -> usize {
    match support_size > 1 {
        true => (support_size - 1).ilog2() as usize + 1,
        false => 1,
    }
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
    fn packed_split_agrees_with_clamp_at_num_vars() {
        for (num_vars, q, s, rb) in cases() {
            let at_clamp = (q * rb).max(1) as u128 + Q_VECTOR_ELEM_BYTES as u128 * (1 << num_vars);
            let unpacked = scan_argmin(num_vars, q, s, rb);

            if unpacked == num_vars {
                let halved = cheapest_split_vars(
                    num_vars,
                    num_vars + 1,
                    q,
                    |c| match c > num_vars {
                        true => rb.div_ceil(2),
                        false => rb,
                    },
                    |_| 0,
                    |_| true,
                )
                .unwrap();

                let cost_up = (q * rb.div_ceil(2)).max(1) as u128
                    + Q_VECTOR_ELEM_BYTES as u128 * (2 << num_vars);

                assert_eq!(
                    halved > num_vars,
                    cost_up < at_clamp,
                    "n={num_vars} q={q} rb={rb}"
                );
            }
        }
    }

    #[test]
    fn cheapest_split_reproduces_unpacked_objective() {
        for (num_vars, q, s, rb) in cases() {
            let floor = support_floor_vars(s).min(num_vars);

            let cheapest = cheapest_split_vars(
                1,
                num_vars,
                q,
                |c| rb << (num_vars - c),
                |_| 0,
                |c| c >= floor,
            );

            assert_eq!(
                cheapest,
                Some(scan_argmin(num_vars, q, s, rb)),
                "n={num_vars} q={q} s={s} rb={rb}"
            );
        }
    }

    #[test]
    fn cheapest_split_is_none_when_nothing_qualifies() {
        assert_eq!(
            cheapest_split_vars(1, 12, 287, |_| 64, |_| 0, |_| false),
            None
        );
    }

    #[test]
    fn claim_bytes_move_split_down() {
        let row_bytes = |c: usize| 4096usize >> c.min(12);
        let free = cheapest_split_vars(1, 12, 4, row_bytes, |_| 0, |_| true).unwrap();
        let charged = cheapest_split_vars(1, 12, 4, row_bytes, |c| 1 << (c + 8), |_| true).unwrap();

        assert!(charged < free, "{charged} against {free}");
    }
}
