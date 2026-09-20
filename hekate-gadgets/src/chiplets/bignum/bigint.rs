// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::errors::{self, Error};

pub const LIMB_BITS: usize = 32;
pub const PAIR_BITS: usize = 64;

/// # Panics
/// `limbs.len() <= 2 * i + 1`.
pub fn limb_pair(limbs: &[u32], i: usize) -> u64 {
    u64::from(limbs[2 * i]) | (u64::from(limbs[2 * i + 1]) << LIMB_BITS)
}

/// `sums[k] = Σ_{2i+j=k} pair_i(a) · b_j`,
/// 64-bit `a` limbs against 32-bit `b` limbs.
pub fn column_sums(a: &[u32], b: &[u32], sums: &mut [u128]) {
    sums.fill(0);

    for i in 0..a.len() / 2 {
        let ai = u128::from(limb_pair(a, i));
        for (j, &bj) in b.iter().enumerate() {
            sums[2 * i + j] += ai * u128::from(bj);
        }
    }
}

/// `add` lands in the low columns;
/// `carries[k]` is the carry into column `k`.
pub fn normalise(sums: &[u128], add: &[u32], digits: &mut [u32], carries: &mut [u128]) {
    carries[0] = 0;

    for k in 0..sums.len() {
        let extra = add.get(k).map_or(0, |&v| u128::from(v));
        let t = sums[k] + extra + carries[k];

        digits[k] = t as u32;
        carries[k + 1] = t >> LIMB_BITS;
    }
}

/// Branch-free binary long division of `t` by `n`;
/// `acc` is `n.len() + 1` limbs of scratch and holds
/// the remainder in its low `n.len()` limbs on return.
///
/// # Errors
/// The quotient exceeds `n.len()` limbs.
pub fn divrem(t: &[u32], n: &[u32], q: &mut [u32], acc: &mut [u32]) -> errors::Result<()> {
    let limbs = n.len();

    q.fill(0);
    acc.fill(0);

    let mut overflow = 0u32;
    for bit in (0..t.len() * LIMB_BITS).rev() {
        let mut carry = (t[bit / LIMB_BITS] >> (bit % LIMB_BITS)) & 1;
        for limb in acc.iter_mut() {
            let out = *limb >> (LIMB_BITS - 1);
            *limb = (*limb << 1) | carry;
            carry = out;
        }

        let take = geq_mask(acc, n);
        sub_masked(acc, n, take);

        let fits = u32::from(bit < limbs * LIMB_BITS);

        overflow |= take & (fits ^ 1);
        q[(bit / LIMB_BITS) % limbs] |= (take & fits) << (bit % LIMB_BITS);
    }

    if overflow != 0 {
        return Err(Error::Protocol {
            protocol: "bignum",
            message: "quotient exceeds the modulus width: operand not reduced",
        });
    }

    Ok(())
}

/// `n - r - 1`, defined for `r < n`.
pub fn sub_one_minus(n: &[u32], r: &[u32], w: &mut [u32]) {
    let mut borrow = 1u32;
    for k in 0..n.len() {
        let (d, b1) = n[k].overflowing_sub(r[k]);
        let (d, b2) = d.overflowing_sub(borrow);

        w[k] = d;
        borrow = u32::from(b1 | b2);
    }
}

pub fn is_less(a: &[u32], b: &[u32]) -> bool {
    let mut borrow = 0u64;
    for k in 0..a.len() {
        let d = u64::from(a[k])
            .wrapping_sub(u64::from(b[k]))
            .wrapping_sub(borrow);

        borrow = d >> 63;
    }

    borrow == 1
}

/// All ones when `r >= n`, zero otherwise: `r - n`
/// borrows out of the top limb exactly when `r < n`.
fn geq_mask(r: &[u32], n: &[u32]) -> u32 {
    let limbs = n.len();

    let mut borrow = 0u64;
    for k in 0..limbs {
        let d = u64::from(r[k])
            .wrapping_sub(u64::from(n[k]))
            .wrapping_sub(borrow);

        borrow = d >> 63;
    }

    let top = u64::from(r[limbs]).wrapping_sub(borrow);
    let negative = (top >> 63) as u32;

    negative.wrapping_sub(1)
}

fn sub_masked(r: &mut [u32], n: &[u32], mask: u32) {
    let limbs = n.len();

    let mut borrow = 0u32;
    for k in 0..limbs {
        let (d, b1) = r[k].overflowing_sub(n[k] & mask);
        let (d, b2) = d.overflowing_sub(borrow);

        r[k] = d;
        borrow = u32::from(b1 | b2);
    }

    r[limbs] = r[limbs].wrapping_sub(borrow);
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn from_u128(value: u128, limbs: usize) -> Vec<u32> {
        (0..limbs)
            .map(|k| match k < 4 {
                true => (value >> (32 * k)) as u32,
                false => 0,
            })
            .collect()
    }

    fn to_u128(limbs: &[u32]) -> u128 {
        limbs
            .iter()
            .enumerate()
            .fold(0u128, |acc, (k, &v)| acc | (u128::from(v) << (32 * k)))
    }

    #[test]
    fn column_sums_match_schoolbook_product() {
        let a = from_u128(0x0123_4567_89ab_cdef, 4);
        let b = from_u128(0xfedc_ba98_7654_3210, 4);

        let mut sums = vec![0u128; 8];

        column_sums(&a, &b, &mut sums);

        let mut digits = vec![0u32; 8];
        let mut carries = vec![0u128; 9];

        normalise(&sums, &[], &mut digits, &mut carries);

        assert_eq!(
            to_u128(&digits[..4]),
            0x0123_4567_89ab_cdefu128 * 0xfedc_ba98_7654_3210
        );
    }

    #[test]
    fn divrem_reproduces_dividend() {
        let t = from_u128(0xdead_beef_cafe_babe_1234_5678_9abc_def0, 8);
        let n = from_u128(0x0000_0000_0000_0001_0000_0000_0000_0003, 4);

        let mut q = vec![0u32; 4];
        let mut acc = vec![0u32; 5];

        divrem(&t, &n, &mut q, &mut acc).unwrap();

        let dividend = to_u128(&t[..4]);
        let modulus = to_u128(&n);

        assert_eq!(to_u128(&q), dividend / modulus);
        assert_eq!(to_u128(&acc[..4]), dividend % modulus);
    }

    #[test]
    fn divrem_rejects_unreduced_operand() {
        let mut t = vec![0u32; 8];
        t[7] = 1;

        let n = from_u128(3, 4);

        let mut q = vec![0u32; 4];
        let mut acc = vec![0u32; 5];

        assert!(divrem(&t, &n, &mut q, &mut acc).is_err());
    }

    #[test]
    fn complement_completes_modulus() {
        let n = from_u128(0x1_0000_0000_0000_0003, 4);
        let r = from_u128(0x0000_0000_dead_beef, 4);

        let mut w = vec![0u32; 4];
        sub_one_minus(&n, &r, &mut w);

        assert_eq!(to_u128(&w), to_u128(&n) - to_u128(&r) - 1);
    }

    #[test]
    fn is_less_orders_limbs() {
        let a = from_u128(0x1_0000_0000, 4);
        let b = from_u128(0x1_0000_0001, 4);

        assert!(is_less(&a, &b));
        assert!(!is_less(&b, &a));
        assert!(!is_less(&a, &a));
    }
}
