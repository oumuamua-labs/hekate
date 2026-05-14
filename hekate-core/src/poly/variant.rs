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

use crate::errors;
use crate::tensor::TensorProduct;
use crate::trace::TraceCompatibleField;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use hekate_math::{Bit, Block8, Block16, Block32, Block64, Block128, Flat};
use zeroize::Zeroize;

/// Failures raised by `PolyVariant` operations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    /// `1 << num_vars` overflowed `usize`.
    DomainTooLarge { num_vars: usize },

    /// Polynomial length disagrees with `2^num_vars`.
    DomainSizeMismatch { expected_len: usize, got_len: usize },

    /// Evaluation point has the
    /// wrong number of coordinates.
    PointDimensionMismatch { expected_len: usize, got_len: usize },

    /// Variant is read-only at fold time.
    UnsupportedFold { kind: &'static str },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DomainTooLarge { num_vars } => {
                write!(
                    f,
                    "Virtual polynomial domain too large: num_vars={num_vars}"
                )
            }
            Self::DomainSizeMismatch {
                expected_len,
                got_len,
            } => write!(
                f,
                "Virtual polynomial domain size mismatch: expected {expected_len}, got {got_len}",
            ),
            Self::PointDimensionMismatch {
                expected_len,
                got_len,
            } => write!(
                f,
                "Virtual polynomial point dimension mismatch: expected {expected_len}, got {got_len}",
            ),
            Self::UnsupportedFold { kind } => {
                write!(
                    f,
                    "Virtual polynomial cannot be folded lazily for kind: {kind}"
                )
            }
        }
    }
}

/// Zero-copy MLE view over a physical trace column.
/// Every variant must deliver `get_at(i)` without
/// heap allocation, the hot path inside Sumcheck.
#[derive(Clone, Debug, Zeroize)]
pub enum PolyVariant<'a, F>
where
    F: TraceCompatibleField,
{
    /// Fully materialized hypercube.
    #[zeroize(skip)]
    Dense(&'a [Flat<F>]),
    #[zeroize(skip)]
    Shifted(&'a [Flat<F>]),

    /// `Eq(x, r)` held lazily as a `TensorProduct`:
    /// `O(num_vars)` memory, `O(1)` fold.
    Eq(TensorProduct<F>),

    /// `(data[i] >> bit_idx) & 1` on a `B8` column.
    #[zeroize(skip)]
    PackedBitB8 {
        data: &'a [Flat<Block8>],
        bit_idx: usize,
    },

    /// `(data[i] >> bit_idx) & 1` on a `B16` column.
    #[zeroize(skip)]
    PackedBitB16 {
        data: &'a [Flat<Block16>],
        bit_idx: usize,
    },

    /// `(data[i] >> bit_idx) & 1` on a `B32` column.
    #[zeroize(skip)]
    PackedBitB32 {
        data: &'a [Flat<Block32>],
        bit_idx: usize,
    },

    /// `(data[i] >> bit_idx) & 1` on a `B64` column.
    #[zeroize(skip)]
    PackedBitB64 {
        data: &'a [Flat<Block64>],
        bit_idx: usize,
    },

    /// `S(i) = Σ_k cols[k][i]` over boolean
    /// columns (one-hot selector groups).
    #[zeroize(skip)]
    CompositeSelector(Vec<&'a [Bit]>),

    /// Mask that is `1` everywhere and
    /// `1 - product_of_challenges` at
    /// index `2^N - 1`. Used to kill
    /// cross-row wrap in Sumcheck.
    #[zeroize(skip)]
    TransitionMask {
        num_vars: usize,
        product_of_challenges: F,
    },

    // ==============================================================
    // Indirect:
    // P(x) = data[indices[x]].
    // Zero-copy permutations and arbitrary wiring.
    // ==============================================================
    #[zeroize(skip)]
    IndirectBit {
        data: &'a [Bit],
        indices: &'a [usize],
    },
    #[zeroize(skip)]
    IndirectB8 {
        data: &'a [Flat<Block8>],
        indices: &'a [usize],
    },
    #[zeroize(skip)]
    IndirectB16 {
        data: &'a [Flat<Block16>],
        indices: &'a [usize],
    },
    #[zeroize(skip)]
    IndirectB32 {
        data: &'a [Flat<Block32>],
        indices: &'a [usize],
    },
    #[zeroize(skip)]
    IndirectB64 {
        data: &'a [Flat<Block64>],
        indices: &'a [usize],
    },
    #[zeroize(skip)]
    IndirectB128 {
        data: &'a [Flat<Block128>],
        indices: &'a [usize],
    },

    // ====================================================
    // Stride Access:
    // P(i) = data[start + i * step].
    // ====================================================
    #[zeroize(skip)]
    StrideBit {
        data: &'a [Bit],
        start: usize,
        step: usize,
        len: usize,
    },
    #[zeroize(skip)]
    StrideB8 {
        data: &'a [Flat<Block8>],
        start: usize,
        step: usize,
        len: usize,
    },
    #[zeroize(skip)]
    StrideB16 {
        data: &'a [Flat<Block16>],
        start: usize,
        step: usize,
        len: usize,
    },
    #[zeroize(skip)]
    StrideB32 {
        data: &'a [Flat<Block32>],
        start: usize,
        step: usize,
        len: usize,
    },
    #[zeroize(skip)]
    StrideB64 {
        data: &'a [Flat<Block64>],
        start: usize,
        step: usize,
        len: usize,
    },
    #[zeroize(skip)]
    StrideB128 {
        data: &'a [Flat<Block128>],
        start: usize,
        step: usize,
        len: usize,
    },

    // ====================================================
    // Cyclic Rotation:
    // P(i) = data[(i + rotation) % len].
    // Uses bitwise masking for modulo (len must be power of 2).
    // ====================================================
    #[zeroize(skip)]
    RotationBit { data: &'a [Bit], rotation: usize },
    #[zeroize(skip)]
    RotationB8 {
        data: &'a [Flat<Block8>],
        rotation: usize,
    },
    #[zeroize(skip)]
    RotationB16 {
        data: &'a [Flat<Block16>],
        rotation: usize,
    },
    #[zeroize(skip)]
    RotationB32 {
        data: &'a [Flat<Block32>],
        rotation: usize,
    },
    #[zeroize(skip)]
    RotationB64 {
        data: &'a [Flat<Block64>],
        rotation: usize,
    },
    #[zeroize(skip)]
    RotationB128 {
        data: &'a [Flat<Block128>],
        rotation: usize,
    },

    // ====================================================
    // Compressed slice views (JIT-promoted to F)
    // ====================================================
    #[zeroize(skip)]
    BitSlice(&'a [Bit]),
    #[zeroize(skip)]
    B8Slice(&'a [Flat<Block8>]),
    #[zeroize(skip)]
    B16Slice(&'a [Flat<Block16>]),
    #[zeroize(skip)]
    B32Slice(&'a [Flat<Block32>]),
    #[zeroize(skip)]
    B64Slice(&'a [Flat<Block64>]),
    #[zeroize(skip)]
    B128Slice(&'a [Flat<Block128>]),

    // ====================================================
    // Same-width views shifted by one row (cyclic)
    // ====================================================
    #[zeroize(skip)]
    ShiftedBitSlice(&'a [Bit]),
    #[zeroize(skip)]
    ShiftedB8Slice(&'a [Flat<Block8>]),
    #[zeroize(skip)]
    ShiftedB16Slice(&'a [Flat<Block16>]),
    #[zeroize(skip)]
    ShiftedB32Slice(&'a [Flat<Block32>]),
    #[zeroize(skip)]
    ShiftedB64Slice(&'a [Flat<Block64>]),
    #[zeroize(skip)]
    ShiftedB128Slice(&'a [Flat<Block128>]),

    #[zeroize(skip)]
    ShiftedPackedBitB8 {
        data: &'a [Flat<Block8>],
        bit_idx: usize,
    },
    #[zeroize(skip)]
    ShiftedPackedBitB16 {
        data: &'a [Flat<Block16>],
        bit_idx: usize,
    },
    #[zeroize(skip)]
    ShiftedPackedBitB32 {
        data: &'a [Flat<Block32>],
        bit_idx: usize,
    },
    #[zeroize(skip)]
    ShiftedPackedBitB64 {
        data: &'a [Flat<Block64>],
        bit_idx: usize,
    },
}

impl<'a, F> PolyVariant<'a, F>
where
    F: TraceCompatibleField,
{
    /// Number of hypercube points (polynomial length).
    pub fn len(&self) -> usize {
        match self {
            Self::Dense(h) => h.len(),
            Self::Shifted(h) => h.len(),
            Self::Eq(t) => 1 << t.num_vars(),
            Self::PackedBitB8 { data, .. } => data.len(),
            Self::PackedBitB16 { data, .. } => data.len(),
            Self::PackedBitB32 { data, .. } => data.len(),
            Self::PackedBitB64 { data, .. } => data.len(),
            Self::TransitionMask { num_vars, .. } => 1 << num_vars,
            Self::CompositeSelector(cols) => {
                if cols.is_empty() {
                    0
                } else {
                    cols[0].len()
                }
            }
            Self::IndirectBit { indices, .. } => indices.len(),
            Self::IndirectB8 { indices, .. } => indices.len(),
            Self::IndirectB16 { indices, .. } => indices.len(),
            Self::IndirectB32 { indices, .. } => indices.len(),
            Self::IndirectB64 { indices, .. } => indices.len(),
            Self::IndirectB128 { indices, .. } => indices.len(),
            Self::StrideBit { len, .. } => *len,
            Self::StrideB8 { len, .. } => *len,
            Self::StrideB16 { len, .. } => *len,
            Self::StrideB32 { len, .. } => *len,
            Self::StrideB64 { len, .. } => *len,
            Self::StrideB128 { len, .. } => *len,
            Self::RotationBit { data, .. } => data.len(),
            Self::RotationB8 { data, .. } => data.len(),
            Self::RotationB16 { data, .. } => data.len(),
            Self::RotationB32 { data, .. } => data.len(),
            Self::RotationB64 { data, .. } => data.len(),
            Self::RotationB128 { data, .. } => data.len(),
            Self::BitSlice(h) => h.len(),
            Self::B8Slice(h) => h.len(),
            Self::B16Slice(h) => h.len(),
            Self::B32Slice(h) => h.len(),
            Self::B64Slice(h) => h.len(),
            Self::B128Slice(h) => h.len(),
            Self::ShiftedBitSlice(h) => h.len(),
            Self::ShiftedB8Slice(h) => h.len(),
            Self::ShiftedB16Slice(h) => h.len(),
            Self::ShiftedB32Slice(h) => h.len(),
            Self::ShiftedB64Slice(h) => h.len(),
            Self::ShiftedB128Slice(h) => h.len(),
            Self::ShiftedPackedBitB8 { data, .. } => data.len(),
            Self::ShiftedPackedBitB16 { data, .. } => data.len(),
            Self::ShiftedPackedBitB32 { data, .. } => data.len(),
            Self::ShiftedPackedBitB64 { data, .. } => data.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read the hypercube value at `index`,
    /// lifted into `Flat<F>`. `O(1)` for
    /// slice variants, `O(num_vars)` for `Eq`.
    #[inline(always)]
    pub fn get_at(&self, index: usize) -> Flat<F> {
        match self {
            Self::Dense(h) => h[index],
            Self::Shifted(h) => {
                let len = h.len();
                let next_idx = if index + 1 == len { 0 } else { index + 1 };
                h[next_idx]
            }
            Self::Eq(t) => t.evaluate_at_index(index),
            Self::PackedBitB8 { data, bit_idx } => {
                let bit = data[index].tower_bit(*bit_idx);
                if bit == 1 {
                    Flat::from_raw(F::ONE)
                } else {
                    Flat::from_raw(F::ZERO)
                }
            }
            Self::PackedBitB16 { data, bit_idx } => {
                let bit = data[index].tower_bit(*bit_idx);
                if bit == 1 {
                    Flat::from_raw(F::ONE)
                } else {
                    Flat::from_raw(F::ZERO)
                }
            }
            Self::PackedBitB32 { data, bit_idx } => {
                let bit = data[index].tower_bit(*bit_idx);
                if bit == 1 {
                    Flat::from_raw(F::ONE)
                } else {
                    Flat::from_raw(F::ZERO)
                }
            }
            Self::PackedBitB64 { data, bit_idx } => {
                let bit = data[index].tower_bit(*bit_idx);
                if bit == 1 {
                    Flat::from_raw(F::ONE)
                } else {
                    Flat::from_raw(F::ZERO)
                }
            }
            Self::TransitionMask {
                num_vars,
                product_of_challenges,
            } => {
                let last_idx: usize = (1 << num_vars) - 1;
                if index == last_idx {
                    Flat::from_raw(F::ONE - *product_of_challenges)
                } else {
                    Flat::from_raw(F::ONE)
                }
            }
            Self::CompositeSelector(cols) => {
                let mut sum = Flat::from_raw(F::default());
                for col in cols {
                    if col[index].0 == 1 {
                        sum += Flat::from_raw(F::ONE);
                    }
                }

                sum
            }

            Self::IndirectBit { data, indices } => Flat::from_raw(F::from(data[indices[index]])),
            Self::IndirectB8 { data, indices } => F::promote_flat(data[indices[index]]),
            Self::IndirectB16 { data, indices } => F::promote_flat(data[indices[index]]),
            Self::IndirectB32 { data, indices } => F::promote_flat(data[indices[index]]),
            Self::IndirectB64 { data, indices } => F::promote_flat(data[indices[index]]),
            Self::IndirectB128 { data, indices } => F::promote_flat(data[indices[index]]),

            Self::StrideBit {
                data, start, step, ..
            } => Flat::from_raw(F::from(data[start + index * step])),
            Self::StrideB8 {
                data, start, step, ..
            } => F::promote_flat(data[start + index * step]),
            Self::StrideB16 {
                data, start, step, ..
            } => F::promote_flat(data[start + index * step]),
            Self::StrideB32 {
                data, start, step, ..
            } => F::promote_flat(data[start + index * step]),
            Self::StrideB64 {
                data, start, step, ..
            } => F::promote_flat(data[start + index * step]),
            Self::StrideB128 {
                data, start, step, ..
            } => F::promote_flat(data[start + index * step]),

            Self::RotationBit { data, rotation } => {
                Flat::from_raw(F::from(data[(index + rotation) & (data.len() - 1)]))
            }
            Self::RotationB8 { data, rotation } => {
                F::promote_flat(data[(index + rotation) & (data.len() - 1)])
            }
            Self::RotationB16 { data, rotation } => {
                F::promote_flat(data[(index + rotation) & (data.len() - 1)])
            }
            Self::RotationB32 { data, rotation } => {
                F::promote_flat(data[(index + rotation) & (data.len() - 1)])
            }
            Self::RotationB64 { data, rotation } => {
                F::promote_flat(data[(index + rotation) & (data.len() - 1)])
            }
            Self::RotationB128 { data, rotation } => {
                F::promote_flat(data[(index + rotation) & (data.len() - 1)])
            }

            Self::BitSlice(s) => Flat::from_raw(F::from(s[index])),
            Self::B8Slice(s) => F::promote_flat(s[index]),
            Self::B16Slice(s) => F::promote_flat(s[index]),
            Self::B32Slice(s) => F::promote_flat(s[index]),
            Self::B64Slice(s) => F::promote_flat(s[index]),
            Self::B128Slice(s) => F::promote_flat(s[index]),

            Self::ShiftedBitSlice(s) => {
                let len = s.len();
                let next_idx = if index + 1 == len { 0 } else { index + 1 };

                Flat::from_raw(F::from(s[next_idx]))
            }
            Self::ShiftedB8Slice(s) => {
                let len = s.len();
                let next_idx = if index + 1 == len { 0 } else { index + 1 };

                F::promote_flat(s[next_idx])
            }
            Self::ShiftedB16Slice(s) => {
                let len = s.len();
                let next_idx = if index + 1 == len { 0 } else { index + 1 };

                F::promote_flat(s[next_idx])
            }
            Self::ShiftedB32Slice(s) => {
                let len = s.len();
                let next_idx = if index + 1 == len { 0 } else { index + 1 };

                F::promote_flat(s[next_idx])
            }
            Self::ShiftedB64Slice(s) => {
                let len = s.len();
                let next_idx = if index + 1 == len { 0 } else { index + 1 };

                F::promote_flat(s[next_idx])
            }
            Self::ShiftedB128Slice(s) => {
                let len = s.len();
                let next_idx = if index + 1 == len { 0 } else { index + 1 };

                F::promote_flat(s[next_idx])
            }
            Self::ShiftedPackedBitB8 { data, bit_idx } => {
                let len = data.len();
                let next_idx = if index + 1 == len { 0 } else { index + 1 };
                let bit = data[next_idx].tower_bit(*bit_idx);

                if bit == 1 {
                    Flat::from_raw(F::ONE)
                } else {
                    Flat::from_raw(F::ZERO)
                }
            }
            Self::ShiftedPackedBitB16 { data, bit_idx } => {
                let len = data.len();
                let next_idx = if index + 1 == len { 0 } else { index + 1 };
                let bit = data[next_idx].tower_bit(*bit_idx);

                if bit == 1 {
                    Flat::from_raw(F::ONE)
                } else {
                    Flat::from_raw(F::ZERO)
                }
            }
            Self::ShiftedPackedBitB32 { data, bit_idx } => {
                let len = data.len();
                let next_idx = if index + 1 == len { 0 } else { index + 1 };
                let bit = data[next_idx].tower_bit(*bit_idx);

                if bit == 1 {
                    Flat::from_raw(F::ONE)
                } else {
                    Flat::from_raw(F::ZERO)
                }
            }
            Self::ShiftedPackedBitB64 { data, bit_idx } => {
                let len = data.len();
                let next_idx = if index + 1 == len { 0 } else { index + 1 };
                let bit = data[next_idx].tower_bit(*bit_idx);

                if bit == 1 {
                    Flat::from_raw(F::ONE)
                } else {
                    Flat::from_raw(F::ZERO)
                }
            }
        }
    }

    /// Evaluate the MLE at an arbitrary point.
    /// `Eq` delegates to `TensorProduct`;
    /// every other variant expands MLE weights
    /// once and does a single `get_at` sweep.
    #[inline(always)]
    pub fn evaluate(&self, point: &[Flat<F>]) -> errors::Result<Flat<F>> {
        match self {
            Self::Eq(t) => Ok(t.evaluate_extension(point)?),
            _ => {
                let num_vars = point.len();
                let got_len = self.len();

                let Some(expected_len) = 1usize.checked_shl(num_vars as u32) else {
                    return Err(Error::DomainTooLarge { num_vars }.into());
                };

                if got_len != expected_len {
                    return Err(Error::DomainSizeMismatch {
                        expected_len,
                        got_len,
                    }
                    .into());
                }

                if num_vars == 0 {
                    return Ok(self.get_at(0));
                }

                let weights = Self::expand_mle_weights(point);
                let mut total = Flat::from_raw(F::ZERO);

                for (i, w) in weights.iter().enumerate() {
                    total += self.get_at(i) * *w;
                }

                Ok(total)
            }
        }
    }

    pub fn expand_mle_weights(r: &[Flat<F>]) -> Vec<Flat<F>> {
        let num_vars = r.len();
        let size = 1 << num_vars;

        let mut weights = vec![Flat::from_raw(F::ZERO); size];
        weights[0] = Flat::from_raw(F::ONE);

        for (i, &rk) in r.iter().enumerate() {
            let one_minus_rk = Flat::from_raw(F::ONE) - rk;
            let current_len = 1 << i;

            for i in 0..current_len {
                let w = weights[i];
                weights[i] = w * one_minus_rk;
                weights[current_len + i] = w * rk;
            }
        }

        weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hekate_math::{Bit, Block8, Block128, FlatPromote, HardwareField, TowerField};

    type F = Block128;

    #[test]
    fn get_at_dense_returns_correct_values() {
        let data: Vec<Flat<F>> = (0..4u128).map(|i| F::from(i * 10).to_hardware()).collect();
        let v = PolyVariant::<F>::Dense(&data);

        assert_eq!(v.get_at(0), F::from(0u128).to_hardware());
        assert_eq!(v.get_at(1), F::from(10u128).to_hardware());
        assert_eq!(v.get_at(2), F::from(20u128).to_hardware());
        assert_eq!(v.get_at(3), F::from(30u128).to_hardware());
    }

    #[test]
    fn get_at_bit_slice_promotes_to_field() {
        let bits = vec![
            Bit::from(0u32),
            Bit::from(1u32),
            Bit::from(1u32),
            Bit::from(0u32),
        ];
        let v = PolyVariant::<F>::BitSlice(&bits);

        assert_eq!(v.get_at(0), Flat::from_raw(F::ZERO));
        assert_eq!(v.get_at(1), Flat::from_raw(F::ONE));
        assert_eq!(v.get_at(2), Flat::from_raw(F::ONE));
        assert_eq!(v.get_at(3), Flat::from_raw(F::ZERO));
    }

    #[test]
    fn get_at_b8_slice_promotes() {
        let data = vec![
            Block8::from(0u8).to_hardware(),
            Block8::from(0xFFu8).to_hardware(),
        ];
        let v = PolyVariant::<F>::B8Slice(&data);

        assert_eq!(
            v.get_at(0),
            F::promote_flat(Block8::from(0u8).to_hardware())
        );
        assert_eq!(
            v.get_at(1),
            F::promote_flat(Block8::from(0xFFu8).to_hardware())
        );
    }

    #[test]
    fn len_matches_data_size() {
        let data = vec![Flat::from_raw(F::ZERO); 16];
        assert_eq!(PolyVariant::<F>::Dense(&data).len(), 16);

        let bits = vec![Bit::from(0u32); 8];
        assert_eq!(PolyVariant::<F>::BitSlice(&bits).len(), 8);

        let eq = TensorProduct::new(vec![Flat::from_raw(F::ONE); 5]);
        assert_eq!(PolyVariant::<F>::Eq(eq).len(), 32);

        let empty: Vec<Flat<F>> = vec![];
        assert!(PolyVariant::<F>::Dense(&empty).is_empty());
    }

    #[test]
    fn evaluate_constant_polynomial() {
        let num_vars = 3;
        let data = vec![F::from(42u128).to_hardware(); 1 << num_vars];
        let v = PolyVariant::<F>::Dense(&data);

        let point: Vec<Flat<F>> = vec![
            Flat::from_raw(F::from(1u128).to_hardware().into_raw()),
            Flat::from_raw(F::from(2u128).to_hardware().into_raw()),
            Flat::from_raw(F::from(3u128).to_hardware().into_raw()),
        ];

        let val = v.evaluate(&point).unwrap();
        assert_eq!(val.into_raw(), F::from(42u128).to_hardware().into_raw());
    }

    #[test]
    fn evaluate_linear_polynomial() {
        let data = vec![F::ZERO.to_hardware(), F::from(10u128).to_hardware()];
        let v = PolyVariant::<F>::Dense(&data);

        let point = vec![Flat::from_raw(F::from(2u128).to_hardware().into_raw())];
        let val = v.evaluate(&point).unwrap();
        assert_eq!(val.into_raw(), F::from(20u128).to_hardware().into_raw());
    }

    #[test]
    fn evaluate_single_row() {
        let data = vec![F::from(99u128).to_hardware()];
        let v = PolyVariant::<F>::Dense(&data);

        let val = v.evaluate(&[]).unwrap();
        assert_eq!(val.into_raw(), F::from(99u128).to_hardware().into_raw());
    }

    #[test]
    fn evaluate_domain_mismatch_rejected() {
        let data = vec![F::ZERO.to_hardware(); 4];
        let v = PolyVariant::<F>::Dense(&data);

        let point = vec![Flat::from_raw(F::ONE); 3];
        assert!(v.evaluate(&point).is_err());
    }

    #[test]
    fn evaluate_eq_polynomial() {
        let r = vec![Flat::from_raw(F::ONE), Flat::from_raw(F::ZERO)];
        let eq = PolyVariant::<F>::Eq(TensorProduct::new(r.clone()));

        let val = eq.evaluate(&r).unwrap();
        assert_eq!(val.into_raw(), F::ONE.to_hardware().into_raw());
    }

    #[test]
    fn expand_mle_weights_single_var() {
        let r = vec![Flat::from_raw(F::from(7u128).to_hardware().into_raw())];
        let w = PolyVariant::<F>::expand_mle_weights(&r);

        assert_eq!(w.len(), 2);
        let r0 = r[0];
        let one = Flat::from_raw(F::ONE);
        assert_eq!(w[0], one - r0);
        assert_eq!(w[1], r0);
    }

    #[test]
    fn expand_mle_weights_zero_vars() {
        let w = PolyVariant::<F>::expand_mle_weights(&[]);
        assert_eq!(w.len(), 1);
        assert_eq!(w[0], Flat::from_raw(F::ONE));
    }

    #[test]
    fn shifted_get_at_wraps_cyclically() {
        let data: Vec<Flat<F>> = (1..=4u128).map(|i| F::from(i).to_hardware()).collect();
        let v = PolyVariant::<F>::Shifted(&data);

        assert_eq!(v.get_at(0), F::from(2u128).to_hardware());
        assert_eq!(v.get_at(1), F::from(3u128).to_hardware());
        assert_eq!(v.get_at(2), F::from(4u128).to_hardware());
        assert_eq!(v.get_at(3), F::from(1u128).to_hardware());
    }

    #[test]
    fn composite_selector_sums_columns() {
        let a = vec![
            Bit::from(1u32),
            Bit::from(0u32),
            Bit::from(1u32),
            Bit::from(0u32),
        ];
        let b = vec![
            Bit::from(0u32),
            Bit::from(1u32),
            Bit::from(1u32),
            Bit::from(0u32),
        ];
        let v = PolyVariant::<F>::CompositeSelector(vec![&a, &b]);

        assert_eq!(v.get_at(0), Flat::from_raw(F::ONE));
        assert_eq!(v.get_at(1), Flat::from_raw(F::ONE));

        let two = F::ONE + F::ONE;
        assert_eq!(v.get_at(2), Flat::from_raw(two));
        assert_eq!(v.get_at(3), Flat::from_raw(F::ZERO));
    }
}
