// SPDX-License-Identifier: Apache-2.0
// This file is part of the hekate project.
// Copyright (C) 2026 Andrei Kochergin <andrei@oumuamua.dev>
// Copyright (C) 2026 Oumuamua Labs <info@oumuamua.dev>.
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

use aes::Aes256;
use aes::cipher::{BlockCipherEncrypt, KeyInit};
use alloc::vec::Vec;
use core::convert::Infallible;
use hekate_math::matrix::ByteSparseMatrix;
use rand::{RngExt, SeedableRng, TryRng};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

// Shared by the parallel and sequential paths,
// generation is bit-identical across feature configs.
const GEN_CHUNK_ROWS: usize = 256;

const AES_BLOCK: usize = 16;

// 8 blocks per batch saturate the AES-NI / ARMv8-CE
// pipeline via instruction-level parallelism.
const AES_BATCH: usize = 8;
const AES_BUF_SIZE: usize = AES_BATCH * AES_BLOCK;

struct AesCtrPrg {
    cipher: Aes256,
    nonce: u64,
    counter: u64,
    buffer: [u8; AES_BUF_SIZE],
    buf_pos: usize,
}

impl AesCtrPrg {
    fn set_stream(&mut self, stream_id: u64) {
        self.nonce = stream_id;
        self.counter = 0;
        self.buf_pos = AES_BUF_SIZE;
    }

    fn refill(&mut self) {
        let nonce_high = (self.nonce as u128) << 64;

        let mut blocks: [aes::Block; AES_BATCH] = Default::default();
        for (i, block) in blocks.iter_mut().enumerate() {
            let val = (self.counter + i as u64) as u128 | nonce_high;
            *block = val.to_le_bytes().into();
        }

        self.cipher.encrypt_blocks(&mut blocks);

        for (i, block) in blocks.iter().enumerate() {
            self.buffer[i * AES_BLOCK..(i + 1) * AES_BLOCK].copy_from_slice(block.as_slice());
        }

        self.counter += AES_BATCH as u64;
        self.buf_pos = 0;
    }
}

impl SeedableRng for AesCtrPrg {
    type Seed = [u8; 32];

    fn from_seed(seed: [u8; 32]) -> Self {
        Self {
            cipher: Aes256::new(&seed.into()),
            nonce: 0,
            counter: 0,
            buffer: [0u8; AES_BUF_SIZE],
            buf_pos: AES_BUF_SIZE,
        }
    }
}

impl TryRng for AesCtrPrg {
    type Error = Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        if self.buf_pos + 4 > AES_BUF_SIZE {
            self.refill();
        }

        let p = self.buf_pos;
        let val = u32::from_le_bytes(core::array::from_fn(|i| self.buffer[p + i]));

        self.buf_pos = p + 4;

        Ok(val)
    }

    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        if self.buf_pos + 8 > AES_BUF_SIZE {
            self.refill();
        }

        let p = self.buf_pos;
        let val = u64::from_le_bytes(core::array::from_fn(|i| self.buffer[p + i]));

        self.buf_pos = p + 8;

        Ok(val)
    }

    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Infallible> {
        let mut written = 0;
        while written < dst.len() {
            if self.buf_pos >= AES_BUF_SIZE {
                self.refill();
            }

            let available = AES_BUF_SIZE - self.buf_pos;
            let copy_len = available.min(dst.len() - written);

            dst[written..written + copy_len]
                .copy_from_slice(&self.buffer[self.buf_pos..self.buf_pos + copy_len]);

            self.buf_pos += copy_len;
            written += copy_len;
        }

        Ok(())
    }
}

/// Sample a degree-regular binary expander matrix
/// from `seed` for Brakedown-style encoding.
///
/// Deterministic in `seed`: prover and verifier
/// derive the identical matrix. Output is bit-identical
/// across `parallel` and sequential builds.
///
/// # Panics
/// Panics on `cols == 0`, `degree > cols`, `degree > 256`,
/// `cols > u32::MAX`, or `rows * degree` overflow.
pub fn generate_expander_matrix(
    rows: usize,
    cols: usize,
    degree: usize,
    seed: [u8; 32],
) -> ByteSparseMatrix {
    const MAX_DEGREE: usize = 256;
    assert!(
        degree <= MAX_DEGREE,
        "Expander degree exceeds stack buffer size"
    );

    assert!(
        cols > 0,
        "Matrix generation requires cols > 0 (division by zero in RNG)"
    );
    assert!(
        degree <= cols,
        "Expander degree cannot exceed cols (would cause infinite loop in generation)"
    );
    assert!(
        cols <= u32::MAX as usize,
        "cols exceeds u32 column-index space"
    );

    let total_elems = rows
        .checked_mul(degree)
        .expect("Matrix size overflow: rows * degree exceeds usize::MAX");

    if total_elems == 0 {
        return ByteSparseMatrix::new(rows, cols, degree, Vec::new(), Vec::new());
    }

    let mut weights: Vec<u8> = Vec::with_capacity(total_elems);
    let mut col_indices: Vec<u32> = Vec::with_capacity(total_elems);

    let weights_uninit = weights.spare_capacity_mut();
    let col_indices_uninit = col_indices.spare_capacity_mut();

    assert!(weights_uninit.len() >= total_elems);
    assert!(col_indices_uninit.len() >= total_elems);

    #[cfg(feature = "parallel")]
    {
        let rows_per_chunk = GEN_CHUNK_ROWS.min(rows.max(1));
        let aligned_chunk_len = rows_per_chunk * degree;

        weights_uninit[..total_elems]
            .par_chunks_mut(aligned_chunk_len)
            .zip(col_indices_uninit[..total_elems].par_chunks_mut(aligned_chunk_len))
            .enumerate()
            .for_each(|(chunk_id, (w_chunk, col_chunk))| {
                let rows_in_this_chunk = w_chunk.len() / degree;

                let mut rng = AesCtrPrg::from_seed(seed);
                rng.set_stream(chunk_id as u64);

                let mut used_cols = [0u32; MAX_DEGREE];
                for r in 0..rows_in_this_chunk {
                    let row_offset = r * degree;

                    for d in 0..degree {
                        w_chunk[row_offset + d].write(1u8);

                        let mut col_idx;
                        loop {
                            col_idx = rng.random_range(0..cols as u32);

                            // Expander collapse: in characteristic 2 a
                            // duplicate column makes X ^ X = 0, erasing
                            // the row's degree and breaking PCS soundness.
                            if !used_cols[..d].contains(&col_idx) {
                                break;
                            }
                        }

                        used_cols[d] = col_idx;
                        col_chunk[row_offset + d].write(col_idx);
                    }
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        let rows_per_chunk = GEN_CHUNK_ROWS.min(rows.max(1));
        let aligned_chunk_len = rows_per_chunk * degree;
        let num_chunks = total_elems.div_ceil(aligned_chunk_len);

        let mut used_cols = [0u32; MAX_DEGREE];
        for chunk_id in 0..num_chunks {
            let mut rng = AesCtrPrg::from_seed(seed);
            rng.set_stream(chunk_id as u64);

            let elem_start = chunk_id * aligned_chunk_len;
            let elem_end = (elem_start + aligned_chunk_len).min(total_elems);
            let rows_in_this_chunk = (elem_end - elem_start) / degree;

            for r in 0..rows_in_this_chunk {
                let row_offset = elem_start + r * degree;

                for d in 0..degree {
                    weights_uninit[row_offset + d].write(1u8);

                    let mut col_idx;
                    loop {
                        col_idx = rng.random_range(0..cols as u32);
                        if !used_cols[..d].contains(&col_idx) {
                            break;
                        }
                    }

                    used_cols[d] = col_idx;
                    col_indices_uninit[row_offset + d].write(col_idx);
                }
            }
        }
    }

    // SAFETY:
    // weights_uninit[..total_elems] and
    // col_indices_uninit[..total_elems]
    // were fully initialized above.
    unsafe {
        weights.set_len(total_elems);
        col_indices.set_len(total_elems);
    }

    ByteSparseMatrix::new(rows, cols, degree, weights, col_indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use hekate_math::{Block128, Flat, HardwareField};
    use proptest::prelude::*;

    #[test]
    #[should_panic(expected = "cols > 0")]
    fn safety_rejects_zero_cols() {
        generate_expander_matrix(10, 0, 5, [1u8; 32]);
    }

    #[test]
    fn accepts_valid_dimensions() {
        let m = generate_expander_matrix(10, 10, 5, [1u8; 32]);
        assert_eq!(m.rows(), 10);
        assert_eq!(m.cols(), 10);
        assert_eq!(m.degree(), 5);
        assert_eq!(m.weights().len(), 50);
    }

    #[test]
    fn accepts_zero_rows_or_degree() {
        let m1 = generate_expander_matrix(0, 10, 5, [1u8; 32]);
        assert_eq!(m1.weights().len(), 0);

        let m2 = generate_expander_matrix(10, 10, 0, [1u8; 32]);
        assert_eq!(m2.weights().len(), 0);
    }

    #[test]
    fn expander_properties_sanity_check() {
        let rows = 4096;
        let cols = 4096;
        let degree = 16;
        let seed = [42u8; 32];

        let matrix = generate_expander_matrix(rows, cols, degree, seed);

        let hamming_weight = |vec: &[Flat<Block128>]| -> usize {
            vec.iter()
                .filter(|&&x| x != Block128::from(0u128).to_hardware())
                .count()
        };

        // Weight-1 input must not vanish:
        // no column maps to empty.
        for i in 0..100 {
            let mut x = vec![Block128::from(0u128).to_hardware(); cols];
            x[i] = Block128::from(1u128).to_hardware();

            let y = matrix.spmv(x.as_slice());
            let w = hamming_weight(&y);

            assert!(w > 0, "Column {} is empty! Information loss", i);
        }

        // Weight-2 input:
        // low neighbour overlap, output ~2*degree.
        let mut rng = AesCtrPrg::from_seed([1u8; 32]);
        let mut total_weight = 0;

        let trials = 100;
        for _ in 0..trials {
            let mut x = vec![Block128::from(0u128).to_hardware(); cols];

            let idx1 = rng.random_range(0..cols);
            let idx2 = (idx1 + 1) % cols;

            x[idx1] = Block128::from(1u128).to_hardware();
            x[idx2] = Block128::from(1u128).to_hardware();

            let y = matrix.spmv(x.as_slice());
            total_weight += hamming_weight(&y);
        }

        let avg_weight = total_weight as f64 / trials as f64;
        let expected_max = (degree * 2) as f64;

        assert!(
            avg_weight > (expected_max * 0.8),
            "Too many collisions! Poor expansion property. Avg: {}",
            avg_weight
        );

        // Avalanche: input weight 10 -> output weight close to 160.
        let input_w = 10;
        let mut x = vec![Block128::from(0u128).to_hardware(); cols];

        for val in x.iter_mut().take(input_w) {
            *val = Block128::from(1u128).to_hardware();
        }

        let y = matrix.spmv(x.as_slice());
        let w_out = hamming_weight(&y);

        assert!(
            w_out > (input_w * degree * 8 / 10),
            "Weight-10 vector collapsed too much! Weight: {}",
            w_out
        );
    }

    #[test]
    fn check_determinism() {
        let seed = [42u8; 32];
        let rows = 1024;
        let cols = 1024;
        let degree = 16;

        let matrix1 = generate_expander_matrix(rows, cols, degree, seed);
        let matrix2 = generate_expander_matrix(rows, cols, degree, seed);

        assert_eq!(
            matrix1.weights(),
            matrix2.weights(),
            "Matrix weights must be deterministic for the same seed"
        );
        assert_eq!(
            matrix1.col_indices(),
            matrix2.col_indices(),
            "Matrix column indices must be deterministic for the same seed"
        );

        #[cfg(feature = "parallel")]
        {
            use rayon::ThreadPoolBuilder;

            let matrix_1thread = ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .unwrap()
                .install(|| generate_expander_matrix(rows, cols, degree, seed));

            let matrix_8threads = ThreadPoolBuilder::new()
                .num_threads(8)
                .build()
                .unwrap()
                .install(|| generate_expander_matrix(rows, cols, degree, seed));

            assert_eq!(
                matrix_1thread.weights(),
                matrix_8threads.weights(),
                "Matrix must be identical regardless of thread count"
            );
            assert_eq!(
                matrix_1thread.col_indices(),
                matrix_8threads.col_indices(),
                "Matrix indices must be identical regardless of thread count"
            );
        }
    }

    #[test]
    fn security_prevent_expander_collapse() {
        // Force degree == cols. In GF(2^k), duplicate indices
        // cause X ^ X = 0, destroying PCS soundness.
        let rows = 1000;
        let cols = 32;
        let degree = 32;
        let seed = [99u8; 32];

        let matrix = generate_expander_matrix(rows, cols, degree, seed);

        for r in 0..rows {
            let row_offset = r * degree;

            let mut row_indices: Vec<u32> =
                matrix.col_indices()[row_offset..row_offset + degree].to_vec();
            row_indices.sort_unstable();

            for d in 0..degree - 1 {
                assert_ne!(
                    row_indices[d],
                    row_indices[d + 1],
                    "Expander Collapse detected in row {}! Duplicate column index {}. \
                     The rejection sampling loop has been compromised.",
                    r,
                    row_indices[d]
                );
            }
        }
    }

    #[test]
    fn cross_feature_determinism_golden() {
        let matrix = generate_expander_matrix(1024, 512, 16, [42u8; 32]);

        #[rustfmt::skip]
        const EXPECTED: [u32; 64] = [
            442, 352, 465,  69, 176, 472, 322, 109,
            349, 216,  74,  35, 206,  50,   7, 443,
            349, 214,  30, 332,  66, 316, 297, 415,
            325,  88, 484, 345,   5, 224, 106, 326,
            454, 345, 295, 443, 267, 264,  91, 333,
            163, 359, 262,  49, 112, 499, 219,  67,
            420, 106, 415,  54, 437, 123, 366, 284,
            503, 249,  26, 353,  90,  29, 311, 111,
        ];

        assert_eq!(&matrix.col_indices()[..64], &EXPECTED);
    }

    // Counter block = (nonce << 64 | counter).to_le_bytes()
    #[test]
    fn aes_ctr_prg_golden() {
        #[rustfmt::skip]
        const EXPECTED: [u8; 128] = [
            // block 0: AES-256([0;32], counter=0)
            0xdc, 0x95, 0xc0, 0x78, 0xa2, 0x40, 0x89, 0x89,
            0xad, 0x48, 0xa2, 0x14, 0x92, 0x84, 0x20, 0x87,
            // block 1: counter=1
            0x52, 0x75, 0xf3, 0xd8, 0x6b, 0x4f, 0xb8, 0x68,
            0x45, 0x93, 0x13, 0x3e, 0xbf, 0xa5, 0x3c, 0xd3,
            // block 2: counter=2
            0x77, 0x9b, 0x38, 0xd1, 0x5b, 0xff, 0xb6, 0x3d,
            0x8d, 0x60, 0x9d, 0x55, 0x1a, 0x5c, 0xc9, 0x8e,
            // block 3: counter=3
            0x39, 0xd6, 0xe9, 0xae, 0x76, 0xa9, 0xb2, 0xf3,
            0xfc, 0x46, 0x26, 0x80, 0xf7, 0x66, 0x72, 0x0e,
            // block 4: counter=4
            0x75, 0xd1, 0x1b, 0x0e, 0x3a, 0x68, 0xc4, 0x22,
            0x3d, 0x88, 0xdb, 0xf0, 0x17, 0x97, 0x7d, 0xd7,
            // block 5: counter=5
            0x84, 0x5c, 0x7d, 0x46, 0x90, 0xfa, 0x59, 0x4f,
            0x90, 0xe6, 0x7f, 0x7b, 0x52, 0x11, 0xa5, 0x1a,
            // block 6: counter=6
            0x6f, 0x87, 0x1f, 0x44, 0x5c, 0x18, 0xaf, 0xc2,
            0xf8, 0x93, 0x7a, 0xf8, 0x41, 0xfd, 0x2a, 0xd0,
            // block 7: counter=7
            0x8d, 0x3a, 0xe1, 0x50, 0x22, 0x15, 0x52, 0x33,
            0x4d, 0xdb, 0x29, 0xfe, 0x36, 0xa0, 0xb7, 0x24,
        ];

        let mut prg = AesCtrPrg::from_seed([0u8; 32]);
        let mut output = [0u8; 128];

        let _ = prg.try_fill_bytes(&mut output);

        assert_eq!(output, EXPECTED);
    }

    #[test]
    fn aes_ctr_prg_stream_isolation() {
        let seed = [0xabu8; 32];

        let mut prg0 = AesCtrPrg::from_seed(seed);
        prg0.set_stream(0);

        let mut out0 = [0u8; 64];
        let _ = prg0.try_fill_bytes(&mut out0);

        let mut prg1 = AesCtrPrg::from_seed(seed);
        prg1.set_stream(1);

        let mut out1 = [0u8; 64];
        let _ = prg1.try_fill_bytes(&mut out1);

        assert_ne!(
            out0, out1,
            "Different streams must produce different output"
        );

        let mut prg0_again = AesCtrPrg::from_seed(seed);
        prg0_again.set_stream(0);

        let mut out0_again = [0u8; 64];
        let _ = prg0_again.try_fill_bytes(&mut out0_again);

        assert_eq!(out0, out0_again, "Same stream must be deterministic");
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]
        #[test]
        fn expansion_proptest(
            seed in any::<[u8; 32]>(),
            random_col in 0..1024usize,
            val_raw in 1..255u128
        ) {
            let rows = 1024;
            let cols = 1024;
            let degree = 16;
            let matrix = generate_expander_matrix(rows, cols, degree, seed);

            let mut x = vec![Block128::from(0u128).to_hardware(); cols];
            x[random_col] = Block128::from(val_raw).to_hardware();

            let y = matrix.spmv(x.as_slice());
            let weight = y.iter().filter(|&&v|
                v != Block128::from(0u128).to_hardware()).count();

            let min_weight = degree / 6;
            prop_assert!(
                weight >= min_weight,
                "Column {} failed expansion: weight {}",
                random_col, weight,
            );
        }
    }
}
