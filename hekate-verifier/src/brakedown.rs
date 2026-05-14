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

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::marker::PhantomData;
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::proofs::{BrakedownCommitment, BrakedownProof};
use hekate_core::utils::compute_split_vars;
use hekate_crypto::merkle::MerkleTree;
use hekate_crypto::transcript::Transcript;
use hekate_crypto::Hasher;
use hekate_math::{Block8, HardwareField, PackableField, TowerField};
use tracing::instrument;

pub type OpenedRows<'a> = &'a [Vec<u8>];

#[derive(Clone, Debug)]
pub struct BrakedownVerifier<F, H: Hasher> {
    _marker: PhantomData<(F, H)>,
}

impl<F, H: Hasher> BrakedownVerifier<F, H>
where
    F: HardwareField + PackableField + From<Block8> + From<u128>,
{
    /// Verifies the Brakedown LDT
    /// opening for random columns.
    ///
    /// # Architecture: LDT Merkle Verification
    ///
    /// The Verifier does not have the original 2D grid.
    /// It only has the Merkle `ROOT`. For each randomly
    /// selected column, the Prover provides the raw column
    /// data and a Merkle path. The Verifier hashes the
    /// column and recomputes the path up to the root
    /// to ensure data integrity.
    ///
    /// ```text
    ///                  [ ROOT (Merkle Root) ]  <-- Known to Verifier
    ///                          /      \
    ///                      Hash_L    Hash_R    <-- Recomputed using Merkle Path
    ///                      /   \      /   \
    ///                    ...   ...  ...   ...
    ///                   /                  \
    ///                 Hash_i                ...
    ///                  |
    /// Opened Column: [ C_i ] <-- Prover provides this column + Path
    /// ```
    #[instrument(skip_all, name = "Brakedown::verify")]
    pub fn verify<'a>(
        commitment: &BrakedownCommitment,
        proof: &'a BrakedownProof<F>,
        transcript: &mut Transcript<H>,
        config: &Config,
    ) -> errors::Result<OpenedRows<'a>> {
        let num_rows = commitment.num_rows;
        let num_vars = num_rows.trailing_zeros() as usize;

        let split_vars = compute_split_vars(num_vars, config.num_queries);
        let grid_cols = 1 << split_vars;
        let encoded_width = grid_cols + config.ldt_blinding_factor;
        let num_queries = config.num_queries;

        let mut random_indices = Vec::with_capacity(num_queries);
        for _ in 0..num_queries {
            let bytes = transcript.challenge_field::<F>(b"idx_query")?.to_bytes();

            let mut rng_val: u64 = 0;
            for (k, &b) in bytes.iter().take(8).enumerate() {
                rng_val |= (b as u64) << (8 * k);
            }

            random_indices.push((rng_val % (encoded_width as u64)) as usize);
        }

        let mut opened_consumed = 0;
        for (ldt_proof_idx, &col_idx) in random_indices.iter().enumerate() {
            // Verify Column Commitment
            // One leaf contains both Base
            // and Shifted encodings.
            verify_single_column::<F, H>(
                &commitment.root,
                col_idx,
                proof,
                opened_consumed,
                ldt_proof_idx,
            )?;

            opened_consumed += 1;
        }

        if opened_consumed != proof.opened_columns.len() {
            return Err(errors::Error::Protocol {
                protocol: "brakedown",
                message: "unused opened columns detected",
            });
        }

        Ok(&proof.opened_columns)
    }
}

/// Helper:
/// Verify a single 2D column
/// using individual merkle path.
fn verify_single_column<F: TowerField, H: Hasher>(
    root: &[u8; 32],
    col_idx: usize,
    proof: &BrakedownProof<F>,
    opened_idx: usize,
    proof_idx: usize,
) -> errors::Result<()> {
    if opened_idx >= proof.opened_columns.len() {
        return Err(errors::Error::Protocol {
            protocol: "brakedown",
            message: "opened columns index out of bounds",
        });
    }
    if proof_idx >= proof.ldt_proofs.len() {
        return Err(errors::Error::Protocol {
            protocol: "brakedown",
            message: "ldt proofs index out of bounds",
        });
    }

    let code_bytes = &proof.opened_columns[opened_idx];
    let merkle_path = &proof.ldt_proofs[proof_idx];

    let leaf_hash = hash_leaf::<H>(code_bytes);

    if !MerkleTree::<F, H>::verify(root, leaf_hash, col_idx, merkle_path) {
        return Err(errors::Error::Protocol {
            protocol: "brakedown",
            message: "merkle proof verification failed",
        });
    }

    Ok(())
}

fn hash_leaf<H: Hasher>(code: &[u8]) -> [u8; 32] {
    let mut h = H::new();
    h.update(&[0u8]); // Domain separator
    h.update(code);

    h.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use hekate_crypto::DefaultHasher;
    use hekate_math::{Block128, CanonicalSerialize};

    type F = Block128;
    type H = DefaultHasher;

    #[test]
    fn brakedown_verify_valid_proof() {
        let config = Config {
            num_queries: 4,
            ldt_blinding_factor: 2, // Enable ZK noise for test
            ..Config::default()
        };

        let num_rows = 16;
        let num_vars = 4; // 2^4 = 16
        let num_cols = 1;
        let field_size = 16; // Block128 size

        let split_vars = compute_split_vars(num_vars, config.num_queries);

        let grid_cols = 1 << split_vars;
        let grid_rows = 1 << (num_vars - split_vars);
        let encoded_width = grid_cols + config.ldt_blinding_factor;

        // 1. Setup Mock Data (Leaves)
        // Verification expects:
        // Leaf = Hash(0 || code_bytes)
        let mut leaves = vec![[0u8; 32]; encoded_width];
        for (i, leaf) in leaves.iter_mut().enumerate() {
            let mut code_bytes = vec![0u8; field_size * grid_rows];

            // Fill mock column data
            for r in 0..grid_rows {
                let idx = r * encoded_width + i;
                code_bytes[r * field_size..(r * field_size) + 8]
                    .copy_from_slice(&idx.to_le_bytes());
            }

            let mut h = H::new();
            h.update(&[0u8]); // Domain separator
            h.update(&code_bytes);

            *leaf = h.finalize();
        }

        let tree = MerkleTree::<F, H>::new(&leaves);
        let root = tree.root();
        let commitment = BrakedownCommitment {
            root,
            num_rows,
            num_cols,
        };

        // 2. Simulate Transcript
        let num_queries = config.num_queries;
        let mut prover_transcript = Transcript::<H>::new(b"test_brakedown");
        let mut query_indices = Vec::new();

        for _ in 0..num_queries {
            let bytes = prover_transcript
                .challenge_field::<F>(b"idx_query")
                .unwrap()
                .to_bytes();

            let mut rng_val: u64 = 0;
            for (k, &b) in bytes.iter().take(8).enumerate() {
                rng_val |= (b as u64) << (8 * k);
            }

            query_indices.push((rng_val % (encoded_width as u64)) as usize);
        }

        // 3. Construct Proof
        let mut merkle_proofs = Vec::new();
        let mut opened_columns = Vec::new();

        // One leaf contains both Base and Shifted encodings.
        // We only push 1 column per query.
        for &col_idx in &query_indices {
            merkle_proofs.push(tree.prove(col_idx).unwrap());

            let mut code_col = Vec::new();
            for r in 0..grid_rows {
                let idx = r * encoded_width + col_idx;

                let mut cd = vec![0u8; field_size];
                cd[0..8].copy_from_slice(&idx.to_le_bytes());

                code_col.extend_from_slice(&cd);
            }

            opened_columns.push(code_col);
        }

        let proof = BrakedownProof::new(merkle_proofs, opened_columns);

        // 4. Verify
        let mut verifier_transcript = Transcript::<H>::new(b"test_brakedown");
        let result = BrakedownVerifier::<F, H>::verify(
            &commitment,
            &proof,
            &mut verifier_transcript,
            &config,
        );

        assert!(result.is_ok(), "Valid Brakedown proof should verify");

        // Ensure we got exactly 1 column per query back
        let cols = result.unwrap();
        assert_eq!(cols.len(), config.num_queries);
    }

    #[test]
    fn brakedown_verify_tampered_data() {
        let config = Config {
            num_queries: 2,
            ..Config::default()
        };
        let num_rows = 16;
        let num_vars = 4;

        let split_vars = compute_split_vars(num_vars, config.num_queries);
        let grid_cols = 1 << split_vars;
        let encoded_width = grid_cols + config.ldt_blinding_factor;

        // Minimal fake tree mapped to encoded_width
        let leaves = vec![[0u8; 32]; encoded_width];
        let tree = MerkleTree::<F, H>::new(&leaves);

        // Mock commitment
        let commitment = BrakedownCommitment {
            root: tree.root(),
            num_rows,
            num_cols: 1,
        };

        // Mock Proof with GARBAGE data (1 column per query)
        let proof = BrakedownProof::new(
            vec![vec![]; config.num_queries], // Empty proofs
            vec![vec![4, 5, 6]; config.num_queries],
        );

        let mut transcript = Transcript::<H>::new(b"test");
        let result =
            BrakedownVerifier::<F, H>::verify(&commitment, &proof, &mut transcript, &config);

        assert!(result.is_err(), "Tampered/Invalid proof should fail");
    }
}
