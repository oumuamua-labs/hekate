// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::marker::PhantomData;
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::proofs::{BrakedownCommitment, BrakedownProof};
use hekate_crypto::Hasher;
use hekate_crypto::merkle::MerkleTree;
use hekate_crypto::transcript::Transcript;
use hekate_math::{Block8, HardwareField, PackableField};
use tracing::instrument;

pub type OpenedRows<'a> = &'a [Vec<u8>];

/// One eval argument's Fiat-Shamir column queries;
/// `slot_map[q]` indexes `distinct` for the q-th query.
pub struct QuerySet {
    pub indices: Vec<usize>,
    pub distinct: Vec<usize>,
    pub slot_map: Vec<usize>,
    pub encoded_width: usize,
}

#[derive(Clone, Debug)]
pub struct BrakedownVerifier<F, H: Hasher> {
    _marker: PhantomData<(F, H)>,
}

impl<F, H: Hasher> BrakedownVerifier<F, H>
where
    F: HardwareField + PackableField + From<Block8> + From<u128>,
{
    /// Draws `num_queries` columns of the codeword domain;
    /// every tree of the table opens at this one set.
    #[instrument(skip_all, name = "Brakedown::draw_queries")]
    pub fn draw_queries(
        transcript: &mut Transcript<H>,
        config: &Config,
        split_vars: usize,
    ) -> errors::Result<QuerySet> {
        let grid_cols = 1 << split_vars;
        let encoded_width = config.encoded_width(grid_cols);
        let num_queries = config.num_queries;

        let mut indices = Vec::with_capacity(num_queries);
        for _ in 0..num_queries {
            let bytes = transcript.challenge_field::<F>(b"idx_query")?.to_bytes();

            let mut rng_val: u64 = 0;
            for (k, &b) in bytes.iter().take(8).enumerate() {
                rng_val |= (b as u64) << (8 * k);
            }

            indices.push((rng_val % (encoded_width as u64)) as usize);
        }

        // One opened column and one octopus leaf
        // per distinct queried index, ascending.
        let mut distinct = indices.clone();
        distinct.sort_unstable();
        distinct.dedup();

        let mut slot_map = Vec::with_capacity(num_queries);
        for col_idx in &indices {
            let slot = distinct
                .binary_search(col_idx)
                .map_err(|_| errors::Error::Protocol {
                    protocol: "brakedown",
                    message: "query index missing from distinct set",
                })?;

            slot_map.push(slot);
        }

        Ok(QuerySet {
            indices,
            distinct,
            slot_map,
            encoded_width,
        })
    }

    /// Verifies one tree's opening at the drawn queries:
    /// hashes each distinct opened column to a leaf and
    /// replays the octopus multiproof against the root.
    #[instrument(skip_all, name = "Brakedown::verify_opening")]
    pub fn verify_opening<'a>(
        commitment: &BrakedownCommitment,
        proof: &'a BrakedownProof<F>,
        queries: &QuerySet,
    ) -> errors::Result<OpenedRows<'a>> {
        let distinct = &queries.distinct;

        if proof.opened_columns.len() != distinct.len() {
            return Err(errors::Error::Protocol {
                protocol: "brakedown",
                message: "opened column count does not match distinct query count",
            });
        }

        #[cfg(feature = "parallel")]
        let leaves: Vec<(usize, [u8; 32])> = {
            use rayon::prelude::*;

            distinct
                .par_iter()
                .zip(proof.opened_columns.par_iter())
                .map(|(&col_idx, col)| (col_idx, hash_leaf::<H>(col)))
                .collect()
        };

        #[cfg(not(feature = "parallel"))]
        let leaves: Vec<(usize, [u8; 32])> = distinct
            .iter()
            .zip(proof.opened_columns.iter())
            .map(|(&col_idx, col)| (col_idx, hash_leaf::<H>(col)))
            .collect();

        let padded_leaves = queries.encoded_width.next_power_of_two();

        if !MerkleTree::<F, H>::verify_batch(
            &commitment.root,
            padded_leaves,
            &leaves,
            &proof.batch_path,
        ) {
            return Err(errors::Error::Protocol {
                protocol: "brakedown",
                message: "batch merkle proof verification failed",
            });
        }

        Ok(&proof.opened_columns)
    }
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
            ldt_support_size: 2, // Enable ZK noise for test
            ..Config::default()
        };

        let num_rows = 16;
        let num_vars = 4; // 2^4 = 16
        let num_cols = 1;
        let field_size = 16; // Block128 size

        let split_vars = hekate_core::utils::compute_split_vars(
            num_vars,
            config.num_queries,
            config.ldt_support_size,
            128,
        );

        let grid_cols = 1 << split_vars;
        let grid_rows = 1 << (num_vars - split_vars);
        let encoded_width = config.encoded_width(grid_cols);

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

        // 3. Construct Proof: one opened column per
        // distinct query, plus one octopus multiproof.
        let mut distinct = query_indices.clone();
        distinct.sort_unstable();
        distinct.dedup();

        let mut opened_columns = Vec::new();
        for &col_idx in &distinct {
            let mut code_col = Vec::new();
            for r in 0..grid_rows {
                let idx = r * encoded_width + col_idx;

                let mut cd = vec![0u8; field_size];
                cd[0..8].copy_from_slice(&idx.to_le_bytes());

                code_col.extend_from_slice(&cd);
            }

            opened_columns.push(code_col);
        }

        let batch_path = tree.prove_batch(&distinct).unwrap();
        let proof = BrakedownProof::new(opened_columns, batch_path);

        // 4. Verify
        let mut verifier_transcript = Transcript::<H>::new(b"test_brakedown");
        let queries =
            BrakedownVerifier::<F, H>::draw_queries(&mut verifier_transcript, &config, split_vars)
                .unwrap();
        let result = BrakedownVerifier::<F, H>::verify_opening(&commitment, &proof, &queries);

        assert!(result.is_ok(), "Valid Brakedown proof should verify");
        assert_eq!(result.unwrap().len(), distinct.len());
        assert_eq!(queries.slot_map.len(), config.num_queries);
    }

    #[test]
    fn brakedown_verify_tampered_data() {
        let config = Config {
            num_queries: 2,
            ..Config::default()
        };
        let num_rows = 16;
        let num_vars = 4;

        let split_vars = hekate_core::utils::compute_split_vars(
            num_vars,
            config.num_queries,
            config.ldt_support_size,
            128,
        );
        let grid_cols = 1 << split_vars;
        let encoded_width = config.encoded_width(grid_cols);

        // Minimal fake tree mapped to encoded_width
        let leaves = vec![[0u8; 32]; encoded_width];
        let tree = MerkleTree::<F, H>::new(&leaves);

        let commitment = BrakedownCommitment {
            root: tree.root(),
            num_rows,
            num_cols: 1,
        };

        // Replay the query draws to learn the distinct
        // count, then submit garbage columns with an
        // empty octopus path: the batch walk starves.
        let mut prover_transcript = Transcript::<H>::new(b"test");
        let mut distinct = Vec::new();

        for _ in 0..config.num_queries {
            let bytes = prover_transcript
                .challenge_field::<F>(b"idx_query")
                .unwrap()
                .to_bytes();

            let mut rng_val: u64 = 0;
            for (k, &b) in bytes.iter().take(8).enumerate() {
                rng_val |= (b as u64) << (8 * k);
            }

            distinct.push((rng_val % (encoded_width as u64)) as usize);
        }

        distinct.sort_unstable();
        distinct.dedup();

        let proof = BrakedownProof::new(vec![vec![4u8, 5, 6]; distinct.len()], Vec::new());

        let mut transcript = Transcript::<H>::new(b"test");
        let queries =
            BrakedownVerifier::<F, H>::draw_queries(&mut transcript, &config, split_vars).unwrap();
        let result = BrakedownVerifier::<F, H>::verify_opening(&commitment, &proof, &queries);

        assert!(result.is_err(), "Tampered/Invalid proof should fail");
    }
}
