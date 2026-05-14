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

use crate::{DefaultHasher, Hasher};
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use hekate_math::TowerField;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    LeafIndexOutOfBounds {
        leaf_index: usize,
        num_leaves: usize,
    },
    SubtreeUnaligned,
    SubtreeInternalIndexOutOfBounds,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LeafIndexOutOfBounds {
                leaf_index,
                num_leaves,
            } => write!(
                f,
                "Merkle leaf index out of bounds: leaf_index={leaf_index}, num_leaves={num_leaves}",
            ),
            Self::SubtreeUnaligned => {
                write!(f, "Merkle subtree range must be aligned to power of 2")
            }
            Self::SubtreeInternalIndexOutOfBounds => {
                write!(f, "Merkle internal node index out of bounds (logic error)")
            }
        }
    }
}

/// Binary Merkle tree over 32-byte leaves.
///
/// Internal node = `H(0x01 || left || right)`.
/// Leaves are expected to already be hashes;
/// callers serialize their field payloads via
/// `hash_leaf_row_blinded` / `hash_leaf_column_encoded`.
#[derive(Clone, Debug)]
pub struct MerkleTree<F: TowerField, H: Hasher = DefaultHasher> {
    nodes: Vec<MaybeUninit<[u8; 32]>>,
    num_leaves: usize,

    /// Guard for the `MaybeUninit` nodes:
    /// reading `root`/`prove`/subtree ops
    /// before `build_layers` runs is UB.
    built: bool,

    _marker: PhantomData<(F, H)>,
}

impl<F: TowerField, H: Hasher> MerkleTree<F, H> {
    /// Build a tree from pre-computed
    /// leaf hashes. Non-power-of-two
    /// inputs pad with zero leaves.
    pub fn new(leaves: &[[u8; 32]]) -> Self {
        let num_leaves = leaves.len();
        if num_leaves == 0 {
            return Self::empty();
        }

        let (mut tree, leaf_offset) = Self::allocate_tree(num_leaves);

        let leaf_layer = tree.leaves_mut(leaf_offset);

        #[cfg(feature = "parallel")]
        {
            leaf_layer
                .par_iter_mut()
                .with_min_len(256)
                .enumerate()
                .for_each(|(i, slot)| {
                    if i < leaves.len() {
                        slot.write(leaves[i]);
                    } else {
                        slot.write([0u8; 32]);
                    }
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for (i, slot) in leaf_layer.iter_mut().enumerate() {
                if i < leaves.len() {
                    slot.write(leaves[i]);
                } else {
                    slot.write([0u8; 32]);
                }
            }
        }

        tree.build_layers(leaf_offset);

        tree
    }

    pub fn num_leaves(&self) -> usize {
        self.num_leaves
    }

    /// Mutable view of the leaf layer for streaming
    /// writes. Slots are `MaybeUninit`, the caller
    /// must populate every slot before `build_layers`.
    pub fn leaves_mut(&mut self, leaf_offset: usize) -> &mut [MaybeUninit<[u8; 32]>] {
        &mut self.nodes[leaf_offset..leaf_offset + self.num_leaves]
    }

    pub fn root(&self) -> [u8; 32] {
        if self.nodes.is_empty() {
            return [0u8; 32];
        }

        // SAFETY:
        // `self.built` means every node
        // slot has been initialized.
        assert!(self.built, "MerkleTree::root called before build_layers");

        unsafe { self.nodes[0].assume_init() }
    }

    /// Sibling path from `leaf_index` up to the root.
    pub fn prove(&self, leaf_index: usize) -> Result<Vec<[u8; 32]>> {
        // SAFETY:
        // see `root`, requires `built`.
        assert!(
            self.nodes.is_empty() || self.built,
            "MerkleTree::prove called before build_layers"
        );

        if leaf_index >= self.num_leaves {
            return Err(Error::LeafIndexOutOfBounds {
                leaf_index,
                num_leaves: self.num_leaves,
            });
        }

        let depth = self.num_leaves.trailing_zeros() as usize;

        let mut proof = Vec::with_capacity(depth);
        let mut node_idx = (self.num_leaves - 1) + leaf_index;

        while node_idx > 0 {
            let sibling_idx = if !node_idx.is_multiple_of(2) {
                node_idx + 1
            } else {
                node_idx - 1
            };

            let sib = unsafe { self.nodes[sibling_idx].assume_init() };
            proof.push(sib);

            node_idx = (node_idx - 1) / 2;
        }

        Ok(proof)
    }

    /// Verify a sibling path against `root`.
    /// `leaf_hash` is already the 32-byte leaf digest.
    pub fn verify(
        root: &[u8; 32],
        leaf_hash: [u8; 32],
        mut leaf_index: usize,
        proof: &[[u8; 32]],
    ) -> bool {
        let mut current_hash = leaf_hash;
        for sibling in proof {
            let mut hasher = H::new();
            hasher.update(&[1u8]);

            if leaf_index.is_multiple_of(2) {
                hasher.update(&current_hash);
                hasher.update(sibling);
            } else {
                hasher.update(sibling);
                hasher.update(&current_hash);
            }

            current_hash = hasher.finalize();
            leaf_index /= 2;
        }

        &current_hash == root
    }

    // =================================
    // Helpers
    // =================================

    fn empty() -> Self {
        Self {
            nodes: vec![],
            num_leaves: 0,
            built: true,
            _marker: PhantomData,
        }
    }

    pub fn allocate_tree(num_leaves: usize) -> (Self, usize) {
        let pow2_leaves = if num_leaves.is_power_of_two() {
            num_leaves
        } else {
            num_leaves.next_power_of_two()
        };

        let num_nodes = 2 * pow2_leaves - 1;
        let leaf_offset = pow2_leaves - 1;

        // SAFETY:
        // elements are `MaybeUninit` and
        // `build_layers` writes every slot
        // before any read.
        let mut nodes: Vec<MaybeUninit<[u8; 32]>> = Vec::with_capacity(num_nodes);
        unsafe {
            nodes.set_len(num_nodes);
        }

        (
            Self {
                nodes,
                num_leaves: pow2_leaves,
                built: false,
                _marker: PhantomData,
            },
            leaf_offset,
        )
    }

    pub fn build_layers(&mut self, leaf_offset: usize) {
        let mut current_layer_size = self.num_leaves;
        let mut current_offset = leaf_offset;

        while current_offset > 0 {
            let parent_layer_size = current_layer_size / 2;
            let parent_offset = current_offset - parent_layer_size;

            let (upper, lower) = self.nodes.split_at_mut(current_offset);
            let parents = &mut upper[parent_offset..parent_offset + parent_layer_size];
            let children = &lower[0..current_layer_size];

            #[cfg(feature = "parallel")]
            {
                parents
                    .par_iter_mut()
                    .with_min_len(256)
                    .enumerate()
                    .for_each(|(i, parent)| {
                        let left = unsafe { children[2 * i].assume_init_ref() };
                        let right = unsafe { children[2 * i + 1].assume_init_ref() };

                        let mut h = H::new();
                        h.update(&[1u8]);
                        h.update(left);
                        h.update(right);

                        parent.write(h.finalize());
                    });
            }

            #[cfg(not(feature = "parallel"))]
            {
                for i in 0..parent_layer_size {
                    let left = unsafe { children[2 * i].assume_init_ref() };
                    let right = unsafe { children[2 * i + 1].assume_init_ref() };

                    let mut h = H::new();
                    h.update(&[1u8]);
                    h.update(left);
                    h.update(right);

                    parents[i].write(h.finalize());
                }
            }

            current_layer_size = parent_layer_size;
            current_offset = parent_offset;
        }

        self.built = true;
    }

    // =================================
    // Subtree proofs
    // =================================

    /// Returns the hash of the subtree root spanning
    /// leaves `[leaf_start_idx, leaf_start_idx + 2^height)`.
    /// Used to batch-commit TensorPCS columns
    /// without emitting one path per leaf.
    pub fn get_internal_root(
        &self,
        leaf_start_idx: usize,
        subtree_height: usize,
    ) -> Result<[u8; 32]> {
        // SAFETY:
        // see `root`, requires `built`.
        assert!(
            self.built,
            "MerkleTree::get_internal_root called before build_layers"
        );

        let num_leaves_in_subtree = 1 << subtree_height;

        if !leaf_start_idx.is_multiple_of(num_leaves_in_subtree) {
            return Err(Error::SubtreeUnaligned);
        }

        if leaf_start_idx + num_leaves_in_subtree > self.num_leaves {
            return Err(Error::LeafIndexOutOfBounds {
                leaf_index: leaf_start_idx + num_leaves_in_subtree,
                num_leaves: self.num_leaves,
            });
        }

        // Heap layout:
        // leaves start at `num_leaves - 1`;
        // walk up `subtree_height` levels
        // to the ancestor.
        let mut node_idx = (self.num_leaves - 1) + leaf_start_idx;
        for _ in 0..subtree_height {
            node_idx = (node_idx - 1) / 2;
        }

        if node_idx >= self.nodes.len() {
            return Err(Error::SubtreeInternalIndexOutOfBounds);
        }

        unsafe { Ok(self.nodes[node_idx].assume_init()) }
    }

    /// Sibling path proving that `subtree_root`
    /// sits at `subtree_height` inside the tree
    /// committed to by `root`.
    pub fn prove_subtree(
        &self,
        leaf_start_idx: usize,
        subtree_height: usize,
    ) -> Result<Vec<[u8; 32]>> {
        // SAFETY:
        // see `root`, requires `built`.
        assert!(
            self.built,
            "MerkleTree::prove_subtree called before build_layers"
        );

        let num_leaves_in_subtree = 1 << subtree_height;
        if !leaf_start_idx.is_multiple_of(num_leaves_in_subtree) {
            return Err(Error::SubtreeUnaligned);
        }

        let mut node_idx = (self.num_leaves - 1) + leaf_start_idx;
        for _ in 0..subtree_height {
            node_idx = (node_idx - 1) / 2;
        }

        let depth = (self.num_leaves.trailing_zeros() as usize) - subtree_height;
        let mut proof = Vec::with_capacity(depth);

        // Odd child index -> node is the left one,
        // sibling is +1.
        // Even child index -> node is the right one,
        // sibling is -1.
        while node_idx > 0 {
            let sibling_idx = if !node_idx.is_multiple_of(2) {
                node_idx + 1
            } else {
                node_idx - 1
            };

            let sib = unsafe { self.nodes[sibling_idx].assume_init() };
            proof.push(sib);

            node_idx = (node_idx - 1) / 2;
        }

        Ok(proof)
    }

    pub fn verify_subtree(
        root: &[u8; 32],
        subtree_root: [u8; 32],
        leaf_start_idx: usize,
        subtree_height: usize,
        proof: &[[u8; 32]],
    ) -> bool {
        // Logical index of the subtree in its layer,
        // e.g. leaves 256..512 with height 8 -> index 1.
        let mut node_logical_idx = leaf_start_idx >> subtree_height;
        let mut current_hash = subtree_root;

        for sibling in proof {
            let mut hasher = H::new();
            hasher.update(&[1u8]);

            if node_logical_idx.is_multiple_of(2) {
                hasher.update(&current_hash);
                hasher.update(sibling);
            } else {
                hasher.update(sibling);
                hasher.update(&current_hash);
            }

            current_hash = hasher.finalize();
            node_logical_idx /= 2;
        }

        &current_hash == root
    }
}

/// Hash one row into a blinded Merkle leaf:
///
/// ```text
/// Leaf = H(
///     0x00
///  || u64_le(len(data || noise))
///  || data_row_bytes
///  || noise_bytes
///  || code_row_bytes
/// )
/// ```
///
/// The length prefix commits to the boundary
/// between `(data || noise)` and `code`, so
/// two rows with shuffled widths cannot collide.
#[inline(always)]
pub fn hash_leaf_row_blinded<H: Hasher>(
    row_idx: usize,
    data_views: &[(&[u8], usize)],
    code_views: &[(&[u8], usize)],
    noise_bytes: &[u8],
) -> [u8; 32] {
    let mut hasher = H::new();

    let physical_data_len: usize = data_views.iter().map(|(_, w)| *w).sum();
    let data_len = physical_data_len + noise_bytes.len();

    hasher.update(&[0u8]);

    let len_bytes = (data_len as u64).to_le_bytes();
    hasher.update(&len_bytes);

    for (base_ptr, width) in data_views {
        let start = row_idx * width;
        let end = start + width;

        // SAFETY:
        // the caller builds each view with a
        // matching `width` and guarantees
        // `row_idx` is in range.
        unsafe {
            let src = base_ptr.get_unchecked(start..end);
            hasher.update(src);
        }
    }

    if !noise_bytes.is_empty() {
        hasher.update(noise_bytes);
    }

    for (base_ptr, width) in code_views {
        let start = row_idx * width;
        let end = start + width;

        // SAFETY:
        // see loop above.
        unsafe {
            let src = base_ptr.get_unchecked(start..end);
            hasher.update(src);
        }
    }

    hasher.finalize()
}

/// Hash one 2D-grid column into a Merkle leaf.
/// Only the encoded codeword bytes are hashed;
/// raw data stays private.
#[inline(always)]
pub fn hash_leaf_column_encoded<H: Hasher>(
    col_idx: usize,
    grid_rows: usize,
    encoded_width: usize,
    code_views: &[(&[u8], usize)],
) -> [u8; 32] {
    let mut hasher = H::new();
    hasher.update(&[0u8]);

    for r in 0..grid_rows {
        for (base_ptr, width) in code_views {
            let start = (r * encoded_width + col_idx) * width;
            let end = start + width;

            // SAFETY:
            // caller guarantees `col_idx`,
            // `grid_rows`, and `encoded_width`
            // match the underlying buffers.
            unsafe {
                hasher.update(base_ptr.get_unchecked(start..end));
            }
        }
    }

    hasher.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hekate_math::Block128;

    type H = DefaultHasher;

    fn hash_bytes(data: &[u8]) -> [u8; 32] {
        let mut hasher = H::new();
        hasher.update(&[0u8]);
        hasher.update(data);

        hasher.finalize()
    }

    #[test]
    fn merkle_tree_basics() {
        let leaves: Vec<[u8; 32]> = (1..=4u8).map(|i| hash_bytes(&[i])).collect();

        let tree = MerkleTree::<Block128, H>::new(&leaves);
        let root = tree.root();

        assert_ne!(root, [0u8; 32]);

        let proof = tree.prove(2).unwrap();
        assert_eq!(proof.len(), 2, "Proof length should be log2(num_leaves)");

        let is_valid = MerkleTree::<Block128, H>::verify(&root, leaves[2], 2, &proof);
        assert!(is_valid, "Merkle Proof rejected a valid leaf");

        let is_invalid = MerkleTree::<Block128, H>::verify(&root, leaves[0], 2, &proof);
        assert!(!is_invalid, "Merkle Proof accepted a wrong leaf");
    }

    #[test]
    fn merkle_odd_leaves() {
        let leaves: Vec<[u8; 32]> = (1..=3u8).map(|i| hash_bytes(&[i])).collect();
        let tree = MerkleTree::<Block128, H>::new(&leaves);

        assert_eq!(tree.num_leaves(), 4);

        let proof0 = tree.prove(0).unwrap();
        assert!(MerkleTree::<Block128, H>::verify(
            &tree.root(),
            leaves[0],
            0,
            &proof0
        ));

        let proof2 = tree.prove(2).unwrap();
        assert!(MerkleTree::<Block128, H>::verify(
            &tree.root(),
            leaves[2],
            2,
            &proof2
        ));
    }

    #[test]
    fn merkle_empty() {
        let leaves: Vec<[u8; 32]> = vec![];
        let tree = MerkleTree::<Block128, H>::new(&leaves);
        assert_eq!(tree.root(), [0u8; 32]);
        assert_eq!(tree.num_leaves, 0);
    }

    #[test]
    fn streaming_build_matches_new() {
        let leaves: Vec<[u8; 32]> = (0..1024u32).map(|i| hash_bytes(&i.to_le_bytes())).collect();

        let tree_ref = MerkleTree::<Block128, H>::new(&leaves);

        let (mut tree_stream, leaf_offset) = MerkleTree::<Block128, H>::allocate_tree(leaves.len());
        let leaf_layer = tree_stream.leaves_mut(leaf_offset);

        for (i, slot) in leaf_layer.iter_mut().enumerate() {
            if i < leaves.len() {
                slot.write(leaves[i]);
            } else {
                slot.write([0u8; 32]);
            }
        }

        tree_stream.build_layers(leaf_offset);

        assert_eq!(tree_stream.root(), tree_ref.root());

        for idx in [0usize, 1, 2, 511, 1023] {
            let proof = tree_stream.prove(idx).unwrap();
            assert!(MerkleTree::<Block128, H>::verify(
                &tree_stream.root(),
                leaves[idx],
                idx,
                &proof
            ));
        }
    }

    #[test]
    fn allocate_tree_padding_behavior_matches_new() {
        for n in [3usize, 5, 6] {
            let leaves: Vec<[u8; 32]> = (0..(n as u32))
                .map(|i| hash_bytes(&i.to_le_bytes()))
                .collect();

            let tree_ref = MerkleTree::<Block128, H>::new(&leaves);

            let (mut tree_stream, leaf_offset) = MerkleTree::<Block128, H>::allocate_tree(n);
            let leaf_layer = tree_stream.leaves_mut(leaf_offset);

            for (i, slot) in leaf_layer.iter_mut().enumerate() {
                if i < leaves.len() {
                    slot.write(leaves[i]);
                } else {
                    slot.write([0u8; 32]);
                }
            }

            tree_stream.build_layers(leaf_offset);

            assert_eq!(tree_stream.num_leaves(), tree_ref.num_leaves());
            assert_eq!(tree_stream.root(), tree_ref.root());

            for (idx, &leaf) in leaves.iter().enumerate() {
                let proof = tree_stream.prove(idx).unwrap();
                assert!(MerkleTree::<Block128, H>::verify(
                    &tree_stream.root(),
                    leaf,
                    idx,
                    &proof
                ));
            }
        }
    }

    #[test]
    fn prove_rejects_oob_leaf_index() {
        let leaves: Vec<[u8; 32]> = (0..8u32).map(|i| hash_bytes(&i.to_le_bytes())).collect();
        let tree = MerkleTree::<Block128, H>::new(&leaves);

        assert!(tree.prove(8).is_err());
        assert!(tree.prove(usize::MAX).is_err());
    }

    #[test]
    fn same_leaves_same_root() {
        let leaves: Vec<[u8; 32]> = (0..64u32).map(|i| hash_bytes(&i.to_le_bytes())).collect();

        let t1 = MerkleTree::<Block128, H>::new(&leaves);
        let t2 = MerkleTree::<Block128, H>::new(&leaves);

        assert_eq!(t1.root(), t2.root());
    }

    #[test]
    fn different_leaf_changes_root() {
        let mut leaves: Vec<[u8; 32]> = (0..64u32).map(|i| hash_bytes(&i.to_le_bytes())).collect();

        let t1 = MerkleTree::<Block128, H>::new(&leaves);

        leaves[17] = hash_bytes(b"different");
        let t2 = MerkleTree::<Block128, H>::new(&leaves);

        assert_ne!(t1.root(), t2.root());
    }

    #[test]
    fn hash_leaf_row_blinded_includes_length_prefix() {
        let data = [1u8, 2u8];
        let code = [3u8, 4u8, 5u8];

        let data_views = vec![(&data[..], data.len())];
        let code_views = vec![(&code[..], code.len())];

        let expected = {
            let mut h = H::new();
            h.update(&[0u8]);
            h.update(&(data.len() as u64).to_le_bytes());
            h.update(&data);
            h.update(&code);

            h.finalize()
        };

        let got = hash_leaf_row_blinded::<H>(0, &data_views, &code_views, &[]);
        assert_eq!(got, expected);
    }

    #[test]
    fn hash_leaf_row_blinded_rejects_ambiguous_concatenation() {
        let data_a = [1u8, 2u8];
        let code_a = [3u8];

        let data_b = [1u8];
        let code_b = [2u8, 3u8];

        let h_a = hash_leaf_row_blinded::<H>(
            0,
            &[(&data_a[..], data_a.len())],
            &[(&code_a[..], code_a.len())],
            &[],
        );

        let h_b = hash_leaf_row_blinded::<H>(
            0,
            &[(&data_b[..], data_b.len())],
            &[(&code_b[..], code_b.len())],
            &[],
        );

        assert_ne!(h_a, h_b);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn root_panics_if_not_built_in_debug() {
        let (tree, _leaf_offset) = MerkleTree::<Block128, H>::allocate_tree(4);
        let _ = tree.root();
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn prove_panics_if_not_built_in_debug() {
        let (tree, _leaf_offset) = MerkleTree::<Block128, H>::allocate_tree(4);
        let _ = tree.prove(0);
    }
}
