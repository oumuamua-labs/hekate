// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use crate::{DefaultHasher, Hasher};
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use hekate_math::TowerField;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Leaves of one subtree a worker hashes whole.
#[cfg(feature = "parallel")]
const SUBTREE_LEAVES: usize = 1024;

/// Scratch bytes per worker for one tile of leaf preimages.
const TILE_BYTES: usize = 256 * 1024;

/// Most leaves in one tile.
const TILE_LEAVES: usize = 64;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    LeafIndexOutOfBounds {
        leaf_index: usize,
        num_leaves: usize,
    },
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
        }
    }
}

/// Binary Merkle tree over 32-byte leaves.
///
/// Internal node = `H(0x01 || left || right)`.
/// Leaves are expected to already be hashes;
/// callers serialize their field payloads via
/// `hash_leaf_row_blinded` / `hash_column_leaves`.
#[derive(Clone, Debug)]
pub struct MerkleTree<F: TowerField, H: Hasher = DefaultHasher> {
    nodes: Vec<MaybeUninit<[u8; 32]>>,
    num_leaves: usize,

    /// Guard for the `MaybeUninit` nodes:
    /// reading `root`/`prove`/`prove_batch`
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
            current_hash = if leaf_index.is_multiple_of(2) {
                hash_node::<H>(&current_hash, sibling)
            } else {
                hash_node::<H>(sibling, &current_hash)
            };

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
        #[cfg(feature = "parallel")]
        let (mut current_layer_size, mut current_offset) = self.build_subtrees(leaf_offset);

        #[cfg(not(feature = "parallel"))]
        let (mut current_layer_size, mut current_offset) = (self.num_leaves, leaf_offset);

        while current_offset > 0 {
            let parent_layer_size = current_layer_size / 2;
            let parent_offset = current_offset - parent_layer_size;

            let (upper, lower) = self.nodes.split_at_mut(current_offset);
            let parents = &mut upper[parent_offset..parent_offset + parent_layer_size];
            let children = &lower[0..current_layer_size];

            hash_layer::<H>(children, parents);

            current_layer_size = parent_layer_size;
            current_offset = parent_offset;
        }

        self.built = true;
    }

    /// Hashes the bottom layers by whole subtrees on the pool
    /// and returns the subtree roots' layer `(size, offset)`.
    #[cfg(feature = "parallel")]
    fn build_subtrees(&mut self, leaf_offset: usize) -> (usize, usize) {
        let num_leaves = leaf_offset + 1;
        let levels = SUBTREE_LEAVES.min(num_leaves).trailing_zeros() as usize;

        let count = num_leaves >> levels;
        let top_offset = count - 1;

        let mut subtrees: Vec<Vec<&mut [MaybeUninit<[u8; 32]>]>> =
            (0..count).map(|_| Vec::with_capacity(levels + 1)).collect();

        let mut rest = &mut self.nodes[top_offset..];
        let mut layers = Vec::with_capacity(levels + 1);

        for t in (0..=levels).rev() {
            let (layer, tail) = rest.split_at_mut(num_leaves >> t);
            layers.push(layer);

            rest = tail;
        }

        for layer in layers.into_iter().rev() {
            let chunk = layer.len() / count;
            for (subtree, part) in subtrees.iter_mut().zip(layer.chunks_mut(chunk)) {
                subtree.push(part);
            }
        }

        subtrees.par_iter_mut().for_each(|layers| {
            for t in 1..layers.len() {
                let (lower, upper) = layers.split_at_mut(t);
                hash_layer::<H>(&lower[t - 1][..], &mut upper[0][..]);
            }
        });

        (count, top_offset)
    }

    // =================================
    // Batch (octopus) proofs
    // =================================

    /// Octopus multiproof: one pruned sibling set opening
    /// every `leaf_indices` entry against the root, each
    /// shared sibling sent once. Emitted in canonical order;
    /// `verify_batch` must consume it in the same order.
    pub fn prove_batch(&self, leaf_indices: &[usize]) -> Result<Vec<[u8; 32]>> {
        // SAFETY:
        // see `root`, requires `built`.
        assert!(
            self.nodes.is_empty() || self.built,
            "MerkleTree::prove_batch called before build_layers"
        );

        let mut frontier: Vec<usize> = Vec::with_capacity(leaf_indices.len());
        for &idx in leaf_indices {
            if idx >= self.num_leaves {
                return Err(Error::LeafIndexOutOfBounds {
                    leaf_index: idx,
                    num_leaves: self.num_leaves,
                });
            }

            frontier.push(idx);
        }

        frontier.sort_unstable();
        frontier.dedup();

        let mut siblings = Vec::new();
        let mut next: Vec<usize> = Vec::with_capacity(frontier.len());
        let mut layer_width = self.num_leaves;

        while layer_width > 1 {
            let layer_offset = layer_width - 1;

            next.clear();

            let mut i = 0;
            while i < frontier.len() {
                let node = frontier[i];
                let sibling = node ^ 1;

                if i + 1 < frontier.len() && frontier[i + 1] == sibling {
                    i += 2;
                } else {
                    // SAFETY:
                    // `built` guarantees every node is
                    // initialized; `sibling < layer_width`
                    // keeps the index inside this layer.
                    siblings.push(unsafe { self.nodes[layer_offset + sibling].assume_init() });

                    i += 1;
                }

                next.push(node >> 1);
            }

            core::mem::swap(&mut frontier, &mut next);

            layer_width >>= 1;
        }

        Ok(siblings)
    }

    /// Verifies a `prove_batch` multiproof against `root`.
    /// `num_leaves` is the padded power-of-two leaf count;
    /// `leaves` must be sorted strictly ascending by index.
    pub fn verify_batch(
        root: &[u8; 32],
        num_leaves: usize,
        leaves: &[(usize, [u8; 32])],
        siblings: &[[u8; 32]],
    ) -> bool {
        if !num_leaves.is_power_of_two() || leaves.is_empty() {
            return false;
        }

        let mut frontier: Vec<(usize, [u8; 32])> = Vec::with_capacity(leaves.len());
        let mut prev: Option<usize> = None;

        for &(idx, hash) in leaves {
            if idx >= num_leaves {
                return false;
            }

            if let Some(p) = prev
                && idx <= p
            {
                return false;
            }

            prev = Some(idx);
            frontier.push((idx, hash));
        }

        let mut sib_pos = 0usize;
        let mut next: Vec<(usize, [u8; 32])> = Vec::with_capacity(leaves.len());
        let mut layer_width = num_leaves;

        while layer_width > 1 {
            next.clear();

            let mut i = 0;
            while i < frontier.len() {
                let (idx, hash) = frontier[i];
                let sibling_idx = idx ^ 1;

                let (left, right) = if i + 1 < frontier.len() && frontier[i + 1].0 == sibling_idx {
                    let sib_hash = frontier[i + 1].1;
                    i += 2;

                    (hash, sib_hash)
                } else {
                    if sib_pos >= siblings.len() {
                        return false;
                    }

                    let sib_hash = siblings[sib_pos];

                    sib_pos += 1;
                    i += 1;

                    if idx.is_multiple_of(2) {
                        (hash, sib_hash)
                    } else {
                        (sib_hash, hash)
                    }
                };

                next.push((idx >> 1, hash_node::<H>(&left, &right)));
            }

            core::mem::swap(&mut frontier, &mut next);

            layer_width >>= 1;
        }

        sib_pos == siblings.len() && frontier.len() == 1 && &frontier[0].1 == root
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

/// One digest per grid column, `H(0x00 ‖ its cells row by row)`,
/// for columns `0..digests.len()`, hashed by tiles of columns.
pub fn hash_column_leaves<H: Hasher>(
    grid_rows: usize,
    encoded_width: usize,
    code_views: &[(&[u8], usize)],
    digests: &mut [[u8; 32]],
) {
    let stride = 1 + grid_rows * code_views.iter().map(|&(_, width)| width).sum::<usize>();
    let tile = (TILE_BYTES / stride).clamp(1, TILE_LEAVES);

    #[cfg(feature = "parallel")]
    digests
        .par_chunks_mut(tile)
        .enumerate()
        .for_each_init(Vec::new, |scratch, (t, out)| {
            hash_tile::<H>(t * tile, grid_rows, encoded_width, code_views, scratch, out);
        });

    #[cfg(not(feature = "parallel"))]
    {
        let mut scratch = Vec::new();
        for (t, out) in digests.chunks_mut(tile).enumerate() {
            hash_tile::<H>(
                t * tile,
                grid_rows,
                encoded_width,
                code_views,
                &mut scratch,
                out,
            );
        }
    }
}

/// Writes one pool leaf per column: each leaf hashes
/// every table's digest of that column, in pool order.
///
/// # Panics
/// If a table holds fewer digests than `leaves`.
pub fn hash_parts_leaves<H: Hasher>(
    table_digests: &[&[[u8; 32]]],
    leaves: &mut [MaybeUninit<[u8; 32]>],
) {
    #[cfg(feature = "parallel")]
    leaves.par_iter_mut().enumerate().for_each(|(j, leaf)| {
        leaf.write(hash_parts_leaf::<H>(table_digests.iter().map(|t| &t[j])));
    });

    #[cfg(not(feature = "parallel"))]
    for (j, leaf) in leaves.iter_mut().enumerate() {
        leaf.write(hash_parts_leaf::<H>(table_digests.iter().map(|t| &t[j])));
    }
}

/// Hashes one pool leaf from the `0x02` tag followed
/// by each table's digest of the column, in pool order.
pub fn hash_parts_leaf<'a, H: Hasher>(digests: impl IntoIterator<Item = &'a [u8; 32]>) -> [u8; 32] {
    let mut hasher = H::new();
    hasher.update(&[2u8]);

    for digest in digests {
        hasher.update(digest);
    }

    hasher.finalize()
}

/// Digests of columns `first..first + out.len()`,
/// their preimages assembled side by side in `scratch`.
fn hash_tile<H: Hasher>(
    first: usize,
    grid_rows: usize,
    encoded_width: usize,
    code_views: &[(&[u8], usize)],
    scratch: &mut Vec<u8>,
    out: &mut [[u8; 32]],
) {
    let stride = 1 + grid_rows * code_views.iter().map(|&(_, width)| width).sum::<usize>();

    scratch.resize(out.len() * stride, 0);

    for prefix in scratch.iter_mut().step_by(stride) {
        *prefix = 0;
    }

    let mut offset = 1;
    for row in 0..grid_rows {
        let base = row * encoded_width + first;
        for &(column, width) in code_views {
            let cells = &column[base * width..(base + out.len()) * width];

            match width {
                4 => scatter::<4>(cells, scratch, stride, offset),
                8 => scatter::<8>(cells, scratch, stride, offset),
                16 => scatter::<16>(cells, scratch, stride, offset),
                _ => {
                    for (i, cell) in cells.chunks_exact(width).enumerate() {
                        let at = i * stride + offset;
                        scratch[at..at + width].copy_from_slice(cell);
                    }
                }
            }

            offset += width;
        }
    }

    for (digest, preimage) in out.iter_mut().zip(scratch.chunks_exact(stride)) {
        *digest = H::digest(preimage);
    }
}

/// One tree layer, each parent the node of its two children.
fn hash_layer<H: Hasher>(
    children: &[MaybeUninit<[u8; 32]>],
    parents: &mut [MaybeUninit<[u8; 32]>],
) {
    for (parent, [lo, hi]) in parents.iter_mut().zip(children.as_chunks::<2>().0) {
        // SAFETY: a layer is hashed only after the one below
        // it is written, the leaves by `build_layers`' caller.
        let (left, right) = unsafe { (lo.assume_init_ref(), hi.assume_init_ref()) };

        parent.write(hash_node::<H>(left, right));
    }
}

/// `H(0x01 ‖ left ‖ right)`, an internal node of [`MerkleTree`].
fn hash_node<H: Hasher>(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut node = [0u8; 65];
    node[0] = 1;
    node[1..33].copy_from_slice(left);
    node[33..].copy_from_slice(right);

    H::digest(&node)
}

/// Copies cell `i` of `cells` to `dst[i * stride + offset..]`.
fn scatter<const CELL: usize>(cells: &[u8], dst: &mut [u8], stride: usize, offset: usize) {
    for (i, cell) in cells.as_chunks::<CELL>().0.iter().enumerate() {
        let at = i * stride + offset;
        dst[at..at + CELL].copy_from_slice(cell);
    }
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

    fn batch_leaves(indices: &[usize], leaf_hashes: &[[u8; 32]]) -> Vec<(usize, [u8; 32])> {
        let mut distinct = indices.to_vec();
        distinct.sort_unstable();
        distinct.dedup();

        distinct.into_iter().map(|i| (i, leaf_hashes[i])).collect()
    }

    fn per_column_leaf(
        col: usize,
        grid_rows: usize,
        encoded_width: usize,
        views: &[(&[u8], usize)],
    ) -> [u8; 32] {
        let mut hasher = H::new();
        hasher.update(&[0u8]);

        for row in 0..grid_rows {
            for &(column, width) in views {
                let start = (row * encoded_width + col) * width;
                hasher.update(&column[start..start + width]);
            }
        }

        hasher.finalize()
    }

    fn code_columns(cells: &[usize], grid_rows: usize, width: usize) -> Vec<Vec<u8>> {
        cells
            .iter()
            .enumerate()
            .map(|(k, &cell)| {
                let mut state =
                    0x9E37_79B9_7F4A_7C15u64 ^ (k as u64 + 1).wrapping_mul(0xD6E8_FEB8_6659_FD93);
                let mut out = vec![0u8; grid_rows * width * cell];

                for chunk in out.chunks_mut(8) {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;

                    chunk.copy_from_slice(&state.to_le_bytes()[..chunk.len()]);
                }

                out
            })
            .collect()
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
    fn batch_proof_verifies_and_matches_single_paths() {
        let leaf_hashes: Vec<[u8; 32]> = (0..64u32).map(|i| hash_bytes(&i.to_le_bytes())).collect();
        let tree = MerkleTree::<Block128, H>::new(&leaf_hashes);
        let root = tree.root();

        for query in [
            &[0usize][..],
            &[63],
            &[0, 1],
            &[0, 63],
            &[7, 7, 7],
            &[0, 1, 2, 5, 17, 63],
            &[1, 3, 5, 7, 9, 11, 40, 41],
        ] {
            let siblings = tree.prove_batch(query).unwrap();
            let leaves = batch_leaves(query, &leaf_hashes);

            assert!(
                MerkleTree::<Block128, H>::verify_batch(&root, 64, &leaves, &siblings),
                "batch proof rejected for {query:?}"
            );

            for &(idx, leaf) in &leaves {
                let single = tree.prove(idx).unwrap();
                assert!(MerkleTree::<Block128, H>::verify(&root, leaf, idx, &single));
            }
        }
    }

    #[test]
    fn batch_proof_full_leaf_set_needs_no_siblings() {
        let leaf_hashes: Vec<[u8; 32]> = (0..32u32).map(|i| hash_bytes(&i.to_le_bytes())).collect();
        let tree = MerkleTree::<Block128, H>::new(&leaf_hashes);

        let all: Vec<usize> = (0..32).collect();
        let siblings = tree.prove_batch(&all).unwrap();

        assert!(siblings.is_empty(), "full leaf set needs zero siblings");

        let leaves = batch_leaves(&all, &leaf_hashes);

        assert!(MerkleTree::<Block128, H>::verify_batch(
            &tree.root(),
            32,
            &leaves,
            &siblings
        ));
    }

    #[test]
    fn batch_proof_single_leaf_sibling_count_is_depth() {
        let leaf_hashes: Vec<[u8; 32]> = (0..64u32).map(|i| hash_bytes(&i.to_le_bytes())).collect();
        let tree = MerkleTree::<Block128, H>::new(&leaf_hashes);

        let siblings = tree.prove_batch(&[42]).unwrap();

        assert_eq!(siblings.len(), 6, "single-leaf octopus is a full path");
        assert_eq!(siblings, tree.prove(42).unwrap());
    }

    #[test]
    fn verify_batch_rejects_tampering() {
        let leaf_hashes: Vec<[u8; 32]> = (0..64u32).map(|i| hash_bytes(&i.to_le_bytes())).collect();
        let tree = MerkleTree::<Block128, H>::new(&leaf_hashes);
        let root = tree.root();

        let query = [3usize, 8, 8, 20, 55];
        let siblings = tree.prove_batch(&query).unwrap();
        let leaves = batch_leaves(&query, &leaf_hashes);

        assert!(MerkleTree::<Block128, H>::verify_batch(
            &root, 64, &leaves, &siblings
        ));

        let mut wrong_leaf = leaves.clone();
        wrong_leaf[1].1 = hash_bytes(b"forged");

        assert!(!MerkleTree::<Block128, H>::verify_batch(
            &root,
            64,
            &wrong_leaf,
            &siblings
        ));

        let mut extra = siblings.clone();
        extra.push([0u8; 32]);

        assert!(
            !MerkleTree::<Block128, H>::verify_batch(&root, 64, &leaves, &extra),
            "unconsumed extra sibling must reject"
        );

        let missing = &siblings[..siblings.len() - 1];

        assert!(
            !MerkleTree::<Block128, H>::verify_batch(&root, 64, &leaves, missing),
            "missing sibling must reject"
        );

        assert!(!MerkleTree::<Block128, H>::verify_batch(
            &[9u8; 32], 64, &leaves, &siblings
        ));

        let unsorted = vec![leaves[2], leaves[0], leaves[1], leaves[3]];

        assert!(
            !MerkleTree::<Block128, H>::verify_batch(&root, 64, &unsorted, &siblings),
            "unsorted leaves must reject"
        );

        let dup = vec![leaves[0], leaves[0], leaves[1]];

        assert!(
            !MerkleTree::<Block128, H>::verify_batch(&root, 64, &dup, &siblings),
            "duplicate leaf indices must reject"
        );

        assert!(
            !MerkleTree::<Block128, H>::verify_batch(&root, 63, &leaves, &siblings),
            "non-power-of-two leaf count must reject"
        );
    }

    #[test]
    fn batch_proof_padded_tree() {
        let leaf_hashes: Vec<[u8; 32]> = (0..5u32).map(|i| hash_bytes(&i.to_le_bytes())).collect();
        let tree = MerkleTree::<Block128, H>::new(&leaf_hashes);
        let padded = tree.num_leaves();

        assert_eq!(padded, 8);

        let query = [0usize, 4];
        let siblings = tree.prove_batch(&query).unwrap();
        let padded_hashes: Vec<[u8; 32]> = (0..padded)
            .map(|i| {
                if i < leaf_hashes.len() {
                    leaf_hashes[i]
                } else {
                    [0u8; 32]
                }
            })
            .collect();

        let leaves = batch_leaves(&query, &padded_hashes);

        assert!(MerkleTree::<Block128, H>::verify_batch(
            &tree.root(),
            padded,
            &leaves,
            &siblings
        ));
    }

    #[test]
    fn prove_batch_rejects_oob_index() {
        let leaf_hashes: Vec<[u8; 32]> = (0..8u32).map(|i| hash_bytes(&i.to_le_bytes())).collect();
        let tree = MerkleTree::<Block128, H>::new(&leaf_hashes);

        assert!(tree.prove_batch(&[0, 8]).is_err());
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

    #[test]
    fn tiled_column_leaves_equal_per_column_leaves() {
        let layouts: [&[usize]; 7] = [
            &[1; 3],
            &[2; 3],
            &[4; 5],
            &[8; 2],
            &[16; 4],
            &[16; 16],
            &[4, 16, 8, 1, 2, 16],
        ];

        for cells in layouts {
            for grid_rows in [1usize, 2, 8, 32] {
                for width in [1usize, 63, 300] {
                    let columns = code_columns(cells, grid_rows, width);
                    let views: Vec<(&[u8], usize)> = columns
                        .iter()
                        .zip(cells)
                        .map(|(column, &cell)| (column.as_slice(), cell))
                        .collect();

                    let mut tiled = vec![[0u8; 32]; width];
                    hash_column_leaves::<H>(grid_rows, width, &views, &mut tiled);

                    for (col, &leaf) in tiled.iter().enumerate() {
                        assert_eq!(
                            leaf,
                            per_column_leaf(col, grid_rows, width, &views),
                            "cells {cells:?} grid_rows {grid_rows} width {width} col {col}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn subtree_layers_equal_level_by_level_layers() {
        for count in [1usize, 2, 3, 1023, 1024, 1025, 4096, 70_000] {
            let leaves: Vec<[u8; 32]> = (0..count as u32)
                .map(|i| hash_bytes(&i.to_le_bytes()))
                .collect();

            let tree = MerkleTree::<Block128, H>::new(&leaves);
            let num_leaves = tree.num_leaves();

            let mut expected = vec![[0u8; 32]; 2 * num_leaves - 1];
            expected[num_leaves - 1..num_leaves - 1 + count].copy_from_slice(&leaves);

            for i in (0..num_leaves - 1).rev() {
                expected[i] = hash_node::<H>(&expected[2 * i + 1], &expected[2 * i + 2]);
            }

            // SAFETY: `new` returns a built tree, every node initialized.
            let built: Vec<[u8; 32]> = tree
                .nodes
                .iter()
                .map(|n| unsafe { n.assume_init() })
                .collect();

            assert_eq!(built, expected, "{count} leaves");
        }
    }

    #[test]
    fn parts_leaves_hash_tagged_table_digests() {
        let tables: Vec<Vec<[u8; 32]>> = (0..3u8)
            .map(|t| (0..37u8).map(|j| [t.wrapping_mul(41) ^ j; 32]).collect())
            .collect();

        let views: Vec<&[[u8; 32]]> = tables.iter().map(Vec::as_slice).collect();

        let mut leaves = vec![MaybeUninit::new([0u8; 32]); 37];
        hash_parts_leaves::<H>(&views, &mut leaves);

        for (j, leaf) in leaves.iter().enumerate() {
            let mut preimage = vec![2u8];
            for table in &tables {
                preimage.extend_from_slice(&table[j]);
            }

            // SAFETY: every slot starts initialized
            let leaf = unsafe { leaf.assume_init() };

            assert_eq!(leaf, H::digest(&preimage));
        }
    }
}
