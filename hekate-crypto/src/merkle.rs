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
use serde::{Deserialize, Serialize};

/// Leaves of one subtree a worker hashes whole.
#[cfg(feature = "parallel")]
const SUBTREE_LEAVES: usize = 1024;

const BATCH: usize = 64;
const NODE_BYTES: usize = 65;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    LeafIndexOutOfBounds {
        leaf_index: usize,
        num_leaves: usize,
    },
    Pruned {
        floor: usize,
    },
    SubtreeMismatch {
        subtree: usize,
    },
    SubtreeTiling {
        num_leaves: usize,
        floor: usize,
        roots: usize,
    },
    SubtreeWidth {
        leaves: usize,
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
            Self::Pruned { floor } => write!(
                f,
                "Merkle tree stores no level below {floor}: open it with prove_batch_with",
            ),
            Self::SubtreeMismatch { subtree } => write!(
                f,
                "Merkle subtree {subtree} rebuilt to a root the tree does not hold",
            ),
            Self::SubtreeTiling {
                num_leaves,
                floor,
                roots,
            } => write!(
                f,
                "{roots} Merkle subtree roots do not tile {num_leaves} leaves at level {floor}",
            ),
            Self::SubtreeWidth { leaves } => {
                write!(f, "Merkle subtree over {leaves} leaves, not a power of two",)
            }
        }
    }
}

/// Last byte of every Merkle preimage, one per kind.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Tag {
    Node = 0x01,
    PoolLeaf = 0x02,
    OuterLeaf = 0x03,
    RoundLeaf = 0x04,
}

/// Opening of a leaf set: the octopus siblings below the [`layer_log`] layer,
/// then the layer nodes no opened path reaches, ascending by position.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct Multiproof {
    pub siblings: Vec<[u8; 32]>,
    pub unreached: Vec<[u8; 32]>,
}

/// Binary Merkle tree over 32-byte leaves.
///
/// Internal node = `H(left || right || 0x01)`.
/// Leaves are expected to already be hashes; callers
/// hash their symbols via `hash_symbol_leaves`.
#[derive(Clone, Debug)]
pub struct MerkleTree<F: TowerField, H: Hasher = DefaultHasher> {
    nodes: Vec<MaybeUninit<[u8; 32]>>,
    num_leaves: usize,

    /// Guard for the `MaybeUninit` nodes:
    /// reading `root`/`prove`/`prove_batch`
    /// before `build_layers` runs is UB.
    built: bool,

    /// Lowest stored level.
    floor: usize,

    _marker: PhantomData<(F, H)>,
}

impl<F: TowerField, H: Hasher> MerkleTree<F, H> {
    /// Build a tree from pre-computed leaf hashes.
    /// Non-power-of-two inputs pad with zero leaves.
    pub fn new(leaves: &[[u8; 32]]) -> Self {
        let num_leaves = leaves.len();
        if num_leaves == 0 {
            return Self::empty();
        }

        let (mut tree, leaf_offset) = Self::allocate_tree(num_leaves);

        let leaf_layer = &mut tree.nodes[leaf_offset..leaf_offset + tree.num_leaves];

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

        tree.hash_layers(leaf_offset);

        tree
    }

    /// Builds the tree over `num_leaves` leaves from its
    /// level-`floor` nodes `roots`, storing levels `floor` and up;
    /// errors unless the roots tile a power-of-two leaf count.
    pub fn from_subtree_roots(num_leaves: usize, floor: usize, roots: &[[u8; 32]]) -> Result<Self> {
        if !(num_leaves.is_power_of_two()
            && floor <= num_leaves.trailing_zeros() as usize
            && roots.len() == num_leaves >> floor)
        {
            return Err(Error::SubtreeTiling {
                num_leaves,
                floor,
                roots: roots.len(),
            });
        }

        let mut tree = Self::new(roots);
        tree.num_leaves = num_leaves;
        tree.floor = floor;

        Ok(tree)
    }

    pub fn num_leaves(&self) -> usize {
        self.num_leaves
    }

    /// Mutable view of the leaf layer for streaming
    /// writes. Slots are `MaybeUninit`, the caller
    /// must populate every slot before `build_layers`.
    pub fn leaves_mut(&mut self, leaf_offset: usize) -> Result<&mut [MaybeUninit<[u8; 32]>]> {
        if self.floor > 0 {
            return Err(Error::Pruned { floor: self.floor });
        }

        Ok(&mut self.nodes[leaf_offset..leaf_offset + self.num_leaves])
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

        if self.floor > 0 {
            return Err(Error::Pruned { floor: self.floor });
        }

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
            floor: 0,
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
                floor: 0,
                _marker: PhantomData,
            },
            leaf_offset,
        )
    }

    pub fn build_layers(&mut self, leaf_offset: usize) -> Result<()> {
        if self.floor > 0 {
            return Err(Error::Pruned { floor: self.floor });
        }

        self.hash_layers(leaf_offset);

        Ok(())
    }

    fn hash_layers(&mut self, leaf_offset: usize) {
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

    /// Multiproof opening every `leaf_indices` entry against
    /// the root for `queries` draws, each shared sibling sent
    /// once; `verify_batch` consumes it in the same order.
    pub fn prove_batch(&self, leaf_indices: &[usize], queries: usize) -> Result<Multiproof> {
        // SAFETY:
        // see `root`, requires `built`.
        assert!(
            self.nodes.is_empty() || self.built,
            "MerkleTree::prove_batch called before build_layers"
        );

        if self.floor > 0 {
            return Err(Error::Pruned { floor: self.floor });
        }

        self.multiproof(leaf_indices, queries, |layer_width, position| {
            // SAFETY:
            // `built` guarantees every node is initialized;
            // `position < layer_width` keeps the index inside the layer.
            Ok(unsafe { self.nodes[layer_width - 1 + position].assume_init() })
        })
    }

    /// Opens `leaf_indices` as `prove_batch` does, rebuilding each
    /// pruned subtree `s` it reads from the hashes `subtree(s, ..)`
    /// writes; errs when a rebuilt root differs from the stored one.
    pub fn prove_batch_with(
        &self,
        leaf_indices: &[usize],
        queries: usize,
        subtree: impl Fn(usize, &mut [[u8; 32]]) + Sync,
    ) -> Result<Multiproof> {
        if self.floor == 0 {
            return self.prove_batch(leaf_indices, queries);
        }

        let depth = self.num_leaves.trailing_zeros() as usize;
        let width = self.num_leaves >> self.floor;
        let span = 1usize << self.floor;

        let frontier = self.frontier(leaf_indices)?;

        let needed: Vec<usize> = match 1usize << layer_log(queries, depth) > width {
            true => (0..width).collect(),
            false => {
                let mut subtrees: Vec<usize> =
                    frontier.iter().map(|&idx| idx >> self.floor).collect();

                subtrees.dedup();

                subtrees
            }
        };

        let rebuild = |&s: &usize| subtree_nodes::<H>(span, |hashes| subtree(s, hashes));

        #[cfg(feature = "parallel")]
        let rebuilt: Vec<Vec<[u8; 32]>> = needed.par_iter().map(rebuild).collect();

        #[cfg(not(feature = "parallel"))]
        let rebuilt: Vec<Vec<[u8; 32]>> = needed.iter().map(rebuild).collect();

        for (&s, nodes) in needed.iter().zip(&rebuilt) {
            // SAFETY:
            // a pruned tree is built when constructed; `s < width`
            // keeps the index inside its lowest stored layer.
            if nodes[0] != unsafe { self.nodes[width - 1 + s].assume_init() } {
                return Err(Error::SubtreeMismatch { subtree: s });
            }
        }

        self.multiproof(leaf_indices, queries, |layer_width, position| {
            if layer_width <= width {
                // SAFETY:
                // a pruned tree is built when constructed;
                // `position < layer_width` keeps the index inside the layer.
                return Ok(unsafe { self.nodes[layer_width - 1 + position].assume_init() });
            }

            let per = layer_width / width;
            let s = position / per;
            let slot = needed
                .binary_search(&s)
                .map_err(|_| Error::SubtreeMismatch { subtree: s })?;

            Ok(rebuilt[slot][per - 1 + position % per])
        })
    }

    /// Verifies a `prove_batch` multiproof for `queries` draws against `root`.
    /// `num_leaves` is the padded power-of-two leaf count;
    /// `leaves` must be sorted strictly ascending by index.
    pub fn verify_batch(
        root: &[u8; 32],
        num_leaves: usize,
        queries: usize,
        leaves: &[(usize, [u8; 32])],
        proof: &Multiproof,
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

        let top_width = 1 << layer_log(queries, num_leaves.trailing_zeros() as usize);
        let siblings = &proof.siblings;

        let mut sib_pos = 0usize;
        let mut next: Vec<(usize, [u8; 32])> = Vec::with_capacity(leaves.len());
        let mut layer_width = num_leaves;

        while layer_width > top_width {
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

        if sib_pos != siblings.len() {
            return false;
        }

        let mut reached = frontier.iter().peekable();
        let mut unreached = proof.unreached.iter();
        let mut layer = Vec::with_capacity(top_width);

        for position in 0..top_width {
            let node = match reached.next_if(|&&(idx, _)| idx == position) {
                Some(&(_, hash)) => Some(hash),
                None => unreached.next().copied(),
            };

            match node {
                Some(hash) => layer.push(hash),
                None => return false,
            }
        }

        if unreached.next().is_some() {
            return false;
        }

        while layer.len() > 1 {
            let parents = layer.len() / 2;
            for i in 0..parents {
                layer[i] = hash_node::<H>(&layer[2 * i], &layer[2 * i + 1]);
            }

            layer.truncate(parents);
        }

        layer.first() == Some(root)
    }

    fn multiproof(
        &self,
        leaf_indices: &[usize],
        queries: usize,
        node: impl Fn(usize, usize) -> Result<[u8; 32]>,
    ) -> Result<Multiproof> {
        let mut frontier = self.frontier(leaf_indices)?;

        let Some(depth) = self.num_leaves.checked_ilog2() else {
            return Ok(Multiproof::default());
        };

        let top_width = 1 << layer_log(queries, depth as usize);

        let mut siblings = Vec::new();
        let mut next: Vec<usize> = Vec::with_capacity(frontier.len());
        let mut layer_width = self.num_leaves;

        while layer_width > top_width {
            next.clear();

            let mut i = 0;
            while i < frontier.len() {
                let current = frontier[i];
                let sibling = current ^ 1;

                if i + 1 < frontier.len() && frontier[i + 1] == sibling {
                    i += 2;
                } else {
                    siblings.push(node(layer_width, sibling)?);

                    i += 1;
                }

                next.push(current >> 1);
            }

            core::mem::swap(&mut frontier, &mut next);

            layer_width >>= 1;
        }

        let mut reached = frontier.iter().copied().peekable();
        let mut unreached = Vec::with_capacity(top_width - frontier.len());

        for position in 0..top_width {
            if reached.next_if_eq(&position).is_none() {
                unreached.push(node(top_width, position)?);
            }
        }

        Ok(Multiproof {
            siblings,
            unreached,
        })
    }

    fn frontier(&self, leaf_indices: &[usize]) -> Result<Vec<usize>> {
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

        Ok(frontier)
    }
}

/// Level of the [`Multiproof`] layer:
/// `⌈log2 queries⌉`, capped at `depth`.
pub fn layer_log(queries: usize, depth: usize) -> usize {
    let ceil_log2 = match queries {
        0 | 1 => 0,
        q => (usize::BITS - (q - 1).leading_zeros()) as usize,
    };

    ceil_log2.min(depth)
}

/// Writes leaf `j` as [`hash_symbol_leaf`] over `bytes[j·run..(j+1)·run]`
/// of every `(bytes, run)` member, in member order.
///
/// # Panics
/// If a member holds fewer than `leaves.len()` runs.
pub fn hash_symbol_leaves<H: Hasher>(
    members: &[(&[u8], usize)],
    tag: Tag,
    leaves: &mut [[u8; 32]],
    preimages: &mut Vec<u8>,
) {
    let len = members.iter().map(|&(_, run)| run).sum::<usize>() + 1;

    for (b, slots) in leaves.chunks_mut(BATCH).enumerate() {
        preimages.clear();
        preimages.reserve_exact(slots.len() * len);

        for j in b * BATCH..b * BATCH + slots.len() {
            for &(bytes, run) in members {
                preimages.extend_from_slice(&bytes[j * run..(j + 1) * run]);
            }

            preimages.push(tag as u8);
        }

        H::digest_many(preimages, len, slots);
    }
}

/// `H(parts ‖ tag)`: one leaf of a tree over codeword symbols.
pub fn hash_symbol_leaf<'a, H: Hasher>(
    parts: impl IntoIterator<Item = &'a [u8]>,
    tag: Tag,
) -> [u8; 32] {
    let mut hasher = H::new();

    for part in parts {
        hasher.update(part);
    }

    hasher.update(&[tag as u8]);

    hasher.finalize()
}

/// Returns the root over the power-of-two leaf hashes `leaves`,
/// reducing them in place; errors on any other count.
pub fn subtree_root<H: Hasher>(leaves: &mut [[u8; 32]]) -> Result<[u8; 32]> {
    if !leaves.len().is_power_of_two() {
        return Err(Error::SubtreeWidth {
            leaves: leaves.len(),
        });
    }

    let mut preimages = [0u8; BATCH * NODE_BYTES];
    let mut digests = [[0u8; 32]; BATCH];

    let mut width = leaves.len();
    while width > 1 {
        let parents = width / 2;

        for first in (0..parents).step_by(BATCH) {
            let count = BATCH.min(parents - first);

            for (k, preimage) in preimages
                .as_chunks_mut::<NODE_BYTES>()
                .0
                .iter_mut()
                .take(count)
                .enumerate()
            {
                preimage[..32].copy_from_slice(&leaves[2 * (first + k)]);
                preimage[32..64].copy_from_slice(&leaves[2 * (first + k) + 1]);
                preimage[64] = Tag::Node as u8;
            }

            H::digest_many(
                &preimages[..count * NODE_BYTES],
                NODE_BYTES,
                &mut digests[..count],
            );

            leaves[first..first + count].copy_from_slice(&digests[..count]);
        }

        width = parents;
    }

    Ok(leaves[0])
}

/// One tree layer, each parent the node of its two children.
fn hash_layer<H: Hasher>(
    children: &[MaybeUninit<[u8; 32]>],
    parents: &mut [MaybeUninit<[u8; 32]>],
) {
    let pairs = children.as_chunks::<2>().0.iter().map(|[lo, hi]| {
        // SAFETY: a layer is hashed only after the one below
        // it is written, the leaves by `build_layers`' caller.
        unsafe { (lo.assume_init_ref(), hi.assume_init_ref()) }
    });

    hash_pairs::<H>(pairs, |k, digest| {
        parents[k].write(digest);
    });
}

fn subtree_nodes<H: Hasher>(span: usize, leaves: impl FnOnce(&mut [[u8; 32]])) -> Vec<[u8; 32]> {
    let mut nodes = vec![[0u8; 32]; 2 * span - 1];
    leaves(&mut nodes[span - 1..]);

    let mut width = span;
    while width > 1 {
        let (upper, lower) = nodes.split_at_mut(width - 1);
        let parents = &mut upper[width / 2 - 1..];

        let pairs = lower[..width]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|[left, right]| (left, right));

        hash_pairs::<H>(pairs, |k, digest| parents[k] = digest);

        width /= 2;
    }

    nodes
}

fn hash_pairs<'a, H: Hasher>(
    pairs: impl Iterator<Item = (&'a [u8; 32], &'a [u8; 32])>,
    mut write: impl FnMut(usize, [u8; 32]),
) {
    let mut preimages = [0u8; BATCH * NODE_BYTES];
    let mut digests = [[0u8; 32]; BATCH];

    let mut filled = 0;
    let mut written = 0;

    let mut flush = |filled: usize, written: usize, preimages: &[u8], digests: &mut [[u8; 32]]| {
        H::digest_many(
            &preimages[..filled * NODE_BYTES],
            NODE_BYTES,
            &mut digests[..filled],
        );

        for (k, digest) in digests[..filled].iter().enumerate() {
            write(written + k, *digest);
        }
    };

    for (left, right) in pairs {
        let preimage = &mut preimages[filled * NODE_BYTES..(filled + 1) * NODE_BYTES];
        preimage[..32].copy_from_slice(left);
        preimage[32..64].copy_from_slice(right);
        preimage[64] = Tag::Node as u8;

        filled += 1;

        if filled == BATCH {
            flush(filled, written, &preimages, &mut digests);

            written += filled;
            filled = 0;
        }
    }

    flush(filled, written, &preimages, &mut digests);
}

/// `H(left ‖ right ‖ 0x01)`, an internal node of [`MerkleTree`].
fn hash_node<H: Hasher>(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut node = [0u8; 65];
    node[..32].copy_from_slice(left);
    node[32..64].copy_from_slice(right);
    node[64] = Tag::Node as u8;

    H::digest(&node)
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

    fn layer_nodes(leaves: &[[u8; 32]], level: usize) -> Vec<[u8; 32]> {
        let mut layer = leaves.to_vec();
        while layer.len() > 1 << level {
            layer = layer
                .as_chunks::<2>()
                .0
                .iter()
                .map(|[left, right]| hash_node::<H>(left, right))
                .collect();
        }

        layer
    }

    fn check_tags<H: Hasher>() {
        let right: [u8; 32] = core::array::from_fn(|i| 0x80 | i as u8);

        for first in [0x00, 0x02] {
            let mut left: [u8; 32] = core::array::from_fn(|i| i as u8);
            left[0] = first;

            let payload = [left, right].concat();

            let tagged = |tag: u8| H::digest(&[&payload[..], &[tag][..]].concat());

            assert_eq!(hash_node::<H>(&left, &right), tagged(0x01));

            for (tag, byte) in [(Tag::PoolLeaf, 0x02), (Tag::RoundLeaf, 0x04)] {
                assert_eq!(
                    hash_symbol_leaf::<H>([&left[..], &right[..]], tag),
                    tagged(byte)
                );
            }
        }
    }

    fn check_symbol_leaves<H: Hasher>() {
        let bytes: Vec<u8> = (0..1u32 << 18)
            .map(|i| (i.wrapping_mul(0x9E37_79B9) >> 11) as u8)
            .collect();

        for runs in [
            &[1usize][..],
            &[16, 4],
            &[134],
            &[135],
            &[8, 64, 16, 40],
            &[1023],
            &[1024],
        ] {
            for count in [1usize, 3, 64, 65, 202] {
                let members: Vec<(&[u8], usize)> = runs
                    .iter()
                    .enumerate()
                    .map(|(m, &run)| (&bytes[977 * m..977 * m + count * run], run))
                    .collect();

                let mut leaves = vec![[0u8; 32]; count];
                hash_symbol_leaves::<H>(&members, Tag::PoolLeaf, &mut leaves, &mut Vec::new());

                for (j, leaf) in leaves.iter().enumerate() {
                    let parts = members.iter().map(|&(b, run)| &b[j * run..(j + 1) * run]);

                    assert_eq!(
                        *leaf,
                        hash_symbol_leaf::<H>(parts, Tag::PoolLeaf),
                        "runs {runs:?}, {count} leaves, leaf {j}"
                    );
                }
            }
        }
    }

    fn check_layers<H: Hasher>() {
        for count in [1usize, 2, 3, 64, 65, 129, 1024, 1025, 4097] {
            let leaves: Vec<[u8; 32]> = (0..count as u32)
                .map(|i| H::digest(&i.to_le_bytes()))
                .collect();

            let mut padded = leaves.clone();
            padded.resize(count.next_power_of_two(), [0u8; 32]);

            let mut layer = padded.clone();
            while layer.len() > 1 {
                layer = layer
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|[left, right]| hash_node::<H>(left, right))
                    .collect();
            }

            let tree = MerkleTree::<Block128, H>::new(&leaves);

            assert_eq!(tree.root(), layer[0], "{count} leaves");
            assert_eq!(
                subtree_root::<H>(&mut padded).unwrap(),
                layer[0],
                "{count} leaves"
            );
        }
    }

    fn pruned(leaf_hashes: &[[u8; 32]], floor: usize) -> MerkleTree<Block128, H> {
        let roots: Vec<[u8; 32]> = leaf_hashes
            .chunks(1 << floor)
            .map(|chunk| subtree_root::<H>(&mut chunk.to_vec()).unwrap())
            .collect();

        MerkleTree::from_subtree_roots(leaf_hashes.len(), floor, &roots).unwrap()
    }

    fn random_indices(state: &mut u64, count: usize, num_leaves: usize) -> Vec<usize> {
        (0..count)
            .map(|_| {
                *state ^= *state << 13;
                *state ^= *state >> 7;
                *state ^= *state << 17;

                *state as usize % num_leaves
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
        let leaf_layer = tree_stream.leaves_mut(leaf_offset).unwrap();

        for (i, slot) in leaf_layer.iter_mut().enumerate() {
            if i < leaves.len() {
                slot.write(leaves[i]);
            } else {
                slot.write([0u8; 32]);
            }
        }

        tree_stream.build_layers(leaf_offset).unwrap();

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
            let leaf_layer = tree_stream.leaves_mut(leaf_offset).unwrap();

            for (i, slot) in leaf_layer.iter_mut().enumerate() {
                if i < leaves.len() {
                    slot.write(leaves[i]);
                } else {
                    slot.write([0u8; 32]);
                }
            }

            tree_stream.build_layers(leaf_offset).unwrap();

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
            let proof = tree.prove_batch(query, query.len()).unwrap();
            let leaves = batch_leaves(query, &leaf_hashes);

            assert!(
                MerkleTree::<Block128, H>::verify_batch(&root, 64, query.len(), &leaves, &proof),
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
        let proof = tree.prove_batch(&all, all.len()).unwrap();

        assert_eq!(proof, Multiproof::default());

        let leaves = batch_leaves(&all, &leaf_hashes);

        assert!(MerkleTree::<Block128, H>::verify_batch(
            &tree.root(),
            32,
            all.len(),
            &leaves,
            &proof
        ));
    }

    #[test]
    fn batch_proof_single_leaf_sibling_count_is_depth() {
        let leaf_hashes: Vec<[u8; 32]> = (0..64u32).map(|i| hash_bytes(&i.to_le_bytes())).collect();
        let tree = MerkleTree::<Block128, H>::new(&leaf_hashes);

        let proof = tree.prove_batch(&[42], 1).unwrap();

        assert_eq!(
            proof.siblings.len(),
            6,
            "single-leaf octopus is a full path"
        );
        assert_eq!(proof.siblings, tree.prove(42).unwrap());
        assert!(proof.unreached.is_empty());
    }

    #[test]
    fn verify_batch_rejects_tampering() {
        let leaf_hashes: Vec<[u8; 32]> = (0..64u32).map(|i| hash_bytes(&i.to_le_bytes())).collect();
        let tree = MerkleTree::<Block128, H>::new(&leaf_hashes);
        let root = tree.root();

        let query = [3usize, 8, 8, 20, 55];
        let q = query.len();
        let proof = tree.prove_batch(&query, q).unwrap();
        let leaves = batch_leaves(&query, &leaf_hashes);

        assert!(MerkleTree::<Block128, H>::verify_batch(
            &root, 64, q, &leaves, &proof
        ));

        let mut wrong_leaf = leaves.clone();
        wrong_leaf[1].1 = hash_bytes(b"forged");

        assert!(!MerkleTree::<Block128, H>::verify_batch(
            &root,
            64,
            q,
            &wrong_leaf,
            &proof
        ));

        let mut extra = proof.clone();
        extra.siblings.push([0u8; 32]);

        assert!(
            !MerkleTree::<Block128, H>::verify_batch(&root, 64, q, &leaves, &extra),
            "unconsumed extra sibling must reject"
        );

        let mut missing = proof.clone();
        missing.siblings.pop();

        assert!(
            !MerkleTree::<Block128, H>::verify_batch(&root, 64, q, &leaves, &missing),
            "missing sibling must reject"
        );

        assert!(!MerkleTree::<Block128, H>::verify_batch(
            &[9u8; 32], 64, q, &leaves, &proof
        ));

        let unsorted = vec![leaves[2], leaves[0], leaves[1], leaves[3]];

        assert!(
            !MerkleTree::<Block128, H>::verify_batch(&root, 64, q, &unsorted, &proof),
            "unsorted leaves must reject"
        );

        let dup = vec![leaves[0], leaves[0], leaves[1]];

        assert!(
            !MerkleTree::<Block128, H>::verify_batch(&root, 64, q, &dup, &proof),
            "duplicate leaf indices must reject"
        );

        assert!(
            !MerkleTree::<Block128, H>::verify_batch(&root, 63, q, &leaves, &proof),
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
        let proof = tree.prove_batch(&query, query.len()).unwrap();
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
            query.len(),
            &leaves,
            &proof
        ));
    }

    #[test]
    fn prove_batch_rejects_oob_index() {
        let leaf_hashes: Vec<[u8; 32]> = (0..8u32).map(|i| hash_bytes(&i.to_le_bytes())).collect();
        let tree = MerkleTree::<Block128, H>::new(&leaf_hashes);

        assert!(tree.prove_batch(&[0, 8], 2).is_err());
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

    #[cfg(feature = "blake3")]
    #[test]
    fn blake3_preimages_end_in_their_tags() {
        check_tags::<crate::blake3::Blake3Hasher>();
    }

    #[cfg(feature = "sha2")]
    #[test]
    fn sha256_preimages_end_in_their_tags() {
        check_tags::<crate::sha256::Sha256Hasher>();
    }

    #[cfg(feature = "sha3")]
    #[test]
    fn sha3_preimages_end_in_their_tags() {
        check_tags::<crate::sha3::Sha3_256Hasher>();
    }

    #[test]
    fn layer_log_is_ceil_log2_capped_at_depth() {
        for (queries, depth, want) in [
            (0, 15, 0),
            (1, 15, 0),
            (2, 15, 1),
            (5, 15, 3),
            (121, 13, 7),
            (153, 16, 8),
            (256, 15, 8),
            (287, 15, 9),
            (513, 15, 10),
            (287, 6, 6),
            (4, 0, 0),
        ] {
            assert_eq!(
                layer_log(queries, depth),
                want,
                "{queries} queries, depth {depth}"
            );
        }
    }

    #[test]
    fn unreached_section_is_layer_minus_reached_positions() {
        let leaf_hashes: Vec<[u8; 32]> =
            (0..256u32).map(|i| hash_bytes(&i.to_le_bytes())).collect();
        let tree = MerkleTree::<Block128, H>::new(&leaf_hashes);

        for query in [
            &[200usize][..],
            &[0, 255],
            &[9, 9, 9, 9, 9],
            &[3, 8, 8, 20, 55, 130, 131, 254],
            &[1, 17, 33, 49, 65, 81, 97, 113, 129, 145, 161, 177, 193],
        ] {
            let level = layer_log(query.len(), 8);
            let layer = layer_nodes(&leaf_hashes, level);

            let expected: Vec<[u8; 32]> = (0..1usize << level)
                .filter(|&p| query.iter().all(|&idx| idx >> (8 - level) != p))
                .map(|p| layer[p])
                .collect();

            let proof = tree.prove_batch(query, query.len()).unwrap();

            assert_eq!(proof.unreached, expected, "{query:?}");
            assert!(MerkleTree::<Block128, H>::verify_batch(
                &tree.root(),
                256,
                query.len(),
                &batch_leaves(query, &leaf_hashes),
                &proof
            ));
        }
    }

    #[test]
    fn layer_at_depth_ships_unopened_leaves() {
        let leaf_hashes: Vec<[u8; 32]> = (0..16u32).map(|i| hash_bytes(&i.to_le_bytes())).collect();
        let tree = MerkleTree::<Block128, H>::new(&leaf_hashes);

        let query = [2usize, 7, 11].repeat(6);
        let proof = tree.prove_batch(&query, query.len()).unwrap();

        let unopened: Vec<[u8; 32]> = (0..16)
            .filter(|i| ![2, 7, 11].contains(i))
            .map(|i| leaf_hashes[i])
            .collect();

        assert_eq!(layer_log(query.len(), 4), 4);
        assert!(proof.siblings.is_empty());
        assert_eq!(proof.unreached, unopened);
        assert!(MerkleTree::<Block128, H>::verify_batch(
            &tree.root(),
            16,
            query.len(),
            &batch_leaves(&query, &leaf_hashes),
            &proof
        ));
    }

    #[test]
    fn verify_batch_rejects_forged_unreached_section() {
        let leaf_hashes: Vec<[u8; 32]> = (0..64u32).map(|i| hash_bytes(&i.to_le_bytes())).collect();
        let tree = MerkleTree::<Block128, H>::new(&leaf_hashes);
        let root = tree.root();

        let query = [3usize, 8, 8, 20, 55];
        let q = query.len();
        let proof = tree.prove_batch(&query, q).unwrap();
        let leaves = batch_leaves(&query, &leaf_hashes);

        assert_eq!(proof.unreached.len(), 4);

        let mut flipped = proof.clone();
        flipped.unreached[2][0] ^= 1;

        let mut extra = proof.clone();
        extra.unreached.push([0u8; 32]);

        let mut missing = proof.clone();
        missing.unreached.pop();

        let mut swapped = proof.clone();
        swapped.unreached.swap(0, 1);

        for forged in [flipped, extra, missing, swapped] {
            assert!(!MerkleTree::<Block128, H>::verify_batch(
                &root, 64, q, &leaves, &forged
            ));
        }

        for other in [4, 9] {
            assert!(!MerkleTree::<Block128, H>::verify_batch(
                &root, 64, other, &leaves, &proof
            ));
        }
    }

    #[test]
    fn multiproof_roundtrips_across_depths_and_draw_counts() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for depth in 0..=9 {
            let num_leaves = 1usize << depth;
            let leaf_hashes: Vec<[u8; 32]> = (0..num_leaves as u32)
                .map(|i| hash_bytes(&i.to_le_bytes()))
                .collect();
            let tree = MerkleTree::<Block128, H>::new(&leaf_hashes);

            for draws in [1usize, 2, 3, 7, 64, 287, 600] {
                let query: Vec<usize> = (0..draws)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;

                        state as usize % num_leaves
                    })
                    .collect();

                let level = layer_log(draws, depth);

                let mut reached: Vec<usize> = query.iter().map(|i| i >> (depth - level)).collect();
                reached.sort_unstable();
                reached.dedup();

                let proof = tree.prove_batch(&query, draws).unwrap();

                assert_eq!(proof.unreached.len(), (1 << level) - reached.len());
                assert!(
                    MerkleTree::<Block128, H>::verify_batch(
                        &tree.root(),
                        num_leaves,
                        draws,
                        &batch_leaves(&query, &leaf_hashes),
                        &proof
                    ),
                    "depth {depth}, {draws} draws"
                );
            }
        }
    }

    #[test]
    fn symbol_leaves_concatenate_member_runs_under_their_tag() {
        let wide: Vec<u8> = (0..64u8).collect();
        let narrow: Vec<u8> = (100..116u8).collect();
        let members = [(wide.as_slice(), 16), (narrow.as_slice(), 4)];

        let mut leaves = vec![[0u8; 32]; 4];
        hash_symbol_leaves::<H>(&members, Tag::PoolLeaf, &mut leaves, &mut Vec::new());

        for (j, leaf) in leaves.iter().enumerate() {
            let mut preimage = wide[16 * j..16 * (j + 1)].to_vec();
            preimage.extend_from_slice(&narrow[4 * j..4 * (j + 1)]);
            preimage.push(Tag::PoolLeaf as u8);

            assert_eq!(*leaf, H::digest(&preimage));
        }

        let parts = [&wide[..16], &narrow[..4]];

        assert_ne!(
            hash_symbol_leaf::<H>(parts, Tag::PoolLeaf),
            hash_symbol_leaf::<H>(parts, Tag::RoundLeaf)
        );
    }

    #[cfg(feature = "blake3")]
    #[test]
    fn blake3_symbol_leaves_match_per_leaf_hash() {
        check_symbol_leaves::<crate::blake3::Blake3Hasher>();
    }

    #[cfg(feature = "sha2")]
    #[test]
    fn sha256_symbol_leaves_match_per_leaf_hash() {
        check_symbol_leaves::<crate::sha256::Sha256Hasher>();
    }

    #[cfg(feature = "sha3")]
    #[test]
    fn sha3_symbol_leaves_match_per_leaf_hash() {
        check_symbol_leaves::<crate::sha3::Sha3_256Hasher>();
    }

    #[cfg(feature = "blake3")]
    #[test]
    fn blake3_tree_root_matches_per_node_hash() {
        check_layers::<crate::blake3::Blake3Hasher>();
    }

    #[cfg(feature = "sha2")]
    #[test]
    fn sha256_tree_root_matches_per_node_hash() {
        check_layers::<crate::sha256::Sha256Hasher>();
    }

    #[cfg(feature = "sha3")]
    #[test]
    fn sha3_tree_root_matches_per_node_hash() {
        check_layers::<crate::sha3::Sha3_256Hasher>();
    }

    #[test]
    fn pruned_tree_root_and_multiproof_match_full_tree() {
        let depth = 10;
        let leaf_hashes: Vec<[u8; 32]> = (0..1u32 << depth)
            .map(|i| hash_bytes(&i.to_le_bytes()))
            .collect();

        let full = MerkleTree::<Block128, H>::new(&leaf_hashes);

        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for floor in [0, 1, 6, depth] {
            let tree = pruned(&leaf_hashes, floor);
            let span = 1 << floor;

            assert_eq!(tree.root(), full.root(), "floor {floor}");

            for draws in [1usize, 2, 3, 15, 16, 17, 64, 287, 2000] {
                let query = random_indices(&mut state, draws, leaf_hashes.len());

                let proof = tree.prove_batch_with(&query, draws, |s, hashes| {
                    hashes.copy_from_slice(&leaf_hashes[s * span..(s + 1) * span]);
                });

                assert_eq!(
                    proof,
                    full.prove_batch(&query, draws),
                    "floor {floor}, {draws} draws"
                );
            }
        }
    }

    #[test]
    fn prove_batch_with_rejects_wrong_subtree_hashes() {
        let leaf_hashes: Vec<[u8; 32]> =
            (0..1024u32).map(|i| hash_bytes(&i.to_le_bytes())).collect();

        let full = MerkleTree::<Block128, H>::new(&leaf_hashes);
        let tree = pruned(&leaf_hashes, 6);

        let forged = |s: usize, hashes: &mut [[u8; 32]]| {
            hashes.copy_from_slice(&leaf_hashes[64 * s..64 * (s + 1)]);

            if s == 3 {
                hashes[17][0] ^= 1;
            }
        };

        assert_eq!(
            tree.prove_batch_with(&[64 * 3 + 5], 1, forged),
            Err(Error::SubtreeMismatch { subtree: 3 })
        );
        assert_eq!(
            tree.prove_batch_with(&[5], 17, forged),
            Err(Error::SubtreeMismatch { subtree: 3 })
        );
        assert_eq!(
            tree.prove_batch_with(&[5], 1, forged),
            full.prove_batch(&[5], 1)
        );
    }

    #[test]
    fn pruned_tree_rejects_prove_and_prove_batch() {
        let leaf_hashes: Vec<[u8; 32]> = (0..64u32).map(|i| hash_bytes(&i.to_le_bytes())).collect();
        let tree = pruned(&leaf_hashes, 2);

        assert_eq!(tree.prove(5), Err(Error::Pruned { floor: 2 }));
        assert_eq!(
            tree.prove_batch(&[5, 9], 2),
            Err(Error::Pruned { floor: 2 })
        );
    }
}
