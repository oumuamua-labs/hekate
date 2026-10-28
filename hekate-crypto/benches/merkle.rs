// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use core::hint::black_box;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use hekate_crypto::merkle::{MerkleTree, Tag};
#[cfg(feature = "parallel")]
use hekate_crypto::merkle::{hash_symbol_leaf, hash_symbol_leaves};
use hekate_crypto::{DefaultHasher, Hasher, default_hasher_name};
use hekate_math::Block128;
use rand::{Rng, SeedableRng, rngs::StdRng};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::time::Duration;

#[cfg(feature = "parallel")]
const BATCH_VARS: usize = 22;

#[cfg(feature = "parallel")]
const BATCH: usize = 64;

#[cfg(feature = "parallel")]
const LEAF_RUN: usize = 128;

#[cfg(feature = "parallel")]
const NODE_BYTES: usize = 65;

type H = DefaultHasher;
type F = Block128;

#[inline(always)]
fn hash_leaf(payload: &[u8]) -> [u8; 32] {
    let mut h = H::new();
    h.update(payload);
    h.update(&[Tag::PoolLeaf as u8]);

    h.finalize()
}

#[inline(always)]
fn hash_node(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut h = H::new();
    h.update(left);
    h.update(right);
    h.update(&[Tag::Node as u8]);

    h.finalize()
}

fn build_leaves(num_leaves: usize) -> Vec<[u8; 32]> {
    let mut rng = StdRng::seed_from_u64(42);

    // We intentionally keep the per-leaf
    // payload small here. The Merkle tree
    // itself only sees 32-byte leaf hashes.
    let mut payload = [0u8; 32];

    let mut leaves = Vec::with_capacity(num_leaves);
    for _ in 0..num_leaves {
        rng.fill_bytes(&mut payload);
        leaves.push(hash_leaf(&payload));
    }

    leaves
}

fn bench_merkle_hashing(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("Merkle/{} hashing", default_hasher_name()));
    group.measurement_time(Duration::from_secs(10));

    for &payload_len in &[32usize, 64, 256, 1024, 4096] {
        group.throughput(Throughput::Bytes(payload_len as u64));

        group.bench_with_input(
            BenchmarkId::new("leaf_hash", payload_len),
            &payload_len,
            |b, &len| {
                let mut payload = vec![0u8; len];
                StdRng::seed_from_u64(42).fill_bytes(&mut payload);

                b.iter(|| black_box(hash_leaf(black_box(&payload))))
            },
        );
    }

    group.throughput(Throughput::Bytes((32 + 32 + 1) as u64));
    group.bench_function("node_hash", |b| {
        let left = hash_leaf(b"left");
        let right = hash_leaf(b"right");

        b.iter(|| black_box(hash_node(black_box(&left), black_box(&right))))
    });

    group.finish();
}

fn bench_merkle_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("Merkle/build");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    for &num_vars in &[10usize, 14, 18] {
        let num_leaves = 1usize << num_vars;
        let leaves = build_leaves(num_leaves);

        group.throughput(Throughput::Elements(num_leaves as u64));

        group.bench_with_input(BenchmarkId::new("new", num_vars), &num_vars, |b, _| {
            b.iter(|| {
                let tree = MerkleTree::<F, H>::new(black_box(&leaves));
                black_box(tree.root())
            })
        });

        group.bench_with_input(
            BenchmarkId::new("streaming", num_vars),
            &num_vars,
            |b, _| {
                b.iter(|| {
                    let (mut tree, leaf_offset) = MerkleTree::<F, H>::allocate_tree(num_leaves);
                    let leaf_layer = tree.leaves_mut(leaf_offset).unwrap();

                    for (i, slot) in leaf_layer.iter_mut().enumerate() {
                        slot.write(leaves[i]);
                    }

                    tree.build_layers(leaf_offset).unwrap();

                    black_box(tree.root())
                })
            },
        );
    }

    group.finish();
}

fn bench_merkle_prove_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("Merkle/prove+verify");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(15));

    for &num_vars in &[10usize, 14, 18] {
        let num_leaves = 1usize << num_vars;
        let leaves = build_leaves(num_leaves);
        let tree = MerkleTree::<F, H>::new(&leaves);
        let root = tree.root();

        let mut rng = StdRng::seed_from_u64(123);
        let idx = (rng.next_u64() as usize) & (num_leaves - 1);

        // Precompute one proof
        // for verify-only bench.
        let proof = tree.prove(idx).unwrap();

        group.bench_with_input(BenchmarkId::new("prove", num_vars), &num_vars, |b, _| {
            b.iter(|| black_box(tree.prove(black_box(idx)).unwrap()))
        });

        // Throughput:
        // number of sibling hashes checked.
        group.throughput(Throughput::Elements(proof.len() as u64));
        group.bench_with_input(BenchmarkId::new("verify", num_vars), &num_vars, |b, _| {
            b.iter(|| {
                black_box(MerkleTree::<F, H>::verify(
                    black_box(&root),
                    black_box(leaves[idx]),
                    black_box(idx),
                    black_box(&proof),
                ))
            })
        });
    }

    group.finish();
}

#[cfg(feature = "parallel")]
fn bench_batched_leaves(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("Merkle/{} leaves", default_hasher_name()));
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    let count = 1usize << BATCH_VARS;
    group.throughput(Throughput::Elements(count as u64));

    let mut symbols = vec![0u8; count * LEAF_RUN];
    StdRng::seed_from_u64(42).fill_bytes(&mut symbols);

    let mut leaves = vec![[0u8; 32]; count];

    group.bench_with_input(
        BenchmarkId::new("per_leaf", BATCH_VARS),
        &BATCH_VARS,
        |b, _| {
            b.iter(|| {
                leaves.par_iter_mut().enumerate().for_each(|(j, slot)| {
                    let run = &symbols[j * LEAF_RUN..(j + 1) * LEAF_RUN];
                    *slot = hash_symbol_leaf::<H>([run], Tag::PoolLeaf);
                });

                black_box(&leaves);
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("hash_symbol_leaves", BATCH_VARS),
        &BATCH_VARS,
        |b, _| {
            b.iter(|| {
                leaves
                    .par_chunks_mut(BATCH)
                    .zip(symbols.par_chunks(BATCH * LEAF_RUN))
                    .for_each_init(Vec::new, |preimages, (out, bytes)| {
                        hash_symbol_leaves::<H>(&[(bytes, LEAF_RUN)], Tag::PoolLeaf, out, preimages)
                    });

                black_box(&leaves);
            })
        },
    );

    group.finish();
}

#[cfg(feature = "parallel")]
fn bench_batched_nodes(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("Merkle/{} nodes", default_hasher_name()));
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    let count = 1usize << BATCH_VARS;
    group.throughput(Throughput::Elements(count as u64));

    let mut preimages = vec![0u8; count * NODE_BYTES];
    StdRng::seed_from_u64(42).fill_bytes(&mut preimages);

    let mut digests = vec![[0u8; 32]; count];

    group.bench_with_input(
        BenchmarkId::new("digest", BATCH_VARS),
        &BATCH_VARS,
        |b, _| {
            b.iter(|| {
                digests
                    .par_chunks_mut(BATCH)
                    .zip(preimages.par_chunks(BATCH * NODE_BYTES))
                    .for_each(|(out, msgs)| {
                        for (slot, msg) in out.iter_mut().zip(msgs.as_chunks::<NODE_BYTES>().0) {
                            *slot = H::digest(msg);
                        }
                    });

                black_box(&digests);
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("digest_many", BATCH_VARS),
        &BATCH_VARS,
        |b, _| {
            b.iter(|| {
                digests
                    .par_chunks_mut(BATCH)
                    .zip(preimages.par_chunks(BATCH * NODE_BYTES))
                    .for_each(|(out, msgs)| H::digest_many(msgs, NODE_BYTES, out));

                black_box(&digests);
            })
        },
    );

    group.finish();
}

fn merkle_benches(c: &mut Criterion) {
    #[cfg(feature = "parallel")]
    {
        eprintln!("rayon threads: {}", rayon::current_num_threads());
    }

    bench_merkle_hashing(c);
    bench_merkle_build(c);
    bench_merkle_prove_verify(c);

    #[cfg(feature = "parallel")]
    {
        bench_batched_leaves(c);
        bench_batched_nodes(c);
    }
}

criterion_group!(benches, merkle_benches);
criterion_main!(benches);
