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

use core::hint::black_box;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hekate_crypto::merkle::MerkleTree;
use hekate_crypto::{default_hasher_name, DefaultHasher, Hasher};
use hekate_math::Block128;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Duration;

type H = DefaultHasher;
type F = Block128;

#[inline(always)]
fn hash_leaf(payload: &[u8]) -> [u8; 32] {
    let mut h = H::new();
    h.update(&[0u8]);
    h.update(&(payload.len() as u64).to_le_bytes());
    h.update(payload);

    h.finalize()
}

#[inline(always)]
fn hash_node(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut h = H::new();
    h.update(&[1u8]);
    h.update(left);
    h.update(right);

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

    group.throughput(Throughput::Bytes((1 + 32 + 32) as u64));
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
                    let leaf_layer = tree.leaves_mut(leaf_offset);

                    for (i, slot) in leaf_layer.iter_mut().enumerate() {
                        slot.write(leaves[i]);
                    }

                    tree.build_layers(leaf_offset);

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

fn merkle_benches(c: &mut Criterion) {
    #[cfg(feature = "parallel")]
    {
        eprintln!("rayon threads: {}", rayon::current_num_threads());
    }

    bench_merkle_hashing(c);
    bench_merkle_build(c);
    bench_merkle_prove_verify(c);
}

criterion_group!(benches, merkle_benches);
criterion_main!(benches);
