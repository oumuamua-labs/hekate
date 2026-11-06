// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec;
use alloc::vec::Vec;
use hekate_core::config::MAX_FOLD_ARITY_LOG;
use hekate_core::errors;
use hekate_core::fold::{CodeField, FoldLayout, FoldTree, PolyKind, TreeKind};
use hekate_core::proofs::{FoldEvalProof, TreeOpening};
use hekate_core::tensor::TensorProduct;
use hekate_core::trace::{ColumnType, TraceCompatibleField};
use hekate_crypto::merkle::{MerkleTree, Tag, hash_symbol_leaf};
use hekate_crypto::transcript::Transcript;
use hekate_crypto::{Digest, Hasher};
use hekate_math::{BinaryFieldExtras, CantorBasis, Flat, HardwareField, TowerField};
use tracing::{instrument, warn};

const MAX_COSET: usize = 1 << MAX_FOLD_ARITY_LOG;

pub struct FoldClaims<'a, F, H: Hasher> {
    pub trace_root: &'a H::Digest,
    pub helper_root: Option<&'a H::Digest>,
    pub point: &'a [Flat<F>],
    pub masters: &'a [Flat<F>],
}

struct OpenedTree<'p> {
    shift: usize,
    distinct: Vec<usize>,
    leaf_bytes: usize,
    leaves: &'p [u8],
}

impl<'p> OpenedTree<'p> {
    fn leaf(&self, query: usize) -> errors::Result<&'p [u8]> {
        let slot = self
            .distinct
            .binary_search(&(query >> self.shift))
            .map_err(|_| malformed("query leaf missing from its opening"))?;

        Ok(&self.leaves[slot * self.leaf_bytes..(slot + 1) * self.leaf_bytes])
    }
}

struct Kernel<F> {
    basis: CantorBasis<F>,
}

impl<F> Kernel<F>
where
    F: TowerField + HardwareField + BinaryFieldExtras,
{
    fn fold_coset(
        &self,
        values: &[Flat<F>],
        base: usize,
        challenges: &[Flat<F>],
    ) -> errors::Result<Flat<F>> {
        let mut cur = [Flat::from_raw(F::ZERO); MAX_COSET];
        let mut len = values.len();
        let mut at = base;

        cur[..len].copy_from_slice(values);

        for &ch in challenges {
            len /= 2;

            for u in 0..len {
                let twiddle = self
                    .basis
                    .point(at + 2 * u)
                    .map_err(|_| malformed("fold twiddle outside the Cantor domain"))?;

                let f1 = cur[2 * u] + cur[2 * u + 1];
                let f0 = cur[2 * u] + f1 * twiddle;

                cur[u] = f0 + ch * (f0 + f1);
            }

            at >>= 1;
        }

        Ok(cur[0])
    }

    fn encode_final(&self, layer: &[Flat<F>], positions: &[usize]) -> errors::Result<Vec<Flat<F>>> {
        let zero = Flat::from_raw(F::ZERO);

        let mut scratch = vec![zero; layer.len() - 1];
        let mut out = vec![zero; positions.len()];

        self.basis
            .evaluate_at(layer, zero, positions, &mut scratch, &mut out)
            .map_err(|_| malformed("final layer encode outside the Cantor domain"))?;

        Ok(out)
    }
}

#[instrument(skip_all, level = "trace", name = "Fold::verify")]
pub fn verify<F, H>(
    layout: &FoldLayout,
    claims: FoldClaims<'_, F, H>,
    proof: &FoldEvalProof<F>,
    transcript: &mut Transcript<H>,
) -> errors::Result<bool>
where
    F: TowerField + HardwareField + BinaryFieldExtras + TraceCompatibleField,
    H: Hasher,
{
    let a = layout.arity_log();
    let rounds = layout.rounds();
    let stop = layout.stop_vars();
    let fold_vars = layout.num_vars() - stop;
    let polys = layout.polys();

    let symbol_bytes = F::BITS / 8;
    let digest_bytes = H::Digest::BYTES;

    if claims.masters.len() != polys.len() || claims.point.len() != layout.num_vars() {
        return Err(malformed("one master per polynomial at the batching point"));
    }

    if proof.fold_rounds.len() != 2 * fold_vars
        || proof.round_roots.len() != (rounds - 1) * digest_bytes
        || proof.final_layer.len() != 1 << stop
        || proof.openings.len() != layout.trees().len()
    {
        return Err(malformed("fold proof does not match its layout"));
    }

    let round_root = |c: usize| &proof.round_roots[(c - 1) * digest_bytes..c * digest_bytes];

    let kernel = Kernel {
        basis: CantorBasis::new(layout.num_vars() + layout.log_inv_rate())
            .map_err(|_| malformed("fold domain beyond the Cantor basis"))?,
    };

    let lambda = transcript
        .challenge_field::<F>(b"fold_lambda")?
        .to_hardware();

    let mut lambdas = Vec::with_capacity(polys.len());
    let mut power = lambda;

    for _ in 0..polys.len() {
        lambdas.push(power);

        power *= lambda;
    }

    let mut claim = Flat::from_raw(F::ZERO);
    for (&lambda_k, &master) in lambdas.iter().zip(claims.masters) {
        claim += lambda_k * master;
    }

    let mut challenges = Vec::with_capacity(fold_vars);
    for (j, coeffs) in proof.fold_rounds.as_chunks::<2>().0.iter().enumerate() {
        transcript.append_field_list(b"fold_round", coeffs);

        let g1 = coeffs[0].to_hardware();
        let c2 = coeffs[1].to_hardware();

        let r = transcript
            .challenge_field::<F>(b"fold_challenge")?
            .to_hardware();

        let c0 = claim + g1;
        let c1 = g1 + c0 + c2;

        claim = c0 + r * (c1 + c2 * r);

        challenges.push(r);

        if (j + 1) % a == 0 {
            match (j + 1) / a {
                c if c < rounds => {
                    transcript.append_message(b"fold_round_root", round_root(c));
                }
                _ => transcript.append_field_list(b"fold_final_layer", &proof.final_layer),
            }
        }
    }

    let final_layer: Vec<Flat<F>> = proof.final_layer.iter().map(|v| v.to_hardware()).collect();

    let head = TensorProduct::evaluate_eq_slice(&challenges, &claims.point[..fold_vars]);
    let tail = mle_at(&final_layer, &claims.point[fold_vars..]);

    if claim != head * tail {
        warn!("fold sumcheck final claim disagrees with the final layer");
        return Ok(false);
    }

    let queries = transcript.challenge_indices(
        b"fold_query",
        layout.num_queries(),
        1 << (layout.num_vars() + layout.log_inv_rate() - a),
    )?;

    let mut poly_trees = Vec::with_capacity(2);
    let mut round_trees = Vec::with_capacity(rounds - 1);

    for (tree, opening) in layout.trees().iter().zip(&proof.openings) {
        let (root, tag) = match tree.kind {
            TreeKind::Polys(PolyKind::Trace) => (*claims.trace_root, Tag::PoolLeaf),
            TreeKind::Polys(PolyKind::Helper | PolyKind::Mask) => (
                *claims
                    .helper_root
                    .ok_or(malformed("helper tree without a root"))?,
                Tag::PoolLeaf,
            ),
            TreeKind::Round(c) => (
                H::Digest::from_slice(round_root(c)).ok_or(malformed("round root width"))?,
                Tag::RoundLeaf,
            ),
        };

        if matches!(tree.kind, TreeKind::Round(_)) && tree.leaf_bytes != symbol_bytes << a {
            return Err(malformed("round tree leaves do not hold the fold field"));
        }

        let opened = open_tree::<F, H>(tree, opening, &root, tag, a, &queries)?;

        match tree.kind {
            TreeKind::Polys(_) => poly_trees.push((tree, opened)),
            TreeKind::Round(_) => round_trees.push(opened),
        }
    }

    let mut final_positions: Vec<usize> =
        queries.iter().map(|&q| q >> (a * (rounds - 1))).collect();

    final_positions.sort_unstable();
    final_positions.dedup();

    let final_code = kernel.encode_final(&final_layer, &final_positions)?;

    let coset_len = 1 << a;

    for (query, &q) in queries.iter().enumerate() {
        let mut coset = [Flat::from_raw(F::ZERO); MAX_COSET];
        let mut joined = Flat::from_raw(F::ZERO);

        for (tree, opened) in &poly_trees {
            let leaf = opened.leaf(q)?;

            let mut at = 0;
            for &k in &tree.polys {
                let fold = &polys[k];
                let run = 1 << fold.leaf_vars;
                let width = fold.field.bytes();
                let symbols = leaf[at..at + run * width].chunks_exact(width);

                match fold.join_round {
                    0 => {
                        for (slot, bytes) in coset.iter_mut().zip(symbols) {
                            *slot += lambdas[k] * symbol::<F>(fold.field, bytes);
                        }
                    }
                    1 => {
                        let mut values = [Flat::from_raw(F::ZERO); MAX_COSET];
                        for (slot, bytes) in values.iter_mut().zip(symbols) {
                            *slot = symbol::<F>(fold.field, bytes);
                        }

                        let p = fold.prefold_vars;
                        let prefolded =
                            kernel.fold_coset(&values[..run], q << p, &challenges[a - p..a])?;

                        joined += lambdas[k] * prefolded;
                    }
                    _ => return Err(malformed("polynomial joins past the first round")),
                }

                at += run * width;
            }
        }

        let mut value = kernel.fold_coset(&coset[..coset_len], q << a, &challenges[..a])? + joined;
        let mut pos = q;

        for (c, opened) in round_trees.iter().enumerate().map(|(i, t)| (i + 1, t)) {
            let mut word = [Flat::from_raw(F::ZERO); MAX_COSET];
            for (slot, bytes) in word
                .iter_mut()
                .zip(opened.leaf(q)?.chunks_exact(symbol_bytes))
            {
                *slot = fold_symbol::<F>(bytes)?;
            }

            if word[pos & (coset_len - 1)] != value {
                warn!(
                    round = c,
                    query, "fold value disagrees with the committed round"
                );
                return Ok(false);
            }

            value = kernel.fold_coset(
                &word[..coset_len],
                (pos >> a) << a,
                &challenges[c * a..(c + 1) * a],
            )?;

            pos >>= a;
        }

        let slot = final_positions
            .binary_search(&pos)
            .map_err(|_| malformed("query path missing from the final positions"))?;

        if final_code[slot] != value {
            warn!(
                round = rounds,
                query, "fold value disagrees with the final layer"
            );
            return Ok(false);
        }
    }

    Ok(true)
}

fn open_tree<'p, F, H>(
    tree: &FoldTree,
    opening: &'p TreeOpening,
    root: &H::Digest,
    tag: Tag,
    arity_log: usize,
    queries: &[usize],
) -> errors::Result<OpenedTree<'p>>
where
    F: TowerField,
    H: Hasher,
{
    let shift = arity_log * (tree.open_round - 1);

    let mut distinct: Vec<usize> = queries.iter().map(|&q| q >> shift).collect();

    distinct.sort_unstable();
    distinct.dedup();

    let leaf_bytes = tree.leaf_bytes;

    if opening.leaves.len() != distinct.len() * leaf_bytes {
        return Err(malformed(
            "opened leaves do not match the tree's leaf width",
        ));
    }

    let hashes: Vec<(usize, H::Digest)> = distinct
        .iter()
        .zip(opening.leaves.chunks_exact(leaf_bytes))
        .map(|(&index, leaf)| (index, hash_symbol_leaf::<H>([leaf], tag)))
        .collect();

    let accepted = MerkleTree::<F, H>::verify_batch(
        root,
        1 << tree.depth,
        queries.len(),
        &hashes,
        &opening.multiproof,
    );

    if !accepted {
        warn!(kind = ?tree.kind, "fold tree multiproof rejected");
        return Err(malformed("fold tree multiproof rejected"));
    }

    Ok(OpenedTree {
        shift,
        distinct,
        leaf_bytes,
        leaves: &opening.leaves,
    })
}

fn mle_at<F>(values: &[Flat<F>], point: &[Flat<F>]) -> Flat<F>
where
    F: TowerField + HardwareField,
{
    let mut cur = values.to_vec();
    for &x in point {
        let half = cur.len() / 2;
        for u in 0..half {
            cur[u] = cur[2 * u] + (cur[2 * u] + cur[2 * u + 1]) * x;
        }

        cur.truncate(half);
    }

    cur[0]
}

fn fold_symbol<F: TowerField + HardwareField>(bytes: &[u8]) -> errors::Result<Flat<F>> {
    F::deserialize(bytes)
        .map(Flat::from_raw)
        .map_err(|()| malformed("round symbol outside its byte form"))
}

fn symbol<F: TraceCompatibleField>(field: CodeField, bytes: &[u8]) -> Flat<F> {
    let cells = match field {
        CodeField::B32 => ColumnType::B32,
        CodeField::B64 => ColumnType::B64,
        CodeField::B128 => ColumnType::B128,
    };

    cells.parse_from_bytes(bytes)
}

fn malformed(message: &'static str) -> errors::Error {
    errors::Error::Protocol {
        protocol: "fold",
        message,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hekate_math::{
        AdditiveFft, Block32, Block64, Block128, Block256, CanonicalSerialize, FlatPromote,
    };

    type F = Block256;

    fn values(seed: u128, n: usize) -> Vec<Flat<F>> {
        let mut state = seed;

        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;

                Flat::from_raw(Block256::new(
                    Block128(state),
                    Block128(state.rotate_left(64) ^ 0x5DEE_CE66),
                ))
            })
            .collect()
    }

    fn encode(message: &[Flat<F>], log_len: usize) -> Vec<Flat<F>> {
        let mut code = vec![Flat::from_raw(F::ZERO); 1 << log_len];
        code[..message.len()].copy_from_slice(message);

        AdditiveFft::<F>::new(log_len as u32)
            .unwrap()
            .forward_scalar(&mut code)
            .unwrap();

        code
    }

    fn kernel(dim: usize) -> Kernel<F> {
        Kernel {
            basis: CantorBasis::new(dim).unwrap(),
        }
    }

    fn narrow_code_matches_wide<S>(field: CodeField, seed: u128)
    where
        S: TowerField + HardwareField + BinaryFieldExtras,
        F: FlatPromote<S>,
    {
        let (vars, log_len) = (6, 8);

        let message: Vec<Flat<S>> = values(seed, 1 << vars)
            .iter()
            .map(|v| {
                let mut bytes = [0u8; 32];
                bytes.copy_from_slice(&v.into_raw().to_bytes());

                Flat::from_raw(S::from_uniform_bytes(&bytes))
            })
            .collect();

        let mut narrow = vec![Flat::from_raw(S::ZERO); 1 << log_len];
        narrow[..message.len()].copy_from_slice(&message);

        AdditiveFft::<S>::new(log_len as u32)
            .unwrap()
            .forward_scalar(&mut narrow)
            .unwrap();

        let lifted: Vec<Flat<F>> = message.iter().map(|&m| F::promote_flat(m)).collect();
        let wide = encode(&lifted, log_len);

        for (i, (&s, &w)) in narrow.iter().zip(&wide).enumerate() {
            assert_eq!(
                symbol::<F>(field, &s.into_raw().to_bytes()),
                w,
                "{field:?} symbol {i}"
            );
        }
    }

    #[test]
    fn coset_fold_equals_encode_of_folded_message() {
        let (vars, log_inv_rate) = (6, 1);
        let k = kernel(vars + log_inv_rate);

        for a in 1..=MAX_FOLD_ARITY_LOG {
            let seed = 0x94_0400 + 16 * a as u128;
            let message = values(seed, 1 << vars);
            let challenges = values(seed + 1, a);

            let mut folded = message.clone();
            for &r in &challenges {
                folded = (0..folded.len() / 2)
                    .map(|u| folded[2 * u] + r * (folded[2 * u] + folded[2 * u + 1]))
                    .collect();
            }

            let want = encode(&folded, vars - a + log_inv_rate);
            let code = encode(&message, vars + log_inv_rate);

            for (q, &want) in want.iter().enumerate() {
                assert_eq!(
                    k.fold_coset(&code[q << a..(q + 1) << a], q << a, &challenges)
                        .unwrap(),
                    want,
                    "arity 2^{a}, coset {q}"
                );
            }
        }
    }

    #[test]
    fn final_encode_matches_full_transform() {
        let (stop, log_inv_rate) = (5, 2);
        let k = kernel(stop + log_inv_rate + 3);

        let layer = values(0x94_0431, 1 << stop);
        let full = encode(&layer, stop + log_inv_rate);

        let all: Vec<usize> = (0..full.len()).collect();
        let sparse: Vec<usize> = all.iter().copied().step_by(7).collect();
        let edges = vec![0, full.len() - 1];

        for positions in [all, sparse, edges] {
            let want: Vec<Flat<F>> = positions.iter().map(|&i| full[i]).collect();

            assert_eq!(k.encode_final(&layer, &positions).unwrap(), want);
        }
    }

    #[test]
    fn narrow_code_symbols_lift_into_the_fold_field() {
        narrow_code_matches_wide::<Block32>(CodeField::B32, 0x94_0471);
        narrow_code_matches_wide::<Block64>(CodeField::B64, 0x94_0472);
        narrow_code_matches_wide::<Block128>(CodeField::B128, 0x94_0473);
    }
}
