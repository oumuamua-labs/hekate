// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

mod multibus;

use hekate::core::config::Config;
use hekate::crypto::{DefaultHasher, Digest, Hasher};
use hekate::math::Block256;
use hekate_core::errors::Error;
use hekate_math::TowerField;
use hekate_testkit::verdict::verify;
use hekate_verifier::rejection::{FoldStage, PoolStage, Rejection};
use multibus::{LABEL, cfg, multibus_proof};

type F = Block256;

const NODE_BYTES: usize = <<DefaultHasher as Hasher>::Digest as Digest>::BYTES;

/// Five draws on an eight-node layer
/// leave three or more nodes unreached.
fn layer_gap_cfg() -> Config {
    Config {
        num_queries: 5,
        ..cfg()
    }
}

#[test]
fn trace_opening_sibling_tamper_rejected() {
    let (air, instance, config, mut proof) = multibus_proof(cfg());

    let opening = &mut proof.eval_proof.openings[0];

    assert!(!opening.multiproof.siblings.is_empty());

    opening.multiproof.siblings[0] ^= 1;

    let outcome = verify(&air, &instance, &config, &proof, LABEL);

    assert_eq!(outcome, Err(Rejection::Fold(FoldStage::Multiproof)));
}

#[test]
fn opened_column_off_part_layout_rejected() {
    let (air, instance, config, proof) = multibus_proof(cfg());

    let mut long_h = proof.clone();
    long_h.eval_proof.openings[1].leaves.extend([0u8; 16]);

    let mut short_trace = proof;
    short_trace.eval_proof.openings[0].leaves.pop();

    for forged in [long_h, short_trace] {
        let result = verify(&air, &instance, &config, &forged, LABEL);

        assert_eq!(
            result,
            Err(Rejection::Malformed(Error::Protocol {
                protocol: "fold",
                message: "opened leaves do not match the tree's leaf width",
            }))
        );
    }
}

#[test]
fn master_count_mismatch_rejected() {
    let (air, instance, config, proof) = multibus_proof(cfg());

    let mut short = proof.clone();
    short.eval_proof.masters.pop();

    let mut long = proof.clone();
    long.eval_proof.masters.push(F::ONE);

    let mut short_mask = proof.clone();
    short_mask.eval_proof.mask_masters.pop();

    let mut long_mask = proof;
    long_mask.eval_proof.mask_masters.push(F::ONE);

    for forged in [short, long, short_mask, long_mask] {
        let result = verify(&air, &instance, &config, &forged, LABEL);

        assert_eq!(result, Err(Rejection::PoolEval(PoolStage::Masters)));
    }
}

#[test]
fn master_or_fold_vector_tamper_rejected() {
    let (air, instance, config, proof) = multibus_proof(cfg());

    for k in 0..proof.eval_proof.masters.len() {
        let mut forged = proof.clone();
        forged.eval_proof.masters[k] += F::ONE;

        let outcome = verify(&air, &instance, &config, &forged, LABEL);

        assert_eq!(
            outcome,
            Err(Rejection::Fold(FoldStage::FinalClaim)),
            "master {k}"
        );
    }

    for k in 0..proof.eval_proof.mask_masters.len() {
        let mut forged = proof.clone();
        forged.eval_proof.mask_masters[k] += F::ONE;

        let outcome = verify(&air, &instance, &config, &forged, LABEL);

        assert_eq!(
            outcome,
            Err(Rejection::Fold(FoldStage::FinalClaim)),
            "mask master {k}"
        );
    }

    let mut bumped = proof.clone();
    bumped.eval_proof.fold_rounds[0] += F::ONE;

    let outcome = verify(&air, &instance, &config, &bumped, LABEL);

    assert_eq!(outcome, Err(Rejection::Fold(FoldStage::FinalClaim)));

    let mut truncated = proof;
    truncated.eval_proof.fold_rounds.pop();

    let result = verify(&air, &instance, &config, &truncated, LABEL);

    assert_eq!(result, Err(Rejection::Fold(FoldStage::Layout)));
}

#[test]
fn unreached_layer_node_tamper_rejected() {
    let (air, instance, config, proof) = multibus_proof(layer_gap_cfg());

    let mut trace = proof.clone();
    let unreached = &mut trace.eval_proof.openings[0].multiproof.unreached;

    assert!(!unreached.is_empty());

    unreached[0] ^= 1;

    let mut h = proof;
    let unreached = &mut h.eval_proof.openings[1].multiproof.unreached;

    assert!(!unreached.is_empty());

    unreached[0] ^= 1;

    for forged in [trace, h] {
        let outcome = verify(&air, &instance, &config, &forged, LABEL);

        assert_eq!(outcome, Err(Rejection::Fold(FoldStage::Multiproof)));
    }
}

#[test]
fn unreached_section_off_derived_count_rejected() {
    let (air, instance, config, proof) = multibus_proof(layer_gap_cfg());

    assert!(!proof.eval_proof.openings[0].multiproof.unreached.is_empty());

    let mut extra = proof.clone();
    extra.eval_proof.openings[0]
        .multiproof
        .unreached
        .extend_from_slice(&[0u8; NODE_BYTES]);

    let mut short = proof;
    let unreached = &mut short.eval_proof.openings[0].multiproof.unreached;

    unreached.truncate(unreached.len() - NODE_BYTES);

    for forged in [extra, short] {
        let outcome = verify(&air, &instance, &config, &forged, LABEL);

        assert_eq!(outcome, Err(Rejection::Fold(FoldStage::Multiproof)));
    }
}

#[test]
fn eval_skip_round_rejected() {
    let (air, instance, config, mut proof) = multibus_proof(cfg());

    assert!(proof.eval_proof.sumcheck_proof.skip.is_empty());

    proof.eval_proof.sumcheck_proof.skip.push(F::ONE);

    let outcome = verify(&air, &instance, &config, &proof, LABEL);

    assert_eq!(outcome, Err(Rejection::PoolEval(PoolStage::Degree)));
}
