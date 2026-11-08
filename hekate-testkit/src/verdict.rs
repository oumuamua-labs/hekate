// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::config::Config;
use hekate_core::proofs::InnerProof;
use hekate_crypto::DefaultHasher;
use hekate_crypto::transcript::Transcript;
use hekate_math::Block256;
use hekate_program::digest::program_id;
use hekate_program::{Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_sdk::preflight;
use hekate_verifier::HekateVerifier;
use hekate_verifier::rejection::Rejection;

use crate::fault::Fault;

const LABEL: &[u8] = b"hekate-testkit";
const SEED: [u8; 32] = [0x5A; 32];

pub type Statement<'a> = (&'a ProgramInstance<Block256>, &'a ProgramWitness<Block256>);

pub fn assert_verifies(
    air: &(impl Program<Block256> + Sync),
    statement: Statement<'_>,
    base: &Config,
) {
    let (instance, witness) = statement;
    let report = preflight(air, instance, witness).unwrap();

    assert!(report.is_clean(), "{report}");

    for zero_knowledge in [false, true] {
        let outcome = verdict(air, statement, &with_mode(base, zero_knowledge));
        assert_eq!(outcome, Ok(()), "zk {zero_knowledge}");
    }
}

pub fn assert_rejected(
    air: &(impl Program<Block256> + Sync),
    honest: Statement<'_>,
    forged: Statement<'_>,
    fault: Fault<'_>,
    base: &Config,
) {
    let (instance, witness) = forged;
    let report = preflight(air, instance, witness).unwrap();

    fault.assert_sole(&report);

    for zero_knowledge in [false, true] {
        let config = with_mode(base, zero_knowledge);

        let outcome = verdict(air, honest, &config);
        assert_eq!(outcome, Ok(()), "zk {zero_knowledge}");

        let outcome = verdict(air, forged, &config);
        let expected = fault.rejection(air, &report, zero_knowledge);

        assert_eq!(outcome, Err(expected), "zk {zero_knowledge}");
    }
}

pub fn verdict(
    air: &(impl Program<Block256> + Sync),
    (instance, witness): Statement<'_>,
    config: &Config,
) -> Result<(), Rejection> {
    let proof = prove(LABEL, air, instance, witness, config, SEED, None)
        .expect("the prover does not refuse");

    verify(air, instance, config, &proof, LABEL)
}

pub fn verify(
    air: &(impl Program<Block256> + Sync),
    instance: &ProgramInstance<Block256>,
    config: &Config,
    proof: &InnerProof<Block256>,
    label: &'static [u8],
) -> Result<(), Rejection> {
    let mut transcript = Transcript::<DefaultHasher>::new(label);
    let pinned_id = program_id(air).unwrap();

    HekateVerifier::<Block256, DefaultHasher>::verify(
        &pinned_id,
        air,
        instance,
        proof,
        &mut transcript,
        config,
    )
}

fn with_mode(base: &Config, zero_knowledge: bool) -> Config {
    Config {
        zero_knowledge,
        ..base.clone()
    }
}
