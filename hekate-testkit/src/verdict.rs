// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::proofs::InnerProof;
use hekate_crypto::DefaultHasher;
use hekate_crypto::transcript::Transcript;
use hekate_math::Block256;
use hekate_program::digest::program_id;
use hekate_program::{Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_sdk::preflight;
use hekate_verifier::HekateVerifier;

use crate::fault::Fault;
use crate::log::logged;

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
        let (outcome, log) = verdict(air, statement, &with_mode(base, zero_knowledge));
        assert_eq!(outcome, Ok(true), "zk {zero_knowledge}\n{log}");
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

    fault.assert_sole(&preflight(air, instance, witness).unwrap());

    for zero_knowledge in [false, true] {
        let config = with_mode(base, zero_knowledge);

        let (outcome, log) = verdict(air, honest, &config);
        assert_eq!(outcome, Ok(true), "zk {zero_knowledge}\n{log}");

        let (outcome, log) = verdict(air, forged, &config);
        assert!(
            fault.rejects(&outcome, &log, zero_knowledge),
            "zk {zero_knowledge}: {outcome:?}\n{log}"
        );
    }
}

pub fn verdict(
    air: &(impl Program<Block256> + Sync),
    (instance, witness): Statement<'_>,
    config: &Config,
) -> (errors::Result<bool>, String) {
    let proof = prove(LABEL, air, instance, witness, config, SEED, None)
        .expect("the prover does not refuse");

    verify_logged(air, instance, config, &proof, LABEL)
}

pub fn verify_logged(
    air: &(impl Program<Block256> + Sync),
    instance: &ProgramInstance<Block256>,
    config: &Config,
    proof: &InnerProof<Block256>,
    label: &'static [u8],
) -> (errors::Result<bool>, String) {
    let mut transcript = Transcript::<DefaultHasher>::new(label);
    let pinned_id = program_id(air).unwrap();

    logged(|| {
        HekateVerifier::<Block256, DefaultHasher>::verify(
            &pinned_id,
            air,
            instance,
            proof,
            &mut transcript,
            config,
        )
    })
}

pub fn rejected_at(outcome: &errors::Result<bool>, log: &str, site: &str) -> bool {
    matches!(outcome, Ok(false)) && log.contains(site)
}

pub fn rejected_with(outcome: &errors::Result<bool>, expected: &str) -> bool {
    matches!(outcome, Err(errors::Error::Protocol { message, .. }) if *message == expected)
}

fn with_mode(base: &Config, zero_knowledge: bool) -> Config {
    Config {
        zero_knowledge,
        ..base.clone()
    }
}
