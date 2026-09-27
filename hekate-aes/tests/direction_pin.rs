// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! A host with witness selectors is refused at verify
//! entry; a pinned host reading ciphertexts it never
//! requested fails the fixed-column compare.

mod common;

use common::{
    CPU_ROWS, F, FIPS128_CIPHER, H, assert_air_violated, build_cpu_trace_128, fips_call_128,
    make_program_128, prove_and_verify, whitened_128,
};
use hekate_aes::{Aes128Chiplet, AesRound128Air, CpuAes128Columns};
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate_crypto::transcript::Transcript;
use hekate_math::{Bit, Block8, TowerField};
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::digest::program_id;
use hekate_program::permutation::PermutationCheckSpec;
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;

const LABEL: &[u8] = b"AES_Direction_Pin";
const AES_ROWS: usize = 32;

/// The pre-cadence discipline roots, with
/// witness selectors and no schedule pins.
#[derive(Clone)]
struct UnpinnedHost {
    aes: Aes128Chiplet<F>,
}

impl UnpinnedHost {
    fn link_spec() -> PermutationCheckSpec {
        let values: Vec<usize> = (0..16).map(|j| CpuAes128Columns::DATA + j).collect();

        AesRound128Air::link_service()
            .request(&values, CpuAes128Columns::SELECTOR)
            .unwrap()
    }

    fn key_spec() -> PermutationCheckSpec {
        let values: Vec<usize> = (0..16).map(|j| CpuAes128Columns::KEY + j).collect();

        AesRound128Air::key_service()
            .request(&values, CpuAes128Columns::KEY_SELECTOR)
            .unwrap()
    }
}

impl Air<F> for UnpinnedHost {
    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(CpuAes128Columns::build_layout)
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![
            (AesRound128Air::LINK_BUS_ID.into(), Self::link_spec()),
            (AesRound128Air::KEY_BUS_ID.into(), Self::key_spec()),
        ]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();

        let sel = cs.col(CpuAes128Columns::SELECTOR);
        let dir = cs.col(CpuAes128Columns::KEY_SELECTOR);
        let next_sel = cs.next(CpuAes128Columns::SELECTOR);
        let next_dir = cs.next(CpuAes128Columns::KEY_SELECTOR);
        let one = cs.one();

        cs.assert_boolean(sel);
        cs.assert_boolean(dir);
        cs.constrain(dir * (one + sel));
        cs.constrain(sel * (dir + next_dir + next_sel));

        cs.build()
    }
}

impl Program<F> for UnpinnedHost {
    fn num_public_inputs(&self) -> usize {
        0
    }

    fn chiplet_defs(&self) -> errors::Result<Vec<hekate_program::chiplet::ChipletDef<F>>> {
        self.aes.composite().flatten_defs()
    }
}

/// Rows 1 and 3 read the ciphertext.
/// No row carries a plaintext or a key.
fn response_only_cpu_trace() -> ColumnTrace {
    let num_vars = CPU_ROWS.trailing_zeros() as usize;
    let mut tb = TraceBuilder::new(&CpuAes128Columns::build_layout(), num_vars).unwrap();

    for row in [1usize, 3] {
        for (j, &byte) in FIPS128_CIPHER.iter().enumerate() {
            tb.set_b8(CpuAes128Columns::DATA + j, row, Block8(byte))
                .unwrap();
        }

        tb.set_bit(CpuAes128Columns::SELECTOR, row, Bit::ONE)
            .unwrap();
    }

    tb.build()
}

fn unbound_ciphertext_witness(aes: &Aes128Chiplet<F>) -> (ColumnTrace, Vec<ColumnTrace>) {
    let call = fips_call_128();
    let traces = aes.generate_traces(&[call.clone(), call]).unwrap();

    (response_only_cpu_trace(), traces)
}

/// The pinned host proves two honest calls;
/// its unpinned twin is refused at verify entry.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn unpinned_host_is_rejected_at_verify() {
    let pinned = make_program_128(AES_ROWS, 2);
    let unpinned = UnpinnedHost {
        aes: pinned.aes.clone(),
    };

    let call = fips_call_128();
    let whitened = whitened_128();

    let instance = ProgramInstance::new(CPU_ROWS, vec![]);
    let witness = ProgramWitness::new(build_cpu_trace_128(&[
        (whitened, FIPS128_CIPHER),
        (whitened, FIPS128_CIPHER),
    ]))
    .with_chiplets(pinned.aes.generate_traces(&[call.clone(), call]).unwrap());

    let config = Config {
        zero_knowledge: true,
        ..Config::dev()
    };

    let proof = prove(
        LABEL,
        &pinned.program,
        &instance,
        &witness,
        &config,
        [0x5Au8; 32],
        None,
    )
    .expect("the pinned host proves honest calls");

    let mut vt = Transcript::<H>::new(LABEL);
    let verdict = HekateVerifier::<F, H>::verify(
        &program_id(&unpinned).unwrap(),
        &unpinned,
        &instance,
        &proof,
        &mut vt,
        &config,
    );

    assert!(
        matches!(
            &verdict,
            Err(errors::Error::Protocol { message, .. })
                if message.starts_with("bus selector or clock phase reads a witness column")
        ),
        "{verdict:?}"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn cadence_pins_reject_unbound_ciphertext() {
    let air = make_program_128(AES_ROWS, 2);
    let (cpu_trace, traces) = unbound_ciphertext_witness(&air.aes);

    assert_air_violated(&air.program, &cpu_trace, &traces);

    match prove_and_verify(&air.program, cpu_trace, traces) {
        Ok(false) | Err(_) => {}
        Ok(true) => panic!("accepted a ciphertext bound to no CPU request row"),
    }
}
