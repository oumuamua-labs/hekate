// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use std::time::Instant;

use hekate_core::config::Config;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder, TraceColumn};
use hekate_crypto::DefaultHasher;
use hekate_crypto::transcript::Transcript;
use hekate_math::{Bit, Block32, Block128, HardwareField, TowerField};
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::digest::program_id;
use hekate_program::{Air, FixedShape, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_scribble::{MutationKind, ScribbleConfig, assert_all_caught_all_targets};
use hekate_sdk::preflight;
use hekate_sha2::sha256::feed_forward_carries;
use hekate_sha2::{
    BLOCK_WORDS, CpuSha256Block, IV, ROUNDS, STATE_WORDS, Sha256Call, Sha256Chiplet, digest_bytes,
    pad_message, sha256_words,
};
use hekate_verifier::HekateVerifier;
use rand::{Rng, SeedableRng, rngs::StdRng};
use sha2::{Digest, Sha256};

type F = Block128;
type H = DefaultHasher;

const CPU_ACTIVE: usize = CpuSha256Block::COLUMNS;
const CPU_CHAIN: usize = CPU_ACTIVE + 1;

const TWO_BLOCK_MSG: &[u8] = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";

struct Statement {
    program: CircuitProgram<F>,
    sha: Sha256Chiplet<F>,
    block: CpuSha256Block,
    cpu_rows: usize,
    blocks: Vec<[u32; BLOCK_WORDS]>,
    digest: [u32; STATE_WORDS],
}

fn cpu_layout() -> Vec<ColumnType> {
    let mut layout = CpuSha256Block::layout().to_vec();
    layout.push(ColumnType::Bit);
    layout.push(ColumnType::Bit);

    layout
}

fn build(msg: &[u8], rounds_per_row: usize) -> Statement {
    let blocks = pad_message(msg);
    let num_blocks = blocks.len();
    let cpu_rows = num_blocks.next_power_of_two().max(2);
    let chiplet_rows = (num_blocks * (ROUNDS / rounds_per_row)).next_power_of_two();

    let sha = Sha256Chiplet::<F>::new(chiplet_rows, num_blocks, rounds_per_row).unwrap();

    let mut cx = Circuit::<F>::new("Sha256Cpu", cpu_rows).unwrap();

    let block = CpuSha256Block::declare(&mut cx, 0);
    let active = cx.column(ColumnType::Bit);
    let chain = cx.column(ColumnType::Bit);

    block.connect(&mut cx, active).unwrap();

    let prefix = |count: usize| FixedShape::Cadence {
        stride: 1,
        count,
        origin: 0,
        values: vec![F::ONE],
    };

    cx.fix(active, prefix(num_blocks));
    cx.fix(chain, prefix(num_blocks - 1));

    for (i, &iv) in IV.iter().enumerate() {
        cx.boundary(block.h_in_words.at(i), 0, F::from(u128::from(iv)));
    }

    {
        let cs = cx.cs();
        let chain = cs.col(chain.index());

        for i in 0..STATE_WORDS {
            let next_h_in = cs.next(block.h_in_words.at(i).index());
            let h_out = cs.col(block.h_out_words.at(i).index());

            cs.assert_zero_when(chain, next_h_in + h_out);
        }
    }

    for i in 0..STATE_WORDS {
        cx.publish(block.h_out_words.at(i), num_blocks - 1);
    }

    cx.attach(sha.def().unwrap());

    let program = cx.compile().unwrap();

    assert_eq!(program.column_layout(), cpu_layout().as_slice());

    Statement {
        program,
        sha,
        block,
        cpu_rows,
        blocks,
        digest: sha256_words(msg),
    }
}

fn cpu_calls(st: &Statement) -> Vec<Sha256Call> {
    let mut calls = Vec::with_capacity(st.blocks.len());
    let mut h = IV;

    for (b, block) in st.blocks.iter().enumerate() {
        let call = Sha256Call {
            h_in: h,
            block: *block,
            request_idx: b as u32,
        };

        h = call.h_out();

        calls.push(call);
    }

    calls
}

fn cpu_trace(st: &Statement) -> ColumnTrace {
    let num_vars = st.cpu_rows.trailing_zeros() as usize;
    let mut tb = TraceBuilder::new(&cpu_layout(), num_vars).unwrap();

    for (b, call) in cpu_calls(st).iter().enumerate() {
        st.block.write(&mut tb, b, call).unwrap();

        tb.set_bit(CPU_ACTIVE, b, Bit::ONE).unwrap();

        if b + 1 < st.blocks.len() {
            tb.set_bit(CPU_CHAIN, b, Bit::ONE).unwrap();
        }
    }

    tb.build()
}

fn chiplet_trace(st: &Statement) -> ColumnTrace {
    st.sha.trace(&cpu_calls(st)).unwrap()
}

fn instance(st: &Statement) -> ProgramInstance<F> {
    let public: Vec<F> = st
        .digest
        .iter()
        .map(|&word| F::from(u128::from(word)))
        .collect();

    ProgramInstance::new(st.cpu_rows, public)
}

fn prove_and_verify(
    st: &Statement,
    cpu: ColumnTrace,
    chiplet: ColumnTrace,
) -> Result<(bool, usize, f64), String> {
    let instance = instance(st);
    let witness = ProgramWitness::new(cpu).with_chiplets(vec![chiplet]);

    let report =
        preflight(&st.program, &instance, &witness).map_err(|e| format!("preflight: {e:?}"))?;

    if !report.is_clean() {
        for v in &report.constraint_violations {
            eprintln!(
                "constraint={} label={:?} row={}",
                v.constraint_idx, v.label, v.row_idx
            );
        }

        for d in &report.bus_diagnostics {
            for ep in &d.endpoints {
                eprintln!("bus \"{}\": active={}", d.bus_id, ep.active_rows);
            }
        }

        return Err("preflight violations".into());
    }

    verdict_without_preflight(st, witness)
}

fn verdict_without_preflight(
    st: &Statement,
    witness: ProgramWitness<F, ColumnTrace>,
) -> Result<(bool, usize, f64), String> {
    let instance = instance(st);

    let config = Config {
        zero_knowledge: true,
        ..Config::dev()
    };

    let mut blinding_seed = [0u8; 32];
    rand::rng().fill_bytes(&mut blinding_seed);

    let started = Instant::now();
    let proof = prove(
        b"SHA256_E2E",
        &st.program,
        &instance,
        &witness,
        &config,
        blinding_seed,
        None,
    )
    .map_err(|e| format!("prover: {e:?}"))?;
    let prove_secs = started.elapsed().as_secs_f64();

    let proof_bytes = hekate_sdk::serialize_proof_bytes(&proof).len();

    let mut vt = Transcript::<H>::new(b"SHA256_E2E");
    let pinned_id = program_id(&st.program).unwrap();

    let ok = HekateVerifier::<F, H>::verify(
        &pinned_id,
        &st.program,
        &instance,
        &proof,
        &mut vt,
        &config,
    )
    .map_err(|e| format!("verifier: {e:?}"))?;

    Ok((ok, proof_bytes, prove_secs))
}

fn assert_accepted(st: &Statement) {
    match prove_and_verify(st, cpu_trace(st), chiplet_trace(st)) {
        Ok((true, _, _)) => {}
        Ok((false, _, _)) => panic!("verifier rejected an honest proof"),
        Err(e) => panic!("{e}"),
    }
}

fn assert_rejected(st: &Statement, cpu: ColumnTrace, chiplet: ColumnTrace) {
    let witness = ProgramWitness::new(cpu).with_chiplets(vec![chiplet]);

    if let Ok((true, _, _)) = verdict_without_preflight(st, witness) {
        panic!("accepted");
    }
}

fn set_b32(trace: &mut ColumnTrace, col: usize, row: usize, value: u32) {
    let TraceColumn::B32(data) = &mut trace.columns[col] else {
        panic!("expected B32 column at {col}");
    };

    data[row] = Block32(value).to_hardware();
}

fn flip_b32(trace: &mut ColumnTrace, col: usize, row: usize, mask: u32) {
    let TraceColumn::B32(data) = &mut trace.columns[col] else {
        panic!("expected B32 column at {col}");
    };

    let original = data[row].to_tower().0;
    data[row] = Block32(original ^ mask).to_hardware();
}

fn random_message(seed: u64, len: usize) -> Vec<u8> {
    let mut msg = vec![0u8; len];
    StdRng::seed_from_u64(seed).fill_bytes(&mut msg);

    msg
}

fn value_pin_program(rows: usize) -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("ValuePin", rows).unwrap();
    let word = cx.expand_bits(1, ColumnType::B32);
    let packed = cx.reuse_pass_through(&word);

    cx.fix(
        packed.at(0),
        FixedShape::Cadence {
            stride: 4,
            count: rows / 4,
            origin: 0,
            values: hekate_sha2::K[..4]
                .iter()
                .map(|&k| F::from(u128::from(k)))
                .collect(),
        },
    );

    let cs = cx.cs();
    let bit = cs.col(word.bits(0).at(0).index());

    cs.constrain(bit * (bit + cs.one()));

    cx.compile().unwrap()
}

fn value_pin_trace(rows: usize, substitute: Option<(usize, u32)>) -> ColumnTrace {
    let mut tb = TraceBuilder::new(&[ColumnType::B32], rows.trailing_zeros() as usize).unwrap();

    for row in 0..rows {
        let mut value = hekate_sha2::K[row % 4];

        if let Some((at, forged)) = substitute
            && at == row
        {
            value = forged;
        }

        tb.set_b32(0, row, Block32(value)).unwrap();
    }

    tb.build()
}

fn value_pin_verdict(program: &CircuitProgram<F>, trace: ColumnTrace) -> Result<bool, String> {
    let instance = ProgramInstance::new(8, vec![]);
    let witness = ProgramWitness::new(trace);

    let config = Config {
        zero_knowledge: true,
        ..Config::dev()
    };

    let mut seed = [0u8; 32];
    rand::rng().fill_bytes(&mut seed);

    let proof = prove(
        b"VALUE_PIN",
        program,
        &instance,
        &witness,
        &config,
        seed,
        None,
    )
    .map_err(|e| format!("prover: {e:?}"))?;

    let mut vt = Transcript::<H>::new(b"VALUE_PIN");
    let pinned_id = program_id(program).unwrap();

    HekateVerifier::<F, H>::verify(&pinned_id, program, &instance, &proof, &mut vt, &config)
        .map_err(|e| format!("verifier: {e:?}"))
}

#[test]
fn reference_matches_sha2_crate() {
    for (seed, len) in [
        (1, 0),
        (2, 3),
        (3, 55),
        (4, 56),
        (5, 64),
        (6, 200),
        (7, 1_000),
    ] {
        let msg = random_message(seed, len);
        let expected: [u8; 32] = Sha256::digest(&msg).into();

        assert_eq!(digest_bytes(&sha256_words(&msg)), expected, "len {len}");
    }
}

#[test]
fn honest_traces_pass_preflight_at_every_row_shape() {
    let msg = random_message(11, 200);

    for rounds_per_row in [1, 2, 4, 8, 16] {
        let st = build(&msg, rounds_per_row);
        let witness = ProgramWitness::new(cpu_trace(&st)).with_chiplets(vec![chiplet_trace(&st)]);
        let report = preflight(&st.program, &instance(&st), &witness).unwrap();

        assert!(
            report.is_clean(),
            "rounds_per_row = {rounds_per_row}: {} constraint, {} boundary violations",
            report.constraint_violations.len(),
            report.boundary_violations.len()
        );
    }
}

#[test]
fn flipped_carry_fails_preflight() {
    let st = build(b"abc", 8);
    let layout = st.sha.layout();
    let mut chiplet = chiplet_trace(&st);

    flip_b32(&mut chiplet, layout.carry + 3, 2, 1 << 17);

    let witness = ProgramWitness::new(cpu_trace(&st)).with_chiplets(vec![chiplet]);
    let report = preflight(&st.program, &instance(&st), &witness).unwrap();

    assert!(!report.is_clean());
}

#[test]
fn flipped_chiplet_message_word_fails_preflight() {
    let st = build(b"abc", 8);
    let layout = st.sha.layout();
    let mut chiplet = chiplet_trace(&st);

    flip_b32(&mut chiplet, layout.window + 5, 0, 1);

    let witness = ProgramWitness::new(cpu_trace(&st)).with_chiplets(vec![chiplet]);
    let report = preflight(&st.program, &instance(&st), &witness).unwrap();

    assert!(!report.is_clean());
}

#[test]
fn forged_state_out_fails_only_at_chiplet_pin() {
    let mut st = build(TWO_BLOCK_MSG, 8);

    let layout = st.sha.layout();
    let rows_per_block = layout.rows_per_block();
    let calls = cpu_calls(&st);
    let last = calls.len() - 1;
    let call = calls[last];

    let mut forged = call.state_out();
    forged[3] ^= 1 << 5;

    let h_out = hekate_sha2::feed_forward(&call.h_in, &forged);
    let carries = feed_forward_carries(&call.h_in, &forged);

    let mut cpu = cpu_trace(&st);
    let mut chiplet = chiplet_trace(&st);

    for i in 0..STATE_WORDS {
        set_b32(&mut cpu, CpuSha256Block::STATE_OUT + i, last, forged[i]);
        set_b32(&mut cpu, CpuSha256Block::H_OUT + i, last, h_out[i]);
        set_b32(&mut cpu, CpuSha256Block::CARRY + i, last, carries[i]);

        for row in 0..rows_per_block {
            set_b32(
                &mut chiplet,
                layout.state_out + i,
                last * rows_per_block + row,
                forged[i],
            );
        }
    }

    st.digest = h_out;

    let witness = ProgramWitness::new(cpu).with_chiplets(vec![chiplet]);
    let report = preflight(&st.program, &instance(&st), &witness).unwrap();

    assert!(report.boundary_violations.is_empty());
    assert!(report.bus_diagnostics.iter().all(|d| !d.has_failures()));
    assert_eq!(report.constraint_violations.len(), 1);
}

#[test]
fn scribble_flip_selector_caught() {
    let st = build(TWO_BLOCK_MSG, 8);
    let witness = ProgramWitness::new(cpu_trace(&st)).with_chiplets(vec![chiplet_trace(&st)]);

    assert_all_caught_all_targets(
        &st.program,
        &instance(&st),
        &witness,
        ScribbleConfig::default()
            .mutations([MutationKind::FlipSelector])
            .cases(64),
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn abc_e2e() {
    assert_accepted(&build(b"abc", 1));
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn two_block_e2e() {
    assert_accepted(&build(TWO_BLOCK_MSG, 4));
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn random_four_blocks_e2e() {
    let msg = random_message(42, 200);
    let st = build(&msg, 1);

    assert_eq!(digest_bytes(&st.digest), Sha256::digest(&msg).as_slice());
    assert_accepted(&st);
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn wrong_digest_rejected() {
    let st = build(b"abc", 1);
    let mut cpu = cpu_trace(&st);

    flip_b32(&mut cpu, CpuSha256Block::H_OUT + 2, 0, 1 << 9);

    assert_rejected(&st, cpu, chiplet_trace(&st));
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn rounds_per_row_sweep() {
    let msg = random_message(42, 200);

    for rounds_per_row in [1, 2, 4, 8, 16] {
        let st = build(&msg, rounds_per_row);
        let layout = st.sha.layout();

        let (ok, proof_bytes, prove_secs) =
            prove_and_verify(&st, cpu_trace(&st), chiplet_trace(&st)).unwrap();

        assert!(ok, "rounds_per_row = {rounds_per_row}");

        println!(
            "rounds_per_row={rounds_per_row:>2} chiplet_rows={:>4} row_bytes={:>5} \
             roots={:>6} prove={:>6.1} ms proof={:>5} KiB",
            st.sha.num_rows(),
            layout.row_bytes(),
            st.sha.program().constraint_ast().roots.len(),
            prove_secs * 1e3,
            proof_bytes / 1024
        );
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn value_pin_accepts_the_honest_column() {
    let program = value_pin_program(8);

    assert_eq!(
        value_pin_verdict(&program, value_pin_trace(8, None)),
        Ok(true)
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn value_pin_rejects_substituted_constant() {
    let program = value_pin_program(8);
    let forged = value_pin_trace(8, Some((5, hekate_sha2::K[1] ^ (1 << 7))));

    let report = preflight(
        &program,
        &ProgramInstance::new(8, vec![]),
        &ProgramWitness::new(forged.clone()),
    )
    .unwrap();

    assert!(report.constraint_violations.is_empty());
    assert_eq!(report.fixed_column_violations.len(), 1);

    if let Ok(true) = value_pin_verdict(&program, forged) {
        panic!("accepted");
    }
}
