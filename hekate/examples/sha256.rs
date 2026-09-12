// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

#[path = "common/mod.rs"]
mod common;

use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::{Bit, Block128, TowerField};
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::{FixedShape, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_sha2::{
    BLOCK_WORDS, CpuSha256Block, IV, ROUNDS, STATE_WORDS, Sha256Call, Sha256Chiplet, digest_bytes,
    pad_message,
};
use hekate_verifier::HekateVerifier;
use rand::{TryRngCore, rngs::OsRng};

type F = Block128;
type H = DefaultHasher;

const CPU_ACTIVE: usize = CpuSha256Block::COLUMNS;
const CPU_CARRY: usize = CPU_ACTIVE + 1;
const CPU_CHAIN: usize = CPU_CARRY + 1;

struct Statement {
    program: CircuitProgram<F>,
    block: CpuSha256Block,
    calls: Vec<Sha256Call>,
    digest: [u32; STATE_WORDS],
}

fn rounds_per_row() -> usize {
    std::env::var("HEKATE_ROUNDS_PER_ROW")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4)
}

fn cpu_layout() -> Vec<ColumnType> {
    let mut layout = CpuSha256Block::layout().to_vec();
    layout.extend([ColumnType::Bit; 3]);

    layout
}

fn build(message: &[u8], chiplet: &Sha256Chiplet<F>) -> errors::Result<Statement> {
    let blocks = pad_message(message);
    let num_blocks = blocks.len();
    let num_rows = chiplet.num_rows();
    let rows_per_block = chiplet.layout().rows_per_block();

    let mut cx = Circuit::<F>::new("Sha256Inline", num_rows)?;

    let block = CpuSha256Block::declare(&mut cx, 0);
    let active = cx.column(ColumnType::Bit);
    let carry = cx.column(ColumnType::Bit);
    let chain = cx.column(ColumnType::Bit);

    block.connect(&mut cx, active)?;

    let cadence = |count: usize, pred: &dyn Fn(usize) -> bool| FixedShape::Cadence {
        stride: rows_per_block,
        count,
        origin: 0,
        values: (0..rows_per_block)
            .map(|off| if pred(off) { F::ONE } else { F::ZERO })
            .collect(),
    };

    cx.fix(active, cadence(num_blocks, &|off| off == 0));
    cx.fix(carry, cadence(num_blocks, &|off| off + 1 < rows_per_block));
    cx.fix(
        chain,
        cadence(num_blocks - 1, &|off| off + 1 == rows_per_block),
    );

    for (i, &iv) in IV.iter().enumerate() {
        cx.boundary(block.h_in_words.at(i), 0, F::from(u128::from(iv)));
    }

    {
        let cs = cx.cs();
        let carry = cs.col(carry.index());
        let chain = cs.col(chain.index());

        for i in 0..STATE_WORDS {
            let h_out = cs.col(block.h_out_words.at(i).index());

            cs.assert_zero_when(carry, cs.next(block.h_out_words.at(i).index()) + h_out);
            cs.assert_zero_when(chain, cs.next(block.h_in_words.at(i).index()) + h_out);
        }
    }

    for i in 0..STATE_WORDS {
        cx.publish(block.h_out_words.at(i), (num_blocks - 1) * rows_per_block);
    }

    cx.mount(chiplet.def()?);

    let mut calls = Vec::with_capacity(num_blocks);
    let mut h = IV;

    for (b, block) in blocks.iter().enumerate() {
        let call = Sha256Call {
            h_in: h,
            block: *block,
            request_idx: (b * rows_per_block) as u32,
        };

        h = call.h_out();

        calls.push(call);
    }

    Ok(Statement {
        program: cx.compile()?,
        block,
        calls,
        digest: h,
    })
}

fn combined_trace(st: &Statement, chiplet: &Sha256Chiplet<F>) -> errors::Result<ColumnTrace> {
    let num_rows = chiplet.num_rows();
    let num_vars = num_rows.trailing_zeros() as usize;
    let rows_per_block = chiplet.layout().rows_per_block();
    let last_block = st.calls.len() - 1;

    let mut tb = TraceBuilder::new(&cpu_layout(), num_vars)?;

    for (b, call) in st.calls.iter().enumerate() {
        let first_row = b * rows_per_block;
        for row in first_row..first_row + rows_per_block {
            st.block.write(&mut tb, row, call)?;
        }

        tb.set_bit(CPU_ACTIVE, first_row, Bit::ONE)?;

        for row in first_row..first_row + rows_per_block - 1 {
            tb.set_bit(CPU_CARRY, row, Bit::ONE)?;
        }

        if b < last_block {
            tb.set_bit(CPU_CHAIN, first_row + rows_per_block - 1, Bit::ONE)?;
        }
    }

    let mut trace = tb.build();

    let sha = chiplet.trace(&st.calls)?;

    for col in sha.into_columns() {
        trace.add_column(col)?;
    }

    Ok(trace)
}

fn main() {
    common::init("SHA-256");

    let num_vars = common::num_vars(15);
    let num_rows = 1 << num_vars;

    let rounds_per_row = rounds_per_row();
    let max_blocks = num_rows * rounds_per_row / ROUNDS;

    assert!(
        max_blocks > 0,
        "2^{num_vars} rows at {rounds_per_row} rounds per row holds no full block"
    );

    let num_blocks = std::env::var("HEKATE_BLOCKS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(max_blocks)
        .clamp(1, max_blocks);

    let message_len = num_blocks * BLOCK_WORDS * 4 - 9;

    let config = Config {
        zero_knowledge: common::zero_knowledge(),
        ..Config::default()
    };

    let mut blinding_seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut blinding_seed).unwrap();

    let chiplet = Sha256Chiplet::<F>::new(num_rows, num_blocks, rounds_per_row).unwrap();

    println!(
        "Rows: 2^{} ({} compressions, {} bytes hashed, {} rounds per row)",
        num_vars, num_blocks, message_len, rounds_per_row
    );
    println!(
        "Physical columns: {} CPU + {} chiplet, mounted in one table",
        cpu_layout().len(),
        chiplet.layout().columns().len()
    );

    let (st, trace) = common::phase("Trace Generation", || {
        let mut message = vec![0u8; message_len];
        OsRng.try_fill_bytes(&mut message).unwrap();

        let st = build(&message, &chiplet).expect("program build");
        let trace = combined_trace(&st, &chiplet).expect("trace");

        (st, trace)
    });

    print!("SHA-256 digest: 0x");
    for byte in digest_bytes(&st.digest) {
        print!("{byte:02x}");
    }
    println!();

    let public_inputs: Vec<F> = st
        .digest
        .iter()
        .map(|&word| F::from(u128::from(word)))
        .collect();

    let instance = ProgramInstance::new(num_rows, public_inputs);
    let witness = ProgramWitness::new(trace);

    let proof = common::phase("Proving", || {
        prove(
            b"Sha256_E2E",
            &st.program,
            &instance,
            &witness,
            &config,
            blinding_seed,
            None,
        )
        .expect("Prover failed")
    });

    common::proof_breakdown(&proof);

    let mut verifier_transcript = Transcript::<H>::new(b"Sha256_E2E");
    let pinned_id = common::audited_id(&st.program);

    let is_valid = common::phase_with_mem("Verifying", || {
        HekateVerifier::<F, H>::verify(
            &pinned_id,
            &st.program,
            &instance,
            &proof,
            &mut verifier_transcript,
            &config,
        )
        .expect("Verifier failed")
    });

    common::result(is_valid);
}
