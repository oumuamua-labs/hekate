// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::config::Config;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate_crypto::DefaultHasher;
use hekate_crypto::transcript::Transcript;
use hekate_keccak::{CpuKeccakColumns, KeccakChiplet, KeccakWitness, generate_keccak_trace};
use hekate_math::{Bit, Block64, Block128, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram, Col};
use hekate_program::digest::program_id;
use hekate_program::{FixedShape, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;

type F = Block128;
type H = DefaultHasher;

const CALLS: usize = 3;
const ROWS: usize = 128;
const OUTPUT_OFFSET: usize = KeccakChiplet::BLOCK_ROWS - 1;
const BLOCK_ORDERS: [[usize; CALLS]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

fn in_row(call: usize) -> usize {
    call * KeccakChiplet::BLOCK_ROWS
}

fn out_row(call: usize) -> usize {
    in_row(call) + OUTPUT_OFFSET
}

fn build_program() -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("KeccakSqueezeChain", ROWS).unwrap();

    let cpu = cx.schema(&CpuKeccakColumns::build_layout());
    let chain = cx.column(ColumnType::Bit);

    let selector = cpu.at(CpuKeccakColumns::SELECTOR);

    let lane = |i: usize| cpu.at(CpuKeccakColumns::LANES + i);

    let values: Vec<Col> = (0..25).map(lane).collect();

    cx.call(&KeccakChiplet::service(), &values, selector)
        .unwrap();

    cx.fix(
        selector,
        KeccakChiplet::host_selector_shape(KeccakChiplet::BLOCK_ROWS, CALLS),
    );
    cx.fix(
        chain,
        FixedShape::Sparse((0..CALLS - 1).map(|k| (out_row(k), F::ONE)).collect()),
    );

    {
        let cs = cx.cs();

        for i in 0..25 {
            let next_input = cs.next(lane(i).index());
            let output = cs.col(lane(i).index());

            cs.assert_zero_when(cs.col(chain.index()), next_input + output);
        }
    }

    for i in 0..25 {
        cx.publish(lane(i), in_row(0));
    }

    for call in 0..CALLS {
        cx.publish(lane(0), out_row(call));
    }

    cx.attach(ChipletDef::from_air(&KeccakChiplet::new(ROWS, CALLS)).unwrap());

    cx.compile().unwrap()
}

fn permute(mut state: [u64; 25]) -> [u64; 25] {
    for rc in KeccakChiplet::ROUND_CONSTANTS {
        state = KeccakWitness::keccak_f_round(state, rc);
    }

    state
}

fn seed() -> [u64; 25] {
    core::array::from_fn(|i| (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
}

fn iterates() -> [[u64; 25]; CALLS + 1] {
    let mut out = [seed(); CALLS + 1];

    for j in 1..=CALLS {
        out[j] = permute(out[j - 1]);
    }

    out
}

fn cpu_trace(output_iterate: [usize; CALLS]) -> ColumnTrace {
    let states = iterates();
    let num_vars = ROWS.trailing_zeros() as usize;

    let mut layout = CpuKeccakColumns::build_layout();
    layout.push(ColumnType::Bit);

    let chain_col = CpuKeccakColumns::NUM_COLUMNS;

    let mut tb = TraceBuilder::new(&layout, num_vars).unwrap();

    for call in 0..CALLS {
        let input = match call {
            0 => states[0],
            _ => states[output_iterate[call - 1]],
        };
        let output = states[output_iterate[call]];

        for i in 0..25 {
            tb.set_b64(CpuKeccakColumns::LANES + i, in_row(call), Block64(input[i]))
                .unwrap();
            tb.set_b64(
                CpuKeccakColumns::LANES + i,
                out_row(call),
                Block64(output[i]),
            )
            .unwrap();
        }

        tb.set_bit(CpuKeccakColumns::SELECTOR, in_row(call), Bit::ONE)
            .unwrap();
        tb.set_bit(CpuKeccakColumns::SELECTOR, out_row(call), Bit::ONE)
            .unwrap();

        if call + 1 < CALLS {
            tb.set_bit(chain_col, out_row(call), Bit::ONE).unwrap();
        }
    }

    tb.build()
}

fn public_inputs(output_iterate: [usize; CALLS]) -> Vec<F> {
    let states = iterates();

    states[0]
        .iter()
        .chain(output_iterate.iter().map(|&p| &states[p][0]))
        .map(|&v| F::from(Block64(v)))
        .collect()
}

fn verdicts(output_iterate: [usize; CALLS], block_order: [usize; CALLS]) -> [bool; 2] {
    let states = iterates();

    let blocks: Vec<[Block64; 25]> = block_order
        .iter()
        .map(|&k| states[k].map(Block64))
        .collect();

    let air = build_program();

    let instance = ProgramInstance::new(ROWS, public_inputs(output_iterate));
    let witness = ProgramWitness::new(cpu_trace(output_iterate))
        .with_chiplets(vec![generate_keccak_trace(&blocks, ROWS).unwrap()]);

    [false, true].map(|zero_knowledge| {
        let config = Config {
            zero_knowledge,
            ..Config::dev()
        };

        let proof = prove(
            b"Keccak_Chain_Rotation",
            &air,
            &instance,
            &witness,
            &config,
            [0x5Au8; 32],
            None,
        )
        .expect("the prover proves the witness it is handed");

        let mut vt = Transcript::<H>::new(b"Keccak_Chain_Rotation");
        let pinned_id = program_id(&air).unwrap();

        HekateVerifier::<F, H>::verify(&pinned_id, &air, &instance, &proof, &mut vt, &config)
            .unwrap_or(false)
    })
}

#[test]
fn honest_squeeze_verifies() {
    assert_eq!(
        verdicts([1, 2, 3], [0, 1, 2]),
        [true, true],
        "the harness itself is broken, the rejections below are unattributable"
    );
}

#[test]
fn rotated_squeeze_rejected() {
    for block_order in BLOCK_ORDERS {
        assert_eq!(
            verdicts([2, 1, 3], block_order),
            [false, false],
            "block order {block_order:?}"
        );
    }
}
