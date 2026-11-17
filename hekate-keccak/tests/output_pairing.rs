// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

mod common;

use common::keccak_f;
use hekate_core::config::Config;
use hekate_core::trace::{ColumnTrace, TraceBuilder};
use hekate_keccak::{CpuKeccakColumns, KeccakChiplet, generate_keccak_trace};
use hekate_math::{Bit, Block64, Block256, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram, Col};
use hekate_program::{ProgramInstance, ProgramWitness};
use hekate_testkit::fault::Fault;
use hekate_testkit::verdict::{assert_rejected, assert_verifies};

type F = Block256;
type Request = ([u64; 25], [u64; 25]);

const CALLS: usize = 2;
const ROWS: usize = 64;
const OUTPUT_OFFSET: usize = KeccakChiplet::BLOCK_ROWS - 1;
const BLOCK_ORDERS: [[usize; CALLS]; 2] = [[0, 1], [1, 0]];

fn build_program() -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("KeccakOutputPairing", ROWS).unwrap();

    let cpu = cx.schema(&CpuKeccakColumns::build_layout());

    let selector = cpu.at(CpuKeccakColumns::SELECTOR);

    let lane = |i: usize| cpu.at(CpuKeccakColumns::LANES + i);

    let values: Vec<Col> = (0..25).map(lane).collect();

    cx.call(&KeccakChiplet::service(), &values, selector)
        .unwrap();

    cx.fix(
        selector,
        KeccakChiplet::host_selector_shape(KeccakChiplet::BLOCK_ROWS, CALLS),
    );

    for call in 0..CALLS {
        let row = call * KeccakChiplet::BLOCK_ROWS;
        for i in 0..25 {
            cx.publish(lane(i), row);
        }

        cx.publish(lane(0), row + OUTPUT_OFFSET);
    }

    cx.attach(ChipletDef::from_air(&KeccakChiplet::new(ROWS, CALLS)).unwrap());

    cx.compile().unwrap()
}

fn inputs() -> [[u64; 25]; CALLS] {
    core::array::from_fn(|call| {
        core::array::from_fn(|i| ((25 * call + i) as u64).wrapping_mul(0x9E37_79B9) | 1)
    })
}

fn cpu_trace(requests: &[Request]) -> ColumnTrace {
    let num_vars = ROWS.trailing_zeros() as usize;
    let mut tb = TraceBuilder::new(&CpuKeccakColumns::build_layout(), num_vars).unwrap();

    for (call, (input, output)) in requests.iter().enumerate() {
        let row = call * KeccakChiplet::BLOCK_ROWS;
        for i in 0..25 {
            tb.set_b64(CpuKeccakColumns::LANES + i, row, Block64(input[i]))
                .unwrap();
            tb.set_b64(
                CpuKeccakColumns::LANES + i,
                row + OUTPUT_OFFSET,
                Block64(output[i]),
            )
            .unwrap();
        }

        tb.set_bit(CpuKeccakColumns::SELECTOR, row, Bit::ONE)
            .unwrap();
        tb.set_bit(CpuKeccakColumns::SELECTOR, row + OUTPUT_OFFSET, Bit::ONE)
            .unwrap();
    }

    tb.build()
}

fn public_inputs(requests: &[Request]) -> Vec<F> {
    requests
        .iter()
        .flat_map(|(input, output)| input.iter().chain([&output[0]]))
        .map(|&v| F::from(Block64(v)))
        .collect()
}

fn statement(
    digest_of: [usize; CALLS],
    block_order: [usize; CALLS],
) -> (ProgramInstance<F>, ProgramWitness<F>) {
    let inputs = inputs();
    let outputs = inputs.map(keccak_f);

    let requests: Vec<Request> = (0..CALLS)
        .map(|k| (inputs[k], outputs[digest_of[k]]))
        .collect();

    let blocks: Vec<[Block64; 25]> = block_order
        .iter()
        .map(|&k| inputs[k].map(Block64))
        .collect();

    (
        ProgramInstance::new(ROWS, public_inputs(&requests)),
        ProgramWitness::new(cpu_trace(&requests))
            .with_chiplets(vec![generate_keccak_trace(&blocks, ROWS).unwrap()]),
    )
}

fn config() -> Config {
    Config {
        num_queries: 2,
        fold_arity_log: 1,
        early_stop_vars: 1,
        ..Config::dev()
    }
}

#[test]
fn honest_pairing_verifies() {
    let (instance, witness) = statement([0, 1], [0, 1]);

    assert_verifies(&build_program(), (&instance, &witness), &config());
}

#[test]
fn traded_outputs_rejected() {
    let outputs = inputs().map(keccak_f);

    assert_ne!(outputs[0][0], outputs[1][0]);

    let (honest_instance, honest) = statement([0, 1], [0, 1]);
    let fault = Fault {
        buses: &[KeccakChiplet::BUS_ID],
        ..Fault::NONE
    };

    for block_order in BLOCK_ORDERS {
        let (instance, witness) = statement([1, 0], block_order);

        assert_rejected(
            &build_program(),
            (&honest_instance, &honest),
            (&instance, &witness),
            fault,
            &config(),
        );
    }
}
