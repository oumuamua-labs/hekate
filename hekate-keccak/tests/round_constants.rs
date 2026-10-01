// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::config::Config;
use hekate_core::trace::{ColumnTrace, ColumnType, Trace, TraceBuilder};
use hekate_crypto::DefaultHasher;
use hekate_crypto::transcript::Transcript;
use hekate_keccak::{
    CpuKeccakColumns, KeccakChiplet, KeccakWitness, PhysKeccakColumns, generate_keccak_trace,
};
use hekate_math::{Bit, Block32, Block64, Block128, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram, Col};
use hekate_program::digest::program_id;
use hekate_program::{Air, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;

type F = Block128;
type H = DefaultHasher;

const ROUNDS: usize = 24;

const ROWS_PER_CALL: usize = ROUNDS + 1;
const ROWS: usize = 32;

/// Which row sends the state and which receives
/// it is the statement, not a witness value.
fn build_program() -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("KeccakRoundConstants", ROWS).unwrap();
    let cpu = cx.schema(&CpuKeccakColumns::build_layout());

    let selector = cpu.at(CpuKeccakColumns::SELECTOR);

    let call_values: Vec<Col> = (0..25)
        .map(|lane| cpu.at(CpuKeccakColumns::LANES + lane))
        .collect();

    cx.call(&KeccakChiplet::service(), &call_values, selector)
        .unwrap();

    cx.fix(
        selector,
        KeccakChiplet::host_selector_shape(KeccakChiplet::BLOCK_ROWS, 1),
    );

    cx.attach(ChipletDef::from_air(&KeccakChiplet::new(ROWS, 1)).unwrap());

    cx.compile().unwrap()
}

fn test_input() -> [u64; 25] {
    core::array::from_fn(|i| (i as u64).wrapping_mul(0x9E37_79B9) | 1)
}

fn constant_schedule(constants: &[u64; ROUNDS]) -> Vec<(u32, u64)> {
    constants
        .iter()
        .enumerate()
        .map(|(round, &rc)| (1u32 << round, rc))
        .collect()
}

fn subset_schedule(offsets: &[usize], steps: usize) -> Vec<(u32, u64)> {
    (0..steps)
        .map(|step| {
            offsets.iter().fold((0u32, 0u64), |(word, rc), &offset| {
                (
                    word | (1u32 << (offset + step)),
                    rc ^ KeccakChiplet::ROUND_CONSTANTS[offset + step],
                )
            })
        })
        .collect()
}

fn cpu_trace(input: [u64; 25], output: [u64; 25]) -> ColumnTrace {
    let num_vars = ROWS.trailing_zeros() as usize;
    let mut tb = TraceBuilder::new(&CpuKeccakColumns::build_layout(), num_vars).unwrap();

    for i in 0..25 {
        tb.set_b64(CpuKeccakColumns::LANES + i, 0, Block64(input[i]))
            .unwrap();
        tb.set_b64(CpuKeccakColumns::LANES + i, ROUNDS, Block64(output[i]))
            .unwrap();
    }

    tb.set_bit(CpuKeccakColumns::SELECTOR, 0, Bit::ONE).unwrap();
    tb.set_bit(CpuKeccakColumns::SELECTOR, ROUNDS, Bit::ONE)
        .unwrap();

    tb.build()
}

/// One block run under `schedule`, emitting
/// on its first row and on its last.
fn chiplet_trace(input: [u64; 25], schedule: &[(u32, u64)]) -> (ColumnTrace, [u64; 25]) {
    let layout = Air::<F>::column_layout(&KeccakChiplet::new(ROWS, 1)).to_vec();
    let num_vars = ROWS.trailing_zeros() as usize;

    let mut tb = TraceBuilder::new(&layout, num_vars).unwrap();
    let mut state = input;

    for (row, &(word, rc)) in schedule.iter().enumerate() {
        for (lane, &value) in state.iter().enumerate() {
            tb.set_b64(lane, row, Block64(value)).unwrap();
        }

        tb.set_b32(PhysKeccakColumns::P_ROUND, row, Block32::from(word))
            .unwrap();
        tb.set_bit(PhysKeccakColumns::P_S_ROUND, row, Bit::ONE)
            .unwrap();

        if row == 0 {
            tb.set_bit(PhysKeccakColumns::P_S_IN_OUT, row, Bit::ONE)
                .unwrap();
        }

        state = KeccakWitness::keccak_f_round(state, rc);
    }

    let out_row = schedule.len();

    for (lane, &value) in state.iter().enumerate() {
        tb.set_b64(lane, out_row, Block64(value)).unwrap();
    }

    tb.set_bit(PhysKeccakColumns::P_S_IN_OUT, out_row, Bit::ONE)
        .unwrap();

    (tb.build(), state)
}

fn proves_and_verifies(schedule: &[(u32, u64)]) -> bool {
    let input = test_input();
    let (chiplet, output) = chiplet_trace(input, schedule);

    run(input, output, chiplet)
}

fn run(cpu_input: [u64; 25], cpu_output: [u64; 25], chiplet: ColumnTrace) -> bool {
    let air = build_program();
    let instance = ProgramInstance::new(ROWS, vec![]);
    let witness =
        ProgramWitness::new(cpu_trace(cpu_input, cpu_output)).with_chiplets(vec![chiplet]);

    let config = Config {
        zero_knowledge: true,
        ..Config::dev()
    };

    let proof = match prove(
        b"Keccak_RC",
        &air,
        &instance,
        &witness,
        &config,
        [0xA5u8; 32],
        None,
    ) {
        Ok(proof) => proof,
        Err(e) => {
            println!("prover refused: {e:?}");
            return false;
        }
    };

    let mut vt = Transcript::<H>::new(b"Keccak_RC");
    let pinned_id = program_id(&air).unwrap();

    HekateVerifier::<F, H>::verify(&pinned_id, &air, &instance, &proof, &mut vt, &config)
        .unwrap_or(false)
}

#[test]
fn layout_matches_crate() {
    let chiplet = KeccakChiplet::new(ROWS, 1);
    let layout = Air::<F>::column_layout(&chiplet);

    assert_eq!(layout.len(), PhysKeccakColumns::P_S_IN_OUT + 1);
    assert!(
        layout[..PhysKeccakColumns::P_ROUND]
            .iter()
            .all(|c| *c == ColumnType::B64)
    );
    assert_eq!(layout[PhysKeccakColumns::P_ROUND], ColumnType::B32);
    assert_eq!(layout[PhysKeccakColumns::P_S_ROUND], ColumnType::Bit);
    assert_eq!(layout[PhysKeccakColumns::P_S_IN_OUT], ColumnType::Bit);
}

#[test]
fn harness_matches_generate_keccak_trace() {
    let input = test_input();
    let block: [Block64; 25] = core::array::from_fn(|i| Block64(input[i]));

    let reference = generate_keccak_trace(&[block], ROWS).unwrap();
    let (mine, _) = chiplet_trace(input, &constant_schedule(&KeccakChiplet::ROUND_CONSTANTS));

    for col in 0..=PhysKeccakColumns::P_S_IN_OUT {
        for row in 0..ROWS_PER_CALL {
            assert_eq!(
                reference.get_element::<F>(col, row).unwrap(),
                mine.get_element::<F>(col, row).unwrap(),
                "column {col} row {row} diverges from generate_keccak_trace"
            );
        }
    }
}

#[test]
fn canonical_schedule_verifies() {
    assert!(
        proves_and_verifies(&constant_schedule(&KeccakChiplet::ROUND_CONSTANTS)),
        "the harness itself is broken, every rejection below is unattributable"
    );
}

/// The state chain is honest under a zero schedule,
/// the trace is consistent while the permutation
/// is not Keccak-f[1600].
#[test]
fn zero_schedule_rejected() {
    assert!(!proves_and_verifies(&constant_schedule(&[0u64; ROUNDS])));
}

#[test]
fn flipped_constant_rejected() {
    let mut constants = KeccakChiplet::ROUND_CONSTANTS;
    constants[7] ^= 1;

    assert!(!proves_and_verifies(&constant_schedule(&constants)));
}

/// The activity constraint is a char-2 parity:
/// an odd-weight set of round bits satisfies it,
/// shifts in lockstep and feeds iota the XOR of its
/// constants. Held by the input-row exactness pin.
#[test]
fn odd_weight_schedule_rejected() {
    assert!(!proves_and_verifies(&subset_schedule(&[0, 5, 9], 15)));
}

/// Nothing forces the chain to reach round 23 unless the
/// terminal constraint pins `next_s_round` on every round row.
#[test]
fn truncated_schedule_rejected() {
    assert!(!proves_and_verifies(&subset_schedule(&[0], 5)));
}

/// The host reads the block backwards, `f(x)` in
/// and `x` out; rank 0 holds `x` and both emits miss.
#[test]
fn backwards_block_rejected() {
    let x = test_input();
    let (chiplet, y) = chiplet_trace(x, &constant_schedule(&KeccakChiplet::ROUND_CONSTANTS));

    assert!(!run(y, x, chiplet));
}
