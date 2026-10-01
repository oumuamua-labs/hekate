// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate::core::config::Config;
use hekate::core::trace::{ColumnTrace, ColumnType};
use hekate::crypto::DefaultHasher;
use hekate::crypto::transcript::Transcript;
use hekate::math::{Block128, TowerField};
use hekate_core::errors;
use hekate_core::trace::TraceBuilder;
use hekate_gadgets::{CpuFetchColumns, Instruction, RomChiplet, generate_rom_trace};
use hekate_math::{Bit, Block32};
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::Circuit;
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::digest::program_id;
use hekate_program::permutation::PermutationCheckSpec;
use hekate_program::{Air, FixedColumn, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_verifier::HekateVerifier;

type F = Block128;
type H = DefaultHasher;

const TEST_NUM_VARS: usize = 6;

fn test_config() -> Config {
    Config {
        num_queries: 4,
        min_security_bits: 0,
        zero_knowledge: true,
        ldt_support_size: 4,
        ..Config::default()
    }
}

fn test_instructions(count: usize) -> Vec<Instruction> {
    (0..count)
        .map(|i| Instruction::new(i as u32, 1, [0, 0, 0]))
        .collect()
}

/// Minimal main trace:
/// single Bit column, all zeros.
fn dummy_main_trace(num_vars: usize) -> ColumnTrace {
    let layout = [ColumnType::Bit];
    let tb = TraceBuilder::new(&layout, num_vars).unwrap();

    tb.build()
}

fn prove_and_verify<P: Program<F>>(
    air: &P,
    instance: &ProgramInstance<F>,
    witness: &ProgramWitness<F, ColumnTrace>,
    config: &Config,
) -> bool {
    let seed = [0xCCu8; 32];
    let proof =
        prove(b"ChipletBuses", air, instance, witness, config, seed, None).expect("proving failed");

    let mut vt = Transcript::<H>::new(b"ChipletBuses");
    let pinned_id = program_id(air).unwrap();

    HekateVerifier::<F, H>::verify(&pinned_id, air, instance, &proof, &mut vt, config)
        .unwrap_or(false)
}

fn try_prove_and_verify<P: Program<F>>(
    air: &P,
    instance: &ProgramInstance<F>,
    witness: &ProgramWitness<F, ColumnTrace>,
    config: &Config,
) -> Result<bool, errors::Error> {
    let seed = [0xCCu8; 32];
    let proof =
        prove(b"ChipletBuses", air, instance, witness, config, seed, None).map_err(|_| {
            errors::Error::Protocol {
                protocol: "ffi",
                message: "prove failed",
            }
        })?;

    let mut vt = Transcript::<H>::new(b"ChipletBuses");
    let pinned_id = program_id(air).unwrap();

    HekateVerifier::<F, H>::verify(&pinned_id, air, instance, &proof, &mut vt, config)
}

// ==========================================================
// AIR:
// no main-trace permutation checks,
// N chiplets via chiplet_defs().
// ==========================================================

#[derive(Clone)]
struct BareMainAir {
    defs: Vec<ChipletDef<F>>,
}

impl Air<F> for BareMainAir {
    fn num_columns(&self) -> usize {
        1
    }

    fn column_layout(&self) -> &[ColumnType] {
        &[ColumnType::Bit]
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(0));

        cs.build()
    }
}

impl Program<F> for BareMainAir {
    fn chiplet_defs(&self) -> errors::Result<Vec<ChipletDef<F>>> {
        Ok(self.defs.clone())
    }
}

// ==========================================================
// Verifier bus matching
// ==========================================================

/// Two chiplets with matching bus_ids
/// and matching products, no main-trace
/// counterpart. Verify passes.
#[test]
fn chiplet_to_chiplet_bus_valid() {
    let num_rows = 1 << TEST_NUM_VARS;
    let instructions = test_instructions(num_rows);

    let rom1 = RomChiplet::new(num_rows, num_rows);
    let rom2 = RomChiplet::new(num_rows, num_rows);

    let air = BareMainAir {
        defs: vec![
            ChipletDef::from_air(&rom1).unwrap(),
            ChipletDef::from_air(&rom2).unwrap(),
        ],
    };

    let trace1 = generate_rom_trace(&instructions, num_rows).unwrap();
    let trace2 = generate_rom_trace(&instructions, num_rows).unwrap();

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness =
        ProgramWitness::new(dummy_main_trace(TEST_NUM_VARS)).with_chiplets(vec![trace1, trace2]);

    assert!(
        prove_and_verify(&air, &instance, &witness, &test_config()),
        "chiplet<>chiplet bus with matching products must verify"
    );
}

/// Two chiplets with same bus_id but
/// different products (different instructions).
/// Verify rejects.
#[test]
fn chiplet_to_chiplet_bus_product_mismatch() {
    let num_rows = 1 << TEST_NUM_VARS;

    let instructions_a = test_instructions(num_rows);
    let instructions_b: Vec<Instruction> = (0..num_rows)
        .map(|i| Instruction::new((i * 4 + 999) as u32, 2, [1, 2, 3]))
        .collect();

    let rom1 = RomChiplet::new(num_rows, num_rows);
    let rom2 = RomChiplet::new(num_rows, num_rows);

    let air = BareMainAir {
        defs: vec![
            ChipletDef::from_air(&rom1).unwrap(),
            ChipletDef::from_air(&rom2).unwrap(),
        ],
    };

    let trace1 = generate_rom_trace(&instructions_a, num_rows).unwrap();
    let trace2 = generate_rom_trace(&instructions_b, num_rows).unwrap();

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness =
        ProgramWitness::new(dummy_main_trace(TEST_NUM_VARS)).with_chiplets(vec![trace1, trace2]);

    let result = try_prove_and_verify(&air, &instance, &witness, &test_config());
    assert!(
        result.is_err() || result == Ok(false),
        "mismatched chiplet bus products must fail verification"
    );
}

/// Chiplet declares bus_id with no
/// counterpart anywhere. Verify
/// rejects with "dangling endpoint".
#[test]
fn dangling_chiplet_bus() {
    let num_rows = 1 << TEST_NUM_VARS;
    let instructions = test_instructions(num_rows);

    let rom = RomChiplet::new(num_rows, num_rows);

    // Single chiplet, bus
    // "rom_link" has no counterpart.
    let air = BareMainAir {
        defs: vec![ChipletDef::from_air(&rom).unwrap()],
    };

    let trace = generate_rom_trace(&instructions, num_rows).unwrap();

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(dummy_main_trace(TEST_NUM_VARS)).with_chiplets(vec![trace]);

    let result = try_prove_and_verify(&air, &instance, &witness, &test_config());
    assert!(
        result.is_err() || result == Ok(false),
        "dangling chiplet bus must fail verification"
    );
}

/// Three specs share a bus_id. Third
/// triggers "more than 2 endpoints" error.
#[test]
fn three_endpoint_bus_rejected() {
    let num_rows = 1 << TEST_NUM_VARS;
    let instructions = test_instructions(num_rows);

    let rom1 = RomChiplet::new(num_rows, num_rows);
    let rom2 = RomChiplet::new(num_rows, num_rows);
    let rom3 = RomChiplet::new(num_rows, num_rows);

    let air = BareMainAir {
        defs: vec![
            ChipletDef::from_air(&rom1).unwrap(),
            ChipletDef::from_air(&rom2).unwrap(),
            ChipletDef::from_air(&rom3).unwrap(),
        ],
    };

    let t1 = generate_rom_trace(&instructions, num_rows).unwrap();
    let t2 = generate_rom_trace(&instructions, num_rows).unwrap();
    let t3 = generate_rom_trace(&instructions, num_rows).unwrap();

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness =
        ProgramWitness::new(dummy_main_trace(TEST_NUM_VARS)).with_chiplets(vec![t1, t2, t3]);

    let result = try_prove_and_verify(&air, &instance, &witness, &test_config());
    assert!(
        result.is_err() || result == Ok(false),
        "3-endpoint bus must fail verification"
    );
}

/// Existing main<>chiplet pattern
/// still works unchanged.
#[test]
fn main_chiplet_bus_backward_compat() {
    let num_rows = 1 << TEST_NUM_VARS;
    let instructions = test_instructions(num_rows);

    #[derive(Clone)]
    struct ClassicAir {
        num_rows: usize,
    }

    impl Air<F> for ClassicAir {
        fn num_columns(&self) -> usize {
            CpuFetchColumns::NUM_COLUMNS
        }

        fn column_layout(&self) -> &[ColumnType] {
            static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
            LAYOUT.get_or_init(CpuFetchColumns::build_layout)
        }

        fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
            vec![(RomChiplet::BUS_ID.into(), RomChiplet::cpu_linking_spec())]
        }

        fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
            vec![FixedColumn::prefix(
                CpuFetchColumns::SELECTOR,
                self.num_rows,
            )]
        }

        fn constraint_ast(&self) -> ConstraintAst<F> {
            ConstraintSystem::<F>::new().build()
        }
    }

    impl Program<F> for ClassicAir {
        fn chiplet_defs(&self) -> errors::Result<Vec<ChipletDef<F>>> {
            Ok(vec![ChipletDef::from_air(&RomChiplet::new(
                self.num_rows,
                self.num_rows,
            ))?])
        }
    }

    // Build CPU trace via TraceBuilder
    let layout = CpuFetchColumns::build_layout();

    let mut tb = TraceBuilder::new(&layout, TEST_NUM_VARS).unwrap();

    for (i, instr) in instructions.iter().enumerate() {
        let bytes = instr.pc_bytes();
        tb.set_b32(CpuFetchColumns::PC_B0, i, Block32::from(bytes[0] as u32))
            .unwrap();
        tb.set_b32(CpuFetchColumns::PC_B1, i, Block32::from(bytes[1] as u32))
            .unwrap();
        tb.set_b32(CpuFetchColumns::PC_B2, i, Block32::from(bytes[2] as u32))
            .unwrap();
        tb.set_b32(CpuFetchColumns::PC_B3, i, Block32::from(bytes[3] as u32))
            .unwrap();
        tb.set_b32(
            CpuFetchColumns::OPCODE,
            i,
            Block32::from(instr.opcode as u32),
        )
        .unwrap();

        let args = instr.args();
        tb.set_b32(CpuFetchColumns::ARG0, i, Block32::from(args[0] as u32))
            .unwrap();
        tb.set_b32(CpuFetchColumns::ARG1, i, Block32::from(args[1] as u32))
            .unwrap();
        tb.set_b32(CpuFetchColumns::ARG2, i, Block32::from(args[2] as u32))
            .unwrap();
        tb.set_bit(CpuFetchColumns::SELECTOR, i, Bit::ONE).unwrap();
    }

    let cpu_trace = tb.build();
    let rom_trace = generate_rom_trace(&instructions, num_rows).unwrap();

    let air = ClassicAir { num_rows };
    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![rom_trace]);

    assert!(
        prove_and_verify(&air, &instance, &witness, &test_config()),
        "classic main<>chiplet pattern must still work"
    );
}

/// Full E2E prove/verify of two RomChiplets
/// attached under one namespace, with identical
/// data on their internal chiplet<>chiplet bus.
#[test]
fn namespaced_end_to_end_prove_verify() {
    let num_rows = 1 << TEST_NUM_VARS;
    let instructions = test_instructions(num_rows);

    let mut cx = Circuit::<F>::new("DualRom", num_rows).unwrap();

    cx.column(ColumnType::Bit);
    cx.attach_namespaced(
        "dual_rom",
        vec![
            ChipletDef::from_air(&RomChiplet::new(num_rows, num_rows)).unwrap(),
            ChipletDef::from_air(&RomChiplet::new(num_rows, num_rows)).unwrap(),
        ],
        &[],
    )
    .unwrap();

    let air = cx.compile().unwrap();

    let trace1 = generate_rom_trace(&instructions, num_rows).unwrap();
    let trace2 = generate_rom_trace(&instructions, num_rows).unwrap();

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness =
        ProgramWitness::new(dummy_main_trace(TEST_NUM_VARS)).with_chiplets(vec![trace1, trace2]);

    assert!(prove_and_verify(&air, &instance, &witness, &test_config()));
}
