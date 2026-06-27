//! Layer-1 + Layer-2 scribble coverage for the
//! RAM chiplet from `hekate-gadgets`. Random
//! shallow tampers plus hand-crafted exploits
//! that target the chiplet's load-bearing
//! invariants: address-sorted ordering, value
//! consistency at the same address, padding-row
//! shadow columns, and the committed Q_LAST
//! row-position selector.

use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate_gadgets::{CpuMemColumns, CpuMemoryUnit, MemoryEvent, RamChiplet, generate_ram_trace};
use hekate_math::{Bit, Block32, Block128, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::permutation::PermutationCheckSpec;
use hekate_program::{Air, Program, ProgramInstance, ProgramWitness};
use hekate_scribble::{
    Mutation, MutationKind, ScribbleConfig, Target, assert_all_caught, check_single_mutation,
};

// AUX_INV:
// free witness on same-addr transitions
// (d=0 zeroes the gate).
const PHY_AUX_INV: usize = 15;

type F = Block128;

#[derive(Clone)]
struct RamTestProgram {
    num_rows: usize,
}

impl Air<F> for RamTestProgram {
    fn num_columns(&self) -> usize {
        CpuMemColumns::NUM_COLUMNS
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(CpuMemColumns::build_layout)
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![(RamChiplet::BUS_ID.into(), CpuMemoryUnit::linking_spec())]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();

        let s = cs.col(CpuMemColumns::SELECTOR);
        cs.assert_boolean(s);
        cs.assert_boolean(cs.col(CpuMemColumns::IS_WRITE));

        let not_active = cs.one() - s;
        cs.assert_zero_when(not_active, cs.col(CpuMemColumns::IS_WRITE));

        cs.build()
    }
}

impl Program<F> for RamTestProgram {
    fn chiplet_defs(&self) -> hekate_core::errors::Result<Vec<ChipletDef<F>>> {
        let ram = RamChiplet::new(self.num_rows);
        Ok(vec![ChipletDef::from_air(&ram)?])
    }
}

fn generate_cpu_trace(events: &[MemoryEvent], num_rows: usize) -> ColumnTrace {
    let num_vars = num_rows.trailing_zeros() as usize;
    let mut tb = TraceBuilder::new(&CpuMemColumns::build_layout(), num_vars).unwrap();

    for (i, event) in events.iter().enumerate() {
        let addr_bytes = event.addr_bytes();
        let val_bytes = event.val_bytes();

        for j in 0..4 {
            tb.set_b32(
                CpuMemColumns::ADDR_B0 + j,
                i,
                Block32::from(addr_bytes[j] as u32),
            )
            .unwrap();
            tb.set_b32(
                CpuMemColumns::VAL_B0 + j,
                i,
                Block32::from(val_bytes[j] as u32),
            )
            .unwrap();
        }

        tb.set_bit(
            CpuMemColumns::IS_WRITE,
            i,
            if event.is_write { Bit::ONE } else { Bit::ZERO },
        )
        .unwrap();
        tb.set_bit(CpuMemColumns::SELECTOR, i, Bit::ONE).unwrap();
    }

    tb.build()
}

fn build_fixture(
    events: &[MemoryEvent],
    num_rows: usize,
) -> (
    RamTestProgram,
    ProgramInstance<F>,
    ProgramWitness<F, ColumnTrace>,
) {
    let air = RamTestProgram { num_rows };
    let cpu_trace = generate_cpu_trace(events, num_rows);
    let ram_trace = generate_ram_trace(events, num_rows).expect("ram trace gen");

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![ram_trace]);

    (air, instance, witness)
}

fn setup_dense_fixture() -> (
    RamTestProgram,
    ProgramInstance<F>,
    ProgramWitness<F, ColumnTrace>,
) {
    let events = vec![
        MemoryEvent::write(0x1000, 0, 42),
        MemoryEvent::write(0x2000, 1, 99),
        MemoryEvent::read(0x1000, 2, 42),
        MemoryEvent::read(0x2000, 3, 99),
    ];

    build_fixture(&events, 4)
}

fn setup_consistency_window_fixture() -> (
    RamTestProgram,
    ProgramInstance<F>,
    ProgramWitness<F, ColumnTrace>,
) {
    // write A -> read A -> write A (new) -> read A
    // exercises the same-address value-equality
    // chain that the chiplet must enforce.
    let events = vec![
        MemoryEvent::write(0x1000, 0, 42),
        MemoryEvent::read(0x1000, 1, 42),
        MemoryEvent::write(0x1000, 2, 99),
        MemoryEvent::read(0x1000, 3, 99),
    ];

    build_fixture(&events, 4)
}

fn setup_padding_fixture() -> (
    RamTestProgram,
    ProgramInstance<F>,
    ProgramWitness<F, ColumnTrace>,
) {
    let events = vec![
        MemoryEvent::write(0x1000, 0, 42),
        MemoryEvent::write(0x2000, 1, 99),
        MemoryEvent::read(0x1000, 2, 42),
    ];

    // 5 events would round to 8; force 4 padding
    // rows out of 8 to exercise shadow columns.
    build_fixture(&events, 8)
}

#[test]
fn ram_chiplet_survives_chaos() {
    let (air, instance, witness) = setup_dense_fixture();

    let config = ScribbleConfig::default()
        .target(Target::Chiplet(0))
        .mutations([
            MutationKind::BitFlip,
            MutationKind::OutOfBounds,
            MutationKind::FlipSelector,
            MutationKind::DuplicateRow,
            MutationKind::SwapRows,
        ])
        // PHY_AUX_INV is unconstrained on inactive
        // gates; random mutations there escape.
        .exclude_cols([PHY_AUX_INV])
        .cases(256);

    assert_all_caught(&air, &instance, &witness, config);
}

#[test]
fn ram_padding_fixture_survives_chaos() {
    let (air, instance, witness) = setup_padding_fixture();

    // Rows 3..8 are SELECTOR=0 padding
    // (bus silenced, cells free).
    let config = ScribbleConfig::default()
        .target(Target::Chiplet(0))
        .mutations([
            MutationKind::BitFlip,
            MutationKind::OutOfBounds,
            MutationKind::FlipSelector,
            MutationKind::DuplicateRow,
        ])
        // PHY_AUX_INV is unconstrained on inactive
        // gates; random mutations there escape.
        .exclude_cols([PHY_AUX_INV])
        .include_rows(0..3)
        .cases(512);

    assert_all_caught(&air, &instance, &witness, config);
}

#[test]
fn ram_consistency_window_survives_chaos() {
    let (air, instance, witness) = setup_consistency_window_fixture();

    let config = ScribbleConfig::default()
        .target(Target::Chiplet(0))
        .mutations([
            MutationKind::BitFlip,
            MutationKind::OutOfBounds,
            MutationKind::FlipSelector,
            MutationKind::DuplicateRow,
            MutationKind::SwapRows,
        ])
        // PHY_AUX_INV is unconstrained on inactive
        // gates; random mutations there escape.
        .exclude_cols([PHY_AUX_INV])
        .cases(256);

    assert_all_caught(&air, &instance, &witness, config);
}

#[test]
fn ram_value_consistency_tamper_caught() {
    // Honest fixture has read(0x1000) returning 42.
    // Flip the low byte of VAL_B0 on the chiplet
    // read row to make the read return 43 while
    // the bus still binds it to 42 (the CPU side
    // is unchanged). The chiplet's value-equality
    // chain across same-address rows must catch it.
    let (air, instance, witness) = setup_dense_fixture();

    let mutation = Mutation::BitFlip {
        target: Target::Chiplet(0),
        col: 8,
        row: 2,
        mask: 1,
    };

    let result = check_single_mutation(&air, &instance, &witness, &mutation);
    assert!(result.is_ok(), "value-consistency tamper escaped preflight");
}

/// Documents the Q_LAST sum-to-one gap.
/// Honest: q_last[N-1] = 1, else 0.
/// Adversarial: q_last ≡ 0 everywhere.
#[test]
fn ram_q_last_uniform_zero_caught() {
    let (air, instance, witness) = setup_dense_fixture();

    const PHY_Q_LAST: usize = 20;

    let mutation = Mutation::ColumnUniformWrite {
        target: Target::Chiplet(0),
        col: PHY_Q_LAST,
        value: 0,
    };

    let result = check_single_mutation(&air, &instance, &witness, &mutation);
    assert!(
        result.is_ok(),
        "Q_LAST uniform-zero attack escaped preflight"
    );
}

/// Trace-wide row-position-selector collapse.
/// q_last ≡ 0, q_step ≡ 1, q_first ≡ 0
/// satisfies boolean, complement, wrap, and
/// no-consecutive without anchoring Σ q_last = 1.
#[test]
fn ram_q_chain_uniform_collapse_caught() {
    let (air, instance, witness) = setup_dense_fixture();

    const PHY_Q_STEP: usize = 18;
    const PHY_Q_FIRST: usize = 19;
    const PHY_Q_LAST: usize = 20;

    let mutation = Mutation::Compound(vec![
        Mutation::ColumnUniformWrite {
            target: Target::Chiplet(0),
            col: PHY_Q_LAST,
            value: 0,
        },
        Mutation::ColumnUniformWrite {
            target: Target::Chiplet(0),
            col: PHY_Q_STEP,
            value: 1,
        },
        Mutation::ColumnUniformWrite {
            target: Target::Chiplet(0),
            col: PHY_Q_FIRST,
            value: 0,
        },
    ]);

    let result = check_single_mutation(&air, &instance, &witness, &mutation);
    assert!(
        result.is_ok(),
        "q_last/q_step/q_first uniform-collapse escaped preflight"
    );
}

#[test]
fn ram_address_tamper_on_chiplet_caught() {
    // Flipping ADDR_B0 of a chiplet row reroutes
    // the event to a fictitious address. Bus on
    // the chiplet side now emits a key the CPU
    // partner never requested -> multiset break.
    let (air, instance, witness) = setup_dense_fixture();

    let mutation = Mutation::BitFlip {
        target: Target::Chiplet(0),
        col: 0,
        row: 0,
        mask: 1,
    };

    let result = check_single_mutation(&air, &instance, &witness, &mutation);
    assert!(result.is_ok(), "address tamper escaped preflight");
}
