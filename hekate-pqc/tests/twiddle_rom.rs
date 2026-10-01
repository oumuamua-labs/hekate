// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::config::Config;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder, TraceColumn};
use hekate_crypto::DefaultHasher;
use hekate_crypto::transcript::Transcript;
use hekate_math::{Bit, Block32, Block128, Flat, TowerField};
use hekate_pqc::ntt::{NttRun, NttSchedule};
use hekate_pqc::twiddle_rom::{
    TWIDDLE_W_BINDING_BUS_ID, TwiddleEntry, TwiddleRomChiplet, TwiddleRomColumns,
    generate_twiddle_rom_trace,
};
use hekate_program::chiplet::ChipletDef;
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::digest::program_id;
use hekate_program::permutation::PermutationCheckSpec;
use hekate_program::{Air, FixedColumn, FixedShape, Program, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_scribble::{MutationKind, ScribbleConfig, Target, assert_all_caught};
use hekate_verifier::HekateVerifier;
use rand::{TryRngCore, rngs::OsRng};
use std::sync::OnceLock;

type F = Block128;
type H = DefaultHasher;

const Q: u32 = 3329;

const CPU_BFLY: usize = 0;
const CPU_W: usize = 1;
const CPU_SEL: usize = 2;
const CPU_NUM_COLS: usize = 3;

fn cpu_layout() -> Vec<ColumnType> {
    vec![ColumnType::B32, ColumnType::B32, ColumnType::Bit]
}

fn cpu_w_binding_spec() -> PermutationCheckSpec {
    TwiddleRomChiplet::w_binding_service()
        .request(&[CPU_BFLY, CPU_W], CPU_SEL)
        .expect("w_binding_service slots match the requester columns")
}

fn forward_schedule() -> NttSchedule {
    NttSchedule::new(7, vec![NttRun::Forward { instances: 1 }])
}

fn honest_entries() -> Vec<TwiddleEntry> {
    let mut entries = Vec::with_capacity(1792);
    for k in 0..896usize {
        entries.push(TwiddleEntry {
            layer: (k / 128) as u32,
            butterfly_idx: (k % 128) as u32,
            w: (k as u32) % Q,
            is_mulonly: false,
            active: true,
        });
        entries.push(TwiddleEntry {
            layer: 0,
            butterfly_idx: 0,
            w: 0,
            is_mulonly: false,
            active: false,
        });
    }

    entries
}

fn set_b32(trace: &mut ColumnTrace, col: usize, row: usize, val: u32) {
    match &mut trace.columns[col] {
        TraceColumn::B32(data) => {
            data[row] = Flat::from_raw(Block32::from(val));
        }
        _ => panic!("expected B32 column at {col}"),
    }
}

fn set_bit(trace: &mut ColumnTrace, col: usize, row: usize, val: Bit) {
    match &mut trace.columns[col] {
        TraceColumn::Bit(data) => data[row] = val,
        _ => panic!("expected Bit column at {col}"),
    }
}

#[derive(Clone)]
struct ShadowExploitProgram {
    twiddle_rows: usize,
}

impl Air<F> for ShadowExploitProgram {
    fn num_columns(&self) -> usize {
        CPU_NUM_COLS
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: OnceLock<Vec<ColumnType>> = OnceLock::new();
        LAYOUT.get_or_init(cpu_layout).as_slice()
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![(TWIDDLE_W_BINDING_BUS_ID.into(), cpu_w_binding_spec())]
    }

    fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
        vec![FixedColumn {
            col_idx: CPU_SEL,
            shape: FixedShape::Sparse(Vec::new()),
        }]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
    }
}

impl Program<F> for ShadowExploitProgram {
    fn chiplet_defs(&self) -> hekate_core::errors::Result<Vec<ChipletDef<F>>> {
        let twiddle = TwiddleRomChiplet::new(Q, self.twiddle_rows, forward_schedule());
        let mut def = ChipletDef::from_air(&twiddle)?;

        // No NTT chiplet here, so the
        // layer-lookup bus has no partner.
        def.permutation_checks
            .retain(|(id, _)| id == TWIDDLE_W_BINDING_BUS_ID);

        Ok(vec![def])
    }
}

#[derive(Clone)]
struct TwiddleRomIsolatedProgram {
    twiddle_rows: usize,
}

impl Air<F> for TwiddleRomIsolatedProgram {
    fn num_columns(&self) -> usize {
        1
    }

    fn column_layout(&self) -> &[ColumnType] {
        &[ColumnType::Bit]
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        Vec::new()
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
    }
}

impl Program<F> for TwiddleRomIsolatedProgram {
    fn chiplet_defs(&self) -> hekate_core::errors::Result<Vec<ChipletDef<F>>> {
        let twiddle = TwiddleRomChiplet::new(Q, self.twiddle_rows, forward_schedule());

        let mut def = ChipletDef::from_air(&twiddle)?;
        def.permutation_checks.clear();

        Ok(vec![def])
    }
}

fn run_e2e<T>(tamper: T) -> bool
where
    T: FnOnce(&mut ColumnTrace, &mut ColumnTrace),
{
    let entries = honest_entries();
    let twiddle_rows: usize = 2048;
    let cpu_rows: usize = 4;

    let mut twiddle_trace = generate_twiddle_rom_trace(&entries, twiddle_rows)
        .expect("honest twiddle trace generation");

    let cpu_tb = TraceBuilder::new(&cpu_layout(), cpu_rows.trailing_zeros() as usize)
        .expect("cpu trace builder");

    let mut cpu_trace = cpu_tb.build();

    tamper(&mut twiddle_trace, &mut cpu_trace);

    let air = ShadowExploitProgram { twiddle_rows };
    let instance = ProgramInstance::new(cpu_rows, Vec::new());
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![twiddle_trace]);

    let config = Config {
        zero_knowledge: true,
        ..Config::dev()
    };

    let mut seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut seed).unwrap();

    let proof = match prove(
        b"TwiddleRom_Shadow",
        &air,
        &instance,
        &witness,
        &config,
        seed,
        None,
    ) {
        Ok(p) => p,
        Err(_) => return false,
    };

    let mut vt = Transcript::<H>::new(b"TwiddleRom_Shadow");
    let pinned_id = program_id(&air).unwrap();

    HekateVerifier::<F, H>::verify(&pinned_id, &air, &instance, &proof, &mut vt, &config)
        .unwrap_or(false)
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn shadow_honest_baseline_verifies() {
    assert!(
        run_e2e(|_t, _c| {}),
        "honest twiddle ROM + empty CPU side must verify"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn shadow_exploit_must_be_rejected() {
    // Forge MULONLY=1 on a non-basemul
    // row + matching CPU partner.
    let accepted = run_e2e(|twiddle, cpu| {
        set_bit(twiddle, TwiddleRomColumns::MULONLY_SELECTOR, 0, Bit::ONE);

        set_b32(cpu, CPU_BFLY, 0, 0);
        set_b32(cpu, CPU_W, 0, 1);
        set_bit(cpu, CPU_SEL, 0, Bit::ONE);
    });

    assert!(
        !accepted,
        "forged twiddle_w_binding emission must be rejected"
    );
}

#[test]
fn scribble_twiddle_rom_mulonly_selector_focused() {
    let entries = honest_entries();
    let twiddle_rows: usize = 2048;
    let cpu_rows: usize = 4;

    let twiddle_trace = generate_twiddle_rom_trace(&entries, twiddle_rows)
        .expect("honest twiddle trace generation");

    let cpu_tb = TraceBuilder::new(&cpu_layout(), cpu_rows.trailing_zeros() as usize)
        .expect("cpu trace builder");
    let cpu_trace = cpu_tb.build();

    let air = TwiddleRomIsolatedProgram { twiddle_rows };
    let instance = ProgramInstance::new(cpu_rows, Vec::new());
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![twiddle_trace]);

    assert_all_caught(
        &air,
        &instance,
        &witness,
        ScribbleConfig::default()
            .target(Target::Chiplet(0))
            .mutations([MutationKind::FlipSelector])
            .include_cols([TwiddleRomColumns::MULONLY_SELECTOR])
            .cases(32),
    );
}
