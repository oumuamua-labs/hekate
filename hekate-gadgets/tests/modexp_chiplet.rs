// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use hekate_core::config::Config;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder, TraceColumn};
use hekate_crypto::DefaultHasher;
use hekate_crypto::transcript::Transcript;
use hekate_gadgets::chiplets::bignum::modexp::{
    self, BUS_ID, LIMBS32, Modexp, ModexpChiplet, NUM_ROWS, RESULT_ROW,
};
use hekate_math::{Bit, Block32, Block128, HardwareField, TowerField};
use hekate_program::circuit::{Circuit, CircuitProgram, Col};
use hekate_program::digest::program_id;
use hekate_program::{Air, FixedShape, ProgramInstance, ProgramWitness};
use hekate_prover_sys::prove;
use hekate_scribble::{MutationKind, ScribbleConfig, assert_all_caught_all_targets};
use hekate_sdk::preflight;
use hekate_verifier::HekateVerifier;

type F = Block128;
type H = DefaultHasher;

const CPU_ROWS: usize = 2;
const CPU_N: usize = 0;
const CPU_S: usize = LIMBS32;
const CPU_R: usize = 2 * LIMBS32;
const CPU_ACTIVE: usize = 3 * LIMBS32;

/// `R = pow(S, 65537, N)` from CPython's `pow`,
/// `random.seed(20260911)`.
#[rustfmt::skip]
const N: [u32; LIMBS32] = [
    0xcf15a761, 0x9a82b719, 0x5cb61f8e, 0x03e92daa,
    0x72d5f70d, 0xe4f190b1, 0x175e869b, 0xa99aba73,
    0xebf57857, 0x2c7e3121, 0x813898f6, 0xd18004ee,
    0x68e2a888, 0x52a46bec, 0x7d2dfaf7, 0x40b801c3,
    0x22cac669, 0x918e832a, 0xc319a980, 0x45044e00,
    0xc0979096, 0x8142e55c, 0x165ffd4f, 0x6ff16c0c,
    0x608c5982, 0x1f154fc1, 0x97d2f980, 0x8905d6f2,
    0xa732080a, 0xcd271ab5, 0xf2ba6ed7, 0x817fbeab,
    0x7c5d45a5, 0xaf691825, 0x9eb8884c, 0x8c834a46,
    0xbb70ef68, 0xb9a3fc36, 0x9b35b98d, 0x5153fd41,
    0x74c9025c, 0x0ec8abe5, 0xf747adc1, 0x6a4c422a,
    0xc0c39a56, 0xa155acde, 0xcda82707, 0xc2b9fd18,
    0x289ef76b, 0x6bacb5b8, 0xed0a08c3, 0xf29100a3,
    0x23677dd1, 0xdbbdfda2, 0x952fdfe8, 0x610904ee,
    0x06607b5e, 0x47a5802b, 0xaa5e5a73, 0xe9df4ead,
    0xe2135545, 0xa5f14445, 0x0e361d03, 0x8fd5da91,
];

#[rustfmt::skip]
const S: [u32; LIMBS32] = [
    0x6d45c8c3, 0x25812b9b, 0x33e57098, 0x29813ced,
    0xbc75fe20, 0x1f46bf7c, 0xad1af473, 0xf3257bad,
    0x6cb538a2, 0xe3df1fda, 0x0d1c0f05, 0x64e412c2,
    0x3f9f681e, 0x954fbd5f, 0xfd836bba, 0xef019957,
    0x6bfe120f, 0x17a3d2cc, 0xc79795c9, 0x642636c2,
    0xef2032b5, 0xf78206cf, 0x927e56f6, 0x3a0f0d1b,
    0xce6e7cd6, 0xb79a7944, 0x2b5dbfe4, 0x92f70f4b,
    0x071f4e74, 0x1cda0d2a, 0x6998aceb, 0x463e29ca,
    0x1602f718, 0x4665336d, 0x12875775, 0xb4e304dc,
    0x7dd1caa6, 0x703f79a3, 0x419b29a7, 0x284e41c9,
    0x5737eff9, 0x77490bb2, 0xba0d832b, 0x3ebc1585,
    0x2775b4f9, 0xf54618ef, 0x868d15f6, 0x99a6e0f7,
    0x16767ed3, 0x49f64bb9, 0x9265838c, 0x60b4fd35,
    0x677919ee, 0x34b43fd1, 0xd4f47315, 0xb596c45e,
    0x4c044463, 0xf78703cf, 0x1a3c6f25, 0x963f9414,
    0x49fab4e3, 0xc28b2a14, 0xf778ad20, 0x2059ef86,
];

#[rustfmt::skip]
const R: [u32; LIMBS32] = [
    0x3b6597f5, 0xc2c81f7b, 0xcbf46aa2, 0xa2462c14,
    0x8023ef5e, 0x36c52b92, 0xd0baedcb, 0xc2dda1d3,
    0x1e83a702, 0x657d267b, 0xc1ba9909, 0x2dff540d,
    0x3ea46713, 0x296dcfb5, 0xc67cc521, 0x0b575f8c,
    0x4f450e50, 0xf06d6ed3, 0x70219d9c, 0xf022658e,
    0x483ec1f6, 0xf52bab4b, 0x3b43099f, 0xe50b76de,
    0x76dcf70c, 0x26b57ff4, 0x879110ef, 0x534b51f0,
    0x4475526a, 0x8c64b7e5, 0x35af90e3, 0xf4f72d2f,
    0xa0fff1fb, 0x85a23410, 0xfa6e9984, 0xa0c00f9f,
    0x8343fdc6, 0x7e91a3ae, 0x2ff6a24d, 0x59477d9b,
    0x99cd3d88, 0x18c1ddf9, 0x2855ad24, 0xd3f2ef22,
    0xd973d58d, 0x541a1ea4, 0x60498be8, 0x6e5b2226,
    0x3de165bd, 0x8e43fcf7, 0x55035fc1, 0x372e0652,
    0xdc3e474a, 0x74f42f0b, 0x7928e6bf, 0x767b5182,
    0xe9c7e8ab, 0xea2da551, 0x7e8a8bbf, 0xf6d31bb3,
    0x88a5a28b, 0x53beccb1, 0x57592cea, 0x77bd87d4,
];

struct Statement {
    program: CircuitProgram<F>,
    chiplet: ModexpChiplet,
    modexp: Modexp,
}

fn cpu_layout() -> Vec<ColumnType> {
    let mut layout = vec![ColumnType::B32; 3 * LIMBS32];
    layout.push(ColumnType::Bit);

    layout
}

fn build() -> Statement {
    let chiplet = ModexpChiplet::new().unwrap();
    let modexp = Modexp::new(&N, &S).unwrap();

    let mut cx = Circuit::<F>::new("ModexpCpu", CPU_ROWS).unwrap();

    let n = cx.columns(LIMBS32, ColumnType::B32);
    let s = cx.columns(LIMBS32, ColumnType::B32);
    let r = cx.columns(LIMBS32, ColumnType::B32);
    let active = cx.column(ColumnType::Bit);

    cx.fix(active, FixedShape::Sparse(vec![(0, F::ONE)]));

    let values: Vec<Col> = n.iter().chain(s.iter()).chain(r.iter()).collect();

    cx.call(&modexp::service(), &values, active).unwrap();

    for j in 0..LIMBS32 {
        cx.publish(r.at(j), 0);
    }

    cx.attach(chiplet.def().unwrap());

    let program = cx.compile().unwrap();

    assert_eq!(program.column_layout(), cpu_layout().as_slice());

    Statement {
        program,
        chiplet,
        modexp,
    }
}

fn cpu_trace(st: &Statement) -> ColumnTrace {
    let mut tb = TraceBuilder::new(&cpu_layout(), CPU_ROWS.trailing_zeros() as usize).unwrap();

    let groups = [
        (CPU_N, st.modexp.modulus()),
        (CPU_S, st.modexp.base()),
        (CPU_R, st.modexp.result()),
    ];

    for (base, limbs) in groups {
        for (j, &limb) in limbs.iter().enumerate() {
            tb.set_b32(base + j, 0, Block32::from(limb)).unwrap();
        }
    }

    tb.set_bit(CPU_ACTIVE, 0, Bit::ONE).unwrap();

    tb.build()
}

fn chiplet_trace(st: &Statement) -> ColumnTrace {
    st.chiplet.trace(&st.modexp, 0).unwrap()
}

fn instance(st: &Statement) -> ProgramInstance<F> {
    let public: Vec<F> = st.modexp.result().iter().map(|&l| F::from(l)).collect();

    ProgramInstance::new(CPU_ROWS, public)
}

fn witness(cpu: ColumnTrace, chiplet: ColumnTrace) -> ProgramWitness<F, ColumnTrace> {
    ProgramWitness::new(cpu).with_chiplets(vec![chiplet])
}

fn prove_and_verify(
    st: &Statement,
    cpu: ColumnTrace,
    chiplet: ColumnTrace,
) -> Result<bool, String> {
    let instance = instance(st);
    let witness = witness(cpu, chiplet);

    let report =
        preflight(&st.program, &instance, &witness).map_err(|e| format!("preflight: {e:?}"))?;

    if !report.is_clean() {
        for v in &report.constraint_violations {
            eprintln!("constraint={} row={}", v.constraint_idx, v.row_idx);
        }

        return Err("preflight violations".into());
    }

    verdict_without_preflight(st, instance, witness)
}

fn verdict_without_preflight(
    st: &Statement,
    instance: ProgramInstance<F>,
    witness: ProgramWitness<F, ColumnTrace>,
) -> Result<bool, String> {
    let config = Config {
        zero_knowledge: true,
        ..Config::dev()
    };

    let proof = prove(
        b"Modexp_E2E",
        &st.program,
        &instance,
        &witness,
        &config,
        [7u8; 32],
        None,
    )
    .map_err(|e| format!("prover: {e:?}"))?;

    let mut vt = Transcript::<H>::new(b"Modexp_E2E");
    let pinned_id = program_id(&st.program).unwrap();

    HekateVerifier::<F, H>::verify(&pinned_id, &st.program, &instance, &proof, &mut vt, &config)
        .map_err(|e| format!("verifier: {e:?}"))
}

fn flip_b32(trace: &mut ColumnTrace, col: usize, row: usize, mask: u32) {
    let TraceColumn::B32(data) = &mut trace.columns[col] else {
        panic!("expected B32 column at {col}");
    };

    let original = data[row].to_tower().0;
    data[row] = Block32(original ^ mask).to_hardware();
}

fn assert_verifier_rejects(st: &Statement, cpu: ColumnTrace, chiplet: ColumnTrace) {
    assert_eq!(
        verdict_without_preflight(st, instance(st), witness(cpu, chiplet)),
        Ok(false)
    );
}

#[test]
fn result_matches_reference_exponentiation() {
    let modexp = Modexp::new(&N, &S).unwrap();
    assert_eq!(modexp.result(), &R);
}

#[test]
fn unreduced_base_rejected() {
    assert!(Modexp::new(&S, &N).is_err());
}

#[test]
fn geometry_matches_prototype() {
    let chiplet = ModexpChiplet::new().unwrap();
    let ast = chiplet.program().constraint_ast();

    assert_eq!(NUM_ROWS, 1024);
    assert_eq!(RESULT_ROW, 543);
    assert_eq!(ast.roots.len(), 3684);
    assert_eq!(ast.max_degree(), 5);
    assert_eq!(chiplet.column_layout().len(), 3447);
    assert_eq!(chiplet.b128_columns(), 3002);
    assert_eq!(chiplet.row_bytes(), 49739);
}

#[test]
fn one_emit_on_result_row() {
    let chiplet = ModexpChiplet::new().unwrap();
    let specs = chiplet.program().permutation_checks();

    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].0, BUS_ID);
    assert_eq!(specs[0].1.num_sources(), 3 * LIMBS32 + 1);

    let modexp = Modexp::new(&N, &S).unwrap();
    let trace = chiplet.trace(&modexp, 0).unwrap();

    let TraceColumn::Bit(emit) = &trace.columns[chiplet.emit_column()] else {
        panic!("emit is not a Bit column");
    };

    assert_eq!(emit.iter().filter(|&&b| b == Bit::ONE).count(), 1);
    assert_eq!(emit[RESULT_ROW], Bit::ONE);
}

#[test]
fn honest_proof_verifies() {
    let st = build();

    match prove_and_verify(&st, cpu_trace(&st), chiplet_trace(&st)) {
        Ok(true) => {}
        Ok(false) => panic!("verifier rejected an honest proof"),
        Err(e) => panic!("{e}"),
    }
}

#[test]
fn flipped_remainder_bit_rejected() {
    let st = build();
    let mut chiplet = chiplet_trace(&st);

    flip_b32(&mut chiplet, st.chiplet.result_column(0), RESULT_ROW, 1);

    assert_verifier_rejects(&st, cpu_trace(&st), chiplet);
}

#[test]
fn forged_cpu_result_rejected() {
    let st = build();
    let mut cpu = cpu_trace(&st);

    flip_b32(&mut cpu, CPU_R, 0, 1);

    assert_verifier_rejects(&st, cpu, chiplet_trace(&st));
}

#[test]
fn scribble_flip_selector_caught() {
    let st = build();

    assert_all_caught_all_targets(
        &st.program,
        &instance(&st),
        &witness(cpu_trace(&st), chiplet_trace(&st)),
        ScribbleConfig::default()
            .cases(64)
            .mutations([MutationKind::FlipSelector]),
    );
}
