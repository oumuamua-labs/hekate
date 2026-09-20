// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Wire format round-trip tests.
//!
//! Serialize -> deserialize -> assert structural equality.
//! Covers:
//! main-trace-only, single chiplet with virtual
//! expansion, and multi-chiplet programs.

use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::proofs::{
    BrakedownCommitment, BrakedownProof, EvalBatchProof, InnerProof, LogUpAux, MasterEvals,
    OuterOpening, OuterProof, SumcheckProof,
};
use hekate_core::trace::ColumnTrace;
use hekate_core::trace::{ColumnType, Trace, TraceBuilder, TraceColumn};
use hekate_gadgets::{
    ArithmeticOpcode, CpuArithColumns, CpuFetchColumns, CpuMemColumns, Instruction,
    IntArithmeticChiplet, IntArithmeticLayout, IntArithmeticOp, MemoryEvent, RamChiplet,
    RomChiplet, generate_arithmetic_trace, generate_ram_trace, generate_rom_trace,
};
use hekate_math::{Bit, Block32, Block128, CanonicalDeserialize, CanonicalSerialize, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::Circuit;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::constraint::{
    BoundaryConstraint, BoundaryTarget, ConstraintAst, ConstraintExpr, ExprId,
};
use hekate_program::define_columns;
use hekate_program::digest::program_id;
use hekate_program::permutation::{BusKind, PermutationCheckSpec, Service, ServiceSlot, Source};
use hekate_program::{Air, FixedColumn, FixedShape, Program, ProgramInstance, ProgramWitness};
use hekate_sdk::{
    BundleProgram, DeserializedBundle, deserialize_bundle, deserialize_proof, serialize_bundle,
    serialize_proof_bytes,
};

type F = Block128;

// =================================================================
// Test Programs
// =================================================================

// 1. Fibonacci (main trace only, no chiplets)

define_columns! {
    FibColumns {
        A: B32,
        B: B32,
        Q: Bit,
    }
}

#[derive(Clone)]
struct FibProgram {
    num_rows: usize,
}

impl Air<F> for FibProgram {
    fn num_columns(&self) -> usize {
        FibColumns::NUM_COLUMNS
    }

    fn boundary_constraints(&self) -> Vec<BoundaryConstraint<F>> {
        vec![BoundaryConstraint {
            col_idx: FibColumns::B,
            row_idx: self.num_rows - 1,
            target: BoundaryTarget::PublicInput(0),
        }]
    }

    fn column_layout(&self) -> &[ColumnType] {
        &[ColumnType::B32, ColumnType::B32, ColumnType::Bit]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::new();

        let [a, b, q] = [
            cs.col(FibColumns::A),
            cs.col(FibColumns::B),
            cs.col(FibColumns::Q),
        ];
        let [na, nb] = [cs.next(FibColumns::A), cs.next(FibColumns::B)];

        cs.constrain(q * (na + b));
        cs.constrain(q * (nb + a + b));

        cs.build()
    }
}

impl Program<F> for FibProgram {
    fn num_public_inputs(&self) -> usize {
        1
    }
}

fn fib_trace(num_vars: usize) -> ColumnTrace {
    let num_rows = 1 << num_vars;

    let mut tb = TraceBuilder::new(&FibColumns::build_layout(), num_vars).unwrap();

    let (mut a, mut b) = (Block32::ZERO, Block32::ONE);
    for i in 0..num_rows {
        tb.set_b32(FibColumns::A, i, a).unwrap();
        tb.set_b32(FibColumns::B, i, b).unwrap();

        let temp = a + b;
        a = b;
        b = temp;
    }

    tb.fill_selector(FibColumns::Q, num_rows - 1).unwrap();

    tb.build()
}

// 2. RAM isolated chiplet (virtual expansion + LogUp)

#[derive(Clone)]
struct RamProgram {
    ram_num_rows: usize,
    ram_num_events: usize,
}

impl Air<F> for RamProgram {
    fn num_columns(&self) -> usize {
        CpuMemColumns::NUM_COLUMNS
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(CpuMemColumns::build_layout)
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![(RamChiplet::BUS_ID.into(), RamChiplet::cpu_linking_spec())]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(CpuMemColumns::SELECTOR));
        cs.assert_boolean(cs.col(CpuMemColumns::IS_WRITE));

        cs.build()
    }
}

impl Program<F> for RamProgram {
    fn chiplet_defs(&self) -> errors::Result<Vec<ChipletDef<F>>> {
        let ram = RamChiplet::new(self.ram_num_rows, self.ram_num_events);
        Ok(vec![ChipletDef::from_air(&ram)?])
    }
}

fn ram_traces(num_rows: usize) -> (ColumnTrace, ColumnTrace) {
    let num_vars = num_rows.trailing_zeros() as usize;
    let mut tb = TraceBuilder::new(&CpuMemColumns::build_layout(), num_vars).unwrap();

    let events = vec![
        MemoryEvent::write(0x1000, 0, 42),
        MemoryEvent::write(0x2000, 1, 99),
        MemoryEvent::read(0x1000, 2, 42),
        MemoryEvent::read(0x2000, 3, 99),
    ];

    for (i, event) in events.iter().enumerate() {
        let addr = event.addr_bytes();
        let val = event.val_bytes();

        for j in 0..4 {
            tb.set_b32(CpuMemColumns::ADDR_B0 + j, i, Block32::from(addr[j] as u32))
                .unwrap();
            tb.set_b32(CpuMemColumns::VAL_B0 + j, i, Block32::from(val[j] as u32))
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

    let cpu = tb.build();
    let ram = generate_ram_trace(&events, num_rows).unwrap();

    (cpu, ram)
}

// 3. Many chiplets (ROM + Arithmetic + RAM)

const MC_CPU_FETCH: usize = 0;
const MC_CPU_ARITH: usize = CpuFetchColumns::NUM_COLUMNS;
const MC_CPU_MEM: usize = MC_CPU_ARITH + CpuArithColumns::NUM_COLUMNS;
const MC_NUM_CPU_COLS: usize = MC_CPU_MEM + CpuMemColumns::NUM_COLUMNS;

#[derive(Clone)]
struct ManyChipletsProgram {
    num_ops: usize,
    rom_num_rows: usize,
    arith_num_rows: usize,
    ram_num_rows: usize,
}

impl Air<F> for ManyChipletsProgram {
    fn num_columns(&self) -> usize {
        MC_NUM_CPU_COLS
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(|| {
            let mut cols = CpuFetchColumns::build_layout();
            cols.extend(CpuArithColumns::build_layout());
            cols.extend(CpuMemColumns::build_layout());

            cols
        })
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        let cpu_fetch = RomChiplet::cpu_linking_spec();

        let mut cpu_arith = IntArithmeticChiplet::cpu_linking_spec();
        cpu_arith.shift_column_indices(MC_CPU_ARITH);

        let mut cpu_mem = RamChiplet::cpu_linking_spec();
        cpu_mem.shift_column_indices(MC_CPU_MEM);

        vec![
            (RomChiplet::BUS_ID.into(), cpu_fetch),
            (IntArithmeticChiplet::BUS_ID.into(), cpu_arith),
            (RamChiplet::BUS_ID.into(), cpu_mem),
        ]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();

        cs.assert_boolean(cs.col(MC_CPU_FETCH + CpuFetchColumns::SELECTOR));
        cs.assert_boolean(cs.col(MC_CPU_ARITH + CpuArithColumns::SELECTOR));
        cs.assert_boolean(cs.col(MC_CPU_MEM + CpuMemColumns::SELECTOR));

        cs.build()
    }
}

impl Program<F> for ManyChipletsProgram {
    fn chiplet_defs(&self) -> errors::Result<Vec<ChipletDef<F>>> {
        let rom = RomChiplet::new(self.rom_num_rows, self.num_ops);
        let arith = IntArithmeticChiplet::new(32, self.arith_num_rows, self.num_ops)
            .expect("IntArithmeticChiplet::new(32, arith_num_rows, num_ops)");
        let ram = RamChiplet::new(self.ram_num_rows, self.num_ops);

        Ok(vec![
            ChipletDef::from_air(&rom)?,
            ChipletDef::from_air(&arith)?,
            ChipletDef::from_air(&ram)?,
        ])
    }
}

fn many_chiplets_traces(
    num_ops: usize,
    cpu_rows: usize,
    rom_rows: usize,
    arith_rows: usize,
    ram_rows: usize,
) -> (ColumnTrace, Vec<ColumnTrace>) {
    let mut instrs = Vec::new();
    let mut ariths = Vec::new();
    let mut mems = Vec::new();

    for i in 0..num_ops {
        instrs.push(Instruction::new((i * 4) as u32, 0x01, [0, 0, 0]));

        let val_a = (i * 100) as u32;
        let val_b = 0xAABB;

        ariths.push(IntArithmeticOp::U32 {
            op: ArithmeticOpcode::ADD,
            a: val_a,
            b: val_b,
            request_idx: i as u32,
        });

        let result = val_a.wrapping_add(val_b);
        mems.push(MemoryEvent::write((i * 4) as u32, (i * 4) as u32, result));
    }

    let cpu_vars = cpu_rows.trailing_zeros() as usize;

    let mut layout = CpuFetchColumns::build_layout();
    layout.extend(CpuArithColumns::build_layout());
    layout.extend(CpuMemColumns::build_layout());

    let mut tb = TraceBuilder::new(&layout, cpu_vars).unwrap();
    for (idx, _instr) in instrs.iter().enumerate() {
        let r = idx;
        if r >= cpu_rows {
            break;
        }

        let instr = &instrs[idx];
        let pc = instr.pc_bytes();
        let args = instr.args();

        tb.set_b32(
            MC_CPU_FETCH + CpuFetchColumns::PC_B0,
            r,
            Block32::from(pc[0] as u32),
        )
        .unwrap();
        tb.set_b32(
            MC_CPU_FETCH + CpuFetchColumns::PC_B1,
            r,
            Block32::from(pc[1] as u32),
        )
        .unwrap();
        tb.set_b32(
            MC_CPU_FETCH + CpuFetchColumns::PC_B2,
            r,
            Block32::from(pc[2] as u32),
        )
        .unwrap();
        tb.set_b32(
            MC_CPU_FETCH + CpuFetchColumns::PC_B3,
            r,
            Block32::from(pc[3] as u32),
        )
        .unwrap();
        tb.set_b32(
            MC_CPU_FETCH + CpuFetchColumns::OPCODE,
            r,
            Block32::from(instr.opcode as u32),
        )
        .unwrap();
        tb.set_b32(
            MC_CPU_FETCH + CpuFetchColumns::ARG0,
            r,
            Block32::from(args[0] as u32),
        )
        .unwrap();
        tb.set_b32(
            MC_CPU_FETCH + CpuFetchColumns::ARG1,
            r,
            Block32::from(args[1] as u32),
        )
        .unwrap();
        tb.set_b32(
            MC_CPU_FETCH + CpuFetchColumns::ARG2,
            r,
            Block32::from(args[2] as u32),
        )
        .unwrap();
        tb.set_bit(MC_CPU_FETCH + CpuFetchColumns::SELECTOR, r, Bit::ONE)
            .unwrap();

        let IntArithmeticOp::U32 { op, a, b, .. } = ariths[idx] else {
            unreachable!("wire_roundtrip many_chiplets is u32-only");
        };

        tb.set_b32(MC_CPU_ARITH + CpuArithColumns::VAL_A, r, Block32::from(a))
            .unwrap();
        tb.set_b32(MC_CPU_ARITH + CpuArithColumns::VAL_B, r, Block32::from(b))
            .unwrap();

        let res = a.wrapping_add(b);
        tb.set_b32(
            MC_CPU_ARITH + CpuArithColumns::VAL_RES,
            r,
            Block32::from(res),
        )
        .unwrap();
        tb.set_b32(
            MC_CPU_ARITH + CpuArithColumns::OPCODE,
            r,
            Block32::from(op as u8 as u32),
        )
        .unwrap();
        tb.set_bit(MC_CPU_ARITH + CpuArithColumns::SELECTOR, r, Bit::ONE)
            .unwrap();

        let mem = &mems[idx];

        let addr = mem.addr_bytes();
        let val = mem.val_bytes();

        tb.set_b32(
            MC_CPU_MEM + CpuMemColumns::ADDR_B0,
            r,
            Block32::from(addr[0] as u32),
        )
        .unwrap();
        tb.set_b32(
            MC_CPU_MEM + CpuMemColumns::ADDR_B1,
            r,
            Block32::from(addr[1] as u32),
        )
        .unwrap();
        tb.set_b32(
            MC_CPU_MEM + CpuMemColumns::ADDR_B2,
            r,
            Block32::from(addr[2] as u32),
        )
        .unwrap();
        tb.set_b32(
            MC_CPU_MEM + CpuMemColumns::ADDR_B3,
            r,
            Block32::from(addr[3] as u32),
        )
        .unwrap();
        tb.set_b32(
            MC_CPU_MEM + CpuMemColumns::VAL_B0,
            r,
            Block32::from(val[0] as u32),
        )
        .unwrap();
        tb.set_b32(
            MC_CPU_MEM + CpuMemColumns::VAL_B1,
            r,
            Block32::from(val[1] as u32),
        )
        .unwrap();
        tb.set_b32(
            MC_CPU_MEM + CpuMemColumns::VAL_B2,
            r,
            Block32::from(val[2] as u32),
        )
        .unwrap();
        tb.set_b32(
            MC_CPU_MEM + CpuMemColumns::VAL_B3,
            r,
            Block32::from(val[3] as u32),
        )
        .unwrap();
        tb.set_bit(MC_CPU_MEM + CpuMemColumns::IS_WRITE, r, Bit::ONE)
            .unwrap();
        tb.set_bit(MC_CPU_MEM + CpuMemColumns::SELECTOR, r, Bit::ONE)
            .unwrap();
    }

    let cpu_trace = tb.build();
    let rom_trace = generate_rom_trace(&instrs, rom_rows).unwrap();
    let ram_trace = generate_ram_trace(&mems, ram_rows).unwrap();

    let arith_layout = IntArithmeticLayout::compute(32);
    let arith_trace = generate_arithmetic_trace(&ariths, &arith_layout, arith_rows).unwrap();

    (cpu_trace, vec![rom_trace, arith_trace, ram_trace])
}

// =================================================================
// Assertion Helpers
// =================================================================

fn assert_layout_eq(a: &[ColumnType], b: &[ColumnType], ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx}: layout length mismatch");

    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x, y, "{ctx}: column {i} type mismatch");
    }
}

fn assert_trace_eq(original: &ColumnTrace, restored: &ColumnTrace, ctx: &str) {
    let orig_cols = original.columns();
    let rest_cols = restored.columns();

    assert_eq!(
        orig_cols.len(),
        rest_cols.len(),
        "{ctx}: column count mismatch"
    );

    for (i, (oc, rc)) in orig_cols.iter().zip(rest_cols.iter()).enumerate() {
        assert_eq!(
            oc.column_type(),
            rc.column_type(),
            "{ctx}: column {i} type mismatch"
        );
        assert_trace_column_data_eq(oc, rc, &format!("{ctx} col {i}"));
    }
}

fn assert_trace_column_data_eq(a: &TraceColumn, b: &TraceColumn, ctx: &str) {
    match (a, b) {
        (TraceColumn::Bit(va), TraceColumn::Bit(vb)) => {
            assert_eq!(va.len(), vb.len(), "{ctx}: Bit length");
            assert_eq!(va, vb, "{ctx}: Bit data mismatch");
        }
        (TraceColumn::B8(va), TraceColumn::B8(vb)) => {
            assert_eq!(va.len(), vb.len(), "{ctx}: B8 length");
            assert_eq!(va, vb, "{ctx}: B8 data mismatch");
        }
        (TraceColumn::B16(va), TraceColumn::B16(vb)) => {
            assert_eq!(va.len(), vb.len(), "{ctx}: B16 length");
            assert_eq!(va, vb, "{ctx}: B16 data mismatch");
        }
        (TraceColumn::B32(va), TraceColumn::B32(vb)) => {
            assert_eq!(va.len(), vb.len(), "{ctx}: B32 length");
            assert_eq!(va, vb, "{ctx}: B32 data mismatch");
        }
        (TraceColumn::B64(va), TraceColumn::B64(vb)) => {
            assert_eq!(va.len(), vb.len(), "{ctx}: B64 length");
            assert_eq!(va, vb, "{ctx}: B64 data mismatch");
        }
        (TraceColumn::B128(va), TraceColumn::B128(vb)) => {
            assert_eq!(va.len(), vb.len(), "{ctx}: B128 length");
            assert_eq!(va, vb, "{ctx}: B128 data mismatch");
        }
        _ => panic!("{ctx}: column type mismatch in data comparison"),
    }
}

fn assert_ast_eq(original: &ConstraintAst<F>, restored: &ConstraintAst<F>, ctx: &str) {
    assert_eq!(
        original.arena.len(),
        restored.arena.len(),
        "{ctx}: AST node count"
    );
    assert_eq!(
        original.roots.len(),
        restored.roots.len(),
        "{ctx}: AST root count"
    );

    for i in 0..original.roots.len() {
        assert_eq!(
            original.roots[i], restored.roots[i],
            "{ctx}: root {i} index"
        );
    }

    for i in 0..original.arena.len() {
        let id = ExprId(i as u32);
        let orig = original.arena.get(id);
        let rest = restored.arena.get(id);

        assert_expr_eq(orig, rest, &format!("{ctx} node {i}"));
    }
}

fn assert_expr_eq(a: &ConstraintExpr<F>, b: &ConstraintExpr<F>, ctx: &str) {
    match (a, b) {
        (ConstraintExpr::Cell(ca), ConstraintExpr::Cell(cb)) => {
            assert_eq!(ca.col_idx, cb.col_idx, "{ctx}: Cell col_idx");
            assert_eq!(ca.next_row, cb.next_row, "{ctx}: Cell next_row");
        }
        (ConstraintExpr::Const(va), ConstraintExpr::Const(vb)) => {
            assert_eq!(va.to_bytes(), vb.to_bytes(), "{ctx}: Const value");
        }
        (ConstraintExpr::Add(la, ra), ConstraintExpr::Add(lb, rb)) => {
            assert_eq!(la, lb, "{ctx}: Add left");
            assert_eq!(ra, rb, "{ctx}: Add right");
        }
        (ConstraintExpr::Mul(la, ra), ConstraintExpr::Mul(lb, rb)) => {
            assert_eq!(la, lb, "{ctx}: Mul left");
            assert_eq!(ra, rb, "{ctx}: Mul right");
        }
        (ConstraintExpr::Scale(sa, ca), ConstraintExpr::Scale(sb, cb)) => {
            assert_eq!(sa.to_bytes(), sb.to_bytes(), "{ctx}: Scale scalar");
            assert_eq!(ca, cb, "{ctx}: Scale child");
        }
        (ConstraintExpr::Sum(ca), ConstraintExpr::Sum(cb)) => {
            assert_eq!(ca.len(), cb.len(), "{ctx}: Sum children count");

            for (i, (x, y)) in ca.iter().zip(cb.iter()).enumerate() {
                assert_eq!(x, y, "{ctx}: Sum child {i}");
            }
        }
        _ => panic!("{ctx}: expression type mismatch: {a:?} vs {b:?}"),
    }
}

fn assert_bundle_eq<P: Program<F>>(
    program: &P,
    bundle: &DeserializedBundle<F>,
    witness: &ProgramWitness<F>,
    ctx: &str,
) {
    assert_eq!(
        program.num_columns(),
        bundle.num_columns,
        "{ctx}: num_columns"
    );
    assert_eq!(
        program.num_public_inputs(),
        bundle.num_public_inputs,
        "{ctx}: num_public_inputs"
    );

    assert_layout_eq(program.column_layout(), &bundle.column_layout, ctx);
    assert_layout_eq(
        program.virtual_column_layout(),
        &bundle.virtual_column_layout,
        &format!("{ctx} virtual"),
    );

    let orig_ast = program.constraint_ast();
    assert_ast_eq(&orig_ast, &bundle.constraint_ast, ctx);

    let orig_bounds = program.boundary_constraints();
    assert_eq!(
        orig_bounds.len(),
        bundle.boundary_constraints.len(),
        "{ctx}: boundary count"
    );
    for (i, (ob, rb)) in orig_bounds
        .iter()
        .zip(&bundle.boundary_constraints)
        .enumerate()
    {
        assert_eq!(ob.col_idx, rb.col_idx, "{ctx}: boundary {i} col_idx");
        assert_eq!(ob.row_idx, rb.row_idx, "{ctx}: boundary {i} row_idx");
        assert_eq!(ob.target, rb.target, "{ctx}: boundary {i} target");
    }

    let orig_perms = program.permutation_checks();
    assert_eq!(
        orig_perms.len(),
        bundle.permutation_checks.len(),
        "{ctx}: permutation check count"
    );

    for (i, ((ob_id, ob_spec), (rb_id, rb_spec))) in orig_perms
        .iter()
        .zip(&bundle.permutation_checks)
        .enumerate()
    {
        assert_eq!(ob_id, rb_id, "{ctx}: bus_id {i}");
        assert_eq!(ob_spec.kind, rb_spec.kind, "{ctx}: bus {i} kind");
        assert_eq!(
            ob_spec.selector, rb_spec.selector,
            "{ctx}: bus {i} selector"
        );
        assert_eq!(
            ob_spec.recv_selector, rb_spec.recv_selector,
            "{ctx}: bus {i} recv_selector"
        );
    }

    let orig_chiplets = program.chiplet_defs().expect("chiplet_defs");
    assert_eq!(
        orig_chiplets.len(),
        bundle.chiplet_defs.len(),
        "{ctx}: chiplet count"
    );

    for (i, (oc, rc)) in orig_chiplets.iter().zip(&bundle.chiplet_defs).enumerate() {
        let oc_name = Air::<F>::name(oc);
        let rc_name = Air::<F>::name(rc);

        assert_eq!(oc_name, rc_name, "{ctx}: chiplet {i} name");
        assert_eq!(
            oc.num_columns(),
            rc.num_columns(),
            "{ctx}: chiplet {i} num_columns"
        );

        assert_layout_eq(
            oc.column_layout(),
            rc.column_layout(),
            &format!("{ctx} chiplet {i}"),
        );

        let oc_ast = oc.constraint_ast();
        let rc_ast = rc.constraint_ast();

        assert_ast_eq(&oc_ast, &rc_ast, &format!("{ctx} chiplet {i}"));

        let oc_bounds = Air::<F>::boundary_constraints(oc);
        let rc_bounds = Air::<F>::boundary_constraints(rc);

        assert_eq!(
            oc_bounds.len(),
            rc_bounds.len(),
            "{ctx}: chiplet {i} boundary count"
        );

        for (j, (ob, rb)) in oc_bounds.iter().zip(&rc_bounds).enumerate() {
            assert_eq!(ob.col_idx, rb.col_idx, "{ctx}: chiplet {i} bnd {j} col_idx");
            assert_eq!(ob.row_idx, rb.row_idx, "{ctx}: chiplet {i} bnd {j} row_idx");
            assert_eq!(ob.target, rb.target, "{ctx}: chiplet {i} bnd {j} target");
        }
    }

    assert_trace_eq(
        &witness.trace,
        &bundle.witness.trace,
        &format!("{ctx} main trace"),
    );

    assert_eq!(
        witness.chiplet_traces.len(),
        bundle.witness.chiplet_traces.len(),
        "{ctx}: chiplet trace count"
    );

    for (i, (ot, rt)) in witness
        .chiplet_traces
        .iter()
        .zip(&bundle.witness.chiplet_traces)
        .enumerate()
    {
        assert_trace_eq(ot, rt, &format!("{ctx} chiplet trace {i}"));
    }
}

fn wide(i: u128) -> F {
    F::from((i << 64) | (i.wrapping_mul(0x9E37_79B9) + 1))
}

// =================================================================
// Tests
// =================================================================

#[test]
fn roundtrip_fibonacci_main_trace_only() {
    let num_vars = 10;
    let num_rows = 1 << num_vars;

    let program = FibProgram { num_rows };
    let trace = fib_trace(num_vars);

    let expected_result = trace.get_element::<F>(1, num_rows - 1).unwrap().to_tower();
    let instance = ProgramInstance::new(num_rows, vec![expected_result]);
    let witness = ProgramWitness::new(trace);
    let config = Config::default();

    let bytes = serialize_bundle(&program, &instance, &witness, &config).unwrap();
    let restored: DeserializedBundle<F> = deserialize_bundle(&bytes).unwrap();

    assert_bundle_eq(&program, &restored, &witness, "fibonacci");

    assert_eq!(restored.instance.num_rows(), num_rows);
    assert_eq!(restored.instance.public_inputs().len(), 1);
    assert_eq!(
        restored.instance.public_inputs()[0].to_bytes(),
        expected_result.to_bytes()
    );
}

#[test]
fn roundtrip_ram_isolated_chiplet() {
    let num_rows = 1 << 10;

    let program = RamProgram {
        ram_num_rows: num_rows,
        ram_num_events: 4,
    };

    let (cpu_trace, ram_trace) = ram_traces(num_rows);

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(vec![ram_trace]);

    let config = Config::default();

    let bytes = serialize_bundle(&program, &instance, &witness, &config).unwrap();
    let restored: DeserializedBundle<F> = deserialize_bundle(&bytes).unwrap();

    assert_bundle_eq(&program, &restored, &witness, "ram");

    assert_eq!(restored.chiplet_defs.len(), 1);
    assert!(
        restored.chiplet_defs[0].virtual_expander().is_some(),
        "ram chiplet must have virtual expander"
    );
}

#[test]
fn roundtrip_many_chiplets() {
    let num_ops = 8;
    let cpu_rows = 1 << 8;
    let rom_rows = 1 << 8;
    let arith_rows = 1 << 8;
    let ram_rows = 1 << 10;

    let program = ManyChipletsProgram {
        num_ops,
        rom_num_rows: rom_rows,
        arith_num_rows: arith_rows,
        ram_num_rows: ram_rows,
    };

    let (cpu_trace, chiplet_traces) =
        many_chiplets_traces(num_ops, cpu_rows, rom_rows, arith_rows, ram_rows);

    let instance = ProgramInstance::new(cpu_rows, vec![]);
    let witness = ProgramWitness::new(cpu_trace).with_chiplets(chiplet_traces);
    let config = Config::default();

    let bytes = serialize_bundle(&program, &instance, &witness, &config).unwrap();
    let restored: DeserializedBundle<F> = deserialize_bundle(&bytes).unwrap();

    assert_bundle_eq(&program, &restored, &witness, "many_chiplets");

    assert_eq!(restored.chiplet_defs.len(), 3, "ROM + Arith + RAM");
    assert_eq!(restored.witness.chiplet_traces.len(), 3);
    assert_eq!(restored.permutation_checks.len(), 3, "3 LogUp buses");
}

#[test]
fn malformed_bundle_returns_error() {
    let result = deserialize_bundle::<F>(&[0xFF, 0x00, 0x42]);
    assert!(result.is_err(), "garbage bytes must fail");

    let result = deserialize_bundle::<F>(&[]);
    assert!(result.is_err(), "empty bytes must fail");
}

#[test]
fn config_roundtrip() {
    let num_rows = 1 << 10;
    let program = FibProgram { num_rows };
    let trace = fib_trace(10);
    let result = trace.get_element::<F>(1, num_rows - 1).unwrap().to_tower();
    let instance = ProgramInstance::new(num_rows, vec![result]);
    let witness = ProgramWitness::new(trace);

    let config = Config {
        num_queries: 200,
        ldt_support_size: 250,
        min_security_bits: 120,
        outer_queries: 300,
        zero_knowledge: false,
    };

    let bytes = serialize_bundle(&program, &instance, &witness, &config).unwrap();
    let restored: DeserializedBundle<F> = deserialize_bundle(&bytes).unwrap();

    assert_eq!(restored.config.num_queries, 200);
    assert_eq!(restored.config.ldt_support_size, 250);
    assert_eq!(restored.config.min_security_bits, 120);
    assert_eq!(restored.config.outer_queries, 300);
    assert!(!restored.config.zero_knowledge);
}

#[test]
fn serialize_bundle_preserves_raw_config() {
    let num_vars = 10;
    let num_rows = 1 << num_vars;
    let program = FibProgram { num_rows };

    let trace = fib_trace(num_vars);
    let result = trace.get_element::<F>(1, num_rows - 1).unwrap().to_tower();

    let instance = ProgramInstance::new(num_rows, vec![result]);
    let witness = ProgramWitness::new(trace);

    let config = Config {
        ldt_support_size: 199,
        ..Config::default()
    };

    let bytes = serialize_bundle(&program, &instance, &witness, &config).unwrap();
    let restored: DeserializedBundle<F> = deserialize_bundle(&bytes).unwrap();

    assert_eq!(
        restored.config.ldt_support_size, 199,
        "serialize_bundle must preserve the caller's config verbatim"
    );

    assert_bundle_eq(&program, &restored, &witness, "raw_serialize");
}

// =================================================================
// BusKind round-trip
// =================================================================

define_columns! {
    MixedKindCols {
        A: B32,
        B: B32,
        SEL_PERM: Bit,
        SEL_LOOKUP: Bit,
    }
}

#[derive(Clone)]
struct MixedKindProgram;

impl Air<F> for MixedKindProgram {
    fn num_columns(&self) -> usize {
        MixedKindCols::NUM_COLUMNS
    }

    fn column_layout(&self) -> &[ColumnType] {
        &[
            ColumnType::B32,
            ColumnType::B32,
            ColumnType::Bit,
            ColumnType::Bit,
        ]
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![
            (
                "perm_bus".into(),
                PermutationCheckSpec::new(
                    vec![(Source::Column(MixedKindCols::A), b"k_a")],
                    Some(MixedKindCols::SEL_PERM),
                )
                .with_clock_waiver(
                    "see wire_roundtrip.rs: BusKind round-trip only, not run through prover",
                ),
            ),
            (
                "lookup_bus".into(),
                PermutationCheckSpec::new_lookup(
                    vec![(Source::Column(MixedKindCols::B), b"k_b")],
                    Some(MixedKindCols::SEL_LOOKUP),
                ),
            ),
        ]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(MixedKindCols::SEL_PERM));
        cs.assert_boolean(cs.col(MixedKindCols::SEL_LOOKUP));

        cs.build()
    }
}

impl Program<F> for MixedKindProgram {}

#[test]
fn bus_kind_round_trips_through_wire() {
    let num_vars = 4;
    let num_rows = 1 << num_vars;

    let layout = MixedKindCols::build_layout();
    let trace = TraceBuilder::new(&layout, num_vars).unwrap().build();

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(trace);
    let config = Config::default();

    let bytes = serialize_bundle(&MixedKindProgram, &instance, &witness, &config).unwrap();
    let restored: DeserializedBundle<F> = deserialize_bundle(&bytes).unwrap();

    let perms = &restored.permutation_checks;
    assert_eq!(perms.len(), 2);

    let by_id: std::collections::HashMap<&str, BusKind> = perms
        .iter()
        .map(|(id, spec)| (id.as_str(), spec.kind))
        .collect();

    assert_eq!(by_id["perm_bus"], BusKind::Permutation);
    assert_eq!(by_id["lookup_bus"], BusKind::Lookup);
}

// =================================================================
// clock_waiver round-trip
// =================================================================

define_columns! {
    WaiverCols {
        A: B32,
        SEL: Bit,
    }
}

#[derive(Clone)]
struct WaiverProgram;

impl Air<F> for WaiverProgram {
    fn num_columns(&self) -> usize {
        WaiverCols::NUM_COLUMNS
    }

    fn column_layout(&self) -> &[ColumnType] {
        &[ColumnType::B32, ColumnType::Bit]
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![
            (
                "no_waiver_bus".into(),
                PermutationCheckSpec::new(
                    vec![
                        (Source::Column(WaiverCols::A), b"k_a"),
                        (Source::RowIndexLeBytes(4), b"k_clk"),
                    ],
                    Some(WaiverCols::SEL),
                ),
            ),
            (
                "waiver_bus".into(),
                PermutationCheckSpec::new(
                    vec![(Source::Column(WaiverCols::A), b"k_a")],
                    Some(WaiverCols::SEL),
                )
                .with_clock_waiver("see waiver_program.rs:1: structurally unique by AIR body"),
            ),
        ]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();
        cs.assert_boolean(cs.col(WaiverCols::SEL));

        cs.build()
    }
}

impl Program<F> for WaiverProgram {}

#[test]
fn clock_waiver_round_trips_through_wire() {
    let num_vars = 4;
    let num_rows = 1 << num_vars;

    let layout = WaiverCols::build_layout();
    let trace = TraceBuilder::new(&layout, num_vars).unwrap().build();

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(trace);
    let config = Config::default();

    let bytes = serialize_bundle(&WaiverProgram, &instance, &witness, &config).unwrap();
    let restored: DeserializedBundle<F> = deserialize_bundle(&bytes).unwrap();

    let perms = &restored.permutation_checks;
    assert_eq!(perms.len(), 2);

    let by_id: std::collections::HashMap<&str, &PermutationCheckSpec> =
        perms.iter().map(|(id, spec)| (id.as_str(), spec)).collect();

    assert_eq!(by_id["no_waiver_bus"].clock_waiver, None);
    assert_eq!(
        by_id["waiver_bus"].clock_waiver.as_deref(),
        Some("see waiver_program.rs:1: structurally unique by AIR body")
    );
}

// =================================================================
// FIXED COLUMN ROUND-TRIP
// =================================================================

define_columns! {
    PinCols {
        A: Bit,
        B: Bit,
        C: Bit,
    }
}

#[derive(Clone)]
struct PinProgram {
    pins: Vec<FixedColumn<F>>,
}

impl Air<F> for PinProgram {
    fn num_columns(&self) -> usize {
        PinCols::NUM_COLUMNS
    }

    fn column_layout(&self) -> &[ColumnType] {
        &[ColumnType::Bit, ColumnType::Bit, ColumnType::Bit]
    }

    fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
        self.pins.clone()
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
    }
}

impl Program<F> for PinProgram {}

fn pin_bundle_roundtrip(pins: Vec<FixedColumn<F>>) -> Vec<FixedColumn<F>> {
    let num_vars = 3;
    let num_rows = 1 << num_vars;
    let layout = PinCols::build_layout();
    let trace = TraceBuilder::new(&layout, num_vars).unwrap().build();

    let program = PinProgram { pins };
    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(trace);
    let config = Config::default();

    let bytes = serialize_bundle(&program, &instance, &witness, &config).unwrap();
    let restored: DeserializedBundle<F> = deserialize_bundle(&bytes).unwrap();

    restored.fixed_columns
}

#[test]
fn fixed_column_last_row_round_trips() {
    let pins = vec![FixedColumn::last_row(PinCols::A)];
    let restored = pin_bundle_roundtrip(pins.clone());

    assert_eq!(restored, pins);
}

#[test]
fn fixed_column_first_row_round_trips() {
    let pins = vec![FixedColumn::first_row(PinCols::B)];
    let restored = pin_bundle_roundtrip(pins.clone());

    assert_eq!(restored, pins);
}

#[test]
fn fixed_column_custom_round_trips() {
    let bits = vec![true, false, true];
    let pins = vec![FixedColumn::custom(PinCols::C, bits.clone())];
    let restored = pin_bundle_roundtrip(pins.clone());

    assert_eq!(restored.len(), 1);
    assert_eq!(restored[0].col_idx, PinCols::C);

    match &restored[0].shape {
        FixedShape::Custom(b) => assert_eq!(b, &bits),
        other => panic!("expected Custom, got {:?}", other),
    }
}

#[test]
fn fixed_column_mixed_variants_round_trip_in_order() {
    let bits = vec![false, true, false];
    let pins = vec![
        FixedColumn::last_row(PinCols::A),
        FixedColumn::first_row(PinCols::B),
        FixedColumn::custom(PinCols::C, bits.clone()),
    ];

    let restored = pin_bundle_roundtrip(pins.clone());
    assert_eq!(restored, pins);
}

#[test]
fn fixed_column_empty_round_trips() {
    let restored = pin_bundle_roundtrip(Vec::new());
    assert!(restored.is_empty());
}

#[test]
fn fixed_column_dense_round_trips() {
    let cols = vec![FixedColumn::dense(PinCols::A, (0..8).map(wide).collect())];
    assert_eq!(pin_bundle_roundtrip(cols.clone()), cols);
}

#[test]
fn fixed_column_periodic_round_trips() {
    let cols = vec![FixedColumn::periodic(
        PinCols::B,
        4,
        vec![wide(1), wide(2), wide(3), wide(4)],
    )];

    assert_eq!(pin_bundle_roundtrip(cols.clone()), cols);
}

#[test]
fn fixed_column_sparse_round_trips() {
    let cols = vec![FixedColumn::sparse(
        PinCols::C,
        vec![(0, wide(7)), (3, wide(11)), (7, wide(13))],
    )];

    assert_eq!(pin_bundle_roundtrip(cols.clone()), cols);
}

#[test]
fn fixed_column_cadence_round_trips() {
    let cols = vec![hekate_program::fix(
        PinCols::A,
        FixedShape::Cadence {
            stride: 3,
            count: 2,
            origin: 1,
            values: vec![wide(1), wide(0), wide(1)],
        },
    )];

    assert_eq!(pin_bundle_roundtrip(cols.clone()), cols);
}

// =================================================================
// Chiplet boundary (BoundaryTarget::Constant) round-trip
// =================================================================

define_columns! {
    BndChipletCols {
        FLAG: Bit,
    }
}

#[derive(Clone)]
struct BndChiplet {
    pinned: F,
}

impl Air<F> for BndChiplet {
    fn num_columns(&self) -> usize {
        BndChipletCols::NUM_COLUMNS
    }

    fn boundary_constraints(&self) -> Vec<BoundaryConstraint<F>> {
        vec![BoundaryConstraint::with_constant(
            BndChipletCols::FLAG,
            0,
            self.pinned,
        )]
    }

    fn column_layout(&self) -> &[ColumnType] {
        &[ColumnType::Bit]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
    }
}

#[derive(Clone)]
struct BndHostProgram {
    pinned: F,
}

impl Air<F> for BndHostProgram {
    fn num_columns(&self) -> usize {
        1
    }

    fn column_layout(&self) -> &[ColumnType] {
        &[ColumnType::Bit]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
    }
}

impl Program<F> for BndHostProgram {
    fn chiplet_defs(&self) -> errors::Result<Vec<ChipletDef<F>>> {
        Ok(vec![ChipletDef::from_air(&BndChiplet {
            pinned: self.pinned,
        })?])
    }
}

fn chiplet_bnd_roundtrip(pinned: F) -> F {
    let num_vars = 3;
    let num_rows = 1 << num_vars;

    let main = TraceBuilder::new(&[ColumnType::Bit], num_vars)
        .unwrap()
        .build();

    let chip = TraceBuilder::new(&BndChipletCols::build_layout(), num_vars)
        .unwrap()
        .build();

    let program = BndHostProgram { pinned };
    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(main).with_chiplets(vec![chip]);
    let config = Config::default();

    let bytes = serialize_bundle(&program, &instance, &witness, &config).unwrap();
    let restored: DeserializedBundle<F> = deserialize_bundle(&bytes).unwrap();

    assert_bundle_eq(&program, &restored, &witness, "chiplet_constant_bnd");

    let bcs = Air::<F>::boundary_constraints(&restored.chiplet_defs[0]);
    assert_eq!(bcs.len(), 1, "single boundary preserved");

    match &bcs[0].target {
        BoundaryTarget::Constant(v) => *v,
        other => panic!("expected Constant target, got {other:?}"),
    }
}

#[test]
fn chiplet_boundary_constant_zero_round_trips() {
    let restored = chiplet_bnd_roundtrip(F::ZERO);
    assert_eq!(restored, F::ZERO);
}

#[test]
fn chiplet_boundary_constant_one_round_trips() {
    let restored = chiplet_bnd_roundtrip(F::ONE);
    assert_eq!(restored, F::ONE);
}

#[test]
fn chiplet_boundary_constant_nontrivial_round_trips() {
    let mut bytes = [0u8; 16];
    bytes[0] = 0xDE;
    bytes[1] = 0xAD;
    bytes[2] = 0xBE;
    bytes[3] = 0xEF;
    bytes[7] = 0x01;
    bytes[15] = 0xA5;

    let val = F::deserialize(&bytes).unwrap();

    let restored = chiplet_bnd_roundtrip(val);
    assert_eq!(
        restored, val,
        "non-trivial 128-bit constant must survive byte-order"
    );
}

// =================================================================
// Paired-spec round-trip
// =================================================================

define_columns! {
    PairedRtCols {
        KEY: B32,
        S_SEND: Bit,
        S_RECV: Bit,
    }
}

#[derive(Clone)]
struct PairedRtProgram {
    kind: BusKind,
}

impl Air<F> for PairedRtProgram {
    fn num_columns(&self) -> usize {
        PairedRtCols::NUM_COLUMNS
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(PairedRtCols::build_layout)
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        let sources = vec![
            (Source::Column(PairedRtCols::KEY), b"k_a" as &[u8]),
            (Source::RowIndexLeBytes(4), b"k_clk" as &[u8]),
        ];

        vec![(
            "rt_paired_bus".into(),
            PermutationCheckSpec::new_paired(
                sources,
                PairedRtCols::S_SEND,
                PairedRtCols::S_RECV,
                self.kind,
            ),
        )]
    }

    fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
        vec![
            FixedColumn::sparse(PairedRtCols::S_SEND, vec![(0, F::ONE)]),
            FixedColumn::sparse(PairedRtCols::S_RECV, vec![(1, F::ONE)]),
        ]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
    }
}

impl Program<F> for PairedRtProgram {}

fn paired_spec_roundtrip(kind: BusKind) {
    let num_vars = 3;
    let num_rows = 1 << num_vars;

    let program = PairedRtProgram { kind };
    let trace = TraceBuilder::new(&PairedRtCols::build_layout(), num_vars)
        .unwrap()
        .build();

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(trace);
    let config = Config::default();

    let bytes = serialize_bundle(&program, &instance, &witness, &config).unwrap();
    let restored: DeserializedBundle<F> = deserialize_bundle(&bytes).unwrap();

    assert_bundle_eq(&program, &restored, &witness, "paired_spec");

    let (_, spec) = restored
        .permutation_checks
        .iter()
        .find(|(id, _)| id == "rt_paired_bus")
        .expect("paired bus must round-trip");

    assert!(spec.has_paired(), "recv_selector must round-trip set");
    assert_eq!(spec.selector, Some(PairedRtCols::S_SEND));
    assert_eq!(spec.recv_selector, Some(PairedRtCols::S_RECV));
    assert_eq!(spec.kind, kind);
}

#[test]
fn paired_permutation_spec_round_trips() {
    paired_spec_roundtrip(BusKind::Permutation);
}

#[test]
fn paired_lookup_spec_round_trips() {
    paired_spec_roundtrip(BusKind::Lookup);
}

// =================================================================
// PhaseColumn round-trip
// =================================================================

define_columns! {
    PhasedRtCols {
        KEY: B32,
        PHASE: B32,
        SEL: Bit,
    }
}

#[derive(Clone)]
struct PhasedRtProgram;

impl Air<F> for PhasedRtProgram {
    fn num_columns(&self) -> usize {
        PhasedRtCols::NUM_COLUMNS
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(PhasedRtCols::build_layout)
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        let service = Service {
            bus_id: "rt_phased_bus",
            kind: BusKind::Permutation,
            slots: vec![
                ServiceSlot::Value(b"k_key"),
                ServiceSlot::RequestIdxBytes { num_bytes: 4 },
            ],
            clock_waiver: None,
        };

        let spec = service
            .request_phased(&[PhasedRtCols::KEY], PhasedRtCols::PHASE, PhasedRtCols::SEL)
            .unwrap();

        vec![("rt_phased_bus".into(), spec)]
    }

    fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
        vec![
            FixedColumn::periodic(PhasedRtCols::PHASE, 2, vec![F::ZERO, F::ONE]),
            FixedColumn::sparse(PhasedRtCols::SEL, vec![(0, F::ONE)]),
        ]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
    }
}

impl Program<F> for PhasedRtProgram {}

#[test]
fn phased_clock_spec_round_trips() {
    let num_vars = 3;
    let num_rows = 1 << num_vars;

    let program = PhasedRtProgram;
    let trace = TraceBuilder::new(&PhasedRtCols::build_layout(), num_vars)
        .unwrap()
        .build();

    let instance = ProgramInstance::new(num_rows, vec![]);
    let witness = ProgramWitness::new(trace);
    let config = Config::default();

    let bytes = serialize_bundle(&program, &instance, &witness, &config).unwrap();
    let restored: DeserializedBundle<F> = deserialize_bundle(&bytes).unwrap();

    assert_bundle_eq(&program, &restored, &witness, "phased_spec");

    let (_, spec) = restored
        .permutation_checks
        .iter()
        .find(|(id, _)| id == "rt_phased_bus")
        .expect("phased bus must round-trip");

    let sources: Vec<Source> = spec.sources.iter().map(|(s, _)| s.clone()).collect();

    assert_eq!(
        sources,
        vec![
            Source::Column(PhasedRtCols::KEY),
            Source::RowIndexByte(0),
            Source::RowIndexByte(1),
            Source::RowIndexByte(2),
            Source::PhaseColumn(PhasedRtCols::PHASE),
        ],
    );

    let origin = program.permutation_checks();
    let labels: Vec<&[u8]> = spec.sources.iter().map(|(_, l)| *l).collect();

    assert_eq!(
        labels,
        origin[0]
            .1
            .sources
            .iter()
            .map(|(_, l)| *l)
            .collect::<Vec<&[u8]>>(),
    );
}

// =================================================================
// Proof wire round-trip (tensor_vec / master_evals)
// =================================================================

fn empty_sumcheck() -> SumcheckProof<F> {
    SumcheckProof {
        round_polys: vec![],
        claimed_evaluation: F::ZERO,
    }
}

fn eval_proof_with(
    tensor_vec: Vec<F>,
    master_evals: Option<MasterEvals<F>>,
    h_ldt_proof: Option<BrakedownProof<F>>,
) -> EvalBatchProof<F> {
    EvalBatchProof::new(
        empty_sumcheck(),
        BrakedownProof::new(vec![], vec![]),
        (vec![wide(1)], vec![wide(2), wide(3)]),
        tensor_vec,
        master_evals,
        h_ldt_proof,
    )
}

fn dummy_commitment() -> BrakedownCommitment {
    BrakedownCommitment {
        root: [7u8; 32],
        num_rows: 1 << 10,
        num_cols: 8,
    }
}

#[test]
fn proof_master_evals_round_trip() {
    let evals = MasterEvals {
        whole: wide(10),
        ring: wide(11),
    };

    let main_eval = eval_proof_with(vec![wide(1), wide(2)], Some(evals), None);
    let chiplet_eval = eval_proof_with(vec![wide(5)], None, None);

    let proof = InnerProof::new(
        dummy_commitment(),
        empty_sumcheck(),
        LogUpAux::new(vec![], vec![]),
        main_eval,
        vec![dummy_commitment()],
        vec![empty_sumcheck()],
        vec![LogUpAux::new(vec![], vec![])],
        vec![chiplet_eval],
        None,
        None,
    );

    let bytes = serialize_proof_bytes(&proof);
    let restored: InnerProof<F> = deserialize_proof(&bytes).unwrap();

    let bytes_of = |v: &[F]| v.iter().map(|f| f.to_bytes()).collect::<Vec<_>>();

    assert_eq!(
        bytes_of(&restored.eval_proof.tensor_vec),
        bytes_of(&[wide(1), wide(2)]),
        "main tensor_vec must survive the wire",
    );
    assert_eq!(
        restored.eval_proof.master_evals,
        Some(evals),
        "main master_evals must survive the wire",
    );

    let chip = &restored.chiplet_eval_proofs[0];

    assert_eq!(
        bytes_of(&chip.tensor_vec),
        bytes_of(&[wide(5)]),
        "chiplet tensor_vec must survive the wire",
    );
    assert!(
        chip.master_evals.is_none(),
        "absent master_evals must stay absent",
    );
}

#[test]
fn proof_logup_h_binding_round_trips() {
    let main_aux = LogUpAux {
        h_evals: vec![("bus".to_string(), wide(42))],
        claimed_sums: vec![("bus".to_string(), F::ZERO)],
        h_commitment: Some(dummy_commitment()),
    };

    let h_opening = BrakedownProof::new(vec![vec![1u8, 2, 3], vec![4, 5, 6]], vec![[9u8; 32]]);

    let proof = InnerProof::new(
        dummy_commitment(),
        empty_sumcheck(),
        main_aux,
        eval_proof_with(vec![wide(1)], None, Some(h_opening)),
        vec![],
        vec![],
        vec![],
        vec![],
        None,
        None,
    );

    let bytes = serialize_proof_bytes(&proof);
    let restored: InnerProof<F> = deserialize_proof(&bytes).unwrap();

    let comm = restored
        .main_logup_aux
        .h_commitment
        .as_ref()
        .expect("h_commitment must survive the wire");

    assert_eq!(comm.root, [7u8; 32]);
    assert_eq!(comm.num_cols, 8);

    let opening = restored
        .eval_proof
        .h_ldt_proof
        .as_ref()
        .expect("h opening must survive the wire");

    assert_eq!(opening.opened_columns, vec![vec![1u8, 2, 3], vec![4, 5, 6]]);
    assert_eq!(opening.batch_path, vec![[9u8; 32]]);
}

#[test]
fn proof_absent_h_binding_stays_none() {
    let proof = InnerProof::new(
        dummy_commitment(),
        empty_sumcheck(),
        LogUpAux::new(vec![], vec![]),
        eval_proof_with(vec![wide(1)], None, None),
        vec![],
        vec![],
        vec![],
        vec![],
        None,
        None,
    );

    let bytes = serialize_proof_bytes(&proof);
    let restored: InnerProof<F> = deserialize_proof(&bytes).unwrap();

    assert!(restored.main_logup_aux.h_commitment.is_none());
    assert!(restored.eval_proof.h_ldt_proof.is_none());
    assert!(restored.pad_root.is_none());
    assert!(restored.outer.is_none());
}

#[test]
fn proof_outer_segment_round_trips() {
    let opening = |seed: u128| OuterOpening {
        columns: vec![3, 17, 40],
        values: (0..6).map(|i| wide(seed + i)).collect(),
        siblings: vec![[9u8; 32], [10u8; 32]],
    };

    let outer = OuterProof {
        aux_root: [4u8; 32],
        interleaved: vec![wide(100), wide(101)],
        linear: vec![wide(200)],
        quadratic: vec![wide(300), wide(301), wide(302)],
        pad_opening: opening(1_000),
        aux_opening: opening(2_000),
    };

    let proof = InnerProof::new(
        dummy_commitment(),
        empty_sumcheck(),
        LogUpAux::new(vec![], vec![]),
        eval_proof_with(vec![wide(1)], None, None),
        vec![],
        vec![],
        vec![],
        vec![],
        Some([5u8; 32]),
        Some(outer),
    );

    let bytes = serialize_proof_bytes(&proof);
    let restored: InnerProof<F> = deserialize_proof(&bytes).unwrap();

    assert_eq!(restored.pad_root, Some([5u8; 32]));

    let outer = restored.outer.expect("outer segment must survive the wire");
    let bytes_of = |v: &[F]| v.iter().map(|f| f.to_bytes()).collect::<Vec<_>>();

    assert_eq!(outer.aux_root, [4u8; 32]);
    assert_eq!(
        bytes_of(&outer.interleaved),
        bytes_of(&[wide(100), wide(101)])
    );
    assert_eq!(bytes_of(&outer.linear), bytes_of(&[wide(200)]));
    assert_eq!(
        bytes_of(&outer.quadratic),
        bytes_of(&[wide(300), wide(301), wide(302)])
    );

    for (opening, seed) in [(&outer.pad_opening, 1_000), (&outer.aux_opening, 2_000)] {
        assert_eq!(opening.columns, vec![3, 17, 40]);
        assert_eq!(
            bytes_of(&opening.values),
            bytes_of(&(0..6).map(|i| wide(seed + i)).collect::<Vec<_>>())
        );
        assert_eq!(opening.siblings, vec![[9u8; 32], [10u8; 32]]);
    }
}

/// Every `program_id` input must survive the wire, or
/// the prover and the verifier bind different transcripts.
#[test]
fn program_id_survives_the_wire() {
    let num_rows = 1 << 10;

    let mut cx = Circuit::<F>::new("DistinctlyNamedProgram", num_rows).unwrap();

    let cpu = cx.schema(&CpuMemColumns::build_layout());

    cx.bus(RamChiplet::BUS_ID, RamChiplet::cpu_linking_spec());

    cx.fix(
        cpu.at(CpuMemColumns::SELECTOR),
        FixedShape::Cadence {
            stride: 1,
            count: 4,
            origin: 0,
            values: vec![F::ONE],
        },
    );

    let cs = cx.cs();

    cs.assert_boolean(cs.col(CpuMemColumns::IS_WRITE));

    cx.attach(ChipletDef::from_air(&RamChiplet::new(num_rows, 4)).unwrap());
    cx.publish(cpu.at(CpuMemColumns::VAL_B0), num_rows - 1);

    let program = cx.compile().unwrap();

    let (trace, ram_trace) = ram_traces(num_rows);

    let instance = ProgramInstance::new(num_rows, vec![F::ZERO]);
    let witness = ProgramWitness::new(trace).with_chiplets(vec![ram_trace]);

    let bytes = serialize_bundle(&program, &instance, &witness, &Config::default()).unwrap();
    let restored: DeserializedBundle<F> = deserialize_bundle(&bytes).unwrap();
    let rebuilt = BundleProgram::from_bundle(&restored);

    assert_eq!(Air::<F>::name(&rebuilt), "DistinctlyNamedProgram");
    assert_eq!(
        program_id::<F, _>(&program).unwrap(),
        program_id::<F, _>(&rebuilt).unwrap()
    );
}
