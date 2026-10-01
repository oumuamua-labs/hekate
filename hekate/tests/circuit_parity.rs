// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Circuit-vs-hand artifact parity.
//!
//! Each hand program below freezes the pre-`Circuit` authoring
//! of a shipped host. The circuit build must reproduce its
//! artifacts bit-for-bit: equal `program_id`, equal kernel hints.

use hekate_core::trace::ColumnType;
use hekate_gadgets::IntArithmeticChiplet;
use hekate_gadgets::atoms::int_arith;
use hekate_keccak::{CpuKeccakColumns, KECCAK_LANE_LABELS, KeccakChiplet, KeccakColumns};
use hekate_math::{Block128, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::circuit::{Circuit, CircuitProgram, Col};
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::digest::program_id;
use hekate_program::expander::VirtualExpander;
use hekate_program::permutation::{EMIT_RANK_LABEL, PermutationCheckSpec, Side, Source};
use hekate_program::{Air, FixedColumn, FixedShape, InlineKernelHint, Program, fix};

type F = Block128;

const KECCAK_OFFSET: usize = CpuKeccakColumns::NUM_COLUMNS;
const NUM_ROWS: usize = 256;

// =================================================================
// Hand authoring:
// keccak_inline host without the Circuit layer.
// =================================================================

#[derive(Clone)]
struct HandKeccakInline {
    num_rows: usize,
}

impl HandKeccakInline {
    fn cpu_spec() -> PermutationCheckSpec {
        let mut sources = Vec::with_capacity(26);

        for (i, label) in KECCAK_LANE_LABELS.iter().enumerate() {
            sources.push((Source::Column(CpuKeccakColumns::LANES + i), *label));
        }

        sources.push((Source::EmitRank(Side::Request), EMIT_RANK_LABEL));

        PermutationCheckSpec::new(sources, Some(CpuKeccakColumns::SELECTOR))
    }

    fn cpu_pins(num_blocks: usize) -> Vec<FixedColumn<F>> {
        vec![fix(
            CpuKeccakColumns::SELECTOR,
            FixedShape::Cadence {
                stride: 25,
                count: num_blocks,
                origin: 0,
                values: (0..25)
                    .map(|off| {
                        if off == 0 || off == 24 {
                            F::ONE
                        } else {
                            F::ZERO
                        }
                    })
                    .collect(),
            },
        )]
    }
}

impl Air<F> for HandKeccakInline {
    fn num_columns(&self) -> usize {
        CpuKeccakColumns::NUM_COLUMNS + KeccakColumns::NUM_COLUMNS
    }

    fn boundary_constraints(&self) -> Vec<hekate_program::constraint::BoundaryConstraint<F>> {
        let last_output_row = 25 * (self.num_rows / 25) - 1;

        (0..4)
            .map(|i| {
                hekate_program::constraint::BoundaryConstraint::with_public_input(
                    CpuKeccakColumns::LANES + i,
                    last_output_row,
                    i,
                )
            })
            .collect()
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(|| {
            let mut cols = CpuKeccakColumns::build_layout();
            cols.extend_from_slice(KeccakChiplet::physical_layout());

            cols
        })
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        let mut keccak_spec = KeccakChiplet::linking_spec();
        keccak_spec.shift_column_indices(KECCAK_OFFSET);

        vec![
            (KeccakChiplet::BUS_ID.into(), Self::cpu_spec()),
            (KeccakChiplet::BUS_ID.into(), keccak_spec),
        ]
    }

    fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
        let num_blocks = self.num_rows / 25;
        let mut pins = Self::cpu_pins(num_blocks);

        let chiplet = KeccakChiplet::new(self.num_rows, num_blocks);
        pins.extend(Air::<F>::fixed_columns(&chiplet).into_iter().map(|mut p| {
            p.col_idx += KECCAK_OFFSET;
            p
        }));

        pins
    }

    fn virtual_expander(&self) -> Option<&VirtualExpander> {
        static E: std::sync::OnceLock<VirtualExpander> = std::sync::OnceLock::new();
        Some(E.get_or_init(|| {
            let cpu = VirtualExpander::new()
                .pass_through(25, ColumnType::B64)
                .control_bits(1);

            KeccakChiplet::expand_into(cpu, KECCAK_OFFSET)
                .build()
                .expect("keccak inline expander")
        }))
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let mut ast = ConstraintSystem::<F>::new().build();
        let mut keccak_ast = KeccakChiplet::new(self.num_rows, self.num_rows / 25).constraint_ast();

        keccak_ast.arena.shift_cells(KECCAK_OFFSET);
        ast.merge(keccak_ast);

        ast
    }

    fn inline_chiplets(&self) -> hekate_core::errors::Result<Vec<ChipletDef<F>>> {
        Ok(vec![ChipletDef::from_air(&KeccakChiplet::new(
            self.num_rows,
            self.num_rows / 25,
        ))?])
    }

    fn inline_chiplet_kernels(&self) -> Vec<InlineKernelHint> {
        vec![InlineKernelHint {
            chiplet_idx: 0,
            root_offset: 0,
            column_offset: KECCAK_OFFSET,
        }]
    }
}

impl Program<F> for HandKeccakInline {
    fn num_public_inputs(&self) -> usize {
        4
    }
}

fn circuit_keccak_inline(num_rows: usize) -> CircuitProgram<F> {
    let num_blocks = num_rows / KeccakChiplet::BLOCK_ROWS;

    let mut cx = Circuit::<F>::new("HekateAir", num_rows).unwrap();

    let cpu = cx.schema(&CpuKeccakColumns::build_layout());

    let selector = cpu.at(CpuKeccakColumns::SELECTOR);

    let call_values: Vec<Col> = (0..25)
        .map(|lane| cpu.at(CpuKeccakColumns::LANES + lane))
        .collect();

    cx.call(&KeccakChiplet::service(), &call_values, selector)
        .unwrap();

    cx.fix(
        selector,
        KeccakChiplet::host_selector_shape(KeccakChiplet::BLOCK_ROWS, num_blocks),
    );

    cx.mount(ChipletDef::from_air(&KeccakChiplet::new(num_rows, num_blocks)).unwrap());

    let last_output_row = KeccakChiplet::BLOCK_ROWS * num_blocks - 1;
    for i in 0..4 {
        cx.publish(cpu.at(CpuKeccakColumns::LANES + i), last_output_row);
    }

    cx.compile().unwrap()
}

#[test]
fn keccak_inline_circuit_matches_hand_authoring() {
    let hand = HandKeccakInline { num_rows: NUM_ROWS };
    let circuit = circuit_keccak_inline(NUM_ROWS);

    assert_eq!(
        circuit.inline_chiplet_kernels().len(),
        hand.inline_chiplet_kernels().len()
    );

    for (c, h) in circuit
        .inline_chiplet_kernels()
        .iter()
        .zip(hand.inline_chiplet_kernels())
    {
        assert_eq!(c.chiplet_idx, h.chiplet_idx);
        assert_eq!(c.root_offset, h.root_offset);
        assert_eq!(c.column_offset, h.column_offset);
    }

    assert_eq!(
        circuit.constraint_ast().to_constraints(),
        hand.constraint_ast().to_constraints()
    );

    assert_eq!(
        program_id::<F, _>(&circuit).unwrap(),
        program_id::<F, _>(&hand).unwrap()
    );
}

// =================================================================
// Frozen hand authoring:
// fibonacci host as shipped before the Circuit layer
// =================================================================

#[derive(Clone)]
struct HandFib {
    num_rows: usize,
    chiplet: IntArithmeticChiplet,
    layout: Vec<ColumnType>,
    expander: VirtualExpander,
}

impl HandFib {
    fn new(num_rows: usize) -> Self {
        let chiplet = IntArithmeticChiplet::new(32, num_rows, num_rows - 1).unwrap();

        let layout = <IntArithmeticChiplet as Air<F>>::column_layout(&chiplet).to_vec();

        let expander = VirtualExpander::new()
            .append(<IntArithmeticChiplet as Air<F>>::virtual_expander(&chiplet).unwrap())
            .build()
            .expect("HandFib expander");

        Self {
            num_rows,
            chiplet,
            layout,
            expander,
        }
    }
}

impl Air<F> for HandFib {
    fn boundary_constraints(&self) -> Vec<hekate_program::constraint::BoundaryConstraint<F>> {
        vec![
            hekate_program::constraint::BoundaryConstraint::with_constant(
                self.chiplet.layout().val_a,
                0,
                F::ZERO,
            ),
            hekate_program::constraint::BoundaryConstraint::with_constant(
                self.chiplet.layout().val_b,
                0,
                F::ONE,
            ),
            hekate_program::constraint::BoundaryConstraint::with_public_input(
                self.chiplet.layout().val_b,
                self.num_rows - 1,
                0,
            ),
        ]
    }

    fn column_layout(&self) -> &[ColumnType] {
        &self.layout
    }

    fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
        vec![
            FixedColumn::prefix(self.chiplet.layout().s_output, self.num_rows - 1),
            FixedColumn::last_row(self.chiplet.layout().s_add),
        ]
    }

    fn virtual_expander(&self) -> Option<&VirtualExpander> {
        Some(&self.expander)
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let mut ast = <IntArithmeticChiplet as Air<F>>::constraint_ast(&self.chiplet);

        let layout = self.chiplet.layout();
        let cs = ConstraintSystem::<F>::new();

        let s_add = cs.col(layout.s_add);
        let val_b = cs.col(layout.val_b);
        let val_res = cs.col(layout.val_res);
        let next_val_a = cs.next(layout.val_a);
        let next_val_b = cs.next(layout.val_b);

        cs.constrain(s_add * (next_val_a + val_b));
        cs.constrain(s_add * (next_val_b + val_res));

        cs.assert_zero_when(cs.one() + s_add, val_res);

        ast.merge(cs.build());

        ast
    }

    fn inline_chiplets(&self) -> hekate_core::errors::Result<Vec<ChipletDef<F>>> {
        Ok(vec![ChipletDef::from_air(&self.chiplet)?])
    }

    fn inline_chiplet_kernels(&self) -> Vec<InlineKernelHint> {
        vec![InlineKernelHint {
            chiplet_idx: 0,
            root_offset: 0,
            column_offset: 0,
        }]
    }
}

impl Program<F> for HandFib {
    fn num_public_inputs(&self) -> usize {
        1
    }
}

fn circuit_fib(num_rows: usize) -> CircuitProgram<F> {
    let chiplet = IntArithmeticChiplet::new(32, num_rows, num_rows - 1).unwrap();
    let layout = chiplet.layout().clone();

    let mut cx = Circuit::<F>::new("HekateAir", num_rows).unwrap();
    let arith = cx.mount_unlinked(ChipletDef::from_air(&chiplet).unwrap());

    let cs = cx.cs();

    let s_add = cs.col(layout.s_add);
    let val_b = cs.col(layout.val_b);
    let val_res = cs.col(layout.val_res);
    let next_val_a = cs.next(layout.val_a);
    let next_val_b = cs.next(layout.val_b);

    cs.constrain(s_add * (next_val_a + val_b));
    cs.constrain(s_add * (next_val_b + val_res));

    cs.assert_zero_when(cs.one() + s_add, val_res);

    cx.boundary(arith.col(layout.val_a), 0, F::ZERO);
    cx.boundary(arith.col(layout.val_b), 0, F::ONE);
    cx.fix(arith.col(layout.s_add), FixedShape::LastRow);
    cx.publish(arith.col(layout.val_b), num_rows - 1);

    cx.compile().unwrap()
}

#[test]
fn fibonacci_circuit_matches_hand_authoring() {
    let hand = HandFib::new(NUM_ROWS);
    let circuit = circuit_fib(NUM_ROWS);

    assert_eq!(
        circuit.constraint_ast().to_constraints(),
        hand.constraint_ast().to_constraints()
    );

    assert_eq!(
        program_id::<F, _>(&circuit).unwrap(),
        program_id::<F, _>(&hand).unwrap()
    );
}

// =================================================================
// Frozen hand authoring:
// fibonacci_raw host as shipped before the Circuit layer.
// =================================================================

hekate_program::define_columns! {
    FibIntPhys {
        A: B32,
        B: B32,
        SUM: B32,
        CARRY: B32,
        Q: Bit,
    }
}

hekate_program::define_columns! {
    FibIntVirt {
        A_BITS: [Bit; 32],
        B_BITS: [Bit; 32],
        SUM_BITS: [Bit; 32],
        CARRY_BITS: [Bit; 32],
        A_PACKED: B32,
        B_PACKED: B32,
        SUM_PACKED: B32,
        CARRY_PACKED: B32,
        Q: Bit,
    }
}

#[derive(Clone)]
struct HandFibRaw {
    num_rows: usize,
}

impl Air<F> for HandFibRaw {
    fn boundary_constraints(&self) -> Vec<hekate_program::constraint::BoundaryConstraint<F>> {
        vec![
            hekate_program::constraint::BoundaryConstraint::with_constant(
                FibIntVirt::A_PACKED,
                0,
                F::ZERO,
            ),
            hekate_program::constraint::BoundaryConstraint::with_constant(
                FibIntVirt::B_PACKED,
                0,
                F::ONE,
            ),
            hekate_program::constraint::BoundaryConstraint::with_public_input(
                FibIntVirt::B_PACKED,
                self.num_rows - 1,
                0,
            ),
        ]
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
        LAYOUT.get_or_init(FibIntPhys::build_layout)
    }

    fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
        vec![FixedColumn::last_row(FibIntVirt::Q)]
    }

    fn virtual_expander(&self) -> Option<&VirtualExpander> {
        static E: std::sync::OnceLock<VirtualExpander> = std::sync::OnceLock::new();
        Some(E.get_or_init(|| {
            VirtualExpander::new()
                .expand_bits(4, ColumnType::B32)
                .reuse_pass_through(0, 4)
                .control_bits(1)
                .build()
                .expect("HandFibRaw expander")
        }))
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        let cs = ConstraintSystem::<F>::new();

        let a_bits: Vec<_> = (0..32).map(|i| cs.col(FibIntVirt::A_BITS + i)).collect();
        let b_bits: Vec<_> = (0..32).map(|i| cs.col(FibIntVirt::B_BITS + i)).collect();
        let sum_bits: Vec<_> = (0..32).map(|i| cs.col(FibIntVirt::SUM_BITS + i)).collect();
        let carry_v: Vec<_> = (0..32)
            .map(|i| cs.col(FibIntVirt::CARRY_BITS + i))
            .collect();

        let zero = cs.constant(F::ZERO);

        let mut carry = Vec::with_capacity(33);
        carry.push(zero);
        carry.extend(carry_v.iter().copied());

        int_arith::add_carry_chain_with_carry_in(&cs, &a_bits, &b_bits, &sum_bits, &carry);

        let q = cs.col(FibIntVirt::Q);
        let b_packed = cs.col(FibIntVirt::B_PACKED);
        let sum_packed = cs.col(FibIntVirt::SUM_PACKED);
        let next_a = cs.next(FibIntVirt::A_PACKED);
        let next_b = cs.next(FibIntVirt::B_PACKED);

        cs.constrain(q * (next_a + b_packed));
        cs.constrain(q * (next_b + sum_packed));

        cs.build()
    }
}

impl Program<F> for HandFibRaw {
    fn num_public_inputs(&self) -> usize {
        1
    }
}

fn circuit_fib_raw(num_rows: usize) -> CircuitProgram<F> {
    let mut cx = Circuit::<F>::new("HekateAir", num_rows).unwrap();

    let words = cx.expand_bits(4, ColumnType::B32);
    let packed = cx.reuse_pass_through(&words);

    let q = cx.column(ColumnType::Bit);

    let [a_packed, b_packed_col, sum_packed_col] = [packed.at(0), packed.at(1), packed.at(2)];

    let cs = cx.cs();

    let a_bits: Vec<_> = words.bits(0).iter().map(|c| cs.col(c.index())).collect();
    let b_bits: Vec<_> = words.bits(1).iter().map(|c| cs.col(c.index())).collect();
    let sum_bits: Vec<_> = words.bits(2).iter().map(|c| cs.col(c.index())).collect();
    let carry_v: Vec<_> = words.bits(3).iter().map(|c| cs.col(c.index())).collect();

    let zero = cs.constant(F::ZERO);

    let mut carry = Vec::with_capacity(33);
    carry.push(zero);
    carry.extend(carry_v.iter().copied());

    int_arith::add_carry_chain_with_carry_in(cs, &a_bits, &b_bits, &sum_bits, &carry);

    let q_cell = cs.col(q.index());
    let b_packed = cs.col(b_packed_col.index());
    let sum_packed = cs.col(sum_packed_col.index());
    let next_a = cs.next(a_packed.index());
    let next_b = cs.next(b_packed_col.index());

    cs.constrain(q_cell * (next_a + b_packed));
    cs.constrain(q_cell * (next_b + sum_packed));

    cx.fix(q, FixedShape::LastRow);

    cx.boundary(a_packed, 0, F::ZERO);
    cx.boundary(b_packed_col, 0, F::ONE);

    cx.publish(b_packed_col, num_rows - 1);

    cx.compile().unwrap()
}

#[test]
fn fibonacci_raw_circuit_matches_hand_authoring() {
    let hand = HandFibRaw { num_rows: NUM_ROWS };
    let circuit = circuit_fib_raw(NUM_ROWS);

    assert_eq!(
        circuit.constraint_ast().to_constraints(),
        hand.constraint_ast().to_constraints()
    );

    assert_eq!(
        program_id::<F, _>(&circuit).unwrap(),
        program_id::<F, _>(&hand).unwrap()
    );
}
