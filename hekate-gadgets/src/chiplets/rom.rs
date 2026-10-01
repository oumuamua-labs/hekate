// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! ROM Chiplet:
//! committed program image, linked to the CPU
//! fetch stream via `BusKind::Lookup`.
//!
//! # Architecture
//! The ROM chiplet stores a 1:1 positionally
//! aligned copy of the CPU's fetch stream as a
//! separate, independently committed trace.
//!
//! # Why Separate ROM?
//! 1. **Trace Compression**: ROM does
//!    not need CPU-side state columns.
//! 2. **Multi-Program Support**: Different ROM
//!    traces can be paired with the same CPU AIR.
//! 3. **Commitment Reuse**: ROM can be committed
//!    once and used across executions.
//!
//! # Linking Protocol
//! - CPU side:
//!   emits a fetch event per row.
//! - ROM side:
//!   holds the fetched instruction at
//!   the same row index.
//! - `BusKind::Lookup` forces pointwise equality
//!   of `h` values on the padded hypercube.
//!
//! # Content Binding
//! This chiplet proves "CPU fetched exactly what
//! the ROM trace says at each row". It does **not**
//! bind the ROM trace to a public program image;
//!
//! # Byte Stitching
//! PC is 4 little-endian bytes `[pc_b0..pc_b3]`.
//!
//! Key:
//! `K = Σᵢ κᵢ·pc_bᵢ + κ₄·opcode + Σⱼ κ₅₊ⱼ·argⱼ`.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use hekate_core::errors::Error;
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate_math::{Block32, TowerField};
use hekate_program::constraint::ConstraintAst;
use hekate_program::constraint::builder::ConstraintSystem;
use hekate_program::define_columns;
use hekate_program::permutation::{BusKind, PermutationCheckSpec, Service, ServiceSlot};
use hekate_program::{Air, FixedColumn};

define_columns! {
    pub RomColumns {
        PC_B0: B32,
        PC_B1: B32,
        PC_B2: B32,
        PC_B3: B32,
        OPCODE: B32,
        ARG0: B32,
        ARG1: B32,
        ARG2: B32,
        SELECTOR: Bit,
    }
}

/// ROM chiplet column layout.
///
/// # Column Schema
/// ```text
/// | pc_b0 | pc_b1 | pc_b2 | pc_b3 | opcode | arg0 | arg1 | arg2 | s_rom |
/// |-------|-------|-------|-------|--------|------|------|------|-------|
/// |  0x00 |  0x01 |  0x00 |  0x00 |  0x01  | ...  | ...  | ...  |   1   |
/// |  0x04 |  0x01 |  0x00 |  0x00 |  0x02  | ...  | ...  | ...  |   1   |
/// |  ...  |  ...  |  ...  |  ...  |  ...   | ...  | ...  | ...  |  ...  |
/// |  0x00 |  0x00 |  0x00 |  0x00 |  0x00  | 0x00 | 0x00 | 0x00 |   0   | <- Padding
/// ```
///
/// # Columns (9 total)
/// - `pc_b0..pc_b3`: Program counter bytes (little-endian)
/// - `opcode`: Instruction opcode (1 byte)
/// - `arg0..arg2`: Instruction operands (3 bytes)
/// - `s_rom`: Selector (1 = valid instruction, 0 = padding)
#[derive(Clone, Debug)]
pub struct RomChiplet {
    /// Number of rows (power of 2)
    pub num_rows: usize,
    pub num_instructions: usize,
}

impl RomChiplet {
    /// Canonical bus identifier shared between
    /// CPU-side and chiplet-side specs.
    pub const BUS_ID: &'static str = "rom_link";

    pub fn new(num_rows: usize, num_instructions: usize) -> Self {
        assert!(num_rows.is_power_of_two(), "ROM size must be power of 2");
        assert!(
            num_instructions <= num_rows,
            "instruction count exceeds ROM rows"
        );

        Self {
            num_rows,
            num_instructions,
        }
    }

    /// Both endpoints derive from this schema:
    /// the address bytes, the opcode, the three arguments.
    pub fn service() -> Service {
        Service {
            bus_id: Self::BUS_ID,
            kind: BusKind::Lookup,
            slots: vec![
                ServiceSlot::Value(b"kappa_pc_b0"),
                ServiceSlot::Value(b"kappa_pc_b1"),
                ServiceSlot::Value(b"kappa_pc_b2"),
                ServiceSlot::Value(b"kappa_pc_b3"),
                ServiceSlot::Value(b"kappa_opcode"),
                ServiceSlot::Value(b"kappa_arg0"),
                ServiceSlot::Value(b"kappa_arg1"),
                ServiceSlot::Value(b"kappa_arg2"),
            ],
        }
    }

    /// Returns the `Lookup`-kind permutation
    /// check spec for ROM-CPU linking.
    pub fn linking_spec() -> PermutationCheckSpec {
        Self::service()
            .respond(
                &[
                    RomColumns::PC_B0,
                    RomColumns::PC_B1,
                    RomColumns::PC_B2,
                    RomColumns::PC_B3,
                    RomColumns::OPCODE,
                    RomColumns::ARG0,
                    RomColumns::ARG1,
                    RomColumns::ARG2,
                ],
                RomColumns::SELECTOR,
            )
            .expect("service slots match the responder columns")
    }

    /// Requester endpoint over `CpuFetchColumns`.
    pub fn cpu_linking_spec() -> PermutationCheckSpec {
        Self::service()
            .request(
                &[
                    CpuFetchColumns::PC_B0,
                    CpuFetchColumns::PC_B1,
                    CpuFetchColumns::PC_B2,
                    CpuFetchColumns::PC_B3,
                    CpuFetchColumns::OPCODE,
                    CpuFetchColumns::ARG0,
                    CpuFetchColumns::ARG1,
                    CpuFetchColumns::ARG2,
                ],
                CpuFetchColumns::SELECTOR,
            )
            .expect("service slots match the requester columns")
    }

    /// Returns the number of rows in this ROM.
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Returns the number of columns in ROM traces.
    pub fn num_columns(&self) -> usize {
        RomColumns::NUM_COLUMNS
    }
}

define_columns! {
    pub CpuFetchColumns {
        PC_B0: B32,
        PC_B1: B32,
        PC_B2: B32,
        PC_B3: B32,
        OPCODE: B32,
        ARG0: B32,
        ARG1: B32,
        ARG2: B32,
        SELECTOR: Bit,
    }
}

/// Instruction representation for ROM.
///
/// This is the prover's witness format before trace generation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Instruction {
    /// Program counter (32-bit address)
    pub pc: u32,

    /// Opcode byte
    pub opcode: u8,

    /// Operand bytes (up to 3)
    pub args: [u8; 3],
}

impl Instruction {
    /// Creates a new instruction.
    pub fn new(pc: u32, opcode: u8, args: [u8; 3]) -> Self {
        Self { pc, opcode, args }
    }

    /// Extracts PC bytes (little-endian).
    pub fn pc_bytes(&self) -> [u8; 4] {
        self.pc.to_le_bytes()
    }

    /// Returns the opcode.
    pub fn opcode(&self) -> u8 {
        self.opcode
    }

    /// Returns the arguments.
    pub fn args(&self) -> [u8; 3] {
        self.args
    }
}

/// Converts a list of instructions into a ROM trace.
///
/// Uses `TraceBuilder` for schema-driven construction:
/// columns are indexed by `RomColumns` constants,
/// ordering is guaranteed, padding is automatic.
pub fn generate_rom_trace(
    instructions: &[Instruction],
    num_rows: usize,
) -> hekate_core::errors::Result<ColumnTrace> {
    assert!(
        num_rows.is_power_of_two(),
        "ROM trace size must be power of 2, got {num_rows}"
    );

    if instructions.len() > num_rows {
        return Err(Error::Protocol {
            protocol: "rom",
            message: "too many instructions for trace size",
        });
    }

    let num_vars = num_rows.trailing_zeros() as usize;
    let mut tb = TraceBuilder::new(&RomColumns::build_layout(), num_vars)?;

    for (i, instr) in instructions.iter().enumerate() {
        let pc = instr.pc_bytes();
        tb.set_b32(RomColumns::PC_B0, i, Block32::from(pc[0] as u32))?;
        tb.set_b32(RomColumns::PC_B1, i, Block32::from(pc[1] as u32))?;
        tb.set_b32(RomColumns::PC_B2, i, Block32::from(pc[2] as u32))?;
        tb.set_b32(RomColumns::PC_B3, i, Block32::from(pc[3] as u32))?;
        tb.set_b32(RomColumns::OPCODE, i, Block32::from(instr.opcode as u32))?;
        tb.set_b32(RomColumns::ARG0, i, Block32::from(instr.args[0] as u32))?;
        tb.set_b32(RomColumns::ARG1, i, Block32::from(instr.args[1] as u32))?;
        tb.set_b32(RomColumns::ARG2, i, Block32::from(instr.args[2] as u32))?;
    }

    tb.fill_selector(RomColumns::SELECTOR, instructions.len())?;

    Ok(tb.build())
}

// =========================================================
// AIR TRAIT IMPLEMENTATION
// =========================================================

impl<F: TowerField> Air<F> for RomChiplet {
    fn name(&self) -> String {
        "RomChiplet".to_string()
    }

    fn num_columns(&self) -> usize {
        RomColumns::NUM_COLUMNS
    }

    fn column_layout(&self) -> &[ColumnType] {
        static LAYOUT: once_cell::race::OnceBox<Vec<ColumnType>> = once_cell::race::OnceBox::new();
        LAYOUT.get_or_init(|| Box::new(RomColumns::build_layout()))
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        vec![(Self::BUS_ID.into(), Self::linking_spec())]
    }

    fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
        vec![FixedColumn::prefix(
            RomColumns::SELECTOR,
            self.num_instructions,
        )]
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        ConstraintSystem::<F>::new().build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hekate_core::trace::Trace;
    use hekate_math::Bit;
    use hekate_program::permutation::Source;

    #[test]
    fn rom_chiplet_column_count() {
        let rom = RomChiplet::new(16, 16);
        assert_eq!(rom.num_columns(), 9);
        assert_eq!(rom.num_rows(), 16);
    }

    #[test]
    fn rom_linking_spec_structure() {
        let spec = RomChiplet::linking_spec();

        assert_eq!(spec.num_sources(), 8);

        // Selector present
        assert!(spec.has_selector());
        assert_eq!(spec.selector, Some(RomColumns::SELECTOR));
    }

    #[test]
    fn cpu_fetch_linking_spec_matches_rom() {
        let rom_spec = RomChiplet::linking_spec();
        let cpu_spec = RomChiplet::cpu_linking_spec();

        // Both specs must have same number of sources
        assert_eq!(rom_spec.num_sources(), cpu_spec.num_sources());

        // Challenge labels must match exactly
        for i in 0..rom_spec.num_sources() {
            assert_eq!(rom_spec.sources[i].1, cpu_spec.sources[i].1);
        }
    }

    #[test]
    fn instruction_pc_bytes() {
        let instr = Instruction::new(0x01020304, 0xAB, [0x11, 0x22, 0x33]);

        let pc_bytes = instr.pc_bytes();
        assert_eq!(pc_bytes, [0x04, 0x03, 0x02, 0x01]); // Little-endian
    }

    #[test]
    fn generate_rom_trace_basic() {
        let instructions = vec![
            Instruction::new(0x0100, 0x01, [0x11, 0x12, 0x13]),
            Instruction::new(0x0104, 0x02, [0x21, 0x22, 0x23]),
            Instruction::new(0x0108, 0x03, [0x31, 0x32, 0x33]),
        ];

        // 8 rows -> 2^3 vars
        let trace = generate_rom_trace(&instructions, 8).unwrap();

        assert_eq!(trace.num_cols(), RomColumns::NUM_COLUMNS);
        assert_eq!(trace.num_rows().unwrap(), 8);

        // Helper to safe access typed columns
        let pc_b0 = trace.columns[RomColumns::PC_B0]
            .as_b32_slice()
            .expect("Wrong type");
        let pc_b1 = trace.columns[RomColumns::PC_B1]
            .as_b32_slice()
            .expect("Wrong type");
        let opcode = trace.columns[RomColumns::OPCODE]
            .as_b32_slice()
            .expect("Wrong type");
        let selector = trace.columns[RomColumns::SELECTOR]
            .as_bit_slice()
            .expect("Wrong type");

        // Check first instruction
        assert_eq!(pc_b0[0].to_tower(), Block32::from(0x00u32)); // LSB of 0x0100
        assert_eq!(pc_b1[0].to_tower(), Block32::from(0x01u32)); // Next byte
        assert_eq!(opcode[0].to_tower(), Block32::from(0x01u32));
        assert_eq!(selector[0], Bit::ONE);

        // Check padding row (last row index 7)
        assert_eq!(pc_b0[7].to_tower(), Block32::ZERO);
        assert_eq!(selector[7], Bit::ZERO);
    }

    #[test]
    #[should_panic(expected = "ROM trace size must be power of 2")]
    fn generate_rom_trace_non_power_of_two() {
        let instructions = vec![Instruction::new(0, 0, [0, 0, 0])];
        let _ = generate_rom_trace(&instructions, 7);
    }

    #[test]
    fn generate_rom_trace_overflow() {
        let instructions = vec![
            Instruction::new(0, 0, [0, 0, 0]),
            Instruction::new(1, 1, [1, 1, 1]),
            Instruction::new(2, 2, [2, 2, 2]),
        ];

        // 2 rows < 3 instructions -> Err
        let result = generate_rom_trace(&instructions, 2);
        assert!(result.is_err());
    }

    /// Test:
    /// Byte Stitching Demonstration (No RowIndex for ROM).
    ///
    /// This test explicitly demonstrates that:
    /// 1. ROM linking uses byte-stitched PC (4 bytes)
    /// 2. NO RowIndex is needed (ROM is static, not time-dependent)
    ///    This is a spec check, no proving needed.
    #[test]
    fn byte_stitching_no_row_index() {
        let rom_spec = RomChiplet::linking_spec();
        let cpu_spec = RomChiplet::cpu_linking_spec();

        // Verify that specs have 5 sources:
        // 4 PC bytes + 1 opcode
        assert_eq!(
            rom_spec.num_sources(),
            8,
            "ROM spec should have 8 sources (PC, opcode, and 3 args)"
        );
        assert_eq!(
            cpu_spec.num_sources(),
            8,
            "CPU spec should have 8 sources (PC, opcode, and 3 args)"
        );

        // Verify that NO RowIndexLeBytes source is used
        for (source, _label) in &rom_spec.sources {
            if let Source::RowIndexLeBytes(_) = source {
                panic!("ROM should NOT use RowIndexLeBytes - it's static code!");
            }
        }

        for (source, _label) in &cpu_spec.sources {
            if let Source::RowIndexLeBytes(_) = source {
                panic!("CPU fetch should NOT use RowIndexLeBytes for ROM linking!");
            }
        }

        // Verify challenge labels match between ROM and CPU
        for i in 0..rom_spec.num_sources() {
            assert_eq!(
                rom_spec.sources[i].1, cpu_spec.sources[i].1,
                "Challenge labels must match at index {}",
                i
            );
        }
    }
}
