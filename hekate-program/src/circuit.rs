// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Single-pass program authoring.
//!
//! A `Circuit` records declarations in order and compiles
//! them into a `CircuitProgram`. Offsets, bus endpoints,
//! kernel hints, and the expander are engine-derived;
//! public inputs exist only through `publish`.

use crate::chiplet::ChipletDef;
use crate::constraint::builder::ConstraintSystem;
use crate::constraint::{BoundaryConstraint, BoundaryTarget, ConstraintAst};
use crate::expander::VirtualExpander;
use crate::permutation::{
    PermutationCheckSpec, Service, Source, validate_bus_set, validate_fixed_selectors,
};
use crate::{Air, FixedColumn, FixedShape, InlineKernelHint, Program, validate_fixed_columns};
use alloc::string::String;
use alloc::vec::Vec;
use hekate_core::errors;
use hekate_core::trace::ColumnType;
use hekate_math::{HardwareField, TowerField};

/// Virtual column handle issued by `Circuit`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Col(usize);

impl Col {
    #[inline(always)]
    pub fn index(self) -> usize {
        self.0
    }
}

/// Contiguous run of virtual columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColRange {
    start: usize,
    count: usize,
}

impl ColRange {
    pub fn at(self, i: usize) -> Col {
        assert!(i < self.count, "ColRange::at out of range");
        Col(self.start + i)
    }

    #[inline(always)]
    pub fn start(self) -> usize {
        self.start
    }

    #[inline(always)]
    pub fn len(self) -> usize {
        self.count
    }

    #[inline(always)]
    pub fn is_empty(self) -> bool {
        self.count == 0
    }

    pub fn iter(self) -> impl Iterator<Item = Col> {
        (self.start..self.start + self.count).map(Col)
    }
}

/// Handle to a bit-expanded physical column group.
#[derive(Clone, Copy, Debug)]
pub struct Packed {
    phys_start: usize,
    count: usize,
    bits_per: usize,
    bits: ColRange,
}

impl Packed {
    /// Bit columns of the group's `i`-th physical column.
    ///
    /// # Panics
    /// `i >= count`.
    pub fn bits(&self, i: usize) -> ColRange {
        assert!(i < self.count, "Packed::bits out of range");
        ColRange {
            start: self.bits.start + i * self.bits_per,
            count: self.bits_per,
        }
    }
}

/// Contiguous run of committed columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhysRange {
    start: usize,
    count: usize,
}

impl PhysRange {
    /// Columns `[start, start + count)` of this run.
    pub fn slice(self, start: usize, count: usize) -> PhysRange {
        assert!(
            start
                .checked_add(count)
                .is_some_and(|end| end <= self.count),
            "PhysRange::slice out of range"
        );

        PhysRange {
            start: self.start + start,
            count,
        }
    }

    #[inline(always)]
    pub fn len(self) -> usize {
        self.count
    }

    #[inline(always)]
    pub fn is_empty(self) -> bool {
        self.count == 0
    }
}

impl From<&Packed> for PhysRange {
    fn from(packed: &Packed) -> Self {
        PhysRange {
            start: packed.phys_start,
            count: packed.count,
        }
    }
}

/// Handle to an inline-mounted chiplet.
#[derive(Clone, Copy, Debug)]
pub struct Mounted {
    offset: usize,
    num_columns: usize,
    physical: PhysRange,
}

impl Mounted {
    /// # Panics
    /// `local >= num_columns()`.
    pub fn col(&self, local: usize) -> Col {
        assert!(local < self.num_columns, "Mounted::col out of range");
        Col(self.offset + local)
    }

    #[inline(always)]
    pub fn offset(&self) -> usize {
        self.offset
    }

    #[inline(always)]
    pub fn num_columns(&self) -> usize {
        self.num_columns
    }

    /// The chiplet's committed columns, in its `column_layout()` order.
    /// A reuse view must slice them within one expander entry.
    #[inline(always)]
    pub fn physical(&self) -> PhysRange {
        self.physical
    }
}

/// Recorder for one program: main-table declarations
/// plus mounted and attached chiplets.
pub struct Circuit<F: TowerField> {
    name: String,
    num_rows: usize,
    cs: ConstraintSystem<F>,
    physical: Vec<ColumnType>,
    expander: VirtualExpander,
    pending_run: Option<(ColumnType, usize)>,
    num_virtual: usize,
    fixed: Vec<FixedColumn<F>>,
    boundaries: Vec<BoundaryConstraint<F>>,
    buses: Vec<(String, PermutationCheckSpec)>,
    num_public_inputs: usize,
    inline_chiplets: Vec<ChipletDef<F>>,
    inline_hints: Vec<InlineKernelHint>,
    chiplet_defs: Vec<ChipletDef<F>>,
}

impl<F: TowerField + HardwareField> Circuit<F> {
    /// # Errors
    /// `num_rows` not a power of two.
    pub fn new(name: &str, num_rows: usize) -> errors::Result<Self> {
        if !num_rows.is_power_of_two() {
            return Err(errors::Error::Protocol {
                protocol: "circuit",
                message: "num_rows must be a power of two",
            });
        }

        Ok(Self {
            name: String::from(name),
            num_rows,
            cs: ConstraintSystem::new(),
            physical: Vec::new(),
            expander: VirtualExpander::new(),
            pending_run: None,
            num_virtual: 0,
            fixed: Vec::new(),
            boundaries: Vec::new(),
            buses: Vec::new(),
            num_public_inputs: 0,
            inline_chiplets: Vec::new(),
            inline_hints: Vec::new(),
            chiplet_defs: Vec::new(),
        })
    }

    #[inline(always)]
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Constraint DSL over the recorded columns.
    /// Column handles convert via `Col::index`.
    pub fn cs(&self) -> &ConstraintSystem<F> {
        &self.cs
    }

    pub fn column(&mut self, ty: ColumnType) -> Col {
        self.columns(1, ty).at(0)
    }

    /// Records a `define_columns!` layout 1:1, physical to
    /// virtual. Index the range with the schema's own constants.
    pub fn schema(&mut self, layout: &[ColumnType]) -> ColRange {
        let start = self.num_virtual;

        for &ty in layout {
            self.column(ty);
        }

        ColRange {
            start,
            count: layout.len(),
        }
    }

    pub fn columns(&mut self, count: usize, ty: ColumnType) -> ColRange {
        let start = self.num_virtual;

        self.physical.extend(core::iter::repeat_n(ty, count));
        self.num_virtual += count;

        match self.pending_run.take() {
            Some((run_ty, run_count)) if run_ty == ty => {
                self.pending_run = Some((ty, run_count + count));
            }
            prior => {
                self.flush_run(prior);

                self.pending_run = Some((ty, count));
            }
        }

        ColRange { start, count }
    }

    /// Commits `count` `storage` columns that constraints see
    /// whole, as one expander entry: `columns` merges adjacent
    /// same-type runs and the entries enter `program_id`.
    pub fn pass_through(&mut self, count: usize, storage: ColumnType) -> ColRange {
        self.flush_pending();

        let cols = ColRange {
            start: self.num_virtual,
            count,
        };

        self.physical.extend(core::iter::repeat_n(storage, count));

        self.num_virtual += count;

        let expander = core::mem::take(&mut self.expander);

        self.expander = expander.pass_through(count, storage);

        cols
    }

    /// Commits `count` `storage` columns (`B8` to `B64`) that
    /// constraints see as bits: `bits(i).at(k)` is bit `k` of
    /// column `i`. `reuse_pass_through` also shows them whole.
    pub fn expand_bits(&mut self, count: usize, storage: ColumnType) -> Packed {
        self.flush_pending();

        let phys_start = self.physical.len();
        let bits_per = storage.byte_size() * 8;
        let bits = ColRange {
            start: self.num_virtual,
            count: count * bits_per,
        };

        self.physical.extend(core::iter::repeat_n(storage, count));
        self.num_virtual += bits.count;

        let expander = core::mem::take(&mut self.expander);

        self.expander = expander.expand_bits(count, storage);

        Packed {
            phys_start,
            count,
            bits_per,
            bits,
        }
    }

    /// Lets constraints read committed columns `src` whole
    /// without committing them again: the `Packed` from
    /// `expand_bits` or a slice of `Mounted::physical()`.
    pub fn reuse_pass_through(&mut self, src: impl Into<PhysRange>) -> ColRange {
        let src = src.into();

        self.flush_pending();

        let view = ColRange {
            start: self.num_virtual,
            count: src.count,
        };

        self.num_virtual += src.count;

        let expander = core::mem::take(&mut self.expander);

        self.expander = expander.reuse_pass_through(src.start, src.count);

        view
    }

    /// Lets constraints read committed columns `src` as bits
    /// without committing them again. Bit numbering and the
    /// `B8` to `B64` limit are those of `expand_bits`.
    pub fn reuse_expand_bits(&mut self, src: impl Into<PhysRange>) -> Packed {
        let src = src.into();

        self.flush_pending();

        let bits_per = self
            .physical
            .get(src.start)
            .map_or(0, |storage| storage.byte_size() * 8);

        let bits = ColRange {
            start: self.num_virtual,
            count: src.count * bits_per,
        };

        self.num_virtual += bits.count;

        let expander = core::mem::take(&mut self.expander);

        self.expander = expander.reuse_expand_bits(src.start, src.count);

        Packed {
            phys_start: src.start,
            count: src.count,
            bits_per,
            bits,
        }
    }

    /// Committed columns behind `cols`, a range of
    /// declared columns or of a whole-column view.
    pub fn physical(&self, cols: ColRange) -> errors::Result<PhysRange> {
        let pending = self.pending_run.map_or(0, |(_, count)| count);
        let flushed = self.num_virtual - pending;
        let tail = self.physical.len() - pending;

        let whole = |virt: usize| match virt {
            v if v >= self.num_virtual => None,
            v if v >= flushed => Some(tail + (v - flushed)),
            v => self.expander.whole_column(v),
        };

        let start = whole(cols.start)
            .filter(|&start| (1..cols.count).all(|i| whole(cols.start + i) == Some(start + i)));

        match start {
            Some(start) => Ok(PhysRange {
                start,
                count: cols.count,
            }),
            None => Err(errors::Error::Protocol {
                protocol: "circuit",
                message: "columns are not one run of whole committed columns",
            }),
        }
    }

    /// Pins `col` to `shape`; the verifier evaluates
    /// the shape's MLE for itself at `r_final`.
    pub fn fix(&mut self, col: Col, shape: FixedShape<F>) {
        self.fixed.push(FixedColumn {
            col_idx: col.index(),
            shape,
        });
    }

    /// Constant-target pin;
    /// public inputs enter only through `publish`.
    pub fn boundary(&mut self, col: Col, row: usize, value: F) {
        self.boundaries.push(BoundaryConstraint {
            col_idx: col.index(),
            row_idx: row,
            target: BoundaryTarget::Constant(value),
        });
    }

    /// Requester endpoint of `service` on this table.
    pub fn call(&mut self, service: &Service, values: &[Col], selector: Col) -> errors::Result<()> {
        let cols: Vec<usize> = values.iter().map(|c| c.index()).collect();
        let spec = service.request(&cols, selector.index())?;

        self.buses.push((String::from(service.bus_id), spec));

        Ok(())
    }

    /// Raw bus endpoint for shapes `Service` does
    /// not model (positional, paired, waivered).
    pub fn bus(&mut self, bus_id: &str, spec: PermutationCheckSpec) {
        self.buses.push((String::from(bus_id), spec));
    }

    /// Mounts a chiplet inline: its artifacts and bus endpoints
    /// merge into this table at engine-computed offsets.
    pub fn mount(&mut self, def: ChipletDef<F>) -> Mounted {
        self.mount_inner(def, true)
    }

    /// Mounts a chiplet used as the program's own
    /// trace, whose declared bus has no partner here.
    pub fn mount_unlinked(&mut self, def: ChipletDef<F>) -> Mounted {
        self.mount_inner(def, false)
    }

    /// Attaches a chiplet as an independent table
    /// with its own trace, commitment, and ZeroCheck.
    pub fn attach(&mut self, def: ChipletDef<F>) {
        self.chiplet_defs.push(def);
    }

    /// Pins `col` at `row` to a fresh public input slot,
    /// returned. Whether the AIR determines the pinned
    /// cell is the audited circuit's burden, not checked.
    pub fn publish(&mut self, col: Col, row: usize) -> usize {
        let slot = self.next_slot();

        self.boundaries.push(BoundaryConstraint::with_public_input(
            col.index(),
            row,
            slot,
        ));

        slot
    }

    /// Derives the program artifacts, running every
    /// structural validator on the recording.
    pub fn compile(mut self) -> errors::Result<CircuitProgram<F>> {
        self.flush_pending();

        let expander = self.expander.build()?;
        let width = self.num_virtual;
        let num_vars = self.num_rows.trailing_zeros() as usize;

        if expander.num_virtual_columns() != width {
            return Err(errors::Error::Protocol {
                protocol: "circuit",
                message: "expander virtual width diverged from declared columns",
            });
        }

        let one_to_one = expander.num_virtual_columns() == expander.num_physical_columns();
        let virtual_layout = expander.virtual_layout().to_vec();

        let ast = self.cs.build();

        for (bus_id, spec) in &self.buses {
            spec.validate_clock_stitching(bus_id)?;

            for (source, _) in &spec.sources {
                validate_source_range(source, width)?;
            }
        }

        validate_fixed_selectors(&self.buses, &self.fixed)?;

        for bc in &self.boundaries {
            if bc.col_idx >= width || bc.row_idx >= self.num_rows {
                return Err(errors::Error::Protocol {
                    protocol: "circuit",
                    message: "boundary constraint outside the trace",
                });
            }
        }

        validate_fixed_columns(&self.fixed, &virtual_layout, Some(num_vars))?;

        for bc in &self.boundaries {
            if let BoundaryTarget::PublicInput(slot) = bc.target
                && slot >= self.num_public_inputs
            {
                return Err(errors::Error::Protocol {
                    protocol: "circuit",
                    message: "boundary public input slot out of range",
                });
            }
        }

        let chiplet_specs: Vec<Vec<(String, PermutationCheckSpec)>> = self
            .chiplet_defs
            .iter()
            .map(|cd| cd.permutation_checks())
            .collect();

        let endpoints = self
            .buses
            .iter()
            .chain(chiplet_specs.iter().flatten())
            .map(|(id, s)| (id.as_str(), s));

        validate_bus_set(endpoints)?;

        let table = ChipletDef::from_parts(
            self.name,
            width,
            ast,
            self.physical,
            virtual_layout,
            self.boundaries,
            self.fixed,
            (!one_to_one).then_some(expander),
            self.buses,
        );

        Ok(CircuitProgram {
            table,
            num_public_inputs: self.num_public_inputs,
            inline_chiplets: self.inline_chiplets,
            inline_kernels: self.inline_hints,
            chiplet_defs: self.chiplet_defs,
        })
    }

    fn mount_inner(&mut self, def: ChipletDef<F>, link: bool) -> Mounted {
        self.flush_pending();

        let chiplet_idx = self.inline_chiplets.len();
        let offset = self.num_virtual;
        let num_columns = Air::<F>::num_columns(&def);

        let physical = PhysRange {
            start: self.physical.len(),
            count: def.column_layout().len(),
        };

        let mut ast = def.constraint_ast();
        ast.arena.shift_cells(offset);

        self.inline_hints.push(InlineKernelHint {
            chiplet_idx,
            root_offset: self.cs.num_roots(),
            column_offset: offset,
        });

        self.cs.merge_ast(ast);

        for mut fc in Air::<F>::fixed_columns(&def) {
            fc.col_idx += offset;

            self.fixed.push(fc);
        }

        for mut bc in def.boundary_constraints() {
            bc.col_idx += offset;

            self.boundaries.push(bc);
        }

        match def.virtual_expander() {
            Some(e) => {
                let expander = core::mem::take(&mut self.expander);

                self.expander = expander.append(e);

                self.physical.extend_from_slice(def.column_layout());

                self.num_virtual += num_columns;
            }
            None => {
                self.schema(def.column_layout());
            }
        }

        if link {
            for (bus_id, mut spec) in def.permutation_checks() {
                spec.shift_column_indices(offset);
                self.buses.push((bus_id, spec));
            }
        }

        self.inline_chiplets.push(def);

        Mounted {
            offset,
            num_columns,
            physical,
        }
    }

    fn next_slot(&mut self) -> usize {
        let slot = self.num_public_inputs;
        self.num_public_inputs += 1;

        slot
    }

    fn flush_pending(&mut self) {
        let prior = self.pending_run.take();
        self.flush_run(prior);
    }

    fn flush_run(&mut self, run: Option<(ColumnType, usize)>) {
        let Some((ty, count)) = run else {
            return;
        };

        let expander = core::mem::take(&mut self.expander);

        self.expander = match ty {
            ColumnType::Bit => expander.control_bits(count),
            _ => expander.pass_through(count, ty),
        };
    }
}

/// A `Circuit`'s compiled program.
#[derive(Clone)]
pub struct CircuitProgram<F: TowerField> {
    table: ChipletDef<F>,
    num_public_inputs: usize,
    inline_chiplets: Vec<ChipletDef<F>>,
    inline_kernels: Vec<InlineKernelHint>,
    chiplet_defs: Vec<ChipletDef<F>>,
}

impl<F: TowerField> Air<F> for CircuitProgram<F> {
    fn name(&self) -> String {
        self.table.name()
    }

    fn num_columns(&self) -> usize {
        self.table.num_columns()
    }

    fn boundary_constraints(&self) -> Vec<BoundaryConstraint<F>> {
        self.table.boundary_constraints()
    }

    fn column_layout(&self) -> &[ColumnType] {
        self.table.column_layout()
    }

    fn virtual_column_layout(&self) -> &[ColumnType] {
        self.table.virtual_column_layout()
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        self.table.permutation_checks()
    }

    fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
        self.table.fixed_columns()
    }

    fn virtual_expander(&self) -> Option<&VirtualExpander> {
        self.table.virtual_expander()
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        self.table.constraint_ast()
    }

    fn inline_chiplets(&self) -> errors::Result<Vec<ChipletDef<F>>> {
        Ok(self.inline_chiplets.clone())
    }

    fn inline_chiplet_kernels(&self) -> Vec<InlineKernelHint> {
        self.inline_kernels.clone()
    }
}

impl<F: TowerField> Program<F> for CircuitProgram<F> {
    fn num_public_inputs(&self) -> usize {
        self.num_public_inputs
    }

    fn chiplet_defs(&self) -> errors::Result<Vec<ChipletDef<F>>> {
        Ok(self.chiplet_defs.clone())
    }
}

fn validate_source_range(source: &Source, width: usize) -> errors::Result<()> {
    let out_of_range = match source {
        Source::Column(idx) | Source::PhaseColumn(idx) => *idx >= width,
        Source::Columns(indices) => indices.iter().any(|&idx| idx >= width),
        Source::RowIndexLeBytes(_) | Source::RowIndexByte(_) | Source::Const(_) => false,
    };

    if out_of_range {
        return Err(errors::Error::Protocol {
            protocol: "circuit",
            message: "bus source column out of range",
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::digest::program_id;
    use crate::permutation::{REQUEST_IDX_LABEL, Source};
    use alloc::string::ToString;
    use alloc::vec;
    use hekate_math::Block128;

    type F = Block128;

    #[derive(Clone)]
    struct MiniChiplet;

    impl Air<F> for MiniChiplet {
        fn name(&self) -> String {
            "MiniChiplet".to_string()
        }

        fn num_columns(&self) -> usize {
            2
        }

        fn boundary_constraints(&self) -> Vec<BoundaryConstraint<F>> {
            vec![BoundaryConstraint::with_constant(0, 0, F::ZERO)]
        }

        fn column_layout(&self) -> &[ColumnType] {
            &[ColumnType::B32, ColumnType::Bit]
        }

        fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
            vec![(
                "mini_bus".to_string(),
                PermutationCheckSpec::new(
                    vec![
                        (Source::Column(0), b"k_v"),
                        (Source::RowIndexLeBytes(4), REQUEST_IDX_LABEL),
                    ],
                    Some(1),
                ),
            )]
        }

        fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
            vec![FixedColumn::prefix(1, 1)]
        }

        fn constraint_ast(&self) -> ConstraintAst<F> {
            let cs = ConstraintSystem::<F>::new();
            cs.assert_boolean(cs.col(1));

            cs.build()
        }
    }

    fn mini_def() -> ChipletDef<F> {
        ChipletDef::from_air(&MiniChiplet).unwrap()
    }

    fn clocked_spec(key_col: usize, selector: usize) -> PermutationCheckSpec {
        PermutationCheckSpec::new(
            vec![
                (Source::Column(key_col), b"k_v"),
                (Source::RowIndexLeBytes(4), REQUEST_IDX_LABEL),
            ],
            Some(selector),
        )
    }

    #[test]
    fn mount_matches_manual_shift() {
        let mut cx = Circuit::<F>::new("mini_host", 16).unwrap();
        let host_key = cx.column(ColumnType::B32);

        let cs = cx.cs();
        cs.constrain(cs.col(host_key.index()) * cs.col(host_key.index()));

        let mounted = cx.mount(mini_def());

        assert_eq!(mounted.offset(), 1);

        let program = cx.compile().unwrap();

        assert_eq!(
            program.column_layout(),
            &[ColumnType::B32, ColumnType::B32, ColumnType::Bit]
        );

        let hints = program.inline_chiplet_kernels();

        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].chiplet_idx, 0);
        assert_eq!(hints[0].root_offset, 1);
        assert_eq!(hints[0].column_offset, 1);

        let mut manual = {
            let cs = ConstraintSystem::<F>::new();
            cs.constrain(cs.col(0) * cs.col(0));

            cs.build()
        };

        let mut chiplet_ast = Air::<F>::constraint_ast(&MiniChiplet);
        chiplet_ast.arena.shift_cells(1);

        manual.merge(chiplet_ast);

        assert_eq!(
            program.constraint_ast().to_constraints(),
            manual.to_constraints()
        );

        let specs = program.permutation_checks();

        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].1.selector, Some(2));
        assert_eq!(specs[0].1.sources[0].0, Source::Column(1));

        let pins = program.fixed_columns();

        assert_eq!(pins.len(), 1);
        assert_eq!(pins[0].col_idx, 2);

        let boundaries = program.boundary_constraints();

        assert_eq!(boundaries.len(), 1);
        assert_eq!(boundaries[0].col_idx, 1);
    }

    #[test]
    fn mount_unlinked_keeps_columns_and_drops_endpoint() {
        let mut cx = Circuit::<F>::new("isolated_host", 16).unwrap();
        let mounted = cx.mount_unlinked(mini_def());

        let program = cx.compile().unwrap();

        assert_eq!(mounted.offset(), 0);
        assert_eq!(program.num_columns(), 2);
        assert!(program.permutation_checks().is_empty());
        assert_eq!(program.fixed_columns().len(), 1);
    }

    #[test]
    fn plain_columns_elide_expander() {
        let mut cx = Circuit::<F>::new("plain", 16).unwrap();
        cx.columns(3, ColumnType::B64);
        cx.column(ColumnType::Bit);

        let program = cx.compile().unwrap();

        assert!(program.virtual_expander().is_none());
        assert_eq!(program.num_columns(), 4);
        assert_eq!(program.column_layout(), program.virtual_column_layout());
    }

    #[test]
    fn publish_assigns_sequential_slots() {
        let mut cx = Circuit::<F>::new("slots", 16).unwrap();
        let val = cx.column(ColumnType::B32);

        let s0 = cx.publish(val, 3);
        let s1 = cx.publish(val, 5);

        assert_eq!((s0, s1), (0, 1));

        let program = cx.compile().unwrap();

        assert_eq!(program.num_public_inputs(), 2);

        let boundaries = program.boundary_constraints();
        assert_eq!(boundaries.len(), 2);
        assert_eq!(boundaries[0].target, BoundaryTarget::PublicInput(0));
        assert_eq!(boundaries[1].target, BoundaryTarget::PublicInput(1));
    }

    #[test]
    fn compile_rejects_witness_bus_selector() {
        let mut cx = Circuit::<F>::new("discipline", 16).unwrap();
        let key = cx.column(ColumnType::B32);
        let sel = cx.column(ColumnType::Bit);

        cx.bus("lone_bus", clocked_spec(key.index(), sel.index()));

        assert!(cx.compile().is_err());
    }

    #[test]
    fn compile_keeps_pinned_selector_unrooted() {
        let mut cx = Circuit::<F>::new("pinned", 16).unwrap();
        let key = cx.column(ColumnType::B32);
        let sel = cx.column(ColumnType::Bit);

        cx.fix(sel, FixedShape::Sparse(vec![(0, F::ONE)]));
        cx.bus("lone_bus", clocked_spec(key.index(), sel.index()));

        assert!(cx.compile().unwrap().constraint_ast().roots.is_empty());
    }

    #[test]
    fn compile_rejects_out_of_range_bus_source() {
        let mut cx = Circuit::<F>::new("oor", 16).unwrap();
        let sel = cx.column(ColumnType::Bit);

        cx.bus("bad_bus", clocked_spec(99, sel.index()));

        assert!(cx.compile().is_err());
    }

    #[test]
    fn mounted_expander_is_kept_and_offset() {
        let expander_def = {
            #[derive(Clone)]
            struct Packed {
                expander: VirtualExpander,
            }

            impl Air<F> for Packed {
                fn num_columns(&self) -> usize {
                    self.expander.num_virtual_columns()
                }

                fn column_layout(&self) -> &[ColumnType] {
                    &[ColumnType::B32, ColumnType::Bit]
                }

                fn virtual_expander(&self) -> Option<&VirtualExpander> {
                    Some(&self.expander)
                }

                fn constraint_ast(&self) -> ConstraintAst<F> {
                    ConstraintSystem::<F>::new().build()
                }
            }

            ChipletDef::from_air(&Packed {
                expander: VirtualExpander::new()
                    .expand_bits(1, ColumnType::B32)
                    .control_bits(1)
                    .build()
                    .unwrap(),
            })
            .unwrap()
        };

        let mut cx = Circuit::<F>::new("packed_host", 16).unwrap();

        let host = cx.column(ColumnType::B64);
        let mounted = cx.mount(expander_def);

        assert_eq!(host.index(), 0);
        assert_eq!(mounted.offset(), 1);
        assert_eq!(mounted.num_columns(), 33);

        let program = cx.compile().unwrap();

        assert!(program.virtual_expander().is_some());
        assert_eq!(program.num_columns(), 34);
        assert_eq!(program.virtual_column_layout()[0], ColumnType::B64);
        assert_eq!(program.virtual_column_layout()[1], ColumnType::Bit);
        assert_eq!(program.virtual_column_layout()[33], ColumnType::Bit);
    }

    #[test]
    fn mount_keeps_nested_inline_chiplets_on_def() {
        let inner = {
            let mut cx = Circuit::<F>::new("inner", 16).unwrap();
            let own = cx.column(ColumnType::B32);

            let cs = cx.cs();
            cs.constrain(cs.col(own.index()) * cs.col(own.index()));

            cx.mount(mini_def());
            cx.compile().unwrap()
        };

        let def = ChipletDef::from_air(&inner).unwrap();
        let nested_hints = vec![InlineKernelHint {
            chiplet_idx: 0,
            root_offset: 1,
            column_offset: 1,
        }];

        assert_eq!(def.inline_chiplet_kernels(), nested_hints);

        let bare = ChipletDef::<F>::from_wire(
            Air::<F>::name(&def),
            Air::<F>::num_columns(&def),
            def.constraint_ast(),
            def.column_layout().to_vec(),
            def.virtual_column_layout().to_vec(),
            def.boundary_constraints(),
            Air::<F>::fixed_columns(&def),
            def.virtual_expander().cloned(),
            def.permutation_checks(),
            Vec::new(),
            Vec::new(),
        )
        .unwrap();

        let outer = |mounted_def: ChipletDef<F>| {
            let mut cx = Circuit::<F>::new("outer", 16).unwrap();
            let own = cx.column(ColumnType::B64);

            let cs = cx.cs();
            cs.constrain(cs.col(own.index()) * cs.col(own.index()));

            let mounted = cx.mount(mounted_def);

            (mounted, cx.compile().unwrap())
        };

        let (mounted, program) = outer(def);

        assert_eq!(mounted.physical(), PhysRange { start: 1, count: 3 });
        assert_eq!(
            program.inline_chiplet_kernels(),
            vec![InlineKernelHint {
                chiplet_idx: 0,
                root_offset: 1,
                column_offset: 1,
            }]
        );

        let inline_defs = Air::<F>::inline_chiplets(&program).unwrap();

        assert_eq!(inline_defs.len(), 1);
        assert_eq!(inline_defs[0].inline_hints(), nested_hints);

        let mut mini_ast = Air::<F>::constraint_ast(&MiniChiplet);
        mini_ast.arena.shift_cells(2);

        assert_eq!(
            program.constraint_ast().to_constraints()[2..],
            mini_ast.to_constraints()[..]
        );

        assert_eq!(
            program_id::<F, _>(&program).unwrap(),
            program_id::<F, _>(&outer(bare).1).unwrap()
        );

        let resnapshot = ChipletDef::from_air(&program).unwrap();

        assert_eq!(resnapshot.inline_defs()[0].inline_hints(), nested_hints);
    }

    #[test]
    fn reuse_views_match_hand_built_expander() {
        let mut cx = Circuit::<F>::new("views", 16).unwrap();

        let lanes = cx.pass_through(3, ColumnType::B64);

        cx.pass_through(2, ColumnType::B64);

        let flags = cx.schema(&[ColumnType::B16, ColumnType::B32]);
        let mounted = cx.mount(mini_def());

        let lane_bits = cx.reuse_expand_bits(cx.physical(lanes).unwrap().slice(1, 2));

        cx.reuse_expand_bits(cx.physical(flags).unwrap().slice(1, 1));
        cx.reuse_expand_bits(mounted.physical().slice(0, 1));
        cx.reuse_pass_through(&lane_bits);

        let program = cx.compile().unwrap();

        let hand = VirtualExpander::new()
            .pass_through(3, ColumnType::B64)
            .pass_through(2, ColumnType::B64)
            .pass_through(1, ColumnType::B16)
            .pass_through(1, ColumnType::B32)
            .pass_through(1, ColumnType::B32)
            .control_bits(1)
            .reuse_expand_bits(1, 2)
            .reuse_expand_bits(6, 1)
            .reuse_expand_bits(7, 1)
            .reuse_pass_through(1, 2)
            .build()
            .unwrap();

        let built = program.virtual_expander().unwrap();

        assert_eq!(built.expansion_entries(), hand.expansion_entries());
        assert_eq!(built.virtual_layout(), hand.virtual_layout());
        assert_eq!(program.column_layout().len(), hand.num_physical_columns());
    }

    #[test]
    fn physical_maps_whole_columns_only() {
        let mut cx = Circuit::<F>::new("physical", 16).unwrap();

        let packed = cx.expand_bits(1, ColumnType::B32);
        let pending = cx.columns(2, ColumnType::B64);

        assert_eq!(
            cx.physical(pending).unwrap(),
            PhysRange { start: 1, count: 2 }
        );

        assert!(cx.physical(packed.bits(0)).is_err());

        let view = cx.reuse_pass_through(&packed);

        assert_eq!(cx.physical(view).unwrap(), PhysRange { start: 0, count: 1 });
    }

    #[test]
    fn reuse_rejects_range_across_expander_entries() {
        let across_schema = {
            let mut cx = Circuit::<F>::new("across_schema", 16).unwrap();
            let flags = cx.schema(&[ColumnType::B16, ColumnType::B32]);

            cx.reuse_pass_through(cx.physical(flags).unwrap());
            cx.compile().err()
        };

        let across_mount = {
            let mut cx = Circuit::<F>::new("across_mount", 16).unwrap();
            let mounted = cx.mount(mini_def());

            cx.reuse_pass_through(mounted.physical());
            cx.compile().err()
        };

        let rejected = Some(errors::Error::Protocol {
            protocol: "virtual_expand",
            message: "reuse: source columns not found in any single fresh entry",
        });

        assert_eq!(across_schema, rejected);
        assert_eq!(across_mount, rejected);
    }
}
