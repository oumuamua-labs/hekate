// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Independent AIR Chiplet definitions.
//!
//! A `ChipletDef` snapshots a chiplet's full AIR
//! (constraints, layout, bus specs) into an owned struct.
//! The prover runs an independent ZeroCheck per chiplet.
//! The LogUp bus reconnects chiplets to the main trace.

use alloc::string::String;
use alloc::vec::Vec;
use hekate_core::errors;
use hekate_core::poly::PolyVariant;
use hekate_core::trace::{ColumnTrace, ColumnType, Trace, TraceCompatibleField};
use hekate_math::{Flat, HardwareField, PackableField, TowerField};

use crate::constraint::{BoundaryConstraint, BoundaryTarget, ConstraintAst};
use crate::expander::VirtualExpander;
use crate::outer::TableStatics;
use crate::permutation::{
    PermutationCheckSpec, RankTable, TableHeight, validate_fixed_selectors, validate_ordered_buses,
};
use crate::{Air, FixedColumn, InlineKernelHint, validate_fixed_columns};

/// Pre-computed chiplet AIR definition.
#[derive(Clone)]
pub struct ChipletDef<F: TowerField> {
    name: String,
    num_columns: usize,
    constraint_ast: ConstraintAst<F>,
    column_layout: Vec<ColumnType>,
    virtual_column_layout: Vec<ColumnType>,
    boundary_constraints: Vec<BoundaryConstraint<F>>,
    fixed_columns: Vec<FixedColumn<F>>,
    expander: Option<VirtualExpander>,
    inline_chiplets: Vec<ChipletDef<F>>,
    inline_kernels: Vec<InlineKernelHint>,

    pub permutation_checks: Vec<(String, PermutationCheckSpec)>,
}

impl<F: TowerField> ChipletDef<F> {
    /// Snapshot a chiplet's full AIR definition.
    /// Call once at setup; the source chiplet can be dropped after.
    pub fn from_air<P: Air<F> + Send + 'static>(p: &P) -> errors::Result<Self>
    where
        F: TraceCompatibleField + PackableField + HardwareField + 'static,
        <F as PackableField>::Packed: Copy + Send + Sync,
    {
        let permutation_checks = p.permutation_checks();
        for (bus_id, spec) in &permutation_checks {
            spec.validate_clock_stitching(bus_id)?;
        }

        let constraint_ast = p.constraint_ast();
        let boundary_constraints = p.boundary_constraints();
        let fixed_columns = p.fixed_columns();
        let inline_chiplets = p.inline_chiplets()?;
        let inline_kernels = p.inline_chiplet_kernels();

        validate_fixed_selectors(&permutation_checks, &fixed_columns)?;
        validate_chiplet_boundaries(&boundary_constraints, p.num_columns())?;
        validate_fixed_columns(&fixed_columns, p.virtual_column_layout(), None)?;
        validate_ordered_buses(&[RankTable {
            specs: &permutation_checks,
            fixed: &fixed_columns,
            height: TableHeight::Chiplet(None),
        }])?;
        validate_expander_coverage(p.virtual_expander(), p.column_layout())?;
        validate_column_count(p.num_columns(), p.virtual_column_layout().len())?;
        validate_inline_kernels(
            &inline_kernels,
            &inline_chiplets,
            constraint_ast.roots.len(),
            p.num_columns(),
        )?;

        Ok(Self {
            name: p.name(),
            num_columns: p.num_columns(),
            constraint_ast,
            column_layout: p.column_layout().to_vec(),
            virtual_column_layout: p.virtual_column_layout().to_vec(),
            boundary_constraints,
            fixed_columns,
            expander: p.virtual_expander().cloned(),
            inline_chiplets,
            inline_kernels,
            permutation_checks,
        })
    }

    /// Borrowed views of what the `Air` getters clone.
    pub fn ast(&self) -> &ConstraintAst<F> {
        &self.constraint_ast
    }

    pub fn boundaries(&self) -> &[BoundaryConstraint<F>] {
        &self.boundary_constraints
    }

    pub fn pins(&self) -> &[FixedColumn<F>] {
        &self.fixed_columns
    }

    pub fn statics(&self) -> TableStatics<'_, F> {
        TableStatics {
            ast: &self.constraint_ast,
            specs: &self.permutation_checks,
            fixed: &self.fixed_columns,
            boundary: &self.boundary_constraints,
        }
    }

    pub fn inline_defs(&self) -> &[ChipletDef<F>] {
        &self.inline_chiplets
    }

    pub fn inline_hints(&self) -> &[InlineKernelHint] {
        &self.inline_kernels
    }

    /// Prefixes internal bus_ids with a namespace.
    /// Bus_ids listed in `exempt` are left unchanged.
    pub fn prefix_bus_ids(&mut self, prefix: &str, exempt: &[String]) {
        for (bus_id, _) in &mut self.permutation_checks {
            if !exempt.contains(bus_id) {
                let mut prefixed = String::from(prefix);
                prefixed.push_str("::");
                prefixed.push_str(bus_id);

                *bus_id = prefixed;
            }
        }
    }

    /// Expand physical ColumnTrace into virtual PolyVariants.
    /// Uses embedded expander if present, else 1:1 mapping.
    pub fn expand_variants<'a>(
        &self,
        trace: &'a ColumnTrace,
    ) -> errors::Result<Vec<PolyVariant<'a, F>>>
    where
        F: TraceCompatibleField + 'static,
    {
        match &self.expander {
            Some(e) => e.expand_variants(trace, 0),
            None => trace.get_poly_variants::<F>(),
        }
    }

    /// Reconstruct from deserialized wire data.
    /// Validates every embedded `PermutationCheckSpec`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_wire(
        name: String,
        num_columns: usize,
        constraint_ast: ConstraintAst<F>,
        column_layout: Vec<ColumnType>,
        virtual_column_layout: Vec<ColumnType>,
        boundary_constraints: Vec<BoundaryConstraint<F>>,
        fixed_columns: Vec<FixedColumn<F>>,
        expander: Option<VirtualExpander>,
        permutation_checks: Vec<(String, PermutationCheckSpec)>,
        inline_chiplets: Vec<ChipletDef<F>>,
        inline_kernels: Vec<InlineKernelHint>,
    ) -> errors::Result<Self> {
        for (bus_id, spec) in &permutation_checks {
            spec.validate_clock_stitching(bus_id)?;
        }

        validate_fixed_selectors(&permutation_checks, &fixed_columns)?;
        validate_chiplet_boundaries(&boundary_constraints, num_columns)?;

        let virt_layout = match &expander {
            Some(e) => e.virtual_layout(),
            None => virtual_column_layout.as_slice(),
        };

        validate_fixed_columns(&fixed_columns, virt_layout, None)?;
        validate_ordered_buses(&[RankTable {
            specs: &permutation_checks,
            fixed: &fixed_columns,
            height: TableHeight::Chiplet(None),
        }])?;
        validate_column_count(num_columns, virt_layout.len())?;
        validate_inline_kernels(
            &inline_kernels,
            &inline_chiplets,
            constraint_ast.roots.len(),
            num_columns,
        )?;

        Ok(Self {
            inline_chiplets,
            inline_kernels,
            ..Self::from_parts(
                name,
                num_columns,
                constraint_ast,
                column_layout,
                virtual_column_layout,
                boundary_constraints,
                fixed_columns,
                expander,
                permutation_checks,
            )
        })
    }

    /// Main-table construction path; callers own
    /// validation (chiplet boundary rules do not apply).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        name: String,
        num_columns: usize,
        constraint_ast: ConstraintAst<F>,
        column_layout: Vec<ColumnType>,
        virtual_column_layout: Vec<ColumnType>,
        boundary_constraints: Vec<BoundaryConstraint<F>>,
        fixed_columns: Vec<FixedColumn<F>>,
        expander: Option<VirtualExpander>,
        permutation_checks: Vec<(String, PermutationCheckSpec)>,
    ) -> Self {
        Self {
            name,
            num_columns,
            constraint_ast,
            column_layout,
            virtual_column_layout,
            boundary_constraints,
            fixed_columns,
            expander,
            inline_chiplets: Vec::new(),
            inline_kernels: Vec::new(),
            permutation_checks,
        }
    }
}

impl<F: TowerField> Air<F> for ChipletDef<F> {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn num_columns(&self) -> usize {
        self.num_columns
    }

    fn boundary_constraints(&self) -> Vec<BoundaryConstraint<F>> {
        self.boundary_constraints.clone()
    }

    fn column_layout(&self) -> &[ColumnType] {
        &self.column_layout
    }

    fn virtual_column_layout(&self) -> &[ColumnType] {
        match &self.expander {
            Some(e) => e.virtual_layout(),
            None => &self.virtual_column_layout,
        }
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        self.permutation_checks.clone()
    }

    fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
        self.fixed_columns.clone()
    }

    fn virtual_expander(&self) -> Option<&VirtualExpander> {
        self.expander.as_ref()
    }

    fn parse_virtual_row(&self, bytes: &[u8], res: &mut Vec<Flat<F>>)
    where
        F: TraceCompatibleField,
    {
        if let Some(e) = &self.expander {
            res.clear();

            e.parse_row(bytes, res)
                .expect("committed row byte length must match physical_row_bytes");
            return;
        }

        res.clear();

        let mut offset = 0;
        for col_type in &self.column_layout {
            let size = col_type.byte_size();
            if offset + size <= bytes.len() {
                res.push(col_type.parse_from_bytes(&bytes[offset..offset + size]));
                offset += size;
            }
        }
    }

    fn constraint_ast(&self) -> ConstraintAst<F> {
        self.constraint_ast.clone()
    }

    fn inline_chiplets(&self) -> errors::Result<Vec<ChipletDef<F>>> {
        Ok(self.inline_chiplets.clone())
    }

    fn inline_chiplet_kernels(&self) -> Vec<InlineKernelHint> {
        self.inline_kernels.clone()
    }
}

/// Chiplets carry no `public_inputs`; a `PublicInput`
/// boundary target is unsatisfiable; reject it at
/// snapshot time. Also rejects out-of-range `col_idx`.
fn validate_chiplet_boundaries<F>(
    boundaries: &[BoundaryConstraint<F>],
    num_columns: usize,
) -> errors::Result<()> {
    for bc in boundaries {
        if bc.col_idx >= num_columns {
            return Err(errors::Error::Protocol {
                protocol: "boundary",
                message: "chiplet boundary col_idx out of range",
            });
        }

        if matches!(bc.target, BoundaryTarget::PublicInput(_)) {
            return Err(errors::Error::Protocol {
                protocol: "boundary",
                message: "chiplet boundaries must use BoundaryTarget::Constant",
            });
        }
    }

    Ok(())
}

/// Physical columns outside every expansion entry enter
/// no master fold; nothing binds their committed cells.
fn validate_expander_coverage(
    expander: Option<&VirtualExpander>,
    layout: &[ColumnType],
) -> errors::Result<()> {
    let covered = match expander {
        Some(e) => e.num_physical_columns(),
        None => layout.len(),
    };

    if covered != layout.len() {
        return Err(errors::Error::Protocol {
            protocol: "chiplet",
            message: "virtual_expander does not tile column_layout",
        });
    }

    Ok(())
}

fn validate_column_count(num_columns: usize, virtual_columns: usize) -> errors::Result<()> {
    if num_columns != virtual_columns {
        return Err(errors::Error::Protocol {
            protocol: "chiplet",
            message: "num_columns does not match the virtual column layout",
        });
    }

    Ok(())
}

fn validate_inline_kernels<F: TowerField>(
    hints: &[InlineKernelHint],
    chiplets: &[ChipletDef<F>],
    num_roots: usize,
    num_columns: usize,
) -> errors::Result<()> {
    for hint in hints {
        let inside = chiplets.get(hint.chiplet_idx).is_some_and(|cd| {
            let roots_end = hint.root_offset.checked_add(cd.constraint_ast.roots.len());
            let columns_end = hint.column_offset.checked_add(cd.num_columns);

            roots_end.is_some_and(|end| end <= num_roots)
                && columns_end.is_some_and(|end| end <= num_columns)
        });

        if !inside {
            return Err(errors::Error::Protocol {
                protocol: "chiplet",
                message: "inline kernel hint outside the snapshot",
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::builder::ConstraintSystem;
    use crate::define_columns;
    use crate::permutation::{
        BusKind, ChallengeLabel, EMIT_RANK_LABEL, PermutationCheckSpec, Side, Source,
    };
    use crate::{ConstraintAst, FixedShape};
    use alloc::string::String;
    use alloc::vec;
    use hekate_core::trace::ColumnType;
    use hekate_math::Block128;

    type F = Block128;

    define_columns! {
        PairedAirCols {
            KEY: B32,
            S_SEND: Bit,
            S_RECV: Bit,
        }
    }

    #[derive(Clone)]
    struct ExpanderAir {
        expander: VirtualExpander,
    }

    impl Air<F> for ExpanderAir {
        fn num_columns(&self) -> usize {
            self.expander.virtual_layout().len()
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

    #[derive(Clone)]
    struct OneBusAir {
        spec: PermutationCheckSpec,
    }

    impl Air<F> for OneBusAir {
        fn num_columns(&self) -> usize {
            2
        }

        fn column_layout(&self) -> &[ColumnType] {
            &[ColumnType::B32, ColumnType::Bit]
        }

        fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
            vec![("test_bus".into(), self.spec.clone())]
        }

        fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
            liveness_pin(1)
        }

        fn constraint_ast(&self) -> ConstraintAst<F> {
            ConstraintSystem::<F>::new().build()
        }
    }

    #[derive(Clone)]
    struct PairedAir {
        pinned: bool,
    }

    impl Air<F> for PairedAir {
        fn num_columns(&self) -> usize {
            PairedAirCols::NUM_COLUMNS
        }

        fn column_layout(&self) -> &[ColumnType] {
            static LAYOUT: std::sync::OnceLock<Vec<ColumnType>> = std::sync::OnceLock::new();
            LAYOUT.get_or_init(PairedAirCols::build_layout)
        }

        fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
            let sources = vec![
                (Source::Column(PairedAirCols::KEY), b"k_a" as ChallengeLabel),
                (Source::RowIndexLeBytes(4), b"k_clk" as ChallengeLabel),
            ];

            vec![(
                "paired_test_bus".into(),
                PermutationCheckSpec::new_paired(
                    sources,
                    PairedAirCols::S_SEND,
                    PairedAirCols::S_RECV,
                    BusKind::Permutation,
                ),
            )]
        }

        fn fixed_columns(&self) -> Vec<FixedColumn<F>> {
            match self.pinned {
                true => vec![
                    FixedColumn::sparse(PairedAirCols::S_SEND, vec![(0, F::ONE)]),
                    FixedColumn::sparse(PairedAirCols::S_RECV, vec![(1, F::ONE)]),
                ],
                false => Vec::new(),
            }
        }

        fn constraint_ast(&self) -> ConstraintAst<F> {
            ConstraintSystem::<F>::new().build()
        }
    }

    #[derive(Clone)]
    struct HintedAir {
        hints: Vec<InlineKernelHint>,
    }

    impl Air<F> for HintedAir {
        fn num_columns(&self) -> usize {
            2
        }

        fn column_layout(&self) -> &[ColumnType] {
            &[ColumnType::B32, ColumnType::Bit]
        }

        fn constraint_ast(&self) -> ConstraintAst<F> {
            let cs = ConstraintSystem::<F>::new();
            cs.assert_boolean(cs.col(1));

            cs.build()
        }

        fn inline_chiplets(&self) -> errors::Result<Vec<ChipletDef<F>>> {
            match self.hints.is_empty() {
                true => Ok(Vec::new()),
                false => Ok(vec![ChipletDef::from_air(&HintedAir {
                    hints: Vec::new(),
                })?]),
            }
        }

        fn inline_chiplet_kernels(&self) -> Vec<InlineKernelHint> {
            self.hints.clone()
        }
    }

    fn key_only() -> Vec<(Source, ChallengeLabel)> {
        vec![(Source::Column(0), b"k_a")]
    }

    fn key_with_clock() -> Vec<(Source, ChallengeLabel)> {
        vec![
            (Source::Column(0), b"k_a"),
            (Source::RowIndexLeBytes(4), b"k_clk"),
        ]
    }

    fn key_with_rank() -> Vec<(Source, ChallengeLabel)> {
        vec![
            (Source::Column(0), b"k_a"),
            (Source::EmitRank(Side::Response), EMIT_RANK_LABEL),
        ]
    }

    fn snapshot(spec: PermutationCheckSpec) -> errors::Result<ChipletDef<F>> {
        ChipletDef::from_air(&OneBusAir { spec })
    }

    fn assert_logup_bus_err<T>(res: errors::Result<T>) {
        match res {
            Err(errors::Error::Protocol { protocol, .. }) => {
                assert_eq!(protocol, "logup_bus");
            }
            Ok(_) => panic!("expected Err(Protocol {{ protocol: \"logup_bus\", .. }})"),
            Err(other) => panic!("expected Err(Protocol), got {:?}", other),
        }
    }

    fn liveness_pin(col: usize) -> Vec<FixedColumn<F>> {
        vec![FixedColumn::sparse(col, vec![(0, F::ONE)])]
    }

    #[test]
    fn def_rejects_permutation_without_clock() {
        let spec = PermutationCheckSpec::new(key_only(), Some(1));
        assert_logup_bus_err(snapshot(spec));
    }

    #[test]
    fn def_rejects_permutation_with_empty_waiver() {
        let spec = PermutationCheckSpec::new(key_only(), Some(1)).with_clock_waiver("");
        assert_logup_bus_err(snapshot(spec));
    }

    #[test]
    fn def_rejects_permutation_with_clock_and_waiver() {
        let spec =
            PermutationCheckSpec::new(key_with_clock(), Some(1)).with_clock_waiver("redundant");
        assert_logup_bus_err(snapshot(spec));
    }

    #[test]
    fn def_rejects_lookup_with_waiver() {
        let spec = PermutationCheckSpec::new_lookup(key_only(), Some(1)).with_clock_waiver("nope");
        assert_logup_bus_err(snapshot(spec));
    }

    #[test]
    fn def_accepts_permutation_with_row_index() {
        let spec = PermutationCheckSpec::new(key_with_clock(), Some(1));
        snapshot(spec).expect("permutation bus with row-index source must accept");
    }

    #[test]
    fn def_accepts_permutation_with_clock_waiver() {
        let spec = PermutationCheckSpec::new(key_only(), Some(1))
            .with_clock_waiver("see foo.rs:42: structurally unique by AIR body");
        snapshot(spec).expect("permutation bus with non-empty clock_waiver must accept");
    }

    #[test]
    fn def_accepts_lookup_without_clock() {
        let spec = PermutationCheckSpec::new_lookup(key_only(), Some(1));
        snapshot(spec).expect("lookup bus without clock must accept");
    }

    #[test]
    fn def_rejects_permutation_with_too_short_waiver() {
        let spec = PermutationCheckSpec::new(key_only(), Some(1)).with_clock_waiver("see x.rs");
        assert_logup_bus_err(snapshot(spec));
    }

    #[test]
    fn def_rejects_permutation_with_missing_see_citation() {
        let spec = PermutationCheckSpec::new(key_only(), Some(1))
            .with_clock_waiver("structurally unique by AIR body but no file citation prefix here");
        assert_logup_bus_err(snapshot(spec));
    }

    #[test]
    fn chiplet_def_rejects_paired_spec_with_witness_selectors() {
        assert_logup_bus_err(ChipletDef::<F>::from_air(&PairedAir { pinned: false }));
    }

    #[test]
    fn chiplet_def_accepts_paired_spec_with_fixed_selectors() {
        ChipletDef::<F>::from_air(&PairedAir { pinned: true })
            .expect("paired AIR with fixed selectors must snapshot");
    }

    #[test]
    fn chiplet_def_requires_expander_to_tile_the_layout() {
        let tiled = ExpanderAir {
            expander: VirtualExpander::new()
                .expand_bits(1, ColumnType::B32)
                .control_bits(1)
                .build()
                .unwrap(),
        };

        ChipletDef::<F>::from_air(&tiled).expect("tiling expander must snapshot");

        let short = ExpanderAir {
            expander: VirtualExpander::new()
                .expand_bits(1, ColumnType::B32)
                .build()
                .unwrap(),
        };

        assert!(ChipletDef::<F>::from_air(&short).is_err());
    }

    #[test]
    fn chiplet_def_rejects_hint_outside_the_snapshot() {
        let hint = |chiplet_idx, root_offset, column_offset| HintedAir {
            hints: vec![InlineKernelHint {
                chiplet_idx,
                root_offset,
                column_offset,
            }],
        };

        let def = ChipletDef::<F>::from_air(&hint(0, 0, 0)).unwrap();

        assert_eq!(def.inline_chiplet_kernels(), hint(0, 0, 0).hints);

        for stray in [
            hint(1, 0, 0),
            hint(0, 1, 0),
            hint(0, 0, 1),
            hint(0, usize::MAX, 0),
        ] {
            assert!(ChipletDef::<F>::from_air(&stray).is_err());
        }
    }

    #[test]
    fn validator_rejects_recv_selector_without_send_selector() {
        let spec = PermutationCheckSpec {
            sources: vec![
                (Source::Column(PairedAirCols::KEY), b"k_a" as ChallengeLabel),
                (Source::RowIndexLeBytes(4), b"k_clk" as ChallengeLabel),
            ],
            selector: None,
            recv_selector: Some(PairedAirCols::S_RECV),
            kind: BusKind::Permutation,
            clock_waiver: None,
        };

        assert_logup_bus_err(validate_fixed_selectors::<F>(
            &[("asym_bus".into(), spec)],
            &liveness_pin(PairedAirCols::S_RECV),
        ));
    }

    #[test]
    fn def_rejects_witness_selector() {
        let spec = PermutationCheckSpec::new(key_with_clock(), Some(0));
        assert_logup_bus_err(snapshot(spec));
    }

    #[test]
    fn def_accepts_absent_selector() {
        let spec = PermutationCheckSpec::new(key_with_clock(), None);
        snapshot(spec).expect("selector-free bus must accept");
    }

    #[test]
    fn validator_rejects_selector_pinned_to_substituted_shape() {
        let spec = PermutationCheckSpec::new(key_with_clock(), Some(1));

        for shape in [
            FixedShape::FirstRow,
            FixedShape::LastRow,
            FixedShape::Custom(vec![true, false]),
        ] {
            assert_logup_bus_err(validate_fixed_selectors(
                &[("svc".into(), spec.clone())],
                &[FixedColumn::<F> { col_idx: 1, shape }],
            ));
        }
    }

    #[test]
    fn validator_accepts_selector_pinned_to_overlay_shape() {
        let spec = PermutationCheckSpec::new(key_with_clock(), Some(1));

        validate_fixed_selectors(&[("svc".into(), spec)], &liveness_pin(1))
            .expect("overlay-pinned selector must accept");
    }

    #[test]
    fn from_wire_rejects_hint_outside_snapshot() {
        let bare = HintedAir { hints: Vec::new() };

        let wire = |chiplet_idx, root_offset| {
            ChipletDef::<F>::from_wire(
                String::from("wired"),
                2,
                bare.constraint_ast(),
                bare.column_layout().to_vec(),
                bare.column_layout().to_vec(),
                Vec::new(),
                Vec::new(),
                None,
                Vec::new(),
                vec![ChipletDef::from_air(&bare).unwrap()],
                vec![InlineKernelHint {
                    chiplet_idx,
                    root_offset,
                    column_offset: 0,
                }],
            )
        };

        assert!(wire(0, 0).is_ok());
        assert!(wire(1, 0).is_err());
        assert!(wire(0, 1).is_err());
    }

    #[test]
    fn def_runs_ordered_bus_predicate() {
        snapshot(PermutationCheckSpec::new(key_with_rank(), Some(1))).unwrap();

        let lookup = PermutationCheckSpec::new_lookup(key_with_rank(), Some(1));

        assert_logup_bus_err(snapshot(lookup.clone()));

        let air = OneBusAir { spec: lookup };

        assert_logup_bus_err(ChipletDef::<F>::from_wire(
            String::from("wired"),
            2,
            air.constraint_ast(),
            air.column_layout().to_vec(),
            air.column_layout().to_vec(),
            Vec::new(),
            air.fixed_columns(),
            None,
            Air::<F>::permutation_checks(&air),
            Vec::new(),
            Vec::new(),
        ));
    }
}
