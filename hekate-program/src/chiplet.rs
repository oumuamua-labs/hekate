// SPDX-License-Identifier: Apache-2.0
// This file is part of the hekate project.
// Copyright (C) 2026 Andrei Kochergin <andrei@oumuamua.dev>
// Copyright (C) 2026 Oumuamua Labs <info@oumuamua.dev>. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Independent AIR Chiplet definitions.
//!
//! A `ChipletDef` snapshots a chiplet's full AIR
//! (constraints, layout, bus specs) into an owned struct.
//! The prover runs an independent ZeroCheck per chiplet.
//! The bus (GPA) reconnects chiplets to the main trace.

use crate::constraint::{
    BoundaryConstraint, BoundaryTarget, ConstraintAst, ConstraintExpr, ExprId,
};
use crate::expander::VirtualExpander;
use crate::permutation::PermutationCheckSpec;
use crate::{Air, LagrangePin, ProgramCell, validate_lagrange_pins};
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use hekate_core::errors;
use hekate_core::poly::PolyVariant;
use hekate_core::trace::{ColumnTrace, ColumnType, Trace, TraceCompatibleField};
use hekate_math::{Flat, HardwareField, PackableField, TowerField};

/// Pre-computed chiplet AIR definition.
pub struct ChipletDef<F: TowerField> {
    name: String,
    num_columns: usize,
    constraint_ast: ConstraintAst<F>,
    column_layout: Vec<ColumnType>,
    virtual_column_layout: Vec<ColumnType>,
    boundary_constraints: Vec<BoundaryConstraint<F>>,
    lagrange_pins: Vec<LagrangePin>,
    expander: Option<VirtualExpander>,
    pub permutation_checks: Vec<(String, PermutationCheckSpec)>,
}

impl<F: TowerField> ChipletDef<F> {
    /// Snapshot a chiplet's full AIR definition.
    /// Call once at setup; the source
    /// chiplet can be dropped after.
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
        validate_paired_bus_mutex(&permutation_checks, &constraint_ast)?;

        let boundary_constraints = p.boundary_constraints();
        validate_chiplet_boundaries(&boundary_constraints, p.num_columns())?;

        let lagrange_pins = p.lagrange_pinned_columns();
        validate_lagrange_pins(&lagrange_pins, p.num_columns(), None)?;

        Ok(Self {
            name: p.name(),
            num_columns: p.num_columns(),
            constraint_ast,
            column_layout: p.column_layout().to_vec(),
            virtual_column_layout: p.virtual_column_layout().to_vec(),
            boundary_constraints,
            lagrange_pins,
            expander: p.virtual_expander().cloned(),
            permutation_checks,
        })
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

    /// Expand physical ColumnTrace into virtual
    /// PolyVariants. Uses embedded expander
    /// if present, else 1:1 mapping.
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
        lagrange_pins: Vec<LagrangePin>,
        expander: Option<VirtualExpander>,
        permutation_checks: Vec<(String, PermutationCheckSpec)>,
    ) -> errors::Result<Self> {
        for (bus_id, spec) in &permutation_checks {
            spec.validate_clock_stitching(bus_id)?;
        }

        validate_paired_bus_mutex(&permutation_checks, &constraint_ast)?;
        validate_chiplet_boundaries(&boundary_constraints, num_columns)?;
        validate_lagrange_pins(&lagrange_pins, num_columns, None)?;

        Ok(Self {
            name,
            num_columns,
            constraint_ast,
            column_layout,
            virtual_column_layout,
            boundary_constraints,
            lagrange_pins,
            expander,
            permutation_checks,
        })
    }
}

impl<F: TowerField> Clone for ChipletDef<F> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            num_columns: self.num_columns,
            constraint_ast: self.constraint_ast.clone(),
            column_layout: self.column_layout.clone(),
            virtual_column_layout: self.virtual_column_layout.clone(),
            boundary_constraints: self.boundary_constraints.clone(),
            lagrange_pins: self.lagrange_pins.clone(),
            expander: self.expander.clone(),
            permutation_checks: self.permutation_checks.clone(),
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

    fn lagrange_pinned_columns(&self) -> Vec<LagrangePin> {
        self.lagrange_pins.clone()
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
}

// =================================================================
// Composite Chiplet Composition
// =================================================================

/// Factory trait for deterministic
/// ChipletDef construction.
trait AirFactory<F: TowerField>: Send + Sync {
    fn build(&self) -> errors::Result<ChipletDef<F>>;
    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)>;
    fn clone_box(&self) -> Box<dyn AirFactory<F>>;
}

impl<F, A> AirFactory<F> for A
where
    F: TowerField + TraceCompatibleField + PackableField + HardwareField + 'static,
    <F as PackableField>::Packed: Copy + Send + Sync,
    A: Air<F> + Clone + Send + Sync + 'static,
{
    fn build(&self) -> errors::Result<ChipletDef<F>> {
        ChipletDef::from_air(self)
    }

    fn permutation_checks(&self) -> Vec<(String, PermutationCheckSpec)> {
        Air::permutation_checks(self)
    }

    fn clone_box(&self) -> Box<dyn AirFactory<F>> {
        Box::new(self.clone())
    }
}

impl<F: TowerField> Clone for Box<dyn AirFactory<F>> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

struct CompositeEntry<F: TraceCompatibleField> {
    air: Box<dyn AirFactory<F>>,
}

impl<F: TraceCompatibleField> Clone for CompositeEntry<F> {
    fn clone(&self) -> Self {
        Self {
            air: self.air.clone_box(),
        }
    }
}

/// Build-time composition of peer
/// chiplets into a single unit.
///
/// Flattens into standard `ChipletDef` entries
/// for the prover, no protocol-level awareness
/// of hierarchy. Internal buses are namespaced
/// with `"{name}::"` to prevent cross-composite
/// collisions. External buses pass through
/// unchanged.
///
/// All chiplets within a composite are peers, no mandatory root.
pub struct CompositeChiplet<F: TraceCompatibleField> {
    name: String,
    chiplets: Vec<CompositeEntry<F>>,
    external_bus_ids: Vec<String>,
    external_buses: Vec<(String, PermutationCheckSpec)>,
}

impl<F: TraceCompatibleField> Clone for CompositeChiplet<F> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            chiplets: self.chiplets.clone(),
            external_bus_ids: self.external_bus_ids.clone(),
            external_buses: self.external_buses.clone(),
        }
    }
}

impl<F: TraceCompatibleField> CompositeChiplet<F> {
    /// Start building a composite
    /// with the given namespace.
    pub fn builder(name: &str) -> CompositeChipletBuilder<F> {
        CompositeChipletBuilder {
            name: String::from(name),
            chiplets: Vec::new(),
            external_bus_ids: Vec::new(),
            external_buses: Vec::new(),
        }
    }

    /// Produce fresh ChipletDefs for `Program::chiplet_defs()`.
    pub fn flatten_defs(&self) -> errors::Result<Vec<ChipletDef<F>>> {
        let mut out = Vec::with_capacity(self.chiplets.len());
        for entry in &self.chiplets {
            let mut def = entry.air.build()?;
            def.prefix_bus_ids(&self.name, &self.external_bus_ids);

            out.push(def);
        }

        Ok(out)
    }

    /// External buses for `Program::permutation_checks()`.
    ///
    /// These are main-trace-side specs, column indices
    /// reference the main trace, not any chiplet trace.
    pub fn external_buses(&self) -> Vec<(String, PermutationCheckSpec)> {
        self.external_buses.clone()
    }

    /// Number of flattened
    /// chiplets in this composite.
    pub fn len(&self) -> usize {
        self.chiplets.len()
    }

    /// Returns true if this
    /// composite contains no chiplets.
    pub fn is_empty(&self) -> bool {
        self.chiplets.is_empty()
    }

    /// The composite's namespace.
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Builder for `CompositeChiplet`.
pub struct CompositeChipletBuilder<F: TraceCompatibleField> {
    name: String,
    chiplets: Vec<CompositeEntry<F>>,
    external_bus_ids: Vec<String>,
    external_buses: Vec<(String, PermutationCheckSpec)>,
}

impl<F: TraceCompatibleField> CompositeChipletBuilder<F> {
    /// Add a sub-chiplet.
    pub fn chiplet<A>(mut self, air: A) -> Self
    where
        A: Air<F> + Clone + Send + Sync + 'static,
        F: TraceCompatibleField + PackableField + HardwareField + 'static,
        <F as PackableField>::Packed: Copy + Send + Sync,
    {
        self.chiplets.push(CompositeEntry { air: Box::new(air) });

        self
    }

    /// Declare an external bus
    /// (connects to the main trace).
    pub fn external_bus(mut self, bus_id: &str, spec: PermutationCheckSpec) -> Self {
        self.external_bus_ids.push(String::from(bus_id));
        self.external_buses.push((String::from(bus_id), spec));

        self
    }

    /// Finalize the composite.
    ///
    /// Validates selector orthogonality:
    /// two specs on different bus_ids must
    /// not share a selector column index.
    /// Same bus_id is exempt for
    /// dual-spec intra-table check.
    pub fn build(self) -> errors::Result<CompositeChiplet<F>> {
        for (bus_id, spec) in &self.external_buses {
            spec.validate_clock_stitching(bus_id)?;
        }

        for entry in &self.chiplets {
            let checks = entry.air.permutation_checks();
            for i in 0..checks.len() {
                for j in (i + 1)..checks.len() {
                    if checks[i].0 == checks[j].0 {
                        continue;
                    }

                    if let (Some(sel_i), Some(sel_j)) = (checks[i].1.selector, checks[j].1.selector)
                        && sel_i == sel_j
                    {
                        return Err(errors::Error::Protocol {
                            protocol: "composite_chiplet",
                            message: "different bus_ids share a selector column",
                        });
                    }
                }
            }
        }

        Ok(CompositeChiplet {
            name: self.name,
            chiplets: self.chiplets,
            external_bus_ids: self.external_bus_ids,
            external_buses: self.external_buses,
        })
    }
}

// =================================================================
// Multi-Composite Helpers
// =================================================================

/// Flatten multiple composites
/// into a single chiplet def list.
///
/// Validates that no two composites
/// share the same name (would cause
/// bus namespace collisions).
pub fn compose_chiplet_defs<F: TraceCompatibleField>(
    composites: &[&CompositeChiplet<F>],
) -> errors::Result<Vec<ChipletDef<F>>> {
    for i in 0..composites.len() {
        for j in (i + 1)..composites.len() {
            if composites[i].name == composites[j].name {
                return Err(errors::Error::Protocol {
                    protocol: "composite_chiplet",
                    message: "duplicate composite name in compose_chiplet_defs",
                });
            }
        }
    }

    let mut defs = Vec::new();
    for composite in composites {
        defs.extend(composite.flatten_defs()?);
    }

    let endpoints = defs
        .iter()
        .flat_map(|d| d.permutation_checks.iter().map(|(id, s)| (id.as_str(), s)));

    crate::permutation::validate_bus_set(endpoints)?;

    Ok(defs)
}

/// Collect external buses
/// from multiple composites.
pub fn compose_external_buses<F: TraceCompatibleField>(
    composites: &[&CompositeChiplet<F>],
) -> Vec<(String, PermutationCheckSpec)> {
    let mut buses = Vec::new();
    for composite in composites {
        buses.extend(composite.external_buses());
    }

    buses
}

/// Without the mutex root, both selectors high
/// collapse the bus numerator to zero in char-2;
/// without the boolean roots, the mutex admits
/// non-zero field-element selectors that bypass
/// binary on/off semantics.
pub fn validate_paired_bus_mutex<F: TowerField>(
    specs: &[(String, PermutationCheckSpec)],
    ast: &ConstraintAst<F>,
) -> errors::Result<()> {
    for (_bus_id, spec) in specs {
        let (s_send, s_recv) = match (spec.selector, spec.recv_selector) {
            (Some(send), Some(recv)) => (send, recv),
            (None, Some(_)) => {
                return Err(errors::Error::Protocol {
                    protocol: "logup_bus",
                    message: "paired bus has recv_selector without send selector",
                });
            }
            _ => continue,
        };

        if !ast_contains_mutex_root(ast, s_send, s_recv) {
            return Err(errors::Error::Protocol {
                protocol: "logup_bus",
                message: "paired bus requires `s_send · s_recv = 0` mutex root in the AST",
            });
        }

        if !ast_contains_boolean_root(ast, s_send) {
            return Err(errors::Error::Protocol {
                protocol: "logup_bus",
                message: "paired bus requires boolean-assertion root for s_send",
            });
        }

        if !ast_contains_boolean_root(ast, s_recv) {
            return Err(errors::Error::Protocol {
                protocol: "logup_bus",
                message: "paired bus requires boolean-assertion root for s_recv",
            });
        }
    }

    Ok(())
}

/// Chiplets carry no `public_inputs`, so a `PublicInput`
/// boundary target is unsatisfiable; reject it at snapshot
/// time. Also rejects out-of-range `col_idx`.
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

/// Non-paired `Bit` selectors with no direct
/// `s·s + s` boolean root, each tagged with the
/// declaring `bus_id`. Advisory only: booleanness
/// can hold indirectly (one-hot, disjoint
/// products), callers warn rather than reject.
pub fn unconstrained_bit_selectors<'a, F: TowerField>(
    specs: &'a [(String, PermutationCheckSpec)],
    ast: &ConstraintAst<F>,
    virtual_layout: &[ColumnType],
) -> Vec<(usize, &'a str)> {
    let mut flagged: Vec<(usize, &str)> = Vec::new();
    for (bus_id, spec) in specs {
        if spec.recv_selector.is_some() {
            continue;
        }

        let Some(sel) = spec.selector else {
            continue;
        };

        if virtual_layout.get(sel) != Some(&ColumnType::Bit) {
            continue;
        }

        if !ast_contains_boolean_root(ast, sel) && !flagged.iter().any(|(s, _)| *s == sel) {
            flagged.push((sel, bus_id.as_str()));
        }
    }

    flagged
}

fn ast_contains_mutex_root<F: TowerField>(
    ast: &ConstraintAst<F>,
    s_send: usize,
    s_recv: usize,
) -> bool {
    ast.roots
        .iter()
        .any(|root| is_mutex_product(ast, *root, s_send, s_recv))
}

fn ast_contains_boolean_root<F: TowerField>(ast: &ConstraintAst<F>, col: usize) -> bool {
    ast.roots
        .iter()
        .any(|root| is_boolean_assertion(ast, *root, col))
}

fn is_boolean_assertion<F: TowerField>(ast: &ConstraintAst<F>, id: ExprId, col: usize) -> bool {
    let ConstraintExpr::Add(a, b) = ast.arena.get(id) else {
        return false;
    };

    matches_boolean_pair(ast, *a, *b, col) || matches_boolean_pair(ast, *b, *a, col)
}

fn matches_boolean_pair<F: TowerField>(
    ast: &ConstraintAst<F>,
    sq_id: ExprId,
    cell_id: ExprId,
    col: usize,
) -> bool {
    let ConstraintExpr::Mul(x, y) = ast.arena.get(sq_id) else {
        return false;
    };

    current_col_idx(ast, *x) == Some(col)
        && current_col_idx(ast, *y) == Some(col)
        && current_col_idx(ast, cell_id) == Some(col)
}

fn is_mutex_product<F: TowerField>(
    ast: &ConstraintAst<F>,
    id: ExprId,
    s_send: usize,
    s_recv: usize,
) -> bool {
    let ConstraintExpr::Mul(a, b) = ast.arena.get(id) else {
        return false;
    };

    let lhs = current_col_idx(ast, *a);
    let rhs = current_col_idx(ast, *b);

    matches!(
        (lhs, rhs),
        (Some(x), Some(y)) if (x == s_send && y == s_recv) || (x == s_recv && y == s_send)
    )
}

fn current_col_idx<F: TowerField>(ast: &ConstraintAst<F>, id: ExprId) -> Option<usize> {
    match ast.arena.get(id) {
        ConstraintExpr::Cell(ProgramCell {
            col_idx,
            next_row: false,
        }) => Some(*col_idx),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ConstraintAst;
    use crate::constraint::builder::ConstraintSystem;
    use crate::define_columns;
    use crate::permutation::{BusKind, ChallengeLabel, PermutationCheckSpec, Source};
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

        fn constraint_ast(&self) -> ConstraintAst<F> {
            ConstraintSystem::<F>::new().build()
        }
    }

    #[derive(Clone)]
    struct PairedAir {
        with_mutex: bool,
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

        fn constraint_ast(&self) -> ConstraintAst<F> {
            let cs = ConstraintSystem::<F>::new();

            cs.assert_boolean(cs.col(PairedAirCols::S_SEND));
            cs.assert_boolean(cs.col(PairedAirCols::S_RECV));

            if self.with_mutex {
                cs.constrain_named(
                    "paired_bus_mutex",
                    cs.col(PairedAirCols::S_SEND) * cs.col(PairedAirCols::S_RECV),
                );
            }

            cs.build()
        }
    }

    #[derive(Clone)]
    struct PairedNoBoolAir;

    impl Air<F> for PairedNoBoolAir {
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

        fn constraint_ast(&self) -> ConstraintAst<F> {
            let cs = ConstraintSystem::<F>::new();

            cs.constrain_named(
                "paired_bus_mutex",
                cs.col(PairedAirCols::S_SEND) * cs.col(PairedAirCols::S_RECV),
            );

            cs.build()
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
    fn paired_bus_emits_mutex_and_boolean_assertions() {
        let cs = ConstraintSystem::<F>::new();

        cs.assert_paired_bus_mutex(PairedAirCols::S_SEND, PairedAirCols::S_RECV);

        let ast = cs.build();

        let labels: Vec<_> = ast.labels.iter().filter_map(|l| *l).collect();

        assert_eq!(
            labels.iter().filter(|l| **l == "boolean").count(),
            2,
            "gadget must emit two boolean assertions"
        );
        assert_eq!(
            labels.iter().filter(|l| **l == "paired_bus_mutex").count(),
            1,
            "gadget must emit exactly one mutex root"
        );
    }

    #[test]
    fn paired_bus_shares_cell_nodes() {
        let cs = ConstraintSystem::<F>::new();

        let send_first = cs.col(PairedAirCols::S_SEND);
        let recv_first = cs.col(PairedAirCols::S_RECV);

        cs.assert_paired_bus_mutex(PairedAirCols::S_SEND, PairedAirCols::S_RECV);

        let send_again = cs.col(PairedAirCols::S_SEND);
        let recv_again = cs.col(PairedAirCols::S_RECV);

        assert_eq!(send_first.id, send_again.id, "S_SEND must dedup");
        assert_eq!(recv_first.id, recv_again.id, "S_RECV must dedup");
    }

    #[test]
    fn chiplet_def_rejects_paired_spec_without_mutex() {
        let bad = PairedAir { with_mutex: false };
        assert_logup_bus_err(ChipletDef::<F>::from_air(&bad));
    }

    #[test]
    fn chiplet_def_accepts_paired_spec_with_mutex() {
        let good = PairedAir { with_mutex: true };
        ChipletDef::<F>::from_air(&good).expect("paired AIR with mutex must snapshot");
    }

    #[test]
    fn chiplet_def_rejects_paired_spec_without_boolean_roots() {
        assert_logup_bus_err(ChipletDef::<F>::from_air(&PairedNoBoolAir));
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

        let ast = ConstraintSystem::<F>::new().build();

        assert_logup_bus_err(validate_paired_bus_mutex(
            &[("asym_bus".into(), spec)],
            &ast,
        ));
    }

    #[test]
    fn flags_bit_selector_without_boolean_root() {
        let ast = ConstraintSystem::<F>::new().build();
        let layout = vec![ColumnType::B32, ColumnType::Bit];
        let specs = vec![(
            String::from("bus"),
            PermutationCheckSpec::new(vec![(Source::Column(0), b"k" as ChallengeLabel)], Some(1)),
        )];

        assert_eq!(
            unconstrained_bit_selectors(&specs, &ast, &layout),
            vec![(1, "bus")]
        );
    }

    #[test]
    fn boolean_root_clears_bit_selector() {
        let cs = ConstraintSystem::<F>::new();
        let sel = cs.col(1);
        cs.assert_boolean(sel);
        let ast = cs.build();

        let layout = vec![ColumnType::B32, ColumnType::Bit];
        let specs = vec![(
            String::from("bus"),
            PermutationCheckSpec::new(vec![(Source::Column(0), b"k" as ChallengeLabel)], Some(1)),
        )];

        assert!(unconstrained_bit_selectors(&specs, &ast, &layout).is_empty());
    }
}
