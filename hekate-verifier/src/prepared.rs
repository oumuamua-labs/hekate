// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec::Vec;
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::ligero::RowEncoder;
use hekate_core::outer::OuterGeometry;
use hekate_core::trace::ColumnType;
use hekate_math::{BinaryFieldExtras, HardwareField, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::expander::RingSwitchPlan;
use hekate_program::outer::{TableParts, TableShape};
use hekate_program::permutation::validate_fixed_selectors;
use hekate_program::{Air, Program, digest};

/// What verifying `program` under one `Config` needs before
/// any proof: chiplet definitions, table parts, ring-switch
/// plans, shapes without heights, and the program id.
pub struct PreparedProgram<F: TowerField> {
    pub(crate) config: Config,
    pub(crate) program_id: [u8; 32],
    pub(crate) num_public_inputs: usize,
    pub(crate) virtual_column_layout: Vec<ColumnType>,
    pub(crate) main: TableParts<F>,
    pub(crate) main_plan: RingSwitchPlan,
    pub(crate) main_shape: TableShape,
    pub(crate) chiplets: Vec<ChipletDef<F>>,
    pub(crate) chiplet_plans: Vec<RingSwitchPlan>,
    pub(crate) chiplet_shapes: Vec<TableShape>,
}

impl<F: TowerField> PreparedProgram<F> {
    pub fn new<P: Program<F>>(program: &P, config: &Config) -> errors::Result<Self> {
        let main = TableParts::of(program);
        let chiplets = program.chiplet_defs()?;

        validate_fixed_selectors(&main.specs, &main.fixed)?;

        for def in &chiplets {
            validate_fixed_selectors(&def.permutation_checks, def.pins())?;
        }

        let main_entries = program.virtual_expander().map(|e| e.expansion_entries());
        let main_plan = RingSwitchPlan::new(
            program.column_layout(),
            main_entries.as_deref(),
            config.blind_units(),
            main.specs.len(),
        )?;

        let main_shape = TableShape::from_air(program, 0, &main.statics())?;

        let mut chiplet_plans = Vec::with_capacity(chiplets.len());
        let mut chiplet_shapes = Vec::with_capacity(chiplets.len());

        for def in &chiplets {
            let entries = Air::<F>::virtual_expander(def).map(|e| e.expansion_entries());

            chiplet_plans.push(RingSwitchPlan::new(
                Air::<F>::column_layout(def),
                entries.as_deref(),
                config.blind_units(),
                def.permutation_checks.len(),
            )?);

            chiplet_shapes.push(TableShape::from_air(def, 0, &def.statics())?);
        }

        let program_id = digest::program_id_of(program, &chiplets, &program.inline_chiplets()?);

        Ok(Self {
            config: config.clone(),
            program_id,
            num_public_inputs: program.num_public_inputs(),
            virtual_column_layout: program.virtual_column_layout().to_vec(),
            main,
            main_plan,
            main_shape,
            chiplets,
            chiplet_plans,
            chiplet_shapes,
        })
    }

    pub fn program_id(&self) -> &[u8; 32] {
        &self.program_id
    }
}

/// State one thread reuses across verifies:
/// the outer encoder of the last geometry seen.
pub struct VerifierScratch<F> {
    outer: Option<(OuterGeometry, RowEncoder<F>)>,
}

impl<F: BinaryFieldExtras + HardwareField> VerifierScratch<F> {
    pub fn new() -> Self {
        Self { outer: None }
    }

    pub(crate) fn encoder(&mut self, geom: &OuterGeometry) -> errors::Result<&RowEncoder<F>> {
        let entry = match self.outer.take() {
            Some((cached, encoder)) if cached == *geom => (cached, encoder),
            _ => (*geom, RowEncoder::new(geom)?),
        };

        Ok(&self.outer.insert(entry).1)
    }
}

impl<F: BinaryFieldExtras + HardwareField> Default for VerifierScratch<F> {
    fn default() -> Self {
        Self::new()
    }
}
