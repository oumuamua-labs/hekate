// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec::Vec;
use hekate_core::config::Config;
use hekate_core::errors;
use hekate_core::ligero::RowEncoder;
use hekate_core::outer::OuterGeometry;
use hekate_math::{BinaryFieldExtras, HardwareField, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::expander::RingSwitchPlan;
use hekate_program::outer::{TableShape, table_plan};
use hekate_program::permutation::validate_fixed_selectors;
use hekate_program::{Program, digest};

/// What verifying `program` under one `Config` needs before
/// any proof: chiplet definitions, table parts, ring-switch
/// plans, shapes without heights, and the program id.
pub struct PreparedProgram<F: TowerField> {
    pub(crate) config: Config,
    pub(crate) program_id: [u8; 32],
    pub(crate) num_public_inputs: usize,
    pub(crate) main: ChipletDef<F>,
    pub(crate) main_plan: RingSwitchPlan,
    pub(crate) main_shape: TableShape,
    pub(crate) chiplets: Vec<ChipletDef<F>>,
    pub(crate) chiplet_plans: Vec<RingSwitchPlan>,
    pub(crate) chiplet_shapes: Vec<TableShape>,
}

impl<F: TowerField> PreparedProgram<F> {
    pub fn new<P: Program<F>>(program: &P, config: &Config) -> errors::Result<Self> {
        // One read: the id hashes the table the proofs are checked against
        let main = ChipletDef::from_main_table(program);
        let chiplets = program.chiplet_defs()?;
        let num_public_inputs = program.num_public_inputs();

        validate_fixed_selectors(&main.permutation_checks, main.pins())?;

        for def in &chiplets {
            validate_fixed_selectors(&def.permutation_checks, def.pins())?;
        }

        let main_plan = table_plan(&main, main.permutation_checks.len(), config)?;
        let main_shape = TableShape::from_air(&main, &main.statics())?;

        let mut chiplet_plans = Vec::with_capacity(chiplets.len());
        let mut chiplet_shapes = Vec::with_capacity(chiplets.len());

        for def in &chiplets {
            chiplet_plans.push(table_plan(def, def.permutation_checks.len(), config)?);
            chiplet_shapes.push(TableShape::from_air(def, &def.statics())?);
        }

        let program_id = digest::program_id_from_parts(
            &main,
            num_public_inputs,
            &chiplets,
            &program.inline_chiplets()?,
            &program.inline_chiplet_kernels(),
        );

        Ok(Self {
            config: config.clone(),
            program_id,
            num_public_inputs,
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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::sync::Arc;
    use core::sync::atomic::{AtomicBool, Ordering};
    use hekate_core::trace::ColumnType;
    use hekate_math::Block128;
    use hekate_program::Air;
    use hekate_program::constraint::ConstraintAst;
    use hekate_program::constraint::builder::ConstraintSystem;

    type F = Block128;

    #[derive(Clone, Default)]
    struct OrderDependent {
        chiplets_read: Arc<AtomicBool>,
    }

    impl Air<F> for OrderDependent {
        fn column_layout(&self) -> &[ColumnType] {
            &[ColumnType::B32, ColumnType::B32]
        }

        fn constraint_ast(&self) -> ConstraintAst<F> {
            let cs = ConstraintSystem::<F>::new();
            cs.constrain(cs.col(0) + cs.col(1));

            if self.chiplets_read.load(Ordering::Relaxed) {
                cs.constrain(cs.col(0) * cs.col(1));
            }

            cs.build()
        }
    }

    impl Program<F> for OrderDependent {
        fn chiplet_defs(&self) -> errors::Result<Vec<ChipletDef<F>>> {
            self.chiplets_read.store(true, Ordering::Relaxed);

            Ok(Vec::new())
        }
    }

    #[test]
    fn program_id_binds_checked_main_table() {
        let program = OrderDependent::default();
        let prepared = PreparedProgram::new(&program, &Config::default()).unwrap();
        let audited = digest::program_id::<F, _>(&OrderDependent::default()).unwrap();

        assert_eq!(prepared.main.ast().roots.len(), 1);
        assert_ne!(prepared.program_id, audited);
        assert_eq!(
            prepared.program_id,
            digest::program_id_from_parts(&prepared.main, 0, &[], &[], &[])
        );
    }
}
