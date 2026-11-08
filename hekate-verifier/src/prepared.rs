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
use hekate_program::constraint::{ConstraintAst, ConstraintExpr, ExprId};
use hekate_program::digest::{self, ProgramId};
use hekate_program::expander::RingSwitchPlan;
use hekate_program::outer::{TableShape, table_plan};
use hekate_program::permutation::{validate_bus_tags, validate_fixed_selectors};
use hekate_program::{Air, Program, validate_fixed_columns};

/// What verifying `program` under one `Config` needs before
/// any proof: chiplet definitions, table parts, ring-switch
/// plans, shapes without heights, and the program id.
pub struct PreparedProgram<F: TowerField> {
    pub(crate) config: Config,
    pub(crate) program_id: ProgramId,
    pub(crate) num_public_inputs: usize,
    pub(crate) main: ChipletDef<F>,
    pub(crate) main_plan: RingSwitchPlan,
    pub(crate) main_shape: TableShape,
    pub(crate) chiplets: Vec<ChipletDef<F>>,
    pub(crate) chiplet_plans: Vec<RingSwitchPlan>,
    pub(crate) chiplet_shapes: Vec<TableShape>,
}

impl<F: TowerField> PreparedProgram<F> {
    /// Checks `program` and `config` and derives what
    /// every proof of the program is verified against.
    pub fn new<P: Program<F>>(program: &P, config: &Config) -> errors::Result<Self> {
        if config.num_queries == 0 {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "num_queries cannot be zero",
            });
        }

        // One read: the id hashes the table the proofs are checked against
        let main = ChipletDef::from_main_table(program);
        let chiplets = program.chiplet_defs()?;
        let num_public_inputs = program.num_public_inputs();

        for def in [&main].into_iter().chain(&chiplets) {
            validate_fixed_columns(def.pins(), Air::<F>::virtual_column_layout(def), None)?;
            validate_fixed_selectors(&def.permutation_checks, def.pins())?;
        }

        let endpoints = main
            .permutation_checks
            .iter()
            .chain(chiplets.iter().flat_map(|def| &def.permutation_checks))
            .map(|(bus_id, spec)| (bus_id.as_str(), spec));

        validate_bus_tags::<F>(endpoints)?;

        let main_statics = main.statics();
        let main_shape = TableShape::from_air(&main, &main_statics)?;
        let main_plan = table_plan(&main, &main_statics, &main_shape, config)?;

        check_next_reads(main_statics.ast, &main_plan)?;

        let mut chiplet_plans = Vec::with_capacity(chiplets.len());
        let mut chiplet_shapes = Vec::with_capacity(chiplets.len());

        for def in &chiplets {
            let statics = def.statics();
            let shape = TableShape::from_air(def, &statics)?;
            let plan = table_plan(def, &statics, &shape, config)?;

            check_next_reads(statics.ast, &plan)?;

            chiplet_plans.push(plan);
            chiplet_shapes.push(shape);
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

    pub fn program_id(&self) -> &ProgramId {
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

/// Errs unless `plan` ships a claim for each next-row cell
/// of `ast`. The match has no wildcard arm: a new expression
/// kind fails to compile until it says whether it reads one.
fn check_next_reads<F: TowerField>(
    ast: &ConstraintAst<F>,
    plan: &RingSwitchPlan,
) -> errors::Result<()> {
    let slots = plan.next_slots();

    for id in 0..ast.arena.len() as u32 {
        let shipped = match ast.arena.get(ExprId(id)) {
            ConstraintExpr::Cell(cell) if cell.next_row => {
                slots.get(cell.col_idx).copied().flatten().is_some()
            }
            ConstraintExpr::Cell(_)
            | ConstraintExpr::Const(_)
            | ConstraintExpr::Add(..)
            | ConstraintExpr::Mul(..)
            | ConstraintExpr::Scale(..)
            | ConstraintExpr::Sum(_) => true,
        };

        if !shipped {
            return Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "AST reads a next-row cell its plan ships no claim for",
            });
        }
    }

    Ok(())
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

    #[test]
    fn plan_without_next_row_slot_rejected() {
        let cs = ConstraintSystem::<F>::new();
        cs.constrain(cs.next(1) + cs.col(0));

        let ast = cs.build();
        let layout = [ColumnType::B32; 2];

        let shipped = RingSwitchPlan::new(&layout, None, 0, 0, &[false, true]).unwrap();
        let unshipped = RingSwitchPlan::new(&layout, None, 0, 0, &[false; 2]).unwrap();

        assert_eq!(check_next_reads(&ast, &shipped), Ok(()));
        assert_eq!(
            check_next_reads(&ast, &unshipped),
            Err(errors::Error::Protocol {
                protocol: "verifier",
                message: "AST reads a next-row cell its plan ships no claim for",
            })
        );
    }
}
