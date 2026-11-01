// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec;
use alloc::vec::Vec;
use hekate_core::config::{self, Config};
use hekate_core::errors::{self, Error};
use hekate_core::fold::{CodeField, FoldLayout, PolyKind, PolyShape, noise_cells_of};
use hekate_core::tensor::TensorProduct;
use hekate_core::trace::ColumnType;
use hekate_math::{Flat, HardwareField};

use crate::expander::RingSwitchPlan;

/// Per-unit claim weights `η^σ(u)·eq(ρ_k, b(u))`
/// and the next-row factor `η^Σ`.
pub struct UnitWeights<F> {
    /// One weight per unit of a table, in plan order.
    pub units: Vec<Flat<F>>,

    /// `η^Σ`, scaling every next-row claim.
    pub shift: Flat<F>,
}

#[derive(Clone, Copy)]
struct Placement {
    poly: usize,
    block: usize,
    sigma: usize,
}

/// One table's units placed in its trace polynomial
/// and, with a bus, its helper polynomial.
pub struct PoolTable {
    /// Units, blinds and `h` columns of the table.
    pub plan: RingSwitchPlan,

    /// `n_i`, the table's row variables.
    pub num_vars: usize,

    /// The trace polynomial's index in the layout.
    pub trace: usize,

    /// The helper polynomial's index, if the table has `h`.
    pub helper: Option<usize>,

    units: Vec<Option<Placement>>,
}

/// Every table's committed polynomials in one fold:
/// the layout, each unit's placement and role.
pub struct FoldPool {
    /// Geometry of the fold over every polynomial.
    layout: FoldLayout,

    /// Tables in pool order: chiplets, then main.
    tables: Vec<PoolTable>,

    /// Leading noise blocks of each data polynomial.
    noise_blocks: Vec<usize>,

    /// `Σ`, the roles over the pool.
    num_sigma: usize,

    /// `θ̄`, the block challenges drawn.
    rho_vars: usize,
}

impl FoldPool {
    /// Lays out every table's polynomials in one fold,
    /// in pool order; errors as [`FoldLayout::new`] does.
    pub fn new(tables: &[(&RingSwitchPlan, usize)], config: &Config) -> errors::Result<Self> {
        let log_inv_rate = config.code_rate.log_inv_rate() as usize;

        if let Some(&(_, num_vars)) = tables
            .iter()
            .find(|&&(_, num_vars)| num_vars + log_inv_rate >= usize::BITS as usize)
        {
            return Err(config::Error::PoolAboveWord {
                num_vars,
                log_inv_rate,
            }
            .into());
        }

        let plans: Vec<RingSwitchPlan> = tables
            .iter()
            .map(|&(plan, _)| match config.zero_knowledge {
                true => plan.pooled(),
                false => Ok(plan.clone()),
            })
            .collect::<errors::Result<_>>()?;

        let mut polys: Vec<(PolyKind, usize, usize, usize, CodeField)> = plans
            .iter()
            .zip(tables)
            .map(|(plan, &(_, num_vars))| {
                (
                    PolyKind::Trace,
                    num_vars,
                    plan.trace_blocks(),
                    plan.blind_cols,
                    data_cells(plan),
                )
            })
            .collect();

        let mut helpers = Vec::with_capacity(tables.len());
        for (plan, &(_, num_vars)) in plans.iter().zip(tables) {
            let helper = match plan.h_cols {
                0 => None,
                h_cols => {
                    polys.push((
                        PolyKind::Helper,
                        num_vars,
                        h_cols + plan.h_blinds,
                        plan.h_blinds,
                        CodeField::B128,
                    ));

                    Some(polys.len() - 1)
                }
            };

            helpers.push(helper);
        }

        let shapes_with = |noise: &[usize]| -> Vec<PolyShape> {
            polys
                .iter()
                .zip(noise)
                .map(
                    |(&(kind, row_vars, blocks, blinds, cells), &noise)| PolyShape {
                        kind,
                        row_vars,
                        block_vars: ceil_log2(noise + blocks),
                        uniform_blocks: noise + blinds,
                        cells,
                    },
                )
                .collect()
        };

        let mut noise = vec![0; polys.len()];
        let shapes = loop {
            let shapes = shapes_with(&noise);

            let grown: Vec<usize> = noise_cells_of(config, &shapes)?
                .into_iter()
                .zip(&polys)
                .zip(&noise)
                .map(|((cells, &(_, row_vars, _, blinds, _)), &noise)| {
                    noise.max(cells.div_ceil(1 << row_vars).saturating_sub(blinds))
                })
                .collect();

            if grown == noise {
                break shapes;
            }

            noise = grown;
        };

        let layout = FoldLayout::new(config, &shapes)?;

        let mut poly_roles = vec![0usize; shapes.len()];
        let mut placed = Vec::with_capacity(tables.len());

        for (i, plan) in plans.iter().enumerate() {
            let helper_noise = helpers[i].map_or(0, |h| noise[h]);

            let mut trace_fill = vec![0usize; noise[i] + plan.trace_blocks()];
            let mut helper_fill = vec![0usize; helper_noise + plan.h_cols + plan.h_blinds];
            let mut units = Vec::with_capacity(plan.num_units);

            for placement in plan.unit_blocks(noise[i], helper_noise) {
                let Some((kind, block)) = placement else {
                    units.push(None);
                    continue;
                };

                let (poly, fill) = match (kind, helpers[i]) {
                    (PolyKind::Trace, _) => (i, &mut trace_fill),
                    (PolyKind::Helper, Some(helper)) => (helper, &mut helper_fill),
                    (PolyKind::Helper, None) => {
                        return Err(Error::Protocol {
                            protocol: "fold_pool",
                            message: "helper unit without a helper polynomial",
                        });
                    }
                    (PolyKind::Mask, _) => {
                        return Err(Error::Protocol {
                            protocol: "fold_pool",
                            message: "table unit placed in a mask",
                        });
                    }
                };

                let role = fill[block];

                fill[block] += 1;
                poly_roles[poly] = poly_roles[poly].max(role + 1);

                units.push(Some((poly, block, role)));
            }

            placed.push(units);
        }

        let mut role_base = vec![0usize; shapes.len() + 1];
        for (k, &roles) in poly_roles.iter().enumerate() {
            role_base[k + 1] = role_base[k] + roles;
        }

        let tables = plans
            .into_iter()
            .zip(tables)
            .zip(placed)
            .zip(helpers)
            .enumerate()
            .map(|(i, (((plan, &(_, num_vars)), units), helper))| PoolTable {
                plan,
                num_vars,
                trace: i,
                helper,
                units: units
                    .into_iter()
                    .map(|unit| {
                        unit.map(|(poly, block, role)| Placement {
                            poly,
                            block,
                            sigma: role_base[poly] + role,
                        })
                    })
                    .collect(),
            })
            .collect();

        let rho_vars = layout
            .polys()
            .iter()
            .map(|p| p.vars - p.shape.row_vars)
            .max()
            .unwrap_or(0);

        Ok(Self {
            layout,
            tables,
            noise_blocks: noise,
            num_sigma: role_base[shapes.len()],
            rho_vars,
        })
    }

    /// Returns the fold's geometry over every polynomial.
    pub fn layout(&self) -> &FoldLayout {
        &self.layout
    }

    /// Returns the pooled tables: chiplets, then main.
    pub fn tables(&self) -> &[PoolTable] {
        &self.tables
    }

    /// Returns the noise blocks leading polynomial `poly`, none for a mask.
    pub fn noise_blocks(&self, poly: usize) -> usize {
        self.noise_blocks.get(poly).copied().unwrap_or(0)
    }

    /// Returns `Σ`, the count of unit roles across the pool.
    pub fn num_sigma(&self) -> usize {
        self.num_sigma
    }

    /// Returns `θ̄`, the count of block challenges the eval draws.
    pub fn rho_vars(&self) -> usize {
        self.rho_vars
    }

    /// Returns whether some table carries a unit over a proper subfield.
    pub fn has_subfield_units(&self) -> bool {
        self.tables.iter().any(|t| t.plan.has_subfield_units())
    }

    /// Returns `θ'_k`, polynomial `poly`'s block variables after padding.
    pub fn block_vars(&self, poly: usize) -> usize {
        let fold = &self.layout.polys()[poly];

        fold.vars - fold.shape.row_vars
    }

    /// Computes table `table`'s per-unit claim weights and next-row factor;
    /// errors unless `rho` holds the pool's `θ̄` block challenges.
    pub fn unit_weights<F: HardwareField>(
        &self,
        table: usize,
        eta: Flat<F>,
        rho: &[Flat<F>],
    ) -> errors::Result<UnitWeights<F>> {
        if rho.len() != self.rho_vars {
            return Err(Error::Protocol {
                protocol: "fold_pool",
                message: "block challenge count does not match the pool",
            });
        }

        let one = Flat::from_raw(F::ONE);

        let mut pows = Vec::with_capacity(self.num_sigma + 1);
        let mut pow = one;

        for _ in 0..=self.num_sigma {
            pows.push(pow);
            pow *= eta;
        }

        let t = &self.tables[table];

        let eq_of = |poly: usize| TensorProduct::new(rho[..self.block_vars(poly)].to_vec());

        let trace_eq = eq_of(t.trace);
        let helper_eq = t.helper.map(eq_of);

        let units = t
            .units
            .iter()
            .map(|placement| match placement {
                Some(p) => {
                    let eq = match (&helper_eq, p.poly == t.trace) {
                        (Some(helper_eq), false) => helper_eq,
                        _ => &trace_eq,
                    };

                    pows[p.sigma] * eq.evaluate_at_index(p.block)
                }
                None => Flat::from_raw(F::ZERO),
            })
            .collect();

        Ok(UnitWeights {
            units,
            shift: pows[self.num_sigma],
        })
    }

    /// Sums `weights` per block of polynomial `poly`, one vector
    /// per unit field it holds; a block without units weighs zero.
    pub fn block_weights<F: HardwareField>(
        &self,
        table: usize,
        poly: usize,
        weights: &UnitWeights<F>,
    ) -> Vec<(ColumnType, Vec<Flat<F>>)> {
        let t = &self.tables[table];
        let zero = Flat::from_raw(F::ZERO);
        let blocks = 1 << self.block_vars(poly);

        let mut fields: Vec<(ColumnType, Vec<Flat<F>>)> = Vec::new();
        for (u, unit) in t.plan.units.iter().enumerate() {
            let Some(p) = t.units[u].filter(|p| p.poly == poly) else {
                continue;
            };

            let k = match fields.iter().position(|(field, _)| *field == unit.field) {
                Some(k) => k,
                None => {
                    fields.push((unit.field, vec![zero; blocks]));
                    fields.len() - 1
                }
            };

            fields[k].1[p.block] += weights.units[u];
        }

        fields
    }

    /// Evaluates [`Self::block_weights`] of polynomial
    /// `poly` at the block point `r_b`, field by field.
    pub fn block_weights_at<F: HardwareField>(
        &self,
        table: usize,
        poly: usize,
        weights: &UnitWeights<F>,
        r_b: &[Flat<F>],
    ) -> Vec<(ColumnType, Flat<F>)> {
        let t = &self.tables[table];
        let eq = TensorProduct::new(r_b.to_vec());

        let mut fields: Vec<(ColumnType, Flat<F>)> = Vec::new();
        for (u, unit) in t.plan.units.iter().enumerate() {
            let Some(p) = t.units[u].filter(|p| p.poly == poly) else {
                continue;
            };

            let weight = weights.units[u] * eq.evaluate_at_index(p.block);

            match fields.iter_mut().find(|(field, _)| *field == unit.field) {
                Some((_, sum)) => *sum += weight,
                None => fields.push((unit.field, weight)),
            }
        }

        fields
    }
}

/// `⌈log2 n⌉`, 0 at `n = 0`.
fn ceil_log2(n: usize) -> usize {
    n.next_power_of_two().trailing_zeros() as usize
}

/// Returns the narrowest code field holding `plan`'s
/// data columns, its blind columns left out.
fn data_cells(plan: &RingSwitchPlan) -> CodeField {
    plan.phys_rs[..plan.phys_rs.len() - plan.blind_cols]
        .iter()
        .map(|ct| match ct {
            ColumnType::B64 => CodeField::B64,
            ColumnType::B128 => CodeField::B128,
            _ => CodeField::B32,
        })
        .max()
        .unwrap_or(CodeField::B32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expander::VirtualExpander;
    use hekate_core::config::MAX_FOLD_ARITY_LOG;
    use hekate_core::trace::ColumnType;
    use hekate_math::{Block128, TowerField};

    type F = Block128;

    fn xorshift(seed: u128) -> impl FnMut() -> Flat<F> {
        let mut state = seed;

        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;

            Block128(state).to_hardware()
        }
    }

    fn small(zero_knowledge: bool) -> Config {
        Config {
            num_queries: 2,
            fold_arity_log: 1,
            early_stop_vars: 0,
            min_security_bits: 0,
            zero_knowledge,
            ..Config::prod()
        }
    }

    #[test]
    fn block_weights_reproduce_batched_claims() {
        let num_vars = 4;
        let rows = 1 << num_vars;

        let plan = RingSwitchPlan::new(&[ColumnType::B32; 3], None, 1, 2).unwrap();
        let pool = FoldPool::new(&[(&plan, num_vars)], &small(true)).unwrap();
        let table = &pool.tables()[0];

        let trace_noise = pool.noise_blocks(table.trace);
        let helper_noise = pool.noise_blocks(table.helper.unwrap());

        let mut next = xorshift(0x94_0301);
        let mut column = || -> Vec<Flat<F>> { (0..rows).map(|_| next()).collect() };

        let trace: Vec<Vec<Flat<F>>> = (0..plan.phys_rs.len()).map(|_| column()).collect();
        let helper: Vec<Vec<Flat<F>>> =
            (0..plan.h_cols + plan.h_blinds).map(|_| column()).collect();

        let point: Vec<Flat<F>> = (0..num_vars).map(|_| next()).collect();
        let eta = next();
        let rho: Vec<Flat<F>> = (0..pool.rho_vars()).map(|_| next()).collect();
        let weights = pool.unit_weights(0, eta, &rho).unwrap();

        let eq = TensorProduct::new(point);

        let omega = |x: usize| {
            eq.evaluate_at_index(x) + weights.shift * eq.evaluate_at_index((x + rows - 1) % rows)
        };

        let zero = Flat::from_raw(F::ZERO);

        let claim = |f: &[Flat<F>]| (0..rows).fold(zero, |acc, x| acc + omega(x) * f[x]);

        let placements = table.plan.unit_blocks(trace_noise, helper_noise);

        let mut target = zero;
        for (u, placement) in placements.into_iter().enumerate() {
            let Some((kind, block)) = placement else {
                continue;
            };

            let (columns, noise) = match kind {
                PolyKind::Trace => (&trace, trace_noise),
                PolyKind::Helper => (&helper, helper_noise),
                PolyKind::Mask => unreachable!("a plan places no unit in a mask"),
            };

            target += weights.units[u] * claim(&columns[block - noise]);
        }

        let mut sum = zero;
        for (poly, columns) in [(table.trace, &trace), (table.helper.unwrap(), &helper)] {
            let noise = pool.noise_blocks(poly);

            for (_, field_weights) in pool.block_weights(0, poly, &weights) {
                for (block, &w) in field_weights.iter().enumerate() {
                    match block.checked_sub(noise).and_then(|p| columns.get(p)) {
                        Some(f) => sum += w * claim(f),
                        None => assert_eq!(w, zero, "block {block} holds no unit"),
                    }
                }
            }
        }

        assert_eq!(sum, target);
    }

    #[test]
    fn units_sharing_block_take_distinct_roles() {
        let expander = VirtualExpander::new()
            .pass_through(2, ColumnType::B32)
            .reuse_pass_through(0, 1)
            .build()
            .unwrap();

        let plan = RingSwitchPlan::new(
            &[ColumnType::B32; 2],
            Some(&expander.expansion_entries()),
            0,
            0,
        )
        .unwrap();

        let pool = FoldPool::new(&[(&plan, 3)], &small(false)).unwrap();

        let mut next = xorshift(0x94_0302);

        let eta = next();
        let rho: Vec<Flat<F>> = (0..pool.rho_vars()).map(|_| next()).collect();
        let weights = pool.unit_weights(0, eta, &rho).unwrap();

        assert_eq!(pool.num_sigma(), 2);
        assert_eq!(weights.shift, eta * eta);
        assert_ne!(weights.units[0], weights.units[2]);
        assert_eq!(weights.units[2], eta * weights.units[0]);
    }

    #[test]
    fn roles_run_over_polynomials_and_small_tables_pad() {
        let wide = RingSwitchPlan::new(&[ColumnType::B32; 3], None, 1, 2).unwrap();
        let narrow = RingSwitchPlan::new(&[ColumnType::B32; 1], None, 1, 0).unwrap();

        let pool = FoldPool::new(&[(&wide, 6), (&narrow, 4)], &small(true)).unwrap();
        let [wide_table, narrow_table] = pool.tables() else {
            panic!("two tables");
        };

        assert_eq!((wide_table.trace, narrow_table.trace), (0, 1));
        assert_eq!((wide_table.helper, narrow_table.helper), (Some(2), None));
        assert_eq!(pool.num_sigma(), 3);

        let top = pool.layout().num_vars();
        let narrow_poly = &pool.layout().polys()[narrow_table.trace];

        assert_eq!(top, 6 + 2);
        assert_eq!(narrow_poly.vars, top - 1);
        assert_eq!(pool.block_vars(narrow_table.trace), top - 1 - 4);
        assert_eq!(pool.rho_vars(), pool.block_vars(narrow_table.trace));
    }

    #[test]
    fn block_weights_at_reads_dense_weights_at_point() {
        let expander = VirtualExpander::new()
            .expand_bits(1, ColumnType::B32)
            .pass_through(2, ColumnType::B32)
            .build()
            .unwrap();

        let plan = RingSwitchPlan::new(
            &[ColumnType::B32; 3],
            Some(&expander.expansion_entries()),
            1,
            2,
        )
        .unwrap();

        let pool = FoldPool::new(&[(&plan, 4)], &small(true)).unwrap();
        let table = &pool.tables()[0];

        let mut next = xorshift(0x94_0303);

        let eta = next();
        let rho: Vec<Flat<F>> = (0..pool.rho_vars()).map(|_| next()).collect();
        let weights = pool.unit_weights(0, eta, &rho).unwrap();

        for poly in [table.trace, table.helper.unwrap()] {
            let r_b: Vec<Flat<F>> = (0..pool.block_vars(poly)).map(|_| next()).collect();
            let eq = TensorProduct::new(r_b.clone());

            let at_point = |dense: &[Flat<F>]| {
                dense
                    .iter()
                    .enumerate()
                    .fold(Flat::from_raw(F::ZERO), |acc, (b, &w)| {
                        acc + w * eq.evaluate_at_index(b)
                    })
            };

            let dense: Vec<(ColumnType, Flat<F>)> = pool
                .block_weights(0, poly, &weights)
                .iter()
                .map(|(field, w)| (*field, at_point(w)))
                .collect();

            assert_eq!(pool.block_weights_at(0, poly, &weights, &r_b), dense);
        }

        assert!(plan.has_subfield_units());
    }

    #[test]
    fn blinds_carry_noise_of_tall_tables() {
        let plan = RingSwitchPlan::new(&[ColumnType::B32; 3], None, 1, 2).unwrap();
        let pool = FoldPool::new(&[(&plan, 12)], &Config::prod()).unwrap();
        let table = &pool.tables()[0];
        let helper = table.helper.unwrap();

        assert_eq!(pool.noise_blocks(table.trace), 0);
        assert_eq!(pool.noise_blocks(helper), 0);
        assert_eq!(pool.block_vars(table.trace), 1);
        assert_eq!(pool.block_vars(helper), 2);
    }

    #[test]
    fn short_table_noise_fills_blocks_past_blinds() {
        let row_vars = 8;
        let config = Config::prod();

        let plan = RingSwitchPlan::new(&[ColumnType::B32; 3], None, 1, 2).unwrap();
        let pool = FoldPool::new(&[(&plan, row_vars)], &config).unwrap();
        let table = &pool.tables()[0];

        for poly in [table.trace, table.helper.unwrap()] {
            let leaf = 1 << pool.layout().polys()[poly].leaf_vars;
            let uniform = ((config.num_queries + 2) * leaf).div_ceil(1 << row_vars);

            assert_eq!(pool.noise_blocks(poly), uniform - 1);
        }
    }

    #[test]
    fn table_past_word_rejected() {
        let plan = RingSwitchPlan::new(&[ColumnType::B32; 3], None, 1, 2).unwrap();
        let num_vars = usize::BITS as usize - 1;

        assert!(matches!(
            FoldPool::new(&[(&plan, num_vars)], &Config::prod()),
            Err(Error::Config(config::Error::PoolAboveWord { .. }))
        ));
    }

    #[test]
    fn arity_outside_cap_rejected_before_noise_layout() {
        let plan = RingSwitchPlan::new(&[ColumnType::B32; 3], None, 1, 2).unwrap();

        for fold_arity_log in [0, MAX_FOLD_ARITY_LOG + 1] {
            let pool = FoldPool::new(
                &[(&plan, 8)],
                &Config {
                    fold_arity_log,
                    ..Config::prod()
                },
            );

            assert!(matches!(
                pool,
                Err(Error::Config(config::Error::FoldArity { .. }))
            ));
        }
    }
}
