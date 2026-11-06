// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::BTreeSet;

use hekate_core::errors;
use hekate_math::Block256;
use hekate_sdk::preflight::{PreflightReport, TableId};

use crate::site::{BUS_SUM, CHIPLET_ZEROCHECK, FIXED_COLUMN, OUTER_LINEAR, ZEROCHECK};
use crate::verdict::{rejected_at, rejected_with};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Check {
    Zerocheck,
    FixedColumn,
}

#[derive(Clone, Copy)]
pub struct Fault<'a> {
    pub constraint_tables: &'a [TableId],
    pub constraint_rows: &'a [(TableId, usize, Option<&'a str>)],
    pub constraint_labels: &'a [(TableId, &'a str)],
    pub boundary_columns: &'a [(TableId, usize)],
    pub pin_columns: &'a [(TableId, usize)],
    pub pin_cells: &'a [(TableId, usize, usize)],
    pub pin_tables: &'a [TableId],
    pub buses: &'a [&'a str],
    pub first_check: Option<Check>,
}

impl<'a> Fault<'a> {
    pub const NONE: Fault<'a> = Fault {
        constraint_tables: &[],
        constraint_rows: &[],
        constraint_labels: &[],
        boundary_columns: &[],
        pin_columns: &[],
        pin_cells: &[],
        pin_tables: &[],
        buses: &[],
        first_check: None,
    };

    pub fn assert_sole(self, report: &PreflightReport<Block256>) {
        let tables: BTreeSet<TableId> = report
            .constraint_violations
            .iter()
            .map(|v| v.table)
            .collect();
        let boundaries: BTreeSet<(TableId, usize)> = report
            .boundary_violations
            .iter()
            .map(|v| (v.table, v.col_idx))
            .collect();
        let pins: BTreeSet<(TableId, usize)> = report
            .fixed_column_violations
            .iter()
            .map(|v| (v.table, v.col_idx))
            .filter(|(table, _)| !self.pin_tables.contains(table))
            .collect();
        let pin_tables: BTreeSet<TableId> = report
            .fixed_column_violations
            .iter()
            .map(|v| v.table)
            .collect();
        let buses: BTreeSet<&str> = report
            .bus_diagnostics
            .iter()
            .map(|d| d.bus_id.as_str())
            .collect();

        assert_eq!(tables, self.failing_constraint_tables(), "{report}");

        if !self.constraint_rows.is_empty() {
            let mut rows: Vec<(TableId, usize, Option<&str>)> = report
                .constraint_violations
                .iter()
                .map(|v| (v.table, v.row_idx, v.label))
                .collect();

            let mut expected = self.constraint_rows.to_vec();

            rows.sort();
            expected.sort();

            assert_eq!(rows, expected, "{report}");
        }

        if !self.constraint_labels.is_empty() {
            let labels: BTreeSet<(TableId, Option<&str>)> = report
                .constraint_violations
                .iter()
                .map(|v| (v.table, v.label))
                .collect();

            assert_eq!(
                labels,
                self.constraint_labels
                    .iter()
                    .map(|&(table, label)| (table, Some(label)))
                    .collect(),
                "{report}"
            );
        }

        assert_eq!(
            boundaries,
            self.boundary_columns.iter().copied().collect(),
            "{report}"
        );
        assert_eq!(pins, self.pinned_columns(), "{report}");
        assert_eq!(pin_tables, self.pinned_tables(), "{report}");

        if !self.pin_cells.is_empty() {
            let cells: BTreeSet<(TableId, usize, usize)> = report
                .fixed_column_violations
                .iter()
                .map(|v| (v.table, v.col_idx, v.row_idx))
                .collect();

            assert_eq!(cells, self.pin_cells.iter().copied().collect(), "{report}");
        }
        assert_eq!(buses, self.buses.iter().copied().collect(), "{report}");
    }

    pub fn rejects(self, outcome: &errors::Result<bool>, log: &str, zero_knowledge: bool) -> bool {
        if zero_knowledge {
            return rejected_at(outcome, log, OUTER_LINEAR);
        }

        match self.first_failing_table() {
            Some(TableId::Main) => rejected_at(outcome, log, self.site(TableId::Main)),
            Some(TableId::Chiplet(index)) => {
                let field = format!("chiplet_idx={index}");

                rejected_with(outcome, CHIPLET_ZEROCHECK)
                    && log.split_whitespace().any(|token| token == field)
                    && log.contains(self.site(TableId::Chiplet(index)))
            }
            None => rejected_with(outcome, BUS_SUM),
        }
    }

    fn failing_constraint_tables(self) -> BTreeSet<TableId> {
        self.constraint_tables
            .iter()
            .copied()
            .chain(self.constraint_rows.iter().map(|&(table, _, _)| table))
            .chain(self.constraint_labels.iter().map(|&(table, _)| table))
            .collect()
    }

    fn pinned_columns(self) -> BTreeSet<(TableId, usize)> {
        self.pin_columns
            .iter()
            .copied()
            .chain(self.pin_cells.iter().map(|&(table, col, _)| (table, col)))
            .collect()
    }

    fn pinned_tables(self) -> BTreeSet<TableId> {
        self.pin_tables
            .iter()
            .copied()
            .chain(self.pinned_columns().into_iter().map(|(table, _)| table))
            .collect()
    }

    fn first_failing_table(self) -> Option<TableId> {
        let tables: BTreeSet<TableId> = self
            .failing_constraint_tables()
            .into_iter()
            .chain(self.boundary_columns.iter().map(|&(table, _)| table))
            .chain(self.pinned_tables())
            .collect();

        tables
            .iter()
            .copied()
            .find(|&table| table != TableId::Main)
            .or(tables.first().copied())
    }

    fn site(self, table: TableId) -> &'static str {
        let pinned = self.pinned_tables().contains(&table);

        let check = match (self.first_check, pinned) {
            (Some(check), _) => check,
            (None, false) => Check::Zerocheck,
            (None, true) => panic!("fault on pinned table {table} names its first check"),
        };

        match check {
            Check::Zerocheck => ZEROCHECK,
            Check::FixedColumn => FIXED_COLUMN,
        }
    }
}
