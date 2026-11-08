// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::BTreeSet;

use hekate_math::Block256;
use hekate_program::Program;
use hekate_sdk::preflight::{PreflightReport, TableId};
use hekate_verifier::rejection::{OuterStage, Rejection, Table, ZerocheckStage};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Check {
    Sumcheck,
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

    pub fn rejection(
        self,
        air: &impl Program<Block256>,
        report: &PreflightReport<Block256>,
        zero_knowledge: bool,
    ) -> Rejection {
        if zero_knowledge {
            return Rejection::Outer(OuterStage::Linear);
        }

        match self.first_failing_table() {
            Some(table) => Rejection::Zerocheck {
                table: match table {
                    TableId::Main => Table::Main,
                    TableId::Chiplet(index) => Table::Chiplet(index),
                },
                stage: self.stage(air, report, table),
            },
            None => Rejection::BusSum,
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

    fn stage(
        self,
        air: &impl Program<Block256>,
        report: &PreflightReport<Block256>,
        table: TableId,
    ) -> ZerocheckStage {
        let pinned = self.pinned_tables().contains(&table);

        let check = match (self.first_check, pinned) {
            (Some(check), _) => check,
            (None, false) => Check::Sumcheck,
            (None, true) => panic!("fault on pinned table {table} names its first check"),
        };

        match check {
            Check::Sumcheck => ZerocheckStage::Sumcheck,
            Check::FixedColumn => {
                ZerocheckStage::FixedColumn(first_violated_pin(air, report, table))
            }
        }
    }
}

fn first_violated_pin(
    air: &impl Program<Block256>,
    report: &PreflightReport<Block256>,
    table: TableId,
) -> usize {
    let violated: BTreeSet<usize> = report
        .fixed_column_violations
        .iter()
        .filter(|v| v.table == table)
        .map(|v| v.col_idx)
        .collect();

    let pins = match table {
        TableId::Main => air.fixed_columns(),
        TableId::Chiplet(index) => air.chiplet_defs().unwrap()[index].pins().to_vec(),
    };

    pins.iter()
        .map(|pin| pin.col_idx)
        .find(|col| violated.contains(col))
        .unwrap_or_else(|| panic!("no pin of table {table} is violated"))
}
