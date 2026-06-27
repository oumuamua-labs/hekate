// SPDX-License-Identifier: Apache-2.0
// This file is part of the hekate project.
// Copyright (C) 2026 Andrei Kochergin <andrei@oumuamua.dev>
// Copyright (C) 2026 Oumuamua Labs <info@oumuamua.dev>.
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

use crate::mutation::MutationKind;
use crate::target::Target;

/// Empty `targets` / `kinds` / `include_*`
/// mean "no restriction"; populated lists
/// are filters.
#[derive(Clone, Debug)]
pub struct ScribbleConfig {
    cases: u32,
    targets: Vec<Target>,
    kinds: Vec<MutationKind>,
    include_cols: Vec<usize>,
    exclude_cols: Vec<usize>,
    include_rows: Vec<usize>,
    exclude_rows: Vec<usize>,
}

impl Default for ScribbleConfig {
    fn default() -> Self {
        Self {
            cases: 256,
            targets: Vec::new(),
            kinds: Vec::new(),
            include_cols: Vec::new(),
            exclude_cols: Vec::new(),
            include_rows: Vec::new(),
            exclude_rows: Vec::new(),
        }
    }
}

impl ScribbleConfig {
    pub fn cases(mut self, cases: u32) -> Self {
        self.cases = cases;
        self
    }

    pub fn target(mut self, target: Target) -> Self {
        self.targets = vec![target];
        self
    }

    pub fn targets<I: IntoIterator<Item = Target>>(mut self, targets: I) -> Self {
        self.targets = targets.into_iter().collect();
        self
    }

    pub fn mutations<I: IntoIterator<Item = MutationKind>>(mut self, kinds: I) -> Self {
        self.kinds = kinds.into_iter().collect();
        self
    }

    pub fn include_cols<I: IntoIterator<Item = usize>>(mut self, cols: I) -> Self {
        self.include_cols = cols.into_iter().collect();
        self
    }

    pub fn exclude_cols<I: IntoIterator<Item = usize>>(mut self, cols: I) -> Self {
        self.exclude_cols = cols.into_iter().collect();
        self
    }

    pub fn include_rows<I: IntoIterator<Item = usize>>(mut self, rows: I) -> Self {
        self.include_rows = rows.into_iter().collect();
        self
    }

    pub fn exclude_rows<I: IntoIterator<Item = usize>>(mut self, rows: I) -> Self {
        self.exclude_rows = rows.into_iter().collect();
        self
    }

    pub fn case_count(&self) -> u32 {
        self.cases
    }

    pub fn is_target_allowed(&self, target: Target) -> bool {
        self.targets.is_empty() || self.targets.contains(&target)
    }

    pub fn is_kind_allowed(&self, kind: MutationKind) -> bool {
        self.kinds.is_empty() || self.kinds.contains(&kind)
    }

    pub fn is_col_allowed(&self, col: usize) -> bool {
        if !self.include_cols.is_empty() && !self.include_cols.contains(&col) {
            return false;
        }

        !self.exclude_cols.contains(&col)
    }

    pub fn is_row_allowed(&self, row: usize) -> bool {
        if !self.include_rows.is_empty() && !self.include_rows.contains(&row) {
            return false;
        }

        !self.exclude_rows.contains(&row)
    }
}
