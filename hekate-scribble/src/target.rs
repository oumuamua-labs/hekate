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

use hekate_core::trace::ColumnTrace;
use hekate_math::TowerField;
use hekate_program::ProgramWitness;

/// Selects which trace
/// a mutation operates on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Target {
    Main,
    Chiplet(usize),
}

impl Target {
    /// Borrows the underlying
    /// [`ColumnTrace`] this target selects.
    ///
    /// # Panics
    ///
    /// Panics if `Chiplet(idx)` is out of
    /// range for the witness's chiplet table.
    pub fn resolve_mut<'w, F: TowerField>(
        &self,
        witness: &'w mut ProgramWitness<F, ColumnTrace>,
    ) -> &'w mut ColumnTrace {
        match self {
            Self::Main => &mut witness.trace,
            Self::Chiplet(idx) => {
                let n = witness.chiplet_traces.len();
                witness.chiplet_traces.get_mut(*idx).unwrap_or_else(|| {
                    panic!("Target::Chiplet({idx}) out of bounds (witness has {n} chiplet traces)")
                })
            }
        }
    }

    /// Immutable counterpart to
    /// [`resolve_mut`](Self::resolve_mut).
    ///
    /// # Panics
    ///
    /// Panics if `Chiplet(idx)` is out of
    /// range for the witness's chiplet table.
    pub fn resolve<'w, F: TowerField>(
        &self,
        witness: &'w ProgramWitness<F, ColumnTrace>,
    ) -> &'w ColumnTrace {
        match self {
            Self::Main => &witness.trace,
            Self::Chiplet(idx) => {
                let n = witness.chiplet_traces.len();
                witness.chiplet_traces.get(*idx).unwrap_or_else(|| {
                    panic!("Target::Chiplet({idx}) out of bounds (witness has {n} chiplet traces)")
                })
            }
        }
    }
}
