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
