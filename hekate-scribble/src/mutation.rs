use crate::target::Target;

/// A trace tamper. `Clone + Debug` are
/// required for proptest shrinking output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mutation {
    /// XOR `mask` (truncated to column width)
    /// into a single cell.
    BitFlip {
        target: Target,
        col: usize,
        row: usize,
        mask: u128,
    },

    /// Inject a value the AIR packing
    /// accepts but range-checks must reject.
    OutOfBounds {
        target: Target,
        col: usize,
        row: usize,
        value: u128,
    },

    /// Toggle a `Bit` cell. Targets
    /// must be `Bit`-typed columns.
    FlipSelector {
        target: Target,
        col: usize,
        row: usize,
    },

    /// Swap every column between two rows.
    SwapRows {
        target: Target,
        row_a: usize,
        row_b: usize,
    },

    /// Dispatch-swap primitive:
    /// rearrange data while leaving the columns
    /// outside `cols` (selectors, RAM bindings)
    /// in place.
    SwapColumns {
        target: Target,
        cols: Vec<usize>,
        row_a: usize,
        row_b: usize,
    },

    /// Char-2 duplication:
    /// copy every column of `src_row` onto `dst_row`.
    DuplicateRow {
        target: Target,
        src_row: usize,
        dst_row: usize,
    },
    CopyColumns {
        target: Target,
        cols: Vec<usize>,
        src_row: usize,
        dst_row: usize,
    },

    /// Overwrite every row of `col` with the
    /// same `value` (truncated to column width).
    /// Catches "column should be non-trivial
    /// somewhere" gaps the row-local AIR misses.
    ColumnUniformWrite {
        target: Target,
        col: usize,
        value: u128,
    },

    /// Zero every cell in the cross product
    /// of `rows` and `cols`. Catches padding-
    /// block forgeries and trace-tail filler.
    RowSegmentZero {
        target: Target,
        rows: Vec<usize>,
        cols: Vec<usize>,
    },

    /// Replace `col` with `base + i * step`
    /// at row `i` (truncated to column width).
    /// Catches CLK rewinds, address-sorted-
    /// permutation forgeries, monotonic-counter
    /// bypasses.
    MonotonicReplace {
        target: Target,
        col: usize,
        base: u128,
        step: u128,
    },

    /// Apply mutations as one tamper.
    /// Enables coordinated cross-trace
    /// (chiplet + main) attacks the
    /// per-table checks alone cannot catch.
    Compound(Vec<Mutation>),
}

/// Layer-1 (proptest-discoverable) variants.
/// `SwapColumns`, `CopyColumns`, and `Compound`
/// are excluded, their search space defeats
/// random discovery.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MutationKind {
    BitFlip,
    OutOfBounds,
    FlipSelector,
    SwapRows,
    DuplicateRow,
    ColumnUniformWrite,
    RowSegmentZero,
    MonotonicReplace,
}

impl Mutation {
    /// `Some` for Layer-1 variants;
    /// `None` for `SwapColumns`, `CopyColumns`,
    /// and `Compound` (Layer 2, hand-crafted only).
    pub fn kind(&self) -> Option<MutationKind> {
        match self {
            Mutation::BitFlip { .. } => Some(MutationKind::BitFlip),
            Mutation::OutOfBounds { .. } => Some(MutationKind::OutOfBounds),
            Mutation::FlipSelector { .. } => Some(MutationKind::FlipSelector),
            Mutation::SwapRows { .. } => Some(MutationKind::SwapRows),
            Mutation::DuplicateRow { .. } => Some(MutationKind::DuplicateRow),
            Mutation::ColumnUniformWrite { .. } => Some(MutationKind::ColumnUniformWrite),
            Mutation::RowSegmentZero { .. } => Some(MutationKind::RowSegmentZero),
            Mutation::MonotonicReplace { .. } => Some(MutationKind::MonotonicReplace),
            Mutation::SwapColumns { .. } | Mutation::CopyColumns { .. } | Mutation::Compound(_) => {
                None
            }
        }
    }
}
