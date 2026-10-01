// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec;
use alloc::vec::Vec;
use core::iter::repeat_n;
use hekate_core::config::{Config, FoldShape, MIN_PRODUCTION_BITS};
use hekate_core::errors::Error;
use hekate_core::poly::PolyVariant;
use hekate_core::trace::{ColumnType, Trace, TraceColumn, TraceCompatibleField};
use hekate_core::utils::{compute_split_vars, support_floor_vars};
use hekate_math::{
    Bit, Block8, Block16, Block32, Block64, Block128, Flat, HardwareField, TowerField,
};

pub const RING_BLIND_BITS: usize = 128;

/// Serializable expansion step descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpansionEntry {
    ExpandBits {
        count: usize,
        storage: ColumnType,
    },
    PassThrough {
        count: usize,
        storage: ColumnType,
    },
    ControlBits {
        count: usize,
    },
    ReusePassThrough {
        phy_col_start: usize,
        count: usize,
        storage: ColumnType,
    },
    ReuseExpandBits {
        phy_col_start: usize,
        count: usize,
        storage: ColumnType,
    },
}

/// Physical-to-virtual column mapping rule.
#[derive(Clone, Copy, Debug)]
enum EntryKind {
    /// N physical columns to N ×
    /// bit_width virtual Bit columns.
    ExpandBits { count: usize, storage: ColumnType },

    /// N physical columns to N virtual
    /// columns of the same type.
    PassThrough { count: usize, storage: ColumnType },

    /// N physical Bit columns
    /// to N virtual Bit columns.
    ControlBits { count: usize },
}

impl EntryKind {
    fn count(&self) -> usize {
        match self {
            Self::ExpandBits { count, .. }
            | Self::PassThrough { count, .. }
            | Self::ControlBits { count } => *count,
        }
    }

    fn storage(&self) -> ColumnType {
        match self {
            Self::ExpandBits { storage, .. } | Self::PassThrough { storage, .. } => *storage,
            Self::ControlBits { .. } => ColumnType::Bit,
        }
    }
}

/// Pre-computed expansion entry
/// with frozen byte/column offsets.
#[derive(Clone, Copy, Debug)]
struct CompiledEntry {
    /// Physical column index,
    /// relative to `phy_start_idx`.
    phy_col_start: usize,

    /// Byte offset in the committed row.
    byte_offset: usize,
    kind: EntryKind,

    /// True if this entry reuses physical
    /// columns declared by a prior entry.
    reuse: bool,
}

/// Declarative physical->virtual
/// column expander for chiplets.
///
/// Built once per chiplet, generates
/// `virtual_layout()`, `parse_row()`,
/// and `expand_variants()` from the
/// same packing specification.
#[derive(Clone, Debug)]
pub struct VirtualExpander {
    entries: Vec<CompiledEntry>,
    num_virtual: usize,
    num_physical: usize,
    physical_row_bytes: usize,
    virtual_layout: Vec<ColumnType>,
    error: Option<Error>,
}

impl VirtualExpander {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            num_virtual: 0,
            num_physical: 0,
            physical_row_bytes: 0,
            virtual_layout: Vec::new(),
            error: None,
        }
    }

    /// Finalize the builder. Returns `Err` if any
    /// builder step recorded a validation error.
    pub fn build(self) -> Result<Self, Error> {
        match self.error {
            Some(e) => Err(e),
            None => Ok(self),
        }
    }

    /// N physical columns of `storage` type
    /// to N × bit_width virtual Bit columns.
    pub fn expand_bits(mut self, count: usize, storage: ColumnType) -> Self {
        if self.error.is_some() {
            return self;
        }

        let bits_per = match expand_bit_width(storage) {
            Ok(v) => v,
            Err(e) => {
                self.error = Some(e);
                return self;
            }
        };

        let byte_offset = self.physical_row_bytes;
        let phy_col_start = self.num_physical;

        self.entries.push(CompiledEntry {
            phy_col_start,
            byte_offset,
            kind: EntryKind::ExpandBits { count, storage },
            reuse: false,
        });

        let virt_count = count * bits_per;
        self.virtual_layout
            .extend(repeat_n(ColumnType::Bit, virt_count));

        self.num_virtual += virt_count;
        self.num_physical += count;
        self.physical_row_bytes += count * storage.byte_size();

        self
    }

    /// N physical columns pass through
    /// 1:1 as virtual columns.
    pub fn pass_through(mut self, count: usize, storage: ColumnType) -> Self {
        let byte_offset = self.physical_row_bytes;
        let phy_col_start = self.num_physical;

        self.entries.push(CompiledEntry {
            phy_col_start,
            byte_offset,
            kind: EntryKind::PassThrough { count, storage },
            reuse: false,
        });

        self.virtual_layout.extend(repeat_n(storage, count));

        self.num_virtual += count;
        self.num_physical += count;
        self.physical_row_bytes += count * storage.byte_size();

        self
    }

    /// N physical Bit columns pass through 1:1.
    pub fn control_bits(mut self, count: usize) -> Self {
        let byte_offset = self.physical_row_bytes;
        let phy_col_start = self.num_physical;

        self.entries.push(CompiledEntry {
            phy_col_start,
            byte_offset,
            kind: EntryKind::ControlBits { count },
            reuse: false,
        });

        self.virtual_layout.extend(repeat_n(ColumnType::Bit, count));

        self.num_virtual += count;
        self.num_physical += count;
        self.physical_row_bytes += count;

        self
    }

    /// Emit pass-through for columns already
    /// declared by a prior fresh entry.
    /// Does not advance the physical cursor.
    pub fn reuse_pass_through(mut self, phy_col_start: usize, count: usize) -> Self {
        if self.error.is_some() {
            return self;
        }

        if phy_col_start + count > self.num_physical {
            self.error = Some(Error::Protocol {
                protocol: "virtual_expand",
                message: "reuse_pass_through: range exceeds declared physical columns",
            });
            return self;
        }

        let (byte_offset, storage) = match self.find_phy_source(phy_col_start, count) {
            Ok(v) => v,
            Err(e) => {
                self.error = Some(e);
                return self;
            }
        };

        self.entries.push(CompiledEntry {
            phy_col_start,
            byte_offset,
            kind: EntryKind::PassThrough { count, storage },
            reuse: true,
        });

        self.virtual_layout.extend(repeat_n(storage, count));

        self.num_virtual += count;

        self
    }

    /// Emit bit-expansion for columns already
    /// declared by a prior fresh entry.
    /// Does not advance the physical cursor.
    pub fn reuse_expand_bits(mut self, phy_col_start: usize, count: usize) -> Self {
        if self.error.is_some() {
            return self;
        }

        if phy_col_start + count > self.num_physical {
            self.error = Some(Error::Protocol {
                protocol: "virtual_expand",
                message: "reuse_expand_bits: range exceeds declared physical columns",
            });
            return self;
        }

        let (byte_offset, storage) = match self.find_phy_source(phy_col_start, count) {
            Ok(v) => v,
            Err(e) => {
                self.error = Some(e);
                return self;
            }
        };

        let bits_per = match expand_bit_width(storage) {
            Ok(v) => v,
            Err(e) => {
                self.error = Some(e);
                return self;
            }
        };

        self.entries.push(CompiledEntry {
            phy_col_start,
            byte_offset,
            kind: EntryKind::ExpandBits { count, storage },
            reuse: true,
        });

        let virt_count = count * bits_per;
        self.virtual_layout
            .extend(repeat_n(ColumnType::Bit, virt_count));

        self.num_virtual += virt_count;

        self
    }

    /// Append another expander's entries after this
    /// one's, shifting its physical references onto
    /// this expander's coordinate space.
    pub fn append(mut self, other: &VirtualExpander) -> Self {
        if self.error.is_some() {
            return self;
        }

        if let Some(e) = &other.error {
            self.error = Some(*e);
            return self;
        }

        let phy_shift = self.num_physical;
        let byte_shift = self.physical_row_bytes;

        for entry in &other.entries {
            self.entries.push(CompiledEntry {
                phy_col_start: entry.phy_col_start + phy_shift,
                byte_offset: entry.byte_offset + byte_shift,
                kind: entry.kind,
                reuse: entry.reuse,
            });
        }

        self.virtual_layout.extend_from_slice(&other.virtual_layout);

        self.num_virtual += other.num_virtual;
        self.num_physical += other.num_physical;
        self.physical_row_bytes += other.physical_row_bytes;

        self
    }

    #[inline]
    pub fn num_virtual_columns(&self) -> usize {
        self.num_virtual
    }

    #[inline]
    pub fn num_physical_columns(&self) -> usize {
        self.num_physical
    }

    #[inline]
    pub fn physical_row_bytes(&self) -> usize {
        self.physical_row_bytes
    }

    #[inline]
    pub fn virtual_layout(&self) -> &[ColumnType] {
        &self.virtual_layout
    }

    /// Verifier-side:
    /// parse committed physical row bytes
    /// into virtual field elements.
    pub fn parse_row<F: TraceCompatibleField>(
        &self,
        bytes: &[u8],
        res: &mut Vec<Flat<F>>,
    ) -> Result<(), Error> {
        if bytes.len() != self.physical_row_bytes {
            return Err(Error::Protocol {
                protocol: "virtual_expand",
                message: "parse_row: byte slice length mismatch",
            });
        }

        res.reserve(self.num_virtual);

        for entry in &self.entries {
            let off = entry.byte_offset;
            match entry.kind {
                EntryKind::ExpandBits { count, storage } => {
                    let bsz = storage.byte_size();
                    let bits = expand_bit_width(storage)?;

                    for i in 0..count {
                        let start = off + i * bsz;
                        for bit_idx in 0..bits {
                            let bit = parse_tower_bit(storage, &bytes[start..start + bsz], bit_idx);
                            res.push(Flat::from_raw(F::from(Bit::from(bit))));
                        }
                    }
                }
                EntryKind::PassThrough { count, storage } => {
                    let bsz = storage.byte_size();
                    for i in 0..count {
                        let start = off + i * bsz;
                        res.push(storage.parse_from_bytes(&bytes[start..start + bsz]));
                    }
                }
                EntryKind::ControlBits { count } => {
                    for i in 0..count {
                        res.push(Flat::from_raw(F::from(Bit::from(bytes[off + i] & 1))));
                    }
                }
            }
        }

        Ok(())
    }

    /// Prover-side:
    /// expand physical `ColumnTrace`
    /// into virtual `PolyVariant`s.
    pub fn expand_variants<'a, F, T: Trace + ?Sized>(
        &self,
        trace: &'a T,
        phy_start_idx: usize,
    ) -> Result<Vec<PolyVariant<'a, F>>, Error>
    where
        F: TraceCompatibleField + 'static,
    {
        let columns = trace.columns();

        let mut variants = Vec::with_capacity(self.num_virtual);
        for entry in &self.entries {
            let base = phy_start_idx + entry.phy_col_start;
            match entry.kind {
                EntryKind::ExpandBits { count, storage } => {
                    let bits = expand_bit_width(storage)?;
                    for i in 0..count {
                        let col = columns.get(base + i).ok_or(Error::Protocol {
                            protocol: "virtual_expand",
                            message: "missing physical column for ExpandBits",
                        })?;

                        for bit_idx in 0..bits {
                            variants.push(expand_packed_bit(col, storage, bit_idx)?);
                        }
                    }
                }
                EntryKind::PassThrough { count, storage } => {
                    for i in 0..count {
                        let col = columns.get(base + i).ok_or(Error::Protocol {
                            protocol: "virtual_expand",
                            message: "missing physical column for PassThrough",
                        })?;

                        variants.push(expand_pass_through(col, storage)?);
                    }
                }
                EntryKind::ControlBits { count } => {
                    for i in 0..count {
                        let col = columns.get(base + i).ok_or(Error::Protocol {
                            protocol: "virtual_expand",
                            message: "missing physical column for ControlBits",
                        })?;
                        let data = col.as_bit_slice().ok_or(Error::Protocol {
                            protocol: "virtual_expand",
                            message: "control column must be Bit",
                        })?;

                        variants.push(PolyVariant::BitSlice(data));
                    }
                }
            }
        }

        Ok(variants)
    }

    /// Wire-format serialization descriptor.
    pub fn expansion_entries(&self) -> Vec<ExpansionEntry> {
        self.entries
            .iter()
            .map(|e| match (e.kind, e.reuse) {
                (EntryKind::PassThrough { count, storage }, true) => {
                    ExpansionEntry::ReusePassThrough {
                        phy_col_start: e.phy_col_start,
                        count,
                        storage,
                    }
                }
                (EntryKind::ExpandBits { count, storage }, true) => {
                    ExpansionEntry::ReuseExpandBits {
                        phy_col_start: e.phy_col_start,
                        count,
                        storage,
                    }
                }
                (EntryKind::ExpandBits { count, storage }, false) => {
                    ExpansionEntry::ExpandBits { count, storage }
                }
                (EntryKind::PassThrough { count, storage }, false) => {
                    ExpansionEntry::PassThrough { count, storage }
                }
                (EntryKind::ControlBits { count }, _) => ExpansionEntry::ControlBits { count },
            })
            .collect()
    }

    pub(crate) fn whole_column(&self, virt: usize) -> Option<usize> {
        let mut base = 0usize;
        for entry in &self.entries {
            let width = match entry.kind {
                EntryKind::ExpandBits { count, storage } => count * storage.byte_size() * 8,
                EntryKind::PassThrough { count, .. } | EntryKind::ControlBits { count } => count,
            };

            if virt < base + width {
                return match entry.kind {
                    EntryKind::ExpandBits { .. } => None,
                    EntryKind::PassThrough { .. } | EntryKind::ControlBits { .. } => {
                        Some(entry.phy_col_start + (virt - base))
                    }
                };
            }

            base += width;
        }

        None
    }

    // Fresh entries have phy_col_start == running_phy;
    // reuse entries point backward.
    fn find_phy_source(
        &self,
        target_start: usize,
        target_count: usize,
    ) -> Result<(usize, ColumnType), Error> {
        let mut running_phy = 0usize;
        for entry in &self.entries {
            if entry.phy_col_start != running_phy {
                continue;
            }

            let entry_count = entry.kind.count();
            let entry_end = running_phy + entry_count;

            if target_start >= running_phy && target_start + target_count <= entry_end {
                let storage = entry.kind.storage();
                let offset_in_entry = target_start - running_phy;

                return Ok((
                    entry.byte_offset + offset_in_entry * storage.byte_size(),
                    storage,
                ));
            }

            running_phy = entry_end;
        }

        Err(Error::Protocol {
            protocol: "virtual_expand",
            message: "reuse: source columns not found in any single fresh entry",
        })
    }
}

impl Default for VirtualExpander {
    fn default() -> Self {
        Self::new()
    }
}

/// Maps the claimed virtual evals and the committed columns
/// onto ring-switch binding units. `eta^k` runs once per
/// unit (in claim order); a bit-expanded physical column
/// is one Ring unit consuming its `bits` claims.
pub struct RingSwitchPlan {
    pub num_units: usize,
    pub units: Vec<(bool, usize)>,
    pub phys_rs: Vec<ColumnType>,
    pub blind_slots: usize,
    pub h_cols: usize,

    phys_bit: Vec<Vec<usize>>,
    phys_whole: Vec<Vec<usize>>,
    h_whole: Vec<usize>,
}

impl RingSwitchPlan {
    pub fn new(
        layout: &[ColumnType],
        entries: Option<&[ExpansionEntry]>,
        num_blind: usize,
        num_h: usize,
    ) -> Result<Self, Error> {
        let num_phys = layout.len();
        let total = num_phys + num_blind;

        let mut phys_bit = vec![Vec::new(); total];
        let mut phys_whole = vec![Vec::new(); total];
        let mut units: Vec<(bool, usize)> = Vec::new();

        let mut phys_rs: Vec<ColumnType> = layout.iter().map(|ct| ct.rs_field()).collect();
        phys_rs.extend((0..num_blind).map(|_| ColumnType::B128));

        let bounds = |upper: usize| -> Result<(), Error> {
            if upper > num_phys {
                return Err(Error::Protocol {
                    protocol: "ring_switch_plan",
                    message: "expansion entry exceeds the physical column layout",
                });
            }

            Ok(())
        };

        match entries {
            Some(entries) => {
                let mut running = 0usize;
                for e in entries {
                    match *e {
                        ExpansionEntry::ExpandBits { count, storage } => {
                            let bits = expand_bit_width(storage)?;

                            bounds(running + count)?;

                            for j in 0..count {
                                phys_bit[running + j].push(units.len());
                                units.push((true, bits));
                            }

                            running += count;
                        }
                        ExpansionEntry::PassThrough { count, .. }
                        | ExpansionEntry::ControlBits { count } => {
                            bounds(running + count)?;

                            for j in 0..count {
                                phys_whole[running + j].push(units.len());
                                units.push((false, 1));
                            }

                            running += count;
                        }
                        ExpansionEntry::ReusePassThrough {
                            phy_col_start,
                            count,
                            ..
                        } => {
                            bounds(phy_col_start + count)?;

                            for j in 0..count {
                                phys_whole[phy_col_start + j].push(units.len());
                                units.push((false, 1));
                            }
                        }
                        ExpansionEntry::ReuseExpandBits {
                            phy_col_start,
                            count,
                            storage,
                        } => {
                            let bits = expand_bit_width(storage)?;

                            bounds(phy_col_start + count)?;

                            for j in 0..count {
                                phys_bit[phy_col_start + j].push(units.len());
                                units.push((true, bits));
                            }
                        }
                    }
                }

                if running != num_phys {
                    return Err(Error::Protocol {
                        protocol: "ring_switch_plan",
                        message: "expansion entries do not cover the physical column layout",
                    });
                }
            }
            None => {
                for pw in phys_whole.iter_mut().take(num_phys) {
                    pw.push(units.len());
                    units.push((false, 1));
                }
            }
        }

        let has_ring = units.iter().any(|(is_ring, _)| *is_ring);

        for b in 0..num_blind {
            phys_whole[num_phys + b].push(units.len());
            units.push((false, 1));
        }

        let mut blind_slots = num_blind;

        // A uniform L-valued unit
        if has_ring && num_blind > 0 {
            phys_bit.push(vec![units.len()]);
            phys_whole.push(Vec::new());
            phys_rs.push(ColumnType::B128);
            units.push((true, RING_BLIND_BITS));
            blind_slots += 1;
        }

        let mut h_whole = Vec::with_capacity(num_h);
        for _ in 0..num_h {
            h_whole.push(units.len());
            units.push((false, 1));
        }

        let num_units = units.len();

        Ok(Self {
            units,
            phys_bit,
            phys_whole,
            h_whole,
            phys_rs,
            blind_slots,
            h_cols: num_h,
            num_units,
        })
    }

    pub fn has_ring(&self) -> bool {
        self.units.iter().any(|(is_ring, _)| *is_ring)
    }

    pub fn has_ring_blind(&self) -> bool {
        self.blind_slots > 0 && self.has_ring()
    }

    pub fn total_claims(&self) -> usize {
        self.units.iter().map(|(_, n)| n).sum()
    }

    pub fn leaf_row_bytes(&self) -> usize {
        self.phys_rs.iter().map(|ct| ct.byte_size()).sum()
    }

    pub fn h_leaf_row_bytes(&self) -> usize {
        self.h_cols * ColumnType::B128.byte_size()
    }

    pub fn opened_row_bytes(&self) -> usize {
        self.leaf_row_bytes() + self.h_leaf_row_bytes()
    }

    /// `log2(grid_cols)` for this table: the proof-size optimum,
    /// stepped down, narrowing the grid to buy proximity bits,
    /// to the widest split that clears `MIN_PRODUCTION_BITS`.
    pub fn split_vars(&self, num_vars: usize, field_bits: usize, config: &Config) -> usize {
        let split = compute_split_vars(
            num_vars,
            config.num_queries,
            config.ldt_support_size,
            self.opened_row_bytes(),
        );

        let floor = support_floor_vars(config.ldt_support_size)
            .clamp(1, num_vars.max(1))
            .min(split);

        let bits = |c: usize| {
            config.estimated_security_bits(
                field_bits,
                FoldShape {
                    grid_cols: 1 << c,
                    grid_rows: 1 << (num_vars - c),
                    units: self.num_units,
                },
            )
        };

        // Never config.min_security_bits: unabsorbed, leaving the
        // two sides free to derive different grids for one proof.
        (floor..=split)
            .rev()
            .find(|&c| bits(c) >= MIN_PRODUCTION_BITS)
            .unwrap_or(split)
    }

    /// Per committed column, trace leaf then h leaf:
    /// its base `eta` coefficients in the ring and whole
    /// masters, plus `eta^U` (the next-row shift multiplier).
    pub fn column_coeffs<F>(&self, eta: Flat<F>) -> (Vec<Flat<F>>, Vec<Flat<F>>, Flat<F>)
    where
        F: HardwareField,
    {
        let mut eta_pows = Vec::with_capacity(self.num_units + 1);
        let mut e = Flat::from_raw(F::ONE);

        for _ in 0..=self.num_units {
            eta_pows.push(e);
            e *= eta;
        }

        let num_phys = self.phys_rs.len();
        let total = num_phys + self.h_cols;

        let mut coeff_bit = vec![Flat::from_raw(F::ZERO); total];
        let mut coeff_whole = vec![Flat::from_raw(F::ZERO); total];

        for p in 0..num_phys {
            for &u in &self.phys_bit[p] {
                coeff_bit[p] += eta_pows[u];
            }

            for &u in &self.phys_whole[p] {
                coeff_whole[p] += eta_pows[u];
            }
        }

        for (p, &u) in self.h_whole.iter().enumerate() {
            coeff_whole[num_phys + p] += eta_pows[u];
        }

        (coeff_bit, coeff_whole, eta_pows[self.num_units])
    }
}

fn expand_bit_width(storage: ColumnType) -> Result<usize, Error> {
    match storage {
        ColumnType::B8 => Ok(8),
        ColumnType::B16 => Ok(16),
        ColumnType::B32 => Ok(32),
        ColumnType::B64 => Ok(64),
        _ => Err(Error::Protocol {
            protocol: "virtual_expand",
            message: "ExpandBits requires B8/B16/B32/B64",
        }),
    }
}

/// Tower-basis bit extraction from LE bytes.
fn parse_tower_bit(storage: ColumnType, bytes: &[u8], bit_idx: usize) -> u8 {
    match storage {
        ColumnType::B8 => Flat::from_raw(Block8(bytes[0])).tower_bit(bit_idx),
        ColumnType::B16 => {
            let mut arr = [0u8; 2];
            arr.copy_from_slice(bytes);

            Flat::from_raw(Block16(u16::from_le_bytes(arr))).tower_bit(bit_idx)
        }
        ColumnType::B32 => {
            let mut arr = [0u8; 4];
            arr.copy_from_slice(bytes);

            Flat::from_raw(Block32(u32::from_le_bytes(arr))).tower_bit(bit_idx)
        }
        ColumnType::B64 => {
            let mut arr = [0u8; 8];
            arr.copy_from_slice(bytes);

            Flat::from_raw(Block64(u64::from_le_bytes(arr))).tower_bit(bit_idx)
        }
        _ => unreachable!(),
    }
}

fn expand_packed_bit<F: TraceCompatibleField + 'static>(
    col: &'_ TraceColumn,
    storage: ColumnType,
    bit_idx: usize,
) -> Result<PolyVariant<'_, F>, Error> {
    match storage {
        ColumnType::B8 => {
            let data = col.as_b8_slice().ok_or(Error::Protocol {
                protocol: "virtual_expand",
                message: "ExpandBits B8: column type mismatch",
            })?;

            Ok(PolyVariant::PackedBitB8 { data, bit_idx })
        }
        ColumnType::B16 => {
            let data = col.as_b16_slice().ok_or(Error::Protocol {
                protocol: "virtual_expand",
                message: "ExpandBits B16: column type mismatch",
            })?;

            Ok(PolyVariant::PackedBitB16 { data, bit_idx })
        }
        ColumnType::B32 => {
            let data = col.as_b32_slice().ok_or(Error::Protocol {
                protocol: "virtual_expand",
                message: "ExpandBits B32: column type mismatch",
            })?;

            Ok(PolyVariant::PackedBitB32 { data, bit_idx })
        }
        ColumnType::B64 => {
            let data = col.as_b64_slice().ok_or(Error::Protocol {
                protocol: "virtual_expand",
                message: "ExpandBits B64: column type mismatch",
            })?;

            Ok(PolyVariant::PackedBitB64 { data, bit_idx })
        }
        _ => unreachable!(),
    }
}

fn expand_pass_through<F: TraceCompatibleField + 'static>(
    col: &TraceColumn,
    storage: ColumnType,
) -> Result<PolyVariant<'_, F>, Error> {
    match storage {
        ColumnType::Bit => {
            let data = col.as_bit_slice().ok_or(Error::Protocol {
                protocol: "virtual_expand",
                message: "PassThrough Bit: column type mismatch",
            })?;

            Ok(PolyVariant::BitSlice(data))
        }
        ColumnType::B8 => {
            let data = col.as_b8_slice().ok_or(Error::Protocol {
                protocol: "virtual_expand",
                message: "PassThrough B8: column type mismatch",
            })?;

            Ok(PolyVariant::B8Slice(data))
        }
        ColumnType::B16 => {
            let data = col.as_b16_slice().ok_or(Error::Protocol {
                protocol: "virtual_expand",
                message: "PassThrough B16: column type mismatch",
            })?;

            Ok(PolyVariant::B16Slice(data))
        }
        ColumnType::B32 => {
            let data = col.as_b32_slice().ok_or(Error::Protocol {
                protocol: "virtual_expand",
                message: "PassThrough B32: column type mismatch",
            })?;

            Ok(PolyVariant::B32Slice(data))
        }
        ColumnType::B64 => {
            let data = col.as_b64_slice().ok_or(Error::Protocol {
                protocol: "virtual_expand",
                message: "PassThrough B64: column type mismatch",
            })?;

            Ok(PolyVariant::B64Slice(data))
        }
        ColumnType::B128 => {
            let data = col.as_b128_slice().ok_or(Error::Protocol {
                protocol: "virtual_expand",
                message: "PassThrough B128: column type mismatch",
            })?;

            Ok(PolyVariant::B128Slice(data))
        }
    }
}

pub fn eq_tensor_b(r: &[Block128]) -> Vec<Block128> {
    let mut t = vec![Block128::ONE];
    for &ri in r {
        let len = t.len();
        let mut nt = Vec::with_capacity(len * 2);

        for &v in &t {
            nt.push(v * (Block128::ONE + ri));
        }

        for &v in &t {
            nt.push(v * ri);
        }

        t = nt;
    }

    t
}

/// Σ_u eq(r'',u)·ŝ_u for one ring unit, ŝ_u = Σ_v bit_u(c_v)·2^v.
pub fn ring_batch_b(bit_claims: &[Block128], eq_mix: &[Block128]) -> Block128 {
    let mut acc = Block128::ZERO;
    for (u, &m) in eq_mix.iter().enumerate() {
        let mut shat = 0u128;
        for (v, cv) in bit_claims.iter().enumerate() {
            shat |= ((cv.0 >> u) & 1) << v;
        }

        acc += m * Block128(shat);
    }

    acc
}

/// Reconstructs the sumcheck's initial claim from the claimed
/// virtual evals, in the tower basis. Ring units contribute
/// `eta·Σ_u eq(r'',u) ŝ_u`; whole units contribute `eta·c'`.
pub fn ring_target<F>(
    plan: &RingSwitchPlan,
    claims: &[Flat<F>],
    eta_tower: F,
    r_mix: &[Block128],
    shifted_claims: bool,
) -> Block128
where
    F: HardwareField + Into<Block128>,
{
    let claim_halves = if shifted_claims { 2 } else { 1 };
    let half = claims.len() / claim_halves;
    let eq_mix = eq_tensor_b(r_mix);
    let eta: Block128 = eta_tower.into();

    let mut eta_pows = Vec::with_capacity(plan.num_units + 1);
    let mut e = Block128::ONE;

    for _ in 0..=plan.num_units {
        eta_pows.push(e);
        e *= eta;
    }

    let eta_shift = eta_pows[plan.num_units];

    let base = [(0usize, Block128::ONE)];
    let base_and_shift = [(0usize, Block128::ONE), (half, eta_shift)];
    let offsets: &[(usize, Block128)] = if shifted_claims {
        &base_and_shift
    } else {
        &base
    };

    let mut target = Block128::ZERO;
    for &(offset, shift_mul) in offsets {
        let half_claims = &claims[offset..offset + half];

        let mut ci = 0usize;
        for (unit_idx, &(is_ring, num_claims)) in plan.units.iter().enumerate() {
            let weight = eta_pows[unit_idx] * shift_mul;
            if is_ring {
                let bits: Vec<Block128> = half_claims[ci..ci + num_claims]
                    .iter()
                    .map(|f| f.to_tower().into())
                    .collect();

                target += weight * ring_batch_b(&bits, &eq_mix);
            } else {
                let c: Block128 = half_claims[ci].to_tower().into();
                target += weight * c;
            }

            ci += num_claims;
        }
    }

    target
}

/// Per-claim weight `a_c` such that
/// `ring_target = Σ_c a_c · (ring ? φ_{r''}(c_c) : c_c)`,
/// with `φ_{r''}(x) = Σ_u eq(r'',u)·bit_u(x)`.
pub fn claim_weights<F>(
    plan: &RingSwitchPlan,
    eta_tower: F,
    shifted_claims: bool,
) -> Vec<(bool, Block128)>
where
    F: HardwareField + Into<Block128>,
{
    let eta: Block128 = eta_tower.into();

    let mut eta_pows = Vec::with_capacity(plan.num_units + 1);
    let mut e = Block128::ONE;

    for _ in 0..=plan.num_units {
        eta_pows.push(e);
        e *= eta;
    }

    let shifts: &[Block128] = if shifted_claims {
        &[Block128::ONE, eta_pows[plan.num_units]]
    } else {
        &[Block128::ONE]
    };

    let mut weights = Vec::with_capacity(plan.total_claims() * shifts.len());
    for &shift_mul in shifts {
        for (unit_idx, &(is_ring, num_claims)) in plan.units.iter().enumerate() {
            let weight = eta_pows[unit_idx] * shift_mul;
            for v in 0..num_claims {
                let basis = if is_ring {
                    Block128(1u128 << v)
                } else {
                    Block128::ONE
                };

                weights.push((is_ring, weight * basis));
            }
        }
    }

    weights
}

#[cfg(test)]
mod tests {
    use super::*;
    use hekate_core::trace::TraceBuilder;
    use hekate_math::{Block128, TowerField};

    fn keccak_expander() -> VirtualExpander {
        VirtualExpander::new()
            .expand_bits(25, ColumnType::B64)
            .expand_bits(1, ColumnType::B64)
            .reuse_pass_through(0, 25)
            .control_bits(2)
            .build()
            .unwrap()
    }

    fn keccak_physical_layout() -> Vec<ColumnType> {
        let mut layout = vec![ColumnType::B64; 26];
        layout.extend(repeat_n(ColumnType::Bit, 2));

        layout
    }

    fn plan_of(cols: usize) -> RingSwitchPlan {
        RingSwitchPlan::new(&vec![ColumnType::B128; cols], None, 1, 1).unwrap()
    }

    #[test]
    fn ram_layout() {
        let e = VirtualExpander::new()
            .expand_bits(2, ColumnType::B32)
            .pass_through(13, ColumnType::B32)
            .pass_through(1, ColumnType::B128)
            .control_bits(4)
            .build()
            .unwrap();

        assert_eq!(e.num_virtual_columns(), 82);
        assert_eq!(e.num_physical_columns(), 20);
        assert_eq!(e.physical_row_bytes(), 80);

        let layout = e.virtual_layout();

        assert_eq!(layout.len(), 82);
        assert!(layout[..64].iter().all(|&t| t == ColumnType::Bit));
        assert!(layout[64..77].iter().all(|&t| t == ColumnType::B32));
        assert_eq!(layout[77], ColumnType::B128);
        assert!(layout[78..82].iter().all(|&t| t == ColumnType::Bit));
    }

    #[test]
    fn keccak_layout() {
        let e = VirtualExpander::new()
            .expand_bits(25, ColumnType::B64)
            .expand_bits(1, ColumnType::B64)
            .reuse_pass_through(0, 25)
            .control_bits(2)
            .build()
            .unwrap();

        assert_eq!(e.num_virtual_columns(), 1691);
        assert_eq!(e.num_physical_columns(), 28);
        assert_eq!(e.physical_row_bytes(), 210);

        let layout = e.virtual_layout();

        assert_eq!(layout.len(), 1691);
        assert!(layout[..1600].iter().all(|&t| t == ColumnType::Bit));
        assert!(layout[1600..1664].iter().all(|&t| t == ColumnType::Bit));
        assert!(layout[1664..1689].iter().all(|&t| t == ColumnType::B64));
        assert!(layout[1689..1691].iter().all(|&t| t == ColumnType::Bit));
    }

    #[test]
    fn ring_switch_plan_rejects_uncovered_columns() {
        let expander = keccak_expander();
        let entries = expander.expansion_entries();

        let mut layout = keccak_physical_layout();

        assert_eq!(expander.num_physical_columns(), layout.len());
        assert!(RingSwitchPlan::new(&layout, Some(&entries), 0, 0).is_ok());

        layout.push(ColumnType::B64);

        assert!(RingSwitchPlan::new(&layout, Some(&entries), 0, 0).is_err());
    }

    #[test]
    fn ring_switch_plan_folds_every_committed_column() {
        let entries = keccak_expander().expansion_entries();
        let layout = keccak_physical_layout();

        let plan = RingSwitchPlan::new(&layout, Some(&entries), 2, 0).unwrap();

        let zero = Flat::from_raw(Block128::ZERO);
        let eta = Block128(0x2545F4914F6CDD1D_517CC1B727220A95).to_hardware();
        let (coeff_bit, coeff_whole, _) = plan.column_coeffs::<Block128>(eta);

        for p in 0..plan.phys_rs.len() {
            assert!(
                coeff_bit[p] != zero || coeff_whole[p] != zero,
                "committed column {p} enters no master fold"
            );
        }
    }

    #[test]
    fn h_units_trail_ring_blind() {
        let layout = [ColumnType::B32, ColumnType::B64];
        let expander = VirtualExpander::new()
            .expand_bits(1, ColumnType::B32)
            .pass_through(1, ColumnType::B64)
            .build()
            .unwrap();
        let entries = expander.expansion_entries();

        let bare = RingSwitchPlan::new(&layout, Some(&entries), 1, 0).unwrap();
        let plan = RingSwitchPlan::new(&layout, Some(&entries), 1, 2).unwrap();

        assert_eq!(plan.h_cols, 2);
        assert_eq!(plan.phys_rs, bare.phys_rs);
        assert_eq!(plan.total_claims(), bare.total_claims() + 2);
        assert_eq!(&plan.units[plan.num_units - 2..], &[(false, 1), (false, 1)]);
        assert_eq!(plan.units[plan.num_units - 3], (true, RING_BLIND_BITS));
        assert_eq!(plan.leaf_row_bytes(), bare.opened_row_bytes());
        assert_eq!(plan.opened_row_bytes(), bare.opened_row_bytes() + 32);

        let eta = Block128::from(0x9E37_79B9u128).to_hardware();
        let (coeff_bit, coeff_whole, eta_shift) = plan.column_coeffs::<Block128>(eta);
        let n = plan.phys_rs.len();

        let mut first_h = Flat::from_raw(Block128::ONE);
        for _ in 0..plan.num_units - 2 {
            first_h *= eta;
        }

        assert_eq!(coeff_whole.len(), n + 2);
        assert_eq!(coeff_bit[n..], [Flat::from_raw(Block128::ZERO); 2]);
        assert_eq!(coeff_whole[n], first_h);
        assert_eq!(coeff_whole[n + 1], first_h * eta);
        assert_eq!(eta_shift, first_h * eta * eta);
    }

    #[test]
    fn blinded_ring_plan_ends_in_ring_blind_unit() {
        let entries = keccak_expander().expansion_entries();
        let layout = keccak_physical_layout();

        let raw = RingSwitchPlan::new(&layout, Some(&entries), 0, 0).unwrap();
        let blinded = RingSwitchPlan::new(&layout, Some(&entries), 2, 0).unwrap();
        let whole_only = RingSwitchPlan::new(&[ColumnType::B32; 3], None, 2, 0).unwrap();

        assert_eq!(raw.blind_slots, 0);
        assert_eq!(raw.phys_rs.len(), layout.len());

        assert_eq!(blinded.blind_slots, 3);
        assert_eq!(blinded.phys_rs.len(), layout.len() + 3);
        assert_eq!(blinded.units.last(), Some(&(true, RING_BLIND_BITS)));
        assert_eq!(
            blinded.total_claims(),
            raw.total_claims() + 2 + RING_BLIND_BITS
        );

        assert_eq!(whole_only.blind_slots, 2);
        assert_eq!(whole_only.units.last(), Some(&(false, 1)));

        let eta = Block128(0x2545F4914F6CDD1D_517CC1B727220A95).to_hardware();
        let (coeff_bit, coeff_whole, _) = blinded.column_coeffs::<Block128>(eta);
        let last = blinded.phys_rs.len() - 1;

        assert_ne!(coeff_bit[last], Flat::from_raw(Block128::ZERO));
        assert_eq!(coeff_whole[last], Flat::from_raw(Block128::ZERO));
    }

    #[test]
    fn reuse_partial_range() {
        let e = VirtualExpander::new()
            .expand_bits(10, ColumnType::B32)
            .reuse_pass_through(3, 4)
            .build()
            .unwrap();

        assert_eq!(e.num_virtual_columns(), 324);
        assert_eq!(e.num_physical_columns(), 10);
        assert_eq!(e.physical_row_bytes(), 40);

        let layout = e.virtual_layout();

        assert_eq!(layout[320..324].len(), 4);
        assert!(layout[320..324].iter().all(|&t| t == ColumnType::B32));
    }

    #[test]
    fn reuse_exceeds_declared() {
        let result = VirtualExpander::new()
            .expand_bits(5, ColumnType::B32)
            .reuse_pass_through(3, 5)
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn reuse_expand_bits_from_pass_through() {
        let e = VirtualExpander::new()
            .pass_through(4, ColumnType::B64)
            .reuse_expand_bits(0, 4)
            .build()
            .unwrap();

        assert_eq!(e.num_physical_columns(), 4);
        assert_eq!(e.physical_row_bytes(), 32);
        assert_eq!(e.num_virtual_columns(), 4 + 256);

        let layout = e.virtual_layout();

        assert!(layout[0..4].iter().all(|&t| t == ColumnType::B64));
        assert!(layout[4..260].iter().all(|&t| t == ColumnType::Bit));
    }

    #[test]
    fn reuse_expand_bits_exceeds_declared() {
        let result = VirtualExpander::new()
            .pass_through(4, ColumnType::B64)
            .reuse_expand_bits(2, 4)
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn reuse_expand_bits_rejects_b128_source() {
        let result = VirtualExpander::new()
            .pass_through(1, ColumnType::B128)
            .reuse_expand_bits(0, 1)
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn expand_rejects_bit() {
        let result = VirtualExpander::new()
            .expand_bits(1, ColumnType::Bit)
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn expand_rejects_b128() {
        let result = VirtualExpander::new()
            .expand_bits(1, ColumnType::B128)
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn empty_expander() {
        let e = VirtualExpander::new();
        assert_eq!(e.num_virtual_columns(), 0);
        assert_eq!(e.num_physical_columns(), 0);
        assert_eq!(e.physical_row_bytes(), 0);
        assert!(e.virtual_layout().is_empty());
    }

    #[test]
    fn parse_row_b32_roundtrip() {
        let expander = VirtualExpander::new()
            .expand_bits(1, ColumnType::B32)
            .pass_through(1, ColumnType::B32)
            .control_bits(1)
            .build()
            .unwrap();

        let val: u32 = 0xDEAD_BEEF;
        let pass_val: u32 = 0x1234_5678;

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&val.to_le_bytes());
        bytes.extend_from_slice(&pass_val.to_le_bytes());
        bytes.push(1);

        let mut res: Vec<Flat<Block128>> = Vec::new();
        expander.parse_row(&bytes, &mut res).unwrap();

        assert_eq!(res.len(), 34);

        for (bit_idx, elem) in res.iter().enumerate().take(32) {
            let expected = Flat::from_raw(Block32(val)).tower_bit(bit_idx);
            let got = elem.tower_bit(0);
            assert_eq!(got, expected, "bit {bit_idx} mismatch");
        }

        let pass = res[32];
        assert_eq!(
            pass,
            <Block128 as hekate_math::FlatPromote<Block32>>::promote_flat(Flat::from_raw(Block32(
                pass_val
            )))
        );

        let ctrl = res[33].tower_bit(0);
        assert_eq!(ctrl, 1);
    }

    #[test]
    fn expand_variants_b32() {
        let expander = VirtualExpander::new()
            .expand_bits(1, ColumnType::B32)
            .pass_through(1, ColumnType::B32)
            .control_bits(1)
            .build()
            .unwrap();

        let layout = [ColumnType::B32, ColumnType::B32, ColumnType::Bit];
        let num_vars = 2;

        let mut tb = TraceBuilder::new(&layout, num_vars).unwrap();
        tb.set_b32(0, 0, Block32(0xAAAA_BBBB)).unwrap();
        tb.set_b32(1, 0, Block32(0x1111_2222)).unwrap();
        tb.set_bit(2, 0, Bit::ONE).unwrap();

        let trace = tb.build();

        let variants: Vec<PolyVariant<'_, Block128>> = expander.expand_variants(&trace, 0).unwrap();

        assert_eq!(variants.len(), 34);

        for (i, v) in variants.iter().enumerate().take(32) {
            assert!(matches!(v, PolyVariant::PackedBitB32 { bit_idx, .. } if *bit_idx == i));
        }

        assert!(matches!(variants[32], PolyVariant::B32Slice(_)));
        assert!(matches!(variants[33], PolyVariant::BitSlice(_)));
    }

    #[test]
    fn claim_weights_reproduce_ring_target() {
        let layout = [
            ColumnType::B32,
            ColumnType::B32,
            ColumnType::B64,
            ColumnType::Bit,
        ];

        let expander = VirtualExpander::new()
            .expand_bits(2, ColumnType::B32)
            .pass_through(1, ColumnType::B64)
            .control_bits(1)
            .build()
            .unwrap();

        let entries = expander.expansion_entries();
        let plan = RingSwitchPlan::new(&layout, Some(&entries), 2, 0).unwrap();

        let mut state = 0x9e37_79b9_7f4a_7c15_0123_4567_89ab_cdefu128;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;

            Block128(state)
        };

        let claims: Vec<Block128> = (0..2 * plan.total_claims()).map(|_| next()).collect();
        let eta = next();
        let r_mix: Vec<Block128> = (0..7).map(|_| next()).collect();
        let eq_mix = eq_tensor_b(&r_mix);

        let flat: Vec<Flat<Block128>> = claims.iter().map(|c| c.to_hardware()).collect();
        let target = ring_target::<Block128>(&plan, &flat, eta, &r_mix, true);

        let mut sum = Block128::ZERO;
        for ((is_ring, weight), claim) in claim_weights::<Block128>(&plan, eta, true)
            .into_iter()
            .zip(&claims)
        {
            let value = match is_ring {
                true => ring_batch_b(&[*claim], &eq_mix),
                false => *claim,
            };

            sum += weight * value;
        }

        assert_eq!(sum, target);
    }

    /// The scan is only correct if it finds the best split, not
    /// merely a better one: `bits` is not unimodal in the split,
    /// and a plateau must not end the search on a local peak.
    #[test]
    fn adaptive_split_clears_floor_whenever_any_split_can() {
        let config = Config::prod();

        let mut stepped_down = 0;
        for cols in [1usize, 4, 16, 64, 128] {
            let plan = plan_of(cols);
            for num_vars in 10usize..=30 {
                let chosen = plan.split_vars(num_vars, 128, &config);
                let bits = |c: usize| {
                    config.estimated_security_bits(
                        128,
                        FoldShape {
                            grid_cols: 1 << c,
                            grid_rows: 1 << (num_vars - c),
                            units: plan.num_units,
                        },
                    )
                };

                let optimal = compute_split_vars(
                    num_vars,
                    config.num_queries,
                    config.ldt_support_size,
                    plan.opened_row_bytes(),
                );

                assert!(chosen <= optimal, "cols={cols} n={num_vars}: stepped up");
                assert!(
                    chosen >= support_floor_vars(config.ldt_support_size).min(optimal),
                    "cols={cols} n={num_vars}: below the support floor"
                );

                if chosen < optimal {
                    stepped_down += 1;
                }

                let best = (1..=optimal).map(bits).max().unwrap();

                if best >= MIN_PRODUCTION_BITS {
                    assert!(
                        bits(chosen) >= MIN_PRODUCTION_BITS,
                        "cols={cols} n={num_vars}: chose {chosen} at {} bits, \
                         but some split reaches {best}",
                        bits(chosen),
                    );
                }
            }
        }

        assert!(stepped_down > 0, "the adaptive walk never fired");
    }
}
