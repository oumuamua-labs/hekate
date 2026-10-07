// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec;
use alloc::vec::Vec;
use core::iter::repeat_n;
use core::ops::Range;
use hekate_core::config::{Config, FoldShape, MIN_PRODUCTION_BITS};
use hekate_core::errors::Error;
use hekate_core::poly::PolyVariant;
use hekate_core::tensor::TensorProduct;
use hekate_core::trace::{ColumnType, Trace, TraceColumn, TraceCompatibleField};
use hekate_core::utils::{cheapest_split_vars, support_floor_vars};
use hekate_math::{
    Bit, Block8, Block16, Block32, Block64, Block128, Flat, HardwareField, TowerField,
};

use crate::linearized::{BITS, LinearMap};

pub const RING_BLIND_BITS: usize = 128;

/// Claim count from which [`ring_target`] switches
/// to byte tables indexed by the claims: variable-time.
const RING_TABLE_MIN_CLAIMS: usize = 300;

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

/// Maps the claimed virtual evals and the committed
/// columns onto ring-switch binding units, in claim
/// order; a bit-expanded physical column is one
/// Ring unit consuming its `bits` claims.
#[derive(Clone)]
pub struct RingSwitchPlan {
    pub num_units: usize,
    pub units: Vec<(bool, usize)>,
    pub phys_rs: Vec<ColumnType>,
    pub blind_cols: usize,
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

        let mut phys_bit = vec![Vec::new(); num_phys];
        let mut phys_whole = vec![Vec::new(); num_phys];
        let mut units: Vec<(bool, usize)> = Vec::new();

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
                for pw in phys_whole.iter_mut() {
                    pw.push(units.len());
                    units.push((false, 1));
                }
            }
        }

        let mut plan = Self {
            num_units: units.len(),
            units,
            phys_rs: layout.iter().map(|ct| ct.rs_field()).collect(),
            blind_cols: 0,
            h_cols: 0,
            phys_bit,
            phys_whole,
            h_whole: Vec::new(),
        };

        plan.append_tail(num_blind, num_h);

        Ok(plan)
    }

    pub fn has_ring(&self) -> bool {
        self.units.iter().any(|(is_ring, _)| *is_ring)
    }

    pub fn has_ring_blind(&self) -> bool {
        self.blind_cols > 0 && self.has_ring()
    }

    pub fn whole_blinds(&self) -> usize {
        self.blind_cols - usize::from(self.has_ring_blind())
    }

    pub fn total_claims(&self) -> usize {
        self.units.iter().map(|(_, n)| n).sum()
    }

    pub fn total_claims_at(&self, num_vars: usize, split_vars: usize) -> usize {
        let pack = 1usize << split_vars.saturating_sub(num_vars);

        self.total_claims() + self.whole_blinds() * (pack - 1)
    }

    pub fn opened_row_bytes(&self) -> usize {
        let trace: usize = self.phys_rs.iter().map(|ct| ct.byte_size()).sum();

        trace + self.h_cols * ColumnType::B128.byte_size()
    }

    /// Committed columns in codeword slots at `split_vars`;
    /// past `num_vars` a slot packs `2^(split - num_vars)` columns
    /// of one class, zero blocks closing a class's last slot.
    pub fn at_split(&self, num_vars: usize, split_vars: usize) -> SlotLayout {
        let pack_vars = split_vars.saturating_sub(num_vars);
        let pack = 1usize << pack_vars;

        let plan = match pack {
            1 => self.clone(),
            _ => self.with_whole_blinds(self.whole_blinds() * pack),
        };

        let num_cols = plan.phys_rs.len();

        let mut open: Vec<(SlotClass, usize)> = Vec::new();
        let mut slot_rs = Vec::new();
        let mut slot_roles = Vec::new();
        let mut slot_fill: Vec<usize> = Vec::new();
        let mut col_slot = Vec::with_capacity(num_cols);

        for p in 0..num_cols {
            let class = plan.slot_class(p);
            let current = open.iter().position(|(c, _)| *c == class);

            let slot = match current {
                Some(i) if slot_fill[open[i].1] < pack => open[i].1,
                _ => {
                    let s = slot_rs.len();

                    slot_rs.push(class.0);
                    slot_roles.push((class.1, class.2));
                    slot_fill.push(0);

                    match current {
                        Some(i) => open[i].1 = s,
                        None => open.push((class, s)),
                    }

                    s
                }
            };

            col_slot.push((slot, slot_fill[slot]));

            slot_fill[slot] += 1;
        }

        let num_slots = slot_rs.len();
        let h_slots = plan.h_cols.div_ceil(pack);

        let mut slot_start = Vec::with_capacity(num_slots + 1);
        slot_start.push(0);

        for &fill in &slot_fill {
            slot_start.push(slot_start[slot_start.len() - 1] + fill);
        }

        let mut slot_cols = vec![0; num_cols];
        for (p, &(slot, block)) in col_slot.iter().enumerate() {
            slot_cols[slot_start[slot] + block] = p;
        }

        let mut group_base = Vec::with_capacity(num_slots + h_slots);
        let mut groups = 0;

        for &(ring, whole) in &slot_roles {
            group_base.push(groups);

            groups += ring + whole;
        }

        for _ in 0..h_slots {
            group_base.push(groups);

            groups += 1;
        }

        let mut unit_home = vec![(0, 0, 0, false); plan.num_units];

        for (p, &(slot, block)) in col_slot.iter().enumerate() {
            let rings = plan.phys_bit[p].len();

            for (i, &u) in plan.phys_bit[p].iter().enumerate() {
                unit_home[u] = (slot, block, group_base[slot] + i, true);
            }

            for (i, &u) in plan.phys_whole[p].iter().enumerate() {
                unit_home[u] = (slot, block, group_base[slot] + rings + i, false);
            }
        }

        for (k, &u) in plan.h_whole.iter().enumerate() {
            let slot = num_slots + k / pack;
            unit_home[u] = (slot, k % pack, group_base[slot], false);
        }

        let mut group_sigma: Vec<Option<usize>> = vec![None; groups];
        let mut sigma_slot = Vec::with_capacity(groups);
        let mut unit_sigma = Vec::with_capacity(plan.num_units);
        let mut unit_block = Vec::with_capacity(plan.num_units);

        for &(slot, block, group, ring) in &unit_home {
            let sigma = *group_sigma[group].get_or_insert_with(|| {
                sigma_slot.push((slot, ring));

                sigma_slot.len() - 1
            });

            unit_sigma.push(sigma);
            unit_block.push(block);
        }

        SlotLayout {
            plan,
            num_vars,
            split_vars,
            slot_rs,
            h_slots,
            num_sigma: sigma_slot.len(),
            slot_start,
            slot_cols,
            unit_sigma,
            unit_block,
            sigma_base: 0,
            sigma_shift: sigma_slot.len(),
            sigma_slot,
        }
    }

    fn append_tail(&mut self, num_blind: usize, num_h: usize) {
        let has_ring = self.has_ring();

        for _ in 0..num_blind {
            self.phys_bit.push(Vec::new());
            self.phys_whole.push(vec![self.units.len()]);
            self.phys_rs.push(ColumnType::B128);
            self.units.push((false, 1));
        }

        self.blind_cols = num_blind;

        // A uniform L-valued unit
        if has_ring && num_blind > 0 {
            self.phys_bit.push(vec![self.units.len()]);
            self.phys_whole.push(Vec::new());
            self.phys_rs.push(ColumnType::B128);
            self.units.push((true, RING_BLIND_BITS));

            self.blind_cols += 1;
        }

        for _ in 0..num_h {
            self.h_whole.push(self.units.len());
            self.units.push((false, 1));
        }

        self.h_cols = num_h;
        self.num_units = self.units.len();
    }

    fn with_whole_blinds(&self, num_blind: usize) -> Self {
        let data_cols = self.phys_rs.len() - self.blind_cols;
        let data_units = self.num_units - self.blind_cols - self.h_cols;

        let mut plan = Self {
            num_units: data_units,
            units: self.units[..data_units].to_vec(),
            phys_rs: self.phys_rs[..data_cols].to_vec(),
            blind_cols: 0,
            h_cols: 0,
            phys_bit: self.phys_bit[..data_cols].to_vec(),
            phys_whole: self.phys_whole[..data_cols].to_vec(),
            h_whole: Vec::new(),
        };

        plan.append_tail(num_blind, self.h_cols);

        plan
    }

    fn slot_class(&self, col: usize) -> SlotClass {
        (
            self.phys_rs[col],
            self.phys_bit[col].len(),
            self.phys_whole[col].len(),
            col >= self.phys_rs.len() - self.blind_cols,
        )
    }

    fn census(&self) -> Census {
        let mut data: Vec<(SlotClass, usize)> = Vec::new();
        for col in 0..self.phys_rs.len() - self.blind_cols {
            let class = self.slot_class(col);

            match data.iter_mut().find(|(c, _)| *c == class) {
                Some((_, count)) => *count += 1,
                None => data.push((class, 1)),
            }
        }

        Census {
            data,
            whole_blinds: self.whole_blinds(),
            ring_blind: self.has_ring_blind(),
            h_cols: self.h_cols,
        }
    }
}

type SlotClass = (ColumnType, usize, usize, bool);

/// A [`RingSwitchPlan`] at one table height: committed columns
/// in codeword slots, block `b` of a slot at message offset
/// `b · 2^num_vars`, and one `eta` exponent per slot role.
pub struct SlotLayout {
    pub plan: RingSwitchPlan,
    pub num_vars: usize,
    pub split_vars: usize,
    pub slot_rs: Vec<ColumnType>,
    pub h_slots: usize,
    pub num_sigma: usize,

    slot_start: Vec<usize>,
    slot_cols: Vec<usize>,
    unit_sigma: Vec<usize>,
    unit_block: Vec<usize>,
    sigma_slot: Vec<(usize, bool)>,
    sigma_base: usize,
    sigma_shift: usize,
}

impl SlotLayout {
    pub fn pack_vars(&self) -> usize {
        self.split_vars.saturating_sub(self.num_vars)
    }

    pub fn pack(&self) -> usize {
        1 << self.pack_vars()
    }

    pub fn grid_cols(&self) -> usize {
        1 << self.split_vars
    }

    pub fn grid_rows(&self) -> usize {
        1 << self.num_vars.saturating_sub(self.split_vars)
    }

    /// Plan columns of trace slot `slot`, in block order.
    pub fn slot_columns(&self, slot: usize) -> &[usize] {
        &self.slot_cols[self.slot_start[slot]..self.slot_start[slot + 1]]
    }

    pub fn h_slot_columns(&self, slot: usize) -> Range<usize> {
        let pack = self.pack();

        slot * pack..((slot + 1) * pack).min(self.plan.h_cols)
    }

    pub fn leaf_row_bytes(&self) -> usize {
        self.slot_rs.iter().map(|ct| ct.byte_size()).sum()
    }

    pub fn h_leaf_row_bytes(&self) -> usize {
        self.h_slots * ColumnType::B128.byte_size()
    }

    pub fn unit_block(&self, unit: usize) -> usize {
        self.unit_block[unit]
    }

    /// PRF salt of blind column `blind`'s message,
    /// past every slot's support salt.
    pub fn blind_salt(&self, blind: usize) -> usize {
        self.slot_rs.len() + blind
    }

    /// Per-unit claim weights `eta^σ(u) · eq(rho, b(u))`
    /// and the next-row factor `eta^#σ`. Errors unless
    /// `rho` holds one challenge per pack variable.
    pub fn unit_weights<F: HardwareField>(
        &self,
        eta: Flat<F>,
        rho: &[Flat<F>],
    ) -> Result<(Vec<Flat<F>>, Flat<F>), Error> {
        if rho.len() != self.pack_vars() {
            return Err(Error::Protocol {
                protocol: "ring_switch_plan",
                message: "block challenge count does not match the pack factor",
            });
        }

        let (mut weights, shift) = self.sigma_weights(eta);

        let eq = TensorProduct::new(rho.to_vec());

        for (weight, &block) in weights.iter_mut().zip(&self.unit_block) {
            *weight *= eq.evaluate_at_index(block);
        }

        Ok((weights, shift))
    }

    pub fn sigma_weights<F: HardwareField>(&self, eta: Flat<F>) -> (Vec<Flat<F>>, Flat<F>) {
        let mut pows = Vec::with_capacity(self.num_sigma);
        let mut e = eta_pow(eta, self.sigma_base);

        for _ in 0..self.num_sigma {
            pows.push(e);

            e *= eta;
        }

        let weights = self.unit_sigma.iter().map(|&sigma| pows[sigma]).collect();

        (weights, eta_pow(eta, self.sigma_shift))
    }

    /// Ring and whole fold coefficients per slot, trace slots
    /// then `h` slots, and the next-row factor `eta^#σ`.
    pub fn slot_coeffs<F: HardwareField>(
        &self,
        eta: Flat<F>,
    ) -> (Vec<Flat<F>>, Vec<Flat<F>>, Flat<F>) {
        let zero = Flat::from_raw(F::ZERO);
        let num_slots = self.slot_rs.len() + self.h_slots;

        let mut coeff_bit = vec![zero; num_slots];
        let mut coeff_whole = vec![zero; num_slots];

        let mut e = eta_pow(eta, self.sigma_base);
        for &(slot, ring) in &self.sigma_slot {
            match ring {
                true => coeff_bit[slot] += e,
                false => coeff_whole[slot] += e,
            }

            e *= eta;
        }

        (coeff_bit, coeff_whole, eta_pow(eta, self.sigma_shift))
    }

    fn weights_b128<F>(
        &self,
        eta: F,
        rho: &[F],
    ) -> Result<(Vec<Flat<Block128>>, Flat<Block128>), Error>
    where
        F: HardwareField + Into<Block128>,
    {
        let flat = |x: F| Into::<Block128>::into(x).to_hardware();
        let rho: Vec<Flat<Block128>> = rho.iter().map(|&r| flat(r)).collect();

        self.unit_weights(flat(eta), &rho)
    }
}

pub struct PoolLayout {
    pub split_vars: usize,
    pub tables: Vec<SlotLayout>,
}

impl PoolLayout {
    pub fn new(tables: &[(&RingSwitchPlan, usize)], field_bits: usize, config: &Config) -> Self {
        Self::at_split(tables, Self::split_vars_for(tables, field_bits, config))
    }

    /// The split [`Self::new`] lays out, priced on
    /// census counts alone: it allocates no blind column.
    pub fn split_vars_for(
        tables: &[(&RingSwitchPlan, usize)],
        field_bits: usize,
        config: &Config,
    ) -> usize {
        let census: Vec<(Census, usize)> = tables
            .iter()
            .map(|&(plan, num_vars)| (plan.census(), num_vars))
            .collect();

        let masters: usize = tables
            .iter()
            .map(|(plan, _)| 1 + usize::from(plan.has_ring()))
            .sum();

        let max_vars = census.iter().map(|&(_, n)| n).max().unwrap_or(0);

        let widest = census
            .iter()
            .map(|(c, n)| n + c.max_pack_vars())
            .fold(max_vars, usize::max);

        let support_floor = support_floor_vars(config.ldt_support_size);

        let row_bytes = |c: usize| -> usize {
            census
                .iter()
                .map(|(census, n)| {
                    (1usize << n.saturating_sub(c)) * census.row_bytes(1 << c.saturating_sub(*n))
                })
                .sum()
        };

        let claim_bytes = |c: usize| -> usize {
            census
                .iter()
                .map(|(census, n)| census.blind_claim_bytes(1 << c.saturating_sub(*n)))
                .sum()
        };

        let secure = |c: usize| {
            let units: usize = census
                .iter()
                .map(|(census, n)| census.num_sigma(1 << c.saturating_sub(*n)))
                .sum();

            let shape = FoldShape {
                grid_cols: 1 << c,
                grid_rows: 1 << max_vars.saturating_sub(c),
                units: units + masters.saturating_sub(1),
            };

            // Never config.min_security_bits:
            // unabsorbed, prover and verifier would diverge.
            c >= support_floor
                && config.estimated_security_bits(field_bits, shape) >= MIN_PRODUCTION_BITS
        };

        let num_queries = config.num_queries;

        cheapest_split_vars(1, widest, num_queries, row_bytes, claim_bytes, secure)
            .or_else(|| {
                cheapest_split_vars(1, widest, num_queries, row_bytes, claim_bytes, |_| true)
            })
            .unwrap_or(0)
    }

    pub fn at_split(tables: &[(&RingSwitchPlan, usize)], split_vars: usize) -> Self {
        let mut layouts: Vec<SlotLayout> = tables
            .iter()
            .map(|&(plan, num_vars)| plan.at_split(num_vars, split_vars))
            .collect();

        let sigma_shift = layouts.iter().map(|t| t.num_sigma).sum();

        let mut sigma_base = 0;
        for layout in &mut layouts {
            layout.sigma_base = sigma_base;
            layout.sigma_shift = sigma_shift;

            sigma_base += layout.num_sigma;
        }

        Self {
            split_vars,
            tables: layouts,
        }
    }

    pub fn num_vars(&self) -> usize {
        self.tables
            .iter()
            .map(|t| t.num_vars)
            .fold(self.split_vars, usize::max)
    }

    pub fn grid_cols(&self) -> usize {
        1 << self.split_vars
    }

    pub fn grid_rows(&self) -> usize {
        1 << (self.num_vars() - self.split_vars)
    }

    pub fn num_sigma(&self) -> usize {
        self.tables.iter().map(|t| t.num_sigma).sum()
    }

    pub fn num_masters(&self) -> usize {
        self.tables
            .iter()
            .map(|t| 1 + usize::from(t.plan.has_ring()))
            .sum()
    }

    pub fn rho_vars(&self) -> usize {
        self.tables
            .iter()
            .map(SlotLayout::pack_vars)
            .max()
            .unwrap_or(0)
    }

    pub fn fold_shape(&self) -> FoldShape {
        FoldShape {
            grid_cols: self.grid_cols(),
            grid_rows: self.grid_rows(),
            units: self.num_sigma() + self.num_masters().saturating_sub(1),
        }
    }
}

struct Census {
    data: Vec<(SlotClass, usize)>,
    whole_blinds: usize,
    ring_blind: bool,
    h_cols: usize,
}

impl Census {
    fn row_bytes(&self, pack: usize) -> usize {
        let data: usize = self
            .data
            .iter()
            .map(|&((rs, ..), count)| count.div_ceil(pack) * rs.byte_size())
            .sum();

        let wide = self.whole_blinds + usize::from(self.ring_blind) + self.h_cols.div_ceil(pack);

        data + wide * ColumnType::B128.byte_size()
    }

    fn num_sigma(&self, pack: usize) -> usize {
        let data: usize = self
            .data
            .iter()
            .map(|&((_, ring, whole, _), count)| count.div_ceil(pack) * (ring + whole))
            .sum();

        data + self.whole_blinds + usize::from(self.ring_blind) + self.h_cols.div_ceil(pack)
    }

    fn blind_claim_bytes(&self, pack: usize) -> usize {
        2 * self.whole_blinds * (pack - 1) * ColumnType::B128.byte_size()
    }

    fn max_pack_vars(&self) -> usize {
        let widest = self
            .data
            .iter()
            .map(|&(_, count)| count)
            .max()
            .unwrap_or(0)
            .max(self.h_cols);

        match widest > 1 {
            true => (widest - 1).ilog2() as usize + 1,
            false => 0,
        }
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
/// virtual evals, in the tower basis. Under its layout weight `w`
/// a ring unit contributes `w·Σ_u eq(r'',u) ŝ_u`, a whole unit `w·c'`.
pub fn ring_target<F>(
    layout: &SlotLayout,
    claims: &[Flat<F>],
    eta_tower: F,
    rho: &[F],
    r_mix: &[Block128],
    shifted_claims: bool,
) -> Result<Block128, Error>
where
    F: HardwareField + Into<Block128>,
{
    let (weights, shift) = layout.weights_b128(eta_tower, rho)?;

    let eq_mix = eq_tensor_b(r_mix);
    let plan = &layout.plan;

    match claims.len() < RING_TABLE_MIN_CLAIMS {
        true => Ok(ring_target_transposed(
            plan,
            claims,
            &weights,
            shift,
            &eq_mix,
            shifted_claims,
        )),
        false => Ok(ring_target_tabulated(
            plan,
            claims,
            &weights,
            shift,
            &eq_mix,
            shifted_claims,
        )),
    }
}

/// Per-claim weight `a_c` such that
/// `ring_target = Σ_c a_c · (ring ? φ_{r''}(c_c) : c_c)`,
/// with `φ_{r''}(x) = Σ_u eq(r'',u)·bit_u(x)`.
pub fn claim_weights<F>(
    layout: &SlotLayout,
    eta_tower: F,
    rho: &[F],
    shifted_claims: bool,
) -> Result<Vec<(bool, Block128)>, Error>
where
    F: HardwareField + Into<Block128>,
{
    let (unit_weights, shift) = layout.weights_b128(eta_tower, rho)?;

    let plan = &layout.plan;

    let shifts: &[Block128] = if shifted_claims {
        &[Block128::ONE, shift.to_tower()]
    } else {
        &[Block128::ONE]
    };

    let mut weights = Vec::with_capacity(plan.total_claims() * shifts.len());
    for &shift_mul in shifts {
        for (&(is_ring, num_claims), unit_weight) in plan.units.iter().zip(&unit_weights) {
            let weight = unit_weight.to_tower() * shift_mul;
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

    Ok(weights)
}

/// [`ring_target`] computed in the tower basis by
/// transposing each ring unit's claim bits into `ŝ_u`.
fn ring_target_transposed<F>(
    plan: &RingSwitchPlan,
    claims: &[Flat<F>],
    unit_weights: &[Flat<Block128>],
    shift: Flat<Block128>,
    eq_mix: &[Block128],
    shifted_claims: bool,
) -> Block128
where
    F: HardwareField + Into<Block128>,
{
    let claim_halves = if shifted_claims { 2 } else { 1 };
    let half = claims.len() / claim_halves;

    let base = [(0usize, Block128::ONE)];
    let base_and_shift = [(0usize, Block128::ONE), (half, shift.to_tower())];

    let offsets: &[(usize, Block128)] = if shifted_claims {
        &base_and_shift
    } else {
        &base
    };

    let mut target = Block128::ZERO;
    for &(offset, shift_mul) in offsets {
        let half_claims = &claims[offset..offset + half];

        let mut ci = 0usize;
        for (&(is_ring, num_claims), unit_weight) in plan.units.iter().zip(unit_weights) {
            let weight = unit_weight.to_tower() * shift_mul;
            if is_ring {
                let bits: Vec<Block128> = half_claims[ci..ci + num_claims]
                    .iter()
                    .map(|f| f.to_tower().into())
                    .collect();

                target += weight * ring_batch_b(&bits, eq_mix);
            } else {
                let c: Block128 = half_claims[ci].to_tower().into();
                target += weight * c;
            }

            ci += num_claims;
        }
    }

    target
}

/// [`ring_target`] computed in the flat basis, each
/// ring unit as `Σ_v 2^v · φ_{r''}(c_v)` with `φ_{r''}`
/// read from byte tables on the claims' raw bits.
fn ring_target_tabulated<F>(
    plan: &RingSwitchPlan,
    claims: &[Flat<F>],
    unit_weights: &[Flat<Block128>],
    shift: Flat<Block128>,
    eq_mix: &[Block128],
    shifted_claims: bool,
) -> Block128
where
    F: HardwareField + Into<Block128>,
{
    let zero = Flat::from_raw(Block128::ZERO);
    let one = Flat::from_raw(Block128::ONE);

    let eq_flat: Vec<Flat<Block128>> = eq_mix.iter().map(|m| m.to_hardware()).collect();

    let phi = LinearMap::new(|x: Flat<F>| {
        let tower: Block128 = x.to_tower().into();

        let mut acc = zero;
        for (u, &m) in eq_flat.iter().enumerate() {
            if (tower.0 >> u) & 1 == 1 {
                acc += m;
            }
        }

        acc
    });

    let basis: Vec<Flat<Block128>> = (0..BITS)
        .map(|v| Block128(1u128 << v).to_hardware())
        .collect();

    let claim_halves = if shifted_claims { 2 } else { 1 };
    let half = claims.len() / claim_halves;

    let base = [(0usize, one)];
    let base_and_shift = [(0usize, one), (half, shift)];

    let offsets: &[(usize, Flat<Block128>)] = if shifted_claims {
        &base_and_shift
    } else {
        &base
    };

    let mut target = zero;
    for &(offset, shift_mul) in offsets {
        let half_claims = &claims[offset..offset + half];

        let mut ci = 0usize;
        for (&(is_ring, num_claims), &unit_weight) in plan.units.iter().zip(unit_weights) {
            let unit_claims = &half_claims[ci..ci + num_claims];
            let value = if is_ring {
                let mut acc = zero;
                for (&c, &b) in unit_claims.iter().zip(&basis) {
                    acc += b * phi.apply(c);
                }

                acc
            } else {
                Into::<Block128>::into(unit_claims[0].to_tower()).to_hardware()
            };

            target += unit_weight * shift_mul * value;
            ci += num_claims;
        }
    }

    target.to_tower()
}

fn eta_pow<F: HardwareField>(eta: Flat<F>, exp: usize) -> Flat<F> {
    let mut acc = Flat::from_raw(F::ONE);
    for _ in 0..exp {
        acc *= eta;
    }

    acc
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

    fn mixed_plan(num_blind: usize, num_h: usize) -> RingSwitchPlan {
        let expander = VirtualExpander::new()
            .expand_bits(5, ColumnType::B32)
            .expand_bits(3, ColumnType::B64)
            .pass_through(3, ColumnType::B64)
            .reuse_pass_through(5, 3)
            .reuse_pass_through(8, 1)
            .control_bits(2)
            .pass_through(1, ColumnType::B128)
            .build()
            .unwrap();

        let mut layout = vec![ColumnType::B32; 5];
        layout.extend(repeat_n(ColumnType::B64, 6));
        layout.extend(repeat_n(ColumnType::Bit, 2));
        layout.push(ColumnType::B128);

        RingSwitchPlan::new(
            &layout,
            Some(&expander.expansion_entries()),
            num_blind,
            num_h,
        )
        .unwrap()
    }

    fn xorshift(seed: u128) -> impl FnMut() -> Block128 {
        let mut state = seed;
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;

            Block128(state)
        }
    }

    fn narrow_plan(cols: usize, config: &Config) -> RingSwitchPlan {
        RingSwitchPlan::new(&vec![ColumnType::B32; cols], None, config.blind_units(), 0).unwrap()
    }

    fn cheapest_split(
        tables: &[(&RingSwitchPlan, usize)],
        config: &Config,
        admissible: impl Fn(FoldShape) -> bool,
    ) -> Option<usize> {
        let hi = tables
            .iter()
            .map(|(plan, num_vars)| num_vars + plan.census().max_pack_vars())
            .max()
            .unwrap_or(0);

        let row_bytes = |c: usize| {
            tables
                .iter()
                .map(|&(plan, num_vars)| {
                    let at = plan.at_split(num_vars, c);

                    at.grid_rows() * (at.leaf_row_bytes() + at.h_leaf_row_bytes())
                })
                .sum()
        };

        let claim_bytes = |c: usize| {
            tables
                .iter()
                .map(|&(plan, num_vars)| {
                    plan.census()
                        .blind_claim_bytes(1 << c.saturating_sub(num_vars))
                })
                .sum()
        };

        cheapest_split_vars(1, hi, config.num_queries, row_bytes, claim_bytes, |c| {
            admissible(PoolLayout::at_split(tables, c).fold_shape())
        })
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

        for split in [4, 5, 7] {
            let at = plan.at_split(4, split);
            let (coeff_bit, coeff_whole, _) = at.slot_coeffs::<Block128>(eta);

            for s in 0..at.slot_rs.len() {
                assert!(
                    coeff_bit[s] != zero || coeff_whole[s] != zero,
                    "split {split}: committed slot {s} enters no master fold"
                );
            }
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
        assert_eq!(plan.opened_row_bytes(), bare.opened_row_bytes() + 32);

        let at = plan.at_split(3, 3);

        assert_eq!(at.leaf_row_bytes(), bare.opened_row_bytes());
        assert_eq!(at.h_leaf_row_bytes(), 32);

        let eta = Block128::from(0x9E37_79B9u128).to_hardware();
        let (coeff_bit, coeff_whole, eta_shift) = at.slot_coeffs::<Block128>(eta);
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

        assert_eq!(raw.blind_cols, 0);
        assert_eq!(raw.phys_rs.len(), layout.len());

        assert_eq!(blinded.blind_cols, 3);
        assert_eq!(blinded.whole_blinds(), 2);
        assert_eq!(blinded.phys_rs.len(), layout.len() + 3);
        assert_eq!(blinded.units.last(), Some(&(true, RING_BLIND_BITS)));
        assert_eq!(
            blinded.total_claims(),
            raw.total_claims() + 2 + RING_BLIND_BITS
        );

        assert_eq!(whole_only.blind_cols, 2);
        assert_eq!(whole_only.whole_blinds(), 2);
        assert_eq!(whole_only.units.last(), Some(&(false, 1)));

        let eta = Block128(0x2545F4914F6CDD1D_517CC1B727220A95).to_hardware();
        let at = blinded.at_split(4, 4);
        let (coeff_bit, coeff_whole, _) = at.slot_coeffs::<Block128>(eta);
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

        for split in [4, 5] {
            let at = plan.at_split(4, split);

            let claims: Vec<Block128> = (0..2 * at.plan.total_claims()).map(|_| next()).collect();
            let eta = next();
            let rho: Vec<Block128> = (0..at.pack_vars()).map(|_| next()).collect();
            let r_mix: Vec<Block128> = (0..7).map(|_| next()).collect();
            let eq_mix = eq_tensor_b(&r_mix);

            let flat: Vec<Flat<Block128>> = claims.iter().map(|c| c.to_hardware()).collect();
            let target = ring_target::<Block128>(&at, &flat, eta, &rho, &r_mix, true).unwrap();

            let mut sum = Block128::ZERO;
            for ((is_ring, weight), claim) in claim_weights::<Block128>(&at, eta, &rho, true)
                .unwrap()
                .into_iter()
                .zip(&claims)
            {
                let value = match is_ring {
                    true => ring_batch_b(&[*claim], &eq_mix),
                    false => *claim,
                };

                sum += weight * value;
            }

            assert_eq!(sum, target, "split {split}");
        }
    }

    /// Whenever a split passes `check_security`,
    /// the pool takes the cheapest split that passes.
    #[test]
    fn adaptive_split_clears_floor_whenever_any_split_can() {
        let zk = Config::prod();
        let base = Config {
            zero_knowledge: false,
            ..Config::prod()
        };

        let wide: Vec<RingSwitchPlan> = [1, 4, 16, 64, 128].into_iter().map(plan_of).collect();
        let narrow_zk: Vec<RingSwitchPlan> = (1..=3).map(|cols| narrow_plan(cols, &zk)).collect();
        let narrow_base: Vec<RingSwitchPlan> =
            (1..=3).map(|cols| narrow_plan(cols, &base)).collect();

        let mixed = mixed_plan(zk.blind_units(), 3);

        let mut cases: Vec<(&Config, Vec<(&RingSwitchPlan, usize)>)> = Vec::new();

        for plan in &wide {
            cases.extend((10..=30).map(|n| (&zk, vec![(plan, n)])));
        }

        for (plan_zk, plan_base) in narrow_zk.iter().zip(&narrow_base) {
            cases.extend((1..=16).map(|n| (&zk, vec![(plan_zk, n)])));
            cases.extend((1..=16).map(|n| (&base, vec![(plan_base, n)])));
        }

        for n in 4..=8 {
            cases.push((&zk, vec![(&mixed, 16), (&narrow_zk[2], n)]));
            cases.push((
                &zk,
                vec![(&wide[1], 12), (&narrow_zk[0], n), (&narrow_zk[1], n + 1)],
            ));
        }

        let mut filtered = 0;
        for (config, tables) in &cases {
            let heights: Vec<usize> = tables.iter().map(|&(_, n)| n).collect();
            let chosen = PoolLayout::new(tables, 128, config).split_vars;

            let passing = cheapest_split(tables, config, |shape| {
                config.check_security(128, shape).is_ok()
            });

            if let Some(best) = passing {
                assert_eq!(chosen, best, "heights {heights:?}");
            }

            if Some(chosen) != cheapest_split(tables, config, |_| true) {
                filtered += 1;
            }
        }

        assert!(filtered > 0, "the security filter never fired");
    }

    #[test]
    fn tabulated_ring_target_equals_transposed() {
        let layout = [
            ColumnType::B8,
            ColumnType::B16,
            ColumnType::B32,
            ColumnType::B64,
            ColumnType::B64,
            ColumnType::Bit,
        ];

        let expander = VirtualExpander::new()
            .expand_bits(1, ColumnType::B8)
            .expand_bits(1, ColumnType::B16)
            .expand_bits(1, ColumnType::B32)
            .expand_bits(1, ColumnType::B64)
            .pass_through(1, ColumnType::B64)
            .control_bits(1)
            .build()
            .unwrap();

        let entries = expander.expansion_entries();

        let mut state = 0x2545_f491_4f6c_dd1d_9e37_79b9_7f4a_7c15u128;

        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;

            Block128(state)
        };

        let cases = [
            (0, 0, false, 4),
            (2, 1, true, 4),
            (1, 3, false, 4),
            (1, 3, true, 5),
            (2, 3, false, 6),
        ];

        for (num_blind, num_h, shifted, split) in cases {
            let plan = RingSwitchPlan::new(&layout, Some(&entries), num_blind, num_h).unwrap();
            let at = plan.at_split(4, split);

            let halves = if shifted { 2 } else { 1 };
            let claims: Vec<Flat<Block128>> = (0..halves * at.plan.total_claims())
                .map(|_| next().to_hardware())
                .collect();

            let eta = next();
            let rho: Vec<Block128> = (0..at.pack_vars()).map(|_| next()).collect();
            let (weights, shift) = at.weights_b128(eta, &rho).unwrap();

            let r_mix: Vec<Block128> = (0..7).map(|_| next()).collect();
            let eq_mix = eq_tensor_b(&r_mix);

            assert_eq!(
                ring_target_tabulated(&at.plan, &claims, &weights, shift, &eq_mix, shifted),
                ring_target_transposed(&at.plan, &claims, &weights, shift, &eq_mix, shifted)
            );
        }
    }

    #[test]
    fn unpacked_layout_keeps_eta_walk() {
        let plans = [
            mixed_plan(1, 2),
            mixed_plan(0, 0),
            RingSwitchPlan::new(
                &keccak_physical_layout(),
                Some(&keccak_expander().expansion_entries()),
                1,
                1,
            )
            .unwrap(),
            plan_of(7),
        ];

        let eta = Block128(0x2545F4914F6CDD1D_517CC1B727220A95).to_hardware();
        let zero = Flat::from_raw(Block128::ZERO);

        for plan in &plans {
            let at = plan.at_split(6, 4);

            assert_eq!(at.plan.units, plan.units);
            assert_eq!(at.slot_rs, plan.phys_rs);
            assert_eq!(at.h_slots, plan.h_cols);
            assert_eq!(at.num_sigma, plan.num_units);
            assert_eq!(
                at.leaf_row_bytes() + at.h_leaf_row_bytes(),
                plan.opened_row_bytes()
            );

            let mut pows = vec![Flat::from_raw(Block128::ONE)];
            for _ in 0..plan.num_units {
                pows.push(pows[pows.len() - 1] * eta);
            }

            let (weights, shift) = at.unit_weights(eta, &[]).unwrap();

            assert_eq!(weights, pows[..plan.num_units]);
            assert_eq!(shift, pows[plan.num_units]);

            let (coeff_bit, coeff_whole, eta_shift) = at.slot_coeffs(eta);
            let num_phys = plan.phys_rs.len();

            for p in 0..num_phys {
                let bit = plan.phys_bit[p].iter().fold(zero, |acc, &u| acc + pows[u]);
                let whole = plan.phys_whole[p]
                    .iter()
                    .fold(zero, |acc, &u| acc + pows[u]);

                assert_eq!(coeff_bit[p], bit);
                assert_eq!(coeff_whole[p], whole);
            }

            for (k, &u) in plan.h_whole.iter().enumerate() {
                assert_eq!(coeff_whole[num_phys + k], pows[u]);
            }

            assert_eq!(eta_shift, shift);
        }
    }

    #[test]
    fn packed_layout_groups_columns_by_class() {
        let plan = mixed_plan(1, 3);

        for pack_vars in 1..=3 {
            let at = plan.at_split(5, 5 + pack_vars);
            let pack = 1 << pack_vars;
            let num_cols = at.plan.phys_rs.len();

            assert_eq!(at.plan.whole_blinds(), pack);
            assert_eq!(at.h_slots, 3usize.div_ceil(pack));

            let mut seen = vec![false; num_cols];
            let mut first = 0;

            for s in 0..at.slot_rs.len() {
                let members = at.slot_columns(s);

                assert!(!members.is_empty() && members.len() <= pack);
                assert!(members[0] >= first, "slots out of first-column order");

                first = members[0];

                for &p in members {
                    assert!(!seen[p]);
                    assert_eq!(at.plan.slot_class(p), at.plan.slot_class(members[0]));
                    assert_eq!(at.slot_rs[s], at.plan.phys_rs[p]);

                    seen[p] = true;
                }
            }

            assert!(seen.iter().all(|&s| s));

            let data_cols = num_cols - at.plan.blind_cols;
            let blind_slots: Vec<usize> = (0..at.slot_rs.len())
                .filter(|&s| at.slot_columns(s)[0] >= data_cols)
                .collect();

            let full = blind_slots
                .iter()
                .filter(|&&s| at.slot_columns(s).len() == pack)
                .count();

            assert_eq!(
                blind_slots.len(),
                2,
                "one whole blind slot, one ring blind slot"
            );
            assert_eq!(full, 1);
            assert_eq!(at.blind_salt(0), at.slot_rs.len());
        }
    }

    #[test]
    fn census_prices_built_layout() {
        let plans = [mixed_plan(1, 3), mixed_plan(0, 1), plan_of(13), plan_of(64)];

        for plan in &plans {
            let census = plan.census();

            for pack_vars in 0..=census.max_pack_vars() {
                let at = plan.at_split(9, 9 + pack_vars);
                let pack = 1 << pack_vars;

                assert_eq!(
                    census.row_bytes(pack),
                    at.leaf_row_bytes() + at.h_leaf_row_bytes()
                );
                assert_eq!(census.num_sigma(pack), at.num_sigma);
            }
        }
    }

    #[test]
    fn packed_unit_weights_factor_through_slot_coeffs() {
        let plan = mixed_plan(1, 3);
        let mut next = xorshift(0x85_5107);

        for pack_vars in 0..=3 {
            let at = plan.at_split(4, 4 + pack_vars);

            let eta = next().to_hardware();
            let rho: Vec<Flat<Block128>> = (0..pack_vars).map(|_| next().to_hardware()).collect();

            let (weights, shift) = at.unit_weights(eta, &rho).unwrap();
            let (coeff_bit, coeff_whole, eta_shift) = at.slot_coeffs(eta);

            assert_eq!(shift, eta_shift);

            let eq = TensorProduct::new(rho.clone());
            let zero = Flat::from_raw(Block128::ZERO);
            let num_slots = at.slot_rs.len();

            for s in 0..num_slots {
                for (block, &p) in at.slot_columns(s).iter().enumerate() {
                    let eq_b = eq.evaluate_at_index(block);
                    let bit = at.plan.phys_bit[p]
                        .iter()
                        .fold(zero, |acc, &u| acc + weights[u]);
                    let whole = at.plan.phys_whole[p]
                        .iter()
                        .fold(zero, |acc, &u| acc + weights[u]);

                    assert_eq!(bit, coeff_bit[s] * eq_b, "slot {s} block {block}");
                    assert_eq!(whole, coeff_whole[s] * eq_b, "slot {s} block {block}");
                }
            }

            for (k, &u) in at.plan.h_whole.iter().enumerate() {
                let slot = num_slots + k / at.pack();
                let eq_b = eq.evaluate_at_index(k % at.pack());

                assert_eq!(weights[u], coeff_whole[slot] * eq_b);
            }
        }
    }

    #[test]
    fn same_role_units_keep_distinct_exponents() {
        let plan = mixed_plan(1, 0);
        let eta = Block128(0x9E37_79B9_7F4A_7C15).to_hardware();
        let zero = Flat::from_raw(Block128::ZERO);

        for pack_vars in 0..=2 {
            let at = plan.at_split(4, 4 + pack_vars);
            let doubled = &at.plan.phys_whole[8];

            assert_eq!(doubled.len(), 2);
            assert_ne!(at.unit_sigma[doubled[0]], at.unit_sigma[doubled[1]]);

            let (_, coeff_whole, _) = at.slot_coeffs(eta);
            let slot = (0..at.slot_rs.len())
                .find(|&s| at.slot_columns(s).contains(&8))
                .unwrap();

            assert_ne!(coeff_whole[slot], zero);
        }
    }

    #[test]
    fn at_split_packs_past_widest_class() {
        let plan = mixed_plan(1, 3);
        let widest = plan.census().max_pack_vars();
        let classes = plan.census().data.len();

        let at = plan.at_split(6, 7 + widest);

        assert_eq!(at.slot_rs.len(), classes + 2);
        assert_eq!(at.h_slots, 1);
        assert_eq!(at.plan.whole_blinds(), 2 << widest);

        let single = plan_of(1).at_split(9, 10);

        assert_eq!(single.slot_rs.len(), 2);
        assert_eq!(single.h_slots, 1);
    }

    #[test]
    fn pool_of_one_table_keeps_its_weights() {
        let plan = mixed_plan(1, 3);
        let eta = xorshift(0x85_9001)().to_hardware();

        for (num_vars, split) in [(6, 4), (6, 6), (4, 6)] {
            let alone = plan.at_split(num_vars, split);
            let pool = PoolLayout::at_split(&[(&plan, num_vars)], split);

            assert!(pool.tables[0].sigma_weights(eta) == alone.sigma_weights(eta));
            assert!(pool.tables[0].slot_coeffs(eta) == alone.slot_coeffs(eta));
        }
    }

    #[test]
    fn pool_exponents_run_across_tables() {
        let (wide, narrow) = (mixed_plan(1, 3), plan_of(3));
        let eta = xorshift(0x85_9002)().to_hardware();

        let pool = PoolLayout::at_split(&[(&wide, 6), (&narrow, 4)], 5);
        let shift = eta_pow(eta, pool.num_sigma());

        let cases = [
            (&pool.tables[0], wide.at_split(6, 5), 0),
            (
                &pool.tables[1],
                narrow.at_split(4, 5),
                pool.tables[0].num_sigma,
            ),
        ];

        for (pooled, alone, base) in cases {
            let scale = eta_pow(eta, base);

            let (weights, weight_shift) = pooled.sigma_weights(eta);
            let (alone_weights, _) = alone.sigma_weights(eta);

            assert!(weight_shift == shift);
            assert!(
                weights
                    .iter()
                    .zip(&alone_weights)
                    .all(|(&w, &a)| w == scale * a)
            );

            let (bit, whole, coeff_shift) = pooled.slot_coeffs(eta);
            let (alone_bit, alone_whole, _) = alone.slot_coeffs(eta);

            assert!(coeff_shift == shift);
            assert!(bit.iter().zip(&alone_bit).all(|(&p, &a)| p == scale * a));
            assert!(
                whole
                    .iter()
                    .zip(&alone_whole)
                    .all(|(&p, &a)| p == scale * a)
            );
        }
    }

    #[test]
    fn pool_shape_charges_line_walk() {
        let (wide, narrow) = (mixed_plan(1, 3), plan_of(3));
        let pool = PoolLayout::at_split(&[(&wide, 6), (&narrow, 4)], 5);

        assert_eq!(pool.num_masters(), 3);
        assert_eq!(pool.num_vars(), 6);
        assert_eq!(pool.rho_vars(), 1);
        assert_eq!(pool.fold_shape().grid_rows, 2);
        assert_eq!(pool.fold_shape().units, pool.num_sigma() + 2);
    }

    #[test]
    fn pool_packs_short_tables_over_full_width() {
        let config = Config::prod();
        let (tall, short) = (mixed_plan(1, 3), mixed_plan(1, 3));

        let pool = PoolLayout::new(&[(&tall, 16), (&short, 8)], 128, &config);
        let split = pool.split_vars;

        assert!(split > 8 && split <= 16);
        assert_eq!(pool.tables[0].grid_rows(), 1 << (16 - split));
        assert_eq!(pool.tables[1].pack(), 1 << (split - 8));
        assert_eq!(pool.tables[1].grid_rows(), 1);
        assert!(config.check_security(128, pool.fold_shape()).is_ok());
    }

    #[test]
    fn census_prices_blind_claims_of_built_layout() {
        let plans = [
            mixed_plan(1, 3),
            mixed_plan(2, 1),
            mixed_plan(0, 1),
            plan_of(13),
        ];

        for plan in &plans {
            let census = plan.census();

            for pack_vars in 0..=4 {
                let at = plan.at_split(9, 9 + pack_vars);
                let added = at.plan.total_claims() - plan.total_claims();

                assert_eq!(
                    census.blind_claim_bytes(1 << pack_vars),
                    2 * added * ColumnType::B128.byte_size()
                );
            }
        }
    }

    #[test]
    fn total_claims_at_matches_packed_plan() {
        let plans = [
            mixed_plan(1, 3),
            mixed_plan(2, 1),
            mixed_plan(0, 1),
            plan_of(13),
        ];

        for plan in &plans {
            for split_vars in 7..=13 {
                assert_eq!(
                    plan.total_claims_at(9, split_vars),
                    plan.at_split(9, split_vars).plan.total_claims(),
                    "split {split_vars}"
                );
            }
        }
    }

    #[test]
    fn split_ignores_unabsorbed_min_security_bits() {
        let prod = Config::prod();

        let lenient = Config {
            min_security_bits: 0,
            ..Config::prod()
        };

        let strict = Config {
            min_security_bits: 128,
            ..Config::prod()
        };

        let mut bound = 0;
        for cols in [1, 4, 16, 64, 128] {
            let plan = plan_of(cols);

            for num_vars in 10..=30 {
                let tables = [(&plan, num_vars)];
                let chosen = PoolLayout::new(&tables, 128, &prod).split_vars;

                assert_eq!(PoolLayout::new(&tables, 128, &lenient).split_vars, chosen);
                assert_eq!(PoolLayout::new(&tables, 128, &strict).split_vars, chosen);

                if Some(chosen) != cheapest_split(&tables, &prod, |_| true) {
                    bound += 1;
                }
            }
        }

        assert!(bound > 0, "no split was moved by the security filter");
    }

    #[test]
    fn unit_weights_scale_with_units_not_pack() {
        let base = Config {
            zero_knowledge: false,
            ..Config::prod()
        };

        let at = narrow_plan(3, &base).at_split(0, 40);
        let rho: Vec<Flat<Block128>> = (0..40u128).map(|i| Block128(i + 3).to_hardware()).collect();

        let (weights, _) = at.unit_weights(Block128(5).to_hardware(), &rho).unwrap();

        assert_eq!(weights.len(), at.plan.num_units);
    }
}
