// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec;
use alloc::vec::Vec;
use core::iter::repeat_n;
use core::ops::Range;
use hekate_core::errors::Error;
use hekate_core::fold::PolyKind;
use hekate_core::poly::PolyVariant;
use hekate_core::trace::{ColumnType, Trace, TraceColumn, TraceCompatibleField};
use hekate_math::{
    Bit, Block8, Block16, Block32, Block64, Block128, Flat, HardwareField, TowerField,
};

use crate::fold_pool::UnitWeights;
use crate::linearized::{BITS, LinearMap};

pub const RING_BLIND_BITS: usize = 128;
pub const EVAL_SUMCHECK_DEGREE: usize = 2;

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

/// Ties whole unit `unit` to `width` planes of ring unit
/// `ring` from `first` on: `c = Σ_v π_v c[first + v]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaneBinding {
    pub unit: usize,
    pub ring: usize,
    pub first: usize,
    pub width: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClaimBinding {
    pub whole: usize,
    pub first: usize,
    pub width: usize,
}

/// Claims of the columns one block packs over `field`:
/// claim `m` at bit `m·w` of each cell, `w` the field's
/// width. A unit over GF(2^128) is one whole column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unit {
    pub field: ColumnType,
    pub claims: Vec<usize>,
}

enum UnitOp {
    Ring(usize, usize),
    Whole(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Home {
    Column(usize),
    Helper(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Place {
    Trace(usize),
    Helper(usize),
    Bound,
}

/// Declarative physical->virtual column expander for chiplets.
///
/// Built once per chiplet, generates `virtual_layout()`, `parse_row()`,
/// and `expand_variants()` from the same packing specification.
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
        F: TraceCompatibleField,
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

/// Maps the claimed virtual evals and the committed columns onto
/// units: a bit-expanded column's planes join a GF(2) unit, and
/// a column read only whole joins a unit over its own field.
#[derive(Clone)]
pub struct RingSwitchPlan {
    pub num_units: usize,
    pub units: Vec<Unit>,
    pub phys_rs: Vec<ColumnType>,
    pub blind_cols: usize,
    pub h_cols: usize,
    pub h_blinds: usize,

    layout: Vec<ColumnType>,
    entries: Option<Vec<ExpansionEntry>>,
    num_blind: usize,
    num_claims: usize,
    expands: bool,

    places: Vec<Place>,
    blocks: Vec<Vec<(usize, usize)>>,
    bindings: Vec<PlaneBinding>,
}

impl RingSwitchPlan {
    pub fn new(
        layout: &[ColumnType],
        entries: Option<&[ExpansionEntry]>,
        num_blind: usize,
        num_h: usize,
    ) -> Result<Self, Error> {
        Self::build(layout, entries, num_blind, num_h, false)
    }

    /// Packs aligned bit planes, and the columns read only
    /// whole and once by field, into shared blocks. Zero
    /// knowledge only: base mode checks no bindings.
    pub fn pooled(&self) -> Result<Self, Error> {
        Self::build(
            &self.layout,
            self.entries.as_deref(),
            self.num_blind,
            self.h_cols,
            true,
        )
    }

    pub fn has_subfield_units(&self) -> bool {
        self.units.iter().any(|u| u.field != ColumnType::B128)
    }

    pub fn expands_bits(&self) -> bool {
        self.expands
    }

    pub fn has_ring_blind(&self) -> bool {
        self.blind_cols > self.num_blind
    }

    pub fn whole_blinds(&self) -> usize {
        self.num_blind
    }

    pub fn total_claims(&self) -> usize {
        self.num_claims
    }

    /// Returns the polynomial and plan block that commit each unit, noise
    /// not counted; `None` for a whole unit bound to its column's planes.
    pub fn unit_blocks(&self) -> Vec<Option<(PolyKind, usize)>> {
        self.places
            .iter()
            .map(|place| match *place {
                Place::Trace(block) => Some((PolyKind::Trace, block)),
                Place::Helper(slot) => Some((PolyKind::Helper, slot)),
                Place::Bound => None,
            })
            .collect()
    }

    pub fn row_bytes(&self) -> usize {
        let trace: usize = self.phys_rs.iter().map(|ct| ct.byte_size()).sum();

        trace + self.h_cols * ColumnType::B128.byte_size()
    }

    pub fn trace_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn blocks(&self) -> &[Vec<(usize, usize)>] {
        &self.blocks
    }

    pub fn binding_claims(&self) -> Vec<ClaimBinding> {
        self.bindings
            .iter()
            .map(|b| ClaimBinding {
                whole: self.units[b.unit].claims[0],
                first: self.units[b.ring].claims[b.first],
                width: b.width,
            })
            .collect()
    }

    fn build(
        layout: &[ColumnType],
        entries: Option<&[ExpansionEntry]>,
        num_blind: usize,
        num_h: usize,
        pack: bool,
    ) -> Result<Self, Error> {
        let num_data = layout.len();
        let ops = unit_ops(layout, entries)?;

        let mut rings = vec![0usize; num_data];
        let mut reads = vec![0usize; num_data];

        for op in &ops {
            match *op {
                UnitOp::Ring(p, _) => rings[p] += 1,
                UnitOp::Whole(p) => reads[p] += 1,
            }
        }

        let expands = rings.iter().any(|&n| n > 0);
        let ring_blind = expands && num_blind > 0;
        let blind_cols = num_blind + usize::from(ring_blind);
        let num_phys = num_data + blind_cols;

        let mut b = UnitBuilder {
            units: Vec::new(),
            homes: Vec::new(),
            whole_of: vec![Vec::new(); num_data],
            slots: vec![None; num_phys],
            open: None,
            open_fields: [None; 8],
            claim: 0,
        };

        for op in ops {
            match op {
                UnitOp::Ring(p, width) => b.planes(p, width, pack && rings[p] <= 1),
                UnitOp::Whole(p) if rings[p] > 0 || layout[p] == ColumnType::B128 => b.whole(p),
                UnitOp::Whole(p) => b.member(p, layout[p], pack && reads[p] == 1),
            }
        }

        for k in 0..num_blind {
            b.push(ColumnType::B128, 1, Home::Column(num_data + k));
        }

        // A uniform L-valued unit
        if ring_blind {
            b.push(
                ColumnType::Bit,
                RING_BLIND_BITS,
                Home::Column(num_data + num_blind),
            );
        }

        let h_blinds = match num_h > 0 {
            true => num_blind,
            false => 0,
        };

        for k in 0..num_h {
            b.push(ColumnType::B128, 1, Home::Helper(h_blinds + k));
        }

        for j in 0..h_blinds {
            b.push(ColumnType::B128, 1, Home::Helper(j));
        }

        let unbound = b.units.iter().zip(&b.homes).any(|(unit, home)| {
            matches!(*home, Home::Column(p) if p < num_data
                && unit.field == ColumnType::B128
                && rings[p] == 0
                && layout[p] != ColumnType::B128)
        });

        if unbound {
            return Err(Error::Protocol {
                protocol: "ring_switch_plan",
                message: "narrow column is read whole outside a unit over its field",
            });
        }

        let mut block_of = vec![0usize; num_phys];
        let mut blocks: Vec<Vec<(usize, usize)>> = Vec::new();
        let mut placed = vec![false; num_phys];

        for p in (num_data..num_phys).chain(0..num_data) {
            if placed[p] {
                continue;
            }

            let members: Vec<(usize, usize)> = match b.slots[p] {
                Some((unit, _, _)) => (p..num_phys)
                    .filter_map(|q| match b.slots[q] {
                        Some((u, first, _)) if u == unit => Some((q, first)),
                        _ => None,
                    })
                    .collect(),
                None => vec![(p, 0)],
            };

            for &(q, _) in &members {
                block_of[q] = blocks.len();
                placed[q] = true;
            }

            blocks.push(members);
        }

        let mut bindings = Vec::new();
        for p in 0..num_data {
            if let (Some((ring, first, width)), true) = (b.slots[p], blocks[block_of[p]].len() > 1)
            {
                for &unit in &b.whole_of[p] {
                    bindings.push(PlaneBinding {
                        unit,
                        ring,
                        first,
                        width,
                    });
                }
            }
        }

        let places = b
            .units
            .iter()
            .zip(&b.homes)
            .map(|(unit, home)| match *home {
                Home::Helper(slot) => Place::Helper(slot),
                Home::Column(p) => {
                    match unit.field == ColumnType::B128 && blocks[block_of[p]].len() > 1 {
                        true => Place::Bound,
                        false => Place::Trace(block_of[p]),
                    }
                }
            })
            .collect();

        let phys_rs = layout
            .iter()
            .map(|ct| ct.rs_field())
            .chain(repeat_n(ColumnType::B128, blind_cols))
            .collect();

        Ok(Self {
            num_units: b.units.len(),
            units: b.units,
            phys_rs,
            blind_cols,
            h_cols: num_h,
            h_blinds,
            layout: layout.to_vec(),
            entries: entries.map(<[ExpansionEntry]>::to_vec),
            num_blind,
            num_claims: b.claim,
            expands,
            places,
            blocks,
            bindings,
        })
    }
}

/// `φ_K` per subfield `K` that [`ring_target`] reads,
/// tabulated on first use and indexed by the claims'
/// raw bits: variable-time, for public claims only.
pub struct RingMaps<'a, F> {
    r_mix: &'a [Block128],
    fields: Vec<(ColumnType, LinearMap<F, Block128>)>,
    basis: Vec<Flat<Block128>>,
}

impl<'a, F> RingMaps<'a, F>
where
    F: HardwareField + Into<Block128>,
{
    pub fn new(r_mix: &'a [Block128]) -> Self {
        Self {
            r_mix,
            fields: Vec::new(),
            basis: (0..BITS)
                .map(|v| Block128(1u128 << v).to_hardware())
                .collect(),
        }
    }

    fn tabulate(&mut self, plan: &RingSwitchPlan) {
        let zero = Flat::from_raw(Block128::ZERO);

        for unit in &plan.units {
            if unit.field == ColumnType::B128
                || self.fields.iter().any(|(field, _)| *field == unit.field)
            {
                continue;
            }

            let coeffs: Vec<Flat<Block128>> = field_coeffs(unit.field, self.r_mix)
                .iter()
                .map(|m| m.to_hardware())
                .collect();

            let map = LinearMap::new(|x: Flat<F>| {
                let tower: Block128 = x.to_tower().into();

                let mut acc = zero;
                for (u, &m) in coeffs.iter().enumerate() {
                    if (tower.0 >> u) & 1 == 1 {
                        acc += m;
                    }
                }

                acc
            });

            self.fields.push((unit.field, map));
        }
    }
}

struct UnitBuilder {
    units: Vec<Unit>,
    homes: Vec<Home>,
    whole_of: Vec<Vec<usize>>,
    slots: Vec<Option<(usize, usize, usize)>>,
    open: Option<usize>,
    open_fields: [Option<usize>; 8],
    claim: usize,
}

impl UnitBuilder {
    fn push(&mut self, field: ColumnType, claims: usize, home: Home) -> usize {
        self.homes.push(home);
        self.units.push(Unit {
            field,
            claims: (self.claim..self.claim + claims).collect(),
        });

        self.claim += claims;

        self.units.len() - 1
    }

    fn planes(&mut self, p: usize, width: usize, packable: bool) {
        let unit = match self.open {
            Some(u)
                if packable
                    && self.units[u].claims.len().is_multiple_of(width)
                    && self.units[u].claims.len() + width <= BITS =>
            {
                u
            }
            _ => self.push(ColumnType::Bit, 0, Home::Column(p)),
        };

        let first = self.units[unit].claims.len();

        self.units[unit]
            .claims
            .extend(self.claim..self.claim + width);

        self.claim += width;

        self.open = match packable {
            true => {
                self.slots[p] = Some((unit, first, width));

                Some(unit)
            }
            false => None,
        };
    }

    fn whole(&mut self, p: usize) {
        self.open = None;

        let unit = self.push(ColumnType::B128, 1, Home::Column(p));

        self.whole_of[p].push(unit);
    }

    fn member(&mut self, p: usize, field: ColumnType, packable: bool) {
        self.open = None;

        let width = field_bits(field);
        let slot = width.trailing_zeros() as usize;

        let unit = match self.open_fields[slot] {
            Some(u) if packable && (self.units[u].claims.len() + 1) * width <= BITS => u,
            _ => self.push(field, 0, Home::Column(p)),
        };

        let first = self.units[unit].claims.len() * width;

        self.units[unit].claims.push(self.claim);

        self.claim += 1;

        if packable {
            self.slots[p] = Some((unit, first, width));
            self.open_fields[slot] = Some(unit);
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

fn expand_packed_bit<F: TraceCompatibleField>(
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

fn expand_pass_through<F: TraceCompatibleField>(
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

/// `m_u` with `φ_K(x) = Σ_u m_u·bit_u(x)` over the tower bits,
/// `K` the field of `field`: `eq(r'', ⌊u/w⌋)·π_{u mod w}`
/// on the first `log2(128/w)` coordinates of `r''`.
pub fn field_coeffs(field: ColumnType, r_mix: &[Block128]) -> Vec<Block128> {
    let width = field_bits(field);
    let eq = eq_tensor_b(&r_mix[..mix_vars(field)]);

    (0..BITS)
        .map(|u| eq[u / width] * Block128(1u128 << (u % width)))
        .collect()
}

/// Reconstructs the sumcheck's initial claim from the claimed
/// virtual evals, in the tower basis: under its layout weight
/// `a` a unit over `K` adds `a·Σ_m π_{m·w}·φ_K(c_m)`.
pub fn ring_target<F>(
    plan: &RingSwitchPlan,
    weights: &UnitWeights<Block128>,
    claims: &[Flat<F>],
    maps: &mut RingMaps<'_, F>,
    shifted_claims: bool,
) -> Block128
where
    F: HardwareField + Into<Block128>,
{
    maps.tabulate(plan);

    let zero = Flat::from_raw(Block128::ZERO);
    let one = Flat::from_raw(Block128::ONE);

    let half = plan.total_claims();

    let base = [(0usize, one)];
    let base_and_shift = [(0usize, one), (half, weights.shift)];

    let offsets: &[(usize, Flat<Block128>)] = if shifted_claims {
        &base_and_shift
    } else {
        &base
    };

    let mut target = zero;
    for &(offset, shift_mul) in offsets {
        for (unit, &unit_weight) in plan.units.iter().zip(&weights.units) {
            let width = field_bits(unit.field);
            let map = maps
                .fields
                .iter()
                .find(|(field, _)| *field == unit.field)
                .map(|(_, map)| map);

            let mut value = zero;
            for (m, &c) in unit.claims.iter().enumerate() {
                let claim = claims[offset + c];
                let image = match map {
                    Some(map) => map.apply(claim),
                    None => Into::<Block128>::into(claim.to_tower()).to_hardware(),
                };

                value += maps.basis[m * width] * image;
            }

            target += unit_weight * shift_mul * value;
        }
    }

    target.to_tower()
}

/// Per-claim `(field, a_c)` with `ring_target = Σ_c a_c·φ_K(c_c)`,
/// `K` the field of the claim's unit, `φ` the identity on GF(2^128).
pub fn claim_weights(
    plan: &RingSwitchPlan,
    unit_weights: &UnitWeights<Block128>,
    shifted_claims: bool,
) -> Vec<(ColumnType, Block128)> {
    let shifts: &[Block128] = if shifted_claims {
        &[Block128::ONE, unit_weights.shift.to_tower()]
    } else {
        &[Block128::ONE]
    };

    let half = plan.total_claims();

    let mut weights = vec![(ColumnType::B128, Block128::ZERO); half * shifts.len()];
    for (s, &shift_mul) in shifts.iter().enumerate() {
        for (unit, unit_weight) in plan.units.iter().zip(&unit_weights.units) {
            let weight = unit_weight.to_tower() * shift_mul;
            let width = field_bits(unit.field);

            for (m, &c) in unit.claims.iter().enumerate() {
                weights[s * half + c] = (unit.field, weight * Block128(1u128 << (m * width)));
            }
        }
    }

    weights
}

pub fn field_bits(field: ColumnType) -> usize {
    match field {
        ColumnType::Bit => 1,
        ColumnType::B8 => 8,
        ColumnType::B16 => 16,
        ColumnType::B32 => 32,
        ColumnType::B64 => 64,
        ColumnType::B128 => BITS,
    }
}

fn unit_ops(
    layout: &[ColumnType],
    entries: Option<&[ExpansionEntry]>,
) -> Result<Vec<UnitOp>, Error> {
    let num_phys = layout.len();

    let Some(entries) = entries else {
        return Ok((0..num_phys).map(UnitOp::Whole).collect());
    };

    let covers = |start: usize, count: usize, storage: ColumnType| -> Result<Range<usize>, Error> {
        let end = start
            .checked_add(count)
            .filter(|&end| end <= num_phys)
            .ok_or(Error::Protocol {
                protocol: "ring_switch_plan",
                message: "expansion entry exceeds the physical column layout",
            })?;

        if layout[start..end].iter().any(|&ct| ct != storage) {
            return Err(Error::Protocol {
                protocol: "ring_switch_plan",
                message: "expansion entry storage differs from the column layout",
            });
        }

        Ok(start..end)
    };

    let mut ops = Vec::new();
    let mut running = 0usize;

    for e in entries {
        match *e {
            ExpansionEntry::ExpandBits { count, storage } => {
                let bits = expand_bit_width(storage)?;
                let cols = covers(running, count, storage)?;

                running = cols.end;

                ops.extend(cols.map(|p| UnitOp::Ring(p, bits)));
            }
            ExpansionEntry::PassThrough { count, storage } => {
                let cols = covers(running, count, storage)?;

                running = cols.end;

                ops.extend(cols.map(UnitOp::Whole));
            }
            ExpansionEntry::ControlBits { count } => {
                let cols = covers(running, count, ColumnType::Bit)?;

                running = cols.end;

                ops.extend(cols.map(UnitOp::Whole));
            }
            ExpansionEntry::ReusePassThrough {
                phy_col_start,
                count,
                storage,
            } => {
                ops.extend(covers(phy_col_start, count, storage)?.map(UnitOp::Whole));
            }
            ExpansionEntry::ReuseExpandBits {
                phy_col_start,
                count,
                storage,
            } => {
                let bits = expand_bit_width(storage)?;
                let cols = covers(phy_col_start, count, storage)?;

                ops.extend(cols.map(|p| UnitOp::Ring(p, bits)));
            }
        }
    }

    if running != num_phys {
        return Err(Error::Protocol {
            protocol: "ring_switch_plan",
            message: "expansion entries do not cover the physical column layout",
        });
    }

    Ok(ops)
}

fn mix_vars(field: ColumnType) -> usize {
    (BITS / field_bits(field)).trailing_zeros() as usize
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

    fn xorshift(seed: u128) -> impl FnMut() -> Block128 {
        let mut state = seed;
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;

            Block128(state)
        }
    }

    /// `Σ_m π_{m·w}·φ_K(c_m)` for one unit over `field`.
    fn unit_batch(field: ColumnType, claims: &[Block128], r_mix: &[Block128]) -> Block128 {
        let width = field_bits(field);

        let mut acc = Block128::ZERO;
        for (j, &m) in eq_tensor_b(&r_mix[..mix_vars(field)]).iter().enumerate() {
            let mut shat = 0u128;
            for (k, c) in claims.iter().enumerate() {
                shat |= chunk(c.0, j, width) << (k * width);
            }

            acc += m * Block128(shat);
        }

        acc
    }

    /// [`ring_target`] in the tower basis, each unit's
    /// claim chunks transposed by [`unit_batch`].
    fn ring_target_transposed<F>(
        plan: &RingSwitchPlan,
        weights: &UnitWeights<Block128>,
        claims: &[Flat<F>],
        r_mix: &[Block128],
        shifted_claims: bool,
    ) -> Block128
    where
        F: HardwareField + Into<Block128>,
    {
        let half = plan.total_claims();

        let base = [(0usize, Block128::ONE)];
        let base_and_shift = [(0usize, Block128::ONE), (half, weights.shift.to_tower())];

        let offsets: &[(usize, Block128)] = if shifted_claims {
            &base_and_shift
        } else {
            &base
        };

        let mut values: Vec<Block128> = Vec::with_capacity(BITS);
        let mut target = Block128::ZERO;

        for &(offset, shift_mul) in offsets {
            for (unit, unit_weight) in plan.units.iter().zip(&weights.units) {
                values.clear();
                values.extend(
                    unit.claims
                        .iter()
                        .map(|&c| -> Block128 { claims[offset + c].to_tower().into() }),
                );

                target +=
                    unit_weight.to_tower() * shift_mul * unit_batch(unit.field, &values, r_mix);
            }
        }

        target
    }

    fn chunk(x: u128, j: usize, width: usize) -> u128 {
        match width {
            BITS => x,
            _ => (x >> (j * width)) & ((1u128 << width) - 1),
        }
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
        let blocks = plan.unit_blocks();

        for p in 0..plan.phys_rs.len() {
            assert!(
                blocks.contains(&Some((PolyKind::Trace, p))),
                "committed column {p} enters no unit"
            );
        }
    }

    #[test]
    fn h_units_and_blinds_trail_ring_blind() {
        let layout = [ColumnType::B32, ColumnType::B64];
        let expander = VirtualExpander::new()
            .expand_bits(1, ColumnType::B32)
            .pass_through(1, ColumnType::B64)
            .build()
            .unwrap();

        let entries = expander.expansion_entries();

        let bare = RingSwitchPlan::new(&layout, Some(&entries), 1, 0).unwrap();
        let plan = RingSwitchPlan::new(&layout, Some(&entries), 1, 2).unwrap();
        let tail = plan.num_units - 3;

        assert_eq!((plan.h_cols, plan.h_blinds), (2, 1));
        assert_eq!((bare.h_cols, bare.h_blinds), (0, 0));
        assert_eq!(plan.phys_rs, bare.phys_rs);
        assert_eq!(plan.total_claims(), bare.total_claims() + 2 + 1);
        assert!(
            plan.units[tail..]
                .iter()
                .all(|u| u.field == ColumnType::B128 && u.claims.len() == 1)
        );
        assert_eq!(plan.units[tail - 1].field, ColumnType::Bit);
        assert_eq!(plan.units[tail - 1].claims.len(), RING_BLIND_BITS);
        assert_eq!(plan.row_bytes(), bare.row_bytes() + 32);

        let n = plan.phys_rs.len();
        let blocks = plan.unit_blocks();

        assert_eq!(
            &blocks[tail..],
            &[
                Some((PolyKind::Helper, 1)),
                Some((PolyKind::Helper, 2)),
                Some((PolyKind::Helper, 0))
            ]
        );
        assert_eq!(blocks[tail - 1], Some((PolyKind::Trace, 1)));
        assert!(
            blocks[..tail]
                .iter()
                .all(|b| matches!(b, Some((PolyKind::Trace, block)) if (0..n).contains(block)))
        );
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
        assert_eq!(
            blinded.units.last().map(|u| (u.field, u.claims.len())),
            Some((ColumnType::Bit, RING_BLIND_BITS))
        );
        assert_eq!(
            blinded.total_claims(),
            raw.total_claims() + 2 + RING_BLIND_BITS
        );

        assert_eq!(whole_only.blind_cols, 2);
        assert_eq!(whole_only.whole_blinds(), 2);
        assert_eq!(
            whole_only.units.last().map(|u| (u.field, u.claims.len())),
            Some((ColumnType::B128, 1))
        );

        assert_eq!(
            blinded.unit_blocks().last(),
            Some(&Some((PolyKind::Trace, blinded.whole_blinds())))
        );
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

        let mut next = xorshift(0x9e37_79b9_7f4a_7c15_0123_4567_89ab_cdef);
        for num_h in [0, 2] {
            let plan = RingSwitchPlan::new(&layout, Some(&entries), 2, num_h).unwrap();

            let claims: Vec<Block128> = (0..2 * plan.total_claims()).map(|_| next()).collect();
            let weights = UnitWeights {
                units: (0..plan.num_units).map(|_| next().to_hardware()).collect(),
                shift: next().to_hardware(),
            };

            let r_mix: Vec<Block128> = (0..7).map(|_| next()).collect();
            let flat: Vec<Flat<Block128>> = claims.iter().map(|c| c.to_hardware()).collect();
            let target = ring_target(&plan, &weights, &flat, &mut RingMaps::new(&r_mix), true);

            let mut sum = Block128::ZERO;
            for ((field, weight), claim) in claim_weights(&plan, &weights, true)
                .into_iter()
                .zip(&claims)
            {
                sum += weight * unit_batch(field, &[*claim], &r_mix);
            }

            assert_eq!(sum, target, "{num_h} h columns");
        }
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

        let mut next = xorshift(0x2545_f491_4f6c_dd1d_9e37_79b9_7f4a_7c15);

        let cases = [
            (0, 0, false),
            (2, 1, true),
            (1, 3, false),
            (1, 3, true),
            (2, 3, false),
        ];

        for (num_blind, num_h, shifted) in cases {
            let plan = RingSwitchPlan::new(&layout, Some(&entries), num_blind, num_h).unwrap();

            for plan in [plan.pooled().unwrap(), plan] {
                let halves = if shifted { 2 } else { 1 };
                let claims: Vec<Flat<Block128>> = (0..halves * plan.total_claims())
                    .map(|_| next().to_hardware())
                    .collect();

                let weights = UnitWeights {
                    units: (0..plan.num_units).map(|_| next().to_hardware()).collect(),
                    shift: next().to_hardware(),
                };

                let r_mix: Vec<Block128> = (0..7).map(|_| next()).collect();

                assert_eq!(
                    ring_target(
                        &plan,
                        &weights,
                        &claims,
                        &mut RingMaps::new(&r_mix),
                        shifted
                    ),
                    ring_target_transposed(&plan, &weights, &claims, &r_mix, shifted)
                );
            }
        }
    }

    #[test]
    fn whole_columns_pack_by_field() {
        let layout = [
            ColumnType::B32,
            ColumnType::Bit,
            ColumnType::B32,
            ColumnType::B8,
            ColumnType::Bit,
        ];

        let units = |plan: &RingSwitchPlan| -> Vec<(ColumnType, Vec<usize>)> {
            plan.units
                .iter()
                .map(|u| (u.field, u.claims.clone()))
                .collect()
        };

        let plan = RingSwitchPlan::new(&layout, None, 1, 0).unwrap();

        assert_eq!(plan.trace_blocks(), 6);
        assert_eq!(
            units(&plan),
            [
                (ColumnType::B32, vec![0]),
                (ColumnType::Bit, vec![1]),
                (ColumnType::B32, vec![2]),
                (ColumnType::B8, vec![3]),
                (ColumnType::Bit, vec![4]),
                (ColumnType::B128, vec![5]),
            ]
        );

        let pooled = plan.pooled().unwrap();

        assert_eq!(pooled.trace_blocks(), 4);
        assert_eq!(
            units(&pooled),
            [
                (ColumnType::B32, vec![0, 2]),
                (ColumnType::Bit, vec![1, 4]),
                (ColumnType::B8, vec![3]),
                (ColumnType::B128, vec![5]),
            ]
        );
        assert!(pooled.binding_claims().is_empty());
        assert!(pooled.blocks().contains(&vec![(0, 0), (2, 32)]));
        assert!(pooled.blocks().contains(&vec![(1, 0), (4, 1)]));
    }

    #[test]
    fn pooled_keccak_pairs_lanes_and_packs_control_bits() {
        let entries = keccak_expander().expansion_entries();
        let plan = RingSwitchPlan::new(&keccak_physical_layout(), Some(&entries), 1, 0).unwrap();

        assert_eq!(plan.trace_blocks(), 30);

        let lanes: Vec<ClaimBinding> = (0..25)
            .map(|l| ClaimBinding {
                whole: 1664 + l,
                first: 64 * l,
                width: 64,
            })
            .collect();

        let pooled = plan.pooled().unwrap();

        assert_eq!(pooled.trace_blocks(), 16);
        assert_eq!(pooled.binding_claims(), lanes);
        assert!(
            pooled
                .units
                .iter()
                .any(|u| u.field == ColumnType::Bit && u.claims == [1689, 1690])
        );
    }

    #[test]
    fn entry_storage_differing_from_layout_rejected() {
        use ColumnType::{B8, B16, B32, B64, B128};
        use ExpansionEntry::*;

        let cases = [
            (B32, vec![ControlBits { count: 1 }]),
            (
                B32,
                vec![ExpandBits {
                    count: 1,
                    storage: B64,
                }],
            ),
            (
                B128,
                vec![PassThrough {
                    count: 1,
                    storage: B32,
                }],
            ),
            (
                B16,
                vec![
                    PassThrough {
                        count: 1,
                        storage: B16,
                    },
                    ReuseExpandBits {
                        phy_col_start: 0,
                        count: 1,
                        storage: B8,
                    },
                ],
            ),
            (
                B16,
                vec![
                    ExpandBits {
                        count: 1,
                        storage: B16,
                    },
                    ReusePassThrough {
                        phy_col_start: 0,
                        count: 1,
                        storage: B32,
                    },
                ],
            ),
        ];

        for (declared, entries) in cases {
            let plan = RingSwitchPlan::new(&[declared], Some(&entries), 0, 0);

            assert!(
                matches!(
                    plan,
                    Err(Error::Protocol {
                        message: "expansion entry storage differs from the column layout",
                        ..
                    })
                ),
                "{declared:?} under {entries:?}"
            );
        }
    }

    #[test]
    fn ring_maps_shared_across_plans_match_fresh() {
        let whole = RingSwitchPlan::new(&[ColumnType::B64, ColumnType::B16], None, 0, 0).unwrap();

        let expander = VirtualExpander::new()
            .expand_bits(1, ColumnType::B32)
            .build()
            .unwrap();

        let planes = RingSwitchPlan::new(
            &[ColumnType::B32],
            Some(&expander.expansion_entries()),
            0,
            0,
        )
        .unwrap();

        let mut next = xorshift(0x5eed_0f0f_1234_5678_9abc_def0_2468_ace0);
        let r_mix: Vec<Block128> = (0..7).map(|_| next()).collect();

        let mut shared = RingMaps::new(&r_mix);
        for plan in [&whole, &planes, &whole] {
            let claims: Vec<Flat<Block128>> = (0..2 * plan.total_claims())
                .map(|_| next().to_hardware())
                .collect();

            let weights = UnitWeights {
                units: (0..plan.num_units).map(|_| next().to_hardware()).collect(),
                shift: next().to_hardware(),
            };

            assert_eq!(
                ring_target(plan, &weights, &claims, &mut shared, true),
                ring_target(plan, &weights, &claims, &mut RingMaps::new(&r_mix), true)
            );
        }
    }
}
