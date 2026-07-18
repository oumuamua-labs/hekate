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

use alloc::vec;
use alloc::vec::Vec;
use core::iter::repeat_n;
use hekate_core::errors::Error;
use hekate_core::poly::PolyVariant;
use hekate_core::trace::{ColumnType, Trace, TraceColumn, TraceCompatibleField};
use hekate_math::{Bit, Block8, Block16, Block32, Block64, Flat, HardwareField};

/// Serializable expansion step descriptor.
#[derive(Clone, Copy, Debug)]
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
    phys_bit: Vec<Vec<usize>>,
    phys_whole: Vec<Vec<usize>>,
}

impl RingSwitchPlan {
    pub fn new(
        layout: &[ColumnType],
        entries: Option<&[ExpansionEntry]>,
        num_blind: usize,
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
            }
            None => {
                for pw in phys_whole.iter_mut().take(num_phys) {
                    pw.push(units.len());
                    units.push((false, 1));
                }
            }
        }

        for b in 0..num_blind {
            phys_whole[num_phys + b].push(units.len());
            units.push((false, 1));
        }

        let num_units = units.len();

        Ok(Self {
            units,
            phys_bit,
            phys_whole,
            phys_rs,
            num_units,
        })
    }

    pub fn has_ring(&self) -> bool {
        self.units.iter().any(|(is_ring, _)| *is_ring)
    }

    pub fn total_claims(&self) -> usize {
        self.units.iter().map(|(_, n)| n).sum()
    }

    /// Per committed column:
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

        let total = self.phys_rs.len();

        let mut coeff_bit = vec![Flat::from_raw(F::ZERO); total];
        let mut coeff_whole = vec![Flat::from_raw(F::ZERO); total];

        for p in 0..total {
            for &u in &self.phys_bit[p] {
                coeff_bit[p] += eta_pows[u];
            }

            for &u in &self.phys_whole[p] {
                coeff_whole[p] += eta_pows[u];
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use hekate_core::trace::TraceBuilder;
    use hekate_math::{Block128, TowerField};

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
}
