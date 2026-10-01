// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec::Vec;
use hekate_core::errors::{Error, Result};
use hekate_core::trace::ColumnType;
use hekate_math::{HardwareField, TowerField};
use hekate_program::circuit::{Circuit, Col, ColRange};

use super::{HintShape, Kind, Kinds, ZShape, packing, widths};
use crate::mlkem;

pub(crate) const Z_WIDTH: usize = 23;
pub(crate) const BYTE: usize = 8;
pub(crate) const HINT_SLOTS: usize = 4;
pub(crate) const KEM_WIDTH: usize = 12;

pub(crate) const Q_SHIFTS: [usize; (mlkem::Q >> 1).count_ones() as usize] = {
    let mut shifts = [0; (mlkem::Q >> 1).count_ones() as usize];

    let (mut s, mut i) = (1, 0);
    while s < KEM_WIDTH {
        if (mlkem::Q >> s) & 1 == 1 {
            shifts[i] = s;
            i += 1;
        }

        s += 1;
    }

    shifts
};

#[derive(Clone, Debug)]
pub(crate) struct ZField {
    pub(crate) y: usize,
    pub(crate) lo_result: usize,
    pub(crate) lo_borrow: usize,
    pub(crate) hi_result: usize,
    pub(crate) hi_borrow: usize,
    pub(crate) z: usize,
    pub(crate) carry: usize,
    pub(crate) flag: usize,
    pub(crate) rng_result: usize,
    pub(crate) rng_borrow: usize,
}

impl ZField {
    fn alloc(shape: ZShape, alloc: &mut impl FnMut(usize) -> usize) -> Self {
        let w = shape.width;

        Self {
            y: alloc(w),
            lo_result: alloc(w),
            lo_borrow: alloc(w + 1),
            hi_result: alloc(w),
            hi_borrow: alloc(w + 1),
            z: alloc(Z_WIDTH),
            carry: alloc(Z_WIDTH + 2),
            flag: alloc(1),
            rng_result: alloc(Z_WIDTH),
            rng_borrow: alloc(Z_WIDTH + 1),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct HintSlot {
    pub(crate) s: usize,
    pub(crate) lt_result: Vec<usize>,
    pub(crate) lt_borrow: Vec<usize>,
    pub(crate) p: usize,
    pub(crate) same: usize,
    pub(crate) inc_result: usize,
    pub(crate) inc_borrow: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct HintBits {
    pub(crate) counts: usize,
    pub(crate) mono_result: Vec<usize>,
    pub(crate) mono_borrow: Vec<usize>,
    pub(crate) last_result: usize,
    pub(crate) last_borrow: usize,
    pub(crate) slots: Vec<HintSlot>,
}

impl HintBits {
    fn alloc(shape: HintShape, alloc: &mut impl FnMut(usize) -> usize) -> Self {
        let k = shape.k;

        let counts = alloc(k * BYTE);

        let mono_result = (1..k).map(|_| alloc(BYTE)).collect();
        let mono_borrow = (1..k).map(|_| alloc(BYTE + 1)).collect();

        let last_result = alloc(BYTE);
        let last_borrow = alloc(BYTE + 1);

        let slots = (0..HINT_SLOTS)
            .map(|_| HintSlot {
                s: alloc(BYTE),
                lt_result: (0..k).map(|_| alloc(BYTE)).collect(),
                lt_borrow: (0..k).map(|_| alloc(BYTE + 1)).collect(),
                p: alloc(k),
                same: alloc(1),
                inc_result: alloc(BYTE),
                inc_borrow: alloc(BYTE + 1),
            })
            .collect();

        Self {
            counts,
            mono_result,
            mono_borrow,
            last_result,
            last_borrow,
            slots,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HintCols {
    pub hp: ColRange,
    pub hm: ColRange,
    pub idx: ColRange,
    pub pad: ColRange,
    pub count_role: ColRange,
    pub slot_value: ColRange,
    pub pair: ColRange,
    pub base: Col,
    pub cont: Col,
    pub kind: Col,
}

#[derive(Clone, Debug)]
pub(crate) struct Byte12Field {
    pub(crate) b: usize,
    pub(crate) x: usize,
    pub(crate) flag: usize,
    pub(crate) carry: usize,
    pub(crate) rng_result: usize,
    pub(crate) rng_borrow: usize,
}

impl Byte12Field {
    fn alloc(alloc: &mut impl FnMut(usize) -> usize) -> Self {
        Self {
            b: alloc(KEM_WIDTH),
            x: alloc(KEM_WIDTH),
            flag: alloc(1),
            carry: alloc(KEM_WIDTH + 1),
            rng_result: alloc(KEM_WIDTH),
            rng_borrow: alloc(KEM_WIDTH + 1),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DecompressField {
    pub(crate) y: usize,
    pub(crate) r: usize,
    pub(crate) x: usize,
    pub(crate) sums: Vec<usize>,
    pub(crate) carries: Vec<usize>,
}

impl DecompressField {
    fn alloc(d: usize, alloc: &mut impl FnMut(usize) -> usize) -> Self {
        let w = d + KEM_WIDTH;
        let chains = Q_SHIFTS.len();

        Self {
            y: alloc(d),
            r: alloc(d),
            x: alloc(KEM_WIDTH),
            sums: (1..chains).map(|_| alloc(w)).collect(),
            carries: (0..chains).map(|_| alloc(w + 1)).collect(),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CompressField {
    pub(crate) x: usize,
    pub(crate) yp: usize,
    pub(crate) r: usize,
    pub(crate) sums: Vec<usize>,
    pub(crate) carries: Vec<usize>,
    pub(crate) lift: Option<(usize, usize)>,
    pub(crate) rng_result: usize,
    pub(crate) rng_borrow: usize,
}

impl CompressField {
    fn alloc(d: usize, alloc: &mut impl FnMut(usize) -> usize) -> Self {
        let v = d + KEM_WIDTH + 1;
        let chains = Q_SHIFTS.len() + 1;

        Self {
            x: alloc(KEM_WIDTH),
            yp: alloc(d + 1),
            r: alloc(KEM_WIDTH),
            sums: (1..chains).map(|_| alloc(v)).collect(),
            carries: (0..chains).map(|_| alloc(v + 1)).collect(),
            lift: (half_q() >> d != 0).then(|| (alloc(KEM_WIDTH + 1), alloc(KEM_WIDTH + 2))),
            rng_result: alloc(KEM_WIDTH),
            rng_borrow: alloc(KEM_WIDTH + 1),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Compress1Field {
    pub(crate) x: usize,
    pub(crate) lo_result: usize,
    pub(crate) lo_borrow: usize,
    pub(crate) hi_result: usize,
    pub(crate) hi_borrow: usize,
}

impl Compress1Field {
    fn alloc(alloc: &mut impl FnMut(usize) -> usize) -> Self {
        Self {
            x: alloc(KEM_WIDTH),
            lo_result: alloc(KEM_WIDTH),
            lo_borrow: alloc(KEM_WIDTH + 1),
            hi_result: alloc(KEM_WIDTH),
            hi_borrow: alloc(KEM_WIDTH + 1),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DecompressBlock {
    pub(crate) d: usize,
    pub(crate) on: Col,
    pub(crate) fields: Vec<DecompressField>,
}

#[derive(Clone, Debug)]
pub(crate) struct CompressBlock {
    pub(crate) d: usize,
    pub(crate) on: Col,
    pub(crate) fields: Vec<CompressField>,
}

#[derive(Clone, Debug)]
pub(crate) struct Compress1Block {
    pub(crate) on: Col,
    pub(crate) fields: Vec<Compress1Field>,
}

#[derive(Clone, Debug)]
pub struct CodecLayout {
    pub words: usize,
    pub coefs: usize,

    pub(crate) z: Vec<ZField>,
    pub(crate) hint: Option<HintBits>,
    pub(crate) byte12: Vec<Byte12Field>,
    pub(crate) decompress: Vec<DecompressBlock>,
    pub(crate) decompress1: Option<Col>,
    pub(crate) compress: Vec<CompressBlock>,
    pub(crate) compress1: Option<Compress1Block>,

    pub num_bits: usize,
    pub num_packed: usize,

    pub word_bits: usize,
    pub coef: ColRange,

    pub poly: Col,
    pub pos: ColRange,
    pub wstream: Col,
    pub widx: ColRange,

    pub wsel: ColRange,
    pub csel: ColRange,
    pub t1: Option<Col>,
    pub zk: Option<Col>,
    pub k12: Option<Col>,
    pub reduce: Option<Col>,
    pub hint_cols: Option<HintCols>,
    pub twin: Option<(Col, Col)>,

    pub word: ColRange,
}

impl CodecLayout {
    pub(crate) fn declare<F: TowerField + HardwareField>(
        cx: &mut Circuit<F>,
        kinds: &Kinds,
    ) -> Self {
        let words = kinds.words();
        let coefs = kinds.coefs();

        let mut next = 0usize;

        let mut alloc = |width: usize| {
            let start = next;
            next += width;

            start
        };

        let z = match kinds.z {
            Some(shape) => (0..shape.fields)
                .map(|_| ZField::alloc(shape, &mut alloc))
                .collect(),
            None => Vec::new(),
        };

        let hint = kinds.hint.map(|shape| HintBits::alloc(shape, &mut alloc));

        let byte12 = match kinds.byte12 {
            true => (0..packing(KEM_WIDTH).1)
                .map(|_| Byte12Field::alloc(&mut alloc))
                .collect(),
            false => Vec::new(),
        };

        let decompress: Vec<(usize, Vec<DecompressField>)> = widths(kinds.decompress)
            .map(|d| d as usize)
            .filter(|&d| d > 1)
            .map(|d| {
                let fields = (0..packing(d).1)
                    .map(|_| DecompressField::alloc(d, &mut alloc))
                    .collect();

                (d, fields)
            })
            .collect();

        let compress: Vec<(usize, Vec<CompressField>)> = widths(kinds.compress)
            .map(|d| d as usize)
            .filter(|&d| d > 1)
            .map(|d| {
                let fields = (0..packing(d).1)
                    .map(|_| CompressField::alloc(d, &mut alloc))
                    .collect();

                (d, fields)
            })
            .collect();

        let compress1: Option<Vec<Compress1Field>> = ((kinds.compress >> 1) & 1 == 1).then(|| {
            (0..packing(1).1)
                .map(|_| Compress1Field::alloc(&mut alloc))
                .collect()
        });

        let num_bits = next;
        let num_packed = num_bits.div_ceil(32);

        cx.expand_bits(num_packed, ColumnType::B32);

        let word_packed = cx.expand_bits(words, ColumnType::B32);
        let coef = cx.columns(coefs, ColumnType::B32);

        let poly = cx.column(ColumnType::B16);
        let pos = cx.columns(coefs, ColumnType::B16);
        let wstream = cx.column(ColumnType::B16);
        let widx = cx.columns(words, ColumnType::B16);

        let hint_labels = kinds.hint.map(|_| {
            (
                cx.columns(HINT_SLOTS, ColumnType::B16),
                cx.columns(HINT_SLOTS, ColumnType::B16),
                cx.columns(HINT_SLOTS, ColumnType::B16),
                cx.column(ColumnType::B16),
            )
        });

        let wsel = cx.columns(words, ColumnType::Bit);
        let csel = cx.columns(coefs, ColumnType::Bit);

        let t1 = kinds.t1.then(|| cx.column(ColumnType::Bit));
        let zk = kinds.z.map(|_| cx.column(ColumnType::Bit));
        let k12 = kinds.byte12.then(|| cx.column(ColumnType::Bit));
        let reduce = kinds.byte12.then(|| cx.column(ColumnType::Bit));

        let hint_cols = match (kinds.hint, hint_labels) {
            (Some(shape), Some((hp, hm, slot_value, base))) => Some(HintCols {
                hp,
                hm,
                slot_value,
                base,
                idx: cx.columns(HINT_SLOTS, ColumnType::Bit),
                pad: cx.columns(HINT_SLOTS, ColumnType::Bit),
                count_role: cx.columns(shape.k, ColumnType::Bit),
                pair: cx.columns(HINT_SLOTS, ColumnType::Bit),
                cont: cx.column(ColumnType::Bit),
                kind: cx.column(ColumnType::Bit),
            }),
            _ => None,
        };

        let decompress = decompress
            .into_iter()
            .map(|(d, fields)| DecompressBlock {
                d,
                on: cx.column(ColumnType::Bit),
                fields,
            })
            .collect();

        let decompress1 = ((kinds.decompress >> 1) & 1 == 1).then(|| cx.column(ColumnType::Bit));

        let compress = compress
            .into_iter()
            .map(|(d, fields)| CompressBlock {
                d,
                on: cx.column(ColumnType::Bit),
                fields,
            })
            .collect();

        let compress1 = compress1.map(|fields| Compress1Block {
            on: cx.column(ColumnType::Bit),
            fields,
        });

        let twin = kinds
            .twin
            .then(|| (cx.column(ColumnType::B16), cx.column(ColumnType::Bit)));

        let word = cx.reuse_pass_through(&word_packed);

        Self {
            words,
            coefs,
            z,
            hint,
            byte12,
            decompress,
            decompress1,
            compress,
            compress1,
            num_bits,
            num_packed,
            word_bits: word_packed.bits(0).start(),
            coef,
            poly,
            pos,
            wstream,
            widx,
            wsel,
            csel,
            t1,
            zk,
            k12,
            reduce,
            hint_cols,
            twin,
            word,
        }
    }

    pub fn physical(&self, col: Col) -> Result<usize> {
        let packed = self.num_packed + self.words;

        match col.index() {
            i if i >= self.word.start() => Ok(self.num_packed + i - self.word.start()),
            i if i >= 32 * packed => Ok(i - 31 * packed),
            _ => Err(Error::Protocol {
                protocol: "codec_layout",
                message: "virtual bit has no committed column of its own",
            }),
        }
    }

    pub(crate) fn word_bit(&self, bit: usize) -> usize {
        self.word_bits + bit
    }

    pub(super) fn kind_flags(&self) -> Vec<Col> {
        let decompress = self.decompress.iter().map(|b| b.on);
        let compress = self.compress.iter().map(|b| b.on);

        [self.t1, self.zk, self.k12, self.reduce, self.decompress1]
            .into_iter()
            .flatten()
            .chain(decompress)
            .chain(compress)
            .chain(self.compress1.as_ref().map(|b| b.on))
            .collect()
    }

    pub(super) fn flags_of(&self, kind: Kind) -> Vec<Col> {
        let decompress = |d: u32| {
            self.decompress
                .iter()
                .find(|b| b.d == d as usize)
                .map(|b| b.on)
        };

        let compress = |d: u32| {
            self.compress
                .iter()
                .find(|b| b.d == d as usize)
                .map(|b| b.on)
        };

        let flags = match kind {
            Kind::T1 => [self.t1, None],
            Kind::Z(_) => [self.zk, None],
            Kind::Hint(..) => [None, None],
            Kind::Decode12 { reduce: true } => [self.k12, self.reduce],
            Kind::Decode12 { reduce: false } | Kind::Encode12 => [self.k12, None],
            Kind::Decompress(1) => [self.decompress1, None],
            Kind::Decompress(d) => [decompress(d), None],
            Kind::Compress(1) => [self.compress1.as_ref().map(|b| b.on), None],
            Kind::Compress(d) => [compress(d), None],
        };

        flags.into_iter().flatten().collect()
    }
}

pub(crate) fn half_q() -> usize {
    (mlkem::Q as usize - 1) / 2
}
