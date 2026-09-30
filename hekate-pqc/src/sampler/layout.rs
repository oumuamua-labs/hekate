// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Column positions of the Sampler table: packed scratch bits per kind,
//! the window and sign lanes with their views, then the plain columns.

use alloc::vec::Vec;
use hekate_core::errors::{Error, Result};
use hekate_core::trace::ColumnType;
use hekate_math::{HardwareField, TowerField};
use hekate_program::circuit::{Circuit, Col, ColRange, PhysRange};

use super::{BallShape, Kinds, RejShape, STEP_ACCESSES, TAIL_LANES, cbd_slots};

pub(crate) const LANES: usize = 25;
pub(crate) const SEED_LANES: usize = 8;
pub(crate) const WINDOW: usize = 3;
pub(crate) const COUNT_BITS: usize = 9;
pub(crate) const BYTE: usize = 8;
pub(crate) const TIME_BITS: usize = 9;

#[derive(Clone, Debug)]
pub(crate) struct RejSlot {
    /// Bits of (q − 1) − c.
    pub(crate) result: usize,

    /// Its borrows; the top one rejects c.
    pub(crate) borrow: usize,

    /// Carries of n + u.
    pub(crate) carry: usize,

    pub(crate) u: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct RejBits {
    pub(crate) shape: RejShape,

    /// n at each slot boundary: `count(m)` before slot m.
    pub(crate) counts: usize,
    pub(crate) slots: Vec<RejSlot>,
}

impl RejBits {
    fn alloc(shape: RejShape, alloc: &mut impl FnMut(usize) -> usize) -> Self {
        let counts = alloc((shape.slots + 1) * COUNT_BITS);

        let slots = (0..shape.slots)
            .map(|_| RejSlot {
                result: alloc(shape.width),
                borrow: alloc(shape.width + 1),
                carry: alloc(COUNT_BITS),
                u: alloc(1),
            })
            .collect();

        Self {
            shape,
            counts,
            slots,
        }
    }

    /// First bit of n before slot m;
    /// m = slots is n after the last.
    pub(crate) fn count(&self, m: usize) -> usize {
        self.counts + COUNT_BITS * m
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CbdSlot {
    /// Popcount of the slot's first η bits.
    pub(crate) x: usize,

    /// Popcount of its next η bits.
    pub(crate) y: usize,

    /// Whether x < y.
    pub(crate) sign: usize,

    /// |x − y|.
    pub(crate) diff: usize,

    /// Carry of lo + a out of bit 0.
    pub(crate) carry: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct CbdBits {
    pub(crate) eta: usize,
    pub(crate) slots: Vec<CbdSlot>,
}

impl CbdBits {
    fn alloc(eta: usize, alloc: &mut impl FnMut(usize) -> usize) -> Self {
        let slots = (0..cbd_slots(eta))
            .map(|_| CbdSlot {
                x: alloc(2),
                y: alloc(2),
                sign: alloc(1),
                diff: alloc(2),
                carry: alloc(1),
            })
            .collect();

        Self { eta, slots }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SortBits {
    pub(crate) addr: usize,
    pub(crate) time: usize,
    pub(crate) write: usize,

    /// Whether the next sorted access has this address.
    pub(crate) same: usize,
    pub(crate) addr_diff: usize,

    /// Borrows of addr − next addr;
    /// the top one means the next is larger.
    pub(crate) addr_borrow: usize,
    pub(crate) time_diff: usize,

    /// Borrows of time − next time;
    /// the top one means the next is later.
    pub(crate) time_borrow: usize,
}

impl SortBits {
    fn alloc(alloc: &mut impl FnMut(usize) -> usize) -> Self {
        Self {
            addr: alloc(BYTE),
            time: alloc(TIME_BITS),
            write: alloc(1),
            same: alloc(1),
            addr_diff: alloc(BYTE),
            addr_borrow: alloc(BYTE + 1),
            time_diff: alloc(TIME_BITS),
            time_borrow: alloc(TIME_BITS + 1),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct BallBits {
    pub(crate) shape: BallShape,

    /// i before and after the row's byte.
    pub(crate) counts: usize,

    /// The candidate byte j.
    pub(crate) byte: usize,
    pub(crate) diff: usize,

    /// Borrows of i − j;
    /// the top one rejects j.
    pub(crate) borrow: usize,
    pub(crate) carry: usize,
    pub(crate) u: usize,
    pub(crate) sort: SortBits,
}

impl BallBits {
    fn alloc(shape: BallShape, alloc: &mut impl FnMut(usize) -> usize) -> Self {
        Self {
            shape,
            counts: alloc(2 * COUNT_BITS),
            byte: alloc(BYTE),
            diff: alloc(BYTE),
            borrow: alloc(BYTE + 1),
            carry: alloc(COUNT_BITS),
            u: alloc(1),
            sort: SortBits::alloc(alloc),
        }
    }

    /// First bit of i before the row's byte (m = 0) or after it (m = 1).
    pub(crate) fn count(&self, m: usize) -> usize {
        self.counts + COUNT_BITS * m
    }
}

#[derive(Clone, Debug)]
pub struct RejCols {
    pub key_poly: ColRange,
    pub key_pos: ColRange,
    pub key_value: ColRange,
    pub work: Col,
}

#[derive(Clone, Debug)]
pub struct CbdCols {
    pub pos: ColRange,
    pub value: ColRange,
    pub work: Col,
}

#[derive(Clone, Debug)]
pub struct TailCols {
    pub lanes: ColRange,
    pub capture: Col,
    pub keep: Col,
    pub straddle: Col,
}

#[derive(Clone, Debug)]
pub struct SignCols {
    /// The sign register S, whole.
    pub lane: Col,

    /// S as 64 virtual bits.
    pub bits: ColRange,

    /// ±1 from bit 0 of S, the value a step writes to `c[j]`.
    pub value: Col,
    pub capture: Col,
    pub keep: Col,
    pub shift: Col,
}

#[derive(Clone, Debug)]
pub struct MemCols {
    /// j on step rows, p on readout rows.
    pub addr: Col,

    /// Value of the read; a step row also writes it to `c[i_t]`.
    pub value: Col,

    /// Link position N + i_t on step rows, p on readout rows.
    pub pos: Col,

    /// i_t, the cell a step writes.
    pub cell: Col,

    /// Times 3t, 3t + 1 and 3t + 2 on step row t;
    /// 3τ + p in the first on readout row p.
    pub times: ColRange,
    pub step: Col,

    /// Step and readout rows, the rows whose first access reads.
    pub read: Col,
    pub readout: Col,
}

#[derive(Clone, Debug)]
pub struct SortCols {
    pub time: Col,
    pub addr: Col,
    pub value: Col,

    /// same · value, what the next read must return.
    pub carried: Col,
    pub sorted: Col,

    /// Sorted rows followed by a sorted row.
    pub link: Col,
    pub first: Col,
}

/// SampleInBall columns. A taken byte emits (key_poly, key_pos,
/// key_value) = (POLY, N + i, j); every other byte emits JUNK.
#[derive(Clone, Debug)]
pub struct BallCols {
    pub key_poly: Col,
    pub key_pos: Col,
    pub key_value: Col,
    pub work: Col,

    /// One-hot byte select on candidate rows.
    pub bytes: ColRange,
    pub sign: SignCols,
    pub mem: MemCols,
    pub sort: SortCols,
}

impl BallCols {
    /// Allocates the SampleInBall columns around the sign register,
    /// whose view `lane` and bits `bits` the wide group already holds.
    fn declare<F: TowerField + HardwareField>(
        cx: &mut Circuit<F>,
        lane: Col,
        bits: ColRange,
    ) -> Self {
        let key_poly = cx.column(ColumnType::B16);
        let key_pos = cx.column(ColumnType::B16);
        let key_value = cx.column(ColumnType::B16);
        let sign_value = cx.column(ColumnType::B32);

        let mem = MemCols {
            addr: cx.column(ColumnType::B16),
            value: cx.column(ColumnType::B32),
            pos: cx.column(ColumnType::B16),
            cell: cx.column(ColumnType::B16),
            times: cx.columns(STEP_ACCESSES, ColumnType::B16),
            step: cx.column(ColumnType::Bit),
            read: cx.column(ColumnType::Bit),
            readout: cx.column(ColumnType::Bit),
        };

        let sort = SortCols {
            time: cx.column(ColumnType::B16),
            addr: cx.column(ColumnType::B16),
            value: cx.column(ColumnType::B32),
            carried: cx.column(ColumnType::B32),
            sorted: cx.column(ColumnType::Bit),
            link: cx.column(ColumnType::Bit),
            first: cx.column(ColumnType::Bit),
        };

        Self {
            key_poly,
            key_pos,
            key_value,
            work: cx.column(ColumnType::Bit),
            bytes: cx.columns(BYTE, ColumnType::Bit),
            sign: SignCols {
                lane,
                bits,
                value: sign_value,
                capture: cx.column(ColumnType::Bit),
                keep: cx.column(ColumnType::Bit),
                shift: cx.column(ColumnType::Bit),
            },
            mem,
            sort,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SamplerLayout {
    pub(crate) rej: Option<RejBits>,
    pub(crate) cbd: Vec<CbdBits>,
    pub(crate) ball: Option<BallBits>,

    pub num_bits: usize,
    pub num_packed: usize,
    pub num_wide: usize,

    /// First virtual bit of window lane 0.
    pub window_bits: usize,
    pub state: ColRange,
    pub seed: ColRange,

    /// (lane, pad column) for each state lane with pad bytes.
    pub pads: Vec<(usize, Col)>,

    pub rej_cols: Option<RejCols>,
    pub cbd_cols: Vec<CbdCols>,
    pub tail: Option<TailCols>,
    pub ball_cols: Option<BallCols>,

    /// Output label on work, step and readout rows.
    pub poly: Col,
    pub seed_stream: Col,
    pub seed_index: ColRange,

    pub kec: Col,
    pub absorb: Col,

    /// The state holds into the next row.
    pub keep: Col,

    /// The seed register holds into the next row.
    pub seed_keep: Col,

    /// The use counters carry into the next row.
    pub carry: Col,
    pub offsets: Vec<(usize, Col)>,
    pub seed_use: ColRange,
    pub seed_load: ColRange,
    pub seed_zero: ColRange,

    pub window: ColRange,
}

impl SamplerLayout {
    /// Allocates the columns for `kinds`, the pinned
    /// pad lanes and the window offsets in use.
    pub(crate) fn declare<F: TowerField + HardwareField>(
        cx: &mut Circuit<F>,
        kinds: &Kinds,
        pad_lanes: &[usize],
        offsets: &[usize],
    ) -> Self {
        let mut next = 0usize;

        let mut alloc = |width: usize| {
            let start = next;
            next += width;

            start
        };

        let rej = kinds.rej.map(|shape| RejBits::alloc(shape, &mut alloc));
        let cbd: Vec<CbdBits> = kinds
            .etas()
            .map(|eta| CbdBits::alloc(eta, &mut alloc))
            .collect();
        let ball = kinds.ball.map(|shape| BallBits::alloc(shape, &mut alloc));

        let num_bits = next;
        let num_packed = num_bits.div_ceil(32);
        let num_wide = WINDOW + ball.is_some() as usize;

        // Virtual order: scratch bits, wide bits, the wide
        // views, then plain columns; `physical` depends on it.
        cx.expand_bits(num_packed, ColumnType::B32);

        let wide = cx.expand_bits(num_wide, ColumnType::B64);
        let wide_cols = PhysRange::from(&wide);

        let window = cx.reuse_pass_through(wide_cols.slice(0, WINDOW));
        let sign_lane = ball
            .is_some()
            .then(|| cx.reuse_pass_through(wide_cols.slice(WINDOW, 1)).at(0));

        let state = cx.columns(LANES, ColumnType::B64);
        let seed = cx.columns(kinds.seed_lanes, ColumnType::B64);
        let pad_cols = cx.columns(pad_lanes.len(), ColumnType::B64);

        let tail_lanes = kinds.eta3.then(|| cx.columns(TAIL_LANES, ColumnType::B64));

        let rej_keys = kinds.rej.map(|shape| {
            (
                cx.columns(shape.slots, ColumnType::B16),
                cx.columns(shape.slots, ColumnType::B16),
                cx.columns(shape.slots, ColumnType::B32),
            )
        });

        let cbd_keys: Vec<(ColRange, ColRange)> = cbd
            .iter()
            .map(|block| {
                let slots = block.slots.len();

                (
                    cx.columns(slots, ColumnType::B16),
                    cx.columns(slots, ColumnType::B32),
                )
            })
            .collect();

        let ball_cols = sign_lane.map(|lane| BallCols::declare(cx, lane, wide.bits(WINDOW)));

        let poly = cx.column(ColumnType::B16);
        let seed_stream = cx.column(ColumnType::B16);
        let seed_index = cx.columns(kinds.seed_lanes, ColumnType::B16);

        let kec = cx.column(ColumnType::Bit);
        let absorb = cx.column(ColumnType::Bit);
        let keep = cx.column(ColumnType::Bit);
        let seed_keep = cx.column(ColumnType::Bit);
        let carry = cx.column(ColumnType::Bit);
        let offset_cols = cx.columns(offsets.len(), ColumnType::Bit);
        let seed_use = cx.columns(kinds.seed_lanes, ColumnType::Bit);
        let seed_load = cx.columns(kinds.seed_lanes, ColumnType::Bit);
        let seed_zero = cx.columns(kinds.seed_lanes, ColumnType::Bit);

        let rej_cols = rej_keys.map(|(key_poly, key_pos, key_value)| RejCols {
            key_poly,
            key_pos,
            key_value,
            work: cx.column(ColumnType::Bit),
        });

        let cbd_cols = cbd_keys
            .into_iter()
            .map(|(pos, value)| CbdCols {
                pos,
                value,
                work: cx.column(ColumnType::Bit),
            })
            .collect();

        let tail = tail_lanes.map(|lanes| TailCols {
            lanes,
            capture: cx.column(ColumnType::Bit),
            keep: cx.column(ColumnType::Bit),
            straddle: cx.column(ColumnType::Bit),
        });

        Self {
            rej,
            cbd,
            ball,
            num_bits,
            num_packed,
            num_wide,
            window_bits: wide.bits(0).start(),
            state,
            seed,
            pads: pad_lanes
                .iter()
                .enumerate()
                .map(|(i, &lane)| (lane, pad_cols.at(i)))
                .collect(),
            rej_cols,
            cbd_cols,
            tail,
            ball_cols,
            poly,
            seed_stream,
            seed_index,
            kec,
            absorb,
            keep,
            seed_keep,
            carry,
            offsets: offsets
                .iter()
                .enumerate()
                .map(|(i, &o)| (o, offset_cols.at(i)))
                .collect(),
            seed_use,
            seed_load,
            seed_zero,
            window,
        }
    }

    /// Committed column behind `col`, for trace writes and forged cells.
    pub fn physical(&self, col: Col) -> Result<usize> {
        let (views, index) = (self.window.start(), col.index());

        // A wide column adds 63 bits and one view ahead of the plain columns
        match index {
            i if i < views => Err(Error::Protocol {
                protocol: "sampler_layout",
                message: "virtual bit has no committed column of its own",
            }),
            i if i < views + self.num_wide => Ok(self.num_packed + i - views),
            i => Ok(i - 31 * self.num_packed - 64 * self.num_wide),
        }
    }

    /// Committed column and bit of the sorted rows' write flag.
    pub fn sort_write_cell(&self) -> Result<(usize, usize)> {
        let ball = self.ball.as_ref().ok_or(Error::Protocol {
            protocol: "sampler_layout",
            message: "table holds no SampleInBall step",
        })?;

        self.bit_cell(ball.sort.write)
    }

    fn bit_cell(&self, k: usize) -> Result<(usize, usize)> {
        let wide = 32 * self.num_packed;

        match k {
            k if k < wide => Ok((k / 32, k % 32)),
            k if k < self.window.start() => {
                Ok((self.num_packed + (k - wide) / 64, (k - wide) % 64))
            }
            _ => Err(Error::Protocol {
                protocol: "sampler_layout",
                message: "column is not a virtual bit",
            }),
        }
    }
}
