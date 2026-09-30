// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec;
use alloc::vec::Vec;
use hekate_core::errors::{self, Error};
use hekate_core::trace::{ColumnTrace, TraceBuilder};
use hekate_math::{Block32, TowerField};
use hekate_program::Air;
use hekate_program::circuit::Col;
use subtle::{
    Choice, ConditionallySelectable, ConstantTimeEq, ConstantTimeGreater, ConstantTimeLess,
};
use zeroize::Zeroizing;

use super::layout::{
    BYTE, Byte12Field, CodecLayout, Compress1Field, CompressField, DecompressField, HINT_SLOTS,
    HintBits, HintCols, KEM_WIDTH, Q_SHIFTS, Z_WIDTH, ZField, half_q,
};
use super::{
    CodecChiplet, CodecForgery, CodecStep, HintRole, HintShape, Kind, Kinds, T1_SHIFT, ZShape,
    hint_role,
};
use crate::mldsa::Q;
use crate::mlkem;
use crate::utils::{
    Writer, fill_add_carry_packed, fill_sub_borrow_packed, flush_bit_buffer, pack_bits, pack_one,
};
use crate::wiring::{N, PolyValues, WordValues};

struct Grid {
    words: Zeroizing<Vec<u32>>,
    coefs: Zeroizing<Vec<u32>>,
    counts: Zeroizing<Vec<u8>>,
    kinds: Vec<Option<(Kind, usize)>>,
}

pub(super) fn generate<F: TowerField>(
    chiplet: &CodecChiplet<F>,
    words: &mut WordValues,
    values: &mut PolyValues,
    checked: bool,
    forgeries: &[CodecForgery],
) -> errors::Result<(ColumnTrace, Zeroizing<Vec<[bool; N]>>)> {
    let ly = &chiplet.layout;
    let num_rows = chiplet.num_rows;

    let mut tb = TraceBuilder::new_secret(
        chiplet.program.column_layout(),
        num_rows.trailing_zeros() as usize,
    )?;

    let mut grid = Grid {
        words: Zeroizing::new(vec![0; num_rows * ly.words]),
        coefs: Zeroizing::new(vec![0; num_rows * ly.coefs]),
        counts: Zeroizing::new(vec![0; num_rows * chiplet.kinds.hint.map_or(0, |s| s.k)]),
        kinds: vec![None; num_rows],
    };

    let hint_polys = chiplet
        .steps
        .iter()
        .map(|step| match step.kind {
            Kind::Hint(shape, _) => shape.k,
            _ => 0,
        })
        .sum();

    let mut hints = Zeroizing::new(Vec::with_capacity(hint_polys));
    let mut origin = 0;

    for step in &chiplet.steps {
        for lr in 0..step.rows() {
            grid.kinds[origin + lr] = Some((step.kind, lr));
        }

        fill_step(
            step,
            origin,
            (words, values),
            (&mut grid, &mut hints),
            ly,
            (checked, forgeries),
        )?;

        write_labels(&mut tb, ly, step, origin)?;

        origin += step.rows();
    }

    let mut bits = Zeroizing::new(vec![0u32; ly.num_packed]);
    for r in 0..num_rows {
        bits.fill(0);

        let tokens = fill_row(&mut bits, ly, &chiplet.kinds, &grid, r, forgeries);
        write_row(&mut tb, &bits, ly, &grid, r, &tokens)?;
    }

    Ok((tb.build(), hints))
}

fn fill_step(
    step: &CodecStep,
    origin: usize,
    (words, values): (&mut WordValues, &mut PolyValues),
    (grid, hints): (&mut Grid, &mut Vec<[bool; N]>),
    ly: &CodecLayout,
    (checked, forgeries): (bool, &[CodecForgery]),
) -> errors::Result<()> {
    match step.kind {
        Kind::T1 => {
            decode(step, origin, words, grid, ly, values, |v, _| {
                (v << T1_SHIFT, Choice::from(1))
            })?;
        }
        Kind::Z(shape) => {
            let bound = 2 * shape.gamma1 - shape.beta;
            let valid = decode(step, origin, words, grid, ly, values, |y, (row, j)| {
                let lifted = forgeries.contains(&CodecForgery::Lifted { row, field: j });

                (
                    z_value(shape, y, lifted).0,
                    y.ct_gt(&shape.beta) & y.ct_lt(&bound),
                )
            })?;

            check(
                checked,
                valid,
                "signature z coefficient is outside the norm bound",
            )?;
        }
        Kind::Decode12 { reduce } => {
            let valid = decode(step, origin, words, grid, ly, values, |b, (row, j)| {
                let unreduced = forgeries.contains(&CodecForgery::Unreduced { row, field: j });
                let over = !b.ct_lt(&mlkem::Q) & Choice::from(!unreduced as u8);
                let x = u32::conditional_select(&b, &b.wrapping_sub(mlkem::Q), over);

                (x, !over | Choice::from(reduce as u8))
            })?;

            check(
                checked,
                valid,
                "encapsulation key coefficient is not below q",
            )?;
        }
        Kind::Decompress(d) => {
            decode(step, origin, words, grid, ly, values, |y, _| {
                (mlkem::decompress(d, y), Choice::from(1))
            })?;
        }
        Kind::Encode12 => {
            canonical(step, values, checked)?;
            encode(step, origin, words, grid, ly, values, |x, _| x)?;
        }
        Kind::Compress(d) => {
            canonical(step, values, checked)?;
            encode(
                step,
                origin,
                words,
                grid,
                ly,
                values,
                |x, (row, j)| match forgeries.contains(&CodecForgery::Borrowed { row, field: j }) {
                    true => (quotient(d as usize, x, true).0 & ((1 << d) - 1)) as u32,
                    false => mlkem::compress(d, x),
                },
            )?;
        }
        Kind::Hint(shape, _) => {
            let stream = load(step, origin, words, grid, ly)?;

            let mut bytes = Zeroizing::new(Vec::with_capacity(4 * stream.len()));
            bytes.extend(stream.iter().flat_map(|w| w.to_le_bytes()));

            if checked {
                check_hint(shape, &bytes)?;
            }

            hint_vectors(shape, &bytes, hints);

            let counts = &bytes[shape.omega..shape.omega + shape.k];

            for lr in 0..shape.rows() {
                grid.counts[(origin + lr) * shape.k..][..shape.k].copy_from_slice(counts);
            }

            for f in forgeries {
                if let CodecForgery::Count { row, poly, value } = *f
                    && (origin..origin + shape.rows()).contains(&row)
                    && poly < shape.k
                {
                    grid.counts[row * shape.k + poly] = value;
                }
            }
        }
    }

    Ok(())
}

fn fill_row(
    bits: &mut [u32],
    ly: &CodecLayout,
    kinds: &Kinds,
    grid: &Grid,
    r: usize,
    forgeries: &[CodecForgery],
) -> [(u16, u16); HINT_SLOTS] {
    let kind = grid.kinds[r].map(|(kind, _)| kind);
    let row_words = &grid.words[r * ly.words..][..ly.words];
    let row_coefs = &grid.coefs[r * ly.coefs..][..ly.coefs];

    if let Some(shape) = kinds.z {
        for (j, f) in ly.z.iter().enumerate() {
            let (y, lifted) = match kind {
                Some(Kind::Z(_)) => (
                    field(row_words, shape.width * j, shape.width),
                    forgeries.contains(&CodecForgery::Lifted { row: r, field: j }),
                ),
                _ => (shape.beta + 1, false),
            };

            fill_z(bits, f, shape, y, lifted);
        }
    }

    let tokens = match (kinds.hint, &ly.hint) {
        (Some(shape), Some(hb)) => fill_hint(bits, hb, shape, grid, ly.words, r),
        _ => [(0, 0); HINT_SLOTS],
    };

    for (j, f) in ly.byte12.iter().enumerate() {
        let (b, unreduced) = match kind {
            Some(Kind::Decode12 { .. } | Kind::Encode12) => (
                field(row_words, KEM_WIDTH * j, KEM_WIDTH),
                forgeries.contains(&CodecForgery::Unreduced { row: r, field: j }),
            ),
            _ => (0, false),
        };

        fill_byte12(bits, f, b, unreduced);
    }

    for block in &ly.decompress {
        for (j, f) in block.fields.iter().enumerate() {
            let y = match kind {
                Some(Kind::Decompress(d)) if d as usize == block.d => {
                    field(row_words, block.d * j, block.d)
                }
                _ => 0,
            };

            fill_decompress(bits, f, block.d, y);
        }
    }

    for block in &ly.compress {
        for (j, f) in block.fields.iter().enumerate() {
            let (x, borrowed) = match kind {
                Some(Kind::Compress(d)) if d as usize == block.d => (
                    row_coefs[j],
                    forgeries.contains(&CodecForgery::Borrowed { row: r, field: j }),
                ),
                _ => (0, false),
            };

            fill_compress(bits, f, block.d, x, borrowed);
        }
    }

    if let Some(block) = &ly.compress1 {
        for (j, f) in block.fields.iter().enumerate() {
            let x = match kind {
                Some(Kind::Compress(1)) => row_coefs[j],
                _ => 0,
            };

            fill_compress1(bits, f, x);
        }
    }

    tokens
}

fn write_row(
    tb: &mut TraceBuilder,
    bits: &[u32],
    ly: &CodecLayout,
    grid: &Grid,
    r: usize,
    tokens: &[(u16, u16); HINT_SLOTS],
) -> errors::Result<()> {
    flush_bit_buffer(bits, tb, r)?;

    for (i, &word) in grid.words[r * ly.words..][..ly.words].iter().enumerate() {
        tb.set_b32(ly.num_packed + i, r, Block32::from(word))?;
    }

    let mut w = Writer {
        tb,
        physical: |col| ly.physical(col),
        row: r,
    };

    for (j, &c) in grid.coefs[r * ly.coefs..][..ly.coefs].iter().enumerate() {
        w.word(ly.coef.at(j), c)?;
    }

    if let Some(hc) = &ly.hint_cols {
        for (t, &(hp, hm)) in tokens.iter().enumerate() {
            w.label(hc.hp.at(t), hp)?;
            w.label(hc.hm.at(t), hm)?;
        }
    }

    Ok(())
}

fn load<'w>(
    step: &CodecStep,
    origin: usize,
    words: &'w WordValues,
    grid: &mut Grid,
    ly: &CodecLayout,
) -> errors::Result<&'w [u32]> {
    let stream = words.get(step.words)?;
    let wpr = step.words_per_row();

    if stream.len() != step.rows() * wpr {
        return Err(Error::Protocol {
            protocol: "codec_chiplet",
            message: "word stream length does not match its step",
        });
    }

    for (r, chunk) in stream.chunks(wpr).enumerate() {
        grid.words[(origin + r) * ly.words..][..wpr].copy_from_slice(chunk);
    }

    Ok(stream)
}

fn decode(
    step: &CodecStep,
    origin: usize,
    words: &WordValues,
    grid: &mut Grid,
    ly: &CodecLayout,
    values: &mut PolyValues,
    value: impl Fn(u32, (usize, usize)) -> (u32, Choice),
) -> errors::Result<Choice> {
    let stream = load(step, origin, words, grid, ly)?;

    let (wpr, fields, width) = (step.words_per_row(), step.fields(), step.field_width());
    let per_poly = N / fields;

    let mut valid = Choice::from(1);
    for (p, &out) in step.polys.iter().enumerate() {
        let mut coeffs = Zeroizing::new([0u32; N]);
        for r in 0..per_poly {
            let lr = p * per_poly + r;
            let row = &stream[lr * wpr..][..wpr];

            for j in 0..fields {
                let (c, ok) = value(field(row, width * j, width), (origin + lr, j));

                valid &= ok;
                coeffs[fields * r + j] = c;
                grid.coefs[(origin + lr) * ly.coefs + j] = c;
            }
        }

        values.insert(out, *coeffs)?;
    }

    Ok(valid)
}

fn encode(
    step: &CodecStep,
    origin: usize,
    words: &mut WordValues,
    grid: &mut Grid,
    ly: &CodecLayout,
    values: &PolyValues,
    value: impl Fn(u32, (usize, usize)) -> u32,
) -> errors::Result<()> {
    let (wpr, fields, width) = (step.words_per_row(), step.fields(), step.field_width());
    let per_poly = N / fields;

    let mut stream = Zeroizing::new(vec![0u32; step.rows() * wpr]);
    for (p, &input) in step.polys.iter().enumerate() {
        for (i, &x) in values.get(input)?.iter().enumerate() {
            let (lr, j) = (p * per_poly + i / fields, i % fields);

            grid.coefs[(origin + lr) * ly.coefs + j] = x;

            put_field(
                &mut stream[lr * wpr..][..wpr],
                width * j,
                width,
                value(x, (origin + lr, j)),
            );
        }
    }

    for (r, chunk) in stream.chunks(wpr).enumerate() {
        grid.words[(origin + r) * ly.words..][..wpr].copy_from_slice(chunk);
    }

    if let Some(twin) = step.twin {
        words.insert(twin, stream.to_vec())?;
    }

    words.insert(step.words, core::mem::take(&mut *stream))
}

fn canonical(step: &CodecStep, values: &PolyValues, checked: bool) -> errors::Result<()> {
    let mut valid = Choice::from(1);
    for &poly in &step.polys {
        valid &= values
            .get(poly)?
            .iter()
            .fold(Choice::from(1), |acc, x| acc & x.ct_lt(&mlkem::Q));
    }

    check(checked, valid, "coefficient to encode is not below q")
}

fn write_labels(
    tb: &mut TraceBuilder,
    ly: &CodecLayout,
    step: &CodecStep,
    origin: usize,
) -> errors::Result<()> {
    let (wpr, fields) = (step.words_per_row(), step.fields());
    let flags = ly.flags_of(step.kind);

    for lr in 0..step.rows() {
        let mut w = Writer {
            tb: &mut *tb,
            physical: |col| ly.physical(col),
            row: origin + lr,
        };

        w.label(ly.wstream, step.words.id())?;

        if let (Some((stream, sel)), Some(twin)) = (ly.twin, step.twin) {
            w.label(stream, twin.id())?;
            w.flag(sel, true)?;
        }

        for i in 0..wpr {
            w.label(ly.widx.at(i), (wpr * lr + i) as u16)?;
            w.flag(ly.wsel.at(i), true)?;
        }

        match step.kind {
            Kind::Hint(shape, base) => {
                if let Some(hc) = &ly.hint_cols {
                    write_hint_labels(&mut w, hc, (shape, base), lr)?;
                }
            }
            _ => {
                let per_poly = N / fields;

                w.label(ly.poly, step.polys[lr / per_poly].id())?;

                for j in 0..fields {
                    w.label(ly.pos.at(j), (fields * (lr % per_poly) + j) as u16)?;
                    w.flag(ly.csel.at(j), true)?;
                }
            }
        }

        for &col in &flags {
            w.flag(col, true)?;
        }
    }

    Ok(())
}

fn write_hint_labels<P: Fn(Col) -> errors::Result<usize>>(
    w: &mut Writer<'_, P>,
    hc: &HintCols,
    (shape, base): (HintShape, u16),
    lr: usize,
) -> errors::Result<()> {
    w.flag(hc.kind, true)?;
    w.flag(hc.cont, lr + 1 < shape.rows())?;

    w.label(hc.base, base)?;

    for t in 0..HINT_SLOTS {
        match hint_role(shape, lr, t) {
            HintRole::Index(s) => {
                let pair = matches!(hint_role(shape, lr, t + 1), HintRole::Index(_));

                w.flag(hc.idx.at(t), true)?;
                w.flag(hc.pair.at(t), pair)?;
                w.label(hc.slot_value.at(t), s as u16)?;
            }
            HintRole::Count(i) => w.flag(hc.count_role.at(i), true)?,
            HintRole::Pad => w.flag(hc.pad.at(t), true)?,
        }
    }

    Ok(())
}

fn fill_z(bits: &mut [u32], f: &ZField, shape: ZShape, y: u32, lifted: bool) {
    let w = shape.width;
    let (z, flag) = z_value(shape, y, lifted);

    pack_bits(bits, f.y, y as u64, w);

    fill_sub_borrow_packed(
        bits,
        f.lo_result,
        f.lo_borrow,
        w,
        y as u64,
        (shape.beta + 1) as u64,
    );
    fill_sub_borrow_packed(
        bits,
        f.hi_result,
        f.hi_borrow,
        w,
        (2 * shape.gamma1 - shape.beta - 1) as u64,
        y as u64,
    );

    pack_bits(bits, f.z, z as u64, Z_WIDTH);
    pack_one(bits, f.flag, flag);

    fill_add_carry_packed(bits, f.carry, Z_WIDTH + 2, z as u64, y as u64);
    fill_sub_borrow_packed(
        bits,
        f.rng_result,
        f.rng_borrow,
        Z_WIDTH,
        (Q - 1) as u64,
        z as u64,
    );
}

fn fill_hint(
    bits: &mut [u32],
    hb: &HintBits,
    shape: HintShape,
    grid: &Grid,
    width: usize,
    row: usize,
) -> [(u16, u16); HINT_SLOTS] {
    let k = shape.k;
    let next = (row + 1) % grid.kinds.len();
    let counts = &grid.counts[row * k..][..k];
    let next_counts = &grid.counts[next * k..][..k];

    let bytes = grid.words[row * width].to_le_bytes();
    let following = grid.words[next * width] as u8;

    let (local, base) = match grid.kinds[row] {
        Some((Kind::Hint(_, base), lr)) => (Some(lr), base),
        _ => (None, 0),
    };

    for (i, &c) in counts.iter().enumerate() {
        pack_bits(bits, hb.counts + BYTE * i, c as u64, BYTE);
    }

    for i in 1..k {
        fill_sub_borrow_packed(
            bits,
            hb.mono_result[i - 1],
            hb.mono_borrow[i - 1],
            BYTE,
            counts[i] as u64,
            counts[i - 1] as u64,
        );
    }

    fill_sub_borrow_packed(
        bits,
        hb.last_result,
        hb.last_borrow,
        BYTE,
        shape.omega as u64,
        counts[k - 1] as u64,
    );

    let mut tokens = [(0, 0); HINT_SLOTS];
    for (t, slot) in hb.slots.iter().enumerate() {
        let roles = local.map(|lr| (hint_role(shape, lr, t), hint_role(shape, lr, t + 1)));

        let (s, idx, pair) = match roles {
            Some((HintRole::Index(s), HintRole::Index(_))) => (s, true, true),
            Some((HintRole::Index(s), _)) => (s, true, false),
            _ => (0, false, false),
        };

        pack_bits(bits, slot.s, s as u64, BYTE);

        for (i, &c) in counts.iter().enumerate() {
            fill_sub_borrow_packed(
                bits,
                slot.lt_result[i],
                slot.lt_borrow[i],
                BYTE,
                s as u64,
                c as u64,
            );
        }

        let (after, after_counts) = match t + 1 < HINT_SLOTS {
            true => (bytes[t + 1], counts),
            false => (following, next_counts),
        };

        let p = owners(s, counts);
        let same = pair && (p & owners(s + 1, after_counts)).count_ones() % 2 == 1;

        pack_bits(bits, slot.p, p as u64, k);
        pack_one(bits, slot.same, same);

        fill_sub_borrow_packed(
            bits,
            slot.inc_result,
            slot.inc_borrow,
            BYTE,
            bytes[t] as u64,
            after as u64,
        );

        if idx {
            tokens[t] = (owner_label(p, base), bytes[t] as u16);
        }
    }

    tokens
}

fn fill_byte12(bits: &mut [u32], f: &Byte12Field, b: u32, unreduced: bool) {
    let reduced = !b.ct_lt(&mlkem::Q) & Choice::from(!unreduced as u8);
    let q = u32::conditional_select(&0, &mlkem::Q, reduced);
    let x = b - q;

    pack_bits(bits, f.b, b as u64, KEM_WIDTH);
    pack_bits(bits, f.x, x as u64, KEM_WIDTH);
    pack_one(bits, f.flag, bool::from(reduced));

    fill_add_carry_packed(bits, f.carry, KEM_WIDTH + 1, x as u64, q as u64);
    fill_sub_borrow_packed(
        bits,
        f.rng_result,
        f.rng_borrow,
        KEM_WIDTH,
        (mlkem::Q - 1) as u64,
        x as u64,
    );
}

fn fill_decompress(bits: &mut [u32], f: &DecompressField, d: usize, y: u32) {
    let w = d + KEM_WIDTH;
    let y = y as u64;

    pack_bits(bits, f.y, y, d);

    let mut acc = y + (1 << (d - 1));
    for (i, s) in Q_SHIFTS.into_iter().enumerate() {
        let term = y << s;

        fill_add_carry_packed(bits, f.carries[i], w + 1, acc, term);

        acc += term;

        if let Some(&start) = f.sums.get(i) {
            pack_bits(bits, start, acc, w);
        }
    }

    pack_bits(bits, f.r, acc & ((1 << d) - 1), d);
    pack_bits(bits, f.x, acc >> d, KEM_WIDTH);
}

fn fill_compress(bits: &mut [u32], f: &CompressField, d: usize, x: u32, borrowed: bool) {
    let v = d + KEM_WIDTH + 1;
    let q = mlkem::Q as u64;

    let (yp, r) = quotient(d, x, borrowed);

    let x = x as u64;

    pack_bits(bits, f.x, x, KEM_WIDTH);
    pack_bits(bits, f.yp, yp, d + 1);
    pack_bits(bits, f.r, r, KEM_WIDTH);

    let terms = core::iter::once(r).chain(Q_SHIFTS.into_iter().map(|s| yp << s));

    let mut acc = yp;
    for (i, term) in terms.enumerate() {
        fill_add_carry_packed(bits, f.carries[i], v + 1, acc, term);

        acc += term;

        if let Some(&start) = f.sums.get(i) {
            pack_bits(bits, start, acc, v);
        }
    }

    if let Some((sum, carry)) = f.lift {
        let high = (half_q() >> d) as u64;

        pack_bits(bits, sum, x + high, KEM_WIDTH + 1);
        fill_add_carry_packed(bits, carry, KEM_WIDTH + 2, x, high);
    }

    fill_sub_borrow_packed(bits, f.rng_result, f.rng_borrow, KEM_WIDTH, q - 1, r);
}

fn fill_compress1(bits: &mut [u32], f: &Compress1Field, x: u32) {
    let x = x as u64;

    pack_bits(bits, f.x, x, KEM_WIDTH);

    fill_sub_borrow_packed(
        bits,
        f.lo_result,
        f.lo_borrow,
        KEM_WIDTH,
        x,
        mlkem::Q.div_ceil(4) as u64,
    );
    fill_sub_borrow_packed(
        bits,
        f.hi_result,
        f.hi_borrow,
        KEM_WIDTH,
        (3 * mlkem::Q / 4) as u64,
        x,
    );
}

fn check_hint(shape: HintShape, bytes: &[u8]) -> errors::Result<()> {
    let (omega, k) = (shape.omega, shape.k);
    let counts = &bytes[omega..omega + k];

    let bounded = counts
        .windows(2)
        .fold(!(counts[k - 1] as u32).ct_gt(&(omega as u32)), |acc, c| {
            acc & !c[1].ct_lt(&c[0])
        });

    let mut increasing = Choice::from(1);
    let mut cleared = Choice::from(1);

    for s in 0..omega {
        let p = owners(s, counts);
        let same = !(p & owners(s + 1, counts)).ct_eq(&0);

        increasing &= !same | bytes[s].ct_lt(&bytes[s + 1]);
        cleared &= !p.ct_eq(&0) | bytes[s].ct_eq(&0);
    }

    let padded = bytes[omega + k..]
        .iter()
        .fold(Choice::from(1), |acc, b| acc & b.ct_eq(&0));

    check(true, bounded, "hint counts decrease or exceed omega")?;
    check(
        true,
        increasing,
        "hint positions of one polynomial are not strictly increasing",
    )?;
    check(true, cleared, "unused hint bytes are not zero")?;
    check(true, padded, "bytes past the hint encoding are not zero")
}

fn hint_vectors(shape: HintShape, bytes: &[u8], h: &mut Vec<[bool; N]>) {
    let counts = &bytes[shape.omega..shape.omega + shape.k];
    let start = h.len();

    h.resize(start + shape.k, [false; N]);

    for (s, &pos) in bytes[..shape.omega].iter().enumerate() {
        let p = owners(s, counts);

        for (i, poly) in h[start..].iter_mut().enumerate() {
            let owned = Choice::from(((p >> i) & 1) as u8);

            for (j, bit) in poly.iter_mut().enumerate() {
                *bit |= bool::from(owned & pos.ct_eq(&(j as u8)));
            }
        }
    }
}

fn owners(s: usize, counts: &[u8]) -> u32 {
    let lt = counts.iter().enumerate().fold(0u32, |acc, (i, &c)| {
        acc | (((s as u32).ct_lt(&(c as u32)).unwrap_u8() as u32) << i)
    });

    lt & !(lt << 1)
}

fn owner_label(p: u32, base: u16) -> u16 {
    (0..u32::BITS).fold(0, |acc, i| {
        let owned = Choice::from(((p >> i) & 1) as u8);

        acc ^ u16::conditional_select(&0, &(base ^ (i as u16 + 1)), owned)
    })
}

fn z_value(shape: ZShape, y: u32, lifted: bool) -> (u32, bool) {
    let flag = y.ct_gt(&shape.gamma1) | Choice::from(lifted as u8);
    let z = u32::conditional_select(&shape.gamma1.wrapping_sub(y), &(shape.gamma1 + Q - y), flag);

    (z, bool::from(flag))
}

fn quotient(d: usize, x: u32, borrowed: bool) -> (u64, u64) {
    let lhs = ((x as u64) << d) + half_q() as u64;
    let yp = mlkem::div_q(lhs).saturating_sub(borrowed as u64);

    (yp, lhs - yp * mlkem::Q as u64)
}

fn field(words: &[u32], start: usize, width: usize) -> u32 {
    (0..width).fold(0, |acc, t| {
        let b = start + t;

        acc | (((words[b / 32] >> (b % 32)) & 1) << t)
    })
}

fn put_field(words: &mut [u32], start: usize, width: usize, value: u32) {
    for t in 0..width {
        let b = start + t;

        words[b / 32] |= ((value >> t) & 1) << (b % 32);
    }
}

fn check(checked: bool, valid: Choice, message: &'static str) -> errors::Result<()> {
    match checked && !bool::from(valid) {
        true => Err(Error::Protocol {
            protocol: "codec_chiplet",
            message,
        }),
        false => Ok(()),
    }
}
