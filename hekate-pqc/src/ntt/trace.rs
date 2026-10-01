// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec;
use alloc::vec::Vec;
use hekate_core::errors::{self, Error};
use hekate_core::trace::{ColumnTrace, TraceBuilder};
use hekate_math::{Bit, Block16, Block32, TowerField};
use hekate_program::Air;
use hekate_program::circuit::{CircuitProgram, Col};
use subtle::{Choice, ConstantTimeGreater, ConstantTimeLess};
use zeroize::Zeroizing;

use super::layout::NttLayout;
use super::params::NttParams;
use super::{
    HALF, Mac, NttForgery, NttKind, NttSchedule, NttStep, RowPlan, Transform, add_plan,
    transform_plan,
};
use crate::gadgets::MulModCols;
use crate::utils::{
    fill_mod_add, fill_mul_mod, fill_sub_borrow_packed, flush_bit_buffer, pack_bits,
};
use crate::wiring::{N, Poly, PolyValues};

#[derive(Clone, Copy)]
struct Gadget {
    a: u32,
    b: u32,
    w: u32,
    x: u32,
    y: u32,
    s1: u32,
    add_flag: bool,
    s2: u32,
    sub_flag: bool,
    quot: u32,
    p: u32,
}

impl Gadget {
    fn eval(params: &NttParams, a: u32, b: u32, w: u32, gs: bool) -> Self {
        let x = if gs { params.sub(a, b) } else { b };

        let (quot, p) = params.divmod(w as u64 * x as u64);

        let y = if gs { b } else { p };

        Self {
            a,
            b,
            w,
            x,
            y,
            s1: params.add(a, y),
            add_flag: bool::from(!(a + y).ct_lt(&params.q)),
            s2: params.sub(a, y),
            sub_flag: bool::from(y.ct_gt(&a)),
            quot,
            p,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Shifts {
    register: u32,
    operand: u32,
    start: u32,
    addend: u32,
}

impl Shifts {
    fn at(params: &NttParams, (step, forgeries): (usize, &[NttForgery]), row: usize) -> Self {
        let mut s = Self::default();
        for f in forgeries {
            match *f {
                NttForgery::Register {
                    step: at,
                    row: r,
                    delta,
                } if (at, r) == (step, row) => s.register = params.add(s.register, delta),
                NttForgery::Operand {
                    step: at,
                    row: r,
                    delta,
                } if (at, r) == (step, row) => s.operand = params.add(s.operand, delta),
                NttForgery::Start {
                    step: at,
                    row: r,
                    delta,
                } if (at, r) == (step, row) => s.start = params.add(s.start, delta),
                NttForgery::Addend {
                    step: at,
                    row: r,
                    delta,
                } if (at, r) == (step, row) => s.addend = params.add(s.addend, delta),
                _ => {}
            }
        }

        s
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Flags {
    gs: bool,
    mac: bool,
    neg: bool,
    za: bool,
    c1: bool,
    c2: bool,
    p1: bool,
    p2: bool,
    bcont: bool,
    rcont: bool,
}

struct RowValues<'r> {
    gadget: Gadget,
    v_in1: u32,
    v_out1: u32,
    v_out2: u32,
    tw: u32,
    polys: [u16; 3],
    pos: [u16; 2],
    flags: Flags,
    sel: Option<usize>,
    selin: bool,
    regs: &'r [u32],
}

struct RowWriter<'w> {
    tb: &'w mut TraceBuilder,
    ly: &'w NttLayout,
    mm: MulModCols,
    params: NttParams,
    bits: Zeroizing<Vec<u32>>,
    row: usize,
}

impl RowWriter<'_> {
    fn write(&mut self, v: &RowValues<'_>) -> errors::Result<()> {
        let ly = self.ly;
        let bw = ly.bit_width;
        let q = self.params.q;
        let g = &v.gadget;

        self.bits.fill(0);

        let bits = &mut self.bits;

        pack_bits(bits, ly.a, g.a as u64, bw);
        pack_bits(bits, ly.b, g.b as u64, bw);
        pack_bits(bits, ly.w, g.w as u64, bw);
        pack_bits(bits, ly.x, g.x as u64, bw);
        pack_bits(bits, ly.y, g.y as u64, bw);
        pack_bits(bits, ly.s1, g.s1 as u64, bw);
        pack_bits(bits, ly.s2, g.s2 as u64, bw);

        fill_mod_add(bits, q, bw, (g.a, g.y, g.s1), g.add_flag, &ly.s1_add);
        fill_mod_add(bits, q, bw, (g.s2, g.y, g.a), g.sub_flag, &ly.s2_sub);

        fill_sub_borrow_packed(
            bits,
            ly.s2_range_result,
            ly.s2_range_borrow,
            bw,
            (q - 1) as u64,
            g.s2 as u64,
        );

        fill_mul_mod(bits, &self.mm, q, (g.w, g.x), (g.quot, g.p));

        flush_bit_buffer(&self.bits, self.tb, self.row)?;

        self.word(ly.v_in1, v.v_in1)?;
        self.word(ly.v_in2, g.b)?;
        self.word(ly.v_out1, v.v_out1)?;
        self.word(ly.v_out2, v.v_out2)?;
        self.word(ly.tw, v.tw)?;

        for r in 0..ly.registers {
            self.word(ly.reg.at(r), v.regs.get(r).copied().unwrap_or(0))?;
            self.flag(ly.sel.at(r), v.sel == Some(r))?;
            self.flag(ly.selin.at(r), v.sel == Some(r) && v.selin)?;
        }

        self.label(ly.poly1, v.polys[0])?;
        self.label(ly.poly2, v.polys[1])?;
        self.label(ly.poly3, v.polys[2])?;
        self.label(ly.pos1, v.pos[0])?;
        self.label(ly.pos2, v.pos[1])?;

        let f = v.flags;

        self.flag(ly.gs, f.gs)?;
        self.flag(ly.mac, f.mac)?;
        self.flag(ly.neg, f.neg)?;
        self.flag(ly.za, f.za)?;
        self.flag(ly.c1, f.c1)?;
        self.flag(ly.c2, f.c2)?;
        self.flag(ly.p1, f.p1)?;
        self.flag(ly.p2, f.p2)?;
        self.flag(ly.bcont, f.bcont)?;
        self.flag(ly.rcont, f.rcont)?;

        self.row += 1;

        Ok(())
    }

    fn padding(&mut self) -> errors::Result<()> {
        self.write(&RowValues {
            gadget: Gadget::eval(&self.params, 0, 0, 0, false),
            v_in1: 0,
            v_out1: 0,
            v_out2: 0,
            tw: 0,
            polys: [0; 3],
            pos: [0; 2],
            flags: Flags::default(),
            sel: None,
            selin: false,
            regs: &[],
        })
    }

    fn word(&mut self, col: Col, value: u32) -> errors::Result<()> {
        self.tb
            .set_b32(self.ly.physical(col)?, self.row, Block32::from(value))
    }

    fn label(&mut self, col: Col, value: u16) -> errors::Result<()> {
        self.tb
            .set_b16(self.ly.physical(col)?, self.row, Block16(value))
    }

    fn flag(&mut self, col: Col, on: bool) -> errors::Result<()> {
        self.tb
            .set_bit(self.ly.physical(col)?, self.row, Bit::from(on as u8))
    }
}

pub(super) fn generate<F: TowerField>(
    program: &CircuitProgram<F>,
    schedule: &NttSchedule,
    ly: &NttLayout,
    num_rows: usize,
    values: &mut PolyValues,
    forgeries: &[NttForgery],
) -> errors::Result<ColumnTrace> {
    let num_vars = num_rows.trailing_zeros() as usize;
    let mut tb = TraceBuilder::new_secret(program.column_layout(), num_vars)?;

    {
        let mut writer = RowWriter {
            tb: &mut tb,
            ly,
            mm: ly.mul_mod(),
            params: schedule.params(),
            bits: Zeroizing::new(vec![0u32; ly.num_packed]),
            row: 0,
        };

        for (step, entry) in schedule.steps.iter().enumerate() {
            match entry {
                NttStep::Transform(t) => transform(
                    &mut writer,
                    schedule,
                    (step, forgeries),
                    t,
                    &transform_plan(&schedule.params(), t),
                    values,
                )?,
                NttStep::Mac(m) => mac(&mut writer, m, values, (step, forgeries))?,
                NttStep::Add { a, b, out } => add(&mut writer, [*a, *b, *out], values)?,
            }
        }

        while writer.row < num_rows {
            writer.padding()?;
        }
    }

    Ok(tb.build())
}

fn transform(
    writer: &mut RowWriter<'_>,
    schedule: &NttSchedule,
    (step, forgeries): (usize, &[NttForgery]),
    t: &Transform,
    plan: &[RowPlan],
    values: &mut PolyValues,
) -> errors::Result<()> {
    let params = schedule.params();

    let mut f = Zeroizing::new(canonical(values, t.input, params.q)?);

    for layer in 0..params.layers {
        let (input, output) = schedule.layer_polys(step, t, layer);

        for rp in &plan[layer * HALF..(layer + 1) * HALF] {
            let gadget = Gadget::eval(&params, f[rp.pos1], f[rp.pos2], rp.tw, rp.gs);
            let out2 = if rp.gs { gadget.p } else { gadget.s2 };

            writer.write(&RowValues {
                gadget,
                v_in1: gadget.a,
                v_out1: gadget.s1,
                v_out2: out2,
                tw: rp.tw,
                polys: [input.id(), input.id(), output.id()],
                pos: [rp.pos1 as u16, rp.pos2 as u16],
                flags: Flags {
                    gs: rp.gs,
                    c1: true,
                    c2: true,
                    p1: true,
                    p2: true,
                    ..Flags::default()
                },
                sel: None,
                selin: false,
                regs: &[],
            })?;

            f[rp.pos1] = gadget.s1;
            f[rp.pos2] = out2;
        }
    }

    if t.kind == NttKind::Forward {
        return values.insert(t.output, *f);
    }

    let addend = Zeroizing::new(match t.addend {
        Some(poly) => Some(canonical(values, poly, params.q)?),
        None => None,
    });

    let first = params.layers * HALF;

    let out = scale(
        writer,
        plan[first..].iter().copied(),
        (*addend).as_ref(),
        &f,
        [
            t.addend.map_or(0, Poly::id),
            schedule.scale_input(step).id(),
            t.output.id(),
        ],
        |row| Shifts::at(&params, (step, forgeries), first + row).addend,
    )?;

    values.insert(t.output, *out)
}

fn add(
    writer: &mut RowWriter<'_>,
    [a, b, out]: [Poly; 3],
    values: &mut PolyValues,
) -> errors::Result<()> {
    let q = writer.params.q;

    let x = Zeroizing::new(canonical(values, a, q)?);
    let y = Zeroizing::new(canonical(values, b, q)?);

    let sum = scale(
        writer,
        add_plan(),
        Some(&x),
        &y,
        [a.id(), b.id(), out.id()],
        |_| 0,
    )?;

    values.insert(out, *sum)
}

fn scale(
    writer: &mut RowWriter<'_>,
    plan: impl Iterator<Item = RowPlan>,
    addend: Option<&[u32; N]>,
    input: &[u32; N],
    polys: [u16; 3],
    shift: impl Fn(usize) -> u32,
) -> errors::Result<Zeroizing<[u32; N]>> {
    let params = writer.params;
    let mut out = Zeroizing::new([0u32; N]);

    for (row, rp) in plan.enumerate() {
        let a = params.add(addend.map_or(0, |c| c[rp.pos1]), shift(row));
        let gadget = Gadget::eval(&params, a, input[rp.pos1], rp.tw, false);
        let result = if rp.neg { gadget.s2 } else { gadget.s1 };

        writer.write(&RowValues {
            gadget,
            v_in1: a,
            v_out1: result,
            v_out2: gadget.s2,
            tw: rp.tw,
            polys,
            pos: [rp.pos1 as u16, rp.pos2 as u16],
            flags: Flags {
                neg: rp.neg,
                za: rp.za,
                c1: rp.c1,
                c2: true,
                p1: true,
                ..Flags::default()
            },
            sel: None,
            selin: false,
            regs: &[],
        })?;

        out[rp.pos1] = result;
    }

    Ok(out)
}

fn mac(
    writer: &mut RowWriter<'_>,
    m: &Mac,
    values: &mut PolyValues,
    (step, forgeries): (usize, &[NttForgery]),
) -> errors::Result<()> {
    let params = writer.params;
    let (terms, outputs) = (m.terms(), m.outputs());
    let rows = m.rows();

    let mut a = Zeroizing::new(Vec::with_capacity(outputs));
    for row in &m.a {
        let mut coeffs = Zeroizing::new(Vec::with_capacity(terms));
        for &poly in row {
            coeffs.push(canonical(values, poly, params.q)?);
        }

        a.push(core::mem::take(&mut *coeffs));
    }

    let mut b = Zeroizing::new(Vec::with_capacity(terms));
    for &poly in &m.b {
        b.push(canonical(values, poly, params.q)?);
    }

    let mut regs = Zeroizing::new(vec![0u32; writer.ly.registers]);
    let mut out = Zeroizing::new(vec![[0u32; N]; outputs]);

    let mut k = 0;

    for pos in 0..N {
        for j in 0..terms {
            for i in 0..outputs {
                let shifts = Shifts::at(&params, (step, forgeries), k);

                regs[i] = params.add(regs[i], shifts.register);

                let acc = if j == 0 { 0 } else { regs[i] };
                let gadget = Gadget::eval(
                    &params,
                    params.add(acc, shifts.start),
                    params.add(b[j][pos], shifts.operand),
                    a[i][j][pos],
                    false,
                );

                let result = if m.negate[j] { gadget.s2 } else { gadget.s1 };

                writer.write(&RowValues {
                    gadget,
                    v_in1: gadget.w,
                    v_out1: result,
                    v_out2: gadget.s2,
                    tw: 0,
                    polys: [m.a[i][j].id(), m.b[j].id(), m.out[i].id()],
                    pos: [pos as u16, pos as u16],
                    flags: Flags {
                        mac: true,
                        neg: m.negate[j],
                        c1: true,
                        c2: i == 0,
                        p1: j + 1 == terms,
                        bcont: i + 1 < outputs,
                        rcont: k + 1 < rows,
                        ..Flags::default()
                    },
                    sel: Some(i),
                    selin: j > 0,
                    regs: &regs,
                })?;

                regs[i] = result;

                if j + 1 == terms {
                    out[i][pos] = result;
                }

                k += 1;
            }
        }
    }

    for (&poly, coeffs) in m.out.iter().zip(out.iter()) {
        values.insert(poly, *coeffs)?;
    }

    Ok(())
}

fn canonical(values: &PolyValues, poly: Poly, q: u32) -> errors::Result<[u32; N]> {
    let coeffs = *values.get(poly)?;
    let in_range = coeffs
        .iter()
        .fold(Choice::from(1), |acc, c| acc & c.ct_lt(&q));

    if !bool::from(in_range) {
        return Err(Error::Protocol {
            protocol: "ntt_chiplet",
            message: "input coefficient is not below q",
        });
    }

    Ok(coeffs)
}
