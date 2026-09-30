// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec;
use alloc::vec::Vec;
use hekate_core::errors::{self, Error};
use hekate_core::trace::{ColumnTrace, TraceBuilder};
use hekate_math::TowerField;
use hekate_program::Air;
use hekate_program::circuit::CircuitProgram;
use subtle::{Choice, ConstantTimeLess};
use zeroize::Zeroizing;

use super::layout::PolyArithLayout;
use super::{BaseCaseMac, PAIRS, PolyArithForgery};
use crate::ntt::NttParams;
use crate::utils::{Writer, fill_mod_add, fill_mul_mod, flush_bit_buffer, pack_bits};
use crate::wiring::{N, Poly, PolyValues};

#[derive(Clone, Copy, Default)]
struct Row {
    a: (u32, u32),
    b: (u32, u32),
    acc: (u32, u32),
    gamma: u32,
    t00: (u32, u32),
    t11: (u32, u32),
    g: (u32, u32),
    t01: (u32, u32),
    t10: (u32, u32),
    u0: (u32, bool),
    out0: (u32, bool),
    u1: (u32, bool),
    out1: (u32, bool),
}

impl Row {
    fn eval(params: &NttParams, a: (u32, u32), b: (u32, u32), acc: (u32, u32), gamma: u32) -> Self {
        let mul = |x: u32, y: u32| params.divmod(x as u64 * y as u64);
        let add = |x: u32, y: u32| (params.add(x, y), bool::from(!(x + y).ct_lt(&params.q)));

        let t00 = mul(a.0, b.0);
        let t11 = mul(a.1, b.1);
        let g = mul(t11.1, gamma);
        let t01 = mul(a.0, b.1);
        let t10 = mul(a.1, b.0);

        let u0 = add(acc.0, t00.1);
        let out0 = add(u0.0, g.1);
        let u1 = add(acc.1, t01.1);
        let out1 = add(u1.0, t10.1);

        Self {
            a,
            b,
            acc,
            gamma,
            t00,
            t11,
            g,
            t01,
            t10,
            u0,
            out0,
            u1,
            out1,
        }
    }

    fn out(&self) -> (u32, u32) {
        (self.out0.0, self.out1.0)
    }

    fn fill(&self, bits: &mut [u32], ly: &PolyArithLayout, q: u32) {
        let bw = ly.bit_width;

        for (start, value) in [
            (ly.a0, self.a.0),
            (ly.a1, self.a.1),
            (ly.b0, self.b.0),
            (ly.b1, self.b.1),
            (ly.in0, self.acc.0),
            (ly.in1, self.acc.1),
            (ly.gamma, self.gamma),
            (ly.u0, self.u0.0),
            (ly.out0, self.out0.0),
            (ly.u1, self.u1.0),
            (ly.out1, self.out1.0),
        ] {
            pack_bits(bits, start, value as u64, bw);
        }

        fill_mul_mod(bits, &ly.t00, q, (self.a.0, self.b.0), self.t00);
        fill_mul_mod(bits, &ly.t11, q, (self.a.1, self.b.1), self.t11);
        fill_mul_mod(bits, &ly.g, q, (self.t11.1, self.gamma), self.g);
        fill_mul_mod(bits, &ly.t01, q, (self.a.0, self.b.1), self.t01);
        fill_mul_mod(bits, &ly.t10, q, (self.a.1, self.b.0), self.t10);

        let adds = [
            (self.acc.0, self.t00.1, self.u0, &ly.u0_add),
            (self.u0.0, self.g.1, self.out0, &ly.out0_add),
            (self.acc.1, self.t01.1, self.u1, &ly.u1_add),
            (self.u1.0, self.t10.1, self.out1, &ly.out1_add),
        ];

        for (x, y, (sum, flag), cols) in adds {
            fill_mod_add(bits, q, bw, (x, y, sum), flag, cols);
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Pin {
    polys: [u16; 4],
    pos: [u16; 2],
    reg: Option<usize>,
    reads: bool,
    active: bool,
    btake: bool,
    seed: bool,
    emit: bool,
    bcont: bool,
    rcont: bool,
    copy: u16,
}

pub(super) fn generate<F: TowerField>(
    program: &CircuitProgram<F>,
    steps: &[BaseCaseMac],
    ly: &PolyArithLayout,
    num_rows: usize,
    values: &mut PolyValues,
    forgeries: &[PolyArithForgery],
) -> errors::Result<ColumnTrace> {
    let params = NttParams::ML_KEM;
    let gammas = params.gammas();
    let num_vars = num_rows.trailing_zeros() as usize;

    let mut tb = TraceBuilder::new_secret(program.column_layout(), num_vars)?;

    let mut bits = Zeroizing::new(vec![0u32; ly.num_packed]);
    let mut regs = Zeroizing::new(vec![(0u32, 0u32); ly.registers]);
    let mut r = 0;

    for (step, s) in steps.iter().enumerate() {
        let (terms, outputs) = (s.terms(), s.outputs());
        let rows = s.rows();

        let mut m = Zeroizing::new(Vec::with_capacity(outputs));
        for row in &s.m {
            m.push(core::mem::take(&mut *canonicals(values, row, params.q)?));
        }

        let v = canonicals(values, &s.v, params.q)?;

        let seed = match &s.seed {
            Some(polys) => canonicals(values, polys, params.q)?,
            None => Zeroizing::new(Vec::new()),
        };

        let seed_labels = s.seed.as_ref();

        let mut out = Zeroizing::new(vec![[0u32; N]; outputs]);

        for p in 0..PAIRS {
            let pair = |c: &[u32; N]| (c[2 * p], c[2 * p + 1]);

            for j in 0..terms {
                for i in 0..outputs {
                    let local = (p * terms + j) * outputs + i;

                    let (reg, operand, start) =
                        forgeries
                            .iter()
                            .fold((0, 0, 0), |(reg, operand, start), f| match *f {
                                PolyArithForgery::Register {
                                    step: at,
                                    row,
                                    delta,
                                } if (at, row) == (step, local) => {
                                    (params.add(reg, delta), operand, start)
                                }
                                PolyArithForgery::Operand {
                                    step: at,
                                    row,
                                    delta,
                                } if (at, row) == (step, local) => {
                                    (reg, params.add(operand, delta), start)
                                }
                                PolyArithForgery::Start {
                                    step: at,
                                    row,
                                    delta,
                                } if (at, row) == (step, local) => {
                                    (reg, operand, params.add(start, delta))
                                }
                                _ => (reg, operand, start),
                            });

                    regs[i].0 = params.add(regs[i].0, reg);

                    let acc = match (j, seed.get(i)) {
                        (0, Some(sd)) => pair(sd),
                        (0, None) => (0, 0),
                        _ => regs[i],
                    };

                    let b = pair(&v[j]);

                    let row = Row::eval(
                        &params,
                        pair(&m[i][j]),
                        (params.add(b.0, operand), b.1),
                        (params.add(acc.0, start), acc.1),
                        gammas[p],
                    );

                    let pin = Pin {
                        polys: [
                            s.m[i][j].id(),
                            s.v[j].id(),
                            seed_labels.map_or(0, |polys| polys[i].id()),
                            s.out[i].id(),
                        ],
                        pos: [2 * p as u16, 2 * p as u16 + 1],
                        reg: Some(i),
                        reads: j > 0,
                        active: true,
                        btake: i == 0,
                        seed: j == 0 && seed_labels.is_some(),
                        emit: j + 1 == terms,
                        bcont: i + 1 < outputs,
                        rcont: local + 1 < rows,
                        copy: match (&s.copy, i) {
                            (Some(copies), 0) => copies[j].id(),
                            _ => 0,
                        },
                    };

                    write_row(&mut tb, &mut bits, ly, r, (&row, &pin), &regs)?;

                    regs[i] = row.out();

                    if pin.emit {
                        out[i][2 * p] = row.out0.0;
                        out[i][2 * p + 1] = row.out1.0;
                    }

                    r += 1;
                }
            }
        }

        for (&poly, coeffs) in s.out.iter().zip(out.iter()) {
            values.insert(poly, *coeffs)?;
        }

        if let Some(copies) = &s.copy {
            for (&poly, coeffs) in copies.iter().zip(v.iter()) {
                values.insert(poly, *coeffs)?;
            }
        }
    }

    let padding = Row::eval(&params, (0, 0), (0, 0), (0, 0), 0);
    let zeros = vec![(0u32, 0u32); ly.registers];

    while r < num_rows {
        write_row(
            &mut tb,
            &mut bits,
            ly,
            r,
            (&padding, &Pin::default()),
            &zeros,
        )?;

        r += 1;
    }

    Ok(tb.build())
}

fn write_row(
    tb: &mut TraceBuilder,
    bits: &mut [u32],
    ly: &PolyArithLayout,
    r: usize,
    (row, pin): (&Row, &Pin),
    regs: &[(u32, u32)],
) -> errors::Result<()> {
    bits.fill(0);
    row.fill(bits, ly, NttParams::ML_KEM.q);

    flush_bit_buffer(bits, tb, r)?;

    let mut w = Writer {
        tb,
        physical: |col| ly.physical(col),
        row: r,
    };

    for (col, value) in [
        (ly.v_a0, row.a.0),
        (ly.v_a1, row.a.1),
        (ly.v_b0, row.b.0),
        (ly.v_b1, row.b.1),
        (ly.v_in0, row.acc.0),
        (ly.v_in1, row.acc.1),
        (ly.v_out0, row.out0.0),
        (ly.v_out1, row.out1.0),
        (ly.v_gamma, row.gamma),
    ] {
        w.word(col, value)?;
    }

    for (k, &(h0, h1)) in regs.iter().enumerate() {
        w.word(ly.h0.at(k), h0)?;
        w.word(ly.h1.at(k), h1)?;
        w.flag(ly.sel.at(k), pin.reg == Some(k))?;
        w.flag(ly.selin.at(k), pin.reg == Some(k) && pin.reads)?;
    }

    for (col, label) in [
        (ly.poly_a, pin.polys[0]),
        (ly.poly_b, pin.polys[1]),
        (ly.poly_s, pin.polys[2]),
        (ly.poly_o, pin.polys[3]),
        (ly.pos0, pin.pos[0]),
        (ly.pos1, pin.pos[1]),
    ] {
        w.label(col, label)?;
    }

    for (col, on) in [
        (ly.active, pin.active),
        (ly.btake, pin.btake),
        (ly.seed, pin.seed),
        (ly.emit, pin.emit),
        (ly.bcont, pin.bcont),
        (ly.rcont, pin.rcont),
    ] {
        w.flag(col, on)?;
    }

    if let Some((label, sel)) = ly.copy {
        w.label(label, pin.copy)?;
        w.flag(sel, pin.copy != 0)?;
    }

    Ok(())
}

fn canonicals(
    values: &PolyValues,
    polys: &[Poly],
    q: u32,
) -> errors::Result<Zeroizing<Vec<[u32; N]>>> {
    let mut coeffs = Zeroizing::new(Vec::with_capacity(polys.len()));
    let mut in_range = Choice::from(1);

    for &poly in polys {
        let c = *values.get(poly)?;

        in_range &= c.iter().fold(Choice::from(1), |acc, x| acc & x.ct_lt(&q));
        coeffs.push(c);
    }

    if !bool::from(in_range) {
        return Err(Error::Protocol {
            protocol: "poly_arith_chiplet",
            message: "input coefficient is not below q",
        });
    }

    Ok(coeffs)
}
