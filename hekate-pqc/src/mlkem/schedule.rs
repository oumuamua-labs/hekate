// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec;
use alloc::vec::Vec;
use hekate_math::TowerField;
use hekate_program::{CadenceSegment, FixedShape};

use super::MlKemLevel;
use crate::utils::{SolidRun, ones_pattern, push_seg, segments_shape, sponge_calls};

const N: usize = 256;
const FWD_NONFC: usize = 7 * 128;
const INV_NONFC: usize = 2 * 7 * 128 + N;
const BLOCK_FULL: usize = 107;
const BLOCK_HASH_CT: usize = 53;
const SHA3_512_RATE: usize = 72;
const SHA3_256_RATE: usize = 136;
const SHAKE_256_RATE: usize = 136;
const SHAKE_128_RATE: usize = 168;
const G_INPUT_BYTES: usize = 64;

#[derive(Clone, Copy, Debug)]
enum Run {
    Io { ct_chunks: usize, pad_chunks: usize },
    KeccakFull { blocks: usize },
    KeccakHashCt { blocks: usize },
    TagSel,
    Ram { rows: usize },
    NttRam { rows: usize },
    WBindRam { rows: usize },
    Basemul { rows: usize },
    Boundary { instances: usize },
    Compare,
}

impl Run {
    fn rows(self) -> usize {
        match self {
            Run::Io {
                ct_chunks,
                pad_chunks,
            } => ct_chunks + pad_chunks,
            Run::KeccakFull { blocks } => blocks * BLOCK_FULL,
            Run::KeccakHashCt { blocks } => blocks * BLOCK_HASH_CT,
            Run::TagSel => 1,
            Run::Ram { rows }
            | Run::NttRam { rows }
            | Run::WBindRam { rows }
            | Run::Basemul { rows } => rows,
            Run::Boundary { instances } => instances * 2 * N,
            Run::Compare => 3,
        }
    }

    fn ram_rows(self) -> usize {
        match self {
            Run::KeccakFull { blocks } => blocks * full_ram_offsets().count(),
            Run::KeccakHashCt { blocks } => blocks * hash_ct_ram_offsets().count(),
            Run::TagSel | Run::Compare => 0,
            _ => self.rows(),
        }
    }
}

/// Row layout of one decapsulation's ctrl dispatch,
/// mirroring the trace generator's phase-sorted order;
/// a diverging trace fails the fixed-column check.
#[derive(Clone, Debug)]
pub(crate) struct MlKemCtrlSchedule {
    runs: Vec<Run>,
    keccak_calls: usize,
}

pub(crate) struct MlKemCtrlShapes<F> {
    pub(crate) io: FixedShape<F>,
    pub(crate) keccak: FixedShape<F>,
    pub(crate) kec_is_output: FixedShape<F>,
    pub(crate) kec_input_ref: FixedShape<F>,
    pub(crate) kec_bind_lo: FixedShape<F>,
    pub(crate) basemul: FixedShape<F>,
    pub(crate) ntt: FixedShape<F>,
    pub(crate) w_bind: FixedShape<F>,
    pub(crate) ram: FixedShape<F>,
    pub(crate) bound_in: FixedShape<F>,
    pub(crate) bound_out: FixedShape<F>,
    pub(crate) ss_out: FixedShape<F>,
}

impl MlKemCtrlSchedule {
    pub(crate) fn for_level(level: MlKemLevel) -> Self {
        let k = level.k;
        let ct = level.ct_bytes();

        let pad = {
            let r = ct % SHA3_256_RATE;
            if r == 0 {
                SHA3_256_RATE
            } else {
                SHA3_256_RATE - r
            }
        };

        let g_calls = sponge_calls(SHA3_512_RATE, G_INPUT_BYTES, 64);
        let enc_calls = k * k * sponge_calls(SHAKE_128_RATE, 34, 3 * N)
            + k * sponge_calls(SHAKE_256_RATE, 33, 64 * level.eta1)
            + (k + 1) * sponge_calls(SHAKE_256_RATE, 33, 64 * level.eta2);
        let h_ct_calls = sponge_calls(SHA3_256_RATE, ct, 32);
        let j_calls = sponge_calls(SHAKE_256_RATE, 32 + ct, 32);

        let mut runs = vec![Run::Io {
            ct_chunks: ct / 4,
            pad_chunks: pad / 4,
        }];

        runs.push(Run::Basemul { rows: k * N });
        runs.push(Run::Ram { rows: k * N });
        runs.push(Run::Ram { rows: 2 * k * N });

        for _ in 0..k {
            runs.push(Run::NttRam { rows: N });
            runs.push(Run::WBindRam { rows: N });
        }

        runs.push(Run::NttRam {
            rows: k * FWD_NONFC + INV_NONFC,
        });
        runs.push(Run::Boundary { instances: k });

        runs.push(Run::KeccakFull { blocks: g_calls });
        runs.push(Run::TagSel);
        runs.push(Run::Ram {
            rows: G_INPUT_BYTES,
        });

        runs.push(Run::KeccakFull { blocks: enc_calls });
        runs.push(Run::Basemul {
            rows: (k * k + k) * N,
        });
        runs.push(Run::Ram { rows: k * k * N });
        runs.push(Run::Ram { rows: k * N });
        runs.push(Run::Ram { rows: 2 * k * N });

        for _ in 0..k {
            for _ in 0..k {
                runs.push(Run::NttRam { rows: N });
                runs.push(Run::WBindRam { rows: N });
            }

            runs.push(Run::Ram { rows: N });
        }

        for _ in 0..k {
            runs.push(Run::NttRam { rows: N });
            runs.push(Run::WBindRam { rows: N });
        }

        runs.push(Run::NttRam {
            rows: k * FWD_NONFC + (k + 1) * INV_NONFC,
        });
        runs.push(Run::Boundary { instances: k });

        runs.push(Run::KeccakHashCt { blocks: h_ct_calls });
        runs.push(Run::TagSel);
        runs.push(Run::KeccakFull { blocks: h_ct_calls });
        runs.push(Run::TagSel);
        runs.push(Run::KeccakFull { blocks: j_calls });
        runs.push(Run::TagSel);

        runs.push(Run::Compare);

        Self {
            runs,
            keccak_calls: g_calls + enc_calls + 2 * h_ct_calls + j_calls,
        }
    }

    #[cfg(test)]
    pub(crate) fn empty() -> Self {
        Self {
            runs: Vec::new(),
            keccak_calls: 0,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }

    pub(crate) fn keccak_calls(&self) -> usize {
        self.keccak_calls
    }

    pub(crate) fn active_rows(&self) -> usize {
        self.runs.iter().map(|run| run.rows()).sum()
    }

    pub(crate) fn ram_events(&self) -> usize {
        self.runs.iter().map(|run| run.ram_rows()).sum()
    }

    pub(crate) fn basemul_ops(&self) -> usize {
        self.runs
            .iter()
            .map(|run| match run {
                Run::Basemul { rows } => *rows,
                _ => 0,
            })
            .sum()
    }

    pub(crate) fn fixed_shapes<F: TowerField>(&self) -> MlKemCtrlShapes<F> {
        let mut io = Vec::new();
        let mut keccak = Vec::new();
        let mut kec_is_output = Vec::new();
        let mut kec_input_ref = Vec::new();
        let mut kec_bind_lo = Vec::new();
        let mut basemul = Vec::new();
        let mut ntt = Vec::new();
        let mut w_bind = Vec::new();
        let mut ram: Vec<CadenceSegment<F>> = Vec::new();
        let mut bound_in = Vec::new();
        let mut bound_out = Vec::new();

        let mut solid = SolidRun::default();
        let mut ss_out_row = 0usize;
        let mut origin = 0usize;

        for run in self.runs.iter().copied() {
            match run {
                Run::Io { ct_chunks, .. } => {
                    push_seg(&mut io, origin, 1, ct_chunks, &[F::ONE]);
                    solid.extend(origin, run.rows());
                }
                Run::KeccakFull { blocks } => {
                    solid.flush(&mut ram);

                    push_seg(
                        &mut keccak,
                        origin,
                        BLOCK_FULL,
                        blocks,
                        &ones_pattern(BLOCK_FULL, [42, BLOCK_FULL - 1]),
                    );
                    push_seg(
                        &mut kec_is_output,
                        origin,
                        BLOCK_FULL,
                        blocks,
                        &ones_pattern(BLOCK_FULL, [BLOCK_FULL - 1]),
                    );
                    push_seg(
                        &mut kec_input_ref,
                        origin,
                        BLOCK_FULL,
                        blocks,
                        &ones_pattern(BLOCK_FULL, 43..64),
                    );
                    push_seg(
                        &mut kec_bind_lo,
                        origin,
                        BLOCK_FULL,
                        blocks,
                        &ones_pattern(BLOCK_FULL, (64..106).step_by(2)),
                    );
                    push_seg(
                        &mut ram,
                        origin,
                        BLOCK_FULL,
                        blocks,
                        &ones_pattern(BLOCK_FULL, full_ram_offsets()),
                    );
                }
                Run::KeccakHashCt { blocks } => {
                    solid.flush(&mut ram);

                    push_seg(
                        &mut keccak,
                        origin,
                        BLOCK_HASH_CT,
                        blocks,
                        &ones_pattern(BLOCK_HASH_CT, [0, BLOCK_HASH_CT - 1]),
                    );
                    push_seg(
                        &mut kec_is_output,
                        origin,
                        BLOCK_HASH_CT,
                        blocks,
                        &ones_pattern(BLOCK_HASH_CT, [BLOCK_HASH_CT - 1]),
                    );
                    push_seg(
                        &mut kec_input_ref,
                        origin,
                        BLOCK_HASH_CT,
                        blocks,
                        &ones_pattern(BLOCK_HASH_CT, 1..18),
                    );
                    push_seg(
                        &mut kec_bind_lo,
                        origin,
                        BLOCK_HASH_CT,
                        blocks,
                        &ones_pattern(BLOCK_HASH_CT, (18..52).step_by(2)),
                    );
                    push_seg(
                        &mut ram,
                        origin,
                        BLOCK_HASH_CT,
                        blocks,
                        &ones_pattern(BLOCK_HASH_CT, hash_ct_ram_offsets()),
                    );
                }
                Run::TagSel => solid.flush(&mut ram),
                Run::Ram { rows } => solid.extend(origin, rows),
                Run::NttRam { rows } => {
                    push_seg(&mut ntt, origin, 1, rows, &[F::ONE]);
                    solid.extend(origin, rows);
                }
                Run::WBindRam { rows } => {
                    push_seg(&mut w_bind, origin, 1, rows, &[F::ONE]);
                    solid.extend(origin, rows);
                }
                Run::Basemul { rows } => {
                    push_seg(&mut basemul, origin, 1, rows, &[F::ONE]);
                    solid.extend(origin, rows);
                }
                Run::Boundary { instances } => {
                    for i in 0..instances {
                        let base = origin + i * 2 * N;

                        push_seg(&mut bound_in, base, 1, N, &[F::ONE]);
                        push_seg(&mut bound_out, base + N, 1, N, &[F::ONE]);
                    }

                    solid.extend(origin, run.rows());
                }
                Run::Compare => {
                    solid.flush(&mut ram);
                    ss_out_row = origin + 2;
                }
            }

            origin += run.rows();
        }

        solid.flush(&mut ram);

        MlKemCtrlShapes {
            io: segments_shape(io),
            keccak: segments_shape(keccak),
            kec_is_output: segments_shape(kec_is_output),
            kec_input_ref: segments_shape(kec_input_ref),
            kec_bind_lo: segments_shape(kec_bind_lo),
            basemul: segments_shape(basemul),
            ntt: segments_shape(ntt),
            w_bind: segments_shape(w_bind),
            ram: segments_shape(ram),
            bound_in: segments_shape(bound_in),
            bound_out: segments_shape(bound_out),
            ss_out: FixedShape::Sparse(vec![(ss_out_row, F::ONE)]),
        }
    }
}

fn full_ram_offsets() -> impl Iterator<Item = usize> + Clone {
    (0..42).chain(64..106)
}

fn hash_ct_ram_offsets() -> impl Iterator<Item = usize> + Clone {
    18..52
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlkem::MlKemLevel;

    #[test]
    fn keccak_calls_match_fips203_kat_baseline() {
        let cases = [
            (MlKemLevel::MLKEM_512, 46),
            (MlKemLevel::MLKEM_768, 80),
            (MlKemLevel::MLKEM_1024, 126),
        ];

        for (level, expected) in cases {
            assert_eq!(
                MlKemCtrlSchedule::for_level(level).keccak_calls(),
                expected,
                "k={}",
                level.k
            );
        }
    }
}
