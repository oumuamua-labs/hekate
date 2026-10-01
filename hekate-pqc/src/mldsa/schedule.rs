// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec;
use alloc::vec::Vec;
use hekate_math::TowerField;
use hekate_program::{CadenceSegment, FixedShape};

use super::{MLDSA_Q, MlDsaLevel};
use crate::utils::{SolidRun, push_seg, segments_shape, sponge_calls};

const N: usize = 256;
const FWD_NONFC: usize = 8 * 128;
const INV_NONFC: usize = 2 * 8 * 128 + N;
const SHAKE_256_RATE: usize = 136;
const SHAKE_128_RATE: usize = 168;

#[derive(Clone, Copy, Debug)]
enum Run {
    Io { chunks: usize },
    Keccak { calls: usize },
    NttRam { rows: usize },
    PairAlt { pairs: usize },
    BoundaryIn,
    BoundaryOut,
    Ram { rows: usize },
    HighBits { rows: usize },
    NormCheck { rows: usize },
    Separator,
    Cmp,
}

impl Run {
    fn rows(self) -> usize {
        match self {
            Run::Io { chunks } => chunks,
            Run::Keccak { calls } => 2 * calls,
            Run::NttRam { rows } | Run::Ram { rows } => rows,
            Run::PairAlt { pairs } => 2 * pairs,
            Run::BoundaryIn | Run::BoundaryOut => N,
            Run::HighBits { rows } | Run::NormCheck { rows } => rows,
            Run::Separator | Run::Cmp => 1,
        }
    }

    fn ram_rows(self) -> usize {
        match self {
            Run::Io { .. } | Run::Keccak { .. } | Run::Separator | Run::Cmp => 0,
            Run::PairAlt { pairs } => pairs,
            _ => self.rows(),
        }
    }
}

/// Row layout of one verification's ctrl dispatch,
/// mirroring the trace generator's (phase, instance)
/// sort; a diverging trace fails the fixed-column check.
#[derive(Clone, Debug)]
pub(crate) struct MlDsaCtrlSchedule {
    runs: Vec<Run>,
    keccak_calls: usize,
}

pub(crate) struct MlDsaCtrlShapes<F> {
    pub(crate) io: FixedShape<F>,
    pub(crate) keccak: FixedShape<F>,
    pub(crate) kec_is_output: FixedShape<F>,
    pub(crate) ntt: FixedShape<F>,
    pub(crate) w_bind: FixedShape<F>,
    pub(crate) ram: FixedShape<F>,
    pub(crate) bound_in: FixedShape<F>,
    pub(crate) bound_out: FixedShape<F>,
    pub(crate) norm_check: FixedShape<F>,
    pub(crate) highbits: FixedShape<F>,
}

impl MlDsaCtrlSchedule {
    pub(crate) fn for_level(level: MlDsaLevel, msg_len: usize) -> Self {
        let k = level.k;
        let l = level.l;

        let c_tilde = match k {
            4 => 32,
            6 => 48,
            _ => 64,
        };

        let tr = sponge_calls(SHAKE_256_RATE, level.pk_bytes(), 64);
        let mu = sponge_calls(SHAKE_256_RATE, 66 + msg_len, 64);
        let sib = sponge_calls(SHAKE_256_RATE, c_tilde, 8 + 3 * level.tau);
        let ea = k * l * sponge_calls(SHAKE_128_RATE, 34, 3 * N + 128);
        let es_calls = tr + mu + sib + ea;

        let w1_bits = if level.gamma2 == (MLDSA_Q - 1) / 88 {
            6
        } else {
            4
        };
        let hc_calls = sponge_calls(SHAKE_256_RATE, 64 + k * N * w1_bits / 8, c_tilde);

        let mut runs = vec![Run::Io {
            chunks: c_tilde / 4,
        }];

        runs.push(Run::Keccak { calls: es_calls });

        for _ in 0..1 + l {
            runs.push(Run::NttRam { rows: FWD_NONFC });
            runs.push(Run::BoundaryIn);
            runs.push(Run::BoundaryOut);
        }

        runs.push(Run::NttRam { rows: FWD_NONFC });
        runs.push(Run::BoundaryIn);
        runs.push(Run::BoundaryOut);

        for _ in 0..k - 1 {
            runs.push(Run::PairAlt { pairs: (l + 1) * N });
            runs.push(Run::NttRam { rows: FWD_NONFC });
            runs.push(Run::BoundaryIn);
            runs.push(Run::BoundaryOut);
        }

        runs.push(Run::PairAlt { pairs: (l + 1) * N });

        for i in 0..k {
            runs.push(Run::NttRam { rows: INV_NONFC });

            if i + 1 < k {
                runs.push(Run::Separator);
            }
        }

        runs.push(Run::Ram { rows: k * N });

        runs.push(Run::Ram { rows: k * N });
        runs.push(Run::HighBits { rows: k * N });

        runs.push(Run::Keccak { calls: hc_calls });
        runs.push(Run::Cmp);

        runs.push(Run::Ram { rows: l * N });
        runs.push(Run::NormCheck { rows: l * N });

        Self {
            runs,
            keccak_calls: es_calls + hc_calls,
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

    pub(crate) fn norm_check_ops(&self) -> usize {
        self.runs
            .iter()
            .map(|run| match run {
                Run::NormCheck { rows } => *rows,
                _ => 0,
            })
            .sum()
    }

    pub(crate) fn highbits_ops(&self) -> usize {
        self.runs
            .iter()
            .map(|run| match run {
                Run::HighBits { rows } => *rows,
                _ => 0,
            })
            .sum()
    }

    pub(crate) fn fixed_shapes<F: TowerField>(&self) -> MlDsaCtrlShapes<F> {
        let mut io = Vec::new();
        let mut keccak = Vec::new();
        let mut kec_is_output = Vec::new();
        let mut ntt = Vec::new();
        let mut w_bind = Vec::new();
        let mut ram: Vec<CadenceSegment<F>> = Vec::new();
        let mut bound_in = Vec::new();
        let mut bound_out = Vec::new();
        let mut norm_check = Vec::new();
        let mut highbits = Vec::new();

        let mut solid = SolidRun::default();
        let mut origin = 0usize;

        for run in self.runs.iter().copied() {
            match run {
                Run::Io { chunks } => {
                    push_seg(&mut io, origin, 1, chunks, &[F::ONE]);
                    solid.flush(&mut ram);
                }
                Run::Keccak { calls } => {
                    push_seg(&mut keccak, origin, 1, 2 * calls, &[F::ONE]);
                    push_seg(&mut kec_is_output, origin, 2, calls, &[F::ZERO, F::ONE]);
                    solid.flush(&mut ram);
                }
                Run::NttRam { rows } => {
                    push_seg(&mut ntt, origin, 1, rows, &[F::ONE]);
                    solid.extend(origin, rows);
                }
                Run::PairAlt { pairs } => {
                    push_seg(&mut ntt, origin, 2, pairs, &[F::ONE, F::ZERO]);
                    push_seg(&mut w_bind, origin, 2, pairs, &[F::ZERO, F::ONE]);

                    solid.flush(&mut ram);
                    push_seg(&mut ram, origin, 2, pairs, &[F::ONE, F::ZERO]);
                }
                Run::BoundaryIn => {
                    push_seg(&mut bound_in, origin, 1, N, &[F::ONE]);
                    solid.extend(origin, N);
                }
                Run::BoundaryOut => {
                    push_seg(&mut bound_out, origin, 1, N, &[F::ONE]);
                    solid.extend(origin, N);
                }
                Run::Ram { rows } => solid.extend(origin, rows),
                Run::HighBits { rows } => {
                    push_seg(&mut highbits, origin, 1, rows, &[F::ONE]);
                    solid.extend(origin, rows);
                }
                Run::NormCheck { rows } => {
                    push_seg(&mut norm_check, origin, 1, rows, &[F::ONE]);
                    solid.extend(origin, rows);
                }
                Run::Separator | Run::Cmp => solid.flush(&mut ram),
            }

            origin += run.rows();
        }

        solid.flush(&mut ram);

        MlDsaCtrlShapes {
            io: segments_shape(io),
            keccak: segments_shape(keccak),
            kec_is_output: segments_shape(kec_is_output),
            ntt: segments_shape(ntt),
            w_bind: segments_shape(w_bind),
            ram: segments_shape(ram),
            bound_in: segments_shape(bound_in),
            bound_out: segments_shape(bound_out),
            norm_check: segments_shape(norm_check),
            highbits: segments_shape(highbits),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mldsa::MlDsaLevel;

    #[test]
    fn keccak_calls_match_fips204_kat_baseline() {
        let cases = [
            (MlDsaLevel::MLDSA_44, 115),
            (MlDsaLevel::MLDSA_65, 205),
            (MlDsaLevel::MLDSA_87, 368),
        ];

        for (level, expected) in cases {
            assert_eq!(
                MlDsaCtrlSchedule::for_level(level, 0).keccak_calls(),
                expected,
                "k={}",
                level.k()
            );
        }
    }

    #[test]
    fn message_hash_costs_one_block_per_shake256_rate() {
        let level = MlDsaLevel::MLDSA_65;
        let base = MlDsaCtrlSchedule::for_level(level, 0).keccak_calls();

        for (msg_len, extra) in [(0usize, 0), (69, 0), (70, 1), (205, 1), (206, 2)] {
            assert_eq!(
                MlDsaCtrlSchedule::for_level(level, msg_len).keccak_calls(),
                base + extra,
                "msg_len={msg_len}"
            );
        }
    }
}
