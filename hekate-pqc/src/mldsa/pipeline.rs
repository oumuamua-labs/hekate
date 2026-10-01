// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec;
use alloc::vec::Vec;
use core::array;
use hekate_core::errors::{self, Error};
use hekate_core::trace::{ColumnTrace, TraceCompatibleField};
use hekate_keccak::{KeccakChiplet, generate_keccak_trace};
use hekate_math::{Block64, Flat, HardwareField, PackableField, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::permutation::{BusKind, Service, ServiceSlot};
use subtle::ConstantTimeEq;
use zeroize::{Zeroize, ZeroizeOnDrop, Zeroizing};

use super::MlDsaParams;
use crate::codec::{CodecChiplet, CodecStep};
use crate::ctrl::{Assembler, CtrlChiplet, Io, RATE};
use crate::high_bits::HighBitsChiplet;
use crate::ntt::{Mac, NttChiplet, NttParams, NttSchedule, NttStep, Transform};
use crate::sampler::{SamplerChiplet, SamplerStep};
use crate::utils::{height, le_lanes, le_words};
use crate::wiring::{LaneValues, N, Poly, PolyLabels, PolyValues, Stream, WordValues};

/// Bus of [`service`]. Pass it as `external` to
/// `Circuit::attach_namespaced` to keep it unprefixed.
pub const MLDSA_DATA_BUS_ID: &str = "ml_dsa_data";

const SHAKE: u8 = 0x1f;
const SEED_BYTES: usize = 32;
const DIGEST_LANES: usize = 8;
const DIGEST_WORDS: usize = 2 * DIGEST_LANES;
const MAX_CALLS: usize = 256;

/// One call: the encoded public key, M' as FIPS 204 Algorithm 3
/// forms it from M and the context, and the encoded signature.
#[derive(Clone, Copy)]
pub struct MlDsaInput<'a> {
    pub pk: &'a [u8],
    pub message: &'a [u8],
    pub signature: &'a [u8],
}

/// The digests of one verified call: tr = H(pk, 64) and
/// μ = H(tr ‖ M′, 64), as FIPS 204 Algorithm 8 computes them.
#[derive(Clone, PartialEq, Eq, Zeroize, ZeroizeOnDrop)]
pub struct MlDsaOutput {
    pub tr: [u8; 64],
    pub mu: [u8; 64],
}

/// Traces in `defs` order, the host's words in request order
/// (per call pk, M' and σ, then tr and μ), and each call's outputs.
pub struct MlDsaWitness {
    pub traces: Vec<ColumnTrace>,
    pub words: Zeroizing<Vec<u32>>,
    pub outputs: Zeroizing<Vec<MlDsaOutput>>,
}

#[derive(Clone, Debug)]
struct Call {
    msg_len: usize,
    rho: Stream,
    ct_lanes: Stream,
    t1_words: Stream,
    z_words: Stream,
    h_words: Stream,
    ct_words: Stream,
    tr_words: Stream,
    mu_words: Stream,
    t1: Vec<Poly>,
    z: Vec<Poly>,
    c: Poly,
    a: Vec<Vec<Poly>>,
    t1_hat: Vec<Poly>,
    z_hat: Vec<Poly>,
    c_hat: Poly,
    w_hat: Vec<Poly>,
    w: Vec<Poly>,
}

impl Call {
    fn new(params: &MlDsaParams, msg_len: usize, labels: &mut PolyLabels) -> errors::Result<Self> {
        let (k, l) = (params.k(), params.l());

        let polys = |n: usize, labels: &mut PolyLabels| -> errors::Result<Vec<Poly>> {
            (0..n).map(|_| labels.fresh()).collect()
        };

        Ok(Self {
            msg_len,
            rho: labels.stream()?,
            ct_lanes: labels.stream()?,
            t1_words: labels.stream()?,
            z_words: labels.stream()?,
            h_words: labels.stream()?,
            ct_words: labels.stream()?,
            tr_words: labels.stream()?,
            mu_words: labels.stream()?,
            t1: polys(k, labels)?,
            z: polys(l, labels)?,
            c: labels.fresh()?,
            a: (0..k)
                .map(|_| polys(l, labels))
                .collect::<errors::Result<Vec<Vec<Poly>>>>()?,
            t1_hat: polys(k, labels)?,
            z_hat: polys(l, labels)?,
            c_hat: labels.fresh()?,
            w_hat: polys(k, labels)?,
            w: polys(k, labels)?,
        })
    }
}

pub struct Forgery<'a> {
    calls: &'a [Call],
    values: &'a mut PolyValues,
    hints: &'a mut [[bool; N]],
    hosted: &'a mut [Vec<u32>],
}

impl Forgery<'_> {
    pub fn a_hat(&mut self, call: usize, i: usize, j: usize) -> errors::Result<&mut [u32; N]> {
        let poly = self
            .calls
            .get(call)
            .and_then(|c| c.a.get(i))
            .and_then(|row| row.get(j))
            .copied()
            .ok_or(outside())?;

        self.values.get_mut(poly)
    }

    pub fn hint(&mut self, call: usize, i: usize) -> errors::Result<&mut [bool; N]> {
        let k = self.calls.first().map_or(0, |c| c.w.len());

        match i < k {
            true => self.hints.get_mut(call * k + i).ok_or(outside()),
            false => Err(outside()),
        }
    }

    pub fn host(&mut self, call: usize) -> errors::Result<&mut [u32]> {
        self.hosted
            .get_mut(call)
            .map(Vec::as_mut_slice)
            .ok_or(outside())
    }
}

/// ML-DSA.Verify_internal (FIPS 204 Algorithm 8)
/// for 1 to 256 signatures, as the tables of `defs`.
#[derive(Clone)]
pub struct MlDsaChiplet<F: TowerField + TraceCompatibleField> {
    params: MlDsaParams,
    calls: Vec<Call>,
    w1: Stream,
    ctrl: CtrlChiplet<F>,
    sampler: SamplerChiplet<F>,
    keccak: KeccakChiplet,
    keccak_rows: usize,
    codec: CodecChiplet<F>,
    ntt: NttChiplet<F>,
    high_bits: HighBitsChiplet<F>,
}

impl<F> MlDsaChiplet<F>
where
    F: TowerField + TraceCompatibleField + PackableField + HardwareField + Send + 'static,
    <F as PackableField>::Packed: Copy + Send + Sync,
    Flat<F>: Send + Sync,
{
    /// Compiles the tables that verify one signature per
    /// entry of `msg_lens`, which gives that call's M′
    /// length in bytes. Takes 1 to 256 entries, 1 to 227
    /// at ML-DSA-87, where the polynomial labels run out.
    pub fn new(params: MlDsaParams, msg_lens: &[usize]) -> errors::Result<Self> {
        if msg_lens.is_empty() || msg_lens.len() > MAX_CALLS {
            return Err(Error::Protocol {
                protocol: "mldsa_chiplet",
                message: "mldsa pipeline serves 1 to 256 calls",
            });
        }

        let (k, l) = (params.k(), params.l());

        let mut labels = PolyLabels::new();

        let calls = msg_lens
            .iter()
            .map(|&len| Call::new(&params, len, &mut labels))
            .collect::<errors::Result<Vec<Call>>>()?;

        let w1 = labels.stream()?;

        let mut codec_steps = Vec::with_capacity(3 * calls.len());
        let mut sampler_steps = Vec::with_capacity((k * l + 1) * calls.len());

        let (mut forward, mut macs, mut inverse) = (Vec::new(), Vec::new(), Vec::new());

        for (i, call) in calls.iter().enumerate() {
            codec_steps.push(CodecStep::t1(call.t1_words, call.t1.clone()));
            codec_steps.push(CodecStep::z(&params, call.z_words, call.z.clone()));
            codec_steps.push(CodecStep::hint(&params, call.h_words, i as u8));

            for r in 0..k {
                for s in 0..l {
                    sampler_steps.push(SamplerStep::expand_a(
                        call.rho,
                        r as u8,
                        s as u8,
                        call.a[r][s],
                    ));
                }
            }

            sampler_steps.push(SamplerStep::sample_in_ball(&params, call.ct_lanes, call.c)?);

            for j in 0..l {
                forward.push(NttStep::Transform(Transform::forward(
                    call.z[j],
                    call.z_hat[j],
                )));
            }

            forward.push(NttStep::Transform(Transform::forward(call.c, call.c_hat)));

            for i in 0..k {
                forward.push(NttStep::Transform(Transform::forward(
                    call.t1[i],
                    call.t1_hat[i],
                )));
            }

            let a = (0..k)
                .map(|i| call.a[i].iter().copied().chain([call.t1_hat[i]]).collect())
                .collect();
            let b = call.z_hat.iter().copied().chain([call.c_hat]).collect();
            let negate = (0..=l).map(|j| j == l).collect();

            macs.push(NttStep::Mac(Mac::new(a, b, negate, call.w_hat.clone())?));

            for i in 0..k {
                inverse.push(NttStep::Transform(Transform::inverse(
                    call.w_hat[i],
                    call.w[i],
                )));
            }
        }

        let codec_rows = codec_steps.iter().map(CodecStep::rows).sum::<usize>();
        let codec = CodecChiplet::new(codec_steps, height(codec_rows))?;

        let sampler_rows = sampler_steps.iter().map(SamplerStep::rows).sum::<usize>();
        let sampler_blocks = sampler_steps.iter().map(SamplerStep::blocks).sum::<usize>();
        let sampler = SamplerChiplet::new(sampler_steps, height(sampler_rows))?;

        let steps = forward.into_iter().chain(macs).chain(inverse).collect();
        let schedule = NttSchedule::new(NttParams::ML_DSA, steps, &mut labels)?;
        let ntt_rows = height(schedule.rows());
        let ntt = NttChiplet::new(schedule, ntt_rows)?;

        let inputs: Vec<Poly> = calls.iter().flat_map(|c| c.w.iter().copied()).collect();
        let high_bits_rows = height(inputs.len() * N);
        let high_bits = HighBitsChiplet::new(params, inputs, w1, high_bits_rows)?;

        let lanes_per_call = k * N * params.w1_bits() / 64;

        let mut asm = Assembler::new();
        for (i, call) in calls.iter().enumerate() {
            assemble(&mut asm, &params, call, w1, i * lanes_per_call)?;
        }

        let rows = asm.into_rows();
        let ctrl_rows = height(rows.len());
        let ctrl = CtrlChiplet::new("MlDsaCtrl", &service(), rows, ctrl_rows)?;

        let blocks = ctrl.blocks() + sampler_blocks;
        let keccak_rows = height(blocks * KeccakChiplet::BLOCK_ROWS);
        let keccak = KeccakChiplet::new(keccak_rows, blocks);

        Ok(Self {
            params,
            calls,
            w1,
            ctrl,
            sampler,
            keccak,
            keccak_rows,
            codec,
            ntt,
            high_bits,
        })
    }

    /// The tables as chiplets for a host program to attach,
    /// in the order `trace` returns their traces.
    pub fn defs(&self) -> errors::Result<Vec<ChipletDef<F>>> {
        Ok(vec![
            self.ctrl.def()?,
            self.sampler.def()?,
            ChipletDef::from_air(&self.keccak)?,
            self.codec.def()?,
            self.ntt.def()?,
            self.high_bits.def()?,
        ])
    }

    pub fn params(&self) -> MlDsaParams {
        self.params
    }

    /// Traces every table for one input per call, in `new`'s
    /// order, with the host's words and each call's tr and μ.
    pub fn trace(&self, inputs: &[MlDsaInput<'_>]) -> errors::Result<MlDsaWitness> {
        self.witness(inputs, |_| Ok(()), true)
    }

    /// `trace` with the c̃ check skipped and `forge` applied
    /// after the Codec and Sampler traces, to the values
    /// the NTT, HighBits and Ctrl tables trace from.
    #[cfg(feature = "forgery")]
    pub fn trace_forged(
        &self,
        inputs: &[MlDsaInput<'_>],
        forge: impl FnOnce(&mut Forgery<'_>) -> errors::Result<()>,
    ) -> errors::Result<MlDsaWitness> {
        self.witness(inputs, forge, false)
    }

    fn witness(
        &self,
        inputs: &[MlDsaInput<'_>],
        forge: impl FnOnce(&mut Forgery<'_>) -> errors::Result<()>,
        checked: bool,
    ) -> errors::Result<MlDsaWitness> {
        if inputs.len() != self.calls.len() {
            return Err(Error::Protocol {
                protocol: "mldsa_chiplet",
                message: "one input per call of the mldsa pipeline",
            });
        }

        let params = &self.params;
        let (k, l) = (params.k(), params.l());

        let mut words = WordValues::default();
        let mut seeds = LaneValues::default();

        let mut hosted: Zeroizing<Vec<Vec<u32>>> = Zeroizing::new(Vec::with_capacity(inputs.len()));
        let mut ctildes = Zeroizing::new(Vec::with_capacity(inputs.len()));

        for (call, input) in self.calls.iter().zip(inputs) {
            if input.pk.len() != params.pk_bytes()
                || input.signature.len() != params.sig_bytes()
                || input.message.len() != call.msg_len
            {
                return Err(Error::Protocol {
                    protocol: "mldsa_chiplet",
                    message: "public key, message or signature length differs from the call",
                });
            }

            let (rho, t1) = input.pk.split_at(SEED_BYTES);
            let (ct, rest) = input.signature.split_at(params.lambda() / 4);
            let (z, h) = rest.split_at(32 * l * params.z_bits());

            words.insert(call.t1_words, le_words(t1).collect())?;
            words.insert(call.z_words, le_words(z).collect())?;
            words.insert(call.h_words, le_words(h).collect())?;

            seeds.insert(call.rho, le_lanes(rho).collect())?;
            seeds.insert(call.ct_lanes, le_lanes(ct).collect())?;

            let parts = [input.pk, input.message, ct, z, h];
            let mut host = Vec::with_capacity(parts.iter().map(|p| p.len().div_ceil(4)).sum());

            host.extend(parts.into_iter().flat_map(le_words));
            hosted.push(host);

            ctildes.push(le_words(ct).collect::<Vec<u32>>());
        }

        let mut values = PolyValues::default();

        let (codec, mut hints) = self.codec.trace(&mut words, &mut values)?;
        let (sampler, sampled) = self.sampler.trace(&seeds, &mut values)?;

        forge(&mut Forgery {
            calls: &self.calls,
            values: &mut values,
            hints: hints.as_mut_slice(),
            hosted: hosted.as_mut_slice(),
        })?;

        let ntt = self.ntt.trace(&mut values)?;
        let high_bits = self.high_bits.trace(&values, &hints)?;

        let per_poly = N * params.w1_bits() / 64;

        let mut w1 = vec![0u64; self.calls.len() * k * per_poly];
        for (call, (hints, lanes)) in self
            .calls
            .iter()
            .zip(hints.chunks(k).zip(w1.chunks_mut(k * per_poly)))
        {
            for ((&w, h), out) in call.w.iter().zip(hints).zip(lanes.chunks_mut(per_poly)) {
                let w = values.get(w)?;
                let mut used: [u32; N] = array::from_fn(|m| params.use_hint(h[m], w[m]));

                params.w1_lanes(&used, out)?;

                used.zeroize();
            }
        }

        let mut lanes = LaneValues::default();
        lanes.insert(self.w1, w1)?;

        let ctrl = self
            .ctrl
            .trace(&Zeroizing::new(hosted.concat()), &lanes, &mut words)?;

        let (sent, received) = self.ctrl.host_words();

        let mut emitted = Zeroizing::new(Vec::with_capacity(sent + received));
        let mut outputs = Zeroizing::new(Vec::with_capacity(inputs.len()));

        for ((call, host), ct) in self.calls.iter().zip(hosted.iter()).zip(ctildes.iter()) {
            if checked && !bool::from(words.get(call.ct_words)?.ct_eq(ct.as_slice())) {
                return Err(Error::Protocol {
                    protocol: "mldsa_chiplet",
                    message: "signature does not verify: c̃' differs from c̃",
                });
            }

            let tr = words.get(call.tr_words)?;
            let mu = words.get(call.mu_words)?;

            emitted.extend(host);
            emitted.extend(tr);
            emitted.extend(mu);

            outputs.push(MlDsaOutput {
                tr: le_bytes(tr)?,
                mu: le_bytes(mu)?,
            });
        }

        let calls: Zeroizing<Vec<[Block64; 25]>> = Zeroizing::new(
            ctrl.keccak
                .iter()
                .chain(sampled.iter())
                .map(|state| state.map(Block64))
                .collect(),
        );

        let keccak = generate_keccak_trace(&calls, self.keccak_rows)?;

        Ok(MlDsaWitness {
            traces: vec![ctrl.trace, sampler, keccak, codec, ntt, high_bits],
            words: emitted,
            outputs,
        })
    }
}

/// The service a host calls with each call's pk, M′
/// and σ words; it answers with the tr and μ words.
pub fn service() -> Service {
    Service {
        bus_id: MLDSA_DATA_BUS_ID,
        kind: BusKind::Permutation,
        slots: vec![
            ServiceSlot::Value(b"kappa_mldsa_word"),
            ServiceSlot::EmitRank,
        ],
    }
}

fn assemble(
    asm: &mut Assembler,
    params: &MlDsaParams,
    call: &Call,
    w1: Stream,
    w1_base: usize,
) -> errors::Result<()> {
    let seed_words = SEED_BYTES / 4;
    let t1_words = (params.pk_bytes() - SEED_BYTES) / 4;

    asm.reset(RATE)?;

    for j in 0..seed_words {
        let lane = (j % 2 == 1).then_some((call.rho, (j / 2) as u16));

        asm.absorb_word(Io::Input, None, lane, 4)?;
    }

    for j in 0..t1_words {
        asm.absorb_word(Io::Input, Some((call.t1_words, j as u16)), None, 4)?;
    }

    asm.finish(&[SHAKE])?;
    asm.split(DIGEST_LANES, call.tr_words, None)?;
    asm.reset_prefix()?;

    let message_words = call.msg_len.div_ceil(4);

    for j in 0..message_words {
        let bytes = match (j + 1 == message_words, call.msg_len % 4) {
            (true, tail) if tail != 0 => tail,
            _ => 4,
        };

        asm.absorb_word(Io::Input, None, None, bytes)?;
    }

    asm.finish(&[SHAKE])?;
    asm.split(DIGEST_LANES, call.mu_words, None)?;

    let ct_words = params.lambda() / 16;
    let z_words = 8 * params.l() * params.z_bits();
    let h_words = (params.omega() + params.k()).div_ceil(4);

    for (stream, count) in [
        (call.ct_words, ct_words),
        (call.z_words, z_words),
        (call.h_words, h_words),
    ] {
        for j in 0..count {
            asm.input(Some((stream, j as u16)));
        }
    }

    asm.reset_prefix()?;

    let w1_lanes = params.k() * N * params.w1_bits() / 64;

    for j in 0..w1_lanes {
        asm.absorb_lane((w1, (w1_base + j) as u16))?;
    }

    asm.finish(&[SHAKE])?;
    asm.split(params.lambda() / 32, call.ct_words, Some(call.ct_lanes))?;

    for stream in [call.tr_words, call.mu_words] {
        for j in 0..DIGEST_WORDS {
            asm.output((stream, j as u16));
        }
    }

    Ok(())
}

fn le_bytes(words: &[u32]) -> errors::Result<[u8; 64]> {
    let mut out = [0u8; 64];

    if words.len() != DIGEST_WORDS {
        return Err(Error::Protocol {
            protocol: "mldsa_chiplet",
            message: "a digest spans 16 words",
        });
    }

    for (chunk, word) in out.chunks_mut(4).zip(words) {
        chunk.copy_from_slice(&word.to_le_bytes());
    }

    Ok(out)
}

fn outside() -> Error {
    Error::Protocol {
        protocol: "mldsa_chiplet",
        message: "the forgery names a call or polynomial outside the pipeline",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hekate_core::trace::{ColumnType, TraceBuilder};
    use hekate_math::{Bit, Block16, Block32, Block128};
    use hekate_program::circuit::Circuit;
    use hekate_program::{Air, FixedShape, ProgramInstance, ProgramWitness};
    use hekate_scribble::{MutationKind, ScribbleConfig, Target, assert_all_caught};
    use hekate_sdk::preflight::preflight;

    use crate::census::{census, unbalanced};
    use crate::sampler::SIB_BUS_ID;
    use crate::wiring::{
        COEF_BUS_ID, LANE_BUS_ID, WORD_BUS_ID, lane_spec, pinned_shape, word_spec,
    };

    type F = Block128;

    const HOST_LAYOUT: [ColumnType; 10] = [
        ColumnType::B32,
        ColumnType::Bit,
        ColumnType::B16,
        ColumnType::B16,
        ColumnType::B32,
        ColumnType::Bit,
        ColumnType::B16,
        ColumnType::B16,
        ColumnType::B64,
        ColumnType::Bit,
    ];

    fn junk(i: usize) -> u32 {
        (i as u32).wrapping_mul(0x9e37_79b9) ^ 0x5a5a
    }

    fn lanes_of(words: &[u32]) -> Vec<u64> {
        words
            .chunks(2)
            .map(|w| w[0] as u64 | (w[1] as u64) << 32)
            .collect()
    }

    fn producers(pipeline: &MlDsaChiplet<F>) -> Vec<(String, Vec<(&'static str, u16)>)> {
        vec![
            (
                pipeline.ctrl.def().unwrap().name(),
                pipeline.ctrl.produced(),
            ),
            (
                pipeline.sampler.def().unwrap().name(),
                pipeline.sampler.produced(),
            ),
            (
                pipeline.codec.def().unwrap().name(),
                pipeline.codec.produced(),
            ),
            (pipeline.ntt.def().unwrap().name(), pipeline.ntt.produced()),
            (
                pipeline.high_bits.def().unwrap().name(),
                pipeline.high_bits.produced(),
            ),
        ]
    }

    #[test]
    fn no_key_misses_or_repeats_emitter() {
        for params in [
            MlDsaParams::ML_DSA_44,
            MlDsaParams::ML_DSA_65,
            MlDsaParams::ML_DSA_87,
        ] {
            let pipeline = MlDsaChiplet::<F>::new(params, &[13, 40]).unwrap();
            let keys = census(
                &pipeline.defs().unwrap(),
                &pipeline.sampler,
                &producers(&pipeline),
            )
            .unwrap();

            let broken = unbalanced(&keys);
            assert!(broken.is_empty(), "{broken:?}");

            for bus in [COEF_BUS_ID, WORD_BUS_ID, LANE_BUS_ID, SIB_BUS_ID] {
                assert!(keys.keys().any(|(b, _, _)| b == bus));
            }
        }
    }

    #[test]
    fn scribble_ctrl_flip_selector_caught() {
        let params = MlDsaParams::ML_DSA_44;
        let pipeline = MlDsaChiplet::<F>::new(params, &[13]).unwrap();
        let call = &pipeline.calls[0];

        let t1_words = (params.pk_bytes() - SEED_BYTES) / 4;
        let z_words = 8 * params.l() * params.z_bits();
        let h_words = (params.omega() + params.k()).div_ceil(4);
        let w1_lanes = params.k() * N * params.w1_bits() / 64;

        let pk: Vec<u32> = (0..SEED_BYTES / 4 + t1_words).map(junk).collect();
        let message = [junk(900), junk(901), junk(902), 0x7f];
        let z: Vec<u32> = (0..z_words).map(|i| junk(1000 + i)).collect();
        let h: Vec<u32> = (0..h_words).map(|i| junk(3000 + i)).collect();
        let w1: Vec<u64> = (0..w1_lanes as u64)
            .map(|i| i.wrapping_mul(0x9e37_79b9_7f4a_7c15))
            .collect();

        let mut lanes = LaneValues::default();
        lanes.insert(pipeline.w1, w1.clone()).unwrap();

        let inputs = |ct: &[u32]| [pk.as_slice(), &message, ct, &z, &h].concat();

        let probe = vec![0; params.lambda() / 16];

        let mut probed = WordValues::default();

        pipeline
            .ctrl
            .trace(&inputs(&probe), &lanes, &mut probed)
            .unwrap();

        let ct = probed.get(call.ct_words).unwrap().to_vec();

        let mut produced = WordValues::default();

        let ctrl = pipeline
            .ctrl
            .trace(&inputs(&ct), &lanes, &mut produced)
            .unwrap();

        let hosted = [
            inputs(&ct),
            produced.get(call.tr_words).unwrap().to_vec(),
            produced.get(call.mu_words).unwrap().to_vec(),
        ]
        .concat();

        let word_tokens: Vec<(Stream, u16, u32)> = [
            (call.t1_words, &pk[SEED_BYTES / 4..]),
            (call.z_words, &z),
            (call.h_words, &h),
        ]
        .into_iter()
        .flat_map(|(stream, words)| {
            words
                .iter()
                .enumerate()
                .map(move |(j, &w)| (stream, j as u16, w))
        })
        .collect();

        let lane_tokens: Vec<(Stream, u16, u64)> = [
            (call.rho, lanes_of(&pk[..SEED_BYTES / 4])),
            (call.ct_lanes, lanes_of(&ct)),
            (pipeline.w1, w1),
        ]
        .into_iter()
        .flat_map(|(stream, lanes)| {
            lanes
                .into_iter()
                .enumerate()
                .map(move |(j, lane)| (stream, j as u16, lane))
        })
        .collect();

        let rows = hosted
            .len()
            .max(word_tokens.len())
            .max(lane_tokens.len())
            .next_power_of_two();

        let blocks = ctrl.keccak.len();
        let keccak_rows = (blocks * KeccakChiplet::BLOCK_ROWS).next_power_of_two();

        let pin = |values: Vec<u64>| pinned_shape::<F>(values);

        let mut cx = Circuit::<F>::new("CtrlHost", rows).unwrap();

        let cols = cx.schema(&HOST_LAYOUT);

        cx.fix(
            cols.at(1),
            FixedShape::Cadence {
                stride: 1,
                count: hosted.len(),
                origin: 0,
                values: vec![F::ONE],
            },
        );
        cx.fix(
            cols.at(2),
            pin(word_tokens.iter().map(|t| t.0.id() as u64).collect()),
        );
        cx.fix(
            cols.at(3),
            pin(word_tokens.iter().map(|t| t.1 as u64).collect()),
        );
        cx.fix(cols.at(5), pin(vec![1; word_tokens.len()]));
        cx.fix(
            cols.at(6),
            pin(lane_tokens.iter().map(|t| t.0.id() as u64).collect()),
        );
        cx.fix(
            cols.at(7),
            pin(lane_tokens.iter().map(|t| t.1 as u64).collect()),
        );
        cx.fix(cols.at(9), pin(vec![1; lane_tokens.len()]));

        cx.call(&service(), &[cols.at(0)], cols.at(1)).unwrap();

        cx.bus(WORD_BUS_ID, word_spec(2, 3, 4, 5));
        cx.bus(LANE_BUS_ID, lane_spec(6, 7, 8, 9));

        cx.attach(pipeline.ctrl.def().unwrap());
        cx.attach(ChipletDef::from_air(&KeccakChiplet::new(keccak_rows, blocks)).unwrap());

        let program = cx.compile().unwrap();

        let mut tb = TraceBuilder::new(&HOST_LAYOUT, rows.trailing_zeros() as usize).unwrap();

        for (r, &w) in hosted.iter().enumerate() {
            tb.set_b32(0, r, Block32::from(w)).unwrap();
            tb.set_bit(1, r, Bit::from(1u8)).unwrap();
        }

        for (r, &(stream, j, w)) in word_tokens.iter().enumerate() {
            tb.set_b16(2, r, Block16(stream.id())).unwrap();
            tb.set_b16(3, r, Block16(j)).unwrap();
            tb.set_b32(4, r, Block32::from(w)).unwrap();
            tb.set_bit(5, r, Bit::from(1u8)).unwrap();
        }

        for (r, &(stream, j, lane)) in lane_tokens.iter().enumerate() {
            tb.set_b16(6, r, Block16(stream.id())).unwrap();
            tb.set_b16(7, r, Block16(j)).unwrap();
            tb.set_b64(8, r, Block64(lane)).unwrap();
            tb.set_bit(9, r, Bit::from(1u8)).unwrap();
        }

        let inputs: Vec<[Block64; 25]> = ctrl.keccak.iter().map(|s| s.map(Block64)).collect();
        let keccak = generate_keccak_trace(&inputs, keccak_rows).unwrap();

        let instance = ProgramInstance::new(rows, Vec::new());
        let witness = ProgramWitness::new(tb.build()).with_chiplets(vec![ctrl.trace, keccak]);

        assert!(preflight(&program, &instance, &witness).unwrap().is_clean());

        assert_all_caught(
            &program,
            &instance,
            &witness,
            ScribbleConfig::default()
                .mutations([MutationKind::FlipSelector])
                .target(Target::Chiplet(0))
                .cases(64),
        );
    }
}
