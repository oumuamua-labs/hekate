// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec;
use alloc::vec::Vec;
use hekate_core::errors::{self, Error};
use hekate_core::trace::{ColumnTrace, TraceCompatibleField};
use hekate_keccak::{KeccakChiplet, generate_keccak_trace};
use hekate_math::{Block64, Flat, HardwareField, PackableField, TowerField};
use hekate_program::chiplet::ChipletDef;
use hekate_program::permutation::{BusKind, Service, ServiceSlot};
use subtle::{ConditionallySelectable, ConstantTimeEq};
use zeroize::{Zeroize, ZeroizeOnDrop, Zeroizing};

use super::MlKemParams;
use super::reference::{
    Coeffs, G_RATE, SHA3, SHAKE, add, byte_decode, byte_encode, canonical12, compress_poly,
    decompress_poly, g, h, j, mac, sub,
};
use crate::codec::{CodecChiplet, CodecStep};
use crate::ctrl::{Assembler, CtrlChiplet, Io, RATE};
use crate::kem_select::{CipherPart, KemSelectChiplet, KemSelectStep};
use crate::ntt::{NttChiplet, NttParams, NttSchedule, NttStep, Transform};
use crate::poly_arith::{BaseCaseMac, PolyArithChiplet};
use crate::sampler::{SamplerChiplet, SamplerStep};
use crate::utils::{height, le_lanes, le_words};
use crate::wiring::{LaneValues, N, Poly, PolyLabels, PolyValues, Stream, WordValues};

/// Bus of [`service`]. Pass it as `external` to
/// `Circuit::attach_namespaced` to keep it unprefixed.
pub const MLKEM_DATA_BUS_ID: &str = "ml_kem_data";

const SEED_BYTES: usize = 32;
const SEED_WORDS: usize = SEED_BYTES / 4;
const SEED_LANES: usize = SEED_BYTES / 8;

const POLY_BYTES: usize = 384;
const POLY_WORDS: usize = POLY_BYTES / 4;

const MAX_CALLS: usize = 256;

/// The FIPS 203 operation a call proves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlKemCall {
    /// ML-KEM.KeyGen_internal (Algorithm 16) from the seed d,
    /// returning ek, H(ek) and dk_PKE; the caller appends z.
    KeyGen,

    /// ML-KEM.Encaps_internal (Algorithm 17):
    /// the host sends ek, m and H(ek), and gets K and c.
    Encaps,

    /// ML-KEM.Decaps_internal (Algorithm 18):
    /// K, and whether c survived the re-encryption check.
    Decaps,
}

/// One call's input, matching its [`MlKemCall`].
#[derive(Clone, Copy)]
pub enum MlKemInput<'a> {
    KeyGen {
        d: &'a [u8; 32],
    },

    /// `m` is the encapsulation randomness: fresh, uniform per call.
    Encaps {
        ek: &'a [u8],
        m: &'a [u8; 32],
    },

    /// `dk` is FIPS 203's dk_PKE ‖ ek ‖ H(ek) ‖ z.
    Decaps {
        dk: &'a [u8],
        c: &'a [u8],
    },
}

/// One call's outputs, zeroized on drop.
#[derive(Zeroize, ZeroizeOnDrop)]
pub enum MlKemOutput {
    /// FIPS 203's dk is dk_pke ‖ ek ‖ h ‖ z,
    /// with the z the caller keeps.
    KeyGen {
        ek: Vec<u8>,
        dk_pke: Vec<u8>,
        h: [u8; 32],
    },
    Encaps {
        key: [u8; 32],
        c: Vec<u8>,
    },

    /// `valid` is false when c failed the re-encryption check;
    /// `key` is then the implicit-rejection key J(z ‖ c).
    Decaps {
        key: [u8; 32],
        valid: bool,
    },
}

/// Traces in `defs` order, the host's words in service order
/// (each call's inputs, then its outputs) and each call's outputs.
pub struct MlKemWitness {
    pub traces: Vec<ColumnTrace>,
    pub words: Zeroizing<Vec<u32>>,
    pub outputs: Zeroizing<Vec<MlKemOutput>>,
}

/// A change to one call for tests expecting rejection.
#[derive(Clone, Copy)]
pub enum Forgery {
    /// The Codec decodes μ from `m` while G absorbs the host's m.
    Message { call: usize, m: [u8; 32] },

    /// G absorbs `h`, and the host sends it, in place of H(ek).
    KeyHash { call: usize, h: [u8; 32] },

    /// Decaps: the Codec decodes ŝ from dk_PKE with word `word`
    /// XORed by `mask`, while the host sends the key's dk_PKE.
    DkPke { call: usize, word: usize, mask: u32 },

    /// The Sampler expands Â from `rho` while the Ctrl table
    /// relays the host key's ρ, or G outputs the one it derives.
    Rho { call: usize, rho: [u8; 32] },
}

#[derive(Clone, Debug)]
enum Call {
    KeyGen(KeyGen),
    Encaps(Encaps),
    Decaps(Decaps),
}

#[derive(Clone, Debug)]
struct KeyGen {
    rho: Stream,
    sigma: Stream,
    rho_words: Stream,
    ek_words: Stream,
    dk_words: Stream,
    h_words: Stream,
    a: Vec<Vec<Poly>>,
    s: Vec<Poly>,
    e: Vec<Poly>,
    s_hat: Vec<Poly>,
    e_hat: Vec<Poly>,
    t_hat: Vec<Poly>,
    s_copy: Vec<Poly>,
}

impl KeyGen {
    fn new(k: usize, labels: &mut PolyLabels) -> errors::Result<Self> {
        Ok(Self {
            rho: labels.stream()?,
            sigma: labels.stream()?,
            rho_words: labels.stream()?,
            ek_words: labels.stream()?,
            dk_words: labels.stream()?,
            h_words: labels.stream()?,
            a: matrix(k, labels)?,
            s: polys(k, labels)?,
            e: polys(k, labels)?,
            s_hat: polys(k, labels)?,
            e_hat: polys(k, labels)?,
            t_hat: polys(k, labels)?,
            s_copy: polys(k, labels)?,
        })
    }

    fn sampling(&self, params: &MlKemParams) -> Vec<SamplerStep> {
        let noise = self
            .s
            .iter()
            .chain(&self.e)
            .enumerate()
            .map(|(n, &out)| SamplerStep::prf(self.sigma, n as u8, params.eta1(), out));

        sample_a(self.rho, &self.a).chain(noise).collect()
    }

    fn steps(&self, params: &MlKemParams, steps: &mut Steps) -> errors::Result<()> {
        steps.sampler.extend(self.sampling(params));

        let inputs = self.s.iter().chain(&self.e);
        let outputs = self.s_hat.iter().chain(&self.e_hat);

        for (&input, &output) in inputs.zip(outputs) {
            steps
                .forward
                .push(NttStep::Transform(Transform::forward(input, output)));
        }

        steps.products.push(
            BaseCaseMac::new(
                self.a.clone(),
                self.s_hat.clone(),
                Some(self.e_hat.clone()),
                self.t_hat.clone(),
            )?
            .with_copy(self.s_copy.clone())?,
        );

        steps
            .codec
            .push(CodecStep::encode12(self.t_hat.clone(), self.ek_words));
        steps
            .codec
            .push(CodecStep::encode12(self.s_copy.clone(), self.dk_words));

        Ok(())
    }

    fn assemble(&self, asm: &mut Assembler) -> errors::Result<()> {
        let k = self.s.len();

        asm.reset(G_RATE)?;

        for _ in 0..SEED_WORDS {
            asm.absorb_word(Io::Input, None, None, 4)?;
        }

        asm.finish(&[k as u8, SHA3])?;
        asm.split(SEED_LANES, self.rho_words, Some(self.rho))?;
        asm.squeeze_lanes(SEED_LANES, self.sigma)?;

        asm.reset(RATE)?;

        for (stream, count) in [
            (self.ek_words, k * POLY_WORDS),
            (self.rho_words, SEED_WORDS),
        ] {
            for i in 0..count {
                asm.absorb_word(Io::Output, Some((stream, i as u16)), None, 4)?;
            }
        }

        asm.finish(&[SHA3])?;
        asm.split(SEED_LANES, self.h_words, None)?;

        for (stream, count) in [(self.h_words, SEED_WORDS), (self.dk_words, k * POLY_WORDS)] {
            for i in 0..count {
                asm.output((stream, i as u16));
            }
        }

        Ok(())
    }
}

#[derive(Clone, Debug)]
struct Encrypt {
    ek_words: Stream,
    rho_words: Stream,
    rho: Stream,
    h_words: Stream,
    r: Stream,
    c1: Stream,
    c2: Stream,
    t_hat: Vec<Poly>,
    a: Vec<Vec<Poly>>,
    y: Vec<Poly>,
    e1: Vec<Poly>,
    e2: Poly,
    y_hat: Vec<Poly>,
    mu: Poly,
    sum: Poly,
    u_hat: Vec<Poly>,
    v_hat: Poly,
    u: Vec<Poly>,
    v: Poly,
}

impl Encrypt {
    fn new(k: usize, labels: &mut PolyLabels) -> errors::Result<Self> {
        Ok(Self {
            ek_words: labels.stream()?,
            rho_words: labels.stream()?,
            rho: labels.stream()?,
            h_words: labels.stream()?,
            r: labels.stream()?,
            c1: labels.stream()?,
            c2: labels.stream()?,
            t_hat: polys(k, labels)?,
            a: matrix(k, labels)?,
            y: polys(k, labels)?,
            e1: polys(k, labels)?,
            e2: labels.fresh()?,
            y_hat: polys(k, labels)?,
            mu: labels.fresh()?,
            sum: labels.fresh()?,
            u_hat: polys(k, labels)?,
            v_hat: labels.fresh()?,
            u: polys(k, labels)?,
            v: labels.fresh()?,
        })
    }

    fn sampling(&self, params: &MlKemParams) -> Vec<SamplerStep> {
        let noise = self
            .y
            .iter()
            .map(|&out| (params.eta1(), out))
            .chain(self.e1.iter().map(|&out| (params.eta2(), out)))
            .chain([(params.eta2(), self.e2)])
            .enumerate()
            .map(|(n, (eta, out))| SamplerStep::prf(self.r, n as u8, eta, out));

        sample_a(self.rho, &self.a).chain(noise).collect()
    }

    fn steps(&self, params: &MlKemParams, steps: &mut Steps) -> errors::Result<()> {
        let k = self.y.len();

        steps.sampler.extend(self.sampling(params));

        for (&y, &y_hat) in self.y.iter().zip(&self.y_hat) {
            steps
                .forward
                .push(NttStep::Transform(Transform::forward(y, y_hat)));
        }

        steps.adds.push(NttStep::Add {
            a: self.e2,
            b: self.mu,
            out: self.sum,
        });

        for ((&u_hat, &e1), &u) in self.u_hat.iter().zip(&self.e1).zip(&self.u) {
            steps
                .inverse
                .push(NttStep::Transform(Transform::inverse_plus(u_hat, e1, u)));
        }

        steps
            .inverse
            .push(NttStep::Transform(Transform::inverse_plus(
                self.v_hat, self.sum, self.v,
            )));

        let transposed = (0..k)
            .map(|i| (0..k).map(|j| self.a[j][i]).collect())
            .chain([self.t_hat.clone()])
            .collect();

        let out = self.u_hat.iter().copied().chain([self.v_hat]).collect();

        steps
            .products
            .push(BaseCaseMac::new(transposed, self.y_hat.clone(), None, out)?);

        steps
            .codec
            .push(CodecStep::compress(params.du(), self.u.clone(), self.c1)?);
        steps
            .codec
            .push(CodecStep::compress(params.dv(), vec![self.v], self.c2)?);

        Ok(())
    }

    fn hash_ek(&self, asm: &mut Assembler) -> errors::Result<()> {
        let k = self.y.len();

        for (stream, count) in [
            (self.ek_words, k * POLY_WORDS),
            (self.rho_words, SEED_WORDS),
        ] {
            for i in 0..count {
                asm.absorb_word(Io::Input, Some((stream, i as u16)), None, 4)?;
            }
        }

        asm.finish(&[SHA3])?;
        asm.split(SEED_LANES, self.h_words, None)?;

        asm.reset(RATE)?;

        for i in 0..SEED_WORDS {
            let lane = (i % 2 == 1).then_some((self.rho, (i / 2) as u16));

            asm.absorb_word(Io::None, Some((self.rho_words, i as u16)), lane, 4)?;
        }

        Ok(())
    }
}

#[derive(Clone, Debug)]
struct Encaps {
    enc: Encrypt,
    m_words: Stream,
    k_words: Stream,
}

impl Encaps {
    fn new(k: usize, labels: &mut PolyLabels) -> errors::Result<Self> {
        Ok(Self {
            enc: Encrypt::new(k, labels)?,
            m_words: labels.stream()?,
            k_words: labels.stream()?,
        })
    }

    fn steps(&self, params: &MlKemParams, steps: &mut Steps) -> errors::Result<()> {
        let enc = &self.enc;

        steps.codec.push(CodecStep::decode12_canonical(
            enc.ek_words,
            enc.t_hat.clone(),
        ));
        steps
            .codec
            .push(CodecStep::decompress(1, self.m_words, vec![enc.mu])?);

        enc.steps(params, steps)
    }

    fn assemble(&self, params: &MlKemParams, asm: &mut Assembler) -> errors::Result<()> {
        let enc = &self.enc;

        asm.reset(RATE)?;

        enc.hash_ek(asm)?;

        asm.reset(G_RATE)?;

        for stream in [self.m_words, enc.h_words] {
            for i in 0..SEED_WORDS {
                asm.absorb_word(Io::Input, Some((stream, i as u16)), None, 4)?;
            }
        }

        asm.finish(&[SHA3])?;
        asm.split(SEED_LANES, self.k_words, None)?;
        asm.squeeze_lanes(SEED_LANES, enc.r)?;

        for (stream, count) in [
            (self.k_words, SEED_WORDS),
            (enc.c1, c1_bytes(params) / 4),
            (enc.c2, c2_bytes(params) / 4),
        ] {
            for i in 0..count {
                asm.output((stream, i as u16));
            }
        }

        Ok(())
    }
}

#[derive(Clone, Debug)]
struct Decaps {
    enc: Encrypt,
    dk_words: Stream,
    c1: Stream,
    c2: Stream,
    c1_relay: Stream,
    c2_relay: Stream,
    m_words: Stream,
    m_twin: Stream,
    k_prime: Stream,
    k_bar: Stream,
    k: Stream,
    valid: Stream,
    s_hat: Vec<Poly>,
    u_prime: Vec<Poly>,
    v_prime: Poly,
    u_prime_hat: Vec<Poly>,
    product: Poly,
    w: Poly,
}

impl Decaps {
    fn new(k: usize, labels: &mut PolyLabels) -> errors::Result<Self> {
        Ok(Self {
            enc: Encrypt::new(k, labels)?,
            dk_words: labels.stream()?,
            c1: labels.stream()?,
            c2: labels.stream()?,
            c1_relay: labels.stream()?,
            c2_relay: labels.stream()?,
            m_words: labels.stream()?,
            m_twin: labels.stream()?,
            k_prime: labels.stream()?,
            k_bar: labels.stream()?,
            k: labels.stream()?,
            valid: labels.stream()?,
            s_hat: polys(k, labels)?,
            u_prime: polys(k, labels)?,
            v_prime: labels.fresh()?,
            u_prime_hat: polys(k, labels)?,
            product: labels.fresh()?,
            w: labels.fresh()?,
        })
    }

    fn steps(&self, params: &MlKemParams, steps: &mut Steps) -> errors::Result<()> {
        let enc = &self.enc;
        let (du, dv) = (params.du(), params.dv());

        steps.codec.extend([
            CodecStep::decode12(self.dk_words, self.s_hat.clone()),
            CodecStep::decode12(enc.ek_words, enc.t_hat.clone()),
            CodecStep::decompress(du, self.c1_relay, self.u_prime.clone())?,
            CodecStep::decompress(dv, self.c2_relay, vec![self.v_prime])?,
            CodecStep::compress(1, vec![self.w], self.m_words)?.with_twin(self.m_twin),
            CodecStep::decompress(1, self.m_twin, vec![enc.mu])?,
        ]);

        for (&u, &u_hat) in self.u_prime.iter().zip(&self.u_prime_hat) {
            steps
                .forward
                .push(NttStep::Transform(Transform::forward(u, u_hat)));
        }

        steps
            .inverse
            .push(NttStep::Transform(Transform::inverse_minus(
                self.product,
                self.v_prime,
                self.w,
            )));

        steps.products.push(BaseCaseMac::new(
            vec![self.s_hat.clone()],
            self.u_prime_hat.clone(),
            None,
            vec![self.product],
        )?);

        steps.selects.push(KemSelectStep {
            parts: vec![
                CipherPart {
                    c: self.c1,
                    c_prime: enc.c1,
                    relay: self.c1_relay,
                    words: c1_bytes(params) / 4,
                },
                CipherPart {
                    c: self.c2,
                    c_prime: enc.c2,
                    relay: self.c2_relay,
                    words: c2_bytes(params) / 4,
                },
            ],
            k_prime: self.k_prime,
            k_bar: self.k_bar,
            k: self.k,
            valid: self.valid,
        });

        enc.steps(params, steps)
    }

    fn assemble(&self, params: &MlKemParams, asm: &mut Assembler) -> errors::Result<()> {
        let enc = &self.enc;

        asm.reset(RATE)?;

        for i in 0..params.k() * POLY_WORDS {
            asm.input(Some((self.dk_words, i as u16)));
        }

        enc.hash_ek(asm)?;

        asm.reset(G_RATE)?;

        for i in 0..SEED_WORDS {
            asm.absorb_word(Io::None, Some((self.m_words, i as u16)), None, 4)?;
        }

        for i in 0..SEED_WORDS {
            asm.absorb_word(Io::Input, Some((enc.h_words, i as u16)), None, 4)?;
        }

        asm.finish(&[SHA3])?;
        asm.split(SEED_LANES, self.k_prime, None)?;
        asm.squeeze_lanes(SEED_LANES, enc.r)?;

        asm.reset(RATE)?;

        for _ in 0..SEED_WORDS {
            asm.absorb_word(Io::Input, None, None, 4)?;
        }

        for (stream, count) in [
            (self.c1, c1_bytes(params) / 4),
            (self.c2, c2_bytes(params) / 4),
        ] {
            for i in 0..count {
                asm.absorb_word(Io::Input, Some((stream, i as u16)), None, 4)?;
            }
        }

        asm.finish(&[SHAKE])?;
        asm.split(SEED_LANES, self.k_bar, None)?;

        for i in 0..SEED_WORDS {
            asm.output((self.k, i as u16));
        }

        asm.output((self.valid, 0));

        Ok(())
    }
}

#[derive(Default)]
struct Steps {
    sampler: Vec<SamplerStep>,
    codec: Vec<CodecStep>,
    forward: Vec<NttStep>,
    adds: Vec<NttStep>,
    inverse: Vec<NttStep>,
    products: Vec<BaseCaseMac>,
    selects: Vec<KemSelectStep>,
}

struct Reference<'f> {
    params: MlKemParams,
    checked: bool,
    forgeries: &'f [Forgery],
    values: PolyValues,
    words: WordValues,
    lanes: LaneValues,
    inputs: Zeroizing<Vec<u32>>,
    hosted: Zeroizing<Vec<u32>>,
    outputs: Zeroizing<Vec<MlKemOutput>>,
}

impl Reference<'_> {
    fn keygen(&mut self, call: usize, kg: &KeyGen, d: &[u8; SEED_BYTES]) -> errors::Result<()> {
        let k = self.params.k();

        let seed = g(&Zeroizing::new([&d[..], &[k as u8]].concat()));
        let (rho, sigma) = seed.split_at(SEED_BYTES);

        let expanded = self.forged_rho(call);

        self.lanes.insert(
            kg.rho,
            le_lanes(expanded.as_ref().map_or(rho, |forged| &forged[..])).collect(),
        )?;
        self.lanes.insert(kg.sigma, le_lanes(sigma).collect())?;
        self.words.insert(kg.rho_words, le_words(rho).collect())?;

        for step in kg.sampling(&self.params) {
            step.sample(&self.lanes, &mut self.values)?;
        }

        for i in 0..k {
            let s_hat = NttParams::ML_KEM.ntt(self.values.get(kg.s[i])?);
            let e_hat = NttParams::ML_KEM.ntt(self.values.get(kg.e[i])?);

            self.values.insert(kg.s_hat[i], s_hat)?;
            self.values.insert(kg.s_copy[i], s_hat)?;
            self.values.insert(kg.e_hat[i], e_hat)?;
        }

        for i in 0..k {
            let t_hat = self.dot(kg.a[i].iter().copied(), &kg.s_hat, Some(kg.e_hat[i]))?;

            self.values.insert(kg.t_hat[i], *t_hat)?;
        }

        let mut ek = vec![0u8; k * POLY_BYTES + SEED_BYTES];
        let mut dk_pke = Zeroizing::new(vec![0u8; k * POLY_BYTES]);

        for i in 0..k {
            let at = i * POLY_BYTES..(i + 1) * POLY_BYTES;

            byte_encode(12, self.values.get(kg.t_hat[i])?, &mut ek[at.clone()]);
            byte_encode(12, self.values.get(kg.s_hat[i])?, &mut dk_pke[at]);
        }

        ek[k * POLY_BYTES..].copy_from_slice(rho);

        let hash = h(&ek);

        self.words
            .insert(kg.ek_words, le_words(&ek[..k * POLY_BYTES]).collect())?;
        self.words
            .insert(kg.dk_words, le_words(&dk_pke).collect())?;
        self.words
            .insert(kg.h_words, le_words(&hash[..]).collect())?;

        self.host(&[d], &[&ek, &hash[..], &dk_pke]);

        self.outputs.push(MlKemOutput::KeyGen {
            ek,
            dk_pke: core::mem::take(&mut *dk_pke),
            h: *hash,
        });

        Ok(())
    }

    fn encaps(
        &mut self,
        call: usize,
        en: &Encaps,
        ek: &[u8],
        m: &[u8; SEED_BYTES],
    ) -> errors::Result<()> {
        let k = self.params.k();

        if ek.len() != k * POLY_BYTES + SEED_BYTES {
            return Err(rejected(
                "encapsulation key length differs from the parameter set",
            ));
        }

        let (ek_t, rho) = ek.split_at(k * POLY_BYTES);

        if self.checked && !bool::from(canonical12(ek_t)) {
            return Err(rejected(
                "encapsulation key fails the FIPS 203 modulus check",
            ));
        }

        let split = h(ek);

        let mut message = Zeroizing::new(*m);
        let mut hash = split.clone();

        for forgery in self.forgeries {
            match *forgery {
                Forgery::Message { call: at, m } if at == call => *message = m,
                Forgery::KeyHash { call: at, h } if at == call => *hash = h,
                _ => {}
            }
        }

        let seed = g(&Zeroizing::new([&m[..], &hash[..]].concat()));
        let (key, r) = seed.split_at(SEED_BYTES);

        self.hash_ek(call, &en.enc, ek_t, rho, &split[..])?;

        self.words
            .insert(en.m_words, le_words(&message[..]).collect())?;
        self.words.insert(en.k_words, le_words(key).collect())?;
        self.lanes.insert(en.enc.r, le_lanes(r).collect())?;

        self.values
            .insert(en.enc.mu, decompress_poly(1, &byte_decode(1, &message[..])))?;

        let mut c = vec![0u8; c1_bytes(&self.params) + c2_bytes(&self.params)];
        self.encrypt(&en.enc, ek_t, &mut c)?;

        self.host(&[ek, m, &hash[..]], &[key, &c]);

        self.outputs.push(MlKemOutput::Encaps {
            key: core::array::from_fn(|i| key[i]),
            c,
        });

        Ok(())
    }

    fn decaps(&mut self, call: usize, de: &Decaps, dk: &[u8], c: &[u8]) -> errors::Result<()> {
        let (k, du, dv) = (self.params.k(), self.params.du(), self.params.dv());
        let c1_len = c1_bytes(&self.params);

        if dk.len() != 2 * k * POLY_BYTES + 3 * SEED_BYTES
            || c.len() != c1_len + c2_bytes(&self.params)
        {
            return Err(rejected(
                "decapsulation key or ciphertext length differs from the parameter set",
            ));
        }

        let (dk_pke, rest) = dk.split_at(k * POLY_BYTES);
        let (ek, rest) = rest.split_at(k * POLY_BYTES + SEED_BYTES);
        let (hash, z) = rest.split_at(SEED_BYTES);

        let split = h(ek);

        if self.checked && !bool::from(split[..].ct_eq(hash)) {
            return Err(rejected("decapsulation key fails the FIPS 203 hash check"));
        }

        let enc = &de.enc;
        let (ek_t, rho) = ek.split_at(k * POLY_BYTES);
        let (c1, c2) = c.split_at(c1_len);
        let c1_poly = c1_len / k;

        self.hash_ek(call, enc, ek_t, rho, &split[..])?;

        let mut decoded = Zeroizing::new(dk_pke.to_vec());

        for forgery in self.forgeries {
            match *forgery {
                Forgery::DkPke {
                    call: at,
                    word,
                    mask,
                } if at == call => {
                    for (byte, m) in decoded.iter_mut().skip(4 * word).zip(mask.to_le_bytes()) {
                        *byte ^= m;
                    }
                }
                _ => {}
            }
        }

        self.words
            .insert(de.dk_words, le_words(&decoded).collect())?;

        for i in 0..k {
            let s_hat = byte_decode(12, &decoded[i * POLY_BYTES..(i + 1) * POLY_BYTES]);
            let u = byte_decode(du, &c1[i * c1_poly..(i + 1) * c1_poly]);
            let u_prime = decompress_poly(du, &u);

            self.values.insert(de.s_hat[i], s_hat)?;
            self.values.insert(de.u_prime[i], u_prime)?;
            self.values
                .insert(de.u_prime_hat[i], NttParams::ML_KEM.ntt(&u_prime))?;
        }

        let v_prime = decompress_poly(dv, &byte_decode(dv, c2));
        let product = self.dot(de.s_hat.iter().copied(), &de.u_prime_hat, None)?;
        let w = Zeroizing::new(sub(&v_prime, &NttParams::ML_KEM.intt(&product)));

        self.values.insert(de.v_prime, v_prime)?;
        self.values.insert(de.product, *product)?;
        self.values.insert(de.w, *w)?;

        let mut m = Zeroizing::new([0u8; SEED_BYTES]);
        byte_encode(1, &compress_poly(1, &w), &mut m[..]);

        self.words.insert(de.m_words, le_words(&m[..]).collect())?;
        self.words.insert(de.m_twin, le_words(&m[..]).collect())?;
        self.values
            .insert(enc.mu, decompress_poly(1, &byte_decode(1, &m[..])))?;

        let seed = g(&Zeroizing::new([&m[..], hash].concat()));
        let (k_prime, r) = seed.split_at(SEED_BYTES);
        let k_bar = j(&Zeroizing::new([z, c].concat()));

        self.words.insert(de.k_prime, le_words(k_prime).collect())?;
        self.words
            .insert(de.k_bar, le_words(&k_bar[..]).collect())?;
        self.lanes.insert(enc.r, le_lanes(r).collect())?;

        for (stream, part) in [
            (de.c1, c1),
            (de.c1_relay, c1),
            (de.c2, c2),
            (de.c2_relay, c2),
        ] {
            self.words.insert(stream, le_words(part).collect())?;
        }

        let mut c_prime = Zeroizing::new(vec![0u8; c.len()]);
        self.encrypt(enc, ek_t, &mut c_prime)?;

        let valid = c.ct_eq(&c_prime[..]);
        let key = Zeroizing::new(core::array::from_fn::<u8, SEED_BYTES, _>(|i| {
            u8::conditional_select(&k_bar[i], &k_prime[i], valid)
        }));

        self.words.insert(de.k, le_words(&key[..]).collect())?;
        self.words
            .insert(de.valid, vec![u32::from(valid.unwrap_u8())])?;

        self.host(&[dk_pke, ek, hash, z, c], &[&key[..], &[valid.unwrap_u8()]]);

        self.outputs.push(MlKemOutput::Decaps {
            key: *key,
            valid: bool::from(valid),
        });

        Ok(())
    }

    fn hash_ek(
        &mut self,
        call: usize,
        enc: &Encrypt,
        ek_t: &[u8],
        rho: &[u8],
        hash: &[u8],
    ) -> errors::Result<()> {
        self.words.insert(enc.ek_words, le_words(ek_t).collect())?;
        self.words.insert(enc.rho_words, le_words(rho).collect())?;
        self.words.insert(enc.h_words, le_words(hash).collect())?;

        let expanded = self.forged_rho(call);
        let seed = expanded.as_ref().map_or(rho, |forged| &forged[..]);

        self.lanes.insert(enc.rho, le_lanes(seed).collect())
    }

    fn forged_rho(&self, call: usize) -> Option<[u8; SEED_BYTES]> {
        self.forgeries.iter().find_map(|f| match *f {
            Forgery::Rho { call: at, rho } if at == call => Some(rho),
            _ => None,
        })
    }

    fn encrypt(&mut self, enc: &Encrypt, ek_t: &[u8], c: &mut [u8]) -> errors::Result<()> {
        let (k, du, dv) = (self.params.k(), self.params.du(), self.params.dv());
        let c1_len = c1_bytes(&self.params);
        let c1_poly = c1_len / k;

        for i in 0..k {
            let t_hat = byte_decode(12, &ek_t[i * POLY_BYTES..(i + 1) * POLY_BYTES]);

            self.values.insert(enc.t_hat[i], t_hat)?;
        }

        for step in enc.sampling(&self.params) {
            step.sample(&self.lanes, &mut self.values)?;
        }

        for i in 0..k {
            let y_hat = NttParams::ML_KEM.ntt(self.values.get(enc.y[i])?);

            self.values.insert(enc.y_hat[i], y_hat)?;
        }

        for i in 0..k {
            let u_hat = self.dot((0..k).map(|j| enc.a[j][i]), &enc.y_hat, None)?;
            let u = add(&NttParams::ML_KEM.intt(&u_hat), self.values.get(enc.e1[i])?);

            self.values.insert(enc.u_hat[i], *u_hat)?;
            self.values.insert(enc.u[i], u)?;

            byte_encode(
                du,
                &compress_poly(du, &u),
                &mut c[i * c1_poly..(i + 1) * c1_poly],
            );
        }

        let v_hat = self.dot(enc.t_hat.iter().copied(), &enc.y_hat, None)?;
        let sum = add(self.values.get(enc.e2)?, self.values.get(enc.mu)?);
        let v = add(&NttParams::ML_KEM.intt(&v_hat), &sum);

        self.values.insert(enc.v_hat, *v_hat)?;
        self.values.insert(enc.sum, sum)?;
        self.values.insert(enc.v, v)?;

        byte_encode(dv, &compress_poly(dv, &v), &mut c[c1_len..]);

        self.words
            .insert(enc.c1, le_words(&c[..c1_len]).collect())?;
        self.words.insert(enc.c2, le_words(&c[c1_len..]).collect())
    }

    fn dot(
        &self,
        row: impl IntoIterator<Item = Poly>,
        column: &[Poly],
        seed: Option<Poly>,
    ) -> errors::Result<Zeroizing<Coeffs>> {
        let mut acc = Zeroizing::new(match seed {
            Some(poly) => *self.values.get(poly)?,
            None => [0; N],
        });

        for (a, &b) in row.into_iter().zip(column) {
            mac(&mut acc, self.values.get(a)?, self.values.get(b)?);
        }

        Ok(acc)
    }

    fn host(&mut self, inputs: &[&[u8]], outputs: &[&[u8]]) {
        for bytes in inputs {
            self.inputs.extend(le_words(bytes));
            self.hosted.extend(le_words(bytes));
        }

        for bytes in outputs {
            self.hosted.extend(le_words(bytes));
        }
    }
}

/// ML-KEM KeyGen, Encaps and Decaps (FIPS 203)
/// for 1 to 256 calls, as the tables of `defs`.
#[derive(Clone)]
pub struct MlKemChiplet<F: TowerField + TraceCompatibleField> {
    params: MlKemParams,
    calls: Vec<Call>,
    ctrl: CtrlChiplet<F>,
    sampler: SamplerChiplet<F>,
    keccak: KeccakChiplet,
    keccak_rows: usize,
    codec: CodecChiplet<F>,
    ntt: NttChiplet<F>,
    poly_arith: PolyArithChiplet<F>,
    kem_select: Option<KemSelectChiplet<F>>,
}

impl<F> MlKemChiplet<F>
where
    F: TowerField + TraceCompatibleField + PackableField + HardwareField + Send + 'static,
    <F as PackableField>::Packed: Copy + Send + Sync,
    Flat<F>: Send + Sync,
{
    /// Compiles the tables proving `calls`, in order,
    /// under one parameter set. Takes 1 to 256 calls.
    pub fn new(params: MlKemParams, calls: &[MlKemCall]) -> errors::Result<Self> {
        if calls.is_empty() || calls.len() > MAX_CALLS {
            return Err(rejected("mlkem pipeline serves 1 to 256 calls"));
        }

        let k = params.k();
        let mut labels = PolyLabels::new();

        let calls = calls
            .iter()
            .map(|call| match call {
                MlKemCall::KeyGen => KeyGen::new(k, &mut labels).map(Call::KeyGen),
                MlKemCall::Encaps => Encaps::new(k, &mut labels).map(Call::Encaps),
                MlKemCall::Decaps => Decaps::new(k, &mut labels).map(Call::Decaps),
            })
            .collect::<errors::Result<Vec<Call>>>()?;

        let mut steps = Steps::default();
        let mut asm = Assembler::new();

        for call in &calls {
            match call {
                Call::KeyGen(kg) => {
                    kg.steps(&params, &mut steps)?;
                    kg.assemble(&mut asm)?;
                }
                Call::Encaps(en) => {
                    en.steps(&params, &mut steps)?;
                    en.assemble(&params, &mut asm)?;
                }
                Call::Decaps(de) => {
                    de.steps(&params, &mut steps)?;
                    de.assemble(&params, &mut asm)?;
                }
            }
        }

        let Steps {
            sampler,
            codec,
            forward,
            adds,
            inverse,
            products,
            selects,
        } = steps;

        let codec_rows = codec.iter().map(CodecStep::rows).sum::<usize>();
        let codec = CodecChiplet::new(codec, height(codec_rows))?;

        let sampler_rows = sampler.iter().map(SamplerStep::rows).sum::<usize>();
        let sampler_blocks = sampler.iter().map(SamplerStep::blocks).sum::<usize>();
        let sampler = SamplerChiplet::new(sampler, height(sampler_rows))?;

        let schedule = forward.into_iter().chain(adds).chain(inverse).collect();
        let schedule = NttSchedule::new(NttParams::ML_KEM, schedule, &mut labels)?;
        let ntt_rows = height(schedule.rows());
        let ntt = NttChiplet::new(schedule, ntt_rows)?;

        let product_rows = products.iter().map(BaseCaseMac::rows).sum::<usize>();
        let poly_arith = PolyArithChiplet::new(products, height(product_rows))?;

        let kem_select = match selects.is_empty() {
            true => None,
            false => {
                let rows = selects.iter().map(KemSelectStep::rows).sum::<usize>();

                Some(KemSelectChiplet::new(selects, height(rows))?)
            }
        };

        let rows = asm.into_rows();
        let ctrl_rows = height(rows.len());
        let ctrl = CtrlChiplet::new("MlKemCtrl", &service(), rows, ctrl_rows)?;

        let blocks = ctrl.blocks() + sampler_blocks;
        let keccak_rows = height(blocks * KeccakChiplet::BLOCK_ROWS);
        let keccak = KeccakChiplet::new(keccak_rows, blocks);

        Ok(Self {
            params,
            calls,
            ctrl,
            sampler,
            keccak,
            keccak_rows,
            codec,
            ntt,
            poly_arith,
            kem_select,
        })
    }

    /// The tables as chiplets for a host program to attach,
    /// in the order `trace` returns their traces; KemSelect
    /// comes last and only when a call decapsulates.
    pub fn defs(&self) -> errors::Result<Vec<ChipletDef<F>>> {
        let mut defs = vec![
            self.ctrl.def()?,
            self.sampler.def()?,
            ChipletDef::from_air(&self.keccak)?,
            self.codec.def()?,
            self.ntt.def()?,
            self.poly_arith.def()?,
        ];

        if let Some(kem_select) = &self.kem_select {
            defs.push(kem_select.def()?);
        }

        Ok(defs)
    }

    /// Traces every table for one input per call, after
    /// the FIPS 203 input checks: the modulus check on
    /// an Encaps ek and the hash check on a Decaps dk.
    pub fn trace(&self, inputs: &[MlKemInput<'_>]) -> errors::Result<MlKemWitness> {
        self.witness(inputs, &[], true)
    }

    /// `trace` without the FIPS 203 modulus and hash checks and
    /// with `forgeries` applied, for tests that expect rejection.
    #[cfg(feature = "forgery")]
    pub fn trace_forged(
        &self,
        inputs: &[MlKemInput<'_>],
        forgeries: &[Forgery],
    ) -> errors::Result<MlKemWitness> {
        self.witness(inputs, forgeries, false)
    }

    fn witness(
        &self,
        inputs: &[MlKemInput<'_>],
        forgeries: &[Forgery],
        checked: bool,
    ) -> errors::Result<MlKemWitness> {
        if inputs.len() != self.calls.len() {
            return Err(rejected("one input per call of the mlkem pipeline"));
        }

        for forgery in forgeries {
            let fits = match *forgery {
                Forgery::Message { call, .. } | Forgery::KeyHash { call, .. } => {
                    matches!(self.calls.get(call), Some(Call::Encaps(_)))
                }
                Forgery::DkPke { call, .. } => {
                    matches!(self.calls.get(call), Some(Call::Decaps(_)))
                }
                Forgery::Rho { call, .. } => call < self.calls.len(),
            };

            if !fits {
                return Err(rejected("the forgery names a call of another kind"));
            }
        }

        let (sent, received) = self.ctrl.host_words();

        let mut rf = Reference {
            params: self.params,
            checked,
            forgeries,
            values: PolyValues::default(),
            words: WordValues::default(),
            lanes: LaneValues::default(),
            inputs: Zeroizing::new(Vec::with_capacity(sent)),
            hosted: Zeroizing::new(Vec::with_capacity(sent + received)),
            outputs: Zeroizing::new(Vec::with_capacity(inputs.len())),
        };

        for (i, (call, input)) in self.calls.iter().zip(inputs).enumerate() {
            match (call, input) {
                (Call::KeyGen(kg), MlKemInput::KeyGen { d }) => rf.keygen(i, kg, d)?,
                (Call::Encaps(en), MlKemInput::Encaps { ek, m }) => rf.encaps(i, en, ek, m)?,
                (Call::Decaps(de), MlKemInput::Decaps { dk, c }) => rf.decaps(i, de, dk, c)?,
                _ => return Err(rejected("input kind differs from its call")),
            }
        }

        let (sampler, sampled) = self.sampler.trace(&rf.lanes, &mut rf.values)?;

        let (codec, _) = self
            .codec
            .trace_checked(&mut rf.words, &mut rf.values, checked)?;

        let ntt = self.ntt.trace(&mut rf.values)?;
        let poly_arith = self.poly_arith.trace(&mut rf.values)?;

        let kem_select = self
            .kem_select
            .as_ref()
            .map(|table| table.trace(&mut rf.words))
            .transpose()?;

        let ctrl = self
            .ctrl
            .trace(&rf.inputs, &LaneValues::default(), &mut rf.words)?;

        let states: Zeroizing<Vec<[Block64; 25]>> = Zeroizing::new(
            ctrl.keccak
                .iter()
                .chain(sampled.iter())
                .map(|state| state.map(Block64))
                .collect(),
        );

        let keccak = generate_keccak_trace(&states, self.keccak_rows)?;

        let mut traces = vec![ctrl.trace, sampler, keccak, codec, ntt, poly_arith];
        traces.extend(kem_select);

        Ok(MlKemWitness {
            traces,
            words: rf.hosted,
            outputs: rf.outputs,
        })
    }
}

/// The service a host calls with each call's input words;
/// it answers with the call's output words.
pub fn service() -> Service {
    Service {
        bus_id: MLKEM_DATA_BUS_ID,
        kind: BusKind::Permutation,
        slots: vec![
            ServiceSlot::Value(b"kappa_mlkem_word"),
            ServiceSlot::EmitRank,
        ],
    }
}

fn sample_a(rho: Stream, a: &[Vec<Poly>]) -> impl Iterator<Item = SamplerStep> + '_ {
    a.iter().enumerate().flat_map(move |(i, row)| {
        row.iter()
            .enumerate()
            .map(move |(j, &out)| SamplerStep::sample_ntt(rho, i as u8, j as u8, out))
    })
}

fn polys(n: usize, labels: &mut PolyLabels) -> errors::Result<Vec<Poly>> {
    (0..n).map(|_| labels.fresh()).collect()
}

fn matrix(k: usize, labels: &mut PolyLabels) -> errors::Result<Vec<Vec<Poly>>> {
    (0..k).map(|_| polys(k, labels)).collect()
}

fn c1_bytes(params: &MlKemParams) -> usize {
    32 * params.du() as usize * params.k()
}

fn c2_bytes(params: &MlKemParams) -> usize {
    32 * params.dv() as usize
}

fn rejected(message: &'static str) -> Error {
    Error::Protocol {
        protocol: "mlkem_chiplet",
        message,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeMap;
    use alloc::string::String;
    use hekate_core::trace::{ColumnType, TraceBuilder};
    use hekate_math::{Bit, Block16, Block32, Block128};
    use hekate_program::circuit::Circuit;
    use hekate_program::{Air, FixedShape, ProgramInstance, ProgramWitness};
    use hekate_scribble::{MutationKind, ScribbleConfig, Target, assert_all_caught};
    use hekate_sdk::preflight::preflight;

    use crate::census::{Key, census, unbalanced};
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

    fn crossings(call: &Call) -> (Vec<Stream>, Vec<Stream>) {
        match call {
            Call::KeyGen(kg) => (vec![kg.ek_words, kg.dk_words], vec![kg.rho, kg.sigma]),
            Call::Encaps(en) => (
                vec![en.enc.ek_words, en.m_words, en.enc.c1, en.enc.c2],
                vec![en.enc.rho, en.enc.r],
            ),
            Call::Decaps(de) => (
                vec![
                    de.dk_words,
                    de.enc.ek_words,
                    de.m_words,
                    de.k_prime,
                    de.c1,
                    de.c2,
                    de.k_bar,
                    de.k,
                    de.valid,
                ],
                vec![de.enc.rho, de.enc.r],
            ),
        }
    }

    fn producers(pipeline: &MlKemChiplet<F>) -> Vec<(String, Vec<(&'static str, u16)>)> {
        let mut tables = vec![
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
                pipeline.poly_arith.def().unwrap().name(),
                pipeline.poly_arith.produced(),
            ),
        ];

        if let Some(table) = &pipeline.kem_select {
            tables.push((table.def().unwrap().name(), table.produced()));
        }

        tables
    }

    fn encaps_with(
        table: &str,
        edit: impl FnOnce(&mut ChipletDef<F>),
    ) -> Result<BTreeMap<Key, Vec<(String, bool)>>, String> {
        let pipeline =
            MlKemChiplet::<F>::new(MlKemParams::ML_KEM_512, &[MlKemCall::Encaps]).unwrap();

        let mut defs = pipeline.defs().unwrap();

        edit(defs.iter_mut().find(|def| def.name() == table).unwrap());

        census(&defs, &pipeline.sampler, &producers(&pipeline))
    }

    #[test]
    fn no_key_misses_or_repeats_emitter() {
        let calls = [
            MlKemCall::KeyGen,
            MlKemCall::Encaps,
            MlKemCall::Decaps,
            MlKemCall::Decaps,
        ];

        for params in [
            MlKemParams::ML_KEM_512,
            MlKemParams::ML_KEM_768,
            MlKemParams::ML_KEM_1024,
        ] {
            let pipeline = MlKemChiplet::<F>::new(params, &calls).unwrap();
            let keys = census(
                &pipeline.defs().unwrap(),
                &pipeline.sampler,
                &producers(&pipeline),
            )
            .unwrap();

            let broken = unbalanced(&keys);
            assert!(broken.is_empty(), "{broken:?}");

            for bus in [COEF_BUS_ID, WORD_BUS_ID, LANE_BUS_ID] {
                assert!(keys.keys().any(|(b, _, _)| b == bus));
            }
        }
    }

    #[test]
    fn census_flags_key_without_producer() {
        let pipeline =
            MlKemChiplet::<F>::new(MlKemParams::ML_KEM_512, &[MlKemCall::Encaps]).unwrap();

        let mut tables = producers(&pipeline);
        tables.retain(|(name, _)| name != "NttChiplet");

        let keys = census(&pipeline.defs().unwrap(), &pipeline.sampler, &tables).unwrap();

        assert!(!unbalanced(&keys).is_empty());
    }

    #[test]
    fn census_refuses_idle_producer() {
        let pipeline =
            MlKemChiplet::<F>::new(MlKemParams::ML_KEM_512, &[MlKemCall::Encaps]).unwrap();

        let mut tables = producers(&pipeline);
        tables[0].1.push((COEF_BUS_ID, u16::MAX));

        assert!(census(&pipeline.defs().unwrap(), &pipeline.sampler, &tables).is_err());
    }

    #[test]
    fn census_flags_doubled_endpoint() {
        let keys = encaps_with("NttChiplet", |ntt| {
            let doubled = ntt
                .permutation_checks
                .iter()
                .find(|(bus, _)| bus == COEF_BUS_ID)
                .cloned()
                .unwrap();

            ntt.permutation_checks.push(doubled);
        })
        .unwrap();

        assert!(!unbalanced(&keys).is_empty());
    }

    #[test]
    fn census_flags_dropped_endpoint() {
        let keys = encaps_with("NttChiplet", |ntt| {
            let at = ntt
                .permutation_checks
                .iter()
                .position(|(bus, _)| bus == COEF_BUS_ID)
                .unwrap();

            ntt.permutation_checks.remove(at);
        })
        .unwrap();

        assert!(!unbalanced(&keys).is_empty());
    }

    #[test]
    fn census_refuses_unpinned_label() {
        let refused = encaps_with("NttChiplet", |ntt| {
            let (_, spec) = ntt
                .permutation_checks
                .iter_mut()
                .find(|(bus, _)| bus == COEF_BUS_ID)
                .unwrap();

            spec.sources[0].0 = spec.sources[2].0.clone();
        });

        assert!(refused.is_err());
    }

    #[test]
    fn census_refuses_witness_label_beside_unpinned_position() {
        let refused = encaps_with("SamplerChiplet", |sampler| {
            let (_, spec) = sampler
                .permutation_checks
                .iter_mut()
                .find(|(bus, _)| bus == COEF_BUS_ID)
                .unwrap();

            spec.sources[1].0 = spec.sources[2].0.clone();
        });

        assert!(refused.is_err());
    }

    #[test]
    fn scribble_ctrl_flip_selector_caught() {
        let params = MlKemParams::ML_KEM_512;
        let calls = [MlKemCall::KeyGen, MlKemCall::Encaps, MlKemCall::Decaps];
        let pipeline = MlKemChiplet::<F>::new(params, &calls).unwrap();

        let [Call::KeyGen(kg), Call::Encaps(en), Call::Decaps(de)] = &pipeline.calls[..] else {
            unreachable!()
        };

        let mut rf = Reference {
            params,
            checked: true,
            forgeries: &[],
            values: PolyValues::default(),
            words: WordValues::default(),
            lanes: LaneValues::default(),
            inputs: Zeroizing::new(Vec::new()),
            hosted: Zeroizing::new(Vec::new()),
            outputs: Zeroizing::new(Vec::new()),
        };

        rf.keygen(0, kg, &[0x5d; 32]).unwrap();

        let MlKemOutput::KeyGen { ek, dk_pke, h } = &rf.outputs[0] else {
            unreachable!()
        };

        let dk = [dk_pke.as_slice(), ek, h, &[0x2e; 32]].concat();
        let ek = ek.clone();

        rf.encaps(1, en, &ek, &[0xa7; 32]).unwrap();

        let MlKemOutput::Encaps { c, .. } = &rf.outputs[1] else {
            unreachable!()
        };

        let c = c.clone();

        rf.decaps(2, de, &dk, &c).unwrap();

        let ctrl = pipeline
            .ctrl
            .trace(&rf.inputs, &LaneValues::default(), &mut rf.words)
            .unwrap();

        let mut words = Vec::new();
        let mut lanes = Vec::new();

        for call in &pipeline.calls {
            let (word_streams, lane_streams) = crossings(call);

            for stream in word_streams {
                for (j, &w) in rf.words.get(stream).unwrap().iter().enumerate() {
                    words.push((stream, j as u16, w));
                }
            }

            for stream in lane_streams {
                for (j, &lane) in rf.lanes.get(stream).unwrap().iter().enumerate() {
                    lanes.push((stream, j as u16, lane));
                }
            }
        }

        let hosted = &rf.hosted;

        let rows = hosted
            .len()
            .max(words.len())
            .max(lanes.len())
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
            pin(words.iter().map(|t| t.0.id() as u64).collect()),
        );
        cx.fix(cols.at(3), pin(words.iter().map(|t| t.1 as u64).collect()));
        cx.fix(cols.at(5), pin(vec![1; words.len()]));
        cx.fix(
            cols.at(6),
            pin(lanes.iter().map(|t| t.0.id() as u64).collect()),
        );
        cx.fix(cols.at(7), pin(lanes.iter().map(|t| t.1 as u64).collect()));
        cx.fix(cols.at(9), pin(vec![1; lanes.len()]));

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

        for (r, &(stream, j, w)) in words.iter().enumerate() {
            tb.set_b16(2, r, Block16(stream.id())).unwrap();
            tb.set_b16(3, r, Block16(j)).unwrap();
            tb.set_b32(4, r, Block32::from(w)).unwrap();
            tb.set_bit(5, r, Bit::from(1u8)).unwrap();
        }

        for (r, &(stream, j, lane)) in lanes.iter().enumerate() {
            tb.set_b16(6, r, Block16(stream.id())).unwrap();
            tb.set_b16(7, r, Block16(j)).unwrap();
            tb.set_b64(8, r, Block64(lane)).unwrap();
            tb.set_bit(9, r, Bit::from(1u8)).unwrap();
        }

        let states: Vec<[Block64; 25]> = ctrl.keccak.iter().map(|s| s.map(Block64)).collect();
        let keccak = generate_keccak_trace(&states, keccak_rows).unwrap();

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
