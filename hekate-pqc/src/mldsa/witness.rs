// SPDX-License-Identifier: Apache-2.0
// This file is part of the hekate project.
// Copyright (C) 2026 Andrei Kochergin <andrei@oumuamua.dev>
// Copyright (C) 2026 Oumuamua Labs <info@oumuamua.dev>. All rights reserved.
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

use super::arithmetic::{
    decode_hint, decode_t1, decode_z_poly, expand_a, mod_add, mod_mul, mod_neg, mod_pow, mod_sub,
    poly_add, poly_sub, sample_in_ball, use_hint, w1_encode, zeta_powers,
};
use super::{MLDSA_Q, MlDsaLevel, N, Phase};
use crate::high_bits::HighBitsOp;
use crate::norm_check::NormCheckOp;
use crate::ntt;
use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;
use core::iter::repeat_n;
use hekate_gadgets::chiplets::ram;
use hekate_keccak as keccak;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum KeccakCallTag {
    ExpandA(u8, u8),
    HashPk,
    HashMu,
    SampleInBall,
    HashCompare,
    Other,
}

// =================================================================
// Key / Signature Types
// =================================================================

/// ML-DSA public key (FIPS 204 §5.2).
#[derive(Clone)]
pub struct MlDsaPublicKey {
    pub level: MlDsaLevel,

    /// Seed for matrix A.
    pub rho: [u8; 32],

    /// t1 polynomials (k polys, 10-bit coefficients).
    pub t1: Vec<[u32; N]>,

    /// Raw pk bytes (for hashing).
    pub pk_bytes: Vec<u8>,
}

impl MlDsaPublicKey {
    pub fn from_bytes(level: MlDsaLevel, bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), level.pk_bytes());

        let rho: [u8; 32] = bytes[..32].try_into().unwrap();

        let t1_bytes = &bytes[32..];
        let poly_bytes = 10 * N / 8; // 320 bytes per poly

        let mut t1 = Vec::with_capacity(level.k);
        for i in 0..level.k {
            t1.push(decode_t1(&t1_bytes[i * poly_bytes..(i + 1) * poly_bytes]));
        }

        Self {
            level,
            rho,
            t1,
            pk_bytes: bytes.to_vec(),
        }
    }
}

/// ML-DSA signature (FIPS 204 §5.2).
#[derive(Clone)]
pub struct MlDsaSignature {
    pub level: MlDsaLevel,

    /// Commitment hash c̃.
    pub c_tilde: Vec<u8>,

    /// Response vector z (l polynomials).
    pub z: Vec<[u32; N]>,

    /// Hint vector h (k polynomials of booleans).
    pub h: Vec<[bool; N]>,
}

impl MlDsaSignature {
    pub fn from_bytes(level: MlDsaLevel, bytes: &[u8]) -> Option<Self> {
        if bytes.len() != level.sig_bytes() {
            return None;
        }

        // c̃ length depends on security level
        let c_tilde_len = match level.k {
            4 => 32,
            6 => 48,
            8 => 64,
            _ => return None,
        };

        let c_tilde = bytes[..c_tilde_len].to_vec();

        // z polynomials
        let gamma1_bits = if level.gamma1 == (1 << 17) { 18 } else { 20 };
        let z_poly_bytes = gamma1_bits * N / 8;

        let mut z = Vec::with_capacity(level.l);
        let z_start = c_tilde_len;

        for i in 0..level.l {
            let start = z_start + i * z_poly_bytes;
            z.push(decode_z_poly(
                &bytes[start..start + z_poly_bytes],
                level.gamma1,
            ));
        }

        // Hint vector
        let h_start = z_start + level.l * z_poly_bytes;
        let h = decode_hint(&bytes[h_start..], level.k, level.omega)?;

        Some(Self {
            level,
            c_tilde,
            z,
            h,
        })
    }
}

// =================================================================
// ML-DSA Verify Result
// =================================================================

/// All intermediate data from ML-DSA
/// verification, needed for trace generation.
pub struct MlDsaVerifyResult {
    #[allow(dead_code)]
    pub valid: bool,

    pub keccak_calls: Vec<keccak::KeccakCall>,
    pub keccak_sponge_meta: Vec<(bool, bool, bool)>,
    pub ram_events: Vec<ram::MemoryEvent>,
    pub norm_check_ops: Vec<NormCheckOp>,
    pub highbits_ops: Vec<HighBitsOp>,

    pub c_tilde: Vec<u8>,

    /// Hash comparison (c̃ vs c̃').
    #[allow(dead_code)]
    pub c_tilde_prime: Vec<u8>,

    pub ntt_phases: Vec<Phase>,
    pub keccak_phases: Vec<Phase>,
    pub ram_phases: Vec<Phase>,
    pub norm_phases: Vec<Phase>,
    pub highbits_phases: Vec<Phase>,

    pub ntt_ram_bindings: Vec<(usize, usize)>,
    pub w_side_bindings: Vec<(usize, u32)>,
    pub ntt_boundary_bindings: Vec<(u32, u32, usize, bool)>,

    #[allow(dead_code)]
    pub next_free_addr: u32,

    /// Keccak call tagging for BIND_SEEN constraints.
    #[allow(dead_code)]
    pub call_tag: Vec<KeccakCallTag>,
}

// =================================================================
// ML-DSA.Verify (FIPS 204 Algorithm 3)
// =================================================================

pub fn ml_dsa_verify_traced(
    pk: &MlDsaPublicKey,
    sig: &MlDsaSignature,
    msg: &[u8],
) -> (MlDsaVerifyResult, Vec<ntt::NttOp>) {
    const SLOT: u32 = 256;

    let level = pk.level;

    let mut keccak_calls = Vec::new();
    let mut keccak_sponge_meta: Vec<(bool, bool, bool)> = Vec::new();
    let mut ntt_ops = Vec::new();
    let mut ram_events: Vec<ram::MemoryEvent> = Vec::new();
    let mut ram_clk = 0u32;

    let mut ntt_phases: Vec<Phase> = Vec::new();
    let mut keccak_phases: Vec<Phase> = Vec::new();
    let mut ram_phases: Vec<Phase> = Vec::new();
    let mut norm_phases: Vec<Phase> = Vec::new();
    let mut highbits_phases: Vec<Phase> = Vec::new();

    let mut norm_check_ops: Vec<NormCheckOp> = Vec::new();
    let mut highbits_ops: Vec<HighBitsOp> = Vec::new();

    let mut ntt_ram_bindings: Vec<(usize, usize)> = Vec::new();
    let mut w_side_bindings: Vec<(usize, u32)> = Vec::new();
    let mut ntt_boundary_bindings: Vec<(u32, u32, usize, bool)> = Vec::new();
    let mut call_tag: Vec<KeccakCallTag> = Vec::new();

    let mut bfly_offset: u32 = 0;
    let mut ntt_instance_id: u32 = 0;

    #[allow(unused_assignments)]
    let mut current_phase = Phase::Io;

    // Reserve IO slots for pk + sig + msg
    let io_bytes = pk.pk_bytes.len() + sig.level.sig_bytes() + msg.len();
    let mut next_slot = io_bytes.div_ceil(SLOT as usize) as u32;

    let mut alloc_slot = || {
        let s = next_slot;
        next_slot += 1;

        s * SLOT
    };

    // Helper:
    // write polynomial to RAM
    let write_poly = |poly: &[u32; N],
                      base_addr: u32,
                      clk: &mut u32,
                      events: &mut Vec<ram::MemoryEvent>,
                      phases: &mut Vec<Phase>,
                      phase: Phase| {
        for (j, &coeff) in poly.iter().enumerate() {
            events.push(ram::MemoryEvent::write(base_addr + j as u32, *clk, coeff));
            phases.push(phase);

            *clk += 1;
        }
    };

    // Helper:
    // read polynomial from RAM
    let read_poly = |poly: &[u32; N],
                     base_addr: u32,
                     clk: &mut u32,
                     events: &mut Vec<ram::MemoryEvent>,
                     phases: &mut Vec<Phase>,
                     phase: Phase| {
        for (j, &coeff) in poly.iter().enumerate() {
            events.push(ram::MemoryEvent::read(base_addr + j as u32, *clk, coeff));
            phases.push(phase);

            *clk += 1;
        }
    };

    // =========================================================
    // Step 1:
    // tr = H(pk, 64) via SHAKE-256
    // =========================================================

    current_phase = Phase::ExpandSample;

    let (tr_out, tr_calls) = keccak::shake256(&pk.pk_bytes, 64);
    keccak_phases.extend(repeat_n(current_phase, tr_calls.len()));
    keccak_calls.extend_from_slice(&tr_calls);

    for (i, _) in tr_calls.iter().enumerate() {
        keccak_sponge_meta.push((i == 0, false, false));
        call_tag.push(KeccakCallTag::HashPk);
    }

    let mut tr = [0u8; 64];
    tr.copy_from_slice(&tr_out[..64]);

    // =========================================================
    // Step 2:
    // μ = H(tr ∥ M', 64) via SHAKE-256
    // FIPS 204 §5.4:
    // M' = 0x00 ∥ len(ctx) ∥ ctx ∥ M
    // Empty context:
    // M' = 0x00 ∥ 0x00 ∥ M
    // =========================================================

    let mut mu_input = Vec::with_capacity(64 + 2 + msg.len());
    mu_input.extend_from_slice(&tr);
    mu_input.push(0x00); // context mode byte
    mu_input.push(0x00); // context length (empty)
    mu_input.extend_from_slice(msg);

    let (mu_out, mu_calls) = keccak::shake256(&mu_input, 64);
    keccak_phases.extend(repeat_n(current_phase, mu_calls.len()));
    keccak_calls.extend_from_slice(&mu_calls);

    for (i, _) in mu_calls.iter().enumerate() {
        keccak_sponge_meta.push((i == 0, false, false));
        call_tag.push(KeccakCallTag::HashMu);
    }

    let mut mu = [0u8; 64];
    mu.copy_from_slice(&mu_out[..64]);

    // =========================================================
    // Step 3:
    // c = SampleInBall(c̃, τ)
    // =========================================================

    let sib_kc_start = keccak_calls.len();
    let c_poly = sample_in_ball(
        &sig.c_tilde,
        level.tau,
        &mut keccak_calls,
        &mut keccak_sponge_meta,
    );

    let sib_kc_count = keccak_calls.len() - sib_kc_start;
    keccak_phases.extend(repeat_n(current_phase, sib_kc_count));

    for _ in 0..sib_kc_count {
        call_tag.push(KeccakCallTag::SampleInBall);
    }

    // =========================================================
    // Step 4:
    // A_hat = ExpandA(ρ, k, l)
    // =========================================================

    let ea_kc_start = keccak_calls.len();
    let a_hat = expand_a(
        &pk.rho,
        level.k,
        level.l,
        &mut keccak_calls,
        &mut keccak_sponge_meta,
    );

    let ea_kc_count = keccak_calls.len() - ea_kc_start;
    keccak_phases.extend(repeat_n(current_phase, ea_kc_count));

    // Tag each ExpandA call with (row, col)
    let mut ea_tag_idx = 0;
    for r in 0..level.k as u8 {
        for s in 0..level.l as u8 {
            // Each (r,s) pair may produce multiple
            // Keccak calls (multi-block SHAKE-128).
            // Count from sponge_meta.
            let mut found_first = false;
            while ea_tag_idx < ea_kc_count {
                call_tag.push(KeccakCallTag::ExpandA(r, s));
                ea_tag_idx += 1;

                if found_first {
                    // After first, keep going
                    // until next "is_first".
                    let meta_idx = ea_kc_start + ea_tag_idx;
                    if meta_idx < keccak_sponge_meta.len() && keccak_sponge_meta[meta_idx].0 {
                        break;
                    }
                } else {
                    found_first = true;
                }
            }
        }
    }

    // Fill any remaining untagged calls
    while call_tag.len() < keccak_calls.len() {
        call_tag.push(KeccakCallTag::Other);
    }

    // =========================================================
    // Step 5:
    // NTT forward passes
    //   ĉ = NTT(c)
    //   ẑ[i] = NTT(z[i]) for i = 0..l
    // =========================================================

    current_phase = Phase::NttForward;

    // NTT(c)
    let mut c_hat = c_poly;
    let c_input_addr = alloc_slot();
    let c_input_ram_base = ram_events.len();

    write_poly(
        &c_hat,
        c_input_addr,
        &mut ram_clk,
        &mut ram_events,
        &mut ram_phases,
        current_phase,
    );

    for pos in 0..N {
        ntt_boundary_bindings.push((ntt_instance_id, pos as u32, c_input_ram_base + pos, true));
    }

    let ntt_before = ntt_ops.len();
    let c_ntt_id = ntt_instance_id;

    ntt_forward_traced(&mut c_hat, &mut ntt_ops, ntt_instance_id, &mut bfly_offset);

    ntt_phases.extend(repeat_n(current_phase, ntt_ops.len() - ntt_before));
    ntt_instance_id += 1;

    let c_output_addr = alloc_slot();
    let c_output_ram_base = ram_events.len();

    write_poly(
        &c_hat,
        c_output_addr,
        &mut ram_clk,
        &mut ram_events,
        &mut ram_phases,
        current_phase,
    );

    for pos in 0..N {
        ntt_boundary_bindings.push((c_ntt_id, pos as u32, c_output_ram_base + pos, false));
    }

    // NTT(z[i])
    let mut z_hat = Vec::with_capacity(level.l);
    let mut z_hat_addrs = Vec::with_capacity(level.l);
    let mut z_input_addrs = Vec::with_capacity(level.l);

    for i in 0..level.l {
        let mut zi = sig.z[i];

        let input_addr = alloc_slot();
        let input_ram_base = ram_events.len();

        z_input_addrs.push(input_addr);

        write_poly(
            &zi,
            input_addr,
            &mut ram_clk,
            &mut ram_events,
            &mut ram_phases,
            current_phase,
        );

        for pos in 0..N {
            ntt_boundary_bindings.push((ntt_instance_id, pos as u32, input_ram_base + pos, true));
        }

        let ntt_before = ntt_ops.len();
        let this_ntt_id = ntt_instance_id;

        ntt_forward_traced(&mut zi, &mut ntt_ops, ntt_instance_id, &mut bfly_offset);

        ntt_phases.extend(repeat_n(current_phase, ntt_ops.len() - ntt_before));
        ntt_instance_id += 1;

        let output_addr = alloc_slot();
        let output_ram_base = ram_events.len();

        write_poly(
            &zi,
            output_addr,
            &mut ram_clk,
            &mut ram_events,
            &mut ram_phases,
            current_phase,
        );

        for pos in 0..N {
            ntt_boundary_bindings.push((this_ntt_id, pos as u32, output_ram_base + pos, false));
        }

        z_hat_addrs.push(output_addr);
        z_hat.push(zi);
    }

    // =========================================================
    // Step 6:
    // NTT(t1[i] · 2^d) and pointwise
    //   ŵ[i] = Σ_j A_hat[i][j] · ẑ[j] - ĉ · NTT(t1[i] · 2^d)
    // =========================================================

    current_phase = Phase::PointwiseMul;

    let two_d = mod_pow(2, level.d as u32);
    let mut w_hat = vec![[0u32; N]; level.k];

    for i in 0..level.k {
        // t1[i] · 2^d
        let mut t1_scaled = [0u32; N];
        for j in 0..N {
            t1_scaled[j] = mod_mul(pk.t1[i][j], two_d);
        }

        // NTT(t1[i] · 2^d)
        let t1_input_addr = alloc_slot();
        let t1_input_ram_base = ram_events.len();

        write_poly(
            &t1_scaled,
            t1_input_addr,
            &mut ram_clk,
            &mut ram_events,
            &mut ram_phases,
            current_phase,
        );

        for pos in 0..N {
            ntt_boundary_bindings.push((
                ntt_instance_id,
                pos as u32,
                t1_input_ram_base + pos,
                true,
            ));
        }

        let ntt_before = ntt_ops.len();
        let t1_ntt_id = ntt_instance_id;

        ntt_forward_traced(
            &mut t1_scaled,
            &mut ntt_ops,
            ntt_instance_id,
            &mut bfly_offset,
        );

        ntt_phases.extend(repeat_n(current_phase, ntt_ops.len() - ntt_before));
        ntt_instance_id += 1;

        let t1_output_addr = alloc_slot();
        let t1_output_ram_base = ram_events.len();

        write_poly(
            &t1_scaled,
            t1_output_addr,
            &mut ram_clk,
            &mut ram_events,
            &mut ram_phases,
            current_phase,
        );

        for pos in 0..N {
            ntt_boundary_bindings.push((t1_ntt_id, pos as u32, t1_output_ram_base + pos, false));
        }

        // Σ_j A_hat[i][j] · ẑ[j]
        let mut acc = [0u32; N];
        for j in 0..level.l {
            let a_ij = &a_hat[i * level.l + j];

            // Read ẑ[j] from RAM for binding
            let ram_base = ram_events.len();
            read_poly(
                &z_hat[j],
                z_hat_addrs[j],
                &mut ram_clk,
                &mut ram_events,
                &mut ram_phases,
                current_phase,
            );

            // Pointwise A_hat[i][j] · ẑ[j]
            let ntt_before = ntt_ops.len();
            let prod = pointwise_mul_traced(
                a_ij,
                &z_hat[j],
                &mut ntt_ops,
                &mut bfly_offset,
                ntt_instance_id,
            );

            ntt_phases.extend(repeat_n(current_phase, ntt_ops.len() - ntt_before));

            // Bind MulOnly ops to ẑ[j] RAM reads (b-side)
            for coeff_idx in 0..N {
                ntt_ram_bindings.push((ntt_before + coeff_idx, ram_base + coeff_idx));
            }

            // W-side:
            // A_hat values are the twiddles
            for coeff_idx in 0..N {
                w_side_bindings.push((
                    ram_base + coeff_idx,
                    bfly_offset - N as u32 + coeff_idx as u32,
                ));
            }

            acc = poly_add(&acc, &prod);
        }

        // - ĉ · NTT(t1[i] · 2^d)
        // Read ĉ from RAM
        let c_ram_base = ram_events.len();
        read_poly(
            &c_hat,
            c_output_addr,
            &mut ram_clk,
            &mut ram_events,
            &mut ram_phases,
            current_phase,
        );

        let ntt_before = ntt_ops.len();
        let ct1_prod = pointwise_mul_traced(
            &t1_scaled,
            &c_hat,
            &mut ntt_ops,
            &mut bfly_offset,
            ntt_instance_id,
        );

        ntt_phases.extend(repeat_n(current_phase, ntt_ops.len() - ntt_before));

        for coeff_idx in 0..N {
            ntt_ram_bindings.push((ntt_before + coeff_idx, c_ram_base + coeff_idx));
        }

        for coeff_idx in 0..N {
            w_side_bindings.push((
                c_ram_base + coeff_idx,
                bfly_offset - N as u32 + coeff_idx as u32,
            ));
        }

        w_hat[i] = poly_sub(&acc, &ct1_prod);
    }

    // =========================================================
    // Step 7:
    // w_approx[i] = INTT(ŵ[i])
    // =========================================================

    current_phase = Phase::NttInverse;

    let mut w_approx = vec![[0u32; N]; level.k];
    let mut w_approx_addrs = Vec::with_capacity(level.k);

    for i in 0..level.k {
        w_approx[i] = w_hat[i];

        let ntt_before = ntt_ops.len();
        ntt_inverse_traced(
            &mut w_approx[i],
            &mut ntt_ops,
            ntt_instance_id,
            &mut bfly_offset,
        );

        ntt_phases.extend(repeat_n(current_phase, ntt_ops.len() - ntt_before));

        ntt_instance_id += 1;

        let wa_addr = alloc_slot();
        w_approx_addrs.push(wa_addr);

        write_poly(
            &w_approx[i],
            wa_addr,
            &mut ram_clk,
            &mut ram_events,
            &mut ram_phases,
            current_phase,
        );
    }

    // =========================================================
    // Step 8:
    // w'_1[i] = UseHint(h[i], w_approx[i])
    // =========================================================

    current_phase = Phase::UseHint;

    let gamma2 = level.gamma2;

    let mut w1_prime = vec![[0u32; N]; level.k];
    let mut hb_idx = 0u32;

    for i in 0..level.k {
        for j in 0..N {
            let w1 = use_hint(sig.h[i][j], w_approx[i][j], gamma2);
            w1_prime[i][j] = w1;

            let wa_ram_addr = w_approx_addrs[i] + j as u32;

            ram_events.push(ram::MemoryEvent {
                addr: wa_ram_addr,
                clk: ram_clk,
                val: w_approx[i][j],
                is_write: false,
            });
            ram_phases.push(current_phase);
            ram_clk += 1;

            highbits_ops.push(HighBitsOp {
                r: w_approx[i][j],
                idx: hb_idx,
                ram_addr: wa_ram_addr,
                h_bit: sig.h[i][j],
                w1_prime: w1,
            });
            highbits_phases.push(current_phase);

            hb_idx += 1;
        }
    }

    // =========================================================
    // Step 9:
    // c̃' = H(μ ∥ w1Encode(w'_1), c̃_len)
    // =========================================================

    current_phase = Phase::HashCompare;

    let w1_encoded = w1_encode(&w1_prime, gamma2);

    let mut hash_input = Vec::with_capacity(64 + w1_encoded.len());
    hash_input.extend_from_slice(&mu);
    hash_input.extend_from_slice(&w1_encoded);

    let c_tilde_len = sig.c_tilde.len();
    let (hash_out, hash_calls) = keccak::shake256(&hash_input, c_tilde_len);

    keccak_phases.extend(repeat_n(current_phase, hash_calls.len()));
    keccak_calls.extend_from_slice(&hash_calls);

    for (i, _) in hash_calls.iter().enumerate() {
        keccak_sponge_meta.push((i == 0, false, false));
        call_tag.push(KeccakCallTag::HashCompare);
    }

    let c_tilde_prime = hash_out[..c_tilde_len].to_vec();

    // =========================================================
    // Step 10:
    // Norm check ‖z‖_∞ < γ₁ - β
    // =========================================================

    current_phase = Phase::NormCheck;

    let z_bound = level.z_bound();

    let mut norm_ok = true;
    let mut nc_idx = 0u32;

    for i in 0..level.l {
        for j in 0..N {
            let coeff = sig.z[i][j];

            // Map to centered representation
            let centered = if coeff > MLDSA_Q / 2 {
                MLDSA_Q - coeff
            } else {
                coeff
            };

            if centered >= z_bound {
                norm_ok = false;
            }

            let z_ram_addr = z_input_addrs[i] + j as u32;

            // RAM read event for NC-RAM binding
            ram_events.push(ram::MemoryEvent {
                addr: z_ram_addr,
                clk: ram_clk,
                val: coeff,
                is_write: false,
            });

            ram_phases.push(current_phase);

            ram_clk += 1;

            norm_check_ops.push(NormCheckOp {
                value: coeff,
                idx: nc_idx,
                ram_addr: z_ram_addr,
            });

            norm_phases.push(current_phase);

            nc_idx += 1;
        }
    }

    // =========================================================
    // Step 11:
    // Check hint weight ≤ ω
    // =========================================================

    let hint_weight: usize = sig.h.iter().map(|p| p.iter().filter(|&&b| b).count()).sum();
    let hint_ok = hint_weight <= level.omega;

    // =========================================================
    // Verdict
    // =========================================================

    let valid = sig.c_tilde == c_tilde_prime && norm_ok && hint_ok;

    // Bind standalone NTT ops to RAM.
    // Write NTT_B to a unique address,
    // preventing dispatch row swaps.
    {
        let ntt_bind_base = next_slot * SLOT;
        let bound_ntt: BTreeSet<usize> = ntt_ram_bindings.iter().map(|&(n, _)| n).collect();

        let mut ntt_bind_count = 0u32;
        for (i, op) in ntt_ops.iter().enumerate() {
            if matches!(op, ntt::NttOp::FlowCompanion(_)) {
                continue;
            }

            if bound_ntt.contains(&i) {
                continue;
            }

            let b_val = match op {
                ntt::NttOp::Butterfly(bfly) => bfly.b,
                ntt::NttOp::MulOnly(mul) => mul.b,
                ntt::NttOp::FlowCompanion(_) => unreachable!(),
            };

            let ram_idx = ram_events.len();
            ram_events.push(ram::MemoryEvent {
                addr: ntt_bind_base + ntt_bind_count,
                clk: 0,
                val: b_val,
                is_write: true,
            });

            ram_phases.push(ntt_phases[i]);
            ntt_ram_bindings.push((i, ram_idx));

            ntt_bind_count += 1;
        }

        next_slot += ntt_bind_count.div_ceil(SLOT);
    }

    let result = MlDsaVerifyResult {
        valid,
        keccak_calls,
        keccak_sponge_meta,
        ram_events,
        norm_check_ops,
        highbits_ops,
        c_tilde: sig.c_tilde.clone(),
        c_tilde_prime,
        ntt_phases,
        keccak_phases,
        ram_phases,
        norm_phases,
        highbits_phases,
        ntt_ram_bindings,
        w_side_bindings,
        ntt_boundary_bindings,
        next_free_addr: next_slot * SLOT,
        call_tag,
    };

    (result, ntt_ops)
}

// =================================================================
// NTT Tracing (8-layer, ζ = 512th root)
// =================================================================

/// Forward NTT with traced butterfly ops.
/// 8 layers (len = 128 → 1) for ML-DSA.
fn ntt_forward_traced(
    f: &mut [u32; N],
    ops: &mut Vec<ntt::NttOp>,
    instance_id: u32,
    bfly_offset: &mut u32,
) {
    let zetas = zeta_powers();
    let max_layer = 7u32;

    let mut k = 0usize;
    let mut len = 128;
    let mut layer = 0u32;

    while len >= 1 {
        let mut start = 0;
        let mut bfly_idx = 0u32;

        while start < N {
            k += 1;
            let zeta = zetas[k];

            for j in start..start + len {
                let a = f[j];
                let b = f[j + len];
                let t = mod_mul(zeta, b);

                f[j] = mod_add(a, t);
                f[j + len] = mod_sub(a, t);

                let pa = j as u32;
                let pb = (j + len) as u32;

                ops.push(ntt::NttOp::Butterfly(ntt::NttButterfly {
                    a,
                    b,
                    w: zeta,
                    layer,
                    butterfly_idx: *bfly_offset + bfly_idx,
                    is_forward: true,
                    ntt_instance: instance_id,
                    pos_a: pa,
                    pos_b: pb,
                }));

                let sl = if layer > 0 { layer - 1 } else { 0 };
                ops.push(ntt::NttOp::FlowCompanion(ntt::NttFlowCompanion {
                    b_in: b,
                    b_out: mod_sub(a, t),
                    layer,
                    ntt_instance: instance_id,
                    pos: pb,
                    src_layer: sl,
                    is_flow_output: layer < max_layer,
                    is_flow_input: layer > 0,
                    is_forward: true,
                }));

                bfly_idx += 1;
            }

            start += 2 * len;
        }

        *bfly_offset += bfly_idx;
        layer += 1;
        len >>= 1;
    }
}

/// Inverse NTT with GS decomposition.
///
/// Each GS butterfly decomposes into:
///   1. CT butterfly(a, b, w=1) -> (a+b, a-b)
///   2. MulOnly(diff, zeta) -> zeta*(a-b)
///      with FlowCompanion for pos_b input
///      at layers > 0.
fn ntt_inverse_traced(
    f: &mut [u32; N],
    ops: &mut Vec<ntt::NttOp>,
    instance_id: u32,
    bfly_offset: &mut u32,
) {
    let zetas = zeta_powers();

    let mut k = 256usize;
    let mut len = 1;
    let mut layer = 0u32;

    while len <= 128 {
        let mut start = 0;

        while start < N {
            k -= 1;

            let zeta = mod_neg(zetas[k]);

            for j in start..start + len {
                let a = f[j];
                let b = f[j + len];

                f[j] = mod_add(a, b);

                let diff = mod_sub(a, b);
                f[j + len] = mod_mul(zeta, diff);

                let pa = j as u32;
                let pb = (j + len) as u32;
                let sl = if layer > 0 { layer - 1 } else { 0 };

                // CT butterfly(w=1):
                // a_out = a+b, b_out = a-b
                ops.push(ntt::NttOp::Butterfly(ntt::NttButterfly {
                    a,
                    b,
                    w: 1,
                    layer,
                    butterfly_idx: *bfly_offset,
                    is_forward: false,
                    ntt_instance: instance_id,
                    pos_a: pa,
                    pos_b: pb,
                }));

                if layer > 0 {
                    ops.push(ntt::NttOp::FlowCompanion(ntt::NttFlowCompanion {
                        b_in: b,
                        b_out: 0,
                        layer,
                        ntt_instance: instance_id,
                        pos: pb,
                        src_layer: sl,
                        is_flow_output: false,
                        is_flow_input: true,
                        is_forward: false,
                    }));
                }

                *bfly_offset += 1;

                // MulOnly(diff, zeta) at pos_b
                ops.push(ntt::NttOp::MulOnly(ntt::NttMulOnly {
                    b: diff,
                    w: zeta,
                    layer,
                    butterfly_idx: *bfly_offset,
                    is_basemul: false,
                    flow_pos: Some(pb),
                    flow_instance: instance_id,
                    flow_src_layer: sl,
                }));

                *bfly_offset += 1;
            }

            start += 2 * len;
        }

        layer += 1;
        len <<= 1;
    }

    // Normalization:
    // 256^{-1} mod q
    let norm = mod_pow(256, MLDSA_Q - 2);

    for j in 0..N {
        let old = f[j];
        f[j] = mod_mul(old, norm);

        ops.push(ntt::NttOp::MulOnly(ntt::NttMulOnly {
            b: old,
            w: norm,
            layer,
            butterfly_idx: *bfly_offset,
            is_basemul: false,
            flow_pos: None,
            flow_instance: instance_id,
            flow_src_layer: layer,
        }));

        *bfly_offset += 1;
    }
}

/// Pointwise coefficient-by-coefficient
/// multiply with traced MulOnly ops.
fn pointwise_mul_traced(
    a: &[u32; N],
    b: &[u32; N],
    ops: &mut Vec<ntt::NttOp>,
    bfly_offset: &mut u32,
    instance_id: u32,
) -> [u32; N] {
    let mut r = [0u32; N];
    for i in 0..N {
        r[i] = mod_mul(a[i], b[i]);

        ops.push(ntt::NttOp::MulOnly(ntt::NttMulOnly {
            b: b[i],
            w: a[i],
            layer: crate::twiddle_rom::BASEMUL_LAYER_MARKER,
            butterfly_idx: *bfly_offset,
            is_basemul: true,
            flow_pos: None,
            flow_instance: instance_id,
            flow_src_layer: 0,
        }));

        *bfly_offset += 1;
    }

    r
}

// =================================================================
// Convenience wrapper (non-traced, for tests)
// =================================================================

#[cfg(test)]
pub fn ml_dsa_verify(pk: &MlDsaPublicKey, sig: &MlDsaSignature, msg: &[u8]) -> bool {
    ml_dsa_verify_traced(pk, sig, msg).0.valid
}

// =================================================================
// Tests
// =================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mldsa::MlDsaLevel;
    use ml_dsa::signature::{Keypair, Signer};
    use ml_dsa::{B32, MlDsa65, SigningKey};
    use rand::{TryRngCore, rngs::OsRng};

    fn nist_mldsa_65(msg: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let mut xi = [0u8; 32];
        OsRng.try_fill_bytes(&mut xi).unwrap();

        let key = SigningKey::<MlDsa65>::from_seed(&B32::from(xi));
        let pk = key.verifying_key().encode();
        let sig = key.sign(msg).encode();

        (pk.as_slice().to_vec(), sig.as_slice().to_vec())
    }

    #[test]
    fn verify_result_struct_populated() {
        // Minimal smoke test: construct a fake signature
        // that will fail verification but exercises the code path.
        let level = MlDsaLevel::MLDSA_65;

        let pk_bytes = vec![0u8; level.pk_bytes()];
        let pk = MlDsaPublicKey::from_bytes(level, &pk_bytes);

        let sig_bytes = vec![0u8; level.sig_bytes()];

        // Set hint section to valid (all zeros = no hints, ω+k bytes)
        // The last k bytes are cumulative indices, all 0.
        // This makes decode_hint succeed.

        let sig = MlDsaSignature::from_bytes(level, &sig_bytes);
        assert!(sig.is_some(), "zero signature should parse");

        let sig = sig.unwrap();
        let msg = b"test message";

        let (result, ntt_ops) = ml_dsa_verify_traced(&pk, &sig, msg);

        // Verification should fail (all-zero sig is invalid)
        assert!(!result.valid);

        // But the traced data should be populated
        assert!(!result.keccak_calls.is_empty());
        assert!(!ntt_ops.is_empty());
        assert!(!result.norm_check_ops.is_empty());
        assert!(!result.highbits_ops.is_empty());
    }

    #[test]
    fn nist_reference_mldsa65_verify() {
        let msg = b"Hekate ML-DSA test vector";
        let (nist_pk, nist_sig) = nist_mldsa_65(msg);

        let pk = MlDsaPublicKey::from_bytes(MlDsaLevel::MLDSA_65, &nist_pk);
        let sig = MlDsaSignature::from_bytes(MlDsaLevel::MLDSA_65, &nist_sig)
            .expect("NIST signature should parse");

        let valid = ml_dsa_verify(&pk, &sig, msg);
        assert!(valid, "NIST-generated signature should verify");
    }

    #[test]
    fn nist_reference_mldsa65_reject_tampered() {
        let msg = b"Hekate ML-DSA test vector";
        let (nist_pk, nist_sig) = nist_mldsa_65(msg);

        let pk = MlDsaPublicKey::from_bytes(MlDsaLevel::MLDSA_65, &nist_pk);

        // Tamper with message
        let bad_msg = b"Hekate ML-DSA test vector TAMPERED";

        let sig = MlDsaSignature::from_bytes(MlDsaLevel::MLDSA_65, &nist_sig)
            .expect("NIST signature should parse");

        let valid = ml_dsa_verify(&pk, &sig, bad_msg);
        assert!(!valid, "Tampered message should be rejected");
    }

    #[test]
    fn nist_reference_multiple_keypairs() {
        for i in 0..5 {
            let msg = format!("test message {i}");
            let (nist_pk, nist_sig) = nist_mldsa_65(msg.as_bytes());

            let pk = MlDsaPublicKey::from_bytes(MlDsaLevel::MLDSA_65, &nist_pk);
            let sig = MlDsaSignature::from_bytes(MlDsaLevel::MLDSA_65, &nist_sig)
                .expect("NIST signature should parse");

            assert!(
                ml_dsa_verify(&pk, &sig, msg.as_bytes()),
                "Keypair {i}: valid signature rejected",
            );

            // Tampered signature (flip first byte of c̃)
            let mut bad_sig_bytes = nist_sig.clone();
            bad_sig_bytes[0] ^= 0x01;

            if let Some(bad_sig) = MlDsaSignature::from_bytes(MlDsaLevel::MLDSA_65, &bad_sig_bytes)
            {
                assert!(
                    !ml_dsa_verify(&pk, &bad_sig, msg.as_bytes()),
                    "Keypair {i}: tampered signature accepted",
                );
            }
        }
    }

    #[test]
    fn traced_ops_counts_reasonable() {
        let msg = b"counting ops";
        let (nist_pk, nist_sig) = nist_mldsa_65(msg);

        let pk = MlDsaPublicKey::from_bytes(MlDsaLevel::MLDSA_65, &nist_pk);
        let sig = MlDsaSignature::from_bytes(MlDsaLevel::MLDSA_65, &nist_sig).unwrap();

        let (result, ntt_ops) = ml_dsa_verify_traced(&pk, &sig, msg);

        let level = MlDsaLevel::MLDSA_65;

        // NTT calls:
        // 1(c) + 5(z) + 6(t1) + 6(INTT w) = 18 NTTs
        // Each forward NTT:
        // 8 layers × 128 butterflies = 1024 bfly ops
        // Plus flow companions:
        // 1024 per NTT
        // Plus normalization muls for INTT:
        // 256 per INTT
        assert!(ntt_ops.len() > 10_000, "too few NTT ops: {}", ntt_ops.len());

        // Norm checks:
        // l × N = 5 × 256 = 1280
        assert_eq!(result.norm_check_ops.len(), level.l * N);

        // HighBits:
        // k × N = 6 × 256 = 1536
        assert_eq!(result.highbits_ops.len(), level.k * N);

        // Keccak calls should exist
        assert!(result.keccak_calls.len() > 10);

        // RAM events should exist
        assert!(result.ram_events.len() > 1000);
    }

    /// Cross-implementation check:
    /// our SHAKE-256 matches PQClean's
    /// stored tr inside the sk.
    #[test]
    #[allow(deprecated)]
    fn shake256_tr_matches_pqclean() {
        let mut xi = [0u8; 32];
        OsRng.try_fill_bytes(&mut xi).unwrap();

        let key = SigningKey::<MlDsa65>::from_seed(&B32::from(xi));

        let pk = key.verifying_key().encode();
        let (tr_out, _) = keccak::shake256(pk.as_slice(), 64);

        // FIPS-204 sk layout:
        // ρ(32) + K(32) + tr(64) + s1 + s2 + t0
        let sk_bytes = key.expanded_key().to_expanded();
        let nist_tr = &sk_bytes[64..128];

        assert_eq!(&tr_out[..64], nist_tr, "tr = H(pk) mismatch",);
    }
}
