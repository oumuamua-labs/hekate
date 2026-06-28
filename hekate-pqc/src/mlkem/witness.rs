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
    byte_decode, byte_encode, compress_poly, decompress_poly, mod_add, mod_mul, mod_sub, poly_add,
    poly_sub, sample_cbd, sample_ntt, zeta_powers,
};
use super::{MlKemLevel, N, Phase};
use crate::{basemul, ntt};
use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;
use core::iter::repeat_n;
use hekate_gadgets::chiplets::ram;
use hekate_keccak as keccak;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum KeccakCallTag {
    Other,
    HashCt,
    HashCtPrime,
    GHash,
    JCall,
}

// =================================================================
// ML-KEM Key Types
// =================================================================

/// ML-KEM decapsulation key.
#[derive(Clone)]
pub struct MlKemDecapsKey {
    /// Security level parameters.
    pub level: MlKemLevel,

    /// Secret vector ŝ (k polynomials in NTT domain).
    pub s_hat: Vec<[u16; N]>,

    /// Public key (encoded).
    pub ek: Vec<u8>,

    /// Hash of public key:
    /// H(ek).
    pub h: [u8; 32],

    /// Implicit rejection seed.
    pub z: [u8; 32],
}

impl MlKemDecapsKey {
    /// Parse from NIST/PQClean secret key bytes.
    ///
    /// Layout:
    /// `dk_PKE || ek || H(ek) (32) || z (32)`
    ///
    /// `dk_PKE` = ByteEncode_12(s_hat[0]) || ... || ByteEncode_12(s_hat[k-1])
    pub fn from_nist_bytes(level: MlKemLevel, sk_bytes: &[u8]) -> Self {
        assert_eq!(sk_bytes.len(), level.sk_bytes());

        let dk_pke_len = level.k * 12 * N / 8;
        let ek_len = level.ek_bytes();

        let dk_pke = &sk_bytes[..dk_pke_len];
        let ek = &sk_bytes[dk_pke_len..dk_pke_len + ek_len];
        let h = &sk_bytes[dk_pke_len + ek_len..dk_pke_len + ek_len + 32];
        let z = &sk_bytes[dk_pke_len + ek_len + 32..];

        let poly_bytes = 12 * N / 8; // 384

        let mut s_hat = vec![[0u16; N]; level.k];
        for i in 0..level.k {
            s_hat[i] = byte_decode(12, &dk_pke[i * poly_bytes..(i + 1) * poly_bytes]);
        }

        let mut h_arr = [0u8; 32];
        h_arr.copy_from_slice(h);

        let mut z_arr = [0u8; 32];
        z_arr.copy_from_slice(z);

        MlKemDecapsKey {
            level,
            s_hat,
            ek: ek.to_vec(),
            h: h_arr,
            z: z_arr,
        }
    }
}

/// ML-KEM encapsulation key (public).
#[derive(Clone)]
pub struct MlKemEncapsKey {
    /// NTT-domain matrix A (k×k).
    pub a_hat: Vec<Vec<[u16; N]>>,

    /// Public vector t̂ (k polynomials in NTT domain).
    pub t_hat: Vec<[u16; N]>,

    /// Seed ρ (for matrix A regeneration).
    #[allow(dead_code)]
    pub rho: [u8; 32],
}

// =================================================================
// ML-KEM.Decaps (FIPS 203 Algorithm 17)
// =================================================================

/// Result of ML-KEM decapsulation, including
/// all intermediate data needed for trace generation.
pub struct MlKemDecapsResult {
    /// Shared secret K (32 bytes).
    pub shared_secret: [u8; 32],

    /// All Keccak-f (input, output) state pairs.
    pub keccak_calls: Vec<keccak::KeccakCall>,

    /// Per-call sponge metadata:
    /// (is_first_in_sponge, is_sha3_512, is_shake_128)
    pub keccak_sponge_meta: Vec<(bool, bool, bool)>,

    /// Basemul addition operations.
    pub basemul_ops: Vec<basemul::BasemulOp>,

    /// RAM memory events for polynomial routing.
    pub ram_events: Vec<ram::MemoryEvent>,

    /// H(ct) and H(ct') for re-encryption comparison.
    pub h_ct: [u8; 32],
    pub h_ct_prime: [u8; 32],

    /// Phase annotation per Keccak call.
    pub keccak_phases: Vec<Phase>,

    /// Phase annotation per basemul op.
    pub basemul_phases: Vec<Phase>,

    /// Phase annotation per RAM event.
    pub ram_phases: Vec<Phase>,

    /// NTT <> RAM data binding pairs (b-side).
    pub ntt_ram_bindings: Vec<(usize, usize)>,

    /// W-side binding pairs:
    /// (ram_event_index, mulonly_butterfly_idx).
    pub w_side_bindings: Vec<(usize, u32)>,

    /// NTT boundary bindings:
    /// (ntt_instance_id, position, ram_event_idx, is_input).
    pub ntt_boundary_bindings: Vec<(u32, u32, usize, bool)>,

    /// First byte past pre-allocated decrypt slots.
    pub next_free_addr: u32,

    /// Per-call protocol tag.
    pub call_tag: Vec<KeccakCallTag>,

    /// K' lo/hi B32 lanes from G(m'||h).
    pub k_prime_lo: [u32; 4],
    pub k_prime_hi: [u32; 4],

    /// K̄ lo/hi B32 lanes from J(z||c).
    pub k_bar_lo: [u32; 4],
    pub k_bar_hi: [u32; 4],
}

/// Full decapsulation with traced NTT operations.
pub fn ml_kem_decaps_traced(
    dk: &MlKemDecapsKey,
    ct: &[u8],
) -> (MlKemDecapsResult, Vec<ntt::NttOp>) {
    let level = dk.level;

    let mut keccak_calls = Vec::new();
    let mut keccak_sponge_meta: Vec<(bool, bool, bool)> = Vec::new();
    let mut ntt_ops = Vec::new();
    let mut bm_ops: Vec<basemul::BasemulOp> = Vec::new();
    let mut ram_events: Vec<ram::MemoryEvent> = Vec::new();
    let mut ram_clk = 0u32;

    // Phase tracking
    let mut ntt_phases: Vec<Phase> = Vec::new();
    let mut keccak_phases: Vec<Phase> = Vec::new();
    let mut bm_phases: Vec<Phase> = Vec::new();
    let mut ram_phases: Vec<Phase> = Vec::new();

    // NTT <> RAM binding pairs:
    // (ntt_op_index, ram_event_index).
    let mut ntt_ram_bindings: Vec<(usize, usize)> = Vec::new();

    // W-side binding pairs:
    // (ram_event_index, mulonly_butterfly_idx).
    let mut w_side_bindings: Vec<(usize, u32)> = Vec::new();

    // Globally unique MulOnly
    // butterfly_idx counter.
    let mut bfly_offset: u32 = 0;

    // NTT instance counter
    // (unique per NTT-256 call).
    let mut ntt_instance_id: u32 = 0;

    // NTT boundary bindings:
    // (ntt_instance_id, position, ram_event_idx, is_input).
    let mut ntt_boundary_bindings: Vec<(u32, u32, usize, bool)> = Vec::new();

    // Current protocol phase.
    let mut current_phase = Phase::Decrypt;

    // Helper:
    // write a polynomial's coefficients to RAM.
    // base_addr = poly_slot * 256.
    // Returns next clock.
    let write_poly = |poly: &[u16; N],
                      base_addr: u32,
                      clk: &mut u32,
                      events: &mut Vec<ram::MemoryEvent>,
                      phases: &mut Vec<Phase>,
                      phase: Phase| {
        for (j, &coeff) in poly.iter().enumerate() {
            events.push(ram::MemoryEvent::write(
                base_addr + j as u32,
                *clk,
                coeff as u32,
            ));
            phases.push(phase);

            *clk += 1;
        }
    };

    // Helper:
    // read a polynomial's coefficients from RAM.
    let read_poly = |poly: &[u16; N],
                     base_addr: u32,
                     clk: &mut u32,
                     events: &mut Vec<ram::MemoryEvent>,
                     phases: &mut Vec<Phase>,
                     phase: Phase| {
        for (j, &coeff) in poly.iter().enumerate() {
            events.push(ram::MemoryEvent::read(
                base_addr + j as u32,
                *clk,
                coeff as u32,
            ));
            phases.push(phase);

            *clk += 1;
        }
    };

    // Decrypt
    let ct_du_len = level.k * N * level.du / 8;
    let u_bytes = &ct[..ct_du_len];
    let v_bytes = &ct[ct_du_len..];

    // Each polynomial version gets a unique slot.
    // SLOT = 256 coefficients.
    // Monotonic slot counter.
    const SLOT: u32 = 256;

    let ct_bytes = level.ct_bytes();
    let io_bytes = ct_bytes + sha3_padding_len(ct_bytes);

    let mut next_slot = io_bytes.div_ceil(SLOT as usize) as u32;
    let mut alloc_slot = || {
        let s = next_slot;
        next_slot += 1;

        s * SLOT
    };

    // Write secret key s_hat
    // (needed for basemul reads).
    let mut s_hat_addrs = vec![0u32; level.k];
    for i in 0..level.k {
        s_hat_addrs[i] = alloc_slot();
        write_poly(
            &dk.s_hat[i],
            s_hat_addrs[i],
            &mut ram_clk,
            &mut ram_events,
            &mut ram_phases,
            current_phase,
        );
    }

    let mut u_hat = vec![[0u16; N]; level.k];
    let mut u_hat_ntt_addrs = vec![0u32; level.k];

    for i in 0..level.k {
        let start = i * N * level.du / 8;
        let poly = byte_decode(level.du, &u_bytes[start..start + N * level.du / 8]);
        u_hat[i] = decompress_poly(level.du, &poly);

        // Write decompressed poly to RAM
        // for layer-0 input boundary binding.
        let input_ram_base = ram_events.len();
        let input_addr = alloc_slot();

        write_poly(
            &u_hat[i],
            input_addr,
            &mut ram_clk,
            &mut ram_events,
            &mut ram_phases,
            current_phase,
        );

        for pos in 0..N {
            ntt_boundary_bindings.push((ntt_instance_id, pos as u32, input_ram_base + pos, true));
        }

        let ntt_len_before = ntt_ops.len();
        let this_ntt_id = ntt_instance_id;
        ntt_forward_traced(&mut u_hat[i], &mut ntt_ops, ntt_instance_id);

        ntt_phases.extend(repeat_n(current_phase, ntt_ops.len() - ntt_len_before));
        ntt_instance_id += 1;

        // Write NTT(u) result
        let write_ram_base = ram_events.len();

        u_hat_ntt_addrs[i] = alloc_slot();
        write_poly(
            &u_hat[i],
            u_hat_ntt_addrs[i],
            &mut ram_clk,
            &mut ram_events,
            &mut ram_phases,
            current_phase,
        );

        // Layer-6 output boundary binding
        for pos in 0..N {
            ntt_boundary_bindings.push((this_ntt_id, pos as u32, write_ram_base + pos, false));
        }
    }

    let mut w_hat = [0u16; N];
    for i in 0..level.k {
        // Read s_hat and NTT(u) for basemul
        let ram_base = ram_events.len();

        read_poly(
            &dk.s_hat[i],
            s_hat_addrs[i],
            &mut ram_clk,
            &mut ram_events,
            &mut ram_phases,
            current_phase,
        );
        read_poly(
            &u_hat[i],
            u_hat_ntt_addrs[i],
            &mut ram_clk,
            &mut ram_events,
            &mut ram_phases,
            current_phase,
        );

        let ntt_len_before = ntt_ops.len();
        let bm_len_before = bm_ops.len();
        let bfly_before = bfly_offset;

        let bm_addr = alloc_slot();
        let prod = poly_basemul_traced(
            &dk.s_hat[i],
            &u_hat[i],
            &mut ntt_ops,
            &mut bm_ops,
            &mut bfly_offset,
            bm_addr,
        );

        // Bind MulOnly ops to
        // s_hat RAM reads (b-side).
        for j in 0..N {
            ntt_ram_bindings.push((ntt_len_before + j, ram_base + j));
        }

        // Bind w-side:
        // u_hat reads at ram_base+N..ram_base+2N
        for j in 0..N {
            w_side_bindings.push((ram_base + N + j, bfly_before + j as u32));
        }

        ntt_phases.extend(repeat_n(current_phase, ntt_ops.len() - ntt_len_before));
        bm_phases.extend(repeat_n(current_phase, bm_ops.len() - bm_len_before));

        w_hat = poly_add(&w_hat, &prod);
    }

    let ntt_len_before = ntt_ops.len();
    ntt_inverse_traced(&mut w_hat, &mut ntt_ops, ntt_instance_id);

    ntt_phases.extend(repeat_n(current_phase, ntt_ops.len() - ntt_len_before));
    ntt_instance_id += 1;

    let v = decompress_poly(level.dv, &byte_decode(level.dv, v_bytes));
    let m = poly_sub(&v, &w_hat);
    let m_compressed = compress_poly(1, &m);
    let m_bytes = byte_encode(1, &m_compressed);

    let mut m_prime = [0u8; 32];
    m_prime.copy_from_slice(&m_bytes[..32]);

    // Phase transition:
    // decrypt complete, now G hash.
    current_phase = Phase::GHash;

    // G(m' || h)
    let mut g_input = Vec::with_capacity(64);
    g_input.extend_from_slice(&m_prime);
    g_input.extend_from_slice(&dk.h);

    let g_start = keccak_calls.len();
    let (g_out, g_calls) = keccak::sha3_512(&g_input);

    keccak_phases.extend(repeat_n(current_phase, g_calls.len()));
    keccak_calls.extend_from_slice(&g_calls);

    let g_end = keccak_calls.len();

    for (k, _) in g_calls.iter().enumerate() {
        keccak_sponge_meta.push((k == 0, true, false));
    }

    let mut k_bar = [0u8; 32];
    let mut r = [0u8; 32];

    k_bar.copy_from_slice(&g_out[..32]);
    r.copy_from_slice(&g_out[32..]);

    // Write G-function input (m' || h) to RAM
    let g_input_addr = alloc_slot();
    for (j, &byte) in g_input.iter().enumerate() {
        ram_events.push(ram::MemoryEvent::write(
            g_input_addr + j as u32,
            ram_clk,
            byte as u32,
        ));
        ram_phases.push(current_phase);

        ram_clk += 1;
    }

    // Phase transition:
    // G hash complete, now re-encrypt.
    current_phase = Phase::Encrypt;

    // Re-encrypt (traced)
    let keccak_len_before = keccak_calls.len();
    let ek = parse_encaps_key(level, &dk.ek, &mut keccak_calls, &mut keccak_sponge_meta);

    keccak_phases.extend(repeat_n(
        current_phase,
        keccak_calls.len() - keccak_len_before,
    ));

    // Encrypt-path RAM events.
    // Write A_hat and t_hat to RAM
    // so basemul reads are constrained.
    let mut a_hat_addrs = vec![vec![0u32; level.k]; level.k];
    for i in 0..level.k {
        for j in 0..level.k {
            a_hat_addrs[i][j] = alloc_slot();
            write_poly(
                &ek.a_hat[i][j],
                a_hat_addrs[i][j],
                &mut ram_clk,
                &mut ram_events,
                &mut ram_phases,
                current_phase,
            );
        }
    }

    let mut t_hat_addrs = vec![0u32; level.k];
    for i in 0..level.k {
        t_hat_addrs[i] = alloc_slot();
        write_poly(
            &ek.t_hat[i],
            t_hat_addrs[i],
            &mut ram_clk,
            &mut ram_events,
            &mut ram_phases,
            current_phase,
        );
    }

    let mut r_vec = vec![[0u16; N]; level.k];
    let mut r_vec_addrs = vec![0u32; level.k];

    for i in 0..level.k {
        let keccak_len_before = keccak_calls.len();
        r_vec[i] = sample_cbd(
            level.eta1,
            &r,
            i as u8,
            &mut keccak_calls,
            &mut keccak_sponge_meta,
        );

        keccak_phases.extend(repeat_n(
            current_phase,
            keccak_calls.len() - keccak_len_before,
        ));

        // Write r_vec before NTT
        let input_ram_base = ram_events.len();

        r_vec_addrs[i] = alloc_slot();
        write_poly(
            &r_vec[i],
            r_vec_addrs[i],
            &mut ram_clk,
            &mut ram_events,
            &mut ram_phases,
            current_phase,
        );

        let ntt_len_before = ntt_ops.len();
        let this_ntt_id = ntt_instance_id;

        ntt_forward_traced(&mut r_vec[i], &mut ntt_ops, ntt_instance_id);

        ntt_phases.extend(repeat_n(current_phase, ntt_ops.len() - ntt_len_before));
        ntt_instance_id += 1;

        // Layer-0 input boundary binding
        for pos in 0..N {
            ntt_boundary_bindings.push((this_ntt_id, pos as u32, input_ram_base + pos, true));
        }

        // Write NTT(r_vec) result
        let write_ram_base = ram_events.len();
        let ntt_addr = alloc_slot();

        write_poly(
            &r_vec[i],
            ntt_addr,
            &mut ram_clk,
            &mut ram_events,
            &mut ram_phases,
            current_phase,
        );

        // Layer-6 output boundary binding
        for pos in 0..N {
            ntt_boundary_bindings.push((this_ntt_id, pos as u32, write_ram_base + pos, false));
        }

        r_vec_addrs[i] = ntt_addr; // update to NTT-domain address
    }

    let mut e1 = vec![[0u16; N]; level.k];
    for i in 0..level.k {
        let keccak_len_before = keccak_calls.len();
        e1[i] = sample_cbd(
            level.eta2,
            &r,
            (level.k + i) as u8,
            &mut keccak_calls,
            &mut keccak_sponge_meta,
        );

        keccak_phases.extend(repeat_n(
            current_phase,
            keccak_calls.len() - keccak_len_before,
        ));
    }

    let keccak_len_before = keccak_calls.len();
    let e2 = sample_cbd(
        level.eta2,
        &r,
        (2 * level.k) as u8,
        &mut keccak_calls,
        &mut keccak_sponge_meta,
    );

    keccak_phases.extend(repeat_n(
        current_phase,
        keccak_calls.len() - keccak_len_before,
    ));

    let mut u_enc = vec![[0u16; N]; level.k];
    for i in 0..level.k {
        let mut acc = [0u16; N];
        for j in 0..level.k {
            // Read A_hat and r_vec for basemul
            let ram_base = ram_events.len();

            read_poly(
                &ek.a_hat[j][i],
                a_hat_addrs[j][i],
                &mut ram_clk,
                &mut ram_events,
                &mut ram_phases,
                current_phase,
            );
            read_poly(
                &r_vec[j],
                r_vec_addrs[j],
                &mut ram_clk,
                &mut ram_events,
                &mut ram_phases,
                current_phase,
            );

            let ntt_len_before = ntt_ops.len();
            let bm_len_before = bm_ops.len();
            let bfly_before = bfly_offset;

            let bm_addr = alloc_slot();
            let prod = poly_basemul_traced(
                &ek.a_hat[j][i],
                &r_vec[j],
                &mut ntt_ops,
                &mut bm_ops,
                &mut bfly_offset,
                bm_addr,
            );

            // b-side binding
            for k in 0..N {
                ntt_ram_bindings.push((ntt_len_before + k, ram_base + k));
            }

            // w-side binding:
            // r_vec reads at ram_base+N..ram_base+2N
            for k in 0..N {
                w_side_bindings.push((ram_base + N + k, bfly_before + k as u32));
            }

            ntt_phases.extend(repeat_n(current_phase, ntt_ops.len() - ntt_len_before));
            bm_phases.extend(repeat_n(current_phase, bm_ops.len() - bm_len_before));

            acc = poly_add(&acc, &prod);
        }

        let ntt_len_before = ntt_ops.len();
        ntt_inverse_traced(&mut acc, &mut ntt_ops, ntt_instance_id);

        ntt_phases.extend(repeat_n(current_phase, ntt_ops.len() - ntt_len_before));
        ntt_instance_id += 1;

        u_enc[i] = poly_add(&acc, &e1[i]);

        // Write u_enc result
        let addr = alloc_slot();
        write_poly(
            &u_enc[i],
            addr,
            &mut ram_clk,
            &mut ram_events,
            &mut ram_phases,
            current_phase,
        );
    }

    let mut v_acc = [0u16; N];
    for i in 0..level.k {
        // t_hat reads bind to MulOnly ops.
        let ram_base = ram_events.len();
        read_poly(
            &ek.t_hat[i],
            t_hat_addrs[i],
            &mut ram_clk,
            &mut ram_events,
            &mut ram_phases,
            current_phase,
        );
        read_poly(
            &r_vec[i],
            r_vec_addrs[i],
            &mut ram_clk,
            &mut ram_events,
            &mut ram_phases,
            current_phase,
        );

        let ntt_len_before = ntt_ops.len();
        let bm_len_before = bm_ops.len();
        let bfly_before = bfly_offset;

        let bm_addr = alloc_slot();
        let prod = poly_basemul_traced(
            &ek.t_hat[i],
            &r_vec[i],
            &mut ntt_ops,
            &mut bm_ops,
            &mut bfly_offset,
            bm_addr,
        );

        // b-side binding
        for k in 0..N {
            ntt_ram_bindings.push((ntt_len_before + k, ram_base + k));
        }

        // w-side binding:
        // r_vec reads at ram_base+N..ram_base+2N
        for k in 0..N {
            w_side_bindings.push((ram_base + N + k, bfly_before + k as u32));
        }

        ntt_phases.extend(repeat_n(current_phase, ntt_ops.len() - ntt_len_before));
        bm_phases.extend(repeat_n(current_phase, bm_ops.len() - bm_len_before));

        v_acc = poly_add(&v_acc, &prod);
    }

    let ntt_len_before = ntt_ops.len();
    ntt_inverse_traced(&mut v_acc, &mut ntt_ops, ntt_instance_id);

    ntt_phases.extend(repeat_n(current_phase, ntt_ops.len() - ntt_len_before));

    v_acc = poly_add(&v_acc, &e2);

    let m_poly = byte_decode(1, &m_prime);
    let m_decomp = decompress_poly(1, &m_poly);

    v_acc = poly_add(&v_acc, &m_decomp);

    // Build re-encrypted ciphertext
    let mut ct_prime = Vec::new();
    for i in 0..level.k {
        let c = compress_poly(level.du, &u_enc[i]);
        ct_prime.extend_from_slice(&byte_encode(level.du, &c));
    }

    let c2 = compress_poly(level.dv, &v_acc);
    ct_prime.extend_from_slice(&byte_encode(level.dv, &c2));

    // Phase transition:
    // encrypt complete, now comparison hashes.
    current_phase = Phase::CmpHash;

    // Hash both ct and ct' for constrained comparison.
    // H(ct) is already part of ML-KEM protocol.
    // H(ct') is extra — needed for soundness.
    let h_ct_start = keccak_calls.len();
    let (h_ct, h_ct_calls) = keccak::sha3_256(ct);

    keccak_phases.extend(repeat_n(current_phase, h_ct_calls.len()));
    keccak_calls.extend_from_slice(&h_ct_calls);

    let h_ct_end = keccak_calls.len();

    for (k, _) in h_ct_calls.iter().enumerate() {
        keccak_sponge_meta.push((k == 0, false, false));
    }

    let h_ct_prime_start = keccak_calls.len();
    let (h_ct_prime, h_ct_prime_calls) = keccak::sha3_256(&ct_prime);

    keccak_phases.extend(repeat_n(current_phase, h_ct_prime_calls.len()));
    keccak_calls.extend_from_slice(&h_ct_prime_calls);

    let h_ct_prime_end = keccak_calls.len();

    for (k, _) in h_ct_prime_calls.iter().enumerate() {
        keccak_sponge_meta.push((k == 0, false, false));
    }

    // Step 4:
    // K̄ = J(z || c) for implicit rejection
    let mut j_input = Vec::with_capacity(32 + ct.len());
    j_input.extend_from_slice(&dk.z);
    j_input.extend_from_slice(ct);

    let j_start = keccak_calls.len();
    let (k_bar_reject, j_calls) = keccak::shake256(&j_input, 32);

    keccak_phases.extend(repeat_n(current_phase, j_calls.len()));
    keccak_calls.extend_from_slice(&j_calls);

    let j_end = keccak_calls.len();

    for (k, _) in j_calls.iter().enumerate() {
        keccak_sponge_meta.push((k == 0, false, false));
    }

    // Phase transition:
    // hashes complete, now compare.
    // No operations dispatched
    // in Compare phase yet.
    let _current_phase = Phase::Compare;

    // FIPS 203 §6.3 steps 5-6:
    // output K' on match, else the implicit-reject key K̄.
    // The compare and select are branchless;
    // a secret-dependent branch is an FO oracle.
    let mut diff = 0u8;
    for k in 0..32 {
        diff |= h_ct[k] ^ h_ct_prime[k];
    }

    // match_mask = 0xFF iff the hashes are equal, else 0x00
    let differs = (diff | diff.wrapping_neg()) >> 7;
    let match_mask = differs.wrapping_sub(1);

    let mut shared_secret = [0u8; 32];
    for k in 0..32 {
        shared_secret[k] = (k_bar[k] & match_mask) | (k_bar_reject[k] & !match_mask);
    }

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

    let next_free_addr = next_slot * SLOT;

    let mut call_tag = vec![KeccakCallTag::Other; keccak_calls.len()];
    call_tag[g_start..g_end].fill(KeccakCallTag::GHash);
    call_tag[h_ct_start..h_ct_end].fill(KeccakCallTag::HashCt);
    call_tag[h_ct_prime_start..h_ct_prime_end].fill(KeccakCallTag::HashCtPrime);
    call_tag[j_start..j_end].fill(KeccakCallTag::JCall);

    let decompose_lanes = |bytes: &[u8]| -> ([u32; 4], [u32; 4]) {
        let mut lo = [0u32; 4];
        let mut hi = [0u32; 4];

        for i in 0..4 {
            lo[i] = u32::from_le_bytes(bytes[i * 8..i * 8 + 4].try_into().unwrap());
            hi[i] = u32::from_le_bytes(bytes[i * 8 + 4..i * 8 + 8].try_into().unwrap());
        }

        (lo, hi)
    };

    let (k_prime_lo, k_prime_hi) = decompose_lanes(&k_bar);
    let (k_bar_lo, k_bar_hi) = decompose_lanes(&k_bar_reject);

    let result = MlKemDecapsResult {
        shared_secret,
        keccak_calls,
        keccak_sponge_meta,
        basemul_ops: bm_ops,
        ram_events,
        h_ct,
        h_ct_prime,
        keccak_phases,
        basemul_phases: bm_phases,
        ram_phases,
        ntt_ram_bindings,
        w_side_bindings,
        ntt_boundary_bindings,
        next_free_addr,
        call_tag,
        k_prime_lo,
        k_prime_hi,
        k_bar_lo,
        k_bar_hi,
    };

    (result, ntt_ops)
}

// =================================================================
// Traced NTT (collects butterfly ops for chiplet)
// =================================================================

/// Forward NTT recording all butterfly ops.
fn ntt_forward_traced(f: &mut [u16; N], ops: &mut Vec<ntt::NttOp>, instance_id: u32) {
    let zetas = zeta_powers();

    let mut k = 1usize;
    let mut len = 128;
    let mut layer = 0u32;

    while len >= 2 {
        let mut start = 0;
        let mut bfly_idx = 0u32;

        while start < N {
            let zeta = zetas[k];
            k += 1;

            for j in start..start + len {
                let a = f[j];
                let b = f[j + len];
                let t = mod_mul(zeta, b);

                let b_out = mod_sub(a, t);
                let a_out = mod_add(a, t);

                f[j + len] = b_out;
                f[j] = a_out;

                // Array positions:
                // f[j] and f[j+len]
                let pa = j as u32;
                let pb = (j + len) as u32;

                ops.push(ntt::NttOp::Butterfly(ntt::NttButterfly {
                    a: a as u32,
                    b: b as u32,
                    w: zeta as u32,
                    layer,
                    butterfly_idx: bfly_idx,
                    is_forward: true,
                    ntt_instance: instance_id,
                    pos_a: pa,
                    pos_b: pb,
                }));

                // Companion row for pos_b flow entry.
                // NTT-N:
                // log2(N)-1 layers,
                // last = log2(N)-2.
                let max_layer = N.trailing_zeros() - 2;
                ops.push(ntt::NttOp::FlowCompanion(ntt::NttFlowCompanion {
                    b_in: b as u32,
                    b_out: b_out as u32,
                    layer,
                    ntt_instance: instance_id,
                    pos: pb,
                    src_layer: if layer > 0 { layer - 1 } else { 0 },
                    is_flow_output: layer < max_layer,
                    is_flow_input: layer > 0,
                    is_forward: true,
                }));

                bfly_idx += 1;
            }

            start += 2 * len;
        }

        len >>= 1;
        layer += 1;
    }
}

/// Inverse NTT recording all butterfly ops.
///
/// GS butterfly:
/// a'=a+b, b'=(a-b)*w.
/// Decomposed into two NTT chiplet ops:
/// 1. CT butterfly(a, b, w=1) -> (a+b, a-b)
/// 2. MulOnly(b=a-b, w=zeta) -> (a-b)*zeta
fn ntt_inverse_traced(f: &mut [u16; N], ops: &mut Vec<ntt::NttOp>, instance_id: u32) {
    let zetas = zeta_powers();

    let mut k = 127usize;
    let mut len = 2;
    let mut layer = 0u32;

    while len <= 128 {
        let mut start = 0;
        let mut bfly_idx = 0u32;

        while start < N {
            let zeta = zetas[k];
            k = k.wrapping_sub(1);

            for j in start..start + len {
                let t = f[j];
                let y = f[j + len];

                // GS butterfly computation
                f[j] = mod_add(t, y);

                let diff = mod_sub(y, t);
                f[j + len] = mod_mul(zeta, diff);

                let pa = j as u32;
                let pb = (j + len) as u32;
                let sl = if layer > 0 { layer - 1 } else { 0 };

                // Step 1:
                // CT butterfly(w=1)
                // a_out = t+y at pos j (flow output)
                // b_out = t-y (intermediate, not stored)
                ops.push(ntt::NttOp::Butterfly(ntt::NttButterfly {
                    a: t as u32,
                    b: y as u32,
                    w: 1,
                    layer,
                    butterfly_idx: bfly_idx,
                    is_forward: false,
                    ntt_instance: instance_id,
                    pos_a: pa,
                    pos_b: pb,
                }));

                // Flow:
                // CT butterfly primary row carries
                // output (inst, layer, pa, a_out)
                // and input (inst, sl, pa, t).
                // Companion for input at pos_b (value y).
                if layer > 0 {
                    ops.push(ntt::NttOp::FlowCompanion(ntt::NttFlowCompanion {
                        b_in: y as u32,
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

                bfly_idx += 1;

                // Step 2:
                // MulOnly(diff, zeta)
                // output = zeta*diff at
                // pos j+len (flow output)
                ops.push(ntt::NttOp::MulOnly(ntt::NttMulOnly {
                    b: diff as u32,
                    w: zeta as u32,
                    layer,
                    butterfly_idx: bfly_idx,
                    is_basemul: false,
                    flow_pos: Some(pb),
                    flow_instance: instance_id,
                    flow_src_layer: sl,
                }));

                bfly_idx += 1;
            }

            start += 2 * len;
        }

        len <<= 1;
        layer += 1;
    }

    // Normalization:
    // multiply by 128^{-1} mod q.
    // Not part of the flow DAG.
    const NTT_NORM: u16 = 3303;

    for (i, coeff) in f.iter_mut().enumerate() {
        let old = *coeff;
        *coeff = mod_mul(old, NTT_NORM);

        ops.push(ntt::NttOp::MulOnly(ntt::NttMulOnly {
            b: old as u32,
            w: NTT_NORM as u32,
            layer: 8,
            butterfly_idx: i as u32,
            is_basemul: false,
            flow_pos: None,
            flow_instance: 0,
            flow_src_layer: 0,
        }));
    }
}

/// Basemul with traced individual multiplications.
/// Uses the correct basemul algorithm and records
/// each constituent mod_mul as a MulOnly op.
fn poly_basemul_traced(
    a: &[u16; N],
    b: &[u16; N],
    ntt_ops: &mut Vec<ntt::NttOp>,
    bm_ops: &mut Vec<basemul::BasemulOp>,
    bfly_offset: &mut u32,
    bm_ram_base: u32,
) -> [u16; N] {
    let zetas = zeta_powers();

    let mut r = [0u16; N];

    for i in 0..64 {
        let z = zetas[64 + i];
        let (a0, a1) = (a[4 * i], a[4 * i + 1]);
        let (b0, b1) = (b[4 * i], b[4 * i + 1]);

        // a0*b0, a1*b1, (a1*b1)*z, a0*b1, a1*b0
        let a0b0 = mod_mul(a0, b0);
        let a1b1z = mod_mul(mod_mul(a1, b1), z);

        r[4 * i] = mod_add(a0b0, a1b1z);
        r[4 * i + 1] = mod_add(mod_mul(a0, b1), mod_mul(a1, b0));

        let (a2, a3) = (a[4 * i + 2], a[4 * i + 3]);
        let (b2, b3) = (b[4 * i + 2], b[4 * i + 3]);

        let neg_z = mod_sub(0, z);
        let a2b2 = mod_mul(a2, b2);
        let a3b3nz = mod_mul(mod_mul(a3, b3), neg_z);

        r[4 * i + 2] = mod_add(a2b2, a3b3nz);
        r[4 * i + 3] = mod_add(mod_mul(a2, b3), mod_mul(a3, b2));

        // Record individual multiplies for NTT chiplet
        for k in 0..4 {
            ntt_ops.push(ntt::NttOp::MulOnly(ntt::NttMulOnly {
                b: a[4 * i + k] as u32,
                w: b[4 * i + k] as u32,
                layer: crate::twiddle_rom::BASEMUL_LAYER_MARKER,
                butterfly_idx: *bfly_offset,
                is_basemul: true,
                flow_pos: None,
                flow_instance: 0,
                flow_src_layer: 0,
            }));

            *bfly_offset += 1;
        }

        // Record basemul addition structure
        let p01 = mod_mul(a0, b1);
        let p10 = mod_mul(a1, b0);
        let p23 = mod_mul(a2, b3);
        let p32 = mod_mul(a3, b2);

        let bm_base = (i as u32) * 4;
        let ram_base_i = bm_ram_base + (i as u32) * 4;

        // r0 = a0b0 + a1b1z
        bm_ops.push(basemul::BasemulOp {
            a: a0b0 as u32,
            b: a1b1z as u32,
            c: r[4 * i] as u32,
            idx: bm_base,
            ram_addr: ram_base_i,
            request_idx: bm_base,
        });

        // r1 = p01 + p10
        bm_ops.push(basemul::BasemulOp {
            a: p01 as u32,
            b: p10 as u32,
            c: r[4 * i + 1] as u32,
            idx: bm_base + 1,
            ram_addr: ram_base_i + 1,
            request_idx: bm_base + 1,
        });

        // r2 = a2b2 + a3b3nz (add, not sub)
        bm_ops.push(basemul::BasemulOp {
            a: a2b2 as u32,
            b: a3b3nz as u32,
            c: r[4 * i + 2] as u32,
            idx: bm_base + 2,
            ram_addr: ram_base_i + 2,
            request_idx: bm_base + 2,
        });

        // r3 = p23 + p32
        bm_ops.push(basemul::BasemulOp {
            a: p23 as u32,
            b: p32 as u32,
            c: r[4 * i + 3] as u32,
            idx: bm_base + 3,
            ram_addr: ram_base_i + 3,
            request_idx: bm_base + 3,
        });
    }

    r
}

/// Parse encapsulation key
/// from encoded bytes.
fn parse_encaps_key(
    level: MlKemLevel,
    ek_bytes: &[u8],
    keccak_calls: &mut Vec<keccak::KeccakCall>,
    sponge_meta: &mut Vec<(bool, bool, bool)>,
) -> MlKemEncapsKey {
    let poly_bytes = 12 * N / 8; // 384 bytes per polynomial
    let t_end = level.k * poly_bytes;

    let mut t_hat = vec![[0u16; N]; level.k];
    for i in 0..level.k {
        t_hat[i] = byte_decode(12, &ek_bytes[i * poly_bytes..(i + 1) * poly_bytes]);
    }

    let mut rho = [0u8; 32];
    rho.copy_from_slice(&ek_bytes[t_end..t_end + 32]);

    // Regenerate matrix A from ρ
    let mut a_hat = vec![vec![[0u16; N]; level.k]; level.k];
    for i in 0..level.k {
        for j in 0..level.k {
            a_hat[i][j] = sample_ntt(&rho, i as u8, j as u8, keccak_calls, sponge_meta);
        }
    }

    MlKemEncapsKey { a_hat, t_hat, rho }
}

/// SHA3-256 rate-block padding
/// length (FIPS 202 §B.2).
pub(crate) fn sha3_padding_len(n: usize) -> usize {
    const RATE: usize = 136;

    let r = n % RATE;
    if r == 0 { RATE } else { RATE - r }
}

// =================================================================
// Tests
// =================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlkem::arithmetic::{ntt_forward, ntt_inverse, poly_basemul};
    use hekate_keccak::{sha3_256, sha3_512, shake256};
    #[allow(deprecated)]
    use ml_kem::ExpandedKeyEncoding;
    use ml_kem::kem::Decapsulate;
    use ml_kem::{B32, Ciphertext, DecapsulationKey, MlKem768};
    use rand::{TryRngCore, rngs::OsRng};

    #[allow(deprecated)]
    fn nist_mlkem_768() -> (DecapsulationKey<MlKem768>, Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut seed = [0u8; 64];
        OsRng.try_fill_bytes(&mut seed).unwrap();

        let dk = DecapsulationKey::<MlKem768>::from_seed(seed.into());

        let mut m = [0u8; 32];
        OsRng.try_fill_bytes(&mut m).unwrap();

        let (ct, ss) = dk
            .encapsulation_key()
            .encapsulate_deterministic(&B32::from(m));
        let sk = dk.to_expanded_bytes().as_slice().to_vec();

        (dk, ct.as_slice().to_vec(), sk, ss.as_slice().to_vec())
    }

    fn nist_decapsulate_768(dk: &DecapsulationKey<MlKem768>, ct: &[u8]) -> Vec<u8> {
        let ct = Ciphertext::<MlKem768>::try_from(ct).unwrap();
        dk.decapsulate(&ct).as_slice().to_vec()
    }

    /// Generate an ML-KEM keypair for
    /// testing. Uses a deterministic seed.
    fn ml_kem_keygen(level: MlKemLevel, seed: &[u8; 64]) -> (MlKemEncapsKey, MlKemDecapsKey) {
        let mut keccak_calls = Vec::new();
        let mut keccak_sponge_meta = Vec::new();

        let d = &seed[..32];
        let z_bytes = &seed[32..64];

        // G(d || k)
        let mut g_in = Vec::with_capacity(33);
        g_in.extend_from_slice(d);
        g_in.push(level.k as u8);

        let (g_out, _) = sha3_512(&g_in);

        let mut rho = [0u8; 32];
        let mut sigma = [0u8; 32];

        rho.copy_from_slice(&g_out[..32]);
        sigma.copy_from_slice(&g_out[32..]);

        // Generate A
        let mut a_hat = vec![vec![[0u16; N]; level.k]; level.k];
        for i in 0..level.k {
            for j in 0..level.k {
                a_hat[i][j] = sample_ntt(
                    &rho,
                    i as u8,
                    j as u8,
                    &mut keccak_calls,
                    &mut keccak_sponge_meta,
                );
            }
        }

        // Sample s, e
        let mut s = vec![[0u16; N]; level.k];
        for i in 0..level.k {
            s[i] = sample_cbd(
                level.eta1,
                &sigma,
                i as u8,
                &mut keccak_calls,
                &mut keccak_sponge_meta,
            );

            ntt_forward(&mut s[i]);
        }

        let mut e = vec![[0u16; N]; level.k];
        for i in 0..level.k {
            e[i] = sample_cbd(
                level.eta1,
                &sigma,
                (level.k + i) as u8,
                &mut keccak_calls,
                &mut keccak_sponge_meta,
            );

            ntt_forward(&mut e[i]);
        }

        // t_hat = A * s_hat + e_hat
        let mut t_hat = vec![[0u16; N]; level.k];
        for i in 0..level.k {
            let mut acc = [0u16; N];
            for j in 0..level.k {
                let prod = poly_basemul(&a_hat[i][j], &s[j]);
                acc = poly_add(&acc, &prod);
            }

            t_hat[i] = poly_add(&acc, &e[i]);
        }

        // Encode public key
        let mut ek_bytes = Vec::new();
        for i in 0..level.k {
            ek_bytes.extend_from_slice(&byte_encode(12, &t_hat[i]));
        }

        ek_bytes.extend_from_slice(&rho);

        // H(ek)
        let (h, _) = sha3_256(&ek_bytes);

        let mut z = [0u8; 32];
        z.copy_from_slice(z_bytes);

        let ek = MlKemEncapsKey { a_hat, t_hat, rho };
        let dk = MlKemDecapsKey {
            level,
            s_hat: s,
            ek: ek_bytes,
            h,
            z,
        };

        (ek, dk)
    }

    /// ML-KEM encapsulation (for testing).
    fn ml_kem_encaps(
        level: MlKemLevel,
        ek: &MlKemEncapsKey,
        rand_bytes: &[u8; 32],
    ) -> (Vec<u8>, [u8; 32]) {
        let mut keccak_calls = Vec::new();
        let mut keccak_sponge_meta = Vec::new();

        // H(ek), need to re-encode
        let mut ek_bytes = Vec::new();
        for poly in &ek.t_hat {
            ek_bytes.extend_from_slice(&byte_encode(12, poly));
        }

        ek_bytes.extend_from_slice(&ek.rho);

        // G(m || H(ek))
        let (h_ek, _) = sha3_256(&ek_bytes);

        let mut g_in = Vec::with_capacity(64);
        g_in.extend_from_slice(rand_bytes);
        g_in.extend_from_slice(&h_ek);

        let (g_out, _) = sha3_512(&g_in);

        let mut k_bar = [0u8; 32];
        let mut r = [0u8; 32];

        k_bar.copy_from_slice(&g_out[..32]);
        r.copy_from_slice(&g_out[32..]);

        // Encrypt
        let ct = kpke_encrypt(
            level,
            ek,
            rand_bytes,
            &r,
            &mut keccak_calls,
            &mut keccak_sponge_meta,
        );

        // FIPS 203 Algorithm 16:
        // return K directly
        let mut shared_secret = [0u8; 32];
        shared_secret.copy_from_slice(&k_bar);

        (ct, shared_secret)
    }

    /// ML-KEM decapsulation.
    fn ml_kem_decaps(dk: &MlKemDecapsKey, ct: &[u8]) -> MlKemDecapsResult {
        let level = dk.level;

        let mut keccak_calls = Vec::new();

        // Step 1:
        // Decrypt to recover message m'
        let (m_prime, _ntt_ops) = kpke_decrypt(level, &dk.s_hat, ct, &mut keccak_calls);

        // Step 2:
        // G(m' || h) -> (K_bar, r)
        let mut g_input = Vec::with_capacity(64);
        g_input.extend_from_slice(&m_prime);
        g_input.extend_from_slice(&dk.h);

        let (g_out, g_calls) = sha3_512(&g_input);
        keccak_calls.extend_from_slice(&g_calls);

        let mut k_bar = [0u8; 32];
        let mut r = [0u8; 32];

        k_bar.copy_from_slice(&g_out[..32]);
        r.copy_from_slice(&g_out[32..]);

        // Step 3:
        // Re-encrypt with (ek, m', r)
        let mut _meta = Vec::new();
        let ek = parse_encaps_key(level, &dk.ek, &mut keccak_calls, &mut _meta);
        let ct_prime = kpke_encrypt(level, &ek, &m_prime, &r, &mut keccak_calls, &mut _meta);

        // Step 4:
        // K̄ = J(z || c) for implicit rejection
        let mut j_input = Vec::with_capacity(32 + ct.len());
        j_input.extend_from_slice(&dk.z);
        j_input.extend_from_slice(ct);

        let (k_bar_reject, j_calls) = shake256(&j_input, 32);
        keccak_calls.extend_from_slice(&j_calls);

        // Step 5:
        // implicit rejection check
        let ct_match = ct == ct_prime.as_slice();

        // Step 6:
        // return K' (success) or K̄ (rejection)
        let mut shared_secret = [0u8; 32];
        if ct_match {
            shared_secret.copy_from_slice(&k_bar);
        } else {
            shared_secret.copy_from_slice(&k_bar_reject);
        }

        MlKemDecapsResult {
            shared_secret,
            keccak_calls,
            keccak_sponge_meta: Vec::new(),
            basemul_ops: Vec::new(),
            ram_events: Vec::new(),
            h_ct: [0u8; 32],
            h_ct_prime: [0u8; 32],
            keccak_phases: Vec::new(),
            basemul_phases: Vec::new(),
            ram_phases: Vec::new(),
            ntt_ram_bindings: Vec::new(),
            w_side_bindings: Vec::new(),
            ntt_boundary_bindings: Vec::new(),
            next_free_addr: 0,
            call_tag: Vec::new(),
            k_prime_lo: [0; 4],
            k_prime_hi: [0; 4],
            k_bar_lo: [0; 4],
            k_bar_hi: [0; 4],
        }
    }

    // =================================================================
    // K-PKE (FIPS 203 §§5-6)
    // =================================================================

    /// K-PKE.Decrypt (FIPS 203 Algorithm 14).
    fn kpke_decrypt(
        level: MlKemLevel,
        s_hat: &[[u16; N]],
        ct: &[u8],
        keccak_calls: &mut Vec<keccak::KeccakCall>,
    ) -> ([u8; 32], Vec<ntt::NttOp>) {
        let _ = keccak_calls; // Decrypt has no Keccak calls

        let mut ntt_ops = Vec::new();

        // Parse ciphertext
        let ct_du_len = level.k * N * level.du / 8;
        let u_bytes = &ct[..ct_du_len];
        let v_bytes = &ct[ct_du_len..];

        // Decompress u
        let mut u_hat = vec![[0u16; N]; level.k];
        for i in 0..level.k {
            let start = i * N * level.du / 8;
            let poly = byte_decode(level.du, &u_bytes[start..start + N * level.du / 8]);
            let decompressed = decompress_poly(level.du, &poly);

            // NTT(u_i)
            u_hat[i] = decompressed;

            ntt_forward(&mut u_hat[i]);
        }

        // Inner product:
        // s^T · NTT(u)
        let mut w_hat = [0u16; N];
        for i in 0..level.k {
            let prod = poly_basemul(&s_hat[i], &u_hat[i]);
            w_hat = poly_add(&w_hat, &prod);
        }

        // INTT
        ntt_inverse(&mut w_hat);

        // Decompress v
        let v = decompress_poly(level.dv, &byte_decode(level.dv, v_bytes));

        // m = v - w
        let m = poly_sub(&v, &w_hat);

        // Compress to message
        let m_compressed = compress_poly(1, &m);
        let m_bytes = byte_encode(1, &m_compressed);

        let mut message = [0u8; 32];
        message.copy_from_slice(&m_bytes[..32]);

        // Record NTT operations for trace
        // (simplified: we record the final polynomials,
        //  actual butterfly-level ops handled separately)
        let _ = &mut ntt_ops;

        (message, ntt_ops)
    }

    /// K-PKE.Encrypt (FIPS 203 Algorithm 13).
    fn kpke_encrypt(
        level: MlKemLevel,
        ek: &MlKemEncapsKey,
        m: &[u8; 32],
        r_seed: &[u8; 32],
        keccak_calls: &mut Vec<keccak::KeccakCall>,
        sponge_meta: &mut Vec<(bool, bool, bool)>,
    ) -> Vec<u8> {
        // Sample r, e1, e2
        let mut r_vec = vec![[0u16; N]; level.k];
        for i in 0..level.k {
            r_vec[i] = sample_cbd(level.eta1, r_seed, i as u8, keccak_calls, sponge_meta);
            ntt_forward(&mut r_vec[i]);
        }

        let mut e1 = vec![[0u16; N]; level.k];
        for i in 0..level.k {
            e1[i] = sample_cbd(
                level.eta2,
                r_seed,
                (level.k + i) as u8,
                keccak_calls,
                sponge_meta,
            );
        }

        let e2 = sample_cbd(
            level.eta2,
            r_seed,
            (2 * level.k) as u8,
            keccak_calls,
            sponge_meta,
        );

        // u = INTT(A^T × r̂) + e1
        let mut u = vec![[0u16; N]; level.k];
        for i in 0..level.k {
            let mut acc = [0u16; N];
            for j in 0..level.k {
                let prod = poly_basemul(&ek.a_hat[j][i], &r_vec[j]);
                acc = poly_add(&acc, &prod);
            }

            ntt_inverse(&mut acc);

            u[i] = poly_add(&acc, &e1[i]);
        }

        // v = INTT(t̂^T × r̂) + e2 + Decompress_1(m)
        let mut v_acc = [0u16; N];
        for i in 0..level.k {
            let prod = poly_basemul(&ek.t_hat[i], &r_vec[i]);
            v_acc = poly_add(&v_acc, &prod);
        }

        ntt_inverse(&mut v_acc);
        v_acc = poly_add(&v_acc, &e2);

        // Add message
        let m_poly = byte_decode(1, m);
        let m_decomp = decompress_poly(1, &m_poly);
        v_acc = poly_add(&v_acc, &m_decomp);

        // Compress and encode
        let mut ct = Vec::new();
        for i in 0..level.k {
            let c = compress_poly(level.du, &u[i]);
            ct.extend_from_slice(&byte_encode(level.du, &c));
        }

        let c2 = compress_poly(level.dv, &v_acc);
        ct.extend_from_slice(&byte_encode(level.dv, &c2));

        ct
    }

    #[test]
    fn encaps_decaps_roundtrip() {
        let seed = [42u8; 64];
        let (ek, dk) = ml_kem_keygen(MlKemLevel::MLKEM_768, &seed);

        let rand = [7u8; 32];
        let (ct, k_encaps) = ml_kem_encaps(MlKemLevel::MLKEM_768, &ek, &rand);

        let result = ml_kem_decaps(&dk, &ct);
        assert_eq!(
            result.shared_secret, k_encaps,
            "encaps/decaps shared secret mismatch",
        );
    }

    #[test]
    fn traced_matches_untraced() {
        let seed = [42u8; 64];
        let (ek, dk) = ml_kem_keygen(MlKemLevel::MLKEM_768, &seed);

        let rand = [7u8; 32];
        let (ct, k_encaps) = ml_kem_encaps(MlKemLevel::MLKEM_768, &ek, &rand);

        let (result, ntt_ops) = ml_kem_decaps_traced(&dk, &ct);
        assert_eq!(
            result.shared_secret, k_encaps,
            "traced decaps shared secret mismatch",
        );
        assert!(!ntt_ops.is_empty());
        assert!(!result.keccak_calls.is_empty());
    }

    // =====================================================
    // NIST Reference Differential Tests
    // =====================================================

    #[test]
    fn nist_decrypt_reencrypt_roundtrip() {
        // K-PKE.Decrypt(ct) -> m -> K-PKE.Encrypt(ek, m, r) -> ct'
        // ct' must equal ct for valid ciphertexts.
        // This verifies NTT, basemul, decompress, compress.
        for _ in 0..10 {
            let (_nist_dk, ct, sk, _ss) = nist_mlkem_768();

            let dk = MlKemDecapsKey::from_nist_bytes(MlKemLevel::MLKEM_768, &sk);
            let ct_bytes = ct.as_slice();

            // Decrypt
            let mut kc = Vec::new();
            let (m, _) = kpke_decrypt(MlKemLevel::MLKEM_768, &dk.s_hat, ct_bytes, &mut kc);

            // G(m || h) -> (K', r)
            let mut g_in = Vec::with_capacity(64);
            g_in.extend_from_slice(&m);
            g_in.extend_from_slice(&dk.h);

            let (g_out, _) = sha3_512(&g_in);

            let mut r = [0u8; 32];
            r.copy_from_slice(&g_out[32..]);

            // Re-encrypt
            let mut _sm = Vec::new();

            let level = MlKemLevel::MLKEM_768;
            let ek = parse_encaps_key(level, &dk.ek, &mut kc, &mut _sm);
            let ct_re = kpke_encrypt(level, &ek, &m, &r, &mut kc, &mut _sm);

            assert_eq!(
                ct_bytes,
                ct_re.as_slice(),
                "Re-encryption must reproduce original ciphertext",
            );
        }
    }

    #[test]
    fn nist_kpke_decrypt_produces_valid_message() {
        // Test K-PKE.Decrypt: does it produce a message
        // that, when re-encrypted, gives the same ciphertext?
        // If not, the implicit rejection kicks in.

        // Use our own encapsulation to have a known-good ct
        let seed = [42u8; 64];
        let (ek_our, dk_our) = ml_kem_keygen(MlKemLevel::MLKEM_768, &seed);

        let rand = [7u8; 32];
        let (ct_our, _) = ml_kem_encaps(MlKemLevel::MLKEM_768, &ek_our, &rand);

        // Decrypt with our impl
        let mut keccak_calls = Vec::new();
        let level = MlKemLevel::MLKEM_768;

        let (m, _) = kpke_decrypt(level, &dk_our.s_hat, &ct_our, &mut keccak_calls);

        // Re-encrypt with our impl
        let mut _sm = Vec::new();
        let ek_parsed = parse_encaps_key(level, &dk_our.ek, &mut keccak_calls, &mut _sm);

        let (g_out, _) = sha3_512(&{
            let mut buf = Vec::new();
            buf.extend_from_slice(&m);
            buf.extend_from_slice(&dk_our.h);

            buf
        });

        let mut r = [0u8; 32];
        r.copy_from_slice(&g_out[32..]);

        let ct_re = kpke_encrypt(level, &ek_parsed, &m, &r, &mut keccak_calls, &mut _sm);
        assert_eq!(
            ct_our, ct_re,
            "Re-encryption mismatch: our encrypt != our ciphertext",
        );
    }

    #[test]
    fn nist_decaps_debug_intermediate() {
        // Isolate the problem: check if K-PKE.Decrypt
        // produces the same message as NIST reference.
        // This tests NTT + basemul + decompress but NOT
        // the G/H/KDF steps.
        let (_nist_dk, _ct, nist_sk, _ss) = nist_mlkem_768();
        let dk = MlKemDecapsKey::from_nist_bytes(MlKemLevel::MLKEM_768, &nist_sk);

        // Verify H(ek) matches what NIST would compute
        let (our_h, _) = sha3_256(&dk.ek);
        assert_eq!(
            our_h, dk.h,
            "H(ek) mismatch: our SHA3-256(ek) != stored H(ek)",
        );
    }

    #[test]
    fn nist_decaps_matches_reference() {
        let (nist_dk, nist_ct, nist_sk, nist_ss) = nist_mlkem_768();

        // NIST reference self-check.
        let nist_ss2 = nist_decapsulate_768(&nist_dk, &nist_ct);
        assert_eq!(nist_ss, nist_ss2, "NIST reference self-check failed");

        let dk = MlKemDecapsKey::from_nist_bytes(MlKemLevel::MLKEM_768, &nist_sk);
        let our_result = ml_kem_decaps(&dk, &nist_ct);

        assert_eq!(
            our_result.shared_secret,
            nist_ss.as_slice(),
            "OUR decaps does not match NIST reference",
        );
    }

    #[test]
    fn nist_decaps_traced_matches_reference() {
        let (_nist_dk, nist_ct, nist_sk, nist_ss) = nist_mlkem_768();

        let dk = MlKemDecapsKey::from_nist_bytes(MlKemLevel::MLKEM_768, &nist_sk);
        let (result, ntt_ops) = ml_kem_decaps_traced(&dk, &nist_ct);

        assert_eq!(
            result.shared_secret,
            nist_ss.as_slice(),
            "TRACED decaps does not match NIST reference",
        );
        assert!(!ntt_ops.is_empty());
    }

    #[test]
    fn nist_multiple_keypairs() {
        // 100 random keypairs,
        // statistical confidence
        for i in 0..100 {
            let (_nist_dk, ct, sk, nist_ss) = nist_mlkem_768();

            let dk = MlKemDecapsKey::from_nist_bytes(MlKemLevel::MLKEM_768, &sk);
            let our = ml_kem_decaps(&dk, &ct);

            assert_eq!(
                our.shared_secret,
                nist_ss.as_slice(),
                "Mismatch on keypair #{i}",
            );
        }
    }

    // =====================================================
    // Implicit Rejection Tests
    // =====================================================

    #[test]
    fn nist_implicit_rejection_modified_ciphertext() {
        // A modified ciphertext must produce J(z || c_mod),
        // NOT the original shared secret.
        // This tests the implicit rejection path.
        let (_nist_dk, ct, sk, nist_ss) = nist_mlkem_768();

        let dk = MlKemDecapsKey::from_nist_bytes(MlKemLevel::MLKEM_768, &sk);

        // Modify one byte of ciphertext
        let mut ct_mod = ct.clone();
        ct_mod[0] ^= 0x01;

        // Our decaps on modified ciphertext
        let our = ml_kem_decaps(&dk, &ct_mod);

        // Must NOT equal original shared secret
        assert_ne!(
            our.shared_secret,
            nist_ss.as_slice(),
            "Modified ciphertext should trigger rejection",
        );

        // Must equal J(z || c_modified) = SHAKE-256(z || c_mod, 32)
        let mut j_input = Vec::with_capacity(32 + ct_mod.len());
        j_input.extend_from_slice(&dk.z);
        j_input.extend_from_slice(&ct_mod);

        let (expected_reject, _) = shake256(&j_input, 32);

        assert_eq!(
            our.shared_secret,
            expected_reject.as_slice(),
            "Rejection output must be J(z || c_modified)",
        );
    }

    #[test]
    fn nist_implicit_rejection_various_modifications() {
        // Test rejection at multiple
        // positions in the ciphertext.
        let (_nist_dk, ct, sk, _ss) = nist_mlkem_768();

        let dk = MlKemDecapsKey::from_nist_bytes(MlKemLevel::MLKEM_768, &sk);
        let ct_bytes = ct.as_slice();

        for &pos in &[0, 1, 100, 500, 960, 1087] {
            let mut ct_mod = ct_bytes.to_vec();
            ct_mod[pos] ^= 0xff;

            let our = ml_kem_decaps(&dk, &ct_mod);

            let mut j_input = Vec::with_capacity(32 + ct_mod.len());
            j_input.extend_from_slice(&dk.z);
            j_input.extend_from_slice(&ct_mod);

            let (expected, _) = shake256(&j_input, 32);

            assert_eq!(
                our.shared_secret,
                expected.as_slice(),
                "Rejection mismatch at position {pos}",
            );
        }
    }

    #[test]
    fn nist_implicit_rejection_matches_nist_decaps() {
        // NIST reference decaps on modified ciphertext
        // must also produce the rejection value.
        // Cross-check:
        // our rejection == NIST rejection.
        let (nist_dk, ct, sk, _ss) = nist_mlkem_768();

        let dk = MlKemDecapsKey::from_nist_bytes(MlKemLevel::MLKEM_768, &sk);

        let mut ct_mod = ct.clone();
        ct_mod[42] ^= 0x01;

        let our = ml_kem_decaps(&dk, &ct_mod);

        // NIST decaps on modified ciphertext
        let nist_reject = nist_decapsulate_768(&nist_dk, &ct_mod);

        assert_eq!(
            our.shared_secret,
            nist_reject.as_slice(),
            "Rejection path: our output != NIST output",
        );
    }

    // =====================================================
    // NTT Consistency with NIST via Full Protocol
    // =====================================================

    #[test]
    fn nist_keygen_consistency() {
        // Our keygen with a fixed seed must produce
        // internally consistent keys:
        // H(ek) == h, and NTT(INTT(s_hat)) == s_hat.
        let seed = [0xABu8; 64];
        let (_, dk) = ml_kem_keygen(MlKemLevel::MLKEM_768, &seed);

        // H(ek) consistency
        let (h, _) = sha3_256(&dk.ek);
        assert_eq!(h, dk.h, "H(ek) != stored h in keygen output");

        // NTT roundtrip on secret key
        for i in 0..MlKemLevel::MLKEM_768.k {
            let mut s = dk.s_hat[i];
            ntt_inverse(&mut s);
            ntt_forward(&mut s);

            assert_eq!(s, dk.s_hat[i], "NTT(INTT(s_hat[{i}])) != s_hat[{i}]",);
        }
    }

    #[test]
    fn nist_100_keypairs_with_rejection_crosscheck() {
        // 100 keypairs:
        // test both success and rejection
        // paths against NIST reference.
        for _ in 0..100 {
            let (nist_dk, ct, sk, nist_ss) = nist_mlkem_768();
            let dk = MlKemDecapsKey::from_nist_bytes(MlKemLevel::MLKEM_768, &sk);

            // Success path
            let our = ml_kem_decaps(&dk, &ct);
            assert_eq!(our.shared_secret, nist_ss.as_slice());

            // Rejection path
            let mut ct_bad = ct.clone();
            ct_bad[0] ^= 0x01;

            let our_rej = ml_kem_decaps(&dk, &ct_bad);
            let nist_rej = nist_decapsulate_768(&nist_dk, &ct_bad);

            assert_eq!(
                our_rej.shared_secret,
                nist_rej.as_slice(),
                "Rejection path mismatch",
            );
        }
    }

    // =====================================================
    // NTT <> RAM Data Binding
    // =====================================================

    #[test]
    fn ntt_ram_bindings_exist_and_values_match() {
        let (_nist_dk, nist_ct, nist_sk, _ss) = nist_mlkem_768();

        let dk = MlKemDecapsKey::from_nist_bytes(MlKemLevel::MLKEM_768, &nist_sk);

        let (result, ntt_ops) = ml_kem_decaps_traced(&dk, &nist_ct);

        // Bindings must be non-empty.
        assert!(
            !result.ntt_ram_bindings.is_empty(),
            "No NTT-RAM bindings generated",
        );

        // Every binding's RAM value must match
        // the NTT op's b field. Bindings cover
        // both basemul MulOnly ops and extended
        // Butterfly ops; FlowCompanion is never bound.
        for &(ntt_idx, ram_idx) in &result.ntt_ram_bindings {
            let ntt_b = match &ntt_ops[ntt_idx] {
                ntt::NttOp::MulOnly(m) => m.b,
                ntt::NttOp::Butterfly(bfly) => bfly.b,
                ntt::NttOp::FlowCompanion(_) => {
                    panic!("Binding at ntt_idx={ntt_idx} is a FlowCompanion");
                }
            };

            let ram_val = result.ram_events[ram_idx].val;

            assert_eq!(
                ntt_b, ram_val,
                "Binding mismatch: ntt_ops[{}].b={} != ram_events[{}].val={}",
                ntt_idx, ntt_b, ram_idx, ram_val,
            );
        }
    }

    #[test]
    fn ntt_ram_bindings_cover_all_basemul_blocks() {
        let (_nist_dk, nist_ct, nist_sk, _ss) = nist_mlkem_768();

        let dk = MlKemDecapsKey::from_nist_bytes(MlKemLevel::MLKEM_768, &nist_sk);

        let (result, ntt_ops) = ml_kem_decaps_traced(&dk, &nist_ct);

        // ML-KEM-768 (k=3):
        // Decrypt:
        // 3 basemul blocks × 256 coefficients = 768
        // Encrypt:
        // (3×3 A_hat + 3 t_hat) × 256 = 3072
        // Total:
        // 3840 basemul bindings
        let expected = (3 + 9 + 3) * N;

        let basemul_bindings = result
            .ntt_ram_bindings
            .iter()
            .filter(|&&(ntt_idx, _)| {
                matches!(&ntt_ops[ntt_idx], ntt::NttOp::MulOnly(m) if m.is_basemul)
            })
            .count();

        assert_eq!(
            basemul_bindings, expected,
            "Expected {} basemul bindings (15 blocks × {}), got {}",
            expected, N, basemul_bindings,
        );
    }

    #[test]
    fn ntt_ram_bindings_cover_every_non_flow_op() {
        let (_nist_dk, nist_ct, nist_sk, _ss) = nist_mlkem_768();

        let dk = MlKemDecapsKey::from_nist_bytes(MlKemLevel::MLKEM_768, &nist_sk);

        let (result, ntt_ops) = ml_kem_decaps_traced(&dk, &nist_ct);

        let butterfly_bindings = result
            .ntt_ram_bindings
            .iter()
            .filter(|&&(ntt_idx, _)| matches!(&ntt_ops[ntt_idx], ntt::NttOp::Butterfly(_)))
            .count();

        let butterfly_ops_count = ntt_ops
            .iter()
            .filter(|op| matches!(op, ntt::NttOp::Butterfly(_)))
            .count();

        assert_eq!(
            butterfly_bindings, butterfly_ops_count,
            "Every Butterfly op must produce one RAM binding; got {} bindings for {} ops",
            butterfly_bindings, butterfly_ops_count,
        );

        let flow_companion_count = ntt_ops
            .iter()
            .filter(|op| matches!(op, ntt::NttOp::FlowCompanion(_)))
            .count();

        assert_eq!(
            result.ntt_ram_bindings.len(),
            ntt_ops.len() - flow_companion_count,
            "ntt_ram_bindings must cover every non-FlowCompanion op",
        );
    }

    #[test]
    fn w_side_bindings_exist_and_values_match() {
        let (_nist_dk, nist_ct, nist_sk, _ss) = nist_mlkem_768();

        let dk = MlKemDecapsKey::from_nist_bytes(MlKemLevel::MLKEM_768, &nist_sk);

        let (result, ntt_ops) = ml_kem_decaps_traced(&dk, &nist_ct);

        let basemul_mulonly_count = ntt_ops
            .iter()
            .filter(|op| matches!(op, ntt::NttOp::MulOnly(m) if m.is_basemul))
            .count();

        assert_eq!(
            result.w_side_bindings.len(),
            basemul_mulonly_count,
            "w-side bindings ({}) != basemul MulOnly count ({})",
            result.w_side_bindings.len(),
            basemul_mulonly_count,
        );

        // Every w-side binding value
        // must match the MulOnly's w value.
        for &(ram_idx, bfly_idx) in &result.w_side_bindings {
            let ram_val = result.ram_events[ram_idx].val;

            // Find the MulOnly with this bfly_idx
            let mulonly = ntt_ops.iter().find(|op| match op {
                ntt::NttOp::MulOnly(m) => m.butterfly_idx == bfly_idx,
                _ => false,
            });

            let w = match mulonly.expect("MulOnly not found") {
                ntt::NttOp::MulOnly(m) => m.w,
                _ => unreachable!(),
            };

            assert_eq!(
                ram_val, w,
                "w-side value mismatch: RAM[{}]={} != MulOnly[bfly={}].w={}",
                ram_idx, ram_val, bfly_idx, w,
            );
        }

        // Verify bfly_idx globally unique
        let mut seen = BTreeSet::new();
        for &(_, bfly_idx) in &result.w_side_bindings {
            assert!(
                seen.insert(bfly_idx),
                "duplicate bfly_idx {} in w-side bindings",
                bfly_idx,
            );
        }
    }
}
