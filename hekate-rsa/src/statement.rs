// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! `s^65537 mod N == PKCS1-v1_5(H)`, `H` chained over committed
//! block words; FIPS 180-4 padding is the caller's, not the AIR's.
//!
//! `N` is public; the signature and the message are witness. The host
//! table runs the SHA-256 chaining, requests both chiplets, and compares
//! the modexp output against the encoding on its last block row.

use alloc::vec;
use alloc::vec::Vec;
use hekate_core::errors::{self, Error};
use hekate_core::trace::{ColumnTrace, ColumnType, TraceBuilder};
use hekate_gadgets::chiplets::bignum::modexp::LIMBS32;
use hekate_gadgets::{CpuModexpBlock, Modexp, ModexpChiplet};
use hekate_math::{Bit, Block128, TowerField};
use hekate_program::circuit::{Circuit, CircuitProgram};
use hekate_program::{Air, FixedShape, ProgramInstance, ProgramWitness};
use hekate_sha2::{CpuSha256Block, IV, STATE_WORDS, Sha256Call, Sha256Chiplet, pad_message};

use crate::pkcs1::{DIGEST_LIMBS, digest_word_of_limb, padding_limbs};

type F = Block128;

const SHA_ACTIVE: usize = CpuSha256Block::COLUMNS + CpuModexpBlock::COLUMNS;
const SHA_CHAIN: usize = SHA_ACTIVE + 1;
const VERIFY: usize = SHA_CHAIN + 1;

/// Host table plus the SHA-256 and modexp chiplets,
/// compiled into one program with both buses wired.
pub struct Pkcs1Statement {
    program: CircuitProgram<F>,
    sha: Sha256Chiplet<F>,
    modexp: ModexpChiplet,
    sha_block: CpuSha256Block,
    modexp_block: CpuModexpBlock,
    num_blocks: usize,
    cpu_rows: usize,
}

impl Pkcs1Statement {
    /// Compiles the statement for a `num_blocks`-block message.
    ///
    /// # Errors
    /// `num_blocks` zero or above `cpu_rows`, a height that is
    /// not a power of two, or a chiplet rejects its parameters.
    pub fn new(
        num_blocks: usize,
        rounds_per_row: usize,
        sha_rows: usize,
        cpu_rows: usize,
    ) -> errors::Result<Self> {
        if num_blocks == 0 || num_blocks > cpu_rows || !cpu_rows.is_power_of_two() {
            return Err(Error::Protocol {
                protocol: "rsa_pkcs1",
                message: "cpu_rows must be a power of two at least num_blocks",
            });
        }

        let sha = Sha256Chiplet::<F>::new(sha_rows, num_blocks, rounds_per_row)?;
        let modexp = ModexpChiplet::new()?;

        let verify_row = num_blocks - 1;

        let mut cx = Circuit::<F>::new("RsaPkcs1", cpu_rows)?;

        let sha_block = CpuSha256Block::declare(&mut cx, 0);
        let modexp_block = CpuModexpBlock::declare(&mut cx, CpuSha256Block::COLUMNS);

        let sha_active = cx.column(ColumnType::Bit);
        let sha_chain = cx.column(ColumnType::Bit);
        let verify = cx.column(ColumnType::Bit);

        let prefix = |count: usize| FixedShape::Cadence {
            stride: 1,
            count,
            origin: 0,
            values: vec![F::ONE],
        };

        cx.fix(sha_active, prefix(num_blocks));
        cx.fix(sha_chain, prefix(num_blocks - 1));
        cx.fix(verify, FixedShape::Sparse(vec![(verify_row, F::ONE)]));

        sha_block.connect(&mut cx, sha_active)?;
        modexp_block.connect(&mut cx, verify)?;

        for (i, &iv) in IV.iter().enumerate() {
            cx.boundary(sha_block.h_in_words.at(i), 0, F::from(u128::from(iv)));
        }

        for j in 0..LIMBS32 {
            cx.publish(modexp_block.modulus.at(j), verify_row);
        }

        constrain(
            &cx,
            &sha_block,
            &modexp_block,
            sha_chain.index(),
            verify.index(),
        );

        cx.attach(sha.def()?);
        cx.attach(modexp.def()?);

        let program = cx.compile()?;

        if program.column_layout() != cpu_layout().as_slice() {
            return Err(Error::Protocol {
                protocol: "rsa_pkcs1",
                message: "host layout diverged from the circuit declaration",
            });
        }

        Ok(Self {
            program,
            sha,
            modexp,
            sha_block,
            modexp_block,
            num_blocks,
            cpu_rows,
        })
    }

    pub fn program(&self) -> &CircuitProgram<F> {
        &self.program
    }

    pub fn instance(&self, modulus: &[u32; LIMBS32]) -> ProgramInstance<F> {
        let public = modulus
            .iter()
            .map(|&limb| F::from(u128::from(limb)))
            .collect();

        ProgramInstance::new(self.cpu_rows, public)
    }

    /// Traces the host table and both chiplets for one signature.
    ///
    /// # Errors
    /// `message` does not pad to the declared block count,
    /// the signature is not reduced, or a trace overflows.
    pub fn witness(
        &self,
        message: &[u8],
        modulus: &[u32; LIMBS32],
        signature: &[u32; LIMBS32],
    ) -> errors::Result<ProgramWitness<F, ColumnTrace>> {
        let blocks = pad_message(message);

        if blocks.len() != self.num_blocks {
            return Err(Error::Protocol {
                protocol: "rsa_pkcs1",
                message: "message does not pad to the declared block count",
            });
        }

        let modexp = Modexp::new(modulus, signature)?;

        let mut calls = Vec::with_capacity(self.num_blocks);
        let mut h = IV;

        for (b, block) in blocks.iter().enumerate() {
            let call = Sha256Call {
                h_in: h,
                block: *block,
                request_idx: b as u32,
            };

            h = call.h_out();

            calls.push(call);
        }

        let verify_row = self.num_blocks - 1;

        let mut tb = TraceBuilder::new(&cpu_layout(), self.cpu_rows.trailing_zeros() as usize)?;

        for (b, call) in calls.iter().enumerate() {
            self.sha_block.write(&mut tb, b, call)?;

            tb.set_bit(SHA_ACTIVE, b, Bit::ONE)?;

            if b + 1 < self.num_blocks {
                tb.set_bit(SHA_CHAIN, b, Bit::ONE)?;
            }
        }

        self.modexp_block.write(&mut tb, verify_row, &modexp)?;

        tb.set_bit(VERIFY, verify_row, Bit::ONE)?;

        let sha_trace = self.sha.trace(&calls)?;
        let modexp_trace = self.modexp.trace(&modexp, verify_row as u32)?;

        Ok(ProgramWitness::new(tb.build()).with_chiplets(vec![sha_trace, modexp_trace]))
    }
}

pub fn cpu_layout() -> Vec<ColumnType> {
    let mut layout = CpuSha256Block::layout().to_vec();
    layout.extend(CpuModexpBlock::layout());
    layout.extend([ColumnType::Bit; 3]);

    layout
}

fn constrain(
    cx: &Circuit<F>,
    sha_block: &CpuSha256Block,
    modexp_block: &CpuModexpBlock,
    sha_chain: usize,
    verify: usize,
) {
    let cs = cx.cs();

    let chain = cs.col(sha_chain);
    let verify = cs.col(verify);
    let padding = padding_limbs();

    for i in 0..STATE_WORDS {
        let h_out = cs.col(sha_block.h_out_words.at(i).index());

        cs.assert_zero_when(chain, cs.next(sha_block.h_in_words.at(i).index()) + h_out);
    }

    for j in 0..DIGEST_LIMBS {
        let limb = cs.col(modexp_block.result.at(j).index());
        let word = cs.col(sha_block.h_out_words.at(digest_word_of_limb(j)).index());

        cs.assert_zero_when(verify, limb + word);
    }

    for (j, &constant) in padding.iter().enumerate().skip(DIGEST_LIMBS) {
        let limb = cs.col(modexp_block.result.at(j).index());
        let expected = cs.constant(F::from(u128::from(constant)));

        cs.assert_zero_when(verify, limb + expected);
    }
}
