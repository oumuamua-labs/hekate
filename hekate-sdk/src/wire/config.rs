// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use flatbuffers::{Allocator, FlatBufferBuilder};
use hekate_core::config::{CodeRate, Config};
use hekate_core::errors::Result;

use crate::generated::program as fb;

pub fn serialize_config<'a, A: Allocator + 'a>(
    fbb: &mut FlatBufferBuilder<'a, A>,
    config: &Config,
) -> flatbuffers::WIPOffset<fb::Config<'a>> {
    fb::Config::create(
        fbb,
        &fb::ConfigArgs {
            num_queries: config.num_queries as u32,
            min_security_bits: config.min_security_bits as u32,
            outer_queries: config.outer_queries as u32,
            zero_knowledge: config.zero_knowledge,
            log_inv_rate: config.code_rate.log_inv_rate(),
            fold_arity_log: config.fold_arity_log as u32,
            early_stop_vars: config.early_stop_vars as u32,
        },
    )
}

pub fn deserialize_config(fb_config: fb::Config<'_>) -> Result<Config> {
    Ok(Config {
        code_rate: CodeRate::try_from(fb_config.log_inv_rate())?,
        num_queries: fb_config.num_queries() as usize,
        fold_arity_log: fb_config.fold_arity_log() as usize,
        early_stop_vars: fb_config.early_stop_vars() as usize,
        min_security_bits: fb_config.min_security_bits() as usize,
        outer_queries: fb_config.outer_queries() as usize,
        zero_knowledge: fb_config.zero_knowledge(),
    })
}
