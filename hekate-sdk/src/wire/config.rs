// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use flatbuffers::{Allocator, FlatBufferBuilder};
use hekate_core::config::Config;
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
            ldt_support_size: config.ldt_support_size as u32,
            min_security_bits: config.min_security_bits as u32,
            outer_queries: config.outer_queries as u32,
            zero_knowledge: config.zero_knowledge,
        },
    )
}

pub fn deserialize_config(fb_config: fb::Config<'_>) -> Result<Config> {
    Ok(Config {
        num_queries: fb_config.num_queries() as usize,
        ldt_support_size: fb_config.ldt_support_size() as usize,
        min_security_bits: fb_config.min_security_bits() as usize,
        outer_queries: fb_config.outer_queries() as usize,
        zero_knowledge: fb_config.zero_knowledge(),
    })
}
