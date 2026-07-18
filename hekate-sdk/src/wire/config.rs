// SPDX-License-Identifier: Apache-2.0
// This file is part of the hekate project.
// Copyright (C) 2026 Andrei Kochergin <andrei@oumuamua.dev>
// Copyright (C) 2026 Oumuamua Labs <info@oumuamua.dev>.
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

use flatbuffers::FlatBufferBuilder;
use hekate_core::config::Config;
use hekate_core::errors::Result;

use crate::generated::program as fb;

pub fn serialize_config<'a>(
    fbb: &mut FlatBufferBuilder<'a>,
    config: &Config,
) -> flatbuffers::WIPOffset<fb::Config<'a>> {
    fb::Config::create(
        fbb,
        &fb::ConfigArgs {
            inv_rate: config.inv_rate as u32,
            num_queries: config.num_queries as u32,
            sumcheck_blinding_factor: config.sumcheck_blinding_factor as u32,
            ldt_support_size: config.ldt_support_size as u32,
            min_security_bits: config.min_security_bits as u32,
        },
    )
}

pub fn deserialize_config(fb_config: fb::Config<'_>) -> Result<Config> {
    Ok(Config {
        inv_rate: fb_config.inv_rate() as usize,
        num_queries: fb_config.num_queries() as usize,
        sumcheck_blinding_factor: fb_config.sumcheck_blinding_factor() as usize,
        ldt_support_size: fb_config.ldt_support_size() as usize,
        min_security_bits: fb_config.min_security_bits() as usize,
    })
}
