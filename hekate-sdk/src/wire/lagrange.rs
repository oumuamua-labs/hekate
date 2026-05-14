// SPDX-License-Identifier: Apache-2.0
// This file is part of the hekate-math project.
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

use alloc::vec::Vec;
use flatbuffers::FlatBufferBuilder;
use hekate_core::errors::Result;
use hekate_program::{LagrangePin, LagrangePoint};

use super::wire_err;
use crate::generated::program as fb;

pub fn serialize_lagrange_pin<'a>(
    fbb: &mut FlatBufferBuilder<'a>,
    pin: &LagrangePin,
) -> flatbuffers::WIPOffset<fb::LagrangePin<'a>> {
    let (kind, custom_bits) = match &pin.point {
        LagrangePoint::LastRow => (fb::LagrangePointKind::LastRow, None),
        LagrangePoint::FirstRow => (fb::LagrangePointKind::FirstRow, None),
        LagrangePoint::Custom(bits) => {
            let bytes: Vec<u8> = bits.iter().map(|&b| b as u8).collect();
            let off = fbb.create_vector(&bytes);

            (fb::LagrangePointKind::Custom, Some(off))
        }
    };

    fb::LagrangePin::create(
        fbb,
        &fb::LagrangePinArgs {
            col_idx: pin.col_idx as u32,
            kind,
            custom_bits,
        },
    )
}

pub fn deserialize_lagrange_pin(fb_pin: fb::LagrangePin<'_>) -> Result<LagrangePin> {
    let col_idx = fb_pin.col_idx() as usize;

    let point = match fb_pin.kind() {
        fb::LagrangePointKind::LastRow => LagrangePoint::LastRow,
        fb::LagrangePointKind::FirstRow => LagrangePoint::FirstRow,
        fb::LagrangePointKind::Custom => {
            let bytes = fb_pin
                .custom_bits()
                .ok_or(wire_err("missing custom_bits for Custom Lagrange pin"))?;

            let mut bits = Vec::with_capacity(bytes.len());
            for i in 0..bytes.len() {
                let b = bytes.get(i);
                if b > 1 {
                    return Err(wire_err("Custom Lagrange pin bit must be 0 or 1"));
                }

                bits.push(b == 1);
            }

            LagrangePoint::Custom(bits)
        }
        _ => return Err(wire_err("unknown LagrangePointKind")),
    };

    Ok(LagrangePin { col_idx, point })
}

pub fn serialize_pins<'a>(
    fbb: &mut FlatBufferBuilder<'a>,
    pins: &[LagrangePin],
) -> flatbuffers::WIPOffset<
    flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<fb::LagrangePin<'a>>>,
> {
    let offsets: Vec<_> = pins
        .iter()
        .map(|pin| serialize_lagrange_pin(fbb, pin))
        .collect();

    fbb.create_vector(&offsets)
}

pub fn deserialize_pins(
    fb_pins: flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<fb::LagrangePin<'_>>>,
) -> Result<Vec<LagrangePin>> {
    let mut out = Vec::with_capacity(fb_pins.len());
    for i in 0..fb_pins.len() {
        out.push(deserialize_lagrange_pin(fb_pins.get(i))?);
    }

    Ok(out)
}
