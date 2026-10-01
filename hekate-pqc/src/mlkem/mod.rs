// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! ML-KEM (FIPS 203): parameter sets and the
//! tables that prove KeyGen, Encaps and Decaps.

mod params;
mod pipeline;
mod reference;

pub use params::{MlKemParams, Q};
pub use pipeline::{
    Forgery, MLKEM_DATA_BUS_ID, MlKemCall, MlKemChiplet, MlKemInput, MlKemOutput, MlKemWitness,
    service,
};

pub(crate) use params::{compress, decompress, div_q};
