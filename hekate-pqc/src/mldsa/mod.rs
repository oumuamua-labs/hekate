// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! ML-DSA (FIPS 204): parameter sets and
//! the tables that prove Verify_internal.

mod params;
mod pipeline;

pub use params::{MlDsaParams, Q};
pub use pipeline::{
    Forgery, MLDSA_DATA_BUS_ID, MlDsaChiplet, MlDsaInput, MlDsaOutput, MlDsaWitness, service,
};
