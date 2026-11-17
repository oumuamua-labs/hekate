// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

//! Test oracles for Hekate programs: honest statements must
//! verify and forged traces must fail for the expected reason.

pub mod cell;
pub mod fault;
pub mod verdict;
