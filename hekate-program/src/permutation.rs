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

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use hekate_core::errors;

/// Challenge label for Fiat-Shamir transcript.
pub type ChallengeLabel = &'static [u8];

/// Shared label for the `request_idx` clock
/// source on stateless-service buses; same
/// label on both endpoints is load-bearing
/// for `β`-mix alignment.
pub const REQUEST_IDX_LABEL: ChallengeLabel = b"kappa_request_idx";

/// LogUp bus semantics.
///
/// Cross-bus check sums `claimed_sum` over all
/// endpoints sharing a `bus_id` and rejects if
/// the total is non-zero in `GF(2^128)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BusKind {
    #[default]
    Permutation,
    Lookup,
}

/// Byte source for the LogUp
/// key `Σ β^j · source_j(i)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// Single byte column from the trace.
    Column(usize),

    /// Multiple byte columns stitched into one
    /// key segment via the global `β` schedule.
    Columns(Vec<usize>),

    /// Virtual clock derived from the row index;
    /// `num_bytes` controls the byte width.
    /// Required for `Permutation`-kind buses to
    /// pin per-row uniqueness and prevent char-2
    /// even-multiplicity parity collapse.
    RowIndexLeBytes(usize),

    /// Constant byte for cross-table
    /// domain separation.
    Const(u128),

    /// Virtual single byte `k` of the row index.
    RowIndexByte(usize),
}

/// One endpoint of a LogUp bus.
///
/// Per row `i`:
/// `h(i) = s(i) / (γ + Σ β^j · source_j(i))`.
/// The endpoint contributes `claimed_sum = Σ_i h(i)`
/// (Permutation) or `Σ_i Eq(r_bus, i) · h(i)` (Lookup)
/// to the cross-bus check.
///
/// # Security
/// `Permutation` kind cancels in char-2 by
/// multiset parity, so endpoints with even
/// per-key multiplicity collapse silently.
/// Use `Source::RowIndexLeBytes` in the key
/// to force per-row uniqueness, or switch to
/// `BusKind::Lookup` for positional binding.
#[derive(Clone, Debug)]
pub struct PermutationCheckSpec {
    pub kind: BusKind,

    /// Key sources stitched via the global
    /// `β` schedule using each label.
    pub sources: Vec<(Source, ChallengeLabel)>,

    /// Selector column gating the row's `h`.
    /// `None` means the row is unconditionally active.
    pub selector: Option<usize>,

    /// Receive-side selector for paired buses.
    /// AIR must enforce `s_send · s_recv = 0` (char-2 mutex).
    pub recv_selector: Option<usize>,

    /// Audited carve-out citation for `Permutation`
    /// specs that intentionally omit a row-index source.
    pub clock_waiver: Option<String>,
}

impl PermutationCheckSpec {
    pub fn new(sources: Vec<(Source, ChallengeLabel)>, selector: Option<usize>) -> Self {
        Self {
            sources,
            selector,
            recv_selector: None,
            kind: BusKind::Permutation,
            clock_waiver: None,
        }
    }

    /// Endpoints on a `Lookup` bus must be
    /// pointwise-equal on the padded hypercube;
    /// use only when positional binding holds.
    pub fn new_lookup(sources: Vec<(Source, ChallengeLabel)>, selector: Option<usize>) -> Self {
        Self {
            sources,
            selector,
            recv_selector: None,
            kind: BusKind::Lookup,
            clock_waiver: None,
        }
    }

    /// Caller must enforce `s_send · s_recv = 0` in the AIR;
    /// without it, rows with both selectors high collapse to
    /// zero in char-2 and slip past cross-bus cancellation.
    pub fn new_paired(
        sources: Vec<(Source, ChallengeLabel)>,
        s_send: usize,
        s_recv: usize,
        kind: BusKind,
    ) -> Self {
        Self {
            sources,
            selector: Some(s_send),
            recv_selector: Some(s_recv),
            kind,
            clock_waiver: None,
        }
    }

    /// Audited escape hatch. `reason` must start
    /// with `"see "` and cite the load-bearing AIR
    /// constraint (`see <path>:<line>: <argument>`).
    pub fn with_clock_waiver(mut self, reason: impl Into<String>) -> Self {
        self.clock_waiver = Some(reason.into());
        self
    }

    pub fn num_sources(&self) -> usize {
        self.sources.len()
    }

    pub fn has_selector(&self) -> bool {
        self.selector.is_some()
    }

    pub fn has_paired(&self) -> bool {
        self.recv_selector.is_some()
    }

    /// Reindexes column references when the
    /// chiplet is embedded into a wider trace.
    pub fn shift_column_indices(&mut self, offset: usize) {
        for (source, _) in &mut self.sources {
            match source {
                Source::Column(idx) => *idx += offset,
                Source::Columns(indices) => {
                    for idx in indices {
                        *idx += offset;
                    }
                }
                _ => {}
            }
        }

        if let Some(sel_idx) = &mut self.selector {
            *sel_idx += offset;
        }

        if let Some(sel_idx) = &mut self.recv_selector {
            *sel_idx += offset;
        }
    }

    /// Only structural guarantor of per-row uniqueness;
    /// label-only stitching is forgeable.
    pub fn has_real_clock_source(&self) -> bool {
        self.sources
            .iter()
            .any(|(src, _)| matches!(src, Source::RowIndexLeBytes(_) | Source::RowIndexByte(_)))
    }

    pub fn has_request_idx_column(&self) -> bool {
        self.sources
            .iter()
            .any(|(src, label)| matches!(src, Source::Column(_)) && *label == REQUEST_IDX_LABEL)
    }

    /// Per-spec only. `validate_bus_set` runs the
    /// cross-endpoint check that closes label spoofing.
    pub fn validate_clock_stitching(&self, _bus_id: &str) -> errors::Result<()> {
        let waiver_status = self.clock_waiver.as_deref().map(WaiverStatus::classify);

        let has_clock_marker = self.has_real_clock_source() || self.has_request_idx_column();

        match (self.kind, has_clock_marker, waiver_status) {
            (BusKind::Lookup, _, Some(_)) => Err(errors::Error::Protocol {
                protocol: "logup_bus",
                message: "lookup bus carries a clock_waiver; waivers only apply \
                          to Permutation kind, drop the .with_clock_waiver(...) call",
            }),
            (BusKind::Permutation, true, Some(_)) => Err(errors::Error::Protocol {
                protocol: "logup_bus",
                message: "permutation bus carries both a clock source and a \
                          clock_waiver; pick one shape",
            }),
            (BusKind::Permutation, false, Some(WaiverStatus::Empty)) => {
                Err(errors::Error::Protocol {
                    protocol: "logup_bus",
                    message: "permutation bus has an empty clock_waiver; provide a \
                              non-empty reason citing the load-bearing AIR constraint",
                })
            }
            (BusKind::Permutation, false, Some(WaiverStatus::TooShort)) => {
                Err(errors::Error::Protocol {
                    protocol: "logup_bus",
                    message: "permutation bus has an under-specified clock_waiver; \
                              the reason must be at least 32 chars and cite a file/line \
                              of the load-bearing AIR constraint",
                })
            }
            (BusKind::Permutation, false, Some(WaiverStatus::MissingCitation)) => {
                Err(errors::Error::Protocol {
                    protocol: "logup_bus",
                    message: "permutation bus clock_waiver lacks a 'see <path>' citation; \
                              waiver text must start with 'see ' followed by the file path \
                              of the load-bearing AIR constraint",
                })
            }
            (BusKind::Permutation, false, None) => Err(errors::Error::Protocol {
                protocol: "logup_bus",
                message: "permutation bus lacks per-row clock stitching; add \
                          Source::RowIndexLeBytes, pair both endpoints with a \
                          committed B32 column labelled REQUEST_IDX_LABEL whose \
                          value matches the partner row index, switch to \
                          BusKind::Lookup via new_lookup, or document the \
                          carve-out via .with_clock_waiver(reason)",
            }),
            _ => Ok(()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WaiverStatus {
    Empty,
    TooShort,
    MissingCitation,
    Ok,
}

impl WaiverStatus {
    const MIN_WAIVER_LEN: usize = 32;
    const REQUIRED_PREFIX: &'static str = "see ";

    fn classify(s: &str) -> Self {
        if s.is_empty() {
            Self::Empty
        } else if s.len() < Self::MIN_WAIVER_LEN {
            Self::TooShort
        } else if !s.starts_with(Self::REQUIRED_PREFIX) {
            Self::MissingCitation
        } else {
            Self::Ok
        }
    }
}

/// Every multi-endpoint `Permutation` `bus_id`
/// must have at least one endpoint owning a real
/// `RowIndexLeBytes`/`RowIndexByte` clock;
/// otherwise label-only stitching admits
/// char-2 parity collapse.
pub fn validate_bus_set<'a, I>(endpoints: I) -> errors::Result<()>
where
    I: IntoIterator<Item = (&'a str, &'a PermutationCheckSpec)>,
{
    let mut by_bus: BTreeMap<&'a str, Vec<&'a PermutationCheckSpec>> = BTreeMap::new();

    for (bus_id, spec) in endpoints {
        if !bus_id.bytes().all(|b| b.is_ascii_graphic()) {
            return Err(errors::Error::Protocol {
                protocol: "logup_bus",
                message: "bus_id has a non-graphic byte; bus ids must be \
                          graphic ASCII (0x21..=0x7E). A space, zero-width, \
                          or homoglyph byte forges a visually identical \
                          second bus that balances on its own",
            });
        }

        by_bus.entry(bus_id).or_default().push(spec);
    }

    for (bus_id, specs) in &by_bus {
        let any_lookup = specs.iter().any(|s| s.kind == BusKind::Lookup);
        let any_perm = specs.iter().any(|s| s.kind == BusKind::Permutation);

        if any_lookup && any_perm {
            return Err(errors::Error::Protocol {
                protocol: "logup_bus",
                message: "bus_id has mixed BusKind across endpoints; \
                          all endpoints must agree on Permutation or Lookup",
            });
        }

        if any_lookup {
            continue;
        }

        if specs.len() < 2 {
            continue;
        }

        let any_real_clock = specs.iter().any(|s| s.has_real_clock_source());
        let all_waivered = specs.iter().all(|s| s.clock_waiver.is_some());

        if !any_real_clock && !all_waivered {
            let _ = bus_id;
            return Err(errors::Error::Protocol {
                protocol: "logup_bus",
                message: "permutation bus_id has no endpoint owning a real \
                          Source::RowIndexLeBytes/RowIndexByte clock and not all \
                          endpoints declare a clock_waiver; label-only stitching \
                          is forgeable and admits char-2 parity collapse",
            });
        }
    }

    Ok(())
}

/// Folds `table_rows` into `heights` for each
/// `Lookup`-kind spec, taking the running max.
/// Used to derive the per-bus `N_max` absorbed
/// into the transcript before `r_bus` is drawn.
pub fn accumulate_lookup_heights(
    specs: &[(String, PermutationCheckSpec)],
    table_rows: u64,
    heights: &mut BTreeMap<String, u64>,
) {
    for (bus_id, spec) in specs {
        if spec.kind == BusKind::Lookup {
            let entry = heights.entry(bus_id.clone()).or_insert(0);
            *entry = (*entry).max(table_rows);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutation_spec_creation() {
        let sources = vec![
            (Source::Column(0), b"kappa_0" as ChallengeLabel),
            (Source::Column(1), b"kappa_1" as ChallengeLabel),
            (Source::RowIndexLeBytes(4), b"kappa_clk" as ChallengeLabel),
        ];

        let spec = PermutationCheckSpec::new(sources, Some(2));

        assert_eq!(spec.num_sources(), 3);
        assert!(spec.has_selector());
        assert_eq!(spec.selector, Some(2));
    }

    #[test]
    fn source_variants() {
        let col = Source::Column(5);
        let cols = Source::Columns(vec![0, 1, 2, 3]);
        let clock = Source::RowIndexLeBytes(4);
        let constant = Source::Const(0x01);

        assert_eq!(col, Source::Column(5));
        assert_eq!(cols, Source::Columns(vec![0, 1, 2, 3]));
        assert_eq!(clock, Source::RowIndexLeBytes(4));
        assert_eq!(constant, Source::Const(0x01));
    }

    #[test]
    fn new_paired_populates_both_selectors() {
        let spec = PermutationCheckSpec::new_paired(
            vec![(Source::Column(0), b"k_a" as ChallengeLabel)],
            3,
            5,
            BusKind::Permutation,
        );

        assert!(spec.has_selector());
        assert!(spec.has_paired());
        assert_eq!(spec.selector, Some(3));
        assert_eq!(spec.recv_selector, Some(5));
        assert_eq!(spec.kind, BusKind::Permutation);
    }

    #[test]
    fn new_defaults_recv_selector_none() {
        let spec =
            PermutationCheckSpec::new(vec![(Source::Column(0), b"k_a" as ChallengeLabel)], Some(1));

        assert!(!spec.has_paired());
        assert_eq!(spec.recv_selector, None);
    }

    #[test]
    fn new_lookup_defaults_recv_selector_none() {
        let spec = PermutationCheckSpec::new_lookup(
            vec![(Source::Column(0), b"k_a" as ChallengeLabel)],
            Some(1),
        );

        assert!(!spec.has_paired());
        assert_eq!(spec.recv_selector, None);
    }

    #[test]
    fn shift_column_indices_covers_recv_selector() {
        let mut spec = PermutationCheckSpec::new_paired(
            vec![
                (Source::Column(0), b"k_a" as ChallengeLabel),
                (Source::Columns(vec![1, 2]), b"k_b" as ChallengeLabel),
            ],
            3,
            5,
            BusKind::Lookup,
        );

        spec.shift_column_indices(10);

        assert_eq!(spec.selector, Some(13));
        assert_eq!(spec.recv_selector, Some(15));

        match &spec.sources[0].0 {
            Source::Column(idx) => assert_eq!(*idx, 10),
            other => panic!("expected Column, got {other:?}"),
        }

        match &spec.sources[1].0 {
            Source::Columns(idxs) => assert_eq!(idxs, &vec![11, 12]),
            other => panic!("expected Columns, got {other:?}"),
        }
    }

    #[test]
    fn validate_bus_set_rejects_homoglyph_split_bus() {
        let clockless =
            PermutationCheckSpec::new(vec![(Source::Column(0), b"k" as ChallengeLabel)], Some(1));
        let other = clockless.clone();

        assert!(validate_bus_set(vec![("ram_link", &clockless), ("ram_link", &other)]).is_err());

        // A zero-width twin splits this into two
        // single-endpoint buses that skip the
        // clock gate; the ASCII check blocks it.
        assert!(
            validate_bus_set(vec![("ram_link", &clockless), ("ram_link\u{200b}", &other)]).is_err()
        );
    }

    #[test]
    fn validate_bus_set_accepts_ascii_bus_id() {
        let spec =
            PermutationCheckSpec::new(vec![(Source::Column(0), b"k" as ChallengeLabel)], Some(1));

        assert!(validate_bus_set(vec![("ram_link", &spec)]).is_ok());
    }
}
