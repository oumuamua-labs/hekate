// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use hekate_core::errors;
use hekate_math::{Flat, HardwareField, TowerField};

use crate::FixedColumn;

mod rank;

pub use rank::{
    OMEGA, OMEGA_ORDER, RankClock, RankTable, TableHeight, rank_clocks, validate_ordered_buses,
    validate_ordered_emits,
};

/// Slot identity;
/// position fixes the `β` power, never the label.
pub type ChallengeLabel = &'static [u8];

/// Clock slot of an ordered service bus, on both endpoints.
pub const EMIT_RANK_LABEL: ChallengeLabel = b"kappa_emit_rank";

/// `eval_row_idx_le_mle` folds at most eight
/// bytes; a wider clock truncates silently.
pub const MAX_ROW_INDEX_BYTES: usize = 8;

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

/// The half of an ordered bus an endpoint belongs to;
/// the `t`-th request pairs with the `t`-th response.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Side {
    Request,
    Response,
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
    /// `num_bytes` controls the byte width. Pins per-row
    /// uniqueness against char-2 parity collapse.
    RowIndexLeBytes(usize),

    /// Constant byte for cross-table domain separation.
    Const(u128),

    /// Virtual single byte `k` of the row index.
    RowIndexByte(usize),

    /// Column replacing the top byte of
    /// a requester's row-index clock;
    /// must be an overlay `fixed_columns()` pin.
    PhaseColumn(usize),

    /// `ω^(base + t)` on the endpoint's `t`-th emit,
    /// computed by the verifier from the pinned selector.
    EmitRank(Side),
}

/// One endpoint of a LogUp bus.
///
/// Per row `i`:
/// `h(i) = s(i) / (γ + Σ β^j · source_j(i))`.
/// The endpoint contributes `claimed_sum = Σ_i h(i)` (Permutation)
/// or `Σ_i Eq(r_bus, i) · h(i)` (Lookup) to the cross-bus check.
///
/// # Security
/// `Permutation` kind cancels in char-2 by multiset parity;
/// endpoints with even per-key multiplicity collapse silently.
/// Use `Source::EmitRank` or a row-index source in the key to force
/// per-row uniqueness, or switch to `BusKind::Lookup` for positional binding.
#[derive(Clone, Debug)]
pub struct PermutationCheckSpec {
    pub kind: BusKind,

    /// Key sources;
    /// slot order fixes the `β` schedule.
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
    /// constraint (`see <path>: <argument>`).
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
                Source::Column(idx) | Source::PhaseColumn(idx) => *idx += offset,
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
        self.sources.iter().any(|(src, _)| {
            matches!(
                src,
                Source::RowIndexLeBytes(_) | Source::RowIndexByte(_) | Source::EmitRank(_)
            )
        })
    }

    /// A Permutation spec needs a row-index or rank source or
    /// a well-formed waiver, never both; a Lookup spec takes no waiver.
    pub fn validate_clock_stitching(&self, _bus_id: &str) -> errors::Result<()> {
        for (src, _) in &self.sources {
            let width_ok = match src {
                Source::RowIndexLeBytes(n) => *n >= 1 && *n <= MAX_ROW_INDEX_BYTES,
                Source::RowIndexByte(k) => *k < MAX_ROW_INDEX_BYTES,
                Source::Column(_)
                | Source::Columns(_)
                | Source::Const(_)
                | Source::PhaseColumn(_)
                | Source::EmitRank(_) => true,
            };

            if !width_ok {
                return Err(errors::Error::Protocol {
                    protocol: "logup_bus",
                    message: "row-index clock wider than 8 bytes; the row-index \
                              MLE folds at most 8 and truncates the rest silently",
                });
            }
        }

        let waiver_status = self.clock_waiver.as_deref().map(WaiverStatus::classify);

        match (self.kind, self.has_real_clock_source(), waiver_status) {
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
                message: "permutation bus lacks per-row clock stitching; give the \
                          Service an EmitRank slot, give a raw spec a RowIndexLeBytes \
                          or RowIndexByte source, or add a clock_waiver citing the AIR \
                          constraint that forces per-row uniqueness",
            }),
            _ => Ok(()),
        }
    }
}

/// One key slot; its position fixes the `β` power.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServiceSlot {
    /// Bound to a table column on each endpoint.
    Value(ChallengeLabel),

    /// Same constant on both endpoints
    /// (cross-table domain separation).
    Const(ChallengeLabel, u128),

    /// Ordered-bus clock: the verifier supplies `ω^(base + t)`
    /// on both endpoints, with no committed column on either side.
    EmitRank,
}

/// Bus key schema declared once by the serving
/// chiplet; both endpoints derive their spec from it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Service {
    pub bus_id: &'static str,
    pub kind: BusKind,
    pub slots: Vec<ServiceSlot>,
}

impl Service {
    pub fn num_values(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| matches!(s, ServiceSlot::Value(_)))
            .count()
    }

    /// Requester endpoint: `values` bind the `Value` slots
    /// in order; the `EmitRank` slot takes the request side.
    pub fn request(
        &self,
        values: &[usize],
        selector: usize,
    ) -> errors::Result<PermutationCheckSpec> {
        self.spec(values, Side::Request, selector)
    }

    /// Responder endpoint: `values` bind the `Value` slots
    /// in order; the `EmitRank` slot takes the response side.
    pub fn respond(
        &self,
        values: &[usize],
        selector: usize,
    ) -> errors::Result<PermutationCheckSpec> {
        self.spec(values, Side::Response, selector)
    }

    fn spec(
        &self,
        values: &[usize],
        side: Side,
        selector: usize,
    ) -> errors::Result<PermutationCheckSpec> {
        let rank_slots = self
            .slots
            .iter()
            .filter(|s| matches!(s, ServiceSlot::EmitRank))
            .count();

        let clock_ok = matches!(
            (self.kind, rank_slots),
            (BusKind::Permutation, 1) | (BusKind::Lookup, 0)
        );

        if !clock_ok {
            return Err(errors::Error::Protocol {
                protocol: "service",
                message: "Permutation schema needs exactly one EmitRank slot; \
                          Lookup schema must have none",
            });
        }

        if values.len() != self.num_values() {
            return Err(errors::Error::Protocol {
                protocol: "service",
                message: "value column count does not match the schema's Value slots",
            });
        }

        let mut sources = Vec::with_capacity(self.slots.len());
        let mut next_value = values.iter();

        for slot in &self.slots {
            match slot {
                ServiceSlot::Value(label) => {
                    let col = next_value.next().ok_or(errors::Error::Protocol {
                        protocol: "service",
                        message: "value column count does not match the schema's Value slots",
                    })?;

                    sources.push((Source::Column(*col), *label));
                }
                ServiceSlot::Const(label, v) => {
                    sources.push((Source::Const(*v), *label));
                }
                ServiceSlot::EmitRank => {
                    sources.push((Source::EmitRank(side), EMIT_RANK_LABEL));
                }
            }
        }

        Ok(match self.kind {
            BusKind::Permutation => PermutationCheckSpec::new(sources, Some(selector)),
            BusKind::Lookup => PermutationCheckSpec::new_lookup(sources, Some(selector)),
        })
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

/// A witness selector gives the prover freedom over which
/// rows join the bus; every selector must be `None` or an
/// overlay-pinned member of the table's `fixed_columns()`.
pub fn validate_fixed_selectors<F>(
    specs: &[(String, PermutationCheckSpec)],
    fixed: &[FixedColumn<F>],
) -> errors::Result<()> {
    for (_, spec) in specs {
        if spec.selector.is_none() && spec.recv_selector.is_some() {
            return Err(errors::Error::Protocol {
                protocol: "logup_bus",
                message: "paired bus has recv_selector without send selector",
            });
        }

        let phases = spec.sources.iter().filter_map(|(src, _)| match src {
            Source::PhaseColumn(col) => Some(*col),
            _ => None,
        });

        for sel in [spec.selector, spec.recv_selector]
            .into_iter()
            .flatten()
            .chain(phases)
        {
            let Some(fc) = fixed.iter().find(|fc| fc.col_idx == sel) else {
                return Err(errors::Error::Protocol {
                    protocol: "logup_bus",
                    message: "bus selector or clock phase reads a witness column; \
                              every selector, recv_selector and PhaseColumn must be \
                              None or a member of the table's fixed_columns",
                });
            };

            if !fc.shape.is_overlay() {
                return Err(errors::Error::Protocol {
                    protocol: "logup_bus",
                    message: "bus selector or clock phase is pinned to a substituted \
                              shape; FirstRow/LastRow/Custom replace the committed \
                              column with a virtual poly the bus cannot read; pin \
                              with Cadence, Segments, Periodic, Sparse or Dense",
                });
            }
        }
    }

    Ok(())
}

/// Rejects a `bus_id` with a non-graphic byte, or
/// one whose endpoints mix `Permutation` and `Lookup`.
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

    for specs in by_bus.values() {
        let any_lookup = specs.iter().any(|s| s.kind == BusKind::Lookup);
        let any_perm = specs.iter().any(|s| s.kind == BusKind::Permutation);

        if any_lookup && any_perm {
            return Err(errors::Error::Protocol {
                protocol: "logup_bus",
                message: "bus_id has mixed BusKind across endpoints; \
                          all endpoints must agree on Permutation or Lookup",
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

/// MLE of `Source::RowIndexLeBytes` at `r_final`. Linear
/// in `r_final` because `F::from` is XOR-additive over char-2.
pub fn eval_row_idx_le_mle<F>(num_bytes: usize, r_final: &[Flat<F>]) -> Flat<F>
where
    F: TowerField + HardwareField + From<u128>,
{
    let total_bits = (num_bytes.min(8) * 8).min(r_final.len());

    let mut acc = Flat::from_raw(F::ZERO);
    for (i, r) in r_final.iter().enumerate().take(total_bits) {
        acc += F::from(1u128 << i).to_hardware() * *r;
    }

    acc
}

/// MLE of `Source::RowIndexByte` at `r_final`. Same char-2
/// shortcut as `eval_row_idx_le_mle`, restricted to one byte.
pub fn eval_row_idx_byte_mle<F>(byte_idx: usize, r_final: &[Flat<F>]) -> Flat<F>
where
    F: TowerField + HardwareField + From<u128>,
{
    let bit_start = byte_idx.saturating_mul(8);
    if bit_start >= r_final.len() {
        return Flat::from_raw(F::ZERO);
    }

    let end = (bit_start + 8).min(r_final.len());

    let mut acc = Flat::from_raw(F::ZERO);
    for (j, i) in (bit_start..end).enumerate() {
        acc += F::from(1u128 << j).to_hardware() * r_final[i];
    }

    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FixedShape;
    use hekate_math::Block128;

    fn test_service() -> Service {
        Service {
            bus_id: "svc",
            kind: BusKind::Permutation,
            slots: vec![
                ServiceSlot::Value(b"k_a"),
                ServiceSlot::Const(b"k_dom", 7),
                ServiceSlot::EmitRank,
                ServiceSlot::Value(b"k_dir"),
            ],
        }
    }

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

        // A zero-width twin splits this into
        // two buses; the ASCII check blocks it.
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

    #[test]
    fn service_endpoints_share_schema() {
        let svc = test_service();

        let req = svc.request(&[10, 11], 12).unwrap();
        let resp = svc.respond(&[20, 21], 23).unwrap();

        let req_labels: Vec<_> = req.sources.iter().map(|(_, l)| *l).collect();
        let resp_labels: Vec<_> = resp.sources.iter().map(|(_, l)| *l).collect();

        assert_eq!(req_labels, resp_labels);
        assert_eq!(req_labels[2], EMIT_RANK_LABEL);

        assert_eq!(req.sources[0].0, Source::Column(10));
        assert_eq!(req.sources[1].0, Source::Const(7));
        assert_eq!(req.sources[2].0, Source::EmitRank(Side::Request));
        assert_eq!(req.sources[3].0, Source::Column(11));
        assert_eq!(req.selector, Some(12));

        assert_eq!(resp.sources[0].0, Source::Column(20));
        assert_eq!(resp.sources[1].0, Source::Const(7));
        assert_eq!(resp.sources[2].0, Source::EmitRank(Side::Response));
        assert_eq!(resp.sources[3].0, Source::Column(21));
        assert_eq!(resp.selector, Some(23));

        req.validate_clock_stitching("svc").unwrap();
        resp.validate_clock_stitching("svc").unwrap();

        validate_bus_set(vec![("svc", &req), ("svc", &resp)]).unwrap();
    }

    #[test]
    fn service_rejects_schema_mismatches() {
        let svc = test_service();

        assert!(svc.request(&[10], 12).is_err());
        assert!(svc.respond(&[20, 21, 22], 23).is_err());

        let lookup = Service {
            bus_id: "rom",
            kind: BusKind::Lookup,
            slots: vec![ServiceSlot::Value(b"k_a"), ServiceSlot::EmitRank],
        };

        assert!(lookup.respond(&[3], 4).is_err());

        let clockless = Service {
            bus_id: "bad",
            kind: BusKind::Permutation,
            slots: vec![ServiceSlot::Value(b"k_a")],
        };

        assert!(clockless.request(&[1], 2).is_err());

        let two_ranks = Service {
            bus_id: "bad",
            kind: BusKind::Permutation,
            slots: vec![
                ServiceSlot::EmitRank,
                ServiceSlot::Value(b"k_a"),
                ServiceSlot::EmitRank,
            ],
        };

        assert!(two_ranks.request(&[1], 2).is_err());
    }

    #[test]
    fn lookup_service_derives_lookup_specs() {
        let svc = Service {
            bus_id: "rom",
            kind: BusKind::Lookup,
            slots: vec![ServiceSlot::Value(b"k_a")],
        };

        let req = svc.request(&[1], 2).unwrap();
        let resp = svc.respond(&[3], 4).unwrap();

        assert_eq!(req.kind, BusKind::Lookup);
        assert_eq!(resp.kind, BusKind::Lookup);
        assert_eq!(req.sources.len(), 1);
    }

    #[test]
    fn column_clock_needs_waiver_whatever_its_label() {
        let spoofed = PermutationCheckSpec::new(
            vec![
                (Source::Column(0), b"k_a" as ChallengeLabel),
                (Source::Column(1), EMIT_RANK_LABEL),
            ],
            Some(2),
        );

        assert!(spoofed.validate_clock_stitching("svc").is_err());

        spoofed
            .with_clock_waiver(
                "see permutation/mod.rs: a body constraint forces per-row uniqueness",
            )
            .validate_clock_stitching("svc")
            .unwrap();
    }

    #[test]
    fn witness_clock_phase_is_rejected() {
        let phased = PermutationCheckSpec::new(
            vec![
                (Source::Column(10), b"k_a" as ChallengeLabel),
                (Source::RowIndexByte(0), b"k_clk_b0" as ChallengeLabel),
                (Source::PhaseColumn(60), b"k_clk_b1" as ChallengeLabel),
            ],
            Some(12),
        );
        let specs = vec![(String::from("svc"), phased)];

        let selector_only = vec![FixedColumn::<Block128> {
            col_idx: 12,
            shape: FixedShape::Periodic {
                period: 1,
                values: vec![Block128::ONE],
            },
        }];

        assert!(validate_fixed_selectors(&specs, &selector_only).is_err());

        let mut pinned = selector_only;
        pinned.push(FixedColumn {
            col_idx: 60,
            shape: FixedShape::Periodic {
                period: 2,
                values: vec![Block128::ZERO, Block128::ONE],
            },
        });

        validate_fixed_selectors(&specs, &pinned).unwrap();

        let mut substituted = pinned;
        substituted[1].shape = FixedShape::LastRow;

        assert!(validate_fixed_selectors(&specs, &substituted).is_err());
    }
}
