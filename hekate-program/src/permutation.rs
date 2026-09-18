// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use hekate_core::errors;
use hekate_math::{Flat, HardwareField, TowerField};

use crate::FixedColumn;

/// Slot identity;
/// position fixes the `β` power, never the label.
pub type ChallengeLabel = &'static [u8];

/// Clock slot of a stateless-service bus, on both endpoints.
pub const REQUEST_IDX_LABEL: ChallengeLabel = b"kappa_request_idx";

/// One label per byte of a byte-split
/// clock, in little-endian order.
pub const REQUEST_IDX_BYTE_LABELS: [ChallengeLabel; 4] = [
    b"kappa_request_idx_b0",
    b"kappa_request_idx_b1",
    b"kappa_request_idx_b2",
    b"kappa_request_idx_b3",
];

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

    /// Constant byte for cross-table domain separation.
    Const(u128),

    /// Virtual single byte `k` of the row index.
    RowIndexByte(usize),

    /// Column replacing the top byte of
    /// a requester's row-index clock;
    /// must be an overlay `fixed_columns()` pin.
    PhaseColumn(usize),
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
/// Use `Source::RowIndexLeBytes` in the key to force per-row
/// uniqueness, or switch to `BusKind::Lookup` for positional binding.
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
        self.sources
            .iter()
            .any(|(src, _)| matches!(src, Source::RowIndexLeBytes(_) | Source::RowIndexByte(_)))
    }

    pub fn has_request_idx_column(&self) -> bool {
        self.sources
            .iter()
            .any(|(src, label)| matches!(src, Source::Column(_)) && is_request_idx_label(label))
    }

    /// Per-spec only. `validate_bus_set` runs the
    /// cross-endpoint check that closes label spoofing.
    pub fn validate_clock_stitching(&self, _bus_id: &str) -> errors::Result<()> {
        for (src, _) in &self.sources {
            let width_ok = match src {
                Source::RowIndexLeBytes(n) => *n >= 1 && *n <= MAX_ROW_INDEX_BYTES,
                Source::RowIndexByte(k) => *k < MAX_ROW_INDEX_BYTES,
                Source::Column(_)
                | Source::Columns(_)
                | Source::Const(_)
                | Source::PhaseColumn(_) => true,
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
                message: "permutation bus lacks per-row clock stitching; give the \
                          Service a RequestIdx or RequestIdxBytes slot, or a \
                          clock_waiver citing the AIR constraint that forces \
                          per-row uniqueness",
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

    /// Stateless-service clock: the requester folds
    /// its own row index, the responder commits a
    /// column carrying the requester's row.
    RequestIdx { num_bytes: usize },

    /// The same clock split one slot per byte, for
    /// a responder whose AIR reads the bytes separately.
    RequestIdxBytes { num_bytes: usize },
}

/// Bus key schema declared once by the serving
/// chiplet; both endpoints derive their spec from it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Service {
    pub bus_id: &'static str,
    pub kind: BusKind,
    pub slots: Vec<ServiceSlot>,

    /// Per-row uniqueness argued by a body
    /// constraint rather than a clock slot.
    pub clock_waiver: Option<&'static str>,
}

impl Service {
    pub fn num_values(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| matches!(s, ServiceSlot::Value(_)))
            .count()
    }

    /// Committed columns a responder must
    /// supply for the schema's clock slot.
    pub fn clock_columns(&self) -> usize {
        self.slots
            .iter()
            .map(|s| match s {
                ServiceSlot::RequestIdx { .. } => 1,
                ServiceSlot::RequestIdxBytes { num_bytes } => *num_bytes,
                _ => 0,
            })
            .sum()
    }

    /// Requester endpoint: `values` bind the `Value` slots in
    /// order; a clock slot folds the requester's own row index.
    pub fn request(
        &self,
        values: &[usize],
        selector: usize,
    ) -> errors::Result<PermutationCheckSpec> {
        self.spec(values, None, None, selector)
    }

    /// Requester spec whose clock replaces the row index's
    /// top byte with `phase_col`, leaving `8 · (num_bytes - 1)`
    /// row-index bits. Unchecked: a phase another requester
    /// on this bus repeats annihilates both emits.
    pub fn request_phased(
        &self,
        values: &[usize],
        phase_col: usize,
        selector: usize,
    ) -> errors::Result<PermutationCheckSpec> {
        self.spec(values, None, Some(phase_col), selector)
    }

    /// Responder endpoint: `clock_cols` are the committed
    /// columns mirroring the requester's clock sources,
    /// one per `clock_columns()`.
    pub fn respond(
        &self,
        values: &[usize],
        clock_cols: &[usize],
        selector: usize,
    ) -> errors::Result<PermutationCheckSpec> {
        self.spec(values, Some(clock_cols), None, selector)
    }

    fn spec(
        &self,
        values: &[usize],
        respond_idx: Option<&[usize]>,
        phase: Option<usize>,
        selector: usize,
    ) -> errors::Result<PermutationCheckSpec> {
        let clock_slots = self
            .slots
            .iter()
            .filter(|s| {
                matches!(
                    s,
                    ServiceSlot::RequestIdx { .. } | ServiceSlot::RequestIdxBytes { .. }
                )
            })
            .count();

        let clock_ok = matches!(
            (self.kind, clock_slots, self.clock_waiver),
            (BusKind::Permutation, 1, None)
                | (BusKind::Permutation, 0, Some(_))
                | (BusKind::Lookup, 0, None)
        );

        if !clock_ok {
            return Err(errors::Error::Protocol {
                protocol: "service",
                message: "Permutation schema needs exactly one clock slot or a \
                          clock_waiver, never both; Lookup schema must have neither",
            });
        }

        if phase.is_some() {
            let clock_bytes = self.slots.iter().find_map(|s| match s {
                ServiceSlot::RequestIdxBytes { num_bytes } => Some(*num_bytes),
                _ => None,
            });

            match clock_bytes {
                None => {
                    return Err(errors::Error::Protocol {
                        protocol: "service",
                        message: "a phased clock needs a byte-split clock slot; \
                                  a whole-word, waived or lookup schema has \
                                  no top byte to replace",
                    });
                }
                Some(num_bytes) if num_bytes < 2 => {
                    return Err(errors::Error::Protocol {
                        protocol: "service",
                        message: "a phased clock needs at least two clock bytes",
                    });
                }
                Some(_) => {}
            }
        }

        if let Some(cols) = respond_idx
            && cols.len() != self.clock_columns()
        {
            return Err(errors::Error::Protocol {
                protocol: "service",
                message: "clock column count does not match the schema's clock slot",
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
        let mut next_clock = respond_idx.unwrap_or(&[]).iter();

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
                ServiceSlot::RequestIdx { num_bytes } => {
                    if *num_bytes == 0 || *num_bytes > MAX_ROW_INDEX_BYTES {
                        return Err(errors::Error::Protocol {
                            protocol: "service",
                            message: "RequestIdx num_bytes must be 1..=8; a wider \
                                      clock is silently truncated by the row-index MLE",
                        });
                    }

                    let source = match respond_idx {
                        None => Source::RowIndexLeBytes(*num_bytes),
                        Some(_) => {
                            Source::Column(*next_clock.next().ok_or(errors::Error::Protocol {
                                protocol: "service",
                                message: "RequestIdx binds exactly one committed \
                                          clock column on the responder",
                            })?)
                        }
                    };

                    sources.push((source, REQUEST_IDX_LABEL));
                }
                ServiceSlot::RequestIdxBytes { num_bytes } => {
                    if *num_bytes == 0 || *num_bytes > REQUEST_IDX_BYTE_LABELS.len() {
                        return Err(errors::Error::Protocol {
                            protocol: "service",
                            message: "byte-split clock wider than the label family",
                        });
                    }

                    for (byte, label) in REQUEST_IDX_BYTE_LABELS.iter().enumerate().take(*num_bytes)
                    {
                        let top = byte + 1 == *num_bytes;
                        let source = match (respond_idx, phase) {
                            (Some(_), _) => Source::Column(*next_clock.next().ok_or(
                                errors::Error::Protocol {
                                    protocol: "service",
                                    message: "byte-split clock binds one committed \
                                              column per byte on the responder",
                                },
                            )?),
                            (None, Some(col)) if top => Source::PhaseColumn(col),
                            (None, _) => Source::RowIndexByte(byte),
                        };

                        sources.push((source, *label));
                    }
                }
            }
        }

        let spec = match self.kind {
            BusKind::Permutation => PermutationCheckSpec::new(sources, Some(selector)),
            BusKind::Lookup => PermutationCheckSpec::new_lookup(sources, Some(selector)),
        };

        Ok(match self.clock_waiver {
            Some(reason) => spec.with_clock_waiver(reason),
            None => spec,
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

/// Whether `label` marks a committed column
/// as carrying the partner's row index.
pub fn is_request_idx_label(label: ChallengeLabel) -> bool {
    label == REQUEST_IDX_LABEL || REQUEST_IDX_BYTE_LABELS.contains(&label)
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
                ServiceSlot::RequestIdx { num_bytes: 4 },
                ServiceSlot::Value(b"k_dir"),
            ],
            clock_waiver: None,
        }
    }

    fn byte_split_service() -> Service {
        Service {
            bus_id: "svc",
            kind: BusKind::Permutation,
            slots: vec![
                ServiceSlot::Value(b"k_a"),
                ServiceSlot::Const(b"k_dom", 7),
                ServiceSlot::RequestIdxBytes { num_bytes: 4 },
                ServiceSlot::Value(b"k_dir"),
            ],
            clock_waiver: None,
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

    #[test]
    fn service_endpoints_share_schema() {
        let svc = test_service();

        let req = svc.request(&[10, 11], 12).unwrap();
        let resp = svc.respond(&[20, 21], &[22], 23).unwrap();

        let req_labels: Vec<_> = req.sources.iter().map(|(_, l)| *l).collect();
        let resp_labels: Vec<_> = resp.sources.iter().map(|(_, l)| *l).collect();

        assert_eq!(req_labels, resp_labels);

        assert_eq!(req.sources[0].0, Source::Column(10));
        assert_eq!(req.sources[1].0, Source::Const(7));
        assert_eq!(req.sources[2].0, Source::RowIndexLeBytes(4));
        assert_eq!(req.sources[3].0, Source::Column(11));
        assert_eq!(req.selector, Some(12));

        assert_eq!(resp.sources[0].0, Source::Column(20));
        assert_eq!(resp.sources[1].0, Source::Const(7));
        assert_eq!(resp.sources[2].0, Source::Column(22));
        assert_eq!(resp.sources[3].0, Source::Column(21));
        assert_eq!(resp.selector, Some(23));

        req.validate_clock_stitching("svc").unwrap();
        resp.validate_clock_stitching("svc").unwrap();
    }

    #[test]
    fn service_rejects_schema_mismatches() {
        let svc = test_service();

        assert!(svc.request(&[10], 12).is_err());
        assert!(svc.respond(&[20, 21], &[], 23).is_err());

        let lookup = Service {
            bus_id: "rom",
            kind: BusKind::Lookup,
            slots: vec![ServiceSlot::Value(b"k_a")],
            clock_waiver: None,
        };

        assert!(lookup.respond(&[3], &[9], 4).is_err());

        let clockless = Service {
            bus_id: "bad",
            kind: BusKind::Permutation,
            slots: vec![ServiceSlot::Value(b"k_a")],
            clock_waiver: None,
        };

        assert!(clockless.request(&[1], 2).is_err());
    }

    #[test]
    fn lookup_service_derives_lookup_specs() {
        let svc = Service {
            bus_id: "rom",
            kind: BusKind::Lookup,
            slots: vec![ServiceSlot::Value(b"k_a")],
            clock_waiver: None,
        };

        let req = svc.request(&[1], 2).unwrap();
        let resp = svc.respond(&[3], &[], 4).unwrap();

        assert_eq!(req.kind, BusKind::Lookup);
        assert_eq!(resp.kind, BusKind::Lookup);
        assert_eq!(req.sources.len(), 1);
    }

    #[test]
    fn request_phased_replaces_only_top_clock_byte() {
        let svc = byte_split_service();

        let phased = svc.request_phased(&[10, 11], 60, 12).unwrap();
        let plain = svc.request(&[10, 11], 12).unwrap();

        let labels = |spec: &PermutationCheckSpec| -> Vec<ChallengeLabel> {
            spec.sources.iter().map(|(_, l)| *l).collect()
        };

        assert_eq!(labels(&phased), labels(&plain));

        assert_eq!(phased.sources[2].0, Source::RowIndexByte(0));
        assert_eq!(phased.sources[3].0, Source::RowIndexByte(1));
        assert_eq!(phased.sources[4].0, Source::RowIndexByte(2));
        assert_eq!(phased.sources[5].0, Source::PhaseColumn(60));
        assert_eq!(plain.sources[5].0, Source::RowIndexByte(3));

        assert!(phased.has_real_clock_source());

        phased.validate_clock_stitching("svc").unwrap();

        let responder = svc.respond(&[20, 21], &[40, 41, 42, 43], 23).unwrap();

        validate_bus_set(vec![("svc", &phased), ("svc", &responder)]).unwrap();
    }

    #[test]
    fn request_phased_rejects_whole_word_clock() {
        assert!(test_service().request_phased(&[10, 11], 60, 12).is_err());
    }

    #[test]
    fn request_phased_rejects_clockless_schema() {
        let waived = Service {
            bus_id: "svc",
            kind: BusKind::Permutation,
            slots: vec![ServiceSlot::Value(b"k_a")],
            clock_waiver: Some("see permutation.rs: a body constraint forces per-row uniqueness"),
        };

        assert!(waived.request_phased(&[10], 60, 12).is_err());
        assert!(waived.request(&[10], 12).is_ok());

        let lookup = Service {
            bus_id: "svc",
            kind: BusKind::Lookup,
            slots: vec![ServiceSlot::Value(b"k_a")],
            clock_waiver: None,
        };

        assert!(lookup.request_phased(&[10], 60, 12).is_err());
        assert!(lookup.request(&[10], 12).is_ok());
    }

    #[test]
    fn request_phased_rejects_one_byte_clock() {
        let svc = Service {
            slots: vec![ServiceSlot::RequestIdxBytes { num_bytes: 1 }],
            ..byte_split_service()
        };

        assert!(svc.request_phased(&[], 60, 12).is_err());
        assert!(svc.request(&[], 12).is_ok());
    }

    #[test]
    fn witness_clock_phase_is_rejected() {
        let svc = byte_split_service();
        let phased = svc.request_phased(&[10, 11], 60, 12).unwrap();
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
