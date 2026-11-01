// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use alloc::vec::Vec;
use hekate_crypto::merkle::layer_log;
use hekate_math::{Block256, TowerField};

use crate::config::{self, Config, ldt_bits};
use crate::errors;

/// Bytes of one Merkle digest.
const DIGEST_BYTES: usize = 32;

const MASKS: usize = 2;

/// The phase a polynomial is committed in; a tree holds one kind.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolyKind {
    /// A table's packed trace, committed before `γ`, `β`.
    Trace,

    /// A bus-carrying table's `h`, committed after `γ`, `β`.
    Helper,

    /// A uniform mask over `ℓ − a` variables, committed with
    /// the helpers; it hides every fold round from round 1.
    Mask,
}

/// The field of a polynomial's Reed-Solomon code.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum CodeField {
    /// GF(2^32): Bit, B8, B16 and B32 cells.
    B32,

    /// GF(2^64): B64 cells.
    B64,

    /// GF(2^128): every cell type.
    B128,
}

impl CodeField {
    /// Returns the field's width in bits.
    pub fn bits(self) -> usize {
        match self {
            Self::B32 => 32,
            Self::B64 => 64,
            Self::B128 => 128,
        }
    }

    /// Returns the width of one symbol in bytes.
    pub fn bytes(self) -> usize {
        self.bits() / 8
    }

    /// Returns the narrowest field, `self` or wider, whose
    /// Cantor basis spans a `2^domain_vars`-point domain.
    pub fn holding(self, domain_vars: usize) -> Self {
        [Self::B32, Self::B64, Self::B128]
            .into_iter()
            .find(|&f| f >= self && domain_vars <= f.bits())
            .unwrap_or(Self::B128)
    }
}

/// A committed polynomial as its table lays it out:
/// `2^block_vars` blocks of `2^row_vars` cells.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PolyShape {
    /// Picks the polynomial's trees.
    pub kind: PolyKind,

    /// `n_i`, the table's row variables.
    pub row_vars: usize,

    /// `θ_k`, the variables indexing its blocks.
    pub block_vars: usize,

    /// Leading blocks of uniform cells: noise, then blinds.
    pub uniform_blocks: usize,

    /// The narrowest field holding every cell.
    pub cells: CodeField,
}

impl PolyShape {
    /// `ℓ_k`, before the layout pads it.
    pub fn vars(&self) -> usize {
        self.row_vars + self.block_vars
    }
}

/// Where one polynomial enters the fold,
/// derived by [`FoldLayout::new`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PolyFold {
    /// The shape it was derived from.
    pub shape: PolyShape,

    /// `ℓ_k`, zero-padded to `ℓ − a` when at or below it.
    pub vars: usize,

    /// `b_k`, the round it joins at; 0 in the top class.
    pub join_round: usize,

    /// `p_k`, its own variables folded before it joins.
    pub prefold_vars: usize,

    /// `log2 o_k`, its symbols opened per query.
    pub leaf_vars: usize,

    /// Fewest cells its uniform blocks hold in
    /// zero-knowledge mode, 0 in base mode.
    pub noise_cells: usize,

    /// The code's field: GF(2^128) in zero-knowledge mode, else
    /// the cells' field widened until its basis spans the domain.
    pub field: CodeField,
}

/// What one fold tree commits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TreeKind {
    /// Every polynomial of one kind; the helper tree
    /// also holds the masks in zero-knowledge mode.
    Polys(PolyKind),

    /// `H_c` of committed round `c ≥ 1`, its symbols in the fold field.
    Round(usize),
}

/// One Merkle tree of the fold and the geometry its opening follows.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FoldTree {
    /// What the tree commits.
    pub kind: TreeKind,

    /// Members in derived order; empty for the rounds.
    pub polys: Vec<usize>,

    /// `c`: a query opens the tree at its index `i_c`.
    pub open_round: usize,

    /// Symbols per leaf: each member's `o_k` in turn.
    pub leaf_symbols: usize,

    /// Bytes per leaf, each symbol at its code field's width.
    pub leaf_bytes: usize,

    /// `log2` of the leaf count.
    pub depth: usize,

    /// Level of the opening's multiproof layer.
    pub layer_log: usize,
}

/// The fold's derived geometry: stop, rounds,
/// joins and trees, checked against its config.
#[derive(Clone, Debug)]
pub struct FoldLayout {
    /// `ℓ`, the variables of the largest polynomial.
    num_vars: usize,

    /// `s`, the variables of the final layer.
    stop_vars: usize,

    /// `a`, the variables one committed round binds.
    arity_log: usize,

    /// `ω`, the code's log inverse rate.
    log_inv_rate: usize,

    /// `t`, the query paths through every round.
    num_queries: usize,

    /// One fold per committed polynomial, derived order.
    polys: Vec<PolyFold>,

    /// Every tree in commitment and opening order.
    trees: Vec<FoldTree>,
}

impl FoldLayout {
    /// Lays out `shapes`, given in derived order. Errors on
    /// an arity or stop bound out of range, a pool below one
    /// round, uniform blocks below the noise, or weak bits.
    pub fn new(config: &Config, shapes: &[PolyShape]) -> errors::Result<Self> {
        config.check_fold()?;

        let arity_log = config.fold_arity_log;
        let num_vars = shapes.iter().map(PolyShape::vars).max().unwrap_or(0);

        if num_vars < arity_log {
            return Err(config::Error::PoolBelowArity {
                num_vars,
                fold_arity_log: arity_log,
            }
            .into());
        }

        let log_inv_rate = config.code_rate.log_inv_rate() as usize;

        if num_vars + log_inv_rate >= usize::BITS as usize {
            return Err(config::Error::PoolAboveWord {
                num_vars,
                log_inv_rate,
            }
            .into());
        }

        let ceiling = config.early_stop_vars.min(num_vars - arity_log);
        let stop_vars = num_vars - (num_vars - ceiling).div_ceil(arity_log) * arity_log;

        let num_queries = config.num_queries;

        let mask = PolyShape {
            kind: PolyKind::Mask,
            row_vars: num_vars - arity_log,
            block_vars: 0,
            uniform_blocks: 0,
            cells: CodeField::B128,
        };

        let masks = match config.zero_knowledge {
            true => MASKS,
            false => 0,
        };

        let polys: Vec<PolyFold> = shapes
            .iter()
            .copied()
            .chain(core::iter::repeat_n(mask, masks))
            .map(|shape| fold_of(config, num_vars, shape))
            .collect();

        for (poly, fold) in polys.iter().enumerate() {
            if fold.shape.uniform_blocks << fold.shape.row_vars < fold.noise_cells {
                return Err(config::Error::TableBelowNoise {
                    poly,
                    row_vars: fold.shape.row_vars,
                    noise_cells: fold.noise_cells,
                }
                .into());
            }
        }

        let tree = |kind, polys, open_round, leaf_symbols, leaf_bytes| {
            let depth = num_vars + log_inv_rate - arity_log * open_round;

            FoldTree {
                kind,
                polys,
                open_round,
                leaf_symbols,
                leaf_bytes,
                depth,
                layer_log: layer_log(num_queries, depth),
            }
        };

        let mut trees = Vec::new();
        for kind in [PolyKind::Trace, PolyKind::Helper] {
            let members: Vec<usize> = polys
                .iter()
                .enumerate()
                .filter(|(_, fold)| {
                    matches!(
                        (kind, fold.shape.kind),
                        (PolyKind::Trace, PolyKind::Trace)
                            | (PolyKind::Helper, PolyKind::Helper | PolyKind::Mask)
                    )
                })
                .map(|(k, _)| k)
                .collect();

            if members.is_empty() {
                continue;
            }

            let leaf_symbols = members.iter().map(|&k| 1 << polys[k].leaf_vars).sum();
            let leaf_bytes = members
                .iter()
                .map(|&k| polys[k].field.bytes() << polys[k].leaf_vars)
                .sum();

            trees.push(tree(
                TreeKind::Polys(kind),
                members,
                1,
                leaf_symbols,
                leaf_bytes,
            ));
        }

        let rounds = (num_vars - stop_vars) / arity_log;
        let fold_bytes = Block256::BITS / 8;

        for round in 1..rounds {
            trees.push(tree(
                TreeKind::Round(round),
                Vec::new(),
                round + 1,
                1 << arity_log,
                fold_bytes << arity_log,
            ));
        }

        let layout = Self {
            num_vars,
            stop_vars,
            arity_log,
            log_inv_rate,
            num_queries,
            polys,
            trees,
        };

        let terms = [
            ("fold queries", layout.query_bits()),
            ("fold proximity", layout.proximity_bits()),
        ];

        for (round, bits) in terms {
            if bits < config.min_security_bits {
                return Err(config::Error::SecurityTooLow {
                    round,
                    estimated_bits: bits,
                    min_bits: config.min_security_bits,
                }
                .into());
            }
        }

        Ok(layout)
    }

    /// Returns `ℓ`, the variables of the largest polynomial.
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Returns `s`, the variables of the final layer.
    pub fn stop_vars(&self) -> usize {
        self.stop_vars
    }

    /// Returns `a`, the variables one committed round binds.
    pub fn arity_log(&self) -> usize {
        self.arity_log
    }

    /// Returns `R = (ℓ − s) / a`, the fold rounds, at least 1.
    pub fn rounds(&self) -> usize {
        (self.num_vars - self.stop_vars) / self.arity_log
    }

    /// Returns `ω`, the log inverse rate of every code.
    pub fn log_inv_rate(&self) -> usize {
        self.log_inv_rate
    }

    /// Returns `t`, the query indices drawn once and shared by every tree.
    pub fn num_queries(&self) -> usize {
        self.num_queries
    }

    /// Returns one fold per committed polynomial, in derived order.
    pub fn polys(&self) -> &[PolyFold] {
        &self.polys
    }

    /// Returns every tree in commitment and opening order.
    pub fn trees(&self) -> &[FoldTree] {
        &self.trees
    }

    /// Returns the fold's security estimate, the
    /// weaker of its query and proximity terms.
    pub fn estimated_security_bits(&self) -> usize {
        self.query_bits().min(self.proximity_bits())
    }

    /// Returns an upper bound on the fold's proof bytes: roots,
    /// opened leaves and paths, the fold rounds and the final layer.
    pub fn proof_bytes(&self) -> usize {
        let fold_bytes = Block256::BITS / 8;

        let opened: usize = self
            .trees
            .iter()
            .map(|tree| {
                let leaves = self
                    .num_queries
                    .min(1usize.checked_shl(tree.depth as u32).unwrap_or(usize::MAX));

                let digests =
                    1 + self.num_queries * (tree.depth - tree.layer_log) + (1 << tree.layer_log);

                leaves * tree.leaf_bytes + digests * DIGEST_BYTES
            })
            .sum();

        let rounds = 2 * (self.num_vars - self.stop_vars) * fold_bytes;
        let final_layer = fold_bytes << self.stop_vars;

        opened + rounds + final_layer
    }

    /// Returns the query term, `⌊t · log2(2 / (1 + ρ))⌋` (DP24 Prop 4.24).
    fn query_bits(&self) -> usize {
        ldt_bits(self.num_queries, 1, 1 << self.log_inv_rate)
    }

    /// Returns the proximity term, `256 − ⌈log2((|K| + 2) · |D_ℓ|)⌉`:
    /// `|K|` for the `λ` curve (BCIKS20 Thm 1.5), one for the folds
    /// (DP24 Prop 4.23) and one the bound does not need.
    fn proximity_bits(&self) -> usize {
        let domain_vars = self.num_vars + self.log_inv_rate;

        Block256::BITS.saturating_sub(ceil_log2(self.polys.len() + 2) + domain_vars)
    }
}

/// Returns the noise cells of each of `shapes`
/// as [`FoldLayout::new`] lays them out together;
/// errors as [`Config::check_fold`] does.
pub fn noise_cells_of(config: &Config, shapes: &[PolyShape]) -> errors::Result<Vec<usize>> {
    config.check_fold()?;

    let num_vars = shapes.iter().map(PolyShape::vars).max().unwrap_or(0);

    Ok(shapes
        .iter()
        .map(|&shape| fold_of(config, num_vars, shape).noise_cells)
        .collect())
}

/// Returns `(t + 2) · 2^leaf_vars`, the uniform cells that hide a polynomial
/// opened at `2^leaf_vars` symbols per query in zero knowledge.
pub fn noise_cells(num_queries: usize, leaf_vars: usize) -> usize {
    num_queries.saturating_add(2).saturating_mul(1 << leaf_vars)
}

fn fold_of(config: &Config, num_vars: usize, shape: PolyShape) -> PolyFold {
    let arity_log = config.fold_arity_log;
    let log_inv_rate = config.code_rate.log_inv_rate() as usize;

    let vars = shape.vars().max(num_vars.saturating_sub(arity_log));
    let lift = num_vars - vars;
    let partial = lift % arity_log;

    let join_round = lift / arity_log + usize::from(partial > 0);
    let prefold_vars = (arity_log - partial) % arity_log;

    let leaf_vars = match join_round {
        0 => arity_log,
        _ => prefold_vars,
    };

    let field = match config.zero_knowledge {
        true => CodeField::B128,
        false => shape.cells.holding(vars + log_inv_rate),
    };

    PolyFold {
        shape,
        vars,
        join_round,
        prefold_vars,
        leaf_vars,
        noise_cells: match (config.zero_knowledge, shape.kind) {
            (true, PolyKind::Trace | PolyKind::Helper) => {
                noise_cells(config.num_queries, leaf_vars)
            }
            _ => 0,
        },
        field,
    }
}

/// Returns `⌈log2 n⌉`, saturating at `usize::BITS`.
fn ceil_log2(n: usize) -> usize {
    n.checked_next_power_of_two()
        .map_or(usize::BITS, usize::ilog2) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CodeRate, MAX_FOLD_ARITY_LOG};

    fn config(arity_log: usize, early_stop_vars: usize, zero_knowledge: bool) -> Config {
        Config {
            fold_arity_log: arity_log,
            early_stop_vars,
            zero_knowledge,
            min_security_bits: 0,
            ..Config::prod()
        }
    }

    fn trace(row_vars: usize, block_vars: usize) -> PolyShape {
        PolyShape {
            kind: PolyKind::Trace,
            row_vars,
            block_vars,
            uniform_blocks: 1,
            cells: CodeField::B128,
        }
    }

    fn helper(row_vars: usize, block_vars: usize) -> PolyShape {
        PolyShape {
            kind: PolyKind::Helper,
            row_vars,
            block_vars,
            uniform_blocks: 1,
            cells: CodeField::B128,
        }
    }

    fn rejection(config: &Config, shapes: &[PolyShape]) -> config::Error {
        match FoldLayout::new(config, shapes) {
            Err(errors::Error::Config(e)) => e,
            other => panic!("expected a config rejection, got {other:?}"),
        }
    }

    #[test]
    fn stop_is_largest_whole_round_stop_under_bound() {
        for arity_log in 1..=MAX_FOLD_ARITY_LOG {
            for early_stop_vars in arity_log - 1..=12 {
                for num_vars in arity_log..=40 {
                    let layout = FoldLayout::new(
                        &config(arity_log, early_stop_vars, false),
                        &[trace(num_vars, 0)],
                    )
                    .unwrap();

                    let s = layout.stop_vars();
                    let ceiling = early_stop_vars.min(num_vars - arity_log);

                    assert!(s <= ceiling);
                    assert!(s + arity_log > ceiling);
                    assert_eq!((num_vars - s) % arity_log, 0);
                    assert!(layout.rounds() >= 1);
                }
            }
        }
    }

    #[test]
    fn stop_examples() {
        let cases = [
            (2, 10, 20, 10, 5),
            (2, 10, 21, 9, 6),
            (2, 10, 5, 3, 1),
            (2, 10, 2, 0, 1),
            (3, 10, 23, 8, 5),
            (1, 0, 7, 0, 7),
        ];

        for (arity_log, early_stop_vars, num_vars, stop_vars, rounds) in cases {
            let layout = FoldLayout::new(
                &config(arity_log, early_stop_vars, false),
                &[trace(num_vars, 0)],
            )
            .unwrap();

            assert_eq!(
                (layout.stop_vars(), layout.rounds()),
                (stop_vars, rounds),
                "a = {arity_log}, bound {early_stop_vars}, {num_vars} variables",
            );
        }
    }

    #[test]
    fn polys_below_top_join_at_round_one() {
        let shapes = [
            trace(20, 0),
            trace(19, 0),
            trace(17, 0),
            trace(15, 0),
            trace(4, 0),
        ];
        let layout = FoldLayout::new(&config(3, 10, false), &shapes).unwrap();

        assert_eq!((layout.stop_vars(), layout.rounds()), (8, 4));

        let joins: Vec<(usize, usize, usize, usize)> = layout
            .polys()
            .iter()
            .map(|p| (p.vars, p.join_round, p.prefold_vars, p.leaf_vars))
            .collect();

        assert_eq!(
            joins,
            [
                (20, 0, 0, 3),
                (19, 1, 2, 2),
                (17, 1, 0, 0),
                (17, 1, 0, 0),
                (17, 1, 0, 0),
            ]
        );
    }

    #[test]
    fn arity_outside_cap_rejected() {
        for arity_log in [0, MAX_FOLD_ARITY_LOG + 1] {
            assert_eq!(
                rejection(&config(arity_log, 12, false), &[trace(20, 0)]),
                config::Error::FoldArity {
                    fold_arity_log: arity_log
                }
            );
        }
    }

    #[test]
    fn stop_bound_below_arity_rejected() {
        assert_eq!(
            rejection(&config(3, 1, false), &[trace(20, 0)]),
            config::Error::EarlyStopBelowArity {
                early_stop_vars: 1,
                fold_arity_log: 3,
            }
        );
    }

    #[test]
    fn pool_below_arity_rejected() {
        assert_eq!(
            rejection(&config(3, 10, false), &[trace(1, 1)]),
            config::Error::PoolBelowArity {
                num_vars: 2,
                fold_arity_log: 3,
            }
        );

        assert_eq!(
            rejection(&config(1, 10, false), &[]),
            config::Error::PoolBelowArity {
                num_vars: 0,
                fold_arity_log: 1,
            }
        );
    }

    #[test]
    fn noise_rule_holds_without_security_floor() {
        let zk = config(2, 10, true);
        let noise = noise_cells(zk.num_queries, 2);
        let floor = ceil_log2(noise);

        assert_eq!(zk.min_security_bits, 0);
        assert_eq!(
            rejection(&zk, &[trace(floor - 1, 3)]),
            config::Error::TableBelowNoise {
                poly: 0,
                row_vars: floor - 1,
                noise_cells: noise,
            }
        );

        assert!(FoldLayout::new(&zk, &[trace(floor, 2)]).is_ok());
        assert!(FoldLayout::new(&config(2, 10, false), &[trace(floor - 1, 3)]).is_ok());
    }

    #[test]
    fn joining_class_needs_its_own_noise_only() {
        let zk = config(2, 10, true);
        let noise = noise_cells(zk.num_queries, 0);
        let floor = ceil_log2(noise);
        let joining = |row_vars| [trace(14, 2), trace(row_vars, 16 - row_vars - 2)];

        assert!(FoldLayout::new(&zk, &joining(floor)).is_ok());
        assert_eq!(
            rejection(&zk, &joining(floor - 1)),
            config::Error::TableBelowNoise {
                poly: 1,
                row_vars: floor - 1,
                noise_cells: noise,
            }
        );
    }

    #[test]
    fn trees_follow_absorb_order() {
        let shapes = [
            trace(18, 2),
            trace(16, 2),
            trace(17, 3),
            helper(17, 3),
            helper(15, 2),
        ];

        let layout = FoldLayout::new(&config(2, 10, true), &shapes).unwrap();

        assert_eq!(
            (layout.num_vars(), layout.stop_vars(), layout.rounds()),
            (20, 10, 5)
        );

        let trees: Vec<(TreeKind, Vec<usize>, usize, usize, usize, usize)> = layout
            .trees()
            .iter()
            .map(|t| {
                (
                    t.kind,
                    t.polys.clone(),
                    t.open_round,
                    t.leaf_symbols,
                    t.leaf_bytes,
                    t.depth,
                )
            })
            .collect();

        let mut expected = vec![
            (
                TreeKind::Polys(PolyKind::Trace),
                vec![0, 1, 2],
                1,
                4 + 1 + 4,
                16 * (4 + 1 + 4),
                19,
            ),
            (
                TreeKind::Polys(PolyKind::Helper),
                vec![3, 4, 5, 6],
                1,
                4 + 1 + 1 + 1,
                16 * (4 + 1 + 1 + 1),
                19,
            ),
        ];

        for (round, depth) in [(1, 17), (2, 15), (3, 13), (4, 11)] {
            expected.push((TreeKind::Round(round), vec![], round + 1, 4, 32 * 4, depth));
        }

        assert_eq!(trees, expected);
    }

    #[test]
    fn query_term_reaches_floor_at_note_counts() {
        let bits = |code_rate, num_queries| {
            let config = Config {
                code_rate,
                num_queries,
                ..config(2, 10, false)
            };

            FoldLayout::new(&config, &[trace(20, 0)])
                .unwrap()
                .query_bits()
        };

        assert!(bits(CodeRate::Half, 241) >= 100);
        assert!(bits(CodeRate::Half, 240) < 100);
        assert!(bits(CodeRate::Quarter, 148) >= 100);
        assert!(bits(CodeRate::Quarter, 147) < 100);
    }

    #[test]
    fn proximity_term_charges_batch_and_domain() {
        let shapes = vec![trace(29, 0); 64];
        let layout = FoldLayout::new(&config(2, 10, false), &shapes).unwrap();

        assert_eq!(layout.proximity_bits(), 256 - 7 - 30);
    }

    #[test]
    fn security_floor_names_failing_term() {
        let weak = Config {
            num_queries: 4,
            min_security_bits: 100,
            ..config(2, 10, false)
        };

        assert!(matches!(
            rejection(&weak, &[trace(20, 0)]),
            config::Error::SecurityTooLow {
                round: "fold queries",
                ..
            }
        ));
    }

    #[test]
    fn proof_bytes_counts_every_tree() {
        let tiny = Config {
            num_queries: 2,
            ..config(1, 1, false)
        };
        let layout = FoldLayout::new(&tiny, &[trace(4, 0)]).unwrap();

        assert_eq!((layout.stop_vars(), layout.rounds()), (1, 3));
        assert_eq!(layout.proof_bytes(), 352 + 352 + 288 + 192 + 64);
    }

    #[test]
    fn masks_join_helper_tree_in_zero_knowledge_only() {
        let shapes = [trace(14, 6), helper(14, 2)];

        let zk = FoldLayout::new(&config(4, 11, true), &shapes).unwrap();
        let masks: Vec<&PolyFold> = zk
            .polys()
            .iter()
            .filter(|p| p.shape.kind == PolyKind::Mask)
            .collect();

        assert_eq!(masks.len(), MASKS);
        assert_eq!(zk.polys().len(), shapes.len() + MASKS);

        for mask in masks {
            assert_eq!(
                (
                    mask.vars,
                    mask.join_round,
                    mask.prefold_vars,
                    mask.leaf_vars
                ),
                (16, 1, 0, 0)
            );
            assert_eq!((mask.field, mask.noise_cells), (CodeField::B128, 0));
        }

        assert_eq!(zk.trees()[1].polys, vec![1, 2, 3]);

        let raw = FoldLayout::new(&config(4, 11, false), &shapes).unwrap();

        assert_eq!(raw.polys().len(), shapes.len());
        assert_eq!(raw.trees()[1].polys, vec![1]);
    }

    #[test]
    fn code_field_widens_to_hold_domain() {
        assert_eq!(CodeField::B32.holding(32), CodeField::B32);
        assert_eq!(CodeField::B32.holding(33), CodeField::B64);
        assert_eq!(CodeField::B64.holding(12), CodeField::B64);
        assert_eq!(CodeField::B64.holding(65), CodeField::B128);
        assert_eq!(CodeField::B128.holding(1), CodeField::B128);

        let narrow = PolyShape {
            cells: CodeField::B32,
            ..trace(30, 2)
        };

        let layout = FoldLayout::new(&config(2, 10, false), &[narrow]).unwrap();

        assert_eq!(layout.polys()[0].field, CodeField::B64);
    }

    #[test]
    fn leaf_bytes_follow_member_fields() {
        let wide = trace(18, 2);
        let lanes = PolyShape {
            cells: CodeField::B64,
            ..trace(17, 3)
        };
        let bits = PolyShape {
            cells: CodeField::B32,
            ..trace(16, 2)
        };

        let layout = FoldLayout::new(&config(2, 10, false), &[wide, lanes, bits]).unwrap();
        let fields: Vec<CodeField> = layout.polys().iter().map(|p| p.field).collect();

        assert_eq!(fields, [CodeField::B128, CodeField::B64, CodeField::B32]);
        assert_eq!(layout.trees()[0].leaf_symbols, 4 + 4 + 1);
        assert_eq!(layout.trees()[0].leaf_bytes, 16 * 4 + 8 * 4 + 4);
    }

    #[test]
    fn noise_holds_two_leaves_past_queries() {
        let t = 241;

        assert_eq!(noise_cells(t, 0), t + 2);
        assert_eq!(noise_cells(t, 4), 16 * (t + 2));
    }

    #[test]
    fn zero_knowledge_codes_every_polynomial_over_gf128() {
        let lanes = PolyShape {
            cells: CodeField::B64,
            ..trace(17, 3)
        };

        let bits = PolyShape {
            cells: CodeField::B32,
            ..trace(16, 2)
        };

        let fields = |zero_knowledge| -> Vec<(CodeField, usize)> {
            FoldLayout::new(&config(2, 10, zero_knowledge), &[lanes, bits])
                .unwrap()
                .polys()
                .iter()
                .take(2)
                .map(|p| (p.field, p.noise_cells))
                .collect()
        };

        let t = Config::prod().num_queries;

        assert_eq!(
            fields(true),
            [(CodeField::B128, 4 * (t + 2)), (CodeField::B128, t + 2)]
        );
        assert_eq!(fields(false), [(CodeField::B64, 0), (CodeField::B32, 0)]);
    }

    #[test]
    fn pool_past_word_rejected() {
        let wide = usize::BITS as usize - 1;

        assert_eq!(
            rejection(&config(2, 10, false), &[trace(wide, 0)]),
            config::Error::PoolAboveWord {
                num_vars: wide,
                log_inv_rate: 1,
            }
        );
    }
}
