# One support cell short: a zero-knowledge postmortem

In 0.35.0 and 0.36.0, a zero-knowledge proof could confirm a guess of its own witness. Both
releases committed the LogUp helper columns of every bus-carrying table under Merkle leaves
without salts, and on every table whose grid has at least 2^11 columns they gave each
message 287 random support cells, as many as the verifier draws columns. On a proof whose
287 draws for such a table were pairwise distinct, every leaf of that table's helper tree,
and its root, followed from the table's bus keys and what the proof shows. Anyone holding a
candidate for those keys could check it offline.

The gap leaves soundness alone: it concerns what a valid proof reveals, and no forgery
follows from it. No shipped example is recoverable either, because the smallest private
input behind any exposed table is a 256-bit seed. What failed is the zero-knowledge
guarantee itself. A statement about a short secret, such as a PIN, a password or a choice
among a few options, could be recovered by enumeration from any proof that exposed a table
carrying it.

0.37.0 commits 288 support cells over 287 queries, and with a nonzero security floor both
parties reject a support that does not exceed the query count. The fix costs one field
element per proof.

For readers new to this series: Hekate commits a table by encoding each message, `S` random
support cells followed by `L` data cells, into a Reed-Solomon codeword of `W` symbols. The
verifier draws `t` of the `W` columns at random, opens them and checks them against a folded
vector the prover sends. Each column of the encoded matrix is one Merkle leaf, the hash of
its symbols. In 0.35.0 and 0.36.0 the LogUp helper columns `h = s / (gamma + key)` of a
bus-carrying table sat in a tree of their own, opened at the same columns as the table's
trace.

## Summary

| # | Gap                                                                    | Effect                                                                | Against 0.36.0                                      |
|:--|:-----------------------------------------------------------------------|:----------------------------------------------------------------------|:----------------------------------------------------|
| 1 | At `S = t`, on distinct draws, every unopened leaf of a table's helper tree was fixed | The root, or one sibling digest the proof ships, confirms a key guess | Exposure measured on real proofs, rebuilt at toy scale |
| 2 | The same held for the trace tree of a table on one grid row without ring units | The table's whole trace becomes checkable against a guess     | Established by reading                              |
| 3 | The hiding rule counted the opened columns only                        | `S >= t` passed two postmortems and the configuration check           | Established by reading                              |

Gap 1 is the mechanism and gap 2 the same mechanism on a rarer shape. Gap 3 is why it
shipped: every statement of the rule was true of the symbols the verifier opens, and none
mentioned the leaves of the columns it leaves closed.

## Affected versions

| Running            | Exposed to                                                         | Action                          |
|:-------------------|:-------------------------------------------------------------------|:--------------------------------|
| 0.31.0 to 0.34.0   | None by default: `Config::prod()` ran 200 support cells over 176 queries; a custom configuration with equal counts was accepted | Upgrade to 0.37.0 and re-prove. |
| 0.35.0, 0.36.0     | 1, 2, 3                                                            | Upgrade to 0.37.0 and re-prove. |
| 0.37.0             | The limits below                                                   | Current.                        |

Releases before 0.31.0 hid the witness with a different encoding, which this postmortem
does not examine.

No proof crosses this boundary. `ldt_support_size` is absorbed into the transcript before
any challenge and it moved, which means a 0.36.0 proof fails replay at the first challenge.
The proof wire format moves from v6 to v7, carried by the pooled commitment shipping
alongside, which puts every table under one tree per oracle. The pinned prover release is
0.15.0.

Upgrading protects new proofs only. A proof already published under 0.35.0 or 0.36.0 keeps
its root and its sibling digests and, where its draws for a table were distinct, stays
checkable against a guess of its witness.

The third row states a limit. Proved: with one support cell more than the query count, the
symbol of every message at every unopened column is uniform given the view and every data
cell, unless fewer than two messages carry nonzero weight in the fold, which happens with
probability below 2^-118 on every example. Not examined here: the linear part of the view,
which rests on the blinds and on the outer argument, and the outer argument's own oracles.

## Where this one sits

The five postmortems before this one were about soundness. This is the first about zero
knowledge, and two of them already state the rule it corrects.

The 0.32.0 postmortem described the support block, which shipped in 0.31.0, and what it
buys: "Any `num_queries` opened columns are jointly uniform, and no committed column contains
cleartext witness symbols." Both halves hold at `S = t`. The 0.35.0 postmortem restated the
rule as that release raised the query count: "The support block hiding opened columns must
be at least as large as the query count." Also true, of the opened columns. The README says
the tree covers "encoded columns only (raw trace never hashed, true ZK)". A column whose
randomness the view pins is the raw trace under a public affine map, and its hash hides
nothing a guess cannot reach.

## 1. At `S = t` the helper tree's leaves were fixed

A message is a support `xi` of `S` cells and data `d` of `L` cells, and its symbol at a
point `alpha` of the codeword domain is `sum_{k<S} xi_k X_k(alpha) + sum_{k<L} d_k
X_{S+k}(alpha)`, where `X_k` is the novel polynomial basis and `deg X_k = k`. When
`S = t` and all `t` draws are distinct, the `S x t` matrix `(X_k(alpha_j))` is
invertible, since a nonzero polynomial of degree below `S` has fewer than `S` roots. The
support is then an affine function of the data and the opened symbols, and every other
symbol of the message follows.

The helper tree holds one column per bus endpoint on the table's grid. The table's bus keys
and selectors, with the public challenges `gamma` and `beta`, fix every data cell. 0.35.0
and 0.36.0 placed the blinds in the trace tree, one blind column per table and one more for
ring-switched units, and gave the helper tree none. Its messages were a support and helper
data, and nothing else.

Leaf `j` hashes `0x00` and the symbols of every helper message at column `j`; an inner node
hashes `0x01` and its two children. A guess of the bus keys therefore yields the helper
data, the support of every message, every unopened symbol, every leaf and the root. The
root is in the proof, and the octopus multiproof sends more: the digest of every unopened
sibling of an opened leaf, up to 287 per tree. A guess can be tested against any one of
them.

The verifier draws the 287 column indices uniformly with replacement. When two draws
coincide, fewer than `S` columns open, each message keeps a free support cell and the
leaves hide. A table was exposed exactly on the proofs whose draws were pairwise distinct:

| Codeword width `W` |   2^12 |  2^13 |  2^14 |  2^15 |  2^16 |  2^17 |  2^18 |
|:-------------------|-------:|------:|------:|------:|------:|------:|------:|
| All draws distinct |  0.003% |  0.6% |  8.0% | 28.5% | 53.4% | 73.1% | 85.5% |

Tables at split `c >= 11` ran `S = 287` at `W = 2L`. Narrower grids ran `L` support cells,
512 or 1,024, at `W = 4L`, and were not exposed. A table with no bus has no helper tree.

## 2. A trace on one grid row without ring units

The trace tree carries uniform cells: its blind column. On a table whose grid has one row
and no ring-switched units, that blind is a single message, and the fold vector fixes its
data, which enters the fold with a public nonzero coefficient beside terms the witness
determines. At `S = t` its support follows as in gap 1, and the whole trace tree is fixed
by the table's witness and the view. One shipped table has this shape, the ML-KEM-768
receiver's main table at 2^11 rows, exposed on about 1 proof in 28,600. A second grid row
or a ring unit leaves the trace tree free cells at any support size.

## 3. The rule counted the opened columns

Every statement of the hiding rule was about the symbols the verifier opens. The
configuration check rejected `ldt_support_size < num_queries` under the comment "opened
columns exhaust the noise budget and witness data leaks". The two postmortems above said the
same. At `S = t`, the `t` opened symbols of a message are a bijective image of its support
and carry nothing about its data. The other `W - t` columns reach the verifier through their
leaves, and no statement counted them.

## What a proof exposed

Measured on 0.36.0 and its pinned prover release 0.14.0, the fifteen example workloads with
zero knowledge on, and the verifier logging each table's geometry and its count of distinct
draws:

| Workload               | Helper trees at `S = t` | Some tree exposed, per proof | Private input a guess must fix         |
|:-----------------------|------------------------:|-----------------------------:|:---------------------------------------|
| ML-KEM-768 sender      |                       3 |                         8.1% | `m` and `ek`, from 256-bit seeds       |
| ML-KEM-768 receiver    |                       5 |                        29.4% | `d`, 256 bits; `z` too for three of the five |
| ML-DSA-44              |                       4 |                        28.9% | the public key and the signature       |
| ML-DSA-65              |                       5 |                        34.7% | the public key and the signature       |
| ML-DSA-87              |                       5 |                        57.7% | the public key and the signature       |
| AES-128, AES-256       |                       3 |                        94.8% | the key and 31,250 random blocks       |
| RSA-2048               |                       0 |                            0 |                                        |
| SHA-256, 2^11 / 2^21   |                       1 |                0.003% / 85.5% | the whole preimage, 2.5 KB / 8.4 MB    |
| Keccak-f, 2^15 / 2^20  |                       1 |                28.5% / 73.1% | the whole message, ~178 KB / ~5.4 MB   |
| Fibonacci              |                       0 |                            0 | none                                   |

In that run, the proofs of ML-DSA-87, AES-128, AES-256 and SHA-256 at 2^21 rows each had a
helper tree with 287 distinct draws. Every RSA-2048 table has at most 2^10 rows and ran 512
or 1,024 support cells. The ML-DSA examples prove verification, with the public key and the
signature as witness.

A test costs little. Once per proof, the attacker inverts the `S x S` matrix of the opened
points, about 24 million multiplications in GF(2^128). Per guess, it computes the table's
helper values, `M (L + t)` multiplications for the symbols of the `M` helper messages at one
sibling column, and one hash of `16 M + 1` bytes. Across the exposed tables of the examples
that runs from 2,335 multiplications, on the ML-KEM-768 sender's Keccak table, to 4.2 million
on SHA-256 at 2^21 rows. A guess that changes `u` key cells updates in `O(u)` field
operations per helper column and one hash.

## How we found out

The gap surfaced during work on our note on the pooled commitment. The check against
shipped code ran on 6 October 2026. The configuration, leaf hash and query draw were read at
the 0.35.0 and 0.36.0 tags, and the pinned prover releases 0.13.0 and 0.14.0 from their
source. The fifteen example workloads were then proved on 0.36.0
with prover 0.14.0, the verifier logging per table its split, support size, helper column
count and distinct draws.

## How the fix is checked

**Configuration.** `Config::prod()` sets 288 support cells over 287 queries. `check_security`
rejects `ldt_support_size <= num_queries` with `InsufficientSupport` whenever
`min_security_bits > 0`, and the verifier runs
it on the pool's shape before it reads an opening. `support_below_queries_rejected` and
`support_equal_to_queries_rejected` in `hekate-core/src/config.rs` pin the rejection below
and at the query count. The prover runs the same check on the same shape.

**Leaves at `S = t + 1`.** Polynomials of degree below `S >= t + 1` interpolate any `t + 1`
values, and the symbol at an unopened point is independent of the opened ones. The fold
vector reveals one linear combination of the supports. When two messages carry nonzero
weight in it, every unopened symbol of every message stays uniform given the view and every
data cell. Every pool has a data slot and a blind slot per table, whose weights are nonzero
polynomials in the challenges; the condition fails with probability below 2^-118 on every
example.

**Reconstruction.** At toy scale (`c = 4`, `W = 32`, `t = 6`, a table of 2^6 rows with two
buses), the helper tree's leaves and root rebuild exactly from the witness and the six
opened symbols at `S = t`, and a flipped witness bit misses. At `S = t + 1`, and at `S = t`
with one repeated draw, two supports agreeing on every opened symbol give two different
roots.

## What correctness cost

One support cell per message. The fold vector grows by one element, 16 bytes on the wire
per proof under the pooled commitment. Each message grows by one cell in 4,384 at the
narrowest deployed grid, `c = 12`. The code length does not move at any deployed split, and
the query term moves by under 0.1 bit.

## What changed for circuit authors

**Nothing in the API.** `Config::prod()` carries the new support size.

**Custom configurations.** A `Config` with `ldt_support_size <= num_queries` now fails on
both sides with `ldt_support_size (N) must be > num_queries (M)` whenever
`min_security_bits > 0`. `Config::dev()` waives the floor and proves without zero knowledge.

**Everything re-proves.** The support size is transcript-bound and it moved.

## What this release does not promise

**Published proofs.** Upgrading does not reach a proof published under 0.35.0 or 0.36.0.

**Zero knowledge at a zero security floor.** `min_security_bits = 0` skips the support
check, also with zero knowledge on.

**A complete zero-knowledge proof.** The leaves are covered by the argument above. The
linear part of the view rests on the blinds and on the outer argument, whose zero knowledge
is argued separately, and this postmortem revisits neither.

**Independent audit.** This workspace has not had one.

## For other implementers

**Count every symbol the verifier can recompute.** A support of `S` cells hides `t` opened
symbols at `S = t`. An unsalted leaf of an unopened column is in the view too, through the
root and the multiproof, and hides only when each message keeps a free cell after the
openings: `S >= t + 1`, or a salt per leaf.

**A tree without blind rows has only its support.** Helper and lookup columns are functions
of the witness and public challenges, and their tree inherits nothing from the trace tree's
blinds.

**Repeated draws are no margin.** Exposure needs distinct draws, and at large widths they
are the common case: 73 percent of proofs at `W = 2^17`.

**A multiproof ships digests of unopened leaves.** A guess can be tested against any one of
them.

**A spare cell or a salt per leaf.** Salts cost `t` per opening. The spare cell costs one
element of the fold vector, and we took it.

## Scope

The previous postmortems are at [postmortem-0.36](postmortem-0.36.md),
[postmortem-0.35](postmortem-0.35.md), [postmortem-0.34](postmortem-0.34.md),
[postmortem-0.33](postmortem-0.33.md) and [postmortem-0.32](postmortem-0.32.md).

This one covers `hekate-core`, `hekate-program`, `hekate-crypto` and `hekate-verifier`, all
open. The prover reaches this workspace as a signed shared library built from a closed
repository. The facts used here about its commitment, a support slot in every message and a
helper tree without a blind column, are visible through the opening format the verifier
enforces: every opened helper column parses as exactly its symbols. The exposure figures
come from 0.36.0, its pinned prover release 0.14.0, and a verifier that additionally logs
each table's geometry and distinct draw count.