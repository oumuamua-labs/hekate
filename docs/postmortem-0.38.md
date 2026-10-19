# Two checks on one weight: a soundness postmortem of our logUp implementation

In every release from 0.29.1 to 0.37.0, a cheating prover could make Hekate's verifier accept a
proof that two tables agree when they do not, every time, without guessing. Each bus between
tables is checked by two equations, and the verifier folded both into its final check under the
same random weight. An error in one equation could then pay for an error in the other. 0.38.0
gives the two their own weights, and proofs keep their size.

For readers new to this series: Hekate is a zero-knowledge proof system over binary fields. It
splits a computation into tables, such as a CPU, a memory, or a unit that computes Keccak, and
checks each table on its own. Tables exchange values over buses. A CPU row that reads memory emits
a message carrying the address and the value it got, and a memory row emits the same message from
its side. The proof has to show that every message sent on a bus is received.
[logUp buses](https://oumuamua.dev/hekate/docs/basics/logup-buses) covers the mechanism in depth.

## Summary

| # | Gap                                                                           | Effect                                                                       | Evidence                                                      |
|:--|:------------------------------------------------------------------------------|:-----------------------------------------------------------------------------|:--------------------------------------------------------------|
| 1 | The consistency check and the bus-sum check of each bus shared one power of α | A constant consistency miss, added to the claimed sum, cancels any imbalance | Forged on the 0.37.0 weights, with and without zero knowledge |

## Affected versions

| Running          | Exposed to                                       | Action                          |
|:-----------------|:-------------------------------------------------|:--------------------------------|
| 0.29.1 to 0.37.0 | 1, on every bus, with and without zero knowledge | Upgrade to 0.38.0 and re-prove. |
| 0.38.0           | None known                                       | Current.                        |

0.38.0 changes the bus weights, and a separate change to Merkle openings moves the proof wire to v8.
No proof crosses this boundary.

Upgrading protects new proofs only. A verifier from 0.29.1 to 0.37.0 accepts forged bus relations,
and a proof it accepted is no evidence of what its buses carried. A statement that matters should be
proved again under 0.38.0.

## Two checks per bus

logUp turns "every message sent is received" into arithmetic. The verifier draws a random `γ`, and
each message `m` on one side of a bus adds the fraction `1 / (γ + m)` to that side's sum. Two sides
holding the same messages produce the same sum, and a message missing on one side changes it for
all but a negligible set of `γ`. In Hekate's field a number added to itself gives zero, and the
check becomes: the sums reported by the sides of a bus add up to zero.

The verifier cannot add up a million fractions itself. The prover commits a helper column `h` with
one fraction per row, `h = s / (γ + key)`, where `key` is the row's message and `s` is 1 on a row
that emits. Two claims about `h` then carry the bus:

1. **Consistency.** Every row satisfies `(γ + key) · h = s`: each `h` is the right fraction.
2. **Bus sum.** The values of `h`, each times `eq_lookup`, which is 1 on a permutation bus and
   `eq(r_bus, x)` for a second random point `r_bus` on a lookup bus, add up to the sum the prover
   reports for its side of the bus.

A table carries many other checks, from its constraints, its boundary values and its blinding
terms. The verifier does not run them one at a time. It draws a random `α` and folds them
into one equation, the first check times 1, the second times `α`, the third times `α²`, one power
per check. A nonzero polynomial in `α` has few roots, and one wrong check makes the folded equation
fail for all but a negligible set of `α`. That guarantee holds while no two checks share a power.

## What went wrong

logUp itself is sound. Hekate gave the consistency check and the bus-sum check of a bus one power of `α`.

Picture a cashier who counts two tills into one pile. A shortfall in one till and an equal surplus
in the other give the right pile, and the cashier never sees either. With one shared weight, the
verifier counted the consistency check and the bus-sum check of a bus into one pile.

The prover makes the shortfall the same on every row. It commits `h = (s + e) / (γ + key)` for a
constant `e` of its choosing, and every row then misses the consistency equation by exactly `e`.
A wrong `h` usually shows up at the verifier's random point as a random-looking error. A miss of
exactly `e` on every row shows up as `e` at every point, random or not. The prover adds the same
`e` to the bus's reported sum, and the pile comes out right.

On a bus whose sides disagree, the prover picks the `e` that makes the reported sums add up to
zero, and the cross-bus check passes. Nothing else catches it. The commitment to `h` is real, the
verifier opens it at its random point, and the opened value is the true one. A lookup bus weights
its sum with a second random point, drawn before `h` is committed, and the prover knows it when it
picks `e`.

We ran the forgery on a toy bus between two columns with one wrong message, and the verifier
accepted it with and without zero knowledge. The algebra is the same on every bus. In Hekate that
covers every relation that crosses one, from memory reads against writes and lookups to calls into
the hash, cipher, ML-KEM and ML-DSA tables.

## Where this one sits

The 0.32.0 postmortem fixed an earlier hole in the same check: `h` was not committed at all, and a
prover could choose its value after seeing the verifier's random point. That postmortem printed the
formula of the bus term, and the formula already shows this bug. It describes the bus term as "affine in
`h_eval` with coefficient `alpha_bus · ((γ + key)·eq_zc + eq_lookup)`" (postmortem-0.32, section 2).
`alpha_bus` is one weight across the consistency part, `(γ + key)·eq_zc`, and the bus-sum part,
`eq_lookup`.

0.32.0 bound `h` to a commitment and kept the one weight. Committing `h` stops a prover from fitting
its value after the challenge. The offset is fixed with `h`, before `α` and the random point exist,
and reads the same at every point. The shared weight predates the first tag, 0.29.1.

## How we found out

Most of the earlier postmortems came out of targeted security audits of Hekate. This one arrived
unplanned. We were working on another system that builds on Hekate's core, and its verifier replays
Hekate's logUp check through `hekate_verifier::logup::expected_bus_contribution`. A security review
of that system found the shared weight on 17 October 2026 and confirmed the forgery end to end
against its verifier. We reproduced it the same day against Hekate's own verifier, with a
proof-of-concept prover, with and without zero knowledge.

## The fix

Each bus now takes two consecutive powers of `α`:

| Term of bus `k`                                | Weight       |
|:-----------------------------------------------|:-------------|
| consistency, `eq(r_zc, x) · (γ·h + h·key + s)` | `α^(o+2k)`   |
| bus sum, `h · eq_lookup`                       | `α^(o+2k+1)` |
| claimed sum, in the initial claim              | `α^(o+2k+1)` |

`o` counts the constraint roots, boundary constraints and blinding units ahead of the buses. With
separate powers the folded equation splits into one equation per check. The consistency
equation must vanish at the random point `r_zc`, which a nonzero residual does with probability at
most `num_vars / 2^128`, and the bus-sum equation pins the reported sum to the honest one.

One function defines the weights, `permutation::bus_alpha_weights` in `hekate-program`. The
verifier's ZeroCheck and the zero-knowledge outer statement take every bus weight from it.

## How the fix is checked

**The forgery.** The proof-of-concept prover's forged proof is rejected with and without zero
knowledge. Merge the two powers back into one inside `bus_alpha_weights`, and the forged proof
verifies again in both modes.

**A control.** The forging prover runs its own sumcheck. On an honest bus that sumcheck produces
proofs that verify in both modes, and the rejection of the forgery comes from the weights.

**The schedule.** `bus_alpha_weights_take_consecutive_powers` in
`hekate-program/src/permutation/mod.rs` checks that each bus takes two consecutive powers and that
the next bus starts two powers later. `expected_contribution_matches_formula` in
`hekate-verifier/src/logup.rs` gives the two terms different weights.

**Honest proofs.** The end-to-end and adversarial suites of the workspace and of every chiplet crate
pass under the new weights.

## What correctness cost

Nothing in proof size and no new challenge. The verifier spends two more field multiplications per
bus, three with zero knowledge.

## What changed for circuit authors

**Nothing in the circuit API.** AIRs, buses and `program_id` are unchanged. Other changes in 0.38.0
do reach circuits, and its release notes list them.

**Everything re-proves.** No proof crosses into 0.38.0.

**Verifiers outside this workspace.** A verifier that replays the ZeroCheck, or the outer statement
under zero knowledge, takes the weights in the table above. `logup::BusSpecEvaluation` in
`hekate-verifier` gains `alpha_cons` beside `alpha_bus`, and `permutation::bus_alpha_weights` yields
both.

## What this release does not promise

**Published proofs.** Upgrading does not reach a proof that a verifier from 0.29.1 to 0.37.0 has
already accepted.

**Independent audit.** This workspace has not had one.

## For other implementers

**Give every check its own power.** A random linear combination tells checks apart only when no two
of them share a coefficient. Write down the map from checks to powers, and test that it is one to
one.

**A constant looks the same at every point.** A ZeroCheck shows that a residual's multilinear
extension vanishes at a random point. A constant residual takes its value everywhere, and only a
target of zero that nothing else can move rejects it. If a value the prover reports shares the
residual's weight, the constant moves into that value.

**Re-derive the whole check when you fix one value in it.** The 0.32.0 fix bound `h` and kept the
formula around it, and that formula, printed in the postmortem of the fix, carried this bug.

**Read your checks through a consumer's verifier too.** Targeted audits of Hekate itself produced
most of the earlier postmortems and did not catch this one. A review of a system that replays
Hekate's check in its own verifier did.

## Scope

The previous postmortems are at [postmortem-0.37](postmortem-0.37.md),
[postmortem-0.36](postmortem-0.36.md), [postmortem-0.35](postmortem-0.35.md),
[postmortem-0.34](postmortem-0.34.md), [postmortem-0.33](postmortem-0.33.md) and
[postmortem-0.32](postmortem-0.32.md).

This one covers `hekate-program` and `hekate-verifier`, both open. The prover reaches this
workspace as a signed shared library built from a closed repository, and the weights it must use
are the ones the verifier enforces.