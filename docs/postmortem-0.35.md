# The estimate that chose its own geometry: a soundness postmortem

Hekate accepted proofs at a 110-bit floor and logged figures just above it. Both
terms feeding those figures were optimistic: the query term read 131 to 173 per
table where the theorems it cited give 57 to 73, and the term beside it was
missing two factors worth 4 to 7 bits.

No forgery is known that exploits this. Every value the verifier computed for
this argument was computed correctly, and a proof that verified was a real proof
of a real statement. What was wrong is the number attached to the answer, and in
a proving system that number is the product.

0.35.0 restates the claim at 100 bits that follow from published theorems, and
pays 287 column queries for them instead of 176.

## Summary

| # | Gap                                                     | Effect                                                        |
|:--|:--------------------------------------------------------|:--------------------------------------------------------------|
| 1 | Column queries priced at capacity, not unique decoding  | Query term read 131 to 173 where the papers give 57 to 73     |
| 2 | The same formula selected the encoding geometry         | Overstating picked the cheaper encoding, then graded it safe  |
| 3 | The proximity term omitted its tensor and curve factors | 4 to 7 bits unaccounted, worst on wide tables                 |
| 4 | The LogUp challenge term was never counted              | A term reaching 102 bits at 2^26 rows appeared in no estimate |
| 5 | The reported figure covered one table and one argument  | Chiplet tables and the outer argument sat outside the number  |

Gap 1 is the arithmetic. Gap 2 is why it was not caught: the formula that
reported security also chose the geometry it reported on, which cannot
contradict itself. Gaps 3, 4 and 5 are terms missing from the sum.

## Affected versions

| Running            | Exposed to                                         | Action                          |
|:-------------------|:---------------------------------------------------|:--------------------------------|
| 0.34.0 and earlier | 1, 2, 3, 4, 5                                      | Upgrade to 0.35.0 and re-prove. |
| 0.35.0             | 100 bits proven; 128 needs a wider challenge field | Current.                        |

No proof crosses this boundary. `num_queries`, `ldt_support_size` and
`outer_queries` are absorbed into the transcript before any challenge and all
three moved, which means a 0.34.0 proof fails transcript replay at the first
challenge rather than somewhere subtle. The proof wire format moves v5 to v6 and
the bundle format v4 to v5, carried by the proof-size work shipping alongside.
The pinned prover release is 0.13.0.

The second row states a ceiling, not an absence. Checked: every term we can
identify is now computed, charged, and logged per table on both sides. Not
checked: whether a term exists that we have not identified. Four postmortems in
this series are evidence that the second category is never empty.

## Where this one sits

0.32.0 was about values reaching the verdict with nothing binding them to the
committed witness. 0.33.0 was about committed columns no constraint policed.
0.34.0 was about the frame: who decides which rows are live, and which circuit is
being verified. All three are questions about the protocol.

This one is about the measurement of the protocol. The earlier three each had a
witness, a column or a validator you could point at. This one had a function
returning a number nothing in the system could contradict, because the number and
the thing it measured came out of the same expression.

> A security parameter is a claim about the worst case. If the function that
> reports it also selects the configuration it reports on, the report cannot
> fail, and its agreement with itself is not evidence.

The tell is mechanical. If a threshold on a security estimate appears where a
performance or layout decision is made, the estimate has stopped being a
measurement.

## 1. Column queries were priced at capacity

These commitments are checked by opening `t` random columns of an encoded matrix.
Security is `t` times the negative log of the per-query pass probability of a
cheating prover, and the question is which pass probability the papers license.

`Config::ldt_bits` computed `t · log2(1 / rho)` with `rho = msg_len / code_width`,
pricing each query at `rho`, which is **1.0 bit per query at rate 1/2**. That is
list-decoding-capacity pricing, and the citation attached to it pointed at a
section of Brakedown bounding a different quantity.

What the literature gives, unanimously:

| Source                                           | Statement                                                        |
|:-------------------------------------------------|:-----------------------------------------------------------------|
| Brakedown (Golovnev et al. 2022), App. B Lemma 4 | Decoding radius capped at `(1 - rho) / 2`                        |
| Brakedown, App. B Theorem 7                      | Per-query term `(rho + delta)`, minimised at `(1 + rho) / 2`     |
| Ligero journal version, Lemma 4.6                | Same radius                                                      |
| Ligerito (Novakovic and Angeris, 2025/1187)      | `ceil(-(lambda + log l) / log((1 + rho) / 2))` queries           |
| Diamond and Gruen 2024/1351, Corollary 3.7       | Tensor-fold proximity gaps at exactly the unique-decoding radius |

At rate 1/2 the correct term is **0.4150 bits per query**, not 1.0. Brakedown's
own concrete parameters follow their theorem: 309 queries at rate 1/2 and 189 at
rate 1/4 for 128 bits, each exactly `ceil(128 / -log2((1 + rho) / 2))`. Nobody in
this family sizes queries from `rho` alone.

Shipped tables at the old `t = 176`:

| Table                  |    rho | Query term logged | By the papers | Overstated |
|:-----------------------|-------:|------------------:|--------------:|-----------:|
| Keccak 2^15 main       | 0.5122 |               170 |        **71** |      2.39x |
| ML-DSA chiplet         | 0.5977 |               131 |        **57** |      2.30x |
| ML-DSA main, full-half | 0.5000 |               176 |        **73** |      2.41x |

**Reach.** Every table in every release of this series. The inner commitment is
the only place column queries are priced.

**Fix.** One line, the shape the outer argument already used:

```rust
let log2_ratio = log2_ratio_fixed(2 * code_width as u128, (code_width + msg_len) as u128);
```

## 2. The formula also chose the geometry

Each table picks one of two encodings. A fractional geometry carries a fixed
support block at rate 1/2. A full-half geometry doubles the codeword width, is
safer, and costs twice the encoded matrix. The choice was made like this:

```rust
if frac.support_size <= grid_cols
    && self.ldt_bits(frac_msg, frac.encoded_width) >= FRACTIONAL_MODE_BITS
```

`FRACTIONAL_MODE_BITS` was 128. The overstated `ldt_bits` therefore did two jobs:
it reported a table's security, and it decided whether that table got the cheap
encoding. Overstating by 2.4x meant every table cleared a 128-bit gate it could
not really clear, took the cheaper geometry, and was then graded safe in it by
the same overstating function.

The smallest fractional table cleared that gate by two bits, 130 against 128.
Under the correct formula it needs 396 queries to stay there.

This is also why the correction could not land alone. Fixing the formula with `t`
at 176 drops every fractional table to full-half, doubling every encoded matrix,
a measured 36 to 50 percent increase in peak memory on an edge prover. There is
no intermediate state where the formula is right and the system is shippable, and
no "measure the formula first" step that avoids confounding a query cost with a
geometry change.

**Fix.** The gate now reads `MIN_PRODUCTION_BITS`, the floor it actually serves,
and the formula and the query bump land together. A memory decision gated on a
security formula, in that formula's units, is an accident waiting for the units
to change, and the units changed.

## 3. The proximity term was missing two factors

Beside the query term sits an additive term: the probability that the random fold
is unrepresentative. It was `field_bits - log2(width)`, the single-line term from
BCIKS20, which omits two factors this protocol incurs:

- the eq-tensor factor over the grid's rows, charged by Diamond and Gruen as
  `theta` for a tensor over `2^theta` rows;
- the curve factor over the `eta` walk, BCIKS20 Theorem 1.5, charging the number
  of units combined.

```
field_bits - log2(width) - log2(log2(rows) + units - 1)
```

Worth 4 to 7 bits. A wide table at 80 units and 2^7 rows drops from 110 to 103.6.
Every rounding is a ceiling against us, and the arithmetic is integer only,
because both sides must derive identical geometry and `powf` is not
bit-reproducible across platforms.

This term is why large traces need an adaptive grid. It is 104.2 bits at a
codeword of 2^21 and falls about a bit per doubling, reaching 100 near 2^25.
`RingSwitchPlan::split_vars` now scans down from the proof-size optimum and takes
the widest grid still clearing the floor. Narrowing buys proximity bits and costs
proof bytes, which is a trade to make automatically rather than discover in
production.

## 4. The LogUp challenge term was never counted

Tables are joined by a bus argument whose soundness rests on a global challenge
`gamma` avoiding every `gamma + key[i]` denominator across every bus row in the
proof. That is `field_bits - log2(bus rows)`: 108 bits at 2^20 rows, 102 at 2^26.

It appeared in no estimate and no check. `Config::check_logup_security` now
computes it from the program's bus counts and the committed table heights, and
both sides run it before `gamma` is drawn. The prover runs it before any
commitment work, turning a rejection that cost a full proving run into one that
costs milliseconds.

## 5. The reported figure covered one table and one argument

The old `System Security` line reported one number for the main table's inner
argument. A proof has one main table, up to seven chiplet tables, and an outer
argument, each with its own geometry and its own numbers.

Per-table security is now logged on both sides. The difference is not cosmetic.
On a current ML-KEM-768 proof:

```
main table security       bits=107  ldt_query=119  fold_gap=112  logup_gamma=107
table security chiplet=1  bits=100  ldt_query=100  fold_gap=110
table security chiplet=3  bits=110
```

The main table reports 107. The proof is worth 100, set by one chiplet, with no
margin. Anyone reading the old single line would have reported the first number.

## How we know the old number was wrong

Two pricings differing by 99 bits at `t = 176` differ by only 10x to 25x in
acceptance at `t = 8`, which is why no existing test separated them and why the
question had to be settled by measurement rather than by reading.

`hekate/tests/mixture_proximity.rs` builds the shipped encode, reconstructs its
evaluation domain, and measures the per-query pass rate of two cheating
strategies over 200,000 query sets:

- **chosen**: commit a real codeword, send a wrong fold. Passes at `rho`.
- **mixture**: commit a word `delta`-far from the code and open it as the far
  codeword, corruption placed at the crossing point of `1 - delta` and
  `rho + delta`. Passes at `(1 + rho) / 2`.

| Table                  |    rho | Strategy | Measured |  Bound | Accept at t=8 |
|:-----------------------|-------:|:---------|---------:|-------:|--------------:|
| Keccak 2^15 main       | 0.5122 | chosen   |   0.5121 | 0.5122 |         0.48% |
| Keccak 2^15 main       | 0.5122 | mixture  |   0.7560 | 0.7561 |    **10.60%** |
| ML-DSA chiplet         | 0.5977 | chosen   |   0.5972 | 0.5977 |         1.63% |
| ML-DSA chiplet         | 0.5977 | mixture  |   0.7983 | 0.7988 |    **16.40%** |
| ML-DSA main, full-half | 0.5000 | chosen   |   0.4995 | 0.5000 |         0.39% |
| ML-DSA main, full-half | 0.5000 | mixture  |   0.7495 | 0.7500 |     **9.87%** |

Both columns reproduce their bound to three decimals. The decision rule was fixed
before the run: acceptance near 10 percent at `t = 8` settles it.

Two properties of that probe matter more than its result, because they are how a
measurement like this fails silently:

**The worst case has to be constructed.** The probe's first version used two
independent random messages and measured agreement of 0.0000, because two random
codewords coincide nowhere. Reaching either bound needs the deviation to be a
minimum-weight codeword. A guard built on random deviations passes forever while
measuring nothing.

**The probe has to validate its own domain.** It asserts each subspace vanishing
polynomial has exactly the zero set the shipped encode implies. Without that it
could faithfully measure a code that is not the one being shipped.

## Why 100 bits and not 128

128 proven bits are not reachable over GF(2^128) at any query count. Every
additive term has the form `n / |F|` and sits at `128 - log2(n)`: the fold
proximity term, the LogUp challenge term and the outer field term all lose bits
to the size of the object they range over, and queries shrink none of them.
Reaching 128 needs challenges from a 256-bit field plus roughly 309 queries, and
costs about 2.25x in proof size on a 2^24 trace.

Where the deployed field sits:

| System                         | Level | Basis                                                     |
|:-------------------------------|:------|:----------------------------------------------------------|
| SP1 Hypercube                  | 100   | Proven, unique decoding (16 of the 100 are grinding bits) |
| SP1 Turbo, Plonky3 example FRI | 100   | Conjectured                                               |
| Ligerito                       | 100   | Proven, unique decoding, 148 queries at rate 1/4          |
| Ethereum Foundation zkEVM      | 100   | Provable target for M2, 128 provable for M3               |
| Halo2 / Zcash                  | ~126  | Discrete log, not post-quantum                            |

Binius64 sets `lambda = 128` over GF(2^128) and writes the total as
`O(l)/|K| + 2^-lambda`. Its fold's proximity error is additive and queries cannot
shrink it, landing near 2^-100 or worse at large codewords, which makes that 128
a parameter rather than an end-to-end bound. Its FRI code prices queries at the
unique-decoding radius, as does Plonky3's binary-tower path, whose security
module opens by stating that only unique decoding is supported.

A larger radius is not available to us:

| Route          | Why it is unavailable                                                                                                                                 |
|:---------------|:------------------------------------------------------------------------------------------------------------------------------------------------------|
| Johnson radius | Buys about 10 percent per query, loses more in the batching term: 2^-89.8 against unique decoding's 2^-104.2 on our geometry, and we take the minimum |
| Capacity       | Refuted in 2025, with counterexamples built over characteristic-2 fields                                                                              |
| STIR, WHIR     | Fold over multiplicative cosets of 2-power order; `\|F*\| = 2^k - 1` is odd in every binary field                                                     |

For calibration, an unconditional floor from Elias list-decoding capacity puts any
rate-1/2 scheme of this shape at 103 queries minimum, whatever is eventually
proved. At 287 we are within 2.8x of a bound nobody goes below.

## What correctness cost

287 is the binding number, and the derivation is worth stating because the obvious
answer is wrong twice.

The support block hiding opened columns must be at least as large as the query
count. Raising `t` therefore widens every message, worsening `rho`, lowering the
bits each query is worth. Sweeping `t` from 200 to 1200, the smallest fractional
grid peaks at 98 bits and never reaches 100 at any query count. It changes mode,
the binding grid moves up one doubling, and that grid needs exactly 287.

Lower is worse, not cheaper. The unconstrained minimum for 100 bits is 241
queries, at which every grid falls to full-half and every encoded matrix doubles.
Query count and memory move in opposite directions: minimising queries maximises
memory.

Proof size goes as `sqrt(t)`, not `t`, because the grid split re-optimises against
it. A 63 percent query increase therefore costs about 25 percent, not 5x.

| Workload       | 0.34.0 proof, ZK | 0.35.0 proof, ZK | Change |
|:---------------|-----------------:|-----------------:|-------:|
| ML-KEM-768     |        2,732 KiB |        3,483 KiB | +27.5% |
| ML-DSA-44      |        3,328 KiB |        4,184 KiB | +25.7% |
| ML-DSA-65      |        3,338 KiB |        4,193 KiB | +25.6% |
| ML-DSA-87      |        4,280 KiB |        5,484 KiB | +28.1% |
| AES-128        |        3,726 KiB |        4,692 KiB | +25.9% |
| AES-256        |        3,923 KiB |        4,982 KiB | +27.0% |
| Keccak 2^15    |          759 KiB |          864 KiB | +13.8% |
| Keccak 2^20    |        2,735 KiB |        3,536 KiB | +29.3% |
| Fibonacci 2^20 |          962 KiB |        1,145 KiB | +19.0% |
| Fibonacci 2^24 |        3,369 KiB |        4,100 KiB | +21.7% |

Prove time and peak memory land inside the benchmark noise floors, 5 and 10
percent, on every workload. That is what choosing 287 over the unconstrained 241
bought: the geometry did not move.

A 2^26 Fibonacci trace is absent from that table. Earlier in this release cycle
the proof-size work widened long tables by one doubling, which pushed that trace
below the then-current floor: it rejected with
`SecurityTooLow { estimated_bits: 109, min_bits: 110 }`. The honest proximity term
and the adaptive grid restored it, and it verifies at 100 bits today. Its figure
in the 0.34.0 tables was measured under a different geometry and is not a
like-for-like baseline.

## What changed for circuit authors

**Minimum table height is 512 rows.** A table needs at least as many rows as there
are queries, and 287 rounds up to 2^9. A shorter table aborts with
`ldt_support_size (256) must be >= num_queries (287)`. One shipped example needed
its tables raised.

**The outer statement gained capacity.** Lowering outer queries from 155 to 121
raises the message length, pushing the cap on multiplication wires from 173,098 to
11,183,509, a 64x gain.

**Everything re-proves.** The configuration is transcript-bound and all of it
moved.

## What this release does not promise

**128 bits.** It needs a wider challenge field, which is a protocol change rather
than a parameter change.

**Grinding.** We do not use proof-of-work to buy query bits, and the reference
point we cite for 100 proven bits does. The 1:1 subtraction is a theorem about
probability, not about work: an adversary guesses one nonce, passes a `g`-bit
grind with probability `2^-g` at constant cost, and nothing in the grind touches
the query term. The average-case reading behind the folklore is real but is a
weaker claim than a query bound, and Grover halves it, which matters for a
post-quantum-positioned system. If we adopt it, those bits will be reported
separately.

**Terms we have not identified.** Five terms were wrong or missing and we found
them. How many terms a soundness argument has is not something the argument tells
you.

**Circuit semantics and determination of published cells.** Unchanged from 0.34.0.

**Independent audit.** This workspace has not had one.

## For other implementers

**Name the radius your query count is priced at.** Capacity, Johnson and unique
decoding differ by more than 2x in bits per query, and the difference is invisible
in any test small enough to run: at eight queries two pricings differ by a factor,
at 176 by 99 bits. Write the per-query term next to its citation, and check the
citation bounds the quantity you are using it for.

**A number that gates a decision cannot grade that decision.** If a security
estimate appears in a threshold selecting geometry, rate or layout, it is part of
the machine and its agreement with itself is worth nothing. Keep the reported
figure and any configuration gate on separate expressions.

**Sum every term, not the one you tuned.** Ours spread across a query bound, an
additive term missing two factors, a bus challenge term counted nowhere, and a log
line covering one table out of eight. Each was individually plausible. A soundness
figure is a minimum over terms, and a term you never wrote down is not
conservatively estimated, it is absent.

**Make the adversary the worst case, not a random one.** The measurement that
settled this was nearly useless: random deviations gave a pass rate of zero and
would have passed as a guard forever. The bound is attained only by a constructed
minimum-weight deviation. A probe validating a soundness claim must also validate
that it measures the shipped object.

## Scope

This covers `hekate-core`, `hekate-program` and `hekate-verifier`, all open, and
the matching parameter derivation in the prover. The prover reaches this workspace
as a signed shared library built from a closed repository. The security accounting
lives entirely in the open crates, both sides derive identical geometry from the
same functions, and the measurement above ships in the public test suite.

Previous postmortems: [0.34](postmortem-0.34.md), [0.33](postmortem-0.33.md),
[0.32](postmortem-0.32.md). The first asked what binds each value entering the
verdict. The second asked whether the witnesses satisfying a circuit are the ones
it meant to admit. The third asked who decides. This one asks what the answer is
worth, and answers that a security level is a claim like any other, and a claim
that cannot fail its own check has not been checked.