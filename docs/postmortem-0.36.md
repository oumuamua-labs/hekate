# Columns only the bus read: a soundness postmortem

Each of the last four releases shipped with a soundness postmortem, and 0.36.0 was
supposed to be the one that did not. It ships with this one. Through 0.35.0, any two
calls to the Keccak or AES chiplets in one proof could swap their answers, and the ML-KEM
and ML-DSA circuits proved almost nothing about the values they claimed to compute: an
ML-DSA proof verified with any signature hash its prover chose to publish.

Most of it comes down to one mechanism. A bus proves that the values at its two ends
agree. When no constraint reads the column at one end, that agreement binds nothing:
the column holds whatever the prover wrote, and the value it carries goes nowhere.

The first gap surfaced from a downstream project's report on the Keccak bus. The fix has
the verifier compute every call's clock itself. With that in place we went back to the
post-quantum circuits to fix gaps we had recorded in August and left open for two
releases. A full audit found far more than August had, and the post-quantum crate is
rebuilt from its tables up.

For readers new to this series: a Hekate program is a set of tables. A chiplet is a table
that serves calls from other tables, the way a coprocessor serves a CPU. The caller emits
each request on a bus, the chiplet emits the matching response, and the bus, a LogUp
multiset argument in characteristic two, balances when the keys emitted on its two sides
cancel in pairs. A committed column is one the prover writes; a fixed, or pinned, column
is one the program determines and the verifier recomputes.

## Summary

| # | Gap                                                        | Effect                                                                 | Against 0.35.0         |
|:--|:-----------------------------------------------------------|:-----------------------------------------------------------------------|:-----------------------|
| 1 | A call's response was not tied to its request              | Two Keccak or AES calls swap outputs; a squeeze reorders its blocks    | Forgery accepted       |
| 2 | AES keys were paired on a clock of their own               | A block encrypts under another call's key                              | Forgery accepted       |
| 3 | ML-DSA's published c̃ reached no check                     | The statement's only public value was decorative                       | Forgery accepted       |
| 4 | Memory carried bytes no constraint read                    | Every intermediate value in both circuits was free                     | Established by reading |
| 5 | Most of FIPS 203 and 204 ran only in the witness generator | Encoding, sampling, NTT wiring and the base-case product went unproved | Established by reading |

The forgeries for gaps 1 to 3 were run on 30 September against the 0.35.0 source and the
signed prover release it pins, 0.13.0, with zero knowledge off and on, and the verifier
accepted every one. Each forged witness satisfies every constraint and balances every bus
exactly. To the prover it is an ordinary witness for a false statement, and no challenge
or configuration can reject it. 0.36.0 rejects the same Keccak and AES shapes. The ML-DSA
circuit the third forgery ran against no longer exists.

Gaps 4 and 5 are established from the constraint lists. No forgery was built for them:
the decision to rebuild replaced the plan to confirm them one at a time.

Gaps 1 and 2 reach every AES-128 and AES-256 program and every program that calls Keccak
over its bus, including the sponges inside ML-KEM and ML-DSA. Services whose one emit
carries a call's inputs and outputs together were not affected: SHA-256, IntArith, modexp
and the RSA statement built on it, and the AES S-box table. Together, gaps 3 to 5 reach
every ML-KEM and ML-DSA statement proved with `hekate-pqc` through 0.35.0.

## Affected versions

| Running            | Exposed to                                                 | Action                          |
|:-------------------|:-----------------------------------------------------------|:--------------------------------|
| 0.35.0 and earlier | 1 to 5                                                     | Upgrade to 0.36.0 and re-prove. |
| 0.36.0             | Host emit order and circuit semantics stay with the author | Current.                        |

Every program that embeds Keccak, AES, IntArith, SHA-256, modexp, RAM or a post-quantum
chiplet gets a new `program_id`, and its proofs must be made again. The proof wire format
(v6) and the transcript do not move. The bundle format moves from 5 to 7 in two steps, 6
for a chiplet-composition change shipping alongside and 7 for the rank clock below; no
release carried format 6. The pinned prover release is 0.14.0. Any deployment verifying
Keccak, AES, ML-KEM or ML-DSA statements has to upgrade.

The second row states a limit, as this series has since 0.33.0. Checked: every
call-and-return bus pairs request `t` with response `t` through a clock the verifier
computes, and every post-quantum value travels from the row that produces it to the row
that consumes it. Not checked: whether a host emits its requests in the order it means,
and whether a host program computes what its name says. Both are about intent, which the
engine cannot see.

0.36.0 also wipes every chiplet trace and its witness buffers on drop, and removes
secret-dependent branches from several witness generators. That work closes none of the
gaps here.

## Where this one sits

0.33.0 was about committed columns no constraint policed and bus keys too poor to tell two
emissions apart. This one returns to that ground, and 0.33.0 already contains it. Its
first lesson for other implementers ends: "A committed column that no root reads is an
input from the adversary." Its section on the Keccak bus key says: "The chiplet proved
its emitted multiset was closed under the 24-round map. It did not prove that the
consumer's output was the image of the consumer's input."

The Keccak clock was a committed column no root read, and the second sentence stayed true
after the fix that followed it. That fix added a bit to the key telling the two ends of
one block apart. The clock column meant to tie a block to a call stayed free in both
chiplets, and 0.33.0 had named it as the lever of a forgery twice, once in Keccak and
once in AES. Each time the fix removed that one use of the lever.

The post-quantum circuits are the same lesson at the scale of a crate. Their memory bus
carried value bytes that no constraint read, beside a packed value that every constraint
read and nothing tied to the bytes.

## 1. A call's response was not tied to its request

A Keccak call spans two rows of its host: the input state on one, the output state 24
rows later. The chiplet answers with the two ends of one 25-row block, and the bus pairs
the four emits. In 0.35.0 each emit's key was 25 lanes, a clock and a direction bit.

The host's clock was its own row index, which the verifier computes. The chiplet's clock
was `REQUEST_IDX`, a committed column. No constraint of the chiplet read it and no fixed
column pinned it; the trace generator filled it from pairs of row indices the caller
passed in. The bus is a characteristic-two multiset check. The host's keys are distinct
by row index, and the chiplet only had to produce the same multiset, which it could,
because each of its emits could take whichever clock matched.

The chiplet's round constraints tie a block's output to the same block's input. Nothing
tied a block to a call:

```
 host emits                       chiplet emits
 (I0,    clk 0,  input)           (I0,    REQUEST_IDX 0,  input)    block 0
 (f(I1), clk 24, output)          (f(I0), REQUEST_IDX 49, output)   block 0
 (I1,    clk 25, input)           (I1,    REQUEST_IDX 25, input)    block 1
 (f(I0), clk 49, output)          (f(I1), REQUEST_IDX 24, output)   block 1
```

Same multiset, every constraint satisfied, and the host's first call reads keccak-f of
the second call's input.

A single sponge is exposed as well. A sponge squeezing three blocks binds each call's
input to the previous call's output. With `S1 = f(S0)`, `S2 = f(S1)` and `S3 = f(S2)`, a
host claiming the output blocks `S2, S1, S3` makes the calls `(S0 → S2)`, `(S2 → S1)` and
`(S1 → S3)`. Their inputs are `{S0, S2, S1}` and their outputs `{S2, S1, S3}`, exactly
the multisets the chiplet's three honest blocks emit. The chain binding holds on every
row, and the first two output blocks trade places.

AES's data bus had the same shape. A block's input state and its ciphertext are two
emits, at offsets 0 and 10 of its 11 rows, 0 and 14 of 15 on AES-256, each clocked by
`REQUEST_IDX_LINK`, which no constraint read. Two AES calls swap ciphertexts the way two
Keccak calls swap digests.

**Reach.** Every AES program, and every program that calls Keccak over its bus. The
ML-KEM and ML-DSA control tables built their Keccak requests by hand in the same shape,
which puts every sponge call in both circuits in reach. A service whose one emit carries
a call's inputs and outputs together cannot be split this way: whichever clock its
responder writes, the emit it matches belongs to its own call.

**The check that passed it.** The build-time validator requires a clock on every
permutation bus. It counted `REQUEST_IDX` as one because of its label: a committed column
whose label marked it as a request index satisfied the check, and the label was the
whole test.

**Confirmed.** Both forgeries ran against the 0.35.0 source and its pinned prover 0.13.0
(sha256 `96417510…`, checked against the release manifest by the 0.35.0 build), under
the production configuration at 512 rows, with zero knowledge off and on. Every block
order was accepted: both for the swap, all six for the squeeze. The honest pairings
verify under the same harness.

## 2. AES keys were paired on a clock of their own

An AES block takes its key on a second bus, one emit per block on its input row, clocked
by a second committed column, `REQUEST_IDX_KEY`, which no constraint read either. The data
pairing and the key pairing were independent of each other.

Take two calls with different keys and let the host swap the keys between them. Both
blocks run honestly under their own keys, the key bus sees the same two keys on each
side, and the verifier accepts. Each call now claims a ciphertext under a key that did
not produce it. A host that binds its key to anything else, a key exchange or a
commitment, proves a ciphertext under a key the binding never saw.

**Confirmed.** Accepted against 0.35.0 and prover 0.13.0 at both key sizes, in both
block orders, with zero knowledge off and on, together with the ciphertext swap of gap 1,
under the AES crate's own test configuration.

## The fix: the verifier counts the calls

0.36.0 takes the clock away from both sides.

`Source::EmitRank` gives the `t`-th emit of an endpoint the label `ω^(base + t)`, where
`t` counts the rows above it on which the endpoint's selector is 1. The selector is a
fixed column whose shape the verifier evaluates for itself, or it is absent. `ω` is an
element of GF(2^128) of prime order `p = 67,280,421,310,721`, about 2^45.9, a factor of
`2^64 + 1 = 274177 · p`; labels stay distinct while `base + t < p`. An endpoint's `base`
is the number of emits before it on its side of the bus, counted over the program's
endpoints in a fixed order, and the verifier derives it from the program and the
instance. Nothing in the proof or the bundle carries it. A bus keyed by this clock is an
ordered bus.

Within one side every label is distinct. The bus sum cancels only when each key meets an
equal key, and request `t` can meet only response `t`. A chiplet whose constraints tie
its emits `2b` and `2b + 1` into one block carries that tie to the host's emits `2b` and
`2b + 1`, one call. The parity of the rank says which end of the call an emit is, and the
direction bit is gone.

The verifier needs the clock's multilinear extension at the sumcheck point without a pass
over the rows, and `ω^(a + b) = ω^a · ω^b` makes that cheap. For a selector that fires on
every row it is a product over the row bits:

```
clock(r) = ω^base · Π_t ((1 + r_t) + r_t · ω^(2^t))
```

For a strided selector it is an automaton over the row bits whose state is the residue
modulo the stride, at `O(n · s)` per run for `n` row bits and stride `s`.

`HekateVerifier::verify` checks the shape of every ordered bus before any transcript
work, and the check is total and fails closed. Every endpoint of an ordered bus carries
exactly one rank source, at the same key position and with the same key width. An
ordered endpoint is a permutation endpoint with no receive selector. Its selector is
absent or a fixed column pinned to a strided, segmented, periodic or sparse schedule
whose values are 0 or 1. Each side emits fewer than `p` times.

One rule came out of review. The prover chooses the height of a chiplet table. An
endpoint whose selector repeats down the whole table emits as many times as the prover
likes, and every base after it on the same side moves with that count. An odd count ahead
of two-emit calls shifts each later call by one rank and pairs its request with the
output of one block and the input of the next. A chiplet endpoint whose emit count
follows its own height must be the last endpoint on its side, and the verifier rejects
any other order.

Every call-and-return bus in the workspace moved to the rank clock: the Keccak bus, the
AES data, key and S-box buses, IntArith, SHA-256, modexp and the post-quantum service
buses. Every responder clock column is deleted, Keccak also loses its direction column,
and the service type accepts rank clocks only. RAM answers in address order and its
consistency check needs real timestamps. It keeps its committed clock, which its sort
constraints force strictly increasing over active rows, on a raw bus spec under a waiver
citing those constraints. A label no longer makes a column a clock anywhere.

## What we knew about the post-quantum circuits, and when

On 27 August, during the work that became 0.34.0, a review of the ML-DSA control table
found that the published c̃ reached no check, that the rate lanes absorbing the message,
the public key and `w1` appeared in no constraint, and that memory addresses in both
post-quantum circuits were free. The note written that day concluded that a prover
satisfies the ML-DSA circuit without a signature or a key.

We did not patch. The address question was open, and a partial fix would have left most
of the statement outside the hash while reading as fixed. We deferred the disclosure
decision until the address question was answered, and it never was. 0.34.0 shipped on
10 September and 0.35.0 on 22 September, and neither release nor either postmortem said
a word about any of it. The 0.34.0 postmortem illustrated the limit of the engine with a
hypothetical "SHA3" circuit missing a round. Our own ML-DSA circuit was a live instance
of the same limit, and it shipped in that release.

What users were told in that window: the post-quantum crate's README opened with
"**Experimental.** This crate exists to demonstrate that Hekate can prove lattice-based
cryptography natively in binary fields." The same note ended: "Treat it as a working
example, not a production dependency." The workspace README listed "ML-DSA (Dilithium)
signature verification, ML-KEM (Kyber) decapsulation, AES-128/256, all proven natively in
binary fields", with performance tables for both post-quantum circuits. A performance row
claims that the circuit proves what the row names, and those rows did not.

0.32.0 and 0.33.0 set the standard for this series: a gap in a shipped circuit is
disclosed with its reach in the release that follows its discovery. Two releases went out
below it. This section is that disclosure, a month and two releases late.

The audit that followed, on 27 September, read every constraint list, pin and bus spec
in both circuits with the ordered buses in place. The August list was a small part of
what it found. Gaps 3 to 5 are the findings, grouped.

## 3. ML-DSA's published value reached no check

The ML-DSA statement published one value, the signature's challenge hash c̃. It traveled
from the host's public cells over the data bus into the control table, where one
constraint copied it into a memory value column on rows that were forbidden to touch
memory. That copy fed nothing. The comparison `c̃ = c̃′` compared the recomputed hash
against a column that only the comparison read.

The public value and the check never met. A proof said that some ML-DSA-shaped
computation had run on inputs the prover picked, and nothing about the value it
published. Given one valid signature from any key on any message, a prover publishes any
c̃ it likes.

**Confirmed.** Run against 0.35.0 and prover 0.13.0 at ML-DSA-65, under the
configuration its example ships with, zero knowledge off and on. The harness signs a
message with an independent ML-DSA implementation and builds the honest witness. It then
complements every word of the published c̃, all 48 bytes, in the public input, in the
host's cells and in the control table's copy, and leaves the rest of the witness honest.
The verifier accepts. Complementing the public input alone is rejected by the boundary
check, and complementing the host's cells without the control table's copy is rejected
by the bus. Both mechanisms worked. The value went nowhere after them.

Reading the constraints, the comparison could also be skipped: its row was not pinned,
and the flag recording that it had run had no anchor on row 0. We did not build that
forgery. The comparison also covered only 32 bytes of c̃, which FIPS 204 sets at 48 bytes
for ML-DSA-65 and 64 for ML-DSA-87. Neither mattered while the published value went
unread.

## 4. Memory carried bytes no constraint read

Both circuits routed every intermediate value through a general-purpose memory table. The
control table's bus to memory carried an address, four value bytes and a write flag.
Every constraint that used a value read a separate packed column, and no constraint tied
the packed column to the four bytes on the bus. The addresses appeared in no constraint
either.

Offline memory checking proves that each read returns the last write to its address.
Here it proved that about bytes the computation never used. Every hop through memory
delivered a free value to the next step, and a pinned address would not have helped: it
would have delivered a free value from the right address.

ML-DSA's hash input had the same shape one level down. The rate lanes of the rows
absorbing the message, the public key and `w1` appeared in no constraint. The bus meant
to bind them had selected on columns the trace generator never wrote. 0.34.0 deleted that
bus as inert and its postmortem reported the deletion, without saying what the bus had
been meant to bind.

## 5. Most of FIPS 203 and 204 ran in the witness generator

The witness generator computed FIPS 203 and 204 correctly. The constraints stated much
less:

- The base-case multiplication table proved `a + b ≡ c (mod q)` and no product. Its
  documentation said the products were checked by the NTT table. The witness generator
  sent that table only the diagonal products `a_j · b_j`, and those went nowhere. The
  cross terms and the multiplication by `ζ` were never computed in the circuit.
- Each NTT row proved one butterfly, and no constraint said which rows fed which. The
  routing labels were free, the flow bus balanced for any routing, and the twiddle
  factors came from a table whose values no constraint read.
- ByteDecode, ByteEncode, Compress, Decompress, SampleNTT, CBD, pkDecode, sigDecode and
  rejection sampling ran only in the witness generator.
- The ML-DSA norm check and the high-bits decomposition ran on bits that no constraint
  connected to the values on their buses.
- ML-KEM published only the ciphertext. The shared secret left on a bus no host
  published, and the decapsulation key, the encapsulation key and its hash were bound by
  nothing.

The norm-check bus carried a waiver reading "bus_idx is positional, both endpoints force
one row per (idx) value by AIR rather than by a clock slot", and the high-bits bus an
equivalent one. No constraint read `bus_idx`. In 0.33.0 we burned a waiver whose citation
pointed at a deleted file. These two made an argument no constraint backed. The build
checks that a waiver starts with `see ` and runs 32 characters, and both did.

## The rebuild

The August note had proposed replacing memory with a single-assignment bus. After the
September audit we rebuilt both circuits and every table under them except Keccak, on
four rules.

**A table is a function.** Each table proves `outputs = f(inputs)` for every call on its
service bus, the way the Keccak chiplet proves `out = keccak-f(in)`. Every input enters
on the bus and every output leaves on it. A table publishes nothing: the host program
decides what is public, what stays hidden and what is chained to what.

**No memory table.** Every value moves as a token from the row that produces it to the
row that consumes it, keyed by label, position and value. Labels and positions are pinned
at both ends, or forced onto their scheduled values where a table picks them at run time.
Each meaningful key has one producing slot and one consuming slot, and balance forces the
two to carry equal values. The one exception is SampleInBall, whose data-dependent access
pattern gets a sorted read-after-write argument over 256 cells inside the sampling table.

**Data-dependent routing keeps fixed counts.** Rejection sampling, SampleInBall and hint
decoding emit one token per fixed slot. A slot with nothing to carry emits a constant key
that no meaningful key equals, and the schedule keeps the count of those keys on each bus
even, which cancels them in characteristic two.

**Every coefficient is canonical.** Each producer range-checks the coefficients it emits
below `q`, and consumers rely on exact matching.

ML-DSA runs six tables: control, sampling, Keccak, encoding, NTT and high bits. ML-KEM
runs control, sampling, Keccak, encoding, NTT and a base-case multiplication table that
multiplies, plus a comparison table when a call decapsulates. Memory, the twiddle table,
the norm-check table and the old base-case table are gone from both. The circuits serve
the FIPS internal functions over an ordered service bus that trades 32-bit words with the
host: ML-KEM `KeyGen`, `Encaps` and `Decaps`, and ML-DSA `Verify`. Decapsulation returns
`K` and a validity bit, and `K` is the implicit-rejection key when the re-encryption check
fails. ML-DSA compares `c̃′` with `c̃` over the full length of `c̃`.

## Why the test suites were green

Every exploit test in the old post-quantum suites changed one cell of an honest witness
and asserted that something rejected it. None built a consistent forged witness. The c̃
experiment above could not be written in the old harness at all, because the harness
built the public input from the honest signature. In August we found six exploit tests
whose names described attacks they did not perform. One was named for a c̃ substitution;
it changed one cell and was caught by the sponge's carry chain. We renamed all six. The
c̃ attack stayed open until this release.

The Keccak and AES suites had no test that put two calls on one bus and asked whether
they could exchange answers. The control-language enumeration added in 0.33.0 walks the
transition relation with the data held at zero, and a bus key sits outside it, as that
postmortem stated. The single-cell mutation harness cannot express a swapped pair, which
needs coordinated clock values on two rows.

The suites also proved real signatures and real key exchanges, and passed. An honest
witness satisfies a base-case table that proves an addition as readily as one that
multiplies, and a suite of honest proofs tests completeness alone.

## How the fixes are checked

**Before and after.** The Keccak and AES forgeries that 0.35.0 accepts are rejected by
0.36.0 and prover 0.14.0, in every block order and at both AES key sizes, with zero
knowledge off and on (`hekate-keccak/tests/output_pairing.rs`, `chain_rotation.rs`,
`hekate-aes/tests/call_pairing.rs`). The 0.35.0 probes use the 0.35.0 API and are not
part of the 0.36.0 suite.

**The engine.** `hekate/tests/ordered_bus.rs` runs 15 end-to-end scenarios around a
synthetic squaring service behind dense and sparse hosts and a tiled responder. Honest
calls verify. Swapped responses, a rotated chain, two requester tables colliding and a
responder taller than its calls are rejected. Three hand-written twins of an honest
program, each with one malformed ordered bus, are rejected at verify entry; each twin is
pinned to its own `program_id`, which leaves the shape check as the only thing that can
reject it. The clock's multilinear extension is checked against a brute-force column for
every selector shape.

**The post-quantum pipelines.** ML-KEM `KeyGen`, `Encaps`, a valid `Decaps` and a
rejected one produce the same `ek`, `c` and `K`, the rejection key included, as the
`ml-kem` crate 0.3.2 at all three parameter sets. ML-DSA-44, -65 and -87 signatures made
by the `ml-dsa` crate 0.1.1 prove and verify, and at ML-DSA-44 a proof for a message the
signer never signed is rejected at the `c̃` comparison. Seventy forgery cases across the
pipelines and their tables each prove a false witness and require the verifier to reject
it. Most change one intermediate value in the witness generator and recompute everything
downstream, and most require that the named check is the only one that fails. With each
root under test removed by hand, the same witness passed preflight; no test repeats that
step. A census reads every labelled endpoint from the compiled definitions at all six
parameter sets and fails on an unpinned label, a key with no producer and a producer that
emits nothing.

## What correctness cost

Both columns below are published README figures: Apple M3 Max (16 cores),
`--release`, features `std parallel blake3 table-math`, `Config::prod()`, best of three
on an idle machine, peak memory as the larger of the peak physical footprint and the peak
resident set. The 0.35.0 column ran on prover 0.13.0 and the 0.36.0 column on 0.14.0.
They are two measurement sets taken on one machine. Cells read zero knowledge / base.

### The rank clock

| Workload                  | Proof 0.35.0        | Proof 0.36.0        |
|:--------------------------|:--------------------|:--------------------|
| Keccak-f[1600], 2^15 rows | 864 / 682 KiB       | 851 / 672 KiB       |
| Keccak-f[1600], 2^20 rows | 3,536 / 3,285 KiB   | 3,545 / 3,223 KiB   |
| SHA-256, 2^21 rows        | 5,507 / 5,183 KiB   | 5,479 / 5,174 KiB   |
| AES-128, 31,250 blocks    | 4,692 / 4,390 KiB   | 4,674 / 4,362 KiB   |
| AES-256, 31,250 blocks    | 4,982 / 4,677 KiB   | 4,959 / 4,647 KiB   |
| RSA-2048 PKCS#1 v1.5      | 14,532 / 14,353 KiB | 14,585 / 14,348 KiB |

Every table that committed a clock or direction column lost it: Keccak goes from 211 to
206 bytes per row, the AES-128 round table from 103 to 95, the S-box table from 69 to 65.
Proof size moves between −1.9 and +0.4 percent, and RSA's three runs spread from 14,585
to 14,914 KiB, wider than its change. Prove time moves within measurement noise
everywhere except RSA-2048, 5 to 6 percent slower and not yet profiled. Peak memory stays
inside run-to-run spread: Keccak at 2^20 rows peaks at 2,418 / 2,417 MiB against
2,431 / 2,454.

### The post-quantum rebuild

| Workload          | Proving       | Verify         | Proof size        | Peak memory   |
|:------------------|:--------------|:---------------|:------------------|:--------------|
| ML-DSA-44, 0.35.0 | 883 / 809 ms  | 44.8 / 23.6 ms | 4,184 / 3,689 KiB | 477 / 469 MiB |
| ML-DSA-44, 0.36.0 | 558 / 467 ms  | 51.6 / 22.2 ms | 2,877 / 2,449 KiB | 280 / 220 MiB |
| ML-DSA-65, 0.35.0 | 946 / 849 ms  | 51.1 / 24.2 ms | 4,193 / 3,706 KiB | 512 / 466 MiB |
| ML-DSA-65, 0.36.0 | 617 / 521 ms  | 53.8 / 23.5 ms | 2,958 / 2,543 KiB | 274 / 239 MiB |
| ML-DSA-87, 0.35.0 | 1.34 / 1.24 s | 48.1 / 25.3 ms | 5,484 / 4,909 KiB | 811 / 787 MiB |
| ML-DSA-87, 0.36.0 | 776 / 689 ms  | 58.5 / 28.4 ms | 3,326 / 2,918 KiB | 361 / 303 MiB |

With zero knowledge, ML-DSA proves 35 to 42 percent faster, its proof is 29 to 39 percent
smaller and its peak memory 41 to 55 percent lower. Verification with zero knowledge
takes 2.7 to 10.4 ms longer. The rebuilt circuit runs six tables where the old one ran
seven, and none of the six is a memory table. These rows compare two different circuits:
the 0.36.0 row proves ML-DSA verification, and the 0.35.0 row proved much less.

ML-KEM has no like-for-like row. The 0.35.0 table measured a decapsulation circuit whose
multiplication table proved an addition and whose shared secret no host published. 0.36.0
measures two new examples. A sender proving `Encaps` and AES-256-CTR over an 87-byte
message takes 573 / 450 ms, 3,512 / 2,962 KiB and 248 / 165 MiB. A receiver chaining
`KeyGen` into `Decaps` takes 746 / 605 ms, 3,895 / 3,359 KiB and 359 / 255 MiB.

## What changed for circuit authors

**Caught by the compiler.** `Service::respond` takes `(values, selector)`, and the
request-index slots, `request_phased` and `clock_columns` are gone. Keccak hosts lose
their direction column, and the Keccak and AES trace generators lose their clock-pair
arguments. The post-quantum constructors take a call list:
`MlKemChiplet::new(params, &[MlKemCall])` and `MlDsaChiplet::new(params, &[msg_len])`.

**Caught at build and again at verify entry.** A malformed ordered bus: a missing or
doubled rank source, a witness column in the clock slot, a selector that is not a pinned
0/1 schedule, or a height-dependent chiplet endpoint that is not last on its side.

**Not caught anywhere.** The order of a host's emits. A responder answers requests in
order, its `k`-th emit answering the host's `k`-th request. Emitting each call's request
before its response, and not interleaving calls, is the host's job. Rows in another order
pair in that order, and the engine cannot tell intent from a mistake.

## What this release does not promise

This series' lists of what it does not check have been accurate. 0.33.0 named bus keys
as unchecked, and gaps 1 and 2 are bus keys. 0.34.0 and 0.35.0 named circuit semantics as
the author's burden, and the author of the post-quantum circuits was us.

**Reference coverage.** ML-KEM is checked against one independent implementation on fixed
seeds, and the NIST test vector files are not run. ML-DSA's `tr` and `μ` are checked only
against our own SHAKE. Three of the decapsulation word streams have no forgery of their
own. The label census runs only as a test; it accepts a key with two producers and does
not model the hint bus.

**Independent audit.** This workspace has not had one.

## For other implementers

**Search for every bus source.** For each column in a bus key, find the constraint that
reads it on its own table. A column that appears in the bus spec, the trace generator and
nowhere else is a value the prover picks, whatever the bus sum says about it. Both halves
of this release would have turned up in that search.

**A clock the prover writes is a label.** A call that spans more than one emit needs an
identity its halves share, fixed by something the prover does not write. Counting emits
over a pinned schedule provides one: the verifier computes it and nothing is committed.
Then check what the prover still controls. Ours was a table's height, which moved every
count after it.

**Bind the lever when you find it.** When a forgery works through a free column, closing
that forgery leaves the column free for the next one.

**A key field can be the only binding of a column.** Moving to rank clocks made the
Keccak direction bit redundant in the bus key. In the old post-quantum control tables
that bit was the only constraint on a column the sponge logic read, and deleting it would
have freed the column. A fixed pin replaced it before anything shipped, and the rebuilt
tables no longer have the column. Before deleting a field from a key, list everything it
was binding.

**Test the function, and forge consistently.** A circuit whose only oracle is its own
witness generator agrees with itself. Check outputs against an independent
implementation, and build forged witnesses that are consistent everywhere except the one
relation under test.

**Route single-assignment dataflow over labelled buses.** A general-purpose memory fails
open on a missing wire: an address nobody reads costs nothing and proves nothing. A bus
keyed by labels pinned at both ends fails closed: a consumer without a producer leaves
the bus unbalanced.

## Scope

The previous postmortems are at [postmortem-0.35](postmortem-0.35.md),
[postmortem-0.34](postmortem-0.34.md), [postmortem-0.33](postmortem-0.33.md) and
[postmortem-0.32](postmortem-0.32.md).

This one covers `hekate-program`, `hekate-verifier`, `hekate-sdk` and the chiplet crates
`hekate-keccak`, `hekate-aes`, `hekate-gadgets`, `hekate-sha2`, `hekate-rsa` and
`hekate-pqc`, all of which are open. The prover reaches this workspace as a signed shared
library built from a closed repository. Soundness here is a property of the verifier, the
AIR and the bus, all public, and every forgery above ran through the real prover and the
real verifier: gaps 1 to 3 against the 0.35.0 release and prover 0.13.0, and the fixed
shapes against 0.36.0 and prover 0.14.0.