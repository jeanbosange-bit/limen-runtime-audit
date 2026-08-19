# REPORT — Reference Landmark Network V1 (Lane B Out-of-Sample Validation)

Status: **EXECUTED, READ-ONLY**. 0 new generations. Reuses Lane B's 144 real captures
and Atlas Coverage Analysis V1's independently-recomputed 305 landmark events
(cross-verified, never modified). Frozen node set (L0/L1/L5/L20), frozen tie notation
reused from the surviving canonical V0 contract/test files.

## Primary question: do the replicated landmarks form statistically supported recurrent
route edges in Lane B?

**Yes, partially.** 2 of 12 predeclared directed edges — `L0→L5` and `L0→L20` — clear a
conservative, Holm-corrected (family of 18), 10,000-permutation, group-preserving null
test, each observed in 8 of 12 prompt families, all 3 seeds, and 17-18 distinct prompts
(not driven by a single prompt). Both are `STABLE` under leave-one-family-out. **This is
the first statistically validated directed edge anywhere in this program** — V0's own
gate validated zero.

## Edge results (`EDGE_VALIDATION_RESULTS.json`, `EDGE_NULL_RESULTS.csv`)

| Edge | Observed | Holm p | Families | Seeds | Prompts | Status |
|---|---|---|---|---|---|---|
| L0→L5 | 28/69 | 0.0054 | 8/12 | 3/3 | 18 | **SUPPORTED** |
| L0→L20 | 24/69 | 0.0054 | 8/12 | 3/3 | 17 | **SUPPORTED** |
| L0→L1 | 27/69 | 0.606 | — | — | — | LOW_SUPPORT |
| L1→L5, L1→L0, L1→L20, L5→L0, L5→L1, L5→L20, L20→L0, L20→L1, L20→L5 | 6-23/69 | 1.0 | — | — | — | LOW_SUPPORT |

The reverse directions of both supported edges (`L5→L0` n=11, `L20→L0` n=6) are
themselves only `LOW_SUPPORT` — the asymmetry is real in this data, not an artifact of
symmetric co-occurrence being mislabeled as directional.

**Undirected co-occurrence**: all 6 pairs `LOW_SUPPORT` — high raw counts (23-35 of 69
runs) but Holm p=1.0 throughout, because the frozen null already expects substantial
co-occurrence by chance among only 4 possible labels reshuffled within short per-run
label sets. Disclosed as a property of the conservative null, not evidence against
co-occurrence.

## Route results (`ROUTE_VALIDATION_RESULTS.json`, `ROUTE_SIGNATURES.csv`)

60 unique signatures (69 edge-eligible + non-eligible runs' `NONE` routes accounted for
separately). **1 `REFERENCE_ROUTE_SUPPORTED`**: `L0-L20` (3 runs, 2 families, 2 seeds,
built on the one already-supported constituent edge). 10 `REFERENCE_ROUTE_LOW_SUPPORT`.
48 `REFERENCE_ROUTE_NOT_SUPPORTED` (mostly single-run unique routes). No route was
promoted on frequency alone without its constituent edges qualifying, per the frozen
rule.

## Cross-family route audit (mission section 11)

The 10 cross-family signatures previously observed descriptively in
`reports/atlas-coverage-analysis-v1/` (different, non-canonical `>`/`~` notation used in
that gate) are referenced in `CROSS_FAMILY_ROUTE_AUDIT.json` for continuity. None is
promoted here solely for having been cross-family — this gate's own independently
computed `L0-L20` route earns its status from the frozen edge/route criteria, not from
matching a prior descriptive observation.

## Family/seed robustness

`EDGE_FAMILY_SUPPORT.csv` and `EDGE_SEED_SUPPORT.csv` give the full breakdown for all
12 edges. The 2 supported edges' per-seed counts (L0→L5: 6/8/14 across seeds 2101/2102
/2103; L0→L20: 5/6/13) show real presence in every seed, with some skew toward seed
2103 — disclosed, not hidden.

## Leave-family-out stability (`EDGE_LEAVE_FAMILY_OUT.csv`)

Both supported edges are `STABLE` — removing any single one of the 12 families never
drops their support below the 3-run stability floor.

## V0 comparison (`V0_V1_COMPARISON.json`, historical reference only, not pooled)

| | V0 (30 runs) | V1 Lane B (144 runs, 69 edge-eligible) |
|---|---|---|
| Runs with ≥1 landmark | 18 | 87 |
| Runs with ≥2 landmarks | 10 | 62 |
| Unique route signatures | 11 | 60 |
| Validated edges | 0 | **2** |
| Low-support edges | 5 | 10 |
| Verdict | PARTIAL | (this gate's own verdict below) |

No p-value or count is combined across the two gates. V0's own directory and its
`-claude-conflict` quarantine are unmodified.

## Star Command product contract (`STAR_COMMAND_NETWORK_CONTRACT.json`)

Human-facing labels only: `CONFIRMED LANDMARK`, `SUPPORTED REFERENCE CONNECTION` (2),
`LOW-SUPPORT CONNECTION` (10), `OBSERVED ROUTE`/`REFERENCE ROUTE` (1 supported). Edge
p-values live only in an expandable technical-detail block, never as the primary UI
text. The fallback message is defined but does not apply this gate (2 supported edges
exist).

## What this gate does not establish

No causal claim. `L0→L5` and `L0→L20` are described only as "supported descriptive
transitions" — never reasoning steps, decisions, mechanisms, or safe/unsafe routes.
`CONTROL: NO`. `SRA-X: NOT_TESTED`. `VBE: NOT_TESTED_IN_THIS_GATE`. No intervention, no
steering, no navigation policy.

## Verdict

**`REFERENCE_LANDMARK_NETWORK_PARTIAL`.** 2 of 12 directed edges and 1 route cleared
conservative, corrected, stable statistical validation — real progress from V0's zero —
but 10 of 12 edges remain low-support and the network as a whole is still sparse, not a
fully coherent recurrent structure. `SUPPORTED` was not forced; `PARTIAL` is the honest
description of a real, asymmetric, two-edge-anchored network.
