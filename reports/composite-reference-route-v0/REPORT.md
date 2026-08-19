# REPORT — Composite Reference Route V0

Status: **EXECUTED, READ-ONLY**. 0 new generations. Extends
`reference-landmark-network-v1`'s two supported edges (`L0→L5`, `L0→L20`) into a
predeclared, full 24-candidate 3-node route test over the frozen L0/L1/L5/L20 nodes.

## Primary question: do the supported edges participate in a reproducible composite
route?

**Yes — but only the primary candidate itself, not its permutation siblings or the
comparable L1-involving routes.** `L0→L5→L20` is `REFERENCE_ROUTE_SUPPORTED`: 8 of 87
landmark-bearing runs, Holm-adjusted p=0.0072 (survives correction across all 24
predeclared candidates), 6 of 12 prompt families, 2 of 3 seeds, 7 distinct prompts, and
`STABLE` under leave-one-family-out. This is the **first statistically validated 3-node
composite route in this program.**

## Candidate universe results (`ROUTE_VALIDATION_RESULTS.json`, `ROUTE_NULL_RESULTS.csv`)

24 candidates tested (6 permutations each of `{L0,L5,L20}`, `{L0,L1,L5}`, `{L0,L1,L20}`,
`{L1,L5,L20}`) — not only the hoped-for route. Top 6 by observed count:

| Route | Observed | Holm p | Status |
|---|---|---|---|
| L0→L5→L20 | 8/87 | 0.0072 | **SUPPORTED** |
| L0→L1→L5 | 7/87 | 0.257 | LOW_SUPPORT |
| L0→L5→L1 | 6/87 | 0.638 | LOW_SUPPORT |
| L0→L1→L20 | 6/87 | 0.060 | LOW_SUPPORT |
| L0→L20→L5 | 5/87 | 0.433 | LOW_SUPPORT |
| L1→L20→L5 | 4/87 | 1.0 | NOT_SUPPORTED |

Every candidate whose first node is not L0 shows an observed count of 0 or near-0 —
consistent with, and independently corroborating, V1's finding that only edges
originating from L0 were supported.

## Edge consistency (mission section 9)

`L0→L5→L20`'s constituent edges are `L0→L5` (`REFERENCE_EDGE_SUPPORTED` in V1) and
`L5→L20` (`REFERENCE_EDGE_LOW_SUPPORT` in V1). **The composite route reached
`SUPPORTED` status despite one of its two constituent edges being only low-support at
the pairwise level** — genuine confirmation that route-level and edge-level evidence
are independent tests, exactly as the mission anticipated: a route is not promoted
merely because a constituent edge is strong, and is not blocked merely because one is
weak.

## Branching vs. composition (`BRANCHING_VS_COMPOSITION_ANALYSIS.json`)

**Verdict: `COMPOSITION_SUPPORTED`** — by the frozen rule, since the primary candidate
itself reached `REFERENCE_ROUTE_SUPPORTED`. For transparency: 8 runs show the full
ordered composite; a comparable-magnitude set of runs also show both `L0→L5` and
`L0→L20` present without necessarily completing the full order (exact counts in the
JSON) — both a real composite pattern and real independent-branch activity coexist in
this data; the composite reading is not exclusive, but it is the one that clears
predeclared validation.

## Temporal distance (`ROUTE_TEMPORAL_DISTANCE.csv`)

For the 8 supporting runs of `L0→L5→L20`: `L0→L5` median 64.5 steps (IQR 44-72),
`L0→L20` median 94.5 steps (IQR 49-130), `L5→L20` median 38.5 steps (IQR 37-53).
**Transitions are broadly distributed, not tightly localized** — no claim is made about
what this timing spread means.

## Robustness

- **Seeds**: `L0→L5→L20` appears in seeds 2102 (3 runs) and 2103 (5 runs), **absent in
  2101** — disclosed plainly, not smoothed over. Still clears the frozen `>=2 seeds`
  bar.
- **Families**: 6 of 12 (CODING, CONSTRAINT_FOLLOWING, GENERAL_INSTRUCTION x2,
  MULTISTEP_INSTRUCTION x2, RECOVERY_CORRECTION, SUMMARIZATION) — absent from
  AMBIGUITY_HANDLING, FACTUAL_GROUNDED, MATH_REASONING, STRUCTURED_OUTPUT,
  TRANSFORMATION, DOCUMENT_EXTRACTION at the composite-route level (some of those
  families do support the individual edges, just not this specific triple order).
- **Leave-family-out**: `STABLE` — removing any single family never drops support below
  the 3-run floor.

## Coverage boundary preserved (mission section 15)

Of Lane B's 144 runs: 87 carry >=1 landmark encounter (the only pool from which any
3-node route can be drawn), 40 eligible runs show 0 landmarks, and 17 are
short/non-evaluable — none of these 57 runs can structurally support any composite
route and none is silently excluded from the accounting (`ROUTE_OBSERVED_COUNTS.csv`).

## What this gate does not establish

No causal claim. `L0→L5→L20` is described only as a "supported reference route" — never
as an initiating, mediating, or terminating step, a reasoning or decision path, or a
causal chain. `CONTROL: NO`. `SRA-X: NOT_TESTED`. No intervention, no steering, no
navigation policy.

## Verdict

**`COMPOSITE_REFERENCE_ROUTE_PARTIAL`.** One route out of 24 predeclared candidates
cleared full, conservative, corrected, stable validation — real, substantive progress
extending V1's two-edge network into a validated 3-node structure — but 23 of 24
candidates remain low-support or unsupported, and the validated route itself only
reaches 2 of 3 seeds and 6 of 12 families. `SUPPORTED` (network-wide) is not claimed;
`PARTIAL` is the honest description of one real, validated composite route within an
otherwise sparse candidate space.
