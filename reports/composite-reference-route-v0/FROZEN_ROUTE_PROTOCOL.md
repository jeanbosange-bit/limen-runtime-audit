# FROZEN_ROUTE_PROTOCOL

Frozen before any route count, p-value, or temporal distance is inspected. Extension of
`reports/reference-landmark-network-v1/FROZEN_NETWORK_PROTOCOL.md` to 3-node composite
routes — same principles, not re-derived from scratch.

## Data source (read-only)

`reports/reference-landmark-network-v1/RELATIONS_CACHE.json` (per-run tie-grouped
ordered encounter sequences, 87 runs with >=1 landmark) and
`reports/atlas-coverage-analysis-v1/PASS1_CACHE.json` (305 raw events with exact step
numbers, needed for temporal-distance measurement). Neither file is modified.

## Frozen node set

L0, L1, L5, L20 — unchanged, no new landmark, no threshold retuning (identical
inheritance chain as every prior gate in this line).

## Route candidate universe

24 ordered 3-node routes (`ROUTE_CANDIDATE_UNIVERSE.json`) — all 6 permutations of each
of the 4 possible 3-node subsets of the frozen node set. The primary candidate
(`L0→L5→L20`) is tested identically to the other 23, not specially favored.

## Route type definitions (mission section 5, frozen)

- `DIRECT_COMPOSITE_ROUTE`: the three nodes' tie-groups appear as three *consecutive*
  groups in the run's ordered sequence, in the candidate's exact order, with no other
  landmark group between any of the three.
- `ORDERED_ROUTE_WITH_GAPS`: the three nodes' groups appear in the candidate's exact
  relative order somewhere in the run, but with >=1 other landmark group (any node,
  including a repeat of one of the three) between at least one adjacent pair.
- `SAME_STEP_TIE_ROUTE`: reuses `reference-landmark-network-v1/SAME_STEP_TIE_POLICY.md`
  exactly — if any two of the three candidate nodes are tied (same step), no ordering
  claim is made between that tied pair; the run is recorded under this category rather
  than being force-fit into direct/gapped succession, and does **not** count toward
  either `DIRECT_COMPOSITE_ROUTE` or `ORDERED_ROUTE_WITH_GAPS` for that route.
- `NO_COMPOSITE_ROUTE`: the three nodes' groups do not appear in the candidate's order
  anywhere in the run (or fewer than all three nodes appear at all).

## Primary route statistic

Observed count = number of distinct runs classified as either `DIRECT_COMPOSITE_ROUTE`
or `ORDERED_ROUTE_WITH_GAPS` for that candidate (the two "route holds" categories
combined — mirrors V1's direct+indirect combination for single edges). The
direct-vs-gapped split is reported descriptively alongside, not separately tested.

## Null model (reused from V1, extended to 3-node statistic)

**Within-run landmark-label permutation** — identical mechanism to
`reference-landmark-network-v1`: for each run with >=1 landmark encounter, step
positions and tie-group structure are held fixed; which node-label occupies which
position is randomly permuted. This preserves run identity, per-run encounter count,
tie structure, prompt family, seed, execution-length class, and (in aggregate) the
global landmark marginal frequency — destroying only the ordering a route test needs.

- **Permutations**: 10,000 (confirmed computationally feasible by timing dry-run,
  consistent with V1's ~25s/10,000-perm runtime for a comparable in-memory operation).
- **RNG seed**: 20260818 (same convention as every prior gate).
- **Sidedness**: one-sided (enrichment).
- **p-value**: `(permuted statistic >= observed + 1) / (n_perm + 1)`.

## Multiple-comparison correction

**Holm**, one family of all 24 predeclared route candidates — same method as V1 (chosen
for consistency and because it strictly controls family-wise error on this small, fully
enumerated hypothesis set).

## Route support thresholds (mission section 8, frozen — minimum, not loosened after
results)

- `REFERENCE_ROUTE_SUPPORTED`: Holm-adjusted p<0.05 AND >=2 distinct prompt families AND
  >=2 distinct seeds AND >=2 distinct supporting prompt_ids (not driven by one prompt)
  AND ordering stable under leave-one-family-out (status unchanged in all 12
  leave-one-out recomputations).
- `REFERENCE_ROUTE_LOW_SUPPORT`: observed count >=1 but fails the `SUPPORTED` bar.
- `REFERENCE_ROUTE_NOT_SUPPORTED`: observed count is 0, or Holm p>=0.05 with no
  qualification met.
- `NOT_EVALUABLE`: data-integrity failure specific to that route (checked, not expected).

## Edge-consistency disclosure (mission section 9)

Per-route constituent edge statuses (from
`reference-landmark-network-v1/EDGE_VALIDATION_RESULTS.json`) are reported alongside
every route result for interpretive context, but **route promotion depends only on the
route-level statistical test above** — a route is never promoted merely because one or
both constituent edges are independently `SUPPORTED`, and is never blocked from testing
because a constituent edge is `LOW_SUPPORT` (mission section 9 explicit: "A route may
only be promoted if route-level evidence itself passes the predeclared validation").

## Branching vs. composition (mission section 10, frozen verdict rule)

Given `L0→L5→L20` is the primary candidate: compare (a) the route-level observed count
and p-value for `L0→L5→L20` directly, against (b) the product of evidence for the two
independent edges `L0→L5` and `L0→L20` occurring in the *same* run without the
`L5→L20` order necessarily holding (i.e., runs supporting both edges but not the
composite order). Frozen verdict rule:
- `COMPOSITION_SUPPORTED`: `L0→L5→L20` itself reaches `REFERENCE_ROUTE_SUPPORTED`.
- `BRANCHING_MORE_CONSISTENT`: the composite route is `NOT_SUPPORTED` or `LOW_SUPPORT`,
  but runs supporting both `L0→L5` and `L0→L20` independently (as a same-run
  co-occurrence, regardless of L5-L20 order) substantially outnumber runs supporting the
  full ordered composite.
- `BOTH_PARTIAL`: composite route is `LOW_SUPPORT` and independent-branch co-occurrence
  is also modest — neither picture dominates.
- `NOT_ESTABLISHED`: sample too small to distinguish (fewer than 5 runs in either
  comparison population).

## Temporal distance (mission section 11)

For each pairwise gap within a tested route (e.g. `L0→L5`, `L0→L20`, `L5→L20` for the
primary candidate), step distance = |step(target) - step(source)| computed from the
first qualifying occurrence of each node in a route-holding run. Reported as
min/median/IQR/max, pooled and by family. Purely descriptive — no claim about what a
short or long gap means.

## Length bins (mission section 14, frozen, reused verbatim — not redefined)

`SHORT_1_9` (n_steps<=9), `MEDIUM_10_49` (10<=n_steps<=49), `LONG_50_PLUS`
(n_steps>=50) — identical bin edges already used in
`reports/bcc-atlas-expansion-v1/ATLAS_DEEP/build_lane_b_audit.py` and
`reports/atlas-coverage-analysis-v1/`.

## Coverage boundary preserved (mission section 15)

No-landmark runs (40), single-landmark runs, and short/non-evaluable runs (17) remain
explicitly accounted for in `ROUTE_OBSERVED_COUNTS.csv`'s denominator reporting — they
structurally cannot support any 3-node route and are never silently dropped from the
coverage picture.

## Forbidden / allowed vocabulary (mission section 16)

Forbidden: "initiates", "mediates", "terminates", "decision path", "reasoning path",
"causal chain", "control corridor", "safe route". Allowed: "observed ordered route",
"recurrent transition", "supported reference route", "branch-like descriptive
structure".
