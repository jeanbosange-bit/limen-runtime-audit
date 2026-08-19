# FROZEN_NETWORK_PROTOCOL

Frozen before any edge count, p-value, or route frequency is inspected. Validation gate
only — no new landmark search, no threshold retuning.

## Data source (read-only)

`reports/atlas-coverage-analysis-v1/PASS1_CACHE.json` — the 305 landmark encounter
events (run_id, node, layer, step, family, seed, prompt_id), independently recomputed
by that gate from Lane B's raw captures and cross-verified to match Lane B's own
`LANDMARK_RECURRENCE_AUDIT.json` exactly. `reports/atlas-coverage-analysis-v1/RUN_ELIGIBILITY_AUDIT.csv`
for run-level eligibility/coverage accounting. Neither file is modified.

## Eligibility (mission section 4)

- **Edge-eligible**: a run with **>=2 total landmark encounter events** (any node,
  including repeats of the same node) — literal mission wording. Runs with 0 or 1
  encounter, or `LANDMARK_NOT_EVALUABLE_SHORT_EXECUTION` (n_steps<4), remain in
  coverage accounting (`ELIGIBILITY_AUDIT.csv`) but contribute **no** edge observation
  (never fabricated).
- Self-pairs (a node's repeat encounter with itself) never contribute to any of the 12
  directed cross-node edges (`EDGE_UNIVERSE.json`) or 6 undirected pairs by
  construction, since none of those 18 predeclared tests involves a node paired with
  itself.

## Tie handling

See `SAME_STEP_TIE_POLICY.md` (frozen separately, reusing V0's canonical `+`/`-`
notation). Tied nodes generate `CO_OCCURRENCE` only, never succession, in either
direction.

## Edge type definitions (mission section 6, frozen)

- `DIRECT_SUCCESSION(A,B)`: in a run's own ordered (tie-group-collapsed) encounter
  sequence, a group containing A is immediately followed by a group containing B, with
  no other landmark group in between.
- `INDIRECT_SUCCESSION(A,B)`: a group containing A precedes a group containing B
  somewhere in the run, with >=1 other landmark group between them.
- `CO_OCCURRENCE(A,B)`: A and B are in the same tie-group (same step) in a run, OR (for
  the separate 6-pair undirected analysis only) A and B both appear anywhere in the same
  edge-eligible run, with no direction claim.
- `NO_OBSERVED_LINK`: the pair never appears together in any edge-eligible run.

## Primary directed-edge statistic (mission section 9)

For each of the 12 directed pairs: **observed count** = number of distinct edge-eligible
runs in which A precedes B (direct or indirect succession combined — this is the single
predeclared primary statistic; the direct/indirect split is reported descriptively
alongside, not separately tested, to keep the corrected family at exactly 18 tests).

## Null model (mission section 8, frozen)

**Within-run landmark-label permutation.** For each edge-eligible run, the exact
step-positions (including tie-group structure) at which encounters occurred are held
fixed, and the exact multiset of node-labels observed in that run is held fixed, but
which label occupies which position (respecting tie-groups as single units, per
`SAME_STEP_TIE_POLICY.md` rule 5) is randomly permuted. This preserves: run identity
(permutation is always within one run), per-run encounter count, per-run tie structure,
prompt family and seed (labels never move across runs), execution-length class (a run's
own step positions never change), and — as an aggregate consequence of only permuting
existing labels — the global landmark marginal frequency. It destroys exactly the
ordered temporal relationship a directed edge test needs to be non-trivial.

- **Permutations**: 10,000 (computationally feasible — confirmed by a timing dry-run
  before the full run; disclosed in `EDGE_NULL_RESULTS.csv` if a lower count was
  actually used).
- **RNG seed**: 20260818 (same convention as every prior gate in this program).
- **Statistic**: same as the observed primary statistic (run-count supporting A→B).
- **Sidedness**: one-sided (testing for *enrichment* of the observed direction — mission
  section 9 asks for "effect size / enrichment", not a two-sided departure test).
- **p-value**: `(count of permuted statistic >= observed statistic + 1) / (n_perm + 1)`.

## Multiple-comparison correction (mission section 9, frozen)

**Holm** (step-down, uniformly more powerful than Bonferroni while still controlling
family-wise error rate), applied across all 18 predeclared tests (12 directed + 6
undirected) as one family. Chosen over Benjamini-Hochberg because this gate cares about
strict false-positive control on a small, fully-enumerated hypothesis set (network edges
that may get cited downstream), not maximizing discoveries in a large exploratory panel.

## Edge status thresholds (mission section 10, frozen — minimum + no loosening after
results)

- `REFERENCE_EDGE_SUPPORTED`: Holm-corrected p < 0.05 AND observed in >=2 distinct
  prompt families AND observed across >=2 distinct seeds AND not driven by a single
  prompt_id alone (>=2 distinct supporting prompt_ids).
- `REFERENCE_EDGE_LOW_SUPPORT`: observed count >=1 but fails the `SUPPORTED` bar (either
  statistically, or confined to 1 family/1 seed/1 prompt_id).
- `REFERENCE_EDGE_NOT_SUPPORTED`: observed count is 0, or Holm-corrected p >= 0.05 with
  no family/seed/prompt qualification met.
- `NOT_EVALUABLE`: reserved for a data-integrity failure specific to that edge (checked,
  not expected to trigger).

## Route status thresholds (mission section 12, frozen)

- `REFERENCE_ROUTE_SUPPORTED`: route observed in >=3 edge-eligible runs, >=2 distinct
  prompt families, >=2 distinct seeds, AND every constituent directed edge in the route
  (adjacent pairs, tie-groups excluded from the edge requirement) is at least
  `REFERENCE_EDGE_LOW_SUPPORT` (a route cannot be `SUPPORTED` purely on its own frequency
  if its edges are `NOT_SUPPORTED` — mission section 12 explicit).
- `REFERENCE_ROUTE_LOW_SUPPORT`: observed in >=2 runs but fails the `SUPPORTED` bar.
- `REFERENCE_ROUTE_NOT_SUPPORTED`: observed in exactly 1 run, or 0.

## Seed robustness (mission section 14)

For each of the 12 directed edges, observed count is additionally broken out by seed
(2101/2102/2103). An edge appearing in only one seed is `LOW_SUPPORT` regardless of its
aggregate p-value, per mission's explicit instruction — no exception is predeclared.

## Leave-family-out stability (mission section 15)

For each directed edge, the primary statistic is recomputed 12 times, once per family
left out. Labeled `STABLE` if the edge's qualitative status (supported vs. not) is
unchanged in all 12 leave-one-out recomputations; `FAMILY_SENSITIVE` if removing any
single family flips the status; `NOT_EVALUABLE` if the edge has too few supporting runs
(<3) for a leave-one-out recomputation to be meaningful.

## Explicit non-claims (mission section 18)

No edge or route is ever labeled with: causes, drives, forces, mechanism established,
reasoning path, decision path, failure path, safe route, unsafe route. Allowed:
precedes, follows, co-occurs, recurs, supported descriptive transition, reference route.
