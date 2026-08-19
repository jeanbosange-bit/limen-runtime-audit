# FROZEN_PROPERTY_PROTOCOL

Frozen before any tensor property or group comparison is inspected. Extends
`reports/composite-reference-route-v0/`'s validated route (`L0→L5→L20`,
`REFERENCE_ROUTE_SUPPORTED`) with a property-characterization analysis. This is a
characterization gate, not causal explanation, failure prediction, or control.

## Data sources (all read-only)

- `reports/composite-reference-route-v0/OBS_CACHE.json` — which runs support `L0-L5-L20`.
- `reports/reference-landmark-network-v1/RELATIONS_CACHE.json` — which runs support
  `L0→L5` / `L0→L20` (direct or indirect succession).
- `reports/atlas-coverage-analysis-v1/RUN_ELIGIBILITY_AUDIT.csv` — run-level
  family/seed/length/eligibility metadata for all 144 runs.
- `reports/atlas-coverage-analysis-v1/{H,Q,K,V,ATTENTION,ATTENTION_OUTPUT,LOGIT_PROBABILITY}_LOCAL_PROPERTIES.csv`
  — already-computed local-window (t-1/t/t+1) tensor descriptors for every one of the
  305 real landmark encounter events across all 4 frozen layers. **Reused directly, not
  recomputed** — no npz reload in this gate.

## Frozen comparison groups (mission section 2, exact partition — decided before any
effect is inspected)

- **A. ROUTE_FULL**: the 8 runs supporting the ordered composite `L0→L5→L20`
  (`composite-reference-route-v0`'s own definition, unaltered).
- **B. L0_L5_ONLY**: runs supporting `L0→L5` **and NOT** `L0→L20`, and not in A.
- **C. L0_L20_ONLY**: runs supporting `L0→L20` **and NOT** `L0→L5`, and not in A.
- **D. OTHER_LANDMARK_ROUTE**: landmark-bearing eligible runs not in A/B/C — includes
  runs supporting *both* `L0→L5` and `L0→L20` without completing the full ordered route
  (the "branch-only" runs identified in `composite-reference-route-v0/BRANCHING_VS_COMPOSITION_ANALYSIS.json`),
  plus every other landmark-bearing run not involving these specific edges.
- **E. NO_LANDMARK**: the 40 eligible runs with zero landmark encounters.
- **F. SHORT_NON_EVALUABLE**: the 17 runs with `n_steps<4` — coverage accounting only,
  never used in a tensor-property comparison.

**Disclosed interpretation note**: mission section 2 names groups B/C "`_ONLY`" — read
literally as mutually exclusive with each other (a run with *both* edges present but
not completing the full route goes to D, not double-counted into both B and C). This
keeps A-F a strict partition of all 144 runs, which the mission's own group F
("coverage accounting") implies is the intended structure.

## Local windows (mission section 5)

Reused verbatim from `atlas-coverage-analysis-v1`'s already-computed
`*_LOCAL_PROPERTIES.csv` files, filtered to the specific `(run_id, layer, step)` triples
that define each group's route-relevant landmark occurrence (the *first* occurrence of
each node per run, matching the exact convention already used to classify routes in
`composite-reference-route-v0`). No step is fabricated; unavailable `t-1`/`t+1` edges
(e.g. `step=0`) remain `null`, exactly as already encoded in the source CSVs'
`available` column.

## Statistical plan (mission sections 11-12, frozen before inspecting any comparison)

- **Test**: Mann-Whitney U (two-sided), non-parametric — identical choice to
  `atlas-coverage-analysis-v1`'s landmark-vs-non-landmark comparison, for methodological
  continuity.
- **Comparisons (test family)**: ROUTE_FULL vs. each of {L0_L5_ONLY, L0_L20_ONLY,
  OTHER_LANDMARK_ROUTE, NO_LANDMARK} — 4 comparison groups — for each of 12 predeclared
  properties (`H_norm`, `H_curvature`, `Q_norm`, `K_norm`, `V_norm`,
  `Attention_dispersion`, `Attention_concentration`, `Oattn_norm`, `Oattn_delta_norm`,
  `probability_entropy`, `probability_top2_gap`, `logit_norm`) = **48 predeclared tests**,
  evaluated at the `t=0` (encounter step) value pooled across the route's own 3 landmark
  layers (0, 5, 20) for ROUTE_FULL's own rows, and at the corresponding available layers
  for each comparison group's own landmark rows.
- **Note on `Attention_dispersion`/`Attention_concentration` field names**: this gate's
  source CSVs use `A_dispersion`/`A_concentration` column names (verified directly
  against `ATTENTION_LOCAL_PROPERTIES.csv`'s header before writing any code) — the
  mismatch disclosed in `atlas-coverage-analysis-v1/EPISTEMIC_STATUS.md` is deliberately
  NOT repeated here; the correct column names are used from the start.
- **Multiple-comparison correction**: Holm, one family of 48.
- **Effect size**: rank-biserial correlation, reported with every p-value (raw and
  Holm-adjusted).
- **Minimum sample requirement**: both groups must have `n>=5` valid (non-null)
  observations for a comparison to be evaluated at all; below that, the property is
  `NOT_EVALUABLE` for that pairing — never silently converted into "no difference."

## Route property status thresholds (mission section 13, frozen)

- `ROUTE_PROPERTY_SUPPORTED`: the ROUTE_FULL-vs-comparison direction is consistent
  across **all 4** comparison groups (same sign of effect), Holm-adjusted p<0.05 in
  **at least the NO_LANDMARK and OTHER_LANDMARK_ROUTE comparisons**, not driven by a
  single seed (property still shows the same-direction effect with seed 2101 excluded —
  ROUTE_FULL only has 2101 absent to begin with, so this reduces to: consistent with
  both 2102-only and 2103-only subsets where n allows), and not driven by a single
  family (removing any one of the 6 supporting families does not flip the sign).
- `ROUTE_PROPERTY_LOW_SUPPORT`: a real, consistent-direction effect exists in at least
  one comparison at raw p<0.05, but does not meet the full `SUPPORTED` bar above.
- `ROUTE_PROPERTY_NOT_ESTABLISHED`: comparisons were run (sufficient n) but showed no
  consistent or significant effect, OR effects existed but disappeared under family/seed
  conditioning (labeled `FAMILY_CONFOUNDED` / `SEED_SENSITIVE` specifically, a subtype of
  this status).
- `NOT_EVALUABLE`: insufficient sample size in one or more comparison groups (n<5)
  — never converted into "no difference."

## Family/seed/length conditioning (mission sections 14-16, frozen rule)

For every property reaching at least `ROUTE_PROPERTY_LOW_SUPPORT`: recompute the
ROUTE_FULL-vs-NO_LANDMARK comparison after removing each of ROUTE_FULL's 6 supporting
families one at a time (leave-one-family-out) and after splitting by each of the 2
supporting seeds. A property whose effect direction flips or whose significance is lost
under family removal is labeled `FAMILY_CONFOUNDED`; under seed split, `SEED_SENSITIVE`.
Generation-length conditioning uses the frozen Atlas Coverage Analysis V1 bins
(`SHORT_1_9`, `MEDIUM_10_49`, `LONG_50_PLUS`, unaltered) — if ROUTE_FULL's own length
distribution is concentrated in one bin relative to comparison groups and the property
difference disappears when comparison groups are restricted to the same bin, labeled
`LENGTH_CONFOUNDED`.

## Positional language only (mission section 17)

Landmarks are described only by their position in the route: "L0 occurs first",
"L5 is the intermediate confirmed landmark in this route", "L20 occurs later" — never as
initiating, transforming, or terminating any process.
