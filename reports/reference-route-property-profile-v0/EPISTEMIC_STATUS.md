# EPISTEMIC_STATUS

**Important distinction maintained throughout this gate**: the route's own SEQUENCE
validation (`L0→L5→L20` is `REFERENCE_ROUTE_SUPPORTED`, established in
`composite-reference-route-v0`) is **not the same question** as this gate's own finding
(whether the route carries a *distinctive measurable tensor property profile*, which it
mostly does not yet). Both are reported, never conflated.

## SUPPORTED

- The route itself remains `REFERENCE_ROUTE_SUPPORTED` (inherited, unaltered).

## LOW_SUPPORT

- **9 of 12 tensor properties**: `H_norm`, `H_curvature`, `Q_norm`, `V_norm`,
  `A_dispersion`, `A_concentration`, `Oattn_delta_norm`, `probability_entropy`,
  `probability_top2_gap` — each shows a real, non-trivial difference from at least one
  comparison group at raw p<0.05, but none meets the full `SUPPORTED` bar (consistent
  direction + significant after Holm correction against **both** NO_LANDMARK and
  OTHER_LANDMARK_ROUTE).
- **`Oattn_delta_norm`'s NO_LANDMARK comparison is the single Holm-significant result in
  this gate** (p=0.0398, effect -0.43) — but it is **not distinctive to this route**:
  `Oattn_delta_norm` is one of the 4 signals used to *define* a landmark encounter in
  the first place, so any landmark-bearing group looks different from NO_LANDMARK on
  this measure by construction. Directly confirmed: ROUTE_FULL vs. `L0_L5_ONLY` on the
  same property shows p=0.50, essentially identical medians (0.313 vs 0.305) — the
  route's landmark moments look like *any* landmark moment, not something special to
  this specific 3-node ordering. Also flagged `FAMILY_CONFOUNDED` and `SEED_SENSITIVE`.

## NOT_ESTABLISHED

- **3 properties**: `K_norm`, `Oattn_norm`, `logit_norm` — no comparison reached even
  raw p<0.05 in any pairing.

## NOT_EVALUABLE

- None this gate — all 48 predeclared comparisons had sufficient sample size (n>=5 in
  both groups) to be evaluated, given the `NO_LANDMARK` population's per-run-layer-mean
  substitution (disclosed below).

## Disclosed methodological asymmetry (not a bug, a design necessity)

`NO_LANDMARK` runs have zero landmark encounters by construction — there is no
encounter-step from which to extract a genuine `t=0` local window. This gate's
`NO_LANDMARK` comparison population instead uses each run's own per-layer **mean**
tensor value across its full generation (already computed by
`atlas-coverage-analysis-v1`, reused read-only), pooled across layers 0/5/20. This is a
coarser statistic than the specific-encounter-step values used for every other group,
disclosed explicitly in `FROZEN_PROPERTY_PROTOCOL.md` and
`ROUTE_GROUP_COMPARISONS.json` before any p-value was computed — not discovered after
the fact.

## Confounds found (real, not hidden)

- `H_curvature`, `Oattn_delta_norm`, `probability_entropy`: `FAMILY_CONFOUNDED` —
  removing at least one of ROUTE_FULL's 6 supporting families flips their significance
  against NO_LANDMARK.
- `Oattn_delta_norm`, `probability_entropy`, `probability_top2_gap`: `SEED_SENSITIVE` —
  splitting by seed 2102 vs. 2103 changes significance for at least one seed.
- `LENGTH_CONFOUND_AUDIT.json`: ROUTE_FULL's generation lengths (median 130 steps) are
  concentrated toward the long end of Lane B's distribution — flagged as a standing risk
  for every LOW_SUPPORT+ property, not exhaustively tested (n=8 too small to stratify
  further).

## No implementation bug found this gate

The `Attention_dispersion`/`Attention_concentration` field-name mismatch disclosed in
`atlas-coverage-analysis-v1/EPISTEMIC_STATUS.md` was deliberately avoided here by
verifying the correct source column names (`A_dispersion`/`A_concentration`) before
writing any code, per `FROZEN_PROPERTY_PROTOCOL.md`'s explicit note.

## Never silently converted

No property or route claim in this gate's outputs equates recurrence with mechanism, a
landmark with a cognitive stage, or a property difference with causation. Landmarks are
described only positionally ("L0 occurs first", "L5 is the intermediate confirmed
landmark", "L20 occurs later").
