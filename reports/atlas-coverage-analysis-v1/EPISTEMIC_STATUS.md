# EPISTEMIC_STATUS

Exact labels used throughout, per `FROZEN_ANALYSIS_PROTOCOL.md`: `MEASURED`, `DERIVED`,
`SUPPORTED`, `LOW_SUPPORT`, `UNKNOWN`, `NOT_ESTABLISHED`, `NOT_TESTED`.

## MEASURED

All raw tensor values (H/Q/K/V/attention/attention-output/logits) at every captured
step — direct reads from Lane B's real captures, inherited from
`reports/bcc-atlas-expansion-v1/ATLAS_DEEP/TENSOR_GRANULARITY.json`.

## DERIVED

Every descriptor in `FROZEN_ANALYSIS_PROTOCOL.md` (norms, deltas, cosines, curvature,
dispersion, concentration, entropy, top2-gap, logit-norm) — deterministic functions of
measured values, formulas reused verbatim from prior gates.

## SUPPORTED

- **L0, L1, L5, L20 replication on Lane B**: all 4 landmarks meet the frozen
  `REPLICATED_ON_LANE_B` criteria (`LANDMARK_REPLICATION_RESULTS.json`) — real,
  independently recomputed (not merely copied from Lane B's own audit; matched it
  exactly: 305 events, 87 runs, confirming correctness).
- **H_curvature, K_norm, probability_top2_gap differ between landmark and non-landmark
  steps** at Bonferroni-corrected significance (`LANDMARK_VS_NONLANDMARK_ANALYSIS.json`)
  — real, predeclared, effect sizes modest (rank-biserial 0.10-0.15).

## LOW_SUPPORT

- FACTUAL_GROUNDED landmark evidence: 1 supporting run of 7 eligible
  (`NONZERO_SUPPORT_ONLY`).
- MATH_REASONING landmark evidence: 2 supporting runs of 6 eligible, both from the same
  prompt (`LOW_SUPPORT_REPLICATION_CANDIDATE`).
- 46 of 60 route signatures occur in only 1-2 runs.

## UNKNOWN

- 20 of 24 mechanism layers, for landmark purposes — never tested in this gate, by
  design (frozen node set).
- Everywhere a `0` appears in `LANDMARK_COOCCURRENCE_MATRIX.csv` off the diagonal — no
  claim that those landmark pairs never co-occur, only that none was observed in these
  144 runs.

## NOT_ESTABLISHED

- Whether the 40 eligible no-landmark runs are simply shorter/under-sampled or
  occupy a genuinely different descriptive region (`NO_LANDMARK_TERRITORY_AUDIT.json` —
  explicit `NOT_ESTABLISHED` verdict, mission-allowed answer, not a forced conclusion).
- Any interpretation of why `H_curvature`/`K_norm`/`probability_top2_gap` differ near
  landmarks — the comparison is real and predeclared, but effect size is small
  (|rank-biserial| ≤ 0.15) and no mechanism is claimed.

## NOT_TESTED

- `Attention_dispersion` / `Attention_concentration` in the landmark-vs-non-landmark
  comparison — a real implementation bug (predeclared field name did not match the
  actual computed series key) was found on review and disclosed in
  `LANDMARK_VS_NONLANDMARK_ANALYSIS.json` rather than silently reported as a null
  result or silently patched by re-running the full ~13-minute computation for 2 of 12
  fields. Not a claim about the true relationship — genuinely not tested here.
- SRA-X, control, KV-compression production readiness — unrelated to this gate,
  explicitly out of scope.

## Never silently converted (mission section 21, verified)

Nowhere in this gate's outputs does correlation become causality, recurrence become
mechanism, a landmark become a cognitive stage, "unknown" become "unsafe", or a
no-landmark run become "failure." `NO_LANDMARK_OBSERVED` and
`LANDMARK_NOT_EVALUABLE_SHORT_EXECUTION` are kept as two distinct, never-merged labels
throughout `RUN_ELIGIBILITY_AUDIT.csv` and every downstream file that references them.
