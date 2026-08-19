# REPORT — Atlas Coverage Analysis V1

Status: **EXECUTED, READ-ONLY**. `new_generations = 0`. Independent recomputation from
Lane B's 144 real raw captures (`reports/bcc-atlas-expansion-v1/ATLAS_DEEP/`), which are
never modified. No new landmark candidate search, no edge promotion, no control/causal
claim.

## Primary question: what parts of the observed execution space are now well covered, low
support, or unknown?

**Well covered**: run-level coverage for all 12 prompt families (each has ≥6 eligible
runs, ≥2 prompt_ids, all 3 seeds — `FAMILY_COVERAGE_MATRIX.csv`), all 7 tensor types at
full native resolution, and the 4 frozen landmark nodes (each replicates in 30-47% of
127 eligible runs). **Low support**: FACTUAL_GROUNDED and MATH_REASONING *landmark
evidence specifically* (as opposed to run coverage, which is fine) — 1 and 2 supporting
runs respectively. **Unknown**: 20 of 24 mechanism layers for landmark purposes (by
design), most of the 60 route signatures (46 seen only 1-2 times).

## Secondary questions

**1. Do the 4 frozen confirmed landmarks replicate?** Yes — all 4
(`LANDMARK_REPLICATION_RESULTS.json`): L0 47.2%, L1 43.3%, L5 33.9%, L20 29.9%
recurrence among the 127 eligible runs, each in 10-12 of 12 families and all 3 seeds.
`REPLICATED_ON_LANE_B` under the frozen threshold for all 4. The prior canonical
`REFERENCE_LANDMARK_SUPPORTED` status is unaffected — this is an independent audit, not
a re-validation with authority to change it.

**2. Which route signatures recur descriptively?** 60 unique signatures across 144 runs.
The single largest category is `NONE` (57 runs — 40 eligible-no-landmark + 17
non-evaluable). Among landmark-bearing routes, single-node routes dominate (`L0`: 9
runs, `L1`: 7 runs). The most-repeated multi-node pattern is `L0>L0` (a run touching L0
twice, 4 occurrences, 4 families) — a real, recurring self-revisit pattern, reported
descriptively only. 4 signatures qualify as `COMMON_ROUTE` (≥3 runs, ≥2 families); 10
signatures are cross-family. No edge is promoted from any of this (`ROUTE_SIGNATURES.csv`).

**3. Which tensor properties differ locally around landmarks?**
`H_curvature`, `K_norm`, and `probability_top2_gap` differ between landmark and
non-landmark steps at Bonferroni-corrected significance (p<0.00417), with modest effect
sizes (rank-biserial 0.10-0.15) — `LANDMARK_VS_NONLANDMARK_ANALYSIS.json`. 9 other
predeclared comparisons were not significant after correction; 2 (`Attention_dispersion`,
`Attention_concentration`) were not tested due to a disclosed implementation bug (field
name mismatch), not insufficient data.

**4. What characterizes no-landmark territory?** 40 eligible runs with zero landmark
encounters — on average shorter (mean generation length) than landmark-bearing runs, but
whether that is under-sampling or a genuinely different region is **`NOT_ESTABLISHED`**
(`NO_LANDMARK_TERRITORY_AUDIT.json`) — the mission-allowed honest answer, not forced
toward a conclusion.

**5. Which prompt families / length regimes remain under-covered?** FACTUAL_GROUNDED (5
of 12 runs non-evaluable, shortest median length of any family at 4.0 steps) and
MATH_REASONING (6 of 12 non-evaluable, median 3.5 steps) — both structurally short by
the nature of their expected answers (single facts/numbers), not a capture defect.

## Factual/Math special audit (mission section 18)

**FACTUAL_GROUNDED: `NONZERO_SUPPORT_ONLY`** — 1 supporting run (FG10-V2, seed 2102,
landmarks L1+L5). **MATH_REASONING: `LOW_SUPPORT_REPLICATION_CANDIDATE`** — 2 supporting
runs, both from prompt MR07 (seed 2102: L0; seed 2103: L1+L20) — 2 distinct seeds but
only 1 distinct prompt_id, and different landmark nodes each time, so it does not clear
the `PARTIAL_REPLICATION` bar (`FACTUAL_MATH_SUPPORT_AUDIT.json`). Both are **new,
real, non-zero observations** — the first time either family has shown any landmark
support in this program — but neither is replication yet, exactly as the mission
instructed not to overclaim.

## Coverage holes (ranked by absence of observation, not effect size)

59 entries in `COVERAGE_HOLES.csv`: short/non-evaluable runs concentrated in
FACTUAL_GROUNDED (5) and MATH_REASONING (6); 46 low-support/unique route signatures; 20
mechanism layers entirely untested for landmark purposes by design; the 17
`LANDMARK_NOT_EVALUABLE_SHORT_EXECUTION` runs as a structural class.

## What this gate does not establish

No causal claim anywhere. No landmark, route, or local-property difference is
interpreted as a reasoning step, decision, or cognitive stage. No new landmark or
network edge was created. Effect sizes for the landmark-vs-non-landmark comparison are
real but modest — not read as strong discriminators. `SRA-X: NOT_TESTED`. `CONTROL: NO`.
`VBE: NOT_TESTED_IN_THIS_GATE`. `KV_COMPRESSION: NO_NEW_CLAIM`.

## Verdict

**`ATLAS_COVERAGE_SUBSTANTIALLY_EXPANDED`** — 144 new, real, fully-captured runs (H/Q/K/V
/attention/attention-output/logits, Q captured for the first time), all 4 canonical
landmarks independently replicated, and non-zero landmark evidence obtained in two
families (FACTUAL_GROUNDED, MATH_REASONING) that had zero evidence in every prior gate.
This concerns data coverage only — not model quality, not control capability.
