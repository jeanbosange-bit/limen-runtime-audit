# EPISTEMIC_STATUS

## SUPPORTED

- **All 4 nodes** (L0, L1, L5, L20) — inherited `REFERENCE_LANDMARK_SUPPORTED` +
  independently confirmed `REPLICATED_ON_LANE_B`.
- **2 of 12 directed edges**: `L0→L5` (n=28, Holm p=0.0054, 8 families, 3 seeds, 18
  prompt_ids) and `L0→L20` (n=24, Holm p=0.0054, 8 families, 3 seeds, 17 prompt_ids).
  Both `STABLE` under leave-family-out. This is the **first statistically validated
  directed edge in this entire program** — V0 had zero.
- **1 route**: `L0-L20` (3 runs, 2 families, 2 seeds, built on the one supported
  constituent edge).

## LOW_SUPPORT

- 10 of 12 directed edges — real, non-zero observed counts (6-27 runs), broad family/
  seed spread in several cases, but did not clear the Holm-corrected p<0.05 bar against
  the (deliberately conservative) within-run label-shuffle null.
- All 6 undirected co-occurrence pairs — high raw counts (23-35 runs) but Holm p=1.0 in
  every case: the null model itself expects substantial co-occurrence by chance given
  how few distinct labels (4) are being reshuffled within short per-run label sets. This
  is a real, disclosed property of the chosen null, not a data quality problem.
- 10 route signatures.

## NOT_ESTABLISHED / UNKNOWN

- Whether the 2 supported edges' asymmetry (L0→L5/L0→L20 supported, but L5→L0/L20→L0
  not) reflects anything beyond a descriptive temporal regularity — no mechanism,
  reasoning, or causal claim is made or supported.
- 48 route signatures — `REFERENCE_ROUTE_NOT_SUPPORTED` (mostly 1-run
  `UNIQUE_ROUTE`s) — real observations, insufficient recurrence to validate.

## DERIVED

Every observed count, null distribution, Holm-adjusted p-value, and leave-family-out
label in this gate is a deterministic function of the 305 landmark encounter events
(themselves independently recomputed and cross-verified by
`reports/atlas-coverage-analysis-v1/`) — no new tensor computation, no npz reload.

## Bug disclosure (mission section 22)

**No implementation bug was found in this gate's own computation.** Two prior-gate bugs
are referenced for continuity but were already disclosed and resolved in their own
gates, not re-triggered here:
- `reports/multi-tensor-landmarks-v0/`'s step-alignment bug (fixed there, results used
  here are from the corrected re-run).
- `reports/atlas-coverage-analysis-v1/`'s `Attention_dispersion`/`Attention_concentration`
  field-name mismatch (disclosed there as `NOT_TESTED`; this gate's landmark-edge
  statistics do not depend on those two fields at all, so the gap does not propagate
  here).

One naming/notation decision was disclosed, not a bug: `SAME_STEP_TIE_POLICY.md`'s
extension for repeated-node routes (not evidenced by the surviving V0 files, decided and
frozen before analysis).

## Never silently converted (mission section 18/21)

No edge or route in any output file uses "causes", "drives", "forces", "mechanism
established", "reasoning path", "decision path", "failure path", "safe route", or
"unsafe route". `NETWORK_GRAPH.json` and `STAR_COMMAND_NETWORK_CONTRACT.json` use only
"precedes"/"transition"/"connection"/"route" vocabulary throughout.
