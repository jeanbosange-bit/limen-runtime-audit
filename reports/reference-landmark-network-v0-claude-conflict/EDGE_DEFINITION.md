# EDGE_DEFINITION

Frozen before any edge is computed. Applies only to the 4 nodes in
`FROZEN_NODE_REGISTRY.json` (L0, L1, L5, L20).

## Source events

A landmark "encounter" = one raw event from
`reports/multi-tensor-atlas-v0/LANDMARK_CANDIDATES_RAW.json` (the same 107-event list
underlying the validated landmark gate), filtered to `layer in {0,1,5,20}`. 35 of the 107
events qualify. Each encounter has a `(run_id, layer, step)`.

## Per-run ordered sequence

For each run, sort its qualifying encounters by `step` ascending. Ties (two different
landmarks at the identical step) are **not** ordered — see CO_OCCURRENCE below.

## Edge types (mutually exclusive per observed instance)

- **CO_OCCURRENCE**: two different landmarks (A, B) encountered at the exact same `step`
  within the same run. Ordering is not inferred.
- **DIRECT_SUCCESSION**: landmark B's encounter is the very next entry (by step) after
  landmark A's encounter in that run's own sorted qualifying-encounter list — i.e., no
  other qualifying landmark encounter (of A, B, or either other node) falls at a step
  strictly between them. A same-landmark repeat (e.g. A occurring twice with nothing else
  between) is not itself a DIRECT_SUCCESSION edge between distinct nodes; it is recorded
  separately as a repeat in `ROUTE_INSTANCES.csv`.
- **INDIRECT_SUCCESSION**: landmark B is encountered after landmark A in the same run, but
  with at least one other qualifying landmark encounter (any node, including a repeat of A
  or B) at an intervening step.
- **NO_OBSERVED_LINK**: the ordered pair (A→B) never occurs in any run under either
  succession type above.

## What is NOT an edge

- No edge is inferred across runs (a run's landmark sequence never connects to another
  run's).
- No edge is inferred from raw coordinate-row proximity — only from confirmed landmark
  encounters (the same 4-node, frozen-threshold events used in the validated landmark
  gate).
- No edge is ever labeled causal. Allowed direction language: "A precedes B in run X" —
  never "A causes B."

## Per-edge (ordered pair) reporting unit

There are 4x3 = **12 possible ordered pairs** among the 4 nodes (every A≠B direction
counted separately, since DIRECT/INDIRECT succession is directional). For each ordered
pair, `EDGE_MATRIX.csv` reports:

- `n_runs_supporting` (DIRECT + INDIRECT succession combined, and separately DIRECT only)
- `n_prompt_ids_supporting`, `n_families_supporting`
- `median_step_distance`, `step_distance_std` (over DIRECT-succession instances only,
  since INDIRECT distances are not a meaningful single-hop measure)
- `direction_consistency`: whether the reverse pair (B→A) is also observed anywhere, and
  at what relative frequency
- `n_co_occurrence_instances` (reported once per unordered pair, not duplicated per
  direction)

## Support unit (mission section 7 — explicit, not row-count-based)

Primary support unit is **RUN** (one run supports an edge at most once per succession
type, regardless of how many times the pattern repeats within that run) and, separately,
**PROMPT_ID / PROMPT_FAMILY** (does the edge recur across more than one prompt group, not
just multiple seeds of the same prompt).
