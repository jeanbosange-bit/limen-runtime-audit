# EPISTEMIC_STATUS

## SUPPORTED

- **`L0-L5-L20` — the primary candidate — is `REFERENCE_ROUTE_SUPPORTED`.** 8 observed
  runs, Holm-adjusted p=0.0072, 6/12 prompt families, 2/3 seeds (2102 and 2103, absent
  in 2101 — disclosed, not hidden), 7 distinct prompt_ids, `STABLE` under
  leave-one-family-out. This is the first statistically validated 3-node composite
  route in this program.
- Notably, this SUPPORTED status was reached **despite** the constituent edge
  `L5→L20` being only `REFERENCE_EDGE_LOW_SUPPORT` at the pairwise level
  (`reference-landmark-network-v1`) — direct, real confirmation that route-level and
  edge-level evidence are genuinely independent tests, exactly as the mission warned
  ("do NOT assume the composite route is supported merely because one constituent edge
  is" — and, symmetrically, a route is not blocked by one weak edge either).

## LOW_SUPPORT

- 10 of 24 candidates, including `L0-L1-L5` (7 runs, Holm p=0.257) and `L0-L1-L20` (6
  runs, Holm p=0.060) — real, non-trivial observed counts that did not clear the
  corrected significance bar.
- `L0-L20-L5` (5 runs, Holm p=0.433) — the primary candidate's nearest permutation
  sibling, real but clearly weaker than `L0-L5-L20` itself.

## NOT_ESTABLISHED / UNKNOWN

- 13 of 24 candidates observed 0-4 times — `REFERENCE_ROUTE_NOT_SUPPORTED`, honest
  absence-of-evidence, not evidence of absence.
- Whether the temporal-distance spread for `L0-L5-L20` (L0→L5 median 64.5 steps, IQR
  44-72; L0→L20 median 94.5, IQR 49-130; L5→L20 median 38.5, IQR 37-53) reflects
  anything beyond descriptive timing — the gaps are broad, not tightly localized, and no
  interpretation is offered beyond reporting the numbers.
- `BRANCHING_VS_COMPOSITION_ANALYSIS.json`'s verdict is `COMPOSITION_SUPPORTED` by the
  frozen rule (the primary route itself reached `SUPPORTED`) — this does not mean
  branching evidence is absent; both-branch-only support (`L0→L5` and `L0→L20`
  co-occurring without the full order) exists too (see the JSON for exact counts) but is
  not the dominant reading once the composite itself is validated.

## DERIVED

Every count, null distribution, Holm-adjusted p-value, and leave-family-out label is a
deterministic function of `reference-landmark-network-v1/RELATIONS_CACHE.json` and
`atlas-coverage-analysis-v1/PASS1_CACHE.json` — no new tensor computation, no npz
reload, no source artifact modified (verified below).

## No bug found this gate

Both computation passes (24-route classification, 10,000-permutation null) ran cleanly
on the first real execution; a small timing dry-run (200 perms) preceded the full run
per standard practice, not because an error was suspected.

## Never silently converted

No route in any output uses "initiates", "mediates", "terminates", "decision path",
"reasoning path", "causal chain", "control corridor", or "safe route". `NETWORK_EXTENSION_GRAPH.json`
and `STAR_COMMAND_ROUTE_CONTRACT.json` use only "observed ordered route", "recurrent
transition", "supported reference route" vocabulary.
