# TOPOLOGY_EPISTEMIC_STATUS

## Nodes: CONFIRMATORY (inherited)

All 4 nodes (L0, L1, L5, L20) carry `REFERENCE_LANDMARK_SUPPORTED` status, established in
`reports/multi-tensor-landmarks-v0/`. Not re-validated here, only reused as the frozen
node set.

## Routes: MEASURED / DERIVED

`ROUTE_INSTANCES.csv` and `ROUTE_SIGNATURES.json` are direct, deterministic derivations
from the same raw event list already used to validate the landmarks — no new statistical
test, no ambiguity in what is reported. Route signatures with `n_runs>=3` and
`n_prompt_ids>=2` are `COMMON_ROUTE` by a frozen, disclosed rule (mission section 11); the
one such route found (`L20` alone, 6 runs) is real but **`FAMILY_CONDITIONED`**
(STRUCTURED_OUTPUT only, both its prompt_ids) — not cross-family.

## Edges: TESTED, NOT_ESTABLISHED at the frozen bar

All 12 ordered pairs among the 4 nodes are `OBSERVED` (each pair occurs in >=1 run), but
**zero clear the frozen Bonferroni-corrected significance bar** (`EDGE_VALIDATION_RESULTS.json`),
and only one (`L0->L1`, p=0.0647) would even have cleared an *uncorrected* p<0.05. This is
a genuine, disclosed negative result at this sample size (21 eligible runs, 10 runs with
>=2 encounters) — **not evidence that no real succession relationship exists**, only that
this specific frozen test does not detect one here. Edge `direction_consistency`,
`step_distance`, and raw support counts in `EDGE_MATRIX.csv` remain useful descriptive
information even though no edge is `VALIDATED`.

## Node centrality: DESCRIPTIVE_NETWORK_PROPERTY only

Betweenness centrality is 0.0 for all 4 nodes (`NODE_PROPERTIES.json`) because the
observed-any graph is near-complete (every ordered pair has >=1 supporting run), so no
node lies exclusively on a shortest path between two others. This is a real, if
unremarkable, descriptive property — not evidence of any node's "importance."

## Local signatures: MIXED GRANULARITY, disclosed

`LANDMARK_LOCAL_SIGNATURES.json` mixes genuinely step-resolved observables (H, attention,
attention-output, probability, all at each landmark's own real encounter events) with
(run,layer)-structural K/V observables broadcast across step — the file's own
`granularity_disclosure` field states this explicitly, consistent with
`reports/multi-tensor-atlas-v0/MULTI_TENSOR_SCHEMA.json`'s standing warning.

## Overlays: SECONDARY / EXPLORATORY, NOT_ESTABLISHED

Both `TRANSITION_OVERLAY.md` and `ATTENTION_SENSITIVITY_ROUTE_OVERLAY.md` are single
descriptive lookups against already-frozen external results. Neither ran a new
significance test, and neither may be used to redefine any node or edge (mission sections
14-15, honored). The L0/attention-sensitivity and L0-L1/T1-transition co-locations are
real, disclosed, numerically notable — but explicitly `NOT_ESTABLISHED` as anything beyond
a co-location observation.

## Network verdict basis

Given 4/4 confirmed nodes but 0/12 statistically validated edges, the correct verdict
(per `REPORT.md`'s allowed-status list) is **PARTIAL**, not SUPPORTED and not
NOT_SUPPORTED — nodes are solid, connective structure between them is observed but not
statistically confirmed at this sample size.
