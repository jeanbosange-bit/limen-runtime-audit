# CONFLICT_DISCLOSURE

## What happened

A duplicate execution of the "REFERENCE LANDMARK NETWORK V0" mission was started by
Claude in `reports/reference-landmark-network-v0/` **after** a canonical execution of the
same mission had already been completed earlier in this project session by Codex.

- The duplicate mission was started without first listing the contents of
  `reports/reference-landmark-network-v0/`. Had that been done, the pre-existing
  canonical files (timestamped ~18:53, roughly 25 minutes before Claude's first write)
  would have been visible before any write occurred.
- Because the directory was not inspected first, Claude wrote its own version of every
  data/report file the mission specifies (`FROZEN_NODE_REGISTRY.json`, `ROUTE_INSTANCES.csv`,
  `ROUTE_SIGNATURES.json`, `EDGE_DEFINITION.md`, `EDGE_MATRIX.csv`,
  `EDGE_VALIDATION_PLAN.md`, `EDGE_VALIDATION_RESULTS.json`, `NODE_PROPERTIES.json`,
  `PROMPT_FAMILY_ROUTE_MATRIX.csv`, `LANDMARK_LOCAL_SIGNATURES.json`,
  `TRANSITION_OVERLAY.md`, `ATTENTION_SENSITIVITY_ROUTE_OVERLAY.md`,
  `UNKNOWN_TERRITORY_AUDIT.json`, `REFERENCE_LANDMARK_NETWORK_SCHEMA.json`,
  `TOPOLOGY_EPISTEMIC_STATUS.md`, `REPORT.md`, `build_network.py`) directly on top of the
  canonical files at those same paths, using a different internal schema (e.g. `node_id`
  instead of `landmark_id`, `~` instead of `+` for same-step ties, `from`/`to` instead of
  `source`/`target` edge fields). **These canonical data files were overwritten and their
  original content is not recoverable from this machine** — there is no git repository in
  this project and no other reachable agent session (`ListAgents` returned no reachable
  agents at the time of discovery).
- Four canonical metadata/test files survived untouched, because Claude had not yet
  reached them in its write sequence when the collision was noticed and the user
  intervened: `MANIFEST.json`, `SHA256SUMS.txt`, `STAR_COMMAND_INTEGRATION_CONTRACT.json`,
  and `test_network.py`. These four files still contain the canonical Codex run's
  hashes, contract, and test assertions, and were left exactly as found — nothing in
  them was edited, regenerated, or used to reconstruct the missing files.
- **Claude's duplicate schema is incompatible with the surviving canonical
  `test_network.py`** — that test suite asserts field names and conventions (e.g.
  `primary_nodes[].landmark_id`, `route_signature` using `"L0+L1"` for same-step ties,
  `edges[].source`/`target`/`supporting_runs`/`support_prompt_groups`) that do not match
  what Claude's duplicate script produced. Running `test_network.py` against Claude's
  duplicate output would fail; it has not been run against the duplicate output, and the
  duplicate output has been removed from the canonical directory specifically to avoid
  that confusion.
- No attempt was made to edit, re-hash, or otherwise rewrite the canonical provenance
  files after this was discovered. No attempt was made to infer or reconstruct the
  original canonical data files' content from the surviving metadata/test files.

## Known canonical result (as reported by the user, from the surviving `MANIFEST.json`
and the canonical run's own reporting, not re-derived by Claude)

```
REFERENCE_LANDMARK_NETWORK_PARTIAL
RUNS ANALYZED: 30
FROZEN NODES: L0 / L1 / L5 / L20
RUNS WITH >=1 LANDMARK: 18
RUNS WITH >=2 LANDMARKS: 10
UNIQUE ROUTE SIGNATURES: 11
COMMON ROUTES: 1
VALIDATED EDGES: 0
LOW-SUPPORT EDGES: 5
CROSS-FAMILY EDGES: 5
MOST BROADLY OBSERVED LANDMARK: L20 (10 runs)
MOST FREQUENT ROUTE: L20 (6 runs)
ROUTES WITH NO LANDMARK: 12
TESTS: 9/9 PASS
HASH VERIFICATION: 20/20 PASS
```

## Contents of this quarantine directory

The 17 files below are Claude's duplicate-attempt output, moved here in full (not
deleted, not edited since being moved) so nothing produced during the incident is lost,
while keeping the canonical directory free of incompatible duplicate schema files:

- ATTENTION_SENSITIVITY_ROUTE_OVERLAY.md
- build_network.py
- EDGE_DEFINITION.md
- EDGE_MATRIX.csv
- EDGE_VALIDATION_PLAN.md
- EDGE_VALIDATION_RESULTS.json
- FROZEN_NODE_REGISTRY.json
- LANDMARK_LOCAL_SIGNATURES.json
- NODE_PROPERTIES.json
- PROMPT_FAMILY_ROUTE_MATRIX.csv
- REFERENCE_LANDMARK_NETWORK_SCHEMA.json
- REPORT.md
- ROUTE_INSTANCES.csv
- ROUTE_SIGNATURES.json
- TOPOLOGY_EPISTEMIC_STATUS.md
- TRANSITION_OVERLAY.md
- UNKNOWN_TERRITORY_AUDIT.json

These files reflect real computation on the same underlying real, existing capture data
(same source facts: 30 runs, 4 confirmed landmarks, 18/30 runs with >=1 landmark, etc. —
Claude's independently-computed numbers agree with the canonical result on every one of
these headline figures) but use a different internal schema and are **not** part of, and
must not be treated as, the canonical deliverable set.

## Lesson for future sessions (also saved to persistent memory)

Before writing multiple files into any `reports/<gate-name>/` output directory for a
mission of this kind, always run `ls -la` on that directory first. This program
repeatedly reuses a small set of conventional filenames (`MANIFEST.json`,
`SHA256SUMS.txt`, `REPORT.md`, etc.) across many gates; a directory that already exists
with content must be treated as someone else's (or one's own prior) in-progress or
completed work until proven otherwise, never as an empty target.
