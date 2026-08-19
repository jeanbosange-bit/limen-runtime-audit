# SAME_STEP_TIE_POLICY

Frozen before any edge statistic is computed. Reuses the canonical convention evidenced
in the surviving Reference Landmark Network V0 contract/test files
(`reports/reference-landmark-network-v0/test_network.py`,
`STAR_COMMAND_INTEGRATION_CONTRACT.json`) — **not invented for this gate**.

## Evidence for the canonical convention (quoted from the surviving V0 test suite)

```python
def test_routes_use_first_actual_step_not_row_position(self):
    ...
    self.assertEqual(route["landmark_sequence"], "-".join(expected) or "NONE")

def test_same_step_encounters_are_not_forced_into_succession(self):
    route = next(r for r in self.routes if r["run_id"] == "KVC_GI02_1102")
    self.assertEqual(route["route_signature"], "L0+L1")
    edge = next(e for e in self.edges["edges"] if e["source"] == "L0" and e["target"] == "L1")
    self.assertNotIn("KVC_GI02_1102", edge["supporting_runs"])
```

And `MANIFEST.json`'s `"alignment": "ACTUAL_STEP_KEYED_WITH_TIES_PRESERVED"`.

## Frozen rules (reused verbatim from the above)

1. **What counts as same-step**: two (or more) landmark encounters in the same run
   sharing the identical decode-step index.
2. **Notation**: same-step ties are joined with `+` (e.g. `L0+L1`), alphabetically
   ordered within the tie group. Successive (different-step) encounter groups are
   joined with `-` (e.g. `L0-L5+L20-L1` = L0 first, then L5 and L20 tied, then L1) —
   `-` is the separator already used in V0's `landmark_sequence` field; `+` is the
   separator already used in V0's `route_signature` field for ties. Both are reused
   here unmodified.
3. **Same-step nodes generate CO_OCCURRENCE only.** They never generate
   `DIRECT_SUCCESSION` or `INDIRECT_SUCCESSION` between themselves, in either
   direction. This is directly verified by the surviving V0 test: `KVC_GI02_1102` (L0
   and L1 tied at step 12) must NOT appear in `supporting_runs` for the `L0→L1` edge.
4. **Ordering behavior when ties occur**: a tie group is treated as a single
   indivisible position in the run's ordered encounter sequence for succession
   purposes — nodes before the tie group precede every node in it (as a group);
   nodes after the tie group follow every node in it (as a group). No order is
   claimed *within* the tie group.
5. **Null model entry**: the group-preserving permutation (`FROZEN_NETWORK_PROTOCOL.md`)
   shuffles which node-label occupies which step-position within a run, but **tie-group
   structure itself (which step-positions are grouped together) is preserved** — a
   permutation never creates a new tie or dissolves an existing one; it only relabels
   which node(s) occupy the already-existing single/tied positions.

## One disclosed extension (not evidenced by the surviving V0 files, decided now, frozen
before analysis)

The surviving V0 files evidence `landmark_sequence` as a **deduplicated, first-occurrence
-only** ordered list (one entry per distinct node, at its earliest step) and
`route_signature` as encounter-preserving for the one tie example available, but do not
evidence how V0 handled a **repeated visit to the same node** (e.g. a run touching L0,
then L5, then L0 again). This gate's `ROUTE_SIGNATURES.csv` preserves every encounter
group in temporal order, **including repeats** (e.g. `L0-L5-L0`), since that is more
information-preserving than deduplication and does not contradict any evidenced V0
behavior. This choice is disclosed here, not silently assumed.
