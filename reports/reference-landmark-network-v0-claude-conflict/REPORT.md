# REPORT — Reference Landmark Network V0

Status: **EXECUTED**. `real_new_generations = 0`. Builds the first network among the 4
validated `REFERENCE_LANDMARK_SUPPORTED` landmarks (L0, L1, L5, L20) from
`reports/multi-tensor-landmarks-v0/`, using only already-existing raw event/coordinate
data. Build script: `build_network.py`.

## 1. Are the four validated landmarks connected by reproducible routes?

**Weakly.** All 4 nodes are real and independently validated (inherited). All 12 possible
ordered pairs among them are *observed* in at least one run, but **zero edges clear the
frozen, Bonferroni-corrected significance bar** against a within-run label-shuffle null
(`EDGE_VALIDATION_RESULTS.json`) — the closest, `L0->L1`, reaches p=0.0647, short of even
the uncorrected 0.05. Connectivity exists descriptively; it is not statistically confirmed
at this sample size (21 eligible runs, only 10 with >=2 encounters).

## 2. What ordered landmark sequences actually occur?

12 distinct route signatures across 30 runs (`ROUTE_SIGNATURES.json`), real, not
illustrative. Examples actually observed: `L0-L20` (GI01), `L5-L1` (GI01), `L0~L1-L0`
(GI02, a genuine tie: L0 and L1 fire at the identical step 12), `L20-L0-L1` (GI02),
`L1-L5-L1` (AH01, a bounce/repeat), and the single most complete route in the whole
corpus, `L0-L5-L1-L20-L5` (AH02_1102, all 4 nodes touched). **The illustrative
`L0→L1→L5→L20` sequence from the mission brief was never observed as-is in any run.**

## 3. Which edges recur across runs?

None beyond chance at the frozen bar. Raw (unvalidated) support: `L0->L1` (4 runs, the
strongest raw signal), `L5->L1` (3 runs), `L5->L0` / `L1->L5` / `L0->L20` / `L20->L0` (2
runs each), the remaining 6 ordered pairs at 1 run each. Full table: `EDGE_MATRIX.csv`.

## 4. Which edges recur across prompt families?

`L0->L1`'s 4 supporting runs span AH01, AH02, GI01, GI02 — 2 families
(AMBIGUITY_HANDLING, GENERAL_INSTRUCTION) — the broadest family spread of any edge, but
still not statistically validated. No edge has STRUCTURED_OUTPUT or FACTUAL_GROUNDED
support (STRUCTURED_OUTPUT's only landmark presence is the single-node `L20` route, which
by definition has no edges).

## 5. Are there multiple route families?

Yes, but almost all are singletons. Only one signature (`L20` alone) clears the frozen
`COMMON_ROUTE` bar (>=3 runs, >=2 prompt_ids): 6 runs, both SO01 and SO02. One
`LOW_SUPPORT_ROUTE` pair of signatures (`L5` alone: 2 runs; `L5-L0`: 2 runs). The other 9
signatures are each `UNIQUE_ROUTE` (1 run only).

## 6. Are some landmarks frequently bypassed?

Every run bypasses at least 1 of the 4 nodes except one (`KVC_AH02_1102`, full coverage).
Across the 21 eligible runs, the modal pattern is bypassing 2-3 of the 4 nodes per run —
no single node is "always" reached even among runs that reach the network at all.

## 7. Which landmark appears most broadly across runs/families?

**L20** — run coverage 10/21 eligible runs (highest of the 4), 3 prompt families
(AMBIGUITY_HANDLING, GENERAL_INSTRUCTION, STRUCTURED_OUTPUT — the only node with
STRUCTURED_OUTPUT support at all). L0 is close behind (8 runs, 2 families). Full detail:
`NODE_PROPERTIES.json`.

## 8. Are transition patterns enriched near specific landmarks?

Yes, asymmetrically (`TRANSITION_OVERLAY.md`). **T1 (attention precedes attention-output)
is strongly enriched at L0** (the single highest `lag1` value, 0.587, of any of the 24
layers tested) and present at L1 (CO-VARIES); **absent at L5 and L20** (L20 even shows the
opposite sign). **T0 (entropy precedes curvature) is not enriched at any of the 4
landmarks** — its validated pattern lives at layers 7-14/17-19, entirely non-overlapping
with the confirmed network's own layers.

## 9. Is attention-sensitivity associated descriptively with any route/landmark?

Yes, one clear co-location (`ATTENTION_SENSITIVITY_ROUTE_OVERLAY.md`): **L0 sits at the
layer with by far the lowest mean `ATTENTION_OUTPUT_DIVERGENCE` of all 24 layers** (0.124
vs. a 1.932 corpus mean — roughly an order of magnitude lower). L1, L5, L20 show no
notable deviation from the corpus mean. This is a co-location observation only — not a
compressibility, fragility, or safety claim about L0.

## 10. How much territory remains outside the landmark network?

**Most of it.** 12 of 30 runs show zero confirmed-landmark encounters (all of MR01, MR02,
FG02, and — notably — all of FG01 despite FG01 being scored). FACTUAL_GROUNDED and
MATH_REASONING contribute **zero** routes to this network. Even among the 18 runs that do
touch the network, only 1 reaches all 4 nodes; total landmark-encounter step-density
across the whole 751-decode-step corpus is 4.7% (35/751). Full accounting:
`UNKNOWN_TERRITORY_AUDIT.json`.

## Network verdict

**REFERENCE_LANDMARK_NETWORK_PARTIAL.** All 4 nodes are solid (inherited validation);
observed connective structure between them is real and disclosed in full, but no edge
reaches the frozen statistical bar for a validated succession relationship at this sample
size. A sparse, largely family-restricted (AMBIGUITY_HANDLING/GENERAL_INSTRUCTION, plus
STRUCTURED_OUTPUT for L20 alone), mostly-unconnected-by-significance network is the honest
result — not forced toward SUPPORTED.

## Explicit non-claims

No node or edge in this network is a reasoning step, cognitive stage, decision, concept,
truth state, or hallucination state. No causal claim anywhere. `SRA-X: NOT_TESTED`.
`CONTROL: NO`.
