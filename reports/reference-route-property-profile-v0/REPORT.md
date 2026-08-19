# REPORT — Reference Route Property Profile V0

Status: **EXECUTED, READ-ONLY**. 0 new generations. Characterizes what measurable
tensor properties accompany the validated composite route `L0→L5→L20`
(`REFERENCE_ROUTE_SUPPORTED`, `composite-reference-route-v0`). This is a property
characterization gate — not causal explanation, failure prediction, or control.

## Primary question: what measurable properties characterize ROUTE_FULL runs?

**Mostly: nothing yet distinctive.** Of 12 predeclared tensor properties tested across
4 comparison groups each (48 tests, Holm-corrected), **0 reach `ROUTE_PROPERTY_SUPPORTED`**,
9 reach `ROUTE_PROPERTY_LOW_SUPPORT`, 3 `ROUTE_PROPERTY_NOT_ESTABLISHED`. The route's
own sequence-level validation (established in the prior gate) is real and unaffected by
this finding — but this gate's own contribution is that the route does **not yet** carry
a distinctive local tensor signature beyond what any landmark-bearing execution shows.

## Groups (`ROUTE_GROUP_REGISTRY.json`, partition verified over all 144 runs)

| Group | n runs |
|---|---|
| A. ROUTE_FULL | 8 |
| B. L0_L5_ONLY | 12 |
| C. L0_L20_ONLY | 8 |
| D. OTHER_LANDMARK_ROUTE | 59 |
| E. NO_LANDMARK | 40 |
| F. SHORT_NON_EVALUABLE | 17 |

## Route occurrence profile (`ROUTE_OCCURRENCE_PROFILE.json`)

8 runs, 7 distinct prompts, 6/12 families, 2/3 seeds (2102: 3, 2103: 5 — **absent in
2101**). Generation lengths: min 48, median 130, max 224 steps — concentrated toward
the long end of Lane B's overall distribution. `L0` occurs almost immediately (step 1
in 6 of 8 runs). Step-distance medians: `L0→L5` 64.5 (IQR 44-72), `L0→L20` 94.5 (IQR
49-130), `L5→L20` 38.5 (IQR 37-53) — broadly distributed, not tightly localized.

## Tensor property comparisons (`ROUTE_GROUP_COMPARISONS.json`, `ROUTE_PROPERTY_REGISTRY.json`)

The single Holm-significant result across all 48 tests — `Oattn_delta_norm` vs.
`NO_LANDMARK` (p=0.0398, effect -0.43) — is **not distinctive to this route**: the same
property is statistically indistinguishable between ROUTE_FULL and `L0_L5_ONLY`
(p=0.50, near-identical medians 0.313 vs. 0.305). Since `Oattn_delta_norm` is literally
one of the 4 signals that *defines* a landmark encounter, this result confirms the
detection mechanism is working, not that this specific 3-node route has a unique
signature. No other property reached full statistical support after correction.

| Status | Count | Properties |
|---|---|---|
| SUPPORTED | 0 | — |
| LOW_SUPPORT | 9 | H_norm, H_curvature, Q_norm, V_norm, A_dispersion, A_concentration, Oattn_delta_norm, probability_entropy, probability_top2_gap |
| NOT_ESTABLISHED | 3 | K_norm, Oattn_norm, logit_norm |

## Confound audits (`FAMILY_CONFOUND_AUDIT.json`, `SEED_ROBUSTNESS_AUDIT.json`,
`LENGTH_CONFOUND_AUDIT.json`)

3 LOW_SUPPORT properties (`H_curvature`, `Oattn_delta_norm`, `probability_entropy`) are
`FAMILY_CONFOUNDED` — their apparent significance depends on which family is included.
3 (`Oattn_delta_norm`, `probability_entropy`, `probability_top2_gap`) are
`SEED_SENSITIVE`. ROUTE_FULL's own length distribution (median 130 steps) is flagged as
a standing `LENGTH_CONFOUND` risk for every LOW_SUPPORT+ property — not exhaustively
tested given n=8 is too small to stratify further, disclosed rather than ignored.

## Methodological note (disclosed before, not after, seeing results)

`NO_LANDMARK` runs have no landmark encounter by construction, so their comparison
population uses per-run-layer **mean** values (already computed, reused read-only) at
layers 0/5/20 rather than a specific encounter-step value used for every other group —
a coarser, asymmetric comparison, stated plainly in `FROZEN_PROPERTY_PROTOCOL.md` before
any p-value existed.

## Product translation (`ROUTE_PASSPORT_L0_L5_L20.json`, `STAR_COMMAND_ROUTE_CARD_CONTRACT.json`,
`ROUTE_MAP_LINK.json`)

Three-layer passport (human meaning / explanation / technical detail) built per the
mandatory language rules — no p-values or tensor jargon at Level 1, mandatory French UX
phrases included verbatim, forbidden labels (SAFE/NORMAL/GOOD/ERROR/REASONING ROUTE)
never used. The route card correctly shows "Supported reference route" (the route's own
validated sequence status) while Level 2's "general property profile" text honestly
notes that no tensor property has yet been shown distinctive to this specific ordering
— the two facts coexist without contradiction.

## Positional language only

L0, L5, L20 are described only by their position in the route ("L0 occurs first", "L5
is the intermediate confirmed landmark", "L20 occurs later") — never as initiating,
transforming, or terminating anything.

## What this gate does not establish

No causal claim. No failure prediction. No control. No safe/unsafe classification.
`CONTROL: NO`. `SRA-X: NOT_TESTED`. `VBE: NOT_TESTED`.

## Verdict

**`REFERENCE_ROUTE_PROPERTY_PROFILE_PARTIAL`.** The route's sequence-level validity is
inherited and unaffected. This gate's own contribution — a genuine, non-inflated
characterization — finds real but non-distinctive tensor differences (9 LOW_SUPPORT
properties, several confound-flagged) and 3 properties with no detectable difference at
all. `SUPPORTED` (property-level) is not claimed for any property; `PARTIAL` honestly
reflects a route that is statistically real but not yet measurably distinctive at the
tensor level.
