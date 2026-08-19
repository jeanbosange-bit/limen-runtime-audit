# ATTENTION_SENSITIVITY_ROUTE_OVERLAY — exploratory, descriptive only

Maps V1's `ATTENTION_OUTPUT_DIVERGENCE` (the frozen, `ATTENTION_SENSITIVITY_SIGNAL_SUPPORTED`
compression-sensitivity target) onto the 4 confirmed landmark layers, using the same
region-level rows already computed in `reports/atlas-kv-compression-v0/execution/ANALYSIS_TABLE.json`
(20,736 rows, 864 rows per layer). No new computation of the sensitivity target itself.

## Mean ATTENTION_OUTPUT_DIVERGENCE by landmark layer vs. corpus

| Landmark | Layer | Mean ATTENTION_OUTPUT_DIVERGENCE (n=864 rows) | Corpus-wide mean (all 24 layers, n=20736) |
|---|---|---|---|
| L0 | 0 | **0.124** | 1.932 |
| L1 | 1 | 1.618 | 1.932 |
| L5 | 5 | 1.360 | 1.932 |
| L20 | 20 | 2.015 | 1.932 |

**L0 sits at the layer with by far the lowest mean attention-output divergence of all 24
layers** (next-lowest is layer 16 at 1.088 — L0's 0.124 is roughly an order of magnitude
below every other layer, including its own network neighbors). L1, L5, and L20 are all
close to or slightly above the 24-layer corpus mean — unremarkable by this measure.

## Reading — explicitly bounded

This is a **co-location observation only**: L0 happens to sit at a layer that, under V1's
tested compression protocol, shows unusually low measured attention-output sensitivity.
**This does NOT mean**:

- L0 is "compressible" — compressibility was never tested as a property of L0 specifically,
  only of the layer-wide row population it belongs to.
- L0 is "fragile" or "safe" for any compression decision.
- Any landmark's presence predicts or is predicted by attention-output sensitivity — no
  such test was run.

L1, L5, L20 show no notable enrichment or depletion relative to the corpus mean. This
overlay is reported once, descriptively, and is not used anywhere else in this gate to
define, validate, or rank a landmark or edge.
