# FROZEN_ANALYSIS_PROTOCOL

Frozen before any Lane B data is disaggregated or inspected beyond the aggregate
Lane B manifest numbers already reported. This is a READ-ONLY analysis gate — no new
generations, no modification of any source artifact.

## Data source (read-only)

`reports/bcc-atlas-expansion-v1/ATLAS_DEEP/run_index.jsonl` (144 entries) and
`raw_captures/*.npz` (144 files). Verified against the existing `SHA256SUMS.txt`
(160/160) before use — see `SOURCE_PROVENANCE.json`.

## Frozen landmark registry (reused, not redefined)

L0 (layer 0), L1 (layer 1), L5 (layer 5), L20 (layer 20) — from
`reports/multi-tensor-landmarks-v0/FROZEN_LANDMARK_REGISTRY.json`. Detection rule reused
verbatim: within (run, layer), z-score `H_curvature`, `Oattn_delta_norm`, `A_dispersion`,
`probability_entropy` across decode steps; a step is an encounter if `|z|>1.5` on `>=3`
of the 4 signals. No threshold, layer, or signal is altered here.

## Eligibility labeling (mandatory, mission section 4)

- `n_steps < 4`: **`LANDMARK_NOT_EVALUABLE_SHORT_EXECUTION`** — the run's generation was
  too short for the within-run z-score to be computed at all. Never called failed or
  invalid.
- `n_steps >= 4` AND no encounter found at any of the 4 nodes: **`NO_LANDMARK_OBSERVED`**
  — the run WAS evaluable and simply did not produce a qualifying encounter.
- `n_steps >= 4` AND >=1 encounter: normal landmark-bearing run.

These two null categories (`LANDMARK_NOT_EVALUABLE_SHORT_EXECUTION` vs.
`NO_LANDMARK_OBSERVED`) are never merged or conflated anywhere in this gate's outputs.

## Route signature notation (frozen, per mission section 8)

- `>` : one landmark's encounter step strictly precedes another's in the same run
  (temporal succession, any distance — this gate does not distinguish direct/indirect
  succession the way `reference-landmark-network-v0` did; it records the full ordered
  sequence).
- `~` : two (or more) landmarks encountered at the exact identical step (tie) — joined
  alphabetically by node id for a canonical string.
- Examples (illustrative only, per mission text): `L0`, `L0>L5`, `L1~L5>L20`.
- A route with zero encounters is recorded as the literal string `NONE`, distinct from
  any real signature.

## Local window definition (mission section 9)

For every landmark encounter at `(run_id, layer, step)`, the local window is
`{t-1, t, t+1}` in decode-step index, where available (a window truncates at the run's
own generation boundary — e.g. `t-1` is absent for `step=0`; nothing is fabricated to
fill a missing edge).

## Descriptor formulas (reused verbatim, mission sections 10-14 — "use existing validated
metric implementations")

All formulas below are copied unchanged from `reports/multi-tensor-atlas-v0/build_atlas.py`
and `reports/atlas-kv-compression-v0/execution/kv_capture_lib.py` — no new geometry
metric is invented:

- `H_norm(l,t) = ||h[l,t]||_2`; `H_delta_norm = H_norm(t) - H_norm(t-1)`;
  `H_cosine_step_movement = cos(h[t-1], h[t])`;
  `H_curvature = 1 - cos(h[t]-h[t-1], h[t-1]-h[t-2])`.
- `Q_norm`, `K_norm`, `V_norm`: `||x[l,t]||_2` per tensor (Q/K flattened across their own
  head dimension before norming, matching K/V's existing head-explicit storage
  convention); `*_delta_norm` and `*_cosine_step_movement` analogous to H's.
- `Attention_dispersion` = mean-over-heads row-normalized entropy over causal keys
  `0..t`; `Attention_concentration` = mean-over-heads max attention weight;
  `Attention_max_mass` = the single largest per-head weight at that step.
- `Oattn_norm`, `Oattn_delta_norm`, `Oattn_cosine_step_movement`: identical formulas to H,
  applied to `attn_output`.
- `probability_entropy`, `probability_top2_gap`: `entropy_from_logits` /
  `top2_gap_from_logits`, reused verbatim.
- `logit_norm`: `||logits[step]||_2` over the full 151,936-dim vector, reported as a
  DERIVED quantity distinct from the probability summaries (mission section 14 —
  "keep logits and probability summaries distinct").

## Epistemic labels (mission section 21, exact set, no substitutions)

`MEASURED`, `DERIVED`, `SUPPORTED`, `LOW_SUPPORT`, `UNKNOWN`, `NOT_ESTABLISHED`,
`NOT_TESTED`. Never: correlation→causality, recurrence→mechanism, landmark→cognitive
stage, unknown→unsafe, no-landmark→failure.

## What this gate does not do

No new generation. No new landmark candidate search. No edge promotion. No control,
causal, or SRA-X claim. No ML/classifier/predictor training. No redefinition of the
frozen landmark thresholds or region rule.
