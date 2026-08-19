# Tensor Runtime Fingerprint Feasibility V0 — frozen protocol

Frozen on 2026-08-18 before inspecting fingerprint class separation.

## Question and units

The primary unit is one real Lane B run. The primary comparison is Euclidean distance after train-independent robust scaling between bounded deterministic summaries: same prompt/different seed versus different prompt. Prompt IDs, not rows, are the challenge classes. No raw flattened tensor is a fingerprint.

## Prefixes

`FULL` uses every available generated position. `FIRST_2`, `FIRST_4`, and `FIRST_8` use the first 2/4/8 generated positions, starting at `prompt_len`. A run is eligible only when `n_steps >= k`; ineligible runs remain reported and are never globally discarded. Tensor positions from the prompt are not substituted for unavailable generated steps.

## Frozen features

Feature families and statuses are frozen in `FEATURE_REGISTRY.json`. Vector tensors use norm, delta-norm, cosine-change, curvature where defined, and across-position/layer dispersion summaries. Q/K/V additionally retain head dispersion. Attention uses normalized entropy, concentration, maximum mass, and normalized position dispersion. Logits use entropy, top-1 probability, top-2 gap, norm, delta-norm, and cosine-change. Features are layer-resolved where the source is layer-resolved.

## Scaling and distance

Distance analyses use robust scaling (median and IQR, IQR floor `1e-9`) fitted across the eligible run set for that frozen analysis. Missing higher-order summaries caused by short prefixes are imputed to the eligible-set training median and accompanied by explicit eligibility counts. Constant features are removed. Euclidean distance is divided by the square root of retained feature count.

Primary distributions are all 144 intra-prompt seed pairs and all inter-prompt run pairs. Controls use different prompts within the same family and generated-length matched pairs (`absolute n_steps difference <= 2`). Reported effect is common-language separation `P(inter distance > intra distance)` and rank-biserial `2P-1`; 95% bootstrap intervals use 2,000 resamples, RNG seed `20260818`.

## Identification frozen after distance definition

For each held-out seed, the two other seeds form one prompt centroid. Scaling is fit on reference runs only. The held-out run is ranked against all 48 centroids. The three seeds rotate. Same-family identification restricts candidates to the four prompt IDs in the true family. No identical prompt ID crosses from held-out run to reference except through the explicitly permitted other seeds that define the challenge reference.

## Layer and ablation analysis

Ablations are `H`, `Q`, `K`, `V`, `ATTENTION`, `ATTENTION_OUTPUT`, `LOGITS`, `H+Q+K+V`, `ATTENTION+ATTENTION_OUTPUT`, and `ALL`. Layer localization repeats distance separation per layer for H/Q/K/V/Attention/Attention Output. It is descriptive and not causal.

## Frozen signal labels

- `SUPPORTED`: full-run common-language separation >= 0.75, same-family separation >= 0.65, global top-1 >= 20%, and same-family top-1 >= 50%.
- `PARTIAL`: common-language separation >= 0.65 or global top-1 >= 10%, without meeting `SUPPORTED`.
- `NOT_SUPPORTED`: neither threshold is met.
- `CONFOUNDED`: a globally supported-looking signal has same-family top-1 <= 30%, or absolute length-distance Spearman rho >= 0.50 with length-matched separation below 0.65.
- `TECHNICALLY_INVALID`: source/shape/hash integrity or required analysis fails.

Length confound is `YES` under the `CONFOUNDED` rule, `PARTIAL` for absolute rho >= 0.25 or a >=0.10 fall in length-matched separation, otherwise `NO`.

## Boundaries

The result is not authentication, attestation security, cryptographic entropy, ownership proof, control, steering, or a random-number source. No new generations are authorized. Landmark relation is secondary and cannot alter the primary verdict.
