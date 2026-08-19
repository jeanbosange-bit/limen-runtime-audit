# Cross-Environment Fingerprint Validation V0 — frozen protocol

Frozen on 2026-08-19 before campaign generation.

## Panel and execution

The panel contains exactly one challenge per each of 12 families. Selection uses only V0 seed-stability class and observed generation length: prefer `MODERATE`, then minimize distance to the within-family median length, then prefer the lower maximum length. It never uses fingerprint separation. Seeds remain 2101/2102/2103 so environment differences can be paired at identical challenge and seed. This design leaves only one identification candidate per family, so same-family identification is structurally non-discriminative and will be reported as `NOT_ESTABLISHED`, not as a misleading 100% score.

The existing Lane B CUDA/bfloat16/eager captures are the reference environment. New AVAILABLE_NOW environments are CPU/eager with float32, bfloat16, and float16. Planned new generations: `12 × 3 × 3 = 108`. No model, tokenizer, generation setting, prompt, feature definition, or attention implementation changes.

## Frozen fingerprint

The exact `reports/tensor-runtime-fingerprint-v0/FEATURE_REGISTRY.json` registry is reused. Prefixes are FULL and first 2/4/8 generated positions. No raw tensor flattening and no new feature selection are allowed. Scaling is fitted on the baseline reference only.

## Distances

- `D_seed`: same challenge, baseline environment, different seed.
- `D_env`: same challenge and seed, baseline versus alternate environment.
- `D_challenge`: different challenge, baseline environment.

Euclidean distance is divided by the square root of retained robust-scaled feature count. Robust median/IQR scaling is fitted only on baseline runs.

## Frozen stability classes

Baseline thresholds are computed before alternate-environment labels are inspected.

- `FINGERPRINT_PRESERVED`: median D_env ≤ baseline D_seed p90 and cross-environment top-1 is no more than 15 percentage points below baseline panel top-1.
- `FINGERPRINT_SHIFTED`: not preserved, but median D_env < median D_challenge and research verification ROC-AUC ≥ 0.70.
- `FINGERPRINT_BROKEN`: neither condition.
- `NOT_TESTED`: no valid probe set.

Overall `SUPPORTED` requires at least two of three alternate environments preserved plus ALL top-1 ≥50% and ROC-AUC ≥0.70. Same-family identification is excluded from the verdict because this frozen panel contains only one challenge per family. `PARTIAL` requires any preserved/shifted environment and ROC-AUC ≥0.70. Otherwise `NOT_SUPPORTED`. A signal is `CONFOUNDED` if environment identity alone explains separation while challenge recovery collapses to chance.

## Identification and verification proxy

Baseline enrollment is the mean of three seed fingerprints per challenge. Alternate-environment runs are probes. Report top-1/top-5; report same-family top-1 as `NOT_ESTABLISHED` for this one-candidate-per-family panel. The research-verification threshold is selected from baseline-only genuine/impostor comparisons at the baseline EER operating point, then frozen and applied to alternate probes. FAR-like, FRR-like, ROC-AUC, and cross-set EER are research proxies—not authentication metrics.

## Boundaries

No authentication, PQC, entropy, ownership, unclonability, anti-replay, attestation security, production security, or control claim is permitted.
