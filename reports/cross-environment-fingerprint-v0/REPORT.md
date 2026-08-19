# Cross-Environment Fingerprint Validation V0

## Verdict

**CROSS_ENV_FINGERPRINT_ROBUSTNESS_SUPPORTED** under the tested comparison: historical CUDA bfloat16/eager enrollment versus CPU float32, bfloat16, and float16/eager probes for 12 frozen challenges and three seeds.

Baseline rotating-seed top-1 is 94.44%; pooled cross-environment top-1 is 92.59% and top-5 is 100.00%. The research verification proxy gives FAR-like 7.66%, FRR-like 13.89%, ROC-AUC 0.9306, and EER 10.19%.

Same-family identification is **NOT_ESTABLISHED**: one frozen challenge per family makes the candidate set non-discriminative. This is not reported as a synthetic 100% result.

## Boundaries

The baseline environment is historical hash-verified evidence, because CUDA was unavailable on the current host. Therefore the observed change combines device and dtype for float32/float16 comparisons; CPU bfloat16 versus historical CUDA bfloat16 isolates device more closely but not temporal/library drift. Quantization and alternate attention backends were unavailable. No authentication, PQC, security-product, entropy, ownership, anti-replay, or control claim is made.
