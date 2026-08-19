# SHORT_EXECUTION_POLICY

## The problem this fixes

`reports/atlas-kv-compression-v0/`'s region-eligibility rule (`n_steps>=9`, 8-decode-step
window + 1 downstream token) structurally excluded MATH_REASONING entirely (MR01/MR02,
`n_steps=3`) and most of FACTUAL_GROUNDED (FG02 fully excluded, FG01 down to a single
seed) from every downstream analysis built on that windowed region — including the
Attention-Sensitivity V1 gate and, indirectly, the Multi-Tensor Landmark Network. This
campaign exists partly to stop repeating that exclusion.

## Rule

**A run producing as few as 1-2 generated tokens is a complete, valid Atlas observation.**
Lane B captures the run's full actual tensors (whatever `seq_len` results) with no
downstream-window minimum imposed at capture time. Nothing is padded, truncated, or
discarded for being short.

## Labeling

- **SHORT_EXECUTION**: 1-9 generated tokens (below the old V0 windowed-region minimum) —
  a real, complete, labeled observation. Fully included in every descriptive coverage
  statistic Lane A/B produce.
- Never labeled **NON_ANALYZABLE** by default. That label is reserved for a specific
  future downstream experiment that has its own explicit, pre-registered minimum-length
  requirement (e.g. a future re-run of the exact 8-step windowed-region rule, if ever
  reused) — and even then, the exclusion applies only to *that* experiment's derived
  table, never to the raw capture itself.

## Consequence for family coverage

MATH_REASONING and FACTUAL_GROUNDED prompts, which produced the shortest generations in
`reports/atlas-kv-compression-v0/` (3-9 steps), are captured on equal footing with every
other family in this campaign. Whether this closes the coverage gap identified in
`reports/multi-tensor-atlas-v0/UNKNOWN_REGION_AUDIT.json` and
`reports/reference-landmark-network-v0/`'s zero-FACTUAL_GROUNDED/zero-MATH_REASONING
landmark finding is an empirical question for Lane A/B's own results — not assumed here.
