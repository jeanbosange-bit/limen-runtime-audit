# RESOURCE_ESTIMATE

All numbers below are computed from real, existing empirical data — no simulation.
Sources: `star_command_bcc_v2/raw/generations.jsonl` (243 real Qwen2.5-0.5B-Instruct
generations against this exact panel) and `reports/atlas-kv-compression-v0/execution/raw_captures/`
(one real captured file's exact per-tensor byte breakdown).

## Lane A — BCC-WIDE (729 lightweight generations)

**Runtime.** BCC V2's real 243-prompt run on Qwen2.5-0.5B-Instruct: mean
`generation_time_s` = 3.443s, total 836.66s for 243 generations (10.60 tok/s mean,
36.5 mean / 22 median / 224 max generated tokens). Lane A repeats this 3x (three seeds):

- **Estimated generation time**: ~3 × 836.7s ≈ **2,510s (~42 minutes)**.
- **Added overhead**: per-step entropy/top1/top2-gap capture requires `output_scores=True`
  during `generate()` — modest additional compute (softmax over the vocab per step,
  already this program's standard practice) and no new forward passes. Budgeting +20-30%
  for this plus I/O: **~50-65 minutes total, round to ~1 hour.**
- **Storage**: lightweight only (no tensors). Per-generation estimate: token ids +
  per-step 3-float probability summary + text/metadata ≈ 8 KB/generation →
  729 × 8 KB ≈ **6 MB**, rounded generously to **under 50 MB** for JSON/JSONL overhead
  and metadata.

## Lane B — ATLAS-DEEP (144 deep runs)

**Storage — two estimates, both reported per mission section 15's ">2-3GB: report exact
estimate" trigger (both exceed it):**

1. **V0-reference-based** (flat extrapolation from `reports/atlas-kv-compression-v0`'s
   own empirical 393MB/30runs = 13.10 MB/run average, plus the new Q tensor): Q adds
   exactly the same element-count as attn_output per run (both are
   `24 × seq_len × 896` elements), a computed **+12.8% size uplift** →
   **14.78 MB/run × 144 = ~2.13 GB**. This likely *understates* Lane B, since V0's KVC
   prompts generated far fewer tokens on average than Panel V2's (V0 KVC max n_steps=48;
   Panel V2's real BCC V2 run shows generations up to 224 tokens, median 22).
2. **Stratified realistic estimate** (48 short/48 medium/48 long, `seq_len` ≈ 50/100/200
   respectively, exact per-tensor byte formula including the quadratic `seq_len²`
   attention-weight tensor and the new Q tensor, then the same 66.6% compression ratio
   measured on one real V0 `.npz` file): SHORT ≈9.3 MB/run, MEDIUM ≈22.9 MB/run,
   LONG ≈58.9 MB/run → **weighted total ≈ 4.38 GB**.

**Reported range: ~2.1–4.4 GB**, with **4.4 GB the more realistic figure** given Panel V2's
longer typical generations. Disk has 599 GB free (32% used) — comfortably sufficient
either way; this is reported for transparency per the mission's explicit threshold rule,
not because it is a capacity risk.

**Runtime.** V0's own per-run tensor-capture forward-pass overhead was small (~0.1-0.15s
observed in `RUN_MANIFEST.json`, on top of generation time) — hooks are cheap relative to
generation itself. Using the same 10.60 tok/s throughput and the 3 length buckets above
(15/40/100 generated tokens): weighted mean generation time ≈ 4.9s/run. Adding capture +
compressed-`.npz` write overhead (larger files than Lane A, but still small relative to
generation for a 0.5B model on local NVMe): **estimated ~20-40 minutes total** for 144
runs (assumes the model is loaded once and reused across all 144 runs, not reloaded
per-run — V0's own `RUN_MANIFEST.json` shows evidence of expensive per-run reload
overhead, ~20+ seconds/run, that a Lane B implementation should explicitly avoid).

## GPU / RAM

Model: 558.5M parameters, measured peak memory 1,034.9 MB (`star_command_bcc_v2`'s model
registry, device-tested on this same Jetson). Both lanes run comfortably within the
7.4 GiB total / ~5.3 GiB currently-available RAM observed at preflight time. No GPU
memory figures are available from `nvidia-smi` on this Jetson (unified-memory platform,
`memory.total`/`memory.used` report `[N/A]`) — RAM is the binding constraint, and 1 GB
peak for a single loaded model leaves ample headroom.

## Disclosed uncertainty

Lane B's real average `seq_len` is not known until Lane A actually runs — the stratified
estimate's bucket assumptions (50/100/200) are a reasonable planning prior from Panel
V2's `max_new_tokens` configuration (64-224 across families) and BCC V2's real observed
distribution (up to 224 generated tokens), not a guarantee. Per the mission's execution
order, Lane B's estimate should be **re-checked against Lane A's actual completion audit
before the Lane B GO is requested**, and Lane B stops again if the real projection
differs materially from this document.
