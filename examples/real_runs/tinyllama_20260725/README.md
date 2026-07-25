# TinyLlama real-run example — 2026-07-25

This is the first public LIMEN example computed from a real open-weight model
trajectory rather than synthetic demonstration arrays.

## Provenance

| Field | Value |
|---|---|
| Model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Exact revision | `fe8a4ea1ffedaf415f4da2f062534de366a451e6` |
| Device | NVIDIA Jetson Orin |
| Generation | Greedy, 16 generated tokens |
| Hidden states | `[16 tokens, 22 layers, 2048 dimensions]` |
| Logits | `[16 tokens, 32000 vocabulary entries]` |
| Source SHA-256 | `22e46f57d76d8c031ad81954fbd86c8510fd75083e1eefef908cb1782985baf2` |
| Extraction errors | 0 |

The recorded vector for each generated token is the state at the final context
position for every transformer layer, immediately before greedy next-token
selection. The embedding output is excluded.

## Results

The table summarizes the distribution across the 16 generated tokens.

| Measurement | Median | Q1–Q3 | Unit or scale |
|---|---:|---:|---|
| Path length | 140.429 | 138.404–142.088 | L2 distance in raw activation space |
| First-to-last displacement | 86.355 | 86.033–86.706 | L2 distance in raw activation space |
| Tortuosity | 1.628 | 1.591–1.642 | Ratio |
| Mean layer step | 6.687 | 6.591–6.766 | L2 distance per layer transition |
| Layer-step coefficient of variation | 2.095 | 2.055–2.122 | Ratio |
| Mean second difference | 7.862 | 7.688–8.051 | Raw activation-space scale |
| Mean turning angle | 1.563 | 1.550–1.580 | Radians |
| Output entropy | 0.784 | 0.271–1.869 | Nats |
| Top-1 probability | 0.853 | 0.517–0.960 | Probability |
| Top-1/top-2 margin | 0.776 | 0.346–0.940 | Probability difference |

## Plain-language reading

- **The route is not straight.** A median tortuosity of `1.628` means that the
  accumulated layer-to-layer path is about 1.63 times the direct first-to-last
  distance for this run.
- **Layer-wise movement is uneven.** The median step coefficient of variation
  is `2.095`, so a small number of layer transitions may contribute much more
  movement than others. The per-layer profile must be inspected before naming
  any specific transition.
- **Output confidence varies substantially between tokens.** The entropy
  interquartile range is `0.271–1.869 nats`; the top-1 probability ranges from a
  first quartile of `0.517` to a third quartile of `0.960`.
- **This is a fingerprint, not a quality score.** With one prompt and one
  trajectory, no value can be labelled “good”, “bad”, “stable”, or “abnormal”.

## What can be claimed

- `[I]` The extraction completed with the documented shapes, revision and
  checksum.
- `[I]` The table reports directly computed descriptive measurements for this
  exact trajectory.
- `[III]` The measurements form a useful baseline fingerprint for matched
  future runs.

## What cannot be claimed

This run does not demonstrate functional localization, a semantic state,
causality, reasoning quality, correctness, controllability, or a universal
invariant. Raw activation-space distances cannot be compared across models,
normalizations or extraction points as though they shared an absolute unit.

## Next matched comparison

The next useful trajectory should change one factor at a time while keeping the
model revision, tokenizer, extraction point and token budget fixed. Good first
comparisons are:

1. the same prompt repeated to verify deterministic replication;
2. a deliberately false or unanswerable prompt;
3. a matched prompt answered incorrectly;
4. the same prompt on a second open architecture.

Machine-readable values are in [`summary.json`](summary.json).

## Replication status

The locked extraction was run a second time and produced bitwise-identical
hidden states, logits and token IDs, including an identical NPZ checksum.
See [`REPLICATION.md`](REPLICATION.md) and
[`replication.json`](replication.json).
