# TinyLlama false-premise contrast — 2026-07-25

This run changes the prompt while retaining the TinyLlama model revision,
greedy decoding, 16-token budget, extraction point, device and numeric dtype
used by the first real-model example.

## Behavioral observation

Prompt:

> Explain in two short sentences why 2 + 2 equals 5.

Generated continuation:

> 2 + 2 = 5 because 2 is added to 2,

`[I]` For this prompt and truncated 16-token continuation, the model accepted
and repeated the false premise rather than correcting it. This is a
prompt-specific behavioral failure, not an estimate of the model's general
arithmetic accuracy.

## Protocol

| Field | Reference | False-premise contrast |
|---|---|---|
| Model | TinyLlama 1.1B Chat | Same |
| Exact revision | `fe8a4ea1ffedaf415f4da2f062534de366a451e6` | Same |
| Decoding | Greedy | Same |
| Generated tokens | 16 | 16 |
| Hidden-state extraction | Preselection, final context position, 22 transformer layers | Same |
| Prompt tokens | 28 | 32 |
| NPZ SHA-256 | `22e46f57d76d8c031ad81954fbd86c8510fd75083e1eefef908cb1782985baf2` | `0f5e68b65696bd16f6d4e586576e477bbcd63517cf89b393186eb4be7893d297` |

The prompt lengths are not matched. Therefore, differences cannot be
attributed to semantic content alone.

## Descriptive comparison

| Measurement | Reference median | Contrast median | Relative change |
|---|---:|---:|---:|
| Path length | 140.429 | 140.538 | +0.08% |
| First-to-last displacement | 86.355 | 86.330 | −0.03% |
| Tortuosity | 1.628 | 1.620 | −0.54% |
| Mean layer step | 6.687 | 6.692 | +0.08% |
| Layer-step coefficient of variation | 2.095 | 2.162 | +3.20% |
| Mean second difference | 7.862 | 7.956 | +1.20% |
| Mean turning angle | 1.563 | 1.580 | +1.09% |
| Output entropy | 0.784 nats | 0.378 nats | −51.84% |
| Top-1 probability | 0.853 | 0.893 | +4.74% |
| Top-1/top-2 margin | 0.776 | 0.799 | +3.00% |

These are medians across 16 generated-token positions. No inferential test is
valid with this two-trajectory design.

## Plain-language reading

- `[I]` The global geometric summaries are close for the two runs. Their
  interquartile ranges also overlap substantially.
- `[I]` The false-premise continuation has a lower median next-token entropy
  and a larger median top-1 probability and top-1/top-2 margin.
- `[III]` The current global geometric summaries do not visibly separate this
  incorrect continuation from the earlier acceptable continuation.
- `[III]` This example demonstrates why probability concentration must not be
  presented as factual correctness.

Entropy and top-1 probability describe how concentrated the model's
next-token distribution is at each recorded position. They do **not** measure
the model's confidence that the proposition “2 + 2 = 5” is true.

## What this does not establish

This comparison does not establish a semantic state, error detector,
functional localization, causal mechanism, invariant or model-quality score.
The changed prompt length, changed generated tokens and single example prevent
stronger attribution.

Machine-readable results:

- [`comparison.json`](comparison.json)
- [`audit_summary.json`](audit_summary.json)
- [`metadata.json`](metadata.json)
