# Metric reference

LIMEN v0.1 treats the layer axis as an ordered sequence. For each token, the
hidden state is an array \(h_\ell\), where \(\ell\) is the layer index.

This is a **layer-wise geometric path**. It must not be confused with a
continuous-time physical trajectory or with the token-to-token generation
trajectory.

## Descriptive trajectory metrics

| Metric | Definition | Useful for | Does not establish |
|---|---|---|---|
| Path length | \(\sum_\ell \lVert h_{\ell+1}-h_\ell\rVert_2\) | Total movement through layers | Reasoning effort |
| Displacement | \(\lVert h_L-h_0\rVert_2\) | Net first-to-last change | Semantic progress |
| Tortuosity | Path length / displacement | Route indirectness | Confusion or intelligence |
| Mean speed | Mean adjacent-layer distance | Typical layer-wise movement | Runtime latency |
| Speed CV | Standard deviation / mean speed | Movement variability | Instability in a control-theory sense |
| Mean acceleration | Mean norm of the second difference | Changes in movement | Physical acceleration |
| Turning angle | Mean angle between consecutive differences | Directional change | A cognitive transition |

Tortuosity is undefined when displacement is zero and is reported as `NaN` in
the per-token arrays. Summary statistics omit non-finite values.

## Probability baselines

Entropy is reported in natural units (`nats`). The top-1/top-2 margin is the
difference between the two largest softmax probabilities.

These baselines are included because many apparent geometric effects can be
explained by ordinary changes in model confidence. A trajectory metric should
not be presented as incrementally useful until it is compared against them on
held-out data.

## Required controls for stronger claims

- exact model and tokenizer revisions;
- fixed extraction location and layer indexing;
- response length and token-position controls;
- prompt-family controls;
- repeated seeds when sampling is enabled;
- shuffled layer and token-order baselines;
- held-out evaluation;
- cross-checks on more than one architecture.

No v0.1 metric is a validated universal invariant, functional locator or
causal control signal.
