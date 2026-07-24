# LIMEN Runtime Audit

`limen-runtime-audit` is a small, reproducible toolkit for inspecting
layer-wise activation trajectories exported from open-weight language models.

It is an **audit and observability prototype**, not a validated controller.
Its outputs do not establish cognitive states, functional localization,
semantic identity, causality, reasoning, consciousness, or universal
attractors.

## Why this exists

Most model evaluations inspect the generated answer. LIMEN adds a second view:
how the model's internal activation vector changes from one layer to the next
for every generated token.

In plain language, the first release can help answer:

- Did two checkpoints produce different internal profiles?
- Is one prompt family associated with more irregular layer-wise movement?
- Is a trajectory an outlier relative to a reference run?
- Did an extraction pipeline silently change shape or produce invalid values?

It does **not** answer whether a response is true, safe, intelligent, or
causally controlled.

## What v0.1 does

- reads hidden states shaped `[tokens, layers, hidden_dim]`;
- computes path length, displacement, tortuosity, speed variability,
  acceleration and turning angle;
- computes entropy and top-1/top-2 probability margin when logits are supplied;
- records the source checksum and extraction metadata;
- writes machine-readable JSON and a compact Markdown report.

## Quick start

```bash
git clone https://github.com/jeanbosange-bit/limen-runtime-audit.git
cd limen-runtime-audit
python -m venv .venv
source .venv/bin/activate
pip install -e .
python examples/make_demo.py
limen-audit examples/data/demo_trajectory.npz \
  --metadata examples/extraction_manifest.example.json \
  --output examples/output
PYTHONPATH=src python -m unittest discover -s tests -v
```

The input is an `.npz` file containing:

- `hidden_states` — required, `[tokens, layers, hidden_dim]`;
- `logits` — optional, `[tokens, vocabulary]`.

## Output vocabulary

| Metric | Plain-language meaning |
|---|---|
| Path length | Total layer-to-layer movement |
| Displacement | Direct distance from first to last layer |
| Tortuosity | Indirectness of the layer-wise route |
| Mean speed | Average movement between adjacent layers |
| Speed CV | Variability of that movement |
| Mean acceleration | Change in layer-to-layer movement |
| Turning angle | Change in movement direction |
| Entropy | Uncertainty of the output distribution |
| Top-1/top-2 margin | Separation between the two leading token probabilities |

These are descriptive measurements. See
[`docs/METRICS.md`](docs/METRICS.md) before interpreting them.

## Reproducibility contract

Every scientific run should pin:

- model and tokenizer identifiers and exact revisions;
- library versions;
- chat template and tokenized input;
- seed and generation parameters;
- `model()` versus `generate()`;
- cache configuration and prefill/decode phase;
- the exact extraction point and layer indexing;
- normalization applied before analysis.

Missing information must be marked `to-confirm`, never reconstructed from
memory.

## Scientific status

The metrics are descriptive observations `[I]`. A stable association supported
by held-out tests and appropriate controls may become `[II]`. Functional names
remain interpretations `[III]` or hypotheses `[IV]` until independently tested.

The next validation milestone is to show that trajectory metrics detect or
predict held-out behavioral regressions beyond entropy, probability margin,
token position, prompt family, model family and shuffled layer/time baselines.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The public test suite checks geometry, numerical validation, probability
baselines, report creation and the audit schema. GitHub Actions runs it on
Python 3.10, 3.11 and 3.12.

## Research background

LIMEN grew from an independent empirical research programme on runtime
activation dynamics:

- [Four Dynamical Regimes in Large Language Models](https://doi.org/10.5281/zenodo.20348878)
- [Conditional Dynamic Signatures in Large Language Models](https://doi.org/10.5281/zenodo.20361289)
- [Dynamic-Layer Controllability without Universal Semantic Recovery](https://doi.org/10.5281/zenodo.20400171)
- [A Runtime Trajectory Dynamics Framework for Large Language Models](https://doi.org/10.5281/zenodo.20602685)

The papers preserve the terminology used during the original experiments.
This repository uses narrower engineering language where later controls showed
that stronger interpretations were not justified.

## Public boundary

This repository contains only the public LLM audit layer. Non-LLM experiments
and unaudited control or compression work are intentionally excluded.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
