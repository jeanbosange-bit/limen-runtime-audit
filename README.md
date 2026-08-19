---
title: LIMEN Runtime Audit
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
short_description: Descriptive LLM trajectory auditing + live evidence capture
---

# LIMEN Runtime Audit

`limen-runtime-audit` is a small, reproducible toolkit for inspecting
layer-wise activation trajectories exported from open-weight language
models, and for capturing a live, synchronized, hash-chained evidence
record from a real generation.

It is an **audit and observability prototype**, not a validated controller.
Its outputs do not establish cognitive states, functional localization,
semantic identity, causality, reasoning, consciousness, or universal
attractors.

## Frozen research reports

The [`reports/`](reports/) archive contains the frozen protocols, derived results, provenance, manifests, and SHA-256 registries for the August 2026 Atlas expansion, landmark/route validation, and runtime-fingerprint feasibility gates. Raw multi-gigabyte tensor captures are intentionally excluded from Git; see [`docs/data/RAW_CAPTURE_STORAGE_POLICY.md`](docs/data/RAW_CAPTURE_STORAGE_POLICY.md). A conservative milestone summary is available in [`docs/research/2026-08-18_ATLAS_DATA_AND_SECURITY_MILESTONE.md`](docs/research/2026-08-18_ATLAS_DATA_AND_SECURITY_MILESTONE.md).

## Why this exists

Most model evaluations inspect the generated answer. LIMEN adds a second
view: how the model's internal activation vector changes from one layer to
the next for every generated token, and how several observation signals
(what was generated, how the output distribution looked, how far internal
representations moved) can be linked under one explicit, tamper-evident
identity instead of being silently glued together by file or array order.

In plain language, this Space can help answer:

- Did two checkpoints produce different internal profiles?
- Is one prompt family associated with more irregular layer-wise movement?
- Is a trajectory an outlier relative to a reference run?
- Did an extraction pipeline silently change shape or produce invalid values?
- If several signals are captured from the same generation, can I trust
  that they are actually about the same event, and can I detect if that
  record was tampered with afterward?

It does **not** answer whether a response is true, safe, intelligent, or
causally controlled, and it does not recommend or apply any intervention.

## What this Space does

**Descriptive audit (upload).** Unchanged from v0.1: upload a
`trajectory.npz` file and get path length, displacement, tortuosity, speed
variability, acceleration, turning angle, and (if logits are supplied)
entropy and top-1/top-2 margin.

**Live evidence capture (new in v0.2).** Runs a small public
instruction-tuned model (`Qwen/Qwen2.5-0.5B-Instruct`) on this Space's
GPU for one greedy generation, and records three synchronized channels per
token — generation, probability, and a coarse layer-sampled latent signal —
under one explicit identity (`run_id -> event_id -> step_index`), sealed
with a SHA-256 hash chain. Tampering with or reordering a recorded step is
detectable, not silently accepted (see `tests/test_evidence.py`).

Every channel reports one of four honest states: `AVAILABLE`,
`NOT_CAPTURED` (this run's profile never intended to record it),
`NOT_SUPPORTED` (the backend cannot produce it here), or `FAILED` (the
profile required it and it is missing — an anomaly, not a design choice). A
missing signal is never treated as zero.

This tab is a public, adapted demonstration of an evidence-architecture
*pattern* used in a larger private research programme. It intentionally
does not include that programme's trained scoring models, calibrated
decision thresholds, feature-engineering details, or raw experiment
archives — only the observation-and-honest-absence pattern itself.

## Quick start (library)

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

The uploaded/audited input is an `.npz` file containing:

- `hidden_states` — required, `[tokens, layers, hidden_dim]`;
- `logits` — optional, `[tokens, vocabulary]`.

The live evidence-capture tab needs no upload; it runs the model directly
on the Space's GPU (`pip install -e ".[space]"` for local use outside HF
Spaces).

The exact real-model extraction script, offline tests and locked TinyLlama
command are available in [`scripts/`](scripts/).

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

## Evidence tiers

| Tier | Meaning |
|---|---|
| **[I]** | Directly observed — recorded from a real run, not inferred. |
| **[II]** | Statistically supported — held up under held-out testing and stated controls. |
| **[III]** | Interpretation — a reasonable reading of [I]/[II] evidence, not itself tested. |
| **[IV]** | Hypothesis — proposed, not yet tested. |

Everything this Space produces directly (both tabs) is tier **[I]**. No
tier **[II]** claim (a validated predictive or causal relationship) is made
anywhere in this repository.

## Real-model example

The repository includes a first real-model audit from
`TinyLlama/TinyLlama-1.1B-Chat-v1.0`, with the exact revision, extraction
semantics, source checksum, numerical summary and a plain-language
interpretation:

- [`examples/real_runs/tinyllama_20260725/`](examples/real_runs/tinyllama_20260725/)
- [`examples/real_runs/tinyllama_false_premise_20260725/`](examples/real_runs/tinyllama_false_premise_20260725/)

It is a single-run activation fingerprint, not a model-quality score or a
reference distribution.

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

The metrics and evidence-chain records are descriptive observations `[I]`.
A stable association supported by held-out tests and appropriate controls
may become `[II]`. Functional names remain interpretations `[III]` or
hypotheses `[IV]` until independently tested.

The next validation milestone is to show that trajectory metrics detect or
predict held-out behavioral regressions beyond entropy, probability margin,
token position, prompt family, model family and shuffled layer/time
baselines.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The public test suite checks geometry, numerical validation, probability
baselines, report creation, the audit schema, and the evidence-chain
integrity guarantees (hash-chain tamper/reorder detection, honest-absence
bookkeeping). GitHub Actions runs it on Python 3.10, 3.11 and 3.12.

## Research background

LIMEN grew from an independent empirical research programme on runtime
activation dynamics:

- [Four Dynamical Regimes in Large Language Models](https://doi.org/10.5281/zenodo.20348878)
- [Conditional Dynamic Signatures in Large Language Models](https://doi.org/10.5281/zenodo.20361289)
- [Dynamic-Layer Controllability without Universal Semantic Recovery](https://doi.org/10.5281/zenodo.20400171)
- [A Runtime Trajectory Dynamics Framework for Large Language Models](https://doi.org/10.5281/zenodo.20602685)

A further series of companion papers on runtime actuator research (compute
budget, self-verification, retry, and evidence-gated recommendation
policies) is in preparation; DOIs will be added here once deposited.

The papers preserve the terminology used during the original experiments.
This repository uses narrower engineering language where later controls
showed that stronger interpretations were not justified.

## Public boundary

This repository contains only the public LLM audit and evidence-pattern
layer. Trained scoring models, calibrated decision thresholds,
feature-engineering details, raw experiment archives, and unaudited
control or compression work are intentionally excluded.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
