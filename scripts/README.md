# Real-model trajectory extraction

This directory contains the extractor used for the public TinyLlama real-run
examples.

## Locked reference configuration

| Field | Value |
|---|---|
| Model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Revision | `fe8a4ea1ffedaf415f4da2f062534de366a451e6` |
| Decoding | Greedy |
| Generated tokens | 16 |
| Hidden-state point | Final context position immediately before each next-token selection |
| Included layers | 22 transformer layers |
| Embedding output | Excluded |
| Logits | Included |

`extract_limen_trajectory.py` writes:

- `hidden_states`: `[generated_tokens, transformer_layers, hidden_dim]`;
- `logits`: `[generated_tokens, vocabulary]`;
- `token_ids`: `[generated_tokens]`;
- a separate JSON metadata file with revisions, shapes and source SHA-256.

## Install

The extractor requires Python 3.10 or later, PyTorch, NumPy and Transformers.
Install the repository itself plus the model dependencies in an isolated
environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install torch transformers
```

## Verify the extractor

```bash
python -m py_compile scripts/extract_limen_trajectory.py
python -m unittest -v scripts/test_extract_limen_trajectory.py
```

## Reproduce the first real run

```bash
python scripts/extract_limen_trajectory.py \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --revision fe8a4ea1ffedaf415f4da2f062534de366a451e6 \
  --prompt "Explain in two short sentences why the sky appears blue." \
  --max-new-tokens 16 \
  --device auto \
  --dtype auto \
  --output trajectory.npz \
  --metadata-output trajectory.metadata.json
```

Then run the public descriptive audit:

```bash
limen-audit trajectory.npz \
  --metadata trajectory.metadata.json \
  --output audit_output
```

The published reference payload has SHA-256:

```text
22e46f57d76d8c031ad81954fbd86c8510fd75083e1eefef908cb1782985baf2
```

## Reproducibility boundary

The exact-replication result currently applies to two greedy executions on the
documented Jetson environment. It does not establish bitwise reproducibility
across devices, PyTorch or Transformers versions, sampled decoding, model
revisions or architectures.

The exported arrays and their descriptive metrics do not establish functional
localization, semantic identity, causal mechanisms, reasoning or model quality.
