# Scalar readout contract

## Inputs

The historical script reads `RUN_INDEX.csv`, selects panel models from
`_core_extended.json`, and processes successful rows having a non-empty
`model_deep_path`. For scalar readout it uses only:

- `token_ids` from the archived model-deep NPZ;
- `generated_logits_fp16`, cast to float32 before scoring;
- the shape of `hidden_states_all_layers_fp16` to obtain `num_layers`, sequence
  length, and therefore `prompt_len = sequence_length - number_of_logit_steps`;
- `probability_metrics` in the archived model-wide JSON, joined by integer `step`,
  solely for commitment, branching, concentration, and collapse fields;
- model/run/prompt/family/seed/path fields from `RUN_INDEX.csv`.

It does not load a tokenizer, model, language-model head, or final-normalization
module. It performs no forward pass. `generated_logits_fp16` is the captured
**GENERATION_NATIVE_OUTPUT**. It is distinct from the
**FINAL_LAYER_RECOMPUTED_READOUT** emitted by `extract_core.py` after applying a
model head/final norm to a captured hidden state.

Runtime flag is positional: `extended` by default; `core` selects the core panel.
Models are sorted lexically. Runs retain `RUN_INDEX.csv` order. A missing/deleted NPZ
returns an explicit error; missing required `token_ids` or hidden states also errors.
If logits are absent, the historical control flow emits no scalar rows for that run.

## Row identity and boundaries

For each generation step, `position = prompt_len + step`. Rows with a position beyond
the stored sequence are skipped. `layer_index = num_layers` for every scalar row.
There is no `target_kind` column; `readout_type` is the categorical distinction.

`FINAL_NATIVE` targets `token_ids[position + 1]`. If that token does not exist, no
FINAL_NATIVE row is emitted. For offset `k` in 2 or 3, the row key retains the current
`token_step`, while logits and target token come from `future_step = step + k` and
`token_ids[prompt_len + future_step + 1]`. A future row is omitted unless both the
future logit row and target token exist.

## Output columns

Common identity columns are string `model_id`, `run_id`, `prompt_id`,
`prompt_family`, `seed`; int64 `token_step`, `layer_index`, `num_layers`; and string
`readout_type`.

For FINAL_NATIVE:

- logits are converted float32 → float64;
- probability is `exp(logit - max(logit)) / sum(exp(logit - max(logit)))`;
- entropy is `-sum(p * log(p + 1e-9))`;
- ordering is `np.argsort(-logits)`;
- top-1/top-2 token IDs are the first two indices in that ordering;
- target rank is the zero-based position of the target ID in that full ordering;
- `is_target_top1` is exact token-ID equality;
- target/top logits and probabilities are float64;
- commitment, branching, concentration, and collapse are relayed from model-wide
  metrics or null when the matching metric/key is unavailable.

For FUTURE_T_PLUS_2 and FUTURE_T_PLUS_3, only target rank, target logit, and target
probability are populated. Top-1/top-2 fields, entropy, `is_target_top1`, and the four
model-wide diagnostic fields are null. The explicit PyArrow schema is 24 columns:
strings, int64 identifiers/rank, boolean top-1 indicator, and float64 numeric scores.
After pandas loading, nullable integer/boolean fields may appear as float64/object;
this serialization behavior matches history.

