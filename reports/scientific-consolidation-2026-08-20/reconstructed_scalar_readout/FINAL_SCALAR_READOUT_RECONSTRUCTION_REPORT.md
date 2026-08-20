# Final scalar readout reconstruction report

## Verdict

The final `extract_features.py` is an exact recovery. Independent replay from archived
NPZ logits, token IDs, model-wide JSON, and the run index produced 153,279 scalar rows
with exact historical row order and zero keyed value mismatches. Replaying the exact
finalizer with these parts reproduced the complete 185,005-row
`readout_features.parquet` byte-for-byte. Overall chain verdict:
**FULLY_RECONSTRUCTIBLE**.

## Scalar replay

| Readout type | Rows |
|---|---:|
| GENERATION_NATIVE_OUTPUT (`FINAL_NATIVE` in scalar script) | 52,167 |
| `FUTURE_T_PLUS_2` | 50,874 |
| `FUTURE_T_PLUS_3` | 50,238 |
| Total | 153,279 |

Coverage is 18 models and 648 runs. Qwen2.5-0.5B-Instruct has no scalar rows because
its archived deep files lack generation-native logits; OPT-350M has no deep run input.
Neither was repaired. The replay matches the historical scalar prefix on 24 columns,
column order, dtypes, row keys/order, null mask, categories, token IDs, ranks, logits,
probabilities, entropy, diagnostic relays, and future boundaries. Numeric mismatch
count is 0; maximum and mean absolute differences are 0. Nonnumeric mismatch count is 0.
Classification: **NUMERICALLY_IDENTICAL** (the deleted original scalar part files have
no surviving bytes against which to classify the part serialization itself).

## Source precedence

Recovered code directly establishes:

- `INTERMEDIATE_PROJECTED`: `extract_core.py`.
- `FINAL_LAYER_RECOMPUTED_READOUT`: the strided `FINAL_NATIVE` rows emitted by
  `extract_core.py`; this label describes provenance and is not the stored historical
  `readout_type` string.
- `GENERATION_NATIVE_OUTPUT`: dense `FINAL_NATIVE` from `extract_features.py`.
- `FUTURE_T_PLUS_2` and `FUTURE_T_PLUS_3`: `extract_features.py`.

The finalizer first merges `_part_core_readout.parquet` and
`_part_extended_readout.parquet` in lexical filename order, producing 153,279 rows.
It then appends every intermediate row and only those recomputed FINAL_NATIVE rows whose
four-field key is absent from the scalar union. This adds 31,329 intermediate rows and
397 unmatched recomputed final-layer rows. Dense generation-native rows therefore have
precedence whenever both sources share a key.

Phase 9 filters `readout_type == "INTERMEDIATE_PROJECTED"` before analysis. It consumes
the intermediate projected rows, not generation-native FINAL_NATIVE, future rows, or
the recomputed final-layer rows. Scalar reconstruction therefore closes provenance for
the package but does not change Phase 9 numerical inputs or interpretation.

## Packaging order

Historical ordering is fully recovered: sorted core models and RUN_INDEX order within
`_part_core_readout`, followed by sorted extended models and run-index order within
`_part_extended_readout`; the finalizer lexically sorts part filenames, concatenates
core before extended, writes the 153,279-row scalar table, and appends the filtered
`extract_core` rows in their archived core-model order. ParquetWriter writes one table
per run. No filesystem completion order affects the result.

The complete replay SHA-256 is
`e45ecf217851bb701a1f9b1929d5151334f40cdd08b707220971960e27ac773d`, identical to
history. Classification: **BYTE_IDENTICAL**.

## Scientific impact

- Phase 9 readout interpretation: **SUPPORTED_DESCRIPTIVELY**, unchanged; it concerns
  intermediate projected readout only.
- Final-native versus intermediate-projected distinction: **SUPPORTED_DESCRIPTIVELY**
  as an explicit provenance distinction. It does not establish a sequential mechanism.
- RPTA: **PARTIALLY_SUPPORTED**, unchanged.
- Publication wording: provenance may state that the complete Phase 9 input chain is
  replay-validated. Ordering, cognition, causality, prediction, control, unseen-prompt
  generalization, and universal-transition claims remain **NOT_ESTABLISHED** or
  **NOT_TESTED** as previously reported.

Replay parquets remain local under `replay/` and are excluded from Git.
