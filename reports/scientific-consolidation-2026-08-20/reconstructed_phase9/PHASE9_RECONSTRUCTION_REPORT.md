# Phase 9 reconstruction report

## Verdict

`FULLY_REPRODUCIBLE`

The recovered final script compiled and reproduced all five files it generates as exact
byte matches to the canonical Phase 9 files. The threshold-free scientific verdict
`MODEL_SPECIFIC_ORDERING_ONLY` remains unchanged.

## Script recovery

Source session:
`~/.claude/projects/-home-jeanbatuli-V20-probabilistic-dynamics/f29e2fd7-545d-43ae-a5c9-c406f52382e9.jsonl`.

Recovery events:

- JSONL event/line 287: `Write` of `phase9_logit_lens.py` into the session scratchpad.
- JSONL event/line 300: one `Edit`, replacing an invalid/unsupported PyArrow table-filter
  expression with `to_pandas()` followed by pandas `model_id.isin(TO_COMPUTE)` filtering.
- JSONL events/lines 296 and 304: execution commands for the script.

Known edits: exactly the one recovered edit above within the source session.  
Uncertain edits: none observed after that edit in the complete matching session history.
The script was absent from both repositories, so an out-of-session edit cannot be
independently excluded; the byte-identical replay strongly validates the recovered final
state against all five canonical outputs.

Recovered script SHA-256:
`5d63c66b108f34fe12d5e8ded0853a6219e5982f0ee8a234333b7cd37c24a40e`.

The recovered file was materialized byte-for-byte. Because it hardcodes the historical
output directory, `replay_phase9.py` replaced only that exact `OUT` assignment in memory
with the provenance replay directory. It copied the existing `pilot_metrics.csv` into
the replay directory because the recovered script reads that file from `OUT` as an input.
No other source text, parameter, seed, split, threshold, scoring rule, or input path was
changed.

## Inputs

- `/home/jeanbatuli/limen-runtime-audit/reports/deep-feature-extraction-2026-08-19/readout_features.parquet`
- `/home/jeanbatuli/limen-runtime-audit/reports/deep-feature-extraction-2026-08-19/h_random_projection.parquet`
- Canonical `reports/decodability-v0-2026-08-19/pilot_metrics.csv`, copied byte-for-byte
  into `replay/` as a required logical input.

All inputs were read-only. No missing cell was repaired or interpolated. The recovered
script reused the three pilot-model ORIGINAL rows and refit the other five models exactly
as originally encoded.

## Replay command

```bash
python reports/scientific-consolidation-2026-08-20/reconstructed_phase9/replay_phase9.py \
  --script reports/scientific-consolidation-2026-08-20/reconstructed_phase9/phase9_logit_lens.py \
  --canonical-dir reports/decodability-v0-2026-08-19 \
  --replay-dir reports/scientific-consolidation-2026-08-20/reconstructed_phase9/replay
```

Comparison command:

```bash
python reports/scientific-consolidation-2026-08-20/reconstructed_phase9/compare_phase9_replay.py \
  --canonical-dir reports/decodability-v0-2026-08-19 \
  --replay-dir reports/scientific-consolidation-2026-08-20/reconstructed_phase9/replay \
  --output reports/scientific-consolidation-2026-08-20/reconstructed_phase9/REPLAY_VERIFICATION.csv
```

## Output-by-output comparison

| output | rows × columns | primary key | null pattern | max numeric difference | SHA-256 | status |
|---|---:|---|---|---:|---|---|
| `logit_lens_model_coverage.csv` | 8 × 14 | `model_id` | exact | 0 | `550456a2…655ed1` | `BYTE_IDENTICAL` |
| `phase9_decodability_all8.csv` | 40 × 9 | `model_id, representation, depth_bin` | exact | 0 | `8eab11db…dba5df` | `BYTE_IDENTICAL` |
| `phase9_readout_agg.csv` | 34 × 12 | `model_id, depth_bin` | exact | 0 | `077fd495…40362b` | `BYTE_IDENTICAL` |
| `decodability_vs_readout.csv` | 40 × 20 | `model_id, representation, depth_bin` | exact, including 6 missing readout cells | 0 | `643c4e6e…0a7690` | `BYTE_IDENTICAL` |
| `aligned_depth_summary.csv` | 40 × 20 | `model_id, representation, depth_bin` | exact, including 6 missing readout cells | 0 | `643c4e6e…0a7690` | `BYTE_IDENTICAL` |

For every output, filename, byte content, row count, column names/order, ordered unique
primary keys, model/depth cells, null pattern, pandas semantic equality, numeric values,
and SHA-256 match. Full hashes are in `REPLAY_VERIFICATION.csv`.

## Mismatches

None. There are zero differing numeric cells and maximum absolute numeric difference is
zero for all five outputs. Serialization is also identical, so no lower match category is
needed.

The script does not generate `model_ordering_summary.csv`, `correlation_summary.csv`, or
the narrative Phase 9 reports. Those downstream historical artifacts were not claimed as
outputs of this recovered pipeline and were not overwritten or replay-classified here.

## Scientific impact

Reconstruction validates the numerical provenance of the core Phase 9 coverage,
decodability, readout aggregation, and aligned tables. It does not validate the later
half-maximum categorical post-processing as a preferred scientific analysis. The existing
threshold-free audit still finds heterogeneous model-specific curve relations and retains
`MODEL_SPECIFIC_ORDERING_ONLY`.

This replay adds reproducibility evidence only. It does not change any result, repair
missingness, test unseen prompts, establish a common ordering, or support a causal,
predictive, cognitive, or control claim.

