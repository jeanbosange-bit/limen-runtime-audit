# Compact packaging contract

```text
SOURCE EXTRACTION
  -> _part_core_proj                         (extract_core.py)
  -> _part_core_ortho                        (extract_core.py)
  -> _part_core_readout_intermediate         (extract_core.py)
  -> _part_{core,extended}_readout            (extract_features.py scalar path)
PROJECTED H
  -> h_random_projection.parquet             (finalize_and_quality.py)
ORTHOGONAL + GEOMETRY CONTROL
  -> orthogonal_subset.parquet                (rotation first, control second)
READOUT
  -> scalar readout parts merged in sorted filename order
  -> INTERMEDIATE_PROJECTED appended
  -> extract_core FINAL_NATIVE appended only when its four-field key is absent
RECAST
  -> all proj_ columns in H/orthogonal tables cast to float32
PACKAGE
  -> SHA-256 manifest over the declared deliverables
```

`INTERMEDIATE_PROJECTED` comes from `_part_core_readout_intermediate.parquet`.
Dense `FINAL_NATIVE`, `FUTURE_T_PLUS_2`, and `FUTURE_T_PLUS_3` rows come from the
scalar `extract_features.py` path. The recovered finalizer does not deduplicate or
sort. It concatenates scalar parts in lexically sorted filename order, then appends
all non-FINAL rows from the core readout part. Core `FINAL_NATIVE` rows are appended
only when `(model_id, run_id, token_step, layer_index)` is absent from the scalar
union. Thus the scalar dense FINAL_NATIVE row has precedence.

Pandas concat supplies four null scalar-only columns on intermediate rows; nulls are
preserved. Column order follows the scalar table, then any core-only columns. Pandas
`to_parquet(index=False)` supplies writer defaults; no explicit compression, row-group,
schema, sorting, or metadata options were archived. The transfer script does not alter
tables; it hashes existing deliverables.

The original scalar part parquets were deleted. Their value union is recoverable from
the packaged table, but their original part boundary and physical row interleaving are
not independently recoverable from current artifacts. This is the only remaining
packaging limitation found.

