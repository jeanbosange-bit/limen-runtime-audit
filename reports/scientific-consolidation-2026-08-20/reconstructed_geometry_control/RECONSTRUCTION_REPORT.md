# Remaining-chain reconstruction report

## Verdict

Overall Phase 9 chain: **PARTIALLY_RECONSTRUCTIBLE**. The complete numerical input used
by Phase 9 is recoverable and remains supported. The geometry-control gap is closed.
The remaining gap is independent reproduction of the dense scalar-readout extraction
parts and their original physical row interleaving; using the immutable packaged scalar
rows as a packaging proxy cannot establish that upstream stage independently.

Overall provenance verdict: **PARTIALLY_REPRODUCIBLE**. This is not a numerical failure.
Two final vector tables are byte-identical and the readout table is keyed-numerically
and semantically identical, but its bytes and row order differ.

## Geometry-control replay

- 183,627 rows; 4 models; 144 runs; 128 dimensions.
- Keys and row order equal to the historical control subset.
- Immediate replay dtype float64; archived final dtype float32 after exact recast.
- After float32: mismatch count 0; maximum absolute difference 0; mean absolute
  difference 0.
- Null patterns and representation labels match.
- Classification: **BYTE_IDENTICAL** as part of the final
  `orthogonal_subset.parquet` (SHA-256
  `13414276f1a476d5e856f0f41cc63d0f00f49ed0eccad5ae1389450ddfd7b543`).

## Final tables

| Table | Semantic | Numerical | Byte | Classification |
|---|---|---|---|---|
| `h_random_projection.parquet` | Yes | Exact | Exact | BYTE_IDENTICAL |
| `orthogonal_subset.parquet` | Yes | Exact | Exact | BYTE_IDENTICAL |
| `readout_features.parquet` | Yes, keyed | Exact, keyed | No | NUMERICALLY_IDENTICAL |

The readout replay has 185,005 rows and 24 columns, identical schema, dtypes, key set,
null counts, categorical values, and keyed numeric values. Its row order differs from
history beginning inside the scalar/intermediate boundary; SHA-256 is replay
`71152ef5e7028b6925d435d6d0b18831c52432c109d08cd18baa5be80faf92f1`
versus historical
`e45ecf217851bb701a1f9b1929d5151334f40cdd08b707220971960e27ac773d`.
This is attributable to reconstructing the deleted scalar-part union from the packaged
table, not to value differences. It does not change Phase 9 grouping or numerical input.

## Scientific impact

| Area | Status | Basis |
|---|---|---|
| Geometry findings | SUPPORTED_DESCRIPTIVELY | Projection table is byte-identical; provenance does not promote the claim. |
| Orthogonal invariance | SUPPORTED_DESCRIPTIVELY | Rotated inputs reproduce within the byte-identical combined table. |
| Geometry-destroying control conclusion | SUPPORTED_DESCRIPTIVELY | Control rows reproduce exactly after archived recast. |
| Decodability | SUPPORTED_DESCRIPTIVELY | Its feature inputs are unchanged; unseen-prompt generalization remains untested. |
| Phase 9 | SUPPORTED_DESCRIPTIVELY | Intermediate/readout values used by the analysis are unchanged. |
| RPTA | PARTIALLY_SUPPORTED | Provenance does not establish a single sequence or mechanism. |
| Atlas | SUPPORTED_DESCRIPTIVELY | Scalar comparisons are unaffected; semantic landmark equivalence is not established. |
| Downstream publication claims | PARTIALLY_SUPPORTED | Numerical provenance is strong, but scalar extraction must be independently replayed before full-chain reproducibility is claimed. |

Replay parquets are retained under `replay/` for local verification and excluded from Git.
