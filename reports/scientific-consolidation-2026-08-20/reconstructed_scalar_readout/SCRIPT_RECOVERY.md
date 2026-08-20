# extract_features.py recovery

Source session ID: `5d793c3e-5e08-4a66-9200-8acfbc5066c9`.

| Event | UTC timestamp | Operation |
|---:|---|---|
| 3993 | 2026-08-19 11:34:55.925 | Initial 17,229-byte `Write` |
| 4060 | 2026-08-19 11:39:01.852 | One `Edit`: replace inferred first-batch schemas with explicit PyArrow schemas |

The final file was recovered directly from Claude file-history object
`ebddff34bcd29b5a@v2`. No later `Write` or `Edit` was found. Classification:
**EXACT_RECOVERY**. Final SHA-256:
`c8fa3842e991ba99f1ba5987b26d4c549437dc86e5549ece08d849e08e41a80a`.
The materialized file is byte-identical to that archive object and passes
`python -m py_compile`.

Replay environment: Python 3.10.12, numpy 1.26.4, pandas 2.3.3, pyarrow 23.0.1,
torch 2.8.0, Linux aarch64. The script imports numpy, pandas, PyArrow, and torch;
the scalar path uses numpy/PyArrow and JSON/CSV standard-library modules. Pandas and
torch are not needed for scalar calculation. There are no imported project helpers.

Implicit assumptions include the two hard-coded report roots, the scratch directory,
NPZ array names, model-wide JSON shape, `RUN_INDEX.csv` field names, lexical model
sorting, run-index iteration order, and PyArrow ParquetWriter defaults.

