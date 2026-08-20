# Reconstruction environment

- Session performing this replay: Claude Code, 2026-08-20 (this consolidation session).
- Source session for recovered scripts: Claude session
  `5d793c3e-5e08-4a66-9200-8acfbc5066c9` (2026-08-19), the same session
  `EXTRACT_CORE_CONTRACT.md` recovered `extract_core.py`/`extract_features.py`/
  `find_final_norm.py` from.
- Inputs consumed: `../reconstructed_extract_core/replay/isolated_*/_part_core_proj.parquet`,
  `_part_core_ortho.parquet`, `_part_core_readout_intermediate.parquet` (Codex's own
  already-replayed, per-model-isolated `extract_core.py` outputs) and canonical
  `~/limen-runtime-audit/reports/deep-feature-extraction-2026-08-19/h_random_projection.parquet`,
  `orthogonal_subset.parquet`, `readout_features.parquet` (comparison targets only, read-only).
- No model was loaded. No generation occurred. No historical artifact was modified —
  every write in this gate targeted `reconstructed_geometry_and_packaging/replay/`
  (a new, dedicated directory) or this directory itself.
- Python 3.10, pandas, pyarrow, numpy (system-installed versions used by the rest of this
  session; not independently pinned/frozen in this gate -- the same disclosed limitation
  `ANALYSIS_PROVENANCE_AUDIT.md` priority #4 already names for the broader pipeline).
- Bulky parquet replay outputs (`replay/*.parquet`) are intentionally **not** committed to
  git and remain only in this working-directory copy, per the standing "no
  parquet/tensors/raw binaries in version control" rule. Only this report, the two
  recovered scripts, the recast script, and this file are committed.
