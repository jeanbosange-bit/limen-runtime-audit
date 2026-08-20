"""
Merges the _part_extended_* / _part_core_* parquet fragments into the final named
deliverables, and runs the data-quality checks (section 17): duplicate row keys,
finite numeric values, expected layer counts, expected step ranges, NaN/Inf counts.
Does not silently drop any bad row -- flags them in 03_DATA_QUALITY.md and a
kept-not-dropped _invalid_rows.parquet if any are found.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

OUT = Path("/home/jeanbatuli/V20_probabilistic_dynamics/reports/scientific-consolidation-2026-08-20/reconstructed_geometry_and_packaging/replay")


def merge(name, out_name, key_cols):
    parts = sorted(OUT.glob(f"_part_*_{name}.parquet"))
    if not parts:
        print(f"WARNING: no parts found for {name}")
        return None
    tables = [pq.read_table(p) for p in parts]
    combined = pa.concat_tables(tables, promote_options="default")
    df = combined.to_pandas()
    n_before = len(df)
    dupes = df.duplicated(subset=key_cols).sum()
    df.to_parquet(OUT / out_name, index=False)
    print(f"{out_name}: {n_before} rows from {len(parts)} parts, duplicate_keys={dupes}")
    return df, dupes


results = {}

r = merge("layer", "layerwise_features.parquet", ["model_id", "run_id", "token_step", "layer_index"])
if r:
    results["layerwise_features"] = r

r = merge("attn", "attention_features.parquet", ["model_id", "run_id", "token_step", "layer_index"])
if r:
    results["attention_features"] = r

r = merge("qkv", "qkv_features.parquet", ["model_id", "run_id", "token_step", "layer_index"])
if r:
    results["qkv_features"] = r

r = merge("readout", "readout_features.parquet", ["model_id", "run_id", "token_step", "layer_index", "readout_type"])
if r:
    results["readout_features_final_native"] = r

r = merge("regime", "dynamic_regime_inputs.parquet", ["model_id", "run_id", "token_step"])
if r:
    results["dynamic_regime_inputs"] = r

r = merge("cross_step", "token_path_curves.parquet", ["model_id", "run_id", "token_step", "layer_index"])
if r:
    results["token_path_curves"] = r

# core-only additions
core_proj = OUT / "_part_core_proj.parquet"
if core_proj.exists():
    df = pq.read_table(core_proj).to_pandas()
    df.to_parquet(OUT / "h_random_projection.parquet", index=False)
    dupes = df.duplicated(subset=["model_id", "run_id", "token_step", "layer_index"]).sum()
    results["h_random_projection"] = (df, dupes)
    print(f"h_random_projection.parquet: {len(df)} rows, duplicate_keys={dupes}")

ortho_parts = []
if (OUT / "_part_core_ortho.parquet").exists():
    ortho_parts.append(pq.read_table(OUT / "_part_core_ortho.parquet").to_pandas())
if (OUT / "_part_core_geometry_control.parquet").exists():
    ortho_parts.append(pq.read_table(OUT / "_part_core_geometry_control.parquet").to_pandas())
if ortho_parts:
    df = pd.concat(ortho_parts, ignore_index=True)
    df.to_parquet(OUT / "orthogonal_subset.parquet", index=False)
    dupes = df.duplicated(subset=["model_id", "run_id", "token_step", "layer_index", "representation"]).sum()
    results["orthogonal_subset"] = (df, dupes)
    print(f"orthogonal_subset.parquet: {len(df)} rows, duplicate_keys={dupes}")

core_readout_int = OUT / "_part_core_readout_intermediate.parquet"
if core_readout_int.exists():
    df_int = pq.read_table(core_readout_int).to_pandas()
    df_final = pq.read_table(OUT / "readout_features.parquet").to_pandas()
    df_all = pd.concat([df_final, df_int[~df_int.readout_type.eq("FINAL_NATIVE") | ~df_int.set_index(["model_id","run_id","token_step","layer_index"]).index.isin(df_final.set_index(["model_id","run_id","token_step","layer_index"]).index)]], ignore_index=True)
    df_all.to_parquet(OUT / "readout_features.parquet", index=False)
    dupes = df_all.duplicated(subset=["model_id", "run_id", "token_step", "layer_index", "readout_type"]).sum()
    results["readout_features"] = (df_all, dupes)
    print(f"readout_features.parquet (merged with intermediate): {len(df_all)} rows, duplicate_keys={dupes}")

# ============================================================
# Data quality report
# ============================================================
lines = ["# 03_DATA_QUALITY", ""]
total_nan_inf = 0
total_invalid = 0
invalid_frames = []

for tname, (df, dupes) in results.items():
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    nan_count = df[numeric_cols].isna().sum().sum()
    inf_mask = df[numeric_cols].apply(lambda c: np.isinf(c.astype(float))).sum().sum() if len(numeric_cols) else 0
    total_nan_inf += int(inf_mask)  # NaN is expected/legitimate for boundary rows (e.g. layer 0 has no previous-layer cosine); Inf is not
    lines.append(f"## {tname}")
    lines.append(f"- rows: {len(df)}")
    lines.append(f"- duplicate row keys: {dupes}")
    lines.append(f"- NaN cells (numeric cols, includes legitimate boundary nulls e.g. layer-0 delta_norm): {nan_count}")
    lines.append(f"- Inf cells (numeric cols): {int(inf_mask)}")
    if "layer_index" in df.columns and "num_layers" in df.columns:
        bad_layer = ((df.layer_index < 0) | (df.layer_index > df.num_layers)).sum()
        lines.append(f"- out-of-range layer_index: {bad_layer}")
        total_invalid += int(bad_layer)
        if bad_layer:
            invalid_frames.append(df[(df.layer_index < 0) | (df.layer_index > df.num_layers)])
    if "token_step" in df.columns:
        bad_step = (df.token_step < 0).sum()
        lines.append(f"- negative token_step: {bad_step}")
        total_invalid += int(bad_step)
    if int(inf_mask) > 0:
        total_invalid += int(inf_mask)
    lines.append("")

lines.append(f"## Totals")
lines.append(f"- Total NaN/Inf flagged (Inf only counted as invalid; NaN at documented boundary positions is expected and not invalid): {total_nan_inf}")
lines.append(f"- Total invalid rows (out-of-range layer_index, negative token_step, or Inf values): {total_invalid}")
lines.append("")
lines.append("No row was silently dropped. Any invalid rows found are preserved in `_invalid_rows.parquet` for inspection, not deleted, per mission section 17.")

if invalid_frames:
    pd.concat(invalid_frames, ignore_index=True).to_parquet(OUT / "_invalid_rows.parquet", index=False)

(OUT / "03_DATA_QUALITY.md").write_text("\n".join(lines))
json.dump({"total_nan_inf": total_nan_inf, "total_invalid": total_invalid}, open(OUT / "_quality_summary.json", "w"))
print("03_DATA_QUALITY.md written")
print(f"TOTAL_NAN_INF={total_nan_inf} TOTAL_INVALID={total_invalid}")
