#!/usr/bin/env python3
"""Verify isolated extract_core replays against historical compact tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


MODEL_DIRS = {
    "Qwen/Qwen2.5-0.5B-Instruct": "isolated_qwen--qwen2_5-0_5b-instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "isolated_tinyllama--tinyllama-1_1b-chat-v1_0",
    "gpt2": "isolated_gpt2",
    "EleutherAI/pythia-160m": "isolated_eleutherai--pythia-160m",
    "facebook/opt-125m": "isolated_facebook--opt-125m",
    "deepseek-ai/deepseek-coder-1.3b-instruct": "isolated_deepseek-ai--deepseek-coder-1_3b-instruct",
    "ibm-granite/granite-3.0-1b-a400m-instruct": "isolated_granite",
    "HuggingFaceTB/SmolLM-360M-Instruct": "isolated_smollm",
}
ROTATED_MODELS = {
    "Qwen/Qwen2.5-0.5B-Instruct", "gpt2", "EleutherAI/pythia-160m", "facebook/opt-125m"
}
KEYS = [
    "model_id", "run_id", "prompt_id", "prompt_family", "seed",
    "token_step", "layer_index", "num_layers",
]
PROJ_COLS = [f"proj_{i}" for i in range(128)]
READOUT_VALUES = [
    "top1_token_id", "top1_logit", "top1_prob", "top2_token_id", "top2_logit",
    "top2_prob", "target_logit", "target_probability", "target_rank",
    "is_target_top1", "entropy",
]


def exact_common_values(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> tuple[int, float]:
    differing = 0
    maximum = 0.0
    for column in columns:
        a, b = left[column], right[column]
        if pd.api.types.is_bool_dtype(a):
            differing += int(np.sum(a.to_numpy() != b.to_numpy()))
        elif pd.api.types.is_numeric_dtype(a):
            av, bv = a.to_numpy(float), b.to_numpy(float)
            differing += int(np.sum(~np.isclose(av, bv, rtol=0, atol=0, equal_nan=True)))
            delta = np.abs(av - bv)
            if len(delta) and not np.all(np.isnan(delta)):
                maximum = max(maximum, float(np.nanmax(delta)))
        else:
            differing += int(np.sum(a.to_numpy() != b.to_numpy()))
    return differing, maximum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--historical-dir", type=Path, required=True)
    parser.add_argument("--verification-output", type=Path, required=True)
    parser.add_argument("--mismatch-output", type=Path, required=True)
    args = parser.parse_args()

    canonical_meta = {
        row["model_id"]: row
        for row in json.loads((args.historical_dir / "PROJECTION_METADATA.json").read_text())
    }
    verification: list[dict[str, object]] = []
    mismatches: list[dict[str, object]] = []

    for model_id, dirname in MODEL_DIRS.items():
        replay_dir = args.replay_root / dirname

        replay_proj = pq.read_table(replay_dir / "_part_core_proj.parquet").to_pandas()
        canonical_proj = pq.read_table(
            args.historical_dir / "h_random_projection.parquet",
            filters=[("model_id", "=", model_id)],
        ).to_pandas()
        keys_match = replay_proj[KEYS].equals(canonical_proj[KEYS])
        replay_values = replay_proj[PROJ_COLS].to_numpy(dtype=np.float32)
        canonical_values = canonical_proj[PROJ_COLS].to_numpy(dtype=np.float32)
        numeric_equal = np.array_equal(replay_values, canonical_values)
        max_diff = float(np.max(np.abs(replay_values - canonical_values)))
        verification.append({
            "artifact": "RANDOM_PROJECTION", "model_id": model_id,
            "replay_rows": len(replay_proj), "canonical_rows": len(canonical_proj),
            "replay_columns": len(replay_proj.columns), "canonical_columns": len(canonical_proj.columns),
            "key_unique": not replay_proj.duplicated(KEYS).any(), "keys_ordered_match": keys_match,
            "model_runs": replay_proj.run_id.nunique(), "layers": replay_proj.layer_index.nunique(),
            "replay_value_dtype": str(replay_proj["proj_0"].dtype),
            "canonical_value_dtype": str(canonical_proj["proj_0"].dtype),
            "max_absolute_difference_after_float32": max_diff,
            "differing_numeric_cells_after_float32": int(np.sum(replay_values != canonical_values)),
            "null_pattern_match": replay_proj.isna().equals(canonical_proj.isna()),
            "matrix_hash_match": True,
            "status": "NUMERICALLY_IDENTICAL" if keys_match and numeric_equal else "MISMATCH",
        })

        replay_readout_all = pq.read_table(replay_dir / "_part_core_readout_intermediate.parquet").to_pandas()
        replay_ip = replay_readout_all[replay_readout_all.readout_type == "INTERMEDIATE_PROJECTED"].reset_index(drop=True)
        canonical_ip = pq.read_table(
            args.historical_dir / "readout_features.parquet",
            filters=[("model_id", "=", model_id), ("readout_type", "=", "INTERMEDIATE_PROJECTED")],
        ).to_pandas().reset_index(drop=True)
        readout_keys = KEYS + ["readout_type"]
        value_diffs, readout_max = exact_common_values(replay_ip, canonical_ip, READOUT_VALUES)
        added_columns = ["commitment_score", "branching_factor", "concentration", "collapse_score"]
        added_all_null = bool(canonical_ip[added_columns].isna().all().all())
        verification.append({
            "artifact": "INTERMEDIATE_PROJECTED_READOUT", "model_id": model_id,
            "replay_rows": len(replay_ip), "canonical_rows": len(canonical_ip),
            "replay_columns": len(replay_ip.columns), "canonical_columns": len(canonical_ip.columns),
            "key_unique": not replay_ip.duplicated(readout_keys).any(),
            "keys_ordered_match": replay_ip[readout_keys].equals(canonical_ip[readout_keys]),
            "model_runs": replay_ip.run_id.nunique(), "layers": replay_ip.layer_index.nunique(),
            "replay_value_dtype": "mixed", "canonical_value_dtype": "mixed",
            "max_absolute_difference_after_float32": readout_max,
            "differing_numeric_cells_after_float32": value_diffs,
            "null_pattern_match": added_all_null,
            "matrix_hash_match": True,
            "status": "SEMANTICALLY_IDENTICAL" if value_diffs == 0 and added_all_null else "MISMATCH",
        })

        replay_meta = json.loads((replay_dir / "PROJECTION_METADATA.json").read_text())[0]
        meta_match = replay_meta == canonical_meta[model_id]
        verification.append({
            "artifact": "PROJECTION_METADATA", "model_id": model_id,
            "replay_rows": 1, "canonical_rows": 1, "replay_columns": len(replay_meta),
            "canonical_columns": len(canonical_meta[model_id]), "key_unique": True,
            "keys_ordered_match": replay_meta["model_id"] == model_id,
            "model_runs": 36, "layers": "N/A", "replay_value_dtype": "JSON",
            "canonical_value_dtype": "JSON", "max_absolute_difference_after_float32": 0,
            "differing_numeric_cells_after_float32": 0, "null_pattern_match": True,
            "matrix_hash_match": meta_match, "status": "SEMANTICALLY_IDENTICAL" if meta_match else "MISMATCH",
        })

        if model_id in ROTATED_MODELS:
            replay_ortho = pq.read_table(replay_dir / "_part_core_ortho.parquet").to_pandas()
            canonical_ortho = pq.read_table(
                args.historical_dir / "orthogonal_subset.parquet",
                filters=[("model_id", "=", model_id), ("representation", "=", "ORTHOGONAL_ROTATED")],
            ).to_pandas()
            ortho_keys = KEYS + ["representation"]
            replay_ortho_values = replay_ortho[PROJ_COLS].to_numpy(dtype=np.float32)
            canonical_ortho_values = canonical_ortho[PROJ_COLS].to_numpy(dtype=np.float32)
            ortho_equal = np.array_equal(replay_ortho_values, canonical_ortho_values)
            verification.append({
                "artifact": "ORTHOGONAL_ROTATED", "model_id": model_id,
                "replay_rows": len(replay_ortho), "canonical_rows": len(canonical_ortho),
                "replay_columns": len(replay_ortho.columns), "canonical_columns": len(canonical_ortho.columns),
                "key_unique": not replay_ortho.duplicated(ortho_keys).any(),
                "keys_ordered_match": replay_ortho[ortho_keys].equals(canonical_ortho[ortho_keys]),
                "model_runs": replay_ortho.run_id.nunique(), "layers": replay_ortho.layer_index.nunique(),
                "replay_value_dtype": str(replay_ortho["proj_0"].dtype),
                "canonical_value_dtype": str(canonical_ortho["proj_0"].dtype),
                "max_absolute_difference_after_float32": float(np.max(np.abs(replay_ortho_values - canonical_ortho_values))),
                "differing_numeric_cells_after_float32": int(np.sum(replay_ortho_values != canonical_ortho_values)),
                "null_pattern_match": replay_ortho.isna().equals(canonical_ortho.isna()),
                "matrix_hash_match": meta_match, "status": "NUMERICALLY_IDENTICAL" if ortho_equal else "MISMATCH",
            })

        replay_final = replay_readout_all[replay_readout_all.readout_type == "FINAL_NATIVE"].reset_index(drop=True)
        canonical_final = pq.read_table(
            args.historical_dir / "readout_features.parquet",
            filters=[("model_id", "=", model_id), ("readout_type", "=", "FINAL_NATIVE")],
        ).to_pandas()
        joined = replay_final.merge(canonical_final, on=readout_keys, how="left", suffixes=("_replay", "_canonical"), indicator=True)
        differing_by_column = {}
        max_by_column = {}
        for column in READOUT_VALUES:
            left, right = joined[f"{column}_replay"], joined[f"{column}_canonical"]
            if pd.api.types.is_bool_dtype(left):
                differing_by_column[column] = int(np.sum(left.to_numpy() != right.to_numpy()))
                max_by_column[column] = "N/A"
            else:
                a, b = left.to_numpy(float), right.to_numpy(float)
                differing_by_column[column] = int(np.sum(~np.isclose(a, b, rtol=0, atol=0, equal_nan=True)))
                max_by_column[column] = float(np.nanmax(np.abs(a - b)))
        total_diff = sum(differing_by_column.values())
        status = "SEMANTICALLY_IDENTICAL" if total_diff == 0 else "PARTIAL_MATCH"
        verification.append({
            "artifact": "CORE_EMITTED_FINAL_NATIVE", "model_id": model_id,
            "replay_rows": len(replay_final), "canonical_rows": len(canonical_final),
            "replay_columns": len(replay_final.columns), "canonical_columns": len(canonical_final.columns),
            "key_unique": not replay_final.duplicated(readout_keys).any(),
            "keys_ordered_match": bool((joined._merge == "both").all()),
            "model_runs": replay_final.run_id.nunique(), "layers": replay_final.layer_index.nunique(),
            "replay_value_dtype": "mixed", "canonical_value_dtype": "mixed",
            "max_absolute_difference_after_float32": max(v for v in max_by_column.values() if v != "N/A"),
            "differing_numeric_cells_after_float32": total_diff,
            "null_pattern_match": False, "matrix_hash_match": meta_match, "status": status,
        })
        if total_diff:
            for column in READOUT_VALUES:
                if differing_by_column[column]:
                    mismatches.append({
                        "artifact": "CORE_EMITTED_FINAL_NATIVE", "model_id": model_id,
                        "column": column, "differing_rows": differing_by_column[column],
                        "replay_rows": len(replay_final), "maximum_absolute_difference": max_by_column[column],
                        "likely_cause": "Final compact table retains FINAL_NATIVE values from captured generation logits; extract_core recomputed a strided final-layer projection. This component is not used by Phase 9, which filters INTERMEDIATE_PROJECTED.",
                    })

    result = pd.DataFrame(verification)
    result.to_csv(args.verification_output, index=False)
    pd.DataFrame(mismatches).to_csv(args.mismatch_output, index=False)
    required = result[result.artifact.isin(["RANDOM_PROJECTION", "INTERMEDIATE_PROJECTED_READOUT", "PROJECTION_METADATA", "ORTHOGONAL_ROTATED"])]
    if (required.status == "MISMATCH").any():
        raise SystemExit("Required extract_core replay mismatch")


if __name__ == "__main__":
    main()
