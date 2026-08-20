#!/usr/bin/env python3
"""Compare isolated replay tables with immutable historical compact tables."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = {
    "h_random_projection.parquet": ["model_id", "run_id", "token_step", "layer_index"],
    "orthogonal_subset.parquet": ["model_id", "run_id", "token_step", "layer_index", "representation"],
    "readout_features.parquet": ["model_id", "run_id", "token_step", "layer_index", "readout_type"],
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(name: str, replay_dir: Path, historical_dir: Path) -> dict[str, object]:
    replay_path, historical_path = replay_dir / name, historical_dir / name
    replay, historical = pd.read_parquet(replay_path), pd.read_parquet(historical_path)
    same_columns = list(replay.columns) == list(historical.columns)
    same_dtypes = replay.dtypes.astype(str).tolist() == historical.dtypes.astype(str).tolist()
    same_nulls = replay.isna().sum().tolist() == historical.isna().sum().tolist()
    same_keys = replay[KEYS[name]].equals(historical[KEYS[name]])
    numeric = [c for c in replay.columns if c in historical and pd.api.types.is_numeric_dtype(replay[c])]
    replay_keyed = replay.sort_values(KEYS[name], kind="stable").reset_index(drop=True)
    historical_keyed = historical.sort_values(KEYS[name], kind="stable").reset_index(drop=True)
    key_sets_equal = replay_keyed[KEYS[name]].equals(historical_keyed[KEYS[name]])
    maximum = mean = 0.0
    mismatches = 0
    if len(replay) == len(historical) and numeric:
        left = replay_keyed[numeric].to_numpy(dtype=float)
        right = historical_keyed[numeric].to_numpy(dtype=float)
        difference = np.abs(left - right)
        finite = np.isfinite(difference)
        maximum = float(difference[finite].max()) if finite.any() else 0.0
        mean = float(difference[finite].mean()) if finite.any() else 0.0
        mismatches = int((~np.isclose(left, right, rtol=0, atol=0, equal_nan=True)).sum())
    nonnumeric_equal = all(
        replay_keyed[c].fillna("<NULL>").equals(historical_keyed[c].fillna("<NULL>"))
        for c in replay.columns
        if c not in numeric
    )
    semantic = len(replay) == len(historical) and same_columns and key_sets_equal and same_nulls and nonnumeric_equal
    numerical = semantic and mismatches == 0
    byte = digest(replay_path) == digest(historical_path)
    status = "BYTE_IDENTICAL" if byte else "NUMERICALLY_IDENTICAL" if numerical else "SEMANTICALLY_IDENTICAL" if semantic else "MISMATCH"
    return {
        "filename": name,
        "status": status,
        "replay_rows": len(replay),
        "historical_rows": len(historical),
        "columns": len(replay.columns),
        "column_order_equal": same_columns,
        "dtype_equal": same_dtypes,
        "row_order_and_keys_equal": same_keys,
        "keyed_semantic_equality": semantic,
        "null_pattern_equal": same_nulls,
        "numeric_mismatch_count": mismatches,
        "max_abs_difference": maximum,
        "mean_abs_difference": mean,
        "replay_sha256": digest(replay_path),
        "historical_sha256": digest(historical_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--historical-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [compare(name, args.replay_dir, args.historical_dir) for name in KEYS]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
