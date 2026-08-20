#!/usr/bin/env python3
"""Compare canonical and replayed Phase 9 CSV outputs without modifying either."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUTS = [
    ("logit_lens_model_coverage.csv", ["model_id"]),
    ("phase9_decodability_all8.csv", ["model_id", "representation", "depth_bin"]),
    ("phase9_readout_agg.csv", ["model_id", "depth_bin"]),
    ("decodability_vs_readout.csv", ["model_id", "representation", "depth_bin"]),
    ("aligned_depth_summary.csv", ["model_id", "representation", "depth_bin"]),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for filename, keys in OUTPUTS:
        canonical_path = args.canonical_dir / filename
        replay_path = args.replay_dir / filename
        if not canonical_path.exists() or not replay_path.exists():
            rows.append({"filename": filename, "status": "NOT_REPRODUCED"})
            continue

        canonical = pd.read_csv(canonical_path)
        replay = pd.read_csv(replay_path)
        columns_match = list(canonical.columns) == list(replay.columns)
        row_count_match = len(canonical) == len(replay)
        key_columns_present = all(k in canonical.columns and k in replay.columns for k in keys)
        key_match = bool(
            key_columns_present
            and row_count_match
            and canonical[keys].equals(replay[keys])
            and not canonical.duplicated(keys).any()
            and not replay.duplicated(keys).any()
        )
        null_pattern_match = bool(
            columns_match and row_count_match and canonical.isna().equals(replay.isna())
        )
        semantic_match = bool(columns_match and row_count_match and canonical.equals(replay))

        numeric_columns = [
            c for c in canonical.columns
            if columns_match and pd.api.types.is_numeric_dtype(canonical[c])
        ]
        max_abs_diff = 0.0
        differing_numeric_cells = 0
        if columns_match and row_count_match:
            for column in numeric_columns:
                left = canonical[column].to_numpy(dtype=float)
                right = replay[column].to_numpy(dtype=float)
                differences = np.abs(left - right)
                if not np.all(np.isnan(differences)):
                    max_abs_diff = max(max_abs_diff, float(np.nanmax(differences)))
                differing_numeric_cells += int(
                    np.sum(~np.isclose(left, right, rtol=0, atol=0, equal_nan=True))
                )

        canonical_hash = sha256(canonical_path)
        replay_hash = sha256(replay_path)
        byte_identical = canonical_hash == replay_hash
        if byte_identical:
            status = "BYTE_IDENTICAL"
        elif semantic_match and differing_numeric_cells == 0:
            status = "NUMERICALLY_IDENTICAL"
        elif semantic_match:
            status = "SEMANTICALLY_IDENTICAL"
        elif columns_match and row_count_match and key_match:
            status = "PARTIAL_MATCH"
        else:
            status = "MISMATCH"

        rows.append(
            {
                "filename": filename,
                "status": status,
                "canonical_rows": len(canonical),
                "replay_rows": len(replay),
                "canonical_columns": len(canonical.columns),
                "replay_columns": len(replay.columns),
                "column_names_and_order_match": columns_match,
                "primary_key": "|".join(keys),
                "primary_keys_unique_and_ordered_match": key_match,
                "null_pattern_match": null_pattern_match,
                "semantic_dataframe_equality": semantic_match,
                "differing_numeric_cells": differing_numeric_cells,
                "maximum_absolute_numeric_difference": max_abs_diff,
                "canonical_sha256": canonical_hash,
                "replay_sha256": replay_hash,
            }
        )

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    if not (result["status"] == "BYTE_IDENTICAL").all():
        raise SystemExit("One or more Phase 9 replay outputs are not byte-identical")


if __name__ == "__main__":
    main()
