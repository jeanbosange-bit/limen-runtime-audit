#!/usr/bin/env python3
"""Assemble model-isolated scalar shards in historical order and verify them."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


KEYS = ["model_id", "run_id", "token_step", "layer_index", "readout_type"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--historical-readout", type=Path, required=True)
    parser.add_argument("--verification-csv", type=Path, required=True)
    args = parser.parse_args()
    shards = args.replay_root / "shards"
    isolated = args.replay_root / "isolated"
    core_names = [
        "_part_core_readout__EleutherAI--pythia-160m--HuggingFaceTB--SmolLM-360M-Instruct--Qwen--Qwen2.5-0.5B-Instruct--Qwen--Qwen2.5-1.5B-Instruct--TinyLlama--TinyLlama-1.1B-Chat-v1.0.parquet",
        "_part_core_readout__deepseek-ai--deepseek-coder-1.3b-instruct--facebook--opt-125m--gpt2--ibm-granite--granite-3.0-1b-a400m-instruct--meta-llama--Llama-3.2-1B.parquet",
        "_part_core_readout__microsoft--phi-1_5--tiiuae--Falcon3-1B-Instruct.parquet",
    ]
    extended_paths = [
        shards / "_part_extended_readout__EleutherAI--gpt-neo-125m--EleutherAI--pythia-410m.parquet",
        isolated / "_part_extended_readout__EleutherAI--pythia-70m.parquet",
        shards / "_part_extended_readout__HuggingFaceTB--SmolLM2-135M--Qwen--Qwen2.5-Coder-0.5B-Instruct--bigscience--bloomz-560m--distilgpt2.parquet",
    ]
    core_paths = [shards / name for name in core_names]
    core = pa.concat_tables([pq.read_table(path) for path in core_paths], promote_options="default")
    extended = pa.concat_tables([pq.read_table(path) for path in extended_paths], promote_options="default")
    core_path = args.replay_root / "_part_core_readout_complete.parquet"
    extended_path = args.replay_root / "_part_extended_readout_complete.parquet"
    pq.write_table(core, core_path)
    pq.write_table(extended, extended_path)

    replay = pa.concat_tables([core, extended], promote_options="default").to_pandas()
    combined_path = args.replay_root / "scalar_readout_combined.parquet"
    replay.to_parquet(combined_path, index=False)
    packaged = pd.read_parquet(args.historical_readout)
    historical = packaged.iloc[:153279].reset_index(drop=True)
    replay = replay.reset_index(drop=True)

    numeric = [column for column in replay if pd.api.types.is_numeric_dtype(replay[column])]
    left = replay[numeric].to_numpy(dtype=float)
    right = historical[numeric].to_numpy(dtype=float)
    mismatch = ~np.isclose(left, right, rtol=0, atol=0, equal_nan=True)
    difference = np.abs(left - right)
    finite = np.isfinite(difference)
    nonnumeric = [column for column in replay if column not in numeric]
    nonnumeric_mismatches = sum(
        int((replay[column].fillna("<NULL>") != historical[column].fillna("<NULL>")).sum())
        for column in nonnumeric
    )
    row = {
        "classification": "BYTE_IDENTICAL" if sha256(combined_path) == sha256(args.historical_readout) else
                          "NUMERICALLY_IDENTICAL" if not mismatch.any() and nonnumeric_mismatches == 0 else "MISMATCH",
        "replay_rows": len(replay), "historical_scalar_rows": len(historical),
        "models": replay.model_id.nunique(), "runs": replay.run_id.nunique(),
        "columns": len(replay.columns), "column_order_equal": list(replay.columns) == list(historical.columns),
        "dtype_equal": replay.dtypes.astype(str).tolist() == historical.dtypes.astype(str).tolist(),
        "row_keys_equal": replay[KEYS].equals(historical[KEYS]),
        "null_pattern_equal": replay.isna().equals(historical.isna()),
        "numeric_mismatch_count": int(mismatch.sum()),
        "max_abs_difference": float(difference[finite].max()) if finite.any() else 0.0,
        "mean_abs_difference": float(difference[finite].mean()) if finite.any() else 0.0,
        "nonnumeric_mismatch_count": nonnumeric_mismatches,
        "replay_sha256": sha256(combined_path),
        "historical_full_sha256": sha256(args.historical_readout),
    }
    with args.verification_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader(); writer.writerow(row)
    print(row)
    print(replay.groupby("readout_type").size())


if __name__ == "__main__":
    main()
