#!/usr/bin/env python3
"""Replay the recovered geometry-control/finalization chain into an isolated directory.

This utility never writes to the historical feature directory. The scalar FINAL_NATIVE
part is reconstructed from the immutable packaged table solely to test finalizer row
precedence; this is not claimed as an independent replay of scalar extraction.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


MODEL_DIRS = [
    "isolated_qwen--qwen2_5-0_5b-instruct",
    "isolated_tinyllama--tinyllama-1_1b-chat-v1_0",
    "isolated_gpt2",
    "isolated_eleutherai--pythia-160m",
    "isolated_facebook--opt-125m",
    "isolated_deepseek-ai--deepseek-coder-1_3b-instruct",
    "isolated_granite",
    "isolated_smollm",
]


def concat_parts(root: Path, name: str, destination: Path) -> None:
    tables = []
    for directory in MODEL_DIRS:
        source = root / directory / name
        if source.exists():
            tables.append(pq.read_table(source))
    pq.write_table(pa.concat_tables(tables, promote_options="default"), destination)


def execute_recovered(script: Path, replay: Path) -> None:
    source = script.read_text()
    historical = 'OUT = Path("/home/jeanbatuli/limen-runtime-audit/reports/deep-feature-extraction-2026-08-19")'
    replacement = f"OUT = Path({str(replay)!r})"
    if source.count(historical) != 1:
        raise RuntimeError(f"expected one historical OUT assignment in {script}")
    code = compile(source.replace(historical, replacement), str(script), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(script)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isolated-root", type=Path, required=True)
    parser.add_argument("--historical-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    args = parser.parse_args()
    args.replay_dir.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent

    concat_parts(args.isolated_root, "_part_core_proj.parquet", args.replay_dir / "_part_core_proj.parquet")
    concat_parts(args.isolated_root, "_part_core_ortho.parquet", args.replay_dir / "_part_core_ortho.parquet")
    concat_parts(
        args.isolated_root,
        "_part_core_readout_intermediate.parquet",
        args.replay_dir / "_part_core_readout_intermediate.parquet",
    )

    # The deleted scalar parts are unavailable. Filtering the immutable final package
    # reconstructs their semantic union for the narrow purpose of replaying precedence.
    readout = pd.read_parquet(args.historical_dir / "readout_features.parquet")
    scalar = readout[readout["readout_type"] != "INTERMEDIATE_PROJECTED"].reset_index(drop=True)
    scalar.to_parquet(args.replay_dir / "_part_core_readout.parquet", index=False)

    execute_recovered(here / "build_geometry_control.py", args.replay_dir)
    execute_recovered(here / "finalize_and_quality.py", args.replay_dir)

    # Exact archived post-finalization recast operation.
    for filename in ["h_random_projection.parquet", "orthogonal_subset.parquet"]:
        frame = pd.read_parquet(args.replay_dir / filename)
        for column in [c for c in frame.columns if c.startswith("proj_")]:
            frame[column] = frame[column].astype("float32")
        frame.to_parquet(args.replay_dir / filename, index=False)


if __name__ == "__main__":
    main()
