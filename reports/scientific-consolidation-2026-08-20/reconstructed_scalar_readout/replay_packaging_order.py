#!/usr/bin/env python3
"""Replay the exact recovered readout finalizer using independently replayed parts."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pyarrow.parquet as pq


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scalar-replay", type=Path, required=True)
    parser.add_argument("--core-readout", type=Path, required=True)
    parser.add_argument("--finalizer", type=Path, required=True)
    parser.add_argument("--historical-readout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pq.read_table(args.scalar_replay / "_part_core_readout_complete.parquet"),
                   args.output_dir / "_part_core_readout.parquet")
    pq.write_table(pq.read_table(args.scalar_replay / "_part_extended_readout_complete.parquet"),
                   args.output_dir / "_part_extended_readout.parquet")
    pq.write_table(pq.read_table(args.core_readout),
                   args.output_dir / "_part_core_readout_intermediate.parquet")

    source = args.finalizer.read_text()
    historical = 'OUT = Path("/home/jeanbatuli/limen-runtime-audit/reports/deep-feature-extraction-2026-08-19")'
    source = source.replace(historical, f"OUT = Path({str(args.output_dir)!r})")
    exec(compile(source, str(args.finalizer), "exec"), {"__name__": "__main__", "__file__": str(args.finalizer)})
    replay = args.output_dir / "readout_features.parquet"
    print({
        "replay_sha256": digest(replay),
        "historical_sha256": digest(args.historical_readout),
        "byte_identical": digest(replay) == digest(args.historical_readout),
    })


if __name__ == "__main__":
    main()
