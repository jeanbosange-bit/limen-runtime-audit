#!/usr/bin/env python3
"""Execute the byte-preserved recovered Phase 9 script with a safe replay output.

Only the exact hardcoded OUT assignment is replaced in memory. The historical source
file and historical outputs remain unchanged. pilot_metrics.csv is copied as a required
read-only logical input into the replay directory before execution.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


HISTORICAL_OUT = (
    'OUT = "/home/jeanbatuli/V20_probabilistic_dynamics/'
    'reports/decodability-v0-2026-08-19"'
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    args = parser.parse_args()

    args.replay_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        args.canonical_dir / "pilot_metrics.csv", args.replay_dir / "pilot_metrics.csv"
    )
    source = args.script.read_text(encoding="utf-8")
    if source.count(HISTORICAL_OUT) != 1:
        raise RuntimeError("Expected exactly one historical OUT assignment")
    safe_out = f'OUT = "{args.replay_dir.resolve()}"'
    replay_source = source.replace(HISTORICAL_OUT, safe_out, 1)
    code = compile(replay_source, str(args.script), "exec")
    namespace = {"__name__": "__main__", "__file__": str(args.script)}
    exec(code, namespace)


if __name__ == "__main__":
    main()
