#!/usr/bin/env python3
"""Replay historically successful extract_core models in isolated processes."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SUCCESSFUL_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "gpt2",
    "EleutherAI/pythia-160m",
    "facebook/opt-125m",
    "deepseek-ai/deepseek-coder-1.3b-instruct",
]


def slug(model_id: str) -> str:
    return model_id.lower().replace("/", "--").replace(".", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--session-jsonl", type=Path, required=True)
    parser.add_argument("--historical-feature-dir", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    args = parser.parse_args()

    for model_id in SUCCESSFUL_MODELS:
        output = args.replay_root / f"isolated_{slug(model_id)}"
        command = [
            sys.executable,
            str(args.runner),
            "--script", str(args.script),
            "--session-jsonl", str(args.session_jsonl),
            "--historical-feature-dir", str(args.historical_feature_dir),
            "--replay-dir", str(output),
            "--models", model_id,
        ]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
