#!/usr/bin/env python3
"""Safely replay the byte-preserved archived extract_core.py.

The archived dependency bodies are recovered by replaying their Claude-session Write/Edit
events. Only output/scratch/import paths are substituted in memory. Historical inputs and
outputs remain read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


SESSION_ID = "5d793c3e-5e08-4a66-9200-8acfbc5066c9"
HISTORICAL_SCRATCH = (
    "/tmp/claude-1000/-home-jeanbatuli-V20-probabilistic-dynamics/"
    f"{SESSION_ID}/scratchpad"
)
HISTORICAL_OUT_ASSIGNMENT = (
    'OUT = Path("/home/jeanbatuli/limen-runtime-audit/'
    'reports/deep-feature-extraction-2026-08-19")'
)
EXPECTED_HASHES = {
    "extract_core.py": "18033dbd8be2a3684f9b53e4c756a5e358de32bf2c3811b517935ad270d37a8c",
    "extract_features.py": "c8fa3842e991ba99f1ba5987b26d4c549437dc86e5549ece08d849e08e41a80a",
    "find_final_norm.py": "93753eb3477bbc8d9cbe8edda940410ca654530b47fb258f6713c155551e7d86",
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def recover_final_body(session_path: Path, filename: str) -> str:
    content: str | None = None
    for line in session_path.open(encoding="utf-8", errors="ignore"):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = event.get("message", {})
        blocks = message.get("content", []) if isinstance(message, dict) else []
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tool = block.get("name")
            tool_input = block.get("input", {})
            if not str(tool_input.get("file_path", "")).endswith(filename):
                continue
            if tool == "Write":
                content = tool_input["content"]
            elif tool == "Edit" and content is not None:
                old = tool_input["old_string"]
                if content.count(old) != 1:
                    raise RuntimeError(f"Cannot replay unique edit for {filename}")
                content = content.replace(old, tool_input["new_string"], 1)
    if content is None:
        raise RuntimeError(f"No archived body found for {filename}")
    if sha256_text(content) != EXPECTED_HASHES[filename]:
        raise RuntimeError(f"Recovered hash mismatch for {filename}")
    return content


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"Expected one {label}, found {source.count(old)}")
    return source.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--session-jsonl", type=Path, required=True)
    parser.add_argument("--historical-feature-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional model isolation for replaying the unchanged per-model extraction after an unstable full-panel GPU run.",
    )
    args = parser.parse_args()

    replay_dir = args.replay_dir.resolve()
    support_dir = replay_dir / "support"
    state_dir = replay_dir / "state"
    support_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        args.historical_feature_dir / "_core_extended.json",
        replay_dir / "_core_extended.json",
    )
    if args.models:
        panel = json.loads((replay_dir / "_core_extended.json").read_text())
        unknown = sorted(set(args.models) - set(panel["core"]))
        if unknown:
            raise RuntimeError(f"Requested models are not in historical core panel: {unknown}")
        panel["core"] = args.models
        (replay_dir / "_core_extended.json").write_text(
            json.dumps(panel, indent=2), encoding="utf-8"
        )

    archived_core = recover_final_body(args.session_jsonl, "extract_core.py")
    materialized_core = args.script.read_text(encoding="utf-8")
    if materialized_core != archived_core:
        raise RuntimeError("Materialized extract_core.py is not byte-identical to archive")

    extract_features = recover_final_body(args.session_jsonl, "extract_features.py")
    extract_features = replace_once(
        extract_features,
        HISTORICAL_OUT_ASSIGNMENT,
        f'OUT = Path("{replay_dir}")',
        "extract_features OUT assignment",
    )
    extract_features = extract_features.replace(HISTORICAL_SCRATCH, str(state_dir))
    (support_dir / "extract_features.py").write_text(extract_features, encoding="utf-8")
    (support_dir / "find_final_norm.py").write_text(
        recover_final_body(args.session_jsonl, "find_final_norm.py"), encoding="utf-8"
    )

    replay_core = replace_once(
        archived_core,
        f'sys.path.insert(0, "{HISTORICAL_SCRATCH}")',
        f'sys.path.insert(0, "{support_dir}")',
        "extract_core import path",
    )
    replay_core = replace_once(
        replay_core,
        f'SCRATCH = Path("{HISTORICAL_SCRATCH}")',
        f'SCRATCH = Path("{state_dir}")',
        "extract_core scratch path",
    )
    code = compile(replay_core, str(args.script), "exec")
    namespace = {"__name__": "__main__", "__file__": str(args.script)}
    exec(code, namespace)


if __name__ == "__main__":
    main()
