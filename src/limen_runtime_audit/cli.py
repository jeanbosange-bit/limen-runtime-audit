from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audit import audit_arrays, write_report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit an exported LLM trajectory")
    p.add_argument("input", type=Path, help="NPZ with hidden_states and optional logits")
    p.add_argument("--output", type=Path, default=Path("limen_audit_output"))
    p.add_argument("--metadata", type=Path, help="JSON extraction manifest")
    return p


def main() -> None:
    args = parser().parse_args()
    metadata = {}
    if args.metadata:
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    with np.load(args.input, allow_pickle=False) as data:
        if "hidden_states" not in data:
            raise SystemExit("input NPZ is missing 'hidden_states'")
        hidden_states = data["hidden_states"]
        logits = data["logits"] if "logits" in data else None
    audit = audit_arrays(hidden_states, logits, metadata)
    write_report(audit, args.output, args.input)
    print(f"Audit written to {args.output}")


if __name__ == "__main__":
    main()
