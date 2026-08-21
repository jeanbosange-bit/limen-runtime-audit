#!/usr/bin/env python3
"""Audit per-model future-token class support under nested prompt-held-out folds."""
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    prompt_order = sorted(manifest.prompt_id.dropna().unique(), key=lambda p: hashlib.sha256(p.encode()).hexdigest())
    records = []
    for row in manifest.itertuples(index=False):
        if not row.hidden_state_available or not row.token_ids_available or not row.generation_steps:
            continue
        path = Path(row.artifact_path)
        if not path.is_absolute():
            path = args.source_dir / path
        with np.load(path) as archive:
            token_ids = archive["token_ids"]
        steps = int(row.generation_steps)
        prompt_len = len(token_ids) - steps
        for offset in (1, 2):
            for step in range(max(0, steps - offset)):
                records.append((row.model, row.run_id, row.prompt_id, offset, int(token_ids[prompt_len + step + offset])))
    frame = pd.DataFrame(records, columns=["model", "run_id", "prompt_id", "offset", "target_token_id"])
    rows = []
    for (model, offset), group in frame.groupby(["model", "offset"]):
        counts = Counter(group.target_token_id)
        fold_coverages = []
        fold_majorities = []
        for index, test_prompt in enumerate(prompt_order):
            validation_prompt = prompt_order[(index + 1) % len(prompt_order)]
            train = group[~group.prompt_id.isin([test_prompt, validation_prompt])]
            test = group[group.prompt_id.eq(test_prompt)]
            if test.empty or train.empty:
                continue
            train_classes = set(train.target_token_id)
            fold_coverages.append(test.target_token_id.isin(train_classes).mean())
            majority = train.target_token_id.value_counts().index[0]
            fold_majorities.append((test.target_token_id == majority).mean())
        rows.append({
            "model": model, "target": f"T_PLUS_{offset}_TOKEN_ID", "samples": len(group),
            "prompts": group.prompt_id.nunique(), "runs": group.run_id.nunique(),
            "unique_classes": len(counts), "singleton_classes": sum(value == 1 for value in counts.values()),
            "singleton_sample_fraction": sum(value for value in counts.values() if value == 1) / len(group),
            "global_majority_fraction": max(counts.values()) / len(group),
            "mean_outer_test_seen_in_train_fraction": float(np.mean(fold_coverages)) if fold_coverages else "",
            "min_outer_test_seen_in_train_fraction": float(np.min(fold_coverages)) if fold_coverages else "",
            "mean_outer_majority_accuracy": float(np.mean(fold_majorities)) if fold_majorities else "",
            "outer_folds_with_test_rows": len(fold_coverages),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} rows; token samples={len(frame)}")


if __name__ == "__main__":
    main()
