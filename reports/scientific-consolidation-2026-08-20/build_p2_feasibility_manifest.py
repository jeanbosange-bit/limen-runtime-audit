#!/usr/bin/env python3
"""Read-only inventory of archived hidden-state artifacts for P2 feasibility."""
from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np


def npy_header(npz_path: Path, name: str):
    try:
        with zipfile.ZipFile(npz_path) as archive:
            member = f"{name}.npy"
            if member not in archive.namelist():
                return None
            with archive.open(member) as handle:
                version = np.lib.format.read_magic(handle)
                shape, _, dtype = np.lib.format._read_array_header(handle, version)
                return shape, str(dtype)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = list(csv.DictReader((args.source_dir / "RUN_INDEX.csv").open()))
    fields = [
        "model", "run_id", "prompt_id", "task_family", "seed", "generation_id",
        "prompt_sha256", "artifact_path", "artifact_sha256", "run_status",
        "hidden_state_available", "hidden_state_array", "hidden_state_dtype",
        "token_ids_available", "generation_token_ids_available", "t_plus_1_available",
        "t_plus_2_available", "structured_label_available", "regime_label_available",
        "regime_metrics_available", "layer_count", "sequence_length", "hidden_width",
        "generation_steps", "t_plus_1_step_samples", "t_plus_2_step_samples",
        "prompt_seed_replicates", "duplicate_run_id", "duplicate_artifact_sha256",
        "eligible", "exclusion_reason",
    ]
    run_id_counts = {}
    hash_counts = {}
    for run in runs:
        run_id_counts[run["run_id"]] = run_id_counts.get(run["run_id"], 0) + 1
        if run["raw_sha256"]:
            hash_counts[run["raw_sha256"]] = hash_counts.get(run["raw_sha256"], 0) + 1
    replicate_counts = {}
    for run in runs:
        key = (run["model_id"], run["prompt_id"])
        replicate_counts[key] = replicate_counts.get(key, 0) + 1

    output_rows = []
    for run in runs:
        deep = args.source_dir / run["model_deep_path"] if run["model_deep_path"] else None
        wide = args.source_dir / run["model_wide_path"] if run["model_wide_path"] else None
        hidden = npy_header(deep, "hidden_states_all_layers_fp16") if deep and deep.exists() else None
        tokens = npy_header(deep, "token_ids") if deep and deep.exists() else None
        generated_logits = npy_header(deep, "generated_logits_fp16") if deep and deep.exists() else None
        logits_region = npy_header(deep, "logits_region_fp16") if deep and deep.exists() else None
        metadata = {}
        if wide and wide.exists():
            try:
                metadata = json.loads(wide.read_text())
            except Exception:
                metadata = {}
        generated = metadata.get("generated_token_ids")
        generation_steps = len(generated) if isinstance(generated, list) else metadata.get("n_steps")
        if generation_steps is None and generated_logits:
            generation_steps = generated_logits[0][0]
        if generation_steps is None and logits_region:
            generation_steps = logits_region[0][0]
        hidden_steps = None
        if hidden and generation_steps is not None:
            inferred_prompt_len = metadata.get("prompt_len", hidden[0][1] - int(generation_steps))
            hidden_steps = max(0, hidden[0][1] - int(inferred_prompt_len))
        aligned_steps = min(int(generation_steps or 0), int(hidden_steps or 0))
        t1 = max(0, aligned_steps - 1)
        t2 = max(0, aligned_steps - 2)
        reasons = []
        if not hidden:
            reasons.append("HIDDEN_STATE_MISSING")
        if not tokens and not generated:
            reasons.append("TOKEN_IDS_MISSING")
        if t1 == 0:
            reasons.append("NO_ALIGNED_T_PLUS_1")
        taxonomy = metadata.get("taxonomy_status")
        probability_metrics = metadata.get("probability_metrics")
        output_rows.append({
            "model": run["model_id"], "run_id": run["run_id"], "prompt_id": run["prompt_id"],
            "task_family": run["family"], "seed": run["seed"],
            "generation_id": metadata.get("generation_id", ""),
            "prompt_sha256": metadata.get("prompt_sha256", ""),
            "artifact_path": run["model_deep_path"], "artifact_sha256": run["raw_sha256"],
            "run_status": run["status"], "hidden_state_available": bool(hidden),
            "hidden_state_array": "hidden_states_all_layers_fp16" if hidden else "",
            "hidden_state_dtype": hidden[1] if hidden else "", "token_ids_available": bool(tokens),
            "generation_token_ids_available": isinstance(generated, list),
            "t_plus_1_available": t1 > 0, "t_plus_2_available": t2 > 0,
            "structured_label_available": False,
            "regime_label_available": bool(taxonomy and taxonomy != "DEFERRED"),
            "regime_metrics_available": bool(probability_metrics),
            "layer_count": hidden[0][0] - 1 if hidden else "",
            "sequence_length": hidden[0][1] if hidden else "",
            "hidden_width": hidden[0][2] if hidden else "",
            "generation_steps": generation_steps if generation_steps is not None else "",
            "t_plus_1_step_samples": t1, "t_plus_2_step_samples": t2,
            "prompt_seed_replicates": replicate_counts[(run["model_id"], run["prompt_id"])],
            "duplicate_run_id": run_id_counts[run["run_id"]] > 1,
            "duplicate_artifact_sha256": bool(run["raw_sha256"] and hash_counts[run["raw_sha256"]] > 1),
            "eligible": not reasons, "exclusion_reason": ";".join(reasons),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
