#!/usr/bin/env python3
"""Replay only extract_features.py's generation-native/future scalar readout path.

No model is loaded and no inference or generation occurs. Hidden-state values are not
read; only the archived NPY header is inspected to recover layer/sequence dimensions.
"""
from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


READOUT_SCHEMA = pa.schema([
    ("model_id", pa.string()), ("run_id", pa.string()), ("prompt_id", pa.string()),
    ("prompt_family", pa.string()), ("seed", pa.string()), ("token_step", pa.int64()),
    ("layer_index", pa.int64()), ("num_layers", pa.int64()), ("readout_type", pa.string()),
    ("top1_token_id", pa.int64()), ("top1_logit", pa.float64()), ("top1_prob", pa.float64()),
    ("top2_token_id", pa.int64()), ("top2_logit", pa.float64()), ("top2_prob", pa.float64()),
    ("target_logit", pa.float64()), ("target_probability", pa.float64()),
    ("target_rank", pa.int64()), ("is_target_top1", pa.bool_()), ("entropy", pa.float64()),
    ("commitment_score", pa.float64()), ("branching_factor", pa.float64()),
    ("concentration", pa.float64()), ("collapse_score", pa.float64()),
])


def npy_shape(npz_path: Path, array_name: str) -> tuple[int, ...]:
    with zipfile.ZipFile(npz_path) as archive, archive.open(f"{array_name}.npy") as handle:
        version = np.lib.format.read_magic(handle)
        shape, _, _ = np.lib.format._read_array_header(handle, version)
    return shape


def entropy_from_logits(row: np.ndarray) -> float:
    row = row.astype(np.float64)
    probability = np.exp(row - row.max())
    probability = probability / probability.sum()
    return float(-(probability * np.log(probability + 1e-9)).sum())


def top_stats_from_logits(row: np.ndarray, target_id: int) -> dict[str, object]:
    row = row.astype(np.float64)
    probability = np.exp(row - row.max())
    probability = probability / probability.sum()
    order = np.argsort(-row)
    top1_id, top2_id = int(order[0]), int(order[1])
    rank = int(np.where(order == target_id)[0][0])
    return {
        "top1_token_id": top1_id, "top1_logit": float(row[top1_id]), "top1_prob": float(probability[top1_id]),
        "top2_token_id": top2_id, "top2_logit": float(row[top2_id]), "top2_prob": float(probability[top2_id]),
        "target_logit": float(row[target_id]), "target_probability": float(probability[target_id]),
        "target_rank": rank, "is_target_top1": bool(target_id == top1_id),
        "entropy": entropy_from_logits(row),
    }


def model_wide_by_step(source_dir: Path, run: dict[str, str]) -> dict[int, dict[str, object]]:
    if not run["model_wide_path"]:
        return {}
    path = source_dir / run["model_wide_path"]
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return {metric["step"]: metric for metric in data.get("probability_metrics", [])}


def extract_run(source_dir: Path, model_id: str, run: dict[str, str]) -> list[dict[str, object]]:
    path = source_dir / run["model_deep_path"]
    with np.load(path) as archive:
        token_ids = archive["token_ids"]
        if "generated_logits_fp16" not in archive.files:
            return []
        logits = archive["generated_logits_fp16"].astype(np.float32)
    hidden_shape = npy_shape(path, "hidden_states_all_layers_fp16")
    num_layers = hidden_shape[0] - 1
    sequence_length = hidden_shape[1]
    num_steps = logits.shape[0]
    prompt_length = sequence_length - num_steps
    dynamic = model_wide_by_step(source_dir, run)
    rows: list[dict[str, object]] = []

    for step in range(num_steps):
        position = prompt_length + step
        if position >= sequence_length:
            continue
        target_id = int(token_ids[position + 1]) if position + 1 < len(token_ids) else None
        if target_id is not None:
            stats = top_stats_from_logits(logits[step], target_id)
            regime = dynamic.get(step, {})
            rows.append({
                "model_id": model_id, "run_id": run["run_id"], "prompt_id": run["prompt_id"],
                "prompt_family": run["family"], "seed": run["seed"], "token_step": step,
                "layer_index": num_layers, "num_layers": num_layers, "readout_type": "FINAL_NATIVE",
                **stats, "commitment_score": regime.get("commitment_score"),
                "branching_factor": regime.get("branching_factor"),
                "concentration": regime.get("concentration"), "collapse_score": regime.get("collapse_score"),
            })

        for offset in (2, 3):
            future_step = step + offset
            future_position = prompt_length + future_step
            if future_step < logits.shape[0] and future_position + 1 < len(token_ids):
                future_target = int(token_ids[future_position + 1])
                row = logits[future_step].astype(np.float64)
                probability = np.exp(row - row.max())
                probability = probability / probability.sum()
                order = np.argsort(-row)
                rows.append({
                    "model_id": model_id, "run_id": run["run_id"], "prompt_id": run["prompt_id"],
                    "prompt_family": run["family"], "seed": run["seed"], "token_step": step,
                    "layer_index": num_layers, "num_layers": num_layers,
                    "readout_type": f"FUTURE_T_PLUS_{offset}", "target_rank": int(np.where(order == future_target)[0][0]),
                    "target_logit": float(row[future_target]), "target_probability": float(probability[future_target]),
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--panel-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="*")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = json.loads(args.panel_file.read_text())
    run_index = list(csv.DictReader((args.source_dir / "RUN_INDEX.csv").open()))

    for panel_name in ("core", "extended"):
        models = sorted(set(panel[panel_name]))
        if args.models:
            models = [model for model in models if model in set(args.models)]
        if not models:
            continue
        suffix = "" if not args.models else "__" + "--".join(model.replace("/", "--") for model in models)
        output = args.output_dir / f"_part_{panel_name}_readout{suffix}.parquet"
        writer = pq.ParquetWriter(output, READOUT_SCHEMA)
        count = 0
        for model_id in models:
            print(f"starting {model_id}", flush=True)
            runs = [r for r in run_index if r["model_id"] == model_id and r["status"] == "SUCCESS" and r["model_deep_path"]]
            for run in runs:
                rows = extract_run(args.source_dir, model_id, run)
                if rows:
                    writer.write_table(pa.Table.from_pylist(rows, schema=READOUT_SCHEMA))
                    count += len(rows)
        writer.close()
        print(f"{output.name}: {count} rows")


if __name__ == "__main__":
    main()
