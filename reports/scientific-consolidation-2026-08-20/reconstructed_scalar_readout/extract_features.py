"""
Main feature-extraction engine for LIMEN Deep Feature Extraction V0.
Read-only against the campaign raw data. Writes only into
reports/deep-feature-extraction-2026-08-19/ and this scratchpad.

Formulas reused verbatim from reports/multi-tensor-atlas-v0/build_atlas.py
(this program's canonical LIMEN geometry/attention formulas):
  H_curvature: 1 - cos(delta_t, delta_t-1) via consecutive second-difference
  Oattn_delta_norm: ||O_t|| - ||O_t-1||
  A_dispersion: mean per-head entropy of the row-normalized causal attention row
  A_concentration: mean per-head max attention weight
  entropy/top2_gap: from softmax(logits)
"""
import csv
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

SRC = Path("/home/jeanbatuli/limen-runtime-audit/reports/multimodel-expansion-2026-08-19")
OUT = Path("/home/jeanbatuli/limen-runtime-audit/reports/deep-feature-extraction-2026-08-19")
SCRATCH = Path("/tmp/claude-1000/-home-jeanbatuli-V20-probabilistic-dynamics/5d793c3e-5e08-4a66-9200-8acfbc5066c9/scratchpad")

core_extended = json.loads((OUT / "_core_extended.json").read_text())
CORE_MODELS = set(core_extended["core"])
EXTENDED_MODELS = set(core_extended["extended"])
ALL_MODELS = CORE_MODELS | EXTENDED_MODELS

run_index = list(csv.DictReader(open(SRC / "RUN_INDEX.csv")))
runs_by_model = {}
for r in run_index:
    if r["model_id"] in ALL_MODELS and r["status"] == "SUCCESS" and r["model_deep_path"]:
        runs_by_model.setdefault(r["model_id"], []).append(r)

LOG_PATH = OUT / "extraction_run.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def entropy_from_logits(row):
    row = row.astype(np.float64)
    p = np.exp(row - row.max())
    p = p / p.sum()
    return float(-(p * np.log(p + 1e-9)).sum())


def top_stats_from_logits(row, target_id):
    row = row.astype(np.float64)
    p = np.exp(row - row.max())
    p = p / p.sum()
    order = np.argsort(-row)
    top1_id, top2_id = int(order[0]), int(order[1])
    rank = int(np.searchsorted(-row[order], -row[target_id], side="left")) if False else int(np.where(order == target_id)[0][0])
    return {
        "top1_token_id": top1_id, "top1_logit": float(row[top1_id]), "top1_prob": float(p[top1_id]),
        "top2_token_id": top2_id, "top2_logit": float(row[top2_id]), "top2_prob": float(p[top2_id]),
        "target_logit": float(row[target_id]), "target_probability": float(p[target_id]),
        "target_rank": rank, "is_target_top1": bool(target_id == top1_id),
        "entropy": entropy_from_logits(row),
    }


# ============================================================
# Per-run feature extraction (geometry + attention + QKV + final-native readout)
# ============================================================
def process_run(model_id, run, model_wide_by_step=None, logit_lens_ctx=None):
    """Returns list of layerwise row dicts, list of attention row dicts, list of qkv row dicts,
    list of readout row dicts (final + optionally intermediate), dynamic_regime rows, warnings."""
    npz_path = SRC / run["model_deep_path"]
    if not npz_path.exists():
        return None, f"RAW_FILE_MISSING:{npz_path}"
    try:
        npz = np.load(npz_path)
    except Exception as e:
        return None, f"LOAD_ERROR:{e}"

    required = ["token_ids", "hidden_states_all_layers_fp16"]
    for k in required:
        if k not in npz.files:
            return None, f"MISSING_ARRAY:{k}"

    token_ids = npz["token_ids"]
    H = npz["hidden_states_all_layers_fp16"].astype(np.float32)  # (n_layers+1, seq_len, d_model)
    n_layers = H.shape[0] - 1
    seq_len = H.shape[1]

    logits = npz["generated_logits_fp16"].astype(np.float32) if "generated_logits_fp16" in npz.files else None
    A = npz["attentions_all_layers_fp16"].astype(np.float32) if "attentions_all_layers_fp16" in npz.files else None
    O = npz["attn_output_all_layers_fp16"].astype(np.float32) if "attn_output_all_layers_fp16" in npz.files else None
    Q = npz["q_all_layers_fp16"].astype(np.float32) if "q_all_layers_fp16" in npz.files else None
    K = npz["k_all_layers_fp16"].astype(np.float32) if "k_all_layers_fp16" in npz.files else (npz["k_all_layers_fp32"].astype(np.float32) if "k_all_layers_fp32" in npz.files else None)
    V = npz["v_all_layers_fp16"].astype(np.float32) if "v_all_layers_fp16" in npz.files else (npz["v_all_layers_fp32"].astype(np.float32) if "v_all_layers_fp32" in npz.files else None)

    n_steps = logits.shape[0] if logits is not None else (seq_len - (seq_len - H.shape[1]))
    prompt_len = seq_len - n_steps if logits is not None else None

    run_id = run["run_id"]
    prompt_id = run["prompt_id"]
    family = run["family"]
    seed = run["seed"]

    layer_rows, attn_rows, qkv_rows, readout_rows, regime_rows = [], [], [], [], []
    nan_inf_count = 0

    # ---- H geometry (CROSS_LAYER, per step at final generated position + full grid) ----
    for step in range(n_steps if prompt_len is not None else 0):
        pos = prompt_len + step
        if pos >= seq_len:
            continue
        for layer in range(n_layers + 1):
            h_t = H[layer, pos]
            if not np.all(np.isfinite(h_t)):
                nan_inf_count += 1
                continue
            norm_t = float(np.linalg.norm(h_t))
            h0 = H[0, pos]
            h_final = H[n_layers, pos]
            norm0 = float(np.linalg.norm(h0)) + 1e-9
            normf = float(np.linalg.norm(h_final)) + 1e-9
            cos_init = float(np.dot(h_t, h0) / (norm_t * norm0 + 1e-9))
            cos_final = float(np.dot(h_t, h_final) / (norm_t * normf + 1e-9))
            displacement = float(np.linalg.norm(h_t - h0))
            delta_norm = cos_prev = curvature = None
            if layer - 1 >= 0:
                h_prev = H[layer - 1, pos]
                norm_prev = float(np.linalg.norm(h_prev))
                delta_norm = norm_t - norm_prev
                cos_prev = float(np.dot(h_t, h_prev) / (norm_t * norm_prev + 1e-9))
            if layer - 2 >= 0:
                h_prev, h_prev2 = H[layer - 1, pos], H[layer - 2, pos]
                d1, d0 = h_t - h_prev, h_prev - h_prev2
                n1, n0 = np.linalg.norm(d1), np.linalg.norm(d0)
                curvature = 1 - float(np.dot(d0, d1) / (n0 * n1 + 1e-9))
            layer_rows.append({
                "model_id": model_id, "run_id": run_id, "prompt_id": prompt_id, "prompt_family": family,
                "seed": seed, "token_step": step, "layer_index": layer, "num_layers": n_layers,
                "normalized_depth": layer / n_layers if n_layers else None,
                "hidden_norm": norm_t, "hidden_mean": float(h_t.mean()), "hidden_std": float(h_t.std()),
                "delta_norm_cross_layer": delta_norm, "cosine_to_previous_layer": cos_prev, "curvature_cross_layer": curvature,
                "trajectory_displacement_from_initial": displacement, "cosine_to_initial_layer": cos_init, "cosine_to_final_layer": cos_final,
            })

        # ---- QKV compact (per layer, this step) ----
        for layer in range(n_layers):
            row = {"model_id": model_id, "run_id": run_id, "prompt_id": prompt_id, "prompt_family": family,
                   "seed": seed, "token_step": step, "layer_index": layer, "num_layers": n_layers}
            for name, arr in [("Q", Q), ("K", K), ("V", V)]:
                if arr is None:
                    row[f"{name}_norm_mean"] = None
                    row[f"{name}_norm_std"] = None
                    row[f"{name}_available"] = False
                    continue
                vec = arr[layer, :, pos, :]  # (n_heads, head_dim)
                norms = np.linalg.norm(vec, axis=-1)
                row[f"{name}_norm_mean"] = float(norms.mean())
                row[f"{name}_norm_std"] = float(norms.std())
                row[f"{name}_available"] = True
            row["num_heads"] = Q.shape[1] if Q is not None else (K.shape[1] if K is not None else None)
            qkv_rows.append(row)

        # ---- Attention compact (per layer, this step) ----
        if A is not None and O is not None:
            for layer in range(n_layers):
                attn_row = A[layer, :, pos, :pos + 1]
                concentration = float(attn_row.max(axis=-1).mean())
                row_sums = attn_row.sum(axis=-1, keepdims=True)
                probs = attn_row / (row_sums + 1e-9)
                ent = -(probs * np.log(probs + 1e-9)).sum(axis=-1)
                ent_mean, ent_std = float(ent.mean()), float(ent.std())
                topk = min(5, attn_row.shape[-1])
                topk_mass = float(np.sort(attn_row, axis=-1)[:, -topk:].sum(axis=-1).mean())
                o_t = O[layer, pos]
                o_norm = float(np.linalg.norm(o_t))
                o_delta = o_cos_prev = None
                if layer - 1 >= 0:
                    o_prev = O[layer - 1, pos]
                    o_prev_norm = float(np.linalg.norm(o_prev))
                    o_delta = o_norm - o_prev_norm
                    o_cos_prev = float(np.dot(o_t, o_prev) / (o_norm * o_prev_norm + 1e-9))
                attn_rows.append({
                    "model_id": model_id, "run_id": run_id, "prompt_id": prompt_id, "prompt_family": family,
                    "seed": seed, "token_step": step, "layer_index": layer, "num_layers": n_layers,
                    "attention_entropy_mean": ent_mean, "attention_entropy_std": ent_std,
                    "attention_max_mean": concentration, "attention_concentration_top5": topk_mass,
                    "attention_output_norm": o_norm, "attention_output_delta_norm": o_delta,
                    "attention_output_cosine_prev_layer": o_cos_prev, "num_heads": A.shape[1],
                })

        # ---- readout FINAL_NATIVE (this step) ----
        if logits is not None and step < logits.shape[0]:
            target_id = int(token_ids[pos + 1]) if pos + 1 < len(token_ids) else None
            if target_id is not None:
                stats = top_stats_from_logits(logits[step], target_id)
                dr = (model_wide_by_step or {}).get(step, {})
                readout_rows.append({
                    "model_id": model_id, "run_id": run_id, "prompt_id": prompt_id, "prompt_family": family,
                    "seed": seed, "token_step": step, "layer_index": n_layers, "num_layers": n_layers,
                    "readout_type": "FINAL_NATIVE", **stats,
                    "commitment_score": dr.get("commitment_score"), "branching_factor": dr.get("branching_factor"),
                    "concentration": dr.get("concentration"), "collapse_score": dr.get("collapse_score"),
                })
                regime_rows.append({
                    "model_id": model_id, "run_id": run_id, "prompt_id": prompt_id, "prompt_family": family,
                    "seed": seed, "token_step": step,
                    "entropy": dr.get("entropy", stats["entropy"]), "top1_prob": dr.get("top1_prob", stats["top1_prob"]),
                    "top2_prob": dr.get("top2_prob"), "top2_gap": dr.get("top2_gap"),
                    "collapse_score": dr.get("collapse_score"), "commitment_score": dr.get("commitment_score"),
                    "concentration": dr.get("concentration"), "branching_factor": dr.get("branching_factor"),
                    "kappa_G_input": dr.get("kappa_G_input"), "kappa_D_input": dr.get("kappa_D_input"),
                    "kappa_sync_input": dr.get("kappa_sync_input"),
                    "source": "MODEL_WIDE_PRECOMPUTED" if dr else "NOT_AVAILABLE",
                })

        # ---- future readout t+2 / t+3 (descriptive only) ----
        if logits is not None:
            for k in (2, 3):
                fstep = step + k
                fpos = prompt_len + fstep
                if fstep < logits.shape[0] and fpos + 1 < len(token_ids):
                    tid = int(token_ids[fpos + 1])
                    row = logits[fstep].astype(np.float64)
                    p = np.exp(row - row.max()); p = p / p.sum()
                    order = np.argsort(-row)
                    rank = int(np.where(order == tid)[0][0])
                    readout_rows.append({
                        "model_id": model_id, "run_id": run_id, "prompt_id": prompt_id, "prompt_family": family,
                        "seed": seed, "token_step": step, "layer_index": n_layers, "num_layers": n_layers,
                        "readout_type": f"FUTURE_T_PLUS_{k}",
                        "target_rank": rank, "target_logit": float(row[tid]), "target_probability": float(p[tid]),
                    })

    # ---- CROSS_STEP geometry (same layer, consecutive steps) ----
    cross_step_rows = []
    if prompt_len is not None:
        for layer in range(n_layers + 1):
            for step in range(1, n_steps):
                pos, pos_prev = prompt_len + step, prompt_len + step - 1
                if pos >= seq_len:
                    continue
                h_t, h_prev = H[layer, pos], H[layer, pos_prev]
                nt, np_ = float(np.linalg.norm(h_t)), float(np.linalg.norm(h_prev))
                cos_step = float(np.dot(h_t, h_prev) / (nt * np_ + 1e-9))
                delta_step = nt - np_
                curvature_step = None
                if step >= 2:
                    pos_prev2 = prompt_len + step - 2
                    if pos_prev2 >= 0:
                        h_prev2 = H[layer, pos_prev2]
                        d1, d0 = h_t - h_prev, h_prev - h_prev2
                        n1, n0 = np.linalg.norm(d1), np.linalg.norm(d0)
                        curvature_step = 1 - float(np.dot(d0, d1) / (n0 * n1 + 1e-9))
                cross_step_rows.append({
                    "model_id": model_id, "run_id": run_id, "prompt_id": prompt_id, "prompt_family": family,
                    "seed": seed, "token_step": step, "layer_index": layer, "num_layers": n_layers,
                    "cosine_step_to_previous_step": cos_step, "delta_norm_cross_step": delta_step,
                    "curvature_cross_step": curvature_step,
                })

    npz.close()
    return {
        "layer_rows": layer_rows, "attn_rows": attn_rows, "qkv_rows": qkv_rows,
        "readout_rows": readout_rows, "regime_rows": regime_rows, "cross_step_rows": cross_step_rows,
        "nan_inf_count": nan_inf_count, "n_layers": n_layers, "n_steps": n_steps,
    }, None


def load_model_wide_by_step(run):
    if not run["model_wide_path"]:
        return {}
    p = SRC / run["model_wide_path"]
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
    except Exception:
        return {}
    return {m["step"]: m for m in d.get("probability_metrics", [])}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "extended"
    models = sorted(EXTENDED_MODELS) if which == "extended" else sorted(CORE_MODELS)
    log(f"Starting extraction for {which}: {len(models)} models")

    # Explicit schemas (not inferred from the first batch) -- required because some models
    # (e.g. gpt-neo-125m, bloomz-560m) have Q entirely unavailable (all-None columns), which
    # would otherwise make pyarrow infer a `null` column type from whichever model is processed
    # first, then fail to accept a later model's real float values for that same column.
    STR = pa.string()
    F64 = pa.float64()
    I64 = pa.int64()
    BOOL = pa.bool_()
    SCHEMAS = {
        "layer": pa.schema([
            ("model_id", STR), ("run_id", STR), ("prompt_id", STR), ("prompt_family", STR), ("seed", STR),
            ("token_step", I64), ("layer_index", I64), ("num_layers", I64), ("normalized_depth", F64),
            ("hidden_norm", F64), ("hidden_mean", F64), ("hidden_std", F64), ("delta_norm_cross_layer", F64),
            ("cosine_to_previous_layer", F64), ("curvature_cross_layer", F64),
            ("trajectory_displacement_from_initial", F64), ("cosine_to_initial_layer", F64), ("cosine_to_final_layer", F64),
        ]),
        "attn": pa.schema([
            ("model_id", STR), ("run_id", STR), ("prompt_id", STR), ("prompt_family", STR), ("seed", STR),
            ("token_step", I64), ("layer_index", I64), ("num_layers", I64),
            ("attention_entropy_mean", F64), ("attention_entropy_std", F64), ("attention_max_mean", F64),
            ("attention_concentration_top5", F64), ("attention_output_norm", F64), ("attention_output_delta_norm", F64),
            ("attention_output_cosine_prev_layer", F64), ("num_heads", I64),
        ]),
        "qkv": pa.schema([
            ("model_id", STR), ("run_id", STR), ("prompt_id", STR), ("prompt_family", STR), ("seed", STR),
            ("token_step", I64), ("layer_index", I64), ("num_layers", I64),
            ("Q_norm_mean", F64), ("Q_norm_std", F64), ("Q_available", BOOL),
            ("K_norm_mean", F64), ("K_norm_std", F64), ("K_available", BOOL),
            ("V_norm_mean", F64), ("V_norm_std", F64), ("V_available", BOOL), ("num_heads", I64),
        ]),
        "readout": pa.schema([
            ("model_id", STR), ("run_id", STR), ("prompt_id", STR), ("prompt_family", STR), ("seed", STR),
            ("token_step", I64), ("layer_index", I64), ("num_layers", I64), ("readout_type", STR),
            ("top1_token_id", I64), ("top1_logit", F64), ("top1_prob", F64),
            ("top2_token_id", I64), ("top2_logit", F64), ("top2_prob", F64),
            ("target_logit", F64), ("target_probability", F64), ("target_rank", I64), ("is_target_top1", BOOL),
            ("entropy", F64), ("commitment_score", F64), ("branching_factor", F64), ("concentration", F64), ("collapse_score", F64),
        ]),
        "regime": pa.schema([
            ("model_id", STR), ("run_id", STR), ("prompt_id", STR), ("prompt_family", STR), ("seed", STR),
            ("token_step", I64), ("entropy", F64), ("top1_prob", F64), ("top2_prob", F64), ("top2_gap", F64),
            ("collapse_score", F64), ("commitment_score", F64), ("concentration", F64), ("branching_factor", F64),
            ("kappa_G_input", F64), ("kappa_D_input", F64), ("kappa_sync_input", F64), ("source", STR),
        ]),
        "cross_step": pa.schema([
            ("model_id", STR), ("run_id", STR), ("prompt_id", STR), ("prompt_family", STR), ("seed", STR),
            ("token_step", I64), ("layer_index", I64), ("num_layers", I64),
            ("cosine_step_to_previous_step", F64), ("delta_norm_cross_step", F64), ("curvature_cross_step", F64),
        ]),
    }

    writers = {}

    def get_writer(name, rows):
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=SCHEMAS[name])
        out_path = OUT / f"_part_{which}_{name}.parquet"
        if name not in writers:
            writers[name] = pq.ParquetWriter(out_path, SCHEMAS[name])
        writers[name].write_table(table)

    total_rows = {"layer": 0, "attn": 0, "qkv": 0, "readout": 0, "regime": 0, "cross_step": 0}
    n_runs_ok = n_runs_fail = 0
    fail_log = []

    for model_id in models:
        runs = runs_by_model.get(model_id, [])
        log(f"=== {model_id}: {len(runs)} runs ===")
        for run in runs:
            mw = load_model_wide_by_step(run)
            result, err = process_run(model_id, run, model_wide_by_step=mw)
            if err:
                n_runs_fail += 1
                fail_log.append({"model_id": model_id, "run_id": run["run_id"], "error": err})
                log(f"FAIL {run['run_id']}: {err}")
                continue
            n_runs_ok += 1
            get_writer("layer", result["layer_rows"]); total_rows["layer"] += len(result["layer_rows"])
            get_writer("attn", result["attn_rows"]); total_rows["attn"] += len(result["attn_rows"])
            get_writer("qkv", result["qkv_rows"]); total_rows["qkv"] += len(result["qkv_rows"])
            get_writer("readout", result["readout_rows"]); total_rows["readout"] += len(result["readout_rows"])
            get_writer("regime", result["regime_rows"]); total_rows["regime"] += len(result["regime_rows"])
            get_writer("cross_step", result["cross_step_rows"]); total_rows["cross_step"] += len(result["cross_step_rows"])
        gc.collect()

    for w in writers.values():
        w.close()

    json.dump({"total_rows": total_rows, "n_runs_ok": n_runs_ok, "n_runs_fail": n_runs_fail, "fail_log": fail_log},
               open(SCRATCH / f"_extraction_summary_{which}.json", "w"), indent=2)
    log(f"DONE {which}: ok={n_runs_ok} fail={n_runs_fail} rows={total_rows}")
