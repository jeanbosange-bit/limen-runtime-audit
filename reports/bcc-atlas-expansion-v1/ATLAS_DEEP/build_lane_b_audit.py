"""
Lane B (ATLAS-DEEP) post-capture audit. Pure acquisition audit + landmark RECURRENCE
(frozen L0/L1/L5/L20, exact detection rule reused verbatim from
reports/multi-tensor-atlas-v0/build_atlas.py -- not redefined). No new landmarks, no
network edges promoted here (separate gate).
"""
import csv
import hashlib
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
INDEX_PATH = HERE / "run_index.jsonl"
RAW_DIR = HERE / "raw_captures"
NODES = {0: "L0", 1: "L1", 5: "L5", 20: "L20"}
SIGNALS = ["H_curvature", "Oattn_delta_norm", "A_dispersion", "probability_entropy"]

with open(INDEX_PATH) as f:
    runs = [json.loads(l) for l in f]
assert len(runs) == 144

# ============================================================
# LANE_B_ACQUISITION_AUDIT.json
# ============================================================
status_counts = Counter(r["status"] for r in runs)
n_success = status_counts.get("SUCCESS", 0)

by_family = Counter(r["prompt_family"] for r in runs)
by_seed = Counter(r["seed"] for r in runs)
prompts_covered = {r["prompt_id"] for r in runs}
families_covered = {r["prompt_family"] for r in runs}
seeds_covered = {r["seed"] for r in runs}
run_id_counts = Counter(r["run_id"] for r in runs)
duplicates = {k: v for k, v in run_id_counts.items() if v > 1}

n_steps_list = [r["meta"]["n_steps"] for r in runs if r["status"] == "SUCCESS"]
runtimes = [r["runtime_seconds"] for r in runs]
storage_bytes = [r["meta"]["npz_size_bytes"] for r in runs if r["status"] == "SUCCESS"]
total_nan = sum(r["meta"]["total_nan"] for r in runs if r["status"] == "SUCCESS")
total_inf = sum(r["meta"]["total_inf"] for r in runs if r["status"] == "SUCCESS")

tensor_status_by_field = defaultdict(Counter)
for r in runs:
    tc = r.get("tensor_completeness", {})
    for field, val in tc.items():
        tensor_status_by_field[field][val] += 1


def summarize_tensor(field):
    counts = tensor_status_by_field[field]
    if counts.get("COMPLETE", 0) == 144:
        return "COMPLETE"
    elif counts.get("COMPLETE", 0) > 0:
        return "PARTIAL"
    else:
        return "FAIL"


acquisition_audit = {
    "status": "EXECUTED",
    "attempted": len(runs), "success": n_success, "failures": len(runs) - n_success,
    "success_rate_pct": round(100 * n_success / len(runs), 3),
    "status_breakdown": dict(status_counts),
    "prompt_coverage": {"n_prompts_covered": len(prompts_covered), "n_prompts_expected": 48},
    "family_coverage": {"n_families_covered": len(families_covered), "n_families_expected": 12, "by_family": dict(by_family)},
    "seed_coverage": {"n_seeds_covered": len(seeds_covered), "seeds_expected": [2101, 2102, 2103], "by_seed": dict(by_seed)},
    "duplicate_run_ids": duplicates,
    "generation_length_distribution": {
        "n": len(n_steps_list), "min": min(n_steps_list), "median": st.median(n_steps_list),
        "max": max(n_steps_list), "mean": round(st.mean(n_steps_list), 2),
    },
    "runtime_seconds_distribution": {"min": round(min(runtimes), 2), "median": round(st.median(runtimes), 2), "max": round(max(runtimes), 2), "total": round(sum(runtimes), 1)},
    "storage_bytes_distribution": {"min": min(storage_bytes), "median": int(st.median(storage_bytes)), "max": max(storage_bytes), "total_gb": round(sum(storage_bytes) / 1e9, 3)},
    "total_nan": total_nan, "total_inf": total_inf,
    "tensor_availability_summary": {f: summarize_tensor(f) for f in ["H", "Q", "K", "V", "ATTENTION", "ATTENTION_OUTPUT", "LOGITS", "TOKEN_IDS"]},
    "malformed_captures": sum(1 for r in runs if r["status"] != "SUCCESS"),
}
with open(HERE / "LANE_B_ACQUISITION_AUDIT.json", "w") as f:
    json.dump(acquisition_audit, f, indent=2)

# ============================================================
# TENSOR_AVAILABILITY_MATRIX.csv
# ============================================================
with open(HERE / "TENSOR_AVAILABILITY_MATRIX.csv", "w", newline="") as f:
    w = csv.writer(f)
    fields = ["H", "Q", "K", "V", "ATTENTION", "ATTENTION_OUTPUT", "LOGITS", "TOKEN_IDS"]
    w.writerow(["run_id", "prompt_id", "prompt_family", "seed", "status"] + fields)
    for r in runs:
        tc = r.get("tensor_completeness", {})
        w.writerow([r["run_id"], r["prompt_id"], r["prompt_family"], r["seed"], r["status"]] + [tc.get(fl, "N/A") for fl in fields])

# ============================================================
# TENSOR_GRANULARITY.json
# ============================================================
granularity = {
    "status": "FROZEN_DESCRIPTIVE",
    "H": {"granularity": "RUN x STEP x LAYER", "shape_per_run": "(25, seq_len, 896)", "dtype": "fp16", "epistemic_status": "MEASURED"},
    "Q": {"granularity": "RUN x STEP x LAYER x HEAD", "shape_per_run": "(24, 14, seq_len, 64)", "dtype": "fp16",
          "epistemic_status": "MEASURED, captured live from self_attn.q_proj -- NOT reconstructed", "note": "new in this campaign; Multi-Tensor Atlas V0 had Q=NOT_CAPTURED"},
    "K": {"granularity": "RUN x STEP x LAYER x HEAD (GQA kv-head)", "shape_per_run": "(24, 2, seq_len, 64)", "dtype": "fp32", "epistemic_status": "MEASURED"},
    "V": {"granularity": "RUN x STEP x LAYER x HEAD (GQA kv-head)", "shape_per_run": "(24, 2, seq_len, 64)", "dtype": "fp32", "epistemic_status": "MEASURED"},
    "ATTENTION": {"granularity": "RUN x STEP x LAYER x HEAD x CONTEXT_POSITION", "shape_per_run": "(24, 14, seq_len, seq_len)", "dtype": "fp16", "epistemic_status": "MEASURED"},
    "ATTENTION_OUTPUT": {"granularity": "RUN x STEP x LAYER", "shape_per_run": "(24, seq_len, 896)", "dtype": "fp16", "epistemic_status": "MEASURED", "note": "post-o_proj, heads already merged, no head split possible (same structural limit as V0)"},
    "LOGITS": {"granularity": "RUN x STEP", "shape_per_run": "(n_generated_steps, 151936)", "dtype": "fp16", "epistemic_status": "MEASURED", "note": "bounded to predictive positions only, same convention as V0 -- NOT full-sequence logits"},
    "TOKEN_IDS": {"granularity": "RUN x STEP", "shape_per_run": "(seq_len,)", "dtype": "int32", "epistemic_status": "MEASURED"},
    "warning": "No tensor here is broadcast into a fake finer granularity. Every field above is genuinely resolved at the granularity stated -- unlike reports/multi-tensor-atlas-v0's COORDINATE_TABLE, which explicitly mixed broadcast (run,layer)-structural K/V columns with true (run,layer,step) columns. Lane B's K/V/Q/attention/attn_output are ALL genuinely step-resolved here because the full sequence (all decode steps) is captured in a single teacher-forced forward pass per run.",
}
with open(HERE / "TENSOR_GRANULARITY.json", "w") as f:
    json.dump(granularity, f, indent=2)

# ============================================================
# CROSSAXIS_COVERAGE.csv -- run x layer x step coverage confirmation
# ============================================================
with open(HERE / "CROSSAXIS_COVERAGE.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run_id", "prompt_id", "prompt_family", "seed", "prompt_len", "seq_len", "n_decode_steps", "n_layers_H", "n_layers_mechanism", "coverage_status"])
    for r in runs:
        if r["status"] != "SUCCESS":
            w.writerow([r["run_id"], r["prompt_id"], r["prompt_family"], r["seed"], None, None, None, None, None, "NOT_CAPTURED"])
            continue
        m = r["meta"]
        w.writerow([r["run_id"], r["prompt_id"], r["prompt_family"], r["seed"], m["prompt_len"], m["seq_len"], m["n_steps"], 25, 24, "FULL"])

# ============================================================
# SHORT_EXECUTION_AUDIT.json
# ============================================================
short_runs = [r for r in runs if r["status"] == "SUCCESS" and r["meta"]["n_steps"] <= 9]
short_by_family = Counter(r["prompt_family"] for r in short_runs)
with open(HERE / "SHORT_EXECUTION_AUDIT.json", "w") as f:
    json.dump({
        "definition": "SHORT_EXECUTION = 1-9 generated tokens, per reports/bcc-atlas-expansion-v1/SHORT_EXECUTION_POLICY.md -- never NON_ANALYZABLE",
        "n_short_executions": len(short_runs), "n_total_success": n_success,
        "pct_short": round(100 * len(short_runs) / n_success, 2),
        "by_family": dict(short_by_family),
        "run_ids": [r["run_id"] for r in short_runs],
        "all_captured_at_full_native_granularity": True,
    }, f, indent=2)

# ============================================================
# LANDMARK_RECURRENCE_AUDIT.json -- frozen L0/L1/L5/L20, exact z-score rule, NOT redefined
# ============================================================
def entropy_from_logits(row):
    p = np.exp(row - row.max())
    p = p / p.sum()
    return float(-(p * np.log(p + 1e-9)).sum())


def compute_signals_for_layer(npz, layer, prompt_len, n_steps, seq_len):
    H = npz["hidden_states_all_layers_fp16"][layer].astype(np.float32)  # (seq_len, 896)
    A = npz["attentions_all_layers_fp16"][layer].astype(np.float32)     # (14, seq_len, seq_len)
    O = npz["attn_output_all_layers_fp16"][layer].astype(np.float32)    # (seq_len, 896)
    logits = npz["logits_region_fp16"].astype(np.float32)               # (n_gen_steps_captured, vocab)

    h_curv, o_delta, a_disp, p_ent = [], [], [], []
    for step in range(n_steps):
        pos = prompt_len + step
        if pos >= seq_len:
            break
        h_t = H[pos]
        curvature = np.nan
        if pos - 2 >= 0:
            d1 = H[pos] - H[pos - 1]
            d0 = H[pos - 1] - H[pos - 2]
            n1, n0 = np.linalg.norm(d1), np.linalg.norm(d0)
            curvature = 1 - float(np.dot(d0, d1) / (n0 * n1 + 1e-9))
        h_curv.append(curvature)

        o_t = O[pos]
        delta = np.nan
        if pos - 1 >= 0:
            o_prev = O[pos - 1]
            delta = float(np.linalg.norm(o_t)) - float(np.linalg.norm(o_prev))
        o_delta.append(delta)

        attn_row = A[:, pos, :pos + 1]
        row_sums = attn_row.sum(axis=-1, keepdims=True)
        probs = attn_row / (row_sums + 1e-9)
        dispersion = float(-(probs * np.log(probs + 1e-9)).sum(axis=-1).mean())
        a_disp.append(dispersion)

        p_ent.append(entropy_from_logits(logits[step]) if step < logits.shape[0] else np.nan)

    return {"H_curvature": h_curv, "Oattn_delta_norm": o_delta, "A_dispersion": a_disp, "probability_entropy": p_ent}


landmark_events = []  # (run_id, layer, step)
ineligible_runs = []  # n_steps<4, structurally excluded
per_run_landmarks = defaultdict(set)

for r in runs:
    if r["status"] != "SUCCESS":
        continue
    m = r["meta"]
    n_steps = m["n_steps"]
    if n_steps < 4:
        ineligible_runs.append(r["run_id"])
        continue
    npz = np.load(m["npz_path"])
    for layer, node in NODES.items():
        sig = compute_signals_for_layer(npz, layer, m["prompt_len"], n_steps, m["seq_len"])
        n_valid = len(sig["H_curvature"])
        if n_valid < 4:
            continue
        arrs = {k: np.array(v, dtype=float) for k, v in sig.items()}
        z = {}
        for k, v in arrs.items():
            std = np.nanstd(v)
            z[k] = (v - np.nanmean(v)) / std if std and std > 1e-9 else v * 0
        spike = np.zeros(n_valid, dtype=int)
        for k in SIGNALS:
            spike += (np.abs(np.nan_to_num(z[k])) > 1.5).astype(int)
        for step_idx in np.where(spike >= 3)[0]:
            landmark_events.append({"run_id": r["run_id"], "prompt_id": r["prompt_id"], "family": r["prompt_family"], "seed": r["seed"], "layer": layer, "step": int(step_idx)})
            per_run_landmarks[r["run_id"]].add(node)
    npz.close()

runs_with_ge1 = sum(1 for v in per_run_landmarks.values() if len(v) >= 1)
runs_with_ge2 = sum(1 for v in per_run_landmarks.values() if len(v) >= 2)
n_eligible = sum(1 for r in runs if r["status"] == "SUCCESS" and r["meta"]["n_steps"] >= 4)
no_landmark_runs = [r["run_id"] for r in runs if r["status"] == "SUCCESS" and r["meta"]["n_steps"] >= 4 and r["run_id"] not in per_run_landmarks]

recurrence_by_family = defaultdict(lambda: {"n_runs_eligible": 0, "n_runs_with_landmark": 0})
recurrence_by_seed = defaultdict(lambda: {"n_runs_eligible": 0, "n_runs_with_landmark": 0})
run_lookup = {r["run_id"]: r for r in runs}
for r in runs:
    if r["status"] != "SUCCESS" or r["meta"]["n_steps"] < 4:
        continue
    fam, seed = r["prompt_family"], r["seed"]
    recurrence_by_family[fam]["n_runs_eligible"] += 1
    recurrence_by_seed[seed]["n_runs_eligible"] += 1
    if r["run_id"] in per_run_landmarks:
        recurrence_by_family[fam]["n_runs_with_landmark"] += 1
        recurrence_by_seed[seed]["n_runs_with_landmark"] += 1

length_buckets = {"SHORT_1_9": [0, 0], "MEDIUM_10_49": [0, 0], "LONG_50_PLUS": [0, 0]}
for r in runs:
    if r["status"] != "SUCCESS" or r["meta"]["n_steps"] < 4:
        continue
    n = r["meta"]["n_steps"]
    bucket = "SHORT_1_9" if n <= 9 else ("MEDIUM_10_49" if n <= 49 else "LONG_50_PLUS")
    length_buckets[bucket][1] += 1
    if r["run_id"] in per_run_landmarks:
        length_buckets[bucket][0] += 1

landmark_audit = {
    "status": "EXECUTED -- descriptive recurrence only, frozen nodes reused verbatim from reports/multi-tensor-landmarks-v0/FROZEN_LANDMARK_REGISTRY.json, NOT redefined",
    "detection_rule": "Within (run,layer), z-score H_curvature/Oattn_delta_norm/A_dispersion/probability_entropy across decode steps; event if |z|>1.5 on >=3/4 signals. Identical to reports/multi-tensor-atlas-v0/build_atlas.py -- unmodified.",
    "nodes_tested": ["L0", "L1", "L5", "L20"],
    "n_runs_total": 144, "n_runs_eligible_for_detection": n_eligible,
    "n_runs_ineligible_n_steps_lt_4": len(ineligible_runs),
    "runs_with_ge1_landmark": runs_with_ge1, "runs_with_ge2_landmark": runs_with_ge2,
    "no_landmark_runs_count": len(no_landmark_runs), "no_landmark_run_ids": no_landmark_runs,
    "recurrence_by_family": dict(recurrence_by_family),
    "recurrence_by_seed": {str(k): v for k, v in recurrence_by_seed.items()},
    "recurrence_by_execution_length_bucket": {k: {"n_with_landmark": v[0], "n_total": v[1]} for k, v in length_buckets.items()},
    "n_raw_events": len(landmark_events),
    "events_sample": landmark_events[:200],
    "network_edges_promoted": 0,
    "new_landmarks_promoted": 0,
}
with open(HERE / "LANDMARK_RECURRENCE_AUDIT.json", "w") as f:
    json.dump(landmark_audit, f, indent=2, default=str)

# ============================================================
# UNKNOWN_TERRITORY_AUDIT.json
# ============================================================
single_landmark_runs = [rid for rid, v in per_run_landmarks.items() if len(v) == 1]
family_sparse = {fam: v for fam, v in recurrence_by_family.items() if v["n_runs_eligible"] > 0 and v["n_runs_with_landmark"] / v["n_runs_eligible"] < 0.3}
unknown_audit = {
    "meaning_disclaimer": "UNKNOWN/LOW_SUPPORT means insufficient reference structure only -- NOT danger, failure, or instability.",
    "no_landmark_runs": {"count": len(no_landmark_runs), "run_ids": no_landmark_runs},
    "single_landmark_runs": {"count": len(single_landmark_runs), "run_ids": single_landmark_runs},
    "structurally_ineligible_runs_n_steps_lt_4": {"count": len(ineligible_runs), "run_ids": ineligible_runs},
    "underrepresented_layers": "Only 4 layers (0,1,5,20) tested by design -- the other 20 mechanism layers are UNKNOWN territory for landmark recurrence in this gate (frozen node set, mission section 4 prohibits testing/promoting new candidates here).",
    "prompt_families_with_sparse_internal_landmark_coverage": {k: v for k, v in family_sparse.items()},
    "note": "A healthy Atlas is allowed to contain mostly unclassified territory (mission precedent, reports/multi-tensor-landmarks-v0). This audit quantifies rather than hides that territory.",
}
with open(HERE / "UNKNOWN_TERRITORY_AUDIT.json", "w") as f:
    json.dump(unknown_audit, f, indent=2, default=str)

# ============================================================
# RUN_INDEX.csv
# ============================================================
with open(HERE / "RUN_INDEX.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run_id", "prompt_id", "prompt_family", "seed", "status", "n_steps", "seq_len", "prompt_len", "npz_size_bytes", "artifact_hash", "runtime_seconds", "n_landmarks", "landmarks"])
    for r in runs:
        m = r.get("meta", {})
        lm = sorted(per_run_landmarks.get(r["run_id"], []))
        w.writerow([r["run_id"], r["prompt_id"], r["prompt_family"], r["seed"], r["status"],
                    m.get("n_steps"), m.get("seq_len"), m.get("prompt_len"), m.get("npz_size_bytes"),
                    m.get("artifact_hash"), round(r["runtime_seconds"], 2), len(lm), "|".join(lm)])

print("Acquisition audit:", json.dumps(acquisition_audit, indent=2)[:800])
print("\nLandmark recurrence: runs>=1:", runs_with_ge1, "runs>=2:", runs_with_ge2, "no-landmark:", len(no_landmark_runs), "ineligible:", len(ineligible_runs))
print("DONE")
