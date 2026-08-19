"""
Atlas Coverage Analysis V1 -- downstream analysis. Reloads the 144 Lane B npz files
(read-only) once, recomputing the full per-run per-landmark-layer descriptor time
series (same formulas as build_coverage_analysis.py, frozen in FROZEN_ANALYSIS_PROTOCOL.md),
and produces every remaining deliverable in one pass.
"""
import csv
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
LANE_B = HERE.parent / "bcc-atlas-expansion-v1" / "ATLAS_DEEP"
NODES = {0: "L0", 1: "L1", 5: "L5", 20: "L20"}
SIGNALS = ["H_curvature", "Oattn_delta_norm", "A_dispersion", "probability_entropy"]

with open(LANE_B / "run_index.jsonl") as f:
    runs = [json.loads(l) for l in f]
with open(HERE / "SUPPORT_THRESHOLD_FREEZE.json") as f:
    THRESH = json.load(f)

DESCRIPTOR_KEYS = ["H_norm", "H_delta_norm", "H_cos", "H_curvature",
                   "Q_norm", "Q_delta_norm", "Q_cos",
                   "K_norm", "K_delta_norm", "K_cos",
                   "V_norm", "V_delta_norm", "V_cos",
                   "A_dispersion", "A_concentration", "A_max_mass",
                   "Oattn_norm", "Oattn_delta_norm", "Oattn_cos",
                   "probability_entropy", "probability_top2_gap", "logit_norm"]


def entropy_from_logits(row):
    p = np.exp(row - row.max()); p = p / p.sum()
    return float(-(p * np.log(p + 1e-9)).sum())


def top2_gap_from_logits(row):
    p = np.exp(row - row.max()); p = p / p.sum()
    idx = np.argsort(-row)[:2]
    return float(p[idx[0]] - p[idx[1]])


def safe_cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb + 1e-9))


run_layer_series = {}
run_landmarks = defaultdict(dict)
run_eligibility = {}
events = []
run_meta_by_id = {r["run_id"]: r for r in runs}

print("Recomputing full descriptor series for all runs...", flush=True)
n_done = 0
for r in runs:
    m = r["meta"]; n_steps = m["n_steps"]; run_id = r["run_id"]
    if n_steps < 4:
        run_eligibility[run_id] = "LANDMARK_NOT_EVALUABLE_SHORT_EXECUTION"
        n_done += 1
        continue
    run_eligibility[run_id] = "ELIGIBLE"
    npz = np.load(m["npz_path"])
    prompt_len, seq_len = m["prompt_len"], m["seq_len"]
    H_all = npz["hidden_states_all_layers_fp16"].astype(np.float32)
    Q_all = npz["q_all_layers_fp16"].astype(np.float32)
    K_all = npz["k_all_layers_fp32"]
    V_all = npz["v_all_layers_fp32"]
    A_all = npz["attentions_all_layers_fp16"].astype(np.float32)
    O_all = npz["attn_output_all_layers_fp16"].astype(np.float32)
    logits = npz["logits_region_fp16"].astype(np.float32)

    for layer, node in NODES.items():
        H = H_all[layer]
        Q = Q_all[layer].transpose(1, 0, 2).reshape(seq_len, -1)
        K = K_all[layer].transpose(1, 0, 2).reshape(seq_len, -1)
        V = V_all[layer].transpose(1, 0, 2).reshape(seq_len, -1)
        A = A_all[layer]
        O = O_all[layer]
        series = {k: [] for k in DESCRIPTOR_KEYS}
        for step in range(n_steps):
            pos = prompt_len + step
            if pos >= seq_len:
                break
            series["H_norm"].append(float(np.linalg.norm(H[pos])))
            series["Q_norm"].append(float(np.linalg.norm(Q[pos])))
            series["K_norm"].append(float(np.linalg.norm(K[pos])))
            series["V_norm"].append(float(np.linalg.norm(V[pos])))
            series["Oattn_norm"].append(float(np.linalg.norm(O[pos])))
            for name, arr in [("H", H), ("Q", Q), ("K", K), ("V", V), ("Oattn", O)]:
                dn, cs = np.nan, np.nan
                if pos - 1 >= 0:
                    dn = series[f"{name}_norm"][-1] - float(np.linalg.norm(arr[pos - 1]))
                    cs = safe_cos(arr[pos - 1], arr[pos])
                series[f"{name}_delta_norm"].append(dn)
                series[f"{name}_cos"].append(cs)
            curvature = np.nan
            if pos - 2 >= 0:
                d1 = H[pos] - H[pos - 1]; d0 = H[pos - 1] - H[pos - 2]
                curvature = 1 - safe_cos(d0, d1)
            series["H_curvature"].append(curvature)
            attn_row = A[:, pos, :pos + 1]
            row_sums = attn_row.sum(axis=-1, keepdims=True)
            probs = attn_row / (row_sums + 1e-9)
            series["A_dispersion"].append(float(-(probs * np.log(probs + 1e-9)).sum(axis=-1).mean()))
            series["A_concentration"].append(float(attn_row.max(axis=-1).mean()))
            series["A_max_mass"].append(float(attn_row.max()))
            if step < logits.shape[0]:
                row = logits[step]
                series["probability_entropy"].append(entropy_from_logits(row))
                series["probability_top2_gap"].append(top2_gap_from_logits(row))
                series["logit_norm"].append(float(np.linalg.norm(row)))
            else:
                series["probability_entropy"].append(np.nan)
                series["probability_top2_gap"].append(np.nan)
                series["logit_norm"].append(np.nan)
        run_layer_series[(run_id, layer)] = series

        n_valid = len(series["H_curvature"])
        if n_valid >= 4:
            z = {}
            for k in SIGNALS:
                v = np.array(series[k], dtype=float)
                std = np.nanstd(v)
                z[k] = (v - np.nanmean(v)) / std if std and std > 1e-9 else v * 0
            spike = np.zeros(n_valid, dtype=int)
            for k in SIGNALS:
                spike += (np.abs(np.nan_to_num(z[k])) > 1.5).astype(int)
            for step_idx in np.where(spike >= 3)[0]:
                events.append({"run_id": run_id, "node": node, "layer": layer, "step": int(step_idx),
                                "family": r["prompt_family"], "seed": r["seed"], "prompt_id": r["prompt_id"]})
                run_landmarks[run_id].setdefault(node, []).append(int(step_idx))
    npz.close()
    n_done += 1
    if n_done % 30 == 0:
        print(f"  {n_done}/144 runs processed", flush=True)

print(f"Recompute done: {len(events)} events, {len(run_landmarks)} runs with >=1 landmark", flush=True)
assert len(events) == 305 and len(run_landmarks) == 87, "recomputation mismatch vs pass1 -- STOP"

# ============================================================
# FAMILY_COVERAGE_MATRIX.csv
# ============================================================
family_stats = defaultdict(lambda: {"runs": 0, "eligible": 0, "short_noneval": 0, "ge1": 0, "ge2": 0,
                                      "no_landmark": 0, "lengths": [], "seeds": set(), "prompt_ids": set()})
for r in runs:
    fam = r["prompt_family"]; run_id = r["run_id"]; m = r["meta"]
    fs = family_stats[fam]
    fs["runs"] += 1
    fs["lengths"].append(m["n_steps"])
    fs["seeds"].add(r["seed"])
    fs["prompt_ids"].add(r["prompt_id"])
    if run_eligibility[run_id] == "LANDMARK_NOT_EVALUABLE_SHORT_EXECUTION":
        fs["short_noneval"] += 1
    else:
        fs["eligible"] += 1
        n_lm = len(run_landmarks.get(run_id, {}))
        if n_lm >= 1:
            fs["ge1"] += 1
        if n_lm >= 2:
            fs["ge2"] += 1
        if n_lm == 0:
            fs["no_landmark"] += 1

with open(HERE / "FAMILY_COVERAGE_MATRIX.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["family", "n_runs", "n_eligible", "n_short_noneval", "n_ge1_landmark", "n_ge2_landmark",
                "n_no_landmark", "min_len", "median_len", "max_len", "n_seeds", "n_prompt_ids", "layer_coverage_note"])
    for fam in sorted(family_stats):
        fs = family_stats[fam]
        w.writerow([fam, fs["runs"], fs["eligible"], fs["short_noneval"], fs["ge1"], fs["ge2"], fs["no_landmark"],
                    min(fs["lengths"]), st.median(fs["lengths"]), max(fs["lengths"]), len(fs["seeds"]), len(fs["prompt_ids"]),
                    "L0/L1/L5/L20 only (frozen node set) -- 20 other mechanism layers UNKNOWN by design"])

# ============================================================
# LANDMARK_RECURRENCE_BY_FAMILY.csv + LANDMARK_REPLICATION_RESULTS.json + COOCCURRENCE
# ============================================================
per_node_family = defaultdict(lambda: defaultdict(lambda: {"eligible": 0, "with_node": set()}))
per_node_seed = defaultdict(lambda: defaultdict(set))
per_node_total_runs = defaultdict(set)
n_eligible_total = sum(1 for r in runs if run_eligibility[r["run_id"]] == "ELIGIBLE")

for r in runs:
    run_id = r["run_id"]
    if run_eligibility[run_id] != "ELIGIBLE":
        continue
    fam, seed = r["prompt_family"], r["seed"]
    for node in NODES.values():
        per_node_family[node][fam]["eligible"] += 1
    lm = run_landmarks.get(run_id, {})
    for node in lm:
        per_node_family[node][fam]["with_node"].add(run_id)
        per_node_seed[node][fam].add(seed)  # placeholder, fixed below
        per_node_total_runs[node].add(run_id)

# fix per_node_seed properly: seeds where node observed
per_node_seed = defaultdict(set)
for r in runs:
    run_id = r["run_id"]
    lm = run_landmarks.get(run_id, {})
    for node in lm:
        per_node_seed[node].add(r["seed"])

with open(HERE / "LANDMARK_RECURRENCE_BY_FAMILY.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["node", "family", "n_eligible", "n_with_node", "recurrence_rate"])
    for node in NODES.values():
        for fam in sorted(family_stats):
            d = per_node_family[node][fam]
            rate = round(len(d["with_node"]) / d["eligible"], 4) if d["eligible"] else None
            w.writerow([node, fam, d["eligible"], len(d["with_node"]), rate])

replication_results = {}
for node in NODES.values():
    n_runs_with_node = len(per_node_total_runs[node])
    n_families = sum(1 for fam in family_stats if len(per_node_family[node][fam]["with_node"]) > 0)
    n_seeds = len(per_node_seed[node])
    rate = n_runs_with_node / n_eligible_total if n_eligible_total else 0
    if n_families >= 3 and n_seeds >= 2 and rate >= 0.15:
        verdict = "REPLICATED_ON_LANE_B"
    elif (n_families >= 2 or n_seeds >= 2) and rate >= 0.05:
        verdict = "PARTIAL_REPLICATION"
    else:
        verdict = "NOT_REPLICATED"
    replication_results[node] = {
        "n_runs_with_node": n_runs_with_node, "n_eligible_total": n_eligible_total,
        "recurrence_rate": round(rate, 4), "n_families_with_node": n_families, "n_seeds_with_node": n_seeds,
        "verdict": verdict, "canonical_status_unchanged": "REFERENCE_LANDMARK_SUPPORTED (reports/multi-tensor-landmarks-v0, not altered by this audit)",
    }
with open(HERE / "LANDMARK_REPLICATION_RESULTS.json", "w") as f:
    json.dump({"thresholds_used": THRESH["landmark_replication_verdict_thresholds"], "results": replication_results}, f, indent=2)

# co-occurrence matrix (4x4, count of runs where both nodes observed)
coocc = {a: {b: 0 for b in NODES.values()} for a in NODES.values()}
for run_id, lm in run_landmarks.items():
    present = list(lm.keys())
    for a in present:
        for b in present:
            coocc[a][b] += 1
with open(HERE / "LANDMARK_COOCCURRENCE_MATRIX.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([""] + list(NODES.values()))
    for a in NODES.values():
        w.writerow([a] + [coocc[a][b] for b in NODES.values()])

print("Family/replication/cooccurrence written", flush=True)

# ============================================================
# ROUTE_SIGNATURES.csv + ROUTE_FAMILY_MATRIX.csv (reuse RUN_ELIGIBILITY_AUDIT.csv's route col)
# ============================================================
with open(HERE / "RUN_ELIGIBILITY_AUDIT.csv") as f:
    elig_rows = list(csv.DictReader(f))

route_groups = defaultdict(list)
for row in elig_rows:
    route_groups[row["route_signature"]].append(row)

with open(HERE / "ROUTE_SIGNATURES.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["route_signature", "n_runs", "n_families", "n_seeds", "n_prompt_ids", "route_length", "category"])
    for sig, rows in sorted(route_groups.items(), key=lambda x: -len(x[1])):
        fams = {r["prompt_family"] for r in rows}
        seeds = {r["seed"] for r in rows}
        pids = {r["prompt_id"] for r in rows}
        length = 0 if sig == "NONE" else (sig.count(">") + 1)
        if sig == "NONE":
            cat = "NO_LANDMARK_ROUTE"
        elif length == 1:
            cat = "SINGLE_LANDMARK_ROUTE"
        elif len(rows) >= 3 and len(fams) >= 2:
            cat = "COMMON_ROUTE"
        elif len(rows) >= 2:
            cat = "LOW_SUPPORT_ROUTE"
        else:
            cat = "UNIQUE_ROUTE"
        w.writerow([sig, len(rows), len(fams), len(seeds), len(pids), length, cat])

with open(HERE / "ROUTE_FAMILY_MATRIX.csv", "w", newline="") as f:
    w = csv.writer(f)
    fams_sorted = sorted(family_stats)
    w.writerow(["route_signature"] + fams_sorted)
    for sig, rows in route_groups.items():
        fam_counts = Counter(r["prompt_family"] for r in rows)
        w.writerow([sig] + [fam_counts.get(fam, 0) for fam in fams_sorted])

n_cross_family_routes = sum(1 for sig, rows in route_groups.items() if sig != "NONE" and len({r["prompt_family"] for r in rows}) >= 2)
print(f"Route signatures written: {len(route_groups)} unique, {n_cross_family_routes} cross-family", flush=True)

# ============================================================
# LOCAL_LANDMARK_WINDOWS.csv + per-tensor _LOCAL_PROPERTIES.csv
# ============================================================
window_rows = []
tensor_prop_rows = {"H": [], "Q": [], "K": [], "V": [], "ATTENTION": [], "ATTENTION_OUTPUT": [], "LOGIT_PROBABILITY": []}

for ev in events:
    key = (ev["run_id"], ev["layer"])
    series = run_layer_series[key]
    n_valid = len(series["H_norm"])
    event_id = f"{ev['run_id']}_L{ev['layer']}_s{ev['step']}"
    for off in [-1, 0, 1]:
        idx = ev["step"] + off
        avail = 0 <= idx < n_valid
        row_common = {"event_id": event_id, "run_id": ev["run_id"], "node": ev["node"], "layer": ev["layer"],
                      "step": ev["step"], "offset": off, "family": ev["family"], "seed": ev["seed"], "available": avail}
        window_rows.append({**row_common, **{k: (series[k][idx] if avail else None) for k in DESCRIPTOR_KEYS}})
        tensor_prop_rows["H"].append({**row_common, "H_norm": series["H_norm"][idx] if avail else None,
                                        "H_delta_norm": series["H_delta_norm"][idx] if avail else None,
                                        "H_cos": series["H_cos"][idx] if avail else None,
                                        "H_curvature": series["H_curvature"][idx] if avail else None})
        tensor_prop_rows["Q"].append({**row_common, "Q_norm": series["Q_norm"][idx] if avail else None,
                                        "Q_delta_norm": series["Q_delta_norm"][idx] if avail else None,
                                        "Q_cos": series["Q_cos"][idx] if avail else None})
        tensor_prop_rows["K"].append({**row_common, "K_norm": series["K_norm"][idx] if avail else None,
                                        "K_delta_norm": series["K_delta_norm"][idx] if avail else None,
                                        "K_cos": series["K_cos"][idx] if avail else None})
        tensor_prop_rows["V"].append({**row_common, "V_norm": series["V_norm"][idx] if avail else None,
                                        "V_delta_norm": series["V_delta_norm"][idx] if avail else None,
                                        "V_cos": series["V_cos"][idx] if avail else None})
        tensor_prop_rows["ATTENTION"].append({**row_common, "A_dispersion": series["A_dispersion"][idx] if avail else None,
                                                "A_concentration": series["A_concentration"][idx] if avail else None,
                                                "A_max_mass": series["A_max_mass"][idx] if avail else None})
        tensor_prop_rows["ATTENTION_OUTPUT"].append({**row_common, "Oattn_norm": series["Oattn_norm"][idx] if avail else None,
                                                       "Oattn_delta_norm": series["Oattn_delta_norm"][idx] if avail else None,
                                                       "Oattn_cos": series["Oattn_cos"][idx] if avail else None})
        tensor_prop_rows["LOGIT_PROBABILITY"].append({**row_common, "probability_entropy": series["probability_entropy"][idx] if avail else None,
                                                        "probability_top2_gap": series["probability_top2_gap"][idx] if avail else None,
                                                        "logit_norm": series["logit_norm"][idx] if avail else None})

with open(HERE / "LOCAL_LANDMARK_WINDOWS.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(window_rows[0].keys()) if window_rows else [])
    w.writeheader()
    for row in window_rows:
        w.writerow(row)

for name, rows in tensor_prop_rows.items():
    fname = f"{name}_LOCAL_PROPERTIES.csv"
    with open(HERE / fname, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        for row in rows:
            w.writerow(row)

print(f"Local window CSVs written: {len(window_rows)} rows total", flush=True)

# ============================================================
# LANDMARK_VS_NONLANDMARK_ANALYSIS.json
# ============================================================
landmark_step_keys = set()
for ev in events:
    landmark_step_keys.add((ev["run_id"], ev["layer"], ev["step"]))

pop_a = defaultdict(list)  # landmark
pop_b = defaultdict(list)  # non-landmark eligible
for (run_id, layer), series in run_layer_series.items():
    n_valid = len(series["H_norm"])
    for step in range(n_valid):
        is_landmark = (run_id, layer, step) in landmark_step_keys
        target = pop_a if is_landmark else pop_b
        for k in DESCRIPTOR_KEYS:
            v = series[k][step]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                target[k].append(v)

COMPARISON_FIELDS = THRESH["landmark_vs_nonlandmark_statistical_plan"]["predeclared_comparisons"]
alpha_corr = THRESH["landmark_vs_nonlandmark_statistical_plan"]["n_comparisons"]
alpha_corrected = 0.05 / alpha_corr
comparison_results = {}
for field in COMPARISON_FIELDS:
    a = np.array(pop_a.get(field, []))
    b = np.array(pop_b.get(field, []))
    if len(a) < 5 or len(b) < 5:
        comparison_results[field] = {"n_a": len(a), "n_b": len(b), "status": "INSUFFICIENT_N"}
        continue
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    rank_biserial = 1 - (2 * stat) / (len(a) * len(b))
    comparison_results[field] = {
        "n_a_landmark": len(a), "n_b_nonlandmark": len(b),
        "median_a": round(float(np.median(a)), 5), "median_b": round(float(np.median(b)), 5),
        "U_statistic": float(stat), "p_raw": float(p),
        "p_significant_bonferroni": bool(p < alpha_corrected),
        "effect_size_rank_biserial": round(float(rank_biserial), 4),
    }
with open(HERE / "LANDMARK_VS_NONLANDMARK_ANALYSIS.json", "w") as f:
    json.dump({"predeclared_plan": THRESH["landmark_vs_nonlandmark_statistical_plan"], "alpha_corrected": alpha_corrected,
               "results": comparison_results,
               "interpretation_caveat": "Descriptive/exploratory only. No causal claim. Significance alone is never sufficient -- effect size (rank-biserial) reported alongside every p-value."}, f, indent=2)

print("Landmark vs non-landmark comparison written", flush=True)

with open(HERE / "PASS2_SERIES_SUMMARY_CACHE.json", "w") as f:
    json.dump({"note": "run_layer_series itself is large and not persisted verbatim; per-run/per-layer summary means below for downstream reuse",
               "per_run_layer_means": {f"{k[0]}|{k[1]}": {kk: (float(np.nanmean(vv)) if vv else None) for kk, vv in v.items()} for k, v in run_layer_series.items()}}, f)

print("DOWNSTREAM PART 1 DONE", flush=True)
