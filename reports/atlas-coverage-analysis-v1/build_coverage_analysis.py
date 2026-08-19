"""
Atlas Coverage Analysis V1 -- read-only analysis of Lane B's 144 real deep captures.
No new generation. No source artifact modified. Frozen landmarks (L0/L1/L5/L20) and
detection rule reused verbatim from reports/multi-tensor-landmarks-v0/.
"""
import csv
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LANE_B = HERE.parent / "bcc-atlas-expansion-v1" / "ATLAS_DEEP"
NODES = {0: "L0", 1: "L1", 5: "L5", 20: "L20"}
SIGNALS = ["H_curvature", "Oattn_delta_norm", "A_dispersion", "probability_entropy"]
LOCAL_WINDOW_OFFSETS = [-1, 0, 1]

with open(LANE_B / "run_index.jsonl") as f:
    runs = [json.loads(l) for l in f]
assert len(runs) == 144


def entropy_from_logits(row):
    p = np.exp(row - row.max())
    p = p / p.sum()
    return float(-(p * np.log(p + 1e-9)).sum())


def top2_gap_from_logits(row):
    p = np.exp(row - row.max())
    p = p / p.sum()
    idx = np.argsort(-row)[:2]
    return float(p[idx[0]] - p[idx[1]])


def safe_cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb + 1e-9))


# ============================================================
# PASS 1: per-run, per-landmark-layer full descriptor time series + landmark encounters
# ============================================================
run_layer_series = {}   # (run_id, layer) -> dict of arrays over decode steps
run_landmarks = defaultdict(dict)  # run_id -> {node: [steps]}
run_eligibility = {}    # run_id -> "ELIGIBLE" | "LANDMARK_NOT_EVALUABLE_SHORT_EXECUTION"
events = []             # list of dicts: run_id, node, layer, step, family, seed, prompt_id

for r in runs:
    m = r["meta"]
    n_steps = m["n_steps"]
    run_id = r["run_id"]
    if n_steps < 4:
        run_eligibility[run_id] = "LANDMARK_NOT_EVALUABLE_SHORT_EXECUTION"
        continue
    run_eligibility[run_id] = "ELIGIBLE"
    npz = np.load(m["npz_path"])
    prompt_len, seq_len = m["prompt_len"], m["seq_len"]
    H_all = npz["hidden_states_all_layers_fp16"].astype(np.float32)
    Q_all = npz["q_all_layers_fp16"].astype(np.float32)   # (24,14,seq,64)
    K_all = npz["k_all_layers_fp32"]                       # (24,2,seq,64)
    V_all = npz["v_all_layers_fp32"]
    A_all = npz["attentions_all_layers_fp16"].astype(np.float32)
    O_all = npz["attn_output_all_layers_fp16"].astype(np.float32)
    logits = npz["logits_region_fp16"].astype(np.float32)

    for layer, node in NODES.items():
        H = H_all[layer]
        Q = Q_all[layer].transpose(1, 0, 2).reshape(seq_len, -1)  # (seq,14*64)
        K = K_all[layer].transpose(1, 0, 2).reshape(seq_len, -1)  # (seq,2*64)
        V = V_all[layer].transpose(1, 0, 2).reshape(seq_len, -1)
        A = A_all[layer]  # (14, seq, seq)
        O = O_all[layer]

        series = {k: [] for k in ["H_norm", "H_delta_norm", "H_cos", "H_curvature",
                                    "Q_norm", "Q_delta_norm", "Q_cos",
                                    "K_norm", "K_delta_norm", "K_cos",
                                    "V_norm", "V_delta_norm", "V_cos",
                                    "A_dispersion", "A_concentration", "A_max_mass",
                                    "Oattn_norm", "Oattn_delta_norm", "Oattn_cos",
                                    "probability_entropy", "probability_top2_gap", "logit_norm"]}
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
                d1 = H[pos] - H[pos - 1]
                d0 = H[pos - 1] - H[pos - 2]
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

        # landmark z-score detection (frozen rule, only on the 4 canonical signals)
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

print(f"PASS 1 done: {len(events)} landmark events across {len(run_landmarks)} runs")

# ============================================================
# RUN_ELIGIBILITY_AUDIT.csv
# ============================================================
with open(HERE / "RUN_ELIGIBILITY_AUDIT.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run_id", "prompt_id", "prompt_family", "seed", "n_steps", "eligibility", "n_landmarks_observed", "landmarks", "route_signature"])
    for r in runs:
        run_id = r["run_id"]
        m = r["meta"]
        elig = run_eligibility[run_id]
        lm = run_landmarks.get(run_id, {})
        if elig == "LANDMARK_NOT_EVALUABLE_SHORT_EXECUTION":
            route = "NONE"
            n_lm = 0
            lm_str = ""
        else:
            n_lm = len(lm)
            lm_str = "|".join(sorted(lm.keys()))
            if n_lm == 0:
                route = "NONE"
            else:
                enc = []
                for node, steps in lm.items():
                    for s in steps:
                        enc.append((s, node))
                enc.sort()
                tokens = []
                i = 0
                while i < len(enc):
                    j = i
                    tie = [enc[i][1]]
                    while j + 1 < len(enc) and enc[j + 1][0] == enc[i][0]:
                        j += 1
                        tie.append(enc[j][1])
                    tokens.append("~".join(sorted(set(tie))))
                    i = j + 1
                route = ">".join(tokens)
        w.writerow([run_id, r["prompt_id"], r["prompt_family"], r["seed"], m["n_steps"], elig, n_lm, lm_str, route])

# reload for downstream use
with open(HERE / "RUN_ELIGIBILITY_AUDIT.csv") as f:
    elig_rows = list(csv.DictReader(f))

print("RUN_ELIGIBILITY_AUDIT.csv written")

with open(HERE / "PASS1_CACHE.json", "w") as f:
    json.dump({"n_events": len(events), "events": events}, f)

print("Checkpoint saved (PASS1_CACHE.json). Continuing to downstream analyses...")
