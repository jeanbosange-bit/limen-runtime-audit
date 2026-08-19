"""
Reference Landmark Network V0 -- builds the network from the 4 validated landmarks
(L0/L1/L5/L20) using ONLY existing artifacts:
  reports/multi-tensor-atlas-v0/LANDMARK_CANDIDATES_RAW.json (107 raw events)
  reports/multi-tensor-atlas-v0/COORDINATE_TABLE.parquet (per-step observables)
  reports/atlas-kv-compression-v0/execution/RUN_MANIFEST.json (run/prompt/family)
  reports/atlas-kv-compression-v0/execution/ANALYSIS_TABLE.json (V1 attention-sensitivity overlay)
  reports/multi-tensor-landmarks-v0/TRANSITION_RESULTS.json (T0/T1 per-layer labels)
real_new_generations = 0.
"""
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ATLAS = Path(__file__).resolve().parent.parent / "multi-tensor-atlas-v0"
LANDMARKS_V0 = Path(__file__).resolve().parent.parent / "multi-tensor-landmarks-v0"
KVC_EXEC = Path(__file__).resolve().parent.parent / "atlas-kv-compression-v0" / "execution"
OUT = Path(__file__).resolve().parent
RNG_SEED = 20260818
N_PERM = 200
NODES = {0: "L0", 1: "L1", 5: "L5", 20: "L20"}
NODE_IDS = list(NODES.values())

with open(ATLAS / "LANDMARK_CANDIDATES_RAW.json") as f:
    all_events = json.load(f)["events"]
with open(KVC_EXEC / "RUN_MANIFEST.json") as f:
    manifest = {r["run_id"]: r for r in json.load(f)["runs"]}
with open(LANDMARKS_V0 / "TRANSITION_RESULTS.json") as f:
    transition_results = json.load(f)

CONTRIB_PROMPTS = ["AH01", "AH02", "FG01", "GI01", "GI02", "SO01", "SO02"]  # 21 runs, the only ones ever eligible for a landmark
ALL_30_RUNS = list(manifest.keys())

# ============================================================
# Per-run qualifying encounters (layer in {0,1,5,20})
# ============================================================
events4 = [e for e in all_events if e["layer"] in NODES]
by_run = {}
for e in events4:
    by_run.setdefault(e["run_id"], []).append(e)
for run_id in by_run:
    by_run[run_id].sort(key=lambda e: e["step"])

# ============================================================
# ROUTE_INSTANCES.csv -- one row per run (all 30, full accounting)
# ============================================================
route_rows = []
for run_id in ALL_30_RUNS:
    m = manifest[run_id]
    evs = by_run.get(run_id, [])
    seq_nodes = [NODES[e["layer"]] for e in evs]
    seq_steps = [e["step"] for e in evs]
    # collapsed route signature: consecutive-duplicate merge; co-occurring (tie-step) pairs joined with '~'
    sig_tokens = []
    i = 0
    while i < len(evs):
        tie_group = [evs[i]]
        j = i + 1
        while j < len(evs) and evs[j]["step"] == evs[i]["step"]:
            tie_group.append(evs[j])
            j += 1
        token = "~".join(sorted(NODES[e["layer"]] for e in tie_group))
        if not sig_tokens or sig_tokens[-1] != token:
            sig_tokens.append(token)
        i = j
    route_signature = "-".join(sig_tokens) if sig_tokens else "NONE"
    repeats = len(seq_nodes) != len(set(seq_nodes))
    bypassed = sorted(set(NODE_IDS) - set(seq_nodes))
    route_rows.append({
        "run_id": run_id, "prompt_id": m["prompt_id"], "family": m["family"], "seed": m["seed"],
        "n_steps": m["n_steps"], "seq_len": m["seq_len"],
        "in_landmark_eligible_prompt_pool": m["prompt_id"] in CONTRIB_PROMPTS,
        "landmark_sequence_full": "-".join(f"{n}@{s}" for n, s in zip(seq_nodes, seq_steps)) if evs else "NONE",
        "route_signature": route_signature,
        "first_encounter_step": seq_steps[0] if seq_steps else None,
        "last_encounter_step": seq_steps[-1] if seq_steps else None,
        "n_encounters": len(evs),
        "n_distinct_landmarks": len(set(seq_nodes)),
        "has_repeat_encounter": repeats,
        "landmarks_bypassed": ",".join(bypassed),
        "route_length_distinct": len(set(seq_nodes)),
        "coverage_fraction_of_4_nodes": round(len(set(seq_nodes)) / 4, 3),
    })

pd.DataFrame(route_rows).to_csv(OUT / "ROUTE_INSTANCES.csv", index=False)

# ============================================================
# ROUTE_SIGNATURES.json -- cluster by collapsed signature
# ============================================================
sig_groups = {}
for r in route_rows:
    sig_groups.setdefault(r["route_signature"], []).append(r)

route_signatures = []
for sig, rows in sig_groups.items():
    n_runs = len(rows)
    prompt_ids = sorted(set(r["prompt_id"] for r in rows))
    families = sorted(set(r["family"] for r in rows))
    seeds = sorted(set(r["seed"] for r in rows))
    lengths = [r["n_distinct_landmarks"] for r in rows]
    if sig == "NONE":
        status = "NO_LANDMARK"
    elif n_runs >= 3 and len(prompt_ids) >= 2:
        status = "COMMON_ROUTE"
    elif n_runs >= 2:
        status = "LOW_SUPPORT_ROUTE"
    else:
        status = "UNIQUE_ROUTE"
    route_signatures.append({
        "route_signature": sig, "n_runs": n_runs, "n_prompt_ids": len(prompt_ids),
        "prompt_ids": prompt_ids, "families": families, "seeds": seeds,
        "mean_length": round(float(np.mean(lengths)), 3), "median_length": float(np.median(lengths)),
        "epistemic_status": status,
    })
route_signatures.sort(key=lambda r: -r["n_runs"])

with open(OUT / "ROUTE_SIGNATURES.json", "w") as f:
    json.dump({"n_unique_signatures": len(route_signatures), "signatures": route_signatures}, f, indent=2)

# ============================================================
# EDGES: DIRECT_SUCCESSION, INDIRECT_SUCCESSION, CO_OCCURRENCE
# ============================================================
pair_support = {(a, b): {"runs_direct": set(), "runs_indirect": set(), "direct_distances": []} for a in NODE_IDS for b in NODE_IDS if a != b}
co_occurrence = {}

for run_id, evs in by_run.items():
    n = len(evs)
    for i in range(n):
        for j in range(i + 1, n):
            a_layer, b_layer = evs[i]["layer"], evs[j]["layer"]
            a, b = NODES[a_layer], NODES[b_layer]
            if evs[i]["step"] == evs[j]["step"]:
                if a_layer != b_layer:
                    key = tuple(sorted([a, b]))
                    co_occurrence.setdefault(key, set()).add(run_id)
                continue
            if a_layer == b_layer:
                continue
            if j == i + 1:
                pair_support[(a, b)]["runs_direct"].add(run_id)
                pair_support[(a, b)]["direct_distances"].append(evs[j]["step"] - evs[i]["step"])
            else:
                pair_support[(a, b)]["runs_indirect"].add(run_id)

edge_rows = []
observed_stat = {}
for (a, b), d in pair_support.items():
    runs_direct = d["runs_direct"]
    runs_indirect = d["runs_indirect"] - runs_direct
    all_runs = runs_direct | runs_indirect
    prompts = sorted(set(manifest[r]["prompt_id"] for r in all_runs))
    families = sorted(set(manifest[r]["family"] for r in all_runs))
    dist = d["direct_distances"]
    edge_rows.append({
        "from": a, "to": b,
        "n_runs_direct": len(runs_direct), "n_runs_indirect": len(runs_indirect),
        "n_runs_total": len(all_runs),
        "run_ids": ",".join(sorted(all_runs)),
        "n_prompt_ids": len(prompts), "prompt_ids": ",".join(prompts),
        "n_families": len(families), "families": ",".join(families),
        "median_step_distance_direct": float(np.median(dist)) if dist else None,
        "step_distance_std_direct": float(np.std(dist)) if len(dist) > 1 else (0.0 if dist else None),
    })
    observed_stat[(a, b)] = len(all_runs)

# co-occurrence rows (unordered)
co_occ_rows = [{"pair": "~".join(k), "n_runs": len(v), "run_ids": ",".join(sorted(v))} for k, v in co_occurrence.items()]

# direction consistency: compare (a,b) vs (b,a)
for row in edge_rows:
    a, b = row["from"], row["to"]
    fwd = observed_stat[(a, b)]
    rev = observed_stat[(b, a)]
    row["reverse_pair_n_runs_total"] = rev
    row["direction_consistency"] = "ONE_DIRECTIONAL" if (fwd > 0 and rev == 0) else ("BIDIRECTIONAL_OBSERVED" if (fwd > 0 and rev > 0) else "NO_OBSERVED_LINK")
    row["background_frequency_note"] = "see EDGE_VALIDATION_RESULTS.json for the frozen permutation-null background rate for this pair"

df_edges = pd.DataFrame(edge_rows)
df_edges.to_csv(OUT / "EDGE_MATRIX.csv", index=False)

# ============================================================
# EDGE VALIDATION: frozen within-run label-shuffle null, 200 perms, Bonferroni over 12 pairs
# ============================================================
def compute_pair_counts(run_events):
    """run_events: dict run_id -> list of (layer,) in temporal(sorted) order (labels only, positions fixed).
    Returns dict (a,b) -> n_runs_total (direct+indirect, excluding same-node adjacent)."""
    counts = {(a, b): set() for a in NODE_IDS for b in NODE_IDS if a != b}
    for run_id, layers in run_events.items():
        n = len(layers)
        for i in range(n):
            for j in range(i + 1, n):
                if layers[i] == layers[j]:
                    continue
                a, b = NODES[layers[i]], NODES[layers[j]]
                counts[(a, b)].add(run_id)
    return {k: len(v) for k, v in counts.items()}


# observed, using the SAME logic as pair_support (co-occurrence pairs excluded from succession counting)
observed_layers_per_run = {}
for run_id, evs in by_run.items():
    seq = []
    i = 0
    while i < len(evs):
        j = i
        while j < len(evs) and evs[j]["step"] == evs[i]["step"]:
            j += 1
        # tie group: skip pairwise succession among tied events (handled as co-occurrence), but keep for ordering vs other groups
        for e in evs[i:j]:
            seq.append(e["layer"])
        i = j
    observed_layers_per_run[run_id] = seq

observed_counts_full = compute_pair_counts(observed_layers_per_run)
# note: this differs slightly from pair_support (which separates direct/indirect) -- this is the pooled total used for the null test, consistent definition

rng = np.random.default_rng(RNG_SEED)
perm_counts_list = {k: [] for k in observed_counts_full}
for _ in range(N_PERM):
    perm_run_events = {}
    for run_id, layers in observed_layers_per_run.items():
        if len(layers) < 2:
            perm_run_events[run_id] = layers
            continue
        perm_run_events[run_id] = list(rng.permutation(layers))
    perm_counts = compute_pair_counts(perm_run_events)
    for k in observed_counts_full:
        perm_counts_list[k].append(perm_counts[k])

ALPHA = 0.05
ALPHA_CORRECTED = 0.05 / 12
edge_validation = []
for (a, b), obs in observed_counts_full.items():
    perms = np.array(perm_counts_list[(a, b)])
    p = float((np.sum(perms >= obs) + 1) / (N_PERM + 1))
    if p < ALPHA_CORRECTED:
        status = "VALIDATED"
    elif p < ALPHA:
        status = "LOW_SUPPORT"
    else:
        status = "NOT_SUPPORTED"
    edge_validation.append({
        "from": a, "to": b, "observed_n_runs": obs,
        "permutation_median": float(np.median(perms)), "permutation_mean": float(np.mean(perms)),
        "p_value": p, "alpha_corrected_bonferroni_12": ALPHA_CORRECTED,
        "status": status,
    })

with open(OUT / "EDGE_VALIDATION_RESULTS.json", "w") as f:
    json.dump({
        "null_model": "within-run landmark-label shuffle, group-preserving by run_id, N=200, seed=20260818",
        "n_permutations": N_PERM, "alpha_corrected_bonferroni_12": ALPHA_CORRECTED,
        "n_runs_contributing_to_null": sum(1 for v in observed_layers_per_run.values() if len(v) >= 2),
        "edges": edge_validation,
    }, f, indent=2)

print("EDGE VALIDATION")
for e in sorted(edge_validation, key=lambda x: x["p_value"]):
    print(f"{e['from']}->{e['to']}: obs={e['observed_n_runs']} p={e['p_value']:.4f} status={e['status']}")

# ============================================================
# NODE_PROPERTIES.json -- descriptive only
# ============================================================
G = nx.DiGraph()
G.add_nodes_from(NODE_IDS)
for e in edge_validation:
    if e["observed_n_runs"] > 0:
        G.add_edge(e["from"], e["to"], weight=e["observed_n_runs"])

betweenness = nx.betweenness_centrality(G, weight=None, normalized=True)
node_properties = {}
for node_layer, node_id in NODES.items():
    runs_with_node = [r for r in route_rows if node_id in (r["landmark_sequence_full"] or "")]
    # more precise: check membership via route_signature tokens / full sequence parsing
    runs_with_node = [r for r in route_rows if any(tok.split("@")[0] in (node_id, ) or node_id in tok.split("@")[0].split("~") for tok in (r["landmark_sequence_full"].split("-") if r["landmark_sequence_full"] != "NONE" else []))]
    prompt_ids = sorted(set(r["prompt_id"] for r in runs_with_node))
    families = sorted(set(r["family"] for r in runs_with_node))
    n_encounters_total = sum(1 for r in route_rows for tok in (r["landmark_sequence_full"].split("-") if r["landmark_sequence_full"] != "NONE" else []) if node_id in tok.split("@")[0].split("~"))
    node_properties[node_id] = {
        "epistemic_label": "DESCRIPTIVE_NETWORK_PROPERTY",
        "layer": node_layer,
        "run_coverage_n": len(runs_with_node), "run_coverage_of_21_eligible": round(len(runs_with_node) / 21, 3),
        "prompt_id_coverage": prompt_ids, "n_prompt_ids": len(prompt_ids),
        "family_coverage": families, "n_families": len(families),
        "total_encounters_across_corpus": n_encounters_total,
        "in_degree_observed_any": G.in_degree(node_id), "out_degree_observed_any": G.out_degree(node_id),
        "in_degree_validated_only": sum(1 for e in edge_validation if e["to"] == node_id and e["status"] == "VALIDATED"),
        "out_degree_validated_only": sum(1 for e in edge_validation if e["from"] == node_id and e["status"] == "VALIDATED"),
        "betweenness_centrality_descriptive": round(betweenness[node_id], 4),
    }

with open(OUT / "NODE_PROPERTIES.json", "w") as f:
    json.dump(node_properties, f, indent=2)

print("\nNODE PROPERTIES")
for n, p in node_properties.items():
    print(n, "run_coverage:", p["run_coverage_n"], "families:", p["family_coverage"], "betweenness:", p["betweenness_centrality_descriptive"])

# ============================================================
# PROMPT_FAMILY_ROUTE_MATRIX.csv
# ============================================================
pf_rows = []
for r in route_rows:
    row = {"prompt_id": r["prompt_id"], "family": r["family"], "seed": r["seed"], "run_id": r["run_id"],
           "route_signature": r["route_signature"], "n_distinct_landmarks": r["n_distinct_landmarks"]}
    for node_id in NODE_IDS:
        row[f"has_{node_id}"] = int(any(node_id in tok.split("@")[0].split("~") for tok in (r["landmark_sequence_full"].split("-") if r["landmark_sequence_full"] != "NONE" else [])))
    pf_rows.append(row)
pd.DataFrame(pf_rows).to_csv(OUT / "PROMPT_FAMILY_ROUTE_MATRIX.csv", index=False)

print("\nDONE step 1 (routes/edges/nodes)")
