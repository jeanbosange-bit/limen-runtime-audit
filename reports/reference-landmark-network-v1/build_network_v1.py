"""
Reference Landmark Network V1 -- Lane B out-of-sample validation.
Read-only: reuses reports/atlas-coverage-analysis-v1/PASS1_CACHE.json (305 landmark
encounter events, independently recomputed and cross-verified against Lane B's own
audit) and reports/atlas-coverage-analysis-v1/RUN_ELIGIBILITY_AUDIT.csv. No npz reload,
no source artifact modified.
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COVERAGE = HERE.parent / "atlas-coverage-analysis-v1"
NODES = ["L0", "L1", "L5", "L20"]
NODE_LAYER = {"L0": 0, "L1": 1, "L5": 5, "L20": 20}
RNG_SEED = 20260818
N_PERM = 10000

with open(COVERAGE / "PASS1_CACHE.json") as f:
    events = json.load(f)["events"]
with open(COVERAGE / "RUN_ELIGIBILITY_AUDIT.csv") as f:
    elig_rows = list(csv.DictReader(f))
run_meta = {r["run_id"]: r for r in elig_rows}

# ============================================================
# Build per-run ordered, tie-grouped encounter sequence
# ============================================================
run_events = defaultdict(list)
for ev in events:
    run_events[ev["run_id"]].append(ev)

run_groups = {}   # run_id -> list of sets (tie groups) in temporal order
run_labels_flat = {}  # run_id -> flat list of (step, node) for permutation bookkeeping
for run_id, evs in run_events.items():
    evs_sorted = sorted(evs, key=lambda e: e["step"])
    groups = []
    i = 0
    while i < len(evs_sorted):
        j = i
        tie = {evs_sorted[i]["node"]}
        while j + 1 < len(evs_sorted) and evs_sorted[j + 1]["step"] == evs_sorted[i]["step"]:
            j += 1
            tie.add(evs_sorted[j]["node"])
        groups.append(sorted(tie))
        i = j + 1
    run_groups[run_id] = groups

# ============================================================
# ELIGIBILITY_AUDIT.csv (edge-eligibility layer on top of atlas-coverage's eligibility)
# ============================================================
edge_eligible_runs = set()
with open(HERE / "ELIGIBILITY_AUDIT.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run_id", "prompt_id", "prompt_family", "seed", "n_steps", "landmark_eligibility",
                "n_landmark_encounters", "edge_eligible", "route_groups"])
    for r in elig_rows:
        run_id = r["run_id"]
        n_enc = len(run_events.get(run_id, []))
        edge_elig = n_enc >= 2
        if edge_elig:
            edge_eligible_runs.add(run_id)
        groups = run_groups.get(run_id, [])
        route_str = "-".join("+".join(g) for g in groups) if groups else "NONE"
        w.writerow([run_id, r["prompt_id"], r["prompt_family"], r["seed"], r["n_steps"], r["eligibility"],
                    n_enc, edge_elig, route_str])

print(f"Edge-eligible runs: {len(edge_eligible_runs)}")

# ============================================================
# Extract observed edge/co-occurrence instances per run
# ============================================================
DIRECTED = [(a, b) for a in NODES for b in NODES if a != b]
UNDIRECTED = [tuple(sorted((a, b))) for i, a in enumerate(NODES) for b in NODES[i + 1:]]


def extract_run_relations(groups):
    """Returns (direct_pairs, indirect_pairs, tie_pairs) as sets of (a,b) tuples (directed for direct/indirect,
    unordered-as-sorted-tuple for tie)."""
    direct, indirect, tie = set(), set(), set()
    for g in groups:
        for i, a in enumerate(g):
            for b in g[i + 1:]:
                tie.add(tuple(sorted((a, b))))
    for i in range(len(groups)):
        for a in groups[i]:
            for j in range(i + 1, len(groups)):
                for b in groups[j]:
                    if a == b:
                        continue
                    if j == i + 1:
                        direct.add((a, b))
                    else:
                        indirect.add((a, b))
    return direct, indirect, tie


observed_direct = defaultdict(set)
observed_indirect = defaultdict(set)
observed_tie = defaultdict(set)
for run_id in edge_eligible_runs:
    d, i, t = extract_run_relations(run_groups[run_id])
    for pair in d:
        observed_direct[pair].add(run_id)
    for pair in i:
        observed_indirect[pair].add(run_id)
    for pair in t:
        observed_tie[pair].add(run_id)


def primary_support(pair):
    return observed_direct.get(pair, set()) | observed_indirect.get(pair, set())


observed_counts = {pair: len(primary_support(pair)) for pair in DIRECTED}
# co-occurrence = any relation (direct, indirect in either direction, or tie) between the pair in the same run
occ_runs = {}
for pair in UNDIRECTED:
    a, b = pair
    runs = observed_tie.get(pair, set()) | primary_support((a, b)) | primary_support((b, a))
    occ_runs[pair] = runs
observed_cooccurrence = {pair: len(runs) for pair, runs in occ_runs.items()}

print("Observed directed counts:", observed_counts)
print("Observed co-occurrence counts:", observed_cooccurrence)

with open(HERE / "EDGE_OBSERVED_COUNTS.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["source", "target", "n_direct", "n_indirect", "n_primary_total"])
    for a, b in DIRECTED:
        w.writerow([a, b, len(observed_direct.get((a, b), set())), len(observed_indirect.get((a, b), set())), observed_counts[(a, b)]])

print("PART 1 (relation extraction) DONE")
with open(HERE / "RELATIONS_CACHE.json", "w") as f:
    json.dump({
        "edge_eligible_runs": sorted(edge_eligible_runs),
        "run_groups": run_groups,
        "observed_direct": {f"{a}|{b}": sorted(v) for (a, b), v in observed_direct.items()},
        "observed_indirect": {f"{a}|{b}": sorted(v) for (a, b), v in observed_indirect.items()},
        "observed_tie": {f"{a}|{b}": sorted(v) for (a, b), v in observed_tie.items()},
        "occ_runs": {f"{a}|{b}": sorted(v) for (a, b), v in occ_runs.items()},
    }, f)
