"""Composite Reference Route V0 -- part 2: permutation testing, Holm correction, family/seed/length
breakdowns, leave-family-out, temporal distance, branching-vs-composition."""
import csv
import json
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NET_V1 = HERE.parent / "reference-landmark-network-v1"
COVERAGE = HERE.parent / "atlas-coverage-analysis-v1"
NODES = ["L0", "L1", "L5", "L20"]
RNG_SEED = 20260818
N_PERM = 10000

with open(NET_V1 / "RELATIONS_CACHE.json") as f:
    net_cache = json.load(f)
with open(COVERAGE / "PASS1_CACHE.json") as f:
    events = json.load(f)["events"]
with open(COVERAGE / "RUN_ELIGIBILITY_AUDIT.csv") as f:
    elig_rows = list(csv.DictReader(f))
with open(NET_V1 / "EDGE_VALIDATION_RESULTS.json") as f:
    edge_val = json.load(f)
with open(HERE / "OBS_CACHE.json") as f:
    obs_cache = json.load(f)

run_groups = net_cache["run_groups"]
run_meta = {r["run_id"]: r for r in elig_rows}
run_family = {r["run_id"]: r["prompt_family"] for r in elig_rows}
run_seed = {r["run_id"]: r["seed"] for r in elig_rows}
run_prompt = {r["run_id"]: r["prompt_id"] for r in elig_rows}
run_nsteps = {r["run_id"]: int(r["n_steps"]) for r in elig_rows}
FAMILIES = sorted(set(run_family.values()))
SEEDS = ["2101", "2102", "2103"]

run_raw_events = defaultdict(list)
for ev in events:
    run_raw_events[ev["run_id"]].append(ev)

SUBSETS = [("L0", "L5", "L20"), ("L0", "L1", "L5"), ("L0", "L1", "L20"), ("L1", "L5", "L20")]
CANDIDATES = [p for s in SUBSETS for p in permutations(s)]
all_runs_with_landmarks = list(run_groups.keys())


def classify_run_for_route(groups, route):
    a, b, c = route
    idx_of = {}
    for gi, g in enumerate(groups):
        for node in (a, b, c):
            if node in g and node not in idx_of:
                idx_of[node] = gi
    if not all(n in idx_of for n in (a, b, c)):
        return "NO_COMPOSITE_ROUTE"
    ia, ib, ic = idx_of[a], idx_of[b], idx_of[c]
    if len({ia, ib, ic}) < 3:
        return "SAME_STEP_TIE_ROUTE"
    if not (ia < ib < ic):
        return "NO_COMPOSITE_ROUTE"
    return "DIRECT_COMPOSITE_ROUTE" if (ib == ia + 1 and ic == ib + 1) else "ORDERED_ROUTE_WITH_GAPS"


def route_holds(c):
    return c in ("DIRECT_COMPOSITE_ROUTE", "ORDERED_ROUTE_WITH_GAPS")


def permute_run_groups(groups, rng):
    flat = [lab for g in groups for lab in g]
    perm_labels = rng.permutation(flat)
    new_groups = []
    idx = 0
    for g in groups:
        size = len(g)
        new_groups.append(sorted(perm_labels[idx:idx + size]))
        idx += size
    return new_groups


observed_counts = {route: obs_cache["observed_counts"]["-".join(route)] for route in CANDIDATES}

print(f"Running {N_PERM} permutations for 24 route candidates...", flush=True)
rng = np.random.default_rng(RNG_SEED)
perm_counts = {route: np.zeros(N_PERM, dtype=int) for route in CANDIDATES}

for p in range(N_PERM):
    perm_groups_by_run = {run_id: permute_run_groups(run_groups[run_id], rng) for run_id in all_runs_with_landmarks}
    for route in CANDIDATES:
        n = 0
        for run_id in all_runs_with_landmarks:
            if route_holds(classify_run_for_route(perm_groups_by_run[run_id], route)):
                n += 1
        perm_counts[route][p] = n
    if (p + 1) % 2000 == 0:
        print(f"  {p+1}/{N_PERM} done", flush=True)

print("Permutations complete", flush=True)

raw_p = {}
for route in CANDIDATES:
    obs = observed_counts[route]
    perm = perm_counts[route]
    raw_p[route] = float((np.sum(perm >= obs) + 1) / (N_PERM + 1))

items = sorted(raw_p.items(), key=lambda x: x[1])
n = len(items)
holm_p = {}
running_max = 0.0
for rank, (route, p) in enumerate(items, start=1):
    adj = min(1.0, p * (n - rank + 1))
    running_max = max(running_max, adj)
    holm_p[route] = running_max

with open(HERE / "ROUTE_NULL_RESULTS.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["route", "observed", "null_mean", "null_median", "raw_p", "holm_adjusted_p", "n_permutations"])
    for route in CANDIDATES:
        perm = perm_counts[route]
        w.writerow(["-".join(route), observed_counts[route], round(float(np.mean(perm)), 3), float(np.median(perm)),
                    raw_p[route], holm_p[route], N_PERM])

print("ROUTE_NULL_RESULTS.csv written", flush=True)
for route in sorted(CANDIDATES, key=lambda r: -observed_counts[r])[:6]:
    print("-".join(route), "obs=", observed_counts[route], "holm_p=", round(holm_p[route], 4))

with open(HERE / "NULL_CACHE.json", "w") as f:
    json.dump({"raw_p": {"-".join(k): v for k, v in raw_p.items()}, "holm_p": {"-".join(k): v for k, v in holm_p.items()}}, f)

print("PART 2 DONE")
