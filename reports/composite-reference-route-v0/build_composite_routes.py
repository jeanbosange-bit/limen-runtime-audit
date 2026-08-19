"""
Composite Reference Route V0 -- 3-node composite route validation over the frozen
L0/L1/L5/L20 landmark network. Read-only reuse of reference-landmark-network-v1's
RELATIONS_CACHE.json and atlas-coverage-analysis-v1's PASS1_CACHE.json.
"""
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

run_groups = net_cache["run_groups"]  # run_id -> list of tie-groups (sorted node lists), only for runs w/ >=1 encounter
run_meta = {r["run_id"]: r for r in elig_rows}

# raw events with step numbers, grouped by run
run_raw_events = defaultdict(list)
for ev in events:
    run_raw_events[ev["run_id"]].append(ev)

# candidate universe
SUBSETS = [("L0", "L5", "L20"), ("L0", "L1", "L5"), ("L0", "L1", "L20"), ("L1", "L5", "L20")]
CANDIDATES = []
for subset in SUBSETS:
    for perm in permutations(subset):
        CANDIDATES.append(perm)
assert len(CANDIDATES) == 24


def classify_run_for_route(groups, route):
    """groups: list of tie-groups (each a list of node names) in temporal order.
    route: tuple of 3 node names (a,b,c).
    Returns one of DIRECT_COMPOSITE_ROUTE / ORDERED_ROUTE_WITH_GAPS / SAME_STEP_TIE_ROUTE / NO_COMPOSITE_ROUTE."""
    a, b, c = route
    # find group indices containing each node (first occurrence per node, per V0 canonical convention)
    idx_of = {}
    for gi, g in enumerate(groups):
        for node in (a, b, c):
            if node in g and node not in idx_of:
                idx_of[node] = gi
    if not all(n in idx_of for n in (a, b, c)):
        return "NO_COMPOSITE_ROUTE"
    ia, ib, ic = idx_of[a], idx_of[b], idx_of[c]
    # tie check: any two of the three sharing the same group index
    if len({ia, ib, ic}) < 3:
        return "SAME_STEP_TIE_ROUTE"
    if not (ia < ib < ic):
        return "NO_COMPOSITE_ROUTE"
    if ib == ia + 1 and ic == ib + 1:
        return "DIRECT_COMPOSITE_ROUTE"
    return "ORDERED_ROUTE_WITH_GAPS"


def route_holds(classification):
    return classification in ("DIRECT_COMPOSITE_ROUTE", "ORDERED_ROUTE_WITH_GAPS")


# ============================================================
# Observed classification for all runs x all 24 candidates
# ============================================================
all_runs_with_landmarks = list(run_groups.keys())
observed_classification = {}  # (route, run_id) -> classification
for route in CANDIDATES:
    for run_id in all_runs_with_landmarks:
        observed_classification[(route, run_id)] = classify_run_for_route(run_groups[run_id], route)

observed_counts = {}
observed_support_runs = {}
for route in CANDIDATES:
    supp = {r for r in all_runs_with_landmarks if route_holds(observed_classification[(route, r)])}
    observed_support_runs[route] = supp
    observed_counts[route] = len(supp)

with open(HERE / "ROUTE_OBSERVED_COUNTS.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["route", "n_direct", "n_gapped", "n_tie", "n_none", "n_holding", "n_runs_with_landmarks_total",
                "n_no_landmark_runs", "n_single_landmark_runs", "n_short_noneval_runs"])
    n_no_landmark = sum(1 for r in elig_rows if r["eligibility"] == "ELIGIBLE" and r["n_landmarks_observed"] == "0")
    n_single = sum(1 for r in elig_rows if r["n_landmarks_observed"] == "1")
    n_short_noneval = sum(1 for r in elig_rows if r["eligibility"] != "ELIGIBLE")
    for route in CANDIDATES:
        cls = [observed_classification[(route, r)] for r in all_runs_with_landmarks]
        c = Counter(cls)
        w.writerow(["-".join(route), c.get("DIRECT_COMPOSITE_ROUTE", 0), c.get("ORDERED_ROUTE_WITH_GAPS", 0),
                    c.get("SAME_STEP_TIE_ROUTE", 0), c.get("NO_COMPOSITE_ROUTE", 0), observed_counts[route],
                    len(all_runs_with_landmarks), n_no_landmark, n_single, n_short_noneval])

print("ROUTE_OBSERVED_COUNTS.csv written")
for route in CANDIDATES:
    print("-".join(route), observed_counts[route])

with open(HERE / "OBS_CACHE.json", "w") as f:
    json.dump({
        "routes": ["-".join(r) for r in CANDIDATES],
        "observed_counts": {"-".join(r): observed_counts[r] for r in CANDIDATES},
        "observed_support_runs": {"-".join(r): sorted(v) for r, v in observed_support_runs.items()},
    }, f)

print("PART 1 DONE")
