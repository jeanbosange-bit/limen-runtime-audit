"""Composite Reference Route V0 -- part 3: family/seed/length breakdowns, leave-family-out,
temporal distance, branching-vs-composition, final route validation results."""
import csv
import json
import statistics as st
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

HERE = Path(__file__).resolve().parent
NET_V1 = HERE.parent / "reference-landmark-network-v1"
COVERAGE = HERE.parent / "atlas-coverage-analysis-v1"

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
with open(HERE / "NULL_CACHE.json") as f:
    null_cache = json.load(f)

run_groups = net_cache["run_groups"]
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


def length_bin(n):
    return "SHORT_1_9" if n <= 9 else ("MEDIUM_10_49" if n <= 49 else "LONG_50_PLUS")


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


all_runs_with_landmarks = list(run_groups.keys())
support_runs = {route: set(obs_cache["observed_support_runs"]["-".join(route)]) for route in CANDIDATES}

# ============================================================
# ROUTE_FAMILY_SUPPORT.csv / ROUTE_SEED_SUPPORT.csv / ROUTE_LENGTH_SUPPORT.csv
# ============================================================
with open(HERE / "ROUTE_FAMILY_SUPPORT.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["route"] + FAMILIES)
    for route in CANDIDATES:
        fam_counts = Counter(run_family[r] for r in support_runs[route])
        w.writerow(["-".join(route)] + [fam_counts.get(fam, 0) for fam in FAMILIES])

with open(HERE / "ROUTE_SEED_SUPPORT.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["route"] + SEEDS)
    for route in CANDIDATES:
        seed_counts = Counter(str(run_seed[r]) for r in support_runs[route])
        w.writerow(["-".join(route)] + [seed_counts.get(s, 0) for s in SEEDS])

with open(HERE / "ROUTE_LENGTH_SUPPORT.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["route", "SHORT_1_9", "MEDIUM_10_49", "LONG_50_PLUS"])
    for route in CANDIDATES:
        len_counts = Counter(length_bin(run_nsteps[r]) for r in support_runs[route])
        w.writerow(["-".join(route), len_counts.get("SHORT_1_9", 0), len_counts.get("MEDIUM_10_49", 0), len_counts.get("LONG_50_PLUS", 0)])

print("Family/seed/length support CSVs written")

# ============================================================
# LEAVE_FAMILY_OUT_RESULTS.csv
# ============================================================
with open(HERE / "LEAVE_FAMILY_OUT_RESULTS.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["route", "baseline_observed"] + [f"loo_{fam}" for fam in FAMILIES] + ["stability_label"])
    for route in CANDIDATES:
        supp = support_runs[route]
        baseline = len(supp)
        if baseline < 3:
            w.writerow(["-".join(route), baseline] + ["NOT_EVALUABLE"] * len(FAMILIES) + ["NOT_EVALUABLE"])
            continue
        loo_vals = []
        flips = 0
        for fam in FAMILIES:
            n = len({r for r in supp if run_family[r] != fam})
            loo_vals.append(n)
            if n < 3:
                flips += 1
        stability = "STABLE" if flips == 0 else "FAMILY_SENSITIVE"
        w.writerow(["-".join(route), baseline] + loo_vals + [stability])

print("LEAVE_FAMILY_OUT_RESULTS.csv written")

# ============================================================
# ROUTE_TEMPORAL_DISTANCE.csv (for the primary candidate + its permutation siblings, and top candidates)
# ============================================================
def first_step(run_id, node):
    steps = [e["step"] for e in run_raw_events[run_id] if e["node"] == node]
    return min(steps) if steps else None


temporal_rows = []
for route in CANDIDATES:
    supp = support_runs[route]
    if not supp:
        continue
    a, b, c = route
    pairs = [(a, b), (a, c), (b, c)]
    for (x, y) in pairs:
        dists = []
        for run_id in supp:
            sx, sy = first_step(run_id, x), first_step(run_id, y)
            if sx is not None and sy is not None:
                dists.append(abs(sy - sx))
        if dists:
            dists_sorted = sorted(dists)
            q1 = dists_sorted[len(dists_sorted) // 4]
            q3 = dists_sorted[(3 * len(dists_sorted)) // 4]
            temporal_rows.append({"route": "-".join(route), "pair": f"{x}->{y}", "n": len(dists),
                                    "min": min(dists), "median": st.median(dists), "q1": q1, "q3": q3, "max": max(dists)})

with open(HERE / "ROUTE_TEMPORAL_DISTANCE.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["route", "pair", "n", "min", "median", "q1", "q3", "max"])
    w.writeheader()
    for row in temporal_rows:
        w.writerow(row)

print(f"ROUTE_TEMPORAL_DISTANCE.csv written: {len(temporal_rows)} rows")

# ============================================================
# ROUTE_VALIDATION_RESULTS.json
# ============================================================
holm_p = null_cache["holm_p"]
raw_p = null_cache["raw_p"]
loo_map = {}
with open(HERE / "LEAVE_FAMILY_OUT_RESULTS.csv") as f:
    for r in csv.DictReader(f):
        loo_map[r["route"]] = r["stability_label"]

route_results = {}
for route in CANDIDATES:
    key = "-".join(route)
    supp = support_runs[route]
    obs = len(supp)
    n_fam = len(set(run_family[r] for r in supp))
    n_seed = len(set(run_seed[r] for r in supp))
    n_prompt = len(set(run_prompt[r] for r in supp))
    p_holm = holm_p[key]
    stability = loo_map.get(key, "NOT_EVALUABLE")
    if obs == 0:
        status = "REFERENCE_ROUTE_NOT_SUPPORTED"
    elif p_holm < 0.05 and n_fam >= 2 and n_seed >= 2 and n_prompt >= 2 and stability == "STABLE":
        status = "REFERENCE_ROUTE_SUPPORTED"
    elif obs >= 2:
        status = "REFERENCE_ROUTE_LOW_SUPPORT"
    else:
        status = "REFERENCE_ROUTE_NOT_SUPPORTED"
    a, b, c = route
    edges_needed = [f"{a}->{b}", f"{b}->{c}", f"{a}->{c}"]
    edge_statuses = {e: edge_val["directed_edges"].get(e, {}).get("status", "N/A") for e in [f"{a}->{b}", f"{b}->{c}"]}
    route_results[key] = {
        "route": list(route), "observed_count": obs, "n_families": n_fam, "n_seeds": n_seed, "n_prompt_ids": n_prompt,
        "raw_p": raw_p[key], "holm_adjusted_p": p_holm, "leave_family_out_stability": stability,
        "constituent_edge_statuses": edge_statuses, "status": status,
    }

with open(HERE / "ROUTE_VALIDATION_RESULTS.json", "w") as f:
    json.dump(route_results, f, indent=2)

n_supported = sum(1 for v in route_results.values() if v["status"] == "REFERENCE_ROUTE_SUPPORTED")
n_low = sum(1 for v in route_results.values() if v["status"] == "REFERENCE_ROUTE_LOW_SUPPORT")
print(f"ROUTE_VALIDATION_RESULTS.json written: {n_supported} SUPPORTED, {n_low} LOW_SUPPORT")
for k, v in route_results.items():
    if v["status"] == "REFERENCE_ROUTE_SUPPORTED":
        print("SUPPORTED:", k, v)

# ============================================================
# BRANCHING_VS_COMPOSITION_ANALYSIS.json
# ============================================================
primary = ("L0", "L5", "L20")
primary_key = "L0-L5-L20"
l0_l5_supp = set()
l0_l20_supp = set()
for run_id in all_runs_with_landmarks:
    groups = run_groups[run_id]
    # direct/gapped succession L0->L5 (any distance, ordered)
    idx = {}
    for gi, g in enumerate(groups):
        for node in ("L0", "L5", "L20"):
            if node in g and node not in idx:
                idx[node] = gi
    if "L0" in idx and "L5" in idx and idx["L0"] < idx["L5"]:
        l0_l5_supp.add(run_id)
    if "L0" in idx and "L20" in idx and idx["L0"] < idx["L20"]:
        l0_l20_supp.add(run_id)

both_branches = l0_l5_supp & l0_l20_supp
composite_supp = support_runs[primary]
branch_only = both_branches - composite_supp  # both edges present, but not in the L0->L5->L20 order

primary_status = route_results[primary_key]["status"]
if primary_status == "REFERENCE_ROUTE_SUPPORTED":
    branching_verdict = "COMPOSITION_SUPPORTED"
elif len(both_branches) < 5 and len(composite_supp) < 5:
    branching_verdict = "NOT_ESTABLISHED"
elif len(both_branches) > 2 * max(len(composite_supp), 1):
    branching_verdict = "BRANCHING_MORE_CONSISTENT"
else:
    branching_verdict = "BOTH_PARTIAL"

branching_analysis = {
    "primary_candidate": "L0-L5-L20", "primary_candidate_status": primary_status,
    "runs_supporting_L0_before_L5": len(l0_l5_supp), "runs_supporting_L0_before_L20": len(l0_l20_supp),
    "runs_supporting_both_branches_any_order": len(both_branches),
    "runs_supporting_full_composite_order": len(composite_supp),
    "runs_with_both_branches_but_not_composite_order": len(branch_only),
    "branch_only_run_ids": sorted(branch_only),
    "verdict": branching_verdict,
    "verdict_rule": "reports/composite-reference-route-v0/FROZEN_ROUTE_PROTOCOL.md#branching-vs-composition",
}
with open(HERE / "BRANCHING_VS_COMPOSITION_ANALYSIS.json", "w") as f:
    json.dump(branching_analysis, f, indent=2)

print("BRANCHING_VS_COMPOSITION_ANALYSIS.json written:", branching_verdict)
print("PART 3 DONE")
