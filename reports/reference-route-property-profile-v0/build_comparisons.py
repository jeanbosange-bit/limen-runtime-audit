"""
Part 3: statistical comparisons (Mann-Whitney U, Holm, effect size), family/seed/length
confound checks, final ROUTE_PROPERTY_REGISTRY.json.

NO_LANDMARK comparison population note (disclosed): NO_LANDMARK runs have zero landmark
encounters by construction, so there is no encounter-step to extract a t=0 local window
from. Its comparison population instead uses reports/atlas-coverage-analysis-v1's
already-computed per-run-layer MEAN values (PASS2_SERIES_SUMMARY_CACHE.json) at the 3
route-relevant layers (0, 5, 20) -- a coarser, disclosed asymmetry versus the other
groups' specific-encounter-step values, not a silent substitution.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
COVERAGE = HERE.parent / "atlas-coverage-analysis-v1"

with open(HERE / "GROUPS_CACHE.json") as f:
    GROUPS = json.load(f)
with open(COVERAGE / "PASS2_SERIES_SUMMARY_CACHE.json") as f:
    means_cache = json.load(f)["per_run_layer_means"]
with open(COVERAGE / "RUN_ELIGIBILITY_AUDIT.csv") as f:
    elig_rows = list(csv.DictReader(f))
run_meta = {r["run_id"]: r for r in elig_rows}

PROPERTIES = ["H_norm", "H_curvature", "Q_norm", "K_norm", "V_norm", "A_dispersion", "A_concentration",
              "Oattn_norm", "Oattn_delta_norm", "probability_entropy", "probability_top2_gap", "logit_norm"]
COMPARISON_GROUPS = ["L0_L5_ONLY", "L0_L20_ONLY", "OTHER_LANDMARK_ROUTE", "NO_LANDMARK"]
ROUTE_LAYERS = [0, 5, 20]

TENSOR_ROUTE_FILES = {
    "H_norm": "H_ROUTE_PROPERTIES.csv", "H_curvature": "H_ROUTE_PROPERTIES.csv",
    "Q_norm": "Q_ROUTE_PROPERTIES.csv", "K_norm": "K_ROUTE_PROPERTIES.csv", "V_norm": "V_ROUTE_PROPERTIES.csv",
    "A_dispersion": "ATTENTION_ROUTE_PROPERTIES.csv", "A_concentration": "ATTENTION_ROUTE_PROPERTIES.csv",
    "Oattn_norm": "ATTENTION_OUTPUT_ROUTE_PROPERTIES.csv", "Oattn_delta_norm": "ATTENTION_OUTPUT_ROUTE_PROPERTIES.csv",
    "probability_entropy": "LOGIT_PROBABILITY_ROUTE_PROPERTIES.csv", "probability_top2_gap": "LOGIT_PROBABILITY_ROUTE_PROPERTIES.csv",
    "logit_norm": "LOGIT_PROBABILITY_ROUTE_PROPERTIES.csv",
}

_file_cache = {}


def load_route_file(fname):
    if fname not in _file_cache:
        with open(HERE / fname) as f:
            _file_cache[fname] = list(csv.DictReader(f))
    return _file_cache[fname]


def group_values(group, prop, restrict_runs=None, restrict_seed=None, restrict_family_exclude=None):
    """Returns list of float values for a property, for a given group (t=0 rows for A-D, per-run-layer means for E)."""
    vals = []
    if group == "NO_LANDMARK":
        runs = GROUPS["NO_LANDMARK"] if restrict_runs is None else [r for r in GROUPS["NO_LANDMARK"] if r in restrict_runs]
        for run_id in runs:
            if restrict_seed is not None and run_meta[run_id]["seed"] != restrict_seed:
                continue
            if restrict_family_exclude is not None and run_meta[run_id]["prompt_family"] == restrict_family_exclude:
                continue
            for layer in ROUTE_LAYERS:
                key = f"{run_id}|{layer}"
                if key in means_cache and prop in means_cache[key] and means_cache[key][prop] is not None:
                    vals.append(means_cache[key][prop])
    else:
        fname = TENSOR_ROUTE_FILES[prop]
        rows = load_route_file(fname)
        for row in rows:
            if row["group"] != group or int(row["offset"]) != 0 or row["available"] != "True":
                continue
            run_id = row["run_id"]
            if restrict_runs is not None and run_id not in restrict_runs:
                continue
            if restrict_seed is not None and run_meta[run_id]["seed"] != restrict_seed:
                continue
            if restrict_family_exclude is not None and run_meta[run_id]["prompt_family"] == restrict_family_exclude:
                continue
            v = row.get(prop)
            if v is not None and v != "":
                vals.append(float(v))
    return vals


def mwu_test(a, b):
    a, b = np.array(a), np.array(b)
    if len(a) < 5 or len(b) < 5:
        return None
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    rank_biserial = 1 - (2 * stat) / (len(a) * len(b))
    return {"n_a": len(a), "n_b": len(b), "median_a": round(float(np.median(a)), 5), "median_b": round(float(np.median(b)), 5),
            "U": float(stat), "p_raw": float(p), "effect_size_rank_biserial": round(float(rank_biserial), 4)}


# ============================================================
# ROUTE_GROUP_COMPARISONS.json -- 48 predeclared tests
# ============================================================
route_full_vals = {prop: group_values("ROUTE_FULL", prop) for prop in PROPERTIES}
all_results = {}
raw_p_list = []
for prop in PROPERTIES:
    a = route_full_vals[prop]
    for comp_group in COMPARISON_GROUPS:
        b = group_values(comp_group, prop)
        res = mwu_test(a, b)
        key = f"{prop}|{comp_group}"
        all_results[key] = res
        if res is not None:
            raw_p_list.append((key, res["p_raw"]))

items = sorted(raw_p_list, key=lambda x: x[1])
n = len(items)
holm_p = {}
running_max = 0.0
for rank, (key, p) in enumerate(items, start=1):
    adj = min(1.0, p * (n - rank + 1))
    running_max = max(running_max, adj)
    holm_p[key] = running_max

for key, res in all_results.items():
    if res is not None:
        res["holm_adjusted_p"] = holm_p[key]

with open(HERE / "ROUTE_GROUP_COMPARISONS.json", "w") as f:
    json.dump({
        "n_predeclared_tests": len(PROPERTIES) * len(COMPARISON_GROUPS),
        "n_evaluable_tests": n,
        "no_landmark_population_note": "NO_LANDMARK uses per-run-layer MEAN values (atlas-coverage-analysis-v1 PASS2_SERIES_SUMMARY_CACHE.json), not encounter-step values -- disclosed asymmetry, see FROZEN_PROPERTY_PROTOCOL.md",
        "results": all_results,
    }, f, indent=2, default=str)

print(f"ROUTE_GROUP_COMPARISONS.json written: {n} evaluable of {len(PROPERTIES)*len(COMPARISON_GROUPS)} predeclared")
for key, res in all_results.items():
    if res is not None and res["holm_adjusted_p"] < 0.05:
        print("SIGNIFICANT:", key, res)

with open(HERE / "COMPARISON_CACHE.json", "w") as f:
    json.dump({"route_full_vals": route_full_vals, "all_results": all_results}, f, default=str)

print("PART 3 STAGE 1 DONE")
