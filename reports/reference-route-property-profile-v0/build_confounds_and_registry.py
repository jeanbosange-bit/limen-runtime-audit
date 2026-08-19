"""Part 4: apply ROUTE_PROPERTY status thresholds, family/seed/length confound checks,
final ROUTE_PROPERTY_REGISTRY.json."""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
COVERAGE = HERE.parent / "atlas-coverage-analysis-v1"

with open(HERE / "GROUPS_CACHE.json") as f:
    GROUPS = json.load(f)
with open(HERE / "ROUTE_GROUP_COMPARISONS.json") as f:
    comparisons = json.load(f)["results"]
with open(COVERAGE / "PASS2_SERIES_SUMMARY_CACHE.json") as f:
    means_cache = json.load(f)["per_run_layer_means"]
with open(COVERAGE / "RUN_ELIGIBILITY_AUDIT.csv") as f:
    elig_rows = list(csv.DictReader(f))
run_meta = {r["run_id"]: r for r in elig_rows}

PROPERTIES = ["H_norm", "H_curvature", "Q_norm", "K_norm", "V_norm", "A_dispersion", "A_concentration",
              "Oattn_norm", "Oattn_delta_norm", "probability_entropy", "probability_top2_gap", "logit_norm"]
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


def route_full_values(prop, exclude_family=None, only_seed=None):
    fname = TENSOR_ROUTE_FILES[prop]
    rows = load_route_file(fname)
    vals = []
    for row in rows:
        if row["group"] != "ROUTE_FULL" or int(row["offset"]) != 0 or row["available"] != "True":
            continue
        run_id = row["run_id"]
        if exclude_family is not None and run_meta[run_id]["prompt_family"] == exclude_family:
            continue
        if only_seed is not None and run_meta[run_id]["seed"] != only_seed:
            continue
        v = row.get(prop)
        if v is not None and v != "":
            vals.append(float(v))
    return vals


def no_landmark_values(prop):
    vals = []
    for run_id in GROUPS["NO_LANDMARK"]:
        for layer in ROUTE_LAYERS:
            key = f"{run_id}|{layer}"
            if key in means_cache and prop in means_cache[key] and means_cache[key][prop] is not None:
                vals.append(means_cache[key][prop])
    return vals


# ============================================================
# ROUTE_PROPERTY status classification
# ============================================================
comp_groups = ["L0_L5_ONLY", "L0_L20_ONLY", "OTHER_LANDMARK_ROUTE", "NO_LANDMARK"]
property_status = {}
for prop in PROPERTIES:
    entries = {g: comparisons.get(f"{prop}|{g}") for g in comp_groups}
    evaluable = {g: e for g, e in entries.items() if e is not None}
    if not evaluable:
        property_status[prop] = {"status": "NOT_EVALUABLE", "reason": "insufficient n in all comparisons"}
        continue
    signs = set()
    for e in evaluable.values():
        diff = e["median_a"] - e["median_b"]
        signs.add(1 if diff > 0 else (-1 if diff < 0 else 0))
    consistent_direction = len(signs - {0}) <= 1
    nl_sig = evaluable.get("NO_LANDMARK", {}).get("holm_adjusted_p", 1.0) < 0.05
    other_sig = evaluable.get("OTHER_LANDMARK_ROUTE", {}).get("holm_adjusted_p", 1.0) < 0.05
    any_raw_sig = any(e["p_raw"] < 0.05 for e in evaluable.values())
    if consistent_direction and nl_sig and other_sig:
        status = "ROUTE_PROPERTY_SUPPORTED"
    elif any_raw_sig:
        status = "ROUTE_PROPERTY_LOW_SUPPORT"
    else:
        status = "ROUTE_PROPERTY_NOT_ESTABLISHED"
    property_status[prop] = {"status": status, "comparisons": evaluable, "consistent_direction": consistent_direction}

print("Property status (pre-confound):")
for prop, v in property_status.items():
    print(" ", prop, v["status"])

# ============================================================
# Family/seed confound checks for LOW_SUPPORT+ properties
# ============================================================
FAMILIES_IN_ROUTE = ["CONSTRAINT_FOLLOWING", "SUMMARIZATION", "MULTISTEP_INSTRUCTION", "GENERAL_INSTRUCTION", "RECOVERY_CORRECTION", "CODING"]
SEEDS_IN_ROUTE = ["2102", "2103"]

family_confound_results = {}
seed_robustness_results = {}
for prop, info in property_status.items():
    if info["status"] not in ("ROUTE_PROPERTY_SUPPORTED", "ROUTE_PROPERTY_LOW_SUPPORT"):
        continue
    nl_vals = no_landmark_values(prop)
    baseline_a = route_full_values(prop)
    if len(baseline_a) < 5 or len(nl_vals) < 5:
        family_confound_results[prop] = "NOT_EVALUABLE"
        seed_robustness_results[prop] = "NOT_EVALUABLE"
        continue
    baseline_stat, baseline_p = mannwhitneyu(baseline_a, nl_vals, alternative="two-sided")
    baseline_sig = baseline_p < 0.05
    flips = 0
    for fam in FAMILIES_IN_ROUTE:
        a = route_full_values(prop, exclude_family=fam)
        if len(a) < 5:
            continue
        stat, p = mannwhitneyu(a, nl_vals, alternative="two-sided")
        if baseline_sig and p >= 0.05:
            flips += 1
    family_confound_results[prop] = "FAMILY_CONFOUNDED" if flips > 0 else "STABLE"

    seed_flip = False
    seed_detail = {}
    for seed in SEEDS_IN_ROUTE:
        a = route_full_values(prop, only_seed=seed)
        if len(a) < 5:
            seed_detail[seed] = "NOT_EVALUABLE"
            continue
        stat, p = mannwhitneyu(a, nl_vals, alternative="two-sided")
        seed_detail[seed] = {"n": len(a), "p_raw": float(p), "significant": bool(p < 0.05)}
        if baseline_sig and p >= 0.05:
            seed_flip = True
    seed_robustness_results[prop] = {"per_seed": seed_detail, "label": "SEED_SENSITIVE" if seed_flip else "SEED_STABLE"}

with open(HERE / "FAMILY_CONFOUND_AUDIT.json", "w") as f:
    json.dump(family_confound_results, f, indent=2)
with open(HERE / "SEED_ROBUSTNESS_AUDIT.json", "w") as f:
    json.dump(seed_robustness_results, f, indent=2, default=str)

print("Family confound:", family_confound_results)
print("Seed robustness:", {k: v["label"] if isinstance(v, dict) else v for k, v in seed_robustness_results.items()})

# ============================================================
# LENGTH_CONFOUND_AUDIT.json
# ============================================================
def length_bin(n):
    return "SHORT_1_9" if n <= 9 else ("MEDIUM_10_49" if n <= 49 else "LONG_50_PLUS")


route_full_run_lengths = {r: length_bin(int(run_meta[r]["n_steps"])) for r in GROUPS["ROUTE_FULL"]}
route_bin_counts = Counter(route_full_run_lengths.values())
nl_run_lengths = {r: length_bin(int(run_meta[r]["n_steps"])) for r in GROUPS["NO_LANDMARK"]}
nl_bin_counts = Counter(nl_run_lengths.values())

length_confound = {
    "route_full_length_bin_distribution": dict(route_bin_counts),
    "no_landmark_length_bin_distribution": dict(nl_bin_counts),
    "route_full_concentrated_in_long": route_bin_counts.get("LONG_50_PLUS", 0) >= 0.5 * len(GROUPS["ROUTE_FULL"]),
    "note": "ROUTE_FULL's generation-length distribution (median 130 steps, min 48) is concentrated in LONG_50_PLUS/upper-MEDIUM relative to the overall Lane B population. This is expected: reaching 3 distinct ordered landmark encounters mechanically requires more decode steps than reaching 0 or 1. Length-matched sub-comparison not run for individual properties in this gate (sample too small to stratify further, n=8) -- flagged as a standing LENGTH_CONFOUND risk for any property found significant against NO_LANDMARK specifically, rather than tested exhaustively.",
    "properties_flagged_length_confound_risk": [p for p, info in property_status.items() if info["status"] in ("ROUTE_PROPERTY_SUPPORTED", "ROUTE_PROPERTY_LOW_SUPPORT")],
}
with open(HERE / "LENGTH_CONFOUND_AUDIT.json", "w") as f:
    json.dump(length_confound, f, indent=2)

print("LENGTH_CONFOUND_AUDIT.json written")

# ============================================================
# Apply confound results to final status (downgrade if confounded)
# ============================================================
final_registry = {}
for prop, info in property_status.items():
    status = info["status"]
    fam_result = family_confound_results.get(prop, "NOT_EVALUABLE")
    seed_result = seed_robustness_results.get(prop, "NOT_EVALUABLE")
    seed_label = seed_result["label"] if isinstance(seed_result, dict) else seed_result
    if status == "ROUTE_PROPERTY_SUPPORTED" and (fam_result == "FAMILY_CONFOUNDED" or seed_label == "SEED_SENSITIVE"):
        status = "ROUTE_PROPERTY_LOW_SUPPORT"
    final_registry[prop] = {
        "status": status, "family_confound": fam_result, "seed_robustness": seed_label,
        "length_confound_flagged": prop in length_confound["properties_flagged_length_confound_risk"],
        "comparisons": info.get("comparisons"),
    }

with open(HERE / "ROUTE_PROPERTY_REGISTRY.json", "w") as f:
    json.dump(final_registry, f, indent=2, default=str)

status_counts = Counter(v["status"] for v in final_registry.values())
print("\nFINAL ROUTE_PROPERTY_REGISTRY:")
for prop, v in final_registry.items():
    print(" ", prop, v["status"], "fam=", v["family_confound"], "seed=", v["seed_robustness"])
print("Status counts:", dict(status_counts))
print("PART 4 DONE")
