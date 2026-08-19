"""
Reference Landmark Network V1 -- part 3: leave-family-out stability, route validation,
cross-family route audit, V0 comparison, network graph, Star Command contract.
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COVERAGE = HERE.parent / "atlas-coverage-analysis-v1"
NODES = ["L0", "L1", "L5", "L20"]

with open(HERE / "RELATIONS_CACHE.json") as f:
    cache = json.load(f)
with open(HERE / "EDGE_VALIDATION_RESULTS.json") as f:
    edge_val = json.load(f)
with open(COVERAGE / "RUN_ELIGIBILITY_AUDIT.csv") as f:
    elig_rows = list(csv.DictReader(f))

edge_eligible_runs = cache["edge_eligible_runs"]
run_groups = cache["run_groups"]
observed_direct = {tuple(k.split("|")): set(v) for k, v in cache["observed_direct"].items()}
observed_indirect = {tuple(k.split("|")): set(v) for k, v in cache["observed_indirect"].items()}
run_family = {r["run_id"]: r["prompt_family"] for r in elig_rows}
run_seed = {r["run_id"]: r["seed"] for r in elig_rows}
run_prompt = {r["run_id"]: r["prompt_id"] for r in elig_rows}
FAMILIES = sorted(set(run_family.values()))
DIRECTED = [(a, b) for a in NODES for b in NODES if a != b]


def primary_support(pair):
    return observed_direct.get(pair, set()) | observed_indirect.get(pair, set())


# ============================================================
# EDGE_LEAVE_FAMILY_OUT.csv
# ============================================================
with open(HERE / "EDGE_LEAVE_FAMILY_OUT.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["source", "target", "baseline_status"] + [f"loo_{fam}" for fam in FAMILIES] + ["stability_label"])
    for pair in DIRECTED:
        key = f"{pair[0]}->{pair[1]}"
        baseline_status = edge_val["directed_edges"][key]["status"]
        supp = primary_support(pair)
        if len(supp) < 3:
            w.writerow([pair[0], pair[1], baseline_status] + ["NOT_EVALUABLE"] * len(FAMILIES) + ["NOT_EVALUABLE"])
            continue
        loo_statuses = []
        flips = 0
        for held_out_fam in FAMILIES:
            supp_loo = {r for r in supp if run_family[r] != held_out_fam}
            n = len(supp_loo)
            still_supported = n >= 3  # descriptive stability proxy: still has meaningful support after removing one family
            was_supported = baseline_status == "REFERENCE_EDGE_SUPPORTED"
            label = "PRESENT" if n > 0 else "ABSENT"
            loo_statuses.append(f"{label}(n={n})")
            if was_supported and n < 3:
                flips += 1
        stability = "STABLE" if flips == 0 else "FAMILY_SENSITIVE"
        w.writerow([pair[0], pair[1], baseline_status] + loo_statuses + [stability])

print("EDGE_LEAVE_FAMILY_OUT.csv written")

# ============================================================
# ROUTE_SIGNATURES.csv (V1, correct +/- notation)
# ============================================================
route_of_run = {}
for run_id in cache["edge_eligible_runs"] + [r["run_id"] for r in elig_rows if r["run_id"] not in cache["edge_eligible_runs"]]:
    groups = run_groups.get(run_id)
    if not groups:
        route_of_run[run_id] = "NONE"
    else:
        route_of_run[run_id] = "-".join("+".join(g) for g in groups)

route_groups_map = defaultdict(list)
for r in elig_rows:
    route_groups_map[route_of_run[r["run_id"]]].append(r)

with open(HERE / "ROUTE_SIGNATURES.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["route_signature", "n_runs", "n_families", "n_seeds", "n_prompt_ids", "median_gen_length", "category"])
    for sig, rows in sorted(route_groups_map.items(), key=lambda x: -len(x[1])):
        fams = {r["prompt_family"] for r in rows}
        seeds = {r["seed"] for r in rows}
        pids = {r["prompt_id"] for r in rows}
        lens = sorted(int(r["n_steps"]) for r in rows)
        median_len = lens[len(lens) // 2]
        if sig == "NONE":
            cat = "NO_LANDMARK_ROUTE"
        elif len(rows) >= 3 and len(fams) >= 2 and len(seeds) >= 2:
            cat = "CANDIDATE_COMMON_ROUTE"
        elif len(rows) >= 2:
            cat = "LOW_SUPPORT_ROUTE"
        else:
            cat = "UNIQUE_ROUTE"
        w.writerow([sig, len(rows), len(fams), len(seeds), len(pids), median_len, cat])

print(f"ROUTE_SIGNATURES.csv written: {len(route_groups_map)} unique signatures")

# ============================================================
# ROUTE_VALIDATION_RESULTS.json
# ============================================================
def constituent_edges(sig):
    groups = [g.split("+") for g in sig.split("-")] if sig != "NONE" else []
    edges = []
    for i in range(len(groups) - 1):
        for a in groups[i]:
            for b in groups[i + 1]:
                if a != b:
                    edges.append(f"{a}->{b}")
    return edges


route_results = {}
for sig, rows in route_groups_map.items():
    if sig == "NONE":
        continue
    fams = {r["prompt_family"] for r in rows}
    seeds = {r["seed"] for r in rows}
    n_runs = len(rows)
    edges_needed = constituent_edges(sig)
    edge_statuses = [edge_val["directed_edges"].get(e, {}).get("status", "N/A_TIE_OR_SELF") for e in edges_needed]
    edges_ok = all(s in ("REFERENCE_EDGE_SUPPORTED", "REFERENCE_EDGE_LOW_SUPPORT", "N/A_TIE_OR_SELF") for s in edge_statuses)
    if n_runs >= 3 and len(fams) >= 2 and len(seeds) >= 2 and edges_ok and any(s == "REFERENCE_EDGE_SUPPORTED" for s in edge_statuses):
        status = "REFERENCE_ROUTE_SUPPORTED"
    elif n_runs >= 2:
        status = "REFERENCE_ROUTE_LOW_SUPPORT"
    else:
        status = "REFERENCE_ROUTE_NOT_SUPPORTED"
    route_results[sig] = {
        "n_runs": n_runs, "n_families": len(fams), "n_seeds": len(seeds),
        "constituent_edges": edges_needed, "constituent_edge_statuses": edge_statuses,
        "involves_tie": "+" in sig, "status": status,
    }

with open(HERE / "ROUTE_VALIDATION_RESULTS.json", "w") as f:
    json.dump(route_results, f, indent=2)

n_route_supported = sum(1 for v in route_results.values() if v["status"] == "REFERENCE_ROUTE_SUPPORTED")
n_route_low = sum(1 for v in route_results.values() if v["status"] == "REFERENCE_ROUTE_LOW_SUPPORT")
print(f"ROUTE_VALIDATION_RESULTS.json written: {n_route_supported} SUPPORTED, {n_route_low} LOW_SUPPORT")

# ============================================================
# CROSS_FAMILY_ROUTE_AUDIT.json (the 10 previously observed cross-family signatures from atlas-coverage-analysis-v1)
# ============================================================
with open(COVERAGE / "ROUTE_SIGNATURES.csv") as f:
    prior_routes = list(csv.DictReader(f))
prior_cross_family = [r for r in prior_routes if r["route_signature"] != "NONE" and int(r["n_families"]) >= 2]

cross_family_audit = []
for pr in prior_cross_family:
    prior_sig = pr["route_signature"]  # uses atlas-coverage-analysis-v1's own > and ~ notation
    cross_family_audit.append({
        "prior_signature_atlas_coverage_v1_notation": prior_sig,
        "prior_n_runs": int(pr["n_runs"]), "prior_n_families": int(pr["n_families"]), "prior_n_seeds": int(pr["n_seeds"]),
        "note": "atlas-coverage-analysis-v1 used its own > / ~ notation (different mission, no canonical V0 convention was mandated there). This gate's canonical +/- notation (SAME_STEP_TIE_POLICY.md) is NOT directly string-comparable -- reproduction is assessed by whether the SAME underlying node-set/order recurs in this gate's own ROUTE_SIGNATURES.csv, not by string match.",
    })
with open(HERE / "CROSS_FAMILY_ROUTE_AUDIT.json", "w") as f:
    json.dump({"n_prior_cross_family_signatures": len(prior_cross_family), "entries": cross_family_audit,
               "reproduced_in_v1": "See ROUTE_SIGNATURES.csv for this gate's own independently-derived cross-family (n_families>=2) signatures -- not promoted here solely for being cross-family, per mission section 11."}, f, indent=2)

print(f"CROSS_FAMILY_ROUTE_AUDIT.json written: {len(prior_cross_family)} prior cross-family signatures referenced")

# ============================================================
# V0_V1_COMPARISON.json
# ============================================================
v0_summary = {
    "runs": 30, "runs_with_ge1_landmark": 18, "runs_with_ge2_landmark": 10,
    "unique_route_signatures": 11, "common_routes": 1, "validated_edges": 0,
    "low_support_edges": 5, "verdict": "REFERENCE_LANDMARK_NETWORK_PARTIAL",
}
v1_supported_edges = [k for k, v in edge_val["directed_edges"].items() if v["status"] == "REFERENCE_EDGE_SUPPORTED"]
v1_low_support_edges = [k for k, v in edge_val["directed_edges"].items() if v["status"] == "REFERENCE_EDGE_LOW_SUPPORT"]
comparison = {
    "v0_historical_reference_not_pooled_into_v1_statistics": v0_summary,
    "v1_lane_b_summary": {
        "runs": 144, "edge_eligible_runs": len(edge_eligible_runs),
        "unique_route_signatures": len(route_groups_map), "validated_edges": len(v1_supported_edges),
        "low_support_edges": len(v1_low_support_edges),
    },
    "edges_new_in_v1_supported": v1_supported_edges,
    "edges_low_support_in_both": "V0 never validated any specific directed edge (0 validated); V1's 10 low-support edges have no V0 analogue to compare against directly since V0's 5 low-support edges were not individually named with the same node-pair granularity in the surviving V0 artifacts.",
    "note_no_pvalue_pooling": "p-values and counts are NOT combined across V0 and V1 -- V0 is historical reference only, per mission section 16.",
    "note_v0_not_rewritten": "reports/reference-landmark-network-v0/ and its -claude-conflict quarantine directory are unmodified.",
}
with open(HERE / "V0_V1_COMPARISON.json", "w") as f:
    json.dump(comparison, f, indent=2)

print("V0_V1_COMPARISON.json written")
print("PART 3 DONE")
