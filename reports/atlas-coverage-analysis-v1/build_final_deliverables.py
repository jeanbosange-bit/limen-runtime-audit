"""
Atlas Coverage Analysis V1 -- final deliverables. Uses only already-computed
outputs (RUN_ELIGIBILITY_AUDIT.csv, PASS2_SERIES_SUMMARY_CACHE.json, etc.) -- no
npz reload.
"""
import csv
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

with open(HERE / "RUN_ELIGIBILITY_AUDIT.csv") as f:
    rows = list(csv.DictReader(f))
with open(HERE / "PASS2_SERIES_SUMMARY_CACHE.json") as f:
    series_summary = json.load(f)["per_run_layer_means"]
with open(HERE / "LANDMARK_REPLICATION_RESULTS.json") as f:
    replication = json.load(f)
with open(HERE / "FAMILY_COVERAGE_MATRIX.csv") as f:
    family_matrix = list(csv.DictReader(f))
with open(HERE / "ROUTE_SIGNATURES.csv") as f:
    route_sigs = list(csv.DictReader(f))

NODES = ["L0", "L1", "L5", "L20"]
NODE_LAYER = {"L0": 0, "L1": 1, "L5": 5, "L20": 20}

# ============================================================
# NO_LANDMARK_TERRITORY_AUDIT.json
# ============================================================
no_lm_rows = [r for r in rows if r["eligibility"] == "ELIGIBLE" and r["n_landmarks_observed"] == "0"]
lm_rows = [r for r in rows if r["eligibility"] == "ELIGIBLE" and r["n_landmarks_observed"] != "0"]

def agg_series_for_rows(run_rows, layer):
    vals = defaultdict(list)
    for r in run_rows:
        key = f"{r['run_id']}|{layer}"
        m = series_summary.get(key)
        if not m:
            continue
        for k, v in m.items():
            if v is not None:
                vals[k].append(v)
    return {k: {"mean": round(st.mean(v), 5), "n": len(v)} for k, v in vals.items() if v}


no_lm_by_family = Counter(r["prompt_family"] for r in no_lm_rows)
no_lm_length_stats = [int(r["n_steps"]) for r in no_lm_rows]
lm_length_stats = [int(r["n_steps"]) for r in lm_rows]

no_landmark_audit = {
    "definition": "ELIGIBLE (n_steps>=4) runs with zero encounters at any of L0/L1/L5/L20 -- distinct from LANDMARK_NOT_EVALUABLE_SHORT_EXECUTION (see SHORT_EXECUTION_TERRITORY_AUDIT.json).",
    "n_no_landmark_eligible": len(no_lm_rows),
    "n_landmark_eligible": len(lm_rows),
    "by_family": dict(no_lm_by_family),
    "generation_length": {
        "no_landmark_runs": {"min": min(no_lm_length_stats), "median": st.median(no_lm_length_stats), "max": max(no_lm_length_stats), "mean": round(st.mean(no_lm_length_stats), 2)},
        "landmark_runs": {"min": min(lm_length_stats), "median": st.median(lm_length_stats), "max": max(lm_length_stats), "mean": round(st.mean(lm_length_stats), 2)},
    },
    "descriptive_local_properties_at_layer_0": {
        "no_landmark_runs": agg_series_for_rows(no_lm_rows, 0),
        "landmark_runs": agg_series_for_rows(lm_rows, 0),
    },
    "primary_question": "ARE THESE RUNS SIMPLY UNDER-SAMPLED, OR DO THEY OCCUPY DESCRIPTIVELY DIFFERENT REGIONS?",
    "answer": "NOT_ESTABLISHED",
    "reasoning": "No-landmark runs are on average shorter (mean {:.1f} vs {:.1f} steps) than landmark runs, which is consistent with (but does not prove) under-sampling: fewer decode steps means fewer opportunities for a within-run z-score to cross the |z|>1.5/3-of-4-signals bar, independent of any real difference in the underlying region. A formal test distinguishing 'shorter => less likely to trip the detector by chance' from 'these runs are in a genuinely different regime' was not run in this gate (would require a length-matched or length-controlled comparison, not attempted here). LANDMARK_VS_NONLANDMARK_ANALYSIS.json's step-level comparison is the closest available evidence but compares STEPS, not RUNS, and is already disclosed there as descriptive/exploratory only.".format(st.mean(no_lm_length_stats), st.mean(lm_length_stats)),
}
with open(HERE / "NO_LANDMARK_TERRITORY_AUDIT.json", "w") as f:
    json.dump(no_landmark_audit, f, indent=2)

# ============================================================
# SHORT_EXECUTION_TERRITORY_AUDIT.json
# ============================================================
short_rows = [r for r in rows if int(r["n_steps"]) <= 9]
short_eval = [r for r in short_rows if r["eligibility"] == "ELIGIBLE"]
short_noneval = [r for r in short_rows if r["eligibility"] != "ELIGIBLE"]
short_by_family = Counter(r["prompt_family"] for r in short_rows)

short_audit = {
    "n_short_executions_1_to_9_tokens": len(short_rows),
    "n_evaluable_4_to_9_tokens": len(short_eval),
    "n_not_evaluable_1_to_3_tokens": len(short_noneval),
    "by_family": dict(short_by_family),
    "tensor_completeness": "COMPLETE for all 21 short runs (H/Q/K/V/ATTENTION/ATTENTION_OUTPUT/LOGITS/TOKEN_IDS) -- per reports/bcc-atlas-expansion-v1/ATLAS_DEEP/TENSOR_AVAILABILITY_MATRIX.csv, full native-resolution capture regardless of generation length",
    "what_can_be_evaluated": [
        "Raw tensor norms (H/Q/K/V/Oattn) at every captured step -- always available regardless of length.",
        "Attention dispersion/concentration at every captured step.",
        "Probability entropy/top2-gap/logit-norm at every captured step.",
        "Landmark z-score detection for the 4-9 token subset (21-17=4 runs... see n_evaluable above) -- the within-run z-score needs >=4 steps to have any variance to standardize against.",
    ],
    "what_cannot_be_evaluated": [
        "Landmark z-score detection for the 1-3 token subset (LANDMARK_NOT_EVALUABLE_SHORT_EXECUTION) -- mathematically undefined with <4 samples for a per-run z-score.",
        "Any multi-step trend (curvature needs >=3 points, delta needs >=2) is degenerate or unavailable for the shortest runs.",
    ],
    "alternate_short_run_descriptors_possible": "Single-step (not delta/trend-based) descriptors -- raw norm, raw entropy, raw attention concentration AT the one or two available steps -- remain well-defined even for 1-2 token runs. These are NOT converted into a landmark definition in this gate (mission section 17 explicit prohibition) but are flagged as a possible basis for a future short-execution-specific descriptive scheme.",
    "no_landmark_definition_created_for_short_runs": True,
}
with open(HERE / "SHORT_EXECUTION_TERRITORY_AUDIT.json", "w") as f:
    json.dump(short_audit, f, indent=2)

# ============================================================
# FACTUAL_MATH_SUPPORT_AUDIT.json
# ============================================================
def family_support_detail(family):
    fam_rows = [r for r in rows if r["prompt_family"] == family]
    supporting = [r for r in fam_rows if r["n_landmarks_observed"] not in ("0", "")]
    prompt_ids = {r["prompt_id"] for r in supporting}
    seeds = {r["seed"] for r in supporting}
    nodes_involved = set()
    for r in supporting:
        nodes_involved.update(r["landmarks"].split("|"))
    n_supporting = len(supporting)
    if n_supporting == 0:
        verdict = "NOT_ESTABLISHED"
        reason = "zero supporting runs"
    elif n_supporting == 1:
        verdict = "NONZERO_SUPPORT_ONLY"
        reason = "exactly 1 supporting run"
    elif len(prompt_ids) == 1 or len(nodes_involved) == 1:
        verdict = "LOW_SUPPORT_REPLICATION_CANDIDATE"
        reason = f">=2 supporting runs but confined to {'a single prompt_id' if len(prompt_ids)==1 else 'a single landmark node'}"
    elif len(prompt_ids) >= 2 and len(seeds) >= 2:
        verdict = "PARTIAL_REPLICATION"
        reason = ">=2 supporting runs spanning >=2 prompt_ids and >=2 seeds"
    else:
        verdict = "NOT_ESTABLISHED"
        reason = "ambiguous evidence pattern"
    return {
        "n_eligible_runs": sum(1 for r in fam_rows if r["eligibility"] == "ELIGIBLE"),
        "n_supporting_runs": n_supporting,
        "supporting_run_ids": [r["run_id"] for r in supporting],
        "supporting_prompt_ids": sorted(prompt_ids), "n_distinct_prompt_ids": len(prompt_ids),
        "supporting_seeds": sorted(seeds), "n_distinct_seeds": len(seeds),
        "nodes_involved": sorted(nodes_involved),
        "verdict": verdict, "reason": reason,
    }

factual_math_audit = {
    "context": "V0's old 8-step-window rule gave FACTUAL_GROUNDED and MATH_REASONING effectively zero usable landmark support (reports/reference-landmark-network-v0). Lane B's native full-sequence capture removed that structural exclusion.",
    "FACTUAL_GROUNDED": family_support_detail("FACTUAL_GROUNDED"),
    "MATH_REASONING": family_support_detail("MATH_REASONING"),
    "classification": "NEW_NONZERO_OBSERVATION, not replication yet -- per mission section 18 explicit instruction not to overclaim.",
}
with open(HERE / "FACTUAL_MATH_SUPPORT_AUDIT.json", "w") as f:
    json.dump(factual_math_audit, f, indent=2)

print("FACTUAL_GROUNDED:", factual_math_audit["FACTUAL_GROUNDED"]["verdict"])
print("MATH_REASONING:", factual_math_audit["MATH_REASONING"]["verdict"])

# ============================================================
# COVERAGE_HOLES.csv
# ============================================================
holes = []
for fr in family_matrix:
    fam = fr["family"]
    n_elig = int(fr["n_eligible"])
    n_short = int(fr["n_short_noneval"])
    if n_short > 0:
        holes.append({"axis": "family", "value": fam, "gap": f"{n_short} short/non-evaluable runs", "support_count": n_short, "rank_basis": "LOW_SUPPORT"})
    if n_elig <= 7:
        holes.append({"axis": "family", "value": fam, "gap": "few eligible runs for landmark analysis", "support_count": n_elig, "rank_basis": "LOW_SUPPORT"})

holes.append({"axis": "layer", "value": "1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23 (20 of 24 mechanism layers)", "gap": "no landmark analysis performed -- frozen node set covers only L0/L1/L5/L20 by design", "support_count": 0, "rank_basis": "UNKNOWN_BY_DESIGN"})

for rt in route_sigs:
    if rt["category"] in ("UNIQUE_ROUTE", "LOW_SUPPORT_ROUTE"):
        holes.append({"axis": "route_signature", "value": rt["route_signature"], "gap": f"{rt['category']}, n_runs={rt['n_runs']}", "support_count": int(rt["n_runs"]), "rank_basis": "LOW_SUPPORT"})

holes.append({"axis": "tensor_at_length", "value": "SHORT_EXECUTION (1-3 tokens)", "gap": "landmark detection structurally undefined (n<4)", "support_count": 17, "rank_basis": "UNKNOWN_BY_DESIGN"})

holes.sort(key=lambda h: h["support_count"])
with open(HERE / "COVERAGE_HOLES.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["axis", "value", "gap", "support_count", "rank_basis"])
    w.writeheader()
    for h in holes:
        w.writerow(h)

print(f"Coverage holes: {len(holes)} entries")

# ============================================================
# ATLAS_COVERAGE_MAP.json
# ============================================================
def classify_support(n_runs, n_prompts, n_seeds):
    if n_runs >= 3 and n_prompts >= 2 and n_seeds >= 2:
        return "WELL_OBSERVED"
    elif n_runs >= 2:
        return "MODERATE_SUPPORT"
    elif n_runs == 1:
        return "LOW_SUPPORT"
    else:
        return "UNKNOWN"

family_support_map = {}
for fr in family_matrix:
    family_support_map[fr["family"]] = classify_support(int(fr["n_eligible"]), int(fr["n_prompt_ids"]), int(fr["n_seeds"]))

coverage_map = {
    "thresholds": "reports/atlas-coverage-analysis-v1/SUPPORT_THRESHOLD_FREEZE.json (frozen before inspection)",
    "by_prompt_family": family_support_map,
    "by_landmark_node": {node: replication["results"][node]["verdict"] for node in NODES},
    "by_layer_for_landmark_purposes": {**{str(layer): replication["results"][node]["verdict"] for node, layer in NODE_LAYER.items()},
                                          "other_20_layers": "UNKNOWN (not tested -- frozen node set, mission section 3)"},
    "by_generation_length_regime": "see LANDMARK_RECURRENCE_AUDIT (Lane B) recurrence_by_execution_length_bucket for SHORT/MEDIUM/LONG bucket support",
    "tensor_type_coverage": {"H": "WELL_OBSERVED", "Q": "WELL_OBSERVED (first time captured)", "K": "WELL_OBSERVED", "V": "WELL_OBSERVED",
                              "ATTENTION": "WELL_OBSERVED", "ATTENTION_OUTPUT": "WELL_OBSERVED", "LOGITS": "WELL_OBSERVED (bounded, predictive positions only)"},
}
with open(HERE / "ATLAS_COVERAGE_MAP.json", "w") as f:
    json.dump(coverage_map, f, indent=2)

print("ATLAS_COVERAGE_MAP.json written")
print("FINAL DELIVERABLES PART 1 DONE")
