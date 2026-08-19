"""
Reference Route Property Profile V0 -- part 1: group construction, route occurrence
profile, local-window extraction (reused from atlas-coverage-analysis-v1's already
-computed per-tensor CSVs, no npz reload).
"""
import csv
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
NET_V1 = HERE.parent / "reference-landmark-network-v1"
ROUTE_V0 = HERE.parent / "composite-reference-route-v0"
COVERAGE = HERE.parent / "atlas-coverage-analysis-v1"
NODE_LAYER = {"L0": 0, "L1": 1, "L5": 5, "L20": 20}

with open(ROUTE_V0 / "OBS_CACHE.json") as f:
    route_obs = json.load(f)
with open(NET_V1 / "RELATIONS_CACHE.json") as f:
    net_cache = json.load(f)
with open(COVERAGE / "RUN_ELIGIBILITY_AUDIT.csv") as f:
    elig_rows = list(csv.DictReader(f))
with open(COVERAGE / "PASS1_CACHE.json") as f:
    events = json.load(f)["events"]

run_meta = {r["run_id"]: r for r in elig_rows}
run_raw_events = defaultdict(list)
for ev in events:
    run_raw_events[ev["run_id"]].append(ev)

ROUTE_FULL = set(route_obs["observed_support_runs"]["L0-L5-L20"])
l0_l5 = set(net_cache["observed_direct"].get("L0|L5", [])) | set(net_cache["observed_indirect"].get("L0|L5", []))
l0_l20 = set(net_cache["observed_direct"].get("L0|L20", [])) | set(net_cache["observed_indirect"].get("L0|L20", []))
landmark_bearing = set(net_cache["run_groups"].keys())

L0_L5_ONLY = (l0_l5 - l0_l20) - ROUTE_FULL
L0_L20_ONLY = (l0_l20 - l0_l5) - ROUTE_FULL
OTHER_LANDMARK_ROUTE = landmark_bearing - ROUTE_FULL - L0_L5_ONLY - L0_L20_ONLY
NO_LANDMARK = {r["run_id"] for r in elig_rows if r["eligibility"] == "ELIGIBLE" and r["n_landmarks_observed"] == "0"}
SHORT_NON_EVALUABLE = {r["run_id"] for r in elig_rows if r["eligibility"] != "ELIGIBLE"}

GROUPS = {"ROUTE_FULL": ROUTE_FULL, "L0_L5_ONLY": L0_L5_ONLY, "L0_L20_ONLY": L0_L20_ONLY,
          "OTHER_LANDMARK_ROUTE": OTHER_LANDMARK_ROUTE, "NO_LANDMARK": NO_LANDMARK,
          "SHORT_NON_EVALUABLE": SHORT_NON_EVALUABLE}

total_partitioned = sum(len(v) for v in GROUPS.values())
assert total_partitioned == 144, f"partition check failed: {total_partitioned} != 144"
assert len(set.union(*GROUPS.values())) == 144, "overlap detected in group partition"

registry = {"status": "FROZEN_POST_CONSTRUCTION", "groups": {k: {"n_runs": len(v), "run_ids": sorted(v)} for k, v in GROUPS.items()},
            "partition_verified": True, "total_runs": 144}
with open(HERE / "ROUTE_GROUP_REGISTRY.json", "w") as f:
    json.dump(registry, f, indent=2)

for k, v in GROUPS.items():
    print(k, len(v))
print("ROUTE_GROUP_REGISTRY.json written, partition verified")

# ============================================================
# ROUTE_OCCURRENCE_PROFILE.json
# ============================================================
def first_step(run_id, node):
    steps = [e["step"] for e in run_raw_events[run_id] if e["node"] == node]
    return min(steps) if steps else None


families = Counter(run_meta[r]["prompt_family"] for r in ROUTE_FULL)
seeds = Counter(run_meta[r]["seed"] for r in ROUTE_FULL)
prompts = sorted({run_meta[r]["prompt_id"] for r in ROUTE_FULL})
lengths = sorted(int(run_meta[r]["n_steps"]) for r in ROUTE_FULL)

step_positions = {"L0": [], "L5": [], "L20": []}
distances = {"L0_L5": [], "L5_L20": [], "L0_L20": []}
route_first_steps = {}
for r in ROUTE_FULL:
    s0, s5, s20 = first_step(r, "L0"), first_step(r, "L5"), first_step(r, "L20")
    step_positions["L0"].append(s0)
    step_positions["L5"].append(s5)
    step_positions["L20"].append(s20)
    distances["L0_L5"].append(s5 - s0)
    distances["L5_L20"].append(s20 - s5)
    distances["L0_L20"].append(s20 - s0)
    route_first_steps[r] = {"L0": s0, "L5": s5, "L20": s20}


def dist_summary(vals):
    vals_sorted = sorted(vals)
    return {"min": min(vals), "median": st.median(vals), "q1": vals_sorted[len(vals_sorted) // 4],
            "q3": vals_sorted[(3 * len(vals_sorted)) // 4], "max": max(vals)}


occurrence_profile = {
    "run_count": len(ROUTE_FULL), "prompt_count": len(prompts), "prompt_ids": prompts,
    "family_breadth": len(families), "families": dict(families),
    "seed_breadth": len(seeds), "seeds": dict(seeds),
    "generation_length_distribution": {"min": min(lengths), "median": st.median(lengths), "max": max(lengths)},
    "step_positions": {k: sorted(v) for k, v in step_positions.items()},
    "step_distance_distributions": {k: dist_summary(v) for k, v in distances.items()},
    "per_run_first_steps": route_first_steps,
}
with open(HERE / "ROUTE_OCCURRENCE_PROFILE.json", "w") as f:
    json.dump(occurrence_profile, f, indent=2, default=str)

print("ROUTE_OCCURRENCE_PROFILE.json written")

with open(HERE / "GROUPS_CACHE.json", "w") as f:
    json.dump({k: sorted(v) for k, v in GROUPS.items()}, f)
with open(HERE / "ROUTE_FIRST_STEPS_CACHE.json", "w") as f:
    json.dump(route_first_steps, f)

print("PART 1 DONE")
