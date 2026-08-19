"""
Lane A acquisition audit -- pure data-quality/coverage accounting only.
No Atlas route interpretation, no landmark promotion.
"""
import hashlib
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PANEL_PATH = ROOT / "star_command_panel_v2_authoring" / "FREEZE" / "PANEL_V2_FROZEN_CANONICAL.jsonl"
GEN_PATH = HERE / "lane_a_generations.jsonl"
SEEDS = [2101, 2102, 2103]

with open(PANEL_PATH) as f:
    panel = [json.loads(l) for l in f]
panel_ids = {r["candidate_prompt_id"] for r in panel}
panel_family = {r["candidate_prompt_id"]: r["family"] for r in panel}

rows = []
with open(GEN_PATH) as f:
    for line in f:
        rows.append(json.loads(line))

# ---- attempted / success / failure ----
status_counts = Counter(r["status"] for r in rows)
n_attempted = len(rows)
n_success = status_counts.get("SUCCESS", 0)
n_fail = n_attempted - n_success

# ---- duplicate audit ----
run_id_counts = Counter(r["run_id"] for r in rows)
duplicates = {k: v for k, v in run_id_counts.items() if v > 1}

# ---- missing audit (expected 243 prompts x 3 seeds = 729 run_ids) ----
expected_run_ids = {f"BAX1_{pid}_{seed}" for pid in panel_ids for seed in SEEDS}
present_run_ids = set(run_id_counts.keys())
missing_run_ids = expected_run_ids - present_run_ids
unexpected_run_ids = present_run_ids - expected_run_ids

# ---- coverage ----
prompts_covered = {r["prompt_id"] for r in rows}
families_covered = {r["prompt_family"] for r in rows}
seeds_covered = {r["seed"] for r in rows}

by_family = Counter(r["prompt_family"] for r in rows)
by_family_success = Counter(r["prompt_family"] for r in rows if r["status"] == "SUCCESS")
by_seed = Counter(r["seed"] for r in rows)
by_seed_success = Counter(r["seed"] for r in rows if r["status"] == "SUCCESS")

# ---- length distribution (success only) ----
lengths = [r["token_count_generated"] for r in rows if r["status"] == "SUCCESS"]
short_execs = [l for l in lengths if l <= 9]

# ---- runtime / memory distribution ----
runtimes = [r["runtime_seconds"] for r in rows if r["status"] == "SUCCESS" and r.get("runtime_seconds") is not None]
mems = [r["peak_memory_mb"] for r in rows if r["status"] == "SUCCESS" and r.get("peak_memory_mb") is not None]

# ---- probability capture completeness ----
prob_complete = 0
prob_incomplete = 0
for r in rows:
    if r["status"] != "SUCCESS":
        continue
    pss = r.get("per_step_probability_summary")
    if pss and len(pss.get("entropy", [])) == r["token_count_generated"] and len(pss.get("top1_probability", [])) == r["token_count_generated"] and len(pss.get("top2_gap", [])) == r["token_count_generated"]:
        prob_complete += 1
    else:
        prob_incomplete += 1

# ---- completion reason breakdown ----
completion_reasons = Counter(r.get("completion_reason") for r in rows if r["status"] == "SUCCESS")

# ---- hashes ----
gen_file_sha256 = hashlib.sha256(GEN_PATH.read_bytes()).hexdigest()
per_run_hash_present = sum(1 for r in rows if r.get("hashes", {}).get("output_text_sha256"))

audit = {
    "status": "EXECUTED",
    "attempted": n_attempted,
    "expected": 729,
    "success": n_success,
    "failures": n_fail,
    "success_rate_pct": round(100 * n_success / n_attempted, 3) if n_attempted else None,
    "status_breakdown": dict(status_counts),
    "prompt_coverage": {"n_prompts_covered": len(prompts_covered), "n_prompts_expected": len(panel_ids), "missing_prompt_ids": sorted(panel_ids - prompts_covered)},
    "family_coverage": {"n_families_covered": len(families_covered), "n_families_expected": 12, "by_family_attempted": dict(by_family), "by_family_success": dict(by_family_success)},
    "seed_coverage": {"n_seeds_covered": len(seeds_covered), "seeds_expected": SEEDS, "by_seed_attempted": dict(by_seed), "by_seed_success": dict(by_seed_success)},
    "duplicate_run_audit": {"n_duplicate_run_ids": len(duplicates), "duplicates": duplicates},
    "missing_run_audit": {"n_missing": len(missing_run_ids), "missing_run_ids": sorted(missing_run_ids), "n_unexpected": len(unexpected_run_ids), "unexpected_run_ids": sorted(unexpected_run_ids)},
    "generation_length_distribution": {
        "n_success_rows": len(lengths),
        "min": min(lengths) if lengths else None, "median": st.median(lengths) if lengths else None,
        "max": max(lengths) if lengths else None, "mean": round(st.mean(lengths), 2) if lengths else None,
        "stdev": round(st.stdev(lengths), 2) if len(lengths) > 1 else None,
    },
    "short_execution_count": {"n_short_1_to_9_tokens": len(short_execs), "definition": "SHORT_EXECUTION per SHORT_EXECUTION_POLICY.md = 1-9 generated tokens, never labeled NON_ANALYZABLE"},
    "runtime_distribution_seconds": {
        "min": round(min(runtimes), 3) if runtimes else None, "median": round(st.median(runtimes), 3) if runtimes else None,
        "max": round(max(runtimes), 3) if runtimes else None, "mean": round(st.mean(runtimes), 3) if runtimes else None,
        "total": round(sum(runtimes), 1) if runtimes else None,
    },
    "memory_distribution_mb": {
        "n_with_memory_captured": len(mems),
        "min": round(min(mems), 1) if mems else None, "median": round(st.median(mems), 1) if mems else None,
        "max": round(max(mems), 1) if mems else None,
    },
    "probability_capture_completeness": {"complete": prob_complete, "incomplete": prob_incomplete, "status": "YES" if prob_incomplete == 0 and prob_complete == n_success else ("PARTIAL" if prob_complete > 0 else "NO")},
    "completion_reason_breakdown": dict(completion_reasons),
    "hashes": {"lane_a_generations_jsonl_sha256": gen_file_sha256, "n_rows_with_output_text_hash": per_run_hash_present},
}

with open(HERE / "LANE_A_ACQUISITION_AUDIT.json", "w") as f:
    json.dump(audit, f, indent=2)

# ============================================================
# Lane B structural candidate identification -- STRUCTURAL ONLY
# (family, length bucket, coverage/diversity) -- NEVER outcome/landmark/entropy-based
# ============================================================
per_prompt_lengths = defaultdict(list)
for r in rows:
    if r["status"] == "SUCCESS":
        per_prompt_lengths[r["prompt_id"]].append(r["token_count_generated"])

prompt_mean_length = {pid: st.mean(v) for pid, v in per_prompt_lengths.items()}

family_to_prompts = defaultdict(list)
for pid in panel_ids:
    family_to_prompts[panel_family[pid]].append(pid)

lane_b_candidates = {}
for family, pids in family_to_prompts.items():
    pids_with_len = [(pid, prompt_mean_length.get(pid)) for pid in pids if pid in prompt_mean_length]
    pids_with_len.sort(key=lambda x: x[1])
    n = len(pids_with_len)
    if n == 0:
        lane_b_candidates[family] = {"selected": [], "note": "no successful generations for this family"}
        continue
    # structural rule: pick 4 prompts spanning this family's OWN observed length range (min, ~33%, ~66%, max) -- purely positional, not outcome-based
    idxs = sorted(set([0, max(0, n // 3), max(0, (2 * n) // 3), n - 1]))
    while len(idxs) < min(4, n):
        for cand in range(n):
            if cand not in idxs:
                idxs.append(cand)
                break
        idxs = sorted(set(idxs))
    idxs = idxs[:4]
    selected = [{"prompt_id": pids_with_len[i][0], "mean_generated_tokens": round(pids_with_len[i][1], 1)} for i in idxs]
    lane_b_candidates[family] = {"selected": selected, "n_family_prompts_with_data": n}

with open(HERE / "LANE_B_STRUCTURAL_CANDIDATES.json", "w") as f:
    json.dump({
        "status": "STRUCTURAL_CANDIDATE_LIST -- NOT a Lane B authorization or final selection",
        "selection_method": "For each of the 12 families, rank that family's own prompts by mean observed generated-token-count across the 3 seeds (structural, not outcome-based), then pick up to 4 prompts spanning the family's own min/lower-third/upper-third/max positions in that ranking. No landmark occurrence, entropy, effect size, or output quality used anywhere in this selection.",
        "n_families": len(lane_b_candidates),
        "total_candidates": sum(len(v["selected"]) for v in lane_b_candidates.values()),
        "by_family": lane_b_candidates,
    }, f, indent=2)

print(json.dumps(audit, indent=2))
print("\nLane B structural candidates written -- total:", sum(len(v["selected"]) for v in lane_b_candidates.values()))
