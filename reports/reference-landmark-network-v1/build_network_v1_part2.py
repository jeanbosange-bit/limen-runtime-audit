"""
Reference Landmark Network V1 -- part 2: permutation testing, Holm correction,
family/seed/leave-family-out breakdowns, route validation, V0 comparison, network graph,
Star Command contract. Reuses RELATIONS_CACHE.json from part 1 (no npz reload).
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COVERAGE = HERE.parent / "atlas-coverage-analysis-v1"
NODES = ["L0", "L1", "L5", "L20"]
RNG_SEED = 20260818
N_PERM = 10000

with open(HERE / "RELATIONS_CACHE.json") as f:
    cache = json.load(f)
with open(COVERAGE / "PASS1_CACHE.json") as f:
    events = json.load(f)["events"]
with open(COVERAGE / "RUN_ELIGIBILITY_AUDIT.csv") as f:
    elig_rows = list(csv.DictReader(f))
run_meta = {r["run_id"]: r for r in elig_rows}

edge_eligible_runs = cache["edge_eligible_runs"]
run_groups = cache["run_groups"]
DIRECTED = [(a, b) for a in NODES for b in NODES if a != b]
UNDIRECTED = [tuple(sorted((a, b))) for i, a in enumerate(NODES) for b in NODES[i + 1:]]

observed_direct = {tuple(k.split("|")): set(v) for k, v in cache["observed_direct"].items()}
observed_indirect = {tuple(k.split("|")): set(v) for k, v in cache["observed_indirect"].items()}
observed_tie = {tuple(k.split("|")): set(v) for k, v in cache["observed_tie"].items()}
occ_runs = {tuple(k.split("|")): set(v) for k, v in cache["occ_runs"].items()}


def primary_support(od, oi, pair):
    return od.get(pair, set()) | oi.get(pair, set())


observed_counts = {pair: len(primary_support(observed_direct, observed_indirect, pair)) for pair in DIRECTED}
observed_cooccurrence = {pair: len(occ_runs.get(pair, set())) for pair in UNDIRECTED}


def relations_for_groups(groups):
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
                    (direct if j == i + 1 else indirect).add((a, b))
    return direct, indirect, tie


def permute_run_groups(groups, rng):
    """Shuffle which label occupies which position, preserving tie-group sizes/positions."""
    flat_labels = [lab for g in groups for lab in g]
    perm_labels = rng.permutation(flat_labels)
    new_groups = []
    idx = 0
    for g in groups:
        size = len(g)
        new_groups.append(sorted(perm_labels[idx:idx + size]))
        idx += size
    return new_groups


print(f"Running {N_PERM} permutations (RNG seed {RNG_SEED})...", flush=True)
rng = np.random.default_rng(RNG_SEED)
perm_counts_directed = {pair: np.zeros(N_PERM, dtype=int) for pair in DIRECTED}
perm_counts_undirected = {pair: np.zeros(N_PERM, dtype=int) for pair in UNDIRECTED}

for p in range(N_PERM):
    supp_direct = defaultdict(set)
    supp_indirect = defaultdict(set)
    supp_any = defaultdict(set)
    for run_id in edge_eligible_runs:
        groups = run_groups[run_id]
        pg = permute_run_groups(groups, rng)
        d, i, t = relations_for_groups(pg)
        for pair in d:
            supp_direct[pair].add(run_id)
        for pair in i:
            supp_indirect[pair].add(run_id)
        for pair in {tuple(sorted(x)) for x in (d | i | t)}:
            supp_any[pair].add(run_id)
    for pair in DIRECTED:
        perm_counts_directed[pair][p] = len(supp_direct.get(pair, set()) | supp_indirect.get(pair, set()))
    for pair in UNDIRECTED:
        perm_counts_undirected[pair][p] = len(supp_any.get(pair, set()))
    if (p + 1) % 2000 == 0:
        print(f"  {p+1}/{N_PERM} permutations done", flush=True)

print("Permutations complete.", flush=True)

# p-values (one-sided, enrichment)
raw_p = {}
for pair in DIRECTED:
    obs = observed_counts[pair]
    perm = perm_counts_directed[pair]
    raw_p[("directed", pair)] = float((np.sum(perm >= obs) + 1) / (N_PERM + 1))
for pair in UNDIRECTED:
    obs = observed_cooccurrence[pair]
    perm = perm_counts_undirected[pair]
    raw_p[("undirected", pair)] = float((np.sum(perm >= obs) + 1) / (N_PERM + 1))

# Holm correction across all 18
items = sorted(raw_p.items(), key=lambda x: x[1])
n = len(items)
holm_p = {}
running_max = 0.0
for rank, (key, p) in enumerate(items, start=1):
    adj = min(1.0, p * (n - rank + 1))
    running_max = max(running_max, adj)
    holm_p[key] = running_max

with open(HERE / "EDGE_NULL_RESULTS.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["type", "source", "target", "observed", "null_mean", "null_median", "raw_p", "holm_adjusted_p", "n_permutations"])
    for pair in DIRECTED:
        perm = perm_counts_directed[pair]
        w.writerow(["directed", pair[0], pair[1], observed_counts[pair], round(float(np.mean(perm)), 3), float(np.median(perm)),
                    raw_p[("directed", pair)], holm_p[("directed", pair)], N_PERM])
    for pair in UNDIRECTED:
        perm = perm_counts_undirected[pair]
        w.writerow(["undirected", pair[0], pair[1], observed_cooccurrence[pair], round(float(np.mean(perm)), 3), float(np.median(perm)),
                    raw_p[("undirected", pair)], holm_p[("undirected", pair)], N_PERM])

print("EDGE_NULL_RESULTS.csv written", flush=True)

# ============================================================
# Family/seed breakdown per directed edge
# ============================================================
run_family = {r["run_id"]: r["prompt_family"] for r in elig_rows}
run_seed = {r["run_id"]: r["seed"] for r in elig_rows}
run_prompt = {r["run_id"]: r["prompt_id"] for r in elig_rows}
FAMILIES = sorted(set(run_family.values()))
SEEDS = ["2101", "2102", "2103"]

with open(HERE / "EDGE_FAMILY_SUPPORT.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["source", "target"] + FAMILIES)
    for pair in DIRECTED:
        supp = primary_support(observed_direct, observed_indirect, pair)
        fam_counts = Counter(run_family[r] for r in supp)
        w.writerow([pair[0], pair[1]] + [fam_counts.get(fam, 0) for fam in FAMILIES])

with open(HERE / "EDGE_SEED_SUPPORT.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["source", "target"] + SEEDS)
    for pair in DIRECTED:
        supp = primary_support(observed_direct, observed_indirect, pair)
        seed_counts = Counter(str(run_seed[r]) for r in supp)
        w.writerow([pair[0], pair[1]] + [seed_counts.get(s, 0) for s in SEEDS])

print("Family/seed support CSVs written", flush=True)

# ============================================================
# Edge status classification
# ============================================================
edge_results = {}
for pair in DIRECTED:
    supp = primary_support(observed_direct, observed_indirect, pair)
    n_fam = len(set(run_family[r] for r in supp))
    n_seed = len(set(run_seed[r] for r in supp))
    n_prompt = len(set(run_prompt[r] for r in supp))
    obs = observed_counts[pair]
    p_holm = holm_p[("directed", pair)]
    if obs == 0:
        status = "REFERENCE_EDGE_NOT_SUPPORTED"
    elif p_holm < 0.05 and n_fam >= 2 and n_seed >= 2 and n_prompt >= 2:
        status = "REFERENCE_EDGE_SUPPORTED"
    elif p_holm >= 0.05 and not (n_fam >= 2 and n_seed >= 2 and n_prompt >= 2):
        status = "REFERENCE_EDGE_NOT_SUPPORTED" if obs <= 1 else "REFERENCE_EDGE_LOW_SUPPORT"
    else:
        status = "REFERENCE_EDGE_LOW_SUPPORT"
    edge_results[f"{pair[0]}->{pair[1]}"] = {
        "source": pair[0], "target": pair[1], "observed_count": obs,
        "eligible_run_support": obs, "n_families": n_fam, "n_seeds": n_seed, "n_prompt_ids": n_prompt,
        "raw_p": raw_p[("directed", pair)], "holm_adjusted_p": p_holm,
        "n_direct": len(observed_direct.get(pair, set())), "n_indirect": len(observed_indirect.get(pair, set())),
        "status": status,
    }

co_results = {}
for pair in UNDIRECTED:
    runs = occ_runs.get(pair, set())
    n_fam = len(set(run_family[r] for r in runs))
    n_seed = len(set(run_seed[r] for r in runs))
    n_prompt = len(set(run_prompt[r] for r in runs))
    obs = observed_cooccurrence[pair]
    p_holm = holm_p[("undirected", pair)]
    if obs == 0:
        status = "REFERENCE_EDGE_NOT_SUPPORTED"
    elif p_holm < 0.05 and n_fam >= 2 and n_seed >= 2 and n_prompt >= 2:
        status = "REFERENCE_EDGE_SUPPORTED"
    else:
        status = "REFERENCE_EDGE_LOW_SUPPORT" if obs >= 2 else "REFERENCE_EDGE_NOT_SUPPORTED"
    co_results[f"{pair[0]}~{pair[1]}"] = {
        "pair": list(pair), "observed_count": obs, "n_families": n_fam, "n_seeds": n_seed, "n_prompt_ids": n_prompt,
        "raw_p": raw_p[("undirected", pair)], "holm_adjusted_p": p_holm, "status": status,
    }

with open(HERE / "EDGE_VALIDATION_RESULTS.json", "w") as f:
    json.dump({"directed_edges": edge_results, "cooccurrence_pairs": co_results,
               "correction": "Holm, family of 18 (12 directed + 6 undirected)",
               "thresholds": "reports/reference-landmark-network-v1/FROZEN_NETWORK_PROTOCOL.md#edge-status-thresholds"}, f, indent=2)

print("EDGE_VALIDATION_RESULTS.json written", flush=True)
for k, v in edge_results.items():
    print(k, v["status"], "p_holm=", round(v["holm_adjusted_p"], 4), "n=", v["observed_count"])

print("PART 2 STAGE 1 DONE")
