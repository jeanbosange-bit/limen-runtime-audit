"""
Part 2: extract local windows for all 6 groups by filtering atlas-coverage-analysis-v1's
already-computed per-tensor _LOCAL_PROPERTIES.csv files (read-only reuse, no npz reload).
For groups B/C/D, uses ALL landmark encounter events belonging to their member runs
(every node, every occurrence) since those groups are not defined by a single route
ordering the way ROUTE_FULL is.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
COVERAGE = HERE.parent / "atlas-coverage-analysis-v1"

with open(HERE / "GROUPS_CACHE.json") as f:
    GROUPS = json.load(f)
with open(HERE / "ROUTE_FIRST_STEPS_CACHE.json") as f:
    route_first_steps = json.load(f)

TENSOR_FILES = {
    "H": "H_LOCAL_PROPERTIES.csv", "Q": "Q_LOCAL_PROPERTIES.csv", "K": "K_LOCAL_PROPERTIES.csv",
    "V": "V_LOCAL_PROPERTIES.csv", "ATTENTION": "ATTENTION_LOCAL_PROPERTIES.csv",
    "ATTENTION_OUTPUT": "ATTENTION_OUTPUT_LOCAL_PROPERTIES.csv",
    "LOGIT_PROBABILITY": "LOGIT_PROBABILITY_LOCAL_PROPERTIES.csv",
}

# Which (run_id, layer, step) triples belong to each group, and which group each belongs to
route_full_keys = set()
for run_id, steps in route_first_steps.items():
    for node, layer in [("L0", 0), ("L5", 5), ("L20", 20)]:
        s = steps.get(node)
        if s is not None:
            route_full_keys.add((run_id, layer, s))

run_to_group = {}
for g in ("ROUTE_FULL", "L0_L5_ONLY", "L0_L20_ONLY", "OTHER_LANDMARK_ROUTE", "NO_LANDMARK", "SHORT_NON_EVALUABLE"):
    for run_id in GROUPS[g]:
        run_to_group[run_id] = g

group_rows_all_tensors = {g: [] for g in run_to_group.values()}
group_rows_all_tensors.setdefault("ROUTE_FULL", [])

route_local_windows = []
tensor_group_rows = {t: defaultdict(list) for t in TENSOR_FILES}

for tname, fname in TENSOR_FILES.items():
    with open(COVERAGE / fname) as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        run_id = row["run_id"]
        layer = int(row["layer"])
        step = int(row["step"])
        offset = int(row["offset"])
        group = run_to_group.get(run_id)
        if group is None:
            continue
        if group == "ROUTE_FULL":
            # only keep rows whose (run_id, layer, base step) matches the route's own first-occurrence triple
            base_step = step - offset
            if (run_id, layer, base_step) not in route_full_keys:
                continue
        tensor_group_rows[tname][group].append(row)
        if tname == "H":
            route_local_windows.append({**row, "group": group})

with open(HERE / "ROUTE_LOCAL_WINDOWS.csv", "w", newline="") as f:
    if route_local_windows:
        w = csv.DictWriter(f, fieldnames=list(route_local_windows[0].keys()))
        w.writeheader()
        for row in route_local_windows:
            w.writerow(row)

for tname, fname_out in [("H", "H_ROUTE_PROPERTIES.csv"), ("Q", "Q_ROUTE_PROPERTIES.csv"), ("K", "K_ROUTE_PROPERTIES.csv"),
                          ("V", "V_ROUTE_PROPERTIES.csv"), ("ATTENTION", "ATTENTION_ROUTE_PROPERTIES.csv"),
                          ("ATTENTION_OUTPUT", "ATTENTION_OUTPUT_ROUTE_PROPERTIES.csv"),
                          ("LOGIT_PROBABILITY", "LOGIT_PROBABILITY_ROUTE_PROPERTIES.csv")]:
    all_rows = []
    for group, rows in tensor_group_rows[tname].items():
        for row in rows:
            all_rows.append({**row, "group": group})
    with open(HERE / fname_out, "w", newline="") as f:
        if all_rows:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            for row in all_rows:
                w.writerow(row)
    print(fname_out, "written:", len(all_rows), "rows")

with open(HERE / "TENSOR_GROUP_ROWS_CACHE.json", "w") as f:
    json.dump({t: {g: len(rows) for g, rows in gd.items()} for t, gd in tensor_group_rows.items()}, f, indent=2)

print("PART 2 DONE")
