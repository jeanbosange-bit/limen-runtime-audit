"""Reference Landmark Network V1 -- part 4: NETWORK_GRAPH.json + STAR_COMMAND_NETWORK_CONTRACT.json."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
with open(HERE / "EDGE_VALIDATION_RESULTS.json") as f:
    edge_val = json.load(f)
with open(HERE / "ROUTE_VALIDATION_RESULTS.json") as f:
    route_val = json.load(f)
with open(HERE / "FROZEN_NODE_REGISTRY.json") as f:
    nodes_reg = json.load(f)

NODES = ["L0", "L1", "L5", "L20"]
NODE_LAYER = {"L0": 0, "L1": 1, "L5": 5, "L20": 20}

network_graph = {
    "schema": "ReferenceLandmarkNetworkV1 -- read-only descriptive graph, NOT a general reasoning graph",
    "nodes": [{"id": n, "layer": NODE_LAYER[n], "status": "REFERENCE_LANDMARK_SUPPORTED",
               "lane_b_replication": "REPLICATED_ON_LANE_B"} for n in NODES],
    "edges": [
        {"source": e["source"], "target": e["target"], "edge_type": "SUPPORTED_DESCRIPTIVE_TRANSITION" if e["status"] == "REFERENCE_EDGE_SUPPORTED" else ("LOW_SUPPORT_DESCRIPTIVE_TRANSITION" if e["status"] == "REFERENCE_EDGE_LOW_SUPPORT" else "NO_OBSERVED_LINK"),
         "observed_count": e["observed_count"], "n_families": e["n_families"], "n_seeds": e["n_seeds"], "n_prompt_ids": e["n_prompt_ids"],
         "holm_adjusted_p": e["holm_adjusted_p"], "status": e["status"]}
        for e in edge_val["directed_edges"].values()
    ],
    "cooccurrence_links": [
        {"pair": v["pair"], "observed_count": v["observed_count"], "status": v["status"]}
        for v in edge_val["cooccurrence_pairs"].values()
    ],
    "reference_routes": [{"signature": k, **v} for k, v in route_val.items() if v["status"] == "REFERENCE_ROUTE_SUPPORTED"],
    "explicit_non_claims": ["No edge is causal.", "No node is a reasoning step, decision, or cognitive stage.",
                              "This is not a general-purpose reasoning graph for the model."],
}
with open(HERE / "NETWORK_GRAPH.json", "w") as f:
    json.dump(network_graph, f, indent=2)

n_supported = sum(1 for e in edge_val["directed_edges"].values() if e["status"] == "REFERENCE_EDGE_SUPPORTED")

# ============================================================
# STAR_COMMAND_NETWORK_CONTRACT.json
# ============================================================
def human_edge(e):
    if e["status"] == "REFERENCE_EDGE_SUPPORTED":
        label = "SUPPORTED REFERENCE CONNECTION"
        text = f"We have repeatedly observed this transition across {e['observed_count']} runs, {e['n_families']} prompt families, and {e['n_seeds']} seeds."
    elif e["status"] == "REFERENCE_EDGE_LOW_SUPPORT":
        label = "LOW-SUPPORT CONNECTION"
        text = f"This transition has been observed {e['observed_count']} time(s) across {e['n_families']} prompt families and {e['n_seeds']} seed(s) -- not yet enough to call it a stable reference connection."
    else:
        label = "UNKNOWN CONNECTION"
        text = "No transition of this kind has been observed yet."
    return {"source": e["source"], "target": e["target"], "ui_label": label, "ui_text": text,
            "technical_detail_expandable": {"observed_count": e["observed_count"], "holm_adjusted_p": e["holm_adjusted_p"],
                                              "n_families": e["n_families"], "n_seeds": e["n_seeds"], "n_prompt_ids": e["n_prompt_ids"]}}


contract = {
    "contract": "STAR_COMMAND_REFERENCE_LANDMARK_NETWORK_V1",
    "mode": "READ_ONLY",
    "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
    "confirmed_landmarks": [{"id": n, "ui_label": "CONFIRMED LANDMARK", "layer": NODE_LAYER[n]} for n in NODES],
    "connections": [human_edge(e) for e in edge_val["directed_edges"].values()],
    "observed_routes_ui_label": "OBSERVED ROUTE",
    "reference_routes_ui_label": "REFERENCE ROUTE",
    "reference_routes": [{"signature": k, "ui_text": f"Observed {v['n_runs']} times across {v['n_families']} prompt families and {v['n_seeds']} seeds."} for k, v in route_val.items() if v["status"] == "REFERENCE_ROUTE_SUPPORTED"],
    "fallback_message_if_no_supported_edge": "Confirmed points of reference exist, but stable connections between them are not yet established.",
    "fallback_message_applies_this_gate": n_supported == 0,
    "ui_rule": "Edge p-value is NEVER the primary UI element -- it lives only in technical_detail_expandable. The primary UI text is always the plain-language run/family/seed count sentence.",
    "control": "NO", "sra_x": "NOT_TESTED", "causal_claim": "NO",
}
with open(HERE / "STAR_COMMAND_NETWORK_CONTRACT.json", "w") as f:
    json.dump(contract, f, indent=2)

print("NETWORK_GRAPH.json and STAR_COMMAND_NETWORK_CONTRACT.json written")
print(f"n_supported edges: {n_supported}, fallback_applies: {n_supported == 0}")
