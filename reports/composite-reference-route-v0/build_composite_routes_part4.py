"""Composite Reference Route V0 -- part 4: NETWORK_EXTENSION_GRAPH.json + STAR_COMMAND_ROUTE_CONTRACT.json."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NET_V1 = HERE.parent / "reference-landmark-network-v1"

with open(HERE / "ROUTE_VALIDATION_RESULTS.json") as f:
    route_val = json.load(f)
with open(HERE / "BRANCHING_VS_COMPOSITION_ANALYSIS.json") as f:
    branching = json.load(f)
with open(NET_V1 / "FROZEN_NODE_REGISTRY.json") as f:
    nodes_reg = json.load(f)

NODE_LAYER = {"L0": 0, "L1": 1, "L5": 5, "L20": 20}
supported_routes = {k: v for k, v in route_val.items() if v["status"] == "REFERENCE_ROUTE_SUPPORTED"}
low_support_routes = {k: v for k, v in route_val.items() if v["status"] == "REFERENCE_ROUTE_LOW_SUPPORT"}

extension_graph = {
    "schema": "NetworkExtensionGraph -- 3-node composite routes over the frozen landmark network, extends reference-landmark-network-v1/NETWORK_GRAPH.json",
    "base_network_supported_edges": ["L0->L5", "L0->L20"],
    "composite_routes": [
        {"route": v["route"], "signature": k, "status": v["status"], "observed_count": v["observed_count"],
         "n_families": v["n_families"], "n_seeds": v["n_seeds"], "holm_adjusted_p": v["holm_adjusted_p"]}
        for k, v in route_val.items() if v["status"] in ("REFERENCE_ROUTE_SUPPORTED", "REFERENCE_ROUTE_LOW_SUPPORT")
    ],
    "branching_vs_composition": branching,
    "explicit_non_claims": ["No route is causal.", "No node initiates, mediates, or terminates anything.",
                              "This is not a reasoning path or decision path."],
}
with open(HERE / "NETWORK_EXTENSION_GRAPH.json", "w") as f:
    json.dump(extension_graph, f, indent=2)

# ============================================================
# STAR_COMMAND_ROUTE_CONTRACT.json
# ============================================================
def human_route(k, v):
    if v["status"] == "REFERENCE_ROUTE_SUPPORTED":
        label = "SUPPORTED REFERENCE ROUTE"
        text = f"This ordered route has been repeatedly observed across {v['observed_count']} runs, {v['n_families']} prompt families, and {v['n_seeds']} seeds."
    elif v["status"] == "REFERENCE_ROUTE_LOW_SUPPORT":
        label = "LOW-SUPPORT ROUTE"
        text = f"This ordered route has been observed {v['observed_count']} time(s) across {v['n_families']} prompt families and {v['n_seeds']} seed(s) -- not yet enough to call it a stable reference route."
    else:
        label = "UNKNOWN CONNECTION"
        text = "This ordered route has not been observed."
    return {"route": v["route"], "signature": k, "ui_label": label, "ui_text": text,
            "technical_detail_expandable": {"observed_count": v["observed_count"], "holm_adjusted_p": v["holm_adjusted_p"],
                                              "n_families": v["n_families"], "n_seeds": v["n_seeds"], "n_prompt_ids": v["n_prompt_ids"]}}


n_supported = len(supported_routes)
contract = {
    "contract": "STAR_COMMAND_COMPOSITE_ROUTE_V0",
    "mode": "READ_ONLY",
    "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
    "base_confirmed_landmarks": ["L0", "L1", "L5", "L20"],
    "base_supported_connections_ui_label": "SUPPORTED REFERENCE CONNECTION",
    "base_supported_connections": ["L0->L5", "L0->L20"],
    "composite_routes": [human_route(k, v) for k, v in route_val.items() if v["status"] in ("REFERENCE_ROUTE_SUPPORTED", "REFERENCE_ROUTE_LOW_SUPPORT")],
    "branching_pattern_ui_label": "BRANCHING PATTERN",
    "branching_pattern_note": branching["verdict"],
    "fallback_message_if_no_composite_route_supported": "Two validated transitions originate from L0, but a stable three-landmark route has not yet been established.",
    "supported_message_if_composite_route_supported": "This ordered route has been repeatedly observed across multiple prompts, families and seeds.",
    "fallback_applies_this_gate": n_supported == 0,
    "ui_rule": "Route p-value is NEVER the primary UI element -- it lives only in technical_detail_expandable. The primary UI text is always the plain-language run/family/seed count sentence. No mechanistic language (initiates/mediates/terminates/decision path/reasoning path/causal chain/control corridor/safe route) anywhere.",
    "control": "NO", "sra_x": "NOT_TESTED", "causal_claim": "NO",
}
with open(HERE / "STAR_COMMAND_ROUTE_CONTRACT.json", "w") as f:
    json.dump(contract, f, indent=2)

print(f"NETWORK_EXTENSION_GRAPH.json and STAR_COMMAND_ROUTE_CONTRACT.json written. n_supported={n_supported}")
