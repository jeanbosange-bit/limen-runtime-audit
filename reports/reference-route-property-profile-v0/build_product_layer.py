"""Part 5: ROUTE_PASSPORT_L0_L5_L20.json, STAR_COMMAND_ROUTE_CARD_CONTRACT.json, ROUTE_MAP_LINK.json."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUTE_V0 = HERE.parent / "composite-reference-route-v0"
NET_V1 = HERE.parent / "reference-landmark-network-v1"

with open(HERE / "ROUTE_OCCURRENCE_PROFILE.json") as f:
    occ = json.load(f)
with open(HERE / "ROUTE_PROPERTY_REGISTRY.json") as f:
    prop_reg = json.load(f)
with open(ROUTE_V0 / "ROUTE_VALIDATION_RESULTS.json") as f:
    route_val = json.load(f)["L0-L5-L20"]

n_supported = sum(1 for v in prop_reg.values() if v["status"] == "ROUTE_PROPERTY_SUPPORTED")
n_low = sum(1 for v in prop_reg.values() if v["status"] == "ROUTE_PROPERTY_LOW_SUPPORT")
n_none = sum(1 for v in prop_reg.values() if v["status"] == "ROUTE_PROPERTY_NOT_ESTABLISHED")

# ============================================================
# ROUTE_PASSPORT_L0_L5_L20.json -- 3 layers
# ============================================================
passport = {
    "route_id": "reference_route_a_l0_l5_l20",
    "level_1_human_meaning": {
        "name": "Reference Route A",
        "technical_notation": "L0 -> L5 -> L20",
        "human_description": "A recurring internal path observed during multiple model executions.",
        "utility": "This route helps compare a new execution with internal paths already observed in the model.",
        "landmark_explanation": "Points of reference are recurring measurable internal patterns. When several appear in a consistent order, Star Command calls that order a route.",
        "critical_statement": "A route does not show what the model is thinking. It is a reference pattern for comparing executions.",
    },
    "level_2_explanation": {
        "how_often_observed": f"{occ['run_count']} runs",
        "prompt_families_count": occ["family_breadth"],
        "seeds_count": occ["seed_breadth"],
        "general_property_profile": f"{n_supported} strongly supported measurable properties, {n_low} properties with limited/mixed evidence, {n_none} properties not yet established -- the route's internal tensor profile is largely similar to other landmark-bearing executions, not yet shown to be distinctive to this specific ordering.",
        "current_evidence_level": "SUPPORTED" if route_val["status"] == "REFERENCE_ROUTE_SUPPORTED" else ("LIMITED SUPPORT" if route_val["status"] == "REFERENCE_ROUTE_LOW_SUPPORT" else "NOT ESTABLISHED"),
        "known_limitations": [
            "Not observed in every task family (6 of 12) or every seed (2 of 3 -- absent in seed 2101).",
            "Its own generation lengths are longer than typical (median 130 decode steps) -- whether this route requires long executions to appear, or is simply easier to detect in them, is not established.",
            "No measurable internal property has yet been shown to be uniquely distinctive to this route versus other recurring internal patterns.",
        ],
    },
    "level_3_technical_detail": {
        "route": "L0 -> L5 -> L20",
        "run_count": occ["run_count"], "prompt_count": occ["prompt_count"], "prompt_ids": occ["prompt_ids"],
        "family_coverage": occ["families"], "seed_coverage": occ["seeds"],
        "median_step_distances": {k: v["median"] for k, v in occ["step_distance_distributions"].items()},
        "step_distance_distributions": occ["step_distance_distributions"],
        "effect_sizes_and_adjusted_p_values": "see ROUTE_GROUP_COMPARISONS.json and ROUTE_PROPERTY_REGISTRY.json",
        "route_validation_reference": {"holm_adjusted_p": route_val["holm_adjusted_p"], "status": route_val["status"],
                                          "constituent_edge_statuses": route_val["constituent_edge_statuses"]},
        "tensor_property_registry_summary": {k: v["status"] for k, v in prop_reg.items()},
        "source_files": ["reports/composite-reference-route-v0/ROUTE_VALIDATION_RESULTS.json",
                          "reports/reference-route-property-profile-v0/ROUTE_PROPERTY_REGISTRY.json",
                          "reports/reference-route-property-profile-v0/ROUTE_GROUP_COMPARISONS.json"],
        "provenance": "reports/reference-route-property-profile-v0/SOURCE_PROVENANCE.json",
        "hashes_file": "reports/reference-route-property-profile-v0/SHA256SUMS.txt",
    },
}
with open(HERE / "ROUTE_PASSPORT_L0_L5_L20.json", "w") as f:
    json.dump(passport, f, indent=2, default=str)

print("ROUTE_PASSPORT_L0_L5_L20.json written")

# ============================================================
# STAR_COMMAND_ROUTE_CARD_CONTRACT.json
# ============================================================
card_contract = {
    "contract": "STAR_COMMAND_ROUTE_CARD_V0",
    "mode": "READ_ONLY",
    "card": {
        "title": "Reference Route A",
        "notation": "L0 -> L5 -> L20",
        "status_label": "Supported reference route",
        "what_this_means": "A recurring internal path observed in multiple tasks.",
        "observed_across": {"task_families": occ["family_breadth"], "independent_seeds": occ["seed_breadth"]},
        "why_it_matters": "Helps compare new executions with previously observed routes.",
        "known_limitation": "Not observed in every task or every seed.",
        "actions": ["Why this matters", "View evidence", "Technical details"],
    },
    "forbidden_labels": ["SAFE ROUTE", "NORMAL ROUTE", "GOOD ROUTE", "ERROR ROUTE", "REASONING ROUTE"],
    "forbidden_labels_used_this_gate": False,
    "mandatory_ux_phrases_fr": {
        "landmark_route_explanation": "Les points de repère indiquent où Star Command reconnaît un motif interne déjà observé. Les routes montrent comment ces points se succèdent pendant une exécution.",
        "utility": "Les routes servent à comparer une exécution actuelle aux parcours déjà observés.",
        "limitation": "Elles ne révèlent pas ce que le modèle pense et ne constituent pas une explication causale.",
    },
    "avoided_first_line_terms": ["tensor", "manifold", "latent geometry", "Holm correction", "edge", "null model"],
    "control": "NO", "sra_x": "NOT_TESTED", "causal_claim": "NO",
}
with open(HERE / "STAR_COMMAND_ROUTE_CARD_CONTRACT.json", "w") as f:
    json.dump(card_contract, f, indent=2, ensure_ascii=False)

print("STAR_COMMAND_ROUTE_CARD_CONTRACT.json written")

# ============================================================
# ROUTE_MAP_LINK.json
# ============================================================
map_link = {
    "route_id": "reference_route_a_l0_l5_l20",
    "landmarks": [{"id": "L0", "layer": 0, "position_in_route": "occurs first"},
                  {"id": "L5", "layer": 5, "position_in_route": "intermediate confirmed landmark in this route"},
                  {"id": "L20", "layer": 20, "position_in_route": "occurs later in this route"}],
    "support": {"run_count": occ["run_count"], "prompt_count": occ["prompt_count"]},
    "families": occ["families"], "seeds": occ["seeds"],
    "properties": {k: v["status"] for k, v in prop_reg.items()},
    "evidence_refs": [
        "reports/composite-reference-route-v0/ROUTE_VALIDATION_RESULTS.json",
        "reports/reference-landmark-network-v1/EDGE_VALIDATION_RESULTS.json",
        "reports/atlas-coverage-analysis-v1/LANDMARK_REPLICATION_RESULTS.json",
        "reports/multi-tensor-landmarks-v0/FROZEN_LANDMARK_REGISTRY.json",
    ],
    "unknowns": ["6 of 12 prompt families never observed with this route", "seed 2101 never observed with this route",
                  "no tensor property yet shown uniquely distinctive to this route vs. other landmark-bearing executions"],
    "crossaxis_coordinate_system": "(run_id, step, layer) -- canonical linking coordinate, per reports/multi-tensor-atlas-v0",
    "drill_down_path": ["Map", "Route", "Landmark", "Step", "Layer", "Tensor", "Evidence"],
    "drill_up_path": ["Tensor observation", "Landmark", "Route", "Model execution", "Mission / Agent (when available)"],
    "control": "NO", "sra_x": "NOT_TESTED",
}
with open(HERE / "ROUTE_MAP_LINK.json", "w") as f:
    json.dump(map_link, f, indent=2, default=str)

print("ROUTE_MAP_LINK.json written")
