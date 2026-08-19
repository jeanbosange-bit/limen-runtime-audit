#!/usr/bin/env python3
import csv, hashlib, json, unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; LM=ROOT/"reports/multi-tensor-landmarks-v0"
def read_json(name): return json.loads((HERE/name).read_text())

class ReferenceLandmarkNetworkChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nodes=read_json("FROZEN_NODE_REGISTRY.json");cls.edges=read_json("EDGE_VALIDATION_RESULTS.json");cls.manifest=read_json("MANIFEST.json");cls.unknown=read_json("UNKNOWN_TERRITORY_AUDIT.json")
        with (HERE/"ROUTE_INSTANCES.csv").open() as f:cls.routes=list(csv.DictReader(f))
        cls.landmarks=json.loads((LM/"LANDMARK_RESULTS.json").read_text())

    def test_only_four_confirmed_primary_nodes(self):
        self.assertEqual([n["landmark_id"] for n in self.nodes["primary_nodes"]],["L0","L1","L5","L20"])
        self.assertTrue(all(n["validation_status"]=="REFERENCE_LANDMARK_SUPPORTED" for n in self.nodes["primary_nodes"]))

    def test_low_support_and_non_replicated_are_not_promoted(self):
        self.assertEqual(self.nodes["secondary_low_support_count"],11);self.assertEqual(self.nodes["not_replicated_count"],9)
        self.assertFalse(any(n["validation_status"] in {"RECURRING_BUT_LOW_SUPPORT","NOT_REPLICATED"} for n in self.nodes["primary_nodes"]))

    def test_routes_use_first_actual_step_not_row_position(self):
        candidates={f'L{x["layer"]}':x for x in self.landmarks["candidates"] if x["promotion_status"]=="REFERENCE_LANDMARK_SUPPORTED"}
        for route in self.routes:
            first={}
            for lid,candidate in candidates.items():
                steps=[int(e["step"]) for split in ("split_A","split_B") for e in candidate[split]["events"] if e["run_id"]==route["run_id"]]
                if steps:first[lid]=min(steps)
            expected=sorted(first,key=lambda x:(first[x],int(x[1:])))
            self.assertEqual(route["landmark_sequence"],"-".join(expected) or "NONE")

    def test_run_and_prompt_dependence_are_preserved(self):
        source={(r["run_id"],r["prompt_id"],r["prompt_family"],r["seed"]) for r in self.routes}
        self.assertEqual(len(source),30)
        for edge in self.edges["edges"]:
            self.assertEqual(edge["support_prompt_groups"],len(edge["supporting_prompt_ids"]))

    def test_only_final_corrected_transition_inputs_are_used(self):
        paths=[x["path"] for x in self.manifest["source_inputs"]]
        self.assertIn("reports/multi-tensor-landmarks-v0/TRANSITION_RESULTS.json",paths)
        self.assertFalse(any("BUGFIX_DISCLOSURE" in x for x in paths));self.assertIn("BUGFIX_DISCLOSURE_buggy_transition_results.json",self.manifest["obsolete_inputs_ignored"])
        self.assertEqual(self.manifest["alignment"],"ACTUAL_STEP_KEYED_WITH_TIES_PRESERVED")

    def test_same_step_encounters_are_not_forced_into_succession(self):
        route=next(r for r in self.routes if r["run_id"]=="KVC_GI02_1102")
        self.assertEqual(route["route_signature"],"L0+L1")
        edge=next(e for e in self.edges["edges"] if e["source"]=="L0" and e["target"]=="L1")
        self.assertNotIn("KVC_GI02_1102",edge["supporting_runs"])

    def test_evidence_hashes_match(self):
        for item in self.manifest["source_inputs"]:
            actual=hashlib.sha256((ROOT/item["path"]).read_bytes()).hexdigest();self.assertEqual(actual,item["sha256"])

    def test_unknown_territory_is_preserved(self):
        self.assertEqual(self.unknown["status"],"UNKNOWN_TERRITORY_PRESERVED");self.assertEqual(len(self.unknown["runs_with_no_confirmed_landmark"]),12)
        self.assertNotIn("unsafe",self.unknown["meaning"].lower())

    def test_no_causal_or_cognitive_edge_vocabulary(self):
        forbidden={"CAUSES","LEADS_TO_REASONING","DECISION_EDGE","REASONING_EDGE","COGNITIVE_STAGE"}
        self.assertFalse(any(edge["edge_type"] in forbidden for edge in self.edges["edges"]))
        self.assertEqual(self.manifest["causal_claim"],"NO");self.assertEqual(self.manifest["control"],"NO")

if __name__=="__main__":unittest.main()
