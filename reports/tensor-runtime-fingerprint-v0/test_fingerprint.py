#!/usr/bin/env python3
import csv,hashlib,json,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def j(name):return json.loads((HERE/name).read_text())
class FingerprintChecks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.m=j("MANIFEST.json");cls.p=j("SOURCE_PROVENANCE.json");cls.f=j("FEATURE_REGISTRY.json");cls.ident=j("CHALLENGE_IDENTIFICATION_RESULTS.json");cls.prefix=j("PREFIX_FINGERPRINT_RESULTS.json");cls.ablation=j("TENSOR_ABLATION_RESULTS.json")
 def test_source_shape_and_zero_generation(self):self.assertEqual((self.m["source_runs"],self.m["prompt_challenges"],len(self.m["seeds"]),self.m["real_new_generations"]),(144,48,3,0))
 def test_canonical_source_hashes_verify(self):
  sums={line.split()[1]:line.split()[0] for line in (ROOT/"reports/bcc-atlas-expansion-v1/ATLAS_DEEP/SHA256SUMS.txt").read_text().splitlines() if line.strip()}
  for run_id,digest in self.p["artifact_hashes"].items():self.assertEqual(digest,sums[f"raw_captures/{run_id}.npz"])
 def test_features_are_bounded_summaries(self):self.assertFalse(self.f["raw_flattened_tensors_used"]);self.assertEqual(set(self.f["families"]),{"H","Q","K","V","ATTENTION","ATTENTION_OUTPUT","LOGITS"})
 def test_strict_seed_holdout_and_baselines(self):self.assertIn("strict rotating seed holdout",self.ident["validation"]);self.assertAlmostEqual(self.ident["random_baselines"]["global_top1"],1/48);self.assertEqual(self.ident["random_baselines"]["same_family_top1"],.25)
 def test_prefix_eligibility_is_explicit(self):self.assertEqual(self.prefix["FULL"]["distance"]["eligible_runs"],144);self.assertLess(self.prefix["FIRST_8"]["distance"]["eligible_runs"],144);self.assertLess(self.prefix["FIRST_8"]["identification"]["eligible_prompt_challenges"],48)
 def test_all_frozen_ablations_exist(self):self.assertEqual(set(self.ablation),{"H","Q","K","V","ATTENTION","ATTENTION_OUTPUT","LOGITS","H+Q+K+V","ATTENTION+ATTENTION_OUTPUT","ALL"})
 def test_security_boundary(self):self.assertEqual(self.m["cryptographic_authentication"],"NOT_TESTED");self.assertEqual(self.m["pqc"],"NOT_TESTED");self.assertEqual(self.m["entropy_source"],"NOT_TESTED");self.assertEqual(self.m["security_product_claim"],"NO");self.assertNotIn("AUTHENTICATED",j("STAR_COMMAND_SECURITY_FEATURE_CONTRACT.json")["human_status_labels"])
 def test_output_hashes_verify(self):
  for line in (HERE/"SHA256SUMS.txt").read_text().splitlines():
   digest,name=line.split();self.assertEqual(hashlib.sha256((HERE/name).read_bytes()).hexdigest(),digest)
if __name__=="__main__":unittest.main()
