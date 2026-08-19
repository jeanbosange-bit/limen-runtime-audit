#!/usr/bin/env python3
"""Deterministic integrity checks for the completed cross-environment gate."""
import csv,hashlib,json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def load(n):return json.loads((HERE/n).read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
panel=load('CHALLENGE_PANEL.json')['challenges'];plan=load('EXECUTION_PLAN.json');manifest=load('MANIFEST.json');ident=load('IDENTIFICATION_RESULTS.json');prefix=load('PREFIX_ROBUSTNESS.json');tensor=load('TENSOR_ROBUSTNESS.json')
assert len(panel)==12==len({x['prompt_family'] for x in panel})
assert plan['planned_new_generations']==108<=216 and manifest['new_generations']==108
assert sha(ROOT/manifest['feature_registry'])==manifest['feature_registry_sha256']
rows=list(csv.DictReader((HERE/'RUN_INDEX.csv').open()));assert len(rows)==144
assert len([x for x in rows if x['source']=='NEW_BOUNDED_CAMPAIGN'])==108
assert len({(x['environment_id'],x['prompt_id'],x['seed']) for x in rows})==144
assert all(sha(ROOT/x['artifact_path'])==x['verified_sha256'] for x in rows)
assert ident['same_family_cross_environment']['status']=='NOT_ESTABLISHED'
assert ident['same_family_cross_environment']['candidate_count_per_family']==1
assert prefix['FIRST_2']['eligible_cross_environment_probes']==108
assert set(tensor)=={'H','Q','K','V','ATTENTION','ATTENTION_OUTPUT','LOGITS','ALL'}
assert all(v['status'] in {'FINGERPRINT_PRESERVED','FINGERPRINT_SHIFTED','FINGERPRINT_BROKEN'} for v in tensor.values())
assert load('VERIFICATION_PROXY_RESULTS.json')['authentication'] is False
security=(HERE/'SECURITY_BOUNDARY.md').read_text();report=(HERE/'REPORT.md').read_text()
assert 'not authentication' in security.lower() and 'PQC' in security
assert not re.search(r'\bAUTHENTICATED\b',security+report)
assert manifest['pqc']=='NOT_TESTED' and manifest['authentication']=='NOT_TESTED' and manifest['security_product_claim']=='NO'
assert manifest['source_artifacts_modified'] is False
print('CROSS_ENV_VALIDATION_TESTS 15/15 PASS')
