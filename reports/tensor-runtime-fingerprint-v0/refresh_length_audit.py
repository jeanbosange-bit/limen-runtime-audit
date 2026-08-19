#!/usr/bin/env python3
"""Refresh the three predeclared length correlations from emitted FULL fingerprints."""
import hashlib,json,math
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
HERE=Path(__file__).resolve().parent
df=pd.read_csv(HERE/"RUN_FINGERPRINTS.csv");meta=["run_id","prompt_id","prompt_family","seed","n_steps","seq_len","prompt_len","n_landmarks","source_sha256"]
X=df.drop(columns=meta).to_numpy(float);med=np.nanmedian(X,axis=0);q1=np.nanquantile(X,.25,axis=0);q3=np.nanquantile(X,.75,axis=0);keep=np.isfinite(med)&np.isfinite(q1)&np.isfinite(q3)&((q3-q1)>1e-9);X=X[:,keep];med=med[keep];scale=(q3-q1)[keep];X=np.where(np.isfinite(X),X,med);Z=(X-med)/scale
d=[];gen=[];prompt=[];total=[]
for i in range(len(df)):
 for j in range(i+1,len(df)):
  if df.prompt_id.iloc[i]==df.prompt_id.iloc[j]:continue
  d.append(np.linalg.norm(Z[i]-Z[j])/math.sqrt(Z.shape[1]));gen.append(abs(df.n_steps.iloc[i]-df.n_steps.iloc[j]));prompt.append(abs(df.prompt_len.iloc[i]-df.prompt_len.iloc[j]));total.append(abs(df.seq_len.iloc[i]-df.seq_len.iloc[j]))
result=json.loads((HERE/"INTRA_INTER_DISTANCE_RESULTS.json").read_text());primary=result["result"];primary["length_distance_spearman_rho"]=float(spearmanr(d,gen).statistic);primary["prompt_length_distance_spearman_rho"]=float(spearmanr(d,prompt).statistic);primary["total_sequence_length_distance_spearman_rho"]=float(spearmanr(d,total).statistic)
primary["ordering_overlap_rate"]=1-primary["common_language_separation"]
for name in ("PREFIX_FINGERPRINT_RESULTS.json","TENSOR_ABLATION_RESULTS.json"):
 data=json.loads((HERE/name).read_text())
 for value in data.values():value["distance"]["ordering_overlap_rate"]=1-value["distance"]["common_language_separation"]
 (HERE/name).write_text(json.dumps(data,indent=2)+"\n")
(HERE/"INTRA_INTER_DISTANCE_RESULTS.json").write_text(json.dumps(result,indent=2)+"\n")
audit=json.loads((HERE/"LENGTH_CONFOUND_AUDIT.json").read_text());audit["generated_length"]["spearman_rho_distance_vs_absolute_length_difference"]=primary["length_distance_spearman_rho"];audit["prompt_token_count"]={"spearman_rho_distance_vs_absolute_length_difference":primary["prompt_length_distance_spearman_rho"],"status":"DERIVED_CONFOUND_DIAGNOSTIC","limitation":"prompt_len is nearly prompt-identity-deterministic and cannot independently identify mechanism."};audit["total_sequence_length"]={"spearman_rho_distance_vs_absolute_length_difference":primary["total_sequence_length_distance_spearman_rho"],"status":"DERIVED_CONFOUND_DIAGNOSTIC","limitation":"seq_len combines prompt identity and generation length."};(HERE/"LENGTH_CONFOUND_AUDIT.json").write_text(json.dumps(audit,indent=2)+"\n")
required=["FROZEN_PROTOCOL.md","SOURCE_PROVENANCE.json","FEATURE_REGISTRY.json","RUN_FINGERPRINTS.csv","INTRA_INTER_DISTANCE_RESULTS.json","SAME_FAMILY_CONTROL.json","LENGTH_CONFOUND_AUDIT.json","SEED_STABILITY.csv","PREFIX_FINGERPRINT_RESULTS.json","TENSOR_ABLATION_RESULTS.json","LAYER_LOCALIZATION.csv","CHALLENGE_IDENTIFICATION_RESULTS.json","LANDMARK_RELATION_AUDIT.json","SECURITY_INTERPRETATION_BOUNDARY.md","FUTURE_ATTESTATION_ARCHITECTURE.md","EPISTEMIC_STATUS.md","REPORT.md","MANIFEST.json","STAR_COMMAND_SECURITY_FEATURE_CONTRACT.json","build_fingerprint.py","refresh_length_audit.py","test_fingerprint.py"]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
(HERE/"SHA256SUMS.txt").write_text("".join(f"{digest(HERE/name)}  {name}\n" for name in required))
print(json.dumps({"generated":primary["length_distance_spearman_rho"],"prompt":primary["prompt_length_distance_spearman_rho"],"total":primary["total_sequence_length_distance_spearman_rho"]},indent=2))
