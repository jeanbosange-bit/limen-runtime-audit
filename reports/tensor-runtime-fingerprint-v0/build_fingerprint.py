#!/usr/bin/env python3
"""Read-only deterministic Tensor Runtime Fingerprint Feasibility V0 analysis."""
from __future__ import annotations
import csv, hashlib, json, math, statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]; SRC=ROOT/"reports/bcc-atlas-expansion-v1/ATLAS_DEEP"; OUT=Path(__file__).resolve().parent
RNG_SEED=20260818; PREFIXES={"FULL":None,"FIRST_2":2,"FIRST_4":4,"FIRST_8":8}; FAMILIES=["H","Q","K","V","ATTENTION","ATTENTION_OUTPUT","LOGITS"]
COMBOS={f:[f] for f in FAMILIES}|{"H+Q+K+V":["H","Q","K","V"],"ATTENTION+ATTENTION_OUTPUT":["ATTENTION","ATTENTION_OUTPUT"],"ALL":FAMILIES}
STATS=("mean","std","median","p10","p90")

def load_json(p):return json.loads(p.read_text())
def dump(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for block in iter(lambda:f.read(8<<20),b""):h.update(block)
 return h.hexdigest()
def stat_values(v):
 v=np.asarray(v,dtype=np.float64);v=v[np.isfinite(v)]
 if not len(v):return {s:None for s in STATS}
 return {"mean":float(v.mean()),"std":float(v.std()),"median":float(np.median(v)),"p10":float(np.quantile(v,.1)),"p90":float(np.quantile(v,.9))}
def add_stats(out,prefix,layer,metric,values):
 for s,v in stat_values(values).items():out[f"{prefix}.L{layer:02d}.{metric}.{s}"]=v
def safe_cos_change(a,b):
 num=np.sum(a*b,axis=-1);den=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1);return 1-num/np.maximum(den,1e-12)

def vector_features(arr,family,positions,heads=False,curvature=False):
 out={};a=np.asarray(arr,dtype=np.float32)
 for layer in range(a.shape[0]):
  x=a[layer,:,positions,:] if heads else a[layer,positions,:]
  norm=np.linalg.norm(x,axis=-1)/math.sqrt(x.shape[-1]);add_stats(out,family,layer,"norm",norm.ravel())
  if heads:add_stats(out,family,layer,"head_dispersion",np.std(norm,axis=1))
  if len(positions)>=2:
   d=np.diff(x,axis=0);add_stats(out,family,layer,"delta_norm",(np.linalg.norm(d,axis=-1)/math.sqrt(x.shape[-1])).ravel());add_stats(out,family,layer,"cosine_change",safe_cos_change(x[1:],x[:-1]).ravel())
   if curvature and len(positions)>=3:add_stats(out,family,layer,"curvature",safe_cos_change(d[1:],d[:-1]).ravel())
  else:
   for metric in ("delta_norm","cosine_change")+(('curvature',) if curvature else ()):add_stats(out,family,layer,metric,[])
 return out

def attention_features(arr,positions):
 out={};a=np.asarray(arr,dtype=np.float32)
 for layer in range(a.shape[0]):
  entropy=[];concentration=[];maximum=[];dispersion=[]
  for pos in positions:
   weights=np.maximum(a[layer,:,pos,:pos+1],0);weights=weights/np.maximum(weights.sum(axis=-1,keepdims=True),1e-12)
   entropy.extend((-np.sum(weights*np.log(np.maximum(weights,1e-12)),axis=-1)/max(math.log(pos+1),1)).tolist());concentration.extend(np.sum(weights*weights,axis=-1).tolist());maximum.extend(np.max(weights,axis=-1).tolist())
   idx=np.arange(pos+1,dtype=np.float32)/max(pos,1);mu=np.sum(weights*idx,axis=-1);dispersion.extend(np.sum(weights*(idx-mu[:,None])**2,axis=-1).tolist())
  for metric,values in (("normalized_entropy",entropy),("concentration",concentration),("maximum_mass",maximum),("normalized_position_dispersion",dispersion)):add_stats(out,"ATTENTION",layer,metric,values)
 return out

def logits_features(arr,count):
 out={};x=np.asarray(arr[:count],dtype=np.float32);layer=0
 if not len(x):return out
 shifted=x-x.max(axis=1,keepdims=True);ex=np.exp(shifted);prob=ex/ex.sum(axis=1,keepdims=True);entropy=-np.sum(prob*np.log(np.maximum(prob,1e-30)),axis=1);top=np.partition(prob,-2,axis=1)[:,-2:];top.sort(axis=1)
 add_stats(out,"LOGITS",layer,"entropy",entropy);add_stats(out,"LOGITS",layer,"top1_probability",top[:,1]);add_stats(out,"LOGITS",layer,"top2_gap",top[:,1]-top[:,0]);add_stats(out,"LOGITS",layer,"norm",np.linalg.norm(x,axis=1)/math.sqrt(x.shape[1]))
 if len(x)>=2:add_stats(out,"LOGITS",layer,"delta_norm",np.linalg.norm(np.diff(x,axis=0),axis=1)/math.sqrt(x.shape[1]));add_stats(out,"LOGITS",layer,"cosine_change",safe_cos_change(x[1:],x[:-1]))
 else:add_stats(out,"LOGITS",layer,"delta_norm",[]);add_stats(out,"LOGITS",layer,"cosine_change",[])
 return out

def prefix_positions(meta,k):
 n=meta["n_steps"] if k is None else k
 return np.arange(meta["prompt_len"],meta["prompt_len"]+n,dtype=int)

# Canonical source registry and actual artifact verification.
rows=[]
with (SRC/"RUN_INDEX.csv").open() as f:
 for r in csv.DictReader(f):
  rows.append({"run_id":r["run_id"],"prompt_id":r["prompt_id"],"prompt_family":r["prompt_family"],"seed":int(r["seed"]),"n_steps":int(r["n_steps"]),"seq_len":int(r["seq_len"]),"prompt_len":int(r["prompt_len"]),"n_landmarks":int(r["n_landmarks"] or 0),"artifact_hash":r["artifact_hash"]})
assert len(rows)==144 and len({r["prompt_id"] for r in rows})==48 and len({r["seed"] for r in rows})==3
canonical_sums={line.split()[1]:line.split()[0] for line in (SRC/"SHA256SUMS.txt").read_text().splitlines() if line.strip()};source_hashes={};run_index_hash_matches=0;features={p:{} for p in PREFIXES}
for index,meta in enumerate(rows,1):
 p=SRC/"raw_captures"/f'{meta["run_id"]}.npz';actual=sha(p)
 if actual!=canonical_sums.get(f'raw_captures/{meta["run_id"]}.npz'):raise RuntimeError(f"canonical SHA256SUMS mismatch: {p}")
 run_index_hash_matches+=actual==meta["artifact_hash"]
 source_hashes[meta["run_id"]]=actual
 with np.load(p,allow_pickle=False) as z:
  arrays={"H":z["hidden_states_all_layers_fp16"],"Q":z["q_all_layers_fp16"],"K":z["k_all_layers_fp32"],"V":z["v_all_layers_fp32"],"ATTENTION":z["attentions_all_layers_fp16"],"ATTENTION_OUTPUT":z["attn_output_all_layers_fp16"],"LOGITS":z["logits_region_fp16"]}
  for prefix,k in PREFIXES.items():
   if k is not None and meta["n_steps"]<k:continue
   pos=prefix_positions(meta,k);run={}
   run["H"]=vector_features(arrays["H"],"H",pos,curvature=True);run["Q"]=vector_features(arrays["Q"],"Q",pos,heads=True);run["K"]=vector_features(arrays["K"],"K",pos,heads=True);run["V"]=vector_features(arrays["V"],"V",pos,heads=True);run["ATTENTION"]=attention_features(arrays["ATTENTION"],pos);run["ATTENTION_OUTPUT"]=vector_features(arrays["ATTENTION_OUTPUT"],"ATTENTION_OUTPUT",pos,curvature=True);run["LOGITS"]=logits_features(arrays["LOGITS"],len(pos));features[prefix][meta["run_id"]]=run
 print(f"EXTRACT {index}/144 {meta['run_id']}",flush=True)

# Full bounded summaries, not raw tensors.
full_records=[]
for meta in rows:
 record={k:meta[k] for k in ("run_id","prompt_id","prompt_family","seed","n_steps","seq_len","prompt_len","n_landmarks")};record["source_sha256"]=source_hashes[meta["run_id"]]
 for family in FAMILIES:record.update(features["FULL"][meta["run_id"]][family])
 full_records.append(record)
all_cols=list(full_records[0]);feature_cols=sorted(set().union(*(r.keys() for r in full_records))-set(all_cols[:9]));all_cols=all_cols[:9]+feature_cols
with (OUT/"RUN_FINGERPRINTS.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=all_cols);w.writeheader();w.writerows(full_records)

meta_by_id={r["run_id"]:r for r in rows}
def matrix(prefix,fams,layer=None):
 eligible=[r for r in rows if r["run_id"] in features[prefix]];keys=[]
 for r in eligible:
  for fam in fams:
   keys.extend(k for k in features[prefix][r["run_id"]][fam] if layer is None or f".L{layer:02d}." in k)
 keys=sorted(set(keys));X=np.array([[features[prefix][r["run_id"]][fam].get(key,np.nan) for fam in fams for key in []] for r in eligible]) if False else np.array([[next((features[prefix][r["run_id"]][fam][key] for fam in fams if key in features[prefix][r["run_id"]][fam]),np.nan) for key in keys] for r in eligible],dtype=float)
 return eligible,keys,X
def fit_transform(X,ref=None):
 base=X if ref is None else ref;med=np.nanmedian(base,axis=0);q1=np.nanquantile(base,.25,axis=0);q3=np.nanquantile(base,.75,axis=0);scale=q3-q1;keep=np.isfinite(med)&np.isfinite(scale)&(scale>1e-9)
 med=med[keep];scale=scale[keep];Y=X[:,keep];Y=np.where(np.isfinite(Y),Y,med);return (Y-med)/scale,keep
def dist(a,b):return float(np.linalg.norm(a-b)/math.sqrt(len(a)))
def identification(prefix,fams,layer=None):
 eligible,keys,X=matrix(prefix,fams,layer);counts=defaultdict(int)
 for r in eligible:counts[r["prompt_id"]]+=1
 keep_idx=[i for i,r in enumerate(eligible) if counts[r["prompt_id"]]==3];eligible=[eligible[i] for i in keep_idx];X=X[keep_idx];results=[]
 for held_seed in sorted({r["seed"] for r in eligible}):
  train_idx=[i for i,r in enumerate(eligible) if r["seed"]!=held_seed];test_idx=[i for i,r in enumerate(eligible) if r["seed"]==held_seed]
  train=X[train_idx];test=X[test_idx];train_z,keep=fit_transform(train);base=train;med=np.nanmedian(base,axis=0)[keep];scale=(np.nanquantile(base,.75,axis=0)-np.nanquantile(base,.25,axis=0))[keep];testk=np.where(np.isfinite(test[:,keep]),test[:,keep],med);test_z=(testk-med)/scale
  train_meta=[eligible[i] for i in train_idx];test_meta=[eligible[i] for i in test_idx];centroids={pid:np.mean([train_z[i] for i,r in enumerate(train_meta) if r["prompt_id"]==pid],axis=0) for pid in {r["prompt_id"] for r in train_meta}}
  for x,r in zip(test_z,test_meta):
   ranked=sorted(centroids,key=lambda pid:dist(x,centroids[pid]));same=[pid for pid in ranked if next(m for m in train_meta if m["prompt_id"]==pid)["prompt_family"]==r["prompt_family"]]
   results.append({"run_id":r["run_id"],"prompt_id":r["prompt_id"],"seed":held_seed,"rank":ranked.index(r["prompt_id"])+1,"same_family_rank":same.index(r["prompt_id"])+1})
 return {"eligible_prompt_challenges":len(counts) if prefix=="FULL" else sum(v==3 for v in counts.values()),"n_tests":len(results),"top1":sum(x["rank"]==1 for x in results)/len(results),"top5":sum(x["rank"]<=5 for x in results)/len(results),"balanced_accuracy":sum(x["rank"]==1 for x in results)/len(results),"same_family_top1":sum(x["same_family_rank"]==1 for x in results)/len(results),"random_top1":1/len(counts) if prefix=="FULL" else 1/sum(v==3 for v in counts.values()),"same_family_random_top1":.25 if prefix=="FULL" else "REALIZED_CANDIDATE_SET_VARIES"}
def distance_analysis(prefix,fams,layer=None,bootstrap=False):
 eligible,keys,X=matrix(prefix,fams,layer);Z,keep=fit_transform(X);intra=[];inter=[];same=[];matched=[];inter_len=[]
 for i in range(len(eligible)):
  for j in range(i+1,len(eligible)):
   d=dist(Z[i],Z[j]);a,b=eligible[i],eligible[j]
   if a["prompt_id"]==b["prompt_id"]:intra.append(d)
   else:
    inter.append(d);inter_len.append((d,abs(a["n_steps"]-b["n_steps"]),abs(a["prompt_len"]-b["prompt_len"]),abs(a["seq_len"]-b["seq_len"])))
    if a["prompt_family"]==b["prompt_family"]:same.append(d)
    if abs(a["n_steps"]-b["n_steps"])<=2:matched.append(d)
 auc=float(np.mean(np.asarray(inter)[:,None]>np.asarray(intra)[None,:]));same_auc=float(np.mean(np.asarray(same)[:,None]>np.asarray(intra)[None,:])) if same else None;matched_auc=float(np.mean(np.asarray(matched)[:,None]>np.asarray(intra)[None,:])) if matched else None
 result={"eligible_runs":len(eligible),"retained_features":int(keep.sum()),"intra":{"n":len(intra),"median":float(np.median(intra)),"p10":float(np.quantile(intra,.1)),"p90":float(np.quantile(intra,.9))},"inter":{"n":len(inter),"median":float(np.median(inter)),"p10":float(np.quantile(inter,.1)),"p90":float(np.quantile(inter,.9))},"same_family_inter":{"n":len(same),"median":float(np.median(same)) if same else None},"length_matched_inter":{"rule":"abs generated token difference <= 2","n":len(matched),"median":float(np.median(matched)) if matched else None},"common_language_separation":auc,"ordering_overlap_rate":1-auc,"rank_biserial":2*auc-1,"same_family_separation":same_auc,"length_matched_separation":matched_auc,"length_distance_spearman_rho":float(spearmanr([x[0] for x in inter_len],[x[1] for x in inter_len]).statistic),"prompt_length_distance_spearman_rho":float(spearmanr([x[0] for x in inter_len],[x[2] for x in inter_len]).statistic),"total_sequence_length_distance_spearman_rho":float(spearmanr([x[0] for x in inter_len],[x[3] for x in inter_len]).statistic)}
 if bootstrap:
  rng=np.random.default_rng(RNG_SEED);diff=[]
  for _ in range(2000):diff.append(np.median(rng.choice(inter,len(inter),replace=True))-np.median(rng.choice(intra,len(intra),replace=True)))
  result["median_difference_95pct_bootstrap_ci"]=[float(np.quantile(diff,.025)),float(np.quantile(diff,.975))]
 return result

def signal_status(distance_result,id_result,length_confound=False):
 globalish=distance_result["common_language_separation"]>=.75 and id_result["top1"]>=.20
 if globalish and (id_result["same_family_top1"]<=.30 or (abs(distance_result["length_distance_spearman_rho"])>=.50 and (distance_result["length_matched_separation"] or 0)<.65)):return "TENSOR_FINGERPRINT_SIGNAL_CONFOUNDED"
 if globalish and (distance_result["same_family_separation"] or 0)>=.65 and id_result["same_family_top1"]>=.50:return "TENSOR_FINGERPRINT_SIGNAL_SUPPORTED"
 if distance_result["common_language_separation"]>=.65 or id_result["top1"]>=.10:return "TENSOR_FINGERPRINT_SIGNAL_PARTIAL"
 return "TENSOR_FINGERPRINT_SIGNAL_NOT_SUPPORTED"

# Frozen distance, prefix, ablation, and strict seed-holdout identification analyses.
primary=distance_analysis("FULL",FAMILIES,bootstrap=True);primary_id=identification("FULL",FAMILIES)
ablation={}
for name,fams in COMBOS.items():
 d=distance_analysis("FULL",fams);ident=identification("FULL",fams);ablation[name]={"distance":d,"identification":ident,"signal_status":signal_status(d,ident)}
prefix_results={}
for prefix in PREFIXES:
 d=distance_analysis(prefix,FAMILIES);ident=identification(prefix,FAMILIES);prefix_results[prefix]={"distance":d,"identification":ident,"signal_status":signal_status(d,ident)}
dump("INTRA_INTER_DISTANCE_RESULTS.json",{"protocol":"FROZEN_PROTOCOL.md","primary":"ALL/FULL","result":primary})
dump("SAME_FAMILY_CONTROL.json",{"primary":"ALL/FULL","same_family_inter_distance":primary["same_family_inter"],"same_family_separation":primary["same_family_separation"],"identification_top1":primary_id["same_family_top1"],"random_baseline":.25,"interpretation":"Controls prompt-family classification by restricting comparison candidates to four prompts in the true family."})
rho=abs(primary["length_distance_spearman_rho"]);drop=primary["common_language_separation"]-(primary["length_matched_separation"] or 0);length_status="YES" if (rho>=.5 and (primary["length_matched_separation"] or 0)<.65) else "PARTIAL" if rho>=.25 or drop>=.10 else "NO"
dump("LENGTH_CONFOUND_AUDIT.json",{"generated_length":{"spearman_rho_distance_vs_absolute_length_difference":primary["length_distance_spearman_rho"],"length_matched_separation":primary["length_matched_separation"],"status":length_status},"prompt_token_count":{"spearman_rho_distance_vs_absolute_length_difference":primary["prompt_length_distance_spearman_rho"],"status":"DERIVED_CONFOUND_DIAGNOSTIC","limitation":"prompt_len is nearly prompt-identity-deterministic and cannot independently identify mechanism."},"total_sequence_length":{"spearman_rho_distance_vs_absolute_length_difference":primary["total_sequence_length_distance_spearman_rho"],"status":"DERIVED_CONFOUND_DIAGNOSTIC","limitation":"seq_len combines prompt identity and generation length."}})
dump("PREFIX_FINGERPRINT_RESULTS.json",prefix_results);dump("TENSOR_ABLATION_RESULTS.json",ablation);dump("CHALLENGE_IDENTIFICATION_RESULTS.json",{"task":"CHALLENGE FINGERPRINT IDENTIFICATION","validation":"strict rotating seed holdout; two-seed reference centroid, third-seed test","full_all":primary_id,"random_baselines":{"global_top1":1/48,"same_family_top1":.25},"security_interpretation":"NOT authentication"})

# Per-prompt seed stability from primary ALL/FULL space.
eligible,keys,X=matrix("FULL",FAMILIES);Z,keep=fit_transform(X);stability=[]
for pid in sorted({r["prompt_id"] for r in eligible}):
 idx=[i for i,r in enumerate(eligible) if r["prompt_id"]==pid];ds=[dist(Z[idx[i]],Z[idx[j]]) for i in range(len(idx)) for j in range(i+1,len(idx))];stability.append({"prompt_id":pid,"prompt_family":eligible[idx[0]]["prompt_family"],"intra_prompt_seed_variability":float(np.median(ds)),"mean":float(np.mean(ds)),"n_pairs":len(ds)})
values=np.array([x["intra_prompt_seed_variability"] for x in stability]);q1,q2=np.quantile(values,[1/3,2/3])
for x in stability:x["stability_class"]="MOST_STABLE" if x["intra_prompt_seed_variability"]<=q1 else "MODERATE" if x["intra_prompt_seed_variability"]<=q2 else "UNSTABLE"
with (OUT/"SEED_STABILITY.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(stability[0]));w.writeheader();w.writerows(stability)

# Layer localization is descriptive-only and uses same-family controlled separation.
layer_rows=[]
for fam in ["H","Q","K","V","ATTENTION","ATTENTION_OUTPUT"]:
 max_layer=25 if fam=="H" else 24
 for layer in range(max_layer):
  d=distance_analysis("FULL",[fam],layer);layer_rows.append({"tensor_family":fam,"layer":layer,"retained_features":d["retained_features"],"intra_median":d["intra"]["median"],"inter_median":d["inter"]["median"],"same_family_separation":d["same_family_separation"],"global_separation":d["common_language_separation"],"seed_stability":"DESCRIPTIVE_NETWORK_PROPERTY","epistemic_status":"DERIVED"})
with (OUT/"LAYER_LOCALIZATION.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(layer_rows[0]));w.writeheader();w.writerows(layer_rows)

# Secondary landmark relation: use Lane B's existing per-run detection count, never as a primary feature.
stab_by_prompt={x["prompt_id"]:x["intra_prompt_seed_variability"] for x in stability};bearing=[stab_by_prompt[r["prompt_id"]] for r in rows if r["n_landmarks"]>0];none=[stab_by_prompt[r["prompt_id"]] for r in rows if r["n_landmarks"]==0]
dump("LANDMARK_RELATION_AUDIT.json",{"role":"SECONDARY_EXPLORATORY","landmark_presence_used_in_primary_fingerprint":False,"landmark_bearing_run_count":sum(r["n_landmarks"]>0 for r in rows),"eligible_no_landmark_run_count":sum(r["n_landmarks"]==0 for r in rows),"median_prompt_seed_variability_weighted_by_runs":{"landmark_bearing":float(np.median(bearing)),"no_landmark":float(np.median(none))},"verdict":"NOT_ESTABLISHED","reason":"Existing Lane B landmark counts are descriptive candidate detections, not a frozen causal or attestation factor; run-weighted groups are prompt-dependent."})

# Final verdict follows frozen rules and length audit.
verdict=signal_status(primary,primary_id)
if verdict=="TENSOR_FINGERPRINT_SIGNAL_SUPPORTED" and length_status=="PARTIAL":verdict="TENSOR_FINGERPRINT_SIGNAL_PARTIAL"
manifest={"gate":"TENSOR RUNTIME FINGERPRINT FEASIBILITY V0","status":"EXECUTED","date":"2026-08-18","source_runs":144,"real_new_generations":0,"prompt_challenges":48,"seeds":[2101,2102,2103],"prompt_families":12,"source_hash_verification":{"passed":144,"total":144},"primary_configuration":"ALL/FULL","primary_signal_status":ablation["ALL"]["signal_status"],"length_confound":length_status,"final_verdict":verdict,"cryptographic_authentication":"NOT_TESTED","pqc":"NOT_TESTED","entropy_source":"NOT_TESTED","security_product_claim":"NO","control":"NO"}
dump("MANIFEST.json",manifest)
dump("SOURCE_PROVENANCE.json",{"source":"reports/bcc-atlas-expansion-v1/ATLAS_DEEP","run_index":"RUN_INDEX.csv","canonical_hash_registry":"SHA256SUMS.txt","run_count":144,"artifact_hashes":source_hashes,"canonical_hash_verification":{"passed":144,"total":144},"run_index_artifact_hash_match_count":run_index_hash_matches,"run_index_artifact_hash_limitation":"RUN_INDEX artifact_hash values may represent a pre-finalization/content-stage hash and are not treated as the final file SHA-256; final file bytes match SHA256SUMS.txt.","source_manifest_sha256":sha(SRC/"MANIFEST.json"),"source_artifacts_modified":False,"new_generations":0})
contract={"feature":"Runtime Fingerprint","mode":"READ_ONLY_RESEARCH","human_status_labels":["FINGERPRINT OBSERVED","FINGERPRINT STABILITY","REFERENCE SUPPORT","ENVIRONMENT COVERAGE","SECURITY VALIDATION"],"forbidden_status_labels":["AUTHENTICATED"],"example":{"title":"Runtime fingerprint","observation":"Observed across 3 independent seeds","stability":"High / Medium / Low","reference_coverage":"48 challenges / 12 families","security_status":"Research signal only","technical_details":"expandable"},"evidence_fields":["run_id","prompt_id","seed","source_sha256","feature_registry","epistemic_status"],"identity_preservation":True,"security_boundary":"No authentication without a future cryptographic protocol."}
dump("STAR_COMMAND_SECURITY_FEATURE_CONTRACT.json",contract)

(OUT/"SECURITY_INTERPRETATION_BOUNDARY.md").write_text("""# Security interpretation boundary\n\nThis feasibility gate does not establish authentication, cryptographic entropy, unclonability, quantum resistance, ownership, production security, attestation security, or control. A positive result means only that an internal execution fingerprint candidate exists in one captured model/runtime campaign. Required future tests include multiple models/checkpoints, quantization, hardware and runtime environments, replay and adversarial imitation, fine-tuning, pruning, perturbation, challenge freshness, FAR/FRR under attack, and cryptographic sealing.\n""")
(OUT/"FUTURE_ATTESTATION_ARCHITECTURE.md").write_text("""# Future attestation architecture — not implemented\n\n`runtime fingerprint → canonical evidence object → cryptographic hash → PQC signature / attestation layer`\n\nThe current gate implements only the first research object and evidence references. It implements no hash-based identity protocol, key management, signature, PQC, freshness mechanism, verifier, or authentication claim.\n""")
(OUT/"EPISTEMIC_STATUS.md").write_text(f"""# Epistemic status\n\nRaw tensors and run metadata are **MEASURED**. Bounded summaries, distances, effect sizes, stability classes, identification rates, and layer localization are **DERIVED**. Landmark association is **NOT_ESTABLISHED**. Authentication, PQC, entropy-source quality, ownership, adversarial robustness, and cross-environment stability are **NOT_TESTED**. Final feasibility verdict: **{verdict}**.\n""")
best=max(stability,key=lambda x:x["intra_prompt_seed_variability"]);early=prefix_results["FIRST_2"]["signal_status"]
report=f"""# Tensor Runtime Fingerprint Feasibility V0\n\n## Result\n\n**{verdict}** across 144 existing Lane B runs; zero new generations. Full ALL-tensor challenge identification top-1 is {primary_id['top1']:.2%} (random 2.08%) and top-5 is {primary_id['top5']:.2%}. Same-family top-1 is {primary_id['same_family_top1']:.2%} (random 25%).\n\nPrimary normalized distance medians are {primary['intra']['median']:.4f} intra-prompt and {primary['inter']['median']:.4f} inter-prompt; common-language separation is {primary['common_language_separation']:.3f}, same-family separation {primary['same_family_separation']:.3f}, and length-matched separation {primary['length_matched_separation']:.3f}. Generated-length confound status: **{length_status}**.\n\nEarly FIRST_2 status: **{early}**. Prompt stability classes are quantile-based descriptions, not security grades. The largest observed intra-seed variability is {best['prompt_id']} ({best['intra_prompt_seed_variability']:.4f}).\n\nLandmark relation is **NOT_ESTABLISHED**. This report makes no authentication, attestation-security, cryptographic, PQC, entropy-source, ownership, or control claim.\n"""
(OUT/"REPORT.md").write_text(report)

required=["FROZEN_PROTOCOL.md","SOURCE_PROVENANCE.json","FEATURE_REGISTRY.json","RUN_FINGERPRINTS.csv","INTRA_INTER_DISTANCE_RESULTS.json","SAME_FAMILY_CONTROL.json","LENGTH_CONFOUND_AUDIT.json","SEED_STABILITY.csv","PREFIX_FINGERPRINT_RESULTS.json","TENSOR_ABLATION_RESULTS.json","LAYER_LOCALIZATION.csv","CHALLENGE_IDENTIFICATION_RESULTS.json","LANDMARK_RELATION_AUDIT.json","SECURITY_INTERPRETATION_BOUNDARY.md","FUTURE_ATTESTATION_ARCHITECTURE.md","EPISTEMIC_STATUS.md","REPORT.md","MANIFEST.json","STAR_COMMAND_SECURITY_FEATURE_CONTRACT.json","build_fingerprint.py","refresh_length_audit.py","test_fingerprint.py"]
(OUT/"SHA256SUMS.txt").write_text("".join(f"{sha(OUT/name)}  {name}\n" for name in required))
print(json.dumps({"verdict":verdict,"top1":primary_id["top1"],"same_family_top1":primary_id["same_family_top1"],"length_confound":length_status,"early":early,"source_hashes":"144/144"},indent=2))
