#!/usr/bin/env python3
"""Cross-environment analysis using the frozen V0 feature registry."""
from __future__ import annotations
import csv, hashlib, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
BASE=ROOT/"reports/bcc-atlas-expansion-v1/ATLAS_DEEP"
V0=ROOT/"reports/tensor-runtime-fingerprint-v0"
PREFIXES={"FULL":None,"FIRST_2":2,"FIRST_4":4,"FIRST_8":8}
FAMILIES=["H","Q","K","V","ATTENTION","ATTENTION_OUTPUT","LOGITS"]
STATS=("mean","std","median","p10","p90")
BASE_ENV="baseline_cuda_bfloat16_eager"

def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def dump(name,obj):(HERE/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
def stats(v):
 v=np.asarray(v,dtype=np.float64);v=v[np.isfinite(v)]
 if not len(v):return {s:None for s in STATS}
 return dict(mean=float(v.mean()),std=float(v.std()),median=float(np.median(v)),p10=float(np.quantile(v,.1)),p90=float(np.quantile(v,.9)))
def add(out,prefix,layer,metric,values):
 for s,v in stats(values).items():out[f"{prefix}.L{layer:02d}.{metric}.{s}"]=v
def coschg(a,b):return 1-np.sum(a*b,axis=-1)/np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1),1e-12)
def vector(arr,family,pos,heads=False,curvature=False):
 out={};a=np.asarray(arr,dtype=np.float32)
 for layer in range(a.shape[0]):
  x=a[layer,:,pos,:] if heads else a[layer,pos,:];norm=np.linalg.norm(x,axis=-1)/math.sqrt(x.shape[-1]);add(out,family,layer,"norm",norm.ravel())
  if heads:add(out,family,layer,"head_dispersion",np.std(norm,axis=1))
  if len(pos)>=2:
   d=np.diff(x,axis=0);add(out,family,layer,"delta_norm",(np.linalg.norm(d,axis=-1)/math.sqrt(x.shape[-1])).ravel());add(out,family,layer,"cosine_change",coschg(x[1:],x[:-1]).ravel())
   if curvature and len(pos)>=3:add(out,family,layer,"curvature",coschg(d[1:],d[:-1]).ravel())
  else:
   for metric in ("delta_norm","cosine_change")+(("curvature",) if curvature else ()):add(out,family,layer,metric,[])
 return out
def attention(arr,pos):
 out={};a=np.asarray(arr,dtype=np.float32)
 for layer in range(a.shape[0]):
  ent=[];conc=[];maximum=[];disp=[]
  for p in pos:
   w=np.maximum(a[layer,:,p,:p+1],0);w=w/np.maximum(w.sum(axis=-1,keepdims=True),1e-12);ent.extend((-np.sum(w*np.log(np.maximum(w,1e-12)),axis=-1)/max(math.log(p+1),1)).tolist());conc.extend(np.sum(w*w,axis=-1).tolist());maximum.extend(np.max(w,axis=-1).tolist());idx=np.arange(p+1,dtype=np.float32)/max(p,1);mu=np.sum(w*idx,axis=-1);disp.extend(np.sum(w*(idx-mu[:,None])**2,axis=-1).tolist())
  for metric,vals in (("normalized_entropy",ent),("concentration",conc),("maximum_mass",maximum),("normalized_position_dispersion",disp)):add(out,"ATTENTION",layer,metric,vals)
 return out
def logits(arr,count):
 out={};x=np.asarray(arr[:count],dtype=np.float32)
 if not len(x):return out
 shifted=x-x.max(1,keepdims=True);ex=np.exp(shifted);p=ex/ex.sum(1,keepdims=True);e=-np.sum(p*np.log(np.maximum(p,1e-30)),1);top=np.partition(p,-2,axis=1)[:,-2:];top.sort(1)
 add(out,"LOGITS",0,"entropy",e);add(out,"LOGITS",0,"top1_probability",top[:,1]);add(out,"LOGITS",0,"top2_gap",top[:,1]-top[:,0]);add(out,"LOGITS",0,"norm",np.linalg.norm(x,axis=1)/math.sqrt(x.shape[1]))
 if len(x)>=2:add(out,"LOGITS",0,"delta_norm",np.linalg.norm(np.diff(x,axis=0),axis=1)/math.sqrt(x.shape[1]));add(out,"LOGITS",0,"cosine_change",coschg(x[1:],x[:-1]))
 else:add(out,"LOGITS",0,"delta_norm",[]);add(out,"LOGITS",0,"cosine_change",[])
 return out
def extract(path,meta,k):
 n=meta["n_steps"] if k is None else k;pos=np.arange(meta["prompt_len"],meta["prompt_len"]+n,dtype=int)
 with np.load(path,allow_pickle=False) as z:
  return {"H":vector(z["hidden_states_all_layers_fp16"],"H",pos,curvature=True),"Q":vector(z["q_all_layers_fp16"],"Q",pos,heads=True),"K":vector(z["k_all_layers_fp32"],"K",pos,heads=True),"V":vector(z["v_all_layers_fp32"],"V",pos,heads=True),"ATTENTION":attention(z["attentions_all_layers_fp16"],pos),"ATTENTION_OUTPUT":vector(z["attn_output_all_layers_fp16"],"ATTENTION_OUTPUT",pos,curvature=True),"LOGITS":logits(z["logits_region_fp16"],n)}
def robust_fit(X):
 med=np.nanmedian(X,0);q1=np.nanquantile(X,.25,axis=0);q3=np.nanquantile(X,.75,axis=0);scale=q3-q1;keep=np.isfinite(med)&np.isfinite(scale)&(scale>1e-9);return med[keep],scale[keep],keep
def transform(X,fit):
 med,scale,keep=fit;Y=X[:,keep];return (np.where(np.isfinite(Y),Y,med)-med)/scale
def distance(a,b):return float(np.linalg.norm(a-b)/math.sqrt(len(a)))
def summarize(v):
 v=np.asarray(v);return {"n":len(v),"median":float(np.median(v)),"p10":float(np.quantile(v,.1)),"p90":float(np.quantile(v,.9))}

panel=json.loads((HERE/"CHALLENGE_PANEL.json").read_text())["challenges"];pids={x["prompt_id"] for x in panel};family={x["prompt_id"]:x["prompt_family"] for x in panel}
baseline=[]
with (BASE/"RUN_INDEX.csv").open() as f:
 for x in csv.DictReader(f):
  if x["prompt_id"] in pids:baseline.append({"run_id":x["run_id"],"environment_id":BASE_ENV,"model_id":"Qwen/Qwen2.5-0.5B-Instruct","checkpoint_revision":"7ae557604adf67be50417f59c2c2f167def9a775","prompt_id":x["prompt_id"],"prompt_family":x["prompt_family"],"seed":int(x["seed"]),"dtype":"bfloat16","device":"cuda","attention_backend":"eager","prompt_len":int(x["prompt_len"]),"seq_len":int(x["seq_len"]),"n_steps":int(x["n_steps"]),"artifact_path":f"reports/bcc-atlas-expansion-v1/ATLAS_DEEP/raw_captures/{x['run_id']}.npz","artifact_sha256":x["artifact_hash"],"source":"REUSED_V0_SOURCE"})
new=[json.loads(x) for x in (HERE/"new_run_index.jsonl").read_text().splitlines() if x.strip()]
assert len(baseline)==36 and len(new)==108 and all(x["status"]=="SUCCESS" for x in new)
rows=baseline+[{**x,"source":"NEW_BOUNDED_CAMPAIGN"} for x in new];assert len(rows)==144
canonical={line.split()[1]:line.split()[0] for line in (BASE/"SHA256SUMS.txt").read_text().splitlines() if line.strip()}
for r in rows:
 p=ROOT/r["artifact_path"]
 actual=sha(p)
 if r["source"]=="NEW_BOUNDED_CAMPAIGN" and actual!=r["artifact_sha256"]:raise RuntimeError(f"hash mismatch {p}")
 if r["source"]=="REUSED_V0_SOURCE" and actual!=canonical.get(f"raw_captures/{r['run_id']}.npz"):raise RuntimeError(f"canonical source hash mismatch {p}")
 r["verified_sha256"]=actual

fields=["run_id","environment_id","model_id","checkpoint_revision","prompt_id","prompt_family","seed","dtype","device","attention_backend","prompt_len","seq_len","n_steps","artifact_path","verified_sha256","source"]
with (HERE/"RUN_INDEX.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:r[k] for k in fields} for r in rows])

env_inv=json.loads((HERE/"ENVIRONMENT_INVENTORY.json").read_text());tested=[x for x in env_inv["candidate_environments"] if x["environment_id"] in {r["environment_id"] for r in rows}]
env_fields=["environment_id","availability","availability_scope","model_id","checkpoint_revision","weights_sha256","tokenizer_json_sha256","tokenizer_config_sha256","dtype","device","hardware_identifier","torch_version","transformers_version","attention_backend","quantization","generation_config"]
with (HERE/"ENVIRONMENT_METADATA.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=env_fields);w.writeheader()
 for e in tested:
  historical=e["device"]=="cuda";rec={"environment_id":e["environment_id"],"availability":e["status"],"availability_scope":e["availability_scope"],"model_id":env_inv["base_model"],"checkpoint_revision":env_inv["checkpoint_revision"],"weights_sha256":env_inv["checkpoint_sha256"],"tokenizer_json_sha256":env_inv["tokenizer_sha256"],"tokenizer_config_sha256":env_inv["tokenizer_config_sha256"],"dtype":e["dtype"],"device":e["device"],"hardware_identifier":"NOT_RECORDED_IN_SOURCE" if historical else env_inv["system"]["hardware_identifier"],"torch_version":"NOT_RECORDED_IN_SOURCE" if historical else env_inv["system"]["torch"],"transformers_version":"NOT_RECORDED_IN_SOURCE" if historical else env_inv["system"]["transformers"],"attention_backend":e["attention_backend"],"quantization":"NONE","generation_config":json.dumps(json.loads((HERE/"EXECUTION_PLAN.json").read_text())["generation_config"],sort_keys=True)};w.writerow(rec)

features={p:{} for p in PREFIXES}
for i,r in enumerate(rows,1):
 for prefix,k in PREFIXES.items():
  if k is None or r["n_steps"]>=k:features[prefix][r["run_id"]]=extract(ROOT/r["artifact_path"],r,k)
 print(f"EXTRACT {i}/144 {r['run_id']}",flush=True)

def matrix(prefix,fams,layer=None):
 eligible=[r for r in rows if r["run_id"] in features[prefix]];keys=sorted({key for r in eligible for fam in fams for key in features[prefix][r["run_id"]][fam] if layer is None or f".L{layer:02d}." in key})
 X=np.asarray([[next((features[prefix][r["run_id"]][fam][key] for fam in fams if key in features[prefix][r["run_id"]][fam]),np.nan) for key in keys] for r in eligible],float);return eligible,keys,X
def evaluate(prefix,fams,layer=None):
 eligible,keys,X=matrix(prefix,fams,layer);base_idx=[i for i,r in enumerate(eligible) if r["environment_id"]==BASE_ENV];fit=robust_fit(X[base_idx]);Z=transform(X,fit);z={r["run_id"]:Z[i] for i,r in enumerate(eligible)};by={(r["environment_id"],r["prompt_id"],r["seed"]):r for r in eligible}
 seed=[];challenge=[]
 base=[r for r in eligible if r["environment_id"]==BASE_ENV]
 for i,a in enumerate(base):
  for b in base[i+1:]:
   d=distance(z[a["run_id"]],z[b["run_id"]])
   if a["prompt_id"]==b["prompt_id"]:seed.append(d)
   else:challenge.append(d)
 counts=defaultdict(int)
 for r in base:counts[r["prompt_id"]]+=1
 active_pids={pid for pid,n in counts.items() if n==3}
 env_results={};centroids={pid:np.mean([z[r["run_id"]] for r in base if r["prompt_id"]==pid],0) for pid in active_pids}
 # rotating seed holdout in the baseline panel
 branks=[]
 for probe in [r for r in base if r["prompt_id"] in active_pids]:
  cents={pid:np.mean([z[r["run_id"]] for r in base if r["prompt_id"]==pid and r["seed"]!=probe["seed"]],0) for pid in active_pids}
  ranked=sorted(active_pids,key=lambda pid:distance(z[probe["run_id"]],cents[pid]));branks.append(ranked.index(probe["prompt_id"])+1)
 baseline_id={"n":len(branks),"top1":sum(x==1 for x in branks)/len(branks),"top5":sum(x<=5 for x in branks)/len(branks)}
 for env in sorted({r["environment_id"] for r in eligible}-{BASE_ENV}):
  probes=[r for r in eligible if r["environment_id"]==env and r["prompt_id"] in active_pids];ds=[];ranks=[]
  for r in probes:
   b=by.get((BASE_ENV,r["prompt_id"],r["seed"]));
   if b:ds.append(distance(z[r["run_id"]],z[b["run_id"]]))
   ranked=sorted(active_pids,key=lambda pid:distance(z[r["run_id"]],centroids[pid]));ranks.append(ranked.index(r["prompt_id"])+1)
  top1=sum(x==1 for x in ranks)/len(ranks);status="FINGERPRINT_PRESERVED" if np.median(ds)<=np.quantile(seed,.9) and top1>=baseline_id["top1"]-.15 else "FINGERPRINT_SHIFTED" if np.median(ds)<np.median(challenge) else "FINGERPRINT_BROKEN"
  env_results[env]={"matched_environment_distance":summarize(ds),"identification":{"n":len(ranks),"top1":top1,"top5":sum(x<=5 for x in ranks)/len(ranks)},"stability_class":status}
 return {"retained_features":int(fit[2].sum()),"D_seed":summarize(seed),"D_challenge":summarize(challenge),"baseline_identification":baseline_id,"environments":env_results},z,centroids,eligible

primary,z,centroids,eligible=evaluate("FULL",FAMILIES);dump("CROSS_ENV_DISTANCE_RESULTS.json",{"primary":"ALL/FULL","distance":"frozen robust-scaled Euclidean / sqrt(features)","baseline_scaler_only":True,"result":primary})
env_rows=[]
for env,v in primary["environments"].items():env_rows.append({"environment_id":env,"variation":"dtype+device_from_historical_baseline","D_env_median":v["matched_environment_distance"]["median"],"D_seed_p90_threshold":primary["D_seed"]["p90"],"D_challenge_median":primary["D_challenge"]["median"],"baseline_top1":primary["baseline_identification"]["top1"],"cross_environment_top1":v["identification"]["top1"],"stability_class":v["stability_class"],"epistemic_status":"DERIVED"})
with (HERE/"ENVIRONMENT_STABILITY_MATRIX.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(env_rows[0]));w.writeheader();w.writerows(env_rows)
cross_n=sum(v["identification"]["n"] for v in primary["environments"].values());cross_top=sum(v["identification"]["top1"]*v["identification"]["n"] for v in primary["environments"].values())/cross_n;cross_top5=sum(v["identification"]["top5"]*v["identification"]["n"] for v in primary["environments"].values())/cross_n
ident={"task":"CHALLENGE FINGERPRINT IDENTIFICATION — NOT AUTHENTICATION","baseline":primary["baseline_identification"],"alternate_environments":{k:v["identification"] for k,v in primary["environments"].items()},"pooled_cross_environment":{"n":cross_n,"top1":cross_top,"top5":cross_top5},"random_top1":1/12,"same_family_cross_environment":{"status":"NOT_ESTABLISHED","reason":"Frozen panel has exactly one challenge per family; within-family candidate count is one.","candidate_count_per_family":1}}
dump("IDENTIFICATION_RESULTS.json",ident)

# Research verification proxy. The threshold is frozen from leave-one-seed baseline distances.
base=[r for r in eligible if r["environment_id"]==BASE_ENV];genuine=[];impostor=[]
for probe in base:
 cents={pid:np.mean([z[r["run_id"]] for r in base if r["prompt_id"]==pid and r["seed"]!=probe["seed"]],0) for pid in pids}
 genuine.append(distance(z[probe["run_id"]],cents[probe["prompt_id"]]));impostor.extend(distance(z[probe["run_id"]],cents[pid]) for pid in pids if pid!=probe["prompt_id"])
candidates=sorted(set(genuine+impostor));threshold=min(candidates,key=lambda t:abs(np.mean(np.asarray(impostor)<=t)-np.mean(np.asarray(genuine)>t)))
probes=[r for r in eligible if r["environment_id"]!=BASE_ENV];cg=[];ci=[]
for r in probes:
 cg.append(distance(z[r["run_id"]],centroids[r["prompt_id"]]));ci.extend(distance(z[r["run_id"]],centroids[pid]) for pid in pids if pid!=r["prompt_id"])
far=float(np.mean(np.asarray(ci)<=threshold));frr=float(np.mean(np.asarray(cg)>threshold));labels=np.r_[np.ones(len(cg)),np.zeros(len(ci))];scores=-np.r_[cg,ci];auc=float(roc_auc_score(labels,scores));eer_t=min(sorted(set(cg+ci)),key=lambda t:abs(np.mean(np.asarray(ci)<=t)-np.mean(np.asarray(cg)>t)));eer=float((np.mean(np.asarray(ci)<=eer_t)+np.mean(np.asarray(cg)>eer_t))/2)
dump("VERIFICATION_PROXY_RESULTS.json",{"label":"RESEARCH VERIFICATION PROXY","authentication":False,"threshold":{"value":threshold,"selection":"baseline-only closest FAR-like/FRR-like operating point","frozen_before_alternate_labels":True},"alternate_probes":{"genuine_n":len(cg),"impostor_n":len(ci),"far_like":far,"frr_like":frr,"roc_auc":auc,"eer":eer,"eer_statistically_meaningful":len(cg)>=30}})

prefix={}
for name in PREFIXES:
 res,_,_,_=evaluate(name,FAMILIES);n=sum(v["identification"]["n"] for v in res["environments"].values());top=sum(v["identification"]["top1"]*v["identification"]["n"] for v in res["environments"].values())/n;classes=[v["stability_class"] for v in res["environments"].values()];prefix[name]={"eligible_cross_environment_probes":n,"cross_environment_top1":top,"environment_classes":dict((k,v["stability_class"]) for k,v in res["environments"].items()),"status":"FINGERPRINT_PRESERVED" if classes.count("FINGERPRINT_PRESERVED")>=2 else "FINGERPRINT_SHIFTED" if any(x!="FINGERPRINT_BROKEN" for x in classes) else "FINGERPRINT_BROKEN"}
dump("PREFIX_ROBUSTNESS.json",prefix)

tensor={}
for fam in FAMILIES+["ALL"]:
 fams=FAMILIES if fam=="ALL" else [fam];res,_,_,_=evaluate("FULL",fams);classes=[v["stability_class"] for v in res["environments"].values()];tensor[fam]={"baseline_identification":res["baseline_identification"],"environments":res["environments"],"status":"FINGERPRINT_PRESERVED" if classes.count("FINGERPRINT_PRESERVED")>=2 else "FINGERPRINT_SHIFTED" if any(x!="FINGERPRINT_BROKEN" for x in classes) else "FINGERPRINT_BROKEN"}
dump("TENSOR_ROBUSTNESS.json",tensor)

layers=[]
for fam in ["H","Q","K","V","ATTENTION","ATTENTION_OUTPUT"]:
 for layer in range(25 if fam=="H" else 24):
  res,_,_,_=evaluate("FULL",[fam],layer);classes=[v["stability_class"] for v in res["environments"].values()];layers.append({"tensor_family":fam,"layer":layer,"retained_features":res["retained_features"],"D_seed_p90":res["D_seed"]["p90"],"D_challenge_median":res["D_challenge"]["median"],"preserved_environment_count":classes.count("FINGERPRINT_PRESERVED"),"shifted_environment_count":classes.count("FINGERPRINT_SHIFTED"),"broken_environment_count":classes.count("FINGERPRINT_BROKEN"),"descriptive_class":"ENVIRONMENT_STABLE_LAYERS" if classes.count("FINGERPRINT_PRESERVED")>=2 else "ENVIRONMENT_SENSITIVE_LAYERS","epistemic_status":"DERIVED"})
with (HERE/"LAYER_ROBUSTNESS.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(layers[0]));w.writeheader();w.writerows(layers)

preserved=sum(x["stability_class"]=="FINGERPRINT_PRESERVED" for x in env_rows)
verdict="CROSS_ENV_FINGERPRINT_ROBUSTNESS_SUPPORTED" if preserved>=2 and cross_top>=.5 and auc>=.7 else "CROSS_ENV_FINGERPRINT_ROBUSTNESS_PARTIAL" if any(x["stability_class"]!="FINGERPRINT_BROKEN" for x in env_rows) and auc>=.7 else "CROSS_ENV_FINGERPRINT_ROBUSTNESS_NOT_SUPPORTED"
dump("FUTURE_ATTESTATION_CONTRACT.json",{"status":"CONCEPT_ONLY","flow":["challenge","runtime fingerprint","canonical evidence object","hash","PQC signature","verification record"],"implemented_here":["challenge","runtime fingerprint","canonical evidence object references"],"not_implemented":["PQC","signature","authentication","freshness","anti-replay","attestation security"]})
(HERE/"SECURITY_BOUNDARY.md").write_text("# Security boundary\n\nThis robustness gate is not authentication, attestation security, PQC, entropy generation, ownership proof, anti-replay, unclonability, or a control mechanism. A positive result establishes only runtime fingerprint robustness under the explicitly tested conditions. FAR-like and FRR-like values are a **RESEARCH VERIFICATION PROXY**.\n")
(HERE/"EPISTEMIC_STATUS.md").write_text(f"# Epistemic status\n\nRaw executions and tensors are **MEASURED**. Frozen summaries, distances, identification rates, verification proxies, and stability classes are **DERIVED**. Same-family discrimination is **NOT_ESTABLISHED** because the frozen panel has one challenge per family. Current-device CUDA, CPU/GPU isolation, quantization, alternative attention backends, PQC, and authentication are **NOT_TESTED**. Verdict: **{verdict}**.\n")
report=f"""# Cross-Environment Fingerprint Validation V0

## Verdict

**{verdict}** under the tested comparison: historical CUDA bfloat16/eager enrollment versus CPU float32, bfloat16, and float16/eager probes for 12 frozen challenges and three seeds.

Baseline rotating-seed top-1 is {primary['baseline_identification']['top1']:.2%}; pooled cross-environment top-1 is {cross_top:.2%} and top-5 is {cross_top5:.2%}. The research verification proxy gives FAR-like {far:.2%}, FRR-like {frr:.2%}, ROC-AUC {auc:.4f}, and EER {eer:.2%}.

Same-family identification is **NOT_ESTABLISHED**: one frozen challenge per family makes the candidate set non-discriminative. This is not reported as a synthetic 100% result.

## Boundaries

The baseline environment is historical hash-verified evidence, because CUDA was unavailable on the current host. Therefore the observed change combines device and dtype for float32/float16 comparisons; CPU bfloat16 versus historical CUDA bfloat16 isolates device more closely but not temporal/library drift. Quantization and alternate attention backends were unavailable. No authentication, PQC, security-product, entropy, ownership, anti-replay, or control claim is made.
"""
(HERE/"REPORT.md").write_text(report)
dump("MANIFEST.json",{"gate":"CROSS-ENVIRONMENT FINGERPRINT VALIDATION V0","date":"2026-08-19","base_model":"Qwen/Qwen2.5-0.5B-Instruct","environments_available_now":4,"environments_tested":4,"new_generations":108,"challenges":12,"seeds":3,"feature_registry":"reports/tensor-runtime-fingerprint-v0/FEATURE_REGISTRY.json","feature_registry_sha256":"f9213d0c2181d87775d9628ca4124de119f7ed98d96ace66cf449b4a5373ae71","raw_capture_hash_verification":{"passed":144,"total":144},"source_artifacts_modified":False,"baseline_identification":primary["baseline_identification"]["top1"],"cross_environment_identification":cross_top,"same_family_cross_environment":"NOT_ESTABLISHED","far_like_proxy":far,"frr_like_proxy":frr,"roc_auc":auc,"eer":eer,"final_verdict":verdict,"pqc":"NOT_TESTED","authentication":"NOT_TESTED","security_product_claim":"NO"})
print(json.dumps({"verdict":verdict,"baseline_top1":primary["baseline_identification"]["top1"],"cross_top1":cross_top,"far":far,"frr":frr,"auc":auc,"eer":eer},indent=2))
