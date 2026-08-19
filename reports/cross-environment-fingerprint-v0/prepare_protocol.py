#!/usr/bin/env python3
import hashlib,json,platform,sys
from pathlib import Path
import pandas as pd,torch,transformers
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;SRC=ROOT/"reports/bcc-atlas-expansion-v1/ATLAS_DEEP";CACHE=Path("/home/jeanbatuli/.cache/huggingface/hub")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
models=[]
for repo in sorted(CACHE.glob("models--*")):
 snaps=list((repo/"snapshots").glob("*")) if (repo/"snapshots").exists() else []
 for snap in snaps:
  names={x.name for x in snap.iterdir()};has_weight=any(n.endswith((".safetensors",".bin",".pt")) for n in names) or "model.safetensors.index.json" in names;has_tok=bool({"tokenizer.json","tokenizer.model","vocab.json"}&names);status="AVAILABLE_NOW" if has_weight and "config.json" in names and has_tok else "NOT_AVAILABLE"
  models.append({"repository":repo.name.removeprefix("models--").replace("--","/"),"revision":snap.name,"status":status,"has_weights":has_weight,"has_tokenizer":has_tok,"has_config":"config.json" in names})
snap=CACHE/"models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
envs=[
 {"environment_id":"baseline_cuda_bfloat16_eager","status":"AVAILABLE_NOW","availability_scope":"existing hash-verified Lane B evidence; device not currently executable","device":"cuda","dtype":"bfloat16","attention_backend":"eager","new_execution":False},
 {"environment_id":"cpu_float32_eager","status":"AVAILABLE_NOW","availability_scope":"integrity forward passed locally","device":"cpu","dtype":"float32","attention_backend":"eager","new_execution":True},
 {"environment_id":"cpu_bfloat16_eager","status":"AVAILABLE_NOW","availability_scope":"integrity forward passed locally","device":"cpu","dtype":"bfloat16","attention_backend":"eager","new_execution":True},
 {"environment_id":"cpu_float16_eager","status":"AVAILABLE_NOW","availability_scope":"integrity forward passed locally","device":"cpu","dtype":"float16","attention_backend":"eager","new_execution":True},
 {"environment_id":"cuda_current_session","status":"NOT_AVAILABLE","reason":"torch.cuda.is_available() false; NvMap/NVML initialization unavailable"},
 {"environment_id":"sdpa_full_attention_capture","status":"NOT_AVAILABLE","reason":"does not provide a distinct comparable full per-head attention capture; eager fallback would erase backend difference"},
 {"environment_id":"quantized_qwen_0.5b","status":"NOT_AVAILABLE","reason":"no complete local quantized checkpoint found"}
]
inventory={"date":"2026-08-19","base_model":"Qwen/Qwen2.5-0.5B-Instruct","checkpoint_revision":"7ae557604adf67be50417f59c2c2f167def9a775","checkpoint_sha256":sha(snap/"model.safetensors"),"tokenizer_sha256":sha(snap/"tokenizer.json"),"tokenizer_config_sha256":sha(snap/"tokenizer_config.json"),"model_config_sha256":sha(snap/"config.json"),"system":{"platform":platform.platform(),"machine":platform.machine(),"cpu":"ARM Cortex-A78AE, 6 cores","hardware_identifier":"Linux Tegra host; CPU execution","python":sys.version.split()[0],"torch":str(torch.__version__),"transformers":transformers.__version__,"cuda_build":torch.version.cuda,"cuda_available":torch.cuda.is_available()},"candidate_environments":envs,"cached_checkpoint_inventory":models,"classification_counts":{k:int(v) for k,v in pd.Series([x["status"] for x in envs]).value_counts().items()},"large_downloads_performed":False}
(OUT/"ENVIRONMENT_INVENTORY.json").write_text(json.dumps(inventory,indent=2)+"\n")
idx=pd.read_csv(SRC/"RUN_INDEX.csv");stab=pd.read_csv(ROOT/"reports/tensor-runtime-fingerprint-v0/SEED_STABILITY.csv");agg=idx.groupby(["prompt_family","prompt_id"],as_index=False).agg(mean_steps=("n_steps","mean"),min_steps=("n_steps","min"),max_steps=("n_steps","max")).merge(stab[["prompt_id","stability_class","intra_prompt_seed_variability"]],on="prompt_id")
chosen=[]
for family,g in agg.groupby("prompt_family"):
 med=float(g.mean_steps.median());g=g.assign(stability_preference=(g.stability_class!="MODERATE").astype(int),length_deviation=(g.mean_steps-med).abs().round(6)).sort_values(["stability_preference","length_deviation","max_steps","prompt_id"]);r=g.iloc[0]
 chosen.append({"prompt_id":r.prompt_id,"prompt_family":family,"selection_basis":"MODERATE preferred, then closest to family median length, then lower maximum length","v0_stability_class":r.stability_class,"v0_seed_variability":float(r.intra_prompt_seed_variability),"baseline_mean_steps":float(r.mean_steps),"baseline_min_steps":int(r.min_steps),"baseline_max_steps":int(r.max_steps)})
panel={"status":"FROZEN_PRE_EXECUTION","selection_uses_fingerprint_separation":False,"n_challenges":12,"n_families":12,"challenges":chosen}
(OUT/"CHALLENGE_PANEL.json").write_text(json.dumps(panel,indent=2)+"\n")
plan={"status":"FROZEN_PRE_EXECUTION","baseline_reused_runs":36,"new_environments":[x["environment_id"] for x in envs if x.get("new_execution")],"challenges":12,"seeds":[2101,2102,2103],"planned_new_generations":108,"authorization_stop_threshold":216,"threshold_exceeded":False,"generation_config":{"do_sample":True,"temperature":1.0,"top_p":1.0,"top_k":0,"repetition_penalty":1.0,"max_new_tokens":"frozen per prompt from Panel V2"},"capture":"same H/Q/K/V/Attention/Attention Output/Logits teacher-forced Lane B mechanism","feature_registry":"reports/tensor-runtime-fingerprint-v0/FEATURE_REGISTRY.json"}
(OUT/"EXECUTION_PLAN.json").write_text(json.dumps(plan,indent=2)+"\n")
print(json.dumps({"selected":[x["prompt_id"] for x in chosen],"planned_new_generations":108,"environments":4},indent=2))
