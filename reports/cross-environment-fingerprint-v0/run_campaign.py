#!/usr/bin/env python3
"""Bounded 108-generation CPU campaign using the frozen Lane B capture mechanism."""
import gc,hashlib,json,time
from pathlib import Path
import numpy as np,torch
from transformers import AutoModelForCausalLM,AutoTokenizer
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];RAW=HERE/"raw_captures";INDEX=HERE/"new_run_index.jsonl"
MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct";REVISION="7ae557604adf67be50417f59c2c2f167def9a775";SEEDS=[2101,2102,2103];LAYERS=24
ENVS=[("cpu_float32_eager",torch.float32),("cpu_bfloat16_eager",torch.bfloat16),("cpu_float16_eager",torch.float16)]
GEN={"do_sample":True,"temperature":1.0,"top_p":1.0,"top_k":0,"repetition_penalty":1.0}
class Capture:
 def __init__(self):self.attn={};self.q={}
 def install(self,model):
  hs=[]
  for i in range(LAYERS):
   hs.append(model.model.layers[i].self_attn.register_forward_hook(lambda m,inp,out,i=i:self.attn.__setitem__(i,(out[0] if isinstance(out,tuple) else out).detach())))
   hs.append(model.model.layers[i].self_attn.q_proj.register_forward_hook(lambda m,inp,out,i=i:self.q.__setitem__(i,out.detach())))
  return hs
def digest(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def already():
 if not INDEX.exists():return set()
 return {json.loads(x)["run_id"] for x in INDEX.read_text().splitlines() if x.strip() and json.loads(x).get("status")=="SUCCESS"}
def main():
 panel_ids=[x["prompt_id"] for x in json.loads((HERE/"CHALLENGE_PANEL.json").read_text())["challenges"]];panel={}
 for line in (ROOT/"star_command_panel_v2_authoring/FREEZE/PANEL_V2_FROZEN_CANONICAL.jsonl").read_text().splitlines():
  x=json.loads(line);panel[x["candidate_prompt_id"]]=x
 assert len(panel_ids)==12 and all(x in panel for x in panel_ids);RAW.mkdir(exist_ok=True);done=already();tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=REVISION,local_files_only=True)
 if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
 planned=108;completed=len(done);print(f"FROZEN_PLAN new_generations={planned} completed={completed}",flush=True)
 for env_id,dtype in ENVS:
  print(f"LOAD {env_id}",flush=True);model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=REVISION,local_files_only=True,torch_dtype=dtype,attn_implementation="eager",low_cpu_mem_usage=True).eval()
  for seed in SEEDS:
   for pid in panel_ids:
    run_id=f"CEFV0_{env_id}_{pid}_{seed}"
    if run_id in done:continue
    rec=panel[pid];prompt=tokenizer.apply_chat_template([{"role":"user","content":rec["prompt_text"]}],tokenize=False,add_generation_prompt=True);inputs=tokenizer(prompt,return_tensors="pt");prompt_len=int(inputs["input_ids"].shape[1]);torch.manual_seed(seed);t0=time.time()
    entry={"run_id":run_id,"environment_id":env_id,"model_id":MODEL_ID,"checkpoint_revision":REVISION,"prompt_id":pid,"prompt_family":rec["family"],"prompt_hash":rec["prompt_hash"],"seed":seed,"dtype":str(dtype).removeprefix("torch."),"device":"cpu","attention_backend":"eager","generation_config":GEN|{"max_new_tokens":rec["max_new_tokens"]}}
    try:
     with torch.no_grad():gen=model.generate(**inputs,max_new_tokens=rec["max_new_tokens"],**GEN,return_dict_in_generate=True,pad_token_id=tokenizer.pad_token_id)
     ids=gen.sequences;n_steps=int(ids.shape[1]-prompt_len);cap=Capture();handles=cap.install(model)
     try:
      with torch.no_grad():out=model(input_ids=ids,use_cache=True,output_attentions=True,output_hidden_states=True)
     finally:
      for h in handles:h.remove()
     seq_len=int(ids.shape[1]);arrays={"token_ids":ids[0].numpy().astype(np.int32),"logits_region_fp16":out.logits[0,prompt_len-1:seq_len].half().numpy(),"k_all_layers_fp32":np.stack([out.past_key_values[l][0][0].float().numpy() for l in range(LAYERS)]),"v_all_layers_fp32":np.stack([out.past_key_values[l][1][0].float().numpy() for l in range(LAYERS)]),"q_all_layers_fp16":np.stack([cap.q[l][0].view(seq_len,14,64).transpose(0,1).half().numpy() for l in range(LAYERS)]),"attentions_all_layers_fp16":np.stack([out.attentions[l][0].half().numpy() for l in range(LAYERS)]),"hidden_states_all_layers_fp16":np.stack([x[0].half().numpy() for x in out.hidden_states]),"attn_output_all_layers_fp16":np.stack([cap.attn[l][0].half().numpy() for l in range(LAYERS)])}
     target=RAW/f"{run_id}.npz";np.savez_compressed(target,**arrays);finite=all(np.isfinite(x).all() for x in arrays.values());entry|={"status":"SUCCESS" if finite and n_steps>=1 else "TECHNICALLY_INVALID","prompt_len":prompt_len,"seq_len":seq_len,"n_steps":n_steps,"runtime_seconds":time.time()-t0,"artifact_path":str(target.relative_to(ROOT)),"artifact_sha256":digest(target),"tensor_shapes":{k:list(v.shape) for k,v in arrays.items()},"finite":finite}
     del out,gen,ids,arrays
    except Exception as exc:entry|={"status":"TECHNICAL_FAILURE","runtime_seconds":time.time()-t0,"error":f"{type(exc).__name__}: {exc}"}
    with INDEX.open("a") as f:f.write(json.dumps(entry)+"\n")
    completed+=entry["status"]=="SUCCESS";print(f"RUN {completed}/{planned} {run_id} {entry['status']} steps={entry.get('n_steps')} seconds={entry['runtime_seconds']:.1f}",flush=True);gc.collect()
    if entry["status"]!="SUCCESS":raise SystemExit(f"campaign stopped on {run_id}: {entry}")
  del model;gc.collect()
 print(f"COMPLETE {completed}/{planned}")
if __name__=="__main__":main()
