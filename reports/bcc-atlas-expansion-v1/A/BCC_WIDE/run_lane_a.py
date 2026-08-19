"""
BCC Atlas Expansion V1 -- Lane A (BCC-WIDE) execution.
Frozen per reports/bcc-atlas-expansion-v1/{CAMPAIGN_PLAN.md,SEED_REGISTRY.json,LANE_A_CAPTURE_SCHEMA.json}.

243 frozen Panel V2 prompts x 3 new seeds (2101/2102/2103) = 729 real generations.
Lightweight capture only -- no K/V/attention/hidden-state tensors persisted.

Generation-config note (frozen HERE, before any run, disclosed in the execution report):
BCC V2's own protocol (star_command_bcc_v2/PROTOCOL/BCC_V2_PROTOCOL.json) used
do_sample=False and never used a seed (seed=None in all 243 real BCC V2 rows for this
model) -- greedy decoding would make Lane A's 3 seeds produce IDENTICAL outputs, defeating
the seed-independence/behavioral-variation objective this campaign was designed for
(SEED_REGISTRY.json, CAMPAIGN_PLAN.md). Lane A therefore uses the SAME numeric
hyperparameters as BCC V2 (temperature=1.0, top_p=1.0, top_k=None, repetition_penalty=1.0)
but with do_sample=True, so each seed genuinely diverges. This is a necessary, disclosed
adaptation, not an unauthorized deviation -- BCC V2 itself has no seed convention to be
consistent with.
"""
import gc
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]  # BCC_WIDE -> A -> bcc-atlas-expansion-v1 -> reports -> V20_probabilistic_dynamics
PANEL_PATH = ROOT / "star_command_panel_v2_authoring" / "FREEZE" / "PANEL_V2_FROZEN_CANONICAL.jsonl"
OUT_JSONL = HERE / "lane_a_generations.jsonl"
LOG_PATH = HERE / "lane_a_run.log"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SEEDS = [2101, 2102, 2103]

GEN_KWARGS_FIXED = {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0}
# top_k=0 is HF transformers' documented way to fully disable top-k filtering (BCC V2 protocol
# states top_k: null i.e. no restriction). NOTE: the model's own generation_config.json default
# (top_k=20, top_p=0.8, temperature=0.7, repetition_penalty=1.1) is explicitly overridden here to
# match BCC V2's stated hyperparameters -- confirmed necessary by a smoke test that showed omitting
# top_k silently applies the model's default top_k=20 instead of the intended unrestricted sampling.


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def entropy_from_logits(logit_row, eps=1e-9):
    p = torch.softmax(logit_row.float(), dim=-1)
    return float(-(p * torch.log(p + eps)).sum())


def top2_gap_from_logits(logit_row):
    p = torch.softmax(logit_row.float(), dim=-1)
    top2 = torch.topk(p, 2)
    return float(top2.values[0] - top2.values[1]), float(top2.values[0])


def load_prompts():
    prompts = []
    with open(PANEL_PATH) as f:
        for line in f:
            prompts.append(json.loads(line))
    assert len(prompts) == 243, f"expected 243 frozen prompts, got {len(prompts)}"
    return prompts


def already_done_run_ids():
    done = set()
    if OUT_JSONL.exists():
        with open(OUT_JSONL) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add(r["run_id"])
                except Exception:
                    pass
    return done


def main():
    log("Loading Panel V2 frozen canonical (read-only)...")
    prompts = load_prompts()
    log(f"{len(prompts)} prompts loaded, {len(set(p['family'] for p in prompts))} families")

    log(f"Loading tokenizer/model {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype)
    except Exception:
        dtype = torch.float16
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    has_chat_template = tok.chat_template is not None
    log(f"Model loaded. device={device} dtype={dtype} attn_impl={model.config._attn_implementation} chat_template={'YES' if has_chat_template else 'NO'}")

    done_ids = already_done_run_ids()
    if done_ids:
        log(f"Resuming: {len(done_ids)} run_ids already present in {OUT_JSONL.name}, will skip them.")

    n_attempted = n_success = n_fail = 0
    out_f = open(OUT_JSONL, "a")

    for seed in SEEDS:
        log(f"=== SEED {seed} starting ===")
        for i, rec in enumerate(prompts):
            run_id = f"BAX1_{rec['candidate_prompt_id']}_{seed}"
            if run_id in done_ids:
                continue
            n_attempted += 1
            prompt_text = rec["prompt_text"]
            family = rec["family"]
            max_new_tokens = rec["max_new_tokens"]
            prompt_hash = rec["prompt_hash"]
            candidate_prompt_id = rec["candidate_prompt_id"]

            status = "OTHER"
            entry = {
                "run_id": run_id, "prompt_id": candidate_prompt_id, "prompt_family": family,
                "seed": seed, "prompt_hash": prompt_hash, "max_new_tokens_config": max_new_tokens,
                "status": status,
            }
            start_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            t0 = time.time()
            try:
                torch.manual_seed(seed)
                if device == "cuda":
                    torch.cuda.manual_seed_all(seed)

                if has_chat_template:
                    wrapped = tok.apply_chat_template([{"role": "user", "content": prompt_text}], tokenize=False, add_generation_prompt=True)
                    inputs = tok(wrapped, return_tensors="pt", add_special_tokens=False).to(device)
                    prompt_wrapping = "CHAT_TEMPLATE"
                else:
                    inputs = tok(prompt_text, return_tensors="pt").to(device)
                    prompt_wrapping = "RAW"

                prompt_token_count = inputs["input_ids"].shape[1]

                with torch.no_grad():
                    gen_out = model.generate(
                        **inputs, max_new_tokens=max_new_tokens,
                        do_sample=GEN_KWARGS_FIXED["do_sample"], temperature=GEN_KWARGS_FIXED["temperature"],
                        top_p=GEN_KWARGS_FIXED["top_p"], top_k=GEN_KWARGS_FIXED["top_k"],
                        repetition_penalty=GEN_KWARGS_FIXED["repetition_penalty"],
                        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                        output_scores=True, return_dict_in_generate=True,
                    )
                runtime_s = time.time() - t0

                full_ids = gen_out.sequences[0]
                generated_ids = full_ids[prompt_token_count:]
                generated_token_count = int(generated_ids.shape[0])
                scores = gen_out.scores  # tuple of (1, vocab) logits, one per generated step

                if generated_token_count != len(scores):
                    status = "INVALID_OUTPUT"
                    entry["invalid_reason"] = f"generated_token_count({generated_token_count}) != len(scores)({len(scores)})"
                else:
                    entropies, top1_probs, top2_gaps = [], [], []
                    for step_idx, logit_row in enumerate(scores):
                        row = logit_row[0]
                        ent = entropy_from_logits(row)
                        gap, top1p = top2_gap_from_logits(row)
                        entropies.append(ent)
                        top1_probs.append(top1p)
                        top2_gaps.append(gap)

                    try:
                        generated_text = tok.decode(generated_ids, skip_special_tokens=True)
                    except Exception as dec_e:
                        generated_text = None

                    if generated_text is None:
                        status = "INVALID_OUTPUT"
                        entry["invalid_reason"] = "decode_failed"
                    else:
                        eos_hit = bool(tok.eos_token_id is not None and generated_ids.numel() > 0 and int(generated_ids[-1]) == tok.eos_token_id)
                        completion_reason = "eos_token" if (eos_hit and generated_token_count < max_new_tokens) else ("max_new_tokens" if generated_token_count >= max_new_tokens else ("eos_token" if eos_hit else "other"))
                        peak_mem_mb = float(torch.cuda.max_memory_allocated() / 1e6) if device == "cuda" else None

                        entry.update({
                            "status": "SUCCESS",
                            "token_ids_prompt": inputs["input_ids"][0].tolist(),
                            "token_ids_generated": generated_ids.tolist(),
                            "generated_text": generated_text,
                            "per_step_probability_summary": {
                                "entropy": entropies, "top1_probability": top1_probs, "top2_gap": top2_gaps,
                            },
                            "token_count_prompt": prompt_token_count,
                            "token_count_generated": generated_token_count,
                            "timestamps": {"start_iso": start_iso, "end_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                            "runtime_seconds": runtime_s,
                            "peak_memory_mb": peak_mem_mb,
                            "completion_reason": completion_reason,
                            "model_config": {
                                "model_id": MODEL_ID, "torch_dtype": str(dtype), "attn_implementation": model.config._attn_implementation,
                                "max_new_tokens_config": max_new_tokens, "prompt_wrapping": prompt_wrapping, **GEN_KWARGS_FIXED,
                            },
                            "hashes": {"output_text_sha256": hashlib.sha256(generated_text.encode()).hexdigest()},
                        })
                        status = "SUCCESS"
                        if device == "cuda":
                            torch.cuda.reset_peak_memory_stats()

            except torch.cuda.OutOfMemoryError as e:
                status = "OOM"
                entry["error"] = str(e)[:500]
                if device == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                status = "TECHNICAL_FAILURE"
                entry["error"] = f"{type(e).__name__}: {str(e)[:500]}"
            finally:
                entry["status"] = status
                if "runtime_seconds" not in entry:
                    entry["runtime_seconds"] = time.time() - t0

            out_f.write(json.dumps(entry) + "\n")
            out_f.flush()

            if status == "SUCCESS":
                n_success += 1
            else:
                n_fail += 1
                log(f"NON-SUCCESS run_id={run_id} status={status} detail={entry.get('error') or entry.get('invalid_reason')}")

            if n_attempted % 25 == 0:
                log(f"progress: attempted={n_attempted} success={n_success} fail={n_fail} (seed={seed}, prompt {i+1}/243)")

            del gen_out
            gc.collect()

        log(f"=== SEED {seed} complete: checkpoint flushed to {OUT_JSONL.name} ===")

    out_f.close()
    log(f"LANE A COMPLETE. attempted={n_attempted} success={n_success} fail={n_fail}")


if __name__ == "__main__":
    main()
