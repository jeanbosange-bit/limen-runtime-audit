"""
BCC Atlas Expansion V1 -- Lane B (ATLAS-DEEP) execution.
48 structural candidates (reports/bcc-atlas-expansion-v1/A/BCC_WIDE/LANE_B_STRUCTURAL_CANDIDATES.json,
frozen, unaltered) x 3 seeds (2101/2102/2103, same as Lane A) = 144 real deep captures.

Protocol continuity with Lane A (mission explicit): do_sample=True, temperature=1.0,
top_p=1.0, top_k=0, repetition_penalty=1.0, same seeds.

Capture mechanism: reuses reports/atlas-kv-compression-v0/execution/kv_capture_lib.py's
proven pattern verbatim (attn_implementation="eager" -- required for real output_attentions;
one sampled generate() call + one teacher-forced full-sequence forward pass with
output_attentions=True/output_hidden_states=True/use_cache=True; AttnOutputCapture hook on
self_attn for attention output), EXTENDED with a new QCapture hook on self_attn.q_proj
(empirically verified available in the Lane B preflight, LANE_B_CAPTURE_SCHEMA.json).

max_new_tokens is per-prompt from the frozen Panel V2 record (NOT V0's flat 48-token cap).
"""
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PANEL_PATH = ROOT / "star_command_panel_v2_authoring" / "FREEZE" / "PANEL_V2_FROZEN_CANONICAL.jsonl"
CANDIDATES_PATH = HERE.parent / "A" / "BCC_WIDE" / "LANE_B_STRUCTURAL_CANDIDATES.json"
RAW_DIR = HERE / "raw_captures"
RUN_INDEX_PATH = HERE / "run_index.jsonl"
LOG_PATH = HERE / "lane_b_run.log"

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SEEDS = [2101, 2102, 2103]
NUM_LAYERS = 24
GEN_KWARGS_FIXED = {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


class AttnOutputCapture:
    """Verbatim from kv_capture_lib.py: hooks self_attn, records post-o_proj output."""
    def __init__(self):
        self.outputs = {}

    def make_hook(self, layer_idx):
        def hook(module, inp, out):
            attn_out = out[0] if isinstance(out, tuple) else out
            self.outputs[layer_idx] = attn_out.detach()
        return hook

    def install(self, model):
        return [model.model.layers[l].self_attn.register_forward_hook(self.make_hook(l)) for l in range(NUM_LAYERS)]


class QCapture:
    """New: hooks self_attn.q_proj (pre-reshape Q, empirically verified in Lane B preflight)."""
    def __init__(self):
        self.outputs = {}

    def make_hook(self, layer_idx):
        def hook(module, inp, out):
            self.outputs[layer_idx] = out.detach()
        return hook

    def install(self, model):
        return [model.model.layers[l].self_attn.q_proj.register_forward_hook(self.make_hook(l)) for l in range(NUM_LAYERS)]


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    last_exc = None
    for dt, dt_name in [(torch.bfloat16, "bfloat16"), (torch.float16, "float16")]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, torch_dtype=dt, device_map="cuda", low_cpu_mem_usage=True,
                trust_remote_code=True, attn_implementation="eager",
            )
            model.eval()
            return model, tokenizer, dt_name
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"model load failed: {last_exc}")


def render_user(tokenizer, text):
    return tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)


def capture_and_persist(model, tokenizer, prompt_text, seed, max_new_tokens, run_id, device="cuda"):
    """Returns (status, meta_dict, tensor_completeness_dict)."""
    tensor_completeness = {k: "NOT_ATTEMPTED" for k in ["H", "Q", "K", "V", "ATTENTION", "ATTENTION_OUTPUT", "LOGITS", "TOKEN_IDS"]}
    prompt_str = render_user(tokenizer, prompt_text)
    inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
    prompt_len = int(inputs["input_ids"].shape[1])

    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    t0 = time.monotonic()
    with torch.no_grad():
        gen_out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=GEN_KWARGS_FIXED["do_sample"], temperature=GEN_KWARGS_FIXED["temperature"],
            top_p=GEN_KWARGS_FIXED["top_p"], top_k=GEN_KWARGS_FIXED["top_k"],
            repetition_penalty=GEN_KWARGS_FIXED["repetition_penalty"],
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_wall_s = time.monotonic() - t0

    full_ids = gen_out.sequences
    n_steps = int(full_ids.shape[1] - prompt_len)
    seq_len = int(full_ids.shape[1])
    response_text = tokenizer.decode(full_ids[0, prompt_len:].cpu(), skip_special_tokens=True)
    tensor_completeness["TOKEN_IDS"] = "COMPLETE"

    if n_steps < 1:
        return "INVALID_OUTPUT", {"error": "n_steps<1, generation produced no tokens"}, tensor_completeness

    attn_cap = AttnOutputCapture()
    q_cap = QCapture()
    handles = attn_cap.install(model) + q_cap.install(model)
    t1 = time.monotonic()
    try:
        with torch.no_grad():
            out_full = model(input_ids=full_ids, use_cache=True, output_attentions=True, output_hidden_states=True)
    finally:
        for h in handles:
            h.remove()
    forward_wall_s = time.monotonic() - t1

    if len(attn_cap.outputs) != NUM_LAYERS:
        tensor_completeness["ATTENTION_OUTPUT"] = f"PARTIAL ({len(attn_cap.outputs)}/{NUM_LAYERS} layers)"
    else:
        tensor_completeness["ATTENTION_OUTPUT"] = "COMPLETE"
    if len(q_cap.outputs) != NUM_LAYERS:
        tensor_completeness["Q"] = f"PARTIAL ({len(q_cap.outputs)}/{NUM_LAYERS} layers)"
    else:
        tensor_completeness["Q"] = "COMPLETE"
    if out_full.attentions is not None and len(out_full.attentions) == NUM_LAYERS:
        tensor_completeness["ATTENTION"] = "COMPLETE"
    else:
        tensor_completeness["ATTENTION"] = "PARTIAL_OR_MISSING"
    if out_full.hidden_states is not None and len(out_full.hidden_states) == NUM_LAYERS + 1:
        tensor_completeness["H"] = "COMPLETE"
    else:
        tensor_completeness["H"] = "PARTIAL_OR_MISSING"
    if out_full.past_key_values is not None:
        tensor_completeness["K"] = "COMPLETE"
        tensor_completeness["V"] = "COMPLETE"
    else:
        tensor_completeness["K"] = "MISSING"
        tensor_completeness["V"] = "MISSING"
    tensor_completeness["LOGITS"] = "COMPLETE" if out_full.logits is not None else "MISSING"

    any_missing = any(v not in ("COMPLETE",) for v in tensor_completeness.values())

    token_ids = full_ids[0].cpu().numpy().astype(np.int32)
    logits_region = out_full.logits[0][prompt_len - 1:seq_len, :].half().cpu().numpy()
    k_layers = np.stack([out_full.past_key_values[l][0][0].float().cpu().numpy() for l in range(NUM_LAYERS)], axis=0)
    v_layers = np.stack([out_full.past_key_values[l][1][0].float().cpu().numpy() for l in range(NUM_LAYERS)], axis=0)
    attn_layers = np.stack([out_full.attentions[l][0].half().cpu().numpy() for l in range(NUM_LAYERS)], axis=0)
    hidden_layers = np.stack([out_full.hidden_states[l][0].half().cpu().numpy() for l in range(len(out_full.hidden_states))], axis=0)
    attn_out_layers = np.stack([attn_cap.outputs[l][0].half().cpu().numpy() for l in range(NUM_LAYERS)], axis=0)
    # Q: pre-reshape (1,seq,896) -> reshape to (14 heads, seq, 64) per layer, matching K/V's head-explicit convention
    q_layers_list = []
    for l in range(NUM_LAYERS):
        qt = q_cap.outputs[l][0]  # (seq_len, 896)
        qt = qt.view(seq_len, 14, 64).transpose(0, 1)  # (14, seq_len, 64)
        q_layers_list.append(qt.half().cpu().numpy())
    q_layers = np.stack(q_layers_list, axis=0)  # (24, 14, seq_len, 64)

    nan_inf = {}
    for name, arr in [("token_ids", token_ids), ("logits", logits_region), ("k", k_layers), ("v", v_layers),
                       ("q", q_layers), ("attentions", attn_layers), ("hidden_states", hidden_layers), ("attn_output", attn_out_layers)]:
        nan_inf[name] = {"nan": int(np.isnan(arr.astype(np.float64)).sum()) if arr.dtype.kind == "f" else 0,
                          "inf": int(np.isinf(arr.astype(np.float64)).sum()) if arr.dtype.kind == "f" else 0}

    arrays = {
        "token_ids": token_ids, "logits_region_fp16": logits_region,
        "k_all_layers_fp32": k_layers, "v_all_layers_fp32": v_layers,
        "q_all_layers_fp16": q_layers,
        "attentions_all_layers_fp16": attn_layers,
        "hidden_states_all_layers_fp16": hidden_layers,
        "attn_output_all_layers_fp16": attn_out_layers,
    }
    npz_path = RAW_DIR / f"{run_id}.npz"
    np.savez_compressed(npz_path, **arrays)
    artifact_hash = hashlib.sha256(
        b"".join(np.ascontiguousarray(a).tobytes() for a in
                 [token_ids, logits_region, k_layers, v_layers, q_layers, attn_layers, hidden_layers, attn_out_layers])
    ).hexdigest()

    total_nan = sum(v["nan"] for v in nan_inf.values())
    total_inf = sum(v["inf"] for v in nan_inf.values())

    status = "SUCCESS"
    if any_missing:
        status = "CAPTURE_FAILURE"
    elif total_nan > 0 or total_inf > 0:
        status = "CAPTURE_FAILURE"

    meta = {
        "prompt_len": prompt_len, "seq_len": seq_len, "n_steps": n_steps,
        "response_text": response_text,
        "npz_path": str(npz_path), "npz_size_bytes": npz_path.stat().st_size,
        "artifact_hash": artifact_hash,
        "shapes": {k: list(v.shape) for k, v in arrays.items()},
        "nan_inf": nan_inf, "total_nan": total_nan, "total_inf": total_inf,
        "gen_wall_s": gen_wall_s, "forward_wall_s": forward_wall_s,
        "tensor_completeness": tensor_completeness,
    }
    return status, meta, tensor_completeness


def early_integrity_check(records):
    """Audit the first 3 completed runs' shapes/NaN-Inf/alignment/storage. Returns (ok, report)."""
    report = {"n_checked": len(records), "checks": []}
    ok = True
    for r in records:
        c = {"run_id": r["run_id"], "status": r["status"]}
        if r["status"] != "SUCCESS":
            c["pass"] = False
            c["reason"] = f"status={r['status']}"
            ok = False
        else:
            shapes = r["meta"]["shapes"]
            seq_len = r["meta"]["seq_len"]
            checks = {
                "H_layers_25": shapes["hidden_states_all_layers_fp16"][0] == 25,
                "Q_layers_24_heads_14": shapes["q_all_layers_fp16"][:2] == [24, 14],
                "K_layers_24_kvheads_2": shapes["k_all_layers_fp32"][:2] == [24, 2],
                "V_layers_24_kvheads_2": shapes["v_all_layers_fp32"][:2] == [24, 2],
                "attn_layers_24_heads_14": shapes["attentions_all_layers_fp16"][:2] == [24, 14],
                "attn_out_layers_24": shapes["attn_output_all_layers_fp16"][0] == 24,
                "seq_len_alignment_H": shapes["hidden_states_all_layers_fp16"][1] == seq_len,
                "seq_len_alignment_Q": shapes["q_all_layers_fp16"][2] == seq_len,
                "seq_len_alignment_K": shapes["k_all_layers_fp32"][2] == seq_len,
                "seq_len_alignment_attn": shapes["attentions_all_layers_fp16"][2] == seq_len and shapes["attentions_all_layers_fp16"][3] == seq_len,
                "zero_nan": r["meta"]["total_nan"] == 0,
                "zero_inf": r["meta"]["total_inf"] == 0,
                "npz_size_reasonable": 100_000 < r["meta"]["npz_size_bytes"] < 500_000_000,
            }
            c["checks"] = checks
            c["pass"] = all(checks.values())
            if not c["pass"]:
                ok = False
        report["checks"].append(c)
    report["overall_pass"] = ok
    return ok, report


def main():
    log("Loading Lane B structural candidates (frozen, read-only)...")
    with open(CANDIDATES_PATH) as f:
        cand_doc = json.load(f)
    candidate_ids = []
    for fam, v in cand_doc["by_family"].items():
        for entry in v["selected"]:
            candidate_ids.append(entry["prompt_id"])
    assert len(candidate_ids) == 48, f"expected 48 candidates, got {len(candidate_ids)}"
    log(f"48 candidates loaded across {len(cand_doc['by_family'])} families")

    with open(PANEL_PATH) as f:
        panel = {json.loads(l)["candidate_prompt_id"]: json.loads(l) for l in f}

    log(f"Loading model {MODEL_ID} (attn_implementation=eager, required for real attention-weight capture)...")
    model, tokenizer, dtype_used = load_model()
    log(f"Model loaded. dtype={dtype_used}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    index_f = open(RUN_INDEX_PATH, "a")

    n_attempted = n_success = n_fail = 0
    completed_records = []
    integrity_checked = False

    for seed in SEEDS:
        log(f"=== SEED {seed} starting ===")
        for pid in candidate_ids:
            rec = panel[pid]
            run_id = f"BAX1DEEP_{pid}_{seed}"
            n_attempted += 1
            entry = {"run_id": run_id, "prompt_id": pid, "prompt_family": rec["family"], "seed": seed,
                      "prompt_hash": rec["prompt_hash"], "max_new_tokens_config": rec["max_new_tokens"]}
            t0 = time.time()
            try:
                status, meta, tensor_completeness = capture_and_persist(
                    model, tokenizer, rec["prompt_text"], seed, rec["max_new_tokens"], run_id
                )
                entry.update({"status": status, "meta": meta, "tensor_completeness": tensor_completeness})
            except torch.cuda.OutOfMemoryError as e:
                status = "OOM"
                entry.update({"status": status, "error": str(e)[:500]})
                torch.cuda.empty_cache()
            except Exception as e:
                status = "TECHNICAL_FAILURE"
                entry.update({"status": status, "error": f"{type(e).__name__}: {str(e)[:500]}"})
            entry["runtime_seconds"] = time.time() - t0

            index_f.write(json.dumps(entry) + "\n")
            index_f.flush()

            if status == "SUCCESS":
                n_success += 1
            else:
                n_fail += 1
                log(f"NON-SUCCESS run_id={run_id} status={status} detail={entry.get('error') or entry.get('meta', {}).get('error')}")

            completed_records.append(entry)
            log(f"[{n_attempted}/144] {run_id} status={status} runtime={entry['runtime_seconds']:.1f}s")

            if not integrity_checked and n_attempted == 3:
                log("=== EARLY INTEGRITY CHECK (first 3 runs) ===")
                ok, report = early_integrity_check(completed_records[-3:])
                with open(HERE / "EARLY_INTEGRITY_CHECK.json", "w") as f:
                    json.dump(report, f, indent=2, default=str)
                if ok:
                    log("EARLY INTEGRITY CHECK: PASS -- resuming automatically.")
                else:
                    log("EARLY INTEGRITY CHECK: FAIL -- STOPPING per protocol. See EARLY_INTEGRITY_CHECK.json.")
                    index_f.close()
                    raise SystemExit("Early integrity check failed -- execution halted before contaminating the full campaign.")
                integrity_checked = True

            gc.collect()
            torch.cuda.empty_cache()

        log(f"=== SEED {seed} complete ===")

    index_f.close()
    log(f"LANE B COMPLETE. attempted={n_attempted} success={n_success} fail={n_fail}")


if __name__ == "__main__":
    main()
