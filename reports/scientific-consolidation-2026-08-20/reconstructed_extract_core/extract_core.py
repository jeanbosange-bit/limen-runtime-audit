"""
CORE-panel-only extraction: intermediate logit-lens readout (section 7),
compact random-projection representation (section 12), and orthogonal-invariance
preparation + geometry-destroying control (section 13).

Model loading here is READ-ONLY weight introspection + a real forward pass on the
ALREADY-CAPTURED prompt/generated token sequence to identify the final-norm module
(find_final_norm_module) and to obtain lm_head/final-norm weights. No new sampling,
no new generation, nothing stored as a scientific result beyond derived features of
the existing captured hidden states.
"""
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/tmp/claude-1000/-home-jeanbatuli-V20-probabilistic-dynamics/5d793c3e-5e08-4a66-9200-8acfbc5066c9/scratchpad")
from find_final_norm import find_final_norm_module  # noqa: E402
import extract_features as ef  # noqa: E402

OUT = ef.OUT
SRC = ef.SRC
SCRATCH = Path("/tmp/claude-1000/-home-jeanbatuli-V20-probabilistic-dynamics/5d793c3e-5e08-4a66-9200-8acfbc5066c9/scratchpad")
LOG_PATH = OUT / "extraction_run.log"

PROJ_DIM = 128
ORTHOGONAL_SUBSET = [
    "Qwen/Qwen2.5-0.5B-Instruct", "meta-llama/Llama-3.2-1B", "gpt2", "EleutherAI/pythia-160m",
    "facebook/opt-125m", "microsoft/phi-1_5",
]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def deterministic_seed(model_id):
    return int(hashlib.sha256(model_id.encode()).hexdigest()[:8], 16)


def load_model(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    last_exc = None
    for dt in (torch.bfloat16, torch.float16):
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dt).to("cuda").eval()
            return model, tok, dt
        except Exception as e:
            last_exc = e
    raise RuntimeError(f"load failed for {model_id}: {last_exc}")


def process_core_model(model_id):
    log(f"=== CORE {model_id}: loading model ===")
    model, tok, dt = load_model(model_id)
    norm_name, norm_module = find_final_norm_module(model, tok)
    lm_head = model.get_output_embeddings()
    d_model = lm_head.in_features
    log(f"{model_id}: final_norm_module={norm_name} d_model={d_model} dtype={dt}")

    rng = np.random.default_rng(deterministic_seed(model_id))
    proj_matrix = rng.standard_normal((d_model, PROJ_DIM)).astype(np.float32) / np.sqrt(PROJ_DIM)
    proj_hash = hashlib.sha256(proj_matrix.tobytes()).hexdigest()

    ortho_matrix = None
    ortho_hash = None
    if model_id in ORTHOGONAL_SUBSET:
        rng2 = np.random.default_rng(deterministic_seed(model_id) + 1)
        M = rng2.standard_normal((d_model, d_model)).astype(np.float64)
        Qo, _ = np.linalg.qr(M)
        ortho_matrix = Qo.astype(np.float32)
        ortho_hash = hashlib.sha256(ortho_matrix.tobytes()).hexdigest()

    proj_meta = {
        "model_id": model_id, "d_model": d_model, "proj_dim": PROJ_DIM,
        "seed": deterministic_seed(model_id), "proj_matrix_sha256": proj_hash,
        "orthogonal_subset_member": model_id in ORTHOGONAL_SUBSET,
        "orthogonal_matrix_sha256": ortho_hash, "final_norm_module": norm_name,
    }

    runs = ef.runs_by_model.get(model_id, [])
    proj_rows = []
    ortho_rows = []
    readout_rows_intermediate = []
    n_ok = n_fail = 0

    param_dtype = next(model.parameters()).dtype
    with torch.no_grad():
        norm_t = norm_module.to("cuda") if norm_module is not None else None
        for run in runs:
            npz_path = SRC / run["model_deep_path"]
            if not npz_path.exists():
                n_fail += 1
                continue
            try:
                npz = np.load(npz_path)
                H = npz["hidden_states_all_layers_fp16"].astype(np.float32)
                token_ids = npz["token_ids"]
                # Two capture-schema naming conventions exist in this corpus: this campaign's
                # own runs use "generated_logits_fp16"; Qwen2.5-0.5B-Instruct's REUSED_CANONICAL_
                # LANE_B_READ_ONLY runs (from a separate, earlier gate) use "logits_region_fp16".
                # Both hold the same semantic content (per-generated-step logits over the vocab).
                logits_key = "generated_logits_fp16" if "generated_logits_fp16" in npz.files else (
                    "logits_region_fp16" if "logits_region_fp16" in npz.files else None)
                logits = npz[logits_key].astype(np.float32) if logits_key else None
            except Exception as e:
                n_fail += 1
                log(f"FAIL {run['run_id']}: {e}")
                continue
            n_layers = H.shape[0] - 1
            seq_len = H.shape[1]
            n_steps = logits.shape[0] if logits is not None else 0
            prompt_len = seq_len - n_steps

            # ---- dense random projection + orthogonal transform: vectorized across the whole run ----
            positions = np.arange(prompt_len, min(prompt_len + n_steps, seq_len))
            H_run = H[:, positions, :]  # (n_layers+1, n_steps_eff, d_model)
            proj_run = np.einsum("lsd,dp->lsp", H_run, proj_matrix)  # (n_layers+1, n_steps_eff, PROJ_DIM)
            for li, layer in enumerate(range(n_layers + 1)):
                for si, step in enumerate(range(len(positions))):
                    proj_rows.append({
                        "model_id": model_id, "run_id": run["run_id"], "prompt_id": run["prompt_id"],
                        "prompt_family": run["family"], "seed": run["seed"], "token_step": step,
                        "layer_index": layer, "num_layers": n_layers,
                        **{f"proj_{i}": float(proj_run[li, si, i]) for i in range(PROJ_DIM)},
                    })
            if ortho_matrix is not None:
                H_rot = np.einsum("lsd,de->lse", H_run, ortho_matrix)
                proj_rot_run = np.einsum("lse,ep->lsp", H_rot, proj_matrix)
                for li, layer in enumerate(range(n_layers + 1)):
                    for si, step in enumerate(range(len(positions))):
                        ortho_rows.append({
                            "model_id": model_id, "run_id": run["run_id"], "prompt_id": run["prompt_id"],
                            "prompt_family": run["family"], "seed": run["seed"], "token_step": step,
                            "layer_index": layer, "num_layers": n_layers, "representation": "ORTHOGONAL_ROTATED",
                            **{f"proj_{i}": float(proj_rot_run[li, si, i]) for i in range(PROJ_DIM)},
                        })

            # ---- intermediate logit-lens readout (section 7) -- batched per run on GPU ----
            # Stride disclosed in 02_FEATURE_SCHEMA.md: every 3rd step x every 4th layer + always
            # the true final layer, since a dense vocab-size matmul at every (t,l) across 12 models
            # is not tractable in this session's wall-clock budget; geometry/proj features stay dense.
            readout_layers = [l for l in range(n_layers + 1) if l % 4 == 0 or l == n_layers]
            readout_steps = [s for s in range(len(positions)) if s % 3 == 0]
            if readout_steps and readout_layers and logits is not None:
                target_ids = []
                batch_h = []
                batch_meta = []
                for step in readout_steps:
                    pos = positions[step]
                    if pos + 1 >= len(token_ids):
                        continue
                    tid = int(token_ids[pos + 1])
                    for layer in readout_layers:
                        batch_h.append(H[layer, pos])
                        batch_meta.append((step, layer))
                        target_ids.append(tid)
                if batch_h:
                    h_batch = torch.from_numpy(np.stack(batch_h)).to("cuda", dtype=param_dtype)
                    normed = norm_t(h_batch) if norm_t is not None else h_batch
                    logit_batch = lm_head(normed).float()  # (n, vocab)
                    tid_t = torch.tensor(target_ids, device="cuda")
                    target_logit = logit_batch.gather(1, tid_t.unsqueeze(1)).squeeze(1)
                    top2 = torch.topk(logit_batch, 2, dim=1)
                    top1_id, top2_id = top2.indices[:, 0], top2.indices[:, 1]
                    top1_logit, top2_logit = top2.values[:, 0], top2.values[:, 1]
                    log_probs = torch.log_softmax(logit_batch, dim=1)
                    entropy = -(log_probs.exp() * log_probs).sum(dim=1)
                    target_logprob = log_probs.gather(1, tid_t.unsqueeze(1)).squeeze(1)
                    rank = (logit_batch > target_logit.unsqueeze(1)).sum(dim=1)
                    is_top1 = (top1_id == tid_t)
                    probs_top = torch.softmax(logit_batch, dim=1).gather(1, torch.stack([top1_id, top2_id], dim=1))
                    for i, (step, layer) in enumerate(batch_meta):
                        readout_rows_intermediate.append({
                            "model_id": model_id, "run_id": run["run_id"], "prompt_id": run["prompt_id"],
                            "prompt_family": run["family"], "seed": run["seed"], "token_step": step,
                            "layer_index": layer, "num_layers": n_layers,
                            "readout_type": "FINAL_NATIVE" if layer == n_layers else "INTERMEDIATE_PROJECTED",
                            "top1_token_id": int(top1_id[i]), "top1_logit": float(top1_logit[i]),
                            "top1_prob": float(probs_top[i, 0]),
                            "top2_token_id": int(top2_id[i]), "top2_logit": float(top2_logit[i]),
                            "top2_prob": float(probs_top[i, 1]),
                            "target_logit": float(target_logit[i]), "target_probability": float(target_logprob[i].exp()),
                            "target_rank": int(rank[i]), "is_target_top1": bool(is_top1[i]),
                            "entropy": float(entropy[i]),
                        })
            npz.close()
            n_ok += 1

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return proj_meta, proj_rows, ortho_rows, readout_rows_intermediate, n_ok, n_fail


STR, F64, I64, BOOL = pa.string(), pa.float64(), pa.int64(), pa.bool_()
PROJ_SCHEMA = pa.schema(
    [("model_id", STR), ("run_id", STR), ("prompt_id", STR), ("prompt_family", STR), ("seed", STR),
     ("token_step", I64), ("layer_index", I64), ("num_layers", I64)]
    + [(f"proj_{i}", F64) for i in range(PROJ_DIM)]
)
ORTHO_SCHEMA = pa.schema(
    [("model_id", STR), ("run_id", STR), ("prompt_id", STR), ("prompt_family", STR), ("seed", STR),
     ("token_step", I64), ("layer_index", I64), ("num_layers", I64), ("representation", STR)]
    + [(f"proj_{i}", F64) for i in range(PROJ_DIM)]
)
READOUT_SCHEMA = pa.schema([
    ("model_id", STR), ("run_id", STR), ("prompt_id", STR), ("prompt_family", STR), ("seed", STR),
    ("token_step", I64), ("layer_index", I64), ("num_layers", I64), ("readout_type", STR),
    ("top1_token_id", I64), ("top1_logit", F64), ("top1_prob", F64),
    ("top2_token_id", I64), ("top2_logit", F64), ("top2_prob", F64),
    ("target_logit", F64), ("target_probability", F64), ("target_rank", I64), ("is_target_top1", BOOL),
    ("entropy", F64),
])


if __name__ == "__main__":
    core_extended = json.loads((OUT / "_core_extended.json").read_text())
    CORE_MODELS = core_extended["core"]

    proj_writer = None
    ortho_writer = None
    readout_writer = None
    all_proj_meta = []
    total_n_ok = total_n_fail = 0

    completed_path = SCRATCH / "_core_completed_models.json"
    completed = json.loads(completed_path.read_text()) if completed_path.exists() else []
    model_failures = json.loads((SCRATCH / "_core_model_failures.json").read_text()) if (SCRATCH / "_core_model_failures.json").exists() else {}

    for model_id in CORE_MODELS:
        if model_id in completed:
            log(f"SKIP {model_id}: already completed in a prior run of this script")
            continue
        try:
            proj_meta, proj_rows, ortho_rows, readout_rows, n_ok, n_fail = process_core_model(model_id)
        except Exception as e:
            log(f"CORE MODEL FAILED (real, disclosed, not retried blindly): {model_id}: {type(e).__name__}: {e}")
            model_failures[model_id] = f"{type(e).__name__}: {e}"
            (SCRATCH / "_core_model_failures.json").write_text(json.dumps(model_failures, indent=2))
            gc.collect()
            torch.cuda.empty_cache()
            continue
        all_proj_meta.append(proj_meta)
        total_n_ok += n_ok
        total_n_fail += n_fail
        completed.append(model_id)
        completed_path.write_text(json.dumps(completed, indent=2))

        if proj_rows:
            t = pa.Table.from_pylist(proj_rows, schema=PROJ_SCHEMA)
            if proj_writer is None:
                proj_writer = pq.ParquetWriter(OUT / "_part_core_proj.parquet", PROJ_SCHEMA)
            proj_writer.write_table(t)
        if ortho_rows:
            t = pa.Table.from_pylist(ortho_rows, schema=ORTHO_SCHEMA)
            if ortho_writer is None:
                ortho_writer = pq.ParquetWriter(OUT / "_part_core_ortho.parquet", ORTHO_SCHEMA)
            ortho_writer.write_table(t)
        if readout_rows:
            t = pa.Table.from_pylist(readout_rows, schema=READOUT_SCHEMA)
            if readout_writer is None:
                readout_writer = pq.ParquetWriter(OUT / "_part_core_readout_intermediate.parquet", READOUT_SCHEMA)
            readout_writer.write_table(t)

        log(f"{model_id}: proj_rows={len(proj_rows)} ortho_rows={len(ortho_rows)} readout_rows={len(readout_rows)} n_ok={n_ok} n_fail={n_fail}")

    for w in (proj_writer, ortho_writer, readout_writer):
        if w:
            w.close()

    json.dump(all_proj_meta, open(OUT / "PROJECTION_METADATA.json", "w"), indent=2)
    json.dump({"n_ok": total_n_ok, "n_fail": total_n_fail}, open(SCRATCH / "_extraction_summary_core.json", "w"), indent=2)
    log(f"DONE core: ok={total_n_ok} fail={total_n_fail}")
