from __future__ import annotations

"""Live, on-Space generation capture for the evidence-architecture demo.

Loads one small, public instruction-tuned model and runs a single greedy
generation, recording three observation channels per generated token:

  - generation  : which token, in which order (always available here).
  - probability : the output distribution's entropy and top-1/top-2 margin.
  - latent      : how far a small, fixed sample of hidden-state layers moved
                   relative to the previous token (coarse, descriptive only).

This module is deliberately narrow. It contains no retry logic, no
actuator-selection policy, and no trained scoring model -- it only produces
a synchronized, hash-chained observation record via ``evidence.EvidenceRun``.
Heavier torch/transformers imports are deferred into the function body so
the rest of the Space (and its test suite) can run without a GPU or without
torch installed at all.
"""

import time
from typing import Any

from .evidence import EvidenceRun

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS_LIMIT = 96
SAMPLE_LAYER_COUNT = 4

_MODEL = None
_TOKENIZER = None


def _load_model():
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
        _MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16
        )
        if torch.cuda.is_available():
            _MODEL = _MODEL.to("cuda")
        _MODEL.eval()
    return _MODEL, _TOKENIZER


def capture_generation(prompt: str, max_new_tokens: int = 48) -> dict[str, Any]:
    """Run one greedy generation and return a sealed, three-channel
    evidence bundle (generation, probability, layer-sampled latent).

    This demonstrates the *shape* of a synchronized multi-signal evidence
    record, not a validated research result. No decision is made from it.
    """
    import torch

    max_new_tokens = max(1, min(int(max_new_tokens), MAX_NEW_TOKENS_LIMIT))
    model, tokenizer = _load_model()

    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    prompt_len = input_ids.shape[1]

    t0 = time.monotonic()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
    latency_s = time.monotonic() - t0

    sequences = out.sequences[0]
    generated_ids = sequences[prompt_len:]
    n_steps = int(generated_ids.shape[0])

    n_layers = model.config.num_hidden_layers
    layer_indices = sorted(
        {int(i) for i in torch.linspace(0, n_layers, steps=min(SAMPLE_LAYER_COUNT, n_layers + 1)).tolist()}
    )

    run = EvidenceRun(capture_profile="full_sampled")
    prev_hidden_by_layer: dict[int, "torch.Tensor"] = {}

    for t in range(n_steps):
        step = run.new_step()
        token_id = int(generated_ids[t].item())
        token_text = tokenizer.decode([token_id])
        step.set_channel(
            "generation",
            "AVAILABLE",
            {"token_index": t, "token_id": token_id, "token_text": token_text},
        )

        scores = out.scores[t][0]
        probs = torch.softmax(scores.float(), dim=-1)
        top5 = torch.topk(probs, k=5)
        entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())
        margin = float((top5.values[0] - top5.values[1]).item())
        step.set_channel(
            "probability",
            "AVAILABLE",
            {
                "entropy_nats": entropy,
                "top1_top2_margin": margin,
                "top5_token_ids": [int(i) for i in top5.indices.tolist()],
                "top5_probs": [float(p) for p in top5.values.tolist()],
            },
        )

        step_hidden = out.hidden_states[t]
        deltas = []
        for layer_idx in layer_indices:
            h = step_hidden[layer_idx][0, -1, :].float()
            prev_h = prev_hidden_by_layer.get(layer_idx)
            if prev_h is not None:
                delta_norm = float(torch.linalg.norm(h - prev_h).item())
                cosine = float(torch.nn.functional.cosine_similarity(h, prev_h, dim=0).item())
            else:
                delta_norm, cosine = None, None
            deltas.append(
                {"layer_index": layer_idx, "delta_norm": delta_norm, "cosine_to_previous": cosine}
            )
            prev_hidden_by_layer[layer_idx] = h

        step.set_channel(
            "latent",
            "AVAILABLE",
            {"sampled_layers": layer_indices, "layer_deltas": deltas},
        )

        run.append(step)

    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    bundle = run.to_dict()
    bundle["run_metadata"] = {
        "model": MODEL_ID,
        "prompt": prompt,
        "response_text": response_text,
        "n_generated_tokens": n_steps,
        "latency_s": latency_s,
        "decoding": "greedy",
        "sampled_layer_indices": layer_indices,
        "note": (
            "Layer-sampled latent deltas are a coarse descriptive signal "
            "(norm/cosine of hidden-state movement between consecutive "
            "tokens at a few fixed layers). They are not a scoring model "
            "and imply no functional interpretation."
        ),
    }
    return bundle
