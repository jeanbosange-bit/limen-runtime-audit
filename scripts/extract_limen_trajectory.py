#!/usr/bin/env python3
"""Extract an autoregressive LLM trajectory compatible with LIMEN Runtime Audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
DEFAULT_PROMPT = "Explain in two short sentences why the sky appears blue."


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def choose_dtype(device: torch.device, requested: str) -> torch.dtype:
    if requested == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return mapping[requested]


def validate_arrays(hidden_states: np.ndarray, logits: np.ndarray | None) -> None:
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [tokens, layers, hidden_dim]")
    if hidden_states.shape[0] < 1:
        raise ValueError("at least one generated token is required")
    if hidden_states.shape[1] < 3:
        raise ValueError("at least three transformer layers are required")
    if not np.isfinite(hidden_states).all():
        raise ValueError("hidden_states contains NaN or infinity")
    if logits is not None:
        if logits.ndim != 2:
            raise ValueError("logits must have shape [tokens, vocabulary]")
        if logits.shape[0] != hidden_states.shape[0]:
            raise ValueError("hidden_states and logits token counts do not match")
        if logits.shape[1] < 2:
            raise ValueError("logits must contain at least two vocabulary entries")
        if not np.isfinite(logits).all():
            raise ValueError("logits contains NaN or infinity")


def build_model_input(tokenizer: Any, prompt: str, device: torch.device) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        encoded = tokenizer(prompt, return_tensors="pt").input_ids
    return encoded.to(device)


@torch.inference_mode()
def extract_autoregressive(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    include_logits: bool,
) -> tuple[np.ndarray, np.ndarray | None, list[int], str, int]:
    input_ids = build_model_input(tokenizer, prompt, device)
    prompt_tokens = int(input_ids.shape[1])
    generated_ids: list[int] = []
    hidden_rows: list[np.ndarray] = []
    logit_rows: list[np.ndarray] = []
    past_key_values = None
    current_ids = input_ids

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=current_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        if outputs.hidden_states is None or len(outputs.hidden_states) < 4:
            raise RuntimeError("model did not return at least three transformer layers")

        # Exclude the embedding output. Each row describes the current context
        # immediately before selecting the next generated token.
        layer_vector = torch.stack(
            [layer[:, -1, :].squeeze(0) for layer in outputs.hidden_states[1:]],
            dim=0,
        )
        next_logits = outputs.logits[:, -1, :].squeeze(0)
        next_token = int(torch.argmax(next_logits).item())

        hidden_rows.append(layer_vector.float().cpu().numpy())
        if include_logits:
            logit_rows.append(next_logits.float().cpu().numpy())
        generated_ids.append(next_token)

        past_key_values = outputs.past_key_values
        current_ids = torch.tensor([[next_token]], device=device, dtype=input_ids.dtype)
        if tokenizer.eos_token_id is not None and next_token == tokenizer.eos_token_id:
            break

    if not hidden_rows:
        raise RuntimeError("generation produced no trajectory row")

    hidden_states = np.stack(hidden_rows, axis=0).astype(np.float32, copy=False)
    logits = (
        np.stack(logit_rows, axis=0).astype(np.float32, copy=False)
        if include_logits
        else None
    )
    validate_arrays(hidden_states, logits)
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return hidden_states, logits, generated_ids, generated_text, prompt_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "float32", "bfloat16"],
        default="auto",
    )
    parser.add_argument("--output", type=Path, default=Path("limen_tinyllama_trajectory.npz"))
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--no-logits", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")

    device = choose_device(args.device)
    dtype = choose_dtype(device, args.dtype)
    started = time.time()
    print("=== LIMEN TRAJECTORY EXTRACTION ===", flush=True)
    print(f"Model: {args.model_id}", flush=True)
    print(f"Requested revision: {args.revision}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Dtype: {dtype}", flush=True)
    print(f"Max new tokens: {args.max_new_tokens}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    )
    model.to(device)
    model.eval()

    hidden_states, logits, token_ids, generated_text, prompt_tokens = extract_autoregressive(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        include_logits=not args.no_logits,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "hidden_states": hidden_states,
        "token_ids": np.asarray(token_ids, dtype=np.int64),
    }
    if logits is not None:
        arrays["logits"] = logits
    np.savez_compressed(args.output, **arrays)

    resolved_revision = (
        getattr(model.config, "_commit_hash", None)
        or getattr(tokenizer, "_commit_hash", None)
        or args.revision
    )
    metadata_path = args.metadata_output or args.output.with_suffix(".metadata.json")
    metadata = {
        "schema_version": "limen.extraction.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "prompt": args.prompt,
        "prompt_token_count": prompt_tokens,
        "generated_token_count": len(token_ids),
        "generated_token_ids": token_ids,
        "generated_text": generated_text,
        "extraction_protocol": "autoregressive_greedy_preselection_hidden_states_v1",
        "hidden_states_semantics": (
            "Transformer-layer states at the current final context position, "
            "recorded immediately before greedy selection of each generated token."
        ),
        "embedding_output_included": False,
        "logits_included": logits is not None,
        "hidden_states_shape": list(hidden_states.shape),
        "logits_shape": list(logits.shape) if logits is not None else None,
        "device": str(device),
        "dtype": str(dtype),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "runtime_seconds": time.time() - started,
        "source_npz": args.output.name,
        "source_npz_sha256": sha256_file(args.output),
        "scientific_status": (
            "Descriptive extraction only; no causal or functional-localization claim."
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=== EXTRACTION COMPLETE ===", flush=True)
    print(f"Hidden states: {hidden_states.shape}", flush=True)
    print(f"Logits: {None if logits is None else logits.shape}", flush=True)
    print(f"Generated text: {generated_text!r}", flush=True)
    print(f"NPZ: {args.output.resolve()}", flush=True)
    print(f"SHA-256: {metadata['source_npz_sha256']}", flush=True)
    print(f"Metadata: {metadata_path.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
