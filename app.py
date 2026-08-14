from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import spaces

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from limen_runtime_audit import audit_arrays  # noqa: E402
from limen_runtime_audit.live_capture import (  # noqa: E402
    MAX_NEW_TOKENS_LIMIT,
    capture_generation,
)

MAX_UPLOAD_BYTES = 250 * 1024 * 1024


def _create_demo_file() -> Path:
    """Create a deterministic synthetic trajectory for interface testing."""
    demo_dir = Path(tempfile.gettempdir()) / "limen-runtime-audit"
    demo_dir.mkdir(parents=True, exist_ok=True)
    demo_path = demo_dir / "demo_trajectory.npz"

    rng = np.random.default_rng(20260725)
    n_tokens, n_layers, hidden_dim, vocabulary = 16, 12, 32, 64
    layer_drift = rng.normal(0.0, 0.10, size=(n_tokens, n_layers, hidden_dim))
    token_context = rng.normal(0.0, 0.20, size=(n_tokens, 1, hidden_dim))
    hidden_states = (token_context + np.cumsum(layer_drift, axis=1)).astype(np.float32)
    logits = rng.normal(0.0, 1.0, size=(n_tokens, vocabulary)).astype(np.float32)

    np.savez_compressed(
        demo_path,
        hidden_states=hidden_states,
        logits=logits,
    )
    return demo_path


DEMO_PATH = _create_demo_file()
DEMO_METADATA = json.dumps(
    {
        "model": "synthetic-demonstration",
        "prompt_id": "demo-001",
        "note": "Deterministic synthetic arrays; not measurements from a real model.",
    }
)

PLAIN_LANGUAGE = {
    "step_norm": "How far the representation moves between consecutive tokens",
    "curvature": "How sharply the trajectory changes direction",
    "residual_ratio": "How much movement remains after removing the dominant depth profile",
    "entropy": "How uncertain the model output distribution is",
    "margin": "Gap between the two most likely output tokens",
}


def _format_number(value: Any) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "—"
    return f"{number:.6g}"


def _summary_rows(audit: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    metrics = audit.get("summary", {})
    if not isinstance(metrics, dict):
        return rows
    for name, values in metrics.items():
        if not isinstance(values, dict):
            continue
        if not any(key in values for key in ("median", "mean", "q1", "q3", "n")):
            continue
        rows.append([
            name,
            PLAIN_LANGUAGE.get(name, "Descriptive measurement"),
            values.get("n", "—"),
            _format_number(values.get("median", values.get("mean"))),
            _format_number(values.get("q1")),
            _format_number(values.get("q3")),
        ])
    return rows


def _markdown_report(audit: dict[str, Any], source_name: str) -> str:
    rows = _summary_rows(audit)
    lines = [
        "# LIMEN Runtime Audit",
        "",
        f"Source: `{source_name}`",
        "",
        "This report describes activation-trajectory geometry. It does not establish "
        "functional localization, causality, reasoning ability, or correctness.",
        "",
        "| Metric | Plain-language meaning | N | Median | Q1 | Q3 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    if not rows:
        lines.append("| No summary metric found | — | — | — | — | — |")
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "These measurements are descriptive signals. Comparisons require matched "
        "prompts, extraction settings, model revisions, and appropriate controls.",
        "",
    ])
    return "\n".join(lines)


def run_audit(file_path: str | None, metadata_text: str) -> tuple:
    if not file_path:
        raise gr.Error("Choose a trajectory.npz file first.")

    source = Path(file_path)
    if source.stat().st_size > MAX_UPLOAD_BYTES:
        raise gr.Error("The file is larger than the 250 MB Space limit.")

    try:
        metadata = json.loads(metadata_text) if metadata_text.strip() else {}
    except json.JSONDecodeError as exc:
        raise gr.Error(f"Metadata must be valid JSON: {exc.msg}") from exc
    if not isinstance(metadata, dict):
        raise gr.Error("Metadata JSON must be an object.")

    try:
        with np.load(source, allow_pickle=False) as data:
            if "hidden_states" not in data.files:
                raise gr.Error("The NPZ file must contain a 'hidden_states' array.")
            hidden_states = np.asarray(data["hidden_states"])
            logits = np.asarray(data["logits"]) if "logits" in data.files else None
        audit = audit_arrays(hidden_states, logits=logits, metadata=metadata)
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Audit failed: {type(exc).__name__}: {exc}") from exc

    output_dir = Path(tempfile.mkdtemp(prefix="limen-audit-"))
    json_path = output_dir / "audit.json"
    report_path = output_dir / "report.md"
    json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(_markdown_report(audit, source.name), encoding="utf-8")
    return _summary_rows(audit), audit, str(json_path), str(report_path)


@spaces.GPU(duration=60)
def run_live_capture(prompt: str, max_new_tokens: int) -> tuple[str, dict[str, Any], str]:
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Enter a prompt first.")
    try:
        bundle = capture_generation(prompt, max_new_tokens=int(max_new_tokens))
    except Exception as exc:  # pragma: no cover - surfaced to the UI
        raise gr.Error(f"Capture failed: {type(exc).__name__}: {exc}") from exc

    meta = bundle["run_metadata"]
    cov = bundle["coverage"]
    chain = bundle["chain_integrity"]
    summary = (
        f"**Response:** {meta['response_text']}\n\n"
        f"- Model: `{meta['model']}` (public checkpoint, greedy decoding)\n"
        f"- Generated tokens: {meta['n_generated_tokens']} in {meta['latency_s']:.2f} s\n"
        f"- Evidence coverage: {cov['available']}/{cov['expected_total']} "
        f"expected channel-slots available "
        f"({cov['coverage_fraction']:.0%} of this run's declared capture profile)\n"
        f"- Hash-chain integrity: {'valid' if chain['valid'] else 'BROKEN at ' + str(chain['first_broken_event_id'])}\n\n"
        "Tier: **[I] directly observed.** No decision policy or trained "
        "threshold was applied to this output."
    )

    output_dir = Path(tempfile.mkdtemp(prefix="limen-evidence-"))
    json_path = output_dir / "evidence_bundle.json"
    json_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary, bundle, str(json_path)


ABOUT_MARKDOWN = """
## Evidence tiers used throughout this project

| Tier | Meaning |
|---|---|
| **[I]** | Directly observed — recorded from a real run, not inferred. |
| **[II]** | Statistically supported — held up under held-out testing and stated controls. |
| **[III]** | Interpretation — a reasonable reading of [I]/[II] evidence, not itself tested. |
| **[IV]** | Hypothesis — proposed, not yet tested. |

## Absence is not zero

Every observation channel in the evidence tab reports one of four states:
`AVAILABLE`, `NOT_CAPTURED` (this run's profile never intended to record it),
`NOT_SUPPORTED` (the backend cannot produce it here), or `FAILED` (the
profile required it and it is missing — an anomaly, not a design choice).
A missing signal is never silently treated as zero, and no link between
signals is invented from file order or array position — identity is always
explicit (`run_id -> event_id -> step_index`).

## What this Space intentionally does not include

This is a public, adapted demonstration of an evidence-architecture
*pattern* used in a larger private research programme on LLM runtime
observability and actuator research (compute budget, self-verification,
retry, and evidence-gated recommendation policies). That programme's
trained models, calibrated thresholds, feature-engineering details, and raw
experiment archives are not published here. The companion papers, once
deposited, will be linked from the README.

## Scientific background

- [Four Dynamical Regimes in Large Language Models](https://doi.org/10.5281/zenodo.20348878)
- [Conditional Dynamic Signatures in Large Language Models](https://doi.org/10.5281/zenodo.20361289)
- [Dynamic-Layer Controllability without Universal Semantic Recovery](https://doi.org/10.5281/zenodo.20400171)
- [A Runtime Trajectory Dynamics Framework for Large Language Models](https://doi.org/10.5281/zenodo.20602685)
"""

with gr.Blocks(title="LIMEN Runtime Audit") as demo:
    gr.Markdown("""
# LIMEN Runtime Audit

Two tools in one Space: a descriptive geometry audit over uploaded
activation trajectories, and a live evidence-capture demo that runs a
small public model on this Space's GPU and records a synchronized,
hash-chained, multi-channel observation record.

Neither tool proves where a function is located, why the model answered,
or whether a representation caused an output.
""")

    with gr.Tab("Descriptive audit (upload)"):
        gr.Markdown("""
Upload a `trajectory.npz` produced from an open transformer model. The file must
contain `hidden_states` with shape `[tokens, layers, hidden_dim]`; `logits`
with shape `[tokens, vocabulary]` is optional.
""")
        with gr.Row():
            trajectory = gr.File(
                label="Trajectory file (.npz)",
                file_types=[".npz"],
                type="filepath",
            )
            metadata = gr.Textbox(
                label="Optional metadata (JSON)",
                value='{"model": "model-name", "prompt_id": "example-001"}',
                lines=5,
            )
        gr.Markdown(
            "**No trajectory yet?** Download the synthetic demonstration file below, "
            "or click the example to load it automatically. It verifies the interface "
            "but is not a scientific model result."
        )
        demo_download = gr.File(
            value=str(DEMO_PATH),
            label="Download demo_trajectory.npz",
            interactive=False,
        )
        gr.Examples(
            examples=[[str(DEMO_PATH), DEMO_METADATA]],
            inputs=[trajectory, metadata],
            label="Ready-to-use demonstration",
        )
        audit_button = gr.Button("Run descriptive audit", variant="primary")
        summary = gr.Dataframe(
            headers=["Metric", "Plain-language meaning", "N", "Median", "Q1", "Q3"],
            datatype=["str", "str", "str", "str", "str", "str"],
            interactive=False,
            label="Summary",
        )
        raw_json = gr.JSON(label="Complete audit")
        with gr.Row():
            json_download = gr.File(label="Download audit.json")
            report_download = gr.File(label="Download report.md")

        audit_button.click(
            fn=run_audit,
            inputs=[trajectory, metadata],
            outputs=[summary, raw_json, json_download, report_download],
            show_progress="full",
        )

        gr.Markdown("""
### Responsible interpretation

Compare matched runs and keep the model revision, prompt, tokenization and
extraction protocol fixed. Decodability is not functional localization, and a
statistic geometric pattern is not automatically a dynamic mechanism.
""")

    with gr.Tab("Live evidence capture"):
        gr.Markdown(f"""
Runs `Qwen/Qwen2.5-0.5B-Instruct` (a small public instruction-tuned model)
on this Space's GPU for one greedy generation, and records three
synchronized observation channels per generated token: **generation**
(which token), **probability** (output-distribution entropy and top-1/top-2
margin), and **latent** (movement of hidden states at a few fixed, sampled
layers). Every channel is hash-chained; tampering or reordering steps is
detectable, not silently accepted.

This is an observability demo, not a decision system. Max {MAX_NEW_TOKENS_LIMIT} new tokens.
""")
        with gr.Row():
            live_prompt = gr.Textbox(
                label="Prompt",
                value="Explain briefly why the sky is blue.",
                lines=3,
            )
            live_max_tokens = gr.Slider(
                minimum=8,
                maximum=MAX_NEW_TOKENS_LIMIT,
                value=32,
                step=1,
                label="Max new tokens",
            )
        live_button = gr.Button("Run live capture", variant="primary")
        live_summary = gr.Markdown(label="Result")
        live_json = gr.JSON(label="Evidence bundle")
        live_download = gr.File(label="Download evidence_bundle.json")

        live_button.click(
            fn=run_live_capture,
            inputs=[live_prompt, live_max_tokens],
            outputs=[live_summary, live_json, live_download],
            show_progress="full",
        )

    with gr.Tab("About / evidence tiers"):
        gr.Markdown(ABOUT_MARKDOWN)


if __name__ == "__main__":
    demo.launch()
