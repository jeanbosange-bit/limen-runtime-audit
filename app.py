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

MAX_UPLOAD_BYTES = 250 * 1024 * 1024

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
    metrics = audit.get("metrics", audit)
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


@spaces.GPU(duration=60)
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


with gr.Blocks(title="LIMEN Runtime Audit") as demo:
    gr.Markdown("""
# LIMEN Runtime Audit

Upload a `trajectory.npz` produced from an open transformer model. The file must
contain `hidden_states` with shape `[tokens, layers, hidden_dim]`; `logits`
with shape `[tokens, vocabulary]` is optional.

The tool reports descriptive measurements of how activations evolve across
tokens and layers. It does **not** prove where a function is located, why the
model answered, or whether a representation caused an output.
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
    )

    gr.Markdown("""
### Responsible interpretation

Compare matched runs and keep the model revision, prompt, tokenization and
extraction protocol fixed. Decodability is not functional localization, and a
static geometric pattern is not automatically a dynamic mechanism.
""")


if __name__ == "__main__":
    demo.launch()
