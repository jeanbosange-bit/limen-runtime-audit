from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import layer_trajectory_metrics, probability_baselines


def _summary(x: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(x, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"n": 0, "median": None, "q1": None, "q3": None}
    return {
        "n": int(finite.size),
        "median": float(np.median(finite)),
        "q1": float(np.quantile(finite, 0.25)),
        "q3": float(np.quantile(finite, 0.75)),
    }


def audit_arrays(
    hidden_states: np.ndarray,
    logits: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trajectory = layer_trajectory_metrics(hidden_states)
    baselines = probability_baselines(logits) if logits is not None else {}
    return {
        "schema_version": "limen.audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "input": {
            "hidden_states_shape": list(hidden_states.shape),
            "logits_shape": list(logits.shape) if logits is not None else None,
            "metadata": metadata or {},
        },
        "trajectory_metrics": {k: v.tolist() for k, v in trajectory.items()},
        "probability_baselines": {k: v.tolist() for k, v in baselines.items()},
        "summary": {
            **{f"trajectory.{k}": _summary(v) for k, v in trajectory.items()},
            **{f"baseline.{k}": _summary(v) for k, v in baselines.items()},
        },
        "interpretation_boundary": (
            "Descriptive audit only. These measurements do not establish "
            "functional localization, semantic state identity, causality, "
            "reasoning, or controllability."
        ),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(audit: dict[str, Any], output_dir: Path, source: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit["input"]["source_file"] = source.name
    audit["input"]["source_sha256"] = sha256_file(source)
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# LIMEN Runtime Audit",
        "",
        f"- Schema: `{audit['schema_version']}`",
        f"- Source: `{source.name}`",
        f"- SHA-256: `{audit['input']['source_sha256']}`",
        f"- Hidden states: `{audit['input']['hidden_states_shape']}`",
        f"- Logits: `{audit['input']['logits_shape']}`",
        "",
        "## Metric summaries",
        "",
        "| Metric | N | Median | Q1 | Q3 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in audit["summary"].items():
        def fmt(value: Any) -> str:
            return "NA" if value is None else f"{value:.6g}"
        lines.append(
            f"| `{name}` | {values['n']} | {fmt(values['median'])} | "
            f"{fmt(values['q1'])} | {fmt(values['q3'])} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        audit["interpretation_boundary"],
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
