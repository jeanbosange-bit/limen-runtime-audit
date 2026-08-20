import hashlib
import json
import subprocess
from pathlib import Path

OUT = Path("/home/jeanbatuli/limen-runtime-audit/reports/deep-feature-extraction-2026-08-19")

# Files to include in the transfer package -- all reports, all compact feature files,
# schemas, indices, hashes, projection metadata. Excludes _part_*.parquet fragments
# (superseded by the merged deliverables) and any scratch/log noise.
DELIVERABLES = [
    "00_INPUT_INVENTORY.md", "01_CORE_PANEL.md", "02_FEATURE_SCHEMA.md", "03_DATA_QUALITY.md",
    "04_DECODABILITY_TARGET_FEASIBILITY.md", "05_ORTHOGONAL_PREPARATION.md", "06_DYNAMIC_REGIME_READINESS.md",
    "07_TRANSFER_PACKAGE_REPORT.md",
    "layerwise_features.parquet", "attention_features.parquet", "qkv_features.parquet",
    "readout_features.parquet", "h_random_projection.parquet", "orthogonal_subset.parquet",
    "dynamic_regime_inputs.parquet", "token_path_curves.parquet",
    "PROJECTION_METADATA.json", "_input_inventory.json", "_core_extended.json", "_quality_summary.json",
]


def build_inventory_and_hashes():
    lines = ["relative_path\tsize_bytes\tsha256"]
    hash_lines = []
    total = 0
    for name in DELIVERABLES:
        p = OUT / name
        if not p.exists():
            print(f"WARNING: missing deliverable {name}")
            continue
        size = p.stat().st_size
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{name}\t{size}\t{h}")
        hash_lines.append(f"{h}  {name}")
        total += size
    (OUT / "TRANSFER_SHA256SUMS.txt").write_text("\n".join(hash_lines) + "\n")
    print(f"TRANSFER_SHA256SUMS.txt: {len(hash_lines)} files, {total} bytes total")
    return total


if __name__ == "__main__":
    total = build_inventory_and_hashes()
    print(f"TOTAL_DELIVERABLE_BYTES={total}")
