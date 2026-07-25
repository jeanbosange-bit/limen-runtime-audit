#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$HOME/limen-runtime-audit/scripts"
ENV_ACTIVATE="$HOME/sra_env/bin/activate"
REFERENCE_DIR="$SCRIPT_DIR/limen_extraction_output/tinyllama_demo_20260725"
REPLICATION_DIR="$SCRIPT_DIR/limen_extraction_output/tinyllama_replication_20260725"
PYTHON_SCRIPT="$SCRIPT_DIR/extract_limen_trajectory.py"
TEST_SCRIPT="$SCRIPT_DIR/test_extract_limen_trajectory.py"
LOG_FILE="$REPLICATION_DIR/extraction.log"

mkdir -p "$REPLICATION_DIR"
cd "$SCRIPT_DIR"

echo "=== SOURCES ET REFERENCE ==="
sha256sum "$PYTHON_SCRIPT" "$TEST_SCRIPT" "$0"
if [[ ! -f "$REFERENCE_DIR/trajectory.npz" ]]; then
    echo "ERREUR: trajectoire de référence absente : $REFERENCE_DIR/trajectory.npz" >&2
    exit 1
fi
sha256sum "$REFERENCE_DIR/trajectory.npz"

if [[ ! -f "$ENV_ACTIVATE" ]]; then
    echo "ERREUR: environnement absent : $ENV_ACTIVATE" >&2
    exit 1
fi
source "$ENV_ACTIVATE"

echo "=== COMPILATION ET TESTS ==="
python -m py_compile "$PYTHON_SCRIPT" "$TEST_SCRIPT"
python -m unittest -v test_extract_limen_trajectory.py

echo "=== REPLICATION STRICTE TINYLLAMA ==="
set +e
python -u "$PYTHON_SCRIPT" \
    --model-id "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --revision "fe8a4ea1ffedaf415f4da2f062534de366a451e6" \
    --prompt "Explain in two short sentences why the sky appears blue." \
    --max-new-tokens 16 \
    --device auto \
    --dtype auto \
    --output "$REPLICATION_DIR/trajectory.npz" \
    --metadata-output "$REPLICATION_DIR/trajectory.metadata.json" \
    2>&1 | tee "$LOG_FILE"
python_status=${PIPESTATUS[0]}
set -e

echo "Code de sortie Python : $python_status"
if [[ "$python_status" -ne 0 ]]; then
    exit "$python_status"
fi

echo "=== COMPARAISON BIT-A-BIT DES TABLEAUX ==="
python - \
    "$REFERENCE_DIR/trajectory.npz" \
    "$REPLICATION_DIR/trajectory.npz" \
    "$REFERENCE_DIR/trajectory.metadata.json" \
    "$REPLICATION_DIR/trajectory.metadata.json" \
    "$REPLICATION_DIR/comparison.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

reference_path = Path(sys.argv[1])
replication_path = Path(sys.argv[2])
reference_metadata_path = Path(sys.argv[3])
replication_metadata_path = Path(sys.argv[4])
comparison_path = Path(sys.argv[5])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

with np.load(reference_path, allow_pickle=False) as reference:
    with np.load(replication_path, allow_pickle=False) as replication:
        reference_keys = sorted(reference.files)
        replication_keys = sorted(replication.files)
        keys_match = reference_keys == replication_keys
        arrays = {}
        for key in sorted(set(reference_keys) & set(replication_keys)):
            left = np.asarray(reference[key])
            right = np.asarray(replication[key])
            arrays[key] = {
                "shape_reference": list(left.shape),
                "shape_replication": list(right.shape),
                "dtype_reference": str(left.dtype),
                "dtype_replication": str(right.dtype),
                "array_equal": bool(np.array_equal(left, right)),
                "max_absolute_difference": (
                    0.0
                    if np.array_equal(left, right)
                    else float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
                ),
            }

reference_metadata = json.loads(reference_metadata_path.read_text(encoding="utf-8"))
replication_metadata = json.loads(replication_metadata_path.read_text(encoding="utf-8"))
fixed_fields = [
    "model_id",
    "requested_revision",
    "resolved_revision",
    "prompt",
    "prompt_token_count",
    "generated_token_count",
    "generated_token_ids",
    "generated_text",
    "extraction_protocol",
    "hidden_states_semantics",
    "embedding_output_included",
    "logits_included",
    "hidden_states_shape",
    "logits_shape",
    "dtype",
]
metadata_matches = {
    key: reference_metadata.get(key) == replication_metadata.get(key)
    for key in fixed_fields
}
all_arrays_equal = keys_match and all(item["array_equal"] for item in arrays.values())
all_fixed_metadata_equal = all(metadata_matches.values())

comparison = {
    "schema_version": "limen.replication.v1",
    "reference": {
        "path": str(reference_path),
        "npz_sha256": sha256(reference_path),
    },
    "replication": {
        "path": str(replication_path),
        "npz_sha256": sha256(replication_path),
    },
    "keys_match": keys_match,
    "arrays": arrays,
    "fixed_metadata_matches": metadata_matches,
    "all_arrays_bitwise_equal": all_arrays_equal,
    "all_fixed_metadata_equal": all_fixed_metadata_equal,
    "verdict": (
        "EXACT_REPLICATION"
        if all_arrays_equal and all_fixed_metadata_equal
        else "REPLICATION_DIFFERENCE_DETECTED"
    ),
    "note": (
        "NPZ archive hashes may differ even when every stored array is identical; "
        "the scientific comparison is performed on decompressed arrays."
    ),
}
comparison_path.write_text(
    json.dumps(comparison, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(json.dumps(comparison, indent=2, ensure_ascii=False))
if comparison["verdict"] != "EXACT_REPLICATION":
    raise SystemExit(2)
PY

echo "=== TERMINE ==="
echo "Trajectoire répliquée : $REPLICATION_DIR/trajectory.npz"
echo "Métadonnées : $REPLICATION_DIR/trajectory.metadata.json"
echo "Comparaison : $REPLICATION_DIR/comparison.json"
