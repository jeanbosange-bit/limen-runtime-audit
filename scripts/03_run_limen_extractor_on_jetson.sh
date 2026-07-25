#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$HOME/limen-runtime-audit/scripts"
ENV_ACTIVATE="$HOME/sra_env/bin/activate"
OUTPUT_DIR="$SCRIPT_DIR/limen_extraction_output/tinyllama_demo_20260725"
PYTHON_SCRIPT="$SCRIPT_DIR/extract_limen_trajectory.py"
TEST_SCRIPT="$SCRIPT_DIR/test_extract_limen_trajectory.py"
LOG_FILE="$OUTPUT_DIR/extraction.log"

mkdir -p "$OUTPUT_DIR"
cd "$SCRIPT_DIR"

echo "=== EMPREINTES DES SOURCES ==="
sha256sum "$PYTHON_SCRIPT" "$TEST_SCRIPT" "$0"

if [[ ! -f "$ENV_ACTIVATE" ]]; then
    echo "ERREUR: environnement absent : $ENV_ACTIVATE" >&2
    exit 1
fi
source "$ENV_ACTIVATE"

echo "=== ENVIRONNEMENT ==="
python --version
python - <<'PY'
import numpy
import torch
import transformers
print("numpy:", numpy.__version__)
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device:", torch.cuda.get_device_name(0))
PY

echo "=== COMPILATION PYTHON ==="
python -m py_compile "$PYTHON_SCRIPT" "$TEST_SCRIPT"

echo "=== TESTS UNITAIRES HORS LIGNE ==="
python -m unittest -v test_extract_limen_trajectory.py

echo "=== EXTRACTION TINYLLAMA ==="
set +e
python -u "$PYTHON_SCRIPT" \
    --model-id "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --revision "fe8a4ea1ffedaf415f4da2f062534de366a451e6" \
    --prompt "Explain in two short sentences why the sky appears blue." \
    --max-new-tokens 16 \
    --device auto \
    --dtype auto \
    --output "$OUTPUT_DIR/trajectory.npz" \
    --metadata-output "$OUTPUT_DIR/trajectory.metadata.json" \
    2>&1 | tee "$LOG_FILE"
python_status=${PIPESTATUS[0]}
set -e

echo "Code de sortie Python : $python_status"
echo "Journal : $LOG_FILE"
if [[ "$python_status" -ne 0 ]]; then
    exit "$python_status"
fi

echo "=== VALIDATION DU FICHIER POUR LE SPACE ==="
python - "$OUTPUT_DIR/trajectory.npz" <<'PY'
import hashlib
import sys
from pathlib import Path
import numpy as np

path = Path(sys.argv[1])
with np.load(path, allow_pickle=False) as data:
    print("arrays:", data.files)
    print("hidden_states:", data["hidden_states"].shape, data["hidden_states"].dtype)
    print("logits:", data["logits"].shape, data["logits"].dtype)
    print("token_ids:", data["token_ids"].shape, data["token_ids"].dtype)
    assert data["hidden_states"].ndim == 3
    assert data["logits"].ndim == 2
    assert data["hidden_states"].shape[0] == data["logits"].shape[0]
    assert np.isfinite(data["hidden_states"]).all()
    assert np.isfinite(data["logits"]).all()
digest = hashlib.sha256(path.read_bytes()).hexdigest()
print("sha256:", digest)
print("size_bytes:", path.stat().st_size)
PY

echo "=== TERMINE ==="
echo "Fichier à transférer vers le PC puis à déposer dans le Space :"
echo "$OUTPUT_DIR/trajectory.npz"
echo "Métadonnées :"
echo "$OUTPUT_DIR/trajectory.metadata.json"
