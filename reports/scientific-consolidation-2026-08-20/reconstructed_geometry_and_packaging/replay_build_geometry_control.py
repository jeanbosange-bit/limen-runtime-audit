"""
Geometry-destroying control (mission section 13). Reads the already-written
_part_core_proj.parquet for the orthogonal-invariance subset models and produces
a per-dimension independent shuffle: for each of the PROJ_DIM feature columns,
independently permute its values ACROSS SAMPLES (rows). This destroys all
cross-dimension covariance/geometric structure for every sample (the joint
distribution is broken) while preserving each dimension's own marginal
distribution -- a real geometry-destroying control.

Explicitly distinguished from what the mission calls out as NOT geometry-destroying:
a single global orthogonal rotation (used separately in _part_core_ortho.parquet)
preserves all pairwise distances/angles by construction; permuting the COORDINATES
of one vector is likewise just a relabeling. What is done here is different: each
dimension is shuffled independently and separately, which does NOT preserve
inter-sample geometry (distances/cosines between samples are not preserved).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OUT = Path("/home/jeanbatuli/V20_probabilistic_dynamics/reports/scientific-consolidation-2026-08-20/reconstructed_geometry_and_packaging/replay")
ORTHOGONAL_SUBSET = [
    "Qwen/Qwen2.5-0.5B-Instruct", "meta-llama/Llama-3.2-1B", "gpt2", "EleutherAI/pythia-160m",
    "facebook/opt-125m", "microsoft/phi-1_5",
]

df = pq.read_table(OUT / "_part_core_proj.parquet").to_pandas()
df = df[df.model_id.isin(ORTHOGONAL_SUBSET)].reset_index(drop=True)

proj_cols = [c for c in df.columns if c.startswith("proj_")]
rng = np.random.default_rng(20260819)

shuffled = df.copy()
for col in proj_cols:
    shuffled[col] = rng.permutation(shuffled[col].values)

shuffled["representation"] = "GEOMETRY_DESTROYING_CONTROL_PER_DIMENSION_SHUFFLE"
shuffled.to_parquet(OUT / "_part_core_geometry_control.parquet", index=False)
print(f"geometry control written: {len(shuffled)} rows, {len(proj_cols)} shuffled dims, models={sorted(shuffled.model_id.unique())}")
