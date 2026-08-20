#!/usr/bin/env python3
"""Recovered float32 recast command (originally an inline `python3 -c` Bash call,
Claude session 5d793c3e-5e08-4a66-9200-8acfbc5066c9, line 4474, 2026-08-19T12:26:43.694Z).
Saved here as a named script only for replay convenience -- the historical command was
unnamed and inline. Logic is byte-identical to the recovered command: cast every
proj_* column of the two named files to float32, in place. Run from the directory
containing h_random_projection.parquet and orthogonal_subset.parquet.
"""
import pandas as pd

for fname in ["h_random_projection.parquet", "orthogonal_subset.parquet"]:
    df = pd.read_parquet(fname)
    proj_cols = [c for c in df.columns if c.startswith("proj_")]
    for c in proj_cols:
        df[c] = df[c].astype("float32")
    df.to_parquet(fname, index=False)
    print(fname, "recast", len(proj_cols), "columns to float32")
