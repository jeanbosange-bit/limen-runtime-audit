# Geometry-control contract

Input is `_part_core_proj.parquet`, before the later float32 recast. Rows are filtered
to the declared six-model orthogonal subset. Only four models had successful extracted
rows: Qwen2.5-0.5B-Instruct, GPT-2, Pythia-160m, and OPT-125M. Missing Llama and Phi
rows remain missing.

The script preserves all non-projection columns and row order. For each of 128 `proj_`
columns, in column order, it independently applies `numpy.random.default_rng(20260819).permutation`
over all 183,627 selected rows. One generator is advanced continuously across columns.
It then sets `representation` to
`GEOMETRY_DESTROYING_CONTROL_PER_DIMENSION_SHUFFLE` and writes pandas parquet without
the index. This preserves each coordinate marginal but breaks the original row-wise
joint combinations. No matrix is generated, so matrix/hash equality is not applicable.

Coverage is 183,627 rows, 4 models, 144 runs (36/model), and 128 vector dimensions.
Qwen covers layers 0–24; GPT-2, Pythia, and OPT cover layers 0–12. Input and immediate
control projection columns are float64. The post-finalization archived recast changes
all `proj_` columns in the combined table to float32. Keys are
`model_id, run_id, token_step, layer_index, representation`; no key duplicates or
null-pattern changes were observed.

