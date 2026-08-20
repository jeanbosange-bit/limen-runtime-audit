# extract_core input/output contract

## Recovered source

Claude session JSONL:
`~/.claude/projects/-home-jeanbatuli-V20-probabilistic-dynamics/5d793c3e-5e08-4a66-9200-8acfbc5066c9.jsonl`.

- Initial `Write`: session event/line 4030.
- Seven subsequent `Edit` events: lines 4046, 4082, 4089, 4196, 4204, 4258, 4270.
- Final recovered SHA-256:
  `18033dbd8be2a3684f9b53e4c756a5e358de32bf2c3811b517935ad270d37a8c`.
- Known edits: all seven session edits were replayed in order and each matched exactly one
  source span.
- Uncertain edits: none observed later in the matching session. Out-of-session editing
  cannot be independently excluded, but the replay comparisons validate the relevant
  outputs.

Archived runtime dependencies recovered by the replay launcher:

- `extract_features.py`: final archive SHA-256
  `c8fa3842e991ba99f1ba5987b26d4c549437dc86e5549ece08d849e08e41a80a`.
- `find_final_norm.py`: archive SHA-256
  `93753eb3477bbc8d9cbe8edda940410ca654530b47fb258f6713c155551e7d86`.

## Inputs

### Panel and identifiers

- Core panel is read from `_core_extended.json`: 12 models, 36 runs/model.
- Run records come from the successful deep rows in multi-model `RUN_INDEX.csv`, grouped
  by `model_id` through archived `extract_features.py`.
- Row identifiers: `model_id`, `run_id`, `prompt_id`, `prompt_family`, string `seed`,
  `token_step`, `layer_index`, `num_layers`; `representation` or `readout_type` is added
  where relevant.

### Tensor inputs

- Required per-run NPZ array: `hidden_states_all_layers_fp16`, loaded as float32,
  shape `(num_layers + 1, sequence_length, d_model)`.
- `token_ids` identifies target token `token_ids[position + 1]`.
- Logits length is obtained from `generated_logits_fp16`; reused Qwen captures may instead
  use `logits_region_fp16`. Logit values themselves establish generated-step count here;
  intermediate readout values are recomputed from H through final norm and LM head.
- `prompt_len = sequence_length - n_generated_steps`; processed positions span generated
  token positions available in H.

### Model/checkpoint inputs

- `AutoTokenizer` and `AutoModelForCausalLM` load the existing checkpoint.
- Load attempts use bfloat16 then float16, move the model to CUDA, and set evaluation mode.
- `find_final_norm_module` identifies the final norm via a read-only forward pass.
- `model.get_output_embeddings()` supplies the LM head.

### Layer and depth logic

- Hidden-state layer indices are `0..num_layers` inclusive; layer 0 is the captured
  embedding output.
- Dense projection and rotation cover every available layer and generated step.
- The script does not write a `normalized_depth` column. Downstream analyses derive it as
  `layer_index / num_layers` and retain exact layer indices.
- Intermediate readout layers are indices divisible by 4 plus the final layer; steps are
  divisible by 3. Final-layer projected rows are labeled `FINAL_NATIVE` by this script;
  all earlier selected layers are `INTERMEDIATE_PROJECTED`.

### Projection and rotation

- Projection dimension: 128.
- Seed: `int(sha256(model_id)[:8], 16)`.
- Projection: float32 Gaussian matrix divided by `sqrt(128)`, then `H @ R`.
- Orthogonal subset: six preselected models; only four completed historically.
- Orthogonal matrix: float64 Gaussian with seed+1, NumPy QR, cast to float32.
- Rotated projection: `(H @ Q) @ R`.
- Projection and orthogonal matrix byte hashes are recorded per model.

### Missingness/failure policy

- Missing NPZ paths and per-run load errors increment `n_fail`; values are not synthesized.
- Missing logits produce dense projection rows but no readout rows.
- Model-level exceptions are logged and recorded; no hidden retry occurs within the script.
- Historical/replayed model-reload failures remain explicit for Qwen2.5-1.5B, Llama-3.2
  1B, Phi-1.5, and Falcon3-1B.

## Direct script outputs

| file | direct content | replay count/schema |
|---|---|---|
| `_part_core_proj.parquet` | Dense random projection | 392,856 rows across 8 successful models when concatenated; 136 columns; float64 parquet projection fields before packaging recast |
| `_part_core_ortho.parquet` | `ORTHOGONAL_ROTATED` only | 183,627 rows across 4 successful subset models; 137 columns; float64 projection fields before recast |
| `_part_core_readout_intermediate.parquet` | Strided projected readout, including intermediate and final-layer labels | 38,233 rows: 31,329 `INTERMEDIATE_PROJECTED`, 6,904 script-labeled `FINAL_NATIVE`; 20 columns |
| `PROJECTION_METADATA.json` | Per-successful-model dimensions, seeds, matrix hashes, final norm | 8 records in historical successful panel |
| `extraction_run.log` | Timestamped progress/failures | execution-dependent text |
| scratch `_core_completed_models.json` | successfully completed model IDs | execution state |
| scratch `_core_model_failures.json` | caught model-level failures | execution state |
| scratch `_extraction_summary_core.json` | total successful/failed run counts | written only on normal process completion |

Primary keys:

- Projection: identifier fields through `num_layers`.
- Rotation: projection key plus `representation`.
- Readout: projection key plus `readout_type`.
- Metadata: `model_id`.

Direct part files contain no intended nulls in projection/rotation values. The direct
readout schema has no commitment/branching/concentration/collapse fields.

## Historical compact packaging outputs

- `h_random_projection.parquet`: 392,856 × 136, projection values recast to float32;
  SHA-256 `0b5c204a…f177b`.
- `orthogonal_subset.parquet`: 367,254 × 137. It combines the 183,627 rotated rows with
  183,627 separately generated geometry-destroying-control rows, then recasts to float32;
  SHA-256 `13414276…b543`.
- `readout_features.parquet`: 185,005 × 24 across the broader extraction. The 31,329
  intermediate rows receive four explicit null columns during schema merge; SHA-256
  `e45ecf21…773d`.
- `PROJECTION_METADATA.json`: 8 records; SHA-256 `c1d68c74…f236`.

`build_geometry_control.py`, `finalize_and_quality.py`, and the later float32 recast
command are separate historical stages. Therefore byte identity between direct part
parquets and packaged parquets is not expected or scientifically meaningful.

