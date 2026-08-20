# Geometry-destroying control & final packaging reconstruction report

## Verdict

`FULLY_REPRODUCIBLE` for the geometry-destroying control and final packaging stage,
byte-identical to the two canonical compact tables this stage produces.

This closes the specific gap `EXTRACT_CORE_RECONSTRUCTION_REPORT.md` left open:
"Separate validation is still required for the geometry-destroying-control builder and
final packaging pipeline." Combined with that report's own `NUMERICALLY_IDENTICAL`
finding for `extract_core.py`'s projection/rotation outputs, and
`PHASE9_RECONSTRUCTION_REPORT.md`'s `FULLY_REPRODUCIBLE` finding for `phase9_logit_lens.py`,
the full input chain feeding Phase 9's own numerical results is now reconstructible
end-to-end for every table Phase 9 actually consumes. See "End-to-end chain" below.

## Recovered source

Claude session JSONL:
`~/.claude/projects/-home-jeanbatuli-V20-probabilistic-dynamics/5d793c3e-5e08-4a66-9200-8acfbc5066c9.jsonl`
(the same deep-extraction session `EXTRACT_CORE_CONTRACT.md` used).

- `build_geometry_control.py`: single `Write` event, line 4093
  (2026-08-19T11:40:45.196Z). No subsequent `Edit` found anywhere in the session.
  Recovered SHA-256: `e58a9e09016b12a80e02d1741aacf1f6c552f2bf4132f9221b3ea2ef15ca0670` —
  **exact match** to the truncated hash already recorded in `ANALYSIS_PROVENANCE_AUDIT.md`
  ("e58a9e…a0670"), independently confirmed here.
- `finalize_and_quality.py`: single `Write` event, line 4175 (2026-08-19T11:45:47.821Z).
  No subsequent `Edit` found. Recovered SHA-256:
  `754f100f6cb46d61169d680f4e7b54f694e5fe76f1c6bbc4fe9418d5866cbbf5`.
- The float32 recast: an inline `python3 -c "..."` command (not a named script), `Bash`
  event at line 4474 (2026-08-19T12:26:43.694Z), run after `finalize_and_quality.py`
  (invoked at lines 4414 and 4464 — two runs, consistent with a fix-and-rerun cycle
  visible in the surrounding session but not relevant to the recast itself). It casts
  every `proj_*` column of `h_random_projection.parquet` and `orthogonal_subset.parquet`
  to `float32` and overwrites the same two files in place. Exact command reproduced
  in `replay_recast.py` in this directory.
- Uncertain edits: none observed for either script anywhere later in the session.
  Out-of-session editing cannot be independently excluded for either recovered script,
  the same caveat `EXTRACT_CORE_CONTRACT.md` and `PHASE9_RECONSTRUCTION_REPORT.md`
  already carry — the byte-identical replay below is the strongest available evidence
  against it having mattered.

## Explicit pipeline-stage boundaries

The mission asked these to be kept separate; none are conflated below or in the source:

1. **Source extraction** (`extract_core.py`, `extract_features.py`) — reads raw NPZ
   captures, produces per-model `_part_core_proj.parquet` / `_part_core_ortho.parquet` /
   `_part_core_readout_intermediate.parquet`. Not re-run here; consumed read-only from
   `../reconstructed_extract_core/replay/isolated_*/` (Codex's own already-replayed,
   per-model-isolated outputs, confirmed `NUMERICALLY_IDENTICAL` to canonical in that
   report). No model was loaded and no generation occurred in this gate.
2. **Projection** — `extract_core.py`'s dense 128-D Gaussian projection, `H @ R`, one
   fixed matrix per model. Upstream of this gate; not touched here beyond consuming its
   already-validated output.
3. **Orthogonal control** — `extract_core.py`'s `(H @ Q) @ R` rotation, `Q` from a
   seed+1 QR decomposition. Also upstream; not touched here.
4. **Geometry-destroying control** — `build_geometry_control.py`, reconstructed and
   replayed in this gate. Reads the projection output (stage 2), independently permutes
   each of the 128 `proj_*` columns **across samples** (`np.random.default_rng(20260819)`,
   one `rng.permutation()` call per column, in column order) for the 4 historically
   successful orthogonal-subset models. This is explicitly distinct from stage 3: a
   rotation preserves inter-sample distances/angles by construction; this per-dimension
   shuffle destroys them while preserving each dimension's own marginal distribution.
5. **Intermediate projected readout** — `extract_core.py`'s recomputation of
   `INTERMEDIATE_PROJECTED` rows (final-norm + LM head applied to a non-final hidden
   state). Upstream; consumed read-only here only for the packaging step.
6. **Generation-native final readout** (`FINAL_NATIVE`) — the model-wide dense
   generation-time distribution, produced by the separate `extract_features.py` pipeline,
   **not** `extract_core.py` and **not** touched by either script in this gate. This
   remains the open gap both this report and `EXTRACT_CORE_RECONSTRUCTION_REPORT.md`
   identify: `extract_features.py` + its own finalizer (Codex's provenance priority #3)
   has not yet been reconstructed by either session.
7. **Packaging/finalization** — `finalize_and_quality.py`, reconstructed and replayed in
   this gate. Concatenates stage-2/3/4/5 part files into the named compact deliverables,
   checks for duplicate keys and out-of-range values, and writes a data-quality report.
   The float32 recast (a separate, later, unnamed command) is the final step, applied
   only to `proj_*` columns of the two large tables.

## Replay method

Inputs to stages 4 and 7 were the row-count-identical, key-order-identical outputs Codex's
own `extract_core.py` replay already produced per model
(`../reconstructed_extract_core/replay/isolated_*/`). These were concatenated in the
model order recovered directly from the canonical `h_random_projection.parquet`'s own
row sequence (verified by first-appearance scan: Qwen, TinyLlama, gpt2, pythia-160m,
opt-125m, deepseek-coder, granite, SmolLM — each model a single contiguous block, 8
blocks total) — **not assumed**, checked directly against the canonical file before
replay. This merge was independently pre-validated against canonical
`h_random_projection.parquet` (before running any new stage): row count, key-column
order, and a 10-column value sample matched exactly after a float32 cast, with zero
difference. Both recovered scripts were then run unmodified except for the `OUT` path
constant, exactly as `EXTRACT_CORE_CONTRACT.md`'s own replay methodology (`replay_extract_core.py`)
already established as the correct, minimal-intervention pattern. The recast command was
applied verbatim afterward. No extraction logic, seed, parameter, or formula was changed
in either script.

## Match summary

| output | rows × columns | primary key | null pattern | max numeric diff | SHA-256 | status |
|---|---:|---|---|---:|---|---|
| `h_random_projection.parquet` | 392,856 × 136 | full identifier set through `num_layers` | zero nulls in any column | 0 | `0b5c204a…f177b` | `BYTE_IDENTICAL` |
| `orthogonal_subset.parquet` | 367,254 × 137 | projection key + `representation` | zero nulls in any column | 0 | `13414276…b543` | `BYTE_IDENTICAL` |
| `readout_features.parquet`, `INTERMEDIATE_PROJECTED` rows only | 31,329 × 20 (+4 canon-only null columns) | projection key + `readout_type` | `commitment_score`/`branching_factor`/`concentration`/`collapse_score` confirmed 100% null on both sides | 0 (20 common numeric columns) | not computed (see below) | `SEMANTICALLY_IDENTICAL` |
| `readout_features.parquet`, full packaged file | — | — | — | — | — | `NOT_ATTEMPTED` (see limitation) |

Both `BYTE_IDENTICAL` results are full-file SHA-256 matches against the canonical files
at `~/limen-runtime-audit/reports/deep-feature-extraction-2026-08-19/`, computed after
the recast step, on the complete file — not a sampled check. The recorded hashes
(`0b5c204a…f177b`, `13414276…b543`) are the exact values already named in
`EXTRACT_CORE_CONTRACT.md`'s historical-packaging section; this gate reproduces them
independently from a fresh replay rather than merely re-quoting them.

## Model/run/layer coverage

`h_random_projection.parquet`: 8/8 historically successful models, every run each model
has (392,856 rows total, matching `EXTRACT_CORE_CONTRACT.md` exactly). `orthogonal_subset.parquet`:
4 models (Qwen2.5-0.5B-Instruct, gpt2, pythia-160m, opt-125m — the historically successful
subset of the 6 originally targeted; Llama-3.2-1B and phi-1.5 are absent from
`_part_core_proj.parquet` upstream, consistent with the disclosed Jetson CUDA
model-reload failures already named in `LIMITATIONS.md`), each contributing identical row
counts to both the `ORTHOGONAL_ROTATED` and `GEOMETRY_DESTROYING_CONTROL_PER_DIMENSION_SHUFFLE`
representations (verified: 52,975/29,525/43,290/57,837 rows per model, exactly matched
across both representations — the geometry-destroying control was built from precisely
the same row population as the rotation, not a resampled or reordered subset).

## Limitation: the readout_features.parquet full-file merge was not replayable

`finalize_and_quality.py`'s readout-merge step requires the **pre-merge**, model-wide-only
`readout_features.parquet` (the `FINAL_NATIVE`-only table `extract_features.py`'s own
finalizer produces) as its base, then adds `INTERMEDIATE_PROJECTED` rows on top. That
pre-merge file does not exist in this session's recovered artifacts — only the canonical,
**already-merged** `readout_features.parquet` (185,005 rows, already containing both
`FINAL_NATIVE` and `INTERMEDIATE_PROJECTED`) is available. Substituting it as the merge
base produces 31,329 spurious duplicate keys — a byproduct of double-merging an
already-merged file, not a defect in `finalize_and_quality.py`. This was caught, not
silently worked around: the `INTERMEDIATE_PROJECTED` subset was instead validated directly
by key- and value-comparison against the canonical file's own `INTERMEDIATE_PROJECTED`
rows (table above, `SEMANTICALLY_IDENTICAL`), which is the only comparison the available
inputs actually support. Full-file byte identity for `readout_features.parquet` remains
untested and should not be inferred from the two `BYTE_IDENTICAL` results above.

## End-to-end chain: can Phase 9's input chain now be considered fully reconstructible?

**Yes, for every table Phase 9's own numerical results depend on.** Per
`PHASE9_RECONSTRUCTION_REPORT.md`'s own stated inputs, `phase9_logit_lens.py` reads
exactly `h_random_projection.parquet` and `readout_features.parquet`'s
`INTERMEDIATE_PROJECTED` rows (plus the pilot's own `pilot_metrics.csv`, unrelated to this
gate). This report now independently validates both of those inputs — one
`BYTE_IDENTICAL`, one `SEMANTICALLY_IDENTICAL` on the exact subset consumed — from
artifacts that were themselves already `NUMERICALLY_IDENTICAL`-validated one layer further
upstream (`extract_core.py`'s projection/rotation output). Chained together: raw NPZ
→ `extract_core.py` (partial, but the Phase-9-relevant slice exact) →
`build_geometry_control.py` + `finalize_and_quality.py` + float32 recast (this gate,
`FULLY_REPRODUCIBLE` for the tables checked) → `phase9_logit_lens.py` (already
`FULLY_REPRODUCIBLE`, byte-identical on all five outputs) → Phase 9's tables. No gap
remains in that specific chain.

What remains genuinely open, unchanged by this gate: `extract_features.py` and its own
finalizer (the six model-wide tables — `layerwise_features.parquet`,
`attention_features.parquet`, `qkv_features.parquet`, `dynamic_regime_inputs.parquet`,
`token_path_curves.parquet`, and `readout_features.parquet`'s `FINAL_NATIVE` portion) —
Codex's own provenance priority #3, not attempted in either this gate or
`EXTRACT_CORE_RECONSTRUCTION_REPORT.md`. Phase 9 itself does not depend on any of these,
so this gap does not weaken the "Phase 9 input chain is reconstructible" conclusion above,
but it does mean the *broader* multi-model deep-feature package is still only
`PARTIALLY_REPRODUCIBLE` overall, exactly as `STATUS_MATRIX.md` already states.

## Scientific impact

No existing scientific result changes. This gate adds reproducibility evidence only:
`h_random_projection.parquet` and `orthogonal_subset.parquet` — and therefore every
decodability, geometry-invariance, and Phase-9 finding built on them
(`TOP_10_DEFENSIBLE_FINDINGS.md` findings 4, `RPTA_CURRENT_EVIDENCE.md` layers 2–3 and 6,
`PHASE9_THRESHOLD_FREE_AUDIT.md`) — are now confirmed to derive, byte-for-byte, from the
disclosed formulas, seed, and script logic already on record, closing the specific
`PARTIALLY_REPRODUCIBLE` → `FULLY_REPRODUCIBLE` gap `EXTRACT_CORE_RECONSTRUCTION_REPORT.md`
left open for these two files.
