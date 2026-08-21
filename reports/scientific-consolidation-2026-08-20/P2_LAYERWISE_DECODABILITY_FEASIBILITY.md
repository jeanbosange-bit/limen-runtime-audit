# P2 layerwise decodability feasibility

## Verdict

**GO_PARTIAL_PANEL**. Archived data support a strict prompt-held-out, per-model pilot
for future-token identity without new inference. The archive does not support a strict
held-out-prompt task-family probe, because its 12 prompt IDs are in one-to-one
correspondence with 12 task families. Exact token targets are reconstructible but
class-sparse and tokenizer-specific, so initial claims must be per-model and narrowly
scoped.

This is a feasibility audit only. No probe was trained.

## 1. Available models and runs

`P2_FEASIBILITY_MANIFEST.csv` contains all 687 indexed attempts. All are marked SUCCESS
in the index, but three OPT-350M rows have no deep artifact. Hidden states and t+1
targets are technically eligible for 684 runs across 19 models. Every eligible model
has 12 prompts × 3 seeds = 36 runs.

The recommended initial comparison panel is the eight models already represented in
the validated 128-dimensional projection table: Pythia-160M, SmolLM-360M-Instruct,
Qwen2.5-0.5B-Instruct, TinyLlama-1.1B-Chat, DeepSeek-Coder-1.3B-Instruct, OPT-125M,
GPT-2, and Granite-3.0-1B-A400M-Instruct. This alignment permits later comparison with
Phase 9 without claiming that P2 was demonstrated across the 18-model scalar panel.

For a compute-bounded first execution, start with GPT-2, Pythia-160M, OPT-125M, and
SmolLM-360M-Instruct. Expansion to the remaining four is conditional on protocol and
permutation controls passing.

| Model | Deep runs | Layers | Width | t+1 steps | P2 status |
|---|---:|---:|---:|---:|---|
| EleutherAI/gpt-neo-125m | 36 | 12 | 768 | 4,454 | Eligible |
| EleutherAI/pythia-70m | 36 | 6 | 512 | 4,228 | Eligible |
| EleutherAI/pythia-160m | 36 | 12 | 768 | 4,039 | Eligible; initial panel |
| EleutherAI/pythia-410m | 36 | 24 | 1,024 | 4,277 | Eligible |
| HuggingFaceTB/SmolLM-360M-Instruct | 36 | 32 | 960 | 2,608 | Eligible; initial panel |
| HuggingFaceTB/SmolLM2-135M | 36 | 30 | 576 | 4,385 | Eligible |
| Qwen/Qwen2.5-0.5B-Instruct | 36 | 24 | 896 | 1,145 | Eligible from reused token IDs/logit-region boundary |
| Qwen/Qwen2.5-1.5B-Instruct | 36 | 28 | 1,536 | 865 | Eligible |
| Qwen/Qwen2.5-Coder-0.5B-Instruct | 36 | 24 | 896 | 902 | Eligible |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 36 | 22 | 2,048 | 1,863 | Eligible |
| bigscience/bloomz-560m | 36 | 24 | 1,024 | 638 | Eligible for H; three runs lack t+2 only |
| deepseek-ai/deepseek-coder-1.3b-instruct | 36 | 24 | 2,048 | 1,781 | Eligible |
| distilgpt2 | 36 | 6 | 768 | 4,240 | Eligible |
| facebook/opt-125m | 36 | 12 | 768 | 3,294 | Eligible; initial panel |
| facebook/opt-350m | 0 | — | — | 0 | Excluded: no deep artifact |
| gpt2 | 36 | 12 | 768 | 4,413 | Eligible; initial panel |
| ibm-granite/granite-3.0-1b-a400m-instruct | 36 | 24 | 1,024 | 1,279 | Eligible |
| meta-llama/Llama-3.2-1B | 36 | 16 | 2,048 | 3,335 | Eligible |
| microsoft/phi-1_5 | 36 | 24 | 2,048 | 4,476 | Eligible |
| tiiuae/Falcon3-1B-Instruct | 36 | 18 | 2,048 | 1,090 | Eligible |

## 2. Hidden-state inventory

- Array: `hidden_states_all_layers_fp16`.
- Dtype: float16 in all 684 available archives.
- Transformer layer counts: 6, 12, 16, 18, 22, 24, 28, 30, or 32; the array also
  includes layer 0, so analyzed layer rows are `num_layers + 1`.
- Hidden widths: 512–2048.
- Sequence lengths: 23–255.
- Generation steps: 2–224.
- Uncompressed hidden-state volume: approximately 3.76 GiB total; maximum single run
  approximately 24.5 MiB.
- Potential aligned layer rows: 999,664 for t+1 and 985,516 for t+2 after including
  Qwen reused captures.

Streaming one run at a time is safe under Jetson constraints. Loading the full archive
or a cross-model float32 matrix is unnecessary. The existing fixed 128-dimensional
random projections contain 392,856 rows for 8 models/288 runs and are suitable for a
low-memory screening probe. A projection result is a lower-bound test of information
retained by that fixed linear map, not proof about all of raw H.

## 3. Prompt identity coverage

There are exactly 12 unique prompt IDs, 12 unique prompt hashes, and 12 families. Each
model/prompt cell has three seeds. Frozen prompt text and source prompt ID are available
in `PROMPT_PANEL_REFERENCE.json`; generation metadata carries prompt hashes for new
model-wide captures. Reused Qwen rows lack model-wide prompt hashes, but their run IDs,
prompt IDs, seed registry, and frozen panel mapping identify the source prompt.

No duplicate prompt hash exists among the 12 panel entries. A normalized lexical
similarity audit found no near-identical pair; the maximum pairwise SequenceMatcher
ratio was 0.455. Modified prompt IDs retain their distinct source IDs in the frozen
panel. This is evidence against within-panel text duplication, not a semantic guarantee
against every possible template relation.

## 4. Leakage audit

Mandatory split group is `prompt_id` across all seeds, runs, token steps, layers, and
models used in an analysis. No derivative row from a prompt may cross partitions.
Random token-row, run-only, or seed-only splitting is prohibited.

There are no duplicate run IDs. Fifteen repeated artifact-hash groups cover 38 runs;
each group is confined to one model/prompt and represents identical outputs across
seeds. Prompt grouping keeps every such duplicate in one partition, so it cannot cross
train/test. These rows must not be counted as independent prompt evidence.

The critical limitation is family identity: `prompt_id ↔ task_family` is one-to-one.
A held-out prompt is therefore also a completely unseen class for task-family
classification. Task-family probing with the required split is not executable on this
archive. Leave-one-seed-out would answer a different, weaker question and is rejected
for P2.

## 5. Proposed strict split

Use nested 12-fold leave-one-prompt-out evaluation, independently per model and target.
Order prompts by SHA-256 of the literal prompt ID. In fold `i`:

- TEST = prompt `i`;
- VALIDATION = prompt `(i + 1) mod 12`;
- TRAIN = the remaining 10 prompts.

All three seeds and all token/layer rows for a prompt remain together. Preprocessing is
fit on TRAIN only. Regularization is chosen on VALIDATION only. TEST is evaluated once
per outer fold. Aggregate the 12 outer predictions and bootstrap the 12 prompt clusters,
not token rows. Twelve prompt clusters remain small; confidence intervals will be wide
and are descriptive.

Family stratification is impossible because there is one prompt per family. The split
therefore measures transfer to unseen prompts that are also unseen families; it cannot
separate prompt shift from family shift.

## 6. Target availability

### Future token t+1

Available for all 684 deep runs and 19 models: 53,312 aligned step samples before layer
replication. The target is the per-model tokenizer ID at `position + 1`. Cross-model
token IDs are not comparable. Per model there are 303–1,807 observed classes; on
average 23.4% of samples belong to singleton classes. Under the proposed outer folds,
only 62.6% of test token rows on average have a token class observed in TRAIN; the
worst model/prompt fold is 30.3%.

A full-vocabulary linear classifier is therefore statistically weak and potentially
expensive. If executed, primary accuracy must be unconditional over every test row,
with unseen-training classes necessarily scored incorrect. Seen-class coverage and
conditional seen-class accuracy may be secondary diagnostics. A top-k frequent-token
subset is scientifically a restricted-vocabulary question and must be labeled as such;
it cannot replace the full target silently.

### Future token t+2

Available for 681 runs: 52,628 aligned step samples. Three two-token BLOOMZ runs lack a
t+2 row. Class sparsity and tokenizer-specificity are essentially the same as t+1.
Treat t+2 as secondary after t+1 alignment is validated.

### Task family

Real labels exist, but strict held-out-prompt classification is **NOT FEASIBLE** because
each family has one prompt. Excluded from initial P2.

### Structured versus unstructured

No clean binary label exists. `STRUCTURED_OUTPUT`, `CODING`, and number-only prompt
requirements are not an archived common response-structure label. Heuristic output
parsing would create a new label. Excluded.

### Future Dynamic Regime

Probability metrics exist for most new runs, but `taxonomy_status` is DEFERRED and no
clean regime label exists. Continuous future metrics could be derived later with strict
feature-time auditing, but are not part of the initial execution.

## 7. Experimental unit and alignment

Primary unit is `(model, run_id, prompt_id, generation_id when available, token_step,
layer_index, normalized_depth, hidden_vector, target, outer_fold)`. Generation ID is
missing for reused Qwen and must remain null rather than synthesized. For Qwen, the
`logits_region_fp16` step dimension supplies the prompt/generation boundary; token IDs
supply targets.

Use token hidden state at position `prompt_len + step`; t+1 is the token at the next
position and t+2 is two positions later. Boundary rows without a target are omitted
explicitly. Do not pool across steps or layers in the primary test. Fit per actual layer
within each model; normalized depth is for post-hoc cross-model plotting, not semantic
layer equivalence.

## 8. Estimated samples and memory

Across all eligible archives: 53,312 t+1 and 52,628 t+2 step samples; after layer
expansion approximately 1.00M and 0.99M rows. The eight-model projected table is 323 MiB
on disk and uses 128 float32 dimensions. A single model/depth/fold matrix is generally
well below 20 MiB when streamed. Full raw H should be loaded one run at a time and never
assembled cross-model in RAM.

## 9. Exact initial probe protocol

1. Per model; no raw-feature pooling across architectures.
2. Primary target: exact per-model t+1 token ID, unconditional test scoring.
3. Secondary target: t+2 only after t+1 alignment checks pass.
4. Start with fixed 128-dimensional projections for the four-model bounded panel;
   optionally confirm selected results on raw H with streaming.
5. Evaluate each actual layer. If compute requires a screen, predeclare nearest layers
   to normalized depths 0, 0.25, 0.5, 0.75, and 1.0; do not average layers.
6. Train-only `StandardScaler`; L2-regularized multinomial linear classifier. Candidate
   `C ∈ {0.1, 1, 10}` selected on VALIDATION. If multinomial optimization is infeasible,
   a fixed regularized linear SGD classifier is an implementation alternative that must
   be disclosed, not mixed within a model.
7. Metrics: unconditional accuracy, top-5 accuracy, macro F1 over the union of test
   labels, class-frequency baseline, seen-training-class coverage, and conditional
   seen-class accuracy. Report class imbalance and class counts.
8. Controls: majority/class-frequency baseline; deterministic TRAIN-label permutation;
   mandatory prompt-held-out folds; prompt-cluster bootstrap intervals. Random-feature
   control is optional because fixed random projections are already the screening input.
9. Report early, peak, and final depth; non-monotonicity; prompt-bootstrap interval;
   permutation result. Do not assume monotonicity.
10. A one-hidden-layer constrained MLP is deferred until the linear protocol is frozen
    and only distinguishes linear from limited nonlinear accessibility.

## 10. Exclusions

- OPT-350M: no deep hidden-state artifact.
- Task-family target: one prompt per class makes held-out-prompt evaluation undefined.
- Structured/unstructured: no registered binary label.
- Dynamic regime: no frozen regime label and temporal-leakage risk.
- Correct/incorrect: intentionally excluded due evaluator and task confounds.
- Cross-model exact token classifier: token-ID spaces differ.
- Any row-level or seed-only split: prompt leakage.

## 11. New generation requirement

No new generation is required for the scoped future-token pilot. New generation would
be required to test task-family decodability with held-out prompts while retaining each
class in training, because that requires multiple independent prompts per family. It is
not authorized or recommended in this mission.

## 12. Scientific boundary

`READOUT ACCESSIBILITY != HELD-OUT FUNCTIONAL DECODABILITY != PREDICTION != CAUSAL CONTROL`.
The proposed probe asks whether a simple learned function transfers across the twelve
archived prompts. It does not establish prediction in deployment, mechanism, or control.
