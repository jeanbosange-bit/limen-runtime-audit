# CAMPAIGN_PLAN — BCC Atlas Data Expansion V1

Status: **PREFLIGHT ONLY**. `real_new_generations = 0`. No execution has occurred.

## Objective

Maximize useful Atlas territory coverage via two independent lanes, both against the
same frozen 243-prompt Panel V2 canonical set, on a single model
(Qwen/Qwen2.5-0.5B-Instruct) in Phase 1. Not a benchmark race — the objective is
*coverage variation* (family, length, structure), not scoring.

## Lane A — BCC-WIDE

- **Prompts**: all 243 Panel V2 frozen prompts, verbatim, read-only
  (`PROMPT_SOURCE_REGISTRY.json`).
- **Seeds**: 3 new seeds — 2101, 2102, 2103 (`SEED_REGISTRY.json`), verified absent from
  every seed value used anywhere else in this project.
- **Generations**: 243 × 3 = **729**.
- **Capture**: lightweight only — text, token ids, per-step probability summaries
  (entropy/top1/top2-gap), runtime/memory metadata, hashes. **No K/V/attention tensors.**
  Full schema: `LANE_A_CAPTURE_SCHEMA.json`.
- **Purpose**: route density, family/length coverage, candidate-route discovery, and —
  critically — informs which 48 prompts Lane B captures deeply.

## Lane B — ATLAS-DEEP

- **Candidate target**: 48 prompts × 3 seeds (same seed registry as Lane A) = **144
  deep runs**.
- **Selection rule (frozen, structural, NOT outcome-based)**: 4 prompts per family × 12
  families = 48. Within each family's 4, prefer spanning that family's own observed
  short/medium/long generation-length range from Lane A's real results (not from any
  scientific effect, not from landmark presence/absence, not from any V1
  attention-sensitivity value). **The actual 48 prompt_ids are NOT selected in this
  preflight** — selection happens after Lane A completes and its real length
  distribution is known (see Execution Order below). This preflight fixes the *rule*,
  not the *list*.
- **Capture**: full synchronized tensors — H, Q (newly available, see
  `LANE_B_CAPTURE_SCHEMA.json`'s audit), K, V, attention weights, attention output,
  bounded logits, token ids — at true native granularity, explicit per-field metadata.
- **No 8-step (or any-length) minimum window** — every run, however short, is a valid
  capture (`SHORT_EXECUTION_POLICY.md`).

## Execution order (frozen, mission section 16)

1. Preflight (**this document set** — complete)
2. Lane A generation (729 lightweight generations)
3. Lane A completion audit (real length distribution, family coverage, technical-error
   rate)
4. Structural selection of the Lane B 48-prompt subset, from step 3's real data
5. **Human GO** required before Lane B (mission section 16 explicit — even if Lane A was
   pre-authorized for rapid execution, Lane B needs its own GO, and STOPS again if
   projected storage/runtime differs materially from `RESOURCE_ESTIMATE.md`)
6. Lane B tensor capture

**Lane A may proceed after a successful preflight if the user has authorized rapid data
acquisition** (mission section 16) — this preflight does not itself authorize Lane A
execution; that authorization is a separate human decision this document surfaces but
does not grant.

## Independent axes maintained (mission section 14)

Lane A never merges behavioral outcome, runtime, and probability/trajectory data into a
single score. Six axes are kept structurally separate throughout: BEHAVIOR, RUNTIME,
PROBABILITY, TRAJECTORY, TENSOR (Lane B only), EVIDENCE (hashes/provenance).

## What this campaign does NOT do

- Does not modify BCC V2, Panel V2 frozen artifacts, Atlas-KV V0, Attention-Sensitivity
  V1, Multi-Tensor Atlas V0, or Landmark Validation V0 (`PROVENANCE_RULES.md`).
- Does not select any prompt based on the 4 confirmed landmarks (L0/L1/L5/L20) —
  landmark recurrence is measured as an OUTCOME of this campaign, never a selection
  criterion (mission section 12).
- Does not promote any new landmark/edge — that requires its own separate validation
  gate, exactly as `reports/multi-tensor-landmarks-v0/` and
  `reports/reference-landmark-network-v0/` already did for the prior capture.
- Does not add a second model in Phase 1.
