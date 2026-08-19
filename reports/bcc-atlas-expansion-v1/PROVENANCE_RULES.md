# PROVENANCE_RULES

## Hard rule (mission section 0, restated and honored)

**Before writing any output directory: `ls -la <target-directory>`. If it already
contains files: STOP and inspect before writing.**

This rule exists because of a real, disclosed incident
(`reports/reference-landmark-network-v0-claude-conflict/CONFLICT_DISCLOSURE.md`,
2026-08-18): a duplicate mission execution overwrote a canonical prior deliverable
because the target directory was never listed before writing began. That incident's
lesson is now also recorded in persistent memory
(`methodology_output_dir_collision`). Applied here: `reports/bcc-atlas-expansion-v1/`
was confirmed empty via `ls -la` immediately before this preflight's first write.

## Canonical artifacts this campaign must never modify

| Path | What it is | Why it's protected |
|---|---|---|
| `star_command_bcc_v2/` | Canonical BCC V2 campaign (6 models, 1215 completed generations) | Explicit mission instruction: "DO NOT modify canonical BCC V2" |
| `star_command_panel_v2_authoring/FREEZE/` | Panel V2 frozen canonical prompt set (243 prompts) | Frozen artifact, sha256-pinned (`4172b4e0...b249`); this campaign's entire prompt source |
| `reports/atlas-kv-compression-v0/` | Atlas-KV V0 (30 real captures, K5-vs-K0 gate) | Frozen, hash-verified prior gate |
| `reports/attention-sensitivity-v1/` | Attention-Sensitivity V1 (SUPPORTED verdict) | Frozen, hash-verified prior gate |
| `reports/multi-tensor-atlas-v0/` | Multi-Tensor Atlas V0 | Frozen, hash-verified prior gate |
| `reports/multi-tensor-landmarks-v0/` | Landmark Validation V0 (4 confirmed landmarks) | Frozen, hash-verified prior gate |
| `reports/reference-landmark-network-v0/` | Canonical Reference Landmark Network (Codex) | Canonical survivor of the recent conflict incident — not this campaign's concern directly, but listed here for completeness since it shares the same landmark IDs (L0/L1/L5/L20) this campaign will test recurrence against |

All of the above are read from, where relevant (prompt text, seed lists, per-tensor
storage benchmarks, landmark definitions), and never written to.

## New-campaign namespace

All new output lives under `reports/bcc-atlas-expansion-v1/` exclusively:
- `A/BCC_WIDE/` — Lane A generation output (not yet populated at preflight time)
- `B/ATLAS_DEEP/` — Lane B tensor capture output (not yet populated at preflight time)
- Preflight deliverables (this file and its 10 siblings) at the top level of
  `reports/bcc-atlas-expansion-v1/`

## Before any future write into this campaign's own directories

Even within `reports/bcc-atlas-expansion-v1/` itself, the same `ls -la`-first discipline
applies to every subsequent phase (Lane A execution, Lane A audit, Lane B execution) —
each phase should re-verify its own target subdirectory is in the expected state before
writing, not assume it from the plan alone.
