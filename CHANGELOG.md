# Changelog

## Unreleased

- Added the first real-model example from a revision-pinned TinyLlama run.
- Added machine-readable descriptive summaries and a plain-language report.
- Added an exact, bitwise replication record for the first TinyLlama run.
- Added a prompt-contrast example where TinyLlama accepts a false arithmetic
  premise, with probability and geometry measurements kept separate.
- Clarified that raw activation distances are protocol-dependent and that a
  single trajectory is not a quality score or reference distribution.
- Added `tests/test_invariance.py`: 11 tests responding directly to an
  independent technical review (July 2026), covering path-length/
  displacement/velocity/acceleration scaling behavior, tortuosity and
  turning-angle invariance under translation/rotation/uniform scale, and
  the zero-motion turning-angle convention. All 20 tests (11 new + 9
  existing in test_metrics.py) pass.

## 0.1.0 - 2026-07-24

- Added the first public LLM-only audit contract.
- Added descriptive layer-trajectory metrics.
- Added entropy and probability-margin baselines.
- Added immutable source hashing and JSON/Markdown reports.
- Added extraction-manifest example and unit tests.
- Explicitly excluded causal, semantic and controllability claims.
