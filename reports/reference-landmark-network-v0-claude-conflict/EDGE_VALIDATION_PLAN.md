# EDGE_VALIDATION_PLAN

Frozen before any edge significance is inspected.

## Null model

**Within-run landmark-label shuffle.** For each run with >=2 qualifying encounters,
keep that run's own multiset of step-positions at which a landmark was encountered fixed,
and keep the run's own multiset of landmark labels fixed (e.g. a run with encounters
`[L0, L5, L1, L20, L5]` keeps exactly those 5 labels and those 5 step-positions), but
randomly permute **which label attaches to which step-position** within that run. This
is a group-preserving null (grouped by `run_id`): it destroys any true within-run
ordering bias between specific landmark pairs while preserving each run's own tempo
(how many encounters, how far apart in steps) and each landmark's own overall frequency
across the corpus. Runs with 0 or 1 qualifying encounter contribute no directed pairs and
are unaffected by the shuffle (nothing to permute).

## Procedure

1. Compute the observed `n_runs_supporting` (DIRECT_SUCCESSION + INDIRECT_SUCCESSION
   combined) for each of the 12 ordered pairs, per `EDGE_DEFINITION.md`.
2. Run 200 permutations of the null above (RNG seed 20260818, consistent with every prior
   gate in this program).
3. For each ordered pair, p = (count of permutations with permuted support >= observed
   support + 1) / 201.
4. **Multiple-comparison correction**: Bonferroni across the 12 ordered pairs, frozen
   before testing: alpha_corrected = 0.05 / 12 = 0.004167. An edge is
   `VALIDATED` only if p < 0.004167. `LOW_SUPPORT` if 0.004167 <= p < 0.05 (i.e. it would
   have passed an uncorrected test but not the corrected one). `NOT_SUPPORTED` if p >= 0.05.

## What this can and cannot detect

With only 21 runs and 35 total qualifying encounters (10 runs with >=2 encounters, the
only runs that can contribute a directed-pair observation at all), this null is
necessarily low-powered. A `NOT_SUPPORTED` or `LOW_SUPPORT` result does not mean no real
relationship exists — it means this specific frozen test, at this sample size, does not
clear the pre-registered bar. This is disclosed here, before any result is seen, precisely
so it cannot be used to explain away an inconvenient result after the fact.

## Explicit prohibition

No edge is fished by trying multiple thresholds and reporting the one that produces the
densest network. The Bonferroni-corrected 0.004167 bar and the 200-permutation,
20260818-seeded null are frozen and will be used exactly once, as specified above.
