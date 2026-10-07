# LIMEN Viability Test V0

Date: 2026-10-07

## Question

Can existing LIMEN/SRA trajectory data support a credible case that runtime dynamics contain useful early information, without launching a new inference campaign?

This V0 deliberately separates three evidentiary questions:

1. same-run dynamic metrics -> output properties;
2. trajectory prefix -> final dynamic endpoint;
3. intervention-associated dynamic changes -> output changes.

It does **not** claim early prediction of semantic failure, and it does **not** claim causality.

## Data integrity decision

The V14.18 output-linkage campaign and V14.14 temporal-control campaign share the keys `model/prompt/condition`, but they are distinct campaigns and their aggregate values do not exactly match.

Therefore they were **not joined as if they represented the same generation**.

- V14.18 is used for same-run dynamics -> output linkage.
- V14.14 is used for prefix -> final dynamic endpoint prediction.
- V14.18 paired baseline/intervention records are used for intervention-associated delta analysis.

This avoids a false early-output prediction result caused by cross-campaign pseudo-pairing.

## Part A — Dynamics -> output, n=72

Validation: out-of-fold logistic regression with GroupKFold grouped by prompt.

High repetition was defined prospectively within this analysis as the top quartile of repetition rate (threshold 0.202625).

Results:

| Target | Full dynamics AUROC | Entropy-only AUROC | Interpretation |
|---|---:|---:|---|
| High repetition | 0.641 | 0.566 | modest useful signal |
| Any marker | 0.206 | 0.209 | no useful signal |
| Uncertainty marker | 0.156 | 0.193 | no useful signal |
| Contradiction marker | 0.336 | 0.252 | no useful signal |
| Refusal marker | not estimable | not estimable | only 2 positives |

Selected continuous associations:

- ratio_norm vs repetition_rate: Spearman rho 0.299, p=0.0106;
- oscillation_rate vs repetition_rate: rho 0.270, p=0.0216;
- oscillation_rate vs output_length: rho 0.300, p=0.0105;
- logit_entropy_mean vs output_length: rho 0.347, p=0.00286.

These are exploratory and not multiplicity-corrected.

## Part B — Early prefix -> final dynamic endpoint, n=72

Target 1: final ratio above the cohort median.

| Prefix observed | Full prefix AUROC | Entropy-only AUROC |
|---|---:|---:|
| 25% | 0.630 | 0.503 |
| 50% | 0.698 | 0.607 |
| 75% | 0.776 | 0.617 |

Target 2: final mean entropy above the cohort median.

| Prefix observed | Full prefix AUROC | Entropy-only AUROC |
|---|---:|---:|
| 25% | 0.719 | 0.742 |
| 50% | 0.863 | 0.863 |
| 75% | 0.975 | 0.958 |

Interpretation:

- Early trajectory information predicts later dynamic state better than chance for final ratio.
- The multivariate trajectory representation adds clear value over entropy alone for final ratio, especially at 75%.
- For final entropy, entropy alone already carries most of the signal; the full vector adds little.
- Therefore the useful LIMEN proposition is not "more features always win"; it is that some future dynamic properties are encoded in trajectory shape beyond scalar entropy.

## Part C — Intervention-associated deltas, 54 paired intervention-vs-baseline comparisons

The intervention analysis is not randomized causal identification. It asks whether within-prompt intervention-associated dynamic changes covary with output changes.

Notable observations:

- CONSTRAIN reduced oscillation_rate by mean -0.0698, Wilcoxon p=0.0115.
- CONSTRAIN shortened outputs by mean -5.5 tokens, p=0.0305.
- Across all paired interventions, delta oscillation_rate correlated with delta repetition_rate: rho 0.323, p=0.0170.
- delta recovery_rate correlated with delta uncertainty markers: rho 0.284, p=0.0377.
- Other dynamic-output delta relationships were weak or inconsistent.

Because interventions can directly affect output form, these results do not establish that changing the latent dynamics caused the output change.

## Robustness check

A small prompt-grouped permutation check was run as an exploratory sanity test, limited to 10 permutations to avoid turning V0 into a heavy Jetson campaign.

Observed vs permutation mean AUROC:

- high repetition, full dynamics: 0.641 vs 0.463;
- final ratio at 25%, full prefix: 0.630 vs 0.493;
- final ratio at 50%, full prefix: 0.698 vs 0.538;
- final ratio at 75%, full prefix: 0.776 vs 0.496.

With only 10 permutations the smallest attainable one-sided empirical p-value is 0.0909. These checks support directionality only; they are not publication-grade significance tests.
## Gate decision

**V0 result: CONDITIONAL PASS for a new prospective early-warning experiment.**

What is supported:

- trajectory prefixes contain information about later dynamic state;
- a multivariate prefix can outperform scalar entropy for at least one meaningful dynamic endpoint;
- some dynamic metrics show same-run association with output repetition/length;
- paired interventions alter both dynamics and outputs in structured ways.

What is **not** supported:

- reliable early prediction of semantic failure;
- reliable early prediction of contradiction or uncertainty;
- causal control of output quality through latent intervention;
- a deployable autonomous runtime controller;
- a universal viability boundary.

The next experiment must therefore directly record, in the **same generation**:

1. token/layer trajectory features;
2. frozen prefix checkpoints at 25/50/75%;
3. final semantic outcome labels;
4. failure/recovery type;
5. intervention/no-intervention assignment if control is tested.

Primary prospective question:

> Can a trajectory prefix predict final failure/recovery on the same generation, with useful lead time and better calibration than entropy/confidence baselines?

Suggested promotion gate:

- grouped held-out AUROC materially above entropy/confidence baseline;
- stable performance across prompt families;
- acceptable false-positive interception rate;
- measurable lead time before terminal failure;
- calibration error reported;
- no controller deployment until prospective prediction is replicated.

## Meta-cognition implication

V0 supports a **machine-readable system self-state** as an architectural object, but not an autonomous self-governing controller.

Safe now:

- represent mission state, remaining functions, evidence, authority, active workers, observed failures, reusable work products, budget and runtime viability as explicit state;
- ingest LIMEN signals as advisory observations with provenance and confidence;
- compare current state against mission constraints;
- propose an adaptation;
- require human/policy authorization before consequential action;
- write the observation, proposal, decision and outcome to the Ledger.

Not justified yet:

- allowing LIMEN to unilaterally stop or redirect an enterprise mission;
- treating a low viability score as semantic incorrectness;
- self-modifying allocation policy from one run;
- autonomous authority escalation;
- calling this model introspection or consciousness.

The evidence therefore supports the control loop:

`OBSERVE -> SELF_STATE -> DIAGNOSE -> PROPOSE -> GATE -> ACT -> LEDGER`

with LIMEN initially supplying **observations**, not authority.

## Bottom line

The central viability hypothesis survives V0 in a narrower and more defensible form:

> Runtime trajectory dynamics contain early information about later dynamic state, and some trajectory-derived signals are associated with output behavior. Existing data justify a same-run prospective failure/recovery prediction study, but do not yet justify claims of semantic-failure prediction, causal control, or autonomous meta-cognition.
