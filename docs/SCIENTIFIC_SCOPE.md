# Scientific scope

## Claim levels

- **[I] Observation:** directly measured from the supplied arrays.
- **[II] Controlled result:** supported by appropriate statistics and controls.
- **[III] Interpretation:** compatible with the observations but not uniquely
  established by them.
- **[IV] Hypothesis:** a proposed explanation requiring a new test.

All metrics emitted by v0.1 are [I]. The software does not automatically
promote them to [II].

## Current supported use

LIMEN v0.1 is suitable for descriptive comparison of small open-weight models,
checkpoints, prompt groups and extraction pipelines. It can expose numerical
errors, profile differences and outlying runs.

## Unsupported conclusions

The following conclusions are outside the evidence produced by this package:

- a trajectory region is a semantic thought state;
- decodability identifies functional localization;
- smoothness means correctness;
- irregularity means hallucination;
- an activation perturbation caused an output improvement;
- a metric is invariant across models or transformations;
- a layer-wise path is a time-domain physical trajectory.

## Publication boundary

The public package is LLM-only. Non-LLM experiments and unaudited control or
compression claims are outside its scope.
