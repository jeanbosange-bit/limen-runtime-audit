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

## Global dynamic-control research framework

The broader LIMEN research programme now includes a formally separated
conceptual/experimental framework covering:

- composite Dynamic State candidates;
- **Flow vs Flux** as distinct observable families;
- **intra-step vs inter-step** dynamics;
- **CrossAxis** as a candidate `generation-time × architecture` coordinate
  system for execution audit and navigation;
- Conditional vs Universal Dynamic State hypotheses;
- Dynamic Coupling Map / Response Atlas concepts;
- incremental predictive value and effective predictive dimension;
- the staged path `Cartography → Response Mapping → System Identification →
  Controllability Analysis`;
- replication-source-independence discipline.

The canonical global research note is
[`DYNAMIC_CONTROL_RESEARCH_FRAMEWORK.md`](DYNAMIC_CONTROL_RESEARCH_FRAMEWORK.md).

This framework does **not** promote those concepts to validated properties of
LLMs. In particular, descriptive or navigational usefulness is kept separate
from predictive usefulness and control evidence.

## Unsupported conclusions

The following conclusions are outside the evidence produced by this package:

- a trajectory region is a semantic thought state;
- decodability identifies functional localization;
- smoothness means correctness;
- irregularity means hallucination;
- an activation perturbation caused an output improvement;
- a metric is invariant across models or transformations;
- a layer-wise path is a time-domain physical trajectory;
- CrossAxis measures cognition or "allocation of intelligence";
- Flow proxies directly measure internal information flow;
- Flux proxies establish a general law of model dynamics;
- a Dynamic State, Dynamic Coupling Map or Dynamic State Embedding is validated;
- system identification, controllability or closed-loop control has been achieved.

## Publication boundary

The public package is LLM-only. Non-LLM experiments and unaudited control or
compression claims are outside its scope.
