# Dynamic Control Research Framework

Status: **GLOBAL LIMEN RESEARCH FRAMEWORK — CONCEPTUAL + EMPIRICAL STATUS SEPARATED**  
Concept attribution: **Jean-Denis Bosange Batuli**  
Date: 2026-08-17

## 1. Motivation

Across LIMEN perturbation/control experiments, a recurring pattern appeared: an intervention could produce a measurable local or downstream effect while the full dynamic object being modified remained only partially characterized.

This document formalizes the resulting research programme without claiming that control has been achieved.

Core distinction:

```text
observable signal != complete dynamics
perturbation != influence != control
```

## 2. Composite dynamic configuration

Candidate execution configuration:

```text
D_t = {
  State,
  Geometry,
  Representation,
  Trajectory,
  Flow,
  Flux,
  Coupling,
  Regime,
  Environment,
  Response
}
```

This is a conceptual decomposition, not a validated complete state representation.

The representation is not forced to be a single vector. Components may naturally be tensors, sequences, matrices, graphs, categorical variables, or contextual structures.

## 3. State vs Dynamic State

`hidden_state_t` is an instantaneous model representation.

A candidate Dynamic State is broader: it may combine current state, recent movement, architectural propagation, tensor-magnitude evolution, coupling, regime, and environment.

Do not claim that a universal Dynamic State exists.

Competing hypotheses:

- **H_UNIVERSAL:** a relatively stable set of dynamic observables is sufficient across multiple response/intervention classes.
- **H_CONDITIONAL:** the minimal predictive set depends on intervention class, layer, regime, timing, model, or environment.

## 4. Flow vs Flux

### Flow

Flow describes the **local/structural organization and propagation of computation through an architecture**.

Candidate architecture-dependent observables include:

- layer-to-layer relations
- residual pathways
- module interactions
- attention structures where applicable
- head/neuron/module contribution where validly measured

Current experiments generally use **Flow proxies**, not direct Flow measurement.

### Flux

Flux describes broader **transport-like or magnitude evolution of tensor activity or computation**.

Keep two levels distinct:

```text
TENSOR FLUX
OPERATIONAL / COMPUTATIONAL FLUX
```

Tensor Flux may include cumulative magnitude movement or normalized activation movement.

Operational Flux requires actual runtime measurements such as throughput, latency, memory, or compute load. Do not infer compute allocation from activation norm alone.

## 5. Intra-step vs inter-step

### Intra-step dynamics

Dynamics occurring inside a single generation step across architectural components before token emission.

Typical candidate observables:

- layer-to-layer propagation
- residual/module changes
- architecture-local Flow proxies
- local tensor Flux proxies

### Inter-step dynamics

Dynamics describing how a configuration evolves from generation step `t` to `t+1`.

Typical candidate observables:

- trajectory
- direction
- curvature
- cumulative movement
- regime transitions
- temporal persistence
- temporal Flux

## 6. CrossAxis

Introduce a local candidate coordinate:

```text
D_(t,l)
```

where:

- `t` = generation-time/token-step axis
- `l` = architectural layer/module axis

CrossAxis should not be treated only as a scalar predictor. It may function as coordinate infrastructure for execution audit and visualization.

Core framing:

> **CrossAxis may be infrastructure, not signal.**

## 7. Execution Dynamic Map

The candidate Execution Dynamic Map is:

```text
Generation-Time Axis × Architectural Axis
```

with evidence-backed overlays such as:

```text
State
Geometry
Trajectory
Flow proxy
Flux proxy
Coupling
Regime
Perturbation
Response
Recovery
```

The map is not a claim that full model dynamics have been reconstructed.

## 8. Token relation

A token is a discrete observable output of generation.

It is not the Dynamic State.

Careful framing:

> Each token emission can be treated as a discrete observable associated with a transition between successive dynamic configurations.

Do not use this framework to claim reasoning, thought, or internal cognition reconstruction.

## 9. Dynamic State Embedding — candidate only

If an explicit projection is later constructed:

```text
z_t = Phi(D_t)
```

then `z_t` may be called a **Dynamic State Embedding** if it is demonstrably a compact representation of heterogeneous dynamic observables.

Current status:

```text
CONCEPTUAL CANDIDATE
NOT VALIDATED
NOT IMPLEMENTED AS A GENERAL LIMEN PRIMITIVE
```

The useful scientific question is not whether a "true" finite/infinite dynamic space exists. It is whether a compact representation preserves reproducible predictive information for a defined target/intervention class.

## 10. Effective predictive dimension

Do not infer ontological dimensionality from PCA saturation/non-saturation.

Instead study:

- incremental predictive value
- redundancy
- conditional sufficiency
- held-out performance
- effective predictive dimension

Candidate nested comparison:

```text
M0 = State
M1 = State + Geometry
M2 = State + Geometry + Trajectory
M3 = M2 + Flux proxies
M4 = M3 + Flow proxies
M5 = M4 + CrossAxis
M6 = M5 + Coupling proxies
```

The exact sets are experiment-specific and must be frozen before outcome inspection.

## 11. PredictiveGain

Conceptual quantity:

```text
PredictiveGain(Component | ExistingSet)
```

Question:

> Does adding this observable family provide reproducible held-out information beyond what is already known?

Low correlation alone does not establish independent information.

## 12. Dynamic Coupling Map

A **Dynamic Coupling Map** is a candidate informational interface between cartography and future response-predictive intervention.

It may be intervention-conditional:

```text
M_t^(u)
```

where `u` is an intervention class.

Possible functions:

```text
READ
LOCATE
RELATE
SENSITIVITY
PREDICT
UPDATE
```

This is not assumed necessary for control and is not currently validated.

## 13. Response Atlas

A future Response Atlas may describe historical response distributions conditioned on dynamic region/context and intervention class:

```text
ResponseAtlas(
  dynamic_region,
  intervention_class,
  response_distribution,
  uncertainty,
  context
)
```

Concept only.

## 14. Research progression

```text
Observation
→ Dynamic Decomposition
→ Cartography
→ Execution Dynamic Map
→ Response Mapping
→ System Identification
→ Controllability Analysis
→ Local Control ?
→ Closed-Loop Control ?
```

Each arrow is an empirical gate and may fail.

System Identification is a future methodological stage, not a current capability.

Candidate model families may later include local linear/Jacobian models, Volterra models, kernels, learned state-space models, autoregressive dynamic models, or nonlinear regressors. None is privileged before empirical comparison.

## 15. Control maturity

Suggested maturity vocabulary:

```text
C0 OBSERVATION
C1 CARTOGRAPHY
C2 PERTURBATION RESPONSE
C3 RESPONSE PREDICTION
C4 LOCAL TARGETED CONTROL
C5 REPEATABLE CONTROL
C6 CLOSED-LOOP CONTROL
```

LIMEN must not claim C4+ without dedicated evidence.

## 16. Current empirical anchor: D0B dynamic-gain test

The first test of this decomposition reused 1014 existing D0B artifacts with zero new model generation.

Narrow result:

- Flow proxies: no standalone predictive gain established on tested targets.
- CrossAxis: no standalone predictive gain established on tested targets.
- Coupling proxies: no standalone predictive gain established on tested targets.
- `X3_flux_cumulative_magnitude_shift`: positive incremental predictive gain beyond the tested State/Geometry/Trajectory baseline on one downstream signed-projection target.

Reference figures from the frozen analysis:

```text
M2 R² ≈ -0.0958
M2 + X3 R² ≈ +0.1129
ΔR² ≈ +0.2087
```

This result is target-specific and was unreplicated at the time this framework was written.

It does not validate Flux generally, Dynamic State, control, or system identification.

## 17. D0 replication lesson

A strict D0 replication attempt was stopped because D0 was not source-independent from D0B: overlapping baseline/random-direction captures were byte-identical.

Result:

```text
X3 CONFIRMED: NO
X3 REFUTED: NO
X3 STATUS: UNREPLICATED
```

General lesson:

> Campaign identity does not establish statistical independence. Independence must be established at the experimental-unit and artifact level.

## 18. Current independent replication programme

A source-independent replication was preregistered with:

- new discovery/evaluation prompts
- new disjoint seeds
- independently constructed/frozen TRUE_DIRECTION under the same method
- new random-direction realizations under the same method
- frozen X3
- frozen M2 baseline
- frozen downstream target
- frozen analysis pipeline

The replication tests **X3 only**, not the entire framework.

## 19. Operational usefulness vs predictive usefulness

Mandatory distinction:

```text
descriptive utility != navigational utility != predictive utility != control utility
```

A family that does not improve R² may still be useful for audit/navigation.

CrossAxis is especially important here: it may provide the coordinate surface on which other measurements become navigable.

## 20. Preferred product/research language

Prefer:

```text
spatiotemporal execution navigation
dynamic activity allocation
computational activity distribution
runtime transformation map
execution topology
layer-time activity map
```

Avoid unsupported labels:

```text
cognitive navigation
allocation of intelligence
reasoning map
where the model thinks
```

## 21. Attribution

The composite Dynamic State framing, Flow/Flux distinction, Conditional Dynamic State hypothesis, intra-step/inter-step decomposition, CrossAxis-as-coordinate-infrastructure framing, Dynamic Coupling Map concept, and staged path from cartography to system identification/controllability are proposed here by **Jean-Denis Bosange Batuli** as a LIMEN research framework derived from recurring control/perturbation observations.

No universal scientific priority claim is made without a dedicated literature review.

## 22. Non-claims

This framework does not establish:

- reasoning reconstruction
- cognitive-state measurement
- semantic truth
- complete LLM dynamics
- universal Dynamic State
- direct Flow measurement
- general Flux law
- system identification
- controllability
- closed-loop control
