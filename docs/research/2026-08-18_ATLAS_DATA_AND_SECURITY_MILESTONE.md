# Atlas Data and Security Milestone — 2026-08-18/19

## Data expansion

The frozen expansion contains 729 successful Lane A generations and 144 Lane B full-tensor captures, plus 108 bounded cross-environment fingerprint generations. Lane B captured H, Q, K, V, attention, attention output, and logits without recorded NaN/Inf. Raw `.npz` payloads remain local; indexes, manifests, provenance, and hash references are versioned.

## Atlas replication and routes

Lane B replicated the four frozen reference candidates: L0 (47.2%), L1 (43.3%), L5 (33.9%), and L20 (29.9%). The reference-network gate supports the descriptive directional connections L0→L5 and L0→L20. The composite gate supports L0→L5→L20 across 8 runs, 6/12 families, and 2/3 seeds after Holm correction against 23 alternatives.

The route-property profile is deliberately negative/conservative: no distinctive tensor signature for the route is established. In particular, the disclosed attention-output difference against no-landmark territory did not distinguish the full route from L0-L5-only territory (p=0.50).

## Runtime fingerprint research

The 48-challenge feasibility gate supports a challenge fingerprint signal (77.78% full-execution top-1 versus 2.08% random; 84.03% within-family versus 25% random), including a supported two-step prefix. Cross-environment probes across historical CUDA bfloat16 and new CPU float32/bfloat16/float16 conditions retained 92.59% top-1 from a 94.44% panel baseline; ROC-AUC was 0.9306. Hardware, dtype, runtime, and time effects are not fully isolated, and within-family cross-environment discrimination is not established because the compact panel contains one challenge per family.

## Boundaries and incident record

These gates establish no causality, control, steering, SRA-X validation, authentication, PQC, cryptographic entropy, unclonability, ownership proof, or security-product claim. The incomplete Reference Landmark Network V0 directory and the quarantined duplicate execution remain preserved with their conflict disclosure; missing canonical files were not reconstructed.
