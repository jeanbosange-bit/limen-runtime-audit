# Raw Capture Storage Policy

Raw tensor captures are intentionally excluded from normal Git history. The Lane B and cross-environment `.npz` payloads remain preserved in their local campaign directories; Git stores frozen protocols, analysis scripts, schemas, run indexes, metadata, derived results, manifests, provenance, and SHA-256 references.

This repository does not bypass GitHub file-size limits and does not activate Git LFS automatically. A future preservation decision may use a dedicated artifact store, release archive, Git LFS, Zenodo, or an equivalent system. No raw capture is uploaded by this integration.

Expected local locations are `reports/bcc-atlas-expansion-v1/ATLAS_DEEP/raw_captures/` and `reports/cross-environment-fingerprint-v0/raw_captures/`. Their canonical hash registries remain in the corresponding report directories.
