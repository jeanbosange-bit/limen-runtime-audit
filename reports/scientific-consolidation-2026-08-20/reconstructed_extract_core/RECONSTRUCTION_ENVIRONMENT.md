# extract_core reconstruction environment

Recovery verification timestamp: `2026-08-20T21:19:32+02:00`  
Platform: `Linux-5.15.185-tegra-aarch64-with-glibc2.35`  
Python: CPython `3.10.12`, GCC 11.4.0  
GPU path: NVIDIA Jetson CUDA, accessed outside the filesystem sandbox because the
historical script explicitly uses `cuda`.

| dependency | replay version | directly used |
|---|---:|---|
| numpy | 1.26.4 | yes |
| pandas | 2.3.3 | imported by recovered script; dependency module also uses it |
| pyarrow | 23.0.1 | yes |
| torch | 2.8.0 | yes |
| transformers | 4.46.3 | yes |
| safetensors | 0.7.0 | checkpoint loading dependency |
| accelerate | 1.13.0 | installed model-loading dependency |

The original session did not freeze all package versions. These are the successful replay
versions. Four historical full-panel model failures and two replay-process terminations
show that the Jetson allocator state is not reliably stable across a long multi-model
process. Per-model isolated fresh processes were therefore used to close valid replay
parquets without changing per-model extraction logic.

No generation, sampling, network-dependent acquisition, fine-tuning, or raw-tensor write
occurred. Cached checkpoints were loaded only to obtain final-normalization and LM-head
weights and to run the historical module-identification forward pass.

