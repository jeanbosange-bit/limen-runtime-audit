# Phase 9 reconstruction environment

Recovery/replay timestamp: `2026-08-20T17:39:17+02:00`  
Timezone: `Europe/Brussels`  
Platform: `Linux-5.15.185-tegra-aarch64-with-glibc2.35`  
Python implementation: `CPython`  
Python executable selected by shell: `/home/jeanbatuli/sra_env/bin/python`, resolving to
`/usr/bin/python3.10`  
Python version: `3.10.12 (GCC 11.4.0)`

## Imported runtime dependencies

| dependency | replay version | import location | touched by recovered script |
|---|---|---|---|
| numpy | 1.26.4 | `/home/jeanbatuli/sra_env/lib/python3.10/site-packages` | yes |
| pandas | 2.3.3 | `/home/jeanbatuli/.local/lib/python3.10/site-packages` | yes |
| scikit-learn | 1.7.2 | `/home/jeanbatuli/sra_env/lib/python3.10/site-packages` | yes |
| pyarrow | 23.0.1 | `/home/jeanbatuli/.local/lib/python3.10/site-packages` | yes |
| scipy | 1.15.3 | `/home/jeanbatuli/.local/lib/python3.10/site-packages` | indirect scikit-learn dependency; not directly imported |
| torch | 2.8.0 | `/home/jeanbatuli/sra_env/lib/python3.10/site-packages` | no; recorded for completeness only |

Standard-library dependencies: `warnings`. The replay launcher additionally uses
`argparse`, `pathlib`, and `shutil`. The verification script uses `argparse`, `hashlib`,
`pathlib`, NumPy, and pandas.

The original 2026-08-19 session does not contain a package-version capture. Therefore
these are the exact successful replay versions, not asserted original-run versions. The
five byte-identical outputs demonstrate that this environment reproduces the preserved
canonical outputs for the executed code and inputs.

No CUDA, torch operation, model loading, network access, or generation occurred during
the replay.

