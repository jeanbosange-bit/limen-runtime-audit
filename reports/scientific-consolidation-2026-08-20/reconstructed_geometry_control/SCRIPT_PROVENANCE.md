# Script provenance

Recovery performed 2026-08-20 (Europe/Brussels) from Claude session JSONL
`5d793c3e-5e08-4a66-9200-8acfbc5066c9.jsonl` under
`/home/jeanbatuli/.claude/projects/-home-jeanbatuli-V20-probabilistic-dynamics/`.

| Script or operation | Session event | Archived time (UTC) | SHA-256 | Recovery | Later edit found |
|---|---:|---|---|---|---|
| `build_geometry_control.py` | 4094 | 2026-08-19 11:40:45 | `e58a9e09016b12a80e02d1741aacf1f6c552f2bf4132f9221b3ea2ef15ca0670` | EXACT RECOVERY | No |
| `finalize_and_quality.py` | 4176 | 2026-08-19 11:45:47 | `754f100f6cb46d61169d680f4e7b54f694e5fe76f1c6bbc4fe9418d5866cbbf5` | EXACT RECOVERY | No |
| `build_transfer_report_and_package.py` | 4232 | 2026-08-19 11:49:30 | `730d5a6236f22ee011a4b3b1d8ba3ea436d77b31589f33f005dd914bad54f653` | EXACT RECOVERY | No |
| float32 recast | 4475 | 2026-08-19 12:26:43 | Not a file | EXACT COMMAND RECOVERY | No later recast found |

All three Python files pass `python -m py_compile`. No recovered source line was
inferred or edited. Replay redirects the hard-coded output directory in memory; the
materialized recovered files retain their exact historical text and hashes.

Chronology: geometry control ran at event 4411; finalization ran at 4415; core scalar
extraction was then run at 4419; finalization was rerun at 4465; the two projection
tables were recast at 4475; the transfer hash package ran at 4504. The exact recast
loop loaded `h_random_projection.parquet` and `orthogonal_subset.parquet`, cast every
`proj_` column to `float32`, and rewrote each with pandas `to_parquet(index=False)`.

Replay environment: Python 3.10.12; numpy 1.26.4; pandas 2.3.3; pyarrow 23.0.1;
scipy 1.15.3; scikit-learn 1.7.2; torch 2.8.0. The recovered scripts directly touch
numpy, pandas, and pyarrow; torch is not touched by these remaining stages.

